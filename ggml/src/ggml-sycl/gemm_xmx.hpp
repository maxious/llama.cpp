#ifndef GGML_SYCL_GEMM_XMX_HPP
#define GGML_SYCL_GEMM_XMX_HPP

#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_MATRIX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif

// XMX-accelerated Graph-Compatible GEMM: C = alpha * A * B^T + beta * C
// Uses Intel XMX (Xe Matrix eXtensions) via SYCL joint_matrix
// Row-major matrices: A[M,K], B[N,K] (for B^T), C[M,N]
//
// NOTE: This uses BF16 intermediate precision for XMX compatibility.
// F32 inputs are converted to BF16, computed via XMX, accumulated to F32.
// This trades some precision (~3 decimal digits) for ~3-4x performance.
//
// Graph-compatible (no oneMKL dependencies).

#ifdef SYCL_EXT_ONEAPI_MATRIX

using xmx_bfloat16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

// XMX tile sizes for Intel Xe2 (Arc Battlemage / Alchemist)
// joint_matrix dimensions: A[TM,TK] @ B[TK,TN] -> C[TM,TN]
constexpr int XMX_TM = 8;
constexpr int XMX_TN = 16;
constexpr int XMX_TK = 16;

// Block sizes - how many XMX tiles per workgroup
// Reduced from 64x64 to fit within workgroup size limits (256 threads)
constexpr int XMX_BM = 32;   // 4 TM tiles = 32 rows
constexpr int XMX_BN = 64;   // 4 TN tiles = 64 cols  
constexpr int XMX_BK = 32;   // 2 TK tiles = 32 K elements

// Subgroups per workgroup
constexpr int XMX_NUM_SG_M = XMX_BM / XMX_TM;  // 4
constexpr int XMX_NUM_SG_N = XMX_BN / XMX_TN;  // 4
constexpr int XMX_NUM_SG = XMX_NUM_SG_M * XMX_NUM_SG_N;  // 16

// Subgroup size for Intel GPUs
constexpr int XMX_SG_SIZE = 16;

// Threads per workgroup (16 subgroups * 16 threads = 256, within limits)
constexpr int XMX_WG_SIZE = XMX_NUM_SG * XMX_SG_SIZE;  // 256

// Shared memory sizes with padding to avoid bank conflicts
constexpr int XMX_A_STRIDE = XMX_BK + 4;  // Padding for bank conflicts
constexpr int XMX_B_STRIDE = XMX_BN + 4;  // Padding for B (stored row-major after transpose)

template <bool transpose_B = true>
inline void gemm_xmx_kernel(
    sycl::nd_item<1> it,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc,
    xmx_bfloat16 * shA,  // Shared memory for A tile [BM x A_STRIDE]
    xmx_bfloat16 * shB,  // Shared memory for B tile [BK x B_STRIDE]
    float * shC          // Shared memory for C output [NUM_SG x TM x TN]
) {
    auto sg = it.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    const int lid = it.get_local_id(0);
    
    // 2D grid mapping: workgroup (block_m, block_n)
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int block_m = it.get_group(0) / grid_n;
    const int block_n = it.get_group(0) % grid_n;
    
    const int row_base = block_m * XMX_BM;
    const int col_base = block_n * XMX_BN;
    
    if (row_base >= M) return;
    
    // Which subgroup tile within the block (sg_m in [0,3], sg_n in [0,3])
    const int sg_m = sg_id / XMX_NUM_SG_N;
    const int sg_n = sg_id % XMX_NUM_SG_N;
    
    // Initialize accumulator to zero
    joint_matrix<sycl::sub_group, float, use::accumulator, XMX_TM, XMX_TN> matC;
    joint_matrix_fill(sg, matC, 0.0f);
    
    // Loop over K dimension in blocks
    const int num_k_blocks = (K + XMX_BK - 1) / XMX_BK;
    
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_base = k_block * XMX_BK;
        
        // === Load A tile [BM x BK] into shared memory as BF16 ===
        // A is row-major: A[m,k] at A[m * lda + k]
        // Store in shared as row-major: shA[m * A_STRIDE + k]
        constexpr int A_TILE_ELEMS = XMX_BM * XMX_BK;
        for (int idx = lid; idx < A_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_m = idx / XMX_BK;
            const int tile_k = idx % XMX_BK;
            const int global_m = row_base + tile_m;
            const int global_k = k_base + tile_k;
            
            float val = 0.0f;
            if (global_m < M && global_k < K) {
                val = A[global_m * lda + global_k];
            }
            shA[tile_m * XMX_A_STRIDE + tile_k] = xmx_bfloat16(val);
        }
        
        // === Load B tile [BN x BK] -> store as [BK x BN] for col-major access ===
        // B is row-major: B[n,k] at B[n * ldb + k] (when transpose_B)
        // For joint_matrix use::b with col_major layout, we need B stored as [K x N]
        // So we store: shB[k * B_STRIDE + n] = B[n,k]
        constexpr int B_TILE_ELEMS = XMX_BK * XMX_BN;
        for (int idx = lid; idx < B_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_k = idx / XMX_BN;
            const int tile_n = idx % XMX_BN;
            const int global_k = k_base + tile_k;
            const int global_n = col_base + tile_n;
            
            float val = 0.0f;
            if constexpr (transpose_B) {
                // B^T: input B is [N,K] row-major, we want B^T[K,N]
                if (global_n < N && global_k < K) {
                    val = B[global_n * ldb + global_k];
                }
            } else {
                // No transpose: B is [K,N] row-major
                if (global_k < K && global_n < N) {
                    val = B[global_k * ldb + global_n];
                }
            }
            shB[tile_k * XMX_B_STRIDE + tile_n] = xmx_bfloat16(val);
        }
        
        sycl::group_barrier(it.get_group());
        
        // === Compute using XMX ===
        joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, XMX_TM, XMX_TK, layout::row_major> matA;
        joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, XMX_TK, XMX_TN, layout::row_major> matB;
        
        // Loop over K tiles within this K block
        #pragma unroll
        for (int k_tile = 0; k_tile < XMX_BK; k_tile += XMX_TK) {
            // Load A[sg_m*TM : sg_m*TM+TM, k_tile : k_tile+TK]
            auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(shA + sg_m * XMX_TM * XMX_A_STRIDE + k_tile);
            joint_matrix_load(sg, matA, a_ptr, XMX_A_STRIDE);
            
            // Load B[k_tile : k_tile+TK, sg_n*TN : sg_n*TN+TN] (stored row-major)
            auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(shB + k_tile * XMX_B_STRIDE + sg_n * XMX_TN);
            joint_matrix_load(sg, matB, b_ptr, XMX_B_STRIDE);
            
            // Multiply-accumulate: C += A @ B
            joint_matrix_mad(sg, matC, matA, matB, matC);
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    // === Store results ===
    const int out_row = row_base + sg_m * XMX_TM;
    const int out_col = col_base + sg_n * XMX_TN;
    
    // Store accumulator to dedicated shared memory region (no aliasing with shA/shB)
    float * my_out = shC + sg_id * XMX_TM * XMX_TN;
    auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
        sycl::access::decorated::no>(my_out);
    joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
    
    sycl::group_barrier(it.get_group());
    
    // Write to global memory with bounds checking
    // Each lane writes multiple elements
    for (int i = lane_id; i < XMX_TM * XMX_TN; i += XMX_SG_SIZE) {
        const int local_m = i / XMX_TN;
        const int local_n = i % XMX_TN;
        const int global_m = out_row + local_m;
        const int global_n = out_col + local_n;
        
        if (global_m < M && global_n < N) {
            const int c_idx = global_m * ldc + global_n;
            float val = my_out[i];
            if (beta == 0.0f) {
                C[c_idx] = alpha * val;
            } else {
                C[c_idx] = alpha * val + beta * C[c_idx];
            }
        }
    }
}

template <bool transpose_B = true>
inline void launch_gemm_xmx(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int grid_m = (M + XMX_BM - 1) / XMX_BM;
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int num_wg = grid_m * grid_n;
    
    sycl::range<1> global(num_wg * XMX_WG_SIZE);
    sycl::range<1> local(XMX_WG_SIZE);
    
    // Shared memory sizes
    constexpr size_t shA_size = XMX_BM * XMX_A_STRIDE;           // A tile [BM x A_STRIDE]
    constexpr size_t shB_size = XMX_BK * XMX_B_STRIDE;           // B tile [BK x B_STRIDE]
    constexpr size_t shC_size = XMX_NUM_SG * XMX_TM * XMX_TN;    // C output [NUM_SG tiles]
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<xmx_bfloat16, 1> shA_acc(sycl::range<1>(shA_size), cgh);
        sycl::local_accessor<xmx_bfloat16, 1> shB_acc(sycl::range<1>(shB_size), cgh);
        sycl::local_accessor<float, 1> shC_acc(sycl::range<1>(shC_size), cgh);
        
        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) 
            [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
            gemm_xmx_kernel<transpose_B>(
                it, A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc,
                shA_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shB_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shC_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw()
            );
        });
    });
}

// Check if XMX GEMM is available at runtime
inline bool xmx_gemm_available([[maybe_unused]] sycl::queue * stream) {
    // XMX is available if SYCL_EXT_ONEAPI_MATRIX is defined at compile time
    // Runtime check would be: device.has(sycl::aspect::ext_oneapi_matrix)
    // but that aspect isn't available in all SYCL versions
    return true;
}

// ============================================================================
// F16 XMX GEMM: C = alpha * A * B^T + beta * C
// Native F16 inputs - no conversion overhead. Ideal for flash attention.
// A[M,K] and B[N,K] are sycl::half, C[M,N] is float
// ============================================================================

template <bool transpose_B = true>
inline void gemm_xmx_f16_kernel(
    sycl::nd_item<1> it,
    const sycl::half * __restrict__ A,
    const sycl::half * __restrict__ B,
    float * __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc,
    sycl::half * shA,   // Shared memory for A tile [BM x A_STRIDE]
    sycl::half * shB,   // Shared memory for B tile [BK x B_STRIDE]
    float * shC         // Shared memory for C output [NUM_SG x TM x TN]
) {
    auto sg = it.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    const int lid = it.get_local_id(0);
    
    // 2D grid mapping: workgroup (block_m, block_n)
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int block_m = it.get_group(0) / grid_n;
    const int block_n = it.get_group(0) % grid_n;
    
    const int row_base = block_m * XMX_BM;
    const int col_base = block_n * XMX_BN;
    
    if (row_base >= M) return;
    
    // Which subgroup tile within the block (sg_m in [0,3], sg_n in [0,3])
    const int sg_m = sg_id / XMX_NUM_SG_N;
    const int sg_n = sg_id % XMX_NUM_SG_N;
    
    // Initialize accumulator to zero
    joint_matrix<sycl::sub_group, float, use::accumulator, XMX_TM, XMX_TN> matC;
    joint_matrix_fill(sg, matC, 0.0f);
    
    // Loop over K dimension in blocks
    const int num_k_blocks = (K + XMX_BK - 1) / XMX_BK;
    
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_base = k_block * XMX_BK;
        
        // === Load A tile [BM x BK] directly as F16 (no conversion) ===
        constexpr int A_TILE_ELEMS = XMX_BM * XMX_BK;
        for (int idx = lid; idx < A_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_m = idx / XMX_BK;
            const int tile_k = idx % XMX_BK;
            const int global_m = row_base + tile_m;
            const int global_k = k_base + tile_k;
            
            sycl::half val = sycl::half(0.0f);
            if (global_m < M && global_k < K) {
                val = A[global_m * lda + global_k];
            }
            shA[tile_m * XMX_A_STRIDE + tile_k] = val;
        }
        
        // === Load B tile [BN x BK] -> store as [BK x BN] ===
        constexpr int B_TILE_ELEMS = XMX_BK * XMX_BN;
        for (int idx = lid; idx < B_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_k = idx / XMX_BN;
            const int tile_n = idx % XMX_BN;
            const int global_k = k_base + tile_k;
            const int global_n = col_base + tile_n;
            
            sycl::half val = sycl::half(0.0f);
            if constexpr (transpose_B) {
                // B^T: input B is [N,K] row-major
                if (global_n < N && global_k < K) {
                    val = B[global_n * ldb + global_k];
                }
            } else {
                // No transpose: B is [K,N] row-major
                if (global_k < K && global_n < N) {
                    val = B[global_k * ldb + global_n];
                }
            }
            shB[tile_k * XMX_B_STRIDE + tile_n] = val;
        }
        
        sycl::group_barrier(it.get_group());
        
        // === Compute using XMX with F16 inputs ===
        joint_matrix<sycl::sub_group, sycl::half, use::a, XMX_TM, XMX_TK, layout::row_major> matA;
        joint_matrix<sycl::sub_group, sycl::half, use::b, XMX_TK, XMX_TN, layout::row_major> matB;
        
        #pragma unroll
        for (int k_tile = 0; k_tile < XMX_BK; k_tile += XMX_TK) {
            auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(shA + sg_m * XMX_TM * XMX_A_STRIDE + k_tile);
            joint_matrix_load(sg, matA, a_ptr, XMX_A_STRIDE);
            
            auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(shB + k_tile * XMX_B_STRIDE + sg_n * XMX_TN);
            joint_matrix_load(sg, matB, b_ptr, XMX_B_STRIDE);
            
            joint_matrix_mad(sg, matC, matA, matB, matC);
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    // === Store results ===
    const int out_row = row_base + sg_m * XMX_TM;
    const int out_col = col_base + sg_n * XMX_TN;
    
    float * my_out = shC + sg_id * XMX_TM * XMX_TN;
    auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
        sycl::access::decorated::no>(my_out);
    joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
    
    sycl::group_barrier(it.get_group());
    
    // Write to global memory with bounds checking
    for (int i = lane_id; i < XMX_TM * XMX_TN; i += XMX_SG_SIZE) {
        const int local_m = i / XMX_TN;
        const int local_n = i % XMX_TN;
        const int global_m = out_row + local_m;
        const int global_n = out_col + local_n;
        
        if (global_m < M && global_n < N) {
            const int c_idx = global_m * ldc + global_n;
            float val = my_out[i];
            if (beta == 0.0f) {
                C[c_idx] = alpha * val;
            } else {
                C[c_idx] = alpha * val + beta * C[c_idx];
            }
        }
    }
}

// Launch F16 XMX GEMM: A and B are sycl::half, C is float
template <bool transpose_B = true>
inline void launch_gemm_xmx_f16(
    sycl::queue * stream,
    const sycl::half * A, const sycl::half * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int grid_m = (M + XMX_BM - 1) / XMX_BM;
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int num_wg = grid_m * grid_n;
    
    sycl::range<1> global(num_wg * XMX_WG_SIZE);
    sycl::range<1> local(XMX_WG_SIZE);
    
    constexpr size_t shA_size = XMX_BM * XMX_A_STRIDE;
    constexpr size_t shB_size = XMX_BK * XMX_B_STRIDE;
    constexpr size_t shC_size = XMX_NUM_SG * XMX_TM * XMX_TN;
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> shA_acc(sycl::range<1>(shA_size), cgh);
        sycl::local_accessor<sycl::half, 1> shB_acc(sycl::range<1>(shB_size), cgh);
        sycl::local_accessor<float, 1> shC_acc(sycl::range<1>(shC_size), cgh);
        
        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) 
            [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
            gemm_xmx_f16_kernel<transpose_B>(
                it, A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc,
                shA_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shB_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shC_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw()
            );
        });
    });
}

// ============================================================================
// F16 XMX GEMM with F16 output: C = alpha * A * B^T + beta * C
// All tensors are sycl::half. Useful when output doesn't need F32 precision.
// ============================================================================

template <bool transpose_B = true>
inline void gemm_xmx_f16_f16_kernel(
    sycl::nd_item<1> it,
    const sycl::half * __restrict__ A,
    const sycl::half * __restrict__ B,
    sycl::half * __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc,
    sycl::half * shA,
    sycl::half * shB,
    float * shC
) {
    auto sg = it.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    const int lid = it.get_local_id(0);
    
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int block_m = it.get_group(0) / grid_n;
    const int block_n = it.get_group(0) % grid_n;
    
    const int row_base = block_m * XMX_BM;
    const int col_base = block_n * XMX_BN;
    
    if (row_base >= M) return;
    
    const int sg_m = sg_id / XMX_NUM_SG_N;
    const int sg_n = sg_id % XMX_NUM_SG_N;
    
    joint_matrix<sycl::sub_group, float, use::accumulator, XMX_TM, XMX_TN> matC;
    joint_matrix_fill(sg, matC, 0.0f);
    
    const int num_k_blocks = (K + XMX_BK - 1) / XMX_BK;
    
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_base = k_block * XMX_BK;
        
        // Load A tile
        constexpr int A_TILE_ELEMS = XMX_BM * XMX_BK;
        for (int idx = lid; idx < A_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_m = idx / XMX_BK;
            const int tile_k = idx % XMX_BK;
            const int global_m = row_base + tile_m;
            const int global_k = k_base + tile_k;
            
            sycl::half val = sycl::half(0.0f);
            if (global_m < M && global_k < K) {
                val = A[global_m * lda + global_k];
            }
            shA[tile_m * XMX_A_STRIDE + tile_k] = val;
        }
        
        // Load B tile
        constexpr int B_TILE_ELEMS = XMX_BK * XMX_BN;
        for (int idx = lid; idx < B_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_k = idx / XMX_BN;
            const int tile_n = idx % XMX_BN;
            const int global_k = k_base + tile_k;
            const int global_n = col_base + tile_n;
            
            sycl::half val = sycl::half(0.0f);
            if constexpr (transpose_B) {
                if (global_n < N && global_k < K) {
                    val = B[global_n * ldb + global_k];
                }
            } else {
                if (global_k < K && global_n < N) {
                    val = B[global_k * ldb + global_n];
                }
            }
            shB[tile_k * XMX_B_STRIDE + tile_n] = val;
        }
        
        sycl::group_barrier(it.get_group());
        
        // XMX compute
        joint_matrix<sycl::sub_group, sycl::half, use::a, XMX_TM, XMX_TK, layout::row_major> matA;
        joint_matrix<sycl::sub_group, sycl::half, use::b, XMX_TK, XMX_TN, layout::row_major> matB;
        
        #pragma unroll
        for (int k_tile = 0; k_tile < XMX_BK; k_tile += XMX_TK) {
            auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(shA + sg_m * XMX_TM * XMX_A_STRIDE + k_tile);
            joint_matrix_load(sg, matA, a_ptr, XMX_A_STRIDE);
            
            auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                sycl::access::decorated::no>(shB + k_tile * XMX_B_STRIDE + sg_n * XMX_TN);
            joint_matrix_load(sg, matB, b_ptr, XMX_B_STRIDE);
            
            joint_matrix_mad(sg, matC, matA, matB, matC);
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    // Store to shared memory
    const int out_row = row_base + sg_m * XMX_TM;
    const int out_col = col_base + sg_n * XMX_TN;
    
    float * my_out = shC + sg_id * XMX_TM * XMX_TN;
    auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
        sycl::access::decorated::no>(my_out);
    joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
    
    sycl::group_barrier(it.get_group());
    
    // Write to global memory as F16
    for (int i = lane_id; i < XMX_TM * XMX_TN; i += XMX_SG_SIZE) {
        const int local_m = i / XMX_TN;
        const int local_n = i % XMX_TN;
        const int global_m = out_row + local_m;
        const int global_n = out_col + local_n;
        
        if (global_m < M && global_n < N) {
            const int c_idx = global_m * ldc + global_n;
            float val = my_out[i];
            if (beta == 0.0f) {
                C[c_idx] = sycl::half(alpha * val);
            } else {
                C[c_idx] = sycl::half(alpha * val + beta * float(C[c_idx]));
            }
        }
    }
}

// Launch F16->F16 XMX GEMM
template <bool transpose_B = true>
inline void launch_gemm_xmx_f16_f16(
    sycl::queue * stream,
    const sycl::half * A, const sycl::half * B, sycl::half * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int grid_m = (M + XMX_BM - 1) / XMX_BM;
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int num_wg = grid_m * grid_n;
    
    sycl::range<1> global(num_wg * XMX_WG_SIZE);
    sycl::range<1> local(XMX_WG_SIZE);
    
    constexpr size_t shA_size = XMX_BM * XMX_A_STRIDE;
    constexpr size_t shB_size = XMX_BK * XMX_B_STRIDE;
    constexpr size_t shC_size = XMX_NUM_SG * XMX_TM * XMX_TN;
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> shA_acc(sycl::range<1>(shA_size), cgh);
        sycl::local_accessor<sycl::half, 1> shB_acc(sycl::range<1>(shB_size), cgh);
        sycl::local_accessor<float, 1> shC_acc(sycl::range<1>(shC_size), cgh);
        
        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) 
            [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
            gemm_xmx_f16_f16_kernel<transpose_B>(
                it, A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc,
                shA_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shB_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shC_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw()
            );
        });
    });
}

#else // !SYCL_EXT_ONEAPI_MATRIX

// Stub when XMX not available
template <bool transpose_B = true>
inline void launch_gemm_xmx(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    (void)stream; (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)alpha; (void)beta;
    (void)lda; (void)ldb; (void)ldc;
    // XMX not available - caller should check xmx_gemm_available() first
}

inline bool xmx_gemm_available(sycl::queue *) {
    return false;
}

template <bool transpose_B = true>
inline void launch_gemm_xmx_f16(
    sycl::queue * stream,
    const sycl::half * A, const sycl::half * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    (void)stream; (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)alpha; (void)beta;
    (void)lda; (void)ldb; (void)ldc;
}

template <bool transpose_B = true>
inline void launch_gemm_xmx_f16_f16(
    sycl::queue * stream,
    const sycl::half * A, const sycl::half * B, sycl::half * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    (void)stream; (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;
    (void)alpha; (void)beta;
    (void)lda; (void)ldb; (void)ldc;
}

#endif // SYCL_EXT_ONEAPI_MATRIX

#endif // GGML_SYCL_GEMM_XMX_HPP
