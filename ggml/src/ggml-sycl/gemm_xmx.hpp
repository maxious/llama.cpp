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
constexpr int XMX_BM = 32;   // 4 TM tiles = 32 rows
constexpr int XMX_BN = 64;   // 4 TN tiles = 64 cols  
constexpr int XMX_BK = 64;   // 4 TK tiles = 64 K elements (larger K block reduces barrier overhead)

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

template <bool transpose_A = false, bool transpose_B = true>
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
            if constexpr (transpose_A) {
                if (global_m < M && global_k < K) {
                    val = A[global_k * lda + global_m];
                }
            } else {
                if (global_m < M && global_k < K) {
                    val = A[global_m * lda + global_k];
                }
            }
            shA[tile_m * XMX_A_STRIDE + tile_k] = xmx_bfloat16(val);
        }
        
        // === Load B tile: coalesced along K for transpose_B ===
        // When transpose_B: B is [N,K] row-major. Threads iterate (k,n) with n
        // varying fastest. Each thread loads B[n, k] = B[global_n * ldb + global_k].
        // For consecutive thread indices, global_n increments by 1, so loads are
        // at stride-1 (contiguous) along the K dimension within each row — coalesced.
        // Store as [BK x BN] row-major in SLM for joint_matrix row-major load.
        constexpr int B_TILE_ELEMS = XMX_BK * XMX_BN;
        for (int idx = lid; idx < B_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_k = idx / XMX_BN;
            const int tile_n = idx % XMX_BN;
            const int global_k = k_base + tile_k;
            const int global_n = col_base + tile_n;
            
            float val = 0.0f;
            if constexpr (transpose_B) {
                if (global_n < N && global_k < K) {
                    val = B[global_n * ldb + global_k];
                }
            } else {
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
    
    // For the common case (beta==0, alpha==1), try direct global store
    // to avoid the SLM round-trip through shC
    if (beta == 0.0f && alpha == 1.0f &&
        out_row + XMX_TM <= M && out_col + XMX_TN <= N) {
        // Full tile, no bounds checking needed — store directly to global
        auto c_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(C + out_row * ldc + out_col);
        joint_matrix_store(sg, matC, c_ptr, ldc, layout::row_major);
    } else {
        // Edge tile or non-trivial alpha/beta: use SLM staging
        float * my_out = shC + sg_id * XMX_TM * XMX_TN;
        auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
            sycl::access::decorated::no>(my_out);
        joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
        
        // Only sub-group barrier needed — shC regions don't overlap between subgroups
        sycl::group_barrier(sg);
        
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
}

template <bool transpose_A = false, bool transpose_B = true>
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
            gemm_xmx_kernel<transpose_A, transpose_B>(
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

template <bool transpose_A = false, bool transpose_B = true>
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
            if constexpr (transpose_A) {
                if (global_m < M && global_k < K) {
                    val = A[global_k * lda + global_m];
                }
            } else {
                if (global_m < M && global_k < K) {
                    val = A[global_m * lda + global_k];
                }
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
    
    if (beta == 0.0f && alpha == 1.0f &&
        out_row + XMX_TM <= M && out_col + XMX_TN <= N) {
        auto c_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(C + out_row * ldc + out_col);
        joint_matrix_store(sg, matC, c_ptr, ldc, layout::row_major);
    } else {
        float * my_out = shC + sg_id * XMX_TM * XMX_TN;
        auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
            sycl::access::decorated::no>(my_out);
        joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
        
        sycl::group_barrier(sg);
        
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
}

// Launch F16 XMX GEMM: A and B are sycl::half, C is float
template <bool transpose_A = false, bool transpose_B = true>
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
            gemm_xmx_f16_kernel<transpose_A, transpose_B>(
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

template <bool transpose_A = false, bool transpose_B = true>
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
            if constexpr (transpose_A) {
                if (global_m < M && global_k < K) {
                    val = A[global_k * lda + global_m];
                }
            } else {
                if (global_m < M && global_k < K) {
                    val = A[global_m * lda + global_k];
                }
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
    
    const int out_row = row_base + sg_m * XMX_TM;
    const int out_col = col_base + sg_n * XMX_TN;
    
    // F16 output requires F32->F16 conversion, so always use SLM staging
    float * my_out = shC + sg_id * XMX_TM * XMX_TN;
    auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
        sycl::access::decorated::no>(my_out);
    joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
    
    sycl::group_barrier(sg);
    
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
template <bool transpose_A = false, bool transpose_B = true>
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
            gemm_xmx_f16_f16_kernel<transpose_A, transpose_B>(
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
// Indirect (MoE) XMX GEMM variants
// Same as above but M is read from device pointer (expert row count)
// and offset applied to A/C for expert packing
// ============================================================================

// F32 indirect: C = alpha * A * B^T + beta * C (F32 inputs, BF16 XMX compute)
template <bool transpose_A = false, bool transpose_B = true>
inline void gemm_xmx_kernel_indirect(
    sycl::nd_item<1> it,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const int * __restrict__ M_ptr,
    const int * __restrict__ offset_ptr,
    const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc,
    xmx_bfloat16 * shA,
    xmx_bfloat16 * shB,
    float * shC
) {
    const int M = *M_ptr;
    if (M <= 0) return;

    const int offset = (offset_ptr) ? *offset_ptr : 0;
    const float * A_ptr = A;
    float * C_ptr = C;
    if (offset > 0) {
        if constexpr (transpose_A) {
            A_ptr += offset;
        } else {
            A_ptr += offset * lda;
        }
        C_ptr += offset * ldc;
    }

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

        constexpr int A_TILE_ELEMS = XMX_BM * XMX_BK;
        for (int idx = lid; idx < A_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_m = idx / XMX_BK;
            const int tile_k = idx % XMX_BK;
            const int global_m = row_base + tile_m;
            const int global_k = k_base + tile_k;

            float val = 0.0f;
            if constexpr (transpose_A) {
                if (global_m < M && global_k < K) val = A_ptr[global_k * lda + global_m];
            } else {
                if (global_m < M && global_k < K) val = A_ptr[global_m * lda + global_k];
            }
            shA[tile_m * XMX_A_STRIDE + tile_k] = xmx_bfloat16(val);
        }

        constexpr int B_TILE_ELEMS = XMX_BK * XMX_BN;
        for (int idx = lid; idx < B_TILE_ELEMS; idx += XMX_WG_SIZE) {
            const int tile_k = idx / XMX_BN;
            const int tile_n = idx % XMX_BN;
            const int global_k = k_base + tile_k;
            const int global_n = col_base + tile_n;

            float val = 0.0f;
            if constexpr (transpose_B) {
                if (global_n < N && global_k < K) val = B[global_n * ldb + global_k];
            } else {
                if (global_k < K && global_n < N) val = B[global_k * ldb + global_n];
            }
            shB[tile_k * XMX_B_STRIDE + tile_n] = xmx_bfloat16(val);
        }

        sycl::group_barrier(it.get_group());

        joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, XMX_TM, XMX_TK, layout::row_major> matA;
        joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, XMX_TK, XMX_TN, layout::row_major> matB;

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

    const int out_row = row_base + sg_m * XMX_TM;
    const int out_col = col_base + sg_n * XMX_TN;

    if (beta == 0.0f && alpha == 1.0f &&
        out_row + XMX_TM <= M && out_col + XMX_TN <= N) {
        auto c_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(C_ptr + out_row * ldc + out_col);
        joint_matrix_store(sg, matC, c_ptr, ldc, layout::row_major);
    } else {
        float * my_out = shC + sg_id * XMX_TM * XMX_TN;
        auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
            sycl::access::decorated::no>(my_out);
        joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);

        sycl::group_barrier(sg);

        for (int i = lane_id; i < XMX_TM * XMX_TN; i += XMX_SG_SIZE) {
            const int local_m = i / XMX_TN;
            const int local_n = i % XMX_TN;
            const int global_m = out_row + local_m;
            const int global_n = out_col + local_n;

            if (global_m < M && global_n < N) {
                const int c_idx = global_m * ldc + global_n;
                float val = my_out[i];
                if (beta == 0.0f) {
                    C_ptr[c_idx] = alpha * val;
                } else {
                    C_ptr[c_idx] = alpha * val + beta * C_ptr[c_idx];
                }
            }
        }
    }
}

template <bool transpose_B = true>
inline void launch_gemm_xmx_indirect(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int * M_ptr, const int * offset_ptr,
    const int max_M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int grid_m = (max_M + XMX_BM - 1) / XMX_BM;
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int num_wg = grid_m * grid_n;

    sycl::range<1> global(num_wg * XMX_WG_SIZE);
    sycl::range<1> local(XMX_WG_SIZE);

    constexpr size_t shA_size = XMX_BM * XMX_A_STRIDE;
    constexpr size_t shB_size = XMX_BK * XMX_B_STRIDE;
    constexpr size_t shC_size = XMX_NUM_SG * XMX_TM * XMX_TN;

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<xmx_bfloat16, 1> shA_acc(sycl::range<1>(shA_size), cgh);
        sycl::local_accessor<xmx_bfloat16, 1> shB_acc(sycl::range<1>(shB_size), cgh);
        sycl::local_accessor<float, 1> shC_acc(sycl::range<1>(shC_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it)
            [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
            gemm_xmx_kernel_indirect<false, transpose_B>(
                it, A, B, C, M_ptr, offset_ptr,
                N, K, alpha, beta, lda, ldb, ldc,
                shA_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shB_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw(),
                shC_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw()
            );
        });
    });
}

// F16 indirect: A is F32 activations, B is F16 weights, C is F32 output
// Both A and B are converted to BF16 in SLM for uniform XMX compute
template <bool transpose_B = true>
inline void launch_gemm_xmx_indirect_f32_f16(
    sycl::queue * stream,
    const float * A, const sycl::half * B, float * C,
    const int * M_ptr, const int * offset_ptr,
    const int max_M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int grid_m = (max_M + XMX_BM - 1) / XMX_BM;
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int num_wg = grid_m * grid_n;

    sycl::range<1> global(num_wg * XMX_WG_SIZE);
    sycl::range<1> local(XMX_WG_SIZE);

    constexpr size_t shA_size = XMX_BM * XMX_A_STRIDE;
    constexpr size_t shB_size = XMX_BK * XMX_B_STRIDE;
    constexpr size_t shC_size = XMX_NUM_SG * XMX_TM * XMX_TN;

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<xmx_bfloat16, 1> shA_acc(sycl::range<1>(shA_size), cgh);
        sycl::local_accessor<xmx_bfloat16, 1> shB_acc(sycl::range<1>(shB_size), cgh);
        sycl::local_accessor<float, 1> shC_acc(sycl::range<1>(shC_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it)
            [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
            // Read M and offset from device pointers
            const int M = *M_ptr;
            if (M <= 0) return;
            const int offset = (offset_ptr) ? *offset_ptr : 0;

            const float * A_ptr = A + (offset > 0 ? offset * lda : 0);
            float * C_ptr = C + (offset > 0 ? offset * ldc : 0);

            auto sg = it.get_sub_group();
            const int sg_id = sg.get_group_linear_id();
            const int lane_id = sg.get_local_linear_id();
            const int lid = it.get_local_id(0);

            const int grid_n_local = (N + XMX_BN - 1) / XMX_BN;
            const int block_m = it.get_group(0) / grid_n_local;
            const int block_n = it.get_group(0) % grid_n_local;
            const int row_base = block_m * XMX_BM;
            const int col_base = block_n * XMX_BN;
            if (row_base >= M) return;

            const int sg_m = sg_id / XMX_NUM_SG_N;
            const int sg_n = sg_id % XMX_NUM_SG_N;

            auto * shA = shA_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw();
            auto * shB = shB_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw();
            auto * shC = shC_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw();

            joint_matrix<sycl::sub_group, float, use::accumulator, XMX_TM, XMX_TN> matC;
            joint_matrix_fill(sg, matC, 0.0f);

            const int num_k_blocks = (K + XMX_BK - 1) / XMX_BK;
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int k_base = k_block * XMX_BK;

                // Load A (F32) -> BF16
                constexpr int A_TILE_ELEMS = XMX_BM * XMX_BK;
                for (int idx = lid; idx < A_TILE_ELEMS; idx += XMX_WG_SIZE) {
                    const int tile_m = idx / XMX_BK, tile_k = idx % XMX_BK;
                    const int gm = row_base + tile_m, gk = k_base + tile_k;
                    float val = (gm < M && gk < K) ? A_ptr[gm * lda + gk] : 0.0f;
                    shA[tile_m * XMX_A_STRIDE + tile_k] = xmx_bfloat16(val);
                }

                // Load B (F16) -> BF16
                constexpr int B_TILE_ELEMS = XMX_BK * XMX_BN;
                for (int idx = lid; idx < B_TILE_ELEMS; idx += XMX_WG_SIZE) {
                    const int tile_k = idx / XMX_BN, tile_n = idx % XMX_BN;
                    const int gk = k_base + tile_k, gn = col_base + tile_n;
                    float val = 0.0f;
                    if constexpr (transpose_B) {
                        if (gn < N && gk < K) val = float(B[gn * ldb + gk]);
                    } else {
                        if (gk < K && gn < N) val = float(B[gk * ldb + gn]);
                    }
                    shB[tile_k * XMX_B_STRIDE + tile_n] = xmx_bfloat16(val);
                }

                sycl::group_barrier(it.get_group());

                joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, XMX_TM, XMX_TK, layout::row_major> matA;
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, XMX_TK, XMX_TN, layout::row_major> matB;

                #pragma unroll
                for (int k_tile = 0; k_tile < XMX_BK; k_tile += XMX_TK) {
                    auto a_p = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(shA + sg_m * XMX_TM * XMX_A_STRIDE + k_tile);
                    joint_matrix_load(sg, matA, a_p, XMX_A_STRIDE);
                    auto b_p = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(shB + k_tile * XMX_B_STRIDE + sg_n * XMX_TN);
                    joint_matrix_load(sg, matB, b_p, XMX_B_STRIDE);
                    joint_matrix_mad(sg, matC, matA, matB, matC);
                }
                sycl::group_barrier(it.get_group());
            }

            const int out_row = row_base + sg_m * XMX_TM;
            const int out_col = col_base + sg_n * XMX_TN;

            if (beta == 0.0f && alpha == 1.0f &&
                out_row + XMX_TM <= M && out_col + XMX_TN <= N) {
                auto c_gptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(C_ptr + out_row * ldc + out_col);
                joint_matrix_store(sg, matC, c_gptr, ldc, layout::row_major);
            } else {
                float * my_out = shC + sg_id * XMX_TM * XMX_TN;
                auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(my_out);
                joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
                sycl::group_barrier(sg);

                for (int i = lane_id; i < XMX_TM * XMX_TN; i += XMX_SG_SIZE) {
                    const int lm = i / XMX_TN, ln = i % XMX_TN;
                    const int gm = out_row + lm, gn = out_col + ln;
                    if (gm < M && gn < N) {
                        float val = my_out[i];
                        C_ptr[gm * ldc + gn] = (beta == 0.0f) ? alpha * val : alpha * val + beta * C_ptr[gm * ldc + gn];
                    }
                }
            }
        });
    });
}

// BF16 indirect: A is F32 activations, B is BF16 weights, C is F32 output
// BF16 is the native XMX compute type — B loads with zero conversion overhead
template <bool transpose_B = true>
inline void launch_gemm_xmx_indirect_f32_bf16(
    sycl::queue * stream,
    const float * A, const sycl::ext::oneapi::bfloat16 * B, float * C,
    const int * M_ptr, const int * offset_ptr,
    const int max_M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int grid_m = (max_M + XMX_BM - 1) / XMX_BM;
    const int grid_n = (N + XMX_BN - 1) / XMX_BN;
    const int num_wg = grid_m * grid_n;

    sycl::range<1> global(num_wg * XMX_WG_SIZE);
    sycl::range<1> local(XMX_WG_SIZE);

    constexpr size_t shA_size = XMX_BM * XMX_A_STRIDE;
    constexpr size_t shB_size = XMX_BK * XMX_B_STRIDE;
    constexpr size_t shC_size = XMX_NUM_SG * XMX_TM * XMX_TN;

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<xmx_bfloat16, 1> shA_acc(sycl::range<1>(shA_size), cgh);
        sycl::local_accessor<xmx_bfloat16, 1> shB_acc(sycl::range<1>(shB_size), cgh);
        sycl::local_accessor<float, 1> shC_acc(sycl::range<1>(shC_size), cgh);

        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it)
            [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
            const int M = *M_ptr;
            if (M <= 0) return;
            const int offset = (offset_ptr) ? *offset_ptr : 0;

            const float * A_ptr = A + (offset > 0 ? offset * lda : 0);
            float * C_ptr = C + (offset > 0 ? offset * ldc : 0);

            auto sg = it.get_sub_group();
            const int sg_id = sg.get_group_linear_id();
            const int lane_id = sg.get_local_linear_id();
            const int lid = it.get_local_id(0);

            const int grid_n_local = (N + XMX_BN - 1) / XMX_BN;
            const int block_m = it.get_group(0) / grid_n_local;
            const int block_n = it.get_group(0) % grid_n_local;
            const int row_base = block_m * XMX_BM;
            const int col_base = block_n * XMX_BN;
            if (row_base >= M) return;

            const int sg_m = sg_id / XMX_NUM_SG_N;
            const int sg_n = sg_id % XMX_NUM_SG_N;

            auto * shA = shA_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw();
            auto * shB = shB_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw();
            auto * shC = shC_acc.template get_multi_ptr<sycl::access::decorated::no>().get_raw();

            joint_matrix<sycl::sub_group, float, use::accumulator, XMX_TM, XMX_TN> matC;
            joint_matrix_fill(sg, matC, 0.0f);

            const int num_k_blocks = (K + XMX_BK - 1) / XMX_BK;
            for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
                const int k_base = k_block * XMX_BK;

                // Load A (F32) -> BF16
                constexpr int A_TILE_ELEMS = XMX_BM * XMX_BK;
                for (int idx = lid; idx < A_TILE_ELEMS; idx += XMX_WG_SIZE) {
                    const int tile_m = idx / XMX_BK, tile_k = idx % XMX_BK;
                    const int gm = row_base + tile_m, gk = k_base + tile_k;
                    float val = (gm < M && gk < K) ? A_ptr[gm * lda + gk] : 0.0f;
                    shA[tile_m * XMX_A_STRIDE + tile_k] = xmx_bfloat16(val);
                }

                // Load B (BF16) -> BF16 (native, zero conversion)
                constexpr int B_TILE_ELEMS = XMX_BK * XMX_BN;
                for (int idx = lid; idx < B_TILE_ELEMS; idx += XMX_WG_SIZE) {
                    const int tile_k = idx / XMX_BN, tile_n = idx % XMX_BN;
                    const int gk = k_base + tile_k, gn = col_base + tile_n;
                    xmx_bfloat16 val = xmx_bfloat16(0.0f);
                    if constexpr (transpose_B) {
                        if (gn < N && gk < K) val = B[gn * ldb + gk];
                    } else {
                        if (gk < K && gn < N) val = B[gk * ldb + gn];
                    }
                    shB[tile_k * XMX_B_STRIDE + tile_n] = val;
                }

                sycl::group_barrier(it.get_group());

                joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, XMX_TM, XMX_TK, layout::row_major> matA;
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, XMX_TK, XMX_TN, layout::row_major> matB;

                #pragma unroll
                for (int k_tile = 0; k_tile < XMX_BK; k_tile += XMX_TK) {
                    auto a_p = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(shA + sg_m * XMX_TM * XMX_A_STRIDE + k_tile);
                    joint_matrix_load(sg, matA, a_p, XMX_A_STRIDE);
                    auto b_p = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(shB + k_tile * XMX_B_STRIDE + sg_n * XMX_TN);
                    joint_matrix_load(sg, matB, b_p, XMX_B_STRIDE);
                    joint_matrix_mad(sg, matC, matA, matB, matC);
                }
                sycl::group_barrier(it.get_group());
            }

            const int out_row = row_base + sg_m * XMX_TM;
            const int out_col = col_base + sg_n * XMX_TN;

            if (beta == 0.0f && alpha == 1.0f &&
                out_row + XMX_TM <= M && out_col + XMX_TN <= N) {
                auto c_gptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(C_ptr + out_row * ldc + out_col);
                joint_matrix_store(sg, matC, c_gptr, ldc, layout::row_major);
            } else {
                float * my_out = shC + sg_id * XMX_TM * XMX_TN;
                auto out_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(my_out);
                joint_matrix_store(sg, matC, out_ptr, XMX_TN, layout::row_major);
                sycl::group_barrier(sg);

                for (int i = lane_id; i < XMX_TM * XMX_TN; i += XMX_SG_SIZE) {
                    const int lm = i / XMX_TN, ln = i % XMX_TN;
                    const int gm = out_row + lm, gn = out_col + ln;
                    if (gm < M && gn < N) {
                        float val = my_out[i];
                        C_ptr[gm * ldc + gn] = (beta == 0.0f) ? alpha * val : alpha * val + beta * C_ptr[gm * ldc + gn];
                    }
                }
            }
        });
    });
}

#else // !SYCL_EXT_ONEAPI_MATRIX

// Stub when XMX not available
template <bool transpose_A = false, bool transpose_B = true>
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

template <bool transpose_A = false, bool transpose_B = true>
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

template <bool transpose_A = false, bool transpose_B = true>
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

template <bool transpose_B = true>
inline void launch_gemm_xmx_indirect(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int * M_ptr, const int * offset_ptr,
    const int max_M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    (void)stream; (void)A; (void)B; (void)C; (void)M_ptr; (void)offset_ptr;
    (void)max_M; (void)N; (void)K; (void)alpha; (void)beta;
    (void)lda; (void)ldb; (void)ldc;
}

template <bool transpose_B = true>
inline void launch_gemm_xmx_indirect_f32_f16(
    sycl::queue * stream,
    const float * A, const sycl::half * B, float * C,
    const int * M_ptr, const int * offset_ptr,
    const int max_M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    (void)stream; (void)A; (void)B; (void)C; (void)M_ptr; (void)offset_ptr;
    (void)max_M; (void)N; (void)K; (void)alpha; (void)beta;
    (void)lda; (void)ldb; (void)ldc;
}

template <bool transpose_B = true>
inline void launch_gemm_xmx_indirect_f32_bf16(
    sycl::queue * stream,
    const float * A, const sycl::ext::oneapi::bfloat16 * B, float * C,
    const int * M_ptr, const int * offset_ptr,
    const int max_M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    (void)stream; (void)A; (void)B; (void)C; (void)M_ptr; (void)offset_ptr;
    (void)max_M; (void)N; (void)K; (void)alpha; (void)beta;
    (void)lda; (void)ldb; (void)ldc;
}

#endif // SYCL_EXT_ONEAPI_MATRIX

#endif // GGML_SYCL_GEMM_XMX_HPP
