#ifndef GGML_SYCL_GEMM_XMX_HPP
#define GGML_SYCL_GEMM_XMX_HPP

#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif

// XMX-accelerated Graph-Compatible GEMM: C = alpha * A * B^T + beta * C
// Uses Intel XMX (Xe Matrix eXtensions) via SYCL joint_matrix
// Row-major matrices: A[M,K], B[N,K] (for B^T), C[M,N]
//
// This provides much higher performance than scalar tiled GEMM by using
// hardware matrix units. Still graph-compatible (no oneMKL).

#ifdef SYCL_EXT_COOPERATIVE_MATRICES

using xmx_bfloat16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

// XMX tile sizes for Intel Xe2 (Arc)
// TM x TK @ TK x TN -> TM x TN accumulator
constexpr int XMX_TM = 8;
constexpr int XMX_TN = 16;
constexpr int XMX_TK = 16;

// Block sizes - how many XMX tiles per workgroup
constexpr int XMX_BM = 64;   // 8 TM tiles = 64 rows
constexpr int XMX_BN = 64;   // 4 TN tiles = 64 cols
constexpr int XMX_BK = 32;   // 2 TK tiles = 32 K elements

// Subgroups per workgroup
constexpr int XMX_NUM_SG_M = XMX_BM / XMX_TM;  // 8
constexpr int XMX_NUM_SG_N = XMX_BN / XMX_TN;  // 4
constexpr int XMX_NUM_SG = XMX_NUM_SG_M * XMX_NUM_SG_N;  // 32

// Subgroup size for Intel GPUs
constexpr int XMX_SG_SIZE = 16;

// Threads per workgroup
constexpr int XMX_WG_SIZE = XMX_NUM_SG * XMX_SG_SIZE;  // 512

// Shared memory strides with padding to avoid bank conflicts
constexpr int XMX_A_STRIDE = XMX_BK + 8;  // Avoid 32-byte bank conflicts
constexpr int XMX_B_STRIDE = XMX_BK + 8;

template <bool transpose_B = true>
inline void gemm_xmx_kernel(
    sycl::nd_item<1> it,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc,
    xmx_bfloat16 * shA,  // Shared memory for A tile [BM x BK]
    xmx_bfloat16 * shB   // Shared memory for B tile [BN x BK]
) {
    auto sg = it.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    const int lid = it.get_local_id(0);
    
    // Which block this workgroup handles
    const int block_m = it.get_group(0) / ((N + XMX_BN - 1) / XMX_BN);
    const int block_n = it.get_group(0) % ((N + XMX_BN - 1) / XMX_BN);
    
    const int row_base = block_m * XMX_BM;
    const int col_base = block_n * XMX_BN;
    
    if (row_base >= M) return;
    
    // Which subgroup tile within the block
    const int sg_m = sg_id / XMX_NUM_SG_N;  // 0..7
    const int sg_n = sg_id % XMX_NUM_SG_N;  // 0..3
    
    // Initialize accumulator
    joint_matrix<sycl::sub_group, float, use::accumulator, XMX_TM, XMX_TN> matC;
    joint_matrix_fill(sg, matC, 0.0f);
    
    // Loop over K in blocks
    const int num_k_blocks = (K + XMX_BK - 1) / XMX_BK;
    
    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        const int k_base = k_block * XMX_BK;
        
        // Cooperatively load A tile [BM x BK] into shared memory as bfloat16
        // Each thread loads multiple elements
        const int a_loads = (XMX_BM * XMX_BK + XMX_WG_SIZE - 1) / XMX_WG_SIZE;
        for (int load = 0; load < a_loads; ++load) {
            const int flat_idx = lid + load * XMX_WG_SIZE;
            if (flat_idx < XMX_BM * XMX_BK) {
                const int tile_m = flat_idx / XMX_BK;
                const int tile_k = flat_idx % XMX_BK;
                const int global_m = row_base + tile_m;
                const int global_k = k_base + tile_k;
                
                float val = 0.0f;
                if (global_m < M && global_k < K) {
                    val = A[global_m * lda + global_k];
                }
                shA[tile_m * XMX_A_STRIDE + tile_k] = xmx_bfloat16(val);
            }
        }
        
        // Cooperatively load B tile [BN x BK] into shared memory as bfloat16
        const int b_loads = (XMX_BN * XMX_BK + XMX_WG_SIZE - 1) / XMX_WG_SIZE;
        for (int load = 0; load < b_loads; ++load) {
            const int flat_idx = lid + load * XMX_WG_SIZE;
            if (flat_idx < XMX_BN * XMX_BK) {
                const int tile_n = flat_idx / XMX_BK;
                const int tile_k = flat_idx % XMX_BK;
                const int global_n = col_base + tile_n;
                const int global_k = k_base + tile_k;
                
                float val = 0.0f;
                if constexpr (transpose_B) {
                    // B^T: B is [N,K] row-major
                    if (global_n < N && global_k < K) {
                        val = B[global_n * ldb + global_k];
                    }
                } else {
                    // B: B is [K,N] row-major
                    if (global_k < K && global_n < N) {
                        val = B[global_k * ldb + global_n];
                    }
                }
                // Store B in column-major for joint_matrix_load with use::b
                shB[tile_k * XMX_BN + tile_n] = xmx_bfloat16(val);
            }
        }
        
        sycl::group_barrier(it.get_group());
        
        // Compute using XMX
        // Each subgroup handles one TM x TN output tile
        joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, XMX_TM, XMX_TK, layout::row_major> matA;
        joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, XMX_TK, XMX_TN, layout::col_major> matB;
        
        // Loop over K tiles within this K block
        for (int k_tile = 0; k_tile < XMX_BK; k_tile += XMX_TK) {
            // Load A tile for this subgroup's row
            const xmx_bfloat16 * a_ptr = shA + sg_m * XMX_TM * XMX_A_STRIDE + k_tile;
            joint_matrix_load(sg, matA, a_ptr, XMX_A_STRIDE);
            
            // Load B tile for this subgroup's column (B is stored col-major in shB)
            const xmx_bfloat16 * b_ptr = shB + k_tile * XMX_BN + sg_n * XMX_TN;
            joint_matrix_load(sg, matB, b_ptr, XMX_BN);
            
            // Multiply-accumulate
            joint_matrix_mad(sg, matC, matA, matB, matC);
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    // Store results to global memory
    const int out_row = row_base + sg_m * XMX_TM;
    const int out_col = col_base + sg_n * XMX_TN;
    
    if (out_row < M && out_col < N) {
        // Extract accumulator values and write to C
        // joint_matrix_store requires contiguous memory, so we store to shared mem first
        float * out_ptr = reinterpret_cast<float*>(shA);  // Reuse shared memory
        joint_matrix_store(sg, matC, out_ptr + sg_id * XMX_TM * XMX_TN, XMX_TN, layout::row_major);
        
        sycl::group_barrier(it.get_group());
        
        // Write to global memory with bounds checking
        for (int i = lane_id; i < XMX_TM * XMX_TN; i += XMX_SG_SIZE) {
            const int local_m = i / XMX_TN;
            const int local_n = i % XMX_TN;
            const int global_m = out_row + local_m;
            const int global_n = out_col + local_n;
            
            if (global_m < M && global_n < N) {
                const int c_idx = global_m * ldc + global_n;
                float val = out_ptr[sg_id * XMX_TM * XMX_TN + i];
                if (beta == 0.0f) {
                    C[c_idx] = alpha * val;
                } else {
                    C[c_idx] = alpha * val + beta * C[c_idx];
                }
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
    constexpr size_t shA_size = XMX_BM * XMX_A_STRIDE;
    constexpr size_t shB_size = XMX_BK * XMX_BN;  // Col-major B storage
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<xmx_bfloat16, 1> shA_acc(sycl::range<1>(shA_size), cgh);
        sycl::local_accessor<xmx_bfloat16, 1> shB_acc(sycl::range<1>(shB_size), cgh);
        
        cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) 
            [[intel::reqd_sub_group_size(XMX_SG_SIZE)]] {
            gemm_xmx_kernel<transpose_B>(
                it, A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc,
                shA_acc.get_pointer(),
                shB_acc.get_pointer()
            );
        });
    });
}

#endif // SYCL_EXT_COOPERATIVE_MATRICES

#endif // GGML_SYCL_GEMM_XMX_HPP
