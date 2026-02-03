#ifndef GGML_SYCL_GEMM_F16_F32_TILED_HPP
#define GGML_SYCL_GEMM_F16_F32_TILED_HPP

#include "common.hpp"
#include <sycl/sycl.hpp>

// Tiled Graph-Compatible GEMM for F16 inputs -> F32 output
// C = alpha * A * B^T + beta * C   (when transpose_B = true)
// OR C = alpha * A * B + beta * C   (when transpose_B = false)
// A: M x K (F16), B: either (N, K) or (K, N) (F16), C: M x N (F32)
// All matrices are row-major with leading dimensions lda, ldb, ldc (in elements).
//
// Tile configuration (optimized for Intel Arc):
// - 64x64 tiles, 4x4 elements per thread, BK=32
// - 256 threads (16x16 workgroup)
// - Shared memory: 2 * (64*32) * sizeof(float) = 16KB (stores as float after conversion)
// - Fully graph-compatible (USM, no external events)

constexpr int GEMM_F16_F32_BM = 64;
constexpr int GEMM_F16_F32_BN = 64;
constexpr int GEMM_F16_F32_BK = 32;
constexpr int GEMM_F16_F32_TM = 4;
constexpr int GEMM_F16_F32_TN = 4;

constexpr int GEMM_F16_F32_WG_M = GEMM_F16_F32_BM / GEMM_F16_F32_TM;
constexpr int GEMM_F16_F32_WG_N = GEMM_F16_F32_BN / GEMM_F16_F32_TN;

template <int BM, int BN, int BK, int TM, int TN, bool transpose_A = false, bool transpose_B = true>
inline void gemm_f16_f32_tiled_kernel(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const sycl::half * __restrict__ A,
    const sycl::half * __restrict__ B,
    float * __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int block_row = it.get_group(0);
    const int block_col = it.get_group(1);
    const int thread_row = it.get_local_id(0);
    const int thread_col = it.get_local_id(1);
    const int thread_id = thread_row * WG_N + thread_col;
    
    const int row_start = block_row * BM + thread_row * TM;
    const int col_start = block_col * BN + thread_col * TN;
    
    float acc[TM][TN];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            acc[tm][tn] = 0.0f;
        }
    }
    
    constexpr int WG_SIZE = WG_M * WG_N;
    constexpr int A_TILE_SIZE = BM * BK;
    constexpr int B_TILE_SIZE = BN * BK;
    constexpr int A_LOADS_PER_THREAD = (A_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    constexpr int B_LOADS_PER_THREAD = (B_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    
    const int num_k_tiles = (K + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_start = k_tile * BK;
        
        // Load A tile [BM x BK] from global (F16) and convert to float in shared memory
        // A is indexed as A[row][k] with row in [0,M), k in [0,K)
        #pragma unroll
        for (int load = 0; load < A_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < A_TILE_SIZE) {
                const int tile_m = flat_idx / BK;
                const int tile_k = flat_idx % BK;
                const int global_row = block_row * BM + tile_m;
                const int global_k = k_start + tile_k;
                
                float val = 0.0f;
                if (global_row < M && global_k < K) {
                    sycl::half h;
                    if constexpr (transpose_A) {
                        h = A[global_k * lda + global_row];
                    } else {
                        h = A[global_row * lda + global_k];
                    }
                    val = static_cast<float>(h);
                }
                tile_A[tile_m * BK + tile_k] = val;
            }
        }
        
        // Load B tile [BN x BK] from global (F16) and convert to float
        // If transpose_B = true, B is stored as (K, N) row-major: we need B^T, so access B[global_k][global_col] -> B[global_k * ldb + global_col]
        // If transpose_B = false, B is stored as (N, K) row-major: access B[global_col][global_k] -> B[global_col * ldb + global_k]
        #pragma unroll
        for (int load = 0; load < B_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < B_TILE_SIZE) {
                const int tile_n = flat_idx / BK;
                const int tile_k = flat_idx % BK;
                const int global_col = block_col * BN + tile_n;
                const int global_k = k_start + tile_k;
                
                float val = 0.0f;
                if (global_col < N && global_k < K) {
                    sycl::half h;
                    if constexpr (transpose_B) {
                        // B stored as (K, N): row = global_k, col = global_col
                        h = B[global_k * ldb + global_col];
                    } else {
                        // B stored as (N, K): row = global_col, col = global_k
                        h = B[global_col * ldb + global_k];
                    }
                    val = static_cast<float>(h);
                }
                tile_B[tile_n * BK + tile_k] = val;
            }
        }
        
        sycl::group_barrier(it.get_group());
        
        // Compute outer products
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = tile_A[(thread_row * TM + tm) * BK + k];
            }
            
            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = tile_B[(thread_col * TN + tn) * BK + k];
            }
            
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm][tn] = sycl::fma(a_reg[tm], b_reg[tn], acc[tm][tn]);
                }
            }
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    // Write results
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int global_row = row_start + tm;
        if (global_row >= M) continue;
        
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            const int global_col = col_start + tn;
            if (global_col >= N) continue;
            
            const int c_idx = global_row * ldc + global_col;
            if (beta == 0.0f) {
                C[c_idx] = alpha * acc[tm][tn];
            } else {
                C[c_idx] = alpha * acc[tm][tn] + beta * C[c_idx];
            }
        }
    }
}

// Launch function for F16->F32 GEMM (non-batched, batch=1)
inline void launch_gemm_f16_f32_tiled(
    sycl::queue * stream,
    const void * A_src, const void * B_src, void * C_dst,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc,
    bool transpose_B = true
) {
    const sycl::half * A = static_cast<const sycl::half*>(A_src);
    const sycl::half * B = static_cast<const sycl::half*>(B_src);
    float * C = static_cast<float*>(C_dst);
    
    constexpr int BM = GEMM_F16_F32_BM;
    constexpr int BN = GEMM_F16_F32_BN;
    constexpr int BK = GEMM_F16_F32_BK;
    constexpr int TM = GEMM_F16_F32_TM;
    constexpr int TN = GEMM_F16_F32_TN;
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    
    sycl::range<2> global(grid_m * WG_M, grid_n * WG_N);
    sycl::range<2> local(WG_M, WG_N);
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BN * BK), cgh);
        
        if (transpose_B) {
            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
                gemm_f16_f32_tiled_kernel<BM, BN, BK, TM, TN, false, true>(
                    it, tile_A, tile_B,
                    A, B, C,
                    M, N, K,
                    alpha, beta,
                    lda, ldb, ldc
                );
            });
        } else {
            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
                gemm_f16_f32_tiled_kernel<BM, BN, BK, TM, TN, false, false>(
                    it, tile_A, tile_B,
                    A, B, C,
                    M, N, K,
                    alpha, beta,
                    lda, ldb, ldc
                );
            });
        }
    });
}

// Variant for A^T * B (transpose_A = true, transpose_B = false)
inline void launch_gemm_f16_f32_tiled_A_transposed(
    sycl::queue * stream,
    const void * A_src, const void * B_src, void * C_dst,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const sycl::half * A = static_cast<const sycl::half*>(A_src);
    const sycl::half * B = static_cast<const sycl::half*>(B_src);
    float * C = static_cast<float*>(C_dst);
    
    constexpr int BM = GEMM_F16_F32_BM;
    constexpr int BN = GEMM_F16_F32_BN;
    constexpr int BK = GEMM_F16_F32_BK;
    constexpr int TM = GEMM_F16_F32_TM;
    constexpr int TN = GEMM_F16_F32_TN;
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    
    sycl::range<2> global(grid_m * WG_M, grid_n * WG_N);
    sycl::range<2> local(WG_M, WG_N);
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BN * BK), cgh);
        
        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_f16_f32_tiled_kernel<BM, BN, BK, TM, TN, true, false>(
                it, tile_A, tile_B,
                A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

#endif // GGML_SYCL_GEMM_F16_F32_TILED_HPP
