#ifndef GGML_SYCL_GEMM_TILED_HPP
#define GGML_SYCL_GEMM_TILED_HPP

#include "common.hpp"
#include <sycl/sycl.hpp>

// Tiled Graph-Compatible GEMM: C = alpha * A * B^T + beta * C
// Row-major matrices: A[M,K], B[N,K] (for B^T), C[M,N]
//
// Optimizations:
// - Tiled algorithm with shared memory for data reuse
// - Configurable tile sizes (BM, BN, BK)
// - Each thread computes TM x TN output elements
// - Coalesced memory access patterns
// - No oneMKL calls - fully graph-compatible

// Default tile configuration for Intel Arc (Xe2)
// BM x BN = output tile size per workgroup
// BK = reduction tile size
// TM x TN = elements per thread
// Benchmark results on Arc Pro B60:
// - BK=32 slightly better than BK=16
// - 64x64 tiles with 4x4 elements/thread = 256 threads (16x16 workgroup)
// - Achieves ~3500 GFLOPS (~30% of oneMKL's ~12000 GFLOPS)
// - Shared memory usage: (64*32 + 64*32) * 4 = 16KB per workgroup
constexpr int GEMM_BM = 64;
constexpr int GEMM_BN = 64;
constexpr int GEMM_BK = 32;
constexpr int GEMM_TM = 4;
constexpr int GEMM_TN = 4;

// Workgroup size: (BM/TM) x (BN/TN) threads = 16x16 = 256 threads
constexpr int GEMM_WG_M = GEMM_BM / GEMM_TM;
constexpr int GEMM_WG_N = GEMM_BN / GEMM_TN;
constexpr int GEMM_WG_SIZE = GEMM_WG_M * GEMM_WG_N;

template <int BM, int BN, int BK, int TM, int TN, bool transpose_B = true>
inline void gemm_tiled_kernel(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const float * __restrict__ A,
    const float * __restrict__ B,
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
    
    // Output position for this thread
    const int row_start = block_row * BM + thread_row * TM;
    const int col_start = block_col * BN + thread_col * TN;
    
    // Accumulator registers for TM x TN output elements
    float acc[TM][TN];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            acc[tm][tn] = 0.0f;
        }
    }
    
    // Number of threads in workgroup
    constexpr int WG_SIZE = WG_M * WG_N;
    
    // Load iterations for tiles
    constexpr int A_TILE_SIZE = BM * BK;
    constexpr int B_TILE_SIZE = BN * BK;
    constexpr int A_LOADS_PER_THREAD = (A_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    constexpr int B_LOADS_PER_THREAD = (B_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    
    // Loop over K dimension in tiles
    const int num_k_tiles = (K + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_start = k_tile * BK;
        
        // Cooperative load of A tile [BM x BK] into shared memory
        // A is row-major: A[row][k]
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
                    val = A[global_row * lda + global_k];
                }
                tile_A[tile_m * BK + tile_k] = val;
            }
        }
        
        // Cooperative load of B tile [BN x BK] into shared memory
        // For transpose_B: B is [N,K] row-major, we load B[n][k]
        // For non-transpose: B is [K,N] row-major, we load B[k][n]
        #pragma unroll
        for (int load = 0; load < B_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < B_TILE_SIZE) {
                const int tile_n = flat_idx / BK;
                const int tile_k = flat_idx % BK;
                const int global_col = block_col * BN + tile_n;
                const int global_k = k_start + tile_k;
                
                float val = 0.0f;
                if constexpr (transpose_B) {
                    // B^T: original B is [N,K], access B[global_col][global_k]
                    if (global_col < N && global_k < K) {
                        val = B[global_col * ldb + global_k];
                    }
                } else {
                    // B: original B is [K,N], access B[global_k][global_col]
                    if (global_k < K && global_col < N) {
                        val = B[global_k * ldb + global_col];
                    }
                }
                tile_B[tile_n * BK + tile_k] = val;
            }
        }
        
        // Synchronize to ensure tiles are loaded
        sycl::group_barrier(it.get_group());
        
        // Compute partial results for this k-tile
        // Each thread computes TM x TN elements
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A values for this k into registers
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = tile_A[(thread_row * TM + tm) * BK + k];
            }
            
            // Load B values for this k into registers
            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = tile_B[(thread_col * TN + tn) * BK + k];
            }
            
            // Outer product accumulation
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm][tn] = sycl::fma(a_reg[tm], b_reg[tn], acc[tm][tn]);
                }
            }
        }
        
        // Synchronize before loading next tiles
        sycl::group_barrier(it.get_group());
    }
    
    // Write results to global memory
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

template <bool transpose_B = true>
inline void launch_gemm_tiled(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    constexpr int BM = GEMM_BM;
    constexpr int BN = GEMM_BN;
    constexpr int BK = GEMM_BK;
    constexpr int TM = GEMM_TM;
    constexpr int TN = GEMM_TN;
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    
    sycl::range<2> global(grid_m * WG_M, grid_n * WG_N);
    sycl::range<2> local(WG_M, WG_N);
    
    stream->submit([&](sycl::handler& cgh) {
        // Allocate shared memory for tiles
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BN * BK), cgh);
        
        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_tiled_kernel<BM, BN, BK, TM, TN, transpose_B>(
                it, tile_A, tile_B,
                A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

// Batched version for flash attention
template <bool transpose_B = true>
inline void launch_gemm_tiled_batched(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int batch,
    const int lda, const int ldb, const int ldc,
    const int64_t stride_A, const int64_t stride_B, const int64_t stride_C
) {
    constexpr int BM = GEMM_BM;
    constexpr int BN = GEMM_BN;
    constexpr int BK = GEMM_BK;
    constexpr int TM = GEMM_TM;
    constexpr int TN = GEMM_TN;
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    
    // Use 3D grid: (M tiles, N tiles, batch)
    sycl::range<3> global(batch, grid_m * WG_M, grid_n * WG_N);
    sycl::range<3> local(1, WG_M, WG_N);
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BN * BK), cgh);
        
        cgh.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
            const int b = it.get_group(0);
            
            // Create 2D item for the kernel
            const float * A_b = A + b * stride_A;
            const float * B_b = B + b * stride_B;
            float * C_b = C + b * stride_C;
            
            // Recompute positions for 2D within batch
            constexpr int WG_M_ = BM / TM;
            constexpr int WG_N_ = BN / TN;
            
            const int block_row = it.get_group(1);
            const int block_col = it.get_group(2);
            const int thread_row = it.get_local_id(1);
            const int thread_col = it.get_local_id(2);
            const int thread_id = thread_row * WG_N_ + thread_col;
            
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
            
            constexpr int WG_SIZE = WG_M_ * WG_N_;
            constexpr int A_TILE_SIZE = BM * BK;
            constexpr int B_TILE_SIZE = BN * BK;
            constexpr int A_LOADS = (A_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
            constexpr int B_LOADS = (B_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
            
            const int num_k_tiles = (K + BK - 1) / BK;
            
            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
                const int k_start = k_tile * BK;
                
                #pragma unroll
                for (int load = 0; load < A_LOADS; ++load) {
                    const int flat_idx = thread_id + load * WG_SIZE;
                    if (flat_idx < A_TILE_SIZE) {
                        const int tile_m = flat_idx / BK;
                        const int tile_k = flat_idx % BK;
                        const int global_row = block_row * BM + tile_m;
                        const int global_k = k_start + tile_k;
                        
                        float val = 0.0f;
                        if (global_row < M && global_k < K) {
                            val = A_b[global_row * lda + global_k];
                        }
                        tile_A[tile_m * BK + tile_k] = val;
                    }
                }
                
                #pragma unroll
                for (int load = 0; load < B_LOADS; ++load) {
                    const int flat_idx = thread_id + load * WG_SIZE;
                    if (flat_idx < B_TILE_SIZE) {
                        const int tile_n = flat_idx / BK;
                        const int tile_k = flat_idx % BK;
                        const int global_col = block_col * BN + tile_n;
                        const int global_k = k_start + tile_k;
                        
                        float val = 0.0f;
                        if constexpr (transpose_B) {
                            if (global_col < N && global_k < K) {
                                val = B_b[global_col * ldb + global_k];
                            }
                        } else {
                            if (global_k < K && global_col < N) {
                                val = B_b[global_k * ldb + global_col];
                            }
                        }
                        tile_B[tile_n * BK + tile_k] = val;
                    }
                }
                
                sycl::group_barrier(it.get_group());
                
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
                        C_b[c_idx] = alpha * acc[tm][tn];
                    } else {
                        C_b[c_idx] = alpha * acc[tm][tn] + beta * C_b[c_idx];
                    }
                }
            }
        });
    });
}

#endif // GGML_SYCL_GEMM_TILED_HPP
