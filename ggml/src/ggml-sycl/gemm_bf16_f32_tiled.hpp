#ifndef GGML_SYCL_GEMM_BF16_F32_TILED_HPP
#define GGML_SYCL_GEMM_BF16_F32_TILED_HPP

#include "common.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

// Tiled Graph-Compatible GEMM for BF16 inputs -> F32 output
// C = alpha * A * B^T + beta * C   (when transpose_B = true)
// OR C = alpha * A * B + beta * C   (when transpose_B = false)
// A: M x K (BF16), B: either (N, K) or (K, N) (BF16), C: M x N (F32)
// All matrices are row-major with leading dimensions lda, ldb, ldc (in elements).
//
// Tile configuration (optimized for Intel Arc):
// - 64x64 tiles, 4x4 elements per thread, BK=32
// - 256 threads (16x16 workgroup)
// - Shared memory: 2 * (64*32) * sizeof(float) = 16KB (stores as float after conversion)
// - Fully graph-compatible (USM, no external events)

constexpr int GEMM_BF16_F32_BM = 64;
constexpr int GEMM_BF16_F32_BN = 64;
constexpr int GEMM_BF16_F32_BK = 32;
constexpr int GEMM_BF16_F32_TM = 4;
constexpr int GEMM_BF16_F32_TN = 4;

constexpr int GEMM_BF16_F32_WG_M = GEMM_BF16_F32_BM / GEMM_BF16_F32_TM;
constexpr int GEMM_BF16_F32_WG_N = GEMM_BF16_F32_BN / GEMM_BF16_F32_TN;

using sycl_bfloat16 = sycl::ext::oneapi::bfloat16;

template <int BM, int BN, int BK, int TM, int TN, bool transpose_A = false, bool transpose_B = true>
inline void gemm_bf16_f32_tiled_kernel(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const sycl_bfloat16 * __restrict__ A,
    const sycl_bfloat16 * __restrict__ B,
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
        
        // Load A tile [BM x BK] from global (BF16) and convert to float in shared memory
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
                    sycl_bfloat16 h;
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
        
        // Load B tile [BN x BK] from global (BF16) and convert to float
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
                    sycl_bfloat16 h;
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

// Launch function for BF16->F32 GEMM (non-batched, batch=1)
inline void launch_gemm_bf16_f32_tiled(
    sycl::queue * stream,
    const void * A_src, const void * B_src, void * C_dst,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc,
    bool transpose_B = true
) {
    const sycl_bfloat16 * A = static_cast<const sycl_bfloat16*>(A_src);
    const sycl_bfloat16 * B = static_cast<const sycl_bfloat16*>(B_src);
    float * C = static_cast<float*>(C_dst);
    
    constexpr int BM = GEMM_BF16_F32_BM;
    constexpr int BN = GEMM_BF16_F32_BN;
    constexpr int BK = GEMM_BF16_F32_BK;
    constexpr int TM = GEMM_BF16_F32_TM;
    constexpr int TN = GEMM_BF16_F32_TN;
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
                gemm_bf16_f32_tiled_kernel<BM, BN, BK, TM, TN, false, true>(
                    it, tile_A, tile_B,
                    A, B, C,
                    M, N, K,
                    alpha, beta,
                    lda, ldb, ldc
                );
            });
        } else {
            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
                gemm_bf16_f32_tiled_kernel<BM, BN, BK, TM, TN, false, false>(
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
inline void launch_gemm_bf16_f32_tiled_A_transposed(
    sycl::queue * stream,
    const void * A_src, const void * B_src, void * C_dst,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const sycl_bfloat16 * A = static_cast<const sycl_bfloat16*>(A_src);
    const sycl_bfloat16 * B = static_cast<const sycl_bfloat16*>(B_src);
    float * C = static_cast<float*>(C_dst);
    
    constexpr int BM = GEMM_BF16_F32_BM;
    constexpr int BN = GEMM_BF16_F32_BN;
    constexpr int BK = GEMM_BF16_F32_BK;
    constexpr int TM = GEMM_BF16_F32_TM;
    constexpr int TN = GEMM_BF16_F32_TN;
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
            gemm_bf16_f32_tiled_kernel<BM, BN, BK, TM, TN, true, false>(
                it, tile_A, tile_B,
                A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

template <int BM, int BN, int BK, int TM, int TN, bool transpose_A = false, bool transpose_B = true>
inline void gemm_tiled_kernel_indirect_f32_bf16(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const float * __restrict__ A,
    const sycl_bfloat16 * __restrict__ B,
    float * __restrict__ C,
    const int * __restrict__ M_ptr,
    const int * __restrict__ offset_ptr,
    const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int M = *M_ptr;
    if (M <= 0) return;

    const int offset = (offset_ptr) ? *offset_ptr : 0;
    const float * A_ptr = A;
    const sycl_bfloat16 * B_ptr = B; 
    
    float * C_ptr = C;
    
    if (offset > 0) {
        if constexpr (transpose_A) {
             A_ptr += offset;
        } else {
             A_ptr += offset * lda;
        }
        C_ptr += offset * ldc;
    }

    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int block_row = it.get_group(0);
    const int block_col = it.get_group(1);
    
    if (block_row * BM >= M) return;

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
                    if constexpr (transpose_A) {
                        val = A_ptr[global_k * lda + global_row];
                    } else {
                        val = A_ptr[global_row * lda + global_k];
                    }
                }
                tile_A[tile_m * BK + tile_k] = val;
            }
        }
        
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
                    if (global_col < N && global_k < K) {
                        val = static_cast<float>(B_ptr[global_col * ldb + global_k]);
                    }
                } else {
                    if (global_k < K && global_col < N) {
                        val = static_cast<float>(B_ptr[global_k * ldb + global_col]);
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
                C_ptr[c_idx] = alpha * acc[tm][tn];
            } else {
                C_ptr[c_idx] = alpha * acc[tm][tn] + beta * C_ptr[c_idx];
            }
        }
    }
}

template <bool transpose_B = true>
inline void launch_gemm_tiled_indirect_f32_bf16(
    sycl::queue * stream,
    const float * A, const sycl_bfloat16 * B, float * C,
    const int * M_ptr, const int * offset_ptr, const int max_M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    constexpr int BM = GEMM_BF16_F32_BM;
    constexpr int BN = GEMM_BF16_F32_BN;
    constexpr int BK = GEMM_BF16_F32_BK;
    constexpr int TM = GEMM_BF16_F32_TM;
    constexpr int TN = GEMM_BF16_F32_TN;
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int grid_m = (max_M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    
    sycl::range<2> global(grid_m * WG_M, grid_n * WG_N);
    sycl::range<2> local(WG_M, WG_N);
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BN * BK), cgh);
        
        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_tiled_kernel_indirect_f32_bf16<BM, BN, BK, TM, TN, false, transpose_B>(
                it, tile_A, tile_B,
                A, B, C,
                M_ptr, offset_ptr, N, K,
                alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

// =============================================================================
// Direct-IDs GEMM kernel for MUL_MAT_ID (zero-overhead expert dispatch)
// =============================================================================
// Eliminates ALL pack/unpack overhead by reading ids, src1, and writing dst
// directly in-kernel. BF16 weights are unquantized — each element is a
// sycl_bfloat16 value in row-major [N x K] layout.
//
// Grid: (total_rows, grid_n)
// - group(0) = token index (flat index into ids tensor)
// - group(1) = N tile index
//
// Each workgroup:
// 1. Reads ids[iid1 * ids_nb1 + id * ids_nb0] to find expert_id
// 2. Reads src1 row directly via strides (no packing)
// 3. Computes BF16 dot product with expert weights
// 4. Writes to dst directly via strides (no unpacking)

constexpr int BF16_DIRECT_N_DST   = 8;
constexpr int BF16_DIRECT_N_SG    = 4;
constexpr int BF16_DIRECT_SG_SIZE = WARP_SIZE;

template <int SG_SIZE, int N_DST, int N_SG>
inline void gemm_bf16_f32_sg_kernel_direct(
    sycl::nd_item<2>                  it,
    const char * __restrict__         src1_data,     // raw src1 tensor data
    const char * __restrict__         B_base,        // weight base pointer (all experts)
    char * __restrict__               dst_data,      // raw dst tensor data
    const char * __restrict__         ids_data,      // expert ids tensor data
    const int                         n_ids,         // ids->ne[0]
    const size_t                      ids_nb0,       // ids->nb[0]
    const size_t                      ids_nb1,       // ids->nb[1]
    const int64_t                     ne11,          // src1->ne[1]
    const size_t                      src1_nb1,      // src1->nb[1]
    const size_t                      src1_nb2,      // src1->nb[2]
    const size_t                      dst_nb1,       // dst->nb[1]
    const size_t                      dst_nb2,       // dst->nb[2]
    const size_t                      expert_stride, // bytes between experts in B (src0->nb[2])
    const int                         N,             // output dim (src0->ne[1])
    const int                         K,             // input dim (src0->ne[0])
    const int                         N_local,       // local N for this device (= N if no split)
    const int64_t                     row_low) {     // first row this device handles

    // Decompose flat token index to (id, iid1)
    const int token_idx = it.get_group(0);
    const int id   = token_idx % n_ids;
    const int iid1 = token_idx / n_ids;

    // Read expert assignment from ids tensor
    const int32_t expert_id = *(const int32_t *)(ids_data + iid1 * ids_nb1 + id * ids_nb0);

    // Subgroup info
    sycl::sub_group sg      = it.get_sub_group();
    const int       sg_id   = sg.get_group_id()[0];
    const int       lane_id = sg.get_local_id()[0];

    const int first_col = (it.get_group(1) * N_SG + sg_id) * N_DST;
    if (first_col >= N_local) return;

    // Source row: src1[i11 * nb11 + i12 * nb12] where i11 = id % ne11, i12 = iid1
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const float * y = (const float *)(src1_data + i11 * src1_nb1 + i12 * src1_nb2);

    // Weight pointer for this expert
    const sycl_bfloat16 * B_ptr = (const sycl_bfloat16 *)(B_base + expert_id * expert_stride);

    float sumf[N_DST] = { 0.0f };

    // Each lane handles a strided subset of the K dimension
    for (int k = lane_id; k < K; k += SG_SIZE) {
        const float a_val = y[k];

        for (int col = 0; col < N_DST; ++col) {
            const int local_col = first_col + col;
            if (local_col < N_local) {
                const int global_col = (int)row_low + local_col;
                sumf[col] += a_val * static_cast<float>(B_ptr[global_col * K + k]);
            }
        }
    }

    // Destination: dst[id * nb1 + iid1 * nb2 + col * sizeof(float)]
    float * dst_row = (float *)(dst_data + id * dst_nb1 + iid1 * dst_nb2);

    for (int col = 0; col < N_DST; ++col) {
        const float tot = sycl::reduce_over_group(sg, sumf[col], sycl::plus<float>());

        const int local_col = first_col + col;
        if (lane_id == 0 && local_col < N_local) {
            const int global_col = (int)row_low + local_col;
            dst_row[global_col] = tot;
        }
    }
}

// Launch direct-ids BF16 GEMM - zero pack/unpack overhead
// Reads ids tensor in-kernel, accesses src1 and dst directly via strides.
inline void launch_gemm_direct_ids_bf16(sycl::queue *  stream,
                                        const char *   src1_data,     // raw src1 tensor data
                                        const char *   B_base,        // weight base pointer
                                        char *         dst_data,      // raw dst tensor data
                                        const char *   ids_data,      // expert ids tensor data
                                        const int      n_ids,         // ids->ne[0]
                                        const int      n_batches,     // ids->ne[1]
                                        const size_t   ids_nb0,       // ids->nb[0]
                                        const size_t   ids_nb1,       // ids->nb[1]
                                        const int64_t  ne11,          // src1->ne[1]
                                        const size_t   src1_nb1,      // src1->nb[1]
                                        const size_t   src1_nb2,      // src1->nb[2]
                                        const size_t   dst_nb1,       // dst->nb[1]
                                        const size_t   dst_nb2,       // dst->nb[2]
                                        const size_t   expert_stride, // bytes between experts (src0->nb[2])
                                        const int      N,             // output dim
                                        const int      K,             // input dim
                                        const int      N_local,       // local N for this device
                                        const int64_t  row_low) {     // first row for this device
    constexpr int SG_SIZE = BF16_DIRECT_SG_SIZE;
    constexpr int N_DST   = BF16_DIRECT_N_DST;
    constexpr int N_SG    = BF16_DIRECT_N_SG;
    constexpr int WG_SIZE = N_SG * SG_SIZE;

    const int total_rows  = n_ids * n_batches;
    const int cols_per_wg = N_SG * N_DST;
    const int grid_n      = (N_local + cols_per_wg - 1) / cols_per_wg;

    sycl::range<2> global(total_rows, grid_n * WG_SIZE);
    sycl::range<2> local(1, WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                gemm_bf16_f32_sg_kernel_direct<SG_SIZE, N_DST, N_SG>(
                    it, src1_data, B_base, dst_data, ids_data,
                    n_ids, ids_nb0, ids_nb1, ne11, src1_nb1, src1_nb2,
                    dst_nb1, dst_nb2, expert_stride, N, K, N_local, row_low);
            });
    });
}

#endif // GGML_SYCL_GEMM_BF16_F32_TILED_HPP
