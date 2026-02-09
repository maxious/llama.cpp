#ifndef GGML_SYCL_GEMM_MXFP4_F32_TILED_HPP
#define GGML_SYCL_GEMM_MXFP4_F32_TILED_HPP

#include "common.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

// Tiled Graph-Compatible GEMM for MXFP4 weights x F16 activations -> F32 output
// Fuses dequantization into the GEMM kernel to save memory bandwidth.
//
// llama.cpp convention for mul_mat_tiled:
//   A = src1 (activations, F16 after conversion), N x K
//   B = src0 (weights, MXFP4), M x K (transposed)
//   C = dst (output, F32), N x M
//
// Computes: C = A * B^T = (N x K) * (K x M) = N x M
//
// MXFP4 block_mxfp4: 32 elements per block with one E8M0 scale per block.
// Layout: Row-major blocks. B[row] consists of K/32 blocks.
//
// Optimized for Intel Arc (Xe2)

// Block size matching MXFP4 block size for alignment
constexpr int GEMM_MXFP4_F32_BM = 64;
constexpr int GEMM_MXFP4_F32_BN = 64;
// BK MUST equal QK_MXFP4 (32) to ensure proper block-scale alignment
constexpr int GEMM_MXFP4_F32_BK = 32;
constexpr int GEMM_MXFP4_F32_TM = 4;
constexpr int GEMM_MXFP4_F32_TN = 4;

template <int BN, int BM, int BK, int TN, int TM>
inline void gemm_mxfp4_f32_tiled_kernel(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const sycl::half * __restrict__ A,   // src1 (activations), N x K
    const block_mxfp4 * __restrict__ B,  // src0 (weights, MXFP4), M x K
    float * __restrict__ C,               // dst, N x M
    const int N, const int M, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    // BK must match QK_MXFP4 for correct block alignment
    static_assert(BK == QK_MXFP4, "BK must equal QK_MXFP4 for MXFP4 block alignment");
    
    constexpr int WG_N = BN / TN;  // workgroup tiles in N dimension
    constexpr int WG_M = BM / TM;  // workgroup tiles in M dimension
    
    const int block_n = it.get_group(0);  // N dimension
    const int block_m = it.get_group(1);  // M dimension
    const int thread_n = it.get_local_id(0);
    const int thread_m = it.get_local_id(1);
    const int thread_id = thread_n * WG_M + thread_m;
    
    const int n_start = block_n * BN + thread_n * TN;
    const int m_start = block_m * BM + thread_m * TM;
    
    float acc[TN][TM];
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            acc[tn][tm] = 0.0f;
        }
    }
    
    constexpr int WG_SIZE = WG_N * WG_M;
    constexpr int A_TILE_SIZE = BN * BK;  // tile_A holds [BN x BK]
    constexpr int B_TILE_SIZE = BM * BK;  // tile_B holds [BM x BK]
    constexpr int A_LOADS_PER_THREAD = (A_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    constexpr int B_LOADS_PER_THREAD = (B_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    
    // Number of MXFP4 blocks per row of B
    const int num_blocks_per_row = ldb / QK_MXFP4;
    const int num_k_tiles = (K + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_start = k_tile * BK;
        const int k_block = k_tile;  // Since BK == QK_MXFP4
        
        // Load A tile [BN x BK] from F16 -> float
        #pragma unroll
        for (int load = 0; load < A_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < A_TILE_SIZE) {
                const int tile_n = flat_idx / BK;
                const int tile_k = flat_idx % BK;
                const int global_n = block_n * BN + tile_n;
                const int global_k = k_start + tile_k;
                
                float val = 0.0f;
                if (global_n < N && global_k < K) {
                    // A is row-major: A[n][k] at offset n*lda + k
                    sycl::half h = A[global_n * lda + global_k];
                    val = static_cast<float>(h);
                }
                tile_A[tile_n * BK + tile_k] = val;
            }
        }
        
        // Load B tile [BM x BK] from MXFP4 and dequantize to float
        // Since BK == QK_MXFP4, each k-tile corresponds exactly to one MXFP4 block
        #pragma unroll
        for (int load = 0; load < B_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < B_TILE_SIZE) {
                const int tile_m = flat_idx / BK;
                const int tile_k = flat_idx % BK;
                const int global_m = block_m * BM + tile_m;
                const int global_k = k_start + tile_k;
                
                float val = 0.0f;
                if (global_m < M && global_k < K) {
                    // Block index: B[global_m] has (K/32) blocks
                    // For this k_tile, we're loading exactly one block per row
                    const int block_idx = global_m * num_blocks_per_row + k_block;
                    const block_mxfp4& blk = B[block_idx];
                    
                    // Get the E8M0 scale for this block
                    const float d = ggml_sycl_e8m0_to_fp32(blk.e);
                    
                    // Extract the 4-bit quantized value
                    // MXFP4 layout: qs[j] contains elements j (low nibble) and j+16 (high nibble)
                    uint8_t q4;
                    if (tile_k < 16) {
                        q4 = blk.qs[tile_k] & 0x0F;
                    } else {
                        q4 = blk.qs[tile_k - 16] >> 4;
                    }
                    
                    // Dequantize: scale * lookup * 0.5 (because kvalues_mxfp4 is doubled)
                    val = d * static_cast<float>(kvalues_mxfp4[q4]) * 0.5f;
                }
                tile_B[tile_m * BK + tile_k] = val;
            }
        }
        
        sycl::group_barrier(it.get_group());
        
        // Compute: C[n][m] += sum_k A[n][k] * B[m][k]
        // This is the transposed multiply: C = A * B^T
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                a_reg[tn] = tile_A[(thread_n * TN + tn) * BK + k];
            }
            
            float b_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                b_reg[tm] = tile_B[(thread_m * TM + tm) * BK + k];
            }
            
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                #pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    acc[tn][tm] = sycl::fma(a_reg[tn], b_reg[tm], acc[tn][tm]);
                }
            }
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    // Write results: C[n][m]
    #pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
        const int global_n = n_start + tn;
        if (global_n >= N) continue;
        
        #pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            const int global_m = m_start + tm;
            if (global_m >= M) continue;
            
            const int c_idx = global_n * ldc + global_m;
            if (beta == 0.0f) {
                C[c_idx] = alpha * acc[tn][tm];
            } else {
                C[c_idx] = alpha * acc[tn][tm] + beta * C[c_idx];
            }
        }
    }
}

// Launch function for MXFP4->F32 GEMM
// Computes: C = A * B^T where B is MXFP4
// A: N x K (F16), B: M x K (MXFP4), C: N x M (F32)
inline void launch_gemm_mxfp4_f32_tiled(
    sycl::queue * stream,
    const void * A_src,  // src1_f16 (activations, F16), N x K
    const void * B_src,  // src0 (weights, MXFP4), M x K
    void * C_dst,        // dst (output, F32), N x M
    const int N, const int M, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const sycl::half * A = static_cast<const sycl::half*>(A_src);
    const block_mxfp4 * B = static_cast<const block_mxfp4*>(B_src);
    float * C = static_cast<float*>(C_dst);
    
    constexpr int BN = GEMM_MXFP4_F32_BN;
    constexpr int BM = GEMM_MXFP4_F32_BM;
    constexpr int BK = GEMM_MXFP4_F32_BK;
    constexpr int TN = GEMM_MXFP4_F32_TN;
    constexpr int TM = GEMM_MXFP4_F32_TM;
    constexpr int WG_N = BN / TN;
    constexpr int WG_M = BM / TM;
    
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;
    
    sycl::range<2> global(grid_n * WG_N, grid_m * WG_M);
    sycl::range<2> local(WG_N, WG_M);
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BN * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BM * BK), cgh);
        
        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_mxfp4_f32_tiled_kernel<BN, BM, BK, TN, TM>(
                it, tile_A, tile_B, A, B, C, N, M, K, alpha, beta, lda, ldb, ldc
            );
        });
    });
}

// =============================================================================
// Indirect GEMM for MUL_MAT_ID (MoE expert dispatch)
// =============================================================================
// For MUL_MAT_ID, we compute C = A * B^T for a subset of rows determined by
// expert routing. M_ptr contains the count of rows for this expert, and
// offset_ptr contains the starting offset in the packed buffers.
//
// Convention for MUL_MAT_ID:
//   A = src1_packed (activations, F32), [total_rows x K] but we process [M x K]
//   B = expert weights (MXFP4), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Note: In MUL_MAT_ID context, the dimensions are:
//   M = number of rows assigned to this expert (dynamic, read from M_ptr)
//   N = output dimension (weight matrix rows)  
//   K = input dimension (weight matrix cols)

template <int BM, int BN, int BK, int TM, int TN>
inline void gemm_mxfp4_f32_tiled_kernel_indirect(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const float * __restrict__ A,           // src1_packed (activations, F32), M x K
    const block_mxfp4 * __restrict__ B,     // weights (MXFP4), N x K
    float * __restrict__ C,                  // dst_packed (output, F32), M x N
    const int * __restrict__ M_ptr,         // pointer to row count for this expert
    const int * __restrict__ offset_ptr,    // pointer to row offset in packed buffers
    const int N, const int K,               // N = output dim, K = input dim
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    // BK must match QK_MXFP4 for correct block alignment
    static_assert(BK == QK_MXFP4, "BK must equal QK_MXFP4 for MXFP4 block alignment");
    
    const int M = *M_ptr;
    if (M <= 0) return;
    
    const int offset = (offset_ptr) ? *offset_ptr : 0;
    
    // Apply offset to A and C (packed buffers)
    // B (weights) doesn't need offset - it's passed as base + expert_idx * stride
    const float * A_ptr = A + offset * lda;
    float * C_ptr = C + offset * ldc;
    
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int block_row = it.get_group(0);  // M dimension (rows)
    const int block_col = it.get_group(1);  // N dimension (cols/output)
    
    // Early exit if this workgroup is beyond the actual rows
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
    
    // Number of MXFP4 blocks per row of B
    const int num_blocks_per_row = ldb / QK_MXFP4;
    const int num_k_tiles = (K + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_start = k_tile * BK;
        const int k_block = k_tile;  // Since BK == QK_MXFP4
        
        // Load A tile [BM x BK] from F32
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
                    // A is row-major: A[m][k] at offset m*lda + k
                    val = A_ptr[global_row * lda + global_k];
                }
                tile_A[tile_m * BK + tile_k] = val;
            }
        }
        
        // Load B tile [BN x BK] from MXFP4 and dequantize to float
        // Since BK == QK_MXFP4, each k-tile corresponds exactly to one MXFP4 block
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
                    // Block index: B[global_col] has (K/32) blocks
                    // For this k_tile, we're loading exactly one block per row
                    const int block_idx = global_col * num_blocks_per_row + k_block;
                    const block_mxfp4& blk = B[block_idx];
                    
                    // Get the E8M0 scale for this block
                    const float d = ggml_sycl_e8m0_to_fp32(blk.e);
                    
                    // Extract the 4-bit quantized value
                    // MXFP4 layout: qs[j] contains elements j (low nibble) and j+16 (high nibble)
                    uint8_t q4;
                    if (tile_k < 16) {
                        q4 = blk.qs[tile_k] & 0x0F;
                    } else {
                        q4 = blk.qs[tile_k - 16] >> 4;
                    }
                    
                    // Dequantize: scale * lookup * 0.5 (because kvalues_mxfp4 is doubled)
                    val = d * static_cast<float>(kvalues_mxfp4[q4]) * 0.5f;
                }
                tile_B[tile_n * BK + tile_k] = val;
            }
        }
        
        sycl::group_barrier(it.get_group());
        
        // Compute: C[m][n] += sum_k A[m][k] * B[n][k]
        // This is the transposed multiply: C = A * B^T
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
    
    // Write results: C[m][n]
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

template <int BM, int BN, int BK, int TM, int TN>
inline void gemm_mxfp4_f32_tiled_kernel_indirect_f16(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const sycl::half * __restrict__ A,      // src1_packed (activations, F16), M x K
    const block_mxfp4 * __restrict__ B,     // weights (MXFP4), N x K
    float * __restrict__ C,                  // dst_packed (output, F32), M x N
    const int * __restrict__ M_ptr,         // pointer to row count for this expert
    const int * __restrict__ offset_ptr,    // pointer to row offset in packed buffers
    const int N, const int K,               // N = output dim, K = input dim
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    // BK must match QK_MXFP4 for correct block alignment
    static_assert(BK == QK_MXFP4, "BK must equal QK_MXFP4 for MXFP4 block alignment");

    const int M = *M_ptr;
    if (M <= 0) return;

    const int offset = (offset_ptr) ? *offset_ptr : 0;

    // Apply offset to A and C (packed buffers)
    const sycl::half * A_ptr = A + offset * lda;
    float * C_ptr = C + offset * ldc;

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

    const int num_blocks_per_row = ldb / QK_MXFP4;
    const int num_k_tiles = (K + BK - 1) / BK;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_start = k_tile * BK;
        const int k_block = k_tile;

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
                    val = static_cast<float>(A_ptr[global_row * lda + global_k]);
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
                if (global_col < N && global_k < K) {
                    const int block_idx = global_col * num_blocks_per_row + k_block;
                    const block_mxfp4& blk = B[block_idx];

                    const float d = ggml_sycl_e8m0_to_fp32(blk.e);

                    uint8_t q4;
                    if (tile_k < 16) {
                        q4 = blk.qs[tile_k] & 0x0F;
                    } else {
                        q4 = blk.qs[tile_k - 16] >> 4;
                    }

                    val = d * static_cast<float>(kvalues_mxfp4[q4]) * 0.5f;
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

// Launch function for indirect MXFP4->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is MXFP4, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (MXFP4), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_mxfp4(
    sycl::queue * stream,
    const float * A,              // src1_packed (activations, F32)
    const block_mxfp4 * B,        // weights (MXFP4)
    float * C,                     // dst_packed (output, F32)
    const int * M_ptr,            // pointer to row count for this expert
    const int * offset_ptr,       // pointer to row offset
    const int max_M,              // max possible rows (for grid sizing)
    const int N, const int K,     // N = output dim, K = input dim
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    constexpr int BM = GEMM_MXFP4_F32_BM;  // 64
    constexpr int BN = GEMM_MXFP4_F32_BN;  // 64
    constexpr int BK = GEMM_MXFP4_F32_BK;  // 32 = QK_MXFP4
    constexpr int TM = GEMM_MXFP4_F32_TM;  // 4
    constexpr int TN = GEMM_MXFP4_F32_TN;  // 4
    constexpr int WG_M = BM / TM;          // 16
    constexpr int WG_N = BN / TN;          // 16
    
    const int grid_m = (max_M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    
    sycl::range<2> global(grid_m * WG_M, grid_n * WG_N);
    sycl::range<2> local(WG_M, WG_N);
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BN * BK), cgh);
        
        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_mxfp4_f32_tiled_kernel_indirect<BM, BN, BK, TM, TN>(
                it, tile_A, tile_B,
                A, B, C,
                M_ptr, offset_ptr, N, K,
                alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

// Launch function for indirect MXFP4->F32 GEMM (MUL_MAT_ID) with F16 activations
inline void launch_gemm_tiled_indirect_mxfp4_f16(
    sycl::queue * stream,
    const sycl::half * A,          // src1_packed (activations, F16)
    const block_mxfp4 * B,         // weights (MXFP4)
    float * C,                     // dst_packed (output, F32)
    const int * M_ptr,            // pointer to row count
    const int * offset_ptr,       // pointer to row offset
    const int max_M,              // max possible rows
    const int N, const int K,     // N = output dim, K = input dim
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    constexpr int BM = GEMM_MXFP4_F32_BM;
    constexpr int BN = GEMM_MXFP4_F32_BN;
    constexpr int BK = GEMM_MXFP4_F32_BK;
    constexpr int TM = GEMM_MXFP4_F32_TM;
    constexpr int TN = GEMM_MXFP4_F32_TN;
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
            gemm_mxfp4_f32_tiled_kernel_indirect_f16<BM, BN, BK, TM, TN>(
                it, tile_A, tile_B,
                A, B, C,
                M_ptr, offset_ptr, N, K,
                alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

#endif // GGML_SYCL_GEMM_MXFP4_F32_TILED_HPP
