#ifndef GGML_SYCL_GEMM_MXFP4_F32_TILED_HPP
#define GGML_SYCL_GEMM_MXFP4_F32_TILED_HPP

#include "common.hpp"

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

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
inline void gemm_mxfp4_f32_tiled_kernel(sycl::nd_item<2>               it,
                                        sycl::local_accessor<float, 1> tile_A,
                                        sycl::local_accessor<float, 1> tile_B,
                                        const sycl::half * __restrict__ A,   // src1 (activations), N x K
                                        const block_mxfp4 * __restrict__ B,  // src0 (weights, MXFP4), M x K
                                        float * __restrict__ C,              // dst, N x M
                                        const int   N,
                                        const int   M,
                                        const int   K,
                                        const float alpha,
                                        const float beta,
                                        const int   lda,
                                        const int   ldb,
                                        const int   ldc) {
    // BK must match QK_MXFP4 for correct block alignment
    static_assert(BK == QK_MXFP4, "BK must equal QK_MXFP4 for MXFP4 block alignment");

    constexpr int WG_N = BN / TN;           // workgroup tiles in N dimension
    constexpr int WG_M = BM / TM;           // workgroup tiles in M dimension

    const int block_n   = it.get_group(0);  // N dimension
    const int block_m   = it.get_group(1);  // M dimension
    const int thread_n  = it.get_local_id(0);
    const int thread_m  = it.get_local_id(1);
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

    constexpr int WG_SIZE            = WG_N * WG_M;
    constexpr int A_TILE_SIZE        = BN * BK;  // tile_A holds [BN x BK]
    constexpr int B_TILE_SIZE        = BM * BK;  // tile_B holds [BM x BK]
    constexpr int A_LOADS_PER_THREAD = (A_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    constexpr int B_LOADS_PER_THREAD = (B_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;

    // Number of MXFP4 blocks per row of B
    const int num_blocks_per_row = ldb / QK_MXFP4;
    const int num_k_tiles        = (K + BK - 1) / BK;

    // Double-buffering: allocate second set of tiles in local memory
    // This allows overlapping loads for next k-tile with computation for current tile
    constexpr int TOTAL_A_TILES      = 2;
    constexpr int TOTAL_B_TILES      = 2;
    constexpr int DOUBLE_A_TILE_SIZE = A_TILE_SIZE * TOTAL_A_TILES;
    constexpr int DOUBLE_B_TILE_SIZE = B_TILE_SIZE * TOTAL_B_TILES;

    // Current tile indices (ping-pong between 0 and 1)
    int tile_idx = 0;

    // Preload first k-tile
    {
        const int k_start = 0;
        const int k_block = 0;

// Load A tile [BN x BK] from F16 -> float
#pragma unroll
        for (int load = 0; load < A_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < A_TILE_SIZE) {
                const int tile_n   = flat_idx / BK;
                const int tile_k   = flat_idx % BK;
                const int global_n = block_n * BN + tile_n;
                const int global_k = k_start + tile_k;

                float val = 0.0f;
                if (global_n < N && global_k < K) {
                    sycl::half h = A[global_n * lda + global_k];
                    val          = static_cast<float>(h);
                }
                // Store to first tile (offset 0)
                tile_A[tile_n * BK + tile_k] = val;
            }
        }

// Load B tile [BM x BK] from MXFP4 and dequantize to float
#pragma unroll
        for (int load = 0; load < B_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < B_TILE_SIZE) {
                const int tile_m   = flat_idx / BK;
                const int tile_k   = flat_idx % BK;
                const int global_m = block_m * BM + tile_m;
                const int global_k = k_start + tile_k;

                float val = 0.0f;
                if (global_m < M && global_k < K) {
                    const int           block_idx = global_m * num_blocks_per_row + k_block;
                    const block_mxfp4 & blk       = B[block_idx];
                    const float         d         = ggml_sycl_e8m0_to_fp32(blk.e);
                    uint8_t             q4;
                    if (tile_k < 16) {
                        q4 = blk.qs[tile_k] & 0x0F;
                    } else {
                        q4 = blk.qs[tile_k - 16] >> 4;
                    }
                    val = d * static_cast<float>(kvalues_mxfp4[q4]) * 0.5f;
                }
                // Store to first tile (offset 0)
                tile_B[tile_m * BK + tile_k] = val;
            }
        }
    }

    // Synchronize after initial load
    sycl::group_barrier(it.get_group());

    // Main loop with double-buffering
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Current tile offset (ping-pong: 0 or A_TILE_SIZE)
        const int current_offset = tile_idx * A_TILE_SIZE;
        const int next_offset    = (tile_idx ^ 1) * A_TILE_SIZE;

// Compute using current tile
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TN];
#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                a_reg[tn] = tile_A[current_offset + (thread_n * TN + tn) * BK + k];
            }

            float b_reg[TM];
#pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                b_reg[tm] = tile_B[current_offset + (thread_m * TM + tm) * BK + k];
            }

#pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
#pragma unroll
                for (int tm = 0; tm < TM; ++tm) {
                    acc[tn][tm] = sycl::fma(a_reg[tn], b_reg[tm], acc[tn][tm]);
                }
            }
        }

        // Swap buffers for next iteration
        tile_idx ^= 1;

        // Check if we need to load the next k-tile
        const int next_k_tile = k_tile + 1;
        if (next_k_tile < num_k_tiles) {
            const int k_start = next_k_tile * BK;
            const int k_block = next_k_tile;  // Since BK == QK_MXFP4

// Load next k-tile into the buffer we just freed up
#pragma unroll
            for (int load = 0; load < A_LOADS_PER_THREAD; ++load) {
                const int flat_idx = thread_id + load * WG_SIZE;
                if (flat_idx < A_TILE_SIZE) {
                    const int tile_n   = flat_idx / BK;
                    const int tile_k   = flat_idx % BK;
                    const int global_n = block_n * BN + tile_n;
                    const int global_k = k_start + tile_k;

                    float val = 0.0f;
                    if (global_n < N && global_k < K) {
                        sycl::half h = A[global_n * lda + global_k];
                        val          = static_cast<float>(h);
                    }
                    // Store to next tile (offset for ping-pong)
                    tile_A[next_offset + tile_n * BK + tile_k] = val;
                }
            }

#pragma unroll
            for (int load = 0; load < B_LOADS_PER_THREAD; ++load) {
                const int flat_idx = thread_id + load * WG_SIZE;
                if (flat_idx < B_TILE_SIZE) {
                    const int tile_m   = flat_idx / BK;
                    const int tile_k   = flat_idx % BK;
                    const int global_m = block_m * BM + tile_m;
                    const int global_k = k_start + tile_k;

                    float val = 0.0f;
                    if (global_m < M && global_k < K) {
                        const int           block_idx = global_m * num_blocks_per_row + k_block;
                        const block_mxfp4 & blk       = B[block_idx];
                        const float         d         = ggml_sycl_e8m0_to_fp32(blk.e);
                        uint8_t             q4;
                        if (tile_k < 16) {
                            q4 = blk.qs[tile_k] & 0x0F;
                        } else {
                            q4 = blk.qs[tile_k - 16] >> 4;
                        }
                        val = d * static_cast<float>(kvalues_mxfp4[q4]) * 0.5f;
                    }
                    tile_B[next_offset + tile_m * BK + tile_k] = val;
                }
            }

            // Barrier before computing on the newly loaded tile
            sycl::group_barrier(it.get_group());
        }
    }

// Write results: C[n][m]
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
        const int global_n = n_start + tn;
        if (global_n >= N) {
            continue;
        }

#pragma unroll
        for (int tm = 0; tm < TM; ++tm) {
            const int global_m = m_start + tm;
            if (global_m >= M) {
                continue;
            }

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
inline void launch_gemm_mxfp4_f32_tiled(sycl::queue * stream,
                                        const void *  A_src,  // src1_f16 (activations, F16), N x K
                                        const void *  B_src,  // src0 (weights, MXFP4), M x K
                                        void *        C_dst,  // dst (output, F32), N x M
                                        const int     N,
                                        const int     M,
                                        const int     K,
                                        const float   alpha,
                                        const float   beta,
                                        const int     lda,
                                        const int     ldb,
                                        const int     ldc) {
    const sycl::half *  A = static_cast<const sycl::half *>(A_src);
    const block_mxfp4 * B = static_cast<const block_mxfp4 *>(B_src);
    float *             C = static_cast<float *>(C_dst);

    constexpr int BN   = GEMM_MXFP4_F32_BN;
    constexpr int BM   = GEMM_MXFP4_F32_BM;
    constexpr int BK   = GEMM_MXFP4_F32_BK;
    constexpr int TN   = GEMM_MXFP4_F32_TN;
    constexpr int TM   = GEMM_MXFP4_F32_TM;
    constexpr int WG_N = BN / TN;
    constexpr int WG_M = BM / TM;

    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;

    sycl::range<2> global(grid_n * WG_N, grid_m * WG_M);
    sycl::range<2> local(WG_N, WG_M);

    stream->submit([&](sycl::handler & cgh) {
        // Double-buffering: allocate 2x local memory for ping-pong tiles
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(2 * BN * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(2 * BM * BK), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_mxfp4_f32_tiled_kernel<BN, BM, BK, TN, TM>(it, tile_A, tile_B, A, B, C, N, M, K, alpha, beta, lda, ldb,
                                                            ldc);
        });
    });
}

// =============================================================================
// Subgroup-Optimized Indirect GEMM for MUL_MAT_ID (MoE expert dispatch)
// =============================================================================
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
// Each subgroup computes N_DST output columns for one A row, with lanes splitting
// the K/32 quant blocks and using reduce_over_group for the final sum.
//
// Convention for MUL_MAT_ID:
//   A = src1_packed (activations), [total_rows x K] but we process [M x K]
//   B = expert weights (MXFP4), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// MXFP4 dequantization:
//   val = e8m0_scale * kvalues_mxfp4[q4] * 0.5
//   qs[j] low nibble = element j, high nibble = element j+16

constexpr int MXFP4_N_DST   = 4;
constexpr int MXFP4_N_SG    = 2;
constexpr int MXFP4_SG_SIZE = WARP_SIZE;

// MXFP4 has 32 elements per block, each lane handles NB elements.
// For SG_SIZE=16: 2 lanes per block (each handles 16 elements), stride SG_SIZE/2.
constexpr int MXFP4_NB = QK_MXFP4 / 2;  // 16 elements per lane per block

// Kernel: F32 activations x MXFP4 weights
template <int SG_SIZE, int N_DST, int N_SG, int NB>
inline void gemm_mxfp4_f32_sg_kernel_indirect(
    sycl::nd_item<2>                 it,
    const float * __restrict__       A,          // src1_packed (activations, F32), M x K
    const block_mxfp4 * __restrict__ B,          // weights (MXFP4), N x K
    float * __restrict__             C,          // dst_packed (output, F32), M x N
    const int * __restrict__         M_ptr,      // pointer to row count for this expert
    const int * __restrict__         offset_ptr, // pointer to row offset in packed buffers
    const int                        N,
    const int                        K,
    const float                      alpha,
    const float                      beta,
    const int                        lda,
    const int                        ldb,
    const int                        ldc) {

    const int M = *M_ptr;
    if (M <= 0) return;

    const int offset = (offset_ptr) ? *offset_ptr : 0;
    const float * A_ptr = A + offset * lda;
    float *       C_ptr = C + offset * ldc;

    const int m = it.get_group(0);
    if (m >= M) return;

    sycl::sub_group sg      = it.get_sub_group();
    const int       sg_id   = sg.get_group_id()[0];
    const int       lane_id = sg.get_local_id()[0];

    const int first_col = (it.get_group(1) * N_SG + sg_id) * N_DST;
    if (first_col >= N) return;

    const int nb = ldb / QK_MXFP4;

    const float * y = A_ptr + m * lda;

    // Each lane handles NB=16 elements (half a block).
    // lane/2 = block offset, lane%2 = first or second half (0..15 or 16..31).
    constexpr int LANES_PER_BLOCK = SG_SIZE / (QK_MXFP4 / NB);
    const int     ix = lane_id / (QK_MXFP4 / NB);
    const int     il = lane_id % (QK_MXFP4 / NB);
    const int     elem_offset = il * NB;  // 0 or 16

    const float * yb = y + ix * QK_MXFP4 + elem_offset;

    float sumf[N_DST] = { 0.0f };

    for (int ib = ix; ib < nb; ib += LANES_PER_BLOCK) {
        // Load activation slice
        float yl[NB];
        for (int i = 0; i < NB; ++i) {
            yl[i] = yb[i];
        }

        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col < N) {
                const block_mxfp4 & blk = B[global_col * nb + ib];
                const float d = ggml_sycl_e8m0_to_fp32(blk.e);

                float partial = 0.0f;
                for (int i = 0; i < NB; ++i) {
                    const int idx = elem_offset + i;
                    uint8_t q4;
                    if (idx < 16) {
                        q4 = blk.qs[idx] & 0x0F;
                    } else {
                        q4 = blk.qs[idx - 16] >> 4;
                    }
                    partial += yl[i] * (d * static_cast<float>(kvalues_mxfp4[q4]) * 0.5f);
                }
                sumf[col] += partial;
            }
        }

        yb += LANES_PER_BLOCK * QK_MXFP4;
    }

    for (int col = 0; col < N_DST; ++col) {
        const float tot = sycl::reduce_over_group(sg, sumf[col], sycl::plus<float>());

        const int global_col = first_col + col;
        if (lane_id == 0 && global_col < N) {
            if (beta == 0.0f) {
                C_ptr[m * ldc + global_col] = alpha * tot;
            } else {
                C_ptr[m * ldc + global_col] = alpha * tot + beta * C_ptr[m * ldc + global_col];
            }
        }
    }
}

// Kernel: F16 activations x MXFP4 weights
template <int SG_SIZE, int N_DST, int N_SG, int NB>
inline void gemm_mxfp4_f32_sg_kernel_indirect_f16(
    sycl::nd_item<2>                 it,
    const sycl::half * __restrict__  A,          // src1_packed (activations, F16), M x K
    const block_mxfp4 * __restrict__ B,          // weights (MXFP4), N x K
    float * __restrict__             C,          // dst_packed (output, F32), M x N
    const int * __restrict__         M_ptr,
    const int * __restrict__         offset_ptr,
    const int                        N,
    const int                        K,
    const float                      alpha,
    const float                      beta,
    const int                        lda,
    const int                        ldb,
    const int                        ldc) {

    const int M = *M_ptr;
    if (M <= 0) return;

    const int offset = (offset_ptr) ? *offset_ptr : 0;
    const sycl::half * A_ptr = A + offset * lda;
    float *            C_ptr = C + offset * ldc;

    const int m = it.get_group(0);
    if (m >= M) return;

    sycl::sub_group sg      = it.get_sub_group();
    const int       sg_id   = sg.get_group_id()[0];
    const int       lane_id = sg.get_local_id()[0];

    const int first_col = (it.get_group(1) * N_SG + sg_id) * N_DST;
    if (first_col >= N) return;

    const int nb = ldb / QK_MXFP4;

    const sycl::half * y = A_ptr + m * lda;

    constexpr int LANES_PER_BLOCK = SG_SIZE / (QK_MXFP4 / NB);
    const int     ix = lane_id / (QK_MXFP4 / NB);
    const int     il = lane_id % (QK_MXFP4 / NB);
    const int     elem_offset = il * NB;

    const sycl::half * yb = y + ix * QK_MXFP4 + elem_offset;

    float sumf[N_DST] = { 0.0f };

    for (int ib = ix; ib < nb; ib += LANES_PER_BLOCK) {
        float yl[NB];
        for (int i = 0; i < NB; ++i) {
            yl[i] = static_cast<float>(yb[i]);
        }

        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col < N) {
                const block_mxfp4 & blk = B[global_col * nb + ib];
                const float d = ggml_sycl_e8m0_to_fp32(blk.e);

                float partial = 0.0f;
                for (int i = 0; i < NB; ++i) {
                    const int idx = elem_offset + i;
                    uint8_t q4;
                    if (idx < 16) {
                        q4 = blk.qs[idx] & 0x0F;
                    } else {
                        q4 = blk.qs[idx - 16] >> 4;
                    }
                    partial += yl[i] * (d * static_cast<float>(kvalues_mxfp4[q4]) * 0.5f);
                }
                sumf[col] += partial;
            }
        }

        yb += LANES_PER_BLOCK * QK_MXFP4;
    }

    for (int col = 0; col < N_DST; ++col) {
        const float tot = sycl::reduce_over_group(sg, sumf[col], sycl::plus<float>());

        const int global_col = first_col + col;
        if (lane_id == 0 && global_col < N) {
            if (beta == 0.0f) {
                C_ptr[m * ldc + global_col] = alpha * tot;
            } else {
                C_ptr[m * ldc + global_col] = alpha * tot + beta * C_ptr[m * ldc + global_col];
            }
        }
    }
}

// Launch function for indirect MXFP4->F32 GEMM (MUL_MAT_ID) with F32 activations
inline void launch_gemm_tiled_indirect_mxfp4(sycl::queue *       stream,
                                             const float *       A,           // src1_packed (activations, F32)
                                             const block_mxfp4 * B,           // weights (MXFP4)
                                             float *             C,           // dst_packed (output, F32)
                                             const int *         M_ptr,       // pointer to row count for this expert
                                             const int *         offset_ptr,  // pointer to row offset
                                             const int           max_M,       // max possible rows (for grid sizing)
                                             const int           N,
                                             const int           K,           // N = output dim, K = input dim
                                             const float         alpha,
                                             const float         beta,
                                             const int           lda,
                                             const int           ldb,
                                             const int           ldc) {
    constexpr int SG_SIZE = MXFP4_SG_SIZE;
    constexpr int N_DST   = MXFP4_N_DST;
    constexpr int N_SG    = MXFP4_N_SG;
    constexpr int NB      = MXFP4_NB;
    constexpr int WG_SIZE = N_SG * SG_SIZE;

    const int cols_per_wg = N_SG * N_DST;
    const int grid_n      = (N + cols_per_wg - 1) / cols_per_wg;

    sycl::range<2> global(max_M * 1, grid_n * WG_SIZE);
    sycl::range<2> local(1, WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                gemm_mxfp4_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG, NB>(
                    it, A, B, C, M_ptr, offset_ptr, N, K, alpha, beta, lda, ldb, ldc);
            });
    });
}

// Launch function for indirect MXFP4->F32 GEMM (MUL_MAT_ID) with F16 activations
inline void launch_gemm_tiled_indirect_mxfp4_f16(sycl::queue *       stream,
                                                 const sycl::half *  A,           // src1_packed (activations, F16)
                                                 const block_mxfp4 * B,           // weights (MXFP4)
                                                 float *             C,           // dst_packed (output, F32)
                                                 const int *         M_ptr,       // pointer to row count
                                                 const int *         offset_ptr,  // pointer to row offset
                                                 const int           max_M,       // max possible rows
                                                 const int           N,
                                                 const int           K,           // N = output dim, K = input dim
                                                 const float         alpha,
                                                 const float         beta,
                                                 const int           lda,
                                                 const int           ldb,
                                                 const int           ldc) {
    constexpr int SG_SIZE = MXFP4_SG_SIZE;
    constexpr int N_DST   = MXFP4_N_DST;
    constexpr int N_SG    = MXFP4_N_SG;
    constexpr int NB      = MXFP4_NB;
    constexpr int WG_SIZE = N_SG * SG_SIZE;

    const int cols_per_wg = N_SG * N_DST;
    const int grid_n      = (N + cols_per_wg - 1) / cols_per_wg;

    sycl::range<2> global(max_M * 1, grid_n * WG_SIZE);
    sycl::range<2> local(1, WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                gemm_mxfp4_f32_sg_kernel_indirect_f16<SG_SIZE, N_DST, N_SG, NB>(
                    it, A, B, C, M_ptr, offset_ptr, N, K, alpha, beta, lda, ldb, ldc);
            });
    });
}

#endif  // GGML_SYCL_GEMM_MXFP4_F32_TILED_HPP
