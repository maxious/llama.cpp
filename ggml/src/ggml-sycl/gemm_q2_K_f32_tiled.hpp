#ifndef GGML_SYCL_GEMM_Q2_K_F32_TILED_HPP
#define GGML_SYCL_GEMM_Q2_K_F32_TILED_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Subgroup-Optimized Graph-Compatible GEMM for Q2_K quantized weights x F32 activations -> F32 output
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
//
// Each subgroup computes N_DST output columns for one A row.
// All lanes cooperate within each super-block; work is distributed across lanes.
//
// Q2_K block_q2_K: 256 elements per super-block.
// Layout: 16 sub-blocks of 16 elements each, 2-bit quantized.
//   - scales[16]: packed 4-bit scale (low nibble) and min (high nibble) per sub-block
//   - qs[64]: packed 2-bit quants (4 elements per byte)
//   - dm[0] = dall (super-block scale), dm[1] = dmin (super-block min)
//
// Dequantization (interleaved layout, QK_K=256):
//   For byte qs[32*n + l] (n=0..1, l=0..31), scale index is = 8*n + l/16:
//     y[128*n + l +  0] = dall * (scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (scales[is+0] >> 4)
//     y[128*n + l + 32] = dall * (scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (scales[is+2] >> 4)
//     y[128*n + l + 64] = dall * (scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (scales[is+4] >> 4)
//     y[128*n + l + 96] = dall * (scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (scales[is+6] >> 4)

// Configuration constants
constexpr int Q2_K_N_DST   = 4;
constexpr int Q2_K_N_SG    = 2;
constexpr int Q2_K_SG_SIZE = WARP_SIZE;

// =============================================================================
// Indirect GEMM kernel for MUL_MAT_ID (MoE expert dispatch)
// =============================================================================
// For MUL_MAT_ID, we compute C = A * B^T for a subset of rows determined by
// expert routing. M_ptr contains the count of rows for this expert, and
// offset_ptr contains the starting offset in the packed buffers.
//
// Convention for MUL_MAT_ID:
//   A = src1_packed (activations, F32), [total_rows x K] but we process [M x K]
//   B = expert weights (Q2_K), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Work mapping:
//   group(0) = A row index (m)
//   group(1) = N tile index (each tile covers N_SG * N_DST columns)
//   Each subgroup within a workgroup handles N_DST columns
//   Lanes within a subgroup split the elements within each super-block
template <int SG_SIZE, int N_DST, int N_SG>
inline void gemm_q2_K_f32_sg_kernel_indirect(
    sycl::nd_item<2> it,
    const float * __restrict__ A,         // src1_packed (activations, F32), M x K
    const block_q2_K * __restrict__ B,    // weights (Q2_K), N x K
    float * __restrict__ C,               // dst_packed (output, F32), M x N
    const int * __restrict__ M_ptr,       // pointer to row count for this expert
    const int * __restrict__ offset_ptr,  // pointer to row offset in packed buffers
    const int   N,
    const int   K,                        // N = output dim, K = input dim
    const float alpha,
    const float beta,
    const int   lda,
    const int   ldb,
    const int   ldc) {
    const int M = *M_ptr;
    if (M <= 0) {
        return;
    }

    const int     offset = (offset_ptr) ? *offset_ptr : 0;
    const float * A_ptr  = A + offset * lda;
    float *       C_ptr  = C + offset * ldc;

    // Which A row this workgroup processes
    const int m = it.get_group(0);
    if (m >= M) {
        return;
    }

    // Subgroup info
    sycl::sub_group sg      = it.get_sub_group();
    const int       sg_id   = sg.get_group_id()[0];
    const int       lane_id = sg.get_local_id()[0];

    // First output column for this subgroup
    const int first_col = (it.get_group(1) * N_SG + sg_id) * N_DST;
    if (first_col >= N) {
        return;
    }

    // Number of Q2_K super-blocks per row of B
    const int nb = ldb / QK_K;

    // Pointer to this row's activations
    const float * y = A_ptr + m * lda;

    // Distribute 64 (n,l) positions across SG_SIZE lanes
    // 2 halves * 32 positions = 64, each generating 4 elements = 256 per super-block
    constexpr int Q2K_TOTAL_POS    = 64;
    constexpr int Q2K_POS_PER_LANE = Q2K_TOTAL_POS / SG_SIZE;
    const int     pos_start        = lane_id * Q2K_POS_PER_LANE;

    float sumf[N_DST] = { 0.0f };

    for (int ib = 0; ib < nb; ++ib) {
        const float * act = y + ib * QK_K;

        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col < N) {
                const block_q2_K & blk  = B[global_col * nb + ib];
                const float        dall = blk.dm[0];
                const float        dmin = blk.dm[1];

                float partial = 0.0f;

                for (int p = pos_start; p < pos_start + Q2K_POS_PER_LANE; ++p) {
                    const int     n  = p / 32;
                    const int     l  = p % 32;
                    const uint8_t q  = blk.qs[32 * n + l];
                    const int     is = 8 * n + l / 16;

                    const float sc0 = dall * (float) (blk.scales[is + 0] & 0xF);
                    const float mn0 = dmin * (float) (blk.scales[is + 0] >> 4);
                    const float sc1 = dall * (float) (blk.scales[is + 2] & 0xF);
                    const float mn1 = dmin * (float) (blk.scales[is + 2] >> 4);
                    const float sc2 = dall * (float) (blk.scales[is + 4] & 0xF);
                    const float mn2 = dmin * (float) (blk.scales[is + 4] >> 4);
                    const float sc3 = dall * (float) (blk.scales[is + 6] & 0xF);
                    const float mn3 = dmin * (float) (blk.scales[is + 6] >> 4);

                    partial += act[128 * n + l + 0] * (sc0 * (float) ((q >> 0) & 3) - mn0);
                    partial += act[128 * n + l + 32] * (sc1 * (float) ((q >> 2) & 3) - mn1);
                    partial += act[128 * n + l + 64] * (sc2 * (float) ((q >> 4) & 3) - mn2);
                    partial += act[128 * n + l + 96] * (sc3 * (float) ((q >> 6) & 3) - mn3);
                }

                sumf[col] += partial;
            }
        }
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

// Launch function for indirect Q2_K->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is Q2_K, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (Q2_K), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_q2_K(sycl::queue *      stream,
                                            const float *      A,           // src1_packed (activations, F32)
                                            const block_q2_K * B,           // weights (Q2_K)
                                            float *            C,           // dst_packed (output, F32)
                                            const int *        M_ptr,       // pointer to row count for this expert
                                            const int *        offset_ptr,  // pointer to row offset
                                            const int          max_M,       // max possible rows (for grid sizing)
                                            const int          N,
                                            const int          K,           // N = output dim, K = input dim
                                            const float        alpha,
                                            const float        beta,
                                            const int          lda,
                                            const int          ldb,
                                            const int          ldc) {
    constexpr int SG_SIZE = Q2_K_SG_SIZE;
    constexpr int N_DST   = Q2_K_N_DST;
    constexpr int N_SG    = Q2_K_N_SG;

    // Each workgroup has N_SG subgroups, each of SG_SIZE lanes
    constexpr int WG_SIZE = N_SG * SG_SIZE;

    // Each workgroup covers N_SG * N_DST output columns
    const int cols_per_wg = N_SG * N_DST;
    const int grid_n      = (N + cols_per_wg - 1) / cols_per_wg;

    // One workgroup per A row x N tile
    sycl::range<2> global(max_M * 1, grid_n * WG_SIZE);
    sycl::range<2> local(1, WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                             gemm_q2_K_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG>(it, A, B, C, M_ptr, offset_ptr, N,
                                                                                    K, alpha, beta, lda, ldb, ldc);
                         });
    });
}

#endif  // GGML_SYCL_GEMM_Q2_K_F32_TILED_HPP
