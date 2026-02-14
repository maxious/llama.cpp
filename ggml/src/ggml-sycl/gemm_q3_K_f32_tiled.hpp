#ifndef GGML_SYCL_GEMM_Q3_K_F32_TILED_HPP
#define GGML_SYCL_GEMM_Q3_K_F32_TILED_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Subgroup-Optimized Graph-Compatible GEMM for Q3_K quantized weights x F32 activations -> F32 output
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
//
// Each subgroup computes N_DST output columns for one A row.
// All lanes cooperate within each super-block; work is distributed across lanes.
// sub_group_reduce_add() finalizes per-column results.
//
// Q3_K block_q3_K: 256 elements per super-block.
//   - qs[64]: 2-bit quants (4 per byte)
//   - hmask[32]: high bits (8 per byte)
//   - scales[12]: 16 packed 6-bit scales
//   - d: F16 super-block scale
//
// Dequantization per element at position p (0..255):
//   q2 = (qs[p/4] >> (2*(p%4))) & 3
//   h  = (hmask[p%32] >> (p/32)) & 1
//   scale = unpack_6bit_scale(scales, p/16) - 32
//   val = d * scale * (q2 - (h ? 0 : 4))

// Configuration constants
constexpr int Q3_K_N_DST   = 4;
constexpr int Q3_K_N_SG    = 2;
constexpr int Q3_K_SG_SIZE = WARP_SIZE;

// =============================================================================
// 6-bit scale unpacking helper
// =============================================================================
// The 12-byte scales array encodes 16 6-bit values with complex packing:
//   scales 0..3:  low 4 bits from scales[is], upper 2 bits from scales[is+8] bits 0-1
//   scales 4..7:  low 4 bits from scales[is], upper 2 bits from scales[is+4] bits 2-3
//   scales 8..11: high 4 bits from scales[is-8], upper 2 bits from scales[is] bits 4-5
//   scales 12..15: high 4 bits from scales[is-8], upper 2 bits from scales[is-4] bits 6-7
inline int8_t get_q3_K_scale(const uint8_t * scales, int is) {
    int8_t us;
    if (is < 4) {
        us = (scales[is] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4);
    } else if (is < 8) {
        us = (scales[is] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4);
    } else if (is < 12) {
        us = (scales[is - 8] >> 4) | (((scales[is] >> 4) & 3) << 4);
    } else {
        us = (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
    }
    return us - 32;
}

// =============================================================================
// Indirect GEMM kernel for MUL_MAT_ID (MoE expert dispatch)
// =============================================================================
// For MUL_MAT_ID, we compute C = A * B^T for a subset of rows determined by
// expert routing. M_ptr contains the count of rows for this expert, and
// offset_ptr contains the starting offset in the packed buffers.
//
// Convention for MUL_MAT_ID:
//   A = src1_packed (activations, F32), [total_rows x K] but we process [M x K]
//   B = expert weights (Q3_K), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Work mapping:
//   group(0) = A row index (m)
//   group(1) = N tile index (each tile covers N_SG * N_DST columns)
//   Each subgroup within a workgroup handles N_DST columns
//   Lanes within a subgroup split the elements within each super-block
template <int SG_SIZE, int N_DST, int N_SG>
inline void gemm_q3_K_f32_sg_kernel_indirect(
    sycl::nd_item<2> it,
    const float * __restrict__ A,         // src1_packed (activations, F32), M x K
    const block_q3_K * __restrict__ B,    // weights (Q3_K), N x K
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

    // Number of Q3_K super-blocks per row of B
    const int nb = ldb / QK_K;

    // Pointer to this row's activations
    const float * y = A_ptr + m * lda;

    // Distribute 32 (im,l) positions across SG_SIZE lanes
    // 2 halves * 16 positions = 32, each generating 8 elements = 256 per super-block
    constexpr int Q3K_TOTAL_POS    = 32;
    constexpr int Q3K_POS_PER_LANE = Q3K_TOTAL_POS / SG_SIZE;
    const int     pos_start        = lane_id * Q3K_POS_PER_LANE;

    float sumf[N_DST] = { 0.0f };

    for (int ib = 0; ib < nb; ++ib) {
        const float * yb = y + ib * QK_K;

        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col >= N) {
                continue;
            }

            const block_q3_K & blk = B[global_col * nb + ib];
            const float        d   = static_cast<float>(*(const sycl::half *) &blk.d);

            int8_t scales[16];
#pragma unroll
            for (int s = 0; s < 16; ++s) {
                scales[s] = get_q3_K_scale(blk.scales, s);
            }

            float partial = 0.0f;

            for (int p = pos_start; p < pos_start + Q3K_POS_PER_LANE; ++p) {
                const int im = p / 16;
                const int l  = p % 16;

                const uint8_t hmask_bit = 1 << (4 * im);
                const uint8_t ql        = blk.qs[32 * im + l];
                const uint8_t ql16      = blk.qs[32 * im + l + 16];
                const uint8_t hl        = blk.hmask[l];
                const uint8_t hl16      = blk.hmask[l + 16];

                const int    y_base = 128 * im;
                const int8_t s0     = scales[8 * im + 0];
                const int8_t s1     = scales[8 * im + 1];
                const int8_t s2     = scales[8 * im + 2];
                const int8_t s3     = scales[8 * im + 3];
                const int8_t s4     = scales[8 * im + 4];
                const int8_t s5     = scales[8 * im + 5];
                const int8_t s6     = scales[8 * im + 6];
                const int8_t s7     = scales[8 * im + 7];

                partial +=
                    yb[y_base + l + 0] * (float) s0 * (float) (((ql >> 0) & 3) - (hl & (hmask_bit << 0) ? 0 : 4));
                partial +=
                    yb[y_base + l + 32] * (float) s2 * (float) (((ql >> 2) & 3) - (hl & (hmask_bit << 1) ? 0 : 4));
                partial +=
                    yb[y_base + l + 64] * (float) s4 * (float) (((ql >> 4) & 3) - (hl & (hmask_bit << 2) ? 0 : 4));
                partial +=
                    yb[y_base + l + 96] * (float) s6 * (float) (((ql >> 6) & 3) - (hl & (hmask_bit << 3) ? 0 : 4));
                partial +=
                    yb[y_base + l + 16] * (float) s1 * (float) (((ql16 >> 0) & 3) - (hl16 & (hmask_bit << 0) ? 0 : 4));
                partial +=
                    yb[y_base + l + 48] * (float) s3 * (float) (((ql16 >> 2) & 3) - (hl16 & (hmask_bit << 1) ? 0 : 4));
                partial +=
                    yb[y_base + l + 80] * (float) s5 * (float) (((ql16 >> 4) & 3) - (hl16 & (hmask_bit << 2) ? 0 : 4));
                partial +=
                    yb[y_base + l + 112] * (float) s7 * (float) (((ql16 >> 6) & 3) - (hl16 & (hmask_bit << 3) ? 0 : 4));
            }

            sumf[col] += d * partial;
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

// Launch function for indirect Q3_K->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is Q3_K, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (Q3_K), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_q3_K(sycl::queue *      stream,
                                            const float *      A,           // src1_packed (activations, F32)
                                            const block_q3_K * B,           // weights (Q3_K)
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
    constexpr int SG_SIZE = Q3_K_SG_SIZE;
    constexpr int N_DST   = Q3_K_N_DST;
    constexpr int N_SG    = Q3_K_N_SG;

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
                             gemm_q3_K_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG>(it, A, B, C, M_ptr, offset_ptr, N,
                                                                                    K, alpha, beta, lda, ldb, ldc);
                         });
    });
}

#endif  // GGML_SYCL_GEMM_Q3_K_F32_TILED_HPP
