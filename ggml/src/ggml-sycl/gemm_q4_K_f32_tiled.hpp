#ifndef GGML_SYCL_GEMM_Q4_K_F32_TILED_HPP
#define GGML_SYCL_GEMM_Q4_K_F32_TILED_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Subgroup-Optimized Graph-Compatible GEMM for Q4_K quantized weights x F32 activations -> F32 output
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
//
// Q4_K super-block: 256 elements (QK_K=256), 8 sub-blocks of 32 elements each.
// Each super-block has:
//   - ggml_half2 dm (d = super-block scale, dmin = super-block min)
//   - uint8_t scales[12] (packed 6-bit sub-block scales and mins)
//   - uint8_t qs[128] (4-bit quants, two elements per byte)
//
// The 256 elements are organized as 4 pairs of sub-blocks (pair=0..3), each pair covering 64 elements:
//   get_scale_min_k4(2*pair, scales, sc1, m1)   -> scale/min for low nibble half (elements 0..31)
//   get_scale_min_k4(2*pair+1, scales, sc2, m2) -> scale/min for high nibble half (elements 32..63)
//   low nibble:  val = dall * sc1 * (qs[j] & 0xF) - dmin * m1
//   high nibble: val = dall * sc2 * (qs[j] >> 4)  - dmin * m2
//
// Work distribution: Each super-block is split across all SG_SIZE lanes.
// The 4 pairs are assigned to groups of (SG_SIZE/4) lanes, and within each group,
// each lane handles (32 / (SG_SIZE/4)) qs bytes of the 32-byte qs array per pair.
// For SG_SIZE=16: 4 lanes/pair, 8 qs bytes/lane (16 elements: 8 low + 8 high nibble)
// For SG_SIZE=32: 8 lanes/pair, 4 qs bytes/lane (8 elements: 4 low + 4 high nibble)

// Configuration constants
constexpr int Q4_K_N_DST   = 4;
constexpr int Q4_K_N_SG    = 2;
constexpr int Q4_K_SG_SIZE = WARP_SIZE;

// Helper: extract 6-bit scale and min for sub-block j from packed scales array.
// Defined inline here to keep the header self-contained (mirrors get_scale_min_k4 in dequantize.hpp).
inline void get_scale_min_k4_tiled(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
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
//   B = expert weights (Q4_K), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Work mapping:
//   group(0) = A row index (m)
//   group(1) = N tile index (each tile covers N_SG * N_DST columns)
//   Each subgroup within a workgroup handles N_DST columns
//   Lanes within a subgroup split the elements within each super-block
template <int SG_SIZE, int N_DST, int N_SG>
inline void gemm_q4_K_f32_sg_kernel_indirect(sycl::nd_item<2> it,
                                             const float * __restrict__ A,
                                             const block_q4_K * __restrict__ B,
                                             float * __restrict__ C,
                                             const int * __restrict__ M_ptr,
                                             const int * __restrict__ offset_ptr,
                                             const int   N,
                                             const int   K,
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

    const int m = it.get_group(0);
    if (m >= M) {
        return;
    }

    sycl::sub_group sg      = it.get_sub_group();
    const int       sg_id   = sg.get_group_id()[0];
    const int       lane_id = sg.get_local_id()[0];

    const int first_col = (it.get_group(1) * N_SG + sg_id) * N_DST;
    if (first_col >= N) {
        return;
    }

    const int nb = ldb / QK_K;

    const float * y = A_ptr + m * lda;

    // 4 sub-block pairs distributed across SG_SIZE lanes
    constexpr int LANES_PER_PAIR = SG_SIZE / 4;
    constexpr int ELEMS_PER_LANE = 32 / LANES_PER_PAIR;

    const int pair_id = lane_id / LANES_PER_PAIR;
    const int slice   = lane_id % LANES_PER_PAIR;
    const int j_start = slice * ELEMS_PER_LANE;

    float sumf[N_DST] = { 0.0f };

    // All lanes process every super-block, each handling its own slice
    for (int ib = 0; ib < nb; ++ib) {
        const float * act = y + ib * QK_K + 64 * pair_id;

        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col < N) {
                const block_q4_K & blk  = B[global_col * nb + ib];
                const float        dall = static_cast<float>(blk.dm[0]);
                const float        dmin = static_cast<float>(blk.dm[1]);

                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4_tiled(2 * pair_id, blk.scales, sc1, m1);
                get_scale_min_k4_tiled(2 * pair_id + 1, blk.scales, sc2, m2);

                const float d1   = dall * sc1;
                const float min1 = dmin * m1;
                const float d2   = dall * sc2;
                const float min2 = dmin * m2;

                const uint8_t * qs = blk.qs + 32 * pair_id;

                float partial = 0.0f;
                for (int j = j_start; j < j_start + ELEMS_PER_LANE; ++j) {
                    partial += act[j] * (d1 * (float) (qs[j] & 0xF) - min1);
                    partial += act[j + 32] * (d2 * (float) (qs[j] >> 4) - min2);
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

// Launch function for indirect Q4_K->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is Q4_K, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (Q4_K), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_q4_K(sycl::queue *      stream,
                                            const float *      A,           // src1_packed (activations, F32)
                                            const block_q4_K * B,           // weights (Q4_K)
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
    constexpr int SG_SIZE = Q4_K_SG_SIZE;
    constexpr int N_DST   = Q4_K_N_DST;
    constexpr int N_SG    = Q4_K_N_SG;

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
                             gemm_q4_K_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG>(it, A, B, C, M_ptr, offset_ptr, N,
                                                                                    K, alpha, beta, lda, ldb, ldc);
                         });
    });
}

#endif  // GGML_SYCL_GEMM_Q4_K_F32_TILED_HPP
