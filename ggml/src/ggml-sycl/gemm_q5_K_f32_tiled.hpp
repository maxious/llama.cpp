#ifndef GGML_SYCL_GEMM_Q5_K_F32_TILED_HPP
#define GGML_SYCL_GEMM_Q5_K_F32_TILED_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Subgroup-Optimized Graph-Compatible GEMM for Q5_K quantized weights x F32 activations -> F32 output
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
//
// Q5_K block_q5_K: 256 elements per super-block with 8 sub-blocks of 32 elements.
// Layout: Row-major blocks. B[row] consists of K/256 blocks.
// Each block has:
//   ggml_half2 dm (d, dmin)
//   uint8_t scales[K_SCALE_SIZE=12] (6-bit quantized scales and mins)
//   uint8_t qh[QK_K/8=32] (high bit per element)
//   uint8_t qs[QK_K/2=128] (low 4 bits)
//
// Dequantization: For each group of 64 elements (il=0..3):
//   get_scale_min_k4(2*il, scales) -> sc, m -> d1 = dall*sc, m1 = dmin*m
//   get_scale_min_k4(2*il+1, scales) -> sc, m -> d2 = dall*sc, m2 = dmin*m
//   hm_lo = 1 << (2*il), hm_hi = hm_lo << 1
//   val[offset+0] = d1 * ((ql[0] & 0xF) + (qh[0] & hm_lo ? 16 : 0)) - m1
//   val[offset+32] = d2 * ((ql[0] >> 4) + (qh[0] & hm_hi ? 16 : 0)) - m2
//
// All lanes cooperate within each super-block; work is distributed across lanes.

// Configuration constants
constexpr int Q5_K_N_DST   = 4;
constexpr int Q5_K_N_SG    = 2;
constexpr int Q5_K_SG_SIZE = WARP_SIZE;

// =============================================================================
// Scale/min extraction helper for K-quant formats (Q4_K, Q5_K, Q6_K)
// =============================================================================
// Extracts 6-bit scale and min values from the packed scales[] array.
// j: sub-block index (0..7)
// q: pointer to scales[K_SCALE_SIZE] array
// d: output scale
// m: output min
inline void get_scale_min_k4_sg(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
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
//   B = expert weights (Q5_K), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Work mapping:
//   group(0) = A row index (m)
//   group(1) = N tile index (each tile covers N_SG * N_DST columns)
//   Each subgroup within a workgroup handles N_DST columns
//   Lanes within a subgroup split the elements within each super-block
template <int SG_SIZE, int N_DST, int N_SG>
inline void gemm_q5_K_f32_sg_kernel_indirect(
    sycl::nd_item<2> it,
    const float * __restrict__ A,         // src1_packed (activations, F32), M x K
    const block_q5_K * __restrict__ B,    // weights (Q5_K), N x K
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

    // Number of Q5_K super-blocks per row of B
    const int nb = ldb / QK_K;

    // Pointer to this row's activations
    const float * y = A_ptr + m * lda;

    // Distribute 64 (il,ir) positions across SG_SIZE lanes
    // 4 groups * 16 ir-positions = 64 positions * 4 elements = 256 per super-block
    constexpr int Q5K_TOTAL_POS    = 64;
    constexpr int Q5K_POS_PER_LANE = Q5K_TOTAL_POS / SG_SIZE;
    const int     pos_start        = lane_id * Q5K_POS_PER_LANE;

    float sumf[N_DST] = { 0.0f };

    for (int ib = 0; ib < nb; ++ib) {
        const float * act = y + ib * QK_K;

        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col >= N) {
                continue;
            }

            const block_q5_K & blk  = B[global_col * nb + ib];
            const float        dall = blk.dm[0];
            const float        dmin = blk.dm[1];

            float partial = 0.0f;

            for (int p = pos_start; p < pos_start + Q5K_POS_PER_LANE; ++p) {
                const int il = p / 16;
                const int ir = p % 16;

                uint8_t   sc, m_val;
                const int is = 2 * il;

                get_scale_min_k4_sg(is, blk.scales, sc, m_val);
                const float d1   = dall * sc;
                const float min1 = dmin * m_val;

                get_scale_min_k4_sg(is + 1, blk.scales, sc, m_val);
                const float d2   = dall * sc;
                const float min2 = dmin * m_val;

                const uint8_t hm_lo = 1 << (2 * il);
                const uint8_t hm_hi = hm_lo << 1;

                const uint8_t * ql = blk.qs + 32 * il + 2 * ir;
                const uint8_t * qh = blk.qh + 2 * ir;
                const float *   a  = act + 64 * il + 2 * ir;

                partial += a[0] * (d1 * ((ql[0] & 0xF) + ((qh[0] & hm_lo) ? 16 : 0)) - min1);
                partial += a[1] * (d1 * ((ql[1] & 0xF) + ((qh[1] & hm_lo) ? 16 : 0)) - min1);
                partial += a[32] * (d2 * ((ql[0] >> 4) + ((qh[0] & hm_hi) ? 16 : 0)) - min2);
                partial += a[33] * (d2 * ((ql[1] >> 4) + ((qh[1] & hm_hi) ? 16 : 0)) - min2);
            }

            sumf[col] += partial;
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

// Launch function for indirect Q5_K->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is Q5_K, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (Q5_K), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_q5_K(sycl::queue *      stream,
                                            const float *      A,           // src1_packed (activations, F32)
                                            const block_q5_K * B,           // weights (Q5_K)
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
    constexpr int SG_SIZE = Q5_K_SG_SIZE;
    constexpr int N_DST   = Q5_K_N_DST;
    constexpr int N_SG    = Q5_K_N_SG;

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
                             gemm_q5_K_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG>(it, A, B, C, M_ptr, offset_ptr, N,
                                                                                    K, alpha, beta, lda, ldb, ldc);
                         });
    });
}

#endif  // GGML_SYCL_GEMM_Q5_K_F32_TILED_HPP
