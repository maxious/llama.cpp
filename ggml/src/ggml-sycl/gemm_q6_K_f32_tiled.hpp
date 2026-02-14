#ifndef GGML_SYCL_GEMM_Q6_K_F32_TILED_HPP
#define GGML_SYCL_GEMM_Q6_K_F32_TILED_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Subgroup-Optimized Graph-Compatible GEMM for Q6_K quantized weights x F32 activations -> F32 output
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
//
// Q6_K block_q6_K: 256 elements per super-block, organized as 16 sub-blocks of 16 elements.
// Layout:
//   uint8_t ql[128]   - lower 4 bits of each 6-bit quant
//   uint8_t qh[64]    - upper 2 bits of each 6-bit quant
//   int8_t  scales[16] - per-sub-block scales (8-bit)
//   ggml_half d        - super-block scale
//
// Dequantization (for ip=0..1, il=0..31):
//   is = 8*ip + il/16
//   ql_ptr = ql + 64*ip + il
//   qh_val = qh[32*ip + il]
//   y[128*ip + il +  0] = d * scales[is]   * (((ql_ptr[ 0] & 0xF) | (((qh_val >> 0) & 3) << 4)) - 32)
//   y[128*ip + il + 32] = d * scales[is+2] * (((ql_ptr[32] & 0xF) | (((qh_val >> 2) & 3) << 4)) - 32)
//   y[128*ip + il + 64] = d * scales[is+4] * (((ql_ptr[ 0] >> 4)  | (((qh_val >> 4) & 3) << 4)) - 32)
//   y[128*ip + il + 96] = d * scales[is+6] * (((ql_ptr[32] >> 4)  | (((qh_val >> 6) & 3) << 4)) - 32)

// Configuration constants
// N_DST: number of weight rows (output columns) each subgroup processes
// N_SG: number of subgroups per workgroup
// SG_SIZE: subgroup size (matches WARP_SIZE from build config)
constexpr int Q6_K_N_DST   = 4;
constexpr int Q6_K_N_SG    = 2;
constexpr int Q6_K_SG_SIZE = WARP_SIZE;

// =============================================================================
// Indirect GEMM kernel for MUL_MAT_ID (MoE expert dispatch)
// =============================================================================
// For MUL_MAT_ID, we compute C = A * B^T for a subset of rows determined by
// expert routing. M_ptr contains the count of rows for this expert, and
// offset_ptr contains the starting offset in the packed buffers.
//
// Convention for MUL_MAT_ID:
//   A = src1_packed (activations, F32), [total_rows x K] but we process [M x K]
//   B = expert weights (Q6_K), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Work mapping:
//   group(0) = A row index (m)
//   group(1) = N tile index (each tile covers N_SG * N_DST columns)
//   Each subgroup within a workgroup handles N_DST columns
//   Lanes within a subgroup split the elements within each super-block
template <int SG_SIZE, int N_DST, int N_SG>
inline void gemm_q6_K_f32_sg_kernel_indirect(
    sycl::nd_item<2> it,
    const float * __restrict__ A,         // src1_packed (activations, F32), M x K
    const block_q6_K * __restrict__ B,    // weights (Q6_K), N x K
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

    // Number of Q6_K super-blocks per row of B
    const int nb = ldb / QK_K;

    // Pointer to this row's activations
    const float * y = A_ptr + m * lda;

    // Distribute 64 (ip,il) positions across SG_SIZE lanes
    // 64 positions * 4 elements = 256 elements per super-block
    constexpr int Q6K_TOTAL_POS    = 64;
    constexpr int Q6K_POS_PER_LANE = Q6K_TOTAL_POS / SG_SIZE;
    const int     pos_start        = lane_id * Q6K_POS_PER_LANE;

    float sumf[N_DST] = { 0.0f };

    for (int ib = 0; ib < nb; ++ib) {
        const float * act = y + ib * QK_K;

        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col < N) {
                const block_q6_K & blk = B[global_col * nb + ib];
                const float        d   = static_cast<float>(*(const sycl::half *) &blk.d);

                float partial = 0.0f;

                for (int p = pos_start; p < pos_start + Q6K_POS_PER_LANE; ++p) {
                    const int ip = p / 32;
                    const int il = p % 32;
                    const int is = 8 * ip + il / 16;

                    const uint8_t ql0 = blk.ql[64 * ip + il];
                    const uint8_t ql1 = blk.ql[64 * ip + il + 32];
                    const uint8_t qh  = blk.qh[32 * ip + il];

                    const int base = 128 * ip + il;

                    partial +=
                        act[base] * (d * blk.scales[is] * ((int8_t) ((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
                    partial += act[base + 32] *
                               (d * blk.scales[is + 2] * ((int8_t) ((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
                    partial += act[base + 64] *
                               (d * blk.scales[is + 4] * ((int8_t) ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32));
                    partial += act[base + 96] *
                               (d * blk.scales[is + 6] * ((int8_t) ((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32));
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

// Launch function for indirect Q6_K->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is Q6_K, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (Q6_K), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_q6_K(sycl::queue *      stream,
                                            const float *      A,           // src1_packed (activations, F32)
                                            const block_q6_K * B,           // weights (Q6_K)
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
    constexpr int SG_SIZE = Q6_K_SG_SIZE;
    constexpr int N_DST   = Q6_K_N_DST;
    constexpr int N_SG    = Q6_K_N_SG;

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
                             gemm_q6_K_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG>(it, A, B, C, M_ptr, offset_ptr, N,
                                                                                    K, alpha, beta, lda, ldb, ldc);
                         });
    });
}

#endif  // GGML_SYCL_GEMM_Q6_K_F32_TILED_HPP
