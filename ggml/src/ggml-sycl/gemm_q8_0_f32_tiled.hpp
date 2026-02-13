#ifndef GGML_SYCL_GEMM_Q8_0_F32_TILED_HPP
#define GGML_SYCL_GEMM_Q8_0_F32_TILED_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Subgroup-Optimized Graph-Compatible GEMM for Q8_0 quantized weights x F32 activations -> F32 output
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
//
// Adapted from the OpenCL mul_mv_q8_0_f32.cl pattern:
//   - Each subgroup computes N_DST output columns for one A row
//   - Lanes split the K/32 quant blocks (each lane handles NB=8 quants at a time)
//   - sub_group_reduce_add() finalizes per-column results
//
// Q8_0 block_q8_0: 32 elements per block with one F16 scale per block.
// Layout: Row-major blocks. B[row] consists of K/32 blocks.
// Dequant: val = d * qs[i]

// Configuration constants
// N_DST: number of weight rows (output columns) each subgroup processes
// N_SG: number of subgroups per workgroup
// SG_SIZE: subgroup size (matches WARP_SIZE from build config)
// NB_Q8: number of quants each lane handles per block (32 / (SG_SIZE/4))
//   For SG_SIZE=16: each lane handles 8 quants (4 lanes cover one 32-element block)
//   For SG_SIZE=32: each lane handles 4 quants (8 lanes cover one block)
constexpr int Q8_0_N_DST   = 4;
constexpr int Q8_0_N_SG    = 2;
constexpr int Q8_0_SG_SIZE = WARP_SIZE;
constexpr int Q8_0_NB      = 8;  // quants per lane per block (for SG_SIZE=16)

// =============================================================================
// Indirect GEMM kernel for MUL_MAT_ID (MoE expert dispatch)
// =============================================================================
// For MUL_MAT_ID, we compute C = A * B^T for a subset of rows determined by
// expert routing. M_ptr contains the count of rows for this expert, and
// offset_ptr contains the starting offset in the packed buffers.
//
// Convention for MUL_MAT_ID:
//   A = src1_packed (activations, F32), [total_rows x K] but we process [M x K]
//   B = expert weights (Q8_0), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Work mapping:
//   group(0) = A row index (m)
//   group(1) = N tile index (each tile covers N_SG * N_DST columns)
//   Each subgroup within a workgroup handles N_DST columns
//   Lanes within a subgroup split the K blocks
template <int SG_SIZE, int N_DST, int N_SG, int NB>
inline void gemm_q8_0_f32_sg_kernel_indirect(
    sycl::nd_item<2>                it,
    const float * __restrict__      A,          // src1_packed (activations, F32), M x K
    const block_q8_0 * __restrict__ B,          // weights (Q8_0), N x K
    float * __restrict__            C,          // dst_packed (output, F32), M x N
    const int * __restrict__        M_ptr,      // pointer to row count for this expert
    const int * __restrict__        offset_ptr, // pointer to row offset in packed buffers
    const int                       N,
    const int                       K,          // N = output dim, K = input dim
    const float                     alpha,
    const float                     beta,
    const int                       lda,
    const int                       ldb,
    const int                       ldc) {

    const int M = *M_ptr;
    if (M <= 0) return;

    const int offset = (offset_ptr) ? *offset_ptr : 0;
    const float * A_ptr = A + offset * lda;
    float *       C_ptr = C + offset * ldc;

    // Which A row this workgroup processes
    const int m = it.get_group(0);
    if (m >= M) return;

    // Subgroup info
    sycl::sub_group sg      = it.get_sub_group();
    const int       sg_id   = sg.get_group_id()[0];
    const int       lane_id = sg.get_local_id()[0];

    // First output column for this subgroup
    const int first_col = (it.get_group(1) * N_SG + sg_id) * N_DST;
    if (first_col >= N) return;

    // Number of Q8_0 blocks per row of B
    const int nb = ldb / QK8_0;

    // Pointer to this row's activations
    const float * y = A_ptr + m * lda;

    // Each lane handles NB=8 quants out of 32 per block.
    // For SG_SIZE=16: 4 lanes per block (ix = lane/4, il = lane%4)
    // Stride: SG_SIZE/4 blocks per iteration
    constexpr int LANES_PER_BLOCK = SG_SIZE / (QK8_0 / NB);  // = SG_SIZE/4
    const int     ix = lane_id / (QK8_0 / NB);  // block offset within iteration
    const int     il = lane_id % (QK8_0 / NB);  // which NB-element slice within block

    const float * yb = y + ix * QK8_0 + il * NB;

    float yl[NB];
    float sumf[N_DST] = { 0.0f };

    // Iterate over blocks
    for (int ib = ix; ib < nb; ib += LANES_PER_BLOCK) {
        // Load activation slice
        for (int i = 0; i < NB; ++i) {
            yl[i] = yb[i];
        }

        // Compute partial dot product for each output column
        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col < N) {
                const block_q8_0 & blk = B[global_col * nb + ib];
                const int8_t *     qs  = blk.qs + il * NB;
                const float        d   = static_cast<float>(*(const sycl::half *) &blk.d);

                float sumq = 0.0f;
                for (int i = 0; i < NB; ++i) {
                    sumq += (float) qs[i] * yl[i];
                }
                sumf[col] += sumq * d;
            }
        }

        yb += LANES_PER_BLOCK * QK8_0;
    }

    // Reduce across subgroup lanes and write results
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

// Launch function for indirect Q8_0->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is Q8_0, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (Q8_0), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_q8_0(sycl::queue *      stream,
                                            const float *      A,           // src1_packed (activations, F32)
                                            const block_q8_0 * B,           // weights (Q8_0)
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
    constexpr int SG_SIZE = Q8_0_SG_SIZE;
    constexpr int N_DST   = Q8_0_N_DST;
    constexpr int N_SG    = Q8_0_N_SG;
    constexpr int NB      = Q8_0_NB;

    // Each workgroup has N_SG subgroups, each of SG_SIZE lanes
    constexpr int WG_SIZE = N_SG * SG_SIZE;

    // Each workgroup covers N_SG * N_DST output columns
    const int cols_per_wg = N_SG * N_DST;
    const int grid_n      = (N + cols_per_wg - 1) / cols_per_wg;

    // One workgroup per A row x N tile
    sycl::range<2> global(max_M * 1, grid_n * WG_SIZE);
    sycl::range<2> local(1, WG_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                gemm_q8_0_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG, NB>(
                    it, A, B, C, M_ptr, offset_ptr, N, K, alpha, beta, lda, ldb, ldc);
            });
    });
}

#endif  // GGML_SYCL_GEMM_Q8_0_F32_TILED_HPP
