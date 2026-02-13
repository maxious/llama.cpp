#ifndef GGML_SYCL_GEMM_Q4_0_F32_TILED_HPP
#define GGML_SYCL_GEMM_Q4_0_F32_TILED_HPP

#include "common.hpp"

#include <sycl/sycl.hpp>

// Subgroup-Optimized Graph-Compatible GEMM for Q4_0 quantized weights x F32 activations -> F32 output
// Uses sub_group_reduce to finalize dot products instead of dequantizing into SLM.
//
// Adapted from the OpenCL mul_mv_q4_0_f32.cl pattern:
//   - Each subgroup computes N_DST output columns for one A row
//   - Lanes split the K/32 quant blocks
//   - Vectorized nibble unpacking with pre-scaled activation values
//   - sub_group_reduce_add() finalizes per-column results
//
// Q4_0 block_q4_0: 32 elements per block with one F16 scale per block.
// Layout: Row-major blocks. B[row] consists of K/32 blocks.
// Each block has: ggml_half d (scale), uint8_t qs[16] (4-bit nibbles).
// Dequantization: val = d * ((q & 0xF) - 8) for low nibble,
//                 val = d * ((q >> 4) - 8) for high nibble.

// Configuration constants
// N_DST: number of weight rows (output columns) each subgroup processes
// N_SG: number of subgroups per workgroup
// SG_SIZE: subgroup size (matches WARP_SIZE from build config)
constexpr int Q4_0_N_DST   = 4;
constexpr int Q4_0_N_SG    = 2;
constexpr int Q4_0_SG_SIZE = WARP_SIZE;

// =============================================================================
// Core Q4_0 dot product using the OpenCL pre-scaling trick
// =============================================================================
// Computes the dot product of half a Q4_0 block (16 floats) with activation values.
// The activations are pre-scaled by (1, 1/16, 1/256, 1/4096) so that we can
// extract nibbles using ushort masks and multiply directly.
// il: offset within the block (0 for first half, 8 for second half)
// sumy: sum of unscaled activation values (for the -8 bias correction)
inline float block_q4_0_dot_y(const block_q4_0 & blk, float sumy, const float * yl, int il) {
    const float d = static_cast<float>(*(const sycl::half *) &blk.d);

    // Read quantized bytes as ushort pairs for vectorized nibble extraction
    const uint16_t * qs = (const uint16_t *) (blk.qs + il / 2);

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int i = 0; i < 8; i += 2) {
        // Masks 0x000F, 0x0F00, 0x00F0, 0xF000 extract the four nibbles from
        // a ushort. The pre-scaled yl values account for the bit positions.
        acc0 += yl[i + 0] * (float) (qs[i / 2] & 0x000F)
              + yl[i + 1] * (float) (qs[i / 2] & 0x0F00);
        acc1 += yl[i + 8] * (float) (qs[i / 2] & 0x00F0)
              + yl[i + 9] * (float) (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.0f + acc0 + acc1);
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
//   B = expert weights (Q4_0), [N x K] where N is output dim
//   C = dst_packed (output, F32), [total_rows x N] but we write [M x N]
//
// Work mapping:
//   group(0) = A row index (m)
//   group(1) = N tile index (each tile covers N_SG * N_DST columns)
//   Each subgroup within a workgroup handles N_DST columns
//   Lanes within a subgroup split the K blocks
template <int SG_SIZE, int N_DST, int N_SG>
inline void gemm_q4_0_f32_sg_kernel_indirect(
    sycl::nd_item<2>                it,
    const float * __restrict__      A,          // src1_packed (activations, F32), M x K
    const block_q4_0 * __restrict__ B,          // weights (Q4_0), N x K
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

    // Number of Q4_0 blocks per row of B
    const int nb = ldb / QK4_0;

    // Pointer to this row's activations
    const float * y = A_ptr + m * lda;

    // Each lane handles half a Q4_0 block (16 elements).
    // lane/2 gives the starting block index, lane%2 selects first/second half.
    const int ix = lane_id / 2;
    const int il = 8 * (lane_id % 2);

    const float * yb = y + ix * QK4_0 + il;

    float yl[16];
    float sumf[N_DST] = { 0.0f };

    // Iterate over blocks, each subgroup covers SG_SIZE/2 blocks per iteration
    for (int ib = ix; ib < nb; ib += SG_SIZE / 2) {
        // Load and pre-scale activation values for this half-block
        float sumy = 0.0f;
        for (int i = 0; i < 8; i += 2) {
            sumy   += yb[i] + yb[i + 1];
            yl[i]   = yb[i];
            yl[i+1] = yb[i + 1] / 256.0f;

            sumy    += yb[i + 16] + yb[i + 17];
            yl[i+8]  = yb[i + 16] / 16.0f;
            yl[i+9]  = yb[i + 17] / 4096.0f;
        }

        // Compute dot product for each of the N_DST output columns
        for (int col = 0; col < N_DST; ++col) {
            const int global_col = first_col + col;
            if (global_col < N) {
                sumf[col] += block_q4_0_dot_y(B[global_col * nb + ib], sumy, yl, il);
            }
        }

        yb += QK4_0 * (SG_SIZE / 2);
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

// Launch function for indirect Q4_0->F32 GEMM (MUL_MAT_ID)
// Computes: C = A * B^T where B is Q4_0, for a subset of rows
// A: [max_M x K] (F32), B: [N x K] (Q4_0), C: [max_M x N] (F32)
// M_ptr: pointer to actual row count, offset_ptr: pointer to row offset
inline void launch_gemm_tiled_indirect_q4_0(sycl::queue *      stream,
                                            const float *      A,           // src1_packed (activations, F32)
                                            const block_q4_0 * B,           // weights (Q4_0)
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
    constexpr int SG_SIZE = Q4_0_SG_SIZE;
    constexpr int N_DST   = Q4_0_N_DST;
    constexpr int N_SG    = Q4_0_N_SG;

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
                gemm_q4_0_f32_sg_kernel_indirect<SG_SIZE, N_DST, N_SG>(
                    it, A, B, C, M_ptr, offset_ptr, N, K, alpha, beta, lda, ldb, ldc);
            });
    });
}

#endif  // GGML_SYCL_GEMM_Q4_0_F32_TILED_HPP
