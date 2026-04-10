//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "fattn-xmx.hpp"

#include "fattn-common.hpp"

#include <cfloat>
#include <cmath>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <sycl/sycl.hpp>

// ============================================================================
// XMX Flash Attention for prompt processing
//
// Uses joint_matrix FP16 instructions (DPAS/XMX) for the two heavy matmuls:
//   S = Q * K^T  (score computation)
//   O = P * V    (output accumulation)
// Online softmax and rescaling done in FP32.
//
// Tile dimensions:
//   BQ = 64  (query rows per work-group)  [vLLM uses 128-256]
//   BK = 32  (KV positions per tile)     [vLLM uses 32-64]
//   TM = 8, TK = 16, TN = 16  (FP16 joint_matrix shapes for Intel Xe)
//
// Work-group: 4 sub-groups of size 16 = 64 threads
//   All 4 sub-groups cooperate on Q loading and softmax
//   Each sub-group handles multiple rows in strided access for Q*K^T
//
// Memory layout in SLM:
//   Q_slm[BQ][D_padded]    - FP16, loaded once per Q tile
//   K_slm[BK][D_padded]    - FP16, reloaded per KV tile (used as B for Q*K^T)
//   V_slm[BK][D_padded]    - FP16, reloaded per KV tile
//   S_slm[BQ][BK]          - FP16, scores after softmax (P matrix)
// ============================================================================

#ifdef SYCL_USE_XMX

namespace syclmx = sycl::ext::oneapi::experimental::matrix;

// FP16 joint_matrix tile shapes for Intel Xe
static constexpr int FA_TM = 8;   // rows per tile for A operand
static constexpr int FA_TK = 16;  // K dimension per MAD
static constexpr int FA_TN = 16;  // cols per tile for B operand / accumulator

// Work-group tile sizes
static constexpr int FA_BQ     = 64;  // query rows per work-group (4 * FA_TM)
static constexpr int FA_BK     = 32;  // KV positions per tile (1 * FA_TN)
static constexpr int FA_NWARPS = 4;   // sub-groups per work-group

// ============================================================================
// XMX Flash Attention kernel
// Template params:
//   DKQ  - head dimension for Q/K (must be multiple of FA_TK=16)
//   DV   - head dimension for V (must be multiple of FA_TN=16)
// ============================================================================
template <int DKQ, int DV>
static void flash_attn_xmx_kernel(const char * __restrict__ Q,
                                  const char * __restrict__ K,
                                  const char * __restrict__ V,
                                  const char * __restrict__ mask,
                                  float * __restrict__ dst,
                                  const float              scale,
                                  const int32_t            ne00,  // Q head dim
                                  const int32_t            ne01,  // Q seq len
                                  const int32_t            ne02,  // Q n_heads
                                  const int32_t            ne03,  // Q batch
                                  const int32_t            nb01,  // Q seq stride
                                  const int32_t            nb02,  // Q head stride
                                  const int32_t            nb03,  // Q batch stride
                                  const int32_t            ne10,  // K head dim
                                  const int32_t            ne11,  // K seq len (KV cache length)
                                  const int32_t            ne12,  // K n_kv_heads
                                  const int32_t            nb11,  // K seq stride
                                  const int32_t            nb12,  // K head stride
                                  const int64_t            nb13,  // K batch stride
                                  const int32_t            nb21,  // V seq stride
                                  const int32_t            nb22,  // V head stride
                                  const int64_t            nb23,  // V batch stride
                                  const int32_t            ne31,  // mask seq stride
                                  const int32_t            nb31,  // mask bytes seq stride
                                  const int32_t            nb33,  // mask batch stride
                                  const sycl::nd_item<3> & item_ct1) {
    static_assert(DKQ % FA_TK == 0, "DKQ must be a multiple of FA_TK");
    static_assert(DV % FA_TN == 0, "DV must be a multiple of FA_TN");

    const int sg_id   = item_ct1.get_local_id(1);  // sub-group index [0, FA_NWARPS)
    const int lane_id = item_ct1.get_local_id(2);  // lane within sub-group [0, WARP_SIZE)
    const int lid     = sg_id * WARP_SIZE + lane_id;

    // Grid mapping
    const int q_block   = item_ct1.get_group(2);  // which BQ block of queries
    const int head_seq  = item_ct1.get_group(0);  // linearized (batch, head)
    const int sequence  = head_seq / ne02;
    const int head      = head_seq % ne02;
    const int gqa_ratio = ne02 / ne12;
    const int kv_head   = head / gqa_ratio;

    const int q_start = q_block * FA_BQ;
    if (q_start >= ne01) {
        return;
    }

    auto sg = item_ct1.get_sub_group();

    // SLM layout
    constexpr int DKQ_PAD = DKQ + 4;  // pad to avoid bank conflicts
    constexpr int DV_PAD  = DV + 4;

    // Total SLM in floats (Q is FP16, but we'll use FP32 for some buffers)
    // Q_slm: BQ * DKQ (FP16 = half the float count)
    // K_slm: BK * DKQ (FP16)
    // V_slm: BK * DV  (FP16)
    // S_slm: BQ * BK  (FP16, after softmax)
    // score_f32: BQ * BK (FP32, for softmax computation)
    constexpr int Q_slm_size  = FA_BQ * DKQ_PAD;  // in half elements
    constexpr int K_slm_size  = FA_BK * DKQ_PAD;
    constexpr int V_slm_size  = FA_BK * DV_PAD;
    constexpr int S_slm_size  = FA_BQ * FA_BK;  // scores in FP16
    constexpr int SF_slm_size = FA_BQ * FA_BK;  // scores in FP32

    // Use work_group_static for SLM
    constexpr int    PV_scratch_size = 4 * FA_TM * FA_TN;  // for 4 sub-groups
    constexpr size_t total_slm_bytes = (Q_slm_size + K_slm_size + V_slm_size + S_slm_size) * sizeof(sycl::half) +
                                       SF_slm_size * sizeof(float) + PV_scratch_size * sizeof(float);
    sycl::ext::oneapi::experimental::work_group_static<char[total_slm_bytes]> slm_raw;

    sycl::half * Q_slm      = (sycl::half *) &slm_raw[0];
    sycl::half * K_slm      = Q_slm + Q_slm_size;
    sycl::half * V_slm      = K_slm + K_slm_size;
    sycl::half * S_slm      = V_slm + V_slm_size;
    float *      SF_slm     = (float *) (S_slm + S_slm_size);
    float *      PV_scratch = SF_slm + SF_slm_size;

    // Pointers to source data
    const float *      Q_f         = (const float *) (Q + (int64_t) nb03 * sequence + (int64_t) nb02 * head);
    const sycl::half * K_h         = (const sycl::half *) (K + (int64_t) nb13 * sequence + (int64_t) nb12 * kv_head);
    const sycl::half * V_h         = (const sycl::half *) (V + (int64_t) nb23 * sequence + (int64_t) nb22 * kv_head);
    const sycl::half * mask_h      = mask ? (const sycl::half *) (mask + (int64_t) nb33 * sequence) : nullptr;
    const int          mask_stride = nb31 / sizeof(sycl::half);

    const int stride_K = nb11 / sizeof(sycl::half);
    const int stride_V = nb21 / sizeof(sycl::half);

    // --- Load Q into SLM once (convert F32 -> FP16, apply scale) ---
    for (int i = lid; i < FA_BQ * DKQ; i += FA_NWARPS * WARP_SIZE) {
        const int  row   = i / DKQ;
        const int  col   = i % DKQ;
        const int  q_row = q_start + row;
        sycl::half val(0.0f);
        if (q_row < ne01 && col < DKQ) {
            val = sycl::half(Q_f[q_row * (nb01 / sizeof(float)) + col] * scale);
        }
        Q_slm[row * DKQ_PAD + col] = val;
    }
    // Zero the padding columns of Q_slm
    if constexpr (DKQ_PAD > DKQ) {
        for (int i = lid; i < FA_BQ; i += FA_NWARPS * WARP_SIZE) {
            for (int p = DKQ; p < DKQ_PAD; ++p) {
                Q_slm[i * DKQ_PAD + p] = sycl::half(0.0f);
            }
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Output accumulators: each sub-group (0 and 1) handles FA_TM rows
    // For DV, we tile: DV/FA_TN tiles of FA_TM x FA_TN
    constexpr int DV_TILES          = DV / FA_TN;
    constexpr int PV_ELEMS_PER_LANE = (FA_TM * FA_TN + WARP_SIZE - 1) / WARP_SIZE;
    float         out_frag[DV_TILES][PV_ELEMS_PER_LANE];

    for (int t = 0; t < DV_TILES; ++t) {
        for (int e = 0; e < PV_ELEMS_PER_LANE; ++e) {
            out_frag[t][e] = 0.0f;
        }
    }

    // Per-row online softmax state stored in SLM (shared across all SGs)
    // row_meta[row * 2 + 0] = running max
    // row_meta[row * 2 + 1] = running sum
    // Reuse the SF_slm area for this (only needed between tiles)
    float * row_meta = SF_slm;  // [FA_BQ * 2] floats, fits in SF_slm which is [FA_BQ * FA_BK]

    // Initialize row_meta
    for (int i = lid; i < FA_BQ * 2; i += FA_NWARPS * WARP_SIZE) {
        row_meta[i] = (i % 2 == 0) ? (-FLT_MAX / 2.0f) : 0.0f;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // --- Main loop over KV cache ---
    for (int kv_start = 0; kv_start < ne11; kv_start += FA_BK) {
        const int kv_end = sycl::min(kv_start + FA_BK, ne11);
        const int kv_len = kv_end - kv_start;

        // --- Load K tile into SLM (FP16, column-major for K^T) ---
        // K is [ne11, ne10], we load BK rows starting at kv_start
        // Store K in SLM as K[kv_pos][d] for the matmul Q * K^T
        // Q is [BQ][DKQ], K^T is [DKQ][BK], so we need K stored as B matrix
        // joint_matrix B: [TK x TN] row-major
        // For Q*K^T: A = Q[BQ x DKQ], B = K^T[DKQ x BK]
        // B should be stored as K^T[d][kv_pos] = K[kv_pos][d]
        // i.e., K_slm[d * BK_stride + kv_local]
        // But it's easier to store K as K_slm[kv_pos * DKQ_PAD + d]
        // and then load B from K^T view. For joint_matrix, we need B in row-major:
        // B[k][n] = K^T[k][n] = K[n][k]
        // So store K^T in SLM: KT_slm[d][kv_pos] where d is the row dim (TK) and kv_pos is TN

        // Actually, let's load K into SLM normally and handle the transpose via
        // loading B with column_major layout or by transposing in SLM.
        // Simpler: store K^T directly in SLM as [DKQ][BK_padded]
        // This means: for each (d, kv_pos), KT[d * BK_pad + kv_pos] = K[kv_pos * stride_K + d]

        // Load K^T into K_slm: K_slm[d * BK_pad + kv_pos]
        // But our K_slm is sized [BK * DKQ_PAD] which is wrong for K^T.
        // Reinterpret: K_slm as [DKQ][FA_BK]
        // Total elements needed: DKQ * FA_BK <= BK * DKQ_PAD (if FA_BK <= DKQ_PAD, which is true for DKQ >= 64)
        // Actually we need DKQ * FA_BK elements, and K_slm has BK * DKQ_PAD elements.
        // DKQ * FA_BK = DKQ * 32; BK * DKQ_PAD = 32 * (DKQ + 4)
        // DKQ * 32 <= 32 * (DKQ + 4), always true. OK.

        for (int i = lid; i < DKQ * FA_BK; i += FA_NWARPS * WARP_SIZE) {
            const int  d         = i / FA_BK;
            const int  kv_pos    = i % FA_BK;
            const int  global_kv = kv_start + kv_pos;
            sycl::half val(0.0f);
            if (global_kv < ne11 && d < DKQ) {
                val = K_h[global_kv * stride_K + d];
            }
            K_slm[d * FA_BK + kv_pos] = val;  // K^T layout
        }

        // --- Load V tile into SLM ---
        // V is [ne11, DV], load BK rows starting at kv_start
        // For P*V: A = P[BQ x BK], B = V[BK x DV]
        // V stored as V_slm[kv_pos * DV_PAD + d]
        for (int i = lid; i < FA_BK * DV; i += FA_NWARPS * WARP_SIZE) {
            const int  kv_pos    = i / DV;
            const int  d         = i % DV;
            const int  global_kv = kv_start + kv_pos;
            sycl::half val(0.0f);
            if (global_kv < ne11 && d < DV) {
                val = V_h[global_kv * stride_V + d];
            }
            V_slm[kv_pos * DV_PAD + d] = val;
        }
        // Zero V padding
        if constexpr (DV_PAD > DV) {
            for (int i = lid; i < FA_BK; i += FA_NWARPS * WARP_SIZE) {
                for (int p = DV; p < DV_PAD; ++p) {
                    V_slm[i * DV_PAD + p] = sycl::half(0.0f);
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // ===== Step 1: Q * K^T using XMX =====
        // Result: score[BQ][BK] in FP32
        // Q_slm: [BQ][DKQ_PAD] (FP16)
        // K_slm: [DKQ][FA_BK] (FP16, K^T layout)
        // We tile: for each DKQ chunk of FA_TK:
        //   A = Q[sg_m*TM : (sg_m+1)*TM][dk*TK : (dk+1)*TK]   (TM x TK)
        //   B = KT[dk*TK : (dk+1)*TK][sg_n*TN : (sg_n+1)*TN]  (TK x TN)
        //   C += A * B

        // SG layout for score computation:
        // BQ = 32 = 4 * TM -> 4 SG rows
        // BK = 16 = 1 * TN -> 1 SG col
        // All 4 SGs compute scores. Each handles 8 rows.
        // sg 0: rows 0-7, sg 1: rows 8-15, sg 2: rows 16-23, sg 3: rows 24-31

        if (sg_id < 4) {
            const int sg_m_off = sg_id * FA_TM;  // 0, 8, 16, or 24

            syclmx::joint_matrix<sycl::sub_group, float, syclmx::use::accumulator, FA_TM, FA_TN> mat_score;
            syclmx::joint_matrix_fill(sg, mat_score, 0.0f);

            // Accumulate over D dimension
            for (int dk = 0; dk < DKQ / FA_TK; ++dk) {
                syclmx::joint_matrix<sycl::sub_group, sycl::half, syclmx::use::a, FA_TM, FA_TK,
                                     syclmx::layout::row_major>
                    mat_q;
                syclmx::joint_matrix<sycl::sub_group, sycl::half, syclmx::use::b, FA_TK, FA_TN,
                                     syclmx::layout::row_major>
                    mat_kt;

                // Load Q slice: rows [sg_m_off, sg_m_off+TM), cols [dk*TK, (dk+1)*TK)
                syclmx::joint_matrix_load(
                    sg, mat_q,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        Q_slm + sg_m_off * DKQ_PAD + dk * FA_TK),
                    DKQ_PAD);

                // Load K^T slice: rows [dk*TK, (dk+1)*TK), cols [0, BK)
                syclmx::joint_matrix_load(
                    sg, mat_kt,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        K_slm + dk * FA_TK * FA_BK),
                    FA_BK);

                syclmx::joint_matrix_mad(sg, mat_score, mat_q, mat_kt, mat_score);
            }

            // Store scores to SF_slm (FP32) for softmax computation
            syclmx::joint_matrix_store(
                sg, mat_score,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    SF_slm + sg_m_off * FA_BK),
                FA_BK, syclmx::layout::row_major);
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // ===== Step 2: Softmax in FP32 =====
        // Scores are already FP32 in SF_slm. Apply mask, compute per-row exp/sum.
        // Each SG handles FA_BQ/FA_NWARPS rows cooperatively.

        // tile_meta stores [tile_max, tile_sum] per row, placed after SF_slm scores.
        // PV_scratch area is used for tile_meta (separate from SF_slm).
        float * tile_meta = PV_scratch;  // [FA_BQ * 2] floats

        // Apply mask and compute softmax
        {
            constexpr int rows_per_sg = (FA_BQ + FA_NWARPS - 1) / FA_NWARPS;
            for (int r = 0; r < rows_per_sg; ++r) {
                const int qr = sg_id * rows_per_sg + r;
                if (qr >= FA_BQ) {
                    break;
                }
                const int q_row = q_start + qr;

                // Read FP32 scores, apply mask
                float local_max = -FLT_MAX / 2.0f;
                for (int kp = lane_id; kp < FA_BK; kp += WARP_SIZE) {
                    float     s         = SF_slm[qr * FA_BK + kp];
                    const int global_kv = kv_start + kp;
                    if (mask_h && q_row < ne01 && global_kv < ne11) {
                        s += static_cast<float>(mask_h[q_row * mask_stride + global_kv]);
                    } else if (q_row >= ne01 || global_kv >= ne11) {
                        s = -FLT_MAX / 2.0f;
                    }
                    SF_slm[qr * FA_BK + kp] = s;
                    local_max               = sycl::fmax(local_max, s);
                }
                float tile_max_val = warp_reduce_max<WARP_SIZE>(local_max);

                // Compute exp and sum
                float local_sum = 0.0f;
                for (int kp = lane_id; kp < FA_BK; kp += WARP_SIZE) {
                    float s                 = SF_slm[qr * FA_BK + kp];
                    float p                 = sycl::native::exp(sycl::fmax(s - tile_max_val, SOFTMAX_FTZ_THRESHOLD));
                    SF_slm[qr * FA_BK + kp] = p;
                    local_sum += p;
                }
                float tile_sum_val = warp_reduce_sum<WARP_SIZE>(local_sum, item_ct1);

                if (lane_id == 0) {
                    tile_meta[qr * 2 + 0] = tile_max_val;
                    tile_meta[qr * 2 + 1] = tile_sum_val;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // ===== Step 3: Online rescaling of output accumulators =====
        // Update running max/sum per row and rescale existing output fragments
        if (sg_id < 4) {
            const int sg_m_off = sg_id * FA_TM;
            for (int i = lane_id; i < FA_TM * FA_TN; i += WARP_SIZE) {
                const int   m            = i / FA_TN;
                const int   row_local    = sg_m_off + m;
                const float tile_max_val = tile_meta[row_local * 2 + 0];
                const float m_old        = row_meta[row_local * 2 + 0];
                const float m_new        = sycl::fmax(m_old, tile_max_val);
                const float alpha        = sycl::native::exp(m_old - m_new);
                const int   frag_idx     = i / WARP_SIZE;
                for (int t = 0; t < DV_TILES; ++t) {
                    out_frag[t][frag_idx] *= alpha;
                }
            }
        }

        // Update row_meta (cooperative)
        for (int qr = lid; qr < FA_BQ; qr += FA_NWARPS * WARP_SIZE) {
            const float tile_max_val = tile_meta[qr * 2 + 0];
            const float tile_sum_val = tile_meta[qr * 2 + 1];
            const float m_old        = row_meta[qr * 2 + 0];
            const float s_old        = row_meta[qr * 2 + 1];
            const float m_new        = sycl::fmax(m_old, tile_max_val);
            const float alpha        = sycl::native::exp(m_old - m_new);
            const float beta         = sycl::native::exp(tile_max_val - m_new);
            row_meta[qr * 2 + 0]     = m_new;
            row_meta[qr * 2 + 1]     = s_old * alpha + tile_sum_val * beta;
        }

        // Convert P (FP32 in SF_slm) to FP16 in S_slm for the P*V matmul
        for (int i = lid; i < FA_BQ * FA_BK; i += FA_NWARPS * WARP_SIZE) {
            S_slm[i] = sycl::half(SF_slm[i]);
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // ===== Step 4: P * V using XMX =====
        if (sg_id < 4) {
            const int sg_m_off = sg_id * FA_TM;
            float *   pv_local = PV_scratch + sg_id * FA_TM * FA_TN;

            for (int dt = 0; dt < DV_TILES; ++dt) {
                syclmx::joint_matrix<sycl::sub_group, sycl::half, syclmx::use::a, FA_TM, FA_TK,
                                     syclmx::layout::row_major>
                    mat_p;
                syclmx::joint_matrix<sycl::sub_group, sycl::half, syclmx::use::b, FA_TK, FA_TN,
                                     syclmx::layout::row_major>
                                                                                                     mat_v;
                syclmx::joint_matrix<sycl::sub_group, float, syclmx::use::accumulator, FA_TM, FA_TN> mat_pv;

                syclmx::joint_matrix_fill(sg, mat_pv, 0.0f);

                syclmx::joint_matrix_load(
                    sg, mat_p,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        S_slm + sg_m_off * FA_BK),
                    FA_BK);

                syclmx::joint_matrix_load(
                    sg, mat_v,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        V_slm + dt * FA_TN),
                    DV_PAD);

                syclmx::joint_matrix_mad(sg, mat_pv, mat_p, mat_v, mat_pv);

                // Store PV result to SLM scratch
                syclmx::joint_matrix_store(
                    sg, mat_pv,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        pv_local),
                    FA_TN, syclmx::layout::row_major);

                // Extract and accumulate, rescaling by exp(tile_max - running_max)
                for (int i = lane_id; i < FA_TM * FA_TN; i += WARP_SIZE) {
                    const int   m            = i / FA_TN;
                    const int   n            = i % FA_TN;
                    const float val          = pv_local[m * FA_TN + n];
                    const int   row_local    = sg_m_off + m;
                    const float tile_max_val = tile_meta[row_local * 2 + 0];
                    const float running_max  = row_meta[row_local * 2 + 0];
                    const float beta         = sycl::native::exp(tile_max_val - running_max);
                    out_frag[dt][i / WARP_SIZE] += val * beta;
                }
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // ===== Write output =====
    // Normalize by running row sum and write to dst
    // dst layout from launch_fattn: o_idx = ((seq * ne01 + q_row) * ne02 + head) * DV + d

    if (sg_id < 4) {
        const int sg_m_off = sg_id * FA_TM;

        for (int dt = 0; dt < DV_TILES; ++dt) {
            for (int i = lane_id; i < FA_TM * FA_TN; i += WARP_SIZE) {
                const int m     = i / FA_TN;
                const int n     = i % FA_TN;
                const int q_row = q_start + sg_m_off + m;
                const int d     = dt * FA_TN + n;

                if (q_row < ne01 && d < DV) {
                    const float     running_sum = row_meta[(sg_m_off + m) * 2 + 1];
                    const float     inv_sum     = 1.0f / (running_sum > 1e-10f ? running_sum : 1.0f);
                    const ptrdiff_t o_idx       = ((ptrdiff_t) (sequence * ne01 + q_row) * ne02 + head) * DV + d;
                    dst[o_idx]                  = out_frag[dt][i / WARP_SIZE] * inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// Support check
// ============================================================================
bool ggml_sycl_fattn_xmx_supported(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int DKQ = K->ne[0];
    const int DV  = V->ne[0];

    // Requirements for XMX flash attention:
    // 1. Intel GPU with matrix extension support
    // 2. F16 K/V (or F32 which gets converted to F16)
    // 3. F32 Q
    // 4. DKQ == DV and both multiples of FA_TK=16
    // 5. Common head dimensions
    // 6. Prompt processing (multiple queries)
    // 7. No logit softcap (for now)

    if (g_ggml_sycl_disable_xmx) {
        return false;
    }

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false;
    }

    if (Q->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_F32) {
        return false;
    }
    if (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_F32) {
        return false;
    }
    if (DKQ != DV) {
        return false;
    }
    if (DKQ % FA_TK != 0) {
        return false;
    }
    if (DV % FA_TN != 0) {
        return false;
    }

    // Only for prompt processing (many queries)
    if (Q->ne[1] <= 32) {
        return false;
    }

    // Check device supports matrix extensions
    dpct::queue_ptr stream = ctx.stream();
    if (!stream->get_device().has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }

    return true;
}

// ============================================================================
// Launch wrapper
// ============================================================================
void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);

    const int DKQ = K->ne[0];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();

    // Convert K/V to F16 if needed
    ggml_sycl_pool &                 pool = ctx.pool();
    ggml_sycl_pool_alloc<sycl::half> K_f16(pool);
    ggml_sycl_pool_alloc<sycl::half> V_f16(pool);

    const char * K_data = (const char *) K->data;
    int32_t      nb11   = K->nb[1];
    int32_t      nb12   = K->nb[2];
    int64_t      nb13   = K->nb[3];

    const char * V_data = (const char *) V->data;
    int32_t      nb21   = V->nb[1];
    int32_t      nb22   = V->nb[2];
    int64_t      nb23   = V->nb[3];

    if (K->type == GGML_TYPE_F32) {
        K_f16.alloc(ggml_nelements(K));
        to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
        to_fp16(K_data, K_f16.ptr, ggml_nelements(K), stream);
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);
        nb11            = nb11 * bs * sizeof(sycl::half) / ts;
        nb12            = nb12 * bs * sizeof(sycl::half) / ts;
        nb13            = nb13 * bs * sizeof(sycl::half) / ts;
        K_data          = (char *) K_f16.ptr;
    }

    if (V->type == GGML_TYPE_F32) {
        V_f16.alloc(ggml_nelements(V));
        to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
        to_fp16(V_data, V_f16.ptr, ggml_nelements(V), stream);
        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);
        nb21            = nb21 * bs * sizeof(sycl::half) / ts;
        nb22            = nb22 * bs * sizeof(sycl::half) / ts;
        nb23            = nb23 * bs * sizeof(sycl::half) / ts;
        V_data          = (char *) V_f16.ptr;
    }

    const int n_q_blocks = (Q->ne[1] + FA_BQ - 1) / FA_BQ;
    const int n_heads    = Q->ne[2] * Q->ne[3];

    const sycl::range<3> grid(n_heads, 1, n_q_blocks);
    const sycl::range<3> block(1, FA_NWARPS, WARP_SIZE);

    // Capture values needed by the kernel lambda
    const char *  Q_data    = (const char *) Q->data;
    float *       dst_data  = (float *) dst->data;
    const char *  mask_data = mask ? (const char *) mask->data : nullptr;
    const int32_t ne00 = Q->ne[0], ne01_val = Q->ne[1], ne02_val = Q->ne[2], ne03_val = Q->ne[3];
    const int32_t nb01_val = Q->nb[1], nb02_val = Q->nb[2], nb03_val = Q->nb[3];
    const int32_t ne10 = K->ne[0], ne11_val = K->ne[1], ne12_val = K->ne[2];
    const int32_t ne31     = mask ? mask->ne[1] : 0;
    const int32_t nb31_val = mask ? mask->nb[1] : 0;
    const int64_t nb33_val = mask ? mask->nb[3] : 0;

#    define LAUNCH_FATTN_XMX(D)                                                                                        \
        stream->submit([&](sycl::handler & cgh) {                                                                      \
            cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                                   \
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {                 \
                                 flash_attn_xmx_kernel<D, D>(Q_data, K_data, V_data, mask_data, dst_data, scale, ne00, \
                                                             ne01_val, ne02_val, ne03_val, nb01_val, nb02_val,         \
                                                             nb03_val, ne10, ne11_val, ne12_val, nb11, nb12, nb13,     \
                                                             nb21, nb22, nb23, ne31, nb31_val, nb33_val, item_ct1);    \
                             });                                                                                       \
        })

    switch (DKQ) {
        case 64:
            LAUNCH_FATTN_XMX(64);
            break;
        case 80:
            LAUNCH_FATTN_XMX(80);
            break;
        case 96:
            LAUNCH_FATTN_XMX(96);
            break;
        case 112:
            LAUNCH_FATTN_XMX(112);
            break;
        case 128:
            LAUNCH_FATTN_XMX(128);
            break;
        case 256:
            LAUNCH_FATTN_XMX(256);
            break;
        default:
            GGML_ABORT("XMX flash attention: unsupported head dim");
    }

#    undef LAUNCH_FATTN_XMX
}

#else   // !SYCL_USE_XMX

bool ggml_sycl_fattn_xmx_supported(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("XMX flash attention: not compiled with SYCL_USE_XMX");
}

#endif  // SYCL_USE_XMX
