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

#include "mmq-xmx.hpp"
#include "dequantize.hpp"

#ifdef SYCL_USE_XMX

namespace syclmx = sycl::ext::oneapi::experimental::matrix;

// ============================================================================
// XMX tile dimensions for Intel GPUs (sub-group size 16)
// joint_matrix shapes: A[TM x TK], B[TK x TN], C[TM x TN]
// Using int8 -> int32 accumulation
// ============================================================================
static constexpr int XMX_TM = 8;   // rows per joint_matrix tile
static constexpr int XMX_TK = 32;  // K dimension per XMX op (matches Q8_1 block)
static constexpr int XMX_TN = 16;  // cols per joint_matrix tile

// Work-group tile dimensions
// 4 sub-groups in a 2x2 grid: 2 along M (each XMX_TM=8 rows) x 2 along N (each XMX_TN=16 cols)
static constexpr int XMX_WG_M = 16;   // rows per work-group (2 TM tiles)
static constexpr int XMX_WG_N = 32;   // cols per work-group (2 TN tiles)
static constexpr int XMX_NWARPS = 4;  // sub-groups per work-group

// Number of accumulator elements owned by each lane in a joint_matrix
static constexpr int FRAG_ELEMS = XMX_TM * XMX_TN / WARP_SIZE;

// Helper to cast SLM pointer for joint_matrix_store
template <typename T>
static inline auto slm_ptr(T * p) {
    return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
}

// ============================================================================
// Device capability check
// ============================================================================
bool ggml_sycl_supports_xmx_mmq(const sycl::device & dev, ggml_type type) {
    if (g_ggml_sycl_disable_xmx) {
        return false;
    }

    // Check for Intel matrix extension support
    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }

    switch (type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// Q8_0 XMX kernel
// block_q8_0: d (fp16 scale), qs[32] (int8 quants)
// QK8_0 = 32, maps directly to XMX_TK
// ============================================================================
template <bool need_check>
static void mul_mat_q8_0_xmx(
    const void * __restrict__ vx,        // quantized weights [nrows_x, ncols_x]
    const void * __restrict__ vy,        // Q8_1 quantized activations
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const sycl::nd_item<3> & item_ct1,
    int8_t * tile_a,      // SLM [XMX_WG_M][XMX_TK]
    int8_t * tile_b,      // SLM [XMX_TK][XMX_WG_N]
    float  * tile_a_d,    // SLM [XMX_WG_M] - A scales
    float  * tile_b_d,   // SLM [XMX_WG_N] - B scales
    int32_t * c_scratch)  // SLM [XMX_NWARPS * XMX_TM * XMX_TN] - private scratch per SG
{
    const block_q8_0 * x = (const block_q8_0 *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / QK8_0;
    const int blocks_per_col_y = nrows_y / QK8_1;

    const int row_dst_0 = item_ct1.get_group(2) * XMX_WG_M;
    const int col_dst_0 = item_ct1.get_group(1) * XMX_WG_N;

    const int sg_id   = item_ct1.get_local_id(1);  // sub-group index [0, XMX_NWARPS)
    const int lane_id = item_ct1.get_local_id(2);  // lane within sub-group [0, WARP_SIZE)

    // Each sub-group handles a TM x TN_PER_SG output tile
    // With 4 sub-groups and XMX_WG_N=32, each SG handles one XMX_TN=16 column slice
    // and a portion of the M rows
    const int sg_m_offset = (sg_id / 2) * XMX_TM;  // 2 SGs per M-row group
    const int sg_n_offset = (sg_id % 2) * XMX_TN;
    const int c_scratch_base = sg_id * XMX_TM * XMX_TN;  // private scratch per SG

    auto sg = item_ct1.get_sub_group();

    // Per-lane fragment accumulators
    float acc_frag[FRAG_ELEMS] = {0.0f};

    // Iterate over K dimension in blocks of XMX_TK (= QK8_0 = 32)
    for (int kb = 0; kb < blocks_per_row_x; ++kb) {

        // --- Load A tile: Q8_0 quants to int8 SLM ---
        // Each lane loads elements for multiple rows
        for (int m = lane_id; m < XMX_WG_M; m += WARP_SIZE) {
            const int row = row_dst_0 + m;
            if (need_check && row >= nrows_x) {
                // Zero-fill out-of-bounds rows
                for (int k = 0; k < XMX_TK; ++k) {
                    tile_a[m * XMX_TK + k] = 0;
                }
                tile_a_d[m] = 0.0f;
                continue;
            }
            const block_q8_0 * bx = &x[row * blocks_per_row_x + kb];
            // Direct copy — Q8_0 is already int8
            for (int k = 0; k < QK8_0; ++k) {
                tile_a[m * XMX_TK + k] = bx->qs[k];
            }
            tile_a_d[m] = sycl::vec<sycl::half, 1>(bx->d).convert<float, sycl::rounding_mode::automatic>()[0];
        }

        // --- Load B tile: Q8_1 quants to int8 SLM ---
        for (int n = lane_id; n < XMX_WG_N; n += WARP_SIZE) {
            const int col = col_dst_0 + n;
            const int col_eff = need_check ? sycl::min(col, ncols_y - 1) : col;
            const block_q8_1 * by = &y[col_eff * blocks_per_col_y + kb];
            for (int k = 0; k < QK8_1; ++k) {
                tile_b[k * XMX_WG_N + n] = by->qs[k];
            }
            const sycl::float2 ds = by->ds.convert<float, sycl::rounding_mode::automatic>();
            tile_b_d[n] = ds.x();
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // XMX compute + store to SLM scratch for float epilogue
        {
            syclmx::joint_matrix<sycl::sub_group, int8_t,  syclmx::use::a,           XMX_TM, XMX_TK, syclmx::layout::row_major> mat_a;
            syclmx::joint_matrix<sycl::sub_group, int8_t,  syclmx::use::b,           XMX_TK, XMX_TN, syclmx::layout::row_major> mat_b;
            syclmx::joint_matrix<sycl::sub_group, int32_t, syclmx::use::accumulator, XMX_TM, XMX_TN> mat_c;

            syclmx::joint_matrix_fill(sg, mat_c, 0);
            syclmx::joint_matrix_load(sg, mat_a,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    tile_a + sg_m_offset * XMX_TK),
                XMX_TK);
            syclmx::joint_matrix_load(sg, mat_b,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    tile_b + sg_n_offset),
                XMX_WG_N);

            syclmx::joint_matrix_mad(sg, mat_c, mat_a, mat_b, mat_c);

            syclmx::joint_matrix_store(sg, mat_c,
                slm_ptr(c_scratch + c_scratch_base),
                XMX_TN, syclmx::layout::row_major);

            for (int i = lane_id; i < XMX_TM * XMX_TN; i += WARP_SIZE) {
                const int m = i / XMX_TN;
                const int n = i % XMX_TN;
                const int32_t val = c_scratch[c_scratch_base + m * XMX_TN + n];
                const float a_scale = tile_a_d[sg_m_offset + m];
                const float b_scale = tile_b_d[sg_n_offset + n];
                acc_frag[i / WARP_SIZE] += (float)val * a_scale * b_scale;
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // Write accumulated results to global memory
    for (int i = lane_id; i < XMX_TM * XMX_TN; i += WARP_SIZE) {
        const int m = i / XMX_TN;
        const int n = i % XMX_TN;
        const int row = row_dst_0 + sg_m_offset + m;
        const int col = col_dst_0 + sg_n_offset + n;
        if (row < nrows_dst && col < ncols_y) {
            dst[col * nrows_dst + row] = acc_frag[i / WARP_SIZE];
        }
    }
}


// ============================================================================
// Q4_K / Q5_K unified XMX kernel
// Both have: dm (fp16x2), scales[12] (6-bit scale/min via get_scale_min_k4),
// qs[128] (4-bit low nibbles), 8 sub-blocks of 32.
// Q5_K additionally has qh[32] providing the 5th bit.
// ============================================================================
enum class k_quant_type { Q4_K, Q5_K };

template <k_quant_type qtype, bool need_check>
static void mul_mat_qX_K_xmx(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const sycl::nd_item<3> & item_ct1,
    int8_t * tile_a,      // SLM [XMX_WG_M][XMX_TK]
    int8_t * tile_b,      // SLM [XMX_TK][XMX_WG_N]
    float  * tile_scales,  // SLM [XMX_WG_M * 2] - a_scale and a_min interleaved
    float  * tile_b_meta,  // SLM [XMX_WG_N * 2] - b_d and b_sum
    int32_t * c_scratch)   // SLM [XMX_NWARPS * XMX_TM * XMX_TN] - private scratch per SG
{
    // block_q4_K and block_q5_K have the same dm/scales/qs layout
    using block_t = std::conditional_t<qtype == k_quant_type::Q4_K, block_q4_K, block_q5_K>;

    const block_t    * x = (const block_t *)    vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;

    const int row_dst_0 = item_ct1.get_group(2) * XMX_WG_M;
    const int col_dst_0 = item_ct1.get_group(1) * XMX_WG_N;

    const int sg_id   = item_ct1.get_local_id(1);
    const int lane_id = item_ct1.get_local_id(2);

    const int sg_m_offset = (sg_id / 2) * XMX_TM;
    const int sg_n_offset = (sg_id % 2) * XMX_TN;
    const int c_scratch_base = sg_id * XMX_TM * XMX_TN;

    auto sg = item_ct1.get_sub_group();

    float acc_frag[FRAG_ELEMS] = {0.0f};

    for (int kb = 0; kb < blocks_per_row_x; ++kb) {
        for (int sb = 0; sb < 8; ++sb) {
            const int k_offset = kb * (QK_K / QK8_1) + sb;

            // --- Load A tile: unpack quants to int8 ---
            for (int m = lane_id; m < XMX_WG_M; m += WARP_SIZE) {
                const int row = row_dst_0 + m;
                if (need_check && row >= nrows_x) {
                    for (int k = 0; k < XMX_TK; ++k) {
                        tile_a[m * XMX_TK + k] = 0;
                    }
                    tile_scales[m * 2 + 0] = 0.0f;
                    tile_scales[m * 2 + 1] = 0.0f;
                    continue;
                }

                const block_t * bx = &x[row * blocks_per_row_x + kb];

                uint8_t sc_val, m_val;
                get_scale_min_k4(sb, bx->scales, sc_val, m_val);

                const sycl::float2 dm = bx->dm.template convert<float, sycl::rounding_mode::automatic>();
                tile_scales[m * 2 + 0] = dm.x() * sc_val;
                tile_scales[m * 2 + 1] = dm.y() * m_val;

                const int pair = sb / 2;
                const int hi   = sb & 1;

                if constexpr (qtype == k_quant_type::Q4_K) {
                    for (int k = 0; k < 32; ++k) {
                        const uint8_t byte = bx->qs[32 * pair + k];
                        tile_a[m * XMX_TK + k] = hi ? (int8_t)(byte >> 4) : (int8_t)(byte & 0x0F);
                    }
                } else {
                    const uint8_t bit_mask = 1u << sb;
                    for (int k = 0; k < 32; ++k) {
                        const uint8_t byte = bx->qs[32 * pair + k];
                        int8_t q = hi ? (int8_t)(byte >> 4) : (int8_t)(byte & 0x0F);
                        if (bx->qh[k] & bit_mask) {
                            q += 16;
                        }
                        tile_a[m * XMX_TK + k] = q;
                    }
                }
            }

            // --- Load B tile: Q8_1 quants ---
            for (int n = lane_id; n < XMX_WG_N; n += WARP_SIZE) {
                const int col = col_dst_0 + n;
                const int col_eff = need_check ? sycl::min(col, ncols_y - 1) : col;
                const block_q8_1 * by = &y[col_eff * blocks_per_col_y + k_offset];
                for (int k = 0; k < QK8_1; ++k) {
                    tile_b[k * XMX_WG_N + n] = by->qs[k];
                }
                const sycl::float2 ds = by->ds.convert<float, sycl::rounding_mode::automatic>();
                tile_b_meta[n * 2 + 0] = ds.x();
                tile_b_meta[n * 2 + 1] = ds.y();
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);

            // XMX compute + store to SLM scratch for float epilogue
            {
                syclmx::joint_matrix<sycl::sub_group, int8_t,  syclmx::use::a,           XMX_TM, XMX_TK, syclmx::layout::row_major> mat_a;
                syclmx::joint_matrix<sycl::sub_group, int8_t,  syclmx::use::b,           XMX_TK, XMX_TN, syclmx::layout::row_major> mat_b;
                syclmx::joint_matrix<sycl::sub_group, int32_t, syclmx::use::accumulator, XMX_TM, XMX_TN> mat_c;

                syclmx::joint_matrix_fill(sg, mat_c, 0);
                syclmx::joint_matrix_load(sg, mat_a,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        tile_a + sg_m_offset * XMX_TK),
                    XMX_TK);
                syclmx::joint_matrix_load(sg, mat_b,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        tile_b + sg_n_offset),
                    XMX_WG_N);

                syclmx::joint_matrix_mad(sg, mat_c, mat_a, mat_b, mat_c);

                syclmx::joint_matrix_store(sg, mat_c,
                    slm_ptr(c_scratch + c_scratch_base),
                    XMX_TN, syclmx::layout::row_major);

                for (int i = lane_id; i < XMX_TM * XMX_TN; i += WARP_SIZE) {
                    const int m = i / XMX_TN;
                    const int n = i % XMX_TN;
                    const int32_t val = c_scratch[c_scratch_base + m * XMX_TN + n];
                    const float a_scale = tile_scales[(sg_m_offset + m) * 2 + 0];
                    const float a_min   = tile_scales[(sg_m_offset + m) * 2 + 1];
                    const float b_d     = tile_b_meta[(sg_n_offset + n) * 2 + 0];
                    const float b_sum   = tile_b_meta[(sg_n_offset + n) * 2 + 1];
                    acc_frag[i / WARP_SIZE] += (float)val * a_scale * b_d - a_min * b_sum;
                }
            }

            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }

    for (int i = lane_id; i < XMX_TM * XMX_TN; i += WARP_SIZE) {
        const int m = i / XMX_TN;
        const int n = i % XMX_TN;
        const int row = row_dst_0 + sg_m_offset + m;
        const int col = col_dst_0 + sg_n_offset + n;
        if (row < nrows_dst && col < ncols_y) {
            dst[col * nrows_dst + row] = acc_frag[i / WARP_SIZE];
        }
    }
}


// ============================================================================
// Q6_K XMX kernel
// block_q6_K: ql[128] (low 4 bits), qh[64] (upper 2 bits), scales[16] (int8), d (fp16)
// Scales are per 16 elements, so we split each K=32 slice into two K=16 halves
// and use two XMX ops with zero-padding.
// ============================================================================
template <bool need_check>
static void mul_mat_q6_K_xmx(
    const void * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y,
    const int nrows_dst,
    const sycl::nd_item<3> & item_ct1,
    int8_t * tile_a,       // SLM [XMX_WG_M][XMX_TK]
    int8_t * tile_b,       // SLM [XMX_TK][XMX_WG_N]
    float  * tile_a_d,     // SLM [XMX_WG_M] - super-block scale
    int8_t * tile_a_sc,    // SLM [XMX_WG_M] - per-16 scale for current half
    float  * tile_b_d,    // SLM [XMX_WG_N] - B scales
    int32_t * c_scratch)   // SLM [XMX_NWARPS * XMX_TM * XMX_TN] - private scratch per SG
{
    const block_q6_K * x = (const block_q6_K *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / QK_K;
    const int blocks_per_col_y = nrows_y / QK8_1;

    const int row_dst_0 = item_ct1.get_group(2) * XMX_WG_M;
    const int col_dst_0 = item_ct1.get_group(1) * XMX_WG_N;

    const int sg_id   = item_ct1.get_local_id(1);
    const int lane_id = item_ct1.get_local_id(2);

    const int sg_m_offset = (sg_id / 2) * XMX_TM;
    const int sg_n_offset = (sg_id % 2) * XMX_TN;
    const int c_scratch_base = sg_id * XMX_TM * XMX_TN;

    auto sg = item_ct1.get_sub_group();

    float acc_frag[FRAG_ELEMS] = {0.0f};

    // Q6_K has 256 weights per superblock.
    // Layout: ql[128] stores lower 4 bits, qh[64] stores upper 2 bits.
    // scales[16] are per-16 elements (int8_t).
    // We process 32 elements at a time (one Q8_1 block), with two 16-element scale groups.
    // For each 32-element slice, we do two XMX ops: one per 16-element half with zero-padding.
    //
    // Q6_K ql/qh layout (QK_K=256):
    //   For element idx in [0,255]:
    //     ip = idx / 128  (0 or 1: which 128-element half)
    //     il = idx % 128
    //     il32 = il % 32
    //     For il < 64:  ql_byte = ql[64*ip + il32],        use lower nibble  (& 0xF)
    //     For il >= 64: ql_byte = ql[64*ip + (il32)],      use upper nibble  (>> 4)
    //     qh_byte = qh[32*ip + il32]
    //     qh_shift = 2 * (il / 32) for within the 128-element half
    //     q = ((ql_nibble) | (((qh_byte >> qh_shift) & 3) << 4)) - 32
    //     scale_idx = 8*ip + il/16  (note: il here is idx%128)

    for (int kb = 0; kb < blocks_per_row_x; ++kb) {
        for (int sb_pair = 0; sb_pair < 8; ++sb_pair) {
            const int k_offset = kb * (QK_K / QK8_1) + sb_pair;
            // sb_pair maps to element range [sb_pair*32, sb_pair*32+32) within superblock

            // --- Load B tile (same for both halves) ---
            for (int n = lane_id; n < XMX_WG_N; n += WARP_SIZE) {
                const int col = col_dst_0 + n;
                const int col_eff = need_check ? sycl::min(col, ncols_y - 1) : col;
                const block_q8_1 * by = &y[col_eff * blocks_per_col_y + k_offset];
                for (int k = 0; k < QK8_1; ++k) {
                    tile_b[k * XMX_WG_N + n] = by->qs[k];
                }
                const sycl::float2 ds = by->ds.convert<float, sycl::rounding_mode::automatic>();
                tile_b_d[n] = ds.x();
            }

            // Process two 16-element halves with different scales
            for (int half = 0; half < 2; ++half) {
                // Global element index within the 256-element superblock
                const int elem_base = sb_pair * 32 + half * 16;
                // ip = elem_base / 128 (which 128-element half)
                const int ip = elem_base / 128;
                // il = elem_base % 128
                const int il = elem_base % 128;
                // scale_idx = 8*ip + il/16
                const int sc_idx = 8 * ip + il / 16;
                // ql uses 64 bytes per 128-element half: il%64 maps [0..63]->[0..63], [64..127]->[0..63]
                const int ql_base = il % 64;
                // Whether to use upper or lower nibble of ql
                const bool use_upper_nibble = (il >= 64);
                // qh uses 32 bytes per 128-element half, with 2-bit shifts per 32-element group
                const int qh_base = il % 32;
                const int qh_shift_base = 2 * (il / 32);

                // --- Load A tile ---
                for (int m = lane_id; m < XMX_WG_M; m += WARP_SIZE) {
                    const int row = row_dst_0 + m;
                    if (need_check && row >= nrows_x) {
                        for (int k = 0; k < XMX_TK; ++k) {
                            tile_a[m * XMX_TK + k] = 0;
                        }
                        tile_a_d[m] = 0.0f;
                        tile_a_sc[m] = 0;
                        continue;
                    }

                    const block_q6_K * bx = &x[row * blocks_per_row_x + kb];

                    tile_a_d[m] = sycl::vec<sycl::half, 1>(bx->d).convert<float, sycl::rounding_mode::automatic>()[0];
                    tile_a_sc[m] = bx->scales[sc_idx];

                    // Zero the entire tile row first
                    for (int k = 0; k < XMX_TK; ++k) {
                        tile_a[m * XMX_TK + k] = 0;
                    }

                    // Unpack 6-bit quants for this 16-element half
                    for (int k = 0; k < 16; ++k) {
                        const int ql_idx = 64 * ip + ql_base + k;
                        const uint8_t ql_byte = bx->ql[ql_idx];
                        const uint8_t ql_nibble = use_upper_nibble ? (ql_byte >> 4) : (ql_byte & 0x0F);

                        const int qh_idx = 32 * ip + qh_base + k;
                        const uint8_t qh_bits = (bx->qh[qh_idx] >> qh_shift_base) & 0x03;

                        int8_t q = (int8_t)((ql_nibble) | (qh_bits << 4)) - 32;
                        tile_a[m * XMX_TK + half * 16 + k] = q;
                    }
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);

                // XMX compute + store to SLM scratch for float epilogue
                {
                    syclmx::joint_matrix<sycl::sub_group, int8_t,  syclmx::use::a,           XMX_TM, XMX_TK, syclmx::layout::row_major> mat_a;
                    syclmx::joint_matrix<sycl::sub_group, int8_t,  syclmx::use::b,           XMX_TK, XMX_TN, syclmx::layout::row_major> mat_b;
                    syclmx::joint_matrix<sycl::sub_group, int32_t, syclmx::use::accumulator, XMX_TM, XMX_TN> mat_c;

                    syclmx::joint_matrix_fill(sg, mat_c, 0);
                    syclmx::joint_matrix_load(sg, mat_a,
                        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                            tile_a + sg_m_offset * XMX_TK),
                        XMX_TK);
                    syclmx::joint_matrix_load(sg, mat_b,
                        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                            tile_b + sg_n_offset),
                        XMX_WG_N);

                    syclmx::joint_matrix_mad(sg, mat_c, mat_a, mat_b, mat_c);

                    syclmx::joint_matrix_store(sg, mat_c,
                        slm_ptr(c_scratch + c_scratch_base),
                        XMX_TN, syclmx::layout::row_major);

                    for (int i = lane_id; i < XMX_TM * XMX_TN; i += WARP_SIZE) {
                        const int m = i / XMX_TN;
                        const int n = i % XMX_TN;
                        const int32_t val = c_scratch[c_scratch_base + m * XMX_TN + n];
                        const float d_val   = tile_a_d[sg_m_offset + m];
                        const float sc_val  = (float)tile_a_sc[sg_m_offset + m];
                        const float b_scale = tile_b_d[sg_n_offset + n];
                        acc_frag[i / WARP_SIZE] += (float)val * d_val * sc_val * b_scale;
                    }
                }

                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    for (int i = lane_id; i < XMX_TM * XMX_TN; i += WARP_SIZE) {
        const int m = i / XMX_TN;
        const int n = i % XMX_TN;
        const int row = row_dst_0 + sg_m_offset + m;
        const int col = col_dst_0 + sg_n_offset + n;
        if (row < nrows_dst && col < ncols_y) {
            dst[col * nrows_dst + row] = acc_frag[i / WARP_SIZE];
        }
    }
}


// ============================================================================
// Host-side launch wrappers
// ============================================================================

static void ggml_mul_mat_q8_0_q8_1_xmx_sycl(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    dpct::queue_ptr stream) try {

    const int block_num_x = (nrows_x + XMX_WG_M - 1) / XMX_WG_M;
    const int block_num_y = (ncols_y + XMX_WG_N - 1) / XMX_WG_N;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, XMX_NWARPS, WARP_SIZE);

    const bool need_check = (nrows_x % XMX_WG_M != 0) || (ncols_y % XMX_WG_N != 0);

    stream->submit([&](sycl::handler & cgh) {
        // SLM allocations
        sycl::local_accessor<int8_t, 1>  tile_a_acc(sycl::range<1>(XMX_WG_M * XMX_TK), cgh);
        sycl::local_accessor<int8_t, 1>  tile_b_acc(sycl::range<1>(XMX_TK * XMX_WG_N), cgh);
        sycl::local_accessor<float, 1>   tile_a_d_acc(sycl::range<1>(XMX_WG_M), cgh);
        sycl::local_accessor<float, 1>   tile_b_d_acc(sycl::range<1>(XMX_WG_N), cgh);
        sycl::local_accessor<int32_t, 1> c_scratch_acc(sycl::range<1>(XMX_NWARPS * XMX_TM * XMX_TN), cgh);

        if (need_check) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_q8_0_xmx<true>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst,
                        item_ct1,
                        get_pointer(tile_a_acc), get_pointer(tile_b_acc),
                        get_pointer(tile_a_d_acc), get_pointer(tile_b_d_acc),
                        get_pointer(c_scratch_acc));
                });
        } else {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_q8_0_xmx<false>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst,
                        item_ct1,
                        get_pointer(tile_a_acc), get_pointer(tile_b_acc),
                        get_pointer(tile_a_d_acc), get_pointer(tile_b_d_acc),
                        get_pointer(c_scratch_acc));
                });
        }
    });
}
catch (sycl::exception const & exc) {
    GGML_LOG_ERROR("%s: %s\n", __func__, exc.what());
    GGML_ABORT("SYCL error in XMX Q8_0 MMQ");
}


template <k_quant_type qtype>
static void ggml_mul_mat_qX_K_q8_1_xmx_sycl(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    dpct::queue_ptr stream) try {

    const int block_num_x = (nrows_x + XMX_WG_M - 1) / XMX_WG_M;
    const int block_num_y = (ncols_y + XMX_WG_N - 1) / XMX_WG_N;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, XMX_NWARPS, WARP_SIZE);

    const bool need_check = (nrows_x % XMX_WG_M != 0) || (ncols_y % XMX_WG_N != 0);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int8_t, 1>  tile_a_acc(sycl::range<1>(XMX_WG_M * XMX_TK), cgh);
        sycl::local_accessor<int8_t, 1>  tile_b_acc(sycl::range<1>(XMX_TK * XMX_WG_N), cgh);
        sycl::local_accessor<float, 1>   tile_scales_acc(sycl::range<1>(XMX_WG_M * 2), cgh);
        sycl::local_accessor<float, 1>   tile_b_meta_acc(sycl::range<1>(XMX_WG_N * 2), cgh);
        sycl::local_accessor<int32_t, 1> c_scratch_acc(sycl::range<1>(XMX_NWARPS * XMX_TM * XMX_TN), cgh);

        if (need_check) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_qX_K_xmx<qtype, true>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst,
                        item_ct1,
                        get_pointer(tile_a_acc), get_pointer(tile_b_acc),
                        get_pointer(tile_scales_acc), get_pointer(tile_b_meta_acc),
                        get_pointer(c_scratch_acc));
                });
        } else {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_qX_K_xmx<qtype, false>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst,
                        item_ct1,
                        get_pointer(tile_a_acc), get_pointer(tile_b_acc),
                        get_pointer(tile_scales_acc), get_pointer(tile_b_meta_acc),
                        get_pointer(c_scratch_acc));
                });
        }
    });
}
catch (sycl::exception const & exc) {
    GGML_LOG_ERROR("%s: %s\n", __func__, exc.what());
    GGML_ABORT("SYCL error in XMX K-quant MMQ");
}


static void ggml_mul_mat_q6_K_q8_1_xmx_sycl(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst,
    dpct::queue_ptr stream) try {

    const int block_num_x = (nrows_x + XMX_WG_M - 1) / XMX_WG_M;
    const int block_num_y = (ncols_y + XMX_WG_N - 1) / XMX_WG_N;
    const sycl::range<3> block_nums(1, block_num_y, block_num_x);
    const sycl::range<3> block_dims(1, XMX_NWARPS, WARP_SIZE);

    const bool need_check = (nrows_x % XMX_WG_M != 0) || (ncols_y % XMX_WG_N != 0);

    // Q6_K needs extra SLM for per-16 scales
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int8_t, 1>  tile_a_acc(sycl::range<1>(XMX_WG_M * XMX_TK), cgh);
        sycl::local_accessor<int8_t, 1>  tile_b_acc(sycl::range<1>(XMX_TK * XMX_WG_N), cgh);
        sycl::local_accessor<float, 1>   tile_a_d_acc(sycl::range<1>(XMX_WG_M), cgh);
        sycl::local_accessor<int8_t, 1>  tile_a_sc_acc(sycl::range<1>(XMX_WG_M), cgh);
        sycl::local_accessor<float, 1>   tile_b_d_acc(sycl::range<1>(XMX_WG_N), cgh);
        sycl::local_accessor<int32_t, 1> c_scratch_acc(sycl::range<1>(XMX_NWARPS * XMX_TM * XMX_TN), cgh);

        if (need_check) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_q6_K_xmx<true>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst,
                        item_ct1,
                        get_pointer(tile_a_acc), get_pointer(tile_b_acc),
                        get_pointer(tile_a_d_acc), get_pointer(tile_a_sc_acc),
                        get_pointer(tile_b_d_acc),
                        get_pointer(c_scratch_acc));
                });
        } else {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_q6_K_xmx<false>(
                        vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst,
                        item_ct1,
                        get_pointer(tile_a_acc), get_pointer(tile_b_acc),
                        get_pointer(tile_a_d_acc), get_pointer(tile_a_sc_acc),
                        get_pointer(tile_b_d_acc),
                        get_pointer(c_scratch_acc));
                });
        }
    });
}
catch (sycl::exception const & exc) {
    GGML_LOG_ERROR("%s: %s\n", __func__, exc.what());
    GGML_ABORT("SYCL error in XMX Q6_K MMQ");
}


// ============================================================================
// Public dispatch entry point
// ============================================================================
void ggml_sycl_op_mul_mat_q_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i, const char * src1_ddq_i,
    float * dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr & stream) try {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    int device_id;
    SYCL_CHECK(CHECK_TRY_ERROR(device_id = get_current_device_id()));

    const int64_t nrows_dst = device_id == ctx.device ? ne0 : row_diff;

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            ggml_mul_mat_q8_0_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i,
                ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q4_K:
            ggml_mul_mat_qX_K_q8_1_xmx_sycl<k_quant_type::Q4_K>(src0_dd_i, src1_ddq_i, dst_dd_i,
                ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            ggml_mul_mat_qX_K_q8_1_xmx_sycl<k_quant_type::Q5_K>(src0_dd_i, src1_ddq_i, dst_dd_i,
                ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            ggml_mul_mat_q6_K_q8_1_xmx_sycl(src0_dd_i, src1_ddq_i, dst_dd_i,
                ne00, row_diff, src1_ncols, src1_padded_row_size, nrows_dst, stream);
            break;
        default:
            GGML_ABORT("XMX MMQ: unsupported type");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}
catch (sycl::exception const & exc) {
    GGML_LOG_ERROR("%s: %s\n", __func__, exc.what());
    GGML_ABORT("SYCL error in XMX MMQ dispatch");
}

#else // !SYCL_USE_XMX

bool ggml_sycl_supports_xmx_mmq(const sycl::device & dev, ggml_type type) {
    GGML_UNUSED(dev);
    GGML_UNUSED(type);
    return false;
}

void ggml_sycl_op_mul_mat_q_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i, const char * src1_ddq_i,
    float * dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr & stream) {
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_UNUSED(src0_dd_i); GGML_UNUSED(src1_ddf_i); GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(dst_dd_i); GGML_UNUSED(row_low); GGML_UNUSED(row_high);
    GGML_UNUSED(src1_ncols); GGML_UNUSED(src1_padded_row_size); GGML_UNUSED(stream);
    GGML_ABORT("XMX MMQ: not compiled with SYCL_USE_XMX");
}

#endif // SYCL_USE_XMX
