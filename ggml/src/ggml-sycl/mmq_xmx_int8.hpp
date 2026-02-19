#ifndef GGML_SYCL_MMQ_XMX_INT8_HPP
#define GGML_SYCL_MMQ_XMX_INT8_HPP

#include "common.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

static inline bool has_int8_xmx_support(const dpct::queue_ptr & q) {
    auto dev = q->get_device();
    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }

    auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    for (auto & comb : combinations) {
        bool has_int8_a = (comb.atype == matrix_type::sint8 || comb.atype == matrix_type::uint8);
        bool has_int8_b = (comb.btype == matrix_type::sint8 || comb.btype == matrix_type::uint8);

        if (has_int8_a && has_int8_b) {
            return true;
        }
    }

    return false;
}

struct xmx_int8_tile_config {
    int TM;
    int TN;
    int TK;
};

static inline xmx_int8_tile_config get_int8_xmx_tile_config(const dpct::queue_ptr & q) {
    auto dev          = q->get_device();
    auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    for (auto & comb : combinations) {
        bool has_int8_a = (comb.atype == matrix_type::sint8 || comb.atype == matrix_type::uint8);
        bool has_int8_b = (comb.btype == matrix_type::sint8 || comb.btype == matrix_type::uint8);

        if (has_int8_a && has_int8_b) {
            if (comb.nsize == 16) {
                return { 8, 16, 32 };
            } else if (comb.nsize == 8) {
                return { 8, 8, 32 };
            } else if (comb.nsize == 0) {
                return { 16, 16, 64 };
            }
        }
    }

    return { 8, 16, 32 };
}

// All XMX int8 kernels read src1 (vy) in standard block_q8_1 AoS layout.
// block_q8_1 = { half2 ds; int8_t qs[32]; } = 36 bytes
// src1 is laid out as block_q8_1[N][K_blocks] where K_blocks = K_padded/32.
//
// Both matA (from src0) and matB (from src1) are staged through SLM to ensure
// proper alignment for joint_matrix_load.

// === Optimized multi-subgroup kernel with col_major B layout ===
// N_SG subgroups cover different M-row strips.
// Each subgroup computes TM x (TILES_N * TN) output by loading A once and
// multiplying against TILES_N different B tiles.
// Workgroup covers (N_SG * TM) rows x (TILES_N * TN) columns.
//
// Key optimization: col_major B layout eliminates scatter-transpose.
// block_q8_1.qs[32] is contiguous and TK=32=QK, so each block's qs[] array
// IS a column of B. By storing B as [N][K] in SLM and loading with col_major
// layout, we use vectorized stores instead of per-byte scatter.
//
// SLM layout:
//   B: [TILES_N * TN][TK] — each row = one block's qs[32], col_major load
//   A: per-sg [TM][TK] — row_major
//   C: per-sg [TM][TN] int32 — accumulator scratch

template <int TM, int TN, int TK, int N_SG, int TILES_N>
static void mmq_q8_0_xmx_kernel(const block_q8_0 * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int                K,
                                   const int                K_padded,
                                   const int                M,
                                   const int                N,
                                   const int                ldc,
                                   const sycl::nd_item<2> & item_ct1,
                                   int8_t *                 slm) {
    const auto sg = item_ct1.get_sub_group();

    const int local_linear = item_ct1.get_local_linear_id();
    const int sg_id        = local_linear / 16;
    const int lane_id      = local_linear % 16;

    const int base_m    = item_ct1.get_group(0) * (TM * N_SG);
    const int sg_startx = base_m + sg_id * TM;
    const int base_n    = item_ct1.get_group(1) * (TN * TILES_N);

    const int A_blocks = K / QK8_0;
    const int B_blocks = K_padded / QK8_1;
    const int K_tiles  = K / TK;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t *  slm_B = slm;
    int8_t *  slm_A = slm + TILES_N * TN * TK + sg_id * (TM * TK);
    int32_t * slm_C = (int32_t *) (slm + TILES_N * TN * TK + N_SG * TM * TK) + sg_id * (TM * TN);

    float acc[TILES_N * TM];
    for (int i = 0; i < TILES_N * TM; i++) {
        acc[i] = 0.0f;
    }

    constexpr int total_threads = N_SG * 16;
    constexpr int VEC_K         = 4;

    for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
        // Load B: copy qs[32] contiguously per column (no transpose!)
        // Each of TILES_N * TN columns = one block_q8_1's qs[] array
        // Use vectorized 4-byte loads from contiguous qs[]
        constexpr int total_B_vec = TILES_N * TN * (TK / VEC_K);
        for (int base = local_linear; base < total_B_vec; base += total_threads) {
            int col_idx = base / (TK / VEC_K);  // which column (0..TILES_N*TN-1)
            int k_start = (base % (TK / VEC_K)) * VEC_K;
            int nt      = col_idx / TN;
            int n       = col_idx % TN;
            int col     = base_n + nt * TN + n;
            if (col < N) {
                const block_q8_1 *       blk = &src1_q8[col * B_blocks + k_tile];
                sycl::vec<int8_t, VEC_K> v   = *reinterpret_cast<const sycl::vec<int8_t, VEC_K> *>(&blk->qs[k_start]);
                // Store contiguously: slm_B[col_idx * TK + k]
                *reinterpret_cast<sycl::vec<int8_t, VEC_K> *>(&slm_B[col_idx * TK + k_start]) = v;
            } else {
                for (int k = 0; k < VEC_K; k++) {
                    slm_B[col_idx * TK + k_start + k] = 0;
                }
            }
        }

        // Each subgroup loads its own A tile [TM][TK]
        for (int base = lane_id; base < TM * (TK / VEC_K); base += 16) {
            int i       = base / (TK / VEC_K);
            int k_start = (base % (TK / VEC_K)) * VEC_K;
            int row     = sg_startx + i;
            if (row < M) {
                const block_q8_0 *       blk = &vx[row * A_blocks + k_tile];
                sycl::vec<int8_t, VEC_K> v   = *reinterpret_cast<const sycl::vec<int8_t, VEC_K> *>(&blk->qs[k_start]);
                slm_A[i * TK + k_start + 0]  = v.x();
                slm_A[i * TK + k_start + 1]  = v.y();
                slm_A[i * TK + k_start + 2]  = v.z();
                slm_A[i * TK + k_start + 3]  = v.w();
            } else {
                for (int k = 0; k < VEC_K; k++) {
                    slm_A[i * TK + k_start + k] = 0;
                }
            }
        }

        sycl::group_barrier(item_ct1.get_group());

        // Load A from SLM (row_major as always)
        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        auto pA = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        joint_matrix_load(sg, matA, pA, TK);

        // Multiply against each of TILES_N B tiles using col_major load
        for (int nt = 0; nt < TILES_N; nt++) {
            int8_t * slm_B_nt = slm_B + nt * TN * TK;  // start of this tile's N columns

            joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matB;
            auto pB = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                slm_B_nt);
            // col_major: stride = number of elements between consecutive columns = TK
            joint_matrix_load(sg, matB, pB, TK);

            joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
            joint_matrix_fill(sg, matC, 0);
            joint_matrix_mad(sg, matC, matA, matB, matC);

            auto pC =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);
            joint_matrix_store(sg, matC, pC, TN, layout::row_major);
            sycl::group_barrier(sg);

            int n_base = base_n + nt * TN;
            for (int i = 0; i < TM; i++) {
                int row = sg_startx + i;
                if (row < M && n_base + lane_id < N) {
                    float   scale_a = (float) vx[row * A_blocks + k_tile].d;
                    float   scale_b = (float) src1_q8[(n_base + lane_id) * B_blocks + k_tile].ds[0];
                    int32_t val     = slm_C[i * TN + lane_id];
                    acc[nt * TM + i] += (float) val * scale_a * scale_b;
                }
            }
        }

        sycl::group_barrier(item_ct1.get_group());
    }

    for (int nt = 0; nt < TILES_N; nt++) {
        int n_base = base_n + nt * TN;
        for (int i = 0; i < TM; i++) {
            if (sg_startx + i < M && n_base + lane_id < N) {
                dst[(n_base + lane_id) * ldc + (sg_startx + i)] = acc[nt * TM + i];
            }
        }
    }
}

// === Legacy single-subgroup kernels for other quant types ===
// SLM layout: [TM * TN int32_t for accumulator store] [TM * TK int8_t for matA] [TK * TN int8_t for matB]
// Total SLM int32_t count: TM*TN + ceil((TM*TK + TK*TN) / 4)

template <int TM, int TN, int TK>
static void mmq_q4_0_xmx_kernel(const block_q4_0 * __restrict__ vx,
                                const void * __restrict__ vy,
                                float * __restrict__ dst,
                                const int                K,
                                const int                K_padded,
                                const int                M,
                                const int                N,
                                const int                ldc,
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int A_blocks = K / QK4_0;
    const int B_blocks = K_padded / QK8_1;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);
    int8_t * slm_B = slm_A + TM * TK;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        // Stage matA: dequantize q4_0 into int8 in SLM [TM][TK]
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += sg_size) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q4_0 * block  = &vx[row * A_blocks + k_tile * (TK / QK4_0)];
                uint8_t            qs_val = block->qs[j];
                slm_A[idx * 2 + 0]        = (int8_t) ((qs_val >> 0) & 0x0F) - 8;
                slm_A[idx * 2 + 1]        = (int8_t) ((qs_val >> 4) & 0x0F) - 8;
            } else {
                slm_A[idx * 2 + 0] = 0;
                slm_A[idx * 2 + 1] = 0;
            }
        }

        // Stage matB: load q8_1 quants into SLM [TK][TN]
        for (int idx = lane_id; idx < TK * TN; idx += sg_size) {
            int k   = idx / TN;
            int n   = idx % TN;
            int col = sg_starty + n;
            if (col < N) {
                const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile * (TK / QK8_1) + k / QK8_1];
                slm_B[idx]             = blk->qs[k % QK8_1];
            } else {
                slm_B[idx] = 0;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto pB =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);

        joint_matrix_load(sg, matA, pA, TK);
        joint_matrix_load(sg, matB, pB, TN);

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M && sg_starty + lane_id < N) {
                const block_q4_0 * block   = &vx[row * A_blocks + k_tile * (TK / QK4_0)];
                float              scale_a = (float) block->d;
                float   scale_b = (float) src1_q8[(sg_starty + lane_id) * B_blocks + k_tile * (TK / QK8_1)].ds[0];
                int32_t val     = slm_tile[i * TN + lane_id];
                acc[i] += (float) val * scale_a * scale_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_starty + lane_id) * ldc + (sg_startx + i)] = acc[i];
        }
    }
}

template <int TM, int TN, int TK>
static void mmq_q4_1_xmx_kernel(const block_q4_1 * __restrict__ vx,
                                const void * __restrict__ vy,
                                float * __restrict__ dst,
                                const int                K,
                                const int                K_padded,
                                const int                M,
                                const int                N,
                                const int                ldc,
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int A_blocks = K / QK4_1;
    const int B_blocks = K_padded / QK8_1;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);
    int8_t * slm_B = slm_A + TM * TK;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        // Stage matA: dequantize q4_1 into int8 in SLM [TM][TK]
        // q4_1 stores unsigned 4-bit values; no bias subtraction needed here
        // because the bias term is handled separately via the min value (m)
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += sg_size) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q4_1 * block  = &vx[row * A_blocks + k_tile * (TK / QK4_1)];
                uint8_t            qs_val = block->qs[j];
                slm_A[idx * 2 + 0]        = (int8_t) ((qs_val >> 0) & 0x0F);
                slm_A[idx * 2 + 1]        = (int8_t) ((qs_val >> 4) & 0x0F);
            } else {
                slm_A[idx * 2 + 0] = 0;
                slm_A[idx * 2 + 1] = 0;
            }
        }

        // Stage matB
        for (int idx = lane_id; idx < TK * TN; idx += sg_size) {
            int k   = idx / TN;
            int n   = idx % TN;
            int col = sg_starty + n;
            if (col < N) {
                const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile * (TK / QK8_1) + k / QK8_1];
                slm_B[idx]             = blk->qs[k % QK8_1];
            } else {
                slm_B[idx] = 0;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto pB =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);

        joint_matrix_load(sg, matA, pA, TK);
        joint_matrix_load(sg, matB, pB, TN);

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M && sg_starty + lane_id < N) {
                const block_q4_1 * block   = &vx[row * A_blocks + k_tile * (TK / QK4_1)];
                const sycl::half2  dm      = block->dm;
                float              d       = (float) dm[0];
                float              m       = (float) dm[1];
                sycl::half2        ds8     = src1_q8[(sg_starty + lane_id) * B_blocks + k_tile * (TK / QK8_1)].ds;
                float              scale_b = (float) ds8[0];
                float              sum_b   = (float) ds8[1];
                int32_t            val     = slm_tile[i * TN + lane_id];
                acc[i] += d * (float) val * scale_b + m * sum_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_starty + lane_id) * ldc + (sg_startx + i)] = acc[i];
        }
    }
}

template <int TM, int TN, int TK>
static void mmq_q5_0_xmx_kernel(const block_q5_0 * __restrict__ vx,
                                const void * __restrict__ vy,
                                float * __restrict__ dst,
                                const int                K,
                                const int                K_padded,
                                const int                M,
                                const int                N,
                                const int                ldc,
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int A_blocks = K / QK5_0;
    const int B_blocks = K_padded / QK8_1;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);
    int8_t * slm_B = slm_A + TM * TK;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        // Stage matA: dequantize q5_0 into int8 in SLM [TM][TK]
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += sg_size) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q5_0 * block = &vx[row * A_blocks + k_tile * (TK / QK5_0)];
                const uint8_t *    qs    = block->qs;
                const uint32_t     qh    = *(const uint32_t *) block->qh;

                uint8_t q0         = (qs[j] >> 0) & 0x0F;
                uint8_t q1         = (qs[j] >> 4) & 0x0F;
                uint8_t h0         = (qh >> (j * 2 + 0)) & 0x01;
                uint8_t h1         = (qh >> (j * 2 + 1)) & 0x01;
                slm_A[idx * 2 + 0] = (int8_t) ((q0 | (h0 << 4)) - 16);
                slm_A[idx * 2 + 1] = (int8_t) ((q1 | (h1 << 4)) - 16);
            } else {
                slm_A[idx * 2 + 0] = 0;
                slm_A[idx * 2 + 1] = 0;
            }
        }

        // Stage matB
        for (int idx = lane_id; idx < TK * TN; idx += sg_size) {
            int k   = idx / TN;
            int n   = idx % TN;
            int col = sg_starty + n;
            if (col < N) {
                const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile * (TK / QK8_1) + k / QK8_1];
                slm_B[idx]             = blk->qs[k % QK8_1];
            } else {
                slm_B[idx] = 0;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto pB =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);

        joint_matrix_load(sg, matA, pA, TK);
        joint_matrix_load(sg, matB, pB, TN);

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M && sg_starty + lane_id < N) {
                const block_q5_0 * block   = &vx[row * A_blocks + k_tile * (TK / QK5_0)];
                float              scale_a = (float) block->d;
                float   scale_b = (float) src1_q8[(sg_starty + lane_id) * B_blocks + k_tile * (TK / QK8_1)].ds[0];
                int32_t val     = slm_tile[i * TN + lane_id];
                acc[i] += (float) val * scale_a * scale_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_starty + lane_id) * ldc + (sg_startx + i)] = acc[i];
        }
    }
}

template <int TM, int TN, int TK>
static void mmq_q5_1_xmx_kernel(const block_q5_1 * __restrict__ vx,
                                const void * __restrict__ vy,
                                float * __restrict__ dst,
                                const int                K,
                                const int                K_padded,
                                const int                M,
                                const int                N,
                                const int                ldc,
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int A_blocks = K / QK5_1;
    const int B_blocks = K_padded / QK8_1;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);
    int8_t * slm_B = slm_A + TM * TK;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        // Stage matA: dequantize q5_1 into int8 in SLM [TM][TK]
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += sg_size) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q5_1 * block = &vx[row * A_blocks + k_tile * (TK / QK5_1)];
                const uint8_t *    qs    = block->qs;
                const uint32_t     qh    = *(const uint32_t *) block->qh;

                uint8_t q0         = (qs[j] >> 0) & 0x0F;
                uint8_t q1         = (qs[j] >> 4) & 0x0F;
                uint8_t h0         = (qh >> (j * 2 + 0)) & 0x01;
                uint8_t h1         = (qh >> (j * 2 + 1)) & 0x01;
                slm_A[idx * 2 + 0] = (int8_t) ((q0 | (h0 << 4)) - 16);
                slm_A[idx * 2 + 1] = (int8_t) ((q1 | (h1 << 4)) - 16);
            } else {
                slm_A[idx * 2 + 0] = 0;
                slm_A[idx * 2 + 1] = 0;
            }
        }

        // Stage matB
        for (int idx = lane_id; idx < TK * TN; idx += sg_size) {
            int k   = idx / TN;
            int n   = idx % TN;
            int col = sg_starty + n;
            if (col < N) {
                const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile * (TK / QK8_1) + k / QK8_1];
                slm_B[idx]             = blk->qs[k % QK8_1];
            } else {
                slm_B[idx] = 0;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto pB =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);

        joint_matrix_load(sg, matA, pA, TK);
        joint_matrix_load(sg, matB, pB, TN);

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M && sg_starty + lane_id < N) {
                const block_q5_1 * block   = &vx[row * A_blocks + k_tile * (TK / QK5_1)];
                const sycl::half2  dm      = block->dm;
                float              d       = (float) dm[0];
                float              m       = (float) dm[1];
                sycl::half2        ds8     = src1_q8[(sg_starty + lane_id) * B_blocks + k_tile * (TK / QK8_1)].ds;
                float              scale_b = (float) ds8[0];
                float              sum_b   = (float) ds8[1];
                int32_t            val     = slm_tile[i * TN + lane_id];
                acc[i] += d * (float) val * scale_b + m * sum_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_starty + lane_id) * ldc + (sg_startx + i)] = acc[i];
        }
    }
}

template <int TM, int TN, int TK>
static void mmq_q8_1_xmx_kernel(const block_q8_1 * __restrict__ vx,
                                const void * __restrict__ vy,
                                float * __restrict__ dst,
                                const int                K,
                                const int                K_padded,
                                const int                M,
                                const int                N,
                                const int                ldc,
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int A_blocks = K / QK8_1;         // src0 blocks per row (unpadded)
    const int B_blocks = K_padded / QK8_1;  // src1 blocks per row (padded)

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);
    int8_t * slm_B = slm_A + TM * TK;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        // Stage matA: load q8_1 quants into SLM [TM][TK]
        for (int idx = lane_id; idx < TM * TK; idx += sg_size) {
            int i   = idx / TK;
            int k   = idx % TK;
            int row = sg_startx + i;
            if (row < M) {
                const block_q8_1 * blk = &vx[row * A_blocks + k_tile * (TK / QK8_1) + k / QK8_1];
                slm_A[idx]             = blk->qs[k % QK8_1];
            } else {
                slm_A[idx] = 0;
            }
        }

        // Stage matB: load q8_1 quants into SLM [TK][TN]
        for (int idx = lane_id; idx < TK * TN; idx += sg_size) {
            int k   = idx / TN;
            int n   = idx % TN;
            int col = sg_starty + n;
            if (col < N) {
                const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile * (TK / QK8_1) + k / QK8_1];
                slm_B[idx]             = blk->qs[k % QK8_1];
            } else {
                slm_B[idx] = 0;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        auto pB =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);

        joint_matrix_load(sg, matA, pA, TK);
        joint_matrix_load(sg, matB, pB, TN);

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M && sg_starty + lane_id < N) {
                float   scale_a = (float) vx[row * A_blocks + k_tile].ds[0];
                float   scale_b = (float) src1_q8[(sg_starty + lane_id) * B_blocks + k_tile].ds[0];
                int32_t val     = slm_tile[i * TN + lane_id];
                acc[i] += (float) val * scale_a * scale_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_starty + lane_id) * ldc + (sg_startx + i)] = acc[i];
        }
    }
}

// === K-quant XMX int8 kernels ===
// K-quants use QK_K=256 super-blocks. With TK=32 we iterate 8 k_tiles per super-block.
// Q4_K/Q5_K have 8 sub-blocks of 32 — one sub-block per TK=32 tile (perfect alignment).
// Q6_K has 16 sub-blocks of 16 — two sub-blocks per TK=32 tile (handled with two MADs).

// Helper: extract 6-bit scale and min for Q4_K/Q5_K sub-block j from packed scales array.
static inline void get_scale_min_k4_xmx(int j, const uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Q4_K multi-subgroup XMX kernel with col_major B layout
// k_tile maps: sb = k_tile/8, sb_tile = k_tile%8, pair = sb_tile/2, half = sb_tile%2
// half=0: low nibble of qs[32*pair+j], half=1: high nibble
template <int TM, int TN, int TK, int N_SG, int TILES_N>
static void mmq_q4_K_xmx_kernel(const block_q4_K * __restrict__ vx,
                                 const void * __restrict__ vy,
                                 float * __restrict__ dst,
                                 const int                K,
                                 const int                K_padded,
                                 const int                M,
                                 const int                N,
                                 const int                ldc,
                                 const sycl::nd_item<2> & item_ct1,
                                 int8_t *                 slm) {
    const auto sg = item_ct1.get_sub_group();

    const int local_linear = item_ct1.get_local_linear_id();
    const int sg_id        = local_linear / 16;
    const int lane_id      = local_linear % 16;

    const int base_m    = item_ct1.get_group(0) * (TM * N_SG);
    const int sg_startx = base_m + sg_id * TM;
    const int base_n    = item_ct1.get_group(1) * (TN * TILES_N);

    const int A_blocks = K / QK_K;
    const int B_blocks = K_padded / QK8_1;
    const int K_tiles  = K / TK;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t *  slm_B = slm;
    int8_t *  slm_A = slm + TILES_N * TN * TK + sg_id * (TM * TK);
    int32_t * slm_C = (int32_t *) (slm + TILES_N * TN * TK + N_SG * TM * TK) + sg_id * (TM * TN);

    float acc[TILES_N * TM];
    for (int i = 0; i < TILES_N * TM; i++) {
        acc[i] = 0.0f;
    }

    constexpr int total_threads = N_SG * 16;
    constexpr int VEC_K         = 4;

    for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
        const int sb      = k_tile / 8;
        const int sb_tile = k_tile % 8;
        const int pair    = sb_tile / 2;
        const int half    = sb_tile % 2;

        // Load B: col_major layout (same as Q8_0 kernel)
        constexpr int total_B_vec = TILES_N * TN * (TK / VEC_K);
        for (int base_b = local_linear; base_b < total_B_vec; base_b += total_threads) {
            int col_idx = base_b / (TK / VEC_K);
            int k_start = (base_b % (TK / VEC_K)) * VEC_K;
            int nt      = col_idx / TN;
            int n       = col_idx % TN;
            int col     = base_n + nt * TN + n;
            if (col < N) {
                const block_q8_1 *       blk = &src1_q8[col * B_blocks + k_tile];
                sycl::vec<int8_t, VEC_K> v   = *reinterpret_cast<const sycl::vec<int8_t, VEC_K> *>(&blk->qs[k_start]);
                *reinterpret_cast<sycl::vec<int8_t, VEC_K> *>(&slm_B[col_idx * TK + k_start]) = v;
            } else {
                for (int k = 0; k < VEC_K; k++) {
                    slm_B[col_idx * TK + k_start + k] = 0;
                }
            }
        }

        // Load A: dequantize Q4_K nibbles to unsigned int8 (0-15)
        for (int idx = lane_id; idx < TM * TK; idx += 16) {
            int i   = idx / TK;
            int j   = idx % TK;
            int row = sg_startx + i;
            if (row < M) {
                const block_q4_K * blk    = &vx[row * A_blocks + sb];
                uint8_t            qs_byte = blk->qs[32 * pair + j];
                slm_A[i * TK + j] = (half == 0) ? (int8_t)(qs_byte & 0x0F) : (int8_t)(qs_byte >> 4);
            } else {
                slm_A[i * TK + j] = 0;
            }
        }

        sycl::group_barrier(item_ct1.get_group());

        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        auto pA = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        joint_matrix_load(sg, matA, pA, TK);

        for (int nt = 0; nt < TILES_N; nt++) {
            int8_t * slm_B_nt = slm_B + nt * TN * TK;

            joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matB;
            auto pB = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B_nt);
            joint_matrix_load(sg, matB, pB, TK);

            joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
            joint_matrix_fill(sg, matC, 0);
            joint_matrix_mad(sg, matC, matA, matB, matC);

            auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);
            joint_matrix_store(sg, matC, pC, TN, layout::row_major);
            sycl::group_barrier(sg);

            int n_base = base_n + nt * TN;
            for (int i = 0; i < TM; i++) {
                int row = sg_startx + i;
                if (row < M && n_base + lane_id < N) {
                    const block_q4_K * blk  = &vx[row * A_blocks + sb];
                    float              dall  = (float) blk->dm[0];
                    float              dmin  = (float) blk->dm[1];

                    uint8_t sc, m_val;
                    get_scale_min_k4_xmx(sb_tile, blk->scales, sc, m_val);

                    const block_q8_1 & b_blk   = src1_q8[(n_base + lane_id) * B_blocks + k_tile];
                    float              scale_b  = (float) b_blk.ds[0];
                    float              sum_b    = (float) b_blk.ds[1];

                    int32_t val = slm_C[i * TN + lane_id];
                    acc[nt * TM + i] += dall * (float) sc * (float) val * scale_b - dmin * (float) m_val * sum_b;
                }
            }
        }

        sycl::group_barrier(item_ct1.get_group());
    }

    for (int nt = 0; nt < TILES_N; nt++) {
        int n_base = base_n + nt * TN;
        for (int i = 0; i < TM; i++) {
            if (sg_startx + i < M && n_base + lane_id < N) {
                dst[(n_base + lane_id) * ldc + (sg_startx + i)] = acc[nt * TM + i];
            }
        }
    }
}

// Q5_K multi-subgroup XMX kernel with col_major B layout
// Same sub-block structure as Q4_K but 5-bit values (4 bits from qs + 1 bit from qh)
template <int TM, int TN, int TK, int N_SG, int TILES_N>
static void mmq_q5_K_xmx_kernel(const block_q5_K * __restrict__ vx,
                                 const void * __restrict__ vy,
                                 float * __restrict__ dst,
                                 const int                K,
                                 const int                K_padded,
                                 const int                M,
                                 const int                N,
                                 const int                ldc,
                                 const sycl::nd_item<2> & item_ct1,
                                 int8_t *                 slm) {
    const auto sg = item_ct1.get_sub_group();

    const int local_linear = item_ct1.get_local_linear_id();
    const int sg_id        = local_linear / 16;
    const int lane_id      = local_linear % 16;

    const int base_m    = item_ct1.get_group(0) * (TM * N_SG);
    const int sg_startx = base_m + sg_id * TM;
    const int base_n    = item_ct1.get_group(1) * (TN * TILES_N);

    const int A_blocks = K / QK_K;
    const int B_blocks = K_padded / QK8_1;
    const int K_tiles  = K / TK;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t *  slm_B = slm;
    int8_t *  slm_A = slm + TILES_N * TN * TK + sg_id * (TM * TK);
    int32_t * slm_C = (int32_t *) (slm + TILES_N * TN * TK + N_SG * TM * TK) + sg_id * (TM * TN);

    float acc[TILES_N * TM];
    for (int i = 0; i < TILES_N * TM; i++) {
        acc[i] = 0.0f;
    }

    constexpr int total_threads = N_SG * 16;
    constexpr int VEC_K         = 4;

    for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
        const int sb      = k_tile / 8;
        const int sb_tile = k_tile % 8;
        const int pair    = sb_tile / 2;
        const int half    = sb_tile % 2;

        // Load B: col_major layout
        constexpr int total_B_vec = TILES_N * TN * (TK / VEC_K);
        for (int base_b = local_linear; base_b < total_B_vec; base_b += total_threads) {
            int col_idx = base_b / (TK / VEC_K);
            int k_start = (base_b % (TK / VEC_K)) * VEC_K;
            int nt      = col_idx / TN;
            int n       = col_idx % TN;
            int col     = base_n + nt * TN + n;
            if (col < N) {
                const block_q8_1 *       blk = &src1_q8[col * B_blocks + k_tile];
                sycl::vec<int8_t, VEC_K> v   = *reinterpret_cast<const sycl::vec<int8_t, VEC_K> *>(&blk->qs[k_start]);
                *reinterpret_cast<sycl::vec<int8_t, VEC_K> *>(&slm_B[col_idx * TK + k_start]) = v;
            } else {
                for (int k = 0; k < VEC_K; k++) {
                    slm_B[col_idx * TK + k_start + k] = 0;
                }
            }
        }

        // Load A: dequantize Q5_K to unsigned int8 (0-31)
        for (int idx = lane_id; idx < TM * TK; idx += 16) {
            int i   = idx / TK;
            int j   = idx % TK;
            int row = sg_startx + i;
            if (row < M) {
                const block_q5_K * blk      = &vx[row * A_blocks + sb];
                uint8_t            qs_byte   = blk->qs[32 * pair + j];
                int                elem_idx  = 64 * pair + 32 * half + j;
                uint8_t            qh_bit    = (blk->qh[elem_idx / 8] >> (elem_idx % 8)) & 1;
                uint8_t            q_low     = (half == 0) ? (qs_byte & 0x0F) : (qs_byte >> 4);
                slm_A[i * TK + j] = (int8_t)(q_low | (qh_bit << 4));
            } else {
                slm_A[i * TK + j] = 0;
            }
        }

        sycl::group_barrier(item_ct1.get_group());

        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
        auto pA = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        joint_matrix_load(sg, matA, pA, TK);

        for (int nt = 0; nt < TILES_N; nt++) {
            int8_t * slm_B_nt = slm_B + nt * TN * TK;

            joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matB;
            auto pB = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B_nt);
            joint_matrix_load(sg, matB, pB, TK);

            joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
            joint_matrix_fill(sg, matC, 0);
            joint_matrix_mad(sg, matC, matA, matB, matC);

            auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);
            joint_matrix_store(sg, matC, pC, TN, layout::row_major);
            sycl::group_barrier(sg);

            int n_base = base_n + nt * TN;
            for (int i = 0; i < TM; i++) {
                int row = sg_startx + i;
                if (row < M && n_base + lane_id < N) {
                    const block_q5_K * blk  = &vx[row * A_blocks + sb];
                    float              dall  = (float) blk->dm[0];
                    float              dmin  = (float) blk->dm[1];

                    uint8_t sc, m_val;
                    get_scale_min_k4_xmx(sb_tile, blk->scales, sc, m_val);

                    const block_q8_1 & b_blk   = src1_q8[(n_base + lane_id) * B_blocks + k_tile];
                    float              scale_b  = (float) b_blk.ds[0];
                    float              sum_b    = (float) b_blk.ds[1];

                    int32_t val = slm_C[i * TN + lane_id];
                    acc[nt * TM + i] += dall * (float) sc * (float) val * scale_b - dmin * (float) m_val * sum_b;
                }
            }
        }

        sycl::group_barrier(item_ct1.get_group());
    }

    for (int nt = 0; nt < TILES_N; nt++) {
        int n_base = base_n + nt * TN;
        for (int i = 0; i < TM; i++) {
            if (sg_startx + i < M && n_base + lane_id < N) {
                dst[(n_base + lane_id) * ldc + (sg_startx + i)] = acc[nt * TM + i];
            }
        }
    }
}

// Q6_K single-subgroup XMX kernel (legacy pattern)
// Q6_K has 16 sub-blocks of 16 elements with different scales per sub-block.
// With TK=32, each k_tile covers 2 sub-blocks. We handle this by doing TWO
// MADs per k_tile (each with 16 elements in first half, zero-padded to 32).
template <int TM, int TN, int TK>
static void mmq_q6_K_xmx_kernel(const block_q6_K * __restrict__ vx,
                                 const void * __restrict__ vy,
                                 float * __restrict__ dst,
                                 const int                K,
                                 const int                K_padded,
                                 const int                M,
                                 const int                N,
                                 const int                ldc,
                                 const sycl::nd_item<2> & item_ct1,
                                 int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int A_blocks = K / QK_K;
    const int B_blocks = K_padded / QK8_1;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);
    int8_t * slm_B = slm_A + TM * TK;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    const int K_tiles = K / TK;

    for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
        const int sb      = k_tile / 8;
        const int sb_tile = k_tile % 8;

        // Two sub-block scale indices for this 32-element chunk
        const int scale_idx0 = 2 * sb_tile;
        const int scale_idx1 = 2 * sb_tile + 1;

        // Process two halves of 16 elements each with separate MADs
        for (int sub_half = 0; sub_half < 2; sub_half++) {
            const int scale_idx = (sub_half == 0) ? scale_idx0 : scale_idx1;

            // Stage A: decode Q6_K values for this 16-element half, zero-pad to TK=32
            for (int idx = lane_id; idx < TM * TK; idx += sg_size) {
                int i = idx / TK;
                int j = idx % TK;
                int row = sg_startx + i;
                if (row < M && j < 16) {
                    const block_q6_K * blk = &vx[row * A_blocks + sb];
                    int global_j = sb_tile * 32 + sub_half * 16 + j;
                    int ip      = global_j / 128;
                    int il_base = global_j % 128;

                    int     ql_idx, qh_idx;
                    int8_t  val;
                    if (il_base < 32) {
                        ql_idx = 64 * ip + il_base;
                        qh_idx = 32 * ip + il_base;
                        val = (int8_t)(((blk->ql[ql_idx] & 0xF) | (((blk->qh[qh_idx] >> 0) & 3) << 4)) - 32);
                    } else if (il_base < 64) {
                        ql_idx = 64 * ip + il_base;
                        qh_idx = 32 * ip + (il_base - 32);
                        val = (int8_t)(((blk->ql[ql_idx] & 0xF) | (((blk->qh[qh_idx] >> 2) & 3) << 4)) - 32);
                    } else if (il_base < 96) {
                        ql_idx = 64 * ip + (il_base - 64);
                        qh_idx = 32 * ip + (il_base - 64);
                        val = (int8_t)(((blk->ql[ql_idx] >> 4) | (((blk->qh[qh_idx] >> 4) & 3) << 4)) - 32);
                    } else {
                        ql_idx = 64 * ip + (il_base - 64);
                        qh_idx = 32 * ip + (il_base - 96);
                        val = (int8_t)(((blk->ql[ql_idx] >> 4) | (((blk->qh[qh_idx] >> 6) & 3) << 4)) - 32);
                    }
                    slm_A[i * TK + j] = val;
                } else {
                    slm_A[i * TK + j] = 0;
                }
            }

            // Stage B: load 16 q8_1 values for this half, zero-pad to TK=32
            for (int idx = lane_id; idx < TK * TN; idx += sg_size) {
                int k   = idx / TN;
                int n   = idx % TN;
                int col = sg_starty + n;
                if (col < N && k < 16) {
                    const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile];
                    slm_B[idx] = blk->qs[sub_half * 16 + k];
                } else {
                    slm_B[idx] = 0;
                }
            }
            sycl::group_barrier(item_ct1.get_group());

            joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
            joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

            auto pA = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
            auto pB = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B);

            joint_matrix_load(sg, matA, pA, TK);
            joint_matrix_load(sg, matB, pB, TN);

            joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
            joint_matrix_fill(sg, matC, 0);
            joint_matrix_mad(sg, matC, matA, matB, matC);

            auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
            joint_matrix_store(sg, matC, pC, TN, layout::row_major);
            sycl::group_barrier(item_ct1.get_group());

            for (int i = 0; i < TM; i++) {
                int row = sg_startx + i;
                if (row < M && sg_starty + lane_id < N) {
                    const block_q6_K * blk     = &vx[row * A_blocks + sb];
                    float              d       = (float) *(const sycl::half *) &blk->d;
                    float              scale_a = d * (float) blk->scales[scale_idx];
                    float              scale_b = (float) src1_q8[(sg_starty + lane_id) * B_blocks + k_tile].ds[0];
                    int32_t            val     = slm_tile[i * TN + lane_id];
                    acc[i] += scale_a * (float) val * scale_b;
                }
            }
            sycl::group_barrier(item_ct1.get_group());
        }
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_starty + lane_id) * ldc + (sg_startx + i)] = acc[i];
        }
    }
}

// Q2_K multi-subgroup XMX kernel with col_major B layout and split-MAD
// Q2_K has 16 sub-blocks of 16 elements (QK_K=256), affine: x = d*sc*q - dmin*m
// With TK=32, each k_tile covers 2 sub-blocks. Two MADs per k_tile.
template <int TM, int TN, int TK, int N_SG, int TILES_N>
static void mmq_q2_K_xmx_kernel(const block_q2_K * __restrict__ vx,
                                 const void * __restrict__ vy,
                                 float * __restrict__ dst,
                                 const int                K,
                                 const int                K_padded,
                                 const int                M,
                                 const int                N,
                                 const int                ldc,
                                 const sycl::nd_item<2> & item_ct1,
                                 int8_t *                 slm) {
    const auto sg = item_ct1.get_sub_group();

    const int local_linear = item_ct1.get_local_linear_id();
    const int sg_id        = local_linear / 16;
    const int lane_id      = local_linear % 16;

    const int base_m    = item_ct1.get_group(0) * (TM * N_SG);
    const int sg_startx = base_m + sg_id * TM;
    const int base_n    = item_ct1.get_group(1) * (TN * TILES_N);

    const int A_blocks = K / QK_K;
    const int B_blocks = K_padded / QK8_1;
    const int K_tiles  = K / TK;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t *  slm_B = slm;
    int8_t *  slm_A = slm + TILES_N * TN * TK + sg_id * (TM * TK);
    int32_t * slm_C = (int32_t *) (slm + TILES_N * TN * TK + N_SG * TM * TK) + sg_id * (TM * TN);

    float acc[TILES_N * TM];
    for (int i = 0; i < TILES_N * TM; i++) {
        acc[i] = 0.0f;
    }

    constexpr int total_threads = N_SG * 16;

    for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
        const int sb      = k_tile / 8;
        const int sb_tile = k_tile % 8;

        for (int sub_half = 0; sub_half < 2; sub_half++) {
            const int scale_idx = 2 * sb_tile + sub_half;

            // Load B: col_major layout, 16 active elements + 16 zeros
            constexpr int total_B_elems = TILES_N * TN * TK;
            for (int base_b = local_linear; base_b < total_B_elems; base_b += total_threads) {
                int col_idx = base_b / TK;
                int k       = base_b % TK;
                int nt      = col_idx / TN;
                int n       = col_idx % TN;
                int col     = base_n + nt * TN + n;
                if (col < N && k < 16) {
                    const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile];
                    slm_B[col_idx * TK + k] = blk->qs[sub_half * 16 + k];
                } else {
                    slm_B[col_idx * TK + k] = 0;
                }
            }

            // Load A: dequantize Q2_K 2-bit values
            for (int idx = lane_id; idx < TM * TK; idx += 16) {
                int i   = idx / TK;
                int j   = idx % TK;
                int row = sg_startx + i;
                if (row < M && j < 16) {
                    const block_q2_K * blk = &vx[row * A_blocks + sb];
                    int global_j  = sb_tile * 32 + sub_half * 16 + j;
                    int byte_idx  = global_j / 4;
                    int shift     = 2 * (global_j % 4);
                    slm_A[i * TK + j] = (int8_t)((blk->qs[byte_idx] >> shift) & 0x3);
                } else {
                    slm_A[i * TK + j] = 0;
                }
            }

            sycl::group_barrier(item_ct1.get_group());

            joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
            auto pA = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
            joint_matrix_load(sg, matA, pA, TK);

            for (int nt = 0; nt < TILES_N; nt++) {
                int8_t * slm_B_nt = slm_B + nt * TN * TK;

                joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matB;
                auto pB = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B_nt);
                joint_matrix_load(sg, matB, pB, TK);

                joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
                joint_matrix_fill(sg, matC, 0);
                joint_matrix_mad(sg, matC, matA, matB, matC);

                auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);
                joint_matrix_store(sg, matC, pC, TN, layout::row_major);
                sycl::group_barrier(sg);

                int n_base = base_n + nt * TN;
                for (int i = 0; i < TM; i++) {
                    int row = sg_startx + i;
                    if (row < M && n_base + lane_id < N) {
                        const block_q2_K * blk = &vx[row * A_blocks + sb];
                        float dall  = (float) blk->dm[0];
                        float dmin  = (float) blk->dm[1];
                        uint8_t sc_byte = blk->scales[scale_idx];
                        float sc    = (float)(sc_byte & 0xF);
                        float m_val = (float)(sc_byte >> 4);

                        const block_q8_1 & b_blk  = src1_q8[(n_base + lane_id) * B_blocks + k_tile];
                        float              scale_b = (float) b_blk.ds[0];
                        float              sum_b   = (float) b_blk.ds[1];

                        int32_t val = slm_C[i * TN + lane_id];
                        acc[nt * TM + i] += dall * sc * (float) val * scale_b - dmin * m_val * sum_b;
                    }
                }
            }

            sycl::group_barrier(item_ct1.get_group());
        }
    }

    for (int nt = 0; nt < TILES_N; nt++) {
        int n_base = base_n + nt * TN;
        for (int i = 0; i < TM; i++) {
            if (sg_startx + i < M && n_base + lane_id < N) {
                dst[(n_base + lane_id) * ldc + (sg_startx + i)] = acc[nt * TM + i];
            }
        }
    }
}

// Q3_K multi-subgroup XMX kernel with col_major B layout and split-MAD
// Q3_K has 16 sub-blocks of 16 elements (QK_K=256), pure scale: x = d*scale*q
// 3-bit values: 2 low bits from qs[], 1 high bit from hmask[], centered by -4
// With TK=32, each k_tile covers 2 sub-blocks. Two MADs per k_tile.

// Helper to extract 6-bit scale for Q3_K sub-block
static inline int8_t get_scale_q3_K_xmx(int isc, const uint8_t * __restrict__ scales) {
    int sc_low  = (scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
    int sc_high = ((scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
    return (int8_t)((sc_low | sc_high) - 32);
}

template <int TM, int TN, int TK, int N_SG, int TILES_N>
static void mmq_q3_K_xmx_kernel(const block_q3_K * __restrict__ vx,
                                 const void * __restrict__ vy,
                                 float * __restrict__ dst,
                                 const int                K,
                                 const int                K_padded,
                                 const int                M,
                                 const int                N,
                                 const int                ldc,
                                 const sycl::nd_item<2> & item_ct1,
                                 int8_t *                 slm) {
    const auto sg = item_ct1.get_sub_group();

    const int local_linear = item_ct1.get_local_linear_id();
    const int sg_id        = local_linear / 16;
    const int lane_id      = local_linear % 16;

    const int base_m    = item_ct1.get_group(0) * (TM * N_SG);
    const int sg_startx = base_m + sg_id * TM;
    const int base_n    = item_ct1.get_group(1) * (TN * TILES_N);

    const int A_blocks = K / QK_K;
    const int B_blocks = K_padded / QK8_1;
    const int K_tiles  = K / TK;

    const block_q8_1 * src1_q8 = (const block_q8_1 *) vy;

    int8_t *  slm_B = slm;
    int8_t *  slm_A = slm + TILES_N * TN * TK + sg_id * (TM * TK);
    int32_t * slm_C = (int32_t *) (slm + TILES_N * TN * TK + N_SG * TM * TK) + sg_id * (TM * TN);

    float acc[TILES_N * TM];
    for (int i = 0; i < TILES_N * TM; i++) {
        acc[i] = 0.0f;
    }

    constexpr int total_threads = N_SG * 16;

    for (int k_tile = 0; k_tile < K_tiles; k_tile++) {
        const int sb      = k_tile / 8;
        const int sb_tile = k_tile % 8;

        for (int sub_half = 0; sub_half < 2; sub_half++) {
            const int scale_idx = 2 * sb_tile + sub_half;

            // Load B: col_major layout, 16 active elements + 16 zeros
            constexpr int total_B_elems = TILES_N * TN * TK;
            for (int base_b = local_linear; base_b < total_B_elems; base_b += total_threads) {
                int col_idx = base_b / TK;
                int k       = base_b % TK;
                int nt      = col_idx / TN;
                int n       = col_idx % TN;
                int col     = base_n + nt * TN + n;
                if (col < N && k < 16) {
                    const block_q8_1 * blk = &src1_q8[col * B_blocks + k_tile];
                    slm_B[col_idx * TK + k] = blk->qs[sub_half * 16 + k];
                } else {
                    slm_B[col_idx * TK + k] = 0;
                }
            }

            // Load A: dequantize Q3_K 3-bit values (2 low + 1 high), centered by -4
            for (int idx = lane_id; idx < TM * TK; idx += 16) {
                int i   = idx / TK;
                int j   = idx % TK;
                int row = sg_startx + i;
                if (row < M && j < 16) {
                    const block_q3_K * blk = &vx[row * A_blocks + sb];
                    int global_j  = sb_tile * 32 + sub_half * 16 + j;
                    int byte_idx  = global_j / 4;
                    int shift     = 2 * (global_j % 4);
                    int q_low2    = (blk->qs[byte_idx] >> shift) & 0x3;
                    int q_high    = (blk->hmask[global_j / 8] >> (global_j % 8)) & 1;
                    slm_A[i * TK + j] = (int8_t)((q_low2 | (q_high << 2)) - 4);
                } else {
                    slm_A[i * TK + j] = 0;
                }
            }

            sycl::group_barrier(item_ct1.get_group());

            joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
            auto pA = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
            joint_matrix_load(sg, matA, pA, TK);

            for (int nt = 0; nt < TILES_N; nt++) {
                int8_t * slm_B_nt = slm_B + nt * TN * TK;

                joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matB;
                auto pB = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_B_nt);
                joint_matrix_load(sg, matB, pB, TK);

                joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
                joint_matrix_fill(sg, matC, 0);
                joint_matrix_mad(sg, matC, matA, matB, matC);

                auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_C);
                joint_matrix_store(sg, matC, pC, TN, layout::row_major);
                sycl::group_barrier(sg);

                int n_base = base_n + nt * TN;
                for (int i = 0; i < TM; i++) {
                    int row = sg_startx + i;
                    if (row < M && n_base + lane_id < N) {
                        const block_q3_K * blk = &vx[row * A_blocks + sb];
                        float d = (float) *(const sycl::half *) &blk->d;
                        int8_t scale = get_scale_q3_K_xmx(scale_idx, blk->scales);

                        const block_q8_1 & b_blk  = src1_q8[(n_base + lane_id) * B_blocks + k_tile];
                        float              scale_b = (float) b_blk.ds[0];

                        int32_t val = slm_C[i * TN + lane_id];
                        acc[nt * TM + i] += d * (float) scale * (float) val * scale_b;
                    }
                }
            }

            sycl::group_barrier(item_ct1.get_group());
        }
    }

    for (int nt = 0; nt < TILES_N; nt++) {
        int n_base = base_n + nt * TN;
        for (int i = 0; i < TM; i++) {
            if (sg_startx + i < M && n_base + lane_id < N) {
                dst[(n_base + lane_id) * ldc + (sg_startx + i)] = acc[nt * TM + i];
            }
        }
    }
}

void ggml_sycl_op_mul_mat_q_xmx_int8(ggml_backend_sycl_context & ctx,
                                     const ggml_tensor *         src0,
                                     const ggml_tensor *         src1,
                                     ggml_tensor *               dst,
                                     const char *                src0_dd_i,
                                     const float *               src1_ddf_i,
                                     const char *                src1_ddq_i,
                                     float *                     dst_dd_i,
                                     const int64_t               row_low,
                                     const int64_t               row_high,
                                     const int64_t               src1_ncols,
                                     const int64_t               src1_padded_row_size,
                                     const dpct::queue_ptr &     stream);

#endif
