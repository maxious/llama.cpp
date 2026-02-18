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

template <int TM, int TN, int TK>
static void mmq_q8_0_xmx_kernel(const block_q8_0 * __restrict__ vx,
                                const void * __restrict__ vy,
                                float * __restrict__ dst,
                                const int                K,
                                const int                K_padded,
                                const int                M,
                                const int                N,
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];

    // K_blocks is based on padded K for scale array stride
    const int           K_blocks = K_padded / 32;
    const sycl::half2 * ds_ptr   = (const sycl::half2 *) vy;
    const int8_t *      qs_ptr   = (const int8_t *) (ds_ptr + N * K_blocks);

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        const int8_t * pA_raw = (const int8_t *) &vx[sg_startx * K_blocks + k_tile].qs;
        auto           pA =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pA_raw);

        // Row-major layout: B is stored as [K][N], stride = N
        // Offset: (k_tile * TK) * N + (sg_starty / sg_size) * TN
        const int      sg_size = sg.get_max_local_range()[0];
        const int8_t * pB_raw  = qs_ptr + (k_tile * TK) * N + (sg_starty / sg_size) * TN;
        auto           pB =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);

        joint_matrix_load(sg, matA, pA, K_blocks * 34);
        joint_matrix_load(sg, matB, pB, N);  // stride = N for row_major

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            // scale_a uses K_blocks from src0 (already correct)
            float   scale_a = (float) vx[(sg_startx + i) * (K / 32) + k_tile].d;
            int32_t val     = slm_tile[i * TN + lane_id];
            // Only accumulate if this lane is within bounds
            if (sg_starty + lane_id < N) {
                // scale_b uses K_blocks from padded src1
                float scale_b = (float) ds_ptr[(sg_starty + lane_id) * K_blocks + k_tile][0];
                acc[i] += (float) val * scale_a * scale_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_startx + i) * N + sg_starty + lane_id] = acc[i];
        }
    }
}

template <int TM, int TN, int TK>
static void mmq_q4_0_xmx_kernel(const block_q4_0 * __restrict__ vx,
                                const void * __restrict__ vy,
                                float * __restrict__ dst,
                                const int                K,
                                const int                K_padded,
                                const int                M,
                                const int                N,
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int           K_blocks = K_padded / 32;
    const sycl::half2 * ds_ptr   = (const sycl::half2 *) vy;
    const int8_t *      qs_ptr   = (const int8_t *) (ds_ptr + N * K_blocks);

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
#pragma unroll
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += 16) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q4_0 * block  = &vx[row * (K / QK4_0) + k_tile * (TK / QK4_0)];
                uint8_t            qs_val = block->qs[j];
                slm_A[idx * 2 + 0]        = (int8_t) ((qs_val >> 0) & 0x0F) - 8;
                slm_A[idx * 2 + 1]        = (int8_t) ((qs_val >> 4) & 0x0F) - 8;
            } else {
                slm_A[idx * 2 + 0] = 0;
                slm_A[idx * 2 + 1] = 0;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        joint_matrix_load(sg, matA, pA, TK);

        // Row-major layout: B is stored as [K][N], stride = N
        const int8_t * pB_raw = qs_ptr + (k_tile * TK) * N + (sg_starty / sg_size) * TN;
        auto           pB =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);
        joint_matrix_load(sg, matB, pB, N);  // stride = N for row_major

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M) {
                const block_q4_0 * block   = &vx[row * (K / QK4_0) + k_tile * (TK / QK4_0)];
                float              scale_a = (float) block->d;
                float   scale_b = (float) ds_ptr[(sg_starty + lane_id) * K_blocks + k_tile * (TK / QK4_0)][0];
                int32_t val     = slm_tile[i * TN + lane_id];
                if (sg_starty + lane_id < N) {
                    acc[i] += (float) val * scale_a * scale_b;
                }
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_startx + i) * N + sg_starty + lane_id] = acc[i];
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
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int           K_blocks = K_padded / 32;
    const sycl::half2 * ds_ptr   = (const sycl::half2 *) vy;
    const int8_t *      qs_ptr   = (const int8_t *) (ds_ptr + N * K_blocks);

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
#pragma unroll
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += 16) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q4_1 * block  = &vx[row * (K / QK4_0) + k_tile * (TK / QK4_1)];
                uint8_t            qs_val = block->qs[j];
                slm_A[idx * 2 + 0]        = (int8_t) ((qs_val >> 0) & 0x0F);
                slm_A[idx * 2 + 1]        = (int8_t) ((qs_val >> 4) & 0x0F);
            } else {
                slm_A[idx * 2 + 0] = 0;
                slm_A[idx * 2 + 1] = 0;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        joint_matrix_load(sg, matA, pA, TK);

        // Row-major layout: B is stored as [K][N], stride = N
        const int8_t * pB_raw = qs_ptr + (k_tile * TK) * N + (sg_starty / sg_size) * TN;
        auto           pB =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);
        joint_matrix_load(sg, matB, pB, N);  // stride = N for row_major

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M) {
                const block_q4_1 * block   = &vx[row * (K / QK4_0) + k_tile * (TK / QK4_1)];
                const sycl::half2  dm      = block->dm;
                float              d       = (float) dm[0];
                float              m       = (float) dm[1];
                sycl::half2        ds8     = ds_ptr[(sg_starty + lane_id) * K_blocks + k_tile * (TK / QK4_1)];
                float              scale_b = (float) ds8[0];
                float              s_b     = (float) ds8[1];
                int32_t            val     = slm_tile[i * TN + lane_id];
                acc[i] += d * (float) val * scale_b + m * s_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_startx + i) * N + sg_starty + lane_id] = acc[i];
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
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int           K_blocks = K_padded / 32;
    const sycl::half2 * ds_ptr   = (const sycl::half2 *) vy;
    const int8_t *      qs_ptr   = (const int8_t *) (ds_ptr + N * K_blocks);

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
#pragma unroll
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += 16) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q5_0 * block = &vx[row * (K / QK5_0) + k_tile * (TK / QK5_0)];
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
        sycl::group_barrier(item_ct1.get_group());

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        joint_matrix_load(sg, matA, pA, TK);

        // Row-major layout: B is stored as [K][N], stride = N
        const int8_t * pB_raw = qs_ptr + (k_tile * TK) * N + (sg_starty / sg_size) * TN;
        auto           pB =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);
        joint_matrix_load(sg, matB, pB, N);  // stride = N for row_major

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M) {
                const block_q5_0 * block   = &vx[row * (K / QK5_0) + k_tile * (TK / QK5_0)];
                float              scale_a = (float) block->d;
                float   scale_b = (float) ds_ptr[(sg_starty + lane_id) * K_blocks + k_tile * (TK / QK5_0)][0];
                int32_t val     = slm_tile[i * TN + lane_id];
                if (sg_starty + lane_id < N) {
                    acc[i] += (float) val * scale_a * scale_b;
                }
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_startx + i) * N + sg_starty + lane_id] = acc[i];
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
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int           K_blocks = K_padded / 32;
    const sycl::half2 * ds_ptr   = (const sycl::half2 *) vy;
    const int8_t *      qs_ptr   = (const int8_t *) (ds_ptr + N * K_blocks);

    int8_t * slm_A = (int8_t *) (slm_tile + TM * TN);

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
#pragma unroll
        for (int idx = lane_id; idx < (TM * TK) / 2; idx += 16) {
            int i   = (idx * 2) / TK;
            int j   = (idx * 2) % TK / 2;
            int row = sg_startx + i;
            if (row < M) {
                const block_q5_1 * block = &vx[row * (K / QK4_0) + k_tile * (TK / QK5_1)];
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
        sycl::group_barrier(item_ct1.get_group());

        auto pA =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_A);
        joint_matrix_load(sg, matA, pA, TK);

        // Row-major layout: B is stored as [K][N], stride = N
        const int8_t * pB_raw = qs_ptr + (k_tile * TK) * N + (sg_starty / sg_size) * TN;
        auto           pB =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);
        joint_matrix_load(sg, matB, pB, N);  // stride = N for row_major

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            int row = sg_startx + i;
            if (row < M) {
                const block_q5_1 * block   = &vx[row * (K / QK4_0) + k_tile * (TK / QK5_1)];
                const sycl::half2  dm      = block->dm;
                float              d       = (float) dm[0];
                float              m       = (float) dm[1];
                sycl::half2        ds8     = ds_ptr[(sg_starty + lane_id) * K_blocks + k_tile * (TK / QK5_1)];
                float              scale_b = (float) ds8[0];
                float              s_b     = (float) ds8[1];
                int32_t            val     = slm_tile[i * TN + lane_id];
                acc[i] += d * (float) val * scale_b + m * s_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_startx + i) * N + sg_starty + lane_id] = acc[i];
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
                                const sycl::nd_item<2> & item_ct1,
                                int32_t *                slm_tile) {
    const auto sg = item_ct1.get_sub_group();

    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::row_major> matB;

    float acc[TM];
    for (int i = 0; i < TM; i++) {
        acc[i] = 0.0f;
    }

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id   = sg.get_local_id()[0];
    const int sg_size   = sg.get_max_local_range()[0];

    const int           K_blocks = K_padded / 32;
    const sycl::half2 * ds_ptr   = (const sycl::half2 *) vy;
    const int8_t *      qs_ptr   = (const int8_t *) (ds_ptr + N * K_blocks);

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        const int8_t * pA_raw = (const int8_t *) &vx[sg_startx * K_blocks + k_tile].qs;
        auto           pA =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pA_raw);

        // Row-major layout: B is stored as [K][N], stride = N
        const int8_t * pB_raw = qs_ptr + (k_tile * TK) * N + (sg_starty / sg_size) * TN;
        auto           pB =
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);

        joint_matrix_load(sg, matA, pA, K_blocks * 36);
        joint_matrix_load(sg, matB, pB, N);  // stride = N for row_major

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC =
            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            const sycl::half2 ds      = vx[(sg_startx + i) * (K / 32) + k_tile].ds;
            float             scale_a = (float) ds[0];
            float             scale_b = (float) ds_ptr[(sg_starty + lane_id) * K_blocks + k_tile][0];
            int32_t           val     = slm_tile[i * TN + lane_id];
            if (sg_starty + lane_id < N) {
                acc[i] += (float) val * scale_a * scale_b;
            }
        }
        sycl::group_barrier(item_ct1.get_group());
    }

    for (int i = 0; i < TM; i++) {
        if (sg_startx + i < M && sg_starty + lane_id < N) {
            dst[(sg_startx + i) * N + sg_starty + lane_id] = acc[i];
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
