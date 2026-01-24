#ifndef GGML_SYCL_MMQ_XMX_INT8_HPP
#define GGML_SYCL_MMQ_XMX_INT8_HPP

#include "common.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

static inline bool has_int8_xmx_support(const dpct::queue_ptr &q) {
    auto dev = q->get_device();
    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }

    auto combinations = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    for (auto &comb : combinations) {
        bool has_int8_a = (comb.atype == matrix_type::sint8 ||
                          comb.atype == matrix_type::uint8);
        bool has_int8_b = (comb.btype == matrix_type::sint8 ||
                          comb.btype == matrix_type::uint8);

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

static inline xmx_int8_tile_config get_int8_xmx_tile_config(const dpct::queue_ptr &q) {
    auto dev = q->get_device();
    auto combinations = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    for (auto &comb : combinations) {
        bool has_int8_a = (comb.atype == matrix_type::sint8 ||
                          comb.atype == matrix_type::uint8);
        bool has_int8_b = (comb.btype == matrix_type::sint8 ||
                          comb.btype == matrix_type::uint8);

        if (has_int8_a && has_int8_b) {
            if (comb.nsize == 16) {
                return {8, 16, 32};
            } else if (comb.nsize == 8) {
                return {8, 8, 32};
            } else if (comb.nsize == 0) {
                return {16, 16, 64};
            }
        }
    }

    return {8, 16, 32};
}

template <int TM, int TN, int TK>
static void mmq_q8_0_xmx_kernel(
    const block_q8_0 * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int K, const int M, const int N,
    const sycl::nd_item<2> &item_ct1,
    int32_t * slm_tile) {

    const auto sg = item_ct1.get_sub_group();

    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::ext_intel_packed> matB;

    float acc[TM]; 
    for (int i = 0; i < TM; i++) acc[i] = 0.0f;

    const int sg_startx = item_ct1.get_group(0) * TM;
    const int sg_starty = item_ct1.get_group(1) * TN;
    const int lane_id = sg.get_local_id()[0];

    const int K_blocks = K / 32;
    const sycl::half2 * ds_ptr = (const sycl::half2 *) vy;
    const int8_t * qs_ptr = (const int8_t *) (ds_ptr + N * K_blocks);

    for (int k_tile = 0; k_tile < K / TK; k_tile++) {
        const int8_t* pA_raw = (const int8_t*)&vx[sg_startx * K_blocks + k_tile].qs;
        auto pA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pA_raw);

        const int8_t* pB_raw = qs_ptr + (k_tile * TK / 4) * (N * 4) + sg_starty * 4;
        auto pB = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);

        joint_matrix_load(sg, matA, pA, K_blocks * 34);
        joint_matrix_load(sg, matB, pB, N * 4);

        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;
        joint_matrix_fill(sg, matC, 0);
        joint_matrix_mad(sg, matC, matA, matB, matC);

        auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(slm_tile);
        joint_matrix_store(sg, matC, pC, TN, layout::row_major);
        sycl::group_barrier(item_ct1.get_group());

        for (int i = 0; i < TM; i++) {
            float scale_a = (float)vx[(sg_startx + i) * K_blocks + k_tile].d;
            float scale_b = (float)ds_ptr[(sg_starty + lane_id) * K_blocks + k_tile][0];
            int32_t val = slm_tile[i * TN + lane_id];
            acc[i] += (float)val * scale_a * scale_b;
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
static void mmq_q4_K_xmx_kernel(
    const block_q4_K * __restrict__ vx,
    const void * __restrict__ vy,
    float * __restrict__ dst,
    const int K, const int M, const int N,
    const sycl::nd_item<2> &item_ct1,
    int32_t * slm_tile) {
}

void ggml_sycl_op_mul_mat_q_xmx_int8(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst,
    const char* src0_dd_i,
    const float* src1_ddf_i,
    const char* src1_ddq_i,
    float* dst_dd_i,
    const int64_t row_low,
    const int64_t row_high,
    const int64_t src1_ncols,
    const int64_t src1_padded_row_size,
    const dpct::queue_ptr& stream);

#endif
