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

#include "mmq_xmx_int8.hpp"
#include "mmq.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// Reorder Q8_1 blocks into XMX-friendly VNNI layout
// src: block_q8_1[N][K/32]
// dst: int8_t[K/4][N*4] (VNNI-packed)
// scales: float[N][K/32] (Optional SoA for scales)
// TODO: Implement this function
void reorder_q8_1_xmx_layout(
    const block_q8_1 * __restrict__ src,
    int8_t * __restrict__ dst,
    int K, int N,
    const sycl::nd_item<1> &item_ct1) {

    (void)src;
    (void)dst;
    (void)K;
    (void)N;
    (void)item_ct1;
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
    const dpct::queue_ptr& stream) {

    static bool first_call = true;
    if (first_call) {
        fprintf(stderr, "ggml_sycl: Using INT8 XMX MMQ kernel for src0->type=%d\n", (int)src0->type);
        first_call = false;
    }

    // Check if device supports INT8 XMX
    if (!has_int8_xmx_support(stream)) {
        // Fall back to regular MMQ if INT8 XMX is not supported
        ggml_sycl_op_mul_mat_q(ctx, src0, src1, dst,
                                src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                                row_low, row_high, src1_ncols, src1_padded_row_size,
                                stream);
        return;
    }

    // Get optimal tile configuration for this device
    auto tile_config = get_int8_xmx_tile_config(stream);
    const int TM = tile_config.TM;
    const int TN = tile_config.TN;
    const int TK = tile_config.TK;

    // Get matrix dimensions
    const int64_t ne00 = src0->ne[0];
    const int64_t ne11 = src1->ne[1];

    // Calculate grid dimensions
    const int64_t M = row_high - row_low;
    const int64_t N = ne11;
    const int64_t K = ne00;

    const int64_t nblocks_m = (M + TM - 1) / TM;
    const int64_t nblocks_n = (N + TN - 1) / TN;

    const int sg_size = 16;

    // Launch kernel based on quantization type
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler& cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(8 * 16), cgh);
                    cgh.parallel_for(
                        nd_range<2>({static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size)},
                                    {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q8_0_xmx_kernel<8, 16, 32>(
                                (const block_q8_0*)src0_dd_i,
                                (const block_q8_1*)src1_ddq_i,
                                dst_dd_i,
                                K, M, N,
                                item_ct1,
                                slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q4_0:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler& cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(8 * 16), cgh);
                    cgh.parallel_for(
                        nd_range<2>({static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size)},
                                    {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q4_0_xmx_kernel<8, 16, 32>(
                                (const block_q4_0*)src0_dd_i,
                                (const block_q8_1*)src1_ddq_i,
                                dst_dd_i,
                                K, M, N,
                                item_ct1,
                                slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q4_1:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler& cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(8 * 16), cgh);
                    cgh.parallel_for(
                        nd_range<2>({static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size)},
                                    {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q4_1_xmx_kernel<8, 16, 32>(
                                (const block_q4_1*)src0_dd_i,
                                (const block_q8_1*)src1_ddq_i,
                                dst_dd_i,
                                K, M, N,
                                item_ct1,
                                slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q5_0:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler& cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(8 * 16), cgh);
                    cgh.parallel_for(
                        nd_range<2>({static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size)},
                                    {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q5_0_xmx_kernel<8, 16, 32>(
                                (const block_q5_0*)src0_dd_i,
                                (const block_q8_1*)src1_ddq_i,
                                dst_dd_i,
                                K, M, N,
                                item_ct1,
                                slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q5_1:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler& cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(8 * 16), cgh);
                    cgh.parallel_for(
                        nd_range<2>({static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size)},
                                    {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q5_1_xmx_kernel<8, 16, 32>(
                                (const block_q5_1*)src0_dd_i,
                                (const block_q8_1*)src1_ddq_i,
                                dst_dd_i,
                                K, M, N,
                                item_ct1,
                                slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q8_1:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler& cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(8 * 16), cgh);
                    cgh.parallel_for(
                        nd_range<2>({static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size)},
                                    {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q8_1_xmx_kernel<8, 16, 32>(
                                (const block_q8_1*)src0_dd_i,
                                (const block_q8_1*)src1_ddq_i,
                                dst_dd_i,
                                K, M, N,
                                item_ct1,
                                slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        default:
            // Fall back to regular MMQ for unsupported quantization types
            ggml_sycl_op_mul_mat_q(ctx, src0, src1, dst,
                                    src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                                    row_low, row_high, src1_ncols, src1_padded_row_size,
                                    stream);
            break;
    }
}