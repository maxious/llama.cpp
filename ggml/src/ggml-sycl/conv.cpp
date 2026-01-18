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

#include "conv.hpp"

static  void conv_transpose_1d_kernel(
        const int s0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2,
        const int src1_ne0, const int dst_ne0,
        const float * src0, const float * src1,  float * dst,
        const sycl::nd_item<3> &item_ct1) {
    int global_index = item_ct1.get_local_id(2) +
                       item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if (global_index >= output_size) {
        return;
    }

    int out_index = global_index / dst_ne0;

    float accumulator = 0;

    for (int c = 0; c < src0_ne2; c++) {
        int idx = global_index % dst_ne0;

        int kernel_offset = (src0_ne0 * src0_ne1 * c) + (out_index * src0_ne0);
        int input_offset = src1_ne0 * c;

        for (int i = 0; i < src1_ne0; i++) {
            if (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                continue;
            }
            int weight_idx = idx - i*s0;

            float kernel_weight = src0[kernel_offset + weight_idx];
            float input_value =  src1[input_offset+i];

            accumulator += kernel_weight * input_value;
        }
    }
    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_sycl(
    const int s0, const int output_size,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int dst_ne0,
    const float *src0, const float *src1, float *dst,
    const queue_ptr& stream) {

    const int num_blocks = (output_size + SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE - 1) / SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_size,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

void ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    // Support F16, BF16, and F32
    const bool is_f16 = (src0->type == GGML_TYPE_F16);
    const bool is_bf16 = (src0->type == GGML_TYPE_BF16);
    const bool is_f32 = (src0->type == GGML_TYPE_F32);

#if defined(GGML_SYCL_F16) || defined(GGML_SYCL_BF16)
    GGML_ASSERT(is_f32 || is_f16 || is_bf16);
    GGML_ASSERT(src1->type == src0->type);
    GGML_ASSERT(dst->type == src0->type);
#else
    GGML_ASSERT(is_f32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    float * dst_d = (float *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    const int32_t * opts = (const int32_t *)dst->op_params;
    const int s0 = opts[0];
    const int64_t output_size = ggml_nelements(dst);

    // Allocate temporary F32 buffers for F16/BF16 input/output
    const size_t nbytes0 = ggml_nbytes(src0);
    const size_t nbytes1 = ggml_nbytes(src1);
    float * src0_f32 = nullptr;
    float * src1_f32 = nullptr;
    float * dst_f32 = nullptr;
    bool need_temp_buffers = false;

    if (is_f16 || is_bf16) {
        src0_f32 = (float *)sycl::malloc_device(nbytes0, *stream);
        src1_f32 = (float *)sycl::malloc_device(nbytes1, *stream);
        dst_f32 = (float *)sycl::malloc_device(nbytes0, *stream);
        need_temp_buffers = true;

        // Dequantize src0
        const int64_t n_elements0 = ggml_nelements(src0);
        stream->parallel_for(sycl::range<1>(n_elements0), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            if (is_f16) {
                const sycl::half * src = (const sycl::half *)src0->data;
                src0_f32[idx] = static_cast<float>(src[idx]);
            } else {
                const bfloat16 * src = (const bfloat16 *)src0->data;
                src0_f32[idx] = bf16_to_fp32(src[idx]);
            }
        });

        // Dequantize src1
        const int64_t n_elements1 = ggml_nelements(src1);
        stream->parallel_for(sycl::range<1>(n_elements1), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            if (is_f16) {
                const sycl::half * src = (const sycl::half *)src1->data;
                src1_f32[idx] = static_cast<float>(src[idx]);
            } else {
                const bfloat16 * src = (const bfloat16 *)src1->data;
                src1_f32[idx] = bf16_to_fp32(src[idx]);
            }
        });
    }

    const float * src0_d = is_f32 ? (const float *)src0->data : src0_f32;
    const float * src1_d = is_f32 ? (const float *)src1->data : src1_f32;
    float * dst_ptr = is_f32 ? dst_d : dst_f32;

    conv_transpose_1d_f32_f32_sycl(s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_ptr, stream);

    if (need_temp_buffers) {
        const int64_t n_elements = ggml_nelements(dst);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            float val = dst_f32[idx];
            if (is_f16) {
                ((sycl::half *)dst->data)[idx] = static_cast<sycl::half>(val);
            } else {
                ((bfloat16 *)dst->data)[idx] = fp32_to_bf16(val);
            }
        });
        sycl::free(src0_f32, *stream);
        sycl::free(src1_f32, *stream);
        sycl::free(dst_f32, *stream);
    }
}

