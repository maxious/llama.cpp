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

#include "tsembd.hpp"

static void timestep_embedding_f32(
        const float * timesteps, float * dst, const int nb1,
        const int dim, const int max_period, const sycl::nd_item<3> &item_ct1) {
    // item_ct1.get_group(1)(blockIDx.y): idx of timesteps->ne[0]
    // item_ct1.get_group(2) (blockIDx.x): idx of ((dim + 1) / 2) / BLOCK_SIZE
    int i = item_ct1.get_group(1);
    int j = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    float * embed_data = (float *)((char *)dst +  i*nb1);

    int half = dim / 2;

    if (dim % 2 != 0 && j == half) {
        embed_data[2 * half] = 0.f;
    }

    if (j >= half) {
        return;
    }

    float timestep = timesteps[i];
    float freq = (float)sycl::native::exp(-(sycl::log((float)max_period)) * j / half);
    float arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

static void timestep_embedding_f32_sycl(
        const float * x, float * dst, const int ne00, const int nb1,
        const int dim, const int max_period, const queue_ptr& stream) {
    // As the kernel returns when thread.idx is larger than dim/2, the half_ceil does not need to pad
    int half_ceil = dim / 2;
    int num_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> block_dims(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    sycl::range<3> gridDim(1, ne00, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(
            gridDim * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            timestep_embedding_f32(
                x, dst, nb1, dim, max_period, item_ct1
            );
        });
}

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    const ggml_tensor *  src0   = dst->src[0];

    // Support F16, BF16, and F32
    const bool is_f16 = (src0->type == GGML_TYPE_F16);
    const bool is_bf16 = (src0->type == GGML_TYPE_BF16);
    const bool is_f32 = (src0->type == GGML_TYPE_F32);

#if defined(GGML_SYCL_F16) || defined(GGML_SYCL_BF16)
    GGML_ASSERT(is_f32 || is_f16 || is_bf16);
    GGML_ASSERT(dst->type == src0->type);
#else
    GGML_ASSERT(is_f32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif

    dpct::queue_ptr stream = ctx.stream();

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];

    // Allocate temporary F32 buffers for F16/BF16 input/output
    const size_t nbytes_src = ggml_nbytes(src0);
    const size_t nbytes_dst = ggml_nbytes(dst);
    const float* src0_d = nullptr;
    float* dst_d = nullptr;
    bool need_temp_buffers = false;

    if (is_f16 || is_bf16) {
        float * src0_f32 = (float *)sycl::malloc_device(nbytes_src, *stream);
        float * dst_f32 = (float *)sycl::malloc_device(nbytes_dst, *stream);
        need_temp_buffers = true;

        // Dequantize src0
        const int64_t n_elements = ggml_nelements(src0);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            if (is_f16) {
                const sycl::half * src = (const sycl::half *)src0->data;
                src0_f32[idx] = static_cast<float>(src[idx]);
            } else {
                const bfloat16 * src = (const bfloat16 *)src0->data;
                src0_f32[idx] = bf16_to_fp32(src[idx]);
            }
        }).wait();

        src0_d = src0_f32;
        dst_d = dst_f32;
    } else {
        src0_d = (const float *)src0->data;
        dst_d = (float *)dst->data;
    }

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);

    if (need_temp_buffers) {
        const int64_t n_elements = ggml_nelements(dst);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            float val = dst_d[idx];
            if (is_f16) {
                ((sycl::half *)dst->data)[idx] = static_cast<sycl::half>(val);
            } else {
                ((bfloat16 *)dst->data)[idx] = fp32_to_bf16(val);
            }
        }).wait();
        sycl::free(const_cast<float *>(src0_d), *stream);
        sycl::free(dst_d, *stream);
    }
}
