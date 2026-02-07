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

#include "ggml-impl.h"
#include "common.hpp"
#include "dequantize.hpp"
#include "getrows.hpp"


template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    2;
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_row, ib, iqs, v);

    dst_row[iybs + iqs + 0] = v.x();
    dst_row[iybs + iqs + y_offset] = v.y();
}

template<typename src0_t, typename dst_t>
static void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            const sycl::nd_item<3> &item_ct1/*, size_t s13*/) {

    const int i00 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int i10 = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    const int i11 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) /
                    ne12;
    const int i12 = (item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                     item_ct1.get_local_id(0)) %
                    ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = src0_row[i00];
}

template <int qk, int qr, dequantize_kernel_t dq>
static void get_rows_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst, const void *src0_dd,
                          const int32_t *src1_dd, float *dst_dd,
                          queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + 2*SYCL_GET_ROWS_BLOCK_SIZE - 1) / (2*SYCL_GET_ROWS_BLOCK_SIZE);
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             k_get_rows<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
                         });

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

template <typename src0_t>
static void get_rows_sycl_float(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, queue_ptr stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const sycl::range<3> block_dims(1, 1, SYCL_GET_ROWS_BLOCK_SIZE);
    const int block_num_x = (ne00 + SYCL_GET_ROWS_BLOCK_SIZE - 1) / SYCL_GET_ROWS_BLOCK_SIZE;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                k_get_rows_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, item_ct1);
            });
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(ctx);
}

// K-quant kernels
static void k_get_rows_q4_k(
    const void * src0, const int32_t * src1, float * dst,
    int64_t ne00, int64_t ne12,
    size_t s1, size_t s2, size_t s3,
    size_t nb01, size_t nb02, size_t nb03,
    size_t s10, size_t s11, size_t s12,
    uint8_t * scales_local,
    const sycl::nd_item<3> &item) {

    const int i10 = item.get_group(1);
    const int i11 = item.get_group(0) / ne12;
    const int i12 = item.get_group(0) % ne12;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    float * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    dequantize_block_q4_K(src0_row, dst_row, scales_local, item);
}

static void k_get_rows_q6_k(
    const void * src0, const int32_t * src1, float * dst,
    int64_t ne00, int64_t ne12,
    size_t s1, size_t s2, size_t s3,
    size_t nb01, size_t nb02, size_t nb03,
    size_t s10, size_t s11, size_t s12,
    const sycl::nd_item<3> &item) {

    const int i10 = item.get_group(1);
    const int i11 = item.get_group(0) / ne12;
    const int i12 = item.get_group(0) % ne12;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    float * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    dequantize_block_q6_K<float>(src0_row, dst_row, item);
}

static void k_get_rows_mxfp4(
    const void * src0, const int32_t * src1, float * dst,
    int64_t ne00, int64_t ne12,
    size_t s1, size_t s2, size_t s3,
    size_t nb01, size_t nb02, size_t nb03,
    size_t s10, size_t s11, size_t s12,
    const sycl::nd_item<3> &item) {

    const int i10 = item.get_group(1);
    const int i11 = item.get_group(0) / ne12;
    const int i12 = item.get_group(0) % ne12;

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    float * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    dequantize_block_mxfp4(src0_row, dst_row, item);
}

static void get_rows_sycl_q4_k(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const void *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, queue_ptr stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // Q4_K needs 32 threads per block
    const sycl::range<3> block_dims(1, 1, 32);
    const int block_num_x = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    // strides in elements
    const size_t s1 = nb1 / sizeof(float);
    const size_t s2 = nb2 / sizeof(float);
    const size_t s3 = nb3 / sizeof(float);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint8_t, 1> scales_local(sycl::range<1>(K_SCALE_SIZE), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                k_get_rows_q4_k(
                    src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                    s3, nb01, nb02, nb03, s10, s11, s12, get_pointer(scales_local), item);
            });
    });
}

static void get_rows_sycl_q6_k(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const void *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, queue_ptr stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // Q6_K needs 64 threads per block (QK_K=256, 4 values per thread?)
    // dequantize_block_q6_K logic assumes 64 threads for QK_K=256
    const sycl::range<3> block_dims(1, 1, 64);
    const int block_num_x = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / sizeof(float);
    const size_t s2 = nb2 / sizeof(float);
    const size_t s3 = nb3 / sizeof(float);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                k_get_rows_q6_k(
                    src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                    s3, nb01, nb02, nb03, s10, s11, s12, item);
            });
    });
}

static void get_rows_sycl_mxfp4(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                const ggml_tensor *src1, ggml_tensor *dst,
                                const void *src0_dd, const int32_t *src1_dd,
                                float *dst_dd, queue_ptr stream) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // MXFP4 needs 32 threads per block
    const sycl::range<3> block_dims(1, 1, 32);
    const int block_num_x = ne00 / QK_K;
    const sycl::range<3> block_nums(ne11 * ne12, ne10, block_num_x);

    const size_t s1 = nb1 / sizeof(float);
    const size_t s2 = nb2 / sizeof(float);
    const size_t s3 = nb3 / sizeof(float);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
                k_get_rows_mxfp4(
                    src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                    s3, nb01, nb02, nb03, s10, s11, s12, item);
            });
    });
}

void ggml_sycl_op_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_size(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_size(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) dst->src[1]->data;
    /* TODO: Refactor and remove duplicates */
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::half *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_BF16:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const sycl::ext::oneapi::bfloat16 *)dst->src[0]->data,
                                src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_F32:
            get_rows_sycl_float(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_0:
            get_rows_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_1:
            get_rows_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_0:
            get_rows_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q5_1:
            get_rows_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q8_0:
            get_rows_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (const float *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q4_K:
            get_rows_sycl_q4_k(ctx, dst->src[0], dst->src[1], dst, (const void *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_Q6_K:
            get_rows_sycl_q6_k(ctx, dst->src[0], dst->src[1], dst, (const void *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        case GGML_TYPE_MXFP4:
            get_rows_sycl_mxfp4(ctx, dst->src[0], dst->src[1], dst, (const void *)dst->src[0]->data,
            src1_i32, (float *)dst->data, ctx.stream());
            break;
        default:
            // TODO: other k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}
