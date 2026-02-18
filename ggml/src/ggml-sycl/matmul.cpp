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

#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-sycl.h"
#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/gemm_bf16_f32_tiled.hpp"
#include "ggml-sycl/gemm_f16_f32_tiled.hpp"
#include "ggml-sycl/gemm_mxfp4_f32_tiled.hpp"
#include "ggml-sycl/gemm_q2_K_f32_tiled.hpp"
#include "ggml-sycl/gemm_q3_K_f32_tiled.hpp"
#include "ggml-sycl/gemm_q4_0_f32_tiled.hpp"
#include "ggml-sycl/gemm_q4_K_f32_tiled.hpp"
#include "ggml-sycl/gemm_q5_K_f32_tiled.hpp"
#include "ggml-sycl/gemm_q6_K_f32_tiled.hpp"
#include "ggml-sycl/gemm_q8_0_f32_tiled.hpp"
#include "ggml-sycl/gemm_tiled.hpp"
#include "ggml-sycl/gemm_xmx.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/itt_annotations.hpp"
#include "ggml-sycl/mmq_xmx_int8.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml.h"
#include "sycl_buffer.hpp"
#include "sycl_compute.hpp"
#include "sycl_defs.hpp"
#include "sycl_matmul.hpp"

#include <sycl/half_type.hpp>

template <template <int> typename quantize_f>
static void ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx,
                                 const ggml_tensor *         src0,
                                 const ggml_tensor *         src1,
                                 ggml_tensor *               dst,
                                 ggml_sycl_op_mul_mat_t      op) try {
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_ts = ggml_type_size(src0->type);
    const size_t src0_bs = ggml_blck_size(src0->type);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * src1_extra = (ggml_tensor_extra_gpu *) src1->extra;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const bool split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
    if (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    struct dev_data {
        ggml_sycl_pool_alloc<char>  src0_dd_alloc;
        ggml_sycl_pool_alloc<float> src1_ddf_alloc;
        ggml_sycl_pool_alloc<char>  src1_ddq_alloc;
        ggml_sycl_pool_alloc<float> dst_dd_alloc;

        char *  src0_dd  = nullptr;
        float * src1_ddf = nullptr;  // float
        char *  src1_ddq = nullptr;  // q8_1
        float * dst_dd   = nullptr;

        int64_t row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    int       used_devices = 0;
    queue_ptr main_stream  = ctx.stream();

    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        // by default, use all rows
        dev[i].row_low  = 0;
        dev[i].row_high = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        if (split) {
            const int64_t rounding = get_row_rounding(src0->type, tensor_split);

            if (i != 0) {
                dev[i].row_low = ne01 * tensor_split[i];
                if (dev[i].row_low < ne01) {
                    dev[i].row_low -= dev[i].row_low % rounding;
                }
            }

            if (i != ggml_sycl_info().device_count - 1) {
                dev[i].row_high = ne01 * tensor_split[i + 1];
                if (dev[i].row_high < ne01) {
                    dev[i].row_high -= dev[i].row_high % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled =
        !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>, no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = i == ctx.device;
        const bool dst_on_device  = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        if (src0_is_contiguous) {
            dev[i].src0_dd = (char *) src0->data;
        } else {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (float *) src1->data;
        } else {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        if constexpr (quantize_enabled) {
            dev[i].src1_ddq =
                dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1 * src1_padded_col_size * q8_1_ts / q8_1_bs);

            if (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                try {
                    // kx = row length (ne10 = K dimension), ky = number of rows (nrows1)
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1,
                                                       src1_padded_col_size, stream);
                } catch (const sycl::exception & exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        if (dst_on_device) {
            dev[i].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = split ? (dev[i].row_high - dev[i].row_low) * ne1 : ggml_nelements(dst);
            dev[i].dst_dd             = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    GGML_SYCL_DEBUG(
        "[SYCL][MUL_MAT] split=%d, device_count=%d, used_devices=%d, ne11=%ld, is_max=%d, MUL_MAT_SRC1_COL_STRIDE=%d\n",
        split, ggml_sycl_info().device_count, used_devices, ne11,
        (int) ((ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE), (int) MUL_MAT_SRC1_COL_STRIDE);

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    if (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        GGML_SYCL_DEBUG("[SYCL][MUL_MAT] Recording MAIN device event: dev=%d, events[%d][0]=%p, ne11=%ld, is_max=%d\n",
                        ctx.device, ctx.device, (void *) (src0_extra->events[ctx.device][0]), ne11,
                        (int) ((ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE));
        SYCL_CHECK(CHECK_TRY_ERROR(*src0_extra->events[ctx.device][0] = ctx.stream()->ext_oneapi_submit_barrier()));
    }

    const int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is         = split ? (src1_col_0 / src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if ((!split && i != ctx.device) || dev[i].row_low == dev[i].row_high) {
                continue;
            }

            const bool    src1_on_device = i == ctx.device;
            const bool    dst_on_device  = i == ctx.device;
            const int64_t row_diff       = dev[i].row_high - dev[i].row_low;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            if (split && (i != ctx.device || is != 0)) {
                GGML_SYCL_DEBUG("[SYCL][MUL_MAT]DEVICE %d (stream %d) WAITING on main event dev=%d events[%d][0]=%p\n",
                                i, (int) is, ctx.device, ctx.device, (void *) (src0_extra->events[ctx.device][0]));
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier({ *src0_extra->events[ctx.device][0] })));
            }

            for (int64_t i0 = 0; i0 < ne13 * ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                const size_t src1_ddq_i_offset = (i0 * ne11 + src1_col_0) * src1_padded_col_size * q8_1_ts / q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                char *  src0_dd_i = dev[i].src0_dd + (i0 / i02_divisor) * (ne01 * ne00 * src0_ts) / src0_bs;
                float * src1_ddf_i;
                if (src1->type == GGML_TYPE_F16) {
                    src1_ddf_i =
                        (float *) ((char *) dev[i].src1_ddf + (i0 * ne11 + src1_col_0) * ne10 * sizeof(sycl::half));
                } else {
                    src1_ddf_i = dev[i].src1_ddf + (i0 * ne11 + src1_col_0) * ne10;
                }
                char *  src1_ddq_i = dev[i].src1_ddq + src1_ddq_i_offset;
                float * dst_dd_i   = dev[i].dst_dd + (i0 * ne1 + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (i == ctx.device) {
                    dst_dd_i += dev[i].row_low;  // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (i != ctx.device) {
                        if constexpr (quantize_enabled) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } else {
                            float * src1_ddf_i_source;
                            if (src1->type == GGML_TYPE_F16) {
                                src1_ddf_i_source = (float *) ((char *) src1_extra->data_device[ctx.device] +
                                                               (i0 * ne11 + src1_col_0) * ne10 * sizeof(sycl::half));
                            } else {
                                src1_ddf_i_source = (float *) src1_extra->data_device[ctx.device];
                                src1_ddf_i_source += (i0 * ne11 + src1_col_0) * ne10;
                            }

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(*stream, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
                        }
                    }
                } else {
                    if (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } else {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    if constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        try {
                            quantize_row_q8_1_sycl<quantize_f>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                               src1_padded_col_size, stream);
                        } catch (const sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02 / i02_divisor, dev[i].row_low,
                                                       dev[i].row_high, stream));
                }
                if (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(
                    CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                                       dev[i].row_low, dev[i].row_high, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    if (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02 * nb2 + i03 * nb3);
                        GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                        dhf_dst_i += src1_col_0 * ne0 + dev[i].row_low;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * sizeof(float), dst_dd_i, row_diff * sizeof(float),
                            row_diff * sizeof(float), src1_ncols, dpct::device_to_device, *stream)));
                    } else {
                        float * dhf_dst_i = (float *) ((char *) dst_off_device + i02 * nb2 + i03 * nb3);
                        GGML_ASSERT(dst->nb[1] == ne0 * sizeof(float));
                        dhf_dst_i += src1_col_0 * ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i, src1_ncols * ne0 * sizeof(float)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                if (split && (i != ctx.device || is != 0)) {
                    GGML_SYCL_DEBUG(
                        "[SYCL][MUL_MAT]DEVICE %d (stream %d) RECORDING completion event at events[%d][%d]=%p\n", i,
                        (int) is, i, (int) is, (void *) (src0_extra->events[i][is]));
                    SYCL_CHECK(CHECK_TRY_ERROR(*src0_extra->events[i][is] = stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    if (split && ggml_sycl_info().device_count > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max         = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        GGML_SYCL_DEBUG("[SYCL][MUL_MAT] Main device %d waiting on %d devices, is_max=%d\n", ctx.device,
                        ggml_sycl_info().device_count, (int) is_max);

        ggml_sycl_set_device(ctx.device);
        for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
            if (dev[i].row_low == dev[i].row_high) {
                GGML_SYCL_DEBUG("[SYCL][MUL_MAT] Skipping device %d (row_low==row_high)\n", i);
                continue;
            }
            for (int64_t is = 0; is < is_max; ++is) {
                GGML_SYCL_DEBUG("[SYCL][MUL_MAT] Main waiting on event from dev %d stream %d at ptr=%p\n", i, (int) is,
                                (void *) (src0_extra->events[i][is]));
                SYCL_CHECK(CHECK_TRY_ERROR(ctx.stream()->ext_oneapi_submit_barrier({ *src0_extra->events[i][is] })));
            }
        }
        GGML_SYCL_DEBUG("[SYCL][MUL_MAT] Main device %d finished waiting all devices\n", ctx.device);
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

static void ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx,
                                       const ggml_tensor *         src0,
                                       const ggml_tensor *         src1,
                                       ggml_tensor *               dst) try {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void *  src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx,
                                     const ggml_tensor *         src0,
                                     const ggml_tensor *         src1,
                                     ggml_tensor *               dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];
    const int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    void *  src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x     = nb01 / sizeof(sycl::half);
    const int64_t channel_stride_x = nb02 / sizeof(sycl::half);
    const int64_t channel_stride_y = nb11 / sizeof(float);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12,
                                     channel_stride_x, channel_stride_y, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void k_compute_batched_ptrs(const sycl::half *       src0_as_f16,
                                   const sycl::half *       src1_as_f16,
                                   void *                   dst,
                                   const void **            ptrs_src,
                                   void **                  ptrs_dst,
                                   int64_t                  ne12,
                                   int64_t                  ne13,
                                   int64_t                  ne23,
                                   size_t                   nb02,
                                   size_t                   nb03,
                                   size_t                   nb12,
                                   size_t                   nb13,
                                   size_t                   nbd2,
                                   size_t                   nbd3,
                                   int64_t                  r2,
                                   int64_t                  r3,
                                   const sycl::nd_item<3> & item_ct1) {
    const int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    const int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    const uint8_t * src0_bytes = reinterpret_cast<const uint8_t *>(src0_as_f16);
    const uint8_t * src1_bytes = reinterpret_cast<const uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

static void ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx,
                                           const ggml_tensor *         src0,
                                           const ggml_tensor *         src1,
                                           ggml_tensor *               dst) try {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    const sycl::half * src0_f16 = static_cast<const sycl::half *>(src0->data);
    float *            dst_ddf  = static_cast<float *>(dst->data);

    const sycl::half * src1_f16       = static_cast<const sycl::half *>(src1->data);
    const size_t       type_size_src0 = ggml_type_size(src0->type);
    const size_t       type_size_src1 = ggml_type_size(src1->type);

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    if (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");

#if GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        int    last_dim    = 0;
        int    last_str    = 0;
        size_t largest_str = 0;
        for (int i = 0; i < 4; i++) {
            // last stride is always the largest
            if (src1->nb[i] == largest_str) {
                if (src1->ne[last_dim] == 1) {
                    last_str = i;
                    last_dim = i;
                }
            }
            if (src1->nb[i] > largest_str) {
                largest_str = src1->nb[i];
                last_str    = i;
                last_dim    = i;
            }
        }
        // oneDNN handles strided data and does not need overhead of get_to_fp16_nc_sycl
        const int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
#else
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        const to_fp16_nc_sycl_t to_fp16_nc_sycl = get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f32;
    const void * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1 / nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_disable_dnn) {
        int64_t str_a0 = nb00 / type_size_src0;
        int64_t str_a1 = nb01 / type_size_src0;
        int64_t str_a2 = nb02 / type_size_src0;

        int64_t str_b0 = nb10 / type_size_src1;
        int64_t str_b1 = nb11 / type_size_src1;
        int64_t str_b2 = nb12 / type_size_src1;

        auto launch_gemm_for_batches = [&ctx, queue](const sycl::half * src0, const sycl::half * src1, float * dst,
                                                     int64_t a0, int64_t a1, int64_t batcha, int64_t /*b0*/, int64_t b1,
                                                     int64_t batchb, int64_t sa0, int64_t sa1, int64_t sa2, int64_t sb0,
                                                     int64_t sb1, int64_t sb2, int64_t sd2) {
            bool supported_broadcast = batchb == batcha ? true : batchb == 1 || batcha == 1 ? true : false;
            if (supported_broadcast) {
                DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0, DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                                      DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                                      DnnlGemmWrapper::to_dt<float>(), queue, batcha, batchb);
            } else {
                // iterate over batches from smaller set of matrices (matrix 0)
                int64_t batches0 = batcha;
                int64_t batches1 = batchb;

                if (batches0 > batches1) {
                    int64_t num_mul_mats = batches1;
                    int64_t sub_batch    = batches0 / num_mul_mats;
                    // src0 is batched and bigger, shift and multiply with src1
                    for (int64_t i0 = 0; i0 < num_mul_mats; i0++) {
                        const sycl::half * src0_shifted = src0 + (sa2 * i0 * sub_batch);
                        const sycl::half * src1_shifted = src1 + (sb2 * i0);
                        float *            dst_shifted  = dst + (sd2 * i0 * sub_batch);
                        DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sa0,
                                              sa1, sa2, src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1,
                                              sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(), queue, sub_batch, 1);
                    }
                } else {
                    int64_t num_mul_mats = batches0;
                    int64_t sub_batch    = batches1 / num_mul_mats;
                    // src1 is batched and bigger, shift and multiply with src0
                    for (int64_t i1 = 0; i1 < num_mul_mats; i1++) {
                        const sycl::half * src0_shifted = src0 + (sa2 * i1);
                        const sycl::half * src1_shifted = src1 + (sb2 * i1 * sub_batch);
                        float *            dst_shifted  = dst + (sd2 * i1 * sub_batch);
                        DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sa0,
                                              sa1, sa2, src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1,
                                              sb2, dst_shifted, DnnlGemmWrapper::to_dt<float>(), queue, 1, sub_batch);
                    }
                }
            }
        };

        const bool cont_batches_dim2_a = nb02 * ne02 == nb03;
        const bool cont_batches_dim2_b = nb12 * ne12 == nb13;
        const bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
        const bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
        if (cont_batches_dim2_a && cont_batches_dim2_b) {
            // A batch is considered contiguous if the dimension 2 is not strided
            int64_t batches0 = ne02 * ne03;
            int64_t batches1 = ne12 * ne13;
            launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0, ne10, ne11, batches1, str_a0,
                                    str_a1, str_a2, str_b0, str_b1, str_b2, nb2 / sizeof(float));
        } else if (cont_batches_dim3_a && cont_batches_dim3_b) {
            // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
            int64_t batches0 = ne02 * ne03;
            int64_t batches1 = ne12 * ne13;
            int64_t str_a3   = nb03 / type_size_src0;
            int64_t str_b3   = nb13 / type_size_src1;
            launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0, ne10, ne11, batches1, str_a0,
                                    str_a1, str_a3, str_b0, str_b1, str_b3, nb2 / sizeof(float));
        } else {
            for (int64_t b_a = 0; b_a < ne03; b_a++) {
                const sycl::half * src0_f16_shifted = src0_f16 + (nb03 * b_a / type_size_src0);
                const sycl::half * src1_f16_shifted = src1_f16 + (nb13 * b_a / type_size_src1);
                float *            dst_shifted      = dst_ddf + (nb3 * b_a / sizeof(float));
                int64_t            batches0         = ne02;
                int64_t            batches1         = ne12;
                launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted, ne00, ne01, batches0, ne10,
                                        ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1, str_b2,
                                        nb2 / sizeof(float));
            }
        }

    } else
#endif
    {
        if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            const int64_t sma = ne02 == 1 ? nb03 / nb00 : nb02 / nb00;
            const int64_t smb = ne12 == 1 ? s13 : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma, src1_f16, dpct::library_data_t::real_half,
                s11, smb, beta, dst_ddf, mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } else {
            const int ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<const void *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<void *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<float>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                const void ** ptrs_src_get = ptrs_src.get();
                void **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * sizeof(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * sizeof(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23,
                                           nb02, nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (const void **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (const void **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (void **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

enum class mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

bool ggml_sycl_supports_mmq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_mul_mat_sycl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return !g_ggml_sycl_prioritize_dmmv;
        default:
            return false;
    }
}

inline bool ggml_sycl_supports_reorder_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

static bool ggml_sycl_supports_dmmv(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_F16:
            return true;
        default:
            return false;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
static inline void * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        return syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    return sycl::malloc(size, *stream, sycl::usm::alloc::device);
}

static inline void sycl_ext_free(dpct::queue_ptr stream, void * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async) {
        syclex::async_free(*stream, ptr);
        return;
    }
#else
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    sycl::free(ptr, *stream);
}

static void reorder_qw_q4_0(uint8_t *       data_device,
                            const int       ncols,
                            const int       nrows,
                            size_t          size,
                            size_t          offset,
                            dpct::queue_ptr stream) {
    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % sizeof(block_q4_0) == 0));
    GGML_ASSERT((offset % sizeof(block_q4_0) == 0));
    int  offset_blks = offset / sizeof(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr       = (sycl::half *) (qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event =
        stream->parallel_for(size / sizeof(block_q4_0), [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            const block_q4_0 * x  = (const block_q4_0 *) tmp_buf;
            const int          ib = i;

            for (int j = 0; j < QK4_0 / 2; j++) {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    sycl_ext_free(stream, tmp_buf);
}

static void reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q4_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q4_K) == 0);

    const int nblocks = size / sizeof(block_q4_K);

    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q4_K * x  = (const block_q4_K *) tmp_buf;
        const int          ib = i;

        for (int j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        for (int j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    sycl_ext_free(stream, tmp_buf);
}

static void reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % sizeof(block_q6_K) == 0);
    GGML_ASSERT(offset % sizeof(block_q6_K) == 0);

    const int nblocks = size / sizeof(block_q6_K);

    uint8_t * tmp_buf = static_cast<uint8_t *>(sycl_ext_malloc_device(stream, size));

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    if (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        const block_q6_K * x  = (const block_q6_K *) tmp_buf;
        const int          ib = i;

        const uint8_t * ql              = x[ib].ql;
        const uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        for (int j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        for (int j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        for (int j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    if (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    sycl_ext_free(stream, tmp_buf);
}

static void reorder_qw(const ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t    ncols       = src0->ne[0];
    size_t    nrows       = src0->ne[1];
    size_t    size        = ggml_nbytes(src0);

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
            break;
        case GGML_TYPE_Q4_K:
            reorder_qw_q4_k(data_device, size, 0, stream);
            break;
        case GGML_TYPE_Q6_K:
            reorder_qw_q6_k(data_device, size, 0, stream);
            break;
        default:
            GGML_ABORT("reorder_qw() called with unsupported type");
            break;
    }
}

bool should_reorder_tensor(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    // Disable reordering when graph compatibility is forced, to avoid host sync in reorder_qw
    if (ctx.force_graph_compatible) {
        return false;
    }

    return !g_ggml_sycl_disable_optimize &&  //allow optimize, controlled by $GGML_SYCL_DISABLE_OPT
           ctx.opt_feature.reorder &&        //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&     //limit to some supported cases of Q4_0, to do for more cases.
           dst->src[1]->ne[1] == 1 && dst->src[1]->ne[2] == 1 && dst->src[1]->ne[3] == 1;
}

static void opt_for_reorder(ggml_backend_sycl_context * ctx,
                            const ggml_tensor *         src0,
                            const ggml_tensor * /* src1 */,
                            ggml_tensor * dst,
                            mul_mat_algo  mm_algorithm) {
    if (!should_reorder_tensor(*ctx, dst)) {
        return;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    if (!extra || extra->optimized_feature.reorder) {
        return;  // Skip permutations and already reordered tensors
    }

    switch (mm_algorithm) {
        case mul_mat_algo::DMMV:
            if (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MMVQ:
            if (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                return;
            }
            break;
        case mul_mat_algo::MUL_MAT_SYCL:
            if (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                return;
            }
            break;
    }

    reorder_qw(src0, ctx->stream());
    extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
}

bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % GGML_SYCL_DMMV_X == 0 && src1->ne[1] == 1;
}

bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    return ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

// Wrapper for XMX GEMM to match ggml_sycl_op_mul_mat_t signature
static void ggml_sycl_op_mul_mat_xmx(ggml_backend_sycl_context & ctx,
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
                                     const queue_ptr &           stream) {
    const int64_t M = row_high - row_low;
    const int64_t N = src1_ncols;
    const int64_t K = src0->ne[0];

    // int id = get_current_device_id(); // unused
    // dst is N x M (row-major), ne0=M (inner dim), ne1=N (outer dim)
    const int64_t ldc = dst->ne[0];

    // Compute C^T = src1 * src0^T (N x M)
    // src1 is N x K (row-major). A param. lda = K.
    // src0 is M x K (row-major). B param. ldb = K.
    // gemm_xmx computes A * B^T.
    // So A=src1, B=src0.

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch_gemm_xmx(stream, (const float *) src1_ddf_i, (const float *) src0_dd_i, dst_dd_i, N, M, K, 1.0f, 0.0f, K,
                        K, ldc);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        launch_gemm_xmx_f16(stream, (const sycl::half *) src1_ddf_i, (const sycl::half *) src0_dd_i, dst_dd_i, N, M, K,
                            1.0f, 0.0f, K, K, ldc);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        launch_gemm_xmx_f16_f16(stream, (const sycl::half *) src1_ddf_i, (const sycl::half *) src0_dd_i,
                                (sycl::half *) dst_dd_i, N, M, K, 1.0f, 0.0f, K, K, ldc);
    } else if (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
               src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 || src0->type == GGML_TYPE_Q8_1) {
        // XMX int8 path for quantized types (opt-in via env var, can cause hangs on some configs)
        static bool enable_xmx_int8 = getenv("GGML_SYCL_XMX_INT8") != nullptr;
        if (enable_xmx_int8 && has_int8_xmx_support(stream)) {
            ggml_sycl_op_mul_mat_q_xmx_int8(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low,
                                            row_high, src1_ncols, src1_padded_row_size, stream);
        } else {
            // Fallback to oneMKL (original behavior before this change)
            ggml_sycl_op_mul_mat_sycl(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low,
                                      row_high, src1_ncols, src1_padded_row_size, stream);
        }
    } else {
        // Fallback to oneMKL for unsupported types (mixed, etc.)
        // Note: This fallback is NOT graph-compatible.
        // If graph recording is active, this will crash or produce invalid graph.
        // We rely on backend.cpp to disable graphs for unsupported types if we don't handle them here.
        ggml_sycl_op_mul_mat_sycl(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low, row_high,
                                  src1_ncols, src1_padded_row_size, stream);
    }
}

static void ggml_sycl_op_mul_mat_tiled(ggml_backend_sycl_context & ctx,
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
                                       const queue_ptr &           stream) {
    const int64_t M   = row_high - row_low;
    const int64_t N   = src1_ncols;
    const int64_t K   = src0->ne[0];
    const int64_t ldc = dst->ne[0];

    // Tiled GEMM computes C = alpha * A * B^T + beta * C
    // We map A=src1, B=src0, C=dst.
    // src1 is N x K. src0 is M x K.
    // C = src1 * src0^T = (N x K) * (K x M) = N x M.
    // dst is N x M.

    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        launch_gemm_tiled<true>(stream, (const float *) src1_ddf_i, (const float *) src0_dd_i, dst_dd_i, N, M, K, 1.0f,
                                0.0f, K, K, ldc);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        // F16 weights: convert src1_ddf_i (F32) to F16 temporary
        ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool(), N * K);
        sycl::half *                     src1_f16 = src1_f16_alloc.get();

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(N * K),
                             [=](sycl::id<1> idx) { src1_f16[idx] = (sycl::half) src1_ddf_i[idx]; });
        });

        launch_gemm_f16_f32_tiled(stream, src1_f16, (const sycl::half *) src0_dd_i, dst_dd_i, N, M, K, 1.0f, 0.0f, K, K,
                                  ldc, false);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        // BF16 weights: convert src1_ddf_i (F32) to BF16 temporary
        ggml_sycl_pool_alloc<sycl::ext::oneapi::bfloat16> src1_bf16_alloc(ctx.pool(), N * K);
        sycl::ext::oneapi::bfloat16 *                     src1_bf16 = src1_bf16_alloc.get();

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(N * K),
                             [=](sycl::id<1> idx) { src1_bf16[idx] = (sycl::ext::oneapi::bfloat16) src1_ddf_i[idx]; });
        });

        launch_gemm_bf16_f32_tiled(stream, src1_bf16, (const sycl::ext::oneapi::bfloat16 *) src0_dd_i, dst_dd_i, N, M,
                                   K, 1.0f, 0.0f, K, K, ldc, false);
    } else if (src0->type == GGML_TYPE_MXFP4 && dst->type == GGML_TYPE_F32) {
        // MXFP4 graph support: use fused dequantization kernel
        // Computes: C = A * B^T where B (src0) is MXFP4
        // A = src1 (activations, F16 after conversion), N x K
        // B = src0 (weights, MXFP4), M x K
        // C = dst (output, F32), N x M

        // Convert src1_ddf_i (F32 -> F16)
        ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool(), N * K);
        sycl::half *                     src1_f16 = src1_f16_alloc.get();

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(N * K),
                             [=](sycl::id<1> idx) { src1_f16[idx] = (sycl::half) src1_ddf_i[idx]; });
        });

        // Use fused kernel: A=src1_f16 (N x K), B=src0 (M x K), C=dst (N x M)
        launch_gemm_mxfp4_f32_tiled(stream, src1_f16, src0_dd_i, dst_dd_i, N, M, K, 1.0f, 0.0f, K, K, ldc);
    } else {
        // Fallback for unsupported types
        ggml_sycl_op_mul_mat_sycl(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low, row_high,
                                  src1_ncols, src1_padded_row_size, stream);
    }
}

void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx,
                       const ggml_tensor *         src0,
                       const ggml_tensor *         src1,
                       ggml_tensor *               dst) {
    if (ctx.force_graph_compatible) {
        if (xmx_gemm_available(ctx.stream())) {
            // Check if types are supported by XMX kernels
            bool xmx_types = false;
            if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                xmx_types = true;
            } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
                xmx_types = true;
            }

            if (xmx_types) {
                GGML_SYCL_ITT_MUL_MAT_XMX(f32);
                ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_xmx);
                return;
            }
        }
        GGML_SYCL_ITT_MUL_MAT_TILED(f32);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_tiled);
        return;
    }

    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const bool           split                  = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t              min_compute_capability = INT_MAX;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        for (int id = 0; id < ggml_sycl_info().device_count; ++id) {
            // skip devices that are not going to do any work:
            if (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_count ? tensor_split[id + 1] : 1.0f)) {
                continue;
            }

            if (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } else {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    bool use_mul_mat_q =
        ggml_sycl_supports_mmq(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif  // SYCL_USE_XMX

    // MMQ kernels with need_check=true (when nrows < mmq_y tile size) can have shared memory
    // write collisions that cause GPU hangs on Intel GPUs. Fall back to oneMKL when nrows
    // is too small. The MMQ_Y tile sizes range from 32-128, so we use 128 as the minimum.
    // Also guard against small ne11 (output rows) which causes need_check=true path issues.
    // See AGENTS.md for more details.
    constexpr int64_t MMQ_MIN_NROWS = 128;
    use_mul_mat_q                   = use_mul_mat_q && (src0->ne[1] >= MMQ_MIN_NROWS);
    use_mul_mat_q                   = use_mul_mat_q && (src1->ne[1] >= MMQ_MIN_NROWS);

    // Dispatch becomes obscure with the reorder, MMVQ when the reorder optimization
    // is enabled takes precedence over DMMV, the current if-else implementation
    // requires disabling DMMV if both conditions are met
    if (!g_ggml_sycl_prioritize_dmmv &&
        ((should_reorder_tensor(ctx, dst) && ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
    }

    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        if (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } else {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) &&
               src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) &&
               src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        // NOTE: Experimental tiled GEMM for graph compatibility (F32/F32->F32) is disabled
        // due to precision issues. The tiled kernel produces ~2.0 error vs threshold of 0.0005.
        // Future work: fix the tiled GEMM kernel indexing/transpose handling.
        // For now, F32 MUL_MAT falls through to oneMKL (graph-incompatible but correct).
    } else if (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        GGML_SYCL_ITT_OP(dmmv);
        fprintf(stderr, "ggml_sycl: MUL_MAT DMMV ne=[%ld,%ld,%ld,%ld] type=%s\n", dst->ne[0], dst->ne[1], dst->ne[2],
                dst->ne[3], ggml_type_name(src0->type));
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } else if (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        if (extra && extra->optimized_feature.reorder) {
            GGML_SYCL_ITT_OP(mmvq_reorder);
            fprintf(stderr, "ggml_sycl: MUL_MAT MMVQ_REORDER ne=[%ld,%ld,%ld,%ld] type=%s\n", dst->ne[0], dst->ne[1],
                    dst->ne[2], dst->ne[3], ggml_type_name(src0->type));
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } else {
            GGML_SYCL_ITT_OP(mmvq);
            fprintf(stderr, "ggml_sycl: MUL_MAT MMVQ ne=[%ld,%ld,%ld,%ld] type=%s\n", dst->ne[0], dst->ne[1],
                    dst->ne[2], dst->ne[3], ggml_type_name(src0->type));
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } else if (use_mul_mat_q) {
        GGML_SYCL_ITT_MUL_MAT_MMQ(quantized);
        fprintf(stderr, "ggml_sycl: MUL_MAT MMQ ne=[%ld,%ld,%ld,%ld] type=%s\n", dst->ne[0], dst->ne[1], dst->ne[2],
                dst->ne[3], ggml_type_name(src0->type));
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } else {
        if (xmx_gemm_available(ctx.stream())) {
            // Note: XMX path supports F32, F16, and now Q8_0/Q4_0/etc via int8 XMX
            GGML_SYCL_ITT_MUL_MAT_XMX(f32);
            bool is_quant =
                (src0->type == GGML_TYPE_Q8_0 || src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
                 src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 || src0->type == GGML_TYPE_Q8_1);
            fprintf(stderr, "ggml_sycl: MUL_MAT %s ne=[%ld,%ld,%ld,%ld] type=%s\n", is_quant ? "XMX_INT8" : "XMX",
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], ggml_type_name(src0->type));
            if (is_quant) {
                ggml_sycl_op_mul_mat<quantize_q8_1_for_xmx>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_xmx);
            } else {
                ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_xmx);
            }
        } else {
            GGML_SYCL_ITT_MUL_MAT_MKL(f32);
            fprintf(stderr, "ggml_sycl: MUL_MAT MKL ne=[%ld,%ld,%ld,%ld] type=%s\n", dst->ne[0], dst->ne[1], dst->ne[2],
                    dst->ne[3], ggml_type_name(src0->type));
            ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
        }
    }
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

__dpct_inline__ static void k_copy_src1_to_contiguous(const char * __restrict__ src1_original,
                                                      char * __restrict__ src1_contiguous,
                                                      int * __restrict__ cur_src1_row,
                                                      mmid_row_mapping * __restrict__ row_mapping,
                                                      const char * __restrict ids,
                                                      int64_t                  i02,
                                                      size_t                   ids_nb1,
                                                      size_t                   ids_nb0,
                                                      int64_t                  ne11,
                                                      int64_t                  ne10,
                                                      size_t                   nb11,
                                                      size_t                   nb12,
                                                      const sycl::nd_item<3> & item_ct1,
                                                      int &                    src1_row) {
    int32_t iid1 = item_ct1.get_group(2);
    int32_t id   = item_ct1.get_group(1);

    const int32_t row_id_i = *(const int32_t *) (ids + iid1 * ids_nb1 + id * ids_nb0);

    if (row_id_i != i02) {
        return;
    }

    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;

    if (item_ct1.get_local_id(2) == 0) {
        src1_row              = dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(cur_src1_row, 1);
        row_mapping[src1_row] = { id, iid1 };
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    const float * src1_row_original   = (const float *) (src1_original + i11 * nb11 + i12 * nb12);
    float *       src1_row_contiguous = (float *) (src1_contiguous + src1_row * nb11);

#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < ne10; i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ static void k_copy_dst_from_contiguous(char * __restrict__ dst_original,
                                                       const char * __restrict__ dst_contiguous,
                                                       const mmid_row_mapping * __restrict__ row_mapping,
                                                       int64_t                  ne0,
                                                       size_t                   nb1,
                                                       size_t                   nb2,
                                                       const sycl::nd_item<3> & item_ct1) {
    int32_t i = item_ct1.get_group(2);

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *) (dst_contiguous + i * nb1);
    float *       dst_row_original   = (float *) (dst_original + i1 * nb1 + i2 * nb2);

#pragma unroll
    for (int j = item_ct1.get_local_id(2); j < ne0; j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

static void k_count_experts(const char * __restrict__ ids,
                            int * __restrict__ expert_counts,
                            int * __restrict__ row_dst_index,
                            int              n_ids,
                            int              n_batches,
                            int              n_experts,
                            size_t           ids_nb0,
                            size_t           ids_nb1,
                            sycl::nd_item<1> item) {
    int global_id  = item.get_global_id(0);
    int total_rows = n_ids * n_batches;
    if (global_id >= total_rows) {
        return;
    }

    // Map global_id to (id, iid1)
    // global_id = iid1 * n_ids + id
    int id   = global_id % n_ids;
    int iid1 = global_id / n_ids;

    const int32_t expert_id = *(const int32_t *) (ids + iid1 * ids_nb1 + id * ids_nb0);

    if (expert_id < 0 || expert_id >= n_experts) {
        // Invalid expert ID - skip or handle error
        // For now, set index to -1 to indicate invalid
        row_dst_index[global_id] = -1;
        return;
    }

    // Atomic increment count for this expert
    // row_dst_index stores the index *within* this expert's block
    row_dst_index[global_id] =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&expert_counts[expert_id], 1);
}

static void k_scan_experts(const int * __restrict__ expert_counts,
                           int * __restrict__ expert_offsets,
                           int              n_experts,
                           sycl::nd_item<1> item) {
    if (item.get_global_id(0) == 0) {
        int offset = 0;
        for (int i = 0; i < n_experts; ++i) {
            expert_offsets[i] = offset;
            offset += expert_counts[i];
        }
    }
}

static void k_pack_experts(const char * __restrict__ src1_original,
                           float * __restrict__ src1_packed,
                           mmid_row_mapping * __restrict__ dst_mapping,
                           const char * __restrict__ ids,
                           const int * __restrict__ expert_offsets,
                           const int * __restrict__ row_dst_index,
                           int              n_ids,
                           int              n_batches,
                           int              n_experts,
                           size_t           ids_nb0,
                           size_t           ids_nb1,
                           int64_t          ne10,
                           int64_t          ne11,
                           size_t           nb11,
                           size_t           nb12,
                           sycl::nd_item<2> item) {
    const int global_row = item.get_global_id(0);
    const int global_col = item.get_global_id(1);
    const int total_rows = n_ids * n_batches;
    if (global_row >= total_rows || global_col >= ne10) {
        return;
    }

    const int id   = global_row % n_ids;
    const int iid1 = global_row / n_ids;

    const int32_t expert_id = *(const int32_t *) (ids + iid1 * ids_nb1 + id * ids_nb0);

    // Bounds check
    if (expert_id < 0 || expert_id >= n_experts) {
        return;
    }

    const int row_idx_in_expert = row_dst_index[global_row];
    if (row_idx_in_expert < 0) {
        return;  // Was marked invalid
    }

    const int dst_idx = expert_offsets[expert_id] + row_idx_in_expert;

    if (global_col == 0) {
        dst_mapping[dst_idx] = { id, iid1 };
    }

    const int64_t i11     = id % ne11;
    const int64_t i12     = iid1;
    const float * src_ptr = (const float *) (src1_original + i11 * nb11 + i12 * nb12);
    float *       dst_ptr = src1_packed + dst_idx * ne10;

    dst_ptr[global_col] = src_ptr[global_col];
}

static void k_pack_experts_f16(const char * __restrict__ src1_original,
                               sycl::half * __restrict__ src1_packed,
                               mmid_row_mapping * __restrict__ dst_mapping,
                               const char * __restrict__ ids,
                               const int * __restrict__ expert_offsets,
                               const int * __restrict__ row_dst_index,
                               int              n_ids,
                               int              n_batches,
                               int              n_experts,
                               size_t           ids_nb0,
                               size_t           ids_nb1,
                               int64_t          ne10,
                               int64_t          ne11,
                               size_t           nb11,
                               size_t           nb12,
                               sycl::nd_item<2> item) {
    const int global_row = item.get_global_id(0);
    const int global_col = item.get_global_id(1);
    const int total_rows = n_ids * n_batches;
    if (global_row >= total_rows || global_col >= ne10) {
        return;
    }

    const int id   = global_row % n_ids;
    const int iid1 = global_row / n_ids;

    const int32_t expert_id = *(const int32_t *) (ids + iid1 * ids_nb1 + id * ids_nb0);

    if (expert_id < 0 || expert_id >= n_experts) {
        return;
    }

    const int row_idx_in_expert = row_dst_index[global_row];
    if (row_idx_in_expert < 0) {
        return;
    }

    const int dst_idx = expert_offsets[expert_id] + row_idx_in_expert;

    if (global_col == 0) {
        dst_mapping[dst_idx] = { id, iid1 };
    }

    const int64_t i11     = id % ne11;
    const int64_t i12     = iid1;
    const float * src_ptr = (const float *) (src1_original + i11 * nb11 + i12 * nb12);
    sycl::half *  dst_ptr = src1_packed + dst_idx * ne10;

    dst_ptr[global_col] = static_cast<sycl::half>(src_ptr[global_col]);
}

static void k_unpack_experts(char * __restrict__ dst_original,
                             const char * __restrict__ dst_packed,
                             const mmid_row_mapping * __restrict__ row_mapping,
                             int64_t          ne0,
                             size_t           nb1,
                             size_t           nb2,
                             int64_t          row_low,
                             int64_t          row_high,  // device row range [row_low, row_high)
                             int              total_rows,
                             sycl::nd_item<2> item) {
    const int row = item.get_global_id(0);
    const int col = item.get_global_id(1);
    if (row >= total_rows || col >= ne0) {
        return;
    }

    const int32_t i1 = row_mapping[row].i1;
    const int32_t i2 = row_mapping[row].i2;

    // Skip rows not belonging to this device's split
    if (i1 < row_low || i1 >= row_high) {
        return;
    }

    const float * src_ptr = (const float *) (dst_packed) + row * ne0;
    float *       dst_ptr = (float *) (dst_original + i1 * nb1 + i2 * nb2);

    dst_ptr[col] = src_ptr[col];
}

static void ggml_sycl_mul_mat_id_tiled(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // Weights [K, N, Expert]
    const ggml_tensor * src1 = dst->src[1];  // Input [K, Batch]
    const ggml_tensor * ids  = dst->src[2];  // Expert IDs [Batch]

    const int64_t n_experts  = src0->ne[2];
    const int64_t n_ids      = ids->ne[0];
    const int64_t n_batches  = ids->ne[1];
    const int64_t total_rows = n_ids * n_batches;

    const int64_t K    = src0->ne[0];
    const int64_t N    = src0->ne[1];  // Output dim
    // src1 is [K, total_rows] effectively
    const int64_t ne10 = src1->ne[0];  // K

    // Handle split weights (src0)
    // If src0 is split, each device processes a subset of rows (N).
    // We assume split-mode row, where N is split across devices.
    const bool split    = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t    row_low  = 0;
    int64_t    row_high = N;
    int64_t    N_local  = N;

    if (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;

        // Calculate row range for this device
        // Matches logic in ggml_sycl_op_mul_mat
        const int64_t rounding = 1;  // MUL_MAT_ID supports arbitrary N usually, but let's check alignment

        if (ctx.device != 0) {
            row_low = N * tensor_split[ctx.device];
            if (row_low < N) {
                row_low -= row_low % rounding;
            }
        }

        if (ctx.device != ggml_sycl_info().device_count - 1) {
            row_high = N * tensor_split[ctx.device + 1];
            if (row_high < N) {
                row_high -= row_high % rounding;
            }
        }

        // Ensure valid range
        if (row_low >= row_high) {
            // No work for this device
            // We must still participate in any collective ops if they existed, but here we just return?
            // Wait, we need to ensure stream state is valid.
            return;
        }

        N_local = row_high - row_low;
    }

    // Allocate temp buffers
    // 1. Expert counts and offsets
    ggml_sycl_pool_alloc<int> dev_expert_counts(ctx.pool(), n_experts);
    ggml_sycl_pool_alloc<int> dev_expert_offsets(ctx.pool(), n_experts);
    ggml_sycl_pool_alloc<int> dev_row_dst_index(ctx.pool(), total_rows);

    // 2. Packed buffers
    // src1 is broadcast/shared, so it's full size. Use F16 pack only for MXFP4 path.
    const bool                       use_f16_pack = (src0->type == GGML_TYPE_MXFP4);
    ggml_sycl_pool_alloc<float>      dev_src1_packed_f32(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> dev_src1_packed_f16(ctx.pool());
    if (use_f16_pack) {
        dev_src1_packed_f16.alloc(total_rows * ne10);
    } else {
        dev_src1_packed_f32.alloc(total_rows * ne10);
    }
    // dst is split if N is split. We compute [N_local, total_rows].
    // dev_dst_packed stores the partial result for this device.
    ggml_sycl_pool_alloc<char>             dev_dst_packed(ctx.pool(), sizeof(float) * total_rows * N_local);
    ggml_sycl_pool_alloc<mmid_row_mapping> dev_dst_mapping(ctx.pool(), total_rows);

    queue_ptr stream = ctx.stream();

    // Zero counts
    stream->memset(dev_expert_counts.get(), 0, n_experts * sizeof(int));
    // Initialize mapping with -1 to catch invalid/skipped rows
    stream->memset(dev_dst_mapping.get(), -1, total_rows * sizeof(mmid_row_mapping));

    // Launch Count
    stream->submit([&](sycl::handler & cgh) {
        const char * ids_data      = (const char *) ids->data;
        int *        counts_ptr    = dev_expert_counts.get();
        int *        row_index_ptr = dev_row_dst_index.get();
        size_t       nb0           = ids->nb[0];
        size_t       nb1           = ids->nb[1];

        size_t global_range = ((total_rows + 255) / 256) * 256;
        cgh.parallel_for(sycl::nd_range<1>(global_range, 256), [=](sycl::nd_item<1> item) {
            k_count_experts(ids_data, counts_ptr, row_index_ptr, n_ids, n_batches, n_experts, nb0, nb1, item);
        });
    });

    // Launch Scan (Single thread)
    stream->submit([&](sycl::handler & cgh) {
        int * counts_ptr  = dev_expert_counts.get();
        int * offsets_ptr = dev_expert_offsets.get();
        cgh.parallel_for(sycl::nd_range<1>(1, 1),
                         [=](sycl::nd_item<1> item) { k_scan_experts(counts_ptr, offsets_ptr, n_experts, item); });
    });

    // Launch Pack
    stream->submit([&](sycl::handler & cgh) {
        const char *       src1_data       = (const char *) src1->data;
        float *            packed_data_f32 = dev_src1_packed_f32.get();
        sycl::half *       packed_data_f16 = dev_src1_packed_f16.get();
        mmid_row_mapping * map_data        = dev_dst_mapping.get();
        const char *       ids_data        = (const char *) ids->data;
        int *              offsets_ptr     = dev_expert_offsets.get();
        int *              row_index_ptr   = dev_row_dst_index.get();
        size_t             nb0             = ids->nb[0];
        size_t             nb1             = ids->nb[1];
        int64_t            ne11            = src1->ne[1];
        size_t             src_nb11        = src1->nb[1];
        size_t             src_nb12        = src1->nb[2];

        constexpr int wg_cols     = 128;
        const size_t  global_rows = total_rows;
        const size_t  global_cols = ((ne10 + wg_cols - 1) / wg_cols) * wg_cols;
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols), sycl::range<2>(1, wg_cols)),
            [=](sycl::nd_item<2> item) {
                if (use_f16_pack) {
                    k_pack_experts_f16(src1_data, packed_data_f16, map_data, ids_data, offsets_ptr, row_index_ptr,
                                       n_ids, n_batches, n_experts, nb0, nb1, ne10, ne11, src_nb11, src_nb12, item);
                } else {
                    k_pack_experts(src1_data, packed_data_f32, map_data, ids_data, offsets_ptr, row_index_ptr, n_ids,
                                   n_batches, n_experts, nb0, nb1, ne10, ne11, src_nb11, src_nb12, item);
                }
            });
    });

    // Adjust stride for weights: if split, each expert is smaller [K, N_local]
    // If src0 buffer is split, it is compacted. So stride is K * N_local * sizeof(type).
    // src0->nb[2] usually reflects the FULL stride if not adjusted?
    // Wait, ggml split buffer type adjusts the tensor shape/stride?
    // ggml_backend_sycl_buffer_type splits the storage.
    // But tensor dimensions in `src0` might still reflect the full tensor if not reshaped?
    // In `MUL_MAT`, it uses `src0_dd` allocation which is compact.
    // Here `src0->data` is used.
    // If `src0` is a view on a split buffer, `src0->data` points to device memory.
    // We assume `src0` describes the FULL tensor, but `src0->data` points to the LOCAL slice?
    // No, standard GGML split logic:
    // The backend receives the full tensor struct, but `data` points to the split buffer.
    // We need to calculate the correct offset between experts.
    // Expert size in bytes = K * N_local * type_size.

    size_t expert_stride = src0->nb[2];
    if (split) {
        // If split, the buffer is compact on device.
        // Stride is derived from N_local.
        // Assuming row-major [Expert, N, K] or [Expert, K, N]?
        // src0 is [K, N, Experts].
        // Stride nb[2] is stride between experts.
        // nb[0] = type_size
        // nb[1] = K * type_size
        // nb[2] = N * K * type_size
        // If split, N becomes N_local.
        expert_stride = N_local * ggml_row_size(src0->type, K);
    }

    const float *      src0_base       = (const float *) src0->data;
    const float *      src1_packed_f32 = dev_src1_packed_f32.get();
    const sycl::half * src1_packed_f16 = dev_src1_packed_f16.get();
    float *            dst_packed      = (float *) dev_dst_packed.get();

    for (int i = 0; i < n_experts; ++i) {
        if (src0->type == GGML_TYPE_F16) {
            const sycl::half * weights = (const sycl::half *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_f32_f16(stream, src1_packed_f32, weights, dst_packed,
                                               dev_expert_counts.get() + i, dev_expert_offsets.get() + i,
                                               total_rows,  // max_M
                                               N_local, K, 1.0f, 0.0f, K, K, N_local);
        } else if (src0->type == GGML_TYPE_BF16) {
            const sycl::ext::oneapi::bfloat16 * weights =
                (const sycl::ext::oneapi::bfloat16 *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_f32_bf16(stream, src1_packed_f32, weights, dst_packed,
                                                dev_expert_counts.get() + i, dev_expert_offsets.get() + i,
                                                total_rows,  // max_M
                                                N_local, K, 1.0f, 0.0f, K, K, N_local);
        } else if (src0->type == GGML_TYPE_MXFP4) {
            // MXFP4 weights with fused dequantization + F16 activations
            // ldb = K (the full K dimension, used for block indexing: K/32 blocks per row)
            const block_mxfp4 * weights = (const block_mxfp4 *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_mxfp4_f16(stream, src1_packed_f16, weights, dst_packed,
                                                 dev_expert_counts.get() + i, dev_expert_offsets.get() + i,
                                                 total_rows,  // max_M
                                                 N_local, K, 1.0f, 0.0f, K, K, N_local);
        } else if (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 weights with fused dequantization
            // ldb = K (used for block indexing: K/QK4_0 blocks per row)
            const block_q4_0 * weights = (const block_q4_0 *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_q4_0(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                            dev_expert_offsets.get() + i,
                                            total_rows,  // max_M
                                            N_local, K, 1.0f, 0.0f, K, K, N_local);
        } else if (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 weights with fused dequantization
            // ldb = K (used for block indexing: K/QK8_0 blocks per row)
            const block_q8_0 * weights = (const block_q8_0 *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_q8_0(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                            dev_expert_offsets.get() + i,
                                            total_rows,  // max_M
                                            N_local, K, 1.0f, 0.0f, K, K, N_local);
        } else if (src0->type == GGML_TYPE_Q2_K) {
            const block_q2_K * weights = (const block_q2_K *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_q2_K(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                            dev_expert_offsets.get() + i, total_rows, N_local, K, 1.0f, 0.0f, K, K,
                                            N_local);
        } else if (src0->type == GGML_TYPE_Q3_K) {
            const block_q3_K * weights = (const block_q3_K *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_q3_K(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                            dev_expert_offsets.get() + i, total_rows, N_local, K, 1.0f, 0.0f, K, K,
                                            N_local);
        } else if (src0->type == GGML_TYPE_Q4_K) {
            const block_q4_K * weights = (const block_q4_K *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_q4_K(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                            dev_expert_offsets.get() + i, total_rows, N_local, K, 1.0f, 0.0f, K, K,
                                            N_local);
        } else if (src0->type == GGML_TYPE_Q5_K) {
            const block_q5_K * weights = (const block_q5_K *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_q5_K(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                            dev_expert_offsets.get() + i, total_rows, N_local, K, 1.0f, 0.0f, K, K,
                                            N_local);
        } else if (src0->type == GGML_TYPE_Q6_K) {
            const block_q6_K * weights = (const block_q6_K *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect_q6_K(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                            dev_expert_offsets.get() + i, total_rows, N_local, K, 1.0f, 0.0f, K, K,
                                            N_local);
        } else {
            const float * weights = (const float *) ((const char *) src0_base + i * expert_stride);
            launch_gemm_tiled_indirect(stream, src1_packed_f32, weights, dst_packed, dev_expert_counts.get() + i,
                                       dev_expert_offsets.get() + i,
                                       total_rows,  // max_M
                                       N_local, K, 1.0f, 0.0f, K, K, N_local);
        }
    }

    stream->submit([&](sycl::handler & cgh) {
        char *             dst_data    = (char *) dst->data;
        const char *       packed_data = dev_dst_packed.get();
        mmid_row_mapping * map_data    = dev_dst_mapping.get();
        // int64_t dst_ne0 = dst->ne[0]; // unused
        size_t             dst_nb1     = dst->nb[1];
        size_t             dst_nb2     = dst->nb[2];

        int64_t low  = row_low;
        int64_t high = row_high;

        constexpr int wg_cols     = 128;
        const size_t  global_rows = total_rows;
        const size_t  global_cols = ((N_local + wg_cols - 1) / wg_cols) * wg_cols;
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(global_rows, global_cols), sycl::range<2>(1, wg_cols)),
                         [=](sycl::nd_item<2> item) {
                             k_unpack_experts(dst_data, packed_data, map_data, N_local, dst_nb1, dst_nb2, low, high,
                                              total_rows, item);
                         });
    });
}

void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    // Use tiled path on Xe2 to work around IGC compiler crash in AddRequiredMemoryFences pass.
    // The MMQ kernels trigger a bug in IGC's SLM fence insertion when compiled as part of the
    // MUL_MAT_ID SPIR-V module (same kernels work fine in the MUL_MAT path).
    // Set GGML_SYCL_MUL_MAT_ID_XMX=1 to force XMX path (requires patched IGC to avoid crashes).
    static bool enable_xmx = getenv("GGML_SYCL_MUL_MAT_ID_XMX") != nullptr;
    const bool  use_tiled =
        !enable_xmx && (ctx.force_graph_compatible || ggml_sycl_info().devices[ctx.device].arch == SYCL_ARCH_INTEL_XE2);
    if (use_tiled) {
        GGML_SYCL_ITT_MUL_MAT_ID_TILED(moe);
        fprintf(stderr, "ggml_sycl: MUL_MAT_ID TILED ne=[%ld,%ld,%ld,%ld] type=%s\n", dst->ne[0], dst->ne[1],
                dst->ne[2], dst->ne[3], ggml_type_name(dst->src[0]->type));
        ggml_sycl_mul_mat_id_tiled(ctx, dst);
        return;
    }
    GGML_SYCL_ITT_MUL_MAT_ID_MMQ(moe);
    fprintf(stderr, "ggml_sycl: MUL_MAT_ID MMQ ne=[%ld,%ld,%ld,%ld] type=%s\n", dst->ne[0], dst->ne[1], dst->ne[2],
            dst->ne[3], ggml_type_name(dst->src[0]->type));

    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    const ggml_tensor *  src0 = dst->src[0];
    const ggml_tensor *  src1 = dst->src[1];
    const ggml_tensor *  ids  = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS

    const queue_ptr stream = ctx.stream();

    const int64_t n_as  = ne02;
    const int64_t n_ids = ids->ne[0];

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char *      ids_dev = (const char *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *) dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    if (ne12 == 1) {
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            for (int64_t id = 0; id < n_ids; id++) {
                const int32_t i02 = *(const int32_t *) (ids_host.data() + iid1 * ids->nb[1] + id * ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                const int64_t i11 = id % ne11;
                const int64_t i12 = iid1;

                const int64_t i1 = id;
                const int64_t i2 = i12;

                src0_row.data = src0_original + i02 * nb02;
                src1_row.data = src1_original + i11 * nb11 + i12 * nb12;
                dst_row.data  = dst_original + i1 * nb1 + i2 * nb2;

                ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } else {
        ggml_sycl_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float) * ggml_nelements(src1));
        ggml_sycl_pool_alloc<char> dst_contiguous(ctx.pool(), sizeof(float) * ggml_nelements(dst));

        src1_row.data = src1_contiguous.get();
        dst_row.data  = dst_contiguous.get();

        for (int64_t i02 = 0; i02 < n_as; i02++) {
            int64_t num_src1_rows = 0;
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
                for (int64_t id = 0; id < n_ids; id++) {
                    const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

                    if (row_id_i != i02) {
                        continue;
                    }

                    num_src1_rows++;
                }
            }

            if (num_src1_rows == 0) {
                continue;
            }

            ggml_sycl_pool_alloc<int>              dev_cur_src1_row(ctx.pool(), 1);
            ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), num_src1_rows);
            SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dev_cur_src1_row.get(), 0, sizeof(int))));

            const unsigned int max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
            assert(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int) ne10, max_work_group_size));
                sycl::range<3> grid_dims(1, n_ids, ids->ne[1]);
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<int, 0> src1_row_acc(cgh);

                    char * __restrict src1_contiguous_get             = src1_contiguous.get();
                    int * __restrict dev_cur_src1_row_get             = dev_cur_src1_row.get();
                    mmid_row_mapping * __restrict dev_row_mapping_get = dev_row_mapping.get();
                    size_t ids_nb_ct6                                 = ids->nb[1];
                    size_t ids_nb_ct7                                 = ids->nb[0];

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                            k_copy_src1_to_contiguous(src1_original, src1_contiguous_get, dev_cur_src1_row_get,
                                                      dev_row_mapping_get, ids_dev, i02, ids_nb_ct6, ids_nb_ct7, ne11,
                                                      ne10, nb11, nb12, item_ct1, src1_row_acc);
                        });
                });
            }

            src0_row.data = src0_original + i02 * nb02;

            GGML_ASSERT(nb11 == sizeof(float) * ne10);
            GGML_ASSERT(nb1 == sizeof(float) * ne0);
            src1_row.ne[1] = num_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = num_src1_rows * nb11;
            src1_row.nb[3] = num_src1_rows * nb11;

            dst_row.ne[1] = num_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = num_src1_rows * nb1;
            dst_row.nb[3] = num_src1_rows * nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);

            {
                sycl::range<3> block_dims(1, 1, std::min((unsigned int) ne0, max_work_group_size));
                sycl::range<3> grid_dims(1, 1, num_src1_rows);
                stream->submit([&](sycl::handler & cgh) {
                    const char * __restrict dst_contiguous_get              = dst_contiguous.get();
                    const mmid_row_mapping * __restrict dev_row_mapping_get = dev_row_mapping.get();

                    cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         k_copy_dst_from_contiguous(dst_original, dst_contiguous_get,
                                                                    dev_row_mapping_get, ne0, nb1, nb2, item_ct1);
                                     });
                });
            }

            // IMPORTANT: Synchronize before dev_cur_src1_row and dev_row_mapping go out of scope.
            // These pool allocations are used by the kernels above. Without this wait, the memory
            // may be reused/freed on the next iteration while kernels are still accessing it,
            // causing GPU page faults (use-after-free).
            SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        }
    }
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}
