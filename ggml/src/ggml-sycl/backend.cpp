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

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <float.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <sys/time.h>

#include <sycl/sycl.hpp>
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#include <sycl/half_type.hpp>

#include "ggml-sycl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-sycl/add-id.hpp"
#include "ggml-sycl/backend.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/element_wise.hpp"
#include "ggml-sycl/norm.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/gemm_tiled.hpp"
#include "ggml-sycl/gemm_xmx.hpp"
#include "ggml-sycl/gemm_f16_f32_tiled.hpp"
#include "ggml-sycl/set_rows.hpp"
#include "ggml-sycl/set.hpp"
#include "ggml-sycl/sycl_hw.hpp"
#include "ggml-sycl/getrows.hpp"
#include "ggml-sycl/repeat_back.hpp"
#include "ggml-sycl/quantize.hpp"
#include "ggml-sycl/ssm_conv.hpp"
#include "ggml.h"

#include "sycl_backend.hpp"
#include "sycl_buffer.hpp"
#include "sycl_compute.hpp"
#include "sycl_matmul.hpp"
#include "sycl_defs.hpp"
static void ggml_sycl_set_main_device(const int main_device) try {
    if (dpct::get_current_device_id() == static_cast<unsigned int> (main_device)) {
        return;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    if (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) try {
    if (!g_sycl_loaded) return false;

    if (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_sycl_count_equal(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                return false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            break;
        case GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_sycl_op_top_k(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
} catch (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API void ggml_backend_sycl_get_device_description(int device, char *description,
                                      size_t description_size) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ggml_backend_sycl_get_device_memory(int device, size_t *free,
                                                   size_t *total) try {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ggml_sycl_set_device(device);

    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::dev_mgr::instance().get_device(device).get_memory_info(*free, *total)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    return sycl_ctx->name.c_str();
}

static void ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

static void ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((char *)tensor->data + offset, data, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) try {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (const char *)tensor->data + offset, size)));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *src,
                                               ggml_tensor *dst) try {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    bool is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    if (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        return true;
    }

    return false;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_synchronize(ggml_backend_t backend) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    const queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    GGML_SYCL_DEBUG("[SYCL] %s: about to wait on stream %p for device %d\n", __func__, (void*)stream, sycl_ctx->device);
    
    // Add timeout detection for debugging hangs
    auto start = std::chrono::steady_clock::now();
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (elapsed_ms > 5000) {
        GGML_LOG_WARN("[SYCL] %s: wait took %ld ms (potential hang detected)\n", __func__, (long)elapsed_ms);
    }
    GGML_SYCL_DEBUG("[SYCL] %s: wait completed successfully (took %ld ms)\n", __func__, (long)elapsed_ms);

    GGML_UNUSED(backend);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        bool ok = ggml_sycl_compute_forward(*sycl_ctx, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }
}

#ifdef GGML_SYCL_GRAPH

// Graph compatibility result
enum class graph_compat_t {
    DISABLED,        // Graphs cannot be used
    SINGLE_DEVICE,   // Use single-device graph (original path)
    MULTI_DEVICE     // Use per-device graphs (new path)
};

// Compute a hash of the cgraph topology for graph caching
// This allows reusing compiled graphs when the same structure is encountered
static uint64_t compute_cgraph_hash(const ggml_cgraph * cgraph) {
    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV-1a offset basis
    constexpr uint64_t FNV_PRIME = 0x100000001b3ULL;

    // Hash number of nodes
    hash ^= static_cast<uint64_t>(cgraph->n_nodes);
    hash *= FNV_PRIME;

    // Hash each node's key properties (op, type, shape)
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];

        // Hash operation type
        hash ^= static_cast<uint64_t>(node->op);
        hash *= FNV_PRIME;

        // Hash tensor type
        hash ^= static_cast<uint64_t>(node->type);
        hash *= FNV_PRIME;

        // Hash dimensions
        for (int j = 0; j < GGML_MAX_DIMS; j++) {
            hash ^= static_cast<uint64_t>(node->ne[j]);
            hash *= FNV_PRIME;
        }

        // Hash source tensor types and shapes (important for MUL_MAT variants)
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
            hash ^= static_cast<uint64_t>(node->src[s]->type);
            hash *= FNV_PRIME;
            for (int j = 0; j < GGML_MAX_DIMS; j++) {
                hash ^= static_cast<uint64_t>(node->src[s]->ne[j]);
                hash *= FNV_PRIME;
            }
        }
    }

    return hash;
}

static graph_compat_t check_graph_compatibility(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Heuristic: Disable graphs for very large compute graphs to avoid driver hang/compile explosion.
    // The exact threshold may need tuning per device/driver.
    if (cgraph->n_nodes > 500) {
        GGML_LOG_INFO("%s: disabling SYCL graphs due to large graph size (%d nodes)\n", __func__, cgraph->n_nodes);
        return graph_compat_t::DISABLED;
    }

    bool has_split_buffers = false;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        const ggml_op node_op = node->op;

        // Check for split buffers (multi-device)
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s] && node->src[s]->buffer &&
                ggml_backend_buffer_is_sycl_split(node->src[s]->buffer)) {
                has_split_buffers = true;
            }
        }

        switch (node_op) {
            default:
                break;
            case GGML_OP_CONCAT:
                break;
            case GGML_OP_MUL_MAT_ID:
                // Graph-compatible tiled implementation is now available.
                // It runs entirely on device without host synchronization.
                break;
        case GGML_OP_SET_ROWS:
            // SET_ROWS uses USM memory and has implicit data dependencies.
            // We insert an explicit barrier in ggml_sycl_op_set_rows to ensure ordering.
            //
            // However, when SET_ROWS operates on a VIEW of ROPE output (ROPE → VIEW → SET_ROWS pattern),
            // with large tensors, the Level Zero driver crashes with GPU page faults on the BCS engine.
            // This appears to be a driver bug with memory aliasing in SYCL graphs.
            // See kernel log: "Faulted Address... EngineClass: 3 bcs... Engine memory CAT error"
            //
            // Workaround: disable graphs when SET_ROWS source is a VIEW with large tensors.
            {
                ggml_tensor * src = node->src[0];
                if (src && src->op == GGML_OP_VIEW) {
                    int64_t view_size = ggml_nbytes(src);
                    // Threshold: 1MB - large VIEW operations trigger driver bug
                    if (view_size > 1024 * 1024) {
                        GGML_LOG_INFO("%s: disabling SYCL graphs - SET_ROWS with large VIEW source (%ld bytes) "
                                      "triggers Level Zero driver page fault bug\n", __func__, (long)view_size);
                        return graph_compat_t::DISABLED;
                    }
                }
            }
            break;
        case GGML_OP_OUT_PROD:
            // OUT_PROD uses custom kernel which is graph-compatible.
            break;
        case GGML_OP_MUL_MAT:
            // Disable SYCL graphs for MUL_MAT operations due to driver instability/GPU faults (Level Zero error 20)
            // Observed with various quantized types (Q8_0, MXFP4) and mixed precision (F16/BF16)
            // The crash occurs during stream synchronization after graph execution.
            // Until the root cause (buffer lifetime or kernel issue) is fixed, we disable graphs for MUL_MAT.
            {
                ggml_tensor * src0 = node->src[0];
                GGML_LOG_DEBUG("%s: disabling SYCL graphs for MUL_MAT (type=%s) to prevent GPU faults\n", 
                    __func__, ggml_type_name(src0->type));
                return graph_compat_t::DISABLED;
            }
                {
                    ggml_tensor * src0 = node->src[0];
                    ggml_tensor * src1 = node->src[1];
                    ggml_tensor * dst  = node;

                    // Optional debugging threshold
                    static int max_ne = get_sycl_env("GGML_SYCL_GRAPH_MUL_MAT_MAX_NE", -1);
                    if (max_ne != -1 && ggml_nelements(dst) > max_ne) {
                        GGML_LOG_INFO("%s: disabling SYCL graphs due to MUL_MAT size %ld > %d\n", __func__, (long)ggml_nelements(dst), max_ne);
                        return graph_compat_t::DISABLED;
                    }

                    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
                    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);
                    bool use_mul_mat_q = ggml_sycl_supports_mmq(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

                    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
                    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif

                    // Exclude known problematic MMQ types that cause GPU faults under SYCL graphs
                    // q5_0 and q8_0 with certain shapes have been observed to cause device loss
                    // MXFP4 also causes device loss in graphs
                    if (src0->type == GGML_TYPE_Q5_0 || 
                        src0->type == GGML_TYPE_Q8_0 || 
                        src0->type == GGML_TYPE_MXFP4 ||
                        src0->type == GGML_TYPE_F16 ||
                        src0->type == GGML_TYPE_BF16) {
                        GGML_LOG_INFO("%s: disabling SYCL graphs for problematic MMQ type %s\n",
                            __func__, ggml_type_name(src0->type));
                        return graph_compat_t::DISABLED;
                    }

                    // Reordering logic:
                    // During graph recording, we disable reordering in matmul.cpp by checking force_graph_compatible.
                    // So we don't need to disable graph here. The execution path will just skip reordering.
                    /*
                    if (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) && ggml_sycl_supports_reorder_mmvq(src0->type)))) {
                        GGML_LOG_INFO("%s: disabling SYCL graphs to perform tensor reordering (src0 type=%s)\n",
                                __func__, ggml_type_name(src0->type));
                        return graph_compat_t::DISABLED;
                    }
                    */

                    const bool split = src0->buffer && ggml_backend_buffer_is_sycl_split(src0->buffer);

                    if (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1 && src0->ne[3] == 1 && src1->ne[3] == 1) {
                         break;
                    }
                    if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
                        break;
                    }
                    if (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
                        break;
                    }

                    if (use_dequantize_mul_mat_vec || use_mul_mat_vec_q || use_mul_mat_q) {
                        break;
                    }

#ifdef SYCL_EXT_ONEAPI_MATRIX
                    // Check if XMX is available and types are supported
                    if (xmx_gemm_available(ctx.stream())) {
                        bool xmx_supported = false;
                        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                            xmx_supported = true;
                        } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {
                            xmx_supported = true;
                        }
                        
                        if (xmx_supported) {
                            break; // Compatible
                        }
                    }
#endif

                    // Note: oneDNN Graph API (DnnlGraphWrapper) is NOT compatible with SYCL command graphs
                    // because oneDNN Graph's cp.execute() creates internal SYCL events that break graph recording.
                    // The error "Graph nodes cannot depend on events from outside the graph" occurs.
                    // To enable SYCL graphs for F16/F32 GEMM, a custom graph-compatible kernel would be needed.

                    // oneMKL GEMM operations internally create events and call wait() which cannot
                    // be used during SYCL graph recording. This is a fundamental limitation of oneMKL.
                    //
                    // To enable graphs for this MUL_MAT, one of these conditions must be met:
                    // 1. Use quantized weights (Q4_K, Q8_0, etc.) with batch size 1 -> uses mul_mat_vec_q kernels
                    // 2. Use F16 weights with specific tensor layouts (see conditions above)
                    // 3. Implement a graph-compatible custom GEMM kernel (see ggml-sycl/gemm_tiled.hpp)
                    // 4. Enable oneDNN with Graph API for F16/F32 -> uses DnnlGraphWrapper
                    //
                    // Current tensor: src0=%s [%ldx%ld], src1=F32 [%ldx%ld], batch=%ld
                    
                    // Fallback to graph-compatible Tiled GEMM if allowed
                    // ggml_sycl_mul_mat() will pick the tiled kernel if force_graph_compatible is set.
                    // Check if the tiled GEMM supports this type combination:
                    // - F32 x F32 -> F32: supported
                    // - F16 x F32 -> F32: supported (converts F32 to F16 temp)
                    // - BF16 x F32 -> F32: supported (converts F32 to BF16 temp)
                    // - MXFP4 x F32 -> F32: supported (dequantizes MXFP4)
                    // - F16 x F16 -> F32: NOT supported (src1_ddf_i is float, not f16)
                    // - BF16 x BF16 -> F32: NOT supported (same reason)
                    // Check if the tiled GEMM supports this type combination
                    // Currently only F32xF32 is fully tested for graph compatibility
                    bool tiled_gemm_supported = false;
                    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
                        tiled_gemm_supported = true;
                    }
                    // F16xF32, BF16xF32, MXFP4xF32 fall back to oneMKL (not graph compatible)

                    if (tiled_gemm_supported) {
                        break; // Proceed with graph
                    } else {
                        GGML_LOG_DEBUG("%s: disabling SYCL graphs for unsupported tiled GEMM type combo (src0=%s, src1=%s)\n",
                            __func__, ggml_type_name(src0->type), ggml_type_name(src1->type));
                        return graph_compat_t::DISABLED;
                    } 
                }
        }
    }

    // Determine single vs multi-device mode
    if (ggml_sycl_info().device_count > 1 && has_split_buffers) {
        // Multi-device with split buffers: use per-device graphs
        GGML_LOG_INFO("%s: using multi-device SYCL graphs (%d devices)\n",
                      __func__, ggml_sycl_info().device_count);
        return graph_compat_t::MULTI_DEVICE;
    } else if (ggml_sycl_info().device_count > 1) {
        // Multi-device without split buffers: can still use single graph on primary device
        // But be conservative for now
        GGML_LOG_INFO("%s: multi-device detected but no split buffers, using single-device graph\n", __func__);
        return graph_compat_t::SINGLE_DEVICE;
    }

    return graph_compat_t::SINGLE_DEVICE;
}

// Check if a node involves split buffers (multi-device operation)
static bool node_uses_split_buffer_mdg(const ggml_tensor * node) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->buffer) {
            if (ggml_backend_buffer_is_sycl_split(node->src[i]->buffer)) {
                return true;
            }
        }
    }
    return false;
}

// Partition cgraph nodes by device
// Returns map of device_id -> vector of node indices
static std::map<int, std::vector<int>> partition_nodes_by_device(
    ggml_backend_sycl_context & ctx,
    ggml_cgraph * cgraph
) {
    std::map<int, std::vector<int>> partitions;
    int device_count = ggml_sycl_info().device_count;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node_uses_split_buffer_mdg(node)) {
            // Split buffer nodes execute on ALL devices that have a portion
            // The actual mul_mat split logic handles per-device row ranges
            for (int d = 0; d < device_count; d++) {
                partitions[d].push_back(i);
            }
        } else {
            // Single-device node - assign to context's device
            partitions[ctx.device].push_back(i);
        }
    }

    return partitions;
}

// Record and execute per-device graphs for multi-GPU setups
static ggml_status ggml_backend_sycl_multi_device_graph_compute(
    ggml_backend_sycl_context & ctx,
    ggml_cgraph * cgraph
) {
    int device_count = ggml_sycl_info().device_count;

    GGML_LOG_INFO("[SYCL-MULTI-GRAPH] Starting multi-device graph compute (%d devices, %d nodes)\n",
                  device_count, cgraph->n_nodes);

    // Partition nodes by device
    auto partitions = partition_nodes_by_device(ctx, cgraph);

    // Log partition info
    for (auto & [device_id, node_indices] : partitions) {
        GGML_LOG_DEBUG("[SYCL-MULTI-GRAPH] Device %d: %zu nodes\n", device_id, node_indices.size());
    }

    // Phase 1: Wait for all queues to be idle before recording
    for (int d = 0; d < device_count; d++) {
        if (partitions.find(d) != partitions.end() && !partitions[d].empty()) {
            ctx.stream(d, 0)->wait();
        }
    }

    // Phase 2: Record and finalize graph for each device
    std::map<int, sycl_ex::command_graph<sycl_ex::graph_state::modifiable>> modifiable_graphs;

    for (auto & [device_id, node_indices] : partitions) {
        if (node_indices.empty()) continue;

        ggml_sycl_set_device(device_id);
        queue_ptr stream = ctx.stream(device_id, 0);

        // Check graph support on this device
        if (!dpct::get_device(device_id).has(sycl::aspect::ext_oneapi_limited_graph)) {
            GGML_LOG_WARN("[SYCL-MULTI-GRAPH] Device %d does not support graphs, falling back\n", device_id);
            // Fall back to non-graph execution for this device's nodes
            for (int node_idx : node_indices) {
                ggml_tensor * node = cgraph->nodes[node_idx];
                bool ok = ggml_sycl_compute_forward(ctx, node);
                GGML_ASSERT(ok);
            }
            continue;
        }

        GGML_LOG_DEBUG("[SYCL-MULTI-GRAPH] Recording graph for device %d (%zu nodes)\n",
                       device_id, node_indices.size());

        // Create modifiable graph for this device
        sycl_ex::command_graph device_graph(*stream,
            {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        device_graph.begin_recording(*stream);

        // Record nodes for this device
        for (int node_idx : node_indices) {
            ggml_tensor * node = cgraph->nodes[node_idx];
            bool ok = ggml_sycl_compute_forward(ctx, node);
            if (!ok) {
                GGML_LOG_ERROR("[SYCL-MULTI-GRAPH] Failed to compute node %s\n", node->name);
            }
            GGML_ASSERT(ok);
        }

        device_graph.end_recording();
        modifiable_graphs.emplace(device_id, std::move(device_graph));
    }

    // Phase 3: Finalize and cache executable graphs
    for (auto & [device_id, mod_graph] : modifiable_graphs) {
        ggml_sycl_set_device(device_id);

        bool supports_update = dpct::get_device(device_id).has(sycl::aspect::ext_oneapi_graph);

        auto & cached_graph = ctx.per_device_exec_graphs[device_id];

        if (!cached_graph || !supports_update) {
            auto exec = supports_update ?
                mod_graph.finalize(sycl_ex::property::graph::updatable{}) :
                mod_graph.finalize();
            cached_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(std::move(exec));
            GGML_LOG_DEBUG("[SYCL-MULTI-GRAPH] Finalized new graph for device %d\n", device_id);
        } else {
            try {
                cached_graph->update(mod_graph);
                GGML_LOG_DEBUG("[SYCL-MULTI-GRAPH] Updated existing graph for device %d\n", device_id);
            } catch (sycl::exception const & e) {
                GGML_LOG_DEBUG("[SYCL-MULTI-GRAPH] Graph update failed on device %d: %s, re-finalizing\n",
                               device_id, e.what());
                auto exec = mod_graph.finalize({sycl_ex::property::graph::updatable{}});
                cached_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(std::move(exec));
            }
        }
    }

    // Phase 4: Execute graphs with inter-device synchronization
    // Execute sequentially with barriers to ensure correct ordering.
    // Future optimization: analyze dependencies and overlap where possible.

    std::vector<sycl::event> prev_events;

    for (auto & [device_id, cached_graph] : ctx.per_device_exec_graphs) {
        if (!cached_graph) continue;

        ggml_sycl_set_device(device_id);
        queue_ptr stream = ctx.stream(device_id, 0);

        // Wait for previous device(s) to complete
        if (!prev_events.empty()) {
            stream->ext_oneapi_submit_barrier(prev_events);
        }

        // Execute this device's graph
        stream->ext_oneapi_graph(*cached_graph);

        // Record completion event for next device
        sycl::event completion = stream->ext_oneapi_submit_barrier();
        prev_events.clear();
        prev_events.push_back(completion);

        GGML_LOG_DEBUG("[SYCL-MULTI-GRAPH] Executed graph on device %d\n", device_id);
    }

    ctx.multi_device_graphs_initialized = true;

    GGML_LOG_INFO("[SYCL-MULTI-GRAPH] Multi-device graph execution complete\n");

    return GGML_STATUS_SUCCESS;
}

#endif // GGML_SYCL_GRAPH

static ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

 #ifdef GGML_SYCL_GRAPH
    graph_compat_t graph_mode = g_ggml_sycl_disable_graph ?
        graph_compat_t::DISABLED : check_graph_compatibility(*sycl_ctx, cgraph);

    if (graph_mode == graph_compat_t::MULTI_DEVICE) {
        // ===== MULTI-DEVICE GRAPH PATH =====
        // Use per-device graphs for multi-GPU with split buffers
        GGML_SYCL_DEBUG("[SYCL-MULTI-GRAPH] Starting multi-device graph compute (n_nodes=%d)\n", cgraph->n_nodes);

        sycl_ctx->force_graph_compatible = true;
        ggml_status status = ggml_backend_sycl_multi_device_graph_compute(*sycl_ctx, cgraph);
        sycl_ctx->force_graph_compatible = false;
        return status;

    } else if (graph_mode == graph_compat_t::SINGLE_DEVICE) {
        // ===== SINGLE-DEVICE GRAPH PATH (with hash-based caching) =====
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Starting graph compute (n_nodes=%d)\n", cgraph->n_nodes);
        struct timeval tv_start, tv_end;
        gettimeofday(&tv_start, NULL);

        // Ensure the queue is idle before recording to avoid "depend on events from outside" error.
        // This is a workaround for SYCL graph implicit dependencies on in-order queues.
        sycl_ctx->stream()->wait();
        gettimeofday(&tv_end, NULL);
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Queue wait took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
        gettimeofday(&tv_start, NULL);

        const bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        if (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            return GGML_STATUS_SUCCESS;
        }
        gettimeofday(&tv_end, NULL);
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Graph support check took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
        gettimeofday(&tv_start, NULL);

        // Compute hash of graph topology for cache lookup
        const uint64_t graph_hash = compute_cgraph_hash(cgraph);
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Graph hash: 0x%016llx\n", (unsigned long long)graph_hash);

        // Check if we have a cached graph with matching topology
        auto cache_it = sycl_ctx->graph_cache.find(graph_hash);
        bool use_cached = (cache_it != sycl_ctx->graph_cache.end());

        if (use_cached) {
            // Use cached graph - just execute it directly
            GGML_SYCL_DEBUG("[SYCL-GRAPH] Cache hit, reusing existing graph\n");
            gettimeofday(&tv_end, NULL);
            GGML_SYCL_DEBUG("[SYCL-GRAPH] Cache lookup took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
            gettimeofday(&tv_start, NULL);

            // Still need to execute the compute to update tensor data pointers
            // For now, we re-record and update the cached graph
            sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
            model_sycl_graph.begin_recording(*(sycl_ctx->stream()));

            sycl_ctx->force_graph_compatible = true;
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            sycl_ctx->force_graph_compatible = false;

            model_sycl_graph.end_recording();

            try {
                cache_it->second->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Cache update success\n");
            } catch (sycl::exception const & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Cache update failed: %s, re-finalizing\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                cache_it->second = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(std::move(exec_graph));
            }

            gettimeofday(&tv_end, NULL);
            GGML_SYCL_DEBUG("[SYCL-GRAPH] finalize/update took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
            gettimeofday(&tv_start, NULL);

            sycl_ctx->stream()->ext_oneapi_graph(*(cache_it->second));
        } else {
            // Cache miss - record, finalize, and cache the new graph
            GGML_SYCL_DEBUG("[SYCL-GRAPH] Cache miss, creating new graph\n");

            // Evict oldest entry if cache is full
            if (sycl_ctx->graph_cache.size() >= sycl_ctx->MAX_GRAPH_CACHE_SIZE) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Cache full, evicting oldest entry\n");
                sycl_ctx->graph_cache.erase(sycl_ctx->graph_cache.begin());
            }

            sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});
            gettimeofday(&tv_end, NULL);
            GGML_SYCL_DEBUG("[SYCL-GRAPH] Graph construction took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
            gettimeofday(&tv_start, NULL);

            model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
            gettimeofday(&tv_end, NULL);
            GGML_SYCL_DEBUG("[SYCL-GRAPH] begin_recording took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
            gettimeofday(&tv_start, NULL);

            sycl_ctx->force_graph_compatible = true;
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            sycl_ctx->force_graph_compatible = false;

            gettimeofday(&tv_end, NULL);
            GGML_SYCL_DEBUG("[SYCL-GRAPH] compute_impl took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
            gettimeofday(&tv_start, NULL);

            model_sycl_graph.end_recording();
            gettimeofday(&tv_end, NULL);
            GGML_SYCL_DEBUG("[SYCL-GRAPH] end_recording took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
            gettimeofday(&tv_start, NULL);

            const bool graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);

            auto exec_graph = graph_update_support ?
                model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                model_sycl_graph.finalize();

            // Store in cache
            sycl_ctx->graph_cache[graph_hash] = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(std::move(exec_graph));

            gettimeofday(&tv_end, NULL);
            GGML_SYCL_DEBUG("[SYCL-GRAPH] finalize took %ld us (cache size: %zu)\n",
                (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec),
                sycl_ctx->graph_cache.size());
            gettimeofday(&tv_start, NULL);

            sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->graph_cache[graph_hash]));
        }
        gettimeofday(&tv_end, NULL);
        GGML_SYCL_DEBUG("[SYCL-GRAPH] ext_oneapi_graph took %ld us\n", (tv_end.tv_sec - tv_start.tv_sec)*1000000 + (tv_end.tv_usec - tv_start.tv_usec));
        gettimeofday(&tv_start, NULL);

        return GGML_STATUS_SUCCESS;
    }
    // graph_mode == DISABLED falls through to non-graph path
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    return GGML_STATUS_SUCCESS;
 }

static void ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
try
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    const queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static void ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) try {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    if (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } else
        GGML_ABORT("fatal error");
} catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

static ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_sycl_guid() {
    static ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    return &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

int ggml_backend_sycl_get_device_count() {
    return ggml_sycl_info().device_count;
}


// backend device

struct ggml_backend_sycl_device_context {
    int device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ggml_sycl_set_device(ctx->device);
    SYCL_CHECK(CHECK_TRY_ERROR(
    dpct::dev_mgr::instance().get_device(ctx->device).get_memory_info(*free, *total)));
}

static enum ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    return ggml_backend_sycl_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_sycl_host_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    int device = sycl_ctx->device;
    switch (op->op) {
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];

                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                ggml_type a_type = a->type;
                if (a_type == GGML_TYPE_IQ4_NL  || a_type == GGML_TYPE_IQ4_XS ||
                    a_type == GGML_TYPE_IQ3_XXS || a_type == GGML_TYPE_IQ3_S  ||
                    a_type == GGML_TYPE_IQ2_XXS || a_type == GGML_TYPE_IQ2_XS || a_type == GGML_TYPE_IQ2_S ||
                    a_type == GGML_TYPE_IQ1_S || a_type == GGML_TYPE_IQ1_M
                    ) {
                    if (b->ne[1] == 1 && ggml_nrows(b) > 1) {
                        return false;
                    }
                }
                ggml_type src0_type = op->src[0]->type;
                /*
                if (src0_type == GGML_TYPE_BF16 ) {
                    // TODO: support GGML_TYPE_BF16
                    // FIXME: keep a list of supported types to avoid breaking the backend when a new type is added
                    return false;
                }
                */

                // TODO: The configuration below needs more work to be supported with oneDNN
                if (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  return false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                if (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    return false;
                }
                return true;
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
         case GGML_OP_SET:
               return (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        case GGML_OP_SET_ROWS:
            {
                return ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                         op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q5_0 ||
                         op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_IQ4_NL) &&
                        (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32));
            }
            break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == src1_type && (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) && src0_type != GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if(src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                return false;
            }
        case GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32;
            }
        case GGML_OP_CONCAT:
        case GGML_OP_DUP:
        case GGML_OP_ARGMAX:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ADD_ID:
        case GGML_OP_SUB:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_REPEAT:
            return true;
        case GGML_OP_PAD_REFLECT_1D:
            return ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
#if defined (GGML_SYCL_F16)
            return ((op->type == GGML_TYPE_F32 || op->type == GGML_SYCL_F16) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_SYCL_F16) && (op->type == op->src[0]->type));
#else
            return (op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32) && (op->type == op->src[0]->type);
#endif
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_SCALE:
            return true;
        case GGML_OP_CONT:
            return op->src[0]->type != GGML_TYPE_BF16;
        case GGML_OP_TRI:
            {
                const ggml_tensor * src0 = op->src[0];
                return src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
            return true;
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST && !(op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS);
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return op->src[0]->ne[0] * sizeof(int) <=
                   ggml_sycl_info().devices[device].smpbo;
        case GGML_OP_TOP_K: {
            const ggml_tensor * src0 = op->src[0];
            const int k = op->ne[0];
            return src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= 32;
        }
        case GGML_OP_POOL_2D:
        case GGML_OP_ACC:
            return true;
        case GGML_OP_PAD:
            // TODO: add circular padding support for syscl, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
        case GGML_OP_GATED_LINEAR_ATTN:
            return true;
        case GGML_OP_SSM_CONV:
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_ROLL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        return false;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return buft_ctx->device == sycl_ctx->device;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    return get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    return nullptr;
#else
  sycl::event *event_ptr = new sycl::event();

  return new ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

static void ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  if (event == nullptr) {
    return;
  }

  if (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


static void ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) try {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static const ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

struct ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYCL_NAME;
}

static size_t ggml_backend_sycl_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    GGML_UNUSED(reg);

    if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        return (void *)ggml_backend_sycl_split_buffer_type;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_count,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_sycl_reg_context * ctx = new ggml_backend_sycl_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_sycl_info().device_count; i++) {
                ggml_backend_sycl_device_context * dev_ctx = new ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_sycl_init(int device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = new ggml_backend_sycl_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    };

    ggml_backend_t sycl_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    return sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
