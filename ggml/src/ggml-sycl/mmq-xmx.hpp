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

#ifndef GGML_SYCL_MMQ_XMX_HPP
#define GGML_SYCL_MMQ_XMX_HPP

#include "common.hpp"

// XMX-accelerated quantized matrix multiplication using joint_matrix API.
// Dequantizes to raw int8 in SLM, runs XMX int8 dot products, then applies
// quantization scales in a float epilogue.

// Check if the device supports XMX int8 joint_matrix operations
bool ggml_sycl_supports_xmx_mmq(const sycl::device & dev, ggml_type type);

// XMX-accelerated mul_mat_q dispatch — called from ggml_sycl_op_mul_mat_q
// when XMX is available for the given quant type.
void ggml_sycl_op_mul_mat_q_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    const char * src0_dd_i,
    const float * src1_ddf_i,
    const char * src1_ddq_i,
    float * dst_dd_i,
    const int64_t row_low,
    const int64_t row_high,
    const int64_t src1_ncols,
    const int64_t src1_padded_row_size,
    const dpct::queue_ptr & stream);

#endif // GGML_SYCL_MMQ_XMX_HPP
