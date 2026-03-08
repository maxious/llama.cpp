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

#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "common.hpp"

// XMX-accelerated flash attention for prompt processing.
// Uses joint_matrix FP16 for Q*K^T and P*V matmuls with online softmax in FP32.
// Targets Intel Xe GPUs with ext_intel_matrix support.

// Check if XMX flash attention is available for this device and head dimensions
bool ggml_sycl_fattn_xmx_supported(ggml_backend_sycl_context & ctx, const ggml_tensor * dst);

// Launch XMX flash attention kernel
void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_HPP
