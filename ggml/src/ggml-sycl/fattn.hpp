#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"
#include "fattn_common.hpp"

// Flash attention operation for SYCL backend
// This implements the Flash Attention algorithm optimized for SYCL devices
void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Fused single-kernel flash attention (runtime DQK, DV)
// Template parameter T: element type of Q, K, V (float, sycl::half, sycl::ext::oneapi::bfloat16)
template <typename T>
void ggml_sycl_op_flash_attn_fused(
    sycl::queue* stream,
    const T* Q, const T* K, const T* V,
    float* O,
    int N, int N_kv,
    int n_heads, int n_kv_heads, int gqa_ratio,
    float scale,
    const float* mask, int64_t mask_stride,
    const float* sinks,
    const fattn_tensor_strides& strides,
    int DQK, int DV);

// Check if flash attention is supported for given tensor
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
