#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"
#include "fattn_common.hpp"

// Flash attention operation for SYCL backend
// This implements the Flash Attention algorithm optimized for SYCL devices
void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Fused single-kernel flash attention (runtime DQK, DV)
// Template parameters: QType, KVType, MaskT
template <typename QType, typename KVType = QType, typename MaskT = float>
void ggml_sycl_op_flash_attn_fused(
    sycl::queue* stream,
    const QType* Q, const KVType* K, const KVType* V,
    float* O,
    int N, int N_kv,
    int n_heads, int n_kv_heads, int gqa_ratio,
    float scale,
    const MaskT* mask, int64_t mask_stride,
    const float* sinks,
    const fattn_tensor_strides& strides,
    int DQK, int DV);

// Check if flash attention is supported for given tensor
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
