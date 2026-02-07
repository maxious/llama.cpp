#ifndef GGML_SYCL_FATTN_COMMON_HPP
#define GGML_SYCL_FATTN_COMMON_HPP

#include <cstdint>
#include <cstring>
#include <sycl/sycl.hpp>

// Full header for ggml_tensor definition
#include "common.hpp"

// ============================================================================
// Low-level memory utilities
// ============================================================================

template<int N>
inline void ggml_sycl_memcpy(float* dst, const float* src) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

inline void ggml_sycl_load_vec4(sycl::float4& dst, const float* src) {
    dst = sycl::float4(src[0], src[1], src[2], src[3]);
}

inline void ggml_sycl_store_vec4(float* dst, const sycl::float4& src) {
    dst[0] = src.x();
    dst[1] = src.y();
    dst[2] = src.z();
    dst[3] = src.w();
}

inline void ggml_sycl_load_vec4_half(sycl::float4& dst, const sycl::half* src) {
    sycl::half h0 = src[0], h1 = src[1], h2 = src[2], h3 = src[3];
    dst = sycl::float4(static_cast<float>(h0), static_cast<float>(h1),
                       static_cast<float>(h2), static_cast<float>(h3));
}

// ============================================================================
// Flash Attention Common Definitions
// ============================================================================

enum class fattn_input_type {
    f32,
    f16
};

// Stride parameters for direct loading from ggml tensor layout
struct fattn_tensor_strides {
    int64_t q_stride_seq;    // stride between Q rows (sequence positions)
    int64_t q_stride_head;   // stride between Q heads
    int64_t k_stride_seq;    // stride between K rows
    int64_t k_stride_head;   // stride between K heads
    int64_t v_stride_seq;    // stride between V rows
    int64_t v_stride_head;   // stride between V heads
    int64_t o_stride_seq;    // stride between O rows (output)
    int64_t o_stride_head;   // stride between O heads
};

constexpr int FATTN_BLOCK_R = 32;
constexpr int FATTN_BLOCK_C = 32;

// Get optimal number of KV splits based on context length
inline int64_t get_kv_split_count(int64_t N_kv, int64_t n_heads) {
    constexpr int64_t MIN_KV_FOR_SPLIT = 256;
    if (N_kv < MIN_KV_FOR_SPLIT) return 1;
    constexpr int64_t MIN_KV_PER_SPLIT = 256;
    constexpr int64_t MAX_SPLITS = 16;
    int64_t max_splits_by_kv = N_kv / MIN_KV_PER_SPLIT;
    int64_t splits = std::min(max_splits_by_kv, MAX_SPLITS);
    if (splits >= 16) splits = 16;
    else if (splits >= 8) splits = 8;
    else if (splits >= 4) splits = 4;
    else if (splits >= 2) splits = 2;
    else splits = 1;
    (void)n_heads;
    return splits;
}

inline int64_t get_padded_head_size(int64_t head_size) {
    if (head_size <= 32) return 32;
    if (head_size <= 64) return 64;
    if (head_size <= 80) return 80;
    if (head_size <= 96) return 96;
    if (head_size <= 112) return 112;
    if (head_size <= 128) return 128;
    if (head_size <= 256) return 256;
    if (head_size <= 512) return 512;
    if (head_size <= 576) return 576;
    return 0;
}

inline bool is_head_size_supported(int64_t head_size) {
    return get_padded_head_size(head_size) > 0;
}

// Compute stride parameters for direct loading from ggml tensor layout
inline fattn_tensor_strides compute_tensor_strides(const ggml_tensor * Q, const ggml_tensor * K,
                                                    const ggml_tensor * V, const ggml_tensor * dst) {
    fattn_tensor_strides s;

    // Q strides in elements
    const size_t q_elem_size = ggml_type_size(Q->type);
    s.q_stride_seq = Q->nb[1] / q_elem_size;
    s.q_stride_head = Q->nb[2] / q_elem_size;

    // K strides in elements
    const size_t k_elem_size = ggml_type_size(K->type);
    s.k_stride_seq = K->nb[1] / k_elem_size;
    s.k_stride_head = K->nb[2] / k_elem_size;

    // V strides in elements
    const size_t v_elem_size = ggml_type_size(V->type);
    s.v_stride_seq = V->nb[1] / v_elem_size;
    s.v_stride_head = V->nb[2] / v_elem_size;

    // Output strides in elements (always F32)
    // Output layout is permuted: ne = [DV, n_heads, N, batch]
    // nb[1] = stride between heads (dim 1), nb[2] = stride between seq positions (dim 2)
    s.o_stride_head = dst->nb[1] / sizeof(float);
    s.o_stride_seq  = dst->nb[2] / sizeof(float);

    return s;
}

// ============================================================================
// Combine Splits Kernel (Reduction Phase)
// ============================================================================
template <int DV>
inline void flash_attn_combine_splits_kernel(
    sycl::nd_item<2> it,
    const float * partials,
    float * dst,
    int n_splits,
    int n_heads,
    int N,
    int partial_size,
    int o_head_stride,
    int o_row_stride,
    const float * sinks_d = nullptr
) {
    int q_idx = it.get_group(0);
    int tid = it.get_local_id(1);
    if (q_idx >= n_heads * N) return;
    int head = q_idx / N;
    int q = q_idx % N;
    const float * q_partials = partials + (int64_t)q_idx * n_splits * partial_size;
    float m_max = -1.0e20f;
    for (int i = 0; i < n_splits; ++i) {
        float m = q_partials[i * partial_size + 0];
        m_max = sycl::fmax(m_max, m);
    }
    float l_final = 0.0f;
    float scales[32];
    for (int i = 0; i < n_splits; ++i) {
        float m = q_partials[i * partial_size + 0];
        float l = q_partials[i * partial_size + 1];
        float scale = sycl::exp(m - m_max);
        scales[i] = scale;
        l_final += l * scale;
    }

    // Apply attention sinks if present (like a virtual token with zero value)
    float ms_factor = 1.0f;  // multiplier for O when sink > m_max
    if (sinks_d != nullptr) {
        float sink = sinks_d[head];  // sink per head
        if (sink > m_max) {
            ms_factor = sycl::exp(m_max - sink);
            l_final = l_final * ms_factor + 1.0f;
        } else {
            l_final += sycl::exp(sink - m_max);
        }
    }

    for (int d = tid; d < DV; d += it.get_local_range(1)) {
        float o_sum = 0.0f;
        for (int i = 0; i < n_splits; ++i) {
            float o_val = q_partials[i * partial_size + 2 + d];
            o_sum += o_val * scales[i];
        }
        // If sink was greater than previous max, scale down the O sum to match
        if (sinks_d != nullptr && ms_factor != 1.0f) {
            o_sum *= ms_factor;
        }
        float res = o_sum / (l_final > 1e-10f ? l_final : 1.0f);
        dst[head * o_head_stride + q * o_row_stride + d] = res;
    }
}

// ============================================================================
// XMX (Cooperative Matrix) Support Detection
// ============================================================================
#ifdef SYCL_EXT_COOPERATIVE_MATRICES
#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
namespace cm = sycl::ext::oneapi::experimental::matrix;
using xmx_bfloat16 = sycl::ext::oneapi::bfloat16;
enum class xmx_tile_kind { tile_dg2, tile_pvc };
inline bool ggml_sycl_has_coopmat_support(sycl::device device) {
    return device.has(sycl::aspect::ext_intel_gpu_eu_simd_width) &&
           device.has(sycl::aspect::ext_intel_matrix);
}
#endif // SYCL_EXT_COOPERATIVE_MATRICES

#endif // GGML_SYCL_FATTN_COMMON_HPP
