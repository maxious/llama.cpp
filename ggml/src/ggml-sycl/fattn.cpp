#include "./fattn.hpp"
#include "./fattn_kernel.hpp"
#include "./fattn_common.hpp"
#include "./common.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <sycl/sycl.hpp>
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif

// ============================================================================
// Flash Attention with KV-Split (Flash Decoding)
// ============================================================================
// The oneMKL flash attention implementation uses optional KV-dimension splitting
// for improved GPU utilization. For short contexts (N_kv < 256), it uses a single
// pass (n_splits=1). For longer contexts, it splits KV across multiple chunks.
//
// Flash Decoding splits the KV dimension across multiple workgroups:
// 1. Each workgroup processes a chunk of KV and produces partial results:
//    - M_partial: max logit for the chunk
//    - L_partial: sum of exp(logits - M_partial) for the chunk  
//    - O_partial: weighted sum of V (unnormalized) for the chunk
// 2. A reduction kernel merges partials using online softmax:
//    - M_new = max(M1, M2)
//    - O_new = O1 * exp(M1 - M_new) + O2 * exp(M2 - M_new)
//    - L_new = L1 * exp(M1 - M_new) + L2 * exp(M2 - M_new)
//    - Final: O = O_new / L_new
//
// Reference: https://pytorch.org/blog/flash-decoding/
// ============================================================================

// Get optimal number of KV splits based on context length and device
// Returns 1 for short contexts (no splitting needed), otherwise 2-16 splits
inline int64_t get_kv_split_count(int64_t N_kv, int64_t n_heads) {
    // For very short contexts, no splitting needed - single pass is optimal
    constexpr int64_t MIN_KV_FOR_SPLIT = 256;
    if (N_kv < MIN_KV_FOR_SPLIT) {
        return 1;  // No splitting for short contexts
    }
    
    // Target: enough splits to saturate GPU, but not too many to cause overhead
    // Each split should process at least 256 KV positions for efficiency
    constexpr int64_t MIN_KV_PER_SPLIT = 256;
    constexpr int64_t MAX_SPLITS = 16;
    
    int64_t max_splits_by_kv = N_kv / MIN_KV_PER_SPLIT;
    int64_t splits = std::min(max_splits_by_kv, MAX_SPLITS);
    
    // Round to power of 2 for cleaner division
    if (splits >= 16) splits = 16;
    else if (splits >= 8) splits = 8;
    else if (splits >= 4) splits = 4;
    else if (splits >= 2) splits = 2;
    else splits = 1;
    
    (void)n_heads; // May be used in future for more sophisticated heuristics
    return splits;
}

// Block sizes for flash attention tiling
// These are regular constants, not macros, to avoid conflicts with function parameters
constexpr int FATTN_BLOCK_R = 32;  // Br
constexpr int FATTN_BLOCK_C = 32;  // Bc

// Head size padding support
// Returns the next supported head size for padding, or 0 if not paddable
inline int64_t get_padded_head_size(int64_t head_size) {
    // Supported sizes: 32, 64, 80, 96, 112, 128, 256, 512, 576
    // 576 added for GLM-4.7-Flash which has K head size 576
    if (head_size <= 32) return 32;
    if (head_size <= 64) return 64;
    if (head_size <= 80) return 80;
    if (head_size <= 96) return 96;
    if (head_size <= 112) return 112;
    if (head_size <= 128) return 128;
    if (head_size <= 256) return 256;
    if (head_size <= 512) return 512;
    if (head_size <= 576) return 576;
    return 0;  // Not supported
}

// Check if a head size can be used (either directly or with padding)
inline bool is_head_size_supported(int64_t head_size) {
    return get_padded_head_size(head_size) > 0;
}


// Check if device supports XMX (cooperative matrix) for flash attention
// All Intel GPUs with XMX support can use the fused flash attention kernel,
// but with different tile sizes:
// - DG2/Arc B60/Battlemage (Xe2): 8x16x16 tiles with bf16 A/B
// - PVC (Ponte Vecchio): 8x16x16 tiles
// - Future GPUs: varies
//
// XMX is enabled by default when the device supports it.
// Set GGML_SYCL_FLASH_ATTN_XMX=0 to disable and fall back to oneMKL.
inline bool ggml_sycl_flash_attn_has_xmx(sycl::device device) {
#ifdef SYCL_EXT_COOPERATIVE_MATRICES
    // Check for basic cooperative matrix support
    if (!device.has(sycl::aspect::ext_intel_gpu_eu_simd_width) ||
        !device.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }
    // Check environment variable to disable XMX kernel (enabled by default)
    static bool xmx_enabled = true;
    static bool env_checked = false;
    if (!env_checked) {
        const char* env = getenv("GGML_SYCL_FLASH_ATTN_XMX");
        if (env != nullptr && strcmp(env, "0") == 0) {
            xmx_enabled = false;
            GGML_SYCL_DEBUG("ggml_sycl: XMX flash attention DISABLED by environment variable\n");
        }
        env_checked = true;
    }
    return xmx_enabled;
#else
    return false;
#endif
}

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
// Get the appropriate tile kind for flash attention XMX kernel
// DG2/Arc uses 8x8 tiles, PVC uses 16x16 tiles
inline xmx_tile_kind ggml_sycl_flash_attn_get_tile_kind(sycl::device device) {
    return ggml_sycl_get_tile_kind(device);
}
#endif

// Check if device should use oneMKL BLAS for flash attention (fallback path)
// oneMKL is only used when XMX is not available or explicitly disabled.
// Set GGML_SYCL_FLASH_ATTN_MKL=1 to force oneMKL even when XMX is available.
inline bool ggml_sycl_flash_attn_use_mkl(sycl::device device) {
#ifdef GGML_SYCL_USE_INTEL_ONEMKL
    // Check if user explicitly wants oneMKL
    static bool mkl_forced = false;
    static bool env_checked = false;
    if (!env_checked) {
        const char* env = getenv("GGML_SYCL_FLASH_ATTN_MKL");
        mkl_forced = (env != nullptr && strcmp(env, "1") == 0);
        env_checked = true;
        if (mkl_forced) {
            GGML_SYCL_DEBUG("ggml_sycl: oneMKL flash attention FORCED by environment variable\n");
        }
    }
    if (mkl_forced) {
        return true;
    }
    // Only use MKL if XMX is not available
    if (!ggml_sycl_flash_attn_has_xmx(device)) {
        return true;
    }
#endif
    return false;
}

// Check if direct stride loading is enabled via environment variable
// GGML_SYCL_FLASH_ATTN_DIRECT=1 to enable (disabled by default for safety)
inline bool ggml_sycl_flash_attn_use_direct() {
    static bool direct_enabled = false;
    static bool env_checked = false;
    if (!env_checked) {
        const char* env = getenv("GGML_SYCL_FLASH_ATTN_DIRECT");
        direct_enabled = (env != nullptr && strcmp(env, "1") == 0);
        env_checked = true;
        if (direct_enabled) {
            GGML_SYCL_DEBUG("ggml_sycl: Direct stride flash attention ENABLED by environment variable\n");
        }
    }
    return direct_enabled;
}

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
// Check if tensor layout is compatible with direct stride loading
// Requirements:
// - Inner dimension (dim=0) must be contiguous (nb[0] == element_size)
// - For Q/K/V: layout is [dim, seq, head] with arbitrary strides for seq and head
// Returns true if direct loading is possible
inline bool can_use_direct_loading(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V,
                                    const ggml_tensor * dst) {
    // Check Q/K/V inner dimension is contiguous
    const size_t q_elem_size = ggml_type_size(Q->type);
    const size_t k_elem_size = ggml_type_size(K->type);
    const size_t v_elem_size = ggml_type_size(V->type);
    const size_t o_elem_size = ggml_type_size(dst->type);
    
    if (Q->nb[0] != q_elem_size) return false;
    if (K->nb[0] != k_elem_size) return false;
    if (V->nb[0] != v_elem_size) return false;
    if (dst->nb[0] != o_elem_size) return false;
    
    // All Q/K/V must be same type (F16 or F32)
    const bool is_f16 = (Q->type == GGML_TYPE_F16);
    const bool is_f32 = (Q->type == GGML_TYPE_F32);
    if (!is_f16 && !is_f32) return false;
    
    // K and V must match Q's type (or mixed F32 Q with F16 K/V is also ok)
    if (is_f16) {
        if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) return false;
    } else {
        // For F32 Q, K/V can be F32 or F16
        if (K->type != GGML_TYPE_F32 && K->type != GGML_TYPE_F16) return false;
        if (V->type != GGML_TYPE_F32 && V->type != GGML_TYPE_F16) return false;
    }
    
    // Output must be F32
    if (dst->type != GGML_TYPE_F32) return false;
    
    return true;
}

// Compute stride parameters for direct loading
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
    s.o_stride_seq = dst->nb[1] / sizeof(float);
    s.o_stride_head = dst->nb[2] / sizeof(float);
    
    return s;
}
#endif // SYCL_EXT_COOPERATIVE_MATRICES

bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale, max_bias, logit_softcap;

    std::memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    // Sinks (attention sinks / StreamingLLM) supported in MKL path
    // Sinks tensor has shape [n_heads] - one float per head
    if (sinks != nullptr && sinks->type != GGML_TYPE_F32) {
        return false;
    }

    if (Q == nullptr || K == nullptr || V == nullptr) {
        return false;
    }

    // Batch > 1 not yet supported in oneMKL flash attention path
    if (Q->ne[3] > 1) {
        return false;
    }

    if (mask != nullptr && mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) {
        return false;
    }
    
    const bool is_all_f32 = (Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && V->type == GGML_TYPE_F32);
    const bool is_all_f16 = (Q->type == GGML_TYPE_F16 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    const bool is_mixed_f32_q = (Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    if (!is_all_f32 && !is_all_f16 && !is_mixed_f32_q) {
        return false;
    }

    const size_t q_elem_size = ggml_type_size(Q->type);
    const size_t k_elem_size = ggml_type_size(K->type);
    const size_t v_elem_size = ggml_type_size(V->type);
    if (Q->nb[0] != q_elem_size || K->nb[0] != k_elem_size || V->nb[0] != v_elem_size) {
        return false;
    }

    int64_t DQK = Q->ne[0];
    int64_t DV  = V->ne[0];

    if (!is_head_size_supported(DQK) || !is_head_size_supported(DV)) {
        return false;
    }

    // Only DQK==DV or specifically supported DQK/DV combinations (e.g., 576/512) are handled
    if (DQK != DV && !(DQK == 576 && DV == 512)) {
        return false;
    }

    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];

    if (n_heads % n_kv_heads != 0) {
        return false;
    }

    const int gqa_ratio = n_heads / n_kv_heads;

    if (gqa_ratio > 8 && gqa_ratio % 4 != 0) {
        return false;
    }
    
    if (gqa_ratio > 32) {
        return false;
    }

    return true;
}

template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];

    const bool is_f16 = (Q->type == GGML_TYPE_F16);
    const bool V_is_K_view = V->view_src && V->view_offs == 0 && (V->view_src == K || V->view_src == K->view_src);

    const float * Q_d_f32;
    const float * K_d_f32;
    const float * V_d_f32;
    float * Q_d_f32_alloc = nullptr;
    float * K_d_f32_alloc = nullptr;
    float * V_d_f32_alloc = nullptr;

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];
    const int64_t N_kv = K->ne[1];  // Key/Value sequence length (can differ from N)
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;  // GQA ratio

    const ptrdiff_t q_row_stride = Q->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t k_row_stride = K->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t v_row_stride = V->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t o_row_stride = dst->nb[1] / (ptrdiff_t)sizeof(float);

    // Handle FP16 by dequantizing to F32 first
    if (is_f16) {
        const sycl::half * Q_d = (const sycl::half *) Q->data;
        const sycl::half * K_d = (const sycl::half *) K->data;

        // Allocate F32 buffers on device
        Q_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
        K_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DQK * n_kv_heads * sizeof(float), *stream);
        if (!V_is_K_view) {
            V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
        }

        // Get strides in elements for FP16
        const ptrdiff_t q_row_stride_f16 = Q->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t k_row_stride_f16 = K->nb[1] / (ptrdiff_t)sizeof(sycl::half);

        // Get head strides from tensor layout
        const ptrdiff_t q_head_stride_f16 = Q->nb[2] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t k_head_stride_f16 = K->nb[2] / (ptrdiff_t)sizeof(sycl::half);

        // Dequantize Q heads (all n_heads heads)
        for (int64_t head = 0; head < n_heads; ++head) {
            const int64_t n_elements = N * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / DQK;
                        const int64_t col = idx % DQK;
                        Q_d_f32_alloc[head * N * DQK + idx] = static_cast<float>(
                            Q_d[head * q_head_stride_f16 + row * q_row_stride_f16 + col]);
                    }
                });
            });
        }

        // Dequantize K heads
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N_kv * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / DQK;
                        const int64_t col = idx % DQK;
                        K_d_f32_alloc[head * N_kv * DQK + idx] = static_cast<float>(
                            K_d[head * k_head_stride_f16 + row * k_row_stride_f16 + col]);
                    }
                });
            });
        }

        if (V_is_K_view && DQK == DV) {
            // V is a view of K and they have the same head dimension - can reuse directly
            V_d_f32_alloc = K_d_f32_alloc;
        } else if (V_is_K_view && DQK != DV) {
            // MLA case: V is a view of K but with different head dimension
            // K layout is [kv_lora_scaled (DV), pe (DQK-DV)] = DQK total
            // V uses only the first DV elements of each K row
            V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
            for (int64_t head = 0; head < n_kv_heads; ++head) {
                const int64_t n_elements = N_kv * DV;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t row = idx / DV;
                            const int64_t col = idx % DV;
                            // Read from K's first DV elements per row
                            V_d_f32_alloc[head * N_kv * DV + idx] = K_d_f32_alloc[head * N * DQK + row * DQK + col];
                        }
                    });
                });
            }
        } else {
            const sycl::half * V_d = (const sycl::half *) V->data;
            const ptrdiff_t v_row_stride_f16 = V->nb[1] / (ptrdiff_t)sizeof(sycl::half);
            const ptrdiff_t v_head_stride_f16 = V->nb[2] / (ptrdiff_t)sizeof(sycl::half);

            // Dequantize V heads
            for (int64_t head = 0; head < n_kv_heads; ++head) {
                const int64_t n_elements = N_kv * DV;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t row = idx / DV;
                            const int64_t col = idx % DV;
                            V_d_f32_alloc[head * N_kv * DV + idx] = static_cast<float>(
                                V_d[head * v_head_stride_f16 + row * v_row_stride_f16 + col]);
                        }
                    });
                });
            }
        }

        // Wait for dequantization to complete before proceeding
        stream->wait();

        Q_d_f32 = Q_d_f32_alloc;
        K_d_f32 = K_d_f32_alloc;
        V_d_f32 = V_d_f32_alloc;
    } else {
        // F32 case - direct pointer cast
        Q_d_f32 = (const float *) Q->data;
        K_d_f32 = (const float *) K->data;
        if (V_is_K_view && DQK == DV) {
            V_d_f32 = K_d_f32;
        } else if (V_is_K_view && DQK != DV) {
            // MLA case: V is a view of K but with different head dimension
            // Need to extract first DV elements from each K row
            V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
            const float * K_f32 = (const float *) K->data;
            const ptrdiff_t k_row_stride = K->nb[1] / (ptrdiff_t)sizeof(float);
            const ptrdiff_t k_head_stride = K->nb[2] / (ptrdiff_t)sizeof(float);
            for (int64_t head = 0; head < n_kv_heads; ++head) {
                const int64_t n_elements = N_kv * DV;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t row = idx / DV;
                            const int64_t col = idx % DV;
                            V_d_f32_alloc[head * N_kv * DV + idx] = 
                                K_f32[head * k_head_stride + row * k_row_stride + col];
                        }
                    });
                });
            }
            stream->wait();
            V_d_f32 = V_d_f32_alloc;
        } else {
            V_d_f32 = (const float *) V->data;
        }
    }

    float *       dst_d = (float *) dst->data;

    const int Br = FATTN_BLOCK_R;
    const int Bc = FATTN_BLOCK_C;

    const int Tr = (N + Br - 1) / Br;
    const int Tc = (N_kv + Bc - 1) / Bc;

    // Per-row statistics for online softmax (one per Q row)
    float * l_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);

    // Launch: grid is (Br * Tr, Tc * n_heads) for processing all heads
    sycl::range<2> global(Br * Tr, Tc * n_heads);
    sycl::range<2> local(Br, 1);

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 2> Qtile({Br, DQK}, cgh);
        sycl::local_accessor<float, 2> Ktile({Bc, DQK}, cgh);
        sycl::local_accessor<float, 2> Vtile({Bc, DV}, cgh);
        sycl::local_accessor<float, 2> Stile({Br, Bc}, cgh);
        sycl::local_accessor<float, 1> Ptile({Br * Bc}, cgh);
        sycl::local_accessor<float, 1> m_local({Br}, cgh);
        sycl::local_accessor<float, 1> l_local({Br}, cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {

            float* q_loc = Qtile.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* k_loc = Ktile.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* v_loc = Vtile.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* s_loc = Stile.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* p_loc = Ptile.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* m_loc = m_local.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* l_loc = l_local.template get_multi_ptr<sycl::access::decorated::no>().get();

            auto group = it.get_group();
            int group_id_i = group.get_group_id(0);
            int group_id_j = group.get_group_id(1);

            // Head index from group_id_j
            int head_idx = group_id_j;
            int kv_head_idx = head_idx / gqa_ratio;  // Which K/V head this Q head maps to

            int row0 = group_id_i * Br;
            int col0 = (group_id_j % Tc) * Bc;

            if (row0 >= (int) N || col0 >= (int) N_kv) {
                return;
            }

            // Calculate base pointers for this head
            const float* Q_block = Q_d_f32 + (ptrdiff_t)(head_idx * N + row0) * q_row_stride;
            const float* K_block = K_d_f32 + (ptrdiff_t)(kv_head_idx * N_kv + col0) * k_row_stride;
            const float* V_block = V_d_f32 + (ptrdiff_t)(kv_head_idx * N_kv + col0) * v_row_stride;
            float*       O_block = dst_d + (ptrdiff_t)(head_idx * N + row0) * o_row_stride;

            // Row statistics offsets
            float* l_row = l_d + (ptrdiff_t)(head_idx * N + row0);
            float* m_row = m_d + (ptrdiff_t)(head_idx * N + row0);

            // Copy tiles to local memory
            ggml_sycl_memcpy<Br * DQK>(q_loc, Q_block);
            ggml_sycl_memcpy<Bc * DQK>(k_loc, K_block);
            ggml_sycl_memcpy<Bc * DV>(v_loc, V_block);

            it.barrier(sycl::access::fence_space::local_space);

            // Q @ K^T
            flash_attn_mul_mat_QK_kernel<DQK>(
                it,
                Q_block, q_row_stride,
                K_block, k_row_stride,
                s_loc, (ptrdiff_t)Bc,
                Br, Bc
            );

            it.barrier(sycl::access::fence_space::local_space);

            // Softmax with causal masking and optional sliding window
            flash_attn_softmax_kernel(
                it,
                s_loc, p_loc,
                m_loc, l_loc,
                Br, Bc,
                l_row, m_row,
                0,   // row_offset (not used)
                0    // window_size (0 = no sliding window limit)
            );

            it.barrier(sycl::access::fence_space::local_space);

            // P @ V
            flash_attn_mul_mat_PV_kernel<DV>(
                it,
                p_loc, (ptrdiff_t)Bc,
                V_block, v_row_stride,
                O_block, o_row_stride,
                Br, Bc
            );

            it.barrier(sycl::access::fence_space::local_space);
        });
    });

    // Normalize output by row sum
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(N * n_heads), [=](sycl::id<1> id) {
            int idx = id[0];
            int head_idx = idx / N;
            int row = idx % N;
            float l_val = l_d[idx];

            if (l_val <= 0.0f) {
                return;
            }

            float inv_l = 1.0f / l_val;
            float * o_row = dst_d + (ptrdiff_t)(head_idx * N + row) * o_row_stride;

            for (int col = 0; col < DV; ++col) {
                o_row[col] *= inv_l;
            }
        });
    });

    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);

    // Free F16 dequantization buffers if allocated
    if (is_f16) {
        sycl::free(Q_d_f32_alloc, *stream);
        sycl::free(K_d_f32_alloc, *stream);
        if (!V_is_K_view) {
            sycl::free(V_d_f32_alloc, *stream);
        }
    }
}

#ifdef SYCL_EXT_COOPERATIVE_MATRICES

template<int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM>
void ggml_sycl_op_flash_attn_coopmat_padded(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool q_is_f16 = (Q->type == GGML_TYPE_F16);
    const bool k_is_f16 = (K->type == GGML_TYPE_F16);
    const bool v_is_f16 = (V->type == GGML_TYPE_F16);

    const bool V_is_K_view = V->view_src && V->view_offs == 0 && (V->view_src == K || V->view_src == K->view_src);
    
    // Compile-time flag for MLA zero-copy optimization
    // When HEAD_DIM != V_HEAD_DIM and V is a view of K, kernel reads V from K directly
    constexpr bool V_FROM_K = (HEAD_DIM != V_HEAD_DIM);

    const float * Q_d_f32;
    const float * K_d_f32;
    const float * V_d_f32;
    float * Q_d_f32_alloc = nullptr;
    float * K_d_f32_alloc = nullptr;
    float * V_d_f32_alloc = nullptr;

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];
    const int64_t N_kv = K->ne[1];
    const int64_t n_heads_per_batch = Q->ne[2];
    const int64_t n_kv_heads_per_batch = K->ne[2];
    const int64_t batch = Q->ne[3];
    const int64_t gqa_ratio = n_heads_per_batch / n_kv_heads_per_batch;
    
    const int64_t n_heads = n_heads_per_batch * batch;
    const int64_t n_kv_heads = n_kv_heads_per_batch * batch;

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    if (q_is_f16) {
        const sycl::half * Q_d = (const sycl::half *) Q->data;
        Q_d_f32_alloc = (float *) sycl::malloc_device(N * HEAD_DIM * n_heads * sizeof(float), *stream);
        
        const int64_t q_stride_seq = Q->nb[1] / sizeof(sycl::half);
        const int64_t q_stride_head = Q->nb[2] / sizeof(sycl::half);
        const int64_t q_stride_batch = Q->nb[3] / sizeof(sycl::half);
        
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_heads_per_batch; ++head) {
                const int64_t head_total = b * n_heads_per_batch + head;
                const int64_t n_elements = N * HEAD_DIM;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq = idx / HEAD_DIM;
                            const int64_t dim = idx % HEAD_DIM;
                            Q_d_f32_alloc[head_total * N * HEAD_DIM + seq * HEAD_DIM + dim] = static_cast<float>(
                                Q_d[dim + head * q_stride_head + seq * q_stride_seq + b * q_stride_batch]);
                        }
                    });
                });
            }
        }
        Q_d_f32 = Q_d_f32_alloc;
    } else {
        const float * Q_d = (const float *) Q->data;
        Q_d_f32_alloc = (float *) sycl::malloc_device(N * HEAD_DIM * n_heads * sizeof(float), *stream);
        
        const int64_t q_stride_seq = Q->nb[1] / sizeof(float);
        const int64_t q_stride_head = Q->nb[2] / sizeof(float);
        const int64_t q_stride_batch = Q->nb[3] / sizeof(float);
        
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_heads_per_batch; ++head) {
                const int64_t head_total = b * n_heads_per_batch + head;
                const int64_t n_elements = N * HEAD_DIM;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq = idx / HEAD_DIM;
                            const int64_t dim = idx % HEAD_DIM;
                            Q_d_f32_alloc[head_total * N * HEAD_DIM + seq * HEAD_DIM + dim] = 
                                Q_d[dim + head * q_stride_head + seq * q_stride_seq + b * q_stride_batch];
                        }
                    });
                });
            }
        }
        Q_d_f32 = Q_d_f32_alloc;
    }
    
    if (k_is_f16) {
        const sycl::half * K_d = (const sycl::half *) K->data;
        K_d_f32_alloc = (float *) sycl::malloc_device(N_kv * HEAD_DIM * n_kv_heads * sizeof(float), *stream);
        
        const int64_t k_stride_seq = K->nb[1] / sizeof(sycl::half);
        const int64_t k_stride_head = K->nb[2] / sizeof(sycl::half);
        const int64_t k_stride_batch = K->nb[3] / sizeof(sycl::half);
        
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * HEAD_DIM;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq = idx / HEAD_DIM;
                            const int64_t dim = idx % HEAD_DIM;
                            K_d_f32_alloc[head_total * N_kv * HEAD_DIM + seq * HEAD_DIM + dim] = static_cast<float>(
                                K_d[dim + head * k_stride_head + seq * k_stride_seq + b * k_stride_batch]);
                        }
                    });
                });
            }
        }
        K_d_f32 = K_d_f32_alloc;
    } else {
        const float * K_d = (const float *) K->data;
        K_d_f32_alloc = (float *) sycl::malloc_device(N_kv * HEAD_DIM * n_kv_heads * sizeof(float), *stream);
        
        const int64_t k_stride_seq = K->nb[1] / sizeof(float);
        const int64_t k_stride_head = K->nb[2] / sizeof(float);
        const int64_t k_stride_batch = K->nb[3] / sizeof(float);
        
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * HEAD_DIM;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq = idx / HEAD_DIM;
                            const int64_t dim = idx % HEAD_DIM;
                            K_d_f32_alloc[head_total * N_kv * HEAD_DIM + seq * HEAD_DIM + dim] = 
                                K_d[dim + head * k_stride_head + seq * k_stride_seq + b * k_stride_batch];
                        }
                    });
                });
            }
        }
        K_d_f32 = K_d_f32_alloc;
    }
    
    if (V_is_K_view && V_FROM_K) {
        // MLA zero-copy: kernel reads V directly from K with HEAD_DIM stride
        // V_d_f32 points to K but kernel knows to use V_FROM_K template
        V_d_f32 = K_d_f32;
    } else if (V_is_K_view) {
        // V is view of K with same dimension - direct reuse
        V_d_f32 = K_d_f32;
    } else if (v_is_f16) {
        const sycl::half * V_d = (const sycl::half *) V->data;
        V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * V_HEAD_DIM * n_kv_heads * sizeof(float), *stream);

        const int64_t v_stride_seq = V->nb[1] / sizeof(sycl::half);
        const int64_t v_stride_head = V->nb[2] / sizeof(sycl::half);
        const int64_t v_stride_batch = V->nb[3] / sizeof(sycl::half);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * V_HEAD_DIM;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq = idx / V_HEAD_DIM;
                            const int64_t dim = idx % V_HEAD_DIM;
                            V_d_f32_alloc[head_total * N_kv * V_HEAD_DIM + seq * V_HEAD_DIM + dim] = static_cast<float>(
                                V_d[dim + head * v_stride_head + seq * v_stride_seq + b * v_stride_batch]);
                        }
                    });
                });
            }
        }
        V_d_f32 = V_d_f32_alloc;
    } else {
        const float * V_d = (const float *) V->data;
        V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * V_HEAD_DIM * n_kv_heads * sizeof(float), *stream);

        const int64_t v_stride_seq = V->nb[1] / sizeof(float);
        const int64_t v_stride_head = V->nb[2] / sizeof(float);
        const int64_t v_stride_batch = V->nb[3] / sizeof(float);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * V_HEAD_DIM;
                stream->submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq = idx / V_HEAD_DIM;
                            const int64_t dim = idx % V_HEAD_DIM;
                            V_d_f32_alloc[head_total * N_kv * V_HEAD_DIM + seq * V_HEAD_DIM + dim] =
                                V_d[dim + head * v_stride_head + seq * v_stride_seq + b * v_stride_batch];
                        }
                    });
                });
            }
        }
        V_d_f32 = V_d_f32_alloc;
    }
    
    // Event-based synchronization: collect prep events for kernel dependency
    // The reordering kernels are already in-flight, kernel will depend on them via event
    std::vector<sycl::event> prep_events;

    float * dst_d = (float *) dst->data;

    // Use small tiles for large head dimensions to fit within 128KB SLM
    constexpr bool USE_SMALL_TILES = (PADDED_V_HEAD_DIM >= 512 || PADDED_HEAD_DIM >= 512);
    constexpr int BLOCK_M = USE_SMALL_TILES ? 16 : 32;
    constexpr int BLOCK_N = USE_SMALL_TILES ? 16 : 32;
    constexpr int THREADS_PER_WG = USE_SMALL_TILES ? 32 : 64;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;

    float * l_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);

    sycl::range<2> global(Tr * THREADS_PER_WG, n_heads);
    sycl::range<2> local(THREADS_PER_WG, 1);

    xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(stream->get_device());

    constexpr int Q_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr int V_T_STRIDE = BLOCK_N + 8;

    // Small-tile kernel eliminates shV buffer (loads V directly to shVT)
    // Large-tile kernel still uses shV
    constexpr int V_STRIDE = USE_SMALL_TILES ? 0 : (PADDED_V_HEAD_DIM + 8);
    constexpr size_t BF16_BYTES = (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE +
                                    BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE +
                                    PADDED_V_HEAD_DIM * V_T_STRIDE) * sizeof(sycl::half);
    constexpr size_t FLOAT_BYTES = (BLOCK_M * S_STRIDE + BLOCK_M * 3 + BLOCK_M * PADDED_V_HEAD_DIM) * sizeof(float);
    constexpr size_t SHMEM_SIZE = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    const float * mask_d = nullptr;
    float * mask_d_f32_alloc = nullptr;
    int64_t mask_stride = N_kv;
    if (mask != nullptr && mask->data != nullptr) {
        if (mask->type == GGML_TYPE_F32) {
            mask_d = (const float *) mask->data;
            mask_stride = mask->nb[1] / sizeof(float);
        } else if (mask->type == GGML_TYPE_F16) {
            const int64_t mask_n_kv = mask->ne[0];
            const int64_t mask_n_q = mask->ne[1];
            const int64_t mask_elements = mask_n_kv * mask_n_q;
            
            mask_d_f32_alloc = (float *) sycl::malloc_device(mask_elements * sizeof(float), *stream);
            const sycl::half * mask_f16 = (const sycl::half *) mask->data;
            const ptrdiff_t mask_row_stride_f16 = mask->nb[1] / sizeof(sycl::half);
            
            sycl::event mask_event = stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((mask_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < mask_elements) {
                        const int64_t row = idx / mask_n_kv;
                        const int64_t col = idx % mask_n_kv;
                        mask_d_f32_alloc[idx] = static_cast<float>(mask_f16[row * mask_row_stride_f16 + col]);
                    }
                });
            });
            prep_events.push_back(mask_event);
            
            mask_d = mask_d_f32_alloc;
            mask_stride = mask_n_kv;
        }
    }

    // Allocate temp output buffer and output stride for kernel
    float * O_temp = (float *) sycl::malloc_device(N * V_HEAD_DIM * n_heads * sizeof(float), *stream);
    const int o_row_stride = V_HEAD_DIM;

    // Use V_FROM_K template parameter for MLA zero-copy optimization
    // When V_FROM_K is true, kernel reads V directly from K's memory
    sycl::event xmx_event;
    if constexpr (USE_SMALL_TILES) {
        // Small-tile kernel for large head dimensions (576/512)
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            xmx_event = stream->submit([&](sycl::handler& cgh) {
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_small_tile_n8<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K>(
                        it,
                        Q_d_f32, K_d_f32, V_d_f32, O_temp,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride, o_row_stride,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        } else {
            xmx_event = stream->submit([&](sycl::handler& cgh) {
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_small_tile_n16<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K>(
                        it,
                        Q_d_f32, K_d_f32, V_d_f32, O_temp,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride, o_row_stride,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        }
    } else {
        // Standard 32x32 tile kernel for normal head dimensions
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            xmx_event = stream->submit([&](sycl::handler& cgh) {
                // Depend on prep kernels (Q/K/V reorder, mask conversion)
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_n8_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K>(
                        it,
                        Q_d_f32, K_d_f32, V_d_f32, O_temp,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride, o_row_stride,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        } else {
            xmx_event = stream->submit([&](sycl::handler& cgh) {
                // Depend on prep kernels (Q/K/V reorder, mask conversion)
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_n16_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K>(
                        it,
                        Q_d_f32, K_d_f32, V_d_f32, O_temp,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride, o_row_stride,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        }
    }

    // Output reorder kernels depend on XMX kernel completion
    std::vector<sycl::event> reorder_events;
    const int64_t dst_stride_seq = dst->nb[1] / sizeof(float);
    const int64_t dst_stride_head = dst->nb[2] / sizeof(float);
    const int64_t dst_stride_batch = dst->nb[3] / sizeof(float);
    
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t head = 0; head < n_heads_per_batch; ++head) {
            const int64_t head_total = b * n_heads_per_batch + head;
            const int64_t n_elements = N * V_HEAD_DIM;
            sycl::event reorder_event = stream->submit([&](sycl::handler& cgh) {
                cgh.depends_on(xmx_event);
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t seq = idx / V_HEAD_DIM;
                        const int64_t dim = idx % V_HEAD_DIM;
                        dst_d[dim + head * dst_stride_head + seq * dst_stride_seq + b * dst_stride_batch] = 
                            O_temp[head_total * N * V_HEAD_DIM + seq * V_HEAD_DIM + dim];
                    }
                });
            });
            reorder_events.push_back(reorder_event);
        }
    }
    
    // Wait only at the end when results are needed
    sycl::event::wait(reorder_events);

    sycl::free(O_temp, *stream);
    if (Q_d_f32_alloc) sycl::free(Q_d_f32_alloc, *stream);
    if (K_d_f32_alloc) sycl::free(K_d_f32_alloc, *stream);
    if (V_d_f32_alloc && !V_is_K_view) sycl::free(V_d_f32_alloc, *stream);
    if (mask_d_f32_alloc) sycl::free(mask_d_f32_alloc, *stream);
    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}

// ============================================================================
// Stride-Aware Direct Loading Flash Attention
// ============================================================================
// This variant eliminates the intermediate F32 repack buffers by loading
// directly from ggml's strided tensor layout. This reduces memory bandwidth
// by 2x and eliminates temporary allocations.
//
// Enabled by setting GGML_SYCL_FLASH_ATTN_DIRECT=1
// ============================================================================
template<int64_t DQK, int64_t DV, int BLOCK_M = 32, int BLOCK_N = 32>
void ggml_sycl_op_flash_attn_coopmat_direct(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (getenv("GGML_SYCL_FLASH_ATTN_DEBUG")) {
        GGML_SYCL_DEBUG("[SYCL] Flash attention coopmat_direct kernel: DQK=%ld, DV=%ld, BLOCK_M=%d, BLOCK_N=%d\n", DQK, DV, BLOCK_M, BLOCK_N);
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool is_f16 = (Q->type == GGML_TYPE_F16);
    const bool V_is_K_view = V->view_src && V->view_offs == 0 && (V->view_src == K || V->view_src == K->view_src);

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];
    const int64_t N_kv = K->ne[1];
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Compute strides for direct loading
    fattn_tensor_strides strides = compute_tensor_strides(Q, K, V, dst);

    // Get raw data pointers (type depends on input format)
    const void * Q_data = Q->data;
    const void * K_data = K->data;
    const void * V_data = V_is_K_view ? K->data : V->data;
    float * O_data = (float *) dst->data;

    // Process mask if present
    const float * mask_d = nullptr;
    float * mask_d_f32_alloc = nullptr;
    int64_t mask_stride_val = N_kv;
    if (mask != nullptr && mask->data != nullptr) {
        if (mask->type == GGML_TYPE_F32) {
            mask_d = (const float *) mask->data;
            mask_stride_val = mask->nb[1] / sizeof(float);
        } else if (mask->type == GGML_TYPE_F16) {
            const int64_t mask_n_kv = mask->ne[0];
            const int64_t mask_n_q = mask->ne[1];
            const int64_t mask_elements = mask_n_kv * mask_n_q;
            
            mask_d_f32_alloc = (float *) sycl::malloc_device(mask_elements * sizeof(float), *stream);
            const sycl::half * mask_f16 = (const sycl::half *) mask->data;
            const ptrdiff_t mask_row_stride_f16 = mask->nb[1] / sizeof(sycl::half);
            
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((mask_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < mask_elements) {
                        const int64_t row = idx / mask_n_kv;
                        const int64_t col = idx % mask_n_kv;
                        mask_d_f32_alloc[idx] = static_cast<float>(mask_f16[row * mask_row_stride_f16 + col]);
                    }
                });
            });
            mask_d = mask_d_f32_alloc;
            mask_stride_val = mask_n_kv;
        }
    }

    constexpr int THREADS_PER_WG = 64;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;

    // Unused but needed for API compatibility
    float * l_d = nullptr;
    float * m_d = nullptr;

    sycl::range<2> global(Tr * THREADS_PER_WG, n_heads);
    sycl::range<2> local(THREADS_PER_WG, 1);

    xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(stream->get_device());

    // Shared memory size calculation
    constexpr int Q_STRIDE = DQK + 8;
    constexpr int K_STRIDE = DQK + 8;
    constexpr int V_STRIDE = DV + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;  // BLOCK_N + padding
    constexpr int V_T_STRIDE = BLOCK_N + 8;
    constexpr size_t BF16_BYTES = (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE +
                                    BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE +
                                    DV * V_T_STRIDE) * sizeof(sycl::half);
    constexpr size_t FLOAT_BYTES = (BLOCK_M * (BLOCK_N + 8) + BLOCK_M * 3 + BLOCK_M * DV) * sizeof(float);
    constexpr size_t SHMEM_SIZE = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    static bool first_call = true;
    if (first_call && getenv("GGML_SYCL_FLASH_ATTN_DEBUG")) {
        GGML_SYCL_DEBUG("ggml_sycl: DIRECT XMX flash_attn N=%ld N_kv=%ld n_heads=%ld DQK=%ld DV=%ld is_f16=%d BLOCK_M=%d BLOCK_N=%d SHMEM_SIZE=%lu\n",
                N, N_kv, n_heads, DQK, DV, is_f16, BLOCK_M, BLOCK_N, SHMEM_SIZE * sizeof(float));
        GGML_SYCL_DEBUG("ggml_sycl: Q strides: seq=%ld head=%ld\n", strides.q_stride_seq, strides.q_stride_head);
        GGML_SYCL_DEBUG("ggml_sycl: K strides: seq=%ld head=%ld\n", strides.k_stride_seq, strides.k_stride_head);
        GGML_SYCL_DEBUG("ggml_sycl: V strides: seq=%ld head=%ld\n", strides.v_stride_seq, strides.v_stride_head);
        GGML_SYCL_DEBUG("ggml_sycl: O strides: seq=%ld head=%ld\n", strides.o_stride_seq, strides.o_stride_head);
        first_call = false;
    }

    // Select kernel based on input type and tile kind
    if (is_f16) {
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            stream->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_strided_n8<DQK, DV, fattn_input_type::f16, false, BLOCK_M, BLOCK_N>(
                        it,
                        Q_data, K_data, V_data, O_data,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride_val, strides,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        } else {
            stream->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_strided_n16<DQK, DV, fattn_input_type::f16, false, BLOCK_M, BLOCK_N>(
                        it,
                        Q_data, K_data, V_data, O_data,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride_val, strides,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        }
    } else {
        // F32 input
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            stream->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_strided_n8<DQK, DV, fattn_input_type::f32, false, BLOCK_M, BLOCK_N>(
                        it,
                        Q_data, K_data, V_data, O_data,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride_val, strides,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        } else {
            stream->submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_coopmat_kernel_strided_n16<DQK, DV, fattn_input_type::f32, false, BLOCK_M, BLOCK_N>(
                        it,
                        Q_data, K_data, V_data, O_data,
                        l_d, m_d,
                        N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                        scale, mask_d, mask_stride_val, strides,
                        shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
            });
        }
    }

    stream->wait();

    if (mask_d_f32_alloc) sycl::free(mask_d_f32_alloc, *stream);
}
#endif // SYCL_EXT_COOPERATIVE_MATRICES

#ifdef GGML_SYCL_USE_INTEL_ONEMKL
// ============================================================================
// oneMKL Flash Attention with KV-Split (Flash Decoding)
// ============================================================================
// Optimized flash attention using oneMKL BLAS with optional KV dimension splitting.
// For short contexts (N_kv < 256), uses single pass (n_splits=1).
// For longer contexts, splits KV across multiple chunks for better parallelism.
//
// Algorithm:
// 1. Split KV into n_splits chunks (n_splits=1 for short contexts)
// 2. Each chunk computes partial attention: Q @ K_chunk^T -> softmax -> @ V_chunk
//    Stores partial results: M_partial (max), L_partial (sum), O_partial (output)
// 3. Reduction kernel merges partials using online softmax math
// ============================================================================
template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int64_t N = Q->ne[1];        // Query sequence length
    const int64_t N_kv = K->ne[1];     // KV sequence length
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;

    dpct::queue_ptr stream = ctx.stream();

    // Determine number of KV splits
    const int64_t n_splits = get_kv_split_count(N_kv, n_heads);
    const int64_t kv_per_split = (N_kv + n_splits - 1) / n_splits;

    static bool first_call = true;
    if (first_call) {
        fprintf(stderr, "ggml_sycl: oneMKL flash attention ACTIVE: N=%ld N_kv=%ld n_heads=%ld n_splits=%ld kv_per_split=%ld\n",
                N, N_kv, n_heads, n_splits, kv_per_split);
        first_call = false;
    }

    const bool q_is_f16 = (Q->type == GGML_TYPE_F16);
    const bool k_is_f16 = (K->type == GGML_TYPE_F16);
    const bool v_is_f16 = (V->type == GGML_TYPE_F16);
    const bool V_is_K_view = V->view_src && V->view_offs == 0 && (V->view_src == K || V->view_src == K->view_src);

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Graph optimization: use async_malloc when recording for graph capture
    // The higher-level graph infrastructure (ggml_backend_sycl_graph_compute) handles
    // the actual command graph recording. This op just needs to use graph-compatible
    // memory allocation (async_malloc) to allow the graph to own the buffers.
    bool use_async_mem = false;
#ifdef GGML_SYCL_GRAPH
    // These are defined in ggml-sycl.cpp
    extern int g_ggml_sycl_disable_graph;
    extern int g_ggml_sycl_use_async_mem_op;
    use_async_mem = !g_ggml_sycl_disable_graph && g_ggml_sycl_use_async_mem_op;
#endif

    // Allocate and reorder Q to contiguous layout [n_heads, N, DQK]
    float * Q_d_f32 = nullptr;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async_mem) {
        Q_d_f32 = (float *) syclex::async_malloc(*stream, sycl::usm::alloc::device, N * DQK * n_heads * sizeof(float));
    } else
#endif
    {
        Q_d_f32 = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
    }
    if (q_is_f16) {
        const sycl::half * Q_f16 = (const sycl::half *) Q->data;
        const int64_t q_stride_seq = Q->nb[1] / sizeof(sycl::half);
        const int64_t q_stride_head = Q->nb[2] / sizeof(sycl::half);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N * DQK * n_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N * DQK);
                const int64_t rem = idx % (N * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                Q_d_f32[idx] = static_cast<float>(Q_f16[d + head * q_stride_head + n * q_stride_seq]);
            });
        });
    } else {
        const float * Q_f32 = (const float *) Q->data;
        const int64_t q_stride_seq = Q->nb[1] / sizeof(float);
        const int64_t q_stride_head = Q->nb[2] / sizeof(float);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N * DQK * n_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N * DQK);
                const int64_t rem = idx % (N * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                Q_d_f32[idx] = Q_f32[d + head * q_stride_head + n * q_stride_seq];
            });
        });
    }

    // Allocate and reorder K to contiguous layout [n_kv_heads, N_kv, DQK]
    float * K_d_f32 = nullptr;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async_mem) {
        K_d_f32 = (float *) syclex::async_malloc(*stream, sycl::usm::alloc::device, N_kv * DQK * n_kv_heads * sizeof(float));
    } else
#endif
    {
        K_d_f32 = (float *) sycl::malloc_device(N_kv * DQK * n_kv_heads * sizeof(float), *stream);
    }
    if (k_is_f16) {
        const sycl::half * K_f16 = (const sycl::half *) K->data;
        const int64_t k_stride_seq = K->nb[1] / sizeof(sycl::half);
        const int64_t k_stride_head = K->nb[2] / sizeof(sycl::half);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N_kv * DQK * n_kv_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N_kv * DQK);
                const int64_t rem = idx % (N_kv * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                K_d_f32[idx] = static_cast<float>(K_f16[d + head * k_stride_head + n * k_stride_seq]);
            });
        });
    } else {
        const float * K_f32 = (const float *) K->data;
        const int64_t k_stride_seq = K->nb[1] / sizeof(float);
        const int64_t k_stride_head = K->nb[2] / sizeof(float);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N_kv * DQK * n_kv_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N_kv * DQK);
                const int64_t rem = idx % (N_kv * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                K_d_f32[idx] = K_f32[d + head * k_stride_head + n * k_stride_seq];
            });
        });
    }

    // Allocate and reorder V
    float * V_d_f32 = nullptr;
    float * V_d_f32_alloc = nullptr;
    if (V_is_K_view && DQK == DV) {
        V_d_f32 = K_d_f32;
    } else if (V_is_K_view && DQK != DV) {
        V_d_f32_alloc = nullptr;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        if (use_async_mem) {
            V_d_f32_alloc = (float *) syclex::async_malloc(*stream, sycl::usm::alloc::device, N_kv * DV * n_kv_heads * sizeof(float));
        } else
#endif
        {
            V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
        }
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N_kv * DV * n_kv_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N_kv * DV);
                const int64_t rem = idx % (N_kv * DV);
                const int64_t n = rem / DV;
                const int64_t d = rem % DV;
                V_d_f32_alloc[idx] = K_d_f32[head * N_kv * DQK + n * DQK + d];
            });
        });
        V_d_f32 = V_d_f32_alloc;
    } else {
        V_d_f32_alloc = nullptr;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        if (use_async_mem) {
            V_d_f32_alloc = (float *) syclex::async_malloc(*stream, sycl::usm::alloc::device, N_kv * DV * n_kv_heads * sizeof(float));
        } else
#endif
        {
            V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
        }
        if (v_is_f16) {
            const sycl::half * V_f16 = (const sycl::half *) V->data;
            const int64_t v_stride_seq = V->nb[1] / sizeof(sycl::half);
            const int64_t v_stride_head = V->nb[2] / sizeof(sycl::half);
            stream->submit([&](sycl::handler& cgh) {
                const int64_t total = N_kv * DV * n_kv_heads;
                cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx >= total) return;
                    const int64_t head = idx / (N_kv * DV);
                    const int64_t rem = idx % (N_kv * DV);
                    const int64_t n = rem / DV;
                    const int64_t d = rem % DV;
                    V_d_f32_alloc[idx] = static_cast<float>(V_f16[d + head * v_stride_head + n * v_stride_seq]);
                });
            });
        } else {
            const float * V_f32 = (const float *) V->data;
            const int64_t v_stride_seq = V->nb[1] / sizeof(float);
            const int64_t v_stride_head = V->nb[2] / sizeof(float);
            stream->submit([&](sycl::handler& cgh) {
                const int64_t total = N_kv * DV * n_kv_heads;
                cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx >= total) return;
                    const int64_t head = idx / (N_kv * DV);
                    const int64_t rem = idx % (N_kv * DV);
                    const int64_t n = rem / DV;
                    const int64_t d = rem % DV;
                    V_d_f32_alloc[idx] = V_f32[d + head * v_stride_head + n * v_stride_seq];
                });
            });
        }
        V_d_f32 = V_d_f32_alloc;
    }

    // Process mask if present
    const float * mask_d = nullptr;
    float * mask_d_f32_alloc = nullptr;
    int64_t mask_stride = N_kv;
    if (mask != nullptr && mask->data != nullptr) {
        if (mask->type == GGML_TYPE_F32) {
            mask_d = (const float *) mask->data;
            mask_stride = mask->nb[1] / sizeof(float);
        } else if (mask->type == GGML_TYPE_F16) {
            const int64_t mask_n_kv = mask->ne[0];
            const int64_t mask_n_q = mask->ne[1];
            const int64_t mask_elements = mask_n_kv * mask_n_q;
            mask_d_f32_alloc = nullptr;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
            if (use_async_mem) {
                mask_d_f32_alloc = (float *) syclex::async_malloc(*stream, sycl::usm::alloc::device, mask_elements * sizeof(float));
            } else
#endif
            {
                mask_d_f32_alloc = (float *) sycl::malloc_device(mask_elements * sizeof(float), *stream);
            }
            const sycl::half * mask_f16 = (const sycl::half *) mask->data;
            const ptrdiff_t mask_row_stride_f16 = mask->nb[1] / sizeof(sycl::half);
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((mask_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < mask_elements) {
                        const int64_t row = idx / mask_n_kv;
                        const int64_t col = idx % mask_n_kv;
                        mask_d_f32_alloc[idx] = static_cast<float>(mask_f16[row * mask_row_stride_f16 + col]);
                    }
                });
            });
            mask_d = mask_d_f32_alloc;
            mask_stride = mask_n_kv;
        }
    }

    const float * sinks_d = nullptr;
    if (sinks != nullptr && sinks->data != nullptr) {
        sinks_d = (const float *) sinks->data;
    }

    // Allocate partials buffer: [n_heads, N, n_splits, 2 + DV]
    const int64_t partial_size = 2 + DV;
    const int64_t partials_total = n_heads * N * n_splits * partial_size;
    float * partials = nullptr;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async_mem) {
        partials = (float *) syclex::async_malloc(*stream, sycl::usm::alloc::device, partials_total * sizeof(float));
    } else
#endif
    {
        partials = (float *) sycl::malloc_device(partials_total * sizeof(float), *stream);
    }

    float * S_d = nullptr;
#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    if (use_async_mem) {
        S_d = (float *) syclex::async_malloc(*stream, sycl::usm::alloc::device, n_heads * N * N_kv * sizeof(float));
    } else
#endif
    {
        S_d = (float *) sycl::malloc_device(n_heads * N * N_kv * sizeof(float), *stream);
    }

    const int64_t lda_q = DQK;
    const int64_t lda_k = DQK;
    const int64_t lda_v = DV;

    const int64_t stride_q = N * DQK;
    const int64_t stride_k = N_kv * DQK;
    const int64_t stride_v = N_kv * DV;
    
    const int64_t lda_s = N_kv; 
    const int64_t stride_s = N * N_kv;

    const int64_t ldc_o = n_splits * partial_size;
    const int64_t stride_o = N * ldc_o;

    // Process all KV splits in parallel
    for (int64_t split = 0; split < n_splits; ++split) {
        const int64_t kv_start = split * kv_per_split;
        const int64_t kv_end = std::min(kv_start + kv_per_split, N_kv);
        const int64_t kv_chunk_size = kv_end - kv_start;
        
        if (kv_chunk_size <= 0) continue;

        // Q @ K_chunk^T -> S_d (offset by kv_start)
        if (gqa_ratio == 1) {
            oneapi::mkl::blas::row_major::gemm_batch(*stream,
                oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
                N, kv_chunk_size, DQK,
                scale,
                Q_d_f32, lda_q, stride_q,
                K_d_f32 + kv_start * DQK, lda_k, stride_k,
                0.0f,
                S_d + kv_start, lda_s, stride_s,
                n_heads);
        } else {
            for (int64_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
                const int64_t q_head_start = kv_head * gqa_ratio;
                oneapi::mkl::blas::row_major::gemm_batch(*stream,
                    oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
                    N, kv_chunk_size, DQK,
                    scale,
                    Q_d_f32 + q_head_start * stride_q, lda_q, stride_q,
                    K_d_f32 + kv_head * stride_k + kv_start * DQK, lda_k, 0,
                    0.0f,
                    S_d + q_head_start * stride_s + kv_start, lda_s, stride_s,
                    gqa_ratio);
            }
        }

        // Apply mask and compute softmax for this chunk, store M/L partials
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n_heads * N), [=](sycl::id<1> idx) {
                const int64_t head = idx[0] / N;
                const int64_t q = idx[0] % N;
                float * S_row = S_d + head * stride_s + q * lda_s + kv_start;
                
                float row_max = -1.0e20f;
                for (int64_t k = 0; k < kv_chunk_size; ++k) {
                    float s_val = S_row[k];
                    const int64_t global_k = kv_start + k;
                    if (mask_d != nullptr) {
                        s_val += mask_d[q * mask_stride + global_k];
                    }
                    S_row[k] = s_val;
                    row_max = sycl::fmax(row_max, s_val);
                }
                
                // Handle sinks (only add to first split)
                float sink_contrib = 0.0f;
                if (split == 0 && sinks_d != nullptr) {
                    float sink_val = sinks_d[head];
                    row_max = sycl::fmax(row_max, sink_val);
                    sink_contrib = sycl::exp(sycl::fmax(sink_val - row_max, -20.0f));
                }
                
                float sum = sink_contrib;
                for (int64_t k = 0; k < kv_chunk_size; ++k) {
                    float exp_val = sycl::exp(sycl::fmax(S_row[k] - row_max, -20.0f));
                    S_row[k] = exp_val;
                    sum += exp_val;
                }
                
                float * partial = partials + (head * N + q) * n_splits * partial_size + split * partial_size;
                partial[0] = row_max;
                partial[1] = sum;
            });
        });

        // S_chunk @ V_chunk -> O (directly to partials)
        float * O_ptr = partials + split * partial_size + 2;
        
        if (gqa_ratio == 1) {
            oneapi::mkl::blas::row_major::gemm_batch(*stream,
                oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
                N, DV, kv_chunk_size,
                1.0f,
                S_d + kv_start, lda_s, stride_s,
                V_d_f32 + kv_start * DV, lda_v, stride_v,
                0.0f,
                O_ptr, ldc_o, stride_o,
                n_heads);
        } else {
             for (int64_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
                const int64_t q_head_start = kv_head * gqa_ratio;
                oneapi::mkl::blas::row_major::gemm_batch(*stream,
                    oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
                    N, DV, kv_chunk_size,
                    1.0f,
                    S_d + q_head_start * stride_s + kv_start, lda_s, stride_s,
                    V_d_f32 + kv_head * stride_v + kv_start * DV, lda_v, 0,
                    0.0f,
                    O_ptr + q_head_start * stride_o, ldc_o, stride_o,
                    gqa_ratio);
            }
        }
    }
    
    float * O_d = (float *) dst->data;
    // Flash attention output is permuted: [DV, n_heads, N, batch]
    // nb[1] = stride between heads, nb[2] = stride between sequence positions
    const int64_t o_stride_head = dst->nb[1] / sizeof(float);  // stride between heads = DV
    const int64_t o_stride_seq = dst->nb[2] / sizeof(float);   // stride between rows = DV * n_heads
    
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_heads * N, DV), sycl::range<2>(1, DV)), 
            [=](sycl::nd_item<2> it) {
            flash_attn_combine_splits_kernel<DV>(it, partials, O_d, n_splits, n_heads, N, partial_size, o_stride_head, o_stride_seq);
        });
    });

    // When graph is enabled (g_ggml_sycl_use_async_mem_op != 0), the graph infrastructure
    // owns the allocated buffers and will manage their lifetime. Do not free manually.
    if (!g_ggml_sycl_use_async_mem_op) {
        sycl::free(partials, *stream);
        sycl::free(S_d, *stream);
        sycl::free(Q_d_f32, *stream);
        sycl::free(K_d_f32, *stream);
        if (mask_d_f32_alloc) sycl::free(mask_d_f32_alloc, *stream);
        if (V_d_f32_alloc && !V_is_K_view) sycl::free(V_d_f32_alloc, *stream);
    }
}
#endif // GGML_SYCL_USE_INTEL_ONEMKL

void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];
    (void)K; // Used by direct loading path when SYCL_EXT_COOPERATIVE_MATRICES is defined

    GGML_SYCL_DEBUG("[SYCL][OP] call ggml_sycl_op_flash_attn: Q=[%ld,%ld,%ld] V=[%ld,%ld,%ld]\n",
            Q->ne[0], Q->ne[1], Q->ne[2], V->ne[0], V->ne[1], V->ne[2]);

#ifdef GGML_SYCL_USE_INTEL_ONEMKL
    // Check if oneMKL is forced first (before XMX check)
    static bool sycl_use_mkl = false;
    static bool mkl_checked = false;

    if (!mkl_checked) {
        sycl::device device = ctx.stream()->get_device();
        sycl_use_mkl = ggml_sycl_flash_attn_use_mkl(device);
        mkl_checked = true;
        if (sycl_use_mkl) {
            GGML_SYCL_DEBUG("ggml_sycl: Using oneMKL BLAS for flash attention (device=%s)\n",
                    device.get_info<sycl::info::device::name>().c_str());
        }
    }

    const int64_t DQK = Q->ne[0];
    const int64_t DV = V->ne[0];
    const int64_t N = Q->ne[1];
    const int64_t N_kv = K->ne[1];
    const int64_t n_heads = Q->ne[2];

    // Sinks (attention sinks / StreamingLLM) only supported in MKL path
    // XMX cooperative matrix kernels don't support sinks yet
    const bool use_mkl_for_sinks = (sinks != nullptr);

    // Use MKL for small batch sizes (N < 32) as it's often faster for small batches
    // Set GGML_SYCL_FLASH_ATTN_FORCE_XMX=1 to bypass this and use XMX for all batch sizes
    static bool force_xmx = false;
    static bool force_xmx_checked = false;
    if (!force_xmx_checked) {
        const char* env = getenv("GGML_SYCL_FLASH_ATTN_FORCE_XMX");
        force_xmx = (env != nullptr && strcmp(env, "1") == 0);
        force_xmx_checked = true;
        if (force_xmx) {
            GGML_SYCL_DEBUG("ggml_sycl: XMX flash attention FORCED for all batch sizes by environment variable\n");
        }
    }
    const bool small_batch = !force_xmx && (N < 32);

    // Use oneMKL KV-split path when MKL is available and needed
    // KV-split handles both short and long contexts efficiently (n_splits=1 for short contexts)
    if (sycl_use_mkl || use_mkl_for_sinks || small_batch) {
        if (DQK == 576 && DV == 512) {
            ggml_sycl_op_flash_attn_mkl<576, 512>(ctx, dst);
            return;
        }

        if (DQK == DV) {
            switch (DQK) {
                case 32:
                    ggml_sycl_op_flash_attn_mkl<32, 32>(ctx, dst);
                    return;
                case 40:
                    ggml_sycl_op_flash_attn_mkl<40, 40>(ctx, dst);
                    return;
                case 48:
                    ggml_sycl_op_flash_attn_mkl<48, 48>(ctx, dst);
                    return;
                case 56:
                    ggml_sycl_op_flash_attn_mkl<56, 56>(ctx, dst);
                    return;
                case 64:
                    ggml_sycl_op_flash_attn_mkl<64, 64>(ctx, dst);
                    return;
                case 72:
                    ggml_sycl_op_flash_attn_mkl<72, 72>(ctx, dst);
                    return;
                case 80:
                    ggml_sycl_op_flash_attn_mkl<80, 80>(ctx, dst);
                    return;
                case 88:
                    ggml_sycl_op_flash_attn_mkl<88, 88>(ctx, dst);
                    return;
                case 96:
                    ggml_sycl_op_flash_attn_mkl<96, 96>(ctx, dst);
                    return;
                case 104:
                    ggml_sycl_op_flash_attn_mkl<104, 104>(ctx, dst);
                    return;
                case 112:
                    ggml_sycl_op_flash_attn_mkl<112, 112>(ctx, dst);
                    return;
                case 128:
                    ggml_sycl_op_flash_attn_mkl<128, 128>(ctx, dst);
                    return;
                case 192:
                    ggml_sycl_op_flash_attn_mkl<192, 192>(ctx, dst);
                    return;
                case 256:
                    ggml_sycl_op_flash_attn_mkl<256, 256>(ctx, dst);
                    return;
                case 512:
                    ggml_sycl_op_flash_attn_mkl<512, 512>(ctx, dst);
                    return;
                case 576:
                    ggml_sycl_op_flash_attn_mkl<576, 576>(ctx, dst);
                    return;
                default:
                    GGML_SYCL_DEBUG("ggml_sycl: oneMKL not implemented for head size DQK=%ld DV=%ld\n", DQK, DV);
                    break;
            }
        } else {
            GGML_SYCL_DEBUG("ggml_sycl: oneMKL path requires DQK==DV, got DQK=%ld DV=%ld\n", DQK, DV);
        }

        // If we get here and it was mandatory MKL, then we should probably abort or warn
        if (sycl_use_mkl || use_mkl_for_sinks) {
            GGML_ABORT("ggml_sycl: oneMKL flash attention path failed (unsupported head size); XMX is required but fallback failed\n");
        }
    }
#endif

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
    // Try XMX path if device supports it
    static bool sycl_use_xmx = false;
    static bool xmx_checked = false;
    static bool use_direct_loading = false;
    static bool direct_checked = false;

    if (!xmx_checked) {
        sycl::device device = ctx.stream()->get_device();
        sycl_use_xmx = ggml_sycl_flash_attn_has_xmx(device);
        xmx_checked = true;
        xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(device);
        const char * tile_str = (tile_kind == xmx_tile_kind::tile_dg2) ? "DG2 (nsize=8)" : "PVC/B60 (nsize=16)";
        GGML_SYCL_DEBUG("ggml_sycl: XMX detection: device=%s, has_xmx=%d, tile_kind=%s\n",
                device.get_info<sycl::info::device::name>().c_str(), sycl_use_xmx, tile_str);
        if (sycl_use_xmx) {
            GGML_SYCL_DEBUG("ggml_sycl: Using XMX (cooperative matrix) for flash attention with %s tiles\n", tile_str);
        }
    }

    // Check for direct loading mode (stride-aware kernel)
    if (!direct_checked) {
        use_direct_loading = ggml_sycl_flash_attn_use_direct();
        direct_checked = true;
    }

    if (sycl_use_xmx) {
        const int64_t actual_d = Q->ne[0];
        const int64_t actual_dv = V->ne[0];
        const int64_t padded_d = get_padded_head_size(actual_d);
        const int64_t padded_dv = get_padded_head_size(actual_dv);

        // Try direct loading path first if enabled and compatible
        bool direct_supported_shape = (actual_d == actual_dv && actual_d == padded_d) || (actual_d == 576 && actual_dv == 512);

        if (use_direct_loading && direct_supported_shape && can_use_direct_loading(Q, K, V, dst)) {
            try {
                if (actual_d == 576 && actual_dv == 512) {
                     // Use smaller blocks (8x16) for GLM-4.7 to fit in SLM (DQK=576 needs large SLM)
                     ggml_sycl_op_flash_attn_coopmat_direct<576, 512, 8, 16>(ctx, dst);
                     return;
                }
                switch (actual_d) {
                    case 32:
                        ggml_sycl_op_flash_attn_coopmat_direct<32, 32>(ctx, dst);
                        return;
                    case 64:
                        ggml_sycl_op_flash_attn_coopmat_direct<64, 64>(ctx, dst);
                        return;
                    case 96:
                        ggml_sycl_op_flash_attn_coopmat_direct<96, 96>(ctx, dst);
                        return;
                    case 128:
                        ggml_sycl_op_flash_attn_coopmat_direct<128, 128>(ctx, dst);
                        return;
                    case 256:
                        ggml_sycl_op_flash_attn_coopmat_direct<256, 256>(ctx, dst);
                        return;
                    case 512:
                        ggml_sycl_op_flash_attn_coopmat_direct<512, 512>(ctx, dst);
                        return;
                    default:
                        break;
                }
            } catch (const std::exception& e) {
                GGML_SYCL_DEBUG("ggml_sycl: Direct loading kernel failed: %s, falling back to repack kernel\n", e.what());
            }
        }

        try {
            if (actual_d != actual_dv && padded_d > 0 && padded_dv > 0) {
                // MLA case: DQK != DV (e.g., GLM-4.7-Flash with K=576, V=512)
                // Use padded kernel with V_FROM_K optimization when V is a view of K
                
                // Add more MLA combinations here as needed
                GGML_SYCL_DEBUG("ggml_sycl: XMX MLA not supported for DQK=%ld DV=%ld, falling back\n", actual_d, actual_dv);
            } else if (actual_d == padded_d && actual_dv == padded_dv) {
                // Native head size - use direct loading kernel (replaces old coopmat)
                switch (actual_d) {
                    case 32:
                        ggml_sycl_op_flash_attn_coopmat_direct<32, 32>(ctx, dst);
                        return;
                    case 64:
                        ggml_sycl_op_flash_attn_coopmat_direct<64, 64>(ctx, dst);
                        return;
                    case 96:
                        ggml_sycl_op_flash_attn_coopmat_direct<96, 96>(ctx, dst);
                        return;
                    case 128:
                        ggml_sycl_op_flash_attn_coopmat_direct<128, 128>(ctx, dst);
                        return;
                    case 256:
                        ggml_sycl_op_flash_attn_coopmat_direct<256, 256>(ctx, dst);
                        return;
                    case 512:
                        ggml_sycl_op_flash_attn_coopmat_direct<512, 512>(ctx, dst);
                        return;
                    default:
                        break;
                }
            } else if (padded_d > 0 && padded_dv > 0) {
                // Padded head size - use padded kernel (DQK == DV case)
                switch (actual_d) {
                    case 40:
                        ggml_sycl_op_flash_attn_coopmat_padded<40, 40, 64, 64>(ctx, dst);
                        return;
                    case 48:
                        ggml_sycl_op_flash_attn_coopmat_padded<48, 48, 64, 64>(ctx, dst);
                        return;
                    case 56:
                        ggml_sycl_op_flash_attn_coopmat_padded<56, 56, 64, 64>(ctx, dst);
                        return;
                    case 72:
                        ggml_sycl_op_flash_attn_coopmat_padded<72, 72, 80, 80>(ctx, dst);
                        return;
                    case 88:
                        ggml_sycl_op_flash_attn_coopmat_padded<88, 88, 96, 96>(ctx, dst);
                        return;
                    case 104:
                        ggml_sycl_op_flash_attn_coopmat_padded<104, 104, 112, 112>(ctx, dst);
                        return;
                    default:
                        break;
                }
            }
            // Fall back to non-XMX path for unsupported head sizes
            GGML_SYCL_DEBUG("ggml_sycl: XMX flash attention not supported for head size DQK=%ld DV=%ld, falling back\n", actual_d, actual_dv);
        } catch (const std::exception& e) {
            GGML_SYCL_DEBUG("ggml_sycl: XMX flash attention failed: %s, falling back to non-XMX path\n", e.what());
#ifdef GGML_SYCL_USE_INTEL_ONEMKL
            if (DQK == 576 && DV == 512) {
                ggml_sycl_op_flash_attn_mkl<576, 512>(ctx, dst);
                return;
            }
#endif
        }
    }
#endif

    // Fallback to basic flash attention implementation
    const int64_t fallback_dqk = Q->ne[0];
    const int64_t fallback_dv = V->ne[0];

    if (fallback_dqk != fallback_dv) {
        GGML_ABORT("ggml_sycl: Flash attention fallback requires DQK==DV, got DQK=%ld DV=%ld\n",
                fallback_dqk, fallback_dv);
    }
    
    switch (fallback_dqk) {
        case 32:
            ggml_sycl_op_flash_attn_2< 32,  32>(ctx, dst);
            break;
        case 64:
            ggml_sycl_op_flash_attn_2< 64,  64>(ctx, dst);
            break;
        case 80:
            ggml_sycl_op_flash_attn_2< 80,  80>(ctx, dst);
            break;
        case 96:
            ggml_sycl_op_flash_attn_2< 96,  96>(ctx, dst);
            break;
        case 112:
            ggml_sycl_op_flash_attn_2<112, 112>(ctx, dst);
            break;
        case 128:
            ggml_sycl_op_flash_attn_2<128, 128>(ctx, dst);
            break;
        case 256:
            ggml_sycl_op_flash_attn_2<256, 256>(ctx, dst);
            break;
        case 512:
            ggml_sycl_op_flash_attn_2<512, 512>(ctx, dst);
            break;
        default:
            GGML_SYCL_DEBUG("Warning: Unsupported head size %ld — skipping op\n", fallback_dqk);
            break;
    }
}
