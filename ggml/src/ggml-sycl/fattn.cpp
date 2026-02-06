#include "./fattn.hpp"
#include "./fattn_kernel.hpp"
#include "./fattn_common.hpp"
#include "./fattn_fused.hpp"
#include "./common.hpp"

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
#include "fattn_xmx.hpp"
#endif

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
    static int override_kind = -2;
    if (override_kind == -2) {
        const char * env = getenv("GGML_SYCL_FLASH_ATTN_TILE");
        if (env != nullptr) {
            if (strcmp(env, "dg2") == 0 || strcmp(env, "n8") == 0) override_kind = 0;
            else if (strcmp(env, "pvc") == 0 || strcmp(env, "b60") == 0 || strcmp(env, "n16") == 0) override_kind = 1;
            else override_kind = -1;
        } else {
            override_kind = -1;
        }
    }
    if (override_kind >= 0) {
        return override_kind == 0 ? xmx_tile_kind::tile_dg2 : xmx_tile_kind::tile_pvc;
    }
    auto matrix_info = device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    int nsize = -1;
    for (const auto& comb : matrix_info) {
        if (comb.atype == sycl::ext::oneapi::experimental::matrix::matrix_type::bf16 &&
            comb.btype == sycl::ext::oneapi::experimental::matrix::matrix_type::bf16) {
            nsize = comb.nsize;
            break;
        }
    }
    if (nsize == 8) return xmx_tile_kind::tile_dg2;
    else if (nsize == 16) return xmx_tile_kind::tile_pvc;
    else return xmx_tile_kind::tile_pvc;
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

    // Mask and sinks are not yet supported in SYCL flash attention
    // These features cause NaN outputs and need further debugging
    // TODO: Re-enable after fixing mask/sinks correctness issues
    if (mask != nullptr && mask->data != nullptr) {
        return false;
    }
    if (sinks != nullptr && sinks->data != nullptr) {
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

            float* q_loc = &Qtile[0][0];
            float* k_loc = &Ktile[0][0];
            float* v_loc = &Vtile[0][0];
            float* s_loc = &Stile[0][0];
            float* p_loc = &Ptile[0];
            float* m_loc = &m_local[0];
            float* l_loc = &l_local[0];

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

            float * o_row = dst_d + (ptrdiff_t)(head_idx * N + row) * o_row_stride;

            if (l_val <= 0.0f) {
                // Fully masked row - output should be zero
                for (int col = 0; col < DV; ++col) {
                    o_row[col] = 0.0f;
                }
                return;
            }

            float inv_l = 1.0f / l_val;

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


// XMX flash attention implementations moved to fattn_xmx.cpp

#ifdef GGML_SYCL_USE_INTEL_ONEMKL
#include "fattn_mkl.hpp"
#endif

void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * sinks = dst->src[4];
    (void)K; // Used by direct loading path when SYCL_EXT_COOPERATIVE_MATRICES is defined

    GGML_SYCL_DEBUG("[SYCL][OP] call ggml_sycl_op_flash_attn: Q=[%ld,%ld,%ld] V=[%ld,%ld,%ld]\n",
            Q->ne[0], Q->ne[1], Q->ne[2], V->ne[0], V->ne[1], V->ne[2]);

    // Common tensor dimensions (used by all paths)
    const int64_t DQK = Q->ne[0];
    const int64_t DV = V->ne[0];
    const int64_t N = Q->ne[1];
    const int64_t N_kv = K->ne[1];
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;
    const int64_t n_splits = get_kv_split_count(N_kv, n_heads);
    bool is_f16 = (Q->type == GGML_TYPE_F16);

    // Extract mask and scale
    const ggml_tensor * mask = dst->src[3];
    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

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
    #endif // GGML_SYCL_USE_INTEL_ONEMKL

    // ============================================================================
    // Fused Single-Kernel Flash Attention (Graph-Compatible)
    // ============================================================================
    // This is a new implementation that fuses QK, softmax, and PV into a single
    // kernel without host-side split loops. It uses online softmax and direct
    // stride loading from ggml tensor layout.
    // Supported: head sizes up to 128 (shared memory constraints), F16/BF16/F32.
    // This replaces the host-side sequential KV split loop in the MKL path.
    if (DQK <= 128 && DV <= 128 && DQK == DV) {
        // Prepare mask pointer (only F32 mask supported for now)
        const float* mask_d = nullptr;
        int64_t mask_stride = 0;
        if (mask != nullptr && mask->data != nullptr) {
            if (mask->type == GGML_TYPE_F32) {
                mask_d = (const float*)mask->data;
                mask_stride = mask->nb[1] / sizeof(float);
            } else {
                // Fused path only supports F32 mask; other types will fall back
                mask_d = nullptr;
            }
        }

        // Compute tensor strides for direct loading
        fattn_tensor_strides strides = compute_tensor_strides(Q, K, V, dst);

        // Dispatch based on Q data type
        if (Q->type == GGML_TYPE_F16) {
            GGML_SYCL_DEBUG("ggml_sycl: Using fused flash attention (F16) DQK=%d DV=%d\n", DQK, DV);
            ggml_sycl_op_flash_attn_fused<sycl::half>(
                ctx.stream(),
                (const sycl::half*)Q, (const sycl::half*)K, (const sycl::half*)V,
                (float*)dst->data,
                N, N_kv, n_heads, n_kv_heads, gqa_ratio, scale,
                mask_d, mask_stride,
                sinks ? (const float*)sinks->data : nullptr,
                strides,
                DQK, DV);
            return;
        } else if (Q->type == GGML_TYPE_BF16) {
            #ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
            GGML_SYCL_DEBUG("ggml_sycl: Using fused flash attention (BF16) DQK=%d DV=%d\n", DQK, DV);
            ggml_sycl_op_flash_attn_fused<sycl::ext::oneapi::bfloat16>(
                ctx.stream(),
                (const sycl::ext::oneapi::bfloat16*)Q,
                (const sycl::ext::oneapi::bfloat16*)K,
                (const sycl::ext::oneapi::bfloat16*)V,
                (float*)dst->data,
                N, N_kv, n_heads, n_kv_heads, gqa_ratio, scale,
                mask_d, mask_stride,
                sinks ? (const float*)sinks->data : nullptr,
                strides,
                DQK, DV);
            return;
            #endif
        } else if (Q->type == GGML_TYPE_F32) {
            GGML_SYCL_DEBUG("ggml_sycl: Using fused flash attention (F32) DQK=%d DV=%d\n", DQK, DV);
            ggml_sycl_op_flash_attn_fused<float>(
                ctx.stream(),
                (const float*)Q, (const float*)K, (const float*)V,
                (float*)dst->data,
                N, N_kv, n_heads, n_kv_heads, gqa_ratio, scale,
                mask_d, mask_stride,
                sinks ? (const float*)sinks->data : nullptr,
                strides,
                DQK, DV);
            return;
        }
    }

    GGML_SYCL_DEBUG("ggml_sycl: Fused flash attention not applicable (DQK=%ld DV=%ld type=%s), falling back\n",
            DQK, DV, ggml_type_name(Q->type));

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

        if (use_direct_loading && direct_supported_shape && can_use_direct_loading(Q, K, V, dst) && n_splits == 1) {
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
            // XMX KV-split path for large contexts (n_splits > 1)
            if (n_splits > 1) {
                bool shape_supported = (actual_d == actual_dv && actual_d == padded_d && actual_dv == padded_d);
                if (shape_supported) {
                    if (actual_d == 576 && actual_dv == 512) {
                        if (is_f16) {
                            ggml_sycl_op_flash_attn_coopmat_kvsplit<576, 512, 8, 16, 16, 8, 16, fattn_input_type::f16, true>(ctx, dst);
                        } else {
                            ggml_sycl_op_flash_attn_coopmat_kvsplit<576, 512, 8, 16, 16, 8, 16, fattn_input_type::f32, true>(ctx, dst);
                        }
                        return;
                    }
                    switch (actual_d) {
                        case 32:
                            if (is_f16) {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<32, 32, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(ctx, dst);
                            } else {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<32, 32, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(ctx, dst);
                            }
                            return;
                        case 64:
                            if (is_f16) {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<64, 64, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(ctx, dst);
                            } else {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<64, 64, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(ctx, dst);
                            }
                            return;
                        case 96:
                            if (is_f16) {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<96, 96, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(ctx, dst);
                            } else {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<96, 96, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(ctx, dst);
                            }
                            return;
                        case 128:
                            if (is_f16) {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<128, 128, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(ctx, dst);
                            } else {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<128, 128, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(ctx, dst);
                            }
                            return;
                        case 256:
                            if (is_f16) {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<256, 256, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(ctx, dst);
                            } else {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<256, 256, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(ctx, dst);
                            }
                            return;
                        case 512:
                            if (is_f16) {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<512, 512, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(ctx, dst);
                            } else {
                                ggml_sycl_op_flash_attn_coopmat_kvsplit<512, 512, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(ctx, dst);
                            }
                            return;
                        default:
                            // Unsupported head size for split, continue to fallback
                            break;
                    }
                }
            }
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
