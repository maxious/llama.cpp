#include "./fattn.hpp"
#include "./fattn_kernel.hpp"
#include "./fattn_common.hpp"
#include "./common.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <sycl/sycl.hpp>

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

// Get the appropriate tile kind for flash attention XMX kernel
// DG2/Arc uses 8x8 tiles, PVC uses 16x16 tiles
inline xmx_tile_kind ggml_sycl_flash_attn_get_tile_kind(sycl::device device) {
#ifdef SYCL_EXT_COOPERATIVE_MATRICES
    return ggml_sycl_get_tile_kind(device);
#else
    return xmx_tile_kind::tile_dg2;  // Fallback, won't be used
#endif
}

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

    // Sinks (attention sinks / StreamingLLM) not yet implemented in SYCL flash attention
    if (sinks != nullptr) {
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
        const sycl::half * V_d = (const sycl::half *) V->data;

        // Allocate F32 buffers on device
        Q_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
        K_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DQK * n_kv_heads * sizeof(float), *stream);
        V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);

        // Get strides in elements for FP16
        const ptrdiff_t q_row_stride_f16 = Q->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t k_row_stride_f16 = K->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t v_row_stride_f16 = V->nb[1] / (ptrdiff_t)sizeof(sycl::half);

        // Get head strides from tensor layout
        const ptrdiff_t q_head_stride_f16 = Q->nb[2] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t k_head_stride_f16 = K->nb[2] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t v_head_stride_f16 = V->nb[2] / (ptrdiff_t)sizeof(sycl::half);

        // Dequantize Q heads (all n_heads heads)
        // Use logical width DQK for index decomposition, physical stride for source access
        for (int64_t head = 0; head < n_heads; ++head) {
            const int64_t n_elements = N * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / DQK;  // Logical width for decomposition
                        const int64_t col = idx % DQK;
                        // Use head stride and physical row stride for source access
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

        // Wait for dequantization to complete before proceeding
        stream->wait();

        Q_d_f32 = Q_d_f32_alloc;
        K_d_f32 = K_d_f32_alloc;
        V_d_f32 = V_d_f32_alloc;
    } else {
        // F32 case - direct pointer cast
        Q_d_f32 = (const float *) Q->data;
        K_d_f32 = (const float *) K->data;
        V_d_f32 = (const float *) V->data;
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
        sycl::free(V_d_f32_alloc, *stream);
    }
}

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_coopmat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];  // Mask tensor with precomputed causal/SWA masking

    // Check types for each tensor separately - they can be mixed (Q=F32, K/V=F16 is common)
    const bool q_is_f16 = (Q->type == GGML_TYPE_F16);
    const bool k_is_f16 = (K->type == GGML_TYPE_F16);
    const bool v_is_f16 = (V->type == GGML_TYPE_F16);

    const float * Q_d_f32;
    const float * K_d_f32;
    const float * V_d_f32;
    float * Q_d_f32_alloc = nullptr;
    float * K_d_f32_alloc = nullptr;
    float * V_d_f32_alloc = nullptr;

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;

    // Use the correct KV sequence length (K/V can have different N than Q for KV cache)
    const int64_t N_kv = K->ne[1];

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Dequantize FP16 tensors to F32 with correct tensor layout handling
    // Input tensor layout: [head_dim, seq_len, n_heads, batch]
    // - Q[d, n, h] = Q_data[d + n*n_heads*DQK + h*DQK] (from nb: stride1=n_heads*DQK*elem_size, stride2=DQK*elem_size)
    // - K[d, n, h] = K_data[d + n*n_kv_heads*DQK + h*DQK]
    // - V[d, n, h] = V_data[d + n*n_kv_heads*DV + h*DV]
    // 
    // Output to kernel: contiguous [N_seq, HEAD_DIM] per head
    // - Q_out[head * N * DQK + seq * DQK + dim]
    // - K_out[head * N_kv * DQK + seq * DQK + dim]
    // - V_out[head * N_kv * DV + seq * DV + dim]
    
    if (q_is_f16) {
        const sycl::half * Q_d = (const sycl::half *) Q->data;
        Q_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
        
        const int64_t q_stride_seq = Q->nb[1] / sizeof(sycl::half);
        const int64_t q_stride_head = Q->nb[2] / sizeof(sycl::half);
        
        for (int64_t head = 0; head < n_heads; ++head) {
            const int64_t n_elements = N * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t seq = idx / DQK;   // Sequence position
                        const int64_t dim = idx % DQK;   // Head dimension
                        // Source: Q[dim, seq, head] = Q_d[dim + head*q_stride_head + seq*q_stride_seq]
                        // Dest: Q_out[head * N * DQK + seq * DQK + dim]
                        Q_d_f32_alloc[head * N * DQK + seq * DQK + dim] = static_cast<float>(
                            Q_d[dim + head * q_stride_head + seq * q_stride_seq]);
                    }
                });
            });
        }
        Q_d_f32 = Q_d_f32_alloc;
    } else {
        const float * Q_d = (const float *) Q->data;
        Q_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
        
        const int64_t q_stride_seq = Q->nb[1] / sizeof(float);
        const int64_t q_stride_head = Q->nb[2] / sizeof(float);
        
        for (int64_t head = 0; head < n_heads; ++head) {
            const int64_t n_elements = N * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t seq = idx / DQK;
                        const int64_t dim = idx % DQK;
                        Q_d_f32_alloc[head * N * DQK + seq * DQK + dim] = 
                            Q_d[dim + head * q_stride_head + seq * q_stride_seq];
                    }
                });
            });
        }
        Q_d_f32 = Q_d_f32_alloc;
    }
    
    if (k_is_f16) {
        const sycl::half * K_d = (const sycl::half *) K->data;
        K_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DQK * n_kv_heads * sizeof(float), *stream);
        
        const int64_t k_stride_seq = K->nb[1] / sizeof(sycl::half);
        const int64_t k_stride_head = K->nb[2] / sizeof(sycl::half);
        
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N_kv * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t seq = idx / DQK;
                        const int64_t dim = idx % DQK;
                        K_d_f32_alloc[head * N_kv * DQK + seq * DQK + dim] = static_cast<float>(
                            K_d[dim + head * k_stride_head + seq * k_stride_seq]);
                    }
                });
            });
        }
        K_d_f32 = K_d_f32_alloc;
    } else {
        const float * K_d = (const float *) K->data;
        K_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DQK * n_kv_heads * sizeof(float), *stream);
        
        const int64_t k_stride_seq = K->nb[1] / sizeof(float);
        const int64_t k_stride_head = K->nb[2] / sizeof(float);
        
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N_kv * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t seq = idx / DQK;
                        const int64_t dim = idx % DQK;
                        K_d_f32_alloc[head * N_kv * DQK + seq * DQK + dim] = 
                            K_d[dim + head * k_stride_head + seq * k_stride_seq];
                    }
                });
            });
        }
        K_d_f32 = K_d_f32_alloc;
    }
    
    if (v_is_f16) {
        const sycl::half * V_d = (const sycl::half *) V->data;
        V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
        
        const int64_t v_stride_seq = V->nb[1] / sizeof(sycl::half);
        const int64_t v_stride_head = V->nb[2] / sizeof(sycl::half);
        
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N_kv * DV;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t seq = idx / DV;
                        const int64_t dim = idx % DV;
                        V_d_f32_alloc[head * N_kv * DV + seq * DV + dim] = static_cast<float>(
                            V_d[dim + head * v_stride_head + seq * v_stride_seq]);
                    }
                });
            });
        }
        V_d_f32 = V_d_f32_alloc;
    } else {
        const float * V_d = (const float *) V->data;
        V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
        
        const int64_t v_stride_seq = V->nb[1] / sizeof(float);
        const int64_t v_stride_head = V->nb[2] / sizeof(float);
        
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N_kv * DV;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t seq = idx / DV;
                        const int64_t dim = idx % DV;
                        V_d_f32_alloc[head * N_kv * DV + seq * DV + dim] = 
                            V_d[dim + head * v_stride_head + seq * v_stride_seq];
                    }
                });
            });
        }
        V_d_f32 = V_d_f32_alloc;
    }
    
    // Always wait since we now always do reordering
    stream->wait();

    float *       dst_d = (float *) dst->data;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;

    float * l_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);

    // Work-group size must divide global size evenly on Intel GPUs
    // Each work-group processes one BLOCK_M x BLOCK_N tile
    // Use 64 threads per work-group (4 subgroups of 16)
    constexpr int THREADS_PER_WG = 64;
    
    static bool first_call = true;
    if (first_call && getenv("GGML_SYCL_FLASH_ATTN_DEBUG")) {
        GGML_SYCL_DEBUG("ggml_sycl: XMX flash_attn_coopmat N=%ld N_kv=%ld n_heads=%ld n_kv_heads=%ld DQK=%ld scale=%f\n",
                N, N_kv, n_heads, n_kv_heads, DQK, scale);
        GGML_SYCL_DEBUG("ggml_sycl: types: Q=%d K=%d V=%d (q_f16=%d k_f16=%d v_f16=%d)\n",
                Q->type, K->type, V->type, q_is_f16, k_is_f16, v_is_f16);

        // Debug: check input Q/K/V values after dequantization
        float debug_q[256], debug_k[256], debug_v[256];
        stream->memcpy(debug_q, Q_d_f32, 256 * sizeof(float)).wait();
        stream->memcpy(debug_k, K_d_f32, 256 * sizeof(float)).wait();
        stream->memcpy(debug_v, V_d_f32, 256 * sizeof(float)).wait();
        GGML_SYCL_DEBUG("ggml_sycl: Q[0:4] = [%f, %f, %f, %f]\n",
                debug_q[0], debug_q[1], debug_q[2], debug_q[3]);
        GGML_SYCL_DEBUG("ggml_sycl: K[0:4] = [%f, %f, %f, %f]\n",
                debug_k[0], debug_k[1], debug_k[2], debug_k[3]);
        GGML_SYCL_DEBUG("ggml_sycl: V[0:4] = [%f, %f, %f, %f]\n",
                debug_v[0], debug_v[1], debug_v[2], debug_v[3]);
        // Also print V[1] (second KV position) - offset by HEAD_DIM=128
        float debug_v2[4];
        stream->memcpy(debug_v2, V_d_f32 + 128, 4 * sizeof(float)).wait();
        GGML_SYCL_DEBUG("ggml_sycl: V[kv=1, 0:4] = [%f, %f, %f, %f]\n",
                debug_v2[0], debug_v2[1], debug_v2[2], debug_v2[3]);
        first_call = false;
    }
    
    // Global size: one work-group per row-block per head
    // Dimension 0: Tr work-groups (each handling BLOCK_M rows)
    // Dimension 1: n_heads work-groups
    sycl::range<2> global(Tr * THREADS_PER_WG, n_heads);
    sycl::range<2> local(THREADS_PER_WG, 1);

    // Get tile kind based on device architecture
    xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(stream->get_device());

    // Calculate shared memory size based on HEAD_DIM
    // Layout: [bf16: Q, K, V, P, VT] + [float: S, rowMax, rowSum, rowAlpha, shAcc]
    constexpr int Q_STRIDE = DQK + 8;
    constexpr int K_STRIDE = DQK + 8;
    constexpr int V_STRIDE = DV + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;
    constexpr int V_T_STRIDE = BLOCK_N + 8;  // V col-major: [BLOCK_N rows, HEAD_DIM cols]
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr size_t BF16_BYTES = (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE +
                                    BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE +
                                    DV * V_T_STRIDE) * sizeof(sycl::half);  // Include shVT
    constexpr size_t FLOAT_BYTES = (BLOCK_M * S_STRIDE + BLOCK_M * 3 + BLOCK_M * DV) * sizeof(float);
    constexpr size_t SHMEM_SIZE = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    // Get mask pointer and stride (mask layout is [N_kv, N] or nullptr if no mask)
    const float * mask_d = nullptr;
    float * mask_d_f32_alloc = nullptr;
    int64_t mask_stride = N_kv;  // Stride between query rows in the mask
    if (mask != nullptr && mask->data != nullptr) {
        // Mask can be F32 or F16 - check type and convert if needed
        if (mask->type == GGML_TYPE_F32) {
            mask_d = (const float *) mask->data;
            mask_stride = mask->nb[1] / sizeof(float);  // Row stride in floats
        } else if (mask->type == GGML_TYPE_F16) {
            // Convert F16 mask to F32
            const int64_t mask_n_kv = mask->ne[0];  // N_kv dimension
            const int64_t mask_n_q = mask->ne[1];   // N dimension (padded)
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
            stream->wait();
            
            mask_d = mask_d_f32_alloc;
            mask_stride = mask_n_kv;  // Contiguous after conversion
        }
    }
    
    static bool mask_debug = true;
    if (mask_debug && mask_d != nullptr && getenv("GGML_SYCL_FLASH_ATTN_DEBUG")) {
        GGML_SYCL_DEBUG("ggml_sycl: Using mask tensor: ne=[%ld, %ld, %ld], stride=%ld, type=%d\n",
                mask->ne[0], mask->ne[1], mask->ne[2], mask_stride, mask->type);
        // Debug: print first few mask values
        float debug_mask[16];
        stream->memcpy(debug_mask, mask_d, 16 * sizeof(float)).wait();
        GGML_SYCL_DEBUG("ggml_sycl: mask[row0, 0:4] = [%f, %f, %f, %f]\n",
                debug_mask[0], debug_mask[1], debug_mask[2], debug_mask[3]);
        // Check mask for row 1 (query position 1) - offset by mask_stride
        float debug_mask_row1[8];
        stream->memcpy(debug_mask_row1, mask_d + mask_stride, 8 * sizeof(float)).wait();
        GGML_SYCL_DEBUG("ggml_sycl: mask[row1, 0:8] = [%f, %f, %f, %f, %f, %f, %f, %f]\n",
                debug_mask_row1[0], debug_mask_row1[1], debug_mask_row1[2], debug_mask_row1[3],
                debug_mask_row1[4], debug_mask_row1[5], debug_mask_row1[6], debug_mask_row1[7]);
        mask_debug = false;
    }

    // Allocate temporary output buffer - kernel writes [head, seq, dim] contiguously
    // Then we scatter to the actual output layout [dim, head, seq, batch]
    float * O_temp = (float *) sycl::malloc_device(N * DV * n_heads * sizeof(float), *stream);

if (tile_kind == xmx_tile_kind::tile_dg2) {
        // DG2/Arc B60: Use 8x8x16 tiles
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_n8<DQK, DV>(
                    it,
                    Q_d_f32, K_d_f32, V_d_f32, O_temp,
                    l_d, m_d,
                    N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                    scale, mask_d, mask_stride,
                    shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            });
        });
    } else {
        // PVC and other GPUs: Use 16x16x16 tiles
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_n16<DQK, DV>(
                    it,
                    Q_d_f32, K_d_f32, V_d_f32, O_temp,
                    l_d, m_d,
                    N, N_kv, n_heads, n_kv_heads, gqa_ratio,
                    scale, mask_d, mask_stride,
                    shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            });
        });
    }

    stream->wait();
    
    // Scatter from temp output to correct layout
    // Kernel writes: O_temp[head * N * DV + seq * DV + dim]
    // Output layout: O[dim, head, seq] = dst_d[dim + head*DV + seq*n_heads*DV]
    for (int64_t head = 0; head < n_heads; ++head) {
        const int64_t n_elements = N * DV;
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (idx < n_elements) {
                    const int64_t seq = idx / DV;
                    const int64_t dim = idx % DV;
                    // Source: O_temp[head * N * DV + seq * DV + dim]
                    // Dest: O[dim + head*DV + seq*n_heads*DV]
                    dst_d[dim + head * DV + seq * n_heads * DV] = 
                        O_temp[head * N * DV + seq * DV + dim];
                }
            });
        });
    }
    
    stream->wait();
    
    // Debug: check output after scatter
    static bool output_checked = false;
    if (!output_checked) {
        float debug_o[16];
        stream->memcpy(debug_o, dst_d, 16 * sizeof(float)).wait();
        GGML_SYCL_DEBUG("ggml_sycl: XMX Output O[0:4] = [%f, %f, %f, %f]\n",
                debug_o[0], debug_o[1], debug_o[2], debug_o[3]);
        output_checked = true;
    }

    // Free allocated buffers
    sycl::free(O_temp, *stream);
    if (Q_d_f32_alloc) sycl::free(Q_d_f32_alloc, *stream);
    if (K_d_f32_alloc) sycl::free(K_d_f32_alloc, *stream);
    if (V_d_f32_alloc) sycl::free(V_d_f32_alloc, *stream);
    if (mask_d_f32_alloc) sycl::free(mask_d_f32_alloc, *stream);
    
    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}

// Padded flash attention using XMX for head sizes that don't match tile dimensions
// HEAD_DIM: actual head dimension (e.g., 40)
// PADDED_HEAD_DIM: padded dimension for XMX compute (e.g., 64)
template<int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM>
void ggml_sycl_op_flash_attn_coopmat_padded(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Check types for each tensor separately - they can be mixed (Q=F32, K/V=F16 is common)
    const bool q_is_f16 = (Q->type == GGML_TYPE_F16);
    const bool k_is_f16 = (K->type == GGML_TYPE_F16);
    const bool v_is_f16 = (V->type == GGML_TYPE_F16);
    
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
    
    // Flatten batch into heads for kernel processing
    const int64_t n_heads = n_heads_per_batch * batch;
    const int64_t n_kv_heads = n_kv_heads_per_batch * batch;

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    GGML_SYCL_DEBUG("Flash attn padded: Q shape [%ld, %ld, %ld, %ld], K shape [%ld, %ld, %ld, %ld]\n",
        Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3]);
    GGML_SYCL_DEBUG("Flash attn padded: Q strides [%ld, %ld, %ld, %ld], K strides [%ld, %ld, %ld, %ld]\n",
        (int64_t)Q->nb[0], (int64_t)Q->nb[1], (int64_t)Q->nb[2], (int64_t)Q->nb[3],
        (int64_t)K->nb[0], (int64_t)K->nb[1], (int64_t)K->nb[2], (int64_t)K->nb[3]);
    GGML_SYCL_DEBUG("Flash attn padded: n_heads=%ld (per_batch=%ld), n_kv_heads=%ld (per_batch=%ld), batch=%ld, gqa_ratio=%ld\n",
        n_heads, n_heads_per_batch, n_kv_heads, n_kv_heads_per_batch, batch, gqa_ratio);

    // Dequantize/reorder tensors to contiguous per-head format
    // Input layout: [head_dim, seq_len, n_heads, batch]
    // Output layout: contiguous [N_seq, HEAD_DIM] per head (with batch flattened into heads)
    
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
    
    if (v_is_f16) {
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
    
    stream->wait();

    float * dst_d = (float *) dst->data;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;

    float * l_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);

    constexpr int THREADS_PER_WG = 64;
    sycl::range<2> global(Tr * THREADS_PER_WG, n_heads);
    sycl::range<2> local(THREADS_PER_WG, 1);

    xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(stream->get_device());

    constexpr int Q_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int V_STRIDE = PADDED_V_HEAD_DIM + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr int V_T_STRIDE = BLOCK_N + 8;
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
            stream->wait();
            
            mask_d = mask_d_f32_alloc;
            mask_stride = mask_n_kv;
        }
    }

    // Allocate temp output buffer and output stride for kernel
    float * O_temp = (float *) sycl::malloc_device(N * V_HEAD_DIM * n_heads * sizeof(float), *stream);
    const int o_row_stride = V_HEAD_DIM;

if (tile_kind == xmx_tile_kind::tile_dg2) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_n8_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM>(
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
        // PVC and other GPUs: Use 16x16x16 tiles
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_n16_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM>(
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

    stream->wait();
    
    // Scatter from temp output to correct layout using actual tensor strides
    // Kernel writes: O_temp[head_total * N * V_HEAD_DIM + seq * V_HEAD_DIM + dim]
    // where head_total = batch * n_heads_per_batch + head
    // Output layout uses dst->nb[] strides to handle batch dimension properly
    const int64_t dst_stride_seq = dst->nb[1] / sizeof(float);
    const int64_t dst_stride_head = dst->nb[2] / sizeof(float);
    const int64_t dst_stride_batch = dst->nb[3] / sizeof(float);
    
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t head = 0; head < n_heads_per_batch; ++head) {
            const int64_t head_total = b * n_heads_per_batch + head;
            const int64_t n_elements = N * V_HEAD_DIM;
            stream->submit([&](sycl::handler& cgh) {
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
        }
    }
    
    stream->wait();

    sycl::free(O_temp, *stream);
    if (Q_d_f32_alloc) sycl::free(Q_d_f32_alloc, *stream);
    if (K_d_f32_alloc) sycl::free(K_d_f32_alloc, *stream);
    if (V_d_f32_alloc) sycl::free(V_d_f32_alloc, *stream);
    if (mask_d_f32_alloc) sycl::free(mask_d_f32_alloc, *stream);
    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}
#endif // SYCL_EXT_COOPERATIVE_MATRICES

#ifdef GGML_SYCL_USE_INTEL_ONEMKL
// Optimized oneMKL-based flash attention for Arc B60 (Xe2/Battlemage)
// Performance optimizations:
// 1. Batched GEMM for all heads in one call (eliminates 40 separate GEMM launches)
// 2. Fused softmax kernel (mask + max + exp + sum + normalize in one pass)
// 3. Parallel output scatter (all heads at once)
// 4. Minimal synchronization (only 3 barriers total instead of 160+)
template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t N = Q->ne[1];       // Query sequence length
    const int64_t N_kv = K->ne[1];    // Key/Value sequence length (can differ from N)
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;

    dpct::queue_ptr stream = ctx.stream();

    // Debug: print tensor info on first call (disabled for normal operation)
    static bool first_mkl_call = true;
    if (first_mkl_call && getenv("GGML_SYCL_FLASH_ATTN_DEBUG")) {
        GGML_SYCL_DEBUG("ggml_sycl MKL path (OPTIMIZED): Q ne=[%ld,%ld,%ld,%ld] type=%d\n",
                Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3], Q->type);
        GGML_SYCL_DEBUG("ggml_sycl MKL path: N=%ld N_kv=%ld n_heads=%ld n_kv_heads=%ld gqa_ratio=%ld\n",
                N, N_kv, n_heads, n_kv_heads, gqa_ratio);
        first_mkl_call = false;
    }

    // Handle mixed precision: Q may be F32, K/V may be F16
    const bool q_is_f16 = (Q->type == GGML_TYPE_F16);
    const bool k_is_f16 = (K->type == GGML_TYPE_F16);
    const bool v_is_f16 = (V->type == GGML_TYPE_F16);

    float * Q_d_f32_alloc = nullptr;
    float * K_d_f32_alloc = nullptr;
    float * V_d_f32_alloc = nullptr;

    Q_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
    if (q_is_f16) {
        const sycl::half * Q_f16 = (const sycl::half *) Q->data;
        const int64_t q_stride_seq = Q->nb[1] / sizeof(sycl::half);
        const int64_t q_stride_head = Q->nb[2] / sizeof(sycl::half);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N * DQK * n_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N * DQK);
                const int64_t rem = idx % (N * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                Q_d_f32_alloc[idx] = static_cast<float>(Q_f16[d + head * q_stride_head + n * q_stride_seq]);
            });
        });
    } else {
        const float * Q_f32 = (const float *) Q->data;
        const int64_t q_stride_seq = Q->nb[1] / sizeof(float);
        const int64_t q_stride_head = Q->nb[2] / sizeof(float);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N * DQK * n_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N * DQK);
                const int64_t rem = idx % (N * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                Q_d_f32_alloc[idx] = Q_f32[d + head * q_stride_head + n * q_stride_seq];
            });
        });
    }

    K_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DQK * n_kv_heads * sizeof(float), *stream);
    if (k_is_f16) {
        const sycl::half * K_f16 = (const sycl::half *) K->data;
        const int64_t k_stride_seq = K->nb[1] / sizeof(sycl::half);
        const int64_t k_stride_head = K->nb[2] / sizeof(sycl::half);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N_kv * DQK * n_kv_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N_kv * DQK);
                const int64_t rem = idx % (N_kv * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                K_d_f32_alloc[idx] = static_cast<float>(K_f16[d + head * k_stride_head + n * k_stride_seq]);
            });
        });
    } else {
        const float * K_f32 = (const float *) K->data;
        const int64_t k_stride_seq = K->nb[1] / sizeof(float);
        const int64_t k_stride_head = K->nb[2] / sizeof(float);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N_kv * DQK * n_kv_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N_kv * DQK);
                const int64_t rem = idx % (N_kv * DQK);
                const int64_t n = rem / DQK;
                const int64_t d = rem % DQK;
                K_d_f32_alloc[idx] = K_f32[d + head * k_stride_head + n * k_stride_seq];
            });
        });
    }

    V_d_f32_alloc = (float *) sycl::malloc_device(N_kv * DV * n_kv_heads * sizeof(float), *stream);
    if (v_is_f16) {
        const sycl::half * V_f16 = (const sycl::half *) V->data;
        const int64_t v_stride_seq = V->nb[1] / sizeof(sycl::half);
        const int64_t v_stride_head = V->nb[2] / sizeof(sycl::half);
        stream->submit([&](sycl::handler& cgh) {
            const int64_t total = N_kv * DV * n_kv_heads;
            cgh.parallel_for(sycl::range<1>((total + 255) / 256 * 256), [=](sycl::item<1> it) {
                const int64_t idx = it.get_id(0);
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
                const int64_t idx = it.get_id(0);
                if (idx >= total) return;
                const int64_t head = idx / (N_kv * DV);
                const int64_t rem = idx % (N_kv * DV);
                const int64_t n = rem / DV;
                const int64_t d = rem % DV;
                V_d_f32_alloc[idx] = V_f32[d + head * v_stride_head + n * v_stride_seq];
            });
        });
    }

    float * O_d = (float *) dst->data;

    // Output tensor layout from ggml_flash_attn_ext: ne = { DV, n_heads, N, batch }
    // So nb[1] = stride between heads, nb[2] = stride between sequence positions
    const int64_t o_stride_head = dst->nb[1] / sizeof(float);  // DV for contiguous
    const int64_t o_stride_seq = dst->nb[2] / sizeof(float);   // DV * n_heads for contiguous

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Get mask pointer and stride if available
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

    // Wait for all data preparation to complete (single sync point)
    stream->wait();

    // Allocate S for ALL heads at once: [n_heads x N x N_kv]
    float * S_scores = (float *) sycl::malloc_device(n_heads * N * N_kv * sizeof(float), *stream);
    
    // Allocate output temp for ALL heads: [n_heads x N x DV]
    float * O_temp = (float *) sycl::malloc_device(n_heads * N * DV * sizeof(float), *stream);

    // Zero S_scores using memset (much faster than single_task loop)
    stream->memset(S_scores, 0, n_heads * N * N_kv * sizeof(float));

    // ============================================================
    // PHASE 1: Batched QK^T GEMM for all heads
    // ============================================================
    // For GQA: multiple Q heads share the same K head
    // We process all n_heads Q heads, mapping each to its corresponding KV head
    
    // Row-major leading dimensions
    const int64_t lda_q = DQK;     // Q is [N x DQK]
    const int64_t lda_k = DQK;     // K is [N_kv x DQK]
    const int64_t lda_s = N_kv;    // S is [N x N_kv]
    const int64_t lda_v = DV;      // V is [N_kv x DV]
    const int64_t lda_o = DV;      // O is [N x DV]

    // Strides between heads in the batched arrays
    const int64_t stride_q = N * DQK;        // Distance between Q heads
    const int64_t stride_k = N_kv * DQK;     // Distance between K heads
    const int64_t stride_s = N * N_kv;       // Distance between S heads
    const int64_t stride_v = N_kv * DV;      // Distance between V heads
    const int64_t stride_o = N * DV;         // Distance between O heads

    // For GQA, we need to handle the mapping from Q heads to KV heads
    // gemm_batch with stride doesn't directly support GQA, so we use group API
    // Group each set of gqa_ratio Q heads that share the same KV head
    
    if (gqa_ratio == 1) {
        // MHA: simple batched GEMM - all heads independent
        oneapi::mkl::blas::row_major::gemm_batch(*stream,
            oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
            N, N_kv, DQK,
            scale,
            Q_d_f32_alloc, lda_q, stride_q,
            K_d_f32_alloc, lda_k, stride_k,
            0.0f,
            S_scores, lda_s, stride_s,
            n_heads);
    } else {
        // GQA: process each KV head group separately
        // This is still much faster than 40 individual GEMMs
        for (int64_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            const int64_t q_head_start = kv_head * gqa_ratio;
            
            oneapi::mkl::blas::row_major::gemm_batch(*stream,
                oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
                N, N_kv, DQK,
                scale,
                Q_d_f32_alloc + q_head_start * stride_q, lda_q, stride_q,  // Q heads for this KV head
                K_d_f32_alloc + kv_head * stride_k, lda_k, 0,               // Same K head for all (stride=0)
                0.0f,
                S_scores + q_head_start * stride_s, lda_s, stride_s,
                gqa_ratio);
        }
    }

    // Wait for QK^T to complete
    stream->wait();

    // ============================================================
    // PHASE 2: Fused softmax for ALL heads in parallel
    // ============================================================
    // Each work-item handles one (head, query_position) pair
    // Fused: mask + max + exp + sum + normalize
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n_heads * N), [=](sycl::id<1> idx) {
            const int64_t head = idx[0] / N;
            const int64_t q = idx[0] % N;
            
            float * S_row = S_scores + head * stride_s + q * N_kv;
            
            // Pass 1: Apply mask and find max
            float row_max = -1.0e20f;
            for (int64_t k = 0; k < N_kv; ++k) {
                float s_val = S_row[k];
                if (mask_d != nullptr) {
                    s_val += mask_d[q * mask_stride + k];
                }
                S_row[k] = s_val;
                row_max = sycl::fmax(row_max, s_val);
            }
            
            // Pass 2: Compute exp and sum
            float sum = 0.0f;
            for (int64_t k = 0; k < N_kv; ++k) {
                float exp_val = sycl::exp(sycl::fmax(S_row[k] - row_max, -20.0f));
                S_row[k] = exp_val;
                sum += exp_val;
            }
            
            // Pass 3: Normalize
            if (sum > 1.0e-10f) {
                float inv_sum = 1.0f / sum;
                for (int64_t k = 0; k < N_kv; ++k) {
                    S_row[k] *= inv_sum;
                }
            }
        });
    });

    // Wait for softmax to complete
    stream->wait();

    // ============================================================
    // PHASE 3: Batched PV GEMM for all heads
    // ============================================================
    if (gqa_ratio == 1) {
        // MHA: simple batched GEMM
        oneapi::mkl::blas::row_major::gemm_batch(*stream,
            oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
            N, DV, N_kv,
            1.0f,
            S_scores, lda_s, stride_s,
            V_d_f32_alloc, lda_v, stride_v,
            0.0f,
            O_temp, lda_o, stride_o,
            n_heads);
    } else {
        // GQA: process each KV head group
        for (int64_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            const int64_t q_head_start = kv_head * gqa_ratio;
            
            oneapi::mkl::blas::row_major::gemm_batch(*stream,
                oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
                N, DV, N_kv,
                1.0f,
                S_scores + q_head_start * stride_s, lda_s, stride_s,
                V_d_f32_alloc + kv_head * stride_v, lda_v, 0,  // Same V head (stride=0)
                0.0f,
                O_temp + q_head_start * stride_o, lda_o, stride_o,
                gqa_ratio);
        }
    }

    // Wait for PV GEMM to complete
    stream->wait();

    // ============================================================
    // PHASE 4: Scatter output to correct layout for ALL heads
    // ============================================================
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(n_heads * N * DV), [=](sycl::id<1> idx) {
            const int64_t i = idx[0];
            const int64_t head = i / (N * DV);
            const int64_t rem = i % (N * DV);
            const int64_t row = rem / DV;
            const int64_t col = rem % DV;

            O_d[col + head * o_stride_head + row * o_stride_seq] = O_temp[i];
        });
    });

    stream->wait();

    // Cleanup
    sycl::free(S_scores, *stream);
    sycl::free(O_temp, *stream);
    if (mask_d_f32_alloc) sycl::free(mask_d_f32_alloc, *stream);
    if (Q_d_f32_alloc) sycl::free(Q_d_f32_alloc, *stream);
    if (K_d_f32_alloc) sycl::free(K_d_f32_alloc, *stream);
    if (V_d_f32_alloc) sycl::free(V_d_f32_alloc, *stream);
}
#endif // GGML_SYCL_USE_INTEL_ONEMKL

void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];

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

    // Handle mismatched DQK/DV case (e.g., GLM-4.7-Flash: K=576, V=512)
    // Always use oneMKL for mismatched dimensions since XMX doesn't support them
    if (DQK == 576 && DV == 512) {
        ggml_sycl_op_flash_attn_mkl<576, 512>(ctx, dst);
        return;
    }

    if (sycl_use_mkl) {
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
    }
#endif

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
    // Try XMX path if device supports it
    static bool sycl_use_xmx = false;
    static bool xmx_checked = false;

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

    if (sycl_use_xmx) {
        const int64_t actual_d = Q->ne[0];
        const int64_t actual_dv = V->ne[0];
        const int64_t padded_d = get_padded_head_size(actual_d);
        const int64_t padded_dv = get_padded_head_size(actual_dv);

        try {
            if (actual_d == padded_d && actual_dv == padded_dv) {
                // Native head size - use non-padded kernel
                switch (actual_d) {
                    case 32:
                        ggml_sycl_op_flash_attn_coopmat<32, 32>(ctx, dst);
                        return;
                    case 64:
                        ggml_sycl_op_flash_attn_coopmat<64, 64>(ctx, dst);
                        return;
                    case 96:
                        ggml_sycl_op_flash_attn_coopmat<96, 96>(ctx, dst);
                        return;
                    case 128:
                        ggml_sycl_op_flash_attn_coopmat<128, 128>(ctx, dst);
                        return;
                    case 256:
                        ggml_sycl_op_flash_attn_coopmat<256, 256>(ctx, dst);
                        return;
                    case 512:
                        ggml_sycl_op_flash_attn_coopmat<512, 512>(ctx, dst);
                        return;
                    default:
                        break;
                }
            } else if (padded_d > 0) {
                // Padded head size - use padded kernel
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
            GGML_SYCL_DEBUG("ggml_sycl: XMX flash attention not supported for head size %ld, falling back\n", Q->ne[0]);
        } catch (const std::exception& e) {
            // XMX kernel failed, fall back to oneMKL for mismatched K/V dimensions
            // Disable XMX for subsequent calls to avoid repeated failures
            sycl_use_xmx = false;
            GGML_SYCL_DEBUG("ggml_sycl: XMX kernel failed: %s, falling back to oneMKL path\n", e.what());

            // For mismatched K/V dimensions, use oneMKL directly
            const int64_t DQK = Q->ne[0];
            const int64_t DV = V->ne[0];

            if (DQK == 576 && DV == 512) {
                ggml_sycl_op_flash_attn_mkl<576, 512>(ctx, dst);
                return;
            } else if (DQK != DV) {
                GGML_SYCL_DEBUG("ggml_sycl: oneMKL path does not support DQK=%ld DV=%ld, no fallback available\n", DQK, DV);
                return;
            }
        }
    }
#endif

    // Fallback to basic flash attention implementation
    // Note: basic path doesn't support mismatched DQK/DV (that's handled by MKL above)
    const int64_t fallback_dqk = Q->ne[0];
    const int64_t fallback_dv = V->ne[0];

    if (fallback_dqk != fallback_dv) {
        GGML_SYCL_DEBUG("ggml_sycl: Flash attention fallback path requires DQK==DV, got DQK=%ld DV=%ld\n",
                fallback_dqk, fallback_dv);
        GGML_SYCL_DEBUG("ggml_sycl: This should have been handled by oneMKL path. Check GGML_SYCL_USE_INTEL_ONEMKL.\n");
        return;
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

