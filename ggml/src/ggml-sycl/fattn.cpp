#include "./fattn.hpp"
#include "./fattn_kernel.hpp"
#include "./fattn_common.hpp"

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
    // Supported sizes: 32, 64, 80, 96, 112, 128, 256, 512
    if (head_size <= 32) return 32;
    if (head_size <= 64) return 64;
    if (head_size <= 80) return 80;
    if (head_size <= 96) return 96;
    if (head_size <= 112) return 112;
    if (head_size <= 128) return 128;
    if (head_size <= 256) return 256;
    if (head_size <= 512) return 512;
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
// NOTE: The XMX kernel is currently EXPERIMENTAL and has correctness issues.
// It's disabled by default. Set GGML_SYCL_FLASH_ATTN_XMX=1 to enable for testing.
inline bool ggml_sycl_flash_attn_has_xmx(sycl::device device) {
#ifdef SYCL_EXT_COOPERATIVE_MATRICES
    // Check for basic cooperative matrix support
    if (!device.has(sycl::aspect::ext_intel_gpu_eu_simd_width) ||
        !device.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }
    // Check environment variable to enable experimental XMX kernel
    static bool xmx_enabled = false;
    static bool env_checked = false;
    if (!env_checked) {
        const char* env = getenv("GGML_SYCL_FLASH_ATTN_XMX");
        xmx_enabled = (env != nullptr && strcmp(env, "1") == 0);
        env_checked = true;
        if (xmx_enabled) {
            fprintf(stderr, "ggml_sycl: XMX flash attention ENABLED (experimental)\n");
        }
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
    return xmx_tile_kind::tile_8x8;  // Fallback, won't be used
#endif
}

// Check if device should use oneMKL BLAS for flash attention (fallback path)
// This is kept as a fallback option but the XMX path is now preferred
inline bool ggml_sycl_flash_attn_use_mkl(sycl::device device) {
#ifdef GGML_SYCL_USE_INTEL_ONEMKL
    // Only use MKL if XMX is not available
    if (!ggml_sycl_flash_attn_has_xmx(device)) {
        return true;
    }
#endif
    return false;
}


bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    static FILE *dbg = fopen("/tmp/llama_sycl_support.txt", "a");
    fprintf(dbg, "ggml_sycl_flash_attn_ext_supported: called\n");
    fflush(dbg);
    
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale, max_bias, logit_softcap;

    std::memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if( max_bias != 0.0f || logit_softcap != 0.0f){
        fprintf(dbg, "  rejected: max_bias=%f or logit_softcap=%f\n", max_bias, logit_softcap);
        fflush(dbg);
        return false;
    }

    if (Q == nullptr || K == nullptr || V == nullptr) {
        fprintf(dbg, "  rejected: null tensor\n");
        fflush(dbg);
        return false;
    }

    // Causal masking support: check if mask is present but not custom
    // For custom masks, we still need to check if we support the specific type
    if (mask != nullptr && mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) {
        fprintf(dbg, "  rejected: mask present and not F32/F16, type=%d, ne[0]=%ld\n", mask->type, mask->ne[0]);
        fflush(dbg);
        return false;
    }
    
    // Support F32 or FP16 inputs (FP16 will be dequantized to F32)
    // Also support mixed types: Q can be F32 while K/V are F16 (common pattern)
    const bool is_all_f32 = (Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && V->type == GGML_TYPE_F32);
    const bool is_all_f16 = (Q->type == GGML_TYPE_F16 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    const bool is_mixed_f32_q = (Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    if (!is_all_f32 && !is_all_f16 && !is_mixed_f32_q) {
        fprintf(stderr, "ggml_sycl_flash_attn_ext_supported: rejected, type not F32/F16 (Q=%d, K=%d, V=%d)\n",
                Q->type, K->type, V->type);
        fprintf(dbg, "  rejected: type not F32/F16 (Q=%d, K=%d, V=%d)\n", Q->type, K->type, V->type);
        fflush(dbg);
        return false;
    }

    int64_t DQK = Q->ne[0];
    int64_t DV  = V->ne[0];

    if (DQK != DV){
        fprintf(stderr, "ggml_sycl_flash_attn_ext_supported: rejected, DQK != DV (%ld != %ld)\n", DQK, DV);
        fprintf(dbg, "  rejected: DQK != DV (%ld != %ld)\n", DQK, DV);
        fflush(dbg);
        return false;
    }

    if (!is_head_size_supported(DV)){
        fprintf(stderr, "ggml_sycl_flash_attn_ext_supported: rejected, unsupported head size %ld (not paddable)\n", DV);
        fprintf(dbg, "  rejected: unsupported head size %ld\n", DV);
        fflush(dbg);
        return false;
    }

    // GQA support: n_kv_heads can be less than n_heads
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];

    // n_heads must be divisible by n_kv_heads for GQA/MQA
    if (n_heads % n_kv_heads != 0) {
        return false;
    }

    // GQA ratio (number of Q heads per K/V head)
    const int gqa_ratio = n_heads / n_kv_heads;

    // For now, only support ratio of 1 (MHA) or small ratios
    // Larger ratios would require different memory access patterns
    if (gqa_ratio > 8) {
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

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];
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
        float * Q_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
        float * K_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_kv_heads * sizeof(float), *stream);
        float * V_d_f32_alloc = (float *) sycl::malloc_device(N * DV * n_kv_heads * sizeof(float), *stream);

        // Get strides in elements for FP16
        const ptrdiff_t q_row_stride_f16 = Q->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t k_row_stride_f16 = K->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t v_row_stride_f16 = V->nb[1] / (ptrdiff_t)sizeof(sycl::half);

        // Dequantize Q heads (all n_heads heads)
        for (int64_t head = 0; head < n_heads; ++head) {
            const int64_t n_elements = N * DQK;
            const ptrdiff_t row_stride = q_row_stride_f16;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / row_stride;
                        const int64_t col = idx % row_stride;
                        Q_d_f32_alloc[head * N * DQK + idx] = static_cast<float>(Q_d[head * N * q_row_stride_f16 + row * q_row_stride_f16 + col]);
                    }
                });
            });
        }

        // Dequantize K heads
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N * DQK;
            const ptrdiff_t row_stride = k_row_stride_f16;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / row_stride;
                        const int64_t col = idx % row_stride;
                        K_d_f32_alloc[head * N * DQK + idx] = static_cast<float>(K_d[head * N * k_row_stride_f16 + row * k_row_stride_f16 + col]);
                    }
                });
            });
        }

        // Dequantize V heads
        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N * DV;
            const ptrdiff_t row_stride = v_row_stride_f16;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / row_stride;
                        const int64_t col = idx % row_stride;
                        V_d_f32_alloc[head * N * DV + idx] = static_cast<float>(V_d[head * N * v_row_stride_f16 + row * v_row_stride_f16 + col]);
                    }
                });
            });
        }

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
    const int Tc = (N + Bc - 1) / Bc;

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

            if (row0 >= (int) N || col0 >= (int) N) {
                return;
            }

            // Calculate base pointers for this head
            const float* Q_block = Q_d_f32 + (ptrdiff_t)(head_idx * N + row0) * q_row_stride;
            const float* K_block = K_d_f32 + (ptrdiff_t)(kv_head_idx * N + col0) * k_row_stride;
            const float* V_block = V_d_f32 + (ptrdiff_t)(kv_head_idx * N + col0) * v_row_stride;
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
}

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_coopmat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];

    const bool is_f16 = (Q->type == GGML_TYPE_F16);
    const float * Q_d_f32;
    const float * K_d_f32;
    const float * V_d_f32;

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;

    const ptrdiff_t q_row_stride = Q->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t k_row_stride = K->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t v_row_stride = V->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t o_row_stride = dst->nb[1] / (ptrdiff_t)sizeof(float);

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Handle FP16 by dequantizing to F32 first
    if (is_f16) {
        const sycl::half * Q_d = (const sycl::half *) Q->data;
        const sycl::half * K_d = (const sycl::half *) K->data;
        const sycl::half * V_d = (const sycl::half *) V->data;

        float * Q_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_heads * sizeof(float), *stream);
        float * K_d_f32_alloc = (float *) sycl::malloc_device(N * DQK * n_kv_heads * sizeof(float), *stream);
        float * V_d_f32_alloc = (float *) sycl::malloc_device(N * DV * n_kv_heads * sizeof(float), *stream);

        const ptrdiff_t q_row_stride_f16 = Q->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t k_row_stride_f16 = K->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t v_row_stride_f16 = V->nb[1] / (ptrdiff_t)sizeof(sycl::half);

        for (int64_t head = 0; head < n_heads; ++head) {
            const int64_t n_elements = N * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / q_row_stride_f16;
                        const int64_t col = idx % q_row_stride_f16;
                        Q_d_f32_alloc[head * N * DQK + idx] = static_cast<float>(Q_d[head * N * q_row_stride_f16 + row * q_row_stride_f16 + col]);
                    }
                });
            });
        }

        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N * DQK;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / k_row_stride_f16;
                        const int64_t col = idx % k_row_stride_f16;
                        K_d_f32_alloc[head * N * DQK + idx] = static_cast<float>(K_d[head * N * k_row_stride_f16 + row * k_row_stride_f16 + col]);
                    }
                });
            });
        }

        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N * DV;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / v_row_stride_f16;
                        const int64_t col = idx % v_row_stride_f16;
                        V_d_f32_alloc[head * N * DV + idx] = static_cast<float>(V_d[head * N * v_row_stride_f16 + row * v_row_stride_f16 + col]);
                    }
                });
            });
        }

        Q_d_f32 = Q_d_f32_alloc;
        K_d_f32 = K_d_f32_alloc;
        V_d_f32 = V_d_f32_alloc;
    } else {
        Q_d_f32 = (const float *) Q->data;
        K_d_f32 = (const float *) K->data;
        V_d_f32 = (const float *) V->data;
    }

    float *       dst_d = (float *) dst->data;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;
    const int Tc = (N + BLOCK_N - 1) / BLOCK_N;

    float * l_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);

    // Work-group size must divide global size evenly on Intel GPUs
    // Each work-group processes one BLOCK_M x BLOCK_N tile
    // Use 64 threads per work-group (4 subgroups of 16)
    constexpr int THREADS_PER_WG = 64;
    
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
    constexpr int V_STRIDE = DQK + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;
    constexpr int V_T_STRIDE = BLOCK_N;  // V^T buffer stride
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr size_t BF16_BYTES = (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE + 
                                    BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE +
                                    DQK * V_T_STRIDE) * sizeof(sycl::half);  // Include shVT
    constexpr size_t FLOAT_BYTES = (BLOCK_M * S_STRIDE + BLOCK_M * 3 + BLOCK_M * DQK) * sizeof(float);
    constexpr size_t SHMEM_SIZE = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    if (tile_kind == xmx_tile_kind::tile_8x8) {
        // DG2/Arc B60: Use 8x8x16 tiles
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_dg2<DQK>(
                    it,
                    Q_d_f32, K_d_f32, V_d_f32, dst_d,
                    l_d, m_d,
                    N, n_heads, n_kv_heads, gqa_ratio,
                    scale, 1, 0,
                    shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            });
        });
    } else {
        // PVC and other GPUs: Use 16x16x16 tiles
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_pvc<DQK>(
                    it,
                    Q_d_f32, K_d_f32, V_d_f32, dst_d,
                    l_d, m_d,
                    N, n_heads, n_kv_heads, gqa_ratio,
                    scale, 1, 0,
                    shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            });
        });
    }

    // Note: The coopmat kernel already normalizes by dividing by rowSum internally,
    // so no post-normalization pass is needed. l_d/m_d are unused for XMX path.

    if (is_f16) {
        sycl::free((void *)Q_d_f32, *stream);
        sycl::free((void *)K_d_f32, *stream);
        sycl::free((void *)V_d_f32, *stream);
    }
    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}

// Padded flash attention using XMX for head sizes that don't match tile dimensions
// HEAD_DIM: actual head dimension (e.g., 40)
// PADDED_HEAD_DIM: padded dimension for XMX compute (e.g., 64)
template<int64_t HEAD_DIM, int64_t PADDED_HEAD_DIM>
void ggml_sycl_op_flash_attn_coopmat_padded(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];

    const bool is_f16 = (Q->type == GGML_TYPE_F16);
    const float * Q_d_f32;
    const float * K_d_f32;
    const float * V_d_f32;

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N = Q->ne[1];
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;

    const ptrdiff_t q_row_stride = Q->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t k_row_stride = K->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t v_row_stride = V->nb[1] / (ptrdiff_t)sizeof(float);
    const ptrdiff_t o_row_stride = dst->nb[1] / (ptrdiff_t)sizeof(float);

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Handle FP16 by dequantizing to F32 first
    if (is_f16) {
        const sycl::half * Q_d = (const sycl::half *) Q->data;
        const sycl::half * K_d = (const sycl::half *) K->data;
        const sycl::half * V_d = (const sycl::half *) V->data;

        float * Q_d_f32_alloc = (float *) sycl::malloc_device(N * HEAD_DIM * n_heads * sizeof(float), *stream);
        float * K_d_f32_alloc = (float *) sycl::malloc_device(N * HEAD_DIM * n_kv_heads * sizeof(float), *stream);
        float * V_d_f32_alloc = (float *) sycl::malloc_device(N * HEAD_DIM * n_kv_heads * sizeof(float), *stream);

        const ptrdiff_t q_row_stride_f16 = Q->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t k_row_stride_f16 = K->nb[1] / (ptrdiff_t)sizeof(sycl::half);
        const ptrdiff_t v_row_stride_f16 = V->nb[1] / (ptrdiff_t)sizeof(sycl::half);

        for (int64_t head = 0; head < n_heads; ++head) {
            const int64_t n_elements = N * HEAD_DIM;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / q_row_stride_f16;
                        const int64_t col = idx % q_row_stride_f16;
                        Q_d_f32_alloc[head * N * HEAD_DIM + idx] = static_cast<float>(Q_d[head * N * q_row_stride_f16 + row * q_row_stride_f16 + col]);
                    }
                });
            });
        }

        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N * HEAD_DIM;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / k_row_stride_f16;
                        const int64_t col = idx % k_row_stride_f16;
                        K_d_f32_alloc[head * N * HEAD_DIM + idx] = static_cast<float>(K_d[head * N * k_row_stride_f16 + row * k_row_stride_f16 + col]);
                    }
                });
            });
        }

        for (int64_t head = 0; head < n_kv_heads; ++head) {
            const int64_t n_elements = N * HEAD_DIM;
            stream->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < n_elements) {
                        const int64_t row = idx / v_row_stride_f16;
                        const int64_t col = idx % v_row_stride_f16;
                        V_d_f32_alloc[head * N * HEAD_DIM + idx] = static_cast<float>(V_d[head * N * v_row_stride_f16 + row * v_row_stride_f16 + col]);
                    }
                });
            });
        }

        Q_d_f32 = Q_d_f32_alloc;
        K_d_f32 = K_d_f32_alloc;
        V_d_f32 = V_d_f32_alloc;
    } else {
        Q_d_f32 = (const float *) Q->data;
        K_d_f32 = (const float *) K->data;
        V_d_f32 = (const float *) V->data;
    }

    float *       dst_d = (float *) dst->data;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;
    const int Tc = (N + BLOCK_N - 1) / BLOCK_N;

    float * l_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);
    float * m_d = (float *) sycl::malloc_device(N * n_heads * sizeof(float), *stream);

    constexpr int THREADS_PER_WG = 64;
    sycl::range<2> global(Tr * THREADS_PER_WG, n_heads);
    sycl::range<2> local(THREADS_PER_WG, 1);

    xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(stream->get_device());

    // Calculate shared memory size based on PADDED_HEAD_DIM
    // Layout: [bf16: Q, K, V, P, VT] + [float: S, rowMax, rowSum, rowAlpha, shAcc]
    constexpr int Q_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int V_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr int V_T_STRIDE = BLOCK_N;  // Stride for V^T (stored transposed)
    // shVT has PADDED_HEAD_DIM rows (each row is BLOCK_N elements)
    constexpr size_t BF16_BYTES = (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE +
                                    BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE +
                                    PADDED_HEAD_DIM * V_T_STRIDE) * sizeof(sycl::half);
    constexpr size_t FLOAT_BYTES = (BLOCK_M * S_STRIDE + BLOCK_M * 3 + BLOCK_M * PADDED_HEAD_DIM) * sizeof(float);
    constexpr size_t SHMEM_SIZE = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    if (tile_kind == xmx_tile_kind::tile_8x8) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_dg2_padded<HEAD_DIM, PADDED_HEAD_DIM>(
                    it,
                    Q_d_f32, K_d_f32, V_d_f32, dst_d,
                    l_d, m_d,
                    N, n_heads, n_kv_heads, gqa_ratio,
                    scale, 1, 0,
                    o_row_stride,
                    shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            });
        });
    } else {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                flash_attn_coopmat_kernel_pvc_padded<HEAD_DIM, PADDED_HEAD_DIM>(
                    it,
                    Q_d_f32, K_d_f32, V_d_f32, dst_d,
                    l_d, m_d,
                    N, n_heads, n_kv_heads, gqa_ratio,
                    scale, 1, 0,
                    o_row_stride,
                    shmem.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            });
        });
    }

    // Note: The coopmat kernel already normalizes by dividing by rowSum internally,
    // so no post-normalization pass is needed. l_d/m_d are unused for XMX path.

    if (is_f16) {
        sycl::free((void *)Q_d_f32, *stream);
        sycl::free((void *)K_d_f32, *stream);
        sycl::free((void *)V_d_f32, *stream);
    }
    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}
#endif // SYCL_EXT_COOPERATIVE_MATRICES

#ifdef GGML_SYCL_USE_INTEL_ONEMKL
// oneMKL-based flash attention for Arc B60 (Xe2/Battlemage)
// Uses oneMKL BLAS for QK^T and PV GEMMs, bypassing cooperative matrix issues
template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int64_t N = Q->ne[1];
    const int64_t n_heads = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio = n_heads / n_kv_heads;

    const float * Q_d = (const float *) Q->data;
    const float * K_d = (const float *) K->data;
    const float * V_d = (const float *) V->data;
    float * O_d = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    scale *= sycl::rsqrt(static_cast<float>(DQK));

    float * S_scores = (float *) sycl::malloc_device(N * N * sizeof(float), *stream);
    float * S_max = (float *) sycl::malloc_device(N * sizeof(float), *stream);
    float * S_sum = (float *) sycl::malloc_device(N * sizeof(float), *stream);

    // Leading dimensions - must satisfy: lda >= k (no transpose), ldb >= k (no transpose), ldc >= n
    const int64_t lda_q = DQK;  // Q row stride (columns)
    const int64_t ldb_k = DQK;  // K row stride (columns)
    const int64_t ldc_s = N;    // S is N x N
    const int64_t lda_v = DV;   // V row stride (columns)
    const int64_t ldc_o = DV;   // O is N x DV

    for (int64_t head = 0; head < n_heads; ++head) {
        const int64_t kv_head = head / gqa_ratio;

        const float * Q_head = Q_d + head * N * DQK;
        const float * K_head = K_d + kv_head * N * DQK;
        const float * V_head = V_d + kv_head * N * DV;
        float * O_head = O_d + head * N * DV;

        stream->submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() {
                for (int64_t i = 0; i < N * N; ++i) {
                    S_scores[i] = 0.0f;
                }
            });
        });

        // Q @ K^T GEMM: (N x DQK) @ (DQK x N) -> (N x N)
        // For row-major: C = A @ B^T
        // Here: S = Q @ K^T means S[i,j] = sum_k Q[i,k] * K[j,k]
        // With no transpose: C = A @ B, so we need K^T as input
        // But we have K in (N x DQK) layout, so we transpose K
        oneapi::mkl::blas::gemm(*stream,
             oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
             N, N, DQK,
             scale,
             Q_head, lda_q,
             K_head, ldb_k,
             0.0f,
             S_scores, ldc_s);

        stream->wait_and_throw();

        // Softmax: row-wise max, exp, sum, normalize
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const int64_t row = idx[0];

                float row_max = -1.0e20f;
                for (int64_t col = 0; col < N; ++col) {
                    row_max = sycl::fmax(row_max, S_scores[row * N + col]);
                }
                S_max[row] = row_max;

                float sum = 0.0f;
                for (int64_t col = 0; col < N; ++col) {
                    S_scores[row * N + col] = sycl::exp(sycl::fmax(S_scores[row * N + col] - row_max, -20.0f));
                    sum += S_scores[row * N + col];
                }
                S_sum[row] = sum;
            });
        });

        stream->wait_and_throw();

        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(N * N), [=](sycl::id<1> idx) {
                const int64_t i = idx[0];
                const int64_t row = i / N;
                float row_sum = S_sum[row];
                if (row_sum > 1.0e-10f) {
                    S_scores[i] /= row_sum;
                }
            });
        });

        stream->wait_and_throw();

        // P @ V GEMM: (N x N) @ (N x DV) -> (N x DV)
        oneapi::mkl::blas::gemm(*stream,
             oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
             N, DV, N,
             1.0f,
             S_scores, N,
             V_head, lda_v,
             0.0f,
             O_head, ldc_o);
    }

    stream->wait_and_throw();

    sycl::free(S_scores, *stream);
    sycl::free(S_max, *stream);
    sycl::free(S_sum, *stream);
}
#endif // GGML_SYCL_USE_INTEL_ONEMKL

void ggml_sycl_op_flash_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
    // Try XMX path first if device supports it
    static bool sycl_use_xmx = false;
    static bool xmx_checked = false;

    if (!xmx_checked) {
        sycl::device device = ctx.stream()->get_device();
        sycl_use_xmx = ggml_sycl_flash_attn_has_xmx(device);
        xmx_checked = true;
        xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(device);
        const char * tile_str = (tile_kind == xmx_tile_kind::tile_8x8) ? "8x8x16 (DG2/Arc)" : "16x16x16 (PVC)";
        fprintf(stderr, "ggml_sycl: XMX detection: device=%s, has_xmx=%d, tile_kind=%s\n", 
                device.get_info<sycl::info::device::name>().c_str(), sycl_use_xmx, tile_str);
        if (sycl_use_xmx) {
            fprintf(stderr, "ggml_sycl: Using XMX (cooperative matrix) for flash attention with %s tiles\n", tile_str);
        }
    }

    if (sycl_use_xmx) {
        const int64_t actual_d = Q->ne[0];
        const int64_t padded_d = get_padded_head_size(actual_d);
        
        try {
            if (actual_d == padded_d) {
                // Native head size - use non-padded kernel
                switch (actual_d) {
                    case 32:
                        GGML_ASSERT(V->ne[0] == 32);
                        ggml_sycl_op_flash_attn_coopmat<32, 32>(ctx, dst);
                        return;
                    case 64:
                        GGML_ASSERT(V->ne[0] == 64);
                        ggml_sycl_op_flash_attn_coopmat<64, 64>(ctx, dst);
                        return;
                    case 96:
                        GGML_ASSERT(V->ne[0] == 96);
                        ggml_sycl_op_flash_attn_coopmat<96, 96>(ctx, dst);
                        return;
                    case 128:
                        GGML_ASSERT(V->ne[0] == 128);
                        ggml_sycl_op_flash_attn_coopmat<128, 128>(ctx, dst);
                        return;
                    case 256:
                        GGML_ASSERT(V->ne[0] == 256);
                        ggml_sycl_op_flash_attn_coopmat<256, 256>(ctx, dst);
                        return;
                    default:
                        break;
                }
            } else if (padded_d > 0) {
                // Padded head size - use padded kernel
                switch (actual_d) {
                    case 40:
                        GGML_ASSERT(V->ne[0] == 40);
                        ggml_sycl_op_flash_attn_coopmat_padded<40, 64>(ctx, dst);
                        return;
                    case 48:
                        GGML_ASSERT(V->ne[0] == 48);
                        ggml_sycl_op_flash_attn_coopmat_padded<48, 64>(ctx, dst);
                        return;
                    case 56:
                        GGML_ASSERT(V->ne[0] == 56);
                        ggml_sycl_op_flash_attn_coopmat_padded<56, 64>(ctx, dst);
                        return;
                    case 72:
                        GGML_ASSERT(V->ne[0] == 72);
                        ggml_sycl_op_flash_attn_coopmat_padded<72, 80>(ctx, dst);
                        return;
                    case 88:
                        GGML_ASSERT(V->ne[0] == 88);
                        ggml_sycl_op_flash_attn_coopmat_padded<88, 96>(ctx, dst);
                        return;
                    case 104:
                        GGML_ASSERT(V->ne[0] == 104);
                        ggml_sycl_op_flash_attn_coopmat_padded<104, 112>(ctx, dst);
                        return;
                    default:
                        break;
                }
            }
            // Fall back to non-XMX path for unsupported head sizes
            fprintf(stderr, "ggml_sycl: XMX flash attention not supported for head size %ld, falling back\n", Q->ne[0]);
        } catch (const std::exception& e) {
            // XMX kernel failed, fall back to non-XMX path
            // Disable XMX for subsequent calls to avoid repeated failures
            sycl_use_xmx = false;
            fprintf(stderr, "ggml_sycl: XMX kernel failed: %s, falling back to non-XMX path\n", e.what());
        }
    }
#endif

#ifdef GGML_SYCL_USE_INTEL_ONEMKL
    // Try oneMKL path for Arc B60 and similar devices
    static bool sycl_use_mkl = false;
    static bool mkl_checked = false;

    if (!mkl_checked) {
        sycl::device device = ctx.stream()->get_device();
        sycl_use_mkl = ggml_sycl_flash_attn_use_mkl(device);
        mkl_checked = true;
        if (sycl_use_mkl) {
            fprintf(stderr, "ggml_sycl: Using oneMKL BLAS for flash attention (device=%s)\n",
                    device.get_info<sycl::info::device::name>().c_str());
        }
    }

    if (sycl_use_mkl) {
        const int64_t actual_d = Q->ne[0];
        const int64_t padded_d = get_padded_head_size(actual_d);
        
        if (actual_d == padded_d) {
            switch (actual_d) {
                case 32:
                    GGML_ASSERT(V->ne[0] == 32);
                    ggml_sycl_op_flash_attn_mkl<32, 32>(ctx, dst);
                    return;
                case 64:
                    GGML_ASSERT(V->ne[0] == 64);
                    ggml_sycl_op_flash_attn_mkl<64, 64>(ctx, dst);
                    return;
                case 80:
                    GGML_ASSERT(V->ne[0] == 80);
                    ggml_sycl_op_flash_attn_mkl<80, 80>(ctx, dst);
                    return;
                case 96:
                    GGML_ASSERT(V->ne[0] == 96);
                    ggml_sycl_op_flash_attn_mkl<96, 96>(ctx, dst);
                    return;
                case 112:
                    GGML_ASSERT(V->ne[0] == 112);
                    ggml_sycl_op_flash_attn_mkl<112, 112>(ctx, dst);
                    return;
                case 128:
                    GGML_ASSERT(V->ne[0] == 128);
                    ggml_sycl_op_flash_attn_mkl<128, 128>(ctx, dst);
                    return;
                case 256:
                    GGML_ASSERT(V->ne[0] == 256);
                    ggml_sycl_op_flash_attn_mkl<256, 256>(ctx, dst);
                    return;
                default:
                    break;
            }
        } else if (padded_d > 0) {
            // Padded head sizes require XMX kernel with padding support
            // oneMKL path does not support padding - fall through to basic path
            fprintf(stderr, "ggml_sycl: oneMKL path does not support padded head sizes, falling back\n");
        }
        fprintf(stderr, "ggml_sycl: oneMKL flash attention not supported for head size %ld, falling back\n", actual_d);
    }
#endif

    switch (Q->ne[0]) {
        case 32:
            GGML_ASSERT(V->ne[0] == 32);
            ggml_sycl_op_flash_attn_2< 32,  32>(ctx, dst);
            break;
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_sycl_op_flash_attn_2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_sycl_op_flash_attn_2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_sycl_op_flash_attn_2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_sycl_op_flash_attn_2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_sycl_op_flash_attn_2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_sycl_op_flash_attn_2<256, 256>(ctx, dst);
            break;
        case 576:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_sycl_op_flash_attn_2<512, 512>(ctx, dst);
            break;
        default:
            fprintf(stderr, "Warning: Unsupported head size %ld — skipping op\n", Q->ne[0]);
            break;
    }
}

