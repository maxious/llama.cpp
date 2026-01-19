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


// Check if device supports XMX (cooperative matrix) for flash attention
inline bool ggml_sycl_flash_attn_has_xmx(sycl::device device) {
#ifdef SYCL_EXT_COOPERATIVE_MATRICES
    return device.has(sycl::aspect::ext_intel_gpu_eu_simd_width) &&
           device.has(sycl::aspect::ext_intel_matrix);
#else
    return false;
#endif
}


bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale, max_bias, logit_softcap;

    std::memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if( max_bias != 0.0f || logit_softcap != 0.0f){
        return false;
    }

    if (Q == nullptr || K == nullptr || V == nullptr) {
        return false;
    }

    // Causal masking support: check if mask is present but not custom
    // For custom masks, we still need to check if we support the specific type
    if (mask != 0) {
        // For now, support only causal-like masks (simple boolean or no mask)
        // Custom attention masks with arbitrary patterns require more work
        return false;
    }
    
    // Support F32 or FP16 inputs (FP16 will be dequantized to F32)
    const bool is_f32 = (Q->type == GGML_TYPE_F32 && K->type == GGML_TYPE_F32 && V->type == GGML_TYPE_F32);
    const bool is_f16 = (Q->type == GGML_TYPE_F16 && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    if (!is_f32 && !is_f16) {
        return false;
    }

    int64_t DQK = Q->ne[0];
    int64_t DV  = V->ne[0];

    if (DQK != DV){
        return false;
    }

    if (DV != 32 && DV != 64 && DV != 80 && DV != 96 && DV != 112 && DV != 128 && DV != 256 && DV != 512){
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

    sycl::range<2> global(Tr * BLOCK_M, Tc * n_heads);
    sycl::range<2> local(BLOCK_M, 32);

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> shmem(sycl::range<1>(8192), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            flash_attn_coopmat_kernel<DQK>(
                it,
                Q_d_f32, K_d_f32, V_d_f32, dst_d,
                l_d, m_d,
                N, n_heads, n_kv_heads, gqa_ratio,
                scale, 1, 0,
                shmem.get_pointer()
            );
        });
    });

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(N * n_heads), [=](sycl::id<1> id) {
            int idx = id[0];
            int head_idx = idx / N;
            int row = idx % N;
            float l_val = l_d[idx];

            if (l_val <= 0.0f) return;

            float inv_l = 1.0f / l_val;
            float * o_row = dst_d + (ptrdiff_t)(head_idx * N + row) * o_row_stride;

            for (int col = 0; col < DV; ++col) {
                o_row[col] *= inv_l;
            }
        });
    });

    if (is_f16) {
        sycl::free((void *)Q_d_f32, *stream);
        sycl::free((void *)K_d_f32, *stream);
        sycl::free((void *)V_d_f32, *stream);
    }
    sycl::free(l_d, *stream);
    sycl::free(m_d, *stream);
}
#endif // SYCL_EXT_COOPERATIVE_MATRICES

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
        if (sycl_use_xmx) {
            fprintf(stderr, "ggml_sycl: Using XMX (cooperative matrix) for flash attention\n");
        }
    }

    if (sycl_use_xmx) {
        switch (Q->ne[0]) {
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
        // Fall back to non-XMX path for unsupported head sizes
        fprintf(stderr, "ggml_sycl: XMX flash attention not supported for head size %ld, falling back\n", Q->ne[0]);
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

