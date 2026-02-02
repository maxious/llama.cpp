#ifdef GGML_SYCL_USE_INTEL_ONEMKL
#include "fattn_common.hpp"
#include "common.hpp"
#include <oneapi/mkl.hpp>
#include <cmath>
#include <cstring>

template<int64_t DQK, int64_t DV>
void ggml_sycl_op_flash_attn_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int64_t N = Q->ne[1];
    const int64_t N_kv = K->ne[1];
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

    bool use_async_mem = false;
#ifdef GGML_SYCL_GRAPH
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

    // Allocate partials buffer
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

    // Process all KV splits
    for (int64_t split = 0; split < n_splits; ++split) {
        const int64_t kv_start = split * kv_per_split;
        const int64_t kv_end = std::min(kv_start + kv_per_split, N_kv);
        const int64_t kv_chunk_size = kv_end - kv_start;
        if (kv_chunk_size <= 0) continue;

        // Q @ K_chunk^T
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

        // Softmax and partials
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

        // S @ V to partials
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
    const int64_t o_stride_head = dst->nb[1] / sizeof(float);
    const int64_t o_stride_seq = dst->nb[2] / sizeof(float);

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_heads * N, DV), sycl::range<2>(1, DV)), 
            [=](sycl::nd_item<2> it) {
            flash_attn_combine_splits_kernel<DV>(it, partials, O_d, n_splits, n_heads, N, partial_size, o_stride_head, o_stride_seq);
        });
    });

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
