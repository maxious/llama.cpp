#ifndef GGML_SYCL_FATTN_TILED_HPP
#define GGML_SYCL_FATTN_TILED_HPP

#include "fattn_common.hpp"
#include "gemm_tiled.hpp"

#include <sycl/sycl.hpp>

// Tiled Flash Attention (Graph-Compatible)
// Decomposes Flash Attention into GEMM + Softmax + GEMM using tiled kernels.
// Optimized for small batch sizes where Fused Kernel is inefficient.

template <typename T>
void softmax_fwd_kernel(sycl::nd_item<2> it,
                        const float * __restrict__ S,
                        const T * __restrict__ mask,
                        float * __restrict__ P,
                        const int     N,
                        const int     N_kv,
                        const float   scale,
                        const float   max_bias,
                        const int64_t mask_stride_row) {
    const int row = it.get_group(0);
    if (row >= N) {
        return;
    }

    const int tid        = it.get_local_id(1);
    const int block_size = it.get_local_range(1);

    // Find Max
    float max_val = -INFINITY;
    for (int i = tid; i < N_kv; i += block_size) {
        float val = S[row * N_kv + i] * scale;
        if (mask) {
            // Mask layout: [N_kv, N] typically, or [N, N_kv]
            // If stride is provided, use it.
            // Row-major: mask[row * stride + col]
            // If mask is [N, N_kv], stride = N_kv.
            val += (float) mask[row * mask_stride_row + i];
        }
        max_val = sycl::fmax(max_val, val);
    }
    max_val = sycl::reduce_over_group(it.get_group(), max_val, sycl::maximum<float>());

    // Sum Exp
    float sum_exp = 0.0f;
    for (int i = tid; i < N_kv; i += block_size) {
        float val = S[row * N_kv + i] * scale;
        if (mask) {
            val += (float) mask[row * mask_stride_row + i];
        }
        val               = sycl::exp(val - max_val);
        P[row * N_kv + i] = val;
        sum_exp += val;
    }
    sum_exp = sycl::reduce_over_group(it.get_group(), sum_exp, sycl::plus<float>());

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < N_kv; i += block_size) {
        P[row * N_kv + i] *= inv_sum;
    }
}

template <typename T_Q, typename T_K, typename T_V, typename T_Mask>
void ggml_sycl_op_flash_attn_tiled(ggml_backend_sycl_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int N          = Q->ne[1];
    const int N_kv       = K->ne[1];
    const int DQK        = Q->ne[0];
    const int DV         = V->ne[0];
    const int n_heads    = Q->ne[2];
    const int n_kv_heads = K->ne[2];

    float scale;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();

    // Initialize buffer pool if needed
    if (!ctx.fattn_buffers) {
        ctx.fattn_buffers = std::make_unique<flash_attn_buffers>();
    }

    size_t  S_size = (size_t) n_heads * N * N_kv * sizeof(float);
    float * S_ptr  = ctx.fattn_buffers->get_S(S_size, stream);
    float * P_ptr  = S_ptr;

    // Pointer arrays for GQA
    size_t   ptr_bytes = n_heads * sizeof(float *);
    float ** Q_ptrs    = nullptr;
    float ** K_ptrs    = nullptr;
    float ** S_ptrs    = nullptr;
    float ** V_ptrs    = nullptr;
    float ** O_ptrs    = nullptr;

    if (n_heads != n_kv_heads) {
        ctx.fattn_buffers->get_ptrs(ptr_bytes, stream, &Q_ptrs, &K_ptrs, &S_ptrs, &V_ptrs, &O_ptrs);
    }

    if (n_heads == n_kv_heads) {
        int64_t stride_Q = Q->nb[2] / sizeof(T_Q);
        int64_t stride_K = K->nb[2] / sizeof(T_K);
        int64_t stride_S = N * N_kv;

        launch_gemm_tiled_batched<true>(stream, (const float *) Q->data, (const float *) K->data, S_ptr, N, N_kv, DQK,
                                        1.0f, 0.0f, n_heads, DQK, DQK, N_kv, stride_Q, stride_K, stride_S);
    } else {
        std::vector<float *> h_Q(n_heads);
        std::vector<float *> h_K(n_heads);
        std::vector<float *> h_S(n_heads);

        for (int i = 0; i < n_heads; ++i) {
            h_Q[i]   = (float *) Q->data + i * (Q->nb[2] / sizeof(T_Q));
            int kv_i = i / (n_heads / n_kv_heads);
            h_K[i]   = (float *) K->data + kv_i * (K->nb[2] / sizeof(T_K));
            h_S[i]   = S_ptr + i * N * N_kv;
        }

        stream->memcpy(Q_ptrs, h_Q.data(), ptr_bytes);
        stream->memcpy(K_ptrs, h_K.data(), ptr_bytes);
        stream->memcpy(S_ptrs, h_S.data(), ptr_bytes);

        launch_gemm_tiled_batched_indirect<true>(stream, (const float **) Q_ptrs, (const float **) K_ptrs, S_ptrs, N,
                                                 N_kv, DQK, 1.0f, 0.0f, n_heads, DQK, DQK, N_kv);
    }

    int            total_rows = N * n_heads;
    int            wg_size    = 256;
    sycl::range<2> global_soft(total_rows, wg_size);
    sycl::range<2> local_soft(1, wg_size);

    int64_t mask_stride = mask ? mask->nb[1] / sizeof(T_Mask) : 0;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<2>(global_soft, local_soft), [=](sycl::nd_item<2> it) {
            softmax_fwd_kernel<T_Mask>(it, S_ptr, mask ? (const T_Mask *) mask->data : nullptr, P_ptr, N, N_kv, scale,
                                       0.0f, mask_stride);
        });
    });

    float * O_ptr = (float *) dst->data;

    if (n_heads == n_kv_heads) {
        int64_t stride_P      = N * N_kv;
        int64_t stride_V      = V->nb[2] / sizeof(T_V);
        int64_t stride_O_head = dst->nb[1] / 4;
        int64_t stride_O_seq  = dst->nb[2] / 4;

        launch_gemm_tiled_batched<false>(stream, P_ptr, (const float *) V->data, O_ptr, N, DV, N_kv, 1.0f, 0.0f,
                                         n_heads, N_kv, DV, stride_O_seq, stride_P, stride_V, stride_O_head);
    } else {
        std::vector<float *> h_S(n_heads);  // P reuse S
        std::vector<float *> h_V(n_heads);
        std::vector<float *> h_O(n_heads);

        int64_t stride_O_head = dst->nb[1] / 4;
        int64_t stride_O_seq  = dst->nb[2] / 4;

        for (int i = 0; i < n_heads; ++i) {
            h_S[i]   = P_ptr + i * N * N_kv;
            int kv_i = i / (n_heads / n_kv_heads);
            h_V[i]   = (float *) V->data + kv_i * (V->nb[2] / sizeof(T_V));
            h_O[i]   = O_ptr + i * stride_O_head;
        }
        stream->memcpy(S_ptrs, h_S.data(), ptr_bytes);  // Reuse S_ptrs
        stream->memcpy(V_ptrs, h_V.data(), ptr_bytes);
        stream->memcpy(O_ptrs, h_O.data(), ptr_bytes);

        launch_gemm_tiled_batched_indirect<false>(stream, (const float **) S_ptrs, (const float **) V_ptrs, O_ptrs, N,
                                                  DV, N_kv, 1.0f, 0.0f, n_heads, N_kv, DV, stride_O_seq);
    }

    // Buffers are pooled in ctx.fattn_buffers and will be reused on next call
}

#endif
