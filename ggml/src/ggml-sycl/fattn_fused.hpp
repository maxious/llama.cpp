// fattn_fused.hpp - Single-kernel fused Flash Attention for SYCL
// Online softmax, direct stride loading, graph-compatible
// Version: runtime DQK/DV (no template on dimensions)

#ifndef GGML_SYCL_FATTN_FUSED_HPP
#define GGML_SYCL_FATTN_FUSED_HPP

#include "common.hpp"
#include "fattn_common.hpp"

#include <sycl/sycl.hpp>

// Tile configuration (optimized for head sizes up to 128)
constexpr int FATTN_BQ            = 16;   // Queries per workgroup
constexpr int FATTN_BK            = 32;   // KV tile size
constexpr int FATTN_WG_M          = 16;
constexpr int FATTN_WG_N          = 1;    // Single-column workgroup for full parallelization
constexpr int FATTN_WG_SIZE       = FATTN_WG_M * FATTN_WG_N;
constexpr int FATTN_MAX_HEAD_SIZE = 128;  // Max DQK/DV supported by this kernel

// Compute shared memory size needed (in bytes)
inline size_t compute_shmem_size(int DQK, int DV) {
    size_t size = FATTN_BQ * DQK;  // shQ
    size += FATTN_BK * DQK;        // shK
    size += FATTN_BK * DV;         // shV
    return size * sizeof(float);
}

// Helper: convert any type to float
template <typename T> inline float load_as_float(const T * ptr);

template <> inline float load_as_float<float>(const float * ptr) {
    return *ptr;
}

template <> inline float load_as_float<sycl::half>(const sycl::half * ptr) {
    return static_cast<float>(*ptr);
}

#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <> inline float load_as_float<sycl::ext::oneapi::bfloat16>(const sycl::ext::oneapi::bfloat16 * ptr) {
    return static_cast<float>(*ptr);
}
#endif

// Single-kernel fused flash attention (template DQK, DV)
// QType: element type of Q (float, half, bfloat16)
// KVType: element type of K, V (float, half, bfloat16)
// MaskT: element type of mask (float, half)
template <typename QType, typename KVType, typename MaskT, int DQK, int DV>
inline void ggml_sycl_op_flash_attn_fused(sycl::queue *                stream,
                                          const QType *                Q,
                                          const KVType *               K,
                                          const KVType *               V,
                                          float *                      O,
                                          const int                    N,
                                          const int                    N_kv,
                                          const int                    n_heads,
                                          const int                    n_kv_heads,
                                          const int                    gqa_ratio,
                                          const float                  scale,
                                          const MaskT *                mask,
                                          const int64_t                mask_stride,
                                          const float *                sinks,
                                          const fattn_tensor_strides & strides) {
    (void)
        n_kv_heads;  // Suppress unused parameter warning (GQA ratio uses n_heads/n_kv_heads, but n_kv_heads itself not needed directly)
    constexpr int BQ   = FATTN_BQ;
    constexpr int BK   = FATTN_BK;
    constexpr int WG_M = FATTN_WG_M;
    constexpr int WG_N = FATTN_WG_N;

    const int      num_q_blocks = (N + BQ - 1) / BQ;
    sycl::range<2> global(num_q_blocks * WG_M, n_heads * WG_N);
    sycl::range<2> local(WG_M, WG_N);

    const size_t shmem_elems = compute_shmem_size(DQK, DV) / sizeof(float);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shmem_acc(sycl::range<1>(shmem_elems), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[intel::kernel_args_restrict]] {
            float *   shmem       = &shmem_acc[0];
            const int q_block_idx = it.get_group(0);
            const int head_idx    = it.get_group(1);
            const int lane_in_wg  = it.get_local_linear_id();
            // const int wg_col = it.get_local_id(1); // unused

            const int q_start = q_block_idx * BQ;
            if (q_start >= N) {
                return;
            }
            if (head_idx >= n_heads) {
                return;
            }

            const int kv_head_idx = head_idx / gqa_ratio;

            // Partition shared memory
            float * shQ = shmem;
            float * shK = shQ + BQ * DQK;
            float * shV = shK + BK * DQK;

            // Thread-local accumulators (Registers)
            // Sized from template parameters to force register promotion
            float acc[DV];
            float logits[FATTN_BK];  // Max size 32
            float m_curr = -1.0e20f;
            float l_curr = 0.0f;

// Initialize accumulators
#pragma unroll
            for (int i = 0; i < DV; ++i) {
                acc[i] = 0.0f;
            }

            // Load Q tile (persistent)
            for (int idx = lane_in_wg; idx < BQ * DQK; idx += FATTN_WG_SIZE) {
                int q_local = idx / DQK;
                int d       = idx % DQK;
                int q_idx   = q_start + q_local;
                if (q_idx < N && d < DQK) {
                    ptrdiff_t global_idx =
                        (ptrdiff_t) head_idx * strides.q_stride_head + (ptrdiff_t) q_idx * strides.q_stride_seq + d;
                    shQ[q_local * DQK + d] = load_as_float<QType>(&Q[global_idx]);
                } else {
                    shQ[q_local * DQK + d] = 0.0f;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);

            // Loop over KV blocks
            for (int kv_start = 0; kv_start < N_kv; kv_start += BK) {
                int split    = kv_start / BK;
                int kv_chunk = (kv_start + BK <= N_kv) ? BK : (N_kv - kv_start);

                // Load K tile
                for (int idx = lane_in_wg; idx < BK * DQK; idx += FATTN_WG_SIZE) {
                    int k_local = idx / DQK;
                    int d       = idx % DQK;
                    int kv_idx  = kv_start + k_local;
                    if (kv_idx < N_kv && d < DQK) {
                        ptrdiff_t global_idx = (ptrdiff_t) kv_head_idx * strides.k_stride_head +
                                               (ptrdiff_t) kv_idx * strides.k_stride_seq + d;
                        shK[k_local * DQK + d] = load_as_float<KVType>(&K[global_idx]);
                    } else {
                        shK[k_local * DQK + d] = 0.0f;
                    }
                }

                // Load V tile
                for (int idx = lane_in_wg; idx < BK * DV; idx += FATTN_WG_SIZE) {
                    int v_local = idx / DV;
                    int d       = idx % DV;
                    int kv_idx  = kv_start + v_local;
                    if (kv_idx < N_kv && d < DV) {
                        ptrdiff_t global_idx = (ptrdiff_t) kv_head_idx * strides.v_stride_head +
                                               (ptrdiff_t) kv_idx * strides.v_stride_seq + d;
                        shV[v_local * DV + d] = load_as_float<KVType>(&V[global_idx]);
                    } else {
                        shV[v_local * DV + d] = 0.0f;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // Each row is processed by exactly one thread (thread-local row index)
                // With WG_N=1, all threads participate - no serialization
                const int local_row = it.get_local_id(0);
                for (int q_local = local_row; q_local < BQ; q_local += WG_M) {
                    int q_idx = q_start + q_local;
                    if (q_idx >= N) {
                        continue;
                    }

                    // Compute logits Q @ K^T for this KV chunk
                    float m_split = -1.0e20f;
                    // #pragma unroll // Removed to reduce JIT time
                    for (int k = 0; k < BK; ++k) {
                        if (k >= kv_chunk) {
                            logits[k] = -1.0e20f;
                            continue;
                        }

                        float dot = 0.0f;
                        // #pragma unroll // Removed to reduce JIT time
                        for (int d = 0; d < DQK; ++d) {
                            dot += shQ[q_local * DQK + d] * shK[k * DQK + d];
                        }
                        float logit = scale * dot;

                        if (mask != nullptr) {
                            int global_k = kv_start + k;
                            if (global_k < N_kv) {
                                logit += load_as_float<MaskT>(&mask[q_idx * mask_stride + global_k]);
                            } else {
                                logit = -1.0e20f;  // Masked out
                            }
                        }
                        logits[k] = logit;
                        m_split   = sycl::fmax(m_split, logit);
                    }

                    // Sinks (first split only)
                    if (split == 0 && sinks != nullptr) {
                        float sink_val = sinks[head_idx];
                        m_split        = sycl::fmax(m_split, sink_val);
                    }

                    // Online softmax combine with previous splits
                    float m_prev     = m_curr;
                    float m_new      = sycl::fmax(m_prev, m_split);
                    float alpha_prev = sycl::exp(m_prev - m_new);

                    // Scale previous accumulators
                    // #pragma unroll // Removed to reduce JIT time
                    for (int d = 0; d < DV; ++d) {
                        acc[d] *= alpha_prev;
                    }

                    // Compute exp and accumulate PV
                    float l_split = 0.0f;
                    // #pragma unroll // Removed to reduce JIT time
                    for (int k = 0; k < BK; ++k) {
                        if (k >= kv_chunk)
                            continue;

                        float exp_val = sycl::exp(sycl::fmax(logits[k] - m_new, -20.0f));
                        // Reuse logits register
                        l_split += exp_val;
                        // #pragma unroll // Removed to reduce JIT time
                        for (int d = 0; d < DV; ++d) {
                            acc[d] += exp_val * shV[k * DV + d];
                        }
                    }

                    // Add sink contribution (first split)
                    if (split == 0 && sinks != nullptr) {
                        float sink_val = sinks[head_idx];
                        l_split += sycl::exp(sycl::fmax(sink_val - m_new, -20.0f));
                    }

                    // Update running stats
                    m_curr = m_new;
                    l_curr = l_curr * alpha_prev + l_split;
                }
                it.barrier(sycl::access::fence_space::local_space);
            }  // end splits

            // Final normalization and store
            const int local_row = it.get_local_id(0);
            for (int q_local = local_row; q_local < BQ; q_local += WG_M) {
                int q_idx = q_start + q_local;
                if (q_idx >= N) {
                    continue;
                }

                float inv_l = 1.0f / (l_curr > 1e-10f ? l_curr : 1.0f);
#pragma unroll
                for (int d = 0; d < DV; ++d) {
                    ptrdiff_t o_idx =
                        (ptrdiff_t) head_idx * strides.o_stride_head + (ptrdiff_t) q_idx * strides.o_stride_seq + d;
                    O[o_idx] = acc[d] * inv_l;
                }
            }
        });
    });
}

#endif  // GGML_SYCL_FATTN_FUSED_HPP
