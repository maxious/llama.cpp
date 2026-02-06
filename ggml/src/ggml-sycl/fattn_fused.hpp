// fattn_fused.hpp - Single-kernel fused Flash Attention for SYCL
// Online softmax, direct stride loading, graph-compatible
// Version: runtime DQK/DV (no template on dimensions)

#ifndef GGML_SYCL_FATTN_FUSED_HPP
#define GGML_SYCL_FATTN_FUSED_HPP

#include "fattn_common.hpp"
#include "common.hpp"
#include <sycl/sycl.hpp>

// Tile configuration (optimized for head sizes up to 128)
constexpr int FATTN_BQ = 16;  // Queries per workgroup
constexpr int FATTN_BK = 32;  // KV tile size
constexpr int FATTN_WG_M = 16;
constexpr int FATTN_WG_N = 16;
constexpr int FATTN_WG_SIZE = FATTN_WG_M * FATTN_WG_N;

// Compute shared memory size needed (in bytes)
inline size_t compute_shmem_size(int DQK, int DV) {
    size_t size = FATTN_BQ * DQK;         // shQ
    size += FATTN_BK * DQK;               // shK
    size += FATTN_BK * DV;                // shV
    size += 2 * FATTN_BQ;                 // row_max + row_sum
    size += FATTN_BQ * DV;                // sh_acc
    size += FATTN_BQ * FATTN_BK;          // shS (logits/scores - separate from shK!)
    return size * sizeof(float);
}

// Helper: convert any type to float
template <typename T>
inline float load_as_float(const T* ptr);

template <>
inline float load_as_float<float>(const float* ptr) { return *ptr; }

template <>
inline float load_as_float<sycl::half>(const sycl::half* ptr) { return static_cast<float>(*ptr); }

#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline float load_as_float<sycl::ext::oneapi::bfloat16>(const sycl::ext::oneapi::bfloat16* ptr) {
    return static_cast<float>(*ptr);
}
#endif

// Single-kernel fused flash attention (runtime DQK, DV)
// T: element type of Q, K, V (float, half, bfloat16)
template <typename T>
inline void ggml_sycl_op_flash_attn_fused(
    sycl::queue* stream,
    const T* Q, const T* K, const T* V,
    float* O,
    const int N, const int N_kv,
    const int n_heads, const int n_kv_heads, const int gqa_ratio,
    const float scale,
    const float* mask, const int64_t mask_stride,
    const float* sinks,
    const fattn_tensor_strides& strides,
    int DQK, int DV
) {
    constexpr int BQ = FATTN_BQ;
    constexpr int BK = FATTN_BK;
    constexpr int WG_M = FATTN_WG_M;
    constexpr int WG_N = FATTN_WG_N;

    const int num_q_blocks = (N + BQ - 1) / BQ;
    sycl::range<2> global(num_q_blocks * WG_M, n_heads * WG_N);
    sycl::range<2> local(WG_M, WG_N);

    const size_t shmem_elems = compute_shmem_size(DQK, DV) / sizeof(float);

    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> shmem_acc(sycl::range<1>(shmem_elems), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[intel::kernel_args_restrict]] {
                float* shmem = &shmem_acc[0];
                const int q_block_idx = it.get_group(0);
                const int head_idx = it.get_group(1);
                const int local_id = it.get_local_id(0);
                const int lane_in_wg = it.get_local_linear_id();

                const int q_start = q_block_idx * BQ;
                if (q_start >= N) return;
                if (head_idx >= n_heads) return;

                const int kv_head_idx = head_idx / gqa_ratio;

                // Partition shared memory
                float* shQ = shmem;
                float* shK = shQ + BQ * DQK;
                float* shV = shK + BK * DQK;
                float* row_max = shV + BK * DV;
                float* row_sum = row_max + BQ;
                float* sh_acc = row_sum + BQ;
                float* shS = sh_acc + BQ * DV;  // Separate buffer for logits/scores

                // Initialize per-thread accumulators
                const int rows_per_thread = (BQ + WG_M - 1) / WG_M;
                const int my_row_start = local_id;
                for (int r = 0; r < rows_per_thread; ++r) {
                    int q_local = my_row_start + r * WG_M;
                    if (q_local < BQ) {
                        int q_idx = q_start + q_local;
                        if (q_idx < N) {
                            row_max[q_local] = -1.0e20f;
                            row_sum[q_local] = 0.0f;
                            for (int d = 0; d < DV; ++d) {
                                sh_acc[q_local * DV + d] = 0.0f;
                            }
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // Load Q tile (persistent)
                for (int idx = lane_in_wg; idx < BQ * DQK; idx += FATTN_WG_SIZE) {
                    int q_local = idx / DQK;
                    int d = idx % DQK;
                    int q_idx = q_start + q_local;
                    if (q_idx < N && d < DQK) {
                        ptrdiff_t global_idx = (ptrdiff_t)head_idx * strides.q_stride_head + 
                                               (ptrdiff_t)q_idx * strides.q_stride_seq + d;
                        shQ[q_local * DQK + d] = load_as_float<T>(&Q[global_idx]);
                    } else {
                        shQ[q_local * DQK + d] = 0.0f;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // Iterate over KV splits
                const int num_splits = (N_kv + BK - 1) / BK;
                for (int split = 0; split < num_splits; ++split) {
                    const int kv_start = split * BK;
                    const int kv_end = sycl::min(kv_start + BK, N_kv);
                    const int kv_chunk = kv_end - kv_start;
                    if (kv_chunk <= 0) break;

                    // Load K tile
                    for (int idx = lane_in_wg; idx < kv_chunk * DQK; idx += FATTN_WG_SIZE) {
                        int k_local = idx / DQK;
                        int d = idx % DQK;
                        int kv_idx = kv_start + k_local;
                        if (kv_idx < N_kv && d < DQK) {
                            ptrdiff_t global_idx = (ptrdiff_t)kv_head_idx * strides.k_stride_head + 
                                                   (ptrdiff_t)kv_idx * strides.k_stride_seq + d;
                            shK[k_local * DQK + d] = load_as_float<T>(&K[global_idx]);
                        } else {
                            shK[k_local * DQK + d] = 0.0f;
                        }
                    }

                    // Load V tile
                    for (int idx = lane_in_wg; idx < kv_chunk * DV; idx += FATTN_WG_SIZE) {
                        int v_local = idx / DV;
                        int d = idx % DV;
                        int kv_idx = kv_start + v_local;
                        if (kv_idx < N_kv && d < DV) {
                            ptrdiff_t global_idx = (ptrdiff_t)kv_head_idx * strides.v_stride_head + 
                                                   (ptrdiff_t)kv_idx * strides.v_stride_seq + d;
                            shV[v_local * DV + d] = load_as_float<T>(&V[global_idx]);
                        } else {
                            shV[v_local * DV + d] = 0.0f;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // Process each query row assigned to this thread
                    for (int r = 0; r < rows_per_thread; ++r) {
                        int q_local = my_row_start + r * WG_M;
                        if (q_local >= BQ) continue;
                        int q_idx = q_start + q_local;
                        if (q_idx >= N) continue;

                        // Compute logits Q @ K^T for this KV chunk
                        float m_split = -1.0e20f;
                        for (int k = 0; k < kv_chunk; ++k) {
                            float dot = 0.0f;
                            for (int d = 0; d < DQK; ++d) {
                                dot += shQ[q_local * DQK + d] * shK[k * DQK + d];
                            }
                            float logit = scale * dot;

                            if (mask != nullptr) {
                                int global_k = kv_start + k;
                                if (global_k < N_kv) {
                                    logit += mask[q_idx * mask_stride + global_k];
                                }
                            }
                            shS[q_local * BK + k] = logit;
                            m_split = sycl::fmax(m_split, logit);
                        }

                        // Sinks (first split only)
                        if (split == 0 && sinks != nullptr) {
                            float sink_val = sinks[head_idx];
                            m_split = sycl::fmax(m_split, sink_val);
                        }

                        // Online softmax combine with previous splits
                        float m_prev = row_max[q_local];
                        float m_new = sycl::fmax(m_prev, m_split);
                        float alpha_prev = sycl::exp(m_prev - m_new);

                        // Scale previous accumulators
                        for (int d = 0; d < DV; ++d) {
                            sh_acc[q_local * DV + d] *= alpha_prev;
                        }

                        // Compute exp and accumulate PV
                        float l_split = 0.0f;
                        for (int k = 0; k < kv_chunk; ++k) {
                            float exp_val = sycl::exp(sycl::fmax(shS[q_local * BK + k] - m_new, -20.0f));
                            shS[q_local * BK + k] = exp_val;
                            l_split += exp_val;
                            for (int d = 0; d < DV; ++d) {
                                sh_acc[q_local * DV + d] += exp_val * shV[k * DV + d];
                            }
                        }

                        // Add sink contribution (first split)
                        if (split == 0 && sinks != nullptr) {
                            float sink_val = sinks[head_idx];
                            l_split += sycl::exp(sycl::fmax(sink_val - m_new, -20.0f));
                        }

                        // Update running stats
                        row_max[q_local] = m_new;
                        row_sum[q_local] = row_sum[q_local] * alpha_prev + l_split;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }  // end splits

                // Final normalization and store
                it.barrier(sycl::access::fence_space::local_space);
                for (int r = 0; r < rows_per_thread; ++r) {
                    int q_local = my_row_start + r * WG_M;
                    if (q_local >= BQ) continue;
                    int q_idx = q_start + q_local;
                    if (q_idx >= N) continue;

                    float inv_l = 1.0f / (row_sum[q_local] > 1e-10f ? row_sum[q_local] : 1.0f);
                    for (int d = 0; d < DV; ++d) {
                        ptrdiff_t o_idx = (ptrdiff_t)head_idx * strides.o_stride_head + 
                                          (ptrdiff_t)q_idx * strides.o_stride_seq + d;
                        O[o_idx] = sh_acc[q_local * DV + d] * inv_l;
                    }
                }
            });
    });
}

#endif // GGML_SYCL_FATTN_FUSED_HPP
