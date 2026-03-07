// fattn-fused.hpp - Single-kernel fused Flash Attention for SYCL
// Subgroup-cooperative D-dimension parallelism with online softmax.
// Each subgroup handles one query row; threads within a subgroup split the
// head dimension (D) to reduce register pressure from ~160 to ~24 regs/thread.
// Graph-compatible, no oneMKL dependency.
//
// Optimizations:
// - Q loaded directly into registers (no SLM for Q, saves ROWS_PER_WG*DQK*4 bytes)
// - K/V stored as float in SLM for reliable alignment on all Intel architectures

#ifndef GGML_SYCL_FATTN_FUSED_HPP
#define GGML_SYCL_FATTN_FUSED_HPP

#include "common.hpp"
#include "fattn-common.hpp"

#include <sycl/sycl.hpp>

// Tile configuration
// BK: number of KV positions processed per tile
// ROWS_PER_WG: number of query rows per workgroup (= number of subgroups)
constexpr int FATTN_BK              = 32;
constexpr int FATTN_ROWS_PER_WG     = 32;
constexpr int FATTN_MAX_HEAD_SIZE   = 128;

// Compute shared memory size needed (in bytes)
// Only K and V in SLM; Q is in registers
inline size_t fattn_fused_shmem_size(int DQK, int DV, int rows_per_wg, int BK) {
    (void)rows_per_wg;
    size_t size = 0;
    size += BK * DQK;  // shK
    size += BK * DV;   // shV
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

// Stride parameters for direct loading from ggml tensor layout
struct fattn_tensor_strides {
    int64_t q_stride_seq;    // stride between Q rows (sequence positions)
    int64_t q_stride_head;   // stride between Q heads
    int64_t q_stride_batch;  // stride between Q batches
    int64_t k_stride_seq;    // stride between K rows
    int64_t k_stride_head;   // stride between K heads
    int64_t k_stride_batch;  // stride between K batches
    int64_t v_stride_seq;    // stride between V rows
    int64_t v_stride_head;   // stride between V heads
    int64_t v_stride_batch;  // stride between V batches
    int64_t o_stride_seq;    // stride between O rows (output)
    int64_t o_stride_head;   // stride between O heads
    int64_t o_stride_batch;  // stride between O batches
};

// Compute stride parameters for direct loading from ggml tensor layout
inline fattn_tensor_strides compute_tensor_strides(const ggml_tensor * Q, const ggml_tensor * K,
                                                    const ggml_tensor * V, const ggml_tensor * dst) {
    fattn_tensor_strides s;

    // Q strides in elements
    const size_t q_elem_size = ggml_type_size(Q->type);
    s.q_stride_seq = Q->nb[1] / q_elem_size;
    s.q_stride_head = Q->nb[2] / q_elem_size;
    s.q_stride_batch = Q->nb[3] / q_elem_size;

    // K strides in elements
    const size_t k_elem_size = ggml_type_size(K->type);
    s.k_stride_seq = K->nb[1] / k_elem_size;
    s.k_stride_head = K->nb[2] / k_elem_size;
    s.k_stride_batch = K->nb[3] / k_elem_size;

    // V strides in elements
    const size_t v_elem_size = ggml_type_size(V->type);
    s.v_stride_seq = V->nb[1] / v_elem_size;
    s.v_stride_head = V->nb[2] / v_elem_size;
    s.v_stride_batch = V->nb[3] / v_elem_size;

    // Output strides in elements (always F32)
    // Output layout is permuted: ne = [DV, n_heads, N, batch]
    // nb[1] = stride between heads (dim 1), nb[2] = stride between seq positions (dim 2)
    s.o_stride_head = dst->nb[1] / sizeof(float);
    s.o_stride_seq  = dst->nb[2] / sizeof(float);
    s.o_stride_batch = dst->nb[3] / sizeof(float);

    return s;
}

// Single-kernel fused flash attention with subgroup D-splitting.
template <typename QType, typename KVType, typename MaskT, int DQK, int DV,
          int SG_SIZE = 16, int ROWS_PER_WG = FATTN_ROWS_PER_WG, int BK = FATTN_BK>
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
                                          const int                    batch_size,
                                          const float                  scale,
                                          const MaskT *                mask,
                                          const int64_t                mask_stride_seq,
                                          const int64_t                mask_stride_head,
                                          const int64_t                mask_stride_batch,
                                          const float *                sinks,
                                          const fattn_tensor_strides & strides) {
    (void)n_kv_heads;

    static_assert(DQK % SG_SIZE == 0, "DQK must be a multiple of SG_SIZE");
    static_assert(DV  % SG_SIZE == 0, "DV must be a multiple of SG_SIZE");

    constexpr int WG_SIZE       = SG_SIZE * ROWS_PER_WG;
    constexpr int D_PER_THREAD  = DV / SG_SIZE;
    constexpr int DK_PER_THREAD = DQK / SG_SIZE;

    const int num_q_blocks = (N + ROWS_PER_WG - 1) / ROWS_PER_WG;

    // Work-group: dim0 = ROWS_PER_WG (rows/subgroups), dim1 = SG_SIZE (lanes)
    sycl::range<2> global(num_q_blocks * ROWS_PER_WG, n_heads * batch_size * SG_SIZE);
    sycl::range<2> local(ROWS_PER_WG, SG_SIZE);

    // SLM: only K and V tiles (Q is in registers)
    constexpr size_t shmem_floats = BK * DQK + BK * DV;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shmem_acc(sycl::range<1>(shmem_floats), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                float * shmem = &shmem_acc[0];

                const int row_in_wg   = it.get_local_id(0);   // 0..ROWS_PER_WG-1
                const int lane        = it.get_local_id(1);   // 0..SG_SIZE-1
                const int q_block_idx = it.get_group(0);
                const int head_batch_idx = it.get_group(1);
                
                const int head_idx    = head_batch_idx % n_heads;
                const int batch_idx   = head_batch_idx / n_heads;
                const int lid         = it.get_local_linear_id();

                const int q_start = q_block_idx * ROWS_PER_WG;

                // Entire workgroup out of bounds — safe uniform exit
                if (q_start >= N || head_batch_idx >= n_heads * batch_size) {
                    return;
                }

                const int  q_idx       = q_start + row_in_wg;
                const bool active_row  = (q_idx < N);
                const int  kv_head_idx = head_idx / gqa_ratio;

                // Partition shared memory: K and V only (no Q)
                float * shK = shmem;                // [BK][DQK]
                float * shV = shK + BK * DQK;      // [BK][DV]

                // ============================================================
                // Load Q directly into registers (each lane loads its D-slice)
                // Each lane needs DK_PER_THREAD elements, loaded once and
                // reused across all KV tiles. Saves ROWS_PER_WG*DQK*4 SLM.
                // ============================================================
                float regQ[DK_PER_THREAD];
                #pragma unroll
                for (int i = 0; i < DK_PER_THREAD; ++i) {
                    if (active_row) {
                        const int d = lane + i * SG_SIZE;
                        const ptrdiff_t gi = (ptrdiff_t)batch_idx * strides.q_stride_batch
                                           + (ptrdiff_t)head_idx * strides.q_stride_head
                                           + (ptrdiff_t)q_idx * strides.q_stride_seq + d;
                        regQ[i] = load_as_float<QType>(&Q[gi]);
                    } else {
                        regQ[i] = 0.0f;
                    }
                }

                // Thread-local accumulators (D-split)
                float acc[D_PER_THREAD];
                #pragma unroll
                for (int i = 0; i < D_PER_THREAD; ++i) {
                    acc[i] = 0.0f;
                }
                float m_curr = -1.0e20f;
                float l_curr = 0.0f;

                // ============================================================
                // Loop over KV blocks
                // ============================================================
                for (int kv_start = 0; kv_start < N_kv; kv_start += BK) {
                    const int kv_chunk = sycl::min(BK, N_kv - kv_start);

                    // Load K tile (cooperative)
                    for (int idx = lid; idx < BK * DQK; idx += WG_SIZE) {
                        const int k_local = idx / DQK;
                        const int d       = idx % DQK;
                        const int kv_idx  = kv_start + k_local;
                        if (kv_idx < N_kv) {
                            const ptrdiff_t gi = (ptrdiff_t)batch_idx * strides.k_stride_batch
                                               + (ptrdiff_t)kv_head_idx * strides.k_stride_head
                                               + (ptrdiff_t)kv_idx * strides.k_stride_seq + d;
                            shK[k_local * DQK + d] = load_as_float<KVType>(&K[gi]);
                        } else {
                            shK[k_local * DQK + d] = 0.0f;
                        }
                    }

                    // Load V tile (cooperative)
                    for (int idx = lid; idx < BK * DV; idx += WG_SIZE) {
                        const int v_local = idx / DV;
                        const int d       = idx % DV;
                        const int kv_idx  = kv_start + v_local;
                        if (kv_idx < N_kv) {
                            const ptrdiff_t gi = (ptrdiff_t)batch_idx * strides.v_stride_batch
                                               + (ptrdiff_t)kv_head_idx * strides.v_stride_head
                                               + (ptrdiff_t)kv_idx * strides.v_stride_seq + d;
                            shV[v_local * DV + d] = load_as_float<KVType>(&V[gi]);
                        } else {
                            shV[v_local * DV + d] = 0.0f;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ========================================================
                    // Per-row attention: each subgroup handles one query row
                    // Inactive rows skip math but still hit barriers
                    // ========================================================
                    if (active_row) {
                        const int split = kv_start / BK;

                        for (int k = 0; k < kv_chunk; ++k) {
                            // Q@K^T dot product: Q from registers, K from SLM
                            float partial_dot = 0.0f;
                            #pragma unroll
                            for (int di = 0; di < DK_PER_THREAD; ++di) {
                                const int d = lane + di * SG_SIZE;
                                partial_dot += regQ[di] * shK[k * DQK + d];
                            }
                            float dot = warp_reduce_sum<SG_SIZE>(partial_dot);

                            float logit = scale * dot;

                            if (mask != nullptr) {
                                const int global_k = kv_start + k;
                                if (global_k < N_kv) {
                                    const ptrdiff_t m_idx = (ptrdiff_t)batch_idx * mask_stride_batch 
                                                          + (ptrdiff_t)head_idx * mask_stride_head 
                                                          + (ptrdiff_t)q_idx * mask_stride_seq + global_k;
                                    logit += load_as_float<MaskT>(&mask[m_idx]);
                                } else {
                                    logit = -1.0e20f;
                                }
                            }

                            // Online softmax + P*V accumulation
                            float m_new = sycl::fmax(m_curr, logit);
                            float alpha_prev = sycl::exp(m_curr - m_new);
                            #pragma unroll
                            for (int i = 0; i < D_PER_THREAD; ++i) {
                                acc[i] *= alpha_prev;
                            }
                            l_curr *= alpha_prev;

                            float exp_val = sycl::exp(sycl::fmax(logit - m_new, -20.0f));
                            l_curr += exp_val;

                            #pragma unroll
                            for (int i = 0; i < D_PER_THREAD; ++i) {
                                const int d = lane + i * SG_SIZE;
                                acc[i] += exp_val * shV[k * DV + d];
                            }

                            m_curr = m_new;
                        }

                        // Handle attention sinks (first tile only)
                        if (split == 0 && sinks != nullptr) {
                            float sink_val   = sinks[head_idx];
                            float m_sink     = sycl::fmax(m_curr, sink_val);
                            float alpha_sink = sycl::exp(m_curr - m_sink);
                            #pragma unroll
                            for (int i = 0; i < D_PER_THREAD; ++i) {
                                acc[i] *= alpha_sink;
                            }
                            l_curr = l_curr * alpha_sink + sycl::exp(sycl::fmax(sink_val - m_sink, -20.0f));
                            m_curr = m_sink;
                        }
                    }

                    it.barrier(sycl::access::fence_space::local_space);
                }  // end KV blocks

                // ============================================================
                // Final normalization and store
                // Each lane writes its D-slice of the output
                // ============================================================
                if (active_row) {
                    const float inv_l = 1.0f / (l_curr > 1e-10f ? l_curr : 1.0f);
                    #pragma unroll
                    for (int i = 0; i < D_PER_THREAD; ++i) {
                        const int d = lane + i * SG_SIZE;
                        const ptrdiff_t o_idx = (ptrdiff_t)batch_idx * strides.o_stride_batch
                                              + (ptrdiff_t)head_idx * strides.o_stride_head
                                              + (ptrdiff_t)q_idx * strides.o_stride_seq + d;
                        O[o_idx] = acc[i] * inv_l;
                    }
                }
            });  // end parallel_for
    });  // end submit
}

// Host-side launch wrapper
template <int D, int type_K, int type_V>
void ggml_sycl_flash_attn_ext_fused_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];
    ggml_tensor * mask = dst->src[3];
    ggml_tensor * sinks = dst->src[4];

    float scale, max_bias, logit_softcap;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // Determine types
    using KVType = typename std::conditional<type_K == GGML_TYPE_F16, sycl::half, float>::type;

    const int N = Q->ne[1];
    const int N_kv = K->ne[1];
    const int n_heads = Q->ne[2];
    const int n_kv_heads = K->ne[2];
    const int batch_size = Q->ne[3];
    const int gqa_ratio = n_heads / n_kv_heads;

    fattn_tensor_strides strides = compute_tensor_strides(Q, K, V, dst);

    auto dispatch_mask_q = [&](auto t_q) {
        using QType = decltype(t_q);
        if (mask == nullptr || mask->type == GGML_TYPE_F32) {
            const float * mask_d = mask ? (const float *)mask->data : nullptr;
            const int64_t mask_stride_seq = mask ? mask->nb[1] / sizeof(float) : 0;
            const int64_t mask_stride_head = mask ? (mask->ne[2] > 1 ? mask->nb[2] / sizeof(float) : 0) : 0;
            const int64_t mask_stride_batch = mask ? (mask->ne[3] > 1 ? mask->nb[3] / sizeof(float) : 0) : 0;
            ggml_sycl_op_flash_attn_fused<QType, KVType, float, D, D>(
                ctx.stream(), (const QType *)Q->data, (const KVType *)K->data, (const KVType *)V->data,
                (float *)dst->data, N, N_kv, n_heads, n_kv_heads, gqa_ratio, batch_size, scale, mask_d, mask_stride_seq, mask_stride_head, mask_stride_batch,
                sinks ? (const float *)sinks->data : nullptr, strides);
        } else {
            const sycl::half * mask_d = (const sycl::half *)mask->data;
            const int64_t mask_stride_seq = mask->nb[1] / sizeof(sycl::half);
            const int64_t mask_stride_head = mask->ne[2] > 1 ? mask->nb[2] / sizeof(sycl::half) : 0;
            const int64_t mask_stride_batch = mask->ne[3] > 1 ? mask->nb[3] / sizeof(sycl::half) : 0;
            ggml_sycl_op_flash_attn_fused<QType, KVType, sycl::half, D, D>(
                ctx.stream(), (const QType *)Q->data, (const KVType *)K->data, (const KVType *)V->data,
                (float *)dst->data, N, N_kv, n_heads, n_kv_heads, gqa_ratio, batch_size, scale, mask_d, mask_stride_seq, mask_stride_head, mask_stride_batch,
                sinks ? (const float *)sinks->data : nullptr, strides);
        }
    };

    if (Q->type == GGML_TYPE_F16) {
        dispatch_mask_q(sycl::half{});
    } else {
        dispatch_mask_q(float{});
    }
}

// Extern declarations for explicit instantiations
#define EXTERN_DECL_FATTN_FUSED_CASE(D, type_K, type_V)                         \
    extern template void ggml_sycl_flash_attn_ext_fused_case                    \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

EXTERN_DECL_FATTN_FUSED_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE( 80, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE( 96, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE(112, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)

EXTERN_DECL_FATTN_FUSED_CASE( 64, GGML_TYPE_F32, GGML_TYPE_F32)
EXTERN_DECL_FATTN_FUSED_CASE( 80, GGML_TYPE_F32, GGML_TYPE_F32)
EXTERN_DECL_FATTN_FUSED_CASE( 96, GGML_TYPE_F32, GGML_TYPE_F32)
EXTERN_DECL_FATTN_FUSED_CASE(112, GGML_TYPE_F32, GGML_TYPE_F32)
EXTERN_DECL_FATTN_FUSED_CASE(128, GGML_TYPE_F32, GGML_TYPE_F32)
EXTERN_DECL_FATTN_FUSED_CASE(256, GGML_TYPE_F32, GGML_TYPE_F32)

#endif // GGML_SYCL_FATTN_FUSED_HPP
