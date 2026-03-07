// fattn-fused.hpp - Single-kernel fused Flash Attention for SYCL
// Subgroup-cooperative D-dimension parallelism with online softmax.
// Each subgroup handles one query row; threads within a subgroup split the
// head dimension (D) to reduce register pressure from ~160 to ~24 regs/thread.
//
// Optimizations over the base fattn-vec kernel:
// 1. Subgroup D-splitting: 16 lanes cooperate on D dimension (5-6.5x speedup)
// 2. 32 subgroups per WG share K/V tiles in SLM (17% speedup)
// 3. Q loaded directly into registers, not SLM (5% speedup)
//
// This kernel handles F16 and F32 Q/KV types. For quantized types,
// fall back to fattn-vec.hpp which has the necessary vec_dot functions.

#ifndef GGML_SYCL_FATTN_FUSED_HPP
#define GGML_SYCL_FATTN_FUSED_HPP

#include <climits>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"

namespace syclex = sycl::ext::oneapi::experimental;

// Tile configuration
constexpr int FATTN_FUSED_BK          = 32;   // KV positions per tile
constexpr int FATTN_FUSED_ROWS_PER_WG = 32;   // Query rows per workgroup (= subgroups per WG)

// Helper: convert any type to float
template <typename T> static __dpct_inline__ float load_as_float(const T * ptr);

template <> __dpct_inline__ float load_as_float<float>(const float * ptr) {
    return *ptr;
}

template <> __dpct_inline__ float load_as_float<sycl::half>(const sycl::half * ptr) {
    return static_cast<float>(*ptr);
}

// Single-kernel fused flash attention with subgroup D-splitting.
//
// Template parameters:
//   KVType: element type for K and V (float or sycl::half)
//   DQK: head dimension for Q/K (must be multiple of SG_SIZE)
//   DV:  head dimension for V   (must be multiple of SG_SIZE)
//   SG_SIZE: subgroup size (16 on Intel Xe)
//   ROWS_PER_WG: query rows per workgroup (= subgroups per workgroup)
//   BK: KV tile size
//
// Work-group layout: sycl::range<2>(ROWS_PER_WG, SG_SIZE)
//   dim 0 = which query row in this workgroup (slow-varying)
//   dim 1 = lane within subgroup (fast-varying → maps to subgroup lanes)
//
// Each thread owns:
//   regQ[DQK / SG_SIZE]  (Q row in registers, loaded once)
//   acc[DV / SG_SIZE]    (e.g. 128/16 = 8 floats)
//   m_curr, l_curr       (2 floats)
// Total: ~18 registers for DQK=DV=128
template <typename KVType, int DQK, int DV,
          int SG_SIZE = 16, int ROWS_PER_WG = FATTN_FUSED_ROWS_PER_WG, int BK = FATTN_FUSED_BK>
static void flash_attn_fused_kernel(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        sycl::float2 * __restrict__ dst_meta,
        const float    scale,
        const float    max_bias,
        const float    m0,
        const float    m1,
        const uint32_t n_head_log2,
        const float    logit_softcap,
        const int32_t  ne00,
        const sycl::uint3 ne01,
        const int32_t  ne02,
        const int32_t  ne03,
        const int32_t  nb01,
        const int32_t  nb02,
        const int32_t  nb03,
        const int32_t  ne10,
        const int32_t  ne11,
        const int32_t  ne12,
        const int32_t  ne13,
        const int32_t  nb11,
        const int32_t  nb12,
        const int64_t  nb13,
        const int32_t  nb21,
        const int32_t  nb22,
        const int64_t  nb23,
        const int32_t  ne31,
        const int32_t  ne32,
        const int32_t  ne33,
        const int32_t  nb31,
        const int32_t  nb32,
        const int64_t  nb33) {
#ifdef SYCL_FLASH_ATTN
    GGML_UNUSED(KV_max);
    GGML_UNUSED(dst_meta);
    GGML_UNUSED(max_bias);
    GGML_UNUSED(m0);
    GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2);
    GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne00);
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
    GGML_UNUSED(ne03);
    GGML_UNUSED(ne31);
    GGML_UNUSED(ne32);
    GGML_UNUSED(nb32);

    static_assert(DQK % SG_SIZE == 0, "DQK must be a multiple of SG_SIZE");
    static_assert(DV  % SG_SIZE == 0, "DV must be a multiple of SG_SIZE");

    constexpr int WG_SIZE       = SG_SIZE * ROWS_PER_WG;
    constexpr int D_PER_THREAD  = DV / SG_SIZE;
    constexpr int DK_PER_THREAD = DQK / SG_SIZE;

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    const int row_in_wg   = item_ct1.get_local_id(1);   // 0..ROWS_PER_WG-1
    const int lane        = item_ct1.get_local_id(2);    // 0..SG_SIZE-1
    const int lid         = row_in_wg * SG_SIZE + lane;  // linear local id

    // Grid mapping: group(0) = sequence*ne02+head, group(2) = q_block
    const int q_block_idx = item_ct1.get_group(2);
    const int head_seq    = item_ct1.get_group(0);
    const int sequence    = head_seq / ne02;
    const int head        = head_seq - sequence * ne02;
    const int gqa_ratio   = ne02 / ne12;
    const int kv_head     = head / gqa_ratio;

    const int N    = ne01.z();  // number of Q rows
    const int N_kv = ne11;      // number of KV rows

    const int q_start = q_block_idx * ROWS_PER_WG;

    // Entire workgroup out of bounds — safe uniform exit
    if (q_start >= N) {
        return;
    }

    const int  q_idx      = q_start + row_in_wg;
    const bool active_row = (q_idx < N);

    // Shared memory: K tile [BK][DQK] + V tile [BK][DV], stored as float
    constexpr int shmem_floats = BK * DQK + BK * DV;
    syclex::work_group_static<float[shmem_floats]> shmem;
    float * shK = &shmem[0];           // [BK][DQK]
    float * shV = shK + BK * DQK;     // [BK][DV]

    // ================================================================
    // Load Q directly into registers (each lane loads its D-slice)
    // Each lane needs DK_PER_THREAD elements, loaded once and
    // reused across all KV tiles. Saves ROWS_PER_WG*DQK*4 bytes SLM.
    // ================================================================
    const float * Q_f = (const float *)(Q + nb03 * sequence + nb02 * head);

    float regQ[DK_PER_THREAD];
#pragma unroll
    for (int i = 0; i < DK_PER_THREAD; ++i) {
        if (active_row) {
            const int d = lane + i * SG_SIZE;
            regQ[i] = Q_f[(ptrdiff_t)q_idx * (nb01 / sizeof(float)) + d] * scale;
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

    // ================================================================
    // KV base pointers
    // ================================================================
    const char * K_base = K + nb13 * sequence + nb12 * kv_head;
    const char * V_base = V + nb23 * sequence + nb22 * kv_head;

    const sycl::half * maskh = mask ?
        (const sycl::half *)(mask + nb33 * (sequence % ne33)) : nullptr;
    // Mask element stride per Q row (nb31 is in bytes)
    const int mask_row_stride = nb31 / sizeof(sycl::half);

    // ================================================================
    // Loop over KV blocks
    // ================================================================
    for (int kv_start = 0; kv_start < N_kv; kv_start += BK) {
        const int kv_chunk = sycl::min(BK, N_kv - kv_start);

        // Load K tile cooperatively (all WG_SIZE threads participate)
        for (int idx = lid; idx < BK * DQK; idx += WG_SIZE) {
            const int k_local = idx / DQK;
            const int d       = idx % DQK;
            const int kv_idx  = kv_start + k_local;
            if (kv_idx < N_kv) {
                shK[k_local * DQK + d] = load_as_float<KVType>(
                    (const KVType *)(K_base + (ptrdiff_t)kv_idx * nb11) + d);
            } else {
                shK[k_local * DQK + d] = 0.0f;
            }
        }

        // Load V tile cooperatively
        for (int idx = lid; idx < BK * DV; idx += WG_SIZE) {
            const int v_local = idx / DV;
            const int d       = idx % DV;
            const int kv_idx  = kv_start + v_local;
            if (kv_idx < N_kv) {
                shV[v_local * DV + d] = load_as_float<KVType>(
                    (const KVType *)(V_base + (ptrdiff_t)kv_idx * nb21) + d);
            } else {
                shV[v_local * DV + d] = 0.0f;
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // ============================================================
        // Per-row attention: each subgroup handles one query row
        // Inactive rows skip math but still hit barriers
        // ============================================================
        if (active_row) {
            for (int k = 0; k < kv_chunk; ++k) {
                // Q@K^T dot product: Q from registers, K from SLM
                // Each lane computes partial dot over its D/SG_SIZE slice
                float partial_dot = 0.0f;
#pragma unroll
                for (int di = 0; di < DK_PER_THREAD; ++di) {
                    const int d = lane + di * SG_SIZE;
                    partial_dot += regQ[di] * shK[k * DQK + d];
                }
                // Reduce across subgroup lanes
                float dot = warp_reduce_sum<SG_SIZE>(partial_dot);

                float logit = dot;  // scale already applied to Q

                if (maskh != nullptr) {
                    const int global_k = kv_start + k;
                    if (global_k < N_kv) {
                        logit += static_cast<float>(maskh[q_idx * mask_row_stride + global_k]);
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
            if (kv_start == 0 && sinks != nullptr) {
                float sink_val   = ((const float *)sinks)[head];
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

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }  // end KV blocks

    // ============================================================
    // Final normalization and store
    // Each lane writes its D-slice of the output
    // ============================================================
    if (active_row) {
        const float inv_l = 1.0f / (l_curr > 1e-10f ? l_curr : 1.0f);

        // Output layout: dst[sequence][q_idx][head][d]
        // From the launch_fattn infrastructure:
        //   dst[(sequence * N + ic0 + j) * ne02 + head) * parallel_blocks + block_y) * D + d]
        // With parallel_blocks=1 (single pass), block_y=0:
        //   dst[((sequence * N + q_idx) * ne02 + head) * D + d]
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            const int d = lane + i * SG_SIZE;
            const ptrdiff_t o_idx = ((ptrdiff_t)(sequence * N + q_idx) * ne02 + head) * DV + d;
            dst[o_idx] = acc[i] * inv_l;
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#endif // SYCL_FLASH_ATTN
}

// Host-side launch wrapper
template <int D, int type_K, int type_V>
void ggml_sycl_flash_attn_ext_fused_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    constexpr int SG_SIZE      = WARP_16_SIZE;
    constexpr int ROWS_PER_WG  = FATTN_FUSED_ROWS_PER_WG;
    constexpr int BK           = FATTN_FUSED_BK;

    // Determine KV element type
    using KVType = typename std::conditional<type_K == GGML_TYPE_F16, sycl::half, float>::type;

    const bool need_f16_K = (type_K == GGML_TYPE_F16);
    const bool need_f16_V = (type_V == GGML_TYPE_F16);

    constexpr int nwarps = ROWS_PER_WG;  // Each "warp" is one subgroup = one query row
    constexpr size_t nbytes_shared = 0;  // Using work_group_static, not dynamic SLM

    // Use launch_fattn with ncols1=ROWS_PER_WG so ntiles_x = ceil(N/32),
    // giving each WG 32 Q rows (one per subgroup).
    // Set nbatch_fa to INT_MAX so ntiles_KQ=1 and parallel_blocks=1,
    // because the fused kernel processes all KV positions in a single pass
    // with online softmax (no split reduction needed).
    launch_fattn<D, ROWS_PER_WG, 1,
                 flash_attn_fused_kernel<KVType, D, D, SG_SIZE, ROWS_PER_WG, BK>,
                 SG_SIZE>(
        ctx, dst, nwarps, nbytes_shared, INT_MAX, need_f16_K, need_f16_V, false);
}

// Extern declarations for explicit instantiations
#define EXTERN_DECL_FATTN_FUSED_CASE(D, type_K, type_V)                         \
    extern template void ggml_sycl_flash_attn_ext_fused_case                    \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Only F16 and F32 types for the fused kernel (quantized uses fattn-vec)
EXTERN_DECL_FATTN_FUSED_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE( 80, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE( 96, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE(112, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
EXTERN_DECL_FATTN_FUSED_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)

#endif // GGML_SYCL_FATTN_FUSED_HPP
