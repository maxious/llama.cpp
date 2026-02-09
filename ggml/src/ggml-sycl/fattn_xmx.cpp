// fattn_xmx.cpp - XMX Flash Attention Implementation
// Split from fattn.cpp refactoring

#include "common.hpp"
#include "fattn.hpp"
#include "fattn_common.hpp"

#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_MATRIX
#    include <sycl/ext/oneapi/bfloat16.hpp>
#    include <sycl/ext/oneapi/group_local_memory.hpp>
#    include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#endif

#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif

#ifndef GGML_SYCL_FATTN_XMX_HPP
#    define GGML_SYCL_FATTN_XMX_HPP

#    include "common.hpp"
#    include "fattn_common.hpp"

#    include <sycl/sycl.hpp>

#    ifdef SYCL_EXT_ONEAPI_MATRIX
#        include <sycl/ext/oneapi/bfloat16.hpp>
#        include <sycl/ext/oneapi/group_local_memory.hpp>
#        include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#    endif

#    if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#        include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#    endif

#    ifdef SYCL_EXT_ONEAPI_MATRIX

// ============================================================================
// XMX Flash Attention Kernels
// ============================================================================

// Type alias for bfloat16 used in XMX operations
using xmx_bfloat16 = sycl::ext::oneapi::bfloat16;

// ============================================================================
// flash_attn_coopmat_kernel_padded - Base template for 32x32 tiles
// ============================================================================
// - V_FROM_K: if true, V is extracted from K's first V_HEAD_DIM columns (MLA zero-copy)
// - TM, TN, TK: joint matrix tile sizes
template <int64_t HEAD_DIM,
          int64_t V_HEAD_DIM,
          int64_t PADDED_HEAD_DIM,
          int64_t PADDED_V_HEAD_DIM,
          bool    V_FROM_K = false,
          int     TM       = 8,
          int     TN       = 16,
          int     TK       = 16>
inline void flash_attn_coopmat_kernel_padded(
    sycl::nd_item<2> it,
    const float *    Q,
    const float *    K,
    const float *    V,  // When V_FROM_K=true, this is ignored and V is read from K
    float *          O,
    float *          l_d,
    float *          m_d,
    const int64_t    N,
    const int64_t    N_kv,
    const int        n_heads,
    const int        n_kv_heads,
    const int        gqa_ratio,
    const float      scale,
    const float *    mask,
    const int64_t    mask_stride,
    const int        o_row_stride,
    float *          shmem) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int THREADS = 64;

    // Number of subgroups covering the M dimension
    constexpr int NUM_SG_M = BLOCK_M / TM;

    // Use PADDED_HEAD_DIM for Q/K XMX computations, PADDED_V_HEAD_DIM for V/O XMX computations
    constexpr int Q_STRIDE   = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE   = PADDED_HEAD_DIM + 8;
    constexpr int V_STRIDE   = PADDED_V_HEAD_DIM + 8;
    // V is stored col-major in shVT as [BLOCK_N rows, PADDED_V_HEAD_DIM cols] for P@V
    // For col-major, stride (leading dimension) must be >= number of rows = BLOCK_N
    constexpr int V_T_STRIDE = BLOCK_N + 8;  // Must be >= BLOCK_N for col-major storage
    constexpr int S_STRIDE   = BLOCK_N + 8;
    constexpr int P_STRIDE   = BLOCK_N + 8;

    // Ensure BLOCK_N is divisible by TK for P@V computation
    static_assert(BLOCK_N % TK == 0, "BLOCK_N must be divisible by TK for P@V matmul");
    static_assert(PADDED_HEAD_DIM % TN == 0, "PADDED_HEAD_DIM must be divisible by TN for output tiles");
    static_assert(PADDED_V_HEAD_DIM % TN == 0, "PADDED_V_HEAD_DIM must be divisible by TN for output tiles");

    // Shared memory layout (sized for PADDED_HEAD_DIM)
    // shVT is [BLOCK_N rows x PADDED_HEAD_DIM cols] col-major, size = PADDED_HEAD_DIM * V_T_STRIDE

    (void) l_d;         // suppress unused parameter warning
    (void) m_d;         // suppress unused parameter warning
    (void) n_heads;     // suppress unused parameter warning
    (void) n_kv_heads;  // suppress unused parameter warning

    const int lid   = it.get_local_id(0);
    const int gid_x = it.get_group(0);
    const int gid_y = it.get_group(1);
    auto      sg    = it.get_sub_group();

    const int sg_id = sg.get_group_linear_id();

    if (sg_id >= NUM_SG_M) {
        return;
    }

    const int row0 = gid_x * BLOCK_M;
    if (row0 >= N) {
        return;
    }

    const int head_idx    = gid_y;
    const int kv_head_idx = head_idx / gqa_ratio;

    // Use HEAD_DIM for Q/K actual data access, V_HEAD_DIM for V/O actual data access
    // Use PADDED_HEAD_DIM for Q/K XMX operations, PADDED_V_HEAD_DIM for V/O XMX operations
    const ptrdiff_t q_offset = (ptrdiff_t) (head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t) (kv_head_idx * N_kv) * HEAD_DIM;  // K uses N_kv and HEAD_DIM
    // When V_FROM_K is true, V is read from K's memory (first V_HEAD_DIM columns per row)
    const ptrdiff_t v_offset = V_FROM_K ? k_offset : (ptrdiff_t) (kv_head_idx * N_kv) * V_HEAD_DIM;
    const ptrdiff_t o_offset = (ptrdiff_t) (head_idx * N + row0) * V_HEAD_DIM;  // O uses V_HEAD_DIM

    // For V_FROM_K mode, V data is interleaved in K's layout with HEAD_DIM stride
    constexpr ptrdiff_t V_ROW_STRIDE = V_FROM_K ? HEAD_DIM : V_HEAD_DIM;

    // Bfloat16 shared memory for XMX tiles
    xmx_bfloat16 * shQ  = reinterpret_cast<xmx_bfloat16 *>(shmem);
    xmx_bfloat16 * shK  = shQ + BLOCK_M * Q_STRIDE;
    xmx_bfloat16 * shV  = shK + BLOCK_N * K_STRIDE;
    xmx_bfloat16 * shP  = shV + BLOCK_N * V_STRIDE;
    xmx_bfloat16 * shVT = shP + BLOCK_M * P_STRIDE;  // V^T buffer (separate from shK to avoid corruption)

    // Float shared memory for softmax scores and output accumulator
    // shVT is [PADDED_V_HEAD_DIM x BLOCK_N] col-major, so advance by PADDED_V_HEAD_DIM * V_T_STRIDE
    float * shS      = reinterpret_cast<float *>(shVT + PADDED_V_HEAD_DIM * V_T_STRIDE);
    float * rowMax   = shS + BLOCK_M * S_STRIDE;
    float * rowSum   = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;
    float * shAcc    = rowAlpha + BLOCK_M;  // Sized for PADDED_V_HEAD_DIM

    // Initialize per-row stats and output accumulator
    if (lid < BLOCK_M) {
        rowMax[lid] = -1.0e20f;
        rowSum[lid] = 0.0f;
    }
    for (int i = lid; i < BLOCK_M * PADDED_V_HEAD_DIM; i += THREADS) {
        shAcc[i] = 0.0f;
    }

    // Load Q tiles with xmx_bfloat16 conversion, scale, and padding
    for (int i = lid; i < BLOCK_M * PADDED_HEAD_DIM; i += THREADS) {
        const int r     = i / PADDED_HEAD_DIM;
        const int c     = i % PADDED_HEAD_DIM;
        const int q_row = row0 + r;

        if (q_row < N && c < HEAD_DIM) {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(Q[q_offset + (ptrdiff_t) r * HEAD_DIM + c] * scale);
        } else {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(0.0f);  // Zero for padding
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    const int num_kv_blocks = (N_kv + BLOCK_N - 1) / BLOCK_N;  // Use N_kv for KV cache

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int col0 = kv_block * BLOCK_N;

        // Load K tiles with xmx_bfloat16 conversion and padding
        for (int i = lid; i < BLOCK_N * PADDED_HEAD_DIM; i += THREADS) {
            const int r     = i / PADDED_HEAD_DIM;
            const int c     = i % PADDED_HEAD_DIM;
            const int k_row = col0 + r;

            xmx_bfloat16 k_val = xmx_bfloat16(0.0f);

            if (k_row < N_kv && c < HEAD_DIM) {  // Use N_kv for bounds check
                const ptrdiff_t base = (ptrdiff_t) k_row * HEAD_DIM;
                k_val                = xmx_bfloat16(K[k_offset + base + c]);
            }

            // Store K as K^T col-major: K^T[head_dim, kv_col] at shK[head_dim + kv_col * K_STRIDE]
            shK[c + r * K_STRIDE] = k_val;
        }

        // Load V tiles separately with V_HEAD_DIM and PADDED_V_HEAD_DIM
        // When V_FROM_K is true, read V from K's memory with HEAD_DIM stride (zero-copy MLA)
        for (int i = lid; i < BLOCK_N * PADDED_V_HEAD_DIM; i += THREADS) {
            const int r     = i / PADDED_V_HEAD_DIM;
            const int c     = i % PADDED_V_HEAD_DIM;
            const int k_row = col0 + r;

            xmx_bfloat16 v_val = xmx_bfloat16(0.0f);

            if (k_row < N_kv && c < V_HEAD_DIM) {  // Use N_kv for bounds check
                // V_ROW_STRIDE is HEAD_DIM when V_FROM_K, else V_HEAD_DIM
                const ptrdiff_t base = (ptrdiff_t) k_row * V_ROW_STRIDE;
                // When V_FROM_K, read from K's base pointer with K's stride
                v_val = V_FROM_K ? xmx_bfloat16(K[v_offset + base + c]) : xmx_bfloat16(V[v_offset + base + c]);
            }

            // Store V row-major
            shV[r * V_STRIDE + c] = v_val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Q @ K^T computation ============
        for (int j = 0; j < BLOCK_N; j += TN) {
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN, layout::dynamic> matS;
            joint_matrix_fill(sg, matS, 0.0f);

            for (int k = 0; k < PADDED_HEAD_DIM; k += TK) {
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, TM, TK, layout::row_major> mq;
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, TK, TN, layout::col_major> mk;

                auto mq_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shQ[(sg_id * TM) * Q_STRIDE + k]);
                // K^T is stored col-major, tile [k:k+TK, j:j+TN] starts at shK[k + j * K_STRIDE]
                auto mk_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shK[k + j * K_STRIDE]);

                joint_matrix_load(sg, mq, mq_ptr, Q_STRIDE);
                joint_matrix_load(sg, mk, mk_ptr, K_STRIDE);

                joint_matrix_mad(sg, matS, mq, mk, matS);
            }

            auto shS_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                    &shS[(sg_id * TM) * S_STRIDE + j]);
            joint_matrix_store(sg, matS, shS_ptr, S_STRIDE, layout::row_major);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Online Softmax ============
        if (lid < BLOCK_M) {
            const int row   = lid;
            const int q_row = row0 + row;  // Global query row index
            float     m     = -1.0e20f;

            for (int c = 0; c < BLOCK_N; ++c) {
                float     s_val  = shS[row * S_STRIDE + c];
                const int kv_col = col0 + c;  // Global KV column index

                // Apply mask from precomputed mask tensor
                if (mask != nullptr && kv_col < N_kv && q_row < N) {
                    const float mask_val = mask[q_row * mask_stride + kv_col];
                    s_val += mask_val;  // mask is 0.0 for unmasked, -inf for masked
                }

                shS[row * S_STRIDE + c] = s_val;
                m                       = sycl::fmax(m, s_val);
            }

            float m_prev  = rowMax[row];
            float m_new   = sycl::fmax(m_prev, m);
            float alpha   = sycl::exp(sycl::fmax(m_prev - m_new, -20.0f));
            rowAlpha[row] = alpha;
            rowMax[row]   = m_new;

            float s_sum = 0.0f;
            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val             = shS[row * S_STRIDE + c];
                float val               = sycl::exp(sycl::fmax(s_val - m_new, -20.0f));
                shS[row * S_STRIDE + c] = val;
                s_sum += val;
            }
            rowSum[row] = rowSum[row] * alpha + s_sum;
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Scale previous output by alpha (critical for online softmax correctness) ============
        // The online softmax scales previous probabilities by exp(m_old - m_new) = rowAlpha
        // The P@V accumulator must be scaled by the same factor to maintain correctness
        for (int i = lid; i < BLOCK_M * PADDED_V_HEAD_DIM; i += THREADS) {
            const int row = i / PADDED_V_HEAD_DIM;
            shAcc[i] *= rowAlpha[row];
        }

        // ============ Convert P to xmx_bfloat16 for XMX ============
        for (int i = lid; i < BLOCK_M * BLOCK_N; i += THREADS) {
            const int row             = i / BLOCK_N;
            const int col             = i % BLOCK_N;
            shP[row * P_STRIDE + col] = xmx_bfloat16(shS[row * S_STRIDE + col]);
        }

        // ============ Copy V to shVT for P@V computation ============
        // V is stored as [BLOCK_N x PADDED_V_HEAD_DIM] row-major in shV
        // Store V col-major in shVT as [BLOCK_N rows, PADDED_V_HEAD_DIM cols]
        // V[row, col] stored at shVT[row + col * V_T_STRIDE]
        for (int i = lid; i < BLOCK_N * PADDED_V_HEAD_DIM; i += THREADS) {
            const int row = i / PADDED_V_HEAD_DIM;  // kv position
            const int col = i % PADDED_V_HEAD_DIM;  // head dim
            // Only copy actual V_HEAD_DIM data, padding elements are zero
            if (col < V_HEAD_DIM) {
                shVT[row + col * V_T_STRIDE] = shV[row * V_STRIDE + col];
            } else {
                shVT[row + col * V_T_STRIDE] = xmx_bfloat16(0.0f);  // Zero for padding
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ P @ V computation ============
        // P is [BLOCK_M x BLOCK_N], V is [BLOCK_N x PADDED_V_HEAD_DIM]
        // Output is [BLOCK_M x PADDED_V_HEAD_DIM], computed as tiles of [TM x TN]
        constexpr int NUM_OUT_TILES = PADDED_V_HEAD_DIM / TN;

        for (int out_tile = 0; out_tile < NUM_OUT_TILES; ++out_tile) {
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN, layout::dynamic> matPV;
            joint_matrix_fill(sg, matPV, 0.0f);

            // Sum over k dimension (BLOCK_N sequence positions) in chunks of TK
            for (int k = 0; k < BLOCK_N; k += TK) {
                // P tile: TM x TK (rows from sg_id * TM, cols from k)
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, TM, TK, layout::row_major> mp;
                auto                                                                           shP_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shP[(sg_id * TM) * P_STRIDE + k]);
                joint_matrix_load(sg, mp, shP_ptr, P_STRIDE);

                // V tile: TK x TN from shVT (col-major, V[BLOCK_N rows, PADDED_V_HEAD_DIM cols])
                // V[k:k+TK, out_tile*TN:out_tile*TN+TN] at shVT[k + out_tile*TN * V_T_STRIDE]
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, TK, TN, layout::col_major> mv;
                auto                                                                           shV_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shVT[k + out_tile * TN * V_T_STRIDE]);
                joint_matrix_load(sg, mv, shV_ptr, V_T_STRIDE);

                joint_matrix_mad(sg, matPV, mp, mv, matPV);
            }

            // Add PV contribution to output accumulator using scratch region
            const int scratch_offset = sg_id * TM * TN;
            auto      scratch_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                    &shS[scratch_offset]);
            joint_matrix_store(sg, matPV, scratch_ptr, TN, layout::row_major);
            it.barrier(sycl::access::fence_space::local_space);

            // Add to output accumulator - output column is in head-dim, not sequence
            const int sg_lane = sg.get_local_linear_id();
            const int sg_size = sg.get_local_linear_range();
            for (int idx = sg_lane; idx < TM * TN; idx += sg_size) {
                const int local_row  = idx / TN;
                const int local_col  = idx % TN;
                const int global_row = sg_id * TM + local_row;
                const int global_col = out_tile * TN + local_col;
                if (global_row < BLOCK_M && global_col < PADDED_V_HEAD_DIM) {
                    shAcc[global_row * PADDED_V_HEAD_DIM + global_col] += shS[scratch_offset + idx];
                }
            }

            it.barrier(sycl::access::fence_space::local_space);
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    // ============ Final normalize and store ============
    // Only write actual V_HEAD_DIM elements, skip padding
    for (int i = lid; i < BLOCK_M * V_HEAD_DIM; i += THREADS) {
        const int row   = i / V_HEAD_DIM;
        const int col   = i % V_HEAD_DIM;
        const int q_row = row0 + row;

        if (q_row < N) {
            float s                                            = rowSum[row];
            // shAcc is indexed with PADDED_V_HEAD_DIM stride
            float acc_val                                      = shAcc[row * PADDED_V_HEAD_DIM + col];
            float normalized                                   = acc_val / (s > 1e-10f ? s : 1.0f);
            O[o_offset + (ptrdiff_t) row * o_row_stride + col] = normalized;
        }
    }
}

// ============================================================================
// Wrapper for DG2 (nsize=8) with padding support
// ============================================================================
// V_FROM_K: when true, V is read from K's memory (MLA zero-copy optimization)
template <int64_t HEAD_DIM,
          int64_t V_HEAD_DIM,
          int64_t PADDED_HEAD_DIM,
          int64_t PADDED_V_HEAD_DIM,
          bool    V_FROM_K = false>
inline void flash_attn_coopmat_kernel_n8_padded(sycl::nd_item<2> it,
                                                const float *    Q,
                                                const float *    K,
                                                const float *    V,
                                                float *          O,
                                                float *          l_d,
                                                float *          m_d,
                                                const int64_t    N,
                                                const int64_t    N_kv,
                                                const int        n_heads,
                                                const int        n_kv_heads,
                                                const int        gqa_ratio,
                                                const float      scale,
                                                const float *    mask,
                                                const int64_t    mask_stride,
                                                const int        o_row_stride,
                                                float *          shmem) {
    flash_attn_coopmat_kernel_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads, gqa_ratio, scale, mask, mask_stride, o_row_stride,
        shmem);
}

// ============================================================================
// Wrapper for PVC/B60 (nsize=16) with padding support
// ============================================================================
// V_FROM_K: when true, V is read from K's memory (MLA zero-copy optimization)
template <int64_t HEAD_DIM,
          int64_t V_HEAD_DIM,
          int64_t PADDED_HEAD_DIM,
          int64_t PADDED_V_HEAD_DIM,
          bool    V_FROM_K = false>
inline void flash_attn_coopmat_kernel_n16_padded(sycl::nd_item<2> it,
                                                 const float *    Q,
                                                 const float *    K,
                                                 const float *    V,
                                                 float *          O,
                                                 float *          l_d,
                                                 float *          m_d,
                                                 const int64_t    N,
                                                 const int64_t    N_kv,
                                                 const int        n_heads,
                                                 const int        n_kv_heads,
                                                 const int        gqa_ratio,
                                                 const float      scale,
                                                 const float *    mask,
                                                 const int64_t    mask_stride,
                                                 const int        o_row_stride,
                                                 float *          shmem) {
    flash_attn_coopmat_kernel_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads, gqa_ratio, scale, mask, mask_stride, o_row_stride,
        shmem);
}

// ============================================================================
// flash_attn_coopmat_kernel_small_tile - Base template for 16x16 tiles
// ============================================================================
// This kernel uses BLOCK_M=16, BLOCK_N=16 to fit within 128KB SLM on Arc B60.
// Key optimizations:
// - Eliminates shV buffer entirely - loads V directly into shVT (transposed)
// - Reduces shAcc from 64KB to 32KB (BLOCK_M: 32->16)
// - Total SLM ~93KB vs ~222KB for the 32x32 kernel
// ============================================================================

template <int64_t HEAD_DIM,
          int64_t V_HEAD_DIM,
          int64_t PADDED_HEAD_DIM,
          int64_t PADDED_V_HEAD_DIM,
          bool    V_FROM_K = false,
          int     TM       = 8,
          int     TN       = 16,
          int     TK       = 16>
inline void flash_attn_coopmat_kernel_small_tile(
    sycl::nd_item<2> it,
    const float *    Q,
    const float *    K,
    const float *    V,  // When V_FROM_K=true, this is ignored and V is read from K
    float *          O,
    float *          l_d,
    float *          m_d,
    const int64_t    N,
    const int64_t    N_kv,
    const int        n_heads,
    const int        n_kv_heads,
    const int        gqa_ratio,
    const float      scale,
    const float *    mask,
    const int64_t    mask_stride,
    const int        o_row_stride,
    float *          shmem) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    // Small tiles for large head dimensions (fits 128KB SLM)
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int THREADS = 32;  // 2 subgroups instead of 4

    // Number of subgroups covering the M dimension
    constexpr int NUM_SG_M = BLOCK_M / TM;

    // Stride calculations with +8 padding for bank conflict avoidance
    constexpr int Q_STRIDE   = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE   = PADDED_HEAD_DIM + 8;
    // No V_STRIDE - we eliminate shV and load directly to shVT
    constexpr int V_T_STRIDE = BLOCK_N + 8;  // Col-major stride for shVT
    constexpr int S_STRIDE   = BLOCK_N + 8;
    constexpr int P_STRIDE   = BLOCK_N + 8;

    // Ensure tile dimensions are compatible with XMX
    static_assert(BLOCK_N % TK == 0, "BLOCK_N must be divisible by TK for P@V matmul");
    static_assert(PADDED_HEAD_DIM % TN == 0, "PADDED_HEAD_DIM must be divisible by TN for output tiles");
    static_assert(PADDED_V_HEAD_DIM % TN == 0, "PADDED_V_HEAD_DIM must be divisible by TN for output tiles");

    (void) l_d;
    (void) m_d;
    (void) n_heads;
    (void) n_kv_heads;

    const int lid   = it.get_local_id(0);
    const int gid_x = it.get_group(0);
    const int gid_y = it.get_group(1);
    auto      sg    = it.get_sub_group();

    const int sg_id = sg.get_group_linear_id();

    if (sg_id >= NUM_SG_M) {
        return;
    }

    const int row0 = gid_x * BLOCK_M;
    if (row0 >= N) {
        return;
    }

    const int head_idx    = gid_y;
    const int kv_head_idx = head_idx / gqa_ratio;

    const ptrdiff_t q_offset = (ptrdiff_t) (head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t) (kv_head_idx * N_kv) * HEAD_DIM;
    const ptrdiff_t v_offset = V_FROM_K ? k_offset : (ptrdiff_t) (kv_head_idx * N_kv) * V_HEAD_DIM;
    const ptrdiff_t o_offset = (ptrdiff_t) (head_idx * N + row0) * V_HEAD_DIM;

    constexpr ptrdiff_t V_ROW_STRIDE = V_FROM_K ? HEAD_DIM : V_HEAD_DIM;

    // SLM layout (optimized - no shV buffer):
    // shQ:      BLOCK_M * Q_STRIDE * sizeof(bf16)     = 16 * 584 * 2 = 18,688 bytes
    // shK:      BLOCK_N * K_STRIDE * sizeof(bf16)     = 16 * 584 * 2 = 18,688 bytes
    // shP:      BLOCK_M * P_STRIDE * sizeof(bf16)     = 16 * 24 * 2  = 768 bytes
    // shVT:     PADDED_V_HEAD_DIM * V_T_STRIDE * sizeof(bf16) = 512 * 24 * 2 = 24,576 bytes
    // shS:      BLOCK_M * S_STRIDE * sizeof(float)    = 16 * 24 * 4  = 1,536 bytes
    // rowMax:   BLOCK_M * sizeof(float)               = 16 * 4       = 64 bytes
    // rowSum:   BLOCK_M * sizeof(float)               = 16 * 4       = 64 bytes
    // rowAlpha: BLOCK_M * sizeof(float)               = 16 * 4       = 64 bytes
    // shAcc:    BLOCK_M * PADDED_V_HEAD_DIM * sizeof(float) = 16 * 512 * 4 = 32,768 bytes
    // Total: ~97KB (fits in 128KB)

    xmx_bfloat16 * shQ  = reinterpret_cast<xmx_bfloat16 *>(shmem);
    xmx_bfloat16 * shK  = shQ + BLOCK_M * Q_STRIDE;
    xmx_bfloat16 * shP  = shK + BLOCK_N * K_STRIDE;
    xmx_bfloat16 * shVT = shP + BLOCK_M * P_STRIDE;  // Directly after shP (no shV)

    float * shS      = reinterpret_cast<float *>(shVT + PADDED_V_HEAD_DIM * V_T_STRIDE);
    float * rowMax   = shS + BLOCK_M * S_STRIDE;
    float * rowSum   = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;
    float * shAcc    = rowAlpha + BLOCK_M;

    // Initialize per-row stats and output accumulator
    if (lid < BLOCK_M) {
        rowMax[lid] = -1.0e20f;
        rowSum[lid] = 0.0f;
    }
    for (int i = lid; i < BLOCK_M * PADDED_V_HEAD_DIM; i += THREADS) {
        shAcc[i] = 0.0f;
    }

    // Load Q tiles with scaling and padding
    for (int i = lid; i < BLOCK_M * PADDED_HEAD_DIM; i += THREADS) {
        const int r     = i / PADDED_HEAD_DIM;
        const int c     = i % PADDED_HEAD_DIM;
        const int q_row = row0 + r;

        if (q_row < N && c < HEAD_DIM) {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(Q[q_offset + (ptrdiff_t) r * HEAD_DIM + c] * scale);
        } else {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(0.0f);
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    const int num_kv_blocks = (N_kv + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int col0 = kv_block * BLOCK_N;

        // Load K tiles (stored as K^T col-major)
        for (int i = lid; i < BLOCK_N * PADDED_HEAD_DIM; i += THREADS) {
            const int r     = i / PADDED_HEAD_DIM;
            const int c     = i % PADDED_HEAD_DIM;
            const int k_row = col0 + r;

            xmx_bfloat16 k_val = xmx_bfloat16(0.0f);
            if (k_row < N_kv && c < HEAD_DIM) {
                const ptrdiff_t base = (ptrdiff_t) k_row * HEAD_DIM;
                k_val                = xmx_bfloat16(K[k_offset + base + c]);
            }
            // Store K^T col-major
            shK[c + r * K_STRIDE] = k_val;
        }

        // Load V tiles DIRECTLY into shVT (transposed) - eliminates shV buffer
        // shVT is col-major: [BLOCK_N rows, PADDED_V_HEAD_DIM cols]
        // Index: shVT[row + col * V_T_STRIDE]
        for (int i = lid; i < BLOCK_N * PADDED_V_HEAD_DIM; i += THREADS) {
            const int r     = i / PADDED_V_HEAD_DIM;  // kv position (0..BLOCK_N-1)
            const int c     = i % PADDED_V_HEAD_DIM;  // head dim (0..PADDED_V_HEAD_DIM-1)
            const int k_row = col0 + r;

            xmx_bfloat16 v_val = xmx_bfloat16(0.0f);
            if (k_row < N_kv && c < V_HEAD_DIM) {
                const ptrdiff_t base = (ptrdiff_t) k_row * V_ROW_STRIDE;
                v_val = V_FROM_K ? xmx_bfloat16(K[v_offset + base + c]) : xmx_bfloat16(V[v_offset + base + c]);
            }
            // Store directly in col-major transposed layout
            shVT[r + c * V_T_STRIDE] = v_val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Q @ K^T computation ============
        for (int j = 0; j < BLOCK_N; j += TN) {
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN, layout::dynamic> matS;
            joint_matrix_fill(sg, matS, 0.0f);

            for (int k = 0; k < PADDED_HEAD_DIM; k += TK) {
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, TM, TK, layout::row_major> mq;
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, TK, TN, layout::col_major> mk;

                auto mq_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shQ[(sg_id * TM) * Q_STRIDE + k]);
                auto mk_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shK[k + j * K_STRIDE]);

                joint_matrix_load(sg, mq, mq_ptr, Q_STRIDE);
                joint_matrix_load(sg, mk, mk_ptr, K_STRIDE);
                joint_matrix_mad(sg, matS, mq, mk, matS);
            }

            auto shS_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                    &shS[(sg_id * TM) * S_STRIDE + j]);
            joint_matrix_store(sg, matS, shS_ptr, S_STRIDE, layout::row_major);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Online Softmax ============
        if (lid < BLOCK_M) {
            const int row   = lid;
            const int q_row = row0 + row;
            float     m     = -1.0e20f;

            for (int c = 0; c < BLOCK_N; ++c) {
                float     s_val  = shS[row * S_STRIDE + c];
                const int kv_col = col0 + c;

                if (mask != nullptr && kv_col < N_kv && q_row < N) {
                    s_val += mask[q_row * mask_stride + kv_col];
                }
                shS[row * S_STRIDE + c] = s_val;
                m                       = sycl::fmax(m, s_val);
            }

            float m_prev  = rowMax[row];
            float m_new   = sycl::fmax(m_prev, m);
            float alpha   = sycl::exp(sycl::fmax(m_prev - m_new, -20.0f));
            rowAlpha[row] = alpha;
            rowMax[row]   = m_new;

            float s_sum = 0.0f;
            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val             = shS[row * S_STRIDE + c];
                float val               = sycl::exp(sycl::fmax(s_val - m_new, -20.0f));
                shS[row * S_STRIDE + c] = val;
                s_sum += val;
            }
            rowSum[row] = rowSum[row] * alpha + s_sum;
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Scale previous output by alpha & Convert P to bf16 ============
        for (int i = lid; i < BLOCK_M * PADDED_V_HEAD_DIM; i += THREADS) {
            const int row = i / PADDED_V_HEAD_DIM;
            shAcc[i] *= rowAlpha[row];
        }

        for (int i = lid; i < BLOCK_M * BLOCK_N; i += THREADS) {
            const int row             = i / BLOCK_N;
            const int col             = i % BLOCK_N;
            shP[row * P_STRIDE + col] = xmx_bfloat16(shS[row * S_STRIDE + col]);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ P @ V computation ============
        constexpr int NUM_OUT_TILES = PADDED_V_HEAD_DIM / TN;

        for (int out_tile = 0; out_tile < NUM_OUT_TILES; ++out_tile) {
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN, layout::dynamic> matPV;
            joint_matrix_fill(sg, matPV, 0.0f);

#        pragma unroll 2
            for (int k = 0; k < BLOCK_N; k += TK) {
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, TM, TK, layout::row_major> mp;
                auto                                                                           shP_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shP[(sg_id * TM) * P_STRIDE + k]);
                joint_matrix_load(sg, mp, shP_ptr, P_STRIDE);

                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, TK, TN, layout::col_major> mv;
                auto                                                                           shV_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                        &shVT[k + out_tile * TN * V_T_STRIDE]);
                joint_matrix_load(sg, mv, shV_ptr, V_T_STRIDE);

                joint_matrix_mad(sg, matPV, mp, mv, matPV);
            }

            // Store to scratch and accumulate
            const int scratch_offset = sg_id * TM * TN;
            auto      scratch_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                    &shS[scratch_offset]);
            joint_matrix_store(sg, matPV, scratch_ptr, TN, layout::row_major);
            it.barrier(sycl::access::fence_space::local_space);

            const int sg_lane = sg.get_local_linear_id();
            const int sg_size = sg.get_local_linear_range();
            for (int idx = sg_lane; idx < TM * TN; idx += sg_size) {
                const int local_row  = idx / TN;
                const int local_col  = idx % TN;
                const int global_row = sg_id * TM + local_row;
                const int global_col = out_tile * TN + local_col;
                if (global_row < BLOCK_M && global_col < PADDED_V_HEAD_DIM) {
                    shAcc[global_row * PADDED_V_HEAD_DIM + global_col] += shS[scratch_offset + idx];
                }
            }

            it.barrier(sycl::access::fence_space::local_space);
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    // ============ Final normalize and store ============
    for (int i = lid; i < BLOCK_M * V_HEAD_DIM; i += THREADS) {
        const int row   = i / V_HEAD_DIM;
        const int col   = i % V_HEAD_DIM;
        const int q_row = row0 + row;

        if (q_row < N) {
            float s                                            = rowSum[row];
            float acc_val                                      = shAcc[row * PADDED_V_HEAD_DIM + col];
            float normalized                                   = acc_val / (s > 1e-10f ? s : 1.0f);
            O[o_offset + (ptrdiff_t) row * o_row_stride + col] = normalized;
        }
    }
}

// ============================================================================
// Wrapper for small-tile kernel (DG2/Arc nsize=8 compatible)
// ============================================================================
template <int64_t HEAD_DIM,
          int64_t V_HEAD_DIM,
          int64_t PADDED_HEAD_DIM,
          int64_t PADDED_V_HEAD_DIM,
          bool    V_FROM_K = false>
inline void flash_attn_coopmat_kernel_small_tile_n8(sycl::nd_item<2> it,
                                                    const float *    Q,
                                                    const float *    K,
                                                    const float *    V,
                                                    float *          O,
                                                    float *          l_d,
                                                    float *          m_d,
                                                    const int64_t    N,
                                                    const int64_t    N_kv,
                                                    const int        n_heads,
                                                    const int        n_kv_heads,
                                                    const int        gqa_ratio,
                                                    const float      scale,
                                                    const float *    mask,
                                                    const int64_t    mask_stride,
                                                    const int        o_row_stride,
                                                    float *          shmem) {
    flash_attn_coopmat_kernel_small_tile<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads, gqa_ratio, scale, mask, mask_stride, o_row_stride,
        shmem);
}

// ============================================================================
// Wrapper for small-tile kernel (PVC/B60 nsize=16 compatible)
// ============================================================================
template <int64_t HEAD_DIM,
          int64_t V_HEAD_DIM,
          int64_t PADDED_HEAD_DIM,
          int64_t PADDED_V_HEAD_DIM,
          bool    V_FROM_K = false>
inline void flash_attn_coopmat_kernel_small_tile_n16(sycl::nd_item<2> it,
                                                     const float *    Q,
                                                     const float *    K,
                                                     const float *    V,
                                                     float *          O,
                                                     float *          l_d,
                                                     float *          m_d,
                                                     const int64_t    N,
                                                     const int64_t    N_kv,
                                                     const int        n_heads,
                                                     const int        n_kv_heads,
                                                     const int        gqa_ratio,
                                                     const float      scale,
                                                     const float *    mask,
                                                     const int64_t    mask_stride,
                                                     const int        o_row_stride,
                                                     float *          shmem) {
    flash_attn_coopmat_kernel_small_tile<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads, gqa_ratio, scale, mask, mask_stride, o_row_stride,
        shmem);
}

#    endif  // SYCL_EXT_ONEAPI_MATRIX
#endif      // GGML_SYCL_FATTN_XMX_HPP

// High-level XMX flash attention functions (extracted from fattn.cpp)

#ifdef SYCL_EXT_ONEAPI_MATRIX

template <int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM>
void ggml_sycl_op_flash_attn_coopmat_padded(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool q_is_f16 = (Q->type == GGML_TYPE_F16);
    const bool k_is_f16 = (K->type == GGML_TYPE_F16);
    const bool v_is_f16 = (V->type == GGML_TYPE_F16);

    const bool V_is_K_view = V->view_src && V->view_offs == 0 && (V->view_src == K || V->view_src == K->view_src);

    // Compile-time flag for MLA zero-copy optimization
    // When HEAD_DIM != V_HEAD_DIM and V is a view of K, kernel reads V from K directly
    constexpr bool V_FROM_K = (HEAD_DIM != V_HEAD_DIM);

    const float * Q_d_f32;
    const float * K_d_f32;
    const float * V_d_f32;
    float *       Q_d_f32_alloc = nullptr;
    float *       K_d_f32_alloc = nullptr;
    float *       V_d_f32_alloc = nullptr;

    dpct::queue_ptr stream = ctx.stream();

    // Initialize buffer pool if needed
    if (!ctx.fattn_buffers) {
        ctx.fattn_buffers = std::make_unique<flash_attn_buffers>();
    }

    const int64_t N                    = Q->ne[1];
    const int64_t N_kv                 = K->ne[1];
    const int64_t n_heads_per_batch    = Q->ne[2];
    const int64_t n_kv_heads_per_batch = K->ne[2];
    const int64_t batch                = Q->ne[3];
    const int64_t gqa_ratio            = n_heads_per_batch / n_kv_heads_per_batch;

    const int64_t n_heads    = n_heads_per_batch * batch;
    const int64_t n_kv_heads = n_kv_heads_per_batch * batch;

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Event-based synchronization: collect prep events for kernel dependency
    std::vector<sycl::event> prep_events;

    if (q_is_f16) {
        const sycl::half * Q_d = (const sycl::half *) Q->data;
        Q_d_f32_alloc          = ctx.fattn_buffers->get_Q(N * HEAD_DIM * n_heads * sizeof(float), stream);

        const int64_t q_stride_seq   = Q->nb[1] / sizeof(sycl::half);
        const int64_t q_stride_head  = Q->nb[2] / sizeof(sycl::half);
        const int64_t q_stride_batch = Q->nb[3] / sizeof(sycl::half);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_heads_per_batch; ++head) {
                const int64_t head_total = b * n_heads_per_batch + head;
                const int64_t n_elements = N * HEAD_DIM;
                sycl::event   evt        = stream->submit([&](sycl::handler & cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq                                               = idx / HEAD_DIM;
                            const int64_t dim                                               = idx % HEAD_DIM;
                            Q_d_f32_alloc[head_total * N * HEAD_DIM + seq * HEAD_DIM + dim] = static_cast<float>(
                                Q_d[dim + head * q_stride_head + seq * q_stride_seq + b * q_stride_batch]);
                        }
                    });
                });
                prep_events.push_back(evt);
            }
        }
        Q_d_f32 = Q_d_f32_alloc;
    } else {
        const float * Q_d = (const float *) Q->data;
        Q_d_f32_alloc     = ctx.fattn_buffers->get_Q(N * HEAD_DIM * n_heads * sizeof(float), stream);

        const int64_t q_stride_seq   = Q->nb[1] / sizeof(float);
        const int64_t q_stride_head  = Q->nb[2] / sizeof(float);
        const int64_t q_stride_batch = Q->nb[3] / sizeof(float);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_heads_per_batch; ++head) {
                const int64_t head_total = b * n_heads_per_batch + head;
                const int64_t n_elements = N * HEAD_DIM;
                sycl::event   evt        = stream->submit([&](sycl::handler & cgh) {
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
                prep_events.push_back(evt);
            }
        }
        Q_d_f32 = Q_d_f32_alloc;
    }

    if (k_is_f16) {
        const sycl::half * K_d = (const sycl::half *) K->data;
        K_d_f32_alloc          = ctx.fattn_buffers->get_K(N_kv * HEAD_DIM * n_kv_heads * sizeof(float), stream);

        const int64_t k_stride_seq   = K->nb[1] / sizeof(sycl::half);
        const int64_t k_stride_head  = K->nb[2] / sizeof(sycl::half);
        const int64_t k_stride_batch = K->nb[3] / sizeof(sycl::half);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * HEAD_DIM;
                sycl::event   evt        = stream->submit([&](sycl::handler & cgh) {
                    cgh.parallel_for(sycl::range<1>((n_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                        const int idx = it.get_id(0);
                        if (idx < n_elements) {
                            const int64_t seq                                                  = idx / HEAD_DIM;
                            const int64_t dim                                                  = idx % HEAD_DIM;
                            K_d_f32_alloc[head_total * N_kv * HEAD_DIM + seq * HEAD_DIM + dim] = static_cast<float>(
                                K_d[dim + head * k_stride_head + seq * k_stride_seq + b * k_stride_batch]);
                        }
                    });
                });
                prep_events.push_back(evt);
            }
        }
        K_d_f32 = K_d_f32_alloc;
    } else {
        const float * K_d = (const float *) K->data;
        K_d_f32_alloc     = ctx.fattn_buffers->get_K(N_kv * HEAD_DIM * n_kv_heads * sizeof(float), stream);

        const int64_t k_stride_seq   = K->nb[1] / sizeof(float);
        const int64_t k_stride_head  = K->nb[2] / sizeof(float);
        const int64_t k_stride_batch = K->nb[3] / sizeof(float);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * HEAD_DIM;
                sycl::event   evt        = stream->submit([&](sycl::handler & cgh) {
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
                prep_events.push_back(evt);
            }
        }
        K_d_f32 = K_d_f32_alloc;
    }

    if (V_is_K_view && V_FROM_K) {
        // MLA zero-copy: kernel reads V directly from K with HEAD_DIM stride
        // V_d_f32 points to K but kernel knows to use V_FROM_K template
        V_d_f32 = K_d_f32;
    } else if (V_is_K_view) {
        // V is view of K with same dimension - direct reuse
        V_d_f32 = K_d_f32;
    } else if (v_is_f16) {
        const sycl::half * V_d = (const sycl::half *) V->data;
        V_d_f32_alloc          = ctx.fattn_buffers->get_V(N_kv * V_HEAD_DIM * n_kv_heads * sizeof(float), stream);

        const int64_t v_stride_seq   = V->nb[1] / sizeof(sycl::half);
        const int64_t v_stride_head  = V->nb[2] / sizeof(sycl::half);
        const int64_t v_stride_batch = V->nb[3] / sizeof(sycl::half);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * V_HEAD_DIM;
                sycl::event   evt        = stream->submit([&](sycl::handler & cgh) {
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
                prep_events.push_back(evt);
            }
        }
        V_d_f32 = V_d_f32_alloc;
    } else {
        const float * V_d = (const float *) V->data;
        V_d_f32_alloc     = ctx.fattn_buffers->get_V(N_kv * V_HEAD_DIM * n_kv_heads * sizeof(float), stream);

        const int64_t v_stride_seq   = V->nb[1] / sizeof(float);
        const int64_t v_stride_head  = V->nb[2] / sizeof(float);
        const int64_t v_stride_batch = V->nb[3] / sizeof(float);

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t head = 0; head < n_kv_heads_per_batch; ++head) {
                const int64_t head_total = b * n_kv_heads_per_batch + head;
                const int64_t n_elements = N_kv * V_HEAD_DIM;
                sycl::event   evt        = stream->submit([&](sycl::handler & cgh) {
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
                prep_events.push_back(evt);
            }
        }
        V_d_f32 = V_d_f32_alloc;
    }

    float * dst_d = (float *) dst->data;

    // Use small tiles for large head dimensions to fit within 128KB SLM
    constexpr bool USE_SMALL_TILES = (PADDED_V_HEAD_DIM >= 512 || PADDED_HEAD_DIM >= 512);
    constexpr int  BLOCK_M         = USE_SMALL_TILES ? 16 : 32;
    constexpr int  BLOCK_N         = USE_SMALL_TILES ? 16 : 32;
    constexpr int  THREADS_PER_WG  = USE_SMALL_TILES ? 32 : 64;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;

    float * l_d = ctx.fattn_buffers->get_l(N * n_heads * sizeof(float), stream);
    float * m_d = ctx.fattn_buffers->get_m(N * n_heads * sizeof(float), stream);

    sycl::range<2> global(Tr * THREADS_PER_WG, n_heads);
    sycl::range<2> local(THREADS_PER_WG, 1);

    xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(stream->get_device());

    constexpr int Q_STRIDE   = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE   = PADDED_HEAD_DIM + 8;
    constexpr int P_STRIDE   = BLOCK_N + 8;
    constexpr int S_STRIDE   = BLOCK_N + 8;
    constexpr int V_T_STRIDE = BLOCK_N + 8;

    // Small-tile kernel eliminates shV buffer (loads V directly to shVT)
    // Large-tile kernel still uses shV
    constexpr int    V_STRIDE   = USE_SMALL_TILES ? 0 : (PADDED_V_HEAD_DIM + 8);
    constexpr size_t BF16_BYTES = (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE + BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE +
                                   PADDED_V_HEAD_DIM * V_T_STRIDE) *
                                  sizeof(sycl::half);
    constexpr size_t FLOAT_BYTES = (BLOCK_M * S_STRIDE + BLOCK_M * 3 + BLOCK_M * PADDED_V_HEAD_DIM) * sizeof(float);
    constexpr size_t SHMEM_SIZE  = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    const float * mask_d           = nullptr;
    float *       mask_d_f32_alloc = nullptr;
    int64_t       mask_stride      = N_kv;
    if (mask != nullptr && mask->data != nullptr) {
        if (mask->type == GGML_TYPE_F32) {
            mask_d      = (const float *) mask->data;
            mask_stride = mask->nb[1] / sizeof(float);
        } else if (mask->type == GGML_TYPE_F16) {
            const int64_t mask_n_kv     = mask->ne[0];
            const int64_t mask_n_q      = mask->ne[1];
            const int64_t mask_elements = mask_n_kv * mask_n_q;

            mask_d_f32_alloc                       = ctx.fattn_buffers->get_mask(mask_elements * sizeof(float), stream);
            const sycl::half * mask_f16            = (const sycl::half *) mask->data;
            const ptrdiff_t    mask_row_stride_f16 = mask->nb[1] / sizeof(sycl::half);

            sycl::event mask_event = stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(sycl::range<1>((mask_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < mask_elements) {
                        const int64_t row     = idx / mask_n_kv;
                        const int64_t col     = idx % mask_n_kv;
                        mask_d_f32_alloc[idx] = static_cast<float>(mask_f16[row * mask_row_stride_f16 + col]);
                    }
                });
            });
            prep_events.push_back(mask_event);

            mask_d      = mask_d_f32_alloc;
            mask_stride = mask_n_kv;
        }
    }

    // Allocate temp output buffer and output stride for kernel
    float *   O_temp       = ctx.fattn_buffers->get_O(N * V_HEAD_DIM * n_heads * sizeof(float), stream);
    const int o_row_stride = V_HEAD_DIM;

    // Use V_FROM_K template parameter for MLA zero-copy optimization
    // When V_FROM_K is true, kernel reads V directly from K's memory
    sycl::event xmx_event;
    if constexpr (USE_SMALL_TILES) {
        // Small-tile kernel for large head dimensions (576/512)
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            xmx_event = stream->submit([&](sycl::handler & cgh) {
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     float * shmem_ptr = &shmem[0];
                                     flash_attn_coopmat_kernel_small_tile_n8<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM,
                                                                             PADDED_V_HEAD_DIM, V_FROM_K>(
                                         it, static_cast<const float *>(Q_d_f32), static_cast<const float *>(K_d_f32),
                                         static_cast<const float *>(V_d_f32), O_temp, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride, o_row_stride, shmem_ptr);
                                 });
            });
        } else {
            xmx_event = stream->submit([&](sycl::handler & cgh) {
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     float * shmem_ptr = &shmem[0];
                                     flash_attn_coopmat_kernel_small_tile_n16<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM,
                                                                              PADDED_V_HEAD_DIM, V_FROM_K>(
                                         it, static_cast<const float *>(Q_d_f32), static_cast<const float *>(K_d_f32),
                                         static_cast<const float *>(V_d_f32), O_temp, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride, o_row_stride, shmem_ptr);
                                 });
            });
        }
    } else {
        // Standard 32x32 tile kernel for normal head dimensions
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            xmx_event = stream->submit([&](sycl::handler & cgh) {
                // Depend on prep kernels (Q/K/V reorder, mask conversion)
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     flash_attn_coopmat_kernel_n8_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM,
                                                                         PADDED_V_HEAD_DIM, V_FROM_K>(
                                         it, static_cast<const float *>(Q_d_f32), static_cast<const float *>(K_d_f32),
                                         static_cast<const float *>(V_d_f32), O_temp, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride, o_row_stride,
                                         shmem.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        } else {
            xmx_event = stream->submit([&](sycl::handler & cgh) {
                // Depend on prep kernels (Q/K/V reorder, mask conversion)
                cgh.depends_on(prep_events);
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     flash_attn_coopmat_kernel_n16_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM,
                                                                          PADDED_V_HEAD_DIM, V_FROM_K>(
                                         it, static_cast<const float *>(Q_d_f32), static_cast<const float *>(K_d_f32),
                                         static_cast<const float *>(V_d_f32), O_temp, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride, o_row_stride,
                                         shmem.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
    }

    // Output reorder kernels depend on XMX kernel completion
    std::vector<sycl::event> reorder_events;
    // Output layout is permuted: ne = [DV, n_heads, N, batch]
    // nb[1] = stride between heads (dim 1), nb[2] = stride between seq positions (dim 2)
    const int64_t            dst_stride_head  = dst->nb[1] / sizeof(float);
    const int64_t            dst_stride_seq   = dst->nb[2] / sizeof(float);
    const int64_t            dst_stride_batch = dst->nb[3] / sizeof(float);

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t head = 0; head < n_heads_per_batch; ++head) {
            const int64_t head_total    = b * n_heads_per_batch + head;
            const int64_t n_elements    = N * V_HEAD_DIM;
            sycl::event   reorder_event = stream->submit([&](sycl::handler & cgh) {
                cgh.depends_on(xmx_event);
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
            reorder_events.push_back(reorder_event);
        }
    }

    // Wait only at the end when results are needed
    sycl::event::wait(reorder_events);

    // All buffers are pooled in ctx.fattn_buffers and will be reused on next call
}

// ============================================================================
// Stride-Aware Direct Loading Flash Attention
// ============================================================================
// This variant eliminates the intermediate F32 repack buffers by loading
// directly from ggml's strided tensor layout. This reduces memory bandwidth
// by 2x and eliminates temporary allocations.
//
// Enabled by setting GGML_SYCL_FLASH_ATTN_DIRECT=1
// ============================================================================
template <int64_t DQK, int64_t DV, int BLOCK_M = 32, int BLOCK_N = 32>
void ggml_sycl_op_flash_attn_coopmat_direct(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (getenv("GGML_SYCL_FLASH_ATTN_DEBUG")) {
        GGML_SYCL_DEBUG("[SYCL] Flash attention coopmat_direct kernel: DQK=%ld, DV=%ld, BLOCK_M=%d, BLOCK_N=%d\n", DQK,
                        DV, BLOCK_M, BLOCK_N);
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const bool is_f16      = (Q->type == GGML_TYPE_F16);
    const bool V_is_K_view = V->view_src && V->view_offs == 0 && (V->view_src == K || V->view_src == K->view_src);

    dpct::queue_ptr stream = ctx.stream();

    const int64_t N          = Q->ne[1];
    const int64_t N_kv       = K->ne[1];
    const int64_t n_heads    = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio  = n_heads / n_kv_heads;

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Compute strides for direct loading
    fattn_tensor_strides strides = compute_tensor_strides(Q, K, V, dst);

    // Get raw data pointers (type depends on input format)
    const void * Q_data = Q->data;
    const void * K_data = K->data;
    const void * V_data = V_is_K_view ? K->data : V->data;
    float *      O_data = (float *) dst->data;

    // Initialize buffer pool if needed
    if (!ctx.fattn_buffers) {
        ctx.fattn_buffers = std::make_unique<flash_attn_buffers>();
    }

    // Process mask if present
    const float * mask_d           = nullptr;
    float *       mask_d_f32_alloc = nullptr;
    int64_t       mask_stride_val  = N_kv;
    if (mask != nullptr && mask->data != nullptr) {
        if (mask->type == GGML_TYPE_F32) {
            mask_d          = (const float *) mask->data;
            mask_stride_val = mask->nb[1] / sizeof(float);
        } else if (mask->type == GGML_TYPE_F16) {
            const int64_t mask_n_kv     = mask->ne[0];
            const int64_t mask_n_q      = mask->ne[1];
            const int64_t mask_elements = mask_n_kv * mask_n_q;

            mask_d_f32_alloc                       = ctx.fattn_buffers->get_mask(mask_elements * sizeof(float), stream);
            const sycl::half * mask_f16            = (const sycl::half *) mask->data;
            const ptrdiff_t    mask_row_stride_f16 = mask->nb[1] / sizeof(sycl::half);

            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(sycl::range<1>((mask_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < mask_elements) {
                        const int64_t row     = idx / mask_n_kv;
                        const int64_t col     = idx % mask_n_kv;
                        mask_d_f32_alloc[idx] = static_cast<float>(mask_f16[row * mask_row_stride_f16 + col]);
                    }
                });
            });
            mask_d          = mask_d_f32_alloc;
            mask_stride_val = mask_n_kv;
        }
    }

    constexpr int THREADS_PER_WG = 64;

    const int Tr = (N + BLOCK_M - 1) / BLOCK_M;

    // Unused but needed for API compatibility
    float * l_d = nullptr;
    float * m_d = nullptr;

    sycl::range<2> global(Tr * THREADS_PER_WG, n_heads);
    sycl::range<2> local(THREADS_PER_WG, 1);

    xmx_tile_kind tile_kind = ggml_sycl_flash_attn_get_tile_kind(stream->get_device());

    // Shared memory size calculation
    constexpr int    Q_STRIDE   = DQK + 8;
    constexpr int    K_STRIDE   = DQK + 8;
    constexpr int    V_STRIDE   = DV + 8;
    constexpr int    P_STRIDE   = BLOCK_N + 8;  // BLOCK_N + padding
    constexpr int    V_T_STRIDE = BLOCK_N + 8;
    constexpr size_t BF16_BYTES =
        (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE + BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE + DV * V_T_STRIDE) *
        sizeof(sycl::half);
    constexpr size_t FLOAT_BYTES = (BLOCK_M * (BLOCK_N + 8) + BLOCK_M * 3 + BLOCK_M * DV) * sizeof(float);
    constexpr size_t SHMEM_SIZE  = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    static bool first_call = true;
    if (first_call && getenv("GGML_SYCL_FLASH_ATTN_DEBUG")) {
        GGML_SYCL_DEBUG(
            "ggml_sycl: DIRECT XMX flash_attn N=%ld N_kv=%ld n_heads=%ld DQK=%ld DV=%ld is_f16=%d BLOCK_M=%d "
            "BLOCK_N=%d SHMEM_SIZE=%lu\n",
            N, N_kv, n_heads, DQK, DV, is_f16, BLOCK_M, BLOCK_N, SHMEM_SIZE * sizeof(float));
        GGML_SYCL_DEBUG("ggml_sycl: Q strides: seq=%ld head=%ld\n", strides.q_stride_seq, strides.q_stride_head);
        GGML_SYCL_DEBUG("ggml_sycl: K strides: seq=%ld head=%ld\n", strides.k_stride_seq, strides.k_stride_head);
        GGML_SYCL_DEBUG("ggml_sycl: V strides: seq=%ld head=%ld\n", strides.v_stride_seq, strides.v_stride_head);
        GGML_SYCL_DEBUG("ggml_sycl: O strides: seq=%ld head=%ld\n", strides.o_stride_seq, strides.o_stride_head);
        first_call = false;
    }

    // Select kernel based on input type and tile kind
    // Use padded kernels (direct loading assumes contiguous layout after reorder)
    if (is_f16) {
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     float * shmem_ptr = &shmem[0];
                                     flash_attn_coopmat_kernel_n8_padded<DQK, DV, DQK, DV, false>(
                                         it, static_cast<const float *>(Q_data), static_cast<const float *>(K_data),
                                         static_cast<const float *>(V_data), O_data, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, DV, shmem_ptr);
                                 });
            });
        } else {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     float * shmem_ptr = &shmem[0];
                                     flash_attn_coopmat_kernel_n16_padded<DQK, DV, DQK, DV, false>(
                                         it, static_cast<const float *>(Q_data), static_cast<const float *>(K_data),
                                         static_cast<const float *>(V_data), O_data, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, DV, shmem_ptr);
                                 });
            });
        }
    } else {
        // F32 input
        if (tile_kind == xmx_tile_kind::tile_dg2) {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     float * shmem_ptr = &shmem[0];
                                     flash_attn_coopmat_kernel_n8_padded<DQK, DV, DQK, DV, false>(
                                         it, static_cast<const float *>(Q_data), static_cast<const float *>(K_data),
                                         static_cast<const float *>(V_data), O_data, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, DV, shmem_ptr);
                                 });
            });
        } else {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);

                cgh.parallel_for(sycl::nd_range<2>(global, local),
                                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                                     float * shmem_ptr = &shmem[0];
                                     flash_attn_coopmat_kernel_n16_padded<DQK, DV, DQK, DV, false>(
                                         it, static_cast<const float *>(Q_data), static_cast<const float *>(K_data),
                                         static_cast<const float *>(V_data), O_data, l_d, m_d, N, N_kv, n_heads,
                                         n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, DV, shmem_ptr);
                                 });
            });
        }
    }

    stream->wait();

    // Mask buffer is pooled in ctx.fattn_buffers and will be reused on next call
}

// ============================================================================
// XMX Flash Attention with KV-Split (Flash Decoding)
// ============================================================================
// This function implements the KV-split (flash decoding) path for XMX flash
// attention. It splits the KV dimension across multiple workgroups to improve
// utilization for long contexts.
//
// Template parameters:
// - HEAD_DIM, V_HEAD_DIM: actual head dimensions
// - TM, TN, TK: XMX tile dimensions for joint matrix operations
// - BLOCK_M, BLOCK_N: thread block dimensions (32x32 for standard, 16x16 for small-tile)
// - InputType:fattn_input_type (f32 or f16) for input data type
// - V_FROM_K: bool, when true V is read from K (MLA zero-copy optimization)
// ============================================================================
template <int64_t          HEAD_DIM,
          int64_t          V_HEAD_DIM,
          int              TM,
          int              TN,
          int              TK,
          int              BLOCK_M,
          int              BLOCK_N,
          fattn_input_type InputType,
          bool             V_FROM_K>
void ggml_sycl_op_flash_attn_coopmat_kvsplit(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    dpct::queue_ptr stream = ctx.stream();

    // Initialize buffer pool if needed
    if (!ctx.fattn_buffers) {
        ctx.fattn_buffers = std::make_unique<flash_attn_buffers>();
    }

    const int64_t N          = Q->ne[1];
    const int64_t N_kv       = K->ne[1];
    const int64_t n_heads    = Q->ne[2];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t gqa_ratio  = n_heads / n_kv_heads;

    float scale = 1.0f;
    std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Determine number of splits
    const int64_t n_splits     = get_kv_split_count(N_kv, n_heads);
    const int64_t kv_per_split = (N_kv + n_splits - 1) / n_splits;

    // Partial buffer layout: per-split M, L, O
    // Each split contributes: 1x float (M) + 1x float (L) + (N * V_HEAD_DIM) floats (O)
    const int64_t partial_size   = 2 + (N * V_HEAD_DIM);  // M, L, and O in float
    const int64_t partials_total = n_splits * n_heads * partial_size;

    // Use partials buffer from pool (maps to partials_buf)
    float * partials = ctx.fattn_buffers->get_partials(partials_total * sizeof(float), stream);

    // Allocate l_d, m_d buffers for row statistics from pool
    float * l_d = ctx.fattn_buffers->get_l(N * n_heads * sizeof(float), stream);
    float * m_d = ctx.fattn_buffers->get_m(N * n_heads * sizeof(float), stream);

    // Prepare mask if needed
    const float * mask_d           = nullptr;
    float *       mask_d_f32_alloc = nullptr;
    int64_t       mask_stride_val  = N_kv;
    if (mask != nullptr && mask->data != nullptr) {
        if (mask->type == GGML_TYPE_F32) {
            mask_d          = (const float *) mask->data;
            mask_stride_val = mask->nb[1] / sizeof(float);
        } else if (mask->type == GGML_TYPE_F16) {
            const int64_t mask_n_kv                = mask->ne[0];
            const int64_t mask_n_q                 = mask->ne[1];
            const int64_t mask_elements            = mask_n_kv * mask_n_q;
            mask_d_f32_alloc                       = ctx.fattn_buffers->get_mask(mask_elements * sizeof(float), stream);
            const sycl::half * mask_f16            = (const sycl::half *) mask->data;
            const ptrdiff_t    mask_row_stride_f16 = mask->nb[1] / sizeof(sycl::half);
            stream->submit([&](sycl::handler & cgh) {
                cgh.parallel_for(sycl::range<1>((mask_elements + 255) / 256 * 256), [=](sycl::item<1> it) {
                    const int idx = it.get_id(0);
                    if (idx < mask_elements) {
                        const int64_t row     = idx / mask_n_kv;
                        const int64_t col     = idx % mask_n_kv;
                        mask_d_f32_alloc[idx] = static_cast<float>(mask_f16[row * mask_row_stride_f16 + col]);
                    }
                });
            });
            mask_d          = mask_d_f32_alloc;
            mask_stride_val = mask_n_kv;
        }
    }

    // Prepare sinks if needed (sinks are F32 per-head values)
    const float * sinks_d = nullptr;
    if (sinks != nullptr && sinks->data != nullptr) {
        sinks_d = (const float *) sinks->data;
    }

    // Output buffer
    float *       O_d           = (float *) dst->data;
    // Output layout is permuted: ne = [DV, n_heads, N, batch]
    // nb[1] = stride between heads, nb[2] = stride between seq positions
    const int64_t o_head_stride = dst->nb[1] / sizeof(float);
    const int64_t o_seq_stride  = dst->nb[2] / sizeof(float);
    const int64_t o_row_stride  = V_HEAD_DIM;

    // Compute shared memory size for the kernel
    constexpr int    Q_STRIDE   = HEAD_DIM + 8;
    constexpr int    K_STRIDE   = HEAD_DIM + 8;
    constexpr int    V_STRIDE   = V_HEAD_DIM + 8;
    constexpr int    P_STRIDE   = BLOCK_N + 8;
    constexpr int    V_T_STRIDE = BLOCK_N + 8;
    constexpr int    S_STRIDE   = BLOCK_N + 8;
    constexpr size_t BF16_BYTES =
        (BLOCK_M * Q_STRIDE + BLOCK_N * K_STRIDE + BLOCK_N * V_STRIDE + BLOCK_M * P_STRIDE + V_HEAD_DIM * V_T_STRIDE) *
        sizeof(sycl::half);
    constexpr size_t FLOAT_BYTES = (BLOCK_M * S_STRIDE + BLOCK_M * 3 + BLOCK_M * V_HEAD_DIM) * sizeof(float);
    constexpr size_t SHMEM_SIZE  = (BF16_BYTES + FLOAT_BYTES + sizeof(float) - 1) / sizeof(float);

    constexpr int  THREADS = BLOCK_M * 2;  // 64 for BLOCK_M=32, 32 for BLOCK_M=16
    const int      Tr      = (N + BLOCK_M - 1) / BLOCK_M;
    sycl::range<2> global(Tr * THREADS, n_heads);
    sycl::range<2> local(THREADS, 1);

    // Launch kernels for each split
    for (int64_t split = 0; split < n_splits; ++split) {
        const int64_t kv_start = split * kv_per_split;
        const int64_t kv_end   = std::min(kv_start + kv_per_split, N_kv);
        if (kv_start >= kv_end) {
            continue;
        }

        float * partials_for_split = partials + split * n_heads * N * partial_size;

        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), cgh);
            cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(
                                                                   16)]] {
                // Choose kernel based on BLOCK_M size
                float * shmem_ptr = &shmem[0];
                if constexpr (BLOCK_M == 16) {
                    // Small-tile kernel (16x16) for large head dimensions
                    if (TM == 8) {  // DG2/Arc B60
                        flash_attn_coopmat_kernel_small_tile_n8<HEAD_DIM, V_HEAD_DIM, HEAD_DIM, V_HEAD_DIM, V_FROM_K>(
                            it, static_cast<const float *>(Q->data), static_cast<const float *>(K->data),
                            static_cast<const float *>(V->data), partials_for_split, l_d, m_d, N, N_kv, n_heads,
                            n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, o_row_stride, shmem_ptr);
                    } else {  // PVC/B60 with TM=8 still
                        flash_attn_coopmat_kernel_small_tile_n16<HEAD_DIM, V_HEAD_DIM, HEAD_DIM, V_HEAD_DIM, V_FROM_K>(
                            it, static_cast<const float *>(Q->data), static_cast<const float *>(K->data),
                            static_cast<const float *>(V->data), partials_for_split, l_d, m_d, N, N_kv, n_heads,
                            n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, o_row_stride, shmem_ptr);
                    }
                } else {
                    // Standard 32x32 tile kernel
                    float * shmem_ptr = &shmem[0];
                    if (TM == 8) {  // DG2/Arc B60
                        flash_attn_coopmat_kernel_n8_padded<HEAD_DIM, V_HEAD_DIM, HEAD_DIM, V_HEAD_DIM, V_FROM_K>(
                            it, static_cast<const float *>(Q->data), static_cast<const float *>(K->data),
                            static_cast<const float *>(V->data), partials_for_split, l_d, m_d, N, N_kv, n_heads,
                            n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, o_row_stride, shmem_ptr);
                    } else {  // PVC/B60
                        flash_attn_coopmat_kernel_n16_padded<HEAD_DIM, V_HEAD_DIM, HEAD_DIM, V_HEAD_DIM, V_FROM_K>(
                            it, static_cast<const float *>(Q->data), static_cast<const float *>(K->data),
                            static_cast<const float *>(V->data), partials_for_split, l_d, m_d, N, N_kv, n_heads,
                            n_kv_heads, gqa_ratio, scale, mask_d, mask_stride_val, o_row_stride, shmem_ptr);
                    }
                }
            });
        });
    }

    // Reduction kernel to combine partials into permuted output layout
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_heads * N, V_HEAD_DIM), sycl::range<2>(1, V_HEAD_DIM)),
                         [=](sycl::nd_item<2> it) {
                             flash_attn_combine_splits_kernel<V_HEAD_DIM>(it, partials, O_d, n_splits, n_heads, N,
                                                                          partial_size, o_head_stride, o_seq_stride,
                                                                          sinks_d);
                         });
    });

    // Cleanup
    stream->wait();
    // All buffers are pooled in ctx.fattn_buffers and will be reused on next call
}

// Explicit instantiations for XMX flash attention kernels
template void ggml_sycl_op_flash_attn_coopmat_padded<40, 40, 64, 64>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<48, 48, 64, 64>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<56, 56, 64, 64>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<72, 72, 80, 80>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<88, 88, 96, 96>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<104, 104, 112, 112>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<64, 64, 64, 64>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<96, 96, 96, 96>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<128, 128, 128, 128>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_padded<256, 256, 256, 256>(ggml_backend_sycl_context &, ggml_tensor *);

template void ggml_sycl_op_flash_attn_coopmat_direct<32, 32>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_direct<64, 64>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_direct<96, 96>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_direct<128, 128>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_direct<256, 256>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_direct<512, 512>(ggml_backend_sycl_context &, ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_direct<576, 512, 8, 16>(ggml_backend_sycl_context &, ggml_tensor *);

template void ggml_sycl_op_flash_attn_coopmat_kvsplit<32, 32, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<32, 32, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<64, 64, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<64, 64, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<96, 96, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<96, 96, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<128, 128, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<128, 128, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<256, 256, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<256, 256, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<512, 512, 8, 16, 16, 32, 32, fattn_input_type::f16, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<512, 512, 8, 16, 16, 32, 32, fattn_input_type::f32, false>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<576, 512, 8, 16, 16, 8, 16, fattn_input_type::f16, true>(
    ggml_backend_sycl_context &,
    ggml_tensor *);
template void ggml_sycl_op_flash_attn_coopmat_kvsplit<576, 512, 8, 16, 16, 8, 16, fattn_input_type::f32, true>(
    ggml_backend_sycl_context &,
    ggml_tensor *);

#endif  // SYCL_EXT_ONEAPI_MATRIX
