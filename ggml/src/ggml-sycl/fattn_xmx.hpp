#ifndef GGML_SYCL_FATTN_XMX_HPP
#define GGML_SYCL_FATTN_XMX_HPP

#include "fattn_common.hpp"
#include "common.hpp"
#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif

#if defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif

#ifdef SYCL_EXT_COOPERATIVE_MATRICES

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
template <int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM, bool V_FROM_K = false, int TM = 8, int TN = 16, int TK = 16>
inline void flash_attn_coopmat_kernel_padded(
    sycl::nd_item<2> it,
    const float * Q,
    const float * K,
    const float * V,  // When V_FROM_K=true, this is ignored and V is read from K
    float * O,
    float * l_d,
    float * m_d,
    const int64_t N,
    const int64_t N_kv,
    const int n_heads,
    const int n_kv_heads,
    const int gqa_ratio,
    const float scale,
    const float * mask,
    const int64_t mask_stride,
    const int o_row_stride,
    float * shmem
) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int THREADS = 64;

    // Number of subgroups covering the M dimension
    constexpr int NUM_SG_M = BLOCK_M / TM;

    // Use PADDED_HEAD_DIM for Q/K XMX computations, PADDED_V_HEAD_DIM for V/O XMX computations
    constexpr int Q_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int V_STRIDE = PADDED_V_HEAD_DIM + 8;
    // V is stored col-major in shVT as [BLOCK_N rows, PADDED_V_HEAD_DIM cols] for P@V
    // For col-major, stride (leading dimension) must be >= number of rows = BLOCK_N
    constexpr int V_T_STRIDE = BLOCK_N + 8;  // Must be >= BLOCK_N for col-major storage
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;

    // Ensure BLOCK_N is divisible by TK for P@V computation
    static_assert(BLOCK_N % TK == 0, "BLOCK_N must be divisible by TK for P@V matmul");
    static_assert(PADDED_HEAD_DIM % TN == 0, "PADDED_HEAD_DIM must be divisible by TN for output tiles");
    static_assert(PADDED_V_HEAD_DIM % TN == 0, "PADDED_V_HEAD_DIM must be divisible by TN for output tiles");

    // Shared memory layout (sized for PADDED_HEAD_DIM)
    // shVT is [BLOCK_N rows x PADDED_HEAD_DIM cols] col-major, size = PADDED_HEAD_DIM * V_T_STRIDE

    (void)l_d; // suppress unused parameter warning
    (void)m_d; // suppress unused parameter warning
    (void)n_heads; // suppress unused parameter warning
    (void)n_kv_heads; // suppress unused parameter warning

    const int lid = it.get_local_id(0);
    const int gid_x = it.get_group(0);
    const int gid_y = it.get_group(1);
    auto sg = it.get_sub_group();
    
    const int sg_id = sg.get_group_linear_id();
    
    if (sg_id >= NUM_SG_M) return;

    const int row0 = gid_x * BLOCK_M;
    if (row0 >= N) return;

    const int head_idx = gid_y;
    const int kv_head_idx = head_idx / gqa_ratio;

    // Use HEAD_DIM for Q/K actual data access, V_HEAD_DIM for V/O actual data access
    // Use PADDED_HEAD_DIM for Q/K XMX operations, PADDED_V_HEAD_DIM for V/O XMX operations
    const ptrdiff_t q_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t)(kv_head_idx * N_kv) * HEAD_DIM;  // K uses N_kv and HEAD_DIM
    // When V_FROM_K is true, V is read from K's memory (first V_HEAD_DIM columns per row)
    const ptrdiff_t v_offset = V_FROM_K ? k_offset : (ptrdiff_t)(kv_head_idx * N_kv) * V_HEAD_DIM;
    const ptrdiff_t o_offset = (ptrdiff_t)(head_idx * N + row0) * V_HEAD_DIM;  // O uses V_HEAD_DIM
    
    // For V_FROM_K mode, V data is interleaved in K's layout with HEAD_DIM stride
    constexpr ptrdiff_t V_ROW_STRIDE = V_FROM_K ? HEAD_DIM : V_HEAD_DIM;

    // Bfloat16 shared memory for XMX tiles
    xmx_bfloat16 * shQ = reinterpret_cast<xmx_bfloat16*>(shmem);
    xmx_bfloat16 * shK = shQ + BLOCK_M * Q_STRIDE;
    xmx_bfloat16 * shV = shK + BLOCK_N * K_STRIDE;
    xmx_bfloat16 * shP = shV + BLOCK_N * V_STRIDE;
    xmx_bfloat16 * shVT = shP + BLOCK_M * P_STRIDE;  // V^T buffer (separate from shK to avoid corruption)

    // Float shared memory for softmax scores and output accumulator
    // shVT is [PADDED_V_HEAD_DIM x BLOCK_N] col-major, so advance by PADDED_V_HEAD_DIM * V_T_STRIDE
    float * shS = reinterpret_cast<float*>(shVT + PADDED_V_HEAD_DIM * V_T_STRIDE);
    float * rowMax = shS + BLOCK_M * S_STRIDE;
    float * rowSum = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;
    float * shAcc = rowAlpha + BLOCK_M;  // Sized for PADDED_V_HEAD_DIM

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
        const int r = i / PADDED_HEAD_DIM;
        const int c = i % PADDED_HEAD_DIM;
        const int q_row = row0 + r;

        if (q_row < N && c < HEAD_DIM) {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(Q[q_offset + (ptrdiff_t)r * HEAD_DIM + c] * scale);
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
            const int r = i / PADDED_HEAD_DIM;
            const int c = i % PADDED_HEAD_DIM;
            const int k_row = col0 + r;

            xmx_bfloat16 k_val = xmx_bfloat16(0.0f);

            if (k_row < N_kv && c < HEAD_DIM) {  // Use N_kv for bounds check
                const ptrdiff_t base = (ptrdiff_t)k_row * HEAD_DIM;
                k_val = xmx_bfloat16(K[k_offset + base + c]);
            }

            // Store K as K^T col-major: K^T[head_dim, kv_col] at shK[head_dim + kv_col * K_STRIDE]
            shK[c + r * K_STRIDE] = k_val;
        }

        // Load V tiles separately with V_HEAD_DIM and PADDED_V_HEAD_DIM
        // When V_FROM_K is true, read V from K's memory with HEAD_DIM stride (zero-copy MLA)
        for (int i = lid; i < BLOCK_N * PADDED_V_HEAD_DIM; i += THREADS) {
            const int r = i / PADDED_V_HEAD_DIM;
            const int c = i % PADDED_V_HEAD_DIM;
            const int k_row = col0 + r;

            xmx_bfloat16 v_val = xmx_bfloat16(0.0f);

            if (k_row < N_kv && c < V_HEAD_DIM) {  // Use N_kv for bounds check
                // V_ROW_STRIDE is HEAD_DIM when V_FROM_K, else V_HEAD_DIM
                const ptrdiff_t base = (ptrdiff_t)k_row * V_ROW_STRIDE;
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

                auto mq_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shQ[(sg_id * TM) * Q_STRIDE + k]);
                // K^T is stored col-major, tile [k:k+TK, j:j+TN] starts at shK[k + j * K_STRIDE]
                auto mk_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shK[k + j * K_STRIDE]);

                joint_matrix_load(sg, mq, mq_ptr, Q_STRIDE);
                joint_matrix_load(sg, mk, mk_ptr, K_STRIDE);

                joint_matrix_mad(sg, matS, mq, mk, matS);
            }

            auto shS_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shS[(sg_id * TM) * S_STRIDE + j]);
            joint_matrix_store(sg, matS, shS_ptr, S_STRIDE, layout::row_major);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Online Softmax ============
        if (lid < BLOCK_M) {
            const int row = lid;
            const int q_row = row0 + row;  // Global query row index
            float m = -1.0e20f;

            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val = shS[row * S_STRIDE + c];
                const int kv_col = col0 + c;  // Global KV column index

                // Apply mask from precomputed mask tensor
                if (mask != nullptr && kv_col < N_kv && q_row < N) {
                    const float mask_val = mask[q_row * mask_stride + kv_col];
                    s_val += mask_val;  // mask is 0.0 for unmasked, -inf for masked
                }

                shS[row * S_STRIDE + c] = s_val;
                m = sycl::fmax(m, s_val);
            }

            float m_prev = rowMax[row];
            float m_new = sycl::fmax(m_prev, m);
            float alpha = sycl::exp(sycl::fmax(m_prev - m_new, -20.0f));
            rowAlpha[row] = alpha;
            rowMax[row] = m_new;

            float s_sum = 0.0f;
            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val = shS[row * S_STRIDE + c];
                float val = sycl::exp(sycl::fmax(s_val - m_new, -20.0f));
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
            const int row = i / BLOCK_N;
            const int col = i % BLOCK_N;
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
                auto shP_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shP[(sg_id * TM) * P_STRIDE + k]);
                joint_matrix_load(sg, mp, shP_ptr, P_STRIDE);

                // V tile: TK x TN from shVT (col-major, V[BLOCK_N rows, PADDED_V_HEAD_DIM cols])
                // V[k:k+TK, out_tile*TN:out_tile*TN+TN] at shVT[k + out_tile*TN * V_T_STRIDE]
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, TK, TN, layout::col_major> mv;
                auto shV_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shVT[k + out_tile * TN * V_T_STRIDE]);
                joint_matrix_load(sg, mv, shV_ptr, V_T_STRIDE);

                joint_matrix_mad(sg, matPV, mp, mv, matPV);
            }

            // Add PV contribution to output accumulator using scratch region
            const int scratch_offset = sg_id * TM * TN;
            auto scratch_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shS[scratch_offset]);
            joint_matrix_store(sg, matPV, scratch_ptr, TN, layout::row_major);
            it.barrier(sycl::access::fence_space::local_space);

            // Add to output accumulator - output column is in head-dim, not sequence
            const int sg_lane = sg.get_local_linear_id();
            const int sg_size = sg.get_local_linear_range();
            for (int idx = sg_lane; idx < TM * TN; idx += sg_size) {
                const int local_row = idx / TN;
                const int local_col = idx % TN;
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
        const int row = i / V_HEAD_DIM;
        const int col = i % V_HEAD_DIM;
        const int q_row = row0 + row;

        if (q_row < N) {
            float s = rowSum[row];
            // shAcc is indexed with PADDED_V_HEAD_DIM stride
            float acc_val = shAcc[row * PADDED_V_HEAD_DIM + col];
            float normalized = acc_val / (s > 1e-10f ? s : 1.0f);
            O[o_offset + (ptrdiff_t)row * o_row_stride + col] = normalized;
        }
    }
}

// ============================================================================
// Wrapper for DG2 (nsize=8) with padding support
// ============================================================================
// V_FROM_K: when true, V is read from K's memory (MLA zero-copy optimization)
template <int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM, bool V_FROM_K = false>
inline void flash_attn_coopmat_kernel_n8_padded(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, const int o_row_stride, float * shmem
) {
    flash_attn_coopmat_kernel_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, o_row_stride, shmem
    );
}

// ============================================================================
// Wrapper for PVC/B60 (nsize=16) with padding support
// ============================================================================
// V_FROM_K: when true, V is read from K's memory (MLA zero-copy optimization)
template <int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM, bool V_FROM_K = false>
inline void flash_attn_coopmat_kernel_n16_padded(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, const int o_row_stride, float * shmem
) {
    flash_attn_coopmat_kernel_padded<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, o_row_stride, shmem
    );
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

template <int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM, bool V_FROM_K = false, int TM = 8, int TN = 16, int TK = 16>
inline void flash_attn_coopmat_kernel_small_tile(
    sycl::nd_item<2> it,
    const float * Q,
    const float * K,
    const float * V,  // When V_FROM_K=true, this is ignored and V is read from K
    float * O,
    float * l_d,
    float * m_d,
    const int64_t N,
    const int64_t N_kv,
    const int n_heads,
    const int n_kv_heads,
    const int gqa_ratio,
    const float scale,
    const float * mask,
    const int64_t mask_stride,
    const int o_row_stride,
    float * shmem
) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    // Small tiles for large head dimensions (fits 128KB SLM)
    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;
    constexpr int THREADS = 32;  // 2 subgroups instead of 4

    // Number of subgroups covering the M dimension
    constexpr int NUM_SG_M = BLOCK_M / TM;

    // Stride calculations with +8 padding for bank conflict avoidance
    constexpr int Q_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE = PADDED_HEAD_DIM + 8;
    // No V_STRIDE - we eliminate shV and load directly to shVT
    constexpr int V_T_STRIDE = BLOCK_N + 8;  // Col-major stride for shVT
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;

    // Ensure tile dimensions are compatible with XMX
    static_assert(BLOCK_N % TK == 0, "BLOCK_N must be divisible by TK for P@V matmul");
    static_assert(PADDED_HEAD_DIM % TN == 0, "PADDED_HEAD_DIM must be divisible by TN for output tiles");
    static_assert(PADDED_V_HEAD_DIM % TN == 0, "PADDED_V_HEAD_DIM must be divisible by TN for output tiles");

    (void)l_d; (void)m_d; (void)n_heads; (void)n_kv_heads;

    const int lid = it.get_local_id(0);
    const int gid_x = it.get_group(0);
    const int gid_y = it.get_group(1);
    auto sg = it.get_sub_group();
    
    const int sg_id = sg.get_group_linear_id();
    
    if (sg_id >= NUM_SG_M) return;

    const int row0 = gid_x * BLOCK_M;
    if (row0 >= N) return;

    const int head_idx = gid_y;
    const int kv_head_idx = head_idx / gqa_ratio;

    const ptrdiff_t q_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t)(kv_head_idx * N_kv) * HEAD_DIM;
    const ptrdiff_t v_offset = V_FROM_K ? k_offset : (ptrdiff_t)(kv_head_idx * N_kv) * V_HEAD_DIM;
    const ptrdiff_t o_offset = (ptrdiff_t)(head_idx * N + row0) * V_HEAD_DIM;
    
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

    xmx_bfloat16 * shQ = reinterpret_cast<xmx_bfloat16*>(shmem);
    xmx_bfloat16 * shK = shQ + BLOCK_M * Q_STRIDE;
    xmx_bfloat16 * shP = shK + BLOCK_N * K_STRIDE;
    xmx_bfloat16 * shVT = shP + BLOCK_M * P_STRIDE;  // Directly after shP (no shV)

    float * shS = reinterpret_cast<float*>(shVT + PADDED_V_HEAD_DIM * V_T_STRIDE);
    float * rowMax = shS + BLOCK_M * S_STRIDE;
    float * rowSum = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;
    float * shAcc = rowAlpha + BLOCK_M;

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
        const int r = i / PADDED_HEAD_DIM;
        const int c = i % PADDED_HEAD_DIM;
        const int q_row = row0 + r;

        if (q_row < N && c < HEAD_DIM) {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(Q[q_offset + (ptrdiff_t)r * HEAD_DIM + c] * scale);
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
            const int r = i / PADDED_HEAD_DIM;
            const int c = i % PADDED_HEAD_DIM;
            const int k_row = col0 + r;

            xmx_bfloat16 k_val = xmx_bfloat16(0.0f);
            if (k_row < N_kv && c < HEAD_DIM) {
                const ptrdiff_t base = (ptrdiff_t)k_row * HEAD_DIM;
                k_val = xmx_bfloat16(K[k_offset + base + c]);
            }
            // Store K^T col-major
            shK[c + r * K_STRIDE] = k_val;
        }

        // Load V tiles DIRECTLY into shVT (transposed) - eliminates shV buffer
        // shVT is col-major: [BLOCK_N rows, PADDED_V_HEAD_DIM cols]
        // Index: shVT[row + col * V_T_STRIDE]
        for (int i = lid; i < BLOCK_N * PADDED_V_HEAD_DIM; i += THREADS) {
            const int r = i / PADDED_V_HEAD_DIM;  // kv position (0..BLOCK_N-1)
            const int c = i % PADDED_V_HEAD_DIM;  // head dim (0..PADDED_V_HEAD_DIM-1)
            const int k_row = col0 + r;

            xmx_bfloat16 v_val = xmx_bfloat16(0.0f);
            if (k_row < N_kv && c < V_HEAD_DIM) {
                const ptrdiff_t base = (ptrdiff_t)k_row * V_ROW_STRIDE;
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

                auto mq_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shQ[(sg_id * TM) * Q_STRIDE + k]);
                auto mk_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shK[k + j * K_STRIDE]);

                joint_matrix_load(sg, mq, mq_ptr, Q_STRIDE);
                joint_matrix_load(sg, mk, mk_ptr, K_STRIDE);
                joint_matrix_mad(sg, matS, mq, mk, matS);
            }

            auto shS_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shS[(sg_id * TM) * S_STRIDE + j]);
            joint_matrix_store(sg, matS, shS_ptr, S_STRIDE, layout::row_major);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Online Softmax ============
        if (lid < BLOCK_M) {
            const int row = lid;
            const int q_row = row0 + row;
            float m = -1.0e20f;

            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val = shS[row * S_STRIDE + c];
                const int kv_col = col0 + c;

                if (mask != nullptr && kv_col < N_kv && q_row < N) {
                    s_val += mask[q_row * mask_stride + kv_col];
                }
                shS[row * S_STRIDE + c] = s_val;
                m = sycl::fmax(m, s_val);
            }

            float m_prev = rowMax[row];
            float m_new = sycl::fmax(m_prev, m);
            float alpha = sycl::exp(sycl::fmax(m_prev - m_new, -20.0f));
            rowAlpha[row] = alpha;
            rowMax[row] = m_new;

            float s_sum = 0.0f;
            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val = shS[row * S_STRIDE + c];
                float val = sycl::exp(sycl::fmax(s_val - m_new, -20.0f));
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
            const int row = i / BLOCK_N;
            const int col = i % BLOCK_N;
            shP[row * P_STRIDE + col] = xmx_bfloat16(shS[row * S_STRIDE + col]);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ P @ V computation ============
        constexpr int NUM_OUT_TILES = PADDED_V_HEAD_DIM / TN;
        
        for (int out_tile = 0; out_tile < NUM_OUT_TILES; ++out_tile) {
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN, layout::dynamic> matPV;
            joint_matrix_fill(sg, matPV, 0.0f);

            #pragma unroll 2
            for (int k = 0; k < BLOCK_N; k += TK) {
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::a, TM, TK, layout::row_major> mp;
                auto shP_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shP[(sg_id * TM) * P_STRIDE + k]);
                joint_matrix_load(sg, mp, shP_ptr, P_STRIDE);

                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, TK, TN, layout::col_major> mv;
                auto shV_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shVT[k + out_tile * TN * V_T_STRIDE]);
                joint_matrix_load(sg, mv, shV_ptr, V_T_STRIDE);

                joint_matrix_mad(sg, matPV, mp, mv, matPV);
            }

            // Store to scratch and accumulate
            const int scratch_offset = sg_id * TM * TN;
            auto scratch_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shS[scratch_offset]);
            joint_matrix_store(sg, matPV, scratch_ptr, TN, layout::row_major);
            it.barrier(sycl::access::fence_space::local_space);

            const int sg_lane = sg.get_local_linear_id();
            const int sg_size = sg.get_local_linear_range();
            for (int idx = sg_lane; idx < TM * TN; idx += sg_size) {
                const int local_row = idx / TN;
                const int local_col = idx % TN;
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
        const int row = i / V_HEAD_DIM;
        const int col = i % V_HEAD_DIM;
        const int q_row = row0 + row;

        if (q_row < N) {
            float s = rowSum[row];
            float acc_val = shAcc[row * PADDED_V_HEAD_DIM + col];
            float normalized = acc_val / (s > 1e-10f ? s : 1.0f);
            O[o_offset + (ptrdiff_t)row * o_row_stride + col] = normalized;
        }
    }
}

// ============================================================================
// Wrapper for small-tile kernel (DG2/Arc nsize=8 compatible)
// ============================================================================
template <int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM, bool V_FROM_K = false>
inline void flash_attn_coopmat_kernel_small_tile_n8(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, const int o_row_stride, float * shmem
) {
    flash_attn_coopmat_kernel_small_tile<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, o_row_stride, shmem
    );
}

// ============================================================================
// Wrapper for small-tile kernel (PVC/B60 nsize=16 compatible)
// ============================================================================
template <int64_t HEAD_DIM, int64_t V_HEAD_DIM, int64_t PADDED_HEAD_DIM, int64_t PADDED_V_HEAD_DIM, bool V_FROM_K = false>
inline void flash_attn_coopmat_kernel_small_tile_n16(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, const int o_row_stride, float * shmem
) {
    flash_attn_coopmat_kernel_small_tile<HEAD_DIM, V_HEAD_DIM, PADDED_HEAD_DIM, PADDED_V_HEAD_DIM, V_FROM_K, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, o_row_stride, shmem
    );
}

#endif // SYCL_EXT_COOPERATIVE_MATRICES
#endif // GGML_SYCL_FATTN_XMX_HPP
