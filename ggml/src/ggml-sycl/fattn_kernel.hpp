#ifndef GGML_SYCL_FATTN_KERNEL_HPP
#define GGML_SYCL_FATTN_KERNEL_HPP

#include <sycl/sycl.hpp>

template <int64_t QKD>
inline void flash_attn_mul_mat_QK_kernel(
    sycl::nd_item<2> it,
    const float * Q,  ptrdiff_t q_row_stride,
    const float * K,  ptrdiff_t k_row_stride,
    float * S,        ptrdiff_t s_row_stride,
    const int Br, const int Bc) {

    const int i = it.get_local_id(0);
    if (i >= Br) {
        return;
    }

    const float * q_vec = Q + i * q_row_stride;
    float * s_row       = S + i * s_row_stride;

    for (int j = 0; j < Bc; ++j) {
        const float * k_vec = K + j * k_row_stride;
        float score = 0.0f;

#pragma unroll
        for (int k = 0; k < QKD; ++k) {
            score += q_vec[k] * k_vec[k];
        }

        s_row[j] = score;
    }
}


inline void flash_attn_softmax_kernel(
    sycl::nd_item<2> it,
    float * S, float * P,
    float * m_local, float * l_local,
    const int Br, const int Bc,
    float * l_d, float * m_d,
    const int row_offset,  // Global row offset for this block
    const int window_size  // Sliding window size (0 = no window limit)
) {
    (void)row_offset; // suppress unused parameter warning

    const int li  = it.get_local_id(0);
    const int gi  = it.get_group(0);
    const int gj  = it.get_group(1);  // Block column index
    const int row = gi * Br + li;
    const int col_start = gj * Bc;  // Starting column for this block

    if (li >= Br) {
        return;
    }

    const int local_row_offset = li * Bc;

    float m_old = m_d[row];
    float l_old = l_d[row];

    // Block max (with causal masking and sliding window check)
    const float NEG_INF = -1.0e20f;
    float m_block = NEG_INF;
    for (int j = 0; j < Bc; ++j) {
        const int col = col_start + j;

        // Check if this position should be masked
        bool is_masked = false;

        // Causal mask: col > row means attending to future positions
        if (col > row) {
            is_masked = true;
        }

        // Sliding window: only attend to positions within [row - window + 1, row]
        // (or symmetric bidirectional window if window_size < 0)
        if (window_size > 0 && !is_masked) {
            if (col < row - window_size + 1) {
                is_masked = true;
            }
        }

        const float s_ij = is_masked ? -1.0e20f : S[local_row_offset + j];
        S[local_row_offset + j] = s_ij;  // Store masked value
        m_block = sycl::fmax(m_block, s_ij);
    }

    // Block exp-sum
    float l_block = 0.0f;
    for (int j = 0; j < Bc; ++j) {
        const float s_ij = S[local_row_offset + j];
        const float e = sycl::isfinite(s_ij) ? sycl::exp(s_ij - m_block) : 0.0f;
        P[local_row_offset + j] = e;  // temporary store
        l_block += e;
    }

    // Merge block stats with global (streaming softmax)
    float m_new;
    float l_new;

    if (l_old == 0.0f && m_old == -1.0e20f) {
        // first block for this row
        m_new = m_block;
        l_new = l_block;
    } else {
        m_new = sycl::fmax(m_old, m_block);

        const float alpha = sycl::exp(m_old - m_new);
        const float beta = sycl::exp(m_block - m_new);

        l_new = alpha * l_old + beta * l_block;
    }

    // Store updated global stats
    m_d[row] = m_new;
    l_d[row] = l_new;

    // Convert local e_ij to global probabilities p_ij
    float scale_block = 0.0f;
    if (l_new > 0.0f) {
        scale_block = sycl::exp(m_block - m_new) / l_new;
    }

    for (int j = 0; j < Bc; ++j) {
        P[local_row_offset + j] *= scale_block;
    }

    // Optional: keep local copies
    m_local[li] = m_new;
    l_local[li] = l_new;
}




template <int64_t VD>
inline void flash_attn_mul_mat_PV_kernel(
    sycl::nd_item<2> it,
    const float * P, ptrdiff_t p_row_stride,
    const float * V, ptrdiff_t v_row_stride,
    float * O,       ptrdiff_t o_row_stride,
    const int Br, const int Bc) {

    const int i = it.get_local_id(0);
    if (i >= Br) {
        return;
    }

    const float * p_row = P + i * p_row_stride;
    float * o_row       = O + i * o_row_stride;

    for (int j = 0; j < VD; ++j) {
        float acc = 0.0f;

#pragma unroll
        for (int k = 0; k < Bc; ++k) {
            const float * v_row = V + k * v_row_stride;
            acc += p_row[k] * v_row[j];
        }

        o_row[j] = acc;
    }
}

// FP16 to F32 dequantization kernel
// Converts half precision tensors to float for flash attention compute
inline void flash_attn_dequantize_fp16_kernel(
    sycl::nd_item<1> it,
    const sycl::half * src,     // FP16 input
    float * dst,                // F32 output
    const int64_t n_elements,   // Total elements
    const int64_t row_stride    // Stride between rows (in elements)
) {
    const int idx = it.get_global_id(0);
    if (idx >= n_elements) {
        return;
    }

    const int64_t row = idx / row_stride;
    const int64_t col = idx % row_stride;

    dst[idx] = static_cast<float>(src[row * row_stride + col]);
}

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace cm = sycl::ext::oneapi::experimental::matrix;

// XMX bfloat16 type for cooperative matrices (distinct from common.hpp bfloat16 = uint16_t)
using xmx_bfloat16 = sycl::ext::oneapi::bfloat16;

// Tile size enum for different Intel GPU architectures
// DG2/Arc (Xe2) supports 8x8x16, PVC supports 8x16x16, AMX supports 16x16x32
enum class xmx_tile_kind { tile_8x8, tile_16x16 };

// Check if device supports cooperative matrices
inline bool ggml_sycl_has_coopmat_support(sycl::device device) {
    return device.has(sycl::aspect::ext_intel_gpu_eu_simd_width) &&
           device.has(sycl::aspect::ext_intel_matrix);
}

// Determine tile kind based on device architecture
// Arc B60/Battlemage (Xe2) actually reports nsize=16 in matrix_combinations,
// meaning it uses 8x16x16 tiles like PVC, not 8x8x16 like DG2
inline xmx_tile_kind ggml_sycl_get_tile_kind(sycl::device device) {
    // Query actual matrix combinations to determine tile kind
    auto combinations = device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    for (const auto& c : combinations) {
        // Look for bf16 or fp16 combinations (atype 0 or 1)
        if (c.atype == sycl::ext::oneapi::experimental::matrix::matrix_type::bf16 ||
            c.atype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp16) {
            if (c.nsize == 16 || c.max_nsize == 16) {
                // PVC-like: 8x16x16 tiles
                return xmx_tile_kind::tile_16x16;  // Use 16 for the N dimension
            } else if (c.nsize == 8 || c.max_nsize == 8) {
                // DG2: 8x8x16 tiles
                return xmx_tile_kind::tile_8x8;
            }
        }
    }
    // Default to 16x16 if we can't determine
    return xmx_tile_kind::tile_16x16;
}

// Debug mode for flash attention kernel
// Set GGML_SYCL_FLASH_ATTN_DEBUG=1 to enable debug output
inline bool ggml_sycl_fattn_debug() {
    static bool debug = false;
    static bool checked = false;
    if (!checked) {
        const char* env = getenv("GGML_SYCL_FLASH_ATTN_DEBUG");
        debug = (env != nullptr && strcmp(env, "1") == 0);
        checked = true;
    }
    return debug;
}

// Intel XMX Flash Attention using cooperative matrices
// Uses xmx_bfloat16 for XMX A/B matrices and float for accumulators
// Type conversion happens at shared memory boundaries
// 
// Template parameters:
// - HEAD_DIM: head dimension (32, 64, 96, 128, 256)
// - TM, TN, TK: joint matrix tile sizes (8x8x16 for DG2/Arc, 16x16x16 for PVC)
// - DEBUG: enable debug output
template <int64_t HEAD_DIM, int TM = 8, int TN = 8, int TK = 16, bool DEBUG = false>
inline void flash_attn_coopmat_kernel(
    sycl::nd_item<2> it,
    const float * Q,
    const float * K,
    const float * V,
    float * O,
    float * l_d,
    float * m_d,
    const int64_t N,       // Query sequence length
    const int64_t N_kv,    // Key/Value sequence length (can differ from N for KV cache)
    const int n_heads,
    const int n_kv_heads,
    const int gqa_ratio,
    const float scale,
    const float * mask,    // Precomputed mask tensor [N_kv, N] with 0.0 or -inf
    const int64_t mask_stride,  // Stride between rows in mask (typically N_kv)
    float * shmem
) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int THREADS = 64;

    // Number of subgroups covering the M dimension
    constexpr int NUM_SG_M = BLOCK_M / TM;

    constexpr int Q_STRIDE = HEAD_DIM + 8;  // Padding for bank conflict avoidance
    constexpr int K_STRIDE = HEAD_DIM + 8;
    constexpr int V_STRIDE = HEAD_DIM + 8;
    // V is stored col-major in shVT as [BLOCK_N rows, HEAD_DIM cols] for P@V
    // For col-major, stride (leading dimension) must be >= number of rows = BLOCK_N
    constexpr int V_T_STRIDE = BLOCK_N + 8;  // Must be >= BLOCK_N for col-major storage
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;   // P matrix stride (for bf16 P)

    // Ensure BLOCK_N is divisible by TK for P@V computation
    static_assert(BLOCK_N % TK == 0, "BLOCK_N must be divisible by TK for P@V matmul");
    static_assert(HEAD_DIM % TN == 0, "HEAD_DIM must be divisible by TN for output tiles");

    // Shared memory layout (all sizes in floats/bf16 as noted):
    // [bf16: Q (BLOCK_M * Q_STRIDE)]
    // [bf16: K (BLOCK_N * K_STRIDE)]
    // [bf16: V (BLOCK_N * V_STRIDE)]
    // [bf16: P (BLOCK_M * P_STRIDE)] - for storing bf16 attention probs
    // [bf16: VT (HEAD_DIM * BLOCK_N)] - V^T buffer (separate from K to avoid corruption)
    // [float: S (BLOCK_M * S_STRIDE)] - attention scores
    // [float: rowMax (BLOCK_M)]
    // [float: rowSum (BLOCK_M)]
    // [float: rowAlpha (BLOCK_M)]
    // [float: shAcc (BLOCK_M * HEAD_DIM)] - output accumulator

    (void)l_d; // suppress unused parameter warning
    (void)m_d; // suppress unused parameter warning
    (void)n_heads; // suppress unused parameter warning
    (void)n_kv_heads; // suppress unused parameter warning

    const int lid = it.get_local_id(0);
    const int gid_x = it.get_group(0);
    const int gid_y = it.get_group(1);
    auto sg = it.get_sub_group();
    
    // Get actual subgroup ID - critical for correct tiling
    const int sg_id = sg.get_group_linear_id();
    
    // Each subgroup handles TM rows; guard if we have more subgroups than needed
    if (sg_id >= NUM_SG_M) return;

    const int row0 = gid_x * BLOCK_M;
    if (row0 >= N) return;

    const int head_idx = gid_y;
    const int kv_head_idx = head_idx / gqa_ratio;

    const ptrdiff_t q_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t)(kv_head_idx * N_kv) * HEAD_DIM;  // K uses N_kv
    const ptrdiff_t v_offset = (ptrdiff_t)(kv_head_idx * N_kv) * HEAD_DIM;  // V uses N_kv
    const ptrdiff_t o_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;

    // Bfloat16 shared memory for XMX tiles
    xmx_bfloat16 * shQ = reinterpret_cast<xmx_bfloat16*>(shmem);
    xmx_bfloat16 * shK = shQ + BLOCK_M * Q_STRIDE;
    xmx_bfloat16 * shV = shK + BLOCK_N * K_STRIDE;
    xmx_bfloat16 * shP = shV + BLOCK_N * V_STRIDE;
    xmx_bfloat16 * shVT = shP + BLOCK_M * P_STRIDE;  // V^T buffer (separate from K)

    // Float shared memory for softmax scores and output accumulator
    float * shS = reinterpret_cast<float*>(shVT + HEAD_DIM * V_T_STRIDE);
    float * rowMax = shS + BLOCK_M * S_STRIDE;
    float * rowSum = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;
    float * shAcc = rowAlpha + BLOCK_M;  // Output accumulator [BLOCK_M x HEAD_DIM]

    // Initialize per-row stats and output accumulator
    if (lid < BLOCK_M) {
        rowMax[lid] = -1.0e20f;
        rowSum[lid] = 0.0f;
    }
    for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        shAcc[i] = 0.0f;
    }

    // Load Q tiles with xmx_bfloat16 conversion and scale
    for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        const int r = i / HEAD_DIM;
        const int c = i % HEAD_DIM;
        const int q_row = row0 + r;

        if (q_row < N) {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(Q[q_offset + (ptrdiff_t)r * HEAD_DIM + c] * scale);
        } else {
            shQ[r * Q_STRIDE + c] = xmx_bfloat16(0.0f);
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    const int num_kv_blocks = (N_kv + BLOCK_N - 1) / BLOCK_N;  // Use N_kv for K/V iteration

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int col0 = kv_block * BLOCK_N;

        // Note: We iterate over ALL KV blocks since the mask tensor handles causal/SWA masking.
        // The mask will apply -inf to masked positions during softmax.

        // Load K and V tiles with xmx_bfloat16 conversion
        // K is stored as K^T in col-major: K^T[head_dim, kv] stored at shK[head_dim + kv * K_STRIDE]
        // V is stored row-major: shV[kv_row, head_dim] for later V^T transpose
        for (int i = lid; i < BLOCK_N * HEAD_DIM; i += THREADS) {
            const int r = i / HEAD_DIM;  // r = kv row index within block (0..BLOCK_N-1)
            const int c = i % HEAD_DIM;  // c = head dim index (0..HEAD_DIM-1)
            const int k_row = col0 + r;

            xmx_bfloat16 k_val = xmx_bfloat16(0.0f);
            xmx_bfloat16 v_val = xmx_bfloat16(0.0f);

            if (k_row < N_kv) {  // Use N_kv for K/V bounds
                const ptrdiff_t base = (ptrdiff_t)k_row * HEAD_DIM;
                k_val = xmx_bfloat16(K[k_offset + base + c]);
                v_val = xmx_bfloat16(V[v_offset + base + c]);
            }

            // Store K as K^T col-major: K^T[head_dim, kv_col] at shK[head_dim + kv_col * K_STRIDE]
            // K^T[c, r] = K[r, c] = K[kv, head_dim]
            shK[c + r * K_STRIDE] = k_val;
            // Store V row-major as before
            shV[r * V_STRIDE + c] = v_val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Q @ K^T computation ============
        // Each subgroup computes TM rows of the BLOCK_M x BLOCK_N score matrix
        for (int j = 0; j < BLOCK_N; j += TN) {
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN, layout::dynamic> matS;
            joint_matrix_fill(sg, matS, 0.0f);

            for (int k = 0; k < HEAD_DIM; k += TK) {
                // Q is TM x TK (row-major), K^T is TK x TN (col-major)
                // For K^T, we load K columns as rows using col_major layout
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

            // Store score tile to shared memory
            auto shS_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shS[(sg_id * TM) * S_STRIDE + j]);
            joint_matrix_store(sg, matS, shS_ptr, S_STRIDE, layout::row_major);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Transpose V for P@V computation ============
        // V is currently stored as [BLOCK_N x HEAD_DIM] row-major in shV
        // We store V directly (not transposed) in shVT for P@V: V[kv, head_dim]
        // shVT is indexed as: V[kv_row, head_col] at shVT[kv_row + head_col * V_T_STRIDE]
        // This is col-major storage with V_T_STRIDE >= BLOCK_N (but we use V_T_STRIDE = HEAD_DIM+8)
        // Actually for the P@V matmul, B needs to be [BLOCK_N rows, HEAD_DIM cols]
        // Stored col-major with stride V_T_STRIDE
        for (int i = lid; i < BLOCK_N * HEAD_DIM; i += THREADS) {
            const int row = i / HEAD_DIM;  // kv position 0 to BLOCK_N-1
            const int col = i % HEAD_DIM;  // head_dim 0 to HEAD_DIM-1
            // Store V[row, col] in col-major format: V[row, col] at shVT[row + col * V_T_STRIDE]
            shVT[row + col * V_T_STRIDE] = shV[row * V_STRIDE + col];
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ Online Softmax ============
        // Each thread handles one row of the score matrix
        if (lid < BLOCK_M) {
            const int row = lid;
            const int q_row = row0 + row;  // Global query row index
            float m = -1.0e20f;

            // Find max with masking from precomputed mask tensor
            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val = shS[row * S_STRIDE + c];
                const int kv_col = col0 + c;  // Global KV column index

                // Apply mask from precomputed mask tensor
                // mask layout is [N_kv, N] - mask[q_row * mask_stride + kv_col]
                if (mask != nullptr && kv_col < N_kv && q_row < N) {
                    const float mask_val = mask[q_row * mask_stride + kv_col];
                    s_val += mask_val;  // mask is 0.0 for unmasked, -inf for masked
                }

                shS[row * S_STRIDE + c] = s_val;
                m = sycl::fmax(m, s_val);
            }

            // Online softmax update
            float m_prev = rowMax[row];
            float m_new = sycl::fmax(m_prev, m);
            float alpha = sycl::exp(sycl::fmax(m_prev - m_new, -20.0f));
            rowAlpha[row] = alpha;
            rowMax[row] = m_new;

            // Compute exp and sum
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

        // ============ Scale previous output by alpha ============
        for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
            const int row = i / HEAD_DIM;
            shAcc[i] *= rowAlpha[row];
        }

        // ============ Convert P to xmx_bfloat16 for XMX ============
        for (int i = lid; i < BLOCK_M * BLOCK_N; i += THREADS) {
            const int row = i / BLOCK_N;
            const int col = i % BLOCK_N;
            shP[row * P_STRIDE + col] = xmx_bfloat16(shS[row * S_STRIDE + col]);
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ P @ V computation ============
        // For DG2/Arc: use::a requires TM x TK (8x16), use::b requires TK x TN (16x8)
        // P is [BLOCK_M x BLOCK_N], V is [BLOCK_N x HEAD_DIM]
        // Output is [BLOCK_M x HEAD_DIM], computed as tiles of [TM x TN]
        // Inner loop iterates over BLOCK_N in chunks of TK (16)
        //
        // Each subgroup computes TM rows of output, for each HEAD_DIM/TN output tiles
        constexpr int NUM_OUT_TILES = HEAD_DIM / TN;
        
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

                // V tile: TK x TN from shVT (col-major, V[BLOCK_N rows, HEAD_DIM cols])
                // V[k:k+TK, out_tile*TN:out_tile*TN+TN] at shVT[k + out_tile*TN * V_T_STRIDE]
                joint_matrix<sycl::sub_group, xmx_bfloat16, use::b, TK, TN, layout::col_major> mv;
                auto shV_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shVT[k + out_tile * TN * V_T_STRIDE]);
                joint_matrix_load(sg, mv, shV_ptr, V_T_STRIDE);

                joint_matrix_mad(sg, matPV, mp, mv, matPV);
            }

            // Add PV contribution to output accumulator - use scratch region in shS
            // Each subgroup uses its own scratch to avoid race conditions
            const int scratch_offset = sg_id * TM * TN;
            auto scratch_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shS[scratch_offset]);
            joint_matrix_store(sg, matPV, scratch_ptr, TN, layout::row_major);
            it.barrier(sycl::access::fence_space::local_space);

            // Add to output accumulator
            // matPV is TM x TN, stored row-major with stride TN at subgroup's scratch region
            const int sg_lane = sg.get_local_linear_id();
            const int sg_size = sg.get_local_linear_range();
            for (int idx = sg_lane; idx < TM * TN; idx += sg_size) {
                const int local_row = idx / TN;
                const int local_col = idx % TN;
                const int global_row = sg_id * TM + local_row;
                const int global_col = out_tile * TN + local_col;
                if (global_row < BLOCK_M && global_col < HEAD_DIM) {
                    shAcc[global_row * HEAD_DIM + global_col] += shS[scratch_offset + idx];
                }
            }

            it.barrier(sycl::access::fence_space::local_space);
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    // ============ Final normalize and store ============
    for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        const int q_row = row0 + row;

        if (q_row < N) {
            float s = rowSum[row];
            float acc = shAcc[i];
            float normalized = acc / (s > 1e-10f ? s : 1.0f);
            
            // Debug output for first few elements
            if (DEBUG && gid_x == 0 && gid_y == 0 && q_row < 2 && col < 4) {
                sycl::float2 acc_pair = sycl::float2(acc, normalized);
                // Note: printf in SYCL kernels may not work on all platforms
                // This is a placeholder for debugging
            }
            
            O[o_offset + (ptrdiff_t)row * HEAD_DIM + col] = normalized;
        }
    }
}

// Wrapper for Arc B60/Battlemage (8x16x16 tiles - msize=8, nsize=16, ksize=16)
// Based on runtime query showing nsize=16 for bf16/fp16 on Arc B60
template <int64_t HEAD_DIM>
inline void flash_attn_coopmat_kernel_dg2(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, float * shmem
) {
    // Arc B60 uses 8x16x16 tiles (TM=8, TN=16, TK=16)
    flash_attn_coopmat_kernel<HEAD_DIM, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, shmem
    );
}

// Wrapper for PVC (8x16x16 tiles - same as Arc B60)
template <int64_t HEAD_DIM>
inline void flash_attn_coopmat_kernel_pvc(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, float * shmem
) {
    // PVC also uses 8x16x16 tiles
    flash_attn_coopmat_kernel<HEAD_DIM, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, shmem
    );
}

// Padded flash attention kernel for head sizes that don't evenly divide tile dimensions
// Uses padding to the next supported size (e.g., 40 -> 64)
// Template parameters:
// - HEAD_DIM: actual head dimension (e.g., 40)
// - PADDED_HEAD_DIM: padded head dimension for XMX (e.g., 64)
// - TM, TN, TK: joint matrix tile sizes
template <int64_t HEAD_DIM, int64_t PADDED_HEAD_DIM, int TM = 8, int TN = 16, int TK = 16>
inline void flash_attn_coopmat_kernel_padded(
    sycl::nd_item<2> it,
    const float * Q,
    const float * K,
    const float * V,
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

    // Use PADDED_HEAD_DIM for all XMX computations
    constexpr int Q_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int K_STRIDE = PADDED_HEAD_DIM + 8;
    constexpr int V_STRIDE = PADDED_HEAD_DIM + 8;
    // V is stored col-major in shVT as [BLOCK_N rows, PADDED_HEAD_DIM cols] for P@V
    // For col-major, stride (leading dimension) must be >= number of rows = BLOCK_N
    constexpr int V_T_STRIDE = BLOCK_N + 8;  // Must be >= BLOCK_N for col-major storage
    constexpr int S_STRIDE = BLOCK_N + 8;
    constexpr int P_STRIDE = BLOCK_N + 8;

    // Ensure BLOCK_N is divisible by TK for P@V computation
    static_assert(BLOCK_N % TK == 0, "BLOCK_N must be divisible by TK for P@V matmul");
    static_assert(PADDED_HEAD_DIM % TN == 0, "PADDED_HEAD_DIM must be divisible by TN for output tiles");

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

    // Use HEAD_DIM for actual data access, PADDED_HEAD_DIM for XMX operations
    const ptrdiff_t q_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t)(kv_head_idx * N_kv) * HEAD_DIM;  // K uses N_kv
    const ptrdiff_t v_offset = (ptrdiff_t)(kv_head_idx * N_kv) * HEAD_DIM;  // V uses N_kv
    const ptrdiff_t o_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;

    // Bfloat16 shared memory for XMX tiles
    xmx_bfloat16 * shQ = reinterpret_cast<xmx_bfloat16*>(shmem);
    xmx_bfloat16 * shK = shQ + BLOCK_M * Q_STRIDE;
    xmx_bfloat16 * shV = shK + BLOCK_N * K_STRIDE;
    xmx_bfloat16 * shP = shV + BLOCK_N * V_STRIDE;
    xmx_bfloat16 * shVT = shP + BLOCK_M * P_STRIDE;  // V^T buffer (separate from shK to avoid corruption)

    // Float shared memory for softmax scores and output accumulator
    // shVT is [PADDED_HEAD_DIM x BLOCK_N] col-major, so advance by PADDED_HEAD_DIM * V_T_STRIDE
    float * shS = reinterpret_cast<float*>(shVT + PADDED_HEAD_DIM * V_T_STRIDE);
    float * rowMax = shS + BLOCK_M * S_STRIDE;
    float * rowSum = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;
    float * shAcc = rowAlpha + BLOCK_M;  // Sized for PADDED_HEAD_DIM

    // Initialize per-row stats and output accumulator
    if (lid < BLOCK_M) {
        rowMax[lid] = -1.0e20f;
        rowSum[lid] = 0.0f;
    }
    for (int i = lid; i < BLOCK_M * PADDED_HEAD_DIM; i += THREADS) {
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

        // Load K and V tiles with xmx_bfloat16 conversion and padding
        for (int i = lid; i < BLOCK_N * PADDED_HEAD_DIM; i += THREADS) {
            const int r = i / PADDED_HEAD_DIM;
            const int c = i % PADDED_HEAD_DIM;
            const int k_row = col0 + r;

            xmx_bfloat16 k_val = xmx_bfloat16(0.0f);
            xmx_bfloat16 v_val = xmx_bfloat16(0.0f);

            if (k_row < N_kv && c < HEAD_DIM) {  // Use N_kv for bounds check
                const ptrdiff_t base = (ptrdiff_t)k_row * HEAD_DIM;
                k_val = xmx_bfloat16(K[k_offset + base + c]);
                v_val = xmx_bfloat16(V[v_offset + base + c]);
            }

            // Store K as K^T col-major: K^T[head_dim, kv_col] at shK[head_dim + kv_col * K_STRIDE]
            shK[c + r * K_STRIDE] = k_val;
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
        for (int i = lid; i < BLOCK_M * PADDED_HEAD_DIM; i += THREADS) {
            const int row = i / PADDED_HEAD_DIM;
            shAcc[i] *= rowAlpha[row];
        }

        // ============ Convert P to xmx_bfloat16 for XMX ============
        for (int i = lid; i < BLOCK_M * BLOCK_N; i += THREADS) {
            const int row = i / BLOCK_N;
            const int col = i % BLOCK_N;
            shP[row * P_STRIDE + col] = xmx_bfloat16(shS[row * S_STRIDE + col]);
        }

        // ============ Copy V to shVT for P@V computation ============
        // V is stored as [BLOCK_N x PADDED_HEAD_DIM] row-major in shV
        // Store V col-major in shVT as [BLOCK_N rows, PADDED_HEAD_DIM cols]
        // V[row, col] stored at shVT[row + col * V_T_STRIDE]
        for (int i = lid; i < BLOCK_N * PADDED_HEAD_DIM; i += THREADS) {
            const int row = i / PADDED_HEAD_DIM;  // kv position
            const int col = i % PADDED_HEAD_DIM;  // head dim
            // Only copy actual HEAD_DIM data, padding elements are zero
            if (col < HEAD_DIM) {
                shVT[row + col * V_T_STRIDE] = shV[row * V_STRIDE + col];
            } else {
                shVT[row + col * V_T_STRIDE] = xmx_bfloat16(0.0f);  // Zero for padding
            }
        }

        it.barrier(sycl::access::fence_space::local_space);

        // ============ P @ V computation ============
        // P is [BLOCK_M x BLOCK_N], V is [BLOCK_N x PADDED_HEAD_DIM]
        // Output is [BLOCK_M x PADDED_HEAD_DIM], computed as tiles of [TM x TN]
        constexpr int NUM_OUT_TILES = PADDED_HEAD_DIM / TN;
        
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

                // V tile: TK x TN from shVT (col-major, V[BLOCK_N rows, PADDED_HEAD_DIM cols])
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
                if (global_row < BLOCK_M && global_col < PADDED_HEAD_DIM) {
                    shAcc[global_row * PADDED_HEAD_DIM + global_col] += shS[scratch_offset + idx];
                }
            }

            it.barrier(sycl::access::fence_space::local_space);
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    // ============ Final normalize and store ============
    // Only write actual HEAD_DIM elements, skip padding
    for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        const int q_row = row0 + row;

        if (q_row < N) {
            float s = rowSum[row];
            // shAcc is indexed with PADDED_HEAD_DIM stride
            float acc_val = shAcc[row * PADDED_HEAD_DIM + col];
            float normalized = acc_val / (s > 1e-10f ? s : 1.0f);
            O[o_offset + (ptrdiff_t)row * o_row_stride + col] = normalized;
        }
    }
}

// Wrapper for Arc B60/Battlemage with padding support
template <int64_t HEAD_DIM, int64_t PADDED_HEAD_DIM>
inline void flash_attn_coopmat_kernel_dg2_padded(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, const int o_row_stride, float * shmem
) {
    flash_attn_coopmat_kernel_padded<HEAD_DIM, PADDED_HEAD_DIM, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, o_row_stride, shmem
    );
}

// Wrapper for PVC with padding support
template <int64_t HEAD_DIM, int64_t PADDED_HEAD_DIM>
inline void flash_attn_coopmat_kernel_pvc_padded(
    sycl::nd_item<2> it,
    const float * Q, const float * K, const float * V,
    float * O, float * l_d, float * m_d,
    const int64_t N, const int64_t N_kv, const int n_heads, const int n_kv_heads,
    const int gqa_ratio, const float scale,
    const float * mask, const int64_t mask_stride, const int o_row_stride, float * shmem
) {
    flash_attn_coopmat_kernel_padded<HEAD_DIM, PADDED_HEAD_DIM, 8, 16, 16>(
        it, Q, K, V, O, l_d, m_d, N, N_kv, n_heads, n_kv_heads,
        gqa_ratio, scale, mask, mask_stride, o_row_stride, shmem
    );
}

#endif // SYCL_EXT_COOPERATIVE_MATRICES

#endif // GGML_SYCL_FATTN_KERNEL_HPP


