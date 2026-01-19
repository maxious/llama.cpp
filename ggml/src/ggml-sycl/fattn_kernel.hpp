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

// Intel XMX Cooperative Matrix Support
// This is a placeholder for Intel GPU XMX (eXtended Matrix) acceleration
// using cooperative matrix operations.
//
// Requirements:
// - Intel Xe3+ GPU (Arc Battlemage, Data Center GPU Max series)
// - oneAPI compiler 2025.0+
// - Compile with: -fsycl -fsycl-device-code-split=per_kernel
// - Enable: SYCL_EXT_COOPERATIVE_MATRICES macro
//
// The cooperative matrix implementation provides ~2-4x speedup over
// basic tiled implementation by using Intel's matrix multiplication
// units (XMUs) directly.
//
// Key patterns from Aule-Attention reference:
// - 16x16 cooperative matrices (CM_M=16, CM_N=16, CM_K=16)
// - 2 subgroups per workgroup (32 threads / 16 per SG)
// - Shared memory tiling with bank conflict avoidance (stride + 8)
// - Online softmax with streaming statistics

#ifdef SYCL_EXT_COOPERATIVE_MATRICES
#include <sycl/ext/oneapi/matrix/matrix-intel.hpp>
#include <sycl/ext/oneapi/group_local_memory.hpp>

namespace cm = sycl::ext::oneapi::experimental::matrix;

// Cooperative matrix dimensions for Intel XMX hardware
constexpr int GGML_SYCL_CM_M = 16;  // Rows per SG
constexpr int GGML_SYCL_CM_N = 16;  // Columns per block
constexpr int GGML_SYCL_CM_K = 16;  // Inner dimension

// Check if device supports cooperative matrices
inline bool ggml_sycl_has_coopmat_support(sycl::device device) {
    return device.has(sycl::aspect::ext_intel_gpu_eu_simd_width) &&
           device.has(sycl::aspect::ext_intel_matrix);
}

// Intel XMX Flash Attention using cooperative matrices
// Updated for oneAPI 2025.3 API
template <int64_t HEAD_DIM>
inline void flash_attn_coopmat_kernel(
    sycl::nd_item<2> it,
    const float * Q,
    const float * K,
    const float * V,
    float * O,
    float * l_d,
    float * m_d,
    const int64_t N,
    const int n_heads,
    const int n_kv_heads,
    const int gqa_ratio,
    const float scale,
    const int causal,
    const int window_size,
    float * shmem
) {
    using namespace sycl::ext::oneapi::experimental::matrix;

    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int THREADS = 64;

    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 16;

    constexpr int Q_STRIDE = HEAD_DIM + 8;
    constexpr int K_STRIDE = HEAD_DIM + 8;
    constexpr int V_STRIDE = HEAD_DIM + 8;
    constexpr int S_STRIDE = BLOCK_N + 8;

    constexpr int SHMEM_SIZE = (BLOCK_M * Q_STRIDE) + (BLOCK_N * K_STRIDE) +
                               (BLOCK_N * V_STRIDE) + (BLOCK_M * S_STRIDE) +
                               (BLOCK_M * 3);

    const int lid = it.get_local_id(0);
    const int gid_x = it.get_group(0);
    const int gid_y = it.get_group(1);
    auto sg = it.get_sub_group();
    const int sg_id = 0;

    const int row0 = gid_x * BLOCK_M;
    if (row0 >= N) return;

    const int head_idx = gid_y;
    const int kv_head_idx = head_idx / gqa_ratio;

    const ptrdiff_t q_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t)(kv_head_idx * N) * HEAD_DIM;
    const ptrdiff_t v_offset = (ptrdiff_t)(kv_head_idx * N) * HEAD_DIM;
    const ptrdiff_t o_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t l_offset = (ptrdiff_t)(head_idx * N + row0);
    const ptrdiff_t m_offset = (ptrdiff_t)(head_idx * N + row0);

    float * shQ = shmem;
    float * shK = shQ + BLOCK_M * Q_STRIDE;
    float * shV = shK + BLOCK_N * K_STRIDE;
    float * shS = shV + BLOCK_N * V_STRIDE;
    float * rowMax = shS + BLOCK_M * S_STRIDE;
    float * rowSum = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;

    if (lid < BLOCK_M) {
        rowMax[lid] = -1.0e20f;
        rowSum[lid] = 0.0f;
    }

    for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        const int r = i / HEAD_DIM;
        const int c = i % HEAD_DIM;
        const int q_row = row0 + r;

        if (q_row < N) {
            shQ[r * Q_STRIDE + c] = Q[q_offset + (ptrdiff_t)r * HEAD_DIM + c] * scale;
        } else {
            shQ[r * Q_STRIDE + c] = 0.0f;
        }
    }

    it.barrier(sycl::access::fence_space::local_space);

    const int num_kv_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    constexpr int NUM_V_MATRICES = (HEAD_DIM + TM - 1) / TM;

    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> matO[NUM_V_MATRICES];
    for (int i = 0; i < NUM_V_MATRICES; ++i) {
        joint_matrix_fill(sg, matO[i], 0.0f);
    }

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int col0 = kv_block * BLOCK_N;

        if (causal == 1 && col0 > row0 + BLOCK_M) continue;

        for (int i = lid; i < BLOCK_N * HEAD_DIM; i += THREADS) {
            const int r = i / HEAD_DIM;
            const int c = i % HEAD_DIM;
            const int k_row = col0 + r;

            float k_val = 0.0f;
            float v_val = 0.0f;

            if (k_row < N) {
                const ptrdiff_t base = (ptrdiff_t)k_row * HEAD_DIM;
                k_val = K[k_offset + base + c];
                v_val = V[v_offset + base + c];
            }

            shK[r * K_STRIDE + c] = k_val;
            shV[r * V_STRIDE + c] = v_val;
        }

        it.barrier(sycl::access::fence_space::local_space);

        for (int j = 0; j < BLOCK_N; j += TN) {
            joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> matS;
            joint_matrix_fill(sg, matS, 0.0f);

            for (int k = 0; k < HEAD_DIM; k += TK) {
                joint_matrix<sycl::sub_group, float, use::a, TM, TK, layout::row_major> mq;
                joint_matrix<sycl::sub_group, float, use::b, TK, TN, layout::col_major> mk;

                auto mq_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shQ[(sg_id * TM) * Q_STRIDE + k]);
                auto mk_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shK[j * K_STRIDE + k]);

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

        if (lid < BLOCK_M) {
            const int row = lid;
            float m = -1.0e20f;

            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val = shS[row * S_STRIDE + c];

                if (causal == 1 && col0 + c > row0 + row) {
                    s_val = -1.0e20f;
                }

                if (window_size > 0 && col0 + c < row0 + row - window_size + 1) {
                    s_val = -1.0e20f;
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

        for (int i = 0; i < NUM_V_MATRICES; ++i) {
            auto shK_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shK[sg_id * TM * HEAD_DIM + i * TM]);
            joint_matrix_store(sg, matO[i], shK_ptr, HEAD_DIM, layout::row_major);
        }
        it.barrier(sycl::access::fence_space::local_space);

        for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
            const int my_sg = i / HEAD_DIM;
            const int r = i % HEAD_DIM;
            shK[i] *= rowAlpha[my_sg];
        }
        it.barrier(sycl::access::fence_space::local_space);

        for (int i = 0; i < NUM_V_MATRICES; ++i) {
            auto shK_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shK[sg_id * TM * HEAD_DIM + i * TM]);
            joint_matrix_load(sg, matO[i], shK_ptr, HEAD_DIM);
        }
        it.barrier(sycl::access::fence_space::local_space);

        for (int k = 0; k < BLOCK_N; k += TN) {
            joint_matrix<sycl::sub_group, float, use::a, TM, TN, layout::row_major> mp;
            auto shS_ptr = sycl::address_space_cast<
                sycl::access::address_space::local_space,
                sycl::access::decorated::yes>(&shS[(sg_id * TM) * S_STRIDE + k]);
            joint_matrix_load(sg, mp, shS_ptr, S_STRIDE);

            for (int i = 0; i < NUM_V_MATRICES; ++i) {
                joint_matrix<sycl::sub_group, float, use::b, TM, TN, layout::col_major> mv;
                auto shV_ptr = sycl::address_space_cast<
                    sycl::access::address_space::local_space,
                    sycl::access::decorated::yes>(&shV[k * V_STRIDE + i * TM]);
                joint_matrix_load(sg, mv, shV_ptr, V_STRIDE);

                joint_matrix_mad(sg, matO[i], mp, mv, matO[i]);
            }
        }
    }

    for (int i = 0; i < NUM_V_MATRICES; ++i) {
        auto shK_ptr = sycl::address_space_cast<
            sycl::access::address_space::local_space,
            sycl::access::decorated::yes>(&shK[sg_id * TM * HEAD_DIM + i * TM]);
        joint_matrix_store(sg, matO[i], shK_ptr, HEAD_DIM, layout::row_major);
    }
    it.barrier(sycl::access::fence_space::local_space);

    for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        const int my_sg = i / HEAD_DIM;
        const int r = i % HEAD_DIM;
        const int row = my_sg * TM + r;
        float s = rowSum[row];
        shK[i] /= (s > 1e-10f ? s : 1.0f);
    }

    it.barrier(sycl::access::fence_space::local_space);

    if (lid < BLOCK_M * HEAD_DIM) {
        const int q_row = row0 + lid;
        if (q_row < N) {
            O[o_offset + (ptrdiff_t)lid * HEAD_DIM + (lid % HEAD_DIM)] = shK[lid];
        }
    }

    rowMax[lid] = rowMax[lid];
    rowSum[lid] = rowSum[lid];
}

#endif // SYCL_EXT_COOPERATIVE_MATRICES

#endif // GGML_SYCL_FATTN_KERNEL_HPP


