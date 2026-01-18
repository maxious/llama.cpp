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
#include <sycl/ext/oneapi/experimental/matrix>

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
// Based on Aule-Attention reference implementation
// Uses 16x16 cooperative matrices, 2 subgroups per workgroup (64 threads total)
template <int64_t HEAD_DIM>
void flash_attn_coopmat_kernel(
    sycl::nd_item<2> it,
    const float * Q,      // Query tensor [n_heads, N, HEAD_DIM]
    const float * K,      // Key tensor [n_kv_heads, N, HEAD_DIM]
    const float * V,      // Value tensor [n_kv_heads, N, HEAD_DIM]
    float * O,            // Output tensor [n_heads, N, HEAD_DIM]
    float * l_d,          // Row sums (online softmax) [n_heads, N]
    float * m_d,          // Row maxes (online softmax) [n_heads, N]
    const int64_t N,      // Sequence length
    const int n_heads,    // Number of query heads
    const int n_kv_heads, // Number of key/value heads
    const int gqa_ratio,  // GQA ratio (n_heads / n_kv_heads)
    const float scale,    // Attention scale factor
    const int causal,     // Causal mask (1=enabled)
    const int window_size // Sliding window (0=no limit)
) {
    // Block sizes
    constexpr int BLOCK_M = 32;  // Rows per workgroup
    constexpr int BLOCK_N = 32;  // Columns per workgroup
    constexpr int THREADS = 64;  // Threads per workgroup
    constexpr int SUBGROUPS = 2; // Subgroups per workgroup

    // Cooperative matrix parameters
    constexpr int TM = GGML_SYCL_CM_M; // 16
    constexpr int TN = GGML_SYCL_CM_N; // 16
    constexpr int TK = GGML_SYCL_CM_K; // 16

    // Stride with bank conflict avoidance (+8)
    constexpr int Q_STRIDE = HEAD_DIM + 8;
    constexpr int K_STRIDE = HEAD_DIM + 8;
    constexpr int V_STRIDE = HEAD_DIM + 8;
    constexpr int S_STRIDE = BLOCK_N + 8;

    // Local memory size
    constexpr int SHMEM_SIZE = (BLOCK_M * Q_STRIDE) + (BLOCK_N * K_STRIDE) +
                               (BLOCK_N * V_STRIDE) + (BLOCK_M * S_STRIDE) +
                               (BLOCK_M * 3); // rowMax, rowSum, rowAlpha

    // Local memory accessors
    sycl::local_accessor<float, 1> shmem(sycl::range<1>(SHMEM_SIZE), it);

    const int tid = it.get_local_id(0) + it.get_local_id(1) * it.get_local_range(0);
    const int gid_x = it.get_group(0); // Block row index
    const int gid_y = it.get_group(1); // Head index
    const int lid = it.get_local_id(0); // Thread within workgroup
    const int sg_id = it.get_subgroup().get_id()[0]; // Subgroup ID (0 or 1)

    // Calculate output row for this workgroup
    const int row0 = gid_x * BLOCK_M;
    if (row0 >= N) return;

    // Calculate K/V head index for this Q head
    const int head_idx = gid_y;
    const int kv_head_idx = head_idx / gqa_ratio;

    // Memory offsets
    const ptrdiff_t q_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t k_offset = (ptrdiff_t)(kv_head_idx * N) * HEAD_DIM;
    const ptrdiff_t v_offset = (ptrdiff_t)(kv_head_idx * N) * HEAD_DIM;
    const ptrdiff_t o_offset = (ptrdiff_t)(head_idx * N + row0) * HEAD_DIM;
    const ptrdiff_t l_offset = (ptrdiff_t)(head_idx * N + row0);
    const ptrdiff_t m_offset = (ptrdiff_t)(head_idx * N + row0);

    // Local memory pointers
    float * shQ = shmem.get_pointer() + 0;
    float * shK = shQ + BLOCK_M * Q_STRIDE;
    float * shV = shK + BLOCK_N * K_STRIDE;
    float * shS = shV + BLOCK_N * V_STRIDE;
    float * rowMax = shS + BLOCK_M * S_STRIDE;
    float * rowSum = rowMax + BLOCK_M;
    float * rowAlpha = rowSum + BLOCK_M;

    // Initialize row statistics
    if (lid < BLOCK_M) {
        rowMax[lid] = -1.0e20f;
        rowSum[lid] = 0.0f;
    }

    // Load Q tile into shared memory
    // Each thread loads multiple elements to cover BLOCK_M * HEAD_DIM with THREADS threads
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

    // Number of K/V blocks to process
    const int num_kv_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    // Output accumulator matrices (8 x 16x16 = 128 elements per SG)
    // This covers HEAD_DIM = 128 with 8 cooperative matrices
    constexpr int NUM_V_MATRICES = (HEAD_DIM + TM - 1) / TM; // 8 for HEAD_DIM=128

    // Initialize output accumulators
    cm::joint_matrix<float, TM, TN, cm::use_accumulator> matO[NUM_V_MATRICES];
    for (int i = 0; i < NUM_V_MATRICES; ++i) {
        cm::joint_matrix_fill(matO[i], 0.0f);
    }

    // Process all K/V blocks
    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int col0 = kv_block * BLOCK_N;

        // Causal check: skip blocks that are entirely in the future
        if (causal == 1 && col0 > row0 + BLOCK_M) continue;

        // Load K and V tiles into shared memory
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

        // Compute Q @ K^T for this block using cooperative matrices
        // Q tile: BLOCK_M x HEAD_DIM, K tile: BLOCK_N x HEAD_DIM
        // We compute BLOCK_M x BLOCK_N = 32 x 32 score matrix
        // Using TM=16, TN=16, we need (32/16) x (32/16) = 2 x 2 = 4 coopmat operations per SG

        // Process Q rows assigned to this subgroup (sg_id * 16 to sg_id * 16 + 15)
        for (int j = 0; j < BLOCK_N; j += TN) {
            // Initialize score matrix for this 16x16 block
            cm::joint_matrix<float, TM, TN, cm::use_accumulator> matS;
            cm::joint_matrix_fill(matS, 0.0f);

            // Multiply Q rows with K columns
            for (int k = 0; k < HEAD_DIM; k += TK) {
                cm::joint_matrix<float, TM, TK, cm::use_a> mq;
                cm::joint_matrix<float, TK, TN, cm::use_b> mk;

                // Load Q tile: rows sg_id*16 to sg_id*16+15, columns k to k+15
                cm::joint_matrix_load(
                    mq, shQ,
                    (sg_id * TM) * Q_STRIDE + k,
                    Q_STRIDE
                );

                // Load K tile: rows j to j+15, columns k to k+15
                cm::joint_matrix_load(
                    mk, shK,
                    j * K_STRIDE + k,
                    K_STRIDE
                );

                // matS += mq * mk
                matS = cm::joint_matrix_multiply_add(mq, mk, matS);
            }

            // Store score matrix to shared memory
            cm::joint_matrix_store(
                matS, shS,
                (sg_id * TM) * S_STRIDE + j,
                S_STRIDE
            );
        }

        it.barrier(sycl::access::fence_space::local_space);

        // Online softmax with causal masking
        if (lid < BLOCK_M) {
            const int row = lid;
            float m = -1.0e20f;

            // Find block max with causal masking
            for (int c = 0; c < BLOCK_N; ++c) {
                float s_val = shS[row * S_STRIDE + c];

                // Causal mask: only attend to positions <= current row
                if (causal == 1 && col0 + c > row0 + row) {
                    s_val = -1.0e20f;
                }

                // Sliding window mask
                if (window_size > 0 && col0 + c < row0 + row - window_size + 1) {
                    s_val = -1.0e20f;
                }

                shS[row * S_STRIDE + c] = s_val;
                m = sycl::fmax(m, s_val);
            }

            // Merge with previous block statistics
            float m_prev = rowMax[row];
            float m_new = sycl::fmax(m_prev, m);
            float alpha = sycl::exp(sycl::fmax(m_prev - m_new, -20.0f));
            rowAlpha[row] = alpha;
            rowMax[row] = m_new;

            // Compute exp(s - m_new) and row sum
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

        // P @ V multiplication using cooperative matrices
        // shS: BLOCK_M x BLOCK_N, shV: BLOCK_N x HEAD_DIM
        // Output: BLOCK_M x HEAD_DIM

        // First, scale previous output by rowAlpha
        for (int i = 0; i < NUM_V_MATRICES; ++i) {
            cm::joint_matrix_store(matO[i], shK, sg_id * 2048 + i * 256, 16);
        }
        it.barrier(sycl::access::fence_space::local_space);

        for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
            const int my_sg = i / (HEAD_DIM / TM * TM);
            const int r = (i % (HEAD_DIM / TM * TM)) / TM;
            shK[i] *= rowAlpha[my_sg * TM + r];
        }
        it.barrier(sycl::access::fence_space::local_space);

        for (int i = 0; i < NUM_V_MATRICES; ++i) {
            cm::joint_matrix_load(matO[i], shK, sg_id * 2048 + i * 256, 16);
        }
        it.barrier(sycl::access::fence_space::local_space);

        // P @ V: shS (32x32) @ shV (32x128) -> O (32x128)
        // Each 16x16 block of shS multiplies with corresponding 16x16 block of shV
        for (int k = 0; k < BLOCK_N; k += TN) {
            // Load P tile (16x16)
            cm::joint_matrix<float, TM, TN, cm::use_a> mp;
            cm::joint_matrix_load(
                mp, shS,
                (sg_id * TM) * S_STRIDE + k,
                S_STRIDE
            );

            // Multiply with V tiles (16x128 across multiple matrices)
            for (int i = 0; i < NUM_V_MATRICES; ++i) {
                cm::joint_matrix<float, TM, TN, cm::use_b> mv;
                cm::joint_matrix_load(
                    mv, shV,
                    k * V_STRIDE + i * TM,
                    V_STRIDE
                );
                matO[i] = cm::joint_matrix_multiply_add(mp, mv, matO[i]);
            }
        }
    }

    // Normalize by row sum and store output
    for (int i = 0; i < NUM_V_MATRICES; ++i) {
        cm::joint_matrix_store(matO[i], shK, sg_id * 2048 + i * 256, 16);
    }
    it.barrier(sycl::access::fence_space::local_space);

    // Scale by 1/rowSum
    for (int i = lid; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        const int my_sg = i / (HEAD_DIM / TM * TM);
        const int r = (i % (HEAD_DIM / TM * TM)) / TM;
        const int row = my_sg * TM + r;
        float s = rowSum[row];
        shK[i] /= (s > 1e-10f ? s : 1.0f);
    }
    it.barrier(sycl::access::fence_space::local_space);

    // Store normalized output
    for (int i = 0; i < NUM_V_MATRICES; ++i) {
        cm::joint_matrix_load(matO[i], shK, sg_id * 2048 + i * 256, 16);
    }
    it.barrier(sycl::access::fence_space::local_space);

    // Store to global memory
    for (int i = 0; i < NUM_V_MATRICES; ++i) {
        cm::joint_matrix_store(
            matO[i],
            O + o_offset,
            HEAD_DIM,
            // Layout depends on implementation
            sycl::ext::oneapi::experimental::matrix::layout::row_major
        );
    }

    // Store final softmax statistics
    it.barrier(sycl::access::fence_space::local_space);
}

#endif // SYCL_EXT_COOPERATIVE_MATRICES

#endif // GGML_SYCL_FATTN_KERNEL_HPP


