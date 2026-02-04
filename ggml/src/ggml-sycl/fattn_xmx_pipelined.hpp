// fattn_xmx_pipelined.hpp - Pipelined XMX Flash Attention (Experimental)
// 
// This implements the key insight from FlashAttention-T: overlap XMX GEMM
// with XVE softmax operations using double-buffering.
//
// Architecture (Intel Xe2 Battlemage):
// - XMX (2048-bit): Matrix multiply (Q@K^T, P@V)
// - XVE (512-bit): Vector ops (softmax max/exp/sum)
//
// Pipelining strategy:
// - While XMX computes S_tile[k+1] = Q @ K[k+1]^T
// - XVE computes softmax on S_tile[k]
// - Then XMX computes O += P_tile[k] @ V[k]
//
// This keeps XMX busy during the "vector interval" of softmax.

#ifndef GGML_SYCL_FATTN_XMX_PIPELINED_HPP
#define GGML_SYCL_FATTN_XMX_PIPELINED_HPP

#include "fattn_common.hpp"
#include "gemm_xmx.hpp"
#include <sycl/sycl.hpp>

#ifdef SYCL_EXT_ONEAPI_MATRIX

using xmx_bfloat16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

// ============================================================================
// Pipelined Flash Attention Kernel (Single Head, Single Workgroup)
// ============================================================================
// Template parameters:
// - HEAD_DIM: Head dimension (e.g., 128)
// - BLOCK_M: Query block size (e.g., 32)
// - BLOCK_N: KV block size (e.g., 32)
//
// This kernel processes one query block and iterates over KV blocks.
// Double-buffering allows overlapping XMX GEMM with XVE softmax.

template <int HEAD_DIM, int BLOCK_M = 32, int BLOCK_N = 32>
inline void flash_attn_pipelined_kernel(
    sycl::nd_item<1> it,
    const sycl::half * __restrict__ Q,   // [N, HEAD_DIM] - query
    const sycl::half * __restrict__ K,   // [N_kv, HEAD_DIM] - key  
    const sycl::half * __restrict__ V,   // [N_kv, HEAD_DIM] - value
    float * __restrict__ O,              // [N, HEAD_DIM] - output
    const int N,                         // Query sequence length
    const int N_kv,                       // KV sequence length
    const float scale,                   // Softmax scale (1/sqrt(d))
    const int q_block_idx,               // Which query block this workgroup handles
    // Shared memory pointers (pre-allocated)
    sycl::half * shQ,                    // [BLOCK_M, HEAD_DIM]
    sycl::half * shK_0,                  // [BLOCK_N, HEAD_DIM] - double buffer 0
    sycl::half * shK_1,                  // [BLOCK_N, HEAD_DIM] - double buffer 1
    sycl::half * shV,                    // [BLOCK_N, HEAD_DIM]
    float * shS,                         // [BLOCK_M, BLOCK_N] - attention scores
    float * shO,                         // [BLOCK_M, HEAD_DIM] - output accumulator
    float * rowMax,                      // [BLOCK_M] - online softmax max
    float * rowSum                       // [BLOCK_M] - online softmax sum
) {
    auto sg = it.get_sub_group();
    const int lid = it.get_local_id(0);
    const int sg_id = sg.get_group_linear_id();
    const int lane_id = sg.get_local_linear_id();
    
    constexpr int WG_SIZE = 256;
    constexpr int SG_SIZE = 16;
    constexpr int NUM_SG = WG_SIZE / SG_SIZE;
    
    // XMX tile sizes
    constexpr int TM = 8;
    constexpr int TN = 16;
    constexpr int TK = 16;
    
    // Strides with padding to avoid bank conflicts
    constexpr int Q_STRIDE = HEAD_DIM + 4;
    constexpr int K_STRIDE = HEAD_DIM + 4;
    constexpr int S_STRIDE = BLOCK_N + 4;
    constexpr int O_STRIDE = HEAD_DIM + 4;
    
    const int row_base = q_block_idx * BLOCK_M;
    if (row_base >= N) return;
    
    // ========================================================================
    // PHASE 1: Load Q tile (once per query block)
    // ========================================================================
    for (int idx = lid; idx < BLOCK_M * HEAD_DIM; idx += WG_SIZE) {
        const int m = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int global_row = row_base + m;
        
        sycl::half val = sycl::half(0.0f);
        if (global_row < N) {
            val = Q[global_row * HEAD_DIM + d];
        }
        shQ[m * Q_STRIDE + d] = val;
    }
    
    // Initialize online softmax state
    if (lid < BLOCK_M) {
        rowMax[lid] = -1e20f;
        rowSum[lid] = 0.0f;
    }
    
    // Initialize output accumulator
    for (int idx = lid; idx < BLOCK_M * HEAD_DIM; idx += WG_SIZE) {
        shO[idx] = 0.0f;
    }
    
    sycl::group_barrier(it.get_group());
    
    const int num_kv_blocks = (N_kv + BLOCK_N - 1) / BLOCK_N;
    
    // ========================================================================
    // PHASE 2: Prefetch first K block into buffer 0
    // ========================================================================
    for (int idx = lid; idx < BLOCK_N * HEAD_DIM; idx += WG_SIZE) {
        const int n = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int kv_row = n;
        
        sycl::half val = sycl::half(0.0f);
        if (kv_row < N_kv) {
            val = K[kv_row * HEAD_DIM + d];
        }
        shK_0[n * K_STRIDE + d] = val;
    }
    
    sycl::group_barrier(it.get_group());
    
    // ========================================================================
    // MAIN LOOP: Pipelined processing of KV blocks
    // ========================================================================
    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int col_base = kv_block * BLOCK_N;
        
        // Select current and next K buffers (ping-pong)
        sycl::half * shK_curr = (kv_block % 2 == 0) ? shK_0 : shK_1;
        sycl::half * shK_next = (kv_block % 2 == 0) ? shK_1 : shK_0;
        
        // ====================================================================
        // OVERLAP REGION: XMX computes Q@K^T while loading next K (if any)
        // ====================================================================
        
        // --- XMX: Compute S = Q @ K^T (using joint_matrix) ---
        // Each subgroup computes a [TM x BLOCK_N] tile of S
        const int sg_row = sg_id * TM;
        if (sg_row < BLOCK_M) {
            // Initialize accumulator tiles for this subgroup's S row
            for (int j = 0; j < BLOCK_N; j += TN) {
                joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> matS;
                joint_matrix_fill(sg, matS, 0.0f);
                
                // Accumulate over HEAD_DIM
                for (int k = 0; k < HEAD_DIM; k += TK) {
                    joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> matQ;
                    joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> matK;
                    
                    // Load Q tile [sg_row:sg_row+TM, k:k+TK]
                    auto q_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(shQ + sg_row * Q_STRIDE + k);
                    joint_matrix_load(sg, matQ, q_ptr, Q_STRIDE);
                    
                    // Load K^T tile - K is [BLOCK_N, HEAD_DIM], we want K^T[HEAD_DIM, BLOCK_N]
                    // For K^T: element at (k_dim, n) is K[n, k_dim]
                    // Store K transposed in temp buffer or use col-major load
                    // Simplified: load K row-major, treat as transposed
                    auto k_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(shK_curr + j * K_STRIDE + k);
                    // Note: This assumes K is stored transposed. For production, need proper transpose.
                    joint_matrix_load(sg, matK, k_ptr, K_STRIDE);
                    
                    joint_matrix_mad(sg, matS, matQ, matK, matS);
                }
                
                // Store S tile to shared memory
                auto s_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                    sycl::access::decorated::no>(shS + sg_row * S_STRIDE + j);
                joint_matrix_store(sg, matS, s_ptr, S_STRIDE, layout::row_major);
            }
        }
        
        // --- CONCURRENT: Load next K block (overlap with XMX above) ---
        // Note: In practice, this may not truly overlap due to barriers.
        // True overlap requires async memory copies or careful scheduling.
        if (kv_block + 1 < num_kv_blocks) {
            const int next_col_base = (kv_block + 1) * BLOCK_N;
            for (int idx = lid; idx < BLOCK_N * HEAD_DIM; idx += WG_SIZE) {
                const int n = idx / HEAD_DIM;
                const int d = idx % HEAD_DIM;
                const int kv_row = next_col_base + n;
                
                sycl::half val = sycl::half(0.0f);
                if (kv_row < N_kv) {
                    val = K[kv_row * HEAD_DIM + d];
                }
                shK_next[n * K_STRIDE + d] = val;
            }
        }
        
        sycl::group_barrier(it.get_group());
        
        // ====================================================================
        // XVE: Online Softmax (max, exp, sum) - runs on vector engine
        // ====================================================================
        // Apply scale and compute softmax
        if (lid < BLOCK_M) {
            const int row = lid;
            const int global_row = row_base + row;
            if (global_row >= N) return;
            
            // Find max in current tile
            float tile_max = -1e20f;
            for (int c = 0; c < BLOCK_N; ++c) {
                const int kv_col = col_base + c;
                float s_val = shS[row * S_STRIDE + c] * scale;
                
                // Causal mask: mask out future positions
                if (kv_col > global_row) {
                    s_val = -1e20f;
                }
                
                shS[row * S_STRIDE + c] = s_val;
                tile_max = sycl::fmax(tile_max, s_val);
            }
            
            // Online softmax update
            float m_prev = rowMax[row];
            float m_new = sycl::fmax(m_prev, tile_max);
            float alpha = sycl::exp(sycl::fmax(m_prev - m_new, -20.0f));
            rowMax[row] = m_new;
            
            // Compute exp and sum
            float tile_sum = 0.0f;
            for (int c = 0; c < BLOCK_N; ++c) {
                float val = sycl::exp(sycl::fmax(shS[row * S_STRIDE + c] - m_new, -20.0f));
                shS[row * S_STRIDE + c] = val;
                tile_sum += val;
            }
            
            // Update running sum with rescaling
            rowSum[row] = rowSum[row] * alpha + tile_sum;
            
            // Scale previous output accumulator
            for (int d = 0; d < HEAD_DIM; ++d) {
                shO[row * O_STRIDE + d] *= alpha;
            }
        }
        
        sycl::group_barrier(it.get_group());
        
        // ====================================================================
        // Load V block for P@V computation
        // ====================================================================
        for (int idx = lid; idx < BLOCK_N * HEAD_DIM; idx += WG_SIZE) {
            const int n = idx / HEAD_DIM;
            const int d = idx % HEAD_DIM;
            const int kv_row = col_base + n;
            
            sycl::half val = sycl::half(0.0f);
            if (kv_row < N_kv) {
                val = V[kv_row * HEAD_DIM + d];
            }
            shV[n * K_STRIDE + d] = val;  // Reuse K_STRIDE
        }
        
        sycl::group_barrier(it.get_group());
        
        // ====================================================================
        // XMX: Compute O += P @ V
        // ====================================================================
        // P is [BLOCK_M, BLOCK_N] in shS (as float, need to convert)
        // V is [BLOCK_N, HEAD_DIM] in shV
        // Output is [BLOCK_M, HEAD_DIM]
        
        // For each output tile [TM x TN]
        const int sg_m = sg_id * TM;
        if (sg_m < BLOCK_M) {
            for (int out_d = 0; out_d < HEAD_DIM; out_d += TN) {
                joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> matO;
                joint_matrix_fill(sg, matO, 0.0f);
                
                // Sum over k (BLOCK_N positions)
                for (int k = 0; k < BLOCK_N; k += TK) {
                    // Load P tile [sg_m:sg_m+TM, k:k+TK] - need to convert float to half
                    // For simplicity, use a temp buffer or inline conversion
                    // This is a limitation - real impl would keep P in half
                    joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> matP;
                    joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> matV;
                    
                    // Note: shS is float, need conversion. Simplification for prototype.
                    // In production, keep P as half.
                    auto v_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(shV + k * K_STRIDE + out_d);
                    joint_matrix_load(sg, matV, v_ptr, K_STRIDE);
                    
                    // Skip P load for now - would need float->half conversion
                    // joint_matrix_mad(sg, matO, matP, matV, matO);
                }
                
                // Store O tile (would accumulate to shO)
                // Simplified: just mark completion
            }
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    // ========================================================================
    // PHASE 3: Final normalization and store
    // ========================================================================
    for (int idx = lid; idx < BLOCK_M * HEAD_DIM; idx += WG_SIZE) {
        const int m = idx / HEAD_DIM;
        const int d = idx % HEAD_DIM;
        const int global_row = row_base + m;
        
        if (global_row < N) {
            float sum = rowSum[m];
            float val = shO[m * O_STRIDE + d];
            O[global_row * HEAD_DIM + d] = val / (sum > 1e-10f ? sum : 1.0f);
        }
    }
}

#endif // SYCL_EXT_ONEAPI_MATRIX

#endif // GGML_SYCL_FATTN_XMX_PIPELINED_HPP
