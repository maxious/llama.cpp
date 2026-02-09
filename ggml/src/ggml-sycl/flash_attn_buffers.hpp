#pragma once

#include <memory>
#include <sycl/sycl.hpp>

// Flash Attention buffer pool to avoid repeated malloc/free overhead
// Preallocate buffers once and reuse across FA calls to reduce kernel launch overhead
struct flash_attn_buffers {
    // Device buffers (float pointers for FA computations)
    float * Q_buf        = nullptr;  // Reordered Q [n_heads, N, DQK]
    float * K_buf        = nullptr;  // Reordered K [n_kv_heads, N_kv, DQK]
    float * V_buf        = nullptr;  // Reordered V [n_kv_heads, N_kv, DV]
    float * S_buf        = nullptr;  // Scores matrix [n_heads, N, N_kv]
    float * partials_buf = nullptr;  // Partial sums [n_heads, N, n_splits, 2+DV]
    float * mask_buf     = nullptr;  // Converted mask (if F16) [N, N_kv]
    float * O_buf        = nullptr;  // Output buffer [n_heads, N, DV] for XMX
    float * l_buf        = nullptr;  // Row max stats [n_heads, N] for XMX
    float * m_buf        = nullptr;  // Row sum stats [n_heads, N] for XMX

    // Pointer arrays for GQA in tiled flash attention
    float ** Q_ptrs_buf = nullptr;  // Q pointer array for batched GEMM
    float ** K_ptrs_buf = nullptr;  // K pointer array for batched GEMM
    float ** S_ptrs_buf = nullptr;  // S pointer array for batched GEMM
    float ** V_ptrs_buf = nullptr;  // V pointer array for batched GEMM
    float ** O_ptrs_buf = nullptr;  // O pointer array for batched GEMM

    // Track allocated sizes in bytes
    size_t Q_size        = 0;
    size_t K_size        = 0;
    size_t V_size        = 0;
    size_t S_size        = 0;
    size_t partials_size = 0;
    size_t mask_size     = 0;
    size_t O_size        = 0;
    size_t l_size        = 0;
    size_t m_size        = 0;
    size_t ptrs_size     = 0;  // Size for all pointer arrays (they're same size)

    // Stream for allocation and cleanup
    sycl::queue * q = nullptr;

    ~flash_attn_buffers() {
        if (q) {
            if (Q_buf) {
                sycl::free(Q_buf, *q);
            }
            if (K_buf) {
                sycl::free(K_buf, *q);
            }
            if (V_buf) {
                sycl::free(V_buf, *q);
            }
            if (S_buf) {
                sycl::free(S_buf, *q);
            }
            if (partials_buf) {
                sycl::free(partials_buf, *q);
            }
            if (mask_buf) {
                sycl::free(mask_buf, *q);
            }
            if (O_buf) {
                sycl::free(O_buf, *q);
            }
            if (l_buf) {
                sycl::free(l_buf, *q);
            }
            if (m_buf) {
                sycl::free(m_buf, *q);
            }
            if (Q_ptrs_buf) {
                sycl::free(Q_ptrs_buf, *q);
            }
            if (K_ptrs_buf) {
                sycl::free(K_ptrs_buf, *q);
            }
            if (S_ptrs_buf) {
                sycl::free(S_ptrs_buf, *q);
            }
            if (V_ptrs_buf) {
                sycl::free(V_ptrs_buf, *q);
            }
            if (O_ptrs_buf) {
                sycl::free(O_ptrs_buf, *q);
            }
        }
    }

    // Get or allocate with automatic reallocation if size increased
    float * get_Q(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > Q_size) {
            if (Q_buf) {
                sycl::free(Q_buf, *q);
            }
            Q_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            Q_size = size_bytes;
            q      = stream;
        }
        return Q_buf;
    }

    float * get_K(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > K_size) {
            if (K_buf) {
                sycl::free(K_buf, *q);
            }
            K_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            K_size = size_bytes;
            q      = stream;
        }
        return K_buf;
    }

    float * get_V(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > V_size) {
            if (V_buf) {
                sycl::free(V_buf, *q);
            }
            V_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            V_size = size_bytes;
            q      = stream;
        }
        return V_buf;
    }

    float * get_S(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > S_size) {
            if (S_buf) {
                sycl::free(S_buf, *q);
            }
            S_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            S_size = size_bytes;
            q      = stream;
        }
        return S_buf;
    }

    float * get_partials(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > partials_size) {
            if (partials_buf) {
                sycl::free(partials_buf, *q);
            }
            partials_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            partials_size = size_bytes;
            q             = stream;
        }
        return partials_buf;
    }

    float * get_mask(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > mask_size) {
            if (mask_buf) {
                sycl::free(mask_buf, *q);
            }
            mask_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            mask_size = size_bytes;
            q         = stream;
        }
        return mask_buf;
    }

    float * get_O(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > O_size) {
            if (O_buf) {
                sycl::free(O_buf, *q);
            }
            O_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            O_size = size_bytes;
            q      = stream;
        }
        return O_buf;
    }

    float * get_l(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > l_size) {
            if (l_buf) {
                sycl::free(l_buf, *q);
            }
            l_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            l_size = size_bytes;
            q      = stream;
        }
        return l_buf;
    }

    float * get_m(size_t size_bytes, sycl::queue * stream) {
        if (size_bytes > m_size) {
            if (m_buf) {
                sycl::free(m_buf, *q);
            }
            m_buf  = (float *) sycl::malloc_device(size_bytes, *stream);
            m_size = size_bytes;
            q      = stream;
        }
        return m_buf;
    }

    // Get pointer arrays for GQA (all allocated together)
    // Returns true if allocation was needed, false if reused
    bool get_ptrs(size_t        size_bytes,
                  sycl::queue * stream,
                  float ***     out_Q_ptrs,
                  float ***     out_K_ptrs,
                  float ***     out_S_ptrs,
                  float ***     out_V_ptrs,
                  float ***     out_O_ptrs) {
        bool was_allocated = false;
        if (size_bytes > ptrs_size) {
            if (Q_ptrs_buf) {
                sycl::free(Q_ptrs_buf, *q);
            }
            if (K_ptrs_buf) {
                sycl::free(K_ptrs_buf, *q);
            }
            if (S_ptrs_buf) {
                sycl::free(S_ptrs_buf, *q);
            }
            if (V_ptrs_buf) {
                sycl::free(V_ptrs_buf, *q);
            }
            if (O_ptrs_buf) {
                sycl::free(O_ptrs_buf, *q);
            }
            // Allocate all 5 pointer arrays
            Q_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            K_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            S_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            V_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            O_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            ptrs_size     = size_bytes;
            q             = stream;
            was_allocated = true;
        }
        *out_Q_ptrs = Q_ptrs_buf;
        *out_K_ptrs = K_ptrs_buf;
        *out_S_ptrs = S_ptrs_buf;
        *out_V_ptrs = V_ptrs_buf;
        *out_O_ptrs = O_ptrs_buf;
        return was_allocated;
    }
};
