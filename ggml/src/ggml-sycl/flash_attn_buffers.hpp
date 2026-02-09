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

    // Track allocated sizes in bytes
    size_t Q_size        = 0;
    size_t K_size        = 0;
    size_t V_size        = 0;
    size_t S_size        = 0;
    size_t partials_size = 0;
    size_t mask_size     = 0;

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
};
