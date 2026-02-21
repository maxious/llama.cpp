#pragma once

#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

struct flash_attn_buffers {
    float * Q_buf        = nullptr;
    float * K_buf        = nullptr;
    float * V_buf        = nullptr;
    float * S_buf        = nullptr;
    float * partials_buf = nullptr;
    float * mask_buf     = nullptr;
    float * O_buf        = nullptr;
    float * l_buf        = nullptr;
    float * m_buf        = nullptr;

    float ** Q_ptrs_buf = nullptr;
    float ** K_ptrs_buf = nullptr;
    float ** S_ptrs_buf = nullptr;
    float ** V_ptrs_buf = nullptr;
    float ** O_ptrs_buf = nullptr;

    float ** h_Q_ptrs_buf = nullptr;
    float ** h_K_ptrs_buf = nullptr;
    float ** h_S_ptrs_buf = nullptr;
    float ** h_V_ptrs_buf = nullptr;
    float ** h_O_ptrs_buf = nullptr;

    size_t Q_size        = 0;
    size_t K_size        = 0;
    size_t V_size        = 0;
    size_t S_size        = 0;
    size_t partials_size = 0;
    size_t mask_size     = 0;
    size_t O_size        = 0;
    size_t l_size        = 0;
    size_t m_size        = 0;
    size_t ptrs_size     = 0;

    sycl::queue * q = nullptr;

    void set_graph_recording_mode(bool enabled) { graph_recording_mode = enabled; }

    bool is_graph_recording_mode() const { return graph_recording_mode; }

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
            if (h_Q_ptrs_buf) {
                sycl::free(h_Q_ptrs_buf, *q);
            }
            if (h_K_ptrs_buf) {
                sycl::free(h_K_ptrs_buf, *q);
            }
            if (h_S_ptrs_buf) {
                sycl::free(h_S_ptrs_buf, *q);
            }
            if (h_V_ptrs_buf) {
                sycl::free(h_V_ptrs_buf, *q);
            }
            if (h_O_ptrs_buf) {
                sycl::free(h_O_ptrs_buf, *q);
            }

            free_graph_slots(Q_graph_slots);
            free_graph_slots(K_graph_slots);
            free_graph_slots(V_graph_slots);
            free_graph_slots(S_graph_slots);
            free_graph_slots(partials_graph_slots);
            free_graph_slots(mask_graph_slots);
            free_graph_slots(O_graph_slots);
            free_graph_slots(l_graph_slots);
            free_graph_slots(m_graph_slots);

            free_graph_slots(Q_ptrs_graph_slots);
            free_graph_slots(K_ptrs_graph_slots);
            free_graph_slots(S_ptrs_graph_slots);
            free_graph_slots(V_ptrs_graph_slots);
            free_graph_slots(O_ptrs_graph_slots);
            free_graph_slots(h_Q_ptrs_graph_slots);
            free_graph_slots(h_K_ptrs_graph_slots);
            free_graph_slots(h_S_ptrs_graph_slots);
            free_graph_slots(h_V_ptrs_graph_slots);
            free_graph_slots(h_O_ptrs_graph_slots);
        }
    }

    float * get_Q(size_t size_bytes, sycl::queue * stream) {
        if (graph_recording_mode) {
            return get_graph_usm_slot(Q_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(K_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(V_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(S_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(partials_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(mask_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(O_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(l_graph_slots, size_bytes, stream, false);
        }
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
        if (graph_recording_mode) {
            return get_graph_usm_slot(m_graph_slots, size_bytes, stream, false);
        }
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

    bool get_ptrs(size_t        size_bytes,
                  sycl::queue * stream,
                  float ***     out_Q_ptrs,
                  float ***     out_K_ptrs,
                  float ***     out_S_ptrs,
                  float ***     out_V_ptrs,
                  float ***     out_O_ptrs,
                  float ***     out_h_Q_ptrs = nullptr,
                  float ***     out_h_K_ptrs = nullptr,
                  float ***     out_h_S_ptrs = nullptr,
                  float ***     out_h_V_ptrs = nullptr,
                  float ***     out_h_O_ptrs = nullptr) {
        bool was_allocated = false;

        if (graph_recording_mode) {
            bool allocated_now = false;
            *out_Q_ptrs        = get_graph_usm_slot(Q_ptrs_graph_slots, size_bytes, stream, false, &allocated_now);
            was_allocated      = was_allocated || allocated_now;
            *out_K_ptrs        = get_graph_usm_slot(K_ptrs_graph_slots, size_bytes, stream, false, &allocated_now);
            was_allocated      = was_allocated || allocated_now;
            *out_S_ptrs        = get_graph_usm_slot(S_ptrs_graph_slots, size_bytes, stream, false, &allocated_now);
            was_allocated      = was_allocated || allocated_now;
            *out_V_ptrs        = get_graph_usm_slot(V_ptrs_graph_slots, size_bytes, stream, false, &allocated_now);
            was_allocated      = was_allocated || allocated_now;
            *out_O_ptrs        = get_graph_usm_slot(O_ptrs_graph_slots, size_bytes, stream, false, &allocated_now);
            was_allocated      = was_allocated || allocated_now;
            if (out_h_Q_ptrs) {
                *out_h_Q_ptrs = get_graph_usm_slot(h_Q_ptrs_graph_slots, size_bytes, stream, true, &allocated_now);
                was_allocated = was_allocated || allocated_now;
            }
            if (out_h_K_ptrs) {
                *out_h_K_ptrs = get_graph_usm_slot(h_K_ptrs_graph_slots, size_bytes, stream, true, &allocated_now);
                was_allocated = was_allocated || allocated_now;
            }
            if (out_h_S_ptrs) {
                *out_h_S_ptrs = get_graph_usm_slot(h_S_ptrs_graph_slots, size_bytes, stream, true, &allocated_now);
                was_allocated = was_allocated || allocated_now;
            }
            if (out_h_V_ptrs) {
                *out_h_V_ptrs = get_graph_usm_slot(h_V_ptrs_graph_slots, size_bytes, stream, true, &allocated_now);
                was_allocated = was_allocated || allocated_now;
            }
            if (out_h_O_ptrs) {
                *out_h_O_ptrs = get_graph_usm_slot(h_O_ptrs_graph_slots, size_bytes, stream, true, &allocated_now);
                was_allocated = was_allocated || allocated_now;
            }
            return was_allocated;
        }

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
            if (h_Q_ptrs_buf) {
                sycl::free(h_Q_ptrs_buf, *q);
            }
            if (h_K_ptrs_buf) {
                sycl::free(h_K_ptrs_buf, *q);
            }
            if (h_S_ptrs_buf) {
                sycl::free(h_S_ptrs_buf, *q);
            }
            if (h_V_ptrs_buf) {
                sycl::free(h_V_ptrs_buf, *q);
            }
            if (h_O_ptrs_buf) {
                sycl::free(h_O_ptrs_buf, *q);
            }
            Q_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            K_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            S_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            V_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            O_ptrs_buf    = (float **) sycl::malloc_device(size_bytes, *stream);
            h_Q_ptrs_buf  = (float **) sycl::malloc_host(size_bytes, *stream);
            h_K_ptrs_buf  = (float **) sycl::malloc_host(size_bytes, *stream);
            h_S_ptrs_buf  = (float **) sycl::malloc_host(size_bytes, *stream);
            h_V_ptrs_buf  = (float **) sycl::malloc_host(size_bytes, *stream);
            h_O_ptrs_buf  = (float **) sycl::malloc_host(size_bytes, *stream);
            ptrs_size     = size_bytes;
            q             = stream;
            was_allocated = true;
        }
        *out_Q_ptrs = Q_ptrs_buf;
        *out_K_ptrs = K_ptrs_buf;
        *out_S_ptrs = S_ptrs_buf;
        *out_V_ptrs = V_ptrs_buf;
        *out_O_ptrs = O_ptrs_buf;
        if (out_h_Q_ptrs) {
            *out_h_Q_ptrs = h_Q_ptrs_buf;
        }
        if (out_h_K_ptrs) {
            *out_h_K_ptrs = h_K_ptrs_buf;
        }
        if (out_h_S_ptrs) {
            *out_h_S_ptrs = h_S_ptrs_buf;
        }
        if (out_h_V_ptrs) {
            *out_h_V_ptrs = h_V_ptrs_buf;
        }
        if (out_h_O_ptrs) {
            *out_h_O_ptrs = h_O_ptrs_buf;
        }
        return was_allocated;
    }

  private:
    template <typename T> struct graph_usm_slot {
        T      ptr  = nullptr;
        size_t size = 0;
    };

    template <typename T>
    T get_graph_usm_slot(std::vector<graph_usm_slot<T>> & slots,
                         size_t                           size_bytes,
                         sycl::queue *                    stream,
                         bool                             host,
                         bool *                           was_allocated = nullptr) {
        if (was_allocated) {
            *was_allocated = false;
        }
        for (auto & slot : slots) {
            if (slot.ptr != nullptr && slot.size >= size_bytes) {
                q = stream;
                return slot.ptr;
            }
        }

        T ptr = host ? (T) sycl::malloc_host(size_bytes, *stream) : (T) sycl::malloc_device(size_bytes, *stream);
        slots.push_back({ ptr, size_bytes });
        q = stream;
        if (was_allocated) {
            *was_allocated = true;
        }
        return ptr;
    }

    template <typename T> void free_graph_slots(std::vector<graph_usm_slot<T>> & slots) {
        for (auto & slot : slots) {
            if (slot.ptr) {
                sycl::free(slot.ptr, *q);
            }
        }
        slots.clear();
    }

    bool graph_recording_mode = false;

    std::vector<graph_usm_slot<float *>> Q_graph_slots;
    std::vector<graph_usm_slot<float *>> K_graph_slots;
    std::vector<graph_usm_slot<float *>> V_graph_slots;
    std::vector<graph_usm_slot<float *>> S_graph_slots;
    std::vector<graph_usm_slot<float *>> partials_graph_slots;
    std::vector<graph_usm_slot<float *>> mask_graph_slots;
    std::vector<graph_usm_slot<float *>> O_graph_slots;
    std::vector<graph_usm_slot<float *>> l_graph_slots;
    std::vector<graph_usm_slot<float *>> m_graph_slots;

    std::vector<graph_usm_slot<float **>> Q_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> K_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> S_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> V_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> O_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> h_Q_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> h_K_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> h_S_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> h_V_ptrs_graph_slots;
    std::vector<graph_usm_slot<float **>> h_O_ptrs_graph_slots;
};
