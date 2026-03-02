#pragma once
#include "common.hpp"

struct ggml_backend_sycl_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;

    ggml_backend_sycl_buffer_context(int device, void * dev_ptr, queue_ptr stream);
    ~ggml_backend_sycl_buffer_context();
};

struct ggml_backend_sycl_buffer_type_context {
    int device;
    std::string name;
    queue_ptr stream;
};

const char * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);
ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type();
bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer);
bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer);
bool ggml_backend_buft_is_sycl_split(ggml_backend_buffer_type_t buft);
int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split);
struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

void dev2dev_memcpy(sycl::queue &q_dst, sycl::queue &q_src, void *ptr_dst, const void *ptr_src, size_t size, int dev_src = -1, int dev_dst = -1);
