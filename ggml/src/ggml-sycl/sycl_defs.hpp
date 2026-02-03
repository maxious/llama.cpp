#pragma once
#include "common.hpp"
#include <cstdlib>
#include <cstdio>

static inline int get_sycl_env(const char *env_name, int default_val) {
    char *user_device_string = getenv(env_name);
    int user_number = default_val;
    unsigned n;
    if (user_device_string != NULL &&
        sscanf(user_device_string, " %u", &n) == 1) {
        user_number = (int)n;
    } else {
        user_number = default_val;
    }
    return user_number;
}

inline void check_allow_gpu_index(const int device_index) {
  if (device_index >= ggml_sycl_info().device_count) {
    char error_buf[256];
    snprintf(
        error_buf,
        sizeof(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_count - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    assert(false);
  }
}

extern bool g_sycl_loaded;
void ggml_check_sycl();
extern int g_ggml_sycl_disable_graph;
extern int g_ggml_sycl_disable_dnn;
extern int g_ggml_sycl_prioritize_dmmv;
extern int g_ggml_sycl_use_async_mem_op;
