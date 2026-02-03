#pragma once
#include "common.hpp"

// Kernel wrappers
void ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_set_peer_access(const int n_tokens, int main_device);

void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

dpct::err0 ggml_sycl_cpy_tensor_2d(void *dst, const ggml_tensor *src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, queue_ptr stream);

void ggml_mul_mat_p021_f16_f32_sycl(const void *vx, const float *y, float *dst, const int ne00, const int ne01, const int ne02, const int ne12, const dpct::queue_ptr &stream);
void ggml_mul_mat_vec_nc_f16_f32_sycl(const void *vx, const float *y, float *dst, const int ne00, const int ne01, const int row_stride_x, const int ne02, const int ne12, const int channel_stride_x, const int channel_stride_y, const dpct::queue_ptr &stream);

void ggml_sycl_op_mul_mat_sycl(ggml_backend_sycl_context & ctx, const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst, const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i, float *dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size, const queue_ptr &stream);
