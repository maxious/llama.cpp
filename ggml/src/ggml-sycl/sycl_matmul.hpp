#pragma once
#include "common.hpp"
typedef void (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

void ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

void ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_top_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool can_use_dequantize_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool can_use_mul_mat_vec_q(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_sycl_supports_mmq(enum ggml_type type);
bool ggml_sycl_supports_reorder_mmvq(enum ggml_type type);
bool should_reorder_tensor(ggml_backend_sycl_context& ctx, const ggml_tensor * dst);
