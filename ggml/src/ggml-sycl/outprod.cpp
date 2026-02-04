#include "outprod.hpp"

static void k_out_prod(const float * src0, const char * src1, float * dst,
                       int64_t ne00, int64_t ne01, int64_t ne10,
                       int64_t nb10, int64_t nb11,
                       const sycl::nd_item<1> & item) {
    int64_t idx = item.get_global_linear_id();
    if (idx >= ne00 * ne10) return;

    int64_t i = idx % ne00;
    int64_t j = idx / ne00;

    float sum = 0.0f;
    for (int64_t k = 0; k < ne01; ++k) {
        float v0 = src0[i + k * ne00];
        float v1 = *(const float *)(src1 + j * nb10 + k * nb11);
        sum += v0 * v1;
    }
    dst[idx] = sum;
}

void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    // Get SYCL queue
    dpct::queue_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols

    // Get data pointers
    const float* src0_d = (const float*)src0->data;
    const char* src1_d = (const char*)src1->data;
    float* dst_d = (float*)dst->data;

    int64_t total = ne00 * ne10;
    int64_t block_size = 256;
    int64_t num_blocks = (total + block_size - 1) / block_size;

    stream->parallel_for(sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> item) {
        k_out_prod(src0_d, src1_d, dst_d, ne00, ne01, ne10, nb10, nb11, item);
    });
}
