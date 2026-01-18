#include "outprod.hpp"

void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor *src0 = dst->src[0];
    const ggml_tensor *src1 = dst->src[1];

    // Support F16, BF16, and F32
    const bool is_f16 = (src0->type == GGML_TYPE_F16);
    const bool is_bf16 = (src0->type == GGML_TYPE_BF16);
    const bool is_f32 = (src0->type == GGML_TYPE_F32);

#if defined(GGML_SYCL_F16) || defined(GGML_SYCL_BF16)
    GGML_ASSERT(is_f32 || is_f16 || is_bf16);
    GGML_ASSERT(src1->type == src0->type);
    GGML_ASSERT(dst->type == src0->type);
#else
    GGML_ASSERT(is_f32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    // Get SYCL queue
    dpct::queue_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols

    // Allocate temporary F32 buffers for F16/BF16 input/output
    const size_t nbytes0 = ggml_nbytes(src0);
    const size_t nbytes1 = ggml_nbytes(src1);
    const size_t nbytes_dst = ggml_nbytes(dst);
    const float* src0_d = nullptr;
    const float* src1_d = nullptr;
    float* dst_d = nullptr;
    bool need_temp_buffers = false;

    if (is_f16 || is_bf16) {
        float * src0_f32 = (float *)sycl::malloc_device(nbytes0, *stream);
        float * src1_f32 = (float *)sycl::malloc_device(nbytes1, *stream);
        float * dst_f32 = (float *)sycl::malloc_device(nbytes_dst, *stream);
        need_temp_buffers = true;

        // Dequantize src0
        const int64_t n_elements0 = ggml_nelements(src0);
        stream->parallel_for(sycl::range<1>(n_elements0), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            if (is_f16) {
                const sycl::half * src = (const sycl::half *)src0->data;
                src0_f32[idx] = static_cast<float>(src[idx]);
            } else {
                const bfloat16 * src = (const bfloat16 *)src0->data;
                src0_f32[idx] = bf16_to_fp32(src[idx]);
            }
        }).wait();

        // Dequantize src1
        const int64_t n_elements1 = ggml_nelements(src1);
        stream->parallel_for(sycl::range<1>(n_elements1), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            if (is_f16) {
                const sycl::half * src = (const sycl::half *)src1->data;
                src1_f32[idx] = static_cast<float>(src[idx]);
            } else {
                const bfloat16 * src = (const bfloat16 *)src1->data;
                src1_f32[idx] = bf16_to_fp32(src[idx]);
            }
        }).wait();

        src0_d = src0_f32;
        src1_d = src1_f32;
        dst_d = dst_f32;
    } else {
        src0_d = (const float*)src0->data;
        src1_d = (const float*)src1->data;
        dst_d = (float*)dst->data;
    }

    // GEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Handle transposition of src1
    const bool src1_T = ggml_is_transposed(src1);
    const oneapi::mkl::transpose src1_op = src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);

    try {
        // Perform matrix multiplication using oneMKL GEMM
        oneapi::mkl::blas::column_major::gemm(*stream, oneapi::mkl::transpose::nontrans, src1_op,
                                               ne0, ne1, ne01, alpha, src0_d, ne00, src1_d, ldb, beta, dst_d, ne0);
    }
    catch (sycl::exception const& exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(false);
    }

    if (need_temp_buffers) {
        const int64_t n_elements = ggml_nelements(dst);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            float val = dst_d[idx];
            if (is_f16) {
                ((sycl::half *)dst->data)[idx] = static_cast<sycl::half>(val);
            } else {
                ((bfloat16 *)dst->data)[idx] = fp32_to_bf16(val);
            }
        }).wait();
        sycl::free(const_cast<float *>(src0_d), *stream);
        sycl::free(const_cast<float *>(src1_d), *stream);
        sycl::free(dst_d, *stream);
    }
}
