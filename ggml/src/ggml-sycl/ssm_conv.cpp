#include "ssm_conv.hpp"
#include "common.hpp"

#include <cstdio>

using namespace sycl;

static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs __attribute__((unused)),
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(global_range, local_range),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }

                const int channel = static_cast<int>(idx % d_inner);
                const int token   = static_cast<int>((idx / d_inner) % n_t);
                const int seq     = static_cast<int>(idx / (static_cast<size_t>(d_inner) * static_cast<size_t>(n_t)));

                const float *s = src_data
                    + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
                    + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
                    + static_cast<size_t>(token);

                const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

                float sumf = 0.0f;
                for (int i0 = 0; i0 < d_conv; ++i0) {
                    sumf += s[i0] * c[i0];
                }

                const size_t dst_idx =
                    static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
                    static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
                    static_cast<size_t>(channel);

                dst_data[dst_idx] = sumf;
            }
        );
    });
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

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

    const int d_conv   = src1->ne[0];
    const int ncs      = src0->ne[0];
    const int d_inner  = src0->ne[1];
    const int n_t      = dst->ne[1];
    const int n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * (int)ggml_type_size(src0->type));

    const int src_stride_inner = ncs;
    const int src_stride_seq   = ncs * d_inner;
    const int dst_stride_token = d_inner;
    const int dst_stride_seq   = d_inner * n_t;

    try {
        queue *q = ctx.stream();

        // Allocate temporary F32 buffers for F16/BF16 input/output
        const size_t nbytes0 = ggml_nbytes(src0);
        const size_t nbytes1 = ggml_nbytes(src1);
        const size_t nbytes_dst = ggml_nbytes(dst);
        float *src_data = nullptr;
        float *weights  = nullptr;
        float *dst_data = nullptr;
        bool need_temp_buffers = false;

        if (is_f16 || is_bf16) {
            src_data = (float *)sycl::malloc_device(nbytes0, *q);
            weights  = (float *)sycl::malloc_device(nbytes1, *q);
            dst_data = (float *)sycl::malloc_device(nbytes_dst, *q);
            need_temp_buffers = true;

            // Dequantize src0
            const int64_t n_elements0 = ggml_nelements(src0);
            q->parallel_for(sycl::range<1>(n_elements0), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (is_f16) {
                    const sycl::half * src = (const sycl::half *)src0->data;
                    src_data[idx] = static_cast<float>(src[idx]);
                } else {
                    const bfloat16 * src = (const bfloat16 *)src0->data;
                    src_data[idx] = bf16_to_fp32(src[idx]);
                }
            }).wait();

            // Dequantize src1
            const int64_t n_elements1 = ggml_nelements(src1);
            q->parallel_for(sycl::range<1>(n_elements1), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                if (is_f16) {
                    const sycl::half * src = (const sycl::half *)src1->data;
                    weights[idx] = static_cast<float>(src[idx]);
                } else {
                    const bfloat16 * src = (const bfloat16 *)src1->data;
                    weights[idx] = bf16_to_fp32(src[idx]);
                }
            }).wait();
        } else {
            src_data = const_cast<float *>(static_cast<const float *>(src0->data));
            weights  = const_cast<float *>(static_cast<const float *>(src1->data));
            dst_data = static_cast<float *>(dst->data);
        }

        GGML_ASSERT(src_data && weights && dst_data);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
            d_conv,
            d_inner,
            n_t,
            n_s,
            ncs,
            src_stride_inner,
            src_stride_seq,
            dst_stride_token,
            dst_stride_seq
        );

        if (need_temp_buffers) {
            const int64_t n_elements = ggml_nelements(dst);
            q->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
                const int idx = it.get_id(0);
                float val = dst_data[idx];
                if (is_f16) {
                    ((sycl::half *)dst->data)[idx] = static_cast<sycl::half>(val);
                } else {
                    ((bfloat16 *)dst->data)[idx] = fp32_to_bf16(val);
                }
            }).wait();
            sycl::free(src_data, *q);
            sycl::free(weights, *q);
            sycl::free(dst_data, *q);
        }

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}
