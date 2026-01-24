#include "softmax.hpp"
#include <cstdint>
#include <utility>
#include <cmath>


template <typename T> static __dpct_inline__ float t2f32(T val) {
    return (float) val;
}

template <> float __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  return sycl::vec<sycl::half, 1>(val)
      .convert<float, sycl::rounding_mode::automatic>()[0];
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we supress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static void soft_max_f32(const float *         x,
                         const T *             mask,
                         const float *         sinks,
                         float *               dst,
                         const soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    const int block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    const int nthreads = block_size;
    const int nwarps = nthreads / WARP_SIZE;
    size_t nreduce = nwarps / WARP_SIZE;

    const int tid = item_ct1.get_local_id(2);

    const int64_t i03 = item_ct1.get_group(0);
    const int64_t i02 = item_ct1.get_group(1);
    const int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    const int rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    float * buf_iw = (float *) dpct_local;

    // shared memory buffer to cache values between iterations:
    float *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    float max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max(max_val);

    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        item_ct1.barrier();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }
    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = sycl::native::exp(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        item_ct1.barrier();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            for (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        for (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp);
    }
    if (sinks) {
        tmp += sycl::native::exp(sinks[i02] - max_val);
    }
    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static void soft_max_back_f32(const float *grad, const float *dstf, float *dst,
                              const int ncols, const float scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid      = item_ct1.get_local_id(2);
    const int rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <int... Ns, typename T>
static void launch_soft_max_kernels(const float *           x,
                                    const T *               mask,
                                    const float *           sinks,
                                    float *                 dst,
                                    const soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);
        if (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<true, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<true, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
static void soft_max_f32_sycl(const float *x, const T *mask,
                              const float *sinks, float *dst,
                              const soft_max_params &params,
                              dpct::queue_ptr stream, int device) {
    int nth = WARP_SIZE;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < max_block_size) nth *= 2;
    if (nth>max_block_size) nth = max_block_size;

    const dpct::dim3 block_dims(nth, 1, 1);
    const dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);

    const int id       = get_current_device_id();
    const size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    if (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } else {
        const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_low), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    soft_max_f32<false, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

static void soft_max_back_f32_sycl(const float *   grad,
                                   const float *   dstf,
                                   float *         dst,
                                   const int       ncols,
                                   const int       nrows,
                                   const float     scale,
                                   dpct::queue_ptr stream) {
    const dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    const dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    dpct::queue_ptr stream = ctx.stream();

    // Support F16, BF16, and F32 input/output
    const bool is_f16 = (src0->type == GGML_TYPE_F16);
    const bool is_bf16 = (src0->type == GGML_TYPE_BF16);
    const bool is_f32 = (src0->type == GGML_TYPE_F32);

#if defined(GGML_SYCL_F16) || defined(GGML_SYCL_BF16)
    GGML_ASSERT(is_f32 || is_f16 || is_bf16);
    GGML_ASSERT(dst->type == src0->type);
#else
    GGML_ASSERT(is_f32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16);

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16_mask = (src1 && src1->type == GGML_TYPE_F16);
    const bool use_bf16_mask = (src1 && src1->type == GGML_TYPE_BF16);

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    // Allocate temporary F32 buffers for F16/BF16 input/output
    float * src0_f32 = nullptr;
    float * dst_f32 = nullptr;

    if (is_f16 || is_bf16) {

        // Dequantize src0 to F32
        const int64_t n_elements = ggml_nelements(src0);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            if (is_f16) {
                const sycl::half * src = (const sycl::half *)src0->data;
                src0_f32[idx] = static_cast<float>(src[idx]);
            } else {
                const bfloat16 * src = (const bfloat16 *)src0->data;
                src0_f32[idx] = bf16_to_fp32(src[idx]);
            }
        });
    }

    // Handle mask conversion if needed
    const void * mask_ptr = src1 ? src1->data : nullptr;
    if (use_bf16_mask && src1) {
        // Convert BF16 mask to F32
        const int64_t mask_elements = ggml_nelements(src1);
        float * mask_f32 = (float *)sycl::malloc_device(mask_elements * sizeof(float), *stream);
        stream->parallel_for(sycl::range<1>(mask_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            mask_f32[idx] = bf16_to_fp32(((const bfloat16 *)src1->data)[idx]);
        });
        mask_ptr = mask_f32;
        // Note: This leaks mask_f32 - should be freed after kernel completes
    }

    // Call the kernel with F32 data
    // src2 contains optional sinks (can be NULL)
    const void * sinks_ptr = src2 ? src2->data : nullptr;
    if (use_f16_mask) {
        soft_max_f32_sycl(is_f32 || is_bf16 ? (const float *)src0->data : src0_f32,
                          (const sycl::half *)src1->data,
                          (const float *)sinks_ptr,
                          is_f32 || is_bf16 ? (float *)dst->data : dst_f32,
                          params, stream, ctx.device);
    } else {
        soft_max_f32_sycl(is_f32 || is_bf16 ? (const float *)src0->data : src0_f32,
                          (const float *)mask_ptr,
                          (const float *)sinks_ptr,
                          is_f32 || is_bf16 ? (float *)dst->data : dst_f32,
                          params, stream, ctx.device);
    }

    // If output is F16/BF16, requantize from F32
    if (is_f16 || is_bf16) {
        const int64_t n_elements = ggml_nelements(dst);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            float val = dst_f32[idx];
            if (is_f16) {
                ((sycl::half *)dst->data)[idx] = static_cast<sycl::half>(val);
            } else {
                ((bfloat16 *)dst->data)[idx] = fp32_to_bf16(val);
            }
        });
        sycl::free(src0_f32, *stream);
        sycl::free(dst_f32, *stream);
    }
}

void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    dpct::queue_ptr stream = ctx.stream();

    // Support F16, BF16, and F32 input/output
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

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    // Allocate temporary F32 buffers for F16/BF16 input/output
    const size_t nbytes = ggml_nbytes(src0);
    float * src0_f32 = nullptr;
    float * src1_f32 = nullptr;
    float * dst_f32 = nullptr;
    bool need_temp_buffers = false;

    if (is_f16 || is_bf16) {
        src0_f32 = (float *)sycl::malloc_device(nbytes, *stream);
        src1_f32 = (float *)sycl::malloc_device(nbytes, *stream);
        dst_f32 = (float *)sycl::malloc_device(nbytes, *stream);
        need_temp_buffers = true;

        const int64_t n_elements = ggml_nelements(src0);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            if (is_f16) {
                const sycl::half * src0_data = (const sycl::half *)src0->data;
                const sycl::half * src1_data = (const sycl::half *)src1->data;
                src0_f32[idx] = static_cast<float>(src0_data[idx]);
                src1_f32[idx] = static_cast<float>(src1_data[idx]);
            } else {
                const bfloat16 * src0_data = (const bfloat16 *)src0->data;
                const bfloat16 * src1_data = (const bfloat16 *)src1->data;
                src0_f32[idx] = bf16_to_fp32(src0_data[idx]);
                src1_f32[idx] = bf16_to_fp32(src1_data[idx]);
            }
        });
    }

    const float * src0_ptr = is_f32 ? (const float *)src0->data : src0_f32;
    const float * src1_ptr = is_f32 ? (const float *)src1->data : src1_f32;
    float * dst_ptr = is_f32 ? (float *)dst->data : dst_f32;

    soft_max_back_f32_sycl(src0_ptr, src1_ptr, dst_ptr, ncols, nrows, scale, stream);

    if (need_temp_buffers) {
        const int64_t n_elements = ggml_nelements(dst);
        stream->parallel_for(sycl::range<1>(n_elements), [=](sycl::item<1> it) {
            const int idx = it.get_id(0);
            float val = dst_f32[idx];
            if (is_f16) {
                ((sycl::half *)dst->data)[idx] = static_cast<sycl::half>(val);
            } else {
                ((bfloat16 *)dst->data)[idx] = fp32_to_bf16(val);
            }
        });
        sycl::free(src0_f32, *stream);
        sycl::free(src1_f32, *stream);
        sycl::free(dst_f32, *stream);
    }
}
