//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "mmq_xmx_int8.hpp"

#include "mmq.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// Get the N_SG setting (subgroups per workgroup, default 8)
static int get_xmx_int8_nsg() {
    static int nsg = -1;
    if (nsg < 0) {
        const char * env = getenv("GGML_SYCL_XMX_INT8_NSG");
        nsg = env ? atoi(env) : 8;
        if (nsg != 0 && nsg != 1 && nsg != 2 && nsg != 4 && nsg != 8) {
            fprintf(stderr, "ggml_sycl: GGML_SYCL_XMX_INT8_NSG=%d invalid, must be 0,1,2,4,8. Using 4.\n", nsg);
            nsg = 4;
        }
    }
    return nsg;
}

// Get TILES_N setting (how many TN-wide N tiles per subgroup, default 2)
static int get_xmx_int8_tiles_n() {
    static int tn = -1;
    if (tn < 0) {
        const char * env = getenv("GGML_SYCL_XMX_INT8_TILES_N");
        tn = env ? atoi(env) : 2;
        if (tn != 0 && tn != 1 && tn != 2 && tn != 4 && tn != 8) {
            fprintf(stderr, "ggml_sycl: GGML_SYCL_XMX_INT8_TILES_N=%d invalid, must be 0,1,2,4,8. Using 4.\n", tn);
            tn = 4;
        }
    }
    return tn;
}

// Helper to launch q8_0 kernel (col_major B, multi-subgroup)
template <int N_SG, int TILES_N>
static void launch_q8_0(const char * src0_dd_i, const char * src1_ddq_i, float * dst_dd_i,
                        int64_t K, int64_t K_padded, int64_t M, int64_t N, int ldc,
                        const dpct::queue_ptr & stream) {
    constexpr int TM = 8, TN = 16, TK = 32;
    const int64_t nblocks_m = (M + TM * N_SG - 1) / (TM * N_SG);
    const int64_t nblocks_n = (N + TN * TILES_N - 1) / (TN * TILES_N);

    // SLM: B[TILES_N*TN*TK] + N_SG*A[TM*TK] + N_SG*C[TM*TN*4]
    const int slm_bytes = TILES_N * TN * TK + N_SG * TM * TK + N_SG * TM * TN * (int) sizeof(int32_t);

    stream->submit([&](handler & cgh) {
        sycl::local_accessor<int8_t, 1> slm(range<1>(slm_bytes), cgh);
        cgh.parallel_for(
            nd_range<2>({ static_cast<size_t>(nblocks_m * N_SG), static_cast<size_t>(nblocks_n * 16) },
                        { static_cast<size_t>(N_SG), static_cast<size_t>(16) }),
            [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                mmq_q8_0_xmx_kernel<TM, TN, TK, N_SG, TILES_N>(
                    (const block_q8_0 *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i,
                    K, K_padded, M, N, ldc, item_ct1,
                    slm.get_multi_ptr<access::decorated::no>().get());
            });
    });
}

// Dispatch for a given N_SG, switching on TILES_N
template <int N_SG>
static bool try_launch_q8_0(int tiles_n, const char * src0_dd_i, const char * src1_ddq_i,
                            float * dst_dd_i, int64_t K, int64_t K_padded, int64_t M,
                            int64_t N, int ldc, const dpct::queue_ptr & stream) {
    switch (tiles_n) {
        case 1: launch_q8_0<N_SG, 1>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 2: launch_q8_0<N_SG, 2>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 4: launch_q8_0<N_SG, 4>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 8: launch_q8_0<N_SG, 8>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        default: return false;
    }
}

// Launch helpers for Q4_K
template <int N_SG, int TILES_N>
static void launch_q4_K(const char * src0_dd_i, const char * src1_ddq_i, float * dst_dd_i,
                        int64_t K, int64_t K_padded, int64_t M, int64_t N, int ldc,
                        const dpct::queue_ptr & stream) {
    constexpr int TM = 8, TN = 16, TK = 32;
    const int64_t nblocks_m = (M + TM * N_SG - 1) / (TM * N_SG);
    const int64_t nblocks_n = (N + TN * TILES_N - 1) / (TN * TILES_N);
    const int slm_bytes = TILES_N * TN * TK + N_SG * TM * TK + N_SG * TM * TN * (int) sizeof(int32_t);

    stream->submit([&](handler & cgh) {
        sycl::local_accessor<int8_t, 1> slm(range<1>(slm_bytes), cgh);
        cgh.parallel_for(
            nd_range<2>({ static_cast<size_t>(nblocks_m * N_SG), static_cast<size_t>(nblocks_n * 16) },
                        { static_cast<size_t>(N_SG), static_cast<size_t>(16) }),
            [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                mmq_q4_K_xmx_kernel<TM, TN, TK, N_SG, TILES_N>(
                    (const block_q4_K *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i,
                    K, K_padded, M, N, ldc, item_ct1,
                    slm.get_multi_ptr<access::decorated::no>().get());
            });
    });
}

template <int N_SG>
static bool try_launch_q4_K(int tiles_n, const char * src0_dd_i, const char * src1_ddq_i,
                            float * dst_dd_i, int64_t K, int64_t K_padded, int64_t M,
                            int64_t N, int ldc, const dpct::queue_ptr & stream) {
    switch (tiles_n) {
        case 1: launch_q4_K<N_SG, 1>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 2: launch_q4_K<N_SG, 2>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 4: launch_q4_K<N_SG, 4>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 8: launch_q4_K<N_SG, 8>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        default: return false;
    }
}

// Launch helpers for Q5_K
template <int N_SG, int TILES_N>
static void launch_q5_K(const char * src0_dd_i, const char * src1_ddq_i, float * dst_dd_i,
                        int64_t K, int64_t K_padded, int64_t M, int64_t N, int ldc,
                        const dpct::queue_ptr & stream) {
    constexpr int TM = 8, TN = 16, TK = 32;
    const int64_t nblocks_m = (M + TM * N_SG - 1) / (TM * N_SG);
    const int64_t nblocks_n = (N + TN * TILES_N - 1) / (TN * TILES_N);
    const int slm_bytes = TILES_N * TN * TK + N_SG * TM * TK + N_SG * TM * TN * (int) sizeof(int32_t);

    stream->submit([&](handler & cgh) {
        sycl::local_accessor<int8_t, 1> slm(range<1>(slm_bytes), cgh);
        cgh.parallel_for(
            nd_range<2>({ static_cast<size_t>(nblocks_m * N_SG), static_cast<size_t>(nblocks_n * 16) },
                        { static_cast<size_t>(N_SG), static_cast<size_t>(16) }),
            [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                mmq_q5_K_xmx_kernel<TM, TN, TK, N_SG, TILES_N>(
                    (const block_q5_K *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i,
                    K, K_padded, M, N, ldc, item_ct1,
                    slm.get_multi_ptr<access::decorated::no>().get());
            });
    });
}

template <int N_SG>
static bool try_launch_q5_K(int tiles_n, const char * src0_dd_i, const char * src1_ddq_i,
                            float * dst_dd_i, int64_t K, int64_t K_padded, int64_t M,
                            int64_t N, int ldc, const dpct::queue_ptr & stream) {
    switch (tiles_n) {
        case 1: launch_q5_K<N_SG, 1>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 2: launch_q5_K<N_SG, 2>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 4: launch_q5_K<N_SG, 4>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        case 8: launch_q5_K<N_SG, 8>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); return true;
        default: return false;
    }
}

void ggml_sycl_op_mul_mat_q_xmx_int8(ggml_backend_sycl_context & ctx,
                                     const ggml_tensor *         src0,
                                     const ggml_tensor *         src1,
                                     ggml_tensor *               dst,
                                     const char *                src0_dd_i,
                                     const float *               src1_ddf_i,
                                     const char *                src1_ddq_i,
                                     float *                     dst_dd_i,
                                     const int64_t               row_low,
                                     const int64_t               row_high,
                                     const int64_t               src1_ncols,
                                     const int64_t               src1_padded_row_size,
                                     const dpct::queue_ptr &     stream) {
    static bool first_call = true;
    if (first_call) {
        fprintf(stderr, "ggml_sycl: Using INT8 XMX MMQ kernel for src0->type=%d (N_SG=%d, TILES_N=%d)\n",
                (int) src0->type, get_xmx_int8_nsg(), get_xmx_int8_tiles_n());
        first_call = false;
    }

    // Check if device supports INT8 XMX
    if (!has_int8_xmx_support(stream)) {
        ggml_sycl_op_mul_mat_q(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low, row_high,
                               src1_ncols, src1_padded_row_size, stream);
        return;
    }

    // Get optimal tile configuration for this device
    auto      tile_config = get_int8_xmx_tile_config(stream);
    const int TM          = tile_config.TM;
    const int TN          = tile_config.TN;
    const int TK          = tile_config.TK;

    // Get matrix dimensions
    const int64_t ne00 = src0->ne[0];
    const int64_t ne0  = dst->ne[0];

    const int64_t M        = row_high - row_low;
    const int64_t N        = src1_ncols;
    const int64_t K        = ne00;
    const int64_t K_padded = src1_padded_row_size;
    const int     ldc      = ne0;

    const int nsg      = get_xmx_int8_nsg();
    const int tiles_n  = get_xmx_int8_tiles_n();

    // Use optimized multi-subgroup kernel for Q8_0 when nsg > 1
    if (nsg > 1 && TM == 8 && TN == 16 && TK == 32 && src0->type == GGML_TYPE_Q8_0) {
        if (tiles_n > 0) {
            bool ok = false;
            switch (nsg) {
                case 2: ok = try_launch_q8_0<2>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
                case 4: ok = try_launch_q8_0<4>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
                case 8: ok = try_launch_q8_0<8>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
            }
            if (ok) return;
        }
    }

    // Use optimized multi-subgroup kernel for Q4_K
    if (nsg > 1 && TM == 8 && TN == 16 && TK == 32 && src0->type == GGML_TYPE_Q4_K) {
        if (tiles_n > 0) {
            bool ok = false;
            switch (nsg) {
                case 2: ok = try_launch_q4_K<2>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
                case 4: ok = try_launch_q4_K<4>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
                case 8: ok = try_launch_q4_K<8>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
            }
            if (ok) return;
        }
    }

    // Use optimized multi-subgroup kernel for Q5_K
    if (nsg > 1 && TM == 8 && TN == 16 && TK == 32 && src0->type == GGML_TYPE_Q5_K) {
        if (tiles_n > 0) {
            bool ok = false;
            switch (nsg) {
                case 2: ok = try_launch_q5_K<2>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
                case 4: ok = try_launch_q5_K<4>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
                case 8: ok = try_launch_q5_K<8>(tiles_n, src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream); break;
            }
            if (ok) return;
        }
    }

    // Legacy single-subgroup path for other quant types
    const int64_t nblocks_m = (M + TM - 1) / TM;
    const int64_t nblocks_n = (N + TN - 1) / TN;
    const int sg_size = 16;
    const int slm_size = TM * TN + (TM * TK + TK * TN + 3) / 4;

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler & cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(slm_size), cgh);
                    cgh.parallel_for(
                        nd_range<2>({ static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size) },
                                    { static_cast<size_t>(1), static_cast<size_t>(sg_size) }),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q4_0_xmx_kernel<8, 16, 32>(
                                (const block_q4_0 *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i, K, K_padded,
                                M, N, ldc, item_ct1, slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q4_1:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler & cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(slm_size), cgh);
                    cgh.parallel_for(
                        nd_range<2>({ static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size) },
                                    { static_cast<size_t>(1), static_cast<size_t>(sg_size) }),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q4_1_xmx_kernel<8, 16, 32>(
                                (const block_q4_1 *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i, K, K_padded,
                                M, N, ldc, item_ct1, slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q5_0:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler & cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(slm_size), cgh);
                    cgh.parallel_for(
                        nd_range<2>({ static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size) },
                                    { static_cast<size_t>(1), static_cast<size_t>(sg_size) }),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q5_0_xmx_kernel<8, 16, 32>(
                                (const block_q5_0 *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i, K, K_padded,
                                M, N, ldc, item_ct1, slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q5_1:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler & cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(slm_size), cgh);
                    cgh.parallel_for(
                        nd_range<2>({ static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size) },
                                    { static_cast<size_t>(1), static_cast<size_t>(sg_size) }),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q5_1_xmx_kernel<8, 16, 32>(
                                (const block_q5_1 *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i, K, K_padded,
                                M, N, ldc, item_ct1, slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q8_1:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler & cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(slm_size), cgh);
                    cgh.parallel_for(
                        nd_range<2>({ static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size) },
                                    { static_cast<size_t>(1), static_cast<size_t>(sg_size) }),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q8_1_xmx_kernel<8, 16, 32>(
                                (const block_q8_1 *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i, K, K_padded,
                                M, N, ldc, item_ct1, slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        case GGML_TYPE_Q4_K:
            if (TM == 8 && TN == 16 && TK == 32) {
                launch_q4_K<1, 1>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream);
            }
            break;

        case GGML_TYPE_Q5_K:
            if (TM == 8 && TN == 16 && TK == 32) {
                launch_q5_K<1, 1>(src0_dd_i, src1_ddq_i, dst_dd_i, K, K_padded, M, N, ldc, stream);
            }
            break;

        case GGML_TYPE_Q6_K:
            if (TM == 8 && TN == 16 && TK == 32) {
                stream->submit([&](handler & cgh) {
                    sycl::local_accessor<int32_t, 1> slm_tile(range<1>(slm_size), cgh);
                    cgh.parallel_for(
                        nd_range<2>({ static_cast<size_t>(nblocks_m), static_cast<size_t>(nblocks_n * sg_size) },
                                    { static_cast<size_t>(1), static_cast<size_t>(sg_size) }),
                        [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                            mmq_q6_K_xmx_kernel<8, 16, 32>(
                                (const block_q6_K *) src0_dd_i, (const block_q8_1 *) src1_ddq_i, dst_dd_i, K, K_padded,
                                M, N, ldc, item_ct1, slm_tile.get_multi_ptr<access::decorated::no>().get());
                        });
                });
            }
            break;

        default:
            ggml_sycl_op_mul_mat_q(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low, row_high,
                                   src1_ncols, src1_padded_row_size, stream);
            break;
    }
}
