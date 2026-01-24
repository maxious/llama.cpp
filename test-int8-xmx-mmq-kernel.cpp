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

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    queue q;

    std::cout << "=== INT8 XMX MMQ Kernel Test ===" << std::endl;
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

    // Test parameters
    const int M = 64;
    const int N = 64;
    const int K = 64;

    // Tile sizes for B60 (PVC architecture)
    const int TM = 8;
    const int TN = 16;
    const int TK = 32;

    // Allocate host memory
    std::vector<int8_t> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<int32_t> h_C(M * N, 0);
    std::vector<int32_t> h_C_ref(M * N, 0);

    // Initialize matrices
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<int8_t>(i % 128);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<int8_t>(i % 128);
    }

    // Compute reference result on CPU
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += static_cast<int32_t>(h_A[m * K + k]) *
                       static_cast<int32_t>(h_B[k * N + n]);
            }
            h_C_ref[m * N + n] = sum;
        }
    }

    // Allocate device memory
    buffer<int8_t, 2> d_A(h_A.data(), range<2>(M, K));
    buffer<int8_t, 2> d_B(h_B.data(), range<2>(K, N));
    buffer<int32_t, 2> d_C(h_C.data(), range<2>(M, N));

    // Get tile configuration
    auto combinations = q.get_device().get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    int sg_size = 0;
    for (auto &comb : combinations) {
        bool has_int8_a = (comb.atype == matrix_type::sint8 ||
                          comb.atype == matrix_type::uint8);
        bool has_int8_b = (comb.btype == matrix_type::sint8 ||
                          comb.btype == matrix_type::uint8);

        if (has_int8_a && has_int8_b && comb.nsize == 16) {
            sg_size = 16;
            break;
        }
    }

    if (sg_size == 0) {
        std::cout << "Error: Could not determine sub-group size" << std::endl;
        return 1;
    }

    std::cout << "Sub-group size: " << sg_size << std::endl;
    std::cout << "Tile sizes: " << TM << "x" << TN << "x" << TK << std::endl;

    // Launch kernel
    try {
        q.submit([&](handler &cgh) {
            accessor accA{d_A, cgh, read_only};
            accessor accB{d_B, cgh, read_only};
            accessor accC{d_C, cgh, write_only};

            cgh.parallel_for(
                nd_range<2>({static_cast<size_t>(M / TM), static_cast<size_t>((N / TN) * sg_size)},
                            {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg = item_ct1.get_sub_group();

                    const int global_m = item_ct1.get_group(0) * TM;
                    const int global_n = (item_ct1.get_group(1) / sg_size) * TN;

                    joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
                    joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matB;
                    joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;

                    joint_matrix_fill(sg, matC, 0);

                    for (int k = 0; k < K; k += TK) {
                        joint_matrix_load(sg, matA,
                                         accA.template get_multi_ptr<access::decorated::no>() +
                                         global_m * K + k,
                                         K);
                        joint_matrix_load(sg, matB,
                                         accB.template get_multi_ptr<access::decorated::no>() +
                                         k * N + global_n,
                                         N);

                        joint_matrix_mad(sg, matC, matA, matB, matC);
                    }

                    joint_matrix_store(sg, matC,
                                      accC.template get_multi_ptr<access::decorated::no>() +
                                      global_m * N + global_n,
                                      N, layout::row_major);
                });
        }).wait();

        std::cout << "Kernel execution: SUCCESS" << std::endl;

    } catch (const std::exception &e) {
        std::cout << "Kernel execution: FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Verify results
    bool passed = true;
    int max_error = 0;

    for (int i = 0; i < M * N; i++) {
        int error = std::abs(h_C[i] - h_C_ref[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > 0) {
            passed = false;
        }
    }

    std::cout << "Result verification: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Max error: " << max_error << std::endl;

    if (passed) {
        std::cout << "\n=== INT8 XMX MMQ Kernel Test: PASSED ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== INT8 XMX MMQ Kernel Test: FAILED ===" << std::endl;
        return 1;
    }
}