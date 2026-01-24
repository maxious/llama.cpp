//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    queue q;

    std::cout << "=== INT8 XMX MMQ Kernel Test (SoA Layout) ===" << std::endl;
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

    const int TM = 8;
    const int TN = 16;
    const int TK = 32;

    const int M = TM * 2;
    const int N = TN * 2;
    const int K = TK * 2;

    const int sg_size = 16;

    std::vector<int8_t> h_A_qs(M * K);
    std::vector<int8_t> h_B_qs(N * K);

    std::vector<int32_t> h_C(M * N, 0);
    std::vector<int32_t> h_C_ref(M * N, 0);

    // Initialize A (Weights, row-major)
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            h_A_qs[m * K + k] = static_cast<int8_t>(m + 2 * k);
        }
    }

    // Initialize B (Activations, N-major, so each row of N is contiguous in K)
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            h_B_qs[n * K + k] = static_cast<int8_t>( k + n );
        }
    }

    // Reference CPU
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)h_A_qs[m * K + k] * (int32_t)h_B_qs[n * K + k];
            }
            h_C_ref[m * N + n] = sum;
        }
    }

    try {
        {
            buffer<int8_t, 2> d_A_qs(h_A_qs.data(), range<2>(M, K));
            buffer<int8_t, 2> d_B_qs(h_B_qs.data(), range<2>(N, K));
            buffer<int32_t, 2> d_C(h_C.data(), range<2>(M, N));

            q.submit([&](handler &cgh) {
                accessor accA{d_A_qs, cgh, read_only};
                accessor accB{d_B_qs, cgh, read_only};
                accessor accC{d_C, cgh, write_only};

                cgh.parallel_for(
                    nd_range<2>({static_cast<size_t>(M / TM), static_cast<size_t>((N / TN) * sg_size)},
                                {static_cast<size_t>(1), static_cast<size_t>(sg_size)}),
                    [=](nd_item<2> item_ct1) [[sycl::reqd_sub_group_size(16)]] {
                        auto sg = item_ct1.get_sub_group();

                        const auto sg_startx = item_ct1.get_global_id(0) - item_ct1.get_local_id(0);
                        const auto sg_starty = item_ct1.get_global_id(1) - item_ct1.get_local_id(1);

                        joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matA;
                        joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;

                        joint_matrix_fill(sg, matC, 0);

                        for (int k = 0; k < K / TK; k++) {
                            const int8_t* pA_raw = accA.template get_multi_ptr<access::decorated::no>().get() + 
                                             (sg_startx * TM) * K + k * TK;
                            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pA_raw);

                            const int8_t* pB_raw = accB.template get_multi_ptr<access::decorated::no>().get() +
                                             ((sg_starty / sg_size) * TN) * K + k * TK;
                            auto pB = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(pB_raw);

                            joint_matrix_load(sg, matA, pA, K);
                            
                            joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matB_col;
                            joint_matrix_load(sg, matB_col, pB, K);

                            joint_matrix_mad(sg, matC, matA, matB_col, matC);
                        }

                        joint_matrix_store(sg, matC,
                                          accC.template get_multi_ptr<access::decorated::no>() +
                                          (sg_startx * TM) * N + (sg_starty / sg_size) * TN,
                                          N, layout::row_major);
                    });
            }).wait();
        }

        std::cout << "Kernel execution and sync: SUCCESS" << std::endl;

    } catch (const std::exception &e) {
        std::cout << "Kernel execution: FAILED" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    bool passed = true;
    int max_error = 0;
    for (int i = 0; i < M * N; i++) {
        int error = std::abs(h_C[i] - h_C_ref[i]);
        if (error > max_error) max_error = error;
        if (error > 0) passed = false;
    }

    std::cout << "Result verification: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Max error: " << max_error << std::endl;

    return passed ? 0 : 1;
}
