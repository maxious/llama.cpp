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

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    queue q;

    std::cout << "=== INT8 XMX Support Detection ===" << std::endl;
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
    std::cout << "Vendor: " << q.get_device().get_info<info::device::vendor>() << std::endl;
    std::cout << "Driver version: " << q.get_device().get_info<info::device::driver_version>() << std::endl;

    // Check if device supports matrix extension
    if (!q.get_device().has(sycl::aspect::ext_intel_matrix)) {
        std::cout << "Matrix extension: NOT SUPPORTED" << std::endl;
        return 1;
    }

    std::cout << "Matrix extension: SUPPORTED" << std::endl;

    // Get matrix combinations
    auto combinations = q.get_device().get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    std::cout << "Number of matrix combinations: " << combinations.size() << std::endl;

    bool has_int8_support = false;

    for (size_t i = 0; i < combinations.size(); i++) {
        auto &comb = combinations[i];

        // Check for INT8 support in both operands
        bool has_int8_a = (comb.atype == matrix_type::sint8 ||
                          comb.atype == matrix_type::uint8);
        bool has_int8_b = (comb.btype == matrix_type::sint8 ||
                          comb.btype == matrix_type::uint8);

        if (has_int8_a && has_int8_b) {
            has_int8_support = true;

            std::cout << "\nINT8 Matrix Combination #" << i << ":" << std::endl;
            std::cout << "  Tile sizes: " << comb.msize << "x" << comb.nsize << "x" << comb.ksize << std::endl;
            std::cout << "  Max tile sizes: " << comb.max_msize << "x" << comb.max_nsize << "x" << comb.max_ksize << std::endl;
            std::cout << "  Operand A type: " << (comb.atype == matrix_type::sint8 ? "sint8" : "uint8") << std::endl;
            std::cout << "  Operand B type: " << (comb.btype == matrix_type::sint8 ? "sint8" : "uint8") << std::endl;
            std::cout << "  Accumulator type: " << (comb.ctype == matrix_type::sint32 ? "sint32" : "uint32") << std::endl;

            // Determine architecture
            if (comb.nsize == 16) {
                std::cout << "  Architecture: Intel PVC/B60 (8x16x32 tiles)" << std::endl;
            } else if (comb.nsize == 8) {
                std::cout << "  Architecture: Intel DG2 (8x8x32 tiles)" << std::endl;
            } else if (comb.nsize == 0) {
                std::cout << "  Architecture: Intel AMX (16x16x64 tiles)" << std::endl;
            }
        }
    }

    if (has_int8_support) {
        std::cout << "\n=== INT8 XMX is SUPPORTED ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== INT8 XMX is NOT SUPPORTED ===" << std::endl;
        return 1;
    }
}