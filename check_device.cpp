#include <sycl/sycl.hpp>
#include <iostream>
#include <string>

int main() {
    try {
        sycl::queue q{sycl::gpu_selector_v};
        sycl::device dev = q.get_device();

        std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
        std::cout << "Vendor: " << dev.get_info<sycl::info::device::vendor>() << "\n";
        std::cout << "Vendor ID: 0x" << std::hex << dev.get_info<sycl::info::device::vendor_id>() << std::dec << "\n";

        // Check for Intel-specific aspects
        std::cout << "\nAspect checks:\n";
        try {
            bool has_eu_simd = dev.has(sycl::aspect::ext_intel_gpu_eu_simd_width);
            std::cout << "  ext_intel_gpu_eu_simd_width: " << (has_eu_simd ? "true" : "false") << "\n";
        } catch (...) {
            std::cout << "  ext_intel_gpu_eu_simd_width: NOT AVAILABLE (aspect not defined)\n";
        }

        try {
            bool has_matrix = dev.has(sycl::aspect::ext_intel_matrix);
            std::cout << "  ext_intel_matrix: " << (has_matrix ? "true" : "false") << "\n";
        } catch (...) {
            std::cout << "  ext_intel_matrix: NOT AVAILABLE (aspect not defined)\n";
        }

        // Check other SYCL extensions
        std::cout << "\nOther info:\n";
        try {
            auto matrix_info = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
            std::cout << "  Matrix combinations available: " << matrix_info.size() << "\n";
            for (const auto& comb : matrix_info) {
                std::cout << "    A:" << (int)comb.atype << " B:" << (int)comb.btype << " C:" << (int)comb.ctype
                          << " nsize=" << comb.nsize << "\n";
            }
        } catch (...) {
            std::cout << "  Matrix combinations: NOT AVAILABLE\n";
        }

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
