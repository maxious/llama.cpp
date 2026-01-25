// Minimal test to compile FLASH_ATTN kernels and detect IGC crashes
// This isolates kernel compilation from full test execution

#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <cmath>

// Test parameters from failing test
constexpr int64_t DQK = 64;
constexpr int64_t DV = 64;
constexpr int64_t N = 512;
constexpr int64_t n_heads = 4;
constexpr int64_t N_kv = 512;
constexpr int64_t n_kv_heads = 4;
constexpr int64_t gqa_ratio = 1;

constexpr int FATTN_BLOCK_R = 32;
constexpr int FATTN_BLOCK_C = 32;

// Simplified kernel structure to test compilation
template<int64_t DQK, int64_t DV>
void test_flash_attn_kernel_compilation(sycl::queue& q) {
    try {
        // Create a minimal kernel that mimics the flash attention structure
        auto kernel = [](sycl::nd_item<2> it) {
            const int Br = FATTN_BLOCK_R;
            const int Bc = FATTN_BLOCK_C;

            const int Tr = (N + Br - 1) / Br;
            const int Tc = (N_kv + Bc - 1) / Bc;

            sycl::range<2> global(Br * Tr, Tc * n_heads);
            sycl::range<2> local(Br, 1);

            auto group = it.get_group();
            int group_id_i = group.get_group_id(0);
            int group_id_j = group.get_group_id(1);

            int head_idx = group_id_j;
            int kv_head_idx = head_idx / gqa_ratio;

            int row0 = group_id_i * Br;
            int col0 = (group_id_j % Tc) * Bc;

            if (row0 >= (int) N || col0 >= (int) N_kv) {
                return;
            }

            // Local memory allocations (mimicking the real kernel)
            float Qtile[Br * DQK];
            float Ktile[Bc * DQK];
            float Vtile[Bc * DV];
            float Stile[Br * Bc];
            float Ptile[Br * Bc];
            float m_local[Br];
            float l_local[Br];

            // Simulate some computation
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < DQK; ++j) {
                    Qtile[i * DQK + j] = 0.0f;
                }
            }

            it.barrier(sycl::access::fence_space::local_space);

            // More computation
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < Bc; ++j) {
                    Stile[i * Bc + j] = 0.0f;
                }
            }

            it.barrier(sycl::access::fence_space::local_space);

            // Softmax-like computation
            for (int i = 0; i < Br; ++i) {
                float max_val = -INFINITY;
                for (int j = 0; j < Bc; ++j) {
                    max_val = sycl::fmax(max_val, Stile[i * Bc + j]);
                }
                m_local[i] = max_val;

                float sum = 0.0f;
                for (int j = 0; j < Bc; ++j) {
                    float exp_val = sycl::exp(Stile[i * Bc + j] - max_val);
                    Ptile[i * Bc + j] = exp_val;
                    sum += exp_val;
                }
                l_local[i] = sum;
            }

            it.barrier(sycl::access::fence_space::local_space);

            // Output computation
            for (int i = 0; i < Br; ++i) {
                for (int j = 0; j < DV; ++j) {
                    float val = 0.0f;
                    for (int k = 0; k < Bc; ++k) {
                        val += Ptile[i * Bc + k] * Vtile[k * DV + j];
                    }
                    // Normalize
                    if (l_local[i] > 0.0f) {
                        val /= l_local[i];
                    }
                }
            }
        };

        // Submit kernel to trigger compilation (use minimal range)
        const int Br = FATTN_BLOCK_R;
        const int Bc = FATTN_BLOCK_C;
        const int Tr = (N + Br - 1) / Br;
        const int Tc = (N_kv + Bc - 1) / Bc;

        sycl::range<2> global(Br * Tr, Tc * n_heads);
        sycl::range<2> local(Br, 1);

        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<2>(global, local), kernel);
        });

        q.wait(); // Force compilation to complete

        std::cout << "✓ Kernel compilation succeeded for DQK=" << DQK << ", DV=" << DV << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "✗ Exception during kernel compilation: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    try {
        std::cout << "Testing FLASH_ATTN kernel compilation..." << std::endl;
        std::cout << "Parameters: DQK=" << DQK << ", DV=" << DV << ", N=" << N
                  << ", n_heads=" << n_heads << ", N_kv=" << N_kv << std::endl;

        // Get SYCL device
        auto devices = sycl::device::get_devices();
        sycl::device device;
        bool found_gpu = false;
        for (const auto& dev : devices) {
            if (dev.is_gpu()) {
                device = dev;
                found_gpu = true;
                break;
            }
        }
        if (!found_gpu) {
            throw std::runtime_error("No GPU devices found");
        }
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;

        // Create queue
        sycl::queue q(device, sycl::property::queue::enable_profiling{});

        // Test kernel compilation for different head sizes
        std::cout << "\nTesting kernel compilation for various head sizes:\n" << std::endl;

        // Test the failing case (64, 64)
        test_flash_attn_kernel_compilation<64, 64>(q);

        // Test other common sizes
        test_flash_attn_kernel_compilation<32, 32>(q);
        test_flash_attn_kernel_compilation<40, 40>(q);
        test_flash_attn_kernel_compilation<80, 80>(q);
        test_flash_attn_kernel_compilation<128, 128>(q);

        std::cout << "\n✓ All kernel compilation tests passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed: " << e.what() << std::endl;
        std::cerr << "\nThis indicates an IGC compiler issue." << std::endl;
        std::cerr << "Please report this to Intel oneAPI support with:" << std::endl;
        std::cerr << "  - IGC version from: icpx --version" << std::endl;
        std::cerr << "  - Device info from: sycl-ls" << std::endl;
        std::cerr << "  - Error message above" << std::endl;
        return 1;
    }
}