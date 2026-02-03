// SYCL GEMM Benchmark: Simple vs Tiled vs XMX vs oneMKL
// Build (with XMX + MKL):
//   icpx -fsycl -O3 -DGGML_SYCL_USE_INTEL_ONEMKL -DSYCL_EXT_ONEAPI_MATRIX \
//     tests/test-gemm-sycl.cpp -o test-gemm-sycl \
//     -lmkl_sycl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential
// Build (XMX only, no MKL):
//   icpx -fsycl -O3 -DSYCL_EXT_ONEAPI_MATRIX tests/test-gemm-sycl.cpp -o test-gemm-sycl
// Run: source /opt/intel/oneapi/setvars.sh intel64 && ./test-gemm-sycl

#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <iomanip>
#include <functional>

#ifdef GGML_SYCL_USE_INTEL_ONEMKL
#include <oneapi/mkl.hpp>
#endif

// Include XMX GEMM implementation
#include "../ggml/src/ggml-sycl/gemm_xmx.hpp"

// ============================================================================
// Tiled GEMM Implementation (standalone copy for benchmarking)
// ============================================================================

// Tile configuration - matches gemm_tiled.hpp
// Benchmark results on Arc Pro B60:
// - BK=32 slightly better than BK=16
// - 64x64 tiles with 4x4 elements/thread = 256 threads (16x16 workgroup)
// - Achieves ~3500 GFLOPS (~30% of oneMKL's ~12000 GFLOPS)
constexpr int GEMM_BM = 64;
constexpr int GEMM_BN = 64;
constexpr int GEMM_BK = 32;
constexpr int GEMM_TM = 4;
constexpr int GEMM_TN = 4;
constexpr int GEMM_WG_M = GEMM_BM / GEMM_TM;  // 16
constexpr int GEMM_WG_N = GEMM_BN / GEMM_TN;  // 16

template <int BM, int BN, int BK, int TM, int TN, bool transpose_B = true>
inline void gemm_tiled_kernel(
    sycl::nd_item<2> it,
    sycl::local_accessor<float, 1> tile_A,
    sycl::local_accessor<float, 1> tile_B,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int block_row = it.get_group(0);
    const int block_col = it.get_group(1);
    const int thread_row = it.get_local_id(0);
    const int thread_col = it.get_local_id(1);
    const int thread_id = thread_row * WG_N + thread_col;
    
    const int row_start = block_row * BM + thread_row * TM;
    const int col_start = block_col * BN + thread_col * TN;
    
    float acc[TM][TN];
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            acc[tm][tn] = 0.0f;
        }
    }
    
    constexpr int WG_SIZE = WG_M * WG_N;
    constexpr int A_TILE_SIZE = BM * BK;
    constexpr int B_TILE_SIZE = BN * BK;
    constexpr int A_LOADS_PER_THREAD = (A_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    constexpr int B_LOADS_PER_THREAD = (B_TILE_SIZE + WG_SIZE - 1) / WG_SIZE;
    
    const int num_k_tiles = (K + BK - 1) / BK;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_start = k_tile * BK;
        
        #pragma unroll
        for (int load = 0; load < A_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < A_TILE_SIZE) {
                const int tile_m = flat_idx / BK;
                const int tile_k = flat_idx % BK;
                const int global_row = block_row * BM + tile_m;
                const int global_k = k_start + tile_k;
                
                float val = 0.0f;
                if (global_row < M && global_k < K) {
                    val = A[global_row * lda + global_k];
                }
                tile_A[tile_m * BK + tile_k] = val;
            }
        }
        
        #pragma unroll
        for (int load = 0; load < B_LOADS_PER_THREAD; ++load) {
            const int flat_idx = thread_id + load * WG_SIZE;
            if (flat_idx < B_TILE_SIZE) {
                const int tile_n = flat_idx / BK;
                const int tile_k = flat_idx % BK;
                const int global_col = block_col * BN + tile_n;
                const int global_k = k_start + tile_k;
                
                float val = 0.0f;
                if constexpr (transpose_B) {
                    if (global_col < N && global_k < K) {
                        val = B[global_col * ldb + global_k];
                    }
                } else {
                    if (global_k < K && global_col < N) {
                        val = B[global_k * ldb + global_col];
                    }
                }
                tile_B[tile_n * BK + tile_k] = val;
            }
        }
        
        sycl::group_barrier(it.get_group());
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                a_reg[tm] = tile_A[(thread_row * TM + tm) * BK + k];
            }
            
            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                b_reg[tn] = tile_B[(thread_col * TN + tn) * BK + k];
            }
            
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm][tn] = sycl::fma(a_reg[tm], b_reg[tn], acc[tm][tn]);
                }
            }
        }
        
        sycl::group_barrier(it.get_group());
    }
    
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int global_row = row_start + tm;
        if (global_row >= M) continue;
        
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            const int global_col = col_start + tn;
            if (global_col >= N) continue;
            
            const int c_idx = global_row * ldc + global_col;
            if (beta == 0.0f) {
                C[c_idx] = alpha * acc[tm][tn];
            } else {
                C[c_idx] = alpha * acc[tm][tn] + beta * C[c_idx];
            }
        }
    }
}

template <bool transpose_B = true>
inline void launch_gemm_tiled(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    constexpr int BM = GEMM_BM;
    constexpr int BN = GEMM_BN;
    constexpr int BK = GEMM_BK;
    constexpr int TM = GEMM_TM;
    constexpr int TN = GEMM_TN;
    constexpr int WG_M = BM / TM;
    constexpr int WG_N = BN / TN;
    
    const int grid_m = (M + BM - 1) / BM;
    const int grid_n = (N + BN - 1) / BN;
    
    sycl::range<2> global(grid_m * WG_M, grid_n * WG_N);
    sycl::range<2> local(WG_M, WG_N);
    
    stream->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile_A(sycl::range<1>(BM * BK), cgh);
        sycl::local_accessor<float, 1> tile_B(sycl::range<1>(BN * BK), cgh);
        
        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_tiled_kernel<BM, BN, BK, TM, TN, transpose_B>(
                it, tile_A, tile_B,
                A, B, C,
                M, N, K,
                alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

// Also include the simple GEMM for comparison
template <bool transpose_B = true>
inline void gemm_kernel_simple(
    sycl::nd_item<2> it,
    const float * A, const float * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int row = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
    const int col = it.get_group(1) * it.get_local_range(1) + it.get_local_id(1);

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    if constexpr (transpose_B) {
        for (int k = 0; k < K; ++k) {
            sum += A[row * lda + k] * B[col * ldb + k];
        }
    } else {
        for (int k = 0; k < K; ++k) {
            sum += A[row * lda + k] * B[k * ldb + col];
        }
    }

    const int c_idx = row * ldc + col;
    if (beta == 0.0f) {
        C[c_idx] = alpha * sum;
    } else {
        C[c_idx] = alpha * sum + beta * C[c_idx];
    }
}

template <bool transpose_B = true>
inline void launch_gemm_simple(
    sycl::queue * stream,
    const float * A, const float * B, float * C,
    const int M, const int N, const int K,
    const float alpha, const float beta,
    const int lda, const int ldb, const int ldc
) {
    const int block_size = 16;
    sycl::range<2> global(
        (M + block_size - 1) / block_size * block_size,
        (N + block_size - 1) / block_size * block_size
    );
    sycl::range<2> local(block_size, block_size);

    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
            gemm_kernel_simple<transpose_B>(
                it, A, B, C,
                M, N, K, alpha, beta,
                lda, ldb, ldc
            );
        });
    });
}

// Helper to convert float array to half array
inline void float_to_half(const std::vector<float>& src, sycl::half* dst) {
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = sycl::half(src[i]);
    }
}

class GemmBenchmark {
public:
    sycl::queue q;
    
    GemmBenchmark() : q(sycl::gpu_selector_v, sycl::property::queue::in_order{}) {
        auto device = q.get_device();
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        std::cout << "Max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
        std::cout << "Max workgroup size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n\n";
    }
    
    void verify_correctness(int M, int N, int K) {
        std::cout << "Verifying correctness for M=" << M << " N=" << N << " K=" << K << "...\n";
        
        std::vector<float> h_A(M * K), h_B(N * K), h_C_ref(M * N), h_C_tiled(M * N);
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);
        
        // CPU reference (A * B^T)
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += h_A[m * K + k] * h_B[n * K + k];
                }
                h_C_ref[m * N + n] = sum;
            }
        }
        
        float *d_A = sycl::malloc_device<float>(M * K, q);
        float *d_B = sycl::malloc_device<float>(N * K, q);
        float *d_C = sycl::malloc_device<float>(M * N, q);
        
        q.memcpy(d_A, h_A.data(), M * K * sizeof(float)).wait();
        q.memcpy(d_B, h_B.data(), N * K * sizeof(float)).wait();
        
        launch_gemm_tiled<true>(&q, d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, K, K, N);
        q.wait();
        
        q.memcpy(h_C_tiled.data(), d_C, M * N * sizeof(float)).wait();
        
        float max_diff = 0.0f;
        int errors = 0;
        for (int i = 0; i < M * N; ++i) {
            float diff = std::abs(h_C_ref[i] - h_C_tiled[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > 1e-3f * std::abs(h_C_ref[i]) + 1e-5f) {
                if (errors < 5) {
                    std::cout << "  Mismatch at " << i << ": ref=" << h_C_ref[i] 
                              << " tiled=" << h_C_tiled[i] << " diff=" << diff << "\n";
                }
                errors++;
            }
        }
        
        std::cout << "  Max diff: " << max_diff << ", Errors: " << errors << "/" << M*N << "\n";
        std::cout << (errors == 0 ? "  Tiled PASSED\n" : "  Tiled FAILED\n");
        
#ifdef SYCL_EXT_ONEAPI_MATRIX
        // Verify XMX GEMM (uses BF16 so has lower precision)
        if (xmx_gemm_available(&q)) {
            std::vector<float> h_C_xmx(M * N);
            q.memset(d_C, 0, M * N * sizeof(float)).wait();
            
            launch_gemm_xmx<true>(&q, d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, K, K, N);
            q.wait();
            
            q.memcpy(h_C_xmx.data(), d_C, M * N * sizeof(float)).wait();
            
            float xmx_max_diff = 0.0f;
            double xmx_sum_sq_err = 0.0;
            double xmx_sum_sq_ref = 0.0;
            // BF16 accumulates error across K dimension, so use NMSE metric
            for (int i = 0; i < M * N; ++i) {
                float diff = h_C_ref[i] - h_C_xmx[i];
                xmx_max_diff = std::max(xmx_max_diff, std::abs(diff));
                xmx_sum_sq_err += diff * diff;
                xmx_sum_sq_ref += h_C_ref[i] * h_C_ref[i];
            }
            double xmx_nmse = xmx_sum_sq_err / (xmx_sum_sq_ref + 1e-10);
            // BF16 NMSE should be < 1e-4 for reasonable accuracy
            std::cout << "  XMX max diff: " << xmx_max_diff << ", NMSE: " << xmx_nmse << "\n";
            std::cout << (xmx_nmse < 1e-3 ? "  XMX PASSED\n" : "  XMX FAILED (NMSE too high)\n");
        }
#endif
        std::cout << "\n";
        
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
    }
    
    double benchmark_kernel(const std::string& name,
                           std::function<void()> kernel,
                           int warmup_iters, int bench_iters) {
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            kernel();
        }
        q.wait();
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
            kernel();
        }
        q.wait();
        auto end = std::chrono::high_resolution_clock::now();
        
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return total_ms / bench_iters;
    }
    
    void run_benchmarks(int M, int N, int K, int warmup = 5, int iters = 20) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Benchmarking M=" << M << " N=" << N << " K=" << K << " (" << iters << " iterations)\n";
        
        float *d_A = sycl::malloc_device<float>(M * K, q);
        float *d_B = sycl::malloc_device<float>(N * K, q);
        float *d_C = sycl::malloc_device<float>(M * N, q);
        
        // Initialize with random data
        std::vector<float> h_A(M * K), h_B(N * K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);
        q.memcpy(d_A, h_A.data(), M * K * sizeof(float)).wait();
        q.memcpy(d_B, h_B.data(), N * K * sizeof(float)).wait();
        
        // FLOPS calculation: 2*M*N*K (multiply-add)
        double flops = 2.0 * M * N * K;
        
        // Simple GEMM
        double simple_ms = benchmark_kernel("Simple", [&]() {
            launch_gemm_simple<true>(&q, d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, K, K, N);
        }, warmup, iters);
        double simple_gflops = flops / (simple_ms * 1e6);
        std::cout << "  Simple:  " << simple_ms << " ms (" << simple_gflops << " GFLOPS)\n";
        
        // Tiled GEMM
        double tiled_ms = benchmark_kernel("Tiled", [&]() {
            launch_gemm_tiled<true>(&q, d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, K, K, N);
        }, warmup, iters);
        double tiled_gflops = flops / (tiled_ms * 1e6);
        std::cout << "  Tiled:   " << tiled_ms << " ms (" << tiled_gflops << " GFLOPS)\n";
        std::cout << "  Speedup vs Simple: " << simple_ms / tiled_ms << "x\n";
        
#ifdef SYCL_EXT_ONEAPI_MATRIX
        // XMX GEMM (BF16 intermediate, F32 accumulator)
        if (xmx_gemm_available(&q)) {
            double xmx_ms = benchmark_kernel("XMX-BF16", [&]() {
                launch_gemm_xmx<true>(&q, d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, K, K, N);
            }, warmup, iters);
            double xmx_gflops = flops / (xmx_ms * 1e6);
            std::cout << "  XMX-BF16: " << xmx_ms << " ms (" << xmx_gflops << " GFLOPS)\n";
            std::cout << "  XMX-BF16 vs Tiled: " << tiled_ms / xmx_ms << "x\n";
            
            // F16 XMX GEMM - allocate F16 inputs
            sycl::half *d_A_f16 = sycl::malloc_device<sycl::half>(M * K, q);
            sycl::half *d_B_f16 = sycl::malloc_device<sycl::half>(N * K, q);
            
            // Convert F32 to F16 on host and copy
            std::vector<sycl::half> h_A_f16(M * K), h_B_f16(N * K);
            float_to_half(h_A, h_A_f16.data());
            float_to_half(h_B, h_B_f16.data());
            q.memcpy(d_A_f16, h_A_f16.data(), M * K * sizeof(sycl::half)).wait();
            q.memcpy(d_B_f16, h_B_f16.data(), N * K * sizeof(sycl::half)).wait();
            
            double xmx_f16_ms = benchmark_kernel("XMX-F16", [&]() {
                launch_gemm_xmx_f16<true>(&q, d_A_f16, d_B_f16, d_C, M, N, K, 1.0f, 0.0f, K, K, N);
            }, warmup, iters);
            double xmx_f16_gflops = flops / (xmx_f16_ms * 1e6);
            std::cout << "  XMX-F16:  " << xmx_f16_ms << " ms (" << xmx_f16_gflops << " GFLOPS)\n";
            std::cout << "  XMX-F16 vs XMX-BF16: " << xmx_ms / xmx_f16_ms << "x\n";
            
            sycl::free(d_A_f16, q);
            sycl::free(d_B_f16, q);
        } else {
            std::cout << "  XMX:     (not available on this device)\n";
        }
#endif
        
#ifdef GGML_SYCL_USE_INTEL_ONEMKL
        // oneMKL GEMM for comparison
        // Row-major C = A * B^T becomes column-major C^T = B * A^T
        // C[M,N] row-major with ldc=N -> column-major view is N x M with ldc=N
        // A[M,K] row-major with lda=K -> column-major view is K x M with lda=K  
        // B[N,K] row-major with ldb=K -> column-major view is K x N with ldb=K
        // We want: C = A * B^T
        // In col-major: C^T = (A*B^T)^T = B * A^T
        // gemm(B, A^T) with B as (K x N col-major = N x K row-major), A^T is transposed
        // Result: (N x M col-major) = C^T, which is C in row-major
        double mkl_ms = benchmark_kernel("oneMKL", [&]() {
            oneapi::mkl::blas::gemm(q,
                oneapi::mkl::transpose::trans,     // B (stored as N x K row-major) -> transpose for col-major view
                oneapi::mkl::transpose::nontrans,  // A (stored as M x K row-major) -> no transpose, treat as K x M col-major
                N, M, K,                           // Dimensions swapped for C^T
                1.0f, d_B, K, d_A, K,              // B first, then A
                0.0f, d_C, N);                     // Result is N x M col-major = M x N row-major
        }, warmup, iters);
        double mkl_gflops = flops / (mkl_ms * 1e6);
        std::cout << "  oneMKL:  " << mkl_ms << " ms (" << mkl_gflops << " GFLOPS)\n";
        std::cout << "  Tiled vs MKL: " << mkl_ms / tiled_ms << "x\n";
#endif
        
        std::cout << "\n";
        
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
    }
};

int main() {
    try {
        GemmBenchmark bench;
        
        // Verify correctness first
        bench.verify_correctness(64, 64, 64);
        bench.verify_correctness(128, 128, 128);
        bench.verify_correctness(256, 256, 256);
        bench.verify_correctness(127, 129, 131);  // Non-aligned sizes
        
        std::cout << "=== Performance Benchmarks ===\n\n";
        
        // Flash attention typical sizes (small M, large K)
        bench.run_benchmarks(1, 2048, 128);     // Single query, long context
        bench.run_benchmarks(4, 2048, 128);     // Small batch
        bench.run_benchmarks(32, 2048, 128);    // Medium batch
        bench.run_benchmarks(128, 2048, 128);   // Larger batch
        
        // Square matrices
        bench.run_benchmarks(256, 256, 256);
        bench.run_benchmarks(512, 512, 512);
        bench.run_benchmarks(1024, 1024, 1024);
        bench.run_benchmarks(2048, 2048, 2048);
        
        // Typical LLM MUL_MAT sizes
        bench.run_benchmarks(4096, 4096, 128);  // Large hidden dim, small K
        bench.run_benchmarks(128, 4096, 4096);  // Small batch, large K
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
