// Test XMX Flash Attention GEMM with SYCL Graphs
// 
// This tests graph compatibility of our XMX GEMM kernels for flash attention.
// SYCL command graphs allow recording and replaying kernel sequences for reduced overhead.
//
// Build:
//   icpx -fsycl -O3 -DSYCL_EXT_ONEAPI_MATRIX -I. tests/test-fattn-xmx-graph.cpp -o test-fattn-xmx-graph
//
// Run:
//   ./test-fattn-xmx-graph

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

#include "../ggml/src/ggml-sycl/gemm_xmx.hpp"

#ifdef SYCL_EXT_ONEAPI_GRAPH
#include <sycl/ext/oneapi/experimental/graph.hpp>
namespace syclex = sycl::ext::oneapi::experimental;
#endif

// Flash attention using XMX GEMM with optional graph recording
class FlashAttnXMXGraphTest {
public:
    sycl::queue q;
    bool has_graph_support;
    
    FlashAttnXMXGraphTest() : q(sycl::gpu_selector_v, sycl::property::queue::in_order{}) {
        auto device = q.get_device();
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        
#ifdef SYCL_EXT_ONEAPI_GRAPH
        has_graph_support = device.has(sycl::aspect::ext_oneapi_graph);
        std::cout << "Graph support: " << (has_graph_support ? "YES" : "NO") << "\n\n";
#else
        has_graph_support = false;
        std::cout << "Graph support: NOT COMPILED (need SYCL_EXT_ONEAPI_GRAPH)\n\n";
#endif
    }
    
    // Reference CPU implementation
    void flash_attn_cpu(
        const std::vector<float>& Q,
        const std::vector<float>& K,
        const std::vector<float>& V,
        std::vector<float>& O,
        int N, int N_kv, int D, float scale
    ) {
        std::vector<float> S(N * N_kv);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N_kv; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < D; ++k) {
                    sum += Q[i * D + k] * K[j * D + k];
                }
                S[i * N_kv + j] = sum * scale;
            }
        }
        
        std::vector<float> P(N * N_kv);
        for (int i = 0; i < N; ++i) {
            float max_val = -1e20f;
            for (int j = 0; j < N_kv; ++j) {
                max_val = std::max(max_val, S[i * N_kv + j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < N_kv; ++j) {
                P[i * N_kv + j] = std::exp(S[i * N_kv + j] - max_val);
                sum += P[i * N_kv + j];
            }
            for (int j = 0; j < N_kv; ++j) {
                P[i * N_kv + j] /= sum;
            }
        }
        
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < N_kv; ++j) {
                    sum += P[i * N_kv + j] * V[j * D + d];
                }
                O[i * D + d] = sum;
            }
        }
    }
    
    // Simple softmax kernel for GPU
    void softmax_gpu(float* S, int N, int N_kv) {
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> row) {
                float max_val = -1e20f;
                for (int j = 0; j < N_kv; ++j) {
                    max_val = sycl::fmax(max_val, S[row * N_kv + j]);
                }
                float sum = 0.0f;
                for (int j = 0; j < N_kv; ++j) {
                    float val = sycl::exp(sycl::fmax(S[row * N_kv + j] - max_val, -20.0f));
                    S[row * N_kv + j] = val;
                    sum += val;
                }
                for (int j = 0; j < N_kv; ++j) {
                    S[row * N_kv + j] /= sum;
                }
            });
        });
    }
    
    // Run flash attention with Q@K^T XMX GEMM - no graph
    // Note: P@V is done with a transpose trick - store V^T and use transpose_B=true
    void flash_attn_xmx_no_graph(
        sycl::half* d_Q, sycl::half* d_K, sycl::half* d_V_T,  // V_T is [D, N_kv] = V^T
        float* d_S, sycl::half* d_P, float* d_O,
        int N, int N_kv, int D, float scale
    ) {
#ifdef SYCL_EXT_ONEAPI_MATRIX
        // Step 1: S = Q @ K^T
        // Q[N, D] @ K[N_kv, D]^T = S[N, N_kv]
        launch_gemm_xmx_f16<true>(&q, d_Q, d_K, d_S, N, N_kv, D, scale, 0.0f, D, D, N_kv);
        
        // Step 2: Softmax
        softmax_gpu(d_S, N, N_kv);
        
        // Step 3: Convert S to half and store in P
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(N * N_kv), [=](sycl::id<1> idx) {
                d_P[idx] = sycl::half(d_S[idx]);
            });
        });
        
        // Step 4: O = P @ V
        // P[N, N_kv] @ V[N_kv, D] = O[N, D]
        // Using transpose_B=true: P[N, N_kv] @ V_T[D, N_kv]^T = O[N, D]
        // This works because (V_T)^T = V
        launch_gemm_xmx_f16<true>(&q, d_P, d_V_T, d_O, N, D, N_kv, 1.0f, 0.0f, N_kv, N_kv, D);
        
        q.wait();
#endif
    }
    
#ifdef SYCL_EXT_ONEAPI_GRAPH
    // Run flash attention with SYCL graph
    void flash_attn_xmx_with_graph(
        sycl::half* d_Q, sycl::half* d_K, sycl::half* d_V_T,  // V_T is [D, N_kv] = V^T
        float* d_S, sycl::half* d_P, float* d_O,
        int N, int N_kv, int D, float scale,
        int num_iterations
    ) {
#ifdef SYCL_EXT_ONEAPI_MATRIX
        // Record graph
        syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), q.get_device());
        
        graph.begin_recording(q);
        
        // Step 1: S = Q @ K^T
        launch_gemm_xmx_f16<true>(&q, d_Q, d_K, d_S, N, N_kv, D, scale, 0.0f, D, D, N_kv);
        
        // Step 2: Softmax
        softmax_gpu(d_S, N, N_kv);
        
        // Step 3: Convert S to half
        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(N * N_kv), [=](sycl::id<1> idx) {
                d_P[idx] = sycl::half(d_S[idx]);
            });
        });
        
        // Step 4: O = P @ V (using V^T with transpose)
        launch_gemm_xmx_f16<true>(&q, d_P, d_V_T, d_O, N, D, N_kv, 1.0f, 0.0f, N_kv, N_kv, D);
        
        graph.end_recording(q);
        
        // Finalize graph
        auto exec_graph = graph.finalize();
        
        // Execute graph multiple times
        for (int i = 0; i < num_iterations; ++i) {
            q.ext_oneapi_graph(exec_graph);
        }
        q.wait();
#endif
    }
#endif
    
    void run_test(int N, int N_kv, int D) {
        std::cout << "Testing Flash Attention: N=" << N << " N_kv=" << N_kv << " D=" << D << "\n";
        
        const float scale = 1.0f / std::sqrt(float(D));
        
        // Generate random data
        std::vector<float> h_Q(N * D), h_K(N_kv * D), h_V(N_kv * D);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& v : h_Q) v = dist(rng);
        for (auto& v : h_K) v = dist(rng);
        for (auto& v : h_V) v = dist(rng);
        
        // CPU reference
        std::vector<float> h_O_ref(N * D);
        flash_attn_cpu(h_Q, h_K, h_V, h_O_ref, N, N_kv, D, scale);
        
#ifdef SYCL_EXT_ONEAPI_MATRIX
        // Convert to F16
        std::vector<sycl::half> h_Q_f16(N * D), h_K_f16(N_kv * D);
        for (int i = 0; i < N * D; ++i) h_Q_f16[i] = sycl::half(h_Q[i]);
        for (int i = 0; i < N_kv * D; ++i) h_K_f16[i] = sycl::half(h_K[i]);
        
        // Create V^T (transpose of V) for P@V computation
        // V is [N_kv, D], V^T is [D, N_kv]
        std::vector<sycl::half> h_V_T_f16(D * N_kv);
        for (int row = 0; row < N_kv; ++row) {
            for (int col = 0; col < D; ++col) {
                h_V_T_f16[col * N_kv + row] = sycl::half(h_V[row * D + col]);
            }
        }
        
        // Allocate device memory
        sycl::half* d_Q = sycl::malloc_device<sycl::half>(N * D, q);
        sycl::half* d_K = sycl::malloc_device<sycl::half>(N_kv * D, q);
        sycl::half* d_V_T = sycl::malloc_device<sycl::half>(D * N_kv, q);  // V^T
        float* d_S = sycl::malloc_device<float>(N * N_kv, q);
        sycl::half* d_P = sycl::malloc_device<sycl::half>(N * N_kv, q);
        float* d_O = sycl::malloc_device<float>(N * D, q);
        
        q.memcpy(d_Q, h_Q_f16.data(), N * D * sizeof(sycl::half)).wait();
        q.memcpy(d_K, h_K_f16.data(), N_kv * D * sizeof(sycl::half)).wait();
        q.memcpy(d_V_T, h_V_T_f16.data(), D * N_kv * sizeof(sycl::half)).wait();
        
        // Test without graph
        flash_attn_xmx_no_graph(d_Q, d_K, d_V_T, d_S, d_P, d_O, N, N_kv, D, scale);
        
        std::vector<float> h_O_xmx(N * D);
        q.memcpy(h_O_xmx.data(), d_O, N * D * sizeof(float)).wait();
        
        // Verify correctness
        float max_diff = 0.0f;
        double sum_sq_err = 0.0, sum_sq_ref = 0.0;
        for (int i = 0; i < N * D; ++i) {
            float diff = h_O_ref[i] - h_O_xmx[i];
            max_diff = std::max(max_diff, std::abs(diff));
            sum_sq_err += diff * diff;
            sum_sq_ref += h_O_ref[i] * h_O_ref[i];
        }
        double nmse = sum_sq_err / (sum_sq_ref + 1e-10);
        std::cout << "  No-graph: Max diff=" << max_diff << ", NMSE=" << nmse 
                  << " " << (nmse < 1e-3 ? "PASSED" : "FAILED") << "\n";
        
        // Benchmark without graph
        const int warmup = 5, iters = 20;
        for (int i = 0; i < warmup; ++i) {
            flash_attn_xmx_no_graph(d_Q, d_K, d_V_T, d_S, d_P, d_O, N, N_kv, D, scale);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            flash_attn_xmx_no_graph(d_Q, d_K, d_V_T, d_S, d_P, d_O, N, N_kv, D, scale);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double no_graph_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
        std::cout << "  No-graph time: " << no_graph_ms << " ms\n";
        
#ifdef SYCL_EXT_ONEAPI_GRAPH
        if (has_graph_support) {
            // Test with graph
            flash_attn_xmx_with_graph(d_Q, d_K, d_V_T, d_S, d_P, d_O, N, N_kv, D, scale, 1);
            
            q.memcpy(h_O_xmx.data(), d_O, N * D * sizeof(float)).wait();
            
            max_diff = 0.0f;
            sum_sq_err = 0.0;
            for (int i = 0; i < N * D; ++i) {
                float diff = h_O_ref[i] - h_O_xmx[i];
                max_diff = std::max(max_diff, std::abs(diff));
                sum_sq_err += diff * diff;
            }
            nmse = sum_sq_err / (sum_sq_ref + 1e-10);
            std::cout << "  With-graph: Max diff=" << max_diff << ", NMSE=" << nmse 
                      << " " << (nmse < 1e-3 ? "PASSED" : "FAILED") << "\n";
            
            // Benchmark with graph
            for (int i = 0; i < warmup; ++i) {
                flash_attn_xmx_with_graph(d_Q, d_K, d_V_T, d_S, d_P, d_O, N, N_kv, D, scale, 1);
            }
            
            start = std::chrono::high_resolution_clock::now();
            flash_attn_xmx_with_graph(d_Q, d_K, d_V_T, d_S, d_P, d_O, N, N_kv, D, scale, iters);
            end = std::chrono::high_resolution_clock::now();
            double graph_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
            std::cout << "  With-graph time: " << graph_ms << " ms (speedup: " 
                      << (no_graph_ms / graph_ms) << "x)\n";
        }
#endif
        
        sycl::free(d_Q, q);
        sycl::free(d_K, q);
        sycl::free(d_V_T, q);
        sycl::free(d_S, q);
        sycl::free(d_P, q);
        sycl::free(d_O, q);
#else
        std::cout << "  XMX not available (SYCL_EXT_ONEAPI_MATRIX not defined)\n";
#endif
        std::cout << "\n";
    }
};

int main() {
    try {
        FlashAttnXMXGraphTest test;
        
        std::cout << "=== Flash Attention XMX GEMM + Graph Test ===\n\n";
        
        // Decode phase (single query)
        test.run_test(1, 512, 128);
        test.run_test(1, 2048, 128);
        
        // Small prefill
        test.run_test(32, 32, 128);
        test.run_test(64, 64, 128);
        
        // Larger prefill
        test.run_test(128, 128, 128);
        test.run_test(256, 256, 128);
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
