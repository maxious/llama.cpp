// Benchmark for MUL_MAT_ID direct vs fallback kernel paths on SYCL backend
// Usage: ./build/bin/test-mul-mat-id-bench [--iters N] [--type f32|f16|q8_0|q4_k] [--m M] [--n N] [--k K] [--n_mats N] [--n_used N]

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct bench_params {
    ggml_type type_a = GGML_TYPE_F32;
    int       n_mats = 8;
    int       n_used = 8;
    int64_t   m      = 128;
    int64_t   n      = 128;
    int64_t   k      = 4096;
    int       iters  = 100;
};

void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t                                nels = ggml_nelements(tensor);
    std::vector<float>                    data(nels);
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < nels; i++) {
        data[i] = dist(rng);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
}

void init_tensor_quant(ggml_tensor * tensor) {
    size_t                                nels = ggml_nelements(tensor);
    std::vector<float>                    data(nels);
    std::mt19937                          rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < nels; i++) {
        data[i] = dist(rng);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
}

void init_ids_tensor(ggml_tensor * ids, int n_mats) {
    int64_t                            n      = ids->ne[1];
    int64_t                            n_used = ids->ne[0];
    std::vector<int32_t>               data(n_used * n);
    std::mt19937                       rng(42);
    std::uniform_int_distribution<int> dist(0, n_mats - 1);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n_used; j++) {
            data[i * n_used + j] = dist(rng);
        }
    }
    ggml_backend_tensor_set(ids, data.data(), 0, n_used * n * sizeof(int32_t));
}

ggml_tensor * create_mul_mat_id_tensor(ggml_context * ctx,
                                       ggml_backend_t backend,
                                       ggml_type      type_a,
                                       ggml_type      type_b,
                                       int            n_mats,
                                       int            n_used,
                                       int64_t        m,
                                       int64_t        n,
                                       int64_t        k) {
    ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    // as: [k, m, n_mats] - weight matrices for each expert
    ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
    ggml_set_name(as, "as");
    init_tensor_uniform(as);

    // ids: [n_used, n] - which experts to use for each token
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
    ggml_set_name(ids, "ids");
    init_ids_tensor(ids, n_mats);

    // b: [k, n_used, n] - input activations
    ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, n_used, n);
    ggml_set_name(b, "b");
    init_tensor_uniform(b);

    // out: [m, n_used, n] - output
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_set_name(out, "out");

    return out;
}

void print_usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --type <f32|f16|q8_0|q4_k>  Type of weight matrix (default: f32)\n");
    printf("  --m <N>                     Output dimension (default: 128)\n");
    printf("  --n <N>                     Sequence length (default: 128)\n");
    printf("  --k <N>                     Hidden dimension (default: 4096)\n");
    printf("  --n_mats <N>                Number of experts (default: 8)\n");
    printf("  --n_used <N>                Number of experts to route to (default: 8)\n");
    printf("  --iters <N>                 Number of iterations (default: 100)\n");
}

bench_params parse_args(int argc, char ** argv) {
    bench_params params;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "f32") == 0) {
                params.type_a = GGML_TYPE_F32;
            } else if (strcmp(argv[i], "f16") == 0) {
                params.type_a = GGML_TYPE_F16;
            } else if (strcmp(argv[i], "q8_0") == 0) {
                params.type_a = GGML_TYPE_Q8_0;
            } else if (strcmp(argv[i], "q4_k") == 0) {
                params.type_a = GGML_TYPE_Q4_K;
            } else {
                fprintf(stderr, "Unknown type: %s\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            params.m = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            params.n = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            params.k = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--n_mats") == 0 && i + 1 < argc) {
            params.n_mats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n_used") == 0 && i + 1 < argc) {
            params.n_used = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            params.iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }
    return params;
}

int main(int argc, char ** argv) {
    bench_params params = parse_args(argc, argv);

    printf("=== MUL_MAT_ID Benchmark ===\n");
    printf("Type: %s\n", ggml_type_name(params.type_a));
    printf("m (output): %ld\n", params.m);
    printf("k (hidden): %ld\n", params.k);
    printf("n (seq_len): %ld\n", params.n);
    printf("n_mats (experts): %d\n", params.n_mats);
    printf("n_used (top_k): %d\n", params.n_used);
    printf("Iterations: %d\n", params.iters);
    printf("\n");

    ggml_initialize();
    atexit(ggml_free);

    // List backends
    printf("Available backends:\n");
    int n_backends = ggml_backend_get_count();
    for (int i = 0; i < n_backends; i++) {
        ggml_backend_t backend = ggml_backend_get(i);
        printf("  [%d] %s\n", i, ggml_backend_name(backend));
        ggml_backend_free(backend);
    }
    printf("\n");

    // Find SYCL backend
    ggml_backend_t backend = NULL;
    for (int i = 0; i < n_backends; i++) {
        ggml_backend_t b = ggml_backend_get(i);
        if (strstr(ggml_backend_name(b), "SYCL") != NULL) {
            backend = b;
            break;
        }
        ggml_backend_free(b);
    }

    if (!backend) {
        fprintf(stderr, "SYCL backend not found!\n");
        return 1;
    }

    printf("Using backend: %s\n\n", ggml_backend_name(backend));

    // Create context and allocators
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t      buf  = ggml_backend_buft_alloc_buffer(buft, 128 * 1024 * 1024);
    ggml_context_t             ctx  = ggml_init({ .extra_buffers = &buf, .no_alloc = false });
    if (!ctx) {
        fprintf(stderr, "Failed to create GGML context\n");
        return 1;
    }

    // Create compute graph
    ggml_cgraph   gf  = {};
    ggml_tensor * out = create_mul_mat_id_tensor(ctx, backend, params.type_a, GGML_TYPE_F32, params.n_mats,
                                                 params.n_used, params.m, params.n, params.k);
    ggml_build_forward_expand(&gf, out);

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 5; i++) {
        ggml_backend_graph_compute(backend, &gf);
    }
    ggml_backend_synchronize(backend);

    // Benchmark
    printf("Running benchmark...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < params.iters; i++) {
        ggml_backend_graph_compute(backend, &gf);
    }
    ggml_backend_synchronize(backend);
    auto end = std::chrono::high_resolution_clock::now();

    double total_time  = std::chrono::duration<double>(end - start).count();
    double avg_time_ms = (total_time / params.iters) * 1000.0;
    double tflops      = (2.0 * params.m * params.k * params.n * params.n_used * params.iters) / (total_time * 1e12);

    printf("\n=== Results ===\n");
    printf("Total time: %.3f s\n", total_time);
    printf("Avg time per iteration: %.3f ms\n", avg_time_ms);
    printf("Throughput: %.2f TFLOPS\n", tflops);

    // Calculate expected memory bandwidth usage
    size_t src0_bytes     = ggml_nbytes(out->src[0]);
    size_t src1_bytes     = ggml_nbytes(out->src[1]);
    size_t dst_bytes      = ggml_nbytes(out);
    double bytes_per_iter = src0_bytes + src1_bytes + dst_bytes;
    double gb_s           = (bytes_per_iter * params.iters) / (total_time * 1e9);
    printf("Effective bandwidth: %.2f GB/s\n", gb_s);

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);

    return 0;
}
