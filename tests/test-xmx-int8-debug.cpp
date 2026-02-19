#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <time.h>
#include <vector>

struct matmul_case {
    int64_t k;
    int64_t m;
    int64_t n;
    ggml_type type_a;
};

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    const size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    for (size_t i = 0; i < nels; ++i) {
        data[i] = min + (max - min) * (float) rand() / (float) RAND_MAX;
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
        return;
    }

    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data_f16(nels);
        ggml_fp32_to_fp16_row(data.data(), data_f16.data(), nels);
        ggml_backend_tensor_set(tensor, data_f16.data(), 0, nels * sizeof(ggml_fp16_t));
        return;
    }

    if (ggml_is_quantized(tensor->type)) {
        const size_t blck_size = ggml_blck_size(tensor->type);
        GGML_ASSERT(nels % blck_size == 0);
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, nels / blck_size, blck_size, nullptr);
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
        return;
    }

    fprintf(stderr, "unsupported tensor type for init\n");
    std::abort();
}

static double nmse(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    double sum_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double av = a[i];
        const double bv = b[i];
        const double diff = av - bv;
        sum += av * av;
        sum_err += diff * diff;
    }
    if (sum == 0.0) {
        return sum_err == 0.0 ? 0.0 : sum_err;
    }
    return sum_err / sum;
}

static bool compare_tensor(ggml_tensor * a, ggml_tensor * b, double max_nmse) {
    const size_t n = ggml_nelements(a);
    std::vector<float> fa(n);
    std::vector<float> fb(n);
    ggml_backend_tensor_get(a, fa.data(), 0, n * sizeof(float));
    ggml_backend_tensor_get(b, fb.data(), 0, n * sizeof(float));

    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(fa[i]) || std::isnan(fb[i])) {
            fprintf(stderr, "NaN detected at %zu\n", i);
            return false;
        }
        const bool a_inf = std::isinf(fa[i]);
        const bool b_inf = std::isinf(fb[i]);
        if (a_inf || b_inf) {
            if (!(a_inf && b_inf && std::signbit(fa[i]) == std::signbit(fb[i]))) {
                fprintf(stderr, "Inf mismatch at %zu: %f vs %f\n", i, fa[i], fb[i]);
                return false;
            }
        }
    }

    const double err = nmse(fa.data(), fb.data(), n);
    if (err > max_nmse) {
        fprintf(stderr, "ERR = %.9f > %.9f\n", err, max_nmse);
        return false;
    }
    return true;
}

static ggml_tensor * build_graph(ggml_context * ctx, const matmul_case & test_case, ggml_tensor ** a_out,
                                 ggml_tensor ** b_out) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, test_case.type_a, test_case.k, test_case.m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test_case.k, test_case.n);

    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_set_name(out, "out");

    if (a_out) {
        *a_out = a;
    }
    if (b_out) {
        *b_out = b;
    }

    return out;
}

static bool run_case(const matmul_case & test_case, ggml_backend_t backend, ggml_backend_t backend_cpu) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to init context\n");
        return false;
    }

    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * out = build_graph(ctx, test_case, &a, &b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate tensors for backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    init_tensor_uniform(a);
    init_tensor_uniform(b);

    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend_cpu, gf);
    if (copy.buffer == NULL) {
        fprintf(stderr, "failed to copy graph for CPU\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_graph_compute(backend_cpu, copy.graph);

    ggml_tensor * out_cpu = ggml_get_tensor(copy.ctx_allocated, "out");
    if (!out_cpu) {
        fprintf(stderr, "failed to locate output tensor on CPU\n");
        ggml_backend_graph_copy_free(copy);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    bool ok = compare_tensor(out, out_cpu, 1e-2);

    ggml_backend_graph_copy_free(copy);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return ok;
}

static ggml_type parse_type(const char * value) {
    if (strcmp(value, "q8_0") == 0) {
        return GGML_TYPE_Q8_0;
    }
    if (strcmp(value, "q4_0") == 0) {
        return GGML_TYPE_Q4_0;
    }
    if (strcmp(value, "q4_1") == 0) {
        return GGML_TYPE_Q4_1;
    }
    if (strcmp(value, "q5_0") == 0) {
        return GGML_TYPE_Q5_0;
    }
    if (strcmp(value, "q5_1") == 0) {
        return GGML_TYPE_Q5_1;
    }
    if (strcmp(value, "q8_1") == 0) {
        return GGML_TYPE_Q8_1;
    }
    fprintf(stderr, "unknown type: %s\n", value);
    std::abort();
}

static double run_bench(const matmul_case & test_case, ggml_backend_t backend, int warmup, int iters) {
    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 64 + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "failed to init context\n");
        return -1.0;
    }

    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_tensor * out = build_graph(ctx, test_case, &a, &b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate tensors for backend\n");
        ggml_free(ctx);
        return -1.0;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    init_tensor_uniform(a);
    init_tensor_uniform(b);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }

    // Benchmark
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double avg_ms = elapsed_ms / iters;

    // Compute GFLOPS: 2*M*N*K flops per matmul
    double flops = 2.0 * test_case.m * test_case.n * test_case.k;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return avg_ms;
}

int main(int argc, char ** argv) {
    const char * backend_name = "SYCL0";
    int64_t k = 128;
    int64_t m = 64;
    int64_t n = 64;
    ggml_type type_a = GGML_TYPE_Q8_0;
    bool set_xmx_int8 = true;
    bool set_debug = false;
    bool bench_mode = false;
    int bench_warmup = 5;
    int bench_iters = 20;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = std::strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            m = std::strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = std::strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--type-a") == 0 && i + 1 < argc) {
            type_a = parse_type(argv[++i]);
        } else if (strcmp(argv[i], "--no-xmx-int8") == 0) {
            set_xmx_int8 = false;
        } else if (strcmp(argv[i], "--debug") == 0) {
            set_debug = true;
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = true;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            bench_warmup = std::strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            bench_iters = std::strtol(argv[++i], nullptr, 10);
        } else {
            fprintf(stderr,
                    "Usage: %s [--backend SYCL0] [--k 128] [--m 64] [--n 64] [--type-a q8_0|q4_0|q4_1|q5_0|q5_1|q8_1] "
                    "[--no-xmx-int8] [--debug] [--bench] [--warmup 5] [--iters 20]\n",
                    argv[0]);
            return 1;
        }
    }

    if (set_xmx_int8 && getenv("GGML_SYCL_XMX_INT8") == nullptr) {
        setenv("GGML_SYCL_XMX_INT8", "1", 1);
    }
    if (set_debug && getenv("GGML_SYCL_DEBUG") == nullptr) {
        setenv("GGML_SYCL_DEBUG", "1", 1);
    }

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) {
        fprintf(stderr, "failed to init backend %s\n", backend_name);
        return 1;
    }

    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu) {
        fprintf(stderr, "failed to init CPU backend\n");
        ggml_backend_free(backend);
        return 1;
    }

    using ggml_backend_cpu_set_use_ref_t = void (*)(ggml_backend_t, bool);
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
    auto * set_use_ref = (ggml_backend_cpu_set_use_ref_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
    if (set_use_ref) {
        set_use_ref(backend_cpu, true);
    }

    if (ggml_is_quantized(type_a)) {
        ggml_quantize_init(type_a);
    }

    srand(0);

    matmul_case test_case = {k, m, n, type_a};

    if (bench_mode) {
        printf("Benchmarking K=%ld M=%ld N=%ld type_a=%s (warmup=%d, iters=%d)...\n",
               test_case.k, test_case.m, test_case.n,
               ggml_type_name(test_case.type_a), bench_warmup, bench_iters);

        double avg_ms = run_bench(test_case, backend, bench_warmup, bench_iters);
        double flops = 2.0 * test_case.m * test_case.n * test_case.k;
        double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

        printf("  avg: %.3f ms  (%.2f GFLOPS)\n", avg_ms, gflops);

        if (ggml_is_quantized(type_a)) {
            ggml_quantize_free();
        }
        ggml_backend_free(backend_cpu);
        ggml_backend_free(backend);
        return 0;
    }

    printf("Testing K=%ld M=%ld N=%ld type_a=%s... ",
           test_case.k,
           test_case.m,
           test_case.n,
           ggml_type_name(test_case.type_a));

    bool ok = run_case(test_case, backend, backend_cpu);
    printf(ok ? "OK\n" : "FAIL\n");

    if (ggml_is_quantized(type_a)) {
        ggml_quantize_free();
    }

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend);

    return ok ? 0 : 1;
}
