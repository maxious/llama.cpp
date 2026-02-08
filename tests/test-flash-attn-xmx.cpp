#include <ggml.h>
#include <ggml-backend.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct flash_attn_case {
    int64_t d_qk;
    int64_t d_v;
    int64_t n_heads;
    int64_t n_q;
    int64_t n_kv;
    ggml_type type_q;
    ggml_type type_kv;
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

static ggml_tensor * build_graph(ggml_context * ctx, const flash_attn_case & test_case, ggml_tensor ** q_out,
                                ggml_tensor ** k_out, ggml_tensor ** v_out) {
    const std::array<int64_t, 4> q_dims = {test_case.d_qk, test_case.n_q, test_case.n_heads, 1};
    const std::array<int64_t, 4> kv_dims = {test_case.d_qk, test_case.n_kv, test_case.n_heads, 1};
    const std::array<int64_t, 4> vv_dims = {test_case.d_v, test_case.n_kv, test_case.n_heads, 1};

    ggml_tensor * q = ggml_new_tensor(ctx, test_case.type_q, 4, q_dims.data());
    ggml_tensor * k = ggml_new_tensor(ctx, test_case.type_kv, 4, kv_dims.data());
    ggml_tensor * v = ggml_new_tensor(ctx, test_case.type_kv, 4, vv_dims.data());

    ggml_set_name(q, "q");
    ggml_set_name(k, "k");
    ggml_set_name(v, "v");

    const float scale = 1.0f / sqrtf((float) test_case.d_qk);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_set_name(out, "out");

    if (q_out) {
        *q_out = q;
    }
    if (k_out) {
        *k_out = k;
    }
    if (v_out) {
        *v_out = v;
    }

    return out;
}

static bool run_case(const flash_attn_case & test_case, ggml_backend_t backend, ggml_backend_t backend_cpu) {
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

    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * out = build_graph(ctx, test_case, &q, &k, &v);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "failed to allocate tensors for backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    init_tensor_uniform(q);
    init_tensor_uniform(k);
    init_tensor_uniform(v);

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

    bool ok = compare_tensor(out, out_cpu, 5e-4);

    ggml_backend_graph_copy_free(copy);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return ok;
}

static ggml_type parse_type(const char * value) {
    if (strcmp(value, "f16") == 0) {
        return GGML_TYPE_F16;
    }
    if (strcmp(value, "f32") == 0) {
        return GGML_TYPE_F32;
    }
    fprintf(stderr, "unknown type: %s\n", value);
    std::abort();
}

int main(int argc, char ** argv) {
    const char * backend_name = "SYCL0";
    std::vector<int64_t> heads = {64, 96, 128};
    int64_t n_q = 64;
    std::vector<int64_t> n_kv = {256, 2048};
    ggml_type type_q = GGML_TYPE_F16;
    ggml_type type_kv = GGML_TYPE_F16;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            heads.clear();
            const char * list = argv[++i];
            const char * cursor = list;
            while (*cursor != '\0') {
                while (*cursor == ' ' || *cursor == ',') {
                    ++cursor;
                }
                if (*cursor == '\0') {
                    break;
                }
                char * end = nullptr;
                long value = std::strtol(cursor, &end, 10);
                if (end != cursor) {
                    heads.push_back(value);
                    cursor = end;
                } else {
                    ++cursor;
                }
            }
        } else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc) {
            n_q = std::strtol(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--nkv") == 0 && i + 1 < argc) {
            n_kv.clear();
            const char * list = argv[++i];
            const char * cursor = list;
            while (*cursor != '\0') {
                while (*cursor == ' ' || *cursor == ',') {
                    ++cursor;
                }
                if (*cursor == '\0') {
                    break;
                }
                char * end = nullptr;
                long value = std::strtol(cursor, &end, 10);
                if (end != cursor) {
                    n_kv.push_back(value);
                    cursor = end;
                } else {
                    ++cursor;
                }
            }
        } else if (strcmp(argv[i], "--type-q") == 0 && i + 1 < argc) {
            type_q = parse_type(argv[++i]);
        } else if (strcmp(argv[i], "--type-kv") == 0 && i + 1 < argc) {
            type_kv = parse_type(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--backend SYCL0] [--heads 64,96,128] [--nq 64] [--nkv 256,2048] [--type-q f16|f32] [--type-kv f16|f32]\n", argv[0]);
            return 1;
        }
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

    srand(0);

    bool ok = true;
    for (int64_t head : heads) {
        for (int64_t nkv : n_kv) {
            flash_attn_case test_case = {
                head,
                head,
                32,
                n_q,
                nkv,
                type_q,
                type_kv,
            };

            printf("Testing DQK=%ld DV=%ld N=%ld N_kv=%ld type_q=%s type_kv=%s... ",
                   test_case.d_qk,
                   test_case.d_v,
                   test_case.n_q,
                   test_case.n_kv,
                   test_case.type_q == GGML_TYPE_F16 ? "f16" : "f32",
                   test_case.type_kv == GGML_TYPE_F16 ? "f16" : "f32");

            if (!run_case(test_case, backend, backend_cpu)) {
                printf("FAIL\n");
                ok = false;
                break;
            }
            printf("OK\n");
        }
        if (!ok) {
            break;
        }
    }

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend);

    return ok ? 0 : 1;
}
