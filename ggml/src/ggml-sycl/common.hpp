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

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#include "dpct/helper.hpp"
#include "ggml-sycl.h"
#include "presets.hpp"
#include "sycl_hw.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#if GGML_SYCL_DNNL
#    include "dnnl.hpp"
#    include "dnnl_sycl.hpp"
#endif

#include "flash_attn_buffers.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
/* suppress warning spam */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "ggml-common.h"
#pragma clang diagnostic pop
#include "ggml-impl.h"

void * ggml_sycl_host_malloc(size_t size);
void   ggml_sycl_host_free(void * ptr);

extern int g_ggml_sycl_debug;
extern int g_ggml_sycl_disable_optimize;
extern int g_ggml_sycl_prioritize_dmmv;

#if defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(expr)   __builtin_expect(expr, true)
#    define UNLIKELY(expr) __builtin_expect(expr, false)
#else
#    define LIKELY(expr)   (expr)
#    define UNLIKELY(expr) (expr)
#endif

#define GGML_SYCL_DEBUG(...)              \
    do {                                  \
        if (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)

#define CHECK_TRY_ERROR(expr)                                                                           \
    [&]() {                                                                                             \
        try {                                                                                           \
            expr;                                                                                       \
            return dpct::success;                                                                       \
        } catch (std::exception const & e) {                                                            \
            std::cerr << e.what() << "\nException caught at file:" << __FILE__ << ", line:" << __LINE__ \
                      << ", func:" << __func__ << std::endl;                                            \
            return dpct::default_error;                                                                 \
        }                                                                                               \
    }()

#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC      610                 // todo for hardward optimize.
#define VER_GEN9      700                 // todo for hardward optimize.
#define VER_GEN12     1000000             // todo for hardward optimize.
#define VER_GEN13     (VER_GEN12 + 1030)  // todo for hardward optimize.

#define GGML_SYCL_MAX_NODES 8192          // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.

// max batch size to use MMQ kernels when tensor cores are available
#define MMQ_MAX_BATCH_SIZE 32

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#    define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#    define GGML_SYCL_MMV_Y 1
#endif

typedef sycl::queue * queue_ptr;

enum ggml_sycl_backend_gpu_mode { SYCL_UNSET_GPU_MODE = -1, SYCL_SINGLE_GPU_MODE = 0, SYCL_MUL_GPU_MODE };

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash() {
    int * ptr = NULL;
    *ptr      = 0;
}

[[noreturn]] static void ggml_sycl_error(const char * stmt,
                                         const char * func,
                                         const char * file,
                                         const int    line,
                                         const char * msg) {
    fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
    fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
    GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    do {                                                                                                   \
        auto err_ = (err);                                                                                 \
        if (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100
#    define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#    define GGML_SYCL_ASSUME(x)
#endif  // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half  dfloat;  // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float        dfloat;  // dequantize float
typedef sycl::float2 dfloat2;
#endif  // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE 8

static int  g_all_sycl_device_count                     = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode = SYCL_UNSET_GPU_MODE;

static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size   = 0;  // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static inline void bad_arch(const sycl::stream & stream_ct1) {
    stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                  "current GPU architecture.\n";
    // __trap();
    std::exit(1);

    (void) bad_arch;  // suppress unused function warning
}

int get_current_device_id();

inline dpct::err0 ggml_sycl_set_device(const int device) try {
    int current_device_id;
    SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

    // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
    // current_device_id=%d\n", device, current_device);
    if (device == current_device_id) {
        return 0;
    }

    return CHECK_TRY_ERROR(dpct::select_device(device));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    crash();
    std::exit(1);
}

// SYCL device architecture types for runtime optimization
enum sycl_arch_type {
    SYCL_ARCH_UNKNOWN = 0,
    SYCL_ARCH_INTEL_GEN9,    // Intel Gen9-Gen11 (integrated)
    SYCL_ARCH_INTEL_XE,      // Intel Xe (Alchemist) - Gen12.7 discrete
    SYCL_ARCH_INTEL_XE2,     // Intel Xe2 (Battlemage) - Gen13
    SYCL_ARCH_INTEL_XE_LPG,  // Intel Xe-LPG (Meteor Lake) - integrated Gen12
    SYCL_ARCH_INTEL_PVC,     // Intel Ponte Vecchio (Xe-HPC)
    SYCL_ARCH_AMD_RDNA1,
    SYCL_ARCH_AMD_RDNA2,
    SYCL_ARCH_AMD_RDNA3,
    SYCL_ARCH_NVIDIA_AMPERE,
    SYCL_ARCH_NVIDIA_TURING,
    // Add more as needed
};

// MMQ kernel tile configuration
struct mmq_config {
    int mmq_x;
    int mmq_y;
    int nwarps;
};

////////////////////
struct optimize_feature {
    bool reorder = false;
};

struct sycl_device_info {
    int              cc;   // compute capability
    int              nsm;  // number of streaming multiprocessors (CUDA) maps to the maximum
                           // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t           smpbo;  // max. shared memory per block (with opt-in)
    bool             vmm;    // virtual memory support
    size_t           total_vram;
    //sycl_hw_info hw_info;     \\ device id and aarch, currently not used
    optimize_feature opt_feature;
    mmq_config       mmq;   // MMQ tile configuration (runtime determined)
    sycl_arch_type   arch;  // Device architecture type for feature dispatch
};

struct ggml_sycl_device_info {
    int device_count;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    int max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = { 0 };
};

const ggml_sycl_device_info & ggml_sycl_info();

struct ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void   free(void * ptr, size_t size)            = 0;
};

template <typename T> struct ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool        = nullptr;
    T *              ptr         = nullptr;
    size_t           actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {}

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) { alloc(size); }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        if (ptr) {
            pool->free(ptr, actual_size);
        }
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() { return ptr; }

    ggml_sycl_pool_alloc()                                         = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &)             = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&)                  = delete;
    ggml_sycl_pool_alloc & operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc & operator=(ggml_sycl_pool_alloc &&)      = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
    void *           data_device[GGML_SYCL_MAX_DEVICES];                    // 1 pointer for each device for split
                                                                            // tensors
    dpct::event_ptr  events[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS];  // events for synchronizing multiple GPUs
    optimize_feature optimized_feature;
};

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams = {});

namespace sycl_ex = sycl::ext::oneapi::experimental;

struct ggml_backend_sycl_context {
    int              device;
    std::string      name;
    optimize_feature opt_feature;
    bool             force_graph_compatible = false;
    bool             graph_recording_active = false;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(int device) : device(device), name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(int device, int stream) {
        if (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        return qptrs[device][stream];
    }

    queue_ptr stream() { return stream(device, 0); }

#if GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue * q) {
        // Get the device associated with the queue
        sycl::device       dev = q->get_device();
        // Get the context associated with the queue
        sycl::context      ctx = q->get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        return eng;
    }

    std::unordered_map<sycl::queue *, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue *, dnnl::engine> engine_map;

    dnnl::stream stream_dnnl(int device, int _stream) {
        auto q = stream(device, _stream);
        return stream_dnnl(q);
    }

    dnnl::engine engine_dnnl(sycl::queue * qptr) {
        auto it = engine_map.find(qptr);
        if (it == engine_map.end()) {
            auto eng         = make_engine(qptr);
            engine_map[qptr] = eng;
            return eng;
        } else {
            return it->second;
        }
    }

    dnnl::stream stream_dnnl(sycl::queue * qptr) {
        auto it = stream_map.find(qptr);
        if (it == stream_map.end()) {
            auto eng         = engine_dnnl(qptr);
            auto stream      = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            return stream;
        } else {
            return it->second;
        }
    }

    dnnl::stream stream_dnnl() { return stream_dnnl(device, 0); }

    dnnl::memory get_scratchpad_mem(const dnnl::memory::desc & scratchpad_md,
                                    const dnnl::engine &       eng,
                                    const queue_ptr            q) {
        ggml_sycl_pool_alloc<uint8_t> * pool;
        auto                            it = scratchpad_map.find(q);
        if (it == scratchpad_map.end()) {
            scratchpad_map[q] = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(this->pool());
            pool              = scratchpad_map[q].get();
        } else {
            pool = it->second.get();
        }

        size_t scratchpad_size = scratchpad_md.get_size();
        if (scratchpad_size > pool->actual_size) {
            pool->realloc(scratchpad_size);
        }
        void * mem_ptr = pool->get();
        return dnnl::memory(scratchpad_md, eng, mem_ptr);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool>                                                   pools[GGML_SYCL_MAX_DEVICES];
    std::unordered_map<sycl::queue *, std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>>> scratchpad_map;

    // Flash Attention buffer pool - preallocated to avoid malloc/free overhead per call
    std::unique_ptr<flash_attn_buffers> fattn_buffers;

    // Persistent graph-owned pointer tables for batched/GQA MUL_MAT paths.
    // Keyed by element count to reuse across graph recording/replay and avoid
    // stale RAII pool allocations captured in SYCL graphs.
    std::map<size_t, std::unique_ptr<ggml_sycl_pool_alloc<const void *>>> graph_ptrs_src_cache;
    std::map<size_t, std::unique_ptr<ggml_sycl_pool_alloc<void *>>>       graph_ptrs_dst_cache;

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device, 0), device);
        }
        return *pools[device];
    }

    ggml_sycl_pool & pool() { return pool(device); }

#ifdef GGML_SYCL_GRAPH
    // Single cached graph (legacy, for backward compatibility)
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
    uint64_t exec_graph_hash = 0;  // Hash of graph topology for cache invalidation

    // Multi-device graph support: one executable graph per device
    std::map<int, std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>>> per_device_exec_graphs;
    bool multi_device_graphs_initialized = false;

    // Graph cache: maps topology hash -> executable graph (for fully-compatible graphs)
    // This allows reusing graphs when the same topology is encountered again
    std::map<uint64_t, std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>>> graph_cache;
    // Pointer hash cache: maps topology hash -> USM pointer hash from last recording
    // Used for three-tier cache lookup: pure replay / re-record+update / full record+finalize
    std::map<uint64_t, uint64_t>                                                                  graph_pointer_hashes;
    static constexpr size_t MAX_GRAPH_CACHE_SIZE = 8;  // Limit cache to prevent memory bloat

    // Segmented graph cache: for graphs with immediate-mode nodes, we cache
    // per-segment executable graphs keyed by (topology_hash, segment_index).
    // Each segment is a contiguous range of graph-compatible nodes.
    struct segment_cache_entry {
        std::vector<std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>>> segment_graphs;
        uint64_t                                                                               pointer_hash = 0;
    };

    std::map<uint64_t, segment_cache_entry> segmented_graph_cache;

    // Dedicated graph execution queue with no_immediate_command_list property.
    // Intel discrete GPUs require this for efficient ext_oneapi_graph() submission.
    queue_ptr   graph_queue = nullptr;
    sycl::queue graph_queue_storage;

    queue_ptr graph_exec_stream() {
        if (graph_queue != nullptr) {
            return graph_queue;
        }
        // Create from the same context and device as the primary stream
        queue_ptr     primary = stream();
        sycl::context ctx     = primary->get_context();
        sycl::device  dev     = primary->get_device();
        graph_queue_storage   = sycl::queue(
            ctx, dev,
            [](sycl::exception_list exceptions) {
                for (const auto & e : exceptions) {
                    try {
                        std::rethrow_exception(e);
                    } catch (const sycl::exception & e) {
                        std::cerr << "Caught asynchronous SYCL exception (graph queue):" << std::endl
                                  << e.what() << std::endl;
                    }
                }
            },
            sycl::property_list(sycl::property::queue::in_order{},
                                  sycl::ext::intel::property::queue::no_immediate_command_list{}));
        graph_queue = &graph_queue_storage;
        GGML_SYCL_DEBUG("[SYCL-GRAPH] Created dedicated graph execution queue with no_immediate_command_list\n");
        return graph_queue;
    }
#endif

    ggml_sycl_pool & host_pool(int device) {
        if (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        return *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { return host_pool(device); }

    const void ** get_graph_batched_src_ptrs(size_t count) {
        auto & slot = graph_ptrs_src_cache[count];
        if (!slot) {
            slot = std::make_unique<ggml_sycl_pool_alloc<const void *>>(pool(), count);
        }
        return slot->get();
    }

    void ** get_graph_batched_dst_ptrs(size_t count) {
        auto & slot = graph_ptrs_dst_cache[count];
        if (!slot) {
            slot = std::make_unique<ggml_sycl_pool_alloc<void *>>(pool(), count);
        }
        return slot->get();
    }

    bool enable_op_stats  = false;
    bool enable_op_timing = false;

    struct op_stat_entry {
        uint64_t count    = 0;
        double   total_ms = 0.0;
        double   min_ms   = std::numeric_limits<double>::max();
        double   max_ms   = 0.0;
    };

    std::map<std::string, op_stat_entry> op_stats;

#ifdef GGML_SYCL_GRAPH
    std::map<std::string, uint64_t> graph_fallback_reason_counts;
    std::map<std::string, uint64_t> graph_plan_counts;
    std::map<std::string, uint64_t> graph_plan_timing_us;

    void record_graph_fallback_reason(const std::string & reason) {
        if (!enable_op_stats) {
            return;
        }
        graph_fallback_reason_counts[reason]++;
    }

    void record_graph_plan_stat(const std::string & key, uint64_t value = 1) {
        if (!enable_op_stats) {
            return;
        }
        graph_plan_counts[key] += value;
    }

    void record_graph_plan_timing_us(const std::string & key, uint64_t value) {
        if (!enable_op_stats) {
            return;
        }
        graph_plan_timing_us[key] += value;
    }
#endif

    void record_op_stat(const std::string & key, double duration_ms) {
        if (!enable_op_stats) {
            return;
        }

        auto & entry = op_stats[key];
        entry.count++;
        if (enable_op_timing && duration_ms >= 0.0) {
            entry.total_ms += duration_ms;
            entry.min_ms = std::min(entry.min_ms, duration_ms);
            entry.max_ms = std::max(entry.max_ms, duration_ms);
        }
    }

    void print_op_stats() {
        if (!enable_op_stats || op_stats.empty()) {
            return;
        }

        std::vector<std::pair<std::string, op_stat_entry>> entries(op_stats.begin(), op_stats.end());
        std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) {
            if (a.second.total_ms == b.second.total_ms) {
                return a.second.count > b.second.count;
            }
            return a.second.total_ms > b.second.total_ms;
        });

        std::fprintf(stderr, "\n[SYCL OP STATS]%s\n", enable_op_timing ? " (timing)" : "");
#ifdef GGML_SYCL_GRAPH
        if (!graph_fallback_reason_counts.empty()) {
            std::fprintf(stderr, "[SYCL GRAPH FALLBACK REASONS]\n");
            for (const auto & kv : graph_fallback_reason_counts) {
                std::fprintf(stderr, "%s: count=%" PRIu64 "\n", kv.first.c_str(), kv.second);
            }
        }
        if (!graph_plan_counts.empty()) {
            std::fprintf(stderr, "[SYCL GRAPH PLAN STATS]\n");
            for (const auto & kv : graph_plan_counts) {
                std::fprintf(stderr, "%s: count=%" PRIu64 "\n", kv.first.c_str(), kv.second);
            }
        }
        if (!graph_plan_timing_us.empty()) {
            std::fprintf(stderr, "[SYCL GRAPH PLAN TIMING]\n");
            for (const auto & kv : graph_plan_timing_us) {
                std::fprintf(stderr, "%s: total_us=%" PRIu64 " total_ms=%.3f\n", kv.first.c_str(), kv.second,
                             (double) kv.second / 1000.0);
            }
        }
#endif
        for (const auto & item : entries) {
            const auto & entry = item.second;
            if (enable_op_timing && entry.count > 0) {
                const double avg_ms = entry.total_ms / static_cast<double>(entry.count);
                std::fprintf(stderr, "%s: count=%" PRIu64 " total=%.3fms avg=%.3fms min=%.3fms max=%.3fms\n",
                             item.first.c_str(), entry.count, entry.total_ms, avg_ms,
                             entry.min_ms == std::numeric_limits<double>::max() ? 0.0 : entry.min_ms, entry.max_ms);
            } else {
                std::fprintf(stderr, "%s: count=%" PRIu64 "\n", item.first.c_str(), entry.count);
            }
        }
        std::fprintf(stderr, "\n");
    }
};

// common device functions

static __dpct_inline__ float warp_reduce_sum(float x, const sycl::nd_item<3> & item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;
}

static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3> & item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(), mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(), mask);
    }
    return a;
}

template <int width = WARP_SIZE> static __dpct_inline__ int warp_reduce_sum(int x) {
    return sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

template <int width = WARP_SIZE> static __dpct_inline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
    }
    return x;
}

template <int width = WARP_SIZE> static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        a.x() +=
            dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset, width);
        a.y() +=
            dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset, width);
    }
    return a;
}

template <int width = WARP_SIZE> static __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        a = a + dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset, width);
    }
    return a;
}

static constexpr int ggml_sycl_get_physical_warp_size() {
    // todo: for old iGPU + dGPU case, need to be changed.
    return WARP_SIZE;
}

template <int width = WARP_SIZE> static __dpct_inline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = sycl::fmax(
            x, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width));
    }
    return x;
}

static __dpct_inline__ float warp_reduce_max(float x, const sycl::nd_item<3> & item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask));
    }
    return x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template <int N>
__dpct_inline__ size_t calculate_offset(const std::array<int, N> & strides, const std::array<int, N> & indices) {
    size_t offset = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    return offset;
}

// Helper for vec loading aligned data
template <typename Tp, int n> inline sycl::vec<Tp, n> vec_aligned_load(const Tp * aligned_ptr) {
    return *reinterpret_cast<const sycl::vec<Tp, n> *>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, int dim> static __dpct_inline__ Tp * get_pointer(sycl::local_accessor<Tp, dim> acc) {
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size);

constexpr size_t ceil_div(const size_t m, const size_t n) {
    return (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device & dev);

template <int N, class T> std::string debug_get_array_str(const std::string & prefix, const T array[N]) {
    if (LIKELY(!g_ggml_sycl_debug)) {
        return "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    for (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    if constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    return ss.str();
}

inline std::string debug_get_tensor_str(const std::string & prefix,
                                        const ggml_tensor * tensor,
                                        const std::string & suffix = "") {
    std::stringstream ss;
    if (LIKELY(!g_ggml_sycl_debug)) {
        return ss.str();
    }
    ss << prefix.c_str() << "=";
    if (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        if (!ggml_is_contiguous(tensor)) {
            ss << ";strided";
        }
        if (ggml_is_permuted(tensor)) {
            ss << ";permuted";
        }
    } else {
        ss << "nullptr";
    }
    ss << suffix;
    return ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
struct scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(const std::string_view & func,
                         const std::string_view & func_suffix,
                         const ggml_tensor *      dst,
                         std::size_t              num_src,
                         const std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        if (LIKELY(!g_ggml_sycl_debug)) {
            return;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        if (dst) {
            for (std::size_t i = 0; i < num_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(const std::string_view & func,
                         const ggml_tensor *      dst,
                         std::size_t              num_src,
                         const std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, num_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  private:
    std::string_view func;
    std::string_view func_suffix;
};

static __dpct_inline__ float get_alibi_slope(const float    max_bias,
                                             const uint32_t h,
                                             const uint32_t n_head_log2,
                                             const float    m0,
                                             const float    m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;

    return dpct::pow(base, exph);
}

static const sycl::uint3 init_fastdiv_values(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return sycl::uint3(mp, L, d);
}

static __dpct_inline__ uint32_t fastdiv(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_values.x());
    return (hi + n) >> fastdiv_values.y();
}

static __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z();
    return sycl::uint2(div_val, mod_val);
}

static __dpct_inline__ int ggml_sycl_dp4a(const int a, const int b, int c) {
    return dpct::dp4a(a, b, c);
}

static __dpct_inline__ float ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

#endif  // GGML_SYCL_COMMON_HPP
