//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_DNNL_GRAPH_HPP
#define GGML_SYCL_DNNL_GRAPH_HPP

#if GGML_SYCL_DNNL

#include <unordered_map>
#include <tuple>
#include <string>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "common.hpp"

/**
 * Wrapper for oneDNN Graph API to provide graph-compatible GEMM operations.
 *
 * This class encapsulates the oneDNN Graph API for MatMul operations, providing:
 * - Graph construction and compilation for F16/F32 data types
 * - Caching of compiled_partition objects by shape/type signature
 * - Execution on SYCL queues (compatible with sycl::command_graph recording)
 */
class DnnlGraphWrapper {
public:
    using dims = dnnl::graph::logical_tensor::dims;
    using data_type = dnnl::graph::logical_tensor::data_type;

    /**
     * Construct a new DnnlGraphWrapper
     * @param ctx SYCL backend context (provides engine/stream interop)
     */
    DnnlGraphWrapper(ggml_backend_sycl_context &ctx) : ctx_(ctx) {}

    /**
     * Execute a MatMul operation using oneDNN Graph API
     *
     * This will either retrieve a cached compiled partition or build/compile a new one.
     *
     * @param src_ptr Pointer to source tensor data
     * @param weights_ptr Pointer to weights tensor data
     * @param dst_ptr Pointer to destination tensor data
     * @param src_dims Dimensions of source tensor
     * @param weights_dims Dimensions of weights tensor
     * @param dst_dims Dimensions of destination tensor
     * @param src_data_type oneDNN data type for source (f16 or f32)
     * @param weights_data_type oneDNN data type for weights (f16 or f32)
     * @param dst_data_type oneDNN data type for destination (f16 or f32)
     * @param transpose_a Whether to transpose source (default: false)
     */
    void execute_matmul(
        const void *src_ptr,
        const void *weights_ptr,
        void *dst_ptr,
        const dims &src_dims,
        const dims &weights_dims,
        const dims &dst_dims,
        data_type src_data_type,
        data_type weights_data_type,
        data_type dst_data_type,
        bool transpose_a = false) {

        // Build cache key
        CacheKey key = {
            src_dims,
            weights_dims,
            dst_dims,
            src_data_type,
            weights_data_type,
            dst_data_type,
            transpose_a
        };

        // Get or compile the partition
        const auto &cp = get_or_compile(key);

        // Get SYCL queue from context
        sycl::queue *q = ctx_.stream();
        GGML_ASSERT(q != nullptr);

        // Get oneDNN engine and stream with proper interop
        dnnl::engine eng = ctx_.engine_dnnl(q);
        dnnl::stream strm = ctx_.stream_dnnl(q);

        // Create logical tensors (IDs start at 0)
        size_t id = 0;
        auto src_lt = dnnl::graph::logical_tensor(id++, src_data_type, src_dims, dnnl::graph::logical_tensor::layout_type::strided);
        auto weights_lt = dnnl::graph::logical_tensor(id++, weights_data_type, weights_dims, dnnl::graph::logical_tensor::layout_type::strided);
        auto dst_lt = dnnl::graph::logical_tensor(id++, dst_data_type, dst_dims, dnnl::graph::logical_tensor::layout_type::strided);

        // Create graph tensors with user-provided memory handles
        dnnl::graph::tensor src_tensor(src_lt, eng, const_cast<void*>(src_ptr));
        dnnl::graph::tensor weights_tensor(weights_lt, eng, const_cast<void*>(weights_ptr));
        dnnl::graph::tensor dst_tensor(dst_lt, eng, dst_ptr);

        // Execute: cp.execute(stream, {inputs}, {outputs})
        cp.execute(strm, {src_tensor, weights_tensor}, {dst_tensor});
    }

private:
    ggml_backend_sycl_context &ctx_;

    // Cache key structure
    struct CacheKey {
        dims src_dims;
        dims weights_dims;
        dims dst_dims;
        data_type src_type;
        data_type weights_type;
        data_type dst_type;
        bool transpose_a;

        bool operator==(const CacheKey &other) const {
            return src_dims == other.src_dims &&
                   weights_dims == other.weights_dims &&
                   dst_dims == other.dst_dims &&
                   src_type == other.src_type &&
                   weights_type == other.weights_type &&
                   dst_type == other.dst_type &&
                   transpose_a == other.transpose_a;
        }
    };

    struct CacheKeyHash {
        size_t operator()(const CacheKey &k) const {
            size_t h1 = hash_dims(k.src_dims);
            size_t h2 = hash_dims(k.weights_dims);
            size_t h3 = hash_dims(k.dst_dims);
            size_t h4 = std::hash<int>()(static_cast<int>(k.src_type));
            size_t h5 = std::hash<int>()(static_cast<int>(k.weights_type));
            size_t h6 = std::hash<int>()(static_cast<int>(k.dst_type));
            size_t h7 = std::hash<bool>()(k.transpose_a);

            // Combine hashes (using boost's hash_combine style)
            size_t seed = h1;
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h5 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h6 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h7 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }

        size_t hash_dims(const dims &d) const {
            size_t seed = 0;
            for (auto v : d) {
                seed ^= std::hash<int64_t>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    std::unordered_map<CacheKey, dnnl::graph::compiled_partition, CacheKeyHash> cache_;

    /**
     * Get a compiled partition from cache or build a new one
     */
    const dnnl::graph::compiled_partition &get_or_compile(const CacheKey &key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }

        // Build graph and compile
        dnnl::graph::compiled_partition cp = compile_partition(key);
        auto [new_it, inserted] = cache_.emplace(key, std::move(cp));
        GGML_ASSERT(inserted);
        return new_it->second;
    }

    /**
     * Build a oneDNN Graph for MatMul and compile it
     */
    dnnl::graph::compiled_partition compile_partition(const CacheKey &key) {
        try {
        // Get SYCL queue from context and create oneDNN engine
        sycl::queue *q = ctx_.stream();
        GGML_ASSERT(q != nullptr);
        dnnl::engine eng = ctx_.engine_dnnl(q);
        dnnl::stream strm = ctx_.stream_dnnl(q);

        GGML_LOG_DEBUG("DnnlGraphWrapper: engine kind = %d, queue = %p\n", (int)eng.get_kind(), (void*)q);

        // Incremental ID for logical tensors and ops
            size_t id = 0;

        // Create logical tensors: src, weights, dst
        auto src_lt = dnnl::graph::logical_tensor(id++, key.src_type, key.src_dims, dnnl::graph::logical_tensor::layout_type::strided);
        auto weights_lt = dnnl::graph::logical_tensor(id++, key.weights_type, key.weights_dims, dnnl::graph::logical_tensor::layout_type::strided);
        auto dst_lt = dnnl::graph::logical_tensor(id++, key.dst_type, key.dst_dims, dnnl::graph::logical_tensor::layout_type::strided);

        GGML_LOG_DEBUG("DnnlGraphWrapper: tensors created: src dims=[%ld,%ld,%ld] type=%d, weights dims=[%ld,%ld,%ld] type=%d, dst dims=[%ld,%ld,%ld] type=%d\n",
            key.src_dims.size() > 0 ? key.src_dims[0] : -1,
            key.src_dims.size() > 1 ? key.src_dims[1] : -1,
            key.src_dims.size() > 2 ? key.src_dims[2] : -1, (int)key.src_type,
            key.weights_dims.size() > 0 ? key.weights_dims[0] : -1,
            key.weights_dims.size() > 1 ? key.weights_dims[1] : -1,
            key.weights_dims.size() > 2 ? key.weights_dims[2] : -1, (int)key.weights_type,
            key.dst_dims.size() > 0 ? key.dst_dims[0] : -1,
            key.dst_dims.size() > 1 ? key.dst_dims[1] : -1,
            key.dst_dims.size() > 2 ? key.dst_dims[2] : -1, (int)key.dst_type);

        // Create MatMul op
            dnnl::graph::op matmul_op(id++, dnnl::graph::op::kind::MatMul, "matmul");
            if (key.transpose_a) {
                matmul_op.set_attr<bool>(dnnl::graph::op::attr::transpose_a, true);
            }
            matmul_op.add_inputs({src_lt, weights_lt});
            matmul_op.add_outputs({dst_lt});

            // Build graph with engine kind
            dnnl::graph::graph g(eng.get_kind());
            g.add_op(matmul_op);
            g.finalize();

            // Get partitions (expect exactly 1)
            std::vector<dnnl::graph::partition> partitions = g.get_partitions();
            if (partitions.size() != 1) {
                GGML_LOG_ERROR("DnnlGraphWrapper: Expected 1 partition, got %zu\n", partitions.size());
                GGML_ABORT("Invalid oneDNN Graph partition count");
            }

        // Compile partition with concrete I/O tensors
        std::vector<dnnl::graph::logical_tensor> inputs = {src_lt, weights_lt};
        std::vector<dnnl::graph::logical_tensor> outputs = {dst_lt};
        dnnl::graph::compiled_partition cp = partitions[0].compile(inputs, outputs, eng);

            GGML_LOG_DEBUG("DnnlGraphWrapper: Compiled MatMul graph (src=%s, weights=%s, dst=%s, transpose_a=%d)\n",
                to_string(key.src_type), to_string(key.weights_type), to_string(key.dst_type), key.transpose_a);

            return cp;
        } catch (const dnnl::error &e) {
            GGML_LOG_ERROR("DnnlGraphWrapper: oneDNN error during graph compilation: %s (status=%d)\n", e.message, static_cast<int>(e.status));
            throw; // rethrow
        }
    }

    // Helper to convert data_type to string for logging
    const char* to_string(data_type dt) {
        switch (dt) {
            case data_type::f16: return "f16";
            case data_type::f32: return "f32";
            case data_type::bf16: return "bf16";
            default: return "unknown";
        }
    }
};

#endif // GGML_SYCL_DNNL

#endif // GGML_SYCL_DNNL_GRAPH_HPP
