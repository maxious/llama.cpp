# SYCL Flash Attention: GLM Model Compatibility Issues

## Overview

This document describes the compatibility issues between SYCL flash attention and models using non-contiguous (padded) tensors, particularly the GLM-4.7 family of models, and outlines potential solutions.

## Problem Summary

The SYCL flash attention implementation in llama.cpp currently requires strictly contiguous Q, K, V tensors. Models like GLM-4.7-Flash use padded KV cache layouts that fail the `ggml_is_contiguous()` check, causing flash attention to fall back to CPU execution with severe performance degradation.

## Observed Behavior

### Performance Impact

| Configuration | Generation Speed | Notes |
|--------------|------------------|-------|
| Non-FA baseline | ~27 t/s | Standard GEMM path |
| FA (working) | 17.7 → 27 t/s | After batched GEMM optimization |
| FA (fallback to CPU) | 8.7 t/s | When non-contiguous detected |

### Tensor Layout Analysis

GLM-4.7-Flash produces tensors with padding in their strides:

```
Q tensor:
  ne = [576, 512, 20, 1]     # shape: [head_dim, seq_len, n_heads, batch]
  nb = [4, 46080, 2304, 23592960]  # strides in bytes
  
  Expected contiguous nb[1] = 576 * 4 = 2304
  Actual nb[1] = 46080  ← indicates padding

V tensor:
  ne = [512, 512, 1, 1]
  nb = [2, 1152, 1152, 589824]  # F16 data (element size = 2)
  
  Expected contiguous nb[1] = 512 * 2 = 1024
  Actual nb[1] = 1152  ← indicates padding
```

### Root Cause

In `ggml/src/ggml-sycl/fattn.cpp`, the support check rejects non-contiguous tensors:

```cpp
// Line 158
if (!ggml_is_contiguous(Q) || !ggml_is_contiguous(K) || !ggml_is_contiguous(V)) {
    return false;
}
```

When `ggml_sycl_flash_attn_ext_supported()` returns `false`, the operation falls back to CPU execution rather than using the optimized SYCL kernels.

## CUDA Reference Implementation

The CUDA backend handles this case correctly using two key mechanisms:

### 1. Relaxed Contiguity Check

From `ggml/src/ggml-cuda/fattn-common.cuh`:

```cpp
// Uses ggml_is_contiguously_allocated instead of ggml_is_contiguous
// This checks if data is in one memory block (allowing padding)
```

### 2. Stride-Aware Data Conversion

```cpp
template<int vals_smem, int head_dim>
static __device__ void dequantize_and_convert_to_half_value(
    const void * src, sycl::half dst[vals_smem], 
    int head_dim_actual, int stride_tokens, int stride_heads) {
    // Handles non-contiguous access with explicit strides
}
```

The CUDA implementation passes stride parameters through the kernel launch and uses them for correct memory access patterns.

## Proposed Solutions

### Solution 1: Stride-Aware Reordering Kernels (Recommended)

Modify the SYCL dequantization/reordering kernels to accept stride parameters:

```cpp
// Current implementation (assumes contiguous):
Q_d_f32_alloc[head * N * DQK + seq * DQK + dim] = 
    Q_d[dim + head * q_stride_head + seq * q_stride_seq];

// Modified implementation (stride-aware):
// Use actual byte strides from tensor->nb[] converted to element strides
const int64_t q_stride_elem = Q->nb[1] / sizeof(float);  // Row stride in elements
const int64_t q_head_stride = Q->nb[2] / sizeof(float);  // Head stride in elements

// Access with actual strides
Q_d_f32_alloc[head * N * DQK + seq * DQK + dim] = 
    Q_d[dim + head * q_head_stride + seq * q_stride_elem];
```

**Changes Required:**
1. Replace `ggml_is_contiguous()` check with `ggml_is_contiguously_allocated()` in support function
2. Modify all reordering kernel loops to use tensor stride values from `nb[]`
3. Add stride parameters to kernel launch configurations

### Solution 2: Pre-Compaction Pass

Add a memory compaction step before flash attention that copies padded tensors into contiguous buffers:

```cpp
if (!ggml_is_contiguous(Q) && ggml_is_contiguously_allocated(Q)) {
    // Allocate contiguous buffer
    float* Q_compact = sycl::malloc_device<float>(N * DQK * n_heads, *stream);
    
    // Copy with proper stride handling
    compact_tensor_kernel<<<...>>>(Q_d, Q_compact, Q->ne, Q->nb);
    
    // Use compacted buffer for FA
    Q_d = Q_compact;
}
```

**Trade-offs:**
- Simpler kernel modifications
- Additional memory allocation and copy overhead
- May not be faster than inline stride handling

### Solution 3: Tensor View Optimization (Upstream Fix)

Investigate why GLM models produce padded tensors and whether the padding can be eliminated at the tensor creation level:

- Check KV cache allocation in `llama.cpp`
- Verify if padding is intentional for alignment or a side effect
- Consider model-specific KV cache layouts

## Implementation Priority

1. **Short-term**: Solution 1 (stride-aware kernels) - matches CUDA approach
2. **Medium-term**: Benchmark Solution 2 vs Solution 1 for specific models
3. **Long-term**: Solution 3 if padding is unnecessary

## Testing Recommendations

1. Test with GLM-4.7-Flash (head sizes: K=576, V=512)
2. Test with standard models (Llama, Mistral) to ensure no regression
3. Benchmark all three solutions on Intel Arc GPUs (Battlemage/Xe2)

## Related Files

- `ggml/src/ggml-sycl/fattn.cpp` - SYCL flash attention implementation
- `ggml/src/ggml-sycl/ggml-sycl.cpp` - SYCL backend main file  
- `ggml/src/ggml-cuda/fattn-common.cuh` - CUDA reference implementation
- `ggml/src/ggml.c` - `ggml_is_contiguous()` and `ggml_is_contiguously_allocated()` definitions

## Environment

- Hardware: Dual Intel Arc Pro B60 GPUs (Battlemage/Xe2 architecture)
- Model: GLM-4.7-Flash with asymmetric head sizes (K=576, V=512)
- GQA ratio: 20 (20 Q heads per 1 KV head)
