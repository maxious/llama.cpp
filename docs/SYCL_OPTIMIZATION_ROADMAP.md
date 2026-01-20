# llama.cpp SYCL Backend Optimization Roadmap

Based on analysis of Intel GPU optimization guides and the current llama.cpp SYCL implementation.

## Executive Summary

The llama.cpp SYCL backend has extensive coverage with **XMX-accelerated flash attention now working** (3-4x speedup). This document outlines actionable improvements targeting Intel Arc GPUs (Xe2/Xe3 architecture), focusing on shared local memory (SLM), unified shared memory (USM), and multi-GPU optimizations.

---

## Current Implementation Status

### Implemented Operations (30+ files)

| Category | Operations |
|----------|-----------|
| Matrix Ops | `mul_mat`, `mmq`, `dmmv`, `mmvq`, `out_prod` |
| Attention | `flash_attn_ext`, `fattn`, `gla`, `wkv` |
| Normalization | `rms_norm`, `layer_norm`, `group_norm` |
| Activations | `gelu`, `silu`, `swiglu`, `geglu`, `relu`, +20 more |
| Data Ops | `cpy`, `getrows`, `set_rows`, `concat`, `pad`, `rope` |

### Optimization Status Matrix

| Feature | Status | Priority |
|---------|--------|----------|
| XMX Flash Attention | ✅ **WORKING** (3-4x speedup) | Done |
| XMX Quantized MatMul | ❌ Removed (dp4a optimal) | N/A |
| Shared Local Memory (SLM) | ⚠️ Flash attention only | P1 |
| Shared USM | ❌ Not implemented | P2 |
| Multi-GPU (Xe Link) | ⚠️ Basic support | P2 |
| Async Memory Ops | ⚠️ Limited | P2 |

---

## Completed: XMX Flash Attention (TODO-001)

**Status**: ✅ **COMPLETE**

**Files**: `ggml/src/ggml-sycl/fattn.cpp`, `ggml/src/ggml-sycl/fattn_kernel.hpp`

### Implementation Summary

1. **XMX detection** with runtime query of `matrix_combinations`
2. **Templated kernel** `flash_attn_coopmat_kernel<HEAD_DIM>` with bfloat16 matrices
3. **Architecture-specific paths**:
   - DG2/Arc B60 (Xe2): 8x16x16 tiles
   - PVC (Ponte Vecchio): 16x16x16 tiles
4. **Automatic fallback** to oneMKL when XMX unavailable

### Performance Results (Intel Arc B60)

| Path | Speedup vs oneMKL |
|------|-------------------|
| XMX (cooperative matrix) | **3-4x faster** |
| oneMKL GEMM fallback | Baseline |

### Environment Variables

```bash
# Disable XMX kernel (use oneMKL fallback)
GGML_SYCL_FLASH_ATTN_XMX=0

# Force oneMKL even when XMX is available
GGML_SYCL_FLASH_ATTN_MKL=1

# Enable debug logging
GGML_SYCL_FLASH_ATTN_DEBUG=1
```

---

## Closed: XMX for Quantized MatMul (TODO-002)

**Status**: ❌ **REMOVED** (2026-01-20)

**Rationale**: XMX is not beneficial for quantized matmul because:
1. **Dequantization overhead**: XMX requires bf16/fp16 inputs, so quantized weights must be dequantized first
2. **Batch size mismatch**: XMX tiles (8x16, 16x16) can't be efficiently filled for decode (batch=1)
3. **K-quants complexity**: Per-block scale/min values don't map well to GEMM patterns
4. **dp4a is optimal**: The existing int8 dot product path is hardware-accelerated and avoids dequantization

**Current Implementation**: All quantized types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) use the dp4a path with optimized tile sizes (64x128 for most types).

---

## Closed: Hardware Detection Utility (TODO-003)

**Status**: ✅ **Implemented in common.hpp**

XMX detection is now integrated into `fattn.cpp`:
- `ggml_sycl_flash_attn_has_xmx()` - checks for cooperative matrix support
- `ggml_sycl_flash_attn_get_tile_kind()` - returns architecture-specific tile sizes

The `sycl_hw.cpp/hpp` files remain as stubs for future expansion.

---

## Priority P1: Shared Local Memory (SLM) Optimization

### Issue: SLM Underutilized

Currently, SLM is only used in flash attention. Other operations could benefit significantly.

### Closed: SLM Tiling for Non-Quantized MatMul (TODO-004)

**Status**: ❌ **REMOVED** (2026-01-20)

**Rationale**: The `gemm.hpp` file is a thin wrapper around oneDNN, which already implements SLM-tiled GEMM internally. Writing a custom kernel would be substantial effort with marginal gains. oneDNN's implementation is already optimized for Intel hardware.

---

### TODO-005: SLM for Layer Normalization

**Status**: ✅ **COMPLETE**

**File**: `ggml/src/ggml-sycl/norm.cpp`

**Implementation**: Added `norm_f32_slm` and `rms_norm_f32_slm` kernels that cache input rows in shared local memory (SLM) during the first pass, avoiding a second global memory read.

**Applies to**: Rows with 256-4096 elements (typical transformer hidden sizes: 768, 1024, 2048, 4096)

**Benefit**: 50% reduction in global memory reads for LayerNorm and RMSNorm operations.

**Technical Details**:
```cpp
// First pass: load into SLM and compute statistics
for (int col = tid; col < ncols; col += block_size) {
    const float xi = x[col];
    s_row[col] = xi;  // Cache in SLM (16KB for 4096 floats)
    tmp += xi * xi;   // Compute sum of squares
}
// ... reduction ...
// Second pass: read from SLM instead of global memory
for (int col = tid; col < ncols; col += block_size) {
    dst[col] = scale * s_row[col];
}
```

---

### Closed: Bank Conflict Avoidance (TODO-006)

**Status**: ✅ **COMPLETE** (already implemented in XMX kernel)

**File**: `ggml/src/ggml-sycl/fattn_kernel.hpp`

The XMX flash attention kernel already has bank conflict padding on all SLM arrays:
```cpp
constexpr int Q_STRIDE = HEAD_DIM + 8;  // Padding for bank conflict avoidance
constexpr int K_STRIDE = HEAD_DIM + 8;
constexpr int V_STRIDE = HEAD_DIM + 8;
constexpr int V_T_STRIDE = BLOCK_N + 8;
constexpr int S_STRIDE = BLOCK_N + 8;
constexpr int P_STRIDE = BLOCK_N + 8;
```

The non-XMX fallback kernel allocates SLM but doesn't actually use it for computation (reads from global memory), so adding padding there would have no effect.

---

## Priority P2: Memory Optimizations

### TODO-007: Shared USM for KV Cache

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Current**: Device-only allocations.

**Desired**: Shared USM with explicit prefetching for KV cache.

```cpp
// Shared USM for KV cache - can be accessed by both CPU and GPU
void * kv_cache = sycl::malloc_shared(kv_size, queue);

// Prefetch to device before attention computation
queue.prefetch(kv_cache, kv_size);
```

---

### TODO-008: Async Prefetching

**File**: `ggml/src/ggml-sycl/common.cpp`

**Add**: Overlap data transfers with computation using async prefetch.

```cpp
// Double-buffer pattern
queue.prefetch(next_chunk, size);  // Prefetch next
compute_kernel(current_chunk);     // Compute current
queue.wait();                      // Sync
```

---

### Closed: Memory Alignment (TODO-009)

**Status**: ✅ **COMPLETE** (2026-01-20)

**File**: `ggml/src/ggml-sycl/common.hpp`, `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Implementation**: Added `ggml_sycl_aligned_malloc_device()` helper with 64-byte alignment. Updated the three main allocation sites in `ggml-sycl.cpp`:
- `ggml_backend_sycl_buffer_type_alloc_buffer()` - main buffer allocator
- `ggml_backend_sycl_split_buffer_init_tensor()` - split buffer allocator
- Memory pool allocator

```cpp
constexpr size_t SYCL_DEVICE_MEM_ALIGNMENT = 64;

inline void * ggml_sycl_aligned_malloc_device(size_t size, sycl::queue * q) {
    return sycl::aligned_alloc_device(SYCL_DEVICE_MEM_ALIGNMENT, size, *q);
}
```

---

## Priority P2: Multi-XPU Optimization

### TODO-010: Fix Row Split Mode

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Current**: Row split causes GPU page faults during inference.

**Status**: ⚠️ Blocked by driver/runtime issue (2026-01-20)

**Symptoms**:
- GPU page faults at low virtual addresses (~0x200bb000) on compute shader engine
- dmesg shows `Fault response: Unsuccessful -ENOENT` indicating unmapped memory access
- Faults occur on device 0 (xe 0000:03:00.0) ccs (compute shader) engine
- Kernel submissions succeed but kernel execution fails with invalid memory access
- The faulted addresses are far from valid device allocations (0xffffeaab... range)
- Layer split mode works fine; only row split mode fails

**Root Cause Analysis (Updated 2026-01-20)**:
Extensive debugging with fprintf tracing revealed:

1. **Kernel submission succeeds** - quantize_row_q8_1_sycl and mul_mat kernels are submitted to both devices
2. **Kernel execution fails on device 0** - page faults at ~500MB address range while valid pointers are at ~0xffffeaab...
3. **Cross-device sync works intermittently** - some layers complete successfully before crash
4. **The issue is in kernel execution, not SYCL API usage** - all pointers verified valid via `sycl::get_pointer_type()`

**Technical Details**:
The GPU page faults show:
- `ASID: 260` - Address Space ID for the process
- `Faulted Address: 0x00000000200bb000` - Very low VA, not in USM range
- `EngineClass: 5 ccs` - Compute shader engine
- `FaultLevel: 1` - Page table lookup failure

This pattern suggests the kernel is dereferencing a corrupted or null pointer internally, likely:
- A stack/register corruption from multi-device context switching
- Level Zero runtime bug with USM allocations across multiple devices
- Xe driver bug with compute queue management across GPUs

**Fixes Applied**:
- Disabled P2P enablement (caused device lost errors)
- Removed cross-device event barriers (use direct queue waits)
- Added host-mediated copies for all cross-device transfers
- Verified pointer types and stream devices
- Removed synchronous waits inside compute loop (caused deadlocks)

**Workaround**: Use `--split-mode layer` instead of `--split-mode row`

**Next Steps**:
1. File Intel driver bug report with dmesg output and reproduction steps
2. Test with newer Intel GPU drivers (currently using 1.14.36711+4)
3. Test with compute-runtime 25.x when available
4. Consider allocating split buffers with explicit device placement

**Key Code Locations**:
- `ggml_backend_sycl_split_buffer_init_tensor()` (line 892) - split buffer allocation
- `ggml_sycl_op_mul_mat()` (line 2408) - main mul_mat function
- `quantize_row_q8_1_sycl()` (quantize.hpp:121) - one of the failing kernels
- `dev2dev_memcpy_2d()` (line 485) - host-mediated cross-device copy

**Driver Info**:
- Intel compute-runtime: 1.14.36711+4
- Kernel xe driver: Ubuntu 25.10
- Hardware: Intel Arc Pro B60 (Xe2/Battlemage) x2

---

### TODO-011: Shared USM for Multi-XPU Efficiency

**File**: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Current**: Each device has separate device memory allocations. Cross-device copies go through host fallback.

**Desired**: Use shared USM for tensors accessed by multiple XPUs to avoid explicit copies.

**Implementation**:
```cpp
// Shared USM - accessible from all devices and host
// Ideal for KV cache and split tensors
void * shared_alloc(size_t size, sycl::queue & q) {
    return sycl::malloc_shared(size, q);
}

// Device USM with prefetch hint
void * device_alloc_with_prefetch(size_t size, sycl::queue & q) {
    void * ptr = sycl::malloc_device(size, q);
    q.prefetch(ptr, size);  // Hint to migrate pages to device
    return ptr;
}
```

**Benefits for Multi-XPU**:
- No explicit host-mediated copies needed
- Runtime handles page migration automatically
- Reduces latency for cross-device tensor access

**Trade-offs**:
- Slightly higher latency than device-only memory for single-device access
- Requires USM support (all Intel GPUs support this)

---

## Priority P3: Additional Optimizations

### TODO-012: Kernel Fusion Patterns

**Pattern 1: MatMul + Add + GELU (Gated MLP)**
```
Current: 3 separate kernels
Optimal: 1 fused kernel with SLM
```

**Pattern 2: RMSNorm + Residual**
```
Current: RMSNorm kernel, add kernel
Optimal: Fused kernel
```

---

### TODO-013: Performance Profiling Infrastructure

```cpp
class ggml_sycl_profiler {
public:
    void start_timer(const char * name);
    void stop_timer(const char * name);
    void print_report();  // kernel_name, time, bandwidth, FLOPs
private:
    std::map<std::string, std::vector<sycl::event>> events;
};
```

---

## Build Configuration

### Enable Optimizations

```bash
cmake -B build \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=ON \
    -DGGML_SYCL_TARGET=INTEL \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx

# Environment variables for profiling
export SYCL_PI_LEVEL_ZERO_TRACK_USM_SIZES=1
export SYCL_PI_LEVEL_ZERO_DEBUG=1
```

---

## Testing Checklist

### Unit Tests
- [x] `test-backend-ops -b SYCL0 -o SOFT_MAX` (passes)
- [x] `test-backend-ops -b SYCL0 -o MUL_MAT` (passes)
- [x] `test-backend-ops -b SYCL0 -o FLASH_ATTN` (passes with XMX)

### Integration Tests
- [x] Single GPU inference (working)
- [ ] Dual GPU layer split
- [ ] Dual GPU row split (currently crashes)
- [ ] KV cache prefetching

### Performance Benchmarks
- [x] Flash attention: XMX vs oneMKL (3-4x faster)
- [ ] USM vs device-only memory patterns
- [ ] Multi-GPU scaling efficiency

---

## References

### Intel Documentation
1. [Programming Intel XMX Using SYCL Joint Matrix Extension](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/programming-intel-xmx-using-sycl-joint-matrix.html)
2. [Shared Local Memory Optimization](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html)
3. [Unified Shared Memory Allocations](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/unified-shared-memory-allocations.html)

### Code References
- `ggml/src/ggml-sycl/fattn.cpp` - XMX flash attention (working)
- `ggml/src/ggml-sycl/fattn_kernel.hpp` - XMX kernel implementation
- `ggml/src/ggml-sycl/mmq.cpp` - dp4a quantized matmul (optimized)
- `ggml/src/ggml-sycl/common.hpp` - Hardware detection utilities

---

## Summary

| Priority | Action Item | Status | Notes |
|----------|-------------|--------|-------|
| P0 | XMX flash attention | ✅ Complete | 3-4x speedup |
| P0 | XMX quantized MatMul | ❌ Removed | dp4a is optimal |
| P0 | Hardware detection | ✅ Complete | In fattn.cpp |
| P0 | Dead code cleanup | ✅ Complete | Removed XMX from mmq.cpp |
| P1 | SLM for non-quantized GEMM | ❌ Removed | oneDNN handles internally |
| P1 | SLM for normalization | ✅ Complete | norm.cpp |
| P1 | Bank conflict avoidance | ✅ Complete | fattn_kernel.hpp (XMX path) |
| P2 | Fix multi-XPU row split | ⚠️ Blocked | Driver/runtime issue on Intel Arc |
| P2 | Shared USM for multi-XPU | Pending | ggml-sycl.cpp |
| P2 | Async prefetching | Pending | common.cpp |
| P2 | Memory alignment (64-byte) | ✅ Complete | common.hpp, ggml-sycl.cpp |
| P3 | Kernel fusion | Pending | Multiple files |
| P3 | Profiling infrastructure | Pending | New file |

**Completed**: Flash attention XMX (3-4x speedup), dead code cleanup, debug logging gated, SLM normalization, bank conflict avoidance, 64-byte memory alignment
**Blocked**: Row split mode blocked by driver issue (use `--split-mode layer` as workaround)
**Next Priority**: Async prefetching, shared USM for KV cache

---

*Document updated: 2026-01-20*
*Target hardware: Intel Arc Pro B60 (Xe2), Data Center GPU Flex 140 (Xe2), Data Center GPU Max 1550 (PVC)*
