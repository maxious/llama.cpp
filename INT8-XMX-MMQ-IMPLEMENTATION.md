# INT8 XMX MMQ Implementation Summary

## Overview

This document summarizes the implementation of INT8 XMX (Intel Xe Matrix Extensions) for MMQ (Multi-precision Matrix Quantization) in llama.cpp's SYCL backend.

## Motivation

### Current MMQ Limitations

MMQ currently uses `dp4a` (int8 dot product) which:
- Computes 4 int8 multiplications per instruction
- Limited to small operations per cycle
- Requires unpacking quantized weights element-wise
- Has dequantization overhead

### INT8 XMX Advantages

INT8 XMX provides significant performance improvements:
- **2x higher throughput**: INT8 XMX has 4,096 ops/clock vs 2,048 for FP16/BF16
- **Larger tile processing**: XMX can process 8x16x16 or 16x16x16 tiles
- **Avoids dequantization**: Keeps data in INT8 format throughout
- **Better memory efficiency**: INT8 uses half the memory of FP16

## Implementation Details

### Files Created

1. **`ggml/src/ggml-sycl/mmq_xmx_int8.hpp`**
   - Header file with INT8 XMX MMQ kernel declarations
   - Hardware detection functions
   - Tile configuration utilities

2. **`ggml/src/ggml-sycl/mmq_xmx_int8.cpp`**
   - Implementation of INT8 XMX MMQ kernels
   - Integration with existing MMQ infrastructure
   - Fallback to dp4a when INT8 XMX is not supported

3. **`bench-int8-xmx-mmq.sh`**
   - Benchmark script to compare INT8 XMX vs dp4a performance
   - Tests different batch sizes
   - Generates performance reports

4. **`test-int8-xmx-support.cpp`**
   - Test program to detect INT8 XMX hardware support
   - Displays available tile configurations
   - Identifies supported architectures

### Key Features

#### Hardware Detection

```cpp
bool has_int8_xmx_support(const dpct::queue_ptr &q);
```

Checks if the device supports INT8 XMX by querying matrix combinations.

#### Tile Configuration

```cpp
xmx_int8_tile_config get_int8_xmx_tile_config(const dpct::queue_ptr &q);
```

Returns optimal tile sizes for the device:
- **PVC/B60**: 8x16x32 tiles
- **DG2**: 8x8x32 tiles
- **AMX**: 16x16x64 tiles

#### Kernel Implementation

**Q8_0 Quantization Kernel:**
```cpp
template <int TM, int TN, int TK>
static void mmq_q8_0_xmx_kernel(
    const block_q8_0 * __restrict__ vx,
    const block_q8_1 * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y,
    const sycl::nd_item<3> &item_ct1);
```

Uses INT8 joint_matrix for operands and int32_t for accumulator.

**Q4_K Quantization Kernel:**
```cpp
template <int TM, int TN, int TK>
static void mmq_q4_K_xmx_kernel(
    const block_q4_K * __restrict__ vx,
    const block_q8_1 * __restrict__ vy,
    float * __restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y,
    const sycl::nd_item<3> &item_ct1);
```

Handles more complex quantization with per-block scaling factors.

### SYCL Joint Matrix Usage

```cpp
joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major> matX;
joint_matrix<sub_group, int8_t, use::b, TK, TN, layout::col_major> matY;
joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> matC;

joint_matrix_fill(sg, matC, 0);

for (int k = 0; k < K; k += TK) {
    joint_matrix_load(sg, matX, X_ptr + offset, stride);
    joint_matrix_load(sg, matY, Y_ptr + offset, stride);
    joint_matrix_mad(sg, matC, matX, matY, matC);
}

joint_matrix_store(sg, matC, C_ptr, stride, layout::row_major);
```

## Hardware Support

### Supported Architectures

| Architecture | Tile Size | Peak Performance |
|--------------|-----------|------------------|
| Intel PVC/B60 | 8x16x32 | ~220 TFlops |
| Intel DG2 | 8x8x32 | ~45 TFlops |
| Intel AMX (CPU) | 16x16x64 | ~60 TFlops |

### Data Types

- **Operands**: `int8_t` (signed 8-bit integers)
- **Accumulator**: `int32_t` (32-bit integers to avoid overflow)
- **Result**: Converted to `float` after accumulation

## Performance Expectations

### Theoretical Performance

Based on hardware specifications:
- **INT8 XMX on B60**: ~220 TFlops peak
- **Current dp4a**: Limited by SIMD instruction throughput
- **Expected speedup**: 1.5-2x for large batch sizes

### Factors Affecting Performance

1. **Model size**: Larger models benefit more from XMX acceleration
2. **Batch size**: Larger batches show better speedup
3. **Quantization level**: Q8_0 shows best results, Q4_K needs more optimization
4. **Memory bandwidth**: May be bottleneck for small models

## Integration with Existing Code

### Fallback Mechanism

The implementation includes automatic fallback to dp4a:
1. Check if INT8 XMX is supported
2. If not supported, use existing dp4a implementation
3. If quantization type is not supported, use dp4a

### Build Integration

To enable INT8 XMX MMQ:
1. Add `mmq_xmx_int8.cpp` to CMakeLists.txt
2. Link with SYCL matrix extension
3. Add compile flag for matrix support

## Testing

### Hardware Detection Test

```bash
# Compile and run hardware detection test
icpx -fsycl -fsycl-targets=spir64 test-int8-xmx-support.cpp -o test-int8-xmx
./test-int8-xmx
```

### Benchmark Test

```bash
# Run benchmark comparing INT8 XMX vs dp4a
./bench-int8-xmx-mmq.sh /path/to/model.gguf
```

## Next Steps

### Immediate Tasks

1. **Fix compilation errors**: Resolve LSP errors in implementation files
2. **Add to build system**: Integrate with CMakeLists.txt
3. **Test on B60**: Verify INT8 XMX support and performance
4. **Benchmark**: Run performance comparisons

### Future Optimizations

1. **Complete scaling factor handling**: Implement proper per-block scaling
2. **Add more quantization types**: Support Q2_K, Q3_K, Q5_K, Q6_K
3. **Optimize memory access**: Improve data layout for better cache utilization
4. **Add profiling**: Identify bottlenecks and optimize further

## References

- Intel SYCL Joint Matrix Documentation
- SYCL Test Suite: `~/llvm/sycl/test-e2e/Matrix/`
- Intel Arc B60 Specifications
- ozIMMU and cuMpSGEMM repositories (for reference)

## Conclusion

INT8 XMX MMQ is a promising optimization that can provide significant performance improvements for quantized matrix multiplication on Intel Arc GPUs. The implementation leverages SYCL's joint_matrix extension to avoid dequantization overhead and utilize the 2x higher INT8 throughput of XMX hardware.

The prototype implementation provides a foundation for further optimization and integration into llama.cpp's SYCL backend.