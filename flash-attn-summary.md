# Flash Attention Benchmark Summary

**Hardware**: Intel Arc Pro B60 (Battlemage BMG-G21)  
**Date**: 2025-01-24  
**Backend**: SYCL with XMX (16x16x16 PVC tiles)

## Key Finding: Flash Attention is SLOWER for Llama-3.2-1B

### Llama-3.2-1B (Q8_0) Results

#### Prompt Processing (pp128) - Flash Attention Impact

| Batch Size | FA ON (t/s) | FA OFF (t/s) | Speedup | Notes |
|------------|-------------|--------------|---------|-------|
| 1 | 43.56 | 55.00 | **0.79x** | 21% slower with FA |
| 8 | 75.07 | 78.18 | **0.96x** | 4% slower with FA |
| 16 | 660.57 | 809.92 | **0.82x** | 18% slower with FA |
| 32 | 1300.92 | 1568.34 | **0.83x** | 17% slower with FA |
| 64 | 2220.78 | 2926.69 | **0.76x** | 24% slower with FA |
| 128 | 3672.12 | 4949.51 | **0.74x** | 26% slower with FA |

#### Token Generation (tg32) - Flash Attention Impact

| Batch Size | FA ON (t/s) | FA OFF (t/s) | Speedup | Notes |
|------------|-------------|--------------|---------|-------|
| 1 | 43.67 | 54.91 | **0.80x** | 20% slower with FA |
| 8 | 43.64 | 54.89 | **0.80x** | 20% slower with FA |
| 16 | 43.59 | 55.04 | **0.79x** | 21% slower with FA |
| 32 | 43.48 | 54.97 | **0.79x** | 21% slower with FA |
| 64 | 43.65 | 55.00 | **0.79x** | 21% slower with FA |
| 128 | 43.54 | 55.07 | **0.79x** | 21% slower with FA |

### Analysis

**Why is Flash Attention Slower?**

1. **Wrong Tile Size**: Using 16x16x16 (PVC) tiles instead of 8x8x16 (DG2/Arc B60) tiles
   - The code detects `tile_kind=16x16x16 (PVC)` but B60 should use 8x8x16
   - This mismatch likely causes inefficient XMX utilization

2. **Small Model Overhead**: Llama-3.2-1B is only 1.2B parameters
   - XMX kernel launch overhead outweighs benefits for small models
   - Memory bandwidth bound rather than compute bound

3. **Q8_0 Quantization**: 8-bit quantization reduces compute intensity
   - Flash attention benefits are more pronounced with FP16/BF16
   - Quantized models may not benefit as much from XMX acceleration

### Recommendations

1. **Fix Tile Size Detection**: Ensure B60 uses 8x8x16 tiles, not 16x16x16
2. **Test Larger Models**: Flash attention may benefit larger models (>7B)
3. **Test FP16 Models**: Benefits may be more pronounced with FP16/BF16
4. **Profile Memory vs Compute**: Determine if bottleneck is memory bandwidth

### Next Steps

- Test with GLM-4.7-Flash (larger model, different architecture)
- Test with FP16 quantized models
- Investigate tile size detection issue
- Profile to identify actual bottleneck
