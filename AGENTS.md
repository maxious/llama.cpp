# Project Notes

- For SYCL builds, use `./build-sycl.sh` which configures `build-sycl/` and builds via CMake (`cmake --build`), not Ninja.
- The SYCL build directory `build-sycl/` does not contain `build.ninja` in this setup.
- Multi-device SYCL testing tips:
  - Build with `./build-sycl.sh`.
  - Use `--split-mode layer` or `--split-mode row` plus `--tensor-split` to exercise multi-device scheduling.
  - Enable debug logging with `GGML_SYCL_DEBUG=1` to trace split rows and per-device work; `LLAMA_LOG_LEVEL=debug` prints graph/backend pinning.
  - Flash attention auto-check can disable FA if backend assignment mismatches; look for `Flash Attention was auto` log line.
  - P2P copies are opt-in via `GGML_SYCL_ENABLE_P2P=1` (may cause device-lost on some systems).
- SYCL MMQ kernel known issues:
  - MMQ kernels with `need_check=true` (when nrows < mmq_y tile size) can have shared memory write collisions.
  - Fixed by adding MMQ_MIN_NROWS=128 guard in ggml-sycl.cpp to fall back to oneMKL when nrows is too small.
  - Test with: `./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q4_K"`

## SYCL Performance Profiling

### Available Tools (in ~/pti-gpu/tools/)

| Tool | Purpose | Overhead |
|------|---------|----------|
| `onetrace` | Kernel timing, device timeline, API tracing | Low |
| `unitrace` | Unified tracing with Chrome timeline export | Medium |
| `oneprof` | Hardware metrics (requires metrics-discovery) | High |
| `gpuinfo` | Device information | None |
| `sysmon` | System monitoring | Low |

### Quick Profiling with onetrace (Recommended)

```bash
source /opt/intel/oneapi/setvars.sh intel64

# Device timing summary (most useful)
~/pti-gpu/tools/onetrace/build/onetrace --device-timing --verbose --output profile.log \
  ./build-sycl/bin/llama-completion --model model.gguf --prompt "Hi" -n 10 --no-warmup

# View results
cat profile.log
```

### Profiling with unitrace (Chrome Timeline)

```bash
source /opt/intel/oneapi/setvars.sh intel64

# Generate Chrome trace
~/pti-gpu/tools/unitrace/build/unitrace \
  --chrome-sycl-logging --chrome-kernel-logging --chrome-device-logging \
  -o trace.json \
  ./build-sycl/bin/llama-completion --model model.gguf --prompt "Hi" -n 10

# View in browser: open https://ui.perfetto.dev and load trace.json
```

### Profiling with oneprof (Hardware Metrics)

Requires metrics-discovery library:
```bash
source /opt/intel/oneapi/setvars.sh intel64
export LD_LIBRARY_PATH=~/pti-gpu/metrics-discovery/install/lib:$LD_LIBRARY_PATH

~/pti-gpu/tools/oneprof/build/oneprof --kernel-metrics --output metrics.log \
  ./build-sycl/bin/llama-completion --model model.gguf --prompt "Hi" -n 10
```

### Debug Logging

```bash
# SYCL backend debug (verbose)
GGML_SYCL_DEBUG=1 ./build-sycl/bin/llama-completion ...

# Flash Attention debug
GGML_SYCL_FLASH_ATTN_DEBUG=1 ./build-sycl/bin/llama-completion ...

# General llama.cpp debug
LLAMA_LOG_LEVEL=debug ./build-sycl/bin/llama-completion ...
```

### Key Metrics to Watch

From onetrace output:
- `zeCommandListAppendMemoryCopy(M2D)` - Host→Device memory copies (should be <30% of device time)
- `zeCommandListAppendMemoryCopy(D2D)` - Device→Device copies (cross-GPU overhead)
- `mul_mat_vec_q*` - Matrix-vector multiply kernels (core compute)
- `flash_attn_mkl` / `flash_attn_xmx` - Flash attention kernels
- `reorder_qw_*` - Weight reordering (should only happen once per tensor)
- `quantize_row_q8_1` - On-the-fly quantization

### Performance Bottleneck Analysis

1. **Memory-bound** (>50% on memory copies):
   - Model too large for GPU memory
   - Cross-GPU communication overhead
   - Consider smaller model or single GPU

2. **Compute-bound** (>50% on kernels):
   - Good GPU utilization
   - Optimize hot kernels (mul_mat_vec, flash_attn)

3. **Reorder overhead** (>5% on reorder_qw):
   - Weight reordering should be cached after first run
   - Check `extra->optimized_feature.reorder` flag

### Example Profile Analysis (GLM-4.7 23B Q8_0)

```
Total Execution Time: 9.23 sec
Total Device Time: 1.34 sec (14.5%)

Top kernels:
- zeCommandListAppendMemoryCopy(M2D): 81.12% (BOTTLENECK - too much data movement)
- mul_mat_vec_q8_0_q8_1: 2.67% (compute)
- flash_attn_mkl: 0.11% (attention)
```

Diagnosis: Memory-bound. Model size (23GB) exceeds efficient memory bandwidth.
Solution: Use smaller model (8B) or Q4_K quantization.

## Flash Attention: XMX vs MKL

### Summary

**MKL is the correct default for Arc B60 GPUs.** XMX (cooperative matrix) kernels are optimized for PVC (Ponte Vecchio) data center GPUs, not consumer Arc GPUs.

### Arc B60 Benchmark Results (Llama-3 8B Q4_0, Single GPU)

| Metric | MKL (default) | XMX (JIT) | 
|--------|---------------|-----------|
| Load time | **1,248 ms** | 7,143 ms (5.7x slower) |
| Prompt eval | **8.81 tok/s** | 1.54 tok/s (5.7x slower) |
| Token gen | **10.34 tok/s** | 9.92 tok/s (4% slower) |

### Arc B60 Benchmark Results (Llama-3 8B Q4_0, Dual GPU)

| Metric | MKL (default) | XMX (JIT) | 
|--------|---------------|-----------|
| Load time | **2,082 ms** | 13,886 ms (6.7x slower) |
| Prompt eval | **8.17 tok/s** | 1.22 tok/s (6.7x slower) |
| Token gen | **17.72 tok/s** | 15.89 tok/s (10% slower) |

**Conclusion**: MKL is faster than XMX in all scenarios on Arc B60 GPUs.
- XMX JIT compilation adds ~6-12 seconds of overhead
- XMX prompt eval is ~6x slower than MKL
- XMX token generation is 4-10% slower than MKL

### AOT Compilation Status

AOT (Ahead-of-Time) compilation for Battlemage (`intel_gpu_bmg_g21`) **fails** with exit code 245 during linking. The cooperative matrix (XMX) kernels cannot be AOT-compiled for Battlemage in oneAPI 2025.3.

### Build Options

```bash
# Default build (MKL, no XMX) - RECOMMENDED
./build-sycl.sh --clean

# Build with XE2/XMX support (JIT only) - for testing
./build-sycl.sh --clean --xe2

# AOT build attempt (currently fails for XMX kernels)
./build-sycl.sh --clean --aot
```

### Testing XMX (for benchmarking only)

```bash
# Build with XMX support
./build-sycl.sh --xe2

# Force XMX for all batch sizes (bypasses N < 32 check)
GGML_SYCL_FLASH_ATTN_FORCE_XMX=1 ./build-sycl/bin/llama-completion ...

# Force MKL (default when XMX not compiled)
GGML_SYCL_FLASH_ATTN_MKL=1 ./build-sycl/bin/llama-completion ...
```

### Recommendations for Arc B60

1. **DO NOT enable XMX** (`-DGGML_SYCL_XE2=OFF` is default) - MKL is faster
2. Use oneMKL BLAS for flash attention (default behavior)
3. The `N < 32` small_batch threshold correctly routes to MKL for token generation

