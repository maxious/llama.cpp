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

## KV-Split Flash Attention (Flash Decoding) - Stable

### Status

The KV-split (Flash Decoding) path is now the default implementation for MKL Flash Attention in `ggml/src/ggml-sycl/fattn.cpp`.

- **Function**: `ggml_sycl_op_flash_attn_mkl<DQK, DV>()`
- **Algorithm**: Splits KV dimension across multiple chunks, computes partial attention per chunk, then merges using online softmax reduction

### Current Limitations

The current implementation processes KV splits **sequentially** on the host (using a for-loop with stream->wait() between splits). This means:
- No parallelism benefit yet - all splits run serially
- Adds overhead from extra memory allocations and synchronization

### Future Optimization

To get actual parallelism benefits, the implementation needs:
1. Launch all split kernels in parallel (different workgroups)
2. Use a single reduction kernel after all splits complete
3. Avoid host-side loops - everything should be GPU-side

This would require significant restructuring to:
- Create a kernel that takes (head, kv_split) as workgroup indices
- Implement a separate reduction kernel
- Manage dependencies properly with SYCL event DAG

### When It Would Help

The optimization is beneficial when:
- N (query length) is small (decode phase, N=1-4)
- N_kv (context length) is large (2K+ tokens)
- n_heads is small/moderate (device is underutilized)

## Flash Attention: XMX vs MKL

### Summary

**XMX (cooperative matrix) is now recommended for high-performance models like GLM-4.7 on Arc B60.**
MKL remains a stable fallback but XMX with Direct Loading offers significant performance benefits for memory-bandwidth bound models.

### Arc B60 Benchmark Results (GLM-4.7 23B Q4_K_M)

| Metric | MKL (default) | XMX (Direct Loading) | Improvement |
|--------|---------------|----------------------|-------------|
| Token gen | ~4 tok/s | ~4.5 tok/s | +12.5% |
| Memory Overhead | Low | Optimized (~85KB SLM) | -60% SLM Usage |

### Build Options

```bash
# Default build (MKL, no XMX) - Stable fallback
./build-sycl.sh --clean

# Build with XE2/XMX support - RECOMMENDED for Performance
./build-sycl.sh --clean --xe2
```

### Testing XMX

```bash
# Build with XMX support
./build-sycl.sh --xe2

# Enable Direct Loading (Critical for performance)
export GGML_SYCL_FLASH_ATTN_DIRECT=1
export GGML_SYCL_FLASH_ATTN_FORCE_XMX=1

./build-sycl/bin/llama-completion ...
```

### Recommendations for Arc B60

1. **Enable XMX** (`./build-sycl.sh --xe2`) for best performance on Battlemage.
2. **Enable Direct Loading** (`GGML_SYCL_FLASH_ATTN_DIRECT=1`) to reduce memory bandwidth usage.
3. The system now automatically handles block size optimization for GLM-4.7 to prevent resource exhaustion.

