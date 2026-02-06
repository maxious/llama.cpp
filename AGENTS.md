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
  - Fixed by adding MMQ_MIN_NROWS=128 guard in matmul.cpp to fall back to oneMKL when:
    - `src0->ne[1] < 128` (weight matrix rows)
    - `src1->ne[1] < 128` (output rows / batch size)
  - This is especially important for MUL_MAT_ID where per-expert batches can be small.
  - Test with: `./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q4_K"`
  - Test MUL_MAT_ID_FUSION: `./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID_FUSION -p "type_a=q4_K"`
- SYCL MMQ XE2 tile size fix (Feb 2026):
  - When SYCL_USE_XMX is defined (--xe2 build), MMQ kernel tile sizes were incorrectly set to tiny values (4x32).
  - Fixed by adding XE2-specific defines in mmq_internal.hpp that use 64x128 tiles like RDNA2.
  - Without this fix, caller allocated 128x64 shared memory but kernel expected 32x4, causing GPU page faults.
- SYCL MMVQ quantization fix (Feb 2026):
  - Fixed swapped arguments in `quantize_row_q8_1_sycl` call at line ~200 of matmul.cpp.
  - Bug: `kx=nrows1, ky=ne10` was wrong; correct is `kx=ne10, ky=nrows1`.
  - Symptom: MUL_MAT with quantized types (q4_K, q8_0, etc.) produced `ERR = inf` for n=2-8, m=16.
  - Root cause: With kx=2 < QK8_1=32, `num_quant_blocks = ky * (kx/32) = 0`, so quantization kernel didn't run.
   - Test: `GGML_SYCL_DISABLE_OPT=1 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q4_K"`
- SYCL MXFP4 Graph-Compatible Tiled GEMM (Feb 2026):
  - Added `gemm_mxfp4_f32_tiled.hpp` for SYCL graph-compatible MXFP4 matrix multiplication.
  - MXFP4 uses E2M1 format (4-bit) with per-block E8M0 scale (32 elements/block).
  - Key fix: BK tile size MUST equal QK_MXFP4 (32) to align with MXFP4 block boundaries.
  - Original bug: Argument order was swapped (src0/src1 inverted), causing completely wrong results.
  - Dequantization: `value = e8m0_scale * kvalues_mxfp4[q4] * 0.5` (kvalues_mxfp4 is doubled for DP4A).
  - Layout: `qs[j]` contains elements j (low nibble) and j+16 (high nibble).
  - Test: `GGML_SYCL_DISABLE_GRAPH=0 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=mxfp4"`
- SYCL MXFP4 MUL_MAT_ID Support (Feb 2026):
  - Added `launch_gemm_tiled_indirect_mxfp4` for MoE expert dispatch with MXFP4 weights.
  - Enables SYCL graphs for models like Qwen3-Coder-30B-A3B with MXFP4 quantized MoE experts.
  - Same block alignment constraint (BK=32=QK_MXFP4) and dequantization logic as MUL_MAT.
  - Supported weight types in MUL_MAT_ID: F32, F16, BF16, MXFP4.
  - Test: `GGML_SYCL_DISABLE_GRAPH=0 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID -p "type_a=mxfp4"`

## SYCL Runtime Architecture Detection

The SYCL backend now automatically detects the GPU vendor and architecture at runtime to select optimal MMQ (matrix multiplication with quantization) tile sizes, eliminating the need for compile-time flags like `--xe2`.

### Supported Architectures

| Architecture | Vendor | Vendor ID | MMQ Tile Config (x, y, warps) | Detection Criteria |
|--------------|--------|-----------|------------------------------|--------------------|
| Intel Xe2 (Battlemage) | Intel | 0x8086 | (64, 128, 8) | `major >= 13` |
| Intel Xe (Alchemist) / Xe-LPG (Meteor Lake) | Intel | 0x8086 | (64, 128, 4) | `major == 12` |
| Intel Gen9-Gen11 (integrated) | Intel | 0x8086 | (64, 128, 4) | `major >= 9 && major < 12` |
| AMD RDNA3 | AMD | 0x1002 | (64, 128, 8) | `major >= 11` |
| AMD RDNA2 | AMD | 0x1002 | (64, 128, 8) | `major == 10 && minor >= 30` |
| AMD RDNA1 | AMD | 0x1002 | (64, 64, 8) | `major == 10 && minor < 30` |
| NVIDIA Ampere | NVIDIA | 0x10de | (64, 128, 4) | `major >= 8` |
| NVIDIA Turing | NVIDIA | 0x10de | (64, 128, 4) | `major == 7 && minor >= 5` |

### Implementation Details

- **Detection Code**: `ggml/src/ggml-sycl/ggml-sycl.cpp` → `ggml_sycl_init()`
- **Arch Enum**: `ggml/src/ggml-sycl/common.hpp` → `enum sycl_arch_type`
- **Dispatch**: All MMQ kernel files (`mmq_q*.cpp`) use `switch(dev_info.arch)` to instantiate template specializations with compile-time tile constants.
- **Logging**: Set `GGML_SYCL_DEBUG=1` to see detection output: `Device X: vendor=0xXXXX, arch=Y, mmq={a,b,c}`

### Notes

- Unknown or older GPUs fall back to AMPERE-like config `(64, 128, 4)`.
- The tile size constants are defined in `mmq_internal.hpp` (e.g., `MMQ_X_Q4_0_XE2`, `MMQ_X_Q4_0_AMPERE`, `MMQ_X_Q4_0_RDNA2`, etc.).
- The design removes the compile-time `SYCL_USE_XMX` flag in favor of runtime detection.
- **User Override**: Set `GGML_SYCL_FORCE_ARCH` environment variable to force a specific architecture (e.g., `INTEL_XE2`, `AMD_RDNA3`, `NVIDIA_TURING`) to override automatic detection. Useful for testing or unsupported hardware.

## Multi-Device SYCL Graphs (Experimental)

A proof-of-concept implementation for per-device SYCL graphs in multi-GPU setups.

### How It Works

1. **Detection**: When `device_count > 1` and split buffers are detected, the system uses `graph_compat_t::MULTI_DEVICE` mode
2. **Node Partitioning**: cgraph nodes are partitioned by device:
   - Nodes using split buffers → assigned to ALL devices (each handles its row range)
   - Other nodes → assigned to the context's primary device
3. **Per-Device Recording**: Each device gets its own `command_graph` recorded on its queue
4. **Sequential Execution**: Graphs execute sequentially with `ext_oneapi_submit_barrier()` for inter-device sync

### Testing Multi-Device Graphs

```bash
# Build with graph support (enabled by default)
./build-sycl.sh

# Run with multiple devices visible
GGML_SYCL_DEBUG=1 ./build-sycl/bin/llama-completion \
  --model model.gguf \
  --split-mode layer --tensor-split 0.5,0.5 \
  --prompt "Hello" -n 10

# Look for "[SYCL-MULTI-GRAPH]" log messages
```

### Testing XMX GEMM with SYCL Graphs

```bash
# Build standalone XMX flash attention + graph test
icpx -fsycl -O3 -DSYCL_EXT_ONEAPI_MATRIX -DSYCL_EXT_ONEAPI_GRAPH -I. \
  tests/test-fattn-xmx-graph.cpp -o test-fattn-xmx-graph

# Run the test
./test-fattn-xmx-graph

# Expected output shows:
# - All tests PASSED with NMSE < 1e-7
# - Graph speedup of ~5-11% for small workloads
```

### Current Limitations

- **Sequential execution**: Graphs run one device at a time (no overlap yet)
- **No dependency analysis**: All split-buffer nodes are assigned to all devices
- **Experimental**: May have edge cases with certain model architectures

### Future Optimizations

1. Analyze node dependencies to overlap independent graphs
2. Smarter partitioning based on tensor→device mapping
3. Persistent graph caching across iterations

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

## Graph-Compatible GEMM

### Background

oneMKL GEMM operations (`oneapi::mkl::blas::gemm`) are incompatible with SYCL command graphs because they create internal SYCL events that cannot be captured during graph recording. This causes exceptions when trying to use SYCL graphs for MUL_MAT operations.

### Solution: Tiled Custom GEMM

A graph-compatible tiled GEMM is implemented in `ggml/src/ggml-sycl/gemm_tiled.hpp`:
- Uses shared memory tiling (64x64 output tiles, 32-element K tiles)
- Each thread computes 4x4 output elements
- 16x16 workgroup (256 threads)
- No oneMKL dependencies - fully graph-recordable

### Benchmark Results (Arc Pro B60)

| Size | Simple (naive) | Tiled | oneMKL | Tiled Speedup |
|------|----------------|-------|--------|---------------|
| 512³ | 416 GFLOPS | 2720 GFLOPS | 9064 GFLOPS | 6.5x vs naive |
| 1024³ | 445 GFLOPS | 2910 GFLOPS | 11365 GFLOPS | 6.5x vs naive |
| 2048³ | 458 GFLOPS | 3480 GFLOPS | 11986 GFLOPS | 7.6x vs naive |

**Performance vs oneMKL**: ~30% of oneMKL performance. This is a tradeoff for graph compatibility.

### Why oneMKL is 3x Faster

oneMKL achieves ~12000 GFLOPS vs our ~3500 GFLOPS because it uses:

1. **XMX Hardware Matrix Units** - Intel XMX does 8x16x16 matrix ops per instruction using `joint_matrix`
2. **Vectorized loads** - `float4`/`float8` instead of scalar loads (4-8x memory bandwidth)
3. **Double buffering** - Prefetch next tile while computing current one
4. **Optimal shared memory layout** - Avoids bank conflicts with padding
5. **bf16 intermediate precision** - 2x smaller data movement

### Future Optimization: XMX GEMM

An XMX-accelerated GEMM skeleton is in `ggml/src/ggml-sycl/gemm_xmx.hpp`. To achieve oneMKL-level performance:

1. Use `joint_matrix<sub_group, bfloat16, use::a/b, TM, TK/TN>` for inputs
2. Use `joint_matrix<sub_group, float, use::accumulator, TM, TN>` for output
3. Convert float->bf16 during shared memory loads
4. Use `sycl::address_space_cast` for `joint_matrix_load/store`
5. Requires `SYCL_EXT_COOPERATIVE_MATRICES` define (set in CMakeLists.txt)

### Usage

```cpp
#include "gemm_tiled.hpp"

// Single GEMM: C = alpha * A * B^T + beta * C
launch_gemm_tiled<true>(stream, A, B, C, M, N, K, alpha, beta, lda, ldb, ldc);

// Batched GEMM for flash attention
launch_gemm_tiled_batched<true>(stream, A, B, C, M, N, K, alpha, beta, batch, lda, ldb, ldc, stride_A, stride_B, stride_C);
```

### Testing the GEMM benchmark

```bash
source /opt/intel/oneapi/setvars.sh intel64
icpx -fsycl -O3 -DGGML_SYCL_USE_INTEL_ONEMKL tests/test-gemm-sycl.cpp -o test-gemm-sycl \
  -lmkl_sycl -lmkl_intel_lp64 -lmkl_core -lmkl_sequential
./test-gemm-sycl
```

