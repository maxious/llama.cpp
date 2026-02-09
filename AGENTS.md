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
- SYCL XE2 Q8_0 MMQ Workaround (Feb 2026):
  - Q8_0 MMQ kernels on Intel XE2 GPUs (Arc Battlemage) cause compiler/driver crashes (segfault in libigc.so) for batch sizes >= 128.
  - This affects prompt processing in `llama-bench` with `-b 128`.
  - Fix: Added a specific check in `matmul.cpp` to disable MMQ for Q8_0 on XE2, forcing fallback to oneMKL (or XMX if compatible types) which is stable.
  - Test: `GGML_SYCL_DEBUG=1 ./build-sycl/bin/llama-bench -m models/koboldcpp/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf -p 128 -n 64 -b 128 -ub 128`

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

## SYCL Flash Attention Implementations

The SYCL backend currently maintains **four** distinct Flash Attention implementations. This complexity is necessary to cover the cross-product of **Workload Type** (Prompt vs. Generation), **Execution Mode** (Eager vs. Graph), and **Hardware Capabilities** (XMX vs. Generic).

### Dispatch Logic

| Workload | Batch Size | Execution Mode | Best Implementation | Why? |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt Processing** | `N >= 32` | Any | **XMX** (or Fused) | Max throughput via hardware acceleration. |
| **Token Generation** | `N < 32` | **Eager** (Chat) | **oneMKL** | Fastest possible low-latency math. |
| **Token Generation** | `N < 32` | **Graph** (Bench) | **Tiled** | **Only** fast option that doesn't crash graphs. |

### 1. XMX Flash Attention (`fattn_xmx`)
*   **Purpose:** **Maximum Throughput** on Intel Arc/Data Center GPUs.
*   **When used:** **Prompt Processing** (Large Batches) on supported hardware (Intel Arc, Data Center Max).
*   **Key Advantage:** Leverages specialized XMX matrix hardware instructions (`joint_matrix`), providing significantly higher FLOPS than generic kernels.
*   **Limitation:** Strict shape requirements (head sizes 64, 96, 128), doesn't support attention sinks yet.

### 2. Fused Flash Attention (`fattn_fused`)
*   **Purpose:** **Portable High Performance** for large batches.
*   **When used:** **Prompt Processing** (Large Batches) on:
    *   Non-Intel GPUs (AMD, Nvidia via SYCL).
    *   Intel GPUs when XMX shapes aren't matched.
*   **Key Advantage:** Single kernel efficiency, graph-compatible. Keeps data in registers/SLM.
*   **Limitation:** Poor performance at `N=1` (~3.7 t/s) due to low thread occupancy (most threads idle).

### 3. Tiled Flash Attention (`fattn_tiled`)
*   **Purpose:** **Graph-Compatible Low Latency**.
*   **When used:** **Token Generation** (Small Batches) inside **SYCL Graphs**.
    *   Essential for `llama-bench` and future graph-based runners.
*   **Key Advantage:** Decomposes operation into `GEMM` + `Softmax` + `GEMM` using `gemm_tiled` kernels. Fast enough (~15 t/s) and **compatible with `cudaGraph`-style recording** (unlike oneMKL).
*   **Implementation:** Supports F32 and F16 mixed precision.

### 4. oneMKL Flash Attention (`fattn_mkl`)
*   **Purpose:** **Absolute Lowest Latency** (Eager Mode).
*   **When used:** **Token Generation** (Small Batches) during **Interactive Inference** (Eager Execution).
*   **Key Advantage:** Marginally faster and more robust than "Tiled" for eager execution due to Intel's hand-tuned microkernels.
*   **Limitation:** **Crashes if recorded into a SYCL Graph** because oneMKL creates internal events that the graph API cannot capture.

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

### Debug Logging

```bash
# SYCL backend debug (verbose)
GGML_SYCL_DEBUG=1 ./build-sycl/bin/llama-completion ...

# Flash Attention debug
GGML_SYCL_FLASH_ATTN_DEBUG=1 ./build-sycl/bin/llama-completion ...

# General llama.cpp debug
LLAMA_LOG_LEVEL=debug ./build-sycl/bin/llama-completion ...
```

### ITT Annotations (VTune Semantic Markers)

The SYCL backend includes ITT (Instrumentation and Tracing Technology) annotations for detailed profiling in Intel VTune and similar tools. These annotations mark semantic regions like "which Flash Attention variant is running" directly in the timeline.

**Enable ITT annotations:**

```bash
# Runtime enable (no rebuild needed)
export INTEL_ENABLE_OFFLOAD_ANNOTATIONS=1

# Or build with explicit enable
cmake -DGGML_SYCL_ITT_ANNOTATIONS=1 ...
```

**Profile with VTune:**

```bash
source /opt/intel/oneapi/setvars.sh intel64

# Basic GPU hotspots with ITT regions
vtune -collect gpu-hotspots ./build-sycl/bin/llama-bench -m model.gguf -p 128 -n 64

# Or use the Advisor for roofline analysis
advisor --collect=roofline -- ./build-sycl/bin/llama-bench -m model.gguf
```

**What you'll see in VTune:**

| ITT Region | Description |
|------------|-------------|
| `fattn:xmx:h64` | XMX Flash Attention (head size 64) |
| `fattn:xmx:h128` | XMX Flash Attention (head size 128) |
| `fattn:fused` | Fused single-kernel Flash Attention |
| `fattn:tiled` | Tiled Flash Attention (graph-compatible) |
| `fattn:mkl` | oneMKL Flash Attention fallback |
| `mul_mat:mmq:quantized` | MMQ quantized matrix multiplication |
| `mul_mat:xmx:f32` | XMX GEMM (F32) |
| `mul_mat:mkl:f32` | oneMKL BLAS GEMM |
| `mul_mat_id:tiled:moe` | MoE MUL_MAT_ID with tiled kernels |
| `mul_mat_id:mmq:moe` | MoE MUL_MAT_ID with MMQ |
| `op:dmmv` | Dequantize + mul_mat_vec |
| `op:mmvq` | Vector quantized matmul |

**Implementation location:** `ggml/src/ggml-sycl/itt_annotations.hpp`

### SYCL/IGC Kernel Dump Debugging (Crash Isolation)

When debugging kernel compile crashes (e.g., IGC/Level Zero segfaults), use SYCL + IGC dump variables to capture device images and kernel names.

```bash
export SYCL_DUMP_IMAGES=1
export SYCL_CACHE_DIR=/tmp/sycl-dump
export SYCL_CACHE_TRACE=0x07
export SYCL_UR_TRACE=-1

export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=/tmp/igc-dumps
# Optional: print kernel names during compilation (debug only; invalid binaries)
# export IGC_CompileOneAtTime=1

# Example repro (adjust model/params):
./build-sycl/bin/llama-bench ...
```

Notes:
- IGC dumps include `*_cmd.txt`, `*.spv`, `*_codegen.ll`, `*_beforeUnification.ll`, `*_optimized.ll`, `*.asm`, and `*.zeinfo`.
- `SYCL_DUMP_IMAGES` + `IGC_ShaderDumpEnable` helps narrow which kernel crashes by inspecting the last dumped module.
