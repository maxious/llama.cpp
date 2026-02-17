## Optimization Goals (Feb 2026)

### 1. Optimize Fused Flash Attention (`fattn_fused`)
- **Status**: Graph-compatible (no crashes), but performance is low (~720 t/s) compared to non-FA path (~9200 t/s).
- **Issue**: Likely register pressure (spilling to memory) due to large accumulators (`float acc[128]`, `logits[32]`) or inefficient thread serialization.
- **Goal**: Reach parity with `fa=off` performance (~9000+ t/s) for single-device prompt processing.
- **Strategy**: Tune tile sizes (`BQ=16`, `BK=32`), reduce register usage, investigate vectorization.

### 2. Close Token Generation Gap
- **Status**: Branch (~43 t/s) trails Master SYCL (~54 t/s) and Master Vulkan (~67 t/s).
- **Issue**: Latency regression in small-batch workloads (`n=1`). Possible overhead in kernel launch, graph submission, or `gemv`/`dequantize` kernels.
- **Goal**: Reach ~65 t/s on Llama-3.2-1B to match Vulkan.
- **Strategy**: Profile with `onetrace` specifically for `tg128` workload. Check `gemv` implementation.

### 3. Verify Multi-Device Scaling
- **Status**: Scaling logic implemented but not showing benefits on 1B model (compute bound on single device).
- **Goal**: Confirm scaling efficiency on larger models (Llama-3-8B or 70B).
- **Strategy**: Run benchmarks on larger models. Ensure P2P is active and effective.

## Project Notes

- For SYCL builds, use `./build-sycl.sh` which configures `build-sycl/` and builds via CMake (`cmake --build`), not Ninja.
- The SYCL build directory `build-sycl/` does not contain `build.ninja` in this setup.
- Running benchmarks requires the patched IGC to avoid JIT compile hangs:
  ```bash
  source /opt/intel/oneapi/setvars.sh -i --force
  LD_LIBRARY_PATH=~/igc_workspace/build/IGC/Release:$LD_LIBRARY_PATH python3 bench.py --devices single
  ```
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
- SYCL MMQ IGC compiler crash fix (Feb 2026):
  - MMQ kernels caused IGC compiler crashes (segfault in AddRequiredMemoryFences.cpp) on Intel XE2 GPUs for batch sizes >= 128.
  - Root cause: `item_ct1.barrier()` (no fence space) didn't emit an SLM fence intrinsic that IGC's `IsSlmFence()` recognizes. The pass then tried to insert fences at loop exits, but `getUniqueExitBlocks()` returned empty for the optimized loop, causing an empty-vector dereference.
  - Fix: Changed barriers in `mul_mat_q` (mmq_internal.hpp) to `item_ct1.barrier(sycl::access::fence_space::local_space)`, which emits an explicit SLM fence that the IGC pass recognizes, preventing it from taking the buggy loop-exit path.
  - This eliminated the need for the per-type XE2 workaround in matmul.cpp (Q8_0, Q2_K-Q6_K are now all enabled on XE2).
  - Test: `./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q8_0"`
  - Test: `./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q4_K"`
- SYCL MUL_MAT_ID IGC crash workaround (Feb 2026):
  - MUL_MAT_ID with q8_0 crashed in AddRequiredMemoryFences.cpp despite the barrier(local_space) fix.
  - Root cause: All SYCL `.cpp` files compile into one library (`ggml-sycl`), so all device kernels share one SPIR-V fat binary. When IGC JIT-compiles kernels for MUL_MAT_ID, it processes a different subset/ordering than MUL_MAT, hitting the bug through a different IR path.
  - Fix: On XE2, `ggml_sycl_mul_mat_id()` always uses the tiled path (`ggml_sycl_mul_mat_id_tiled`) instead of the MMQ path. The tiled kernels don't trigger the IGC bug.
  - The tiled path supports all weight types: F32, F16, BF16, MXFP4, Q4_0, Q8_0, Q2_K-Q6_K.
  - This is an upstream IGC bug (empty `getUniqueExitBlocks()` dereference). The workaround can be removed once IGC fixes `AddRequiredMemoryFences.cpp`.
  - A patched IGC is at `~/igc_workspace/` on branch `fix-add-required-memory-fences-crash` (fork: https://github.com/maxious/intel-graphics-compiler).
  - To test with the patched IGC (bypasses the Xe2 workaround):
    ```bash
    # Build the patched IGC (uses icecc for distributed build):
    cd ~/igc_workspace/build
    CCACHE_PREFIX=icecc ICECC_SCHEDULER_HOST=192.168.1.192 make igc_dll -j96
    # Three GCC 15 compat patches are needed in the bundled LLVM 16 code (already applied locally):
    #   - clang/lib/Driver/ToolChains/Arch/X86.cpp: StringMapKeyIterator range-for
    #   - clang/lib/Sema/SemaExpr.cpp: [=, this] capture
    #   - IGC/Compiler/Optimizer/InstructionHoistingOptimization.cpp: const ref vector param

    # Run llama.cpp tests with the patched libigc.so:
    LD_LIBRARY_PATH=~/igc_workspace/build/IGC/Release:$LD_LIBRARY_PATH \
      ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID -p "type_a=q8_0"
    ```
  - Test: `./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID -p "type_a=q8_0"`

## SYCL Flash Attention hsk=40 Investigation (Feb 2026)

**Problem**: Page faults in SYCL Flash Attention for head size 40 (hsk=40) on Intel Arc B60.

**Root Cause**: All XMX tile configurations (32x32, 16x16) cause issues with hsk=40:
- XMX padded kernel: hangs with 64x64 tiles (padded from 40)
- Fused path: not implemented for hsk=40
- MKL/tiled paths: produce wrong results

**LLVM SYCL e2e Tests Reference** (`~/llvm/sycl/test-e2e/Matrix/`):
- Intel AMX: TM=16, TN=16, TK=16/32
- Intel PVC (B60): TM=8, TN=16, TK=16 (nsize=16)
- Intel DG2: TM=8, TN=8, TK=16 (nsize=8)
- Note: Tests don't cover head dimensions as small as 40

**Current Fix**: Disabled hsk=40 in `ggml_sycl_flash_attn_ext_supported()` - returns false so SYCL skips FA and falls back to CPU.

**Potential Future Fixes**:
1. Enable small-tile kernel (16x16) for hsk=40 instead of using 32x32
2. Add explicit handling for head dims that pad to 64
3. Add similar `row_split` logic from Vulkan PR #19625 to avoid cross-workgroup barriers

**Test**:
```bash
./build-sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT -p "hsk=40"
# Should show "not supported [SYCL0]"
```

## SYCL XMX Flash Attention Guardrails

- For MLA head sizes (e.g., 576/512), ensure the XMX allowlist includes the head size or use `GGML_SYCL_FLASH_ATTN_XMX_ALLOWLIST=all`.
- Small-tile kernels do not allocate `shV`. When estimating SLM usage, exclude the `shV` term or the kernel may be rejected despite fitting.
- Masking must clamp out-of-range KV columns (`kv_col >= N_kv`) to a large negative value before softmax, even when `mask == nullptr`.
- Ensure reorder kernels wait on the XMX kernel event (or use explicit event dependencies) before reading shared output buffers.
- When reusing pooled `l_d`/`m_d`, reinitialize them on every call (`l_d = 0`, `m_d = -1e20f`) to avoid stale online softmax state.
- Always include XMX coverage in `test-backend-ops` with `GGML_SYCL_FLASH_ATTN_FORCE_XMX=1` for new head sizes.

## Debugging SYCL SIGSEGV Crashes

When investigating SIGSEGV crashes in the SYCL backend, **always use gdb** rather than relying on stderr output alone. IGC compiler crashes happen inside `libigc.so` during JIT compilation and produce no stderr output — the process just dies with SIGSEGV. Under gdb, the backtrace reveals:
- Which IGC pass crashed (e.g., `AddRequiredMemoryFences.cpp:168`)
- Which kernel was being JIT-compiled (e.g., `launch_q8_0<64, 128, 8>`)
- The full SYCL runtime → Level Zero → IGC call chain

```bash
# Basic crash diagnosis
gdb -batch -ex "set print thread-events off" -ex run -ex bt \
  --args ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID -p "type_a=q8_0"

# With debug logging
GGML_SYCL_DEBUG=1 gdb -batch -ex "set print thread-events off" -ex run -ex bt \
  --args ./build-sycl/bin/test-backend-ops ...
```

Key frames to look for in the backtrace:
- `IGC::AddRequiredMemoryFences::runOnFunction` — SLM fence insertion bug
- `launch_q8_0<...>` / `launch_q4_K<...>` — identifies the crashing kernel template
- `ggml_sycl_op_mul_mat_q` vs `ggml_sycl_op_mul_mat_sycl` — identifies the dispatch path

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

### Notes from Recent Profiling

- `onetrace` worked reliably for `test-backend-ops` and produced a useful kernel timing summary.
- `unitrace` timed out on `test-backend-ops` (no output after ~5 minutes).
- `oneprof` reported `No metrics found`, so it could not generate reports on this system.
- VTune `gpu-hotspots` failed with "analysis type is not applicable" on this machine.

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

How To Use

Set GGML_SYCL_OP_STATS=1 to get counts per implementation/shape/type.
Set GGML_SYCL_OP_STATS_TIMING=1 to also include rough wall-clock timing (adds stream synchronization, so it can slow down execution).
The summary prints when the SYCL backend is destroyed (process exit or backend teardown).

What The Summary Includes

Implementation label: xmx, fused, tiled, mkl.
Parameters: dqk, dv, n, n_kv, heads, kv_heads, q/kv/out types, mask type, sinks, n_splits, graph vs eager, small batch.
Counters and (optionally) timing totals/avg/min/max.
