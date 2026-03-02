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

## XMX Peak Performance Notes

- Intel Xe-HPG (ACM-G10) peak XMX throughput per clock: FP16/BF16 65536 ops, INT8 131072 ops, INT4/INT2 262144 ops.
- 2.1 GHz example peak (XMX Matrix engine): FP16/BF16 137.6 TFLOPS, INT8 275.2 TOPS, INT4 550.4 TOPS.
- Vector (non-XMX) peak at 2.1 GHz for comparison: FP16 39.32 TFLOPS, FP32 19.66 TFLOPS.

## Intel Battlemage (Xe2) XVE Architecture Notes

- Reference: https://chipsandcheese.com/p/intels-battlemage-architecture
- Concurrency: XMX units operate alongside vector (FP) and scalar (INT/EM) units, allowing the XVE to execute different instruction types concurrently.
- Thread Management: XVEs manage multiple threads (up to eight), switching between them to maintain high execution unit utilization.
- Each Xe-core has 8 XVEs. A subgroup of 16 threads runs across 8 XVEs (2 threads per XVE).
- To fully utilize an Xe-core, multiple subgroups should run concurrently (up to 8 threads per XVE = 4 subgroups of 16).

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
  - `llama-bench` uses `;` or `/` as `--tensor-split` separators (e.g., `--tensor-split "0.5;0.5"`).
- SYCL cross-device synchronization rule (Mar 2026):
  - **Never use `ext_oneapi_submit_barrier({event})` across devices with different `sycl::context` objects.** Level Zero events are device-scoped; passing an event from device A's queue as a dependency to device B's queue causes indefinite hangs on discrete GPUs without P2P.
  - **Pattern**: Check `stream->get_context() == other_stream->get_context()` before choosing the sync method:
    - Same context → `stream->ext_oneapi_submit_barrier({event})` (efficient device-side wait).
    - Different context → `event.wait()` (host-side wait, always safe).
  - This applies to all cross-device event dependencies: `ggml_sycl_op_mul_mat` split-tensor sync, multi-device graph fan-in barriers, and any future cross-device coordination.
  - Same-device barriers (recording an event on queue A, waiting on it from queue A or another queue on the same device) are always safe.
  - Reference: `ggml/src/ggml-sycl/matmul.cpp` (lines ~300, ~484), `ggml/src/ggml-sycl/backend.cpp` (multi-device graph `chain_before_submit` and fan-in barrier).
- SYCL split-buffer (row-split) multi-GPU notes (Mar 2026):
  - `src0->data` is a dummy address (`0x1000`) for split tensors — always use `src0_extra->data_device[i]` for per-device pointers.
  - Tensor reordering (`opt_for_reorder`) must be disabled for split tensors to avoid segfault on the dummy address.
  - Cross-device memcpy must use host staging (`dev2dev_memcpy`) when P2P is unavailable; direct `stream->memcpy` between devices hangs.
  - `ggml_backend_sycl_device_supports_buft` must return `true` for `SYCL_Split` buffer types (checked via `ggml_backend_buft_is_sycl_split`).
  - Per-device streams in `split_buffer_context` must be initialized for all devices on first tensor, not appended per-tensor.
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

## SYCL MUL_MAT Path Logging and XMX Int8 (Feb 2026)

### Path Discovery
- Added logging in `ggml_sycl_mul_mat()` to show which implementation is selected:
  - `ggml_sycl: MUL_MAT DMMV ne=[...] type=...` - Dequantize + mul_mat_vec
  - `ggml_sycl: MUL_MAT MMVQ ne=[...] type=...` - Vector quantization mul_mat
  - `ggml_sycl: MUL_MAT MMQ ne=[...] type=...` - Matrix multiplication with quantization
  - `ggml_sycl: MUL_MAT XMX ne=[...] type=...` - XMX hardware accelerated GEMM (F32/F16)
  - `ggml_sycl: MUL_MAT XMX_INT8 ne=[...] type=...` - XMX int8 GEMM (quantized types)
  - `ggml_sycl: MUL_MAT MKL ne=[...] type=...` - oneMKL fallback
- Also logs MUL_MAT_ID paths: `TILED` (MoE tiled) or `MMQ` (MoE MMQ)

### Key Finding: MMQ Row Guard
- MMQ path is guarded by `MMQ_MIN_NROWS=128` in matmul.cpp
- When batch size < 128, MMQ is disabled and falls through to XMX path
- XMX path only supports F32/F16 - for quantized types it falls back to MKL
- This explains Perfetto showing `gemm_kernel` (MKL) for q8_0 at small batch sizes

### XMX Int8 Support (Opt-in)
- XMX int8 kernels exist in `mmq_xmx_int8.cpp` but were not wired up
- Now connected via `ggml_sycl_op_mul_mat_xmx()` for q8_0, q4_0, q4_1, q5_0, q5_1, q8_1
- K-quant support: Q4_K, Q5_K, Q6_K use multi-subgroup col_major B pattern; Q2_K, Q3_K added with split-MAD approach (16-element sub-blocks, two MADs per TK=32 tile)
- Enable with: `GGML_SYCL_XMX_INT8=1 ./build-sycl/bin/llama-bench ...`
- Without the flag, falls back to MKL (original behavior)
- **Supported types**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- Test Q2_K: `GGML_SYCL_XMX_INT8=1 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q2_K"`
- Test Q3_K: `GGML_SYCL_XMX_INT8=1 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q3_K"`

#### XMX Int8 Performance (Feb 2026)

All 37/37 tests pass with optimized defaults. Performance varies by problem size:

**Canonical benchmark (K=2048, M=2048, N=128):**

| Variant | Config | GFLOPS | vs v1 | vs MKL |
|---------|--------|--------|-------|--------|
| v1 (legacy) | N_SG=1 | 460 | 1.0x | 0.05x |
| v2 | N_SG=8, TILES_N=0 | 1026 | 2.2x | 0.11x |
| v3 | N_SG=8, TILES_N=2 | 1928 | 4.2x | 0.20x |
| v4 (prefetch) | N_SG=8, TILES_N=2, PREFETCH=1 | 1928 | 4.2x | 0.20x |
| v5 (col_major B) | N_SG=8, TILES_N=2, COLMAJOR=1 | 2096 | 4.6x | 0.22x |
| MKL baseline | - | 9476 | 20.6x | 1.0x |

**Large problem (K=4096, M=4096, N=256):**

| Variant | Config | GFLOPS | vs MKL |
|---------|--------|--------|--------|
| v3 | N_SG=8, TILES_N=2 | 2323 | 0.14x |
| v3 | N_SG=8, TILES_N=4 | 2631 | 0.16x |
| v5 (col_major B) | N_SG=8, TILES_N=2, COLMAJOR=1 | 2755 | 0.16x |
| v5 (col_major B) | N_SG=8, TILES_N=4, COLMAJOR=1 | 3240 | 0.19x |
| MKL baseline | - | 16834 | 1.0x |

**Key Optimizations:**
- **v2**: Multiple subgroups per workgroup (N_SG=8) sharing B tile, reducing barriers from 3→2 workgroup barriers per K tile
- **v3**: N-dimension expansion (TILES_N=2) — load A once from SLM, reuse across multiple B tiles, getting TILES_N× A data reuse
- **v4**: v3 + L1 prefetch hints for next K tile's blocks using `sycl::ext::oneapi::experimental::prefetch`. Negligible benefit — hardware prefetcher already handles sequential block access effectively.
- **v5**: col_major B layout in SLM — stores each block_q8_1's qs[32] contiguously as one column of B, then uses `joint_matrix_load` with `layout::col_major`. Eliminates the per-byte scatter-transpose from [block][qs] → [K][N], enabling vectorized 4-byte SLM writes instead of 4 scattered byte writes. ~15-25% improvement over v3.

**Optimal NSG/TILES_N by problem shape:**
- Small N (≤128): NSG=8, TILES_N=2 is best
- Large N (≥256): NSG=8, TILES_N=4 is best (more N-reuse)
- Small M (≤512): NSG=4 can beat NSG=8 (fewer idle subgroups at tile boundaries)

**Tuning Knobs (environment variables):**
- `GGML_SYCL_XMX_INT8_NSG` — subgroups per workgroup (default: 8)
- `GGML_SYCL_XMX_INT8_TILES_N` — N tiles per subgroup (default: 2)

**Remaining Gap Analysis:**
The ~5x gap vs MKL is due to per-byte scatter-gather from quantized block AoS structures through SLM. Even with col_major B (v5), A tiles still require scatter loads. MKL operates on dequantized contiguous F32 data via optimized microkernels. The strategic value of XMX int8 is enabling SYCL graphs for quantized models (which MKL doesn't support).

**Potential further optimizations:**
- col_major A: apply same trick to A tiles (store block_q8_0.qs[] contiguously, use col_major joint_matrix_load for matA)
- Accumulator-direct scales: use joint_matrix_apply to apply scales directly in accumulator registers, avoiding SLM C round-trip
- Adaptive tile selection: auto-select NSG/TILES_N based on M/N dimensions at runtime
- Wider vector stores: use 8-byte or 16-byte SLM writes for B data (sycl::vec<int8_t, 8> or vec<int8_t, 16>)

**Test:**
```bash
GGML_SYCL_XMX_INT8=1 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q8_0"
```

### MUL_MAT_ID XMX Env Flag
- Set `GGML_SYCL_MUL_MAT_ID_XMX=1` to bypass Xe2 tiled workaround
- Allows testing MMQ path for MoE (requires patched IGC to avoid AddRequiredMemoryFences crash)
- Default behavior: Xe2 always uses TILED path to avoid IGC crash

### Testing Path Selection
```bash
# Run benchmark - logs show path per operation
./build-sycl/bin/llama-bench -m model.gguf -n 32 -p 32 --split-mode layer --tensor-split 1,1 2>&1 | grep "MUL_MAT"

# Single device, no split (simpler paths)
./build-sycl/bin/llama-bench -m model.gguf -n 16 -p 16 --split-mode none
```

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

## SYCL Graph Optimization Plan (Feb 2026)

### Current State
- SYCL graphs are implemented with queue recording (`begin_recording`/`end_recording`/`finalize`/`ext_oneapi_graph`).
- Graph caching uses FNV-1a topology hashing (op, type, shape) with up to 8 cached executable graphs.
- Whole-graph `update()` is used when `ext_oneapi_graph` aspect is supported; falls back to re-finalize.
- `force_graph_compatible` flag forces graph-safe kernel paths (tiled instead of oneMKL, direct-ids instead of pack/unpack for MoE).
- Graphs are **disabled by default** (`GGML_SYCL_DISABLE_GRAPH=1`).
- Many bail-out heuristics in `check_graph_compatibility()`: n_nodes > 500, Q5_0/Q8_0 types, SET_ROWS with large VIEWs, unsupported MUL_MAT type combos.

### Key Flaws Identified

1. ~~**Re-records every frame even on cache hit**~~ — **FIXED** (commit 327b07c8b). Three-tier cache now skips recording entirely when USM pointers are unchanged (pure replay).
2. ~~**Missing `no_immediate_command_list`**~~ — **FIXED** (commit 327b07c8b). Dedicated graph execution queue with `no_immediate_command_list` + `in_order` properties. Lazily created via `graph_exec_stream()`.
3. ~~**All-or-nothing graph compatibility**~~ — **FIXED**. Was: If any single op is graph-incompatible (oneMKL, oneDNN), the entire graph is disabled. All nodes fall back to eager. Now: uses segmented graph execution.
4. ~~**`force_graph_compatible` forces slow paths**~~ — **FIXED**. Was: MUL_MAT always uses tiled kernels (slow) instead of MMQ (fast) during graph recording. Now: graph-safe dispatch allows DMMV/MMVQ/MMQ/XMX during recording, only falling back to tiled as last resort.

### Completed Fixes

#### Fix 1: Three-tier graph cache with pure replay ✅ (commit 327b07c8b)
- Added `compute_cgraph_pointer_hash()` — FNV-1a hash of all USM `tensor->data` pointers.
- Added `graph_pointer_hashes` map in `common.hpp` alongside `graph_cache`.
- Three-tier cache lookup in single-device path:
  - **Tier 1 (Pure replay)**: topology + pointers match → just `ext_oneapi_graph()`, no recording, no queue wait. ~10-15 µs per call.
  - **Tier 2 (Re-record + update)**: topology match, pointers differ → `begin_recording` + `compute_impl` + `end_recording` + `update()`.
  - **Tier 3 (Full record + finalize)**: cache miss → full pipeline with `finalize(updatable{})`.
- **Verified**: Llama-3.2-1B Q4_K tg16 — 2 cache misses on startup, then 100% pure replay hits. test-backend-ops 35/35 MUL_MAT q4_K tests pass.

#### Fix 2: Dedicated graph execution queue with `no_immediate_command_list` ✅ (commit 327b07c8b)
- Added `graph_exec_stream()` method to `ggml_backend_sycl_context` that lazily creates a `sycl::queue` with `in_order` + `no_immediate_command_list` properties.
- All 3 single-device `ext_oneapi_graph()` calls use this queue. Multi-device path unchanged.
- Per Intel spec: `no_immediate_command_list` uses standard command queues (batched submission) instead of immediate command lists, reducing per-submission overhead.

#### Fix 3: Segmented graph execution for graph-incompatible ops ✅
- **Problem**: Previously, if ANY node was graph-incompatible (Q5_0/Q8_0 MMQ, oneMKL GEMM), check_graph_compatibility() returned DISABLED and the entire graph fell back to eager execution.
- **Solution**: Instead of oneDNN's pause/resume pattern (which doesn't solve the replay problem), implemented **segmented graph execution** that partitions the cgraph into graph segments and immediate-mode nodes.
- **Design choice**: oneDNN's pause/resume records multiple segments into one modifiable graph, but a single executable graph cannot "remember" to do eager work between segments on replay. Segmentation creates separate executable graphs per segment, enabling correct interleaved replay.
- **Key components**:
  - `node_needs_immediate_mode()` — identifies per-node incompatibilities: Q5_0/Q8_0 MMQ types (GPU faults), unsupported type combos that fall to oneMKL GEMM (creates internal events).
  - `graph_exec_step` / `build_graph_exec_plan()` — partitions cgraph nodes into consecutive graph segments and immediate-mode groups.
  - Two execution paths in `ggml_backend_sycl_graph_compute()`:
    - **Monolithic path** (no immediate nodes): existing three-tier cache, unchanged.
    - **Segmented path** (has immediate nodes): per-segment executable graphs cached in `segmented_graph_cache`, interleaved with eager execution of immediate nodes.
  - Segmented cache in `common.hpp`: `segment_cache_entry` stores vector of executable graphs + pointer hash.
  - Three-tier cache per segment: pure replay (topology+pointers match), re-record (pointers changed), cache miss (full record+finalize).
  - Queue synchronization: `stream()->wait()` between graph segments and immediate nodes ensures correct data dependencies.
- **Results**: Q8_0 and Q5_0 MUL_MAT now work with graphs enabled (37/37 tests pass for Q8_0). Previously these types completely disabled graphs.
- **Verified**: test-backend-ops Q8_0 37/37, Q5_0 all pass, Q4_K 34/35 (1 pre-existing tiled precision edge case, improved from 32/35 after Fix 4).
- **Test**:
  ```bash
  GGML_SYCL_DISABLE_GRAPH=0 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q8_0"
  GGML_SYCL_DISABLE_GRAPH=0 GGML_SYCL_DEBUG=1 ./build-sycl/bin/llama-bench -m model.gguf -n 16 -p 0
  # Look for "[SYCL-GRAPH-SEG]" log messages indicating segmented path
  ```

### Remaining Fixes

#### Fix 4: Graph-safe dispatch allows fast kernel paths ✅
- **Problem**: `force_graph_compatible` in matmul.cpp forced all MUL_MAT to use tiled kernels during graph recording, even when fast paths (DMMV, MMVQ, MMQ) are graph-compatible.
- **Original plan**: Use `dynamic_command_group` to register both tiled and MMQ as alternatives. **Abandoned** because `dynamic_command_group` requires explicit graph API (`graph.add()`), not queue recording (`begin_recording`/`end_recording`). The SYCL spec explicitly throws `invalid` if `graph.add(dynamic_command_group)` is called while a queue is recording.
- **Actual fix**: Replaced the `force_graph_compatible` early-return in `ggml_sycl_mul_mat()` (matmul.cpp) with a graph-safe dispatch that mirrors the normal dispatch ordering:
  1. F16 permuted/non-contiguous single-batch paths (custom kernels, graph-safe)
  2. DMMV (`can_use_dequantize_mul_mat_vec`) — graph-safe
  3. MMVQ (`can_use_mul_mat_vec_q`) — graph-safe, no reorder (already disabled)
  4. MMQ (`ggml_sycl_supports_mmq`) — graph-safe for all types **except** Q5_0/Q8_0 (excluded with belt-and-suspenders guard; already segmented out by `node_needs_immediate_mode()`)
  5. XMX (F32/F16) — graph-safe
  6. Tiled GEMM — last-resort fallback only (F32/F16/BF16/MXFP4 × F32)
  - oneMKL is **never** called during graph recording (graph-incompatible due to internal events).
  - Reordering is already disabled by `should_reorder_tensor()` checking `force_graph_compatible`.
- **Results**: All types now pass 100% with graphs enabled:
  - Q4_K: 35/35 MUL_MAT, 72/72 MUL_MAT_ID
  - Q8_0: 37/37, Q4_0: 37/37, Q5_0: 12/12, Q2_K: 11/11, Q6_K: 11/11
  - F32: 186/186, F16: 176/176
- **Segmented execution rules** (`node_needs_immediate_mode` in backend.cpp):
  - MUL_MAT_ID: always immediate (data-dependent expert routing, BCS page faults)
  - Q5_0/Q8_0 MUL_MAT: always immediate (GPU faults under SYCL graphs)
  - F16 permuted single-batch: graph-safe (direct kernels, no pool allocs)
  - Non-contiguous src0/src1: immediate (pool-allocated temporaries freed after recording)
  - Multi-sequence batch (ne[3] > 1): immediate (pool-allocated pointer arrays for batched GEMM)
  - GQA (r2 = src1->ne[2]/src0->ne[2] > 1): immediate (pool-allocated pointer arrays for batched GEMM dispatch)
  - F16/F32 src0: **Only immediate if GQA/batch**; XMX path is graph-safe when r2=1 and r3=1
  - Remaining quantized types (Q4_K, Q4_0, Q2_K-Q6_K): graph-safe via DMMV/MMVQ/MMQ paths
- **Test**:
  ```bash
  GGML_SYCL_DISABLE_GRAPH=0 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=q4_K"
  GGML_SYCL_DISABLE_GRAPH=0 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p "type_a=f32"
  GGML_SYCL_DISABLE_GRAPH=0 ./build-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID -p "type_a=q4_K"
  # Debug output should show "DMMV [graph]", "MMVQ [graph]", "MMQ [graph]" instead of "TILED [graph fallback]"
  ```

#### Fix 5: Enable graphs by default
- Once fixes 1–4 are stable and tested, flip `GGML_SYCL_DISABLE_GRAPH` default from `1` to `0`.
- Add a `GGML_SYCL_GRAPH_MODE` env var with values: `off` (current default), `replay` (fix 1 only), `full` (fixes 1+3).
- **Current status**: Graphs cause ~14% regression on Llama-3.2-1B Q4_K (173 t/s vs 201 t/s with graphs disabled). The segmentation overhead (10 graph + 9 immediate steps) is inherent for GQA models.
- **Updated status (Feb 2026, post-36388b86d)**: `FLASH_ATTN_EXT` is forced to eager in `node_needs_immediate_mode()` to avoid stale pooled pointer arrays during graph replay on GQA/MLA models (DEVICE_LOST). A fragmentation guard also falls back to full eager when the segmented execution plan is too fragmented (`>100` steps), because hundreds of tiny graph/eager transitions are slower than pure eager execution.
- **Implication**: Do **not** enable graphs by default yet. The next step is reducing graph-incompatible pointer-array/pool-allocation patterns rather than expanding eager fallbacks.

### GQA MUL_MAT Graph-Incompatibility (Feb 2026)

**Problem**: F16 MUL_MAT operations with GQA (Grouped Query Attention) cause BCS page faults when recorded into SYCL graphs.

**Root Cause**: GQA with `r2 = n_q_heads / n_kv_heads > 1` requires pool-allocated pointer arrays for batched GEMM dispatch:
```cpp
// matmul.cpp:763-765
ggml_sycl_pool_alloc<const void *> ptrs_src(ctx.pool(), 2 * ne23);
ggml_sycl_pool_alloc<void *>       ptrs_dst(ctx.pool(), 1 * ne23);
```
These RAII allocations are freed after `ggml_sycl_op_mul_mat` returns, but the recorded graph still references those addresses → page fault on BCS engine during replay.

**Key Insight**: Multi-head parallelism (ne[2] > 1) is NOT the same as GQA:
- **Multi-head (MHA)**: ne[2]=32, src0->ne[2]=32, r2=1 → contiguous batched GEMM, graph-safe
- **GQA**: ne[2]=32, src0->ne[2]=8, r2=4 → pool-allocated pointer arrays, graph-incompatible

**Detection**: Check `r2 = src1->ne[2] / src0->ne[2]` instead of raw `ne[2]`:
```cpp
const int64_t r2 = src1->ne[2] / src0->ne[2];  // GQA ratio
const int64_t r3 = src1->ne[3] / src0->ne[3];  // Batch ratio
if (r2 > 1 || r3 > 1) return true;  // Immediate mode required
```

**Result**: Llama-3.2-1B Q4_K generates 19 execution steps (10 graph + 9 immediate). The 9 immediate nodes are GQA MUL_MAT operations that cannot be avoided. This is the correct minimum segmentation for GQA models.

### Implementation Order
1. ~~Fix 2 — trivial, one queue property change~~ ✅
2. ~~Fix 1 — biggest perf win, well-validated by LLVM e2e tests~~ ✅
3. ~~Fix 3 — architecturally important, segmented graph execution~~ ✅
4. ~~Fix 4 — graph-safe dispatch (abandoned `dynamic_command_group`, used direct dispatch instead)~~ ✅
5. Fix 5 — final gate after stability validation

## What To Try Next (Post-36388b86d)

### Recommendation: Do not revert 36388b86d yet
- The latest commit is a **stability guardrail**, not just a pessimization: it prevents `DEVICE_LOST` from stale pointer arrays captured during graph recording (especially GQA/MLA `FLASH_ATTN_EXT`).
- Reverting it likely reintroduces crashes and invalidates performance comparisons (throughput wins are not meaningful if graph replay is unsafe).
- **Better path**: keep the eager fallback + fragmentation guard, then selectively re-enable graph mode only after eliminating the root causes below.

### Root Cause Work (highest priority)
1. **Make graph-replayed ops stop using ephemeral pool pointer arrays**
   - Problem pattern: RAII `ggml_sycl_pool_alloc<...>` pointer arrays / temporaries created during recording and freed before replay.
   - Next fix: allocate graph-stable pointer tables from a persistent cache keyed by graph topology + shape (or pointer hash), and refresh contents in-place each invocation.
   - Targets:
     - `MUL_MAT` batched/GQA paths (pointer arrays for batched GEMM dispatch)
     - `FLASH_ATTN_EXT` tiled/GQA/MLA fallback paths (pointer lists + H2D memcpy during recording)
   - Success criterion: those ops can run under graph replay with no BCS page faults / `DEVICE_LOST`.

2. **Split graph compatibility into finer-grained reasons (for measurement)**
   - Add per-reason counters in `node_needs_immediate_mode()` (e.g. `reason=flash_attn_ptr_arrays`, `reason=gqa_ptr_arrays`, `reason=noncontig_temp`, `reason=mul_mat_id_dynamic_ids`).
   - This makes it obvious which fallback dominates real models and prevents chasing low-impact fixes.
   - Use `GGML_SYCL_OP_STATS=1` + a new graph-plan summary counter to report step counts by reason.

3. **Replace the hard fragmentation cutoff with a cost model**
   - Current `>100` steps guard is a good emergency brake, but too coarse.
   - Next step: estimate cost = `N_graph_segments * graph_submit_overhead + N_boundaries * sync_cost + N_immediate_nodes * eager_cost`, and compare with pure eager.
   - Even a rough heuristic (weighted by node types / bytes moved) is better than a fixed threshold.

### Performance Work That Still Looks High-Value
4. **Token generation gap: profile segmented graph overhead vs eager on B60**
   - Measure with `onetrace` + ITT on `tg128` / `n=1` workloads and record:
     - graph submit latency
     - `stream()->wait()` boundary cost
     - top immediate-mode ops by time
   - Goal: confirm whether the regression is mostly segmentation/sync or kernel-level (`DMMV/MMVQ/dequantize`) cost.

5. **Fused Flash Attention (`fattn_fused`) register-pressure reduction**
   - Prior notes already point to accumulator/logit pressure; this remains likely high ROI for prompt throughput when XMX FA is unavailable.
   - Try:
     - smaller tiles (`BQ/BK`) per architecture
     - partial reduction in registers (chunked softmax / split accumulators)
     - subgroup-private + shared reduction variants
   - Verify with `onetrace` and compiler spill indicators (if available via IGC dumps / asm).

6. **XMX int8: focus on col_major A (not more prefetch tuning)**
   - v5 showed col_major B mattered; prefetch did not.
   - Next likely win is symmetric treatment for A tile layout (reduce scatter on `block_q8_0` loads), then accumulator-direct scale application to avoid SLM round-trip.

### Suggested Experimental Plan (safe sequence)
1. Keep `36388b86d` in place.
2. Add graph fallback reason counters + plan summary logging.
3. Implement persistent graph-owned pointer-table storage for one path first (`MUL_MAT` GQA), test replay stability.
4. If stable, re-enable graph for that path only; benchmark `tg128`.
5. Repeat for `FLASH_ATTN_EXT` tiled/GQA path.
6. Replace `>100` fragmentation guard with heuristic after data is collected.

### Open Question: `MUL_MAT_ID`
- `MUL_MAT_ID` is still correctly treated as immediate for now (data-dependent expert routing / ids content changes).
- If revisiting, treat it as a separate project: graph-safe execution likely needs a persistent indirection buffer strategy or explicit graph parameter updates, not just cache-key changes.

### Testing Strategy
- **Fix 1 correctness**: Run `test-backend-ops -b SYCL0` with `GGML_SYCL_DISABLE_GRAPH=0` and verify all ops pass.
- **Fix 1 perf**: Compare `llama-bench` with `GGML_SYCL_DEBUG=1` to count re-recordings per graph execution. Expect 1 recording on first call, 0 on subsequent calls with stable pointers.
- **Fix 2 perf**: Profile with `onetrace --device-timing` before/after adding `no_immediate_command_list`. Compare kernel launch latency.
- **Fix 3 correctness**: Run models that use oneMKL paths (F16 weights, large batches) with graphs enabled. Verify output matches non-graph path.
- **Fix 3 perf**: Measure overhead of segmented vs. fully-disabled graphs. Segmentation cost should be O(n_segments) per frame.

### Reference Files
- LLVM SYCL graph e2e tests: `~/llvm/sycl/test-e2e/Graph/`
- LLVM SYCL graph spec: `~/llvm/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc`
- LLVM immediate command list spec: `~/llvm/sycl/doc/extensions/supported/sycl_ext_intel_queue_immediate_command_list.asciidoc`
- oneDNN pause/resume pattern: `~/oneDNN/src/gpu/intel/sycl/stream.cpp`
- oneDNN immediate mode for zero pools: `~/oneDNN/src/gpu/intel/compute/zero_pool.cpp`
- oneDNN graph-aware GEMM: `~/oneDNN/src/gpu/intel/gemm/jit.cpp:336`
- Current graph implementation: `ggml/src/ggml-sycl/backend.cpp:595-1191`
- Graph cache/context: `ggml/src/ggml-sycl/common.hpp:410-423`
- Queue creation: `ggml/src/ggml-sycl/dpct/helper.hpp:786-810`

## Dual B60 SYCL Backend Review Follow-Up (Feb 2026)

### Backend structure map (hot paths)
- **Graph orchestration / execution planning**: `ggml/src/ggml-sycl/backend.cpp` (graph compute, segmented graph replay, immediate-mode fallback, multi-device graph path), `ggml/src/ggml-sycl/common.hpp` (context/cache state).
- **Device/arch setup**: `ggml/src/ggml-sycl/ggml-sycl.cpp` (runtime vendor/arch detection, Xe2 selection), `ggml/src/ggml-sycl/sycl_hw.*`.
- **Matmul dispatch + quantized paths**: `ggml/src/ggml-sycl/matmul.cpp` (`ggml_sycl_mul_mat`, `MUL_MAT_ID`, graph-safe dispatch), `ggml/src/ggml-sycl/dmmv.cpp`, `ggml/src/ggml-sycl/mmvq.cpp`, `ggml/src/ggml-sycl/mmq*.cpp`, `ggml/src/ggml-sycl/mmq_internal.hpp`, `ggml/src/ggml-sycl/mmq_xmx_int8.cpp`, `ggml/src/ggml-sycl/gemm_tiled.hpp`, `ggml/src/ggml-sycl/gemm_xmx.hpp`.
- **Flash attention**: `ggml/src/ggml-sycl/fattn.cpp` (dispatch/policy), `ggml/src/ggml-sycl/fattn_xmx*`, `ggml/src/ggml-sycl/fattn_fused.hpp`, `ggml/src/ggml-sycl/fattn_tiled.hpp`, `ggml/src/ggml-sycl/fattn_mkl.*`.
- **Memory/scratch/instrumentation**: `ggml/src/ggml-sycl/pool.cpp`, `ggml/src/ggml-sycl/flash_attn_buffers.hpp`, `ggml/src/ggml-sycl/itt_annotations.hpp`.

### Likely current bottlenecks (dual B60 and graph mode)
1. **Segmented graph boundary overhead** in `ggml/src/ggml-sycl/backend.cpp` (repeated queue waits and graph/eager transitions) is a likely dominant token-generation regression source for `n=1`.
2. **Ephemeral pool allocations captured during graph recording** (notably GQA/batched pointer arrays in `ggml/src/ggml-sycl/matmul.cpp`, FA fallback metadata paths via `ggml/src/ggml-sycl/fattn.cpp` + `flash_attn_buffers.hpp`) force immediate-mode fallback and fragment graphs.
3. **`MMQ_MIN_NROWS=128` guard in `ggml/src/ggml-sycl/matmul.cpp`** disables MMQ on many small-row tg shapes, pushing quantized matmuls to less optimal paths.
4. **XMX int8 quantized path still layout/SLM limited** in `ggml/src/ggml-sycl/mmq_xmx_int8.cpp` (AoS block unpack/scatter cost vs contiguous GEMM).
5. **Fused FA register pressure / occupancy** in `ggml/src/ggml-sycl/fattn_fused.hpp` limits prompt throughput when XMX FA is unavailable.
6. **Multi-device scheduling is too coarse** in `ggml/src/ggml-sycl/backend.cpp` (split-buffer node assignment + sequentialized sync points) and likely underlaps compute with P2P/BCS transfers.

### Prioritized plan (dual B60 prompt + token generation)
1. **Graph-safe persistent metadata/pointer tables (highest priority)**
   - Add graph-owned persistent allocations keyed by graph topology/shape/device for GQA/batched GEMM and FA pointer/metadata tables.
   - Goal: shrink immediate-mode nodes in `ggml/src/ggml-sycl/backend.cpp` plans and reduce segmentation overhead.
2. **Dependency-aware multi-device scheduler for dual B60 prompt throughput**
   - Replace coarse per-device sequencing in `ggml/src/ggml-sycl/backend.cpp` with event-driven per-device subgraph DAG execution and explicit inter-device dependencies.
   - Preserve overlap of compute + P2P/BCS copies; avoid full queue waits between phases.
3. **Cost-model graph planner (replace coarse fragmentation heuristic)**
   - In `ggml/src/ggml-sycl/backend.cpp`, choose eager vs segmented vs monolithic replay from measured submit/wait/eager costs instead of a fixed step-count cutoff.
4. **Small-batch quantized tg path tuning on Xe2**
   - Revisit `MMQ_MIN_NROWS` behavior in `ggml/src/ggml-sycl/matmul.cpp` with shape-specialized kernels/fallbacks; reduce dispatch/submission overhead across `dmmv.cpp` / `mmvq.cpp` / MMQ paths.
5. **Prompt throughput kernel work**
   - Optimize `ggml/src/ggml-sycl/fattn_fused.hpp` (register pressure / tiles / occupancy) and continue `ggml/src/ggml-sycl/mmq_xmx_int8.cpp` layout work (col-major A, scale application without extra SLM traffic).

### Graph-mode regression root causes (architectural)
- **Primary issue is architectural, not only kernel speed**: graph replay is fragmented by graph-incompatible temporary allocation patterns and then pays high boundary synchronization costs in `ggml/src/ggml-sycl/backend.cpp`.
- **Needed changes beyond current guardrails**:
  - graph-lifetime resource ownership (persistent pointer tables / descriptors / scratch metadata),
  - finer-grained graph-safe contracts per op subpath (`ggml/src/ggml-sycl/matmul.cpp`, `ggml/src/ggml-sycl/fattn.cpp`),
  - event-based execution plans instead of frequent full queue waits,
  - unified single-device + multi-device graph planning/caching strategy.
- **Do not enable graphs by default** until the above reduces fragmentation for GQA/MLA-heavy models and closes the `tg` regression on B60.

### Measurement / bench harness gaps to close
- Add **per-reason fallback counters and timing** for `node_needs_immediate_mode()` in `ggml/src/ggml-sycl/backend.cpp` (e.g. GQA pointer arrays, FA pointer arrays, noncontig temps, `MUL_MAT_ID`).
- Add **segmented graph boundary timing attribution** (graph submit time vs `wait()` boundary cost vs eager step time), ideally with ITT regions (`ggml/src/ggml-sycl/itt_annotations.hpp`).
- Create a **dual-B60 benchmark matrix** (graph on/off, split-mode layer/row, tensor-split, P2P on/off, prompt/tg sizes, larger models than 1B) to expose scaling behavior.
- Add **tg-shape microbench coverage** for `MUL_MAT` and `FLASH_ATTN_EXT` hot shapes on Xe2 (not just correctness via `test-backend-ops`).
- Standardize **SYCL vs Vulkan apples-to-apples runs** (same model, quant, warmup, prompt/tg mix, KV settings) before attributing regressions to kernels.
