#!/usr/bin/env python3
"""
Multi-build SYCL/Vulkan benchmark runner for llama-bench.

Runs a test matrix across:
  - Multiple builds (branch SYCL, master SYCL, master Vulkan)
  - All Q4/Q8 GGUF models found in models/
  - SYCL graph: on vs off (only for builds that support it)
  - Flash attention: on vs off

Results are saved as JSON files under bench_results/.
If a result file already exists for a combination, that test is skipped.

Usage:
    source /opt/intel/oneapi/setvars.sh intel64
    python3 bench.py [--dry-run] [--results-dir DIR] [--summary-only]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "bench_results"
DEFAULT_MODELS = SCRIPT_DIR / "models"

PROMPT_SIZES = [128, 512]
GEN_SIZES = [64, 128]
BATCH_SIZE = 512
UBATCH_SIZE = 512
REPETITIONS = 3
N_GPU_LAYERS = 99

BUILDS = [
    {
        "name": "branch_sycl",
        "bench": SCRIPT_DIR / "build-sycl" / "bin" / "llama-bench",
        "backend": "SYCL",
        "supports_graph": True,
        "description": "Local SYCL branch (graph-enabled)",
    },
    {
        "name": "master_sycl",
        "bench": Path("/home/maxious/llama-master/build-sycl/bin/llama-bench"),
        "backend": "SYCL",
        "supports_graph": False,
        "description": "Master branch SYCL build",
    },
    {
        "name": "master_vulkan",
        "bench": Path("/home/maxious/llama-master/build-vulkan/bin/llama-bench"),
        "backend": "Vulkan",
        "supports_graph": False,
        "description": "Master branch Vulkan build",
    },
]


def find_models(models_dir: Path) -> list[Path]:
    patterns = ["*[Qq]4*.gguf", "*[Qq]8*.gguf"]
    found = []
    for pat in patterns:
        for p in sorted(models_dir.glob(pat)):
            if p.stat().st_size < 10 * 1024 * 1024:
                continue
            # Exclude K-quant models (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) due to IGC crash issues
            if re.search(r"[Qq][2-8]_[Kk]", p.name):
                continue
            found.append(p)
    return sorted(set(found))


def model_tag(model_path: Path) -> str:
    name = model_path.stem
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def result_filename(
    build_name: str,
    model_path: Path,
    graph: bool | None,
    fa: bool,
    pp: int,
    tg: int,
    device_mode: str,
) -> str:
    tag = model_tag(model_path)
    if graph is None:
        graph_str = "graph_default"
    else:
        graph_str = "graph_on" if graph else "graph_off"
    fa_str = "fa_on" if fa else "fa_off"
    return (
        f"{build_name}__{tag}__{graph_str}__{fa_str}__pp{pp}_tg{tg}_{device_mode}.json"
    )


def run_bench(
    build: dict,
    model_path: Path,
    graph: bool | None,
    fa: bool,
    pp: int,
    tg: int,
    results_dir: Path,
    device_mode: str,
    dry_run: bool = False,
) -> dict | None:
    fname = result_filename(build["name"], model_path, graph, fa, pp, tg, device_mode)
    out_path = results_dir / fname

    if out_path.exists():
        print(f"  SKIP (exists): {fname}")
        return None

    env = os.environ.copy()

    if build["supports_graph"] and graph is not None:
        env["GGML_SYCL_DISABLE_GRAPH"] = "0" if graph else "1"
    else:
        env.pop("GGML_SYCL_DISABLE_GRAPH", None)

    # Device selection logic
    if device_mode == "single":
        if build["backend"] == "SYCL":
            env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"

    # Add the binary's directory to LD_LIBRARY_PATH so libllama.so is found
    bin_dir = build["bench"].parent
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(bin_dir) + (":" + ld_path if ld_path else "")

    # For Vulkan builds, select Intel Arc GPU via VK_ICD_FILENAMES
    if build["backend"] == "Vulkan":
        icd_path = "/usr/share/vulkan/icd.d/intel_icd.json"
        if os.path.exists(icd_path):
            env["VK_ICD_FILENAMES"] = icd_path
        else:
            # Fallback: try common alternative paths
            alt_paths = [
                "/usr/share/vulkan/icd.d/intel_icd.x86_64.json",
                "/etc/vulkan/icd.d/intel_icd.json",
            ]
            for p in alt_paths:
                if os.path.exists(p):
                    env["VK_ICD_FILENAMES"] = p
                    break
            else:
                print(
                    f"  WARNING: Intel Vulkan ICD not found, using default device selection"
                )

    cmd = [
        str(build["bench"]),
        "-m",
        str(model_path),
        "-p",
        str(pp),
        "-n",
        str(tg),
        "-b",
        str(BATCH_SIZE),
        "-ub",
        str(UBATCH_SIZE),
        "-ngl",
        str(N_GPU_LAYERS),
        "-fa",
        "1" if fa else "0",
        "-r",
        str(REPETITIONS),
        "-o",
        "json",
        "--progress",
    ]

    if build["backend"] == "Vulkan" and device_mode == "single":
        cmd.extend(["--device", "Vulkan0"])

    if graph is None:
        graph_label = "default"
    else:
        graph_label = "on" if graph else "off"

    label = (
        f"[{build['name']}] model={model_path.name} graph={graph_label} "
        f"fa={'on' if fa else 'off'} pp={pp} tg={tg}"
    )

    if dry_run:
        print(f"  DRY-RUN: {label}")
        print(f"    cmd: {' '.join(cmd)}")
        if build["supports_graph"] and graph is not None:
            print(
                f"    env: GGML_SYCL_DISABLE_GRAPH={env.get('GGML_SYCL_DISABLE_GRAPH', 'unset')}"
            )
        return None

    print(f"  RUN: {label}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s: {label}")
        error_result = {
            "error": "timeout",
            "label": label,
            "cmd": cmd,
            "build": build["name"],
        }
        out_path.with_suffix(".error.json").write_text(
            json.dumps(error_result, indent=2)
        )
        return None

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}) after {elapsed:.1f}s")
        print(f"    stderr: {result.stderr[:500]}")
        error_result = {
            "error": f"exit_code_{result.returncode}",
            "label": label,
            "cmd": cmd,
            "build": build["name"],
            "stderr": result.stderr[:2000],
            "stdout": result.stdout[:2000],
        }
        out_path.with_suffix(".error.json").write_text(
            json.dumps(error_result, indent=2)
        )
        return None

    stdout = result.stdout.strip()
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\])", stdout, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            print(f"  ERROR: Could not parse JSON output")
            print(f"    stdout: {stdout[:500]}")
            return None

    wrapped = {
        "meta": {
            "build": build["name"],
            "backend": build["backend"],
            "build_description": build["description"],
            "model": str(model_path),
            "model_name": model_path.name,
            "graph_enabled": graph,
            "flash_attn": fa,
            "pp": pp,
            "tg": tg,
            "batch_size": BATCH_SIZE,
            "ubatch_size": UBATCH_SIZE,
            "repetitions": REPETITIONS,
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "device_mode": device_mode,
        },
        "results": data,
    }

    out_path.write_text(json.dumps(wrapped, indent=2))
    print(f"  OK ({elapsed:.1f}s) -> {fname}")
    return wrapped


def print_summary(results_dir: Path):
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print("\nNo results to summarize.")
        return

    rows = []
    for f in files:
        if ".error." in f.name:
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue

        meta = data.get("meta", {})
        results = data.get("results", [])

        graph_val = meta.get("graph_enabled")
        if graph_val is None:
            graph_str = "def"
        else:
            graph_str = "on" if graph_val else "off"

        # Extract device_mode from meta or filename
        device_mode = meta.get("device_mode")
        if not device_mode:
            # Fallback: parse from filename
            # Format: ..._ppX_tgY_{device_mode}.json
            # But wait, previous filenames didn't have device_mode at all (before my changes).
            # The suffix was just added.
            # Filename: build__tag__graph__fa__pp_tg_mode.json
            parts = f.stem.split("_")
            if parts[-1] in ["single", "multi"]:
                device_mode = parts[-1]
            else:
                # Assume multi if not specified (legacy files renamed to _multi)
                # But wait, I renamed them to _multi.json.
                # So they should have _multi.
                if f.name.endswith("_multi.json"):
                    device_mode = "multi"
                elif f.name.endswith("_single.json"):
                    device_mode = "single"
                else:
                    device_mode = "?"

        for r in results:
            if "test" in r:
                test_name = r["test"]
            else:
                # Infer from n_prompt/n_gen
                np = r.get("n_prompt", 0)
                ng = r.get("n_gen", 0)
                if np > 0 and ng == 0:
                    test_name = f"pp{np}"
                elif ng > 0:
                    test_name = f"tg{ng}"
                else:
                    test_name = "?"

            rows.append(
                {
                    "build": meta.get("build", "?"),
                    "model": meta.get("model_name", "?"),
                    "graph": graph_str,
                    "fa": "on" if meta.get("flash_attn") else "off",
                    "device": device_mode,
                    "test": test_name,
                    "t/s": r.get("avg_ts", r.get("t/s", "?")),
                    "std": r.get("stddev_ts", r.get("stddev", "?")),
                }
            )

    if not rows:
        print("\nNo results to summarize.")
        return

    error_files = list(results_dir.glob("*.error.json"))

    print(f"\n{'=' * 110}")
    print(
        f"BENCHMARK SUMMARY ({len(files) - len(error_files)} result files, {len(error_files)} errors)"
    )
    print(f"{'=' * 110}")
    header = f"{'Build':<16} {'Model':<28} {'Graph':>5} {'FA':>3} {'Dev':>6} {'Test':>8} {'t/s':>10} {'stddev':>10}"
    print(header)
    print("-" * len(header))

    rows.sort(
        key=lambda r: (
            r["build"],
            r["model"],
            r["graph"],
            r["fa"],
            r["device"],
            r["test"],
        )
    )

    for r in rows:
        ts = f"{r['t/s']:.2f}" if isinstance(r["t/s"], (int, float)) else str(r["t/s"])
        std = f"{r['std']:.2f}" if isinstance(r["std"], (int, float)) else str(r["std"])
        print(
            f"{r['build']:<16} {r['model']:<28} {r['graph']:>5} {r['fa']:>3} {r['device']:>6} {r['test']:>8} {ts:>10} {std:>10}"
        )


def main():
    # IGC workaround: use locally compiled patched IGC if available
    igc_lib_dir = Path.home() / "igc_workspace" / "build" / "IGC" / "Release"
    if igc_lib_dir.exists():
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(igc_lib_dir) + (
            ":" + ld_path if ld_path else ""
        )
        print(f"[bench] Using patched IGC from: {igc_lib_dir}")
    else:
        print(
            "[bench] WARNING: Patched IGC not found at ~/igc_workspace/build/IGC/Release, using system IGC"
        )

    parser = argparse.ArgumentParser(
        description="Multi-build llama-bench test matrix runner"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Directory for result JSON files (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS,
        help=f"Directory to search for models (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary of existing results",
    )
    parser.add_argument(
        "--builds",
        nargs="+",
        choices=[b["name"] for b in BUILDS],
        default=None,
        help="Run only specific builds (default: all available)",
    )
    parser.add_argument(
        "--devices",
        choices=["single", "multi", "both"],
        default="both",
        help="Device configuration: single=use one GPU, multi=use multiple GPUs, both=run both (default: both)",
    )
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print_summary(args.results_dir)
        return

    active_builds = []
    for build in BUILDS:
        if args.builds and build["name"] not in args.builds:
            continue
        if not build["bench"].exists():
            print(
                f"WARNING: {build['name']} binary not found at {build['bench']}, skipping"
            )
            continue
        active_builds.append(build)

    if not active_builds:
        print("ERROR: No builds available", file=sys.stderr)
        sys.exit(1)

    models = find_models(args.models_dir)
    if not models:
        print(
            f"ERROR: No Q4/Q8 GGUF models found in {args.models_dir}", file=sys.stderr
        )
        sys.exit(1)

    print(f"Found {len(models)} model(s):")
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        print(f"  {m.name} ({size_mb:.0f} MB)")

    print(f"\nActive builds ({len(active_builds)}):")
    for b in active_builds:
        graph_note = "graph ON/OFF" if b["supports_graph"] else "no graph toggle"
        print(f"  {b['name']}: {b['description']} ({graph_note})")

    fa_opts = [False, True]
    workloads = [(pp, tg) for pp in PROMPT_SIZES for tg in GEN_SIZES]

    # Determine device modes to run
    if args.devices == "both":
        device_modes = ["single", "multi"]
    else:
        device_modes = [args.devices]

    total = 0
    for build in active_builds:
        if build["supports_graph"]:
            graph_opts = [False, True]
        else:
            graph_opts = [None]
        total += (
            len(models)
            * len(graph_opts)
            * len(fa_opts)
            * len(workloads)
            * len(device_modes)
        )

    print(
        f"\nTest matrix: {total} total tests across {len(active_builds)} builds, "
        f"{len(models)} models, {len(fa_opts)} fa, {len(workloads)} workloads, "
        f"{len(device_modes)} device mode(s) ({', '.join(device_modes)})"
    )
    print(f"Results dir: {args.results_dir}")
    print()

    completed = 0
    skipped = 0
    failed = 0

    for device_mode in device_modes:
        print(f"\n{'=' * 60}")
        print(f"DEVICE MODE: {device_mode}")
        print(f"{'=' * 60}")

        for build in active_builds:
            print(f"\n{'=' * 60}")
            print(f"BUILD: {build['name']} -- {build['description']}")
            print(f"{'=' * 60}")

            if build["supports_graph"]:
                graph_opts = [False, True]
            else:
                graph_opts = [None]

            for model in models:
                print(f"\n--- {model.name} ({build['name']}) ---")
                for graph in graph_opts:
                    for fa in fa_opts:
                        for pp, tg in workloads:
                            fname = result_filename(
                                build["name"], model, graph, fa, pp, tg, device_mode
                            )
                            out_path = args.results_dir / fname

                            if out_path.exists():
                                skipped += 1
                                print(f"  SKIP (exists): {fname}")
                                continue

                            result = run_bench(
                                build,
                                model,
                                graph,
                                fa,
                                pp,
                                tg,
                                args.results_dir,
                                device_mode,
                                args.dry_run,
                            )
                            if result is not None:
                                completed += 1
                            elif not args.dry_run and not out_path.exists():
                                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {completed} completed, {skipped} skipped, {failed} failed")
    print(f"{'=' * 60}")

    if not args.dry_run:
        print_summary(args.results_dir)


if __name__ == "__main__":
    main()
