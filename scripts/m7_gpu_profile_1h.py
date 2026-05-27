#!/usr/bin/env python
"""M7 1h Canary GPU wall-clock and profiler orchestrator."""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
from jax import config  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name  # noqa: E402
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational  # noqa: E402


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-26-m7-gpu-profile-prep"
ARTIFACT_ROOT = Path("/tmp/m7_profile_artifacts")
RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DT_S = 10.0
RUN_IDS = {
    "20260429": "20260429_18z_l3_24h_20260524T204451Z",
    "20260509": "20260509_18z_l3_24h_20260511T190519Z",
    "20260521": "20260521_18z_l3_24h_20260522T072630Z",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_id(run_key: str) -> str:
    return RUN_IDS.get(run_key, run_key)


def _hours_for_steps(steps: int) -> float:
    return float(steps) * DT_S / 3600.0


def _steps_for_hours(hours: float) -> int:
    raw = float(hours) * 3600.0 / DT_S
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-8:
        raise ValueError(f"hours={hours} does not align with dt={DT_S}s")
    return rounded


def _build_case(run_key: str) -> tuple[Any, OperationalNamelist, dict[str, Any]]:
    run_id = _run_id(run_key)
    run_dir = RUN_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"missing Gen2 run directory: {run_dir}")
    case = build_replay_case(run_dir)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=DT_S,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
    meta = {
        "run_key": run_key,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "grid": case.metadata["grid"],
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
        },
    }
    return state, namelist, meta


def _timed_forecast(run_key: str, hours: float, annotation: str) -> tuple[float, dict[str, Any]]:
    state, namelist, meta = _build_case(run_key)
    start = time.perf_counter()
    with jax.profiler.TraceAnnotation(annotation):
        result = run_forecast_operational(state, namelist, float(hours))
        block_until_ready(result)
    wall_s = time.perf_counter() - start
    meta["hours"] = float(hours)
    meta["rk_steps"] = _steps_for_hours(hours)
    return wall_s, meta


def measure_one_run(run_key: str, hours: float) -> dict[str, Any]:
    cold_wall_s, meta = _timed_forecast(run_key, hours, f"m7_cold_1h_{run_key}")
    warm_wall_s, _ = _timed_forecast(run_key, hours, f"m7_warm_1h_{run_key}")
    total_steps = int(meta["rk_steps"])
    return {
        **meta,
        "status": "PASS",
        "cold_start_jit_compile_inclusive_wall_s": cold_wall_s,
        "warm_jit_cached_wall_s": warm_wall_s,
        "total_rk_steps": total_steps,
        "per_rk_step_median_wall_ms": (warm_wall_s / max(total_steps, 1)) * 1000.0,
        "per_rk_step_wall_method": (
            "Derived from one warm full-forecast compiled scan divided by RK step count; "
            "the operational loop is intentionally one XLA program, so no host-side "
            "per-step timers are inserted."
        ),
    }


def _child_json(cmd: list[str], stdout_path: Path, stderr_path: Path) -> tuple[int, dict[str, Any] | None]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    payload = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None
    return proc.returncode, payload


def run_wall_clock(hours: float, run_keys: list[str], output: Path) -> dict[str, Any]:
    command_logs = SPRINT / "command_outputs"
    runs: list[dict[str, Any]] = []
    child_results: list[dict[str, Any]] = []
    for run_key in run_keys:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "single-wall-clock",
            "--run-key",
            run_key,
            "--hours",
            str(hours),
        ]
        rc, payload = _child_json(
            cmd,
            command_logs / f"wall_clock_{run_key}.stdout",
            command_logs / f"wall_clock_{run_key}.stderr",
        )
        child_results.append({"run_key": run_key, "command": cmd, "returncode": rc})
        if payload is None:
            runs.append({"run_key": run_key, "status": "FAIL", "error": "child did not emit JSON"})
        else:
            payload["child_returncode"] = rc
            runs.append(payload)
    status = "PASS" if all(run.get("status") == "PASS" and int(run.get("child_returncode", 1)) == 0 for run in runs) else "FAIL"
    payload = {
        "artifact_type": "m7_gpu_profile_wall_clock",
        "status": status,
        "device": visible_gpu_name(),
        "cpu_pinning": "expected taskset -c 0-3 from invocation",
        "hours": float(hours),
        "run_keys": run_keys,
        "run_ids": {key: _run_id(key) for key in run_keys},
        "timing_scope": "run_forecast_operational call plus block_until_ready; replay-case construction is outside timed window",
        "child_processes": child_results,
        "runs": runs,
    }
    _write_json(output, payload)
    return payload


def run_reproducibility(run_key: str, hours: float, runs: int, output: Path) -> dict[str, Any]:
    warmup_wall_s, meta = _timed_forecast(run_key, hours, f"m7_repro_warmup_{run_key}")
    samples: list[float] = []
    for idx in range(1, runs + 1):
        wall_s, _ = _timed_forecast(run_key, hours, f"m7_repro_run_{idx}_{run_key}")
        samples.append(wall_s)
    mean = statistics.fmean(samples) if samples else math.nan
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    cv = stdev / mean if mean and math.isfinite(mean) else math.inf
    payload = {
        "artifact_type": "m7_gpu_profile_reproducibility",
        "status": "PASS" if cv <= 0.05 else "BLOCKED-PERF",
        "device": visible_gpu_name(),
        **meta,
        "warmup_wall_s": warmup_wall_s,
        "warm_runs": runs,
        "warm_wall_s_samples": samples,
        "mean_wall_s": mean,
        "population_stdev_wall_s": stdev,
        "coefficient_of_variation": cv,
        "threshold_cv": 0.05,
        "jit_cache_policy": "No cache wipe between warm reruns.",
    }
    _write_json(output, payload)
    return payload


def _cuda_profiler_call(name: str) -> None:
    library = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(library)
    result = getattr(cudart, name)()
    if int(result) != 0:
        raise RuntimeError(f"{name} failed with CUDA error {result}")


def run_profile_window(
    *,
    run_key: str,
    steps: int,
    warmups: int,
    use_cuda_range: bool,
    output: Path,
    annotation: str = "m7_profile_window",
) -> dict[str, Any]:
    hours = _hours_for_steps(steps)
    warmup_times = []
    for idx in range(1, warmups + 1):
        wall_s, _ = _timed_forecast(run_key, hours, f"{annotation}_warmup_{idx}")
        warmup_times.append(wall_s)
    if use_cuda_range:
        _cuda_profiler_call("cudaProfilerStart")
    profile_wall_s, meta = _timed_forecast(run_key, hours, annotation)
    if use_cuda_range:
        _cuda_profiler_call("cudaProfilerStop")
    payload = {
        "artifact_type": "m7_gpu_profile_window_call_log",
        "status": "PASS",
        "device": visible_gpu_name(),
        **meta,
        "profile_steps": int(steps),
        "profile_wall_s": profile_wall_s,
        "profile_wall_ms_per_rk_step": (profile_wall_s / max(steps, 1)) * 1000.0,
        "warmups_outside_profile_window": int(warmups),
        "warmup_wall_s": warmup_times,
        "cuda_profiler_range_used": bool(use_cuda_range),
        "nvtx_annotation": annotation,
    }
    _write_json(output, payload)
    return payload


def _load_top_kernel_names(path: Path, count: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(row.get("name", ""))
        for row in payload.get("longest_10_kernels", [])[:count]
        if str(row.get("name", ""))
    ]


def _kernel_regex(name: str) -> str:
    # Nsight Compute regexes are RE2-like and can choke on very long C++/XLA
    # symbol spellings. A literal prefix is stable enough for the top-kernel
    # spot check and keeps the command readable in the report.
    prefix = name.strip().split("\n", maxsplit=1)[0][:160]
    if not prefix:
        return ".*"
    return re.escape(prefix)


def _parse_ncu_csv(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"available": False, "reason": "missing log"}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.startswith('"') or "Metric Name" not in text:
            continue
    try:
        reader = csv.DictReader(line for line in text.splitlines() if line.startswith('"'))
        rows = [dict(row) for row in reader if row.get("Metric Name")]
    except csv.Error as exc:
        return {"available": False, "reason": f"csv parse failed: {exc}", "raw_log": str(log_path)}

    metrics = {row.get("Metric Name", ""): row for row in rows}

    def find_value(*needles: str) -> dict[str, Any] | None:
        lowered = [(name.lower(), row) for name, row in metrics.items()]
        for needle in needles:
            needle_lower = needle.lower()
            for name, row in lowered:
                if needle_lower in name:
                    return {
                        "metric_name": row.get("Metric Name"),
                        "unit": row.get("Metric Unit"),
                        "value": row.get("Metric Value"),
                    }
        return None

    return {
        "available": bool(rows),
        "raw_log": str(log_path),
        "registers_per_thread": find_value("registers per thread", "launch__registers_per_thread"),
        "achieved_occupancy": find_value("achieved occupancy", "warps_active", "occupancy"),
        "achieved_memory_bandwidth": find_value("memory throughput", "dram__throughput", "memory bandwidth"),
        "local_memory": find_value("local memory", "local_mem", "lmem"),
        "achieved_flops": find_value("flop", "flops", "sm__throughput"),
        "metric_count": len(rows),
    }


def run_ncu_hot_kernels(
    *,
    nsys_summary: Path,
    output: Path,
    run_key: str,
    steps: int,
    count: int,
) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    kernels = _load_top_kernel_names(nsys_summary, count)
    results: list[dict[str, Any]] = []
    for idx, kernel in enumerate(kernels, start=1):
        report_base = ARTIFACT_ROOT / f"m7_ncu_hot_kernel_{idx}"
        log_path = ARTIFACT_ROOT / f"m7_ncu_hot_kernel_{idx}.csv"
        regex = _kernel_regex(kernel)
        cmd = [
            "ncu",
            "--set",
            "basic",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{regex}",
            "--launch-count",
            "1",
            "--force-overwrite",
            "--export",
            str(report_base),
            "--csv",
            "--page",
            "raw",
            "--log-file",
            str(log_path),
            "taskset",
            "-c",
            "0-3",
            sys.executable,
            str(Path(__file__).resolve()),
            "ncu-window",
            "--run-key",
            run_key,
            "--steps",
            str(steps),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        stdout_path = ARTIFACT_ROOT / f"m7_ncu_hot_kernel_{idx}.stdout"
        stderr_path = ARTIFACT_ROOT / f"m7_ncu_hot_kernel_{idx}.stderr"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        results.append(
            {
                "rank": idx,
                "kernel_name_from_nsys": kernel,
                "kernel_regex": regex,
                "status": "PASS" if proc.returncode == 0 else "BLOCKED-PROFILER",
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "ncu_report_path": str(report_base) + ".ncu-rep",
                "ncu_csv_log": str(log_path),
                "metrics": _parse_ncu_csv(log_path),
            }
        )
    status = "PASS" if results and all(item["status"] == "PASS" for item in results) else "BLOCKED-PROFILER"
    payload = {
        "artifact_type": "m7_ncu_hot_kernels",
        "status": status,
        "device": visible_gpu_name(),
        "nsys_summary": str(nsys_summary),
        "run_key": run_key,
        "profile_steps": int(steps),
        "raw_artifact_root": str(ARTIFACT_ROOT),
        "kernels": results,
    }
    _write_json(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    single = sub.add_parser("single-wall-clock")
    single.add_argument("--run-key", required=True)
    single.add_argument("--hours", type=float, default=1.0)

    wall = sub.add_parser("wall-clock")
    wall.add_argument("--hours", type=float, default=1.0)
    wall.add_argument("--run-key", action="append", dest="run_keys")
    wall.add_argument("--output", type=Path, default=SPRINT / "wall_clock.json")

    repro = sub.add_parser("reproducibility")
    repro.add_argument("--run-key", default="20260521")
    repro.add_argument("--hours", type=float, default=1.0)
    repro.add_argument("--runs", type=int, default=3)
    repro.add_argument("--output", type=Path, default=SPRINT / "reproducibility.json")

    profile = sub.add_parser("profile-window")
    profile.add_argument("--run-key", default="20260521")
    profile.add_argument("--steps", type=int, default=360)
    profile.add_argument("--warmups", type=int, default=2)
    profile.add_argument("--cuda-profiler-range", action="store_true")
    profile.add_argument("--output", type=Path, default=SPRINT / "nsys_capture_call_log.json")

    ncu_window = sub.add_parser("ncu-window")
    ncu_window.add_argument("--run-key", default="20260521")
    ncu_window.add_argument("--steps", type=int, default=100)
    ncu_window.add_argument("--warmups", type=int, default=1)
    ncu_window.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "m7_ncu_window_call_log.json")

    ncu = sub.add_parser("ncu-hot-kernels")
    ncu.add_argument("--nsys-summary", type=Path, default=SPRINT / "nsys_summary.json")
    ncu.add_argument("--output", type=Path, default=SPRINT / "ncu_hot_kernels.json")
    ncu.add_argument("--run-key", default="20260521")
    ncu.add_argument("--steps", type=int, default=100)
    ncu.add_argument("--count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "single-wall-clock":
        payload = measure_one_run(args.run_key, args.hours)
    elif args.cmd == "wall-clock":
        payload = run_wall_clock(args.hours, args.run_keys or list(RUN_IDS), args.output)
    elif args.cmd == "reproducibility":
        payload = run_reproducibility(args.run_key, args.hours, args.runs, args.output)
    elif args.cmd == "profile-window":
        use_range = bool(args.cuda_profiler_range or os.environ.get("GPUWRF_CUDA_PROFILER_RANGE") == "1")
        payload = run_profile_window(
            run_key=args.run_key,
            steps=args.steps,
            warmups=args.warmups,
            use_cuda_range=use_range,
            output=args.output,
            annotation="m7_profile_window",
        )
    elif args.cmd == "ncu-window":
        payload = run_profile_window(
            run_key=args.run_key,
            steps=args.steps,
            warmups=args.warmups,
            use_cuda_range=False,
            output=args.output,
            annotation="m7_ncu_window",
        )
    elif args.cmd == "ncu-hot-kernels":
        payload = run_ncu_hot_kernels(
            nsys_summary=args.nsys_summary,
            output=args.output,
            run_key=args.run_key,
            steps=args.steps,
            count=args.count,
        )
    else:
        raise AssertionError(args.cmd)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
