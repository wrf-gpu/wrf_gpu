#!/usr/bin/env python3
"""Run ADR-007 precision-policy timing probes and write machine-readable artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any


os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ARTIFACT_DIR = ROOT / "artifacts" / "precision-bench"
SCRATCH_DIR = ROOT / "data" / "scratch" / "precision-bench"
GEN2_ENV = Path("/home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh")
GEN2_WRF = Path("/home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF")

KERNELS = ("m2_column", "m4_dycore", "m5_thompson")
GPU_PRECISIONS = {
    "m2_column": ("fp64", "fp32", "bf16"),
    "m4_dycore": ("fp64", "fp32", "bf16"),
    "m5_thompson": ("fp64", "fp32", "bf16"),
}
CPU_BASELINE_PRECISION = "fp64"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    path = path.absolute()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _jax_modules():
    import jax
    from jax import config
    import jax.numpy as jnp

    config.update("jax_enable_x64", True)
    return jax, jnp


def _dtype_for(precision: str):
    _jax, jnp = _jax_modules()
    if precision == "fp64":
        return jnp.float64
    if precision == "fp32":
        return jnp.float32
    if precision == "bf16":
        return jnp.bfloat16
    raise ValueError(f"unknown precision {precision!r}")


def _scalar(dtype: Any, value: float):
    _jax, jnp = _jax_modules()
    return jnp.asarray(value, dtype=dtype)


def _block_until_ready(value: Any) -> None:
    jax, _jnp = _jax_modules()
    jax.tree_util.tree_map(lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item, value)


def _compiled_text(compiled: Any) -> str:
    for kwargs in ({"dialect": "hlo"}, {}):
        try:
            return str(compiled.as_text(**kwargs))
        except TypeError:
            continue
    return str(compiled)


def _kernel_launches(hlo_text: str) -> int:
    from gpuwrf.profiling.budget import kernel_launches_per_step

    return int(kernel_launches_per_step(hlo_text))


def _write_hlo(artifact_dir: Path, backend: str, kernel: str, precision: str, hlo_text: str) -> str:
    path = artifact_dir / "hlo" / f"{backend}-{kernel}-{precision}.txt"
    full_path = SCRATCH_DIR / "hlo-full" / f"{backend}-{kernel}-{precision}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = hlo_text.encode("utf-8")
    if len(encoded) <= 100_000:
        path.write_text(hlo_text, encoding="utf-8")
    else:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(hlo_text, encoding="utf-8")
        head = "\n".join(hlo_text.splitlines()[:1000])
        path.write_text(head + f"\n\n# Truncated for git hygiene; full HLO: {_rel(full_path)}\n", encoding="utf-8")
    return _rel(path)


def _lscpu_model() -> str:
    proc = subprocess.run(["lscpu"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _gpu_model() -> str | None:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip().splitlines()[0]
    return None


def _hardware_record(backend: str) -> dict[str, Any]:
    jax, _jnp = _jax_modules()
    return {
        "backend": backend,
        "cpu_model": _lscpu_model(),
        "gpu_model": _gpu_model(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "platform": platform.platform(),
    }


def _tree_nbytes(value: Any) -> int:
    jax, _jnp = _jax_modules()
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "size") and hasattr(leaf, "dtype"):
            total += int(leaf.size) * int(leaf.dtype.itemsize)
    return int(total)


def _cast_slots(obj: Any, dtype: Any) -> Any:
    values = {name: getattr(obj, name).astype(dtype) for name in obj.__slots__}
    return type(obj)(**values)


def _time_compiled(compiled: Any, dynamic_args: tuple[Any, ...], warmups: int, runs: int) -> tuple[Any, dict[str, float]]:
    result = compiled(*dynamic_args)
    _block_until_ready(result)
    for _ in range(int(warmups)):
        result = compiled(*dynamic_args)
        _block_until_ready(result)
    timings: list[float] = []
    for _ in range(int(runs)):
        start = time.perf_counter_ns()
        result = compiled(*dynamic_args)
        _block_until_ready(result)
        timings.append((time.perf_counter_ns() - start) / 1000.0)
    return result, {
        "kernel_wall_time_us": float(statistics.median(timings)),
        "min_wall_time_us": float(min(timings)),
        "max_wall_time_us": float(max(timings)),
        "mean_wall_time_us": float(statistics.fmean(timings)),
        "runs": float(runs),
        "warmups": float(warmups),
    }


def _profile_limitation() -> str:
    return (
        "Nsight tools are present on this workstation, but perf counters may be blocked by "
        "NVIDIA perfmon policy. Wall time is measured with cached JAX compiled calls; "
        "bandwidth is fallback-derived from estimated bytes touched; registers, occupancy, "
        "and local memory are null unless a follow-up privileged ncu pass is available."
    )


def _base_record(
    *,
    backend: str,
    kernel: str,
    precision: str,
    case: str,
    compiled: Any,
    dynamic_args: tuple[Any, ...],
    result: Any,
    timings: dict[str, float],
    artifact_dir: Path,
    bytes_touched: int,
    extra_artifacts: list[str],
) -> dict[str, Any]:
    hlo_text = _compiled_text(compiled)
    hlo_path = _write_hlo(artifact_dir, backend, kernel, precision, hlo_text)
    wall_us = float(timings["kernel_wall_time_us"])
    bandwidth = (float(bytes_touched) / (wall_us * 1.0e-6) / 1.0e9) if wall_us > 0.0 else 0.0
    return {
        "achieved_bandwidth_gb_per_s": bandwidth,
        "achieved_bandwidth_method": "fallback-derived from input/output/state bytes and median wall time",
        "artifact_paths": [hlo_path, *extra_artifacts],
        "backend": "jax",
        "benchmark": kernel,
        "case": case,
        "cpu_hardware": _lscpu_model(),
        "device_backend": backend,
        "gpu_hardware": _gpu_model(),
        "host_device_transfer_bytes": 0,
        "host_to_device_bytes_post_init": 0,
        "device_to_host_bytes_post_init": 0,
        "kernel_launches": _kernel_launches(hlo_text),
        "kernel_launches_per_step": _kernel_launches(hlo_text),
        "kernel_wall_time_us": wall_us,
        "local_memory_bytes": None,
        "max_wall_time_us": float(timings["max_wall_time_us"]),
        "mean_wall_time_us": float(timings["mean_wall_time_us"]),
        "min_wall_time_us": float(timings["min_wall_time_us"]),
        "occupancy_pct": None,
        "precision": precision,
        "profiler_limitation": _profile_limitation(),
        "registers_per_thread": None,
        "result_bytes": _tree_nbytes(result),
        "status": "ok",
        "warmup_pattern": f"one compile call, {int(timings['warmups'])} warmup calls, median of {int(timings['runs'])} cached block_until_ready calls",
        "workload_bytes_touched_estimate": int(bytes_touched),
    }


def run_m2_column(precision: str, backend: str, artifact_dir: Path, warmups: int, runs: int) -> dict[str, Any]:
    import numpy as np

    _jax, jnp = _jax_modules()
    from gpuwrf.backends.jax.column import column_thermo

    dtype = _dtype_for(precision)
    fixture = ROOT / "fixtures" / "samples" / "analytic-column-thermo-v1.npz"
    with np.load(fixture, allow_pickle=False) as loaded:
        args = (
            jnp.asarray(loaded["temperature_initial"]).astype(dtype),
            jnp.asarray(loaded["qv_initial"]).astype(dtype),
            jnp.asarray(loaded["pressure_initial"]).astype(dtype),
            jnp.asarray(loaded["saturation_qv"]).astype(dtype),
        )
    compiled = column_thermo.lower(*args).compile()
    result, timings = _time_compiled(compiled, args, warmups, runs)
    bytes_touched = _tree_nbytes(args) + _tree_nbytes(result)
    return _base_record(
        backend=backend,
        kernel="m2_column",
        precision=precision,
        case="analytic-column-thermo-v1",
        compiled=compiled,
        dynamic_args=args,
        result=result,
        timings=timings,
        artifact_dir=artifact_dir,
        bytes_touched=bytes_touched,
        extra_artifacts=[_rel(fixture)],
    )


def _make_m4_state_tendencies(dtype: Any):
    _jax, jnp = _jax_modules()
    from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
    from gpuwrf.validation.tier2 import make_ideal_grid

    grid = make_ideal_grid()
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    zero_u = jnp.zeros((nz, ny, nx + 1), dtype=dtype)
    zero_v = jnp.zeros((nz, ny + 1, nx), dtype=dtype)
    zero_w = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    zero_m = jnp.zeros((nz, ny, nx), dtype=dtype)
    zero_ph = jnp.zeros((nz + 1, ny, nx), dtype=dtype)
    zero_mu = jnp.zeros((ny, nx), dtype=dtype)
    tendencies = Tendencies(zero_u, zero_v, zero_w, zero_m, zero_m, zero_m, zero_ph, zero_mu)

    z = jnp.arange(nz, dtype=dtype)[:, None, None]
    y = jnp.arange(ny, dtype=dtype)[None, :, None]
    x = jnp.arange(nx, dtype=dtype)[None, None, :]
    x0 = _scalar(dtype, 0.30) * _scalar(dtype, float(nx))
    z0 = _scalar(dtype, 0.50) * _scalar(dtype, float(nz - 1))
    radius_x = _scalar(dtype, 0.10) * _scalar(dtype, float(nx))
    radius_z = _scalar(dtype, 0.18) * _scalar(dtype, float(nz))
    bump = jnp.exp(-(((x - x0) / radius_x) ** 2 + ((z - z0) / radius_z) ** 2)) + y * _scalar(dtype, 0.0)
    theta = _scalar(dtype, 300.0) + _scalar(dtype, 8.0) * bump
    qv = theta * _scalar(dtype, 0.0) + _scalar(dtype, 1.0e-3)
    p = theta * _scalar(dtype, 0.0)
    mu = jnp.ones((ny, nx), dtype=dtype)
    u = jnp.ones((nz, ny, nx + 1), dtype=dtype) * _scalar(dtype, 5.0)
    arrays = {
        field: jnp.zeros(shape, dtype=dtype)
        for field, shape in _state_field_shapes(grid).items()
    }
    arrays.update(
        {
            "u": u,
            "v": zero_v,
            "w": zero_w,
            "theta": theta,
            "qv": qv,
            "p": p,
            "p_total": p,
            "p_perturbation": p,
            "ph": zero_ph,
            "ph_total": zero_ph,
            "ph_perturbation": zero_ph,
            "mu": mu,
            "mu_total": mu,
            "mu_perturbation": mu,
        }
    )
    state = State(**arrays)
    return grid, state, tendencies


def run_m4_dycore(precision: str, backend: str, artifact_dir: Path, warmups: int, runs: int) -> dict[str, Any]:
    dtype = _dtype_for(precision)
    from gpuwrf.dynamics.step import step

    if precision == "bf16":
        record = {
            "artifact_paths": [],
            "backend": "jax",
            "benchmark": "m4_dycore",
            "case": "m4-dycore-40x80x80-single-step",
            "device_backend": backend,
            "kernel_wall_time_us": None,
            "precision": precision,
            "rationale": "not applicable under ADR-007 candidate policy: M4 dycore includes pressure/acoustic-adjacent and mass-continuity-critical paths that remain FP64-locked; benchmark intentionally not run as a production candidate",
            "status": "not_applicable",
        }
        return record

    grid, state, tendencies = _make_m4_state_tendencies(dtype)
    compiled = step.lower(state, tendencies, grid, 2.0, n_acoustic=4, debug=False).compile()
    result, timings = _time_compiled(compiled, (state, tendencies), warmups, runs)
    bytes_touched = _tree_nbytes((state, tendencies, result))
    return _base_record(
        backend=backend,
        kernel="m4_dycore",
        precision=precision,
        case="m4-dycore-40x80x80-single-step",
        compiled=compiled,
        dynamic_args=(state, tendencies),
        result=result,
        timings=timings,
        artifact_dir=artifact_dir,
        bytes_touched=bytes_touched,
        extra_artifacts=["artifacts/m4/tier2_invariants.json", "artifacts/m4/dycore_profile.json"],
    )


def run_m5_thompson(precision: str, backend: str, artifact_dir: Path, warmups: int, runs: int) -> dict[str, Any]:
    from gpuwrf.physics.thompson_column import step_thompson_column
    from gpuwrf.validation.tier1_thompson import SAMPLE, load_fixture_state

    dtype = _dtype_for(precision)
    state, dt, _expected = load_fixture_state()
    state = _cast_slots(state, dtype)
    compiled = step_thompson_column.lower(state, dt, debug=False).compile()
    result, timings = _time_compiled(compiled, (state,), warmups, runs)
    bytes_touched = _tree_nbytes((state, result))
    return _base_record(
        backend=backend,
        kernel="m5_thompson",
        precision=precision,
        case="analytic-thompson-column-v1",
        compiled=compiled,
        dynamic_args=(state,),
        result=result,
        timings=timings,
        artifact_dir=artifact_dir,
        bytes_touched=bytes_touched,
        extra_artifacts=[_rel(SAMPLE), "artifacts/m5/tier1_thompson_parity.json", "artifacts/m5/tier2_thompson_invariants.json"],
    )


def run_one(kernel: str, precision: str, backend: str, artifact_dir: Path, warmups: int, runs: int) -> dict[str, Any]:
    if kernel == "m2_column":
        return run_m2_column(precision, backend, artifact_dir, warmups, runs)
    if kernel == "m4_dycore":
        return run_m4_dycore(precision, backend, artifact_dir, warmups, runs)
    if kernel == "m5_thompson":
        return run_m5_thompson(precision, backend, artifact_dir, warmups, runs)
    raise ValueError(f"unknown kernel {kernel!r}")


def _record_failure(path: Path, kernel: str, precision: str, backend: str, exc: BaseException) -> dict[str, Any]:
    record = {
        "backend": "jax",
        "benchmark": kernel,
        "device_backend": backend,
        "error": f"{type(exc).__name__}: {exc}",
        "kernel_wall_time_us": None,
        "precision": precision,
        "status": "failed",
    }
    _write_json(path, record)
    return record


def _run_child_cpu(kernel: str, artifact_dir: Path, warmups: int, runs: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cpu-worker",
        "--run-one",
        kernel,
        "--precision",
        CPU_BASELINE_PRECISION,
        "--artifact-dir",
        str(artifact_dir),
        "--warmups",
        str(warmups),
        "--runs",
        str(runs),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    worker_log = artifact_dir / f"cpu-{kernel}-worker.log"
    worker_log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    path = artifact_dir / f"cpu-{kernel}-{CPU_BASELINE_PRECISION}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_paths": [_rel(worker_log)],
        "backend": "jax",
        "benchmark": kernel,
        "device_backend": "cpu",
        "error": f"cpu worker exited {proc.returncode}",
        "kernel_wall_time_us": None,
        "precision": CPU_BASELINE_PRECISION,
        "status": "failed",
    }
    _write_json(path, record)
    return record


def _write_profiler_probe(artifact_dir: Path, timeout_s: int = 90) -> str:
    path = artifact_dir / "profiler-probe.json"
    ncu = shutil.which("ncu")
    nsys = shutil.which("nsys")
    probe: dict[str, Any] = {
        "ncu_path": ncu,
        "nsys_path": nsys,
        "status": "not_run",
    }
    if ncu:
        cmd = [
            ncu,
            "--target-processes",
            "all",
            "--launch-count",
            "1",
            sys.executable,
            "-c",
            "import jax, jax.numpy as jnp; x=jnp.ones((1024,), dtype=jnp.float32); (x+x).block_until_ready()",
        ]
        try:
            proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False)
            probe.update(
                {
                    "command": cmd,
                    "returncode": proc.returncode,
                    "status": "ok" if proc.returncode == 0 else "failed_or_restricted",
                    "stdout_tail": proc.stdout[-4000:],
                    "stderr_tail": proc.stderr[-4000:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            probe.update({"command": cmd, "status": "timeout", "error": str(exc)})
    _write_json(path, probe)
    return _rel(path)


def _write_gen2_probe(artifact_dir: Path) -> str:
    wrfinput = []
    if GEN2_WRF.exists():
        wrfinput = [str(path) for path in GEN2_WRF.rglob("wrfinput*")][:20]
    cmd = f"source {GEN2_ENV} >/tmp/precision_bench_gen2_env.out 2>/tmp/precision_bench_gen2_env.err; env | sort | grep -E '^(WRF|NETCDF|HDF5|PATH)=' | head -50"
    proc = subprocess.run(["bash", "-lc", cmd], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    record = {
        "env_path": str(GEN2_ENV),
        "env_exists": GEN2_ENV.exists(),
        "env_probe_returncode": proc.returncode,
        "env_probe_stdout": proc.stdout,
        "env_probe_stderr": proc.stderr,
        "wrf_exe": str(GEN2_WRF / "main" / "wrf.exe"),
        "wrf_exe_exists": (GEN2_WRF / "main" / "wrf.exe").exists(),
        "wrfinput_candidates": wrfinput,
        "status": "unavailable" if not wrfinput else "inputs_found_not_run",
        "note": "Exact Gen2 WRF Canary 3km CPU timestep baseline was probed but not run here because no wrfinput/wrfbdy files are present under the Gen2 WRF tree. Sprint-local CPU JAX baselines are emitted separately per workload.",
    }
    path = artifact_dir / "cpu-gen2-probe.json"
    _write_json(path, record)
    return _rel(path)


def _write_speedup_summary(artifact_dir: Path, records: list[dict[str, Any]], profiler_probe: str, gen2_probe: str) -> None:
    cpu_by_kernel = {
        record["benchmark"]: record
        for record in records
        if record.get("device_backend") == "cpu" and record.get("precision") == CPU_BASELINE_PRECISION and record.get("status") == "ok"
    }
    rows = []
    for kernel in KERNELS:
        row: dict[str, Any] = {"kernel": kernel, "cpu_fp64_artifact": None, "cells": {}}
        cpu = cpu_by_kernel.get(kernel)
        if cpu:
            row["cpu_fp64_artifact"] = f"artifacts/precision-bench/cpu-{kernel}-fp64.json"
            row["cpu_wall_time_us"] = cpu.get("kernel_wall_time_us")
        for precision in GPU_PRECISIONS[kernel]:
            candidates = [
                record
                for record in records
                if record.get("device_backend") == "gpu"
                and record.get("benchmark") == kernel
                and record.get("precision") == precision
            ]
            if not candidates:
                continue
            record = candidates[0]
            wall = record.get("kernel_wall_time_us")
            speedup = None
            if cpu and isinstance(wall, (float, int)) and wall:
                speedup = float(cpu["kernel_wall_time_us"]) / float(wall)
            row["cells"][precision] = {
                "artifact": f"artifacts/precision-bench/{kernel}-{precision}.json",
                "gpu_wall_time_us": wall,
                "speedup_vs_cpu_fp64": speedup,
                "status": record.get("status"),
            }
        rows.append(row)
    summary = {
        "benchmark": "adr007_precision_speedup_summary",
        "cpu_denominator": "Sprint-local JAX CPU FP64 workload timing on the host CPU; exact Gen2 WRF 3km timestep probe is separate.",
        "gen2_probe_artifact": gen2_probe,
        "profiler_probe_artifact": profiler_probe,
        "rows": rows,
    }
    _write_json(artifact_dir / "projected-speedups.json", summary)


def run_all(artifact_dir: Path, warmups: int, runs: int, skip_profiler_probe: bool) -> int:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    profiler_probe = str(artifact_dir / "profiler-probe.json")
    if not skip_profiler_probe:
        profiler_probe = _write_profiler_probe(artifact_dir)
    gen2_probe = _write_gen2_probe(artifact_dir)

    records: list[dict[str, Any]] = []
    for kernel in KERNELS:
        cpu_record = _run_child_cpu(kernel, artifact_dir, warmups=max(1, min(warmups, 5)), runs=runs)
        records.append(cpu_record)

    for kernel in KERNELS:
        for precision in GPU_PRECISIONS[kernel]:
            path = artifact_dir / f"{kernel}-{precision}.json"
            try:
                record = run_one(kernel, precision, "gpu", artifact_dir, warmups, runs)
                if profiler_probe:
                    record["artifact_paths"].append(profiler_probe)
                if gen2_probe:
                    record["artifact_paths"].append(gen2_probe)
                _write_json(path, record)
            except Exception as exc:
                record = _record_failure(path, kernel, precision, "gpu", exc)
            records.append(record)

    _write_speedup_summary(artifact_dir, records, profiler_probe, gen2_probe)
    required_failures = [
        record
        for record in records
        if record.get("device_backend") == "gpu"
        and record.get("precision") in {"fp64", "fp32"}
        and record.get("status") != "ok"
    ]
    return 1 if required_failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ADR-007 precision benchmark runner.")
    parser.add_argument("--run-all", action="store_true", help="Run CPU baseline plus all GPU precision cells.")
    parser.add_argument("--run-one", choices=KERNELS, help="Run one kernel/precision cell.")
    parser.add_argument("--precision", choices=("fp64", "fp32", "bf16"), default="fp64")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--skip-profiler-probe", action="store_true")
    parser.add_argument("--cpu-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true", help="Print configuration without importing JAX kernels.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        print(json.dumps({"ok": True, "kernels": KERNELS, "gpu_precisions": GPU_PRECISIONS}, indent=2, sort_keys=True))
        return 0
    if args.run_all:
        return run_all(args.artifact_dir, args.warmups, args.runs, args.skip_profiler_probe)
    if args.run_one:
        backend = "cpu" if args.cpu_worker or os.environ.get("JAX_PLATFORMS") == "cpu" else "gpu"
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            record = run_one(args.run_one, args.precision, backend, args.artifact_dir, args.warmups, args.runs)
        except Exception as exc:
            record = _record_failure(args.artifact_dir / f"{backend}-{args.run_one}-{args.precision}.json", args.run_one, args.precision, backend, exc)
            print(json.dumps(record, indent=2, sort_keys=True))
            return 1
        path = args.artifact_dir / f"{backend}-{args.run_one}-{args.precision}.json"
        if backend == "gpu":
            path = args.artifact_dir / f"{args.run_one}-{args.precision}.json"
        _write_json(path, record)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0 if record.get("status") in {"ok", "not_applicable"} else 1
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
