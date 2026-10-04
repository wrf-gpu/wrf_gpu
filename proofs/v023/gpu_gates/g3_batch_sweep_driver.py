#!/usr/bin/env python3
"""F1 G3 GPU throughput driver for homogeneous two-domain batch ensembles.

This proof driver is intentionally separate from the v0.21 canary timing
driver: F1 needs batched carries and per-lane namelists loaded through
``_load_batched_domains`` before ``DomainTree`` sees ``GPUWRF_BATCH_ENSEMBLE``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("GPUWRF_NESTED_AOT", "0")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.integration.nested_pipeline import (  # noqa: E402
    NestedPipelineConfig,
    batch_ensemble_size_from_env,
    _load_batched_domains,
    _load_domains,
    _nested_sync_mode_from_env,
    _output_cadence_steps_by_domain,
    domain_names_for,
    run_batched_operational_domain_tree,
)
from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: E402
from gpuwrf.profiling.transfer_audit import visible_gpu_name  # noqa: E402
from gpuwrf.runtime.domain_tree import (  # noqa: E402
    DomainTree,
    run_operational_domain_tree,
)


_CUDART = None


def _cuda_profiler_start() -> bool:
    global _CUDART
    try:
        if _CUDART is None:
            _CUDART = ctypes.CDLL("libcudart.so")
        return int(_CUDART.cudaProfilerStart()) == 0
    except Exception:
        return False


def _cuda_profiler_stop() -> bool:
    try:
        return _CUDART is not None and int(_CUDART.cudaProfilerStop()) == 0
    except Exception:
        return False


def _gpu_mem_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return -1


def _gpu_util_pct() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return -1


class _Sampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = float(interval_s)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_Sampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(
                {
                    "t": time.time(),
                    "vram_mib": _gpu_mem_mib(),
                    "gpu_util_pct": _gpu_util_pct(),
                }
            )
            self._stop.wait(self.interval_s)

    @property
    def peak_vram_mib(self) -> int:
        values = [int(s["vram_mib"]) for s in self.samples if int(s["vram_mib"]) >= 0]
        return max(values) if values else -1

    @property
    def mean_gpu_util_pct(self) -> float:
        values = [int(s["gpu_util_pct"]) for s in self.samples if int(s["gpu_util_pct"]) >= 0]
        return float(sum(values) / len(values)) if values else float("nan")


def _path_part(part: Any) -> str:
    raw = getattr(part, "name", getattr(part, "key", getattr(part, "idx", part)))
    return str(raw).replace("/", "_").replace(" ", "")


def _state_digest(states: dict[str, Any]) -> dict[str, Any]:
    h = hashlib.sha256()
    leaves = 0
    total_bytes = 0
    for domain, state in sorted(states.items()):
        path_leaves, _treedef = jax.tree_util.tree_flatten_with_path(state)
        for idx, (path, leaf) in enumerate(path_leaves):
            if not hasattr(leaf, "dtype"):
                continue
            arr = np.ascontiguousarray(np.asarray(jax.device_get(leaf)))
            suffix = ".".join(_path_part(part) for part in path) or f"leaf{idx}"
            key = f"{domain}/{idx:04d}/{suffix}"
            leaves += 1
            total_bytes += int(arr.nbytes)
            h.update(key.encode("utf-8"))
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(json.dumps(tuple(int(v) for v in arr.shape)).encode("utf-8"))
            h.update(arr.tobytes(order="C"))
    return {"sha256": h.hexdigest(), "leaves": leaves, "bytes": total_bytes}


def _steps_per_root(hierarchy: Any, names: tuple[str, ...]) -> dict[str, int]:
    steps = {names[0]: 1}
    pending = list(hierarchy.nests)
    while pending:
        progressed = False
        rest = []
        for edge in pending:
            parent = str(edge.parent)
            child = str(edge.child)
            if parent in steps:
                steps[child] = int(steps[parent]) * int(edge.parent_grid_ratio)
                progressed = True
            else:
                rest.append(edge)
        if not progressed:
            raise ValueError("could not resolve domain step ratios from hierarchy")
        pending = rest
    return steps


def _cell_metrics(
    *,
    hierarchy: Any,
    bundles: dict[str, Any],
    names: tuple[str, ...],
    batch_size: int,
    s_per_root_step: float,
) -> dict[str, Any]:
    step_mult = _steps_per_root(hierarchy, names)
    per_domain = {}
    per_lane_cells_per_root = 0
    for name in names:
        grid = bundles[name].grid
        mass_cells = int(grid.nx) * int(grid.ny) * int(grid.nz)
        advanced = int(step_mult[name]) * mass_cells
        per_lane_cells_per_root += advanced
        per_domain[name] = {
            "nx": int(grid.nx),
            "ny": int(grid.ny),
            "nz": int(grid.nz),
            "mass_cells": mass_cells,
            "steps_per_root_step": int(step_mult[name]),
            "advanced_cells_per_root_step_per_lane": advanced,
        }
    total_cells = int(batch_size) * int(per_lane_cells_per_root)
    return {
        "per_domain": per_domain,
        "per_lane_advanced_cells_per_root_step": int(per_lane_cells_per_root),
        "batch_advanced_cells_per_root_step": int(total_cells),
        "warm_cells_per_sec": float(total_cells / s_per_root_step),
    }


def _load_tree(cfg: NestedPipelineConfig, names: tuple[str, ...], batch_size: int):
    if int(batch_size) > 1:
        (
            hierarchy,
            bundles,
            meta,
            run_start,
            dt_by_domain,
            initial_carries,
            batch_namelists,
            batch_input_dirs,
            batch_run_starts,
        ) = _load_batched_domains(cfg, names, batch_size=int(batch_size))
    else:
        hierarchy, bundles, meta, run_start, dt_by_domain, initial_carries = _load_domains(cfg, names)
        batch_namelists = {}
        batch_input_dirs = (Path(cfg.input_dir),)
        batch_run_starts = (run_start,)
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=bool(cfg.feedback))
    return (
        hierarchy,
        bundles,
        meta,
        run_start,
        dt_by_domain,
        initial_carries,
        batch_namelists,
        batch_input_dirs,
        batch_run_starts,
        tree,
    )


def main() -> int:
    batch_size = batch_ensemble_size_from_env()
    max_dom = int(os.environ.get("G3_MAXDOM", "2"))
    input_dir = Path(os.environ["G3_INPUT_DIR"]).resolve()
    out_path = Path(os.environ.get("G3_OUTPUT", f"proofs/v023/gpu_gates/g3_B{batch_size}.json"))
    root_steps = int(os.environ.get("G3_ROOT_STEPS", "200"))
    repeats = int(os.environ.get("G3_REPEATS", "2"))
    if root_steps <= 0:
        raise ValueError("G3_ROOT_STEPS must be positive")
    if repeats < 2:
        raise ValueError("G3_REPEATS must be at least 2")

    names = domain_names_for(max_dom)
    cfg = NestedPipelineConfig(
        input_dir=input_dir,
        output_dir=out_path.parent / f"g3_B{batch_size}_output",
        proof_dir=out_path.parent / f"g3_B{batch_size}_proof",
        scratch_dir=out_path.parent / f"g3_B{batch_size}_scratch",
        hours=1,
        max_dom=max_dom,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.proof_dir.mkdir(parents=True, exist_ok=True)
    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)

    (
        hierarchy,
        bundles,
        meta,
        run_start,
        dt_by_domain,
        initial_carries,
        batch_namelists,
        batch_input_dirs,
        batch_run_starts,
        tree,
    ) = _load_tree(cfg, names, batch_size)

    cadence_run = Gen2Run(input_dir)
    output_cadence, _hist_min = _output_cadence_steps_by_domain(cadence_run, names, dt_by_domain)
    block_between, root_sync_cadence = _nested_sync_mode_from_env()
    own_steps0 = {name: 0 for name in names}

    def one_call(label: str):
        t0 = time.perf_counter()
        run_tree = (
            run_batched_operational_domain_tree
            if int(batch_size) > 1
            else run_operational_domain_tree
        )
        run_kwargs = {}
        if int(batch_size) > 1:
            run_kwargs["batch_namelists"] = batch_namelists
            run_kwargs["batch_size"] = int(batch_size)
        result = run_tree(
            tree,
            root_steps=root_steps,
            feedback_enabled=bool(cfg.feedback),
            output=None,
            output_cadence_steps=output_cadence,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            carries=initial_carries,
            initial_own_steps=own_steps0,
            **run_kwargs,
        )
        jax.block_until_ready(tuple(result.states[name].theta for name in names))
        wall_s = time.perf_counter() - t0
        finite = all(bool(np.isfinite(np.asarray(jax.device_get(result.states[name].theta))).all()) for name in names)
        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        vram_mib = _gpu_mem_mib()
        print(
            f"CALL label={label} B={batch_size} root_steps={root_steps} "
            f"wall_s={wall_s:.4f} finite={finite} vram_mib={vram_mib} rss_peak_mib={rss_mib:.0f}",
            flush=True,
        )
        return wall_s, result, finite, vram_mib, rss_mib

    print(
        f"MARKER:G3_START B={batch_size} maxdom={max_dom} root_steps={root_steps} repeats={repeats} "
        f"input={input_dir} backend={jax.default_backend()} device={visible_gpu_name()}",
        flush=True,
    )
    with _Sampler(interval_s=float(os.environ.get("G3_SAMPLE_INTERVAL_S", "0.5"))) as sampler:
        calls = [one_call("WARMUP")]
        for idx in range(repeats - 1):
            if os.environ.get("G3_CUDA_PROFILER_RANGE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                print("MARKER:CUDA_PROFILER_START", flush=True)
                _cuda_profiler_start()
                try:
                    calls.append(one_call(f"MEASURE_{idx + 1}"))
                finally:
                    _cuda_profiler_stop()
                    print("MARKER:CUDA_PROFILER_STOP", flush=True)
            else:
                calls.append(one_call(f"MEASURE_{idx + 1}"))

    measured = calls[1:]
    measured_wall_s = [float(item[0]) for item in measured]
    measured_results = [item[1] for item in measured]
    s_per_root_step = float(sum(measured_wall_s) / (len(measured_wall_s) * root_steps))
    root = names[0]
    root_dt_s = float(dt_by_domain[root])
    s_per_fc_hour = float(s_per_root_step * (3600.0 / root_dt_s))
    cases_per_gpu_hour = float(batch_size * 3600.0 / s_per_fc_hour)
    cells = _cell_metrics(
        hierarchy=hierarchy,
        bundles=bundles,
        names=names,
        batch_size=batch_size,
        s_per_root_step=s_per_root_step,
    )
    digest = _state_digest(measured_results[-1].states)

    payload = {
        "schema": "v023.f1_g3_batch_sweep",
        "schema_version": 1,
        "verdict": "PASS" if all(bool(item[2]) for item in calls) else "FAIL",
        "batch_size": int(batch_size),
        "backend": jax.default_backend(),
        "device": visible_gpu_name(),
        "input_dir": str(input_dir),
        "batch_input_dirs": [str(path.resolve()) for path in batch_input_dirs],
        "batch_run_starts_utc": [dt.isoformat() for dt in batch_run_starts],
        "run_start_utc": run_start.isoformat(),
        "max_dom": int(max_dom),
        "domains": list(names),
        "root_steps": int(root_steps),
        "root_dt_s": root_dt_s,
        "forecast_hours_per_call": float(root_steps * root_dt_s / 3600.0),
        "repeats": int(repeats),
        "warmup_wall_s": float(calls[0][0]),
        "measured_wall_s": measured_wall_s,
        "s_per_root_step": s_per_root_step,
        "s_per_forecast_hour_per_lane": s_per_fc_hour,
        "cases_per_gpu_hour": cases_per_gpu_hour,
        "peak_vram_mib": int(max([sampler.peak_vram_mib, *(int(item[3]) for item in calls)])),
        "peak_host_rss_mib": float(max(float(item[4]) for item in calls)),
        "mean_gpu_util_pct": sampler.mean_gpu_util_pct,
        "sampler_count": len(sampler.samples),
        "samples": sampler.samples,
        "all_finite": all(bool(item[2]) for item in calls),
        "cell_metrics": cells,
        "state_digest": digest,
        "sync_mode": {
            "block_between": bool(block_between),
            "root_sync_cadence": root_sync_cadence,
        },
        "metadata": meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"SUMMARY B={batch_size} verdict={payload['verdict']} s_per_step={s_per_root_step:.6f} "
        f"cells_per_sec={cells['warm_cells_per_sec']:.3f} cases_per_gpu_hour={cases_per_gpu_hour:.6f} "
        f"peak_vram_mib={payload['peak_vram_mib']} rss_peak_mib={payload['peak_host_rss_mib']:.0f} "
        f"proof={out_path}",
        flush=True,
    )
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
