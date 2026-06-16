"""v0.17 fp32-physics A/B proof harness.

Measures the opt-in ``GPUWRF_FP32_PHYSICS=1`` island against the default fp64
physics path in separate subprocesses so JAX compile caches cannot hide the env
toggle.  The JSON shape mirrors the Sprint-0 harness fields requested for v0.17:
cold/warm ms/step, peak VRAM, CPU/oracle comparison, and measured-vs-projected.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "proofs" / "perf" / "v017"
OUT_JSON = OUT_DIR / "fp32_physics_bench.json"

WS128_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
WS128_RUN_ID = "run_h36"
CANARY_ROOT = Path("/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_allgreen")
CANARY_RUN_ID = "gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _child_env(fp32_physics: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "TF_GPU_ALLOCATOR": "cuda_malloc_async",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "4"),
            "GPUWRF_FP32_PHYSICS": "1" if fp32_physics else "0",
        }
    )
    return env


def _run_child(kind: str, fp32_physics: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--kind",
        kind,
        "--fp32-physics",
        "1" if fp32_physics else "0",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=_child_env(fp32_physics),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    payload: dict[str, Any]
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {
            "status": "failed",
            "error": "child did not emit JSON",
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    payload["returncode"] = int(proc.returncode)
    payload["stderr_tail"] = proc.stderr[-4000:]
    payload["stdout_tail"] = proc.stdout[-4000:]
    if proc.returncode != 0 and payload.get("status") == "ok":
        payload["status"] = "failed"
    return payload


def _import_jax_stack():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _block(tree) -> None:
    import jax

    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _peak_gib() -> float | None:
    import jax

    try:
        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        return float(stats.get("peak_bytes_in_use", 0)) / (1024.0**3)
    except Exception:
        return None


def _reset_peak() -> bool:
    import jax

    try:
        dev = jax.devices()[0]
        if hasattr(dev, "reset_memory_stats"):
            dev.reset_memory_stats()
            return True
    except Exception:
        pass
    return False


def _all_finite(tree) -> bool:
    import jax
    import numpy as np

    for leaf in jax.tree_util.tree_leaves(tree):
        arr = np.asarray(leaf)
        if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
            return False
    return True


def _state_summary(state) -> dict[str, Any]:
    import jax
    import numpy as np

    floating = []
    max_abs = 0.0
    for leaf in jax.tree_util.tree_leaves(state):
        arr = np.asarray(leaf)
        if np.issubdtype(arr.dtype, np.floating):
            floating.append(str(arr.dtype))
            max_abs = max(max_abs, float(np.nanmax(np.abs(arr))) if arr.size else 0.0)
    return {
        "all_finite": _all_finite(state),
        "floating_dtypes": sorted(set(floating)),
        "max_abs_floating_leaf": max_abs,
    }


def _timed_call(fn, *args, **kwargs) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    _block(out)
    return (time.perf_counter() - start), out


def _measure_case(case_name: str, run_root: Path, run_id: str, domain: str) -> dict[str, Any]:
    import jax

    from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
    from gpuwrf.runtime.operational_mode import run_forecast_operational

    cfg = DailyPipelineConfig(run_id=run_id, run_root=run_root, domain=domain, hours=1)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist,
        force_fp64=True,
        run_boundary=False,
        use_noahmp=False,
        gwd_opt=0,
        # Make the radiation/MYNN lever a real per-step fraction in this short
        # benchmark window instead of hiding radiation behind a long cadence.
        radiation_cadence_steps=1,
    )
    dt_s = float(nl.dt_s)
    grid = case.grid

    def fresh_state():
        fresh_case, _ = _build_real_case(cfg)
        return fresh_case.state

    one_step_h = dt_s / 3600.0
    warm_a_steps = 2
    warm_b_steps = 5
    h_a = warm_a_steps * dt_s / 3600.0
    h_b = warm_b_steps * dt_s / 3600.0

    _reset_peak()
    cold_s, cold_out = _timed_call(run_forecast_operational, fresh_state(), nl, one_step_h)
    cold_peak = _peak_gib()
    _timed_call(run_forecast_operational, fresh_state(), nl, h_a)
    warm_a_s, warm_a_out = _timed_call(run_forecast_operational, fresh_state(), nl, h_a)
    _timed_call(run_forecast_operational, fresh_state(), nl, h_b)
    warm_b_s, warm_b_out = _timed_call(run_forecast_operational, fresh_state(), nl, h_b)
    warm_peak = _peak_gib()

    warm_ms = (warm_b_s - warm_a_s) * 1000.0 / float(warm_b_steps - warm_a_steps)
    return {
        "status": "ok",
        "case": case_name,
        "run_dir": str(run_dir),
        "domain": domain,
        "device": str(jax.devices()[0]),
        "grid": {"ny": int(grid.ny), "nx": int(grid.nx), "nz": int(grid.nz), "ncol": int(grid.ny) * int(grid.nx)},
        "dt_s": dt_s,
        "timing_method": {
            "cold": "compile+run one operational step",
            "warm": f"marginal ({warm_b_steps} steps - {warm_a_steps} steps) / {warm_b_steps - warm_a_steps}",
            "radiation_cadence_steps": 1,
            "boundary": "off",
        },
        "ms_per_step": {
            "cold": cold_s * 1000.0,
            "warm": warm_ms,
            "warm_a_total_ms": warm_a_s * 1000.0,
            "warm_b_total_ms": warm_b_s * 1000.0,
        },
        "peak_vram_gib": max(x for x in (cold_peak, warm_peak) if x is not None) if cold_peak is not None or warm_peak is not None else None,
        "finite": {
            "cold": _state_summary(cold_out),
            "warm_a": _state_summary(warm_a_out),
            "warm_b": _state_summary(warm_b_out),
        },
    }


def _measure_dummy_sanity() -> dict[str, Any]:
    import importlib.util
    import numpy as np

    from gpuwrf.contracts.state import Tendencies

    script = ROOT / "scripts" / "m6_run_dummy_coupled.py"
    spec = importlib.util.spec_from_file_location("m6_run_dummy_coupled", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grid = module.make_dummy_grid(8, 8, 8)
    state = module.make_initial_state(grid)
    tendencies = Tendencies.zeros(grid)
    before_mu = float(np.asarray(state.mu).sum())
    before_qv = float(np.asarray(state.qv).sum())
    out = module.run_dummy_coupled(state, tendencies, grid, 1.0, 10, n_acoustic=1, debug=False)
    _block(out)
    after_mu = float(np.asarray(out.mu).sum())
    after_qv = float(np.asarray(out.qv).sum())
    return {
        "status": "ok",
        "case": "dummy_coupled_10step",
        "grid": {"ny": int(grid.ny), "nx": int(grid.nx), "nz": int(grid.nz), "ncol": int(grid.ny) * int(grid.nx)},
        "steps": 10,
        "finite": _state_summary(out),
        "conservation": {
            "mu_sum_before": before_mu,
            "mu_sum_after": after_mu,
            "mu_relative_drift": abs(after_mu - before_mu) / max(abs(before_mu), 1.0),
            "qv_sum_before": before_qv,
            "qv_sum_after": after_qv,
        },
    }


def _measure_oracles() -> dict[str, Any]:
    import jax.numpy as jnp

    from gpuwrf.validation.tier1_mynn import run_tier1 as run_mynn_tier1
    from gpuwrf.validation.tier1_rrtmg import run_tier1_lw, run_tier1_sw

    out_dir = OUT_DIR / "oracles"
    return {
        "rrtmg_sw": run_tier1_sw(out_dir / "tier1_rrtmg_sw_fp32.json", dtype=jnp.float32),
        "rrtmg_lw": run_tier1_lw(out_dir / "tier1_rrtmg_lw_fp32.json", dtype=jnp.float32),
        "mynn": run_mynn_tier1(out_dir / "tier1_mynn_fp32.json", dtype=jnp.float32),
    }


def worker(kind: str, fp32_physics: bool) -> int:
    try:
        if kind == "oracles":
            payload = {"status": "ok", "kind": kind, "fp32_physics": fp32_physics, "oracles": _measure_oracles()}
        elif kind == "dummy_sanity":
            payload = {"status": "ok", "kind": kind, "fp32_physics": fp32_physics, "sanity": _measure_dummy_sanity()}
        elif kind == "ws128":
            payload = _measure_case("ws128", WS128_ROOT, WS128_RUN_ID, "d01")
            payload["fp32_physics"] = fp32_physics
        elif kind == "canary_d02":
            payload = _measure_case("canary_d02", CANARY_ROOT, CANARY_RUN_ID, "d02")
            payload["fp32_physics"] = fp32_physics
        else:
            raise ValueError(f"unknown worker kind {kind}")
        print(json.dumps(payload, sort_keys=True, default=_json_default), flush=True)
        return 0 if payload.get("status") == "ok" else 1
    except Exception as exc:  # noqa: BLE001 - proof must report the actual failure.
        payload = {
            "status": "failed",
            "kind": kind,
            "fp32_physics": fp32_physics,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }
        print(json.dumps(payload, sort_keys=True, default=_json_default), flush=True)
        return 1


def _ab_pair(kind: str) -> dict[str, Any]:
    base = _run_child(kind, False)
    fp32 = _run_child(kind, True)
    out = {"fp64": base, "fp32_physics": fp32}
    try:
        base_warm = float(base["ms_per_step"]["warm"])
        fp32_warm = float(fp32["ms_per_step"]["warm"])
        out["measured_vs_projected"] = {
            "measured_speedup_warm": base_warm / fp32_warm if fp32_warm > 0 else None,
            "projected_speedup": None,
            "projection_source": "not projected in this harness; measured-only",
        }
    except Exception:
        out["measured_vs_projected"] = {
            "measured_speedup_warm": None,
            "projected_speedup": None,
            "projection_source": "unavailable because one side failed",
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--kind", default="all")
    parser.add_argument("--fp32-physics", default="0")
    args = parser.parse_args()
    fp32_physics = args.fp32_physics.strip() not in {"0", "false", "False", "no", "off"}
    if args.worker:
        return worker(args.kind, fp32_physics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v017_fp32_physics_bench_v1",
        "flag": "GPUWRF_FP32_PHYSICS",
        "default_path": "off/fp64",
        "bench_harness": "proofs/perf/v017/fp32_physics_bench.py",
        "notes": [
            "v0.17 bench_harness.py was absent in this v0.16.0 worktree; this mirrors the requested JSON fields.",
            "fp64 and fp32-physics variants run in separate child processes to avoid JIT cache reuse across env flag toggles.",
        ],
        "cpu_compare": {
            "policy": "per-scheme WRF oracle fixtures compare fp32 outputs to established operational tolerances",
            "oracles": _run_child("oracles", True),
        },
        "coupled_short_run_sanity": _run_child("dummy_sanity", True),
        "cases": {
            "ws128": _ab_pair("ws128"),
            "canary_d02": _ab_pair("canary_d02"),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT_JSON), "case_status": {k: {m: v[m].get("status") for m in ("fp64", "fp32_physics")} for k, v in payload["cases"].items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
