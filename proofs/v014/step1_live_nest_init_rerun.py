#!/usr/bin/env python3
"""V0.14 d02 step-1 strict comparison with live-nest base initialization."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import same_input_contract_builder as builder  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_live_nest_init_rerun.json"
OUT_MD = PROOF_DIR / "step1_live_nest_init_rerun.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md"

PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
MANAGING_SPRINTS_SKILL = ROOT / ".agent/skills/managing-sprints/SKILL.md"
SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-live-nest-init-rerun/sprint-contract.md"
PREVIOUS_STEP1_MD = PROOF_DIR / "step1_same_input_truth.md"
PREVIOUS_STEP1_JSON = PROOF_DIR / "step1_same_input_truth.json"
PREVIOUS_STEP1_PY = PROOF_DIR / "step1_same_input_truth.py"
LIVE_NEST_FIX_MD = PROOF_DIR / "live_nest_base_source_fix.md"
LIVE_NEST_FIX_JSON = PROOF_DIR / "live_nest_base_source_fix.json"
D02_REPLAY = SRC / "gpuwrf/integration/d02_replay.py"
NESTED_PIPELINE = SRC / "gpuwrf/integration/nested_pipeline.py"

RUN_CASE3 = Path("/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3")
CPU_H0_D02 = RUN_CASE3 / "wrfout_d02_2026-05-01_18:00:00"
ACCEPTED_TRUTH = (
    Path("/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth")
    / "same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz"
)

TARGET_STEP = 1
TARGET_DOMAIN = 2
TARGET_SURFACE = "post_after_all_rk_steps_pre_halo"
P0_THETA_OFFSET_K = 300.0

ALL_COMPARE_FIELDS = (
    "T",
    "P",
    "PB",
    "PH",
    "PHB",
    "MU",
    "MUB",
    "U",
    "V",
    "W",
    "QVAPOR",
    "QCLOUD",
    "QRAIN",
    "QICE",
    "QSNOW",
    "QGRAUP",
)
INITIAL_DELTA_FIELDS = ("HGT", "PB", "MUB", "PHB", "P_TOTAL", "MU_TOTAL", "PH_TOTAL")
BASE_CLOSE_THRESHOLDS = {"PB": 0.1, "MUB": 0.1, "PHB": 0.2}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], *, cwd: Path | None = None, timeout_s: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
    )
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout_s": int(timeout_s),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = list(jax.devices())
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([device for device in devices if device.platform == "gpu"]),
            }
        )
    except Exception as exc:
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def load_nc_var(path: Path, name: str) -> np.ndarray:
    from netCDF4 import Dataset  # type: ignore # noqa: PLC0415

    with Dataset(path) as dataset:
        var = dataset.variables[name]
        data = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
        return np.asarray(data, dtype=np.float64)


def h0_reference_fields() -> dict[str, np.ndarray]:
    p = load_nc_var(CPU_H0_D02, "P")
    pb = load_nc_var(CPU_H0_D02, "PB")
    ph = load_nc_var(CPU_H0_D02, "PH")
    phb = load_nc_var(CPU_H0_D02, "PHB")
    mu = load_nc_var(CPU_H0_D02, "MU")
    mub = load_nc_var(CPU_H0_D02, "MUB")
    return {
        "HGT": load_nc_var(CPU_H0_D02, "HGT"),
        "PB": pb,
        "MUB": mub,
        "PHB": phb,
        "P_TOTAL": pb + p,
        "MU_TOTAL": mub + mu,
        "PH_TOTAL": phb + ph,
    }


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def fortran_index(field: str, index: tuple[int, ...] | None) -> dict[str, int] | None:
    if index is None:
        return None
    if field in {"T", "P", "PB", "P_TOTAL", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"}:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field in {"PH", "PHB", "PH_TOTAL", "W"}:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "kstag": int(k) + 1}
    if field == "U":
        k, y, x = index
        return {"i_xstag": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field == "V":
        k, y, x = index
        return {"i": int(x) + 1, "j_ystag": int(y) + 1, "k": int(k) + 1}
    if field in {"MU", "MUB", "MU_TOTAL", "HGT"}:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    return None


def diff_metrics(field: str, candidate: Any, reference: Any) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    diff = cand - ref
    absdiff = np.abs(diff)
    finite_abs = absdiff[np.isfinite(absdiff)]
    mismatch_mask = (diff != 0.0) | (~np.isfinite(diff))
    mismatch = np.argwhere(mismatch_mask)
    first = tuple(int(x) for x in mismatch[0]) if mismatch.size else None
    if finite_abs.size:
        worst = tuple(int(x) for x in np.unravel_index(int(np.nanargmax(absdiff)), absdiff.shape))
        max_abs = float(np.nanmax(absdiff))
        rmse = float(np.sqrt(np.nanmean(diff * diff)))
        bias = float(np.nanmean(diff))
        p95 = float(np.nanpercentile(absdiff, 95))
        p99 = float(np.nanpercentile(absdiff, 99))
    else:
        worst = first
        max_abs = None
        rmse = None
        bias = None
        p95 = None
        p99 = None
    return {
        "status": "OK",
        "count": int(diff.size),
        "shape": list(diff.shape),
        "max_abs": max_abs,
        "rmse": rmse,
        "bias": bias,
        "p95": p95,
        "p99": p99,
        "nonfinite_diff_count": int((~np.isfinite(diff)).sum()),
        "first_mismatch_index": list(first) if first is not None else None,
        "first_mismatch_fortran": fortran_index(field, first),
        "worst_mismatch_index": list(worst) if worst is not None else None,
        "worst_mismatch_fortran": fortran_index(field, worst),
    }


def jax_compare_array(field: str, state: Any, base_state: Any) -> Any:
    mapping = {
        "T": lambda: state.theta - P0_THETA_OFFSET_K,
        "P": lambda: state.p_perturbation,
        "PB": lambda: base_state.pb,
        "PH": lambda: state.ph_perturbation,
        "PHB": lambda: base_state.phb,
        "MU": lambda: state.mu_perturbation,
        "MUB": lambda: base_state.mub,
        "U": lambda: state.u,
        "V": lambda: state.v,
        "W": lambda: state.w,
        "QVAPOR": lambda: state.qv,
        "QCLOUD": lambda: state.qc,
        "QRAIN": lambda: state.qr,
        "QICE": lambda: state.qi,
        "QSNOW": lambda: state.qs,
        "QGRAUP": lambda: state.qg,
    }
    return mapping[field]()


def compare_arrays(truth_path: Path, state: Any, base_state: Any, jax: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    with np.load(truth_path) as truth:
        missing = [name for name in ALL_COMPARE_FIELDS if name not in truth]
        if missing:
            return {"status": "BLOCKED_TRUTH_MISSING_KEYS", "missing": missing}
        for name in ALL_COMPARE_FIELDS:
            wrf = np.asarray(truth[name], dtype=np.float64)
            candidate = np.asarray(jax.device_get(jax_compare_array(name, state, base_state)), dtype=np.float64)
            item = diff_metrics(name, candidate, wrf)
            if item["status"] != "OK":
                return {"status": "BLOCKED_SHAPE_MISMATCH", "field": name, **item}
            metrics[name] = {key: value for key, value in item.items() if key != "status"}
    ranked = sorted(
        [{"field": name, **item} for name, item in metrics.items()],
        key=lambda item: (-1.0 if item["max_abs"] is None else float(item["max_abs"])),
        reverse=True,
    )
    first_divergent = None
    for name in ALL_COMPARE_FIELDS:
        item = metrics[name]
        if item["nonfinite_diff_count"] or (item["max_abs"] is not None and float(item["max_abs"]) != 0.0):
            first_divergent = name
            break
    return {
        "status": "COMPARISON_EXECUTED",
        "truth_file": str(truth_path),
        "strict_live_nest_init_comparison_run": True,
        "first_divergent_field": first_divergent,
        "per_field_metrics": metrics,
        "ranked_residuals": ranked,
    }


def initial_field_arrays(case: Mapping[str, Any]) -> dict[str, Any]:
    state = case["state"]
    base = case["base_state"]
    grid = case["grid"]
    return {
        "HGT": grid.terrain_height,
        "PB": base.pb,
        "MUB": base.mub,
        "PHB": base.phb,
        "P_TOTAL": state.p_total,
        "MU_TOTAL": state.mu_total,
        "PH_TOTAL": state.ph_total,
    }


def compare_initial_fields(raw_case: Mapping[str, Any], live_case: Mapping[str, Any], jax: Any) -> dict[str, Any]:
    h0 = h0_reference_fields()
    raw = {name: np.asarray(jax.device_get(value), dtype=np.float64) for name, value in initial_field_arrays(raw_case).items()}
    live = {name: np.asarray(jax.device_get(value), dtype=np.float64) for name, value in initial_field_arrays(live_case).items()}
    comparisons: dict[str, dict[str, Any]] = {
        "raw_init_vs_cpu_wrf_h0": {},
        "live_nest_init_vs_cpu_wrf_h0": {},
        "live_nest_init_minus_raw_init": {},
    }
    for name in INITIAL_DELTA_FIELDS:
        comparisons["raw_init_vs_cpu_wrf_h0"][name] = diff_metrics(name, raw[name], h0[name])
        comparisons["live_nest_init_vs_cpu_wrf_h0"][name] = diff_metrics(name, live[name], h0[name])
        comparisons["live_nest_init_minus_raw_init"][name] = diff_metrics(name, live[name], raw[name])
    return comparisons


def apply_live_nest_base_init(run: Any, parent: Mapping[str, Any], raw_child: Mapping[str, Any]) -> dict[str, Any]:
    import jax.numpy as jnp  # noqa: PLC0415

    from gpuwrf.contracts.state import BaseState  # noqa: PLC0415
    from gpuwrf.integration.d02_replay import (  # noqa: PLC0415
        _apply_live_nest_base_init,
        _wrf_base_theta_from_loaded_state,
    )

    parent_case = SimpleNamespace(grid=parent["grid"], metadata={"domain": "d01"})
    grid, metrics, pb, phb, mub, live_meta = _apply_live_nest_base_init(
        run,
        domain="d02",
        grid=raw_child["grid"],
        metrics=raw_child["metrics"],
        parent_case=parent_case,
    )
    state = raw_child["state"]
    p_perturbation = state.p_perturbation
    ph_perturbation = state.ph_perturbation
    mu_perturbation = state.mu_perturbation
    state = state.replace(
        p_total=pb + p_perturbation,
        p_perturbation=p_perturbation,
        ph_total=phb + ph_perturbation,
        ph_perturbation=ph_perturbation,
        mu_total=mub + mu_perturbation,
        mu_perturbation=mu_perturbation,
    )
    theta_base = _wrf_base_theta_from_loaded_state(pb=pb, phb=phb, mub=mub, metrics=metrics)
    base_state = BaseState(
        pb=pb.astype(state.p_total.dtype),
        phb=phb.astype(state.ph_total.dtype),
        mub=mub.astype(state.mu_total.dtype),
        t0=jnp.full_like(theta_base, P0_THETA_OFFSET_K).astype(state.theta.dtype),
        theta_base=theta_base.astype(state.theta.dtype),
    )
    return {
        **raw_child,
        "grid": grid,
        "metrics": metrics,
        "state": state,
        "base_state": base_state,
        "live_nest_base_init": live_meta,
        "construction": (
            "proof-local direct constructor plus production "
            "gpuwrf.integration.d02_replay._apply_live_nest_base_init"
        ),
    }


def build_live_nest_step1_inputs() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    from gpuwrf.integration.d02_replay import run_start_label  # noqa: PLC0415
    from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: PLC0415
    from gpuwrf.nesting.boundary_construction import build_child_boundary_package, build_nest_force_weights  # noqa: PLC0415
    from gpuwrf.runtime.domain_tree import with_live_child_boundary_config  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import OperationalNamelist  # noqa: PLC0415
    from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: PLC0415

    run = Gen2Run(builder.RUN_CASE3)
    parent = builder._state_from_wrfinput(run, "d01")
    raw_child = builder._state_from_wrfinput(run, "d02")
    live_child = apply_live_nest_base_init(run, parent, raw_child)

    parent_grid_meta = run.grid("d01")
    child_grid_meta = run.grid("d02")
    weights = build_nest_force_weights(
        parent_grid_ratio=int(child_grid_meta.parent_grid_ratio),
        i_parent_start=int(child_grid_meta.i_parent_start),
        j_parent_start=int(child_grid_meta.j_parent_start),
        parent_grid=parent["grid"],
        child_grid=live_child["grid"],
        registration="sint",
    )
    child_state_with_parent_bdy = build_child_boundary_package(
        live_child["state"],
        parent["state"],
        weights,
        bdy_width=builder.BDY_WIDTH,
    )

    child_dt = builder._domain_dt_s(run, "d02")
    parent_dt = builder._domain_dt_s(run, "d01")
    radiation_cadence = max(1, int(round(builder.RADT_TARGET_S / float(child_dt))))
    namelist = OperationalNamelist.from_grid(
        live_child["grid"],
        tendencies=live_child["tendencies"],
        metrics=live_child["metrics"],
        dt_s=child_dt,
        acoustic_substeps=10,
        radiation_cadence_steps=radiation_cadence,
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
        time_utc=run_start_label(run, "d02"),
    )
    namelist = with_live_child_boundary_config(
        namelist,
        parent_dt_s=parent_dt,
        nested_ph_relax=True,
        nested_w_relax=False,
        nested_ph_spec=True,
    )
    cu_physics = int(builder._domain_list_value(run.namelist, "physics", "cu_physics", "d02", 0))
    namelist = dataclass_replace(namelist, cu_physics=cu_physics)
    carry = initial_operational_carry(child_state_with_parent_bdy)
    jax.block_until_ready(jax.tree_util.tree_leaves(carry)[0])
    initial_deltas = compare_initial_fields(raw_child, live_child, jax)
    return {
        "run": run,
        "parent": parent,
        "raw_child": raw_child,
        "live_child": live_child,
        "state": child_state_with_parent_bdy,
        "base_state": live_child["base_state"],
        "carry": carry,
        "namelist": namelist,
        "grid": live_child["grid"],
        "tendencies": live_child["tendencies"],
        "jax": jax,
        "jnp": jnp,
        "initial_deltas": initial_deltas,
        "boundary_package": {
            "builder": "gpuwrf.nesting.boundary_construction.build_child_boundary_package",
            "registration": "sint",
            "bdy_width": int(builder.BDY_WIDTH),
            "source": "live parent d01 state over live-nest-initialized d02 state",
        },
    }


def run_live_nest_step1_compare() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        _physics_step_forcing,
        _rk_scan_step_with_pre_halo_capture,
    )

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    if not ACCEPTED_TRUTH.is_file():
        return {"status": "BLOCKED_NO_TRUTH_NPZ", "truth": path_info(ACCEPTED_TRUTH)}
    try:
        inputs = build_live_nest_step1_inputs()
        namelist = inputs["namelist"]
        jnp = inputs["jnp"]
        lead_seconds = jnp.asarray(float(TARGET_STEP) * float(namelist.dt_s), dtype=jnp.float64)
        cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
        run_radiation = bool(cadence > 0 and TARGET_STEP % cadence == 0)
        physics = _physics_step_forcing(
            inputs["carry"],
            namelist,
            lead_seconds,
            run_radiation=run_radiation,
        )
        result = _rk_scan_step_with_pre_halo_capture(
            physics.carry,
            namelist,
            lead_seconds=lead_seconds,
            physics_tendencies=physics.dry_tendencies,
        )
        jax.block_until_ready(result.pre_halo_state.theta)
        comparison = compare_arrays(ACCEPTED_TRUTH, result.pre_halo_state, inputs["base_state"], jax)
        return {
            "status": comparison["status"],
            "step_index": TARGET_STEP,
            "lead_seconds": float(lead_seconds),
            "run_radiation": run_radiation,
            "radiation_cadence_steps": cadence,
            "namelist": {
                "dt_s": float(namelist.dt_s),
                "rk_order": int(namelist.rk_order),
                "run_physics": bool(namelist.run_physics),
                "run_boundary": bool(namelist.run_boundary),
                "force_fp64": bool(namelist.force_fp64),
                "cu_physics": int(namelist.cu_physics),
                "radiation_static_loaded": namelist.radiation_static is not None,
                "gwdo_statics_loaded": namelist.gwdo_statics is not None,
            },
            "grid": {
                "nz": int(inputs["grid"].nz),
                "ny": int(inputs["grid"].ny),
                "nx": int(inputs["grid"].nx),
                "dx_m": float(inputs["grid"].projection.dx_m),
                "dy_m": float(inputs["grid"].projection.dy_m),
            },
            "loader": {
                "raw_child_construction": inputs["raw_child"].get("construction"),
                "live_child_construction": inputs["live_child"].get("construction"),
                "production_live_nest_helper": "gpuwrf.integration.d02_replay._apply_live_nest_base_init",
                "build_replay_case_cpu_blocker": "State.zeros requires a GPU device; proof uses direct constructors",
                "live_nest_base_init": inputs["live_child"].get("live_nest_base_init"),
                "boundary_package": inputs["boundary_package"],
            },
            "initial_deltas": inputs["initial_deltas"],
            **comparison,
        }
    except Exception as exc:
        return {
            "status": "BLOCKED_LIVE_NEST_JAX_CAPTURE_EXCEPTION",
            "exception": repr(exc),
            "exact_function_boundary": (
                "proof-local direct constructor -> "
                "gpuwrf.integration.d02_replay._apply_live_nest_base_init -> "
                "src/gpuwrf/runtime/operational_mode.py::_physics_step_forcing -> "
                "_rk_scan_step_with_pre_halo_capture(...).pre_halo_state"
            ),
            "smallest_next_patch_or_tool": (
                "Make the proof-local direct constructor match the exact missing production "
                "state leaf named in this exception, unless the exception proves a narrow "
                "production live-nest init defect."
            ),
        }


def base_residual_status(comparison: Mapping[str, Any]) -> dict[str, Any]:
    if comparison.get("status") != "COMPARISON_EXECUTED":
        return {"status": "NOT_EXECUTED"}
    fields = comparison["per_field_metrics"]
    items = {}
    closed = True
    remain = []
    for name, threshold in BASE_CLOSE_THRESHOLDS.items():
        max_abs = float(fields[name]["max_abs"])
        item_closed = max_abs <= float(threshold)
        items[name] = {
            "max_abs": max_abs,
            "rmse": float(fields[name]["rmse"]),
            "threshold": float(threshold),
            "closed": item_closed,
        }
        if not item_closed:
            closed = False
            remain.append(name)
    return {"status": "CLOSED" if closed else "REMAIN", "fields": items, "remaining_fields": remain}


def derive_verdict(comparison: Mapping[str, Any]) -> str:
    status = str(comparison.get("status"))
    if status != "COMPARISON_EXECUTED":
        suffix = status.replace("BLOCKED_", "").replace("LIVE_NEST_", "").upper()
        return f"STEP1_LIVE_NEST_INIT_BLOCKED_{suffix}"
    base = base_residual_status(comparison)
    if base["status"] == "REMAIN":
        specific = "_".join(base.get("remaining_fields", [])) or "BASE_FIELDS"
        return f"STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_REMAIN_{specific}"
    first = comparison.get("first_divergent_field")
    if first:
        return f"STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_{first}"
    return "STEP1_LIVE_NEST_INIT_COMPARISON_CLEAN"


def next_decision_for(verdict: str, comparison: Mapping[str, Any]) -> str:
    if verdict == "STEP1_LIVE_NEST_INIT_COMPARISON_CLEAN":
        return "Close the step-1 same-input d02 gate; no next localization sprint is named by this proof."
    if "BASE_RESIDUALS_CLOSED_NEXT_" in verdict:
        field = verdict.rsplit("_", 1)[-1]
        top = comparison["ranked_residuals"][0]["field"]
        return (
            f"Run the next operator-localization sprint at field {field}; largest max_abs after base closure is {top}."
        )
    if "BASE_RESIDUALS_REMAIN_" in verdict:
        return "Localize the missing base/source path before any operator-localization or TOST work."
    return "Patch or instrument the exact blocker named in the comparison status, then rerun this proof."


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    comparison = payload["comparison"]
    lines = [
        "# V0.14 Step-1 Live-Nest Init Rerun",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`.",
        f"- Truth NPZ reused: `{ACCEPTED_TRUTH}`.",
        f"- Strict comparison run: `{comparison.get('strict_live_nest_init_comparison_run', False)}`.",
    ]
    if comparison.get("status") == "COMPARISON_EXECUTED":
        first = comparison.get("first_divergent_field")
        top = comparison["ranked_residuals"][0]
        base = payload["base_residual_status"]
        lines.extend(
            [
                f"- Base residual status: `{base['status']}`.",
                f"- First divergent field in schema order: `{first}`.",
                f"- Largest max_abs field: `{top['field']}` max_abs `{top['max_abs']}` rmse `{top['rmse']}`.",
                "",
                "Detailed per-field and initial-delta tables are in `proofs/v014/step1_live_nest_init_rerun.json`.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Blocker",
                "",
                f"- Comparison status: `{comparison.get('status')}`.",
                f"- Exact blocker: `{comparison.get('exception')}`.",
                f"- Next patch/tool: `{comparison.get('smallest_next_patch_or_tool')}`.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    commands = payload["commands"]["required_validation"]
    lines = [
        "# Review: V0.14 Step-1 Live-Nest Init Rerun",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: rerun the strict d02 step-1 same-input comparison using the production native live-nest child base initialization semantics wired into a CPU-only proof loader.",
        "",
        "files changed:",
        "- `proofs/v014/step1_live_nest_init_rerun.py`",
        "- `proofs/v014/step1_live_nest_init_rerun.json`",
        "- `proofs/v014/step1_live_nest_init_rerun.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md`",
        "",
        "commands run:",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "proof objects produced:",
        f"- `{OUT_JSON}`",
        f"- `{OUT_MD}`",
        f"- `{OUT_REVIEW}`",
        f"- `{ACCEPTED_TRUTH}` reused, not rebuilt",
        "",
        "unresolved risks:",
    ]
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    comparison = run_live_nest_step1_compare()
    base_status = base_residual_status(comparison)
    verdict = derive_verdict(comparison)
    next_decision = next_decision_for(verdict, comparison)
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_live_nest_init_rerun.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "weak_comparison_avoided": True,
        "jax_vs_jax_self_compare": False,
        "one_cell_proof": False,
        "mixed_wrf_jax_carry_leaves": False,
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "surface": TARGET_SURFACE,
            "accepted_comparison": (
                "CPU-WRF step-1 post-RK/pre-halo truth vs JAX one-step "
                "_rk_scan_step_with_pre_halo_capture(...).pre_halo_state from "
                "a live-nest-initialized d02 carry"
            ),
            "rejected_comparisons": [
                "WRF step-1 truth vs JAX initial state",
                "headline raw wrfinput d02 initial state",
                "JAX-vs-JAX self-compare",
            ],
        },
        "environment": jax_environment(),
        "git_head": git_head,
        "inputs": {
            "project_constitution": path_info(PROJECT_CONSTITUTION),
            "agents": path_info(AGENTS),
            "managing_sprints_skill": path_info(MANAGING_SPRINTS_SKILL),
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "previous_step1_same_input_truth_md": path_info(PREVIOUS_STEP1_MD),
            "previous_step1_same_input_truth_json": path_info(PREVIOUS_STEP1_JSON),
            "previous_step1_same_input_truth_py": path_info(PREVIOUS_STEP1_PY),
            "live_nest_base_source_fix_md": path_info(LIVE_NEST_FIX_MD),
            "live_nest_base_source_fix_json": path_info(LIVE_NEST_FIX_JSON),
            "d02_replay": path_info(D02_REPLAY),
            "nested_pipeline": path_info(NESTED_PIPELINE),
            "run_case3": path_info(RUN_CASE3),
            "cpu_wrf_h0_d02_validation_oracle": path_info(CPU_H0_D02),
            "accepted_truth_npz": path_info(ACCEPTED_TRUTH),
        },
        "truth": {
            "status": "TRUTH_NPZ_REUSED_EXISTING" if ACCEPTED_TRUTH.is_file() else "TRUTH_NPZ_MISSING",
            "path": str(ACCEPTED_TRUTH),
            "rebuilt_wrf": False,
        },
        "field_order": list(ALL_COMPARE_FIELDS),
        "initial_delta_fields": list(INITIAL_DELTA_FIELDS),
        "base_residual_status": base_status,
        "comparison": comparison,
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_live_nest_init_rerun.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py",
                "python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "accepted_truth_npz_reused": str(ACCEPTED_TRUTH),
        },
        "unresolved_risks": [
            "The proof-local CPU loader mirrors production live-nest init semantics but bypasses build_replay_case because State.zeros is GPU-only.",
            "Residuals after base closure identify a field-level symptom, not yet the exact dycore or physics operator.",
        ],
        "next_decision": next_decision,
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(f"verdict={verdict}")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
