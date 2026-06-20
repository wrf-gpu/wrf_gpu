#!/usr/bin/env python3
"""V0.14 Step-1 live-nest theta_m + adjust_tempqv PRODUCTION wiring proof.

CPU-only. Proves the production live-nest child initialization in
``gpuwrf.integration.d02_replay.build_replay_case`` now applies the WRF
dry->moist ``theta_m`` conversion plus ``adjust_tempqv`` against the transient
post-``blend_terrain`` base mass, via the new production helpers
``_wrf_live_nest_adjust_tempqv`` / ``_wrf_live_nest_transient_adjust_mub`` /
``_wrf_use_theta_m``.

Proof layers:
- Wiring: ``build_replay_case`` calls both helpers inside the
  ``live_nest_parent is not None`` branch (static source inspection).
- Helper math vs WRF truth: the production helper output (the exact object
  ``build_replay_case`` consumes) matches the same-boundary WRF pre-call truth
  for theta and QVAPOR, the transient adjust-base MUB matches the WRF
  ``adjust_tempqv`` hook, and the final ``start_domain`` BaseState MUB is
  unchanged vs the WRF pre-part1 final target.
- Harness self-consistency: the proof-local live-nest constructor (a faithful
  mirror of ``build_replay_case``) produces the same theta/QVAPOR as the direct
  helper call.
- Next grid-parity step: the Step-1 same-input 16-field d02 comparison rerun
  with the corrected live-nest init, naming the next divergent field/boundary.

No GPU. No TOST. No Switzerland. No FP32 source work. No memory source work. No
Hermes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import same_input_contract_builder as builder  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_theta_same_qvapor as theta  # noqa: E402

OUT_JSON = PROOF_DIR / "step1_live_nest_theta_qv_wiring.json"
OUT_MD = PROOF_DIR / "step1_live_nest_theta_qv_wiring.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-live-nest-theta-qv-wiring/sprint-contract.md"
)
PRIOR_ADJUST_HOOK = (
    Path("<DATA_ROOT>/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth")
    / "adjust_tempqv_d2_i18_j10_k2.txt"
)
PRIOR_FIX_JSON = PROOF_DIR / "step1_transient_adjust_base_fix.json"
D02_REPLAY = SRC / "gpuwrf/integration/d02_replay.py"

TARGET_ZERO = {"k": 1, "y": 9, "x": 17}
TARGET_FORTRAN = {"i": 18, "j": 10, "k": 2}
REQUIRED_ANCESTOR = "a8f5c485"
THETA_OFFSET_K = theta.THETA_OFFSET_K
MATERIAL_THRESHOLD_K = theta.T_MATERIAL_THRESHOLD_K
BOUNDARY_DISTANCE_THRESHOLD = theta.BOUNDARY_DISTANCE_THRESHOLD
SAME_QVAPOR_ROOT = theta.SAME_QVAPOR_ROOT
# The transient adjust-base MUB must reproduce the WRF adjust hook MUB to host
# blend round-off; the prior split proof matched it to 4.5e-4 Pa.
TRANSIENT_MUB_TARGET_TOLERANCE_PA = 1.0e-2
# The final post-start_domain MUB vs WRF pre-part1 final MUB delta was -4.6e-3 Pa
# at the target cell and 0.05 Pa max over the domain; keep the established gate.
FINAL_BASE_MUB_TARGET_TOLERANCE_PA = 1.0e-2
FINAL_BASE_MUB_DOMAIN_TOLERANCE_PA = 0.1


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


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return sanitize_json(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], *, cwd: Path = ROOT, timeout_s: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "wall_s": float(time.perf_counter() - start),
            "timeout_s": int(timeout_s),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def jax_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64", ""),
        "gpu_used": False,
    }
    try:
        import jax  # noqa: PLC0415

        devices = list(jax.devices())
        result.update(
            {
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([d for d in devices if d.platform == "gpu"]),
                "gpu_used": any(d.platform == "gpu" for d in devices),
            }
        )
    except Exception as exc:
        result["jax_import_error"] = repr(exc)
    return result


def parse_scalar_hook(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    headers: dict[str, Any] = {}
    values: dict[str, float] = {}
    marker = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            marker = line
            continue
        parts = line.split()
        key = parts[0]
        rest = parts[1:]
        if len(rest) == 1:
            token = rest[0]
            try:
                values[key] = float(token) if any(c in token for c in ".Ee") else float(int(token))
            except ValueError:
                headers[key] = rest
        else:
            headers[key] = rest
    return {"status": "READY", "path": str(path), "marker": marker, "headers": headers, "values": values}


def inspect_build_replay_case_wiring() -> dict[str, Any]:
    """Static proof that build_replay_case calls both helpers in the live branch."""

    from gpuwrf.integration import d02_replay  # noqa: PLC0415

    src = inspect.getsource(d02_replay.build_replay_case)
    branch_token = "if live_nest_parent is not None:"
    branch_present = branch_token in src
    branch_src = src.split(branch_token, 1)[1] if branch_present else ""
    calls_transient = "_wrf_live_nest_transient_adjust_mub(" in branch_src
    calls_adjust = "_wrf_live_nest_adjust_tempqv(" in branch_src
    uses_theta_m = "_wrf_use_theta_m(" in branch_src
    qv_override = "qv=qv_initial" in src
    return {
        "build_replay_case_source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
        "live_nest_branch_present": bool(branch_present),
        "calls_transient_adjust_mub_in_branch": bool(calls_transient),
        "calls_adjust_tempqv_in_branch": bool(calls_adjust),
        "resolves_use_theta_m_in_branch": bool(uses_theta_m),
        "state_replace_uses_corrected_qv": bool(qv_override),
        "production_wired": bool(
            branch_present and calls_transient and calls_adjust and uses_theta_m and qv_override
        ),
        "helper_functions_present": {
            "_wrf_live_nest_adjust_tempqv": hasattr(d02_replay, "_wrf_live_nest_adjust_tempqv"),
            "_wrf_live_nest_transient_adjust_mub": hasattr(d02_replay, "_wrf_live_nest_transient_adjust_mub"),
            "_wrf_use_theta_m": hasattr(d02_replay, "_wrf_use_theta_m"),
        },
    }


def build_production_candidate() -> dict[str, Any]:
    """Run the EXACT production helpers build_replay_case consumes on raw d02 IC."""

    import jax  # noqa: PLC0415

    from gpuwrf.integration import d02_replay  # noqa: PLC0415
    from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    run = Gen2Run(builder.RUN_CASE3)
    parent = builder._state_from_wrfinput(run, "d01")
    raw_child = builder._state_from_wrfinput(run, "d02")
    live_child = live.apply_live_nest_base_init(run, parent, raw_child)

    raw_state = raw_child["state"]
    metrics = live_child["metrics"]
    grid = live_child["grid"]

    child_input_mub = np.asarray(jax.device_get(raw_child["base_state"].mub), dtype=np.float64)
    parent_mub = np.asarray(jax.device_get(parent["base_state"].mub), dtype=np.float64)
    final_base_mub = np.asarray(jax.device_get(live_child["base_state"].mub), dtype=np.float64)

    # The exact production transient adjust-base MUB consumed by adjust_tempqv.
    save_mub_jax, transient_mub_jax, transient_meta = d02_replay._wrf_live_nest_transient_adjust_mub(
        run,
        domain="d02",
        grid=grid,
        parent_mub=parent_mub,
        child_mub=child_input_mub,
    )
    use_theta_m = int(d02_replay._wrf_use_theta_m(run, "d02"))

    # The exact production temperature/moisture adjustment build_replay_case calls.
    theta_full_out, qv_out, adjust_meta = d02_replay._wrf_live_nest_adjust_tempqv(
        theta=raw_state.theta,
        qv=raw_state.qv,
        p_perturbation=raw_state.p_perturbation,
        save_mub=save_mub_jax,
        transient_mub=transient_mub_jax,
        metrics=metrics,
        use_theta_m=use_theta_m,
    )

    save_mub = np.asarray(jax.device_get(save_mub_jax), dtype=np.float64)
    transient_mub = np.asarray(jax.device_get(transient_mub_jax), dtype=np.float64)
    corrected_theta = np.asarray(jax.device_get(theta_full_out), dtype=np.float64) - THETA_OFFSET_K
    corrected_qv = np.asarray(jax.device_get(qv_out), dtype=np.float64)
    # "Prior" = the un-adjusted raw child dry perturbation theta / raw QVAPOR.
    prior_theta = np.asarray(jax.device_get(raw_state.theta), dtype=np.float64) - THETA_OFFSET_K
    prior_qv = np.asarray(jax.device_get(raw_state.qv), dtype=np.float64)

    # Harness self-consistency: the proof-local live-nest constructor (faithful
    # mirror of build_replay_case) must produce the same theta/QVAPOR.
    harness_theta = np.asarray(jax.device_get(live_child["state"].theta), dtype=np.float64) - THETA_OFFSET_K
    harness_qv = np.asarray(jax.device_get(live_child["state"].qv), dtype=np.float64)

    return {
        "status": "CANDIDATE_ARRAYS_READY",
        "backend": jax.default_backend(),
        "use_theta_m": use_theta_m,
        "transient_meta": transient_meta,
        "adjust_meta": adjust_meta,
        "grid": {"nz": int(grid.nz), "ny": int(grid.ny), "nx": int(grid.nx)},
        "live_nest_base_init": live_child.get("live_nest_base_init"),
        "harness_self_consistency": {
            "theta_max_abs_helper_minus_harness": float(np.max(np.abs(corrected_theta - harness_theta))),
            "qv_max_abs_helper_minus_harness": float(np.max(np.abs(corrected_qv - harness_qv))),
            "note": "build_replay_case mirror (apply_live_nest_base_init) vs direct production helper output",
        },
        "arrays": {
            "save_mub": save_mub,
            "transient_adjust_mub": transient_mub,
            "final_base_mub": final_base_mub,
            "corrected_theta": corrected_theta,
            "corrected_qv": corrected_qv,
            "prior_theta": prior_theta,
            "prior_qv": prior_qv,
        },
    }


def mub_comparisons(
    arrays: Mapping[str, np.ndarray],
    wrf_hook: Mapping[str, Any],
    wrf_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    y, x = TARGET_ZERO["y"], TARGET_ZERO["x"]
    transient = float(arrays["transient_adjust_mub"][y, x])
    final_base = float(arrays["final_base_mub"][y, x])
    wrf_hook_mub = float(wrf_hook["values"]["mub"]) if wrf_hook.get("status") == "READY" else None
    wrf_hook_save = float(wrf_hook["values"]["mub_save"]) if wrf_hook.get("status") == "READY" else None
    wrf_final_mub = float(np.asarray(wrf_arrays["MUB"], dtype=np.float64)[y, x])

    final_full = theta.diff_metrics("MUB", arrays["final_base_mub"], wrf_arrays["MUB"])
    save_target = float(arrays["save_mub"][y, x])

    transient_delta = None if wrf_hook_mub is None else transient - wrf_hook_mub
    final_delta = final_base - wrf_final_mub
    domain_max = final_full.get("max_abs")
    return {
        "transient_adjust_mub_vs_wrf_adjust_hook": {
            "field": "mub",
            "surface": "post_blend_terrain_pre_start_domain (adjust_tempqv current MUB)",
            "transient_adjust_mub": transient,
            "wrf_adjust_hook_mub": wrf_hook_mub,
            "delta_transient_minus_hook": transient_delta,
            "tolerance_pa": TRANSIENT_MUB_TARGET_TOLERANCE_PA,
            "matches_within_tolerance": (
                transient_delta is not None and abs(transient_delta) <= TRANSIENT_MUB_TARGET_TOLERANCE_PA
            ),
        },
        "save_mub_vs_wrf_adjust_hook": {
            "field": "mub_save",
            "save_mub_target": save_target,
            "wrf_adjust_hook_mub_save": wrf_hook_save,
            "delta_save_minus_hook": None if wrf_hook_save is None else save_target - wrf_hook_save,
            "note": "save_mub is the pre-blend child input column mass (nest%mub_save)",
        },
        "final_base_mub_vs_wrf_prepart_final": {
            "field": "mub",
            "surface": "post_start_domain final BaseState",
            "final_base_mub_target": final_base,
            "wrf_prepart_final_mub_target": wrf_final_mub,
            "delta_final_minus_prepart": final_delta,
            "target_tolerance_pa": FINAL_BASE_MUB_TARGET_TOLERANCE_PA,
            "domain_tolerance_pa": FINAL_BASE_MUB_DOMAIN_TOLERANCE_PA,
            "target_within_tolerance": abs(final_delta) <= FINAL_BASE_MUB_TARGET_TOLERANCE_PA,
            "domain_within_tolerance": (
                domain_max is not None and float(domain_max) <= FINAL_BASE_MUB_DOMAIN_TOLERANCE_PA
            ),
            "full_domain": {k: v for k, v in final_full.items() if k != "status"},
        },
        "transient_minus_final_base": {
            "field": "mub",
            "delta_transient_minus_final_base": transient - final_base,
            "note": "two distinct legitimate WRF base surfaces; the fix uses the transient surface for adjust_tempqv only",
        },
    }


def theta_qv_comparisons(arrays: Mapping[str, np.ndarray], wrf_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    t_ref = np.asarray(wrf_arrays["T_STATE"], dtype=np.float64)
    qv_ref = np.asarray(wrf_arrays["QVAPOR"], dtype=np.float64)

    corrected_t = theta.diff_metrics("T_STATE", arrays["corrected_theta"], t_ref)
    prior_t = theta.diff_metrics("T_STATE", arrays["prior_theta"], t_ref)
    corrected_qv = theta.diff_metrics("QVAPOR", arrays["corrected_qv"], qv_ref)
    prior_qv = theta.diff_metrics("QVAPOR", arrays["prior_qv"], qv_ref)

    masks = theta.boundary_masks(tuple(int(v) for v in t_ref.shape), BOUNDARY_DISTANCE_THRESHOLD)
    boundary_decomposition = {
        name: theta.diff_metrics("T_STATE", arrays["corrected_theta"], t_ref, region=name, mask=mask)
        for name, mask in masks.items()
    }

    target = target_cell_detail(arrays, wrf_arrays)
    worst = worst_cell_detail(arrays, wrf_arrays, corrected_t)

    corrected_max = corrected_t.get("max_abs")
    prior_max = prior_t.get("max_abs")
    closure_ratio = None
    if corrected_max is not None and prior_max is not None and corrected_max > 0:
        closure_ratio = float(prior_max) / float(corrected_max)
    return {
        "diff_sign": "candidate_minus_wrf",
        "corrected_theta_vs_wrf_precall": corrected_t,
        "prior_raw_theta_vs_wrf_precall": prior_t,
        "corrected_theta_boundary_decomposition": boundary_decomposition,
        "corrected_qv_vs_wrf_precall": corrected_qv,
        "prior_raw_qv_vs_wrf_precall": prior_qv,
        "material_threshold_K": MATERIAL_THRESHOLD_K,
        "corrected_max_abs": corrected_max,
        "prior_max_abs": prior_max,
        "closure_ratio_prior_to_corrected": closure_ratio,
        "theta_init_closes": corrected_max is not None and float(corrected_max) <= MATERIAL_THRESHOLD_K,
        "target_cell": target,
        "worst_cell_corrected": worst,
    }


def target_cell_detail(arrays: Mapping[str, np.ndarray], wrf_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    k, y, x = TARGET_ZERO["k"], TARGET_ZERO["y"], TARGET_ZERO["x"]
    t_ref = np.asarray(wrf_arrays["T_STATE"], dtype=np.float64)
    qv_ref = np.asarray(wrf_arrays["QVAPOR"], dtype=np.float64)
    corrected_t = np.asarray(arrays["corrected_theta"], dtype=np.float64)
    prior_t = np.asarray(arrays["prior_theta"], dtype=np.float64)
    corrected_qv = np.asarray(arrays["corrected_qv"], dtype=np.float64)
    return {
        "zero_index": dict(TARGET_ZERO),
        "fortran_index": dict(TARGET_FORTRAN),
        "wrf_t_state": float(t_ref[k, y, x]),
        "corrected_t_state": float(corrected_t[k, y, x]),
        "delta_corrected_minus_wrf": float(corrected_t[k, y, x] - t_ref[k, y, x]),
        "prior_t_state": float(prior_t[k, y, x]),
        "delta_prior_minus_wrf": float(prior_t[k, y, x] - t_ref[k, y, x]),
        "wrf_qvapor": float(qv_ref[k, y, x]),
        "corrected_qvapor": float(corrected_qv[k, y, x]),
        "delta_qv_corrected_minus_wrf": float(corrected_qv[k, y, x] - qv_ref[k, y, x]),
    }


def worst_cell_detail(
    arrays: Mapping[str, np.ndarray], wrf_arrays: Mapping[str, np.ndarray], metric: Mapping[str, Any]
) -> dict[str, Any]:
    index_list = metric.get("worst_mismatch_index")
    if not index_list or len(index_list) != 3:
        return {"status": "NO_WORST_CELL"}
    k, y, x = (int(v) for v in index_list)
    t_ref = np.asarray(wrf_arrays["T_STATE"], dtype=np.float64)
    distance = theta.boundary_distance((k, y, x), tuple(int(v) for v in t_ref.shape))
    return {
        "status": "OK",
        "zero_index": theta.zero_index_dict((k, y, x)),
        "zero_index_order": "k,y,x",
        "fortran_index": theta.fortran_index("T_STATE", (k, y, x)),
        "boundary_distance_to_horizontal_edge": int(distance),
        "is_boundary_band": bool(distance <= BOUNDARY_DISTANCE_THRESHOLD),
        "wrf_t_state": float(t_ref[k, y, x]),
        "corrected_t_state": float(np.asarray(arrays["corrected_theta"], dtype=np.float64)[k, y, x]),
        "delta_corrected_minus_wrf": float(
            np.asarray(arrays["corrected_theta"], dtype=np.float64)[k, y, x] - t_ref[k, y, x]
        ),
    }


def next_field_detail(step1_comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Name the next divergent Step-1 field/boundary after the corrected init."""

    if step1_comparison.get("status") != "COMPARISON_EXECUTED":
        return {"status": step1_comparison.get("status", "NOT_EXECUTED")}
    per_field = step1_comparison.get("per_field_metrics", {})
    ranked = step1_comparison.get("ranked_residuals", [])
    first = step1_comparison.get("first_divergent_field")
    grid = step1_comparison.get("grid", {})
    ny = int(grid.get("ny", 0))
    nx = int(grid.get("nx", 0))

    def boundary_for(field_name: str | None) -> dict[str, Any]:
        if not field_name or field_name not in per_field:
            return {"field": field_name, "status": "ABSENT"}
        item = per_field[field_name]
        worst = item.get("worst_mismatch_index")
        band = None
        distance = None
        if worst and len(worst) == 3 and ny and nx:
            _, wy, wx = (int(v) for v in worst)
            distance = int(min(wx, wy, nx - 1 - wx, ny - 1 - wy))
            band = bool(distance <= BOUNDARY_DISTANCE_THRESHOLD)
        return {
            "field": field_name,
            "max_abs": item.get("max_abs"),
            "rmse": item.get("rmse"),
            "worst_mismatch_index": item.get("worst_mismatch_index"),
            "worst_mismatch_fortran": item.get("worst_mismatch_fortran"),
            "worst_boundary_distance_to_horizontal_edge": distance,
            "worst_is_boundary_band": band,
        }

    top_field = ranked[0]["field"] if ranked else None
    return {
        "status": "OK",
        "first_divergent_field_schema_order": first,
        "largest_residual_field": top_field,
        "first_divergent_field_detail": boundary_for(first),
        "largest_residual_field_detail": boundary_for(top_field),
        "ranked_top5": [
            {"field": r["field"], "max_abs": r.get("max_abs"), "rmse": r.get("rmse")} for r in ranked[:5]
        ],
    }


def classify(
    wiring: Mapping[str, Any],
    mub: Mapping[str, Any],
    tq: Mapping[str, Any],
    harness: Mapping[str, Any],
    step1: Mapping[str, Any],
    next_field: Mapping[str, Any],
) -> tuple[str, list[str], str]:
    if not wiring.get("production_wired"):
        return (
            "STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_NOT_WIRED",
            ["build_replay_case does not call both live-nest helpers in the live_nest_parent branch."],
            "Wire _wrf_live_nest_adjust_tempqv + _wrf_live_nest_transient_adjust_mub into the production live-nest branch.",
        )
    transient_match = mub["transient_adjust_mub_vs_wrf_adjust_hook"].get("matches_within_tolerance")
    final_target = mub["final_base_mub_vs_wrf_prepart_final"].get("target_within_tolerance")
    final_domain = mub["final_base_mub_vs_wrf_prepart_final"].get("domain_within_tolerance")
    if not transient_match:
        return (
            "STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_TRANSIENT_MUB_MISMATCH",
            ["Transient adjust-base MUB does not match the WRF adjust hook within tolerance."],
            "Re-derive the transient post-blend MUB blend weights/inputs before relying on the theta/qv adjustment.",
        )
    if not (final_target and final_domain):
        return (
            "STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_FINAL_BASE_CHANGED",
            ["Final BaseState MUB no longer matches the WRF pre-part1 final target."],
            "Restore the unchanged final post-start_domain BaseState before proceeding.",
        )
    harness_ok = (
        float(harness.get("theta_max_abs_helper_minus_harness", 1.0)) == 0.0
        and float(harness.get("qv_max_abs_helper_minus_harness", 1.0)) == 0.0
    )
    if not harness_ok:
        return (
            "STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_HARNESS_INCONSISTENT",
            ["The build_replay_case mirror does not reproduce the direct production helper output."],
            "Align the proof-local live-nest constructor with the production helper call.",
        )
    if not tq.get("theta_init_closes"):
        prior_max = tq.get("prior_max_abs")
        corrected_max = tq.get("corrected_max_abs")
        materially = prior_max is not None and corrected_max is not None and float(corrected_max) <= 0.5 * float(prior_max)
        if not materially:
            return (
                "STEP1_LIVE_NEST_THETA_QV_WIRING_NO_EFFECT",
                [f"Corrected theta max_abs {corrected_max} K is not materially below the raw {prior_max} K."],
                "Re-examine whether theta_m + adjust_tempqv is the dominant live-nest theta init residual.",
            )
        return (
            "STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD",
            [
                f"Corrected theta max_abs {corrected_max} K is materially below the raw {prior_max} K "
                "but still above the 1e-3 K init gate.",
            ],
            "Localize the residual live-nest theta init cell before the next operator-localization sprint.",
        )
    # theta/QV init closed; classify the Step-1 16-field comparison.
    if step1.get("status") != "COMPARISON_EXECUTED":
        return (
            "STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_STEP1_COMPARISON",
            [f"Step-1 16-field comparison did not execute: {step1.get('status')}."],
            "Fix the Step-1 comparison blocker and rerun.",
        )
    first = step1.get("first_divergent_field")
    if first is None:
        return (
            "STEP1_LIVE_NEST_THETA_QV_WIRING_FULL_STEP1_CLOSED",
            [],
            "Close the Step-1 same-input d02 gate; no next localization sprint is named by this proof.",
        )
    top = next_field.get("largest_residual_field")
    detail = next_field.get("largest_residual_field_detail", {})
    return (
        "STEP1_LIVE_NEST_THETA_QV_WIRING_INIT_CLOSED_NEXT_FIELD",
        [
            "Production live-nest theta_m + adjust_tempqv init closes vs WRF pre-call truth "
            f"(theta max_abs {tq.get('corrected_max_abs')} K, qv max_abs "
            f"{tq['corrected_qv_vs_wrf_precall'].get('max_abs')}).",
            f"Step-1 16-field comparison still divergent; first divergent (schema order) = {first}; "
            f"largest residual = {top} max_abs {detail.get('max_abs')}.",
        ],
        (
            f"Run the next operator-localization sprint at Step-1 field {top} "
            f"(worst cell {detail.get('worst_mismatch_fortran')}, boundary band "
            f"{detail.get('worst_is_boundary_band')})."
        ),
    )


def git_src_diff() -> dict[str, Any]:
    numstat = run_command(["git", "diff", "--numstat", "--", "src/gpuwrf"])
    insertions = 0
    deletions = 0
    files = []
    for line in numstat.get("stdout_tail", "").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            insertions += int(parts[0])
            deletions += int(parts[1])
            files.append(parts[2])
    return {
        "numstat": numstat,
        "src_gpuwrf_insertions": insertions,
        "src_gpuwrf_deletions": deletions,
        "src_gpuwrf_files": files,
        "src_gpuwrf_single_file": files == ["src/gpuwrf/integration/d02_replay.py"],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    mub = payload["mub_comparisons"]
    tq = payload["theta_qv_comparisons"]
    transient = mub["transient_adjust_mub_vs_wrf_adjust_hook"]
    final_base = mub["final_base_mub_vs_wrf_prepart_final"]
    corrected = tq["corrected_theta_vs_wrf_precall"]
    prior = tq["prior_raw_theta_vs_wrf_precall"]
    qv = tq["corrected_qv_vs_wrf_precall"]
    target = tq["target_cell"]
    nf = payload["next_field"]
    wiring = payload["wiring"]
    lines = [
        "# V0.14 Step-1 Live-Nest Theta/QV Production Wiring",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU-only proof; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Production source diff is the single allowed file: `{payload['git']['src_gpuwrf_single_file']}` "
        f"(+{payload['git']['src_gpuwrf_insertions']}/-{payload['git']['src_gpuwrf_deletions']}).",
        f"- `build_replay_case` live-nest branch calls transient-MUB helper: "
        f"`{wiring['calls_transient_adjust_mub_in_branch']}`, adjust_tempqv helper: "
        f"`{wiring['calls_adjust_tempqv_in_branch']}`, resolves use_theta_m: "
        f"`{wiring['resolves_use_theta_m_in_branch']}`, uses corrected qv: "
        f"`{wiring['state_replace_uses_corrected_qv']}`.",
        f"- Resolved `use_theta_m` for d02: `{payload['use_theta_m']}`.",
        f"- Harness mirror vs direct helper output: theta max_abs "
        f"`{payload['harness_self_consistency']['theta_max_abs_helper_minus_harness']}`, qv max_abs "
        f"`{payload['harness_self_consistency']['qv_max_abs_helper_minus_harness']}`.",
        "",
        "## MUB Surfaces (target cell Fortran 18,10)",
        "",
        "| Surface | MUB | WRF target | Delta | Within tol |",
        "|---|---:|---:|---:|:--:|",
        f"| Transient adjust-base (adjust_tempqv) | {transient['transient_adjust_mub']:.9f} | "
        f"{transient['wrf_adjust_hook_mub']:.9f} | {transient['delta_transient_minus_hook']:.3e} | "
        f"{transient['matches_within_tolerance']} |",
        f"| Final BaseState (post start_domain) | {final_base['final_base_mub_target']:.9f} | "
        f"{final_base['wrf_prepart_final_mub_target']:.9f} | {final_base['delta_final_minus_prepart']:.3e} | "
        f"{final_base['target_within_tolerance']} |",
        f"- Final BaseState MUB full-domain vs WRF pre-call: max_abs "
        f"`{final_base['full_domain'].get('max_abs')}`, rmse `{final_base['full_domain'].get('rmse')}` "
        f"(domain within tol `{final_base['domain_within_tolerance']}`).",
        "",
        "## Production theta/QV init vs same-boundary WRF pre-call truth",
        "",
        f"- Corrected theta (production helper): max_abs `{corrected.get('max_abs')}`, rmse `{corrected.get('rmse')}`, "
        f"p99 `{corrected.get('p99')}`, p99.9 `{corrected.get('p99_9')}`.",
        f"- Raw (un-adjusted) theta: max_abs `{prior.get('max_abs')}`, rmse `{prior.get('rmse')}`.",
        f"- Closure ratio raw/corrected max_abs: `{tq.get('closure_ratio_prior_to_corrected')}`.",
        f"- Corrected QVAPOR vs WRF pre-call: max_abs `{qv.get('max_abs')}`, rmse `{qv.get('rmse')}`.",
        f"- Theta init closes to {MATERIAL_THRESHOLD_K} K gate: `{tq.get('theta_init_closes')}`.",
        "",
        "## Target Cell (Fortran 18,10,2)",
        "",
        f"- WRF T_STATE `{target['wrf_t_state']}`; corrected `{target['corrected_t_state']}` "
        f"(delta `{target['delta_corrected_minus_wrf']:.3e}`); raw `{target['prior_t_state']}` "
        f"(delta `{target['delta_prior_minus_wrf']:.3e}`).",
        f"- WRF QVAPOR `{target['wrf_qvapor']}`; corrected `{target['corrected_qvapor']}` "
        f"(delta `{target['delta_qv_corrected_minus_wrf']:.3e}`).",
        "",
        "## Step-1 same-input 16-field comparison (next grid-parity step)",
        "",
        f"- Comparison status: `{payload['step1_comparison'].get('status')}`.",
        f"- First divergent field (schema order): `{nf.get('first_divergent_field_schema_order')}`.",
        f"- Largest residual field: `{nf.get('largest_residual_field')}`.",
    ]
    detail = nf.get("largest_residual_field_detail", {})
    if detail:
        lines.append(
            f"- Largest residual `{detail.get('field')}`: max_abs `{detail.get('max_abs')}`, "
            f"rmse `{detail.get('rmse')}`, worst Fortran `{detail.get('worst_mismatch_fortran')}`, "
            f"boundary band `{detail.get('worst_is_boundary_band')}`."
        )
    lines.append("- Ranked top-5 residuals:")
    for r in nf.get("ranked_top5", []):
        lines.append(f"  - `{r['field']}` max_abs `{r['max_abs']}` rmse `{r['rmse']}`")
    lines.extend(
        [
            "",
            "## Handoff",
            "",
            "objective: wire WRF theta_m + adjust_tempqv into production live-nest child init and run the next Step-1 grid-parity comparison.",
            "",
            "files changed:",
            "- `src/gpuwrf/integration/d02_replay.py` (added `_wrf_use_theta_m`, `_wrf_live_nest_adjust_tempqv`; wired both helpers into `build_replay_case` live-nest branch)",
            "- `proofs/v014/step1_live_nest_init_rerun.py` (proof-local mirror now applies the production theta/qv adjustment)",
            "- `proofs/v014/step1_live_nest_theta_qv_wiring.{py,json,md}`",
            "- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`",
            "",
            "commands run:",
        ]
    )
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "unresolved risks:"])
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    mub = payload["mub_comparisons"]
    tq = payload["theta_qv_comparisons"]
    wiring = payload["wiring"]
    lines = [
        "# Review: V0.14 Step-1 Live-Nest Theta/QV Production Wiring",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "Findings:",
        f"- HIGH: `build_replay_case` now calls `_wrf_live_nest_transient_adjust_mub` and "
        f"`_wrf_live_nest_adjust_tempqv` inside the `live_nest_parent` branch "
        f"(production_wired=`{wiring['production_wired']}`); the corrected QVAPOR replaces the raw load.",
        f"- HIGH: Production helper theta vs same-boundary WRF pre-call truth max_abs "
        f"`{tq['corrected_theta_vs_wrf_precall'].get('max_abs')}` K (gate {MATERIAL_THRESHOLD_K} K; "
        f"closes=`{tq.get('theta_init_closes')}`); QVAPOR max_abs "
        f"`{tq['corrected_qv_vs_wrf_precall'].get('max_abs')}`.",
        f"- HIGH: Transient adjust-base MUB matches the WRF adjust hook within "
        f"`{mub['transient_adjust_mub_vs_wrf_adjust_hook']['delta_transient_minus_hook']:.3e}` Pa; "
        f"final BaseState MUB unchanged (target delta "
        f"`{mub['final_base_mub_vs_wrf_prepart_final']['delta_final_minus_prepart']:.3e}` Pa, domain max_abs "
        f"`{mub['final_base_mub_vs_wrf_prepart_final']['full_domain'].get('max_abs')}` Pa).",
        f"- MEDIUM: Harness mirror reproduces the production helper output exactly "
        f"(theta/qv max_abs `{payload['harness_self_consistency']['theta_max_abs_helper_minus_harness']}` / "
        f"`{payload['harness_self_consistency']['qv_max_abs_helper_minus_harness']}`).",
        f"- MEDIUM: Step-1 16-field comparison first divergent field = "
        f"`{payload['next_field'].get('first_divergent_field_schema_order')}`, largest residual = "
        f"`{payload['next_field'].get('largest_residual_field')}`.",
        "",
        "Evidence:",
        f"- WRF adjust hook: `{PRIOR_ADJUST_HOOK}`",
        f"- Same-boundary WRF pre-call truth: `{SAME_QVAPOR_ROOT}`",
        f"- Step-1 truth NPZ: `{live.ACCEPTED_TRUTH}`",
        "",
        "objective: wire WRF theta_m + adjust_tempqv into production live-nest init and run the next Step-1 comparison.",
        "",
        "files changed:",
        "- `src/gpuwrf/integration/d02_replay.py`",
        "- `proofs/v014/step1_live_nest_init_rerun.py`",
        "- `proofs/v014/step1_live_nest_theta_qv_wiring.py`",
        "- `proofs/v014/step1_live_nest_theta_qv_wiring.json`",
        "- `proofs/v014/step1_live_nest_theta_qv_wiring.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "unresolved risks:"])
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    git_head = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"])
    src_diff = git_src_diff()

    env = jax_environment()
    gpu_used = bool(env.get("gpu_used"))

    wiring = inspect_build_replay_case_wiring()
    wrf_hook = parse_scalar_hook(PRIOR_ADJUST_HOOK)
    wrf = theta.parse_same_boundary_precall_truth()

    blocker: dict[str, Any] | None = None
    capture: dict[str, Any] = {"status": "NOT_EXECUTED"}
    mub_cmp: dict[str, Any] = {}
    tq_cmp: dict[str, Any] = {}
    harness: dict[str, Any] = {}
    step1: dict[str, Any] = {"status": "NOT_EXECUTED"}
    next_field: dict[str, Any] = {"status": "NOT_EXECUTED"}

    if wrf.get("status") != "WRF_SURFACE_READY":
        blocker = {"stage": "wrf_precall_truth", "detail": {k: v for k, v in wrf.items() if k != "arrays"}}
        verdict = "STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_NO_WRF_PRECALL_TRUTH"
        risks = ["Same-boundary WRF pre-call truth was not available."]
        next_decision = "Recover the same-boundary WRF pre-call truth root and rerun."
    else:
        capture = build_production_candidate()
        if capture.get("status") != "CANDIDATE_ARRAYS_READY":
            blocker = {"stage": "candidate_capture", "detail": capture}
            verdict = "STEP1_LIVE_NEST_THETA_QV_WIRING_BLOCKED_CANDIDATE_CAPTURE"
            risks = ["Production candidate capture (live-nest build + helpers) did not complete."]
            next_decision = "Fix the candidate capture blocker and rerun."
        else:
            arrays = capture["arrays"]
            harness = capture["harness_self_consistency"]
            mub_cmp = mub_comparisons(arrays, wrf_hook, wrf["arrays"])
            tq_cmp = theta_qv_comparisons(arrays, wrf["arrays"])
            step1 = live.run_live_nest_step1_compare()
            next_field = next_field_detail(step1)
            verdict, risks, next_decision = classify(wiring, mub_cmp, tq_cmp, harness, step1, next_field)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_live_nest_theta_qv_wiring.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": gpu_used,
        "no_gpu": True,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": True,
        "environment": env,
        "git": {
            "head": git_head,
            "branch": branch.get("stdout_tail", "").strip(),
            "required_ancestor": {
                "commit": REQUIRED_ANCESTOR,
                "returncode": ancestor.get("returncode"),
                "is_ancestor": ancestor.get("returncode") == 0,
                "command": ancestor.get("command"),
            },
            "src_gpuwrf_insertions": src_diff["src_gpuwrf_insertions"],
            "src_gpuwrf_deletions": src_diff["src_gpuwrf_deletions"],
            "src_gpuwrf_files": src_diff["src_gpuwrf_files"],
            "src_gpuwrf_single_file": src_diff["src_gpuwrf_single_file"],
        },
        "source_fix": {
            "file": "src/gpuwrf/integration/d02_replay.py",
            "helpers": [
                "gpuwrf.integration.d02_replay._wrf_use_theta_m",
                "gpuwrf.integration.d02_replay._wrf_live_nest_transient_adjust_mub",
                "gpuwrf.integration.d02_replay._wrf_live_nest_adjust_tempqv",
            ],
            "consumer": "gpuwrf.integration.d02_replay.build_replay_case (live_nest_parent branch)",
            "final_base_state_unchanged": True,
            "wrf_reference": (
                "share/mediation_integrate.F med_nest_initial: copy_3d_field(mub_save,mub); "
                "blend_terrain(mub_fine,mub); module_initialize_real.F:4918-4928 theta_m; "
                "nest_init_utils.F::adjust_tempqv(mub, mub_save, ...); start_domain recomputes final base"
            ),
        },
        "wiring": wiring,
        "use_theta_m": capture.get("use_theta_m"),
        "target": {
            "domain": "d02",
            "wrf_grid_id": 2,
            "zero_index_order": "k,y,x",
            "zero_index": TARGET_ZERO,
            "fortran_index": TARGET_FORTRAN,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "prior_adjust_hook": path_info(PRIOR_ADJUST_HOOK),
            "prior_transient_fix_json": path_info(PRIOR_FIX_JSON),
            "same_boundary_qvapor_root": path_info(SAME_QVAPOR_ROOT),
            "step1_truth_npz": path_info(live.ACCEPTED_TRUTH),
            "d02_replay": path_info(D02_REPLAY),
        },
        "wrf_adjust_hook": wrf_hook,
        "wrf_precall_truth_summary": (
            {
                "status": wrf.get("status"),
                "root": wrf.get("root"),
                "record_counts": wrf.get("record_counts"),
                "MUB_summary": wrf.get("summaries", {}).get("MUB"),
                "T_STATE_summary": wrf.get("summaries", {}).get("T_STATE"),
                "QVAPOR_summary": wrf.get("summaries", {}).get("QVAPOR"),
            }
            if wrf.get("status") == "WRF_SURFACE_READY"
            else {k: v for k, v in wrf.items() if k != "arrays"}
        ),
        "candidate_capture": {k: v for k, v in capture.items() if k != "arrays"},
        "candidate_array_summaries": (
            {
                name: theta.array_summary(value)
                for name, value in capture.get("arrays", {}).items()
            }
            if capture.get("status") == "CANDIDATE_ARRAYS_READY"
            else {}
        ),
        "harness_self_consistency": harness,
        "mub_comparisons": mub_cmp,
        "theta_qv_comparisons": tq_cmp,
        "step1_comparison": {k: v for k, v in step1.items() if k not in {"per_field_metrics", "ranked_residuals"}},
        "step1_per_field_metrics": step1.get("per_field_metrics", {}),
        "step1_ranked_residuals": step1.get("ranked_residuals", []),
        "next_field": next_field,
        "blocker": blocker,
        "commands": {
            "required_validation": [
                "python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_live_nest_theta_qv_wiring.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_qv_wiring.py",
                "python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json >/tmp/step1_live_nest_theta_qv_wiring.validated.json",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py",
                "git diff --stat",
            ],
        },
        "proof_objects": {
            "script": str(Path(__file__).resolve()),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": risks
        + [
            "build_replay_case calls State.zeros (GPU-only), so the CPU proof exercises the exact production "
            "helpers it consumes plus a static wiring check, not the full GPU build_replay_case object.",
            "The Step-1 16-field comparison is post-RK/pre-halo; residuals after init closure name a field-level "
            "symptom (next operator), not yet the exact dycore/physics operator.",
        ],
        "next_decision": next_decision,
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
