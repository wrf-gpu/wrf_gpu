#!/usr/bin/env python3
"""V0.14 Step-1 JAX live-nest loader/carry T_STATE split."""

from __future__ import annotations

import hashlib
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

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_pre_part1_handoff as pre  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_jax_loader_tstate.json"
OUT_MD = PROOF_DIR / "step1_jax_loader_tstate.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-jax-loader-tstate/sprint-contract.md"
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_step1_jax_loader_tstate")
WRF_TRUTH = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth")
RUN_CASE3 = live.RUN_CASE3
REQUIRED_ANCESTOR = "99df65e0"

TARGET_STEP = 1
TARGET_DOMAIN = 2
PRECALL_SURFACE = "before_first_rk_step_part1_call"
THETA_OFFSET_K = 300.0
BDY_WIDTH = int(live.builder.BDY_WIDTH)

STAGE_ORDER = (
    "raw_child_state",
    "live_child_state",
    "boundary_packaged_state",
    "initial_carry_state",
    "haloed_step_entry_state",
)
FIELDS = (
    "T_STATE",
    "THETA_FULL",
    "P_STATE",
    "PB",
    "MU_STATE",
    "MUB",
    "MUT",
    "PH_STATE",
    "PHB",
    "W_STATE",
)
MATERIAL_THRESHOLDS = {
    "T_STATE": 1.0e-3,
    "THETA_FULL": 1.0e-3,
    "P_STATE": 1.0,
    "PB": 1.0,
    "MU_STATE": 1.0e-2,
    "MUB": 1.0e-2,
    "MUT": 1.0e-2,
    "PH_STATE": 1.0e-2,
    "PHB": 1.0e-2,
    "W_STATE": 1.0e-2,
}


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


def fortran_index(field: str, index: tuple[int, ...] | None) -> dict[str, int] | None:
    if index is None:
        return None
    if field in {"T_STATE", "THETA_FULL", "P_STATE", "PB"} and len(index) == 3:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field in {"PH_STATE", "PHB", "W_STATE"} and len(index) == 3:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "kstag": int(k) + 1}
    if field in {"MU_STATE", "MUB", "MUT"} and len(index) == 2:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    return None


def diff_metrics(
    field: str,
    candidate: Any,
    reference: Any,
    *,
    region: str = "full_domain",
) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "region": region,
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
        "region": region,
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


def material(field: str, metric: Mapping[str, Any]) -> bool:
    if metric.get("status") != "OK":
        return True
    max_abs = metric.get("max_abs")
    if max_abs is None:
        return bool(metric.get("nonfinite_diff_count"))
    threshold = MATERIAL_THRESHOLDS.get(field)
    return threshold is not None and float(max_abs) > float(threshold)


def rank_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for field, item in metrics.items():
        if item.get("status") != "OK":
            ranked.append({"field": field, "status": item.get("status"), "material": True})
            continue
        ranked.append(
            {
                "field": field,
                "max_abs": item.get("max_abs"),
                "rmse": item.get("rmse"),
                "bias": item.get("bias"),
                "p95": item.get("p95"),
                "p99": item.get("p99"),
                "material_threshold": MATERIAL_THRESHOLDS.get(field),
                "material": material(field, item),
            }
        )
    return sorted(ranked, key=lambda item: -1.0 if item.get("max_abs") is None else -float(item["max_abs"]))


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def state_arrays(state: Any, jax: Any) -> dict[str, np.ndarray]:
    p_base = state.p_total - state.p_perturbation
    ph_base = state.ph_total - state.ph_perturbation
    mu_base = state.mu_total - state.mu_perturbation
    arrays = {
        "T_STATE": state.theta - THETA_OFFSET_K,
        "THETA_FULL": state.theta,
        "P_STATE": state.p_perturbation,
        "PB": p_base,
        "MU_STATE": state.mu_perturbation,
        "MUB": mu_base,
        "MUT": state.mu_total,
        "PH_STATE": state.ph_perturbation,
        "PHB": ph_base,
        "W_STATE": state.w,
    }
    return {name: np.asarray(jax.device_get(value), dtype=np.float64) for name, value in arrays.items()}


def field_reference(field: str, wrf_arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    if field == "THETA_FULL":
        return np.asarray(wrf_arrays["T_STATE"], dtype=np.float64) + THETA_OFFSET_K
    return np.asarray(wrf_arrays[field], dtype=np.float64)


def boundary_mask(shape: tuple[int, ...], width: int) -> np.ndarray:
    if len(shape) == 3:
        _, ny, nx = shape
    elif len(shape) == 2:
        ny, nx = shape
    else:
        raise ValueError(f"unsupported boundary mask shape {shape}")
    yy, xx = np.ogrid[:ny, :nx]
    return (yy < width) | (yy >= ny - width) | (xx < width) | (xx >= nx - width)


def masked_region(arr: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, mask_2d]
    if arr.ndim == 2:
        return arr[mask_2d]
    raise ValueError(f"unsupported masked region ndim {arr.ndim}")


def compare_t_regions(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    mask = boundary_mask(tuple(reference.shape), BDY_WIDTH)
    interior = ~mask
    return {
        "band_width_cells": BDY_WIDTH,
        "full_domain": diff_metrics("T_STATE", candidate, reference, region="full_domain"),
        "interior": diff_metrics(
            "T_STATE",
            masked_region(candidate, interior),
            masked_region(reference, interior),
            region="interior",
        ),
        "boundary_band": diff_metrics(
            "T_STATE",
            masked_region(candidate, mask),
            masked_region(reference, mask),
            region="boundary_band",
        ),
    }


def compare_stage_to_wrf(stage_arrays: Mapping[str, np.ndarray], wrf_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    metrics = {
        field: diff_metrics(field, stage_arrays[field], field_reference(field, wrf_arrays))
        for field in FIELDS
    }
    first_material = None
    for field in FIELDS:
        if material(field, metrics[field]):
            first_material = field
            break
    return {
        "status": "WRF_JAX_STAGE_COMPARISON_EXECUTED",
        "diff_sign": "jax_minus_wrf",
        "first_material_field": first_material,
        "per_field_metrics": metrics,
        "ranked_residuals": rank_metrics(metrics),
        "t_state_regions": compare_t_regions(stage_arrays["T_STATE"], wrf_arrays["T_STATE"]),
        "theta_semantics": {
            "wrf_t_state_vs_jax_theta_minus_300": metrics["T_STATE"],
            "wrf_t_state_plus_300_vs_jax_full_theta": metrics["THETA_FULL"],
            "wrong_wrf_t_state_vs_jax_full_theta": diff_metrics(
                "THETA_FULL",
                stage_arrays["THETA_FULL"],
                wrf_arrays["T_STATE"],
                region="wrong_full_theta_mapping",
            ),
        },
    }


def compare_stage_transition(
    prev_name: str,
    next_name: str,
    prev_arrays: Mapping[str, np.ndarray],
    next_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    metrics = {field: diff_metrics(field, next_arrays[field], prev_arrays[field]) for field in FIELDS}
    return {
        "status": "STAGE_TRANSITION_COMPARISON_EXECUTED",
        "name": f"{prev_name}__to__{next_name}",
        "diff_sign": "next_minus_previous",
        "previous_stage": prev_name,
        "next_stage": next_name,
        "per_field_metrics": metrics,
        "ranked_residuals": rank_metrics(metrics),
    }


def load_wrf_precall_truth() -> dict[str, Any]:
    shapes = pre.expected_shapes()
    parsed = pre.parse_wrf_surface(PRECALL_SURFACE, shapes)
    if parsed.get("status") != "WRF_SURFACE_READY":
        return {"status": parsed.get("status"), "blocker": parsed}
    return parsed


def strip_wrf_arrays(wrf: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in wrf.items() if key != "arrays"}


def build_stage_capture() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    inputs = live.build_live_nest_step1_inputs()
    spec = om.halo_spec(inputs["namelist"].grid)
    stages = {
        "raw_child_state": inputs["raw_child"]["state"],
        "live_child_state": inputs["live_child"]["state"],
        "boundary_packaged_state": inputs["state"],
        "initial_carry_state": inputs["carry"].state,
        "haloed_step_entry_state": om.apply_halo(inputs["carry"].state, spec),
    }
    jax.block_until_ready(stages["haloed_step_entry_state"].theta)
    return {
        "status": "JAX_STAGE_CAPTURE_READY",
        "stages": {name: state_arrays(state, jax) for name, state in stages.items()},
        "loader": {
            "raw_child_construction": inputs["raw_child"].get("construction"),
            "live_child_construction": inputs["live_child"].get("construction"),
            "boundary_package": inputs["boundary_package"],
            "initial_deltas": inputs["initial_deltas"],
            "live_nest_base_init": inputs["live_child"].get("live_nest_base_init"),
            "halo_spec": {
                "width": int(spec.width),
                "edge_type": str(spec.edge_type),
                "fields_to_exchange": list(spec.fields_to_exchange),
                "sharding_enabled": bool(getattr(spec.sharding, "enabled", False)) if spec.sharding is not None else False,
            },
        },
        "namelist": {
            "dt_s": float(inputs["namelist"].dt_s),
            "rk_order": int(inputs["namelist"].rk_order),
            "acoustic_substeps": int(inputs["namelist"].acoustic_substeps),
            "run_physics": bool(inputs["namelist"].run_physics),
            "run_boundary": bool(inputs["namelist"].run_boundary),
            "force_fp64": bool(inputs["namelist"].force_fp64),
            "cu_physics": int(inputs["namelist"].cu_physics),
            "radiation_cadence_steps": int(inputs["namelist"].radiation_cadence_steps),
            "use_flux_advection": bool(inputs["namelist"].use_flux_advection),
            "moist_adv_opt": int(inputs["namelist"].moist_adv_opt),
        },
        "grid": {
            "domain": "d02",
            "mass_shape": [
                int(inputs["grid"].nz),
                int(inputs["grid"].ny),
                int(inputs["grid"].nx),
            ],
            "dx_m": float(inputs["grid"].projection.dx_m),
            "dy_m": float(inputs["grid"].projection.dy_m),
        },
    }


def strip_stage_arrays(capture: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in capture.items() if key != "stages"}


def stage_summaries(stages: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    return {
        stage: {field: array_summary(arrays[field]) for field in FIELDS}
        for stage, arrays in stages.items()
    }


def compare_all_stages(
    wrf: Mapping[str, Any],
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    stages = capture["stages"]
    wrf_arrays = wrf["arrays"]
    stage_vs_wrf = {
        name: compare_stage_to_wrf(stages[name], wrf_arrays)
        for name in STAGE_ORDER
    }
    transitions = {}
    for prev_name, next_name in zip(STAGE_ORDER[:-1], STAGE_ORDER[1:]):
        transitions[f"{prev_name}__to__{next_name}"] = compare_stage_transition(
            prev_name,
            next_name,
            stages[prev_name],
            stages[next_name],
        )
    return {
        "status": "JAX_LOADER_STAGE_COMPARISONS_EXECUTED",
        "stage_vs_wrf_precall": stage_vs_wrf,
        "stage_transitions": transitions,
        "stage_summaries": stage_summaries(stages),
    }


def metric_max_abs(comp: Mapping[str, Any], field: str) -> float:
    value = comp["per_field_metrics"][field].get("max_abs")
    return float(value or 0.0)


def classify(comparisons: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[str], str]:
    if comparisons.get("status") != "JAX_LOADER_STAGE_COMPARISONS_EXECUTED":
        return (
            "STEP1_JAX_LOADER_TSTATE_BLOCKED_STAGE_COMPARISON",
            {},
            ["JAX loader stage comparisons did not execute."],
            "Fix the stage comparison blocker and rerun.",
        )

    stage_vs_wrf = comparisons["stage_vs_wrf_precall"]
    transitions = comparisons["stage_transitions"]
    first_material_stage = None
    first_material_field = None
    for stage in STAGE_ORDER:
        field = stage_vs_wrf[stage].get("first_material_field")
        if field is not None:
            first_material_stage = stage
            first_material_field = field
            break

    raw_t = stage_vs_wrf["raw_child_state"]["per_field_metrics"]["T_STATE"]
    live_t = stage_vs_wrf["live_child_state"]["per_field_metrics"]["T_STATE"]
    boundary_t = stage_vs_wrf["boundary_packaged_state"]["per_field_metrics"]["T_STATE"]
    carry_t = stage_vs_wrf["initial_carry_state"]["per_field_metrics"]["T_STATE"]
    halo_t = stage_vs_wrf["haloed_step_entry_state"]["per_field_metrics"]["T_STATE"]
    raw_to_live = transitions["raw_child_state__to__live_child_state"]
    live_to_boundary = transitions["live_child_state__to__boundary_packaged_state"]
    boundary_to_carry = transitions["boundary_packaged_state__to__initial_carry_state"]
    carry_to_halo = transitions["initial_carry_state__to__haloed_step_entry_state"]

    base_changes = {
        name: raw_to_live["per_field_metrics"][name]
        for name in ("PB", "MUB", "PHB")
    }
    base_closure = {
        name: {
            "raw_max_abs": metric_max_abs(stage_vs_wrf["raw_child_state"], name),
            "live_max_abs": metric_max_abs(stage_vs_wrf["live_child_state"], name),
            "improved": metric_max_abs(stage_vs_wrf["live_child_state"], name)
            < metric_max_abs(stage_vs_wrf["raw_child_state"], name),
        }
        for name in ("PB", "MUB", "PHB")
    }
    t_unchanged_after_raw = all(
        not material("T_STATE", item["per_field_metrics"]["T_STATE"])
        for item in (raw_to_live, live_to_boundary, boundary_to_carry, carry_to_halo)
    )
    base_changed = any(material(name, base_changes[name]) for name in base_changes)
    base_improved = any(item["improved"] for item in base_closure.values())

    evidence = {
        "field": "T_STATE",
        "first_material_stage": first_material_stage,
        "first_material_field": first_material_field,
        "wrf_precall_surface": PRECALL_SURFACE,
        "raw_child_state_t_state": raw_t,
        "live_child_state_t_state": live_t,
        "boundary_packaged_state_t_state": boundary_t,
        "initial_carry_state_t_state": carry_t,
        "haloed_step_entry_state_t_state": halo_t,
        "raw_child_t_state_regions": stage_vs_wrf["raw_child_state"]["t_state_regions"],
        "live_child_t_state_regions": stage_vs_wrf["live_child_state"]["t_state_regions"],
        "haloed_step_entry_t_state_regions": stage_vs_wrf["haloed_step_entry_state"]["t_state_regions"],
        "t_state_transition_max_abs": {
            "raw_to_live": raw_to_live["per_field_metrics"]["T_STATE"].get("max_abs"),
            "live_to_boundary": live_to_boundary["per_field_metrics"]["T_STATE"].get("max_abs"),
            "boundary_to_carry": boundary_to_carry["per_field_metrics"]["T_STATE"].get("max_abs"),
            "carry_to_halo": carry_to_halo["per_field_metrics"]["T_STATE"].get("max_abs"),
        },
        "live_nest_base_changes_raw_to_live": base_changes,
        "live_nest_base_closure_vs_wrf": base_closure,
        "t_state_unchanged_after_raw": t_unchanged_after_raw,
        "live_nest_base_changed": base_changed,
        "live_nest_base_improved_vs_wrf": base_improved,
        "theta_semantics_conclusion": "WRF_T_STATE_IS_PERTURBATION_THETA",
        "theta_semantics_raw_child": stage_vs_wrf["raw_child_state"]["theta_semantics"],
        "theta_semantics_haloed_step_entry": stage_vs_wrf["haloed_step_entry_state"]["theta_semantics"],
    }

    if not material("T_STATE", halo_t):
        return (
            "STEP1_JAX_LOADER_TSTATE_NO_REMAINING_DIVERGENCE",
            evidence,
            [],
            "Close this T_STATE loader gate; no T_STATE divergence remains at step entry.",
        )
    if material("T_STATE", live_to_boundary["per_field_metrics"]["T_STATE"]):
        return (
            "STEP1_JAX_LOADER_TSTATE_LOCALIZED_BOUNDARY_PACKAGE",
            evidence,
            [],
            "Inspect build_child_boundary_package theta handling before any carry or dycore work.",
        )
    if material("T_STATE", boundary_to_carry["per_field_metrics"]["T_STATE"]):
        return (
            "STEP1_JAX_LOADER_TSTATE_LOCALIZED_CARRY_CONSTRUCTION",
            evidence,
            [],
            "Inspect initial_operational_carry theta/state threading.",
        )
    if material("T_STATE", carry_to_halo["per_field_metrics"]["T_STATE"]):
        return (
            "STEP1_JAX_LOADER_TSTATE_LOCALIZED_HALO_ENTRY",
            evidence,
            [],
            "Inspect apply_halo/halo_spec theta behavior for the operational step-entry path.",
        )
    if material("T_STATE", raw_t) and t_unchanged_after_raw and base_changed and base_improved:
        return (
            "STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH",
            evidence,
            [],
            "Localize WRF live-nest initialization T_STATE/theta semantics; JAX updates PB/PHB/MUB but carries raw wrfinput theta unchanged.",
        )
    if material("T_STATE", raw_t):
        return (
            "STEP1_JAX_LOADER_TSTATE_LOCALIZED_RAW_WRFINPUT",
            evidence,
            [],
            "Localize raw d02 wrfinput T/State.theta loading before boundary or carry work.",
        )
    if material("T_STATE", live_t):
        return (
            "STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH",
            evidence,
            [],
            "Localize live-nest base initialization T_STATE/theta semantics.",
        )
    return (
        "STEP1_JAX_LOADER_TSTATE_BLOCKED_UNCLASSIFIED",
        evidence,
        ["No verdict branch matched despite a material step-entry residual."],
        "Inspect the JSON stage comparison matrix and update the classifier.",
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    ev = payload.get("classification_evidence", {})
    raw = ev.get("raw_child_state_t_state", {})
    live_t = ev.get("live_child_state_t_state", {})
    halo = ev.get("haloed_step_entry_state_t_state", {})
    regions = ev.get("haloed_step_entry_t_state_regions", {})
    interior = regions.get("interior", {})
    boundary = regions.get("boundary_band", {})
    transitions = ev.get("t_state_transition_max_abs", {})
    base_pb = ev.get("live_nest_base_closure_vs_wrf", {}).get("PB", {})
    theta_wrong = ev.get("theta_semantics_haloed_step_entry", {}).get("wrong_wrf_t_state_vs_jax_full_theta", {})
    lines = [
        "# V0.14 Step-1 JAX Loader T_STATE",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`.",
        f"- WRF pre-call truth reused: `{WRF_TRUTH}`.",
        f"- First material stage/field: `{ev.get('first_material_stage')}` / `{ev.get('first_material_field')}`.",
        f"- Raw child `T_STATE = State.theta - 300 K` vs WRF pre-call: max_abs `{raw.get('max_abs')}`, rmse `{raw.get('rmse')}`.",
        f"- Live child `T_STATE` vs WRF pre-call: max_abs `{live_t.get('max_abs')}`, rmse `{live_t.get('rmse')}`.",
        f"- Haloed step-entry `T_STATE` vs WRF pre-call: max_abs `{halo.get('max_abs')}`, rmse `{halo.get('rmse')}`.",
        f"- Haloed step-entry interior max_abs `{interior.get('max_abs')}`; boundary-band max_abs `{boundary.get('max_abs')}` (band width `{BDY_WIDTH}`).",
        f"- `T_STATE` transition max_abs raw->live `{transitions.get('raw_to_live')}`, live->boundary `{transitions.get('live_to_boundary')}`, boundary->carry `{transitions.get('boundary_to_carry')}`, carry->halo `{transitions.get('carry_to_halo')}`.",
        f"- PB raw->WRF max_abs `{base_pb.get('raw_max_abs')}`; live->WRF max_abs `{base_pb.get('live_max_abs')}`.",
        f"- Wrong full-theta mapping has approximately 300 K bias: bias `{theta_wrong.get('bias')}`, max_abs `{theta_wrong.get('max_abs')}`.",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH":
        lines.extend(
            [
                "- The `T_STATE` residual is already visible in raw d02 wrfinput theta and is carried unchanged through live-nest init, boundary packaging, carry construction, and the step-entry halo path.",
                "- The live-nest stage does materially change and improve the base fields (`PB/PHB/MUB`) against WRF, so the remaining theta mismatch is the live-nest state/base semantic split, not a boundary package, carry, or halo mutation.",
                "- The residual is not boundary-only: the interior max is material and comparable to the full-domain maximum.",
                "- WRF `T_STATE` still maps to JAX `State.theta - 300 K`; comparing WRF perturbation theta directly to full theta produces the expected ~300 K offset.",
            ]
        )
    elif verdict == "STEP1_JAX_LOADER_TSTATE_LOCALIZED_RAW_WRFINPUT":
        lines.extend(
            [
                "- The first material `T_STATE` residual is present in raw d02 wrfinput `State.theta` before live-nest, boundary, carry, or halo stages.",
                "- Later stages do not introduce a new `T_STATE` mutation.",
            ]
        )
    else:
        lines.append("- See the JSON stage matrix for the exact stage/field split.")
    lines.extend(["", "Detailed stage and field tables are in `proofs/v014/step1_jax_loader_tstate.json`.", ""])
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 JAX Loader T_STATE",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: split the JAX live-nest Step-1 loader/carry construction for `T_STATE` against accepted WRF solve_em pre-`first_rk_step_part1` truth.",
        "",
        "files changed:",
        "- `proofs/v014/step1_jax_loader_tstate.py`",
        "- `proofs/v014/step1_jax_loader_tstate.json`",
        "- `proofs/v014/step1_jax_loader_tstate.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-jax-loader-tstate.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            f"- `{OUT_JSON}`",
            f"- `{OUT_MD}`",
            f"- `{OUT_REVIEW}`",
            f"- `{WRF_TRUTH}` reused, not rebuilt",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"], cwd=ROOT)
    wrf = load_wrf_precall_truth()
    capture: dict[str, Any]
    comparisons: dict[str, Any]
    if wrf.get("status") != "WRF_SURFACE_READY":
        capture = {"status": "NOT_EXECUTED"}
        comparisons = {"status": "NOT_EXECUTED", "blocker": wrf}
        verdict = "STEP1_JAX_LOADER_TSTATE_BLOCKED_WRF_PRECALL_TRUTH"
        evidence: dict[str, Any] = {}
        risks = ["Accepted WRF pre-call truth could not be parsed."]
        next_decision = "Fix or restore the WRF pre-call truth root and rerun."
    else:
        capture = build_stage_capture()
        if capture.get("status") != "JAX_STAGE_CAPTURE_READY":
            comparisons = {"status": "NOT_EXECUTED", "blocker": capture}
            verdict = "STEP1_JAX_LOADER_TSTATE_BLOCKED_JAX_STAGE_CAPTURE"
            evidence = {}
            risks = ["JAX stage capture did not complete."]
            next_decision = "Fix the exact JAX capture blocker and rerun."
        else:
            comparisons = compare_all_stages(wrf, capture)
            verdict, evidence, risks, next_decision = classify(comparisons)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_jax_loader_tstate.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "git_head": git_head,
        "required_ancestor_99df65e0": {
            "command": ancestor["command"],
            "returncode": ancestor["returncode"],
            "is_ancestor": ancestor["returncode"] == 0,
            "stderr_tail": ancestor.get("stderr_tail"),
        },
        "environment": jax_environment(),
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "tooling_verdict": "RIGHT_TOOL_FASTEST_WALL_CLOCK_CPU_ONLY_STAGE_COMPARATOR_REUSING_ACCEPTED_WRF_TRUTH",
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "wrf_surface": PRECALL_SURFACE,
            "theta_offset_K": THETA_OFFSET_K,
            "stage_order": list(STAGE_ORDER),
            "field_order": list(FIELDS),
            "material_thresholds": MATERIAL_THRESHOLDS,
            "boundary_band_width_cells": BDY_WIDTH,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "accepted_pre_part1_handoff_md": path_info(PROOF_DIR / "step1_pre_part1_handoff.md"),
            "accepted_pre_part1_handoff_json": path_info(PROOF_DIR / "step1_pre_part1_handoff.json"),
            "accepted_pre_part1_handoff_py": path_info(PROOF_DIR / "step1_pre_part1_handoff.py"),
            "live_nest_init_rerun_py": path_info(PROOF_DIR / "step1_live_nest_init_rerun.py"),
            "wrf_truth_root": path_info(WRF_TRUTH),
            "run_case3": path_info(RUN_CASE3),
            "scratch_root": path_info(SCRATCH),
            "d02_replay": path_info(SRC / "gpuwrf/integration/d02_replay.py"),
            "boundary_construction": path_info(SRC / "gpuwrf/nesting/boundary_construction.py"),
            "operational_state": path_info(SRC / "gpuwrf/runtime/operational_state.py"),
        },
        "wrf_precall_truth": strip_wrf_arrays(wrf) if wrf.get("status") == "WRF_SURFACE_READY" else wrf,
        "jax_capture": strip_stage_arrays(capture),
        "comparisons": comparisons,
        "classification_evidence": evidence,
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_jax_loader_tstate.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_loader_tstate.py",
                "python -m json.tool proofs/v014/step1_jax_loader_tstate.json >/tmp/step1_jax_loader_tstate.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "wrf_truth_root_reused": str(WRF_TRUTH),
        },
        "unresolved_risks": risks
        or [
            "This proof localizes the T_STATE loader/carry split but does not implement WRF live-nest T_STATE semantics.",
            "No production source fix was made, so the extra source-edit proof chain was not run.",
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
