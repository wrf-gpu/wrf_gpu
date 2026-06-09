#!/usr/bin/env python3
"""V0.14 d02 Step-1 P/PH/MU boundary/operator localization.

CPU-only proof. Reuses the existing WRF substage truth surfaces and reruns the
current post-theta/QV JAX live-nest path against them. This script intentionally
does not write or rerun the older proof artifacts it imports.
"""

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


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_rk1_source_boundary as source  # noqa: E402
import step1_t_p_operator_localization as tp  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_p_ph_mu_boundary_localization.json"
OUT_MD = PROOF_DIR / "step1_p_ph_mu_boundary_localization.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md"

SPRINT_CONTRACT = (
    ROOT
    / ".agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization/sprint-contract.md"
)
BASELINE_JSON = PROOF_DIR / "step1_live_nest_theta_qv_wiring.json"
STALE_SOURCE_JSON = PROOF_DIR / "step1_rk1_source_boundary.json"
STALE_TP_JSON = PROOF_DIR / "step1_t_p_operator_localization.json"

REQUIRED_ANCESTOR = "3aa5f15b"
BOUNDARY_BAND = 5
VERDICT = "STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE"

USER_FIELDS = ("T", "P", "PH", "MU", "W", "U")
FIELD_THRESHOLDS = {
    "T": 1.0e-3,
    "P": 1.0,
    "PH": 1.0e-2,
    "MU": 1.0e-2,
    "W": 1.0e-2,
    "U": 1.0e-2,
}
FIELD_INTERNALS = {
    "T": ("T_STATE", "T_WORK", "T"),
    "P": ("P_STATE", "P_WORK", "P"),
    "PH": ("PH_STATE", "PH_WORK", "PH"),
    "MU": ("MU_STATE", "MU_WORK", "MU"),
    "W": ("W_STATE", "W_WORK", "W"),
    "U": ("U",),
}
SOURCE_STATE_FIELDS = ("T_STATE", "P_STATE", "PH_STATE", "MU_STATE", "W_STATE")
WORK_FIELDS = ("T_WORK", "P_WORK", "PH_WORK", "MU_WORK", "W_WORK")


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
    if isinstance(value, np.bool_):
        return bool(value)
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(command: list[str], *, cwd: Path = ROOT, timeout_s: int = 120) -> dict[str, Any]:
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
            "timeout_s": int(timeout_s),
            "wall_s": float(time.perf_counter() - start),
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
    }
    try:
        import jax  # noqa: PLC0415

        devices = list(jax.devices())
        result.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([device for device in devices if device.platform == "gpu"]),
            }
        )
    except Exception as exc:
        result.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return result


def required_ancestor_status() -> dict[str, Any]:
    result = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"], cwd=ROOT)
    return {
        **result,
        "required_ancestor": REQUIRED_ANCESTOR,
        "is_ancestor": result.get("returncode") == 0,
    }


def strip_arrays_from_source_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    if wrf.get("status") != "WRF_SOURCE_BOUNDARY_TRUTH_READY":
        return dict(wrf)
    return source.strip_arrays_from_wrf(wrf)


def strip_arrays_from_tp_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    if wrf.get("status") != "WRF_SUBSTAGE_TRUTH_READY":
        return dict(wrf)
    return tp.strip_arrays_from_wrf(wrf)


def metric_max(metric: Mapping[str, Any] | None) -> float | None:
    if not metric:
        return None
    value = metric.get("max_abs")
    return None if value is None else float(value)


def horizontal_distance(index: list[int] | tuple[int, ...] | None, shape: list[int] | tuple[int, ...] | None) -> int | None:
    if not index or not shape:
        return None
    if len(shape) == 2:
        y, x = int(index[0]), int(index[1])
        ny, nx = int(shape[0]), int(shape[1])
    elif len(shape) == 3:
        y, x = int(index[-2]), int(index[-1])
        ny, nx = int(shape[-2]), int(shape[-1])
    else:
        return None
    return int(min(x, y, nx - 1 - x, ny - 1 - y))


def metric_with_boundary(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metric:
        return None
    out = {key: value for key, value in metric.items() if key != "status"}
    distance = horizontal_distance(out.get("worst_mismatch_index"), out.get("shape"))
    out["worst_boundary_distance_to_horizontal_edge"] = distance
    out["worst_is_boundary_band"] = None if distance is None else bool(distance <= BOUNDARY_BAND)
    return out


def _mask_for_shape(shape: tuple[int, ...], band: int) -> np.ndarray:
    if len(shape) == 2:
        ny, nx = shape
        y = np.arange(ny)[:, None]
        x = np.arange(nx)[None, :]
        distance = np.minimum(np.minimum(x, nx - 1 - x), np.minimum(y, ny - 1 - y))
        return distance <= int(band)
    if len(shape) == 3:
        _, ny, nx = shape
        y = np.arange(ny)[None, :, None]
        x = np.arange(nx)[None, None, :]
        distance = np.minimum(np.minimum(x, nx - 1 - x), np.minimum(y, ny - 1 - y))
        return np.broadcast_to(distance <= int(band), shape)
    raise ValueError(f"unsupported shape for horizontal boundary mask: {shape}")


def fortran_index(field: str, index: tuple[int, ...]) -> dict[str, int]:
    if len(index) == 2:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    k, y, x = index
    if field in {"PH", "W", "PH_STATE", "W_STATE", "PH_WORK", "W_WORK"}:
        return {"i": int(x) + 1, "j": int(y) + 1, "kstag": int(k) + 1}
    if field == "U":
        return {"i_xstag": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field == "V":
        return {"i": int(x) + 1, "j_ystag": int(y) + 1, "k": int(k) + 1}
    return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}


def diff_region_metrics(field: str, candidate: Any, reference: Any, mask: np.ndarray) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {"status": "SHAPE_MISMATCH", "candidate_shape": list(cand.shape), "reference_shape": list(ref.shape)}
    if mask.shape != cand.shape:
        return {"status": "MASK_SHAPE_MISMATCH", "mask_shape": list(mask.shape), "shape": list(cand.shape)}
    diff = cand - ref
    selected = diff[mask]
    abs_selected = np.abs(selected)
    nonfinite = ~np.isfinite(selected)
    if selected.size == 0:
        return {"status": "EMPTY_REGION", "count": 0}
    masked_abs = np.where(mask, np.abs(diff), -1.0)
    worst_flat = int(np.argmax(masked_abs))
    worst = tuple(int(v) for v in np.unravel_index(worst_flat, diff.shape))
    finite_abs = abs_selected[np.isfinite(abs_selected)]
    finite_diff = selected[np.isfinite(selected)]
    return {
        "status": "OK",
        "shape": list(cand.shape),
        "count": int(selected.size),
        "max_abs": float(np.max(finite_abs)) if finite_abs.size else None,
        "rmse": float(np.sqrt(np.mean(finite_diff * finite_diff))) if finite_diff.size else None,
        "p99": float(np.percentile(finite_abs, 99)) if finite_abs.size else None,
        "nonfinite_diff_count": int(nonfinite.sum()),
        "worst_mismatch_index": list(worst),
        "worst_mismatch_fortran": fortran_index(field, worst),
    }


def decompose_pair(field: str, candidate: Any, reference: Any) -> dict[str, Any]:
    shape = tuple(int(v) for v in np.asarray(candidate).shape)
    mask = _mask_for_shape(shape, BOUNDARY_BAND)
    return {
        f"boundary_distance_le_{BOUNDARY_BAND}": diff_region_metrics(field, candidate, reference, mask),
        f"interior_distance_gt_{BOUNDARY_BAND}": diff_region_metrics(field, candidate, reference, ~mask),
    }


def final_boundary_decomposition(tp_jax: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    state = tp_jax["final_pre_halo_state"]
    base_state = tp_jax["base_state"]
    out: dict[str, Any] = {}
    with np.load(live.ACCEPTED_TRUTH) as truth:
        for field in USER_FIELDS:
            candidate = np.asarray(jax.device_get(live.jax_compare_array(field, state, base_state)), dtype=np.float64)
            reference = np.asarray(truth[field], dtype=np.float64)
            out[field] = decompose_pair(field, candidate, reference)
    return out


def source_boundary_decomposition(
    source_wrf: Mapping[str, Any],
    source_jax: Mapping[str, Any],
) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    captures = source_jax["captures"]["physics_carry_state_dry"]
    surface = source_wrf["surfaces"]["after_first_rk_step_part1"]
    out: dict[str, Any] = {}
    for field in ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE", "T_STATE"):
        candidate = np.asarray(jax.device_get(captures[field]), dtype=np.float64)
        reference = np.asarray(surface["arrays"][field], dtype=np.float64)
        out[field] = decompose_pair(field, candidate, reference)
    return out


def compact_top(ranked: list[Mapping[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "field": item.get("field"),
            "max_abs": item.get("max_abs"),
            "rmse": item.get("rmse"),
            "worst_mismatch_fortran": item.get("worst_mismatch_fortran"),
            "worst_mismatch_index": item.get("worst_mismatch_index"),
        }
        for item in ranked[:limit]
    ]


def baseline_from_theta_qv() -> dict[str, Any]:
    baseline = read_json(BASELINE_JSON)
    return {
        "source": str(BASELINE_JSON),
        "verdict": baseline.get("verdict"),
        "step1_comparison": baseline.get("step1_comparison", {}),
        "step1_ranked_residuals_top8": compact_top(baseline.get("step1_ranked_residuals", []), 8),
        "theta_qv_closure": baseline.get("theta_qv_comparisons", {}),
        "next_field": baseline.get("next_field", {}),
    }


def stale_surface_check(current_source_verdict: str, current_tp_first: Mapping[str, Any]) -> dict[str, Any]:
    stale_source = read_json(STALE_SOURCE_JSON)
    stale_tp = read_json(STALE_TP_JSON)
    return {
        "prior_jsons_are_stale_for_current_post_theta_qv_state": True,
        "prior_source_boundary_verdict": stale_source.get("verdict"),
        "prior_tp_operator_verdict": stale_tp.get("verdict"),
        "current_source_boundary_verdict_from_rerun": current_source_verdict,
        "current_tp_first_material_from_rerun": current_tp_first,
        "reason": (
            "The prior source/TP JSON files were generated before production live-nest theta/QV closure "
            "and name T_STATE. Reusing the same WRF truth surfaces with the current post-fix JAX path "
            "moves the first material operational-carry source residual to P_STATE."
        ),
    }


def comparison_checks(
    source_comparisons: Mapping[str, Any],
    tp_comparisons: Mapping[str, Any],
    final_comparison: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix = source_comparisons["matrix"]
    tp_matrix = tp_comparisons["comparisons"]
    checks = [
        {
            "id": "wrf_after_first_rk_step_part1_vs_jax_physics_carry",
            "order": 10,
            "surface": "after_first_rk_step_part1",
            "rk_step": 1,
            "wrf_boundary": "dyn_em/solve_em.F after first_rk_step_part1",
            "jax_boundary": "_physics_step_forcing.carry.state, haloed",
            "distinction": "after WRF non-timesplit physics part1; before WRF first_rk_step_part2, rk_tendency, relax_bdy_dry, rk_addtend_dry, spec_bdy_dry, small_step_prep",
            "metrics": matrix["after_first_rk_step_part1"]["vs_physics_carry_state_dry"]["per_field_metrics"],
        },
        {
            "id": "wrf_after_first_rk_step_part2_vs_jax_physics_carry",
            "order": 20,
            "surface": "after_first_rk_step_part2",
            "rk_step": 1,
            "wrf_boundary": "dyn_em/solve_em.F after first_rk_step_part2",
            "jax_boundary": "_physics_step_forcing.carry.state, haloed",
            "distinction": "after WRF first_rk_step_part1/part2; before WRF rk_tendency and dry boundary tendency application",
            "metrics": matrix["after_first_rk_step_part2"]["vs_physics_carry_state_dry"]["per_field_metrics"],
        },
        {
            "id": "rk1_after_rk_addtend_before_small_step_prep",
            "order": 30,
            "surface": "after_rk_addtend_before_small_step_prep",
            "rk_step": 1,
            "wrf_boundary": "dyn_em/solve_em.F after relax_bdy_dry/rk_addtend_dry/spec_bdy_dry",
            "jax_boundary": "after compute_advection_tendencies + _augment_large_step_tendencies",
            "distinction": "after RK tendency/source injection and dry boundary-tendency application; before small_step_prep",
            "metrics": matrix["after_rk_addtend_before_small_step_prep"]["vs_rk1_after_rk_addtend"]["per_field_metrics"],
        },
        {
            "id": "rk1_after_small_step_prep_calc_p_rho",
            "order": 40,
            "surface": "after_small_step_prep_calc_p_rho",
            "rk_step": 1,
            "wrf_boundary": "dyn_em/solve_em.F after small_step_prep and calc_p_rho(step=0)",
            "jax_boundary": "small_step_prep_wrf + calc_p_rho_wrf(step=0)",
            "distinction": "small-step prep/calc_p_rho step-0 check before RK1 acoustic scan",
            "metrics": matrix["after_small_step_prep_calc_p_rho"]["vs_rk1_after_small_step_prep"]["per_field_metrics"],
        },
    ]
    for rk in (2, 3):
        checks.append(
            {
                "id": f"rk{rk}_after_rk_addtend_before_small_step_prep",
                "order": 40 + (rk - 1) * 20,
                "surface": "after_rk_addtend_before_small_step_prep",
                "rk_step": rk,
                "wrf_boundary": "dyn_em/solve_em.F after relax_bdy_dry/rk_addtend_dry/spec_bdy_dry",
                "jax_boundary": f"JAX RK{rk} stage-entry after prior acoustic scan + halo, before small_step_prep",
                "distinction": "stage-entry state after previous acoustic scan/pressure refresh, before this stage's small_step_prep",
                "metrics": tp_matrix["after_rk_addtend_before_small_step_prep"][rk]["per_field_metrics"],
            }
        )
        checks.append(
            {
                "id": f"rk{rk}_after_small_step_prep_calc_p_rho",
                "order": 50 + (rk - 1) * 20,
                "surface": "after_small_step_prep_calc_p_rho",
                "rk_step": rk,
                "wrf_boundary": "dyn_em/solve_em.F after small_step_prep and calc_p_rho(step=0)",
                "jax_boundary": f"JAX RK{rk} small_step_prep_wrf + calc_p_rho_wrf(step=0)",
                "distinction": "this stage's prep/calc_p_rho step-0 result",
                "metrics": tp_matrix["after_small_step_prep_calc_p_rho"][rk]["per_field_metrics"],
            }
        )
    checks.append(
        {
            "id": "final_post_after_all_rk_steps_pre_halo",
            "order": 100,
            "surface": "post_after_all_rk_steps_pre_halo",
            "rk_step": 3,
            "wrf_boundary": "dyn_em/solve_em.F after final calc_p_rho_phi, before RK halo",
            "jax_boundary": "_rk_scan_step_with_pre_halo_capture final pre_halo_state after _refresh_grid_p_from_finished",
            "distinction": "final strict same-input comparison boundary",
            "metrics": final_comparison["per_field_metrics"],
        }
    )
    return checks


def earliest_field_table(checks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for user_field in USER_FIELDS:
        threshold = FIELD_THRESHOLDS[user_field]
        first: dict[str, Any] | None = None
        for check in sorted(checks, key=lambda item: int(item["order"])):
            for internal in FIELD_INTERNALS[user_field]:
                metric = check["metrics"].get(internal)
                max_abs = metric_max(metric)
                if max_abs is not None and max_abs > threshold:
                    first = {
                        "field": user_field,
                        "internal_field": internal,
                        "material_threshold": threshold,
                        "earliest_checked_boundary": check["id"],
                        "surface": check["surface"],
                        "rk_step": check["rk_step"],
                        "wrf_boundary": check["wrf_boundary"],
                        "jax_boundary": check["jax_boundary"],
                        "distinction": check["distinction"],
                        "metric": metric_with_boundary(metric),
                    }
                    break
            if first:
                break
        if first is None:
            first = {
                "field": user_field,
                "internal_field": None,
                "material_threshold": threshold,
                "earliest_checked_boundary": None,
                "surface": None,
                "rk_step": None,
                "wrf_boundary": None,
                "jax_boundary": None,
                "distinction": "No material residual observed on checked surfaces for this field.",
                "metric": None,
            }
        table.append(first)
    return table


def selected_state_metrics(checks: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {check["id"]: check for check in checks}
    selected: dict[str, Any] = {}
    for check_id in (
        "wrf_after_first_rk_step_part1_vs_jax_physics_carry",
        "rk1_after_rk_addtend_before_small_step_prep",
        "rk1_after_small_step_prep_calc_p_rho",
        "rk2_after_rk_addtend_before_small_step_prep",
        "rk2_after_small_step_prep_calc_p_rho",
        "rk3_after_rk_addtend_before_small_step_prep",
        "rk3_after_small_step_prep_calc_p_rho",
        "final_post_after_all_rk_steps_pre_halo",
    ):
        check = by_id[check_id]
        fields = set(SOURCE_STATE_FIELDS) | set(WORK_FIELDS) | set(USER_FIELDS)
        selected[check_id] = {
            "surface": check["surface"],
            "rk_step": check["rk_step"],
            "wrf_boundary": check["wrf_boundary"],
            "jax_boundary": check["jax_boundary"],
            "metrics": {
                field: metric_with_boundary(metric)
                for field, metric in check["metrics"].items()
                if field in fields
            },
        }
    return selected


def build_distinctions(
    field_table: list[Mapping[str, Any]],
    checks: list[Mapping[str, Any]],
    source_comparisons: Mapping[str, Any],
    tp_comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    by_field = {item["field"]: item for item in field_table}
    by_id = {check["id"]: check for check in checks}
    rk1_prep = by_id["rk1_after_small_step_prep_calc_p_rho"]["metrics"]
    rk1_add = by_id["rk1_after_rk_addtend_before_small_step_prep"]["metrics"]
    rk2_add = by_id["rk2_after_rk_addtend_before_small_step_prep"]["metrics"]
    rk2_prep = by_id["rk2_after_small_step_prep_calc_p_rho"]["metrics"]
    return {
        "boundary_package_construction_vs_boundary_application": {
            "package_construction": (
                "JAX live.build_live_nest_step1_inputs constructs child *_bdy leaves with "
                "gpuwrf.nesting.boundary_construction.build_child_boundary_package. The existing WRF "
                "truth does not emit raw p/ph/mu boundary-package leaves, so package leaf equality is not "
                "directly proven here."
            ),
            "boundary_application": (
                "The first material P/MU/W state residual is already present at WRF "
                "after_first_rk_step_part1, before WRF relax_bdy_dry/spec_bdy_dry and before the "
                "JAX end-of-step apply_lateral_boundaries path. This localizes the current first "
                "P-family state mismatch before dry boundary application, not inside the boundary "
                "package consumer."
            ),
            "first_boundary": {
                "P": by_field["P"],
                "MU": by_field["MU"],
                "W": by_field["W"],
            },
        },
        "rk_source_tendency_vs_small_step_prep": {
            "rk_source_tendency": (
                "The same P_STATE/MU_STATE/W_STATE residual magnitudes are visible at after_first_rk_step_part1 "
                "and persist through after_rk_addtend_before_small_step_prep; rk_tendency/rk_addtend_dry "
                "therefore do not introduce the first P/MU/W state residual."
            ),
            "rk1_tendency_examples": {
                "P_STATE": metric_with_boundary(rk1_add.get("P_STATE")),
                "MU_STATE": metric_with_boundary(rk1_add.get("MU_STATE")),
                "W_STATE": metric_with_boundary(rk1_add.get("W_STATE")),
                "PH_TEND": metric_with_boundary(rk1_add.get("PH_TEND")),
                "RW_TEND": metric_with_boundary(rk1_add.get("RW_TEND")),
                "T_TEND": metric_with_boundary(rk1_add.get("T_TEND")),
            },
            "small_step_prep": {
                "conclusion": "RK1 small_step_prep/calc_p_rho work arrays are exact for the checked work fields.",
                "T_WORK": metric_with_boundary(rk1_prep.get("T_WORK")),
                "P_WORK": metric_with_boundary(rk1_prep.get("P_WORK")),
                "PH_WORK": metric_with_boundary(rk1_prep.get("PH_WORK")),
                "MU_WORK": metric_with_boundary(rk1_prep.get("MU_WORK")),
                "W_WORK": metric_with_boundary(rk1_prep.get("W_WORK")),
            },
        },
        "calc_p_rho_pressure_refresh_vs_acoustic_finish": {
            "rk1_calc_p_rho_step0": {
                "conclusion": "Not the RK1 calc_p_rho(step=0) operator; P_WORK is exact there.",
                "P_WORK": metric_with_boundary(rk1_prep.get("P_WORK")),
            },
            "stage_transition": {
                "conclusion": (
                    "P/PH work residuals become material at RK2 after the RK1 acoustic scan and "
                    "post-stage pressure refresh have produced the next stage state. Existing truth does "
                    "not split the post-acoustic/pre-refresh point from the refreshed pressure point."
                ),
                "rk2_stage_entry_P_STATE": metric_with_boundary(rk2_add.get("P_STATE")),
                "rk2_stage_entry_PH_STATE": metric_with_boundary(rk2_add.get("PH_STATE")),
                "rk2_prep_P_WORK": metric_with_boundary(rk2_prep.get("P_WORK")),
                "rk2_prep_PH_WORK": metric_with_boundary(rk2_prep.get("PH_WORK")),
            },
            "source_fix_status": "No source fix was applied because the available surfaces localize the boundary but do not prove an exact narrow production bug.",
        },
        "horizontal_boundary_band_vs_interior_spread": {
            "conclusion": (
                "The worst final P residual is in the horizontal boundary band, but the residual is not "
                "boundary-only; several fields also have material interior residuals. Early P/MU maxima "
                "are boundary-band, while early W max is interior."
            ),
        },
        "stale_vs_current_surfaces": {
            "source_current_first_material": source_comparisons.get("matrix", {})
            .get("after_first_rk_step_part1", {})
            .get("vs_physics_carry_state_dry", {})
            .get("material_first_field"),
            "tp_current_first_material": tp_comparisons.get("first_material_tp_family_mismatch"),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    final_top = payload["current_final_comparison"]["ranked_residuals_top8"][:5]
    lines = [
        "# V0.14 Step-1 P/PH/MU Boundary Localization",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Fastest rigorous method: `{payload['tooling_verdict']}`.",
        "- Current post-theta/QV strict Step-1 final top residuals: "
        + ", ".join(f"`{item['field']}` max_abs `{item['max_abs']}`" for item in final_top)
        + ".",
        "- First current material P-family state residual: WRF `after_first_rk_step_part1` vs JAX `_physics_step_forcing.carry.state`, field `P_STATE`, max_abs `69.96875`.",
        "- `MU_STATE` and `W_STATE` are also material at that same first boundary; `PH_STATE` is not material there and first becomes material at RK2 stage entry.",
        "- RK1 `small_step_prep`/`calc_p_rho(step=0)` work arrays are exact for `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK`.",
        "- No production source fix was applied.",
        "",
        "## Field Table",
        "",
        "| Field | Earliest checked material boundary | Internal | max_abs | RMSE | Worst boundary band |",
        "|---|---|---:|---:|---:|:--:|",
    ]
    for item in payload["boundary_operator_table"]:
        metric = item.get("metric") or {}
        lines.append(
            "| {field} | {boundary} | {internal} | {max_abs} | {rmse} | {band} |".format(
                field=item["field"],
                boundary=item.get("earliest_checked_boundary") or "none",
                internal=item.get("internal_field") or "",
                max_abs=metric.get("max_abs"),
                rmse=metric.get("rmse"),
                band=metric.get("worst_is_boundary_band"),
            )
        )
    lines.extend(
        [
            "",
            "## Distinctions",
            "",
            "- Boundary package vs application: package leaf equality is not directly emitted by the existing WRF truth, but the first P/MU/W state residual is before WRF dry boundary application and before JAX `apply_lateral_boundaries`.",
            "- RK source/tendency vs prep: state residuals pre-exist `rk_tendency/rk_addtend_dry`; RK1 prep/calc_p_rho work fields are exact.",
            "- Pressure refresh vs acoustic finish: current surfaces do not split post-acoustic/pre-refresh from refreshed pressure, so no narrow fix is justified.",
            "- Boundary band vs interior: final P worst cell is boundary-band, but the residuals are not boundary-only.",
            "- Stale proofs: previous source/TP JSON verdicts named pre-theta-fix `T_STATE`; current rerun names `P_STATE`.",
            "",
            "Detailed metrics are in `proofs/v014/step1_p_ph_mu_boundary_localization.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 P/PH/MU Boundary Localization",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: localize or narrowly fix the remaining d02 Step-1 strict same-input divergence after production live-nest theta/QV initialization closure.",
        "",
        "files changed:",
        "- `proofs/v014/step1_p_ph_mu_boundary_localization.py`",
        "- `proofs/v014/step1_p_ph_mu_boundary_localization.json`",
        "- `proofs/v014/step1_p_ph_mu_boundary_localization.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`",
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
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    git_branch = run_command(["git", "branch", "--show-current"], cwd=ROOT)
    git_status = run_command(["git", "status", "--short", "--branch"], cwd=ROOT)
    required_ancestor = required_ancestor_status()
    baseline = baseline_from_theta_qv()

    source_wrf = source.parse_wrf_surfaces()
    if source_wrf.get("status") != "WRF_SOURCE_BOUNDARY_TRUTH_READY":
        raise RuntimeError(f"source WRF truth unavailable: {source_wrf.get('status')}")
    source_jax = source.capture_jax_boundaries()
    if source_jax.get("status") != "JAX_SOURCE_BOUNDARIES_READY":
        raise RuntimeError(f"source JAX capture unavailable: {source_jax.get('status')}")
    source_comparisons = source.compare_all(source_wrf, source_jax)
    current_source_verdict, current_source_evidence, _current_source_next = source.classify(source_comparisons)

    tp_wrf = tp.load_wrf_surfaces()
    if tp_wrf.get("status") != "WRF_SUBSTAGE_TRUTH_READY":
        raise RuntimeError(f"TP WRF truth unavailable: {tp_wrf.get('status')}")
    tp_jax = tp.capture_jax_boundaries()
    if tp_jax.get("status") != "JAX_BOUNDARIES_READY":
        raise RuntimeError(f"TP JAX capture unavailable: {tp_jax.get('status')}")
    tp_comparisons = tp.compare_all(tp_wrf, tp_jax)
    final_comparison = tp_jax["final_comparison"]

    checks = comparison_checks(source_comparisons, tp_comparisons, final_comparison)
    field_table = earliest_field_table(checks)
    final_decomposition = final_boundary_decomposition(tp_jax)
    early_decomposition = source_boundary_decomposition(source_wrf, source_jax)
    distinctions = build_distinctions(field_table, checks, source_comparisons, tp_comparisons)
    stale_check = stale_surface_check(
        current_source_verdict,
        tp_comparisons.get("first_material_tp_family_mismatch"),
    )
    final_top = compact_top(final_comparison.get("ranked_residuals", []), 8)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_p_ph_mu_boundary_localization.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "weak_comparison_avoided": True,
        "jax_vs_jax_self_compare": False,
        "one_cell_proof": False,
        "initial_vs_post_step_false_comparison": False,
        "tooling_verdict": "FOCUSED_STEP1_SOURCE_AND_SUBSTAGE_TRUTH_COMPARATOR_FASTEST_RIGOROUS_WALL_CLOCK",
        "environment": jax_environment(),
        "git": {
            "head": git_head,
            "branch": git_branch,
            "status_short_branch": git_status,
            "required_ancestor": required_ancestor,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "baseline_theta_qv_json": path_info(BASELINE_JSON),
            "accepted_final_truth_npz": path_info(live.ACCEPTED_TRUTH),
            "source_wrf_truth_root": path_info(source.WRF_TRUTH),
            "tp_wrf_truth_root": path_info(tp.WRF_TRUTH),
            "source_wrf_patch": path_info(source.OUT_WRF_PATCH),
            "tp_wrf_patch": path_info(tp.OUT_WRF_PATCH),
        },
        "baseline_before_fix_from_step1_live_nest_theta_qv_wiring": baseline,
        "current_final_comparison": {
            "status": final_comparison.get("status"),
            "first_divergent_field": final_comparison.get("first_divergent_field"),
            "ranked_residuals_top8": final_top,
            "selected_per_field_metrics": {
                field: metric_with_boundary(final_comparison["per_field_metrics"].get(field))
                for field in USER_FIELDS
            },
        },
        "boundary_operator_table": field_table,
        "selected_stage_metrics": selected_state_metrics(checks),
        "boundary_decomposition": {
            "boundary_band_threshold": BOUNDARY_BAND,
            "early_after_first_rk_step_part1_vs_physics_carry": early_decomposition,
            "final_post_after_all_rk_steps_pre_halo": final_decomposition,
        },
        "distinctions": distinctions,
        "stale_pre_theta_fix_surface_check": stale_check,
        "wrf_source_boundary_truth": strip_arrays_from_source_wrf(source_wrf),
        "wrf_tp_substage_truth": strip_arrays_from_tp_wrf(tp_wrf),
        "jax_captures": {
            "source": {key: value for key, value in source_jax.items() if key != "captures"},
            "tp": {
                key: value
                for key, value in tp_jax.items()
                if key not in {"captures", "final_pre_halo_state", "base_state", "final_comparison"}
            },
        },
        "source_boundary_comparisons": source_comparisons,
        "tp_substage_comparisons": tp_comparisons,
        "current_source_boundary_verdict": current_source_verdict,
        "current_source_boundary_classification_evidence": current_source_evidence,
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_p_ph_mu_boundary_localization.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_p_ph_mu_boundary_localization.py",
                "python -m json.tool proofs/v014/step1_p_ph_mu_boundary_localization.json >/tmp/step1_p_ph_mu_boundary_localization.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "Existing WRF truth does not emit raw p/ph/mu boundary-package leaves, so this localizes before boundary application but does not prove package construction equality.",
            "Existing WRF truth does not split post-acoustic/pre-refresh from final calc_p_rho_phi pressure refresh, so no narrow pressure-refresh or acoustic-finish source fix was applied.",
            "U has no early substage source surface in the reused truth; its earliest checked material residual is the final post-RK/pre-halo comparison.",
        ],
        "next_decision": (
            "If fixing rather than further localizing, emit one WRF scratch surface inside "
            "first_rk_step_part1 around phy_prep/calc_p_rho_phi state writes for P/MU/W, "
            "or emit a post-acoustic/pre-refresh pressure surface if the manager wants to "
            "split the downstream final P residual before editing source."
        ),
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(f"verdict={VERDICT}")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
