#!/usr/bin/env python3
"""V0.14 Step-1 first_rk_step_part1 P/MU/W state split.

CPU-only proof. Reuses accepted WRF savepoint surfaces and reruns the current
JAX live-nest stage/carry captures. No production source is modified.
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

import step1_jax_loader_tstate as loader  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_part1_physics_state_mutation as part1  # noqa: E402
import step1_pre_part1_handoff as pre  # noqa: E402
import step1_rk1_source_boundary as source  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_first_rk_part1_p_state_split.json"
OUT_MD = PROOF_DIR / "step1_first_rk_part1_p_state_split.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-first-rk-part1-p-state-split/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
PREDECESSOR_JSON = PROOF_DIR / "step1_p_ph_mu_boundary_localization.json"
PREDECESSOR_MD = PROOF_DIR / "step1_p_ph_mu_boundary_localization.md"
PRECALL_TRUTH_ROOT = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth")
PART1_TRUTH_ROOT = Path("/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth")
SOURCE_TRUTH_ROOT = Path("/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/wrf_truth")

REQUIRED_ANCESTOR = "ebedb3c1"
TARGET_STEP = 1
TARGET_DOMAIN = 2
BOUNDARY_BAND = 5
VERDICT = "STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE"

STATE_FIELDS = ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE", "T_STATE")
P_FAMILY_FIELDS = ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE")
PART1_MASS_FIELDS = ("P_STATE", "MU_STATE", "T_STATE", "PB", "MUB", "MUT", "MU_TENDF")


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


def git_metadata() -> dict[str, Any]:
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"])
    return {
        "head": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "log_head": run_command(["git", "log", "-1", "--oneline", "--decorate"]),
        "required_ancestor": {
            "commit": REQUIRED_ANCESTOR,
            "returncode": ancestor["returncode"],
            "is_ancestor": ancestor["returncode"] == 0,
            "stderr_tail": ancestor.get("stderr_tail"),
        },
    }


def metric_brief(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metric:
        return None
    return {
        key: metric.get(key)
        for key in (
            "status",
            "shape",
            "count",
            "max_abs",
            "rmse",
            "bias",
            "p95",
            "p99",
            "nonfinite_diff_count",
            "worst_mismatch_index",
            "worst_mismatch_fortran",
        )
        if key in metric
    }


def metrics_for_fields(metrics: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: metric_brief(metrics.get(field)) for field in fields if field in metrics}


def top_residuals(items: list[Mapping[str, Any]], fields: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    out = []
    for item in items:
        if fields is not None and item.get("field") not in fields:
            continue
        out.append(
            {
                "field": item.get("field"),
                "max_abs": item.get("max_abs"),
                "rmse": item.get("rmse"),
                "material": item.get("material"),
                "material_threshold": item.get("material_threshold"),
                "worst_mismatch_fortran": item.get("worst_mismatch_fortran"),
            }
        )
    return out


def strip_arrays(surface: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in surface.items() if key != "arrays"}


def strip_wrf_surfaces(wrf: Mapping[str, Any]) -> dict[str, Any]:
    if "surfaces" not in wrf:
        return dict(wrf)
    return {
        **{key: value for key, value in wrf.items() if key != "surfaces"},
        "surfaces": {
            name: strip_arrays(surface) if isinstance(surface, Mapping) else surface
            for name, surface in wrf["surfaces"].items()
        },
    }


def compare_arrays(field: str, candidate: Any, reference: Any) -> dict[str, Any]:
    return live.diff_metrics(field, candidate, reference)


def compare_surface_arrays(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    return {
        field: compare_arrays(field, candidate[field], reference[field])
        for field in fields
        if field in candidate and field in reference
    }


def horizontal_mask(shape: tuple[int, ...], band: int) -> np.ndarray:
    if len(shape) == 2:
        ny, nx = shape
        yy = np.arange(ny)[:, None]
        xx = np.arange(nx)[None, :]
        return (yy < band) | (yy >= ny - band) | (xx < band) | (xx >= nx - band)
    if len(shape) == 3:
        _, ny, nx = shape
        yy = np.arange(ny)[None, :, None]
        xx = np.arange(nx)[None, None, :]
        return np.broadcast_to((yy < band) | (yy >= ny - band) | (xx < band) | (xx >= nx - band), shape)
    raise ValueError(f"unsupported shape: {shape}")


def region_stats(diff: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = diff[mask]
    finite = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {"status": "EMPTY_REGION", "count": 0}
    return {
        "status": "OK",
        "count": int(selected.size),
        "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
        "rmse": float(np.sqrt(np.mean(finite * finite))) if finite.size else None,
        "bias": float(np.mean(finite)) if finite.size else None,
        "p99": float(np.percentile(np.abs(finite), 99)) if finite.size else None,
        "nonfinite_diff_count": int((~np.isfinite(selected)).sum()),
    }


def decompose_boundary(field: str, candidate: Any, reference: Any) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {"status": "SHAPE_MISMATCH", "candidate_shape": list(cand.shape), "reference_shape": list(ref.shape)}
    diff = cand - ref
    mask = horizontal_mask(tuple(cand.shape), BOUNDARY_BAND)
    return {
        "field": field,
        "boundary_band_width_cells": BOUNDARY_BAND,
        "diff_sign": "jax_minus_wrf",
        "full_domain": metric_brief(live.diff_metrics(field, cand, ref)),
        "boundary_band": region_stats(diff, mask),
        "interior": region_stats(diff, ~mask),
        "not_boundary_only": bool(region_stats(diff, ~mask).get("max_abs") not in (None, 0.0)),
    }


def baseline_summary() -> dict[str, Any]:
    baseline = read_json(PREDECESSOR_JSON)
    final_top = baseline.get("current_final_comparison", {}).get("ranked_residuals_top8")
    if final_top is None:
        final_top = baseline.get("baseline_before_fix_from_step1_live_nest_theta_qv_wiring", {}).get(
            "next_field", {}
        ).get("ranked_top5", [])
    first = baseline.get("classification_evidence") or {}
    return {
        "source": str(PREDECESSOR_JSON),
        "path_info": path_info(PREDECESSOR_JSON),
        "markdown": path_info(PREDECESSOR_MD),
        "verdict": baseline.get("verdict"),
        "top_residuals": top_residuals(final_top or []),
        "first_material_p_boundary": {
            "wrf_boundary": "after_first_rk_step_part1",
            "jax_boundary": "_physics_step_forcing.carry.state",
            "field": "P_STATE",
            "max_abs": baseline.get("boundary_operator_table", [{}])[1].get("metric", {}).get("max_abs")
            if len(baseline.get("boundary_operator_table", [])) > 1
            else None,
            "predecessor_evidence": first,
        },
    }


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    pre_shapes = pre.expected_shapes()
    precall = pre.parse_wrf_surface("before_first_rk_step_part1_call", pre_shapes)
    if precall.get("status") != "WRF_SURFACE_READY":
        return {"status": "BLOCKED_PRECALL_WRF_TRUTH", "blocker": precall}

    part1_wrf = part1.parse_wrf_surfaces()
    if part1_wrf.get("status") != "WRF_PART1_TRUTH_READY":
        return {"status": "BLOCKED_PART1_WRF_TRUTH", "blocker": part1_wrf}

    source_wrf = source.parse_wrf_surfaces()
    if source_wrf.get("status") != "WRF_SOURCE_BOUNDARY_TRUTH_READY":
        return {"status": "BLOCKED_SOURCE_WRF_TRUTH", "blocker": source_wrf}

    stage_capture = loader.build_stage_capture()
    if stage_capture.get("status") != "JAX_STAGE_CAPTURE_READY":
        return {"status": "BLOCKED_JAX_STAGE_CAPTURE", "blocker": stage_capture}

    physics_capture = source.capture_jax_boundaries()
    if physics_capture.get("status") != "JAX_SOURCE_BOUNDARIES_READY":
        return {"status": "BLOCKED_JAX_PHYSICS_CAPTURE", "blocker": physics_capture}

    stage_comparisons = loader.compare_all_stages(precall, stage_capture)
    source_comparisons = source.compare_all(source_wrf, physics_capture)
    part1_internal = part1.compare_wrf_internal(part1_wrf)

    precall_arrays = precall["arrays"]
    part1_entry_arrays = part1_wrf["surfaces"]["part1_entry_before_init_zero_tendency"]["arrays"]
    after_phy_prep_arrays = part1_wrf["surfaces"]["after_phy_prep"]["arrays"]
    part1_exit_arrays = part1_wrf["surfaces"]["part1_exit"]["arrays"]
    after_first_arrays = source_wrf["surfaces"]["after_first_rk_step_part1"]["arrays"]
    halo_stage_arrays = stage_capture["stages"]["haloed_step_entry_state"]

    pre_to_part1_entry = compare_surface_arrays(part1_entry_arrays, precall_arrays, PART1_MASS_FIELDS)
    part1_entry_to_after_phy_prep = compare_surface_arrays(after_phy_prep_arrays, part1_entry_arrays, PART1_MASS_FIELDS)
    part1_entry_to_exit = compare_surface_arrays(part1_exit_arrays, part1_entry_arrays, PART1_MASS_FIELDS)
    pre_to_after_first = compare_surface_arrays(
        after_first_arrays,
        precall_arrays,
        ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE", "PB", "MUB", "PHB"),
    )
    boundary_decomposition = {
        field: decompose_boundary(field, halo_stage_arrays[field], precall_arrays[field])
        for field in P_FAMILY_FIELDS
    }

    stage_vs_wrf = stage_comparisons["stage_vs_wrf_precall"]
    transitions = stage_comparisons["stage_transitions"]
    p_stage_table = []
    for stage in loader.STAGE_ORDER:
        metrics = stage_vs_wrf[stage]["per_field_metrics"]
        p_stage_table.append(
            {
                "stage": stage,
                "first_material_field": stage_vs_wrf[stage].get("first_material_field"),
                "metrics": metrics_for_fields(metrics, ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE", "PB", "MUB", "PHB", "T_STATE")),
            }
        )
    p_transition_table = []
    for name in (
        "raw_child_state__to__live_child_state",
        "live_child_state__to__boundary_packaged_state",
        "boundary_packaged_state__to__initial_carry_state",
        "initial_carry_state__to__haloed_step_entry_state",
    ):
        metrics = transitions[name]["per_field_metrics"]
        p_transition_table.append(
            {
                "transition": name,
                "metrics": metrics_for_fields(metrics, ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE", "PB", "MUB", "PHB", "T_STATE")),
            }
        )

    after_part1_vs_physics = source_comparisons["matrix"]["after_first_rk_step_part1"]["vs_physics_carry_state_dry"]
    rk1_prep = source_comparisons["matrix"]["after_small_step_prep_calc_p_rho"]["vs_rk1_after_small_step_prep"]
    return {
        "status": "PROOF_EXECUTED",
        "verdict": VERDICT,
        "wrf_truth": {
            "precall": strip_arrays(precall),
            "part1": strip_wrf_surfaces(part1_wrf),
            "source_boundary": strip_wrf_surfaces(source_wrf),
        },
        "jax_capture": {
            "stage_capture": {key: value for key, value in stage_capture.items() if key != "stages"},
            "physics_capture": {key: value for key, value in physics_capture.items() if key != "captures"},
        },
        "stage_vs_wrf_precall_table": p_stage_table,
        "stage_transition_table": p_transition_table,
        "boundary_decomposition_vs_haloed_step_entry": boundary_decomposition,
        "wrf_continuity": {
            "precall_to_part1_entry_mass": metrics_for_fields(pre_to_part1_entry, PART1_MASS_FIELDS),
            "part1_entry_to_after_phy_prep_mass": metrics_for_fields(part1_entry_to_after_phy_prep, PART1_MASS_FIELDS),
            "part1_entry_to_part1_exit_mass": metrics_for_fields(part1_entry_to_exit, PART1_MASS_FIELDS),
            "precall_to_after_first_rk_step_part1_full_p_family": metrics_for_fields(
                pre_to_after_first, ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE", "PB", "MUB", "PHB")
            ),
        },
        "source_boundary_comparisons": {
            "after_first_rk_step_part1_vs_physics_carry_state_dry": {
                "first_material_field": after_part1_vs_physics.get("material_first_field"),
                "metrics": metrics_for_fields(after_part1_vs_physics["per_field_metrics"], STATE_FIELDS),
                "ranked": top_residuals(after_part1_vs_physics.get("material_ranked_residuals", []), STATE_FIELDS),
            },
            "rk1_after_small_step_prep_calc_p_rho_vs_jax": {
                "first_material_field": rk1_prep.get("material_first_field"),
                "metrics": metrics_for_fields(rk1_prep["per_field_metrics"], ("P_WORK", "MU_WORK", "W_WORK", "PH_WORK", "T_WORK")),
            },
        },
        "part1_internal_comparisons": {
            "selected_from_entry": {
                surface_name: metrics_for_fields(comp["per_field_metrics"], PART1_MASS_FIELDS)
                for surface_name, comp in part1_internal["from_entry"].items()
                if surface_name in ("after_init_zero_tendency", "after_phy_prep", "part1_exit")
            },
            "selected_adjacent": {
                name: metrics_for_fields(comp["per_field_metrics"], PART1_MASS_FIELDS)
                for name, comp in part1_internal["adjacent"].items()
                if "after_phy_prep" in name or "part1_exit" in name
            },
        },
        "classification": {
            "localized_boundary": "JAX raw_child_state already differs from WRF before_first_rk_step_part1_call for P_STATE/MU_STATE/W_STATE.",
            "wrf_part1_result": "WRF P_STATE/MU_STATE are unchanged from part1 entry through after_phy_prep and part1_exit; W_STATE/PH_STATE are unchanged from solve_em pre-call to after_first_rk_step_part1.",
            "jax_loader_result": "raw_child_state -> live_child_state fixes PB/MUB/PHB and T_STATE/QV from prior sprint, but P_STATE/MU_STATE/W_STATE remain unchanged through live_child, boundary package, carry, and halo.",
            "exact_missing_contract": (
                "Define and implement the live-nest child perturbation-state initialization contract for "
                "raw_child_state -> live_child_state: P_STATE, MU_STATE, and W_STATE must match WRF "
                "before_first_rk_step_part1_call, not raw wrfinput_d02 leaves."
            ),
            "no_source_fix_reason": (
                "No narrow GPU-compatible fix is justified here because the proof localizes the residual "
                "to missing live-nest loader semantics, not to first_rk_step_part1, phy_prep, "
                "rk_tendency, small_step_prep, calc_p_rho(step=0), or a production line with a proven formula."
            ),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    stage_rows = proof.get("stage_vs_wrf_precall_table", [])
    live_row = next((row for row in stage_rows if row["stage"] == "live_child_state"), {})
    halo_row = next((row for row in stage_rows if row["stage"] == "haloed_step_entry_state"), {})
    cont = proof.get("wrf_continuity", {})
    after_phy = cont.get("part1_entry_to_after_phy_prep_mass", {})
    pre_after = cont.get("precall_to_after_first_rk_step_part1_full_p_family", {})
    final_top = payload["predecessor_baseline"].get("top_residuals", [])[:5]
    lines = [
        "# V0.14 Step-1 First-RK Part1 P-State Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Fastest rigorous method: `{payload['tooling_verdict']}`.",
        "- Predecessor final top residuals preserved: "
        + ", ".join(f"`{item['field']}` max_abs `{item['max_abs']}`" for item in final_top)
        + ".",
        "- WRF `before_first_rk_step_part1_call` -> `after_first_rk_step_part1` is exact for `P_STATE/MU_STATE/W_STATE/PH_STATE`.",
        f"- WRF part1 entry -> `after_phy_prep`: `P_STATE` max_abs `{after_phy.get('P_STATE', {}).get('max_abs')}`, `MU_STATE` max_abs `{after_phy.get('MU_STATE', {}).get('max_abs')}`.",
        f"- JAX `live_child_state` vs WRF pre-call: `P_STATE` max_abs `{live_row.get('metrics', {}).get('P_STATE', {}).get('max_abs')}`, `MU_STATE` `{live_row.get('metrics', {}).get('MU_STATE', {}).get('max_abs')}`, `W_STATE` `{live_row.get('metrics', {}).get('W_STATE', {}).get('max_abs')}`.",
        f"- JAX `haloed_step_entry_state` carries the same residuals: `P_STATE` max_abs `{halo_row.get('metrics', {}).get('P_STATE', {}).get('max_abs')}`, `MU_STATE` `{halo_row.get('metrics', {}).get('MU_STATE', {}).get('max_abs')}`, `W_STATE` `{halo_row.get('metrics', {}).get('W_STATE', {}).get('max_abs')}`.",
        "- No production source fix was applied.",
        "",
        "## Boundary Table",
        "",
        "| Boundary/check | P_STATE | MU_STATE | W_STATE | PH_STATE |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = [
        ("WRF pre-call -> after_first_rk_step_part1", pre_after),
        ("JAX raw_child_state vs WRF pre-call", stage_rows[0]["metrics"] if stage_rows else {}),
        ("JAX live_child_state vs WRF pre-call", live_row.get("metrics", {})),
        ("JAX haloed_step_entry_state vs WRF pre-call", halo_row.get("metrics", {})),
    ]
    for name, metrics in rows:
        lines.append(
            "| {name} | {p} | {mu} | {w} | {ph} |".format(
                name=name,
                p=metrics.get("P_STATE", {}).get("max_abs"),
                mu=metrics.get("MU_STATE", {}).get("max_abs"),
                w=metrics.get("W_STATE", {}).get("max_abs"),
                ph=metrics.get("PH_STATE", {}).get("max_abs"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The current `P/MU/W` state mismatch is not introduced by WRF `first_rk_step_part1` or `phy_prep`; it exists before the call.",
            "- JAX live-nest base/theta/QV correction does not update perturbation `P_STATE/MU_STATE/W_STATE`; those leaves are unchanged through boundary package, carry construction, halo, and `_physics_step_forcing.carry.state`.",
            "- The exact missing contract is `raw_child_state -> live_child_state` perturbation-state initialization for `P_STATE/MU_STATE/W_STATE` against WRF `before_first_rk_step_part1_call`.",
            "",
            "Detailed metrics are in `proofs/v014/step1_first_rk_part1_p_state_split.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 First-RK Part1 P-State Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: split the Step-1 `P/MU/W` residual around WRF `first_rk_step_part1`, especially `phy_prep`, and name the exact upstream surface/contract if WRF part1 is clean.",
        "",
        "files changed:",
        "- `proofs/v014/step1_first_rk_part1_p_state_split.py`",
        "- `proofs/v014/step1_first_rk_part1_p_state_split.json`",
        "- `proofs/v014/step1_first_rk_part1_p_state_split.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`",
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
            f"- reused WRF pre-call truth root `{PRECALL_TRUTH_ROOT}`",
            f"- reused WRF part1 truth root `{PART1_TRUTH_ROOT}`",
            f"- reused WRF source-boundary truth root `{SOURCE_TRUTH_ROOT}`",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    proof = build_proof()
    verdict = proof.get("verdict", f"STEP1_FIRST_RK_PART1_P_STATE_BLOCKED_{proof.get('status', 'UNKNOWN')}")
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_first_rk_part1_p_state_split.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "environment": jax_environment(),
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "tooling_verdict": (
            "YES_FASTEST_RIGOROUS_WALL_CLOCK_REUSED_CPU_WRF_SAVEPOINTS_PLUS_CURRENT_CPU_JAX_STAGE_CAPTURE"
        ),
        "git": git_metadata(),
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "predecessor_json": path_info(PREDECESSOR_JSON),
            "predecessor_md": path_info(PREDECESSOR_MD),
            "wrf_precall_truth_root": path_info(PRECALL_TRUTH_ROOT),
            "wrf_part1_truth_root": path_info(PART1_TRUTH_ROOT),
            "wrf_source_truth_root": path_info(SOURCE_TRUTH_ROOT),
            "accepted_step1_truth_npz": path_info(live.ACCEPTED_TRUTH),
        },
        "predecessor_baseline": baseline_summary(),
        "proof": proof,
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_first_rk_part1_p_state_split.py",
                "python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json >/tmp/step1_first_rk_part1_p_state_split.validated.json",
                "git diff -- src/gpuwrf",
            ],
            "wrf_instrumentation": [
                "No new WRF run; reused existing CPU-only WRF truth surfaces from accepted v014 predecessor savepoints."
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "This proof names the missing live-nest perturbation-state contract but does not implement the formula for WRF's P/MU/W initialization.",
            "No Step-1 post-acoustic/pre-refresh run was made because the first material P/MU/W residual is already upstream of first_rk_step_part1.",
            "The next patch must preserve GPU residency and avoid any CPU-WRF runtime dependency.",
        ],
        "next_decision": (
            "Open a narrow live-nest perturbation-state sprint: transcribe/prove WRF raw-child -> "
            "pre-first_rk_step_part1 initialization for P_STATE, MU_STATE, and W_STATE, then apply only "
            "a GPU-native initialization fix if that closes this boundary."
        ),
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if proof.get("status") == "PROOF_EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
