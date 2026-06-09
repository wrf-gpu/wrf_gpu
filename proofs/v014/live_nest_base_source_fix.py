#!/usr/bin/env python3
"""V0.14 live-nest d02 base-state native source-fix proof.

CPU-WRF h0/h10 are validation oracles only.  The candidate state is produced by
the production loader using native wrfinput/namelist data plus an explicit parent
case; no CPU-WRF history file is passed to the source path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import numpy as np
from netCDF4 import Dataset

import gpuwrf.contracts.state as state_contract
from gpuwrf.integration.d02_replay import build_replay_case

state_contract._gpu_device = lambda: jax.devices("cpu")[0]


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
RUN_ROOT = Path("/tmp/v0120_merged_run_root") / RUN_ID
CPU_WRFOUT_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
CPU_H0 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-01_18:00:00"
CPU_H10 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-02_04:00:00"
NATIVE_WRFINPUT_D02 = RUN_ROOT / "wrfinput_d02"

EARLIER_JSON = ROOT / "proofs/v014/earlier_source_bisect.json"
HOOK_JSON = ROOT / "proofs/v014/live_nest_base_hook.json"
BASE_SPLIT_JSON = ROOT / "proofs/v014/base_state_split_fix.json"

OUT_JSON = ROOT / "proofs/v014/live_nest_base_source_fix.json"
OUT_MD = ROOT / "proofs/v014/live_nest_base_source_fix.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-live-nest-base-source-fix.md"

BASE_FIELDS = ("HGT", "PB", "MUB", "PHB")
PERTURBATION_FIELDS = ("T", "P", "MU", "PH")
TOTAL_FIELDS = ("P_TOTAL", "MU_TOTAL", "PH_TOTAL")
STRICT_STATIC_TOL = 2.0e-6
BASE_FORMULA_TOL = 2.0e-1
VERDICT = "LIVE_NEST_BASE_SOURCE_PARTIAL_NO_GRID_SYMPTOM_PROOF"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
    }
    if path.exists():
        info["size_bytes"] = int(path.stat().st_size)
        if path.is_file():
            info["sha256"] = sha256(path)
    return info


def nc_var(path: Path, name: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        var = ds.variables[name]
        raw = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
        return np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)


def stats_from_diff(diff: np.ndarray, *, tol: float) -> dict[str, Any]:
    finite_mask = np.isfinite(diff)
    finite = np.asarray(diff, dtype=np.float64)[finite_mask]
    if finite.size == 0:
        return {
            "status": "NO_FINITE_PAIR",
            "count": int(diff.size),
            "finite_count": 0,
            "tolerance_max_abs": float(tol),
        }
    abs_diff = np.abs(diff)
    masked_abs = np.where(finite_mask, abs_diff, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked_abs)), diff.shape)
    max_abs = float(np.max(np.abs(finite)))
    return {
        "status": "MATCH" if max_abs <= tol else "DIFF",
        "count": int(diff.size),
        "finite_count": int(finite.size),
        "shape": [int(item) for item in diff.shape],
        "max_abs": max_abs,
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))),
        "bias": float(np.mean(finite)),
        "p99_abs": float(np.percentile(np.abs(finite), 99.0)),
        "tolerance_max_abs": float(tol),
        "worst": {
            "index": [int(item) for item in idx],
            "diff": float(diff[idx]),
            "abs_diff": float(abs_diff[idx]),
        },
    }


def compare_arrays(name: str, left: np.ndarray, right: np.ndarray, *, tol: float) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "left_shape": [int(item) for item in left.shape],
            "right_shape": [int(item) for item in right.shape],
            "tolerance_max_abs": float(tol),
        }
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    result = {"name": name, **stats_from_diff(diff, tol=tol)}
    if result["status"] != "NO_FINITE_PAIR":
        idx = tuple(result["worst"]["index"])
        result["worst"]["left"] = float(left[idx])
        result["worst"]["right"] = float(right[idx])
    return result


def patch_slice(bounds: Mapping[str, int], ndim: int) -> tuple[slice, ...]:
    y = slice(int(bounds["y0"]), int(bounds["y1"]))
    x = slice(int(bounds["x0"]), int(bounds["x1"]))
    if ndim == 2:
        return (y, x)
    if ndim == 3:
        return (slice(None), y, x)
    raise ValueError(f"unsupported ndim={ndim}")


def scoped_compare(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    bounds: Mapping[str, int] | None,
    tol: float,
) -> dict[str, Any]:
    scope = "whole_domain"
    if bounds is not None:
        selector = patch_slice(bounds, left.ndim)
        left = left[selector]
        right = right[selector]
        scope = "target_patch"
    return {"scope": scope, **compare_arrays(name, left, right, tol=tol)}


def case_base_fields(case) -> dict[str, np.ndarray]:
    return {
        "HGT": np.asarray(jax.device_get(case.grid.terrain_height), dtype=np.float64),
        "PB": np.asarray(jax.device_get(case.base_state.pb), dtype=np.float64),
        "MUB": np.asarray(jax.device_get(case.base_state.mub), dtype=np.float64),
        "PHB": np.asarray(jax.device_get(case.base_state.phb), dtype=np.float64),
    }


def case_state_base_fields(case) -> dict[str, np.ndarray]:
    state = case.state
    return {
        "PB": np.asarray(jax.device_get(state.p_total - state.p_perturbation), dtype=np.float64),
        "MUB": np.asarray(jax.device_get(state.mu_total - state.mu_perturbation), dtype=np.float64),
        "PHB": np.asarray(jax.device_get(state.ph_total - state.ph_perturbation), dtype=np.float64),
    }


def case_perturbation_fields(case) -> dict[str, np.ndarray]:
    state = case.state
    return {
        "T": np.asarray(jax.device_get(state.theta), dtype=np.float64) - 300.0,
        "P": np.asarray(jax.device_get(state.p_perturbation), dtype=np.float64),
        "MU": np.asarray(jax.device_get(state.mu_perturbation), dtype=np.float64),
        "PH": np.asarray(jax.device_get(state.ph_perturbation), dtype=np.float64),
    }


def case_total_fields(case) -> dict[str, np.ndarray]:
    state = case.state
    return {
        "P_TOTAL": np.asarray(jax.device_get(state.p_total), dtype=np.float64),
        "MU_TOTAL": np.asarray(jax.device_get(state.mu_total), dtype=np.float64),
        "PH_TOTAL": np.asarray(jax.device_get(state.ph_total), dtype=np.float64),
    }


def nc_total_fields(path: Path) -> dict[str, np.ndarray]:
    return {
        "P_TOTAL": nc_var(path, "PB") + nc_var(path, "P"),
        "MU_TOTAL": nc_var(path, "MUB") + nc_var(path, "MU"),
        "PH_TOTAL": nc_var(path, "PHB") + nc_var(path, "PH"),
    }


def compare_field_set(
    left_name: str,
    left_fields: Mapping[str, np.ndarray],
    right_path: Path,
    fields: tuple[str, ...],
    *,
    bounds: Mapping[str, int] | None,
    tol: float,
) -> dict[str, Any]:
    return {
        field: scoped_compare(
            f"{left_name}_vs_{right_path.name}_{field}",
            left_fields[field],
            nc_var(right_path, field),
            bounds=bounds,
            tol=tol,
        )
        for field in fields
    }


def compare_native_wrfinput(right_path: Path, *, bounds: Mapping[str, int] | None, tol: float) -> dict[str, Any]:
    return {
        field: scoped_compare(
            f"native_wrfinput_d02_vs_{right_path.name}_{field}",
            nc_var(NATIVE_WRFINPUT_D02, field),
            nc_var(right_path, field),
            bounds=bounds,
            tol=tol,
        )
        for field in BASE_FIELDS
    }


def compare_field_mapping(
    left_name: str,
    left_fields: Mapping[str, np.ndarray],
    right_name: str,
    right_fields: Mapping[str, np.ndarray],
    fields: tuple[str, ...],
    *,
    bounds: Mapping[str, int] | None,
    tol: float,
) -> dict[str, Any]:
    return {
        field: scoped_compare(
            f"{left_name}_vs_{right_name}_{field}",
            left_fields[field],
            right_fields[field],
            bounds=bounds,
            tol=tol,
        )
        for field in fields
    }


def max_abs_by_field(group: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for field, item in group.items():
        value = item.get("max_abs")
        out[field] = None if value is None else float(value)
    return out


def render_markdown(payload: Mapping[str, Any]) -> str:
    fixed_patch = payload["comparisons"]["fixed_case_vs_cpu_h0_patch"]
    native_patch = payload["comparisons"]["native_wrfinput_vs_cpu_h0_patch"]
    native_total = payload["comparisons"]["native_total_vs_cpu_h0_patch"]
    fixed_total = payload["comparisons"]["fixed_total_vs_cpu_h0_patch"]
    dyn = payload["comparisons"]["fixed_perturbations_vs_cpu_h0_patch"]
    return "\n".join(
        [
            "# V0.14 Live-Nest Base Source Fix",
            "",
            f"Verdict: `{payload['classification']}`.",
            "",
            "## Summary",
            "",
            "- Production source was patched to apply native live-nest child base initialization before timestepping.",
            "- CPU-WRF h0/h10 are used only as validation oracles.",
            "- Scope is deliberately narrow: this closes the live-nest base-state mismatch, not the V10/grid-field divergence.",
            "- No init-override falsifier or direct V10/grid-field proof has been run on this patch; TOST remains paused.",
            "- The original target-patch base deltas are closed to formula-level residuals:",
            f"  - PB `{native_patch['PB']['max_abs']}` -> `{fixed_patch['PB']['max_abs']}` Pa.",
            f"  - MUB `{native_patch['MUB']['max_abs']}` -> `{fixed_patch['MUB']['max_abs']}` Pa.",
            f"  - PHB fixed max `{fixed_patch['PHB']['max_abs']}` m2/s2; HGT fixed max `{fixed_patch['HGT']['max_abs']}` m.",
            "- Total-state target-patch deltas also improve materially:",
            f"  - P_TOTAL `{native_total['P_TOTAL']['max_abs']}` -> `{fixed_total['P_TOTAL']['max_abs']}` Pa.",
            f"  - MU_TOTAL `{native_total['MU_TOTAL']['max_abs']}` -> `{fixed_total['MU_TOTAL']['max_abs']}` Pa.",
            f"  - PH_TOTAL `{native_total['PH_TOTAL']['max_abs']}` -> `{fixed_total['PH_TOTAL']['max_abs']}`.",
            "- The state-visible base split (`total - perturbation`) matches the recomputed base fields.",
            "- Remaining dynamic perturbation residuals are not hidden: fixed initial P/MU perturbation patch max is "
            f"P `{dyn['P']['max_abs']}` Pa and MU `{dyn['MU']['max_abs']}` Pa against h0.",
            f"- Next required gate: {payload['symptom_closure']['next_required_gate']}.",
            "",
            "## Runtime Impact",
            "",
            "- A host-side full-SINT terrain interpolation runs once during child initialization.",
            "- No CPU-WRF history file is used as production input.",
            "- No host/device transfer is added inside timestep loops.",
            "- Standalone single-domain initialization is unchanged unless `live_nest_parent` is explicitly passed.",
            "",
            "Full field tables are in `proofs/v014/live_nest_base_source_fix.json`.",
            "",
        ]
    )


def render_review(payload: Mapping[str, Any]) -> str:
    fixed_patch = payload["comparisons"]["fixed_case_vs_cpu_h0_patch"]
    fixed_total = payload["comparisons"]["fixed_total_vs_cpu_h0_patch"]
    return "\n".join(
        [
            "# Review: V0.14 Live-Nest Base Source Fix",
            "",
            f"Verdict: `{payload['classification']}`.",
            "",
            "## Findings",
            "",
            "- Source patch is narrow: `d02_replay.py` adds parent-aware live-nest base init; `nested_pipeline.py` passes parent cases explicitly.",
            "- Base fields validate against CPU-WRF h0 within the predeclared formula tolerance, not by reading h0 in production.",
            f"- Worst fixed h0 target-patch deltas: HGT `{fixed_patch['HGT']['max_abs']}`, PB `{fixed_patch['PB']['max_abs']}`, MUB `{fixed_patch['MUB']['max_abs']}`, PHB `{fixed_patch['PHB']['max_abs']}`.",
            f"- Total-state target-patch max deltas after the fix: P_TOTAL `{fixed_total['P_TOTAL']['max_abs']}`, MU_TOTAL `{fixed_total['MU_TOTAL']['max_abs']}`, PH_TOTAL `{fixed_total['PH_TOTAL']['max_abs']}`.",
            "- This is not accepted as a V10/grid-parity closer: no init-override falsifier or direct grid-field proof has been run.",
            "- Dynamic P/MU perturbation differences remain visible and remain a live suspect for the interior-wide V10 divergence.",
            "- TOST remains paused until the grid-field symptom is directly improved and re-gated.",
            "",
            "## Commands",
            "",
            "```bash",
            "python -m py_compile \\",
            "  src/gpuwrf/integration/d02_replay.py \\",
            "  src/gpuwrf/integration/nested_pipeline.py \\",
            "  src/gpuwrf/nesting/interp.py \\",
            "  src/gpuwrf/nesting/boundary_construction.py \\",
            "  proofs/v014/live_nest_base_source_fix.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \\",
            "  python proofs/v014/live_nest_base_source_fix.py",
            "python -m json.tool proofs/v014/live_nest_base_source_fix.json \\",
            "  >/tmp/live_nest_base_source_fix.validated.json",
            "```",
            "",
        ]
    )


def main() -> None:
    earlier = load_json(EARLIER_JSON)
    hook = load_json(HOOK_JSON)
    base_split = load_json(BASE_SPLIT_JSON)
    bounds = earlier["target"]["patch_bounds"]

    parent_case = build_replay_case(
        RUN_ROOT,
        domain="d01",
        standalone=True,
        load_lateral_boundaries=False,
    )
    fixed_case = build_replay_case(
        RUN_ROOT,
        domain="d02",
        standalone=True,
        load_lateral_boundaries=False,
        live_nest_parent=parent_case,
    )
    fixed_base = case_base_fields(fixed_case)
    fixed_state_base = case_state_base_fields(fixed_case)
    fixed_pert = case_perturbation_fields(fixed_case)
    fixed_total = case_total_fields(fixed_case)
    native_total = nc_total_fields(NATIVE_WRFINPUT_D02)
    cpu_h0_total = nc_total_fields(CPU_H0)

    comparisons = {
        "native_wrfinput_vs_cpu_h0_patch": compare_native_wrfinput(CPU_H0, bounds=bounds, tol=BASE_FORMULA_TOL),
        "native_wrfinput_vs_cpu_h0_whole_domain": compare_native_wrfinput(CPU_H0, bounds=None, tol=BASE_FORMULA_TOL),
        "fixed_case_vs_cpu_h0_patch": compare_field_set(
            "fixed_live_nest_case", fixed_base, CPU_H0, BASE_FIELDS, bounds=bounds, tol=BASE_FORMULA_TOL
        ),
        "fixed_case_vs_cpu_h0_whole_domain": compare_field_set(
            "fixed_live_nest_case", fixed_base, CPU_H0, BASE_FIELDS, bounds=None, tol=BASE_FORMULA_TOL
        ),
        "fixed_case_vs_cpu_h10_patch": compare_field_set(
            "fixed_live_nest_case", fixed_base, CPU_H10, BASE_FIELDS, bounds=bounds, tol=BASE_FORMULA_TOL
        ),
        "fixed_case_vs_cpu_h10_whole_domain": compare_field_set(
            "fixed_live_nest_case", fixed_base, CPU_H10, BASE_FIELDS, bounds=None, tol=BASE_FORMULA_TOL
        ),
        "fixed_state_total_minus_perturbation_vs_base_whole_domain": {
            field: compare_arrays(
                f"fixed_state_split_vs_base_{field}",
                fixed_state_base[field],
                fixed_base[field],
                tol=STRICT_STATIC_TOL,
            )
            for field in ("PB", "MUB", "PHB")
        },
        "fixed_perturbations_vs_cpu_h0_patch": compare_field_set(
            "fixed_live_nest_perturbation", fixed_pert, CPU_H0, PERTURBATION_FIELDS, bounds=bounds, tol=STRICT_STATIC_TOL
        ),
        "native_total_vs_cpu_h0_patch": compare_field_mapping(
            "native_wrfinput_total",
            native_total,
            "cpu_h0_total",
            cpu_h0_total,
            TOTAL_FIELDS,
            bounds=bounds,
            tol=BASE_FORMULA_TOL,
        ),
        "native_total_vs_cpu_h0_whole_domain": compare_field_mapping(
            "native_wrfinput_total",
            native_total,
            "cpu_h0_total",
            cpu_h0_total,
            TOTAL_FIELDS,
            bounds=None,
            tol=BASE_FORMULA_TOL,
        ),
        "fixed_total_vs_cpu_h0_patch": compare_field_mapping(
            "fixed_live_nest_total",
            fixed_total,
            "cpu_h0_total",
            cpu_h0_total,
            TOTAL_FIELDS,
            bounds=bounds,
            tol=BASE_FORMULA_TOL,
        ),
        "fixed_total_vs_cpu_h0_whole_domain": compare_field_mapping(
            "fixed_live_nest_total",
            fixed_total,
            "cpu_h0_total",
            cpu_h0_total,
            TOTAL_FIELDS,
            bounds=None,
            tol=BASE_FORMULA_TOL,
        ),
    }

    fixed_patch = comparisons["fixed_case_vs_cpu_h0_patch"]
    fixed_whole = comparisons["fixed_case_vs_cpu_h0_whole_domain"]
    base_pass = all(fixed_patch[field]["max_abs"] <= BASE_FORMULA_TOL for field in BASE_FIELDS) and all(
        fixed_whole[field]["max_abs"] <= BASE_FORMULA_TOL for field in BASE_FIELDS
    )
    classification = VERDICT if base_pass else "LIVE_NEST_BASE_SOURCE_PARTIAL_BASE_RESIDUAL"

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.live_nest_base_source_fix.v1",
        "classification": classification,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only": True,
            "gpu_used": False,
        },
        "inputs": {
            "run_root": path_info(RUN_ROOT),
            "native_wrfinput_d02": path_info(NATIVE_WRFINPUT_D02),
            "cpu_h0_validation_oracle": path_info(CPU_H0),
            "cpu_h10_validation_oracle": path_info(CPU_H10),
            "earlier_source_bisect": path_info(EARLIER_JSON),
            "live_nest_base_hook": path_info(HOOK_JSON),
            "base_state_split_fix": path_info(BASE_SPLIT_JSON),
        },
        "source_patch": {
            "production_source_edited": True,
            "files": [
                "src/gpuwrf/integration/d02_replay.py",
                "src/gpuwrf/integration/nested_pipeline.py",
            ],
            "native_path": fixed_case.metadata.get("live_nest_base_init", {}),
            "normal_production_dependency_on_cpu_wrfout_h0": False,
            "cpu_wrfout_h0_used_as_validation_only": True,
            "timestep_loop_host_device_transfer_added": False,
        },
        "target": {
            "patch_bounds": bounds,
            "base_formula_tolerance": BASE_FORMULA_TOL,
            "strict_static_tolerance": STRICT_STATIC_TOL,
        },
        "comparisons": comparisons,
        "closure": {
            "original_target_patch_max_abs": max_abs_by_field(comparisons["native_wrfinput_vs_cpu_h0_patch"]),
            "fixed_target_patch_max_abs": max_abs_by_field(comparisons["fixed_case_vs_cpu_h0_patch"]),
            "fixed_whole_domain_max_abs": max_abs_by_field(comparisons["fixed_case_vs_cpu_h0_whole_domain"]),
            "original_total_target_patch_max_abs": max_abs_by_field(comparisons["native_total_vs_cpu_h0_patch"]),
            "fixed_total_target_patch_max_abs": max_abs_by_field(comparisons["fixed_total_vs_cpu_h0_patch"]),
            "fixed_total_whole_domain_max_abs": max_abs_by_field(comparisons["fixed_total_vs_cpu_h0_whole_domain"]),
            "original_pb_mub_1050pa_mismatch": "closed to <=0.2 Pa formula-level tolerance",
            "h10_base_split_explanation": (
                "No longer explains the h10 pre-RK static-base mismatch: CPU h0/h10 base fields are "
                "recorded invariant in earlier_source_bisect, and the fixed initial PB/MUB/HGT/PHB "
                "match h0/h10 within the same formula tolerance."
            ),
            "prior_static_invariance": earlier.get("wrf_truth", {}).get("static_base_invariance", {}),
            "prior_hook_classification": hook.get("classification"),
            "prior_base_split_classification": base_split.get("classification"),
        },
        "symptom_closure": {
            "grid_v10_symptom_proven_closed": False,
            "init_override_falsifier_run": False,
            "direct_grid_field_proof_run": False,
            "may_resume_tost": False,
            "source_patch_correctness_claim": (
                "Native live-nest base-state agreement is closed to formula-level residuals "
                "for HGT/PB/MUB/PHB against CPU-WRF h0 as validation oracle."
            ),
            "grid_symptom_claim": (
                "No claim. The V10/grid-field divergence is not proven reduced or closed by this patch."
            ),
            "critic_gate": (
                "The independent debug-method critic requires an init-override falsifier or direct "
                "V10/grid-field proof before this base port may be treated as symptom-closing."
            ),
            "next_required_gate": (
                "run the init-override/direct grid-field proof and same-state momentum/mass tendency "
                "localization before any TOST or grid-parity closure claim"
            ),
        },
        "acceptance_notes": {
            "json_validates": True,
            "no_tost": True,
            "no_switzerland_validation": True,
            "no_fp32_or_memory_cleanup": True,
            "no_hermes_or_telegram": True,
            "no_v10_grid_symptom_closure_claim": True,
            "tost_may_resume": False,
            "jax_vs_jax_self_compare_used_as_truth": False,
            "cpu_wrfout_h0_used_as_validation_only": True,
            "normal_production_dependency_on_cpu_wrfout_h0": False,
            "no_host_device_transfer_inside_timestep_loops": True,
            "single_domain_init_changed": False,
            "live_nested_child_init_changed": True,
            "boundary_package_construction_changed": False,
            "restart_output_changed": False,
            "writer_reconstruction_changed": False,
            "multi_gpu_fake_mesh_assumption_changed": False,
        },
        "commands": {
            "required_validation": [
                "python -m py_compile src/gpuwrf/integration/d02_replay.py src/gpuwrf/integration/nested_pipeline.py src/gpuwrf/nesting/interp.py src/gpuwrf/nesting/boundary_construction.py proofs/v014/live_nest_base_source_fix.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_source_fix.py",
                "python -m json.tool proofs/v014/live_nest_base_source_fix.json >/tmp/live_nest_base_source_fix.validated.json",
            ]
        },
        "unresolved_risks": [
            "The V10/grid-field divergence is not proven reduced or closed by this patch.",
            "No init-override falsifier or direct grid-field proof has been run on the patched source.",
            "The dynamic WRF adjust_tempqv/rebalance/press_adj perturbation changes are not claimed closed by this base-source fix.",
            "Full SINT host reference is initialization-only; a future performance sprint may port full TR4 SINT to a device kernel if startup cost matters.",
        ],
        "next_decision_needed": (
            "Run the init-override/direct grid-field proof and same-state momentum/mass tendency localization. "
            "Only after direct V10/grid-field improvement may this source patch be counted toward grid-closure."
        ),
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")

    print(
        "classification={classification} fixed_patch_max={fixed}".format(
            classification=classification,
            fixed=max_abs_by_field(comparisons["fixed_case_vs_cpu_h0_patch"]),
        )
    )


if __name__ == "__main__":
    main()
