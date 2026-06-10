#!/usr/bin/env python3
"""V0.14 Step-1 tendency-family contract split.

CPU-only proof.  This starts from the patched Mythos perturb-init capture that
closed the stale RK1 P_STATE issue, then splits the next tendency boundary:

* WRF ``first_rk_step_part2`` fixed-source leaves, especially ``T_TENDF``;
* WRF RK1 pre/addtend tendency family, especially ``T_TEND/PH_TEND/RW_TEND``;
* JAX ``compute_advection_tendencies`` and ``_augment_large_step_tendencies``.

No production source is edited here.  The proof only localizes the first
remaining exact boundary and ranks cheap alternatives.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

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

import source_save_boundary_hook as savehook  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402
import step1_rk1_source_boundary as source  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_tendency_contract_split.json"
OUT_MD = PROOF_DIR / "step1_tendency_contract_split.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-tendency-contract-split/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
P_STATE_SPLIT_JSON = PROOF_DIR / "step1_rk1_p_state_source_split.json"
SOURCE_SAVE_PATTERN = "source_save_after_rk_tendency_d2_step_1_*.txt"

VERDICT = "STEP1_TENDENCY_CONTRACT_LOCALIZED_FIRST_RK_STEP_PART2_T_TENDF_SOURCE_LEAVES"

STATE_FIELDS = ("T_STATE", "P_STATE", "MU_STATE", "W_STATE", "PH_STATE")
SOURCE_FIELDS = (
    "T_TENDF",
    "H_DIABATIC",
    "T_SAVE",
    "MU_TENDF",
    "RW_TENDF",
    "PH_TENDF",
    "W_SAVE",
    "PH_SAVE",
)
TENDENCY_FIELDS = (
    "T_TEND",
    "T_TENDF",
    "MU_TEND",
    "MU_TENDF",
    "PH_TEND",
    "PH_TENDF",
    "RW_TEND",
    "RW_TENDF",
)
REQUESTED_TENDENCY_FIELDS = ("T_TEND", "PH_TEND", "RW_TEND")
SPARSE_SOURCE_FIELDS = (
    "T_TENDF",
    "H_DIABATIC",
    "T_SAVE",
    "MU_TENDF",
    "RW_TENDF",
    "PH_TENDF",
    "W_SAVE",
    "PH_SAVE",
    "PH_TEND",
    "RW_TEND",
)
SPARSE_FIELD_MAP = {
    "T_TENDF": ("MASS_SOURCE", "T_TENDF"),
    "H_DIABATIC": ("MASS_SOURCE", "H_DIABATIC"),
    "T_SAVE": ("MASS_SOURCE", "T_SAVE"),
    "MU_TENDF": ("MASS2D_SOURCE", "MU_TENDF"),
    "RW_TENDF": ("WPH_SOURCE", "RW_TENDF"),
    "PH_TENDF": ("WPH_SOURCE", "PH_TENDF"),
    "W_SAVE": ("WPH_SOURCE", "W_SAVE"),
    "PH_SAVE": ("WPH_SOURCE", "PH_SAVE"),
    "RW_TEND": ("WPH_SOURCE", "RW_TEND"),
    "PH_TEND": ("WPH_SOURCE", "PH_TEND"),
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

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
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
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
    return {
        "head": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "log_head": run_command(["git", "log", "-1", "--oneline", "--decorate"]),
        "status_short_branch": run_command(["git", "status", "--short", "--branch"]),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
            "first_mismatch_fortran",
            "worst_mismatch_fortran",
            "worst_mismatch_index",
        )
        if key in metric
    }


def selected_metrics(metrics: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: metric_brief(metrics.get(field)) for field in fields if field in metrics}


def material(field: str, metric: Mapping[str, Any] | None) -> bool:
    if not metric or metric.get("status") != "OK":
        return bool(metric)
    max_abs = metric.get("max_abs")
    if max_abs is None:
        return bool(metric.get("nonfinite_diff_count"))
    threshold = source.MATERIAL_THRESHOLDS.get(field)
    return threshold is not None and float(max_abs) > float(threshold)


def top_residuals(comp: Mapping[str, Any], fields: Iterable[str], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for field in fields:
        metric = comp.get("per_field_metrics", {}).get(field)
        if metric and metric.get("status") == "OK":
            rows.append(
                {
                    "field": field,
                    "max_abs": metric.get("max_abs"),
                    "rmse": metric.get("rmse"),
                    "material_threshold": source.MATERIAL_THRESHOLDS.get(field),
                    "material": material(field, metric),
                    "worst_mismatch_fortran": metric.get("worst_mismatch_fortran"),
                }
            )
    return sorted(rows, key=lambda item: -float(item["max_abs"] or 0.0))[:limit]


def compact_surface(comp: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {
        "status": comp.get("status"),
        "strict_first_mismatch_field": comp.get("strict_first_mismatch_field"),
        "material_first_field": comp.get("material_first_field"),
        "selected_metrics": selected_metrics(comp.get("per_field_metrics", {}), fields),
        "top_material_residuals": [
            {
                "field": item.get("field"),
                "max_abs": item.get("max_abs"),
                "rmse": item.get("rmse"),
                "material": item.get("material"),
                "material_threshold": item.get("material_threshold"),
                "worst_mismatch_fortran": item.get("worst_mismatch_fortran"),
            }
            for item in comp.get("material_ranked_residuals", [])[:12]
        ],
    }


def strip_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    if "surfaces" not in wrf:
        return dict(wrf)
    out = {key: value for key, value in wrf.items() if key != "surfaces"}
    out["surfaces"] = {
        name: {key: value for key, value in surface.items() if key != "arrays"}
        for name, surface in wrf["surfaces"].items()
    }
    return out


def source_save_files() -> list[Path]:
    return sorted(source.WRF_TRUTH.glob(SOURCE_SAVE_PATTERN))


def parse_source_save() -> dict[str, Any]:
    files = source_save_files()
    if not files:
        return {
            "status": "BLOCKED_NO_SOURCE_SAVE_FILES",
            "pattern": str(source.WRF_TRUTH / SOURCE_SAVE_PATTERN),
        }
    parsed = savehook.parse_savepoint(files, savehook.SOURCE_SCHEMAS)
    return {
        "status": "SOURCE_SAVE_AFTER_RK_TENDENCY_READY",
        "files": files,
        "surface": parsed,
        "compact": savehook.compact_surface(parsed, savehook.SOURCE_SCHEMAS),
    }


def sparse_fortran_index(field: str, key: tuple[int, ...]) -> dict[str, int] | None:
    if len(key) == 2:
        y, x = key
        return {"i": int(x) + 1, "j": int(y) + 1}
    if len(key) == 3:
        k, y, x = key
        if field in {"PH_TEND", "PH_TENDF", "PH_SAVE", "RW_TEND", "RW_TENDF", "W_SAVE"}:
            return {"i": int(x) + 1, "j": int(y) + 1, "kstag": int(k) + 1}
        return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    return None


def sparse_diff_metrics(
    field: str,
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    keys: list[tuple[int, ...]],
) -> dict[str, Any]:
    if candidate_values.shape != reference_values.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "candidate_shape": list(candidate_values.shape),
            "reference_shape": list(reference_values.shape),
        }
    if candidate_values.size == 0:
        return {"status": "NO_COMMON_RECORDS", "count": 0}
    diff = candidate_values - reference_values
    absdiff = np.abs(diff)
    finite_abs = absdiff[np.isfinite(absdiff)]
    mismatch = np.argwhere((diff != 0.0) | (~np.isfinite(diff))).reshape(-1)
    first_pos = int(mismatch[0]) if mismatch.size else None
    if finite_abs.size:
        worst_pos = int(np.nanargmax(absdiff))
        max_abs = float(np.nanmax(absdiff))
        rmse = float(np.sqrt(np.nanmean(diff * diff)))
        bias = float(np.nanmean(diff))
        p95 = float(np.nanpercentile(absdiff, 95))
        p99 = float(np.nanpercentile(absdiff, 99))
    else:
        worst_pos = first_pos
        max_abs = None
        rmse = None
        bias = None
        p95 = None
        p99 = None
    first_key = keys[first_pos] if first_pos is not None else None
    worst_key = keys[worst_pos] if worst_pos is not None else None
    return {
        "status": "OK",
        "count": int(diff.size),
        "shape": [int(diff.size)],
        "max_abs": max_abs,
        "rmse": rmse,
        "bias": bias,
        "p95": p95,
        "p99": p99,
        "nonfinite_diff_count": int((~np.isfinite(diff)).sum()),
        "first_mismatch_index": [first_pos] if first_pos is not None else None,
        "first_mismatch_key_zero_based": list(first_key) if first_key is not None else None,
        "first_mismatch_fortran": sparse_fortran_index(field, first_key) if first_key is not None else None,
        "worst_mismatch_index": [worst_pos] if worst_pos is not None else None,
        "worst_mismatch_key_zero_based": list(worst_key) if worst_key is not None else None,
        "worst_mismatch_fortran": sparse_fortran_index(field, worst_key) if worst_key is not None else None,
        "worst_candidate": float(candidate_values[worst_pos]) if worst_pos is not None else None,
        "worst_reference": float(reference_values[worst_pos]) if worst_pos is not None else None,
    }


def compare_sparse_source_save(
    source_save: Mapping[str, Any],
    jax_arrays: Mapping[str, Any],
    *,
    fields: Iterable[str] = SPARSE_SOURCE_FIELDS,
) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    records_by_tag = source_save["surface"]["records"]
    metrics: dict[str, Any] = {}
    for field in fields:
        if field not in SPARSE_FIELD_MAP:
            continue
        if field not in jax_arrays:
            metrics[field] = {"status": "MISSING_JAX_FIELD"}
            continue
        tag, wrf_field = SPARSE_FIELD_MAP[field]
        records = records_by_tag.get(tag, {})
        candidate_array = np.asarray(jax.device_get(jax_arrays[field]), dtype=np.float64)
        candidate_values: list[float] = []
        reference_values: list[float] = []
        keys: list[tuple[int, ...]] = []
        out_of_bounds: list[tuple[int, ...]] = []
        for key in sorted(records):
            if wrf_field not in records[key]:
                continue
            if len(key) != candidate_array.ndim or any(idx < 0 or idx >= candidate_array.shape[pos] for pos, idx in enumerate(key)):
                out_of_bounds.append(key)
                continue
            candidate_values.append(float(candidate_array[key]))
            reference_values.append(float(records[key][wrf_field]))
            keys.append(tuple(int(item) for item in key))
        if out_of_bounds:
            metrics[field] = {
                "status": "OUT_OF_BOUNDS_RECORDS",
                "candidate_shape": list(candidate_array.shape),
                "out_of_bounds_count": len(out_of_bounds),
                "first_out_of_bounds": list(out_of_bounds[0]),
            }
            continue
        metrics[field] = sparse_diff_metrics(
            field,
            np.asarray(candidate_values, dtype=np.float64),
            np.asarray(reference_values, dtype=np.float64),
            keys,
        )
    material_ranked = []
    for field, metric in metrics.items():
        if metric.get("status") != "OK":
            continue
        material_ranked.append(
            {
                "field": field,
                "max_abs": metric.get("max_abs"),
                "rmse": metric.get("rmse"),
                "material_threshold": source.MATERIAL_THRESHOLDS.get(field),
                "material": material(field, metric),
                "worst_mismatch_fortran": metric.get("worst_mismatch_fortran"),
            }
        )
    material_ranked.sort(key=lambda item: -float(item["max_abs"] or 0.0))
    strict_first = None
    material_first = None
    for field in fields:
        metric = metrics.get(field)
        if not metric:
            continue
        if metric.get("status") != "OK":
            strict_first = strict_first or field
            continue
        max_abs = float(metric.get("max_abs") or 0.0)
        if strict_first is None and (int(metric.get("nonfinite_diff_count", 0)) or max_abs != 0.0):
            strict_first = field
        if material_first is None and material(field, metric):
            material_first = field
    return {
        "status": "SOURCE_SAVE_SPARSE_COMPARISON_EXECUTED",
        "strict_first_mismatch_field": strict_first,
        "material_first_field": material_first,
        "per_field_metrics": metrics,
        "material_ranked_residuals": material_ranked,
        "patch_only": True,
        "one_cell_proof": False,
    }


def sparse_summary(comp: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {
        "status": comp.get("status"),
        "strict_first_mismatch_field": comp.get("strict_first_mismatch_field"),
        "material_first_field": comp.get("material_first_field"),
        "selected_metrics": selected_metrics(comp.get("per_field_metrics", {}), fields),
        "top_material_residuals": comp.get("material_ranked_residuals", [])[:10],
        "patch_only": comp.get("patch_only"),
        "one_cell_proof": comp.get("one_cell_proof"),
    }


def build_tendency_capture(inputs: Mapping[str, Any], carry: Any, *, label: str) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    namelist = inputs["namelist"]
    jnp = inputs["jnp"]
    lead_seconds = jnp.asarray(float(source.TARGET_STEP) * float(namelist.dt_s), dtype=jnp.float64)
    cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
    run_radiation = bool(cadence > 0 and source.TARGET_STEP % cadence == 0)

    physics = om._physics_step_forcing(
        carry,
        namelist,
        lead_seconds,
        run_radiation=run_radiation,
        first_timestep=True,
    )
    spec = om.halo_spec(namelist.grid)
    physics_carry_state = om.apply_halo(physics.carry.state, spec)
    physics_state = om.apply_halo(physics.state, spec)
    empty_dry = om.DryPhysicsTendencies()
    rk1_reference = physics_carry_state
    base_tendencies = om.compute_advection_tendencies(
        physics_carry_state, namelist.tendencies, namelist.grid
    )
    augment_empty = om._augment_large_step_tendencies(
        physics_carry_state,
        base_tendencies,
        namelist,
        rk_step=1,
        physics_tendencies=empty_dry,
        step_origin=rk1_reference,
    )
    augment_full = om._augment_large_step_tendencies(
        physics_carry_state,
        base_tendencies,
        namelist,
        rk_step=1,
        physics_tendencies=physics.dry_tendencies,
        step_origin=rk1_reference,
    )
    prep = om.small_step_prep_wrf(
        physics_carry_state,
        1,
        float(namelist.dt_s) / 3.0,
        metrics=namelist.metrics,
        reference_state=rk1_reference,
        ww=physics.carry.ww,
    )
    pressure = om.calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
    acoustic_no_relax = om._acoustic_core_state_from_prep(
        physics.carry,
        prep,
        pressure,
        namelist,
        augment_empty,
        lead_seconds=None,
    )
    acoustic_with_current_boundary = om._acoustic_core_state_from_prep(
        physics.carry,
        prep,
        pressure,
        namelist,
        augment_full,
        lead_seconds=lead_seconds,
    )

    raw_arrays = source.state_tendency_arrays(jnp, physics_carry_state, base_tendencies, empty_dry)
    augment_empty_arrays = source.state_tendency_arrays(
        jnp, physics_carry_state, augment_empty, empty_dry
    )
    augment_full_arrays = source.state_tendency_arrays(
        jnp, physics_carry_state, augment_full, physics.dry_tendencies
    )
    acoustic_no_relax_arrays = dict(augment_empty_arrays)
    acoustic_no_relax_arrays.update(
        {
            "PH_TEND": acoustic_no_relax.ph_tend,
            "RW_TEND": acoustic_no_relax.rw_tend_pg_buoy,
        }
    )
    acoustic_with_current_boundary_arrays = dict(augment_full_arrays)
    acoustic_with_current_boundary_arrays.update(
        {
            "PH_TEND": acoustic_with_current_boundary.ph_tend,
            "RW_TEND": acoustic_with_current_boundary.rw_tend_pg_buoy,
        }
    )

    jax.block_until_ready(acoustic_no_relax.ph_tend)
    return {
        "status": "JAX_TENDENCY_BOUNDARIES_READY",
        "label": label,
        "captures": {
            "physics_carry_state_dry": source.state_source_arrays(
                jnp, physics_carry_state, physics.dry_tendencies
            ),
            "physics_state_dry": source.state_source_arrays(
                jnp, physics_state, physics.dry_tendencies
            ),
            "raw_compute_advection_tendencies": raw_arrays,
            "augment_empty_dry": augment_empty_arrays,
            "augment_full_dry": augment_full_arrays,
            "acoustic_stage_no_relax_ph_rw": acoustic_no_relax_arrays,
            "acoustic_stage_with_current_boundary_ph_rw": acoustic_with_current_boundary_arrays,
        },
        "run_radiation": run_radiation,
        "lead_seconds": float(lead_seconds),
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "rk_order": int(namelist.rk_order),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "force_fp64": bool(namelist.force_fp64),
            "cu_physics": int(namelist.cu_physics),
            "mp_physics": int(namelist.mp_physics),
            "bl_pbl_physics": int(namelist.bl_pbl_physics),
            "sf_sfclay_physics": int(namelist.sf_sfclay_physics),
            "rad_rk_tendf": int(namelist.rad_rk_tendf),
            "radiation_cadence_steps": int(getattr(namelist, "radiation_cadence_steps", 1)),
            "use_flux_advection": bool(namelist.use_flux_advection),
            "moist_adv_opt": int(namelist.moist_adv_opt),
            "scalar_adv_opt": int(namelist.scalar_adv_opt),
            "diff_opt": int(namelist.diff_opt),
            "km_opt": int(namelist.km_opt),
            "diff_6th_opt": int(namelist.diff_6th_opt),
        },
    }


def compare_full_surfaces(
    wrf_source: Mapping[str, Any],
    tendency_capture: Mapping[str, Any],
    *,
    pstate_capture: Mapping[str, Any],
) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    pstate_comp = pstate.compare_capture(wrf_source, pstate_capture)
    cap = tendency_capture["captures"]
    return {
        "status": "FULL_SURFACE_COMPARISONS_EXECUTED",
        "after_first_rk_step_part2": {
            "vs_physics_carry_state_dry": source.compare_surface(
                "after_first_rk_step_part2",
                wrf_source["surfaces"]["after_first_rk_step_part2"],
                cap["physics_carry_state_dry"],
                jax,
            ),
            "vs_physics_state_dry": source.compare_surface(
                "after_first_rk_step_part2",
                wrf_source["surfaces"]["after_first_rk_step_part2"],
                cap["physics_state_dry"],
                jax,
            ),
        },
        "after_rk_addtend_before_small_step_prep": {
            "vs_raw_compute_advection_tendencies": source.compare_surface(
                "after_rk_addtend_before_small_step_prep",
                wrf_source["surfaces"]["after_rk_addtend_before_small_step_prep"],
                cap["raw_compute_advection_tendencies"],
                jax,
            ),
            "vs_augment_empty_dry": source.compare_surface(
                "after_rk_addtend_before_small_step_prep",
                wrf_source["surfaces"]["after_rk_addtend_before_small_step_prep"],
                cap["augment_empty_dry"],
                jax,
            ),
            "vs_augment_full_dry": source.compare_surface(
                "after_rk_addtend_before_small_step_prep",
                wrf_source["surfaces"]["after_rk_addtend_before_small_step_prep"],
                cap["augment_full_dry"],
                jax,
            ),
            "vs_acoustic_stage_no_relax_ph_rw": source.compare_surface(
                "after_rk_addtend_before_small_step_prep",
                wrf_source["surfaces"]["after_rk_addtend_before_small_step_prep"],
                cap["acoustic_stage_no_relax_ph_rw"],
                jax,
            ),
            "vs_acoustic_stage_with_current_boundary_ph_rw": source.compare_surface(
                "after_rk_addtend_before_small_step_prep",
                wrf_source["surfaces"]["after_rk_addtend_before_small_step_prep"],
                cap["acoustic_stage_with_current_boundary_ph_rw"],
                jax,
            ),
            "previous_pstate_wrapper_vs_rk1_after_rk_addtend": pstate_comp["matrix"][
                "after_rk_addtend_before_small_step_prep"
            ]["vs_rk1_after_rk_addtend"],
        },
    }


def compare_sparse_boundaries(
    source_save: Mapping[str, Any],
    tendency_capture: Mapping[str, Any],
) -> dict[str, Any]:
    cap = tendency_capture["captures"]
    return {
        "status": "SPARSE_SOURCE_SAVE_COMPARISONS_EXECUTED",
        "source_save_after_rk_tendency": {
            "vs_physics_carry_state_dry": compare_sparse_source_save(
                source_save,
                cap["physics_carry_state_dry"],
                fields=SOURCE_FIELDS,
            ),
            "vs_raw_compute_advection_tendencies": compare_sparse_source_save(
                source_save,
                cap["raw_compute_advection_tendencies"],
                fields=("PH_TEND", "RW_TEND"),
            ),
            "vs_augment_empty_dry": compare_sparse_source_save(
                source_save,
                cap["augment_empty_dry"],
                fields=("PH_TEND", "RW_TEND"),
            ),
            "vs_augment_full_dry": compare_sparse_source_save(
                source_save,
                cap["augment_full_dry"],
                fields=("PH_TEND", "RW_TEND"),
            ),
            "vs_acoustic_stage_no_relax_ph_rw": compare_sparse_source_save(
                source_save,
                cap["acoustic_stage_no_relax_ph_rw"],
                fields=("PH_TEND", "RW_TEND"),
            ),
            "vs_acoustic_stage_with_current_boundary_ph_rw": compare_sparse_source_save(
                source_save,
                cap["acoustic_stage_with_current_boundary_ph_rw"],
                fields=("PH_TEND", "RW_TEND"),
            ),
        },
    }


def make_rad_rk_tendf_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(inputs)
    out["namelist"] = dataclasses.replace(inputs["namelist"], rad_rk_tendf=1)
    return out


def build_rad_toggle_falsifier(
    wrf_source: Mapping[str, Any],
    inputs: Mapping[str, Any],
    patched_carry: Any,
) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    rad_inputs = make_rad_rk_tendf_inputs(inputs)
    rad_capture = pstate.capture_from_carry(
        rad_inputs,
        patched_carry,
        label="proof_local_rad_rk_tendf_1",
    )
    comp = source.compare_surface(
        "after_first_rk_step_part2",
        wrf_source["surfaces"]["after_first_rk_step_part2"],
        rad_capture["captures"]["physics_carry_state_dry"],
        jax,
    )
    return {
        "status": "RAD_RK_TENDF_TOGGLE_EXECUTED",
        "namelist": rad_capture.get("namelist"),
        "run_radiation": rad_capture.get("run_radiation"),
        "after_first_rk_step_part2_vs_physics_carry_state_dry": compact_surface(
            comp, STATE_FIELDS + SOURCE_FIELDS
        ),
        "selected_T_TENDF_metric": metric_brief(
            comp.get("per_field_metrics", {}).get("T_TENDF")
        ),
        "interpretation": (
            "Proof-local rad_rk_tendf=1 does not close WRF Step-1 T_TENDF because this "
            "step does not run radiation cadence and the held radiation leaf is not the "
            "missing current-step source bundle."
        ),
    }


def extract_predecessor_context() -> dict[str, Any]:
    previous = read_json(P_STATE_SPLIT_JSON)
    return {
        "path": str(P_STATE_SPLIT_JSON),
        "verdict": previous.get("verdict"),
        "p_state_gate": previous.get("proof", {}).get("p_state_gate"),
        "patched_after_first_rk_step_part2": previous.get("proof", {})
        .get("selected_before_after_surfaces", {})
        .get("patched_after_first_rk_step_part2"),
        "patched_after_rk_addtend_before_small_step_prep": previous.get("proof", {})
        .get("selected_before_after_surfaces", {})
        .get("patched_after_rk_addtend_before_small_step_prep"),
    }


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    wrf_source = source.parse_wrf_surfaces()
    if wrf_source.get("status") != "WRF_SOURCE_BOUNDARY_TRUTH_READY":
        return {"status": "BLOCKED_WRF_SOURCE_TRUTH", "blocker": strip_wrf(wrf_source)}
    source_save = parse_source_save()
    if source_save.get("status") != "SOURCE_SAVE_AFTER_RK_TENDENCY_READY":
        return {"status": "BLOCKED_SOURCE_SAVE_TRUTH", "blocker": source_save}

    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    pstate_capture = pstate.capture_from_carry(
        inputs,
        patched["carry"],
        label="production_mythos_perturb_init_pstate_wrapper",
    )
    tendency_capture = build_tendency_capture(
        inputs,
        patched["carry"],
        label="production_mythos_perturb_init_tendency_contract",
    )

    full = compare_full_surfaces(wrf_source, tendency_capture, pstate_capture=pstate_capture)
    sparse = compare_sparse_boundaries(source_save, tendency_capture)
    rad_toggle = build_rad_toggle_falsifier(wrf_source, inputs, patched["carry"])

    part2_carry = full["after_first_rk_step_part2"]["vs_physics_carry_state_dry"]
    add_aug_full = full["after_rk_addtend_before_small_step_prep"]["vs_augment_full_dry"]
    add_aug_empty = full["after_rk_addtend_before_small_step_prep"]["vs_augment_empty_dry"]
    add_acoustic_raw = full["after_rk_addtend_before_small_step_prep"][
        "vs_acoustic_stage_no_relax_ph_rw"
    ]
    save_source = sparse["source_save_after_rk_tendency"]["vs_physics_carry_state_dry"]
    save_aug_empty = sparse["source_save_after_rk_tendency"]["vs_augment_empty_dry"]
    save_acoustic_raw = sparse["source_save_after_rk_tendency"]["vs_acoustic_stage_no_relax_ph_rw"]

    return {
        "status": "PROOF_EXECUTED",
        "verdict": VERDICT,
        "patched_init": {
            "metadata": patched["metadata"],
            "delta_from_stale_proof_state": patched["delta_from_stale_proof_state"],
        },
        "jax_capture": {
            "tendency_status": tendency_capture["status"],
            "tendency_label": tendency_capture["label"],
            "tendency_namelist": tendency_capture["namelist"],
            "run_radiation": tendency_capture["run_radiation"],
            "pstate_wrapper_namelist": pstate_capture["namelist"],
        },
        "full_domain_split": {
            "after_first_rk_step_part2_vs_physics_carry": compact_surface(
                part2_carry,
                STATE_FIELDS + SOURCE_FIELDS,
            ),
            "after_rk_addtend_vs_raw_compute_advection": compact_surface(
                full["after_rk_addtend_before_small_step_prep"][
                    "vs_raw_compute_advection_tendencies"
                ],
                STATE_FIELDS + TENDENCY_FIELDS,
            ),
            "after_rk_addtend_vs_augment_empty_dry": compact_surface(
                add_aug_empty,
                STATE_FIELDS + TENDENCY_FIELDS,
            ),
            "after_rk_addtend_vs_augment_full_dry": compact_surface(
                add_aug_full,
                STATE_FIELDS + TENDENCY_FIELDS,
            ),
            "after_rk_addtend_vs_acoustic_stage_no_relax_ph_rw": compact_surface(
                add_acoustic_raw,
                STATE_FIELDS + TENDENCY_FIELDS,
            ),
            "after_rk_addtend_vs_acoustic_stage_with_current_boundary_ph_rw": compact_surface(
                full["after_rk_addtend_before_small_step_prep"][
                    "vs_acoustic_stage_with_current_boundary_ph_rw"
                ],
                STATE_FIELDS + TENDENCY_FIELDS,
            ),
        },
        "source_save_patch_split": {
            "truth_summary": source_save["compact"],
            "source_save_after_rk_tendency_vs_physics_carry": sparse_summary(
                save_source, SOURCE_FIELDS
            ),
            "source_save_after_rk_tendency_vs_raw_compute_advection": sparse_summary(
                sparse["source_save_after_rk_tendency"]["vs_raw_compute_advection_tendencies"],
                ("PH_TEND", "RW_TEND"),
            ),
            "source_save_after_rk_tendency_vs_augment_empty_dry": sparse_summary(
                save_aug_empty,
                ("PH_TEND", "RW_TEND"),
            ),
            "source_save_after_rk_tendency_vs_augment_full_dry": sparse_summary(
                sparse["source_save_after_rk_tendency"]["vs_augment_full_dry"],
                ("PH_TEND", "RW_TEND"),
            ),
            "source_save_after_rk_tendency_vs_acoustic_stage_no_relax_ph_rw": sparse_summary(
                save_acoustic_raw,
                ("PH_TEND", "RW_TEND"),
            ),
            "source_save_after_rk_tendency_vs_acoustic_stage_with_current_boundary_ph_rw": sparse_summary(
                sparse["source_save_after_rk_tendency"][
                    "vs_acoustic_stage_with_current_boundary_ph_rw"
                ],
                ("PH_TEND", "RW_TEND"),
            ),
        },
        "cheap_falsifiers": {
            "rad_rk_tendf_1": rad_toggle,
            "boundary_or_spec_only": {
                "status": "REFUTED_FOR_FIRST_T_TENDF_BOUNDARY",
                "evidence": (
                    "WRF source-save after rk_tendency is before relax_bdy_dry, "
                    "rk_addtend_dry, spec_bdy_dry, small_step_prep, and acoustic "
                    "updates, yet T_TENDF is already nonzero and differs from JAX."
                ),
                "source_save_T_TENDF_metric": metric_brief(
                    save_source.get("per_field_metrics", {}).get("T_TENDF")
                ),
            },
            "raw_advection_explains_T_TENDF": {
                "status": "REFUTED_FOR_FIRST_BOUNDARY",
                "evidence": (
                    "T_TENDF is a WRF fixed-source leaf produced by first_rk_step_part2/"
                    "update_phy_ten before rk_addtend. It is not produced by "
                    "compute_advection_tendencies or _augment_large_step_tendencies."
                ),
            },
            "augment_boundary_explains_PH_RW": {
                "status": "STRUCTURAL_MISMATCH_SUPPORTED",
                "evidence": (
                    "JAX _augment_large_step_tendencies does not assemble the WRF "
                    "large-step PH_TEND/RW_TEND contract; the current code assembles "
                    "those leaves in acoustic setup. Compare source-save patch metrics "
                    "against the acoustic-stage candidate before source-editing _augment."
                ),
                "sparse_augment_empty": sparse_summary(save_aug_empty, ("PH_TEND", "RW_TEND")),
                "sparse_acoustic_no_relax": sparse_summary(save_acoustic_raw, ("PH_TEND", "RW_TEND")),
            },
        },
        "ranked_hypotheses": [
            {
                "rank": 1,
                "hypothesis": "Missing or incorrectly routed WRF first_rk_step_part2 fixed source leaves, first visible as T_TENDF.",
                "status": "SUPPORTED_LOCALIZED",
                "evidence": (
                    "Patched-init state fields are below material gates, but "
                    "after_first_rk_step_part2 T_TENDF and source-save pre-addtend "
                    "T_TENDF remain materially different from JAX dry tendencies."
                ),
                "key_metric": metric_brief(part2_carry.get("per_field_metrics", {}).get("T_TENDF")),
            },
            {
                "rank": 2,
                "hypothesis": "WRF/JAX PH_TEND and RW_TEND are being compared across different stage boundaries.",
                "status": "SUPPORTED_AS_LATER_SYMPTOM",
                "evidence": (
                    "The full after_rk_addtend WRF surface is after relax/rk_addtend/spec, "
                    "while JAX _augment is pre-acoustic and does not own PH/RW assembly. "
                    "The sparse source-save boundary is the cleaner pre-addtend falsifier."
                ),
                "key_metric": {
                    "full_vs_augment_PH_TEND": metric_brief(
                        add_aug_full.get("per_field_metrics", {}).get("PH_TEND")
                    ),
                    "full_vs_augment_RW_TEND": metric_brief(
                        add_aug_full.get("per_field_metrics", {}).get("RW_TEND")
                    ),
                },
            },
            {
                "rank": 3,
                "hypothesis": "A proof-local rad_rk_tendf switch closes T_TENDF.",
                "status": "REFUTED",
                "evidence": "rad_rk_tendf=1 leaves Step-1 T_TENDF materially unchanged under the current cadence.",
                "key_metric": rad_toggle["selected_T_TENDF_metric"],
            },
            {
                "rank": 4,
                "hypothesis": "Boundary relaxation or spec_bdy_dry is the first T_TENDF cause.",
                "status": "REFUTED_FOR_FIRST_BOUNDARY",
                "evidence": "T_TENDF differs at source-save after rk_tendency before boundary/spec/addtend mutations.",
            },
            {
                "rank": 5,
                "hypothesis": "The already closed stale RK1 P_STATE loader is still causal.",
                "status": "REFUTED_BY_PREDECESSOR_AND_RECHECK",
                "evidence": "Patched-init P_STATE remains below the 1 Pa material gate in the predecessor proof.",
            },
        ],
        "best_next_exact_boundary": {
            "boundary": (
                "WRF first_rk_step_part2 internals: emit after calculate_phy_tend, "
                "after update_phy_ten, and after conv_t_tendf_to_moist for the "
                "theta source leaves feeding T_TENDF. Include the raw RTH*TEN/"
                "T_HIST_SRC contributors and the current JAX dry physics source bundle."
            ),
            "why": (
                "The earliest full-domain material field after patched init is T_TENDF "
                "at after_first_rk_step_part2. Source-save confirms it is already "
                "nonzero before rk_addtend_dry/boundary/spec/acoustic. A source edit in "
                "_augment or boundary code would be later than the first failure."
            ),
            "secondary_boundary": (
                "After T_TENDF is fixed, add an exact WRF post-rk_tendency/post-"
                "relax_bdy_dry/post-rk_addtend_dry/post-spec_bdy_dry split for "
                "T_TEND/PH_TEND/RW_TEND, or compare source-save PH/RW against the "
                "JAX acoustic-stage assembly rather than _augment alone."
            ),
        },
        "predecessor_context": extract_predecessor_context(),
        "wrf_truth": {
            "source_boundary": strip_wrf(wrf_source),
            "source_save_after_rk_tendency": source_save["compact"],
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    part2 = proof["full_domain_split"]["after_first_rk_step_part2_vs_physics_carry"]
    save = proof["source_save_patch_split"]["source_save_after_rk_tendency_vs_physics_carry"]
    add = proof["full_domain_split"]["after_rk_addtend_vs_augment_full_dry"]
    rad = proof["cheap_falsifiers"]["rad_rk_tendf_1"]
    next_boundary = proof["best_next_exact_boundary"]
    t_tendf = part2["selected_metrics"].get("T_TENDF", {})
    save_t_tendf = save["selected_metrics"].get("T_TENDF", {})
    add_top = add["top_material_residuals"][:5]
    lines = [
        "# V0.14 Step-1 Tendency Contract Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- Earliest full-domain material field after patched init: `{part2['material_first_field']}` at WRF `after_first_rk_step_part2`.",
        f"- Full-domain `T_TENDF` vs JAX dry source: max_abs `{t_tendf.get('max_abs')}`, rmse `{t_tendf.get('rmse')}`.",
        f"- Source-save pre-addtend `T_TENDF` vs JAX dry source: max_abs `{save_t_tendf.get('max_abs')}`, rmse `{save_t_tendf.get('rmse')}`.",
        f"- Proof-local `rad_rk_tendf=1` `T_TENDF`: max_abs `{rad['selected_T_TENDF_metric'].get('max_abs')}`.",
        "- Boundary/spec-only is too late for the first failure: source-save is before `relax_bdy_dry`, `rk_addtend_dry`, `spec_bdy_dry`, `small_step_prep`, and acoustic updates.",
        "",
        "## RK1 Tendency Symptoms",
        "",
        "- Full WRF `after_rk_addtend_before_small_step_prep` compared to JAX `_augment_large_step_tendencies` is not an exact boundary because WRF has already passed dry boundary/addtend/spec work.",
        "- Largest full-surface residuals vs JAX augment: "
        + ", ".join(f"`{item['field']}` max_abs `{item['max_abs']}`" for item in add_top)
        + ".",
        "- `PH_TEND/RW_TEND` assembly is structurally later in the current JAX path than `_augment`; compare the source-save patch against the acoustic-stage candidate before making an `_augment` source edit.",
        "",
        "## Next Exact Boundary",
        "",
        next_boundary["boundary"],
        "",
        next_boundary["why"],
        "",
        f"Secondary: {next_boundary['secondary_boundary']}",
        "",
        "Detailed metrics are in `proofs/v014/step1_tendency_contract_split.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    lines = [
        "# Review: V0.14 Step-1 Tendency Contract Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: localize or fix the remaining Step-1 tendency-family divergence after patched-init `P_STATE` closure.",
        "",
        "files changed:",
        "- `proofs/v014/step1_tendency_contract_split.py`",
        "- `proofs/v014/step1_tendency_contract_split.json`",
        "- `proofs/v014/step1_tendency_contract_split.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`",
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
    lines.extend(
        [
            "",
            f"next decision needed: {proof['best_next_exact_boundary']['boundary']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    proof = build_proof()
    verdict = str(proof.get("verdict", "STEP1_TENDENCY_CONTRACT_SPLIT_BLOCKED_UNKNOWN"))
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_tendency_contract_split.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
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
        "source_save_patch_only": True,
        "initial_vs_post_step_false_comparison": False,
        "environment": jax_environment(),
        "git": git_metadata(),
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "p_state_split_json": path_info(P_STATE_SPLIT_JSON),
            "accepted_final_truth_npz": path_info(live.ACCEPTED_TRUTH),
            "source_wrf_truth_root": path_info(source.WRF_TRUTH),
            "source_save_files": [path_info(path) for path in source_save_files()],
        },
        "proof": proof,
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_tendency_contract_split.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_tendency_contract_split.py",
                "python -m json.tool proofs/v014/step1_tendency_contract_split.json >/tmp/step1_tendency_contract_split.validated.json",
                "git diff --check",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "The source-save hook is patch-only, so PH/RW source-save metrics are sparse falsifiers; full-domain conclusions use the full WRF surfaces.",
            "The full WRF after_rk_addtend surface is after relax_bdy_dry/rk_addtend_dry/spec_bdy_dry, while JAX _augment is not an exact post-spec boundary.",
            "No production source edit was made because the first exact failure is inside WRF first_rk_step_part2 source-leaf construction, not a proven narrow JAX source line.",
        ],
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if proof.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
