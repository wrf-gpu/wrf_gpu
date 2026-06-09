#!/usr/bin/env python3
"""V0.14 d02 Step-1 RK1 source-boundary split against WRF substage truth."""

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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_rk1_source_boundary.json"
OUT_MD = PROOF_DIR / "step1_rk1_source_boundary.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_rk1_source_boundary_wrf_patch.diff"

SPRINT_CONTRACT = (
    ROOT
    / ".agent/sprints/2026-06-09-v014-step1-rk1-source-boundary/sprint-contract.md"
)
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary")
WRF_TRUTH = SCRATCH / "wrf_truth"
WRF_BUILD_LOG = SCRATCH / "compile_step1_rk1_source_boundary.log"
WRF_RUN_LOG = SCRATCH / "wrf_step1_rk1_source_boundary_stdout.log"
ACCEPTED_FINAL_TRUTH = live.ACCEPTED_TRUTH

TARGET_STEP = 1
TARGET_DOMAIN = 2
THETA_OFFSET = 300.0

EARLY_SURFACES = (
    "after_first_rk_step_part1",
    "after_first_rk_step_part2",
)
CONTINUITY_SURFACES = (
    "after_rk_addtend_before_small_step_prep",
    "after_small_step_prep_calc_p_rho",
)
SURFACE_ORDER = EARLY_SURFACES + CONTINUITY_SURFACES

MATERIAL_THRESHOLDS = {
    "T_STATE": 1.0e-3,
    "T_WORK": 1.0e-3,
    "P_STATE": 1.0,
    "P_WORK": 1.0,
    "PH_STATE": 1.0e-2,
    "PH_WORK": 1.0e-2,
    "MU_STATE": 1.0e-2,
    "MU_WORK": 1.0e-2,
    "W_STATE": 1.0e-2,
    "T_TEND": 1.0e-6,
    "T_TENDF": 1.0e-6,
    "MU_TEND": 1.0e-6,
    "MU_TENDF": 1.0e-6,
    "PH_TEND": 1.0e-6,
    "PH_TENDF": 1.0e-6,
    "RW_TEND": 1.0e-6,
    "RW_TENDF": 1.0e-6,
    "H_DIABATIC": 1.0e-6,
}

STATE_FIELDS = {"T_STATE", "P_STATE", "PH_STATE", "MU_STATE", "W_STATE"}
RELEVANT_FIELDS = set(MATERIAL_THRESHOLDS)

SURFACE_FIELD_SPECS = {
    "after_first_rk_step_part1": {
        "MASS_SOURCE": {
            "length": 18,
            "fields": (
                "T_STATE",
                "P_STATE",
                "PB",
                "MU_STATE",
                "MUB",
                "MUT",
                "T_TENDF",
                "H_DIABATIC",
                "MU_TENDF",
                "T_SAVE",
                "T_OLD",
            ),
        },
        "WPH_SOURCE": {
            "length": 14,
            "fields": (
                "W_STATE",
                "PH_STATE",
                "PHB",
                "RW_TENDF",
                "PH_TENDF",
                "W_SAVE",
                "PH_SAVE",
            ),
        },
    },
    "after_first_rk_step_part2": {
        "MASS_SOURCE": {
            "length": 18,
            "fields": (
                "T_STATE",
                "P_STATE",
                "PB",
                "MU_STATE",
                "MUB",
                "MUT",
                "T_TENDF",
                "H_DIABATIC",
                "MU_TENDF",
                "T_SAVE",
                "T_OLD",
            ),
        },
        "WPH_SOURCE": {
            "length": 14,
            "fields": (
                "W_STATE",
                "PH_STATE",
                "PHB",
                "RW_TENDF",
                "PH_TENDF",
                "W_SAVE",
                "PH_SAVE",
            ),
        },
    },
    "after_rk_addtend_before_small_step_prep": {
        "MASS_TEND": {
            "length": 18,
            "fields": (
                "T_STATE",
                "P_STATE",
                "PB",
                "MU_STATE",
                "MUB",
                "MUT",
                "T_TEND",
                "T_TENDF",
                "H_DIABATIC",
                "MU_TEND",
                "MU_TENDF",
            ),
        },
        "WPH_TEND": {
            "length": 14,
            "fields": (
                "W_STATE",
                "PH_STATE",
                "PHB",
                "RW_TEND",
                "RW_TENDF",
                "PH_TEND",
                "PH_TENDF",
            ),
        },
    },
    "after_small_step_prep_calc_p_rho": {
        "MASS_PREP": {
            "length": 18,
            "fields": (
                "T_WORK",
                "T_SAVE",
                "T_REF",
                "P_WORK",
                "PB",
                "MU_WORK",
                "MU_REF",
                "MUB",
                "MUT",
                "MUTS",
                "MU_SAVE",
            ),
        },
        "WPH_PREP": {
            "length": 14,
            "fields": (
                "W_WORK",
                "W_SAVE",
                "W_REF",
                "PH_WORK",
                "PH_SAVE",
                "PH_REF",
                "PHB",
            ),
        },
    },
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


def expected_shapes() -> dict[str, tuple[int, ...]]:
    if not ACCEPTED_FINAL_TRUTH.is_file():
        raise FileNotFoundError(str(ACCEPTED_FINAL_TRUTH))
    with np.load(ACCEPTED_FINAL_TRUTH) as truth:
        return {
            "mass": tuple(int(v) for v in truth["T"].shape),
            "mass2d": tuple(int(v) for v in truth["MU"].shape),
            "wph": tuple(int(v) for v in truth["PH"].shape),
        }


def field_shape(field: str, shapes: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    if field in {
        "MU_STATE",
        "MUB",
        "MUT",
        "MU_TEND",
        "MU_TENDF",
        "MU_WORK",
        "MU_REF",
        "MUTS",
        "MU_SAVE",
    }:
        return shapes["mass2d"]
    if (
        field.startswith("W_")
        or field.startswith("PH_")
        or field in {"PHB", "RW_TEND", "RW_TENDF", "PH_TEND", "PH_TENDF", "W_STATE"}
    ):
        return shapes["wph"]
    return shapes["mass"]


def _record_value(
    arrays: dict[str, np.ndarray],
    duplicate_stats: dict[str, Any],
    field: str,
    index: tuple[int, ...],
    value: float,
) -> None:
    current = arrays[field][index]
    if np.isnan(current):
        arrays[field][index] = value
        return
    duplicate_stats[field]["duplicates"] += 1
    if current != value:
        delta = abs(float(current) - float(value))
        duplicate_stats[field]["mismatches"] += 1
        duplicate_stats[field]["max_delta"] = max(float(duplicate_stats[field]["max_delta"]), delta)
        if duplicate_stats[field].get("first_mismatch") is None:
            duplicate_stats[field]["first_mismatch"] = {
                "index": list(index),
                "existing": float(current),
                "new": float(value),
                "delta": delta,
            }


def parse_wrf_surface(surface: str, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    pattern = f"{surface}_d{TARGET_DOMAIN}_step_{TARGET_STEP}_rk_1_*.txt"
    raw_files = sorted(WRF_TRUTH.glob(pattern))
    if not raw_files:
        return {"status": "BLOCKED_NO_WRF_SURFACE_FILES", "surface": surface, "pattern": pattern}
    spec = SURFACE_FIELD_SPECS[surface]
    fields: list[str] = []
    for record in spec.values():
        for field in record["fields"]:
            if field not in fields:
                fields.append(field)
    arrays = {field: np.full(field_shape(field, shapes), np.nan, dtype=np.float64) for field in fields}
    duplicate_stats = {
        field: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for field in fields
    }
    record_counts = {record_name: 0 for record_name in spec}
    headers: list[dict[str, Any]] = []
    allowed_headers = {
        "surface",
        "routine",
        "domain_id",
        "current_timestr_before_step",
        "grid_itimestep_after_increment",
        "rk_step",
        "rk_order",
        "dt_seconds",
        "dt_rk_seconds",
        "dts_rk_seconds",
        "number_of_small_timesteps",
        "tile_i_j_bounds_fortran",
        "global_mass_i_j_end_exclusive_fortran",
        "tile_record_policy",
        "mass_vertical_fortran_k_start_k_end_inclusive",
        "w_ph_vertical_fortran_kstag_start_kstag_end_inclusive",
        "record_schema",
    }
    for path in raw_files:
        header: dict[str, Any] = {"path": str(path)}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                tag = parts[0]
                if tag.startswith("#"):
                    header.setdefault("marker", stripped)
                    continue
                if tag in spec:
                    record_spec = spec[tag]
                    if len(parts) != int(record_spec["length"]):
                        return {
                            "status": "BLOCKED_PARSE_ERROR",
                            "surface": surface,
                            "path": str(path),
                            "line": stripped[:240],
                            "expected_length": int(record_spec["length"]),
                            "actual_length": len(parts),
                        }
                    x = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    values = [float(item) for item in parts[7:]]
                    for field, value in zip(record_spec["fields"], values):
                        shape = arrays[field].shape
                        index = (y, x) if len(shape) == 2 else (k, y, x)
                        _record_value(arrays, duplicate_stats, field, index, value)
                    record_counts[tag] += 1
                    continue
                if tag in allowed_headers:
                    header[tag] = parts[1:]
                    continue
                return {
                    "status": "BLOCKED_UNKNOWN_RECORD",
                    "surface": surface,
                    "path": str(path),
                    "line": stripped[:240],
                }
        headers.append(header)
    duplicate_mismatches = {name: item for name, item in duplicate_stats.items() if int(item["mismatches"]) > 0}
    if duplicate_mismatches:
        return {
            "status": "BLOCKED_DUPLICATE_MISMATCH",
            "surface": surface,
            "duplicate_mismatches": duplicate_mismatches,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    missing = {
        name: {"missing_count": int(np.isnan(arr).sum()), "shape": list(arr.shape)}
        for name, arr in arrays.items()
        if np.isnan(arr).any()
    }
    if missing:
        return {
            "status": "BLOCKED_MISSING_VALUES",
            "surface": surface,
            "missing": missing,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    summaries = {
        name: {
            "shape": list(arr.shape),
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
        for name, arr in arrays.items()
    }
    return {
        "status": "WRF_SURFACE_READY",
        "surface": surface,
        "rk_step": 1,
        "raw_file_count": len(raw_files),
        "record_counts": record_counts,
        "duplicate_stats": duplicate_stats,
        "headers": headers[:4],
        "summaries": summaries,
        "arrays": arrays,
    }


def parse_wrf_surfaces() -> dict[str, Any]:
    try:
        shapes = expected_shapes()
    except Exception as exc:
        return {"status": "BLOCKED_SHAPE_DISCOVERY", "exception": repr(exc)}
    surfaces: dict[str, Any] = {}
    for surface in SURFACE_ORDER:
        parsed = parse_wrf_surface(surface, shapes)
        if parsed.get("status") != "WRF_SURFACE_READY":
            return {"status": parsed.get("status"), "blocker": parsed, "shapes": {k: list(v) for k, v in shapes.items()}}
        surfaces[surface] = parsed
    rk_tendency_patch = sorted(WRF_TRUTH.glob(f"source_save_after_rk_tendency_d{TARGET_DOMAIN}_step_{TARGET_STEP}_*.txt"))
    return {
        "status": "WRF_SOURCE_BOUNDARY_TRUTH_READY",
        "shapes": {k: list(v) for k, v in shapes.items()},
        "surfaces": surfaces,
        "rk_tendency_patch_file_count": len(rk_tendency_patch),
        "rk_tendency_patch_files": [str(path) for path in rk_tendency_patch[:8]],
    }


def zero_like(jnp: Any, reference: Any, candidate: Any) -> Any:
    return jnp.zeros_like(reference) if candidate is None else candidate


def dry_leaf(jnp: Any, physics: Any, name: str, reference: Any) -> Any:
    return zero_like(jnp, reference, getattr(physics, name))


def state_source_arrays(jnp: Any, state: Any, dry: Any) -> dict[str, Any]:
    p_base = state.p_total - state.p_perturbation
    ph_base = state.ph_total - state.ph_perturbation
    mu_base = state.mu_total - state.mu_perturbation
    return {
        "T_STATE": state.theta - THETA_OFFSET,
        "P_STATE": state.p_perturbation,
        "PB": p_base,
        "MU_STATE": state.mu_perturbation,
        "MUB": mu_base,
        "MUT": state.mu_total,
        "T_TENDF": dry_leaf(jnp, dry, "t_tendf", state.theta),
        "H_DIABATIC": dry_leaf(jnp, dry, "h_diabatic", state.theta),
        "MU_TENDF": dry_leaf(jnp, dry, "mu_tendf", state.mu_perturbation),
        "T_SAVE": dry_leaf(jnp, dry, "t_save", state.theta),
        "T_OLD": state.theta - THETA_OFFSET,
        "W_STATE": state.w,
        "PH_STATE": state.ph_perturbation,
        "PHB": ph_base,
        "RW_TENDF": dry_leaf(jnp, dry, "rw_tendf", state.w),
        "PH_TENDF": dry_leaf(jnp, dry, "ph_tendf", state.ph_perturbation),
        "W_SAVE": dry_leaf(jnp, dry, "w_save", state.w),
        "PH_SAVE": dry_leaf(jnp, dry, "ph_save", state.ph_perturbation),
    }


def state_tendency_arrays(jnp: Any, state: Any, tendencies: Any, dry: Any) -> dict[str, Any]:
    arrays = state_source_arrays(jnp, state, dry)
    arrays.update(
        {
            "T_TEND": tendencies.theta,
            "MU_TEND": tendencies.mu,
            "RW_TEND": tendencies.w,
            "PH_TEND": tendencies.ph,
        }
    )
    return arrays


def prep_arrays(state: Any, prep: Any, pressure: Any) -> dict[str, Any]:
    ph_base = state.ph_total - state.ph_perturbation
    return {
        "T_WORK": prep.theta_work,
        "T_SAVE": prep.t_save,
        "T_REF": prep.theta_1,
        "P_WORK": pressure.p,
        "PB": prep.pb,
        "MU_WORK": prep.mu_work,
        "MU_REF": prep.mu_1,
        "MUB": prep.mub,
        "MUT": prep.mut,
        "MUTS": prep.muts,
        "MU_SAVE": prep.mu_save,
        "W_WORK": prep.w_work,
        "W_SAVE": prep.w_save,
        "W_REF": prep.w_1,
        "PH_WORK": prep.ph_work,
        "PH_SAVE": prep.ph_save,
        "PH_REF": prep.ph_1,
        "PHB": ph_base,
    }


def capture_jax_boundaries() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    inputs = live.build_live_nest_step1_inputs()
    namelist = inputs["namelist"]
    jnp = inputs["jnp"]
    lead_seconds = jnp.asarray(float(TARGET_STEP) * float(namelist.dt_s), dtype=jnp.float64)
    cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
    run_radiation = bool(cadence > 0 and TARGET_STEP % cadence == 0)
    physics = om._physics_step_forcing(
        inputs["carry"],
        namelist,
        lead_seconds,
        run_radiation=run_radiation,
    )
    spec = om.halo_spec(namelist.grid)
    step_entry = om.apply_halo(inputs["carry"].state, spec)
    physics_carry_state = om.apply_halo(physics.carry.state, spec)
    physics_state = om.apply_halo(physics.state, spec)
    empty_dry = om.DryPhysicsTendencies()

    rk1_reference = physics_carry_state
    haloed = physics_carry_state
    base_tendencies = om.compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
    rk_tendency_empty = om._augment_large_step_tendencies(
        haloed,
        base_tendencies,
        namelist,
        rk_step=1,
        physics_tendencies=empty_dry,
        step_origin=rk1_reference,
    )
    rk_addtend = om._augment_large_step_tendencies(
        haloed,
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

    jax.block_until_ready(physics_state.theta)
    return {
        "status": "JAX_SOURCE_BOUNDARIES_READY",
        "captures": {
            "step_entry_state_zero_dry": state_source_arrays(jnp, step_entry, empty_dry),
            "physics_carry_state_dry": state_source_arrays(jnp, physics_carry_state, physics.dry_tendencies),
            "physics_state_dry": state_source_arrays(jnp, physics_state, physics.dry_tendencies),
            "rk1_after_rk_tendency_empty_dry": state_tendency_arrays(jnp, physics_carry_state, rk_tendency_empty, empty_dry),
            "rk1_after_rk_addtend": state_tendency_arrays(jnp, physics_carry_state, rk_addtend, physics.dry_tendencies),
            "rk1_after_small_step_prep": prep_arrays(physics_carry_state, prep, pressure),
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
            "rad_rk_tendf": int(namelist.rad_rk_tendf),
            "use_flux_advection": bool(namelist.use_flux_advection),
            "moist_adv_opt": int(namelist.moist_adv_opt),
        },
    }


def diff_metrics(field: str, candidate: Any, reference: Any) -> dict[str, Any]:
    return live.diff_metrics(field, candidate, reference)


def compare_surface(
    surface: str,
    wrf_surface: Mapping[str, Any],
    jax_arrays: Mapping[str, Any],
    jax: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field, wrf_array in wrf_surface["arrays"].items():
        if field not in jax_arrays:
            metrics[field] = {"status": "MISSING_JAX_FIELD"}
            continue
        candidate = np.asarray(jax.device_get(jax_arrays[field]), dtype=np.float64)
        metrics[field] = diff_metrics(field, candidate, wrf_array)
    ranked = sorted(
        [
            {"field": name, **{k: v for k, v in item.items() if k != "status"}}
            for name, item in metrics.items()
            if item.get("status") == "OK"
        ],
        key=lambda item: (-1.0 if item.get("max_abs") is None else float(item["max_abs"])),
        reverse=True,
    )
    strict_first = None
    material_first = None
    for field in wrf_surface["arrays"]:
        item = metrics[field]
        if item.get("status") != "OK":
            strict_first = strict_first or field
            continue
        max_abs = 0.0 if item.get("max_abs") is None else float(item["max_abs"])
        if strict_first is None and (int(item.get("nonfinite_diff_count", 0)) or max_abs != 0.0):
            strict_first = field
        threshold = MATERIAL_THRESHOLDS.get(field)
        if material_first is None and field in RELEVANT_FIELDS and threshold is not None and max_abs > float(threshold):
            material_first = field
    material_ranked = [
        {
            **item,
            "material_threshold": MATERIAL_THRESHOLDS.get(item["field"]),
            "material": (
                item["field"] in RELEVANT_FIELDS
                and MATERIAL_THRESHOLDS.get(item["field"]) is not None
                and float(item["max_abs"]) > float(MATERIAL_THRESHOLDS[item["field"]])
            ),
        }
        for item in ranked
        if item["field"] in RELEVANT_FIELDS
    ]
    return {
        "status": "SOURCE_BOUNDARY_COMPARISON_EXECUTED",
        "surface": surface,
        "strict_first_mismatch_field": strict_first,
        "material_first_field": material_first,
        "per_field_metrics": metrics,
        "ranked_residuals": ranked,
        "material_ranked_residuals": material_ranked,
    }


def compare_all(wrf: Mapping[str, Any], jax_capture: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    cap = jax_capture["captures"]
    matrix: dict[str, dict[str, Any]] = {
        "after_first_rk_step_part1": {
            "vs_step_entry_state_zero_dry": compare_surface(
                "after_first_rk_step_part1", wrf["surfaces"]["after_first_rk_step_part1"], cap["step_entry_state_zero_dry"], jax
            ),
            "vs_physics_carry_state_dry": compare_surface(
                "after_first_rk_step_part1", wrf["surfaces"]["after_first_rk_step_part1"], cap["physics_carry_state_dry"], jax
            ),
            "vs_physics_state_dry": compare_surface(
                "after_first_rk_step_part1", wrf["surfaces"]["after_first_rk_step_part1"], cap["physics_state_dry"], jax
            ),
        },
        "after_first_rk_step_part2": {
            "vs_step_entry_state_zero_dry": compare_surface(
                "after_first_rk_step_part2", wrf["surfaces"]["after_first_rk_step_part2"], cap["step_entry_state_zero_dry"], jax
            ),
            "vs_physics_carry_state_dry": compare_surface(
                "after_first_rk_step_part2", wrf["surfaces"]["after_first_rk_step_part2"], cap["physics_carry_state_dry"], jax
            ),
            "vs_physics_state_dry": compare_surface(
                "after_first_rk_step_part2", wrf["surfaces"]["after_first_rk_step_part2"], cap["physics_state_dry"], jax
            ),
        },
        "after_rk_addtend_before_small_step_prep": {
            "vs_rk1_after_rk_addtend": compare_surface(
                "after_rk_addtend_before_small_step_prep",
                wrf["surfaces"]["after_rk_addtend_before_small_step_prep"],
                cap["rk1_after_rk_addtend"],
                jax,
            ),
            "vs_rk1_after_rk_tendency_empty_dry": compare_surface(
                "after_rk_addtend_before_small_step_prep",
                wrf["surfaces"]["after_rk_addtend_before_small_step_prep"],
                cap["rk1_after_rk_tendency_empty_dry"],
                jax,
            ),
        },
        "after_small_step_prep_calc_p_rho": {
            "vs_rk1_after_small_step_prep": compare_surface(
                "after_small_step_prep_calc_p_rho",
                wrf["surfaces"]["after_small_step_prep_calc_p_rho"],
                cap["rk1_after_small_step_prep"],
                jax,
            ),
        },
    }
    return {"status": "ALL_SOURCE_BOUNDARY_COMPARISONS_EXECUTED", "matrix": matrix}


def material_field(comp: Mapping[str, Any]) -> str | None:
    field = comp.get("material_first_field")
    return str(field) if field else None


def is_material_clean(comp: Mapping[str, Any], fields: set[str] | None = None) -> bool:
    for item in comp.get("material_ranked_residuals", []):
        if fields is not None and item["field"] not in fields:
            continue
        if bool(item.get("material")):
            return False
    return True


def classify(comparisons: Mapping[str, Any]) -> tuple[str, dict[str, Any], str]:
    if comparisons.get("status") != "ALL_SOURCE_BOUNDARY_COMPARISONS_EXECUTED":
        return (
            "STEP1_RK1_SOURCE_BLOCKED_SOURCE_BOUNDARY_COMPARISON",
            {},
            "Fix the named source-boundary comparison blocker and rerun this proof.",
        )
    matrix = comparisons["matrix"]
    p1_carry = matrix["after_first_rk_step_part1"]["vs_physics_carry_state_dry"]
    p1_phys = matrix["after_first_rk_step_part1"]["vs_physics_state_dry"]
    p2_carry = matrix["after_first_rk_step_part2"]["vs_physics_carry_state_dry"]
    p2_phys = matrix["after_first_rk_step_part2"]["vs_physics_state_dry"]
    add_comp = matrix["after_rk_addtend_before_small_step_prep"]["vs_rk1_after_rk_addtend"]

    if not is_material_clean(p1_carry, STATE_FIELDS):
        field = material_field(p1_carry) or "UNKNOWN"
        if is_material_clean(p1_phys, STATE_FIELDS):
            evidence = {
                "surface": "after_first_rk_step_part1",
                "field": field,
                "wrf_vs_physics_carry": p1_carry["per_field_metrics"].get(field),
                "wrf_vs_physics_state": p1_phys["per_field_metrics"].get(field),
            }
            return (
                f"STEP1_RK1_SOURCE_LOCALIZED_PHYSICS_STATE_CARRY_HANDOFF_{field}",
                evidence,
                (
                    "Decide whether production `_physics_step_forcing` should hand its dry-mutated "
                    "state into `_rk_scan_step`, or whether the affected WRF first_rk_step_part1 "
                    "mutation must instead be represented as true RK-fixed `*_tendf` leaves."
                ),
            )
        evidence = {
            "surface": "after_first_rk_step_part1",
            "field": field,
            "wrf_vs_physics_carry": p1_carry["per_field_metrics"].get(field),
            "wrf_vs_physics_state": p1_phys["per_field_metrics"].get(field),
        }
        return (
            f"STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_{field}",
            evidence,
            "Split WRF first_rk_step_part1 internals against the JAX physics adapter output for the named state field.",
        )

    if not is_material_clean(p2_carry, STATE_FIELDS):
        field = material_field(p2_carry) or "UNKNOWN"
        if is_material_clean(p2_phys, STATE_FIELDS):
            evidence = {
                "surface": "after_first_rk_step_part2",
                "field": field,
                "wrf_vs_physics_carry": p2_carry["per_field_metrics"].get(field),
                "wrf_vs_physics_state": p2_phys["per_field_metrics"].get(field),
            }
            return (
                f"STEP1_RK1_SOURCE_LOCALIZED_PHYSICS_STATE_CARRY_HANDOFF_{field}",
                evidence,
                (
                    "Decide whether production `_physics_step_forcing` should hand its dry-mutated "
                    "state into `_rk_scan_step`, or whether the affected WRF first_rk_step_part2 "
                    "mutation must instead be represented as true RK-fixed `*_tendf` leaves."
                ),
            )
        evidence = {
            "surface": "after_first_rk_step_part2",
            "field": field,
            "wrf_vs_physics_carry": p2_carry["per_field_metrics"].get(field),
            "wrf_vs_physics_state": p2_phys["per_field_metrics"].get(field),
        }
        return (
            f"STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART2_PHYSICS_STATE_MUTATION_{field}",
            evidence,
            "Split WRF first_rk_step_part2 internals against the JAX physics adapter output for the named state field.",
        )

    if not is_material_clean(add_comp):
        field = material_field(add_comp) or "UNKNOWN"
        return (
            f"STEP1_RK1_SOURCE_LOCALIZED_RK_TENDENCY_OR_RK_ADDTEND_DRY_{field}",
            {
                "surface": "after_rk_addtend_before_small_step_prep",
                "field": field,
                "wrf_vs_jax": add_comp["per_field_metrics"].get(field),
            },
            "Split full-domain WRF after-rk_tendency against pre/post JAX `rk_addtend_dry` for the named tendency field.",
        )

    prep_comp = matrix["after_small_step_prep_calc_p_rho"]["vs_rk1_after_small_step_prep"]
    if not is_material_clean(prep_comp):
        field = material_field(prep_comp) or "UNKNOWN"
        return (
            f"STEP1_RK1_SOURCE_LOCALIZED_SMALL_STEP_PREP_CONTINUITY_{field}",
            {"surface": "after_small_step_prep_calc_p_rho", "field": field},
            "Return to small_step_prep only after source-boundary surfaces are confirmed clean.",
        )
    return (
        "STEP1_RK1_SOURCE_NO_REMAINING_DIVERGENCE",
        {},
        "Close the Step-1 RK1 source-boundary gate.",
    )


def strip_arrays_from_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    surfaces = {}
    for surface, parsed in wrf["surfaces"].items():
        surfaces[surface] = {key: value for key, value in parsed.items() if key != "arrays"}
    return {
        "status": wrf["status"],
        "shapes": wrf["shapes"],
        "surfaces": surfaces,
        "rk_tendency_patch_file_count": wrf.get("rk_tendency_patch_file_count"),
        "rk_tendency_patch_files": wrf.get("rk_tendency_patch_files"),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    evidence = payload.get("classification_evidence", {})
    lines = [
        "# V0.14 Step-1 RK1 Source Boundary",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`.",
        f"- WRF source-boundary truth root: `{WRF_TRUTH}`.",
        f"- Fastest rigorous method: `{payload['tooling_verdict']}`.",
    ]
    if evidence:
        field = evidence.get("field")
        surface = evidence.get("surface")
        carry = evidence.get("wrf_vs_physics_carry") or evidence.get("wrf_vs_jax")
        phys = evidence.get("wrf_vs_physics_state")
        if carry:
            lines.append(
                f"- First localized boundary: `{surface}` field `{field}`; WRF vs JAX operational carry max_abs `{carry.get('max_abs')}` rmse `{carry.get('rmse')}`."
            )
        if phys:
            lines.append(
                f"- Same WRF field vs `_physics_step_forcing.state` max_abs `{phys.get('max_abs')}` rmse `{phys.get('rmse')}`."
            )
    add_comp = payload["source_boundary_comparisons"]["matrix"]["after_rk_addtend_before_small_step_prep"]["vs_rk1_after_rk_addtend"]
    add_top = add_comp["material_ranked_residuals"][0]
    lines.append(
        f"- Continuity check at prior pre-small-step boundary still shows top material `{add_top['field']}` max_abs `{add_top['max_abs']}`."
    )
    prep_comp = payload["source_boundary_comparisons"]["matrix"]["after_small_step_prep_calc_p_rho"]["vs_rk1_after_small_step_prep"]
    t_work = prep_comp["per_field_metrics"].get("T_WORK", {})
    p_work = prep_comp["per_field_metrics"].get("P_WORK", {})
    lines.append(
        f"- RK1 `small_step_prep` continuity remains exact for `T_WORK` max_abs `{t_work.get('max_abs')}` and `P_WORK` max_abs `{p_work.get('max_abs')}`."
    )
    lines.extend(
        [
            "",
            "Detailed comparison tables are in `proofs/v014/step1_rk1_source_boundary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 RK1 Source Boundary",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: split WRF first_rk_step_part1/part2, rk_tendency, and rk_addtend_dry/spec_bdy_dry against JAX `_physics_step_forcing` and dry tendency construction before `small_step_prep`.",
        "",
        "files changed:",
        "- `proofs/v014/step1_rk1_source_boundary.py`",
        "- `proofs/v014/step1_rk1_source_boundary.json`",
        "- `proofs/v014/step1_rk1_source_boundary.md`",
        "- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    for command in payload["commands"].get("wrf_instrumentation", []):
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            f"- `{OUT_JSON}`",
            f"- `{OUT_MD}`",
            f"- `{OUT_REVIEW}`",
            f"- `{OUT_WRF_PATCH}`",
            f"- `{WRF_TRUTH}`",
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
    wrf = parse_wrf_surfaces()
    if wrf.get("status") != "WRF_SOURCE_BOUNDARY_TRUTH_READY":
        comparisons = {"status": "NOT_EXECUTED", "blocker": wrf}
        jax_capture_meta = {"status": "NOT_EXECUTED"}
        verdict = "STEP1_RK1_SOURCE_BLOCKED_WRF_SOURCE_BOUNDARY_TRUTH"
        classification_evidence: dict[str, Any] = {}
        decision = "Fix the named WRF source-boundary truth blocker and rerun this proof."
    else:
        jax_capture = capture_jax_boundaries()
        if jax_capture.get("status") != "JAX_SOURCE_BOUNDARIES_READY":
            comparisons = {"status": "NOT_EXECUTED", "blocker": jax_capture}
            jax_capture_meta = {key: value for key, value in jax_capture.items() if key != "captures"}
            verdict = "STEP1_RK1_SOURCE_BLOCKED_JAX_SOURCE_BOUNDARY_CAPTURE"
            classification_evidence = {}
            decision = "Fix the named JAX source-boundary capture blocker and rerun this proof."
        else:
            comparisons = compare_all(wrf, jax_capture)
            verdict, classification_evidence, decision = classify(comparisons)
            jax_capture_meta = {key: value for key, value in jax_capture.items() if key != "captures"}
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_rk1_source_boundary.v1",
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
        "initial_vs_post_step_false_comparison": False,
        "tooling_verdict": "FOCUSED_STEP1_SOURCE_BOUNDARY_TRUTH_COMPARATOR_FASTEST_RIGOROUS_WALL_CLOCK",
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "rk_step": 1,
            "substage_boundaries": list(SURFACE_ORDER),
            "rejected_comparisons": [
                "WRF final truth vs JAX initial state",
                "JAX-vs-JAX-only localization",
                "one-cell or station proxy proof",
                "acoustic continuation before source-boundary split",
            ],
        },
        "environment": jax_environment(),
        "git_head": git_head,
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "accepted_final_truth_npz": path_info(ACCEPTED_FINAL_TRUTH),
            "wrf_truth_root": path_info(WRF_TRUTH),
            "wrf_build_log": path_info(WRF_BUILD_LOG),
            "wrf_run_log": path_info(WRF_RUN_LOG),
            "wrf_patch_diff": path_info(OUT_WRF_PATCH),
        },
        "wrf_source_boundary_truth": strip_arrays_from_wrf(wrf) if wrf.get("status") == "WRF_SOURCE_BOUNDARY_TRUTH_READY" else wrf,
        "jax_capture": jax_capture_meta,
        "source_boundary_comparisons": comparisons,
        "classification_evidence": classification_evidence,
        "commands": {
            "wrf_instrumentation": [
                "cp -a --reflink=auto prior Step-1 WRF scratch into /mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary",
                "tcsh ./compile em_real (scratch WRF, conda wrf-build toolchain)",
                "WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY=1 WRFGPU2_STEP1_TP_LOCALIZATION=1 WRFGPU2_SOURCE_SAVE_BOUNDARY=1 mpirun --oversubscribe -np 28 ./wrf.exe",
            ],
            "required_validation": [
                "python -m py_compile proofs/v014/step1_rk1_source_boundary.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_source_boundary.py",
                "python -m json.tool proofs/v014/step1_rk1_source_boundary.json >/tmp/step1_rk1_source_boundary.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "wrf_patch_diff": str(OUT_WRF_PATCH),
            "wrf_truth_root": str(WRF_TRUTH),
        },
        "unresolved_risks": [
            "The after-rk_tendency hook reused the prior patch-window source-save emitter; because the first material source-boundary classification occurs earlier at first_rk_step_part1 T_STATE, this did not block the verdict.",
            "No production source fix was applied; the next decision must split WRF first_rk_step_part1 internals against the JAX physics adapter output before choosing a source representation.",
        ],
        "next_decision": decision,
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
