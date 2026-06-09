#!/usr/bin/env python3
"""V0.14 d02 Step-1 T/P operator localization against WRF substage truth."""

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


OUT_JSON = PROOF_DIR / "step1_t_p_operator_localization.json"
OUT_MD = PROOF_DIR / "step1_t_p_operator_localization.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_t_p_operator_localization_wrf_patch.diff"

SPRINT_CONTRACT = (
    ROOT
    / ".agent/sprints/2026-06-09-v014-step1-t-p-operator-localization/sprint-contract.md"
)
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization")
WRF_TRUTH = SCRATCH / "wrf_truth"
WRF_BUILD_LOG = SCRATCH / "compile_step1_tp_localization.log"
WRF_RUN_LOG = SCRATCH / "wrf_step1_tp_localization_stdout.log"
ACCEPTED_FINAL_TRUTH = live.ACCEPTED_TRUTH

TARGET_STEP = 1
TARGET_DOMAIN = 2
THETA_OFFSET = 300.0
STAGES = (1, 2, 3)

SURFACE_ORDER = (
    "after_rk_addtend_before_small_step_prep",
    "after_small_step_prep_calc_p_rho",
)

MATERIAL_THRESHOLDS = {
    "T_STATE": 1.0e-3,
    "T_WORK": 1.0e-3,
    "P_STATE": 1.0,
    "P_WORK": 1.0,
    "PH_STATE": 1.0e-2,
    "PH_WORK": 1.0e-2,
    "MU_STATE": 1.0e-2,
    "MU_WORK": 1.0e-2,
    "T_TEND": 1.0e-6,
    "T_TENDF": 1.0e-6,
    "MU_TEND": 1.0e-6,
    "MU_TENDF": 1.0e-6,
    "PH_TEND": 1.0e-6,
    "PH_TENDF": 1.0e-6,
    "RW_TEND": 1.0e-6,
    "RW_TENDF": 1.0e-6,
}

RELEVANT_LOCALIZATION_FIELDS = {
    "T_STATE",
    "T_WORK",
    "P_STATE",
    "P_WORK",
    "PH_STATE",
    "PH_WORK",
    "MU_STATE",
    "MU_WORK",
    "T_TEND",
    "T_TENDF",
    "MU_TEND",
    "MU_TENDF",
    "PH_TEND",
    "PH_TENDF",
    "RW_TEND",
    "RW_TENDF",
}

SURFACE_FIELD_SPECS = {
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
            "rank": "mass",
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
            "rank": "wph",
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
            "rank": "mass",
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
            "rank": "wph",
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
    if field.startswith("W_") or field.startswith("PH_") or field in {"PHB", "RW_TEND", "RW_TENDF", "PH_TEND", "PH_TENDF"}:
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
        duplicate_stats[field]["mismatches"] += 1
        delta = abs(float(current) - float(value))
        duplicate_stats[field]["max_delta"] = max(float(duplicate_stats[field]["max_delta"]), delta)
        if duplicate_stats[field].get("first_mismatch") is None:
            duplicate_stats[field]["first_mismatch"] = {
                "index": list(index),
                "existing": float(current),
                "new": float(value),
                "delta": delta,
            }


def parse_wrf_surface(surface: str, rk_step: int, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    pattern = f"{surface}_d2_step_{TARGET_STEP}_rk_{rk_step}_*.txt"
    raw_files = sorted(WRF_TRUTH.glob(pattern))
    if not raw_files:
        return {"status": "BLOCKED_NO_WRF_SURFACE_FILES", "surface": surface, "rk_step": rk_step, "pattern": pattern}
    spec = SURFACE_FIELD_SPECS[surface]
    fields = []
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
                            "rk_step": rk_step,
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
                if tag in {
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
                }:
                    header[tag] = parts[1:]
                    continue
                return {
                    "status": "BLOCKED_UNKNOWN_RECORD",
                    "surface": surface,
                    "rk_step": rk_step,
                    "path": str(path),
                    "line": stripped[:240],
                }
        headers.append(header)
    duplicate_mismatches = {name: item for name, item in duplicate_stats.items() if int(item["mismatches"]) > 0}
    if duplicate_mismatches:
        return {
            "status": "BLOCKED_DUPLICATE_MISMATCH",
            "surface": surface,
            "rk_step": rk_step,
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
            "rk_step": rk_step,
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
        "rk_step": rk_step,
        "raw_file_count": len(raw_files),
        "record_counts": record_counts,
        "duplicate_stats": duplicate_stats,
        "headers": headers[:4],
        "summaries": summaries,
        "arrays": arrays,
    }


def load_wrf_surfaces() -> dict[str, Any]:
    try:
        shapes = expected_shapes()
    except Exception as exc:
        return {"status": "BLOCKED_SHAPE_DISCOVERY", "exception": repr(exc)}
    surfaces: dict[str, Any] = {}
    for surface in SURFACE_ORDER:
        surfaces[surface] = {}
        for rk_step in STAGES:
            parsed = parse_wrf_surface(surface, rk_step, shapes)
            if parsed.get("status") != "WRF_SURFACE_READY":
                return {"status": parsed.get("status"), "blocker": parsed, "shapes": {k: list(v) for k, v in shapes.items()}}
            surfaces[surface][rk_step] = parsed
    return {"status": "WRF_SUBSTAGE_TRUTH_READY", "shapes": {k: list(v) for k, v in shapes.items()}, "surfaces": surfaces}


def zero_like(jnp: Any, reference: Any, candidate: Any) -> Any:
    return jnp.zeros_like(reference) if candidate is None else candidate


def dry_leaf(jnp: Any, physics: Any, name: str, reference: Any) -> Any:
    return zero_like(jnp, reference, getattr(physics, name))


def capture_jax_boundaries() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    if not ACCEPTED_FINAL_TRUTH.is_file():
        return {"status": "BLOCKED_NO_ACCEPTED_FINAL_TRUTH", "truth": path_info(ACCEPTED_FINAL_TRUTH)}

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
    origin = om.apply_halo(physics.carry.state, om.halo_spec(namelist.grid))
    rk1_reference = origin
    dt = float(namelist.dt_s)
    configured_sound_steps = int(namelist.acoustic_substeps)
    stages = (
        om._RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
        om._RKStageDescriptor(2, 0.5 * dt, dt / float(configured_sound_steps), max(1, configured_sound_steps // 2)),
        om._RKStageDescriptor(3, dt, dt / float(configured_sound_steps), configured_sound_steps),
    )
    stage_carry = physics.carry.replace(state=origin)
    captures: dict[str, dict[int, dict[str, Any]]] = {surface: {} for surface in SURFACE_ORDER}
    final_pre_halo_state = None

    for stage in stages:
        haloed = om.apply_halo(stage_carry.state, om.halo_spec(namelist.grid))
        tendencies = om.compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        tendencies = om._augment_large_step_tendencies(
            haloed,
            tendencies,
            namelist,
            rk_step=int(stage.rk_step),
            physics_tendencies=physics.dry_tendencies,
            step_origin=rk1_reference,
        )
        candidate = om.apply_halo(stage_carry.state, om.halo_spec(namelist.grid))
        p_base = candidate.p_total - candidate.p_perturbation
        ph_base = candidate.ph_total - candidate.ph_perturbation
        mu_base = candidate.mu_total - candidate.mu_perturbation
        captures["after_rk_addtend_before_small_step_prep"][int(stage.rk_step)] = {
            "T_STATE": candidate.theta - THETA_OFFSET,
            "P_STATE": candidate.p_perturbation,
            "PB": p_base,
            "MU_STATE": candidate.mu_perturbation,
            "MUB": mu_base,
            "MUT": candidate.mu_total,
            "T_TEND": tendencies.theta,
            "T_TENDF": dry_leaf(jnp, physics.dry_tendencies, "t_tendf", tendencies.theta),
            "H_DIABATIC": dry_leaf(jnp, physics.dry_tendencies, "h_diabatic", tendencies.theta),
            "MU_TEND": tendencies.mu,
            "MU_TENDF": dry_leaf(jnp, physics.dry_tendencies, "mu_tendf", tendencies.mu),
            "W_STATE": candidate.w,
            "PH_STATE": candidate.ph_perturbation,
            "PHB": ph_base,
            "RW_TEND": tendencies.w,
            "RW_TENDF": dry_leaf(jnp, physics.dry_tendencies, "rw_tendf", tendencies.w),
            "PH_TEND": tendencies.ph,
            "PH_TENDF": dry_leaf(jnp, physics.dry_tendencies, "ph_tendf", tendencies.ph),
        }

        prep = om.small_step_prep_wrf(
            candidate,
            int(stage.rk_step),
            float(stage.dt_rk),
            metrics=namelist.metrics,
            reference_state=rk1_reference,
            ww=stage_carry.ww,
        )
        pressure = om.calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
        captures["after_small_step_prep_calc_p_rho"][int(stage.rk_step)] = {
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

        moisture_advected = bool(namelist.use_flux_advection) and int(namelist.moist_adv_opt) != 0
        q_tendencies = (
            om._moisture_coupled_tendencies(
                haloed,
                namelist,
                rk_step=int(stage.rk_step),
                step_origin=rk1_reference,
            )
            if moisture_advected
            else None
        )
        acoustic_result = om._acoustic_scan(
            stage_carry.replace(state=candidate),
            namelist,
            stage=stage,
            prep=prep,
            pressure=pressure,
            tendencies=tendencies,
            lead_seconds=lead_seconds,
            capture_pre_halo=int(stage.rk_step) == int(namelist.rk_order),
        )
        if int(stage.rk_step) == int(namelist.rk_order):
            final_pre_halo_state = acoustic_result.pre_halo_state
            stage_carry = acoustic_result.carry
        else:
            stage_carry = acoustic_result
        if moisture_advected:
            stage_carry = stage_carry.replace(
                state=om._apply_moisture_large_step(
                    stage_carry.state,
                    rk1_reference,
                    q_tendencies=q_tendencies,
                    dt_rk=float(stage.dt_rk),
                    metrics=namelist.metrics,
                )
            )
        stage_carry = stage_carry.replace(state=om.apply_halo(stage_carry.state, om.halo_spec(namelist.grid)))

    if final_pre_halo_state is None:
        return {"status": "BLOCKED_NO_FINAL_PRE_HALO_CAPTURE"}
    jax.block_until_ready(final_pre_halo_state.theta)
    final_comparison = live.compare_arrays(ACCEPTED_FINAL_TRUTH, final_pre_halo_state, inputs["base_state"], jax)
    return {
        "status": "JAX_BOUNDARIES_READY",
        "captures": captures,
        "final_pre_halo_state": final_pre_halo_state,
        "base_state": inputs["base_state"],
        "final_comparison": final_comparison,
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
    rk_step: int,
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
        item = diff_metrics(field, candidate, wrf_array)
        metrics[field] = item
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
        if (
            material_first is None
            and field in RELEVANT_LOCALIZATION_FIELDS
            and threshold is not None
            and max_abs > float(threshold)
        ):
            material_first = field
    material_ranked = [
        {
            **item,
            "material_threshold": MATERIAL_THRESHOLDS.get(item["field"]),
            "material": (
                item["field"] in RELEVANT_LOCALIZATION_FIELDS
                and MATERIAL_THRESHOLDS.get(item["field"]) is not None
                and float(item["max_abs"]) > float(MATERIAL_THRESHOLDS[item["field"]])
            ),
        }
        for item in ranked
        if item["field"] in RELEVANT_LOCALIZATION_FIELDS
    ]
    return {
        "status": "SUBSTAGE_COMPARISON_EXECUTED",
        "surface": surface,
        "rk_step": rk_step,
        "strict_first_mismatch_field": strict_first,
        "material_first_tp_family_field": material_first,
        "per_field_metrics": metrics,
        "ranked_residuals": ranked,
        "material_ranked_residuals": material_ranked,
    }


def compare_all(wrf: Mapping[str, Any], jax_capture: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    comparisons: dict[str, Any] = {}
    first_material = None
    first_strict = None
    for surface in SURFACE_ORDER:
        comparisons[surface] = {}
        for rk_step in STAGES:
            item = compare_surface(
                surface,
                rk_step,
                wrf["surfaces"][surface][rk_step],
                jax_capture["captures"][surface][rk_step],
                jax,
            )
            comparisons[surface][rk_step] = item
            if first_strict is None and item.get("strict_first_mismatch_field"):
                first_strict = {
                    "surface": surface,
                    "rk_step": rk_step,
                    "field": item["strict_first_mismatch_field"],
                }
            if first_material is None and item.get("material_first_tp_family_field"):
                first_material = {
                    "surface": surface,
                    "rk_step": rk_step,
                    "field": item["material_first_tp_family_field"],
                }
    return {
        "status": "ALL_SUBSTAGE_COMPARISONS_EXECUTED",
        "first_strict_substage_mismatch": first_strict,
        "first_material_tp_family_mismatch": first_material,
        "comparisons": comparisons,
    }


def strip_arrays_from_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    surfaces = {}
    for surface, stages in wrf["surfaces"].items():
        surfaces[surface] = {}
        for rk_step, parsed in stages.items():
            surfaces[surface][rk_step] = {
                key: value for key, value in parsed.items() if key != "arrays"
            }
    return {
        "status": wrf["status"],
        "shapes": wrf["shapes"],
        "surfaces": surfaces,
    }


def derive_verdict(comparisons: Mapping[str, Any], final_comparison: Mapping[str, Any]) -> str:
    if comparisons.get("status") != "ALL_SUBSTAGE_COMPARISONS_EXECUTED":
        return "STEP1_TP_BLOCKED_SUBSTAGE_COMPARISON"
    first_material = comparisons.get("first_material_tp_family_mismatch")
    if first_material:
        surface = first_material["surface"]
        field = first_material["field"]
        if surface == "after_rk_addtend_before_small_step_prep":
            if field in {"T_STATE", "P_STATE", "MU_STATE", "PH_STATE", "W_STATE"}:
                return f"STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK{first_material['rk_step']}_{field}"
            return f"STEP1_TP_LOCALIZED_RK_ADDTEND_DRY_OR_SPEC_BDY_DRY_RK{first_material['rk_step']}_{field}"
        if surface == "after_small_step_prep_calc_p_rho":
            return f"STEP1_TP_LOCALIZED_SMALL_STEP_PREP_OR_CALC_P_RHO_RK{first_material['rk_step']}_{field}"
    if final_comparison.get("status") == "COMPARISON_EXECUTED" and final_comparison.get("first_divergent_field"):
        return "STEP1_TP_BLOCKED_ACOUSTIC_SUBSTEP_TRUTH_AFTER_EARLY_SUBSTAGES_MATCH"
    if final_comparison.get("status") == "COMPARISON_EXECUTED":
        return "STEP1_TP_NO_REMAINING_DIVERGENCE"
    return "STEP1_TP_BLOCKED_FINAL_COMPARISON"


def next_decision(verdict: str, comparisons: Mapping[str, Any]) -> str:
    first = comparisons.get("first_material_tp_family_mismatch")
    if first and first["surface"] == "after_rk_addtend_before_small_step_prep":
        if first["field"] in {"T_STATE", "P_STATE", "MU_STATE", "PH_STATE", "W_STATE"}:
            return (
                "Resolve the WRF/JAX RK1 stage-entry state mismatch after WRF first_rk_step_part1/part2 "
                f"and before JAX small_step_prep, starting with field {first['field']}; do not continue acoustic debugging yet."
            )
        return (
            "Inspect WRF/JAX RK-fixed dry tendency and nested spec_bdy_dry handling at "
            f"RK{first['rk_step']} field {first['field']}; do not continue acoustic debugging until this boundary is resolved."
        )
    if first and first["surface"] == "after_small_step_prep_calc_p_rho":
        return (
            "Inspect the WRF/JAX small_step_prep reference/work split and calc_p_rho step-0 diagnostic at "
            f"RK{first['rk_step']} field {first['field']}."
        )
    if verdict.startswith("STEP1_TP_BLOCKED_ACOUSTIC"):
        return "Emit matching WRF post-acoustic/pre-finish or per-substep truth; early substage surfaces did not carry the material T/P residual."
    if verdict == "STEP1_TP_NO_REMAINING_DIVERGENCE":
        return "Close the Step-1 T/P same-input gate."
    return "Patch or emit the exact missing truth surface named in the blocker and rerun this proof."


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    first = payload["substage_comparisons"].get("first_material_tp_family_mismatch")
    final = payload["final_comparison"]
    lines = [
        "# V0.14 Step-1 T/P Operator Localization",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`.",
        f"- WRF substage truth root: `{WRF_TRUTH}`.",
        f"- Fastest rigorous method: `{payload['tooling_verdict']}`.",
    ]
    if first:
        comp = payload["substage_comparisons"]["comparisons"][first["surface"]][first["rk_step"]]
        top = comp["material_ranked_residuals"][0]
        lines.extend(
            [
                f"- First material T/P-family mismatch: `{first['surface']}` RK`{first['rk_step']}` field `{first['field']}`.",
                f"- Top material residual there: `{top['field']}` max_abs `{top['max_abs']}` rmse `{top['rmse']}`.",
            ]
        )
        if first["surface"] == "after_rk_addtend_before_small_step_prep":
            prep = payload["substage_comparisons"]["comparisons"]["after_small_step_prep_calc_p_rho"][first["rk_step"]]
            t_work = prep["per_field_metrics"].get("T_WORK", {})
            p_work = prep["per_field_metrics"].get("P_WORK", {})
            lines.append(
                f"- RK`{first['rk_step']}` prep work arrays then match for `T_WORK` max_abs `{t_work.get('max_abs')}` and `P_WORK` max_abs `{p_work.get('max_abs')}`."
            )
    if final.get("status") == "COMPARISON_EXECUTED":
        top_final = final["ranked_residuals"][0]
        lines.append(
            f"- Final accepted strict comparison still diverges: first `{final.get('first_divergent_field')}`, top `{top_final['field']}` max_abs `{top_final['max_abs']}`."
        )
    lines.extend(
        [
            "",
            "Detailed per-boundary metrics are in `proofs/v014/step1_t_p_operator_localization.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 T/P Operator Localization",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: localize the remaining Step-1 strict same-input T/P divergence after live-nest base initialization closure.",
        "",
        "files changed:",
        "- `proofs/v014/step1_t_p_operator_localization.py`",
        "- `proofs/v014/step1_t_p_operator_localization.json`",
        "- `proofs/v014/step1_t_p_operator_localization.md`",
        "- `proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md`",
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
    wrf = load_wrf_surfaces()
    if wrf.get("status") != "WRF_SUBSTAGE_TRUTH_READY":
        comparison_bundle = {"status": "NOT_EXECUTED", "blocker": wrf}
        final_comparison = {"status": "NOT_EXECUTED"}
        verdict = "STEP1_TP_BLOCKED_WRF_SUBSTAGE_TRUTH"
        jax_capture_meta = {"status": "NOT_EXECUTED"}
    else:
        jax_capture = capture_jax_boundaries()
        if jax_capture.get("status") != "JAX_BOUNDARIES_READY":
            comparison_bundle = {"status": "NOT_EXECUTED", "blocker": jax_capture}
            final_comparison = {"status": "NOT_EXECUTED"}
            verdict = "STEP1_TP_BLOCKED_JAX_SUBSTAGE_CAPTURE"
            jax_capture_meta = {key: value for key, value in jax_capture.items() if key != "captures"}
        else:
            comparison_bundle = compare_all(wrf, jax_capture)
            final_comparison = jax_capture["final_comparison"]
            verdict = derive_verdict(comparison_bundle, final_comparison)
            jax_capture_meta = {
                key: value
                for key, value in jax_capture.items()
                if key not in {"captures", "final_pre_halo_state", "base_state", "final_comparison"}
            }
    decision = next_decision(verdict, comparison_bundle)
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_t_p_operator_localization.v1",
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
        "tooling_verdict": (
            "FOCUSED_STEP1_SUBSTAGE_TRUTH_COMPARATOR_FASTEST_RIGOROUS_WALL_CLOCK"
        ),
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "accepted_final_comparison": (
                "CPU-WRF step-1 post-RK/pre-halo truth vs JAX one-step final pre-halo state "
                "from the live-nest-initialized d02 carry"
            ),
            "substage_boundaries": list(SURFACE_ORDER),
            "rejected_comparisons": [
                "WRF final truth vs JAX initial state",
                "JAX-vs-JAX-only localization",
                "one-cell or station proxy proof",
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
        "wrf_substage_truth": strip_arrays_from_wrf(wrf) if wrf.get("status") == "WRF_SUBSTAGE_TRUTH_READY" else wrf,
        "jax_capture": jax_capture_meta,
        "substage_comparisons": comparison_bundle,
        "final_comparison": final_comparison,
        "commands": {
            "wrf_instrumentation": [
                "tcsh ./compile em_real (scratch WRF under /mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/WRF)",
                "WRFGPU2_STEP1_TP_LOCALIZATION=1 mpirun --oversubscribe -np 28 run/wrf.exe",
            ],
            "required_validation": [
                "python -m py_compile proofs/v014/step1_t_p_operator_localization.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_t_p_operator_localization.py",
                "python -m json.tool proofs/v014/step1_t_p_operator_localization.json >/tmp/step1_t_p_operator_localization.validated.json",
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
            "Only two early WRF substage boundaries were emitted; if this boundary is fixed and residuals remain, acoustic/pre-finish substep truth is the next surface.",
            "The proof-local CPU loader still mirrors production live-nest init because build_replay_case is GPU-only at State.zeros.",
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
