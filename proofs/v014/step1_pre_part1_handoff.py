#!/usr/bin/env python3
"""V0.14 d02 Step-1 solve_em pre-part1 handoff localization."""

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
import step1_part1_physics_state_mutation as part1  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_pre_part1_handoff.json"
OUT_MD = PROOF_DIR / "step1_pre_part1_handoff.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_pre_part1_handoff_wrf_patch.diff"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-pre-part1-handoff/sprint-contract.md"
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff")
WRF_TRUTH = SCRATCH / "wrf_truth"
WRF_TREE = SCRATCH / "WRF"
WRF_RUN_DIR = SCRATCH / "run"
WRF_BUILD_LOG = SCRATCH / "compile_step1_pre_part1_handoff.log"
WRF_RUN_LOG = SCRATCH / "wrf_step1_pre_part1_handoff_stdout.log"
ACCEPTED_FINAL_TRUTH = live.ACCEPTED_TRUTH

TARGET_STEP = 1
TARGET_DOMAIN = 2
THETA_OFFSET_K = 300.0

SURFACE_ORDER = (
    "after_step_increment",
    "rk_loop_entry_before_rk_step_prep",
    "after_rk_step_prep",
    "after_halo_em_a",
    "after_rk_phys_bc_dry_1",
    "before_first_rk_step_part1_call",
)
PRECALL_SURFACE = "before_first_rk_step_part1_call"
FIRST_SURFACE = "after_step_increment"

FIELDS = (
    "T_STATE",
    "P_STATE",
    "PB",
    "MU_STATE",
    "MUB",
    "MUT",
    "W_STATE",
    "PH_STATE",
    "PHB",
)
JAX_SURFACE_ORDER = (
    "raw_input_state_before_physics",
    "raw_carry_state_before_physics",
    "step_entry_state_zero_dry",
    "physics_carry_state_dry",
    "physics_state_dry",
)
MATERIAL_THRESHOLDS = {
    "T_STATE": 1.0e-3,
    "P_STATE": 1.0,
    "PB": 1.0,
    "MU_STATE": 1.0e-2,
    "MUB": 1.0e-2,
    "MUT": 1.0e-2,
    "W_STATE": 1.0e-2,
    "PH_STATE": 1.0e-2,
    "PHB": 1.0e-2,
}

RECORD_SPECS = {
    "MASS_PREPART": {
        "length": 13,
        "fields": ("T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "MUT"),
    },
    "WPH_PREPART": {
        "length": 10,
        "fields": ("W_STATE", "PH_STATE", "PHB"),
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
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
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
    if field in {"MU_STATE", "MUB", "MUT"}:
        return shapes["mass2d"]
    if field in {"W_STATE", "PH_STATE", "PHB"}:
        return shapes["wph"]
    return shapes["mass"]


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


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
    pattern = f"{surface}_d{TARGET_DOMAIN}_step_{TARGET_STEP}_rk_*_*.txt"
    raw_files = sorted(WRF_TRUTH.glob(pattern))
    if not raw_files:
        return {"status": "BLOCKED_NO_WRF_SURFACE_FILES", "surface": surface, "pattern": pattern}
    arrays = {field: np.full(field_shape(field, shapes), np.nan, dtype=np.float64) for field in FIELDS}
    duplicate_stats = {
        field: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for field in FIELDS
    }
    record_counts = {record: 0 for record in RECORD_SPECS}
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
                if tag in RECORD_SPECS:
                    spec = RECORD_SPECS[tag]
                    if len(parts) != int(spec["length"]):
                        return {
                            "status": "BLOCKED_PARSE_ERROR",
                            "surface": surface,
                            "path": str(path),
                            "line": stripped[:240],
                            "expected_length": int(spec["length"]),
                            "actual_length": len(parts),
                        }
                    x = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    values = [float(item) for item in parts[7:]]
                    for field, value in zip(spec["fields"], values):
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
    return {
        "status": "WRF_SURFACE_READY",
        "surface": surface,
        "rk_values": sorted(
            {
                int((header.get("rk_step") or ["-1"])[0])
                for header in headers
                if header.get("rk_step")
            }
        ),
        "raw_file_count": len(raw_files),
        "record_counts": record_counts,
        "duplicate_stats": duplicate_stats,
        "headers": headers[:4],
        "summaries": {name: array_summary(arr) for name, arr in arrays.items()},
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
            return {
                "status": parsed.get("status"),
                "blocker": parsed,
                "shapes": {key: list(value) for key, value in shapes.items()},
            }
        surfaces[surface] = parsed
    return {
        "status": "WRF_PRE_PART1_TRUTH_READY",
        "shapes": {key: list(value) for key, value in shapes.items()},
        "surfaces": surfaces,
    }


def diff_metrics(field: str, candidate: Any, reference: Any) -> dict[str, Any]:
    return live.diff_metrics(field, candidate, reference)


def material(field: str, metric: Mapping[str, Any]) -> bool:
    if metric.get("status") != "OK":
        return True
    max_abs = metric.get("max_abs")
    if max_abs is None:
        return bool(metric.get("nonfinite_diff_count"))
    threshold = MATERIAL_THRESHOLDS.get(field)
    return threshold is not None and float(max_abs) > float(threshold)


def rank_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
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


def compare_wrf_pair(
    name: str,
    candidate_surface: Mapping[str, Any],
    reference_surface: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = {
        field: diff_metrics(field, candidate_surface["arrays"][field], reference_surface["arrays"][field])
        for field in FIELDS
    }
    first_material_field = None
    for field in FIELDS:
        if material(field, metrics[field]):
            first_material_field = field
            break
    return {
        "status": "WRF_PAIR_COMPARISON_EXECUTED",
        "name": name,
        "diff_sign": "candidate_minus_reference",
        "candidate_surface": candidate_surface["surface"],
        "reference_surface": reference_surface["surface"],
        "first_material_field": first_material_field,
        "per_field_metrics": metrics,
        "ranked_residuals": rank_metrics(metrics),
    }


def compare_wrf_internal(wrf: Mapping[str, Any]) -> dict[str, Any]:
    surfaces = wrf["surfaces"]
    adjacent: dict[str, Any] = {}
    from_first: dict[str, Any] = {}
    first = surfaces[FIRST_SURFACE]
    for previous, current in zip(SURFACE_ORDER[:-1], SURFACE_ORDER[1:]):
        adjacent[f"{previous}__to__{current}"] = compare_wrf_pair(
            f"{previous}__to__{current}", surfaces[current], surfaces[previous]
        )
    for current in SURFACE_ORDER[1:]:
        from_first[current] = compare_wrf_pair(f"{FIRST_SURFACE}__to__{current}", surfaces[current], first)
    return {
        "status": "WRF_INTERNAL_COMPARISONS_EXECUTED",
        "adjacent": adjacent,
        "from_first": from_first,
    }


def state_arrays(jnp: Any, state: Any) -> dict[str, Any]:
    p_base = state.p_total - state.p_perturbation
    ph_base = state.ph_total - state.ph_perturbation
    mu_base = state.mu_total - state.mu_perturbation
    return {
        "T_STATE": state.theta - THETA_OFFSET_K,
        "THETA_FULL": state.theta,
        "P_STATE": state.p_perturbation,
        "PB": p_base,
        "MU_STATE": state.mu_perturbation,
        "MUB": mu_base,
        "MUT": state.mu_total,
        "W_STATE": state.w,
        "PH_STATE": state.ph_perturbation,
        "PHB": ph_base,
    }


def capture_jax_pre_part1_surfaces() -> dict[str, Any]:
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
    spec = om.halo_spec(namelist.grid)
    raw_input_state = inputs["state"]
    raw_carry_state = inputs["carry"].state
    step_entry = om.apply_halo(raw_carry_state, spec)
    physics = om._physics_step_forcing(
        inputs["carry"],
        namelist,
        lead_seconds,
        run_radiation=run_radiation,
    )
    physics_carry_state = om.apply_halo(physics.carry.state, spec)
    physics_state = om.apply_halo(physics.state, spec)
    jax.block_until_ready(physics_state.theta)
    return {
        "status": "JAX_PRE_PART1_SURFACES_READY",
        "captures": {
            "raw_input_state_before_physics": state_arrays(jnp, raw_input_state),
            "raw_carry_state_before_physics": state_arrays(jnp, raw_carry_state),
            "step_entry_state_zero_dry": state_arrays(jnp, step_entry),
            "physics_carry_state_dry": state_arrays(jnp, physics_carry_state),
            "physics_state_dry": state_arrays(jnp, physics_state),
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
        "loader": {
            "raw_child_construction": inputs["raw_child"].get("construction"),
            "live_child_construction": inputs["live_child"].get("construction"),
            "boundary_package": inputs["boundary_package"],
            "initial_deltas": inputs["initial_deltas"],
        },
    }


def compare_surface_to_jax(
    surface: str,
    wrf_surface: Mapping[str, Any],
    jax_arrays: Mapping[str, Any],
    jax: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field in FIELDS:
        if field not in jax_arrays:
            metrics[field] = {"status": "MISSING_JAX_FIELD"}
            continue
        candidate = np.asarray(jax.device_get(jax_arrays[field]), dtype=np.float64)
        metrics[field] = diff_metrics(field, candidate, wrf_surface["arrays"][field])
    first_material_field = None
    for field in FIELDS:
        if metrics[field].get("status") == "OK" and material(field, metrics[field]):
            first_material_field = field
            break
    return {
        "status": "WRF_JAX_COMPARISON_EXECUTED",
        "surface": surface,
        "diff_sign": "jax_minus_wrf",
        "first_material_field": first_material_field,
        "per_field_metrics": metrics,
        "ranked_residuals": rank_metrics(metrics),
    }


def compare_wrf_to_jax(wrf: Mapping[str, Any], jax_capture: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    captures = jax_capture["captures"]
    surfaces = wrf["surfaces"]
    matrix: dict[str, Any] = {}
    for wrf_surface_name in (FIRST_SURFACE, PRECALL_SURFACE):
        matrix[wrf_surface_name] = {}
        for jax_name in JAX_SURFACE_ORDER:
            matrix[wrf_surface_name][f"vs_{jax_name}"] = compare_surface_to_jax(
                wrf_surface_name,
                surfaces[wrf_surface_name],
                captures[jax_name],
                jax,
            )
    return {"status": "WRF_JAX_COMPARISONS_EXECUTED", "matrix": matrix}


def theta_semantics(wrf: Mapping[str, Any], jax_capture: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    wrf_t = wrf["surfaces"][PRECALL_SURFACE]["arrays"]["T_STATE"]
    out: dict[str, Any] = {
        "wrf_field": "grid%t_2 / T_STATE",
        "wrf_interpretation_under_test": "perturbation potential temperature",
        "jax_full_theta_field": "State.theta",
        "jax_perturbation_theta_field": "State.theta - 300 K",
        "theta_offset_K": THETA_OFFSET_K,
        "per_jax_surface": {},
    }
    for name in JAX_SURFACE_ORDER:
        arrays = jax_capture["captures"][name]
        jax_full = np.asarray(jax.device_get(arrays["THETA_FULL"]), dtype=np.float64)
        jax_pert = np.asarray(jax.device_get(arrays["T_STATE"]), dtype=np.float64)
        pert_metric = diff_metrics("T_STATE", jax_pert, wrf_t)
        full_metric = diff_metrics("T_STATE", jax_full, wrf_t + THETA_OFFSET_K)
        wrong_metric = diff_metrics("T_STATE", jax_full, wrf_t)
        same_residual = (
            pert_metric.get("status") == "OK"
            and full_metric.get("status") == "OK"
            and abs(float(pert_metric.get("max_abs") or 0.0) - float(full_metric.get("max_abs") or 0.0)) < 1.0e-9
            and abs(float(pert_metric.get("rmse") or 0.0) - float(full_metric.get("rmse") or 0.0)) < 1.0e-9
        )
        wrong_full_offset = (
            wrong_metric.get("status") == "OK"
            and wrong_metric.get("bias") is not None
            and 250.0 < abs(float(wrong_metric["bias"])) < 350.0
        )
        out["per_jax_surface"][name] = {
            "wrf_t_state_vs_jax_theta_minus_300": pert_metric,
            "wrf_t_state_plus_300_vs_jax_full_theta": full_metric,
            "wrong_wrf_t_state_vs_jax_full_theta": wrong_metric,
            "perturbation_and_full_plus_300_residuals_identical": same_residual,
            "wrong_full_theta_mapping_has_approximately_300K_bias": wrong_full_offset,
            "mapping_conclusion": (
                "WRF_T_STATE_IS_PERTURBATION_THETA"
                if same_residual and wrong_full_offset
                else "MAPPING_NOT_PROVEN"
            ),
        }
    step = out["per_jax_surface"]["step_entry_state_zero_dry"]
    out["overall_conclusion"] = step["mapping_conclusion"]
    return out


def compare_part1_entry_continuity(pre_wrf: Mapping[str, Any]) -> dict[str, Any]:
    try:
        shapes = expected_shapes()
        part1_entry = part1.parse_wrf_surface(
            "part1_entry_before_init_zero_tendency",
            {"mass": shapes["mass"], "mass2d": shapes["mass2d"]},
        )
    except Exception as exc:
        return {"status": "BLOCKED_PART1_ENTRY_PARSE_EXCEPTION", "exception": repr(exc)}
    if part1_entry.get("status") != "WRF_SURFACE_READY":
        return {"status": "BLOCKED_PART1_ENTRY_PARSE", "blocker": part1_entry}
    pre_call = pre_wrf["surfaces"][PRECALL_SURFACE]
    fields = ("T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "MUT")
    metrics = {
        field: diff_metrics(field, part1_entry["arrays"][field], pre_call["arrays"][field])
        for field in fields
    }
    return {
        "status": "PART1_ENTRY_CONTINUITY_EXECUTED",
        "diff_sign": "part1_entry_minus_solve_em_precall",
        "surface_a": "part1_entry_before_init_zero_tendency",
        "surface_b": PRECALL_SURFACE,
        "per_field_metrics": metrics,
        "ranked_residuals": rank_metrics(metrics),
    }


def t_state_delta_from_first(internal: Mapping[str, Any]) -> dict[str, Any]:
    comp = internal["from_first"][PRECALL_SURFACE]
    return comp["per_field_metrics"]["T_STATE"]


def first_material_jax_t(comparisons: Mapping[str, Any]) -> dict[str, Any] | None:
    precall = comparisons["wrf_jax"]["matrix"][PRECALL_SURFACE]
    for name in JAX_SURFACE_ORDER:
        comp = precall[f"vs_{name}"]
        metric = comp["per_field_metrics"]["T_STATE"]
        if material("T_STATE", metric):
            return {"jax_surface": name, "metrics": metric}
    return None


def classify(comparisons: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[str], str]:
    internal = comparisons.get("wrf_internal", {})
    wrf_jax = comparisons.get("wrf_jax", {})
    continuity = comparisons.get("part1_entry_continuity", {})
    semantics = comparisons.get("theta_semantics", {})
    if internal.get("status") != "WRF_INTERNAL_COMPARISONS_EXECUTED":
        return (
            "STEP1_PRE_PART1_BLOCKED_WRF_INTERNAL_COMPARISON",
            {},
            ["WRF pre-part1 internal comparisons did not execute."],
            "Fix the WRF internal comparison blocker and rerun.",
        )
    if continuity.get("status") == "PART1_ENTRY_CONTINUITY_EXECUTED":
        cont_t = continuity["per_field_metrics"]["T_STATE"]
        if material("T_STATE", cont_t):
            return (
            "STEP1_PRE_PART1_LOCALIZED_WRONG_PRIOR_SURFACE_T_STATE",
            {"part1_entry_vs_solve_em_precall_t_state": cont_t},
            [],
            "Rebuild the WRF part1-entry hook against this solve_em pre-call boundary.",
        )
    if wrf_jax.get("status") != "WRF_JAX_COMPARISONS_EXECUTED":
        return (
            "STEP1_PRE_PART1_BLOCKED_JAX_COMPARISON",
            {},
            ["WRF/JAX pre-part1 comparisons did not execute."],
            "Fix the JAX comparison blocker and rerun.",
        )

    wrf_t_delta = t_state_delta_from_first(internal)
    step_entry_t = wrf_jax["matrix"][PRECALL_SURFACE]["vs_step_entry_state_zero_dry"]["per_field_metrics"]["T_STATE"]
    raw_t = wrf_jax["matrix"][PRECALL_SURFACE]["vs_raw_input_state_before_physics"]["per_field_metrics"]["T_STATE"]
    first_jax_t = first_material_jax_t(comparisons)
    mapping = semantics.get("overall_conclusion")

    evidence = {
        "field": "T_STATE",
        "wrf_first_surface": FIRST_SURFACE,
        "wrf_precall_surface": PRECALL_SURFACE,
        "wrf_t_state_delta_after_step_increment_to_precall": wrf_t_delta,
        "wrf_precall_vs_jax_raw_input_state_before_physics": raw_t,
        "wrf_precall_vs_jax_step_entry_state_zero_dry": step_entry_t,
        "first_material_jax_t_state_surface": first_jax_t,
        "theta_semantics_conclusion": mapping,
        "part1_entry_continuity_t_state": continuity.get("per_field_metrics", {}).get("T_STATE"),
    }

    if material("T_STATE", wrf_t_delta):
        return (
            "STEP1_PRE_PART1_LOCALIZED_WRF_PRECALL_MUTATION_T_STATE",
            evidence,
            [],
            "Split the named WRF adjacent solve_em interval for the exact mutation point.",
        )
    if material("T_STATE", step_entry_t):
        if mapping == "WRF_T_STATE_IS_PERTURBATION_THETA":
            return (
                "STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE",
                evidence,
                [],
                "Localize the JAX live-nest Step-1 loader/carry construction for `T_STATE` before `_physics_step_forcing`.",
            )
        return (
            "STEP1_PRE_PART1_LOCALIZED_FIELD_MAPPING_T_STATE",
            evidence,
            [],
            "Resolve the T/theta field mapping contract before changing loader or dynamics code.",
        )
    return (
        "STEP1_PRE_PART1_NO_REMAINING_DIVERGENCE",
        evidence,
        [],
        "Close this handoff gate and return to the next largest field only if required by v0.14.",
    )


def strip_arrays_from_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    if wrf.get("status") != "WRF_PRE_PART1_TRUTH_READY":
        return dict(wrf)
    surfaces = {}
    for surface, parsed in wrf["surfaces"].items():
        surfaces[surface] = {key: value for key, value in parsed.items() if key != "arrays"}
    return {"status": wrf["status"], "shapes": wrf["shapes"], "surfaces": surfaces}


def strip_arrays_from_jax(jax_capture: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in jax_capture.items() if key != "captures"}


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    evidence = payload.get("classification_evidence", {})
    wrf_delta = evidence.get("wrf_t_state_delta_after_step_increment_to_precall", {})
    step_entry = evidence.get("wrf_precall_vs_jax_step_entry_state_zero_dry", {})
    raw = evidence.get("wrf_precall_vs_jax_raw_input_state_before_physics", {})
    continuity = evidence.get("part1_entry_continuity_t_state", {})
    lines = [
        "# V0.14 Step-1 Pre-Part1 Handoff",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`.",
        f"- WRF solve_em truth root: `{WRF_TRUTH}`.",
        f"- WRF patch artifact: `{OUT_WRF_PATCH}`.",
        f"- Full-vs-perturbation theta conclusion: `{evidence.get('theta_semantics_conclusion')}`.",
        f"- WRF `T_STATE` delta from `{FIRST_SURFACE}` to `{PRECALL_SURFACE}`: max_abs `{wrf_delta.get('max_abs')}`, rmse `{wrf_delta.get('rmse')}`.",
        f"- WRF `{PRECALL_SURFACE}` `T_STATE` vs raw JAX live-nest input state (`State.theta - 300 K`): max_abs `{raw.get('max_abs')}`, rmse `{raw.get('rmse')}`.",
        f"- WRF `{PRECALL_SURFACE}` `T_STATE` vs JAX step-entry haloed state (`State.theta - 300 K`): max_abs `{step_entry.get('max_abs')}`, rmse `{step_entry.get('rmse')}`.",
        f"- WRF solve_em pre-call vs prior part1-entry `T_STATE` continuity: max_abs `{continuity.get('max_abs')}`.",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE":
        lines.extend(
            [
                "- WRF does not change `grid%t_2` between the solve_em step boundary and the `CALL first_rk_step_part1` call-site.",
                "- The previous part1-entry surface is continuous with the new solve_em pre-call surface for `T_STATE`.",
                "- The 300 K offset check rejects the full-theta mismatch explanation: WRF `grid%t_2` is perturbation theta and should be compared to JAX `State.theta - 300 K`.",
                "- The residual is already present in the raw JAX live-nest Step-1 state/carry before `_physics_step_forcing`; localize the loader/carry construction next.",
            ]
        )
    else:
        lines.append("- See the JSON classification evidence for the exact first surface and field.")
    lines.extend(["", "Detailed comparison tables are in `proofs/v014/step1_pre_part1_handoff.json`.", ""])
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Pre-Part1 Handoff",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: move one boundary upstream from WRF `first_rk_step_part1` and classify why `T_STATE` already diverges at part1 entry.",
        "",
        "files changed:",
        "- `proofs/v014/step1_pre_part1_handoff.py`",
        "- `proofs/v014/step1_pre_part1_handoff.json`",
        "- `proofs/v014/step1_pre_part1_handoff.md`",
        "- `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-step1-pre-part1-handoff.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["wrf_instrumentation"]:
        lines.append(f"- `{command}`")
    for command in payload["commands"]["required_validation"]:
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
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    ancestor = run_command(["git", "merge-base", "--is-ancestor", "588686d6", "HEAD"], cwd=ROOT)
    wrf = parse_wrf_surfaces()
    comparisons: dict[str, Any]
    jax_capture_meta: dict[str, Any]
    if wrf.get("status") != "WRF_PRE_PART1_TRUTH_READY":
        comparisons = {"status": "NOT_EXECUTED", "blocker": wrf}
        jax_capture_meta = {"status": "NOT_EXECUTED"}
        verdict = "STEP1_PRE_PART1_BLOCKED_WRF_TRUTH"
        classification_evidence: dict[str, Any] = {}
        risks = ["WRF solve_em pre-part1 truth parsing did not complete."]
        decision = "Fix the WRF truth blocker and rerun."
    else:
        internal = compare_wrf_internal(wrf)
        part1_continuity = compare_part1_entry_continuity(wrf)
        jax_capture = capture_jax_pre_part1_surfaces()
        if jax_capture.get("status") != "JAX_PRE_PART1_SURFACES_READY":
            comparisons = {
                "wrf_internal": internal,
                "part1_entry_continuity": part1_continuity,
                "wrf_jax": {"status": "NOT_EXECUTED", "blocker": jax_capture},
            }
            jax_capture_meta = strip_arrays_from_jax(jax_capture)
            verdict = "STEP1_PRE_PART1_BLOCKED_JAX_CAPTURE"
            classification_evidence = {}
            risks = ["JAX live-nest pre-part1 capture did not complete."]
            decision = "Fix the JAX capture blocker and rerun."
        else:
            wrf_jax = compare_wrf_to_jax(wrf, jax_capture)
            semantics = theta_semantics(wrf, jax_capture)
            comparisons = {
                "wrf_internal": internal,
                "part1_entry_continuity": part1_continuity,
                "wrf_jax": wrf_jax,
                "theta_semantics": semantics,
            }
            verdict, classification_evidence, risks, decision = classify(comparisons)
            jax_capture_meta = strip_arrays_from_jax(jax_capture)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_pre_part1_handoff.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "required_ancestor_588686d6": {
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
        "tooling_verdict": "RIGHT_TOOL_FASTEST_WALL_CLOCK_SOLVE_EM_SAVEPOINT_COMPARATOR",
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "accepted_final_truth_npz": path_info(ACCEPTED_FINAL_TRUTH),
            "wrf_truth_root": path_info(WRF_TRUTH),
            "wrf_tree": path_info(WRF_TREE),
            "wrf_run_dir": path_info(WRF_RUN_DIR),
            "wrf_build_log": path_info(WRF_BUILD_LOG),
            "wrf_run_log": path_info(WRF_RUN_LOG),
            "wrf_patch_diff": path_info(OUT_WRF_PATCH),
            "prior_part1_truth_root": path_info(part1.WRF_TRUTH),
        },
        "verdict": verdict,
        "classification_evidence": classification_evidence,
        "wrf_pre_part1_truth": strip_arrays_from_wrf(wrf),
        "jax_capture": jax_capture_meta,
        "comparisons": comparisons,
        "commands": {
            "wrf_instrumentation": [
                "cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/WRF /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF",
                "cp -a --reflink=auto /mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/run /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/run",
                "tcsh ./compile em_real (scratch WRF, conda wrf-build toolchain)",
                "WRFGPU2_STEP1_PRE_PART1_HANDOFF=1 WRFGPU2_STEP1_PRE_PART1_HANDOFF_ROOT=/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth mpirun --oversubscribe -np 28 ./wrf.exe",
            ],
            "required_validation": [
                "python -m py_compile proofs/v014/step1_pre_part1_handoff.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_pre_part1_handoff.py",
                "python -m json.tool proofs/v014/step1_pre_part1_handoff.json >/tmp/step1_pre_part1_handoff.validated.json",
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
        "unresolved_risks": risks
        or [
            "This proof localizes the divergence to the JAX live-nest Step-1 loader/carry boundary, but does not yet split the loader internals.",
            "No production source fix was made or gated.",
        ],
        "next_decision": decision,
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
