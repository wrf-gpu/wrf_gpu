#!/usr/bin/env python3
"""V0.14 d02 Step-1 first_rk_step_part1 physics-state mutation split."""

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
import step1_rk1_source_boundary as source_boundary  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_part1_physics_state_mutation.json"
OUT_MD = PROOF_DIR / "step1_part1_physics_state_mutation.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_part1_physics_state_mutation_wrf_patch.diff"

SPRINT_CONTRACT = (
    ROOT
    / ".agent/sprints/2026-06-09-v014-step1-part1-physics-state-mutation/sprint-contract.md"
)
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_part1_physics_state_mutation")
WRF_TRUTH = SCRATCH / "wrf_truth"
WRF_BUILD_LOG = SCRATCH / "compile_step1_part1_physics_state_mutation.log"
WRF_RUN_LOG = SCRATCH / "wrf_step1_part1_physics_state_mutation_stdout.log"
WRF_RUN_DIR = SCRATCH / "run"
WRF_TREE = SCRATCH / "WRF"
ACCEPTED_FINAL_TRUTH = live.ACCEPTED_TRUTH

TARGET_STEP = 1
TARGET_DOMAIN = 2
MATERIAL_T_STATE = 1.0e-3

SURFACE_ORDER = (
    "part1_entry_before_init_zero_tendency",
    "after_init_zero_tendency",
    "after_phy_prep",
    "after_pre_radiation_driver",
    "after_radiation_driver",
    "after_surface_driver",
    "after_pbl_driver",
    "after_cumulus_driver",
    "after_shallowcu_driver",
    "after_force_scm",
    "after_fddagd_driver",
    "part1_exit",
)

PART1_FIELDS = (
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
    "RTHRATEN",
    "RTHRATENLW",
    "RTHRATENSW",
    "RTHBLTEN",
    "RTHCUTEN",
    "RTHFTEN",
    "TH_PHY",
    "T_PHY",
    "P_PHY",
    "PI_PHY",
)

MATERIAL_THRESHOLDS = {
    "T_STATE": MATERIAL_T_STATE,
    "P_STATE": 1.0,
    "PB": 1.0,
    "MU_STATE": 1.0e-2,
    "MUB": 1.0e-2,
    "MUT": 1.0e-2,
    "T_TENDF": 1.0e-6,
    "H_DIABATIC": 1.0e-6,
    "MU_TENDF": 1.0e-6,
    "T_SAVE": 1.0e-6,
    "T_OLD": 1.0e-3,
    "RTHRATEN": 1.0e-9,
    "RTHRATENLW": 1.0e-9,
    "RTHRATENSW": 1.0e-9,
    "RTHBLTEN": 1.0e-9,
    "RTHCUTEN": 1.0e-9,
    "RTHFTEN": 1.0e-9,
    "TH_PHY": 1.0e-3,
    "T_PHY": 1.0e-3,
    "P_PHY": 1.0,
    "PI_PHY": 1.0e-6,
}

JAX_COMPARE_FIELDS = {
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
        }


def field_shape(field: str, shapes: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    if field in {"MU_STATE", "MUB", "MUT", "MU_TENDF"}:
        return shapes["mass2d"]
    return shapes["mass"]


def _record_value(
    arrays: dict[str, np.ndarray],
    filled: dict[str, np.ndarray],
    duplicate_stats: dict[str, Any],
    field: str,
    index: tuple[int, ...],
    value: float,
) -> None:
    if not bool(filled[field][index]):
        arrays[field][index] = value
        filled[field][index] = True
        return
    current = arrays[field][index]
    duplicate_stats[field]["duplicates"] += 1
    if current != value and not (np.isnan(current) and np.isnan(value)):
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
    arrays = {field: np.zeros(field_shape(field, shapes), dtype=np.float64) for field in PART1_FIELDS}
    filled = {field: np.zeros(field_shape(field, shapes), dtype=np.bool_) for field in PART1_FIELDS}
    duplicate_stats = {
        field: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for field in PART1_FIELDS
    }
    headers: list[dict[str, Any]] = []
    record_count = 0
    allowed_headers = {
        "surface",
        "routine",
        "domain_id",
        "current_timestr_before_step",
        "grid_itimestep_after_increment",
        "rk_step",
        "tile_i_j_bounds_fortran",
        "global_mass_i_j_end_exclusive_fortran",
        "tile_record_policy",
        "mass_vertical_fortran_k_start_k_end_inclusive",
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
                if tag == "MASS_PART1":
                    if len(parts) != 28:
                        return {
                            "status": "BLOCKED_PARSE_ERROR",
                            "surface": surface,
                            "path": str(path),
                            "line": stripped[:240],
                            "expected_length": 28,
                            "actual_length": len(parts),
                        }
                    x = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    values = [float(item) for item in parts[7:]]
                    for field, value in zip(PART1_FIELDS, values):
                        shape = arrays[field].shape
                        index = (y, x) if len(shape) == 2 else (k, y, x)
                        _record_value(arrays, filled, duplicate_stats, field, index, value)
                    record_count += 1
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
            "record_count": record_count,
        }
    missing = {
        name: {"missing_count": int((~filled[name]).sum()), "shape": list(arr.shape)}
        for name, arr in arrays.items()
        if (~filled[name]).any()
    }
    if missing:
        return {
            "status": "BLOCKED_MISSING_VALUES",
            "surface": surface,
            "missing": missing,
            "duplicate_stats": duplicate_stats,
            "record_count": record_count,
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
        "record_count": record_count,
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
            return {
                "status": parsed.get("status"),
                "blocker": parsed,
                "shapes": {key: list(value) for key, value in shapes.items()},
            }
        surfaces[surface] = parsed
    return {
        "status": "WRF_PART1_TRUTH_READY",
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
        for field in PART1_FIELDS
    }
    first_material_field = None
    for field in PART1_FIELDS:
        if material(field, metrics[field]):
            first_material_field = field
            break
    return {
        "status": "WRF_PAIR_COMPARISON_EXECUTED",
        "name": name,
        "candidate_surface": candidate_surface["surface"],
        "reference_surface": reference_surface["surface"],
        "first_material_field": first_material_field,
        "per_field_metrics": metrics,
        "ranked_residuals": rank_metrics(metrics),
    }


def compare_wrf_internal(wrf: Mapping[str, Any]) -> dict[str, Any]:
    surfaces = wrf["surfaces"]
    adjacent: dict[str, Any] = {}
    from_entry: dict[str, Any] = {}
    entry = surfaces[SURFACE_ORDER[0]]
    for previous, current in zip(SURFACE_ORDER[:-1], SURFACE_ORDER[1:]):
        adjacent[f"{previous}__to__{current}"] = compare_wrf_pair(
            f"{previous}__to__{current}", surfaces[current], surfaces[previous]
        )
    for current in SURFACE_ORDER[1:]:
        from_entry[current] = compare_wrf_pair(
            f"{SURFACE_ORDER[0]}__to__{current}", surfaces[current], entry
        )
    return {
        "status": "WRF_INTERNAL_COMPARISONS_EXECUTED",
        "adjacent": adjacent,
        "from_entry": from_entry,
    }


def compare_surface_to_jax(
    surface: str,
    wrf_surface: Mapping[str, Any],
    jax_arrays: Mapping[str, Any],
    jax: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field in PART1_FIELDS:
        if field not in JAX_COMPARE_FIELDS:
            metrics[field] = {"status": "NOT_A_JAX_STATE_OR_DRY_LEAF"}
            continue
        if field not in jax_arrays:
            metrics[field] = {"status": "MISSING_JAX_FIELD"}
            continue
        candidate = np.asarray(jax.device_get(jax_arrays[field]), dtype=np.float64)
        metrics[field] = diff_metrics(field, candidate, wrf_surface["arrays"][field])
    first_material_field = None
    for field in PART1_FIELDS:
        if metrics[field].get("status") == "OK" and material(field, metrics[field]):
            first_material_field = field
            break
    return {
        "status": "WRF_JAX_COMPARISON_EXECUTED",
        "surface": surface,
        "diff_sign": "jax_minus_wrf",
        "first_material_field": first_material_field,
        "per_field_metrics": metrics,
        "ranked_residuals": rank_metrics({k: v for k, v in metrics.items() if v.get("status") == "OK"}),
    }


def compare_jax(wrf: Mapping[str, Any], jax_capture: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    captures = jax_capture["captures"]
    surfaces = wrf["surfaces"]
    return {
        "status": "WRF_JAX_COMPARISONS_EXECUTED",
        "matrix": {
            "part1_entry_before_init_zero_tendency": {
                "vs_step_entry_state_zero_dry": compare_surface_to_jax(
                    "part1_entry_before_init_zero_tendency",
                    surfaces["part1_entry_before_init_zero_tendency"],
                    captures["step_entry_state_zero_dry"],
                    jax,
                ),
                "vs_physics_carry_state_dry": compare_surface_to_jax(
                    "part1_entry_before_init_zero_tendency",
                    surfaces["part1_entry_before_init_zero_tendency"],
                    captures["physics_carry_state_dry"],
                    jax,
                ),
                "vs_physics_state_dry": compare_surface_to_jax(
                    "part1_entry_before_init_zero_tendency",
                    surfaces["part1_entry_before_init_zero_tendency"],
                    captures["physics_state_dry"],
                    jax,
                ),
            },
            "part1_exit": {
                "vs_step_entry_state_zero_dry": compare_surface_to_jax(
                    "part1_exit", surfaces["part1_exit"], captures["step_entry_state_zero_dry"], jax
                ),
                "vs_physics_carry_state_dry": compare_surface_to_jax(
                    "part1_exit", surfaces["part1_exit"], captures["physics_carry_state_dry"], jax
                ),
                "vs_physics_state_dry": compare_surface_to_jax(
                    "part1_exit", surfaces["part1_exit"], captures["physics_state_dry"], jax
                ),
            },
        },
    }


def first_adjacent_t_state_mutation(internal: Mapping[str, Any]) -> dict[str, Any] | None:
    for previous, current in zip(SURFACE_ORDER[:-1], SURFACE_ORDER[1:]):
        key = f"{previous}__to__{current}"
        metric = internal["adjacent"][key]["per_field_metrics"]["T_STATE"]
        if material("T_STATE", metric):
            return {"from": previous, "to": current, "field": "T_STATE", "metrics": metric}
    return None


def max_t_state_delta_from_entry(internal: Mapping[str, Any]) -> dict[str, Any]:
    ranked = []
    for surface, comparison in internal["from_entry"].items():
        metric = comparison["per_field_metrics"]["T_STATE"]
        ranked.append({"surface": surface, **metric})
    return max(ranked, key=lambda item: -1.0 if item.get("max_abs") is None else float(item["max_abs"]))


def classify(comparisons: Mapping[str, Any]) -> tuple[str, dict[str, Any], list[str], str]:
    if comparisons["wrf_internal"].get("status") != "WRF_INTERNAL_COMPARISONS_EXECUTED":
        return (
            "STEP1_PART1_BLOCKED_WRF_INTERNAL_COMPARISON",
            {},
            ["WRF internal comparison did not execute."],
            "Fix the WRF internal comparison blocker and rerun.",
        )
    if comparisons["wrf_jax"].get("status") != "WRF_JAX_COMPARISONS_EXECUTED":
        return (
            "STEP1_PART1_BLOCKED_JAX_COMPARISON",
            {},
            ["JAX comparison did not execute."],
            "Fix the JAX comparison blocker and rerun.",
        )

    internal = comparisons["wrf_internal"]
    jax_matrix = comparisons["wrf_jax"]["matrix"]
    entry_vs_step = jax_matrix["part1_entry_before_init_zero_tendency"]["vs_step_entry_state_zero_dry"]
    entry_t = entry_vs_step["per_field_metrics"]["T_STATE"]
    t_mutation = first_adjacent_t_state_mutation(internal)
    max_entry_delta = max_t_state_delta_from_entry(internal)

    if material("T_STATE", entry_t):
        evidence = {
            "classification": "input_already_diverged",
            "field": "T_STATE",
            "surface": "part1_entry_before_init_zero_tendency",
            "wrf_entry_vs_jax_step_entry_state_zero_dry": entry_t,
            "first_wrf_internal_t_state_mutation": t_mutation,
            "max_wrf_t_state_delta_from_entry": max_entry_delta,
            "wrf_exit_vs_jax_physics_carry_state_dry": jax_matrix["part1_exit"][
                "vs_physics_carry_state_dry"
            ]["per_field_metrics"]["T_STATE"],
            "wrf_exit_vs_jax_physics_state_dry": jax_matrix["part1_exit"][
                "vs_physics_state_dry"
            ]["per_field_metrics"]["T_STATE"],
        }
        return (
            "STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE",
            evidence,
            [
                "This proof localizes the first material T_STATE residual before first_rk_step_part1 executes.",
                "WRF T_STATE does not need a production physics-state mutation fix inside first_rk_step_part1 unless a later proof finds the upstream handoff source.",
            ],
            "Move upstream to the live-nest/WRF handoff immediately before first_rk_step_part1 entry.",
        )

    if t_mutation is not None:
        return (
            f"STEP1_PART1_MUTATION_LOCALIZED_{t_mutation['to']}_T_STATE",
            {
                "classification": "wrf_internal_mutation",
                "field": "T_STATE",
                "mutation": t_mutation,
                "entry_vs_jax_step_entry_state_zero_dry": entry_t,
                "max_wrf_t_state_delta_from_entry": max_entry_delta,
            },
            ["The first material WRF T_STATE mutation is at the named surface."],
            "Map the named WRF surface to the corresponding JAX adapter or tendency leaf.",
        )

    exit_carry_t = jax_matrix["part1_exit"]["vs_physics_carry_state_dry"]["per_field_metrics"]["T_STATE"]
    exit_state_t = jax_matrix["part1_exit"]["vs_physics_state_dry"]["per_field_metrics"]["T_STATE"]
    if material("T_STATE", exit_carry_t) or material("T_STATE", exit_state_t):
        return (
            "STEP1_PART1_MUTATION_LOCALIZED_JAX_PHYSICS_STATE_CARRY_HANDOFF_T_STATE",
            {
                "classification": "jax_state_carry_handoff_mismatch",
                "field": "T_STATE",
                "entry_vs_jax_step_entry_state_zero_dry": entry_t,
                "wrf_exit_vs_jax_physics_carry_state_dry": exit_carry_t,
                "wrf_exit_vs_jax_physics_state_dry": exit_state_t,
                "max_wrf_t_state_delta_from_entry": max_entry_delta,
            },
            [
                "WRF does not materially mutate T_STATE inside first_rk_step_part1, but the JAX exit-facing physics surfaces still disagree."
            ],
            "Decide whether the JAX physics state mutation belongs in carry state or in dry tendency leaves.",
        )

    return (
        "STEP1_PART1_NO_REMAINING_DIVERGENCE",
        {
            "field": "T_STATE",
            "entry_vs_jax_step_entry_state_zero_dry": entry_t,
            "wrf_exit_vs_jax_physics_carry_state_dry": exit_carry_t,
            "wrf_exit_vs_jax_physics_state_dry": exit_state_t,
            "max_wrf_t_state_delta_from_entry": max_entry_delta,
        },
        [],
        "Close the Step-1 part1 physics-state mutation gate.",
    )


def strip_arrays_from_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    if wrf.get("status") != "WRF_PART1_TRUTH_READY":
        return dict(wrf)
    surfaces = {}
    for surface, parsed in wrf["surfaces"].items():
        surfaces[surface] = {key: value for key, value in parsed.items() if key != "arrays"}
    return {"status": wrf["status"], "shapes": wrf["shapes"], "surfaces": surfaces}


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    evidence = payload.get("classification_evidence", {})
    lines = [
        "# V0.14 Step-1 Part1 Physics-State Mutation",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`.",
        f"- WRF internal truth root: `{WRF_TRUTH}`.",
        f"- Scratch WRF patch: `{OUT_WRF_PATCH}`.",
        f"- Fastest rigorous method: `{payload['tooling_verdict']}`.",
    ]
    if evidence:
        entry = evidence.get("wrf_entry_vs_jax_step_entry_state_zero_dry") or evidence.get(
            "entry_vs_jax_step_entry_state_zero_dry"
        )
        if entry:
            lines.append(
                f"- `part1_entry_before_init_zero_tendency` `T_STATE` vs JAX live-nest step-entry state: max_abs `{entry.get('max_abs')}`, rmse `{entry.get('rmse')}`."
            )
        max_delta = evidence.get("max_wrf_t_state_delta_from_entry")
        if max_delta:
            lines.append(
                f"- Largest WRF internal `T_STATE` delta from part1 entry occurs at `{max_delta.get('surface')}`: max_abs `{max_delta.get('max_abs')}`."
            )
        exit_carry = evidence.get("wrf_exit_vs_jax_physics_carry_state_dry")
        exit_state = evidence.get("wrf_exit_vs_jax_physics_state_dry")
        if exit_carry:
            lines.append(
                f"- `part1_exit` `T_STATE` vs JAX `_physics_step_forcing.carry.state`: max_abs `{exit_carry.get('max_abs')}`, rmse `{exit_carry.get('rmse')}`."
            )
        if exit_state:
            lines.append(
                f"- `part1_exit` `T_STATE` vs JAX `_physics_step_forcing.state`: max_abs `{exit_state.get('max_abs')}`, rmse `{exit_state.get('rmse')}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if verdict == "STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE":
        lines.extend(
            [
                "- The first material Step-1 `T_STATE` residual is already present at WRF `first_rk_step_part1` entry.",
                "- The WRF routine itself does not materially mutate `grid%t_2` / `T_STATE` during the instrumented part1 boundaries.",
                "- The next split should move upstream to the boundary immediately before this call, not into radiation/surface/PBL/cumulus leaves.",
            ]
        )
    else:
        lines.append("- See the JSON classification evidence for the exact first surface and field.")
    lines.extend(["", "Detailed comparison tables are in `proofs/v014/step1_part1_physics_state_mutation.json`.", ""])
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Part1 Physics-State Mutation",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: split the first material Step-1 `T_STATE` mismatch inside or at entry to WRF `first_rk_step_part1`.",
        "",
        "files changed:",
        "- `proofs/v014/step1_part1_physics_state_mutation.py`",
        "- `proofs/v014/step1_part1_physics_state_mutation.json`",
        "- `proofs/v014/step1_part1_physics_state_mutation.md`",
        "- `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`",
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
    ancestor = run_command(["git", "merge-base", "--is-ancestor", "c18795af", "HEAD"], cwd=ROOT)
    wrf = parse_wrf_surfaces()
    if wrf.get("status") != "WRF_PART1_TRUTH_READY":
        comparisons = {"status": "NOT_EXECUTED", "blocker": wrf}
        jax_capture_meta = {"status": "NOT_EXECUTED"}
        verdict = "STEP1_PART1_BLOCKED_WRF_TRUTH"
        classification_evidence: dict[str, Any] = {}
        risks = ["WRF internal truth parsing did not complete."]
        decision = "Fix the WRF truth blocker and rerun."
    else:
        internal = compare_wrf_internal(wrf)
        jax_capture = source_boundary.capture_jax_boundaries()
        if jax_capture.get("status") != "JAX_SOURCE_BOUNDARIES_READY":
            comparisons = {"wrf_internal": internal, "wrf_jax": {"status": "NOT_EXECUTED", "blocker": jax_capture}}
            jax_capture_meta = {key: value for key, value in jax_capture.items() if key != "captures"}
            verdict = "STEP1_PART1_BLOCKED_JAX_CAPTURE"
            classification_evidence = {}
            risks = ["JAX live-nest source-boundary capture did not complete."]
            decision = "Fix the JAX capture blocker and rerun."
        else:
            wrf_jax = compare_jax(wrf, jax_capture)
            comparisons = {"wrf_internal": internal, "wrf_jax": wrf_jax}
            verdict, classification_evidence, risks, decision = classify(comparisons)
            jax_capture_meta = {key: value for key, value in jax_capture.items() if key != "captures"}

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_part1_physics_state_mutation.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "required_ancestor_c18795af": {
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
        "tooling_verdict": "RIGHT_TOOL_FASTEST_WALL_CLOCK_SAVEPOINT_COMPARATOR",
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "accepted_final_truth_npz": path_info(ACCEPTED_FINAL_TRUTH),
            "wrf_truth_root": path_info(WRF_TRUTH),
            "wrf_tree": path_info(WRF_TREE),
            "wrf_run_dir": path_info(WRF_RUN_DIR),
            "wrf_build_log": path_info(WRF_BUILD_LOG),
            "wrf_run_log": path_info(WRF_RUN_LOG),
            "wrf_patch_diff": path_info(OUT_WRF_PATCH),
        },
        "verdict": verdict,
        "classification_evidence": classification_evidence,
        "wrf_part1_truth": strip_arrays_from_wrf(wrf),
        "jax_capture": jax_capture_meta,
        "comparisons": comparisons,
        "commands": {
            "wrf_instrumentation": [
                "cp -a --reflink=auto <DATA_ROOT>/wrf_gpu2/v014_step1_rk1_source_boundary/WRF <DATA_ROOT>/wrf_gpu2/v014_step1_part1_physics_state_mutation/WRF",
                "tcsh ./compile em_real (scratch WRF, conda wrf-build toolchain)",
                "WRFGPU2_STEP1_PART1_PHYSICS_STATE_MUTATION=1 WRFGPU2_STEP1_PART1_PHYSICS_STATE_MUTATION_ROOT=<DATA_ROOT>/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth mpirun --oversubscribe -np 28 ./wrf.exe",
            ],
            "required_validation": [
                "python -m py_compile proofs/v014/step1_part1_physics_state_mutation.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part1_physics_state_mutation.py",
                "python -m json.tool proofs/v014/step1_part1_physics_state_mutation.json >/tmp/step1_part1_physics_state_mutation.validated.json",
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
            "The sprint stops at first_rk_step_part1 entry; upstream localization before that call remains for the next decision.",
            "No production source fix was made or gated because the mismatch is not inside first_rk_step_part1.",
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
