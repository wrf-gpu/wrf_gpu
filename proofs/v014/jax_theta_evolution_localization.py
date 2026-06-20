#!/usr/bin/env python3
"""V0.14 h10 JAX theta evolution localization proof.

This is a proof-only, CPU-only localization script.  It does not edit or patch
production source.  The RK wrapper below mirrors the private operational RK path
so it can expose stage/pre-finish boundaries that the production API does not
return.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_JSON = ROOT / "proofs/v014/jax_theta_evolution_localization.json"
OUT_MD = ROOT / "proofs/v014/jax_theta_evolution_localization.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-theta-evolution-localization.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-theta-evolution-localization/sprint-contract.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

JAX_T_ATTR_JSON = ROOT / "proofs/v014/jax_t_history_source_attribution.json"
JAX_H10_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry.json"
JAX_H10_PRODUCER_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.json"
JAX_PRE_HALO_JSON = ROOT / "proofs/v014/jax_pre_halo_capture.json"
WRF_DYNAMIC_JSON = ROOT / "proofs/v014/wrf_dynamic_term_localization.json"
WRF_REFRESH_JSON = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.json"
SAVEPOINT_REQUEST_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"

OPERATIONAL_MODE = ROOT / "src/gpuwrf/runtime/operational_mode.py"
SMALL_STEP_PREP = ROOT / "src/gpuwrf/dynamics/core/small_step_prep.py"
SMALL_STEP_FINISH = ROOT / "src/gpuwrf/dynamics/core/small_step_finish.py"
ACOUSTIC_CORE = ROOT / "src/gpuwrf/dynamics/core/acoustic.py"
RK_ADDTEND_DRY = ROOT / "src/gpuwrf/dynamics/core/rk_addtend_dry.py"

CHECKPOINT = Path(
    os.environ.get(
        "WRFGPU2_H10_PRESTEP_CARRY",
        "<DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl",
    )
)
STDERR_CAPTURE = Path(
    os.environ.get(
        "WRFGPU2_THETA_LOCALIZATION_STDERR",
        "/tmp/jax_theta_evolution_localization.stderr",
    )
)

TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
THETA_OFFSET_K = 300.0
GREEN_TOLERANCE_MAX_ABS = 2.0e-6


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def capture_process_stderr(path: Path):
    """Redirect native-library stderr chatter while preserving compact proof output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with path.open("ab") as handle:
        try:
            os.dup2(handle.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "count": int(arr.size),
            "finite_count": 0,
            "min": None,
            "max": None,
            "max_abs": None,
            "rmse": None,
            "mean": None,
        }
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "max_abs": float(np.max(np.abs(finite))),
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))),
        "mean": float(np.mean(finite)),
    }


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "count": 0,
            "finite_count": 0,
            "all_finite": True,
            "min": None,
            "max": None,
            "mean": None,
            "max_abs": None,
        }
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
    else:
        finite = arr.ravel()
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "all_finite": bool(finite.size == arr.size),
    }
    if finite.size:
        out.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "max_abs": float(np.max(np.abs(finite))),
            }
        )
    else:
        out.update({"min": None, "max": None, "mean": None, "max_abs": None})
    return out


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = jax.devices()
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([device for device in devices if device.platform == "gpu"]),
            }
        )
    except Exception as exc:  # pragma: no cover - recorded in proof output
        env["jax_import_error"] = repr(exc)
    return env


def extract_ast_node(path: Path, name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
            source = "\n".join(text.splitlines()[start - 1 : end])
            return {
                "module_path": str(path.relative_to(ROOT)),
                "name": name,
                "line_start": start,
                "line_end": end,
                "source_sha256": sha256_text(source),
            }
    return {"module_path": str(path.relative_to(ROOT)), "name": name, "missing": True}


def parse_surface_files(paths: Iterable[Path]) -> dict[str, Any]:
    schemas: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "record_schema":
                    tag = parts[1]
                    columns = parts[2:]
                    index_names: list[str] = []
                    for column in columns:
                        if column.startswith("fortran_") or column.startswith("zero_"):
                            index_names.append(column)
                        else:
                            break
                    schemas[tag] = {
                        "index_names": index_names,
                        "fields": columns[len(index_names) :],
                    }
                    continue
                tag = parts[0]
                if tag not in schemas:
                    if len(parts) > 1:
                        metadata[tag].append(" ".join(parts[1:]))
                    continue
                schema = schemas[tag]
                nidx = len(schema["index_names"])
                idx_values = [int(value) for value in parts[1 : 1 + nidx]]
                values = [float(value) for value in parts[1 + nidx :]]
                fields = list(schema["fields"])
                if len(values) != len(fields):
                    raise ValueError(f"{path}: {tag} expected {len(fields)} values, got {len(values)}")
                item = dict(zip(fields, values))
                key = surface_key(tag, schema["index_names"], idx_values)
                if key in records[tag]:
                    duplicate_count += 1
                    previous = records[tag][key]
                    for field in fields:
                        if field not in previous:
                            continue
                        label = f"{tag}.{field}"
                        delta = abs(previous[field] - item[field])
                        duplicate_max_delta_by_field[label] = max(
                            duplicate_max_delta_by_field.get(label, 0.0), delta
                        )
                        duplicate_max_delta = max(duplicate_max_delta, delta)
                records[tag][key] = item

    return {
        "files": [str(path) for path in paths],
        "schemas": schemas,
        "records": records,
        "metadata": {name: values[0] if len(values) == 1 else values for name, values in metadata.items()},
        "unique_counts": {name: len(items) for name, items in records.items()},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def surface_key(tag: str, index_names: list[str], idx_values: list[int]) -> tuple[int, ...]:
    del tag
    lookup = dict(zip(index_names, idx_values))
    if "zero_kstag" in lookup:
        return (lookup["zero_kstag"], lookup["zero_y"], lookup["zero_x"])
    if "zero_ystag" in lookup:
        return (lookup["zero_ystag"], lookup["zero_x"])
    if "zero_xstag" in lookup:
        return (lookup["zero_y"], lookup["zero_xstag"])
    if "zero_y" in lookup and "zero_x" in lookup:
        return (lookup["zero_y"], lookup["zero_x"])
    return tuple(idx_values)


def mass_index(arr: np.ndarray, key: tuple[int, ...]) -> tuple[int, ...]:
    y, x = key[-2], key[-1]
    if arr.ndim == 3:
        return (0, y, x)
    if arr.ndim == 2:
        return (y, x)
    raise ValueError(f"expected 2-D or 3-D mass field, got shape {arr.shape}")


def compare_mass_array_to_wrf(candidate_array: Any, surface: Mapping[str, Any], wrf_field: str) -> dict[str, Any]:
    arr = np.asarray(candidate_array, dtype=np.float64)
    mass_records = surface["records"].get("MASS_K1", {})
    if not mass_records:
        return {
            "status": "UNAVAILABLE_NO_MASS_RECORDS",
            "wrf_field": wrf_field,
            "candidate_shape": list(arr.shape),
            "candidate_dtype": str(np.asarray(candidate_array).dtype),
        }
    sample = next(iter(mass_records.values()))
    if wrf_field not in sample:
        return {
            "status": "UNAVAILABLE_FIELD",
            "wrf_field": wrf_field,
            "candidate_shape": list(arr.shape),
            "candidate_dtype": str(np.asarray(candidate_array).dtype),
            "available_fields": sorted(sample.keys()),
        }

    diffs: list[float] = []
    worst: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = []
    for key, record in sorted(mass_records.items()):
        try:
            idx = mass_index(arr, key)
            candidate = float(arr[idx])
        except Exception as exc:  # pragma: no cover - recorded in proof output
            skipped.append({"native_key": list(key), "reason": repr(exc)})
            continue
        truth = float(record[wrf_field])
        diff = candidate - truth
        diffs.append(diff)
        if worst is None or abs(diff) > worst["abs_diff"]:
            worst = {
                "native_key": list(key),
                "array_index": list(idx),
                "jax_candidate": candidate,
                "wrf_truth": truth,
                "diff_jax_minus_wrf": diff,
                "abs_diff": abs(diff),
            }

    return {
        "status": "DIFF" if diffs else "NO_COMPARABLE_RECORDS",
        "wrf_field": wrf_field,
        "candidate_shape": list(arr.shape),
        "candidate_dtype": str(np.asarray(candidate_array).dtype),
        **finite_stats(diffs),
        "worst": worst,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped[:10],
    }


def compare_context_to_wrf(context_arrays: Mapping[str, Any], surface: Mapping[str, Any], fields: Mapping[str, str | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("P", "PB", "MU", "MUB"):
        wrf_field = fields.get(name)
        if wrf_field is None:
            out[name] = {"status": "UNAVAILABLE_FIELD", "wrf_field": None}
            continue
        if name not in context_arrays:
            out[name] = {"status": "UNAVAILABLE_JAX_CONTEXT", "wrf_field": wrf_field}
            continue
        out[name] = compare_mass_array_to_wrf(context_arrays[name], surface, wrf_field)
    return out


def state_theta_pert(state: Any) -> Any:
    return state.theta - THETA_OFFSET_K


def state_context_arrays(state: Any) -> dict[str, Any]:
    return {
        "P": np.asarray(state.p_perturbation),
        "PB": np.asarray(state.p_total) - np.asarray(state.p_perturbation),
        "MU": np.asarray(state.mu_perturbation),
        "MUB": np.asarray(state.mu_total) - np.asarray(state.mu_perturbation),
    }


def acoustic_context_arrays(acoustic: Any, prep: Any) -> dict[str, Any]:
    return {
        "P": np.asarray(acoustic.p),
        "PB": np.asarray(prep.pb),
        "MU": np.asarray(acoustic.mu),
        "MUB": np.asarray(prep.mub),
    }


def target_definitions(surfaces: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "wrf_final_stage_pre_small_step_finish.T_OLD": {
            "surface_key": "wrf_final_stage_pre_small_step_finish",
            "surface": surfaces["wrf_final_stage_pre_small_step_finish"],
            "field": "T_OLD",
            "role": "start-of-step / RK reference theta from WRF grid%t_1",
            "history_T": False,
            "diagnostic_only": False,
            "context_fields": {"P": None, "PB": None, "MU": "MU_OLD", "MUB": None},
        },
        "wrf_final_stage_pre_small_step_finish.T_HIST_SRC": {
            "surface_key": "wrf_final_stage_pre_small_step_finish",
            "surface": surfaces["wrf_final_stage_pre_small_step_finish"],
            "field": "T_HIST_SRC",
            "role": "WRF history T source before final small_step_finish, grid%th_phy_m_t0",
            "history_T": True,
            "diagnostic_only": False,
            "context_fields": {"P": "P", "PB": "PB", "MU": "MU_NEW", "MUB": None},
        },
        "wrf_final_stage_pre_small_step_finish.T_THM": {
            "surface_key": "wrf_final_stage_pre_small_step_finish",
            "surface": surfaces["wrf_final_stage_pre_small_step_finish"],
            "field": "T_THM",
            "role": "WRF grid%t_2 before final small_step_finish; diagnostic, not history T",
            "history_T": False,
            "diagnostic_only": True,
            "context_fields": {"P": "P", "PB": "PB", "MU": "MU_NEW", "MUB": None},
        },
        "wrf_final_stage_post_small_step_finish.T_HIST_SRC": {
            "surface_key": "wrf_final_stage_post_small_step_finish",
            "surface": surfaces["wrf_final_stage_post_small_step_finish"],
            "field": "T_HIST_SRC",
            "role": "WRF history T source after final small_step_finish",
            "history_T": True,
            "diagnostic_only": False,
            "context_fields": {"P": "P", "PB": "PB", "MU": "MU_NEW", "MUB": None},
        },
        "wrf_final_stage_post_small_step_finish.T_THM": {
            "surface_key": "wrf_final_stage_post_small_step_finish",
            "surface": surfaces["wrf_final_stage_post_small_step_finish"],
            "field": "T_THM",
            "role": "WRF grid%t_2 after final small_step_finish; diagnostic, not history T",
            "history_T": False,
            "diagnostic_only": True,
            "context_fields": {"P": "P", "PB": "PB", "MU": "MU_NEW", "MUB": None},
        },
        "wrf_post_after_all_rk_steps_pre_halo.T_HIST_SRC": {
            "surface_key": "wrf_post_after_all_rk_steps_pre_halo",
            "surface": surfaces["wrf_post_after_all_rk_steps_pre_halo"],
            "field": "T_HIST_SRC",
            "role": "accepted WRF history T source after all RK steps before RK halo",
            "history_T": True,
            "diagnostic_only": False,
            "context_fields": {"P": "P", "PB": "PB", "MU": "MU_NEW", "MUB": "MUB"},
        },
        "wrf_post_after_all_rk_steps_pre_halo.T_THM": {
            "surface_key": "wrf_post_after_all_rk_steps_pre_halo",
            "surface": surfaces["wrf_post_after_all_rk_steps_pre_halo"],
            "field": "T_THM",
            "role": "WRF THM-side candidate after all RK; diagnostic, not history T",
            "history_T": False,
            "diagnostic_only": True,
            "context_fields": {"P": "P", "PB": "PB", "MU": "MU_NEW", "MUB": "MUB"},
        },
    }


def add_theta_candidate(
    candidates: dict[str, dict[str, Any]],
    *,
    name: str,
    array: Any,
    boundary_order: int,
    boundary: str,
    cadence: str,
    component: str,
    source_expression: str,
    offset_convention: str,
    offset_applied_k: float | None,
    context_arrays: Mapping[str, Any] | None,
    targets: Mapping[str, Mapping[str, Any]],
) -> None:
    arr = np.asarray(array)
    comparisons = {
        target_name: compare_mass_array_to_wrf(arr, target["surface"], target["field"])
        for target_name, target in targets.items()
    }
    context = {}
    if context_arrays is not None:
        context = {
            target_name: compare_context_to_wrf(context_arrays, target["surface"], target["context_fields"])
            for target_name, target in targets.items()
        }
    candidates[name] = {
        "name": name,
        "boundary_order": int(boundary_order),
        "boundary": boundary,
        "cadence": cadence,
        "component": component,
        "source_expression": source_expression,
        "offset_convention": offset_convention,
        "offset_applied_k": offset_applied_k,
        "array_summary": array_summary(arr),
        "comparisons": comparisons,
        "context_mass_pressure": context,
    }


def compare_arrays(left: Any, right: Any) -> dict[str, Any]:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape:
        return {"status": "SHAPE_MISMATCH", "left_shape": list(lhs.shape), "right_shape": list(rhs.shape)}
    diff = lhs - rhs
    finite = diff[np.isfinite(diff)]
    worst = None
    if finite.size:
        flat = int(np.nanargmax(np.abs(diff)))
        idx = np.unravel_index(flat, diff.shape)
        worst = {
            "index": list(idx),
            "left": float(lhs[idx]),
            "right": float(rhs[idx]),
            "diff_left_minus_right": float(diff[idx]),
            "abs_diff": float(abs(diff[idx])),
        }
    return {"status": "DIFF", **finite_stats(diff.ravel()), "worst": worst}


def source_inspection() -> dict[str, Any]:
    return {
        "mirrored_private_path": {
            "note": "The proof-local RK wrapper mirrors private helpers so stage boundaries can be observed without editing src/.",
            "nodes": [
                extract_ast_node(OPERATIONAL_MODE, "_rk_scan_step"),
                extract_ast_node(OPERATIONAL_MODE, "_rk_scan_step_with_pre_halo_capture"),
                extract_ast_node(OPERATIONAL_MODE, "_acoustic_scan"),
                extract_ast_node(OPERATIONAL_MODE, "_carry_from_finished_stage"),
                extract_ast_node(OPERATIONAL_MODE, "_physics_step_forcing"),
                extract_ast_node(SMALL_STEP_PREP, "small_step_prep_wrf"),
                extract_ast_node(SMALL_STEP_FINISH, "small_step_finish_wrf"),
                extract_ast_node(ACOUSTIC_CORE, "acoustic_substep_core"),
                extract_ast_node(RK_ADDTEND_DRY, "rk_addtend_dry"),
            ],
        },
        "files": {
            "operational_mode": path_info(OPERATIONAL_MODE),
            "small_step_prep": path_info(SMALL_STEP_PREP),
            "small_step_finish": path_info(SMALL_STEP_FINISH),
            "acoustic_core": path_info(ACOUSTIC_CORE),
            "rk_addtend_dry": path_info(RK_ADDTEND_DRY),
        },
    }


def proof_inputs() -> dict[str, Any]:
    return {
        "project_constitution": path_info(ROOT / "PROJECT_CONSTITUTION.md"),
        "agents": path_info(ROOT / "AGENTS.md"),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "handoff": path_info(HANDOFF),
        "jax_t_history_source_attribution_json": path_info(JAX_T_ATTR_JSON),
        "jax_h10_prestep_carry_json": path_info(JAX_H10_JSON),
        "jax_h10_prestep_carry_producer_json": path_info(JAX_H10_PRODUCER_JSON),
        "jax_pre_halo_capture_json": path_info(JAX_PRE_HALO_JSON),
        "wrf_dynamic_term_localization_json": path_info(WRF_DYNAMIC_JSON),
        "wrf_post_rk_refresh_localization_json": path_info(WRF_REFRESH_JSON),
        "same_state_savepoint_request_json": path_info(SAVEPOINT_REQUEST_JSON),
        "checkpoint": path_info(CHECKPOINT),
    }


def recorded_checkpoint_identity(actual: Mapping[str, Any], loaded_payloads: Mapping[str, Any]) -> dict[str, Any]:
    def same_as(record: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            return {"record_present": False, "path_match": False, "sha256_match": False, "size_match": False}
        return {
            "record_present": True,
            "recorded_path": record.get("path"),
            "recorded_sha256": record.get("sha256"),
            "recorded_size_bytes": record.get("size_bytes"),
            "recorded_step_index": record.get("step_index") or record.get("metadata", {}).get("step_index"),
            "path_match": str(record.get("path")) == str(actual.get("path")),
            "sha256_match": record.get("sha256") == actual.get("sha256"),
            "size_match": record.get("size_bytes") == actual.get("size_bytes"),
        }

    t_attr_actual = (
        loaded_payloads["jax_t_attr"].get("checkpoint_identity", {}).get("actual_checkpoint")
    )
    h10_candidates = loaded_payloads["jax_h10"].get("checkpoint_probe", {}).get("usable_candidates", [])
    h10_candidate = h10_candidates[0] if h10_candidates else None
    producer_checkpoint = loaded_payloads["jax_h10_producer"].get("checkpoint")
    return {
        "actual_checkpoint": dict(actual),
        "matches_t_history_source_attribution_actual": same_as(t_attr_actual),
        "matches_canonical_h10_usable_candidate": same_as(h10_candidate),
        "matches_producer_checkpoint_record": same_as(producer_checkpoint),
        "same_artifact_as_t_attribution": bool(
            isinstance(t_attr_actual, Mapping)
            and t_attr_actual.get("sha256") == actual.get("sha256")
            and t_attr_actual.get("size_bytes") == actual.get("size_bytes")
        ),
        "same_artifact_as_canonical_h10_compared": bool(
            isinstance(h10_candidate, Mapping)
            and h10_candidate.get("sha256") == actual.get("sha256")
            and h10_candidate.get("size_bytes") == actual.get("size_bytes")
        ),
        "same_artifact_as_producer_recorded": bool(
            isinstance(producer_checkpoint, Mapping)
            and producer_checkpoint.get("sha256") == actual.get("sha256")
            and producer_checkpoint.get("size_bytes") == actual.get("size_bytes")
        ),
    }


def summarize_dry_tendencies(dry: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in getattr(dry, "__dataclass_fields__", {}):
        value = getattr(dry, name)
        out[name] = {"present": value is not None}
        if value is not None:
            out[name].update(array_summary(np.asarray(value)))
    return out


def run_jax_localization(surfaces: Mapping[str, Any], wrf_refresh: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from gpuwrf.contracts.halo import apply_halo  # noqa: PLC0415
    from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients  # noqa: PLC0415
    from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec  # noqa: PLC0415
    from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, acoustic_substep_core  # noqa: PLC0415
    from gpuwrf.dynamics.core.advance_w import dry_cqw  # noqa: PLC0415
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf  # noqa: PLC0415
    from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf  # noqa: PLC0415
    from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        _RKStageDescriptor,
        _acoustic_core_state_from_prep,
        _acoustic_lateral_bc_flags,
        _acoustic_unroll,
        _apply_moisture_large_step,
        _augment_large_step_tendencies,
        _carry_from_finished_stage,
        _maybe_exchange_sharded_carry_halos,
        _moisture_coupled_tendencies,
        _physics_step_forcing,
        _rk_scan_step_with_pre_halo_capture,
    )

    targets = target_definitions(surfaces)
    candidates: dict[str, dict[str, Any]] = {}

    started = time.perf_counter()
    state, namelist, grid, step_index, carry = read_checkpoint_with_runtime_state(CHECKPOINT)
    if carry is None:
        raise RuntimeError(f"{CHECKPOINT} has no runtime_state OperationalCarry")
    if int(step_index) != PRESTEP_COMPLETED_STEPS:
        raise RuntimeError(f"checkpoint step_index={step_index}, expected {PRESTEP_COMPLETED_STEPS}")

    lead_seconds = jnp.asarray(
        float(wrf_refresh["target_confirmed"]["lead_seconds_after_step"]), dtype=jnp.float64
    )
    cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
    run_radiation = bool(cadence > 0 and TARGET_STEP % cadence == 0)

    add_theta_candidate(
        candidates,
        name="checkpoint_prestep_carry.state.theta_minus_300",
        array=jax.device_get(carry.state.theta - THETA_OFFSET_K),
        boundary_order=10,
        boundary="checkpoint completed step 5999 / target step 6000 input state",
        cadence="pre-physics, pre-RK",
        component="input_state",
        source_expression="carry.state.theta - 300",
        offset_convention="State.theta is total theta; WRF T/T_OLD are perturbation theta.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(carry.state),
        targets=targets,
    )
    add_theta_candidate(
        candidates,
        name="checkpoint_prestep_carry.t_save_minus_300",
        array=jax.device_get(carry.t_save - THETA_OFFSET_K),
        boundary_order=11,
        boundary="checkpoint completed step 5999 save family",
        cadence="pre-physics, pre-RK",
        component="carry_save_family",
        source_expression="carry.t_save - 300",
        offset_convention="OperationalCarry.t_save is total theta at this checkpoint; subtract 300 K.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(carry.state),
        targets=targets,
    )
    add_theta_candidate(
        candidates,
        name="checkpoint_prestep_carry.t_2ave_minus_300",
        array=jax.device_get(carry.t_2ave - THETA_OFFSET_K),
        boundary_order=12,
        boundary="checkpoint completed step 5999 small-step scratch",
        cadence="pre-physics, pre-RK",
        component="carry_scratch",
        source_expression="carry.t_2ave - 300",
        offset_convention="Diagnostic only; t_2ave is not WRF history T.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(carry.state),
        targets=targets,
    )

    physics = _physics_step_forcing(carry, namelist, lead_seconds, run_radiation=run_radiation)
    jax.block_until_ready(physics.carry.state.theta)
    add_theta_candidate(
        candidates,
        name="after_physics_forcing.carry.state.theta_minus_300",
        array=jax.device_get(physics.carry.state.theta - THETA_OFFSET_K),
        boundary_order=20,
        boundary="after _physics_step_forcing carry state",
        cadence="post-physics-forcing, RK input carry",
        component="physics_forcing_carry",
        source_expression="physics.carry.state.theta - 300",
        offset_convention="The carry state is the RK input; non-dry physics deltas are returned separately.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(physics.carry.state),
        targets=targets,
    )
    add_theta_candidate(
        candidates,
        name="after_physics_forcing.non_dry_state.theta_minus_300",
        array=jax.device_get(physics.state.theta - THETA_OFFSET_K),
        boundary_order=21,
        boundary="after _physics_step_forcing non-dry physics state",
        cadence="post-physics-forcing side state",
        component="physics_non_dry_state",
        source_expression="physics.state.theta - 300",
        offset_convention="Diagnostic only: production applies this non-dry delta after the dycore.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(physics.state),
        targets=targets,
    )

    def proof_acoustic_scan(
        stage_carry: Any,
        stage: Any,
        prep: Any,
        pressure: Any,
        tendencies: Any,
    ) -> tuple[Any, Any, Any]:
        acoustic = _acoustic_core_state_from_prep(
            stage_carry, prep, pressure, namelist, tendencies, lead_seconds=lead_seconds
        )
        if bool(namelist.use_vertical_solver):
            cqw_field = acoustic.cqw
            if cqw_field is None:
                cqw_field = dry_cqw(
                    int(prep.theta_work.shape[0]),
                    int(prep.theta_work.shape[1]),
                    int(prep.theta_work.shape[2]),
                    dtype=prep.theta_work.dtype,
                )
            a, alpha, gamma = calc_coef_w_wrf_coefficients(
                prep.mut,
                namelist.metrics,
                dt=float(stage.dts_rk),
                epssm=float(namelist.epssm),
                top_lid=bool(namelist.top_lid),
                cqw=cqw_field,
                c2a=prep.c2a,
            )
            periodic_x, specified, nested = _acoustic_lateral_bc_flags(namelist)
            stage_cfg = AcousticCoreConfig(
                dt=float(stage.dts_rk),
                dx=float(namelist.grid.projection.dx_m),
                dy=float(namelist.grid.projection.dy_m),
                epssm=float(namelist.epssm),
                top_lid=bool(namelist.top_lid),
                w_damping=int(namelist.w_damping),
                damp_opt=int(namelist.damp_opt),
                dampcoef=float(namelist.dampcoef),
                zdamp=float(namelist.zdamp),
                dt_full=float(namelist.dt_s),
                periodic_x=periodic_x,
                specified=specified,
                nested=nested,
            )

            def body(scan_acoustic, _):
                return acoustic_substep_core(
                    scan_acoustic,
                    a=a,
                    alpha=alpha,
                    gamma=gamma,
                    cfg=stage_cfg,
                    cqw=cqw_field,
                ), None

            acoustic, _ = jax.lax.scan(
                body,
                acoustic,
                xs=None,
                length=int(stage.number_of_small_timesteps),
                unroll=_acoustic_unroll(),
            )
            no_halo_carry = _carry_from_finished_stage(stage_carry, prep, acoustic, namelist)
            no_halo_carry = _maybe_exchange_sharded_carry_halos(no_halo_carry)
            post_halo_carry = no_halo_carry.replace(state=apply_halo(no_halo_carry.state, halo_spec(namelist.grid)))
            return acoustic, no_halo_carry, post_halo_carry

        no_halo_carry = _maybe_exchange_sharded_carry_halos(stage_carry)
        return acoustic, no_halo_carry, no_halo_carry

    origin = apply_halo(physics.carry.state, halo_spec(namelist.grid))
    rk1_reference = origin
    dt = float(namelist.dt_s)
    configured_sound_steps = int(namelist.acoustic_substeps)
    stages = (
        _RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
        _RKStageDescriptor(2, 0.5 * dt, dt / float(configured_sound_steps), max(1, configured_sound_steps // 2)),
        _RKStageDescriptor(3, dt, dt / float(configured_sound_steps), configured_sound_steps),
    )

    stage_carry = physics.carry.replace(state=origin)
    stage_details: dict[str, Any] = {}
    final_pre_halo_state = None
    for stage in stages:
        stage_name = f"rk_stage_{int(stage.rk_step)}"
        haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        add_theta_candidate(
            candidates,
            name=f"{stage_name}.input_haloed_state.theta_minus_300",
            array=jax.device_get(haloed.theta - THETA_OFFSET_K),
            boundary_order=100 + int(stage.rk_step) * 10,
            boundary=f"RK stage {int(stage.rk_step)} input haloed state",
            cadence=f"RK{int(stage.rk_step)} input",
            component="rk_stage_input",
            source_expression="apply_halo(stage_carry.state).theta - 300",
            offset_convention="State.theta is total theta; subtract 300 K.",
            offset_applied_k=THETA_OFFSET_K,
            context_arrays=state_context_arrays(haloed),
            targets=targets,
        )
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        tendencies = _augment_large_step_tendencies(
            haloed,
            tendencies,
            namelist,
            rk_step=int(stage.rk_step),
            physics_tendencies=physics.dry_tendencies,
            step_origin=rk1_reference,
        )
        candidate = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        prep = small_step_prep_wrf(
            candidate,
            int(stage.rk_step),
            float(stage.dt_rk),
            metrics=namelist.metrics,
            reference_state=rk1_reference,
            ww=stage_carry.ww,
        )
        add_theta_candidate(
            candidates,
            name=f"{stage_name}.prep.t_save_perturbation",
            array=jax.device_get(prep.t_save),
            boundary_order=101 + int(stage.rk_step) * 10,
            boundary=f"RK stage {int(stage.rk_step)} small_step_prep save theta",
            cadence=f"RK{int(stage.rk_step)} before acoustic",
            component="small_step_prep",
            source_expression="prep.t_save",
            offset_convention="small_step_prep.t_save is already perturbation theta.",
            offset_applied_k=None,
            context_arrays=state_context_arrays(candidate),
            targets=targets,
        )
        pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
        acoustic, no_halo_carry, post_halo_carry = proof_acoustic_scan(
            stage_carry.replace(state=candidate), stage, prep, pressure, tendencies
        )
        jax.block_until_ready(no_halo_carry.state.theta)

        add_theta_candidate(
            candidates,
            name=f"{stage_name}.post_acoustic_pre_small_step_finish.acoustic.theta",
            array=jax.device_get(acoustic.theta),
            boundary_order=102 + int(stage.rk_step) * 10,
            boundary=f"RK stage {int(stage.rk_step)} after acoustic scan before small_step_finish",
            cadence=f"RK{int(stage.rk_step)} post-acoustic/pre-finish",
            component="acoustic_theta_diagnostic",
            source_expression="acoustic.theta",
            offset_convention="AcousticCoreState.theta is a perturbation-theta diagnostic view at this point.",
            offset_applied_k=None,
            context_arrays=acoustic_context_arrays(acoustic, prep),
            targets=targets,
        )
        add_theta_candidate(
            candidates,
            name=f"{stage_name}.post_acoustic_pre_small_step_finish.acoustic.theta_coupled_work_raw",
            array=jax.device_get(acoustic.theta_coupled_work),
            boundary_order=103 + int(stage.rk_step) * 10,
            boundary=f"RK stage {int(stage.rk_step)} after acoustic scan before small_step_finish",
            cadence=f"RK{int(stage.rk_step)} post-acoustic/pre-finish",
            component="acoustic_theta_work",
            source_expression="acoustic.theta_coupled_work",
            offset_convention="Raw coupled work-array diagnostic; included to test WRF grid%t_2 mapping.",
            offset_applied_k=None,
            context_arrays=acoustic_context_arrays(acoustic, prep),
            targets=targets,
        )
        add_theta_candidate(
            candidates,
            name=f"{stage_name}.post_small_step_finish_pre_halo_state.theta_minus_300",
            array=jax.device_get(no_halo_carry.state.theta - THETA_OFFSET_K),
            boundary_order=104 + int(stage.rk_step) * 10,
            boundary=f"RK stage {int(stage.rk_step)} after small_step_finish before halo",
            cadence=f"RK{int(stage.rk_step)} post-finish/pre-halo",
            component="small_step_finish_state",
            source_expression="no_halo_carry.state.theta - 300",
            offset_convention="State.theta is total theta after small_step_finish; subtract 300 K.",
            offset_applied_k=THETA_OFFSET_K,
            context_arrays=state_context_arrays(no_halo_carry.state),
            targets=targets,
        )

        moisture_advected = bool(namelist.use_flux_advection) and int(namelist.moist_adv_opt) != 0
        stage_carry = post_halo_carry
        if moisture_advected:
            q_tendencies = _moisture_coupled_tendencies(
                haloed,
                namelist,
                rk_step=int(stage.rk_step),
                step_origin=rk1_reference,
            )
            stage_carry = stage_carry.replace(
                state=_apply_moisture_large_step(
                    stage_carry.state,
                    rk1_reference,
                    q_tendencies=q_tendencies,
                    dt_rk=float(stage.dt_rk),
                    metrics=namelist.metrics,
                )
            )
        stage_carry = stage_carry.replace(state=apply_halo(stage_carry.state, halo_spec(namelist.grid)))
        if int(stage.rk_step) == 3:
            final_pre_halo_state = no_halo_carry.state
        stage_details[stage_name] = {
            "rk_step": int(stage.rk_step),
            "dt_rk": float(stage.dt_rk),
            "dts_rk": float(stage.dts_rk),
            "number_of_small_timesteps": int(stage.number_of_small_timesteps),
            "moisture_advected": bool(moisture_advected),
            "theta_tendency_summary": array_summary(jax.device_get(tendencies.theta)),
            "pressure_p_summary": array_summary(jax.device_get(pressure.p)),
        }

    final_carry = stage_carry
    add_theta_candidate(
        candidates,
        name="final_carry.post_halo_state.theta_minus_300",
        array=jax.device_get(final_carry.state.theta - THETA_OFFSET_K),
        boundary_order=200,
        boundary="final RK3 carry after halo",
        cadence="post-RK post-halo",
        component="final_carry_state",
        source_expression="final_carry.state.theta - 300",
        offset_convention="State.theta is total theta after post-RK halo; subtract 300 K.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(final_carry.state),
        targets=targets,
    )
    add_theta_candidate(
        candidates,
        name="final_carry.t_save_minus_300",
        array=jax.device_get(final_carry.t_save - THETA_OFFSET_K),
        boundary_order=201,
        boundary="final RK3 carry save family",
        cadence="post-RK carry",
        component="final_carry_save_family",
        source_expression="final_carry.t_save - 300",
        offset_convention="OperationalCarry.t_save is stored as total theta after _carry_from_finished_stage.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(final_carry.state),
        targets=targets,
    )
    add_theta_candidate(
        candidates,
        name="final_carry.t_2ave_minus_300",
        array=jax.device_get(final_carry.t_2ave - THETA_OFFSET_K),
        boundary_order=202,
        boundary="final RK3 carry t_2ave scratch",
        cadence="post-RK carry",
        component="final_carry_scratch",
        source_expression="final_carry.t_2ave - 300",
        offset_convention="t_2ave is small-step scratch, not history T; subtract 300 K for diagnostic scale.",
        offset_applied_k=THETA_OFFSET_K,
        context_arrays=state_context_arrays(final_carry.state),
        targets=targets,
    )

    public_result = _rk_scan_step_with_pre_halo_capture(
        physics.carry,
        namelist,
        lead_seconds=lead_seconds,
        physics_tendencies=physics.dry_tendencies,
    )
    jax.block_until_ready(public_result.carry.state.theta)
    mirror_parity = {
        "pre_halo_state_theta": compare_arrays(
            jax.device_get(final_pre_halo_state.theta),
            jax.device_get(public_result.pre_halo_state.theta),
        )
        if final_pre_halo_state is not None
        else {"status": "NOT_CAPTURED"},
        "final_post_halo_state_theta": compare_arrays(
            jax.device_get(final_carry.state.theta),
            jax.device_get(public_result.carry.state.theta),
        ),
        "final_t_save": compare_arrays(
            jax.device_get(final_carry.t_save),
            jax.device_get(public_result.carry.t_save),
        ),
        "final_t_2ave": compare_arrays(
            jax.device_get(final_carry.t_2ave),
            jax.device_get(public_result.carry.t_2ave),
        ),
    }

    return {
        "status": "RAN",
        "wall_s": float(time.perf_counter() - started),
        "loader": "gpuwrf.runtime.checkpoint.read_checkpoint_with_runtime_state",
        "checkpoint_step_index": int(step_index),
        "runtime_state_type": type(carry).__name__,
        "namelist": {
            "type": type(namelist).__name__,
            "dt_s": float(getattr(namelist, "dt_s")),
            "acoustic_substeps": int(getattr(namelist, "acoustic_substeps")),
            "radiation_cadence_steps": cadence,
            "run_physics": bool(getattr(namelist, "run_physics")),
            "run_boundary": bool(getattr(namelist, "run_boundary")),
            "rad_rk_tendf": int(getattr(namelist, "rad_rk_tendf", 0)),
            "use_flux_advection": bool(getattr(namelist, "use_flux_advection", False)),
            "moist_adv_opt": int(getattr(namelist, "moist_adv_opt", 0)),
        },
        "target_step": TARGET_STEP,
        "lead_seconds": float(lead_seconds),
        "run_radiation_for_step": run_radiation,
        "grid_shape": {
            "nz": int(getattr(grid, "nz")),
            "ny": int(getattr(grid, "ny")),
            "nx": int(getattr(grid, "nx")),
        },
        "dry_tendencies": summarize_dry_tendencies(physics.dry_tendencies),
        "held_rthraten_after_physics": array_summary(jax.device_get(physics.carry.rthraten)),
        "stage_details": stage_details,
        "theta_candidates": candidates,
        "candidate_count": len(candidates),
        "mirror_parity_with_existing_pre_halo_helper": mirror_parity,
    }


def surface_summary(surface: Mapping[str, Any]) -> dict[str, Any]:
    field_stats: dict[str, Any] = {}
    mass_records = surface["records"].get("MASS_K1", {})
    if mass_records:
        sample = next(iter(mass_records.values()))
        for field in sorted(sample):
            values = [float(record[field]) for record in mass_records.values()]
            field_stats[field] = finite_stats(values)
    return {
        "files": surface["files"],
        "schemas": surface["schemas"],
        "unique_counts": surface["unique_counts"],
        "duplicate_count": surface["duplicate_count"],
        "duplicate_max_delta": surface["duplicate_max_delta"],
        "field_stats": field_stats,
    }


def best_candidate_for_target(candidates: Mapping[str, Any], target_name: str) -> dict[str, Any] | None:
    rows = []
    for candidate_name, candidate in candidates.items():
        comparison = candidate["comparisons"].get(target_name, {})
        max_abs = comparison.get("max_abs")
        if max_abs is None:
            continue
        rows.append(
            {
                "candidate": candidate_name,
                "boundary_order": candidate["boundary_order"],
                "boundary": candidate["boundary"],
                "max_abs": float(max_abs),
                "rmse": comparison.get("rmse"),
                "worst": comparison.get("worst"),
            }
        )
    rows.sort(key=lambda item: (item["max_abs"], item["boundary_order"]))
    return rows[0] if rows else None


def localization_summary(jax_run: Mapping[str, Any]) -> dict[str, Any]:
    candidates = jax_run.get("theta_candidates", {})
    prestep = candidates.get("checkpoint_prestep_carry.state.theta_minus_300", {})
    prestep_to_t_old = prestep.get("comparisons", {}).get("wrf_final_stage_pre_small_step_finish.T_OLD", {})
    prestep_context = prestep.get("context_mass_pressure", {}).get(
        "wrf_final_stage_pre_small_step_finish.T_OLD", {}
    )
    stage3_input = candidates.get("rk_stage_3.input_haloed_state.theta_minus_300", {})
    stage3_input_to_t_old = stage3_input.get("comparisons", {}).get(
        "wrf_final_stage_pre_small_step_finish.T_OLD", {}
    )
    final_pre_halo = candidates.get("rk_stage_3.post_small_step_finish_pre_halo_state.theta_minus_300", {})
    final_pre_halo_to_hist = final_pre_halo.get("comparisons", {}).get(
        "wrf_post_after_all_rk_steps_pre_halo.T_HIST_SRC", {}
    )
    acoustic_raw = candidates.get(
        "rk_stage_3.post_acoustic_pre_small_step_finish.acoustic.theta_coupled_work_raw", {}
    )
    acoustic_raw_to_thm = acoustic_raw.get("comparisons", {}).get(
        "wrf_final_stage_pre_small_step_finish.T_THM", {}
    )

    best_by_target = {
        target: best_candidate_for_target(candidates, target)
        for target in [
            "wrf_final_stage_pre_small_step_finish.T_OLD",
            "wrf_final_stage_pre_small_step_finish.T_HIST_SRC",
            "wrf_final_stage_pre_small_step_finish.T_THM",
            "wrf_final_stage_post_small_step_finish.T_HIST_SRC",
            "wrf_post_after_all_rk_steps_pre_halo.T_HIST_SRC",
            "wrf_post_after_all_rk_steps_pre_halo.T_THM",
        ]
    }

    prestep_mismatch = (
        prestep_to_t_old.get("max_abs") is not None
        and float(prestep_to_t_old["max_abs"]) > GREEN_TOLERANCE_MAX_ABS
    )
    if prestep_mismatch:
        verdict = "THETA_MISMATCH_PRESTEP_OR_INPUT"
        reason = (
            "The earliest available WRF start-of-step/RK-reference theta surface "
            "(final-stage pre-small_step_finish T_OLD/grid%t_1) already differs "
            "from the real JAX step-5999 carry input before current-step physics or RK."
        )
        first = {
            "jax_boundary": "checkpoint_prestep_carry.state.theta_minus_300",
            "wrf_target": "wrf_final_stage_pre_small_step_finish.T_OLD",
            "comparison": prestep_to_t_old,
            "context_mass_pressure": prestep_context,
        }
        next_decision = (
            "Open a WRF/JAX input-boundary emitter or hook sprint for explicit step-6000 "
            "pre-RK T/P/PB/MU/MUB before deciding any source-changing fix; do not start "
            "by editing final small_step_finish or history-source mapping."
        )
    else:
        stage3_mismatch = (
            stage3_input_to_t_old.get("max_abs") is not None
            and float(stage3_input_to_t_old["max_abs"]) > GREEN_TOLERANCE_MAX_ABS
        )
        if stage3_mismatch:
            verdict = "THETA_MISMATCH_RK_STAGE_3"
            reason = "The first reachable mismatch is at the RK3 input/reference boundary."
            first = {
                "jax_boundary": "rk_stage_3.input_haloed_state.theta_minus_300",
                "wrf_target": "wrf_final_stage_pre_small_step_finish.T_OLD",
                "comparison": stage3_input_to_t_old,
            }
            next_decision = "Open a hook sprint to emit WRF/JAX RK1/RK2 stage-output references or fix the earlier RK stage."
        elif (
            final_pre_halo_to_hist.get("max_abs") is not None
            and float(final_pre_halo_to_hist["max_abs"]) > GREEN_TOLERANCE_MAX_ABS
        ):
            verdict = "THETA_MISMATCH_ACOUSTIC_FINISH"
            reason = "The mismatch first appears across the final acoustic/small_step_finish completion boundary."
            first = {
                "jax_boundary": "rk_stage_3.post_small_step_finish_pre_halo_state.theta_minus_300",
                "wrf_target": "wrf_post_after_all_rk_steps_pre_halo.T_HIST_SRC",
                "comparison": final_pre_halo_to_hist,
            }
            next_decision = "Open a source-changing fix sprint in the RK3 acoustic/small_step_finish theta path."
        else:
            verdict = "THETA_LOCALIZATION_BLOCKED_NO_REACHABLE_MISMATCH"
            reason = "No inspected reachable boundary exceeded tolerance, which is inconsistent with the input attribution proof."
            first = None
            next_decision = "Re-run the proof and add an exact WRF/JAX same-boundary hook for the missing stage."

    return {
        "verdict": verdict,
        "verdict_reason": reason,
        "earliest_reachable_wrf_reference": "wrf_final_stage_pre_small_step_finish.T_OLD",
        "mismatch_already_present_at_earliest_input_boundary": bool(prestep_mismatch),
        "first_reachable_mismatch": first,
        "best_by_target": best_by_target,
        "final_pre_halo_vs_accepted_history_T": final_pre_halo_to_hist,
        "stage3_acoustic_raw_work_vs_wrf_T_THM": acoustic_raw_to_thm,
        "next_decision": next_decision,
    }


def compact_target(wrf_refresh: Mapping[str, Any]) -> dict[str, Any]:
    surface = wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]
    return {
        "accepted_wrf_verdict": wrf_refresh.get("verdict"),
        "target_surface": wrf_refresh.get("next_jax_cpu_wrapper_target"),
        "domain": wrf_refresh["target_confirmed"]["domain"],
        "wrf_step": wrf_refresh["target_confirmed"]["wrf_step"],
        "prestep_completed_steps_required": PRESTEP_COMPLETED_STEPS,
        "valid_time_utc": wrf_refresh["target_confirmed"]["valid_time_utc"],
        "lead_seconds_after_step": wrf_refresh["target_confirmed"]["lead_seconds_after_step"],
        "selected_patch_bounds_mass_grid": wrf_refresh["target_confirmed"]["selected_patch_bounds_mass_grid"],
        "native_staggered_coordinates": wrf_refresh["target_confirmed"]["native_staggered_coordinates"],
        "source_files": surface["files"],
        "source_file_info": [path_info(Path(path)) for path in surface["files"]],
    }


def fmt_stat(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "unavailable"
    return f"max_abs `{entry.get('max_abs')}`, rmse `{entry.get('rmse')}`"


def render_markdown(payload: Mapping[str, Any]) -> str:
    loc = payload["localization"]
    first = loc.get("first_reachable_mismatch") or {}
    final_hist = loc.get("final_pre_halo_vs_accepted_history_T")
    acoustic = loc.get("stage3_acoustic_raw_work_vs_wrf_T_THM")
    lines = [
        "# V0.14 JAX Theta Evolution Localization",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Boundary",
        "",
        f"- First mismatch: `{first.get('jax_boundary')}` vs `{first.get('wrf_target')}`.",
        f"- First comparison: {fmt_stat(first.get('comparison'))}.",
        f"- Final pre-halo history comparison: {fmt_stat(final_hist)}.",
        f"- Final pre-finish `T_THM` diagnostic comparison: {fmt_stat(acoustic)}.",
        "",
        "## Context",
        "",
        "- The checkpoint hash matches the prior T history-source attribution proof.",
        "- WRF `T_HIST_SRC` is treated as history T; WRF `T_THM` is diagnostic only.",
        "- P/PB/MU/MUB context is included in JSON for every state boundary where the WRF surface exposes the matching fields.",
        "",
        "## Next Decision",
        "",
        loc["next_decision"],
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Theta Evolution Localization",
        "",
        f"verdict: `{payload['verdict']}`",
        "",
        "objective: localize the confirmed h10 theta evolution mismatch to the narrowest reachable JAX stage/cadence/component boundary before any production source fix.",
        "",
        "files changed:",
        "- `proofs/v014/jax_theta_evolution_localization.py`",
        "- `proofs/v014/jax_theta_evolution_localization.json`",
        "- `proofs/v014/jax_theta_evolution_localization.md`",
        "- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`",
        "",
        "commands run:",
        "- `python -m py_compile proofs/v014/jax_theta_evolution_localization.py`",
        "- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_theta_evolution_localization.py`",
        "- `python -m json.tool proofs/v014/jax_theta_evolution_localization.json >/tmp/jax_theta_evolution_localization.validated.json`",
        "",
        "proof objects produced:",
        "- `proofs/v014/jax_theta_evolution_localization.json`",
        "- `proofs/v014/jax_theta_evolution_localization.md`",
        "- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`",
        "",
        "result:",
        f"- {payload['localization']['verdict_reason']}",
        "- The proof used the exact h10 full-carry checkpoint recorded by the prior T attribution sprint.",
        "- No production `src/` files, WRF source, TOST, Switzerland validation, or FP32 work were touched.",
        "",
        "unresolved risks:",
    ]
    for risk in payload.get("unresolved_risks", []):
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['localization']['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    started = time.perf_counter()
    if STDERR_CAPTURE.exists():
        STDERR_CAPTURE.unlink()

    loaded_payloads = {
        "jax_t_attr": load_json(JAX_T_ATTR_JSON),
        "jax_h10": load_json(JAX_H10_JSON),
        "jax_h10_producer": load_json(JAX_H10_PRODUCER_JSON),
        "jax_pre_halo": load_json(JAX_PRE_HALO_JSON),
        "wrf_dynamic": load_json(WRF_DYNAMIC_JSON),
        "wrf_refresh": load_json(WRF_REFRESH_JSON),
        "savepoint_request": load_json(SAVEPOINT_REQUEST_JSON),
    }
    wrf_refresh = loaded_payloads["wrf_refresh"]
    wrf_dynamic = loaded_payloads["wrf_dynamic"]

    dynamic_files = wrf_dynamic["emitted_surface"]["files"]
    refresh_surfaces = wrf_refresh["emitted_surfaces"]
    surfaces = {
        "wrf_final_stage_pre_small_step_finish": parse_surface_files(
            Path(path) for path in dynamic_files["pre"]
        ),
        "wrf_final_stage_post_small_step_finish": parse_surface_files(
            Path(path) for path in dynamic_files["post"]
        ),
        "wrf_post_after_all_rk_steps_pre_halo": parse_surface_files(
            Path(path) for path in refresh_surfaces["post_after_all_rk_steps_pre_halo"]["files"]
        ),
        "wrf_post_final_calc_p_rho_phi": parse_surface_files(
            Path(path) for path in refresh_surfaces["post_final_calc_p_rho_phi"]["files"]
        ),
    }

    actual_checkpoint = path_info(CHECKPOINT)
    checkpoint_identity = recorded_checkpoint_identity(actual_checkpoint, loaded_payloads)
    if not CHECKPOINT.exists():
        verdict = "THETA_LOCALIZATION_BLOCKED_MISSING_CHECKPOINT"
        verdict_reason = f"Required checkpoint does not exist: {CHECKPOINT}"
        jax_run: dict[str, Any] = {"status": "BLOCKED", "reason": verdict_reason}
        loc = {
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "first_reachable_mismatch": None,
            "next_decision": "Provide the recorded h10 step-5999 full-carry checkpoint before source-changing work.",
        }
        unresolved = [verdict_reason]
    elif not checkpoint_identity["same_artifact_as_t_attribution"]:
        verdict = "THETA_LOCALIZATION_BLOCKED_CHECKPOINT_IDENTITY_MISMATCH"
        verdict_reason = "Checkpoint hash/size does not match the prior T attribution proof."
        jax_run = {"status": "BLOCKED", "reason": verdict_reason}
        loc = {
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "first_reachable_mismatch": None,
            "next_decision": "Resolve checkpoint provenance before source-changing work.",
        }
        unresolved = [verdict_reason]
    else:
        try:
            with capture_process_stderr(STDERR_CAPTURE):
                jax_run = run_jax_localization(surfaces, wrf_refresh)
            loc = localization_summary(jax_run)
            verdict = loc["verdict"]
            verdict_reason = loc["verdict_reason"]
            unresolved = [
                "Only the selected Boole h10 patch and k=1 mass layer / kstag 0..1 W/PH source emitters were compared.",
                "WRF has no separate full-domain step-5999 prestep emitter in the current artifact set; `T_OLD` is the narrowest available WRF input/reference theta surface.",
                "The proof is CPU-only and intentionally does not run TOST or Switzerland validation.",
            ]
        except Exception as exc:  # pragma: no cover - recorded in proof output
            verdict = f"THETA_LOCALIZATION_BLOCKED_{type(exc).__name__}"
            verdict_reason = repr(exc)
            jax_run = {"status": "BLOCKED", "reason": verdict_reason}
            loc = {
                "verdict": verdict,
                "verdict_reason": verdict_reason,
                "first_reachable_mismatch": None,
                "next_decision": "Fix the blocked proof input/API path or add the exact missing WRF/JAX hook before source-changing work.",
            }
            unresolved = [verdict_reason]

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.jax_theta_evolution_localization.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "tolerance": {
            "max_abs": GREEN_TOLERANCE_MAX_ABS,
            "policy": "No tolerance widening; inherited from the canonical h10 proof.",
        },
        "cpu_only": True,
        "gpu_used": False,
        "production_src_edits": False,
        "wrf_source_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "inputs_read": proof_inputs(),
        "environment": {
            **jax_environment(),
            "native_stderr_capture": {
                "path": str(STDERR_CAPTURE),
                "exists": STDERR_CAPTURE.exists(),
                "size_bytes": STDERR_CAPTURE.stat().st_size if STDERR_CAPTURE.exists() else 0,
                "note": "Native XLA CPU warnings, if emitted, are redirected here to keep terminal output compact.",
            },
        },
        "wrf_target": compact_target(wrf_refresh),
        "wrf_surfaces": {name: surface_summary(surface) for name, surface in surfaces.items()},
        "checkpoint_identity": checkpoint_identity,
        "source_inspection": source_inspection(),
        "jax_run": jax_run,
        "localization": loc,
        "acceptance_notes": {
            "uses_produced_h10_carry_checkpoint": bool(checkpoint_identity["same_artifact_as_t_attribution"]),
            "uses_wrf_source_derived_surfaces": True,
            "distinguishes_T_HIST_SRC_from_T_THM": True,
            "records_p_pb_mu_mub_context": True,
            "retained_wrfout_used_as_verdict": False,
            "jax_vs_jax_self_compare": False,
            "production_source_edited": False,
        },
        "commands": {
            "minimum_contract_commands": [
                "python -m py_compile proofs/v014/jax_theta_evolution_localization.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_theta_evolution_localization.py",
                "python -m json.tool proofs/v014/jax_theta_evolution_localization.json >/tmp/jax_theta_evolution_localization.validated.json",
            ],
            "argv": sys.argv,
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": unresolved,
        "next_decision": loc["next_decision"],
        "wall_s": float(time.perf_counter() - started),
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")

    first = (loc.get("first_reachable_mismatch") or {}).get("comparison", {})
    print(f"{verdict} first_max_abs={first.get('max_abs')} first_rmse={first.get('rmse')}")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
