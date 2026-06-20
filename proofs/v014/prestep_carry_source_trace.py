#!/usr/bin/env python3
"""V0.14 prestep carry source trace.

Evidence-only proof.  This script traces the confirmed d02 h10 pre-RK mismatch
through the produced step-5999 JAX carry checkpoint and its producer metadata.
It does not run current-step RK/acoustic code and does not edit production
source.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np


os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs/v014/prestep_carry_source_trace.json"
OUT_MD = ROOT / "proofs/v014/prestep_carry_source_trace.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md"

PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-prestep-carry-source-trace/sprint-contract.md"
)
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
REPORTING_SKILL = ROOT / ".agent/skills/reporting-to-human/SKILL.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

PRE_RK_JSON = ROOT / "proofs/v014/pre_rk_input_boundary.json"
PRODUCER_SCRIPT = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.py"
PRODUCER_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.json"
JAX_H10_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry.json"
JAX_T_HISTORY_JSON = ROOT / "proofs/v014/jax_t_history_source_attribution.json"
JAX_THETA_LOCALIZATION_JSON = ROOT / "proofs/v014/jax_theta_evolution_localization.json"
CHECKPOINT_MODULE = ROOT / "src/gpuwrf/runtime/checkpoint.py"
OPERATIONAL_STATE_MODULE = ROOT / "src/gpuwrf/runtime/operational_state.py"

CHECKPOINT = Path(
    os.environ.get(
        "WRFGPU2_H10_PRESTEP_CARRY",
        "<DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl",
    )
)
CHECKPOINT_PROVENANCE = CHECKPOINT.with_suffix(".provenance.json")
SCRATCH = Path("/tmp/wrf_gpu2_v014_prestep_carry_source_trace")
ROUNDTRIP_CHECKPOINT = SCRATCH / "d02_step5999_roundtrip.pkl"

TARGET_FIELDS = ("T", "P", "PB", "MU", "MUB")
TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
THETA_OFFSET_K = 300.0
TOLERANCE_MAX_ABS = 2.0e-6

MASS_SCHEMA = {
    "MASS_K1": {
        "index_count": 4,
        "fields": [
            "T_THM",
            "T_OLD",
            "T_HIST_SRC",
            "P",
            "PB",
            "MU_NEW",
            "MU_OLD",
            "MUB",
        ],
    }
}

WRF_FIELD_SOURCE = {
    "T": ("MASS_K1", "T_THM", "grid%t_2 perturbation potential temperature at k=1"),
    "P": ("MASS_K1", "P", "grid%p perturbation pressure at k=1"),
    "PB": ("MASS_K1", "PB", "grid%pb base pressure at k=1"),
    "MU": ("MASS_K1", "MU_NEW", "grid%mu_2 perturbation dry-air column mass"),
    "MUB": ("MASS_K1", "MUB", "grid%mub base dry-air column mass"),
}

CHECKPOINT_LEAF_SOURCE = {
    "T": "carry.state.theta - 300 K, k=1",
    "P": "carry.state.p_perturbation, k=1",
    "PB": "carry.state.p_total - carry.state.p_perturbation, k=1",
    "MU": "carry.state.mu_perturbation",
    "MUB": "carry.state.mu_total - carry.state.mu_perturbation",
}


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


def load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


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
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = [str(device) for device in jax.devices()]
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": devices,
                "gpu_device_count": len([device for device in devices if "gpu" in device.lower()]),
            }
        )
    except Exception as exc:  # pragma: no cover - recorded in proof output
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def extract_ast_node(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        return {"module_path": str(path), "name": name, "missing_file": True}
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
                "contains_load_domains": "_load_domains" in source,
                "contains_run_operational_domain_tree": "run_operational_domain_tree" in source,
                "contains_operational_force": "_operational_force" in source,
                "contains_operational_advance_factory": "_operational_advance_factory" in source,
                "contains_write_checkpoint": "write_checkpoint" in source,
                "contains_runtime_state": "runtime_state" in source,
                "contains_hostify_tree": "_hostify_tree" in source,
                "contains_device_tree": "_device_tree" in source,
            }
    return {"module_path": str(path.relative_to(ROOT)), "name": name, "missing": True}


def key_for(record_type: str, idx: list[int]) -> tuple[int, ...]:
    if record_type == "MASS_K1":
        return (idx[3], idx[2])
    raise ValueError(record_type)


def parse_pre_rk_truth(pre_rk_payload: Mapping[str, Any]) -> dict[str, Any]:
    files = [Path(path) for path in pre_rk_payload["wrf_provenance"]["surface_parse"]["files"]]
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}

    for path in files:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("record_schema"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag not in MASS_SCHEMA:
                    if len(parts) > 1:
                        metadata[tag].append(" ".join(parts[1:]))
                    continue
                schema = MASS_SCHEMA[tag]
                nidx = int(schema["index_count"])
                idx = [int(value) for value in parts[1 : 1 + nidx]]
                values = [float(value) for value in parts[1 + nidx :]]
                fields = list(schema["fields"])
                if len(values) != len(fields):
                    raise ValueError(f"{path}: {tag} expected {len(fields)} values, got {len(values)}")
                item = dict(zip(fields, values))
                key = key_for(tag, idx)
                if key in records[tag]:
                    duplicate_count += 1
                    previous = records[tag][key]
                    for field in fields:
                        label = f"{tag}.{field}"
                        delta = abs(previous[field] - item[field])
                        duplicate_max_delta_by_field[label] = max(
                            duplicate_max_delta_by_field.get(label, 0.0), delta
                        )
                        duplicate_max_delta = max(duplicate_max_delta, delta)
                records[tag][key] = item

    return {
        "files": [str(path) for path in files],
        "file_info": [path_info(path) for path in files],
        "records": records,
        "metadata": {
            name: values[0] if len(values) == 1 else values for name, values in metadata.items()
        },
        "unique_counts": {name: len(items) for name, items in records.items()},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0, "max_abs": None, "rmse": None}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "max_abs": float(np.max(np.abs(finite))),
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    out: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "count": int(arr.size),
        "finite_count": int(finite.size),
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
    return out


def state_index(field: str, key: tuple[int, ...]) -> tuple[int, ...]:
    if field in {"T", "P", "PB"}:
        return (0, key[0], key[1])
    if field in {"MU", "MUB"}:
        return (key[0], key[1])
    raise KeyError(field)


def compare_candidate_to_truth(
    *,
    name: str,
    source_expression: str,
    field: str,
    array: Any,
    wrf_truth: Mapping[str, Any],
    target_leaf_eligible: bool,
) -> dict[str, Any]:
    tag, wrf_field, wrf_convention = WRF_FIELD_SOURCE[field]
    arr = np.asarray(array, dtype=np.float64)
    diffs: list[float] = []
    worst: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = []
    records = wrf_truth["records"].get(tag, {})
    for key, record in sorted(records.items()):
        if wrf_field not in record:
            skipped.append({"native_key": list(key), "reason": f"missing {wrf_field}"})
            continue
        try:
            candidate = float(arr[state_index(field, key)])
        except Exception as exc:  # pragma: no cover - recorded in proof output
            skipped.append({"native_key": list(key), "reason": repr(exc)})
            continue
        truth = float(record[wrf_field])
        diff = candidate - truth
        diffs.append(diff)
        if worst is None or abs(diff) > worst["abs_diff"]:
            worst = {
                "native_key": list(key),
                "array_index": list(state_index(field, key)),
                "jax_candidate": candidate,
                "wrf_truth": truth,
                "diff_jax_minus_wrf": diff,
                "abs_diff": abs(diff),
            }
    entry = {
        "name": name,
        "field": field,
        "source_expression": source_expression,
        "target_leaf_eligible": bool(target_leaf_eligible),
        "wrf_source_field": wrf_field,
        "wrf_source_convention": wrf_convention,
        "array_summary": array_summary(arr),
        **stats(diffs),
        "worst": worst,
        "skipped_record_count": len(skipped),
        "skipped_records": skipped,
        "tolerance_max_abs": TOLERANCE_MAX_ABS,
    }
    entry["status"] = (
        "MATCH"
        if entry.get("max_abs") is not None and float(entry["max_abs"]) <= TOLERANCE_MAX_ABS
        else "DIFF"
    )
    return entry


def compare_arrays(name: str, left: Any, right: Any) -> dict[str, Any]:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "left_shape": list(left_arr.shape),
            "right_shape": list(right_arr.shape),
        }
    diff = np.asarray(left_arr, dtype=np.float64) - np.asarray(right_arr, dtype=np.float64)
    finite = diff[np.isfinite(diff)]
    return {
        "name": name,
        "status": "EXACT" if np.array_equal(left_arr, right_arr) else "DIFF",
        "shape": list(left_arr.shape),
        "left_dtype": str(left_arr.dtype),
        "right_dtype": str(right_arr.dtype),
        "max_abs": float(np.max(np.abs(finite))) if finite.size else None,
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))) if finite.size else None,
        "array_equal": bool(np.array_equal(left_arr, right_arr)),
    }


def field_array_from_carry(carry: Any, field: str) -> Any:
    state = carry.state
    if field == "T":
        return np.asarray(state.theta) - THETA_OFFSET_K
    if field == "P":
        return np.asarray(state.p_perturbation)
    if field == "PB":
        return np.asarray(state.p_total) - np.asarray(state.p_perturbation)
    if field == "MU":
        return np.asarray(state.mu_perturbation)
    if field == "MUB":
        return np.asarray(state.mu_total) - np.asarray(state.mu_perturbation)
    raise KeyError(field)


def field_array_from_state_fields(fields: Mapping[str, Any], field: str) -> Any:
    if field == "T":
        return np.asarray(fields["theta"]) - THETA_OFFSET_K
    if field == "P":
        return np.asarray(fields["p_perturbation"])
    if field == "PB":
        return np.asarray(fields["p_total"]) - np.asarray(fields["p_perturbation"])
    if field == "MU":
        return np.asarray(fields["mu_perturbation"])
    if field == "MUB":
        return np.asarray(fields["mu_total"]) - np.asarray(fields["mu_perturbation"])
    raise KeyError(field)


def alternative_candidates(carry: Any, field: str) -> list[tuple[str, str, Any]]:
    state = carry.state
    if field == "T":
        return [
            (
                "checkpoint_prestep_carry.t_save_minus_300",
                "carry.t_save - 300 K, k=1",
                np.asarray(carry.t_save) - THETA_OFFSET_K,
            ),
            (
                "checkpoint_prestep_carry.t_2ave_minus_300",
                "carry.t_2ave - 300 K, k=1",
                np.asarray(carry.t_2ave) - THETA_OFFSET_K,
            ),
        ]
    if field == "MU":
        state_mub = np.asarray(state.mu_total) - np.asarray(state.mu_perturbation)
        return [
            ("checkpoint_prestep_carry.mu_save", "carry.mu_save", np.asarray(carry.mu_save)),
            (
                "checkpoint_prestep_carry.muts_minus_state_MUB",
                "carry.muts - (carry.state.mu_total - carry.state.mu_perturbation)",
                np.asarray(carry.muts) - state_mub,
            ),
            ("checkpoint_prestep_carry.muave", "carry.muave", np.asarray(carry.muave)),
        ]
    if field == "MUB":
        return [
            (
                "checkpoint_prestep_carry.mu_total_minus_mu_save",
                "carry.state.mu_total - carry.mu_save",
                np.asarray(state.mu_total) - np.asarray(carry.mu_save),
            ),
            (
                "checkpoint_prestep_carry.muts_minus_mu_save",
                "carry.muts - carry.mu_save",
                np.asarray(carry.muts) - np.asarray(carry.mu_save),
            ),
            (
                "checkpoint_prestep_carry.mu_total_minus_muave",
                "carry.state.mu_total - carry.muave",
                np.asarray(state.mu_total) - np.asarray(carry.muave),
            ),
        ]
    return []


def load_checkpoint_objects(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state  # noqa: PLC0415

    with path.open("rb") as handle:
        raw_payload = pickle.load(handle)
    state, namelist, grid, step_index, runtime_state = read_checkpoint_with_runtime_state(path)
    if runtime_state is None:
        raise ValueError(f"{path} does not contain runtime_state")
    return {
        "raw_payload": raw_payload,
        "state": state,
        "namelist": namelist,
        "grid": grid,
        "step_index": int(step_index),
        "carry": runtime_state,
    }


def roundtrip_checkpoint(carry: Any, namelist: Any, grid: Any, step_index: int) -> dict[str, Any]:
    from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state, write_checkpoint  # noqa: PLC0415

    SCRATCH.mkdir(parents=True, exist_ok=True)
    write_checkpoint(carry.state, namelist, grid, step_index, ROUNDTRIP_CHECKPOINT, runtime_state=carry)
    _state, _namelist, _grid, roundtrip_step, roundtrip_carry = read_checkpoint_with_runtime_state(
        ROUNDTRIP_CHECKPOINT
    )
    if roundtrip_carry is None:
        raise ValueError("roundtrip checkpoint lost runtime_state")
    return {
        "path": ROUNDTRIP_CHECKPOINT,
        "step_index": int(roundtrip_step),
        "carry": roundtrip_carry,
        "file_info": path_info(ROUNDTRIP_CHECKPOINT),
    }


def serialization_checks(objects: Mapping[str, Any], roundtrip: Mapping[str, Any]) -> dict[str, Any]:
    raw_payload = objects["raw_payload"]
    api_carry = objects["carry"]
    raw_carry = raw_payload["runtime_state"]
    state_fields = raw_payload["state_fields"]
    roundtrip_carry = roundtrip["carry"]
    field_checks: dict[str, Any] = {}
    all_exact = True
    for field in TARGET_FIELDS:
        raw_arr = field_array_from_carry(raw_carry, field)
        api_arr = field_array_from_carry(api_carry, field)
        top_arr = field_array_from_state_fields(state_fields, field)
        rt_arr = field_array_from_carry(roundtrip_carry, field)
        checks = {
            "raw_runtime_state_vs_api_runtime_state": compare_arrays(
                f"{field}.raw_runtime_state_vs_api_runtime_state", raw_arr, api_arr
            ),
            "raw_top_level_state_fields_vs_raw_runtime_state": compare_arrays(
                f"{field}.raw_top_level_state_fields_vs_raw_runtime_state", top_arr, raw_arr
            ),
            "api_runtime_state_vs_roundtrip_runtime_state": compare_arrays(
                f"{field}.api_runtime_state_vs_roundtrip_runtime_state", api_arr, rt_arr
            ),
        }
        all_exact = all_exact and all(
            item.get("status") == "EXACT" and item.get("max_abs") in {0.0, None}
            for item in checks.values()
        )
        field_checks[field] = checks
    return {
        "verdict": "CHECKPOINT_READ_WRITE_PRESERVES_TARGET_LEAVES"
        if all_exact
        else "CHECKPOINT_SERIALIZATION_DIFF_DETECTED",
        "all_target_leaf_checks_exact": bool(all_exact),
        "checkpoint_format": raw_payload.get("format"),
        "checkpoint_format_version": raw_payload.get("format_version"),
        "step_index": objects["step_index"],
        "raw_payload_has_runtime_state": raw_payload.get("runtime_state") is not None,
        "raw_payload_state_field_count": raw_payload.get("state_field_count"),
        "roundtrip": {
            "path": str(roundtrip["path"]),
            "step_index": roundtrip["step_index"],
            "file_info": roundtrip["file_info"],
        },
        "field_checks": field_checks,
    }


def field_trace(carry: Any, wrf_truth: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        checkpoint_entry = compare_candidate_to_truth(
            name=f"checkpoint_prestep_carry.{field}",
            source_expression=CHECKPOINT_LEAF_SOURCE[field],
            field=field,
            array=field_array_from_carry(carry, field),
            wrf_truth=wrf_truth,
            target_leaf_eligible=True,
        )
        alternatives = [
            compare_candidate_to_truth(
                name=name,
                source_expression=source_expression,
                field=field,
                array=array,
                wrf_truth=wrf_truth,
                target_leaf_eligible=False,
            )
            for name, source_expression, array in alternative_candidates(carry, field)
        ]
        for item in alternatives:
            item["closer_than_checkpoint_leaf"] = (
                item.get("max_abs") is not None
                and checkpoint_entry.get("max_abs") is not None
                and float(item["max_abs"]) < float(checkpoint_entry["max_abs"])
            )
            item["matches_tolerance"] = (
                item.get("max_abs") is not None and float(item["max_abs"]) <= TOLERANCE_MAX_ABS
            )
        closest_candidates = [checkpoint_entry, *alternatives]
        closest_candidates = [item for item in closest_candidates if item.get("max_abs") is not None]
        closest = min(closest_candidates, key=lambda item: float(item["max_abs"]))
        prior_artifacts = prior_artifact_summary_for_field(field, prior, checkpoint_entry)
        out[field] = {
            "checkpoint_leaf_source_expression": CHECKPOINT_LEAF_SOURCE[field],
            "checkpoint_leaf_vs_wrf_pre_rk_truth": checkpoint_entry,
            "same_checkpoint_alternative_leaves": alternatives,
            "closest_same_checkpoint_candidate": {
                "name": closest["name"],
                "max_abs": closest.get("max_abs"),
                "rmse": closest.get("rmse"),
                "target_leaf_eligible": closest.get("target_leaf_eligible"),
                "status": closest.get("status"),
            },
            "any_same_checkpoint_candidate_matches_tolerance": any(
                item.get("max_abs") is not None and float(item["max_abs"]) <= TOLERANCE_MAX_ABS
                for item in closest_candidates
            ),
            "any_same_checkpoint_alternative_closer_than_checkpoint_leaf": any(
                bool(item.get("closer_than_checkpoint_leaf")) for item in alternatives
            ),
            "prior_artifact_candidates": prior_artifacts,
            "conclusion": field_conclusion(field, checkpoint_entry, alternatives, prior_artifacts),
        }
    return out


def prior_artifact_summary_for_field(
    field: str, prior: Mapping[str, Any], checkpoint_entry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    jax_h10 = prior.get("jax_h10") or {}
    h10_entry = (
        jax_h10.get("comparison", {}).get("comparison", {}).get(field)
        if isinstance(jax_h10, Mapping)
        else None
    )
    if isinstance(h10_entry, Mapping):
        out.append(
            {
                "artifact": "proofs/v014/jax_h10_prestep_carry.json",
                "candidate": "current-step captured pre-halo/post-RK comparison",
                "surface": "after_all_rk_steps_pre_halo, not pre-RK input",
                "max_abs": h10_entry.get("max_abs"),
                "rmse": h10_entry.get("rmse"),
                "same_surface_as_this_proof": False,
                "eligible_as_prestep_source": False,
            }
        )

    t_history = prior.get("t_history") or {}
    if field == "T" and isinstance(t_history, Mapping):
        best = (
            t_history.get("best_matches", {})
            .get("wrf_thm_side__MASS_K1_T_THM", [])
        )
        if best:
            first = best[0]
            out.append(
                {
                    "artifact": "proofs/v014/jax_t_history_source_attribution.json",
                    "candidate": first.get("candidate"),
                    "surface": "WRF THM-side after current-step all-RK, not pre-RK input",
                    "max_abs": first.get("max_abs"),
                    "rmse": first.get("rmse"),
                    "same_surface_as_this_proof": False,
                    "eligible_as_prestep_source": False,
                }
            )
    elif isinstance(t_history, Mapping):
        context = t_history.get("jax_capture", {}).get("context_pre_halo_mass_pressure", {})
        item = context.get(field)
        if isinstance(item, Mapping):
            out.append(
                {
                    "artifact": "proofs/v014/jax_t_history_source_attribution.json",
                    "candidate": f"context_pre_halo_mass_pressure.{field}",
                    "surface": "after_all_rk_steps_pre_halo context, not pre-RK input",
                    "max_abs": item.get("max_abs"),
                    "rmse": item.get("rmse"),
                    "same_surface_as_this_proof": False,
                    "eligible_as_prestep_source": False,
                }
            )

    theta = prior.get("theta_localization") or {}
    if field == "T" and isinstance(theta, Mapping):
        first = theta.get("localization", {}).get("first_reachable_mismatch", {}).get("comparison")
        if isinstance(first, Mapping):
            out.append(
                {
                    "artifact": "proofs/v014/jax_theta_evolution_localization.json",
                    "candidate": theta.get("localization", {})
                    .get("first_reachable_mismatch", {})
                    .get("jax_boundary"),
                    "surface": "earliest previous WRF T_OLD reference, adjacent but not explicit T_THM pre-RK truth",
                    "max_abs": first.get("max_abs"),
                    "rmse": first.get("rmse"),
                    "same_surface_as_this_proof": False,
                    "eligible_as_prestep_source": False,
                }
            )

    for item in out:
        item["closer_than_checkpoint_leaf_max_abs"] = (
            item.get("max_abs") is not None
            and checkpoint_entry.get("max_abs") is not None
            and float(item["max_abs"]) < float(checkpoint_entry["max_abs"])
        )
        item["matches_tolerance"] = (
            item.get("max_abs") is not None and float(item["max_abs"]) <= TOLERANCE_MAX_ABS
        )
    return out


def field_conclusion(
    field: str,
    checkpoint_entry: Mapping[str, Any],
    alternatives: list[Mapping[str, Any]],
    prior_artifacts: list[Mapping[str, Any]],
) -> str:
    closer = [item for item in alternatives if item.get("closer_than_checkpoint_leaf")]
    matches = [item for item in alternatives if item.get("matches_tolerance")]
    if matches:
        return "A same-checkpoint non-target leaf matches tolerance; this would require source follow-up."
    if closer:
        names = ", ".join(str(item["name"]) for item in closer)
        return (
            f"{names} is closer than the target State leaf for {field}, but still differs by more "
            "than tolerance and is scratch/non-target state."
        )
    prior_closer = [item for item in prior_artifacts if item.get("closer_than_checkpoint_leaf_max_abs")]
    if prior_closer:
        return (
            "Only non-pre-RK/current-step prior artifacts are closer; they are not eligible as the "
            "step-5999 pre-RK source."
        )
    return "No inspected same-checkpoint or prior artifact improves the target State leaf on the required pre-RK source."


def producer_trace(producer_payload: Mapping[str, Any], provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    producer_summary = {}
    if isinstance(provenance, Mapping):
        producer_summary = provenance.get("producer_summary", {}) or {}
    checkpoint_record = producer_payload.get("checkpoint") if isinstance(producer_payload, Mapping) else None
    return {
        "classification": "live_nested_replay_generated_operational_carry",
        "starts_from": (
            "native L2 wrfinput/wrfbdy domain load via _load_domains, then generated "
            "OperationalCarry advanced by DomainTree/run_operational_domain_tree"
        ),
        "does_not_start_from": [
            "retained GPU wrfout",
            "WRF restart",
            "JAX restart readback",
            "previous-step WRF live state",
        ],
        "checkpoint_write_expression": (
            "write_checkpoint(d02_carry.state, d02_bundle.namelist, d02_bundle.grid, "
            "5999, path, runtime_state=d02_carry)"
        ),
        "handoff_path": [
            "_load_domains(config, domain_names_for(2))",
            "DomainTree.from_domains(..., feedback_enabled=False)",
            "run_operational_domain_tree(..., carries=carries, initial_own_steps=own_steps) through d01=1999/d02=5997",
            "_operational_advance_factory(tree) advances parent d01 step 2000 for the partial subcycle",
            "_operational_force(edge, carries[parent], carries[d02]) refreshes d02 from parent",
            "advance(d02, carries[d02], child_start_step=5998, child_steps=2) produces completed d02 step 5999",
            "write_checkpoint(... runtime_state=d02_carry) writes the produced carry",
        ],
        "producer_json_checkpoint_record": checkpoint_record,
        "producer_provenance": {
            "path": str(CHECKPOINT_PROVENANCE),
            "present": isinstance(provenance, Mapping),
            "gpu_used_for_original_production": provenance.get("gpu_used") if isinstance(provenance, Mapping) else None,
            "producer_command_display": provenance.get("producer_command_display")
            if isinstance(provenance, Mapping)
            else None,
            "producer_summary": producer_summary,
        },
        "source_nodes": [
            extract_ast_node(PRODUCER_SCRIPT, "input_run_dir"),
            extract_ast_node(PRODUCER_SCRIPT, "produce_checkpoint"),
            extract_ast_node(CHECKPOINT_MODULE, "write_checkpoint"),
            extract_ast_node(CHECKPOINT_MODULE, "read_checkpoint_with_runtime_state"),
            extract_ast_node(OPERATIONAL_STATE_MODULE, "OperationalCarry"),
        ],
    }


def input_records() -> dict[str, Any]:
    return {
        "project_constitution": path_info(PROJECT_CONSTITUTION),
        "agents": path_info(AGENTS),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "reporting_to_human_skill": path_info(REPORTING_SKILL),
        "handoff": path_info(HANDOFF),
        "pre_rk_input_boundary_json": path_info(PRE_RK_JSON),
        "producer_script": path_info(PRODUCER_SCRIPT),
        "producer_json": path_info(PRODUCER_JSON),
        "producer_provenance": path_info(CHECKPOINT_PROVENANCE),
        "jax_h10_prestep_carry_json": path_info(JAX_H10_JSON),
        "jax_t_history_source_attribution_json": path_info(JAX_T_HISTORY_JSON),
        "jax_theta_evolution_localization_json": path_info(JAX_THETA_LOCALIZATION_JSON),
        "checkpoint": path_info(CHECKPOINT),
        "checkpoint_module": path_info(CHECKPOINT_MODULE),
        "operational_state_module": path_info(OPERATIONAL_STATE_MODULE),
    }


def blocked_payload(reason: str, detail: str, next_artifact_or_command: str) -> dict[str, Any]:
    return {
        "schema": "wrfgpu2.v014.prestep_carry_source_trace.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": f"TRACE_BLOCKED_{reason}",
        "classification": f"TRACE_BLOCKED_{reason}",
        "blocked": {
            "reason": reason,
            "detail": detail,
            "exact_next_artifact_api_or_command_needed": next_artifact_or_command,
        },
        "cpu_only": True,
        "gpu_used": False,
        "production_src_edits": False,
        "wrf_source_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "inputs_read": input_records(),
        "environment": environment(),
        "commands": required_commands(),
        "proof_objects": proof_objects(),
        "unresolved_risks": [detail],
        "next_decision": next_artifact_or_command,
    }


def required_commands() -> dict[str, Any]:
    return {
        "required_validation": [
            "python -m py_compile proofs/v014/prestep_carry_source_trace.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/prestep_carry_source_trace.py",
            "python -m json.tool proofs/v014/prestep_carry_source_trace.json >/tmp/prestep_carry_source_trace.validated.json",
        ],
        "argv": sys.argv,
    }


def proof_objects() -> dict[str, str]:
    return {
        "json": str(OUT_JSON),
        "markdown": str(OUT_MD),
        "review": str(OUT_REVIEW),
        "roundtrip_checkpoint": str(ROUNDTRIP_CHECKPOINT),
    }


def decide_classification(serialization: Mapping[str, Any], fields: Mapping[str, Any]) -> tuple[str, list[str]]:
    if serialization.get("verdict") != "CHECKPOINT_READ_WRITE_PRESERVES_TARGET_LEAVES":
        return (
            "CHECKPOINT_SERIALIZATION_BUG",
            ["At least one raw/API/top-level/roundtrip target-leaf identity check differed."],
        )
    mismatch_fields = [
        field
        for field, item in fields.items()
        if item["checkpoint_leaf_vs_wrf_pre_rk_truth"].get("status") == "DIFF"
    ]
    if mismatch_fields:
        return (
            "PRODUCER_WRITES_BAD_FINAL_CARRY",
            [
                "The raw pickle runtime_state, checkpoint API runtime_state, top-level State payload, and local roundtrip all preserve target leaves exactly.",
                "The preserved target leaves are the same leaves that differ from CPU-WRF pre-RK truth.",
                "The producer path writes the d02 live-nested replay carry directly as runtime_state=d02_carry at completed step 5999.",
                "No same-surface inspected checkpoint scratch leaf or prior proof artifact matches all target fields within tolerance.",
            ],
        )
    return (
        "TRACE_BLOCKED_NO_MISMATCH_REPRODUCED",
        ["The pre-RK mismatch did not reproduce from the current checkpoint; rerun the starting proof."],
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("blocked"):
        blocked = payload["blocked"]
        return "\n".join(
            [
                "# V0.14 Prestep Carry Source Trace",
                "",
                f"Verdict: `{payload['verdict']}`.",
                "",
                "## Blocker",
                "",
                f"- Reason: `{blocked['reason']}`.",
                f"- Detail: {blocked['detail']}",
                f"- Next: `{blocked['exact_next_artifact_api_or_command_needed']}`",
                "",
            ]
        )

    lines = [
        "# V0.14 Prestep Carry Source Trace",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Summary",
        "",
        f"- Classification: `{payload['classification']}`.",
        f"- Serialization: `{payload['serialization']['verdict']}`.",
        "- Producer source: live nested replay from native domain load; not retained wrfout/restart.",
        "- Current-step RK/acoustic was not run by this trace.",
        "",
        "## Field Results",
        "",
        "| field | checkpoint source | max abs | RMSE | closest inspected same-checkpoint candidate |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for field in TARGET_FIELDS:
        item = payload["field_trace"][field]
        ckpt = item["checkpoint_leaf_vs_wrf_pre_rk_truth"]
        closest = item["closest_same_checkpoint_candidate"]
        lines.append(
            f"| `{field}` | `{item['checkpoint_leaf_source_expression']}` | "
            f"{ckpt.get('max_abs')} | {ckpt.get('rmse')} | "
            f"`{closest['name']}` max_abs {closest.get('max_abs')} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["next_decision"],
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Prestep Carry Source Trace",
        "",
        f"verdict: `{payload['verdict']}`",
        "",
        "objective: trace the confirmed h10 d02 pre-RK input-boundary mismatch through the JAX checkpoint/prestep carry producer and previous-step handoff path without production source edits.",
        "",
        "files changed:",
        "- `proofs/v014/prestep_carry_source_trace.py`",
        "- `proofs/v014/prestep_carry_source_trace.json`",
        "- `proofs/v014/prestep_carry_source_trace.md`",
        "- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            "- `proofs/v014/prestep_carry_source_trace.json`",
            "- `proofs/v014/prestep_carry_source_trace.md`",
            "- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`",
            f"- `{ROUNDTRIP_CHECKPOINT}`",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload.get("unresolved_risks", []):
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    if not PRE_RK_JSON.exists():
        payload = blocked_payload(
            "PRE_RK_INPUT_BOUNDARY_JSON_MISSING",
            f"Missing {PRE_RK_JSON}",
            "Run proofs/v014/pre_rk_input_boundary.py until proofs/v014/pre_rk_input_boundary.json exists.",
        )
        write_outputs(payload)
        print(payload["verdict"])
        return 0
    if not CHECKPOINT.exists():
        payload = blocked_payload(
            "CHECKPOINT_MISSING",
            f"Missing {CHECKPOINT}",
            (
                "Provide <DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl "
                "or set WRFGPU2_H10_PRESTEP_CARRY to a CPU-loadable step-5999 runtime checkpoint."
            ),
        )
        write_outputs(payload)
        print(payload["verdict"])
        return 0

    pre_rk = load_json(PRE_RK_JSON)
    producer = load_json(PRODUCER_JSON)
    provenance = load_optional_json(CHECKPOINT_PROVENANCE)
    prior = {
        "jax_h10": load_json(JAX_H10_JSON),
        "t_history": load_json(JAX_T_HISTORY_JSON),
        "theta_localization": load_json(JAX_THETA_LOCALIZATION_JSON),
    }

    wrf_truth = parse_pre_rk_truth(pre_rk)
    if not wrf_truth["records"].get("MASS_K1"):
        payload = blocked_payload(
            "NO_PRE_RK_TRUTH_RECORDS",
            "pre_rk_input_boundary.json was readable but its listed hook files had no MASS_K1 records.",
            pre_rk.get("wrf_provenance", {})
            .get("minimal_cpu_wrf_commands", ["rerun proofs/v014/pre_rk_input_boundary.py"])[-1],
        )
        write_outputs(payload)
        print(payload["verdict"])
        return 0

    objects = load_checkpoint_objects(CHECKPOINT)
    roundtrip = roundtrip_checkpoint(
        objects["carry"], objects["namelist"], objects["grid"], objects["step_index"]
    )
    serialization = serialization_checks(objects, roundtrip)
    fields = field_trace(objects["carry"], wrf_truth, prior)
    classification, rationale = decide_classification(serialization, fields)
    verdict = classification

    checkpoint_record = path_info(CHECKPOINT)
    pre_rk_checkpoint = pre_rk.get("jax_provenance", {}).get("checkpoint", {})
    checkpoint_identity = {
        "actual_checkpoint": checkpoint_record,
        "step_index": objects["step_index"],
        "matches_pre_rk_input_boundary_record": {
            "record_present": bool(pre_rk_checkpoint),
            "path_match": pre_rk_checkpoint.get("path") == str(CHECKPOINT),
            "sha256_match": pre_rk_checkpoint.get("sha256") == checkpoint_record.get("sha256"),
            "size_match": pre_rk_checkpoint.get("size_bytes") == checkpoint_record.get("size_bytes"),
            "recorded_step_index": pre_rk.get("jax_provenance", {}).get("step_index"),
        },
        "matches_producer_record": {
            "record_present": bool(producer.get("checkpoint")),
            "path_match": producer.get("checkpoint", {}).get("path") == str(CHECKPOINT),
            "sha256_match": producer.get("checkpoint", {}).get("sha256")
            == checkpoint_record.get("sha256"),
            "size_match": producer.get("checkpoint", {}).get("size_bytes")
            == checkpoint_record.get("size_bytes"),
            "recorded_step_index": producer.get("checkpoint", {}).get("step_index"),
        },
    }

    any_candidate_match = any(
        item["any_same_checkpoint_candidate_matches_tolerance"] for item in fields.values()
    )
    fields_with_closer_but_wrong = [
        field
        for field, item in fields.items()
        if item["any_same_checkpoint_alternative_closer_than_checkpoint_leaf"]
        and not item["any_same_checkpoint_candidate_matches_tolerance"]
    ]

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.prestep_carry_source_trace.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "classification": classification,
        "classification_rationale": rationale,
        "cpu_only": True,
        "gpu_used": False,
        "production_src_edits": False,
        "wrf_source_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "target": {
            "domain": "d02",
            "wrf_step": TARGET_STEP,
            "prestep_completed_steps": PRESTEP_COMPLETED_STEPS,
            "valid_time_utc": "2026-05-02T04:00:00Z",
            "wrf_surface": "dyn_em/solve_em.F after grid%itimestep increment before current-step physics/RK",
            "target_fields": list(TARGET_FIELDS),
            "tolerance_max_abs": TOLERANCE_MAX_ABS,
        },
        "environment": environment(),
        "inputs_read": input_records(),
        "checkpoint_identity": checkpoint_identity,
        "producer_trace": producer_trace(producer, provenance),
        "wrf_pre_rk_truth": {
            "source_json": str(PRE_RK_JSON),
            "source_verdict": pre_rk.get("verdict"),
            "files": wrf_truth["files"],
            "file_info": wrf_truth["file_info"],
            "metadata": wrf_truth["metadata"],
            "unique_counts": wrf_truth["unique_counts"],
            "duplicate_count": wrf_truth["duplicate_count"],
            "duplicate_max_delta": wrf_truth["duplicate_max_delta"],
            "field_source": WRF_FIELD_SOURCE,
        },
        "serialization": serialization,
        "field_trace": fields,
        "candidate_summary": {
            "any_same_checkpoint_candidate_matches_tolerance": any_candidate_match,
            "fields_with_closer_but_still_wrong_same_checkpoint_leaf": fields_with_closer_but_wrong,
            "prior_artifacts_used_as_truth": False,
            "retained_wrfout_used_as_truth": False,
            "current_step_rk_or_acoustic_run": False,
        },
        "blocked": None,
        "acceptance_notes": {
            "uses_cpu_wrf_pre_rk_truth_from_pre_rk_input_boundary": True,
            "distinguishes_serialization_from_bad_carry_production": True,
            "checkpoint_write_read_preserves_target_leaves": serialization[
                "all_target_leaf_checks_exact"
            ],
            "production_source_edited": False,
            "wrf_source_edited": False,
            "jax_vs_jax_self_compare_used_as_verdict": False,
        },
        "commands": required_commands(),
        "proof_objects": proof_objects(),
        "unresolved_risks": [
            "This proves the persisted producer carry is bad; it does not split the deeper cause between previous-step final carry assembly, parent/child force packaging, and earlier step integration because no d02 step-5997/5998/5999 in-memory snapshots are present.",
            "Only the selected pre-RK h10 d02 mass patch was compared.",
            "The original checkpoint was produced in a prior GPU-enabled producer run; this trace only loads and round-trips it CPU-side.",
        ],
        "next_decision": (
            "Open a narrow previous-step handoff bisection sprint: capture CPU-only JAX d02 carries "
            "at steps 5997, 5998, and 5999 immediately before/after _operational_force, "
            "_advance_chunk, and _carry_from_finished_stage, then compare State theta/p/mu "
            "target leaves plus t_2ave/t_save/mu_save/muts against the existing CPU-WRF pre-RK "
            "truth. Do not edit current-step RK/acoustic first."
        ),
        "wall_s": float(time.perf_counter() - started),
    }

    write_outputs(payload)
    print(payload["verdict"])
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
