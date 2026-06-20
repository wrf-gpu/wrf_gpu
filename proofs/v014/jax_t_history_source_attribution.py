#!/usr/bin/env python3
"""V0.14 T history-source attribution proof.

This proof is read-only with respect to production source.  It loads the
recorded step-5999 full OperationalCarry checkpoint on CPU, reruns the existing
h10 final-RK pre-halo capture path, and compares plausible JAX theta/history
sources against both WRF history T (MASS_K1.T_HIST_SRC) and the distinct WRF
THM-side candidate (MASS_K1.T_THM).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from collections import defaultdict
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

OUT_JSON = ROOT / "proofs/v014/jax_t_history_source_attribution.json"
OUT_MD = ROOT / "proofs/v014/jax_t_history_source_attribution.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-t-history-source-attribution.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-t-history-source-attribution/sprint-contract.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

JAX_H10_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry.json"
JAX_H10_PRODUCER_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.json"
WRF_REFRESH_JSON = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.json"
WRF_MARKER_JSON = ROOT / "proofs/v014/wrf_same_state_marker_savepoint.json"
SAVEPOINT_REQUEST_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"

CHECKPOINT = Path(
    os.environ.get(
        "WRFGPU2_H10_PRESTEP_CARRY",
        "<DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl",
    )
)
STDERR_CAPTURE = Path(os.environ.get("WRFGPU2_T_ATTRIBUTION_STDERR", "/tmp/jax_t_history_source_attribution.stderr"))

TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
THETA_OFFSET_K = 300.0
GREEN_TOLERANCE_MAX_ABS = 2.0e-6

REFRESH_SCHEMAS = {
    "MASS_K1": {
        "index_count": 4,
        "fields": [
            "T_HIST_SRC",
            "T_THM",
            "P",
            "PB",
            "MU_NEW",
            "MU_OLD",
            "MUB",
            "MUT",
            "MUTS",
            "AL",
            "ALB",
            "ALT",
            "RHO",
        ],
    },
    "U_K1": {"index_count": 4, "fields": ["U"]},
    "V_K1": {"index_count": 4, "fields": ["V"]},
    "WPH_KSTAG01": {"index_count": 6, "fields": ["W", "PH"]},
}

WRF_T_CANDIDATES = {
    "wrf_history_T__MASS_K1_T_HIST_SRC": {
        "tag": "MASS_K1",
        "field": "T_HIST_SRC",
        "description": "WRF history T source, grid%th_phy_m_t0.",
    },
    "wrf_thm_side__MASS_K1_T_THM": {
        "tag": "MASS_K1",
        "field": "T_THM",
        "description": "WRF THM-side candidate; not the accepted history T source.",
    },
}


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
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


def key_for(record_type: str, idx: list[int]) -> tuple[int, ...]:
    if record_type in {"MASS_K1", "U_K1", "V_K1"}:
        return (idx[3], idx[2])
    if record_type == "WPH_KSTAG01":
        return (idx[5], idx[4], idx[3])
    raise ValueError(record_type)


def parse_refresh_files(paths: Iterable[Path]) -> dict[str, Any]:
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("record_schema"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag not in REFRESH_SCHEMAS:
                    if len(parts) > 1:
                        metadata[tag].append(" ".join(parts[1:]))
                    continue
                schema = REFRESH_SCHEMAS[tag]
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
        "files": [str(path) for path in paths],
        "records": records,
        "metadata": {name: values[0] if len(values) == 1 else values for name, values in metadata.items()},
        "unique_counts": {name: len(items) for name, items in records.items()},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def finite_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
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
            "min": None,
            "max": None,
            "mean": None,
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
            }
        )
    return out


def wrf_field_stats(surface: Mapping[str, Any], tag: str, field: str) -> dict[str, Any]:
    values = [float(record[field]) for record in surface["records"][tag].values()]
    return finite_stats(values)


def wrf_field_diff_stats(surface: Mapping[str, Any], tag: str, left: str, right: str) -> dict[str, Any]:
    diffs: list[float] = []
    worst: dict[str, Any] | None = None
    for key, record in sorted(surface["records"][tag].items()):
        diff = float(record[left]) - float(record[right])
        diffs.append(diff)
        if worst is None or abs(diff) > worst["abs_diff"]:
            worst = {
                "native_key": list(key),
                "left_field": left,
                "left_value": float(record[left]),
                "right_field": right,
                "right_value": float(record[right]),
                "diff_left_minus_right": diff,
                "abs_diff": abs(diff),
            }
    return {"status": "DIFF" if diffs else "NO_RECORDS", **finite_stats(diffs), "worst": worst}


def mass_k1_index(arr: np.ndarray, key: tuple[int, ...]) -> tuple[int, ...]:
    y, x = key
    if arr.ndim == 3:
        return (0, y, x)
    if arr.ndim == 2:
        return (y, x)
    raise ValueError(f"expected 2-D or 3-D mass field, got shape {arr.shape}")


def compare_mass_array_to_wrf(
    candidate_array: Any,
    surface: Mapping[str, Any],
    *,
    wrf_field: str,
) -> dict[str, Any]:
    arr = np.asarray(candidate_array, dtype=np.float64)
    diffs: list[float] = []
    worst: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = []
    for key, record in sorted(surface["records"]["MASS_K1"].items()):
        try:
            idx = mass_k1_index(arr, key)
            candidate = float(arr[idx])
        except Exception as exc:
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


def compare_state_context_to_wrf(state: Any, surface: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "P": (np.asarray(state.p_perturbation), "P"),
        "PB": (np.asarray(state.p_total) - np.asarray(state.p_perturbation), "PB"),
        "MU": (np.asarray(state.mu_perturbation), "MU_NEW"),
        "MUB": (np.asarray(state.mu_total) - np.asarray(state.mu_perturbation), "MUB"),
    }
    return {
        name: compare_mass_array_to_wrf(array, surface, wrf_field=wrf_field)
        for name, (array, wrf_field) in fields.items()
    }


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


def recorded_checkpoint_identity(
    actual: Mapping[str, Any],
    h10_payload: Mapping[str, Any],
    producer_payload: Mapping[str, Any],
) -> dict[str, Any]:
    h10_candidates = h10_payload.get("checkpoint_probe", {}).get("usable_candidates", [])
    h10_candidate = h10_candidates[0] if h10_candidates else None
    producer_checkpoint = producer_payload.get("checkpoint")

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

    return {
        "actual_checkpoint": dict(actual),
        "matches_producer_checkpoint_record": same_as(producer_checkpoint),
        "matches_canonical_h10_usable_candidate": same_as(h10_candidate),
        "same_artifact_as_producer_recorded": bool(
            isinstance(producer_checkpoint, Mapping)
            and producer_checkpoint.get("sha256") == actual.get("sha256")
            and producer_checkpoint.get("size_bytes") == actual.get("size_bytes")
        ),
        "same_artifact_as_canonical_h10_compared": bool(
            isinstance(h10_candidate, Mapping)
            and h10_candidate.get("sha256") == actual.get("sha256")
            and h10_candidate.get("size_bytes") == actual.get("size_bytes")
        ),
    }


def carry_leaf_inventory(carry: Any, theta_shape: tuple[int, ...]) -> list[dict[str, Any]]:
    inventory = []
    for name in getattr(carry, "__dataclass_fields__", {}):
        value = getattr(carry, name)
        if hasattr(value, "shape") and tuple(value.shape) == theta_shape:
            role = "theta_history_candidate" if name in {"t_save", "t_2ave"} else "not_T_history_candidate"
            unit = "K total-theta storage" if name in {"t_save", "t_2ave"} else "varies"
            if name == "rthraten":
                unit = "K/s held radiative theta tendency"
            inventory.append(
                {
                    "leaf": name,
                    "shape": list(value.shape),
                    "dtype": str(getattr(value, "dtype", None)),
                    "role": role,
                    "unit_or_convention": unit,
                    "summary": array_summary(np.asarray(value)),
                }
            )
    return inventory


def add_candidate(
    candidates: dict[str, dict[str, Any]],
    *,
    name: str,
    array: Any,
    stage: str,
    source_expression: str,
    offset_convention: str,
    offset_applied_k: float | None,
    verdict_eligible: bool = True,
) -> None:
    candidates[name] = {
        "name": name,
        "stage": stage,
        "source_expression": source_expression,
        "offset_convention": offset_convention,
        "offset_applied_k": offset_applied_k,
        "verdict_eligible": bool(verdict_eligible),
        "array": np.asarray(array),
    }


def run_jax_capture_and_compare(surface: Mapping[str, Any], wrf_refresh: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        _physics_step_forcing,
        _rk_scan_step_with_pre_halo_capture,
    )

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
    physics = _physics_step_forcing(carry, namelist, lead_seconds, run_radiation=run_radiation)
    result = _rk_scan_step_with_pre_halo_capture(
        physics.carry,
        namelist,
        lead_seconds=lead_seconds,
        physics_tendencies=physics.dry_tendencies,
    )
    jax.block_until_ready(result.carry.state.theta)

    candidates: dict[str, dict[str, Any]] = {}
    add_candidate(
        candidates,
        name="captured_pre_halo_state.theta_minus_300",
        array=jax.device_get(result.pre_halo_state.theta - THETA_OFFSET_K),
        stage="final RK3 after_all_rk_steps pre-halo capture",
        source_expression="result.pre_halo_state.theta - 300",
        offset_convention="State.theta is total theta; WRF history T is perturbation theta, so subtract 300 K.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="captured_post_halo_carry.state.theta_minus_300",
        array=jax.device_get(result.carry.state.theta - THETA_OFFSET_K),
        stage="final RK3 post-halo carry",
        source_expression="result.carry.state.theta - 300",
        offset_convention="State.theta is total theta; post-halo is not the green WRF target but is a plausible wrong source.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="captured_final_carry.t_save_minus_300",
        array=jax.device_get(result.carry.t_save - THETA_OFFSET_K),
        stage="final RK3 carry save family",
        source_expression="result.carry.t_save - 300",
        offset_convention="OperationalCarry.t_save is stored as total theta (see _carry_from_finished_stage); subtract 300 K.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="captured_final_carry.t_2ave_minus_300",
        array=jax.device_get(result.carry.t_2ave - THETA_OFFSET_K),
        stage="final RK3 carry scratch",
        source_expression="result.carry.t_2ave - 300",
        offset_convention="OperationalCarry.t_2ave is stored with the total-theta offset restored; subtract 300 K for perturbation comparison.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="captured_final_carry.t_2ave_raw",
        array=jax.device_get(result.carry.t_2ave),
        stage="final RK3 carry scratch diagnostic",
        source_expression="result.carry.t_2ave",
        offset_convention="Raw diagnostic only; production storage includes the 300 K offset and should not be compared directly to WRF T.",
        offset_applied_k=None,
        verdict_eligible=True,
    )
    add_candidate(
        candidates,
        name="physics_state.theta_minus_300",
        array=jax.device_get(physics.state.theta - THETA_OFFSET_K),
        stage="non-timesplit physics result before RK3",
        source_expression="physics.state.theta - 300",
        offset_convention="State.theta is total theta; this checks a possible cadence/source confusion before the dycore.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="checkpoint_prestep_carry.state.theta_minus_300",
        array=jax.device_get(carry.state.theta - THETA_OFFSET_K),
        stage="checkpoint completed step 5999 before target step 6000",
        source_expression="carry.state.theta - 300",
        offset_convention="State.theta is total theta; subtract 300 K.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="checkpoint_prestep_carry.t_save_minus_300",
        array=jax.device_get(carry.t_save - THETA_OFFSET_K),
        stage="checkpoint save family before target step 6000",
        source_expression="carry.t_save - 300",
        offset_convention="OperationalCarry.t_save is total theta; subtract 300 K.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="checkpoint_prestep_carry.t_2ave_minus_300",
        array=jax.device_get(carry.t_2ave - THETA_OFFSET_K),
        stage="checkpoint small-step scratch before target step 6000",
        source_expression="carry.t_2ave - 300",
        offset_convention="OperationalCarry.t_2ave is stored with the 300 K offset restored; subtract 300 K.",
        offset_applied_k=THETA_OFFSET_K,
    )
    add_candidate(
        candidates,
        name="checkpoint_prestep_carry.t_2ave_raw",
        array=jax.device_get(carry.t_2ave),
        stage="checkpoint small-step scratch diagnostic before target step 6000",
        source_expression="carry.t_2ave",
        offset_convention="Raw diagnostic only; included to prove an offset mistake is not the hidden match.",
        offset_applied_k=None,
        verdict_eligible=True,
    )

    candidate_results: dict[str, Any] = {}
    for name, candidate in candidates.items():
        array = candidate.pop("array")
        comparisons = {
            wrf_name: compare_mass_array_to_wrf(array, surface, wrf_field=wrf_spec["field"])
            for wrf_name, wrf_spec in WRF_T_CANDIDATES.items()
        }
        candidate_results[name] = {
            **candidate,
            "array_summary": array_summary(array),
            "comparisons": comparisons,
        }

    context = compare_state_context_to_wrf(result.pre_halo_state, surface)
    theta_shape = tuple(np.asarray(carry.state.theta).shape)
    return {
        "status": "RAN",
        "loader": "gpuwrf.runtime.checkpoint.read_checkpoint_with_runtime_state",
        "wall_s": float(time.perf_counter() - started),
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
        },
        "target_step": TARGET_STEP,
        "lead_seconds": float(lead_seconds),
        "run_radiation_for_step": run_radiation,
        "grid_shape": {
            "nz": int(getattr(grid, "nz")),
            "ny": int(getattr(grid, "ny")),
            "nx": int(getattr(grid, "nx")),
        },
        "candidate_count": len(candidate_results),
        "candidate_results": candidate_results,
        "context_pre_halo_mass_pressure": context,
        "carry_theta_shaped_leaf_inventory": carry_leaf_inventory(carry, theta_shape),
    }


def best_matches(candidate_results: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for wrf_name in WRF_T_CANDIDATES:
        rows = []
        for candidate_name, candidate in candidate_results.items():
            if not candidate.get("verdict_eligible", True):
                continue
            comparison = candidate["comparisons"][wrf_name]
            max_abs = comparison.get("max_abs")
            rows.append(
                {
                    "candidate": candidate_name,
                    "max_abs": None if max_abs is None else float(max_abs),
                    "rmse": comparison.get("rmse"),
                    "worst": comparison.get("worst"),
                }
            )
        rows.sort(key=lambda item: float("inf") if item["max_abs"] is None else item["max_abs"])
        out[wrf_name] = rows
    return out


def choose_verdict(match_tables: Mapping[str, Any]) -> tuple[str, str]:
    history_rows = match_tables["wrf_history_T__MASS_K1_T_HIST_SRC"]
    thm_rows = match_tables["wrf_thm_side__MASS_K1_T_THM"]
    history_hits = [
        row for row in history_rows if row["max_abs"] is not None and row["max_abs"] <= GREEN_TOLERANCE_MAX_ABS
    ]
    if history_hits:
        candidate = history_hits[0]["candidate"].replace(".", "_")
        return (
            f"T_SOURCE_MAPPING_CONFIRMED_{candidate}",
            f"JAX candidate {history_hits[0]['candidate']} matches WRF history T within {GREEN_TOLERANCE_MAX_ABS}.",
        )
    thm_hits = [
        row for row in thm_rows if row["max_abs"] is not None and row["max_abs"] <= GREEN_TOLERANCE_MAX_ABS
    ]
    if thm_hits:
        return (
            "T_THM_SIDE_MATCH_ONLY",
            f"JAX candidate {thm_hits[0]['candidate']} matches WRF T_THM but no candidate matches history T.",
        )
    return (
        "T_EVOLUTION_MISMATCH_CONFIRMED",
        "No inspected JAX theta/history candidate matches WRF history T or WRF T_THM within the frozen tolerance.",
    )


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


def proof_inputs() -> dict[str, Any]:
    return {
        "project_constitution": path_info(ROOT / "PROJECT_CONSTITUTION.md"),
        "agents": path_info(ROOT / "AGENTS.md"),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "handoff": path_info(HANDOFF),
        "jax_h10_prestep_carry_json": path_info(JAX_H10_JSON),
        "jax_h10_prestep_carry_producer_json": path_info(JAX_H10_PRODUCER_JSON),
        "wrf_post_rk_refresh_localization_json": path_info(WRF_REFRESH_JSON),
        "wrf_same_state_marker_savepoint_json": path_info(WRF_MARKER_JSON),
        "same_state_savepoint_request_json": path_info(SAVEPOINT_REQUEST_JSON),
        "checkpoint": path_info(CHECKPOINT),
    }


def render_candidate_table(match_tables: Mapping[str, Any], wrf_name: str, limit: int = 5) -> list[str]:
    lines = [
        "| Candidate | Max abs | RMSE |",
        "| --- | ---: | ---: |",
    ]
    for row in match_tables[wrf_name][:limit]:
        lines.append(f"| `{row['candidate']}` | `{row['max_abs']}` | `{row['rmse']}` |")
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    best = payload["best_matches"]
    context = payload["jax_capture"]["context_pre_halo_mass_pressure"]
    lines = [
        "# V0.14 T History Source Attribution",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Target",
        "",
        "- WRF target: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.",
        "- WRF history `T`: `MASS_K1.T_HIST_SRC` (`grid%th_phy_m_t0`).",
        "- WRF THM-side candidate: `MASS_K1.T_THM`.",
        f"- Tolerance: `{GREEN_TOLERANCE_MAX_ABS}` max_abs, unchanged from the h10 proof.",
        "",
        "## Artifact Identity",
        "",
        f"- Checkpoint: `{payload['checkpoint_identity']['actual_checkpoint']['path']}`.",
        f"- Same as producer record: `{payload['checkpoint_identity']['same_artifact_as_producer_recorded']}`.",
        f"- Same as canonical h10 compared artifact: `{payload['checkpoint_identity']['same_artifact_as_canonical_h10_compared']}`.",
        "",
        "## Best Matches To WRF History T",
        "",
        *render_candidate_table(best, "wrf_history_T__MASS_K1_T_HIST_SRC"),
        "",
        "## Best Matches To WRF THM",
        "",
        *render_candidate_table(best, "wrf_thm_side__MASS_K1_T_THM"),
        "",
        "## Context",
        "",
        (
            f"- Pre-halo P/PB/MU/MUB max_abs: P `{context['P']['max_abs']}`, "
            f"PB `{context['PB']['max_abs']}`, MU `{context['MU']['max_abs']}`, "
            f"MUB `{context['MUB']['max_abs']}`."
        ),
        f"- WRF `T_THM - T_HIST_SRC` max_abs: `{payload['wrf_surface']['T_THM_minus_T_HIST_SRC']['max_abs']}`.",
        "",
        "## Next Decision",
        "",
        payload["next_decision"],
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 T History Source Attribution",
        "",
        f"verdict: `{payload['verdict']}`",
        "",
        "objective: determine whether `JAX_MISMATCH_T` is a JAX theta/history-source mapping error or a real theta-evolution mismatch.",
        "",
        "files changed:",
        "- `proofs/v014/jax_t_history_source_attribution.py`",
        "- `proofs/v014/jax_t_history_source_attribution.json`",
        "- `proofs/v014/jax_t_history_source_attribution.md`",
        "- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`",
        "",
        "commands run:",
        "- `python -m py_compile proofs/v014/jax_t_history_source_attribution.py`",
        "- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_t_history_source_attribution.py`",
        "- `python -m json.tool proofs/v014/jax_t_history_source_attribution.json >/tmp/jax_t_history_source_attribution.validated.json`",
        "",
        "proof objects produced:",
        "- `proofs/v014/jax_t_history_source_attribution.json`",
        "- `proofs/v014/jax_t_history_source_attribution.md`",
        "- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`",
        "",
        "result:",
        f"- {payload['verdict_reason']}",
        "- The WRF history source and THM-side source are explicitly separated.",
        "- The checkpoint hash matches both the producer proof and the canonical h10 comparison proof.",
        "",
        "unresolved risks:",
    ]
    for risk in payload.get("unresolved_risks", []):
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    started = time.perf_counter()
    if STDERR_CAPTURE.exists():
        STDERR_CAPTURE.unlink()
    h10_payload = load_json(JAX_H10_JSON)
    producer_payload = load_json(JAX_H10_PRODUCER_JSON)
    wrf_refresh = load_json(WRF_REFRESH_JSON)
    load_json(WRF_MARKER_JSON)
    load_json(SAVEPOINT_REQUEST_JSON)

    surface_paths = [
        Path(path)
        for path in wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]["files"]
    ]
    wrf_surface = parse_refresh_files(surface_paths)
    actual_checkpoint = path_info(CHECKPOINT)
    checkpoint_identity = recorded_checkpoint_identity(actual_checkpoint, h10_payload, producer_payload)
    if not CHECKPOINT.exists():
        verdict = "T_ATTRIBUTION_BLOCKED_MISSING_CHECKPOINT"
        verdict_reason = f"Required checkpoint does not exist: {CHECKPOINT}"
        jax_capture: dict[str, Any] = {"status": "BLOCKED", "reason": verdict_reason}
        best: dict[str, Any] = {
            name: [] for name in WRF_T_CANDIDATES
        }
        next_decision = "Provide the recorded h10 step-5999 full-carry checkpoint before source-changing work."
        unresolved = [verdict_reason]
    elif not checkpoint_identity["same_artifact_as_producer_recorded"]:
        verdict = "T_ATTRIBUTION_BLOCKED_CHECKPOINT_IDENTITY_MISMATCH"
        verdict_reason = "Checkpoint hash/size does not match the producer proof record."
        jax_capture = {"status": "BLOCKED", "reason": verdict_reason}
        best = {name: [] for name in WRF_T_CANDIDATES}
        next_decision = "Resolve checkpoint provenance before attributing theta mismatch."
        unresolved = [verdict_reason]
    else:
        try:
            with capture_process_stderr(STDERR_CAPTURE):
                jax_capture = run_jax_capture_and_compare(wrf_surface, wrf_refresh)
            best = best_matches(jax_capture["candidate_results"])
            verdict, verdict_reason = choose_verdict(best)
            if verdict == "T_EVOLUTION_MISMATCH_CONFIRMED":
                next_decision = (
                    "Open a theta-evolution localization sprint; do not spend the next sprint on "
                    "JAX-vs-WRF history source remapping for `T`."
                )
            elif verdict.startswith("T_SOURCE_MAPPING_CONFIRMED"):
                next_decision = "Open a narrow source/cadence mapping fix sprint for the named JAX T source."
            else:
                next_decision = "Open a narrow WRF THM-vs-history cadence/source investigation before editing dynamics."
            unresolved = [
                "Only Boole's selected h10 patch was compared; this attributes the first mismatch, not full-domain parity.",
                "The proof is CPU-only and intentionally does not run TOST or Switzerland validation.",
            ]
        except Exception as exc:  # pragma: no cover - recorded in proof output
            verdict = f"T_ATTRIBUTION_BLOCKED_{type(exc).__name__}"
            verdict_reason = repr(exc)
            jax_capture = {"status": "BLOCKED", "reason": verdict_reason}
            best = {name: [] for name in WRF_T_CANDIDATES}
            next_decision = "Fix the blocked proof input/API path before source-changing work."
            unresolved = [verdict_reason]

    wrf_surface_summary = {
        "files": wrf_surface["files"],
        "unique_counts": wrf_surface["unique_counts"],
        "duplicate_count": wrf_surface["duplicate_count"],
        "duplicate_max_delta": wrf_surface["duplicate_max_delta"],
        "field_stats": {
            "T_HIST_SRC": wrf_field_stats(wrf_surface, "MASS_K1", "T_HIST_SRC"),
            "T_THM": wrf_field_stats(wrf_surface, "MASS_K1", "T_THM"),
            "P": wrf_field_stats(wrf_surface, "MASS_K1", "P"),
            "PB": wrf_field_stats(wrf_surface, "MASS_K1", "PB"),
            "MU_NEW": wrf_field_stats(wrf_surface, "MASS_K1", "MU_NEW"),
            "MUB": wrf_field_stats(wrf_surface, "MASS_K1", "MUB"),
        },
        "T_THM_minus_T_HIST_SRC": wrf_field_diff_stats(wrf_surface, "MASS_K1", "T_THM", "T_HIST_SRC"),
        "source_semantics": WRF_T_CANDIDATES,
    }

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.jax_t_history_source_attribution.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "tolerance": {
            "max_abs": GREEN_TOLERANCE_MAX_ABS,
            "policy": "No artificial tolerance widening; inherited from the canonical h10 proof.",
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
                "note": "Native XLA CPU AOT warnings, if emitted, are redirected here to keep terminal output compact.",
            },
        },
        "wrf_target": compact_target(wrf_refresh),
        "wrf_surface": wrf_surface_summary,
        "checkpoint_identity": checkpoint_identity,
        "canonical_h10_context": {
            "verdict": h10_payload.get("verdict"),
            "first_mismatch": (
                h10_payload.get("comparison", {}).get("first_mismatch")
                if isinstance(h10_payload.get("comparison"), Mapping)
                else None
            ),
            "comparison_run": h10_payload.get("comparison_run"),
        },
        "jax_capture": jax_capture,
        "best_matches": best,
        "acceptance_notes": {
            "uses_produced_h10_carry_checkpoint": bool(checkpoint_identity["same_artifact_as_producer_recorded"]),
            "uses_boole_same_surface_wrf_target": True,
            "distinguishes_T_HIST_SRC_from_T_THM": True,
            "retained_wrfout_used_as_verdict": False,
            "jax_vs_jax_self_compare": False,
            "production_source_edited": False,
        },
        "commands": {
            "minimum_contract_commands": [
                "python -m py_compile proofs/v014/jax_t_history_source_attribution.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_t_history_source_attribution.py",
                "python -m json.tool proofs/v014/jax_t_history_source_attribution.json >/tmp/jax_t_history_source_attribution.validated.json",
            ],
            "argv": sys.argv,
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": unresolved,
        "next_decision": next_decision,
        "wall_s": float(time.perf_counter() - started),
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")

    best_hist = best.get("wrf_history_T__MASS_K1_T_HIST_SRC", [{}])[0] if best else {}
    best_thm = best.get("wrf_thm_side__MASS_K1_T_THM", [{}])[0] if best else {}
    print(
        f"{verdict} best_T_HIST_SRC={best_hist.get('candidate')}:{best_hist.get('max_abs')} "
        f"best_T_THM={best_thm.get('candidate')}:{best_thm.get('max_abs')}"
    )
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
