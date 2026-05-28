#!/usr/bin/env python
"""Run the Oracle-baseline regression suite."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import yaml
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "tests" / "regression" / "oracle_cases.yaml"
DEFAULT_TOLERANCES = ROOT / "tests" / "regression" / "tolerances.yaml"
DEFAULT_PROOF_DIR = ROOT / "proofs" / "regression"
TIME_RE = re.compile(r"^wrfout_(?P<domain>d\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")
MILESTONE_RE = re.compile(r"^M(?P<number>\d+)$", re.IGNORECASE)

FIELD_VARIABLES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "theta": "T",
    "QVAPOR": "QVAPOR",
    "PSFC": "PSFC",
    "T2": "T2",
    "U10": "U10",
    "V10": "V10",
    "SWDOWN": "SWDOWN",
    "GLW": "GLW",
    "HFX": "HFX",
    "LH": "LH",
    "PBLH": "PBLH",
    "TSK": "TSK",
    "LU_INDEX": "LU_INDEX",
}

THRESHOLD_KEYS = ("max_abs_diff", "rmse", "mean_abs_diff", "bias_abs")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _milestone_label(args: argparse.Namespace) -> str:
    if args.milestone_snapshot:
        return str(args.milestone_snapshot)
    if args.smoke:
        return "SMOKE"
    return "ADHOC"


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))


def _resolve_case_variables(case: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    raw = case.get("expected_variables", "DEFAULT")
    if raw == "DEFAULT":
        raw = manifest.get("default_expected_variables", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{case.get('case_id')} has no expected variables")
    return [str(item) for item in raw]


def _scale_thresholds(spec: Any, scale: float) -> Any:
    if isinstance(spec, dict):
        scaled: dict[str, Any] = {}
        for key, value in spec.items():
            if key in THRESHOLD_KEYS and isinstance(value, (int, float)):
                scaled[key] = float(value) * float(scale)
            else:
                scaled[key] = _scale_thresholds(value, scale)
        return scaled
    if isinstance(spec, list):
        return [_scale_thresholds(item, scale) for item in spec]
    return spec


def _resolve_tolerances(raw: Mapping[str, Any]) -> dict[str, Any]:
    classes = raw.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("tolerances.yaml must define classes")
    resolved: dict[str, Any] = {}
    for name, spec in classes.items():
        if not isinstance(spec, dict):
            raise ValueError(f"tolerance class {name} must be a mapping")
        if "inherits" in spec:
            parent_name = str(spec["inherits"])
            if parent_name not in resolved:
                raise ValueError(f"tolerance class {name} inherits unresolved {parent_name}")
            merged = deepcopy(resolved[parent_name])
            for key, value in spec.items():
                if key not in {"inherits", "scale"}:
                    merged[key] = deepcopy(value)
            if "scale" in spec:
                merged = _scale_thresholds(merged, float(spec["scale"]))
            merged["inherits"] = parent_name
            merged["scale"] = spec.get("scale")
            resolved[str(name)] = merged
        else:
            resolved[str(name)] = deepcopy(spec)
    return resolved


def _field_tolerance(tolerances: Mapping[str, Any], class_name: str, field: str) -> dict[str, Any]:
    class_spec = tolerances.get(class_name)
    if not isinstance(class_spec, dict):
        raise ValueError(f"unknown tolerance class {class_name}")
    default = class_spec.get("default", {})
    fields = class_spec.get("fields", {})
    result: dict[str, Any] = dict(default) if isinstance(default, dict) else {}
    if isinstance(fields, dict) and isinstance(fields.get(field), dict):
        result.update(fields[field])
    return result


def _parse_valid_stamp(path: Path) -> str:
    match = TIME_RE.match(path.name)
    if match is None:
        raise ValueError(f"cannot parse valid time from {path}")
    return match.group("stamp")


def _file_map(paths: list[Path], domain: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        match = TIME_RE.match(path.name)
        if match is not None and match.group("domain") == domain:
            result[match.group("stamp")] = path
    return result


def _wrfout_files(root: Path, domain: str) -> list[Path]:
    return sorted(root.glob(f"wrfout_{domain}_*"), key=_parse_valid_stamp)


def _theta_to_absolute(values: np.ndarray, *, side: str) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[np.isfinite(values)]
    raw_min = float(np.min(finite)) if finite.size else None
    raw_max = float(np.max(finite)) if finite.size else None
    raw_mean = float(np.mean(finite)) if finite.size else None
    base = {"raw_min": raw_min, "raw_max": raw_max, "raw_mean": raw_mean}
    if side == "wrf":
        return values + 300.0, {
            **base,
            "reference_state": "perturbation_from_300K_base",
            "canonical_transform": "theta = T + 300.0",
        }
    if finite.size and float(np.median(finite)) > 150.0:
        return values, {**base, "reference_state": "absolute_theta", "canonical_transform": "theta = T"}
    return values + 300.0, {
        **base,
        "reference_state": "perturbation_from_300K_base",
        "canonical_transform": "theta = T + 300.0",
    }


def _read_field(dataset: Dataset, field: str, *, side: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    variable_name = FIELD_VARIABLES.get(field, field)
    if variable_name not in dataset.variables:
        return None, {"status": "MISSING", "variable": variable_name, "side": side}
    variable = dataset.variables[variable_name]
    values = np.asarray(variable[:])
    if values.ndim > 0 and values.shape[0] == 1:
        values = values[0]
    values = values.astype(np.float64, copy=False)
    transform = "identity"
    convention: dict[str, Any] = {}
    if field == "theta":
        values, convention = _theta_to_absolute(values, side=side)
        transform = convention.get("canonical_transform", "theta canonicalization")
    return values, {
        "status": "OK",
        "variable": variable_name,
        "units": "K" if field == "theta" else str(getattr(variable, "units", "")),
        "stagger": str(getattr(variable, "stagger", "")),
        "side": side,
        "transform": transform,
        **convention,
    }


def _common_arrays(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], bool]:
    shape_match = list(left.shape) == list(right.shape)
    if left.ndim != right.ndim:
        count = min(int(left.size), int(right.size))
        return left.ravel()[:count], right.ravel()[:count], [count], shape_match
    common = [min(int(left.shape[index]), int(right.shape[index])) for index in range(left.ndim)]
    slices = tuple(slice(0, dim) for dim in common)
    return left[slices], right[slices], common, shape_match


def _stats(gpu: np.ndarray, wrf: np.ndarray) -> dict[str, Any]:
    gpu_common, wrf_common, common_shape, shape_match = _common_arrays(gpu, wrf)
    finite_mask = np.isfinite(gpu_common) & np.isfinite(wrf_common)
    finite_count = int(finite_mask.sum())
    total_count = int(finite_mask.size)
    if finite_count:
        delta = gpu_common - wrf_common
        finite_delta = delta[finite_mask]
        abs_delta = np.where(finite_mask, np.abs(delta), -np.inf)
        flat = int(np.argmax(abs_delta))
        max_abs = float(abs_delta.flat[flat])
        argmax = [int(item) for item in np.unravel_index(flat, abs_delta.shape)]
        rmse = float(np.sqrt(np.mean(finite_delta * finite_delta)))
        mean_abs = float(np.mean(np.abs(finite_delta)))
        bias = float(np.mean(finite_delta))
        wrf_scale = float(np.nanmax(np.abs(wrf_common[finite_mask])))
        rel = max_abs / max(wrf_scale, 1.0e-30)
    else:
        max_abs = rmse = mean_abs = bias = rel = float("nan")
        argmax = []
    return {
        "status": "OK",
        "shape_match": bool(shape_match),
        "gpu_shape": list(gpu.shape),
        "wrf_shape": list(wrf.shape),
        "compared_shape": common_shape,
        "finite_count": finite_count,
        "total_count": total_count,
        "gpu_nonfinite_count": int(np.size(gpu_common) - int(np.isfinite(gpu_common).sum())),
        "wrf_nonfinite_count": int(np.size(wrf_common) - int(np.isfinite(wrf_common).sum())),
        "max_abs_diff": max_abs,
        "rmse": rmse,
        "mean_abs_diff": mean_abs,
        "bias": bias,
        "rel_diff": rel,
        "argmax_diff_idx": argmax,
    }


def _compare_field(gpu_ds: Dataset, wrf_ds: Dataset, field: str) -> dict[str, Any]:
    gpu_values, gpu_meta = _read_field(gpu_ds, field, side="gpu")
    wrf_values, wrf_meta = _read_field(wrf_ds, field, side="wrf")
    if gpu_values is None or wrf_values is None:
        return {
            "status": "MISSING",
            "gpu": gpu_meta,
            "wrf": wrf_meta,
            "max_abs_diff": None,
            "rmse": None,
            "mean_abs_diff": None,
            "bias": None,
            "argmax_diff_idx": [],
        }
    return {**_stats(gpu_values, wrf_values), "gpu": gpu_meta, "wrf": wrf_meta}


def _threshold_pass(metric: str, value: float | None, threshold: Any) -> tuple[bool, str]:
    if threshold is None:
        return True, "not_checked"
    if value is None or not math.isfinite(float(value)):
        return False, "nonfinite_metric"
    if float(threshold) == 0.0:
        return (float(value) == 0.0), f"{metric} <= 0"
    return (float(value) <= float(threshold)), f"{metric} <= {threshold}"


def _evaluate_metrics(metrics: Mapping[str, Any], tolerance: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    checked = 0
    for metric in ("max_abs_diff", "rmse", "mean_abs_diff"):
        if metric not in tolerance:
            continue
        threshold = tolerance.get(metric)
        if threshold is None:
            continue
        ok, rule = _threshold_pass(metric, metrics.get(metric), threshold)
        checks[metric] = {"passed": ok, "rule": rule, "observed": metrics.get(metric), "threshold": threshold}
        passed = passed and ok
        checked += 1
    return {"passed": bool(passed), "checked_metric_count": checked, "checks": checks}


def _normalized_score(metrics: Mapping[str, Any], tolerance: Mapping[str, Any]) -> float | None:
    scores: list[float] = []
    for metric in ("rmse", "max_abs_diff", "mean_abs_diff"):
        threshold = tolerance.get(metric)
        observed = metrics.get(metric)
        if threshold is None or observed is None:
            continue
        if not math.isfinite(float(observed)):
            return math.inf
        if float(threshold) == 0.0:
            scores.append(0.0 if float(observed) == 0.0 else math.inf)
        else:
            scores.append(float(observed) / float(threshold))
    return max(scores) if scores else None


def _summarize_field(rows: list[dict[str, Any]], field: str, tolerance: Mapping[str, Any]) -> dict[str, Any]:
    max_abs = -1.0
    hour_of_max: int | None = None
    sum_sq = 0.0
    finite_count = 0
    missing_hours: list[int] = []
    shape_mismatch_hours: list[int] = []
    first_failure_hour: int | None = None

    for row in rows:
        stats = row["fields"].get(field, {})
        hour = int(row["hour"])
        if stats.get("status") != "OK":
            missing_hours.append(hour)
            if first_failure_hour is None:
                first_failure_hour = hour
            continue
        count = int(stats.get("finite_count", 0))
        rmse = stats.get("rmse")
        if count and rmse is not None and math.isfinite(float(rmse)):
            sum_sq += float(rmse) * float(rmse) * count
            finite_count += count
        current_max = stats.get("max_abs_diff")
        if current_max is not None and math.isfinite(float(current_max)) and float(current_max) > max_abs:
            max_abs = float(current_max)
            hour_of_max = hour
        if not bool(stats.get("shape_match", False)):
            shape_mismatch_hours.append(hour)

    metrics = {
        "max_abs_diff": None if max_abs < 0.0 else max_abs,
        "rmse": float(np.sqrt(sum_sq / finite_count)) if finite_count else None,
        "finite_count": finite_count,
    }
    evaluation = _evaluate_metrics(metrics, tolerance)
    passed = bool(evaluation["passed"]) and not missing_hours and not shape_mismatch_hours and finite_count > 0
    if first_failure_hour is None and not passed:
        first_failure_hour = hour_of_max
    score = _normalized_score(metrics, tolerance)
    return {
        "field": field,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "tolerance": dict(tolerance),
        "metrics": metrics,
        "normalized_score": score,
        "max_abs_diff_over_hours": metrics["max_abs_diff"],
        "hour_of_max_abs_diff": hour_of_max,
        "rmse_over_all_hours": metrics["rmse"],
        "finite_count_over_hours": finite_count,
        "missing_hours": missing_hours,
        "shape_mismatch_hours": shape_mismatch_hours,
        "first_failure_hour": first_failure_hour,
        "evaluation": evaluation,
    }


def _load_pipeline_files(pipeline_path: Path, domain: str) -> tuple[list[Path], dict[str, Any]]:
    if not pipeline_path.is_file():
        return [], {"load_status": "MISSING_PIPELINE_RUN", "pipeline_run": str(pipeline_path)}
    payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    listed = [Path(item) for item in payload.get("wrfout_files", [])]
    existing = [path for path in listed if path.is_file() and TIME_RE.match(path.name)]
    return sorted(existing, key=_parse_valid_stamp), {
        "load_status": "OK" if existing else "NO_EXISTING_WRFOUT_FILES",
        "pipeline_run": str(pipeline_path),
        "pipeline_verdict": payload.get("verdict"),
        "run_id": payload.get("run_id"),
        "domain": payload.get("domain"),
        "hours": payload.get("hours"),
        "output_dir": payload.get("output_dir"),
        "listed_file_count": len(listed),
        "existing_file_count": len(existing),
        "wrfout_files": [str(path) for path in existing if path.name.startswith(f"wrfout_{domain}_")],
    }


def _gpu_paths(case: Mapping[str, Any], milestone: str, proof_dir: Path) -> dict[str, Path]:
    case_id = str(case["case_id"])
    safe_milestone = _safe_label(milestone)
    run_proof = proof_dir / "gpu_runs" / f"{case_id}_{safe_milestone}"
    output_dir = Path("/tmp/oracle_regression") / safe_milestone / case_id
    return {
        "proof_dir": run_proof,
        "output_dir": output_dir,
        "pipeline_daily": run_proof / "pipeline_run_20260521.json",
        "pipeline_l2": run_proof / "pipeline_run_l2_d02.json",
    }


def _build_gpu_command(case: Mapping[str, Any], *, hours: int, milestone: str, proof_dir: Path) -> tuple[list[str] | None, Path]:
    gpu = case.get("gpu", {})
    if not isinstance(gpu, dict):
        return None, proof_dir / "missing-pipeline.json"
    runner = str(gpu.get("runner", ""))
    paths = _gpu_paths(case, milestone, proof_dir)
    domain = str(gpu.get("domain", case.get("oracle", {}).get("domain", "d02")))
    run_id = str(gpu.get("run_id", ""))
    run_root = str(gpu.get("run_root", ""))
    if runner == "m7_daily_pipeline":
        command = [
            "taskset",
            "-c",
            "0-3",
            sys.executable,
            str(ROOT / "scripts" / "m7_daily_pipeline.py"),
            "--run-id",
            run_id,
            "--hours",
            str(int(hours)),
            "--output-dir",
            str(paths["output_dir"]),
            "--proof-dir",
            str(paths["proof_dir"]),
            "--run-root",
            run_root,
            "--domain",
            domain,
        ]
        return command, paths["pipeline_daily"]
    if runner == "m7_l2_d02_replay":
        command = [
            "taskset",
            "-c",
            "0-3",
            sys.executable,
            str(ROOT / "scripts" / "m7_l2_d02_replay.py"),
            "--run-id",
            run_id,
            "--hours",
            str(int(hours)),
            "--output-root",
            str(paths["output_dir"].parent),
            "--proof-dir",
            str(paths["proof_dir"]),
            "--run-root",
            run_root,
        ]
        return command, paths["pipeline_daily"]
    return None, paths["proof_dir"] / "unsupported-runner.json"


def _run_gpu(case: Mapping[str, Any], *, hours: int, milestone: str, proof_dir: Path, force: bool) -> dict[str, Any]:
    oracle = case.get("oracle", {})
    domain = str(oracle.get("domain", case.get("gpu", {}).get("domain", "d02")))
    command, pipeline_path = _build_gpu_command(case, hours=hours, milestone=milestone, proof_dir=proof_dir)
    cached_files, cached_meta = _load_pipeline_files(pipeline_path, domain)
    if cached_files and not force:
        return {
            "status": "REUSED_EXISTING_GPU_OUTPUT",
            "command": command,
            "pipeline_run": str(pipeline_path),
            "stdout_log": None,
            "stderr_log": None,
            **cached_meta,
        }
    if pipeline_path.is_file() and not force:
        return {
            "status": "REUSED_EXISTING_GPU_PROOF_WITHOUT_FILES",
            "command": command,
            "pipeline_run": str(pipeline_path),
            "stdout_log": None,
            "stderr_log": None,
            **cached_meta,
        }
    if command is None:
        return {
            "status": "BLOCKED_UNSUPPORTED_GPU_RUNNER",
            "reason": f"unsupported gpu.runner={case.get('gpu', {}).get('runner')}",
            "pipeline_run": str(pipeline_path),
            "command": None,
        }

    paths = _gpu_paths(case, milestone, proof_dir)
    paths["proof_dir"].mkdir(parents=True, exist_ok=True)
    stdout_log = paths["proof_dir"] / "gpu_run.stdout.txt"
    stderr_log = paths["proof_dir"] / "gpu_run.stderr.txt"
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "true")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("OMP_NUM_THREADS", "4")
    start = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, text=True, check=False)
    files, meta = _load_pipeline_files(pipeline_path, domain)
    return {
        "status": "GPU_RUN_COMPLETE_WITH_FILES" if files else "GPU_RUN_NO_FILES",
        "command": command,
        "exit_code": int(completed.returncode),
        "wall_clock_seconds": time.perf_counter() - start,
        "pipeline_run": str(pipeline_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        **meta,
    }


def _blocked_case_payload(
    case: Mapping[str, Any],
    *,
    milestone: str,
    mode: str,
    manifest: Mapping[str, Any],
    tolerances: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    fields = _resolve_case_variables(case, manifest)
    tolerance_class = str(case.get("tolerance_class"))
    tests = {
        f"{case['case_id']}::{field}": {
            "case_id": str(case["case_id"]),
            "field": field,
            "status": "BLOCKED",
            "passed": False,
            "tolerance_class": tolerance_class,
            "tolerance": _field_tolerance(tolerances, tolerance_class, field),
            "metrics": {},
            "normalized_score": None,
            "blocked_reason": case.get("blocked_reason"),
        }
        for field in fields
    }
    return {
        "schema": "OracleBaselineRegressionCase",
        "schema_version": 1,
        "case_id": str(case["case_id"]),
        "case_type": case.get("type"),
        "milestone": milestone,
        "mode": mode,
        "generated_utc": _utc_now(),
        "commit": _git_commit(),
        "case_status": "BLOCKED",
        "passed": False,
        "blocked_reason": case.get("blocked_reason"),
        "oracle": case.get("oracle", {}),
        "tolerance_class": tolerance_class,
        "expected_variables": fields,
        "hours": [],
        "field_summary": {},
        "tests": tests,
        "output_path": str(output_path),
    }


def _compare_case(
    case: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    tolerances: Mapping[str, Any],
    milestone: str,
    mode: str,
    proof_dir: Path,
    force_gpu_run: bool,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    output_path = proof_dir / f"{case_id}_{_safe_label(milestone)}.json"
    if str(case.get("status", "AVAILABLE")).upper() == "BLOCKED":
        payload = _blocked_case_payload(
            case,
            milestone=milestone,
            mode=mode,
            manifest=manifest,
            tolerances=tolerances,
            output_path=output_path,
        )
        _write_json(output_path, payload)
        return payload

    oracle = case.get("oracle", {})
    if not isinstance(oracle, dict):
        raise ValueError(f"{case_id} oracle must be a mapping")
    domain = str(oracle.get("domain", "d02"))
    fields = _resolve_case_variables(case, manifest)
    tolerance_class = str(case.get("tolerance_class"))
    gpu_cfg = case.get("gpu", {})
    run_hours = int(gpu_cfg.get("smoke_hours" if mode == "smoke" else "full_hours", oracle.get("expected_forecast_hours", 1)))
    compare_hours = run_hours

    gpu_run = _run_gpu(case, hours=run_hours, milestone=milestone, proof_dir=proof_dir, force=force_gpu_run)
    pipeline_path = Path(str(gpu_run.get("pipeline_run", "")))
    gpu_files, gpu_meta = _load_pipeline_files(pipeline_path, domain)
    wrf_root = Path(str(oracle.get("run_dir", "")))
    wrf_files = _wrfout_files(wrf_root, domain) if wrf_root.is_dir() else []
    gpu_by_time = _file_map(gpu_files, domain)
    wrf_by_time = _file_map(wrf_files, domain)
    common_times = sorted(set(gpu_by_time).intersection(wrf_by_time))[:compare_hours]

    rows: list[dict[str, Any]] = []
    for hour, stamp in enumerate(common_times, start=1):
        gpu_path = gpu_by_time[stamp]
        wrf_path = wrf_by_time[stamp]
        with Dataset(gpu_path, "r") as gpu_ds, Dataset(wrf_path, "r") as wrf_ds:
            row_fields = {field: _compare_field(gpu_ds, wrf_ds, field) for field in fields}
        rows.append(
            {
                "hour": hour,
                "valid_time_utc": stamp.replace("_", "T") + "+00:00",
                "gpu_path": str(gpu_path),
                "wrf_path": str(wrf_path),
                "fields": row_fields,
            }
        )

    field_summary = {
        field: _summarize_field(rows, field, _field_tolerance(tolerances, tolerance_class, field))
        for field in fields
    }
    missing_inputs = {
        "gpu_only_valid_times": sorted(set(gpu_by_time) - set(wrf_by_time)),
        "wrf_only_valid_times": sorted(set(wrf_by_time) - set(gpu_by_time)),
        "matched_valid_times": common_times,
    }
    tests = {
        f"{case_id}::{field}": {
            "case_id": case_id,
            "field": field,
            "status": summary["status"],
            "passed": bool(summary["passed"]),
            "tolerance_class": tolerance_class,
            "tolerance": summary["tolerance"],
            "metrics": summary["metrics"],
            "normalized_score": summary["normalized_score"],
            "first_failure_hour": summary["first_failure_hour"],
        }
        for field, summary in field_summary.items()
    }
    case_passed = bool(rows) and all(bool(item["passed"]) for item in field_summary.values())
    if not common_times:
        case_status = "BLOCKED_MISSING_INPUTS"
    elif case_passed:
        case_status = "PASS"
    else:
        case_status = "FAIL"

    payload = {
        "schema": "OracleBaselineRegressionCase",
        "schema_version": 1,
        "case_id": case_id,
        "case_type": case.get("type"),
        "milestone": milestone,
        "mode": mode,
        "generated_utc": _utc_now(),
        "commit": _git_commit(),
        "case_status": case_status,
        "passed": case_passed,
        "oracle": {**oracle, "wrfout_file_count": len(wrf_files), "wrfout_files": [str(path) for path in wrf_files]},
        "gpu_run": {**gpu_run, "load_metadata": gpu_meta},
        "tolerance_class": tolerance_class,
        "expected_variables": fields,
        "requested_hours": run_hours,
        "matched_hour_count": len(rows),
        "missing_inputs": missing_inputs,
        "hours": rows,
        "field_summary": field_summary,
        "tests": tests,
        "output_path": str(output_path),
    }
    _write_json(output_path, payload)
    return payload


def _aggregate(
    case_payloads: list[dict[str, Any]],
    *,
    milestone: str,
    mode: str,
    proof_dir: Path,
    snapshot_path: Path | None,
) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    for payload in case_payloads:
        tests.update(payload.get("tests", {}))
    passed = [test for test in tests.values() if test.get("status") == "PASS"]
    failed = [test for test in tests.values() if test.get("status") == "FAIL"]
    blocked_tests = [test for test in tests.values() if test.get("status") == "BLOCKED"]
    blocked_cases = [payload for payload in case_payloads if str(payload.get("case_status", "")).startswith("BLOCKED")]
    executed_cases = [payload for payload in case_payloads if not str(payload.get("case_status", "")).startswith("BLOCKED")]
    aggregate_status = "PASS"
    if failed:
        aggregate_status = "FAIL"
    elif blocked_cases:
        aggregate_status = "PARTIAL_BLOCKED"
    aggregate_path = proof_dir / f"aggregate_{_safe_label(milestone)}.json"
    return {
        "schema": "OracleBaselineRegressionAggregate",
        "schema_version": 1,
        "milestone": milestone,
        "mode": mode,
        "generated_utc": _utc_now(),
        "commit": _git_commit(),
        "aggregate_status": aggregate_status,
        "passed": aggregate_status == "PASS",
        "case_count": len(case_payloads),
        "executed_case_count": len(executed_cases),
        "blocked_case_count": len(blocked_cases),
        "field_test_count": len(tests),
        "passed_test_count": len(passed),
        "failed_test_count": len(failed),
        "blocked_test_count": len(blocked_tests),
        "case_results": [
            {
                "case_id": payload.get("case_id"),
                "case_status": payload.get("case_status"),
                "passed": payload.get("passed"),
                "matched_hour_count": payload.get("matched_hour_count", 0),
                "output_path": payload.get("output_path"),
            }
            for payload in case_payloads
        ],
        "tests": tests,
        "aggregate_path": str(aggregate_path),
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
    }


def _selected_cases(manifest: Mapping[str, Any], *, smoke: bool, requested: list[str]) -> list[dict[str, Any]]:
    raw_cases = manifest.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("oracle_cases.yaml must define cases as a list")
    cases = [dict(item) for item in raw_cases]
    if requested:
        requested_set = set(requested)
        cases = [case for case in cases if str(case.get("case_id")) in requested_set]
    if smoke:
        cases = [case for case in cases if bool(case.get("smoke", False))]
    if not cases:
        raise ValueError("no regression cases selected")
    return cases


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_yaml(args.cases)
    tolerances = _resolve_tolerances(_load_yaml(args.tolerances))
    milestone = _milestone_label(args)
    mode = "smoke" if args.smoke else "full"
    proof_dir = args.proof_dir
    proof_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_cases(manifest, smoke=bool(args.smoke), requested=args.case)
    case_payloads = [
        _compare_case(
            case,
            manifest=manifest,
            tolerances=tolerances,
            milestone=milestone,
            mode=mode,
            proof_dir=proof_dir,
            force_gpu_run=bool(args.force_gpu_run),
        )
        for case in selected
    ]
    snapshot_path = proof_dir / f"snapshot_{_safe_label(milestone)}.json" if args.milestone_snapshot else None
    aggregate = _aggregate(case_payloads, milestone=milestone, mode=mode, proof_dir=proof_dir, snapshot_path=snapshot_path)
    _write_json(Path(str(aggregate["aggregate_path"])), aggregate)
    if snapshot_path is not None:
        snapshot = {
            "schema": "OracleBaselineRegressionSnapshot",
            "schema_version": 1,
            "milestone": milestone,
            "generated_utc": aggregate["generated_utc"],
            "commit": aggregate["commit"],
            "aggregate": {key: aggregate[key] for key in aggregate if key != "tests"},
            "tests": aggregate["tests"],
        }
        _write_json(snapshot_path, snapshot)
    return aggregate


def _previous_milestone_label(current: str, proof_dir: Path) -> str | None:
    match = MILESTONE_RE.match(current)
    if match:
        number = int(match.group("number"))
        preferred = f"M{number - 1}" if number > 0 else None
        if preferred and (proof_dir / f"snapshot_{preferred}.json").is_file():
            return preferred
        candidates: list[tuple[int, str]] = []
        for snapshot in proof_dir.glob("snapshot_M*.json"):
            label = snapshot.stem.replace("snapshot_", "")
            label_match = MILESTONE_RE.match(label)
            if label_match and int(label_match.group("number")) < number:
                candidates.append((int(label_match.group("number")), label))
        return sorted(candidates)[-1][1] if candidates else preferred
    snapshots = sorted(proof_dir.glob("snapshot_M*.json"))
    labels = [snapshot.stem.replace("snapshot_", "") for snapshot in snapshots]
    labels = [label for label in labels if label != current]
    return labels[-1] if labels else None


def _load_snapshot(proof_dir: Path, milestone: str) -> dict[str, Any] | None:
    path = proof_dir / f"snapshot_{_safe_label(milestone)}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_snapshots(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    current_label = str(args.compare_snapshot)
    proof_dir = args.proof_dir
    current = _load_snapshot(proof_dir, current_label)
    previous_label = args.previous_snapshot or _previous_milestone_label(current_label, proof_dir)
    previous = _load_snapshot(proof_dir, previous_label) if previous_label else None
    output_path = proof_dir / (
        f"regression_check_{_safe_label(previous_label)}_to_{_safe_label(current_label)}.json"
        if previous_label
        else f"regression_check_NONE_to_{_safe_label(current_label)}.json"
    )
    if current is None:
        payload = {
            "schema": "OracleBaselineRegressionCheck",
            "schema_version": 1,
            "status": "BLOCKED_MISSING_CURRENT_SNAPSHOT",
            "current_milestone": current_label,
            "previous_milestone": previous_label,
            "output_path": str(output_path),
        }
        _write_json(output_path, payload)
        return payload, 2
    if previous is None:
        payload = {
            "schema": "OracleBaselineRegressionCheck",
            "schema_version": 1,
            "status": "NO_PREVIOUS_SNAPSHOT",
            "current_milestone": current_label,
            "previous_milestone": previous_label,
            "newly_passing": [],
            "newly_failing": [],
            "improved_within_tolerance": [],
            "worsened_within_tolerance": [],
            "pass_count_decreased": False,
            "exit_policy": "zero because no previous snapshot was available",
            "output_path": str(output_path),
        }
        _write_json(output_path, payload)
        return payload, 0

    current_tests = current.get("tests", {})
    previous_tests = previous.get("tests", {})
    newly_passing: list[dict[str, Any]] = []
    newly_failing: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []
    all_ids = sorted(set(current_tests) | set(previous_tests))
    for test_id in all_ids:
        current_test = current_tests.get(test_id)
        previous_test = previous_tests.get(test_id)
        if current_test is None or previous_test is None:
            continue
        current_status = current_test.get("status")
        previous_status = previous_test.get("status")
        row = {
            "test_id": test_id,
            "previous_status": previous_status,
            "current_status": current_status,
            "previous_score": previous_test.get("normalized_score"),
            "current_score": current_test.get("normalized_score"),
        }
        if previous_status != "PASS" and current_status == "PASS":
            newly_passing.append(row)
            continue
        if previous_status == "PASS" and current_status != "PASS":
            newly_failing.append(row)
            continue
        previous_score = previous_test.get("normalized_score")
        current_score = current_test.get("normalized_score")
        if previous_score is None or current_score is None:
            continue
        if float(current_score) < float(previous_score):
            improved.append(row)
        elif float(current_score) > float(previous_score):
            worsened.append(row)

    current_pass_count = int(current.get("aggregate", {}).get("passed_test_count", 0))
    previous_pass_count = int(previous.get("aggregate", {}).get("passed_test_count", 0))
    pass_count_decreased = current_pass_count < previous_pass_count
    status = "PASS" if not newly_failing and not pass_count_decreased else "FAIL"
    payload = {
        "schema": "OracleBaselineRegressionCheck",
        "schema_version": 1,
        "status": status,
        "current_milestone": current_label,
        "previous_milestone": previous_label,
        "current_snapshot": str(proof_dir / f"snapshot_{_safe_label(current_label)}.json"),
        "previous_snapshot": str(proof_dir / f"snapshot_{_safe_label(previous_label)}.json"),
        "current_pass_count": current_pass_count,
        "previous_pass_count": previous_pass_count,
        "pass_count_decreased": pass_count_decreased,
        "newly_passing": newly_passing,
        "newly_failing": newly_failing,
        "improved_within_tolerance": improved,
        "worsened_within_tolerance": worsened,
        "exit_policy": "0 iff no newly-failing tests and aggregate pass-count did not decrease",
        "output_path": str(output_path),
    }
    _write_json(output_path, payload)
    return payload, 0 if status == "PASS" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--tolerances", type=Path, default=DEFAULT_TOLERANCES)
    parser.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--milestone-snapshot", default=None)
    parser.add_argument("--compare-snapshot", default=None)
    parser.add_argument("--previous-snapshot", default=None)
    parser.add_argument("--force-gpu-run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when aggregate status is not PASS.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare_snapshot:
        payload, exit_code = compare_snapshots(args)
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
        return exit_code
    aggregate = run_suite(args)
    print(json.dumps(_jsonable(aggregate), indent=2, sort_keys=True))
    if args.strict and aggregate.get("aggregate_status") != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
