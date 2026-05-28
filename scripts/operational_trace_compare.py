#!/usr/bin/env python
"""Compare GPU hourly wrfout files against WRF hourly wrfout files.

M9.A was pivoted from per-operator Fortran trace comparison to an hourly
``wrfout`` divergence trace because the WRF Canary reference run now provides
full hourly NetCDF history output without requiring a new WRF build.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

DEFAULT_CASE = "20260521"
DEFAULT_DOMAIN = "d02"
DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
DEFAULT_WRF_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3") / DEFAULT_RUN_ID
DEFAULT_GPU_PIPELINE_RUN = (
    ROOT / ".agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json"
)
DEFAULT_OUTPUT = ROOT / "proofs/m9/operational_trace_hourly.json"
TIME_RE = re.compile(r"^wrfout_(?P<domain>d\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")


@dataclass(frozen=True)
class FieldSpec:
    output_name: str
    variable_name: str
    transform: str = "identity"


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("U", "U"),
    FieldSpec("V", "V"),
    FieldSpec("W", "W"),
    FieldSpec("theta", "T", "theta = canonical absolute potential temperature"),
    FieldSpec("QVAPOR", "QVAPOR"),
    FieldSpec("PSFC", "PSFC"),
    FieldSpec("T2", "T2"),
    FieldSpec("U10", "U10"),
    FieldSpec("V10", "V10"),
    FieldSpec("SWDOWN", "SWDOWN"),
    FieldSpec("GLW", "GLW"),
    FieldSpec("HFX", "HFX"),
    FieldSpec("LH", "LH"),
    FieldSpec("PBLH", "PBLH"),
    FieldSpec("TSK", "TSK"),
    FieldSpec("LU_INDEX", "LU_INDEX"),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _parse_valid_stamp(path: Path) -> str:
    match = TIME_RE.match(path.name)
    if match is None:
        raise ValueError(f"cannot parse valid time from {path}")
    return match.group("stamp")


def _file_map(paths: Iterable[Path], domain: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in paths:
        match = TIME_RE.match(path.name)
        if match is None or match.group("domain") != domain:
            continue
        files[match.group("stamp")] = path
    return files


def _load_gpu_files(pipeline_run: Path | None, gpu_root: Path | None, domain: str) -> tuple[list[Path], dict[str, Any]]:
    provenance: dict[str, Any] = {"source": None, "pipeline_run": str(pipeline_run) if pipeline_run else None}
    if pipeline_run is not None and pipeline_run.is_file():
        payload = json.loads(pipeline_run.read_text(encoding="utf-8"))
        listed = [Path(item) for item in payload.get("wrfout_files", [])]
        existing = [path for path in listed if path.is_file()]
        provenance.update(
            {
                "source": "pipeline_run.wrfout_files",
                "run_id": payload.get("run_id"),
                "output_dir": payload.get("output_dir"),
                "verdict": payload.get("verdict"),
                "listed_file_count": len(listed),
                "existing_file_count": len(existing),
            }
        )
        if existing:
            return sorted(existing, key=_parse_valid_stamp), provenance
        if gpu_root is None and payload.get("output_dir"):
            gpu_root = Path(str(payload["output_dir"]))

    if gpu_root is not None:
        files = sorted(gpu_root.glob(f"wrfout_{domain}_*"), key=_parse_valid_stamp)
        provenance.update(
            {
                "source": "gpu_root.glob",
                "output_dir": str(gpu_root),
                "listed_file_count": len(files),
                "existing_file_count": len(files),
            }
        )
        return files, provenance

    provenance["source"] = "none"
    provenance["listed_file_count"] = 0
    provenance["existing_file_count"] = 0
    return [], provenance


def _load_wrf_files(wrf_root: Path, domain: str) -> list[Path]:
    return sorted(wrf_root.glob(f"wrfout_{domain}_*"), key=_parse_valid_stamp)


def _theta_to_absolute(values: np.ndarray, *, side: str) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[np.isfinite(values)]
    raw_min = float(np.min(finite)) if finite.size else None
    raw_max = float(np.max(finite)) if finite.size else None
    raw_mean = float(np.mean(finite)) if finite.size else None

    if side == "wrf":
        return values + 300.0, {
            "reference_state": "perturbation_from_300K_base",
            "canonical_transform": "theta = T + 300.0",
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
        }

    if finite.size and float(np.median(finite)) > 150.0:
        return values, {
            "reference_state": "absolute_theta",
            "canonical_transform": "theta = T",
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
        }
    return values + 300.0, {
        "reference_state": "perturbation_from_300K_base",
        "canonical_transform": "theta = T + 300.0",
        "raw_min": raw_min,
        "raw_max": raw_max,
        "raw_mean": raw_mean,
    }


def _read_field(dataset: Dataset, spec: FieldSpec, *, side: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    if spec.variable_name not in dataset.variables:
        return None, {"status": "MISSING", "variable": spec.variable_name}
    variable = dataset.variables[spec.variable_name]
    values = np.asarray(variable[:])
    if values.ndim > 0 and values.shape[0] == 1:
        values = values[0]
    values = values.astype(np.float64, copy=False)
    convention: dict[str, Any] = {}
    if spec.output_name == "theta":
        values, convention = _theta_to_absolute(values, side=side)
    return values, {
        "status": "OK",
        "variable": spec.variable_name,
        "units": "K" if spec.output_name == "theta" else str(getattr(variable, "units", "")),
        "stagger": str(getattr(variable, "stagger", "")),
        "transform": spec.transform,
        "side": side,
        **convention,
    }


def _common_slices(left: np.ndarray, right: np.ndarray) -> tuple[tuple[int, ...], tuple[slice, ...]]:
    rank = min(left.ndim, right.ndim)
    common = tuple(min(left.shape[index], right.shape[index]) for index in range(rank))
    return common, tuple(slice(0, dim) for dim in common)


def _stats(gpu: np.ndarray, wrf: np.ndarray) -> dict[str, Any]:
    common_shape, slices = _common_slices(gpu, wrf)
    gpu_common = gpu[slices]
    wrf_common = wrf[slices]
    finite_mask = np.isfinite(gpu_common) & np.isfinite(wrf_common)
    finite_count = int(finite_mask.sum())
    total_count = int(finite_mask.size)
    nonfinite_gpu = int(np.size(gpu_common) - int(np.isfinite(gpu_common).sum()))
    nonfinite_wrf = int(np.size(wrf_common) - int(np.isfinite(wrf_common).sum()))

    if finite_count:
        delta = gpu_common - wrf_common
        abs_delta = np.where(finite_mask, np.abs(delta), -np.inf)
        flat = int(np.argmax(abs_delta))
        argmax = [int(item) for item in np.unravel_index(flat, abs_delta.shape)]
        finite_delta = delta[finite_mask]
        max_abs = float(abs_delta.flat[flat])
        rmse = float(np.sqrt(np.mean(finite_delta * finite_delta)))
        mean_abs = float(np.mean(np.abs(finite_delta)))
        bias = float(np.mean(finite_delta))
        wrf_scale = float(np.nanmax(np.abs(wrf_common[finite_mask])))
        rel = max_abs / max(wrf_scale, 1.0e-30)
    else:
        argmax = []
        max_abs = float("nan")
        rmse = float("nan")
        mean_abs = float("nan")
        bias = float("nan")
        rel = float("nan")

    return {
        "status": "OK",
        "shape_match": list(gpu.shape) == list(wrf.shape),
        "gpu_shape": list(gpu.shape),
        "wrf_shape": list(wrf.shape),
        "compared_shape": list(common_shape),
        "finite_count": finite_count,
        "total_count": total_count,
        "gpu_nonfinite_count": nonfinite_gpu,
        "wrf_nonfinite_count": nonfinite_wrf,
        "max_abs_diff": max_abs,
        "rmse": rmse,
        "mean_abs_diff": mean_abs,
        "bias": bias,
        "rel_diff": rel,
        "argmax_diff_idx": argmax,
    }


def _compare_field(gpu_ds: Dataset, wrf_ds: Dataset, spec: FieldSpec) -> dict[str, Any]:
    gpu_values, gpu_meta = _read_field(gpu_ds, spec, side="gpu")
    wrf_values, wrf_meta = _read_field(wrf_ds, spec, side="wrf")
    if gpu_values is None or wrf_values is None:
        return {
            "status": "MISSING",
            "gpu": gpu_meta,
            "wrf": wrf_meta,
            "max_abs_diff": None,
            "rmse": None,
            "argmax_diff_idx": [],
        }
    stats = _stats(gpu_values, wrf_values)
    stats["gpu"] = gpu_meta
    stats["wrf"] = wrf_meta
    return stats


def _field_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for spec in FIELD_SPECS:
        max_abs = -1.0
        hour_of_max: int | None = None
        first_hour: int | None = None
        first_valid_time: str | None = None
        sum_sq = 0.0
        count = 0
        missing_hours: list[int] = []
        shape_mismatch_hours: list[int] = []
        for row in rows:
            stats = row["fields"].get(spec.output_name, {})
            if stats.get("status") != "OK":
                missing_hours.append(int(row["hour"]))
                continue
            finite_count = int(stats["finite_count"])
            rmse = float(stats["rmse"])
            if finite_count and np.isfinite(rmse):
                sum_sq += rmse * rmse * finite_count
                count += finite_count
            current = float(stats["max_abs_diff"])
            if current > max_abs:
                max_abs = current
                hour_of_max = int(row["hour"])
            if first_hour is None and current > 0.0:
                first_hour = int(row["hour"])
                first_valid_time = str(row["valid_time_utc"])
            if not bool(stats["shape_match"]):
                shape_mismatch_hours.append(int(row["hour"]))
        summary[spec.output_name] = {
            "max_abs_diff_over_hours": None if max_abs < 0.0 else max_abs,
            "hour_of_max_abs_diff": hour_of_max,
            "rmse_over_all_hours": float(np.sqrt(sum_sq / count)) if count else None,
            "finite_count_over_hours": count,
            "first_divergence_hour": first_hour,
            "first_divergence_valid_time_utc": first_valid_time,
            "missing_hours": missing_hours,
            "shape_mismatch_hours": shape_mismatch_hours,
        }
    return summary


def _first_divergence(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        for spec in FIELD_SPECS:
            stats = row["fields"].get(spec.output_name, {})
            if stats.get("status") != "OK":
                continue
            if float(stats["max_abs_diff"]) > 0.0:
                return {
                    "hour": int(row["hour"]),
                    "valid_time_utc": row["valid_time_utc"],
                    "field": spec.output_name,
                    "max_abs_diff": float(stats["max_abs_diff"]),
                    "rmse": float(stats["rmse"]),
                    "rel_diff": float(stats["rel_diff"]),
                    "argmax_diff_idx": stats["argmax_diff_idx"],
                }
    return None


def _missing_payload(
    *,
    case: str,
    domain: str,
    gpu_files: list[Path],
    wrf_files: list[Path],
    gpu_provenance: dict[str, Any],
    wrf_root: Path,
    start: float,
) -> dict[str, Any]:
    return {
        "trace_version": "wrfout-hourly-1.0",
        "case": case,
        "domain": domain,
        "commit": _git_commit(),
        "status": "M9A_PARTIAL_MISSING_WRFOUT_INPUT",
        "field_order": [spec.output_name for spec in FIELD_SPECS],
        "gpu_source": {**gpu_provenance, "wrfout_files": [str(path) for path in gpu_files]},
        "wrf_reference": {
            "run_dir": str(wrf_root),
            "wrfout_file_count": len(wrf_files),
            "wrfout_files": [str(path) for path in wrf_files],
        },
        "hours": [],
        "per_field_summary": {},
        "first_divergence": None,
        "blocked_reason": (
            "Need at least one GPU wrfout and one WRF reference wrfout with matching valid time. "
            "Rerun scripts/m7_daily_pipeline.py if the GPU /tmp output has been cleaned."
        ),
        "wall_clock_seconds": time.perf_counter() - start,
    }


def compare_hourly(
    *,
    case: str,
    domain: str,
    gpu_files: list[Path],
    wrf_files: list[Path],
    gpu_provenance: dict[str, Any],
    wrf_root: Path,
    max_hours: int | None,
    start: float,
) -> dict[str, Any]:
    gpu_by_time = _file_map(gpu_files, domain)
    wrf_by_time = _file_map(wrf_files, domain)
    common_times = sorted(set(gpu_by_time).intersection(wrf_by_time))
    if max_hours is not None:
        common_times = common_times[: int(max_hours)]

    rows: list[dict[str, Any]] = []
    for hour, stamp in enumerate(common_times, start=1):
        gpu_path = gpu_by_time[stamp]
        wrf_path = wrf_by_time[stamp]
        with Dataset(gpu_path, "r") as gpu_ds, Dataset(wrf_path, "r") as wrf_ds:
            fields = {spec.output_name: _compare_field(gpu_ds, wrf_ds, spec) for spec in FIELD_SPECS}
        rows.append(
            {
                "hour": int(hour),
                "valid_time_utc": stamp.replace("_", "T") + "+00:00",
                "gpu_path": str(gpu_path),
                "wrf_path": str(wrf_path),
                "fields": fields,
            }
        )

    summary = _field_summary(rows)
    first = _first_divergence(rows)
    missing_inputs = {
        "gpu_only_valid_times": sorted(set(gpu_by_time) - set(wrf_by_time)),
        "wrf_only_valid_times": sorted(set(wrf_by_time) - set(gpu_by_time)),
    }
    return {
        "trace_version": "wrfout-hourly-1.0",
        "case": case,
        "domain": domain,
        "scope_pivot": "hour-by-hour wrfout GPU-vs-WRF divergence trace; no per-operator WRF Fortran recompilation",
        "commit": _git_commit(),
        "status": "PASS" if first is None and rows else "FAIL",
        "field_order": [spec.output_name for spec in FIELD_SPECS],
        "field_mapping": {
            spec.output_name: {
                "gpu_variable": spec.variable_name,
                "wrf_variable": spec.variable_name,
                "transform": spec.transform,
            }
            for spec in FIELD_SPECS
        },
        "gpu_source": {**gpu_provenance, "wrfout_files": [str(path) for path in gpu_files]},
        "wrf_reference": {
            "run_dir": str(wrf_root),
            "wrfout_file_count": len(wrf_files),
            "wrfout_files": [str(path) for path in wrf_files],
        },
        "matched_hour_count": len(rows),
        "missing_inputs": missing_inputs,
        "hours": rows,
        "per_field_summary": summary,
        "first_divergence": first,
        "notes": [
            "Only d02 is compared because the available GPU M7 operational artifact writes d02 wrfout files.",
            "theta is compared as canonical absolute potential temperature: WRF T is T+300 K; GPU T is auto-detected as absolute or perturbation before conversion.",
            "Differences are reported over the common overlapping array shape; shape mismatches are listed per field.",
        ],
        "wall_clock_seconds": time.perf_counter() - start,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--gpu-pipeline-run", type=Path, default=DEFAULT_GPU_PIPELINE_RUN)
    parser.add_argument("--gpu-root", type=Path, default=None)
    parser.add_argument("--wrf-root", type=Path, default=DEFAULT_WRF_ROOT)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    start = time.perf_counter()
    gpu_files, gpu_provenance = _load_gpu_files(args.gpu_pipeline_run, args.gpu_root, args.domain)
    wrf_files = _load_wrf_files(args.wrf_root, args.domain)
    if not gpu_files or not wrf_files or not set(_file_map(gpu_files, args.domain)).intersection(_file_map(wrf_files, args.domain)):
        payload = _missing_payload(
            case=args.case,
            domain=args.domain,
            gpu_files=gpu_files,
            wrf_files=wrf_files,
            gpu_provenance=gpu_provenance,
            wrf_root=args.wrf_root,
            start=start,
        )
    else:
        payload = compare_hourly(
            case=args.case,
            domain=args.domain,
            gpu_files=gpu_files,
            wrf_files=wrf_files,
            gpu_provenance=gpu_provenance,
            wrf_root=args.wrf_root,
            max_hours=args.hours,
            start=start,
        )
    _write_json(args.output, payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
