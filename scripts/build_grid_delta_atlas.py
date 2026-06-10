#!/usr/bin/env python3
"""Build the offline v0.14 CPU-WRF-vs-GPU Grid-Delta Atlas.

This tool compares existing WRF-style NetCDF ``wrfout`` files only. It does not
run WRF, JAX, CUDA, TOST, Switzerland generation, or model kernels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


SCHEMA_VERSION = 1
DEFAULT_PROOF_DIR = Path("proofs/v014/grid_delta_atlas")
DEFAULT_ASSET_DIR = Path("docs/assets/v014/grid_delta_atlas")
WRFOUT_RE = re.compile(
    r"^wrfout_(?P<domain>d\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})(?:\.nc)?$"
)
CASE_INIT_RE = re.compile(r"(20\d{6})_(\d{2})z", re.IGNORECASE)
TIME_METADATA_FIELDS = {"Times"}
DEFAULT_MANDATORY_CORE_FIELDS = (
    "T",
    "U",
    "V",
    "W",
    "QVAPOR",
    "P",
    "PH",
    "MU",
    "U10",
    "V10",
    "T2",
    "PSFC",
    "RAINC",
    "RAINNC",
)
DEFAULT_PLOT_CORE_FIELDS = (
    "T",
    "U",
    "V",
    "W",
    "QVAPOR",
    "P",
    "PH",
    "MU",
    "U10",
    "V10",
    "T2",
    "PSFC",
    "RAINNC",
    "RAINC",
)


def clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return clean_float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def parse_wrfout_name(path: Path) -> tuple[str, datetime] | None:
    match = WRFOUT_RE.match(path.name)
    if not match:
        return None
    valid = datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    return match.group("domain"), valid


def parse_init_from_text(text: str) -> datetime | None:
    match = CASE_INIT_RE.search(text)
    if not match:
        return None
    date_s, hour_s = match.groups()
    return datetime(
        int(date_s[:4]),
        int(date_s[4:6]),
        int(date_s[6:8]),
        int(hour_s),
        tzinfo=timezone.utc,
    )


def parse_iso_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def lead_hour(valid_time: datetime, init_time: datetime) -> int:
    return int(round((valid_time - init_time).total_seconds() / 3600.0))


def fmt_num(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        x = float(value)
        if x == 0.0:
            return "0"
        if abs(x) >= 10000.0 or abs(x) < 0.001:
            return f"{x:.3e}"
        return f"{x:.{digits}f}"
    return "NA"


def is_numeric_dtype(dtype: Any) -> bool:
    return np.dtype(dtype).kind in {"b", "i", "u", "f"}


def dims_without_time(var: Any) -> tuple[str, ...]:
    dims = tuple(str(dim) for dim in var.dimensions)
    if dims and dims[0] == "Time":
        return dims[1:]
    return dims


def shape_without_time(var: Any) -> tuple[int, ...]:
    shape = tuple(int(item) for item in var.shape)
    dims = tuple(str(dim) for dim in var.dimensions)
    if dims and dims[0] == "Time":
        return shape[1:]
    return shape


def variable_metadata(dataset: Dataset, name: str) -> dict[str, Any]:
    var = dataset.variables[name]
    return {
        "dims": list(dims_without_time(var)),
        "shape": list(shape_without_time(var)),
        "dtype": str(var.dtype),
        "units": str(getattr(var, "units", "")),
        "description": str(getattr(var, "description", getattr(var, "FieldType", ""))),
        "stagger": str(getattr(var, "stagger", "")),
    }


def read_variable(dataset: Dataset, name: str) -> np.ndarray:
    var = dataset.variables[name]
    raw = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
    if np.ma.isMaskedArray(raw):
        if is_numeric_dtype(raw.dtype):
            raw = raw.astype(np.float64, copy=False)
            return np.asarray(np.ma.filled(raw, np.nan))
        return np.asarray(np.ma.filled(raw, b""))
    return np.asarray(raw)


def metadata_key(meta: dict[str, Any]) -> str:
    return json.dumps(
        {"dims": meta["dims"], "shape": meta["shape"], "dtype": meta["dtype"]},
        sort_keys=True,
    )


def compatible_metadata(cpu_meta: dict[str, Any], gpu_meta: dict[str, Any]) -> tuple[bool, str | None]:
    if cpu_meta["shape"] != gpu_meta["shape"]:
        return False, "shape_mismatch"
    if cpu_meta["dims"] != gpu_meta["dims"]:
        return False, "dimension_name_mismatch"
    if not is_numeric_dtype(cpu_meta["dtype"]) or not is_numeric_dtype(gpu_meta["dtype"]):
        return False, "non_numeric"
    return True, None


def normalize_tolerance_record(value: Any) -> dict[str, float]:
    if isinstance(value, (int, float)):
        return {"max_abs": float(value)}
    if not isinstance(value, dict):
        return {}
    aliases = {
        "rmse": "rmse",
        "mae": "mae",
        "bias_abs": "bias_abs",
        "p50": "p50_abs",
        "p50_abs": "p50_abs",
        "p95": "p95_abs",
        "p95_abs": "p95_abs",
        "p99": "p99_abs",
        "p99_abs": "p99_abs",
        "p999": "p999_abs",
        "p99.9": "p999_abs",
        "p99_9_abs": "p999_abs",
        "p999_abs": "p999_abs",
        "max_abs": "max_abs",
        "mean_abs_rel": "mean_abs_rel",
        "p95_abs_rel": "p95_abs_rel",
        "max_abs_rel": "max_abs_rel",
        "correlation_min": "correlation_min",
        "pearson_min": "correlation_min",
        "finite_pair_fraction_min": "finite_pair_fraction_min",
    }
    out: dict[str, float] = {}
    for src, dst in aliases.items():
        if src in value:
            converted = clean_float(value[src])
            if converted is not None:
                out[dst] = converted
    return out


def load_tolerances(path: Path | None) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if path is None:
        return {}, {
            "supplied": False,
            "policy": "No tolerance manifest supplied; differences are report-only except inventory/nonfinite failures.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw = None
        for key in ("fields", "variables", "tolerances"):
            if isinstance(payload.get(key), dict):
                raw = payload[key]
                break
        if raw is None:
            raw = payload
    else:
        raw = {}
    normalized = {str(name): normalize_tolerance_record(record) for name, record in raw.items()}
    normalized = {name: spec for name, spec in normalized.items() if spec}
    return normalized, {
        "supplied": True,
        "path": str(path),
        "field_count": len(normalized),
        "policy": "Tolerances are read before comparison and never tuned by this script.",
    }


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    cpu_dir: Path
    gpu_dir: Path
    domains: tuple[str, ...] = ()
    init_time: datetime | None = None


@dataclass(frozen=True)
class PairRecord:
    case_id: str
    domain: str
    valid_time: datetime
    lead_h: int
    cpu_file: Path
    gpu_file: Path
    init_time: datetime
    init_time_source: str

    def context(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "lead_h": int(self.lead_h),
            "valid_time_utc": self.valid_time.isoformat(),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            **self.context(),
            "init_time_utc": self.init_time.isoformat(),
            "init_time_source": self.init_time_source,
            "cpu_file": str(self.cpu_file),
            "gpu_file": str(self.gpu_file),
        }


@dataclass
class StatsAccumulator:
    rel_floor: float = 1.0e-12
    count: int = 0
    finite_cpu: int = 0
    finite_gpu: int = 0
    finite_pair: int = 0
    sum_diff: float = 0.0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0
    sum_gpu: float = 0.0
    sum_cpu: float = 0.0
    sum_gpu_sq: float = 0.0
    sum_cpu_sq: float = 0.0
    sum_gpu_cpu: float = 0.0
    relative_count: int = 0
    sum_abs_rel: float = 0.0
    max_abs_rel: float = 0.0
    abs_chunks: list[np.ndarray] = field(default_factory=list)
    rel_chunks: list[np.ndarray] = field(default_factory=list)
    worst: dict[str, Any] | None = None

    def update(self, gpu: np.ndarray, cpu: np.ndarray, context: dict[str, Any]) -> dict[str, int]:
        g = np.asarray(gpu, dtype=np.float64)
        c = np.asarray(cpu, dtype=np.float64)
        if g.shape != c.shape:
            raise ValueError(f"shape mismatch {g.shape} vs {c.shape}")
        total = int(g.size)
        cpu_finite = np.isfinite(c)
        gpu_finite = np.isfinite(g)
        valid = cpu_finite & gpu_finite
        finite_pair = int(np.sum(valid))
        finite_cpu = int(np.sum(cpu_finite))
        finite_gpu = int(np.sum(gpu_finite))

        self.count += total
        self.finite_cpu += finite_cpu
        self.finite_gpu += finite_gpu
        self.finite_pair += finite_pair
        if finite_pair == 0:
            return {
                "count": total,
                "finite_cpu": finite_cpu,
                "finite_gpu": finite_gpu,
                "finite_pair": 0,
                "nonfinite_cpu": total - finite_cpu,
                "nonfinite_gpu": total - finite_gpu,
            }

        gv = g[valid]
        cv = c[valid]
        diff = gv - cv
        abs_diff = np.abs(diff)
        self.sum_diff += float(np.sum(diff, dtype=np.float64))
        self.sum_abs += float(np.sum(abs_diff, dtype=np.float64))
        self.sum_sq += float(np.sum(diff * diff, dtype=np.float64))
        local_max = float(np.max(abs_diff))
        self.max_abs = max(self.max_abs, local_max)
        self.sum_gpu += float(np.sum(gv, dtype=np.float64))
        self.sum_cpu += float(np.sum(cv, dtype=np.float64))
        self.sum_gpu_sq += float(np.sum(gv * gv, dtype=np.float64))
        self.sum_cpu_sq += float(np.sum(cv * cv, dtype=np.float64))
        self.sum_gpu_cpu += float(np.sum(gv * cv, dtype=np.float64))
        self.abs_chunks.append(abs_diff.astype(np.float64, copy=False))

        denom = np.maximum(np.maximum(np.abs(gv), np.abs(cv)), self.rel_floor)
        meaningful = denom > self.rel_floor
        if np.any(meaningful):
            rel = abs_diff[meaningful] / denom[meaningful]
            self.relative_count += int(rel.size)
            self.sum_abs_rel += float(np.sum(rel, dtype=np.float64))
            self.max_abs_rel = max(self.max_abs_rel, float(np.max(rel)))
            self.rel_chunks.append(rel.astype(np.float64, copy=False))

        if local_max >= float((self.worst or {}).get("abs_diff", -1.0)):
            abs_arr = np.full(g.shape, -1.0, dtype=np.float64)
            abs_arr[valid] = np.abs(g[valid] - c[valid])
            flat_idx = int(np.argmax(abs_arr))
            idx = tuple(int(item) for item in np.unravel_index(flat_idx, g.shape))
            self.worst = {
                **context,
                "index": list(idx),
                "abs_diff": clean_float(abs_arr[idx]),
                "diff": clean_float(g[idx] - c[idx]),
                "gpu_value": clean_float(g[idx]),
                "cpu_value": clean_float(c[idx]),
            }

        return {
            "count": total,
            "finite_cpu": finite_cpu,
            "finite_gpu": finite_gpu,
            "finite_pair": finite_pair,
            "nonfinite_cpu": total - finite_cpu,
            "nonfinite_gpu": total - finite_gpu,
        }

    def finish(self) -> dict[str, Any]:
        if self.finite_pair == 0:
            return {
                "count": int(self.count),
                "finite_cpu_count": int(self.finite_cpu),
                "finite_gpu_count": int(self.finite_gpu),
                "finite_pair_count": 0,
                "nonfinite_cpu_count": int(self.count - self.finite_cpu),
                "nonfinite_gpu_count": int(self.count - self.finite_gpu),
                "finite_pair_fraction": 0.0 if self.count else None,
                "bias": None,
                "rmse": None,
                "mae": None,
                "max_abs": None,
                "p50_abs": None,
                "p95_abs": None,
                "p99_abs": None,
                "p999_abs": None,
                "safe_relative": {
                    "floor": self.rel_floor,
                    "count": 0,
                    "mean_abs_rel": None,
                    "p95_abs_rel": None,
                    "max_abs_rel": None,
                },
                "correlation": None,
                "worst": self.worst,
            }
        abs_all = np.concatenate(self.abs_chunks) if self.abs_chunks else np.asarray([], dtype=np.float64)
        rel_all = np.concatenate(self.rel_chunks) if self.rel_chunks else np.asarray([], dtype=np.float64)
        return {
            "count": int(self.count),
            "finite_cpu_count": int(self.finite_cpu),
            "finite_gpu_count": int(self.finite_gpu),
            "finite_pair_count": int(self.finite_pair),
            "nonfinite_cpu_count": int(self.count - self.finite_cpu),
            "nonfinite_gpu_count": int(self.count - self.finite_gpu),
            "finite_pair_fraction": clean_float(self.finite_pair / self.count) if self.count else None,
            "bias": clean_float(self.sum_diff / self.finite_pair),
            "rmse": clean_float(math.sqrt(self.sum_sq / self.finite_pair)),
            "mae": clean_float(self.sum_abs / self.finite_pair),
            "max_abs": clean_float(self.max_abs),
            "p50_abs": clean_float(np.percentile(abs_all, 50)) if abs_all.size else None,
            "p95_abs": clean_float(np.percentile(abs_all, 95)) if abs_all.size else None,
            "p99_abs": clean_float(np.percentile(abs_all, 99)) if abs_all.size else None,
            "p999_abs": clean_float(np.percentile(abs_all, 99.9)) if abs_all.size else None,
            "safe_relative": {
                "floor": self.rel_floor,
                "count": int(self.relative_count),
                "mean_abs_rel": clean_float(self.sum_abs_rel / self.relative_count) if self.relative_count else None,
                "p95_abs_rel": clean_float(np.percentile(rel_all, 95)) if rel_all.size else None,
                "max_abs_rel": clean_float(self.max_abs_rel) if self.relative_count else None,
            },
            "correlation": pearson_from_sums(
                self.finite_pair,
                self.sum_gpu,
                self.sum_cpu,
                self.sum_gpu_sq,
                self.sum_cpu_sq,
                self.sum_gpu_cpu,
            ),
            "worst": self.worst,
        }


def pearson_from_sums(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float | None:
    if n < 2:
        return None
    cov = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if vx <= 0.0 or vy <= 0.0:
        return None
    return clean_float(cov / math.sqrt(vx * vy))


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0.0:
        return None
    return clean_float(float(np.sum((x - x_mean) * (y - y_mean)) / denom))


def max_lead_jump(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    ordered = sorted((row for row in rows if row.get(metric) is not None), key=lambda row: (row["lead_h"], row.get("case_id", "")))
    if len(ordered) < 2:
        return None
    best: dict[str, Any] | None = None
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        jump = abs(float(cur[metric]) - float(prev[metric]))
        if best is None or jump > float(best["abs_jump"]):
            best = {
                "from_lead_h": int(prev["lead_h"]),
                "to_lead_h": int(cur["lead_h"]),
                "abs_jump": clean_float(jump),
                "from_value": clean_float(prev[metric]),
                "to_value": clean_float(cur[metric]),
            }
    return best


def early_late_delta(rows: list[dict[str, Any]], metric: str) -> float | None:
    ordered = sorted((row for row in rows if row.get(metric) is not None), key=lambda row: row["lead_h"])
    if len(ordered) < 2:
        return None
    midpoint = max(1, len(ordered) // 2)
    early = [float(row[metric]) for row in ordered[:midpoint]]
    late = [float(row[metric]) for row in ordered[midpoint:]]
    if not late:
        late = [float(ordered[-1][metric])]
    return clean_float(float(np.mean(late) - np.mean(early)))


def drift_summary(by_lead: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in by_lead if row.get("rmse") is not None]
    leads = [float(row["lead_h"]) for row in rows]
    rmses = [float(row["rmse"]) for row in rows]
    biases = [float(row["bias"]) for row in rows if row.get("bias") is not None]
    bias_leads = [float(row["lead_h"]) for row in rows if row.get("bias") is not None]
    worst = max(rows, key=lambda row: float(row["rmse"])) if rows else None
    return {
        "lead_count": len(rows),
        "rmse_slope_per_lead_hour": linear_slope(leads, rmses),
        "bias_slope_per_lead_hour": linear_slope(bias_leads, biases),
        "max_rmse_lead_jump": max_lead_jump(rows, "rmse"),
        "max_bias_lead_jump": max_lead_jump(rows, "bias"),
        "late_minus_early_rmse": early_late_delta(rows, "rmse"),
        "late_minus_early_bias": early_late_delta(rows, "bias"),
        "worst_lead_h": int(worst["lead_h"]) if worst else None,
        "worst_lead_rmse": worst.get("rmse") if worst else None,
        "worst_lead_bias": worst.get("bias") if worst else None,
    }


def apply_tolerance(metrics: dict[str, Any], spec: dict[str, float] | None) -> dict[str, Any]:
    if not spec:
        return {"supplied": False, "pass": None, "failures": []}
    checks: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    direct = ("rmse", "mae", "p50_abs", "p95_abs", "p99_abs", "p999_abs", "max_abs")
    for metric in direct:
        if metric not in spec:
            continue
        value = metrics.get(metric)
        limit = spec[metric]
        passed = bool(value is not None and float(value) <= limit)
        checks[metric] = {"value": value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": metric, "value": value, "limit": limit})
    if "bias_abs" in spec:
        value = metrics.get("bias")
        abs_value = abs(float(value)) if value is not None else None
        limit = spec["bias_abs"]
        passed = bool(abs_value is not None and abs_value <= limit)
        checks["bias_abs"] = {"value": abs_value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": "bias_abs", "value": abs_value, "limit": limit})
    rel = metrics.get("safe_relative", {})
    for metric in ("mean_abs_rel", "p95_abs_rel", "max_abs_rel"):
        if metric not in spec:
            continue
        value = rel.get(metric)
        limit = spec[metric]
        passed = bool(value is not None and float(value) <= limit)
        checks[metric] = {"value": value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": metric, "value": value, "limit": limit})
    if "correlation_min" in spec:
        value = metrics.get("correlation")
        limit = spec["correlation_min"]
        passed = bool(value is not None and float(value) >= limit)
        checks["correlation_min"] = {"value": value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": "correlation_min", "value": value, "limit": limit})
    if "finite_pair_fraction_min" in spec:
        value = metrics.get("finite_pair_fraction")
        limit = spec["finite_pair_fraction_min"]
        passed = bool(value is not None and float(value) >= limit)
        checks["finite_pair_fraction_min"] = {"value": value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": "finite_pair_fraction_min", "value": value, "limit": limit})
    return {"supplied": True, "spec": spec, "checks": checks, "pass": not failures, "failures": failures}


def discover_wrfouts(root: Path, domains: tuple[str, ...] = ()) -> tuple[dict[str, dict[datetime, Path]], list[dict[str, Any]]]:
    found: dict[str, dict[datetime, Path]] = {}
    duplicates: list[dict[str, Any]] = []
    wanted = set(domains)
    for path in sorted(root.glob("wrfout_d??_*")):
        if not path.is_file():
            continue
        parsed = parse_wrfout_name(path)
        if parsed is None:
            continue
        domain, valid = parsed
        if wanted and domain not in wanted:
            continue
        by_time = found.setdefault(domain, {})
        if valid in by_time:
            duplicates.append(
                {
                    "domain": domain,
                    "valid_time_utc": valid.isoformat(),
                    "kept": str(by_time[valid]),
                    "ignored": str(path),
                }
            )
            continue
        by_time[valid] = path
    return found, duplicates


def infer_init_time(
    spec: CaseSpec,
    domain: str,
    cpu_map: dict[datetime, Path],
    gpu_map: dict[datetime, Path],
) -> tuple[datetime, str]:
    if spec.init_time is not None:
        return spec.init_time, "case-spec"
    for text in (spec.case_id, str(spec.gpu_dir), str(spec.cpu_dir)):
        parsed = parse_init_from_text(text)
        if parsed is not None:
            return parsed, "case-id-or-directory"
    all_times = sorted(set(cpu_map) | set(gpu_map))
    if not all_times:
        raise ValueError(f"case {spec.case_id} domain {domain} has no wrfout files")
    return all_times[0], "earliest-wrfout-fallback"


def domain_coverage(
    spec: CaseSpec,
    domain: str,
    cpu_map: dict[datetime, Path],
    gpu_map: dict[datetime, Path],
    init_time: datetime,
    init_source: str,
    pairs: list[PairRecord],
    min_lead: int | None,
    max_lead: int | None,
) -> dict[str, Any]:
    common_all = sorted(set(cpu_map) & set(gpu_map))
    filtered_common: list[datetime] = []
    for valid in common_all:
        lead = lead_hour(valid, init_time)
        if min_lead is not None and lead < min_lead:
            continue
        if max_lead is not None and lead > max_lead:
            continue
        filtered_common.append(valid)
    return {
        "case_id": spec.case_id,
        "domain": domain,
        "init_time_utc": init_time.isoformat(),
        "init_time_source": init_source,
        "cpu_file_count": len(cpu_map),
        "gpu_file_count": len(gpu_map),
        "common_file_count_before_lead_filter": len(common_all),
        "paired_file_count": sum(1 for pair in pairs if pair.case_id == spec.case_id and pair.domain == domain),
        "lead_filter": {"min_lead": min_lead, "max_lead": max_lead},
        "paired_leads_h": [pair.lead_h for pair in pairs if pair.case_id == spec.case_id and pair.domain == domain],
        "unmatched_cpu_count": len(set(cpu_map) - set(gpu_map)),
        "unmatched_gpu_count": len(set(gpu_map) - set(cpu_map)),
        "unmatched_cpu_times_utc": [item.isoformat() for item in sorted(set(cpu_map) - set(gpu_map))],
        "unmatched_gpu_times_utc": [item.isoformat() for item in sorted(set(gpu_map) - set(cpu_map))],
        "lead_filtered_common_times_utc": [item.isoformat() for item in common_all if item not in filtered_common],
    }


def build_pairs(case_specs: list[CaseSpec], min_lead: int | None, max_lead: int | None) -> tuple[list[PairRecord], dict[str, Any]]:
    pairs: list[PairRecord] = []
    cases_json: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    for spec in case_specs:
        cpu_files, cpu_dups = discover_wrfouts(spec.cpu_dir, spec.domains)
        gpu_files, gpu_dups = discover_wrfouts(spec.gpu_dir, spec.domains)
        duplicate_records.extend({**item, "case_id": spec.case_id, "side": "cpu"} for item in cpu_dups)
        duplicate_records.extend({**item, "case_id": spec.case_id, "side": "gpu"} for item in gpu_dups)
        domains = sorted(spec.domains or tuple(set(cpu_files) | set(gpu_files)))
        domain_rows = []
        for domain in domains:
            cpu_map = cpu_files.get(domain, {})
            gpu_map = gpu_files.get(domain, {})
            if not cpu_map and not gpu_map:
                continue
            init_time, init_source = infer_init_time(spec, domain, cpu_map, gpu_map)
            for valid in sorted(set(cpu_map) & set(gpu_map)):
                lead = lead_hour(valid, init_time)
                if min_lead is not None and lead < min_lead:
                    continue
                if max_lead is not None and lead > max_lead:
                    continue
                pairs.append(
                    PairRecord(
                        case_id=spec.case_id,
                        domain=domain,
                        valid_time=valid,
                        lead_h=lead,
                        cpu_file=cpu_map[valid],
                        gpu_file=gpu_map[valid],
                        init_time=init_time,
                        init_time_source=init_source,
                    )
                )
            domain_rows.append(domain_coverage(spec, domain, cpu_map, gpu_map, init_time, init_source, pairs, min_lead, max_lead))
        cases_json.append(
            {
                "case_id": spec.case_id,
                "cpu_dir": str(spec.cpu_dir),
                "gpu_dir": str(spec.gpu_dir),
                "domains": domains,
                "domains_coverage": domain_rows,
            }
        )
    pairs = sorted(pairs, key=lambda item: (item.case_id, item.domain, item.valid_time.isoformat()))
    coverage = {
        "case_count": len(case_specs),
        "paired_file_count": len(pairs),
        "domains": sorted({pair.domain for pair in pairs}),
        "lead_hours": sorted({int(pair.lead_h) for pair in pairs}),
        "cases": cases_json,
        "duplicate_wrfouts": duplicate_records,
        "pairs": [pair.to_json() for pair in pairs],
    }
    return pairs, coverage


def case_specs_from_args(args: argparse.Namespace) -> list[CaseSpec]:
    if args.case_json is not None:
        payload = json.loads(args.case_json.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_cases, list):
            raise SystemExit("--case-json must contain a list or an object with a 'cases' list")
        specs = []
        for i, raw in enumerate(raw_cases, start=1):
            if not isinstance(raw, dict):
                raise SystemExit(f"case {i} in --case-json is not an object")
            cpu_dir_value = raw.get("cpu_dir") or raw.get("cpu_root")
            gpu_dir_value = raw.get("gpu_dir") or raw.get("gpu_root")
            if not cpu_dir_value or not gpu_dir_value:
                raise SystemExit(f"case {i} must define cpu_dir and gpu_dir")
            cpu_dir = Path(str(cpu_dir_value))
            gpu_dir = Path(str(gpu_dir_value))
            domains_raw = raw.get("domains", raw.get("domain", ()))
            if isinstance(domains_raw, str):
                domains = (domains_raw,)
            else:
                domains = tuple(str(item) for item in domains_raw)
            init_value = raw.get("init_time_utc", raw.get("init_time", raw.get("init")))
            init_time = parse_iso_time(str(init_value)) if init_value else None
            specs.append(
                CaseSpec(
                    case_id=str(raw.get("case_id", raw.get("run_id", f"case_{i:03d}"))),
                    cpu_dir=cpu_dir,
                    gpu_dir=gpu_dir,
                    domains=domains,
                    init_time=init_time,
                )
            )
        return specs

    if args.cpu_dir is None or args.gpu_dir is None:
        raise SystemExit("provide --cpu-dir/--gpu-dir or --case-json")
    init_time = parse_iso_time(args.init) if args.init else None
    case_id = args.case_id or args.gpu_dir.name or args.cpu_dir.name or "single_case"
    return [
        CaseSpec(
            case_id=case_id,
            cpu_dir=args.cpu_dir,
            gpu_dir=args.gpu_dir,
            domains=tuple(args.domain or ()),
            init_time=init_time,
        )
    ]


def context_with_pair(pair: PairRecord) -> dict[str, Any]:
    return {
        **pair.context(),
        "cpu_file": str(pair.cpu_file),
        "gpu_file": str(pair.gpu_file),
    }


def build_field_union(pairs: list[PairRecord]) -> tuple[list[str], dict[str, Any]]:
    cpu_union: set[str] = set()
    gpu_union: set[str] = set()
    first_metadata: dict[str, dict[str, Any]] = {}
    variable_presence: dict[str, dict[str, int]] = {}
    for pair in pairs:
        with Dataset(pair.cpu_file, "r") as cpu_ds, Dataset(pair.gpu_file, "r") as gpu_ds:
            cpu_names = set(cpu_ds.variables)
            gpu_names = set(gpu_ds.variables)
            cpu_union.update(cpu_names)
            gpu_union.update(gpu_names)
            for name in sorted(cpu_names | gpu_names):
                presence = variable_presence.setdefault(name, {"cpu_present_pairs": 0, "gpu_present_pairs": 0})
                if name in cpu_names:
                    presence["cpu_present_pairs"] += 1
                if name in gpu_names:
                    presence["gpu_present_pairs"] += 1
            for name in sorted(cpu_names | gpu_names):
                if name in first_metadata:
                    continue
                meta: dict[str, Any] = {"first_seen": pair.context()}
                if name in cpu_names:
                    meta["cpu"] = variable_metadata(cpu_ds, name)
                if name in gpu_names:
                    meta["gpu"] = variable_metadata(gpu_ds, name)
                first_metadata[name] = meta
    union = sorted(cpu_union | gpu_union)
    inventory = {
        "cpu_variable_count": len(cpu_union),
        "gpu_variable_count": len(gpu_union),
        "union_variable_count": len(union),
        "common_variable_count": len(cpu_union & gpu_union),
        "cpu_only_count": len(cpu_union - gpu_union),
        "gpu_only_count": len(gpu_union - cpu_union),
        "cpu_only": sorted(cpu_union - gpu_union),
        "gpu_only": sorted(gpu_union - cpu_union),
        "common": sorted(cpu_union & gpu_union),
        "variable_presence": variable_presence,
        "first_metadata": first_metadata,
    }
    return union, inventory


def grouped_lead_key(pair: PairRecord) -> tuple[int, str]:
    return int(pair.lead_h), pair.valid_time.isoformat()


def compare_field(
    name: str,
    pairs: list[PairRecord],
    first_metadata: dict[str, Any],
    tolerance_spec: dict[str, float] | None,
    rel_floor: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    overall = StatsAccumulator(rel_floor=rel_floor)
    by_lead_accs: dict[tuple[int, str], StatsAccumulator] = {}
    by_pair: list[dict[str, Any]] = []
    issues = {
        "missing": [],
        "non_numeric": [],
        "shape_mismatch": [],
        "dimension_name_mismatch": [],
        "nonfinite": [],
        "metadata_variants": [],
    }
    seen_meta_variants: set[str] = set()

    for pair in pairs:
        context = context_with_pair(pair)
        with Dataset(pair.cpu_file, "r") as cpu_ds, Dataset(pair.gpu_file, "r") as gpu_ds:
            cpu_present = name in cpu_ds.variables
            gpu_present = name in gpu_ds.variables
            if not cpu_present or not gpu_present:
                issues["missing"].append(
                    {
                        **context,
                        "field": name,
                        "missing_in": [side for side, present in (("cpu", cpu_present), ("gpu", gpu_present)) if not present],
                    }
                )
                continue

            cpu_meta = variable_metadata(cpu_ds, name)
            gpu_meta = variable_metadata(gpu_ds, name)
            variant_key = metadata_key(cpu_meta) + "|" + metadata_key(gpu_meta)
            if variant_key not in seen_meta_variants:
                seen_meta_variants.add(variant_key)
                issues["metadata_variants"].append({"field": name, "cpu": cpu_meta, "gpu": gpu_meta, "first_seen": context})

            ok, reason = compatible_metadata(cpu_meta, gpu_meta)
            if not ok:
                record = {"field": name, "reason": reason, "cpu": cpu_meta, "gpu": gpu_meta, **context}
                if reason == "non_numeric":
                    issues["non_numeric"].append(record)
                elif reason == "dimension_name_mismatch":
                    issues["dimension_name_mismatch"].append(record)
                else:
                    issues["shape_mismatch"].append(record)
                continue
            cpu = read_variable(cpu_ds, name)
            gpu = read_variable(gpu_ds, name)

        pair_acc = StatsAccumulator(rel_floor=rel_floor)
        counts = pair_acc.update(gpu, cpu, context)
        lead_acc = by_lead_accs.setdefault(grouped_lead_key(pair), StatsAccumulator(rel_floor=rel_floor))
        lead_acc.update(gpu, cpu, context)
        overall.update(gpu, cpu, context)
        pair_metrics = pair_acc.finish()
        by_pair.append(
            {
                **pair.context(),
                "count": counts["count"],
                "finite_cpu_count": counts["finite_cpu"],
                "finite_gpu_count": counts["finite_gpu"],
                "finite_pair_count": counts["finite_pair"],
                "nonfinite_cpu_count": counts["nonfinite_cpu"],
                "nonfinite_gpu_count": counts["nonfinite_gpu"],
                "rmse": pair_metrics["rmse"],
                "bias": pair_metrics["bias"],
                "max_abs": pair_metrics["max_abs"],
            }
        )
        if counts["nonfinite_cpu"] or counts["nonfinite_gpu"]:
            issues["nonfinite"].append(
                {
                    **context,
                    "field": name,
                    "count": counts["count"],
                    "nonfinite_cpu_count": counts["nonfinite_cpu"],
                    "nonfinite_gpu_count": counts["nonfinite_gpu"],
                }
            )

    by_lead = []
    for (lead_h, valid_time_utc), acc in sorted(by_lead_accs.items(), key=lambda item: item[0]):
        metrics = acc.finish()
        if metrics.get("worst"):
            metrics["worst"] = {"field": name, **metrics["worst"]}
        by_lead.append({"lead_h": lead_h, "valid_time_utc": valid_time_utc, **metrics})
    overall_metrics = overall.finish()
    if overall_metrics.get("worst"):
        overall_metrics["worst"] = {"field": name, **overall_metrics["worst"]}
    tolerance_result = apply_tolerance(overall_metrics, tolerance_spec)
    compared = overall_metrics["finite_pair_count"] > 0
    field_summary = {
        "field": name,
        "status": "compared" if compared else "not_compared",
        "metadata": first_metadata,
        "overall": overall_metrics,
        "by_lead": by_lead,
        "by_pair": by_pair,
        "drift": drift_summary(by_lead),
        "tolerance_result": tolerance_result,
        "issue_counts": {
            "missing": len(issues["missing"]),
            "non_numeric": len(issues["non_numeric"]),
            "shape_mismatch": len(issues["shape_mismatch"]),
            "dimension_name_mismatch": len(issues["dimension_name_mismatch"]),
            "nonfinite": len(issues["nonfinite"]),
            "metadata_variants": len(issues["metadata_variants"]),
        },
        "issues": {key: value for key, value in issues.items() if key != "metadata_variants" and value},
        "metadata_variants": issues["metadata_variants"][:8],
    }
    return field_summary, issues


def field_score(summary: dict[str, Any], tolerances_supplied: bool) -> float:
    tol = summary.get("tolerance_result", {})
    if tolerances_supplied and tol.get("supplied"):
        score = 0.0
        for failure in tol.get("failures", []):
            value = failure.get("value")
            limit = failure.get("limit")
            if isinstance(value, (int, float)) and isinstance(limit, (int, float)) and abs(float(limit)) > 0.0:
                score = max(score, abs(float(value)) / abs(float(limit)))
            elif value is not None:
                score = max(score, 1.0)
        return score
    overall = summary.get("overall", {})
    return max(
        float(overall.get("max_abs") or 0.0),
        float(overall.get("p99_abs") or 0.0),
        float(overall.get("rmse") or 0.0),
    )


def check_mandatory_fields(
    pairs: list[PairRecord],
    mandatory_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not mandatory_fields:
        return []
    missing: list[dict[str, Any]] = []
    for pair in pairs:
        with Dataset(pair.cpu_file, "r") as cpu_ds, Dataset(pair.gpu_file, "r") as gpu_ds:
            cpu_names = set(cpu_ds.variables)
            gpu_names = set(gpu_ds.variables)
        for field_name in mandatory_fields:
            cpu_present = field_name in cpu_names
            gpu_present = field_name in gpu_names
            if not cpu_present or not gpu_present:
                missing.append(
                    {
                        **context_with_pair(pair),
                        "field": field_name,
                        "missing_in": [side for side, present in (("cpu", cpu_present), ("gpu", gpu_present)) if not present],
                    }
                )
    return missing


def build_lead_stability(field_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    by_lead: dict[int, list[dict[str, Any]]] = {}
    for field_name, summary in field_metrics.items():
        for row in summary.get("by_lead", []):
            if row.get("rmse") is None:
                continue
            by_lead.setdefault(int(row["lead_h"]), []).append(
                {
                    "field": field_name,
                    "rmse": row.get("rmse"),
                    "bias": row.get("bias"),
                    "max_abs": row.get("max_abs"),
                    "p99_abs": row.get("p99_abs"),
                }
            )
    out = []
    for lead_h, rows in sorted(by_lead.items()):
        worst_rmse = max(rows, key=lambda row: float(row.get("rmse") or 0.0)) if rows else None
        worst_max = max(rows, key=lambda row: float(row.get("max_abs") or 0.0)) if rows else None
        out.append(
            {
                "lead_h": lead_h,
                "compared_field_count": len(rows),
                "mean_rmse_over_fields": clean_float(np.mean([float(row["rmse"]) for row in rows])) if rows else None,
                "worst_rmse_field": worst_rmse,
                "worst_max_abs_field": worst_max,
            }
        )
    return out


def build_summary(
    pairs: list[PairRecord],
    pairing: dict[str, Any],
    inventory: dict[str, Any],
    field_metrics: dict[str, Any],
    issue_records: dict[str, list[dict[str, Any]]],
    tolerances_supplied: bool,
    mandatory_missing: list[dict[str, Any]],
) -> dict[str, Any]:
    compared_fields = [name for name, item in field_metrics.items() if item.get("status") == "compared"]
    ranked = sorted(
        (
            {
                "field": name,
                "status": item.get("status"),
                "rmse": item.get("overall", {}).get("rmse"),
                "bias": item.get("overall", {}).get("bias"),
                "p50_abs": item.get("overall", {}).get("p50_abs"),
                "p95_abs": item.get("overall", {}).get("p95_abs"),
                "p99_abs": item.get("overall", {}).get("p99_abs"),
                "p999_abs": item.get("overall", {}).get("p999_abs"),
                "max_abs": item.get("overall", {}).get("max_abs"),
                "correlation": item.get("overall", {}).get("correlation"),
                "worst": item.get("overall", {}).get("worst"),
                "score": field_score(item, tolerances_supplied),
                "tolerance_pass": item.get("tolerance_result", {}).get("pass"),
                "issue_counts": item.get("issue_counts", {}),
            }
            for name, item in field_metrics.items()
            if item.get("status") == "compared"
        ),
        key=lambda row: (float(row["score"]), row["field"]),
        reverse=True,
    )
    drift_ranked = sorted(
        (
            {
                "field": name,
                "rmse_slope_per_lead_hour": item.get("drift", {}).get("rmse_slope_per_lead_hour"),
                "bias_slope_per_lead_hour": item.get("drift", {}).get("bias_slope_per_lead_hour"),
                "late_minus_early_rmse": item.get("drift", {}).get("late_minus_early_rmse"),
                "late_minus_early_bias": item.get("drift", {}).get("late_minus_early_bias"),
                "worst_lead_h": item.get("drift", {}).get("worst_lead_h"),
                "worst_lead_rmse": item.get("drift", {}).get("worst_lead_rmse"),
                "score": abs(float(item.get("drift", {}).get("rmse_slope_per_lead_hour") or 0.0))
                + abs(float(item.get("drift", {}).get("bias_slope_per_lead_hour") or 0.0)),
            }
            for name, item in field_metrics.items()
            if item.get("status") == "compared"
        ),
        key=lambda row: (float(row["score"]), row["field"]),
        reverse=True,
    )
    tolerance_failures = [
        {"field": name, "failures": item.get("tolerance_result", {}).get("failures", [])}
        for name, item in field_metrics.items()
        if item.get("tolerance_result", {}).get("pass") is False
    ]
    gpu_nonfinite = [
        item for item in issue_records["nonfinite"] if int(item.get("nonfinite_gpu_count") or 0) > 0
    ]
    inventory_failure_count = (
        len(mandatory_missing)
        + len(gpu_nonfinite)
        + len(issue_records["shape_mismatch"])
        + len(issue_records["dimension_name_mismatch"])
    )
    if inventory_failure_count:
        verdict = "FAIL_INVENTORY_OR_NONFINITE"
    elif tolerance_failures:
        verdict = "FAIL_TOLERANCE"
    elif tolerances_supplied:
        verdict = "PASS"
    else:
        verdict = "REPORT_ONLY_NO_TOLERANCE_MANIFEST"
    return {
        "schema": "grid-delta-atlas-summary-v1",
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "pairing": {
            "paired_file_count": pairing["paired_file_count"],
            "case_count": pairing["case_count"],
            "domains": pairing["domains"],
            "lead_hours": pairing["lead_hours"],
        },
        "inventory": {
            "union_variable_count": inventory["union_variable_count"],
            "common_variable_count": inventory["common_variable_count"],
            "cpu_only_count": inventory["cpu_only_count"],
            "gpu_only_count": inventory["gpu_only_count"],
            "compared_numeric_field_count": len(compared_fields),
            "not_compared_field_count": len(field_metrics) - len(compared_fields),
            "missing_record_count": len(issue_records["missing"]),
            "non_numeric_record_count": len(issue_records["non_numeric"]),
            "nonfinite_record_count": len(issue_records["nonfinite"]),
            "shape_mismatch_record_count": len(issue_records["shape_mismatch"]),
            "dimension_name_mismatch_record_count": len(issue_records["dimension_name_mismatch"]),
            "mandatory_missing_record_count": len(mandatory_missing),
        },
        "top_fields_by_severity": ranked[:40],
        "top_drift_signals": drift_ranked[:40],
        "lead_stability": build_lead_stability(field_metrics),
        "tolerance_failure_count": len(tolerance_failures),
        "tolerance_failures": tolerance_failures[:40],
        "mandatory_missing": mandatory_missing[:200],
        "issue_samples": {
            "missing": issue_records["missing"][:40],
            "non_numeric": issue_records["non_numeric"][:40],
            "nonfinite": issue_records["nonfinite"][:40],
            "shape_mismatch": issue_records["shape_mismatch"][:40],
            "dimension_name_mismatch": issue_records["dimension_name_mismatch"][:40],
        },
        "field_metrics": field_metrics,
    }


def import_pyplot() -> tuple[Any | None, str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "grid-delta-atlas-v014"
        import matplotlib.pyplot as plt

        return plt, None
    except Exception as exc:  # pragma: no cover - depends on optional environment
        return None, f"{type(exc).__name__}: {exc}"


def metric_matrix(field_metrics: dict[str, Any], fields: list[str], leads: list[int], metric: str) -> np.ndarray:
    matrix = np.full((len(fields), len(leads)), np.nan, dtype=np.float64)
    lead_index = {lead: i for i, lead in enumerate(leads)}
    for row_idx, field_name in enumerate(fields):
        for row in field_metrics[field_name].get("by_lead", []):
            lead = int(row["lead_h"])
            if lead in lead_index and row.get(metric) is not None:
                matrix[row_idx, lead_index[lead]] = float(row[metric])
    return matrix


def save_heatmap(plt: Any, matrix: np.ndarray, fields: list[str], leads: list[int], title: str, path: Path) -> None:
    fig_h = max(3.2, min(11.0, 0.32 * len(fields) + 1.8))
    fig_w = max(5.0, min(13.0, 0.35 * len(leads) + 3.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Lead hour")
    ax.set_ylabel("Field")
    ax.set_xticks(range(len(leads)))
    ax.set_xticklabels([str(lead) for lead in leads], rotation=45, ha="right")
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def reduce_to_spatial_2d(diff: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(diff, dtype=np.float64)
    if arr.ndim < 2:
        return None
    with np.errstate(all="ignore"):
        while arr.ndim > 2:
            arr = np.nanmax(np.abs(arr), axis=0)
    return np.asarray(arr, dtype=np.float64)


def save_spatial_map(plt: Any, diff2d: np.ndarray, field_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=120)
    image = ax.imshow(diff2d, interpolation="nearest", cmap="magma")
    ax.set_title(f"{field_name} worst-case max abs footprint")
    ax.set_xlabel("west_east")
    ax.set_ylabel("south_north")
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def make_plots(summary: dict[str, Any], asset_dir: Path, no_plots: bool, spatial_limit: int) -> dict[str, Any]:
    if no_plots:
        return {"status": "disabled", "plots": [], "asset_dir": str(asset_dir)}
    plt, error = import_pyplot()
    if plt is None:
        return {"status": "skipped_missing_dependency", "error": error, "plots": [], "asset_dir": str(asset_dir)}

    asset_dir.mkdir(parents=True, exist_ok=True)
    field_metrics = summary["field_metrics"]
    ranked_fields = [row["field"] for row in summary["top_fields_by_severity"] if row["field"] in field_metrics]
    fields = ranked_fields[: min(24, len(ranked_fields))]
    leads = summary["pairing"]["lead_hours"]
    plots: list[dict[str, Any]] = []
    metric_titles = {
        "rmse": "RMSE by field and lead",
        "bias": "Bias by field and lead",
        "p99_abs": "p99 absolute delta by field and lead",
        "max_abs": "Max absolute delta by field and lead",
    }
    if fields and leads:
        for metric, title in metric_titles.items():
            path = asset_dir / f"heatmap_{metric}.png"
            save_heatmap(plt, metric_matrix(field_metrics, fields, leads, metric), fields, leads, title, path)
            plots.append({"kind": "heatmap", "metric": metric, "path": str(path)})

    core_fields = [field for field in DEFAULT_PLOT_CORE_FIELDS if field in field_metrics and field_metrics[field]["status"] == "compared"]
    if core_fields:
        fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=120)
        for field_name in core_fields[:12]:
            rows = [row for row in field_metrics[field_name].get("by_lead", []) if row.get("rmse") is not None]
            if not rows:
                continue
            ax.plot([row["lead_h"] for row in rows], [row["rmse"] for row in rows], marker="o", linewidth=1.3, label=field_name)
        ax.set_title("Core-field RMSE over lead time")
        ax.set_xlabel("Lead hour")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        path = asset_dir / "core_fields_rmse_timeseries.png"
        fig.savefig(path)
        plt.close(fig)
        plots.append({"kind": "timeseries", "metric": "rmse", "path": str(path)})

    if summary["top_fields_by_severity"]:
        top = summary["top_fields_by_severity"][:10]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=120)
        labels = [row["field"] for row in top][::-1]
        max_abs = [float(row.get("max_abs") or 0.0) for row in top][::-1]
        ax0.barh(labels, max_abs, color="#386cb0")
        ax0.set_title("Top max_abs fields")
        ax0.set_xlabel("max_abs")
        ax0.grid(True, axis="x", alpha=0.25)
        ax1.axis("off")
        lines = [
            "Grid-Delta Atlas",
            f"Verdict: {summary['verdict']}",
            f"Pairs: {summary['pairing']['paired_file_count']}",
            f"Fields: {summary['inventory']['compared_numeric_field_count']}",
            f"Leads: {len(summary['pairing']['lead_hours'])}",
            f"Missing records: {summary['inventory']['missing_record_count']}",
            f"Nonfinite records: {summary['inventory']['nonfinite_record_count']}",
            f"Shape mismatches: {summary['inventory']['shape_mismatch_record_count']}",
        ]
        ax1.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")
        fig.tight_layout()
        path = asset_dir / "dashboard.png"
        fig.savefig(path)
        plt.close(fig)
        plots.append({"kind": "dashboard", "path": str(path)})

    spatial_done = 0
    for row in summary["top_fields_by_severity"]:
        if spatial_done >= spatial_limit:
            break
        field_name = row["field"]
        if field_name not in DEFAULT_PLOT_CORE_FIELDS:
            continue
        worst = field_metrics[field_name].get("overall", {}).get("worst")
        if not worst:
            continue
        try:
            with Dataset(worst["cpu_file"], "r") as cpu_ds, Dataset(worst["gpu_file"], "r") as gpu_ds:
                if field_name not in cpu_ds.variables or field_name not in gpu_ds.variables:
                    continue
                cpu = read_variable(cpu_ds, field_name)
                gpu = read_variable(gpu_ds, field_name)
            if cpu.shape != gpu.shape:
                continue
            diff2d = reduce_to_spatial_2d(np.asarray(gpu, dtype=np.float64) - np.asarray(cpu, dtype=np.float64))
            if diff2d is None:
                continue
            path = asset_dir / f"spatial_maxabs_{field_name}.png"
            save_spatial_map(plt, diff2d, field_name, path)
            plots.append({"kind": "spatial_max_abs", "field": field_name, "path": str(path)})
            spatial_done += 1
        except Exception as exc:  # pragma: no cover - defensive for corrupt real files
            plots.append({"kind": "spatial_max_abs", "field": field_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    return {"status": "ok", "plots": plots, "asset_dir": str(asset_dir)}


def write_markdown(summary: dict[str, Any], manifest: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# V0.14 Grid-Delta Atlas",
        "",
        f"Generated UTC: `{manifest['generated_utc']}`",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['verdict']}`",
        f"- paired wrfout files: `{summary['pairing']['paired_file_count']}`",
        f"- compared numeric fields: `{summary['inventory']['compared_numeric_field_count']}`",
        f"- tolerance manifest supplied: `{manifest['tolerances']['supplied']}`",
        f"- plots: `{manifest['plots']['status']}`",
        "",
        "This report is produced by offline tooling from existing CPU-WRF and GPU wrfout files. It does not run model code.",
        "",
        "## Artifacts",
        "",
        f"- manifest: `{manifest['outputs']['manifest_json']}`",
        f"- summary: `{manifest['outputs']['summary_json']}`",
        f"- assets: `{manifest['outputs']['asset_dir']}`",
        "",
        "## Coverage",
        "",
        f"- cases: `{summary['pairing']['case_count']}`",
        f"- domains: `{', '.join(summary['pairing']['domains']) or 'none'}`",
        f"- lead hours: `{summary['pairing']['lead_hours']}`",
        f"- variable union/common: `{summary['inventory']['union_variable_count']}`/`{summary['inventory']['common_variable_count']}`",
        f"- CPU-only/GPU-only variables: `{manifest['inventory']['cpu_only_count']}`/`{manifest['inventory']['gpu_only_count']}`",
        "",
        "## Inventory Issues",
        "",
        f"- missing records: `{summary['inventory']['missing_record_count']}`",
        f"- mandatory-core missing records: `{summary['inventory']['mandatory_missing_record_count']}`",
        f"- non-numeric records: `{summary['inventory']['non_numeric_record_count']}`",
        f"- nonfinite records: `{summary['inventory']['nonfinite_record_count']}`",
        f"- shape mismatches: `{summary['inventory']['shape_mismatch_record_count']}`",
        f"- dimension-name mismatches: `{summary['inventory']['dimension_name_mismatch_record_count']}`",
        "",
        "## Top Field Differences",
        "",
        "| Field | RMSE | Bias | p50 | p95 | p99 | p99.9 | Max abs | Corr | Worst lead |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["top_fields_by_severity"][:20]:
        worst = row.get("worst") or {}
        lines.append(
            f"| `{row['field']}` | {fmt_num(row.get('rmse'))} | {fmt_num(row.get('bias'))} | "
            f"{fmt_num(row.get('p50_abs'))} | {fmt_num(row.get('p95_abs'))} | {fmt_num(row.get('p99_abs'))} | "
            f"{fmt_num(row.get('p999_abs'))} | {fmt_num(row.get('max_abs'))} | {fmt_num(row.get('correlation'))} | "
            f"{worst.get('lead_h', 'NA')} |"
        )
    if not summary["top_fields_by_severity"]:
        lines.append("| none | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    lines.extend(
        [
            "",
            "## Lead-Time Stability",
            "",
            "| Field | RMSE slope/h | Bias slope/h | Late-early RMSE | Late-early bias | Worst lead |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["top_drift_signals"][:12]:
        lines.append(
            f"| `{row['field']}` | {fmt_num(row.get('rmse_slope_per_lead_hour'))} | "
            f"{fmt_num(row.get('bias_slope_per_lead_hour'))} | {fmt_num(row.get('late_minus_early_rmse'))} | "
            f"{fmt_num(row.get('late_minus_early_bias'))} | {row.get('worst_lead_h')} |"
        )
    if not summary["top_drift_signals"]:
        lines.append("| none | NA | NA | NA | NA | NA |")
    lines.extend(["", "## Plot Inventory", ""])
    for plot in manifest["plots"].get("plots", []):
        path_s = plot.get("path")
        if path_s:
            lines.append(f"- `{plot.get('kind')}`: `{path_s}`")
    if not manifest["plots"].get("plots"):
        lines.append(f"- no plots emitted (`{manifest['plots']['status']}`)")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_atlas(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    case_specs = case_specs_from_args(args)
    tolerances, tolerance_meta = load_tolerances(args.tolerance_json)
    pairs, pairing = build_pairs(case_specs, args.min_lead, args.max_lead)
    if not pairs:
        raise SystemExit("no paired wrfout files found after domain/lead filtering")

    fields, inventory = build_field_union(pairs)
    if args.field:
        requested = set(args.field)
        fields = [field_name for field_name in fields if field_name in requested]
    issue_records = {
        "missing": [],
        "non_numeric": [],
        "nonfinite": [],
        "shape_mismatch": [],
        "dimension_name_mismatch": [],
    }
    field_metrics: dict[str, Any] = {}
    for i, field_name in enumerate(fields, start=1):
        field_summary, issues = compare_field(
            field_name,
            pairs,
            inventory["first_metadata"].get(field_name, {}),
            tolerances.get(field_name),
            args.relative_floor,
        )
        field_metrics[field_name] = field_summary
        for key in issue_records:
            issue_records[key].extend(issues[key])
        if args.progress and (i % args.progress == 0 or i == len(fields)):
            print(f"compared {i}/{len(fields)} fields: {field_name}", file=sys.stderr, flush=True)

    mandatory_fields = () if args.no_default_mandatory_fields else DEFAULT_MANDATORY_CORE_FIELDS
    if args.mandatory_field:
        mandatory_fields = tuple(dict.fromkeys((*mandatory_fields, *args.mandatory_field)))
    mandatory_missing = check_mandatory_fields(pairs, mandatory_fields)
    summary = build_summary(
        pairs,
        pairing,
        inventory,
        field_metrics,
        issue_records,
        bool(tolerances),
        mandatory_missing,
    )
    plot_manifest = make_plots(summary, args.asset_dir, args.no_plots, args.spatial_plot_limit)
    generated = datetime.now(timezone.utc).isoformat()
    manifest_path = args.proof_dir / "manifest.json"
    summary_path = args.proof_dir / "grid_delta_summary.json"
    report_path = args.proof_dir / "GRID_DELTA_ATLAS.md"
    elapsed = time.perf_counter() - started
    manifest = {
        "schema": "grid-delta-atlas-manifest-v1",
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "cpu_only": True,
        "gpu_used": False,
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "note": "Offline NetCDF comparison only; no model runtime is invoked.",
        },
        "runtime": {
            "elapsed_seconds": clean_float(elapsed),
            "streaming_policy": "Field-major: one field and one paired wrfout pair are read at a time.",
            "exact_percentiles": True,
            "relative_floor": args.relative_floor,
        },
        "inputs": {
            "case_json": str(args.case_json) if args.case_json else None,
            "cpu_dir": str(args.cpu_dir) if args.cpu_dir else None,
            "gpu_dir": str(args.gpu_dir) if args.gpu_dir else None,
            "domains": args.domain,
            "min_lead": args.min_lead,
            "max_lead": args.max_lead,
            "field_subset": args.field,
            "mandatory_fields": list(mandatory_fields),
        },
        "outputs": {
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
            "markdown_report": str(report_path),
            "asset_dir": str(args.asset_dir),
        },
        "pairing": pairing,
        "inventory": {
            key: value
            for key, value in inventory.items()
            if key not in {"first_metadata", "variable_presence"}
        },
        "tolerances": tolerance_meta,
        "plots": plot_manifest,
        "verdict": summary["verdict"],
    }
    return manifest, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_argument_group("inputs")
    input_group.add_argument("--cpu-dir", type=Path, help="Directory containing CPU-WRF wrfout files for a single case.")
    input_group.add_argument("--gpu-dir", type=Path, help="Directory containing GPU wrfout files for a single case.")
    input_group.add_argument("--case-id", help="Case id for --cpu-dir/--gpu-dir mode.")
    input_group.add_argument(
        "--case-json",
        type=Path,
        help="JSON list, or object with cases list, containing case_id, cpu_dir, gpu_dir, optional domain(s), and optional init_time_utc.",
    )
    input_group.add_argument("--domain", action="append", help="Domain to include, e.g. d01 or d02. Repeatable. Default: all discovered.")
    input_group.add_argument("--init", help="Initialization time for single-case mode, ISO UTC or local-naive UTC.")
    input_group.add_argument("--min-lead", type=int, default=None, help="Minimum lead hour to include.")
    input_group.add_argument("--max-lead", type=int, default=None, help="Maximum lead hour to include.")
    input_group.add_argument("--field", action="append", help="Optional field subset. Repeatable. Default: every discovered field.")
    input_group.add_argument("--mandatory-field", action="append", help="Additional mandatory field. Repeatable.")
    input_group.add_argument(
        "--no-default-mandatory-fields",
        action="store_true",
        help="Disable the built-in v0.14 core-field missing-field hard-fail list.",
    )
    input_group.add_argument("--tolerance-json", type=Path, help="Optional predeclared tolerance manifest.")

    output_group = parser.add_argument_group("outputs")
    output_group.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR, help="Output proof directory.")
    output_group.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR, help="Output plot asset directory.")
    output_group.add_argument("--no-plots", action="store_true", help="Skip optional matplotlib plot generation.")

    parser.add_argument("--relative-floor", type=float, default=1.0e-12, help="Denominator floor for safe relative metrics.")
    parser.add_argument("--spatial-plot-limit", type=int, default=8, help="Maximum core-field worst-case spatial plots.")
    parser.add_argument("--progress", type=int, default=0, help="Print one stderr progress line every N fields.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, summary = build_atlas(args)
    manifest_path = args.proof_dir / "manifest.json"
    summary_path = args.proof_dir / "grid_delta_summary.json"
    report_path = args.proof_dir / "GRID_DELTA_ATLAS.md"
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)
    write_markdown(summary, manifest, report_path)
    compact = {
        "verdict": summary["verdict"],
        "paired_files": summary["pairing"]["paired_file_count"],
        "compared_fields": summary["inventory"]["compared_numeric_field_count"],
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "report": str(report_path),
        "plot_status": manifest["plots"]["status"],
        "plot_count": len(manifest["plots"].get("plots", [])),
    }
    print(json.dumps(compact, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
