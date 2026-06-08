#!/usr/bin/env python3
"""Complete CPU-WRF-vs-GPU wrfout grid comparator.

The comparator reads existing wrfout directories only. It does not run WRF,
JAX, CUDA, or any model code.
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
from netCDF4 import Dataset, chartostring


WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")
CASE_INIT_RE = re.compile(r"(20\d{6})_(\d{2})z", re.IGNORECASE)

TIME_METADATA_FIELDS = {"Times", "XTIME"}
STATIC_FIELD_NAMES = {
    "XLAT",
    "XLONG",
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "HGT",
    "LANDMASK",
    "LU_INDEX",
    "ZNU",
    "ZNW",
    "ZS",
    "DZS",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "MAPFAC_MX",
    "MAPFAC_MY",
    "MAPFAC_UX",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_VY",
    "F",
    "E",
    "SINALPHA",
    "COSALPHA",
    "XLAND",
    "P_TOP",
    "PB",
    "PHB",
    "MUB",
    "MU0",
    "DN",
    "DNW",
    "RDN",
    "RDNW",
    "FNM",
    "FNP",
    "CFN",
    "CFN1",
    "CF1",
    "CF2",
    "CF3",
    "C1H",
    "C2H",
    "C3H",
    "C4H",
    "C1F",
    "C2F",
    "C3F",
    "C4F",
    "RDX",
    "RDY",
    "CLAT",
}


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def parse_wrfout_time(path: Path) -> datetime | None:
    match = WRFOUT_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def parse_domain(path: Path) -> str | None:
    match = WRFOUT_RE.match(path.name)
    return match.group(1) if match else None


def discover_wrfouts(run_dir: Path, domain: str) -> dict[datetime, Path]:
    out: dict[datetime, Path] = {}
    for path in sorted(run_dir.glob(f"wrfout_{domain}_*")):
        if not path.is_file():
            continue
        valid = parse_wrfout_time(path)
        if valid is not None:
            out[valid] = path
    return out


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
    out = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if out.tzinfo is None:
        return out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def infer_init_time(cpu_dir: Path, gpu_dir: Path, cpu_map: dict[datetime, Path], gpu_map: dict[datetime, Path], explicit: str | None) -> tuple[datetime, str]:
    if explicit:
        return parse_iso_time(explicit), "--init"
    for text in (str(gpu_dir), str(cpu_dir)):
        parsed = parse_init_from_text(text)
        if parsed is not None:
            return parsed, "directory-name"
    all_times = sorted(set(cpu_map) | set(gpu_map))
    if not all_times:
        raise SystemExit("no wrfout files found for init-time inference")
    return all_times[0], "earliest-wrfout-fallback"


def lead_hour(valid_time: datetime, init_time: datetime) -> int:
    return int(round((valid_time - init_time).total_seconds() / 3600.0))


def dims_without_time(var: Any) -> tuple[str, ...]:
    dims = tuple(var.dimensions)
    if dims and dims[0] == "Time":
        return dims[1:]
    return dims


def shape_without_time(var: Any) -> tuple[int, ...]:
    shape = tuple(int(x) for x in var.shape)
    dims = tuple(var.dimensions)
    if dims and dims[0] == "Time":
        return shape[1:]
    return shape


def is_numeric_dtype(dtype: Any) -> bool:
    return np.dtype(dtype).kind in {"b", "i", "u", "f", "c"}


def var_metadata(ds: Dataset, name: str) -> dict[str, Any]:
    var = ds.variables[name]
    return {
        "dims": list(dims_without_time(var)),
        "shape": list(shape_without_time(var)),
        "dtype": str(var.dtype),
        "units": str(getattr(var, "units", "")),
        "description": str(getattr(var, "description", getattr(var, "FieldType", ""))),
    }


def read_var(ds: Dataset, name: str) -> np.ndarray:
    var = ds.variables[name]
    raw = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
    return np.asarray(np.ma.filled(raw, np.nan))


def read_string_var(ds: Dataset, name: str) -> str:
    arr = read_var(ds, name)
    try:
        converted = chartostring(arr)
        if np.ndim(converted) == 0:
            return str(converted.item())
        return str(np.asarray(converted).ravel()[0])
    except Exception:
        flat = arr.ravel()
        parts = []
        for item in flat:
            if isinstance(item, bytes):
                parts.append(item.decode("ascii", errors="replace"))
            else:
                parts.append(str(item))
        return "".join(parts)


def arrays_equal_nan(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


def pearson_from_sums(n: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float | None:
    if n < 2:
        return None
    cov = n * sxy - sx * sy
    vx = n * sxx - sx * sx
    vy = n * syy - sy * sy
    if vx <= 0.0 or vy <= 0.0:
        return None
    return clean_float(cov / math.sqrt(vx * vy))


def normalize_tolerance_record(value: Any) -> dict[str, float]:
    if isinstance(value, (int, float)):
        return {"max_abs": float(value), "element_abs": float(value)}
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    aliases = {
        "abs": "element_abs",
        "cell_abs": "element_abs",
        "tolerance_abs": "element_abs",
        "element_abs": "element_abs",
        "max_abs": "max_abs",
        "rmse": "rmse",
        "mae": "mae",
        "p95_abs": "p95_abs",
        "p99_abs": "p99_abs",
        "bias_abs": "bias_abs",
        "pearson_min": "pearson_min",
        "finite_pair_fraction_min": "finite_pair_fraction_min",
    }
    for src, dst in aliases.items():
        if src in value:
            converted = clean_float(value[src])
            if converted is not None:
                out[dst] = converted
    return out


def load_tolerances(path: Path | None) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if path is None:
        return {}, {"supplied": False, "policy": "No tolerance manifest supplied; differences are report-only."}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("fields", "variables", "tolerances"):
            if isinstance(payload.get(key), dict):
                raw = payload[key]
                break
        else:
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
        "supported_metrics": ["rmse", "mae", "bias_abs", "p95_abs", "p99_abs", "max_abs", "element_abs", "pearson_min", "finite_pair_fraction_min"],
    }


@dataclass
class StatsAccumulator:
    element_abs_tolerance: float | None = None
    n: int = 0
    finite_cpu: int = 0
    finite_gpu: int = 0
    finite_pair: int = 0
    sum_diff: float = 0.0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0
    within_element_abs: int = 0
    sum_gpu: float = 0.0
    sum_cpu: float = 0.0
    sum_gpu_sq: float = 0.0
    sum_cpu_sq: float = 0.0
    sum_gpu_cpu: float = 0.0
    abs_chunks: list[np.ndarray] = field(default_factory=list)

    def update(self, gpu: np.ndarray, cpu: np.ndarray, mask2d: np.ndarray | None = None) -> None:
        g = np.asarray(gpu, dtype=np.float64)
        c = np.asarray(cpu, dtype=np.float64)
        if g.shape != c.shape:
            raise ValueError(f"shape mismatch {g.shape} vs {c.shape}")
        cpu_finite = np.isfinite(c)
        gpu_finite = np.isfinite(g)
        valid = cpu_finite & gpu_finite
        if mask2d is not None:
            mask = broadcast_horizontal_mask(mask2d, g.shape)
            cpu_finite &= mask
            gpu_finite &= mask
            valid &= mask
            total = int(np.sum(mask))
        else:
            total = int(g.size)
        self.n += total
        self.finite_cpu += int(np.sum(cpu_finite))
        self.finite_gpu += int(np.sum(gpu_finite))
        if not np.any(valid):
            return
        gv = g[valid]
        cv = c[valid]
        diff = gv - cv
        abs_diff = np.abs(diff)
        self.finite_pair += int(diff.size)
        self.sum_diff += float(np.sum(diff, dtype=np.float64))
        self.sum_abs += float(np.sum(abs_diff, dtype=np.float64))
        self.sum_sq += float(np.sum(diff * diff, dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(abs_diff)))
        if self.element_abs_tolerance is not None:
            self.within_element_abs += int(np.sum(abs_diff <= self.element_abs_tolerance))
        self.sum_gpu += float(np.sum(gv, dtype=np.float64))
        self.sum_cpu += float(np.sum(cv, dtype=np.float64))
        self.sum_gpu_sq += float(np.sum(gv * gv, dtype=np.float64))
        self.sum_cpu_sq += float(np.sum(cv * cv, dtype=np.float64))
        self.sum_gpu_cpu += float(np.sum(gv * cv, dtype=np.float64))
        self.abs_chunks.append(abs_diff.astype(np.float32, copy=False))

    def finish(self) -> dict[str, Any]:
        if self.finite_pair == 0:
            return {
                "n": int(self.n),
                "finite_cpu": int(self.finite_cpu),
                "finite_gpu": int(self.finite_gpu),
                "finite_pair": 0,
                "finite_pair_fraction": 0.0 if self.n else None,
                "bias": None,
                "rmse": None,
                "mae": None,
                "p95_abs": None,
                "p99_abs": None,
                "max_abs": None,
                "pearson_r": None,
                "element_abs_tolerance": self.element_abs_tolerance,
                "frac_within_element_abs_tolerance": None,
            }
        abs_all = np.concatenate(self.abs_chunks) if self.abs_chunks else np.asarray([], dtype=np.float32)
        return {
            "n": int(self.n),
            "finite_cpu": int(self.finite_cpu),
            "finite_gpu": int(self.finite_gpu),
            "finite_pair": int(self.finite_pair),
            "finite_pair_fraction": clean_float(self.finite_pair / self.n) if self.n else None,
            "bias": clean_float(self.sum_diff / self.finite_pair),
            "rmse": clean_float(math.sqrt(self.sum_sq / self.finite_pair)),
            "mae": clean_float(self.sum_abs / self.finite_pair),
            "p95_abs": clean_float(np.percentile(abs_all, 95)) if abs_all.size else None,
            "p99_abs": clean_float(np.percentile(abs_all, 99)) if abs_all.size else None,
            "max_abs": clean_float(self.max_abs),
            "pearson_r": pearson_from_sums(
                self.finite_pair,
                self.sum_gpu,
                self.sum_cpu,
                self.sum_gpu_sq,
                self.sum_cpu_sq,
                self.sum_gpu_cpu,
            ),
            "element_abs_tolerance": self.element_abs_tolerance,
            "frac_within_element_abs_tolerance": clean_float(self.within_element_abs / self.finite_pair)
            if self.element_abs_tolerance is not None
            else None,
        }


def apply_tolerance(stats: dict[str, Any], spec: dict[str, float] | None) -> dict[str, Any]:
    if not spec:
        return {"supplied": False, "pass": None, "failures": []}
    failures: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    for metric in ("rmse", "mae", "p95_abs", "p99_abs", "max_abs"):
        if metric not in spec:
            continue
        value = stats.get(metric)
        limit = spec[metric]
        passed = bool(value is not None and float(value) <= limit)
        checks[metric] = {"value": value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": metric, "value": value, "limit": limit})
    if "bias_abs" in spec:
        value = stats.get("bias")
        abs_value = abs(float(value)) if value is not None else None
        limit = spec["bias_abs"]
        passed = bool(abs_value is not None and abs_value <= limit)
        checks["bias_abs"] = {"value": abs_value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": "bias_abs", "value": abs_value, "limit": limit})
    if "pearson_min" in spec:
        value = stats.get("pearson_r")
        limit = spec["pearson_min"]
        passed = bool(value is not None and float(value) >= limit)
        checks["pearson_min"] = {"value": value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": "pearson_min", "value": value, "limit": limit})
    if "finite_pair_fraction_min" in spec:
        value = stats.get("finite_pair_fraction")
        limit = spec["finite_pair_fraction_min"]
        passed = bool(value is not None and float(value) >= limit)
        checks["finite_pair_fraction_min"] = {"value": value, "limit": limit, "pass": passed}
        if not passed:
            failures.append({"metric": "finite_pair_fraction_min", "value": value, "limit": limit})
    return {"supplied": True, "spec": spec, "checks": checks, "pass": not failures, "failures": failures}


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


def drift_summary(by_lead: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in by_lead if row.get("rmse") is not None]
    leads = [float(row["lead_h"]) for row in rows]
    rmses = [float(row["rmse"]) for row in rows]
    biases = [float(row["bias"]) for row in rows if row.get("bias") is not None]
    bias_leads = [float(row["lead_h"]) for row in rows if row.get("bias") is not None]
    worst = max(rows, key=lambda row: float(row["rmse"])) if rows else None
    signs = []
    for value in biases:
        if value > 0.0:
            signs.append(1)
        elif value < 0.0:
            signs.append(-1)
    pos = signs.count(1)
    neg = signs.count(-1)
    majority = 1 if pos >= neg else -1
    consistency = (max(pos, neg) / len(signs)) if signs else None
    return {
        "rmse_slope_per_hour": linear_slope(leads, rmses),
        "bias_slope_per_hour": linear_slope(bias_leads, biases),
        "worst_lead_h": int(worst["lead_h"]) if worst else None,
        "worst_lead_rmse": worst.get("rmse") if worst else None,
        "worst_lead_bias": worst.get("bias") if worst else None,
        "bias_majority_sign": "+" if majority > 0 else "-",
        "bias_sign_consistency_fraction": clean_float(consistency),
        "lead_count": len(rows),
    }


def field_failure_score(summary: dict[str, Any], tolerance_supplied: bool) -> float:
    overall = summary.get("overall", {})
    tol = summary.get("tolerance_result", {})
    if tolerance_supplied and tol.get("supplied"):
        score = 0.0
        for failure in tol.get("failures", []):
            value = failure.get("value")
            limit = failure.get("limit")
            if isinstance(value, (int, float)) and isinstance(limit, (int, float)) and abs(float(limit)) > 0.0:
                score = max(score, abs(float(value)) / abs(float(limit)))
            elif value is not None:
                score = max(score, 1.0)
        return score
    rmse = float(overall.get("rmse") or 0.0)
    max_abs = float(overall.get("max_abs") or 0.0)
    p99_abs = float(overall.get("p99_abs") or 0.0)
    return max(rmse, p99_abs, max_abs)


def fmt_num(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        x = float(value)
        if x == 0.0:
            return "0"
        if abs(x) >= 10000.0 or abs(x) < 0.001:
            return f"{x:.3e}"
        return f"{x:.{digits}f}"
    return "NA"


def label_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    names = sorted({str(item) for item in labels.ravel().tolist()})
    return {name: labels == name for name in names}


def edge_average_x(arr: np.ndarray) -> np.ndarray:
    out = np.empty((arr.shape[0], arr.shape[1] + 1), dtype=np.float64)
    out[:, 0] = arr[:, 0]
    out[:, -1] = arr[:, -1]
    out[:, 1:-1] = 0.5 * (arr[:, :-1] + arr[:, 1:])
    return out


def edge_average_y(arr: np.ndarray) -> np.ndarray:
    out = np.empty((arr.shape[0] + 1, arr.shape[1]), dtype=np.float64)
    out[0, :] = arr[0, :]
    out[-1, :] = arr[-1, :]
    out[1:-1, :] = 0.5 * (arr[:-1, :] + arr[1:, :])
    return out


def edge_bool_x(arr: np.ndarray) -> np.ndarray:
    return edge_average_x(arr.astype(np.float64)) > 0.5


def edge_bool_y(arr: np.ndarray) -> np.ndarray:
    return edge_average_y(arr.astype(np.float64)) > 0.5


def boundary_mask(shape: tuple[int, int], width: int) -> np.ndarray:
    ny, nx = shape
    width = min(width, max(ny // 2, 0), max(nx // 2, 0))
    mask = np.zeros(shape, dtype=bool)
    if width <= 0:
        return mask
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def split_masks_for_grid(hgt: np.ndarray, land: np.ndarray, lat: np.ndarray, lon: np.ndarray, boundary_width: int) -> dict[str, dict[str, np.ndarray]]:
    land_bool = land > 0.5
    elev = np.full(hgt.shape, "ocean", dtype=object)
    elev[land_bool & (hgt < 300.0)] = "land_0_300m"
    elev[land_bool & (hgt >= 300.0) & (hgt < 1000.0)] = "land_300_1000m"
    elev[land_bool & (hgt >= 1000.0)] = "land_gt_1000m"
    lat_mid = float(np.nanmedian(lat))
    lon_mid = float(np.nanmedian(lon))
    quadrant = np.full(hgt.shape, "SW", dtype=object)
    quadrant[(lat >= lat_mid) & (lon < lon_mid)] = "NW"
    quadrant[(lat >= lat_mid) & (lon >= lon_mid)] = "NE"
    quadrant[(lat < lat_mid) & (lon >= lon_mid)] = "SE"
    frame = boundary_mask(hgt.shape, boundary_width)
    boundary = np.where(frame, f"frame_{boundary_width}cells", f"interior_excluding_{boundary_width}cell_frame")
    return {
        "land_ocean": label_masks(np.where(land_bool, "land", "ocean")),
        "elevation": label_masks(elev),
        "quadrant": label_masks(quadrant),
        "boundary": label_masks(boundary),
    }


def build_spatial_masks(first_cpu: Path, boundary_width: int) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    with Dataset(first_cpu, "r") as ds:
        required = ["HGT", "LANDMASK", "XLAT", "XLONG"]
        missing = [name for name in required if name not in ds.variables]
        if missing:
            return {}, {}, [f"spatial splits disabled: missing {', '.join(missing)} in {first_cpu}"]
        hgt = np.asarray(read_var(ds, "HGT"), dtype=np.float64)
        land = np.asarray(read_var(ds, "LANDMASK"), dtype=np.float64)
        lat = np.asarray(read_var(ds, "XLAT"), dtype=np.float64)
        lon = np.asarray(read_var(ds, "XLONG"), dtype=np.float64)
    mass = split_masks_for_grid(hgt, land, lat, lon, boundary_width)
    u = split_masks_for_grid(edge_average_x(hgt), edge_bool_x(land > 0.5), edge_average_x(lat), edge_average_x(lon), boundary_width)
    v = split_masks_for_grid(edge_average_y(hgt), edge_bool_y(land > 0.5), edge_average_y(lat), edge_average_y(lon), boundary_width)
    counts = {
        kind: {
            split_name: {label: int(np.sum(mask)) for label, mask in split.items()}
            for split_name, split in groups.items()
        }
        for kind, groups in {"mass": mass, "u": u, "v": v}.items()
    }
    counts["mass_grid_shape"] = {"south_north": int(hgt.shape[0]), "west_east": int(hgt.shape[1])}
    return {"mass": mass, "u": u, "v": v}, counts, warnings


def horizontal_kind(dims: tuple[str, ...]) -> str | None:
    if len(dims) < 2:
        return None
    tail = dims[-2:]
    if tail == ("south_north", "west_east"):
        return "mass"
    if tail == ("south_north", "west_east_stag"):
        return "u"
    if tail == ("south_north_stag", "west_east"):
        return "v"
    return None


def broadcast_horizontal_mask(mask2d: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if len(target_shape) < 2:
        raise ValueError(f"cannot apply horizontal mask to shape {target_shape}")
    if tuple(target_shape[-2:]) != tuple(mask2d.shape):
        raise ValueError(f"mask shape {mask2d.shape} incompatible with field shape {target_shape}")
    return np.broadcast_to(mask2d.reshape((1,) * (len(target_shape) - 2) + mask2d.shape), target_shape)


def make_inventory(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_union: set[str] = set()
    gpu_union: set[str] = set()
    first_meta: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, str] = {}
    for pair in pairs:
        with Dataset(pair["cpu_file"], "r") as cds, Dataset(pair["gpu_file"], "r") as gds:
            cpu_names = set(cds.variables)
            gpu_names = set(gds.variables)
            cpu_union.update(cpu_names)
            gpu_union.update(gpu_names)
            for name in sorted(cpu_names & gpu_names):
                if name not in first_meta:
                    first_meta[name] = {"cpu": var_metadata(cds, name), "gpu": var_metadata(gds, name)}
                    first_seen[name] = pair["valid_time_utc"]
    return {
        "cpu_variable_count": len(cpu_union),
        "gpu_variable_count": len(gpu_union),
        "common_variable_count": len(cpu_union & gpu_union),
        "cpu_only_count": len(cpu_union - gpu_union),
        "gpu_only_count": len(gpu_union - cpu_union),
        "cpu_only": sorted(cpu_union - gpu_union),
        "gpu_only": sorted(gpu_union - cpu_union),
        "common": sorted(cpu_union & gpu_union),
        "first_metadata": first_meta,
        "first_seen_utc": first_seen,
    }


def compatible_metadata(cpu_meta: dict[str, Any], gpu_meta: dict[str, Any]) -> tuple[bool, str | None]:
    if cpu_meta["shape"] != gpu_meta["shape"]:
        return False, "shape_mismatch"
    if cpu_meta["dims"] != gpu_meta["dims"]:
        return False, "dimension_name_mismatch"
    if not is_numeric_dtype(cpu_meta["dtype"]) or not is_numeric_dtype(gpu_meta["dtype"]):
        return False, "non_numeric"
    return True, None


def update_worst_cell(current: dict[str, Any] | None, diff: np.ndarray, gpu: np.ndarray, cpu: np.ndarray, lead_h: int) -> dict[str, Any] | None:
    arr = np.asarray(diff, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return current
    abs_arr = np.where(finite, np.abs(arr), -1.0)
    flat_idx = int(np.argmax(abs_arr))
    max_abs = float(abs_arr.ravel()[flat_idx])
    if current is not None and max_abs <= float(current.get("abs_diff", -1.0)):
        return current
    idx = tuple(int(i) for i in np.unravel_index(flat_idx, arr.shape))
    return {
        "lead_h": int(lead_h),
        "index": list(idx),
        "abs_diff": clean_float(max_abs),
        "diff": clean_float(arr[idx]),
        "gpu": clean_float(np.asarray(gpu, dtype=np.float64)[idx]),
        "cpu": clean_float(np.asarray(cpu, dtype=np.float64)[idx]),
    }


def compare_time_metadata(name: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    all_equal = True
    for pair in pairs:
        with Dataset(pair["cpu_file"], "r") as cds, Dataset(pair["gpu_file"], "r") as gds:
            if name not in cds.variables or name not in gds.variables:
                checks.append({"lead_h": pair["lead_h"], "status": "MISSING"})
                all_equal = False
                continue
            cpu = read_string_var(cds, name)
            gpu = read_string_var(gds, name)
        equal = cpu == gpu
        all_equal = all_equal and equal
        checks.append({"lead_h": pair["lead_h"], "status": "OK", "cpu": cpu, "gpu": gpu, "equal": bool(equal)})
    return {"classification": "time_metadata", "all_equal": bool(all_equal), "checks": checks}


def compare_variable(
    name: str,
    pairs: list[dict[str, Any]],
    metadata: dict[str, Any],
    tolerance_spec: dict[str, float] | None,
    spatial_masks: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    element_tol = tolerance_spec.get("element_abs") if tolerance_spec else None
    overall = StatsAccumulator(element_abs_tolerance=element_tol)
    by_lead: list[dict[str, Any]] = []
    missing_leads: list[dict[str, Any]] = []
    incompatible_leads: list[dict[str, Any]] = []
    split_accs: dict[str, dict[str, StatsAccumulator]] = {}
    first_cpu_array: np.ndarray | None = None
    first_gpu_array: np.ndarray | None = None
    prev_cpu: np.ndarray | None = None
    prev_gpu: np.ndarray | None = None
    cpu_time_invariant = True
    gpu_time_invariant = True
    lead_count = 0
    worst_cell: dict[str, Any] | None = None
    dims_tuple: tuple[str, ...] | None = None
    shape_tuple: tuple[int, ...] | None = None

    for pair in pairs:
        lead_h = int(pair["lead_h"])
        with Dataset(pair["cpu_file"], "r") as cds, Dataset(pair["gpu_file"], "r") as gds:
            if name not in cds.variables or name not in gds.variables:
                missing_leads.append({"lead_h": lead_h, "reason": "missing_in_one_paired_file"})
                continue
            cpu_var = cds.variables[name]
            gpu_var = gds.variables[name]
            cpu_meta = var_metadata(cds, name)
            gpu_meta = var_metadata(gds, name)
            ok, reason = compatible_metadata(cpu_meta, gpu_meta)
            if not ok:
                incompatible_leads.append({"lead_h": lead_h, "reason": reason, "cpu": cpu_meta, "gpu": gpu_meta})
                continue
            cpu = read_var(cds, name)
            gpu = read_var(gds, name)
            if not is_numeric_dtype(cpu_var.dtype) or not is_numeric_dtype(gpu_var.dtype):
                incompatible_leads.append({"lead_h": lead_h, "reason": "non_numeric", "cpu": cpu_meta, "gpu": gpu_meta})
                continue

        if first_cpu_array is None:
            first_cpu_array = np.array(cpu, copy=True)
            first_gpu_array = np.array(gpu, copy=True)
            dims_tuple = tuple(cpu_meta["dims"])
            shape_tuple = tuple(cpu_meta["shape"])
            kind = horizontal_kind(dims_tuple)
            if kind in spatial_masks:
                for split_name, masks in spatial_masks[kind].items():
                    split_accs[split_name] = {
                        label: StatsAccumulator(element_abs_tolerance=element_tol) for label in masks
                    }
        if prev_cpu is not None and not arrays_equal_nan(cpu, prev_cpu):
            cpu_time_invariant = False
        if prev_gpu is not None and not arrays_equal_nan(gpu, prev_gpu):
            gpu_time_invariant = False
        prev_cpu = np.array(cpu, copy=True)
        prev_gpu = np.array(gpu, copy=True)
        lead_count += 1

        lead_acc = StatsAccumulator(element_abs_tolerance=element_tol)
        lead_acc.update(gpu, cpu)
        lead_stats = lead_acc.finish()
        lead_stats["lead_h"] = lead_h
        lead_stats["valid_time_utc"] = pair["valid_time_utc"]
        lead_stats["tolerance_result"] = apply_tolerance(lead_stats, tolerance_spec)
        by_lead.append(lead_stats)
        overall.update(gpu, cpu)
        diff = np.asarray(gpu, dtype=np.float64) - np.asarray(cpu, dtype=np.float64)
        worst_cell = update_worst_cell(worst_cell, diff, gpu, cpu, lead_h)

        if split_accs and dims_tuple is not None:
            kind = horizontal_kind(dims_tuple)
            if kind in spatial_masks:
                for split_name, masks in spatial_masks[kind].items():
                    for label, mask in masks.items():
                        split_accs[split_name][label].update(gpu, cpu, mask2d=mask)

    overall_stats = overall.finish()
    known_static = name in STATIC_FIELD_NAMES
    observed_time_invariant = bool(lead_count > 1 and cpu_time_invariant and gpu_time_invariant)
    if name in TIME_METADATA_FIELDS:
        classification = "time_metadata"
    elif known_static:
        classification = "static"
    elif observed_time_invariant:
        classification = "time_invariant"
    else:
        classification = "dynamic"
    split_report = {
        split_name: {label: acc.finish() for label, acc in labels.items()}
        for split_name, labels in split_accs.items()
    }
    worst_region = None
    for split_name, labels in split_report.items():
        for label, stats in labels.items():
            rmse = stats.get("rmse")
            if rmse is None:
                continue
            candidate = {"split": split_name, "label": label, "rmse": rmse, "bias": stats.get("bias"), "finite_pair": stats.get("finite_pair")}
            if worst_region is None or float(rmse) > float(worst_region["rmse"]):
                worst_region = candidate
    out = {
        "classification": classification,
        "known_static_field": bool(known_static),
        "observed_time_invariant": bool(observed_time_invariant),
        "cpu_time_invariant_over_compared_leads": bool(cpu_time_invariant),
        "gpu_time_invariant_over_compared_leads": bool(gpu_time_invariant),
        "metadata": metadata,
        "overall": overall_stats,
        "tolerance_result": apply_tolerance(overall_stats, tolerance_spec),
        "by_lead": by_lead,
        "drift": drift_summary(by_lead),
        "spatial_splits": split_report,
        "worst_region": worst_region,
        "worst_cell": worst_cell,
        "missing_leads": missing_leads,
        "incompatible_leads": incompatible_leads,
        "compared_lead_count": int(lead_count),
    }
    if shape_tuple is not None:
        out["native_shape"] = list(shape_tuple)
    return out


def build_pairs(cpu_dir: Path, gpu_dir: Path, domain: str, init: str | None, min_lead: int | None, max_lead: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cpu_map = discover_wrfouts(cpu_dir, domain)
    gpu_map = discover_wrfouts(gpu_dir, domain)
    if not cpu_map:
        raise SystemExit(f"no CPU wrfout_{domain}_* files found under {cpu_dir}")
    if not gpu_map:
        raise SystemExit(f"no GPU wrfout_{domain}_* files found under {gpu_dir}")
    init_time, init_source = infer_init_time(cpu_dir, gpu_dir, cpu_map, gpu_map, init)
    common = sorted(set(cpu_map) & set(gpu_map))
    filtered = []
    for valid in common:
        lead_h = lead_hour(valid, init_time)
        if min_lead is not None and lead_h < min_lead:
            continue
        if max_lead is not None and lead_h > max_lead:
            continue
        filtered.append(valid)
    pairs = [
        {
            "valid_time_utc": valid.isoformat(),
            "lead_h": lead_hour(valid, init_time),
            "cpu_file": str(cpu_map[valid]),
            "gpu_file": str(gpu_map[valid]),
        }
        for valid in filtered
    ]
    coverage = {
        "domain": domain,
        "init_time_utc": init_time.isoformat(),
        "init_time_source": init_source,
        "cpu_file_count": len(cpu_map),
        "gpu_file_count": len(gpu_map),
        "common_file_count_before_lead_filter": len(common),
        "paired_file_count": len(pairs),
        "lead_filter": {"min_lead": min_lead, "max_lead": max_lead},
        "common_leads_h": [item["lead_h"] for item in pairs],
        "unmatched_cpu_count": len(set(cpu_map) - set(gpu_map)),
        "unmatched_gpu_count": len(set(gpu_map) - set(cpu_map)),
        "unmatched_cpu_times_utc": [t.isoformat() for t in sorted(set(cpu_map) - set(gpu_map))],
        "unmatched_gpu_times_utc": [t.isoformat() for t in sorted(set(gpu_map) - set(cpu_map))],
        "pairs": pairs,
    }
    if not pairs:
        raise SystemExit("no paired wrfout files remain after lead filtering")
    return pairs, coverage


def top_coverage_issues(inventory: dict[str, Any], field_summaries: dict[str, Any], non_numeric: list[dict[str, Any]], incompatible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if inventory["gpu_only_count"]:
        issues.append({"kind": "gpu_only", "count": inventory["gpu_only_count"], "examples": inventory["gpu_only"][:12]})
    if inventory["cpu_only_count"]:
        issues.append({"kind": "cpu_only", "count": inventory["cpu_only_count"], "examples": inventory["cpu_only"][:12]})
    if incompatible:
        issues.append({"kind": "incompatible", "count": len(incompatible), "examples": incompatible[:5]})
    if non_numeric:
        issues.append({"kind": "non_numeric", "count": len(non_numeric), "examples": non_numeric[:5]})
    missing_later = [
        {"field": name, "missing_leads": item.get("missing_leads", [])[:5]}
        for name, item in sorted(field_summaries.items())
        if item.get("missing_leads")
    ]
    if missing_later:
        issues.append({"kind": "missing_on_some_paired_leads", "count": len(missing_later), "examples": missing_later[:5]})
    return issues[:10]


def build_summaries(field_summaries: dict[str, Any], inventory: dict[str, Any], non_numeric: list[dict[str, Any]], incompatible: list[dict[str, Any]], tolerances_supplied: bool) -> dict[str, Any]:
    comparable = [item for item in field_summaries.values() if item.get("overall", {}).get("finite_pair", 0)]
    ranked = sorted(
        (
            {
                "field": name,
                "classification": item.get("classification"),
                "rmse": item.get("overall", {}).get("rmse"),
                "bias": item.get("overall", {}).get("bias"),
                "p99_abs": item.get("overall", {}).get("p99_abs"),
                "max_abs": item.get("overall", {}).get("max_abs"),
                "worst_lead_h": item.get("drift", {}).get("worst_lead_h"),
                "score": field_failure_score(item, tolerances_supplied),
                "tolerance_pass": item.get("tolerance_result", {}).get("pass"),
            }
            for name, item in field_summaries.items()
            if item.get("overall", {}).get("finite_pair", 0)
        ),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    dynamic_ranked = [row for row in ranked if row["classification"] == "dynamic"]
    static_ranked = [row for row in ranked if row["classification"] in {"static", "time_invariant"}]
    drift_ranked = sorted(
        (
            {
                "field": name,
                "classification": item.get("classification"),
                "rmse_slope_per_hour": item.get("drift", {}).get("rmse_slope_per_hour"),
                "bias_slope_per_hour": item.get("drift", {}).get("bias_slope_per_hour"),
                "bias_sign_consistency_fraction": item.get("drift", {}).get("bias_sign_consistency_fraction"),
                "worst_lead_h": item.get("drift", {}).get("worst_lead_h"),
                "worst_lead_rmse": item.get("drift", {}).get("worst_lead_rmse"),
                "score": abs(float(item.get("drift", {}).get("rmse_slope_per_hour") or 0.0))
                + abs(float(item.get("drift", {}).get("bias_slope_per_hour") or 0.0)),
            }
            for name, item in field_summaries.items()
            if item.get("classification") == "dynamic"
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    tolerance_failures = [row for row in ranked if row.get("tolerance_pass") is False]
    if tolerances_supplied:
        verdict = "FAIL" if tolerance_failures else "PASS"
    else:
        verdict = "REPORT_ONLY_NO_TOLERANCE_MANIFEST"
    coverage_issues = top_coverage_issues(inventory, field_summaries, non_numeric, incompatible)
    return {
        "verdict": verdict,
        "comparable_field_count": len(comparable),
        "dynamic_field_count": sum(1 for item in field_summaries.values() if item.get("classification") == "dynamic"),
        "static_or_time_invariant_field_count": sum(1 for item in field_summaries.values() if item.get("classification") in {"static", "time_invariant"}),
        "time_metadata_field_count": sum(1 for item in field_summaries.values() if item.get("classification") == "time_metadata"),
        "top_fields_by_severity": ranked[:25],
        "top_dynamic_fields_by_rmse": dynamic_ranked[:25],
        "top_static_or_time_invariant_differences": static_ranked[:25],
        "top_drift_signals": drift_ranked[:25],
        "coverage_issues": coverage_issues,
        "tolerance_failure_count": len(tolerance_failures),
        "tolerance_failures": tolerance_failures[:25],
    }


def next_debug_recommendation(summaries: dict[str, Any]) -> str:
    static_top = summaries.get("top_static_or_time_invariant_differences", [])
    dynamic_top = summaries.get("top_dynamic_fields_by_rmse", [])
    drift_top = summaries.get("top_drift_signals", [])
    if static_top and float(static_top[0].get("max_abs") or 0.0) > 0.0:
        return (
            "Fix or explicitly root-cause static/grid/base-state mismatches first, then rerun this comparator before "
            "dycore, radiation, FP32, TOST, or Switzerland equivalence work."
        )
    if dynamic_top:
        names = ", ".join(row["field"] for row in dynamic_top[:5])
        return f"With static fields quiet, localize first dynamic divergence in {names} using same-state/tendency probes."
    if drift_top:
        return f"Investigate lead-dependent drift in {drift_top[0]['field']} with a first-bad-lead replay."
    return "Coverage is the current blocker; resolve missing/incompatible fields before interpreting physics parity."


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summaries = report["summaries"]
    pairing = report["pairing"]
    inv = report["inventory"]
    lines: list[str] = [
        "# V0.14 Wrfout Grid Comparison Smoke",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summaries['verdict']}`",
        f"- tolerance manifest: `{report['tolerances']['supplied']}`",
        f"- CPU-only: `{report['cpu_only']}`; GPU used: `{report['gpu_used']}`",
        f"- next: {next_debug_recommendation(summaries)}",
        "",
        "## Coverage",
        "",
        f"- domain `{pairing['domain']}`; paired files `{pairing['paired_file_count']}`; leads `{pairing['common_leads_h'][0]}`-`{pairing['common_leads_h'][-1]}` h",
        f"- CPU files `{pairing['cpu_file_count']}`; GPU files `{pairing['gpu_file_count']}`; unmatched CPU/GPU `{pairing['unmatched_cpu_count']}`/`{pairing['unmatched_gpu_count']}`",
        f"- variables CPU/GPU/common `{inv['cpu_variable_count']}`/`{inv['gpu_variable_count']}`/`{inv['common_variable_count']}`",
        f"- compared numeric `{summaries['comparable_field_count']}`; dynamic `{summaries['dynamic_field_count']}`; static/time-invariant `{summaries['static_or_time_invariant_field_count']}`; time metadata `{summaries['time_metadata_field_count']}`",
        "",
        "## Top 10 Field Differences",
        "",
        "| Field | Class | RMSE | Bias | p99 abs | Max abs | Worst lead |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries["top_fields_by_severity"][:10]:
        lines.append(
            f"| `{row['field']}` | {row['classification']} | {fmt_num(row.get('rmse'))} | {fmt_num(row.get('bias'))} | "
            f"{fmt_num(row.get('p99_abs'))} | {fmt_num(row.get('max_abs'))} | {row.get('worst_lead_h')} |"
        )
    lines.extend(["", "## Top 5 Drift Signals", "", "| Field | RMSE slope/h | Bias slope/h | Sign consistency | Worst lead RMSE |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in summaries["top_drift_signals"][:5]:
        lines.append(
            f"| `{row['field']}` | {fmt_num(row.get('rmse_slope_per_hour'))} | {fmt_num(row.get('bias_slope_per_hour'))} | "
            f"{fmt_num(row.get('bias_sign_consistency_fraction'))} | {fmt_num(row.get('worst_lead_rmse'))} |"
        )
    lines.extend(["", "## Top 5 Coverage Issues", ""])
    for issue in summaries["coverage_issues"][:5]:
        examples = issue.get("examples", [])
        if examples and isinstance(examples[0], str):
            sample = ", ".join(f"`{name}`" for name in examples[:8])
        else:
            sample = json.dumps(examples[:2], default=json_default)
        lines.append(f"- `{issue['kind']}` count `{issue['count']}`: {sample}")
    if not summaries["coverage_issues"]:
        lines.append("- none")
    lines.extend(["", "## Next Debug Recommendation", "", next_debug_recommendation(summaries), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    tolerances, tolerance_meta = load_tolerances(args.tolerance_json)
    pairs, pairing = build_pairs(args.cpu_dir, args.gpu_dir, args.domain, args.init, args.min_lead, args.max_lead)
    inventory = make_inventory(pairs)
    spatial_masks: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    split_counts: dict[str, Any] = {}
    split_warnings: list[str] = []
    if not args.no_spatial_splits:
        spatial_masks, split_counts, split_warnings = build_spatial_masks(Path(pairs[0]["cpu_file"]), args.boundary_width)

    field_summaries: dict[str, Any] = {}
    non_numeric: list[dict[str, Any]] = []
    incompatible: list[dict[str, Any]] = []
    common_names = inventory["common"]
    if args.vars:
        requested = set(args.vars)
        common_names = [name for name in common_names if name in requested]

    for i, name in enumerate(common_names, start=1):
        meta = inventory["first_metadata"][name]
        if name == "Times":
            field_summaries[name] = {
                **compare_time_metadata(name, pairs),
                "metadata": meta,
                "compared_lead_count": len(pairs),
            }
            continue
        ok, reason = compatible_metadata(meta["cpu"], meta["gpu"])
        if not ok:
            item = {"field": name, "reason": reason, "cpu": meta["cpu"], "gpu": meta["gpu"]}
            if reason == "non_numeric":
                non_numeric.append(item)
            else:
                incompatible.append(item)
            continue
        field_summaries[name] = compare_variable(name, pairs, meta, tolerances.get(name), spatial_masks)
        if args.progress and (i % args.progress == 0 or i == len(common_names)):
            print(f"compared {i}/{len(common_names)} fields: {name}", file=sys.stderr, flush=True)

    summaries = build_summaries(field_summaries, inventory, non_numeric, incompatible, bool(tolerances))
    elapsed = time.perf_counter() - start
    report = {
        "schema": "wrfout-grid-comparison-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_used": False,
        "inputs": {
            "cpu_dir": str(args.cpu_dir),
            "gpu_dir": str(args.gpu_dir),
            "domain": args.domain,
        },
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "note": "This comparator imports numpy and netCDF4 only; no JAX/CUDA/model execution is performed.",
        },
        "runtime": {
            "elapsed_seconds": clean_float(elapsed),
            "streaming_policy": "Variable-major: one variable and one lead pair are read at a time; no full campaign state is loaded.",
            "exact_percentiles": True,
            "memory_note": "Exact percentiles keep finite absolute-difference chunks for the current variable only, plus current/previous source arrays for time-invariance checks.",
        },
        "pairing": pairing,
        "inventory": {
            **{k: v for k, v in inventory.items() if k not in {"first_metadata", "first_seen_utc"}},
            "non_numeric_common": non_numeric,
            "incompatible_common": incompatible,
        },
        "tolerances": tolerance_meta,
        "spatial_splits": {
            "enabled": bool(spatial_masks),
            "boundary_width": args.boundary_width,
            "mask_counts": split_counts,
            "warnings": split_warnings,
        },
        "field_summaries": field_summaries,
        "summaries": summaries,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True, help="Directory containing CPU-WRF wrfout files.")
    parser.add_argument("--gpu-dir", type=Path, required=True, help="Directory containing GPU wrfout files.")
    parser.add_argument("--domain", required=True, help="Domain name such as d01, d02, or d03.")
    parser.add_argument("--init", help="Optional initialization time, ISO format. If omitted, inferred from path or earliest wrfout.")
    parser.add_argument("--min-lead", type=int, default=None)
    parser.add_argument("--max-lead", type=int, default=None)
    parser.add_argument("--vars", nargs="+", help="Optional field subset for small smoke/unit runs. Default compares all common fields.")
    parser.add_argument("--tolerance-json", type=Path, help="Optional predeclared tolerance manifest.")
    parser.add_argument("--boundary-width", type=int, default=5)
    parser.add_argument("--no-spatial-splits", action="store_true", help="Disable LANDMASK/HGT/quadrant/boundary split summaries.")
    parser.add_argument("--progress", type=int, default=0, help="Print a stderr progress line every N common fields.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(args)
    write_json(args.out_json, report)
    write_markdown(report, args.out_md)
    compact = {
        "verdict": report["summaries"]["verdict"],
        "paired_files": report["pairing"]["paired_file_count"],
        "compared_fields": report["summaries"]["comparable_field_count"],
        "dynamic_fields": report["summaries"]["dynamic_field_count"],
        "static_or_time_invariant_fields": report["summaries"]["static_or_time_invariant_field_count"],
        "top_field": report["summaries"]["top_fields_by_severity"][0] if report["summaries"]["top_fields_by_severity"] else None,
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    print(json.dumps(compact, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
