#!/usr/bin/env python3
"""CPU-only all-comparable-field grid-cell envelope for V0.14 attribution.

This proof compares the retained Case 3 GPU wrfouts against CPU-WRF truth for
every emitted writer field that is present in truth with compatible dimensions.
Cases without retained GPU wrfouts are represented from their stored case JSON
aggregates only.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 24-31 \
    python proofs/v014/grid_cell_envelope.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset, chartostring

from gpuwrf.io.wrfout_writer import OPERATIONAL_WRFOUT_VARIABLES, WRFOUT_VARIABLE_SPECS


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "proofs/v0120/powered_tost_n15"
OUT_JSON = ROOT / "proofs/v014/grid_cell_envelope.json"
OUT_MD = ROOT / "proofs/v014/grid_cell_envelope.md"
CPU_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output")
GPU_ROOT = Path("/tmp/v0120_powered_tost_runs")
WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")

DIAGNOSTIC_TOLERANCES = {"T2": 2.0, "U10": 2.5, "V10": 2.5}
TOST_MARGINS = {"T2": 0.215, "U10": 0.231, "V10": 0.275}
LEAD_BLOCKS = ("0-6h", "6-12h", "12-24h")

STATIC_AUDIT_FIELDS = {
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
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "F",
    "E",
    "SINALPHA",
    "COSALPHA",
    "XLAND",
    "P_TOP",
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
    "MAPFAC_MX",
    "MAPFAC_MY",
    "MAPFAC_UX",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_VY",
    "RDX",
    "RDY",
    "CLAT",
    "ISEEDARR_SPPT",
    "ISEEDARR_SKEBS",
    "ISEEDARRAY_SPP_CONV",
    "ISEEDARRAY_SPP_PBL",
    "ISEEDARRAY_SPP_LSM",
}

TIME_METADATA_FIELDS = {"Times", "XTIME"}

SURFACE_MINIMUM_FIELDS = (
    "T2",
    "Q2",
    "U10",
    "V10",
    "PSFC",
    "TSK",
    "PBLH",
    "UST",
    "HFX",
    "LH",
    "SWDOWN",
    "GLW",
    "RAINC",
    "RAINNC",
    "RAINSH",
)

DYNAMICS_MINIMUM_FIELDS = (
    "U",
    "V",
    "W",
    "T",
    "QVAPOR",
    "P",
    "PB",
    "PH",
    "PHB",
    "MU",
    "MUB",
)

MICROPHYSICS_MINIMUM_FIELDS = (
    "QCLOUD",
    "QICE",
    "QRAIN",
    "QSNOW",
    "QGRAUP",
    "QNICE",
    "QNRAIN",
    "QNSNOW",
    "QNGRAUPEL",
    "QNCLOUD",
    "QNCCN",
    "QKE",
)


def clean_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def safe_mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def parse_init_time(run_id: str) -> datetime:
    parts = run_id.split("_")
    ds = parts[0]
    hour = int(parts[1].replace("z", ""))
    return datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]), hour, tzinfo=timezone.utc)


def wrfout_map(path: Path, domain: str = "d02") -> dict[datetime, Path]:
    out: dict[datetime, Path] = {}
    for p in sorted(path.glob(f"wrfout_{domain}_*")):
        m = WRFOUT_RE.match(p.name)
        if not m or not p.is_file():
            continue
        vt = datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        out[vt] = p
    return out


def lead_hour(init: datetime, valid_time: datetime) -> int:
    return int(round((valid_time - init).total_seconds() / 3600.0))


def lead_block(hour: int) -> str | None:
    if 0 < hour <= 6:
        return "0-6h"
    if 6 < hour <= 12:
        return "6-12h"
    if 12 < hour <= 24:
        return "12-24h"
    return None


def load_case_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def station_tost_0_24h(case: dict[str, Any]) -> dict[str, Any]:
    tost = case.get("tost_pairs", {}).get("per_block", {})
    out = {}
    for field_name, margin in TOST_MARGINS.items():
        out[field_name] = {
            **tost.get(field_name, {}).get("0-24h", {}),
            "equivalence_margin": margin,
        }
    return out


def var_units(ds: Dataset, name: str) -> str:
    return str(getattr(ds.variables[name], "units", ""))


def read_var(ds: Dataset, name: str) -> np.ndarray:
    var = ds.variables[name]
    if var.dimensions and var.dimensions[0] == "Time":
        arr = var[0]
    else:
        arr = var[:]
    return np.asarray(np.ma.filled(arr, np.nan))


def read_string_var(ds: Dataset, name: str) -> str:
    arr = read_var(ds, name)
    if arr.dtype.kind in {"S", "U"}:
        try:
            converted = chartostring(arr)
            if np.ndim(converted) == 0:
                return str(converted.item())
            return str(np.asarray(converted).ravel()[0])
        except Exception:
            return "".join(x.decode("ascii", errors="replace") if isinstance(x, bytes) else str(x) for x in arr.ravel())
    return str(arr)


def is_numeric_array(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)


def shape_without_time(ds: Dataset, name: str) -> tuple[int, ...]:
    return tuple(read_var(ds, name).shape)


def dims_for(ds: Dataset, name: str) -> tuple[str, ...]:
    dims = tuple(ds.variables[name].dimensions)
    if dims and dims[0] == "Time":
        return dims[1:]
    return dims


@dataclass
class RunningStats:
    tolerance: float | None = None
    n: int = 0
    sum_diff: float = 0.0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0
    within_tol: int = 0
    sum_gpu: float = 0.0
    sum_cpu: float = 0.0
    sum_gpu_sq: float = 0.0
    sum_cpu_sq: float = 0.0
    sum_gpu_cpu: float = 0.0
    abs_chunks: list[np.ndarray] = field(default_factory=list)

    def update(self, gpu: np.ndarray, cpu: np.ndarray, mask2d: np.ndarray | None = None) -> None:
        g = np.asarray(gpu, dtype=np.float64)
        c = np.asarray(cpu, dtype=np.float64)
        valid = np.isfinite(g) & np.isfinite(c)
        if mask2d is not None:
            full_mask = broadcast_horizontal_mask(mask2d, g.shape)
            valid &= full_mask
        if not np.any(valid):
            return
        gv = g[valid]
        cv = c[valid]
        diff = gv - cv
        abs_diff = np.abs(diff)
        self.n += int(diff.size)
        self.sum_diff += float(np.sum(diff, dtype=np.float64))
        self.sum_abs += float(np.sum(abs_diff, dtype=np.float64))
        self.sum_sq += float(np.sum(diff * diff, dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(abs_diff)))
        if self.tolerance is not None:
            self.within_tol += int(np.sum(abs_diff <= self.tolerance))
        self.sum_gpu += float(np.sum(gv, dtype=np.float64))
        self.sum_cpu += float(np.sum(cv, dtype=np.float64))
        self.sum_gpu_sq += float(np.sum(gv * gv, dtype=np.float64))
        self.sum_cpu_sq += float(np.sum(cv * cv, dtype=np.float64))
        self.sum_gpu_cpu += float(np.sum(gv * cv, dtype=np.float64))
        self.abs_chunks.append(abs_diff.astype(np.float32, copy=False))

    def finish(self, include_pearson: bool = True) -> dict[str, Any]:
        if self.n == 0:
            return {
                "count": 0,
                "bias": None,
                "rmse": None,
                "mae": None,
                "p95_abs": None,
                "p99_abs": None,
                "max_abs": None,
                "frac_within_tolerance": None if self.tolerance is None else 0.0,
                "tolerance": self.tolerance,
                "pearson_r": None,
            }
        abs_all = np.concatenate(self.abs_chunks) if self.abs_chunks else np.asarray([], dtype=np.float32)
        pearson = None
        if include_pearson:
            pearson = pearson_from_sums(
                self.n,
                self.sum_gpu,
                self.sum_cpu,
                self.sum_gpu_sq,
                self.sum_cpu_sq,
                self.sum_gpu_cpu,
            )
        return {
            "count": int(self.n),
            "bias": clean_float(self.sum_diff / self.n),
            "rmse": clean_float(math.sqrt(self.sum_sq / self.n)),
            "mae": clean_float(self.sum_abs / self.n),
            "p95_abs": clean_float(np.percentile(abs_all, 95)) if abs_all.size else None,
            "p99_abs": clean_float(np.percentile(abs_all, 99)) if abs_all.size else None,
            "max_abs": clean_float(self.max_abs),
            "frac_within_tolerance": clean_float(self.within_tol / self.n) if self.tolerance is not None else None,
            "tolerance": self.tolerance,
            "pearson_r": pearson,
        }


@dataclass
class PairCorr:
    n: int = 0
    sum_a: float = 0.0
    sum_b: float = 0.0
    sum_a_sq: float = 0.0
    sum_b_sq: float = 0.0
    sum_ab: float = 0.0

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        valid = np.isfinite(aa) & np.isfinite(bb)
        if not np.any(valid):
            return
        av = aa[valid]
        bv = bb[valid]
        self.n += int(av.size)
        self.sum_a += float(np.sum(av, dtype=np.float64))
        self.sum_b += float(np.sum(bv, dtype=np.float64))
        self.sum_a_sq += float(np.sum(av * av, dtype=np.float64))
        self.sum_b_sq += float(np.sum(bv * bv, dtype=np.float64))
        self.sum_ab += float(np.sum(av * bv, dtype=np.float64))

    def finish(self) -> dict[str, Any]:
        return {
            "count": int(self.n),
            "pearson_r": pearson_from_sums(self.n, self.sum_a, self.sum_b, self.sum_a_sq, self.sum_b_sq, self.sum_ab),
        }


def pearson_from_sums(
    n: int,
    sum_a: float,
    sum_b: float,
    sum_a_sq: float,
    sum_b_sq: float,
    sum_ab: float,
) -> float | None:
    if n < 2:
        return None
    cov_num = n * sum_ab - sum_a * sum_b
    var_a = n * sum_a_sq - sum_a * sum_a
    var_b = n * sum_b_sq - sum_b * sum_b
    if var_a <= 0.0 or var_b <= 0.0:
        return None
    return clean_float(cov_num / math.sqrt(var_a * var_b))


def stats_from_arrays(gpu: np.ndarray, cpu: np.ndarray, tolerance: float | None = None) -> dict[str, Any]:
    acc = RunningStats(tolerance=tolerance)
    acc.update(gpu, cpu)
    return acc.finish()


def broadcast_horizontal_mask(mask2d: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if len(target_shape) < 2:
        raise ValueError(f"cannot broadcast horizontal mask to shape {target_shape}")
    if tuple(target_shape[-2:]) != tuple(mask2d.shape):
        raise ValueError(f"mask shape {mask2d.shape} does not match target horizontal shape {target_shape[-2:]}")
    return np.broadcast_to(mask2d.reshape((1,) * (len(target_shape) - 2) + mask2d.shape), target_shape)


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
    numeric = edge_average_x(arr.astype(np.float64))
    return numeric > 0.5


def edge_bool_y(arr: np.ndarray) -> np.ndarray:
    numeric = edge_average_y(arr.astype(np.float64))
    return numeric > 0.5


def dilate_bool(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    out = mask.copy()
    ny, nx = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            src_y0 = max(0, -dy)
            src_y1 = min(ny, ny - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(nx, nx - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(ny, ny + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(nx, nx + dx)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def coast_mask_from_land(land: np.ndarray) -> np.ndarray:
    land_bool = land > 0.5
    boundary = np.zeros_like(land_bool, dtype=bool)
    boundary[:, 1:] |= land_bool[:, 1:] != land_bool[:, :-1]
    boundary[:, :-1] |= land_bool[:, 1:] != land_bool[:, :-1]
    boundary[1:, :] |= land_bool[1:, :] != land_bool[:-1, :]
    boundary[:-1, :] |= land_bool[1:, :] != land_bool[:-1, :]
    return dilate_bool(boundary, radius=2)


def label_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {str(label): labels == label for label in sorted({str(x) for x in labels.ravel().tolist()})}


def make_split_masks(cpu_first: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    with Dataset(cpu_first) as ds:
        hgt = np.asarray(read_var(ds, "HGT"), dtype=np.float64)
        land = np.asarray(read_var(ds, "LANDMASK"), dtype=np.float64)
        lat = np.asarray(read_var(ds, "XLAT"), dtype=np.float64)
        lon = np.asarray(read_var(ds, "XLONG"), dtype=np.float64)

    land_bool = land > 0.5
    elev_labels = np.full(hgt.shape, "ocean", dtype=object)
    elev_labels[land_bool & (hgt < 300.0)] = "land_0_300m"
    elev_labels[land_bool & (hgt >= 300.0) & (hgt < 1000.0)] = "land_300_1000m"
    elev_labels[land_bool & (hgt >= 1000.0)] = "land_gt_1000m"

    lat_mid = float(np.nanmedian(lat))
    lon_mid = float(np.nanmedian(lon))
    quadrant = np.full(hgt.shape, "SW", dtype=object)
    quadrant[(lat >= lat_mid) & (lon < lon_mid)] = "NW"
    quadrant[(lat >= lat_mid) & (lon >= lon_mid)] = "NE"
    quadrant[(lat < lat_mid) & (lon >= lon_mid)] = "SE"

    coast = coast_mask_from_land(land)
    land_labels = np.where(land_bool, "land", "ocean")
    coast_labels = np.where(coast, "coast_band", "non_coast")

    mass = {
        "land_ocean": label_masks(land_labels),
        "elevation": label_masks(elev_labels),
        "quadrant": label_masks(quadrant),
        "coast_band": label_masks(coast_labels),
    }

    u_land = edge_bool_x(land_bool)
    u_hgt = edge_average_x(hgt)
    u_lat = edge_average_x(lat)
    u_lon = edge_average_x(lon)
    u_coast = edge_bool_x(coast)
    u_elev = np.full(u_hgt.shape, "ocean", dtype=object)
    u_elev[u_land & (u_hgt < 300.0)] = "land_0_300m"
    u_elev[u_land & (u_hgt >= 300.0) & (u_hgt < 1000.0)] = "land_300_1000m"
    u_elev[u_land & (u_hgt >= 1000.0)] = "land_gt_1000m"
    u_quad = np.full(u_hgt.shape, "SW", dtype=object)
    u_quad[(u_lat >= lat_mid) & (u_lon < lon_mid)] = "NW"
    u_quad[(u_lat >= lat_mid) & (u_lon >= lon_mid)] = "NE"
    u_quad[(u_lat < lat_mid) & (u_lon >= lon_mid)] = "SE"
    u = {
        "land_ocean": label_masks(np.where(u_land, "land", "ocean")),
        "elevation": label_masks(u_elev),
        "quadrant": label_masks(u_quad),
        "coast_band": label_masks(np.where(u_coast, "coast_band", "non_coast")),
    }

    v_land = edge_bool_y(land_bool)
    v_hgt = edge_average_y(hgt)
    v_lat = edge_average_y(lat)
    v_lon = edge_average_y(lon)
    v_coast = edge_bool_y(coast)
    v_elev = np.full(v_hgt.shape, "ocean", dtype=object)
    v_elev[v_land & (v_hgt < 300.0)] = "land_0_300m"
    v_elev[v_land & (v_hgt >= 300.0) & (v_hgt < 1000.0)] = "land_300_1000m"
    v_elev[v_land & (v_hgt >= 1000.0)] = "land_gt_1000m"
    v_quad = np.full(v_hgt.shape, "SW", dtype=object)
    v_quad[(v_lat >= lat_mid) & (v_lon < lon_mid)] = "NW"
    v_quad[(v_lat >= lat_mid) & (v_lon >= lon_mid)] = "NE"
    v_quad[(v_lat < lat_mid) & (v_lon >= lon_mid)] = "SE"
    v = {
        "land_ocean": label_masks(np.where(v_land, "land", "ocean")),
        "elevation": label_masks(v_elev),
        "quadrant": label_masks(v_quad),
        "coast_band": label_masks(np.where(v_coast, "coast_band", "non_coast")),
    }

    return {"mass": mass, "u": u, "v": v}


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


def field_category(name: str) -> str:
    if name in SURFACE_MINIMUM_FIELDS:
        return "surface_diagnostic"
    if name in DYNAMICS_MINIMUM_FIELDS:
        return "dynamics_thermodynamics"
    if name in MICROPHYSICS_MINIMUM_FIELDS or name == "CLDFRA":
        return "microphysics_cloud"
    if name in {"SWDNB", "SWUPB", "LWDNB", "LWUPB", "SWDNT", "SWUPT", "LWDNT", "LWUPT", "OLR", "SWNORM"}:
        return "radiation_flux"
    if name in {"SNOWNC", "GRAUPELNC", "SR", "SNOWC"}:
        return "precip_snow_diagnostic"
    if name in {
        "QFX",
        "GRDFLX",
        "TH2",
        "TSLB",
        "SMOIS",
        "SH2O",
        "SNOW",
        "SNOWH",
        "CANWAT",
        "SFROFF",
        "UDROFF",
        "ALBEDO",
        "EMISS",
        "TSNO",
        "SNICE",
        "SNLIQ",
        "ZSNSO",
        "ISNOW",
        "SNEQVO",
        "CANLIQ",
        "CANICE",
    }:
        return "land_surface_state"
    if name == "COSZEN":
        return "derived_solar_diagnostic"
    return "other_dynamic"


def comparable_inventory(gpu_first: Path, cpu_first: Path) -> dict[str, Any]:
    with Dataset(gpu_first) as gds, Dataset(cpu_first) as cds:
        gpu_vars = set(gds.variables)
        cpu_vars = set(cds.variables)
        writer_vars = list(OPERATIONAL_WRFOUT_VARIABLES)
        emitted_writer_vars = [name for name in writer_vars if name in gpu_vars]
        missing_from_gpu = [name for name in writer_vars if name not in gpu_vars]
        missing_from_cpu = [name for name in emitted_writer_vars if name not in cpu_vars]

        dynamic_fields: list[str] = []
        static_fields: list[str] = []
        time_fields: list[str] = []
        incompatible: list[dict[str, Any]] = []
        non_numeric: list[dict[str, Any]] = []
        comparable_metadata: dict[str, Any] = {}

        for name in emitted_writer_vars:
            if name not in cpu_vars:
                continue
            gdims = dims_for(gds, name)
            cdims = dims_for(cds, name)
            gshape = shape_without_time(gds, name)
            cshape = shape_without_time(cds, name)
            meta = {
                "gpu_dims": list(gdims),
                "cpu_dims": list(cdims),
                "gpu_shape": list(gshape),
                "cpu_shape": list(cshape),
                "gpu_units": var_units(gds, name),
                "cpu_units": var_units(cds, name),
            }
            if gshape != cshape:
                incompatible.append({"name": name, "reason": "shape_mismatch", **meta})
                continue
            if gdims != cdims:
                incompatible.append({"name": name, "reason": "dimension_name_mismatch", **meta})
                continue
            if name == "Times":
                time_fields.append(name)
                comparable_metadata[name] = meta
                continue

            garr = read_var(gds, name)
            carr = read_var(cds, name)
            if not is_numeric_array(garr) or not is_numeric_array(carr):
                non_numeric.append({"name": name, "reason": "non_numeric_not_times", **meta})
                continue

            comparable_metadata[name] = meta
            if name in TIME_METADATA_FIELDS:
                time_fields.append(name)
            elif name in STATIC_AUDIT_FIELDS:
                static_fields.append(name)
            else:
                dynamic_fields.append(name)

    return {
        "writer_operational_fields": writer_vars,
        "writer_operational_field_count": len(writer_vars),
        "gpu_emitted_writer_fields": emitted_writer_vars,
        "gpu_emitted_writer_field_count": len(emitted_writer_vars),
        "missing_from_gpu": missing_from_gpu,
        "missing_from_cpu_truth": missing_from_cpu,
        "incompatible": incompatible,
        "non_numeric_uncompared": non_numeric,
        "dynamic_fields": dynamic_fields,
        "static_audit_fields": static_fields,
        "time_metadata_fields": time_fields,
        "comparable_metadata": comparable_metadata,
    }


def audit_static_fields(
    fields: list[str],
    metadata: dict[str, Any],
    common: list[datetime],
    gm: dict[datetime, Path],
    cm: dict[datetime, Path],
    init: datetime,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in fields:
        tol = 0.0
        exact_all = True
        max_abs = 0.0
        max_bias = 0.0
        n_checked = 0
        mismatch_leads: list[int] = []
        first_lead_stats: dict[str, Any] | None = None
        for t in common:
            hour = lead_hour(init, t)
            with Dataset(gm[t]) as gds, Dataset(cm[t]) as cds:
                g = read_var(gds, name)
                c = read_var(cds, name)
            if g.shape != c.shape:
                exact_all = False
                mismatch_leads.append(hour)
                continue
            st = stats_from_arrays(g, c, tolerance=tol)
            n_checked += int(st["count"] or 0)
            if first_lead_stats is None:
                first_lead_stats = st
            lead_max = st.get("max_abs")
            lead_bias = st.get("bias")
            if lead_max is not None:
                max_abs = max(max_abs, float(lead_max))
            if lead_bias is not None:
                max_bias = max(max_bias, abs(float(lead_bias)))
            if (lead_max or 0.0) != 0.0:
                exact_all = False
                mismatch_leads.append(hour)
        out[name] = {
            "category": "static_grid_audit",
            "metadata": metadata[name],
            "audited_leads": [lead_hour(init, t) for t in common],
            "count_checked_total": int(n_checked),
            "exact_all_checked_leads": bool(exact_all),
            "max_abs_across_checked_leads": clean_float(max_abs),
            "max_abs_bias_across_checked_leads": clean_float(max_bias),
            "first_mismatch_leads": mismatch_leads[:10],
            "first_lead_stats": first_lead_stats,
            "note": "Static/grid field audited separately from prognostic RMSE.",
        }
    return out


def audit_time_metadata(
    fields: list[str],
    metadata: dict[str, Any],
    common: list[datetime],
    gm: dict[datetime, Path],
    cm: dict[datetime, Path],
    init: datetime,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in fields:
        if name == "Times":
            checks = []
            all_equal = True
            for t in common:
                hour = lead_hour(init, t)
                with Dataset(gm[t]) as gds, Dataset(cm[t]) as cds:
                    gs = read_string_var(gds, name)
                    cs = read_string_var(cds, name)
                equal = gs == cs
                all_equal = all_equal and equal
                checks.append({"lead_h": hour, "gpu": gs, "cpu": cs, "equal": equal})
            out[name] = {
                "category": "time_metadata",
                "metadata": metadata[name],
                "all_equal": all_equal,
                "checks": checks,
            }
            continue

        acc = RunningStats(tolerance=0.0)
        by_lead = []
        for t in common:
            hour = lead_hour(init, t)
            with Dataset(gm[t]) as gds, Dataset(cm[t]) as cds:
                g = read_var(gds, name)
                c = read_var(cds, name)
            st = stats_from_arrays(g, c, tolerance=0.0)
            by_lead.append({"lead_h": hour, **st})
            acc.update(g, c)
        out[name] = {
            "category": "time_metadata",
            "metadata": metadata[name],
            "overall": acc.finish(),
            "by_lead": by_lead,
        }
    return out


def analyze_dynamic_field(
    name: str,
    metadata: dict[str, Any],
    common: list[datetime],
    gm: dict[datetime, Path],
    cm: dict[datetime, Path],
    init: datetime,
    split_masks: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    tol = DIAGNOSTIC_TOLERANCES.get(name)
    overall = RunningStats(tolerance=tol)
    blocks = {block: RunningStats(tolerance=tol) for block in LEAD_BLOCKS}
    by_lead: list[dict[str, Any]] = []
    missing_leads: list[dict[str, Any]] = []

    dims = tuple(metadata[name]["gpu_dims"])
    kind = horizontal_kind(dims)
    split_accs: dict[str, dict[str, RunningStats]] = {}
    if kind is not None:
        for split_name, masks in split_masks[kind].items():
            split_accs[split_name] = {label: RunningStats(tolerance=tol) for label in masks}

    for t in common:
        hour = lead_hour(init, t)
        block = lead_block(hour)
        with Dataset(gm[t]) as gds, Dataset(cm[t]) as cds:
            if name not in gds.variables or name not in cds.variables:
                missing_leads.append({"lead_h": hour, "reason": "field_missing_in_one_file"})
                continue
            g = read_var(gds, name)
            c = read_var(cds, name)
        if g.shape != c.shape:
            missing_leads.append({
                "lead_h": hour,
                "reason": "shape_mismatch",
                "gpu_shape": list(g.shape),
                "cpu_shape": list(c.shape),
            })
            continue
        st = stats_from_arrays(g, c, tolerance=tol)
        by_lead.append({"lead_h": hour, **st})
        overall.update(g, c)
        if block is not None:
            blocks[block].update(g, c)
        if kind is not None:
            for split_name, masks in split_masks[kind].items():
                for label, mask in masks.items():
                    split_accs[split_name][label].update(g, c, mask2d=mask)

    split_report: dict[str, dict[str, Any]] = {}
    for split_name, labels in split_accs.items():
        split_report[split_name] = {label: acc.finish() for label, acc in labels.items()}

    worst_leads = sorted(
        (x for x in by_lead if x.get("rmse") is not None),
        key=lambda x: float(x["rmse"]),
        reverse=True,
    )[:5]
    return {
        "category": field_category(name),
        "metadata": metadata[name],
        "overall": overall.finish(),
        "by_lead": by_lead,
        "by_lead_block": {block: acc.finish() for block, acc in blocks.items()},
        "spatial_splits": split_report,
        "worst_leads_by_rmse": worst_leads,
        "missing_or_bad_leads": missing_leads,
    }


def diff_for_field(gds: Dataset, cds: Dataset, name: str) -> np.ndarray | None:
    if name not in gds.variables or name not in cds.variables:
        return None
    g = read_var(gds, name)
    c = read_var(cds, name)
    if g.shape != c.shape or not is_numeric_array(g) or not is_numeric_array(c):
        return None
    return np.asarray(g, dtype=np.float64) - np.asarray(c, dtype=np.float64)


def cross_field_correlations(common: list[datetime], gm: dict[datetime, Path], cm: dict[datetime, Path]) -> dict[str, Any]:
    fields = ("U10", "V10", "T2", "PSFC")
    pairs = [(a, b) for i, a in enumerate(fields) for b in fields[i + 1 :]]
    accs = {f"d{a}_d{b}": PairCorr() for a, b in pairs}
    by_lead: list[dict[str, Any]] = []
    for t in common:
        with Dataset(gm[t]) as gds, Dataset(cm[t]) as cds:
            diffs = {name: diff_for_field(gds, cds, name) for name in fields}
        lead_item: dict[str, Any] = {}
        for a, b in pairs:
            key = f"d{a}_d{b}"
            if diffs[a] is None or diffs[b] is None:
                continue
            accs[key].update(diffs[a], diffs[b])
            lead_acc = PairCorr()
            lead_acc.update(diffs[a], diffs[b])
            lead_item[key] = lead_acc.finish()["pearson_r"]
        if lead_item:
            lead_item["lead_time"] = t.isoformat()
            by_lead.append(lead_item)
    return {
        "fields": list(fields),
        "overall": {key: acc.finish() for key, acc in accs.items()},
        "by_lead": by_lead,
    }


def aggregate_only_case(case: dict[str, Any], writer_vars: list[str]) -> dict[str, Any]:
    fields = case.get("cell_level", {}).get("field_stats", {})
    aggregate_fields = {}
    for name, st in fields.items():
        aggregate_fields[name] = {
            "overall": {
                "count": st.get("n_cells"),
                "bias": st.get("bias"),
                "rmse": st.get("rmse"),
                "mae": st.get("mae"),
                "p95_abs": st.get("p95"),
                "p99_abs": st.get("p99"),
                "max_abs": st.get("max"),
                "frac_within_tolerance": st.get("frac_within_tol"),
                "tolerance": case.get("cell_level", {}).get("cell_tol", {}).get(name),
                "pearson_r": st.get("pearson_r"),
            },
            "by_lead_block": {
                block: {
                    "count": bst.get("n_cells"),
                    "bias": bst.get("bias"),
                    "rmse": bst.get("rmse"),
                    "mae": bst.get("mae"),
                    "p95_abs": bst.get("p95"),
                    "p99_abs": bst.get("p99"),
                    "max_abs": bst.get("max"),
                    "frac_within_tolerance": bst.get("frac_within_tol"),
                    "tolerance": case.get("cell_level", {}).get("cell_tol", {}).get(name),
                    "pearson_r": None,
                }
                for block, bst in st.get("by_lead_block", {}).items()
            },
        }
    aggregate_names = set(aggregate_fields)
    return {
        "run_id": case.get("run_id"),
        "source": "case_json_aggregate_only",
        "reason_spatial_unavailable": "Retained GPU wrfout directory is not available; only stored case JSON aggregates can be used.",
        "gpu_dir": case.get("gpu_dir"),
        "cpu_dir": case.get("cpu_dir"),
        "aggregate_fields": aggregate_fields,
        "aggregate_field_names": sorted(aggregate_names),
        "spatial_unavailable_writer_fields": [name for name in writer_vars if name not in aggregate_names],
        "station_tost_0_24h": station_tost_0_24h(case),
    }


def summarize_case3(case: dict[str, Any]) -> dict[str, Any]:
    field_metrics = case.get("field_metrics", {})
    ranked = sorted(
        (
            {
                "field": name,
                "category": item.get("category"),
                "rmse": item.get("overall", {}).get("rmse"),
                "bias": item.get("overall", {}).get("bias"),
                "p95_abs": item.get("overall", {}).get("p95_abs"),
                "max_abs": item.get("overall", {}).get("max_abs"),
            }
            for name, item in field_metrics.items()
            if item.get("overall", {}).get("rmse") is not None
        ),
        key=lambda x: float(x["rmse"]),
        reverse=True,
    )
    min_fields = set(SURFACE_MINIMUM_FIELDS + DYNAMICS_MINIMUM_FIELDS + MICROPHYSICS_MINIMUM_FIELDS)
    ranked_min = [x for x in ranked if x["field"] in min_fields]
    static = case.get("static_audit", {})
    static_mismatches = [
        {
            "field": name,
            "max_abs": item.get("max_abs_across_checked_leads"),
            "first_mismatch_leads": item.get("first_mismatch_leads"),
        }
        for name, item in static.items()
        if not item.get("exact_all_checked_leads", False)
    ]
    static_mismatches = sorted(
        static_mismatches,
        key=lambda x: float(x.get("max_abs") or 0.0),
        reverse=True,
    )
    return {
        "worst_dynamic_fields_by_rmse": ranked[:20],
        "worst_minimum_fields_by_rmse": ranked_min[:20],
        "surface_minimum_summary": {
            name: field_metrics[name]["overall"]
            for name in SURFACE_MINIMUM_FIELDS
            if name in field_metrics
        },
        "dynamics_minimum_summary": {
            name: field_metrics[name]["overall"]
            for name in DYNAMICS_MINIMUM_FIELDS
            if name in field_metrics
        },
        "microphysics_minimum_summary": {
            name: field_metrics[name]["overall"]
            for name in MICROPHYSICS_MINIMUM_FIELDS
            if name in field_metrics
        },
        "static_mismatch_count": len(static_mismatches),
        "static_mismatches": static_mismatches[:30],
    }


def ranked_hypotheses(report: dict[str, Any]) -> list[dict[str, Any]]:
    spatial_cases = [c for c in report["cases"] if c.get("source") == "spatial_grid_wrfouts"]
    if not spatial_cases:
        return []
    case = spatial_cases[0]
    metrics = case.get("field_metrics", {})
    summary = case.get("case_summary", {})
    static_mismatch_count = summary.get("static_mismatch_count", 0)
    top_static = summary.get("static_mismatches", [])[:6]
    top_static_text = ", ".join(
        f"{item['field']} max {fmt(item.get('max_abs'))}" for item in top_static
    )

    def rmse(name: str) -> float | None:
        return metrics.get(name, {}).get("overall", {}).get("rmse")

    def bias(name: str) -> float | None:
        return metrics.get(name, {}).get("overall", {}).get("bias")

    u10 = rmse("U10") or 0.0
    v10 = rmse("V10") or 0.0
    psfc = rmse("PSFC") or 0.0
    t2 = rmse("T2") or 0.0
    u3 = rmse("U") or 0.0
    v3 = rmse("V") or 0.0
    swdown = rmse("SWDOWN") or 0.0
    glw = rmse("GLW") or 0.0
    coszen = rmse("COSZEN") or 0.0

    hypotheses: list[dict[str, Any]] = []
    if static_mismatch_count:
        hypotheses.append({
            "rank": 1,
            "hypothesis": "WRF vertical-coordinate / grid-metric payload mismatch is the first root-cause target.",
            "evidence": (
                f"{static_mismatch_count} audited static fields are not exact across checked leads; "
                f"largest: {top_static_text}."
            ),
            "next_probe": (
                "Diff DycoreMetrics/GridSpec against CPU wrfinput or CPU wrfout before any timestep, "
                "then rerun this envelope after the metric payload is exact."
            ),
            "confidence": "high",
        })
    else:
        hypotheses.append({
            "rank": 1,
            "hypothesis": "Primary divergence is in time-stepped dynamics/surface coupling, not static grid placement.",
            "evidence": "All audited static coordinate/metric fields are exact while U10/V10/PSFC/T2 diverge.",
            "next_probe": "Freeze static-grid ownership and bisect the first one to three timesteps through dycore plus surface diagnostics.",
            "confidence": "high",
        })

    if max(u3, v3, psfc) > 1.0 or psfc > 100.0:
        hypotheses.append({
            "rank": 2,
            "hypothesis": "Pressure-gradient / mass-wind coupling in the dycore is the leading operator-level suspect.",
            "evidence": (
                f"3D wind RMSE U={u3:.3g}, V={v3:.3g}; PSFC RMSE={psfc:.3g}; "
                f"surface wind RMSE U10={u10:.3g}, V10={v10:.3g}."
            ),
            "next_probe": "Instrument first-timestep MU/P/PH pressure-gradient tendencies and U/V updates against WRF savepoints.",
            "confidence": "high" if psfc > 100.0 else "medium",
        })
    else:
        hypotheses.append({
            "rank": 2,
            "hypothesis": "Near-surface diagnostic mapping may dominate because 3D winds are smaller than U10/V10 errors.",
            "evidence": f"U10/V10 RMSE are {u10:.3g}/{v10:.3g} while U/V RMSE are {u3:.3g}/{v3:.3g}.",
            "next_probe": "Compare surface-layer input winds, rotation, roughness, stability, and diagnostic U10/V10 at fixed model state.",
            "confidence": "medium",
        })

    hypotheses.append({
        "rank": 3,
        "hypothesis": "Radiation or surface-energy diagnostics are a secondary amplifier of T2/TSK/PBL divergence.",
        "evidence": (
            f"SWDOWN RMSE={swdown:.3g}, GLW RMSE={glw:.3g}, COSZEN RMSE={coszen:.3g}, "
            f"T2 RMSE={t2:.3g}; T2 bias={bias('T2')}."
        ),
        "next_probe": "Keep radiation diagnostics in the next proof gate, but do not fix them before the first-step wind/mass budget is isolated.",
        "confidence": "medium" if max(swdown, glw) > 10.0 else "low",
    })
    return hypotheses


def analyze_spatial_case(case: dict[str, Any], gpu_dir: Path, cpu_dir: Path) -> dict[str, Any]:
    run_id = str(case["run_id"])
    init = parse_init_time(run_id)
    gm = wrfout_map(gpu_dir)
    cm = wrfout_map(cpu_dir)
    common = sorted(t for t in set(gm) & set(cm) if 0 < (t - init).total_seconds() / 3600.0 <= 24)
    if not common:
        return {
            "run_id": run_id,
            "source": "spatial_attempt",
            "status": "NO_COMMON_WRFOUTS",
            "gpu_dir": str(gpu_dir),
            "cpu_dir": str(cpu_dir),
            "station_tost_0_24h": station_tost_0_24h(case),
        }

    inventory = comparable_inventory(gm[common[0]], cm[common[0]])
    split_masks = make_split_masks(cm[common[0]])
    metadata = inventory["comparable_metadata"]

    field_metrics = {}
    for index, name in enumerate(inventory["dynamic_fields"], start=1):
        print(f"[{index:03d}/{len(inventory['dynamic_fields']):03d}] comparing {name}", flush=True)
        field_metrics[name] = analyze_dynamic_field(name, metadata, common, gm, cm, init, split_masks)

    static_audit = audit_static_fields(inventory["static_audit_fields"], metadata, common, gm, cm, init)
    time_metadata = audit_time_metadata(inventory["time_metadata_fields"], metadata, common, gm, cm, init)

    item = {
        "run_id": run_id,
        "source": "spatial_grid_wrfouts",
        "gpu_dir": str(gpu_dir),
        "cpu_dir": str(cpu_dir),
        "domain": "d02",
        "lead_hours": [lead_hour(init, t) for t in common],
        "n_common_leads": len(common),
        "inventory": {k: v for k, v in inventory.items() if k != "comparable_metadata"},
        "field_metrics": field_metrics,
        "static_audit": static_audit,
        "time_metadata": time_metadata,
        "cross_field_correlations": cross_field_correlations(common, gm, cm),
        "station_tost_0_24h": station_tost_0_24h(case),
    }
    item["case_summary"] = summarize_case3(item)
    return item


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = [
        "# V0.14 Grid-Cell Envelope",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "This is a grid-field attribution artifact, not a station-skill result and not an equivalence pass.",
        "",
        "## Coverage",
        "",
        f"- writer operational fields declared: `{report['writer_inventory']['writer_operational_field_count']}`",
        f"- spatial wrfout cases: `{report['summary']['n_spatial_cases']}`",
        f"- aggregate-only cases: `{report['summary']['n_aggregate_only_cases']}`",
        "",
    ]

    for case in report["cases"]:
        lines.append(f"## {case['run_id']}")
        lines.append("")
        lines.append(f"- source: `{case['source']}`")
        if case["source"] == "case_json_aggregate_only":
            lines.append(f"- limitation: {case['reason_spatial_unavailable']}")
            lines.append(f"- aggregate fields available: `{', '.join(case['aggregate_field_names'])}`")
            for name in ("T2", "U10", "V10"):
                st = case["aggregate_fields"].get(name, {}).get("overall", {})
                if st:
                    lines.append(
                        f"- {name}: RMSE `{fmt(st.get('rmse'))}`, bias `{fmt(st.get('bias'))}`, "
                        f"p95 `{fmt(st.get('p95_abs'))}`, max `{fmt(st.get('max_abs'))}`"
                    )
            lines.append(f"- spatial unavailable writer fields: `{len(case['spatial_unavailable_writer_fields'])}`")
            lines.append("")
            continue

        inv = case["inventory"]
        lines.append(f"- common leads compared: `{case['n_common_leads']}` ({case['lead_hours'][0]}-{case['lead_hours'][-1]} h)")
        lines.append(f"- dynamic fields with RMSE envelope: `{len(inv['dynamic_fields'])}`")
        lines.append(f"- static/grid fields audited separately: `{len(inv['static_audit_fields'])}`")
        lines.append(f"- time metadata fields audited: `{len(inv['time_metadata_fields'])}`")
        lines.append(f"- writer fields not emitted in retained GPU wrfouts: `{len(inv['missing_from_gpu'])}`")
        lines.append(f"- emitted writer fields missing in CPU truth: `{len(inv['missing_from_cpu_truth'])}`")
        lines.append(f"- incompatible fields: `{len(inv['incompatible'])}`")
        lines.append("")

        lines.append("### Minimum Dynamic Fields")
        lines.append("")
        lines.append("| field | count | bias | RMSE | MAE | p95 abs | p99 abs | max abs | frac tol | r |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name in SURFACE_MINIMUM_FIELDS + DYNAMICS_MINIMUM_FIELDS + MICROPHYSICS_MINIMUM_FIELDS:
            item = case["field_metrics"].get(name)
            if not item:
                continue
            st = item["overall"]
            lines.append(
                f"| `{name}` | {st.get('count')} | {fmt(st.get('bias'))} | {fmt(st.get('rmse'))} | "
                f"{fmt(st.get('mae'))} | {fmt(st.get('p95_abs'))} | {fmt(st.get('p99_abs'))} | "
                f"{fmt(st.get('max_abs'))} | {fmt(st.get('frac_within_tolerance'))} | {fmt(st.get('pearson_r'))} |"
            )
        lines.append("")

        summary = case["case_summary"]
        lines.append("### Worst Dynamic Fields")
        lines.append("")
        for item in summary["worst_dynamic_fields_by_rmse"][:12]:
            lines.append(
                f"- `{item['field']}` ({item['category']}): RMSE `{fmt(item['rmse'])}`, "
                f"bias `{fmt(item['bias'])}`, p95 `{fmt(item['p95_abs'])}`, max `{fmt(item['max_abs'])}`"
            )
        lines.append("")

        lines.append("### Lead Blocks")
        lines.append("")
        for name in ("T2", "U10", "V10", "PSFC", "U", "V", "W", "T"):
            item = case["field_metrics"].get(name)
            if not item:
                continue
            parts = []
            for block in LEAD_BLOCKS:
                st = item["by_lead_block"].get(block, {})
                parts.append(f"{block} RMSE {fmt(st.get('rmse'))} bias {fmt(st.get('bias'))}")
            lines.append(f"- `{name}`: " + "; ".join(parts))
        lines.append("")

        lines.append("### Spatial Splits")
        lines.append("")
        for name in ("T2", "U10", "V10", "PSFC", "U", "V", "W", "SWDOWN", "GLW"):
            item = case["field_metrics"].get(name)
            if not item:
                continue
            splits = item.get("spatial_splits", {})
            land = splits.get("land_ocean", {})
            elev = splits.get("elevation", {})
            quad = splits.get("quadrant", {})
            lines.append(f"- `{name}` land/ocean: {compact_split(land)}")
            lines.append(f"- `{name}` elevation: {compact_split(elev)}")
            lines.append(f"- `{name}` quadrant: {compact_split(quad)}")
        lines.append("")

        lines.append("### Inventory Exceptions")
        lines.append("")
        lines.append(f"- missing from retained GPU wrfouts: `{', '.join(inv['missing_from_gpu']) or 'none'}`")
        lines.append(f"- emitted but missing from CPU truth: `{', '.join(inv['missing_from_cpu_truth']) or 'none'}`")
        if inv["incompatible"]:
            for bad in inv["incompatible"]:
                lines.append(f"- incompatible `{bad['name']}`: {bad['reason']} GPU {bad['gpu_shape']} CPU {bad['cpu_shape']}")
        else:
            lines.append("- incompatible: `none`")
        mismatches = summary.get("static_mismatches", [])
        lines.append(f"- static/grid mismatch count: `{summary.get('static_mismatch_count')}`")
        for item in mismatches[:12]:
            lines.append(
                f"- static mismatch `{item['field']}`: max abs `{fmt(item.get('max_abs'))}`, "
                f"first leads `{item.get('first_mismatch_leads')}`"
            )
        lines.append("")

    lines.append("## Ranked Root-Cause Hypotheses")
    lines.append("")
    for item in report.get("ranked_root_cause_hypotheses", []):
        lines.append(f"{item['rank']}. **{item['hypothesis']}**")
        lines.append(f"   Evidence: {item['evidence']}")
        lines.append(f"   Next probe: {item['next_probe']}")
        lines.append(f"   Confidence: `{item['confidence']}`")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n")


def fmt(value: Any) -> str:
    x = clean_float(value)
    if x is None:
        return "NA"
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1000.0 or ax < 0.001:
        return f"{x:.3e}"
    return f"{x:.3f}"


def compact_split(split: dict[str, Any]) -> str:
    if not split:
        return "NA"
    parts = []
    for label, st in split.items():
        parts.append(f"{label} RMSE {fmt(st.get('rmse'))} bias {fmt(st.get('bias'))}")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args(argv)

    writer_vars = list(OPERATIONAL_WRFOUT_VARIABLES)
    cases: list[dict[str, Any]] = []
    for case_path in sorted(CASE_DIR.glob("case_*.json")):
        case = load_case_json(case_path)
        run_id = case["run_id"]
        gpu_dir = Path(case.get("gpu_dir", "")) if case.get("gpu_dir") else GPU_ROOT / f"l2_d02_{run_id}"
        cpu_dir = Path(case.get("cpu_dir", "")) if case.get("cpu_dir") else CPU_ROOT / run_id
        if gpu_dir.is_dir() and cpu_dir.is_dir():
            cases.append(analyze_spatial_case(case, gpu_dir, cpu_dir))
        else:
            cases.append(aggregate_only_case(case, writer_vars))

    report = {
        "schema": "v014-grid-cell-envelope",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract": ".agent/sprints/2026-06-08-v014-grid-parity-attribution/sprint-contract.md",
        "writer_inventory": {
            "writer_operational_fields": writer_vars,
            "writer_operational_field_count": len(writer_vars),
            "writer_spec_field_count": len(WRFOUT_VARIABLE_SPECS),
            "source": "src/gpuwrf/io/wrfout_writer.py",
        },
        "inputs": {
            "case_dir": str(CASE_DIR),
            "cpu_root": str(CPU_ROOT),
            "gpu_root": str(GPU_ROOT),
            "diagnostic_tolerances": DIAGNOSTIC_TOLERANCES,
            "note": "Only T2/U10/V10 have predeclared diagnostic tolerances from retained case JSON.",
        },
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_spatial_cases": sum(1 for c in cases if c.get("source") == "spatial_grid_wrfouts"),
            "n_aggregate_only_cases": sum(1 for c in cases if c.get("source") == "case_json_aggregate_only"),
        },
    }
    report["ranked_root_cause_hypotheses"] = ranked_hypotheses(report)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
