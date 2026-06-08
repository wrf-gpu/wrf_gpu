#!/usr/bin/env python3
"""CPU-only dynamic field attribution probe for V0.14 Case 3.

This proof reads retained GPU wrfouts and CPU-WRF truth wrfouts. It does not
run the model, import JAX, edit src, or use a GPU.

Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
    python proofs/v014/dynamic_field_attribution.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DOMAIN = "d02"
CPU_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
GPU_DIR = Path("/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z")
GRID_COMPARATOR_JSON = ROOT / "proofs/v014/grid_comparison_framework_smoke.json"
STATIC_PARITY_JSON = ROOT / "proofs/v014/static_metric_base_parity.json"
WIND_MASS_JSON = ROOT / "proofs/v014/wind_mass_divergence_probe.json"
SAME_STATE_INVENTORY_JSON = ROOT / "proofs/v014/same_state_tendency_inventory.json"
OUT_JSON = ROOT / "proofs/v014/dynamic_field_attribution.json"
OUT_MD = ROOT / "proofs/v014/dynamic_field_attribution.md"
WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")

CORE_FIELDS = (
    "PSFC",
    "MU",
    "P",
    "PH",
    "U",
    "V",
    "U10",
    "V10",
    "T",
    "QVAPOR",
    "W",
    "PBLH",
)

RADIATION_CANDIDATES = (
    "SWDOWN",
    "SWDNB",
    "SWNORM",
    "GLW",
    "OLR",
    "SWUPB",
    "LWUPB",
    "LWDNB",
    "SWDNT",
    "SWUPT",
    "LWDNT",
    "LWUPT",
    "SWDNTC",
    "SWUPTC",
    "LWDNTC",
    "LWUPTC",
    "ACSWDNB",
    "ACSWUPB",
    "ACLWDNB",
    "ACLWUPB",
    "ACSWDNT",
    "ACSWUPT",
    "ACLWDNT",
    "ACLWUPT",
)

STATIC_WRITER_FIELDS = {
    "PB",
    "PHB",
    "MUB",
    "HGT",
    "XLAT",
    "XLONG",
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "MAPFAC_MX",
    "MAPFAC_MY",
    "MAPFAC_UX",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_VY",
    "C1H",
    "C2H",
    "C3H",
    "C4H",
    "C1F",
    "C2F",
    "C3F",
    "C4F",
    "CF1",
    "CF2",
    "CF3",
    "CFN",
    "CFN1",
    "DN",
    "DNW",
    "RDN",
    "RDNW",
    "FNM",
    "FNP",
    "ZNU",
    "ZNW",
}

# Report-only thresholds. They are not pass/fail tolerances and are used only
# to rank leads, fields, and cells for the next same-state localization sprint.
REPORT_RMSE_THRESHOLDS = {
    "U10": 1.5,
    "V10": 1.5,
    "U": 2.0,
    "V": 2.0,
    "W": 0.05,
    "T": 1.0,
    "QVAPOR": 5.0e-4,
    "PSFC": 150.0,
    "MU": 150.0,
    "P": 100.0,
    "PH": 150.0,
    "PBLH": 100.0,
    "SWDOWN": 100.0,
    "SWDNB": 100.0,
    "SWNORM": 100.0,
    "GLW": 50.0,
    "OLR": 50.0,
    "SWUPB": 100.0,
    "LWUPB": 50.0,
    "LWDNB": 50.0,
    "SWDNT": 100.0,
    "SWUPT": 100.0,
    "LWDNT": 50.0,
    "LWUPT": 50.0,
}

CELL_SCORE_SCALES = {
    "dU10": 1.5,
    "dV10": 1.5,
    "dU_k0": 2.0,
    "dV_k0": 2.0,
    "dPSFC": 150.0,
    "dMU": 150.0,
    "dP_k0": 100.0,
    "dPH_k0": 150.0,
    "dW_k0": 0.05,
    "dT_k0": 1.0,
    "dQVAPOR_k0": 5.0e-4,
    "dPBLH": 100.0,
    "dSWDOWN": 100.0,
    "dSWDNB": 100.0,
    "dSWNORM": 100.0,
    "dGLW": 50.0,
}

CELL_SCORE_WEIGHTS = {
    "dV10": 2.0,
    "dU10": 1.5,
    "dV_k0": 2.0,
    "dU_k0": 1.5,
    "dPSFC": 1.0,
    "dMU": 1.0,
    "dP_k0": 1.0,
    "dPH_k0": 1.0,
    "dW_k0": 0.6,
    "dPBLH": 0.5,
}

CORRELATION_NAMES = (
    "dU10",
    "dV10",
    "dU_k0",
    "dV_k0",
    "dPSFC",
    "dMU",
    "dP_k0",
    "dPH_k0",
)

PRIMARY_LOCALIZATION_FIELDS = ("U10", "V10", "U", "V", "PSFC", "MU", "P", "PH")
H10_H14 = range(10, 15)
H8_H14 = range(8, 15)
FRAME_CELLS = 5
LOCALIZATION_INTERIOR_CELLS = 8
PATCH_RADIUS = 8
CELL_CANDIDATE_COUNT = 160
SELECTED_CELL_COUNT = 24


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_init_time(run_id: str) -> datetime:
    parts = run_id.split("_")
    ds = parts[0]
    hour = int(parts[1].replace("z", ""))
    return datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]), hour, tzinfo=timezone.utc)


def wrfout_map(path: Path, domain: str = DOMAIN) -> dict[datetime, Path]:
    out: dict[datetime, Path] = {}
    for p in sorted(path.glob(f"wrfout_{domain}_*")):
        m = WRFOUT_RE.match(p.name)
        if not m or not p.is_file():
            continue
        vt = datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        out[vt] = p
    return out


def read_var(ds: Dataset, name: str) -> tuple[np.ndarray, tuple[str, ...], str | None] | None:
    if name not in ds.variables:
        return None
    var = ds.variables[name]
    if var.dimensions and var.dimensions[0] == "Time":
        arr = var[0]
        dims = tuple(var.dimensions[1:])
    else:
        arr = var[:]
        dims = tuple(var.dimensions)
    units = getattr(var, "units", None)
    data = np.asarray(np.ma.filled(arr, np.nan), dtype=np.float64)
    return data, dims, units


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def clean_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def stats_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    total = int(arr.size)
    vals = finite_values(arr)
    if vals.size == 0:
        return {
            "n": 0,
            "finite_pair": 0,
            "total": total,
            "finite_pair_fraction": 0.0 if total else None,
            "status": "NO_FINITE",
        }
    abs_vals = np.abs(vals)
    finite_mask = np.isfinite(arr)
    max_idx = np.unravel_index(
        int(np.nanargmax(np.where(finite_mask, np.abs(arr), np.nan))),
        arr.shape,
    )
    return {
        "n": int(vals.size),
        "finite_pair": int(vals.size),
        "total": total,
        "finite_pair_fraction": float(vals.size / total) if total else None,
        "bias": float(np.mean(vals)),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
        "mae": float(np.mean(abs_vals)),
        "p95_abs": float(np.percentile(abs_vals, 95.0)),
        "p99_abs": float(np.percentile(abs_vals, 99.0)),
        "max_abs": float(np.max(abs_vals)),
        "max_index": [int(i) for i in max_idx],
        "value_at_max": float(arr[max_idx]),
    }


class StatAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.total = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.sumabs = 0.0
        self.maxabs = 0.0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        self.total += int(arr.size)
        vals = finite_values(arr)
        if vals.size == 0:
            return
        self.n += int(vals.size)
        self.sum += float(np.sum(vals))
        self.sumsq += float(np.sum(vals * vals))
        self.sumabs += float(np.sum(np.abs(vals)))
        self.maxabs = max(self.maxabs, float(np.max(np.abs(vals))))

    def finalize(self) -> dict[str, Any]:
        if self.n == 0:
            return {
                "n": 0,
                "finite_pair": 0,
                "total": int(self.total),
                "finite_pair_fraction": 0.0 if self.total else None,
                "status": "NO_FINITE",
            }
        return {
            "n": int(self.n),
            "finite_pair": int(self.n),
            "total": int(self.total),
            "finite_pair_fraction": float(self.n / self.total) if self.total else None,
            "bias": float(self.sum / self.n),
            "rmse": float(math.sqrt(self.sumsq / self.n)),
            "mae": float(self.sumabs / self.n),
            "max_abs": float(self.maxabs),
        }


class CorrAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        xx = np.asarray(x, dtype=np.float64).ravel()
        yy = np.asarray(y, dtype=np.float64).ravel()
        if xx.shape != yy.shape:
            raise ValueError(f"correlation shape mismatch: {xx.shape} vs {yy.shape}")
        mask = np.isfinite(xx) & np.isfinite(yy)
        if not np.any(mask):
            return
        xx = xx[mask]
        yy = yy[mask]
        self.n += int(xx.size)
        self.sx += float(np.sum(xx))
        self.sy += float(np.sum(yy))
        self.sxx += float(np.sum(xx * xx))
        self.syy += float(np.sum(yy * yy))
        self.sxy += float(np.sum(xx * yy))

    def finalize(self) -> dict[str, Any]:
        if self.n < 2:
            return {"n": int(self.n), "pearson": None}
        cov = self.sxy - (self.sx * self.sy / self.n)
        vx = self.sxx - (self.sx * self.sx / self.n)
        vy = self.syy - (self.sy * self.sy / self.n)
        if vx <= 0.0 or vy <= 0.0:
            return {"n": int(self.n), "pearson": None}
        return {"n": int(self.n), "pearson": float(cov / math.sqrt(vx * vy))}


def make_pair_accumulators(names: tuple[str, ...]) -> dict[str, CorrAccumulator]:
    return {
        f"{a}__{b}": CorrAccumulator()
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def update_pair_accumulators(accs: dict[str, CorrAccumulator], values: Mapping[str, np.ndarray]) -> None:
    names = [name for name in CORRELATION_NAMES if name in values]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}__{b}"
            if key in accs:
                accs[key].update(values[a], values[b])


def finalize_pair_accumulators(accs: dict[str, CorrAccumulator]) -> dict[str, Any]:
    return {key: acc.finalize() for key, acc in sorted(accs.items())}


def corr_matrix(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    accs = make_pair_accumulators(CORRELATION_NAMES)
    update_pair_accumulators(accs, values)
    return finalize_pair_accumulators(accs)


def top_mask(arr: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.abs(np.asarray(arr, dtype=np.float64)).ravel()
    finite_idx = np.flatnonzero(np.isfinite(flat))
    mask = np.zeros(flat.shape, dtype=bool)
    if finite_idx.size == 0:
        return mask.reshape(arr.shape)
    n = max(1, int(round(finite_idx.size * fraction)))
    vals = flat[finite_idx]
    local = np.argpartition(vals, -n)[-n:]
    mask[finite_idx[local]] = True
    return mask.reshape(arr.shape)


def colocation_matrix(values: Mapping[str, np.ndarray], fraction: float = 0.01) -> dict[str, Any]:
    names = [name for name in CORRELATION_NAMES if name in values]
    masks = {name: top_mask(values[name], fraction) for name in names}
    out: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ma = masks[a]
            mb = masks[b]
            inter = int(np.sum(ma & mb))
            union = int(np.sum(ma | mb))
            ca = int(np.sum(ma))
            cb = int(np.sum(mb))
            out[f"{a}__{b}"] = {
                "top_fraction": fraction,
                "count_a": ca,
                "count_b": cb,
                "intersection": inter,
                "jaccard": float(inter / union) if union else None,
                "overlap_fraction_of_smaller_top_set": float(inter / min(ca, cb)) if min(ca, cb) else None,
            }
    return out


def destagger_x(arr: np.ndarray) -> np.ndarray:
    return 0.5 * (arr[..., :-1] + arr[..., 1:])


def destagger_y(arr: np.ndarray) -> np.ndarray:
    return 0.5 * (arr[..., :-1, :] + arr[..., 1:, :])


def destagger_z(arr: np.ndarray) -> np.ndarray:
    return 0.5 * (arr[:-1, ...] + arr[1:, ...])


def as_mass_grid(field: str, diff: np.ndarray) -> np.ndarray | None:
    if field == "U":
        return destagger_x(diff)
    if field == "V":
        return destagger_y(diff)
    if field in ("W", "PH"):
        return destagger_z(diff)
    if diff.ndim in (2, 3):
        return diff
    return None


def level_axis(dims: tuple[str, ...]) -> int | None:
    for i, dim in enumerate(dims):
        if dim in ("bottom_top", "bottom_top_stag", "soil_layers_stag"):
            return i
    return None


def level_slice(arr: np.ndarray, axis: int, k: int) -> np.ndarray:
    return np.take(arr, indices=k, axis=axis)


def mask_values(arr: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[mask2d]
    if arr.ndim == 3:
        return arr[:, mask2d]
    raise ValueError(f"unsupported mask array rank {arr.ndim}")


def make_masks(hgt: np.ndarray, land: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, Any]:
    land_mask = land > 0.5
    ocean_mask = ~land_mask
    lat_mid = float(np.nanmedian(lat))
    lon_mid = float(np.nanmedian(lon))
    sn, we = hgt.shape

    frame = np.zeros_like(hgt, dtype=bool)
    frame[:FRAME_CELLS, :] = True
    frame[-FRAME_CELLS:, :] = True
    frame[:, :FRAME_CELLS] = True
    frame[:, -FRAME_CELLS:] = True

    localization_frame = np.zeros_like(hgt, dtype=bool)
    localization_frame[:LOCALIZATION_INTERIOR_CELLS, :] = True
    localization_frame[-LOCALIZATION_INTERIOR_CELLS:, :] = True
    localization_frame[:, :LOCALIZATION_INTERIOR_CELLS] = True
    localization_frame[:, -LOCALIZATION_INTERIOR_CELLS:] = True

    masks = {
        "land_ocean": {
            "ocean": ocean_mask,
            "land": land_mask,
        },
        "elevation": {
            "ocean": ocean_mask,
            "land_0_300m": land_mask & (hgt < 300.0),
            "land_300_1000m": land_mask & (hgt >= 300.0) & (hgt < 1000.0),
            "land_gt_1000m": land_mask & (hgt >= 1000.0),
        },
        "quadrant": {
            "NW": (lat >= lat_mid) & (lon < lon_mid),
            "NE": (lat >= lat_mid) & (lon >= lon_mid),
            "SW": (lat < lat_mid) & (lon < lon_mid),
            "SE": (lat < lat_mid) & (lon >= lon_mid),
        },
        "boundary": {
            f"frame_{FRAME_CELLS}cells": frame,
            f"interior_excluding_{FRAME_CELLS}cell_frame": ~frame,
            f"interior_excluding_{LOCALIZATION_INTERIOR_CELLS}cell_frame": ~localization_frame,
        },
        "same_state_preference": {
            "interior_ocean_or_low_terrain": (~localization_frame) & (ocean_mask | (hgt < 300.0)),
            "interior_ocean": (~localization_frame) & ocean_mask,
            "interior_low_terrain_land": (~localization_frame) & land_mask & (hgt < 300.0),
        },
    }
    counts = {
        group: {name: int(np.sum(mask)) for name, mask in group_masks.items()}
        for group, group_masks in masks.items()
    }
    counts["grid_shape"] = {"south_north": int(sn), "west_east": int(we)}
    counts["lat_lon_split"] = {"lat_mid": lat_mid, "lon_mid": lon_mid}
    return {"masks": masks, "counts": counts}


def region_bins(y: int, x: int, hgt: np.ndarray, land: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, str]:
    lat_mid = float(np.nanmedian(lat))
    lon_mid = float(np.nanmedian(lon))
    is_land = bool(land[y, x] > 0.5)
    elev = float(hgt[y, x])
    if not is_land:
        elev_bin = "ocean"
    elif elev < 300.0:
        elev_bin = "land_0_300m"
    elif elev < 1000.0:
        elev_bin = "land_300_1000m"
    else:
        elev_bin = "land_gt_1000m"
    if lat[y, x] >= lat_mid and lon[y, x] < lon_mid:
        quad = "NW"
    elif lat[y, x] >= lat_mid:
        quad = "NE"
    elif lon[y, x] < lon_mid:
        quad = "SW"
    else:
        quad = "SE"
    sn, we = hgt.shape
    in_frame = y < FRAME_CELLS or y >= sn - FRAME_CELLS or x < FRAME_CELLS or x >= we - FRAME_CELLS
    in_localization_frame = (
        y < LOCALIZATION_INTERIOR_CELLS
        or y >= sn - LOCALIZATION_INTERIOR_CELLS
        or x < LOCALIZATION_INTERIOR_CELLS
        or x >= we - LOCALIZATION_INTERIOR_CELLS
    )
    return {
        "land_ocean": "land" if is_land else "ocean",
        "elevation": elev_bin,
        "quadrant": quad,
        "boundary": f"frame_{FRAME_CELLS}cells" if in_frame else f"interior_excluding_{FRAME_CELLS}cell_frame",
        "same_state_preference": "edge_excluded" if in_localization_frame else "interior_candidate",
    }


def mass_low_values(mass_diffs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "U10" in mass_diffs:
        out["dU10"] = mass_diffs["U10"]
    if "V10" in mass_diffs:
        out["dV10"] = mass_diffs["V10"]
    if "PSFC" in mass_diffs:
        out["dPSFC"] = mass_diffs["PSFC"]
    if "MU" in mass_diffs:
        out["dMU"] = mass_diffs["MU"]
    if "PBLH" in mass_diffs:
        out["dPBLH"] = mass_diffs["PBLH"]
    for field, out_name in (
        ("U", "dU_k0"),
        ("V", "dV_k0"),
        ("P", "dP_k0"),
        ("PH", "dPH_k0"),
        ("W", "dW_k0"),
        ("T", "dT_k0"),
        ("QVAPOR", "dQVAPOR_k0"),
    ):
        arr = mass_diffs.get(field)
        if arr is not None and arr.ndim == 3 and arr.shape[0] > 0:
            out[out_name] = arr[0]
    for field in ("SWDOWN", "SWDNB", "SWNORM", "GLW"):
        if field in mass_diffs:
            out[f"d{field}"] = mass_diffs[field]
    return out


def safe_native_value(arr: np.ndarray | None, index: tuple[int, ...]) -> float | None:
    if arr is None:
        return None
    try:
        return clean_float(arr[index])
    except IndexError:
        return None


def stagger_context(
    y: int,
    x: int,
    hgt_shape: tuple[int, int],
    native_diffs: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    sn, we = hgt_shape
    y0 = max(0, y - PATCH_RADIUS)
    y1 = min(sn, y + PATCH_RADIUS + 1)
    x0 = max(0, x - PATCH_RADIUS)
    x1 = min(we, x + PATCH_RADIUS + 1)
    u = native_diffs.get("U")
    v = native_diffs.get("V")
    w = native_diffs.get("W")
    ph = native_diffs.get("PH")
    return {
        "mass_cell": {"bottom_top": 0, "south_north": y, "west_east": x},
        "patch_bounds_mass_grid": {
            "halo_radius_cells": PATCH_RADIUS,
            "south_north_start": int(y0),
            "south_north_stop_exclusive": int(y1),
            "west_east_start": int(x0),
            "west_east_stop_exclusive": int(x1),
        },
        "adjacent_native_faces_k0": {
            "U": [
                {"bottom_top": 0, "south_north": y, "west_east_stag": x},
                {"bottom_top": 0, "south_north": y, "west_east_stag": x + 1},
            ],
            "V": [
                {"bottom_top": 0, "south_north_stag": y, "west_east": x},
                {"bottom_top": 0, "south_north_stag": y + 1, "west_east": x},
            ],
            "W": [
                {"bottom_top_stag": 0, "south_north": y, "west_east": x},
                {"bottom_top_stag": 1, "south_north": y, "west_east": x},
            ],
            "PH": [
                {"bottom_top_stag": 0, "south_north": y, "west_east": x},
                {"bottom_top_stag": 1, "south_north": y, "west_east": x},
            ],
        },
        "adjacent_native_diff_values_k0": {
            "U": [
                safe_native_value(u, (0, y, x)),
                safe_native_value(u, (0, y, x + 1)),
            ],
            "V": [
                safe_native_value(v, (0, y, x)),
                safe_native_value(v, (0, y + 1, x)),
            ],
            "W": [
                safe_native_value(w, (0, y, x)),
                safe_native_value(w, (1, y, x)),
            ],
            "PH": [
                safe_native_value(ph, (0, y, x)),
                safe_native_value(ph, (1, y, x)),
            ],
        },
        "vertical_column_context": {
            "all_native_vertical_levels_required_for_operator_probe": True,
            "U_column_faces": {"south_north": y, "west_east_stag_faces": [x, x + 1]},
            "V_column_faces": {"south_north_stag_faces": [y, y + 1], "west_east": x},
            "W_PH_column_faces": {"south_north": y, "west_east": x},
        },
    }


def top_cell_candidates(
    *,
    lead_h: int,
    valid_time: datetime,
    low_values: Mapping[str, np.ndarray],
    native_diffs: Mapping[str, np.ndarray],
    hgt: np.ndarray,
    land: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    top_n: int = CELL_CANDIDATE_COUNT,
) -> list[dict[str, Any]]:
    if not all(name in low_values for name in ("dU10", "dV10", "dPSFC", "dMU", "dP_k0", "dPH_k0")):
        return []
    sn, we = hgt.shape
    score = np.zeros((sn, we), dtype=np.float64)
    hit_count = np.zeros((sn, we), dtype=np.int32)
    normalized: dict[str, np.ndarray] = {}
    for name, arr in low_values.items():
        scale = CELL_SCORE_SCALES.get(name)
        if scale is None:
            continue
        weight = CELL_SCORE_WEIGHTS.get(name, 0.25)
        norm = np.abs(arr) / scale
        norm = np.where(np.isfinite(norm), norm, 0.0)
        normalized[name] = norm
        score += weight * np.clip(norm, 0.0, 12.0)
        hit_count += (norm >= 1.0).astype(np.int32)

    local_frame = np.zeros((sn, we), dtype=bool)
    local_frame[:LOCALIZATION_INTERIOR_CELLS, :] = True
    local_frame[-LOCALIZATION_INTERIOR_CELLS:, :] = True
    local_frame[:, :LOCALIZATION_INTERIOR_CELLS] = True
    local_frame[:, -LOCALIZATION_INTERIOR_CELLS:] = True
    preferred = (~local_frame) & ((land <= 0.5) | (hgt < 300.0))
    finite = np.isfinite(score)
    selection_mask = preferred & finite
    if int(np.sum(selection_mask)) < top_n:
        selection_mask = (~local_frame) & finite
    if int(np.sum(selection_mask)) < top_n:
        selection_mask = finite

    flat = np.where(selection_mask, score, -np.inf).ravel()
    valid_idx = np.flatnonzero(np.isfinite(flat) & (flat > -np.inf))
    if valid_idx.size == 0:
        return []
    n = min(top_n, int(valid_idx.size))
    local = np.argpartition(flat[valid_idx], -n)[-n:]
    flat_idxs = valid_idx[local]
    flat_idxs = flat_idxs[np.argsort(flat[flat_idxs])[::-1]]

    candidates: list[dict[str, Any]] = []
    for rank, flat_idx in enumerate(flat_idxs, start=1):
        y, x = np.unravel_index(int(flat_idx), score.shape)
        component_values: dict[str, Any] = {}
        component_severity: dict[str, Any] = {}
        for name, arr in low_values.items():
            if arr.shape != score.shape:
                continue
            component_values[name] = clean_float(arr[y, x])
            if name in normalized:
                component_severity[name] = clean_float(normalized[name][y, x])
        top_components = sorted(
            [
                {"name": name, "severity_ratio": value}
                for name, value in component_severity.items()
                if value is not None
            ],
            key=lambda item: item["severity_ratio"],
            reverse=True,
        )[:8]
        candidates.append(
            {
                "candidate_rank_in_lead": rank,
                "lead_h": int(lead_h),
                "valid_time_utc": valid_time.isoformat(),
                "mass_index": {"south_north": int(y), "west_east": int(x)},
                "lat": clean_float(lat[y, x]),
                "lon": clean_float(lon[y, x]),
                "hgt_m": clean_float(hgt[y, x]),
                "landmask": clean_float(land[y, x]),
                "region_bins": region_bins(int(y), int(x), hgt, land, lat, lon),
                "co_located_component_hit_count": int(hit_count[y, x]),
                "composite_score": clean_float(score[y, x]),
                "field_diffs": component_values,
                "component_severity_ratios": component_severity,
                "top_components": top_components,
                "stagger_context": stagger_context(int(y), int(x), hgt.shape, native_diffs),
            }
        )
    return candidates


def select_diverse_cells(candidates: list[dict[str, Any]], count: int = SELECTED_CELL_COUNT) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for min_sep in (2, 1, 0):
        for cand in candidates:
            idx = cand["mass_index"]
            y = int(idx["south_north"])
            x = int(idx["west_east"])
            key = (y, x)
            if key in seen:
                continue
            if min_sep > 0:
                too_close = any(
                    max(abs(y - int(sel["mass_index"]["south_north"])), abs(x - int(sel["mass_index"]["west_east"]))) <= min_sep
                    for sel in selected
                )
                if too_close:
                    continue
            selected.append(dict(cand))
            seen.add(key)
            if len(selected) >= count:
                break
        if len(selected) >= count:
            break
    for rank, cand in enumerate(selected, start=1):
        cand["selection_rank"] = rank
    return selected


def detect_fields(first_cpu: Path, first_gpu: Path) -> tuple[list[str], list[str], list[str]]:
    with Dataset(first_cpu, "r") as cds, Dataset(first_gpu, "r") as gds:
        common = set(cds.variables) & set(gds.variables)
    radiation_present = [field for field in RADIATION_CANDIDATES if field in common]
    missing_core = [field for field in CORE_FIELDS if field not in common]
    fields = [field for field in CORE_FIELDS if field in common] + radiation_present
    return fields, radiation_present, missing_core


def summarize_top_levels(per_level: list[dict[str, Any]], key: str, n: int = 10) -> list[dict[str, Any]]:
    valid = [item for item in per_level if item.get("stats", {}).get("n", 0)]
    return sorted(valid, key=lambda x: x["stats"].get(key, -1.0), reverse=True)[:n]


def field_threshold(field: str) -> float | None:
    if field in REPORT_RMSE_THRESHOLDS:
        return REPORT_RMSE_THRESHOLDS[field]
    return None


def severity_ratio(field: str, stat: Mapping[str, Any]) -> float | None:
    threshold = field_threshold(field)
    rmse = clean_float(stat.get("rmse"))
    if threshold is None or rmse is None or threshold <= 0.0:
        return None
    return float(rmse / threshold)


def build_field_summaries(
    *,
    fields: list[str],
    overall: Mapping[str, dict[str, Any]],
    per_lead: Mapping[str, Mapping[str, Any]],
    grid_fields: Mapping[str, Any],
    ignored_static_fields: set[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        ost = overall.get(field)
        if not ost:
            continue
        threshold = field_threshold(field)
        lead_items = []
        for lead_key in sorted(per_lead, key=lambda x: int(x)):
            stat = per_lead[lead_key].get(field, {})
            if stat.get("n", 0):
                lead_items.append({"lead_h": int(lead_key), **stat})
        worst = max(lead_items, key=lambda item: item.get("rmse", -1.0), default=None)
        first_bad = None
        if threshold is not None:
            for item in lead_items:
                rmse = clean_float(item.get("rmse"))
                if rmse is not None and rmse >= threshold:
                    first_bad = {
                        "lead_h": item["lead_h"],
                        "rmse": rmse,
                        "threshold": threshold,
                        "severity_ratio": float(rmse / threshold),
                    }
                    break
        grid_summary = grid_fields.get(field, {})
        comparator_overall = grid_summary.get("overall", {}) if isinstance(grid_summary, dict) else {}
        out[field] = {
            "overall": ost,
            "grid_comparator_overall": {
                key: comparator_overall.get(key)
                for key in ("rmse", "bias", "p99_abs", "max_abs", "pearson_r", "finite_pair_fraction")
                if key in comparator_overall
            },
            "threshold_rmse_report_only": threshold,
            "overall_severity_ratio": severity_ratio(field, ost),
            "first_bad_lead": first_bad,
            "worst_lead_by_rmse": worst,
            "grid_comparator_classification": grid_summary.get("classification") if isinstance(grid_summary, dict) else None,
            "ignored_from_dynamic_ranking": field in ignored_static_fields,
        }
    return out


def build_lead_analysis(per_lead: Mapping[str, Mapping[str, Any]], fields: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lead_key in sorted(per_lead, key=lambda x: int(x)):
        item = per_lead[lead_key]
        hits = []
        ratios = {}
        primary_ratios = []
        for field in fields:
            stat = item.get(field, {})
            ratio = severity_ratio(field, stat)
            if ratio is None:
                continue
            ratios[field] = ratio
            if ratio >= 1.0:
                hits.append(
                    {
                        "field": field,
                        "rmse": clean_float(stat.get("rmse")),
                        "threshold": field_threshold(field),
                        "severity_ratio": ratio,
                    }
                )
            if field in PRIMARY_LOCALIZATION_FIELDS:
                primary_ratios.append(ratio)
        hits.sort(key=lambda x: x["severity_ratio"], reverse=True)
        out[lead_key] = {
            "lead_h": int(lead_key),
            "threshold_hit_count": len(hits),
            "threshold_hits": hits,
            "max_severity_ratio": max(ratios.values()) if ratios else None,
            "primary_localization_score": float(sum(primary_ratios) / len(primary_ratios)) if primary_ratios else None,
            "primary_localization_ratios": ratios,
        }
    return out


def first_materially_bad_lead(lead_analysis: Mapping[str, Any]) -> dict[str, Any] | None:
    for lead_key in sorted(lead_analysis, key=lambda x: int(x)):
        item = lead_analysis[lead_key]
        if item.get("threshold_hit_count", 0) > 0:
            return item
    return None


def worst_lead_by_score(lead_analysis: Mapping[str, Any], allowed: range | None = None) -> dict[str, Any] | None:
    items = []
    for key, item in lead_analysis.items():
        lead_h = int(key)
        if allowed is not None and lead_h not in allowed:
            continue
        score = clean_float(item.get("primary_localization_score"))
        if score is not None:
            items.append(item)
    if not items:
        return None
    return max(items, key=lambda item: item.get("primary_localization_score", -1.0))


def rank_dynamic_fields(field_summaries: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for field, summary in field_summaries.items():
        if summary.get("ignored_from_dynamic_ranking"):
            continue
        ratio = clean_float(summary.get("overall_severity_ratio"))
        worst = summary.get("worst_lead_by_rmse") or {}
        worst_ratio = severity_ratio(field, worst)
        score = max([x for x in (ratio, worst_ratio) if x is not None], default=None)
        if score is None:
            continue
        rows.append(
            {
                "field": field,
                "rank_score": score,
                "overall_rmse": summary["overall"].get("rmse"),
                "overall_bias": summary["overall"].get("bias"),
                "threshold_rmse_report_only": summary.get("threshold_rmse_report_only"),
                "overall_severity_ratio": ratio,
                "worst_lead_h": worst.get("lead_h"),
                "worst_lead_rmse": worst.get("rmse"),
                "worst_lead_severity_ratio": worst_ratio,
                "first_bad_lead": summary.get("first_bad_lead"),
            }
        )
    return sorted(rows, key=lambda item: item["rank_score"], reverse=True)


def rank_regions(
    splits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fields: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for group, bins in splits.items():
        for bin_name, by_field in bins.items():
            for field in fields:
                stat = by_field.get(field, {})
                ratio = severity_ratio(field, stat)
                if ratio is None:
                    continue
                rows.append(
                    {
                        "group": group,
                        "bin": bin_name,
                        "field": field,
                        "rmse": stat.get("rmse"),
                        "bias": stat.get("bias"),
                        "max_abs": stat.get("max_abs"),
                        "n": stat.get("n"),
                        "threshold_rmse_report_only": field_threshold(field),
                        "severity_ratio": ratio,
                    }
                )
    return sorted(rows, key=lambda item: item["severity_ratio"], reverse=True)


def fmt_num(x: Any, digits: int = 3) -> str:
    value = clean_float(x)
    if value is None:
        return "NA"
    if abs(value) >= 10000.0 or (0.0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    grid_comparator = load_json(args.grid_comparator_json)
    static_parity = load_json(args.static_parity_json)
    wind_mass = load_json(args.wind_mass_json)
    same_state_inventory = load_json(args.same_state_inventory_json)
    grid_fields = grid_comparator.get("field_summaries", {})

    init = parse_init_time(args.run_id)
    gm = wrfout_map(args.gpu_dir)
    cm = wrfout_map(args.cpu_dir)
    common = sorted(t for t in set(gm) & set(cm) if (t - init).total_seconds() > 0)
    if args.max_hour is not None:
        common = [t for t in common if (t - init).total_seconds() / 3600.0 <= args.max_hour]
    if not common:
        raise SystemExit("no common positive-lead wrfouts found")

    fields, radiation_present, missing_core = detect_fields(cm[common[0]], gm[common[0]])
    if not fields:
        raise SystemExit("no target fields are common to CPU and retained GPU wrfouts")

    with Dataset(cm[common[0]], "r") as first:
        hgt_info = read_var(first, "HGT")
        land_info = read_var(first, "LANDMASK")
        lat_info = read_var(first, "XLAT")
        lon_info = read_var(first, "XLONG")
    if not all([hgt_info, land_info, lat_info, lon_info]):
        raise SystemExit("missing one of HGT/LANDMASK/XLAT/XLONG in CPU truth")
    hgt = hgt_info[0]
    land = land_info[0]
    lat = lat_info[0]
    lon = lon_info[0]
    mask_bundle = make_masks(hgt, land, lat, lon)
    masks = mask_bundle["masks"]

    comparator_static = {
        field
        for field, summary in grid_fields.items()
        if isinstance(summary, dict)
        and (
            summary.get("classification") != "dynamic"
            or summary.get("known_static_field")
            or summary.get("observed_time_invariant")
        )
    }
    ignored_static_fields = set(STATIC_WRITER_FIELDS) | comparator_static

    overall: dict[str, StatAccumulator] = {field: StatAccumulator() for field in fields}
    h10_h14_overall: dict[str, StatAccumulator] = {field: StatAccumulator() for field in fields}
    h8_h14_overall: dict[str, StatAccumulator] = {field: StatAccumulator() for field in fields}
    per_level: dict[str, dict[int, StatAccumulator]] = defaultdict(dict)
    h10_h14_per_level: dict[str, dict[int, StatAccumulator]] = defaultdict(dict)
    splits: dict[str, dict[str, dict[str, StatAccumulator]]] = {
        group: {
            name: {field: StatAccumulator() for field in fields}
            for name in group_masks
        }
        for group, group_masks in masks.items()
    }
    h10_h14_splits: dict[str, dict[str, dict[str, StatAccumulator]]] = {
        group: {
            name: {field: StatAccumulator() for field in fields}
            for name in group_masks
        }
        for group, group_masks in masks.items()
    }

    corr_pooled = make_pair_accumulators(CORRELATION_NAMES)
    corr_h10_h14 = make_pair_accumulators(CORRELATION_NAMES)
    corr_h8_h14 = make_pair_accumulators(CORRELATION_NAMES)
    coupling_names = (
        "dV10_vs_dV_by_level",
        "dU10_vs_dU_by_level",
        "dPSFC_vs_dP_by_level",
        "dPSFC_vs_dPH_by_level",
        "dV10_vs_dP_by_level",
        "dV10_vs_dPH_by_level",
    )
    coupling: dict[str, list[CorrAccumulator]] = {name: [] for name in coupling_names}
    coupling_h10_h14: dict[str, list[CorrAccumulator]] = {name: [] for name in coupling_names}

    per_lead: dict[str, dict[str, Any]] = {}
    per_lead_correlations: dict[str, Any] = {}
    per_lead_colocation: dict[str, Any] = {}
    cell_candidates_by_lead: dict[str, list[dict[str, Any]]] = {}
    compatibility: dict[str, Any] = {
        "compared": {},
        "skipped": {},
        "mass_grid_note": (
            "Region, correlation, and selected-cell diagnostics destagger U/V horizontally "
            "and W/PH vertically onto the mass grid. Native-shape field stats preserve "
            "original WRF staggering."
        ),
    }

    for t in common:
        lead_h = int(round((t - init).total_seconds() / 3600.0))
        lead_key = str(lead_h)
        lead_stats: dict[str, Any] = {}
        native_diffs: dict[str, np.ndarray] = {}
        dims_by_field: dict[str, tuple[str, ...]] = {}

        with Dataset(gm[t], "r") as gds, Dataset(cm[t], "r") as cds:
            for field in fields:
                g = read_var(gds, field)
                c = read_var(cds, field)
                if g is None or c is None:
                    compatibility["skipped"].setdefault(field, []).append(
                        {"lead_h": lead_h, "reason": "missing in GPU or CPU file"}
                    )
                    continue
                garr, gdims, gunits = g
                carr, cdims, cunits = c
                if garr.shape != carr.shape or gdims != cdims:
                    compatibility["skipped"].setdefault(field, []).append(
                        {
                            "lead_h": lead_h,
                            "reason": "incompatible shape or dimensions",
                            "gpu_shape": list(garr.shape),
                            "cpu_shape": list(carr.shape),
                            "gpu_dims": list(gdims),
                            "cpu_dims": list(cdims),
                        }
                    )
                    continue
                diff = garr - carr
                native_diffs[field] = diff
                dims_by_field[field] = gdims
                lead_stats[field] = stats_array(diff)
                overall[field].update(diff)
                if lead_h in H10_H14:
                    h10_h14_overall[field].update(diff)
                if lead_h in H8_H14:
                    h8_h14_overall[field].update(diff)
                comp = compatibility["compared"].setdefault(
                    field,
                    {
                        "dims": list(gdims),
                        "shape": list(garr.shape),
                        "units_gpu": gunits,
                        "units_cpu": cunits,
                        "leads": [],
                    },
                )
                comp["leads"].append(lead_h)

                axis = level_axis(gdims)
                if axis is not None:
                    for k in range(diff.shape[axis]):
                        per_level[field].setdefault(k, StatAccumulator()).update(level_slice(diff, axis, k))
                        if lead_h in H10_H14:
                            h10_h14_per_level[field].setdefault(k, StatAccumulator()).update(level_slice(diff, axis, k))

        mass_diffs: dict[str, np.ndarray] = {}
        for field, diff in native_diffs.items():
            mg = as_mass_grid(field, diff)
            if mg is not None and mg.shape[-2:] == hgt.shape:
                mass_diffs[field] = mg

        for group, group_masks in masks.items():
            for name, mask in group_masks.items():
                for field, mg in mass_diffs.items():
                    splits[group][name][field].update(mask_values(mg, mask))
                    if lead_h in H10_H14:
                        h10_h14_splits[group][name][field].update(mask_values(mg, mask))

        low_values = mass_low_values(mass_diffs)
        if all(name in low_values for name in CORRELATION_NAMES):
            update_pair_accumulators(corr_pooled, low_values)
            if lead_h in H10_H14:
                update_pair_accumulators(corr_h10_h14, low_values)
            if lead_h in H8_H14:
                update_pair_accumulators(corr_h8_h14, low_values)
            per_lead_correlations[lead_key] = corr_matrix(low_values)
            per_lead_colocation[lead_key] = colocation_matrix(low_values)

        if all(field in mass_diffs for field in ("U", "V", "P", "PH")) and all(
            name in low_values for name in ("dU10", "dV10", "dPSFC")
        ):
            nlev = min(
                mass_diffs["U"].shape[0],
                mass_diffs["V"].shape[0],
                mass_diffs["P"].shape[0],
                mass_diffs["PH"].shape[0],
            )
            for name in coupling_names:
                while len(coupling[name]) < nlev:
                    coupling[name].append(CorrAccumulator())
                    coupling_h10_h14[name].append(CorrAccumulator())
            for k in range(nlev):
                pairs = {
                    "dV10_vs_dV_by_level": (low_values["dV10"], mass_diffs["V"][k]),
                    "dU10_vs_dU_by_level": (low_values["dU10"], mass_diffs["U"][k]),
                    "dPSFC_vs_dP_by_level": (low_values["dPSFC"], mass_diffs["P"][k]),
                    "dPSFC_vs_dPH_by_level": (low_values["dPSFC"], mass_diffs["PH"][k]),
                    "dV10_vs_dP_by_level": (low_values["dV10"], mass_diffs["P"][k]),
                    "dV10_vs_dPH_by_level": (low_values["dV10"], mass_diffs["PH"][k]),
                }
                for name, (x, y) in pairs.items():
                    coupling[name][k].update(x, y)
                    if lead_h in H10_H14:
                        coupling_h10_h14[name][k].update(x, y)

        cell_candidates_by_lead[lead_key] = top_cell_candidates(
            lead_h=lead_h,
            valid_time=t,
            low_values=low_values,
            native_diffs=native_diffs,
            hgt=hgt,
            land=land,
            lat=lat,
            lon=lon,
        )

        lead_stats["_files"] = {"gpu": str(gm[t]), "cpu": str(cm[t])}
        lead_stats["_valid_time_utc"] = t.isoformat()
        per_lead[lead_key] = lead_stats

    overall_final = {
        field: acc.finalize()
        for field, acc in overall.items()
        if acc.n > 0
    }
    h10_h14_overall_final = {
        field: acc.finalize()
        for field, acc in h10_h14_overall.items()
        if acc.n > 0
    }
    h8_h14_overall_final = {
        field: acc.finalize()
        for field, acc in h8_h14_overall.items()
        if acc.n > 0
    }
    per_level_final = {
        field: [
            {"k": int(k), "stats": acc.finalize()}
            for k, acc in sorted(levels.items())
        ]
        for field, levels in sorted(per_level.items())
    }
    h10_h14_per_level_final = {
        field: [
            {"k": int(k), "stats": acc.finalize()}
            for k, acc in sorted(levels.items())
        ]
        for field, levels in sorted(h10_h14_per_level.items())
    }
    splits_final = {
        group: {
            name: {
                field: acc.finalize()
                for field, acc in fields_by_name.items()
                if acc.n > 0
            }
            for name, fields_by_name in group_items.items()
        }
        for group, group_items in splits.items()
    }
    h10_h14_splits_final = {
        group: {
            name: {
                field: acc.finalize()
                for field, acc in fields_by_name.items()
                if acc.n > 0
            }
            for name, fields_by_name in group_items.items()
        }
        for group, group_items in h10_h14_splits.items()
    }
    coupling_final = {
        name: [{"k": k, **acc.finalize()} for k, acc in enumerate(accs)]
        for name, accs in coupling.items()
    }
    coupling_h10_h14_final = {
        name: [{"k": k, **acc.finalize()} for k, acc in enumerate(accs)]
        for name, accs in coupling_h10_h14.items()
    }

    field_summaries = build_field_summaries(
        fields=fields,
        overall=overall_final,
        per_lead=per_lead,
        grid_fields=grid_fields,
        ignored_static_fields=ignored_static_fields,
    )
    lead_analysis = build_lead_analysis(per_lead, fields)
    first_bad = first_materially_bad_lead(lead_analysis)
    worst_all = worst_lead_by_score(lead_analysis)
    worst_h10_h14 = worst_lead_by_score(lead_analysis, H10_H14)
    worst_h8_h14 = worst_lead_by_score(lead_analysis, H8_H14)
    selected_lead = worst_h10_h14 or worst_h8_h14 or worst_all or first_bad
    selected_lead_key = str(selected_lead["lead_h"]) if selected_lead else None
    selected_candidates = cell_candidates_by_lead.get(selected_lead_key or "", [])
    selected_cells = select_diverse_cells(selected_candidates)

    top_vertical_levels = {
        field: {
            "by_rmse": summarize_top_levels(levels, "rmse"),
            "by_max_abs": summarize_top_levels(levels, "max_abs"),
        }
        for field, levels in per_level_final.items()
    }
    recommended_vertical_levels = sorted(
        {
            0,
            1,
            *[
                int(item["k"])
                for field in ("U", "V", "P", "PH", "W")
                for item in top_vertical_levels.get(field, {}).get("by_rmse", [])[:3]
            ],
        }
    )
    dynamic_ranking = rank_dynamic_fields(field_summaries)
    region_ranking = rank_regions(splits_final, fields)

    selected_corr = per_lead_correlations.get(selected_lead_key or "", {})
    selected_colocation = per_lead_colocation.get(selected_lead_key or "", {})
    static_conclusion = static_parity.get("summary", {}).get("conclusion", {})
    prior_top_hypothesis = None
    hypotheses = wind_mass.get("ranked_root_cause_hypotheses", [])
    if hypotheses:
        prior_top_hypothesis = hypotheses[0].get("hypothesis")

    report: dict[str, Any] = {
        "schema": "v014-dynamic-field-attribution-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_used": False,
        "run_id": args.run_id,
        "inputs": {
            "gpu_dir": str(args.gpu_dir),
            "cpu_dir": str(args.cpu_dir),
            "domain": DOMAIN,
            "grid_comparator_json": str(args.grid_comparator_json),
            "static_parity_json": str(args.static_parity_json),
            "wind_mass_probe_json": str(args.wind_mass_json),
            "same_state_inventory_json": str(args.same_state_inventory_json),
        },
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "note": "This script imports numpy and netCDF4 only; no JAX/GPU execution is performed.",
        },
        "common_leads_h": [
            int(round((t - init).total_seconds() / 3600.0))
            for t in common
        ],
        "fields": {
            "core_requested": list(CORE_FIELDS),
            "compared": fields,
            "missing_core_requested": missing_core,
            "radiation_candidates": list(RADIATION_CANDIDATES),
            "radiation_present": radiation_present,
        },
        "report_only_thresholds": {
            "rmse": {field: REPORT_RMSE_THRESHOLDS[field] for field in fields if field in REPORT_RMSE_THRESHOLDS},
            "cell_score_scales": CELL_SCORE_SCALES,
            "cell_score_weights": CELL_SCORE_WEIGHTS,
            "materially_bad_lead_definition": "first lead where any compared target field RMSE meets or exceeds its predeclared report-only threshold",
        },
        "static_writer_exclusions": {
            "ignored_from_dynamic_ranking": sorted(field for field in ignored_static_fields if field in grid_fields or field in fields),
            "static_parity_conclusion": static_conclusion,
            "note": "Known static/time-invariant writer and base-state fields are not ranked as dynamic forecast errors.",
        },
        "mask_counts": mask_bundle["counts"],
        "compatibility": compatibility,
        "overall": overall_final,
        "per_lead": per_lead,
        "field_summaries": field_summaries,
        "lead_analysis": {
            "by_lead": lead_analysis,
            "first_materially_bad_lead": first_bad,
            "worst_lead_by_primary_score_all": worst_all,
            "worst_lead_by_primary_score_h8_h14": worst_h8_h14,
            "worst_lead_by_primary_score_h10_h14": worst_h10_h14,
            "selected_localization_lead": selected_lead,
        },
        "dynamic_field_ranking": dynamic_ranking,
        "per_vertical_level": per_level_final,
        "top_vertical_levels": top_vertical_levels,
        "splits": splits_final,
        "region_ranking": region_ranking,
        "correlations": {
            "pooled_low_level_mass_grid": finalize_pair_accumulators(corr_pooled),
            "h8_h14_low_level_mass_grid": finalize_pair_accumulators(corr_h8_h14),
            "h10_h14_low_level_mass_grid": finalize_pair_accumulators(corr_h10_h14),
            "selected_lead_low_level_mass_grid": selected_corr,
            "per_lead_low_level_mass_grid": per_lead_correlations,
        },
        "colocation": {
            "definition": "For each lead and variable, top 1 percent by absolute mass-grid error is compared by Jaccard and overlap.",
            "selected_lead_top1pct": selected_colocation,
            "per_lead_top1pct": per_lead_colocation,
        },
        "surface_vs_aloft_coupling": {
            "pooled_by_level": coupling_final,
            "h10_h14_by_level": coupling_h10_h14_final,
        },
        "lead_windows": {
            "h8_h14": {
                "lead_hours": list(H8_H14),
                "overall": h8_h14_overall_final,
            },
            "h10_h14": {
                "lead_hours": list(H10_H14),
                "overall": h10_h14_overall_final,
                "per_lead": {
                    str(h): per_lead[str(h)]
                    for h in H10_H14
                    if str(h) in per_lead
                },
                "per_vertical_level": h10_h14_per_level_final,
                "splits": h10_h14_splits_final,
                "correlations_low_level_mass_grid": finalize_pair_accumulators(corr_h10_h14),
                "surface_vs_aloft_coupling_by_level": coupling_h10_h14_final,
            },
        },
        "localization_manifest": {
            "selected_lead_h": selected_lead["lead_h"] if selected_lead else None,
            "selected_valid_time_utc": per_lead.get(selected_lead_key or "", {}).get("_valid_time_utc"),
            "candidate_source": "highest primary localization score in h10-h14; fallback h8-h14/all leads if unavailable",
            "selected_cell_count": len(selected_cells),
            "target_cell_count": SELECTED_CELL_COUNT,
            "candidate_filters": {
                "preferred": f"outside {LOCALIZATION_INTERIOR_CELLS}-cell frame and LANDMASK == 0 or HGT < 300 m",
                "fallback": f"outside {LOCALIZATION_INTERIOR_CELLS}-cell frame, then any finite mass-grid cell",
                "patch_halo_radius_cells": PATCH_RADIUS,
            },
            "recommended_vertical_levels_for_first_probe": recommended_vertical_levels,
            "selected_cells": selected_cells,
            "top_candidates_for_selected_lead": selected_candidates[:32],
            "candidate_counts_by_lead": {
                lead_key: len(cands)
                for lead_key, cands in sorted(cell_candidates_by_lead.items(), key=lambda item: int(item[0]))
            },
        },
        "prior_evidence": {
            "wind_mass_probe_top_hypothesis": prior_top_hypothesis,
            "same_state_inventory_preferred_first_lead": same_state_inventory.get("case", {}).get("preferred_first_lead"),
            "same_state_inventory_truth_requirement": same_state_inventory.get("scope", {}).get("truth_requirement"),
        },
        "next_target": {
            "recommendation": "Build CPU-WRF term savepoints for the selected lead/cells before relying on a JAX-only operator probe.",
            "first_terms_to_measure": [
                "stage_input_parity",
                "mass_coupling",
                "momentum_advection",
                "large_step_horizontal_pgf",
                "coriolis",
                "source_tendency_folding",
                "acoustic_uv",
                "mu_theta",
                "w_ph_vertical",
                "boundary_spec_relax",
            ],
            "rationale": (
                "Surface winds are nearly co-located with low-level prognostic U/V error, while PSFC/MU/P/PH "
                "are already materially bad. Existing proof inventory says JAX-only diagnostics are not WRF "
                "same-state truth."
            ),
        },
        "limits": [
            "This is a wrfout-only attribution and makes no pass/fail equivalence claim.",
            "Hourly wrfouts cannot determine whether wind drives mass/geopotential error or the reverse.",
            "Static/base-state writer mismatches are excluded from dynamic ranking but remain a separate validation gate.",
        ],
    }
    return report


def md_table_top_fields(ranking: list[dict[str, Any]], limit: int = 8) -> list[str]:
    lines = [
        "| Field | score | overall RMSE | bias | worst lead | worst RMSE | first bad |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranking[:limit]:
        first_bad = row.get("first_bad_lead") or {}
        lines.append(
            f"| `{row['field']}` | {fmt_num(row.get('rank_score'))} | "
            f"{fmt_num(row.get('overall_rmse'))} | {fmt_num(row.get('overall_bias'))} | "
            f"{row.get('worst_lead_h', 'NA')} | {fmt_num(row.get('worst_lead_rmse'))} | "
            f"{first_bad.get('lead_h', 'NA')} |"
        )
    return lines


def md_selected_cells(cells: list[dict[str, Any]], limit: int = 12) -> list[str]:
    lines = [
        "| Rank | y | x | lat | lon | score | hits | top components |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for cell in cells[:limit]:
        comps = ", ".join(item["name"] for item in cell.get("top_components", [])[:4])
        idx = cell["mass_index"]
        lines.append(
            f"| {cell['selection_rank']} | {idx['south_north']} | {idx['west_east']} | "
            f"{fmt_num(cell.get('lat'), 4)} | {fmt_num(cell.get('lon'), 4)} | "
            f"{fmt_num(cell.get('composite_score'))} | {cell.get('co_located_component_hit_count')} | `{comps}` |"
        )
    return lines


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lead_info = report["lead_analysis"]
    first_bad = lead_info.get("first_materially_bad_lead") or {}
    selected = lead_info.get("selected_localization_lead") or {}
    worst_h10 = lead_info.get("worst_lead_by_primary_score_h10_h14") or {}
    manifest = report["localization_manifest"]
    cells = manifest.get("selected_cells", [])
    corr = report["correlations"].get("selected_lead_low_level_mass_grid", {})
    top_fields = report["dynamic_field_ranking"]
    regions = report["region_ranking"]
    lines: list[str] = [
        "# V0.14 Dynamic Field Attribution",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "CPU-only wrfout attribution for retained Case 3. This is not an equivalence claim and does not run the model.",
        "",
        "## Verdict",
        "",
        f"- first materially bad lead: `h{first_bad.get('lead_h', 'NA')}` under report-only thresholds",
        f"- selected same-state localization lead: `h{manifest.get('selected_lead_h', 'NA')}` (`{manifest.get('selected_valid_time_utc', 'NA')}`)",
        f"- worst h10-h14 primary lead: `h{worst_h10.get('lead_h', 'NA')}` score `{fmt_num(worst_h10.get('primary_localization_score'))}`",
        f"- selected cells: `{manifest.get('selected_cell_count')}` mass-grid cells with U/V/W/PH native-stagger context",
        "- next target: CPU-WRF term savepoints for selected lead/cells, then JAX same-state term comparison",
        "",
        "## Top Dynamic Fields",
        "",
    ]
    lines.extend(md_table_top_fields(top_fields))
    lines.extend(
        [
            "",
            "## Selected Cells",
            "",
            f"Lead `h{manifest.get('selected_lead_h', 'NA')}`; full selected-cell details and staggered face indices are in JSON.",
            "",
        ]
    )
    lines.extend(md_selected_cells(cells))
    lines.extend(
        [
            "",
            "## Correlation Snapshot",
            "",
            "| Pair | selected-lead r |",
            "| --- | ---: |",
        ]
    )
    for pair in (
        "dU10__dU_k0",
        "dV10__dV_k0",
        "dPSFC__dMU",
        "dPSFC__dP_k0",
        "dPSFC__dPH_k0",
        "dV10__dP_k0",
        "dV10__dPH_k0",
    ):
        lines.append(f"| `{pair}` | {fmt_num(corr.get(pair, {}).get('pearson'))} |")
    lines.extend(["", "## Region Signal", ""])
    for item in regions[:6]:
        lines.append(
            f"- `{item['field']}` in `{item['group']}/{item['bin']}`: RMSE `{fmt_num(item.get('rmse'))}`, "
            f"severity `{fmt_num(item.get('severity_ratio'))}`"
        )
    top_levels = report["top_vertical_levels"]
    lines.extend(["", "## Vertical Targets", ""])
    for field in ("U", "V", "P", "PH", "W"):
        levels = top_levels.get(field, {}).get("by_rmse", [])[:3]
        text = ", ".join(f"k{item['k']} RMSE {fmt_num(item['stats'].get('rmse'))}" for item in levels)
        lines.append(f"- `{field}`: {text}")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Static/time-invariant writer and base-state fields are excluded from dynamic ranking.",
            "- Wrfout-only evidence cannot identify the first failing tendency term.",
            "- Detailed per-lead, per-level, per-region, colocation, and cell tables are in JSON.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--gpu-dir", type=Path, default=GPU_DIR)
    ap.add_argument("--cpu-dir", type=Path, default=CPU_DIR)
    ap.add_argument("--grid-comparator-json", type=Path, default=GRID_COMPARATOR_JSON)
    ap.add_argument("--static-parity-json", type=Path, default=STATIC_PARITY_JSON)
    ap.add_argument("--wind-mass-json", type=Path, default=WIND_MASS_JSON)
    ap.add_argument("--same-state-inventory-json", type=Path, default=SAME_STATE_INVENTORY_JSON)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--max-hour", type=int, default=24)
    args = ap.parse_args(argv)

    report = build_report(args)
    write_json(args.out_json, report)
    write_markdown(report, args.out_md)
    selected_cells = report["localization_manifest"]["selected_cells"]
    summary = {
        "schema": report["schema"],
        "run_id": report["run_id"],
        "common_leads_h": report["common_leads_h"],
        "first_materially_bad_lead": report["lead_analysis"]["first_materially_bad_lead"],
        "selected_localization_lead": report["lead_analysis"]["selected_localization_lead"],
        "selected_cell_count": len(selected_cells),
        "selected_cells_preview": selected_cells[:5],
        "top_dynamic_fields": report["dynamic_field_ranking"][:5],
        "next_target": report["next_target"]["recommendation"],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_jsonable, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
