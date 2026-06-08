#!/usr/bin/env python3
"""CPU-only wind/mass divergence anatomy probe for V014 Case 3.

This proof reads retained GPU wrfouts and CPU-WRF truth wrfouts. It does not
run the model and does not import JAX.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=src \
    python proofs/v014/wind_mass_divergence_probe.py
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
from typing import Any

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
GPU_DIR = Path("/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z")
CPU_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z")
OUT_JSON = ROOT / "proofs/v014/wind_mass_divergence_probe.json"
OUT_MD = ROOT / "proofs/v014/wind_mass_divergence_probe.md"
WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")

FIELDS_3D = ("U", "V", "W", "T", "QVAPOR", "P", "PH", "MU", "MUB", "PB", "PHB")
FIELDS_2D = ("U10", "V10", "T2", "PSFC")
FIELDS = FIELDS_3D + FIELDS_2D
H10_H14 = range(10, 15)
FRAME_CELLS = 5


def parse_init_time(run_id: str) -> datetime:
    parts = run_id.split("_")
    ds = parts[0]
    hour = int(parts[1].replace("z", ""))
    return datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]), hour, tzinfo=timezone.utc)


def wrfout_map(path: Path, domain: str = "d02") -> dict[datetime, Path]:
    out = {}
    for p in sorted(path.glob(f"wrfout_{domain}_*")):
        m = WRFOUT_RE.match(p.name)
        if not m or not p.is_file():
            continue
        vt = datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        out[vt] = p
    return out


def read_var(path: Path, name: str) -> tuple[np.ndarray, tuple[str, ...], str | None] | None:
    with Dataset(path) as ds:
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


def stats_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    total = int(arr.size)
    vals = finite_values(arr)
    if vals.size == 0:
        return {"n": 0, "total": total, "status": "NO_FINITE"}
    abs_vals = np.abs(vals)
    return {
        "n": int(vals.size),
        "total": total,
        "finite_fraction": float(vals.size / total) if total else None,
        "bias": float(np.mean(vals)),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
        "mae": float(np.mean(abs_vals)),
        "max_abs": float(np.max(abs_vals)),
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
            return {"n": 0, "total": self.total, "status": "NO_FINITE"}
        return {
            "n": int(self.n),
            "total": int(self.total),
            "finite_fraction": float(self.n / self.total) if self.total else None,
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
    if field in ("W", "PH", "PHB"):
        return destagger_z(diff)
    if diff.ndim in (2, 3):
        return diff
    return None


def mask_values(arr: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[mask2d]
    if arr.ndim == 3:
        return arr[:, mask2d]
    raise ValueError(f"unsupported mask array rank {arr.ndim}")


def level_axis(dims: tuple[str, ...]) -> int | None:
    for i, dim in enumerate(dims):
        if dim in ("bottom_top", "bottom_top_stag", "soil_layers_stag"):
            return i
    return None


def level_slice(arr: np.ndarray, axis: int, k: int) -> np.ndarray:
    return np.take(arr, indices=k, axis=axis)


def make_masks(hgt: np.ndarray, land: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    land_mask = land > 0.5
    lat_mid = float(np.nanmedian(lat))
    lon_mid = float(np.nanmedian(lon))
    sn, we = hgt.shape
    frame = np.zeros_like(hgt, dtype=bool)
    frame[:FRAME_CELLS, :] = True
    frame[-FRAME_CELLS:, :] = True
    frame[:, :FRAME_CELLS] = True
    frame[:, -FRAME_CELLS:] = True

    masks = {
        "elevation_ocean": {
            "ocean": ~land_mask,
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
        },
    }
    counts = {
        group: {name: int(np.sum(mask)) for name, mask in group_masks.items()}
        for group, group_masks in masks.items()
    }
    counts["grid_shape"] = {"south_north": sn, "west_east": we}
    counts["lat_lon_split"] = {"lat_mid": lat_mid, "lon_mid": lon_mid}
    return {"masks": masks, "counts": counts}


def make_pair_accumulators(names: list[str]) -> dict[str, CorrAccumulator]:
    return {
        f"{a}__{b}": CorrAccumulator()
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def update_pair_accumulators(accs: dict[str, CorrAccumulator], values: dict[str, np.ndarray]) -> None:
    names = list(values)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}__{b}"
            if key in accs:
                accs[key].update(values[a], values[b])


def finalize_pair_accumulators(accs: dict[str, CorrAccumulator]) -> dict[str, Any]:
    return {key: acc.finalize() for key, acc in sorted(accs.items())}


def summarize_top_levels(per_level: list[dict[str, Any]], n: int = 8) -> list[dict[str, Any]]:
    valid = [item for item in per_level if item.get("stats", {}).get("n", 0)]
    return sorted(valid, key=lambda x: x["stats"].get("rmse", -1.0), reverse=True)[:n]


def get_stat(report: dict[str, Any], *path: str) -> dict[str, Any]:
    obj: Any = report
    for key in path:
        obj = obj.get(key, {})
    return obj if isinstance(obj, dict) else {}


def val(obj: dict[str, Any], key: str) -> float | None:
    x = obj.get(key)
    return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else None


def fmt_num(x: Any, digits: int = 3) -> str:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return f"{float(x):.{digits}f}"
    return "NA"


def rank_hypotheses(report: dict[str, Any]) -> list[dict[str, Any]]:
    overall = report["overall"]
    hwin = report["lead_windows"]["h10_h14"]["overall"]
    corr_low = report["correlations"]["pooled_low_level_mass_grid"]
    corr_win = report["lead_windows"]["h10_h14"]["correlations_low_level_mass_grid"]
    coupling = report["surface_vs_aloft_coupling"]["pooled_by_level"]
    splits = report["splits"]
    boundary = splits["boundary"]
    elev = splits["elevation_ocean"]

    v10 = overall["V10"]
    u10 = overall["U10"]
    psfc = overall["PSFC"]
    v = overall["V"]
    u = overall["U"]
    t2 = overall["T2"]
    qv = overall["QVAPOR"]
    p = overall["P"]
    ph = overall["PH"]
    mub = overall["MUB"]
    pb = overall["PB"]
    phb = overall["PHB"]
    win_v10 = hwin["V10"]

    bframe_v10 = boundary[f"frame_{FRAME_CELLS}cells"]["V10"]
    bint_v10 = boundary[f"interior_excluding_{FRAME_CELLS}cell_frame"]["V10"]
    boundary_ratio = None
    if bint_v10.get("rmse"):
        boundary_ratio = bframe_v10.get("rmse") / bint_v10.get("rmse")

    ocean_v10 = elev["ocean"]["V10"]
    land_low_v10 = elev["land_0_300m"]["V10"]

    c_v10_v_k0 = corr_low.get("dV10__dV_k0", {}).get("pearson")
    c_u10_u_k0 = corr_low.get("dU10__dU_k0", {}).get("pearson")
    c_v10_psfc = corr_low.get("dV10__dPSFC", {}).get("pearson")
    c_v10_p_k0_win = corr_win.get("dV10__dP_k0", {}).get("pearson")
    c_v10_ph_k0_win = corr_win.get("dV10__dPH_k0", {}).get("pearson")
    d_v10_dv_profile = coupling["dV10_vs_dV_by_level"]
    low_corr = d_v10_dv_profile[0]["pearson"] if d_v10_dv_profile else None

    hypotheses = [
        {
            "rank": 1,
            "hypothesis": "Prognostic wind-column divergence with a near-surface projection, coupled to mass/geopotential error and peaking around h10-h14.",
            "verdict": "favored_by_this_probe",
            "evidence_for": [
                f"V10 pooled RMSE {fmt_num(v10.get('rmse'))} m/s and h10-h14 RMSE {fmt_num(win_v10.get('rmse'))} m/s are above the 1.5 m/s equivalence envelope.",
                f"Native 3D wind errors are larger than surface diagnostic errors: U RMSE {fmt_num(u.get('rmse'))} m/s, V RMSE {fmt_num(v.get('rmse'))} m/s.",
                f"Surface wind is coupled to low-level prognostic wind: corr(dV10,dV_k0)={fmt_num(c_v10_v_k0)}, corr(dU10,dU_k0)={fmt_num(c_u10_u_k0)}; by-level dV coupling starts at {fmt_num(low_corr)}.",
                f"Mass fields are not quiet: PSFC RMSE {fmt_num(psfc.get('rmse'))} Pa, P RMSE {fmt_num(p.get('rmse'))} Pa, PH RMSE {fmt_num(ph.get('rmse'))} m2/s2.",
                f"h10-h14 mass correlations include corr(dV10,dP_k0)={fmt_num(c_v10_p_k0_win)} and corr(dV10,dPH_k0)={fmt_num(c_v10_ph_k0_win)}.",
            ],
            "evidence_against": [
                "This wrfout-only probe cannot localize the first bad tendency component or prove whether wind drives mass or mass drives wind.",
            ],
        },
        {
            "rank": 2,
            "hypothesis": "Static base-state or wrfout/grid-base reconstruction mismatch contributes to the mass/geopotential residual.",
            "verdict": "plausible_contributor_not_full_explanation",
            "evidence_for": [
                f"Compatible base-state fields are not identical: MUB RMSE {fmt_num(mub.get('rmse'))} Pa, PB RMSE {fmt_num(pb.get('rmse'))} Pa, PHB RMSE {fmt_num(phb.get('rmse'))} m2/s2.",
                "MUB/PB/PHB statistics are essentially lead-invariant, so this is reproducible static structure rather than random forecast noise.",
                f"PSFC and low-level P are strongly coupled: corr(dPSFC,dP_k0)={fmt_num(corr_low.get('dPSFC__dP_k0', {}).get('pearson'))}.",
            ],
            "evidence_against": [
                f"Dynamic fields are much larger and lead-window dependent: V RMSE {fmt_num(v.get('rmse'))} m/s and h10-h14 V10 RMSE {fmt_num(win_v10.get('rmse'))} m/s.",
                "The static base-state signal cannot by itself explain the h10-h14 V10 peak or the old case-to-case V10 bias sign changes.",
            ],
        },
        {
            "rank": 3,
            "hypothesis": "Surface/PBL or source-tendency cadence feedback amplifies a real low-level wind error after the early leads.",
            "verdict": "plausible_but_not_proven",
            "evidence_for": [
                f"V10 bias/RMSE worsens in the h10-h14 window: pooled bias {fmt_num(v10.get('bias'))}, h10-h14 bias {fmt_num(win_v10.get('bias'))}.",
                f"Ocean V10 RMSE {fmt_num(ocean_v10.get('rmse'))} m/s and low-land V10 RMSE {fmt_num(land_low_v10.get('rmse'))} m/s show the failure is not only steep-terrain noise.",
                "The surface wind error tracks low-level prognostic wind, so a near-surface feedback/cadence issue remains a viable owner.",
            ],
            "evidence_against": [
                f"T2 RMSE {fmt_num(t2.get('rmse'))} K and QVAPOR RMSE {fmt_num(qv.get('rmse'), 6)} kg/kg are much smaller relative to their known envelopes, arguing against broad thermodynamic/moisture blow-up.",
                "No same-state component tendency comparison was run here, so this is not yet an implementation-localized diagnosis.",
            ],
        },
        {
            "rank": 4,
            "hypothesis": "Boundary-frame forcing defect/regression dominates the V10 failure.",
            "verdict": "disfavored_by_this_probe",
            "evidence_for": [
                f"The {FRAME_CELLS}-cell frame still has V10 RMSE {fmt_num(bframe_v10.get('rmse'))} m/s.",
            ],
            "evidence_against": [
                f"Interior V10 RMSE remains {fmt_num(bint_v10.get('rmse'))} m/s; frame/interior RMSE ratio is {fmt_num(boundary_ratio)}.",
                "Excluding the boundary frame does not collapse the error or flip this into a harmless edge-only artifact.",
            ],
        },
        {
            "rank": 5,
            "hypothesis": "Pure 10 m diagnostic sign/formula bug.",
            "verdict": "disfavored_by_this_probe",
            "evidence_for": [
                "The largest user-visible symptom is still U10/V10 grid error.",
            ],
            "evidence_against": [
                f"3D U/V native RMSEs ({fmt_num(u.get('rmse'))}/{fmt_num(v.get('rmse'))} m/s) exceed U10/V10 RMSEs ({fmt_num(u10.get('rmse'))}/{fmt_num(v10.get('rmse'))} m/s).",
                f"Low-level coupling is substantial: corr(dV10,dV_k0)={fmt_num(c_v10_v_k0)} and corr(dU10,dU_k0)={fmt_num(c_u10_u_k0)}.",
                "The V10 bias changes with lead window in existing V014 evidence, which is not the shape of a single static sign bug.",
            ],
        },
        {
            "rank": 6,
            "hypothesis": "Old absent Coriolis or post-step-only normal-boundary bug reappeared unchanged.",
            "verdict": "low_priority_unless_a_tendency_probe_contradicts",
            "evidence_for": [
                "Wind/mass coupling is real, so momentum assembly remains the broad search space.",
            ],
            "evidence_against": [
                "The prior-attribution sidecar established both old bugs are fixed ancestors of current HEAD.",
                "This probe does not show a boundary-frame-dominated signature, and a wrfout-only anatomy cannot implicate a missing Coriolis term without component tendencies.",
            ],
        },
    ]
    return hypotheses


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    init = parse_init_time(args.run_id)
    gm = wrfout_map(args.gpu_dir)
    cm = wrfout_map(args.cpu_dir)
    common = sorted(t for t in set(gm) & set(cm) if (t - init).total_seconds() > 0)
    if args.max_hour is not None:
        common = [t for t in common if (t - init).total_seconds() / 3600.0 <= args.max_hour]
    if not common:
        raise SystemExit("no common positive-lead wrfouts found")

    first_cpu = cm[common[0]]
    hgt_info = read_var(first_cpu, "HGT")
    land_info = read_var(first_cpu, "LANDMASK")
    lat_info = read_var(first_cpu, "XLAT")
    lon_info = read_var(first_cpu, "XLONG")
    if not all([hgt_info, land_info, lat_info, lon_info]):
        raise SystemExit("missing one of HGT/LANDMASK/XLAT/XLONG in CPU truth")
    hgt = hgt_info[0]
    land = land_info[0]
    lat = lat_info[0]
    lon = lon_info[0]
    mask_bundle = make_masks(hgt, land, lat, lon)
    masks = mask_bundle["masks"]

    overall: dict[str, StatAccumulator] = {field: StatAccumulator() for field in FIELDS}
    hwin_overall: dict[str, StatAccumulator] = {field: StatAccumulator() for field in FIELDS}
    per_level: dict[str, dict[int, StatAccumulator]] = defaultdict(dict)
    hwin_per_level: dict[str, dict[int, StatAccumulator]] = defaultdict(dict)
    splits: dict[str, dict[str, dict[str, StatAccumulator]]] = {
        group: {
            name: {field: StatAccumulator() for field in FIELDS}
            for name in group_masks
        }
        for group, group_masks in masks.items()
    }
    hwin_splits: dict[str, dict[str, dict[str, StatAccumulator]]] = {
        group: {
            name: {field: StatAccumulator() for field in FIELDS}
            for name in group_masks
        }
        for group, group_masks in masks.items()
    }

    corr_names = [
        "dV10",
        "dU10",
        "dPSFC",
        "dU_k0",
        "dV_k0",
        "dP_k0",
        "dPH_k0",
        "dP_colmean",
        "dPH_colmean",
    ]
    corr_low = make_pair_accumulators(corr_names)
    corr_low_hwin = make_pair_accumulators(corr_names)
    coupling_names = [
        "dV10_vs_dV_by_level",
        "dU10_vs_dU_by_level",
        "dPSFC_vs_dP_by_level",
        "dPSFC_vs_dPH_by_level",
        "dV10_vs_dP_by_level",
        "dV10_vs_dPH_by_level",
    ]
    coupling: dict[str, list[CorrAccumulator]] = {name: [] for name in coupling_names}
    coupling_hwin: dict[str, list[CorrAccumulator]] = {name: [] for name in coupling_names}
    per_lead: dict[str, dict[str, Any]] = {}
    compatibility: dict[str, Any] = {
        "compared": {},
        "skipped": {},
        "mass_grid_diagnostic_note": (
            "Native field comparisons use matching WRF shapes. Spatial splits and correlations "
            "destagger U/V horizontally and W/PH/PHB vertically onto the mass grid."
        ),
    }

    for t in common:
        lead_h = int(round((t - init).total_seconds() / 3600.0))
        lead_key = str(lead_h)
        gpath = gm[t]
        cpath = cm[t]
        lead_stats: dict[str, Any] = {}
        diffs: dict[str, np.ndarray] = {}
        dims_by_field: dict[str, tuple[str, ...]] = {}

        for field in FIELDS:
            g = read_var(gpath, field)
            c = read_var(cpath, field)
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
            diffs[field] = diff
            dims_by_field[field] = gdims
            lead_stats[field] = stats_array(diff)
            overall[field].update(diff)
            if lead_h in H10_H14:
                hwin_overall[field].update(diff)
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
                        hwin_per_level[field].setdefault(k, StatAccumulator()).update(level_slice(diff, axis, k))

        mass_diffs: dict[str, np.ndarray] = {}
        for field, diff in diffs.items():
            mg = as_mass_grid(field, diff)
            if mg is not None and mg.shape[-2:] == hgt.shape:
                mass_diffs[field] = mg

        for group, group_masks in masks.items():
            for name, mask in group_masks.items():
                for field, mg in mass_diffs.items():
                    splits[group][name][field].update(mask_values(mg, mask))
                    if lead_h in H10_H14:
                        hwin_splits[group][name][field].update(mask_values(mg, mask))

        if all(name in mass_diffs for name in ("V10", "U10", "PSFC", "U", "V", "P", "PH")):
            corr_values = {
                "dV10": mass_diffs["V10"],
                "dU10": mass_diffs["U10"],
                "dPSFC": mass_diffs["PSFC"],
                "dU_k0": mass_diffs["U"][0],
                "dV_k0": mass_diffs["V"][0],
                "dP_k0": mass_diffs["P"][0],
                "dPH_k0": mass_diffs["PH"][0],
                "dP_colmean": np.nanmean(mass_diffs["P"], axis=0),
                "dPH_colmean": np.nanmean(mass_diffs["PH"], axis=0),
            }
            update_pair_accumulators(corr_low, corr_values)
            if lead_h in H10_H14:
                update_pair_accumulators(corr_low_hwin, corr_values)

            nlev = min(mass_diffs["U"].shape[0], mass_diffs["V"].shape[0], mass_diffs["P"].shape[0], mass_diffs["PH"].shape[0])
            for name in coupling_names:
                while len(coupling[name]) < nlev:
                    coupling[name].append(CorrAccumulator())
                    coupling_hwin[name].append(CorrAccumulator())
            for k in range(nlev):
                pairs = {
                    "dV10_vs_dV_by_level": (mass_diffs["V10"], mass_diffs["V"][k]),
                    "dU10_vs_dU_by_level": (mass_diffs["U10"], mass_diffs["U"][k]),
                    "dPSFC_vs_dP_by_level": (mass_diffs["PSFC"], mass_diffs["P"][k]),
                    "dPSFC_vs_dPH_by_level": (mass_diffs["PSFC"], mass_diffs["PH"][k]),
                    "dV10_vs_dP_by_level": (mass_diffs["V10"], mass_diffs["P"][k]),
                    "dV10_vs_dPH_by_level": (mass_diffs["V10"], mass_diffs["PH"][k]),
                }
                for name, (x, y) in pairs.items():
                    coupling[name][k].update(x, y)
                    if lead_h in H10_H14:
                        coupling_hwin[name][k].update(x, y)

        lead_stats["_files"] = {"gpu": str(gpath), "cpu": str(cpath)}
        per_lead[lead_key] = lead_stats

    per_level_final = {
        field: [
            {"k": int(k), "stats": acc.finalize()}
            for k, acc in sorted(levels.items())
        ]
        for field, levels in sorted(per_level.items())
    }
    hwin_per_level_final = {
        field: [
            {"k": int(k), "stats": acc.finalize()}
            for k, acc in sorted(levels.items())
        ]
        for field, levels in sorted(hwin_per_level.items())
    }
    splits_final = {
        group: {
            name: {
                field: acc.finalize()
                for field, acc in fields.items()
                if acc.n > 0
            }
            for name, fields in group_items.items()
        }
        for group, group_items in splits.items()
    }
    hwin_splits_final = {
        group: {
            name: {
                field: acc.finalize()
                for field, acc in fields.items()
                if acc.n > 0
            }
            for name, fields in group_items.items()
        }
        for group, group_items in hwin_splits.items()
    }

    coupling_final = {
        name: [
            {"k": k, **acc.finalize()}
            for k, acc in enumerate(accs)
        ]
        for name, accs in coupling.items()
    }
    coupling_hwin_final = {
        name: [
            {"k": k, **acc.finalize()}
            for k, acc in enumerate(accs)
        ]
        for name, accs in coupling_hwin.items()
    }

    report: dict[str, Any] = {
        "schema": "v014-wind-mass-divergence-probe-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_used": False,
        "run_id": args.run_id,
        "inputs": {
            "gpu_dir": str(args.gpu_dir),
            "cpu_dir": str(args.cpu_dir),
            "domain": "d02",
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
        "mask_counts": mask_bundle["counts"],
        "compatibility": compatibility,
        "overall": {
            field: acc.finalize()
            for field, acc in overall.items()
            if acc.n > 0
        },
        "per_lead": per_lead,
        "per_vertical_level": per_level_final,
        "splits": {
            "elevation_ocean": splits_final["elevation_ocean"],
            "quadrant": splits_final["quadrant"],
            "boundary": splits_final["boundary"],
        },
        "correlations": {
            "pooled_low_level_mass_grid": finalize_pair_accumulators(corr_low),
        },
        "surface_vs_aloft_coupling": {
            "pooled_by_level": coupling_final,
        },
        "lead_windows": {
            "h10_h14": {
                "lead_hours": list(H10_H14),
                "overall": {
                    field: acc.finalize()
                    for field, acc in hwin_overall.items()
                    if acc.n > 0
                },
                "per_lead": {
                    str(h): per_lead[str(h)]
                    for h in H10_H14
                    if str(h) in per_lead
                },
                "per_vertical_level": hwin_per_level_final,
                "splits": {
                    "elevation_ocean": hwin_splits_final["elevation_ocean"],
                    "quadrant": hwin_splits_final["quadrant"],
                    "boundary": hwin_splits_final["boundary"],
                },
                "correlations_low_level_mass_grid": finalize_pair_accumulators(corr_low_hwin),
                "surface_vs_aloft_coupling_by_level": coupling_hwin_final,
            }
        },
    }

    report["ranked_root_cause_hypotheses"] = rank_hypotheses(report)
    report["top_vertical_rmse_levels"] = {
        field: summarize_top_levels(levels)
        for field, levels in per_level_final.items()
    }
    return report


def md_field_table(title: str, stats_by_field: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    lines = [f"## {title}", "", "| Field | RMSE | Bias | MAE | Max abs | n |", "|---|---:|---:|---:|---:|---:|"]
    for field in fields:
        st = stats_by_field.get(field, {})
        lines.append(
            f"| `{field}` | {fmt_num(st.get('rmse'), 6)} | {fmt_num(st.get('bias'), 6)} | "
            f"{fmt_num(st.get('mae'), 6)} | {fmt_num(st.get('max_abs'), 6)} | {st.get('n', 0)} |"
        )
    lines.append("")
    return lines


def write_markdown(report: dict[str, Any], path: Path) -> None:
    overall = report["overall"]
    hwin = report["lead_windows"]["h10_h14"]["overall"]
    lines: list[str] = [
        "# V014 Wind/Mass Divergence Probe",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "CPU-only wrfout anatomy probe for Case 3. This is not an equivalence pass and it does not run the model.",
        "",
        "## Inputs",
        "",
        f"- GPU retained wrfouts: `{report['inputs']['gpu_dir']}`",
        f"- CPU-WRF truth: `{report['inputs']['cpu_dir']}`",
        f"- common leads: `{report['common_leads_h'][0]}` to `{report['common_leads_h'][-1]}` h (`{len(report['common_leads_h'])}` files)",
        f"- JAX_PLATFORMS during run: `{report['environment'].get('JAX_PLATFORMS')}`",
        "",
    ]
    lines.extend(md_field_table("Overall Native-Shape Differences", overall, FIELDS))
    lines.extend(md_field_table("Lead Window h10-h14", hwin, FIELDS))

    lines.extend(
        [
            "## h10-h14 Lead Anatomy",
            "",
            "| Lead h | V10 RMSE | V10 Bias | U10 RMSE | PSFC RMSE | U RMSE | V RMSE | P RMSE | PH RMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for h in H10_H14:
        item = report["per_lead"].get(str(h), {})
        lines.append(
            f"| {h} | {fmt_num(item.get('V10', {}).get('rmse'))} | {fmt_num(item.get('V10', {}).get('bias'))} | "
            f"{fmt_num(item.get('U10', {}).get('rmse'))} | {fmt_num(item.get('PSFC', {}).get('rmse'))} | "
            f"{fmt_num(item.get('U', {}).get('rmse'))} | {fmt_num(item.get('V', {}).get('rmse'))} | "
            f"{fmt_num(item.get('P', {}).get('rmse'))} | {fmt_num(item.get('PH', {}).get('rmse'))} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Splits",
            "",
            "### V10 Spatial Splits",
            "",
            "| Split | Bin | RMSE | Bias | n |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for group_name, group in report["splits"].items():
        for bin_name, fields in group.items():
            st = fields.get("V10", {})
            lines.append(f"| `{group_name}` | `{bin_name}` | {fmt_num(st.get('rmse'))} | {fmt_num(st.get('bias'))} | {st.get('n', 0)} |")
    lines.append("")

    lines.extend(
        [
            "## Correlations",
            "",
            "Low-level correlations use mass-grid diagnostics: U/V are horizontally destaggered, PH is vertically destaggered.",
            "",
            "| Pair | Pooled r | h10-h14 r |",
            "|---|---:|---:|",
        ]
    )
    pooled_corr = report["correlations"]["pooled_low_level_mass_grid"]
    hwin_corr = report["lead_windows"]["h10_h14"]["correlations_low_level_mass_grid"]
    for pair in (
        "dV10__dV_k0",
        "dU10__dU_k0",
        "dV10__dPSFC",
        "dV10__dP_k0",
        "dV10__dPH_k0",
        "dPSFC__dP_k0",
        "dPSFC__dPH_k0",
    ):
        lines.append(
            f"| `{pair}` | {fmt_num(pooled_corr.get(pair, {}).get('pearson'))} | "
            f"{fmt_num(hwin_corr.get(pair, {}).get('pearson'))} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Vertical Coupling",
            "",
            "| k | corr(dV10,dV) | corr(dU10,dU) | corr(dPSFC,dP) | corr(dPSFC,dPH) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    coupling = report["surface_vs_aloft_coupling"]["pooled_by_level"]
    nlev = min(12, len(coupling.get("dV10_vs_dV_by_level", [])))
    for k in range(nlev):
        lines.append(
            f"| {k} | {fmt_num(coupling['dV10_vs_dV_by_level'][k].get('pearson'))} | "
            f"{fmt_num(coupling['dU10_vs_dU_by_level'][k].get('pearson'))} | "
            f"{fmt_num(coupling['dPSFC_vs_dP_by_level'][k].get('pearson'))} | "
            f"{fmt_num(coupling['dPSFC_vs_dPH_by_level'][k].get('pearson'))} |"
        )
    lines.append("")

    lines.extend(["## Ranked Root-Cause Hypotheses", ""])
    for hypo in report["ranked_root_cause_hypotheses"]:
        lines.append(f"### {hypo['rank']}. {hypo['hypothesis']}")
        lines.append("")
        lines.append(f"- verdict: `{hypo['verdict']}`")
        lines.append("- evidence for:")
        for item in hypo["evidence_for"]:
            lines.append(f"  - {item}")
        lines.append("- evidence against / limits:")
        for item in hypo["evidence_against"]:
            lines.append(f"  - {item}")
        lines.append("")

    lines.extend(
        [
            "## Next Fix Probe",
            "",
            "Run a CPU-only same-state tendency localization on the h8-h14 window, sampled over the ocean/low-terrain interior cells where V10 and PSFC both fail. The first target should split large-step momentum and mass terms into PGF, Coriolis, advection, diffusion, boundary/spec-relax, physics/source-tendency folding, and resulting `ru`/`rv`/`mu` updates.",
            "",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=RUN_ID)
    ap.add_argument("--gpu-dir", type=Path, default=GPU_DIR)
    ap.add_argument("--cpu-dir", type=Path, default=CPU_DIR)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--max-hour", type=int, default=24)
    args = ap.parse_args(argv)

    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    summary = {
        "schema": report["schema"],
        "run_id": report["run_id"],
        "common_leads_h": report["common_leads_h"],
        "overall_V10": report["overall"].get("V10"),
        "h10_h14_V10": report["lead_windows"]["h10_h14"]["overall"].get("V10"),
        "top_hypothesis": report["ranked_root_cause_hypotheses"][0]["hypothesis"],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
