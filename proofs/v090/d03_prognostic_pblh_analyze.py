#!/usr/bin/env python
"""PART-2 augmentation: d03 1km PBLH + prognostic-level RMSE vs CPU 1km truth.

scripts/d03_replay.py already computes per-lead T2/U10/V10/RAINNC RMSE +
persistence skill into d03_validation.json. This adds the two fields the v0.9.0
gate also wants reported but that the surface-only validator does not cover:
  * PBLH (surface diagnostic)
  * prognostic 3D levels: U / V / W / T(theta-perturbation) / QVAPOR gridded RMSE
    at a representative low / mid / model-top level, every output hour.

Post-processing only (NO new GPU run): consumes the GPU d03 wrfout already
written and the CPU corpus d03 truth.

Usage:
  taskset -c 0-3 python3 proofs/v090/d03_prognostic_pblh_analyze.py \
      --gpu-output-dir /tmp/vburst_d03/d03_<run_id> \
      --cpu-ref-dir <DATA_ROOT>/canairy_meteo/runs/wrf_l3/<run_id> \
      --out proofs/v090/d03_prognostic_pblh.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

# Informational engineering bands for the 1km prognostic comparison. The 1km
# replay is single-domain GPU vs CPU-d03-in-nest; per the speedup-benchmark
# honesty flags it is INDICATIVE, so these bands are conservative not tight.
PBLH_BAR_M = 400.0
PROG_BARS = {"U": 6.0, "V": 6.0, "W": 1.5, "T": 4.0, "QVAPOR": 0.004}  # m/s, m/s, m/s, K(theta-pert), kg/kg
SURF = ("PBLH",)
PROG = ("U", "V", "W", "T", "QVAPOR")


def _pin() -> None:
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0, 1, 2, 3})
        except OSError:
            pass


def _read(path: Path, fields) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with Dataset(path, "r") as ds:
        for name in fields:
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            data = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
            out[name] = np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)
    return out


def _rmse(a, b) -> float:
    d = (np.asarray(a) - np.asarray(b)).ravel()
    f = np.isfinite(d)
    return float(np.sqrt(np.mean(d[f] ** 2))) if f.any() else float("nan")


def _bias(a, b) -> float:
    d = (np.asarray(a) - np.asarray(b)).ravel()
    f = np.isfinite(d)
    return float(np.mean(d[f])) if f.any() else float("nan")


def _common_shape(g, r):
    """Crop to the common min extent on every axis (handles stag/mass off-by-one)."""
    sl = tuple(slice(0, min(gs, rs)) for gs, rs in zip(g.shape, r.shape))
    return g[sl], r[sl]


def _valid_time(path: Path) -> str:
    return path.name.split("wrfout_d03_")[-1].replace("_", "T", 1)


def main(argv=None) -> int:
    _pin()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-output-dir", required=True, type=Path)
    ap.add_argument("--cpu-ref-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--precision", default="gated_fp32")
    args = ap.parse_args(argv)

    gpu_files = sorted(Path(args.gpu_output_dir).glob("wrfout_d03_*"), key=lambda p: p.name)
    if not gpu_files:
        raise FileNotFoundError(f"no GPU wrfout_d03_* in {args.gpu_output_dir}")
    ref_dir = Path(args.cpu_ref_dir)

    # determine level indices from the first GPU prognostic file
    g0 = _read(gpu_files[0], ("T",))
    nz = g0["T"].shape[0] if "T" in g0 else 44
    levels = {"low_k2": min(2, nz - 1), "mid": nz // 2, "top_k-2": max(0, nz - 2)}

    per_lead = []
    failures = []
    nonfinite_leads = 0
    surf_agg = {f: [] for f in SURF}
    prog_agg = {f: {lk: [] for lk in levels} for f in PROG}

    for gf in gpu_files:
        ref_path = ref_dir / gf.name
        if not ref_path.is_file():
            failures.append(f"missing CPU d03 truth for {gf.name}")
            continue
        gpu = _read(gf, SURF + PROG)
        ref = _read(ref_path, SURF + PROG)
        rec = {"valid_time_utc": _valid_time(gf), "surface": {}, "prognostic": {}}
        any_nonfinite = False
        for f in SURF:
            if f not in gpu or f not in ref:
                continue
            g, r = _common_shape(gpu[f], ref[f])
            if not np.isfinite(g).all():
                any_nonfinite = True
            rmse = _rmse(g, r)
            surf_agg[f].append(rmse)
            rec["surface"][f] = {"rmse": rmse, "bias": _bias(g, r),
                                 "bar": PBLH_BAR_M if f == "PBLH" else None,
                                 "within_bar": bool(np.isfinite(rmse) and rmse <= PBLH_BAR_M) if f == "PBLH" else None,
                                 "units": "m"}
        for f in PROG:
            if f not in gpu or f not in ref:
                continue
            g_full, r_full = _common_shape(gpu[f], ref[f])
            if not np.isfinite(g_full).all():
                any_nonfinite = True
            rec["prognostic"][f] = {}
            for lk, lev in levels.items():
                if lev >= g_full.shape[0]:
                    continue
                gl, rl = g_full[lev], r_full[lev]
                rmse = _rmse(gl, rl)
                prog_agg[f][lk].append(rmse)
                bar = PROG_BARS.get(f)
                rec["prognostic"][f][lk] = {
                    "level_index": lev,
                    "rmse": rmse,
                    "bias": _bias(gl, rl),
                    "bar": bar,
                    "within_bar": bool(np.isfinite(rmse) and (bar is None or rmse <= bar)),
                }
        if any_nonfinite:
            nonfinite_leads += 1
        per_lead.append(rec)

    surf_summary = {f: {"mean_rmse": float(np.nanmean(v)) if v else None,
                        "max_rmse": float(np.nanmax(v)) if v else None,
                        "bar": PBLH_BAR_M if f == "PBLH" else None,
                        "all_leads_within_bar": bool(all(x <= PBLH_BAR_M for x in v if np.isfinite(x))) if (v and f == "PBLH") else None}
                    for f, v in surf_agg.items()}
    prog_summary = {}
    for f in PROG:
        prog_summary[f] = {}
        for lk in levels:
            v = [x for x in prog_agg[f][lk] if np.isfinite(x)]
            bar = PROG_BARS.get(f)
            prog_summary[f][lk] = {
                "mean_rmse": float(np.mean(v)) if v else None,
                "max_rmse": float(np.max(v)) if v else None,
                "bar": bar,
                "all_leads_within_bar": bool(all(x <= bar for x in v)) if (v and bar is not None) else None,
            }

    # overall: all prognostic levels + PBLH within their bands, all leads, all finite
    prog_ok = all(prog_summary[f][lk]["all_leads_within_bar"] in (True, None) for f in PROG for lk in levels)
    pblh_ok = surf_summary.get("PBLH", {}).get("all_leads_within_bar") in (True, None)
    status = "PASS" if (prog_ok and pblh_ok and nonfinite_leads == 0 and not failures) else "FAIL"

    payload = {
        "schema": "V090D03PrognosticPBLH",
        "schema_version": 1,
        "status": status,
        "precision_mode": args.precision,
        "gpu_output_dir": str(args.gpu_output_dir),
        "cpu_reference_dir": str(ref_dir),
        "lead_count": len(per_lead),
        "all_leads_finite": nonfinite_leads == 0,
        "nonfinite_leads": nonfinite_leads,
        "level_indices": levels,
        "surface_summary": surf_summary,
        "prognostic_summary": prog_summary,
        "per_lead": per_lead,
        "failures": failures,
        "notes": [
            "Complements scripts/d03_replay.py surface T2/U10/V10/RAINNC validation with PBLH + prognostic-level RMSE.",
            "Prognostic RMSE is gridded GPU vs CPU-d03 truth at a low / mid / near-top model level each output hour.",
            "T is theta-perturbation (WRF convention, +300K offset). Bands are INDICATIVE engineering bands (1km single-domain GPU vs CPU-d03-in-nest).",
            "Common-shape crop handles C-grid staggering off-by-one between GPU and CPU files.",
        ],
        "generated": datetime.utcnow().isoformat() + "Z",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": status, "lead_count": len(per_lead),
                      "all_leads_finite": nonfinite_leads == 0,
                      "surface_summary": surf_summary,
                      "prognostic_mid_T_mean_rmse": prog_summary.get("T", {}).get("mid", {}).get("mean_rmse")},
                     indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
