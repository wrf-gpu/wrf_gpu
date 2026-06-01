#!/usr/bin/env python
"""DIAGNOSIS: score d03 GPU wrfout PSFC + T2 bias against the corpus L3 d03 truth.

Confirms the P0-6 hydrostatic-ph' boundary fix collapsed the +2.6 kPa surface-
pressure error and the Exner-driven T2 warm bias.  CPU-only post-processing of
already-written wrfouts (no GPU/model run).  Pins CPU 0-3.

Usage: d03_psfc_t2_check.py <gpu_output_dir>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

import numpy as np
from netCDF4 import Dataset

CORB = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")


def _tag(p: Path) -> str:
    # wrfout_d03_2026-05-21_19:00:00 -> 2026-05-21_19:00:00
    return p.name.replace("wrfout_d03_", "")


def _psfc(ds: Dataset) -> np.ndarray:
    P = np.asarray(ds.variables["P"][0], dtype=np.float64)
    PB = np.asarray(ds.variables["PB"][0], dtype=np.float64)
    return (P + PB)[0]


def main() -> int:
    out_dir = Path(sys.argv[1])
    gpu_files = sorted(out_dir.glob("wrfout_d03_*"))
    if not gpu_files:
        print(f"NO wrfout in {out_dir}")
        return 1
    print(f"=== d03 PSFC + T2 vs corpus ({len(gpu_files)} leads) ===")
    print(f"{'lead':>21} {'psfc_bias_Pa':>13} {'T2_bias_K':>10} {'T2_rmse_K':>10} {'TSK_bias':>9}")
    psfc_b, t2_b, t2_r = [], [], []
    for gp in gpu_files:
        tag = _tag(gp)
        cp = CORB / f"wrfout_d03_{tag}"
        if not cp.exists():
            continue
        g = Dataset(gp)
        c = Dataset(cp)
        gpsfc = _psfc(g)
        cpsfc = _psfc(c)
        gt2 = np.asarray(g.variables["T2"][0], dtype=np.float64)
        ct2 = np.asarray(c.variables["T2"][0], dtype=np.float64)
        gtsk = np.asarray(g.variables["TSK"][0], dtype=np.float64)
        ctsk = np.asarray(c.variables["TSK"][0], dtype=np.float64)
        pb = float(np.nanmean(gpsfc - cpsfc))
        tb = float(np.nanmean(gt2 - ct2))
        tr = float(np.sqrt(np.nanmean((gt2 - ct2) ** 2)))
        tskb = float(np.nanmean(gtsk - ctsk))
        psfc_b.append(pb)
        t2_b.append(tb)
        t2_r.append(tr)
        print(f"{tag:>21} {pb:>+13.1f} {tb:>+10.3f} {tr:>10.3f} {tskb:>+9.3f}")
        g.close()
        c.close()
    if psfc_b:
        print("---")
        print(f"MEAN  psfc_bias={np.mean(psfc_b):+.1f} Pa   "
              f"T2_bias={np.mean(t2_b):+.3f} K   T2_rmse(mean over leads)={np.mean(t2_r):.3f} K")
        print(f"FINAL lead T2_bias={t2_b[-1]:+.3f} K  T2_rmse={t2_r[-1]:.3f} K  psfc_bias={psfc_b[-1]:+.1f} Pa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
