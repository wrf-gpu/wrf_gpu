#!/usr/bin/env python
"""DIAGNOSIS (2026-06-01 opus t2-bias bisection): offline Exner knockout.

Proves the d03 1km T2 +1.5 K warm bias is a DIAGNOSTIC surface-pressure error
fed into the Exner conversion t2 = th2*(psfc/P0)^kappa, NOT a land/skin solve or
an atmospheric warm bias.

For each forecast hour it:
  1. reports T2 bias, TSK bias, lowest-level theta bias, surface-pressure bias
     vs the corpus L3 d03 truth;
  2. inverts the GPU's own T2 back to its implied 2-m potential temperature th2
     using the GPU surface pressure (= state.p[0], what surface_layer used as
     psfcpa), then RE-converts th2 with the CORPUS surface pressure;
  3. shows the re-Exner'd T2 bias collapses from +1.5..+1.95 K to ~-0.5 K.

Read-only over existing wrfout artifacts; no GPU, no production-code edits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

GPU_DIR = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/tmp/v010_d03_runs/d03_20260521_18z_l3_24h_20260522T133443Z_p1_4a"
)
COR_DIR = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z"
)

R_D = 287.0
CP = 1004.5
KAPPA = R_D / CP
P0 = 1.0e5


def f(ds, n):
    v = ds.variables[n]
    d = v[0] if v.dimensions and v.dimensions[0] == "Time" else v[:]
    return np.asarray(np.ma.filled(d, np.nan), dtype=np.float64)


def main() -> int:
    tags = sorted(p.name.replace("wrfout_d03_", "") for p in GPU_DIR.glob("wrfout_d03_*"))
    print(f"GPU dir: {GPU_DIR}")
    print(
        f"{'hour(valid)':21} {'T2bias':>7} {'TSKb':>6} {'th0b':>6} "
        f"{'Pbias':>8} {'origRMSE':>8} {'corrT2b':>8} {'corrRMSE':>8}"
    )
    for tag in tags:
        gp = GPU_DIR / f"wrfout_d03_{tag}"
        cp = COR_DIR / f"wrfout_d03_{tag}"
        if not cp.is_file():
            continue
        g, c = Dataset(gp), Dataset(cp)
        T2g, T2c = f(g, "T2"), f(c, "T2")
        TSKg, TSKc = f(g, "TSK"), f(c, "TSK")
        th0g = f(g, "T")[0] + 300.0
        th0c = f(c, "T")[0] + 300.0
        psfc_gpu = (f(g, "P") + f(g, "PB"))[0]
        psfc_cor = (f(c, "P") + f(c, "PB"))[0]
        th2 = T2g / (psfc_gpu / P0) ** KAPPA
        T2_corr = th2 * (psfc_cor / P0) ** KAPPA
        print(
            f"{tag:21} {np.nanmean(T2g - T2c):+7.3f} {np.nanmean(TSKg - TSKc):+6.3f} "
            f"{np.nanmean(th0g - th0c):+6.3f} {np.nanmean(psfc_gpu - psfc_cor):+8.1f} "
            f"{np.sqrt(np.nanmean((T2g - T2c) ** 2)):8.3f} "
            f"{np.nanmean(T2_corr - T2c):+8.3f} "
            f"{np.sqrt(np.nanmean((T2_corr - T2c) ** 2)):8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
