#!/usr/bin/env python3
"""Pin the QVAPOR redistribution to PBL entrainment depth, not surface flux.

If the GPU mixed layer is too shallow / too sharply capped, the diagnostic
PBLH should be biased LOW over the same ocean columns where qv is mis-placed,
and the qv-bias sign should flip exactly at the inversion the GPU caps below
the CPU's. Pure file analysis, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

CPU = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/"
           "20260501_18z_l2_72h_20260519T173026Z")
GPU = Path("<DATA_ROOT>/wrf_gpu_validation/v015_canary_d02_72h_allgreen/"
           "gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z")
FRAME = 5


def _open(p):
    return xr.open_dataset(p, decode_times=False)


def _interior(a):
    return a[..., FRAME:-FRAME, FRAME:-FRAME]


def _f(ds, name):
    return np.asarray(ds[name].values[0], dtype=np.float64)


def pblh_check(stamps):
    rows = []
    for stamp in stamps:
        cpuf = CPU / f"wrfout_d02_{stamp}"
        gpuf = GPU / f"wrfout_d02_{stamp}"
        if not cpuf.exists() or not gpuf.exists():
            continue
        c, g = _open(cpuf), _open(gpuf)
        ocean = _interior(_f(c, "XLAND")) > 1.5
        cp = _interior(_f(c, "PBLH"))[ocean]
        gp = _interior(_f(g, "PBLH"))[ocean]
        rows.append({
            "stamp": stamp,
            "cpu_pblh_mean_m": float(np.mean(cp)),
            "gpu_pblh_mean_m": float(np.mean(gp)),
            "pblh_bias_m": float(np.mean(gp - cp)),
            "pblh_pct_bias": float(100.0 * np.mean(gp - cp) / np.mean(cp)),
            "cpu_pblh_median_m": float(np.median(cp)),
            "gpu_pblh_median_m": float(np.median(gp)),
        })
        c.close(); g.close()
    return rows


def main():
    stamps = [
        "2026-05-01_19:00:00",  # h1
        "2026-05-02_06:00:00",  # h12
        "2026-05-02_18:00:00",  # h24
        "2026-05-03_06:00:00",  # h36
        "2026-05-03_18:00:00",  # h48
        "2026-05-04_18:00:00",  # h72
    ]
    res = {"purpose": "PBLH entrainment-depth bias over ocean (QVAPOR redistribution mechanism)",
           "pblh_ocean": pblh_check(stamps)}
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/pblh_entrainment.json")
    out.write_text(json.dumps(res, indent=2))
    print("=== PBLH over ocean (GPU vs CPU) ===")
    for r in res["pblh_ocean"]:
        print(f"  {r['stamp']}  CPU {r['cpu_pblh_mean_m']:7.1f} m  GPU {r['gpu_pblh_mean_m']:7.1f} m  "
              f"bias {r['pblh_bias_m']:+7.1f} m ({r['pblh_pct_bias']:+5.1f}%)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
