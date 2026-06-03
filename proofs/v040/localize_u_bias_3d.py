"""Opus v040 round — spatially localize the standalone U bias in the 3D field.

Compares the JAX standalone forecast wrfout 3D U/V/PSFC against the CPU-WRF
reference wrfout at each available lead.  Splits the bias into:
  * interior (ring-5-excluded) vs boundary-ring (width 5),
  * per vertical level (is the bias surface-trapped or full-column?),
so we can tell whether the +1.2 m/s U10 bias is (a) an interior free-dynamics
imbalance (uniform interior, all levels), (b) boundary-ring leakage spreading
inward, or (c) surface-trapped (a near-surface drag / surface-layer issue).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import netCDF4

JAX_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/v040_loc/20260521_18z_l3_24h_20260522T133443Z_d01")
REF_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
OUT = Path(__file__).resolve().parent / "localize_u_bias_3d.json"


def _interior(ny, nx, ring=5):
    m = np.ones((ny, nx), dtype=bool)
    m[:ring] = m[-ring:] = m[:, :ring] = m[:, -ring:] = False
    return m


def _read(ds, name):
    return np.asarray(ds.variables[name][0], dtype=np.float64)


def _stats3d(diff, ring=5):
    nz, ny, nx = diff.shape
    intr = _interior(ny, nx, ring)
    intr3 = np.broadcast_to(intr[None], diff.shape)
    return {
        "full_mean": float(np.mean(diff)),
        "full_rmse": float(np.sqrt(np.mean(diff**2))),
        "interior_mean": float(np.mean(diff[intr3])),
        "interior_rmse": float(np.sqrt(np.mean(diff[intr3]**2))),
        "boundary_ring_mean": float(np.mean(diff[~intr3])),
        "boundary_ring_rmse": float(np.sqrt(np.mean(diff[~intr3]**2))),
        "k0_interior_mean": float(np.mean(diff[0][intr])),
        "k0_interior_rmse": float(np.sqrt(np.mean(diff[0][intr]**2))),
        "klow_0_4_interior_mean": float(np.mean(diff[:5][np.broadcast_to(intr[None], diff[:5].shape)])),
        "kmid_interior_mean": float(np.mean(diff[nz//2][intr])),
        "ktop_interior_mean": float(np.mean(diff[-1][intr])),
    }


def _stats2d(diff, ring=5):
    ny, nx = diff.shape
    intr = _interior(ny, nx, ring)
    return {
        "full_mean": float(np.mean(diff)), "full_rmse": float(np.sqrt(np.mean(diff**2))),
        "interior_mean": float(np.mean(diff[intr])), "interior_rmse": float(np.sqrt(np.mean(diff[intr]**2))),
        "boundary_ring_mean": float(np.mean(diff[~intr])),
    }


def main() -> int:
    report: dict[str, Any] = {
        "schema": "v040_localize_u_bias_3d.v1", "created_by": "Opus 4.8 MAX",
        "jax_dir": str(JAX_DIR), "ref_dir": str(REF_DIR),
        "note": "JAX standalone wrfout minus CPU-WRF wrfout 3D U/V + PSFC, interior vs boundary-ring + per-level.",
        "leads": {},
    }
    jax_files = sorted(JAX_DIR.glob("wrfout_d01_*"))
    for jf in jax_files:
        stamp = jf.name.replace("wrfout_d01_", "")
        rf = REF_DIR / jf.name
        if not rf.is_file():
            continue
        with netCDF4.Dataset(jf) as jd, netCDF4.Dataset(rf) as rd:
            try:
                ju, jv = _read(jd, "U"), _read(jd, "V")
                ru, rv = _read(rd, "U"), _read(rd, "V")
                jp, rp = _read(jd, "PSFC"), _read(rd, "PSFC")
                ju10, jv10 = _read(jd, "U10"), _read(jd, "V10")
                ru10, rv10 = _read(rd, "U10"), _read(rd, "V10")
            except KeyError as e:
                report["leads"][stamp] = {"error": f"missing var {e}"}
                continue
            report["leads"][stamp] = {
                "U_3d": _stats3d(ju - ru),
                "V_3d": _stats3d(jv - rv),
                "PSFC": _stats2d(jp - rp),
                "U10": _stats2d(ju10 - ru10),
                "V10": _stats2d(jv10 - rv10),
            }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
