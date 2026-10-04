"""Max-Courant diagnostic (TEST INFRA ONLY -- no model code).

Reads a wrfout NetCDF history file and computes the worst-case advective
Courant numbers per frame, per the WRF C-grid staggering:

    Cx = max |U| * dt / dx           (x-momentum, U on x-faces)
    Cy = max |V| * dt / dy           (y-momentum, V on y-faces)
    Cz = max |W| * dt / dz           (w on w-faces; dz from full geopotential)

where dz between mass levels is recovered from the full geopotential
(PH + PHB) / g differenced across the vertical faces.

NO existing CFL diagnostic was trusted (only a mass-continuity stability
fraction in acoustic_wrf.py); this is the K2 evidence readout the prep doc
(proofs/v022/GPU_OPENER_PREP.md) requires.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

G = 9.81


def _read(ds: Dataset, name: str) -> np.ndarray:
    return np.asarray(ds.variables[name][:], dtype=np.float64)


def courant_for_file(path: str | Path, dt_s: float) -> dict:
    """Return per-frame and worst-over-file Courant numbers for one wrfout."""
    path = Path(path)
    with Dataset(str(path)) as ds:
        # dx/dy: global attrs DX/DY (meters); fall back to RDX if needed.
        dx = float(getattr(ds, "DX"))
        dy = float(getattr(ds, "DY"))
        U = _read(ds, "U")  # (Time, z, y, x+1)
        V = _read(ds, "V")  # (Time, z, y+1, x)
        W = _read(ds, "W")  # (Time, z+1, y, x)
        PH = _read(ds, "PH")  # (Time, z+1, y, x) perturbation geopotential
        PHB = _read(ds, "PHB")  # (Time, z+1, y, x) base geopotential

    z_full = (PH + PHB) / G  # geopotential height on w-faces (m)
    # dz between adjacent w-faces (i.e. mass-layer thickness), shape (Time,z,y,x)
    dz = np.diff(z_full, axis=1)
    dz = np.where(np.abs(dz) < 1.0e-6, np.nan, dz)

    ntime = U.shape[0]
    frames = []
    worst = {"Cx": 0.0, "Cy": 0.0, "Cz": 0.0, "C_total": 0.0}
    for t in range(ntime):
        umax = float(np.nanmax(np.abs(U[t])))
        vmax = float(np.nanmax(np.abs(V[t])))
        cx = umax * dt_s / dx
        cy = vmax * dt_s / dy
        # vertical: W is on w-faces (z+1). Use interior faces aligned to dz layers
        # by averaging the two bounding faces onto the mass-layer; conservative max.
        w_layer = 0.5 * (np.abs(W[t, :-1]) + np.abs(W[t, 1:]))
        cz_field = w_layer * dt_s / np.abs(dz[t])
        cz = float(np.nanmax(cz_field))
        c_total = cx + cy + cz
        frame = {
            "frame": t,
            "max_abs_U": umax,
            "max_abs_V": vmax,
            "max_abs_W": float(np.nanmax(np.abs(W[t]))),
            "Cx": cx,
            "Cy": cy,
            "Cz": cz,
            "C_total": c_total,
            "min_dz_m": float(np.nanmin(np.abs(dz[t]))),
        }
        frames.append(frame)
        for k in ("Cx", "Cy", "Cz", "C_total"):
            if frame[k] > worst[k]:
                worst[k] = frame[k]

    return {
        "file": str(path),
        "dt_s": float(dt_s),
        "dx_m": dx,
        "dy_m": dy,
        "n_frames": ntime,
        "worst": worst,
        "frames": frames,
    }


def courant_for_run(wrfout_paths, dt_s: float) -> dict:
    per_file = [courant_for_file(p, dt_s) for p in wrfout_paths]
    worst = {"Cx": 0.0, "Cy": 0.0, "Cz": 0.0, "C_total": 0.0}
    for f in per_file:
        for k in worst:
            worst[k] = max(worst[k], f["worst"][k])
    return {"dt_s": float(dt_s), "worst": worst, "per_file": per_file}
