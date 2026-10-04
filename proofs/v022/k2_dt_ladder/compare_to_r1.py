"""Operational-tolerance comparator vs the R1 baseline (TEST INFRA ONLY).

Given a rung's final wrfout and the R1 baseline's final wrfout (same valid
time, same grid), compute per-field RMSE / max-abs-diff / bias on the
operational fields. K2 is LOSSLESS-but-not-bit-identical: a larger dt yields a
different trajectory, so we do NOT expect zero diff; the gate is that the diff
stays within a strict operational band (the run is the SAME forecast, just a
coarser time step).

Reports surface fields (T2, U10, V10, PSFC) and 3-D dynamics (U, V, W, T,
QVAPOR). The summary verdict flags any field exceeding the band.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# Strict operational band (per field). These are conservative "same forecast,
# coarser dt" bands; a rung that blows past them is NOT operationally equivalent.
BANDS_RMSE = {
    "T2": 0.5,      # K
    "U10": 0.75,    # m/s
    "V10": 0.75,    # m/s
    "PSFC": 60.0,   # Pa
    "T": 0.75,      # K (perturbation theta, 3-D)
    "U": 1.5,       # m/s
    "V": 1.5,       # m/s
    "W": 0.5,       # m/s
    "QVAPOR": 5.0e-4,  # kg/kg
}


def _load(path: str, names) -> dict:
    out = {}
    with Dataset(str(path)) as ds:
        for n in names:
            if n in ds.variables:
                out[n] = np.asarray(ds.variables[n][:], dtype=np.float64)
    return out


def compare(rung_wrfout: str, r1_wrfout: str) -> dict:
    names = list(BANDS_RMSE.keys())
    a = _load(rung_wrfout, names)
    b = _load(r1_wrfout, names)
    fields = {}
    worst_ratio = 0.0
    any_exceed = False
    for n in names:
        if n not in a or n not in b:
            continue
        if a[n].shape != b[n].shape:
            fields[n] = {"shape_mismatch": [list(a[n].shape), list(b[n].shape)]}
            continue
        d = a[n] - b[n]
        rmse = float(np.sqrt(np.nanmean(d * d)))
        maxabs = float(np.nanmax(np.abs(d)))
        bias = float(np.nanmean(d))
        band = BANDS_RMSE[n]
        ratio = rmse / band if band > 0 else 0.0
        exceed = rmse > band
        any_exceed = any_exceed or exceed
        worst_ratio = max(worst_ratio, ratio)
        fields[n] = {
            "rmse": rmse,
            "max_abs_diff": maxabs,
            "bias": bias,
            "band_rmse": band,
            "rmse_over_band": ratio,
            "within_band": not exceed,
        }
    return {
        "rung_wrfout": rung_wrfout,
        "r1_wrfout": r1_wrfout,
        "fields": fields,
        "within_operational_band": not any_exceed,
        "worst_rmse_over_band": worst_ratio,
    }


def main() -> int:
    rung = sys.argv[1]
    r1 = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    res = compare(rung, r1)
    text = json.dumps(res, indent=2, default=str)
    if out:
        Path(out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
