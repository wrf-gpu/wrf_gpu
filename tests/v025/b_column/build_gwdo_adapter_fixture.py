"""Adapter-level GWDO fixture (B42): real PROD d01 State crops + pristine-WRF face increments.

State leaves come straight from the CPU-WRF history frame (THM+300, QVAPOR, P+PB,
PH+PHB, U, V, statics, FNM/FNP, DX). Expected u/v face increments are WRF
add_a2c_u/v of dt * the pristine bl_gwdo_run tendencies (build_gwdo_oracle.py
output for the full domain, so every crop column carries its true column result).

usage: build_gwdo_adapter_fixture.py <oracle dir> <out.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gwdo_oracle import CASE, STATIC  # noqa: E402

CROPS = {"night": ("wrfout_d01_2026-07-26_00:00:00", 19, 103),
         "day": ("wrfout_d01_2026-07-26_12:00:00", 19, 103)}
N = 16


def main():
    import netCDF4
    oracle, out = Path(sys.argv[1]), Path(sys.argv[2])
    arrays = {}
    for case, (frame, j0, i0) in CROPS.items():
        with netCDF4.Dataset(CASE / frame) as d:
            g = lambda n: np.asarray(d.variables[n][0], np.float32)
            nz = g("P").shape[0]
            m = (slice(None), slice(j0, j0 + N), slice(i0, i0 + N))
            arrays.update({
                f"{case}_theta": (g("THM") + np.float32(300.0))[m], f"{case}_qv": g("QVAPOR")[m],
                f"{case}_p": (g("P") + g("PB"))[m], f"{case}_ph": (g("PH") + g("PHB"))[m],
                f"{case}_u": g("U")[:, j0:j0 + N, i0:i0 + N + 1],
                f"{case}_v": g("V")[:, j0:j0 + N + 1, i0:i0 + N],
                f"{case}_fnm": g("FNM"), f"{case}_fnp": g("FNP"),
                f"{case}_dx": np.float32(d.DX), f"{case}_crop": np.asarray([j0, i0]),
                **{f"{case}_st_{n}": g(n)[j0:j0 + N, i0:i0 + N]
                   for n in (*STATIC, "SINALPHA", "COSALPHA")}})
        z = np.load(oracle / case / "columns.npz")
        dt = float(z["dt"])
        cols = (np.arange(j0, j0 + N)[:, None] * 120 + np.arange(i0, i0 + N)[None, :]).ravel()
        du = (z["wrf_rublten"][:, cols].astype(np.float64) * dt).reshape(nz, N, N)
        dv = (z["wrf_rvblten"][:, cols].astype(np.float64) * dt).reshape(nz, N, N)
        eu = np.zeros((nz, N, N + 1))
        ev = np.zeros((nz, N + 1, N))
        eu[:, :, 1:-1] = 0.5 * (du[:, :, :-1] + du[:, :, 1:])
        ev[:, 1:-1, :] = 0.5 * (dv[:, :-1, :] + dv[:, 1:, :])
        arrays.update({f"{case}_expected_du": eu, f"{case}_expected_dv": ev,
                       f"{case}_dt": np.float64(dt)})
    np.savez_compressed(out, **arrays)
    print("wrote", out, {k: v.shape for k, v in arrays.items() if k.startswith("night_") and v.ndim})


if __name__ == "__main__":
    main()
