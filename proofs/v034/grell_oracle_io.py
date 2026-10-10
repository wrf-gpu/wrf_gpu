"""v0.3.4 o1-grell: input generation + I/O for the Grell-family tile oracles.

The oracle executables (``oracle/cumulus_grell/build.sh``) call the UNMODIFIED
pristine WRF ``G3DRV`` + ``conv_grell_spread3d`` (cu_physics=5) and ``GRELLDRV``
(cu_physics=93) on an ``nx x ny`` tile.  This module builds varied but physically
structured soundings (deep / capped-shallow / stable / marginal regimes, land and
water, day and night, random advective/radiative/PBL forcing), writes the binary
input, runs the oracle and parses its binary output.  NumPy only (no JAX import).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

R_D = 287.0
CP = 7.0 * R_D / 2.0
G = 9.81
P0 = 1.0e5
ENSDIM = 144

FIELDS3_IN = ("u", "v", "w", "t", "q", "p", "pi", "rho", "dz8w", "p8w",
              "rthften", "rqvften", "rthraten", "rthblten", "rqvblten")
FIELDS2_IN = ("ht", "xland", "gsw", "kpbl", "htop", "hbot")
FIELDS3_OUT = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "CUGD_TTEN",
               "CUGD_QVTEN", "CUGD_QCTEN", "CUGD_TTENS", "CUGD_QVTENS", "GDC", "GDC2")
FIELDS2_OUT = ("RAINCV", "PRATEC", "DRV_RAINCV", "DRV_PRATEC", "HTOP", "HBOT",
               "MASS_FLUX", "EDT_OUT", "APR_GR", "APR_W", "APR_MC", "APR_ST",
               "APR_AS", "APR_CAPMA", "APR_CAPME", "APR_CAPMI", "XMB_SHALLOW",
               "KTOP_DEEP", "K22_SHALLOW", "KBCON_SHALLOW", "KTOP_SHALLOW")

REGIMES = ("deep", "deep_dryupper", "shallow_capped", "stable", "marginal", "cold_deep")


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    es = np.minimum(es, 0.95 * p)
    return 0.622 * es / (p - es)


def _column(rng, regime, kx, ztop=19500.0):
    """One column: returns dict of (kx+1,) arrays (index kx = top interface pad)."""
    zf = ztop * (np.arange(kx + 1) / kx) ** 1.25          # interface heights (kx+1)
    zm = 0.5 * (zf[:-1] + zf[1:])                          # mass heights (kx)
    j = lambda a, b: rng.uniform(a, b)
    if regime == "deep":
        psfc, tsfc, zml, lap, rh_ml, rh_free, cap, wpk = (j(100600, 101500), j(300, 311),
            j(900, 2000), j(1.2e-3, 3.0e-3), j(0.85, 0.99), j(0.55, 0.9), 0.0, j(0.2, 4.0))
    elif regime == "deep_dryupper":
        psfc, tsfc, zml, lap, rh_ml, rh_free, cap, wpk = (j(100600, 101500), j(301, 310),
            j(900, 1800), j(1.5e-3, 3.5e-3), j(0.85, 0.97), j(0.15, 0.45), 0.0, j(0.0, 2.5))
    elif regime == "shallow_capped":
        psfc, tsfc, zml, lap, rh_ml, rh_free, cap, wpk = (j(100300, 101300), j(294, 303),
            j(500, 1200), j(3.5e-3, 6.0e-3), j(0.8, 0.95), j(0.2, 0.5), j(1.0, 4.0), j(-0.3, 0.6))
    elif regime == "stable":
        psfc, tsfc, zml, lap, rh_ml, rh_free, cap, wpk = (j(99800, 101000), j(275, 292),
            j(100, 400), j(7e-3, 1.1e-2), j(0.3, 0.7), j(0.1, 0.3), 0.0, j(-0.3, 0.05))
    elif regime == "marginal":
        psfc, tsfc, zml, lap, rh_ml, rh_free, cap, wpk = (j(100300, 101300), j(296, 306),
            j(700, 1600), j(2.5e-3, 4.5e-3), j(0.75, 0.95), j(0.35, 0.7), j(0.0, 1.5), j(-0.5, 1.5))
    else:  # cold_deep: freezing-level low enough for ice (T<258) tendencies
        psfc, tsfc, zml, lap, rh_ml, rh_free, cap, wpk = (j(99000, 100800), j(285, 295),
            j(500, 1300), j(1.5e-3, 3.0e-3), j(0.85, 0.98), j(0.6, 0.9), 0.0, j(0.3, 3.0))
    ushr = j(-15, 15)
    vshr = j(-15, 15)
    upper_start = j(8500.0, 11500.0)
    upper_lapse = j(8e-3, 1.4e-2)
    theta_sfc = tsfc * (P0 / psfc) ** (R_D / CP)
    th = np.where(zm <= zml, theta_sfc, theta_sfc + lap * (zm - zml))
    capbump = cap * np.exp(-((zm - (zml + 600.0)) / 450.0) ** 2) * (zm > zml)
    th = th + capbump + np.where(zm > upper_start, upper_lapse * (zm - upper_start), 0.0)
    th = th + rng.normal(0.0, 0.15, kx)
    rh = np.where(zm <= zml, rh_ml, rh_free + (rh_ml - rh_free) * np.exp(-(zm - zml) / j(1500, 3500)))
    rh = np.clip(rh + rng.normal(0.0, 0.03, kx), 0.02, 0.995)
    # hydrostatic integration (interfaces) with virtual temperature
    pf = np.empty(kx + 1)
    pm = np.empty(kx)
    t = np.empty(kx)
    q = np.empty(kx)
    pf[0] = psfc
    for k in range(kx):
        # first guess with dry T, iterate once with moisture
        tvk = th[k] * (pf[k] / P0) ** (R_D / CP)
        for _ in range(3):
            pm_k = pf[k] * np.exp(-G * (zm[k] - zf[k]) / (R_D * tvk))
            t_k = th[k] * (pm_k / P0) ** (R_D / CP)
            q_k = max(1.0e-7, rh[k] * _qsat(t_k, pm_k))
            tvk = t_k * (1.0 + 0.608 * q_k)
        pm[k], t[k], q[k] = pm_k, t_k, q_k
        pf[k + 1] = pf[k] * np.exp(-G * (zf[k + 1] - zf[k]) / (R_D * tvk))
    tv = t * (1.0 + 0.608 * q)
    rho = pm / (R_D * tv)
    pi = (pm / P0) ** (R_D / CP)
    u = ushr * np.minimum(1.0, zm / 10000.0) + rng.normal(0, 0.5, kx)
    v = vshr * np.minimum(1.0, zm / 10000.0) + rng.normal(0, 0.5, kx)
    w = wpk * np.exp(-((zf - j(1200, 4000)) / j(900, 2500)) ** 2) + rng.normal(0.0, 0.05, kx + 1)
    w[0] = 0.0
    pad = lambda a: np.concatenate([a, a[-1:]])
    return dict(u=pad(u), v=pad(v), w=w, t=pad(t), q=pad(q), p=pad(pm), pi=pad(pi),
                rho=pad(rho), dz8w=pad(np.diff(zf)), p8w=pf)


def make_tile(seed, nx=16, ny=16, kx=44, regimes=REGIMES, forcing_scale=1.0):
    """Build an (nx, ny) tile of independent columns with smooth-ish neighbours."""
    rng = np.random.default_rng(seed)
    f3 = {name: np.zeros((nx, kx + 1, ny)) for name in FIELDS3_IN}
    f2 = {name: np.zeros((nx, ny)) for name in FIELDS2_IN}
    reg = np.empty((nx, ny), dtype=object)
    for jj in range(ny):
        for ii in range(nx):
            r = regimes[rng.integers(len(regimes))]
            reg[ii, jj] = r
            col = _column(rng, r, kx)
            for name, val in col.items():
                f3[name][ii, :, jj] = val
            f2["ht"][ii, jj] = rng.uniform(0.0, 1500.0) if rng.random() < 0.6 else 0.0
            f2["xland"][ii, jj] = 1.0 if rng.random() < 0.6 else 2.0
            f2["gsw"][ii, jj] = rng.uniform(100.0, 900.0) if rng.random() < 0.6 else 0.0
            f2["kpbl"][ii, jj] = rng.integers(2, 13)
            f2["hbot"][ii, jj] = float(kx)
            f2["htop"][ii, jj] = 1.0
    s = forcing_scale
    f3["rthften"] = s * rng.normal(0.0, 3e-5, (nx, kx + 1, ny))
    f3["rqvften"] = s * rng.normal(0.0, 4e-8, (nx, kx + 1, ny))
    f3["rthraten"] = s * rng.normal(-1.5e-5, 1e-5, (nx, kx + 1, ny))
    bl = np.zeros((nx, kx + 1, ny))
    for jj in range(ny):
        for ii in range(nx):
            kp = int(f2["kpbl"][ii, jj])
            bl[ii, :kp, jj] = 1.0
    f3["rthblten"] = s * bl * rng.uniform(0.0, 1.5e-4, (nx, 1, ny))
    f3["rqvblten"] = s * bl * rng.uniform(-2e-9, 1.2e-8, (nx, 1, ny))
    return dict(f3=f3, f2=f2, regimes=reg, nx=nx, ny=ny, kx=kx)


def write_input(path, tile, *, dt, dx, ishallow=0, cugd_avedx=1, ichoice=0, periodic=0, itimestep=1):
    nx, ny, kx = tile["nx"], tile["ny"], tile["kx"]
    with open(path, "wb") as fh:
        np.array([nx, ny, kx, ishallow, cugd_avedx, ichoice, periodic, itimestep], "<i4").tofile(fh)
        np.array([dt, dx], "<f8").tofile(fh)
        for name in FIELDS3_IN:
            np.asfortranarray(tile["f3"][name], dtype="<f8").T.tofile(fh)  # Fortran order
        for name in FIELDS2_IN:
            np.asfortranarray(tile["f2"][name], dtype="<f8").T.tofile(fh)


def read_output(path, nx, ny, kx):
    raw = np.fromfile(path, "<f8")
    out = {}
    off = 0
    n3 = nx * (kx + 1) * ny
    for name in FIELDS3_OUT:
        out[name] = raw[off:off + n3].reshape((ny, kx + 1, nx)).transpose(2, 1, 0)[:, :kx, :]
        off += n3
    n2 = nx * ny
    for name in FIELDS2_OUT:
        out[name] = raw[off:off + n2].reshape((ny, nx)).T.copy()
        off += n2
    out["XF_ENS"] = raw[off:off + n2 * ENSDIM].reshape((ENSDIM, ny, nx)).transpose(2, 1, 0).copy()
    off += n2 * ENSDIM
    out["PR_ENS"] = raw[off:off + n2 * ENSDIM].reshape((ENSDIM, ny, nx)).transpose(2, 1, 0).copy()
    off += n2 * ENSDIM
    assert off == raw.size, (off, raw.size)
    return out


def run_oracle(exe, tile, *, workdir=None, cpus="13", **opts):
    exe = str(exe)
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        fin = os.path.join(td, "in.bin")
        fout = os.path.join(td, "out.bin")
        write_input(fin, tile, **opts)
        cmd = ["taskset", "-c", cpus, "nice", "-n", "19", exe, fin, fout]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(f"oracle failed rc={res.returncode}: {res.stderr[-2000:]}")
        return read_output(fout, tile["nx"], tile["ny"], tile["kx"])


def tile_to_npz_dict(tile):
    d = {f"in_{k}": v for k, v in tile["f3"].items()}
    d.update({f"in2_{k}": v for k, v in tile["f2"].items()})
    return d
