#!/usr/bin/env python
"""V0.14 Switzerland theta-advection operator oracle (CPU, WRF-literal numpy vs JAX).

At the CPU-truth h36 state (the reinit anchor), build the WRF transporting
velocities (couple_momentum + calc_ww_cp, literal numpy transcription of
module_big_step_utilities_em.F:640-782) and evaluate the flux-form theta
advection tendency two ways on the SAME inputs:

  1. WRF-literal numpy transcription of advect_scalar (h=5 / v=3,
     module_advect_em.F flux5/flux3, interior stencils);
  2. the production JAX chain (couple_velocities_periodic + advect_scalar_flux).

Any systematic interior-mean difference per level is an operator transcription
defect (the +0.5 K/h interior warm-bias candidate).  Also reports the absolute
interior-mean advective theta tendency per level (decoupled, K/s) from both, and
the CPU actual hourly mean d(theta)/dt for scale.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CPU = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
H36 = "wrfout_d01_2023-01-16_12:00:00"
H37 = "wrfout_d01_2023-01-16_13:00:00"
OUT = Path(__file__).with_suffix(".json")


def load(fname: str) -> dict[str, np.ndarray]:
    ds = Dataset(CPU / fname)
    names = ["U", "V", "T", "MU", "MUB", "QVAPOR", "MAPFAC_MX", "MAPFAC_MY",
             "MAPFAC_UY", "MAPFAC_VX", "DNW", "RDNW", "FNM", "FNP", "C1H", "C2H"]
    out = {}
    for n in names:
        v = ds.variables[n][0]
        out[n] = np.asarray(v, dtype=np.float64)
    ds.close()
    return out


def wrf_flux5(q: np.ndarray, vel: np.ndarray, axis: int) -> np.ndarray:
    """WRF flux5 face value at left face of cell i (periodic roll, interior-valid)."""
    r = lambda s: np.roll(q, s, axis=axis)
    flux6 = (37.0 * (q + r(1)) - 8.0 * (r(-1) + r(2)) + (r(-2) + r(3))) / 60.0
    corr = ((r(-2) - r(3)) - 5.0 * (r(-1) - r(2)) + 10.0 * (q - r(1))) / 60.0
    return flux6 - np.sign(vel) * corr


def main() -> int:
    d36 = load(H36)
    d37 = Dataset(CPU / H37)
    t37 = np.asarray(d37.variables["T"][0], dtype=np.float64)
    d37.close()

    # moist theta_m perturbation (the WRF prognostic t_2): THM = (T+300)(1+1.61qv)-300
    RV_RD = 0.61  # WRF rvovrd - 1... NOTE: WRF uses R_v/R_d = 1.61 -> factor (1+1.61*qv)
    thm = (d36["T"] + 300.0) * (1.0 + 1.608 * d36["QVAPOR"]) - 300.0

    mu = d36["MU"] + d36["MUB"]            # (ny, nx)
    u = d36["U"]                            # (nz, ny, nx+1)
    v = d36["V"]                            # (nz, ny+1, nx)
    nz, ny, nx = thm.shape
    dnw, rdnw = d36["DNW"], d36["RDNW"]
    fnm, fnp = d36["FNM"], d36["FNP"]
    c1h, c2h = d36["C1H"], d36["C2H"]
    msftx = d36["MAPFAC_MX"]
    msfuy = d36["MAPFAC_UY"]               # (ny, nx+1)
    msfvx = d36["MAPFAC_VX"]               # (ny+1, nx)
    dx = 3000.0
    rdx = rdy = 1.0 / dx

    # ---- WRF-literal couple + calc_ww_cp (interior; one-sided edges ignored) ----
    muu = np.empty((ny, nx + 1))
    muu[:, 1:nx] = 0.5 * (mu[:, 1:nx] + mu[:, 0 : nx - 1])
    muu[:, 0] = mu[:, 0]
    muu[:, nx] = mu[:, nx - 1]
    muv = np.empty((ny + 1, nx))
    muv[1:ny, :] = 0.5 * (mu[1:ny, :] + mu[0 : ny - 1, :])
    muv[0, :] = mu[0, :]
    muv[ny, :] = mu[ny - 1, :]

    mass_u = c1h[:, None, None] * muu[None] + c2h[:, None, None]
    mass_v = c1h[:, None, None] * muv[None] + c2h[:, None, None]
    ru = mass_u * u / msfuy[None]          # (nz, ny, nx+1)
    rv = mass_v * v / msfvx[None]          # (nz, ny+1, nx)

    divv = dnw[:, None, None] * msftx[None] * (
        rdx * (ru[:, :, 1:] - ru[:, :, :-1]) + rdy * (rv[:, 1:, :] - rv[:, :-1, :])
    )                                       # (nz, ny, nx)
    dmdt = divv.sum(axis=0)                 # (ny, nx)
    rom = np.zeros((nz + 1, ny, nx))
    for k in range(1, nz):
        rom[k] = rom[k - 1] - dnw[k - 1] * c1h[k - 1] * dmdt - divv[k - 1]

    # ---- WRF-literal advect_scalar tendency (h5 horizontal + v3 vertical) ----
    velx = ru                               # transporting at x faces (nz, ny, nx+1)
    # face value of theta at x face i (between cell i-1, i): use roll-based flux5
    # computed on the mass field; valid in the deep interior only.
    fqx_face = np.empty_like(ru)
    f5 = wrf_flux5(thm, np.sign(ru[:, :, :-1]), axis=2)  # placeholder sign arg
    # build face values with the velocity at each face
    r = lambda s: np.roll(thm, s, axis=2)
    flux6x = (37.0 * (thm + r(1)) - 8.0 * (r(-1) + r(2)) + (r(-2) + r(3))) / 60.0
    corrx = ((r(-2) - r(3)) - 5.0 * (r(-1) - r(2)) + 10.0 * (thm - r(1))) / 60.0
    facex = flux6x - np.sign(ru[:, :, :nx]) * corrx      # face i in 0..nx-1
    fqx = ru[:, :, :nx] * facex
    tend = -msftx[None] * rdx * (np.roll(fqx, -1, axis=2) - fqx)
    ry = lambda s: np.roll(thm, s, axis=1)
    flux6y = (37.0 * (thm + ry(1)) - 8.0 * (ry(-1) + ry(2)) + (ry(-2) + ry(3))) / 60.0
    corry = ((ry(-2) - ry(3)) - 5.0 * (ry(-1) - ry(2)) + 10.0 * (thm - ry(1))) / 60.0
    facey = flux6y - np.sign(rv[:, :ny, :]) * corry
    fqy = rv[:, :ny, :] * facey
    tend = tend - msftx[None] * rdy * (np.roll(fqy, -1, axis=1) - fqy)

    # vertical v3 (WRF advect_scalar): interior faces k=2..nz-2 flux3 with ua=-vel;
    # faces 1 and nz-1 2nd order fnm/fnp; 0 and nz zero.
    vflux = np.zeros((nz + 1, ny, nx))
    q = thm
    for k in range(2, nz - 1):
        vel = rom[k]
        flux4 = (7.0 * (q[k] + q[k - 1]) - (q[k + 1] + q[k - 2])) / 12.0
        corr = ((q[k + 1] - q[k - 2]) - 3.0 * (q[k] - q[k - 1])) / 12.0
        vflux[k] = vel * (flux4 + np.sign(-vel) * corr)
    vflux[1] = rom[1] * (fnm[1] * q[1] + fnp[1] * q[0])
    vflux[nz - 1] = rom[nz - 1] * (fnm[nz - 1] * q[nz - 1] + fnp[nz - 1] * q[nz - 2])
    tend = tend - rdnw[:, None, None] * (vflux[1:] - vflux[:nz])
    tend_np = tend

    # ---- JAX production chain on the same inputs ----
    import jax.numpy as jnp
    from gpuwrf.dynamics.flux_advection import couple_velocities_periodic, advect_scalar_flux

    vel_jax = couple_velocities_periodic(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(mu),
        c1h=jnp.asarray(c1h), c2h=jnp.asarray(c2h), dnw=jnp.asarray(dnw),
        rdx=rdx, rdy=rdy,
        msfuy=jnp.asarray(msfuy), msfvx=jnp.asarray(msfvx),
        msftx=jnp.asarray(msftx), msfux=None, msfvy=None,
    )
    tend_jax = np.asarray(advect_scalar_flux(
        jnp.asarray(thm), vel_jax, mut=jnp.asarray(mu), c1=jnp.asarray(c1h),
        rdx=rdx, rdy=rdy, rdzw=jnp.asarray(rdnw), fzm=jnp.asarray(fnm), fzp=jnp.asarray(fnp),
    ))
    rom_jax = np.asarray(vel_jax.rom)

    mask = np.zeros((ny, nx), dtype=bool)
    mask[10:-10, 10:-10] = True
    mass_h = c1h[:, None, None] * mu[None] + c2h[:, None, None]

    res = {
        "schema": "v014_switzerland_theta_advection_operator_oracle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rom_diff_max": float(np.max(np.abs((rom_jax - rom)[:, mask]))),
        "rom_interior_rms": float(np.sqrt(np.mean(rom[:, mask] ** 2))),
        "actual_cpu_dtheta_dt_k_per_s": [float(np.mean((t37 - d36["T"])[k][mask]) / 3600.0) for k in range(nz)],
    }
    dec_np = tend_np / mass_h
    dec_jax = tend_jax / mass_h
    res["np_minus_jax_mean_K_per_s"] = [float(np.mean((dec_np - dec_jax)[k][mask])) for k in range(nz)]
    res["np_minus_jax_rms_K_per_s"] = [float(np.sqrt(np.mean((dec_np - dec_jax)[k][mask] ** 2))) for k in range(nz)]
    res["jax_adv_mean_K_per_s"] = [float(np.mean(dec_jax[k][mask])) for k in range(nz)]
    res["np_adv_mean_K_per_s"] = [float(np.mean(dec_np[k][mask])) for k in range(nz)]

    OUT.write_text(json.dumps(res, indent=1))
    print("rom diff max (interior):", res["rom_diff_max"], " rom rms:", res["rom_interior_rms"])
    print("k   np-jax_mean   np-jax_rms   jax_adv_mean  np_adv_mean  cpu_actual")
    for k in range(nz):
        print(f"k{k:02d} {res['np_minus_jax_mean_K_per_s'][k]:+12.3e} {res['np_minus_jax_rms_K_per_s'][k]:12.3e} "
              f"{res['jax_adv_mean_K_per_s'][k]:+12.3e} {res['np_adv_mean_K_per_s'][k]:+12.3e} "
              f"{res['actual_cpu_dtheta_dt_k_per_s'][k]:+12.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
