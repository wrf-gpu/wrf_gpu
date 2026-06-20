"""CPU repro for the v0.14 d01 LU16/ISLTYP=1 nonfinite preflight blocker.

Drives the production noah_mp_step (the same kernel the nested pipeline scans)
from the REAL Canary L2 wrfinput_d01 land warm-start with frozen
wrfinput-derived forcing, on CPU fp64, and reports the first step/field/cell
that goes nonfinite. All 51 nonfinite cells in the failing GPU preflight are
exactly the ISLTYP=1 (sand) land cells; this repro isolates whether the
Noah-MP kernel itself produces them without the dycore/radiation in the loop.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from netCDF4 import Dataset

sys.path.insert(0, "src")

from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.physics.noahmp.types import NoahMPForcing

RUN_DIR = "<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z"
DT = 18.0
NSTEPS = 200


def main() -> None:
    land, static, meta = build_noahmp_land_state(RUN_DIR, "d01")

    w = Dataset(f"{RUN_DIR}/wrfinput_d01")

    def s2(name):
        return jnp.asarray(np.asarray(w.variables[name][0], dtype=np.float64))

    # lowest-level forcing from wrfinput (frozen in time; enough for the kernel
    # repro -- the failing cells diverge from internal land-state evolution).
    p0 = jnp.asarray(np.asarray(w.variables["P"][0][0], dtype=np.float64)
                     + np.asarray(w.variables["PB"][0][0], dtype=np.float64))
    th_pert = jnp.asarray(np.asarray(w.variables["T"][0][0], dtype=np.float64))
    qv = jnp.maximum(jnp.asarray(np.asarray(w.variables["QVAPOR"][0][0], dtype=np.float64)), 0.0)
    theta_dry = th_pert + 300.0
    sfctmp = theta_dry * (p0 / 1.0e5) ** (287.0 / 1004.5)
    u = jnp.asarray(0.5 * (np.asarray(w.variables["U"][0][0][:, :-1], dtype=np.float64)
                           + np.asarray(w.variables["U"][0][0][:, 1:], dtype=np.float64)))
    v = jnp.asarray(0.5 * (np.asarray(w.variables["V"][0][0][:-1, :], dtype=np.float64)
                           + np.asarray(w.variables["V"][0][0][1:, :], dtype=np.float64)))
    psfc = s2("PSFC") if "PSFC" in w.variables else p0
    glw = s2("GLW") if "GLW" in w.variables else jnp.full_like(p0, 350.0)
    swd = s2("SWDOWN") if "SWDOWN" in w.variables else jnp.zeros_like(p0)
    # 18z May 1 Canary: sun low; use a small positive cosz so the rad branch runs.
    cosz = jnp.where(swd > 0.0, 0.2, 0.05)

    # lowest half-level height
    ph = np.asarray(w.variables["PH"][0], dtype=np.float64)
    phb = np.asarray(w.variables["PHB"][0], dtype=np.float64)
    z_if = (ph + phb) / 9.81
    zlvl = jnp.asarray(0.5 * (z_if[1] - z_if[0]))

    shape = np.asarray(sfctmp).shape
    forcing = NoahMPForcing(
        sfctmp=sfctmp, sfcprs=p0, psfc=psfc, uu=u, vv=v, qair=qv,
        qc=jnp.zeros(shape), soldn=jnp.maximum(swd, 0.0), lwdn=glw,
        prcpconv=jnp.zeros(shape), prcpnonc=jnp.zeros(shape),
        prcpsnow=jnp.zeros(shape), prcpgrpl=jnp.zeros(shape),
        prcphail=jnp.zeros(shape),
        cosz=cosz, zlvl=zlvl,
        julian=jnp.asarray(120.75), yearlen=jnp.asarray(365.0),
    )

    energy_params, rad_params, _nroot = build_noahmp_params(static)

    isl = np.asarray(static.isltyp)
    lm = np.asarray(static.landmask)
    sand = (isl == 1) & (lm == 1)
    print(f"grid {shape}; land cells {int(lm.sum())}; sand land cells {int(sand.sum())}")

    step = jax.jit(
        lambda ls: noah_mp_step(ls, forcing, static, DT,
                                energy_params=energy_params, rad_params=rad_params)
    )

    watch = ["tsk", "hfx", "lh", "t2"]
    first_bad = None
    for it in range(1, NSTEPS + 1):
        land, fx = step(land)
        bad_report = {}
        for f in watch:
            a = np.asarray(getattr(fx, f))
            nf = ~np.isfinite(a) & (lm == 1)
            if nf.any():
                ys, xs = np.where(nf)
                bad_report[f] = {
                    "n": int(nf.sum()),
                    "n_sand": int((nf & sand).sum()),
                    "first_cells": [[int(y), int(x)] for y, x in zip(ys[:5], xs[:5])],
                }
        # also watch the prognostic carry
        for f in ["tg", "tv", "tslb", "smois", "sneqv"]:
            a = np.asarray(getattr(land, f))
            m = lm == 1
            nf = ~np.isfinite(a) & (m if a.ndim == 2 else m[None])
            if nf.any():
                bad_report[f"carry:{f}"] = {"n": int(nf.sum())}
        if bad_report:
            first_bad = {"step": it, "t_s": it * DT, "fields": bad_report}
            print(json.dumps(first_bad, indent=1))
            break
        if it % 25 == 0:
            tg = np.asarray(land.tg)
            print(f"step {it:4d}: tg sand min/max {tg[sand].min():.2f}/{tg[sand].max():.2f} "
                  f"hfx sand max {np.asarray(fx.hfx)[sand].max():.1f}")

    if first_bad is None:
        print("NO nonfinite within", NSTEPS, "steps -- kernel-only repro insufficient")
    else:
        ys, xs = first_bad["fields"][list(first_bad["fields"])[0]].get("first_cells", [[None, None]])[0], None


if __name__ == "__main__":
    main()
