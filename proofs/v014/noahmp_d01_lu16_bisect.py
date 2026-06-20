"""Stage-level bisection of the step-1 sand-cell smois NaN (d01 LU16 blocker).

Replicates noah_mp_step stage by stage (eager, CPU fp64) on the real
wrfinput_d01 warm-start and reports the FIRST stage/array that goes nonfinite
at the ISLTYP=1 cells, then drills into that stage's internals.
"""

from __future__ import annotations

import sys

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from netCDF4 import Dataset

sys.path.insert(0, "src")

from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.physics.noahmp.types import NoahMPForcing

RUN_DIR = "<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z"
DT = 18.0

land, static, meta = build_noahmp_land_state(RUN_DIR, "d01")
w = Dataset(f"{RUN_DIR}/wrfinput_d01")


def s2(name):
    return jnp.asarray(np.asarray(w.variables[name][0], dtype=np.float64))


p0 = jnp.asarray(np.asarray(w.variables["P"][0][0], dtype=np.float64)
                 + np.asarray(w.variables["PB"][0][0], dtype=np.float64))
th_pert = jnp.asarray(np.asarray(w.variables["T"][0][0], dtype=np.float64))
qv = jnp.maximum(jnp.asarray(np.asarray(w.variables["QVAPOR"][0][0], dtype=np.float64)), 0.0)
sfctmp = (th_pert + 300.0) * (p0 / 1.0e5) ** (287.0 / 1004.5)
u = jnp.asarray(0.5 * (np.asarray(w.variables["U"][0][0][:, :-1], dtype=np.float64)
                       + np.asarray(w.variables["U"][0][0][:, 1:], dtype=np.float64)))
v = jnp.asarray(0.5 * (np.asarray(w.variables["V"][0][0][:-1, :], dtype=np.float64)
                       + np.asarray(w.variables["V"][0][0][1:, :], dtype=np.float64)))
psfc = s2("PSFC") if "PSFC" in w.variables else p0
glw = s2("GLW") if "GLW" in w.variables else jnp.full_like(p0, 350.0)
swd = s2("SWDOWN") if "SWDOWN" in w.variables else jnp.zeros_like(p0)
cosz = jnp.where(swd > 0.0, 0.2, 0.05)
ph = np.asarray(w.variables["PH"][0], dtype=np.float64)
phb = np.asarray(w.variables["PHB"][0], dtype=np.float64)
zif = (ph + phb) / 9.81
zlvl = jnp.asarray(0.5 * (zif[1] - zif[0]))
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

energy_params, rad_params, _ = build_noahmp_params(static)

isl = np.asarray(static.isltyp)
lm = np.asarray(static.landmask)
sand = (isl == 1) & (lm == 1)
ys, xs = np.where(sand)
cy, cx = int(ys[0]), int(xs[0])
print(f"sand cells {sand.sum()}; probe cell ({cy},{cx})")


def chk(name, arr, mask=sand):
    a = np.asarray(arr)
    if a.ndim == 2:
        bad = ~np.isfinite(a) & mask
    elif a.ndim == 3:
        bad = ~np.isfinite(a) & mask[None]
    else:
        bad = ~np.isfinite(a)
    n = int(bad.sum())
    flag = " <-- NONFINITE" if n else ""
    print(f"  {name:28s} nonfinite@sand={n}{flag}  probe={a[..., cy, cx] if a.ndim>=2 else a}")
    return n


# ---- replicate noah_mp_step stages eagerly ----
from gpuwrf.physics.noahmp.noahmp_driver import (
    _dzsnso_from_zsnso,
    _gather_vec,
)
from gpuwrf.physics.noahmp.phenology import ISURBAN_MODIS, noahmp_phenology_table
from gpuwrf.physics.noahmp.precip_heat import noahmp_precip_heat
from gpuwrf.physics.noahmp.energy import noahmp_energy_canopy, thermoprop_full
from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream
from gpuwrf.physics.noahmp.soil_thermo import noahmp_phasechange
from gpuwrf.physics.noahmp.water_hydro import noahmp_water_hydro
from gpuwrf.contracts.noahmp_state import NSNOW

print("== stage 1: phenology ==")
phen = noahmp_phenology_table(land, forcing, static)
chk("phen.lai", phen.lai); chk("phen.fveg", phen.fveg)
land = land.replace(lai=phen.lai, sai=phen.sai)

print("== stage 2: precip_heat ==")
ch2op = _gather_vec(getattr(static.parameters, "ch2op"), jnp.asarray(static.ivgtyp, jnp.int32))
is_lake = jnp.asarray(static.lakemask) > 0.5 if static.lakemask is not None else None
precip, canliq_new, canice_new = noahmp_precip_heat(land, forcing, phen, ch2op, DT, is_lake=is_lake)
chk("precip.qrain", precip.qrain); chk("precip.qsnow", precip.qsnow)
land = land.replace(fwet=precip.fwet, canliq=canliq_new, canice=canice_new)
forcing_e = forcing._replace(pahv=precip.pahv, pahg=precip.pahg, pahb=precip.pahb)

print("== stage 3: radiation + energy ==")
rad, rad_extras = radiation_twostream(land, forcing_e, static, phen, rad_params, DT)
chk("rad.sag", rad.sag)
co2 = 395.0e-06 * forcing.sfcprs
o2 = 0.209 * forcing.sfcprs
foln_v = jnp.ones_like(jnp.asarray(land.tv))
land, ef, et = noahmp_energy_canopy(
    land, forcing_e, static, rad, DT,
    phen=phen, params=energy_params, rad_extras=rad_extras,
    o2air=o2, co2air=co2, foln=foln_v,
    pahv_kw=precip.pahv, pahg_kw=precip.pahg, pahb_kw=precip.pahb,
    isurban=int(getattr(static.parameters, "isurban", ISURBAN_MODIS)),
)
for f in ["fsh", "fgev", "fcev", "fctr", "ssoil", "trad"]:
    chk(f"ef.{f}", getattr(ef, f))
for f in ["ecan", "etran", "edir", "qseva", "btrani", "qsnow", "qmelt"]:
    chk(f"et.{f}", getattr(et, f))
chk("land.tg", land.tg); chk("land.tslb", land.tslb); chk("land.smois", land.smois)
chk("land.sh2o", land.sh2o)

print("== stage 4: phasechange ==")
urban = (jnp.asarray(static.ivgtyp, jnp.int32) == int(getattr(static.parameters, "isurban", ISURBAN_MODIS)))
df_full, hcpct_full, _a, _b, _c = thermoprop_full(land, energy_params, urban)
chk("hcpct_full", hcpct_full)
dzsnso = _dzsnso_from_zsnso(land.zsnso)
stc_full = jnp.concatenate([land.tsno, land.tslb], axis=0)
(stc_pc, snice_pc, snliq_pc, smc_pc, sh2o_pc, sneqv_pc, snowh_pc,
 qmelt, imelt, _pond) = noahmp_phasechange(
    stc_full, land.snice, land.snliq, land.smois, land.sh2o, land.sneqv,
    land.snowh, hcpct_full, dzsnso, land.isnow, DT,
    smcmax=energy_params.smcmax, psisat=energy_params.psisat, bexp=energy_params.bexp)
chk("pc.smc", smc_pc); chk("pc.sh2o", sh2o_pc); chk("pc.stc", stc_pc)
chk("pc.qmelt", qmelt)
land = land.replace(tsno=stc_pc[:NSNOW], tslb=stc_pc[NSNOW:], snice=snice_pc,
                    snliq=snliq_pc, smois=smc_pc, sh2o=sh2o_pc, sneqv=sneqv_pc, snowh=snowh_pc)

print("== stage 5: water ==")
et_w = et._replace(qsnow=precip.qsnow, qmelt=qmelt, imelt=imelt)
forcing_w = forcing._replace(prcpnonc=precip.qrain, prcpconv=jnp.zeros_like(precip.qrain),
                             prcpsnow=precip.qsnow)
land_w = noahmp_water_hydro(land, forcing_w, static, et_w, DT)
chk("water.smois", land_w.smois); chk("water.sh2o", land_w.sh2o)
chk("water.canliq", land_w.canliq)

print("done")
