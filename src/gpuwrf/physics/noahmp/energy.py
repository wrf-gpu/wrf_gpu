"""Noah-MP canopy/ground energy balance (Sprint S1, PRIORITY) — THE HFX FIX.

Faithful JAX port of NOAHMP_SFLX::ENERGY (module_sf_noahmplsm.F:1741-2396) and
its flux callees for the scoped option set
(opt_sfc=1, opt_crs=1, opt_btr=1, opt_rad=3, opt_alb=2):

  - THERMOPROP (:2400-2510)  : DF / HCPCT per layer (needed for CGH and the STC
                               update); CSNOW/TDFCND inlined (snow-free branch).
  - RADIATION  (:2684-2806)  : two-stream chain -> SAV/SAG/PARSUN/PARSHA/albedo
                               (split into ``energy_radiation.py``).
  - VEGE_FLUX  (:3578-4170)  : Newton-Raphson canopy (TV) + canopy-air (TAH/EAH)
                               + ground (TG) energy balance, with SFCDIF1 (M-O
                               drag), RAGRB (under-canopy resistance + RB), and
                               STOMATA (opt_crs=1 Ball-Berry) inside the loop.
  - BARE_FLUX  (:4174-4479)  : bare-tile ground (TG) energy balance.
  - FVEG-weighted tile sum (:2285-2325) + TRAD/EMISSI (:2337-2348).
  - soil_thermo (Sprint S2, TSNOSOI) called for the semi-implicit STC update;
    tolerated as a no-op when S2 is not yet merged (the STC update does not
    affect THIS step's HFX/LH/SSOIL/TRAD, which use the old STC(ISNOW+1) as the
    ground BC).

WHY THIS IS THE FIX (proofs/v010_validation/hfx_overflux_root_cause.json):
  WRF HFX = FSH = FVEG*SHG + (1-FVEG)*SHB + SHC, where the vegetated tile's
  sensible heat is driven by (TAH - SFCTMP) through the canopy-air temperature
  TAH (~5.5 K cooler than the radiative skin TSK at midday over dry sparse-veg
  land). The v0.1.0 bulk path used the radiative TSK and over-fluxed by the
  gradient ratio 1.40. Solving the canopy-air energy balance for TAH closes it.

opt_sfc=1: sfclay supplies the INITIAL CM/CH (carried in ``land_state``), but
SFCDIF1 re-derives the canopy/bare-specific CM/CH each Newton iteration from the
evolving sensible-heat flux (WRF semantics: VEGE_FLUX/BARE_FLUX own the in-loop
drag; sfclay's CM/CH seed the inout slot and remain the ocean/lake path's coeffs).

All operations are fully vectorized over the land-column grid (ny, nx); fp64
(x64 enabled at package import). No host transfer; pure functional pytree I/O.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.energy_radiation import TwoStreamParams, radiation_twostream
from gpuwrf.physics.noahmp.types import (
    NoahMPEnergyFluxes,
    NoahMPEtFluxes,
    NoahMPForcing,
    NoahMPPhenology,
    NoahMPRadInputs,
)

# ==================================================================================
# Module physical constants (module_sf_noahmplsm.F:204-220)
# ==================================================================================
GRAV = 9.80616
SB = 5.67e-08
VKC = 0.40
TFRZ = 273.16
HSUB = 2.8440e06
HVAP = 2.5104e06
CWAT = 4.188e06
CICE = 2.094e06
CPAIR = 1004.64
TKICE = 2.2
RAIR = 287.04
RW = 461.269
DENH2O = 1000.0
DENICE = 917.0
MPE = 1.0e-6

# ENERGY locals (:2028-2030)
Z0_BARE = 0.002  # bare-soil (under-canopy) roughness length [m]

# Newton-Raphson iteration counts (WRF VEGE_FLUX/BARE_FLUX)
NITERC = 20
NITERG = 5
NITERB = 5

# STOMATA Ball-Berry CI iterations (:5045)
STOMATA_NITER = 3


class EnergyParams(NamedTuple):
    """Per-column gathered Noah-MP parameters for the energy balance.

    A self-contained bundle (subset of the frozen ``NoahMPParameters`` plus the
    energy-only extras not yet in that tuple — Ball-Berry, EG, SNOW_EMIS,
    RSURF_EXP, CBIOM). All fields broadcast over the land-column grid (ny, nx);
    per-layer soil fields are (NSOIL, ny, nx). Built once per run by the oracle /
    coupler from MPTABLE/SOILPARM. ``nroot`` is a static python int.
    """

    # --- canopy structure / roughness ---
    z0mvt: jnp.ndarray
    hvt: jnp.ndarray
    cwpvt: jnp.ndarray
    dleaf: jnp.ndarray
    z0sno: jnp.ndarray
    cbiom: jnp.ndarray
    # --- soil thermal / moisture (per layer) ---
    smcmax: jnp.ndarray   # (NSOIL, ny, nx)
    smcref: jnp.ndarray   # (NSOIL, ny, nx)
    smcwlt: jnp.ndarray   # (NSOIL, ny, nx)
    psisat: jnp.ndarray   # (NSOIL, ny, nx)
    bexp: jnp.ndarray     # (NSOIL, ny, nx)
    quartz: jnp.ndarray   # (NSOIL, ny, nx)
    csoil: jnp.ndarray
    nroot: int
    eg: jnp.ndarray
    snow_emis: jnp.ndarray
    rsurf_exp: jnp.ndarray
    # --- Ball-Berry STOMATA (opt_crs=1) ---
    bp: jnp.ndarray
    mp: jnp.ndarray
    folnmx: jnp.ndarray
    qe25: jnp.ndarray
    kc25: jnp.ndarray
    ko25: jnp.ndarray
    akc: jnp.ndarray
    ako: jnp.ndarray
    avcmx: jnp.ndarray
    vcmx25: jnp.ndarray
    c3psn: jnp.ndarray


# ----------------------------------------------------------------------------------
# ESAT (module_sf_noahmplsm.F:4952-5001) — saturation vapor pressure + d(es)/dt
# ----------------------------------------------------------------------------------
_A = (6.107799961, 4.436518521e-01, 1.428945805e-02, 2.650648471e-04,
      3.031240396e-06, 2.034080948e-08, 6.136820929e-11)
_B = (6.109177956, 5.034698970e-01, 1.886013408e-02, 4.176223716e-04,
      5.824720280e-06, 4.838803174e-08, 1.838826904e-10)
_C = (4.438099984e-01, 2.857002636e-02, 7.938054040e-04, 1.215215065e-05,
      1.036561403e-07, 3.532421810e-10, -7.090244804e-13)
_D = (5.030305237e-01, 3.773255020e-02, 1.267995369e-03, 2.477563108e-05,
      3.005693132e-07, 2.158542548e-09, 7.131097725e-12)


def _poly(co, t):
    return co[0] + t * (co[1] + t * (co[2] + t * (co[3] + t * (co[4] + t * (co[5] + t * co[6])))))


def esat(t_c):
    """Returns (esw, esi, desw, desi) in Pa / Pa-per-K; ``t_c`` is degrees C."""
    return (100.0 * _poly(_A, t_c), 100.0 * _poly(_B, t_c),
            100.0 * _poly(_C, t_c), 100.0 * _poly(_D, t_c))


def _tdc(t_k):
    return jnp.minimum(50.0, jnp.maximum(-50.0, t_k - TFRZ))


def _es_dest(t_k):
    """Saturation vapor pressure ESTV and d(es)/dt at temperature t_k [K]."""
    t_c = _tdc(t_k)
    esw, esi, desw, desi = esat(t_c)
    water = t_c > 0.0
    return jnp.where(water, esw, esi), jnp.where(water, desw, desi)


# ----------------------------------------------------------------------------------
# THERMOPROP / TDFCND (module_sf_noahmplsm.F:2400-2510, 2573-2680), soil branch.
# Snow-free land columns (FSNO=0, ISNOW=0): ground BC layer = soil layer 1.
# ----------------------------------------------------------------------------------
def thermoprop_soil(land_state: NoahMPLandState, p: EnergyParams):
    """Soil-layer thermal conductivity DF and heat capacity HCPCT, (NSOIL, ny, nx)."""
    smc = land_state.smois
    sh2o = land_state.sh2o
    sice = smc - sh2o
    hcpct = (sh2o * CWAT + (1.0 - p.smcmax) * p.csoil
             + (p.smcmax - smc) * CPAIR + sice * CICE)
    satratio = smc / p.smcmax
    thkw = 0.57
    thko = 2.0
    thkqtz = 7.7
    thks = (thkqtz ** p.quartz) * (thko ** (1.0 - p.quartz))
    xunfroz = jnp.where(smc > 0.0, sh2o / jnp.maximum(smc, MPE), 1.0)
    xu = xunfroz * p.smcmax
    thksat = thks ** (1.0 - p.smcmax) * TKICE ** (p.smcmax - xu) * thkw ** xu
    gammd = (1.0 - p.smcmax) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    frozen = (sh2o + 0.0005) < smc
    ake_unfroz = jnp.where(satratio > 0.1, jnp.log10(jnp.maximum(satratio, MPE)) + 1.0, 0.0)
    ake = jnp.where(frozen, satratio, ake_unfroz)
    df = ake * (thksat - thkdry) + thkdry
    return df, hcpct


# ----------------------------------------------------------------------------------
# SFCDIF1 (module_sf_noahmplsm.F:4583-4743) — M-O drag CM/CH for one iteration.
# (FM,FH,FM2,FH2,MOZ,MOZSGN,FV) carried explicitly across iterations by the caller.
# ----------------------------------------------------------------------------------
def sfcdif1(it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h, ur,
            moz, mozsgn, fm, fh, fm2, fh2, fv_prev):
    """One SFCDIF1 step. Returns (cm, ch, fv, moz, mozsgn, fm, fh, fm2, fh2)."""
    mozold = moz
    tmpcm = jnp.log((zlvl - zpd) / z0m)
    tmpch = jnp.log((zlvl - zpd) / z0h)
    tmpcm2 = jnp.log((2.0 + z0m) / z0m)
    tmpch2 = jnp.log((2.0 + z0h) / z0h)

    if it == 1:
        moz = jnp.zeros_like(sfctmp)
        moz2 = jnp.zeros_like(sfctmp)
    else:
        tvir = (1.0 + 0.61 * qair) * sfctmp
        tmp1 = VKC * (GRAV / tvir) * h / (rhoair * CPAIR)
        tmp1 = jnp.where(jnp.abs(tmp1) <= MPE, MPE, tmp1)
        mol = -1.0 * fv_prev ** 3 / tmp1
        moz = jnp.minimum((zlvl - zpd) / mol, 1.0)
        moz2 = jnp.minimum((2.0 + z0h) / mol, 1.0)

    mozsgn = jnp.where(mozold * moz < 0.0, mozsgn + 1, mozsgn)
    reset = mozsgn >= 2
    moz = jnp.where(reset, 0.0, moz)
    moz2 = jnp.where(reset, 0.0, moz2)

    unst = moz < 0.0
    tmp1u = (1.0 - 16.0 * jnp.minimum(moz, 0.0)) ** 0.25
    tmp2u = jnp.log((1.0 + tmp1u * tmp1u) / 2.0)
    tmp3u = jnp.log((1.0 + tmp1u) / 2.0)
    fmnew_u = 2.0 * tmp3u + tmp2u - 2.0 * jnp.arctan(tmp1u) + 1.5707963
    fhnew_u = 2.0 * tmp2u
    tmp12 = (1.0 - 16.0 * jnp.minimum(moz2, 0.0)) ** 0.25
    tmp22 = jnp.log((1.0 + tmp12 * tmp12) / 2.0)
    tmp32 = jnp.log((1.0 + tmp12) / 2.0)
    fm2new_u = 2.0 * tmp32 + tmp22 - 2.0 * jnp.arctan(tmp12) + 1.5707963
    fh2new_u = 2.0 * tmp22
    fmnew_s = -5.0 * moz
    fm2new_s = -5.0 * moz2
    fmnew = jnp.where(unst, fmnew_u, fmnew_s)
    fhnew = jnp.where(unst, fhnew_u, fmnew_s)
    fm2new = jnp.where(unst, fm2new_u, fm2new_s)
    fh2new = jnp.where(unst, fh2new_u, fm2new_s)

    if it == 1:
        fm, fh, fm2, fh2 = fmnew, fhnew, fm2new, fh2new
    else:
        fm = 0.5 * (fm + fmnew)
        fh = 0.5 * (fh + fhnew)
        fm2 = 0.5 * (fm2 + fm2new)
        fh2 = 0.5 * (fh2 + fh2new)
    fm = jnp.where(reset, 0.0, fm)
    fh = jnp.where(reset, 0.0, fh)
    fm2 = jnp.where(reset, 0.0, fm2)
    fh2 = jnp.where(reset, 0.0, fh2)

    fh = jnp.minimum(fh, 0.9 * tmpch)
    fm = jnp.minimum(fm, 0.9 * tmpcm)
    fh2 = jnp.minimum(fh2, 0.9 * tmpch2)
    fm2 = jnp.minimum(fm2, 0.9 * tmpcm2)

    cmfm = tmpcm - fm
    chfh = tmpch - fh
    cmfm = jnp.where(jnp.abs(cmfm) <= MPE, MPE, cmfm)
    chfh = jnp.where(jnp.abs(chfh) <= MPE, MPE, chfh)
    cm = VKC * VKC / (cmfm * cmfm)
    ch = VKC * VKC / (cmfm * chfh)
    fv = ur * jnp.sqrt(cm)
    return cm, ch, fv, moz, mozsgn, fm, fh, fm2, fh2


# ----------------------------------------------------------------------------------
# RAGRB (module_sf_noahmplsm.F:4483-4579) — under-canopy resistance + leaf RB
# ----------------------------------------------------------------------------------
def ragrb(it, vaie, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc, z0h, fv, cwp, fhg, dleaf):
    """One RAGRB step. Returns (rahg, rawg, rb, fhg)."""
    if it == 1:
        mozg = jnp.zeros_like(tah)
    else:
        tmp1 = VKC * (GRAV / tah) * hg / (rhoair * CPAIR)
        tmp1 = jnp.where(jnp.abs(tmp1) <= MPE, MPE, tmp1)
        molg = -1.0 * fv ** 3 / tmp1
        mozg = jnp.minimum((zpd - z0mg) / molg, 1.0)
    fhgnew = jnp.where(mozg < 0.0, (1.0 - 15.0 * jnp.minimum(mozg, 0.0)) ** (-0.25),
                       1.0 + 4.7 * mozg)
    if it == 1:
        fhg = fhgnew
    else:
        fhg = 0.5 * (fhg + fhgnew)
    cwpc = (cwp * vaie * hcan * fhg) ** 0.5
    tmp1 = jnp.exp(-cwpc * z0hg / hcan)
    tmp2 = jnp.exp(-cwpc * (z0h + zpd) / hcan)
    tmprah2 = hcan * jnp.exp(cwpc) / cwpc * (tmp1 - tmp2)
    kh = jnp.maximum(VKC * fv * (hcan - zpd), MPE)
    rahg = tmprah2 / kh
    rawg = rahg
    tmprb = cwpc * 50.0 / (1.0 - jnp.exp(-cwpc / 2.0))
    rb = tmprb * jnp.sqrt(dleaf / uc)
    rb = jnp.minimum(jnp.maximum(rb, 5.0), 50.0)
    return rahg, rawg, rb, fhg


# ----------------------------------------------------------------------------------
# STOMATA (module_sf_noahmplsm.F:5005-5137) — Ball-Berry, opt_crs=1
# ----------------------------------------------------------------------------------
def stomata(apar, foln, tv, ei, ea, sfctmp, sfcprs, fveg, o2, co2, igs, rb, btran, p: EnergyParams):
    """Ball-Berry leaf stomatal resistance RS [s/m]. Vectorized."""
    apar_scale = apar / jnp.maximum(fveg, 1.0e-6)
    cf = sfcprs / (8.314 * sfctmp) * 1.0e06
    rs0 = 1.0 / p.bp * cf
    fnf = jnp.minimum(foln / jnp.maximum(MPE, p.folnmx), 1.0)
    tc = tv - TFRZ
    ppf = 4.6 * jnp.maximum(apar_scale, 0.0)
    j = ppf * p.qe25
    f1 = lambda ab, bc: ab ** ((bc - 25.0) / 10.0)
    f2 = lambda ab: 1.0 + jnp.exp((-2.2e05 + 710.0 * (ab + 273.16)) / (8.314 * (ab + 273.16)))
    kc = p.kc25 * f1(p.akc, tc)
    ko = p.ko25 * f1(p.ako, tc)
    awc = kc * (1.0 + o2 / ko)
    cp = 0.5 * kc / ko * o2 * 0.21
    vcmx = p.vcmx25 / f2(tc) * fnf * btran * f1(p.avcmx, tc)
    ci = 0.7 * co2 * p.c3psn + 0.4 * co2 * (1.0 - p.c3psn)
    rlb = rb / cf
    cea = jnp.maximum(0.25 * ei * p.c3psn + 0.40 * ei * (1.0 - p.c3psn), jnp.minimum(ea, ei))

    rs = rs0
    for _ in range(STOMATA_NITER):
        wj = jnp.maximum(ci - cp, 0.0) * j / (ci + 2.0 * cp) * p.c3psn + j * (1.0 - p.c3psn)
        wc = jnp.maximum(ci - cp, 0.0) * vcmx / (ci + awc) * p.c3psn + vcmx * (1.0 - p.c3psn)
        we = 0.5 * vcmx * p.c3psn + 4000.0 * vcmx * ci / sfcprs * (1.0 - p.c3psn)
        psn = jnp.minimum(jnp.minimum(wj, wc), we) * igs
        cs = jnp.maximum(co2 - 1.37 * rlb * sfcprs * psn, MPE)
        a = p.mp * psn * sfcprs * cea / (cs * ei) + p.bp
        b = (p.mp * psn * sfcprs / cs + p.bp) * rlb - 1.0
        c = -rlb
        disc = jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 0.0))
        q = jnp.where(b >= 0.0, -0.5 * (b + disc), -0.5 * (b - disc))
        r1 = q / a
        r2 = c / q
        rs = jnp.maximum(r1, r2)
        ci = jnp.maximum(cs - psn * sfcprs * 1.65 * rs, 0.0)
    rs = rs * cf
    rs = jnp.where(apar_scale <= 0.0, rs0, rs)
    return rs


# ==================================================================================
# Frozen-signature radiation wrapper
# ==================================================================================
def noahmp_radiation_twostream(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    phen: NoahMPPhenology,
    *,
    rad_params: TwoStreamParams,
    dt: float = 1.0,
) -> NoahMPRadInputs:
    """Two-stream canopy radiation (split into ``energy_radiation``). Returns the
    ``NoahMPRadInputs`` only, so S1 can oracle-test the radiation sub-step alone.

    Use ``energy_radiation.radiation_twostream`` directly when the fsun/laisun
    extras are needed (they feed the flux solve via ``rad_extras``).
    """
    rad, _ = radiation_twostream(land_state, forcing, static, phen, rad_params, dt)
    return rad


# ----------------------------------------------------------------------------------
# column-thickness helpers (ZSNSO interface depths <0)
# ----------------------------------------------------------------------------------
def _soil_dz(land_state):
    """Soil-layer thicknesses (NSOIL, ny, nx) from ZSNSO. Soil occupies
    zsnso[NSNOW:]; for ISNOW=0 the soil-surface interface is 0."""
    z = land_state.zsnso
    dz = jnp.zeros((NSOIL,) + land_state.tg.shape)
    dz = dz.at[0].set(-z[NSNOW])
    for j in range(1, NSOIL):
        dz = dz.at[j].set(z[NSNOW + j - 1] - z[NSNOW + j])
    return dz


def _dz1(land_state):
    """DZSNSO(ISNOW+1) for ISNOW=0 = soil layer-1 thickness = -(zsnso[NSNOW])."""
    return -land_state.zsnso[NSNOW]


def _full_dz(land_state):
    z = land_state.zsnso
    dz = jnp.zeros((NSNOW + NSOIL,) + land_state.tg.shape)
    dz = dz.at[0].set(-z[0])
    for j in range(1, NSNOW + NSOIL):
        dz = dz.at[j].set(z[j - 1] - z[j])
    return dz


# ==================================================================================
# VEGE_FLUX / BARE_FLUX tiles
# ==================================================================================
def _vege_flux(land_state, forcing, radd, phen, p, df1, zlvl, zpd, z0m, z0mg,
               hcan, ur, emv, emg, gammav, gammag, rsurf, rhsur, latheav,
               dt, o2air, co2air, foln, btran):
    """VEGE_FLUX (:3578-4170): TV / TAH-EAH / TG Newton-Raphson, vectorized."""
    eair = forcing.qair * forcing.sfcprs / (0.622 + 0.378 * forcing.qair)
    rhoair = (forcing.sfcprs - 0.378 * eair) / (RAIR * forcing.sfctmp)
    sfctmp = forcing.sfctmp
    sfcprs = forcing.sfcprs
    psfc = forcing.psfc
    qair = forcing.qair
    fveg = phen.fveg
    vai = phen.elai + phen.esai
    vaie = jnp.minimum(6.0, vai)
    laisune = jnp.minimum(6.0, radd["laisun"])
    laishae = jnp.minimum(6.0, radd["laisha"])
    fwet = land_state.fwet
    canliq = land_state.canliq
    canice = land_state.canice
    sav = radd["sav"]
    sag = radd["sag"]
    parsun = radd["parsun"]
    parsha = radd["parsha"]
    igs = phen.igs

    tg = land_state.tg
    tv = land_state.tv
    eah = land_state.eah
    tah = land_state.tah
    cm = land_state.cm
    ch = land_state.ch
    stc1 = land_state.tslb[0]
    lwdn = forcing.lwdn

    estg, _ = _es_dest(tg)
    qsfc = 0.622 * eair / (psfc - 0.378 * eair)
    uc = ur * jnp.log((hcan - zpd + z0m) / z0m) / jnp.log(zlvl / z0m)

    air = -emv * (1.0 + (1.0 - emv) * (1.0 - emg)) * lwdn - emv * emg * SB * tg ** 4
    cir = (2.0 - emv * (1.0 - emg)) * emv * SB

    moz = jnp.zeros_like(tah)
    mozsgn = jnp.zeros_like(tah)
    fm = jnp.zeros_like(tah)
    fh = jnp.zeros_like(tah)
    fm2 = jnp.zeros_like(tah)
    fh2 = jnp.zeros_like(tah)
    fhg = jnp.zeros_like(tah)
    fv = 0.1 + jnp.zeros_like(tah)
    h = jnp.zeros_like(tah)
    hg = jnp.zeros_like(tah)
    rssun = jnp.zeros_like(tah)
    rssha = jnp.zeros_like(tah)
    irc = shc = evc = tr = jnp.zeros_like(tah)
    cah = jnp.zeros_like(tah)
    rahg = rawg = rb = jnp.zeros_like(tah)

    z0h = z0m
    z0hg = z0mg

    for it in range(1, NITERC + 1):
        cm, ch, fv, moz, mozsgn, fm, fh, fm2, fh2 = sfcdif1(
            it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h, ur,
            moz, mozsgn, fm, fh, fm2, fh2, fv,
        )
        rahc = jnp.maximum(1.0, 1.0 / (ch * ur))
        rahg, rawg, rb, fhg = ragrb(
            it, vaie, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc, z0h, fv, p.cwpvt, fhg, p.dleaf
        )
        estv, destv = _es_dest(tv)
        if it == 1:
            rssun = stomata(parsun, foln, tv, estv, eah, sfctmp, sfcprs, fveg,
                            o2air, co2air, igs, rb, btran, p)
            rssha = stomata(parsha, foln, tv, estv, eah, sfctmp, sfcprs, fveg,
                            o2air, co2air, igs, rb, btran, p)

        cah = 1.0 / rahc
        cvh = 2.0 * vaie / rb
        cgh = 1.0 / rahg
        cond = cah + cvh + cgh
        ata = (sfctmp * cah + tg * cgh) / cond
        bta = cvh / cond
        csh = (1.0 - bta) * rhoair * CPAIR * cvh

        caw = 1.0 / rahc
        cew = fwet * vaie / rb
        ctw = (1.0 - fwet) * (laisune / (rb + rssun) + laishae / (rb + rssha))
        cgw = 1.0 / (rawg + rsurf)
        cond_w = caw + cew + ctw + cgw
        aea = (eair * caw + estg * cgw) / cond_w
        bea = (cew + ctw) / cond_w
        cev = (1.0 - bea) * cew * rhoair * CPAIR / gammav
        ctr = (1.0 - bea) * ctw * rhoair * CPAIR / gammav

        tah = ata + bta * tv
        eah = aea + bea * estv

        irc = fveg * (air + cir * tv ** 4)
        shc = fveg * rhoair * CPAIR * cvh * (tv - tah)
        evc = fveg * rhoair * CPAIR * cew * (estv - eah) / gammav
        tr = fveg * rhoair * CPAIR * ctw * (estv - eah) / gammav
        evc = jnp.where(tv > TFRZ,
                        jnp.minimum(canliq * latheav / dt, evc),
                        jnp.minimum(canice * latheav / dt, evc))
        hcv = p.cbiom * vaie * CWAT + canliq * CWAT / DENH2O + canice * CICE / DENICE

        b = sav - irc - shc - evc - tr
        a = fveg * (4.0 * cir * tv ** 3 + csh + (cev + ctr) * destv + hcv / dt)
        dtv = b / a

        irc = irc + fveg * 4.0 * cir * tv ** 3 * dtv
        shc = shc + fveg * csh * dtv
        evc = evc + fveg * cev * destv * dtv
        tr = tr + fveg * ctr * destv * dtv
        tv = tv + dtv

        h = rhoair * CPAIR * (tah - sfctmp) / rahc
        hg = rhoair * CPAIR * (tg - tah) / rahg
        qsfc = (0.622 * eah) / (sfcprs - 0.378 * eah)

    # under-canopy ground loop (loop2)
    air_g = -emg * (1.0 - emv) * lwdn - emg * emv * SB * tv ** 4
    cir_g = emg * SB
    csh_g = rhoair * CPAIR / rahg
    cev_g = rhoair * CPAIR / (gammag_of(land_state, forcing) * (rawg + rsurf))
    cgh_g = 2.0 * df1 / _dz1(land_state)
    irg = shg = evg = gh = jnp.zeros_like(tah)
    for _ in range(NITERG):
        estg, destg = _es_dest(tg)
        irg = cir_g * tg ** 4 + air_g
        shg = csh_g * (tg - tah)
        evg = cev_g * (estg * rhsur - eah)
        gh = cgh_g * (tg - stc1)
        b = sag - irg - shg - evg - gh
        a = 4.0 * cir_g * tg ** 3 + csh_g + cev_g * destg + cgh_g
        dtg = b / a
        irg = irg + 4.0 * cir_g * tg ** 3 * dtg
        shg = shg + csh_g * dtg
        evg = evg + cev_g * destg * dtg
        gh = gh + cgh_g * dtg
        tg = tg + dtg

    return {
        "irc": irc, "shc": shc, "evc": evc, "tr": tr,
        "irg": irg, "shg": shg, "evg": evg, "ghv": gh,
        "tv": tv, "tg": tg, "tah": tah, "eah": eah,
        "cm": cm, "chv": cah, "qsfc": qsfc,
    }


def gammag_of(land_state, forcing):
    latheag = jnp.where(land_state.tg > TFRZ, HVAP, HSUB)
    return CPAIR * forcing.sfcprs / (0.622 * latheag)


def _bare_flux(land_state, forcing, radd, p, df1, zlvl, zpdg, z0mg, ur, emg,
               gammag, rsurf, rhsur):
    """BARE_FLUX (:4174-4479): bare-tile ground (TG) Newton-Raphson, vectorized."""
    eair = forcing.qair * forcing.sfcprs / (0.622 + 0.378 * forcing.qair)
    rhoair = (forcing.sfcprs - 0.378 * eair) / (RAIR * forcing.sfctmp)
    sfctmp = forcing.sfctmp
    qair = forcing.qair
    sag = radd["sag"]
    lwdn = forcing.lwdn
    tgb = land_state.tg
    stc1 = land_state.tslb[0]

    cir = emg * SB
    cgh = 2.0 * df1 / _dz1(land_state)

    moz = jnp.zeros_like(tgb)
    mozsgn = jnp.zeros_like(tgb)
    fm = jnp.zeros_like(tgb)
    fh = jnp.zeros_like(tgb)
    fm2 = jnp.zeros_like(tgb)
    fh2 = jnp.zeros_like(tgb)
    fv = 0.1 + jnp.zeros_like(tgb)
    h = jnp.zeros_like(tgb)
    z0m = z0mg
    z0h = z0m
    irb = shb = evb = ghb = jnp.zeros_like(tgb)
    ehb = jnp.zeros_like(tgb)
    cm = land_state.cm

    for it in range(1, NITERB + 1):
        cm, ch, fv, moz, mozsgn, fm, fh, fm2, fh2 = sfcdif1(
            it, sfctmp, rhoair, h, qair, zlvl, zpdg, z0m, z0h, ur,
            moz, mozsgn, fm, fh, fm2, fh2, fv,
        )
        rahb = jnp.maximum(1.0, 1.0 / (ch * ur))
        rawb = rahb
        ehb = 1.0 / rahb
        estg, destg = _es_dest(tgb)
        csh = rhoair * CPAIR / rahb
        cev = rhoair * CPAIR / gammag / (rsurf + rawb)
        irb = cir * tgb ** 4 - emg * lwdn
        shb = csh * (tgb - sfctmp)
        evb = cev * (estg * rhsur - eair)
        ghb = cgh * (tgb - stc1)
        b = sag - irb - shb - evb - ghb
        a = 4.0 * cir * tgb ** 3 + csh + cev * destg + cgh
        dtg = b / a
        irb = irb + 4.0 * cir * tgb ** 3 * dtg
        shb = shb + csh * dtg
        evb = evb + cev * destg * dtg
        ghb = ghb + cgh * dtg
        tgb = tgb + dtg
        h = csh * (tgb - sfctmp)

    return {"irb": irb, "shb": shb, "evb": evb, "ghb": ghb, "tgb": tgb,
            "cm": cm, "ch": ehb}


# ==================================================================================
# Top-level energy balance — THE HFX FIX
# ==================================================================================
def noahmp_energy_canopy(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    rad: NoahMPRadInputs,
    dt: float,
    *,
    phen: NoahMPPhenology,
    params: EnergyParams,
    rad_extras: dict,
    o2air: jnp.ndarray | float = 0.209 * 101325.0,
    co2air: jnp.ndarray | float = 395.0e-06 * 101325.0,
    foln: jnp.ndarray | float = 1.0,
) -> tuple[NoahMPLandState, NoahMPEnergyFluxes, NoahMPEtFluxes]:
    """Canopy/ground surface-energy balance — THE HFX FIX.

    ENERGY (:1741-2396) for opt_sfc=1/opt_crs=1/opt_btr=1/opt_rad=3/opt_alb=2.
    The two-stream ``rad`` is from ``noahmp_radiation_twostream``; ``rad_extras``
    (the fsun/laisun/laisha/tauss/albold bundle radiation also produces) is the
    second element of ``energy_radiation.radiation_twostream`` and is required.

    Returns ``(land_state', energy_fluxes, et_fluxes)``.
    """
    radd = {
        "sav": rad.sav, "sag": rad.sag, "parsun": rad.parsun, "parsha": rad.parsha,
        "laisun": rad_extras["laisun"], "laisha": rad_extras["laisha"],
    }
    p = params
    fveg = phen.fveg
    elai = phen.elai
    esai = phen.esai
    vai = elai + esai
    veg = vai > 0.0
    fsno = rad.fsno

    # roughness / displacement (ENERGY :2075-2109)
    z0mg = Z0_BARE * (1.0 - fsno) + p.z0sno * fsno
    zpdg = land_state.snowh
    z0m = jnp.where(veg, p.z0mvt, z0mg)
    zpd = jnp.where(veg, jnp.maximum(0.65 * p.hvt, land_state.snowh), zpdg)
    zref = forcing.zlvl
    zlvl = jnp.maximum(zpd, p.hvt) + zref
    zlvl = jnp.where(zpdg >= zlvl, zpdg + zref, zlvl)
    hcan = p.hvt
    ur = jnp.maximum(jnp.sqrt(forcing.uu ** 2 + forcing.vv ** 2), 1.0)

    # THERMOPROP DF/HCPCT (soil branch); df1 = DF(1) for ISNOW=0
    df, hcpct = thermoprop_soil(land_state, p)
    df1 = df[0]

    # emissivities (ENERGY :2139-2144)
    emv = 1.0 - jnp.exp(-(elai + esai) / 1.0)
    emg = p.eg * (1.0 - fsno) + p.snow_emis * fsno

    # BTRAN (opt_btr=1) over rooting layers (ENERGY :2148-2171)
    sh2o = land_state.sh2o
    zsoil = static.zsoil
    dz_soil = _soil_dz(land_state)
    nroot = p.nroot
    btran = jnp.zeros_like(fveg)
    btrani = jnp.zeros((NSOIL,) + fveg.shape)
    for iz in range(nroot):
        gx = (sh2o[iz] - p.smcwlt[iz]) / (p.smcref[iz] - p.smcwlt[iz])
        gx = jnp.minimum(1.0, jnp.maximum(0.0, gx))
        bi = jnp.maximum(MPE, dz_soil[iz] / (-zsoil[nroot - 1]) * gx)
        btrani = btrani.at[iz].set(bi)
        btran = btran + bi
    btran = jnp.maximum(MPE, btran)
    for iz in range(nroot):
        btrani = btrani.at[iz].set(btrani[iz] / btran)

    # ground surface resistance + RHSUR (ENERGY :2173-2201, OPT_RSF=1 SZ09)
    sh2o1 = sh2o[0]
    smcmax1 = p.smcmax[0]
    bexp1 = p.bexp[0]
    psisat1 = p.psisat[0]
    l_rsurf = (-zsoil[0]) * (jnp.exp((1.0 - jnp.minimum(1.0, sh2o1 / smcmax1)) ** p.rsurf_exp) - 1.0) / (2.71828 - 1.0)
    d_rsurf = 2.2e-5 * smcmax1 * smcmax1 * (1.0 - p.smcwlt[0] / smcmax1) ** (2.0 + 3.0 / bexp1)
    rsurf = l_rsurf / d_rsurf
    rsurf = jnp.where((sh2o1 < 0.01) & (land_state.snowh == 0.0), 1.0e6, rsurf)
    psi = -psisat1 * (jnp.maximum(0.01, sh2o1) / smcmax1) ** (-bexp1)
    rhsur = fsno + (1.0 - fsno) * jnp.exp(psi * GRAV / (RW * land_state.tg))

    # psychrometric constants (ENERGY :2210-2226)
    latheav = jnp.where(land_state.tv > TFRZ, HVAP, HSUB)
    gammav = CPAIR * forcing.sfcprs / (0.622 * latheav)
    latheag = jnp.where(land_state.tg > TFRZ, HVAP, HSUB)
    gammag = CPAIR * forcing.sfcprs / (0.622 * latheag)

    vf = _vege_flux(land_state, forcing, radd, phen, p, df1, zlvl, zpd, z0m,
                    z0mg, hcan, ur, emv, emg, gammav, gammag, rsurf, rhsur,
                    latheav, dt, o2air, co2air, foln, btran)
    bf = _bare_flux(land_state, forcing, radd, p, df1, zlvl, zpdg, z0mg, ur,
                    emg, gammag, rsurf, rhsur)

    # FVEG-weighted tile sum (ENERGY :2285-2325)
    use_veg = veg & (fveg > 0.0)
    fira = jnp.where(use_veg, fveg * vf["irg"] + (1.0 - fveg) * bf["irb"] + vf["irc"], bf["irb"])
    fsh = jnp.where(use_veg, fveg * vf["shg"] + (1.0 - fveg) * bf["shb"] + vf["shc"], bf["shb"])
    fgev = jnp.where(use_veg, fveg * vf["evg"] + (1.0 - fveg) * bf["evb"], bf["evb"])
    ssoil = jnp.where(use_veg, fveg * vf["ghv"] + (1.0 - fveg) * bf["ghb"], bf["ghb"])
    fcev = jnp.where(use_veg, vf["evc"], 0.0)
    fctr = jnp.where(use_veg, vf["tr"], 0.0)
    tg = jnp.where(use_veg, fveg * vf["tg"] + (1.0 - fveg) * bf["tgb"], bf["tgb"])
    tv = jnp.where(use_veg, vf["tv"], land_state.tv)
    tah = jnp.where(use_veg, vf["tah"], land_state.tah)
    eah = jnp.where(use_veg, vf["eah"], land_state.eah)
    chv = jnp.where(use_veg, vf["chv"], bf["ch"])
    chb = bf["ch"]
    ch = jnp.where(use_veg, fveg * vf["chv"] + (1.0 - fveg) * bf["ch"], bf["ch"])
    cm = jnp.where(use_veg, fveg * vf["cm"] + (1.0 - fveg) * bf["cm"], bf["cm"])
    qsfc = jnp.where(use_veg, vf["qsfc"], land_state.qsfc)
    z0wrf = jnp.where(use_veg, z0m, z0mg)

    # net emissivity + TRAD (ENERGY :2337-2348)
    emissi = fveg * (emg * (1 - emv) + emv + emv * (1 - emv) * (1 - emg)) + (1 - fveg) * emg
    emissi = jnp.where(use_veg, emissi, emg)
    fire = forcing.lwdn + fira
    trad = ((fire - (1.0 - emissi) * forcing.lwdn) / (emissi * SB)) ** 0.25

    # ET partition (NOAHMP_SFLX :982-984, 1061; W/m2 -> kg/m2/s)
    qvap = jnp.maximum(fgev / latheag, 0.0)
    qdew = jnp.abs(jnp.minimum(fgev / latheag, 0.0))
    edir = qvap - qdew
    ecan = fcev / latheav
    etran = fctr / latheav
    qfx = ecan + edir + etran
    lh = fcev + fgev + fctr  # noqa: F841  (LH mapping; coupler reads ef fields)

    # semi-implicit STC update (TSNOSOI / Sprint S2). Fluxes above used the OLD
    # STC(1) as the ground BC, so this does not change this step's HFX/LH/SSOIL/
    # TRAD. Tolerate S2 unmerged (stub raises NotImplementedError).
    tslb_new = land_state.tslb
    try:
        from gpuwrf.physics.noahmp.soil_thermo import noahmp_soil_thermo

        stc = jnp.concatenate([land_state.tsno, land_state.tslb], axis=0)
        dzsnso = _full_dz(land_state)
        df_full = jnp.concatenate([jnp.zeros((NSNOW,) + fveg.shape), df], axis=0)
        hcpct_full = jnp.concatenate([jnp.zeros((NSNOW,) + fveg.shape), hcpct], axis=0)
        stc_new = noahmp_soil_thermo(
            stc, df_full, hcpct_full, ssoil, static.tbot, land_state.zsnso,
            dzsnso, land_state.isnow, dt,
        )
        tslb_new = stc_new[NSNOW:]
    except NotImplementedError:
        pass

    ls = land_state.replace(
        tv=tv, tg=tg, tah=tah, eah=eah, tslb=tslb_new,
        t_skin=trad, emiss=emissi, albedo=rad.albedo, znt=z0wrf,
        qsfc=qsfc, cm=cm, ch=ch,
        tauss=rad_extras["tauss"], albold=rad_extras["albold"],
        lai=phen.lai, sai=phen.sai,
    )
    ef = NoahMPEnergyFluxes(
        fsh=fsh, fcev=fcev, fgev=fgev, fctr=fctr, ssoil=ssoil, fira=fira,
        trad=trad, emissi=emissi, z0wrf=z0wrf, chv=chv, chb=chb,
    )
    et = NoahMPEtFluxes(
        ecan=ecan, etran=etran, edir=edir, qseva=qvap, btrani=btrani,
        qsnow=forcing.prcpsnow, qmelt=jnp.zeros_like(fsh),
        imelt=jnp.zeros((NSNOW + NSOIL,) + fveg.shape, dtype=jnp.int32),
    )
    return ls, ef, et


__all__ = ["noahmp_radiation_twostream", "noahmp_energy_canopy", "EnergyParams"]
