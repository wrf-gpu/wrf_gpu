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

opt_sfc=1 CH/CM SEMANTICS (authoritative; do NOT mis-wire in S6): sfclay only
SEEDS the inout CM/CH slot (carried in ``land_state``). VEGE_FLUX/BARE_FLUX call
SFCDIF1 INSIDE their Newton loops (WRF :3889-3895 / :4359-4365), RE-DERIVE the
canopy/bare drag from the evolving sensible-heat flux, overwrite CH with the
conductance output (:4165-4168 CH=CAH / :4476-4477 CH=EHB), and the top-level
tile-combines CM/CH (:2298-2299). Over LAND the Noah-MP-returned CM/CH (and the
ef.chv/chb diagnostics) are AUTHORITATIVE; sfclay's seed remains only the
ocean/lake path's coeffs. S6 must read back ``land_state.cm/ch`` over land, NOT
the sfclay seed (see ADR-NOAHMP-INTERFACES.md §4, S1-amend).

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
    coupler from MPTABLE/SOILPARM. ``nroot`` is the static loop bound; optional
    ``nroot_cell`` is the WRF per-column VEGPARM/MPTABLE NROOT gather.
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
    nroot_cell: jnp.ndarray | None = None


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
# THERMOPROP / TDFCND (module_sf_noahmplsm.F:2400-2510, 2573-2680).
# Soil branch (TDFCND) for the NSOIL layers + CSNOW (:2514-2569) for active snow
# layers, plus the snow/soil interface conductivity blend (:2503-2507).
# Ground BC layer = STC(ISNOW+1): soil layer 1 when ISNOW=0; the bottom active
# snow layer when ISNOW<0.
# ----------------------------------------------------------------------------------
def _tdfcnd_soil(smc, sh2o, p: EnergyParams):
    """TDFCND (:2573-2680) soil-layer thermal conductivity DF and HCPCT."""
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


def thermoprop_soil(land_state: NoahMPLandState, p: EnergyParams):
    """Soil-layer thermal conductivity DF and heat capacity HCPCT, (NSOIL, ny, nx).

    Snow-free convenience wrapper (used by the snow-free test fixtures and the
    STC update). The snow-aware ground-BC quantities are assembled by
    ``thermoprop_full`` below.
    """
    return _tdfcnd_soil(land_state.smois, land_state.sh2o, p)


def _csnow(land_state):
    """CSNOW (:2514-2569): per-snow-layer thermal conductivity TKSNO and volumetric
    heat capacity CVSNO, (NSNOW, ny, nx). Stieglitz/Yen TKSNO = 3.2217e-6*rho^2.

    Inactive snow slots (above ISNOW) carry harmless values; the caller only reads
    the active-layer / interface values via masked gathers on ISNOW.
    """
    dz = _full_dz(land_state)            # (NSNOW+NSOIL, ny, nx)
    dz_sn = dz[:NSNOW]                   # snow-layer thicknesses
    snice = land_state.snice
    snliq = land_state.snliq
    dzpos = jnp.maximum(dz_sn, MPE)
    snicev = jnp.minimum(1.0, snice / (dzpos * DENICE))
    epore = 1.0 - snicev
    snliqv = jnp.minimum(epore, snliq / (dzpos * DENH2O))
    cvsno = CICE * snicev + CWAT * snliqv
    bdsnoi = (snice + snliq) / dzpos
    tksno = 3.2217e-6 * bdsnoi ** 2.0
    return tksno, cvsno


def thermoprop_full(land_state: NoahMPLandState, p: EnergyParams, urban=None):
    """Full snow+soil DF/HCPCT plus the WRF-faithful ground-BC quantities.

    Returns ``(df_full, hcpct_full, df_top, stc_top, dz_top)`` where:
      - ``df_full``/``hcpct_full`` are (NSNOW+NSOIL, ny, nx) with the snow/soil
        interface conductivity blend applied to the top SOIL layer (:2503-2507);
      - ``df_top``/``stc_top``/``dz_top`` are the ground-BC layer values
        DF(ISNOW+1)/STC(ISNOW+1)/DZSNSO(ISNOW+1) gathered at concat index
        ``NSNOW+ISNOW`` (soil layer 1 when ISNOW=0; bottom snow layer when <0).

    ``urban`` (bool (ny,nx) or None): WRF sets soil DF=3.24 for urban
    (:2468-2472); applied before the snow/soil interface blend.
    """
    df_soil, hcpct_soil = _tdfcnd_soil(land_state.smois, land_state.sh2o, p)
    if urban is not None:
        df_soil = jnp.where(urban[None, ...], 3.24, df_soil)
    tksno, cvsno = _csnow(land_state)
    df_full = jnp.concatenate([tksno, df_soil], axis=0)        # (NSNOW+NSOIL,...)
    hcpct_full = jnp.concatenate([cvsno, hcpct_soil], axis=0)
    dz = _full_dz(land_state)
    snowh = land_state.snowh
    isnow = land_state.isnow                                   # int32 (ny,nx), <=0

    # snow/soil interface conductivity for the TOP soil layer (concat index NSNOW):
    #   ISNOW==0: DF(1)=(DF1*DZ1+0.35*SNOWH)/(SNOWH+DZ1)            (:2504)
    #   ISNOW<0 : DF(1)=(DF1*DZ1+DF0*DZ0)/(DZ0+DZ1)                 (:2506)
    dz1 = dz[NSNOW]                       # soil layer-1 thickness
    df1 = df_soil[0]
    df0 = df_full[NSNOW - 1]              # bottom snow layer conductivity (Fortran DF(0))
    dz0 = dz[NSNOW - 1]                   # bottom snow layer thickness
    iface_nosnow = (df1 * dz1 + 0.35 * snowh) / (snowh + dz1)
    iface_snow = (df1 * dz1 + df0 * dz0) / (dz0 + dz1)
    df1_blend = jnp.where(isnow == 0, iface_nosnow, iface_snow)
    df_full = df_full.at[NSNOW].set(df1_blend)

    # ground-BC layer (ISNOW+1) -> concat index NSNOW+ISNOW, gathered per column.
    bc_idx = NSNOW + isnow               # int32 (ny,nx): 3 (soil1) when ISNOW=0
    iz = jnp.arange(NSNOW + NSOIL).reshape((-1,) + (1,) * isnow.ndim)
    sel = iz == bc_idx[None, ...]        # (NSNOW+NSOIL, ny, nx) one-hot
    stc_full = jnp.concatenate([land_state.tsno, land_state.tslb], axis=0)
    df_top = jnp.sum(jnp.where(sel, df_full, 0.0), axis=0)
    stc_top = jnp.sum(jnp.where(sel, stc_full, 0.0), axis=0)
    dz_top = jnp.sum(jnp.where(sel, dz, 0.0), axis=0)
    return df_full, hcpct_full, df_top, stc_top, dz_top


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
    """Soil-layer thicknesses (NSOIL, ny, nx) = the soil slice of the WRF-faithful
    DZSNSO (:827-833). For ISNOW=0 soil layer 1 is the top active layer
    (DZSNSO = -ZSNSO(1)); for snow columns the soil DZSNSO are the ZSNSO diffs."""
    return _full_dz(land_state)[NSNOW:]


def _full_dz(land_state):
    """DZSNSO over snow+soil (NSNOW+NSOIL, ny, nx), WRF-faithful (:827-833).

    WRF computes DZSNSO only from the top ACTIVE layer ISNOW+1: that layer gets
    ``-ZSNSO(ISNOW+1)`` and every layer below gets ``ZSNSO(IZ-1)-ZSNSO(IZ)``.
    Inactive snow slots above ISNOW are left zero. Implemented vectorized with
    a per-column ISNOW-dependent first-active mask.
    """
    z = land_state.zsnso
    isnow = land_state.isnow                       # int32 (ny,nx) <=0
    first = NSNOW + isnow                           # concat index of ISNOW+1
    nl = NSNOW + NSOIL
    iz = jnp.arange(nl).reshape((-1,) + (1,) * isnow.ndim)
    zprev = jnp.concatenate([jnp.zeros((1,) + z.shape[1:]), z[:-1]], axis=0)
    dz_diff = zprev - z                             # ZSNSO(IZ-1)-ZSNSO(IZ)
    dz_first = -z                                   # -ZSNSO(IZ) for the top active
    is_first = iz == first[None, ...]
    is_active = iz >= first[None, ...]
    dz = jnp.where(is_first, dz_first, dz_diff)
    dz = jnp.where(is_active, dz, 0.0)
    return dz


# ==================================================================================
# VEGE_FLUX / BARE_FLUX tiles
# ==================================================================================
def _vege_flux(land_state, forcing, radd, phen, p, df_top, stc_top, dz_top,
               zlvl, zpd, z0m, z0mg, hcan, ur, emv, emg, gammav, gammag, rsurf,
               rhsur, latheav, dt, o2air, co2air, foln, btran, fsno,
               pahv, pahg):
    """VEGE_FLUX (:3578-4170): TV / TAH-EAH / TG Newton-Raphson, vectorized.

    Newton loop1 uses WRF's finite-iteration semantics (:4075-4080): once
    ``ITER>=5 .AND. |DTV|<=0.01`` the next sweep is the LAST (LITER=1) and the
    loop exits. Implemented branch-free as a fixed NITERC-trip scan with a per-
    column ``active`` mask that freezes a column's state on the iteration AFTER
    its last sweep (matching the Fortran "execute one more loop, then exit").
    """
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
    snowh = land_state.snowh
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
    canhs = jnp.zeros_like(tah)
    rahg = rawg = rb = jnp.zeros_like(tah)

    z0h = z0m
    z0hg = z0mg

    # WRF finite-iteration state (:3817, :4075-4080), per column.
    liter = jnp.zeros_like(tah, dtype=bool)   # this sweep is the last
    active = jnp.ones_like(tah, dtype=bool)    # column still iterating

    for it in range(1, NITERC + 1):
        # frozen carries (restored where a column has stopped iterating)
        prev = (cm, ch, fv, moz, mozsgn, fm, fh, fm2, fh2, fhg, rahg, rawg, rb,
                rssun, rssha, cah, irc, shc, evc, tr, tah, eah, tv, h, hg, qsfc,
                canhs)

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

        b = sav - irc - shc - evc - tr + pahv
        a = fveg * (4.0 * cir * tv ** 3 + csh + (cev + ctr) * destv + hcv / dt)
        dtv = b / a

        irc = irc + fveg * 4.0 * cir * tv ** 3 * dtv
        shc = shc + fveg * csh * dtv
        evc = evc + fveg * cev * destv * dtv
        tr = tr + fveg * ctr * destv * dtv
        canhs = dtv * fveg * hcv / dt
        tv = tv + dtv

        h = rhoair * CPAIR * (tah - sfctmp) / rahc
        hg = rhoair * CPAIR * (tg - tah) / rahg
        qsfc = (0.622 * eah) / (sfcprs - 0.378 * eah)

        # WRF :4075-4080: if this column already had LITER=1, it has now done its
        # one extra sweep -> stop. Else if it converged this sweep, next is last.
        stop_now = liter & active
        active = active & ~liter
        liter = jnp.where((it >= 5) & (jnp.abs(dtv) <= 0.01), True, liter)

        # freeze any column that has stopped (restore its pre-sweep carries).
        keep = active | stop_now  # columns that legitimately ran THIS sweep
        cur = (cm, ch, fv, moz, mozsgn, fm, fh, fm2, fh2, fhg, rahg, rawg, rb,
               rssun, rssha, cah, irc, shc, evc, tr, tah, eah, tv, h, hg, qsfc,
               canhs)
        frozen = [jnp.where(keep, c, o) for c, o in zip(cur, prev)]
        (cm, ch, fv, moz, mozsgn, fm, fh, fm2, fh2, fhg, rahg, rawg, rb,
         rssun, rssha, cah, irc, shc, evc, tr, tah, eah, tv, h, hg, qsfc,
         canhs) = frozen

    # under-canopy ground loop (loop2)
    air_g = -emg * (1.0 - emv) * lwdn - emg * emv * SB * tv ** 4
    cir_g = emg * SB
    csh_g = rhoair * CPAIR / rahg
    cev_g = rhoair * CPAIR / (gammag_of(land_state, forcing) * (rawg + rsurf))
    cgh_g = 2.0 * df_top / dz_top
    irg = shg = evg = gh = jnp.zeros_like(tah)
    estg = destg = jnp.zeros_like(tah)
    for _ in range(NITERG):
        estg, destg = _es_dest(tg)
        irg = cir_g * tg ** 4 + air_g
        shg = csh_g * (tg - tah)
        evg = cev_g * (estg * rhsur - eah)
        gh = cgh_g * (tg - stc_top)
        b = sag - irg - shg - evg - gh + pahg
        a = 4.0 * cir_g * tg ** 3 + csh_g + cev_g * destg + cgh_g
        dtg = b / a
        irg = irg + 4.0 * cir_g * tg ** 3 * dtg
        shg = shg + csh_g * dtg
        evg = evg + cev_g * destg * dtg
        gh = gh + cgh_g * dtg
        tg = tg + dtg

    # snow on ground & TG>TFRZ: reset TG=TFRZ, re-evaluate ground fluxes (opt_stc=1,
    # :4125-4133). estg/destg held at the final loop2 ESTG (TG before clamp).
    snow_clamp = (snowh > 0.05) & (tg > TFRZ)
    tg_c = jnp.where(snow_clamp, TFRZ, tg)
    irg_c = cir_g * tg_c ** 4 - emg * (1.0 - emv) * lwdn - emg * emv * SB * tv ** 4
    shg_c = csh_g * (tg_c - tah)
    evg_c = cev_g * (estg * rhsur - eah)
    gh_c = sag + pahg - (irg_c + shg_c + evg_c)
    tg = tg_c
    irg = jnp.where(snow_clamp, irg_c, irg)
    shg = jnp.where(snow_clamp, shg_c, shg)
    evg = jnp.where(snow_clamp, evg_c, evg)
    gh = jnp.where(snow_clamp, gh_c, gh)

    # 2-m air temperature over the VEGETATED tile (T2MV), opt_sfc=1 (:4148-4163).
    # Uses the SFCDIF1-converged FH2/FV/Z0H (carried through loop1, frozen at each
    # column's last sweep) and the FINAL canopy-air temperature TAH plus the final
    # under-canopy ground SHG and canopy SHC. SHC carries the FVEG weight (it is the
    # tile-summed canopy flux), so WRF divides it back out (SHC/FVEG); SHG is the raw
    # under-canopy ground flux. CAH2 = FV*VKC/(LOG((2+Z0H)/Z0H)-FH2). The CAH2<1e-5
    # branch falls back to T2MV=TAH (degenerate near-zero conductance).
    cah2 = fv * VKC / (jnp.log((2.0 + z0h) / z0h) - fh2)
    fveg_safe = jnp.maximum(fveg, MPE)
    t2mv = jnp.where(
        cah2 < 1.0e-5,
        tah,
        tah - (shg + shc / fveg_safe) / (rhoair * CPAIR) * (1.0 / cah2),
    )

    return {
        "irc": irc, "shc": shc, "evc": evc, "tr": tr,
        "irg": irg, "shg": shg, "evg": evg, "ghv": gh,
        "tv": tv, "tg": tg, "tah": tah, "eah": eah,
        "cm": cm, "chv": cah, "qsfc": qsfc, "canhs": canhs,
        "t2mv": t2mv,
    }


def gammag_of(land_state, forcing):
    latheag = jnp.where(land_state.tg > TFRZ, HVAP, HSUB)
    return CPAIR * forcing.sfcprs / (0.622 * latheag)


def _bare_flux(land_state, forcing, radd, p, df_top, stc_top, dz_top, zlvl,
               zpdg, z0mg, ur, emg, gammag, rsurf, rhsur, pahb):
    """BARE_FLUX (:4174-4479): bare-tile ground (TG) Newton-Raphson, vectorized.

    Recomputes and returns QSFC from the bare-ground saturation path
    (:4428-4437) and applies the opt_stc=1 snow TG clamp (:4444-4451).
    """
    eair = forcing.qair * forcing.sfcprs / (0.622 + 0.378 * forcing.qair)
    rhoair = (forcing.sfcprs - 0.378 * eair) / (RAIR * forcing.sfctmp)
    sfctmp = forcing.sfctmp
    psfc = forcing.psfc
    qair = forcing.qair
    sag = radd["sag"]
    lwdn = forcing.lwdn
    tgb = land_state.tg
    snowh = land_state.snowh

    cir = emg * SB
    cgh = 2.0 * df_top / dz_top

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
    qsfc = land_state.qsfc
    estg = jnp.zeros_like(tgb)
    csh = cev = jnp.zeros_like(tgb)

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
        ghb = cgh * (tgb - stc_top)
        b = sag - irb - shb - evb - ghb + pahb
        a = 4.0 * cir * tgb ** 3 + csh + cev * destg + cgh
        dtg = b / a
        irb = irb + 4.0 * cir * tgb ** 3 * dtg
        shb = shb + csh * dtg
        evb = evb + cev * destg * dtg
        ghb = ghb + cgh * dtg
        tgb = tgb + dtg
        h = csh * (tgb - sfctmp)
        # QSFC from bare-ground saturation at the updated TGB (:4428-4435)
        estg_q, _ = _es_dest(tgb)
        qsfc = 0.622 * (estg_q * rhsur) / (psfc - 0.378 * (estg_q * rhsur))
        estg = estg_q

    # snow on ground & TGB>TFRZ: reset TGB=TFRZ, re-evaluate ground fluxes
    # (opt_stc=1, :4444-4451). ESTG held at the final loop3 saturation value.
    snow_clamp = (snowh > 0.05) & (tgb > TFRZ)
    tgb_c = jnp.where(snow_clamp, TFRZ, tgb)
    # csh/cev held at the final loop3 values (RAHB/RAWB from the last sweep).
    irb_c = cir * tgb_c ** 4 - emg * lwdn
    shb_c = csh * (tgb_c - sfctmp)
    evb_c = cev * (estg * rhsur - eair)
    ghb_c = sag + pahb - (irb_c + shb_c + evb_c)
    tgb = tgb_c
    irb = jnp.where(snow_clamp, irb_c, irb)
    shb = jnp.where(snow_clamp, shb_c, shb)
    evb = jnp.where(snow_clamp, evb_c, evb)
    ghb = jnp.where(snow_clamp, ghb_c, ghb)

    # 2-m air temperature over the BARE tile (T2MB), opt_sfc=1 (:4461-4474). Uses
    # the SFCDIF1-converged FH2/FV/Z0H (Z0H=Z0MG over bare ground; CZIL commented
    # out at WRF :4354-4356) and the final bare-ground SHB / skin TGB. EHB2 =
    # FV*VKC/(LOG((2+Z0H)/Z0H)-FH2); EHB2<1e-5 falls back to T2MB=TGB.
    ehb2 = fv * VKC / (jnp.log((2.0 + z0h) / z0h) - fh2)
    t2mb = jnp.where(
        ehb2 < 1.0e-5,
        tgb,
        tgb - shb / (rhoair * CPAIR) * (1.0 / ehb2),
    )

    return {"irb": irb, "shb": shb, "evb": evb, "ghb": ghb, "tgb": tgb,
            "cm": cm, "ch": ehb, "qsfc": qsfc, "t2mb": t2mb}


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
    pahv_kw: jnp.ndarray | float = 0.0,
    pahg_kw: jnp.ndarray | float = 0.0,
    pahb_kw: jnp.ndarray | float = 0.0,
    isurban: int | None = None,
) -> tuple[NoahMPLandState, NoahMPEnergyFluxes, NoahMPEtFluxes]:
    """Canopy/ground surface-energy balance — THE HFX FIX.

    ENERGY (:1741-2396) for opt_sfc=1/opt_crs=1/opt_btr=1/opt_rad=3/opt_alb=2.
    The two-stream ``rad`` is from ``noahmp_radiation_twostream``; ``rad_extras``
    (the fsun/laisun/laisha/tauss/albold bundle radiation also produces) is the
    second element of ``energy_radiation.radiation_twostream`` and is required.

    Returns ``(land_state', energy_fluxes, et_fluxes)``.

    PAH (precip-advected heat) and the STOMATA carbon forcing (o2air/co2air/foln)
    are taken from the ADDITIVE ``NoahMPForcing`` fields when supplied (S6 path);
    otherwise from the keyword fallbacks below (PAH=0 no-precip; carbon from the
    WRF block ``co2air=395e-6*SFCPRS``, ``o2air=0.209*SFCPRS``).
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
    zero = jnp.zeros_like(fveg)

    # PAH + STOMATA carbon forcing: prefer the additive forcing fields (S6),
    # else the keyword fallbacks. (PAHV/PAHG/PAHB default 0 -> no-precip.)
    pahv = forcing.pahv if forcing.pahv is not None else (pahv_kw + zero)
    pahg = forcing.pahg if forcing.pahg is not None else (pahg_kw + zero)
    pahb = forcing.pahb if forcing.pahb is not None else (pahb_kw + zero)
    o2 = forcing.o2air if forcing.o2air is not None else (o2air + zero)
    co2 = forcing.co2air if forcing.co2air is not None else (co2air + zero)
    foln_v = forcing.foln if forcing.foln is not None else (foln + zero)

    # urban special-case mask (ENERGY :2099-2106, THERMOPROP :2468, FVEG :874).
    # Urban is CUT from the active land scope (ADR §1); this branch is carried
    # only so an urban cell that reaches S1 stays WRF-faithful (Z0MG=Z0MVT,
    # ZPDG=0.65*HVT, soil DF=3.24). ``urban`` defaults False (no urban cells).
    urban = (static.ivgtyp == isurban) if isurban is not None else jnp.zeros_like(veg)

    # roughness / displacement (ENERGY :2075-2109)
    z0mg = Z0_BARE * (1.0 - fsno) + p.z0sno * fsno
    zpdg = land_state.snowh
    z0m = jnp.where(veg, p.z0mvt, z0mg)
    zpd = jnp.where(veg, jnp.maximum(0.65 * p.hvt, land_state.snowh), zpdg)
    # special case for urban (:2101-2106): Z0MG=Z0MVT, ZPDG=0.65*HVT, Z0M=ZPD=...
    z0mg = jnp.where(urban, p.z0mvt, z0mg)
    zpdg = jnp.where(urban, 0.65 * p.hvt, zpdg)
    z0m = jnp.where(urban, p.z0mvt, z0m)
    zpd = jnp.where(urban, 0.65 * p.hvt, zpd)
    zref = forcing.zlvl
    zlvl = jnp.maximum(zpd, p.hvt) + zref
    zlvl = jnp.where(zpdg >= zlvl, zpdg + zref, zlvl)
    hcan = p.hvt
    ur = jnp.maximum(jnp.sqrt(forcing.uu ** 2 + forcing.vv ** 2), 1.0)

    # THERMOPROP DF/HCPCT incl. snow (CSNOW) + snow/soil interface blend; the
    # ground-BC layer DF(ISNOW+1)/STC(ISNOW+1)/DZSNSO(ISNOW+1) for the flux solve.
    df_full, hcpct_full, df_top, stc_top, dz_top = thermoprop_full(land_state, p, urban)
    df = df_full[NSNOW:]   # soil-layer DF (post-interface-blend) for STC update

    # emissivities (ENERGY :2139-2144)
    emv = 1.0 - jnp.exp(-(elai + esai) / 1.0)
    emg = p.eg * (1.0 - fsno) + p.snow_emis * fsno

    # BTRAN (opt_btr=1) over rooting layers (ENERGY :2148-2171)
    sh2o = land_state.sh2o
    zsoil = static.zsoil
    dz_soil = _soil_dz(land_state)
    nroot = max(0, min(NSOIL, int(p.nroot)))
    if p.nroot_cell is None:
        nroot_cell = jnp.full(fveg.shape, nroot, dtype=jnp.int32)
    else:
        nroot_cell = jnp.clip(jnp.asarray(p.nroot_cell, dtype=jnp.int32), 0, nroot)
    root_depth = -zsoil[jnp.maximum(nroot_cell, 1) - 1]
    btran = jnp.zeros_like(fveg)
    btrani = jnp.zeros((NSOIL,) + fveg.shape)
    for iz in range(nroot):
        active_root = (iz + 1) <= nroot_cell
        gx = (sh2o[iz] - p.smcwlt[iz]) / (p.smcref[iz] - p.smcwlt[iz])
        gx = jnp.minimum(1.0, jnp.maximum(0.0, gx))
        bi = jnp.maximum(MPE, dz_soil[iz] / root_depth * gx)
        bi = jnp.where(active_root, bi, 0.0)
        btrani = btrani.at[iz].set(bi)
        btran = btran + bi
    btran = jnp.maximum(MPE, btran)
    for iz in range(nroot):
        active_root = (iz + 1) <= nroot_cell
        btrani = btrani.at[iz].set(jnp.where(active_root, btrani[iz] / btran, 0.0))

    # ground surface resistance + RHSUR (ENERGY :2173-2201, OPT_RSF=1 SZ09)
    sh2o1 = sh2o[0]
    smcmax1 = p.smcmax[0]
    bexp1 = p.bexp[0]
    psisat1 = p.psisat[0]
    l_rsurf = (-zsoil[0]) * (jnp.exp((1.0 - jnp.minimum(1.0, sh2o1 / smcmax1)) ** p.rsurf_exp) - 1.0) / (2.71828 - 1.0)
    d_rsurf = 2.2e-5 * smcmax1 * smcmax1 * (1.0 - p.smcwlt[0] / smcmax1) ** (2.0 + 3.0 / bexp1)
    rsurf = l_rsurf / d_rsurf
    rsurf = jnp.where((sh2o1 < 0.01) & (land_state.snowh == 0.0), 1.0e6, rsurf)
    # urban impervious surface (:2204-2206): RSURF=1e6 when snow-free
    rsurf = jnp.where(urban & (land_state.snowh == 0.0), 1.0e6, rsurf)
    psi = -psisat1 * (jnp.maximum(0.01, sh2o1) / smcmax1) ** (-bexp1)
    rhsur = fsno + (1.0 - fsno) * jnp.exp(psi * GRAV / (RW * land_state.tg))

    # psychrometric constants (ENERGY :2210-2226)
    latheav = jnp.where(land_state.tv > TFRZ, HVAP, HSUB)
    gammav = CPAIR * forcing.sfcprs / (0.622 * latheav)
    latheag = jnp.where(land_state.tg > TFRZ, HVAP, HSUB)
    gammag = CPAIR * forcing.sfcprs / (0.622 * latheag)

    vf = _vege_flux(land_state, forcing, radd, phen, p, df_top, stc_top, dz_top,
                    zlvl, zpd, z0m, z0mg, hcan, ur, emv, emg, gammav, gammag,
                    rsurf, rhsur, latheav, dt, o2, co2, foln_v, btran, fsno,
                    pahv, pahg)
    bf = _bare_flux(land_state, forcing, radd, p, df_top, stc_top, dz_top, zlvl,
                    zpdg, z0mg, ur, emg, gammag, rsurf, rhsur, pahb)

    # FVEG-weighted tile sum (ENERGY :2285-2325)
    use_veg = veg & (fveg > 0.0)
    fira = jnp.where(use_veg, fveg * vf["irg"] + (1.0 - fveg) * bf["irb"] + vf["irc"], bf["irb"])
    fsh = jnp.where(use_veg, fveg * vf["shg"] + (1.0 - fveg) * bf["shb"] + vf["shc"], bf["shb"])
    fgev = jnp.where(use_veg, fveg * vf["evg"] + (1.0 - fveg) * bf["evb"], bf["evb"])
    ssoil = jnp.where(use_veg, fveg * vf["ghv"] + (1.0 - fveg) * bf["ghb"], bf["ghb"])
    fcev = jnp.where(use_veg, vf["evc"], 0.0)
    fctr = jnp.where(use_veg, vf["tr"], 0.0)
    canhs = jnp.where(use_veg, vf["canhs"], 0.0)
    # PAH tile sum (ENERGY :2294/2314): veg = FVEG*PAHG+(1-FVEG)*PAHB+PAHV; bare=PAHB
    pah = jnp.where(use_veg, fveg * pahg + (1.0 - fveg) * pahb + pahv, pahb)
    tg = jnp.where(use_veg, fveg * vf["tg"] + (1.0 - fveg) * bf["tgb"], bf["tgb"])
    # 2-m air temperature tile-combine (ENERGY :2296 vegetated / :2311 bare).
    # T2M = FVEG*T2MV + (1-FVEG)*T2MB over the vegetated tile; T2M = T2MB over
    # bare/FVEG=0 (the ELSE branch). This is the LSM 2-m temperature the WRF
    # surface driver writes back as the land T2 (module_surface_driver.F:3470/3467),
    # OVERWRITING the surface-layer MYNN diagnostic over land.
    t2mv = vf["t2mv"]
    t2mb = bf["t2mb"]
    t2 = jnp.where(use_veg, fveg * t2mv + (1.0 - fveg) * t2mb, t2mb)
    tv = jnp.where(use_veg, vf["tv"], land_state.tv)
    tah = jnp.where(use_veg, vf["tah"], land_state.tah)
    eah = jnp.where(use_veg, vf["eah"], land_state.eah)
    chv = jnp.where(use_veg, vf["chv"], bf["ch"])
    chb = bf["ch"]
    ch = jnp.where(use_veg, fveg * vf["chv"] + (1.0 - fveg) * bf["ch"], bf["ch"])
    cm = jnp.where(use_veg, fveg * vf["cm"] + (1.0 - fveg) * bf["cm"], bf["cm"])
    # QSFC = Q1 tile-combine (ENERGY :2300/2318): veg = FVEG*(EAH canopy-air q) +
    # (1-FVEG)*QSFC_bare; bare = QSFC_bare. The driver writes Q1 back as QSFC
    # (module_sf_noahmpdrv.F:1244).
    q1_canopy = vf["eah"] * 0.622 / (forcing.sfcprs - 0.378 * vf["eah"])
    qsfc = jnp.where(use_veg, fveg * q1_canopy + (1.0 - fveg) * bf["qsfc"], bf["qsfc"])
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

    # urban surface humidity (NOAHMP_SFLX :1062-1064): QSFC=QFX/(RHOAIR*CH)+QAIR.
    rhoair_top = (forcing.sfcprs - 0.378 * forcing.qair * forcing.sfcprs
                  / (0.622 + 0.378 * forcing.qair)) / (RAIR * forcing.sfctmp)
    qsfc = jnp.where(urban, qfx / (rhoair_top * jnp.maximum(ch, MPE)) + forcing.qair, qsfc)

    # semi-implicit STC update (TSNOSOI / Sprint S2). Fluxes above used the OLD
    # STC(1) as the ground BC, so this does not change this step's HFX/LH/SSOIL/
    # TRAD. Tolerate S2 unmerged (stub raises NotImplementedError).
    tslb_new = land_state.tslb
    try:
        from gpuwrf.physics.noahmp.soil_thermo import noahmp_soil_thermo

        stc = jnp.concatenate([land_state.tsno, land_state.tslb], axis=0)
        dzsnso = _full_dz(land_state)
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
        trad=trad, emissi=emissi, z0wrf=z0wrf, chv=chv, chb=chb, canhs=canhs,
        t2mv=t2mv, t2mb=t2mb, t2=t2,
    )
    et = NoahMPEtFluxes(
        ecan=ecan, etran=etran, edir=edir, qseva=qvap, btrani=btrani,
        qsnow=forcing.prcpsnow, qmelt=jnp.zeros_like(fsh),
        imelt=jnp.zeros((NSNOW + NSOIL,) + fveg.shape, dtype=jnp.int32),
    )
    return ls, ef, et


__all__ = ["noahmp_radiation_twostream", "noahmp_energy_canopy", "EnergyParams"]
