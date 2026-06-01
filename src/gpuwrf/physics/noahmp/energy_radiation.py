"""Noah-MP two-stream canopy radiation (Sprint S1 split-out helper).

THE LARGEST SUB-PIECE of ENERGY, split into its own module per ADR-NOAHMP-
INTERFACES.md §6.1 / §7 ("two-stream radiation lives inside S1 (energy) and is
the largest sub-component — flagged for possible split if S1 overruns").

Faithful port of the WRF Noah-MP RADIATION chain
(``module_sf_noahmplsm.F``):
  - RADIATION (:2684-2806) orchestrator
  - ALBEDO    (:2810-2990) : SNOW_AGE + SNOWALB_CLASS (opt_alb=2) + GROUNDALB +
                             two TWOSTREAM calls (direct/diffuse) per band
  - SURRAD    (:2994-3115) : assemble SAV/SAG/PARSUN/PARSHA/FSA/FSR
  - SNOW_AGE  (:3119-3167)
  - SNOWALB_CLASS (:3226-3275, opt_alb=2 CLASS aging)
  - GROUNDALB (:3279-3332)
  - TWOSTREAM (:3336-3574, Dickinson 1983 / Sellers 1985, opt_rad=3 gap=1-FVEG)

Active options: opt_rad = 3 (GAP = KOPEN = 1-FVEG), opt_alb = 2 (CLASS).
NBAND = 2 (1 = vis, 2 = nir). Two-band SOLAD/SOLAI from ATM (:1158-1161):
  SOLAD = SWDOWN*0.7*0.5 (each band); SOLAI = SWDOWN*0.3*0.5 (each band).

All operations are fully vectorized over the land-column grid (ny, nx); fp64
(x64 enabled at package import). No host transfer; pure functional.

WRF semantics surprise worth flagging: the grid albedo SALB reported by the
driver is recomputed in NOAHMP_SFLX as FSR/SWDOWN (:1072-1076), NOT the
two-stream ALBD/ALBI. We therefore return SALB = FSR/SWDOWN to match the
driver's reported ALBEDO.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPForcing, NoahMPPhenology, NoahMPRadInputs

# --- module constants (module_sf_noahmplsm.F:204-220) ---
TFRZ = 273.16
MPE = 1.0e-6

# --- scoped scalar/per-band radiation parameters (MPTABLE.TBL) ---
OMEGAS = (0.8, 0.4)   # two-stream omega for snow (vis, nir)
BETADS = 0.5
BETAIS = 0.5
SWEMX = 1.0           # m water-equivalent fresh-snow full-cover (GLOBAL block)
# SNOW_AGE (BATS) advances TAUSS; opt_alb=2 does not consume FAGE but TAUSS is
# still aged each step.
TAU0 = 1.0e6
GRAIN_GROWTH = 5000.0
EXTRA_GROWTH = 10.0
DIRT_SOOT = 0.3

# MFSNO/SCFFAC default for the FSNO tanh (MPTABLE; sparse-veg ~ 2.5/0.042).
# Snow-free land columns short-circuit to FSNO=0 so this only matters under snow.
_MFSNO = 2.5
_SCFFAC = 0.042


class TwoStreamParams(NamedTuple):
    """Per-column gathered radiation parameters (subset of NoahMPParameters +
    the radiation-only extras not in the frozen tuple).

    Per-band arrays are stacked on a leading length-2 axis (0 = vis, 1 = nir)
    and broadcast over the land-column grid (ny, nx).
    """

    rhol: jnp.ndarray    # (2, ny, nx) leaf reflectance
    rhos: jnp.ndarray    # (2, ny, nx) stem reflectance
    taul: jnp.ndarray    # (2, ny, nx) leaf transmittance
    taus: jnp.ndarray    # (2, ny, nx) stem transmittance
    xl: jnp.ndarray      # (ny, nx) leaf orientation index
    albsat: jnp.ndarray  # (2, ny, nx) saturated soil albedo
    albdry: jnp.ndarray  # (2, ny, nx) dry soil albedo


def _fsno(snowh, sneqv):
    bdsno = sneqv / jnp.maximum(snowh, MPE)
    fmelt = (bdsno / 100.0) ** _MFSNO
    return jnp.tanh(snowh / (_SCFFAC * fmelt))


def snow_age(dt, tg, sneqvo, sneqv, tauss):
    """SNOW_AGE (:3119-3167). Returns (tauss_new, fage)."""
    arg = GRAIN_GROWTH * (1.0 / TFRZ - 1.0 / tg)
    age1 = jnp.exp(arg)
    age2 = jnp.exp(jnp.minimum(0.0, EXTRA_GROWTH * arg))
    age3 = DIRT_SOOT
    tage = age1 + age2 + age3
    dela0 = dt / TAU0
    dela = dela0 * tage
    dels = jnp.maximum(0.0, sneqv - sneqvo) / SWEMX
    sge = (tauss + dela) * (1.0 - dels)
    tauss_new = jnp.where(sneqv <= 0.0, 0.0, jnp.maximum(0.0, sge))
    fage = tauss_new / (tauss_new + 1.0)
    return tauss_new, fage


def snowalb_class(qsnow, dt, albold):
    """SNOWALB_CLASS (:3226-3275, opt_alb=2). Returns (alb, albsnd, albsni)."""
    alb = 0.55 + (albold - 0.55) * jnp.exp(-0.01 * dt / 3600.0)
    alb = jnp.where(
        qsnow > 0.0,
        alb + jnp.minimum(qsnow, SWEMX / dt) * (0.84 - alb) / (SWEMX / dt),
        alb,
    )
    albsn = jnp.stack([alb, alb], axis=0)  # vis, nir identical for CLASS
    return alb, albsn, albsn


def groundalb(smc1, fsno, albsnd, albsni, p: TwoStreamParams):
    """GROUNDALB (:3279-3332), IST=1 soil branch. Returns (albgrd, albgri)."""
    inc = jnp.maximum(0.11 - 0.40 * smc1, 0.0)  # (ny, nx)
    albsod = jnp.minimum(p.albsat + inc[None, ...], p.albdry)  # (2, ny, nx)
    albsoi = albsod
    albgrd = albsod * (1.0 - fsno) + albsnd * fsno
    albgri = albsoi * (1.0 - fsno) + albsni * fsno
    return albgrd, albgri


def twostream(ib, ic, cosz, vai, fwet, tveg, albgr_d, albgr_i, rho, tau, fveg, p: TwoStreamParams):
    """TWOSTREAM (:3336-3574) for band ``ib`` and case ``ic`` (0=direct,
    1=diffuse), opt_rad=3. Vectorized over (ny, nx).

    Returns (fab, fre, ftd, fti, gdir, frev, freg) for this band/case.
    """
    gap = jnp.where(vai == 0.0, 1.0, 1.0 - fveg)
    kopen = jnp.where(vai == 0.0, 1.0, 1.0 - fveg)

    coszi = jnp.maximum(0.001, cosz)
    chil = jnp.minimum(jnp.maximum(p.xl, -0.4), 0.6)
    chil = jnp.where(jnp.abs(chil) <= 0.01, 0.01, chil)
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    gdir = phi1 + phi2 * coszi
    ext = gdir / coszi
    avmu = (1.0 - phi1 / phi2 * jnp.log((phi1 + phi2) / phi1)) / phi2
    omegal = rho + tau
    tmp0 = gdir + phi2 * coszi
    tmp1 = phi1 * coszi
    asu = 0.5 * omegal * gdir / tmp0 * (1.0 - tmp1 / tmp0 * jnp.log((tmp1 + tmp0) / tmp1))
    betadl = (1.0 + avmu * ext) / (omegal * avmu * ext) * asu
    betail = 0.5 * (rho + tau + (rho - tau) * ((1.0 + chil) / 2.0) ** 2) / omegal

    omegas_b = OMEGAS[ib]
    no_snow = tveg > TFRZ
    omega_sn = (1.0 - fwet) * omegal + fwet * omegas_b
    betad_sn = ((1.0 - fwet) * omegal * betadl + fwet * omegas_b * BETADS) / omega_sn
    betai_sn = ((1.0 - fwet) * omegal * betail + fwet * omegas_b * BETAIS) / omega_sn
    omega = jnp.where(no_snow, omegal, omega_sn)
    betad = jnp.where(no_snow, betadl, betad_sn)
    betai = jnp.where(no_snow, betail, betai_sn)

    b = 1.0 - omega + omega * betai
    c = omega * betai
    tmp0 = avmu * ext
    d = tmp0 * omega * betad
    f = tmp0 * omega * (1.0 - betad)
    tmp1 = b * b - c * c
    h = jnp.sqrt(tmp1) / avmu
    sigma = tmp0 * tmp0 - tmp1
    sigma = jnp.where(jnp.abs(sigma) < 1.0e-6, jnp.where(sigma < 0.0, -1.0e-6, 1.0e-6), sigma)
    p1 = b + avmu * h
    p2 = b - avmu * h
    p3 = b + tmp0
    p4 = b - tmp0
    s1 = jnp.exp(-jnp.minimum(h * vai, 40.0))
    s2 = jnp.exp(-jnp.minimum(ext * vai, 40.0))

    albgr = jnp.where(ic == 0, albgr_d, albgr_i)
    u1 = b - c / albgr
    u2 = b - c * albgr
    u3 = f + c * albgr

    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    tmp4 = u2 + avmu * h
    tmp5 = u2 - avmu * h
    d2 = tmp4 / s1 - tmp5 * s1
    h1 = -d * p4 - c * f
    tmp6 = d - h1 * p3 / sigma
    tmp7 = (d - c - h1 / sigma * (u1 + tmp0)) * s2
    h2 = (tmp6 * tmp2 / s1 - p2 * tmp7) / d1
    h3 = -(tmp6 * tmp3 * s1 - p1 * tmp7) / d1
    h4 = -f * p3 - c * d
    tmp8 = h4 / sigma
    tmp9 = (u3 - tmp8 * (u2 - tmp0)) * s2
    h5 = -(tmp8 * tmp4 / s1 + tmp9) / d2
    h6 = (tmp8 * tmp5 * s1 + tmp9) / d2
    h7 = (c * tmp2) / (d1 * s1)
    h8 = (-c * tmp3 * s1) / d1
    h9 = tmp4 / (d2 * s1)
    h10 = (-tmp5 * s1) / d2

    ftds_dir = s2 * (1.0 - gap) + gap
    ftis_dir = (h4 * s2 / sigma + h5 * s1 + h6 / s1) * (1.0 - gap)
    ftds_dif = jnp.zeros_like(s2)
    ftis_dif = (h9 * s1 + h10 / s1) * (1.0 - kopen) + kopen
    ftd = jnp.where(ic == 0, ftds_dir, ftds_dif)
    fti = jnp.where(ic == 0, ftis_dir, ftis_dif)

    fres_dir = (h1 / sigma + h2 + h3) * (1.0 - gap) + albgr_d * gap
    freveg_dir = (h1 / sigma + h2 + h3) * (1.0 - gap)
    frebar_dir = albgr_d * gap
    fres_dif = (h7 + h8) * (1.0 - kopen) + albgr_i * kopen
    freveg_dif = (h7 + h8) * (1.0 - kopen) + albgr_i * kopen
    frebar_dif = jnp.zeros_like(albgr_i)
    fre = jnp.where(ic == 0, fres_dir, fres_dif)
    frev = jnp.where(ic == 0, freveg_dir, freveg_dif)
    freg = jnp.where(ic == 0, frebar_dir, frebar_dif)

    fab = 1.0 - fre - (1.0 - albgr_d) * ftd - (1.0 - albgr_i) * fti
    return fab, fre, ftd, fti, gdir, frev, freg


def radiation_twostream(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    phen: NoahMPPhenology,
    params: TwoStreamParams,
    dt: float,
):
    """RADIATION (:2684-2806). Returns (NoahMPRadInputs, extras dict).

    extras carries ``fsun``/``laisun``/``laisha``/``vai`` for the flux solve and
    the advanced ``tauss``/``albold`` snow-albedo carry.
    """
    cosz = forcing.cosz
    elai = phen.elai
    esai = phen.esai
    fveg = phen.fveg
    tg = land_state.tg
    tv = land_state.tv
    fwet = land_state.fwet
    smc1 = land_state.smois[0]
    sneqv = land_state.sneqv
    sneqvo = land_state.sneqvo
    snowh = land_state.snowh
    qsnow = forcing.prcpsnow

    swdown = jnp.where(cosz <= 0.0, 0.0, forcing.soldn)
    solad = jnp.stack([swdown * 0.7 * 0.5, swdown * 0.7 * 0.5], axis=0)
    solai = jnp.stack([swdown * 0.3 * 0.5, swdown * 0.3 * 0.5], axis=0)

    vai = elai + esai
    fsno = jnp.where(snowh > 0.0, _fsno(snowh, sneqv), 0.0)

    tauss_new, _fage = snow_age(dt, tg, sneqvo, sneqv, land_state.tauss)
    alb_new, albsnd, albsni = snowalb_class(qsnow, dt, land_state.albold)

    wl = elai / jnp.maximum(vai, MPE)
    ws = esai / jnp.maximum(vai, MPE)
    rho = jnp.maximum(params.rhol * wl[None, ...] + params.rhos * ws[None, ...], MPE)
    tau = jnp.maximum(params.taul * wl[None, ...] + params.taus * ws[None, ...], MPE)

    albgrd, albgri = groundalb(smc1, fsno, albsnd, albsni, params)

    fabd, albd, ftdd, ftid = [], [], [], []
    fabi, albi, ftii = [], [], []
    gdir_vis = None
    for ib in range(2):
        fab_d, fre_d, ftd_d, fti_d, gdir, _frv_d, _frg_d = twostream(
            ib, 0, cosz, vai, fwet, tv, albgrd[ib], albgri[ib], rho[ib], tau[ib], fveg, params
        )
        fab_i, fre_i, ftd_i, fti_i, _gd, _frv_i, _frg_i = twostream(
            ib, 1, cosz, vai, fwet, tv, albgrd[ib], albgri[ib], rho[ib], tau[ib], fveg, params
        )
        fabd.append(fab_d)
        albd.append(fre_d)
        ftdd.append(ftd_d)
        ftid.append(fti_d)
        fabi.append(fab_i)
        albi.append(fre_i)
        ftii.append(fti_i)
        if ib == 0:
            gdir_vis = gdir
    fabd = jnp.stack(fabd, 0)
    albd = jnp.stack(albd, 0)
    ftdd = jnp.stack(ftdd, 0)
    ftid = jnp.stack(ftid, 0)
    fabi = jnp.stack(fabi, 0)
    albi = jnp.stack(albi, 0)
    ftii = jnp.stack(ftii, 0)

    ext = gdir_vis / jnp.maximum(cosz, MPE) * jnp.sqrt(jnp.maximum(1.0 - rho[0] - tau[0], 0.0))
    fsun = (1.0 - jnp.exp(-jnp.minimum(ext * vai, 40.0))) / jnp.maximum(ext * vai, MPE)
    fsun = jnp.where(fsun < 0.01, 0.0, fsun)
    fsun = jnp.where(cosz > 0.0, fsun, 0.0)

    fsha = 1.0 - fsun
    laisun = elai * fsun
    laisha = elai * fsha

    sag = jnp.zeros_like(cosz)
    sav = jnp.zeros_like(cosz)
    fsa = jnp.zeros_like(cosz)
    cad, cai = [], []
    for ib in range(2):
        cad_b = solad[ib] * fabd[ib]
        cai_b = solai[ib] * fabi[ib]
        sav = sav + cad_b + cai_b
        fsa = fsa + cad_b + cai_b
        trd = solad[ib] * ftdd[ib]
        tri = solad[ib] * ftid[ib] + solai[ib] * ftii[ib]
        absorb = trd * (1.0 - albgrd[ib]) + tri * (1.0 - albgri[ib])
        sag = sag + absorb
        fsa = fsa + absorb
        cad.append(cad_b)
        cai.append(cai_b)

    laifra = elai / jnp.maximum(vai, MPE)
    parsun = jnp.where(
        fsun > 0.0,
        (cad[0] + fsun * cai[0]) * laifra / jnp.maximum(laisun, MPE),
        0.0,
    )
    parsha = jnp.where(
        fsun > 0.0,
        (fsha * cai[0]) * laifra / jnp.maximum(laisha, MPE),
        (cad[0] + cai[0]) * laifra / jnp.maximum(laisha, MPE),
    )

    rvis = albd[0] * solad[0] + albi[0] * solai[0]
    rnir = albd[1] * solad[1] + albi[1] * solai[1]
    fsr = rvis + rnir

    albedo = jnp.where(swdown != 0.0, fsr / jnp.maximum(swdown, MPE), land_state.albedo)

    rad = NoahMPRadInputs(
        sav=sav,
        sag=sag,
        parsun=parsun,
        parsha=parsha,
        fsa=fsa,
        fsr=fsr,
        albedo=albedo,
        fsno=fsno,
    )
    extras = {
        "fsun": fsun,
        "laisun": laisun,
        "laisha": laisha,
        "tauss": tauss_new,
        "albold": alb_new,
        "vai": vai,
    }
    return rad, extras


__all__ = [
    "TwoStreamParams",
    "radiation_twostream",
    "snow_age",
    "snowalb_class",
    "groundalb",
    "twostream",
]
