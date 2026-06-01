"""Noah-MP precipitation partition + precip-advected heat (Sprint S6a).

Faithful JAX port of the two pristine-WRF NOAHMP_SFLX pre-ENERGY blocks that
S1 ENERGY consumes (PAHV/PAHG/PAHB) and S4 WATER / S3 SNOW consume (canopy
interception QINTR/QDRIP/QTHRO, FWET/CANLIQ/CANICE, QRAIN/QSNOW/SNOWHIN, BDFALL):

  1. ``ATM`` (module_sf_noahmplsm.F:1083-1252), restricted to the scoped options:
     SOLAD/SOLAI band split, FP (fractional precip area, Niu05), the **opt_snf=1
     Jordan-1991** rain/snow partition FPICE, and the Hedstrom-Pomeroy-1998 fresh
     snow bulk density BDFALL.  (opt_snf in {2,3,4,5} CUT — not the Canary scope.)
  2. ``PRECIP_HEAT`` (module_sf_noahmplsm.F:1362-1556): canopy rain/snow
     interception + unloading, the wetted-canopy fraction FWET, the total canopy
     water CMC, and the three precip-advected heat terms PAHV/PAHG/PAHB (with the
     FVEG renormalisation + the ±20 W/m2 stability clamp), QRAIN/QSNOW reaching
     the ground, and SNOWHIN = QSNOW/BDFALL.

All vectorised over the land-column grid ``(ny, nx)``; fp64; branch-free
(``jnp.where`` for the Fortran IFs). Pure functional: returns a small result
container + the (CANLIQ, CANICE) update; the driver writes them into the carry.

The only veg-table parameter PRECIP_HEAT reads is ``CH2OP`` (max intercepted
water per unit LAI+SAI); it is gathered by the driver from ``static.parameters``
(``NoahMPParameters.ch2op``) and passed in as ``ch2op``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

# Physical constants (module_sf_noahmplsm.F:204-220) — identical to energy.py.
TFRZ = 273.16
CWAT = 4.188e06     # specific heat capacity of water [J/m3/K]
CICE = 2.094e06     # specific heat capacity of ice [J/m3/K]


class PrecipHeat(NamedTuple):
    """PRECIP_HEAT + ATM-partition outputs over the land-column grid (ny, nx)."""

    # precip-advected heat (the S1-ENERGY consumers; default-0 before this sprint)
    pahv: jnp.ndarray     # net into vegetation [W/m2]
    pahg: jnp.ndarray     # net into under-canopy ground [W/m2]
    pahb: jnp.ndarray     # net into bare ground [W/m2]
    # ground precip + new-snow geometry (S3 SNOW + S4 WATER consumers)
    qrain: jnp.ndarray    # rain reaching the ground [mm/s]
    qsnow: jnp.ndarray    # snow reaching the ground [mm/s]
    snowhin: jnp.ndarray  # snow-depth increase rate [m/s]
    bdfall: jnp.ndarray   # fresh-snow bulk density [kg/m3]
    fp: jnp.ndarray       # fractional precip area
    fpice: jnp.ndarray    # snow fraction of precip
    rain: jnp.ndarray     # total rainfall [mm/s]
    snow: jnp.ndarray     # total snowfall [mm/s]
    # canopy water (S4 WATER + S1 ENERGY radiation consumers)
    fwet: jnp.ndarray     # wetted/snowed canopy fraction
    canliq: jnp.ndarray   # intercepted liquid [mm]
    canice: jnp.ndarray   # intercepted ice [mm]
    cmc: jnp.ndarray      # total canopy water (canliq + canice) [mm]


def _atm_partition(forcing):
    """ATM (opt_snf=1) precip partition: returns (prcp, rain, snow, fp, fpice, bdfall).

    PRCP = PRCPCONV + PRCPNONC (PRCPSHCV = 0 in scope). All [mm/s].
    """
    sfctmp = jnp.asarray(forcing.sfctmp, dtype=jnp.float64)
    prcpconv = jnp.asarray(forcing.prcpconv, dtype=jnp.float64)
    prcpnonc = jnp.asarray(forcing.prcpnonc, dtype=jnp.float64)

    prcp = prcpconv + prcpnonc
    # QPRECC/QPRECL (opt_snf != 4): 10%/90% convective/large-scale split (:1169-1170).
    qprecc = 0.10 * prcp
    qprecl = 0.90 * prcp

    # FP — fractional area receiving precip (Niu05, :1175-1177).
    denom = 10.0 * qprecc + qprecl
    fp = jnp.where((qprecc + qprecl) > 0.0, (qprecc + qprecl) / jnp.where(denom > 0.0, denom, 1.0), 0.0)

    # opt_snf=1 Jordan-1991 ice fraction FPICE (:1183-1194).
    fpice = jnp.where(
        sfctmp > TFRZ + 2.5, 0.0,
        jnp.where(
            sfctmp <= TFRZ + 0.5, 1.0,
            jnp.where(
                sfctmp <= TFRZ + 2.0, 1.0 - (-54.632 + 0.2 * sfctmp), 0.6
            ),
        ),
    )

    # Hedstrom-Pomeroy fresh-snow bulk density (:1216).
    bdfall = jnp.minimum(120.0, 67.92 + 51.25 * jnp.exp((sfctmp - TFRZ) / 2.59))

    rain = prcp * (1.0 - fpice)
    snow = prcp * fpice
    return prcp, rain, snow, fp, fpice, bdfall


def noahmp_precip_heat(
    land_state,
    forcing,
    phen,
    ch2op: jnp.ndarray,
    dt: float,
    is_lake: jnp.ndarray | None = None,
) -> tuple[PrecipHeat, jnp.ndarray, jnp.ndarray]:
    """ATM partition + PRECIP_HEAT (module_sf_noahmplsm.F:1362-1556), vectorised.

    Parameters
    ----------
    land_state : NoahMPLandState  — reads canliq/canice/tv/tg.
    forcing    : NoahMPForcing    — reads sfctmp/uu/vv/prcp*.
    phen       : NoahMPPhenology  — reads elai/esai/fveg (FVEG = SHDMAX for dveg=4).
    ch2op      : (ny, nx) max intercepted water per unit (ELAI+ESAI) [mm].
    dt         : physics timestep [s].
    is_lake    : optional (ny, nx) bool IST==2 mask (snow zeroed if TG>TFRZ).

    Returns ``(PrecipHeat, canliq_new, canice_new)``. The PAH terms are real
    precip-advected heat; default-0 collapses to the no-precip case exactly.
    """

    elai = jnp.asarray(phen.elai, dtype=jnp.float64)
    esai = jnp.asarray(phen.esai, dtype=jnp.float64)
    fveg = jnp.asarray(phen.fveg, dtype=jnp.float64)
    tv = jnp.asarray(land_state.tv, dtype=jnp.float64)
    tg = jnp.asarray(land_state.tg, dtype=jnp.float64)
    sfctmp = jnp.asarray(forcing.sfctmp, dtype=jnp.float64)
    uu = jnp.asarray(forcing.uu, dtype=jnp.float64)
    vv = jnp.asarray(forcing.vv, dtype=jnp.float64)
    canliq = jnp.asarray(land_state.canliq, dtype=jnp.float64)
    canice = jnp.asarray(land_state.canice, dtype=jnp.float64)
    ch2op = jnp.broadcast_to(jnp.asarray(ch2op, dtype=jnp.float64), fveg.shape)
    dt_a = jnp.asarray(dt, dtype=jnp.float64)

    _prcp, rain, snow, fp, fpice, bdfall = _atm_partition(forcing)

    has_canopy = (elai + esai) > 0.0

    # --------------------------- liquid water (:1450-1471) --------------------
    maxliq = fveg * ch2op * (elai + esai)
    maxliq_safe = jnp.maximum(maxliq, 1.0e-06)
    qintr_cap = (fveg * rain) * fp
    qintr_lim = jnp.where(
        maxliq > 0.0,
        (maxliq - canliq) / dt_a * (1.0 - jnp.exp(-rain * dt_a / maxliq_safe)),
        0.0,
    )
    qintr = jnp.minimum(qintr_cap, qintr_lim)
    qintr = jnp.maximum(qintr, 0.0)
    qintr = jnp.where(has_canopy, qintr, 0.0)
    qdripr_canopy = fveg * rain - qintr
    qthror_canopy = (1.0 - fveg) * rain
    canliq_after = jnp.maximum(0.0, canliq + qintr * dt_a)
    # no-canopy branch: all rain throughfall; buried canopy dumps canliq.
    qdripr_noc = jnp.where(canliq > 0.0, canliq / dt_a, 0.0)
    qthror_noc = rain
    canliq_noc = jnp.where(canliq > 0.0, 0.0, canliq)
    qdripr = jnp.where(has_canopy, qdripr_canopy, qdripr_noc)
    qthror = jnp.where(has_canopy, qthror_canopy, qthror_noc)
    canliq = jnp.where(has_canopy, canliq_after, canliq_noc)

    # heat transported by liquid water (:1465-1471)
    pah_ac = fveg * rain * (CWAT / 1000.0) * (sfctmp - tv)
    pah_cg = qdripr * (CWAT / 1000.0) * (tv - tg)
    pah_ag = qthror * (CWAT / 1000.0) * (sfctmp - tg)

    # --------------------------- canopy ice (:1480-1503) ----------------------
    maxsno = fveg * 6.6 * (0.27 + 46.0 / bdfall) * (elai + esai)
    maxsno_safe = jnp.maximum(maxsno, 1.0e-06)
    qints_cap = (fveg * snow) * fp
    qints_lim = jnp.where(
        maxsno > 0.0,
        (maxsno - canice) / dt_a * (1.0 - jnp.exp(-snow * dt_a / maxsno_safe)),
        0.0,
    )
    qints = jnp.minimum(qints_cap, qints_lim)
    qints = jnp.maximum(qints, 0.0)
    qints = jnp.where(has_canopy, qints, 0.0)
    ft = jnp.maximum(0.0, (tv - 270.15) / 1.87e5)
    fv = jnp.sqrt(uu * uu + vv * vv) / 1.56e5
    icedrip = jnp.maximum(0.0, canice) * (fv + ft)
    icedrip = jnp.minimum(canice / dt_a + qints, icedrip)
    qdrips_canopy = (fveg * snow - qints) + icedrip
    qthros_canopy = (1.0 - fveg) * snow
    canice_after = jnp.maximum(0.0, canice + (qints - icedrip) * dt_a)
    qdrips_noc = jnp.where(canice > 0.0, canice / dt_a, 0.0)
    qthros_noc = snow
    canice_noc = jnp.where(canice > 0.0, 0.0, canice)
    qdrips = jnp.where(has_canopy, qdrips_canopy, qdrips_noc)
    qthros = jnp.where(has_canopy, qthros_canopy, qthros_noc)
    canice = jnp.where(has_canopy, canice_after, canice_noc)

    # wetted canopy fraction (:1505-1510)
    fwet = jnp.where(
        canice > 0.0,
        jnp.maximum(0.0, canice) / maxsno_safe,
        jnp.maximum(0.0, canliq) / maxliq_safe,
    )
    fwet = jnp.minimum(fwet, 1.0) ** 0.667

    cmc = canliq + canice

    # heat transported by snow/ice (:1517-1519) — accumulate onto liquid terms.
    pah_ac = pah_ac + fveg * snow * (CICE / 1000.0) * (sfctmp - tv)
    pah_cg = pah_cg + qdrips * (CICE / 1000.0) * (tv - tg)
    pah_ag = pah_ag + qthros * (CICE / 1000.0) * (sfctmp - tg)

    pahv = pah_ac - pah_cg
    pahg = pah_cg
    pahb = pah_ag

    # FVEG renormalisation (:1524-1535): the tile sum multiplies PAHG by FVEG and
    # PAHB by (1-FVEG) later, so divide here for 0<FVEG<1; handle the FVEG<=0 and
    # FVEG>=1 edge cases exactly as WRF.
    mid = (fveg > 0.0) & (fveg < 1.0)
    low = fveg <= 0.0
    high = fveg >= 1.0
    fveg_safe = jnp.where(fveg > 0.0, fveg, 1.0)
    one_m_fveg_safe = jnp.where((1.0 - fveg) > 0.0, 1.0 - fveg, 1.0)
    pahg_mid = pahg / fveg_safe
    pahb_mid = pahb / one_m_fveg_safe
    # default (mid) values:
    pahg_out = jnp.where(mid, pahg_mid, pahg)
    pahb_out = jnp.where(mid, pahb_mid, pahb)
    # FVEG<=0: PAHB = PAHG+PAHB, PAHG=0, PAHV=0 (buried canopy).
    pahb_out = jnp.where(low, pahg + pahb, pahb_out)
    pahg_out = jnp.where(low, 0.0, pahg_out)
    pahv = jnp.where(low, 0.0, pahv)
    # FVEG>=1: PAHB=0.
    pahb_out = jnp.where(high, 0.0, pahb_out)

    # ±20 W/m2 stability clamp (:1538-1543).
    pahv = jnp.clip(pahv, -20.0, 20.0)
    pahg_out = jnp.clip(pahg_out, -20.0, 20.0)
    pahb_out = jnp.clip(pahb_out, -20.0, 20.0)

    # rain/snow on the ground + snow-depth increase (:1546-1554).
    qrain = qdripr + qthror
    qsnow = qdrips + qthros
    snowhin = qsnow / jnp.maximum(bdfall, 1.0e-06)
    if is_lake is not None:
        lake_warm = jnp.asarray(is_lake, dtype=bool) & (tg > TFRZ)
        qsnow = jnp.where(lake_warm, 0.0, qsnow)
        snowhin = jnp.where(lake_warm, 0.0, snowhin)

    out = PrecipHeat(
        pahv=pahv, pahg=pahg_out, pahb=pahb_out,
        qrain=qrain, qsnow=qsnow, snowhin=snowhin, bdfall=bdfall,
        fp=fp, fpice=fpice, rain=rain, snow=snow,
        fwet=fwet, canliq=canliq, canice=canice, cmc=cmc,
    )
    return out, canliq, canice


__all__ = ["PrecipHeat", "noahmp_precip_heat"]
