"""Noah-MP Schaake96 soil hydrology + runoff (Sprint S4).

Ports WATER (module_sf_noahmplsm.F:5954-6261) restricted to the Schaake branch
(opt_run=3) and SOILWATER (:7234-7556), with opt_inf=1 / opt_frz=1 frozen-soil
treatment. NO GROUNDWATER (SIMGM/SIMTOP CUT).

Steps: canopy interception update (CANLIQ/CANICE/FWET), infiltration, Schaake96
surface + subsurface runoff, the Richards-like soil-moisture tridiagonal solve
(SOILWATER), and supercooled-liquid (SH2O <= SMC). Consumes the transpiration /
evaporation sinks from the energy step (S1) as soil-moisture withdrawals.

FULLY PARALLEL to author (savepoints supply the ET inputs as fixtures); integrates
serially with S1 (consumes its ET). Oracle = WRF SMC/SH2O/SMCWTD/SFCRUNOFF/UDRUNOFF
savepoint parity + a water-mass conservation check (the LH 18x over-flux must
collapse to ~1x once evaporation is soil-hydraulic + canopy-resistance limited).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import config

from gpuwrf.contracts.noahmp_state import NSOIL
from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPEtFluxes, NoahMPForcing

config.update("jax_enable_x64", True)

_TFRZ = 273.16
_HFUS = 0.3336e6
_CWAT = 4.188e6
_CICE = 2.094e6
_DENH2O = 1000.0
_DENICE = 917.0
_REFDK = 2.0e-6
_REFKDT = 3.0
_FRZK = 0.15
_SLOPE_TYPE_1 = 0.1
_SLOPETYP = 1   # WRF default subsurface-runoff slope category (opt_run=3)


def _soil_field(value, template: jnp.ndarray) -> jnp.ndarray:
    """Broadcast a soil-vector/field value to ``(NSOIL, ny, nx)``."""

    arr = jnp.asarray(value, dtype=template.dtype)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template.shape)
    if arr.shape[0] != NSOIL:
        raise ValueError(f"expected first dimension {NSOIL}, got {arr.shape}")
    return jnp.broadcast_to(
        arr.reshape((NSOIL,) + (1,) * (template.ndim - 1)),
        template.shape,
    )


def _surface_field(value, template: jnp.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(value, dtype=template.dtype)
    return jnp.broadcast_to(arr, template.shape)


def _category_index(category: jnp.ndarray, size: int) -> jnp.ndarray:
    """Row index into a 1-BASED parameter table (axis-0 length ncat+1, dummy row 0).

    The frozen ``NoahMPParameters`` soil/veg tables keep WRF's 1-based category
    layout (``tables._parse_soilparm`` fills rows 1..ncat; row 0 is an all-zero
    placeholder), so the category id IS the row index — identical to
    ``noahmp_driver._gather_vec`` and the phenology gathers. The previous
    ``category - 1`` shifted every category one row down: ISLTYP=1 (sand) read
    the all-zero row 0 (SMCMAX=0 -> smc/smcmax=inf -> NaN soil moisture, the
    v0.14 d01 LU16 preflight blocker) and every other soil type silently ran
    WATER with the previous category's hydraulic parameters.
    """
    return jnp.clip(category.astype(jnp.int32), 0, max(size - 1, 0))


def _gather_by_category(
    arr: jnp.ndarray,
    category: jnp.ndarray,
    template: jnp.ndarray,
) -> jnp.ndarray:
    """Gather 1-based category table values using WRF's 1-based category ids."""

    idx = _category_index(category, arr.shape[0])
    return jnp.asarray(jnp.take(arr, idx, axis=0), dtype=template.dtype)


def _soil_param(parameters, name: str, static: NoahMPStatic, template: jnp.ndarray, default) -> jnp.ndarray:
    """Return a soil parameter as ``(NSOIL, ny, nx)``.

    The frozen table bundle stores soil values by category, but oracle fixtures
    often inject already-expanded layer fields. This accepts both without
    widening the public interface.
    """

    arr = jnp.asarray(getattr(parameters, name, default), dtype=template.dtype)
    surface_shape = template.shape[1:]
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template.shape)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == 1:
        if arr.shape[0] == NSOIL:
            return _soil_field(arr, template)
        gathered = _gather_by_category(arr, static.isltyp, template[0])
        return jnp.broadcast_to(gathered[jnp.newaxis, ...], template.shape)
    if arr.ndim == 2:
        if arr.shape[0] == NSOIL and arr.shape[1:] == surface_shape:
            return jnp.asarray(arr, dtype=template.dtype)
        if arr.shape[0] == NSOIL:
            idx = _category_index(static.isltyp, arr.shape[1])
            values = [jnp.take(arr[k], idx, axis=0) for k in range(NSOIL)]
            return jnp.stack(values, axis=0).astype(template.dtype)
        if arr.shape[1] == NSOIL:
            idx = _category_index(static.isltyp, arr.shape[0])
            values = [jnp.take(arr[:, k], idx, axis=0) for k in range(NSOIL)]
            return jnp.stack(values, axis=0).astype(template.dtype)
    if arr.ndim == template.ndim and arr.shape[0] == NSOIL:
        return jnp.broadcast_to(arr, template.shape)
    raise ValueError(f"unsupported {name} parameter shape {arr.shape}")


def _veg_param(parameters, name: str, static: NoahMPStatic, template: jnp.ndarray, default) -> jnp.ndarray:
    arr = jnp.asarray(getattr(parameters, name, default), dtype=template.dtype)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template.shape)
    if arr.shape == template.shape:
        return arr
    if arr.ndim == 1:
        if arr.shape[0] == template.shape[0] and arr.shape == template.shape:
            return arr
        return _gather_by_category(arr, static.ivgtyp, template)
    return jnp.broadcast_to(arr, template.shape)


def _geometry(static: NoahMPStatic, template: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    zsoil = _soil_field(static.zsoil, template)
    dzs = _soil_field(static.dzs, template)
    return zsoil, dzs


def _canwater(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    et_fluxes: NoahMPEtFluxes,
    fveg: jnp.ndarray,
    ch2op: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """WRF CANWATER, with ET already provided as mass fluxes."""

    del static
    canliq = jnp.asarray(land_state.canliq)
    canice = jnp.asarray(land_state.canice)
    tv = jnp.asarray(land_state.tv)
    frozen_canopy = tv <= _TFRZ

    ecan = jnp.asarray(et_fluxes.ecan, dtype=canliq.dtype)
    etran = jnp.maximum(jnp.asarray(et_fluxes.etran, dtype=canliq.dtype), 0.0)
    qevac = jnp.where(frozen_canopy, 0.0, jnp.maximum(ecan, 0.0))
    qdewc = jnp.where(frozen_canopy, 0.0, jnp.maximum(-ecan, 0.0))
    qsubc = jnp.where(frozen_canopy, jnp.maximum(ecan, 0.0), 0.0)
    qfroc = jnp.where(frozen_canopy, jnp.maximum(-ecan, 0.0), 0.0)

    maxliq = fveg * ch2op * (land_state.lai + land_state.sai)
    canliq = jnp.maximum(0.0, canliq + (qdewc - jnp.minimum(canliq / dt, qevac)) * dt)
    canliq = jnp.where(canliq <= 1.0e-6, 0.0, canliq)

    bdfall = jnp.minimum(120.0, 67.92 + 51.25 * jnp.exp((forcing.sfctmp - _TFRZ) / 2.59))
    maxsno = fveg * 6.6 * (0.27 + 46.0 / jnp.maximum(bdfall, 1.0e-12)) * (land_state.lai + land_state.sai)
    canice = jnp.maximum(0.0, canice + (qfroc - jnp.minimum(canice / dt, qsubc)) * dt)
    canice = jnp.where(canice <= 1.0e-6, 0.0, canice)

    fwet_ice = jnp.maximum(0.0, canice) / jnp.maximum(maxsno, 1.0e-6)
    fwet_liq = jnp.maximum(0.0, canliq) / jnp.maximum(maxliq, 1.0e-6)
    fwet = jnp.where((canice > 0.0) & (canice >= canliq), fwet_ice, fwet_liq)
    fwet = jnp.minimum(fwet, 1.0) ** 0.667

    cmc = canliq + canice
    qmeltc = jnp.where(
        (canice > 1.0e-6) & (tv > _TFRZ),
        jnp.minimum(canice / dt, (tv - _TFRZ) * _CICE * canice / _DENICE / (dt * _HFUS)),
        0.0,
    )
    canice_melt = jnp.maximum(0.0, canice - qmeltc * dt)
    canliq_melt = jnp.maximum(0.0, cmc - canice_melt)
    tv_melt = fwet * _TFRZ + (1.0 - fwet) * tv
    canice = jnp.where(qmeltc > 0.0, canice_melt, canice)
    canliq = jnp.where(qmeltc > 0.0, canliq_melt, canliq)
    tv = jnp.where(qmeltc > 0.0, tv_melt, tv)

    cmc = canliq + canice
    qfrzc = jnp.where(
        (canliq > 1.0e-6) & (tv < _TFRZ),
        jnp.minimum(canliq / dt, (_TFRZ - tv) * _CWAT * canliq / _DENH2O / (dt * _HFUS)),
        0.0,
    )
    canliq_frz = jnp.maximum(0.0, canliq - qfrzc * dt)
    canice_frz = jnp.maximum(0.0, cmc - canliq_frz)
    tv_frz = fwet * _TFRZ + (1.0 - fwet) * tv
    canliq = jnp.where(qfrzc > 0.0, canliq_frz, canliq)
    canice = jnp.where(qfrzc > 0.0, canice_frz, canice)
    tv = jnp.where(qfrzc > 0.0, tv_frz, tv)

    return canliq, canice, fwet, tv, etran


def _wdfcnd1(
    smc: jnp.ndarray,
    fcr: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    factr = jnp.maximum(0.01, smc / smcmax)
    wdf = dwsat * factr ** (bexp + 2.0) * (1.0 - fcr)
    wcnd = dksat * factr ** (2.0 * bexp + 3.0) * (1.0 - fcr)
    return wdf, wcnd


def _wdfcnd2(
    smc: jnp.ndarray,
    sice: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    factr1 = jnp.minimum(0.05 / smcmax, jnp.maximum(0.01, smc / smcmax))
    factr2 = jnp.maximum(0.01, smc / smcmax)
    expon = bexp + 2.0
    wdf = dwsat * factr2**expon
    vkwgt = 1.0 / (1.0 + (500.0 * sice) ** 3.0)
    wdf_ice = vkwgt * wdf + (1.0 - vkwgt) * dwsat * factr1**expon
    wdf = jnp.where(sice > 0.0, wdf_ice, wdf)
    wcnd = dksat * factr2 ** (2.0 * bexp + 3.0)
    return wdf, wcnd


def _infil(
    qinsur: jnp.ndarray,
    dt: jnp.ndarray,
    zsoil: jnp.ndarray,
    sh2o: jnp.ndarray,
    sice: jnp.ndarray,
    sicemax: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    smcwlt: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
    kdt: jnp.ndarray,
    frzx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """WRF INFIL for opt_run=3 / opt_inf=1. Returns ``(pddum, runsrf)`` in m/s."""

    smcav = smcmax[0] - smcwlt[0]
    dmax0 = -zsoil[0] * smcav
    dice = -zsoil[0] * sice[0]
    dmax0 = dmax0 * (1.0 - (sh2o[0] + sice[0] - smcwlt[0]) / smcav)
    dd = dmax0
    for k in range(1, NSOIL):
        thickness = zsoil[k - 1] - zsoil[k]
        dice = dice + thickness * sice[k]
        dmax = thickness * smcav
        dmax = dmax * (1.0 - (sh2o[k] + sice[k] - smcwlt[k]) / smcav)
        dd = dd + dmax

    dt1 = dt / 86400.0
    val = 1.0 - jnp.exp(-kdt * dt1)
    ddt = dd * val
    px = jnp.maximum(0.0, qinsur * dt)
    infmax = (px * (ddt / (px + ddt))) / dt

    acrt = 3.0 * frzx / dice
    frozen_sum = 1.0 + acrt + 0.5 * acrt**2
    frozen_factor = 1.0 - jnp.exp(-acrt) * frozen_sum
    fcr = jnp.where(dice > 1.0e-2, frozen_factor, 1.0)
    infmax = infmax * fcr

    _, wcnd = _wdfcnd2(sh2o[0], sicemax, bexp[0], smcmax[0], dksat[0], dwsat[0])
    infmax = jnp.maximum(infmax, wcnd)
    infmax = jnp.minimum(infmax, px / dt)
    runsrf = jnp.maximum(0.0, qinsur - infmax)
    pddum = qinsur - runsrf
    active = qinsur > 0.0
    return jnp.where(active, pddum, 0.0), jnp.where(active, runsrf, 0.0)


def _srt(
    zsoil: jnp.ndarray,
    dt: jnp.ndarray,
    pddum: jnp.ndarray,
    etrani: jnp.ndarray,
    qseva: jnp.ndarray,
    sh2o: jnp.ndarray,
    smc: jnp.ndarray,
    fcr: jnp.ndarray,
    fcrmax: jnp.ndarray,
    slope: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del dt, sh2o, fcrmax
    smx = smc
    wdf, wcnd = _wdfcnd1(smx, fcr, bexp, smcmax, dksat, dwsat)

    denom0 = -zsoil[0]
    temp0 = -zsoil[1]
    ddz0 = 2.0 / temp0
    dsmdz0 = 2.0 * (smx[0] - smx[1]) / temp0
    wflux0 = wdf[0] * dsmdz0 + wcnd[0] - pddum + etrani[0] + qseva

    denom1 = zsoil[0] - zsoil[1]
    temp1 = zsoil[0] - zsoil[2]
    ddz1 = 2.0 / temp1
    dsmdz1 = 2.0 * (smx[1] - smx[2]) / temp1
    wflux1 = wdf[1] * dsmdz1 + wcnd[1] - wdf[0] * dsmdz0 - wcnd[0] + etrani[1]

    denom2 = zsoil[1] - zsoil[2]
    temp2 = zsoil[1] - zsoil[3]
    ddz2 = 2.0 / temp2
    dsmdz2 = 2.0 * (smx[2] - smx[3]) / temp2
    wflux2 = wdf[2] * dsmdz2 + wcnd[2] - wdf[1] * dsmdz1 - wcnd[1] + etrani[2]

    denom3 = zsoil[2] - zsoil[3]
    qdrain = slope * wcnd[3]
    wflux3 = -(wdf[2] * dsmdz2) - wcnd[2] + etrani[3] + qdrain

    denom = jnp.stack([denom0, denom1, denom2, denom3], axis=0)
    ddz = jnp.stack([ddz0, ddz1, ddz2, jnp.zeros_like(ddz2)], axis=0)
    wflux = jnp.stack([wflux0, wflux1, wflux2, wflux3], axis=0)

    ai0 = jnp.zeros_like(wflux0)
    bi0 = wdf[0] * ddz[0] / denom[0]
    ci0 = -bi0
    ai1 = -wdf[0] * ddz[0] / denom[1]
    ci1 = -wdf[1] * ddz[1] / denom[1]
    bi1 = -(ai1 + ci1)
    ai2 = -wdf[1] * ddz[1] / denom[2]
    ci2 = -wdf[2] * ddz[2] / denom[2]
    bi2 = -(ai2 + ci2)
    ai3 = -wdf[2] * ddz[2] / denom[3]
    ci3 = jnp.zeros_like(ai3)
    bi3 = -(ai3 + ci3)

    ai = jnp.stack([ai0, ai1, ai2, ai3], axis=0)
    bi = jnp.stack([bi0, bi1, bi2, bi3], axis=0)
    ci = jnp.stack([ci0, ci1, ci2, ci3], axis=0)
    rhstt = wflux / (-denom)
    return rhstt, ai, bi, ci, qdrain, wcnd


def _rosr12(ai: jnp.ndarray, bi: jnp.ndarray, ci: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    c = ci.at[NSOIL - 1].set(0.0)
    p0 = -c[0] / bi[0]
    d0 = rhs[0] / bi[0]
    p1 = -c[1] / (bi[1] + ai[1] * p0)
    d1 = (rhs[1] - ai[1] * d0) / (bi[1] + ai[1] * p0)
    p2 = -c[2] / (bi[2] + ai[2] * p1)
    d2 = (rhs[2] - ai[2] * d1) / (bi[2] + ai[2] * p1)
    p3 = -c[3] / (bi[3] + ai[3] * p2)
    d3 = (rhs[3] - ai[3] * d2) / (bi[3] + ai[3] * p2)
    x3 = d3
    x2 = p2 * x3 + d2
    x1 = p1 * x2 + d1
    x0 = p0 * x1 + d0
    del p3
    return jnp.stack([x0, x1, x2, x3], axis=0)


def _sstep(
    dt: jnp.ndarray,
    dzs: jnp.ndarray,
    sice: jnp.ndarray,
    sh2o: jnp.ndarray,
    smc: jnp.ndarray,
    ai: jnp.ndarray,
    bi: jnp.ndarray,
    ci: jnp.ndarray,
    rhstt: jnp.ndarray,
    smcmax: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del smc
    ai_dt = ai * dt
    bi_dt = 1.0 + bi * dt
    ci_dt = ci * dt
    rhstt_dt = rhstt * dt

    delta = _rosr12(ai_dt, bi_dt, ci_dt, rhstt_dt)
    sh2o = sh2o + delta

    epore3 = jnp.maximum(1.0e-4, smcmax[3] - sice[3])
    wplus3 = jnp.maximum(sh2o[3] - epore3, 0.0) * dzs[3]
    sh2o = sh2o.at[3].set(jnp.minimum(epore3, sh2o[3]))
    sh2o = sh2o.at[2].add(wplus3 / dzs[2])

    epore2 = jnp.maximum(1.0e-4, smcmax[2] - sice[2])
    wplus2 = jnp.maximum(sh2o[2] - epore2, 0.0) * dzs[2]
    sh2o = sh2o.at[2].set(jnp.minimum(epore2, sh2o[2]))
    sh2o = sh2o.at[1].add(wplus2 / dzs[1])

    epore1 = jnp.maximum(1.0e-4, smcmax[1] - sice[1])
    wplus1 = jnp.maximum(sh2o[1] - epore1, 0.0) * dzs[1]
    sh2o = sh2o.at[1].set(jnp.minimum(epore1, sh2o[1]))
    sh2o = sh2o.at[0].add(wplus1 / dzs[0])

    epore0 = jnp.maximum(1.0e-4, smcmax[0] - sice[0])
    wplus0 = jnp.maximum(sh2o[0] - epore0, 0.0) * dzs[0]
    sh2o_base = sh2o.at[0].set(jnp.minimum(epore0, sh2o[0]))

    sh2o_true = sh2o_base.at[1].add(wplus0 / dzs[1])
    epore1b = jnp.maximum(1.0e-4, smcmax[1] - sice[1])
    wplus1b = jnp.maximum(sh2o_true[1] - epore1b, 0.0) * dzs[1]
    sh2o_true = sh2o_true.at[1].set(jnp.minimum(epore1b, sh2o_true[1]))
    sh2o_true = sh2o_true.at[2].add(wplus1b / dzs[2])

    epore2b = jnp.maximum(1.0e-4, smcmax[2] - sice[2])
    wplus2b = jnp.maximum(sh2o_true[2] - epore2b, 0.0) * dzs[2]
    sh2o_true = sh2o_true.at[2].set(jnp.minimum(epore2b, sh2o_true[2]))
    sh2o_true = sh2o_true.at[3].add(wplus2b / dzs[3])

    epore3b = jnp.maximum(1.0e-4, smcmax[3] - sice[3])
    wplus3b = jnp.maximum(sh2o_true[3] - epore3b, 0.0) * dzs[3]
    sh2o_true = sh2o_true.at[3].set(jnp.minimum(epore3b, sh2o_true[3]))

    mask = wplus0 > 0.0
    sh2o = jnp.where(mask[jnp.newaxis, ...], sh2o_true, sh2o_base)
    wplus = jnp.where(mask, wplus3b, wplus0)
    smc = sh2o + sice
    return sh2o, smc, wplus


def _soilwater(
    qinsur: jnp.ndarray,
    qseva: jnp.ndarray,
    etrani: jnp.ndarray,
    sh2o: jnp.ndarray,
    smc: jnp.ndarray,
    sice: jnp.ndarray,
    zsoil: jnp.ndarray,
    dzs: jnp.ndarray,
    smcwtd: jnp.ndarray,
    bexp: jnp.ndarray,
    smcmax: jnp.ndarray,
    smcref: jnp.ndarray,
    smcwlt: jnp.ndarray,
    dksat: jnp.ndarray,
    dwsat: jnp.ndarray,
    kdt: jnp.ndarray,
    frzx: jnp.ndarray,
    slope: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del smcref, smcwtd
    epore = jnp.maximum(1.0e-4, smcmax - sice)
    rsat = jnp.sum(jnp.maximum(0.0, sh2o - epore) * dzs, axis=0)
    sh2o = jnp.minimum(epore, sh2o)

    ice_frac = jnp.minimum(1.0, sice / smcmax)
    fcr = jnp.maximum(0.0, jnp.exp(-4.0 * (1.0 - ice_frac)) - jnp.exp(-4.0)) / (1.0 - jnp.exp(-4.0))
    sicemax = jnp.max(sice, axis=0)
    fcrmax = jnp.max(fcr, axis=0)

    pddum0, _ = _infil(qinsur, dt, zsoil, sh2o, sice, sicemax, bexp, smcmax, smcwlt, dksat, dwsat, kdt, frzx)
    niter = jnp.where(pddum0 * dt > dzs[0] * smcmax[0], 6.0, 3.0)
    dtfine = dt / niter

    qdrain_save = jnp.zeros_like(qinsur)
    runsrf_save = jnp.zeros_like(qinsur)

    for iter_idx in range(6):
        active = iter_idx < niter
        pddum, runsrf = _infil(qinsur, dtfine, zsoil, sh2o, sice, sicemax, bexp, smcmax, smcwlt, dksat, dwsat, kdt, frzx)
        pddum = jnp.where((qinsur > 0.0) & active, pddum, 0.0)
        runsrf = jnp.where((qinsur > 0.0) & active, runsrf, 0.0)
        rhstt, ai, bi, ci, qdrain, _ = _srt(
            zsoil, dtfine, pddum, etrani, qseva, sh2o, smc, fcr, fcrmax,
            slope, bexp, smcmax, dksat, dwsat,
        )
        sh2o_next, smc_next, wplus = _sstep(dtfine, dzs, sice, sh2o, smc, ai, bi, ci, rhstt, smcmax)
        sh2o = jnp.where(active[jnp.newaxis, ...], sh2o_next, sh2o)
        smc = jnp.where(active[jnp.newaxis, ...], smc_next, smc)
        rsat = rsat + jnp.where(active, wplus, 0.0)
        qdrain_save = qdrain_save + jnp.where(active, qdrain, 0.0)
        runsrf_save = runsrf_save + runsrf

    qdrain = qdrain_save / niter
    runsrf = runsrf_save / niter
    runsrf = runsrf * 1000.0 + rsat * 1000.0 / dt
    qdrain = qdrain * 1000.0

    mliq = sh2o * dzs * 1000.0
    runsub = jnp.zeros_like(qinsur)
    for iz in range(NSOIL - 1):
        xs = jnp.where(mliq[iz] < 0.0, 0.01 - mliq[iz], 0.0)
        mliq = mliq.at[iz].add(xs)
        mliq = mliq.at[iz + 1].add(-xs)

    xs = jnp.where(mliq[NSOIL - 1] < 0.01, 0.01 - mliq[NSOIL - 1], 0.0)
    mliq = mliq.at[NSOIL - 1].add(xs)
    runsub = runsub - xs / dt
    sh2o = mliq / (dzs * 1000.0)
    smc = sh2o + sice
    return sh2o, smc, runsrf, qdrain, runsub


def noahmp_water_hydro(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    et_fluxes: NoahMPEtFluxes,
    dt: float,
) -> NoahMPLandState:
    """Advance soil/canopy water one ``dt`` (Schaake96).

    Returns the land carry with SMC/SH2O/SMCWTD/CANLIQ/CANICE/FWET/SFCRUNOFF/
    UDRUNOFF updated; thermal and snow fields untouched (those are S2/S3).
    """

    smc = jnp.asarray(land_state.smois)
    sh2o = jnp.asarray(land_state.sh2o)
    surface = jnp.asarray(land_state.smcwtd)
    dt_arr = jnp.asarray(dt, dtype=smc.dtype)
    parameters = static.parameters

    zsoil, dzs = _geometry(static, smc)
    bexp = _soil_param(parameters, "bexp", static, smc, 5.0)
    smcmax = _soil_param(parameters, "smcmax", static, smc, 0.45)
    smcref = _soil_param(parameters, "smcref", static, smc, 0.35)
    smcwlt = _soil_param(parameters, "smcwlt", static, smc, 0.10)
    dksat = _soil_param(parameters, "dksat", static, smc, 1.0e-6)
    dwsat = _soil_param(parameters, "dwsat", static, smc, 1.0e-5)

    sice = jnp.maximum(0.0, smc - sh2o)

    ch2op = _veg_param(parameters, "ch2op", static, surface, 0.1)
    # FVEG (dveg=4) = SHDMAX, a per-column wrfinput field on NoahMPStatic (= VEGMAX/100),
    # NOT an MPTABLE parameter (arbiter: module_sf_noahmplsm.F:864). Must match the
    # phenology/energy FVEG. Fall back: instantaneous SHDFAC, then 1.0 (defensive).
    fveg_src = static.shdmax if static.shdmax is not None else static.shdfac
    if fveg_src is None:
        fveg = jnp.broadcast_to(jnp.asarray(1.0, dtype=surface.dtype), surface.shape)
    else:
        fveg = jnp.broadcast_to(
            jnp.asarray(fveg_src, dtype=surface.dtype), surface.shape
        )
        fveg = jnp.where(fveg <= 0.05, 0.05, fveg)
    ivgtyp = jnp.asarray(static.ivgtyp)
    fveg = jnp.where((ivgtyp == 25) | (ivgtyp == 26) | (ivgtyp == 27), 0.0, fveg)
    canliq, canice, fwet, tv, etran = _canwater(land_state, forcing, static, et_fluxes, fveg, ch2op, dt_arr)

    ground_et = jnp.where(jnp.abs(et_fluxes.edir) > 0.0, et_fluxes.edir, et_fluxes.qseva)
    qvap = jnp.maximum(jnp.asarray(ground_et, dtype=surface.dtype), 0.0)
    qdew = jnp.maximum(-jnp.asarray(ground_et, dtype=surface.dtype), 0.0)
    has_snow = land_state.sneqv > 0.0
    qsnsubl = jnp.where(has_snow, jnp.minimum(qvap, land_state.sneqv / dt_arr), 0.0)
    qseva_mm_s = qvap - qsnsubl
    qsdew_mm_s = jnp.where(has_snow, 0.0, qdew)

    frozen_ground = land_state.tg <= _TFRZ
    sice0_frozen = sice[0] + (qsdew_mm_s - qseva_mm_s) * dt_arr / (dzs[0] * 1000.0)
    sh2o0_frozen = sh2o[0] + jnp.minimum(sice0_frozen, 0.0)
    sice0_frozen = jnp.maximum(sice0_frozen, 0.0)
    sh2o = sh2o.at[0].set(jnp.where(frozen_ground, sh2o0_frozen, sh2o[0]))
    sice = sice.at[0].set(jnp.where(frozen_ground, sice0_frozen, sice[0]))
    qseva_mm_s = jnp.where(frozen_ground, 0.0, qseva_mm_s)
    qsdew_mm_s = jnp.where(frozen_ground, 0.0, qsdew_mm_s)
    smc = sh2o + sice

    qrain_total = jnp.maximum(forcing.prcpconv, 0.0) + jnp.maximum(forcing.prcpnonc, 0.0)
    solid_prcp = jnp.maximum(et_fluxes.qsnow, forcing.prcpsnow + forcing.prcpgrpl + forcing.prcphail)
    qrain = jnp.maximum(qrain_total - jnp.maximum(solid_prcp, 0.0), 0.0)
    qinsur_mm_s = jnp.maximum(et_fluxes.qmelt, 0.0) + qsdew_mm_s + jnp.where(land_state.isnow == 0, qrain, 0.0)
    qinsur = qinsur_mm_s * 0.001
    qseva = qseva_mm_s * 0.001

    btrani = _soil_field(et_fluxes.btrani, smc)
    etrani = jnp.maximum(etran, 0.0)[jnp.newaxis, ...] * btrani * 0.001

    kdt_default = _REFKDT * dksat[0] / _REFDK
    smcref_safe = jnp.maximum(smcref[0], 1.0e-12)
    frzx_default = _FRZK * (smcmax[0] / smcref_safe) * (0.412 / 0.468)
    kdt = _surface_field(getattr(parameters, "kdt", kdt_default), surface)
    frzx = _surface_field(getattr(parameters, "frzx", frzx_default), surface)
    # SLOPE: the frozen S0b NoahMPParameters carries the per-slope-type table
    # (nslope+1,); WRF gathers it by SLOPETYP (TRANSFER_MP_PARAMETERS, opt_run=3
    # default SLOPETYP=1). A pre-gathered scalar (oracle fixtures) is used as-is.
    slope_raw = jnp.asarray(getattr(parameters, "slope", _SLOPE_TYPE_1), dtype=surface.dtype)
    if slope_raw.ndim == 1 and slope_raw.shape[0] > 1:
        slope_val = slope_raw[jnp.clip(jnp.int32(_SLOPETYP), 0, slope_raw.shape[0] - 1)]
    else:
        slope_val = slope_raw
    slope = _surface_field(slope_val, surface)

    sh2o, smc, runsrf_mm_s, qdrain_mm_s, runsub_mm_s = _soilwater(
        qinsur, qseva, etrani, sh2o, smc, sice, zsoil, dzs, land_state.smcwtd,
        bexp, smcmax, smcref, smcwlt, dksat, dwsat, kdt, frzx, slope, dt_arr,
    )

    runsrf_m = runsrf_mm_s * dt_arr * 0.001
    runsub_m = (runsub_mm_s + qdrain_mm_s) * dt_arr * 0.001

    return land_state.replace(
        smois=smc,
        sh2o=sh2o,
        canliq=canliq,
        canice=canice,
        fwet=fwet,
        tv=tv,
        sfcrunoff=land_state.sfcrunoff + runsrf_m,
        udrunoff=land_state.udrunoff + runsub_m,
    )


__all__ = ["noahmp_water_hydro"]
