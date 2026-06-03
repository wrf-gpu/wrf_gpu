"""JAX WSM3 simple-ice microphysics (WRF mp_physics=3).

WRF WSM3 has three prognostic moist leaves in the Registry:
``qv,qc,qr``. The scheme names the latter two arrays ``qci`` and ``qrs`` and
interprets them as cloud/rain in warm air and ice/snow in cold air.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import wsm6_constants as C
from gpuwrf.physics.microphysics_wsm6 import (
    _conden,
    _cpmcal,
    _diffac,
    _diffusivity_limit,
    _effective_radii,
    _nislfv_single,
    _plm_qmi_qpi,
    _plm_remap,
    _precip_out,
    _venfac,
    _wi_interp,
    _xlcal,
)


def _qsat_wsm3(t, p):
    """WSM3 saturation over liquid for warm cells and ice for cold cells."""

    hsub = C.XLS
    hvap = C.XLV0
    cvap = C.CPV
    ttp = C.T0C + 0.01
    dldt = cvap - C.CLIQ
    xa = -dldt / C.RV
    xb = xa + hvap / (C.RV * ttp)
    dldti = cvap - C.CICE
    xai = -dldti / C.RV
    xbi = xai + hsub / (C.RV * ttp)
    tr = ttp / t
    es_ice = C.PSAT * jnp.exp(jnp.log(tr) * xai) * jnp.exp(xbi * (1.0 - tr))
    es_liq = C.PSAT * jnp.exp(jnp.log(tr) * xa) * jnp.exp(xb * (1.0 - tr))
    es = jnp.where(t < ttp, es_ice, es_liq)
    es = jnp.minimum(es, 0.99 * p)
    qs = C.EP2 * es / (p - es)
    return jnp.maximum(qs, C.QMIN)


def _qsat_liquid(t, p):
    qsat1, _ = _qsat_pair_liq_ice(t, p)
    return qsat1


def _qsat_pair_liq_ice(t, p):
    from gpuwrf.physics.microphysics_wsm6 import _qsat_fpvs

    return _qsat_fpvs(t, p)


def _slope_wsm3(qrs, den, denfac, t):
    """WRF slope_wsm3: rain distribution in warm cells, snow in cold cells."""

    rain = t >= C.T0C
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)

    lamdar = jnp.sqrt(jnp.sqrt(C.PIDN0R / (qrs * den)))
    rsl_r = jnp.where(qrs <= C.QCRMIN, C.RSLOPERMAX, 1.0 / lamdar)
    rslb_r = jnp.where(qrs <= C.QCRMIN, C.RSLOPERBMAX, rsl_r ** C.BVTR)
    rsl2_r = jnp.where(qrs <= C.QCRMIN, C.RSLOPER2MAX, rsl_r * rsl_r)
    rsl3_r = jnp.where(qrs <= C.QCRMIN, C.RSLOPER3MAX, rsl2_r * rsl_r)

    lamdas = jnp.sqrt(jnp.sqrt(C.PIDN0S * n0sfac / (qrs * den)))
    rsl_s = jnp.where(qrs <= C.QCRMIN, C.RSLOPESMAX, 1.0 / lamdas)
    rslb_s = jnp.where(qrs <= C.QCRMIN, C.RSLOPESBMAX, rsl_s ** C.BVTS)
    rsl2_s = jnp.where(qrs <= C.QCRMIN, C.RSLOPES2MAX, rsl_s * rsl_s)
    rsl3_s = jnp.where(qrs <= C.QCRMIN, C.RSLOPES3MAX, rsl2_s * rsl_s)

    rslope = jnp.where(rain, rsl_r, rsl_s)
    rslopeb = jnp.where(rain, rslb_r, rslb_s)
    rslope2 = jnp.where(rain, rsl2_r, rsl2_s)
    rslope3 = jnp.where(rain, rsl3_r, rsl3_s)
    pvt = jnp.where(rain, C.PVTR, C.PVTS)
    vt = jnp.where(qrs <= 0.0, 0.0, pvt * rslopeb * denfac)
    return rslope, rslopeb, rslope2, rslope3, vt


def _nislfv_wsm3(q, den, denfac, tk, dz, ww_in, dt, iter_n):
    """WSM3 PLM sedimentation: arrival velocity recomputed with slope_wsm3."""

    km = q.shape[0]
    allold = jnp.sum(q)
    zi = jnp.concatenate([jnp.zeros(1, q.dtype), jnp.cumsum(dz)])
    wd = ww_in

    def iterate(ww):
        wi = _wi_interp(ww, dz, km)
        wi = _diffusivity_limit(wi, dz, dt, km)
        za = zi - wi * dt
        dza = jnp.concatenate([za[1:km + 1] - za[0:km], (zi[km] - za[km]).reshape(1)])
        qa_cells = q * dz / dza[0:km]
        qr_cells = qa_cells / den
        return za, dza, qa_cells, qr_cells

    def iter_body(n, state):
        ww, was = state
        za, dza, qa_cells, qr_cells = iterate(ww)
        _r, _rb, _r2, _r3, wa = _slope_wsm3(qr_cells, den, denfac, tk)
        wa = jnp.where(n >= 1, 0.5 * (wa + was), wa)
        return 0.5 * (wd + wa), wa

    ww = wd
    was = jnp.zeros(km, dtype=q.dtype)
    if iter_n >= 1:
        ww, was = jax.lax.fori_loop(0, iter_n, iter_body, (ww, was))

    za, dza, qa_cells, _qr_cells = iterate(ww)
    qa = jnp.concatenate([qa_cells, jnp.zeros(1, q.dtype)])
    qmi, qpi = _plm_qmi_qpi(qa, dza, km)
    qn = _plm_remap(qa, dza, qmi, qpi, zi, za, km)
    precip = _precip_out(qa, dza, za, km)
    q_new = jnp.where(allold <= 0.0, q, qn)
    precip = jnp.where(allold <= 0.0, 0.0, precip)
    return q_new, precip


def _wsm3_deposition_block(cold, qci, qrs, q, rh, work1, satdt, prevp, dtcld,
                           rslope, rslopeb, rslope2, rslope3, work2, den, denfac, supcol, supsat, xni):
    eacrs = jnp.exp(0.07 * (-supcol))
    xmi = den * qci / xni
    diameter = jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX)
    vt2i = 1.49e4 * diameter ** 1.31
    vt2s = C.PVTS * rslopeb * denfac
    acrfac = 2.0 * rslope3 + 2.0 * diameter * rslope2 + diameter * diameter * rslope
    pacr_cold = jnp.minimum(
        C.PI * qci * eacrs * C.N0S
        * jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
        * jnp.abs(vt2s - vt2i) * acrfac / 4.0,
        qci / dtcld,
    )
    pacr = jnp.where(cold & (qrs > C.QCRMIN) & (qci > C.QMIN), pacr_cold, 0.0)

    pisd_raw = 4.0 * (C.DICON * jnp.sqrt(xmi)) * xni * (rh - 1.0) / work1
    pisd_neg = jnp.maximum(jnp.maximum(pisd_raw, satdt / 2.0), -qci / dtcld)
    pisd_pos = jnp.minimum(pisd_raw, satdt / 2.0)
    pisd_v = jnp.where(pisd_raw < 0.0, pisd_neg, pisd_pos)
    do_pisd = cold & (qci > 0.0)
    pisd = jnp.where(do_pisd, pisd_v, 0.0)
    ifsat1 = do_pisd & (jnp.abs(pisd) >= jnp.abs(satdt))

    coeres = rslope2 * jnp.sqrt(rslope * rslopeb)
    pres_cold_raw = (
        (rh - 1.0)
        * jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
        * (C.PRECS1 * rslope2 + C.PRECS2 * work2 * coeres)
        / work1
    )
    supice = satdt - pisd
    pres_cold_neg = jnp.maximum(pres_cold_raw, -qrs / dtcld)
    pres_cold_neg = jnp.maximum(jnp.maximum(pres_cold_neg, satdt / 2.0), supice)
    pres_cold_pos = jnp.minimum(jnp.minimum(pres_cold_raw, satdt / 2.0), supice)
    pres_cold = jnp.where(pres_cold_raw < 0.0, pres_cold_neg, pres_cold_pos)
    do_pres_cold = cold & (qrs > 0.0) & jnp.logical_not(ifsat1)
    pres_ice = jnp.where(do_pres_cold, pres_cold, 0.0)
    ifsat2 = ifsat1 | (do_pres_cold & (jnp.abs(pisd + pres_ice) >= jnp.abs(satdt)))

    supice2 = satdt - pisd - pres_ice
    xni0 = 1.0e3 * jnp.exp(0.1 * supcol)
    roqi0 = 4.92e-11 * xni0 ** 1.33
    pgen_raw = jnp.maximum(0.0, (roqi0 / den - jnp.maximum(qci, 0.0)) / dtcld)
    pgen_v = jnp.minimum(jnp.minimum(pgen_raw, satdt), supice2)
    pgen = jnp.where(cold & (supsat > 0.0) & jnp.logical_not(ifsat2), pgen_v, 0.0)

    qimax = C.ROQIMAX / den
    paut_cold = jnp.where(cold & (qci > 0.0), jnp.maximum(0.0, (qci - qimax) / dtcld), 0.0)
    return pacr, pisd, pres_ice, pgen, paut_cold


def _wsm3_column(t, q, qci, qrs, w, den, p, delz, delt):
    qci = jnp.maximum(qci, 0.0)
    qrs = jnp.maximum(qrs, 0.0)
    cpm = _cpmcal(q)
    xl = _xlcal(t)
    rainncv = jnp.zeros((), t.dtype)
    snowncv = jnp.zeros((), t.dtype)
    tstepsnow = jnp.zeros((), t.dtype)
    sr = jnp.zeros((), t.dtype)

    loops = max(int(round(delt / C.DTCLDCR)), 1)
    dtcld = delt / loops
    if delt <= C.DTCLDCR:
        dtcld = delt

    state = (t, q, qci, qrs, cpm, xl, rainncv, snowncv, tstepsnow, sr)
    for _ in range(loops):
        state = _wsm3_minor_loop(state, w, den, p, delz, dtcld)

    t, q, qci, qrs, cpm, xl, rainncv, snowncv, tstepsnow, sr = state
    qc_re = jnp.where(t >= C.T0C, qci, 0.0)
    qi_re = jnp.where(t < C.T0C, qci, 0.0)
    qs_re = jnp.where(t < C.T0C, qrs, 0.0)
    re_qc, re_qi, re_qs = _effective_radii(t, qc_re, qi_re, qs_re, den)
    return {
        "t": t,
        "qv": q,
        "qc": qci,
        "qr": qrs,
        "rainncv": rainncv,
        "snowncv": snowncv,
        "sr": sr,
        "re_cloud": re_qc,
        "re_ice": re_qi,
        "re_snow": re_qs,
    }


def _wsm3_minor_loop(state, w, den, p, delz, dtcld):
    t, q, qci, qrs, cpm, xl, rainncv, snowncv, tstepsnow, sr = state
    qmin = C.QMIN
    denfac = jnp.sqrt(C.DEN0 / den)
    qs = _qsat_wsm3(t, p)
    rh = jnp.maximum(q / qs, qmin)

    temp = den * jnp.maximum(qci, qmin)
    xni = jnp.minimum(jnp.maximum(5.38e7 * jnp.exp(jnp.log(temp) * 0.75), 1.0e3), 1.0e6)

    rslope, rslopeb, rslope2, rslope3, vt = _slope_wsm3(qrs, den, denfac, t)
    denqrs_new, delqrs = _nislfv_wsm3(den * qrs, den, denfac, t, delz, vt, dtcld, 1)
    qrs = jnp.maximum(denqrs_new / den, 0.0)
    fall = denqrs_new * vt / delz
    fall = fall.at[0].set(delqrs / delz[0] / dtcld)

    xmi = den * qci / xni
    diameter = jnp.maximum(C.DICON * jnp.sqrt(xmi), 1.0e-25)
    work1c = jnp.where((t < C.T0C) & (qci > 0.0), 1.49e4 * jnp.exp(jnp.log(diameter) * 1.31), 0.0)
    denqci_new, delqi = _nislfv_single(den * qci, den, denfac, t, delz, work1c, dtcld, 0)
    qci = jnp.maximum(denqci_new / den, 0.0)
    fallc0 = delqi / delz[0] / dtcld

    km = t.shape[0]
    idx = jnp.arange(km)
    mstep = jnp.max(jnp.where(t >= C.T0C, idx, -1))
    active = mstep >= 0
    kk = jnp.clip(mstep, 0, km - 1)
    k1_raw = kk + jnp.where(active & (w[kk] > 0.0), 1, 0)
    k1 = jnp.clip(k1_raw, 0, km - 1)
    active = active & (k1_raw < km)
    qrsci = qrs[k1] + qci[k1]
    do_fm = active & ((qrsci > 0.0) | (fall[kk] > 0.0))
    frzmlt = jnp.minimum(jnp.maximum(-w[k1] * qrsci / delz[k1], -qrsci / dtcld), qrsci / dtcld)
    snomlt = jnp.minimum(jnp.maximum(fall[kk] / den[kk], -qrs[k1] / dtcld), qrs[k1] / dtcld)
    same = k1 == kk
    dtemp = jnp.zeros_like(t)
    dtemp = dtemp.at[k1].add(jnp.where(do_fm, -C.XLF0 / cpm[k1] * frzmlt * dtcld, 0.0))
    dtemp = dtemp.at[kk].add(jnp.where(do_fm & jnp.logical_not(same), -C.XLF0 / cpm[kk] * snomlt * dtcld, 0.0))
    dtemp = dtemp.at[k1].add(jnp.where(do_fm & same, -C.XLF0 / cpm[k1] * snomlt * dtcld, 0.0))
    t = t + dtemp

    fallsum = fall[0]
    fallsum_qsi = jnp.zeros_like(fallsum)
    cold_surface = (C.T0C - t[0]) > 0.0
    fallsum = jnp.where(cold_surface, fallsum + fallc0, fallsum)
    fallsum_qsi = jnp.where(cold_surface, fall[0] + fallc0, fallsum_qsi)
    add = fallsum * delz[0] / C.DENR * dtcld * 1000.0
    rainncv = jnp.where(fallsum > 0.0, rainncv + add, rainncv)
    add_s = fallsum_qsi * delz[0] / C.DENR * dtcld * 1000.0
    tstepsnow = jnp.where(fallsum_qsi > 0.0, tstepsnow + add_s, tstepsnow)
    snowncv = jnp.where(fallsum_qsi > 0.0, snowncv + add_s, snowncv)
    sr = jnp.where(fallsum > 0.0, snowncv / (rainncv + 1.0e-12), sr)

    rslope, rslopeb, rslope2, rslope3, _vt = _slope_wsm3(qrs, den, denfac, t)
    # WSM3 intentionally keeps the saturation array computed at the start of
    # the minor step here. Near the melting level D89 can move T across T0C,
    # but pristine WRF still feeds the stale qs into diffac before pcond.
    work1_liq = _diffac(xl, p, t, den, qs)
    work1_ice = _diffac(C.XLS, p, t, den, qs)
    work1 = jnp.where(t >= C.T0C, work1_liq, work1_ice)
    work2 = _venfac(p, t, den)

    supsat = jnp.maximum(q, qmin) - qs
    satdt = supsat / dtcld
    warm = t >= C.T0C
    paut_warm = jnp.where(qci > C.QC0, jnp.minimum(C.QCK1 * qci ** (7.0 / 3.0), qci / dtcld), 0.0)
    pacr_warm = jnp.where(
        (qrs > C.QCRMIN) & (qci > qmin),
        jnp.minimum(C.PACRR * rslope3 * rslopeb * qci * denfac, qci / dtcld),
        0.0,
    )
    coeres = rslope2 * jnp.sqrt(rslope * rslopeb)
    pres_raw = (rh - 1.0) * (C.PRECR1 * rslope2 + C.PRECR2 * work2 * coeres) / work1
    pres_warm_neg = jnp.maximum(jnp.maximum(pres_raw, -qrs / dtcld), satdt / 2.0)
    pres_warm_pos = jnp.minimum(pres_raw, satdt / 2.0)
    pres_warm = jnp.where(qrs > 0.0, jnp.where(pres_raw < 0.0, pres_warm_neg, pres_warm_pos), 0.0)

    cold = t < C.T0C
    temp = den * jnp.maximum(qci, qmin)
    xni = jnp.minimum(jnp.maximum(5.38e7 * jnp.exp(jnp.log(temp) * 0.75), 1.0e3), 1.0e6)
    pacr_cold, pisd, pres_ice, pgen, paut_cold = _wsm3_deposition_block(
        cold, qci, qrs, q, rh, work1, satdt, pres_warm, dtcld, rslope, rslopeb,
        rslope2, rslope3, work2, den, denfac, C.T0C - t, supsat, xni
    )
    paut = jnp.where(warm, paut_warm, paut_cold)
    pacr = jnp.where(warm, pacr_warm, pacr_cold)
    pres = jnp.where(warm, pres_warm, pres_ice)

    qciik = jnp.maximum(qmin, qci)
    delqci = (paut + pacr - pgen - pisd) * dtcld
    facqci = jnp.where(delqci >= qciik, qciik / delqci, 1.0)
    paut = paut * facqci
    pacr = pacr * facqci
    pgen = pgen * facqci
    pisd = pisd * facqci
    qik = jnp.maximum(qmin, q)
    delq = (pres + pgen + pisd) * dtcld
    facq = jnp.where(delq >= qik, qik / delq, 1.0)
    pres = pres * facq
    pgen = pgen * facq
    pisd = pisd * facq

    workv = -pres - pgen - pisd
    q = q + workv * dtcld
    qci = jnp.maximum(qci - (paut + pacr - pgen - pisd) * dtcld, 0.0)
    qrs = jnp.maximum(qrs + (paut + pacr + pres) * dtcld, 0.0)
    latent = jnp.where(t < C.T0C, C.XLS, xl)
    t = t - latent * workv / cpm * dtcld

    qsat_liq, _ = _qsat_pair_liq_ice(t, p)
    w1 = _conden(t, q, qsat_liq, xl, cpm)
    pcon = jnp.minimum(jnp.maximum(w1, 0.0), jnp.maximum(q, 0.0)) / dtcld
    pcon = jnp.where((qci > 0.0) & (w1 < 0.0) & (t > C.T0C), jnp.maximum(w1, -qci) / dtcld, pcon)
    q = q - pcon * dtcld
    qci = jnp.maximum(qci + pcon * dtcld, 0.0)
    t = t + pcon * xl / cpm * dtcld

    qci = jnp.where(qci <= qmin, 0.0, qci)
    qrs = jnp.where(qrs <= C.QCRMIN, 0.0, qrs)
    return (t, q, qci, qrs, cpm, xl, rainncv, snowncv, tstepsnow, sr)


_wsm3_columns = jax.jit(
    jax.vmap(_wsm3_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None)),
    static_argnums=(8,),
)


def wsm3_run(t, qv, qc, qr, w, den, p, delz, delt):
    """Run WSM3 on batch-major columns shaped ``(ncol, nlev)``."""

    return _wsm3_columns(t, qv, qc, qr, w, den, p, delz, float(delt))


def wsm3_physics_tendency(theta, qv, qc, qr, w, pii, den, p, delz, delt):
    """Return frozen S0-style in-place replacements for WSM3."""

    t = theta * pii
    out = wsm3_run(t, qv, qc, qr, w, den, p, delz, delt)
    return PhysicsTendency(
        state_replacements={
            "theta": out["t"] / pii,
            "qv": out["qv"],
            "qc": out["qc"],
            "qr": out["qr"],
        },
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["snowncv"],
        },
        diagnostics={
            "re_cloud": out["re_cloud"],
            "re_ice": out["re_ice"],
            "re_snow": out["re_snow"],
            "sr": out["sr"],
        },
    )


__all__ = ["wsm3_run", "wsm3_physics_tendency"]
