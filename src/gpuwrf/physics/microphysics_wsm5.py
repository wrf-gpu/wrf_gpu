"""JAX WSM5 single-moment 5-class microphysics (WRF mp_physics=4).

Faithful column port of WRF ``phys/module_mp_wsm5.F``. WSM5 is the classic
single-moment mixed-ice scheme with active moist leaves ``qv,qc,qr,qi,qs``:
rain and snow sediment separately, there is no graupel state or accumulator.
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
    _effective_radii,
    _nislfv_double,
    _nislfv_single,
    _qsat_fpvs,
    _slope_wsm6,
    _venfac,
    _xlcal,
    _xka,
)


def _wsm5_deposition_block(
    cold,
    qi,
    qs,
    q,
    qmin,
    rh2,
    work1_2,
    satdt2,
    prevp,
    dtcld,
    n0sfac,
    rslope,
    rslopeb,
    rslope2,
    work2,
    den,
    supcol,
    supsat2,
    xni,
):
    """WSM5 sequential ice/snow deposition block with WRF's ifsat short-circuit."""

    xmi = den * qi / xni
    diameter = C.DICON * jnp.sqrt(xmi)

    pidep_raw = 4.0 * diameter * xni * (rh2 - 1.0) / work1_2
    supice1 = satdt2 - prevp
    pidep_neg = jnp.maximum(jnp.maximum(pidep_raw, satdt2 * 0.5), supice1)
    pidep_neg = jnp.maximum(pidep_neg, -qi / dtcld)
    pidep_pos = jnp.minimum(jnp.minimum(pidep_raw, satdt2 * 0.5), supice1)
    pidep_v = jnp.where(pidep_raw < 0.0, pidep_neg, pidep_pos)
    do_pidep = cold & (qi > 0.0)
    pidep = jnp.where(do_pidep, pidep_v, 0.0)
    ifsat1 = do_pidep & (jnp.abs(prevp + pidep) >= jnp.abs(satdt2))

    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psdep_raw = (
        (rh2 - 1.0)
        * n0sfac
        * (C.PRECS1 * rslope2[:, 1] + C.PRECS2 * work2 * coeres_s)
        / work1_2
    )
    supice2 = satdt2 - prevp - pidep
    psdep_neg = jnp.maximum(psdep_raw, -qs / dtcld)
    psdep_neg = jnp.maximum(jnp.maximum(psdep_neg, satdt2 * 0.5), supice2)
    psdep_pos = jnp.minimum(jnp.minimum(psdep_raw, satdt2 * 0.5), supice2)
    psdep_v = jnp.where(psdep_raw < 0.0, psdep_neg, psdep_pos)
    do_psdep = cold & (qs > 0.0) & jnp.logical_not(ifsat1)
    psdep = jnp.where(do_psdep, psdep_v, 0.0)
    ifsat2 = ifsat1 | (do_psdep & (jnp.abs(prevp + pidep + psdep) >= jnp.abs(satdt2)))

    supice3 = satdt2 - prevp - pidep - psdep
    xni0 = 1.0e3 * jnp.exp(0.1 * supcol)
    roqi0 = 4.92e-11 * xni0 ** 1.33
    pigen_raw = jnp.maximum(0.0, (roqi0 / den - jnp.maximum(qi, 0.0)) / dtcld)
    pigen_v = jnp.minimum(jnp.minimum(pigen_raw, satdt2), supice3)
    do_pigen = cold & (supsat2 > 0.0) & jnp.logical_not(ifsat2)
    pigen = jnp.where(do_pigen, pigen_v, 0.0)

    qimax = C.ROQIMAX / den
    psaut = jnp.where(cold & (qi > 0.0), jnp.maximum(0.0, (qi - qimax) / dtcld), 0.0)
    return pidep, psdep, pigen, psaut


def _wsm5_column(t, q, qc, qr, qi, qs, den, p, delz, delt):
    km = t.shape[0]
    qmin = C.QMIN

    qc = jnp.maximum(qc, 0.0)
    qr = jnp.maximum(qr, 0.0)
    qi = jnp.maximum(qi, 0.0)
    qs = jnp.maximum(qs, 0.0)

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

    state = (t, q, qc, qr, qi, qs, cpm, xl, rainncv, snowncv, tstepsnow, sr)
    for _ in range(loops):
        state = _wsm5_minor_loop(state, den, p, delz, dtcld)

    t, q, qc, qr, qi, qs, cpm, xl, rainncv, snowncv, tstepsnow, sr = state
    re_qc, re_qi, re_qs = _effective_radii(t, qc, qi, qs, den)
    return {
        "t": t,
        "qv": q,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "rainncv": rainncv,
        "snowncv": snowncv,
        "sr": sr,
        "re_cloud": re_qc,
        "re_ice": re_qi,
        "re_snow": re_qs,
    }


def _wsm5_minor_loop(state, den, p, delz, dtcld):
    t, q, qc, qr, qi, qs, cpm, xl, rainncv, snowncv, tstepsnow, sr = state
    qmin = C.QMIN

    denfac = jnp.sqrt(C.DEN0 / den)
    qsat1, qsat2 = _qsat_fpvs(t, p)
    rh1 = jnp.maximum(q / qsat1, qmin)
    rh2 = jnp.maximum(q / qsat2, qmin)

    temp = den * jnp.maximum(qi, qmin)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    xni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)

    qg0 = jnp.zeros_like(qs)
    rslope, rslopeb, rslope2, rslope3, vt = _slope_wsm6(qr, qs, qg0, den, denfac, t)
    workr = jnp.where(qr <= 0.0, 0.0, vt[:, 0])
    works = jnp.where(qs <= 0.0, 0.0, vt[:, 1])

    denqrs1_new, delqrs1 = _nislfv_single(den * qr, den, denfac, t, delz, workr, dtcld, 1)
    qs_new, _qg_new, delqrs2, _delqg = _nislfv_double(den * qs, jnp.zeros_like(qs), den, denfac, t, delz, works, dtcld, 1)
    qr = jnp.maximum(denqrs1_new / den, 0.0)
    qs = jnp.maximum(qs_new / den, 0.0)
    fall1 = denqrs1_new * workr / delz
    fall2 = qs_new * works / delz
    fall1 = fall1.at[0].set(delqrs1 / delz[0] / dtcld)
    fall2 = fall2.at[0].set(delqrs2 / delz[0] / dtcld)

    rslope, rslopeb, rslope2, rslope3, _vt2 = _slope_wsm6(qr, qs, qg0, den, denfac, t)

    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    warm = t > C.T0C
    work2 = _venfac(p, t, den)
    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psmlt = (
        _xka(t, den)
        / C.XLF0
        * (C.T0C - t)
        * C.PI
        / 2.0
        * n0sfac
        * (C.PRECS1 * rslope2[:, 1] + C.PRECS2 * work2 * coeres_s)
        / den
    )
    psmlt = jnp.minimum(jnp.maximum(psmlt * dtcld, -qs), 0.0)
    do_psmlt = warm & (qs > 0.0)
    qs = jnp.where(do_psmlt, qs + psmlt, qs)
    qr = jnp.where(do_psmlt, qr - psmlt, qr)
    t = jnp.where(do_psmlt, t + C.XLF0 / cpm * psmlt, t)

    xmi = den * qi / xni
    diameter = jnp.maximum(jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX), 1.0e-25)
    work1c = jnp.where(qi <= 0.0, 0.0, 1.49e4 * jnp.exp(jnp.log(diameter) * 1.31))
    denqci_new, delqi = _nislfv_single(den * qi, den, denfac, t, delz, work1c, dtcld, 0)
    qi = jnp.maximum(denqci_new / den, 0.0)
    fallc0 = delqi / delz[0] / dtcld

    fallsum = fall1[0] + fall2[0] + fallc0
    fallsum_qsi = fall2[0] + fallc0
    add = fallsum * delz[0] / C.DENR * dtcld * 1000.0
    rainncv = jnp.where(fallsum > 0.0, rainncv + add, rainncv)
    add_s = fallsum_qsi * delz[0] / C.DENR * dtcld * 1000.0
    tstepsnow = jnp.where(fallsum_qsi > 0.0, tstepsnow + add_s, tstepsnow)
    snowncv = jnp.where(fallsum_qsi > 0.0, snowncv + add_s, snowncv)
    sr = jnp.where(fallsum > 0.0, snowncv / (rainncv + 1.0e-12), sr)

    supcol = C.T0C - t
    xlf = C.XLS - xl
    xlf = jnp.where(supcol < 0.0, C.XLF0, xlf)
    do_im = (supcol < 0.0) & (qi > 0.0)
    qc = jnp.where(do_im, qc + qi, qc)
    t = jnp.where(do_im, t - xlf / cpm * qi, t)
    qi = jnp.where(do_im, 0.0, qi)
    do_hmf = (supcol > 40.0) & (qc > 0.0)
    qi = jnp.where(do_hmf, qi + qc, qi)
    t = jnp.where(do_hmf, t + xlf / cpm * qc, t)
    qc = jnp.where(do_hmf, 0.0, qc)
    supcolt = jnp.minimum(supcol, 50.0)
    pfrzdtc = jnp.minimum(
        C.PFRZ1 * (jnp.exp(C.PFRZ2 * supcolt) - 1.0) * den / C.DENR / C.XNCR * qc * qc * dtcld,
        qc,
    )
    do_htf = (supcol > 0.0) & (qc > 0.0)
    qi = jnp.where(do_htf, qi + pfrzdtc, qi)
    t = jnp.where(do_htf, t + xlf / cpm * pfrzdtc, t)
    qc = jnp.where(do_htf, qc - pfrzdtc, qc)
    pfrzdtr = jnp.minimum(
        20.0
        * (C.PI * C.PI)
        * C.PFRZ1
        * C.N0R
        * C.DENR
        / den
        * (jnp.exp(C.PFRZ2 * supcolt) - 1.0)
        * (rslope[:, 0] ** 7)
        * dtcld,
        qr,
    )
    do_rfrz = (supcol > 0.0) & (qr > 0.0)
    qs = jnp.where(do_rfrz, qs + pfrzdtr, qs)
    t = jnp.where(do_rfrz, t + xlf / cpm * pfrzdtr, t)
    qr = jnp.where(do_rfrz, qr - pfrzdtr, qr)

    rslope, rslopeb, rslope2, rslope3, _vt3 = _slope_wsm6(qr, qs, qg0, den, denfac, t)
    work1_1 = _diffac(xl, p, t, den, qsat1)
    work1_2 = _diffac(C.XLS, p, t, den, qsat2)
    work2 = _venfac(p, t, den)

    supsat1 = jnp.maximum(q, qmin) - qsat1
    satdt1 = supsat1 / dtcld
    praut = jnp.where(qc > C.QC0, jnp.minimum(C.QCK1 * qc ** (7.0 / 3.0), qc / dtcld), 0.0)
    pracw = jnp.where(
        (qr > C.QCRMIN) & (qc > qmin),
        jnp.minimum(C.PACRR * rslope3[:, 0] * rslopeb[:, 0] * qc * denfac, qc / dtcld),
        0.0,
    )
    coeres_r = rslope2[:, 0] * jnp.sqrt(rslope[:, 0] * rslopeb[:, 0])
    prevp_raw = (rh1 - 1.0) * (C.PRECR1 * rslope2[:, 0] + C.PRECR2 * work2 * coeres_r) / work1_1
    prevp_neg = jnp.maximum(jnp.maximum(prevp_raw, -qr / dtcld), satdt1 / 2.0)
    prevp_pos = jnp.minimum(prevp_raw, satdt1 / 2.0)
    prevp = jnp.where(prevp_raw < 0.0, prevp_neg, prevp_pos)
    prevp = jnp.where(qr > 0.0, prevp, 0.0)

    supcol = C.T0C - t
    cold = supcol > 0.0
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    supsat2 = jnp.maximum(q, qmin) - qsat2
    satdt2 = supsat2 / dtcld
    temp = den * jnp.maximum(qi, qmin)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    xni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)
    eacrs = jnp.exp(0.07 * (-supcol))
    xmi = den * qi / xni
    diameter = jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX)
    vt2i = 1.49e4 * diameter ** 1.31
    vt2s = C.PVTS * rslopeb[:, 1] * denfac
    acrfac = 2.0 * rslope3[:, 1] + 2.0 * diameter * rslope2[:, 1] + diameter * diameter * rslope[:, 1]
    psaci_raw = C.PI * qi * eacrs * C.N0S * n0sfac * jnp.abs(vt2s - vt2i) * acrfac / 4.0
    psaci = jnp.where(cold & (qs > C.QCRMIN) & (qi > qmin), psaci_raw, 0.0)
    psacw = jnp.where(
        (qs > C.QCRMIN) & (qc > qmin),
        jnp.minimum(C.PACRC * n0sfac * rslope3[:, 1] * rslopeb[:, 1] * qc * denfac, qc / dtcld),
        0.0,
    )
    pidep, psdep, pigen, psaut = _wsm5_deposition_block(
        cold, qi, qs, q, qmin, rh2, work1_2, satdt2, prevp, dtcld, n0sfac,
        rslope, rslopeb, rslope2, work2, den, supcol, supsat2, xni
    )
    psevp = jnp.zeros_like(t)  # WRF WSM5 leaves psdep zero in the warm branch.

    cold_branch = t <= C.T0C

    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + psacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut * f
    pracw_c = pracw * f
    psacw_c = psacw * f
    val = jnp.maximum(qmin, qi)
    src = (psaut + psaci - pigen - pidep) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psaut_c = psaut * f
    psaci_c = psaci * f
    pigen_c = pigen * f
    pidep_c = pidep * f
    val = jnp.maximum(qmin, qr)
    src = (-praut_c - pracw_c - prevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut_c * f
    pracw_c = pracw_c * f
    prevp_c = prevp * f
    val = jnp.maximum(qmin, qs)
    src = (-psdep - psaut_c - psaci_c - psacw_c) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psdep_c = psdep * f
    psaut_c = psaut_c * f
    psaci_c = psaci_c * f
    psacw_c = psacw_c * f

    w2_cold = -(prevp_c + psdep_c + pigen_c + pidep_c)
    q_cold = q + w2_cold * dtcld
    qc_cold = jnp.maximum(qc - (praut_c + pracw_c + psacw_c) * dtcld, 0.0)
    qr_cold = jnp.maximum(qr + (praut_c + pracw_c + prevp_c) * dtcld, 0.0)
    qi_cold = jnp.maximum(qi - (psaut_c + psaci_c - pigen_c - pidep_c) * dtcld, 0.0)
    qs_cold = jnp.maximum(qs + (psdep_c + psaut_c + psaci_c + psacw_c) * dtcld, 0.0)
    xlf_cold = C.XLS - xl
    xlwork2_cold = -C.XLS * (psdep_c + pidep_c + pigen_c) - xl * prevp_c - xlf_cold * psacw_c
    t_cold = t - xlwork2_cold / cpm * dtcld

    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + psacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut * f
    pracw_w = pracw * f
    psacw_w = psacw * f
    val = jnp.maximum(qmin, qr)
    src = (-praut_w - pracw_w - prevp - psacw_w) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut_w * f
    pracw_w = pracw_w * f
    prevp_w = prevp * f
    psacw_w = psacw_w * f
    val = jnp.maximum(C.QCRMIN, qs)
    src = (-psevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psevp_w = psevp * f

    w2_warm = -(prevp_w + psevp_w)
    q_warm = q + w2_warm * dtcld
    qc_warm = jnp.maximum(qc - (praut_w + pracw_w + psacw_w) * dtcld, 0.0)
    qr_warm = jnp.maximum(qr + (praut_w + pracw_w + prevp_w + psacw_w) * dtcld, 0.0)
    qs_warm = jnp.maximum(qs + psevp_w * dtcld, 0.0)
    xlwork2_warm = -xl * (prevp_w + psevp_w)
    t_warm = t - xlwork2_warm / cpm * dtcld

    q = jnp.where(cold_branch, q_cold, q_warm)
    qc = jnp.where(cold_branch, qc_cold, qc_warm)
    qr = jnp.where(cold_branch, qr_cold, qr_warm)
    qi = jnp.where(cold_branch, qi_cold, qi)
    qs = jnp.where(cold_branch, qs_cold, qs_warm)
    t = jnp.where(cold_branch, t_cold, t_warm)

    qsat1b, _ = _qsat_fpvs(t, p)
    w1 = _conden(t, q, qsat1b, xl, cpm)
    pcond = jnp.minimum(jnp.maximum(w1 / dtcld, 0.0), jnp.maximum(q, 0.0) / dtcld)
    pcond = jnp.where((qc > 0.0) & (w1 < 0.0), jnp.maximum(w1, -qc) / dtcld, pcond)
    q = q - pcond * dtcld
    qc = jnp.maximum(qc + pcond * dtcld, 0.0)
    t = t + pcond * xl / cpm * dtcld

    qc = jnp.where(qc <= qmin, 0.0, qc)
    qi = jnp.where(qi <= qmin, 0.0, qi)
    return (t, q, qc, qr, qi, qs, cpm, xl, rainncv, snowncv, tstepsnow, sr)


_wsm5_columns = jax.jit(
    jax.vmap(_wsm5_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None)),
    static_argnums=(9,),
)


def wsm5_run(t, qv, qc, qr, qi, qs, den, p, delz, delt):
    """Run WSM5 on a batch of columns shaped ``(ncol, nlev)``."""

    return _wsm5_columns(t, qv, qc, qr, qi, qs, den, p, delz, float(delt))


def wsm5_physics_tendency(theta, qv, qc, qr, qi, qs, pii, den, p, delz, delt):
    """Return frozen S0-style in-place replacements for WSM5."""

    t = theta * pii
    out = wsm5_run(t, qv, qc, qr, qi, qs, den, p, delz, delt)
    return PhysicsTendency(
        state_replacements={
            "theta": out["t"] / pii,
            "qv": out["qv"],
            "qc": out["qc"],
            "qr": out["qr"],
            "qi": out["qi"],
            "qs": out["qs"],
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


__all__ = ["wsm5_run", "wsm5_physics_tendency"]
