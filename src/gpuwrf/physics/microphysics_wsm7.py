"""JAX WSM7 single-moment 7-class microphysics (WRF mp_physics=24).

Faithful column port of WRF ``phys/module_mp_wsm7.F`` (subroutines ``wsm72D`` +
``slope_wsm7`` + the inlined fpvs/diffac/venfac/conden statement functions + the
shared semi-Lagrangian PLM sedimentation ``nislfv_rain_plm`` /
``nislfv_rain_plm6`` and the ``effectRad`` diagnostics), preserving WRF's exact
process order.

WSM7 = the WSM6 single-moment rain/snow/graupel microphysics extended with a
SEPARATE precipitating HAIL class ``qh`` (Bae, Hong, Tao 2018). The precipitating
array in WRF is ``qrs(:,:,1..4) = (rain, snow, graupel, hail)``; cloud
ice/water are ``qci(:,:,1..2) = (cloud water, cloud ice)``. The hail class adds
its own slope (``slope_wsm7`` index 4), a 4th semi-Lagrangian fall channel, and
the hail process terms phaci/phacw/phacr/phacs/phacg (accretion), phaut
(graupel->hail aggregation), phmlt/pheml (melting), phdep (deposition),
phevp (evaporation of melting hail), and the hail/graupel wet-growth coupling
``pgwet``/``phwet``.

WRF process order (wsm72D), preserved exactly:

  padding (qc/qi/qr/qs/qg/qh>=0)
  for loop in 1..loops (minor time steps):
    1. denfac, qsat(liquid,ice), rh, init process arrays, Ni (xni)
    2. slope_wsm7; semi-Lagrangian fallout of rain (plm), snow+graupel (plm6),
       hail (plm,iter=2); slope_wsm7 recompute on post-sedimentation qrs
    3. snow/graupel/hail melting for T>T0 (psmlt/pgmlt/phmlt -> rain)
    4. ice fallout (Vice + plm); surface precip (rain/snow/graupel/hail/sr)
    5. instantaneous melt/freeze (pimlt, pihmf, pihtf -> ice; pgfrz: rain->graupel)
    6. slope_wsm7; work1(diffac) liquid/ice; work2(venfac)
    7. WARM RAIN: praut, pracw, prevp
    8. COLD RAIN: praci/piacr, psaci, pgaci, phaci, psacw/pgacw/paacw/phacw,
       pracs/psacr, pracg/pgacr, pgacs(=0), phacr/phacs/phacg, pgwet/phwet
       (with the hail wet-growth shutoff), pseml/pgeml/pheml,
       deposition (pidep/psdep/pgdep/phdep), pigen, psaut, pgaut, phaut,
       psevp/pgevp/phevp
    9. mass conservation feedback + state update (T<=T0 / T>T0 branches), with
       the delta2/delta3 rain-presence switches
    10. recompute qsat; pcond condensation; small-value padding of qc/qi

The scheme works on TEMPERATURE ``t`` internally; the WRF ``wsm7`` wrapper
converts th<->t via the Exner function ``pii``. The adapter
``wsm7_physics_tendency`` mirrors that and returns a frozen ``PhysicsTendency``
with ``state_replacements`` for theta + the moist species (qv/qc/qr/qi/qs/qg/qh),
``accumulator_increments`` for surface precip, and ``diagnostics`` for the
effective radii.

Validation: per-column WRF savepoint parity against the real Fortran scheme
(proofs/v013/oracle/wsm7_oracle_driver.f90 driving the UNMODIFIED pristine
module_mp_wsm7.F). The port defaults to fp64; the classic WRF scheme runs in
fp32 (bare ``real``), so prognostic-state parity is to a predeclared physical
tolerance, never bitwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import wsm7_constants as C
from gpuwrf.physics.microphysics_wsm6 import (
    _conden,
    _cpmcal,
    _diffac,
    _diffus,
    _effective_radii,
    _nislfv_double,
    _nislfv_single,
    _qsat_fpvs,
    _venfac,
    _xka,
    _xlcal,
)


# --------------------------------------------------------------------------
# slope_wsm7: rain/snow/graupel/hail slope parameters + fall speeds (per column)
# Mirrors slope_wsm7 (4-class). Index 0=rain, 1=snow, 2=graupel, 3=hail.
# --------------------------------------------------------------------------
def _slope_wsm7(qr, qs, qg, qh, den, denfac, t):
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)

    # rain
    lamdar = jnp.sqrt(jnp.sqrt(C.PIDN0R / (qr * den)))
    rsl_r = jnp.where(qr <= C.QCRMIN, C.RSLOPERMAX, 1.0 / lamdar)
    rslb_r = jnp.where(qr <= C.QCRMIN, C.RSLOPERBMAX, rsl_r ** C.BVTR)
    rsl2_r = jnp.where(qr <= C.QCRMIN, C.RSLOPER2MAX, rsl_r * rsl_r)
    rsl3_r = jnp.where(qr <= C.QCRMIN, C.RSLOPER3MAX, rsl2_r * rsl_r)

    # snow
    lamdas = jnp.sqrt(jnp.sqrt(C.PIDN0S * n0sfac / (qs * den)))
    rsl_s = jnp.where(qs <= C.QCRMIN, C.RSLOPESMAX, 1.0 / lamdas)
    rslb_s = jnp.where(qs <= C.QCRMIN, C.RSLOPESBMAX, rsl_s ** C.BVTS)
    rsl2_s = jnp.where(qs <= C.QCRMIN, C.RSLOPES2MAX, rsl_s * rsl_s)
    rsl3_s = jnp.where(qs <= C.QCRMIN, C.RSLOPES3MAX, rsl2_s * rsl_s)

    # graupel
    lamdag = jnp.sqrt(jnp.sqrt(C.PIDN0G / (qg * den)))
    rsl_g = jnp.where(qg <= C.QCRMIN, C.RSLOPEGMAX, 1.0 / lamdag)
    rslb_g = jnp.where(qg <= C.QCRMIN, C.RSLOPEGBMAX, rsl_g ** C.BVTG)
    rsl2_g = jnp.where(qg <= C.QCRMIN, C.RSLOPEG2MAX, rsl_g * rsl_g)
    rsl3_g = jnp.where(qg <= C.QCRMIN, C.RSLOPEG3MAX, rsl2_g * rsl_g)

    # hail
    lamdah = jnp.sqrt(jnp.sqrt(C.PIDN0H / (qh * den)))
    rsl_h = jnp.where(qh <= C.QCRMIN, C.RSLOPEHMAX, 1.0 / lamdah)
    rslb_h = jnp.where(qh <= C.QCRMIN, C.RSLOPEHBMAX, rsl_h ** C.BVTH)
    rsl2_h = jnp.where(qh <= C.QCRMIN, C.RSLOPEH2MAX, rsl_h * rsl_h)
    rsl3_h = jnp.where(qh <= C.QCRMIN, C.RSLOPEH3MAX, rsl2_h * rsl_h)

    rslope = jnp.stack([rsl_r, rsl_s, rsl_g, rsl_h], axis=-1)
    rslopeb = jnp.stack([rslb_r, rslb_s, rslb_g, rslb_h], axis=-1)
    rslope2 = jnp.stack([rsl2_r, rsl2_s, rsl2_g, rsl2_h], axis=-1)
    rslope3 = jnp.stack([rsl3_r, rsl3_s, rsl3_g, rsl3_h], axis=-1)

    vt_r = jnp.where(qr <= 0.0, 0.0, C.PVTR * rslb_r * denfac)
    vt_s = jnp.where(qs <= 0.0, 0.0, C.PVTS * rslb_s * denfac)
    vt_g = jnp.where(qg <= 0.0, 0.0, C.PVTG * rslb_g * denfac)
    vt_h = jnp.where(qh <= 0.0, 0.0, C.PVTH * rslb_h * denfac)
    vt = jnp.stack([vt_r, vt_s, vt_g, vt_h], axis=-1)
    return rslope, rslopeb, rslope2, rslope3, vt


def _slope_hail(q, den, denfac):
    """slope_hail-style arrival-point terminal velocity for the hail nislfv iter."""
    lamda = jnp.sqrt(jnp.sqrt(C.PIDN0H / (q * den)))
    rsl = jnp.where(q <= C.QCRMIN, C.RSLOPEHMAX, 1.0 / lamda)
    rslb = jnp.where(q <= C.QCRMIN, C.RSLOPEHBMAX, rsl ** C.BVTH)
    vt = jnp.where(q <= 0.0, 0.0, C.PVTH * rslb * denfac)
    return vt


def _nislfv_single_hail(q, den, denfac, tk, dz, ww_in, dt, iter_n):
    """nislfv_rain_plm for the HAIL species (iter=2 in WSM7); one column.

    Identical PLM scheme to ``microphysics_wsm6._nislfv_single`` but the
    arrival-point terminal-velocity re-evaluation uses the HAIL slope (slope_hail
    via PIDN0H/PVTH), not the rain slope. Returns ``(q_new, precip)``.
    """
    from gpuwrf.physics.microphysics_wsm6 import (
        _plm_qmi_qpi,
        _plm_remap,
        _precip_out,
        _wi_interp,
        _diffusivity_limit,
    )

    km = q.shape[0]
    allold = jnp.sum(q)

    zi = jnp.concatenate([jnp.zeros(1, q.dtype), jnp.cumsum(dz)])  # length km+1
    wd = ww_in

    def iterate(ww):
        wi = _wi_interp(ww, dz, km)
        wi = _diffusivity_limit(wi, dz, dt, km)
        za = zi - wi * dt
        dza = jnp.concatenate([za[1:km + 1] - za[0:km],
                               (zi[km] - za[km]).reshape(1)])
        qa_cells = q * dz / dza[0:km]
        qr_cells = qa_cells / den
        return wi, za, dza, qa_cells, qr_cells

    def iter_body(n, state):
        ww, was = state
        wi, za, dza, qa_cells, qr_cells = iterate(ww)
        wa = _slope_hail(qr_cells, den, denfac)
        wa = jnp.where(n >= 1, 0.5 * (wa + was), wa)
        ww_new = 0.5 * (wd + wa)
        return ww_new, wa

    ww = wd
    was = jnp.zeros(km, dtype=q.dtype)
    if iter_n >= 1:
        ww, was = jax.lax.fori_loop(0, iter_n, iter_body, (ww, was))

    wi, za, dza, qa_cells, qr_cells = iterate(ww)
    qa = jnp.concatenate([qa_cells, jnp.zeros(1, q.dtype)])
    qmi, qpi = _plm_qmi_qpi(qa, dza, km)
    qn = _plm_remap(qa, dza, qmi, qpi, zi, za, km)
    precip = _precip_out(qa, dza, za, km)

    q_new = jnp.where(allold <= 0.0, q, qn)
    precip = jnp.where(allold <= 0.0, 0.0, precip)
    return q_new, precip


# --------------------------------------------------------------------------
# Core single-column WSM7 (wsm72D for one column, all minor loops)
# --------------------------------------------------------------------------
def _wsm7_column(t, q, qc, qr, qi, qs, qg, qh, den, p, delz, delt):
    """One column. Returns dict of updated fields + surface precip increments."""
    qc = jnp.maximum(qc, 0.0)
    qr = jnp.maximum(qr, 0.0)
    qi = jnp.maximum(qi, 0.0)
    qs = jnp.maximum(qs, 0.0)
    qg = jnp.maximum(qg, 0.0)
    qh = jnp.maximum(qh, 0.0)

    cpm = _cpmcal(q)
    xl = _xlcal(t)

    rainncv = jnp.zeros((), t.dtype)
    snowncv = jnp.zeros((), t.dtype)
    graupelncv = jnp.zeros((), t.dtype)
    hailncv = jnp.zeros((), t.dtype)
    tstepsnow = jnp.zeros((), t.dtype)
    tstepgraup = jnp.zeros((), t.dtype)
    tstephail = jnp.zeros((), t.dtype)
    sr = jnp.zeros((), t.dtype)

    loops = max(int(round(delt / C.DTCLDCR)), 1)
    dtcld = delt / loops
    if delt <= C.DTCLDCR:
        dtcld = delt

    state = (t, q, qc, qr, qi, qs, qg, qh, cpm, xl,
             rainncv, snowncv, graupelncv, hailncv,
             tstepsnow, tstepgraup, tstephail, sr)

    for _loop in range(loops):
        state = _wsm7_minor_loop(state, den, p, delz, dtcld)

    (t, q, qc, qr, qi, qs, qg, qh, cpm, xl,
     rainncv, snowncv, graupelncv, hailncv,
     tstepsnow, tstepgraup, tstephail, sr) = state

    re_qc, re_qi, re_qs = _effective_radii(t, qc, qi, qs, den)

    return {
        "t": t, "qv": q, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "qh": qh,
        "rainncv": rainncv, "snowncv": snowncv, "graupelncv": graupelncv,
        "hailncv": hailncv, "sr": sr,
        "re_cloud": re_qc, "re_ice": re_qi, "re_snow": re_qs,
    }


def _wsm7_minor_loop(state, den, p, delz, dtcld):
    (t, q, qc, qr, qi, qs, qg, qh, cpm, xl,
     rainncv, snowncv, graupelncv, hailncv,
     tstepsnow, tstepgraup, tstephail, sr) = state
    qmin = C.QMIN

    denfac = jnp.sqrt(C.DEN0 / den)

    qsat1, qsat2 = _qsat_fpvs(t, p)
    rh1 = jnp.maximum(q / qsat1, qmin)
    rh2 = jnp.maximum(q / qsat2, qmin)

    # ice crystal number conc (HDC 5c)
    temp = den * jnp.maximum(qi, qmin)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    xni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)

    # ----- sedimentation of rain / (snow+graupel) / hail -----
    rslope, rslopeb, rslope2, rslope3, vt = _slope_wsm7(qr, qs, qg, qh, den, denfac, t)
    workr = jnp.where(qr <= 0.0, 0.0, vt[:, 0])
    workh = jnp.where(qh <= 0.0, 0.0, vt[:, 3])
    qsum = jnp.maximum(qs + qg, 1.0e-15)
    worka = jnp.where(qsum > 1.0e-15, (vt[:, 1] * qs + vt[:, 2] * qg) / qsum, 0.0)

    denqrs1_new, delqrs1 = _nislfv_single(den * qr, den, denfac, t, delz, workr, dtcld, 1)
    qs_new, qg_new, delqrs2, delqrs3 = _nislfv_double(
        den * qs, den * qg, den, denfac, t, delz, worka, dtcld, 1
    )
    denqrs4_new, delqrs4 = _nislfv_single_hail(den * qh, den, denfac, t, delz, workh, dtcld, 2)

    qr = jnp.maximum(denqrs1_new / den, 0.0)
    qs = jnp.maximum(qs_new / den, 0.0)
    qg = jnp.maximum(qg_new / den, 0.0)
    qh = jnp.maximum(denqrs4_new / den, 0.0)
    fall1 = denqrs1_new * workr / delz
    fall2 = qs_new * worka / delz
    fall3 = qg_new * worka / delz
    fall4 = denqrs4_new * workh / delz
    fall1 = fall1.at[0].set(delqrs1 / delz[0] / dtcld)
    fall2 = fall2.at[0].set(delqrs2 / delz[0] / dtcld)
    fall3 = fall3.at[0].set(delqrs3 / delz[0] / dtcld)
    fall4 = fall4.at[0].set(delqrs4 / delz[0] / dtcld)

    rslope, rslopeb, rslope2, rslope3, _vt = _slope_wsm7(qr, qs, qg, qh, den, denfac, t)

    # ----- snow/graupel/hail melting (T>T0) -----
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    warm = t > C.T0C
    xlf = C.XLF0
    work2 = _venfac(p, t, den)

    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psmlt = (_xka(t, den) / xlf * (C.T0C - t) * C.PI / 2.0 * n0sfac
             * (C.PRECS1 * rslope2[:, 1] + C.PRECS2 * work2 * coeres_s) / den)
    psmlt = jnp.minimum(jnp.maximum(psmlt * dtcld, -qs), 0.0)
    do_psmlt = warm & (qs > 0.0)
    qs = jnp.where(do_psmlt, qs + psmlt, qs)
    qr = jnp.where(do_psmlt, qr - psmlt, qr)
    t = jnp.where(do_psmlt, t + xlf / cpm * psmlt, t)

    coeres_g = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgmlt = (_xka(t, den) / xlf * (C.T0C - t)
             * (C.PRECG1 * rslope2[:, 2] + C.PRECG2 * work2 * coeres_g) / den)
    pgmlt = jnp.minimum(jnp.maximum(pgmlt * dtcld, -qg), 0.0)
    do_pgmlt = warm & (qg > 0.0)
    qg = jnp.where(do_pgmlt, qg + pgmlt, qg)
    qr = jnp.where(do_pgmlt, qr - pgmlt, qr)
    t = jnp.where(do_pgmlt, t + xlf / cpm * pgmlt, t)

    coeres_h = rslope2[:, 3] * jnp.sqrt(rslope[:, 3] * rslopeb[:, 3])
    phmlt = (_xka(t, den) / xlf * (C.T0C - t)
             * (C.PRECH1 * rslope2[:, 3] + C.PRECH2 * work2 * coeres_h) / den)
    phmlt = jnp.minimum(jnp.maximum(phmlt * dtcld, -qh), 0.0)
    do_phmlt = warm & (qh > 0.0)
    qh = jnp.where(do_phmlt, qh + phmlt, qh)
    qr = jnp.where(do_phmlt, qr - phmlt, qr)
    t = jnp.where(do_phmlt, t + xlf / cpm * phmlt, t)

    # ----- ice fallout (Vice) -----
    xmi = den * qi / xni
    diameter = jnp.maximum(jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX), 1.0e-25)
    work1c = jnp.where(qi <= 0.0, 0.0, 1.49e4 * jnp.exp(jnp.log(diameter) * 1.31))
    denqci_new, delqi = _nislfv_single(den * qi, den, denfac, t, delz, work1c, dtcld, 0)
    qi = jnp.maximum(denqci_new / den, 0.0)
    fallc0 = delqi / delz[0] / dtcld

    # ----- surface precip -----
    fallsum = fall1[0] + fall2[0] + fall3[0] + fall4[0] + fallc0
    fallsum_qsi = fall2[0] + fallc0
    fallsum_qg = fall3[0]
    fallsum_qh = fall4[0]
    add = fallsum * delz[0] / C.DENR * dtcld * 1000.0
    rainncv = jnp.where(fallsum > 0.0, rainncv + add, rainncv)
    add_s = fallsum_qsi * delz[0] / C.DENR * dtcld * 1000.0
    tstepsnow = jnp.where(fallsum_qsi > 0.0, tstepsnow + add_s, tstepsnow)
    snowncv = jnp.where(fallsum_qsi > 0.0, snowncv + add_s, snowncv)
    add_g = fallsum_qg * delz[0] / C.DENR * dtcld * 1000.0
    tstepgraup = jnp.where(fallsum_qg > 0.0, tstepgraup + add_g, tstepgraup)
    graupelncv = jnp.where(fallsum_qg > 0.0, graupelncv + add_g, graupelncv)
    add_h = fallsum_qh * delz[0] / C.DENR * dtcld * 1000.0
    tstephail = jnp.where(fallsum_qh > 0.0, tstephail + add_h, tstephail)
    hailncv = jnp.where(fallsum_qh > 0.0, hailncv + add_h, hailncv)
    sr = jnp.where(fallsum > 0.0,
                   (tstepsnow + tstepgraup + tstephail) / (rainncv + 1.0e-12), sr)

    # ----- instantaneous melt/freeze (pimlt, pihmf, pihtf, pgfrz) -----
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
        C.PFRZ1 * (jnp.exp(C.PFRZ2 * supcolt) - 1.0) * den / C.DENR / C.XNCR
        * qc * qc * dtcld, qc)
    do_htf = (supcol > 0.0) & (qc > qmin)
    qi = jnp.where(do_htf, qi + pfrzdtc, qi)
    t = jnp.where(do_htf, t + xlf / cpm * pfrzdtc, t)
    qc = jnp.where(do_htf, qc - pfrzdtc, qc)
    # pgfrz: freezing of rain water -> graupel
    temp_r = rslope3[:, 0] * rslope3[:, 0] * rslope[:, 0]
    pfrzdtr = jnp.minimum(
        20.0 * (C.PI * C.PI) * C.PFRZ1 * C.N0R * C.DENR / den
        * (jnp.exp(C.PFRZ2 * supcolt) - 1.0) * temp_r * dtcld, qr)
    do_rfrz = (supcol > 0.0) & (qr > 0.0)
    qg = jnp.where(do_rfrz, qg + pfrzdtr, qg)
    t = jnp.where(do_rfrz, t + xlf / cpm * pfrzdtr, t)
    qr = jnp.where(do_rfrz, qr - pfrzdtr, qr)

    # ----- recompute slopes + work1/work2 for microphysics -----
    rslope, rslopeb, rslope2, rslope3, _vt = _slope_wsm7(qr, qs, qg, qh, den, denfac, t)
    work1_1 = _diffac(xl, p, t, den, qsat1)
    work1_2 = _diffac(C.XLS, p, t, den, qsat2)
    work2 = _venfac(p, t, den)

    # ======================= WARM RAIN =======================
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

    # ======================= COLD RAIN =======================
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
    vt2r = C.PVTR * rslopeb[:, 0] * denfac
    vt2s = C.PVTS * rslopeb[:, 1] * denfac
    vt2g = C.PVTG * rslopeb[:, 2] * denfac
    vt2h = C.PVTH * rslopeb[:, 3] * denfac
    qsum = jnp.maximum(qs + qg, 1.0e-15)
    vt2ave = jnp.where(qsum > 1.0e-15, (vt2s * qs + vt2g * qg) / qsum, 0.0)

    cold_qi = cold & (qi > qmin)

    # praci: accretion of cloud ice by rain (T<T0)
    acrfac_r = 2.0 * rslope3[:, 0] + 2.0 * diameter * rslope2[:, 0] + diameter ** 2 * rslope[:, 0]
    praci_raw = C.PI * qi * C.N0R * jnp.abs(vt2r - vt2i) * acrfac_r / 4.0
    praci_raw = praci_raw * jnp.minimum(jnp.maximum(0.0, qr / jnp.maximum(qi, qmin)), 1.0) ** 2
    praci_raw = jnp.minimum(praci_raw, qi / dtcld)
    praci = jnp.where(cold_qi & (qr > C.QCRMIN), praci_raw, 0.0)
    # piacr: accretion of rain by cloud ice (T<T0)
    piacr_raw = (C.PI ** 2 * C.AVTR * C.N0R * C.DENR * xni * denfac
                 * C.G6PBR * rslope3[:, 0] * rslope3[:, 0] * rslopeb[:, 0] / 24.0 / den)
    piacr_raw = piacr_raw * jnp.minimum(jnp.maximum(0.0, qi / jnp.maximum(qr, qmin)), 1.0) ** 2
    piacr_raw = jnp.minimum(piacr_raw, qr / dtcld)
    piacr = jnp.where(cold_qi & (qr > C.QCRMIN), piacr_raw, 0.0)
    # psaci: accretion of cloud ice by snow
    acrfac_s = 2.0 * rslope3[:, 1] + 2.0 * diameter * rslope2[:, 1] + diameter ** 2 * rslope[:, 1]
    psaci_raw = C.PI * qi * eacrs * C.N0S * n0sfac * jnp.abs(vt2ave - vt2i) * acrfac_s / 4.0
    psaci_raw = jnp.minimum(psaci_raw, qi / dtcld)
    psaci = jnp.where(cold_qi & (qs > C.QCRMIN), psaci_raw, 0.0)
    # pgaci: accretion of cloud ice by graupel
    egi = jnp.exp(0.07 * (-supcol))
    acrfac_g = 2.0 * rslope3[:, 2] + 2.0 * diameter * rslope2[:, 2] + diameter ** 2 * rslope[:, 2]
    pgaci_raw = C.PI * egi * qi * C.N0G * jnp.abs(vt2ave - vt2i) * acrfac_g / 4.0
    pgaci_raw = jnp.minimum(pgaci_raw, qi / dtcld)
    pgaci = jnp.where(cold_qi & (qg > C.QCRMIN), pgaci_raw, 0.0)
    # phaci: accretion of cloud ice by hail
    ehi = jnp.exp(0.07 * (-supcol))
    acrfac_h = 2.0 * rslope3[:, 3] + 2.0 * diameter * rslope2[:, 3] + diameter ** 2 * rslope[:, 3]
    phaci_raw = C.PI * ehi * qi * C.N0H * jnp.abs(vt2h - vt2i) * acrfac_h / 4.0
    phaci_raw = jnp.minimum(phaci_raw, qi / dtcld)
    phaci = jnp.where(cold_qi & (qh > C.QCRMIN), phaci_raw, 0.0)

    # psacw: accretion of cloud water by snow
    psacw = jnp.where(
        (qs > C.QCRMIN) & (qc > qmin),
        jnp.minimum(C.PACRC * n0sfac * rslope3[:, 1] * rslopeb[:, 1]
                    * jnp.minimum(jnp.maximum(0.0, qs / jnp.maximum(qc, qmin)), 1.0) ** 2
                    * qc * denfac, qc / dtcld),
        0.0,
    )
    # pgacw: accretion of cloud water by graupel
    pgacw = jnp.where(
        (qg > C.QCRMIN) & (qc > qmin),
        jnp.minimum(C.PACRG * rslope3[:, 2] * rslopeb[:, 2]
                    * jnp.minimum(jnp.maximum(0.0, qg / jnp.maximum(qc, qmin)), 1.0) ** 2
                    * qc * denfac, qc / dtcld),
        0.0,
    )
    # paacw: accretion of cloud water by averaged snow/graupel
    paacw = jnp.where(qsum > 1.0e-15, (qs * psacw + qg * pgacw) / qsum, 0.0)
    # phacw: accretion of cloud water by hail
    phacw = jnp.where(
        (qh > C.QCRMIN) & (qc > qmin),
        jnp.minimum(C.PACRH * rslope3[:, 3] * rslopeb[:, 3]
                    * jnp.minimum(jnp.maximum(0.0, qh / jnp.maximum(qc, qmin)), 1.0) ** 2
                    * qc * denfac, qc / dtcld),
        0.0,
    )

    # pracs: accretion of snow by rain (T<T0: S->G)
    acrfac_rs = (5.0 * rslope3[:, 1] * rslope3[:, 1] * rslope[:, 0]
                 + 2.0 * rslope3[:, 1] * rslope2[:, 1] * rslope2[:, 0]
                 + 0.5 * rslope2[:, 1] * rslope2[:, 1] * rslope3[:, 0])
    pracs_raw = C.PI ** 2 * C.N0R * C.N0S * n0sfac * jnp.abs(vt2r - vt2ave) * (C.DENS / den) * acrfac_rs
    pracs_raw = pracs_raw * jnp.minimum(jnp.maximum(0.0, qr / jnp.maximum(qs, qmin)), 1.0) ** 2
    pracs_raw = jnp.minimum(pracs_raw, qs / dtcld)
    pracs = jnp.where((qs > C.QCRMIN) & (qr > C.QCRMIN) & cold, pracs_raw, 0.0)
    # psacr: accretion of rain by snow
    acrfac_sr = (5.0 * rslope3[:, 0] * rslope3[:, 0] * rslope[:, 1]
                 + 2.0 * rslope3[:, 0] * rslope2[:, 0] * rslope2[:, 1]
                 + 0.5 * rslope2[:, 0] * rslope2[:, 0] * rslope3[:, 1])
    psacr_raw = C.PI ** 2 * C.N0R * C.N0S * n0sfac * jnp.abs(vt2ave - vt2r) * (C.DENR / den) * acrfac_sr
    psacr_raw = psacr_raw * jnp.minimum(jnp.maximum(0.0, qs / jnp.maximum(qr, qmin)), 1.0) ** 2
    psacr_raw = jnp.minimum(psacr_raw, qr / dtcld)
    psacr = jnp.where((qs > C.QCRMIN) & (qr > C.QCRMIN), psacr_raw, 0.0)

    # pracg: accretion of graupel by rain (T<T0: G->H)
    acrfac_rg = (5.0 * rslope3[:, 2] * rslope3[:, 2] * rslope[:, 0]
                 + 2.0 * rslope3[:, 2] * rslope2[:, 2] * rslope2[:, 0]
                 + 0.5 * rslope2[:, 2] * rslope2[:, 2] * rslope3[:, 0])
    pracg_raw = C.PI ** 2 * C.N0R * C.N0G * jnp.abs(vt2r - vt2ave) * (C.DENG / den) * acrfac_rg
    pracg_raw = pracg_raw * jnp.minimum(jnp.maximum(0.0, qr / jnp.maximum(qg, qmin)), 1.0) ** 2
    pracg_raw = jnp.minimum(pracg_raw, qg / dtcld)
    pracg = jnp.where((qg > C.QCRMIN) & (qr > C.QCRMIN) & cold, pracg_raw, 0.0)
    # pgacr: accretion of rain by graupel
    acrfac_gr = (5.0 * rslope3[:, 0] * rslope3[:, 0] * rslope[:, 2]
                 + 2.0 * rslope3[:, 0] * rslope2[:, 0] * rslope2[:, 2]
                 + 0.5 * rslope2[:, 0] * rslope2[:, 0] * rslope3[:, 2])
    pgacr_raw = C.PI ** 2 * C.N0R * C.N0G * jnp.abs(vt2ave - vt2r) * (C.DENR / den) * acrfac_gr
    pgacr_raw = pgacr_raw * jnp.minimum(jnp.maximum(0.0, qg / jnp.maximum(qr, qmin)), 1.0) ** 2
    pgacr_raw = jnp.minimum(pgacr_raw, qr / dtcld)
    pgacr = jnp.where((qg > C.QCRMIN) & (qr > C.QCRMIN), pgacr_raw, 0.0)

    # pgacs = 0 (eliminated in V3.0)
    pgacs = jnp.zeros_like(t)

    # phacr: accretion of rain by hail
    acrfac_hr = (5.0 * rslope3[:, 0] * rslope3[:, 0] * rslope[:, 3]
                 + 2.0 * rslope3[:, 0] * rslope2[:, 0] * rslope2[:, 3]
                 + 0.5 * rslope2[:, 0] * rslope2[:, 0] * rslope3[:, 3])
    phacr_raw = C.PI ** 2 * C.N0R * C.N0H * jnp.abs(vt2h - vt2r) * (C.DENR / den) * acrfac_hr
    phacr_raw = phacr_raw * jnp.minimum(jnp.maximum(0.0, qh / jnp.maximum(qr, qmin)), 1.0) ** 2
    phacr_raw = jnp.minimum(phacr_raw, qr / dtcld)
    phacr = jnp.where((qh > C.QCRMIN) & (qr > C.QCRMIN), phacr_raw, 0.0)
    # phacs: accretion of snow by hail
    acrfac_hs = (5.0 * rslope3[:, 1] * rslope3[:, 1] * rslope[:, 3]
                 + 2.0 * rslope3[:, 1] * rslope2[:, 1] * rslope2[:, 3]
                 + 0.5 * rslope2[:, 1] * rslope2[:, 1] * rslope3[:, 3])
    phacs_raw = (C.PI ** 2 * C.EACHS * C.N0S * n0sfac * C.N0H * jnp.abs(vt2h - vt2ave)
                 * (C.DENS / den) * acrfac_hs)
    phacs_raw = jnp.minimum(phacs_raw, qs / dtcld)
    phacs = jnp.where((qh > C.QCRMIN) & (qs > C.QCRMIN), phacs_raw, 0.0)
    # phacg: accretion of graupel by hail
    acrfac_hg = (5.0 * rslope3[:, 2] * rslope3[:, 2] * rslope[:, 3]
                 + 2.0 * rslope3[:, 2] * rslope2[:, 2] * rslope2[:, 3]
                 + 0.5 * rslope2[:, 2] * rslope2[:, 2] * rslope3[:, 3])
    phacg_raw = C.PI ** 2 * C.EACHG * C.N0G * C.N0H * jnp.abs(vt2h - vt2ave) * (C.DENG / den) * acrfac_hg
    phacg_raw = jnp.minimum(phacg_raw, qg / dtcld)
    phacg = jnp.where((qh > C.QCRMIN) & (qg > C.QCRMIN), phacg_raw, 0.0)

    # ----- pgwet / phwet wet growth -----
    # rs0 = saturation mixing ratio over liquid at t0c (fpvs liquid at T0C)
    ttp = C.T0C + 0.01
    dldt = C.CPV - C.CLIQ
    xa = -dldt / C.RV
    xb = xa + C.XLV0 / (C.RV * ttp)
    rs0 = C.PSAT * jnp.exp(jnp.log(ttp / C.T0C) * xa) * jnp.exp(xb * (1.0 - ttp / C.T0C))
    rs0 = jnp.minimum(rs0, 0.99 * p)
    rs0 = C.EP2 * rs0 / (p - rs0)
    rs0 = jnp.maximum(rs0, qmin)
    ghw1 = den * C.XLV0 * _diffus(t, p) * (rs0 - q) - _xka(t, den) * (-supcol)
    ghw2 = den * (C.XLF0 + C.CLIQ * (-supcol))
    ghw3 = _venfac(p, t, den) * jnp.sqrt(jnp.sqrt(C.G * den / C.DEN0))
    ghw4 = den * (C.XLF0 - C.CLIQ * supcol + C.CICE * supcol)
    egi_w = jnp.exp(0.07 * (-supcol))
    pgaci_w = jnp.where((qg > C.QCRMIN) & (pgaci > 0.0), pgaci / egi_w, 0.0)
    pgwet = ghw1 / ghw2 * (C.PRECG1 * rslope2[:, 2]
                           + C.PRECG3 * ghw3 * rslope[:, 3] ** 2.75
                           + ghw4 * (pgaci_w + pgacs))
    pgwet = jnp.where(qg > C.QCRMIN, jnp.maximum(pgwet, 0.0), 0.0)
    ehi_w = jnp.exp(0.07 * (-supcol))
    phaci_w = jnp.where((qh > C.QCRMIN) & (phaci > 0.0), phaci / ehi_w, 0.0)
    phwet = ghw1 / ghw2 * (C.PRECH1 * rslope2[:, 3]
                           + C.PRECH3 * ghw3 * rslope[:, 3] ** 2.75
                           + ghw4 * (phaci_w + phacs))
    phwet = jnp.maximum(phwet, 0.0)
    # hail wet-growth shutoff: if phacw+phacr < 0.95*phwet, zero phaci/phacs/phacg
    shutoff = (phacw + phacr) < 0.95 * phwet
    phaci = jnp.where(shutoff, 0.0, phaci)
    phacs = jnp.where(shutoff, 0.0, phacs)
    phacg = jnp.where(shutoff, 0.0, phacg)

    # ----- enhanced melting (T>=T0) -----
    warm2 = supcol <= 0.0
    xlf_w = C.XLF0
    pseml = jnp.where(
        warm2 & (qs > 0.0),
        jnp.minimum(jnp.maximum(C.CLIQ * supcol * (paacw + psacr) / xlf_w, -qs / dtcld), 0.0),
        0.0,
    )
    pgeml = jnp.where(
        warm2 & (qg > 0.0),
        jnp.minimum(jnp.maximum(C.CLIQ * supcol * (paacw + pgacr) / xlf_w, -qg / dtcld), 0.0),
        0.0,
    )
    pheml = jnp.where(
        warm2 & (qh > 0.0),
        jnp.minimum(jnp.maximum(C.CLIQ * supcol * (phacw + phacr) / xlf_w, -qh / dtcld), 0.0),
        0.0,
    )

    # ----- deposition/sublimation + pigen + psaut/pgaut/phaut (T<T0) -----
    pidep, psdep, pgdep, phdep, pigen, psaut, pgaut = _wsm7_deposition_block(
        cold, qi, qs, qg, qh, q, qmin, rh2, work1_2, satdt2, prevp, dtcld,
        n0sfac, rslope, rslopeb, rslope2, work2, den, supcol, supsat2, xni, diameter
    )
    # phaut: graupel->hail aggregation (NOTE: in WRF this is OUTSIDE the supcol>0
    # guard -- it fires whenever qg>0, see module_mp_wsm7.F lines 1460-1466).
    alpha2 = 1.0e-3 * jnp.exp(0.09 * (-supcol))
    phaut = jnp.where(qg > 0.0, jnp.minimum(jnp.maximum(0.0, alpha2 * (qg - C.QS0)), qg / dtcld), 0.0)

    # ----- evaporation of melting snow/graupel/hail (T<0 -> supcol<0) -----
    psevp = jnp.zeros_like(t)
    pgevp = jnp.zeros_like(t)
    phevp = jnp.zeros_like(t)
    sub_cond = (supcol < 0.0) & (rh1 < 1.0)
    coeres_s2 = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psevp_raw = ((rh1 - 1.0) * n0sfac * (C.PRECS1 * rslope2[:, 1] + C.PRECS2 * work2 * coeres_s2)
                 / work1_1)
    psevp = jnp.where(sub_cond & (qs > 0.0),
                      jnp.minimum(jnp.maximum(psevp_raw, -qs / dtcld), 0.0), 0.0)
    coeres_g2 = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgevp_raw = (rh1 - 1.0) * (C.PRECG1 * rslope2[:, 2] + C.PRECG2 * work2 * coeres_g2) / work1_1
    pgevp = jnp.where(sub_cond & (qg > 0.0),
                      jnp.minimum(jnp.maximum(pgevp_raw, -qg / dtcld), 0.0), 0.0)
    coeres_h2 = rslope2[:, 3] * jnp.sqrt(rslope[:, 3] * rslopeb[:, 3])
    phevp_raw = (rh1 - 1.0) * (C.PRECH1 * rslope2[:, 3] + C.PRECH2 * work2 * coeres_h2) / work1_1
    phevp = jnp.where(sub_cond & (qh > 0.0),
                      jnp.minimum(jnp.maximum(phevp_raw, -qh / dtcld), 0.0), 0.0)

    pvapg = jnp.zeros_like(t)  # never set nonzero in WRF wsm72D
    pvaph = jnp.zeros_like(t)
    primh = jnp.zeros_like(t)

    # ======================= mass-conservation feedback + update =======================
    delta2 = jnp.where((qr < 1.0e-4) & (qs < 1.0e-4), 1.0, 0.0)
    delta3 = jnp.where(qr < 1.0e-4, 1.0, 0.0)
    cold_branch = t <= C.T0C

    # ----- COLD branch (T<=T0) clamps -----
    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + paacw + paacw + phacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut * f; pracw_c = pracw * f; paacw_c = paacw * f; phacw_c = phacw * f

    val = jnp.maximum(qmin, qi)
    src = (psaut - pigen - pidep + praci + psaci + pgaci + phaci) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psaut_c = psaut * f; pigen_c = pigen * f; pidep_c = pidep * f
    praci_c = praci * f; psaci_c = psaci * f; pgaci_c = pgaci * f; phaci_c = phaci * f

    val = jnp.maximum(qmin, qr)
    src = (-praut_c - prevp - pracw_c + piacr + psacr + pgacr + phacr) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut_c * f; prevp_c = prevp * f; pracw_c = pracw_c * f
    piacr_c = piacr * f; psacr_c = psacr * f; pgacr_c = pgacr * f; phacr_c = phacr * f

    val = jnp.maximum(qmin, qs)
    src = -(psdep + psaut_c + paacw_c + pvapg + pvaph + psaci_c - pgaut
            - pracs * (1.0 - delta2) + piacr_c * delta3 + praci_c * delta3
            + psacr_c * delta2 - pgacs - phacs) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psdep_c = psdep * f; psaut_c = psaut_c * f; pgaut_c = pgaut * f; paacw_c = paacw_c * f
    pvapg_c = pvapg * f; pvaph_c = pvaph * f; psaci_c = psaci_c * f; piacr_c = piacr_c * f
    praci_c = praci_c * f; psacr_c = psacr_c * f; pracs_c = pracs * f; pgacs_c = pgacs * f
    phacs_c = phacs * f

    val = jnp.maximum(qmin, qg)
    src = -(pgdep + pgaut_c + pgaci_c + paacw_c + pgacs_c
            + piacr_c * (1.0 - delta3) + praci_c * (1.0 - delta3)
            + psacr_c * (1.0 - delta2) + pgacr_c * delta2
            + pracs_c * (1.0 - delta2) - pracg * (1.0 - delta2)
            - phaut - pvapg_c - phacg + primh) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgdep_c = pgdep * f; pgaut_c = pgaut_c * f; phaut_c = phaut * f; piacr_c = piacr_c * f
    praci_c = praci_c * f; pracs_c = pracs_c * f; pracg_c = pracg * f; psacr_c = psacr_c * f
    paacw_c = paacw_c * f; pgaci_c = pgaci_c * f; pgacr_c = pgacr_c * f; pgacs_c = pgacs_c * f
    pvapg_c = pvapg_c * f; phacg_c = phacg * f; primh_c = primh * f

    val = jnp.maximum(qmin, qh)
    src = -(phdep + phaut_c + pgacr_c * (1.0 - delta2) + pracg_c * (1.0 - delta2)
            + phacw_c + phacr_c + phaci_c + phacs_c + phacg_c - pvaph_c - primh_c) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    phdep_c = phdep * f; phaut_c = phaut_c * f; pracg_c = pracg_c * f; pgacr_c = pgacr_c * f
    phacw_c = phacw_c * f; phaci_c = phaci_c * f; phacr_c = phacr_c * f; phacs_c = phacs_c * f
    phacg_c = phacg_c * f; pvaph_c = pvaph_c * f; primh_c = primh_c * f

    work2_cold = -(prevp_c + psdep_c + pgdep_c + phdep_c + pigen_c + pidep_c)
    q_cold = q + work2_cold * dtcld
    qc_cold = jnp.maximum(qc - (praut_c + pracw_c + paacw_c + paacw_c + phacw_c) * dtcld, 0.0)
    qr_cold = jnp.maximum(qr + (praut_c + pracw_c + prevp_c - piacr_c - pgacr_c
                                - psacr_c - phacr_c) * dtcld, 0.0)
    qi_cold = jnp.maximum(qi - (psaut_c + praci_c + psaci_c + pgaci_c + phaci_c
                                - pigen_c - pidep_c) * dtcld, 0.0)
    qs_cold = jnp.maximum(qs + (psdep_c + psaut_c + paacw_c + pvapg_c + pvaph_c - pgaut_c
                                + psaci_c - pgacs_c - phacs_c + piacr_c * delta3
                                + praci_c * delta3 + psacr_c * delta2
                                - pracs_c * (1.0 - delta2)) * dtcld, 0.0)
    qg_cold = jnp.maximum(qg + (pgdep_c + pgaut_c + piacr_c * (1.0 - delta3)
                                + praci_c * (1.0 - delta3) + psacr_c * (1.0 - delta2)
                                + pgacr_c * delta2 + pgaci_c + paacw_c + pgacs_c + primh_c
                                + pracs_c * (1.0 - delta2) - pracg_c * (1.0 - delta2)
                                - phaut_c - pvapg_c - phacg_c) * dtcld, 0.0)
    qh_cold = jnp.maximum(qh + (phdep_c + phaut_c + pgacr_c * (1.0 - delta2)
                                + pracg_c * (1.0 - delta2) + phacw_c + phacr_c + phaci_c
                                + phacs_c + phacg_c - pvaph_c - primh_c) * dtcld, 0.0)
    xlf_cold = C.XLS - xl
    xlwork2_cold = (-C.XLS * (psdep_c + pgdep_c + phdep_c + pidep_c + pigen_c)
                    - xl * prevp_c
                    - xlf_cold * (piacr_c + paacw_c + paacw_c + phacw_c + phacr_c
                                  + pgacr_c + psacr_c))
    t_cold = t - xlwork2_cold / cpm * dtcld

    # ----- WARM branch (T>T0) clamps -----
    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + paacw + paacw + phacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut * f; pracw_w = pracw * f; paacw_w = paacw * f; phacw_w = phacw * f

    val = jnp.maximum(qmin, qr)
    src = (pseml + pgeml + pheml - pracw_w - paacw_w - paacw_w - phacw_w - prevp - praut_w) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut_w * f; prevp_w = prevp * f; pracw_w = pracw_w * f; paacw_w = paacw_w * f
    phacw_w = phacw_w * f; pseml_w = pseml * f; pgeml_w = pgeml * f; pheml_w = pheml * f

    val = jnp.maximum(C.QCRMIN, qs)
    src = (pgacs + phacs - pseml_w - psevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs * f; phacs_w = phacs * f; psevp_w = psevp * f; pseml_w = pseml_w * f

    val = jnp.maximum(C.QCRMIN, qg)
    src = -(pgacs_w + pgevp + pgeml_w - phacg) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs_w * f; pgevp_w = pgevp * f; pgeml_w = pgeml_w * f; phacg_w = phacg * f

    val = jnp.maximum(C.QCRMIN, qh)
    src = -(phacs_w + phacg_w + phevp + pheml_w) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    phacs_w = phacs_w * f; phacg_w = phacg_w * f; phevp_w = phevp * f; pheml_w = pheml_w * f

    work2_warm = -(prevp_w + psevp_w + pgevp_w + phevp_w)
    q_warm = q + work2_warm * dtcld
    qc_warm = jnp.maximum(qc - (praut_w + pracw_w + paacw_w + paacw_w + phacw_w) * dtcld, 0.0)
    qr_warm = jnp.maximum(qr + (praut_w + pracw_w + prevp_w + paacw_w + paacw_w + phacw_w
                                - pseml_w - pgeml_w - pheml_w) * dtcld, 0.0)
    qs_warm = jnp.maximum(qs + (psevp_w + pseml_w - pgacs_w - phacs_w) * dtcld, 0.0)
    qg_warm = jnp.maximum(qg + (pgacs_w + pgevp_w + pgeml_w - phacg_w) * dtcld, 0.0)
    qh_warm = jnp.maximum(qh + (phacs_w + phacg_w + phevp_w + pheml_w) * dtcld, 0.0)
    xlwork2_warm = (-xl * (prevp_w + psevp_w + pgevp_w + phevp_w)
                    - (C.XLS - xl) * (pseml_w + pgeml_w + pheml_w))
    t_warm = t - xlwork2_warm / cpm * dtcld

    q = jnp.where(cold_branch, q_cold, q_warm)
    qc = jnp.where(cold_branch, qc_cold, qc_warm)
    qr = jnp.where(cold_branch, qr_cold, qr_warm)
    qi = jnp.where(cold_branch, qi_cold, qi)
    qs = jnp.where(cold_branch, qs_cold, qs_warm)
    qg = jnp.where(cold_branch, qg_cold, qg_warm)
    qh = jnp.where(cold_branch, qh_cold, qh_warm)
    t = jnp.where(cold_branch, t_cold, t_warm)

    # ----- pcond saturation adjustment -----
    qsat1b, _ = _qsat_fpvs(t, p)
    w1 = _conden(t, q, qsat1b, xl, cpm)
    pcond = jnp.minimum(jnp.maximum(w1 / dtcld, 0.0), jnp.maximum(q, 0.0) / dtcld)
    pcond = jnp.where((qc > 0.0) & (w1 < 0.0), jnp.maximum(w1, -qc) / dtcld, pcond)
    q = q - pcond * dtcld
    qc = jnp.maximum(qc + pcond * dtcld, 0.0)
    t = t + pcond * xl / cpm * dtcld

    qc = jnp.where(qc <= qmin, 0.0, qc)
    qi = jnp.where(qi <= qmin, 0.0, qi)
    return (t, q, qc, qr, qi, qs, qg, qh, cpm, xl,
            rainncv, snowncv, graupelncv, hailncv,
            tstepsnow, tstepgraup, tstephail, sr)


def _wsm7_deposition_block(cold, qi, qs, qg, qh, q, qmin, rh2, work1_2, satdt2,
                           prevp, dtcld, n0sfac, rslope, rslopeb, rslope2, work2,
                           den, supcol, supsat2, xni, diameter):
    """Sequential ice/snow/graupel/hail deposition with WRF's ifsat short-circuit.

    Mirrors the supcol>0 deposition block of wsm72D: pidep, psdep, pgdep, phdep
    each guarded by the cumulative ifsat flag, then pigen, psaut, pgaut.
    """
    # pidep
    pidep_raw = 4.0 * diameter * xni * (rh2 - 1.0) / work1_2
    supice1 = satdt2 - prevp
    pidep_neg = jnp.maximum(jnp.maximum(pidep_raw, satdt2 * 0.5), supice1)
    pidep_neg = jnp.maximum(pidep_neg, -qi / dtcld)
    pidep_pos = jnp.minimum(jnp.minimum(pidep_raw, satdt2 * 0.5), supice1)
    pidep_v = jnp.where(pidep_raw < 0.0, pidep_neg, pidep_pos)
    do_pidep = cold & (qi > 0.0)
    pidep = jnp.where(do_pidep, pidep_v, 0.0)
    ifsat1 = do_pidep & (jnp.abs(prevp + pidep) >= jnp.abs(satdt2))

    # psdep
    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psdep_raw = ((rh2 - 1.0) * n0sfac
                 * (C.PRECS1 * rslope2[:, 1] + C.PRECS2 * work2 * coeres_s) / work1_2)
    supice2 = satdt2 - prevp - pidep
    psdep_neg = jnp.maximum(psdep_raw, -qs / dtcld)
    psdep_neg = jnp.maximum(jnp.maximum(psdep_neg, satdt2 * 0.5), supice2)
    psdep_pos = jnp.minimum(jnp.minimum(psdep_raw, satdt2 * 0.5), supice2)
    psdep_v = jnp.where(psdep_raw < 0.0, psdep_neg, psdep_pos)
    do_psdep = cold & (qs > 0.0) & jnp.logical_not(ifsat1)
    psdep = jnp.where(do_psdep, psdep_v, 0.0)
    ifsat2 = ifsat1 | (do_psdep & (jnp.abs(prevp + pidep + psdep) >= jnp.abs(satdt2)))

    # pgdep
    coeres_g = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgdep_raw = ((rh2 - 1.0)
                 * (C.PRECG1 * rslope2[:, 2] + C.PRECG2 * work2 * coeres_g) / work1_2)
    supice3 = satdt2 - prevp - pidep - psdep
    pgdep_neg = jnp.maximum(pgdep_raw, -qg / dtcld)
    pgdep_neg = jnp.maximum(jnp.maximum(pgdep_neg, satdt2 * 0.5), supice3)
    pgdep_pos = jnp.minimum(jnp.minimum(pgdep_raw, satdt2 * 0.5), supice3)
    pgdep_v = jnp.where(pgdep_raw < 0.0, pgdep_neg, pgdep_pos)
    do_pgdep = cold & (qg > 0.0) & jnp.logical_not(ifsat2)
    pgdep = jnp.where(do_pgdep, pgdep_v, 0.0)
    ifsat3 = ifsat2 | (do_pgdep & (jnp.abs(prevp + pidep + psdep + pgdep) >= jnp.abs(satdt2)))

    # phdep
    coeres_h = rslope2[:, 3] * jnp.sqrt(rslope[:, 3] * rslopeb[:, 3])
    phdep_raw = ((rh2 - 1.0)
                 * (C.PRECH1 * rslope2[:, 3] + C.PRECH2 * work2 * coeres_h) / work1_2)
    supice4 = satdt2 - prevp - pidep - psdep - pgdep
    phdep_neg = jnp.maximum(phdep_raw, -qh / dtcld)
    phdep_neg = jnp.maximum(jnp.maximum(phdep_neg, satdt2 * 0.5), supice4)
    phdep_pos = jnp.minimum(jnp.minimum(phdep_raw, satdt2 * 0.5), supice4)
    phdep_v = jnp.where(phdep_raw < 0.0, phdep_neg, phdep_pos)
    do_phdep = cold & (qh > 0.0) & jnp.logical_not(ifsat3)
    phdep = jnp.where(do_phdep, phdep_v, 0.0)
    ifsat4 = ifsat3 | (do_phdep & (jnp.abs(prevp + pidep + psdep + pgdep + phdep) >= jnp.abs(satdt2)))

    # pigen (NOTE WRF supice here = satdt-prevp-pidep-psdep-pgdep, no phdep term)
    supice5 = satdt2 - prevp - pidep - psdep - pgdep
    xni0 = 1.0e3 * jnp.exp(0.1 * supcol)
    roqi0 = 4.92e-11 * xni0 ** 1.33
    pigen_raw = jnp.maximum(0.0, (roqi0 / den - jnp.maximum(qi, 0.0)) / dtcld)
    pigen_v = jnp.minimum(jnp.minimum(pigen_raw, satdt2), supice5)
    do_pigen = cold & (supsat2 > 0.0) & jnp.logical_not(ifsat4)
    pigen = jnp.where(do_pigen, pigen_v, 0.0)

    # psaut
    qimax = C.ROQIMAX / den
    psaut = jnp.where(cold & (qi > 0.0), jnp.maximum(0.0, (qi - qimax) / dtcld), 0.0)
    # pgaut: snow->graupel aggregation
    alpha2 = 1.0e-3 * jnp.exp(0.09 * (-supcol))
    pgaut = jnp.where(cold & (qs > 0.0),
                      jnp.minimum(jnp.maximum(0.0, alpha2 * (qs - C.QS0)), qs / dtcld), 0.0)

    return pidep, psdep, pgdep, phdep, pigen, psaut, pgaut


_wsm7_columns = jax.jit(
    jax.vmap(_wsm7_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)),
    static_argnums=(11,),
)


def wsm7_run(t, qv, qc, qr, qi, qs, qg, qh, den, p, delz, delt):
    """Run WSM7 on a batch of columns shaped ``(ncol, nlev)``."""

    return _wsm7_columns(t, qv, qc, qr, qi, qs, qg, qh, den, p, delz, float(delt))


def wsm7_physics_tendency(theta, qv, qc, qr, qi, qs, qg, qh, pii, den, p, delz, delt):
    """Return frozen S0-style in-place replacements for WSM7."""

    t = theta * pii
    out = wsm7_run(t, qv, qc, qr, qi, qs, qg, qh, den, p, delz, delt)
    return PhysicsTendency(
        state_replacements={
            "theta": out["t"] / pii,
            "qv": out["qv"],
            "qc": out["qc"],
            "qr": out["qr"],
            "qi": out["qi"],
            "qs": out["qs"],
            "qg": out["qg"],
            "qh": out["qh"],
        },
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["snowncv"],
            "graupel_acc": out["graupelncv"],
            "hail_acc": out["hailncv"],
        },
        diagnostics={
            "re_cloud": out["re_cloud"],
            "re_ice": out["re_ice"],
            "re_snow": out["re_snow"],
            "sr": out["sr"],
        },
    )


__all__ = ["wsm7_run", "wsm7_physics_tendency"]
