"""JAX WDM7 double-moment 7-class hail microphysics (WRF mp_physics=26).

Faithful column port of WRF ``phys/module_mp_wdm7.F`` (subroutines ``wdm72D`` +
``slope_wdm7`` + ``slope_rain`` + ``slope_hail`` + the inlined fpvs/diffac/
venfac/conden statement functions + the semi-Lagrangian PLM sedimentation
``nislfv_rain_plmr`` / ``nislfv_rain_plm6`` and ``effectRad_wdm7``), preserving
WRF's EXACT process order.

WDM7 = WDM6 (DOUBLE-MOMENT warm rain: predicted cloud-droplet number ``Nc``,
rain number ``Nr``, CCN number ``Nn``; single-moment ice/snow/graupel) extended
with a SEPARATE precipitating HAIL class ``qh`` (Bae, Hong, Tao 2018). Hail is
SINGLE-MOMENT -- there is NO hail number ``Nh``. The WRF number array carries
only ``ncr(:,:,1)=Nn (CCN)``, ``ncr(:,:,2)=Nc (cloud)``, ``ncr(:,:,3)=Nr
(rain)``; the precipitating mass array is ``qrs(:,:,1..4) = (rain, snow,
graupel, hail)`` and ``qci(:,:,1..2) = (cloud water, cloud ice)``.

The hail class adds (mirroring WSM7 but with the WDM6 double-moment rain-number
collection kernels): slope index 4 (``slope_hail`` via PIDN0H/PVTH), a 4th
semi-Lagrangian fall channel (``nislfv_rain_plmr`` id=2, iter=1), and the hail
process terms phaci/phacw/phacr/phacs/phacg (accretion, the rain ones carrying
the predicted ``Nr``, plus nhacw/nhacr number sinks), phaut (graupel->hail
aggregation), phmlt/pheml (melting -> rain, with the nhmlt/nheml -> Nr number
transfers), phdep (deposition), phevp (evaporation of melting hail), and the
hail/graupel wet-growth coupling ``pgwet``/``phwet``.

WRF process order (wdm72D), preserved:

  padding (qc/qi/qr/qs/qg/qh>=0; Nn in [1e8,2e10]; Nc,Nr>=0)
  for loop in 1..loops (minor time steps):
    1. denfac, qsat(liquid,ice), rh, init arrays, rslopec, xni
    2. slope_wdm7; rain+Nr in-line mstep fallout; snow+graupel (plm6); hail
       (plmr id=2,iter=1); slope_wdm7 recompute on post-sedimentation qrs
    3. snow/graupel/hail melting for T>T0 (psmlt/pgmlt/phmlt + nsmlt/ngmlt/nhmlt
       -> Nr); note WRF divides phmlt by mstep(i) (carried tracer)
    4. ice fallout (Vice + plmr); surface precip (rain/snow/graupel/hail/sr)
    5. instantaneous melt/freeze (pimlt+nimlt, pihmf+nihmf, pihtf+nihtf,
       pgfrz+ngfrz); clamp Nc,Nr>=0
    6. slope_wdm7; avedia(rain,cloud); rslopec; work1(diffac)/work2(venfac)
    7. WARM RAIN (double-moment): praut+nraut, pracw+nracw, nccol, nrcol,
       prevp + Nrevp(NR->NCCN)
    8. COLD RAIN: praci/piacr/niacr, psaci, pgaci, phaci, psacw/nsacw,
       pgacw/ngacw, paacw/naacw, phacw/nhacw, pracs, psacr/nsacr, pgacr/ngacr,
       pgacs(=0), pracg, phacr/nhacr, phacs, phacg, pgwet/phwet wet growth
       (with the hail wet-growth shutoff), pseml/nseml, pgeml/ngeml,
       pheml/nheml, deposition(pidep/psdep/pgdep/phdep), pigen, psaut, pgaut,
       phaut, psevp/pgevp/phevp
    9. mass+number conservation feedback + state update (T<=T0 / T>T0 branches),
       with the delta2/delta3 rain-presence switches
    10. recompute qsat; small-drop rain->cloud (avedia<=di82); CCN activation
        pcact+ncact; pcond condensation + ncevp(NC->NCCN); small-value padding +
        lamdr/lamdc clamping (adjusts Nr/Nc)

The scheme works on TEMPERATURE ``t`` internally; the WRF ``wdm7`` wrapper
converts th<->t via the Exner function ``pii``. The adapter
``wdm7_physics_tendency`` mirrors that and returns a frozen ``PhysicsTendency``
with ``state_replacements`` for theta + the moist species (qv/qc/qr/qi/qs/qg/qh)
+ the number leaves (Nc, Nr), ``accumulator_increments`` for surface precip
(incl hail_acc), and ``diagnostics`` for Nn (CCN) + the effective radii.

Validation: per-column WRF savepoint parity against the real Fortran scheme
(proofs/v013_wdm7/oracle driving the UNMODIFIED pristine module_mp_wdm7.F). The
port defaults to fp64; the classic WRF scheme runs in fp32 (bare ``real``), so
prognostic-state parity is to a predeclared physical tolerance, never bitwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import wdm7_constants as C
# Reuse the WDM6 thermodynamic helpers, PLM sedimentation primitives, the
# double-moment rain+Nr in-line fallout, and the effective-radii diagnostic.
# WDM7's effectRad_wdm7 is byte-identical to effectRad_wdm6, and all of WDM7's
# constants that those helpers read (PIDNC/PIDN0S/ALPHA/DICON/... and the RE_*
# bounds) have the SAME values in wdm7_constants as in wdm6_constants, so we
# pass the WDM7 constants module ``C`` in explicitly where the helper closes
# over module constants. The PLM primitives are pure (no module-constant use)
# so they are imported directly.
from gpuwrf.physics.microphysics_wdm6 import (
    _conden,
    _cpmcal,
    _diffac,
    _lamdac,
    _nislfv_rain_plm6,
    _nislfv_rain_plmr,
    _plm_qmi_qpi,
    _plm_remap,
    _precip_out,
    _qsat1_only,
    _qsat_fpvs,
    _rain_number_fallout,
    _venfac,
    _wi_interp,
    _diffusivity_limit,
    _xka,
    _xlcal,
)


# --------------------------------------------------------------------------
# slope_wdm7: rain(double-moment)/snow/graupel/HAIL slopes + fall speeds.
# Index 0=rain, 1=snow, 2=graupel, 3=hail. Mirrors slope_wdm7. The rain slope
# uses the PREDICTED rain number nr: lamdar = ((pidnr*nr)/(qr*den))^(1/3),
# rslope = min(1/lamdar, 1e-3) (NOTE: rain_ok uses nr>nrmin like WDM6, NOT the
# WSM7 qr-only test). snow/graupel/hail are single-moment. Returns
# rslope/rslopeb/rslope2/rslope3 (K,4), vt (K,4) and vtn (K) (rain-NUMBER fall).
# --------------------------------------------------------------------------
def _slope_wdm7(qr, qs, qg, qh, nr, den, denfac, t):
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)

    # rain (double-moment): lamdar from predicted number nr; rslope clipped 1e-3
    rain_ok = (qr > C.QCRMIN) & (nr > C.NRMIN)
    lamdar = jnp.exp(jnp.log((C.PIDNR * nr) / (qr * den)) * (1.0 / 3.0))
    rsl_r = jnp.where(rain_ok, jnp.minimum(1.0 / lamdar, 1.0e-3), C.RSLOPERMAX)
    rslb_r = jnp.where(rain_ok, rsl_r ** C.BVTR, C.RSLOPERBMAX)
    rsl2_r = jnp.where(rain_ok, rsl_r * rsl_r, C.RSLOPER2MAX)
    rsl3_r = jnp.where(rain_ok, rsl2_r * rsl_r, C.RSLOPER3MAX)

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
    vtn = jnp.where(nr <= 0.0, 0.0, C.PVTRN * rslb_r * denfac)
    return rslope, rslopeb, rslope2, rslope3, vt, vtn


def _slope_hail_vt(q, den, denfac):
    """slope_hail arrival-point terminal velocity for the hail nislfv iter."""
    lamda = jnp.sqrt(jnp.sqrt(C.PIDN0H / (q * den)))
    rsl = jnp.where(q <= C.QCRMIN, C.RSLOPEHMAX, 1.0 / lamda)
    rslb = jnp.where(q <= C.QCRMIN, C.RSLOPEHBMAX, rsl ** C.BVTH)
    return jnp.where(q <= 0.0, 0.0, C.PVTH * rslb * denfac)


def _nislfv_hail(q, den, denfac, tk, dz, ww_in, dt, iter_n):
    """nislfv_rain_plmr id=2 (hail) for one column. Returns (q_new, precip).

    Identical PLM scheme to ``microphysics_wdm6._nislfv_rain_plmr`` (rid=0) but
    the arrival-point terminal-velocity re-evaluation uses the HAIL slope
    (slope_hail via PIDN0H/PVTH). WDM7 calls this with iter=1.
    """
    km = q.shape[0]
    allold = jnp.sum(q)

    zi = jnp.concatenate([jnp.zeros(1, q.dtype), jnp.cumsum(dz)])
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
        wa = _slope_hail_vt(qr_cells, den, denfac)
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
# Effective radii: effectRad_wdm7 == effectRad_wdm6 (byte-identical). Reuse
# the WDM6 implementation (closes over WDM6 constants whose values equal WDM7's).
# --------------------------------------------------------------------------
def _effective_radii(t, qc, nc, qi, qs, rho):
    from gpuwrf.physics.microphysics_wdm6 import _effective_radii as _erad6
    return _erad6(t, qc, nc, qi, qs, rho)


# --------------------------------------------------------------------------
# Deposition/sublimation block: ice/snow/graupel/HAIL with the cumulative ifsat
# short-circuit, then pigen, psaut, pgaut. Mirrors wdm72D lines ~1721-1817.
# (phaut is OUTSIDE this block in WRF -- it fires whenever qg>0, see below.)
# --------------------------------------------------------------------------
def _wdm7_deposition_block(cold, qi, qs, qg, qh, q, diameter, xni, rh2, work1_2,
                           satdt2, prevp, dtcld, n0sfac, rslope, rslopeb,
                           rslope2, work2, den, supcol, supsat2):
    # pidep
    pidep_raw = 4.0 * diameter * xni * (rh2 - 1.0) / work1_2
    supice1 = satdt2 - prevp
    pidep_neg = jnp.maximum(jnp.maximum(jnp.maximum(pidep_raw, satdt2 / 2.0), supice1),
                            -qi / dtcld)
    pidep_pos = jnp.minimum(jnp.minimum(pidep_raw, satdt2 / 2.0), supice1)
    pidep_v = jnp.where(pidep_raw < 0.0, pidep_neg, pidep_pos)
    do_pidep = cold & (qi > 0.0)
    pidep = jnp.where(do_pidep, pidep_v, 0.0)
    ifsat1 = do_pidep & (jnp.abs(prevp + pidep) >= jnp.abs(satdt2))

    # psdep
    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psdep_raw = ((rh2 - 1.0) * n0sfac * (C.PRECS1 * rslope2[:, 1]
                 + C.PRECS2 * work2 * coeres_s) / work1_2)
    supice2 = satdt2 - prevp - pidep
    psdep_neg = jnp.maximum(jnp.maximum(jnp.maximum(psdep_raw, -qs / dtcld),
                                        satdt2 / 2.0), supice2)
    psdep_pos = jnp.minimum(jnp.minimum(psdep_raw, satdt2 / 2.0), supice2)
    psdep_v = jnp.where(psdep_raw < 0.0, psdep_neg, psdep_pos)
    do_psdep = cold & (qs > 0.0) & jnp.logical_not(ifsat1)
    psdep = jnp.where(do_psdep, psdep_v, 0.0)
    ifsat2 = ifsat1 | (do_psdep & (jnp.abs(prevp + pidep + psdep) >= jnp.abs(satdt2)))

    # pgdep
    coeres_g = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgdep_raw = ((rh2 - 1.0) * (C.PRECG1 * rslope2[:, 2]
                 + C.PRECG2 * work2 * coeres_g) / work1_2)
    supice3 = satdt2 - prevp - pidep - psdep
    pgdep_neg = jnp.maximum(jnp.maximum(jnp.maximum(pgdep_raw, -qg / dtcld),
                                        satdt2 / 2.0), supice3)
    pgdep_pos = jnp.minimum(jnp.minimum(pgdep_raw, satdt2 / 2.0), supice3)
    pgdep_v = jnp.where(pgdep_raw < 0.0, pgdep_neg, pgdep_pos)
    do_pgdep = cold & (qg > 0.0) & jnp.logical_not(ifsat2)
    pgdep = jnp.where(do_pgdep, pgdep_v, 0.0)
    ifsat3 = ifsat2 | (do_pgdep & (jnp.abs(prevp + pidep + psdep + pgdep) >= jnp.abs(satdt2)))

    # phdep
    coeres_h = rslope2[:, 3] * jnp.sqrt(rslope[:, 3] * rslopeb[:, 3])
    phdep_raw = ((rh2 - 1.0) * (C.PRECH1 * rslope2[:, 3]
                 + C.PRECH2 * work2 * coeres_h) / work1_2)
    supice4 = satdt2 - prevp - pidep - psdep - pgdep
    phdep_neg = jnp.maximum(jnp.maximum(jnp.maximum(phdep_raw, -qh / dtcld),
                                        satdt2 / 2.0), supice4)
    phdep_pos = jnp.minimum(jnp.minimum(phdep_raw, satdt2 / 2.0), supice4)
    phdep_v = jnp.where(phdep_raw < 0.0, phdep_neg, phdep_pos)
    do_phdep = cold & (qh > 0.0) & jnp.logical_not(ifsat3)
    phdep = jnp.where(do_phdep, phdep_v, 0.0)
    ifsat4 = ifsat3 | (do_phdep & (jnp.abs(prevp + pidep + psdep + pgdep + phdep) >= jnp.abs(satdt2)))

    # pigen (WRF supice = satdt-prevp-pidep-psdep-pgdep-phdep)
    supice5 = satdt2 - prevp - pidep - psdep - pgdep - phdep
    xni0 = 1.0e3 * jnp.exp(0.1 * supcol)
    roqi0 = 4.92e-11 * xni0 ** 1.33
    pigen_raw = jnp.maximum(0.0, (roqi0 / den - jnp.maximum(qi, 0.0)) / dtcld)
    pigen_v = jnp.minimum(jnp.minimum(pigen_raw, satdt2), supice5)
    do_pigen = cold & (supsat2 > 0.0) & jnp.logical_not(ifsat4)
    pigen = jnp.where(do_pigen, pigen_v, 0.0)

    # psaut (qi->qs aggregation)
    qimax = C.ROQIMAX / den
    psaut_v = jnp.maximum(0.0, (qi - qimax) / dtcld)
    psaut = jnp.where(cold & (qi > 0.0), psaut_v, 0.0)

    # pgaut (qs->qg aggregation)
    alpha2 = 1.0e-3 * jnp.exp(0.09 * (-supcol))
    pgaut_v = jnp.minimum(jnp.maximum(0.0, alpha2 * (qs - C.QS0)), qs / dtcld)
    pgaut = jnp.where(cold & (qs > 0.0), pgaut_v, 0.0)

    return pidep, psdep, pgdep, phdep, pigen, psaut, pgaut


# --------------------------------------------------------------------------
# Core single-column WDM7 (wdm72D for one column, all minor loops)
# --------------------------------------------------------------------------
def _wdm7_column(t, q, qc, qr, qi, qs, qg, qh, nn, nc, nr, den, p, delz, delt, slmsk):
    # padding from dynamics (wdm72D lines ~557-568)
    qc = jnp.maximum(qc, 0.0)
    qr = jnp.maximum(qr, 0.0)
    qi = jnp.maximum(qi, 0.0)
    qs = jnp.maximum(qs, 0.0)
    qg = jnp.maximum(qg, 0.0)
    qh = jnp.maximum(qh, 0.0)
    nn = jnp.minimum(jnp.maximum(nn, 1.0e8), 2.0e10)
    nc = jnp.maximum(nc, 0.0)
    nr = jnp.maximum(nr, 0.0)

    cpm = _cpmcal(q)
    xl = _xlcal(t)
    # qcr threshold: slmsk==2 (water) -> qc0, else qc1
    qcr = jnp.where(slmsk == 2.0, C.QC0, C.QC1)

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

    state = (t, q, qc, qr, qi, qs, qg, qh, nn, nc, nr, cpm, xl,
             rainncv, snowncv, graupelncv, hailncv,
             tstepsnow, tstepgraup, tstephail, sr)

    for _loop in range(loops):
        state = _wdm7_minor_loop(state, den, p, delz, dtcld, qcr)

    (t, q, qc, qr, qi, qs, qg, qh, nn, nc, nr, cpm, xl,
     rainncv, snowncv, graupelncv, hailncv,
     tstepsnow, tstepgraup, tstephail, sr) = state

    re_qc, re_qi, re_qs = _effective_radii(t, qc, nc, qi, qs, den)

    return {
        "t": t, "qv": q, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "qh": qh,
        "nn": nn, "nc": nc, "nr": nr,
        "rainncv": rainncv, "snowncv": snowncv, "graupelncv": graupelncv,
        "hailncv": hailncv, "sr": sr,
        "re_cloud": re_qc, "re_ice": re_qi, "re_snow": re_qs,
    }


def _wdm7_minor_loop(state, den, p, delz, dtcld, qcr):
    (t, q, qc, qr, qi, qs, qg, qh, nn, nc, nr, cpm, xl,
     rainncv, snowncv, graupelncv, hailncv,
     tstepsnow, tstepgraup, tstephail, sr) = state
    qmin = C.QMIN

    denfac = jnp.sqrt(C.DEN0 / den)

    qsat1, qsat2 = _qsat_fpvs(t, p)
    rh1 = jnp.maximum(q / qsat1, qmin)
    rh2 = jnp.maximum(q / qsat2, qmin)

    # rslopec (cloud-droplet slope) from qc,nc
    cloud_ok = (qc > qmin) & (nc > C.NCMIN)
    rslopec = jnp.where(cloud_ok, 1.0 / _lamdac(qc, den, nc), C.RSLOPECMAX)
    rslopec2 = jnp.where(cloud_ok, rslopec * rslopec, C.RSLOPEC2MAX)
    rslopec3 = jnp.where(cloud_ok, rslopec2 * rslopec, C.RSLOPEC3MAX)

    # ice crystal number conc (HDC 5c)
    temp = den * jnp.maximum(qi, qmin)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    xni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)

    # ---- fallout: rain + Nr via in-line mstep loop (wdm72D ~785-857) ----
    rslope, rslopeb, rslope2, rslope3, work1_vt, workn = _slope_wdm7(
        qr, qs, qg, qh, nr, den, denfac, t)
    work1_r = work1_vt[:, 0] / delz   # rain mass fall speed / delz
    workn_r = workn / delz            # rain number fall speed / delz
    numdt = jnp.maximum(jnp.round(jnp.maximum(work1_r, workn_r) * dtcld + 0.5), 1.0)
    mstep = jnp.maximum(jnp.max(numdt), 1.0)

    qr, nr, fall1, falln = _rain_number_fallout(
        qr, nr, qs, qg, den, denfac, t, delz, dtcld, work1_r, workn_r, mstep)

    # snow+graupel semi-Lagrangian (plm6); hail (plmr id=2, iter=1) (~861-898)
    workh = jnp.where(qh <= 0.0, 0.0, work1_vt[:, 3])
    qsum = jnp.maximum(qs + qg, 1.0e-15)
    worka = jnp.where(qsum > 1.0e-15,
                      (work1_vt[:, 1] * qs + work1_vt[:, 2] * qg) / qsum, 0.0)
    denqrs2 = den * qs
    denqrs3 = den * qg
    denqrs4 = den * qh
    denqrs2_new, denqrs3_new, delqrs2, delqrs3 = _nislfv_rain_plm6(
        denqrs2, denqrs3, den, denfac, t, delz, worka, dtcld, 1)
    denqrs4_new, delqrs4 = _nislfv_hail(denqrs4, den, denfac, t, delz, workh, dtcld, 1)
    qs = jnp.maximum(denqrs2_new / den, 0.0)
    qg = jnp.maximum(denqrs3_new / den, 0.0)
    qh = jnp.maximum(denqrs4_new / den, 0.0)
    fall2 = denqrs2_new * worka / delz
    fall3 = denqrs3_new * worka / delz
    fall4 = denqrs4_new * workh / delz
    fall2 = fall2.at[0].set(delqrs2 / delz[0] / dtcld)
    fall3 = fall3.at[0].set(delqrs3 / delz[0] / dtcld)
    fall4 = fall4.at[0].set(delqrs4 / delz[0] / dtcld)

    # slope recompute on post-sedimentation qrs (wdm72D ~910)
    rslope, rslopeb, rslope2, rslope3, work1_vt, workn = _slope_wdm7(
        qr, qs, qg, qh, nr, den, denfac, t)

    # ---- snow/graupel/hail melting for T>T0 (psmlt/pgmlt/phmlt + numbers) ----
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    warm = t > C.T0C
    xlf = C.XLF0
    work2 = _venfac(p, t, den)
    # psmlt
    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psmlt = (_xka(t, den) / xlf * (C.T0C - t) * C.PI / 2.0 * n0sfac
             * (C.PRECS1 * rslope2[:, 1] + C.PRECS2 * work2 * coeres_s) / den)
    psmlt = jnp.minimum(jnp.maximum(psmlt * dtcld, -qs), 0.0)
    do_psmlt = warm & (qs > 0.0)
    sfac_s = rslope[:, 1] * C.N0S * n0sfac / jnp.where(qs > 0.0, qs, 1.0)
    nr = jnp.where(do_psmlt & (qs > C.QCRMIN), nr - sfac_s * psmlt, nr)
    qs = jnp.where(do_psmlt, qs + psmlt, qs)
    qr = jnp.where(do_psmlt, qr - psmlt, qr)
    t = jnp.where(do_psmlt, t + xlf / cpm * psmlt, t)
    # pgmlt
    coeres_g = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgmlt = (_xka(t, den) / xlf * (C.T0C - t)
             * (C.PRECG1 * rslope2[:, 2] + C.PRECG2 * work2 * coeres_g) / den)
    pgmlt = jnp.minimum(jnp.maximum(pgmlt * dtcld, -qg), 0.0)
    do_pgmlt = warm & (qg > 0.0)
    gfac_g = rslope[:, 2] * C.N0G / jnp.where(qg > 0.0, qg, 1.0)
    nr = jnp.where(do_pgmlt & (qg > C.QCRMIN), nr - gfac_g * pgmlt, nr)
    qg = jnp.where(do_pgmlt, qg + pgmlt, qg)
    qr = jnp.where(do_pgmlt, qr - pgmlt, qr)
    t = jnp.where(do_pgmlt, t + xlf / cpm * pgmlt, t)
    # phmlt (NOTE: WRF divides by mstep(i) for both the rate and the cap)
    coeres_h = rslope2[:, 3] * jnp.sqrt(rslope[:, 3] * rslopeb[:, 3])
    phmlt = (_xka(t, den) / xlf * (C.T0C - t)
             * (C.PRECH1 * rslope2[:, 3] + C.PRECH2 * work2 * coeres_h) / den)
    phmlt = jnp.minimum(jnp.maximum(phmlt * dtcld / mstep, -qh / mstep), 0.0)
    do_phmlt = warm & (qh > 0.0)
    qh = jnp.where(do_phmlt, qh + phmlt, qh)
    qr = jnp.where(do_phmlt, qr - phmlt, qr)
    gfac_h = rslope[:, 3] * C.N0H / jnp.where(qh > 0.0, qh, 1.0)
    nr = jnp.where(do_phmlt & (qh > C.QCRMIN), nr - gfac_h * phmlt, nr)
    t = jnp.where(do_phmlt, t + xlf / cpm * phmlt, t)

    # ---- ice fallout (Vice via plmr, rid=0, iter=0) ----
    xmi = den * qi / xni
    diameter = jnp.maximum(jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX), 1.0e-25)
    work1c = jnp.where(qi <= 0.0, 0.0, 1.49e4 * jnp.exp(jnp.log(diameter) * 1.31))
    denqci = den * qi
    denqci_new, delqi = _nislfv_rain_plmr(
        denqci, denqci, den, denfac, t, delz, work1c, dtcld, 0, 0)
    qi = jnp.maximum(denqci_new / den, 0.0)
    fallc0 = delqi / delz[0] / dtcld

    # ---- surface precip accumulation (rain/snow/graupel/hail/sr) ----
    fallsum = fall1[0] + fall2[0] + fall3[0] + fall4[0] + fallc0
    fallsum_qsi = fall2[0] + fallc0
    fallsum_qg = fall3[0]
    fallsum_qh = fall4[0]
    add = fallsum * delz[0] / C.DENR * dtcld * 1000.0
    rainncv = jnp.where(fallsum > 0.0, add + rainncv, rainncv)
    add_si = fallsum_qsi * delz[0] / C.DENR * dtcld * 1000.0
    tstepsnow = jnp.where(fallsum_qsi > 0.0, add_si + tstepsnow, tstepsnow)
    snowncv = jnp.where(fallsum_qsi > 0.0, add_si + snowncv, snowncv)
    add_g = fallsum_qg * delz[0] / C.DENR * dtcld * 1000.0
    tstepgraup = jnp.where(fallsum_qg > 0.0, add_g + tstepgraup, tstepgraup)
    graupelncv = jnp.where(fallsum_qg > 0.0, add_g + graupelncv, graupelncv)
    add_h = fallsum_qh * delz[0] / C.DENR * dtcld * 1000.0
    tstephail = jnp.where(fallsum_qh > 0.0, add_h + tstephail, tstephail)
    hailncv = jnp.where(fallsum_qh > 0.0, add_h + hailncv, hailncv)
    sr = jnp.where(fallsum > 0.0,
                   (tstepsnow + tstepgraup + tstephail) / (rainncv + 1.0e-12), sr)

    # ---- instantaneous melt/freeze (pimlt/nimlt, pihmf/nihmf, pihtf/nihtf,
    #      pgfrz/ngfrz) ----
    supcol = C.T0C - t
    xlf_im = C.XLS - xl
    xlf_im = jnp.where(supcol < 0.0, C.XLF0, xlf_im)
    # pimlt: I->C for supcol<0 ; nimlt: ->NC (Nc += xni)
    do_im = (supcol < 0.0) & (qi > 0.0)
    qc = jnp.where(do_im, qc + qi, qc)
    nc = jnp.where(do_im, nc + xni, nc)
    t = jnp.where(do_im, t - xlf_im / cpm * qi, t)
    qi = jnp.where(do_im, 0.0, qi)
    # pihmf: C->I for supcol>40 ; nihmf: NC->0
    do_hmf = (supcol > 40.0) & (qc > 0.0)
    qi = jnp.where(do_hmf, qi + qc, qi)
    nc = jnp.where(do_hmf & (nc > 0.0), 0.0, nc)
    t = jnp.where(do_hmf, t + xlf_im / cpm * qc, t)
    qc = jnp.where(do_hmf, 0.0, qc)
    # pihtf: heterogeneous freezing C->I for supcol>0 ; nihtf: NC reduced
    supcolt = jnp.minimum(supcol, 70.0)
    pfrzdtc = jnp.minimum(
        C.PI * C.PI * C.PFRZ1 * (jnp.exp(C.PFRZ2 * supcolt) - 1.0) * C.DENR / den
        * nc * rslopec3 * rslopec3 / 18.0 * dtcld, qc)
    do_htf = (supcol > 0.0) & (qc > qmin)
    nfrzdtc = jnp.minimum(
        C.PI * C.PFRZ1 * (jnp.exp(C.PFRZ2 * supcolt) - 1.0) * nc
        * rslopec3 / 6.0 * dtcld, nc)
    nc = jnp.where(do_htf & (nc > C.NCMIN), nc - nfrzdtc, nc)
    qi = jnp.where(do_htf, qi + pfrzdtc, qi)
    t = jnp.where(do_htf, t + xlf_im / cpm * pfrzdtc, t)
    qc = jnp.where(do_htf, qc - pfrzdtc, qc)
    # pgfrz: rain->graupel for supcol>0 ; ngfrz: NR reduced
    rs3r = rslope3[:, 0]
    pfrzdtr = jnp.minimum(
        140.0 * (C.PI * C.PI) * C.PFRZ1 * nr * C.DENR / den
        * (jnp.exp(C.PFRZ2 * supcolt) - 1.0) * rs3r * rs3r * dtcld, qr)
    do_gfrz = (supcol > 0.0) & (qr > 0.0)
    nfrzdtr = jnp.minimum(
        4.0 * C.PI * C.PFRZ1 * nr * (jnp.exp(C.PFRZ2 * supcolt) - 1.0)
        * rs3r * dtcld, nr)
    nr = jnp.where(do_gfrz & (nr > C.NRMIN), nr - nfrzdtr, nr)
    qg = jnp.where(do_gfrz, qg + pfrzdtr, qg)
    t = jnp.where(do_gfrz, t + xlf_im / cpm * pfrzdtr, t)
    qr = jnp.where(do_gfrz, qr - pfrzdtr, qr)

    # clamp Nc,Nr >= 0 (wdm72D ~1148-1153)
    nc = jnp.maximum(nc, 0.0)
    nr = jnp.maximum(nr, 0.0)

    # ---- slope update + avedia + rslopec + work1(diffac)/work2(venfac) ----
    rslope, rslopeb, rslope2, rslope3, _vt, _vtn = _slope_wdm7(
        qr, qs, qg, qh, nr, den, denfac, t)
    avedia_r = rslope[:, 0] * (24.0 ** (1.0 / 3.0))   # mean-volume rain diam
    # WRF reuses THIS slope (qrs_tmp/ncr_tmp from the pre-microphysics state,
    # last set at module_mp_wdm7.F line ~1160) for the post-update small-drop
    # avedia at line ~2222 -- it is NOT recomputed from the post-conservation
    # qr/nr. Snapshot the pre-update rain slope so the small-drop test below
    # matches WRF (else a cell whose rain is depleted by the conservation update
    # gets a tiny post-update avedia and wrongly converts rain->cloud).
    rslope_preupd_r = rslope[:, 0]
    cloud_ok = (qc > qmin) & (nc > C.NCMIN)
    rslopec = jnp.where(cloud_ok, 1.0 / _lamdac(qc, den, nc), C.RSLOPECMAX)
    rslopec2 = jnp.where(cloud_ok, rslopec * rslopec, C.RSLOPEC2MAX)
    rslopec3 = jnp.where(cloud_ok, rslopec2 * rslopec, C.RSLOPEC3MAX)
    avedia_c = rslopec
    work1_1 = _diffac(xl, p, t, den, qsat1)
    work1_2 = _diffac(C.XLS, p, t, den, qsat2)
    work2 = _venfac(p, t, den)

    # ===================== WARM RAIN (double-moment) =====================
    supsat = jnp.maximum(q, qmin) - qsat1
    satdt = supsat / dtcld

    # praut + nraut (LH9/CP17, LH A6/CP18&19)
    lencon = 2.7e-2 * den * qc * (1.0e20 / 16.0 * rslopec2 * rslopec2 - 0.4)
    lenconcr = jnp.maximum(1.2 * lencon, C.QCRMIN)
    praut_ok = (qc > qcr) & (nc > C.NCMIN)
    praut = jnp.where(praut_ok,
                      jnp.minimum(C.QCK1 * qc ** (7.0 / 3.0) * nc ** (-1.0 / 3.0),
                                  qc / dtcld), 0.0)
    nraut_base = 3.5e9 * den * praut
    nraut_alt = jnp.where(qr > lenconcr, nr / jnp.where(qr > 0.0, qr, 1.0) * praut,
                          nraut_base)
    nraut = jnp.where(praut_ok, jnp.minimum(nraut_alt, nc / dtcld), 0.0)

    # pracw + nracw (LH10/CP22&23, LH A9): two branches by avedia_r vs di100
    big = avedia_r >= C.DI100
    nracw_big = jnp.minimum(C.NCRK1 * nc * nr * (rslopec3 + 24.0 * rslope3[:, 0]),
                            nc / dtcld)
    pracw_big = jnp.minimum(C.PI / 6.0 * (C.DENR / den) * C.NCRK1 * nc * nr
                            * rslopec3 * (2.0 * rslopec3 + 24.0 * rslope3[:, 0]),
                            qc / dtcld)
    nracw_sml = jnp.minimum(C.NCRK2 * nc * nr * (2.0 * rslopec3 * rslopec3
                            + 5040.0 * rslope3[:, 0] * rslope3[:, 0]), nc / dtcld)
    pracw_sml = jnp.minimum(C.PI / 6.0 * (C.DENR / den) * C.NCRK2 * nc * nr
                            * rslopec3 * (6.0 * rslopec3 * rslopec3
                            + 5040.0 * rslope3[:, 0] * rslope3[:, 0]), qc / dtcld)
    pracw_v = jnp.where(big, pracw_big, pracw_sml)
    nracw_v = jnp.where(big, nracw_big, nracw_sml)
    pracw_cond = qr >= lenconcr
    pracw = jnp.where(pracw_cond, pracw_v, 0.0)
    nracw = jnp.where(pracw_cond, nracw_v, 0.0)

    # nccol: self-collection of cloud (LH A8/CP24&25)
    bigc = avedia_c >= C.DI100
    nccol = jnp.where(bigc, C.NCRK1 * nc * nc * rslopec3,
                      2.0 * C.NCRK2 * nc * nc * rslopec3 * rslopec3)

    # nrcol: self-collection of rain + breakup (LH A21/CP24&25), 4 branches
    nrcol_sml = 5040.0 * C.NCRK2 * nr * nr * rslope3[:, 0] * rslope3[:, 0]
    nrcol_mid = 24.0 * C.NCRK1 * nr * nr * rslope3[:, 0]
    coecol = -2.5e3 * (avedia_r - C.DI600)
    nrcol_big = 24.0 * jnp.exp(coecol) * C.NCRK1 * nr * nr * rslope3[:, 0]
    nrcol_v = jnp.where(avedia_r < C.DI100, nrcol_sml,
                jnp.where(avedia_r < C.DI600, nrcol_mid,
                  jnp.where(avedia_r < C.DI2000, nrcol_big, 0.0)))
    nrcol = jnp.where(qr >= lenconcr, nrcol_v, 0.0)

    # prevp (HL A41) + Nrevp(NR->NCCN) handled immediately (load-bearing)
    coeres_r = rslope[:, 0] * jnp.sqrt(rslope[:, 0] * rslopeb[:, 0])
    prevp_raw = ((rh1 - 1.0) * nr * (C.PRECR1 * rslope[:, 0]
                 + C.PRECR2 * work2 * coeres_r) / work1_1)
    prevp_neg = jnp.maximum(jnp.maximum(prevp_raw, -qr / dtcld), satdt / 2.0)
    prevp_pos = jnp.minimum(prevp_raw, satdt / 2.0)
    prevp = jnp.where(prevp_raw < 0.0, prevp_neg, prevp_pos)
    prevp = jnp.where(qr > 0.0, prevp, 0.0)
    nrevp_full = (qr > 0.0) & (prevp_raw < 0.0) & (prevp == (-qr / dtcld))
    nn = jnp.where(nrevp_full, nn + nr, nn)
    nr = jnp.where(nrevp_full, 0.0, nr)

    # ===================== COLD RAIN =====================
    supcol = C.T0C - t
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

    cold = supcol > 0.0
    ice_present = cold & (qi > qmin)
    safe_qi = jnp.where(qi > 0.0, qi, 1.0)
    safe_qr = jnp.where(qr > 0.0, qr, 1.0)
    safe_qs = jnp.where(qs > 0.0, qs, 1.0)
    safe_qc = jnp.where(qc > 0.0, qc, 1.0)
    safe_qg = jnp.where(qg > 0.0, qg, 1.0)
    safe_qh = jnp.where(qh > 0.0, qh, 1.0)

    # praci, piacr, niacr (need qr>qcrmin)
    acrfac_r = 6.0 * rslope2[:, 0] + 4.0 * diameter * rslope[:, 0] + diameter ** 2
    praci_raw = C.PI * qi * nr * jnp.abs(vt2r - vt2i) * acrfac_r / 4.0
    praci_raw = praci_raw * jnp.minimum(jnp.maximum(0.0, qr / safe_qi), 1.0) ** 2
    praci_raw = jnp.minimum(praci_raw, qi / dtcld)
    piacr_raw = (C.PI * C.PI * C.AVTR * nr * C.DENR * xni * denfac * C.G7PBR
                 * rslope3[:, 0] * rslope2[:, 0] * rslopeb[:, 0] / 24.0 / den)
    piacr_raw = piacr_raw * jnp.minimum(jnp.maximum(0.0, qi / safe_qr), 1.0) ** 2
    piacr_raw = jnp.minimum(piacr_raw, qr / dtcld)
    raci_cond = ice_present & (qr > C.QCRMIN)
    praci = jnp.where(raci_cond, praci_raw, 0.0)
    piacr = jnp.where(raci_cond, piacr_raw, 0.0)
    niacr_raw = (C.PI * C.AVTR * nr * xni * denfac * C.G4PBR
                 * rslope2[:, 0] * rslopeb[:, 0] / 4.0)
    niacr_raw = niacr_raw * jnp.minimum(jnp.maximum(0.0, qi / safe_qr), 1.0) ** 2
    niacr_raw = jnp.minimum(niacr_raw, nr / dtcld)
    niacr = jnp.where(ice_present & (nr > C.NRMIN), niacr_raw, 0.0)

    # psaci (qs>qcrmin)
    acrfac_s = (2.0 * rslope3[:, 1] + 2.0 * diameter * rslope2[:, 1]
                + diameter ** 2 * rslope[:, 1])
    psaci_raw = (C.PI * qi * eacrs * C.N0S * n0sfac * jnp.abs(vt2ave - vt2i)
                 * acrfac_s / 4.0)
    psaci_raw = jnp.minimum(psaci_raw, qi / dtcld)
    psaci = jnp.where(ice_present & (qs > C.QCRMIN), psaci_raw, 0.0)

    # pgaci (qg>qcrmin)
    acrfac_g = (2.0 * rslope3[:, 2] + 2.0 * diameter * rslope2[:, 2]
                + diameter ** 2 * rslope[:, 2])
    egi = jnp.exp(0.07 * (-supcol))
    pgaci_raw = C.PI * egi * qi * C.N0G * jnp.abs(vt2ave - vt2i) * acrfac_g / 4.0
    pgaci_raw = jnp.minimum(pgaci_raw, qi / dtcld)
    pgaci = jnp.where(ice_present & (qg > C.QCRMIN), pgaci_raw, 0.0)

    # phaci (qh>qcrmin)
    acrfac_h = (2.0 * rslope3[:, 3] + 2.0 * diameter * rslope2[:, 3]
                + diameter ** 2 * rslope[:, 3])
    ehi = jnp.exp(0.07 * (-supcol))
    phaci_raw = C.PI * ehi * qi * C.N0H * jnp.abs(vt2h - vt2i) * acrfac_h / 4.0
    phaci_raw = jnp.minimum(phaci_raw, qi / dtcld)
    phaci = jnp.where(ice_present & (qh > C.QCRMIN), phaci_raw, 0.0)

    # psacw + nsacw (qs>qcrmin & qc>qmin / nc>ncmin)
    psacw_raw = jnp.minimum(
        C.PACRC * n0sfac * rslope3[:, 1] * rslopeb[:, 1]
        * jnp.minimum(jnp.maximum(0.0, qs / safe_qc), 1.0) ** 2 * qc * denfac,
        qc / dtcld)
    psacw = jnp.where((qs > C.QCRMIN) & (qc > qmin), psacw_raw, 0.0)
    nsacw_raw = jnp.minimum(
        C.PACRC * n0sfac * rslope3[:, 1] * rslopeb[:, 1]
        * jnp.minimum(jnp.maximum(0.0, qs / safe_qc), 1.0) ** 2 * nc * denfac,
        nc / dtcld)
    nsacw = jnp.where((qs > C.QCRMIN) & (nc > C.NCMIN), nsacw_raw, 0.0)

    # pgacw + ngacw (qg>qcrmin & qc>qmin / nc>ncmin)
    pgacw_raw = jnp.minimum(
        C.PACRG * rslope3[:, 2] * rslopeb[:, 2]
        * jnp.minimum(jnp.maximum(0.0, qg / safe_qc), 1.0) ** 2 * qc * denfac,
        qc / dtcld)
    pgacw = jnp.where((qg > C.QCRMIN) & (qc > qmin), pgacw_raw, 0.0)
    ngacw_raw = jnp.minimum(
        C.PACRG * rslope3[:, 2] * rslopeb[:, 2]
        * jnp.minimum(jnp.maximum(0.0, qg / safe_qc), 1.0) ** 2 * nc * denfac,
        nc / dtcld)
    ngacw = jnp.where((qg > C.QCRMIN) & (nc > C.NCMIN), ngacw_raw, 0.0)

    # paacw + naacw
    paacw = jnp.where(qsum > 1.0e-15, (qs * psacw + qg * pgacw) / qsum, 0.0)
    naacw = jnp.where(qsum > 1.0e-15, (qs * nsacw + qg * ngacw) / qsum, 0.0)

    # phacw + nhacw (qh>qcrmin & qc>qmin / nc>ncmin)
    phacw_raw = jnp.minimum(
        C.PACRH * rslope3[:, 3] * rslopeb[:, 3]
        * jnp.minimum(jnp.maximum(0.0, qh / safe_qc), 1.0) ** 2 * qc * denfac,
        qc / dtcld)
    phacw = jnp.where((qh > C.QCRMIN) & (qc > qmin), phacw_raw, 0.0)
    nhacw_raw = jnp.minimum(
        C.PACRH * rslope3[:, 3] * rslopeb[:, 3]
        * jnp.minimum(jnp.maximum(0.0, qh / safe_qc), 1.0) ** 2 * nc * denfac,
        nc / dtcld)
    nhacw = jnp.where((qh > C.QCRMIN) & (nc > C.NCMIN), nhacw_raw, 0.0)

    # pracs (qs>qcrmin & qr>qcrmin & supcol>0)
    acrfac_pracs = (5.0 * rslope3[:, 1] * rslope3[:, 1]
                    + 4.0 * rslope3[:, 1] * rslope2[:, 1] * rslope[:, 0]
                    + 1.5 * rslope2[:, 1] * rslope2[:, 1] * rslope2[:, 0])
    pracs_raw = (C.PI * C.PI * nr * C.N0S * n0sfac * jnp.abs(vt2r - vt2ave)
                 * (C.DENS / den) * acrfac_pracs)
    pracs_raw = pracs_raw * jnp.minimum(jnp.maximum(0.0, qr / safe_qs), 1.0) ** 2
    pracs_raw = jnp.minimum(pracs_raw, qs / dtcld)
    sr_qr_cond = (qs > C.QCRMIN) & (qr > C.QCRMIN)
    pracs = jnp.where(sr_qr_cond & (supcol > 0.0), pracs_raw, 0.0)

    # psacr (qs>qcrmin & qr>qcrmin)
    acrfac_psacr = (30.0 * rslope3[:, 0] * rslope2[:, 0] * rslope[:, 1]
                    + 10.0 * rslope2[:, 0] * rslope2[:, 0] * rslope2[:, 1]
                    + 2.0 * rslope3[:, 0] * rslope3[:, 1])
    psacr_raw = (C.PI * C.PI * nr * C.N0S * n0sfac * jnp.abs(vt2ave - vt2r)
                 * (C.DENR / den) * acrfac_psacr)
    psacr_raw = psacr_raw * jnp.minimum(jnp.maximum(0.0, qs / safe_qr), 1.0) ** 2
    psacr_raw = jnp.minimum(psacr_raw, qr / dtcld)
    psacr = jnp.where(sr_qr_cond, psacr_raw, 0.0)

    # nsacr (qs>qcrmin & nr>nrmin)
    acrfac_nsacr = (1.5 * rslope2[:, 0] * rslope[:, 1]
                    + 1.0 * rslope[:, 0] * rslope2[:, 1] + 0.5 * rslope3[:, 1])
    nsacr_raw = (C.PI * nr * C.N0S * n0sfac * jnp.abs(vt2ave - vt2r) * acrfac_nsacr)
    nsacr_raw = nsacr_raw * jnp.minimum(jnp.maximum(0.0, qs / safe_qr), 1.0) ** 2
    nsacr_raw = jnp.minimum(nsacr_raw, nr / dtcld)
    nsacr = jnp.where((qs > C.QCRMIN) & (nr > C.NRMIN), nsacr_raw, 0.0)

    # pracg (qg>qcrmin & qr>qcrmin & supcol>0): G->H
    acrfac_pracg = (5.0 * rslope3[:, 2] * rslope3[:, 2]
                    + 4.0 * rslope3[:, 2] * rslope2[:, 2] * rslope[:, 0]
                    + 1.5 * rslope2[:, 2] * rslope2[:, 2] * rslope2[:, 0])
    pracg_raw = (C.PI * C.PI * nr * C.N0G * jnp.abs(vt2r - vt2ave)
                 * (C.DENG / den) * acrfac_pracg)
    pracg_raw = pracg_raw * jnp.minimum(jnp.maximum(0.0, qr / safe_qg), 1.0) ** 2
    pracg_raw = jnp.minimum(pracg_raw, qg / dtcld)
    grr_cond = (qg > C.QCRMIN) & (qr > C.QCRMIN)
    pracg = jnp.where(grr_cond & (supcol > 0.0), pracg_raw, 0.0)

    # pgacr (qg>qcrmin & qr>qcrmin)
    acrfac_pgacr = (30.0 * rslope3[:, 0] * rslope2[:, 0] * rslope[:, 2]
                    + 10.0 * rslope2[:, 0] * rslope2[:, 0] * rslope2[:, 2]
                    + 2.0 * rslope3[:, 0] * rslope3[:, 2])
    pgacr_raw = (C.PI * C.PI * nr * C.N0G * jnp.abs(vt2ave - vt2r)
                 * (C.DENR / den) * acrfac_pgacr)
    pgacr_raw = pgacr_raw * jnp.minimum(jnp.maximum(0.0, qg / safe_qr), 1.0) ** 2
    pgacr_raw = jnp.minimum(pgacr_raw, qr / dtcld)
    pgacr = jnp.where(grr_cond, pgacr_raw, 0.0)

    # ngacr (qg>qcrmin & nr>nrmin)
    acrfac_ngacr = (1.5 * rslope2[:, 0] * rslope[:, 2]
                    + 1.0 * rslope[:, 0] * rslope2[:, 2] + 0.5 * rslope3[:, 2])
    ngacr_raw = C.PI * nr * C.N0G * jnp.abs(vt2ave - vt2r) * acrfac_ngacr
    ngacr_raw = ngacr_raw * jnp.minimum(jnp.maximum(0.0, qg / safe_qr), 1.0) ** 2
    ngacr_raw = jnp.minimum(ngacr_raw, nr / dtcld)
    ngacr = jnp.where((qg > C.QCRMIN) & (nr > C.NRMIN), ngacr_raw, 0.0)

    # pgacs = 0 (eliminated in V3.0)
    pgacs = jnp.zeros_like(t)

    # phacr + nhacr (qh>qcrmin & qr>qcrmin): R->H
    acrfac_phacr = (30.0 * rslope3[:, 0] * rslope2[:, 0] * rslope[:, 3]
                    + 10.0 * rslope3[:, 0] * rslope[:, 0] * rslope2[:, 3]
                    + 2.0 * rslope3[:, 0] * rslope3[:, 3])
    phacr_raw = (C.PI * C.PI * nr * C.N0H * jnp.abs(vt2h - vt2r)
                 * (C.DENR / den) * acrfac_phacr)
    phacr_raw = phacr_raw * jnp.minimum(jnp.maximum(0.0, qh / safe_qr), 1.0) ** 2
    phacr_raw = jnp.minimum(phacr_raw, qr / dtcld)
    hrr_cond = (qh > C.QCRMIN) & (qr > C.QCRMIN)
    phacr = jnp.where(hrr_cond, phacr_raw, 0.0)
    acrfac_nhacr = (1.5 * rslope2[:, 0] * rslope[:, 3]
                    + 1.0 * rslope[:, 0] * rslope2[:, 3] + 0.5 * rslope3[:, 3])
    nhacr_raw = C.PI * nr * C.N0H * jnp.abs(vt2h - vt2r) * acrfac_nhacr
    nhacr_raw = nhacr_raw * jnp.minimum(jnp.maximum(0.0, qh / safe_qr), 1.0) ** 2
    nhacr_raw = jnp.minimum(nhacr_raw, nr / dtcld)
    nhacr = jnp.where((qh > C.QCRMIN) & (nr > C.NRMIN), nhacr_raw, 0.0)

    # phacs (qh>qcrmin & qs>qcrmin): S->H
    acrfac_phacs = (5.0 * rslope3[:, 1] * rslope3[:, 1] * rslope[:, 3]
                    + 2.0 * rslope3[:, 1] * rslope2[:, 1] * rslope2[:, 3]
                    + 0.5 * rslope2[:, 1] * rslope2[:, 1] * rslope3[:, 3])
    phacs_raw = (C.PI ** 2 * C.EACHS * C.N0S * n0sfac * C.N0H * jnp.abs(vt2h - vt2ave)
                 * (C.DENS / den) * acrfac_phacs)
    phacs_raw = jnp.minimum(phacs_raw, qs / dtcld)
    phacs = jnp.where((qh > C.QCRMIN) & (qs > C.QCRMIN), phacs_raw, 0.0)

    # phacg (qh>qcrmin & qg>qcrmin): G->H
    acrfac_phacg = (5.0 * rslope3[:, 2] * rslope3[:, 2] * rslope[:, 3]
                    + 2.0 * rslope3[:, 2] * rslope2[:, 2] * rslope2[:, 3]
                    + 0.5 * rslope2[:, 2] * rslope2[:, 2] * rslope3[:, 3])
    phacg_raw = (C.PI ** 2 * C.EACHG * C.N0G * C.N0H * jnp.abs(vt2h - vt2ave)
                 * (C.DENG / den) * acrfac_phacg)
    phacg_raw = jnp.minimum(phacg_raw, qg / dtcld)
    phacg = jnp.where((qh > C.QCRMIN) & (qg > C.QCRMIN), phacg_raw, 0.0)

    # ----- pgwet / phwet wet growth (rs0 from fpvs liquid at T0C) -----
    ttp = C.T0C + 0.01
    dldt = C.CPV - C.CLIQ
    xa = -dldt / C.RV
    xb = xa + C.XLV0 / (C.RV * ttp)
    rs0 = C.PSAT * jnp.exp(jnp.log(ttp / C.T0C) * xa) * jnp.exp(xb * (1.0 - ttp / C.T0C))
    rs0 = jnp.minimum(rs0, 0.99 * p)
    rs0 = C.EP2 * rs0 / (p - rs0)
    rs0 = jnp.maximum(rs0, qmin)
    # WRF: diffus(t,p)=8.794e-5*t**1.81/p ; xka(t,den)=1.414e3*viscos(t,den)*den
    diffus_tp = 8.794e-5 * jnp.exp(jnp.log(t) * 1.81) / p
    ghw1 = den * C.XLV0 * diffus_tp * (rs0 - q) - _xka(t, den) * (-supcol)
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

    # ----- enhanced melting (supcol<=0): pseml/nseml, pgeml/ngeml, pheml/nheml --
    warm2 = supcol <= 0.0
    xlf_e = C.XLF0
    pseml = jnp.where(warm2 & (qs > 0.0),
                      jnp.minimum(jnp.maximum(C.CLIQ * supcol * (paacw + psacr) / xlf_e,
                                              -qs / dtcld), 0.0), 0.0)
    sfac_s2 = rslope[:, 1] * C.N0S * n0sfac / safe_qs
    nseml = jnp.where(warm2 & (qs > 0.0) & (qs > C.QCRMIN), -sfac_s2 * pseml, 0.0)
    pgeml = jnp.where(warm2 & (qg > 0.0),
                      jnp.minimum(jnp.maximum(C.CLIQ * supcol * (paacw + pgacr) / xlf_e,
                                              -qg / dtcld), 0.0), 0.0)
    gfac_g2 = rslope[:, 2] * C.N0G / safe_qg
    ngeml = jnp.where(warm2 & (qg > 0.0) & (qg > C.QCRMIN), -gfac_g2 * pgeml, 0.0)
    pheml = jnp.where(warm2 & (qh > 0.0),
                      jnp.minimum(jnp.maximum(C.CLIQ * supcol * (phacw + phacr) / xlf_e,
                                              -qh / dtcld), 0.0), 0.0)
    gfac_h2 = rslope[:, 3] * C.N0H / safe_qh
    nheml = jnp.where(warm2 & (qh > 0.0) & (qh > C.QCRMIN), -gfac_h2 * pheml, 0.0)

    # deposition/sublimation + ice/snow/graupel/hail aggregation
    pidep, psdep, pgdep, phdep, pigen, psaut, pgaut = _wdm7_deposition_block(
        cold, qi, qs, qg, qh, q, diameter, xni, rh2, work1_2, satdt2,
        prevp, dtcld, n0sfac, rslope, rslopeb, rslope2, work2, den, supcol,
        supsat2)

    # phaut: graupel->hail aggregation (WRF: OUTSIDE the supcol>0 guard; fires
    # whenever qg>0 -- module_mp_wdm7.F lines ~1823-1826)
    alpha2 = 1.0e-3 * jnp.exp(0.09 * (-supcol))
    phaut = jnp.where(qg > 0.0,
                      jnp.minimum(jnp.maximum(0.0, alpha2 * (qg - C.QS0)), qg / dtcld), 0.0)

    # psevp/pgevp/phevp (supcol<0, rh1<1)
    sub_cond = (supcol < 0.0) & (rh1 < 1.0)
    coeres_s2 = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psevp_raw = ((rh1 - 1.0) * n0sfac * (C.PRECS1 * rslope2[:, 1]
                 + C.PRECS2 * work2 * coeres_s2) / work1_1)
    psevp_raw = jnp.minimum(jnp.maximum(psevp_raw, -qs / dtcld), 0.0)
    psevp = jnp.where(sub_cond & (qs > 0.0), psevp_raw, 0.0)
    coeres_g2 = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgevp_raw = ((rh1 - 1.0) * (C.PRECG1 * rslope2[:, 2]
                 + C.PRECG2 * work2 * coeres_g2) / work1_1)
    pgevp_raw = jnp.minimum(jnp.maximum(pgevp_raw, -qg / dtcld), 0.0)
    pgevp = jnp.where(sub_cond & (qg > 0.0), pgevp_raw, 0.0)
    coeres_h2 = rslope2[:, 3] * jnp.sqrt(rslope[:, 3] * rslopeb[:, 3])
    phevp_raw = ((rh1 - 1.0) * (C.PRECH1 * rslope2[:, 3]
                 + C.PRECH2 * work2 * coeres_h2) / work1_1)
    phevp_raw = jnp.minimum(jnp.maximum(phevp_raw, -qh / dtcld), 0.0)
    phevp = jnp.where(sub_cond & (qh > 0.0), phevp_raw, 0.0)

    pvapg = jnp.zeros_like(t)  # never set nonzero in WRF wdm72D
    pvaph = jnp.zeros_like(t)
    primh = jnp.zeros_like(t)

    # ===================== conservation feedback + update =====================
    delta2 = jnp.where((qr < 1.0e-4) & (qs < 1.0e-4), 1.0, 0.0)
    delta3 = jnp.where(qr < 1.0e-4, 1.0, 0.0)
    cold_branch = t <= C.T0C

    # ---------- COLD branch (t<=t0c) ----------
    # cloud water (WRF cold-branch source: praut+pracw+paacw+paacw+phacw)
    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + paacw + paacw + phacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut * f; pracw_c = pracw * f; paacw_c = paacw * f; phacw_c = phacw * f
    # cloud ice
    val = jnp.maximum(qmin, qi)
    src = (psaut - pigen - pidep + praci + psaci + pgaci + phaci) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psaut_c = psaut * f; pigen_c = pigen * f; pidep_c = pidep * f
    praci_c = praci * f; psaci_c = psaci * f; pgaci_c = pgaci * f; phaci_c = phaci * f
    # rain (mass)
    val = jnp.maximum(qmin, qr)
    src = (-praut_c - prevp - pracw_c + piacr + psacr + pgacr + phacr) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut_c * f; prevp_c = prevp * f; pracw_c = pracw_c * f
    piacr_c = piacr * f; psacr_c = psacr * f; pgacr_c = pgacr * f; phacr_c = phacr * f
    # snow
    val = jnp.maximum(qmin, qs)
    src = -(psdep + psaut_c - pgaut + paacw_c + piacr_c * delta3
            + praci_c * delta3 + pvapg + pvaph
            - pracs * (1.0 - delta2) + psacr_c * delta2
            + psaci_c - pgacs - phacs) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psdep_c = psdep * f; psaut_c = psaut_c * f; pgaut_c = pgaut * f; paacw_c = paacw_c * f
    piacr_c = piacr_c * f; praci_c = praci_c * f; psaci_c = psaci_c * f
    pracs_c = pracs * f; psacr_c = psacr_c * f; pgacs_c = pgacs * f; phacs_c = phacs * f
    pvapg_c = pvapg * f; pvaph_c = pvaph * f
    # graupel
    val = jnp.maximum(qmin, qg)
    src = -(pgdep + pgaut_c + piacr_c * (1.0 - delta3) + praci_c * (1.0 - delta3)
            + psacr_c * (1.0 - delta2) + pracs_c * (1.0 - delta2)
            + pgaci_c + paacw_c + pgacr_c * delta2 + pgacs_c
            - pracg * (1.0 - delta2) - phacg - phaut
            - pvapg_c + primh) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgdep_c = pgdep * f; pgaut_c = pgaut_c * f; phaut_c = phaut * f; piacr_c = piacr_c * f
    praci_c = praci_c * f; psacr_c = psacr_c * f; pracs_c = pracs_c * f; paacw_c = paacw_c * f
    pgaci_c = pgaci_c * f; pgacr_c = pgacr_c * f; pgacs_c = pgacs_c * f; pracg_c = pracg * f
    phacg_c = phacg * f; pvapg_c = pvapg_c * f; primh_c = primh * f
    # hail
    val = jnp.maximum(qmin, qh)
    src = -(phdep + phaut_c + pgacr_c * (1.0 - delta2) + pracg_c * (1.0 - delta2)
            + phacw_c + phacr_c + phaci_c + phacs_c + phacg_c
            - pvaph_c - primh_c) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    phdep_c = phdep * f; phaut_c = phaut_c * f; pracg_c = pracg_c * f; pgacr_c = pgacr_c * f
    phacw_c = phacw_c * f; phaci_c = phaci_c * f; phacr_c = phacr_c * f; phacs_c = phacs_c * f
    phacg_c = phacg_c * f; pvaph_c = pvaph_c * f; primh_c = primh_c * f
    # cloud number
    val = jnp.maximum(C.NCMIN, nc)
    src = (nraut + nccol + nracw + naacw + naacw + nhacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_c = nraut * f; nccol_c = nccol * f; nracw_c = nracw * f; naacw_c = naacw * f
    nhacw_c = nhacw * f
    # rain number
    val = jnp.maximum(C.NRMIN, nr)
    src = (-nraut_c + nrcol + niacr + nsacr + ngacr + nhacr) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_c = nraut_c * f; nrcol_c = nrcol * f; niacr_c = niacr * f
    nsacr_c = nsacr * f; ngacr_c = ngacr * f; nhacr_c = nhacr * f

    w2_cold = -(prevp_c + psdep_c + pgdep_c + phdep_c + pigen_c + pidep_c)
    q_cold = q + w2_cold * dtcld
    qc_cold = jnp.maximum(qc - (praut_c + pracw_c + paacw_c + paacw_c + phacw_c) * dtcld, 0.0)
    qr_cold = jnp.maximum(qr + (praut_c + pracw_c + prevp_c - piacr_c - pgacr_c
                               - psacr_c - phacr_c) * dtcld, 0.0)
    qi_cold = jnp.maximum(qi - (psaut_c + praci_c + psaci_c + pgaci_c + phaci_c
                               - pigen_c - pidep_c) * dtcld, 0.0)
    qs_cold = jnp.maximum(qs + (psdep_c + psaut_c + paacw_c - pgaut_c
                               + piacr_c * delta3 + praci_c * delta3 + psaci_c
                               - pgacs_c - pracs_c * (1.0 - delta2)
                               + psacr_c * delta2 + pvapg_c + pvaph_c
                               - phacs_c) * dtcld, 0.0)
    qg_cold = jnp.maximum(qg + (pgdep_c + pgaut_c + piacr_c * (1.0 - delta3)
                               + praci_c * (1.0 - delta3) + psacr_c * (1.0 - delta2)
                               + pracs_c * (1.0 - delta2) + pgaci_c + paacw_c
                               + pgacr_c * delta2 + pgacs_c + primh_c
                               - pracg_c * (1.0 - delta2) - phacg_c - phaut_c
                               - pvapg_c) * dtcld, 0.0)
    qh_cold = jnp.maximum(qh + (phdep_c + phaut_c + pgacr_c * (1.0 - delta2)
                               + pracg_c * (1.0 - delta2) + phacw_c + phacr_c
                               + phaci_c + phacs_c + phacg_c
                               - pvaph_c - primh_c) * dtcld, 0.0)
    nc_cold = jnp.maximum(nc + (-nraut_c - nccol_c - nracw_c - naacw_c - naacw_c
                               - nhacw_c) * dtcld, 0.0)
    nr_cold = jnp.maximum(nr + (nraut_c - nrcol_c - niacr_c - nsacr_c - ngacr_c
                               - nhacr_c) * dtcld, 0.0)
    xlf_cold = C.XLS - xl
    xlwork2_cold = (-C.XLS * (psdep_c + pgdep_c + phdep_c + pidep_c + pigen_c)
                    - xl * prevp_c
                    - xlf_cold * (piacr_c + paacw_c + paacw_c + phacw_c
                                  + pgacr_c + psacr_c + phacr_c))
    t_cold = t - xlwork2_cold / cpm * dtcld

    # ---------- WARM branch (t>t0c) ----------
    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + paacw + paacw - phacw) * dtcld   # WRF warm-branch sign
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut * f; pracw_w = pracw * f; paacw_w = paacw * f; phacw_w = phacw * f
    # rain
    val = jnp.maximum(qmin, qr)
    src = (-prevp - praut_w + pseml + pgeml + pheml - pracw_w - paacw_w - paacw_w
           - phacw_w) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut_w * f; prevp_w = prevp * f; pracw_w = pracw_w * f; paacw_w = paacw_w * f
    phacw_w = phacw_w * f; pseml_w = pseml * f; pgeml_w = pgeml * f; pheml_w = pheml * f
    # snow
    val = jnp.maximum(C.QCRMIN, qs)
    src = (pgacs + phacs - pseml_w - psevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs * f; phacs_w = phacs * f; psevp_w = psevp * f; pseml_w = pseml_w * f
    # graupel
    val = jnp.maximum(C.QCRMIN, qg)
    src = -(pgacs_w + pgevp + pgeml_w - phacg) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs_w * f; pgevp_w = pgevp * f; pgeml_w = pgeml_w * f; phacg_w = phacg * f
    # hail
    val = jnp.maximum(C.QCRMIN, qh)
    src = -(phacs_w + phacg_w + phevp + pheml_w) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    phacs_w = phacs_w * f; phacg_w = phacg_w * f; phevp_w = phevp * f; pheml_w = pheml_w * f
    # cloud number
    val = jnp.maximum(C.NCMIN, nc)
    src = (nraut + nccol + nracw + naacw + naacw + nhacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_w = nraut * f; nccol_w = nccol * f; nracw_w = nracw * f; naacw_w = naacw * f
    nhacw_w = nhacw * f
    # rain number
    val = jnp.maximum(C.NRMIN, nr)
    src = (-nraut_w + nrcol - nseml - ngeml - nheml) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_w = nraut_w * f; nrcol_w = nrcol * f; nseml_w = nseml * f
    ngeml_w = ngeml * f; nheml_w = nheml * f

    w2_warm = -(prevp_w + psevp_w + pgevp_w + phevp_w)
    q_warm = q + w2_warm * dtcld
    qc_warm = jnp.maximum(qc - (praut_w + pracw_w + paacw_w + paacw_w + phacw_w) * dtcld, 0.0)
    qr_warm = jnp.maximum(qr + (praut_w + pracw_w + prevp_w + paacw_w + paacw_w
                               + phacw_w - pseml_w - pgeml_w - pheml_w) * dtcld, 0.0)
    qs_warm = jnp.maximum(qs + (psevp_w - pgacs_w - phacs_w + pseml_w) * dtcld, 0.0)
    qg_warm = jnp.maximum(qg + (pgacs_w + pgevp_w + pgeml_w - phacg_w) * dtcld, 0.0)
    qh_warm = jnp.maximum(qh + (phacs_w + phacg_w + phevp_w + pheml_w) * dtcld, 0.0)
    nc_warm = jnp.maximum(nc + (-nraut_w - nccol_w - nracw_w - naacw_w - naacw_w
                               - nhacw_w) * dtcld, 0.0)
    nr_warm = jnp.maximum(nr + (nraut_w - nrcol_w + nseml_w + ngeml_w + nheml_w) * dtcld, 0.0)
    xlf_warm = C.XLS - xl
    xlwork2_warm = (-xl * (prevp_w + psevp_w + pgevp_w + phevp_w)
                    - xlf_warm * (pseml_w + pgeml_w + pheml_w))
    t_warm = t - xlwork2_warm / cpm * dtcld

    # select branch per cell
    q = jnp.where(cold_branch, q_cold, q_warm)
    qc = jnp.where(cold_branch, qc_cold, qc_warm)
    qr = jnp.where(cold_branch, qr_cold, qr_warm)
    qi = jnp.where(cold_branch, qi_cold, qi)   # warm branch leaves qi unchanged
    qs = jnp.where(cold_branch, qs_cold, qs_warm)
    qg = jnp.where(cold_branch, qg_cold, qg_warm)
    qh = jnp.where(cold_branch, qh_cold, qh_warm)
    nc = jnp.where(cold_branch, nc_cold, nc_warm)
    nr = jnp.where(cold_branch, nr_cold, nr_warm)
    t = jnp.where(cold_branch, t_cold, t_warm)

    # ---- recompute qsat; small-drop rain->cloud (avedia<=di82) ----
    # WRF recomputes qsat1 + rh1 (used by pcond). The slope_wdm7 call that feeds
    # this avedia (module_mp_wdm7.F line ~2222) reuses qrs_tmp/ncr_tmp from the
    # PRE-microphysics state (last set ~line 1160), NOT the post-update qr/nr --
    # so we use the snapshot rslope_preupd_r captured above, not a recompute.
    qsat1b = _qsat1_only(t, p)
    rh1b = jnp.maximum(q / qsat1b, qmin)
    avedia_r2 = rslope_preupd_r * (24.0 ** (1.0 / 3.0))
    small = avedia_r2 <= C.DI82
    nc = jnp.where(small, nc + nr, nc)
    nr = jnp.where(small, 0.0, nr)
    qc = jnp.where(small, qc + qr, qc)
    qr = jnp.where(small, 0.0, qr)

    # ---- CCN activation: pcact (QV->QC) + ncact (NCCN->NC) ----
    activate = rh1b > 1.0
    ncact = jnp.maximum(0.0, ((nn + nc) * jnp.minimum(1.0, (rh1b / C.SATMAX) ** C.ACTK)
                              - nc)) / dtcld
    ncact = jnp.minimum(ncact, jnp.maximum(nn, 0.0) / dtcld)
    pcact = jnp.minimum(4.0 * C.PI * C.DENR * (C.ACTR * 1.0e-6) ** 3 * ncact
                        / (3.0 * den), jnp.maximum(q, 0.0) / dtcld)
    q = jnp.where(activate, jnp.maximum(q - pcact * dtcld, 0.0), q)
    qc = jnp.where(activate, jnp.maximum(qc + pcact * dtcld, 0.0), qc)
    nn = jnp.where(activate, jnp.maximum(nn - ncact * dtcld, 0.0), nn)
    nc = jnp.where(activate, jnp.maximum(nc + ncact * dtcld, 0.0), nc)
    t = jnp.where(activate, t + pcact * xl / cpm * dtcld, t)

    # ---- pcond condensation + ncevp(NC->NCCN) ----
    qsat1c = _qsat1_only(t, p)
    w1 = _conden(t, q, qsat1c, xl, cpm)
    pcond = jnp.minimum(jnp.maximum(w1 / dtcld, 0.0), jnp.maximum(q, 0.0) / dtcld)
    pcond = jnp.where((qc > 0.0) & (w1 < 0.0), jnp.maximum(w1, -qc) / dtcld, pcond)
    ncevp_full = pcond == (-qc / dtcld)
    nn = jnp.where(ncevp_full, nn + nc, nn)
    nc = jnp.where(ncevp_full, 0.0, nc)
    q = q - pcond * dtcld
    qc = jnp.maximum(qc + pcond * dtcld, 0.0)
    t = t + pcond * xl / cpm * dtcld

    # ---- small-value padding + lamdr/lamdc clamping (adjust Nr/Nc) ----
    qc = jnp.where(qc <= qmin, 0.0, qc)
    qi = jnp.where(qi <= qmin, 0.0, qi)
    rain_ok = (qr >= C.QCRMIN) & (nr >= C.NRMIN)
    safe_qr2 = jnp.where(qr > 0.0, qr, 1.0)
    lamdr = jnp.exp(jnp.log((C.PIDNR * nr) / (den * safe_qr2)) * (1.0 / 3.0))
    lamdr_lo = lamdr <= C.LAMDARMIN
    lamdr_hi = lamdr >= C.LAMDARMAX
    nr_lo = den * qr * C.LAMDARMIN ** 3 / C.PIDNR
    nr_hi = den * qr * C.LAMDARMAX ** 3 / C.PIDNR
    nr = jnp.where(rain_ok & lamdr_lo, nr_lo,
                   jnp.where(rain_ok & lamdr_hi, nr_hi, nr))
    cloud_ok2 = (qc >= qmin) & (nc >= C.NCMIN)
    safe_qc2 = jnp.where(qc > 0.0, qc, 1.0)
    lamdc = jnp.exp(jnp.log((C.PIDNC * nc) / (den * safe_qc2)) * (1.0 / 3.0))
    lamdc_lo = lamdc <= C.LAMDACMIN
    lamdc_hi = lamdc >= C.LAMDACMAX
    nc_lo = den * qc * C.LAMDACMIN ** 3 / C.PIDNC
    nc_hi = den * qc * C.LAMDACMAX ** 3 / C.PIDNC
    nc = jnp.where(cloud_ok2 & lamdc_lo, nc_lo,
                   jnp.where(cloud_ok2 & lamdc_hi, nc_hi, nc))

    return (t, q, qc, qr, qi, qs, qg, qh, nn, nc, nr, cpm, xl,
            rainncv, snowncv, graupelncv, hailncv,
            tstepsnow, tstepgraup, tstephail, sr)


# vmap the single-column scheme across the column (i) dimension.
_wdm7_columns = jax.jit(
    jax.vmap(_wdm7_column,
             in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0)),
    static_argnums=(14,))


# --------------------------------------------------------------------------
# Public adapter
# --------------------------------------------------------------------------
def wdm7_run(t, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr, den, p, delz, delt, slmsk):
    """Run WDM7 on a batch of columns (shape (ncol, nlev)).

    Inputs are TEMPERATURE-based (t in K), matching the WRF wdm72D internal
    state. ``qh`` is the precipitating hail mixing ratio (single-moment; there
    is NO hail number). ``nn,nc,nr`` are the CCN / cloud-droplet / rain number
    concentrations (# kg^-1). ``slmsk`` is the per-column land/sea mask
    (1=land, 2=water); pass a (ncol,) array. Returns a dict with updated t, qv,
    qc, qr, qi, qs, qg, qh, nn, nc, nr, surface precip increments (mm) for
    rain/snow/graupel/hail, sr, and effective radii.
    """
    return _wdm7_columns(t, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr,
                         den, p, delz, float(delt), slmsk)


def wdm7_physics_tendency(theta, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr,
                          pii, den, p, delz, delt, slmsk):
    """WDM7 adapter returning a frozen PhysicsTendency (in-place replacements).

    ``theta`` and the moist/number species are (ncol, nlev). ``pii`` is the
    Exner function used to convert theta<->t (as the WRF ``wdm7`` wrapper does).
    Surface precip increments are per-call mm (rain/snow/graupel/hail).

    NOTE (Nc/Nn integration): like WDM6, WDM7 carries the double-moment State
    leaves ``Nc`` (qnc) and ``Nn`` (qnn); ``Nr`` (qnr) already exists. This
    adapter returns ``state_replacements`` for the moist mass species (incl
    qh), theta, and the number leaves nc/nr; ``nn`` is the CCN leaf the
    manager-owned State patch materializes, so until then it is carried in
    ``diagnostics``. Hail is single-moment (qh only, no Nh).
    """
    t = theta * pii
    out = wdm7_run(t, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr, den, p, delz, delt, slmsk)
    theta_new = out["t"] / pii

    state_repl = {
        "theta": theta_new,
        "qv": out["qv"], "qc": out["qc"], "qr": out["qr"],
        "qi": out["qi"], "qs": out["qs"], "qg": out["qg"], "qh": out["qh"],
        "Nr": out["nr"],
        "Nc": out["nc"],
    }

    tend = PhysicsTendency(
        state_replacements=state_repl,
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["snowncv"],
            "graupel_acc": out["graupelncv"],
            "hail_acc": out["hailncv"],
        },
        diagnostics={
            "Nn": out["nn"],   # CCN: manager-owned State materialization (S0)
            "re_cloud": out["re_cloud"],
            "re_ice": out["re_ice"],
            "re_snow": out["re_snow"],
            "sr": out["sr"],
        },
    )
    return tend


__all__ = ["wdm7_run", "wdm7_physics_tendency"]
