"""JAX WDM6 double-moment 6-class microphysics (WRF mp_physics=16).

Faithful port of WRF ``phys/module_mp_wdm6.F`` (subroutines ``wdm62D`` +
``slope_wdm6`` + ``slope_rain``/``slope_snow``/``slope_graup`` +
``nislfv_rain_plmr`` + ``nislfv_rain_plm6`` + ``effectRad_wdm6``), preserving
WRF's exact process order.

WDM6 = WSM6 single-moment ICE/SNOW/GRAUPEL microphysics + a DOUBLE-MOMENT warm
rain: predicted cloud-droplet number ``Nc``, rain number ``Nr``, and CCN number
``Nn``. The number-concentration array in WRF packs ``ncr(:,:,1)=Nn`` (CCN),
``ncr(:,:,2)=Nc`` (cloud), ``ncr(:,:,3)=Nr`` (rain). This port carries them as
named columns ``Nn, Nc, Nr``.

WRF process order (wdm62D), preserved:

  padding (qc/qi/qr/qs/qg>=0; Nn in [1e8,2e10]; Nc,Nr>=0)
  for loop in 1..loops (minor time steps):
    1. denfac, qsat(liquid,ice), rh, init process arrays, rslopec, xni
    2. slope_wdm6; semi-Lagrangian fallout of rain+Nr (plmr), then snow+graupel
       (plm6); slope_wdm6 recompute on post-sedimentation qrs
    3. snow/graupel melting for T>T0 (psmlt/pgmlt + nsmlt/ngmlt -> Nr)
    4. ice fallout (Vice + plmr); surface precip (rain/snow/graupel/sr)
    5. instantaneous melt/freeze (pimlt+nimlt, pihmf+nihmf, pihtf+nihtf,
       pgfrz+ngfrz); clamp Nc,Nr>=0
    6. slope_wdm6; avedia(rain,cloud); rslopec; work1(diffac)/work2(venfac)
    7. WARM RAIN (double-moment): praut+nraut, pracw+nracw, nccol, nrcol,
       prevp + Nrevp(NR->NCCN)
    8. COLD RAIN: praci/piacr/niacr, psaci, pgaci, psacw/nsacw, pgacw/ngacw,
       paacw/naacw, pracs, psacr/nsacr, pgacr/ngacr, pgacs(=0),
       pseml/nseml, pgeml/ngeml, deposition(pidep/psdep/pgdep), pigen,
       psaut, pgaut, psevp, pgevp
    9. mass+number conservation feedback + state update (T<=T0 / T>T0 branches)
    10. recompute qsat; small-drop rain->cloud conversion (avedia<=di82);
        CCN activation pcact+ncact; pcond condensation + ncevp(NC->NCCN);
        small-value padding + lamdr/lamdc clamping (adjusts Nr/Nc)

The scheme works on TEMPERATURE ``t`` internally; the WRF ``wdm6`` wrapper
converts th<->t via the Exner function ``pii``. The adapter
``wdm6_physics_tendency`` mirrors that and returns a frozen ``PhysicsTendency``
with ``state_replacements`` for theta + the moist species + the number leaves
(Nc, Nr, Nn), ``accumulator_increments`` for surface precip, and
``diagnostics`` for the effective radii.

Validation: per-column WRF savepoint parity against the real Fortran scheme
(proofs/v060_wdm6/oracle). The port defaults to fp64; the canonical classic
WRF scheme runs in fp32 (bare ``real``), so prognostic-state parity is to a
predeclared physical tolerance, never bitwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import wdm6_constants as C


# --------------------------------------------------------------------------
# Inline thermodynamic helper functions (module_mp_wdm6.F statement functions)
# --------------------------------------------------------------------------
def _cpmcal(q):
    qq = jnp.maximum(q, C.QMIN)
    return C.CPD * (1.0 - qq) + qq * C.CPV


def _xlcal(t):
    return C.XLV0 - C.XLV1 * (t - C.T0C)


def _diffus(x, y):
    return 8.794e-5 * jnp.exp(jnp.log(x) * 1.81) / y


def _viscos(x, y):
    return 1.496e-6 * (x * jnp.sqrt(x)) / (x + 120.0) / y


def _xka(x, y):
    return 1.414e3 * _viscos(x, y) * y


def _diffac(a, b, c, d, e):
    return d * a * a / (_xka(c, d) * C.RV * c * c) + 1.0 / (e * _diffus(c, b))


def _venfac(a, b, c):
    return (jnp.exp(jnp.log(_viscos(b, c) / _diffus(b, a)) * (1.0 / 3.0))
            / jnp.sqrt(_viscos(b, c)) * jnp.sqrt(jnp.sqrt(C.DEN0 / c)))


def _conden(a, b, c, d, e):
    return (jnp.maximum(b, C.QMIN) - c) / (1.0 + d * d / (C.RV * e) * c / (a * a))


def _lamdac(qc, den, nc):
    # cloud-droplet slope: exp(log((pidnc*nc)/(qc*den))/3)  (LH cloud distrib)
    return jnp.exp(jnp.log((C.PIDNC * nc) / (qc * den)) * (1.0 / 3.0))


def _qsat_fpvs(t, p):
    """Inline fpvs expansion: returns (qsat1 over liquid, qsat2 over ice<ttp)."""
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
    qsat1 = C.PSAT * jnp.exp(jnp.log(tr) * xa) * jnp.exp(xb * (1.0 - tr))
    qsat1 = jnp.minimum(qsat1, 0.99 * p)
    qsat1 = C.EP2 * qsat1 / (p - qsat1)
    qsat1 = jnp.maximum(qsat1, C.QMIN)

    qsat2_ice = C.PSAT * jnp.exp(jnp.log(tr) * xai) * jnp.exp(xbi * (1.0 - tr))
    qsat2_liq = C.PSAT * jnp.exp(jnp.log(tr) * xa) * jnp.exp(xb * (1.0 - tr))
    qsat2 = jnp.where(t < ttp, qsat2_ice, qsat2_liq)
    qsat2 = jnp.minimum(qsat2, 0.99 * p)
    qsat2 = C.EP2 * qsat2 / (p - qsat2)
    qsat2 = jnp.maximum(qsat2, C.QMIN)
    return qsat1, qsat2


def _qsat1_only(t, p):
    """Liquid-saturation mixing ratio only (used in the pcond recompute)."""
    cvap = C.CPV
    ttp = C.T0C + 0.01
    dldt = cvap - C.CLIQ
    xa = -dldt / C.RV
    xb = xa + C.XLV0 / (C.RV * ttp)
    tr = ttp / t
    qsat1 = C.PSAT * jnp.exp(jnp.log(tr) * xa) * jnp.exp(xb * (1.0 - tr))
    qsat1 = jnp.minimum(qsat1, 0.99 * p)
    qsat1 = C.EP2 * qsat1 / (p - qsat1)
    qsat1 = jnp.maximum(qsat1, C.QMIN)
    return qsat1


# --------------------------------------------------------------------------
# slope_wdm6: rain(double-moment)/snow/graupel slopes + fall speeds (per col).
# Index 0=rain, 1=snow, 2=graupel. Mirrors slope_wdm6 exactly. The rain slope
# uses the PREDICTED rain number nr: lamdar = ((pidnr*nr)/(qr*den))^(1/3),
# rslope = min(1/lamdar, 1e-3). Returns rslope/rslopeb/rslope2/rslope3 (K,3),
# vt (K,3) and vtn (K) (the rain-NUMBER fall speed).
# --------------------------------------------------------------------------
def _slope_wdm6(qr, qs, qg, nr, den, denfac, t):
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)

    # rain (double-moment): lamdar from predicted number nr
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

    rslope = jnp.stack([rsl_r, rsl_s, rsl_g], axis=-1)
    rslopeb = jnp.stack([rslb_r, rslb_s, rslb_g], axis=-1)
    rslope2 = jnp.stack([rsl2_r, rsl2_s, rsl2_g], axis=-1)
    rslope3 = jnp.stack([rsl3_r, rsl3_s, rsl3_g], axis=-1)

    vt_r = jnp.where(qr <= 0.0, 0.0, C.PVTR * rslb_r * denfac)
    vt_s = jnp.where(qs <= 0.0, 0.0, C.PVTS * rslb_s * denfac)
    vt_g = jnp.where(qg <= 0.0, 0.0, C.PVTG * rslb_g * denfac)
    vt = jnp.stack([vt_r, vt_s, vt_g], axis=-1)
    vtn = jnp.where(nr <= 0.0, 0.0, C.PVTRN * rslb_r * denfac)
    return rslope, rslopeb, rslope2, rslope3, vt, vtn


def _slope_rain_vt(qr, nr, den, denfac):
    """slope_rain: rain mass-fall vt and number-fall vtn from (qr, nr)."""
    rain_ok = (qr > C.QCRMIN) & (nr > C.NRMIN)
    lamdar = jnp.exp(jnp.log((C.PIDNR * nr) / (qr * den)) * (1.0 / 3.0))
    rsl = jnp.where(rain_ok, jnp.minimum(1.0 / lamdar, 1.0e-3), C.RSLOPERMAX)
    rslb = jnp.where(rain_ok, rsl ** C.BVTR, C.RSLOPERBMAX)
    vt = jnp.where(qr <= 0.0, 0.0, C.PVTR * rslb * denfac)
    vtn = jnp.where(nr <= 0.0, 0.0, C.PVTRN * rslb * denfac)
    return vt, vtn


def _slope_snow_vt(q, den, denfac, t):
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    lamda = jnp.sqrt(jnp.sqrt(C.PIDN0S * n0sfac / (q * den)))
    rsl = jnp.where(q <= C.QCRMIN, C.RSLOPESMAX, 1.0 / lamda)
    rslb = jnp.where(q <= C.QCRMIN, C.RSLOPESBMAX, rsl ** C.BVTS)
    return jnp.where(q <= 0.0, 0.0, C.PVTS * rslb * denfac)


def _slope_graup_vt(q, den, denfac):
    lamda = jnp.sqrt(jnp.sqrt(C.PIDN0G / (q * den)))
    rsl = jnp.where(q <= C.QCRMIN, C.RSLOPEGMAX, 1.0 / lamda)
    rslb = jnp.where(q <= C.QCRMIN, C.RSLOPEGBMAX, rsl ** C.BVTG)
    return jnp.where(q <= 0.0, 0.0, C.PVTG * rslb * denfac)


# --------------------------------------------------------------------------
# Semi-Lagrangian PLM sedimentation primitives (shared by plmr/plm6).
# Mirror module_mp_wdm6.F nislfv_rain_plmr / nislfv_rain_plm6; column index
# 0=bottom .. km-1=top.
# --------------------------------------------------------------------------
def _wi_interp(ww, dz, km):
    fa1 = 9.0 / 16.0
    fa2 = 1.0 / 16.0
    wi = jnp.zeros(km + 1, dtype=ww.dtype)
    wi = wi.at[0].set(ww[0])
    wi = wi.at[1].set(0.5 * (ww[1] + ww[0]))
    if km >= 4:
        k = jnp.arange(2, km - 1)
        interior = fa1 * (ww[k] + ww[k - 1]) - fa2 * (ww[k + 1] + ww[k - 2])
        wi = wi.at[2:km - 1].set(interior)
    wi = wi.at[km - 1].set(0.5 * (ww[km - 1] + ww[km - 2]))
    wi = wi.at[km].set(ww[km - 1])
    # terminate at top of raingroup: where ww(k)==0 -> wi(k)=ww(k-1) (k=2..km)
    wi = wi.at[1:km].set(jnp.where(ww[1:km] == 0.0, ww[0:km - 1], wi[1:km]))
    return wi


def _diffusivity_limit(wi, dz, dt, km):
    con1 = 0.05

    def body(k_rev, wi):
        k = km - 1 - k_rev
        decfl = (wi[k + 1] - wi[k]) * dt / dz[k]
        new = wi[k + 1] - con1 * dz[k] / dt
        wi = wi.at[k].set(jnp.where(decfl > con1, new, wi[k]))
        return wi

    return jax.lax.fori_loop(0, km, body, wi)


def _plm_qmi_qpi(qa, dza, km):
    qmi = jnp.zeros(km + 1, dtype=qa.dtype)
    qpi = jnp.zeros(km + 1, dtype=qa.dtype)
    k = jnp.arange(1, km)
    dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k])
    dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k])
    qpi_lin = qa[k] + 0.5 * (dip + dim) * dza[k]
    qmi_lin = 2.0 * qa[k] - qpi_lin
    extremum = (dip * dim) <= 0.0
    bad = (qpi_lin < 0.0) | (qmi_lin < 0.0)
    qpi_k = jnp.where(extremum | bad, qa[k], qpi_lin)
    qmi_k = jnp.where(extremum | bad, qa[k], qmi_lin)
    qmi = qmi.at[1:km].set(qmi_k)
    qpi = qpi.at[1:km].set(qpi_k)
    qmi = qmi.at[0].set(qa[0])
    qpi = qpi.at[0].set(qa[0])
    qmi = qmi.at[km].set(qa[km])
    qpi = qpi.at[km].set(qa[km])
    return qmi, qpi


def _plm_remap(qa, dza, qmi, qpi, zi, za, km):
    def body(k, carry):
        qn, kb, kt = carry
        kb = jnp.maximum(kb - 1, 0)
        kt = jnp.maximum(kt - 1, 0)
        done = zi[k] >= za[km]

        def find_kb(start):
            def cond(state):
                kk, found = state
                return jnp.logical_and(kk < km, jnp.logical_not(found))

            def step(state):
                kk, found = state
                hit = zi[k] <= za[kk + 1]
                return jnp.where(hit, kk, kk + 1), jnp.logical_or(found, hit)

            kk0, _ = jax.lax.while_loop(cond, step, (start, False))
            return kk0

        kb_new = find_kb(kb)
        kt_entry = kt

        def find_kt(start):
            def cond(state):
                kk, found = state
                return jnp.logical_and(kk < km, jnp.logical_not(found))

            def step(state):
                kk, found = state
                hit = zi[k + 1] <= za[kk]
                return jnp.where(hit, kk, kk + 1), jnp.logical_or(found, hit)

            kk0, found = jax.lax.while_loop(cond, step, (start, False))
            return jnp.where(found, kk0, kt_entry) - 1

        kt_new = find_kt(kt)
        kb_c = kb_new
        kt_c = kt_new

        # kt == kb : piecewise method
        tl = (zi[k] - za[kb_c]) / dza[kb_c]
        th_ = (zi[k + 1] - za[kb_c]) / dza[kb_c]
        tl2 = tl * tl
        th2 = th_ * th_
        qqd = 0.5 * (qpi[kb_c] - qmi[kb_c])
        qqh = qqd * th2 + qmi[kb_c] * th_
        qql = qqd * tl2 + qmi[kb_c] * tl
        qn_eq = (qqh - qql) / (th_ - tl)

        # kt > kb : integrate kb partial + full middle cells + kt partial
        tl_b = (zi[k] - za[kb_c]) / dza[kb_c]
        tl2_b = tl_b * tl_b
        qqd_b = 0.5 * (qpi[kb_c] - qmi[kb_c])
        qql_b = qqd_b * tl2_b + qmi[kb_c] * tl_b
        dql = qa[kb_c] - qql_b
        zsum0 = (1.0 - tl_b) * dza[kb_c]
        qsum0 = dql * dza[kb_c]

        midx = jnp.arange(km)
        mid_mask = (midx >= kb_c + 1) & (midx <= kt_c - 1)
        zsum_mid = jnp.sum(jnp.where(mid_mask, dza[0:km], 0.0))
        qsum_mid = jnp.sum(jnp.where(mid_mask, qa[0:km] * dza[0:km], 0.0))

        th_t = (zi[k + 1] - za[kt_c]) / dza[kt_c]
        th2_t = th_t * th_t
        qqd_t = 0.5 * (qpi[kt_c] - qmi[kt_c])
        dqh = qqd_t * th2_t + qmi[kt_c] * th_t
        zsum = zsum0 + zsum_mid + th_t * dza[kt_c]
        qsum = qsum0 + qsum_mid + dqh * dza[kt_c]
        qn_gt = qsum / zsum

        qn_val = jnp.where(kt_c == kb_c, qn_eq, jnp.where(kt_c > kb_c, qn_gt, 0.0))
        qn_val = jnp.where(done, 0.0, qn_val)
        qn = qn.at[k].set(qn_val)
        return qn, kb_c, kt_c

    qn0 = jnp.zeros(km, dtype=qa.dtype)
    qn, _, _ = jax.lax.fori_loop(0, km, body, (qn0, 0, 0))
    return qn


def _precip_out(qa, dza, za, km):
    both_below = (za[0:km] < 0.0) & (za[1:km + 1] < 0.0)
    straddle = (za[0:km] < 0.0) & (za[1:km + 1] >= 0.0)
    contrib_full = jnp.where(both_below, qa[0:km] * dza[0:km], 0.0)
    contrib_str = jnp.where(straddle, qa[0:km] * (0.0 - za[0:km]), 0.0)
    prefix_below = jnp.cumprod(both_below.astype(qa.dtype))
    started = jnp.concatenate([jnp.ones(1, qa.dtype), prefix_below[:-1]])
    return jnp.sum(contrib_full * prefix_below) + jnp.sum(contrib_str * started)


def _nislfv_rain_plmr(q, nr_in, den, denfac, tk, dz, ww_in, dt, iter_n, rid):
    """nislfv_rain_plmr for one species, one column.

    Mirrors module_mp_wdm6.F nislfv_rain_plmr. ``rid``: 1 => the transported
    field is a NUMBER concentration (q is the number; nr_in is the companion
    mixing-ratio used to re-evaluate the slope/velocity). For the WDM6 calls in
    wdm62D, plmr is used with rid=0 for ice fallout (no number); the rain+Nr
    fallout is done in-line in wdm62D's own mstep loop (NOT via plmr). We keep
    the rid plumbing for completeness but only the rid=0 path is exercised in
    wdm62D's process order.
    """
    km = q.shape[0]
    nr = jnp.where(rid == 1, nr_in / den, nr_in)
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
        qr_cells = jnp.where(rid == 1, qa_cells, qa_cells / den)
        return wi, za, dza, qa_cells, qr_cells

    def iter_body(n, state):
        ww, was = state
        wi, za, dza, qa_cells, qr_cells = iterate(ww)
        # rid=1: slope_rain(nr, qr=qa) -> use vtn; rid=0: slope_rain(qr, nr)->vt
        vt_m, vt_n = _slope_rain_vt(jnp.where(rid == 1, nr, qr_cells),
                                    jnp.where(rid == 1, qr_cells, nr),
                                    den, denfac)
        wa = jnp.where(rid == 1, vt_n, vt_m)
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


def _nislfv_rain_plm6(qs, qg, den, denfac, tk, dz, ww_in, dt, iter_n):
    """nislfv_rain_plm6 for snow+graupel, one column. Returns (qs,qg,ps,pg)."""
    km = qs.shape[0]
    allold = jnp.sum(qs) + jnp.sum(qg)

    zi = jnp.concatenate([jnp.zeros(1, qs.dtype), jnp.cumsum(dz)])
    wd = ww_in

    def iterate(ww):
        wi = _wi_interp(ww, dz, km)
        wi = _diffusivity_limit(wi, dz, dt, km)
        za = zi - wi * dt
        dza = jnp.concatenate([za[1:km + 1] - za[0:km],
                               (zi[km] - za[km]).reshape(1)])
        qa_s = qs * dz / dza[0:km]
        qa_g = qg * dz / dza[0:km]
        qr_s = qa_s / den
        qr_g = qa_g / den
        return wi, za, dza, qa_s, qa_g, qr_s, qr_g

    def iter_body(n, state):
        ww, was = state
        wi, za, dza, qa_s, qa_g, qr_s, qr_g = iterate(ww)
        wa_s = _slope_snow_vt(qr_s, den, denfac, tk)
        wa_g = _slope_graup_vt(qr_g, den, denfac)
        tmp = jnp.maximum(qr_s + qr_g, 1.0e-15)
        wa = jnp.where(tmp > 1.0e-15, (wa_s * qr_s + wa_g * qr_g) / tmp, 0.0)
        wa = jnp.where(n >= 1, 0.5 * (wa + was), wa)
        ww_new = 0.5 * (wd + wa)
        return ww_new, wa

    ww = wd
    was = jnp.zeros(km, dtype=qs.dtype)
    if iter_n >= 1:
        ww, was = jax.lax.fori_loop(0, iter_n, iter_body, (ww, was))

    wi, za, dza, qa_s, qa_g, qr_s, qr_g = iterate(ww)

    def remap_species(qa_cells):
        qa = jnp.concatenate([qa_cells, jnp.zeros(1, qs.dtype)])
        qmi, qpi = _plm_qmi_qpi(qa, dza, km)
        qn = _plm_remap(qa, dza, qmi, qpi, zi, za, km)
        precip = _precip_out(qa, dza, za, km)
        return qn, precip

    qs_new, ps = remap_species(qa_s)
    qg_new, pg = remap_species(qa_g)

    qs_out = jnp.where(allold <= 0.0, qs, qs_new)
    qg_out = jnp.where(allold <= 0.0, qg, qg_new)
    ps = jnp.where(allold <= 0.0, 0.0, ps)
    pg = jnp.where(allold <= 0.0, 0.0, pg)
    return qs_out, qg_out, ps, pg


# --------------------------------------------------------------------------
# Effective radii (effectRad_wdm6) -- double-moment cloud uses predicted Nc.
# --------------------------------------------------------------------------
def _effective_radii(t, qc, nc, qi, qs, rho):
    R1 = 1.0e-12
    R2 = 1.0e-6
    obmr = 1.0 / 3.0
    cdm2 = C._rgmma(5.0 / 3.0)  # rgmma(cdm), cdm=5/3

    re_qc = jnp.full_like(t, C.RE_QC_BG)
    re_qi = jnp.full_like(t, C.RE_QI_BG)
    re_qs = jnp.full_like(t, C.RE_QS_BG)

    rqc = jnp.maximum(R1, qc * rho)
    rnc = jnp.maximum(R2, nc * rho)
    rqi = jnp.maximum(R1, qi * rho)
    temp = rho * jnp.maximum(qi, C.QMIN)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    ni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)
    rni = jnp.maximum(R2, ni * rho)
    rqs = jnp.maximum(R1, qs * rho)

    # cloud: lamc = 2*cdm2*(pidnc*nc/rqc)**obmr ; re = 1/lamc
    lamc = 2.0 * cdm2 * (C.PIDNC * nc / rqc) ** obmr
    re_c = jnp.maximum(2.51e-6, jnp.minimum(1.0 / lamc, 50.0e-6))
    re_qc = jnp.where((rqc > R1) & (rnc > R2), re_c, re_qc)

    # ice
    diai = 11.9 * jnp.sqrt(rqi / ni)
    re_i = jnp.maximum(10.01e-6, jnp.minimum(0.75 * 0.163 * diai, 125.0e-6))
    re_qi = jnp.where((rqi > R1) & (rni > R2), re_i, re_qi)

    # snow
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    lamdas = jnp.sqrt(jnp.sqrt(C.PIDN0S * n0sfac / rqs))
    re_s = jnp.maximum(25.0e-6, jnp.minimum(0.5 * (1.0 / lamdas), 999.0e-6))
    re_qs = jnp.where(rqs > R1, re_s, re_qs)

    # driver bounds (wdm6 wrapper)
    re_qc = jnp.maximum(C.RE_QC_BG, jnp.minimum(re_qc, C.RE_QC_MAX))
    re_qi = jnp.maximum(C.RE_QI_BG, jnp.minimum(re_qi, C.RE_QI_MAX))
    re_qs = jnp.maximum(C.RE_QS_BG, jnp.minimum(re_qs, C.RE_QS_MAX))
    return re_qc, re_qi, re_qs


# --------------------------------------------------------------------------
# Deposition/sublimation + ice/snow aggregation (cold rain), ifsat per cell.
# Faithful to wdm62D's cold-rain ordering (pidep->psdep->pgdep->pigen, then
# psaut, pgaut). Number tendencies for these are not produced here (WDM6's
# ice/snow/graupel are single-moment).
# --------------------------------------------------------------------------
def _deposition_block(cold, qi, qs, qg, q, diameter, xni, rh2, work1_2,
                      satdt2, prevp, dtcld, n0sfac, rslope, rslopeb, rslope2,
                      work2, den, supcol, supsat2):
    pidep_raw = 4.0 * diameter * xni * (rh2 - 1.0) / work1_2
    supice1 = satdt2 - prevp
    pidep_neg = jnp.maximum(jnp.maximum(jnp.maximum(pidep_raw, satdt2 / 2.0), supice1),
                            -qi / dtcld)
    pidep_pos = jnp.minimum(jnp.minimum(pidep_raw, satdt2 / 2.0), supice1)
    pidep_v = jnp.where(pidep_raw < 0.0, pidep_neg, pidep_pos)
    do_pidep = cold & (qi > 0.0)
    pidep = jnp.where(do_pidep, pidep_v, 0.0)
    ifsat1 = do_pidep & (jnp.abs(prevp + pidep) >= jnp.abs(satdt2))

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

    supice4 = satdt2 - prevp - pidep - psdep - pgdep
    xni0 = 1.0e3 * jnp.exp(0.1 * supcol)
    roqi0 = 4.92e-11 * xni0 ** 1.33
    pigen_raw = jnp.maximum(0.0, (roqi0 / den - jnp.maximum(qi, 0.0)) / dtcld)
    pigen_v = jnp.minimum(jnp.minimum(pigen_raw, satdt2), supice4)
    do_pigen = cold & (supsat2 > 0.0) & jnp.logical_not(ifsat3)
    pigen = jnp.where(do_pigen, pigen_v, 0.0)

    qimax = C.ROQIMAX / den
    psaut_v = jnp.maximum(0.0, (qi - qimax) / dtcld)
    psaut = jnp.where(cold & (qi > 0.0), psaut_v, 0.0)

    alpha2 = 1.0e-3 * jnp.exp(0.09 * (-supcol))
    pgaut_v = jnp.minimum(jnp.maximum(0.0, alpha2 * (qs - C.QS0)), qs / dtcld)
    pgaut = jnp.where(cold & (qs > 0.0), pgaut_v, 0.0)

    return pidep, psdep, pgdep, pigen, psaut, pgaut


# --------------------------------------------------------------------------
# Core single-column WDM6 (wdm62D for one column, all minor loops)
# --------------------------------------------------------------------------
def _wdm6_column(t, q, qc, qr, qi, qs, qg, nn, nc, nr, den, p, delz, delt, slmsk):
    qmin = C.QMIN

    # padding from dynamics (wdm62D lines ~577-587)
    qc = jnp.maximum(qc, 0.0)
    qr = jnp.maximum(qr, 0.0)
    qi = jnp.maximum(qi, 0.0)
    qs = jnp.maximum(qs, 0.0)
    qg = jnp.maximum(qg, 0.0)
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
    tstepsnow = jnp.zeros((), t.dtype)
    tstepgraup = jnp.zeros((), t.dtype)
    sr = jnp.zeros((), t.dtype)

    loops = max(int(round(delt / C.DTCLDCR)), 1)
    dtcld = delt / loops
    if delt <= C.DTCLDCR:
        dtcld = delt

    state = (t, q, qc, qr, qi, qs, qg, nn, nc, nr, cpm, xl,
             rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr)

    for _loop in range(loops):
        state = _wdm6_minor_loop(state, den, p, delz, dtcld, qcr)

    (t, q, qc, qr, qi, qs, qg, nn, nc, nr, cpm, xl,
     rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr) = state

    re_qc, re_qi, re_qs = _effective_radii(t, qc, nc, qi, qs, den)

    return {
        "t": t, "qv": q, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
        "nn": nn, "nc": nc, "nr": nr,
        "rainncv": rainncv, "snowncv": snowncv, "graupelncv": graupelncv,
        "sr": sr, "re_cloud": re_qc, "re_ice": re_qi, "re_snow": re_qs,
    }


def _wdm6_minor_loop(state, den, p, delz, dtcld, qcr):
    (t, q, qc, qr, qi, qs, qg, nn, nc, nr, cpm, xl,
     rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr) = state
    km = t.shape[0]
    qmin = C.QMIN

    denfac = jnp.sqrt(C.DEN0 / den)

    qsat1, qsat2 = _qsat_fpvs(t, p)
    rh1 = jnp.maximum(q / qsat1, qmin)
    rh2 = jnp.maximum(q / qsat2, qmin)

    # rslopec (cloud-droplet slope) from qc,nc  (wdm62D lines ~760-768)
    cloud_ok = (qc > qmin) & (nc > C.NCMIN)
    rslopec = jnp.where(cloud_ok, 1.0 / _lamdac(qc, den, nc), C.RSLOPECMAX)
    rslopec2 = jnp.where(cloud_ok, rslopec * rslopec, C.RSLOPEC2MAX)
    rslopec3 = jnp.where(cloud_ok, rslopec2 * rslopec, C.RSLOPEC3MAX)

    # ice crystal number conc (HDC 5c)
    temp = den * jnp.maximum(qi, qmin)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    xni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)

    # ---- fallout: rain + Nr via in-line mstep loop (wdm62D ~795-854) ----
    rslope, rslopeb, rslope2, rslope3, work1_vt, workn = _slope_wdm6(
        qr, qs, qg, nr, den, denfac, t)
    work1_r = work1_vt[:, 0] / delz   # rain mass fall speed / delz
    workn_r = workn / delz            # rain number fall speed / delz

    # mstep(i) = max over column of nint(max(work1,workn)*dtcld+0.5), >=1
    # (WRF wdm62D ~801). Data-dependent; the fallout loop bounds the substeps
    # by a static cap and masks inactive substeps, so mstep stays a tracer.
    numdt = jnp.maximum(jnp.round(jnp.maximum(work1_r, workn_r) * dtcld + 0.5), 1.0)
    mstep = jnp.maximum(jnp.max(numdt), 1.0)

    qr, nr, fall1, falln = _rain_number_fallout(
        qr, nr, qs, qg, den, denfac, t, delz, dtcld, work1_r, workn_r, mstep)

    # snow+graupel semi-Lagrangian (plm6) using mean (vt2s*qs+vt2g*qg)/qsum
    qsum = jnp.maximum(qs + qg, 1.0e-15)
    worka = jnp.where(qsum > 1.0e-15,
                      (work1_vt[:, 1] * qs + work1_vt[:, 2] * qg) / qsum, 0.0)
    denqrs2 = den * qs
    denqrs3 = den * qg
    denqrs2_new, denqrs3_new, delqrs2, delqrs3 = _nislfv_rain_plm6(
        denqrs2, denqrs3, den, denfac, t, delz, worka, dtcld, 1)
    qs = jnp.maximum(denqrs2_new / den, 0.0)
    qg = jnp.maximum(denqrs3_new / den, 0.0)
    fall2 = denqrs2_new * worka / delz
    fall3 = denqrs3_new * worka / delz
    fall2 = fall2.at[0].set(delqrs2 / delz[0] / dtcld)
    fall3 = fall3.at[0].set(delqrs3 / delz[0] / dtcld)

    # slope recompute on post-sedimentation qrs (wdm62D ~893)
    rslope, rslopeb, rslope2, rslope3, work1_vt, workn = _slope_wdm6(
        qr, qs, qg, nr, den, denfac, t)

    # ---- snow/graupel melting for T>T0 (psmlt/pgmlt + nsmlt/ngmlt) ----
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
    # nsmlt: ->NR (only when qs>qcrmin)
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

    # ---- ice fallout (Vice via plmr, rid=0, iter=0) ----
    xmi = den * qi / xni
    diameter = jnp.maximum(jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX), 1.0e-25)
    work1c = jnp.where(qi <= 0.0, 0.0, 1.49e4 * jnp.exp(jnp.log(diameter) * 1.31))
    denqci = den * qi
    denqci_new, delqi = _nislfv_rain_plmr(
        denqci, denqci, den, denfac, t, delz, work1c, dtcld, 0, 0)
    qi = jnp.maximum(denqci_new / den, 0.0)
    fallc0 = delqi / delz[0] / dtcld

    # ---- surface precip accumulation ----
    fallsum = fall1[0] + fall2[0] + fall3[0] + fallc0
    fallsum_qsi = fall2[0] + fallc0
    fallsum_qg = fall3[0]
    add = fallsum * delz[0] / C.DENR * dtcld * 1000.0
    rainncv = jnp.where(fallsum > 0.0, add + rainncv, rainncv)
    add_si = fallsum_qsi * delz[0] / C.DENR * dtcld * 1000.0
    tstepsnow = jnp.where(fallsum_qsi > 0.0, add_si + tstepsnow, tstepsnow)
    snowncv = jnp.where(fallsum_qsi > 0.0, add_si + snowncv, snowncv)
    add_g = fallsum_qg * delz[0] / C.DENR * dtcld * 1000.0
    tstepgraup = jnp.where(fallsum_qg > 0.0, add_g + tstepgraup, tstepgraup)
    graupelncv = jnp.where(fallsum_qg > 0.0, add_g + graupelncv, graupelncv)
    sr = jnp.where(fallsum > 0.0, (snowncv + graupelncv) / (rainncv + 1.0e-12), sr)

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

    # clamp Nc,Nr >= 0 (wdm62D ~1101)
    nc = jnp.maximum(nc, 0.0)
    nr = jnp.maximum(nr, 0.0)

    # ---- slope update + avedia + rslopec + work1(diffac)/work2(venfac) ----
    rslope, rslopeb, rslope2, rslope3, _vt, _vtn = _slope_wdm6(
        qr, qs, qg, nr, den, denfac, t)
    avedia_r = rslope[:, 0] * (24.0 ** (1.0 / 3.0))   # mean-volume rain diam
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

    # prevp (HL A41) + Nrevp(NR->NCCN) handled in the update branch
    coeres_r = rslope[:, 0] * jnp.sqrt(rslope[:, 0] * rslopeb[:, 0])
    prevp_raw = ((rh1 - 1.0) * nr * (C.PRECR1 * rslope[:, 0]
                 + C.PRECR2 * work2 * coeres_r) / work1_1)
    prevp_neg = jnp.maximum(jnp.maximum(prevp_raw, -qr / dtcld), satdt / 2.0)
    prevp_pos = jnp.minimum(prevp_raw, satdt / 2.0)
    prevp = jnp.where(prevp_raw < 0.0, prevp_neg, prevp_pos)
    prevp = jnp.where(qr > 0.0, prevp, 0.0)
    # Nrevp (NR->NCCN): full rain evaporation. WRF (wdm62D ~1249): inside the
    # warm-rain do-k loop, if prevp == -qr/dtcld (rain fully evaporates this
    # step) move ALL rain number to CCN and zero Nr -- IMMEDIATELY, BEFORE the
    # cold-rain block runs. This is load-bearing: at cold subsaturated cells the
    # rain evaporates and nr->0 here, so the cold-rain rain-accretion terms
    # (pracs/psacr/...) see nr=0 and do not fire. Applying it later (after the
    # conservation update) wrongly leaves nr>0 in the cold block and wipes snow.
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

    xmi = den * qi / xni
    diameter = jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX)
    vt2i = 1.49e4 * diameter ** 1.31
    vt2r = C.PVTR * rslopeb[:, 0] * denfac
    vt2s = C.PVTS * rslopeb[:, 1] * denfac
    vt2g = C.PVTG * rslopeb[:, 2] * denfac
    qsum = jnp.maximum(qs + qg, 1.0e-15)
    vt2ave = jnp.where(qsum > 1.0e-15, (vt2s * qs + vt2g * qg) / qsum, 0.0)

    cold = supcol > 0.0
    ice_present = cold & (qi > qmin)
    safe_qi = jnp.where(qi > 0.0, qi, 1.0)
    safe_qr = jnp.where(qr > 0.0, qr, 1.0)
    safe_qs = jnp.where(qs > 0.0, qs, 1.0)
    safe_qc = jnp.where(qc > 0.0, qc, 1.0)

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
    eacrs = jnp.exp(0.07 * (-supcol))
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

    # pgacr (qg>qcrmin & qr>qcrmin)
    acrfac_pgacr = (30.0 * rslope3[:, 0] * rslope2[:, 0] * rslope[:, 2]
                    + 10.0 * rslope2[:, 0] * rslope2[:, 0] * rslope2[:, 2]
                    + 2.0 * rslope3[:, 0] * rslope3[:, 2])
    pgacr_raw = (C.PI * C.PI * nr * C.N0G * jnp.abs(vt2ave - vt2r)
                 * (C.DENR / den) * acrfac_pgacr)
    pgacr_raw = pgacr_raw * jnp.minimum(jnp.maximum(0.0, qg / safe_qr), 1.0) ** 2
    pgacr_raw = jnp.minimum(pgacr_raw, qr / dtcld)
    pgr_cond = (qg > C.QCRMIN) & (qr > C.QCRMIN)
    pgacr = jnp.where(pgr_cond, pgacr_raw, 0.0)

    # ngacr (qg>qcrmin & nr>nrmin)
    acrfac_ngacr = (1.5 * rslope2[:, 0] * rslope[:, 2]
                    + 1.0 * rslope[:, 0] * rslope2[:, 2] + 0.5 * rslope3[:, 2])
    ngacr_raw = C.PI * nr * C.N0G * jnp.abs(vt2ave - vt2r) * acrfac_ngacr
    ngacr_raw = ngacr_raw * jnp.minimum(jnp.maximum(0.0, qg / safe_qr), 1.0) ** 2
    ngacr_raw = jnp.minimum(ngacr_raw, nr / dtcld)
    ngacr = jnp.where((qg > C.QCRMIN) & (nr > C.NRMIN), ngacr_raw, 0.0)

    # pgacs = 0
    pgacs = jnp.zeros_like(t)

    # enhanced melting (supcol<=0): pseml/nseml, pgeml/ngeml
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
    safe_qg = jnp.where(qg > 0.0, qg, 1.0)
    gfac_g2 = rslope[:, 2] * C.N0G / safe_qg
    ngeml = jnp.where(warm2 & (qg > 0.0) & (qg > C.QCRMIN), -gfac_g2 * pgeml, 0.0)

    # deposition/sublimation + ice/snow aggregation
    pidep, psdep, pgdep, pigen, psaut, pgaut = _deposition_block(
        cold, qi, qs, qg, q, diameter, xni, rh2, work1_2, satdt2,
        prevp, dtcld, n0sfac, rslope, rslopeb, rslope2, work2, den, supcol,
        supsat2)

    # psevp/pgevp (supcol<0, rh1<1)
    coeres_s2 = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psevp_raw = ((rh1 - 1.0) * n0sfac * (C.PRECS1 * rslope2[:, 1]
                 + C.PRECS2 * work2 * coeres_s2) / work1_1)
    psevp_raw = jnp.minimum(jnp.maximum(psevp_raw, -qs / dtcld), 0.0)
    psevp = jnp.where((supcol < 0.0) & (qs > 0.0) & (rh1 < 1.0), psevp_raw, 0.0)
    coeres_g2 = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgevp_raw = ((rh1 - 1.0) * (C.PRECG1 * rslope2[:, 2]
                 + C.PRECG2 * work2 * coeres_g2) / work1_1)
    pgevp_raw = jnp.minimum(jnp.maximum(pgevp_raw, -qg / dtcld), 0.0)
    pgevp = jnp.where((supcol < 0.0) & (qg > 0.0) & (rh1 < 1.0), pgevp_raw, 0.0)

    # ===================== conservation feedback + update =====================
    delta2 = jnp.where((qr < 1.0e-4) & (qs < 1.0e-4), 1.0, 0.0)
    delta3 = jnp.where(qr < 1.0e-4, 1.0, 0.0)
    cold_branch = t <= C.T0C

    # ---------- COLD branch (t<=t0c) ----------
    # cloud water
    val = jnp.maximum(qmin, qc); src = (praut + pracw + paacw + paacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut * f; pracw_c = pracw * f; paacw_c = paacw * f
    # cloud ice
    val = jnp.maximum(qmin, qi)
    src = (psaut - pigen - pidep + praci + psaci + pgaci) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psaut_c = psaut * f; pigen_c = pigen * f; pidep_c = pidep * f
    praci_c = praci * f; psaci_c = psaci * f; pgaci_c = pgaci * f
    # rain (mass)
    val = jnp.maximum(qmin, qr)
    src = (-praut_c - prevp - pracw_c + piacr + psacr + pgacr) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut_c * f; prevp_c = prevp * f; pracw_c = pracw_c * f
    piacr_c = piacr * f; psacr_c = psacr * f; pgacr_c = pgacr * f
    # snow
    val = jnp.maximum(qmin, qs)
    src = -(psdep + psaut_c - pgaut + paacw_c + piacr_c * delta3
            + praci_c * delta3 - pracs * (1.0 - delta2)
            + psacr_c * delta2 + psaci_c - pgacs) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psdep_c = psdep * f; psaut_c = psaut_c * f; pgaut_c = pgaut * f
    paacw_c = paacw_c * f; piacr_c = piacr_c * f; praci_c = praci_c * f
    psaci_c = psaci_c * f; pracs_c = pracs * f; psacr_c = psacr_c * f
    pgacs_c = pgacs * f
    # graupel
    val = jnp.maximum(qmin, qg)
    src = -(pgdep + pgaut_c + piacr_c * (1.0 - delta3) + praci_c * (1.0 - delta3)
            + psacr_c * (1.0 - delta2) + pracs_c * (1.0 - delta2)
            + pgaci_c + paacw_c + pgacr_c + pgacs_c) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgdep_c = pgdep * f; pgaut_c = pgaut_c * f; piacr_c = piacr_c * f
    praci_c = praci_c * f; psacr_c = psacr_c * f; pracs_c = pracs_c * f
    paacw_c = paacw_c * f; pgaci_c = pgaci_c * f; pgacr_c = pgacr_c * f
    pgacs_c = pgacs_c * f
    # cloud number
    val = jnp.maximum(C.NCMIN, nc)
    src = (nraut + nccol + nracw + naacw + naacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_c = nraut * f; nccol_c = nccol * f; nracw_c = nracw * f; naacw_c = naacw * f
    # rain number
    val = jnp.maximum(C.NRMIN, nr)
    src = (-nraut_c + nrcol + niacr + nsacr + ngacr) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_c = nraut_c * f; nrcol_c = nrcol * f; niacr_c = niacr * f
    nsacr_c = nsacr * f; ngacr_c = ngacr * f

    w2_cold = -(prevp_c + psdep_c + pgdep_c + pigen_c + pidep_c)
    q_cold = q + w2_cold * dtcld
    qc_cold = jnp.maximum(qc - (praut_c + pracw_c + paacw_c + paacw_c) * dtcld, 0.0)
    qr_cold = jnp.maximum(qr + (praut_c + pracw_c + prevp_c - piacr_c - pgacr_c
                               - psacr_c) * dtcld, 0.0)
    qi_cold = jnp.maximum(qi - (psaut_c + praci_c + psaci_c + pgaci_c
                               - pigen_c - pidep_c) * dtcld, 0.0)
    qs_cold = jnp.maximum(qs + (psdep_c + psaut_c + paacw_c - pgaut_c
                               + piacr_c * delta3 + praci_c * delta3 + psaci_c
                               - pgacs_c - pracs_c * (1.0 - delta2)
                               + psacr_c * delta2) * dtcld, 0.0)
    qg_cold = jnp.maximum(qg + (pgdep_c + pgaut_c + piacr_c * (1.0 - delta3)
                               + praci_c * (1.0 - delta3) + psacr_c * (1.0 - delta2)
                               + pracs_c * (1.0 - delta2) + pgaci_c + paacw_c
                               + pgacr_c + pgacs_c) * dtcld, 0.0)
    nc_cold = jnp.maximum(nc + (-nraut_c - nccol_c - nracw_c - naacw_c - naacw_c) * dtcld, 0.0)
    nr_cold = jnp.maximum(nr + (nraut_c - nrcol_c - niacr_c - nsacr_c - ngacr_c) * dtcld, 0.0)
    xlf_cold = C.XLS - xl
    xlwork2_cold = (-C.XLS * (psdep_c + pgdep_c + pidep_c + pigen_c)
                    - xl * prevp_c - xlf_cold * (piacr_c + paacw_c + paacw_c
                                                 + pgacr_c + psacr_c))
    t_cold = t - xlwork2_cold / cpm * dtcld

    # ---------- WARM branch (t>t0c) ----------
    val = jnp.maximum(qmin, qc); src = (praut + pracw + paacw + paacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut * f; pracw_w = pracw * f; paacw_w = paacw * f
    # rain
    val = jnp.maximum(qmin, qr)
    src = (-paacw_w - praut_w + pseml + pgeml - pracw_w - paacw_w - prevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut_w * f; prevp_w = prevp * f; pracw_w = pracw_w * f
    paacw_w = paacw_w * f; pseml_w = pseml * f; pgeml_w = pgeml * f
    # snow
    val = jnp.maximum(C.QCRMIN, qs); src = (pgacs - pseml_w - psevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs * f; psevp_w = psevp * f; pseml_w = pseml_w * f
    # graupel
    val = jnp.maximum(C.QCRMIN, qg); src = -(pgacs_w + pgevp + pgeml_w) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs_w * f; pgevp_w = pgevp * f; pgeml_w = pgeml_w * f
    # cloud number
    val = jnp.maximum(C.NCMIN, nc)
    src = (nraut + nccol + nracw + naacw + naacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_w = nraut * f; nccol_w = nccol * f; nracw_w = nracw * f; naacw_w = naacw * f
    # rain number
    val = jnp.maximum(C.NRMIN, nr)
    src = (-nraut_w + nrcol - nseml - ngeml) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    nraut_w = nraut_w * f; nrcol_w = nrcol * f; nseml_w = nseml * f; ngeml_w = ngeml * f

    w2_warm = -(prevp_w + psevp_w + pgevp_w)
    q_warm = q + w2_warm * dtcld
    qc_warm = jnp.maximum(qc - (praut_w + pracw_w + paacw_w + paacw_w) * dtcld, 0.0)
    qr_warm = jnp.maximum(qr + (praut_w + pracw_w + prevp_w + paacw_w + paacw_w
                               - pseml_w - pgeml_w) * dtcld, 0.0)
    qs_warm = jnp.maximum(qs + (psevp_w - pgacs_w + pseml_w) * dtcld, 0.0)
    qg_warm = jnp.maximum(qg + (pgacs_w + pgevp_w + pgeml_w) * dtcld, 0.0)
    nc_warm = jnp.maximum(nc + (-nraut_w - nccol_w - nracw_w - naacw_w - naacw_w) * dtcld, 0.0)
    nr_warm = jnp.maximum(nr + (nraut_w - nrcol_w + nseml_w + ngeml_w) * dtcld, 0.0)
    xlf_warm = C.XLS - xl
    xlwork2_warm = (-xl * (prevp_w + psevp_w + pgevp_w)
                    - xlf_warm * (pseml_w + pgeml_w))
    t_warm = t - xlwork2_warm / cpm * dtcld

    # select branch per cell
    q = jnp.where(cold_branch, q_cold, q_warm)
    qc = jnp.where(cold_branch, qc_cold, qc_warm)
    qr = jnp.where(cold_branch, qr_cold, qr_warm)
    qi = jnp.where(cold_branch, qi_cold, qi)   # warm branch leaves qi unchanged
    qs = jnp.where(cold_branch, qs_cold, qs_warm)
    qg = jnp.where(cold_branch, qg_cold, qg_warm)
    nc = jnp.where(cold_branch, nc_cold, nc_warm)
    nr = jnp.where(cold_branch, nr_cold, nr_warm)
    t = jnp.where(cold_branch, t_cold, t_warm)

    # ---- recompute qsat(liquid+ice); small-drop rain->cloud (avedia<=di82) ----
    qsat1b, qsat2b = _qsat_fpvs(t, p)
    rh1b = jnp.maximum(q / qsat1b, qmin)
    rslope_b, rslopeb_b, rslope2_b, rslope3_b, _v, _vn = _slope_wdm6(
        qr, qs, qg, nr, den, denfac, t)
    avedia_r2 = rslope_b[:, 0] * (24.0 ** (1.0 / 3.0))
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
    # ncevp: if pcond == -qc/dtcld (full cloud evap) move Nc to Nn
    ncevp_full = pcond == (-qc / dtcld)
    nn = jnp.where(ncevp_full, nn + nc, nn)
    nc = jnp.where(ncevp_full, 0.0, nc)
    q = q - pcond * dtcld
    qc = jnp.maximum(qc + pcond * dtcld, 0.0)
    t = t + pcond * xl / cpm * dtcld

    # ---- small-value padding + lamdr/lamdc clamping (adjust Nr/Nc) ----
    qc = jnp.where(qc <= qmin, 0.0, qc)
    qi = jnp.where(qi <= qmin, 0.0, qi)
    # rain number clamp by slope limits
    rain_ok = (qr >= C.QCRMIN) & (nr >= C.NRMIN)
    safe_qr2 = jnp.where(qr > 0.0, qr, 1.0)
    lamdr = jnp.exp(jnp.log((C.PIDNR * nr) / (den * safe_qr2)) * (1.0 / 3.0))
    lamdr_lo = lamdr <= C.LAMDARMIN
    lamdr_hi = lamdr >= C.LAMDARMAX
    nr_lo = den * qr * C.LAMDARMIN ** 3 / C.PIDNR
    nr_hi = den * qr * C.LAMDARMAX ** 3 / C.PIDNR
    nr = jnp.where(rain_ok & lamdr_lo, nr_lo,
                   jnp.where(rain_ok & lamdr_hi, nr_hi, nr))
    # cloud number clamp by slope limits
    cloud_ok2 = (qc >= qmin) & (nc >= C.NCMIN)
    safe_qc2 = jnp.where(qc > 0.0, qc, 1.0)
    lamdc = jnp.exp(jnp.log((C.PIDNC * nc) / (den * safe_qc2)) * (1.0 / 3.0))
    lamdc_lo = lamdc <= C.LAMDACMIN
    lamdc_hi = lamdc >= C.LAMDACMAX
    nc_lo = den * qc * C.LAMDACMIN ** 3 / C.PIDNC
    nc_hi = den * qc * C.LAMDACMAX ** 3 / C.PIDNC
    nc = jnp.where(cloud_ok2 & lamdc_lo, nc_lo,
                   jnp.where(cloud_ok2 & lamdc_hi, nc_hi, nc))

    return (t, q, qc, qr, qi, qs, qg, nn, nc, nr, cpm, xl,
            rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr)


# --------------------------------------------------------------------------
# rain + Nr in-line sedimentation (wdm62D's own mstep loop, NOT the plm).
# WRF advects rain mass and rain number through a forward-upwind multi-substep
# scheme with mstep substeps, recomputing the rain slope each substep
# (slope_rain). Here mstep is the column-max integer substep count.
# --------------------------------------------------------------------------
def _rain_number_fallout(qr, nr, qs, qg, den, denfac, t, delz, dtcld,
                         work1_r0, workn_r0, mstep):
    km = qr.shape[0]
    # mstep is a per-column scalar (same value used for all k, = column max).
    # WRF uses per-column mstep(i); for a single column this is one integer.
    mstep_i = jnp.maximum(mstep, 1.0)

    def substep(n, carry):
        qr, nr, fall1, falln, work1_r, workn_r = carry
        do_n = (n < mstep_i)  # scalar mask (n is 0-based here vs WRF 1-based)

        # falk/falkn store the per-cell fluxes computed AT EACH CELL'S PROCESSING
        # TIME (WRF semantics): the dqr(k+1) flux-in uses the STORED falk(k+1)
        # and the already-updated qrs(k+1), NOT a recompute from qrs(k+1).
        falk = jnp.zeros(km, qr.dtype)
        falkn = jnp.zeros(km, qr.dtype)

        # top cell (k = kte = km-1)
        falk_top = den[km - 1] * qr[km - 1] * work1_r[km - 1] / mstep_i
        falkn_top = nr[km - 1] * workn_r[km - 1] / mstep_i
        falk = falk.at[km - 1].set(falk_top)
        falkn = falkn.at[km - 1].set(falkn_top)
        fall1 = fall1.at[km - 1].add(jnp.where(do_n, falk_top, 0.0))
        falln = falln.at[km - 1].add(jnp.where(do_n, falkn_top, 0.0))
        qr = qr.at[km - 1].set(jnp.where(do_n,
            jnp.maximum(qr[km - 1] - falk_top * dtcld / den[km - 1], 0.0), qr[km - 1]))
        nr = nr.at[km - 1].set(jnp.where(do_n,
            jnp.maximum(nr[km - 1] - falkn_top * dtcld, 0.0), nr[km - 1]))

        # interior cells, top-1 down to bottom (k = km-2 .. 0)
        def kbody(j, carry2):
            qr, nr, fall1, falln, falk, falkn = carry2
            k = km - 2 - j  # from km-2 down to 0
            falk_k = den[k] * qr[k] * work1_r[k] / mstep_i
            falkn_k = nr[k] * workn_r[k] / mstep_i
            falk = falk.at[k].set(falk_k)
            falkn = falkn.at[k].set(falkn_k)
            fall1 = fall1.at[k].add(jnp.where(do_n, falk_k, 0.0))
            falln = falln.at[k].add(jnp.where(do_n, falkn_k, 0.0))
            # dqr from this cell + flux in from k+1: WRF uses STORED falk(k+1)
            # and the already-updated qr[k+1]/nr[k+1].
            dqr_k = jnp.minimum(falk_k * dtcld / den[k], qr[k])
            dqr_kp1 = jnp.minimum(falk[k + 1] * delz[k + 1] / delz[k] * dtcld / den[k],
                                  qr[k + 1])
            dnr_k = jnp.minimum(falkn_k * dtcld, nr[k])
            dnr_kp1 = jnp.minimum(falkn[k + 1] * delz[k + 1] / delz[k] * dtcld, nr[k + 1])
            qr_new = jnp.maximum(qr[k] - dqr_k + dqr_kp1, 0.0)
            nr_new = jnp.maximum(nr[k] - dnr_k + dnr_kp1, 0.0)
            qr = qr.at[k].set(jnp.where(do_n, qr_new, qr[k]))
            nr = nr.at[k].set(jnp.where(do_n, nr_new, nr[k]))
            return qr, nr, fall1, falln, falk, falkn

        qr, nr, fall1, falln, falk, falkn = jax.lax.fori_loop(
            0, km - 1, kbody, (qr, nr, fall1, falln, falk, falkn))

        # recompute rain slope/vt from updated qr,nr (slope_rain), /delz
        vt_m, vt_n = _slope_rain_vt(qr, nr, den, denfac)
        work1_r = jnp.where(do_n, vt_m / delz, work1_r)
        workn_r = jnp.where(do_n, vt_n / delz, workn_r)
        return qr, nr, fall1, falln, work1_r, workn_r

    fall1 = jnp.zeros(km, qr.dtype)
    falln = jnp.zeros(km, qr.dtype)
    # bound substeps by a static cap (savepoint columns: dt=90, vt small -> few
    # substeps). 64 is a safe ceiling; inactive substeps are masked.
    carry = (qr, nr, fall1, falln, work1_r0, workn_r0)
    qr, nr, fall1, falln, _w1, _wn = jax.lax.fori_loop(0, 64, substep, carry)
    return qr, nr, fall1, falln


# vmap the single-column scheme across the column (i) dimension.
_wdm6_columns = jax.jit(
    jax.vmap(_wdm6_column,
             in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0)),
    static_argnums=(13,))


# --------------------------------------------------------------------------
# Public adapter
# --------------------------------------------------------------------------
def wdm6_run(t, qv, qc, qr, qi, qs, qg, nn, nc, nr, den, p, delz, delt, slmsk):
    """Run WDM6 on a batch of columns (shape (ncol, nlev)).

    Inputs are TEMPERATURE-based (t in K), matching the WRF wdm62D internal
    state. ``nn,nc,nr`` are the CCN / cloud-droplet / rain number concentrations
    (# kg^-1). ``slmsk`` is the per-column land/sea mask (1=land, 2=water);
    pass a (ncol,) array. Returns a dict with updated t, qv, qc, qr, qi, qs, qg,
    nn, nc, nr, surface precip increments (mm), sr, and effective radii.
    """
    return _wdm6_columns(t, qv, qc, qr, qi, qs, qg, nn, nc, nr,
                         den, p, delz, float(delt), slmsk)


def wdm6_physics_tendency(theta, qv, qc, qr, qi, qs, qg, nn, nc, nr,
                          pii, den, p, delz, delt, slmsk):
    """WDM6 adapter returning a frozen PhysicsTendency (in-place replacements).

    ``theta`` and the moist/number species are (ncol, nlev). ``pii`` is the
    Exner function used to convert theta<->t (as the WRF ``wdm6`` wrapper does).
    Surface precip increments are per-call mm.

    NOTE (Nc/Nn integration): WDM6 introduces the additive State leaves ``Nc``
    (qnc) and ``Nn`` (qnn) per the v0.6.0 S0 plan. ``Nr`` (qnr) already exists.
    This adapter returns ``state_replacements`` for nc/nr; ``nn`` is the CCN
    leaf that the manager-owned State.__slots__ patch must materialize before
    integration (it is NOT in STATE_TENDENCY_KEYS yet). Until then ``nn`` is
    returned in ``diagnostics`` so no information is lost. Per-scheme parity is
    validated at the SCHEME-FUNCTION level via ``wdm6_run``.
    """
    t = theta * pii
    out = wdm6_run(t, qv, qc, qr, qi, qs, qg, nn, nc, nr, den, p, delz, delt, slmsk)
    theta_new = out["t"] / pii

    state_repl = {
        "theta": theta_new,
        "qv": out["qv"], "qc": out["qc"], "qr": out["qr"],
        "qi": out["qi"], "qs": out["qs"], "qg": out["qg"],
        "Nr": out["nr"],
    }
    # Nc is an additive State leaf; include it as a replacement when the State
    # patch lands. It is in STATE_TENDENCY_KEYS via NUMBER_SPECIES ("Nc").
    state_repl["Nc"] = out["nc"]

    tend = PhysicsTendency(
        state_replacements=state_repl,
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["snowncv"],
            "graupel_acc": out["graupelncv"],
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


__all__ = ["wdm6_run", "wdm6_physics_tendency"]
