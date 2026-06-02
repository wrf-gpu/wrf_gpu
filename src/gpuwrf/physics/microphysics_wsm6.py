"""JAX WSM6 single-moment 6-class microphysics (WRF mp_physics=6).

Faithful port of WRF ``phys/physics_mmm/mp_wsm6.F90`` (subroutine
``mp_wsm6_run`` + ``slope_wsm6`` + ``nislfv_rain_plm`` + ``nislfv_rain_plm6``)
and ``mp_wsm6_effectRad.F90`` (``mp_wsm6_effectRad_run``).

WRF process order is preserved exactly:

  for loop in 1..loops (minor time steps):
    1. denfac, qsat(liquid,ice), rh, init process arrays, xni
    2. slope_wsm6; semi-Lagrangian fallout of rain / (snow+graupel)
    3. snow/graupel melting for T>T0 (psmlt, pgmlt)
    4. ice fallout (Vice + nislfv_rain_plm)
    5. surface precip accumulation (rain/snow/graupel/sr)
    6. instantaneous melt/freeze (pimlt, pihmf, pihtf, pgfrz)
    7. slope update; work1(diffac), work2(venfac)
    8. warm-rain (praut, pracw, prevp)
    9. cold-rain accretion/deposition/aggregation/enhanced-melting block
    10. mass-conservation feedback + state update (T<=T0 / T>T0 branches)
    11. recompute qsat; pcond condensation; small-value padding

The scheme works on TEMPERATURE ``t`` internally; the WRF ``wsm6`` wrapper
converts th<->t via the Exner function ``pii``. This port mirrors that: the
adapter accepts the column state and returns a ``PhysicsTendency`` with
``state_replacements`` for the updated theta + moist species (WSM6 is an
in-place scheme), ``accumulator_increments`` for surface precip, and
``diagnostics`` for the effective radii.

Validation: per-column WRF savepoint parity against the real Fortran scheme
(proofs/v060/oracle). The port defaults to fp64; the WRF MMM physics runs in
fp32 (kind_phys=selected_real_kind(6)), so parity is to a predeclared physical
tolerance, never bitwise.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import wsm6_constants as C


# --------------------------------------------------------------------------
# Inline thermodynamic helper functions (mp_wsm6.F90 statement functions)
# --------------------------------------------------------------------------
def _cpmcal(q):
    # cpd*(1-max(q,qmin)) + max(q,qmin)*cpv
    qq = jnp.maximum(q, C.QMIN)
    return C.CPD * (1.0 - qq) + qq * C.CPV


def _xlcal(t):
    return C.XLV0 - C.XLV1 * (t - C.T0C)


def _diffus(x, y):
    # diffusion coefficient of water vapor: 8.794e-5 * x**1.81 / y
    return 8.794e-5 * jnp.exp(jnp.log(x) * 1.81) / y


def _viscos(x, y):
    # kinematic viscosity: 1.496e-6 * x**1.5 / (x+120) / y
    return 1.496e-6 * (x * jnp.sqrt(x)) / (x + 120.0) / y


def _xka(x, y):
    return 1.414e3 * _viscos(x, y) * y


def _diffac(a, b, c, d, e):
    # a=xl, b=p, c=t, d=den, e=qsat
    return d * a * a / (_xka(c, d) * C.RV * c * c) + 1.0 / (e * _diffus(c, b))


def _venfac(a, b, c):
    # a=p, b=t, c=den
    return (jnp.exp(jnp.log(_viscos(b, c) / _diffus(b, a)) * (1.0 / 3.0))
            / jnp.sqrt(_viscos(b, c)) * jnp.sqrt(jnp.sqrt(C.DEN0 / c)))


def _conden(a, b, c, d, e):
    # a=t, b=q, c=qsat, d=xl, e=cpm
    return (jnp.maximum(b, C.QMIN) - c) / (1.0 + d * d / (C.RV * e) * c / (a * a))


def _qsat_fpvs(t, p):
    """Inline fpvs expansion (mp_wsm6.F90 lines ~519-547): returns (qsat1, qsat2).

    qsat1 = saturation mixing ratio over LIQUID.
    qsat2 = saturation mixing ratio over ICE when t<ttp else over liquid.
    """
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


# --------------------------------------------------------------------------
# slope_wsm6: rain/snow/graupel slope parameters + fall speeds (per column)
# --------------------------------------------------------------------------
def _slope_wsm6(qr, qs, qg, den, denfac, t):
    """Return rslope, rslopeb, rslope2, rslope3 (each (K,3)) and vt (K,3).

    Index 0=rain, 1=snow, 2=graupel. Mirrors slope_wsm6 exactly.
    """
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

    rslope = jnp.stack([rsl_r, rsl_s, rsl_g], axis=-1)
    rslopeb = jnp.stack([rslb_r, rslb_s, rslb_g], axis=-1)
    rslope2 = jnp.stack([rsl2_r, rsl2_s, rsl2_g], axis=-1)
    rslope3 = jnp.stack([rsl3_r, rsl3_s, rsl3_g], axis=-1)

    vt_r = jnp.where(qr <= 0.0, 0.0, C.PVTR * rslb_r * denfac)
    vt_s = jnp.where(qs <= 0.0, 0.0, C.PVTS * rslb_s * denfac)
    vt_g = jnp.where(qg <= 0.0, 0.0, C.PVTG * rslb_g * denfac)
    vt = jnp.stack([vt_r, vt_s, vt_g], axis=-1)
    return rslope, rslopeb, rslope2, rslope3, vt


def _slope_single(q, den, denfac, t, kind):
    """slope_rain/slope_snow/slope_graup combined; returns vt only (used in nislfv).

    kind: 'r','s','g'. Returns terminal velocity vt(K) for arrival-point re-eval.
    """
    if kind == "r":
        lamda = jnp.sqrt(jnp.sqrt(C.PIDN0R / (q * den)))
        rsl = jnp.where(q <= C.QCRMIN, C.RSLOPERMAX, 1.0 / lamda)
        rslb = jnp.where(q <= C.QCRMIN, C.RSLOPERBMAX, rsl ** C.BVTR)
        vt = jnp.where(q <= 0.0, 0.0, C.PVTR * rslb * denfac)
    elif kind == "s":
        supcol = C.T0C - t
        n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
        lamda = jnp.sqrt(jnp.sqrt(C.PIDN0S * n0sfac / (q * den)))
        rsl = jnp.where(q <= C.QCRMIN, C.RSLOPESMAX, 1.0 / lamda)
        rslb = jnp.where(q <= C.QCRMIN, C.RSLOPESBMAX, rsl ** C.BVTS)
        vt = jnp.where(q <= 0.0, 0.0, C.PVTS * rslb * denfac)
    else:  # graupel
        lamda = jnp.sqrt(jnp.sqrt(C.PIDN0G / (q * den)))
        rsl = jnp.where(q <= C.QCRMIN, C.RSLOPEGMAX, 1.0 / lamda)
        rslb = jnp.where(q <= C.QCRMIN, C.RSLOPEGBMAX, rsl ** C.BVTG)
        vt = jnp.where(q <= 0.0, 0.0, C.PVTG * rslb * denfac)
    return vt


# --------------------------------------------------------------------------
# nislfv_rain_plm: semi-Lagrangian PLM sedimentation (single species).
# Mirrors mp_wsm6.F90 nislfv_rain_plm; column index 0=bottom..km-1=top.
# --------------------------------------------------------------------------
def _wi_interp(ww, dz, km):
    """3rd/2nd-order interface velocity wi (length km+1), per WRF."""
    fa1 = 9.0 / 16.0
    fa2 = 1.0 / 16.0
    # 3rd-order interior (k=3..km-1 in 1-based => indices 2..km-2 0-based)
    wi = jnp.zeros(km + 1, dtype=ww.dtype)
    wi = wi.at[0].set(ww[0])
    wi = wi.at[1].set(0.5 * (ww[1] + ww[0]))
    if km >= 4:
        k = jnp.arange(2, km - 1)  # 0-based -> WRF k=3..km-1
        interior = fa1 * (ww[k] + ww[k - 1]) - fa2 * (ww[k + 1] + ww[k - 2])
        wi = wi.at[2:km - 1].set(interior)
    wi = wi.at[km - 1].set(0.5 * (ww[km - 1] + ww[km - 2]))
    wi = wi.at[km].set(ww[km - 1])
    # terminate at top of raingroup: where ww(k)==0 -> wi(k)=ww(k-1)  (WRF k=2..km)
    k = jnp.arange(1, km)
    wi = wi.at[1:km].set(jnp.where(ww[1:km] == 0.0, ww[0:km - 1], wi[1:km]))
    return wi


def _diffusivity_limit(wi, dz, dt, km):
    """Downward diffusivity limit on wi (decfl<con1). Sequential in k (top->bottom)."""
    con1 = 0.05

    def body(k_rev, wi):
        k = km - 1 - k_rev  # 0-based, from km-1 down to 0
        decfl = (wi[k + 1] - wi[k]) * dt / dz[k]
        new = wi[k + 1] - con1 * dz[k] / dt
        wi = wi.at[k].set(jnp.where(decfl > con1, new, wi[k]))
        return wi

    wi = jax.lax.fori_loop(0, km, body, wi)
    return wi


def _plm_qmi_qpi(qa, dza, km):
    """Monotone PLM edge values qmi,qpi (length km+1) from cell values qa."""
    qmi = jnp.zeros(km + 1, dtype=qa.dtype)
    qpi = jnp.zeros(km + 1, dtype=qa.dtype)
    # interior k=2..km (1-based) -> indices 1..km-1 0-based
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
    """Remap arrival-cell PLM profile back to fixed grid -> qn(km). Sequential."""

    def body(k, carry):
        qn, kb, kt = carry
        kb = jnp.maximum(kb - 1, 0)
        kt = jnp.maximum(kt - 1, 0)
        # if zi(k) >= za(km+1) -> contribution zero, leave qn(k)=0
        done = zi[k] >= za[km]

        # find_kb: first kk in [kb,km-1] with zi(k) <= za(kk+1)
        def find_kb(_):
            def cond(state):
                kk, found = state
                return jnp.logical_and(kk < km, jnp.logical_not(found))

            def step(state):
                kk, found = state
                hit = zi[k] <= za[kk + 1]
                return jnp.where(hit, kk, kk + 1), jnp.logical_or(found, hit)

            kk0, _ = jax.lax.while_loop(cond, step, (kb, False))
            return kk0

        kb_new = find_kb(None)

        # find_kt: first kk in [kt,km-1] with zi(k+1) <= za(kk); then kt=kt-1.
        # WRF: if the condition is never met, kt RETAINS its entry value (the
        # `kt=kk` assignment only fires on a hit), then kt=kt-1.
        kt_entry = kt

        def find_kt(_):
            def cond(state):
                kk, found = state
                return jnp.logical_and(kk < km, jnp.logical_not(found))

            def step(state):
                kk, found = state
                hit = zi[k + 1] <= za[kk]
                return jnp.where(hit, kk, kk + 1), jnp.logical_or(found, hit)

            kk0, found = jax.lax.while_loop(cond, step, (kt, False))
            return jnp.where(found, kk0, kt_entry) - 1

        kt_new = find_kt(None)

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

        # middle cells m = kb+1 .. kt-1 (sum via masked dot over all cells)
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
        # once done, keep kb/kt frozen (further k also done since zi increasing)
        return qn, kb_c, kt_c

    qn0 = jnp.zeros(km, dtype=qa.dtype)
    qn, _, _ = jax.lax.fori_loop(0, km, body, (qn0, 0, 0))
    return qn


def _precip_out(qa, dza, za, km):
    """Sum the mass that falls below the surface (za<0)."""
    # WRF: for k=1..km: if za(k)<0 and za(k+1)<0: precip += qa(k)*dza(k); cycle
    #      elif za(k)<0 and za(k+1)>=0: precip += qa(k)*(0-za(k)); exit
    #      else exit
    both_below = (za[0:km] < 0.0) & (za[1:km + 1] < 0.0)
    straddle = (za[0:km] < 0.0) & (za[1:km + 1] >= 0.0)
    # contribution is taken while contiguous from k=1 upward until exit.
    # Since za is monotincreasing in k (arrival depth decreases upward), the
    # below-surface region is a contiguous prefix. cumulative AND of "started".
    contrib_full = jnp.where(both_below, qa[0:km] * dza[0:km], 0.0)
    contrib_str = jnp.where(straddle, qa[0:km] * (0.0 - za[0:km]), 0.0)
    # active prefix: position k is summed iff za[0..k] satisfies the run.
    # both_below cells form prefix; first straddle ends it.
    prefix_below = jnp.cumprod(both_below.astype(qa.dtype))  # 1 while contiguous
    # straddle cell is the one right after the last both_below (or at k=0)
    started = jnp.concatenate([jnp.ones(1, qa.dtype), prefix_below[:-1]])
    precip = jnp.sum(contrib_full * prefix_below) + jnp.sum(contrib_str * started)
    return precip


def _nislfv_single(q, den, denfac, tk, dz, ww_in, dt, iter_n):
    """nislfv_rain_plm for one species, one column. Returns (q_new, precip)."""
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

    # iteration for mean terminal velocity
    def iter_body(n, state):
        ww, was = state
        wi, za, dza, qa_cells, qr_cells = iterate(ww)
        wa = _slope_single(qr_cells, den, denfac, tk, "r")
        wa = jnp.where(n >= 1, 0.5 * (wa + was), wa)  # n>=2 (1-based) => n>=1 here
        ww_new = 0.5 * (wd + wa)
        return ww_new, wa

    ww = wd
    was = jnp.zeros(km, dtype=q.dtype)
    if iter_n >= 1:
        ww, was = jax.lax.fori_loop(0, iter_n, iter_body, (ww, was))

    wi, za, dza, qa_cells, qr_cells = iterate(ww)
    qa = jnp.concatenate([qa_cells, jnp.zeros(1, q.dtype)])  # qa(km+1)=0
    qmi, qpi = _plm_qmi_qpi(qa, dza, km)
    qn = _plm_remap(qa, dza, qmi, qpi, zi, za, km)
    precip = _precip_out(qa, dza, za, km)

    q_new = jnp.where(allold <= 0.0, q, qn)
    precip = jnp.where(allold <= 0.0, 0.0, precip)
    return q_new, precip


def _nislfv_double(qs, qg, den, denfac, tk, dz, ww_in, dt, iter_n):
    """nislfv_rain_plm6 for snow+graupel, one column. Returns (qs_new, qg_new, ps, pg)."""
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
        wa_s = _slope_single(qr_s, den, denfac, tk, "s")
        wa_g = _slope_single(qr_g, den, denfac, tk, "g")
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
# Effective radii (mp_wsm6_effectRad_run)
# --------------------------------------------------------------------------
def _effective_radii(t, qc, qi, qs, rho):
    R1 = 1.0e-12
    R2 = 1.0e-6
    obmr = 1.0 / 3.0
    nc0 = 3.0e8

    re_qc = jnp.full_like(t, C.RE_QC_BG)
    re_qi = jnp.full_like(t, C.RE_QI_BG)
    re_qs = jnp.full_like(t, C.RE_QS_BG)

    rqc = jnp.maximum(R1, qc * rho)
    rqi = jnp.maximum(R1, qi * rho)
    temp = rho * jnp.maximum(qi, C.QMIN)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    ni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)
    rni = jnp.maximum(R2, ni * rho)
    rqs = jnp.maximum(R1, qs * rho)

    # cloud
    lamdac = (C.PIDNC * nc0 / rqc) ** obmr
    re_c = jnp.maximum(2.51e-6, jnp.minimum(1.5 * (1.0 / lamdac), C.RE_QC_MAX))
    re_qc = jnp.where(rqc > R1, re_c, re_qc)

    # ice
    diai = 11.9 * jnp.sqrt(rqi / ni)
    re_i = jnp.maximum(10.01e-6, jnp.minimum(0.75 * 0.163 * diai, C.RE_QI_MAX))
    re_qi = jnp.where((rqi > R1) & (rni > R2), re_i, re_qi)

    # snow
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    lamdas = jnp.sqrt(jnp.sqrt(C.PIDN0S * n0sfac / rqs))
    re_s = jnp.maximum(25.0e-6, jnp.minimum(0.5 * (1.0 / lamdas), C.RE_QS_MAX))
    re_qs = jnp.where(rqs > R1, re_s, re_qs)

    re_qc = jnp.maximum(C.RE_QC_BG, jnp.minimum(re_qc, C.RE_QC_MAX))
    re_qi = jnp.maximum(C.RE_QI_BG, jnp.minimum(re_qi, C.RE_QI_MAX))
    re_qs = jnp.maximum(C.RE_QS_BG, jnp.minimum(re_qs, C.RE_QS_MAX))
    return re_qc, re_qi, re_qs


# --------------------------------------------------------------------------
# Core single-column WSM6 (mp_wsm6_run for one column, all minor loops)
# --------------------------------------------------------------------------
def _wsm6_column(t, q, qc, qr, qi, qs, qg, den, p, delz, delt):
    """One column. Returns dict of updated fields + surface precip increments."""
    km = t.shape[0]
    qmin = C.QMIN

    # negative padding from dynamics
    qc = jnp.maximum(qc, 0.0)
    qr = jnp.maximum(qr, 0.0)
    qi = jnp.maximum(qi, 0.0)
    qs = jnp.maximum(qs, 0.0)
    qg = jnp.maximum(qg, 0.0)

    cpm = _cpmcal(q)
    xl = _xlcal(t)

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

    state = (t, q, qc, qr, qi, qs, qg, cpm, xl,
             rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr)

    for _loop in range(loops):
        state = _wsm6_minor_loop(state, den, p, delz, dtcld)

    (t, q, qc, qr, qi, qs, qg, cpm, xl,
     rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr) = state

    re_qc, re_qi, re_qs = _effective_radii(t, qc, qi, qs, den)

    return {
        "t": t, "qv": q, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
        "rainncv": rainncv, "snowncv": snowncv, "graupelncv": graupelncv,
        "sr": sr, "re_cloud": re_qc, "re_ice": re_qi, "re_snow": re_qs,
    }


def _wsm6_minor_loop(state, den, p, delz, dtcld):
    (t, q, qc, qr, qi, qs, qg, cpm, xl,
     rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr) = state
    km = t.shape[0]
    qmin = C.QMIN

    denfac = jnp.sqrt(C.DEN0 / den)

    qsat1, qsat2 = _qsat_fpvs(t, p)
    rh1 = jnp.maximum(q / qsat1, qmin)
    rh2 = jnp.maximum(q / qsat2, qmin)

    # ice crystal number conc (HDC 5c)
    temp = den * jnp.maximum(qi, qmin)
    temp = jnp.sqrt(jnp.sqrt(temp * temp * temp))
    xni = jnp.minimum(jnp.maximum(5.38e7 * temp, 1.0e3), 1.0e6)

    # ---- fallout (rain / snow+graupel) ----
    rslope, rslopeb, rslope2, rslope3, work1_vt = _slope_wsm6(qr, qs, qg, den, denfac, t)
    workr = work1_vt[:, 0]
    qsum = jnp.maximum(qs + qg, 1.0e-15)
    worka = jnp.where(qsum > 1.0e-15,
                      (work1_vt[:, 1] * qs + work1_vt[:, 2] * qg) / qsum, 0.0)
    workr = jnp.where(qr <= 0.0, 0.0, workr)

    denqrs1 = den * qr
    denqrs2 = den * qs
    denqrs3 = den * qg

    denqrs1_new, delqrs1 = _nislfv_single(denqrs1, den, denfac, t, delz, workr, dtcld, 1)
    denqrs2_new, denqrs3_new, delqrs2, delqrs3 = _nislfv_double(
        denqrs2, denqrs3, den, denfac, t, delz, worka, dtcld, 1)

    qr = jnp.maximum(denqrs1_new / den, 0.0)
    qs = jnp.maximum(denqrs2_new / den, 0.0)
    qg = jnp.maximum(denqrs3_new / den, 0.0)
    fall1 = denqrs1_new * workr / delz
    fall2 = denqrs2_new * worka / delz
    fall3 = denqrs3_new * worka / delz
    fall1 = fall1.at[0].set(delqrs1 / delz[0] / dtcld)
    fall2 = fall2.at[0].set(delqrs2 / delz[0] / dtcld)
    fall3 = fall3.at[0].set(delqrs3 / delz[0] / dtcld)

    # WRF recomputes slopes from POST-sedimentation qr/qs/qg (slope_wsm6 call #2,
    # mp_wsm6.F90 lines 654-662) before the snow/graupel melting block.
    rslope, rslopeb, rslope2, rslope3, _vt = _slope_wsm6(qr, qs, qg, den, denfac, t)

    # ---- snow/graupel melting for T>T0 (psmlt, pgmlt) ----
    supcol = C.T0C - t
    n0sfac = jnp.maximum(jnp.minimum(jnp.exp(C.ALPHA * supcol), C.N0SMAX / C.N0S), 1.0)
    warm = t > C.T0C
    xlf = C.XLF0
    work2 = _venfac(p, t, den)
    # psmlt
    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psmlt = (_xka(t, den) / xlf * (C.T0C - t) * C.PI / 2.0 * n0sfac
             * (C.PRECS1 * rslope2[:, 1] + C.PRECS2 * work2 * coeres_s) / den)
    psmlt = jnp.minimum(jnp.maximum(psmlt * dtcld, -qs), 0.0)  # mstep=1
    do_psmlt = warm & (qs > 0.0)
    qs = jnp.where(do_psmlt, qs + psmlt, qs)
    qr = jnp.where(do_psmlt, qr - psmlt, qr)
    t = jnp.where(do_psmlt, t + xlf / cpm * psmlt, t)
    # pgmlt
    coeres_g = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgmlt = (_xka(t, den) / xlf * (C.T0C - t)
             * (C.PRECG1 * rslope2[:, 2] + C.PRECG2 * work2 * coeres_g) / den)
    pgmlt = jnp.minimum(jnp.maximum(pgmlt * dtcld, -qg), 0.0)
    do_pgmlt = warm & (qg > 0.0)
    qg = jnp.where(do_pgmlt, qg + pgmlt, qg)
    qr = jnp.where(do_pgmlt, qr - pgmlt, qr)
    t = jnp.where(do_pgmlt, t + xlf / cpm * pgmlt, t)

    # ---- ice fallout (Vice) ----
    xmi = den * qi / xni
    diameter = jnp.maximum(jnp.minimum(C.DICON * jnp.sqrt(xmi), C.DIMAX), 1.0e-25)
    work1c = jnp.where(qi <= 0.0, 0.0, 1.49e4 * jnp.exp(jnp.log(diameter) * 1.31))
    denqci = den * qi
    denqci_new, delqi = _nislfv_single(denqci, den, denfac, t, delz, work1c, dtcld, 0)
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
    # snow present path (we always carry snow/graupel) -> use snowncv+graupelncv
    sr = jnp.where(fallsum > 0.0, (snowncv + graupelncv) / (rainncv + 1.0e-12), sr)

    # ---- instantaneous melt/freeze ----
    supcol = C.T0C - t
    xlf_im = C.XLS - xl
    xlf_im = jnp.where(supcol < 0.0, C.XLF0, xlf_im)
    # pimlt: I->C for supcol<0
    do_im = (supcol < 0.0) & (qi > 0.0)
    qc = jnp.where(do_im, qc + qi, qc)
    t = jnp.where(do_im, t - xlf_im / cpm * qi, t)
    qi = jnp.where(do_im, 0.0, qi)
    # pihmf: C->I for supcol>40
    do_hmf = (supcol > 40.0) & (qc > 0.0)
    qi = jnp.where(do_hmf, qi + qc, qi)
    t = jnp.where(do_hmf, t + xlf_im / cpm * qc, t)
    qc = jnp.where(do_hmf, 0.0, qc)
    # pihtf: heterogeneous freezing C->I for supcol>0
    supcolt = jnp.minimum(supcol, 50.0)
    pfrzdtc = jnp.minimum(
        C.PFRZ1 * (jnp.exp(C.PFRZ2 * supcolt) - 1.0)
        * den / C.DENR / C.XNCR * qc * qc * dtcld, qc)
    do_htf = (supcol > 0.0) & (qc > qmin)
    qi = jnp.where(do_htf, qi + pfrzdtc, qi)
    t = jnp.where(do_htf, t + xlf_im / cpm * pfrzdtc, t)
    qc = jnp.where(do_htf, qc - pfrzdtc, qc)
    # pgfrz: rain->graupel for supcol>0
    rs3 = rslope3[:, 0]
    temp_f = rs3 * rs3 * rslope[:, 0]
    pfrzdtr = jnp.minimum(
        20.0 * (C.PI * C.PI) * C.PFRZ1 * C.N0R * C.DENR / den
        * (jnp.exp(C.PFRZ2 * supcolt) - 1.0) * temp_f * dtcld, qr)
    do_gfrz = (supcol > 0.0) & (qr > 0.0)
    qg = jnp.where(do_gfrz, qg + pfrzdtr, qg)
    t = jnp.where(do_gfrz, t + xlf_im / cpm * pfrzdtr, t)
    qr = jnp.where(do_gfrz, qr - pfrzdtr, qr)

    # ---- slope update + work1(diffac)/work2(venfac) ----
    rslope, rslopeb, rslope2, rslope3, _vt2 = _slope_wsm6(qr, qs, qg, den, denfac, t)
    work1_1 = _diffac(xl, p, t, den, qsat1)
    work1_2 = _diffac(C.XLS, p, t, den, qsat2)
    work2 = _venfac(p, t, den)

    # ---- warm rain ----
    supsat = jnp.maximum(q, qmin) - qsat1
    satdt = supsat / dtcld
    praut = jnp.where(qc > C.QC0,
                      jnp.minimum(C.QCK1 * qc ** (7.0 / 3.0), qc / dtcld), 0.0)
    pracw = jnp.where((qr > C.QCRMIN) & (qc > qmin),
                      jnp.minimum(C.PACRR * rslope3[:, 0] * rslopeb[:, 0]
                                  * qc * denfac, qc / dtcld), 0.0)
    coeres_r = rslope2[:, 0] * jnp.sqrt(rslope[:, 0] * rslopeb[:, 0])
    prevp_raw = ((rh1 - 1.0) * (C.PRECR1 * rslope2[:, 0]
                 + C.PRECR2 * work2 * coeres_r) / work1_1)
    prevp_neg = jnp.maximum(jnp.maximum(prevp_raw, -qr / dtcld), satdt / 2.0)
    prevp_pos = jnp.minimum(prevp_raw, satdt / 2.0)
    prevp = jnp.where(prevp_raw < 0.0, prevp_neg, prevp_pos)
    prevp = jnp.where(qr > 0.0, prevp, 0.0)

    # ---- cold rain block ----
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

    # praci, piacr (need qr>qcrmin)
    acrfac_r = (2.0 * rslope3[:, 0] + 2.0 * diameter * rslope2[:, 0]
                + diameter ** 2 * rslope[:, 0])
    praci_raw = C.PI * qi * C.N0R * jnp.abs(vt2r - vt2i) * acrfac_r / 4.0
    praci_raw = praci_raw * jnp.minimum(jnp.maximum(0.0, qr / qi), 1.0) ** 2
    praci_raw = jnp.minimum(praci_raw, qi / dtcld)
    piacr_raw = (C.PI ** 2 * C.AVTR * C.N0R * C.DENR * xni * denfac * C.G6PBR
                 * rslope3[:, 0] * rslope3[:, 0] * rslopeb[:, 0] / 24.0 / den)
    piacr_raw = piacr_raw * jnp.minimum(jnp.maximum(0.0, qi / qr), 1.0) ** 2
    piacr_raw = jnp.minimum(piacr_raw, qr / dtcld)
    raci_cond = ice_present & (qr > C.QCRMIN)
    praci = jnp.where(raci_cond, praci_raw, 0.0)
    piacr = jnp.where(raci_cond, piacr_raw, 0.0)

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

    # psacw (qs>qcrmin & qc>qmin)
    psacw_raw = jnp.minimum(
        C.PACRC * n0sfac * rslope3[:, 1] * rslopeb[:, 1]
        * jnp.minimum(jnp.maximum(0.0, qs / qc), 1.0) ** 2 * qc * denfac, qc / dtcld)
    psacw = jnp.where((qs > C.QCRMIN) & (qc > qmin), psacw_raw, 0.0)

    # pgacw (qg>qcrmin & qc>qmin)
    pgacw_raw = jnp.minimum(
        C.PACRG * rslope3[:, 2] * rslopeb[:, 2]
        * jnp.minimum(jnp.maximum(0.0, qg / qc), 1.0) ** 2 * qc * denfac, qc / dtcld)
    pgacw = jnp.where((qg > C.QCRMIN) & (qc > qmin), pgacw_raw, 0.0)

    # paacw
    paacw = jnp.where(qsum > 1.0e-15, (qs * psacw + qg * pgacw) / qsum, 0.0)

    # pracs (qs>qcrmin & qr>qcrmin & supcol>0)
    acrfac_pracs = (5.0 * rslope3[:, 1] * rslope3[:, 1] * rslope[:, 0]
                    + 2.0 * rslope3[:, 1] * rslope2[:, 1] * rslope2[:, 0]
                    + 0.5 * rslope2[:, 1] * rslope2[:, 1] * rslope3[:, 0])
    pracs_raw = (C.PI ** 2 * C.N0R * C.N0S * n0sfac * jnp.abs(vt2r - vt2ave)
                 * (C.DENS / den) * acrfac_pracs)
    pracs_raw = pracs_raw * jnp.minimum(jnp.maximum(0.0, qr / qs), 1.0) ** 2
    pracs_raw = jnp.minimum(pracs_raw, qs / dtcld)
    sr_qr_cond = (qs > C.QCRMIN) & (qr > C.QCRMIN)
    pracs = jnp.where(sr_qr_cond & (supcol > 0.0), pracs_raw, 0.0)

    # psacr (qs>qcrmin & qr>qcrmin)
    acrfac_psacr = (5.0 * rslope3[:, 0] * rslope3[:, 0] * rslope[:, 1]
                    + 2.0 * rslope3[:, 0] * rslope2[:, 0] * rslope2[:, 1]
                    + 0.5 * rslope2[:, 0] * rslope2[:, 0] * rslope3[:, 1])
    psacr_raw = (C.PI ** 2 * C.N0R * C.N0S * n0sfac * jnp.abs(vt2ave - vt2r)
                 * (C.DENR / den) * acrfac_psacr)
    psacr_raw = psacr_raw * jnp.minimum(jnp.maximum(0.0, qs / qr), 1.0) ** 2
    psacr_raw = jnp.minimum(psacr_raw, qr / dtcld)
    psacr = jnp.where(sr_qr_cond, psacr_raw, 0.0)

    # pgacr (qg>qcrmin & qr>qcrmin)
    acrfac_pgacr = (5.0 * rslope3[:, 0] * rslope3[:, 0] * rslope[:, 2]
                    + 2.0 * rslope3[:, 0] * rslope2[:, 0] * rslope2[:, 2]
                    + 0.5 * rslope2[:, 0] * rslope2[:, 0] * rslope3[:, 2])
    pgacr_raw = (C.PI ** 2 * C.N0R * C.N0G * jnp.abs(vt2ave - vt2r)
                 * (C.DENR / den) * acrfac_pgacr)
    pgacr_raw = pgacr_raw * jnp.minimum(jnp.maximum(0.0, qg / qr), 1.0) ** 2
    pgacr_raw = jnp.minimum(pgacr_raw, qr / dtcld)
    pgacr = jnp.where((qg > C.QCRMIN) & (qr > C.QCRMIN), pgacr_raw, 0.0)

    # pgacs = 0 (eliminated in V3.0)
    pgacs = jnp.zeros_like(t)

    # enhanced melting (supcol<=0)
    warm2 = supcol <= 0.0
    xlf_e = C.XLF0
    pseml = jnp.where(warm2 & (qs > 0.0),
                      jnp.minimum(jnp.maximum(C.CLIQ * supcol * (paacw + psacr) / xlf_e,
                                              -qs / dtcld), 0.0), 0.0)
    pgeml = jnp.where(warm2 & (qg > 0.0),
                      jnp.minimum(jnp.maximum(C.CLIQ * supcol * (paacw + pgacr) / xlf_e,
                                              -qg / dtcld), 0.0), 0.0)

    # deposition/sublimation block (supcol>0) with sequential ifsat per cell.
    # WRF computes pidep, psdep, pgdep, pigen, psaut, pgaut with an ifsat flag
    # that, once set, zeroes later deposition terms in that cell. We reproduce
    # the exact short-circuit per cell.
    pidep, psdep, pgdep, pigen, psaut, pgaut = _deposition_block(
        cold, qi, qs, qg, q, qmin, diameter, xni, rh2, work1_2, satdt2,
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

    # ---- mass conservation feedback + state update (two branches) ----
    delta2 = jnp.where((qr < 1.0e-4) & (qs < 1.0e-4), 1.0, 0.0)
    delta3 = jnp.where(qr < 1.0e-4, 1.0, 0.0)
    cold_branch = t <= C.T0C

    # COLD branch quantities -------------------------------------------------
    # cloud water
    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + paacw + paacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut * f
    pracw_c = pracw * f
    paacw_c = paacw * f
    # cloud ice
    val = jnp.maximum(qmin, qi)
    src = (psaut - pigen - pidep + praci + psaci + pgaci) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psaut_c = psaut * f
    pigen_c = pigen * f
    pidep_c = pidep * f
    praci_c = praci * f
    psaci_c = psaci * f
    pgaci_c = pgaci * f
    # rain
    val = jnp.maximum(qmin, qr)
    src = (-praut_c - prevp - pracw_c + piacr + psacr + pgacr) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_c = praut_c * f
    prevp_c = prevp * f
    pracw_c = pracw_c * f
    piacr_c = piacr * f
    psacr_c = psacr * f
    pgacr_c = pgacr * f
    # snow
    val = jnp.maximum(qmin, qs)
    src = -(psdep + psaut_c - pgaut + paacw_c + piacr_c * delta3
            + praci_c * delta3 - pracs * (1.0 - delta2)
            + psacr_c * delta2 + psaci_c - pgacs) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    psdep_c = psdep * f
    psaut_c = psaut_c * f
    pgaut_c = pgaut * f
    paacw_c = paacw_c * f
    piacr_c = piacr_c * f
    praci_c = praci_c * f
    psaci_c = psaci_c * f
    pracs_c = pracs * f
    psacr_c = psacr_c * f
    pgacs_c = pgacs * f
    # graupel
    val = jnp.maximum(qmin, qg)
    src = -(pgdep + pgaut_c + piacr_c * (1.0 - delta3) + praci_c * (1.0 - delta3)
            + psacr_c * (1.0 - delta2) + pracs_c * (1.0 - delta2)
            + pgaci_c + paacw_c + pgacr_c + pgacs_c) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgdep_c = pgdep * f
    pgaut_c = pgaut_c * f
    piacr_c = piacr_c * f
    praci_c = praci_c * f
    psacr_c = psacr_c * f
    pracs_c = pracs_c * f
    paacw_c = paacw_c * f
    pgaci_c = pgaci_c * f
    pgacr_c = pgacr_c * f
    pgacs_c = pgacs_c * f

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
    xlf_cold = C.XLS - xl
    xlwork2_cold = (-C.XLS * (psdep_c + pgdep_c + pidep_c + pigen_c)
                    - xl * prevp_c - xlf_cold * (piacr_c + paacw_c + paacw_c
                                                 + pgacr_c + psacr_c))
    t_cold = t - xlwork2_cold / cpm * dtcld

    # WARM branch quantities -------------------------------------------------
    val = jnp.maximum(qmin, qc)
    src = (praut + pracw + paacw + paacw) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut * f
    pracw_w = pracw * f
    paacw_w = paacw * f
    # rain
    val = jnp.maximum(qmin, qr)
    src = (-paacw_w - praut_w + pseml + pgeml - pracw_w - paacw_w - prevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    praut_w = praut_w * f
    prevp_w = prevp * f
    pracw_w = pracw_w * f
    paacw_w = paacw_w * f
    pseml_w = pseml * f
    pgeml_w = pgeml * f
    # snow
    val = jnp.maximum(C.QCRMIN, qs)
    src = (pgacs - pseml_w - psevp) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs * f
    psevp_w = psevp * f
    pseml_w = pseml_w * f
    # graupel
    val = jnp.maximum(C.QCRMIN, qg)
    src = -(pgacs_w + pgevp + pgeml_w) * dtcld
    f = jnp.where(src > val, val / src, 1.0)
    pgacs_w = pgacs_w * f
    pgevp_w = pgevp * f
    pgeml_w = pgeml_w * f

    w2_warm = -(prevp_w + psevp_w + pgevp_w)
    q_warm = q + w2_warm * dtcld
    qc_warm = jnp.maximum(qc - (praut_w + pracw_w + paacw_w + paacw_w) * dtcld, 0.0)
    qr_warm = jnp.maximum(qr + (praut_w + pracw_w + prevp_w + paacw_w + paacw_w
                               - pseml_w - pgeml_w) * dtcld, 0.0)
    qs_warm = jnp.maximum(qs + (psevp_w - pgacs_w + pseml_w) * dtcld, 0.0)
    qg_warm = jnp.maximum(qg + (pgacs_w + pgevp_w + pgeml_w) * dtcld, 0.0)
    xlf_warm = C.XLS - xl
    xlwork2_warm = (-xl * (prevp_w + psevp_w + pgevp_w)
                    - xlf_warm * (pseml_w + pgeml_w))
    t_warm = t - xlwork2_warm / cpm * dtcld

    # select branch per cell
    q = jnp.where(cold_branch, q_cold, q_warm)
    qc = jnp.where(cold_branch, qc_cold, qc_warm)
    qr = jnp.where(cold_branch, qr_cold, qr_warm)
    qi = jnp.where(cold_branch, qi_cold, qi)  # warm branch leaves qi unchanged
    qs = jnp.where(cold_branch, qs_cold, qs_warm)
    qg = jnp.where(cold_branch, qg_cold, qg_warm)
    t = jnp.where(cold_branch, t_cold, t_warm)

    # ---- recompute qsat, pcond condensation ----
    qsat1b, _qsat2b = _qsat_fpvs(t, p)
    w1 = _conden(t, q, qsat1b, xl, cpm)
    pcond = jnp.minimum(jnp.maximum(w1 / dtcld, 0.0), jnp.maximum(q, 0.0) / dtcld)
    pcond = jnp.where((qc > 0.0) & (w1 < 0.0), jnp.maximum(w1, -qc) / dtcld, pcond)
    q = q - pcond * dtcld
    qc = jnp.maximum(qc + pcond * dtcld, 0.0)
    t = t + pcond * xl / cpm * dtcld

    # ---- padding for small values ----
    qc = jnp.where(qc <= qmin, 0.0, qc)
    qi = jnp.where(qi <= qmin, 0.0, qi)

    return (t, q, qc, qr, qi, qs, qg, cpm, xl,
            rainncv, snowncv, graupelncv, tstepsnow, tstepgraup, sr)


def _deposition_block(cold, qi, qs, qg, q, qmin, diameter, xni, rh2, work1_2,
                      satdt2, prevp, dtcld, n0sfac, rslope, rslopeb, rslope2,
                      work2, den, supcol, supsat2):
    """Sequential deposition/sublimation + ice/snow aggregation generation.

    Reproduces the WRF ifsat short-circuit per cell: pidep, psdep, pgdep, and
    pigen are computed in order and a saturation-overshoot flag, once tripped,
    zeroes the remaining deposition terms in THAT cell. psaut/pgaut are
    unaffected by ifsat (only gated by supcol>0).
    """
    zero = jnp.zeros_like(qi)
    satdt = satdt2

    # pidep
    # WRF: pidep=max(max(pidep,satdt/2),supice); then pidep=max(pidep,-qi/dtcld)
    pidep_raw = 4.0 * diameter * xni * (rh2 - 1.0) / work1_2
    supice1 = satdt - prevp
    pidep_neg = jnp.maximum(jnp.maximum(jnp.maximum(pidep_raw, satdt / 2.0), supice1),
                            -qi / dtcld)
    pidep_pos = jnp.minimum(jnp.minimum(pidep_raw, satdt / 2.0), supice1)
    pidep_v = jnp.where(pidep_raw < 0.0, pidep_neg, pidep_pos)
    do_pidep = cold & (qi > 0.0)
    pidep = jnp.where(do_pidep, pidep_v, 0.0)
    ifsat1 = do_pidep & (jnp.abs(prevp + pidep) >= jnp.abs(satdt))

    # psdep (ifsat from pidep blocks it)
    coeres_s = rslope2[:, 1] * jnp.sqrt(rslope[:, 1] * rslopeb[:, 1])
    psdep_raw = ((rh2 - 1.0) * n0sfac * (C.PRECS1 * rslope2[:, 1]
                 + C.PRECS2 * work2 * coeres_s) / work1_2)
    # WRF: psdep=max(psdep,-qs/dtcld); then psdep=max(max(psdep,satdt/2),supice)
    supice2 = satdt - prevp - pidep
    psdep_neg = jnp.maximum(jnp.maximum(jnp.maximum(psdep_raw, -qs / dtcld),
                                        satdt / 2.0), supice2)
    psdep_pos = jnp.minimum(jnp.minimum(psdep_raw, satdt / 2.0), supice2)
    psdep_v = jnp.where(psdep_raw < 0.0, psdep_neg, psdep_pos)
    do_psdep = cold & (qs > 0.0) & jnp.logical_not(ifsat1)
    psdep = jnp.where(do_psdep, psdep_v, 0.0)
    ifsat2 = ifsat1 | (do_psdep & (jnp.abs(prevp + pidep + psdep) >= jnp.abs(satdt)))

    # pgdep
    coeres_g = rslope2[:, 2] * jnp.sqrt(rslope[:, 2] * rslopeb[:, 2])
    pgdep_raw = ((rh2 - 1.0) * (C.PRECG1 * rslope2[:, 2]
                 + C.PRECG2 * work2 * coeres_g) / work1_2)
    supice3 = satdt - prevp - pidep - psdep
    pgdep_neg = jnp.maximum(jnp.maximum(jnp.maximum(pgdep_raw, -qg / dtcld),
                                        satdt / 2.0), supice3)
    pgdep_pos = jnp.minimum(jnp.minimum(pgdep_raw, satdt / 2.0), supice3)
    pgdep_v = jnp.where(pgdep_raw < 0.0, pgdep_neg, pgdep_pos)
    do_pgdep = cold & (qg > 0.0) & jnp.logical_not(ifsat2)
    pgdep = jnp.where(do_pgdep, pgdep_v, 0.0)
    ifsat3 = ifsat2 | (do_pgdep & (jnp.abs(prevp + pidep + psdep + pgdep) >= jnp.abs(satdt)))

    # pigen (supsat>0)
    supice4 = satdt - prevp - pidep - psdep - pgdep
    xni0 = 1.0e3 * jnp.exp(0.1 * supcol)
    roqi0 = 4.92e-11 * xni0 ** 1.33
    pigen_raw = jnp.maximum(0.0, (roqi0 / den - jnp.maximum(qi, 0.0)) / dtcld)
    pigen_v = jnp.minimum(jnp.minimum(pigen_raw, satdt), supice4)
    do_pigen = cold & (supsat2 > 0.0) & jnp.logical_not(ifsat3)
    pigen = jnp.where(do_pigen, pigen_v, 0.0)

    # psaut: I->S (supcol>0, qi>0)
    qimax = C.ROQIMAX / den
    psaut_v = jnp.maximum(0.0, (qi - qimax) / dtcld)
    psaut = jnp.where(cold & (qi > 0.0), psaut_v, 0.0)

    # pgaut: S->G (supcol>0, qs>0)
    alpha2 = 1.0e-3 * jnp.exp(0.09 * (-supcol))
    pgaut_v = jnp.minimum(jnp.maximum(0.0, alpha2 * (qs - C.QS0)), qs / dtcld)
    pgaut = jnp.where(cold & (qs > 0.0), pgaut_v, 0.0)

    return pidep, psdep, pgdep, pigen, psaut, pgaut


# vmap the single-column scheme across the column (i) dimension.
_wsm6_columns = jax.jit(
    jax.vmap(_wsm6_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)),
    static_argnums=(10,))


# --------------------------------------------------------------------------
# Public adapter
# --------------------------------------------------------------------------
def wsm6_run(t, qv, qc, qr, qi, qs, qg, den, p, delz, delt):
    """Run WSM6 on a batch of columns (shape (ncol, nlev)).

    Inputs are TEMPERATURE-based (t in K), matching the WRF mp_wsm6_run
    internal state. Returns a dict with updated t, qv, qc, qr, qi, qs, qg,
    surface precip increments (mm), sr, and effective radii.
    """
    return _wsm6_columns(t, qv, qc, qr, qi, qs, qg, den, p, delz, float(delt))


def wsm6_physics_tendency(theta, qv, qc, qr, qi, qs, qg, pii, den, p, delz, delt):
    """WSM6 adapter returning a frozen PhysicsTendency (in-place replacements).

    ``theta`` and the moist species are (ncol, nlev). ``pii`` is the Exner
    function used to convert theta<->t (as the WRF ``wsm6`` wrapper does).
    Surface precip increments are per-call mm.
    """
    t = theta * pii
    out = wsm6_run(t, qv, qc, qr, qi, qs, qg, den, p, delz, delt)
    theta_new = out["t"] / pii

    tend = PhysicsTendency(
        state_replacements={
            "theta": theta_new,
            "qv": out["qv"],
            "qc": out["qc"],
            "qr": out["qr"],
            "qi": out["qi"],
            "qs": out["qs"],
            "qg": out["qg"],
        },
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["snowncv"],
            "graupel_acc": out["graupelncv"],
        },
        diagnostics={
            "re_cloud": out["re_cloud"],
            "re_ice": out["re_ice"],
            "re_snow": out["re_snow"],
            "sr": out["sr"],
        },
    )
    return tend


__all__ = ["wsm6_run", "wsm6_physics_tendency"]
