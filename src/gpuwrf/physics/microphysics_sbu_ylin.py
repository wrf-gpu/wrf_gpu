"""JAX SBU-YLin single-moment 5-class microphysics (WRF mp_physics=13).

Faithful column port of ``phys/module_mp_sbu_ylin.F`` -- the Stony-Brook-
University variant of Lin et al. (1983) with Y. Lin's ice-Richardson-dependent
snow size distribution (Heymsfield 2007 fall-speed), Liu & Daum (2004)
autoconversion, and Bigg rain freezing. Active moist leaves are
``qv,qc,qr,qi,qs`` (the WRF moist members for the ``sbu_ylinscheme`` package);
the scheme also returns the diagnostic vertical ice-Richardson profile ``Ri``
(WRF ``rimi``/``Ri3D``), which is a pure OUTPUT (computed each step, never read
back as memory), so SBU-YLin needs no extra prognostic State leaf.

Structure mirrors the Fortran ``clphy1d_ylin`` column routine exactly:

1. saturation mixing ratios + viscosity/diffusivity/conductivity coefficients;
2. ice-Richardson ``Ri`` and the Ri-dependent snow ``am_s/bm_s/av_s/bv_s``;
3. data-dependent semi-Lagrangian sedimentation of rain/snow/ice via an
   adaptive Courant sub-step (``jax.lax.while_loop`` per species, one fixed
   maximum trip count so it stays jit/vmap-traceable);
4. per-level mixed-phase process rates (autoconversion, accretion, Bergeron,
   deposition/sublimation, melting/evaporation, Bigg freezing) with WRF's
   sequential depletion limiters;
5. Newton saturation adjustment (``satadj``, fixed 20-iteration sweep) and the
   ice<->water melting/freezing closure.

No masking clamps beyond the WRF source's own ``amax1(0.,.)`` budget guards, no
self-compare: the reference is the Fortran scheme (proofs/v017 savepoints).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import sbu_ylin_constants as C

_F64 = jnp.float64

# Maximum adaptive sedimentation sub-steps. The Fortran ``DO while(notlast)``
# loop accumulates Courant-limited sub-steps until ``t_del_tv >= dtb``; with the
# 0.9*dz/vt safety factor each sub-step advances >=~ one Courant time, so the
# trip count is bounded by the column Courant number. 256 covers a 90 s macro
# step over the fastest rain fall speeds on the thinnest WRF layers with margin;
# the loop predicate stops early (subsequent trips are no-ops) so a generous cap
# costs nothing at runtime once converged.
_MAX_SED_SUBSTEPS = 256


def _parama1(temp):
    """Bergeron crystal-growth coefficient a1 (linear table interp), WRF-faithful.

    Fortran: ``i1=int(-temp)+1; ratio=-temp-(i1-1)``. ``temp`` reaching this is
    clamped by the caller (``temc1=amax1(-30.99,temcc)`` for psfw; pidw guards
    ``-31<temcc<0``), so indices land in [0, 31]; clamp defensively to keep the
    gather in bounds without altering in-range results.
    """

    a1 = jnp.asarray(C.PARAMA1_TABLE, dtype=temp.dtype)
    i1f = jnp.floor(-temp) + 1.0
    ratio = -temp - (i1f - 1.0)
    i1 = jnp.clip(i1f.astype(jnp.int32) - 1, 0, 30)
    return a1[i1] + ratio * (a1[i1 + 1] - a1[i1])


def _parama2(temp):
    """Bergeron crystal-growth coefficient a2 (linear table interp), WRF-faithful."""

    a2 = jnp.asarray(C.PARAMA2_TABLE, dtype=temp.dtype)
    i1f = jnp.floor(-temp) + 1.0
    ratio = -temp - (i1f - 1.0)
    i1 = jnp.clip(i1f.astype(jnp.int32) - 1, 0, 30)
    return a2[i1] + ratio * (a2[i1 + 1] - a2[i1])


def _ggamma_arr(x):
    """Vectorized Hastings gamma -- EXACTLY ``module_mp_sbu_ylin.F:ggamma``.

    Snow processes call ``ggamma`` on level-dependent arguments (``tmp_ss``,
    ``bv_s+tmp_ss`` etc.), so the gamma must be evaluated per level at trace
    time. Reproduces the Fortran reduction-to-(1,2] + 8-term Hastings series.
    The reduction-by-1 loop is data-dependent in Fortran (``DO J=1,200``); the
    arguments here are bounded (PSD shape factors ~ a few), so a fixed sweep of
    16 reductions covers them with the same arithmetic.
    """

    b = (
        -0.577191652, 0.988205891, -0.897056937, 0.918206857,
        -0.756704078, 0.482199394, -0.193527818, 0.035868343,
    )
    x = jnp.asarray(x)
    pf = jnp.ones_like(x)
    temp = x
    for _ in range(16):
        do_red = temp > 2.0
        temp_next = jnp.where(do_red, temp - 1.0, temp)
        pf = jnp.where(do_red, pf * temp_next, pf)
        temp = temp_next
    temp = temp - 1.0
    g1to2 = jnp.ones_like(x)
    for k1 in range(1, 9):
        g1to2 = g1to2 + b[k1 - 1] * temp ** k1
    return pf * g1to2


def _sediment_species(q, vt_coeff_fn, rho, dzw, zz, zsfc, dtb):
    """Adaptive semi-Lagrangian fall of one precipitating species.

    Faithful translation of the WRF ``DO while(notlast)`` block: each sub-step
    computes a per-level fall speed ``vt`` (via ``vt_coeff_fn(q)``, zero where
    ``q<=1e-8``), takes the global Courant-limited ``del_tv = min(del_tv,
    0.9*dz/vt)`` over active levels, then applies the explicit upwind flux
    sweep ``q += del_tv*(fluxin-fluxout)/rho/dzw`` from the top active level
    down, accumulating surface precip ``ppt`` from the lowest level's flux. The
    ``min_q-1`` spill into the level below the lowest active one is folded into
    the same vectorized flux sweep (it is exactly ``fluxin/rho/dzw`` at that
    level, which the upwind sweep already produces).

    Returns ``(q_out, ppt)`` with ``ppt`` in metres (WRF units, *1000 -> mm by
    the caller's accumulator).
    """

    nz = q.shape[0]
    # k index ascending from surface (k=0 lowest), matching Fortran 1-based kts.
    dz_below = jnp.concatenate([
        jnp.array([zz[0] - zsfc], dtype=q.dtype),
        zz[1:] - zz[:-1],
    ])  # 0.9*(zz(k)-zz(k-1)); k=1 uses (zz(1)-zsfc)

    def cond(state):
        _q, _ppt, t_del_tv, _del_tv, notlast = state
        return notlast

    def body(state):
        q_, ppt, t_del_tv, del_tv, _notlast = state
        active = q_ > 1.0e-8                      # qrz(k) > 1.0e-8
        vt = jnp.where(active, vt_coeff_fn(q_), 0.0)
        # WRF only considers k=kts..kte-1 for the Courant limit and the sweep
        # (the top level kte never falls in-place). Mask the top level out.
        top_mask = jnp.arange(nz) < (nz - 1)
        active = active & top_mask
        any_active = jnp.any(active)
        # del_tv starts each MACRO call at dtb; within the while loop it is the
        # running min. Courant limit over active levels.
        safe_vt = jnp.where(active, vt, 1.0)
        courant = 0.9 * dz_below / safe_vt
        del_tv_new = jnp.minimum(del_tv, jnp.min(jnp.where(active, courant, jnp.inf)))
        del_tv_new = jnp.where(any_active, del_tv_new, del_tv)

        t_acc = t_del_tv + del_tv_new
        # WRF: t_del_tv += del_tv (-> t_acc); if t_del_tv >= dtb then
        #      del_tv = dtb + del_tv - t_del_tv. Since that t_del_tv is the
        #      POST-increment value (t_acc), the per-step del_tv cancels and the
        #      finishing sub-step uses exactly the remaining time dtb - t_del_tv
        #      (the value BEFORE this step's increment).
        finishing = t_acc >= dtb
        del_tv_eff = jnp.where(finishing, dtb - t_del_tv, del_tv_new)

        # Upwind flux sweep from top active level down. fluxout(k)=rho*vt*q;
        # flux(k)=(fluxin-fluxout)/rho/dzw; fluxin for level k is fluxout(k+1).
        fluxout = rho * vt * q_                    # 0 where inactive
        fluxin = jnp.concatenate([fluxout[1:], jnp.array([0.0], dtype=q_.dtype)])
        flux = (fluxin - fluxout) / rho / dzw
        q_upd = q_ + del_tv_eff * flux
        q_upd = jnp.maximum(q_upd, 0.0)            # WRF clamps qsz/qiz>=0
        q_new = jnp.where(any_active, q_upd, q_)

        # Surface precip (WRF: ``pptrain += fluxin*del_tv`` when min_q==1, where
        # at the loop's end ``fluxin == fluxout(min_q)`` -- the mass flux leaving
        # the LOWEST active level downward, computed with that level's pre-update
        # q). For the lowest model level k=0 that is ``fluxout[0]``; the upwind
        # sweep keeps fluxin[0] (mass entering from level 1) in level 0 and lets
        # fluxout[0] exit the domain as precip. When min_q>1, WRF spills
        # fluxout(min_q) into level min_q-1 instead -- the vectorized sweep
        # already deposits it there (fluxout is 0 at the empty level below), so
        # only the k=0 outflow is caught here. ppt is in kg/m^2 == mm.
        surf_active = active[0]
        ppt_inc = jnp.where(surf_active & any_active, fluxout[0] * del_tv_eff, 0.0)
        ppt_new = ppt + ppt_inc

        notlast_new = any_active & jnp.logical_not(finishing)
        return (q_new, ppt_new, t_acc, del_tv_new, notlast_new)

    # Bound the data-dependent loop with a fixed cap so it is jit/vmap-traceable.
    def capped_body(i, state):
        return jax.lax.cond(state[4], body, lambda s: s, state)

    init = (q, jnp.zeros((), q.dtype), jnp.zeros((), q.dtype),
            jnp.asarray(dtb, q.dtype), jnp.asarray(True))
    q_out, ppt, _, _, _ = jax.lax.fori_loop(0, _MAX_SED_SUBSTEPS, capped_body, init)
    return q_out, ppt


def _satadj(qvz, qlz, qiz, prez, theiz, tothz):
    """Newton saturation adjustment, vectorized over levels (WRF ``satadj``).

    Mirrors ``module_mp_sbu_ylin.F:satadj``: split condensate into liquid/ice
    fractions (mass-weighted, or a temperature ramp when trace), then a fixed
    20-iteration Newton solve for the saturation temperature with the early-exit
    ``absft<0.01`` replicated as a freeze-once-converged guard. Returns updated
    ``(qvz, qlz, qiz)``.
    """

    thz = theiz - (C.XLV / C.CP * qvz - C.XLF / C.CP * qiz) / tothz
    tem = tothz * thz

    es_w = 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * (tem - C.SVPT0) / (tem - C.SVP3))
    qsat_w = C.EP2 * es_w / (prez - es_w)
    qsat_i = C.RH * C.EP2 * 1000.0 * C.SVP1 / prez * jnp.exp(
        21.8745584 * (tem - 273.15) / (tem - 7.66)
    )
    qsat = jnp.where(tem > 273.15, qsat_w, qsat_i)

    qpz = qvz + qlz + qiz
    # Early unsaturated branch (qpz < qsat): all condensate -> vapor.
    unsat = qpz < qsat

    qlpqi = qlz + qiz
    ratql_mass = qlz / jnp.where(qlpqi > 0.0, qlpqi, 1.0)
    ratqi_mass = qiz / jnp.where(qlpqi > 0.0, qlpqi, 1.0)
    t0, t1 = 273.15, 248.15
    tmp1 = jnp.clip((t0 - tem) / (t0 - t1), 0.0, 1.0)
    ratqi = jnp.where(qlpqi >= 1.0e-5, ratqi_mass, tmp1)
    ratql = jnp.where(qlpqi >= 1.0e-5, ratql_mass, 1.0 - tmp1)

    def newton(state):
        tsat, absft = state
        denom1 = 1.0 / (tsat - C.SVP3)
        denom2 = 1.0 / (tsat - 7.66)
        es = 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * denom1 * (tsat - C.SVPT0))
        qswz = C.EP2 * es / (prez - es)
        es_i = 1000.0 * C.SVP1 * jnp.exp(21.8745584 * denom2 * (tsat - 273.15))
        qsiz_cold = C.EP2 * es_i / (prez - es_i)
        cold = tem < 273.15
        qsiz = jnp.where(cold, qsiz_cold, qswz)
        qswz = jnp.where(cold & (tem < 233.15), qsiz, qswz)
        qvsbar = ratql * qswz + ratqi * qsiz

        dqvsbar = (ratql * qswz * C.SVP2 * 243.5 * denom1 * denom1
                   + ratqi * qsiz * 21.8745584 * 265.5 * denom2 * denom2)
        ftsat = (tsat + (C.XLV / C.CP + ratqi * C.XLF / C.CP) * qvsbar
                 - tothz * theiz - C.XLF / C.CP * ratqi * (qvz + qlz + qiz))
        dftsat = 1.0 + (C.XLV / C.CP + ratqi * C.XLF / C.CP) * dqvsbar
        # WRF exits when absft<0.01: freeze tsat once converged (idempotent).
        converged = absft < 0.01
        tsat_new = jnp.where(converged, tsat, tsat - ftsat / dftsat)
        absft_new = jnp.where(converged, absft, jnp.abs(ftsat))
        return (tsat_new, absft_new), qvsbar

    tsat0 = tem
    state = (tsat0, jnp.full_like(tsat0, 1.0))
    qvsbar_last = jnp.zeros_like(tem)
    for _ in range(20):
        state, qvsbar_last = newton(state)

    qvsbar = qvsbar_last
    sat_more = qpz > qvsbar
    qvz_sat = jnp.where(sat_more, qvsbar, qpz)
    qiz_sat = jnp.where(sat_more, ratqi * (qpz - qvsbar), 0.0)
    qlz_sat = jnp.where(sat_more, ratql * (qpz - qvsbar), 0.0)

    qvz_out = jnp.where(unsat, qpz, qvz_sat)
    qiz_out = jnp.where(unsat, 0.0, qiz_sat)
    qlz_out = jnp.where(unsat, 0.0, qlz_sat)
    return qvz_out, qlz_out, qiz_out


def _clphy1d_ylin(thz, qvz, qlz, qrz, qiz, qsz, tothz, rho, prez, zz, dzw, zsfc, dtb):
    """One-column SBU-YLin microphysics. All inputs/outputs shape ``(nz,)``."""

    odtb = 1.0 / dtb
    orho = 1.0 / rho
    sqrho = jnp.ones_like(rho)  # WRF disables density-dependent fall speed (=1)
    oprez = 1.0 / prez

    xLvocp = C.XLV / C.CP
    xLfocp = C.XLF / C.CP
    ocp = 1.0 / C.CP
    oxLf = 1.0 / C.XLF
    cwoxlf = C.CW / C.XLF
    oxmi = 1.0 / C.XMI

    bp3 = C.BV_R + 3.0
    bp5 = C.BV_R + 5.0
    bp6 = C.BV_R + 6.0

    # --- clamp + saturation mixing ratios (WRF do k loop) ---------------------
    qlz = jnp.maximum(qlz, 0.0)
    qiz = jnp.maximum(qiz, 0.0)
    qvz = jnp.maximum(qvz, C.QVMIN)
    qsz = jnp.maximum(qsz, 0.0)
    qrz = jnp.maximum(qrz, 0.0)

    tem = thz * tothz
    temcc = tem - 273.15
    es_w = 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * temcc / (tem - C.SVP3))
    qswz = C.EP2 * es_w / (prez - es_w)
    es_i = 1000.0 * C.SVP1 * jnp.exp(21.8745584 * (tem - 273.16) / (tem - 7.66))
    qsiz_cold = C.EP2 * es_i / (prez - es_i)
    cold0 = tem < 273.15
    qsiz = jnp.where(cold0, qsiz_cold, qswz)
    qswz = jnp.where(cold0 & (temcc < -40.0), qsiz, qswz)

    qvoqswz = qvz / qswz
    qvoqsiz = qvz / qsiz
    theiz = thz + (xLvocp * qvz - xLfocp * qiz) / tothz

    # --- transport coefficients ----------------------------------------------
    viscmu = C.AVISC * tem ** 1.5 / (tem + 120.0)
    visc = viscmu * orho
    diffwv = C.ADIFFWV * tem ** 1.81 * oprez
    schmidt = visc / diffwv
    xka = C.AXKA * viscmu
    rs0 = C.EP2 * 1000.0 * C.SVP1 / (prez - 1000.0 * C.SVP1)

    # --- ice-Richardson Ri + Ri-dependent snow PSD (Y. Lin) -------------------
    have_both = (rho * qlz > 1e-5) & (rho * qsz > 1e-5)
    ri_raw = 1.0 / (1.0 + 6e-5 / (rho ** 1.170 * jnp.where(have_both, qlz, 1.0)
                                  * jnp.where(have_both, qsz, 1.0) ** 0.170))
    Ri = jnp.where(have_both, ri_raw, 0.0)
    # make Ri non-decreasing downward: Ri(kts..max_ri_k) = max(Ri).
    nz = thz.shape[0]
    max_ri = jnp.max(Ri)
    max_ri_k = jnp.argmax(Ri)  # 0-based index of (first) max
    idx = jnp.arange(nz)
    Ri = jnp.where(idx <= max_ri_k, max_ri, Ri)
    Ri = jnp.clip(Ri, 0.0, 1.0)
    Riz = Ri  # returned diagnostic (WRF Ri3D)

    cap_s = 0.25 * (1.0 + Ri)
    tc0 = jnp.minimum(-0.1, tem - 273.15)
    N0_s = jnp.minimum(2.0e8, 2.0e6 * jnp.exp(-0.12 * tc0))
    am_s = C.AM_C1 + C.AM_C2 * tc0 + C.AM_C3 * Ri * Ri
    am_s = jnp.maximum(0.000023, am_s)
    bm_s = C.BM_C1 + C.BM_C2 * tc0 + C.BM_C3 * Ri
    bm_s = jnp.minimum(bm_s, 3.0)
    am_s = 10.0 ** (2.0 * bm_s - 3.0) * am_s
    aa_s = C.AA_C1 + C.AA_C2 * tc0 + C.AA_C3 * Ri
    ba_s = C.BA_C1 + C.BA_C2 * tc0 + C.BA_C3 * Ri
    aa_s = (1e-2) ** (2.0 - ba_s) * aa_s
    av_s = C.BEST_A * viscmu * (2.0 * C.GRAV * am_s / rho / aa_s / (viscmu ** 2)) ** C.BEST_B
    bv_s = C.BEST_B * (bm_s - ba_s + 2.0) - 1.0
    tmp_ss = bm_s + C.MU_S + 1.0
    tmp_sa = ba_s + C.MU_S + 1.0

    # --- semi-Lagrangian sedimentation (rain, snow, ice) ----------------------
    def vt_rain(q):
        tmp1 = jnp.sqrt(jnp.sqrt(C.PI * C.RHOWATER * C.XNOR / rho / jnp.maximum(q, 1e-30)))
        return C.O6_AVR_GAMBP4 * sqrho / tmp1 ** C.BV_R

    def vt_snow(q):
        tmp1 = (am_s * N0_s * _ggamma_arr(tmp_ss) * orho / jnp.maximum(q, 1e-30)) ** (1.0 / tmp_ss)
        return sqrho * av_s * _ggamma_arr(bv_s + tmp_ss) / _ggamma_arr(tmp_ss) / (tmp1 ** bv_s)

    def vt_ice(q):
        return 3.29 * (rho * jnp.maximum(q, 0.0)) ** 0.16  # Heymsfield-Donner

    qrz, pptrain = _sediment_species(qrz, vt_rain, rho, dzw, zz, zsfc, dtb)
    qsz, pptsnow = _sediment_species(qsz, vt_snow, rho, dzw, zz, zsfc, dtb)
    qiz, pptice = _sediment_species(qiz, vt_ice, rho, dzw, zz, zsfc, dtb)

    # --- per-level process rates (DO 2000 k loop) -----------------------------
    qvzodt = jnp.maximum(0.0, odtb * qvz)
    qlzodt = jnp.maximum(0.0, odtb * qlz)
    qizodt = jnp.maximum(0.0, odtb * qiz)
    qszodt = jnp.maximum(0.0, odtb * qsz)
    qrzodt = jnp.maximum(0.0, odtb * qrz)

    # skip mask: unsaturated AND no condensate -> go to 2000 (all rates stay 0).
    cond_sum = qiz + qlz + qsz + qrz
    skip = (qvz + qlz + qiz < qsiz) & (cond_sum == 0.0)
    active = jnp.logical_not(skip)

    # rain slope / fall speed (only where qrz>1e-8)
    have_r = qrz > 1.0e-8
    tmp1_r = jnp.sqrt(jnp.sqrt(C.PI * C.RHOWATER * C.XNOR * orho / jnp.where(have_r, qrz, 1.0)))
    xlambdar = jnp.where(have_r, tmp1_r, 0.0)
    olambdar = jnp.where(have_r, 1.0 / jnp.where(have_r, xlambdar, 1.0), 0.0)
    vtr = jnp.where(have_r, C.O6_AVR_GAMBP4 * sqrho * olambdar ** C.BV_R, 0.0)

    # snow slope / fall speed (only where qsz>1e-8)
    have_s = qsz > 1.0e-8
    gsss = _ggamma_arr(tmp_ss)
    tmp1_s = (am_s * N0_s * gsss * orho / jnp.where(have_s, qsz, 1.0)) ** (1.0 / tmp_ss)
    olambdas = jnp.where(have_s, 1.0 / jnp.where(have_s, tmp1_s, 1.0), 0.0)
    vts = jnp.where(
        have_s,
        sqrho * av_s * _ggamma_arr(bv_s + tmp_ss) / gsss / (jnp.where(have_s, tmp1_s, 1.0) ** bv_s),
        0.0,
    )

    cold = tem < 273.15

    # ---- COLD branch (T < 0 C) process rates ----
    alpha1 = 1.0e-3 * jnp.exp(0.025 * temcc)
    qic_lt = 1.0e-3 * jnp.exp(-7.6 + 4.0 * jnp.exp(
        -0.2443e-3 * (jnp.abs(temcc) - 20.0) ** 2.455)) * orho
    qic = jnp.where(temcc < -20.0, qic_lt, C.QI0)
    psaut = jnp.maximum(0.0, odtb * (qiz - qic) * (1.0 - jnp.exp(-alpha1 * dtb)))

    # Bergeron Psfw/Psfi (only where qlz>1e-10)
    have_ql10 = qlz > 1.0e-10
    temc1 = jnp.maximum(-30.99, temcc)
    a1b = _parama1(temc1)
    a2b = _parama2(temc1)
    tmpp1 = 1.0 - a2b
    a1b = a1b * 0.001 ** tmpp1
    odtberg = (a1b * tmpp1) / (C.XMI50 ** tmpp1 - C.XMI40 ** tmpp1)
    vti50 = C.AV_I * C.DI50 ** C.BV_I * sqrho
    eiw = 1.0
    save1_b = a1b * C.XMI50 ** a2b
    save2_b = 0.25 * C.PI * eiw * rho * C.DI50 * C.DI50 * vti50
    tmp2_b = save1_b + save2_b * qlz
    xni50mx = qlzodt / tmp2_b
    xni50 = qiz * (1.0 - jnp.exp(-dtb * odtberg)) / C.XMI50
    xni50 = jnp.minimum(xni50, xni50mx)
    tmp3_b = odtb * tmp2_b / save2_b * (1.0 - jnp.exp(-save2_b * xni50 * dtb))
    psfw = jnp.where(have_ql10, jnp.minimum(tmp3_b, qlzodt), 0.0)
    psfi = jnp.where(have_ql10, jnp.minimum(xni50 * C.XMI50 - psfw, qizodt), 0.0)

    # Praci/Piacr (need qrz>0)
    have_qr = qrz > 0.0
    eri = 1.0
    save1_r = C.PI / 4.0 * eri * C.XNOR * C.AV_R * sqrho
    tmp1_pr = save1_r * C.GAMBP3 * olambdar ** bp3
    praci = jnp.where(have_qr, qizodt * (1.0 - jnp.exp(-tmp1_pr * dtb)), 0.0)
    tmp2_pi = qiz * save1_r * rho * C.PI / 6.0 * C.RHOWATER * C.GAMBP6 * oxmi * olambdar ** bp6
    piacr = jnp.where(have_qr, jnp.minimum(tmp2_pi, qrzodt), 0.0)

    # Psaci/Psacw/Psdep/Pssub (need qsz>0)
    have_qs = qsz > 0.0
    esi = jnp.exp(0.025 * temcc)
    save1_s = aa_s * sqrho * N0_s * _ggamma_arr(bv_s + tmp_sa) * olambdas ** (bv_s + tmp_sa)
    psaci = jnp.where(have_qs, qizodt * (1.0 - jnp.exp(-esi * save1_s * dtb)), 0.0)
    psacw_cold = jnp.where(have_qs, qlzodt * (1.0 - jnp.exp(-save1_s * dtb)), 0.0)
    tmpa_d = C.RVAPOR * xka * tem * tem
    tmpb_d = C.XLS * C.XLS * rho * qsiz * diffwv
    tmpc_d = tmpa_d * qsiz * diffwv
    abi = 4.0 * C.PI * cap_s * (qvoqsiz - 1.0) * tmpc_d / (tmpa_d + tmpb_d)
    tmp1_sd = av_s * sqrho * olambdas ** (5 + bv_s + 2 * C.MU_S) / visc
    tmp2_sd = abi * N0_s * (C.VF1S * olambdas * olambdas
                            + C.VF2S * schmidt ** 0.33334 * _ggamma_arr(2.5 + 0.5 * bv_s + C.MU_S)
                            * jnp.sqrt(jnp.maximum(tmp1_sd, 0.0)))
    tmp3_sd = odtb * (qvz - qsiz)
    pssub_neg = jnp.maximum(jnp.maximum(tmp2_sd, tmp3_sd), -qszodt)
    psdep_pos = jnp.minimum(tmp2_sd, tmp3_sd)
    pssub = jnp.where(have_qs & (tmp2_sd <= 0.0), pssub_neg, 0.0)
    psdep = jnp.where(have_qs & (tmp2_sd > 0.0), psdep_pos, 0.0)

    # Pracs/Psacr (need qsz>0 and qrz>0); Pracs forced 0 in WRF.
    have_sr = have_qs & have_qr
    tmpa_x = olambdar * olambdar
    tmpb_x = olambdas * olambdas
    tmpc_x = olambdar * olambdas
    tmp1_x = C.PI * C.PI * 1.0 * C.XNOR * N0_s * jnp.abs(vtr - vts) * orho
    tmp3_psacr = tmpa_x * tmpa_x * olambdas * (5.0 * tmpa_x + 2.0 * tmpc_x + 0.5 * tmpb_x)
    psacr_cold = jnp.where(have_sr, jnp.minimum(tmp1_x * C.RHOWATER * tmp3_psacr, qrzodt), 0.0)
    pracs_cold = jnp.zeros_like(thz)

    # Bigg freezing Pgfr (need qrz>1e-8)
    have_r8 = qrz > 1.0e-8
    Bp, Ap = 100.0, 0.66
    tmp1_g = olambdar * olambdar * olambdar
    tmp2_g = 20.0 * C.PI * C.PI * Bp * C.XNOR * C.RHOWATER * orho * \
        (jnp.exp(-Ap * temcc) - 1.0) * tmp1_g * tmp1_g * olambdar
    pgfr = jnp.where(have_r8, jnp.minimum(tmp2_g, qrzodt), 0.0)

    # ---- WARM branch (T >= 0 C) snow processes ----
    save1_sw = aa_s * sqrho * N0_s * _ggamma_arr(bv_s + tmp_sa) * olambdas ** (bv_s + tmp_sa)
    psacw_warm = jnp.where(have_qs, qlzodt * (1.0 - jnp.exp(-save1_sw * dtb)), 0.0)
    tmp2_psacrw = tmpa_x * tmpa_x * olambdas * (5.0 * tmpa_x + 2.0 * tmpc_x + 0.5 * tmpb_x)
    psacr_warm = jnp.where(have_qs, jnp.minimum(tmp1_x * C.RHOWATER * tmp2_psacrw, qrzodt), 0.0)
    delrs = rs0 - qvz
    term1_m = 2.0 * C.PI * orho * (C.XLV * diffwv * rho * delrs - xka * temcc)
    tmp1_m = av_s * sqrho * olambdas ** (5 + bv_s + 2 * C.MU_S) / visc
    tmp2_m = N0_s * (C.VF1S * olambdas * olambdas
                     + C.VF2S * schmidt ** 0.33334 * _ggamma_arr(2.5 + 0.5 * bv_s + C.MU_S)
                     * jnp.sqrt(jnp.maximum(tmp1_m, 0.0)))
    tmp3_m = term1_m * oxLf * tmp2_m - cwoxlf * temcc * (psacw_warm + psacr_warm)
    psmlt = jnp.where(have_qs, jnp.maximum(jnp.minimum(0.0, tmp3_m), -qszodt), 0.0)
    tmpa_e = C.RVAPOR * xka * tem * tem
    tmpb_e = C.XLV * C.XLV * rho * qswz * diffwv
    tmpc_e = tmpa_e * qswz * diffwv
    tmpd_e = jnp.minimum(0.0, (qvoqswz - 0.90) * qswz * odtb)
    tmp1_se = av_s * sqrho * olambdas ** (5 + bv_s + 2 * C.MU_S) / visc
    tmp2_se = N0_s * (C.VF1S * olambdas * olambdas
                      + C.VF2S * schmidt ** 0.33334 * _ggamma_arr(2.5 + 0.5 * bv_s + C.MU_S)
                      * jnp.sqrt(jnp.maximum(tmp1_se, 0.0)))
    tmp3_se = jnp.maximum(jnp.minimum(0.0, tmp2_se), tmpd_e)
    psmltevp = jnp.where(have_qs, jnp.maximum(tmp3_se, -qszodt), 0.0)

    # select cold vs warm snow rates
    psacw = jnp.where(cold, psacw_cold, psacw_warm)
    psacr = jnp.where(cold, psacr_cold, psacr_warm)
    psmlt = jnp.where(cold, 0.0, psmlt)
    psmltevp = jnp.where(cold, 0.0, psmltevp)
    pracs = jnp.where(cold, pracs_cold, 0.0)
    # cold-only rates forced 0 in warm
    psaut = jnp.where(cold, psaut, 0.0)
    psfw = jnp.where(cold, psfw, 0.0)
    psfi = jnp.where(cold, psfi, 0.0)
    praci = jnp.where(cold, praci, 0.0)
    piacr = jnp.where(cold, piacr, 0.0)
    psaci = jnp.where(cold, psaci, 0.0)
    psdep = jnp.where(cold, psdep, 0.0)
    pssub = jnp.where(cold, pssub, 0.0)
    pgfr = jnp.where(cold, pgfr, 0.0)

    # ---- rain processes (both branches) ----
    # Liu & Daum autoconversion (need qlz>1e-6)
    have_ql6 = qlz > 1.0e-6
    lamc = (C.NT_C * C.RHOWATER * C.PI * C.GGAMMA_4PMUC
            / (6.0 * rho * jnp.where(have_ql6, qlz, 1.0)) / C.GGAMMA_1PMUC) ** 0.3333
    Dc_liu = (C.GGAMMA_7PMUC / C.GGAMMA_1PMUC) ** (1.0 / 6.0) / lamc
    disp = 1.0 / (C.MU_C + 1.0)
    eta = (0.75 / C.PI / (1e-3 * C.RHOWATER)) ** 2 * 1.9e11 * (
        (1.0 + 3.0 * disp) * (1.0 + 4.0 * disp) * (1.0 + 5.0 * disp)
        / (1.0 + disp) / (1.0 + 2.0 * disp))
    praut_raw = eta * (1e-3 * rho * qlz) ** 3 / (1e-6 * C.NT_C) / (1e-3 * rho)
    praut = jnp.where(have_ql6 & (Dc_liu > C.R6C), praut_raw, 0.0)

    # Pracw
    erw = 1.0
    tmp1_pw = C.PI / 4.0 * erw * C.XNOR * C.AV_R * sqrho * C.GAMBP3 * olambdar ** bp3
    pracw = qlzodt * (1.0 - jnp.exp(-tmp1_pw * dtb))

    # Prevp
    tmpa_p = C.RVAPOR * xka * tem * tem
    tmpb_p = C.XLV * C.XLV * rho * qswz * diffwv
    tmpc_p = tmpa_p * qswz * diffwv
    tmpd_p = jnp.minimum(0.0, (qvoqswz - 0.90) * qswz * odtb)
    abr = 2.0 * C.PI * (qvoqswz - 0.90) * tmpc_p / (tmpa_p + tmpb_p)
    tmp1_pv = C.AV_R * sqrho * olambdar ** bp5 / visc
    tmp2_pv = abr * C.XNOR * (C.VF1R * olambdar * olambdar
                             + C.VF2R * schmidt ** 0.33334 * C.GAMBP5O2
                             * jnp.sqrt(jnp.maximum(tmp1_pv, 0.0)))
    tmp3_pv = jnp.maximum(jnp.minimum(0.0, tmp2_pv), tmpd_p)
    prevp = jnp.maximum(tmp3_pv, -qrzodt)

    # ---- sequential depletion limiters + state update (cold vs warm) ----
    def limit(num, denom):
        # factor = denom/num when num>denom else 1
        return jnp.where(num > denom, denom / jnp.where(num > denom, num, 1.0), 1.0)

    # COLD closure
    f = limit(psdep, qvzodt)
    psdep_c = psdep * f
    f = limit(praut + psacw + psfw + pracw, qlzodt)
    praut_c1 = praut * f
    psacw_c1 = psacw * f
    psfw_c = psfw * f
    pracw_c1 = pracw * f
    f = limit(psaut + psaci + praci + psfi, qizodt)
    psaut_c1 = psaut * f
    psaci_c1 = psaci * f
    praci_c1 = praci * f
    psfi_c = psfi * f
    tmp_r_c = piacr + psacr - prevp - praut_c1 - pracw_c1 + pgfr
    f = limit(tmp_r_c, qrzodt)
    piacr_c = piacr * f
    psacr_c1 = psacr * f
    prevp_c1 = prevp * f
    pgfr_c = pgfr * f
    tmp_s_c = -pssub - (psaut_c1 + psaci_c1 + psacw_c1 + psfw_c + pgfr_c + psfi_c
                        + praci_c1 + piacr_c + psdep_c + psacr_c1 - pracs)
    f = limit(tmp_s_c, qszodt)
    pssub_c = pssub * f
    pracs_c = pracs * f

    pvapor_c = -pssub_c - psdep_c - prevp_c1
    qvz_c = jnp.maximum(C.QVMIN, qvz + dtb * pvapor_c)
    pclw_c = -praut_c1 - pracw_c1 - psacw_c1 - psfw_c
    qlz_c = jnp.maximum(0.0, qlz + dtb * pclw_c)
    pcli_c = -psaut_c1 - psfi_c - psaci_c1 - praci_c1
    qiz_c = jnp.maximum(0.0, qiz + dtb * pcli_c)
    tmp_r_c2 = piacr_c + psacr_c1 - prevp_c1 - praut_c1 - pracw_c1 + pgfr_c - pracs_c
    qrz_c = jnp.maximum(0.0, qrz + dtb * (-tmp_r_c2))
    tmp_s_c2 = -pssub_c - (psaut_c1 + psaci_c1 + psacw_c1 + psfw_c + pgfr_c + psfi_c
                           + praci_c1 + piacr_c + psdep_c + psacr_c1 - pracs_c)
    psnow_c = -tmp_s_c2
    qsz_c = jnp.maximum(0.0, qsz + dtb * psnow_c)
    qschg_c = psnow_c
    theiz_c = theiz + dtb * (ocp / tothz * C.XLF * qschg_c)
    thz_c = theiz_c - (xLvocp * qvz_c - xLfocp * qiz_c) / tothz

    # WARM closure
    f = limit(praut + psacw + pracw, qlzodt)
    praut_w = praut * f
    psacw_w = psacw * f
    pracw_w = pracw * f
    f = limit(-(psmlt + psmltevp), qszodt)
    psmlt_w = psmlt * f
    psmltevp_w = psmltevp * f
    tmp_r_w = -prevp - (praut_w + pracw_w + psacw_w - psmlt_w)
    f = limit(tmp_r_w, qrzodt)
    prevp_w = prevp * f

    pvapor_w = -psmltevp_w - prevp_w
    qvz_w = jnp.maximum(C.QVMIN, qvz + dtb * pvapor_w)
    pclw_w = -praut_w - pracw_w - psacw_w
    qlz_w = jnp.maximum(0.0, qlz + dtb * pclw_w)
    qiz_w = qiz  # pcli=0 in warm
    tmp_r_w2 = -prevp_w - (praut_w + pracw_w + psacw_w - psmlt_w)
    qrz_w = jnp.maximum(0.0, qrz + dtb * (-tmp_r_w2))
    psnow_w = psmlt_w + psmltevp_w
    qsz_w = jnp.maximum(0.0, qsz + dtb * psnow_w)
    qschg_w = psnow_w
    theiz_w = theiz + dtb * (ocp / tothz * C.XLF * qschg_w)
    thz_w = theiz_w - (xLvocp * qvz_w - xLfocp * qiz_w) / tothz

    # merge cold/warm budgets (where active)
    qvz_b = jnp.where(cold, qvz_c, qvz_w)
    qlz_b = jnp.where(cold, qlz_c, qlz_w)
    qiz_b = jnp.where(cold, qiz_c, qiz_w)
    qrz_b = jnp.where(cold, qrz_c, qrz_w)
    qsz_b = jnp.where(cold, qsz_c, qsz_w)
    theiz_b = jnp.where(cold, theiz_c, theiz_w)
    thz_b = jnp.where(cold, thz_c, thz_w)

    # apply only on active levels (skipped levels keep pre-process state)
    qvz = jnp.where(active, qvz_b, qvz)
    qlz = jnp.where(active, qlz_b, qlz)
    qiz = jnp.where(active, qiz_b, qiz)
    qrz = jnp.where(active, qrz_b, qrz)
    qsz = jnp.where(active, qsz_b, qsz)
    theiz = jnp.where(active, theiz_b, theiz)
    thz = jnp.where(active, thz_b, thz)
    tem = thz * tothz
    temcc = tem - 273.15

    # qvsbar after the budget (for the saturation test). CRITICAL WRF detail:
    # the COLD branch reuses the ORIGINAL qswz/qsiz (from the top-of-column
    # saturation calc -- NOT recomputed after the budget's temperature change),
    # applying only ``if(temcc<-40) qswz=qsiz`` with the post-budget temcc. The
    # WARM branch DOES recompute qswz/qsiz from the post-budget temperature
    # (qsiz=qswz, qvsbar=qswz). Recomputing qsiz in the cold branch would flip
    # the saturation-test branch at near-saturated ice levels (the level-23
    # case-2 bug). qswz/qsiz here are still the original top-of-function values.
    qswz_cold = jnp.where(temcc < -40.0, qsiz, qswz)
    qsiz_cold = qsiz
    es_w2 = 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * temcc / (tem - C.SVP3))
    qswz_warm = C.EP2 * es_w2 / (prez - es_w2)
    qsiz_warm = qswz_warm
    qswz2 = jnp.where(cold, qswz_cold, qswz_warm)
    qsiz2 = jnp.where(cold, qsiz_cold, qsiz_warm)
    qlpqi = qlz + qiz
    qvsbar = jnp.where(qlpqi == 0.0, qsiz2, (qiz * qsiz2 + qlz * qswz2) / jnp.where(qlpqi > 0.0, qlpqi, 1.0))

    # ---- saturation adjustment ----
    rsat = 1.0
    unsat_sa = (qvz + qlz + qiz) < rsat * qvsbar
    # unsaturated: dump condensate to vapor
    qvz_u = qvz + qlz + qiz
    qlz_u = jnp.zeros_like(qlz)
    qiz_u = jnp.zeros_like(qiz)
    # saturated: Newton satadj
    qvz_s, qlz_s, qiz_s = _satadj(qvz, qlz, qiz, prez, theiz, tothz)

    do_sa = active  # satadj branch only entered on active levels in WRF
    qvz = jnp.where(do_sa & unsat_sa, qvz_u, jnp.where(do_sa, qvz_s, qvz))
    qlz = jnp.where(do_sa & unsat_sa, qlz_u, jnp.where(do_sa, qlz_s, qlz))
    qiz = jnp.where(do_sa & unsat_sa, qiz_u, jnp.where(do_sa, qiz_s, qiz))
    thz = theiz - (xLvocp * qvz - xLfocp * qiz) / tothz
    tem = thz * tothz
    temcc = tem - 273.15

    # ---- ice/water melting + freezing (only saturated-branch levels with q>0) ----
    qlpqi2 = qlz + qiz
    do_mf = do_sa & jnp.logical_not(unsat_sa) & (qlpqi2 > 0.0)
    pihom = jnp.where(temcc < -40.0, qlz * odtb, 0.0)
    pimlt = jnp.where(temcc > 0.0, qiz * odtb, 0.0)
    in_berg = (temcc < 0.0) & (temcc > -31.0)
    a1m = _parama1(temcc)
    a2m = _parama2(temcc)
    a1m = a1m * 0.001 ** (1.0 - a2m)
    xnin = C.XNI0 * jnp.exp(-C.BNI * temcc)
    pidw = jnp.where(in_berg, xnin * orho * (a1m * C.XMNIN ** a2m), 0.0)

    qlz_mf = jnp.maximum(0.0, qlz + dtb * (-pihom + pimlt - pidw))
    qiz_mf = jnp.maximum(0.0, qiz + dtb * (pihom - pimlt + pidw))
    qlz = jnp.where(do_mf, qlz_mf, qlz)
    qiz = jnp.where(do_mf, qiz_mf, qiz)
    # satadj again after melting/freezing
    qvz_s2, qlz_s2, qiz_s2 = _satadj(qvz, qlz, qiz, prez, theiz, tothz)
    qvz = jnp.where(do_mf, qvz_s2, qvz)
    qlz = jnp.where(do_mf, qlz_s2, qlz)
    qiz = jnp.where(do_mf, qiz_s2, qiz)
    thz = theiz - (xLvocp * qvz - xLfocp * qiz) / tothz

    # ---- final qvmin guard (k=kts+1..kte) ----
    below_min = qvz < C.QVMIN
    guard = jnp.arange(nz) >= 1
    qlz = jnp.where(below_min & guard, 0.0, qlz)
    qiz = jnp.where(below_min & guard, 0.0, qiz)
    qvz = jnp.where(below_min & guard, jnp.maximum(C.QVMIN, qvz + qlz + qiz), qvz)

    return thz, qvz, qlz, qrz, qiz, qsz, Riz, pptrain, pptsnow, pptice


def _sbu_ylin_column(th, qv, ql, qr, qi, qs, rho, pii, p, z, dz8w, ht, dt):
    """Single-column SBU-YLin driver: theta/q in -> theta/q out + surface precip."""

    thz, qvz, qlz, qrz, qiz, qsz, Riz, pptrain, pptsnow, pptice = _clphy1d_ylin(
        th, qv, ql, qr, qi, qs, pii, rho, p, z, dz8w, ht, dt
    )
    # WRF: RAINNCV = pptrain+pptsnow+pptice. The sedimentation flux*del_tv has
    # units kg/m^2 == mm of liquid-water equivalent, so no m->mm rescale is
    # applied (the WRF source comment "m to mm" is not reflected in its code; it
    # leaves RAINNCV in the native flux unit, which is already mm).
    rainncv = pptrain + pptsnow + pptice
    return {
        "th": thz,
        "qv": qvz,
        "qc": qlz,
        "qr": qrz,
        "qi": qiz,
        "qs": qsz,
        "ri3d": Riz,
        "rainncv": rainncv,
        "pptrain": pptrain,
        "pptsnow": pptsnow,
        "pptice": pptice,
    }


_sbu_ylin_columns = jax.jit(
    jax.vmap(_sbu_ylin_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)),
    static_argnums=(12,),
)


def sbu_ylin_run(th, qv, qc, qr, qi, qs, rho, pii, p, z, dz8w, ht, dt):
    """Run SBU-YLin on a batch of columns shaped ``(ncol, nlev)``.

    ``th`` is potential temperature, ``pii`` the Exner function, ``z`` the
    geometric mass-level height, ``dz8w`` the layer thickness, ``ht`` a scalar
    per column (surface height). Returns a dict of post-microphysics column
    fields + surface precip (mm).
    """

    return _sbu_ylin_columns(th, qv, qc, qr, qi, qs, rho, pii, p, z, dz8w, ht, float(dt))


def sbu_ylin_physics_tendency(theta, qv, qc, qr, qi, qs, pii, rho, p, z, dz8w, ht, dt):
    """Return frozen S0-style in-place replacements for SBU-YLin.

    The diagnostic ice-Richardson profile ``ri3d`` (WRF ``rimi``/``Ri3D``) is
    returned in ``diagnostics`` -- it is a pure scheme OUTPUT and is NOT a
    prognostic State leaf, so it is not in ``state_replacements`` (which keeps
    SBU-YLin a low-state, contract-clean microphysics: it reads/writes only the
    standard ``qv,qc,qr,qi,qs`` moist members).
    """

    out = sbu_ylin_run(theta, qv, qc, qr, qi, qs, rho, pii, p, z, dz8w, ht, dt)
    return PhysicsTendency(
        state_replacements={
            "theta": out["th"],
            "qv": out["qv"],
            "qc": out["qc"],
            "qr": out["qr"],
            "qi": out["qi"],
            "qs": out["qs"],
        },
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["pptsnow"] + out["pptice"],
        },
        diagnostics={"ri3d": out["ri3d"]},
    )


__all__ = ["sbu_ylin_run", "sbu_ylin_physics_tendency"]
