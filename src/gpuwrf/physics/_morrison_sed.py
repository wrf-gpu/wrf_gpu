"""Morrison sedimentation (Reisner 1998 split-step) + final state update.

Faithful vectorized port of the sedimentation loop (lines ~3356-3667) and the
final state-update / instantaneous-process / slope-recompute / effective-radius
block (lines ~3669-4056) of ``MORR_TWO_MOMENT_MICRO``.

Sedimentation uses a per-column number of split steps NSTEP (data-dependent on
the max fall speed and the CFL); to keep it jittable we cap NSTEP at a static
maximum and run a fixed jax.lax loop, doing nothing on iterations beyond the
column's NSTEP (the Fortran's NSTEP is the same for the whole column).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.physics import morrison_constants as C

_EPS = 1.0e-30
# Static cap on the number of sedimentation sub-steps. The CFL gives
# NSTEP ~ max(fallspeed)*dt/dz; with dt=60s, dz~250-600m and fall speeds capped
# at ~20 m/s (graupel) the worst case is ~5; 32 is a very safe ceiling that the
# parity test cross-checks (NSTEP is reported and asserted < cap).
_NSTEP_MAX = 64


def _gamma(x):
    from gpuwrf.physics.microphysics_morrison import gamma_fn
    return gamma_fn(x)


def _sedimentation(qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
                   qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
                   nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
                   t, p, rho, dz, dt, do_cell):
    """Return sedimentation tendencies and surface precip (PRECRT/SNOWRT/...)."""
    QSMALL = C.QSMALL
    eps = _EPS
    one = jnp.ones_like(t)
    ncol, kx = t.shape

    # post-process-rate provisional values (DUMx = q + qten*dt)
    dumi = qi + qi_ten * dt
    dumqs = qs + qni_ten * dt
    dumr = qr + qr_ten * dt
    dumfni = jnp.maximum(ni + ni_ten * dt, 0.0)
    dumfns = jnp.maximum(ns + ns_ten * dt, 0.0)
    dumfnr = jnp.maximum(nr + nr_ten * dt, 0.0)
    dumc = qc + qc_ten * dt
    dumfnc = nc  # iinum=1: DUMFNC = NC3D
    dumg = qg + qg_ten * dt
    dumfng = jnp.maximum(ng + ng_ten * dt, 0.0)
    dumfnc = jnp.maximum(dumfnc, 0.0)

    # dummy slope (clamped) for sedimentation fall speeds
    def dlam(q, n, cmass, inv_pow, lammin, lammax):
        act = q >= QSMALL
        qsf = jnp.where(act, q, 1.0)
        lam = (cmass * jnp.where(act, n, 0.0) / qsf) ** inv_pow
        lam = jnp.clip(jnp.where(act, lam, lammin), lammin, lammax)
        return jnp.where(act, lam, lammin)

    dlami = dlam(dumi, dumfni, C.CONS12, 1.0 / C.DI, C.LAMMINI, C.LAMMAXI)
    dlamr = dlam(dumr, dumfnr, C.PI * C.RHOW, 1.0 / 3.0, C.LAMMINR, C.LAMMAXR)
    dlams = dlam(dumqs, dumfns, C.CONS1, 1.0 / C.DS, C.LAMMINS, C.LAMMAXS)
    dlamg = dlam(dumg, dumfng, C.CONS2, 1.0 / C.DG, C.LAMMING, C.LAMMAXG)

    # droplet PGAM + DLAMC (uses NC3D, not DUMFNC, for PGAM per the Fortran)
    dum_rho = p / (287.15 * t)
    pgam = 0.0005714 * (nc / 1.0e6 * dum_rho) + 0.2714
    pgam = 1.0 / (pgam * pgam) - 1.0
    pgam = jnp.clip(pgam, 2.0, 10.0)
    g_p1 = _gamma(pgam + 1.0)
    g_p4 = _gamma(pgam + 4.0)
    actc = dumc >= QSMALL
    dumc_s = jnp.where(actc, dumc, 1.0)
    dlamc = (C.CONS26 * dumfnc * g_p4 / (dumc_s * g_p1)) ** (1.0 / 3.0)
    lcmin = (pgam + 1.0) / 60.0e-6
    lcmax = (pgam + 1.0) / 1.0e-6
    dlamc = jnp.clip(jnp.where(actc, dlamc, lcmin), lcmin, lcmax)

    dum_dc = (C.RHOSU / rho) ** 0.54
    mu = 1.496e-6 * t ** 1.5 / (t + 120.0)
    acn = C.G * C.RHOW / (18.0 * mu)
    ain = (C.RHOSU / rho) ** 0.35 * C.AI
    arn = dum_dc * C.AR
    asn = dum_dc * C.AS
    agn = dum_dc * C.AG

    # fall speeds
    g_bc_p1 = _gamma(1.0 + C.BC + pgam)
    g_bc_p4 = _gamma(4.0 + C.BC + pgam)
    unc = jnp.where(actc, acn * g_bc_p1 / (dlamc ** C.BC * g_p1 + eps), 0.0)
    umc = jnp.where(actc, acn * g_bc_p4 / (dlamc ** C.BC * g_p4 + eps), 0.0)
    uni = jnp.where(dumi >= QSMALL, ain * C.CONS27 / (dlami ** C.BI + eps), 0.0)
    umi = jnp.where(dumi >= QSMALL, ain * C.CONS28 / (dlami ** C.BI + eps), 0.0)
    unr = jnp.where(dumr >= QSMALL, arn * C.CONS6 / (dlamr ** C.BR + eps), 0.0)
    umr = jnp.where(dumr >= QSMALL, arn * C.CONS4 / (dlamr ** C.BR + eps), 0.0)
    ums = jnp.where(dumqs >= QSMALL, asn * C.CONS3 / (dlams ** C.BS + eps), 0.0)
    uns = jnp.where(dumqs >= QSMALL, asn * C.CONS5 / (dlams ** C.BS + eps), 0.0)
    umg = jnp.where(dumg >= QSMALL, agn * C.CONS7 / (dlamg ** C.BG + eps), 0.0)
    ung = jnp.where(dumg >= QSMALL, agn * C.CONS8 / (dlamg ** C.BG + eps), 0.0)

    # realistic fall-speed limits
    ums = jnp.minimum(ums, 1.2 * dum_dc)
    uns = jnp.minimum(uns, 1.2 * dum_dc)
    umi = jnp.minimum(umi, 1.2 * (C.RHOSU / rho) ** 0.35)
    uni = jnp.minimum(uni, 1.2 * (C.RHOSU / rho) ** 0.35)
    umr = jnp.minimum(umr, 9.1 * dum_dc)
    unr = jnp.minimum(unr, 9.1 * dum_dc)
    umg = jnp.minimum(umg, 20.0 * dum_dc)
    ung = jnp.minimum(ung, 20.0 * dum_dc)

    fr = umr; fi = umi; fni = uni; fs = ums; fns = uns
    fnr = unr; fc = umc; fnc = unc; fg = umg; fng = ung

    # V3.3: modify fall speed below level of precip (carry down from k+1).
    # k index 0=bottom .. kx-1=top. The Fortran "K.LE.KTE-1" with FR(K+1) refers
    # to the level ABOVE (higher k). We fill any near-zero fall speed at level k
    # from level k+1, sweeping top-down so the carry propagates downward.
    def fill_down(fld):
        def body(kk, arr):
            k = kx - 2 - kk  # from kx-2 down to 0
            above = arr[:, k + 1]
            cur = arr[:, k]
            new = jnp.where(cur < 1.0e-10, above, cur)
            return arr.at[:, k].set(new)
        return jax.lax.fori_loop(0, kx - 1, body, fld)

    fr = fill_down(fr); fi = fill_down(fi); fni = fill_down(fni)
    fs = fill_down(fs); fns = fill_down(fns); fnr = fill_down(fnr)
    fc = fill_down(fc); fnc = fill_down(fnc); fg = fill_down(fg); fng = fill_down(fng)

    # number of split steps (per column = max over k of the CFL count)
    rgvm = fr
    for _fld in (fi, fs, fc, fni, fnr, fns, fnc, fg, fng):
        rgvm = jnp.maximum(rgvm, _fld)
    nstep_k = jnp.floor(rgvm * dt / dz + 1.0).astype(jnp.int32)
    nstep = jnp.maximum(jnp.max(nstep_k, axis=1), 1)  # (ncol,)
    nstep = jnp.minimum(nstep, _NSTEP_MAX)

    # multiply DUM by rho (these are the "in-flight" prognostic amounts)
    dumr = dumr * rho; dumi = dumi * rho; dumfni = dumfni * rho
    dumqs = dumqs * rho; dumfns = dumfns * rho; dumfnr = dumfnr * rho
    dumc = dumc * rho; dumfnc = dumfnc * rho; dumg = dumg * rho; dumfng = dumfng * rho

    # accumulators
    qrsten = jnp.zeros_like(t); qisten = jnp.zeros_like(t); qnisten = jnp.zeros_like(t)
    qcsten = jnp.zeros_like(t); qgsten = jnp.zeros_like(t)
    ni_sed = jnp.zeros_like(t); ns_sed = jnp.zeros_like(t); nr_sed = jnp.zeros_like(t)
    nc_sed = jnp.zeros_like(t); ng_sed = jnp.zeros_like(t)
    precrt = jnp.zeros((ncol,), t.dtype)
    snowrt = jnp.zeros((ncol,), t.dtype)
    snowprt = jnp.zeros((ncol,), t.dtype)
    grplprt = jnp.zeros((ncol,), t.dtype)

    nstep_f = nstep.astype(t.dtype)[:, None]   # (ncol,1)

    # one fall species step. fld is mass/number "*rho" array; F is fall speed.
    # Returns updated fld, the per-level sed tendency increment, and surface flux.
    nstep_col = nstep_f[:, 0]  # (ncol,)

    def fall_one(fld, F):
        falout = F * fld                            # (ncol,kx)
        ktop = kx - 1
        # top level (k = kx-1): only outgoing flux divergence
        faltnd_top = falout[:, ktop] / dz[:, ktop]  # (ncol,)
        ten = jnp.zeros_like(fld)
        ten = ten.at[:, ktop].add(-faltnd_top / nstep_col / rho[:, ktop])
        fld = fld.at[:, ktop].add(-faltnd_top * dt / nstep_col)
        # interior + bottom k=0..kx-2: (falout[k+1]-falout[k])/dz[k]
        fo_kp1 = falout[:, 1:]      # k+1 for k=0..kx-2
        fo_k = falout[:, :-1]       # k
        faltnd = (fo_kp1 - fo_k) / dz[:, :-1]       # (ncol, kx-1)
        ten = ten.at[:, :-1].add(faltnd / nstep_f / rho[:, :-1])
        fld = fld.at[:, :-1].add(faltnd * dt / nstep_f)
        surf = falout[:, 0]                          # FALOUT at KTS (bottom)
        return fld, ten, surf, falout

    def step_body(istep, carry):
        (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
         qrsten, qisten, qnisten, qcsten, qgsten,
         ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
         precrt, snowrt, snowprt, grplprt) = carry
        active = (istep < nstep)[:, None].astype(t.dtype)  # (ncol,1) 1 if this substep runs

        dumr2, tr, sr_r, for_ = fall_one(dumr, fr)
        dumi2, ti, sr_i, foi = fall_one(dumi, fi)
        dumfni2, tni, _, _ = fall_one(dumfni, fni)
        dumqs2, ts, sr_s, fos = fall_one(dumqs, fs)
        dumfns2, tns, _, _ = fall_one(dumfns, fns)
        dumfnr2, tnr, _, _ = fall_one(dumfnr, fnr)
        dumc2, tc, sr_c, foc = fall_one(dumc, fc)
        dumfnc2, tnc, _, _ = fall_one(dumfnc, fnc)
        dumg2, tg, sr_g, fog = fall_one(dumg, fg)
        dumfng2, tng, _, _ = fall_one(dumfng, fng)

        a = active
        am = a  # mask for per-level tendency add (broadcast)
        dumr = jnp.where(a > 0, dumr2, dumr)
        dumi = jnp.where(a > 0, dumi2, dumi)
        dumfni = jnp.where(a > 0, dumfni2, dumfni)
        dumqs = jnp.where(a > 0, dumqs2, dumqs)
        dumfns = jnp.where(a > 0, dumfns2, dumfns)
        dumfnr = jnp.where(a > 0, dumfnr2, dumfnr)
        dumc = jnp.where(a > 0, dumc2, dumc)
        dumfnc = jnp.where(a > 0, dumfnc2, dumfnc)
        dumg = jnp.where(a > 0, dumg2, dumg)
        dumfng = jnp.where(a > 0, dumfng2, dumfng)

        qrsten = qrsten + tr * am
        qisten = qisten + ti * am
        qnisten = qnisten + ts * am
        qcsten = qcsten + tc * am
        qgsten = qgsten + tg * am
        ni_sed = ni_sed + tni * am
        ns_sed = ns_sed + tns * am
        nr_sed = nr_sed + tnr * am
        nc_sed = nc_sed + tnc * am
        ng_sed = ng_sed + tng * am

        amc = active[:, 0]  # (ncol,)
        precrt = precrt + (sr_r + sr_c + sr_s + sr_i + sr_g) * dt / nstep_f[:, 0] * amc
        snowrt = snowrt + (sr_s + sr_i + sr_g) * dt / nstep_f[:, 0] * amc
        snowprt = snowprt + (sr_i + sr_s) * dt / nstep_f[:, 0] * amc
        grplprt = grplprt + (sr_g) * dt / nstep_f[:, 0] * amc

        return (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
                qrsten, qisten, qnisten, qcsten, qgsten,
                ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
                precrt, snowrt, snowprt, grplprt)

    carry = (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
             qrsten, qisten, qnisten, qcsten, qgsten,
             ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
             precrt, snowrt, snowprt, grplprt)
    carry = jax.lax.fori_loop(0, _NSTEP_MAX, step_body, carry)
    (dumr, dumi, dumfni, dumqs, dumfns, dumfnr, dumc, dumfnc, dumg, dumfng,
     qrsten, qisten, qnisten, qcsten, qgsten,
     ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
     precrt, snowrt, snowprt, grplprt) = carry

    # zero out sedimentation for skipped (GOTO 200 / no-hydrometeor) columns?
    # The Fortran computes sed for the whole column whenever LTRUE=1 (any cell).
    # do_cell governs per-cell process rates; sedimentation runs for the column.
    return (qrsten, qisten, qnisten, qcsten, qgsten,
            ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
            precrt, snowrt, snowprt, grplprt)


def _finalize(t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
              qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
              t_ten, qv_ten,
              nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
              xxlv, xxls, xlf, cpm, p, rho, dt, do_cell):
    """Final state update, instantaneous processes, slope recompute, Reff.

    Mirrors module_mp_morr_two_moment.F lines ~3669-4056. ``t``/``qv`` are the
    post-setup values; the *_ten arrays are the FULL accumulated tendencies
    (process rates + saturation adjustment + sedimentation).
    """
    QSMALL = C.QSMALL
    EP_2 = C.EP_2
    R = C.R
    one = jnp.ones_like(t)

    # ice -> snow if mean diameter > 2*dcs (LAMI computed from PRE-update qi/ni,
    # exactly as the Fortran which uses LAMI from the process-rate slope block;
    # here we recompute the pre-update slope since LAMI(K) was last set there).
    lami_pre = jnp.where(qi >= QSMALL,
                         (C.CONS12 * jnp.maximum(ni, 0.0)
                          / jnp.where(qi >= QSMALL, qi, one)) ** (1.0 / C.DI), 0.0)
    lami_pre = jnp.clip(jnp.where(qi >= QSMALL, lami_pre, 0.0), 0.0, None)
    big_ice = (qi >= QSMALL) & (t < 273.15) & (lami_pre >= 1.0e-10) \
        & (1.0 / jnp.where(lami_pre > 0, lami_pre, one) >= 2.0 * C.DCS)
    qni_ten = jnp.where(big_ice, qni_ten + qi / dt + qi_ten, qni_ten)
    ns_ten = jnp.where(big_ice, ns_ten + ni / dt + ni_ten, ns_ten)
    qi_ten = jnp.where(big_ice, -qi / dt, qi_ten)
    ni_ten = jnp.where(big_ice, -ni / dt, ni_ten)

    # apply tendencies to update prognostics
    qc = qc + qc_ten * dt
    qi = qi + qi_ten * dt
    qs = qs + qni_ten * dt
    qr = qr + qr_ten * dt
    nc = nc + nc_ten * dt
    ni = ni + ni_ten * dt
    ns = ns + ns_ten * dt
    nr = nr + nr_ten * dt
    qg = qg + qg_ten * dt    # IGRAUP=0
    ng = ng + ng_ten * dt
    t = t + t_ten * dt
    qv = qv + qv_ten * dt

    # saturation + subsaturation trace removal
    evs = jnp.minimum(0.99 * p, _polysvp0(t))
    eis = jnp.minimum(0.99 * p, _polysvp1(t))
    eis = jnp.where(eis > evs, evs, eis)
    qvs = EP_2 * evs / (p - evs)
    qvi = EP_2 * eis / (p - eis)
    qvqvs = qv / qvs
    qvqvsi = qv / qvi

    def _rm_liq(qv_, t_, qx, thr, cpm_):
        rm = (qvqvs < 0.9) & (qx < thr)
        return (jnp.where(rm, qv_ + qx, qv_),
                jnp.where(rm, t_ - qx * xxlv / cpm_, t_),
                jnp.where(rm, 0.0, qx))

    def _rm_ice(qv_, t_, qx, thr, cpm_):
        rm = (qvqvsi < 0.9) & (qx < thr)
        return (jnp.where(rm, qv_ + qx, qv_),
                jnp.where(rm, t_ - qx * xxls / cpm_, t_),
                jnp.where(rm, 0.0, qx))

    qv, t, qr = _rm_liq(qv, t, qr, 1.0e-8, cpm)
    qv, t, qc = _rm_liq(qv, t, qc, 1.0e-8, cpm)
    qv, t, qi = _rm_ice(qv, t, qi, 1.0e-8, cpm)
    qv, t, qs = _rm_ice(qv, t, qs, 1.0e-8, cpm)
    qv, t, qg = _rm_ice(qv, t, qg, 1.0e-8, cpm)

    # QSMALL zeroing
    def _zero(qx, nx):
        m = qx < QSMALL
        return jnp.where(m, 0.0, qx), jnp.where(m, 0.0, nx)

    qc, nc = _zero(qc, nc)
    qr, nr = _zero(qr, nr)
    qi, ni = _zero(qi, ni)
    qs, ns = _zero(qs, ns)
    qg, ng = _zero(qg, ng)

    # GOTO 500: skip instantaneous processes + slope if cell empty
    empty = ((qc < QSMALL) & (qi < QSMALL) & (qs < QSMALL)
             & (qr < QSMALL) & (qg < QSMALL))
    nonempty = ~empty

    # instantaneous: melt cloud ice -> rain (T >= 273.15)
    melt_i = (qi >= QSMALL) & (t >= 273.15) & nonempty
    qr = jnp.where(melt_i, qr + qi, qr)
    t = jnp.where(melt_i, t - qi * xlf / cpm, t)
    nr = jnp.where(melt_i, nr + ni, nr)
    qi = jnp.where(melt_i, 0.0, qi)
    ni = jnp.where(melt_i, 0.0, ni)

    # homogeneous freezing of cloud water (T <= 233.15)
    hf_c = (t <= 233.15) & (qc >= QSMALL) & nonempty
    qi = jnp.where(hf_c, qi + qc, qi)
    t = jnp.where(hf_c, t + qc * xlf / cpm, t)
    qc = jnp.where(hf_c, 0.0, qc)
    ni = jnp.where(hf_c, ni + nc, ni)
    nc = jnp.where(hf_c, 0.0, nc)

    # homogeneous freezing of rain (IGRAUP=0 -> to graupel)
    hf_r = (t <= 233.15) & (qr >= QSMALL) & nonempty
    qg = jnp.where(hf_r, qg + qr, qg)
    t = jnp.where(hf_r, t + qr * xlf / cpm, t)
    qr = jnp.where(hf_r, 0.0, qr)
    ng = jnp.where(hf_r, ng + nr, ng)
    nr = jnp.where(hf_r, 0.0, nr)

    # number non-negative
    ni = jnp.maximum(ni, 0.0); ns = jnp.maximum(ns, 0.0)
    nc = jnp.maximum(nc, 0.0); nr = jnp.maximum(nr, 0.0); ng = jnp.maximum(ng, 0.0)

    # slope recompute (with clamps; back-adjust number) for Reff
    lami, ni = _slope_final(qi, ni, C.CONS12, 1.0 / C.DI, C.LAMMINI, C.LAMMAXI)
    lamr, nr = _slope_final(qr, nr, C.PI * C.RHOW, 1.0 / 3.0, C.LAMMINR, C.LAMMAXR)
    lams, ns = _slope_final(qs, ns, C.CONS1, 1.0 / C.DS, C.LAMMINS, C.LAMMAXS)
    lamg, ng = _slope_final(qg, ng, C.CONS2, 1.0 / C.DG, C.LAMMING, C.LAMMAXG)

    # droplet slope (PGAM/LAMC) for EFFC
    actc = qc >= QSMALL
    dum_rho = p / (287.15 * t)
    pgam = 0.0005714 * (nc / 1.0e6 * dum_rho) + 0.2714
    pgam = 1.0 / (pgam * pgam) - 1.0
    pgam = jnp.clip(pgam, 2.0, 10.0)
    g_p1 = _gamma(pgam + 1.0)
    g_p3 = _gamma(pgam + 3.0)
    g_p4 = _gamma(pgam + 4.0)
    qcs = jnp.where(actc, qc, 1.0)
    lamc = (C.CONS26 * nc * g_p4 / (qcs * g_p1)) ** (1.0 / 3.0)
    lcmin = (pgam + 1.0) / 60.0e-6
    lcmax = (pgam + 1.0) / 1.0e-6
    too_s = lamc < lcmin
    too_b = lamc > lcmax
    lamc = jnp.where(too_s, lcmin, jnp.where(too_b, lcmax, lamc))
    nc_adj = jnp.exp(3.0 * jnp.log(lamc) + jnp.log(qcs) + jnp.log(g_p1)
                     - jnp.log(g_p4)) / C.CONS26
    nc = jnp.where(actc & (too_s | too_b), nc_adj, nc)
    lamc = jnp.where(actc, lamc, 0.0)

    # effective radii (micron). Default 25 where species absent.
    def reff(active, lam):
        return jnp.where(active, 3.0 / jnp.where(lam > 0, lam, one) / 2.0 * 1.0e6, 25.0)

    effi = reff(qi >= QSMALL, lami)
    effs = reff(qs >= QSMALL, lams)
    effr = reff(qr >= QSMALL, lamr)
    effg = reff(qg >= QSMALL, lamg)
    effc = jnp.where(qc >= QSMALL,
                     g_p4 / g_p3 / jnp.where(lamc > 0, lamc, one) / 2.0 * 1.0e6, 25.0)

    # ice number upper bound; constant droplet number reset (iinum=1)
    ni = jnp.minimum(ni, 0.3e6 / rho)
    nc = C.NDCNST * 1.0e6 / rho

    return (t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
            effc, effi, effs, effr, effg)


def _slope_final(q, n, cmass, inv_pow, lammin, lammax):
    """Slope recompute with clamp + number back-adjust (no n0 needed for Reff)."""
    QSMALL = C.QSMALL
    one = jnp.ones_like(q)
    act = q >= QSMALL
    qsf = jnp.where(act, q, 1.0)
    nn = jnp.maximum(n, 0.0)
    lam = (cmass * jnp.where(act, nn, 0.0) / qsf) ** inv_pow
    lam = jnp.where(act, lam, lammin)
    too_s = lam < lammin
    too_b = lam > lammax
    lam_c = jnp.where(too_s, lammin, jnp.where(too_b, lammax, lam))
    n0_c = lam_c ** 4 * jnp.where(act, q, 0.0) / cmass
    n_out = jnp.where(act & (too_s | too_b), n0_c / lam_c, n)
    return jnp.where(act, lam_c, 0.0), n_out


def _polysvp0(t):
    from gpuwrf.physics.microphysics_morrison import polysvp
    return polysvp(t, 0)


def _polysvp1(t):
    from gpuwrf.physics.microphysics_morrison import polysvp
    return polysvp(t, 1)
