"""Morrison cold-branch (T < 273.15) process rates, conservation, tendencies.

Faithful vectorized port of the ``ELSE`` (temperature < 273.15) block of
``MORR_TWO_MOMENT_MICRO`` (phys/module_mp_morr_two_moment.F lines ~2101-3327),
for the default config (IGRAUP=0, ILIQ=0, INUC=0, iinum=1, IHAIL=0).

Returns the per-second tendency contributions (to be added to the accumulators
in the caller) and the slope-clamped number concentrations for cold cells:
  (qv_t, t_t, qc_t, qr_t, qi_t, qni_t, qg_t,
   nc_t, ni_t, ns_t, nr_t, ng_t,
   ni_clamp, ns_clamp, nr_clamp, ng_clamp)
"""

from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.physics import morrison_constants as C

_EPS = 1.0e-30


def _slope_generic(q, n, cmass, inv_pow, lammin, lammax, qsmall):
    active = q >= qsmall
    qs = jnp.where(active, q, 1.0)
    ns = jnp.maximum(n, 0.0)
    lam = (cmass * jnp.where(active, ns, 0.0) / qs) ** inv_pow
    lam = jnp.where(active, lam, lammin)
    too_small = lam < lammin
    too_big = lam > lammax
    lam_c = jnp.where(too_small, lammin, jnp.where(too_big, lammax, lam))
    n0_c = lam_c ** 4 * jnp.where(active, q, 0.0) / cmass
    n_c = jnp.where(too_small | too_big, n0_c / lam_c, ns)
    n0 = jnp.where(too_small | too_big, n0_c, jnp.where(active, ns * lam, 0.0))
    lam_out = jnp.where(active, lam_c, 0.0)
    n_out = jnp.where(active, n_c, n)
    n0_out = jnp.where(active, n0, 0.0)
    return lam_out, n0_out, n_out


def _gamma(x):
    from gpuwrf.physics.microphysics_morrison import gamma_fn
    return gamma_fn(x)


def _cold_branch(cold, t, qv, qc, qr, qi, qs, qg, nc, ni, ns, nr, ng,
                 qvs, qvi, qvqvs, qvqvsi, ab, abi, rho, dv, mu, sc, kap,
                 ain, arn, asn, agn, acn, dum_dc, xxlv, xxls, xlf, cpm, p, dt):
    QSMALL = C.QSMALL
    PI = C.PI
    eps = _EPS
    one = jnp.ones_like(t)

    niw = jnp.maximum(ni, 0.0)
    nsw = jnp.maximum(ns, 0.0)
    nrw = jnp.maximum(nr, 0.0)
    ngw = jnp.maximum(ng, 0.0)
    ncw = jnp.maximum(nc, 0.0)

    # slope params
    lami, n0i, ni_c = _slope_generic(qi, niw, C.CONS12, 1.0 / C.DI,
                                     C.LAMMINI, C.LAMMAXI, QSMALL)
    lamr, n0rr, nr_c = _slope_generic(qr, nrw, PI * C.RHOW, 1.0 / 3.0,
                                      C.LAMMINR, C.LAMMAXR, QSMALL)
    lams, n0s, ns_c = _slope_generic(qs, nsw, C.CONS1, 1.0 / C.DS,
                                     C.LAMMINS, C.LAMMAXS, QSMALL)
    lamg, n0g, ng_c = _slope_generic(qg, ngw, C.CONS2, 1.0 / C.DG,
                                     C.LAMMING, C.LAMMAXG, QSMALL)

    # droplet PGAM/LAMC/CDIST1
    qc_act = qc >= QSMALL
    dum_rho = p / (287.15 * t)
    pgam = 0.0005714 * (nc / 1.0e6 * dum_rho) + 0.2714
    pgam = 1.0 / (pgam * pgam) - 1.0
    pgam = jnp.clip(pgam, 2.0, 10.0)
    g_p1 = _gamma(pgam + 1.0)
    g_p4 = _gamma(pgam + 4.0)
    g_p5 = _gamma(pgam + 5.0)
    g_p2 = _gamma(pgam + 2.0)
    g_p7 = _gamma(7.0 + pgam)
    qcs = jnp.where(qc_act, qc, 1.0)
    lamc = (C.CONS26 * nc * g_p4 / (qcs * g_p1)) ** (1.0 / 3.0)
    lammin = (pgam + 1.0) / 60.0e-6
    lammax = (pgam + 1.0) / 1.0e-6
    too_s = lamc < lammin
    too_b = lamc > lammax
    lamc = jnp.where(too_s, lammin, jnp.where(too_b, lammax, lamc))
    nc_adj = jnp.exp(3.0 * jnp.log(lamc) + jnp.log(qcs)
                     + jnp.log(g_p1) - jnp.log(g_p4)) / C.CONS26
    nc = jnp.where(qc_act & (too_s | too_b), nc_adj, nc)
    cdist1 = jnp.where(qc_act, nc / g_p1, 0.0)
    lamc = jnp.where(qc_act, lamc, 0.0)

    # ---- contact + immersion freezing of cloud droplets (T < 269.15) ----
    frz = qc_act & (t < 269.15)
    nacnt = jnp.exp(-2.80 + 0.262 * (273.15 - t)) * 1000.0
    dum_mfp = 7.37 * t / (288.0 * 10.0 * p) / 100.0
    dap = C.CONS37 * t * (1.0 + dum_mfp / C.RIN) / mu
    lamc_safe = jnp.where(lamc > 0, lamc, one)
    mnuccc = jnp.where(frz, C.CONS38 * dap * nacnt
                       * jnp.exp(jnp.log(jnp.where(cdist1 > 0, cdist1, one))
                                 + jnp.log(g_p5) - 4.0 * jnp.log(lamc_safe)), 0.0)
    nnuccc = jnp.where(frz, 2.0 * PI * dap * nacnt * cdist1 * g_p2 / lamc_safe, 0.0)
    # immersion (Bigg) addition
    mnuccc = jnp.where(frz, mnuccc + C.CONS39
                       * jnp.exp(jnp.log(jnp.where(cdist1 > 0, cdist1, one))
                                 + jnp.log(g_p7) - 6.0 * jnp.log(lamc_safe))
                       * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0), 0.0)
    nnuccc = jnp.where(frz, nnuccc + C.CONS40
                       * jnp.exp(jnp.log(jnp.where(cdist1 > 0, cdist1, one))
                                 + jnp.log(g_p4) - 3.0 * jnp.log(lamc_safe))
                       * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0), 0.0)
    nnuccc = jnp.minimum(nnuccc, nc / dt)
    nnuccc = jnp.where(frz, nnuccc, 0.0)
    mnuccc = jnp.where(frz, mnuccc, 0.0)

    # ---- autoconversion (KK2000) ----
    qc_ge6 = qc >= 1.0e-6
    prc = jnp.where(qc_ge6, 1350.0 * qc ** 2.47 * (nc / 1.0e6 * rho) ** (-1.79), 0.0)
    nprc1 = jnp.where(qc_ge6, prc / C.CONS29, 0.0)
    nprc = jnp.where(qc_ge6, prc / (qc / jnp.where(nc > 0, nc, one)), 0.0)
    nprc = jnp.minimum(nprc, nc / dt)
    nprc1 = jnp.minimum(nprc1, nprc)

    # ---- snow aggregation ----
    qs8 = qs >= 1.0e-8
    nsagg = jnp.where(qs8, C.CONS15 * asn * rho ** ((2.0 + C.BS) / 3.0)
                      * qs ** ((2.0 + C.BS) / 3.0)
                      * (ns * rho) ** ((4.0 - C.BS) / 3.0) / rho, 0.0)

    # ---- accretion of droplets onto snow (PSACWS) ----
    sn_qc = qs8 & qc_act
    psacws = jnp.where(sn_qc, C.CONS13 * asn * qc * rho * n0s
                       / (lams ** (C.BS + 3.0) + eps), 0.0)
    npsacws = jnp.where(sn_qc, C.CONS13 * asn * nc * rho * n0s
                        / (lams ** (C.BS + 3.0) + eps), 0.0)

    # ---- collection of droplets by graupel (PSACWG) ----
    qg8 = qg >= 1.0e-8
    g_qc = qg8 & qc_act
    psacwg = jnp.where(g_qc, C.CONS14 * agn * qc * rho * n0g
                       / (lamg ** (C.BG + 3.0) + eps), 0.0)
    npsacwg = jnp.where(g_qc, C.CONS14 * agn * nc * rho * n0g
                        / (lamg ** (C.BG + 3.0) + eps), 0.0)

    # ---- cloud ice collecting droplets (PSACWI), only if 1/lami >= 100 micron ----
    qi8 = qi >= 1.0e-8
    lami_safe = jnp.where(lami > 0, lami, one)
    ice_rime = qi8 & qc_act & (1.0 / lami_safe >= 100.0e-6)
    psacwi = jnp.where(ice_rime, C.CONS16 * ain * qc * rho * n0i
                       / (lami ** (C.BI + 3.0) + eps), 0.0)
    npsacwi = jnp.where(ice_rime, C.CONS16 * ain * nc * rho * n0i
                        / (lami ** (C.BI + 3.0) + eps), 0.0)

    # ---- accretion of rain by snow (PRACS), and PSACR ----
    qr8 = qr >= 1.0e-8
    rs = qr8 & qs8
    ums = jnp.minimum(asn * C.CONS3 / (lams ** C.BS + eps), 1.2 * dum_dc)
    umr = jnp.minimum(arn * C.CONS4 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    uns = jnp.minimum(asn * C.CONS5 / (lams ** C.BS + eps), 1.2 * dum_dc)
    unr = jnp.minimum(arn * C.CONS6 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    pracs = jnp.where(
        rs, C.CONS41 * (((1.2 * umr - 0.95 * ums) ** 2 + 0.08 * ums * umr) ** 0.5
                        * rho * n0rr * n0s / (lamr ** 3 + eps)
                        * (5.0 / (lamr ** 3 * lams + eps)
                           + 2.0 / (lamr ** 2 * lams ** 2 + eps)
                           + 0.5 / (lamr * lams ** 3 + eps))), 0.0)
    npracs = jnp.where(
        rs, C.CONS32 * rho * (1.7 * (unr - uns) ** 2 + 0.3 * unr * uns) ** 0.5
        * n0rr * n0s * (1.0 / (lamr ** 3 * lams + eps)
                        + 1.0 / (lamr ** 2 * lams ** 2 + eps)
                        + 1.0 / (lamr * lams ** 3 + eps)), 0.0)
    pracs = jnp.where(rs, jnp.minimum(pracs, qr / dt), 0.0)
    npracs = jnp.where(rs, npracs, 0.0)
    psacr_cond = rs & (qs >= 0.1e-3) & (qr >= 0.1e-3)
    psacr = jnp.where(
        psacr_cond, C.CONS31 * (((1.2 * umr - 0.95 * ums) ** 2 + 0.08 * ums * umr) ** 0.5
                                * rho * n0rr * n0s / (lams ** 3 + eps)
                                * (5.0 / (lams ** 3 * lamr + eps)
                                   + 2.0 / (lams ** 2 * lamr ** 2 + eps)
                                   + 0.5 / (lams * lamr ** 3 + eps))), 0.0)

    # ---- collection of rain by graupel (PRACG) ----
    rg = qr8 & qg8
    umg = jnp.minimum(agn * C.CONS7 / (lamg ** C.BG + eps), 20.0 * dum_dc)
    umr2 = jnp.minimum(arn * C.CONS4 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    ung = jnp.minimum(agn * C.CONS8 / (lamg ** C.BG + eps), 20.0 * dum_dc)
    unr2 = jnp.minimum(arn * C.CONS6 / (lamr ** C.BR + eps), 9.1 * dum_dc)
    pracg = jnp.where(
        rg, C.CONS41 * (((1.2 * umr2 - 0.95 * umg) ** 2 + 0.08 * umg * umr2) ** 0.5
                        * rho * n0rr * n0g / (lamr ** 3 + eps)
                        * (5.0 / (lamr ** 3 * lamg + eps)
                           + 2.0 / (lamr ** 2 * lamg ** 2 + eps)
                           + 0.5 / (lamr * lamg ** 3 + eps))), 0.0)
    npracg = jnp.where(
        rg, C.CONS32 * rho * (1.7 * (unr2 - ung) ** 2 + 0.3 * unr2 * ung) ** 0.5
        * n0rr * n0g * (1.0 / (lamr ** 3 * lamg + eps)
                        + 1.0 / (lamr ** 2 * lamg ** 2 + eps)
                        + 1.0 / (lamr * lamg ** 3 + eps)), 0.0)
    pracg = jnp.where(rg, jnp.minimum(pracg, qr / dt), 0.0)
    npracg = jnp.where(rg, npracg, 0.0)

    # ---- rime-splintering (Hallett-Mossop) for snow and graupel ----
    def _fmult(tt):
        f = jnp.where(tt > 270.16, 0.0,
                      jnp.where(tt > 268.16, (270.16 - tt) / 2.0,
                                jnp.where(tt >= 265.16, (tt - 265.16) / 3.0, 0.0)))
        return f
    in_band = (t < 270.16) & (t > 265.16)
    fmult = jnp.where(in_band, _fmult(t), 0.0)

    # snow splinter
    snow_split = (qs >= 0.1e-3) & ((qc >= 0.5e-3) | (qr >= 0.1e-3)) \
        & ((psacws > 0.0) | (pracs > 0.0)) & in_band
    nmults = jnp.where(snow_split & (psacws > 0.0), 35.0e4 * psacws * fmult * 1000.0, 0.0)
    qmults = nmults * C.MMULT
    qmults = jnp.where(snow_split & (psacws > 0.0), jnp.minimum(qmults, psacws), 0.0)
    psacws = jnp.where(snow_split & (psacws > 0.0), psacws - qmults, psacws)
    nmultr = jnp.where(snow_split & (pracs > 0.0), 35.0e4 * pracs * fmult * 1000.0, 0.0)
    qmultr = nmultr * C.MMULT
    qmultr = jnp.where(snow_split & (pracs > 0.0), jnp.minimum(qmultr, pracs), 0.0)
    pracs = jnp.where(snow_split & (pracs > 0.0), pracs - qmultr, pracs)

    # graupel splinter
    g_split = (qg >= 0.1e-3) & ((qc >= 0.5e-3) | (qr >= 0.1e-3)) \
        & ((psacwg > 0.0) | (pracg > 0.0)) & in_band
    nmultg = jnp.where(g_split & (psacwg > 0.0), 35.0e4 * psacwg * fmult * 1000.0, 0.0)
    qmultg = nmultg * C.MMULT
    qmultg = jnp.where(g_split & (psacwg > 0.0), jnp.minimum(qmultg, psacwg), 0.0)
    psacwg = jnp.where(g_split & (psacwg > 0.0), psacwg - qmultg, psacwg)
    nmultrg = jnp.where(g_split & (pracg > 0.0), 35.0e4 * pracg * fmult * 1000.0, 0.0)
    qmultrg = nmultrg * C.MMULT
    qmultrg = jnp.where(g_split & (pracg > 0.0), jnp.minimum(qmultrg, pracg), 0.0)
    pracg = jnp.where(g_split & (pracg > 0.0), pracg - qmultrg, pracg)

    # ---- conversion of rimed cloud water on snow to graupel (PGSACW) ----
    pgsacw_cond = (psacws > 0.0) & (qs >= 0.1e-3) & (qc >= 0.5e-3)
    pgsacw = jnp.where(
        pgsacw_cond,
        jnp.minimum(psacws, C.CONS17 * dt * n0s * qc * qc * asn * asn
                    / (rho * lams ** (2.0 * C.BS + 2.0) + eps)), 0.0)
    dum_emb = jnp.maximum(C.RHOSN / (C.RHOG - C.RHOSN) * pgsacw, 0.0)
    nscng = jnp.where(pgsacw_cond, dum_emb / C.MG0 * rho, 0.0)
    nscng = jnp.where(pgsacw_cond, jnp.minimum(nscng, ns / dt), 0.0)
    psacws = jnp.where(pgsacw_cond, psacws - pgsacw, psacws)

    # ---- conversion of rimed rain on snow to graupel (PGRACS) ----
    pgracs_cond = (pracs > 0.0) & (qs >= 0.1e-3) & (qr >= 0.1e-3)
    dum_fr = (C.CONS18 * (4.0 / (lams + eps)) ** 3 * (4.0 / (lams + eps)) ** 3) \
        / (C.CONS18 * (4.0 / (lams + eps)) ** 3 * (4.0 / (lams + eps)) ** 3
           + C.CONS19 * (4.0 / (lamr + eps)) ** 3 * (4.0 / (lamr + eps)) ** 3 + eps)
    dum_fr = jnp.clip(dum_fr, 0.0, 1.0)
    pgracs = jnp.where(pgracs_cond, (1.0 - dum_fr) * pracs, 0.0)
    ngracs = jnp.where(pgracs_cond, (1.0 - dum_fr) * npracs, 0.0)
    ngracs = jnp.where(pgracs_cond, jnp.minimum(ngracs, nr / dt), ngracs)
    ngracs = jnp.where(pgracs_cond, jnp.minimum(ngracs, ns / dt), ngracs)
    pracs = jnp.where(pgracs_cond, pracs - pgracs, pracs)
    npracs = jnp.where(pgracs_cond, npracs - ngracs, npracs)
    psacr = jnp.where(pgracs_cond, psacr * (1.0 - dum_fr), psacr)

    # ---- freezing of rain (Bigg, T < 269.15) -> MNUCCR / NNUCCR ----
    rain_frz = (t < 269.15) & (qr >= QSMALL)
    lamr_safe = jnp.where(lamr > 0, lamr, one)
    mnuccr = jnp.where(rain_frz, C.CONS20 * nr * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0)
                       / lamr_safe ** 3 / lamr_safe ** 3, 0.0)
    nnuccr = jnp.where(rain_frz, PI * nr * C.BIMM * (jnp.exp(C.AIMM * (273.15 - t)) - 1.0)
                       / lamr_safe ** 3, 0.0)
    nnuccr = jnp.where(rain_frz, jnp.minimum(nnuccr, nr / dt), 0.0)

    # ---- accretion of cloud water by rain (PRA) ----
    qrqc8 = (qr >= 1.0e-8) & (qc >= 1.0e-8)
    pra = jnp.where(qrqc8, 67.0 * (qc * qr) ** 1.15, 0.0)
    npra = jnp.where(qrqc8, pra / (qc / jnp.where(nc > 0, nc, one)), 0.0)

    # ---- self-collection of rain (NRAGG) ----
    inv_lamr = jnp.where(lamr > 0, 1.0 / lamr_safe, 0.0)
    br_dum = jnp.where(inv_lamr < 300.0e-6, 1.0,
                       2.0 - jnp.exp(2300.0 * (inv_lamr - 300.0e-6)))
    nragg = jnp.where(qr8, -5.78 * br_dum * nr * qr * rho, 0.0)

    # ---- autoconversion of cloud ice to snow (NPRCI, PRCI) ----
    ice_aut = qi8 & (qvqvsi >= 1.0)
    nprci = jnp.where(ice_aut, C.CONS21 * (qv - qvi) * rho * n0i
                      * jnp.exp(-lami * C.DCS) * dv / abi, 0.0)
    prci = jnp.where(ice_aut, C.CONS22 * nprci, 0.0)
    nprci = jnp.where(ice_aut, jnp.minimum(nprci, ni / dt), 0.0)

    # ---- accretion of cloud ice by snow (PRAI) ----
    sn_qi = qs8 & (qi >= QSMALL)
    prai = jnp.where(sn_qi, C.CONS23 * asn * qi * rho * n0s
                     / (lams ** (C.BS + 3.0) + eps), 0.0)
    nprai = jnp.where(sn_qi, C.CONS23 * asn * ni * rho * n0s
                      / (lams ** (C.BS + 3.0) + eps), 0.0)
    nprai = jnp.where(sn_qi, jnp.minimum(nprai, ni / dt), 0.0)

    # ---- collision of rain and ice -> snow or graupel (PIACR/PRACI[S]) ----
    ri = (qr >= 1.0e-8) & (qi >= 1.0e-8) & (t <= 273.15)
    big_rain = ri & (qr >= 0.1e-3)
    small_rain = ri & (qr < 0.1e-3)
    niacr = jnp.where(big_rain, C.CONS24 * ni * n0rr * arn
                      / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    piacr = jnp.where(big_rain, C.CONS25 * ni * n0rr * arn
                      / (lamr ** (C.BR + 3.0) + eps) / (lamr ** 3 + eps) * rho, 0.0)
    praci = jnp.where(big_rain, C.CONS24 * qi * n0rr * arn
                      / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    niacr = jnp.where(big_rain, jnp.minimum(jnp.minimum(niacr, nr / dt), ni / dt), niacr)
    niacrs = jnp.where(small_rain, C.CONS24 * ni * n0rr * arn
                       / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    piacrs = jnp.where(small_rain, C.CONS25 * ni * n0rr * arn
                       / (lamr ** (C.BR + 3.0) + eps) / (lamr ** 3 + eps) * rho, 0.0)
    pracis = jnp.where(small_rain, C.CONS24 * qi * n0rr * arn
                       / (lamr ** (C.BR + 3.0) + eps) * rho, 0.0)
    niacrs = jnp.where(small_rain, jnp.minimum(jnp.minimum(niacrs, nr / dt), ni / dt), niacrs)

    # ---- ice nucleation (INUC=0, Cooper/Rasmussen) ----
    nuc_cond = ((qvqvs >= 0.999) & (t <= 265.15)) | (qvqvsi >= 1.08)
    kc2 = jnp.minimum(0.005 * jnp.exp(0.304 * (273.15 - t)) * 1000.0, 500.0e3)
    kc2 = jnp.maximum(kc2 / rho, 0.0)
    do_nuc = nuc_cond & (kc2 > (ni + ns + ng))
    nnuccd = jnp.where(do_nuc, (kc2 - ni - ns - ng) / dt, 0.0)
    mnuccd = jnp.where(do_nuc, nnuccd * C.MI0, 0.0)

    # ---- evap/sub/dep terms for qi, qni, qg, qr ----
    epsi = jnp.where(qi >= QSMALL, 2.0 * PI * n0i * rho * dv / (lami * lami + eps), 0.0)
    epss = jnp.where(qs8 | (qs >= QSMALL),
                     2.0 * PI * n0s * rho * dv
                     * (C.F1S / (lams * lams + eps)
                        + C.F2S * (asn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                        * C.CONS10 / (lams ** C.CONS35 + eps)), 0.0)
    epss = jnp.where(qs >= QSMALL, epss, 0.0)
    epsg = jnp.where(qg >= QSMALL,
                     2.0 * PI * n0g * rho * dv
                     * (C.F1S / (lamg * lamg + eps)
                        + C.F2S * (agn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                        * C.CONS11 / (lamg ** C.CONS36 + eps)), 0.0)
    epsr = jnp.where(qr >= QSMALL,
                     2.0 * PI * n0rr * rho * dv
                     * (C.F1R / (lamr * lamr + eps)
                        + C.F2R * (arn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                        * C.CONS9 / (lamr ** C.CONS34 + eps)), 0.0)

    qi_ge = qi >= QSMALL
    dum_frac = jnp.where(qi_ge, (1.0 - jnp.exp(-lami * C.DCS) * (1.0 + lami * C.DCS)), 0.0)
    prd = jnp.where(qi_ge, epsi * (qv - qvi) / abi * dum_frac, 0.0)
    qs_ge = qs >= QSMALL
    prds = jnp.where(qs_ge, epss * (qv - qvi) / abi
                     + epsi * (qv - qvi) / abi * (1.0 - dum_frac), 0.0)
    # if snow absent, the (1-dum) ice deposition goes to ice
    prd = jnp.where(qi_ge & (~qs_ge), prd + epsi * (qv - qvi) / abi * (1.0 - dum_frac), prd)
    prdg = epsg * (qv - qvi) / abi
    pre = jnp.where(qv < qvs, jnp.minimum(epsr * (qv - qvs) / ab, 0.0), 0.0)

    # FUDGEF clamp on deposition
    dum_dep = (qv - qvi) / dt
    fudgef = 0.9999
    sum_dep = prd + prds + mnuccd + prdg
    clamp = ((dum_dep > 0.0) & (sum_dep > dum_dep * fudgef)) \
        | ((dum_dep < 0.0) & (sum_dep < dum_dep * fudgef))
    scale = jnp.where(clamp, fudgef * dum_dep / jnp.where(sum_dep != 0, sum_dep, one), 1.0)
    mnuccd = jnp.where(clamp, mnuccd * scale, mnuccd)
    prd = jnp.where(clamp, prd * scale, prd)
    prds = jnp.where(clamp, prds * scale, prds)
    prdg = jnp.where(clamp, prdg * scale, prdg)

    # negative dep -> sublimation
    eprd = jnp.where(prd < 0.0, prd, 0.0)
    prd = jnp.where(prd < 0.0, 0.0, prd)
    eprds = jnp.where(prds < 0.0, prds, 0.0)
    prds = jnp.where(prds < 0.0, 0.0, prds)
    eprdg = jnp.where(prdg < 0.0, prdg, 0.0)
    prdg = jnp.where(prdg < 0.0, 0.0, prdg)

    # ---- conservation of water (negative-process adjustment) ----
    # QC
    dum = (prc + pra + mnuccc + psacws + psacwi + qmults + psacwg + pgsacw + qmultg) * dt
    cqc = (dum > qc) & (qc >= QSMALL)
    rr = jnp.where(cqc, qc / jnp.where(dum != 0, dum, one), 1.0)
    prc = prc * rr; pra = pra * rr; mnuccc = mnuccc * rr; psacws = psacws * rr
    psacwi = psacwi * rr; qmults = qmults * rr; qmultg = qmultg * rr
    psacwg = psacwg * rr; pgsacw = pgsacw * rr

    # QI
    dum = (-prd - mnuccc + prci + prai - qmults - qmultg - qmultr - qmultrg
           - mnuccd + praci + pracis - eprd - psacwi) * dt
    cqi = (dum > qi) & (qi >= QSMALL)
    denom_i = (prci + prai + praci + pracis - eprd)
    ri_ = jnp.where(cqi & (denom_i != 0),
                    (qi / dt + prd + mnuccc + qmults + qmultg + qmultr + qmultrg
                     + mnuccd + psacwi) / jnp.where(denom_i != 0, denom_i, one), 1.0)
    prci = jnp.where(cqi, prci * ri_, prci)
    prai = jnp.where(cqi, prai * ri_, prai)
    praci = jnp.where(cqi, praci * ri_, praci)
    pracis = jnp.where(cqi, pracis * ri_, pracis)
    eprd = jnp.where(cqi, eprd * ri_, eprd)

    # QR
    dum = ((pracs - pre) + (qmultr + qmultrg - prc) + (mnuccr - pra)
           + piacr + piacrs + pgracs + pracg) * dt
    cqr = (dum > qr) & (qr >= QSMALL)
    denom_r = (-pre + qmultr + qmultrg + pracs + mnuccr + piacr + piacrs + pgracs + pracg)
    rr2 = jnp.where(cqr & (denom_r != 0),
                    (qr / dt + prc + pra) / jnp.where(denom_r != 0, denom_r, one), 1.0)
    pre = jnp.where(cqr, pre * rr2, pre)
    pracs = jnp.where(cqr, pracs * rr2, pracs)
    qmultr = jnp.where(cqr, qmultr * rr2, qmultr)
    qmultrg = jnp.where(cqr, qmultrg * rr2, qmultrg)
    mnuccr = jnp.where(cqr, mnuccr * rr2, mnuccr)
    piacr = jnp.where(cqr, piacr * rr2, piacr)
    piacrs = jnp.where(cqr, piacrs * rr2, piacrs)
    pgracs = jnp.where(cqr, pgracs * rr2, pgracs)
    pracg = jnp.where(cqr, pracg * rr2, pracg)

    # QNI (snow), IGRAUP=0
    dum = (-prds - psacws - prai - prci - pracs - eprds + psacr - piacrs - pracis) * dt
    cqs = (dum > qs) & (qs >= QSMALL)
    denom_s = (-eprds + psacr)
    rs2 = jnp.where(cqs & (denom_s != 0),
                    (qs / dt + prds + psacws + prai + prci + pracs + piacrs + pracis)
                    / jnp.where(denom_s != 0, denom_s, one), 1.0)
    eprds = jnp.where(cqs, eprds * rs2, eprds)
    psacr = jnp.where(cqs, psacr * rs2, psacr)

    # QG
    dum = (-psacwg - pracg - pgsacw - pgracs - prdg - mnuccr - eprdg - piacr
           - praci - psacr) * dt
    cqg = (dum > qg) & (qg >= QSMALL)
    rg2 = jnp.where(cqg & (eprdg != 0),
                    (qg / dt + psacwg + pracg + pgsacw + pgracs + prdg + mnuccr
                     + psacr + piacr + praci) / jnp.where(eprdg != 0, -eprdg, one), 1.0)
    eprdg = jnp.where(cqg, eprdg * rg2, eprdg)

    # ---- tendencies ----
    qv_t = (-pre - prd - prds - mnuccd - eprd - eprds - prdg - eprdg)
    t_t = (pre * xxlv
           + (prd + prds + mnuccd + eprd + eprds + prdg + eprdg) * xxls
           + (psacws + psacwi + mnuccc + mnuccr + qmults + qmultg + qmultr + qmultrg
              + pracs + psacwg + pracg + pgsacw + pgracs + piacr + piacrs) * xlf) / cpm
    # PCC (saturation adjustment) is applied AFTER both branches in the caller,
    # so the cold-branch QC tendency here omits the +PCC term that the Fortran
    # adds inline; the caller adds PCC to qc_ten/qv_ten/t_ten for all do_cell.
    qc_t = (-pra - prc - mnuccc
            - psacws - psacwi - qmults - qmultg - psacwg - pgsacw)
    qi_t = (prd + eprd + psacwi + mnuccc - prci - prai
            + qmults + qmultg + qmultr + qmultrg + mnuccd - praci - pracis)
    qr_t = (pre + pra + prc - pracs - mnuccr - qmultr - qmultrg
            - piacr - piacrs - pracg - pgracs)
    qni_t = (prai + psacws + prds + pracs + prci + eprds - psacr + piacrs + pracis)
    ns_t = (nsagg + nprci - nscng - ngracs + niacrs)
    qg_t = (pracg + psacwg + pgsacw + pgracs + prdg + eprdg + mnuccr + piacr
            + praci + psacr)
    ng_t = (nscng + ngracs + nnuccr + niacr)
    nc_t = (-nnuccc - npsacws - npra - nprc - npsacwi - npsacwg)
    ni_t = (nnuccc - nprci - nprai + nmults + nmultg + nmultr + nmultrg
            + nnuccd - niacr - niacrs)
    nr_t = (nprc1 - npracs - nnuccr + nragg - niacr - niacrs - npracg - ngracs)

    # number sublimation (NSUBI/NSUBS/NSUBR/NSUBG)
    nsubi = jnp.where(eprd < 0.0, jnp.maximum(-1.0, eprd * dt / jnp.where(qi > 0, qi, one))
                      * ni / dt, 0.0)
    nsubs = jnp.where(eprds < 0.0, jnp.maximum(-1.0, eprds * dt / jnp.where(qs > 0, qs, one))
                      * ns / dt, 0.0)
    nsubr = jnp.where(pre < 0.0, jnp.maximum(-1.0, pre * dt / jnp.where(qr > 0, qr, one))
                      * nr / dt, 0.0)
    nsubg = jnp.where(eprdg < 0.0, jnp.maximum(-1.0, eprdg * dt / jnp.where(qg > 0, qg, one))
                      * ng / dt, 0.0)
    ni_t = ni_t + nsubi
    ns_t = ns_t + nsubs
    ng_t = ng_t + nsubg
    nr_t = nr_t + nsubr

    return (qv_t, t_t, qc_t, qr_t, qi_t, qni_t, qg_t,
            nc_t, ni_t, ns_t, nr_t, ng_t,
            ni_c, ns_c, nr_c, ng_c)
