"""NSSL 2-moment droplet nucleation / condensation / evaporation: ``NUCOND`` + ``QVEXCESS``.

Port of ``phys/module_mp_nssl_2mom.F`` SUBROUTINE NUCOND (lines 9749-12044) and SUBROUTINE
QVEXCESS (lines 6178-6335) for the WRF-default mp=18 configuration, called by the driver right
after ``nssl_2mom_gs`` (driver lines 3246-3255).

Configuration that is fixed here (module defaults, see ``constants``): ipconc=5, iqvsopt=1 (Bolton),
eqtset=1, rcond=2, imurain=1, iferwisventr=2, irenuc=5 (activated CCN ``lccna``), ac_opt=0,
iqcinit=2, iccwflg=1, imaxsupopt=4, restoreccn=.true., lss=0, lzr=0, nxtra=1, flag_qndrop=.false.
Dead branches dropped: ``lzr>1`` rain-Z block (lzr=0), irenuc /= 5 nucleation variants, ac_opt>0
aerosol paths, eqtset>1 latent-heat set, axtra diagnostics (nxtra=1), ``lss>1`` ssmax storage,
``renucfrac >= 0.999`` hack (renucfrac=0 because the CCN field is on).

Interface (vectorised; ``an`` is the Fortran-indexed state stack (NA+1, ..., nz), density-scaled
number/volume moments as inside the driver):

    nucond(an, dn, t77, pn, w, dtp, C, prec, flag_qndrop=False) -> (an_new, t0_new, ssat)

* NUCOND writes ``t0 = an(lt)*t77`` for every point (before any update) and the driver work array
  ``ssat`` (= dummy ``ssfilt``) = 100*(qv/qvs - 1) for every point; it never reads/writes ``t9``,
  ``t00`` (iqvsopt=1) or ``dz3d`` (only in the irenuc<2 branch), so they are not arguments.
* ``an`` entries written: lt, lv, lc, lr, lnc, lnr, lccna — only at "gathered" points.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from .indices import LT, LV, LC, LR, LNC, LNR, LCCN, LCCNA
from .satfun import dtabqvs, tabqvs

SUPCB = 0.5
SUPMX = 238.0


def _ltemq(temp, C, R):
    """``ltemq = (temp-163.15)/fqsat + 1.5`` assigned to INTEGER (truncation), clamped [1, nqsat]."""
    l = jnp.trunc((temp - jnp.asarray(163.15, R)) / C.fqsat + jnp.asarray(1.5, R)).astype(jnp.int32)
    return jnp.minimum(C.nqsat, jnp.maximum(1, l))


def _ltemq_inv(temp, C, R):
    """``(temp-163.15)*fqsati + 1.5`` variant used inside the RK2c condensation loop."""
    l = jnp.trunc((temp - jnp.asarray(163.15, R)) * C.fqsati + jnp.asarray(1.5, R)).astype(jnp.int32)
    return jnp.minimum(C.nqsat, jnp.maximum(1, l))


def _qvs_bolton(temp, pres, C, R):
    l = _ltemq(temp, C, R)
    tq = tabqvs(l, C, R)
    return C.rdorv * C.esbolton * tq / (pres - C.esbolton * tq), l


def qvexcess(qwvp0, qv0, qcw1, pres, thetap0, theta0, pi0, fcqv1, felvcp, ss1, pk, C, R):
    """SUBROUTINE QVEXCESS (module line 6178): condensate needed to bring vapour down to
    ``(1 + 0.01*ss1)*qvs`` (two fixed iterations); returns ``qvex = max(0, qcw - qcw1)``."""
    tfr = jnp.asarray(273.15, R)  # local PARAMETER tfr
    thetap = thetap0
    theta = thetap + theta0
    qwvp = qwvp0
    qvap = jnp.maximum(qwvp0 + qv0, 0.0)
    temg = theta * pk
    qwv = jnp.maximum(0.0, qvap)
    qcw = jnp.maximum(0.0, qcw1)
    del tfr
    qvs, _ = _qvs_bolton(temg, pres, C, R)
    qss = (0.01 * ss1 + 1.0) * qvs
    for _itertd in range(2):
        dqcw = jnp.zeros_like(qcw)
        dqwv = qwv - qss
        sub = dqwv < 0.0
        enough = qcw > -dqwv
        dqcw_s = jnp.where(enough, dqwv, -qcw)
        dqwv_s = jnp.where(enough, jnp.zeros_like(dqwv), dqwv + qcw)
        dqcw = jnp.where(sub, dqcw_s, dqcw)
        dqwv = jnp.where(sub, dqwv_s, dqwv)
        qwvp = jnp.where(sub, qwvp - dqcw, qwvp)
        qcw = jnp.where(sub, qcw + dqcw, qcw)
        thetap = jnp.where(sub, thetap + 1. / pi0 * (felvcp * dqcw), thetap)
        cond = dqwv >= 0.0
        dqvcnd = dqwv / (1. + fcqv1 * qss / ((temg - C.cbw) ** 2))
        dqcw = jnp.where(cond, dqvcnd, dqcw)
        thetap = jnp.where(cond, thetap + (felvcp * dqcw) / (pi0), thetap)
        qwvp = jnp.where(cond, qwvp - (dqvcnd), qwvp)
        qcw = jnp.where(cond, qcw + dqcw, qcw)
        theta = thetap + theta0
        temg = theta * pk
        qvap = jnp.maximum((qwvp + qv0), 0.0)
        qvs, _ = _qvs_bolton(temg, pres, C, R)
        qcw = jnp.maximum(0.0, qcw)
        qwv = jnp.maximum(0.0, qvap)
        qss = (0.01 * ss1 + 1.0) * qvs
    return jnp.maximum(0.0, qcw - qcw1)


def _kidx(nz, offset, lo, hi):
    return jnp.clip(jnp.arange(nz) + offset, lo, hi)


def nucond(an, dn, t77, pn, w, dtp, C, prec, flag_qndrop=False):
    if flag_qndrop:
        raise NotImplementedError("flag_qndrop (WRF-Chem droplet number) is not part of the default mp=18 port")
    R = prec.R
    dtp = jnp.asarray(dtp, R)
    an = jnp.asarray(an, R)
    nz = an.shape[-1]
    zero = jnp.zeros(an.shape[1:], R)

    # ------------------------------------------------------------------ pre-gather (10030-10050)
    temp1 = an[LT] * t77
    t0 = temp1
    c1, _ = _qvs_bolton(temp1, pn, C, R)  # pb = 0
    ssfilt = jnp.where(c1 > 0., 100. * (an[LV] / c1 - 1.0), zero)

    # ------------------------------------------------------------------ gather mask (10082-10118)
    temg = t0
    qvs, ltemq = _qvs_bolton(temg, pn, C, R)
    qss = qvs
    gathered = ((temg > C.tfrh) | (an[LV] / qvs > C.maxlowtempss)) & (
        (an[LV] > qss) | (an[LC] > C.qxmin[LC]) | ((an[LR] > C.qxmin[LR]) & (C.rcond == 2)))

    # ------------------------------------------------------------------ temporaries (10132-10237)
    qv = an[LV]
    qc = jnp.maximum(an[LC], 0.0)
    qr = jnp.maximum(an[LR], 0.0)
    alpha_r = jnp.full(zero.shape, C.alphar, R)  # imurain == 1
    theta0 = an[LT]
    thetap = zero
    qv0 = qv
    qwvp = qv - qv0
    pres = pn
    rho0 = dn
    rhoinv = 1.0 / rho0
    rhovt = jnp.sqrt(C.rho00 / rho0)
    pi0 = t77
    pk = t77
    pqs = (380.0) / (pres)
    del pqs
    tab = tabqvs(ltemq, C, R)
    qvap = jnp.maximum((qwvp + qv0), 0.0)
    es = C.esbolton * tab
    temgx = jnp.minimum(temg, 313.15)
    temgx = jnp.maximum(temgx, 233.15)
    felv = 2500837.367 * (273.15 / temgx) ** ((0.167) + (3.67e-4) * temgx)
    felvcp = felv * C.cpi  # eqtset <= 1
    fcqv1 = 4098.0258 * felv * C.cpi
    wvdf = (2.11e-05) * ((temg / C.tfr) ** 1.94) * (101325.0 / (pn))
    advisc = C.advisc0 * (416.16 / (temg + 120.0)) * (temg / 296.0) ** (1.5)
    tka = C.tka0 * advisc / C.advisc1

    cx_c = jnp.maximum(an[LNC], 0.0)
    cwnccn = C.cwccn * rho0 / C.rho00
    ccnc = an[LCCN]  # lccn > 1, ac_opt == 0, no UF CCN
    ccna = an[LCCNA]  # predicted activated CCN
    cx_r = jnp.maximum(an[LNR], 0.0)
    # irenuc /= 6, /= 2:
    cnuc = ccnc * (1. - C.renucfrac) + ccnc * C.renucfrac

    xdn_c = jnp.asarray(C.xdn0[LC], R)
    xdn_r = jnp.asarray(C.xdn0[LR], R)
    ventrxn = jnp.asarray(C.ventrn, R)

    # vertical neighbours (10602-10622), 0-based level index
    w_kp1 = jnp.take(w, _kidx(nz, 1, 0, nz - 1), axis=-1)
    w_km1 = jnp.take(w, _kidx(nz, -1, 0, nz - 1), axis=-1)
    wvel = (0.5) * (w_kp1 + w)
    wvelkm1 = (0.5) * (w + w_km1)
    ssat0 = ssfilt
    ssf = ssfilt
    ssfkp1 = jnp.take(ssfilt, jnp.minimum(nz - 2, jnp.arange(nz) + 1), axis=-1)
    ssfkm1 = jnp.take(ssfilt, jnp.maximum(0, jnp.arange(nz) - 1), axis=-1)
    del wvelkm1  # only used by the dead irenuc<2 branch
    kidx = jnp.broadcast_to(jnp.arange(nz), zero.shape)

    # cloud water (10632-10661)
    big = cx_c > 1.0e6  # ipconc >= 2
    cxc_safe = jnp.where(cx_c > 0., cx_c, 1.0)
    xmas_big = jnp.minimum(jnp.maximum(qc * rho0 / cxc_safe, C.cwmasn), C.cwmasx)
    okc = (qc > C.qxmin[LC]) & (cx_c > C.cxmin)
    xmas_ok = jnp.minimum(jnp.maximum(qc * rho0 / cxc_safe, xdn_c * C.xvmn[LC]), xdn_c * C.xvmx[LC])
    cx_ok = qc * rho0 / xmas_ok
    lowc = (qc > C.qxmin[LC]) & (cx_c <= C.cxmin)
    cx_low = jnp.maximum(C.cxmin, rho0 * qc / C.cwmasx)
    xmas_low = jnp.minimum(jnp.maximum(qc * rho0 / cx_low, C.cwmasn), C.cwmasx)
    xmas_c = jnp.where(big, xmas_big,
                       jnp.where(okc, xmas_ok, jnp.where(lowc, xmas_low, jnp.full(zero.shape, C.cwmasn, R))))
    cx_c = jnp.where(big, cx_c, jnp.where(okc, cx_ok, jnp.where(lowc, cx_low, cx_c)))
    xdia_c = (xmas_c * C.cwc1) ** C.c1f3

    # rain (10665-10702), ipconc >= 3
    has_r = qr > C.qxmin[LR]
    xv_r = rho0 * qr / (xdn_r * jnp.maximum(1.0e-9, cx_r))
    hi = xv_r > C.xvmx[LR]
    lo = xv_r < C.xvmn[LR]
    cx_r = jnp.where(has_r & hi, rho0 * qr / (C.xvmx[LR] * xdn_r),
                     jnp.where(has_r & lo, rho0 * qr / (C.xvmn[LR] * xdn_r), cx_r))
    xv_r = jnp.where(hi, C.xvmx[LR], jnp.where(lo, C.xvmn[LR], xv_r))
    xdia_r = jnp.where(
        has_r,
        (6. * C.piinv * xv_r / ((alpha_r + 3.) * (alpha_r + 2.) * (alpha_r + 1.))) ** (1. / 3.),
        jnp.full(zero.shape, 1.e-9, R))

    # ventilation (10708-10723)
    fadvisc = C.advisc0 * (416.16 / (temg + 120.0)) * (temg / 296.0) ** (1.5)
    fakvisc = fadvisc * rhoinv
    fwvdf = (2.11e-05) * ((temg / C.tfr) ** 1.94) * (101325.0 / (pres))
    fschm = (fakvisc / fwvdf)
    fvent = (fschm ** (1. / 3.)) * (fakvisc ** (-0.5))

    # ================================================================== main point loop (10732-11930)
    # CYCLE at 10736 cannot trigger for gathered points (complement of the gather test) but is kept.
    active = gathered & ~((temg <= C.tfrh) & (qv / qvs < C.maxlowtempss))
    supersat = (ssat0 > 0.) | (ssf > 0.)

    # ---------------- (a) sub-saturated: evaporation of cloud water (10746-10850)
    ev = active & ~supersat & ~(qc <= 0.)
    R1 = 1. / (1. + C.caw * (273.15 - C.cbw) * qss * felv / (C.cp * (temg - C.cbw) ** 2))
    QEVAP = jnp.minimum(qc, R1 * (qss - qvap))
    full = qc <= QEVAP
    qctmp = qc
    # full evaporation
    qwvp_f = qwvp + qc
    thetap_f = thetap - felvcp * qc / (pi0)
    ccna_f = ccna - C.restoreccnfrac * cx_c
    # partial evaporation
    qwvp_p = qwvp + QEVAP
    qc_p = qc - QEVAP
    gone = qc_p <= 0.
    tmp_p = 0.9 * QEVAP * cx_c / jnp.where(qctmp != 0., qctmp, 1.0)
    tmp_p = C.restoreccnfrac * tmp_p
    ccna_p = jnp.where(gone, ccna - C.restoreccnfrac * cx_c, ccna - tmp_p)
    cxc_p = jnp.where(gone, zero, cx_c - tmp_p)
    thetap_p = thetap - felvcp * QEVAP / (pi0)
    qwvp_ev = jnp.where(full, qwvp_f, qwvp_p)
    thetap_ev = jnp.where(full, thetap_f, thetap_p)
    qc_ev = jnp.where(full, zero, qc_p)
    ccna_ev = jnp.where(full, ccna_f, ccna_p)
    cxc_ev = jnp.where(full, zero, cxc_p)

    # ---------------- (b) super-saturated with cloud: condensation (10857-11155)
    cd = active & supersat & (qc > C.qxmin[LC]) & (cx_c >= 1.)
    ac1 = felv ** 2 / (tka * C.rw * temg ** 2)
    bc = C.rw * temg / (wvdf * es)
    xdia_c_cd = jnp.where(xdia_c <= 0.0, (C.cwmasn * C.cwc1) ** C.c1f3, xdia_c)
    d1 = jnp.where(qc > C.qxmin[LC],
                   (1. / (ac1 + bc)) * 4.0 * C.pi * C.ventc * 0.5 * xdia_c_cd * cx_c * rhoinv, zero)
    rain_ok = (C.rcond == 2) & (qr > C.qxmin[LR]) & (cx_r > 1.e-9)
    x_r = 1. + alpha_r
    rwvent = (0.78 * x_r + 0.308 * ventrxn * fvent * jnp.sqrt((C.ar * rhovt))
              * (xdia_r ** ((1.0 + C.br) / 2.0)))
    d1r = jnp.where(rain_ok, (1. / (ac1 + bc)) * 4.0 * C.pi * rwvent * 0.5 * xdia_r * cx_r * rhoinv, zero)
    e1 = felvcp / (pi0)
    f1 = pk
    p380 = C.esbolton * C.rdorv / (pres - es)
    ss1 = qv / qvs
    qv1 = qv
    qvs1 = qvs
    dd = d1 * (ss1 - 1.0)
    delta = jnp.where(jnp.abs(ss1 - 1.0) > 1.e-5,
                      0.5 * (qv1 - qvs1) / jnp.where(cd & (dd != 0.), dd, 1.0), 0.1 * dtp)
    delta = jnp.where(cd, delta, 1.0)
    dtcon1 = jnp.minimum(0.05, 0.2 * delta)
    xr_ = (dtp - 4.0 * dtcon1) / delta
    nint_ = jnp.where(xr_ >= 0., jnp.floor(xr_ + 0.5), jnp.ceil(xr_ - 0.5))
    nint_ = jnp.clip(nint_, -1.0e9, 1.0e9).astype(jnp.int32)
    nc = jnp.maximum(5, 2 * nint_)
    dtcon2 = (dtp - 4.0 * dtcon1) / nc.astype(R)

    def _step_init():
        return dict(dt1=zero, n=jnp.ones(zero.shape, jnp.int32), qv1=qv1, qvs1=qvs1, ss1=ss1, ss2=ss1,
                    temp1=temg, temp2=temg, dqc=zero, dqr=zero, dtcon=zero,
                    fresh=jnp.ones(zero.shape, bool), live=cd)

    def _cond(s):
        return jnp.any(s["live"])

    def _body(s):
        live = s["live"]
        dtcon = jnp.where(s["fresh"], jnp.where(s["n"] <= 4, dtcon1, dtcon2), s["dtcon"])
        # predictor (label 609)
        dqv = -(s["ss1"] - 1.) * d1 * dtcon
        dqvr = -(s["ss1"] - 1.) * d1r * dtcon
        dtemp = -0.5 * e1 * f1 * (dqv + dqvr)
        l1m = _ltemq_inv(s["temp1"] + dtemp, C, R)
        dqvs = dtemp * p380 * dtabqvs(l1m, C, R)
        qv1m = s["qv1"] + dqv + dqvr
        qvs1m = s["qvs1"] + dqvs
        ss1m = qv1m / qvs1m
        under = ss1m < 1.  # (dqvii + dqvis) == 0 always (no ice in NUCOND)
        dtcon_h = (0.5 * dtcon)
        retry = under & (dtcon_h >= dtcon1)
        quit_ = under & ~retry
        # corrector
        dqv = -(ss1m - 1.) * d1 * dtcon
        dqvr = -(ss1m - 1.) * d1r * dtcon
        dtemp = -e1 * f1 * (dqv + dqvr)
        l1 = _ltemq_inv(s["temp1"] + dtemp, C, R)
        dqvs = dtemp * p380 * dtabqvs(l1, C, R)
        qv1n = s["qv1"] + dqv + dqvr
        dqcn = s["dqc"] - dqv
        dqrn = s["dqr"] - dqvr
        qvs1n = s["qvs1"] + dqvs
        ss1n = qv1n / qvs1n
        temp1n = s["temp1"] + dtemp
        stop = ((s["temp2"] == temp1n) | (s["ss2"] == ss1n) | (ss1n == 1.00)
                | ((s["n"] > 10) & (ss1n < 1.0005)))
        upd = live & ~under
        adv = upd & ~stop
        dt1n = jnp.where(adv, s["dt1"] + dtcon, s["dt1"])
        out = dict(
            dt1=dt1n,
            n=jnp.where(adv, s["n"] + 1, s["n"]),
            qv1=jnp.where(upd, qv1n, s["qv1"]),
            qvs1=jnp.where(upd, qvs1n, s["qvs1"]),
            ss1=jnp.where(upd, ss1n, s["ss1"]),
            ss2=jnp.where(adv, ss1n, s["ss2"]),
            temp1=jnp.where(upd, temp1n, s["temp1"]),
            temp2=jnp.where(adv, temp1n, s["temp2"]),
            dqc=jnp.where(upd, dqcn, s["dqc"]),
            dqr=jnp.where(upd, dqrn, s["dqr"]),
            dtcon=jnp.where(live & retry, dtcon_h, dtcon),
            fresh=jnp.where(live & retry, False, True),
            live=live & ~quit_ & ~(upd & stop) & ~(adv & ~(dt1n < dtp)),
        )
        return out

    st = lax.while_loop(_cond, _body, _step_init())
    dcloud_cd = st["dqc"]
    dqr_cd = st["dqr"]
    thetap_cd = thetap + e1 * (dcloud_cd + dqr_cd)
    qwvp_cd = qwvp - (dcloud_cd + dqr_cd)
    qc_cd = qc + dcloud_cd
    qr_cd = qr + dqr_cd
    temg_cd = (thetap_cd + theta0) * f1
    qvs_cd, _ = _qvs_bolton(temg_cd, pres, C, R)

    # nucleation after condensation (11295-11807), irenuc == 5
    wpos = wvel > 0.
    to613 = wpos & ((cx_c <= 0.) | ((kidx >= 1) & (qc_cd <= C.qxmin[LC])) | (kidx == 0))
    skip631 = ((ssf <= SUPCB) & wpos) | (
        (kidx >= 1) & (kidx < nz - 2) & ((ssfkp1 >= SUPMX) | (ssf >= SUPMX) | (ssfkm1 >= SUPMX))) | (
        (ssf < 1.E-10) | (ssf >= SUPMX))
    nuc = cd & ~to613 & ~skip631
    expo = (2. / (2. + C.cck))
    CN = jnp.minimum(cnuc, C.CCNE0 * cnuc ** expo * jnp.maximum(0.0, wvel) ** C.cnexp)
    depleted = ccna >= cnuc
    temp1_n = (theta0 + thetap_cd) * pk
    c1n, _ = _qvs_bolton(temp1_n, pres, C, R)
    ssf_n = jnp.where(c1n > 0., jnp.maximum(0.0, 100. * ((qv0 + qwvp_cd) / jnp.where(c1n > 0., c1n, 1.0) - 1.0)),
                      zero)
    CN_dep = jnp.maximum(CN, cnuc * jnp.minimum(C.ssf2kmax, ssf_n ** C.cck))
    CN_dep = jnp.maximum(0.0, CN_dep - ccna)
    CN = jnp.where(depleted, CN_dep, jnp.minimum(CN, cnuc - ccna))
    dcrit = jnp.asarray(2.0 * 2.0e-6, R)
    dcl = 1000. * dcrit ** 3 * C.pi / 6.
    tmpn = jnp.maximum(0.0, rho0 * qc_cd / dcl - cx_c)
    cn_nuc = jnp.minimum(tmpn, CN)
    grow = cn_nuc > 0.0
    dcrit2 = jnp.asarray(2.5e-7, R)
    dcl2 = 1000. * dcrit2 ** 3 * C.pi / 6. * cn_nuc
    cxc_nuc = jnp.where(grow, cx_c + cn_nuc, cx_c)
    qc_nuc = jnp.where(grow, qc_cd + dcl2, qc_cd)
    thetap_nuc = jnp.where(grow, thetap_cd + felvcp * dcl2 / (pi0), thetap_cd)
    qwvp_nuc = jnp.where(grow, qwvp_cd - dcl2, qwvp_cd)
    ccna_nuc = ccna + cn_nuc

    thetap_b = jnp.where(nuc, thetap_nuc, thetap_cd)
    qwvp_b = jnp.where(nuc, qwvp_nuc, qwvp_cd)
    qc_b = jnp.where(nuc, qc_nuc, qc_cd)
    cxc_b = jnp.where(nuc, cxc_nuc, cx_c)
    ccna_b = jnp.where(nuc, ccna_nuc, ccna)

    # ---------------- (c) super-saturated without (enough) cloud: initial condensation (11158-11284)
    ic = active & supersat & ~cd
    ssmx_i = jnp.asarray(C.ssmxinit, R)  # iqcinit == 2
    do_qvex = (ssf > 0.0) & (ssf > ssmx_i) & (ssf < 20.0) & (ccnc > 0.05 * cwnccn)
    dcloud_i = jnp.where(do_qvex,
                         qvexcess(qwvp, qv0, qc, pres, thetap, theta0, pi0, fcqv1, felvcp, ssmx_i, pk, C, R),
                         zero)
    thetap_i = thetap + felvcp * dcloud_i / (pi0)
    qwvp_i = qwvp - dcloud_i
    qc_i = qc + dcloud_i
    temg_i = (thetap_i + theta0) * pk
    qvs_i, _ = _qvs_bolton(temg_i, pres, C, R)
    act = (dcloud_i > C.qxmin[LC]) & wpos
    cn_i = C.CCNE0 * cnuc ** expo * jnp.where(wpos, wvel, 1.0) ** C.cnexp
    cn_i = jnp.minimum(C.cwccn * rho0 / C.rho00,
                       jnp.maximum(cn_i, rho0 * qc_i / (xdn_c * (4. * C.pi / 3.) * (4.e-6) ** 3)))  # iccwflg == 1
    cn_i = jnp.where(act, cn_i, zero)
    cn_i = jnp.where((cn_i > 0.0) & (cn_i > ccnc), ccnc, cn_i)  # ac_opt == 0
    ccna_i = jnp.where(cn_i > 0.0, ccna + cn_i, ccna)
    cxc_i = jnp.where(cn_i > cx_c, cn_i, cx_c)
    cxc_i = jnp.where((cxc_i > 0.) & (qc_i <= C.qxmin[LC]), zero,
                      jnp.minimum(cxc_i, rho0 * jnp.maximum(0.0, qc_i) / C.cwmasn))

    # ---------------- merge branches
    thetap_m = jnp.where(ev, thetap_ev, jnp.where(cd, thetap_b, jnp.where(ic, thetap_i, thetap)))
    qwvp_m = jnp.where(ev, qwvp_ev, jnp.where(cd, qwvp_b, jnp.where(ic, qwvp_i, qwvp)))
    qc_m = jnp.where(ev, qc_ev, jnp.where(cd, qc_b, jnp.where(ic, qc_i, qc)))
    qr_m = jnp.where(cd, qr_cd, qr)
    cxc_m = jnp.where(ev, cxc_ev, jnp.where(cd, cxc_b, jnp.where(ic, cxc_i, cx_c)))
    ccna_m = jnp.where(ev, ccna_ev, jnp.where(cd, ccna_b, jnp.where(ic, ccna_i, ccna)))
    qvs_m = jnp.where(cd, qvs_cd, jnp.where(ic, qvs_i, qvs))

    # ---------------- label 631: cap at maxsupersat (11817-11871), imaxsupopt == 4
    qv1t = qv0 + qwvp_m
    cap = active & (qv1t > (C.maxsupersat * qvs_m))
    ssmx_c = 100. * (jnp.asarray(C.maxsupersat, R) - 1.0)
    qvex = jnp.where(cap, qvexcess(qwvp_m, qv0, qc_m, pres, thetap_m, theta0, pi0, fcqv1, felvcp, ssmx_c, pk,
                                   C, R), zero)
    pos = cap & (qvex > 0.0)
    cwmas20 = jnp.asarray(1000., R) * jnp.asarray(0.523599, R) * (jnp.asarray(2., R) * jnp.asarray(20.e-6, R)) ** 3
    cn_x = jnp.minimum(jnp.maximum(ccnc, cwnccn),
                       rho0 * qvex / jnp.maximum(C.cwmasn5, jnp.maximum(cwmas20, xmas_c)))
    thetap_m = jnp.where(pos, thetap_m + felvcp * qvex / (pi0), thetap_m)
    qwvp_m = jnp.where(pos, qwvp_m - qvex, qwvp_m)
    qc_m = jnp.where(pos, qc_m + qvex, qc_m)
    ccna_m = jnp.where(pos, ccna_m + cn_x, ccna_m)
    cxc_m = jnp.where(pos, cxc_m + cn_x, cxc_m)

    # droplet mass bounds (11881-11893)
    okm = active & (cxc_m > C.cxmin) & (qc_m > C.qxmin[LC])
    xmas_m = rho0 * qc_m / jnp.where(okm, cxc_m, 1.0)
    out_rng = okm & ((xmas_m < C.cwmasn) | (xmas_m > C.cwmasx))
    xmas_cl = jnp.maximum(jnp.minimum(xmas_m, C.cwmasx), C.cwmasn)
    cxc_m = jnp.where(out_rng, rho0 * qc_m / xmas_cl, cxc_m)

    # ------------------------------------------------------------------ scatter (11942-12004)
    g = gathered
    an_new = an
    an_new = an_new.at[LT].set(jnp.where(g, theta0 + thetap_m, an[LT]))
    an_new = an_new.at[LV].set(jnp.where(g, qv0 + qwvp_m, an[LV]))
    an_new = an_new.at[LC].set(jnp.where(g, qc_m + jnp.minimum(an[LC], 0.0), an[LC]))  # ido(lc) == 1
    an_new = an_new.at[LR].set(jnp.where(g, qr_m + jnp.minimum(an[LR], 0.0), an[LR]))  # ido(lr), rcond == 2
    an_new = an_new.at[LNC].set(jnp.where(g, jnp.maximum(cxc_m, 0.0), an[LNC]))
    an_new = an_new.at[LCCNA].set(jnp.where(g, jnp.maximum(0.0, ccna_m), an[LCCNA]))
    an_new = an_new.at[LNR].set(jnp.where(g, jnp.maximum(cx_r, 0.0), an[LNR]))
    return an_new, t0, ssfilt
