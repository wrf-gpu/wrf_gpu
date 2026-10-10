"""WRF CAM longwave radiation (``ra_lw_physics = 3``): radtpl, radoz2, trcpth, trcplk, radems/trcems, radclwmx.

Faithful float64 port of pristine WRF V4.7.1 ``phys/module_ra_cam.F`` / ``module_ra_cam_support.F`` (see
:mod:`gpuwrf.physics.ra_cam_common` for conventions: CAM vertical order, ``lit`` for REAL literals, ``ntoplw = 1``).
Absorptivities (``radabs``) live in :mod:`gpuwrf.physics.ra_cam_lw_abs`.

WRF specifics reproduced: radtpl/radems/radabs use the MODULE ``co2vmr`` (355 ppm, never updated by ``camrad``)
while trcpth uses the CLWRF ``co2mmr``; strat_volcanic = .false. so every ``aer_trn_ttl`` factor is exactly 1.0
(dropped); REAL constant sub-expressions (``1.15*3.42217e3``, ``1./250.``, ``1./.3205``, ``1.0/296.0``) are folded in
single precision by gfortran.  Cumulative vertical paths are sequential Python sums over levels (exact order).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.ra_cam_common import (
    CONST,
    COEFH,
    COEFJ,
    COEFK,
    DLP_H2O,
    DLU_H2O,
    DRH_H2O,
    DTE_H2O,
    DTP_H2O,
    FET,
    MAX_LP_H2O,
    MAX_LU_H2O,
    MAX_RH_H2O,
    MAX_TE_H2O,
    MAX_TP_H2O,
    MIN_LP_H2O,
    MIN_LU_H2O,
    MIN_P_H2O,
    MIN_RH_H2O,
    MIN_TE_H2O,
    MIN_TP_H2O,
    MIN_U_H2O,
    N_P,
    N_RH,
    N_TE,
    N_TP,
    N_U,
    fh2oself,
    lit,
    phi,
    psi,
)

f32 = np.float32


def ffold(*xs) -> float:
    """gfortran folds a product of REAL literals in single precision (left to right)."""

    acc = f32(xs[0])
    for x in xs[1:]:
        acc = f32(acc * f32(x))
    return float(acc)


def fdiv(a, b) -> float:
    return float(f32(f32(a) / f32(b)))


def _pow4(x):
    x2 = x * x
    return x2 * x2


def _pow3(x):
    return (x * x) * x


def _cumsum_levels(first, increments):
    """Sequential Fortran recurrence v(1) = first; v(k+1) = v(k) + inc(k) -> (ncol, n+1)."""

    out = [first]
    for k in range(increments.shape[1]):
        out.append(out[-1] + increments[:, k])
    return jnp.stack(out, axis=1)


# --------------------------------------------------------------------------------------------------------------------- #
# radtpl                                                                                                                #
# --------------------------------------------------------------------------------------------------------------------- #
class RadtplOut(NamedTuple):
    plco2: jnp.ndarray
    plh2o: jnp.ndarray
    tplnka: jnp.ndarray
    s2c: jnp.ndarray
    tcg: jnp.ndarray
    w: jnp.ndarray
    tplnke: jnp.ndarray
    tint: jnp.ndarray
    tint4: jnp.ndarray
    tlayr: jnp.ndarray
    tlayr4: jnp.ndarray
    plh2ob: jnp.ndarray  # (ncol, 2, pverp)
    wb: jnp.ndarray      # (ncol, 2, pverp)


def radtpl(tnm, lwupcgs, qnm, pnm, pmln, piln) -> RadtplOut:
    c = CONST
    repsil = 1.0 / c.epsilo
    cpwpl = c.amco2 / c.amd * 0.5 / (c.gravit * c.p0)
    co2vmr = c.co2vmr_module
    pnm2 = pnm * pnm
    plh2o = _cumsum_levels(c.rgsslp * qnm[:, 0] * pnm[:, 0] * pnm[:, 0],
                           c.rgsslp * (pnm2[:, 1:] - pnm2[:, :-1]) * qnm)
    plco2 = jnp.concatenate([(co2vmr * cpwpl * pnm[:, 0] * pnm[:, 0])[:, None], co2vmr * cpwpl * pnm2[:, 1:]], axis=1)
    tint4_bot = lwupcgs / c.stebol
    tint_bot = jnp.sqrt(jnp.sqrt(tint4_bot))
    t_top = tnm[:, 0]
    dy = (piln[:, 1:-1] - pmln[:, 1:]) / (pmln[:, :-1] - pmln[:, 1:])
    tint_mid = tnm[:, 1:] - dy * (tnm[:, 1:] - tnm[:, :-1])
    tint = jnp.concatenate([t_top[:, None], tint_mid, tint_bot[:, None]], axis=1)
    tint4 = jnp.concatenate([_pow4(t_top)[:, None], _pow4(tint_mid), tint4_bot[:, None]], axis=1)
    tlayr = jnp.concatenate([tint[:, :1], tnm], axis=1)
    tlayr4 = jnp.concatenate([_pow4(t_top)[:, None], _pow4(tnm)], axis=1)
    tplnka = jnp.concatenate([t_top[:, None], 0.5 * (tint[:, 1:] + tint[:, :-1])], axis=1)
    tplnke = t_top
    rga = c.rga
    q0, p0_, t0 = qnm[:, 0], pnm[:, 0], tnm[:, 0]
    w0 = c.sslp * (plh2o[:, 0] * 2.0) / p0_
    dpnm = pnm[:, 1:] - pnm[:, :-1]
    dpnmsq = pnm2[:, 1:] - pnm2[:, :-1]
    dw = rga * qnm * dpnm
    w = _cumsum_levels(w0, dw)
    wb = jnp.stack([_cumsum_levels(w0 * phi(t0, b), dw * phi(tnm, b)) for b in (1, 2)], axis=1)
    dplh2o = plh2o[:, 1:] - plh2o[:, :-1]
    plh2ob = jnp.stack([_cumsum_levels(plh2o[:, 0] * psi(t0, b), dplh2o * psi(tnm, b)) for b in (1, 2)], axis=1)
    tcg = _cumsum_levels(rga * q0 * p0_ * t0, dw * tnm)
    s2c = _cumsum_levels(plh2o[:, 0] * fh2oself(t0) * q0 * repsil,
                         c.rgsslp * dpnmsq * qnm * fh2oself(tnm) * qnm * repsil)
    return RadtplOut(plco2, plh2o, tplnka, s2c, tcg, w, tplnke, tint, tint4, tlayr, tlayr4, plh2ob, wb)


def radoz2(o3vmr, pnm):
    """radoz2 on cgs interface pressures: (plol, plos)."""

    c = CONST
    plos_inc = lit(0.1) * c.cplos * o3vmr * (pnm[:, 1:] - pnm[:, :-1])
    plol_inc = lit(0.01) * c.cplol * o3vmr * (pnm[:, 1:] * pnm[:, 1:] - pnm[:, :-1] * pnm[:, :-1])
    plos = _cumsum_levels(lit(0.1) * c.cplos * o3vmr[:, 0] * pnm[:, 0], plos_inc)
    plol = _cumsum_levels(lit(0.01) * c.cplol * o3vmr[:, 0] * pnm[:, 0] * pnm[:, 0], plol_inc)
    return plol, plos


class TrcpthOut(NamedTuple):
    ucfc11: jnp.ndarray
    ucfc12: jnp.ndarray
    un2o0: jnp.ndarray
    un2o1: jnp.ndarray
    uch4: jnp.ndarray
    uco211: jnp.ndarray
    uco212: jnp.ndarray
    uco213: jnp.ndarray
    uco221: jnp.ndarray
    uco222: jnp.ndarray
    uco223: jnp.ndarray
    bn2o0: jnp.ndarray
    bn2o1: jnp.ndarray
    bch4: jnp.ndarray
    uptype: jnp.ndarray


_CO2_BANDS = ((3.42217e3, 1849.7), (6.02454e3, 2782.1), (5.53143e3, 3723.2),
              (3.88984e3, 1997.6), (3.67108e3, 3843.8), (6.50642e3, 2989.7))


def trcpth(tnm, pnm, cfc11, cfc12, n2o, ch4, qnm, co2mmr) -> TrcpthOut:
    """trcpth; ``co2mmr`` is the module (CLWRF) CO2 mass mixing ratio, scalar or per column ``(ncol,)``."""

    c = CONST
    co2mmr = jnp.asarray(co2mmr, jnp.float64)
    co2mmr_k = co2mmr if co2mmr.ndim == 0 else co2mmr[:, None]
    diff = lit(1.66)
    rga, sslp = c.rga, c.sslp
    t0, p0_, q0 = tnm[:, 0], pnm[:, 0], qnm[:, 0]
    r296 = fdiv(1.0, 296.0)
    top = {}
    top["ucfc11"] = lit(1.8) * cfc11[:, 0] * p0_ * rga
    top["ucfc12"] = lit(1.8) * cfc12[:, 0] * p0_ * rga
    top["un2o0"] = diff * lit(1.02346e5) * n2o[:, 0] * p0_ * rga / jnp.sqrt(t0)
    top["un2o1"] = diff * lit(2.01909) * top["un2o0"] * jnp.exp(-(lit(847.36) / t0))
    top["uch4"] = diff * lit(8.60957e4) * ch4[:, 0] * p0_ * rga / jnp.sqrt(t0)
    co2fac = diff * co2mmr * p0_ * rga
    alpha1_0 = jnp.power(1.0 - jnp.exp(-(lit(1540.0) / t0)), 3.0) / jnp.sqrt(t0)
    alpha2_0 = jnp.power(1.0 - jnp.exp(-(lit(1360.0) / t0)), 3.0) / jnp.sqrt(t0)
    for name, (coef, ex), alpha in zip(("uco211", "uco212", "uco213", "uco221", "uco222", "uco223"), _CO2_BANDS,
                                       (alpha1_0,) * 3 + (alpha2_0,) * 3):
        top[name] = lit(coef) * co2fac * alpha * jnp.exp(-(lit(ex) / t0))
    top["bn2o0"] = diff * lit(19.399) * (p0_ * p0_) * n2o[:, 0] * lit(1.02346e5) * rga / (sslp * t0)
    top["bn2o1"] = top["bn2o0"] * jnp.exp(-(lit(847.36) / tnm[:, 0])) * lit(2.06646e5)
    top["bch4"] = diff * lit(2.94449) * ch4[:, 0] * (p0_ * p0_) * rga * lit(8.60957e4) / (sslp * t0)
    top["uptype"] = diff * q0 * (p0_ * p0_) * jnp.exp(lit(1800.0) * (1.0 / t0 - r296)) * rga / sslp

    rt = 1.0 / tnm
    rsqrt = jnp.sqrt(rt)
    pbar = 0.5 * (pnm[:, 1:] + pnm[:, :-1]) / sslp
    dpnm = (pnm[:, 1:] - pnm[:, :-1]) * rga
    alpha1 = diff * rsqrt * jnp.power(1.0 - jnp.exp(-(lit(1540.0) / tnm)), 3.0)
    alpha2 = diff * rsqrt * jnp.power(1.0 - jnp.exp(-(lit(1360.0) / tnm)), 3.0)
    e847 = jnp.exp(-(lit(847.36) / tnm))
    inc = {}
    inc["ucfc11"] = lit(1.8) * cfc11 * dpnm
    inc["ucfc12"] = lit(1.8) * cfc12 * dpnm
    inc["un2o0"] = diff * lit(1.02346e5) * n2o * rsqrt * dpnm
    inc["un2o1"] = diff * lit(2.06646e5) * n2o * rsqrt * e847 * dpnm
    inc["uch4"] = diff * lit(8.60957e4) * ch4 * rsqrt * dpnm
    for name, (coef, ex), alpha in zip(("uco211", "uco212", "uco213", "uco221", "uco222", "uco223"), _CO2_BANDS,
                                       (alpha1,) * 3 + (alpha2,) * 3):
        inc[name] = ffold(1.15, coef) * alpha * co2mmr_k * jnp.exp(-(lit(ex) / tnm)) * dpnm
    inc["bn2o0"] = diff * lit(19.399) * pbar * rt * lit(1.02346e5) * n2o * dpnm
    inc["bn2o1"] = diff * lit(19.399) * pbar * rt * lit(2.06646e5) * e847 * n2o * dpnm
    inc["bch4"] = diff * lit(2.94449) * rt * pbar * lit(8.60957e4) * ch4 * dpnm
    inc["uptype"] = diff * qnm * jnp.exp(lit(1800.0) * (1.0 / tnm - r296)) * pbar * dpnm
    return TrcpthOut(**{n: _cumsum_levels(top[n], inc[n]) for n in TrcpthOut._fields})


_F1 = np.asarray([5.85713e8, 7.94950e8, 1.47009e9, 1.40031e9, 1.34853e8, 1.05158e9, 3.35370e8, 3.99601e8, 5.35994e8,
                  8.42955e8, 4.63682e8, 5.18944e8, 8.83202e8, 1.03279e9], f32).astype(np.float64)
_F2 = np.asarray([2.02493e11, 3.04286e11, 6.90698e11, 6.47333e11, 2.85744e10, 4.41862e11, 9.62780e10, 1.21618e11,
                  1.79905e11, 3.29029e11, 1.48294e11, 1.72315e11, 3.50140e11, 4.31364e11], f32).astype(np.float64)
_F3 = np.asarray([1383.0, 1531.0, 1879.0, 1849.0, 848.0, 1681.0, 1148.0, 1217.0, 1343.0, 1561.0, 1279.0, 1328.0,
                  1586.0, 1671.0], np.float64)


def trcplk(tint, tlayr, tplnke):
    """trcplk: emplnk (ncol, 14), abplnk1/abplnk2 (ncol, 14, pverp)."""

    f1, f2, f3 = (jnp.asarray(a)[None, :] for a in (_F1, _F2, _F3))
    te = tplnke[:, None]
    emplnk = f1 / (jnp.power(te, 4.0) * (jnp.exp(f3 / te) - 1.0))
    f2c, f3c = f2[:, :, None], f3[:, :, None]

    def ab(t):
        t = t[:, None, :]
        e = jnp.exp(f3c / t) - 1.0
        return (f2c * jnp.exp(f3c / t)) / (jnp.power(t, 5.0) * (e * e))

    return emplnk, ab(tint), ab(tlayr)


# --------------------------------------------------------------------------------------------------------------------- #
# radems (+ trcems)                                                                                                     #
# --------------------------------------------------------------------------------------------------------------------- #
class RademsOut(NamedTuple):
    co2em: jnp.ndarray
    co2eml: jnp.ndarray
    co2t: jnp.ndarray
    h2otr: jnp.ndarray
    abplnk1: jnp.ndarray
    abplnk2: jnp.ndarray
    emstot: jnp.ndarray


def _flat(table):
    return jnp.asarray(np.asarray(table).ravel(order="F"))


def table_index(ip, itp, iu, ite, irh):
    """0-based linear index into a Fortran-ordered (n_p, n_tp, n_u, n_te, n_rh) table from 1-based indices."""

    return (ip - 1) + N_P * ((itp - 1) + N_TP * ((iu - 1) + N_U * ((ite - 1) + N_TE * (irh - 1))))


def interp5(flat, ip, itp, iu, ite, irh, wp, wtp, wu, wte, wrh):
    """CAM's 32-term table interpolation, terms and weight products in the Fortran listing order.

    Each ``w*`` is the weight of the UPPER index (``wtp = dvar - floor(dvar)``); the lower weight is ``1.0 - w``.
    """

    wp1, wtp1, wu1, wte1, wrh1 = 1.0 - wp, 1.0 - wtp, 1.0 - wu, 1.0 - wte, 1.0 - wrh
    pick = lambda lo, w, w1: w1 if lo else w
    acc = None
    for p_lo in (True, False):
        for t_lo in (True, False):
            for u_lo in (True, False):
                for e_lo in (True, False):
                    for r_lo in (True, False):
                        w_te = pick(t_lo, wtp, wtp1) * pick(e_lo, wte, wte1)
                        w_ter = w_te * pick(r_lo, wrh, wrh1)
                        weight = pick(p_lo, wp, wp1) * w_ter
                        idx = table_index(ip + (0 if p_lo else 1), itp + (0 if t_lo else 1), iu + (0 if u_lo else 1),
                                          ite + (0 if e_lo else 1), irh + (0 if r_lo else 1))
                        term = jnp.take(flat, idx, mode="clip") * weight * pick(u_lo, wu, wu1)
                        acc = term if acc is None else acc + term
    return acc


def lookup_index(dvar, n):
    """``min(max(int(aint(dvar)) + 1, 1), n - 1)`` and the upper weight ``dvar - floor(dvar)``."""

    i = jnp.clip(jnp.trunc(dvar).astype(jnp.int32) + 1, 1, n - 1)
    return i, dvar - jnp.floor(dvar)


def _dbvt(t):
    return ((lit(-2.8911366682e-4) + (lit(2.3771251896e-6) + lit(1.1305188929e-10) * t) * t)
            / (1.0 + (lit(-6.1364820707e-3) + lit(1.5550319767e-5) * t) * t))


def _fo3(ux, vx):
    return ux / jnp.sqrt(4.0 + ux * (1.0 + vx))


_G1 = np.asarray([0.0468556, 0.0397454, 0.0407664, 0.0304380, 0.0540398, 0.0321962], f32).astype(np.float64)
_G2 = np.asarray([14.4832, 4.30242, 5.23523, 3.25342, 0.698935, 16.5599], f32).astype(np.float64)
_G3 = np.asarray([26.1898, 18.4476, 15.3633, 12.1927, 9.14992, 8.07092], f32).astype(np.float64)
_G4 = np.asarray([0.0261782, 0.0369516, 0.0307266, 0.0243854, 0.0182932, 0.0161418], f32).astype(np.float64)
_AB = np.asarray([3.0857e-2, 2.3524e-2, 1.7310e-2, 2.6661e-2, 2.8074e-2, 2.2915e-2], f32).astype(np.float64)
_BB = np.asarray([-1.3512e-4, -6.8320e-5, -3.2609e-5, -1.0228e-5, -9.5743e-5, -1.0304e-4], f32).astype(np.float64)
_ABP = np.asarray([2.9129e-2, 2.4101e-2, 1.9821e-2, 2.6904e-2, 2.9458e-2, 1.9892e-2], f32).astype(np.float64)
_BBP = np.asarray([-1.3139e-4, -5.5688e-5, -4.6380e-5, -8.0362e-5, -1.0115e-4, -8.8061e-5], f32).astype(np.float64)


def _func(u, b):
    return u / jnp.sqrt(4.0 + u * (1.0 + 1.0 / b))


def trcems(co2t, pnm, tr: TrcpthOut, w, s2c, up2, emplnk, th2o, tco2, to3):
    """trcems for all levels at once (k = k1 axis); emplnk (ncol, 14) broadcast over levels."""

    sslp = CONST.sslp
    sqti = jnp.sqrt(co2t)
    tt = jnp.abs(co2t - 250.0)
    tw = []
    for l in range(6):
        psi1 = jnp.exp(_ABP[l] * tt + _BBP[l] * tt * tt)
        phi1 = jnp.exp(_AB[l] * tt + _BB[l] * tt * tt)
        p1 = pnm * (psi1 / phi1) / sslp
        w1 = w * phi1
        tw.append(jnp.exp(-(_G1[l] * p1 * (jnp.sqrt(1.0 + _G2[l] * (w1 / p1)) - 1.0)) - _G3[l] * s2c - _G4[l] * tr.uptype))
    em = lambda n: emplnk[:, n - 1][:, None]
    tcfc3 = jnp.exp(-(lit(175.005) * tr.ucfc11))
    tcfc4 = jnp.exp(-(lit(1202.18) * tr.ucfc11))
    tcfc6 = jnp.exp(-(lit(5786.73) * tr.ucfc12))
    tcfc7 = jnp.exp(-(lit(2873.51) * tr.ucfc12))
    tcfc8 = jnp.exp(-(lit(2085.59) * tr.ucfc12))
    ecfc1 = 50.0 * (1.0 - jnp.exp(-(lit(54.09) * tr.ucfc11))) * tw[0] * em(7)
    ecfc2 = 60.0 * (1.0 - jnp.exp(-(lit(5130.03) * tr.ucfc11))) * tw[1] * em(8)
    ecfc3 = 60.0 * (1.0 - tcfc3) * tw[3] * tcfc6 * em(9)
    ecfc4 = 100.0 * (1.0 - tcfc4) * tw[4] * em(10)
    ecfc5 = 45.0 * (1.0 - jnp.exp(-(lit(1272.35) * tr.ucfc12))) * tw[2] * em(11)
    ecfc6 = 50.0 * (1.0 - tcfc6) * tw[3] * em(12)
    ecfc7 = 80.0 * (1.0 - tcfc7) * tw[4] * tcfc4 * em(13)
    ecfc8 = 70.0 * (1.0 - tcfc8) * tw[5] * em(14)
    tlw = jnp.exp(-(1.0 * jnp.sqrt(up2)))
    betac = tr.bch4 / tr.uch4
    ech4 = lit(6.00444) * sqti * jnp.log(1.0 + _func(tr.uch4, betac)) * tlw * em(3)
    tch4 = 1.0 / (1.0 + lit(0.02) * _func(tr.uch4, betac))
    u01, u11 = tr.un2o0, tr.un2o1
    beta01 = tr.bn2o0 / tr.un2o0
    beta11 = tr.bn2o1 / tr.un2o1
    en2o1 = lit(2.35558) * sqti * jnp.log(1.0 + _func(u01, beta01) + _func(u11, beta11)) * tlw * tch4 * em(4)
    u02 = lit(0.100090) * u01
    u12 = lit(0.0992746) * u11
    beta02 = lit(0.964282) * beta01
    en2o2 = lit(2.65581) * sqti * jnp.log(1.0 + _func(u02, beta02) + _func(u12, beta02)) * tco2 * th2o * em(5)
    u03 = lit(0.0333767) * u01
    beta03 = lit(0.982143) * beta01
    en2o3 = lit(2.54034) * sqti * jnp.log(1.0 + _func(u03, beta03)) * tw[5] * tcfc8 * em(6)
    betac1 = lit(2.97558) * pnm / (sslp * sqti)
    betac2 = 2.0 * betac1
    eco21 = (lit(3.7571) * sqti * jnp.log(1.0 + _func(tr.uco211, betac1) + _func(tr.uco212, betac2)
                                         + _func(tr.uco213, betac2)) * to3 * tw[4] * tcfc4 * tcfc7 * em(2))
    eco22 = (lit(3.8443) * sqti * jnp.log(1.0 + _func(tr.uco221, betac1) + _func(tr.uco222, betac1)
                                         + _func(tr.uco223, betac2)) * tw[3] * tcfc3 * tcfc6 * em(1))
    return (ecfc1 + ecfc2 + ecfc3 + ecfc4 + ecfc5 + ecfc6 + ecfc7 + ecfc8 + en2o1 + en2o2 + en2o3 + ech4
            + eco21 + eco22)


def radems(tp: RadtplOut, pnm, plol, plos, tr: TrcpthOut, tables) -> RademsOut:
    c = CONST
    eh2onw, cn_eh2ow, ln_eh2ow = _flat(tables.eh2onw), _flat(tables.cn_eh2ow), _flat(tables.ln_eh2ow)
    estblh2o = jnp.asarray(tables.estblh2o)
    s2c, tcg, w, plh2o, plco2 = tp.s2c, tp.tcg, tp.w, tp.plh2o, tp.plco2
    tint, tint4, tlayr, tlayr4 = tp.tint, tp.tint4, tp.tlayr, tp.tlayr4
    tplnke = tp.tplnke
    pverp = pnm.shape[1]
    r250 = fdiv(1.0, 250.0)
    r300 = fdiv(1.0, 300.0)
    rsslp = 1.0 / c.sslp
    fdif = lit(1.66)
    sslp_mks = c.sslp / 10.0
    omeps = 1.0 - c.epsilo
    rga = c.rga

    ex = jnp.exp(lit(960.0) / tplnke)
    co2plk = lit(5.e8) / (_pow4(tplnke) * (ex - 1.0))
    s = tplnke * pnm[:, 0]
    co2t_cols = [tplnke]
    co2eml_cols = [None] * (pverp - 1)
    for k in range(1, pverp):  # Fortran k = 2..pverp, k1 = pverp+1-k... (k1 = pverp..2 while k = 2..pverp)
        s = s + tlayr[:, k] * (pnm[:, k] - pnm[:, k - 1])
        co2t_cols.append(s / pnm[:, k])
    co2t = jnp.stack(co2t_cols, axis=1)
    exl = jnp.exp(lit(960.0) / tlayr[:, 1:])
    tlayr5 = tlayr[:, 1:] * tlayr4[:, 1:]
    co2eml = lit(1.2e11) * exl / (tlayr5 * ((exl - 1.0) * (exl - 1.0)))
    del co2eml_cols
    dbvtt = _dbvt(tplnke)[:, None]
    emplnk, abplnk1, abplnk2 = trcplk(tint, tlayr, tplnke)

    te_ = tplnke[:, None]
    u = plh2o
    pnew = u / w
    pnew_mks = pnew * sslp_mks
    uc1 = (s2c + lit(1.7e-3) * plh2o) * (1.0 + 2.0 * s2c) / (1.0 + 15.0 * s2c)
    pch2o = s2c
    tpathe = tcg / w
    t_p = jnp.minimum(jnp.maximum(tpathe, MIN_TP_H2O), MAX_TP_H2O)
    iest = (jnp.floor(t_p) - MIN_TP_H2O).astype(jnp.int32)
    esx = jnp.take(estblh2o, iest, mode="clip") + (jnp.take(estblh2o, iest + 1, mode="clip") - jnp.take(estblh2o, iest, mode="clip")) * (
        t_p - MIN_TP_H2O - iest.astype(jnp.float64))
    qsx = c.epsilo * esx / (pnew_mks - omeps * esx)
    q_path = w / pnm / rga
    ub1 = tp.plh2ob[:, 0] / psi(t_p, 1)
    ub2 = tp.plh2ob[:, 1] / psi(t_p, 2)
    pnewb1 = ub1 / tp.wb[:, 0] * phi(t_p, 1)
    pnewb2 = ub2 / tp.wb[:, 1] * phi(t_p, 2)
    dtx = te_ - 250.0
    dty = tpathe - 250.0
    te1 = te_
    te2 = te1 * te1
    te3 = te2 * te1
    te4 = te3 * te1
    te5 = te4 * te1
    itp, wtp = lookup_index((t_p - MIN_TP_H2O) / DTP_H2O, N_TP)
    t_e = jnp.minimum(jnp.maximum(te_ - t_p, MIN_TE_H2O), MAX_TE_H2O)
    ite, wte = lookup_index((t_e - MIN_TE_H2O) / DTE_H2O, N_TE)
    rh_path = jnp.minimum(jnp.maximum(q_path / qsx, MIN_RH_H2O), MAX_RH_H2O)
    irh, wrh = lookup_index((rh_path - MIN_RH_H2O) / DRH_H2O, N_RH)
    fch2o = fh2oself(t_p)
    uch2o = (pch2o * c.epsilo) / (q_path * fch2o)

    def fe_band(ib):
        return (FET[0, ib] + FET[1, ib] * te1 + FET[2, ib] * te2 + FET[3, ib] * te3 + FET[4, ib] * te4
                + FET[5, ib] * te5)

    def u_p_index(ub, pnewb):
        uvar = ub * fdif
        log_u = jnp.minimum(jnp.log10(jnp.maximum(uvar, MIN_U_H2O)), MAX_LU_H2O)
        iu, wu = lookup_index((log_u - MIN_LU_H2O) / DLU_H2O, N_U)
        log_p = jnp.minimum(jnp.log10(jnp.maximum(pnewb, MIN_P_H2O)), MAX_LP_H2O)
        ip, wp = lookup_index((log_p - MIN_LP_H2O) / DLP_H2O, N_P)
        return uvar, iu, wu, ip, wp

    uvar1, iu1, wu1, ip1, wp1 = u_p_index(ub1, pnewb1)
    e_star = interp5(eh2onw, ip1, itp, iu1, ite, irh, wp1, wtp, wu1, wte, wrh)
    emis1 = jnp.minimum(jnp.maximum(fe_band(0) * (1.0 - (1.0 - e_star)), 0.0), 1.0)
    emis1 = jnp.where(uvar1 < MIN_U_H2O, emis1 * (uvar1 / MIN_U_H2O), emis1)
    uvar2, iu2, wu2, ip2, wp2 = u_p_index(ub2, pnewb2)
    log_uc = jnp.minimum(jnp.log10(jnp.maximum(uch2o * fdif, MIN_U_H2O)), MAX_LU_H2O)
    iuc, wuc = lookup_index((log_uc - MIN_LU_H2O) / DLU_H2O, N_U)
    l_star = interp5(ln_eh2ow, ip2, itp, iu2, ite, irh, wp2, wtp, wu2, wte, wrh)
    c_star = interp5(cn_eh2ow, ip2, itp, iuc, ite, irh, wp2, wtp, wuc, wte, wrh)
    emis2 = jnp.minimum(jnp.maximum(fe_band(1) * (1.0 - l_star * c_star), 0.0), 1.0)
    emis2 = jnp.where(uvar2 < MIN_U_H2O, emis2 * (uvar2 / MIN_U_H2O), emis2)
    h2oems = emis1 + emis2

    term7_1 = COEFJ[0, 0] + COEFJ[1, 0] * dty * (1.0 + c.c16 * dty)
    term8_1 = COEFK[0, 0] + COEFK[1, 0] * dty * (1.0 + c.c17 * dty)
    term7_2 = COEFJ[0, 1] + COEFJ[1, 1] * dty * (1.0 + c.c26 * dty)
    term8_2 = COEFK[0, 1] + COEFK[1, 1] * dty * (1.0 + c.c27 * dty)
    sqrtu = jnp.sqrt(u)
    k21 = term7_1 + term8_1 / (1.0 + (c.c30 + c.c31 * (dty - 10.0) * (dty - 10.0)) * sqrtu)
    k22 = term7_2 + term8_2 / (1.0 + (c.c28 + c.c29 * (dty - 10.0)) * sqrtu)
    fwk = c.fwcoef + c.fwc1 / (1.0 + c.fwc2 * u)
    tr1 = jnp.exp(-(k21 * (sqrtu + c.fc1 * fwk * u)))
    tr2 = jnp.exp(-(k22 * (sqrtu + c.fc1 * fwk * u)))
    tr3 = jnp.exp(-((COEFH[0, 0] + COEFH[1, 0] * dtx) * uc1))
    tr4 = jnp.exp(-((COEFH[0, 1] + COEFH[1, 1] * dtx) * uc1))
    tr7 = tr1 * tr3
    tr8 = tr2 * tr4
    troco2 = lit(0.65) * tr7 + lit(0.35) * tr8
    th2o = tr8

    t1i = jnp.exp(-(lit(480.0) / co2t))
    sqti = jnp.sqrt(co2t)
    rsqti = 1.0 / sqti
    et = t1i
    et2 = et * et
    et4 = et2 * et2
    omet = 1.0 - lit(1.5) * et2
    f1co2 = lit(899.70) * omet * (1.0 + lit(1.94774) * et + lit(4.73486) * et2) * rsqti
    sqwp = jnp.sqrt(plco2)
    f1sqwp = f1co2 * sqwp
    t1co2 = 1.0 / (1.0 + lit(245.18) * omet * sqwp * rsqti)
    oneme = 1.0 - et2
    alphat = _pow3(oneme) * rsqti
    wco2 = lit(2.5221) * c.co2vmr_module * pnm * rga
    u7 = lit(4.9411e4) * alphat * et2 * wco2
    u8 = lit(3.9744e4) * alphat * et4 * wco2
    u9 = lit(1.0447e5) * alphat * et4 * et2 * wco2
    u13 = lit(2.8388e3) * alphat * et4 * wco2
    tcrfac = jnp.sqrt((te_ * r250) * (co2t * r300))
    pi_ = pnm * rsslp + 2.0 * c.dpfco2 * tcrfac
    posqt = pi_ / (2.0 * sqti)
    rbeta7 = 1.0 / (lit(5.3288) * posqt)
    rbeta8 = 1.0 / (lit(10.6576) * posqt)
    rbeta9 = rbeta7
    rbeta13 = rbeta9
    f2co2 = (u7 / jnp.sqrt(4.0 + u7 * (1.0 + rbeta7))) + (u8 / jnp.sqrt(4.0 + u8 * (1.0 + rbeta8))) + (
        u9 / jnp.sqrt(4.0 + u9 * (1.0 + rbeta9)))
    f3co2 = u13 / jnp.sqrt(4.0 + u13 * (1.0 + rbeta13))
    tmp1 = jnp.log(1.0 + f1sqwp)
    tmp2 = jnp.log(1.0 + f2co2)
    tmp3 = jnp.log(1.0 + f3co2)
    absbnd = (tmp1 + 2.0 * t1co2 * tmp2 + 2.0 * tmp3) * sqti
    tco2 = 1.0 / (1.0 + 10.0 * (u7 / jnp.sqrt(4.0 + u7 * (1.0 + rbeta7))))
    co2ems = troco2 * absbnd * co2plk[:, None]
    exi = jnp.exp(lit(960.0) / tint)
    exm1sq = (exi - 1.0) * (exi - 1.0)
    co2em = lit(1.2e11) * exi / (tint * tint4 * exm1sq)

    h2otr = jnp.exp(-(12.0 * s2c))
    te = jnp.power(co2t / 293.0, lit(0.7))
    u1 = lit(18.29) * plos / te
    u2 = lit(0.5649) * plos / te
    phat = plos / plol
    tcrfac_o3 = jnp.sqrt(te_ * r250) * te
    beta = fdiv(1.0, 0.3205) * ((1.0 / phat) + (c.dpfo3 * tcrfac_o3))
    realnu = (1.0 / beta) * te
    o3bndi = 74.0 * te * (te_ / 375.0) * jnp.log(1.0 + _fo3(u1, realnu) + _fo3(u2, realnu))
    o3ems = dbvtt * h2otr * o3bndi
    to3 = 1.0 / (1.0 + lit(0.1) * _fo3(u1, realnu) + lit(0.1) * _fo3(u2, realnu))
    emstrc = trcems(co2t, pnm, tr, w, s2c, u, emplnk, th2o, tco2, to3)
    emstot = h2oems + co2ems + o3ems + emstrc
    return RademsOut(co2em, co2eml, co2t, h2otr, abplnk1, abplnk2, emstot)


__all__ = ["RademsOut", "RadtplOut", "TrcpthOut", "interp5", "lookup_index", "radems", "radoz2", "radtpl", "table_index",
           "trcems", "trcpth", "trcplk"]


# --------------------------------------------------------------------------------------------------------------------- #
# radclwmx: longwave fluxes with maximum-random cloud overlap (module_ra_cam.F:4729-5685)                               #
# --------------------------------------------------------------------------------------------------------------------- #
CLDMIN = 1.0e-80  # radclwmx parameter (cldmin = 1.0d-80)


def _take(a, idx):
    """Batched gather along the last axis of ``a`` (ncol, ..., n) with integer ``idx`` broadcast on the leading axes."""

    a = jnp.broadcast_to(a, idx.shape[:-1] + a.shape[-1:])
    return jnp.take_along_axis(a, idx, axis=-1, mode="clip")


def shell_sort_order(values, n):
    """Exact emulation of module_ra_cam_support.F ``sortarray`` (Shell sort, gaps 3h+1) on the first ``n`` entries.

    Each gapped insertion pass is a STABLE sort of every gap chain, so the permutation (including tie order) equals
    a sequence of stable chain sorts for the gaps the Fortran loop visits for this ``n`` (..., 40, 13, 4, 1).
    ``values`` (..., M); returns the permutation (..., M) of positions (identity beyond ``n``).
    """

    m = values.shape[-1]
    pos = jnp.broadcast_to(jnp.arange(m), values.shape)
    perm = pos
    vals = values
    nn = n[..., None]
    for gap in (40, 13, 4, 1):
        if gap >= m:
            continue
        live = pos < nn
        chain = jnp.where(live, pos % gap, gap)
        big = jnp.where(live, vals, jnp.inf)
        order = jnp.lexsort((pos, big, chain), axis=-1)       # chain, then value, then position (stable)
        sc = jnp.take_along_axis(chain, order, axis=-1, mode="clip")
        counts = jnp.sum(chain[..., None, :] == jnp.arange(gap + 1)[:, None], axis=-1)
        starts = jnp.cumsum(counts, axis=-1) - counts
        rank = jnp.arange(m) - jnp.take_along_axis(starts, sc, axis=-1, mode="clip")
        dest = jnp.where(sc < gap, sc + rank * gap, jnp.take_along_axis(pos, order, axis=-1, mode="clip"))
        new_perm = jnp.zeros_like(perm)
        new_vals = jnp.zeros_like(vals)
        src_perm = jnp.take_along_axis(perm, order, axis=-1, mode="clip")
        src_vals = jnp.take_along_axis(vals, order, axis=-1, mode="clip")
        new_perm = jnp.put_along_axis(new_perm, dest, src_perm, axis=-1, inplace=False)
        new_vals = jnp.put_along_axis(new_vals, dest, src_vals, axis=-1, inplace=False)
        apply = (n >= gap) if gap > 1 else (n >= 2)
        perm = jnp.where(apply[..., None], new_perm, perm)
        vals = jnp.where(apply[..., None], new_vals, vals)
    return perm


class LwFluxes(NamedTuple):
    qrl: jnp.ndarray     # (ncol, pver) cgs heating (radctl later: no rescale of qrl)
    qrlcs: jnp.ndarray
    flup: jnp.ndarray    # (ncol, pverp) cgs (radctl scales by 1e-3)
    flupc: jnp.ndarray
    fldn: jnp.ndarray
    fldnc: jnp.ndarray
    flns: jnp.ndarray
    flnt: jnp.ndarray
    flnsc: jnp.ndarray
    flntc: jnp.ndarray
    flwds: jnp.ndarray
    flut: jnp.ndarray
    flutc: jnp.ndarray


def _s_matrix(abstot, absnxt, tint4, tlayr4):
    """radclwmx ``s(k, km)`` (1-based both axes, padded row/col 0): cumulative absorptivity-weighted Planck terms."""

    c = CONST
    ncol, pverp, _ = abstot.shape
    pver = pverp - 1
    one = lambda a: jnp.concatenate([jnp.zeros(a.shape[:1] + (1,) + a.shape[2:], a.dtype), a], axis=1)
    at = jnp.pad(abstot, ((0, 0), (1, 0), (1, 0)))        # at[:, k, km] = abstot(k, km)
    an = one(absnxt)                                       # an[:, k, kn-1] = absnxt(k, kn)
    t4 = one(tint4)
    l4 = one(tlayr4)
    delt = t4[:, pver] - l4[:, pverp]
    delt1 = l4[:, pverp] - t4[:, pverp]
    kk = jnp.arange(pverp + 1)
    bk2 = (at[:, :, pver] + at[:, :, pverp]) * 0.5
    col = c.stebol * (bk2 * delt[:, None] + bk2 * delt1[:, None])
    col = jnp.where(kk == pverp, c.stebol * (delt1 * an[:, pver, 0] + delt * an[:, pver, 3])[:, None], col)
    col = jnp.where(kk == pver, c.stebol * (delt * an[:, pver, 1] + delt1 * an[:, pver, 2])[:, None], col)
    col_top = jnp.where(kk == 0, 0.0, col)
    kms = jnp.arange(pver, 1, -1)                          # km = pver .. 2

    def step(prev, km):
        delt = (jax.lax.dynamic_index_in_dim(t4, km - 1, axis=1, keepdims=False)
                - jax.lax.dynamic_index_in_dim(l4, km, axis=1, keepdims=False))[:, None]
        delt1 = (jax.lax.dynamic_index_in_dim(l4, km, axis=1, keepdims=False)
                 - jax.lax.dynamic_index_in_dim(t4, km, axis=1, keepdims=False))[:, None]
        a_m1 = jax.lax.dynamic_index_in_dim(at, km - 1, axis=2, keepdims=False)
        a_m = jax.lax.dynamic_index_in_dim(at, km, axis=2, keepdims=False)
        n_m1 = jax.lax.dynamic_index_in_dim(an, km - 1, axis=1, keepdims=False)   # (ncol, 4)
        avg = (a_m1 + a_m) * 0.5
        bk2 = jnp.where(kk == km, n_m1[:, 3:4], jnp.where(kk == km - 1, n_m1[:, 1:2], avg))
        bk1 = jnp.where(kk == km, n_m1[:, 0:1], jnp.where(kk == km - 1, n_m1[:, 2:3], bk2))
        new = jnp.where(kk == 0, 0.0, prev + c.stebol * (bk2 * delt + bk1 * delt1))
        return new, new

    _, body = jax.lax.scan(step, col_top, kms)              # body[j] = column km = pver - j
    zero = jnp.zeros_like(col_top)
    stack = [zero, zero] + [body[pver - km] for km in range(2, pver + 1)] + [col_top]
    return jnp.stack(stack, axis=2)   # S[:, k, km]


def _first_valid(cands, valid, fallback_idx):
    """Index of the first candidate (in the given order) with 0 <= emx0 <= 1 among ``valid``; else ``fallback_idx``."""

    ok = valid & (cands >= 0.0) & (cands <= 1.0)
    found = jnp.any(ok, axis=-1)
    first = jnp.argmax(ok, axis=-1)
    return jnp.where(found, first, fallback_idx)


def radclwmx_fluxes(abstot, absnxt, emstot, tp: RadtplOut, lwupcgs, pmid_cgs, pint_cgs, cld, emis, pmxrgn, nmxrgn):
    """Flux part of radclwmx given absorptivities/emissivity; CAM order, cgs. Returns :class:`LwFluxes`."""

    c = CONST
    stebol = c.stebol
    ncol, pver = cld.shape
    pverp = pver + 1
    pverp2 = pver + 2
    S = _s_matrix(abstot, absnxt, tp.tint4, tp.tlayr4)
    pad1 = lambda a: jnp.concatenate([jnp.zeros((ncol, 1), a.dtype), a], axis=1)   # 1-based
    tint4 = pad1(tp.tint4)
    emst = pad1(emstot)
    at = jnp.pad(abstot, ((0, 0), (1, 0), (1, 0)))
    kv = jnp.arange(pverp + 1)
    rows = jnp.arange(ncol)[:, None]
    s_kk1 = S[rows, kv[None, :], jnp.minimum(kv + 1, pverp)[None, :]]      # s(k, k+1)
    # clear sky
    fsul_b = lwupcgs
    tmp = fsul_b - stebol * tint4[:, pverp]
    fsul = fsul_b[:, None] - at[:, :, pverp] * tmp[:, None] + s_kk1
    fsul = fsul.at[:, pverp].set(fsul_b).at[:, 0].set(0.0)
    t4e = stebol * _pow4(tp.tplnke)
    fsdl = t4e[:, None] * emst - (S[:, :, 2] - s_kk1)
    fsdl = fsdl.at[:, 1].set(t4e * emst[:, 1])
    fsdl = fsdl.at[:, pverp].set(t4e * emst[:, pverp] - S[:, pverp, 2]).at[:, 0].set(0.0)

    cldp = jnp.concatenate([jnp.zeros((ncol, 1)), cld, jnp.zeros((ncol, 1))], axis=1)   # cldp(pverp) = 0
    emisp = jnp.concatenate([jnp.zeros((ncol, 1)), emis, jnp.zeros((ncol, 1))], axis=1)
    pmidp = jnp.concatenate([jnp.full((ncol, 1), -jnp.inf), pmid_cgs], axis=1)
    maxcld = jnp.max(cld, axis=1)
    cloudy = maxcld >= CLDMIN
    fclt4 = stebol * tint4[:, 1:]          # fclt4(j) = stebol*tint4(j+1), j = 0..pver   -> index j
    fclb4 = stebol * tint4[:, 1:]          # fclb4(j) = stebol*tint4(j+2), j = -1..pver-1 -> index j+1

    # region bounds (kx1, kx2) for every region, max-overlap regions from cldovrlap
    nreg_max = pverp
    lev = jnp.arange(1, pver + 1, dtype=jnp.int32)
    npm = jnp.sum(pmid_cgs[:, None, :] <= pmxrgn[:, :, None], axis=-1)            # (ncol, pverp): count pmid<=bound
    kx1_l, kx2_l = [], []
    kx2_prev = jnp.zeros((ncol,), jnp.int32)
    for r in range(nreg_max):
        k1 = kx2_prev + 1
        kx2 = jnp.where(npm[:, r] >= k1, npm[:, r], 0).astype(jnp.int32)
        kx1_l.append(k1)
        kx2_l.append(kx2)
        kx2_prev = jnp.where(r < nmxrgn, kx2, kx2_prev)
    kx1_a = jnp.stack(kx1_l, axis=1)
    kx2_a = jnp.stack(kx2_l, axis=1)
    # a region is processed when its predecessor did not reach the surface (kx2(irgn-1) < pver)
    prev_kx2 = jnp.concatenate([jnp.zeros((ncol, 1), jnp.int32), kx2_a[:, :-1]], axis=1)
    reg_active = (jnp.arange(nreg_max)[None, :] < nmxrgn[:, None]) & (prev_kx2 < pver) & cloudy[:, None]
    nmax = jnp.max(jnp.where(cloudy, nmxrgn, 0))

    def region_list(r):
        """Sorted cloudy layers (1-based) of region r: (ncol, pver) padded with pverp, and their count."""

        k1 = jax.lax.dynamic_index_in_dim(kx1_a, r, axis=1, keepdims=False)
        k2 = jax.lax.dynamic_index_in_dim(kx2_a, r, axis=1, keepdims=False)
        inreg = (lev[None, :] >= k1[:, None]) & (lev[None, :] <= k2[:, None]) & (cld >= CLDMIN)
        n = jnp.sum(inreg, axis=1)
        slot = jnp.where(inreg, jnp.cumsum(inreg, axis=1) - 1, pver)
        layers = jnp.full((ncol, pver + 1), pverp, jnp.int32).at[rows, slot].set(
            jnp.broadcast_to(lev, (ncol, pver)), mode="drop")[:, :pver]
        asort = 1.0 - _take(cldp, layers)
        perm = shell_sort_order(asort, n)
        return jnp.take_along_axis(layers, perm, axis=-1, mode="clip"), n, k1, k2

    L = pver + 2   # ksort(0 .. nxsk+1)

    def configurations(ks, nxsk, emx0, X, inc_fs, flux0):
        """The l / l1 configuration sums of radclwmx for one region at every level (vectorized over (ncol, K)).

        ks (ncol,K,L) with ks[...,0] = km1+1 and ks[...,nxsk+1] = pverp; X (ncol,K,L) the l1 flux differences;
        inc_fs (ncol,K) the clear flux at k2; returns the accumulated flux (exact Fortran accumulation order)."""

        emx = jnp.where(jnp.arange(L) == 0, emx0[:, None, None], _take(emisp[:, None, :], ks))
        nl_max = jnp.max(nxsk) + 1

        def l_body(l, state):
            flux, emx, cld0 = state
            act = l <= nxsk + 1
            kl = jax.lax.dynamic_index_in_dim(ks, l, axis=-1, keepdims=False)
            cld1 = _take(cldp[:, None, :], kl[..., None])[..., 0] * jnp.minimum(1, nxsk + 1 - l).astype(jnp.float64)
            upd = act & (cld0 != cld1)
            dc = cld0 - cld1
            flux = jnp.where(upd, flux + dc * inc_fs, flux)

            def l1_body(l1, f):
                e = jax.lax.dynamic_index_in_dim(emx, l1, axis=-1, keepdims=False)
                xx = jax.lax.dynamic_index_in_dim(X, l1, axis=-1, keepdims=False)
                return jnp.where(upd, f + dc * e * xx, f)

            flux = jax.lax.fori_loop(0, l, l1_body, flux)
            cld0 = jnp.where(act, cld1, cld0)
            return flux, emx, cld0

        def l_body_trans(l, state, down):
            flux, emx, cld0 = l_body(l, state)
            nxsk_ = nxsk
            kl = jax.lax.dynamic_index_in_dim(ks, l, axis=-1, keepdims=False)
            trans = 1.0 - _take(emisp[:, None, :], kl[..., None])[..., 0]
            lidx = jnp.arange(L)
            cmp = (ks < kl[..., None]) if down else (ks > kl[..., None])
            mult = (l <= nxsk_)[..., None] & (lidx <= nxsk_[..., None]) & cmp
            emx = jnp.where(mult, emx * trans[..., None], emx)
            return flux, emx, cld0

        return l_body_trans, (flux0, emx, jnp.ones_like(flux0)), nl_max

    def run_config(ks, nxsk, emx0, X, inc_fs, flux0, down):
        body, state, nl_max = configurations(ks, nxsk, emx0, X, inc_fs, flux0)
        flux, _, _ = jax.lax.fori_loop(1, nl_max + 1, lambda l, s: body(l, s, down), state)
        return flux

    def compact(srt, n, incl_fn, km1):
        """ksort(0..nxsk+1) per level k (1..pver): ks[...,0] = km1+1, included sorted layers, then pverp."""

        kcol = lev[None, :, None]
        valid = (jnp.arange(pver)[None, None, :] < n[:, None, None]) & incl_fn(srt[:, None, :], kcol)
        nxsk = jnp.sum(valid, axis=-1)
        slot = jnp.where(valid, jnp.cumsum(valid, axis=-1), L)
        ks = jnp.full((ncol, pver, L), pverp, jnp.int32)
        ks = ks.at[rows[:, :, None], jnp.arange(pver)[None, :, None], slot].set(
            jnp.broadcast_to(srt[:, None, :], (ncol, pver, pver)), mode="drop")
        ks = ks.at[:, :, 0].set((km1 + 1)[:, None])
        return ks, nxsk

    # ---------------- downward pass (regions top -> bottom) ----------------
    fdl0 = jnp.zeros((ncol, pverp + 1)).at[:, 1].set(fsdl[:, 1])
    km1_cands = jnp.arange(-1, pver - 1, dtype=jnp.int32)      # -1 .. pver-2

    def down_region(r, fdl):
        srt, n, k1, k2 = region_list(r)
        act = jax.lax.dynamic_index_in_dim(reg_active, r, axis=1, keepdims=False)
        k3 = k1 + 1
        tmp = S[rows[:, 0], k1, jnp.minimum(k3, pverp)] * jnp.minimum(1, pverp2 - k3).astype(jnp.float64)
        km4 = km1_cands[None, :] + 3
        tmp2 = jnp.take_along_axis(S[rows[:, 0], k1], jnp.minimum(km4, pverp), axis=1, mode="clip") * jnp.minimum(
            1, pverp2 - km4).astype(jnp.float64)
        fdl_k1 = jnp.take_along_axis(fdl, k1[:, None], axis=1, mode="clip")
        fsdl_k1 = jnp.take_along_axis(fsdl, k1[:, None], axis=1, mode="clip")
        fcl = fclb4[:, km1_cands + 1]
        cand = (fdl_k1 - fsdl_k1) / ((fcl - tmp2 + tmp[:, None]) - fsdl_k1)
        valid = km1_cands[None, :] <= (k1 - 2)[:, None]
        last = jnp.maximum(k1 - 2 + 1, 0)                       # index of km1 = k1-2 in km1_cands
        pick = _first_valid(cand, valid, last)
        emx0 = jnp.take_along_axis(cand, pick[:, None], axis=1, mode="clip")[:, 0]
        km1 = km1_cands[pick]
        ks, nxsk = compact(srt, n, lambda layer, k: k >= layer, km1)
        k2v = lev[None, :] + 1                                   # k2 = k + 1
        k3v = k2v + 1
        tmpv = S[rows, k2v, jnp.minimum(k3v, pverp)] * jnp.minimum(1, pverp2 - k3v).astype(jnp.float64)
        km1l = ks - 1
        km4l = km1l + 3
        s_row = S[rows, k2v]                                     # (ncol, pver, pverp+1)
        tmp2l = jnp.take_along_axis(s_row, jnp.minimum(km4l, pverp), axis=-1, mode="clip") * jnp.minimum(
            1, pverp2 - km4l).astype(jnp.float64)
        fsdl_k2 = jnp.take_along_axis(fsdl, k2v, axis=1, mode="clip")
        fclb = jnp.take_along_axis(fclb4[:, None, :], jnp.clip(km1l + 1, 0, pver), axis=-1, mode="clip")
        X = ((fclb - tmp2l) + tmpv[..., None]) - fsdl_k2[..., None]
        flux0 = jnp.take_along_axis(fdl, k2v, axis=1, mode="clip")
        flux = run_config(ks, nxsk, emx0, X, fsdl_k2, flux0, True)
        inreg = (lev[None, :] >= k1[:, None]) & (lev[None, :] <= k2[:, None]) & act[:, None]
        upd = jnp.zeros((ncol, pverp + 1), bool).at[:, 2:].set(inreg)
        newv = jnp.zeros((ncol, pverp + 1)).at[:, 2:].set(flux)
        return jnp.where(upd, newv, fdl)

    fdl = jax.lax.fori_loop(0, nmax, down_region, fdl0)

    # ---------------- upward pass (regions bottom -> top) ----------------
    ful0 = jnp.zeros((ncol, pverp + 1)).at[:, pverp].set(fsul[:, pverp])
    km1_up = jnp.arange(pver - 1, -1, -1, dtype=jnp.int32)      # pver-1 .. 0 (descending search order)

    def up_region(j, ful):
        r = nmax - 1 - j
        srt, n, k1r, k2r = region_list(r)
        act = jax.lax.dynamic_index_in_dim(reg_active, r, axis=1, keepdims=False) & (r < nmxrgn)
        k1 = k2r + 1
        inner = k1 < pverp
        k1c = jnp.minimum(k1, pver)
        km3 = km1_up[None, :] + 2
        s_k1 = S[rows[:, 0], k1c]
        tmp = jnp.take_along_axis(s_k1, jnp.minimum(km3, pverp), axis=1, mode="clip") * jnp.minimum(1, pverp2 - km3).astype(
            jnp.float64)
        ful_k1 = jnp.take_along_axis(ful, k1c[:, None], axis=1, mode="clip")
        fsul_k1 = jnp.take_along_axis(fsul, k1c[:, None], axis=1, mode="clip")
        s_k1k3 = S[rows[:, 0], k1c, k1c + 1][:, None]
        cand = (ful_k1 - fsul_k1) / ((fclt4[:, km1_up] + s_k1k3 - tmp) - fsul_k1)
        valid = km1_up[None, :] >= k2r[:, None]
        last = jnp.maximum(pver - 1 - k2r, 0)                    # index of km1 = kx2 in km1_up
        pick = _first_valid(cand, valid, last)
        emx0 = jnp.where(inner, jnp.take_along_axis(cand, pick[:, None], axis=1, mode="clip")[:, 0], 1.0)
        km1 = jnp.where(inner, jnp.maximum(km1_up[pick], k2r), k1 - 1)
        ks, nxsk = compact(srt, n, lambda layer, k: k <= layer, km1)
        k2v = jnp.broadcast_to(lev[None, :], (ncol, pver))
        k3v = k2v + 1
        s_row = S[rows, k2v]
        s_k2k3 = jnp.take_along_axis(s_row, k3v[..., None], axis=-1, mode="clip")[..., 0]
        km1l = ks - 1
        km3l = km1l + 2
        tmpl = jnp.take_along_axis(s_row, jnp.minimum(km3l, pverp), axis=-1, mode="clip") * jnp.minimum(
            1, pverp2 - km3l).astype(jnp.float64)
        fsul_k2 = jnp.take_along_axis(fsul, k2v, axis=1, mode="clip")
        fclt = jnp.take_along_axis(fclt4[:, None, :], jnp.clip(km1l, 0, pver), axis=-1, mode="clip")
        X = ((fclt + s_k2k3[..., None]) - tmpl) - fsul_k2[..., None]
        flux0 = jnp.take_along_axis(ful, k2v, axis=1, mode="clip")
        flux = run_config(ks, nxsk, emx0, X, fsul_k2, flux0, False)
        inreg = (lev[None, :] >= k1r[:, None]) & (lev[None, :] <= k2r[:, None]) & act[:, None]
        upd = jnp.zeros((ncol, pverp + 1), bool).at[:, 1:pverp].set(inreg)
        newv = jnp.zeros((ncol, pverp + 1)).at[:, 1:pverp].set(flux)
        return jnp.where(upd, newv, ful)

    ful = jax.lax.fori_loop(0, nmax, up_region, ful0)

    fdl = jnp.where(cloudy[:, None], fdl, fsdl)
    ful = jnp.where(cloudy[:, None], ful, fsul)
    fdl, ful, fsdl, fsul = fdl[:, 1:], ful[:, 1:], fsdl[:, 1:], fsul[:, 1:]
    dp = pint_cgs[:, :-1] - pint_cgs[:, 1:]
    qrl = (ful[:, :-1] - fdl[:, :-1] - ful[:, 1:] + fdl[:, 1:]) * lit(1.e-4) * c.gravit / dp
    qrlcs = (fsul[:, :-1] - fsdl[:, :-1] - fsul[:, 1:] + fsdl[:, 1:]) * lit(1.e-4) * c.gravit / dp
    return LwFluxes(qrl=qrl, qrlcs=qrlcs, flup=ful, flupc=fsul, fldn=fdl, fldnc=fsdl,
                    flns=ful[:, -1] - fdl[:, -1], flnt=ful[:, 0] - fdl[:, 0], flnsc=fsul[:, -1] - fsdl[:, -1],
                    flntc=fsul[:, 0] - fsdl[:, 0], flwds=fdl[:, -1], flut=ful[:, 0], flutc=fsul[:, 0])
