"""WRF CAM longwave absorptivities ``radabs`` (+ ``trcab`` / ``trcabn``) as a column-batched float64 JAX kernel.

Source: pristine WRF V4.7.1 ``phys/module_ra_cam.F`` ``radabs`` (lines 2196-3602) and ``phys/module_ra_cam_support.F``
``trcab`` (437-721) / ``trcabn`` (725-1014).  CAM computes in ``real(r8)``: the port is float64, every unsuffixed
Fortran literal is a single-precision constant widened to double (:func:`lit`), Fortran's left-to-right evaluation
order and parentheses are kept, and default-REAL constant expressions (``r293 = 1./293.`` ...) are folded in float32.

Layout: CAM vertical order (index 0 = model top), column batch on the leading axis.

* non-adjacent pairs ``(k1, k2)`` are evaluated at once on ``(ncol, pverp, pverp)`` arrays (axis 1 = k1, axis 2 = k2);
  the diagonal ``k1 == k2`` is WRF's ``inf`` (= ``lit(1.e20)``) marker;
* nearest-layer sub-divided absorptivities on ``(ncol, pver, 4)`` arrays (axis 1 = k2, axis 2 = kn).

WRF quirks reproduced on purpose:

* ``radabs`` reads the MODULE ``co2vmr`` (``3.550e-4`` single literal), not the CLWRF value camrad passes to radctl;
* ``ntoplw = 1``; ``strat_volcanic = .false.`` so every ``aer_trn_ttl`` / ``aer_trn_ngh`` factor is exactly 1.0 and the
  multiplications by it are dropped (exact); ``tw(:,1)`` is multiplied by ``0.7*1.0 + 0.3*1.0`` which is exactly 1.0 in
  double (asserted below);
* non-adjacent CO2: ``sqti`` is replaced by ``sqrt(tlayr(k2))`` when ``k2 >= k1`` (after f1/f2/f3co2 are formed);
* the two ``tco2`` forms differ: non-adjacent ``1/(1 + 10*(u7/sqrt(..)))``, nearest ``1/(1 + (10*u7)/sqrt(..))``;
* ``trcab`` an2o2 multiplies ``th2o*tco2``, ``trcabn`` ``tco2*th2o`` (order kept).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.ra_cam_common import (
    COEFH,
    COEFJ,
    COEFK,
    CONST,
    DLP_H2O,
    DLU_H2O,
    DRH_H2O,
    DTE_H2O,
    DTP_H2O,
    FAT,
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
    CamAbsTables,
    fh2oself,
    lit,
    phi,
    psi,
)

_f32 = np.float32
F64 = jnp.float64

INF = lit(1.0e20)                                   # module_ra_cam_support: real(r8), parameter :: inf = 1.e20
# radabs local constants: default-REAL expressions folded in float32, then stored in r8.
R293 = float(_f32(1.0) / _f32(293.0))
R250 = float(_f32(1.0) / _f32(250.0))
R3205 = float(_f32(1.0) / _f32(0.3205))
R300 = float(_f32(1.0) / _f32(300.0))
# r8 expressions (sslp is r8).
RSSLP = 1.0 / CONST.sslp
R2SSLP = 1.0 / (2.0 * CONST.sslp)
FDIF = lit(1.66)
SSLP_MKS = CONST.sslp / 10.0
OMEPS = 1.0 - CONST.epsilo
CO2VMR = CONST.co2vmr_module                         # MODULE co2vmr (never updated by camrad)

# trcab/trcabn data statements (r8 arrays initialised from single-precision literals).
_G1 = tuple(lit(x) for x in (0.0468556, 0.0397454, 0.0407664, 0.0304380, 0.0540398, 0.0321962))
_G2 = tuple(lit(x) for x in (14.4832, 4.30242, 5.23523, 3.25342, 0.698935, 16.5599))
_G3 = tuple(lit(x) for x in (26.1898, 18.4476, 15.3633, 12.1927, 9.14992, 8.07092))
_G4 = tuple(lit(x) for x in (0.0261782, 0.0369516, 0.0307266, 0.0243854, 0.0182932, 0.0161418))
_AB = tuple(lit(x) for x in (3.0857e-2, 2.3524e-2, 1.7310e-2, 2.6661e-2, 2.8074e-2, 2.2915e-2))
_BB = tuple(lit(x) for x in (-1.3512e-4, -6.8320e-5, -3.2609e-5, -1.0228e-5, -9.5743e-5, -1.0304e-4))
_ABP = tuple(lit(x) for x in (2.9129e-2, 2.4101e-2, 1.9821e-2, 2.6904e-2, 2.9458e-2, 1.9892e-2))
_BBP = tuple(lit(x) for x in (-1.3139e-4, -5.5688e-5, -4.6380e-5, -8.0362e-5, -1.0115e-4, -8.8061e-5))

# tw(:,1) aerosol factor 0.7*aer(650-800) + 0.3*aer(800-1000) with aer == 1.0: exactly 1.0 in double -> dropped.
assert lit(0.7) * 1.0 + lit(0.3) * 1.0 == 1.0


# -------------------------------------------------------------------------------------------------------------------- #
# Helpers                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #
def _dbvt(t):
    """radabs statement function dbvt(t): Planck-function temperature derivative for O3 (single literals)."""

    return (lit(-2.8911366682e-4) + (lit(2.3771251896e-6) + lit(1.1305188929e-10) * t) * t) / \
        (1.0 + (lit(-6.1364820707e-3) + lit(1.5550319767e-5) * t) * t)


def _cell(dvar, n):
    """``i = min(max(int(aint(dvar)) + 1, 1), n - 1)`` (0-based lower index), ``w = dvar - floor(dvar)``, ``1 - w``."""

    i = jnp.clip(jnp.trunc(dvar).astype(jnp.int32), 0, n - 2)
    w = dvar - jnp.floor(dvar)
    return i, w, 1.0 - w


def _wtrh(wtp, wtp1, wte, wte1, wrh, wrh1):
    """Band-independent weights ``w_Y_ZR`` (Y: T_p, Z: T_e, R: RH; 0 = upper weight, 1 = lower weight)."""

    wtpv, wtev, wrhv = (wtp, wtp1), (wte, wte1), (wrh, wrh1)
    out = {}
    for y in (0, 1):
        for z in (0, 1):
            wyz = wtpv[y] * wtev[z]                 # w_Y_Z_
            for r in (0, 1):
                out[(y, z, r)] = wyz * wrhv[r]       # w_Y_ZR
    return out


# Fortran term order of the 32-term sums: (ip,itp) (ip,itp1) (ip1,itp) (ip1,itp1) x (iu, iu1) x (ite,irh) (ite,irh1)
# (ite1,irh) (ite1,irh1).  Index "1" = lower table node (weight w*1), "0" = upper node (weight w*0).
_PT_ORDER = ((1, 1), (1, 0), (0, 1), (0, 0))
_U_ORDER = (1, 0)
_ER_ORDER = ((1, 1), (1, 0), (0, 1), (0, 0))


def _interp32(flat, ip, wp, wp1, itp, w3, iu, wu, wu1, ite, irh):
    """sum_{32} tab(ip.., itp.., iu.., ite.., irh..) * wXY_ZR * wu..  accumulated left to right like the Fortran."""

    wpv, wuv = (wp, wp1), (wu, wu1)
    acc = None
    for x, y in _PT_ORDER:
        bpt = (ip + (1 - x)) * N_TP + (itp + (1 - y))
        wxy = {(z, r): wpv[x] * w3[(y, z, r)] for z, r in _ER_ORDER}    # wXY_ZR
        for uu in _U_ORDER:
            bptu = bpt * N_U + (iu + (1 - uu))
            for z, r in _ER_ORDER:
                idx = (bptu * N_TE + (ite + (1 - z))) * N_RH + (irh + (1 - r))
                term = jnp.take(flat, idx, mode="clip") * wxy[(z, r)] * wuv[uu]
                acc = term if acc is None else acc + term
    return acc


def _fa(ib, te1, te2, te3, te4, te5):
    j = ib - 1
    return FAT[0, j] + FAT[1, j] * te1 + FAT[2, j] * te2 + FAT[3, j] * te3 + FAT[4, j] * te4 + FAT[5, j] * te5


def _log_cell(x, xmin, lmax, lmin, dl, n):
    lg = jnp.minimum(jnp.log10(jnp.maximum(x, xmin)), lmax)
    return _cell((lg - lmin) / dl, n)


def _clip01(x):
    return jnp.minimum(jnp.maximum(x, 0.0), 1.0)


def _func(u, b):
    """trcab/trcabn statement function func(u,b) = u/sqrt(4.0 + u*(1.0 + 1.0 / b))."""

    return u / jnp.sqrt(4.0 + u * (1.0 + 1.0 / b))


def _h2o_weights(t_p, t_e, rh_path):
    itp, wtp, wtp1 = _cell((t_p - MIN_TP_H2O) / DTP_H2O, N_TP)
    ite, wte, wte1 = _cell((t_e - MIN_TE_H2O) / DTE_H2O, N_TE)
    irh, wrh, wrh1 = _cell((rh_path - MIN_RH_H2O) / DRH_H2O, N_RH)
    return itp, ite, irh, _wtrh(wtp, wtp1, wte, wte1, wrh, wrh1)


def _esat_qsat(t_p, pnew_mks, est):
    iest = (jnp.floor(t_p) - MIN_TP_H2O).astype(jnp.int32)        # integer floor - r8 160 -> int (exact)
    e0 = jnp.take(est, iest, mode="clip")
    e1 = jnp.take(est, iest + 1, mode="clip")
    esx = e0 + (e1 - e0) * (t_p - MIN_TP_H2O - iest.astype(F64))
    qsx = CONST.epsilo * esx / (pnew_mks - OMEPS * esx)
    return qsx


def _h2o_overlap_tw(tt, pnew, w1_base, ds2c, duptyp):
    """trcab/trcabn h2o transmission tw(:,l), l = 1..6 (aerosol factors == 1 dropped)."""

    tw = []
    for l in range(6):
        psi1 = jnp.exp(_ABP[l] * tt + _BBP[l] * tt * tt)
        phi1 = jnp.exp(_AB[l] * tt + _BB[l] * tt * tt)
        p1 = pnew * (psi1 / phi1) / CONST.sslp
        w1 = w1_base * phi1
        tw.append(jnp.exp(-_G1[l] * p1 * (jnp.sqrt(1.0 + _G2[l] * (w1 / p1)) - 1.0) - _G3[l] * ds2c - _G4[l] * duptyp))
    return tw


# -------------------------------------------------------------------------------------------------------------------- #
# trcab: non-nearest layers                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #
def _trcab(g1, g2, to3co2, pnm1, pnm2, dw, pnew, dplh2o, abp, tco2, th2o, to3):
    """Pair form: ``g1[name]`` / ``g2[name]`` = gas paths at k1 / k2; ``abp(w)`` = abplnk1(w, :, k2) (1-based w)."""

    sqti = jnp.sqrt(to3co2)
    tt = jnp.abs(to3co2 - 250.0)
    ds2c = jnp.abs(g1["s2c"] - g2["s2c"])
    duptyp = jnp.abs(g1["uptype"] - g2["uptype"])
    tw = _h2o_overlap_tw(tt, pnew, dw, ds2c, duptyp)

    du1 = jnp.abs(g1["ucfc11"] - g2["ucfc11"])
    du2 = jnp.abs(g1["ucfc12"] - g2["ucfc12"])
    tcfc3 = jnp.exp(-lit(175.005) * du1)
    tcfc4 = jnp.exp(-lit(1202.18) * du1)
    tcfc6 = jnp.exp(-lit(5786.73) * du2)
    tcfc7 = jnp.exp(-lit(2873.51) * du2)
    tcfc8 = jnp.exp(-lit(2085.59) * du2)
    acfc1 = 50.0 * (1.0 - jnp.exp(-lit(54.09) * du1)) * tw[0] * abp(7)
    acfc2 = 60.0 * (1.0 - jnp.exp(-lit(5130.03) * du1)) * tw[1] * abp(8)
    acfc3 = 60.0 * (1.0 - tcfc3) * tw[3] * tcfc6 * abp(9)
    acfc4 = 100.0 * (1.0 - tcfc4) * tw[4] * abp(10)
    acfc5 = 45.0 * (1.0 - jnp.exp(-lit(1272.35) * du2)) * tw[2] * abp(11)
    acfc6 = 50.0 * (1.0 - tcfc6) * tw[3] * abp(12)
    acfc7 = 80.0 * (1.0 - tcfc7) * tw[4] * tcfc4 * abp(13)
    acfc8 = 70.0 * (1.0 - tcfc8) * tw[5] * abp(14)

    tlw = jnp.exp(-1.0 * jnp.sqrt(dplh2o))
    duch4 = jnp.abs(g1["uch4"] - g2["uch4"])
    dbetac = jnp.abs(g1["bch4"] - g2["bch4"]) / duch4
    ach4 = lit(6.00444) * sqti * jnp.log(1.0 + _func(duch4, dbetac)) * tlw * abp(3)
    tch4 = 1.0 / (1.0 + lit(0.02) * _func(duch4, dbetac))

    du01 = jnp.abs(g1["un2o0"] - g2["un2o0"])
    du11 = jnp.abs(g1["un2o1"] - g2["un2o1"])
    dbeta01 = jnp.abs(g1["bn2o0"] - g2["bn2o0"]) / du01
    dbeta11 = jnp.abs(g1["bn2o1"] - g2["bn2o1"]) / du11
    an2o1 = lit(2.35558) * sqti * jnp.log(1.0 + _func(du01, dbeta01) + _func(du11, dbeta11)) * tlw * tch4 * abp(4)
    du02 = lit(0.100090) * du01
    du12 = lit(0.0992746) * du11
    dbeta02 = lit(0.964282) * dbeta01
    an2o2 = lit(2.65581) * sqti * jnp.log(1.0 + _func(du02, dbeta02) + _func(du12, dbeta02)) * th2o * tco2 * abp(5)
    du03 = lit(0.0333767) * du01
    dbeta03 = lit(0.982143) * dbeta01
    an2o3 = lit(2.54034) * sqti * jnp.log(1.0 + _func(du03, dbeta03)) * tw[5] * tcfc8 * abp(6)

    du11 = jnp.abs(g1["uco211"] - g2["uco211"])
    du12 = jnp.abs(g1["uco212"] - g2["uco212"])
    du13 = jnp.abs(g1["uco213"] - g2["uco213"])
    dbetc1 = lit(2.97558) * jnp.abs(pnm1 + pnm2) / (2.0 * CONST.sslp * sqti)
    dbetc2 = 2.0 * dbetc1
    aco21 = (lit(3.7571) * sqti * jnp.log(1.0 + _func(du11, dbetc1) + _func(du12, dbetc2) + _func(du13, dbetc2))
             * to3 * tw[4] * tcfc4 * tcfc7 * abp(2))
    du21 = jnp.abs(g1["uco221"] - g2["uco221"])
    du22 = jnp.abs(g1["uco222"] - g2["uco222"])
    du23 = jnp.abs(g1["uco223"] - g2["uco223"])
    aco22 = (lit(3.8443) * sqti * jnp.log(1.0 + _func(du21, dbetc1) + _func(du22, dbetc1) + _func(du23, dbetc2))
             * tw[3] * tcfc3 * tcfc6 * abp(1))
    return (acfc1 + acfc2 + acfc3 + acfc4 + acfc5 + acfc6 + acfc7 + acfc8 + an2o1 + an2o2 + an2o3 + ach4
            + aco21 + aco22)


# -------------------------------------------------------------------------------------------------------------------- #
# trcabn: nearest layer                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #
def _trcabn(dgas, tbar, bpl, winpl, pinpl, tco2, th2o, to3, dw, up2, pnew, uinpl):
    """Nearest-layer form: ``dgas[name] = abs(x(k2+1) - x(k2))``; ``bpl(w)`` = bplnk(w, :, kn) (1-based w)."""

    sqti = jnp.sqrt(tbar)
    rsqti = 1.0 / sqti
    tt = jnp.abs(tbar - 250.0)
    ds2c = dgas["s2c"] * uinpl
    duptyp = dgas["uptype"] * uinpl
    tw = _h2o_overlap_tw(tt, pnew, dw * winpl, ds2c, duptyp)

    du1 = dgas["ucfc11"] * winpl
    du2 = dgas["ucfc12"] * winpl
    tcfc3 = jnp.exp(-lit(175.005) * du1)
    tcfc4 = jnp.exp(-lit(1202.18) * du1)
    tcfc6 = jnp.exp(-lit(5786.73) * du2)
    tcfc7 = jnp.exp(-lit(2873.51) * du2)
    tcfc8 = jnp.exp(-lit(2085.59) * du2)
    acfc1 = 50.0 * (1.0 - jnp.exp(-lit(54.09) * du1)) * tw[0] * bpl(7)
    acfc2 = 60.0 * (1.0 - jnp.exp(-lit(5130.03) * du1)) * tw[1] * bpl(8)
    acfc3 = 60.0 * (1.0 - tcfc3) * tw[3] * tcfc6 * bpl(9)
    acfc4 = 100.0 * (1.0 - tcfc4) * tw[4] * bpl(10)
    acfc5 = 45.0 * (1.0 - jnp.exp(-lit(1272.35) * du2)) * tw[2] * bpl(11)
    acfc6 = 50.0 * (1.0 - tcfc6) * tw[3] * bpl(12)
    acfc7 = 80.0 * (1.0 - tcfc7) * tw[4] * tcfc4 * bpl(13)
    acfc8 = 70.0 * (1.0 - tcfc8) * tw[5] * bpl(14)

    tlw = jnp.exp(-1.0 * jnp.sqrt(up2))
    duch4 = dgas["uch4"] * winpl
    dbetac = lit(2.94449) * pinpl * rsqti / CONST.sslp
    ach4 = lit(6.00444) * sqti * jnp.log(1.0 + _func(duch4, dbetac)) * tlw * bpl(3)
    tch4 = 1.0 / (1.0 + lit(0.02) * _func(duch4, dbetac))

    du01 = dgas["un2o0"] * winpl
    du11 = dgas["un2o1"] * winpl
    dbeta01 = lit(19.399) * pinpl * rsqti / CONST.sslp
    dbeta11 = dbeta01
    an2o1 = lit(2.35558) * sqti * jnp.log(1.0 + _func(du01, dbeta01) + _func(du11, dbeta11)) * tlw * tch4 * bpl(4)
    du02 = lit(0.100090) * du01
    du12 = lit(0.0992746) * du11
    dbeta02 = lit(0.964282) * dbeta01
    an2o2 = lit(2.65581) * sqti * jnp.log(1.0 + _func(du02, dbeta02) + _func(du12, dbeta02)) * tco2 * th2o * bpl(5)
    du03 = lit(0.0333767) * du01
    dbeta03 = lit(0.982143) * dbeta01
    an2o3 = lit(2.54034) * sqti * jnp.log(1.0 + _func(du03, dbeta03)) * tw[5] * tcfc8 * bpl(6)

    du11 = dgas["uco211"] * winpl
    du12 = dgas["uco212"] * winpl
    du13 = dgas["uco213"] * winpl
    dbetc1 = lit(2.97558) * pinpl * rsqti / CONST.sslp
    dbetc2 = 2.0 * dbetc1
    aco21 = (lit(3.7571) * sqti * jnp.log(1.0 + _func(du11, dbetc1) + _func(du12, dbetc2) + _func(du13, dbetc2))
             * to3 * tw[4] * tcfc4 * tcfc7 * bpl(2))
    du21 = dgas["uco221"] * winpl
    du22 = dgas["uco222"] * winpl
    du23 = dgas["uco223"] * winpl
    aco22 = (lit(3.8443) * sqti * jnp.log(1.0 + _func(du21, dbetc1) + _func(du22, dbetc1) + _func(du23, dbetc2))
             * tw[3] * tcfc3 * tcfc6 * bpl(1))
    return (acfc1 + acfc2 + acfc3 + acfc4 + acfc5 + acfc6 + acfc7 + acfc8 + an2o1 + an2o2 + an2o3 + ach4
            + aco21 + aco22)


# -------------------------------------------------------------------------------------------------------------------- #
# Shared H2O 500-800 cm-1 line/continuum overlap (term7/term8/k21/k22/tr*)                                             #
# -------------------------------------------------------------------------------------------------------------------- #
def _terms78(dty):
    t71 = COEFJ[0, 0] + COEFJ[1, 0] * dty * (1.0 + CONST.c16 * dty)
    t81 = COEFK[0, 0] + COEFK[1, 0] * dty * (1.0 + CONST.c17 * dty)
    t72 = COEFJ[0, 1] + COEFJ[1, 1] * dty * (1.0 + CONST.c26 * dty)
    t82 = COEFK[0, 1] + COEFK[1, 1] * dty * (1.0 + CONST.c27 * dty)
    return t71, t81, t72, t82


def _tr_h2o(k21, k22, sqrtu, fwku, dtx, uc1):
    tr1 = jnp.exp(-(k21 * (sqrtu + CONST.fc1 * fwku)))
    tr2 = jnp.exp(-(k22 * (sqrtu + CONST.fc1 * fwku)))
    tr5 = jnp.exp(-((COEFH[0, 2] + COEFH[1, 2] * dtx) * uc1))
    tr6 = jnp.exp(-((COEFH[0, 3] + COEFH[1, 3] * dtx) * uc1))
    tr9 = tr1 * tr5
    tr10 = tr2 * tr6
    trab2 = lit(0.65) * tr9 + lit(0.35) * tr10
    return trab2, tr10                                                   # th2o = tr10


def _gas_names():
    return ("ucfc11", "ucfc12", "un2o0", "un2o1", "uch4", "uco211", "uco212", "uco213", "uco221", "uco222", "uco223",
            "uptype", "bn2o0", "bn2o1", "bch4", "s2c")


# -------------------------------------------------------------------------------------------------------------------- #
# radabs                                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #
def radabs(pbr, pnm, co2em, co2eml, tplnka, s2c, tcg, w, h2otr, plco2, plh2o, co2t, tint, tlayr, plol, plos, pmln,
           piln, ucfc11, ucfc12, un2o0, un2o1, uch4, uco211, uco212, uco213, uco221, uco222, uco223, uptype, bn2o0,
           bn2o1, bch4, abplnk1, abplnk2, plh2ob, wb, tables: CamAbsTables):
    """CAM ``radabs``: returns ``abstot (ncol, pverp, pverp)`` [k1, k2] and ``absnxt (ncol, pver, 4)`` [k, kn].

    Shapes: ``pbr``, ``co2eml``, ``pmln`` (ncol, pver); ``abplnk1/2`` (ncol, 14, pverp); ``plh2ob``/``wb``
    (ncol, 2, pverp); every other array (ncol, pverp).  Pressures in dynes/cm2 (cgs), ``pmln``/``piln`` = ln(Pa).
    """

    a = lambda x: jnp.asarray(x, F64)
    (pbr, pnm, co2em, co2eml, tplnka, s2c, tcg, w, h2otr, plco2, plh2o, co2t, tint, tlayr, plol, plos, pmln, piln,
     abplnk1, abplnk2, plh2ob, wb) = map(a, (pbr, pnm, co2em, co2eml, tplnka, s2c, tcg, w, h2otr, plco2, plh2o, co2t,
                                           tint, tlayr, plol, plos, pmln, piln, abplnk1, abplnk2, plh2ob, wb))
    gas = dict(zip(_gas_names(), map(a, (ucfc11, ucfc12, un2o0, un2o1, uch4, uco211, uco212, uco213, uco221, uco222,
                                         uco223, uptype, bn2o0, bn2o1, bch4, s2c))))
    ah2onw = jnp.asarray(tables.ah2onw, F64).reshape(-1)
    ah2ow = jnp.asarray(tables.ah2ow, F64).reshape(-1)
    cn_ah2ow = jnp.asarray(tables.cn_ah2ow, F64).reshape(-1)
    ln_ah2ow = jnp.asarray(tables.ln_ah2ow, F64).reshape(-1)
    est = jnp.asarray(tables.estblh2o, F64)

    ncol, pverp = pnm.shape
    pver = pverp - 1
    rga = CONST.rga

    dbvtit = _dbvt(tint)                                 # (ncol, pverp)   dbvtit(k), k = 1..pverp
    dbvtly = _dbvt(tlayr[:, 1:])                         # (ncol, pver)    dbvtly(k) = dbvt(tlayr(k+1))

    # ------------------------------------------------------------------------------------------------------------- #
    # Non-adjacent layer absorptivity: all (k1, k2) pairs, axis 1 = k1, axis 2 = k2.                                 #
    # ------------------------------------------------------------------------------------------------------------- #
    def p1(x):
        return x[:, :, None]

    def p2(x):
        return x[:, None, :]

    kk = jnp.arange(pverp)
    k2_lt_k1 = kk[None, :] < kk[:, None]                 # [k1, k2]: k2 < k1
    diag = kk[None, :] == kk[:, None]

    dplh2o = p1(plh2o) - p2(plh2o)
    u = jnp.abs(dplh2o)
    sqrtu = jnp.sqrt(u)
    ds2c = jnp.abs(p1(s2c) - p2(s2c))
    dw = jnp.abs(p1(w) - p2(w))
    uc1 = (ds2c + lit(1.7e-3) * u) * (1.0 + 2.0 * ds2c) / (1.0 + 15.0 * ds2c)
    pch2o = ds2c
    pnew = u / dw
    pnew_mks = pnew * SSLP_MKS
    tpatha = jnp.abs(p1(tcg) - p2(tcg)) / dw
    t_p = jnp.minimum(jnp.maximum(tpatha, MIN_TP_H2O), MAX_TP_H2O)
    qsx = _esat_qsat(t_p, pnew_mks, est)
    q_path = dw / jnp.abs(p1(pnm) - p2(pnm)) / rga
    ub1 = jnp.abs(p1(plh2ob[:, 0]) - p2(plh2ob[:, 0])) / psi(t_p, 1)
    ub2 = jnp.abs(p1(plh2ob[:, 1]) - p2(plh2ob[:, 1])) / psi(t_p, 2)
    pnewb1 = ub1 / jnp.abs(p1(wb[:, 0]) - p2(wb[:, 0])) * phi(t_p, 1)
    pnewb2 = ub2 / jnp.abs(p1(wb[:, 1]) - p2(wb[:, 1])) * phi(t_p, 2)
    dtx = p2(tplnka) - 250.0
    dty = tpatha - 250.0
    fwk = CONST.fwcoef + CONST.fwc1 / (1.0 + CONST.fwc2 * u)
    fwku = fwk * u

    te1 = p2(tplnka)
    te2 = te1 * te1
    te3 = te2 * te1
    te4 = te3 * te1
    te5 = te4 * te1
    t_e = jnp.minimum(jnp.maximum(p2(tplnka) - t_p, MIN_TE_H2O), MAX_TE_H2O)
    rh_path = jnp.minimum(jnp.maximum(q_path / qsx, MIN_RH_H2O), MAX_RH_H2O)
    itp, ite, irh, w3 = _h2o_weights(t_p, t_e, rh_path)
    fch2o = fh2oself(t_p)
    uch2o = (pch2o * CONST.epsilo) / (q_path * fch2o)

    # band 1 (non-window): ah2onw
    uvar1 = ub1 * FDIF
    iu, wu, wu1 = _log_cell(uvar1, MIN_U_H2O, MAX_LU_H2O, MIN_LU_H2O, DLU_H2O, N_U)
    ip, wp, wp1 = _log_cell(pnewb1, MIN_P_H2O, MAX_LP_H2O, MIN_LP_H2O, DLP_H2O, N_P)
    a_star = _interp32(ah2onw, ip, wp, wp1, itp, w3, iu, wu, wu1, ite, irh)
    abso1 = _clip01(_fa(1, te1, te2, te3, te4, te5) * (1.0 - (1.0 - a_star)))          # * aer_trn_ttl (== 1)
    abso1 = jnp.where(uvar1 < MIN_U_H2O, abso1 * (uvar1 / MIN_U_H2O), abso1)

    # band 2 (window): line ln_ah2ow x continuum cn_ah2ow
    uvar2 = ub2 * FDIF
    iu, wu, wu1 = _log_cell(uvar2, MIN_U_H2O, MAX_LU_H2O, MIN_LU_H2O, DLU_H2O, N_U)
    ip, wp, wp1 = _log_cell(pnewb2, MIN_P_H2O, MAX_LP_H2O, MIN_LP_H2O, DLP_H2O, N_P)
    iuc, wuc, wuc1 = _log_cell(uch2o * FDIF, MIN_U_H2O, MAX_LU_H2O, MIN_LU_H2O, DLU_H2O, N_U)
    l_star = _interp32(ln_ah2ow, ip, wp, wp1, itp, w3, iu, wu, wu1, ite, irh)
    c_star = _interp32(cn_ah2ow, ip, wp, wp1, itp, w3, iuc, wuc, wuc1, ite, irh)
    abso2 = _clip01(_fa(2, te1, te2, te3, te4, te5) * (1.0 - l_star * c_star))          # * aer_trn_ttl (== 1)
    abso2 = jnp.where(uvar2 < MIN_U_H2O, abso2 * (uvar2 / MIN_U_H2O), abso2)

    # 500-800 cm-1 h2o rotation band overlap with co2
    t71, t81, t72, t82 = _terms78(dty)
    k21 = t71 + t81 / (1.0 + (CONST.c30 + CONST.c31 * (dty - 10.0) * (dty - 10.0)) * sqrtu)
    k22 = t72 + t82 / (1.0 + (CONST.c28 + CONST.c29 * (dty - 10.0)) * sqrtu)
    trab2, th2o = _tr_h2o(k21, k22, sqrtu, fwku, dtx, uc1)
    to3h2o = jnp.where(k2_lt_k1, p1(h2otr) / p2(h2otr), p2(h2otr) / p1(h2otr))

    # o3 9.6 micrometer band
    dpnm = p1(pnm) - p2(pnm)
    to3co2 = (p1(pnm) * p1(co2t) - p2(pnm) * p2(co2t)) / dpnm
    te = jnp.power(to3co2 * R293, lit(0.7))
    dplos = p1(plos) - p2(plos)
    dplol = p1(plol) - p2(plol)
    u1 = lit(18.29) * jnp.abs(dplos) / te
    u2 = lit(0.5649) * jnp.abs(dplos) / te
    rphat = dplol / dplos
    tlocal = p2(tint)
    tcrfac = jnp.sqrt(tlocal * R250) * te
    beta = R3205 * (rphat + CONST.dpfo3 * tcrfac)
    realnu = te / beta
    tmp1 = u1 / jnp.sqrt(4.0 + u1 * (1.0 + realnu))
    tmp2 = u2 / jnp.sqrt(4.0 + u2 * (1.0 + realnu))
    o3bndi = 74.0 * te * jnp.log(1.0 + tmp1 + tmp2)
    abso3 = o3bndi * to3h2o * p2(dbvtit)
    to3 = 1.0 / (1.0 + lit(0.1) * tmp1 + lit(0.1) * tmp2)

    # co2 15 micrometer band system
    sqwp = jnp.sqrt(jnp.abs(p1(plco2) - p2(plco2)))
    et = jnp.exp(-480.0 / to3co2)
    sqti = jnp.sqrt(to3co2)
    rsqti = 1.0 / sqti
    et2 = et * et
    et4 = et2 * et2
    omet = 1.0 - 1.5 * et2
    f1co2 = lit(899.70) * omet * (1.0 + lit(1.94774) * et + lit(4.73486) * et2) * rsqti
    f1sqwp = f1co2 * sqwp
    t1co2 = 1.0 / (1.0 + (lit(245.18) * omet * sqwp * rsqti))
    oneme = 1.0 - et2
    alphat = oneme * oneme * oneme * rsqti
    pi = jnp.abs(dpnm)
    wco2 = lit(2.5221) * CO2VMR * pi * rga
    u7 = lit(4.9411e4) * alphat * et2 * wco2
    u8 = lit(3.9744e4) * alphat * et4 * wco2
    u9 = lit(1.0447e5) * alphat * et4 * et2 * wco2
    u13 = lit(2.8388e3) * alphat * et4 * wco2
    tpath = to3co2
    tlocal = p2(tint)
    tcrfac = jnp.sqrt(tlocal * R250 * tpath * R300)
    posqt = ((p2(pnm) + p1(pnm)) * R2SSLP + CONST.dpfco2 * tcrfac) * rsqti
    rbeta7 = 1.0 / (lit(5.3228) * posqt)
    rbeta8 = 1.0 / (lit(10.6576) * posqt)
    rbeta9 = rbeta7
    rbeta13 = rbeta9
    f2co2 = ((u7 / jnp.sqrt(4.0 + u7 * (1.0 + rbeta7))) + (u8 / jnp.sqrt(4.0 + u8 * (1.0 + rbeta8)))
             + (u9 / jnp.sqrt(4.0 + u9 * (1.0 + rbeta9))))
    f3co2 = u13 / jnp.sqrt(4.0 + u13 * (1.0 + rbeta13))
    sqti = jnp.where(~k2_lt_k1, p2(jnp.sqrt(tlayr)), sqti)                      # if (k2 >= k1) sqti = sqrt(tlayr(k2))
    tmp1 = jnp.log(1.0 + f1sqwp)
    tmp2 = jnp.log(1.0 + f2co2)
    tmp3 = jnp.log(1.0 + f3co2)
    absbnd = (tmp1 + 2.0 * t1co2 * tmp2 + 2.0 * tmp3) * sqti
    abso4 = trab2 * p2(co2em) * absbnd
    tco2 = 1.0 / (1.0 + 10.0 * (u7 / jnp.sqrt(4.0 + u7 * (1.0 + rbeta7))))

    g1 = {k: p1(v) for k, v in gas.items()}
    g2 = {k: p2(v) for k, v in gas.items()}
    abstrc = _trcab(g1, g2, to3co2, p1(pnm), p2(pnm), dw, pnew, u, lambda wv: p2(abplnk1[:, wv - 1]), tco2, th2o, to3)

    abstot = abso1 + abso2 + abso3 + abso4 + abstrc
    abstot = jnp.where(diag, INF, abstot)

    # ------------------------------------------------------------------------------------------------------------- #
    # Nearest layer: axis 1 = k2 (1..pver), axis 2 = kn (1..4).                                                      #
    # ------------------------------------------------------------------------------------------------------------- #
    lo = lambda x: x[:, :-1]                             # interface quantity at k2
    hi = lambda x: x[:, 1:]                              # interface quantity at k2+1
    st = lambda *xs: jnp.stack(xs, axis=-1)              # (ncol, pver) x 4 -> (ncol, pver, 4)
    c = lambda x: x[:, :, None]                          # kn-independent (ncol, pver) -> broadcast

    tbar1 = 0.5 * (hi(tint) + hi(tlayr))
    emm1 = 0.5 * (hi(co2em) + co2eml)
    tbar2 = 0.5 * (hi(tlayr) + lo(tint))
    emm2 = 0.5 * (lo(co2em) + co2eml)
    tbar3 = 0.5 * (tbar2 + tbar1)
    o3emm1 = 0.5 * (hi(dbvtit) + dbvtly)
    o3emm2 = 0.5 * (lo(dbvtit) + dbvtly)
    tbar = st(tbar1, tbar2, tbar3, tbar3)
    emm = st(emm1, emm2, emm1, emm2)
    o3emm = st(o3emm1, o3emm2, o3emm1, o3emm2)
    temh2o = st(tbar1, tbar2, tbar1, tbar2)
    dpnm_n = hi(pnm) - lo(pnm)
    bpl1 = 0.5 * (abplnk1[:, :, 1:] + abplnk2[:, :, :-1])       # (ncol, 14, pver)
    bpl2 = 0.5 * (abplnk1[:, :, :-1] + abplnk2[:, :, :-1])
    bplnk = jnp.stack((bpl1, bpl2, bpl1, bpl2), axis=-1)       # (ncol, 14, pver, 4)

    pnmsq = pnm * pnm
    rdpnmsq = 1.0 / (hi(pnmsq) - lo(pnmsq))
    rdpnm = 1.0 / dpnm_n
    pp1 = 0.5 * (pbr + hi(pnm))
    pp2 = 0.5 * (pbr + lo(pnm))
    uinpl = st((hi(pnmsq) - pp1 * pp1) * rdpnmsq,
               -(lo(pnmsq) - pp2 * pp2) * rdpnmsq,
               -(lo(pnmsq) - pp1 * pp1) * rdpnmsq,
               (hi(pnmsq) - pp2 * pp2) * rdpnmsq)
    winpl = st((0.5 * (hi(pnm) - pbr)) * rdpnm,
               (0.5 * (-lo(pnm) + pbr)) * rdpnm,
               (0.5 * (hi(pnm) + pbr) - lo(pnm)) * rdpnm,
               (0.5 * (-lo(pnm) - pbr) + hi(pnm)) * rdpnm)
    ztmp1 = 1.0 / (hi(piln) - lo(piln))
    ztmp2 = hi(piln) - pmln
    ztmp3 = lo(piln) - pmln
    zinpl = st((0.5 * ztmp2) * ztmp1,
               (-0.5 * ztmp3) * ztmp1,
               (0.5 * ztmp2 - ztmp3) * ztmp1,
               (ztmp2 - 0.5 * ztmp3) * ztmp1)
    pinpl = st(0.5 * (pp1 + hi(pnm)),
               0.5 * (pp2 + lo(pnm)),
               0.5 * (pp1 + lo(pnm)),
               0.5 * (pp2 + hi(pnm)))

    u = uinpl * c(jnp.abs(lo(plh2o) - hi(plh2o)))
    sqrtu = jnp.sqrt(u)
    dw = c(jnp.abs(lo(w) - hi(w)))
    pnew = u / (winpl * dw)
    pnew_mks = pnew * SSLP_MKS
    t_p = jnp.minimum(jnp.maximum(tbar, MIN_TP_H2O), MAX_TP_H2O)
    qsx = _esat_qsat(t_p, pnew_mks, est)
    q_path = dw / jnp.abs(c(dpnm_n)) / rga
    ds2c = c(jnp.abs(lo(s2c) - hi(s2c)))
    uc1 = uinpl * ds2c
    uc1 = (uc1 + lit(1.7e-3) * u) * (1.0 + 2.0 * uc1) / (1.0 + 15.0 * uc1)
    dtx = temh2o - 250.0
    dty = tbar - 250.0
    fwk = CONST.fwcoef + CONST.fwc1 / (1.0 + CONST.fwc2 * u)
    fwku = fwk * u

    te1 = temh2o
    te2 = te1 * te1
    te3 = te2 * te1
    te4 = te3 * te1
    te5 = te4 * te1
    uvar = u * FDIF
    iu, wu, wu1 = _log_cell(uvar, MIN_U_H2O, MAX_LU_H2O, MIN_LU_H2O, DLU_H2O, N_U)
    ip, wp, wp1 = _log_cell(pnew, MIN_P_H2O, MAX_LP_H2O, MIN_LP_H2O, DLP_H2O, N_P)
    t_e = jnp.minimum(jnp.maximum(temh2o - t_p, MIN_TE_H2O), MAX_TE_H2O)
    rh_path = jnp.minimum(jnp.maximum(q_path / qsx, MIN_RH_H2O), MAX_RH_H2O)
    itp, ite, irh, w3 = _h2o_weights(t_p, t_e, rh_path)
    small = uvar < MIN_U_H2O
    a_star = _interp32(ah2onw, ip, wp, wp1, itp, w3, iu, wu, wu1, ite, irh)
    abso1 = _clip01(_fa(1, te1, te2, te3, te4, te5) * (1.0 - (1.0 - a_star)))          # * aer_trn_ngh (== 1)
    abso1 = jnp.where(small, abso1 * (uvar / MIN_U_H2O), abso1)
    a_star = _interp32(ah2ow, ip, wp, wp1, itp, w3, iu, wu, wu1, ite, irh)
    abso2 = _clip01(_fa(2, te1, te2, te3, te4, te5) * (1.0 - (1.0 - a_star)))          # * aer_trn_ngh (== 1)
    abso2 = jnp.where(small, abso2 * (uvar / MIN_U_H2O), abso2)

    t71, t81, t72, t82 = _terms78(dty)
    dtym10 = dty - 10.0
    denom = 1.0 + (CONST.c30 + CONST.c31 * dtym10 * dtym10) * sqrtu
    k21 = t71 + t81 / denom
    denom = 1.0 + (CONST.c28 + CONST.c29 * dtym10) * sqrtu
    k22 = t72 + t82 / denom
    trab2, th2o = _tr_h2o(k21, k22, sqrtu, fwku, dtx, uc1)

    te = jnp.power(tbar * R293, lit(0.7))
    dplos = c(jnp.abs(hi(plos) - lo(plos)))
    u1 = zinpl * lit(18.29) * dplos / te
    u2 = zinpl * lit(0.5649) * dplos / te
    tlocal = tbar
    tcrfac = jnp.sqrt(tlocal * R250) * te
    beta = R3205 * (pinpl * RSSLP + CONST.dpfo3 * tcrfac)
    realnu = te / beta
    tmp1 = u1 / jnp.sqrt(4.0 + u1 * (1.0 + realnu))
    tmp2 = u2 / jnp.sqrt(4.0 + u2 * (1.0 + realnu))
    o3bndi = 74.0 * te * jnp.log(1.0 + tmp1 + tmp2)
    abso3 = o3bndi * o3emm * c(hi(h2otr) / lo(h2otr))
    to3 = 1.0 / (1.0 + lit(0.1) * tmp1 + lit(0.1) * tmp2)

    dplco2 = c(hi(plco2) - lo(plco2))
    sqwp = jnp.sqrt(uinpl * dplco2)
    et = jnp.exp(-480.0 / tbar)
    sqti = jnp.sqrt(tbar)
    rsqti = 1.0 / sqti
    et2 = et * et
    et4 = et2 * et2
    omet = 1.0 - 1.5 * et2
    f1co2 = lit(899.70) * omet * (1.0 + lit(1.94774) * et + lit(4.73486) * et2) * rsqti
    f1sqwp = f1co2 * sqwp
    t1co2 = 1.0 / (1.0 + (lit(245.18) * omet * sqwp * rsqti))
    oneme = 1.0 - et2
    alphat = oneme * oneme * oneme * rsqti
    pi = jnp.abs(c(dpnm_n)) * winpl
    wco2 = lit(2.5221) * CO2VMR * pi * rga
    u7 = lit(4.9411e4) * alphat * et2 * wco2
    u8 = lit(3.9744e4) * alphat * et4 * wco2
    u9 = lit(1.0447e5) * alphat * et4 * et2 * wco2
    u13 = lit(2.8388e3) * alphat * et4 * wco2
    tpath = tbar
    tlocal = tbar
    tcrfac = jnp.sqrt((tlocal * R250) * (tpath * R300))
    posqt = (pinpl * RSSLP + CONST.dpfco2 * tcrfac) * rsqti
    rbeta7 = 1.0 / (lit(5.3228) * posqt)
    rbeta8 = 1.0 / (lit(10.6576) * posqt)
    rbeta9 = rbeta7
    rbeta13 = rbeta9
    f2co2 = (u7 / jnp.sqrt(4.0 + u7 * (1.0 + rbeta7)) + u8 / jnp.sqrt(4.0 + u8 * (1.0 + rbeta8))
             + u9 / jnp.sqrt(4.0 + u9 * (1.0 + rbeta9)))
    f3co2 = u13 / jnp.sqrt(4.0 + u13 * (1.0 + rbeta13))
    tmp1 = jnp.log(1.0 + f1sqwp)
    tmp2 = jnp.log(1.0 + f2co2)
    tmp3 = jnp.log(1.0 + f3co2)
    absbnd = (tmp1 + 2.0 * t1co2 * tmp2 + 2.0 * tmp3) * sqti
    abso4 = trab2 * emm * absbnd
    tco2 = 1.0 / (1.0 + 10.0 * u7 / jnp.sqrt(4.0 + u7 * (1.0 + rbeta7)))

    dgas = {k: c(jnp.abs(hi(v) - lo(v))) for k, v in gas.items()}
    abstrc = _trcabn(dgas, tbar, lambda wv: bplnk[:, wv - 1], winpl, pinpl, tco2, th2o, to3, dw, u, pnew, uinpl)

    absnxt = abso1 + abso2 + abso3 + abso4 + abstrc
    return abstot, absnxt


__all__ = ["INF", "radabs"]
