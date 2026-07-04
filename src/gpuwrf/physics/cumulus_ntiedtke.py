"""WRF New-Tiedtke cumulus (``cu_physics=16``) -- faithful fp64 column port.

Line-faithful single-column transcription of the UNMODIFIED WRF sources

* ``phys/physics_mmm/cu_ntiedtke.F90``  (CCPP core: ``cu_ntiedtke_run`` and the
  ``cumastrn`` / ``cuinin`` / ``cutypen`` / ``cuascn`` / ``cudlfsn`` /
  ``cuddrafn`` / ``cuflxn`` / ``cudtdqn`` / ``cududvn`` / ``cubasmcn`` /
  ``cuentrn`` / ``cuadjtqn`` subroutine tree and the ``foealfa`` / ``foelhm`` /
  ``foeewm`` / ``foedem`` / ``foeldcpm`` saturation functions), and
* ``phys/module_cu_ntiedtke.F``  (WRF driver: ``cu_ntiedtke_pre_run`` input
  conditioning / bottom-up->top-down flip and ``cu_ntiedtke_post_run`` tendency
  reconstruction),

compiled fp64 (``-DDOUBLE_PRECISION`` => ``kind_phys = selected_real_kind(15)``)
exactly as the committed oracle savepoints were produced
(``proofs/v013/oracle/cumulus/ntiedtke_build_and_run.sh`` ->
``proofs/v013/savepoints/cumulus/ntiedtke_case_*.json``).

This module is the NumPy *reference* transcription (sequential control flow
mirrors the Fortran statement-for-statement; every data-dependent branch is a
real branch, all arrays are 1-based top-down copies of the Fortran locals).  It
is validated against the fp64 WRF oracle savepoints in
``tests/test_ntiedtke_cumulus_oracle.py``.  The scheme is distinct from the old
modified Tiedtke (``cu_physics=6``, :mod:`gpuwrf.physics.cumulus_tiedtke`):
new trigger functions (Jakob-Siebesma / Bechtold), non-equilibrium CAPE
closure, new convective time scale, new entrainment/detrainment, Sundqvist
conversion, and cloud-scale pressure-gradient momentum transport.

Internal convention: Fortran 1-based, TOP-DOWN levels (index 1 = model top,
index ``km`` = lowest model level, ``km+1`` = surface interface).  Arrays are
allocated with a dead 0 slot so the transcription keeps Fortran indices.

No masking, no clamping beyond the WRF source's own limiters, no NaN guards.
"""

from __future__ import annotations

import numpy as np

# --- module cu_ntiedtke_common (verbatim constants) ---------------------------
_F = np.float64


def _f32(x):
    """WRF unsuffixed Fortran literal: default REAL (fp32), promoted to fp64."""
    return _F(np.float32(x))


T13 = _F(np.float32(1.0) / np.float32(3.0))
TMELT = _f32(273.16)
C1ES = _f32(610.78)
C3LES = _f32(17.2693882)
C3IES = _f32(21.875)
C4LES = _f32(35.86)
C4IES = _f32(7.66)

RTWAT = TMELT
RTBER = _F(np.float32(273.16) - np.float32(5.0))
RTICE = _F(np.float32(273.16) - np.float32(23.0))

MOMTRANS = 2
ENTRDD = _f32(2.0e-4)
CMFCMAX = 1.0
CMFCMIN = _f32(1.0e-10)
CMFDEPS = _f32(0.30)
ZDNOPRC = 2.0e4
CPRCON = _f32(1.4e-3)
PGCOEF = _f32(0.7)

NONEQUIL = True
LMFPEN = True
LMFMID = True
LMFSCV = True
LMFDD = True
LMFDUDV = True


class _Const:
    """cu_ntiedtke_init: derived constants from the WRF physical constants.

    The oracle driver passes G=9.81, R_D=287.0, R_V=461.6, CP=3.5*R_D,
    XLV=2.5e6, XLS=2.85e6, XLF=3.5e5 (ntiedtke_oracle_driver.f90).
    """

    def __init__(self, cp=_F(1004.5), rd=_F(287.0), rv=_F(461.6),
                 xlv=_F(2.5e6), xls=_F(2.85e6), xlf=_F(3.5e5), grav=_F(9.81)):
        self.alf = _F(xlf)
        self.als = _F(xls)
        self.alv = _F(xlv)
        self.cpd = _F(cp)
        self.g = _F(grav)
        self.rd = _F(rd)
        self.rv = _F(rv)
        self.rcpd = _F(1.0) / self.cpd
        self.c2es = C1ES * self.rd / self.rv
        self.c5les = C3LES * (TMELT - C4LES)
        self.c5ies = C3IES * (TMELT - C4IES)
        self.r5alvcp = self.c5les * self.alv * self.rcpd
        self.r5alscp = self.c5ies * self.als * self.rcpd
        self.ralvdcp = self.alv * self.rcpd
        self.ralsdcp = self.als * self.rcpd
        self.ralfdcp = self.alf * self.rcpd
        self.vtmpc1 = self.rv / self.rd - _F(1.0)
        self.zrg = _F(1.0) / self.g


# --- saturation functions (verbatim) ------------------------------------------
def _foealfa(c: _Const, tt):
    return min(1.0, ((max(RTICE, min(RTWAT, tt)) - RTICE) / (RTWAT - RTICE)) ** 2)


def _foelhm(c: _Const, tt):
    a = _foealfa(c, tt)
    return a * c.alv + (1.0 - a) * c.als


def _foeewm(c: _Const, tt):
    a = _foealfa(c, tt)
    return c.c2es * (a * np.exp(C3LES * (tt - TMELT) / (tt - C4LES))
                     + (1.0 - a) * np.exp(C3IES * (tt - TMELT) / (tt - C4IES)))


def _foedem(c: _Const, tt):
    a = _foealfa(c, tt)
    return (a * c.r5alvcp * (1.0 / (tt - C4LES) ** 2)
            + (1.0 - a) * c.r5alscp * (1.0 / (tt - C4IES) ** 2))


def _foeldcpm(c: _Const, tt):
    a = _foealfa(c, tt)
    return a * c.ralvdcp + (1.0 - a) * c.ralsdcp


# --- cuadjtqn (single column; kk is a Fortran level index) ---------------------
def _cuadjtqn(c: _Const, kk, psp, pt, pq, ldflag, kcall):
    """Saturation adjustment at level kk.  Mutates pt[kk], pq[kk] in place."""
    if kcall == 1:
        if ldflag:
            zqp = 1.0 / psp
            zl = 1.0 / (pt[kk] - C4LES)
            zi = 1.0 / (pt[kk] - C4IES)
            alfa = _foealfa(c, pt[kk])
            zqsat = c.c2es * (alfa * np.exp(C3LES * (pt[kk] - TMELT) * zl)
                              + (1.0 - alfa) * np.exp(C3IES * (pt[kk] - TMELT) * zi))
            zqsat = zqsat * zqp
            zqsat = min(0.5, zqsat)
            zcor = 1.0 - c.vtmpc1 * zqsat
            zf = (alfa * c.r5alvcp * zl ** 2
                  + (1.0 - alfa) * c.r5alscp * zi ** 2)
            zcond = (pq[kk] * zcor ** 2 - zqsat * zcor) / (zcor ** 2 + zqsat * zf)
            if zcond > 0.0:
                pt[kk] = pt[kk] + _foeldcpm(c, pt[kk]) * zcond
                pq[kk] = pq[kk] - zcond
                zl = 1.0 / (pt[kk] - C4LES)
                zi = 1.0 / (pt[kk] - C4IES)
                alfa = _foealfa(c, pt[kk])
                zqsat = c.c2es * (alfa * np.exp(C3LES * (pt[kk] - TMELT) * zl)
                                  + (1.0 - alfa) * np.exp(C3IES * (pt[kk] - TMELT) * zi))
                zqsat = zqsat * zqp
                zqsat = min(0.5, zqsat)
                zcor = 1.0 - c.vtmpc1 * zqsat
                zf = (alfa * c.r5alvcp * zl ** 2
                      + (1.0 - alfa) * c.r5alscp * zi ** 2)
                zcond1 = (pq[kk] * zcor ** 2 - zqsat * zcor) / (zcor ** 2 + zqsat * zf)
                if abs(zcond) < _f32(1.0e-20):
                    zcond1 = 0.0
                pt[kk] = pt[kk] + _foeldcpm(c, pt[kk]) * zcond1
                pq[kk] = pq[kk] - zcond1
    elif kcall == 2:
        if ldflag:
            zqp = 1.0 / psp
            zqsat = _foeewm(c, pt[kk]) * zqp
            zqsat = min(0.5, zqsat)
            zcor = 1.0 / (1.0 - c.vtmpc1 * zqsat)
            zqsat = zqsat * zcor
            zcond = (pq[kk] - zqsat) / (1.0 + zqsat * zcor * _foedem(c, pt[kk]))
            zcond = min(zcond, 0.0)
            pt[kk] = pt[kk] + _foeldcpm(c, pt[kk]) * zcond
            pq[kk] = pq[kk] - zcond
            zqsat = _foeewm(c, pt[kk]) * zqp
            zqsat = min(0.5, zqsat)
            zcor = 1.0 / (1.0 - c.vtmpc1 * zqsat)
            zqsat = zqsat * zcor
            zcond1 = (pq[kk] - zqsat) / (1.0 + zqsat * zcor * _foedem(c, pt[kk]))
            if abs(zcond) < _f32(1.0e-20):
                zcond1 = min(zcond1, 0.0)
            pt[kk] = pt[kk] + _foeldcpm(c, pt[kk]) * zcond1
            pq[kk] = pq[kk] - zcond1
    elif kcall == 0:
        # kcall=0 ignores ldflag in the Fortran (processes every column).
        zqp = 1.0 / psp
        zqsat = _foeewm(c, pt[kk]) * zqp
        zqsat = min(0.5, zqsat)
        zcor = 1.0 / (1.0 - c.vtmpc1 * zqsat)
        zqsat = zqsat * zcor
        zcond1 = (pq[kk] - zqsat) / (1.0 + zqsat * zcor * _foedem(c, pt[kk]))
        pt[kk] = pt[kk] + _foeldcpm(c, pt[kk]) * zcond1
        pq[kk] = pq[kk] - zcond1
        zqsat = _foeewm(c, pt[kk]) * zqp
        zqsat = min(0.5, zqsat)
        zcor = 1.0 / (1.0 - c.vtmpc1 * zqsat)
        zqsat = zqsat * zcor
        zcond1 = (pq[kk] - zqsat) / (1.0 + zqsat * zcor * _foedem(c, pt[kk]))
        pt[kk] = pt[kk] + _foeldcpm(c, pt[kk]) * zcond1
        pq[kk] = pq[kk] - zcond1


def _z1(n):
    """1-based fp64 zeros of Fortran length n (slot 0 dead)."""
    return np.zeros(n + 1, dtype=_F)


# --- cuinin --------------------------------------------------------------------
def _cuinin(c: _Const, klev, pten, pqen, pqsen, puen, pven, pverv, pgeo,
            paph, pgeoh):
    """Half-level interpolation + updraft/downdraft init (single column).

    Returns (ptenh, pqenh, pqsenh, klwmin, ptu, pqu, ptd, pqd, puu, pvu,
    pud, pvd, plu, klab).  The mass-flux/flux arrays the Fortran also zeroes
    here are allocated fresh by the caller.
    """
    klevm1 = klev - 1
    ptenh = _z1(klev)
    pqenh = _z1(klev)
    pqsenh = _z1(klev)
    for jk in range(2, klev + 1):
        ptenh[jk] = (max(c.cpd * pten[jk - 1] + pgeo[jk - 1],
                         c.cpd * pten[jk] + pgeo[jk]) - pgeoh[jk]) * c.rcpd
        pqenh[jk] = pqen[jk - 1]
        pqsenh[jk] = pqsen[jk - 1]
        zph = paph[jk]
        if jk >= klev - 1 or jk < 2:
            continue
        _cuadjtqn(c, jk, zph, ptenh, pqsenh, True, 0)
        pqenh[jk] = (min(pqen[jk - 1], pqsen[jk - 1])
                     + (pqsenh[jk] - pqsen[jk - 1]))
        pqenh[jk] = max(pqenh[jk], 0.0)

    ptenh[klev] = (c.cpd * pten[klev] + pgeo[klev] - pgeoh[klev]) * c.rcpd
    pqenh[klev] = pqen[klev]
    ptenh[1] = pten[1]
    pqenh[1] = pqen[1]
    klwmin = klev
    zwmax = 0.0

    for jk in range(klevm1, 1, -1):
        zzs = max(c.cpd * ptenh[jk] + pgeoh[jk],
                  c.cpd * ptenh[jk + 1] + pgeoh[jk + 1])
        ptenh[jk] = (zzs - pgeoh[jk]) * c.rcpd

    for jk in range(klev, 2, -1):
        if pverv[jk] < zwmax:
            zwmax = pverv[jk]
            klwmin = jk

    ptu = _z1(klev)
    ptd = _z1(klev)
    pqu = _z1(klev)
    pqd = _z1(klev)
    plu = _z1(klev)
    puu = _z1(klev)
    pud = _z1(klev)
    pvu = _z1(klev)
    pvd = _z1(klev)
    klab = np.zeros(klev + 1, dtype=np.int64)
    for jk in range(1, klev + 1):
        ik = jk - 1
        if jk == 1:
            ik = 1
        ptu[jk] = ptenh[jk]
        ptd[jk] = ptenh[jk]
        pqu[jk] = pqenh[jk]
        pqd[jk] = pqenh[jk]
        plu[jk] = 0.0
        puu[jk] = puen[ik]
        pud[jk] = puen[ik]
        pvu[jk] = pven[ik]
        pvd[jk] = pven[ik]
        klab[jk] = 0
    return ptenh, pqenh, pqsenh, klwmin, ptu, pqu, ptd, pqd, puu, pvu, pud, pvd, plu, klab


# --- cubasmcn (midlevel convection cloud-base; single column, level kk) --------
def _cubasmcn(c: _Const, klev, kk, pten, pqen, pqsen, pverv, pgeo, pgeoh,
              ldcum, ktype, klab, plrain, pmfu, pmfub, kcbot, ptu, pqu, plu,
              pmfus, pmfuq, pmful, pdmfup):
    """Mutates the updraft state in place; returns (ldcum, ktype, kcbot, pmfub)."""
    if (not ldcum) and klab[kk + 1] == 0:
        if (LMFMID and pqen[kk] > _f32(0.80) * pqsen[kk]
                and pgeo[kk] * c.zrg > 5.0e2
                and pgeo[kk] * c.zrg < 1.0e4):
            ptu[kk + 1] = (c.cpd * pten[kk] + pgeo[kk] - pgeoh[kk + 1]) * c.rcpd
            pqu[kk + 1] = pqen[kk]
            plu[kk + 1] = 0.0
            zzzmb = max(CMFCMIN, -pverv[kk] * c.zrg)
            zzzmb = min(zzzmb, CMFCMAX)
            pmfub = zzzmb
            pmfu[kk + 1] = pmfub
            pmfus[kk + 1] = pmfub * (c.cpd * ptu[kk + 1] + pgeoh[kk + 1])
            pmfuq[kk + 1] = pmfub * pqu[kk + 1]
            pmful[kk + 1] = 0.0
            pdmfup[kk + 1] = 0.0
            kcbot = kk
            klab[kk + 1] = 1
            plrain[kk + 1] = 0.0
            ktype = 3
    return ldcum, ktype, kcbot, pmfub


# --- cuentrn -------------------------------------------------------------------
def _cuentrn(c: _Const, kk, kcbot, ldcum, ldwork, pgeoh, pmfu):
    pdmfen = 0.0
    pdmfde = 0.0
    if ldwork:
        zentr = 0.0
        if ldcum:
            zdz = (pgeoh[kk] - pgeoh[kk + 1]) * c.zrg
            zmf = pmfu[kk + 1] * zdz
            if kk < kcbot:
                pdmfen = zentr * zmf
                pdmfde = _f32(0.75e-4) * zmf
    return pdmfen, pdmfde


# --- cutypen (first-guess updraft, cloud base + convection type) ---------------
def _cutypen(c: _Const, klev, pqen, ptenh, pqenh, pqsenh, pgeoh, paph,
             hfx, qfx, pgeo, pqsen, pap, pten, lndj,
             cutu, cuqu, culab, culu):
    """Single-column cutypen.  cutu/cuqu/culu/culab are the caller's updraft
    arrays (cuinin-initialized); they are read as parcel initial values and
    written back exactly as the Fortran intent(out)-but-aliased arrays behave.

    Returns (ldcum, cubot, cutop, ktype, wbase, kdpl).
    """
    klevm1 = klev - 1
    kcbot = klev
    kctop = klev
    kdpl = klev
    ktype = 0
    wbase = 0.0
    ldcum = False

    ptu = _z1(klev)
    pqu = _z1(klev)
    plu = _z1(klev)
    dh = _z1(klev)
    dhen = _z1(klev)
    kup = _z1(klev)
    vptu = _z1(klev)
    vten = _z1(klev)
    zbuo = _z1(klev)
    abuoy = _z1(klev)
    klab = np.zeros(klev + 1, dtype=np.int64)
    for jk in range(1, klev + 1):
        plu[jk] = culu[jk]
        ptu[jk] = cutu[jk]
        pqu[jk] = cuqu[jk]
        klab[jk] = culab[jk]

    zqold = 0.0
    lldcum = False
    loflag = True
    zcbase = 0
    eta = 0.0
    dz = 0.0
    coef = 0.0

    # ---- shallow test parcel from the surface ---------------------------------
    for jk in range(klevm1, 1, -1):
        if jk == klevm1:
            rho = pap[klev] / (c.rd * (pten[klev] * (1.0 + c.vtmpc1 * pqen[klev])))
            part1 = _F(np.float32(1.5) * np.float32(0.4)) * pgeo[klev] / (rho * pten[klev])
            part2 = -hfx * c.rcpd - c.vtmpc1 * pten[klev] * qfx
            root = _f32(0.001) - part1 * part2
            if part2 < 0.0:
                conw = _f32(1.2) * root ** T13
                deltt = max(1.5 * hfx / (rho * c.cpd * conw), 0.0)
                deltq = max(1.5 * qfx / (rho * conw), 0.0)
                kup[klev] = 0.5 * conw ** 2
                pqu[klev] = pqenh[klev] + deltq
                dhen[klev] = pgeoh[klev] + ptenh[klev] * c.cpd
                dh[klev] = dhen[klev] + deltt * c.cpd
                ptu[klev] = (dh[klev] - pgeoh[klev]) * c.rcpd
                vptu[klev] = ptu[klev] * (1.0 + c.vtmpc1 * pqu[klev])
                vten[klev] = ptenh[klev] * (1.0 + c.vtmpc1 * pqenh[klev])
                zbuo[klev] = (vptu[klev] - vten[klev]) / vten[klev]
                klab[klev] = 1
            else:
                loflag = False

        if not loflag:
            break

        eta = _f32(0.8) / (pgeo[jk] * c.zrg) + _f32(2.0e-4)
        dz = (pgeoh[jk] - pgeoh[jk + 1]) * c.zrg
        coef = 0.5 * eta * dz
        dhen[jk] = pgeoh[jk] + c.cpd * ptenh[jk]
        dh[jk] = (coef * (dhen[jk + 1] + dhen[jk])
                  + (1.0 - coef) * dh[jk + 1]) / (1.0 + coef)
        pqu[jk] = (coef * (pqenh[jk + 1] + pqenh[jk])
                   + (1.0 - coef) * pqu[jk + 1]) / (1.0 + coef)
        ptu[jk] = (dh[jk] - pgeoh[jk]) * c.rcpd
        zqold = pqu[jk]
        zph = paph[jk]

        _cuadjtqn(c, jk, zph, ptu, pqu, loflag, 1)

        zdq = max(zqold - pqu[jk], 0.0)
        plu[jk] = plu[jk + 1] + zdq
        zlglac = zdq * ((1.0 - _foealfa(c, ptu[jk]))
                        - (1.0 - _foealfa(c, ptu[jk + 1])))
        plu[jk] = min(plu[jk], _f32(5.0e-3))
        dh[jk] = pgeoh[jk] + c.cpd * (ptu[jk] + c.ralfdcp * zlglac)
        vptu[jk] = (ptu[jk] * (1.0 + c.vtmpc1 * pqu[jk] - plu[jk])
                    + c.ralfdcp * zlglac)
        vten[jk] = ptenh[jk] * (1.0 + c.vtmpc1 * pqenh[jk])
        zbuo[jk] = (vptu[jk] - vten[jk]) / vten[jk]
        abuoy[jk] = (zbuo[jk] + zbuo[jk + 1]) * 0.5 * c.g
        atop1 = 1.0 - 2.0 * coef
        atop2 = 2.0 * dz * abuoy[jk]
        abot = 1.0 + 2.0 * coef
        kup[jk] = (atop1 * kup[jk + 1] + atop2) / abot

        if plu[jk] > 0.0 and klab[jk + 1] == 1:
            ik = jk + 1
            zqsu = _foeewm(c, ptu[ik]) / paph[ik]
            zqsu = min(0.5, zqsu)
            zcor = 1.0 / (1.0 - c.vtmpc1 * zqsu)
            zqsu = zqsu * zcor
            zdq = min(0.0, pqu[ik] - zqsu)
            zalfaw = _foealfa(c, ptu[ik])
            zfacw = c.c5les / ((ptu[ik] - C4LES) ** 2)
            zfaci = c.c5ies / ((ptu[ik] - C4IES) ** 2)
            zfac = zalfaw * zfacw + (1.0 - zalfaw) * zfaci
            zesdp = _foeewm(c, ptu[ik]) / paph[ik]
            zcor = 1.0 / (1.0 - c.vtmpc1 * zesdp)
            zdqsdt = zfac * zcor * zqsu
            zdtdp = c.rd * ptu[ik] / (c.cpd * paph[ik])
            zdp = zdq / (zdqsdt * zdtdp)
            zcbase = int(paph[ik] + zdp)  # Fortran INTEGER zcbase: truncation
            zpdifftop = zcbase - paph[jk]
            zpdiffbot = paph[jk + 1] - zcbase
            if zpdifftop > zpdiffbot and kup[jk + 1] > 0.0:
                ikb = min(klev - 1, jk + 1)
                klab[ikb] = 2
                klab[jk] = 2
                kcbot = ikb
                plu[jk + 1] = _f32(1.0e-8)
            elif zpdifftop <= zpdiffbot and kup[jk] > 0.0:
                klab[jk] = 2
                kcbot = jk

        if kup[jk] < 0.0:
            loflag = False
            if plu[jk + 1] > 0.0:
                kctop = jk
                lldcum = True
            else:
                lldcum = False
        else:
            if plu[jk] > 0.0:
                klab[jk] = 2
            else:
                klab[jk] = 1

    ikb = kcbot
    ikt = kctop
    if paph[ikb] - paph[ikt] > ZDNOPRC:
        lldcum = False
    if lldcum:
        ktype = 2
        ldcum = True
        wbase = np.sqrt(max(2.0 * kup[ikb], 0.0))
        cubot = ikb
        cutop = ikt
        kdpl = klev
    else:
        cutop = -1
        cubot = -1
        kdpl = klev - 1
        ldcum = False
        wbase = 0.0

    for jk in range(klev, 0, -1):
        ikt = kctop
        if jk >= ikt:
            culab[jk] = klab[jk]
            cutu[jk] = ptu[jk]
            cuqu[jk] = pqu[jk]
            culu[jk] = plu[jk]

    # ---- deep test parcels from elevated departure levels ---------------------
    deltt = _f32(0.2)
    deltq = _f32(1.0e-4)
    deepflag = False

    itoppacel = 0
    for jk in range(klev, 0, -1):
        if (paph[klev + 1] - paph[jk]) < 350.0e2:
            itoppacel = jk

    for levels in range(klevm1 - 1, klev // 2, -1):
        for jk in range(1, klev + 1):
            plu[jk] = 0.0
            ptu[jk] = 0.0
            pqu[jk] = 0.0
            dh[jk] = 0.0
            dhen[jk] = 0.0
            kup[jk] = 0.0
            vptu[jk] = 0.0
            vten[jk] = 0.0
            abuoy[jk] = 0.0
            zbuo[jk] = 0.0
            klab[jk] = 0

        kcbot = levels
        kctop = levels
        zqold = 0.0
        lldcum = False
        resetflag = False
        loflag = (not deepflag) and (levels >= itoppacel)

        for jk in range(levels, 1, -1):
            if not loflag:
                break

            if jk == levels:
                if (paph[klev + 1] - paph[jk]) < 60.0e2:
                    tmix = 0.0
                    qmix = 0.0
                    zmix = 0.0
                    pmix = 0.0
                    for nk in range(jk + 2, jk - 1, -1):
                        if pmix < 50.0e2:
                            dp = paph[nk] - paph[nk - 1]
                            tmix = tmix + dp * ptenh[nk]
                            qmix = qmix + dp * pqenh[nk]
                            zmix = zmix + dp * pgeoh[nk]
                            pmix = pmix + dp
                    tmix = tmix / pmix
                    qmix = qmix / pmix
                    zmix = zmix / pmix
                else:
                    tmix = ptenh[jk + 1]
                    qmix = pqenh[jk + 1]
                    zmix = pgeoh[jk + 1]

                pqu[jk + 1] = qmix + deltq
                dhen[jk + 1] = zmix + tmix * c.cpd
                dh[jk + 1] = dhen[jk + 1] + deltt * c.cpd
                ptu[jk + 1] = (dh[jk + 1] - pgeoh[jk + 1]) * c.rcpd
                kup[jk + 1] = 0.5
                klab[jk + 1] = 1
                vptu[jk + 1] = ptu[jk + 1] * (1.0 + c.vtmpc1 * pqu[jk + 1])
                vten[jk + 1] = ptenh[jk + 1] * (1.0 + c.vtmpc1 * pqenh[jk + 1])
                zbuo[jk + 1] = (vptu[jk + 1] - vten[jk + 1]) / vten[jk + 1]

            fscale = min(1.0, (pqsen[jk] / pqsen[levels]) ** 3)
            eta = _f32(1.75e-3) * fscale
            dz = (pgeoh[jk] - pgeoh[jk + 1]) * c.zrg
            coef = 0.5 * eta * dz
            dhen[jk] = pgeoh[jk] + c.cpd * ptenh[jk]
            dh[jk] = (coef * (dhen[jk + 1] + dhen[jk])
                      + (1.0 - coef) * dh[jk + 1]) / (1.0 + coef)
            pqu[jk] = (coef * (pqenh[jk + 1] + pqenh[jk])
                       + (1.0 - coef) * pqu[jk + 1]) / (1.0 + coef)
            ptu[jk] = (dh[jk] - pgeoh[jk]) * c.rcpd
            zqold = pqu[jk]
            zph = paph[jk]

            _cuadjtqn(c, jk, zph, ptu, pqu, loflag, 1)

            zdq = max(zqold - pqu[jk], 0.0)
            plu[jk] = plu[jk + 1] + zdq
            zlglac = zdq * ((1.0 - _foealfa(c, ptu[jk]))
                            - (1.0 - _foealfa(c, ptu[jk + 1])))
            plu[jk] = 0.5 * plu[jk]
            dh[jk] = pgeoh[jk] + c.cpd * (ptu[jk] + c.ralfdcp * zlglac)
            vptu[jk] = (ptu[jk] * (1.0 + c.vtmpc1 * pqu[jk] - plu[jk])
                        + c.ralfdcp * zlglac)
            vten[jk] = ptenh[jk] * (1.0 + c.vtmpc1 * pqenh[jk])
            zbuo[jk] = (vptu[jk] - vten[jk]) / vten[jk]
            abuoy[jk] = (zbuo[jk] + zbuo[jk + 1]) * 0.5 * c.g
            atop1 = 1.0 - 2.0 * coef
            atop2 = 2.0 * dz * abuoy[jk]
            abot = 1.0 + 2.0 * coef
            kup[jk] = (atop1 * kup[jk + 1] + atop2) / abot

            if plu[jk] > 0.0 and klab[jk + 1] == 1:
                ik = jk + 1
                zqsu = _foeewm(c, ptu[ik]) / paph[ik]
                zqsu = min(0.5, zqsu)
                zcor = 1.0 / (1.0 - c.vtmpc1 * zqsu)
                zqsu = zqsu * zcor
                zdq = min(0.0, pqu[ik] - zqsu)
                zalfaw = _foealfa(c, ptu[ik])
                zfacw = c.c5les / ((ptu[ik] - C4LES) ** 2)
                zfaci = c.c5ies / ((ptu[ik] - C4IES) ** 2)
                zfac = zalfaw * zfacw + (1.0 - zalfaw) * zfaci
                zesdp = _foeewm(c, ptu[ik]) / paph[ik]
                zcor = 1.0 / (1.0 - c.vtmpc1 * zesdp)
                zdqsdt = zfac * zcor * zqsu
                zdtdp = c.rd * ptu[ik] / (c.cpd * paph[ik])
                zdp = zdq / (zdqsdt * zdtdp)
                zcbase = int(paph[ik] + zdp)  # Fortran INTEGER zcbase
                zpdifftop = zcbase - paph[jk]
                zpdiffbot = paph[jk + 1] - zcbase
                if zpdifftop > zpdiffbot and kup[jk + 1] > 0.0:
                    ikb = min(klev - 1, jk + 1)
                    klab[ikb] = 2
                    klab[jk] = 2
                    kcbot = ikb
                    plu[jk + 1] = _f32(1.0e-8)
                elif zpdifftop <= zpdiffbot and kup[jk] > 0.0:
                    klab[jk] = 2
                    kcbot = jk

            if kup[jk] < 0.0:
                loflag = False
                if plu[jk + 1] > 0.0:
                    kctop = jk
                    lldcum = True
                else:
                    lldcum = False
            else:
                if plu[jk] > 0.0:
                    klab[jk] = 2
                else:
                    klab[jk] = 1

        needreset = False
        ikb = kcbot
        ikt = kctop
        if paph[ikb] - paph[ikt] < ZDNOPRC:
            lldcum = False
        if lldcum:
            ktype = 1
            ldcum = True
            deepflag = True
            wbase = np.sqrt(max(2.0 * kup[ikb], 0.0))
            cubot = ikb
            cutop = ikt
            kdpl = levels + 1
            needreset = True
            resetflag = True

        if needreset:
            for jk in range(klev, 0, -1):
                if resetflag:
                    ikt = kctop
                    ikb = kdpl
                    if jk <= ikb and jk >= ikt:
                        culab[jk] = klab[jk]
                        cutu[jk] = ptu[jk]
                        cuqu[jk] = pqu[jk]
                        culu[jk] = plu[jk]
                    else:
                        culab[jk] = 1
                        cutu[jk] = ptenh[jk]
                        cuqu[jk] = pqenh[jk]
                        culu[jk] = 0.0
                    if jk < ikt:
                        culab[jk] = 0

    return ldcum, cubot, cutop, ktype, wbase, kdpl


# --- cuascn (cloud ascent for entraining plume) ---------------------------------
def _cuascn(c: _Const, klev, ztmst, ptenh, pqenh, puen, pven, pten, pqen,
            pqsen, pgeo, pgeoh, pap, paph, pqte, pverv, klwmin, ldcum,
            ktype, klab, ptu, pqu, plu, puu, pvu, pmfu, pmfub,
            pmfus, pmfuq, pmful, plude, pdmfup, kcbot, kctop, kctop0,
            lndj, wbase, kdpl):
    """Single-column cuascn.  Mutates the passed arrays in place (including
    ptenh/pqenh -- the Fortran modifies the half-level environment when the
    parcel goes negatively buoyant).  Returns
    (ldcum, ktype, kcbot, kctop, kctop0, wup, plglac, pmfude_rate).
    """
    klevm1 = klev - 1
    zcons2 = 3.0 / (c.g * ztmst)
    zfacbuo = _F(np.float32(0.5) / (np.float32(1.0) + np.float32(0.5)))
    zprcdgw = CPRCON * c.zrg
    z_cldmax = _f32(5.0e-3)
    z_cwifrac = 0.5
    z_cprc2 = 0.5
    z_cwdrag = _F(np.float32(3.0) / np.float32(8.0) * np.float32(0.506) / np.float32(0.2))

    plglac = _z1(klev)
    pmfude_rate = _z1(klev)
    zlrain = _z1(klev)
    zbuo = _z1(klev)
    kup = _z1(klev)
    pdmfen_arr = _z1(klev)

    # 2. set default values
    llo3 = False
    zluold = 0.0
    wup = 0.0
    zdpmean = 0.0
    zoentr = 0.0
    if not ldcum:
        ktype = 0
        kcbot = -1
        pmfub = 0.0
        pqu[klev] = 0.0

    for jk in range(1, klev + 1):
        if jk != kcbot:
            plu[jk] = 0.0
        pmfu[jk] = 0.0
        pmfus[jk] = 0.0
        pmfuq[jk] = 0.0
        pmful[jk] = 0.0
        plude[jk] = 0.0
        plglac[jk] = 0.0
        pdmfup[jk] = 0.0
        zlrain[jk] = 0.0
        zbuo[jk] = 0.0
        kup[jk] = 0.0
        pdmfen_arr[jk] = 0.0
        pmfude_rate[jk] = 0.0
        if (not ldcum) or ktype == 3:
            klab[jk] = 0
        if (not ldcum) and paph[jk] < 4.0e4:
            kctop0 = jk

    if ktype == 3:
        ldcum = False

    # 3. initialize values at cloud base level
    kctop = kcbot
    if ldcum:
        ikb = kcbot
        kup[ikb] = 0.5 * wbase ** 2
        pmfu[ikb] = pmfub
        pmfus[ikb] = pmfub * (c.cpd * ptu[ikb] + pgeoh[ikb])
        pmfuq[ikb] = pmfub * pqu[ikb]
        pmful[ikb] = pmfub * plu[ikb]

    # 4. do ascent
    zdmfen = 0.0
    zdmfde = 0.0
    for jk in range(klevm1, 2, -1):
        # cubasmcn: midlevel convection cloud base
        ldcum, ktype, kcbot, pmfub = _cubasmcn(
            c, klev, jk, pten, pqen, pqsen, pverv, pgeo, pgeoh, ldcum,
            ktype, klab, zlrain, pmfu, pmfub, kcbot, ptu, pqu, plu,
            pmfus, pmfuq, pmful, pdmfup)

        loflag = False
        zprecip = 0.0
        llo1 = False
        is_cnt = int(klab[jk + 1])
        if klab[jk + 1] == 0:
            klab[jk] = 0
        if (ldcum and klab[jk + 1] == 2) or (ktype == 3 and klab[jk + 1] == 1):
            loflag = True
        zph = paph[jk]
        if ktype == 3 and jk == kcbot:
            zmfmax = (paph[jk] - paph[jk - 1]) * zcons2
            if pmfub > zmfmax:
                zfac = zmfmax / pmfub
                pmfu[jk + 1] = pmfu[jk + 1] * zfac
                pmfus[jk + 1] = pmfus[jk + 1] * zfac
                pmfuq[jk + 1] = pmfuq[jk + 1] * zfac
                pmfub = zmfmax
            pmfub = min(pmfub, zmfmax)

        if is_cnt > 0:
            llo3 = True

        # cuentrn: entrainment/detrainment rates
        zdmfen, zdmfde = _cuentrn(c, jk, kcbot, ldcum, llo3, pgeoh, pmfu)

        if llo3:
            zqold = 0.0
            if loflag:
                zdmfde = min(zdmfde, 0.75 * pmfu[jk + 1])
                if jk == kcbot:
                    zoentr = (-_f32(1.75e-3) * (min(1.0, pqen[jk] / pqsen[jk]) - 1.0)
                              * (pgeoh[jk] - pgeoh[jk + 1]) * c.zrg)
                    zoentr = min(_f32(0.4), zoentr) * pmfu[jk + 1]
                if jk < kcbot:
                    zmfmax = (paph[jk] - paph[jk - 1]) * zcons2
                    zxs = max(pmfu[jk + 1] - zmfmax, 0.0)
                    wup = wup + kup[jk + 1] * (pap[jk + 1] - pap[jk])
                    zdpmean = zdpmean + pap[jk + 1] - pap[jk]
                    zdmfen = zoentr
                    if ktype >= 2:
                        zdmfen = 2.0 * zdmfen
                        zdmfde = zdmfen
                    zdmfde = zdmfde * (_f32(1.6) - min(1.0, pqen[jk] / pqsen[jk]))
                    zmftest = pmfu[jk + 1] + zdmfen - zdmfde
                    zchange = max(zmftest - zmfmax, 0.0)
                    zxe = max(zchange - zxs, 0.0)
                    zdmfen = zdmfen - zxe
                    zchange = zchange - zxe
                    zdmfde = zdmfde + zchange
                pdmfen_arr[jk] = zdmfen - zdmfde
                pmfu[jk] = pmfu[jk + 1] + zdmfen - zdmfde
                zqeen = pqenh[jk + 1] * zdmfen
                zseen = (c.cpd * ptenh[jk + 1] + pgeoh[jk + 1]) * zdmfen
                zscde = (c.cpd * ptu[jk + 1] + pgeoh[jk + 1]) * zdmfde
                zqude = pqu[jk + 1] * zdmfde
                plude[jk] = plu[jk + 1] * zdmfde
                zmfusk = pmfus[jk + 1] + zseen - zscde
                zmfuqk = pmfuq[jk + 1] + zqeen - zqude
                zmfulk = pmful[jk + 1] - plude[jk]
                plu[jk] = zmfulk * (1.0 / max(CMFCMIN, pmfu[jk]))
                pqu[jk] = zmfuqk * (1.0 / max(CMFCMIN, pmfu[jk]))
                ptu[jk] = (zmfusk * (1.0 / max(CMFCMIN, pmfu[jk]))
                           - pgeoh[jk]) * c.rcpd
                ptu[jk] = max(100.0, ptu[jk])
                ptu[jk] = min(400.0, ptu[jk])
                zqold = pqu[jk]
                zlrain[jk] = (zlrain[jk + 1] * (pmfu[jk + 1] - zdmfde)
                              * (1.0 / max(CMFCMIN, pmfu[jk])))
                zluold = plu[jk]

            # reset to environmental values if below departure level
            if jk > kdpl:
                ptu[jk] = ptenh[jk]
                pqu[jk] = pqenh[jk]
                plu[jk] = 0.0
                zluold = plu[jk]

            if loflag:
                _cuadjtqn(c, jk, zph, ptu, pqu, loflag, 1)

            if loflag:
                if pqu[jk] != zqold:
                    plglac[jk] = plu[jk] * ((1.0 - _foealfa(c, ptu[jk]))
                                            - (1.0 - _foealfa(c, ptu[jk + 1])))
                    ptu[jk] = ptu[jk] + c.ralfdcp * plglac[jk]

                if pqu[jk] != zqold:
                    klab[jk] = 2
                    plu[jk] = plu[jk] + zqold - pqu[jk]
                    zbc = ptu[jk] * (1.0 + c.vtmpc1 * pqu[jk] - plu[jk + 1]
                                     - zlrain[jk + 1])
                    zbe = ptenh[jk] * (1.0 + c.vtmpc1 * pqenh[jk])
                    zbuo[jk] = zbc - zbe
                    if ktype == 3 and klab[jk + 1] == 1:
                        if zbuo[jk] > -0.5:
                            ldcum = True
                            kctop = jk
                            kup[jk] = 0.5
                        else:
                            klab[jk] = 0
                            pmfu[jk] = 0.0
                            plude[jk] = 0.0
                            plu[jk] = 0.0
                    if klab[jk + 1] == 2:
                        if zbuo[jk] < 0.0:
                            ptenh[jk] = 0.5 * (pten[jk] + pten[jk - 1])
                            pqenh[jk] = 0.5 * (pqen[jk] + pqen[jk - 1])
                            zbuo[jk] = zbc - ptenh[jk] * (1.0 + c.vtmpc1 * pqenh[jk])
                        zbuoc = ((zbuo[jk]
                                  / (ptenh[jk] * (1.0 + c.vtmpc1 * pqenh[jk]))
                                  + zbuo[jk + 1]
                                  / (ptenh[jk + 1] * (1.0 + c.vtmpc1 * pqenh[jk + 1])))
                                 * 0.5)
                        zdkbuo = (pgeoh[jk] - pgeoh[jk + 1]) * zfacbuo * zbuoc
                        if zdmfen > 0.0:
                            zdken = min(1.0, (1.0 + z_cwdrag) * zdmfen
                                        / max(CMFCMIN, pmfu[jk + 1]))
                        else:
                            zdken = min(1.0, (1.0 + z_cwdrag) * zdmfde
                                        / max(CMFCMIN, pmfu[jk + 1]))
                        kup[jk] = (kup[jk + 1] * (1.0 - zdken) + zdkbuo) / (1.0 + zdken)
                        if zbuo[jk] < 0.0:
                            zkedke = kup[jk] / max(_f32(1.0e-10), kup[jk + 1])
                            zkedke = max(0.0, min(1.0, zkedke))
                            zmfun = np.sqrt(zkedke) * pmfu[jk + 1]
                            zdmfde = max(zdmfde, pmfu[jk + 1] - zmfun)
                            plude[jk] = plu[jk + 1] * zdmfde
                            pmfu[jk] = pmfu[jk + 1] + zdmfen - zdmfde
                        if zbuo[jk] > -_f32(0.2):
                            ikb = kcbot
                            zoentr = (_f32(1.75e-3)
                                      * (_f32(0.3) - (min(1.0, pqen[jk - 1] / pqsen[jk - 1]) - 1.0))
                                      * (pgeoh[jk - 1] - pgeoh[jk]) * c.zrg
                                      * min(1.0, pqsen[jk] / pqsen[ikb]) ** 3)
                            zoentr = min(_f32(0.4), zoentr) * pmfu[jk]
                        else:
                            zoentr = 0.0
                        if jk > kdpl:
                            pmfu[jk] = pmfu[jk + 1]
                            kup[jk] = 0.5
                        if kup[jk] > 0.0 and pmfu[jk] > 0.0:
                            kctop = jk
                            llo1 = True
                        else:
                            klab[jk] = 0
                            pmfu[jk] = 0.0
                            kup[jk] = 0.0
                            zdmfde = pmfu[jk + 1]
                            plude[jk] = plu[jk + 1] * zdmfde
                        if pmfu[jk + 1] > 0.0:
                            pmfude_rate[jk] = zdmfde
                elif ktype == 2 and pqu[jk] == zqold:
                    klab[jk] = 0
                    pmfu[jk] = 0.0
                    kup[jk] = 0.0
                    zdmfde = pmfu[jk + 1]
                    plude[jk] = plu[jk + 1] * zdmfde
                    pmfude_rate[jk] = zdmfde

            if llo1:
                # precipitation conversion (Sundqvist) above threshold lwc
                if lndj == 1:
                    zdshrd = _f32(5.0e-4)
                else:
                    zdshrd = _f32(3.0e-4)
                ikb = kcbot
                if plu[jk] > zdshrd:
                    zwu = min(15.0, np.sqrt(2.0 * max(_f32(0.1), kup[jk + 1])))
                    zprcon = zprcdgw / (0.75 * zwu)
                    zdt = min(RTBER - RTICE, max(RTBER - ptu[jk], 0.0))
                    zcbf = 1.0 + z_cprc2 * np.sqrt(zdt)
                    zzco = zprcon * zcbf
                    zlcrit = zdshrd / zcbf
                    zdfi = pgeoh[jk] - pgeoh[jk + 1]
                    zc = plu[jk] - zluold
                    zarg = (plu[jk] / zlcrit) ** 2
                    if zarg < 25.0:
                        zd = zzco * (1.0 - np.exp(-zarg)) * zdfi
                    else:
                        zd = zzco * zdfi
                    zint = np.exp(-zd)
                    zlnew = zluold * zint + zc / zd * (1.0 - zint)
                    zlnew = max(0.0, min(plu[jk], zlnew))
                    zlnew = min(z_cldmax, zlnew)
                    zprecip = max(0.0, zluold + zc - zlnew)
                    pdmfup[jk] = zprecip * pmfu[jk]
                    zlrain[jk] = zlrain[jk] + zprecip
                    plu[jk] = zlnew

            if llo1:
                if zlrain[jk] > 0.0:
                    zvw = _f32(21.18) * zlrain[jk] ** _f32(0.2)
                    zvi = z_cwifrac * zvw
                    zalfaw = _foealfa(c, ptu[jk])
                    zvv = zalfaw * zvw + (1.0 - zalfaw) * zvi
                    zrold = zlrain[jk] - zprecip
                    zc = zprecip
                    zwu = min(15.0, np.sqrt(2.0 * max(_f32(0.1), kup[jk])))
                    zd = zvv / zwu
                    zint = np.exp(-zd)
                    zrnew = zrold * zint + zc / zd * (1.0 - zint)
                    zrnew = max(0.0, min(zlrain[jk], zrnew))
                    zlrain[jk] = zrnew

            if loflag:
                pmful[jk] = plu[jk] * pmfu[jk]
                pmfus[jk] = (c.cpd * ptu[jk] + pgeoh[jk]) * pmfu[jk]
                pmfuq[jk] = pqu[jk] * pmfu[jk]

    # 5. final calculations
    if kctop == -1:
        ldcum = False
    kcbot = max(kcbot, kctop)
    if ldcum:
        wup = max(_f32(1.0e-2), wup / max(1.0, zdpmean))
        wup = np.sqrt(2.0 * wup)

    return ldcum, ktype, kcbot, kctop, kctop0, pmfub, wup, plglac, pmfude_rate


# --- cudlfsn (level of free sinking) --------------------------------------------
def _cudlfsn(c: _Const, klev, kcbot, kctop, lndj, ldcum, ptenh, pqenh,
             puen, pven, pten, pqsen, pgeo, pgeoh, paph, ptu, pqu, plu,
             puu, pvu, pmfub, prfl, ptd, pqd, pud, pvd):
    """Single-column cudlfsn.  ptd/pqd/pud/pvd are the cuinin-initialized
    downdraft arrays (Fortran intent(out)-but-aliased storage: only the LFS
    level is overwritten here, other levels keep their prior values).
    Returns (pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl)."""
    pmfd = _z1(klev)
    pmfds = _z1(klev)
    pmfdq = _z1(klev)
    pdmfdp = _z1(klev)
    ztenwb = _z1(klev)
    zqenwb = _z1(klev)

    lddraf = False
    kdtop = klev + 1
    ikhsmin = klev + 1
    zhsmin = 1.0e8

    for jk in range(3, klev - 1):
        zhsk = (c.cpd * pten[jk] + pgeo[jk]
                + _foelhm(c, pten[jk]) * pqsen[jk])
        if zhsk < zhsmin:
            zhsmin = zhsk
            ikhsmin = jk

    ike = klev - 3
    for jk in range(3, ike + 1):
        ztenwb[jk] = ptenh[jk]
        zqenwb[jk] = pqenh[jk]
        zph = paph[jk]
        llo2 = (ldcum and prfl > 0.0 and (not lddraf)
                and (jk < kcbot and jk > kctop) and jk >= ikhsmin)
        if not llo2:
            continue

        _cuadjtqn(c, jk, zph, ztenwb, zqenwb, llo2, 2)

        zttest = 0.5 * (ptu[jk] + ztenwb[jk])
        zqtest = 0.5 * (pqu[jk] + zqenwb[jk])
        zbuo = (zttest * (1.0 + c.vtmpc1 * zqtest)
                - ptenh[jk] * (1.0 + c.vtmpc1 * pqenh[jk]))
        zcond = pqenh[jk] - zqenwb[jk]
        zmftop = -CMFDEPS * pmfub
        if zbuo < 0.0 and prfl > 10.0 * zmftop * zcond:
            kdtop = jk
            lddraf = True
            ptd[jk] = zttest
            pqd[jk] = zqtest
            pmfd[jk] = zmftop
            pmfds[jk] = pmfd[jk] * (c.cpd * ptd[jk] + pgeoh[jk])
            pmfdq[jk] = pmfd[jk] * pqd[jk]
            pdmfdp[jk - 1] = -0.5 * pmfd[jk] * zcond
            prfl = prfl + pdmfdp[jk - 1]

    return pmfd, pmfds, pmfdq, pdmfdp, kdtop, lddraf, prfl


# --- cuddrafn (downdraft descent) ------------------------------------------------
def _cuddrafn(c: _Const, klev, lddraf, ptenh, pqenh, puen, pven, pgeo,
              pgeoh, paph, prfl, ptd, pqd, pud, pvd, pmfu, pmfd, pmfds,
              pmfdq, pdmfdp):
    """Single-column cuddrafn.  Mutates the downdraft arrays in place.
    Returns (prfl, pmfdde_rate)."""
    pmfdde_rate = _z1(klev)
    zoentr = 0.0
    zbuoy = 0.0
    zdmfen = 0.0
    zdmfde = 0.0

    itopde = 0
    for jk in range(klev, 0, -1):
        pmfdde_rate[jk] = 0.0
        if (paph[klev + 1] - paph[jk]) < 60.0e2:
            itopde = jk

    for jk in range(3, klev + 1):
        zph = paph[jk]
        llo2 = lddraf and pmfd[jk - 1] < 0.0
        if not llo2:
            continue

        zentr = (ENTRDD * pmfd[jk - 1]
                 * (pgeoh[jk - 1] - pgeoh[jk]) * c.zrg)
        zdmfen = zentr
        zdmfde = zentr

        if jk > itopde:
            zdmfen = 0.0
            zdmfde = (pmfd[itopde] * (paph[jk] - paph[jk - 1])
                      / (paph[klev + 1] - paph[itopde]))

        if jk <= itopde:
            zdz = -(pgeoh[jk - 1] - pgeoh[jk]) * c.zrg
            zzentr = zoentr * zdz * pmfd[jk - 1]
            zdmfen = zdmfen + zzentr
            zdmfen = max(zdmfen, _f32(0.3) * pmfd[jk - 1])
            zdmfen = max(zdmfen, -0.75 * pmfu[jk] - (pmfd[jk - 1] - zdmfde))
            zdmfen = min(zdmfen, 0.0)

        pmfd[jk] = pmfd[jk - 1] + zdmfen - zdmfde
        zseen = (c.cpd * ptenh[jk - 1] + pgeoh[jk - 1]) * zdmfen
        zqeen = pqenh[jk - 1] * zdmfen
        zsdde = (c.cpd * ptd[jk - 1] + pgeoh[jk - 1]) * zdmfde
        zqdde = pqd[jk - 1] * zdmfde
        zmfdsk = pmfds[jk - 1] + zseen - zsdde
        zmfdqk = pmfdq[jk - 1] + zqeen - zqdde
        pqd[jk] = zmfdqk * (1.0 / min(-CMFCMIN, pmfd[jk]))
        ptd[jk] = (zmfdsk * (1.0 / min(-CMFCMIN, pmfd[jk])) - pgeoh[jk]) * c.rcpd
        ptd[jk] = min(400.0, ptd[jk])
        ptd[jk] = max(100.0, ptd[jk])
        zcond = pqd[jk]

        _cuadjtqn(c, jk, zph, ptd, pqd, llo2, 2)

        zcond = zcond - pqd[jk]
        zbuo = (ptd[jk] * (1.0 + c.vtmpc1 * pqd[jk])
                - ptenh[jk] * (1.0 + c.vtmpc1 * pqenh[jk]))
        if prfl > 0.0 and pmfu[jk] > 0.0:
            zrain = prfl / pmfu[jk]
            zbuo = zbuo - ptd[jk] * zrain
        if zbuo >= 0.0 or prfl <= (pmfd[jk] * zcond):
            pmfd[jk] = 0.0
            zbuo = 0.0
        pmfds[jk] = (c.cpd * ptd[jk] + pgeoh[jk]) * pmfd[jk]
        pmfdq[jk] = pqd[jk] * pmfd[jk]
        zdmfdp = -pmfd[jk] * zcond
        pdmfdp[jk - 1] = zdmfdp
        prfl = prfl + zdmfdp

        zbuoyz = zbuo / ptenh[jk]
        zbuoyz = min(zbuoyz, 0.0)
        zdz = -(pgeo[jk - 1] - pgeo[jk])
        zbuoy = zbuoy + zbuoyz * zdz
        zoentr = c.g * zbuoyz * 0.5 / (1.0 + zbuoy)
        pmfdde_rate[jk] = -zdmfde

    return prfl, pmfdde_rate


# --- cuflxn (final convective fluxes) --------------------------------------------
def _cuflxn(c: _Const, klev, ztmst, pten, pqen, pqsen, ptenh, pqenh,
            paph, pap, pgeoh, lndj, ldcum, kcbot, kctop, kdtop,
            ktype, lddraf, pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq,
            pmful, plude, pdmfup, pdmfdp, plglac, pmfdde_rate):
    """Single-column cuflxn.  Mutates flux arrays (incl. pqsen, plglac,
    pmfdde_rate) in place.  Returns
    (ktopm2, ktype, lddraf, prain, pdpmel, pmflxr, pmflxs)."""
    ztaumel = 18000.0
    zcons1a = c.cpd / (c.alf * c.g * ztaumel)
    zcons2 = 3.0 / (c.g * ztmst)
    zcucov = _f32(0.05)
    zcpecons = _f32(5.44e-4) / c.g

    pdpmel = _z1(klev)
    pmflxr = _z1(klev + 1)
    pmflxs = _z1(klev + 1)

    prain = 0.0
    if (not ldcum) or kdtop < kctop:
        lddraf = False
    if not ldcum:
        ktype = 0
    idbas = klev
    if lndj == 1:
        rhevap = _f32(0.7)
    else:
        rhevap = _f32(0.9)

    ktopm2 = 2
    for jk in range(ktopm2, klev + 1):
        ikb = min(jk + 1, klev)
        pmflxr[jk] = 0.0
        pmflxs[jk] = 0.0
        pdpmel[jk] = 0.0
        if ldcum and jk >= kctop:
            pmfus[jk] = pmfus[jk] - pmfu[jk] * (c.cpd * ptenh[jk] + pgeoh[jk])
            pmfuq[jk] = pmfuq[jk] - pmfu[jk] * pqenh[jk]
            plglac[jk] = pmfu[jk] * plglac[jk]
            llddraf = lddraf and jk >= kdtop
            if llddraf and jk >= kdtop:
                pmfds[jk] = pmfds[jk] - pmfd[jk] * (c.cpd * ptenh[jk] + pgeoh[jk])
                pmfdq[jk] = pmfdq[jk] - pmfd[jk] * pqenh[jk]
            else:
                pmfd[jk] = 0.0
                pmfds[jk] = 0.0
                pmfdq[jk] = 0.0
                pdmfdp[jk - 1] = 0.0
            if llddraf and pmfd[jk] < 0.0 and abs(pmfd[ikb]) < _f32(1.0e-20):
                idbas = jk
        else:
            pmfu[jk] = 0.0
            pmfd[jk] = 0.0
            pmfus[jk] = 0.0
            pmfds[jk] = 0.0
            pmfuq[jk] = 0.0
            pmfdq[jk] = 0.0
            pmful[jk] = 0.0
            plglac[jk] = 0.0
            pdmfup[jk - 1] = 0.0
            pdmfdp[jk - 1] = 0.0
            plude[jk - 1] = 0.0

    pmflxr[klev + 1] = 0.0
    pmflxs[klev + 1] = 0.0

    if ldcum:
        ikb = kcbot
        ik = ikb + 1
        zzp = ((paph[klev + 1] - paph[ik]) / (paph[klev + 1] - paph[ikb]))
        if ktype == 3:
            zzp = zzp ** 2
        pmfu[ik] = pmfu[ikb] * zzp
        pmfus[ik] = (pmfus[ikb] - _foelhm(c, ptenh[ikb]) * pmful[ikb]) * zzp
        pmfuq[ik] = (pmfuq[ikb] + pmful[ikb]) * zzp
        pmful[ik] = 0.0

    for jk in range(ktopm2, klev + 1):
        if ldcum and jk > kcbot + 1:
            ikb = kcbot + 1
            zzp = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ikb]))
            if ktype == 3:
                zzp = zzp ** 2
            pmfu[jk] = pmfu[ikb] * zzp
            pmfus[jk] = pmfus[ikb] * zzp
            pmfuq[jk] = pmfuq[ikb] * zzp
            pmful[jk] = 0.0
        ik = idbas
        llddraf = lddraf and jk > ik and ik < klev
        if llddraf and ik == kcbot + 1:
            zzp = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ik]))
            if ktype == 3:
                zzp = zzp * zzp
            pmfd[jk] = pmfd[ik] * zzp
            pmfds[jk] = pmfds[ik] * zzp
            pmfdq[jk] = pmfdq[ik] * zzp
            pmfdde_rate[jk] = -(pmfd[jk - 1] - pmfd[jk])
        elif llddraf and ik != kcbot + 1 and jk == ik + 1:
            pmfdde_rate[jk] = -(pmfd[jk - 1] - pmfd[jk])

    # 2. rain/snow fall rates, melting, evaporation
    for jk in range(ktopm2, klev + 1):
        if ldcum and jk >= kctop - 1:
            prain = prain + pdmfup[jk]
            if pmflxs[jk] > 0.0 and pten[jk] > TMELT:
                zcons1 = zcons1a * (1.0 + 0.5 * (pten[jk] - TMELT))
                zfac = zcons1 * (paph[jk + 1] - paph[jk])
                zsnmlt = min(pmflxs[jk], zfac * (pten[jk] - TMELT))
                pdpmel[jk] = zsnmlt
                pqsen[jk] = _foeewm(c, pten[jk] - zsnmlt / zfac) / pap[jk]
            zalfaw = _foealfa(c, pten[jk])
            # no liquid precipitation above melting level
            if pten[jk] < TMELT and zalfaw > 0.0:
                plglac[jk] = plglac[jk] + zalfaw * (pdmfup[jk] + pdmfdp[jk])
                zalfaw = 0.0
            pmflxr[jk + 1] = (pmflxr[jk] + zalfaw * (pdmfup[jk] + pdmfdp[jk])
                              + pdpmel[jk])
            pmflxs[jk + 1] = (pmflxs[jk] + (1.0 - zalfaw) * (pdmfup[jk] + pdmfdp[jk])
                              - pdpmel[jk])
            if pmflxr[jk + 1] + pmflxs[jk + 1] < 0.0:
                pdmfdp[jk] = -(pmflxr[jk] + pmflxs[jk] + pdmfup[jk])
                pmflxr[jk + 1] = 0.0
                pmflxs[jk + 1] = 0.0
                pdpmel[jk] = 0.0
            elif pmflxr[jk + 1] < 0.0:
                pmflxs[jk + 1] = pmflxs[jk + 1] + pmflxr[jk + 1]
                pmflxr[jk + 1] = 0.0
            elif pmflxs[jk + 1] < 0.0:
                pmflxr[jk + 1] = pmflxr[jk + 1] + pmflxs[jk + 1]
                pmflxs[jk + 1] = 0.0

    for jk in range(ktopm2, klev + 1):
        if ldcum and jk >= kcbot:
            zrfl = pmflxr[jk] + pmflxs[jk]
            if zrfl > _f32(1.0e-20):
                zdrfl1 = (zcpecons * max(0.0, pqsen[jk] - pqen[jk]) * zcucov
                          * (np.sqrt(paph[jk] / paph[klev + 1]) / _f32(5.09e-3)
                             * zrfl / zcucov) ** _f32(0.5777)
                          * (paph[jk + 1] - paph[jk]))
                zrnew = zrfl - zdrfl1
                zrmin = (zrfl - zcucov * max(0.0, rhevap * pqsen[jk] - pqen[jk])
                         * zcons2 * (paph[jk + 1] - paph[jk]))
                zrnew = max(zrnew, zrmin)
                zrfln = max(zrnew, 0.0)
                zdrfl = min(0.0, zrfln - zrfl)
                zdenom = 1.0 / max(_f32(1.0e-20), pmflxr[jk] + pmflxs[jk])
                zalfaw = _foealfa(c, pten[jk])
                if pten[jk] < TMELT:
                    zalfaw = 0.0
                zpdr = zalfaw * pdmfdp[jk]
                zpds = (1.0 - zalfaw) * pdmfdp[jk]
                pmflxr[jk + 1] = (pmflxr[jk] + zpdr + pdpmel[jk]
                                  + zdrfl * pmflxr[jk] * zdenom)
                pmflxs[jk + 1] = (pmflxs[jk] + zpds - pdpmel[jk]
                                  + zdrfl * pmflxs[jk] * zdenom)
                pdmfup[jk] = pdmfup[jk] + zdrfl
                if pmflxr[jk + 1] + pmflxs[jk + 1] < 0.0:
                    pdmfup[jk] = pdmfup[jk] - (pmflxr[jk + 1] + pmflxs[jk + 1])
                    pmflxr[jk + 1] = 0.0
                    pmflxs[jk + 1] = 0.0
                    pdpmel[jk] = 0.0
                elif pmflxr[jk + 1] < 0.0:
                    pmflxs[jk + 1] = pmflxs[jk + 1] + pmflxr[jk + 1]
                    pmflxr[jk + 1] = 0.0
                elif pmflxs[jk + 1] < 0.0:
                    pmflxr[jk + 1] = pmflxr[jk + 1] + pmflxs[jk + 1]
                    pmflxs[jk + 1] = 0.0
            else:
                pmflxr[jk + 1] = 0.0
                pmflxs[jk + 1] = 0.0
                pdmfdp[jk] = 0.0
                pdpmel[jk] = 0.0

    return ktopm2, ktype, lddraf, prain, pdpmel, pmflxr, pmflxs


# --- cudtdqn (T/q tendencies) -----------------------------------------------------
def _cudtdqn(c: _Const, klev, ktopm2, kctop, kdtop, ldcum, lddraf, ztmst,
             paph, pgeoh, pgeo, pten, ptenh, pqen, pqenh, pqsen, plglac,
             plude, pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq, pmful,
             pdmfup, pdmfdp, pdpmel, ptent, ptenq, pcte):
    zdp = _z1(klev)
    zdtdt = _z1(klev)
    zdqdt = _z1(klev)
    for jk in range(1, klev + 1):
        if ldcum:
            zdp[jk] = c.g / (paph[jk + 1] - paph[jk])
    for jk in range(ktopm2, klev + 1):
        if jk < klev:
            if ldcum:
                zalv = _foelhm(c, pten[jk])
                zdtdt[jk] = (zdp[jk] * c.rcpd
                             * (pmfus[jk + 1] - pmfus[jk] + pmfds[jk + 1]
                                - pmfds[jk] + c.alf * plglac[jk]
                                - c.alf * pdpmel[jk]
                                - zalv * (pmful[jk + 1] - pmful[jk]
                                          - plude[jk] - pdmfup[jk] - pdmfdp[jk])))
                zdqdt[jk] = (zdp[jk] * (pmfuq[jk + 1] - pmfuq[jk]
                                        + pmfdq[jk + 1] - pmfdq[jk]
                                        + pmful[jk + 1] - pmful[jk]
                                        - plude[jk] - pdmfup[jk] - pdmfdp[jk]))
        else:
            if ldcum:
                zalv = _foelhm(c, pten[jk])
                zdtdt[jk] = (-zdp[jk] * c.rcpd
                             * (pmfus[jk] + pmfds[jk] + c.alf * pdpmel[jk]
                                - zalv * (pmful[jk] + pdmfup[jk]
                                          + pdmfdp[jk] + plude[jk])))
                zdqdt[jk] = (-zdp[jk] * (pmfuq[jk] + plude[jk] + pmfdq[jk]
                                         + (pmful[jk] + pdmfup[jk] + pdmfdp[jk])))
    for jk in range(ktopm2, klev + 1):
        if ldcum:
            ptent[jk] = ptent[jk] + zdtdt[jk]
            ptenq[jk] = ptenq[jk] + zdqdt[jk]
            pcte[jk] = zdp[jk] * plude[jk]


# --- cududvn (u/v tendencies) ------------------------------------------------------
def _cududvn(c: _Const, klev, ktopm2, ktype, kcbot, kctop, ldcum, ztmst,
             paph, puen, pven, pmfu, pmfd, puu, pud, pvu, pvd, ptenu, ptenv):
    zuen = _z1(klev)
    zven = _z1(klev)
    zdp = _z1(klev)
    zmfuu = _z1(klev)
    zmfdu = _z1(klev)
    zmfuv = _z1(klev)
    zmfdv = _z1(klev)
    zdudt = _z1(klev)
    zdvdt = _z1(klev)
    for jk in range(1, klev + 1):
        if ldcum:
            zuen[jk] = puen[jk]
            zven[jk] = pven[jk]
            zdp[jk] = c.g / (paph[jk + 1] - paph[jk])
    for jk in range(ktopm2, klev + 1):
        ik = jk - 1
        if ldcum:
            zmfuu[jk] = pmfu[jk] * (puu[jk] - zuen[ik])
            zmfuv[jk] = pmfu[jk] * (pvu[jk] - zven[ik])
            zmfdu[jk] = pmfd[jk] * (pud[jk] - zuen[ik])
            zmfdv[jk] = pmfd[jk] * (pvd[jk] - zven[ik])
    for jk in range(ktopm2, klev + 1):
        if ldcum and jk > kcbot:
            ikb = kcbot
            zzp = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ikb]))
            if ktype == 3:
                zzp = zzp * zzp
            zmfuu[jk] = zmfuu[ikb] * zzp
            zmfuv[jk] = zmfuv[ikb] * zzp
            zmfdu[jk] = zmfdu[ikb] * zzp
            zmfdv[jk] = zmfdv[ikb] * zzp
    for jk in range(ktopm2, klev + 1):
        if jk < klev:
            ik = jk + 1
            if ldcum:
                zdudt[jk] = zdp[jk] * (zmfuu[ik] - zmfuu[jk] + zmfdu[ik] - zmfdu[jk])
                zdvdt[jk] = zdp[jk] * (zmfuv[ik] - zmfuv[jk] + zmfdv[ik] - zmfdv[jk])
        else:
            if ldcum:
                zdudt[jk] = -zdp[jk] * (zmfuu[jk] + zmfdu[jk])
                zdvdt[jk] = -zdp[jk] * (zmfuv[jk] + zmfdv[jk])
    for jk in range(ktopm2, klev + 1):
        if ldcum:
            ptenu[jk] = ptenu[jk] + zdudt[jk]
            ptenv[jk] = ptenv[jk] + zdvdt[jk]


# --- cumastrn (master routine) -----------------------------------------------------
def _cumastrn(c: _Const, klev, pten, pqen, puen, pven, pverv, pqsen,
              pqhfl, ztmst, pap, paph, pgeo, ptte, pqte, pvom, pvol,
              phhfl, lndj, zgeoh, dx, scale_fac, scale_fac2):
    """Single-column cumastrn.  Mutates ptte/pqte/pvom/pvol/pqsen in place.

    Returns (prsfc, pssfc, ldcum, ktype, kcbot, kctop, ptu, pqu, plu,
    plude, pmfu, pmfd, prain, pcte)."""
    klevp1 = klev + 1
    klevm1 = klev - 1
    zcons = 1.0 / (c.g * ztmst)
    zcons2 = 3.0 / (c.g * ztmst)

    prsfc = 0.0
    pssfc = 0.0
    prain = 0.0
    pcte = _z1(klev)

    # 2. cuinin
    (ztenh, zqenh, zqsenh, ilwmin, ptu, pqu, ztd, zqd, zuu, zvu, zud, zvd,
     plu, ilab) = _cuinin(c, klev, pten, pqen, pqsen, puen, pven, pverv,
                          pgeo, paph, zgeoh)

    # 3.0 cloud base calculations (cutypen)
    ldcum, kcbot, ictop0, ktype, wbase, kdpl = _cutypen(
        c, klev, pqen, ztenh, zqenh, zqsenh, zgeoh, paph, phhfl, pqhfl,
        pgeo, pqsen, pap, pten, lndj, ptu, pqu, ilab, plu)

    # 3(b) first-guess cloud-base mass flux
    zdhpbl = 0.0
    upbl = 0.0
    idtop = 0
    for jk in range(2, klev + 1):
        if jk >= kcbot and ldcum:
            zdhpbl = zdhpbl + ((c.alv * pqte[jk] + c.cpd * ptte[jk])
                               * (paph[jk + 1] - paph[jk]))
            if lndj == 0:
                wspeed = np.sqrt(puen[jk] ** 2 + pven[jk] ** 2)
                upbl = upbl + wspeed * (paph[jk + 1] - paph[jk])

    zmfub = 0.0
    if ldcum:
        ikb = kcbot
        zmfmax = (paph[ikb] - paph[ikb - 1]) * zcons2
        if ktype == 1:
            zmfub = _f32(0.1) * zmfmax
        elif ktype == 2:
            zqumqe = pqu[ikb] + plu[ikb] - zqenh[ikb]
            zdqmin = max(_f32(0.01) * zqenh[ikb], _f32(1.0e-10))
            zdh = c.cpd * (ptu[ikb] - ztenh[ikb]) + c.alv * zqumqe
            zdh = c.g * max(zdh, 1.0e5 * zdqmin)
            if zdhpbl > 0.0:
                zmfub = zdhpbl / zdh
                zmfub = min(zmfub, zmfmax)
            else:
                zmfub = _f32(0.1) * zmfmax
                ldcum = False
    else:
        zmfub = 0.0

    # 4.0 cloud ascent
    zmfus = _z1(klev)
    zmfuq = _z1(klev)
    zmful = _z1(klev)
    plude = _z1(klev)
    zdmfup = _z1(klev)
    pmfu = _z1(klev)
    zhcbase = 0.0
    icum = 0
    kctop = 0
    (ldcum, ktype, kcbot, kctop, ictop0, zmfub, wup, zlglac,
     pmfude_rate) = _cuascn(
        c, klev, ztmst, ztenh, zqenh, puen, pven, pten, pqen, pqsen,
        pgeo, zgeoh, pap, paph, pqte, pverv, ilwmin, ldcum, ktype, ilab,
        ptu, pqu, plu, zuu, zvu, pmfu, zmfub, zmfus, zmfuq, zmful,
        plude, zdmfup, kcbot, kctop, ictop0, lndj, wbase, kdpl)

    # 5. cloud depth check, adjust ktype, precipitation rate
    if ldcum:
        ikb = kcbot
        itopm2 = kctop
        zpbmpt = paph[ikb] - paph[itopm2]
        if ktype == 1 and zpbmpt < ZDNOPRC:
            ktype = 2
        if ktype == 2 and zpbmpt >= ZDNOPRC:
            ktype = 1
        ictop0 = kctop
    zrfl = zdmfup[1]
    for jk in range(2, klev + 1):
        zrfl = zrfl + zdmfup[jk]

    pmfd = _z1(klev)
    zmfds = _z1(klev)
    zmfdq = _z1(klev)
    zdmfdp = _z1(klev)
    zdpmel = _z1(klev)
    pmfdde_rate = _z1(klev)
    loddraf = False

    # 6.0 downdrafts
    if LMFDD:
        (pmfd, zmfds, zmfdq, zdmfdp, idtop, loddraf, zrfl) = _cudlfsn(
            c, klev, kcbot, kctop, lndj, ldcum, ztenh, zqenh, puen, pven,
            pten, pqsen, pgeo, zgeoh, paph, ptu, pqu, plu, zuu, zvu,
            zmfub, zrfl, ztd, zqd, zud, zvd)
        zrfl, pmfdde_rate = _cuddrafn(
            c, klev, loddraf, ztenh, zqenh, puen, pven, pgeo, zgeoh, paph,
            zrfl, ztd, zqd, zud, zvd, pmfu, pmfd, zmfds, zmfdq, zdmfdp)

    # 6.1 CAPE closure for deep convection
    zheat = 0.0
    zcape = 0.0
    zcape1 = 0.0
    zcape2 = 0.0
    zmfub1 = zmfub
    ztauc = 0.0
    ztaubl = 0.0
    if ldcum and ktype == 1:
        ikb = kcbot
        ikt = kctop
        ztauc = (zgeoh[ikt] - zgeoh[ikb]) / ((2.0 + min(15.0, wup)) * c.g)
        if lndj == 0:
            upbl = 2.0 + upbl / (paph[klev + 1] - paph[ikb])
            ztaubl = (zgeoh[ikb] - zgeoh[klev + 1]) / (c.g * upbl)
            ztaubl = min(300.0, ztaubl)
        else:
            ztaubl = ztauc

    for jk in range(1, klev + 1):
        llo1 = ldcum and ktype == 1
        if llo1 and jk <= kcbot and jk > kctop:
            ikb = kcbot
            zdz = pgeo[jk - 1] - pgeo[jk]
            zdp_ = pap[jk] - pap[jk - 1]
            zheat = zheat + (((pten[jk - 1] - pten[jk] + zdz * c.rcpd) / ztenh[jk]
                              + c.vtmpc1 * (pqen[jk - 1] - pqen[jk]))
                             * (c.g * (pmfu[jk] + pmfd[jk])))
            zcape1 = zcape1 + (((ptu[jk] - ztenh[jk]) / ztenh[jk]
                                + c.vtmpc1 * (pqu[jk] - zqenh[jk]) - plu[jk])
                               * zdp_)
        if llo1 and jk >= kcbot:
            if (paph[klev + 1] - paph[kdpl]) < 50.0e2:
                zdp_ = paph[jk + 1] - paph[jk]
                zcape2 = zcape2 + (ztaubl
                                   * ((1.0 + c.vtmpc1 * pqen[jk]) * ptte[jk]
                                      + c.vtmpc1 * pten[jk] * pqte[jk]) * zdp_)

    if ldcum and ktype == 1:
        ikb = kcbot
        ikt = kctop
        ztauc = max(ztmst, ztauc)
        ztauc = max(360.0, ztauc)
        ztauc = min(10800.0, ztauc)
        ztau = ztauc * scale_fac
        if NONEQUIL:
            zcape2 = max(0.0, zcape2)
            zcape = max(0.0, min(zcape1 - zcape2, 5000.0))
        else:
            zcape = max(0.0, min(zcape1, 5000.0))
        zheat = max(_f32(1.0e-4), zheat)
        zmfub1 = (zcape * zmfub) / (zheat * ztau)
        zmfub1 = max(zmfub1, _f32(0.001))
        zmfmax = (paph[ikb] - paph[ikb - 1]) * zcons2
        zmfub1 = min(zmfub1, zmfmax)

    # 6.2 shallow closure (moist static energy budget with downdrafts)
    if ldcum and ktype == 2:
        ikb = kcbot
        if pmfd[ikb] < 0.0 and loddraf:
            zeps = -pmfd[ikb] / max(zmfub, CMFCMIN)
        else:
            zeps = 0.0
        zqumqe = (pqu[ikb] + plu[ikb]
                  - zeps * zqd[ikb] - (1.0 - zeps) * zqenh[ikb])
        zdqmin = max(_f32(0.01) * zqenh[ikb], CMFCMIN)
        zmfmax = (paph[ikb] - paph[ikb - 1]) * zcons2
        zdh = (c.cpd * (ptu[ikb] - zeps * ztd[ikb]
                        - (1.0 - zeps) * ztenh[ikb]) + c.alv * zqumqe)
        zdh = c.g * max(zdh, 1.0e5 * zdqmin)
        if zdhpbl > 0.0:
            zmfub1 = zdhpbl / zdh
        else:
            zmfub1 = zmfub
        zmfub1 = zmfub1 / scale_fac2
        zmfub1 = min(zmfub1, zmfmax)

    # 6.3 mid-level convection
    if ldcum and ktype == 3:
        zmfub1 = zmfub

    # 6.4 scale the downdraft mass flux
    for jk in range(1, klev + 1):
        if ldcum:
            zfac = zmfub1 / max(zmfub, CMFCMIN)
            pmfd[jk] = pmfd[jk] * zfac
            zmfds[jk] = zmfds[jk] * zfac
            zmfdq[jk] = zmfdq[jk] * zfac
            zdmfdp[jk] = zdmfdp[jk] * zfac
            pmfdde_rate[jk] = pmfdde_rate[jk] * zfac

    # 6.5 scale the updraft mass flux
    zmfs = 1.0
    if ldcum:
        zmfs = zmfub1 / max(CMFCMIN, zmfub)
    for jk in range(2, klev + 1):
        if ldcum and jk >= kctop - 1:
            ikb = kcbot
            if jk > ikb:
                zdz = ((paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[ikb]))
                pmfu[jk] = pmfu[ikb] * zdz
            zmfmax = (paph[jk] - paph[jk - 1]) * zcons2
            if pmfu[jk] * zmfs > zmfmax:
                zmfs = min(zmfs, zmfmax / pmfu[jk])
    for jk in range(2, klev + 1):
        if ldcum and jk <= kcbot and jk >= kctop - 1:
            pmfu[jk] = pmfu[jk] * zmfs
            zmfus[jk] = zmfus[jk] * zmfs
            zmfuq[jk] = zmfuq[jk] * zmfs
            zmful[jk] = zmful[jk] * zmfs
            zdmfup[jk] = zdmfup[jk] * zmfs
            plude[jk] = plude[jk] * zmfs
            pmfude_rate[jk] = pmfude_rate[jk] * zmfs

    # 6.6 if ktype=2, kcbot=kctop is not allowed
    if ktype == 2 and kcbot == kctop and kcbot >= klev - 1:
        ldcum = False
        ktype = 0

    # (lmfscv and lmfpen are both .true. -> the switch-off block is inert)

    # 6.7 set downdraft mass fluxes to zero above cloud top
    if loddraf and idtop <= kctop:
        idtop = kctop + 1
    for jk in range(2, klev + 1):
        if loddraf:
            if jk < idtop:
                pmfd[jk] = 0.0
                zmfds[jk] = 0.0
                zmfdq[jk] = 0.0
                pmfdde_rate[jk] = 0.0
                zdmfdp[jk] = 0.0
            elif jk == idtop:
                pmfdde_rate[jk] = 0.0

    # 7.0 final convective fluxes in cuflxn
    itopm2, ktype, loddraf, prain, zdpmel, pmflxr, pmflxs = _cuflxn(
        c, klev, ztmst, pten, pqen, pqsen, ztenh, zqenh, paph, pap, zgeoh,
        lndj, ldcum, kcbot, kctop, idtop, ktype, loddraf, pmfu, pmfd,
        zmfus, zmfds, zmfuq, zmfdq, zmful, plude, zdmfup, zdmfdp,
        zlglac, pmfdde_rate)

    # some adjustments needed
    zmfs = 1.0
    zmfuub = 0.0
    for jk in range(2, klev + 1):
        if loddraf and jk >= idtop - 1:
            zmfmax = pmfu[jk] * _f32(0.98)
            if pmfd[jk] + zmfmax + _f32(1.0e-15) < 0.0:
                zmfs = min(zmfs, -zmfmax / pmfd[jk])

    for jk in range(2, klev + 1):
        if zmfs < 1.0 and jk >= idtop - 1:
            pmfd[jk] = pmfd[jk] * zmfs
            zmfds[jk] = zmfds[jk] * zmfs
            zmfdq[jk] = zmfdq[jk] * zmfs
            pmfdde_rate[jk] = pmfdde_rate[jk] * zmfs
            zmfuub = zmfuub - (1.0 - zmfs) * zdmfdp[jk]
            pmflxr[jk + 1] = pmflxr[jk + 1] + zmfuub
            zdmfdp[jk] = zdmfdp[jk] * zmfs

    for jk in range(2, klev):
        if loddraf and jk >= idtop - 1:
            zerate = -pmfd[jk] + pmfd[jk - 1] + pmfdde_rate[jk]
            if zerate < 0.0:
                pmfdde_rate[jk] = pmfdde_rate[jk] - zerate
        if ldcum and jk >= kctop - 1:
            zerate = pmfu[jk] - pmfu[jk + 1] + pmfude_rate[jk]
            if zerate < 0.0:
                pmfude_rate[jk] = pmfude_rate[jk] - zerate
            zdmfup[jk] = (pmflxr[jk + 1] + pmflxs[jk + 1]
                          - pmflxr[jk] - pmflxs[jk])
            zdmfdp[jk] = 0.0

    # avoid negative humidities at ddraught top
    if loddraf:
        jk = idtop
        ik = min(jk + 1, klev)
        if zmfdq[jk] < _f32(0.3) * zmfdq[ik]:
            zmfdq[jk] = _f32(0.3) * zmfdq[ik]

    # avoid negative humidities near cloud top
    for jk in range(2, klev + 1):
        if ldcum and jk >= kctop - 1 and jk < kcbot:
            zdz = ztmst * c.g / (paph[jk + 1] - paph[jk])
            zmfa = (zmfuq[jk + 1] + zmfdq[jk + 1] - zmfuq[jk] - zmfdq[jk]
                    + zmful[jk + 1] - zmful[jk] + zdmfup[jk])
            zmfa = (zmfa - plude[jk]) * zdz
            if pqen[jk] + zmfa < 0.0:
                plude[jk] = plude[jk] + 2.0 * (pqen[jk] + zmfa) / zdz
            if plude[jk] < 0.0:
                plude[jk] = 0.0
        if not ldcum:
            pmfude_rate[jk] = 0.0
        if abs(pmfd[jk - 1]) < _f32(1.0e-20):
            pmfdde_rate[jk] = 0.0

    prsfc = pmflxr[klev + 1]
    pssfc = pmflxs[klev + 1]

    # 8.0 update T and q tendencies
    _cudtdqn(c, klev, itopm2, kctop, idtop, ldcum, loddraf, ztmst, paph,
             zgeoh, pgeo, pten, ztenh, pqen, zqenh, pqsen, zlglac, plude,
             pmfu, pmfd, zmfus, zmfds, zmfuq, zmfdq, zmful, zdmfup,
             zdmfdp, zdpmel, ptte, pqte, pcte)

    # 9.0 update u and v tendencies
    if LMFDUDV:
        for jk in range(klev - 1, 1, -1):
            ik = jk + 1
            if ldcum:
                if jk == kcbot and ktype < 3:
                    ikb = kdpl
                    zuu[jk] = puen[ikb - 1]
                    zvu[jk] = pven[ikb - 1]
                elif jk == kcbot and ktype == 3:
                    zuu[jk] = puen[jk - 1]
                    zvu[jk] = pven[jk - 1]
                if jk < kcbot and jk >= kctop:
                    if MOMTRANS == 1:
                        zfac = 0.0
                        if ktype == 1 or ktype == 3:
                            zfac = 2.0
                        if ktype == 1 and jk <= kctop + 2:
                            zfac = 3.0
                        zerate = (pmfu[jk] - pmfu[ik]
                                  + (1.0 + zfac) * pmfude_rate[jk])
                        zderate = (1.0 + zfac) * pmfude_rate[jk]
                        zmfa = 1.0 / max(CMFCMIN, pmfu[jk])
                        zuu[jk] = (zuu[ik] * pmfu[ik] + zerate * puen[jk]
                                   - zderate * zuu[ik]) * zmfa
                        zvu[jk] = (zvu[ik] * pmfu[ik] + zerate * pven[jk]
                                   - zderate * zvu[ik]) * zmfa
                    else:
                        pgf_u = (-PGCOEF * 0.5
                                 * (pmfu[ik] * (puen[ik] - puen[jk])
                                    + pmfu[jk] * (puen[jk] - puen[jk - 1])))
                        pgf_v = (-PGCOEF * 0.5
                                 * (pmfu[ik] * (pven[ik] - pven[jk])
                                    + pmfu[jk] * (pven[jk] - pven[jk - 1])))
                        zerate = pmfu[jk] - pmfu[ik] + pmfude_rate[jk]
                        zderate = pmfude_rate[jk]
                        zmfa = 1.0 / max(CMFCMIN, pmfu[jk])
                        zuu[jk] = (zuu[ik] * pmfu[ik] + zerate * puen[jk]
                                   - zderate * zuu[ik] + pgf_u) * zmfa
                        zvu[jk] = (zvu[ik] * pmfu[ik] + zerate * pven[jk]
                                   - zderate * zvu[ik] + pgf_v) * zmfa

        if LMFDD:
            for jk in range(3, klev + 1):
                ik = jk - 1
                if ldcum:
                    if jk == idtop:
                        zud[jk] = 0.5 * (zuu[jk] + puen[ik])
                        zvd[jk] = 0.5 * (zvu[jk] + pven[ik])
                    elif jk > idtop:
                        zerate = -pmfd[jk] + pmfd[ik] + pmfdde_rate[jk]
                        zmfa = 1.0 / min(-CMFCMIN, pmfd[jk])
                        zud[jk] = (zud[ik] * pmfd[ik] - zerate * puen[ik]
                                   + pmfdde_rate[jk] * zud[ik]) * zmfa
                        zvd[jk] = (zvd[ik] * pmfd[ik] - zerate * pven[ik]
                                   + pmfdde_rate[jk] * zvd[ik]) * zmfa

        # rescale massfluxes for stability in momentum
        zmfs = 1.0
        for jk in range(2, klev + 1):
            if ldcum and jk >= kctop - 1:
                zmfmax = (paph[jk] - paph[jk - 1]) * zcons
                if pmfu[jk] > zmfmax and jk >= kctop:
                    zmfs = min(zmfs, zmfmax / pmfu[jk])
        zmfuus = _z1(klev)
        zmfdus = _z1(klev)
        for jk in range(1, klev + 1):
            zmfuus[jk] = pmfu[jk]
            zmfdus[jk] = pmfd[jk]
            if ldcum and jk >= kctop - 1:
                zmfuus[jk] = pmfu[jk] * zmfs
                zmfdus[jk] = pmfd[jk] * zmfs

        # 9.1 update u and v in cududvn
        ztenu = _z1(klev)
        ztenv = _z1(klev)
        for jk in range(1, klev + 1):
            ztenu[jk] = pvom[jk]
            ztenv[jk] = pvol[jk]

        _cududvn(c, klev, itopm2, ktype, kcbot, kctop, ldcum, ztmst, paph,
                 puen, pven, zmfuus, zmfdus, zuu, zud, zvu, zvd, pvom, pvol)

        # KE dissipation
        zsum12 = 0.0
        zsum22 = 0.0
        zuv2 = _z1(klev)
        for jk in range(1, klev + 1):
            zuv2[jk] = 0.0
            if ldcum and jk >= kctop - 1:
                zdz = paph[jk + 1] - paph[jk]
                zduten = pvom[jk] - ztenu[jk]
                zdvten = pvol[jk] - ztenv[jk]
                zuv2[jk] = np.sqrt(zduten ** 2 + zdvten ** 2)
                zsum22 = zsum22 + zuv2[jk] * zdz
                zsum12 = zsum12 - (puen[jk] * zduten + pven[jk] * zdvten) * zdz
        for jk in range(1, klev + 1):
            if ldcum and jk >= kctop - 1:
                ztdis = c.rcpd * zsum12 * zuv2[jk] / max(_f32(1.0e-15), zsum22)
                ptte[jk] = ptte[jk] + ztdis

    # 10. (lmfscv/lmfpen both .true. -> a-posteriori zeroing block is inert)

    return (prsfc, pssfc, ldcum, ktype, kcbot, kctop, ptu, pqu, plu,
            plude, pmfu, pmfd, prain, pcte)


# --- cu_ntiedtke_run (scheme interface) ----------------------------------------
def _cu_ntiedtke_run(c: _Const, km, dt, dx, evap, hfx, lndj,
                     pu, pv, pt, pqv, pqc, pqi, pqvf, ptf, poz, pzz, pomg,
                     pap, paph):
    """Single-column cu_ntiedtke_run.  All level arrays are 1-based TOP-DOWN
    (index 1 = model top, km = lowest level, km1 = surface interface).
    Mutates pu/pv/pt/pqv/pqc/pqi in place; returns zprecc (m of precip per
    delt) plus (ktype, ldcum) diagnostics."""
    km1 = km + 1
    ztmst = dt

    # scale-dependency factor
    dxref = 15000.0
    if dx < dxref:
        scale_fac = (_f32(1.06133) + np.log(dxref / dx)) ** 3
        scale_fac2 = scale_fac ** 0.5
    else:
        scale_fac = 1.0 + _f32(1.33e-5) * dx
        scale_fac2 = 1.0

    zrain = 0.0
    prsfc = 0.0
    pssfc = 0.0
    pqhfl = evap
    phhfl = hfx
    pgeoh = _z1(km1)
    pgeoh[km1] = c.g * pzz[km1]

    pcte = _z1(km)
    pvom = _z1(km)
    pvol = _z1(km)
    ztp1 = _z1(km)
    zqp1 = _z1(km)
    pum1 = _z1(km)
    pvm1 = _z1(km)
    pverv = _z1(km)
    pgeo = _z1(km)
    zqsat = _z1(km)
    pqte = _z1(km)
    zqq = _z1(km)
    ptte = _z1(km)
    ztt = _z1(km)
    for k in range(1, km + 1):
        pcte[k] = 0.0
        pvom[k] = 0.0
        pvol[k] = 0.0
        ztp1[k] = pt[k]
        zqp1[k] = pqv[k] / (1.0 + pqv[k])
        pum1[k] = pu[k]
        pvm1[k] = pv[k]
        pverv[k] = pomg[k]
        pgeo[k] = c.g * poz[k]
        pgeoh[k] = c.g * pzz[k]
        tt = ztp1[k]
        zew = _foeewm(c, tt)
        zqs = zew / pap[k]
        zqs = min(0.5, zqs)
        zcor = 1.0 / (1.0 - c.vtmpc1 * zqs)
        zqsat[k] = zqs * zcor
        pqte[k] = pqvf[k]
        zqq[k] = pqte[k]
        ptte[k] = ptf[k]
        ztt[k] = ptte[k]

    # 2. cumastrn
    (prsfc, pssfc, ldcum, ktype, kcbot, kctop, ztu, zqu, zlu, zlude,
     zmfu, zmfd, zrain, pcte) = _cumastrn(
        c, km, ztp1, zqp1, pum1, pvm1, pverv, zqsat, pqhfl, ztmst, pap,
        paph, pgeo, ptte, pqte, pvom, pvol, phhfl, lndj, pgeoh, dx,
        scale_fac, scale_fac2)

    # include the cloud water and cloud ice detrained from convection
    for k in range(1, km + 1):
        if pcte[k] > 0.0:
            fliq = _foealfa(c, ztp1[k])
            fice = 1.0 - fliq
            pqc[k] = pqc[k] + fliq * pcte[k] * ztmst
            pqi[k] = pqi[k] + fice * pcte[k] * ztmst

    for k in range(1, km + 1):
        pt[k] = ztp1[k] + (ptte[k] - ztt[k]) * ztmst
        zqp1[k] = zqp1[k] + (pqte[k] - zqq[k]) * ztmst
        pqv[k] = zqp1[k] / (1.0 - zqp1[k])

    # Fortran: zprecc = amax1(0.0, (prsfc+pssfc)*ztmst) -- AMAX1 is the
    # REAL*4-specific intrinsic, so gfortran demotes the fp64 expression to
    # fp32 for the max and the result (verified bit-exact vs the oracle).
    zprecc = _F(max(np.float32(0.0), np.float32((prsfc + pssfc) * ztmst)))

    if LMFDUDV:
        for k in range(1, km + 1):
            pu[k] = pu[k] + pvom[k] * ztmst
            pv[k] = pv[k] + pvol[k] * ztmst

    return zprecc, ktype, ldcum


# --- public column API (WRF bottom-up orientation, savepoint-compatible) --------
def ntiedtke_column(t, qv, qc, qi, p, p8w, dz, rho, pi, u, v, w,
                    qvften, thften, qfx, hfx, xland, dt,
                    dx=9000.0, stepcu=5, itimestep=2,
                    cp=1004.5, rd=287.0, rv=461.6,
                    xlv=2.5e6, xls=2.85e6, xlf=3.5e5, grav=9.81):
    """Faithful fp64 New-Tiedtke single-column step (cu_physics=16).

    Inputs are WRF bottom-up columns exactly as stored in the oracle
    savepoints: ``t``..``thften`` of shape (kx,), ``p8w``/``w`` of shape
    (kx+1,).  Implements ``cu_ntiedtke_pre_run`` -> ``cu_ntiedtke_run`` ->
    ``cu_ntiedtke_post_run`` from ``module_cu_ntiedtke.F``.

    Returns a dict with RTHCUTEN, RQVCUTEN, RQCCUTEN, RQICUTEN, RUCUTEN,
    RVCUTEN (bottom-up, shape (kx,)), RAINCV, PRATEC, KTYPE, LDCUM.
    """
    c = _Const(cp=cp, rd=rd, rv=rv, xlv=xlv, xls=xls, xlf=xlf, grav=grav)
    t = np.asarray(t, dtype=_F)
    qv = np.asarray(qv, dtype=_F)
    qc = np.asarray(qc, dtype=_F)
    qi = np.asarray(qi, dtype=_F)
    p = np.asarray(p, dtype=_F)
    p8w = np.asarray(p8w, dtype=_F)
    dz = np.asarray(dz, dtype=_F)
    rho = np.asarray(rho, dtype=_F)
    pi = np.asarray(pi, dtype=_F)
    u = np.asarray(u, dtype=_F)
    v = np.asarray(v, dtype=_F)
    w = np.asarray(w, dtype=_F)
    qvften = np.asarray(qvften, dtype=_F)
    thften = np.asarray(thften, dtype=_F)
    kx = t.shape[0]
    kx1 = kx + 1
    if p8w.shape[0] != kx1 or w.shape[0] != kx1:
        raise ValueError("p8w and w must be interface columns of length kx+1")

    # --- cu_ntiedtke_pre_run ---------------------------------------------------
    delt = _F(dt) * stepcu
    slimsk = int(abs(_F(xland) - 2.0))  # Fortran INTEGER truncation

    zi = np.zeros(kx1, dtype=_F)
    for k in range(kx):
        zi[k + 1] = zi[k] + dz[k]
    zl = np.zeros(kx, dtype=_F)
    dot = np.zeros(kx, dtype=_F)
    for k in range(kx):
        zl[k] = 0.5 * (zi[k] + zi[k + 1])
        dot[k] = -0.5 * c.g * rho[k] * (w[k] + w[k + 1])

    # flip bottom-up (0-based) -> top-down (1-based)
    ghti = _z1(kx1)
    prsi = _z1(kx1)
    for k in range(kx1):
        zz = kx1 - k
        ghti[zz] = zi[k]
        prsi[zz] = p8w[k]
    ghtl = _z1(kx)
    omg = _z1(kx)
    prsl = _z1(kx)
    tf = _z1(kx)
    qvf = _z1(kx)
    qcf = _z1(kx)
    qif = _z1(kx)
    uf = _z1(kx)
    vf = _z1(kx)
    qvftenz = _z1(kx)
    thftenz = _z1(kx)
    for k in range(kx):
        zz = kx - k
        ghtl[zz] = zl[k]
        omg[zz] = dot[k]
        prsl[zz] = p[k]
        tf[zz] = t[k]
        qvf[zz] = qv[k]
        qcf[zz] = qc[k]
        qif[zz] = qi[k]
        uf[zz] = u[k]
        vf[zz] = v[k]
    if itimestep == 1:
        for k in range(1, kx + 1):
            qvftenz[k] = 0.0
            thftenz[k] = 0.0
    else:
        for k in range(kx):
            zz = kx - k
            qvftenz[zz] = qvften[k]
            thftenz[zz] = thften[k]

    # --- cu_ntiedtke_run ---------------------------------------------------------
    rn, ktype, ldcum = _cu_ntiedtke_run(
        c, kx, delt, _F(dx), _F(qfx), _F(hfx), slimsk,
        uf, vf, tf, qvf, qcf, qif, qvftenz, thftenz, ghtl, ghti, omg,
        prsl, prsi)

    # --- cu_ntiedtke_post_run ----------------------------------------------------
    rdelt = 1.0 / delt
    raincv = rn / stepcu
    pratec = rn / (stepcu * _F(dt))
    rthcuten = np.zeros(kx, dtype=_F)
    rqvcuten = np.zeros(kx, dtype=_F)
    rqccuten = np.zeros(kx, dtype=_F)
    rqicuten = np.zeros(kx, dtype=_F)
    rucuten = np.zeros(kx, dtype=_F)
    rvcuten = np.zeros(kx, dtype=_F)
    for k in range(kx):
        zz = kx - k
        rthcuten[k] = (tf[zz] - t[k]) / pi[k] * rdelt
        rqvcuten[k] = (qvf[zz] - qv[k]) * rdelt
        rqccuten[k] = (qcf[zz] - qc[k]) * rdelt
        rqicuten[k] = (qif[zz] - qi[k]) * rdelt
        rucuten[k] = (uf[zz] - u[k]) * rdelt
        rvcuten[k] = (vf[zz] - v[k]) * rdelt

    return {
        "RTHCUTEN": rthcuten,
        "RQVCUTEN": rqvcuten,
        "RQCCUTEN": rqccuten,
        "RQICUTEN": rqicuten,
        "RUCUTEN": rucuten,
        "RVCUTEN": rvcuten,
        "RAINCV": float(raincv),
        "PRATEC": float(pratec),
        "KTYPE": int(ktype),
        "LDCUM": bool(ldcum),
    }


__all__ = ["ntiedtke_column"]
