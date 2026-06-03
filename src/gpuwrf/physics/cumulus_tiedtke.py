"""WRF modified-Tiedtke cumulus column adapter.

This module is a faithful single-column transcription of WRF
``phys/module_cu_tiedtke.F`` for the v0.6.0 savepoint lane. The implementation
keeps WRF's top-down mass-flux indexing internally and exposes bottom-up WRF/JAX
columns at the boundary, matching the oracle driver in ``proofs/v060/oracle``.

The current implementation is intentionally CPU-oriented Python/JAX glue: it
returns JAX arrays and the frozen ``PhysicsStepResult`` contract, but it is not
yet a ``jit``/``vmap`` production kernel. The parity gate is still meaningful
because the numerical algorithm below is independent from the compiled Fortran
oracle and is checked against savepoints produced by the unmodified WRF source.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsCarry, PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency


RD = 287.0
RV = 461.6
CPD = 7.0 * RD / 2.0
ALV = 2.5e6
ALS = 2.85e6
ALF = 3.5e5
G = 9.81
RCPD = 1.0 / CPD
VTMP_C1 = RV / RD - 1.0
T000 = 273.15
TMELT = 273.16
HGFR = 233.15
ZRG = 1.0 / G
C1ES = 610.78
C2ES = C1ES * RD / RV
C3LES = 17.269
C3IES = 21.875
C4LES = 35.86
C4IES = 7.66
C5LES = C3LES * (TMELT - C4LES)
C5IES = C3IES * (TMELT - C4IES)

ENTRPEN = 1.0e-4
ENTRSCV = 1.2e-3
ENTRMID = 1.0e-4
ENTRDD = 2.0e-4
CMFCTOP = 0.30
CMFCMAX = 1.0
CMFCMIN = 1.0e-10
CMFDEPS = 0.30
CPRCON = 1.1e-3 / G
ZDNOPRC = 1.5e4
RHC = 0.80
RHM = 1.0
ZBUO0 = 0.50
CRIRH = 0.70
FDBK = 1.0
ZTAU = 2400.0
CEVAPCU1 = 1.93e-6 * 261.0 * 0.5 / G
CEVAPCU2 = 1.0e3 / (38.3 * 0.293)

LMFPEN = True
LMFMID = True
LMFSCV = True
LMFDD = True
LMFDUDV = True


@dataclass
class _TieState:
    pu: np.ndarray
    pv: np.ndarray
    pt: np.ndarray
    pqv: np.ndarray
    pqc: np.ndarray
    pqi: np.ndarray
    pqvf: np.ndarray
    pqvbl: np.ndarray
    poz: np.ndarray
    pomg: np.ndarray
    pap: np.ndarray
    paph: np.ndarray
    evap: float
    lndj: int
    sig1: np.ndarray
    dt: float


def _zeros(klev: int) -> np.ndarray:
    return np.zeros(klev + 2, dtype=np.float64)


def _ints(klev: int) -> np.ndarray:
    return np.zeros(klev + 2, dtype=np.int32)


def _tlucua(tt: float) -> float:
    if tt - TMELT > 0.0:
        zcvm3 = C3LES
        zcvm4 = C4LES
    else:
        zcvm3 = C3IES
        zcvm4 = C4IES
    return C2ES * np.exp(zcvm3 * (tt - TMELT) / (tt - zcvm4))


def _tlucub(tt: float) -> float:
    z5alvcp = C5LES * ALV / CPD
    z5alscp = C5IES * ALS / CPD
    if tt - TMELT > 0.0:
        zcvm4 = C4LES
        zcvm5 = z5alvcp
    else:
        zcvm4 = C4IES
        zcvm5 = z5alscp
    return zcvm5 * (1.0 / (tt - zcvm4)) ** 2


def _tlucuc(tt: float) -> float:
    return ALV / CPD if tt - TMELT > 0.0 else ALS / CPD


def _cuadjtq(klev: int, kk: int, pp: float, pt: np.ndarray, pq: np.ndarray, ldflag: bool, kcall: int) -> None:
    """WRF cuadjtq for one column and one level."""

    if kcall in (1, 2) and not ldflag:
        return

    def adjust_once(limit: str | None) -> float:
        tt = float(pt[kk])
        zqp = 1.0 / pp
        zqsat = min(0.5, _tlucua(tt) * zqp)
        zcor = 1.0 / (1.0 - VTMP_C1 * zqsat)
        zqsat = zqsat * zcor
        zcond = (pq[kk] - zqsat) / (1.0 + zqsat * zcor * _tlucub(tt))
        if limit == "positive":
            zcond = max(zcond, 0.0)
        elif limit == "negative":
            zcond = min(zcond, 0.0)
        pt[kk] = pt[kk] + _tlucuc(tt) * zcond
        pq[kk] = pq[kk] - zcond
        return float(zcond)

    if kcall == 1:
        zcond = adjust_once("positive")
        if zcond != 0.0:
            adjust_once(None)
    elif kcall == 2:
        zcond = adjust_once("negative")
        if zcond != 0.0:
            adjust_once(None)
    elif kcall in (0, 4):
        zcond = adjust_once(None)
        if zcond != 0.0 or kcall == 4:
            adjust_once(None)
    else:
        raise ValueError(f"unsupported cuadjtq kcall={kcall}")


def _cuini(klev: int, pten, pqen, pqsen, puen, pven, pverv, pgeo, paph):
    klevm1 = klev - 1
    pgeoh = _zeros(klev)
    ptenh = _zeros(klev)
    pqenh = _zeros(klev)
    pqsenh = _zeros(klev)
    ptu = _zeros(klev)
    pqu = _zeros(klev)
    ptd = _zeros(klev)
    pqd = _zeros(klev)
    puu = _zeros(klev)
    pvu = _zeros(klev)
    pud = _zeros(klev)
    pvd = _zeros(klev)
    pmfu = _zeros(klev)
    pmfd = _zeros(klev)
    pmfus = _zeros(klev)
    pmfds = _zeros(klev)
    pmfuq = _zeros(klev)
    pmfdq = _zeros(klev)
    pmful = _zeros(klev)
    pdmfup = _zeros(klev)
    pdmfdp = _zeros(klev)
    pdpmel = _zeros(klev)
    plu = _zeros(klev)
    plude = _zeros(klev)
    klab = _ints(klev)

    for jk in range(2, klev + 1):
        pgeoh[jk] = pgeo[jk] + (pgeo[jk - 1] - pgeo[jk]) * 0.5
        ptenh[jk] = (max(CPD * pten[jk - 1] + pgeo[jk - 1], CPD * pten[jk] + pgeo[jk]) - pgeoh[jk]) * RCPD
        pqsenh[jk] = pqsen[jk - 1]
        _cuadjtq(klev, jk, paph[jk], ptenh, pqsenh, True, 0)
        pqenh[jk] = min(pqen[jk - 1], pqsen[jk - 1]) + (pqsenh[jk] - pqsen[jk - 1])
        pqenh[jk] = max(pqenh[jk], 0.0)

    ptenh[klev] = (CPD * pten[klev] + pgeo[klev] - pgeoh[klev]) * RCPD
    pqenh[klev] = pqen[klev]
    ptenh[1] = pten[1]
    pqenh[1] = pqen[1]
    pgeoh[1] = pgeo[1]
    klwmin = klev
    zwmax = 0.0

    for jk in range(klevm1, 1, -1):
        zzs = max(CPD * ptenh[jk] + pgeoh[jk], CPD * ptenh[jk + 1] + pgeoh[jk + 1])
        ptenh[jk] = (zzs - pgeoh[jk]) * RCPD

    for jk in range(klev, 2, -1):
        if pverv[jk] < zwmax:
            zwmax = pverv[jk]
            klwmin = jk

    for jk in range(1, klev + 1):
        ik = jk - 1
        if jk == 1:
            ik = 1
        ptu[jk] = ptenh[jk]
        ptd[jk] = ptenh[jk]
        pqu[jk] = pqenh[jk]
        pqd[jk] = pqenh[jk]
        puu[jk] = puen[ik]
        pud[jk] = puen[ik]
        pvu[jk] = pven[ik]
        pvd[jk] = pven[ik]

    return {
        "pgeoh": pgeoh,
        "ptenh": ptenh,
        "pqenh": pqenh,
        "pqsenh": pqsenh,
        "ptu": ptu,
        "pqu": pqu,
        "ptd": ptd,
        "pqd": pqd,
        "puu": puu,
        "pvu": pvu,
        "pud": pud,
        "pvd": pvd,
        "pmfu": pmfu,
        "pmfd": pmfd,
        "pmfus": pmfus,
        "pmfds": pmfds,
        "pmfuq": pmfuq,
        "pmfdq": pmfdq,
        "pmful": pmful,
        "pdmfup": pdmfup,
        "pdmfdp": pdmfdp,
        "pdpmel": pdpmel,
        "plu": plu,
        "plude": plude,
        "klab": klab,
        "klwmin": klwmin,
    }


def _cubase(klev: int, ptenh, pqenh, pgeoh, paph, ptu, pqu, plu, puen, pven, puu, pvu):
    klevm1 = klev - 1
    klab = _ints(klev)
    zqold = _zeros(klev)
    klab[klev] = 1
    kcbot = klevm1
    ldcum = False
    ldbase = False
    puu[klev] = puen[klev] * (paph[klev + 1] - paph[klev])
    pvu[klev] = pven[klev] * (paph[klev + 1] - paph[klev])

    for jk in range(klevm1, 1, -1):
        loflag = klab[jk + 1] == 1 or (ldcum and kcbot == jk + 1)
        if not loflag:
            continue
        if LMFDUDV and not ldbase:
            puu[klev] += puen[jk] * (paph[jk + 1] - paph[jk])
            pvu[klev] += pven[jk] * (paph[jk + 1] - paph[jk])
        pqu[jk] = pqu[jk + 1]
        ptu[jk] = (CPD * ptu[jk + 1] + pgeoh[jk + 1] - pgeoh[jk]) * RCPD
        zqold[jk] = pqu[jk]
        _cuadjtq(klev, jk, paph[jk], ptu, pqu, True, 1)
        if pqu[jk] == zqold[jk]:
            zbuo = ptu[jk] * (1.0 + VTMP_C1 * pqu[jk]) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk]) + ZBUO0
            if zbuo > 0.0:
                klab[jk] = 1
        else:
            klab[jk] = 2
            plu[jk] += zqold[jk] - pqu[jk]
            zbuo = ptu[jk] * (1.0 + VTMP_C1 * pqu[jk]) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk]) + ZBUO0
            if zbuo > 0.0 and klab[jk + 1] == 1:
                kcbot = jk
                ldcum = True
                ldbase = True

    if LMFDUDV:
        if ldcum:
            zz = 1.0 / (paph[klev + 1] - paph[kcbot])
            puu[klev] *= zz
            pvu[klev] *= zz
        else:
            puu[klev] = puen[klevm1]
            pvu[klev] = pven[klevm1]
    return ldcum, kcbot, klab


def _cubasmc(klev: int, kk: int, pten, pqen, pqsen, puen, pven, pverv, pgeo, pgeoh,
             ldcum, ktype, klab, pmfu, pmfub, pentr, kcbot, ptu, pqu, plu,
             puu, pvu, pmfus, pmfuq, pmful, pdmfup, pmfuu, pmfuv):
    if (not ldcum) and klab[kk + 1] == 0 and pqen[kk] > 0.80 * pqsen[kk]:
        ptu[kk + 1] = (CPD * pten[kk] + pgeo[kk] - pgeoh[kk + 1]) * RCPD
        pqu[kk + 1] = pqen[kk]
        plu[kk + 1] = 0.0
        zzzmb = max(CMFCMIN, -pverv[kk] / G)
        zzzmb = min(zzzmb, CMFCMAX)
        pmfub = zzzmb
        pmfu[kk + 1] = pmfub
        pmfus[kk + 1] = pmfub * (CPD * ptu[kk + 1] + pgeoh[kk + 1])
        pmfuq[kk + 1] = pmfub * pqu[kk + 1]
        pmful[kk + 1] = 0.0
        pdmfup[kk + 1] = 0.0
        kcbot = kk
        klab[kk + 1] = 1
        ktype = 3
        pentr = ENTRMID
        if LMFDUDV:
            puu[kk + 1] = puen[kk]
            pvu[kk + 1] = pven[kk]
            pmfuu = pmfub * puu[kk + 1]
            pmfuv = pmfub * pvu[kk + 1]
    return ldcum, ktype, pmfub, pentr, kcbot, pmfuu, pmfuv


def _cuentr_new(klev: int, kk: int, ptenh, paph, pap, pgeoh, klwmin, ldcum, ktype,
                kcbot, kctop0, pmfu, pentr, zodetr, khmin):
    zpbase = paph[kcbot]
    zrrho = (RD * ptenh[kk + 1]) / paph[kk + 1]
    zdprho = (paph[kk + 1] - paph[kk]) * ZRG
    zpmid = 0.5 * (zpbase + paph[kctop0])
    zentr = pentr * pmfu[kk + 1] * zdprho * zrrho
    llo1 = kk < kcbot and ldcum
    zdmfde = zentr if llo1 else 0.0
    zdmfen = 0.0
    if llo1 and ktype == 2 and ((zpbase - paph[kk]) < ZDNOPRC or paph[kk] > zpmid):
        zdmfen = zentr
    iklwmin = max(klwmin, kctop0 + 2)
    if llo1 and ktype == 3 and (kk >= iklwmin or pap[kk] > zpmid):
        zdmfen = zentr
    if llo1 and ktype == 1:
        zdmfen = zentr
    zodetr[kk] = 0.0
    if llo1 and ktype == 1 and kk <= khmin and kk >= kctop0:
        ikt = kctop0
        ikh = khmin
        if ikh > ikt:
            zzmzk = -(pgeoh[ikh] - pgeoh[kk]) * ZRG
            ztmzk = -(pgeoh[ikh] - pgeoh[ikt]) * ZRG
            arg = 3.1415 * (zzmzk / ztmzk) * 0.5
            zorgde = np.tan(arg) * 3.1415 * 0.5 / ztmzk
            zdprho2 = (paph[kk + 1] - paph[kk]) * (ZRG * zrrho)
            zodetr[kk] = min(zorgde, 1.0e-3) * pmfu[kk + 1] * zdprho2
    return zpbase, zdmfen, zdmfde


def _cuasc_new(klev: int, ptenh, pqenh, puen, pven, pten, pqen, pqsen, pgeo, pgeoh,
               pap, paph, pqte, pverv, klwmin, ldcum, phcbase, ktype, klab,
               ptu, pqu, plu, puu, pvu, pmfu, pmfub, pentr, pmfus, pmfuq,
               pmful, plude, pdmfup, kcbot, kctop, kctop0, ztmst, khmin,
               phhatt, pqsenh):
    klevm1 = klev - 1
    zcons2 = 1.0 / (G * ztmst)
    zmfuu = 0.0
    zmfuv = 0.0
    zbuoy_acc = 0.0
    if not ldcum:
        ktype = 0
    zodetr = _zeros(klev)
    zoentr = _zeros(klev)

    for jk in range(1, klev + 1):
        plu[jk] = 0.0
        pmfu[jk] = 0.0
        pmfus[jk] = 0.0
        pmfuq[jk] = 0.0
        pmful[jk] = 0.0
        plude[jk] = 0.0
        pdmfup[jk] = 0.0
        zodetr[jk] = 0.0
        zoentr[jk] = 0.0
        if (not ldcum) or ktype == 3:
            klab[jk] = 0
        if (not ldcum) and paph[jk] < 4.0e4:
            kctop0 = jk

    kctop = klevm1
    if not ldcum:
        kcbot = klevm1
        pmfub = 0.0
        pqu[klev] = 0.0
    pmfu[klev] = pmfub
    pmfus[klev] = pmfub * (CPD * ptu[klev] + pgeoh[klev])
    pmfuq[klev] = pmfub * pqu[klev]
    if LMFDUDV:
        zmfuu = pmfub * puu[klev]
        zmfuv = pmfub * pvu[klev]

    if ktype == 1:
        ikb = kcbot
        zbuoy0 = G * ((ptu[ikb] - ptenh[ikb]) / ptenh[ikb] + 0.608 * (pqu[ikb] - pqenh[ikb]))
        if zbuoy0 > 0.0 and ikb > 1:
            zdz = (pgeo[ikb - 1] - pgeo[ikb]) * ZRG
            zdrodz = -np.log(pten[ikb - 1] / pten[ikb]) / zdz - G / (RD * ptenh[ikb])
            zoentr[ikb - 1] = zbuoy0 * 0.5 / (1.0 + zbuoy0 * zdz) + zdrodz
            zoentr[ikb - 1] = min(max(zoentr[ikb - 1], 0.0), 1.0e-3)

    for jk in range(klevm1, 1, -1):
        if LMFMID and jk < klevm1 and jk > klev - 13:
            ldcum, ktype, pmfub, pentr, kcbot, zmfuu, zmfuv = _cubasmc(
                klev, jk, pten, pqen, pqsen, puen, pven, pverv, pgeo, pgeoh,
                ldcum, ktype, klab, pmfu, pmfub, pentr, kcbot, ptu, pqu, plu,
                puu, pvu, pmfus, pmfuq, pmful, pdmfup, zmfuu, zmfuv
            )

        zqold = 0.0
        isum = int(klab[jk + 1])
        if klab[jk + 1] == 0:
            klab[jk] = 0
        loflag = klab[jk + 1] > 0
        if ktype == 3 and jk == kcbot:
            zmfmax = (paph[jk] - paph[jk - 1]) * zcons2
            if pmfub > zmfmax and pmfub != 0.0:
                zfac = zmfmax / pmfub
                pmfu[jk + 1] *= zfac
                pmfus[jk + 1] *= zfac
                pmfuq[jk + 1] *= zfac
                zmfuu *= zfac
                zmfuv *= zfac
                pmfub = zmfmax
        if isum == 0:
            continue

        zpbase, zdmfen, zdmfde = _cuentr_new(
            klev, jk, ptenh, paph, pap, pgeoh, klwmin, ldcum, ktype,
            kcbot, kctop0, pmfu, pentr, zodetr, khmin
        )
        if loflag:
            if jk < kcbot:
                zmftest = pmfu[jk + 1] + zdmfen - zdmfde
                zmfmax = min(zmftest, (paph[jk] - paph[jk - 1]) * zcons2)
                zdmfen = max(zdmfen - max(zmftest - zmfmax, 0.0), 0.0)
            zdmfde = min(zdmfde, 0.75 * pmfu[jk + 1])
            pmfu[jk] = pmfu[jk + 1] + zdmfen - zdmfde
            if jk < kcbot:
                zdprho = (pgeoh[jk] - pgeoh[jk + 1]) * ZRG
                zoentr[jk] = zoentr[jk] * zdprho * pmfu[jk + 1]
                zmftest = pmfu[jk] + zoentr[jk] - zodetr[jk]
                zmfmax = min(zmftest, (paph[jk] - paph[jk - 1]) * zcons2)
                zoentr[jk] = max(zoentr[jk] - max(zmftest - zmfmax, 0.0), 0.0)
            if ktype == 1 and jk < kcbot and jk <= khmin:
                zmse = CPD * ptu[jk + 1] + ALV * pqu[jk + 1] + pgeoh[jk + 1]
                ikt = kctop0
                znevn = (pgeoh[ikt] - pgeoh[jk + 1]) * (zmse - phhatt[jk + 1]) * ZRG
                if znevn <= 0.0:
                    znevn = 1.0
                zdprho = (pgeoh[jk] - pgeoh[jk + 1]) * ZRG
                zodmax = ((phcbase - zmse) / znevn) * zdprho * pmfu[jk + 1]
                zodetr[jk] = min(zodetr[jk], max(zodmax, 0.0))
            zodetr[jk] = min(zodetr[jk], 0.75 * pmfu[jk])
            pmfu[jk] = pmfu[jk] + zoentr[jk] - zodetr[jk]
            zqeen = pqenh[jk + 1] * zdmfen + pqenh[jk + 1] * zoentr[jk]
            zseen = (CPD * ptenh[jk + 1] + pgeoh[jk + 1]) * zdmfen
            zseen += (CPD * ptenh[jk + 1] + pgeoh[jk + 1]) * zoentr[jk]
            zscde = (CPD * ptu[jk + 1] + pgeoh[jk + 1]) * zdmfde
            zga = ALV * pqsenh[jk + 1] / (RV * (ptenh[jk + 1] ** 2))
            zdt = (plu[jk + 1] - 0.608 * (pqsenh[jk + 1] - pqenh[jk + 1])) / (
                1.0 / ptenh[jk + 1] + 0.608 * zga
            )
            zscod = CPD * ptenh[jk + 1] + pgeoh[jk + 1] + CPD * zdt
            zscde += zodetr[jk] * zscod
            zqude = pqu[jk + 1] * zdmfde
            zqcod = pqsenh[jk + 1] + zga * zdt
            zqude += zodetr[jk] * zqcod
            plude[jk] = plu[jk + 1] * zdmfde + plu[jk + 1] * zodetr[jk]
            zmfusk = pmfus[jk + 1] + zseen - zscde
            zmfuqk = pmfuq[jk + 1] + zqeen - zqude
            zmfulk = pmful[jk + 1] - plude[jk]
            denom = max(CMFCMIN, pmfu[jk])
            plu[jk] = zmfulk / denom
            pqu[jk] = zmfuqk / denom
            ptu[jk] = (zmfusk / denom - pgeoh[jk]) * RCPD
            ptu[jk] = min(max(ptu[jk], 100.0), 400.0)
            zqold = pqu[jk]

        _cuadjtq(klev, jk, paph[jk], ptu, pqu, loflag, 1)
        if loflag and pqu[jk] != zqold:
            klab[jk] = 2
            plu[jk] += zqold - pqu[jk]
            zbuo = ptu[jk] * (1.0 + VTMP_C1 * pqu[jk] - plu[jk]) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk])
            if klab[jk + 1] == 1:
                zbuo += ZBUO0
            if zbuo > 0.0 and pmfu[jk] > 0.01 * pmfub and jk >= kctop0:
                kctop = jk
                ldcum = True
                zprcon = CPRCON if (zpbase - paph[jk]) >= ZDNOPRC else 0.0
                zlnew = plu[jk] / (1.0 + zprcon * (pgeoh[jk] - pgeoh[jk + 1]))
                pdmfup[jk] = max(0.0, (plu[jk] - zlnew) * pmfu[jk])
                plu[jk] = zlnew
            else:
                klab[jk] = 0
                pmfu[jk] = 0.0
        if loflag:
            pmful[jk] = plu[jk] * pmfu[jk]
            pmfus[jk] = (CPD * ptu[jk] + pgeoh[jk]) * pmfu[jk]
            pmfuq[jk] = pqu[jk] * pmfu[jk]

        if LMFDUDV:
            zdmfen_m = zdmfen + zoentr[jk]
            zdmfde_m = zdmfde + zodetr[jk]
            if loflag:
                if ktype in (1, 3):
                    zz = 3.0 if zdmfen_m <= 1.0e-20 else 2.0
                else:
                    zz = 1.0 if zdmfen_m <= 1.0e-20 else 0.0
                zdmfeu = zdmfen_m + zz * zdmfde_m
                zdmfdu = min(zdmfde_m + zz * zdmfde_m, 0.75 * pmfu[jk + 1])
                zmfuu += zdmfeu * puen[jk] - zdmfdu * puu[jk + 1]
                zmfuv += zdmfeu * pven[jk] - zdmfdu * pvu[jk + 1]
                if pmfu[jk] > 0.0:
                    puu[jk] = zmfuu / pmfu[jk]
                    pvu[jk] = zmfuv / pmfu[jk]

        if loflag and ktype == 1 and jk > 1:
            zbuoyz = G * ((ptu[jk] - ptenh[jk]) / ptenh[jk] + 0.608 * (pqu[jk] - pqenh[jk]) - plu[jk])
            zbuoyz = max(zbuoyz, 0.0)
            zdz = (pgeo[jk - 1] - pgeo[jk]) * ZRG
            zdrodz = -np.log(pten[jk - 1] / pten[jk]) / zdz - G / (RD * ptenh[jk])
            zbuoy_acc += zbuoyz * zdz
            zoentr[jk - 1] = zbuoyz * 0.5 / (1.0 + zbuoy_acc) + zdrodz
            zoentr[jk - 1] = min(max(zoentr[jk - 1], 0.0), 1.0e-3)

    if kctop == klevm1:
        ldcum = False
    kcbot = max(kcbot, kctop)
    kcum = 1 if ldcum else 0
    if kcum == 0:
        return ldcum, ktype, kcbot, kctop, kctop0, kcum

    jk = kctop - 1
    if jk >= 1:
        zdmfde = (1.0 - CMFCTOP) * pmfu[jk + 1]
        plude[jk] = zdmfde * plu[jk + 1]
        pmfu[jk] = pmfu[jk + 1] - zdmfde
        pmfus[jk] = (CPD * ptu[jk] + pgeoh[jk]) * pmfu[jk]
        pmfuq[jk] = pqu[jk] * pmfu[jk]
        pmful[jk] = plu[jk] * pmfu[jk]
        plude[jk - 1] = pmful[jk]
        pdmfup[jk] = 0.0
        if LMFDUDV:
            puu[jk] = puu[jk + 1]
            pvu[jk] = pvu[jk + 1]
    return ldcum, ktype, kcbot, kctop, kctop0, kcum


def _cudlfs(klev: int, ptenh, pqenh, puen, pven, pgeoh, paph, ptu, pqu, puu, pvu,
            ldcum, kcbot, kctop, pmfub, prfl, ptd, pqd, pud, pvd, pmfd, pmfds,
            pmfdq, pdmfdp):
    kdtop = klev + 1
    lddraf = False
    if not LMFDD:
        return kdtop, lddraf
    for jk in range(3, klev - 2 + 1):
        llo2 = ldcum and prfl > 0.0 and (not lddraf) and (jk < kcbot and jk > kctop)
        if not llo2:
            continue
        ztenwb = ptenh.copy()
        zqenwb = pqenh.copy()
        _cuadjtq(klev, jk, paph[jk], ztenwb, zqenwb, True, 2)
        zttest = 0.5 * (ptu[jk] + ztenwb[jk])
        zqtest = 0.5 * (pqu[jk] + zqenwb[jk])
        zbuo = zttest * (1.0 + VTMP_C1 * zqtest) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk])
        zcond = pqenh[jk] - zqenwb[jk]
        zmftop = -CMFDEPS * pmfub
        if zbuo < 0.0 and prfl > 10.0 * zmftop * zcond:
            kdtop = jk
            lddraf = True
            ptd[jk] = zttest
            pqd[jk] = zqtest
            pmfd[jk] = zmftop
            pmfds[jk] = pmfd[jk] * (CPD * ptd[jk] + pgeoh[jk])
            pmfdq[jk] = pmfd[jk] * pqd[jk]
            pdmfdp[jk - 1] = -0.5 * pmfd[jk] * zcond
            prfl += pdmfdp[jk - 1]
        if LMFDUDV and pmfd[jk] < 0.0:
            pud[jk] = 0.5 * (puu[jk] + puen[jk - 1])
            pvd[jk] = 0.5 * (pvu[jk] + pven[jk - 1])
    return kdtop, lddraf


def _cuddraf(klev: int, ptenh, pqenh, puen, pven, pgeoh, paph, prfl, lddraf,
             ptd, pqd, pud, pvd, pmfd, pmfds, pmfdq, pdmfdp):
    for jk in range(3, klev + 1):
        llo2 = lddraf and pmfd[jk - 1] < 0.0
        if not llo2:
            continue
        zentr = ENTRDD * pmfd[jk - 1] * RD * ptenh[jk - 1] / (G * paph[jk - 1]) * (paph[jk] - paph[jk - 1])
        zdmfen = zentr
        zdmfde = zentr
        itopde = klev - 2
        if jk > itopde:
            zdmfen = 0.0
            zdmfde = pmfd[itopde] * (paph[jk] - paph[jk - 1]) / (paph[klev + 1] - paph[itopde])
        pmfd[jk] = pmfd[jk - 1] + zdmfen - zdmfde
        zseen = (CPD * ptenh[jk - 1] + pgeoh[jk - 1]) * zdmfen
        zqeen = pqenh[jk - 1] * zdmfen
        zsdde = (CPD * ptd[jk - 1] + pgeoh[jk - 1]) * zdmfde
        zqdde = pqd[jk - 1] * zdmfde
        zmfdsk = pmfds[jk - 1] + zseen - zsdde
        zmfdqk = pmfdq[jk - 1] + zqeen - zqdde
        denom = min(-CMFCMIN, pmfd[jk])
        pqd[jk] = zmfdqk / denom
        ptd[jk] = (zmfdsk / denom - pgeoh[jk]) * RCPD
        ptd[jk] = min(max(ptd[jk], 100.0), 400.0)
        zcond = pqd[jk]
        _cuadjtq(klev, jk, paph[jk], ptd, pqd, True, 2)
        zcond = zcond - pqd[jk]
        zbuo = ptd[jk] * (1.0 + VTMP_C1 * pqd[jk]) - ptenh[jk] * (1.0 + VTMP_C1 * pqenh[jk])
        if zbuo >= 0.0 or prfl <= (pmfd[jk] * zcond):
            pmfd[jk] = 0.0
        pmfds[jk] = (CPD * ptd[jk] + pgeoh[jk]) * pmfd[jk]
        pmfdq[jk] = pqd[jk] * pmfd[jk]
        pdmfdp[jk - 1] = -pmfd[jk] * zcond
        prfl += pdmfdp[jk - 1]
        if LMFDUDV and pmfd[jk] < 0.0:
            zmfduk = pmfd[jk - 1] * pud[jk - 1] + zdmfen * puen[jk - 1] - zdmfde * pud[jk - 1]
            zmfdvk = pmfd[jk - 1] * pvd[jk - 1] + zdmfen * pven[jk - 1] - zdmfde * pvd[jk - 1]
            pud[jk] = zmfduk / min(-CMFCMIN, pmfd[jk])
            pvd[jk] = zmfdvk / min(-CMFCMIN, pmfd[jk])
    return prfl


def _cuflx(klev: int, pqen, pqsen, ptenh, pqenh, paph, pgeoh, kcbot, kctop,
           kdtop, ktype, lddraf, ldcum, pmfu, pmfd, pmfus, pmfds, pmfuq, pmfdq,
           pmful, plude, pdmfup, pdmfdp, pten, pdpmel, ztmst, sig1):
    zcons1 = CPD / (ALF * G * ztmst)
    zcons2 = 1.0 / (G * ztmst)
    zcucov = 0.05
    ztmelp2 = TMELT + 2.0
    prfl = 0.0
    psfl = 0.0
    prain = 0.0
    if (not LMFSCV) and ktype == 2:
        ldcum = False
        lddraf = False
    itop = kctop
    if (not ldcum) or kdtop < kctop:
        lddraf = False
    if not ldcum:
        ktype = 0
    ktopm2 = itop - 2

    for jk in range(ktopm2, klev + 1):
        if ldcum and jk >= kctop - 1:
            pmfus[jk] = pmfus[jk] - pmfu[jk] * (CPD * ptenh[jk] + pgeoh[jk])
            pmfuq[jk] = pmfuq[jk] - pmfu[jk] * pqenh[jk]
            if lddraf and jk >= kdtop:
                pmfds[jk] = pmfds[jk] - pmfd[jk] * (CPD * ptenh[jk] + pgeoh[jk])
                pmfdq[jk] = pmfdq[jk] - pmfd[jk] * pqenh[jk]
            else:
                pmfd[jk] = 0.0
                pmfds[jk] = 0.0
                pmfdq[jk] = 0.0
                pdmfdp[jk - 1] = 0.0
        else:
            pmfu[jk] = 0.0
            pmfd[jk] = 0.0
            pmfus[jk] = 0.0
            pmfds[jk] = 0.0
            pmfuq[jk] = 0.0
            pmfdq[jk] = 0.0
            pmful[jk] = 0.0
            pdmfup[jk - 1] = 0.0
            pdmfdp[jk - 1] = 0.0
            plude[jk - 1] = 0.0

    for jk in range(ktopm2, klev + 1):
        if ldcum and jk > kcbot:
            zzp = (paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[kcbot])
            if ktype == 3:
                zzp = zzp ** 2
            pmfu[jk] = pmfu[kcbot] * zzp
            pmfus[jk] = pmfus[kcbot] * zzp
            pmfuq[jk] = pmfuq[kcbot] * zzp
            pmful[jk] = pmful[kcbot] * zzp
        if ldcum:
            prain += pdmfup[jk]
            if pten[jk] > TMELT:
                prfl += pdmfup[jk] + pdmfdp[jk]
                if psfl > 0.0 and pten[jk] > ztmelp2:
                    zfac = zcons1 * (paph[jk + 1] - paph[jk])
                    zsnmlt = min(psfl, zfac * (pten[jk] - ztmelp2))
                    pdpmel[jk] = zsnmlt
                    psfl -= zsnmlt
                    prfl += zsnmlt
            else:
                psfl += pdmfup[jk] + pdmfdp[jk]

    prfl = max(prfl, 0.0)
    psfl = max(psfl, 0.0)
    zpsubcl = prfl + psfl
    for jk in range(ktopm2, klev + 1):
        if ldcum and jk >= kcbot and zpsubcl > 1.0e-20:
            zrfl = zpsubcl
            cevapcu = CEVAPCU1 * np.sqrt(CEVAPCU2 * np.sqrt(sig1[jk]))
            zrnew = (max(0.0, np.sqrt(zrfl / zcucov) - cevapcu * (paph[jk + 1] - paph[jk]) * max(0.0, pqsen[jk] - pqen[jk]))) ** 2 * zcucov
            zrmin = zrfl - zcucov * max(0.0, 0.8 * pqsen[jk] - pqen[jk]) * zcons2 * (paph[jk + 1] - paph[jk])
            zrnew = max(zrnew, zrmin)
            zrfln = max(zrnew, 0.0)
            zdrfl = min(0.0, zrfln - zrfl)
            pdmfup[jk] += zdrfl
            zpsubcl = zrfln
    zdpevap = zpsubcl - (prfl + psfl)
    denom = max(1.0e-20, prfl + psfl)
    prfl = prfl + zdpevap * prfl / denom
    psfl = psfl + zdpevap * psfl / denom
    return ktopm2, ktype, lddraf, ldcum, prfl, psfl, prain


def _cudtdq(klev: int, ktopm2: int, paph, ldcum, pten, ptte, pqte, pmfus, pmfds,
            pmfuq, pmfdq, pmful, pdmfup, pdmfdp, ztmst, pdpmel, prain, prfl,
            psfl, pqen, pqsen, plude, pcte):
    zdiagw = ztmst / 1000.0
    zsheat = 0.0
    zmelt = 0.0
    prsfc = 0.0
    pssfc = 0.0
    paprc = 0.0
    paprsm = 0.0
    paprs = 0.0
    if not ldcum:
        return prsfc, pssfc
    for jk in range(ktopm2, klev + 1):
        zalv = ALV if pten[jk] > TMELT else ALS
        rhk = min(1.0, pqen[jk] / pqsen[jk])
        rhcoe = max(0.0, (rhk - RHC) / (RHM - RHC))
        pldfd = max(0.0, rhcoe * FDBK * plude[jk])
        if jk < klev:
            zdtdt = (G / (paph[jk + 1] - paph[jk])) * RCPD * (
                pmfus[jk + 1] - pmfus[jk] + pmfds[jk + 1] - pmfds[jk]
                - ALF * pdpmel[jk] - zalv * (pmful[jk + 1] - pmful[jk] - pldfd - (pdmfup[jk] + pdmfdp[jk]))
            )
            zdqdt = (G / (paph[jk + 1] - paph[jk])) * (
                pmfuq[jk + 1] - pmfuq[jk] + pmfdq[jk + 1] - pmfdq[jk]
                + pmful[jk + 1] - pmful[jk] - pldfd - (pdmfup[jk] + pdmfdp[jk])
            )
        else:
            zdtdt = -(G / (paph[jk + 1] - paph[jk])) * RCPD * (
                pmfus[jk] + pmfds[jk] + ALF * pdpmel[jk]
                - zalv * (pmful[jk] + pdmfup[jk] + pdmfdp[jk] + pldfd)
            )
            zdqdt = -(G / (paph[jk + 1] - paph[jk])) * (
                pmfuq[jk] + pmfdq[jk] + pldfd + (pmful[jk] + pdmfup[jk] + pdmfdp[jk])
            )
        ptte[jk] += zdtdt
        pqte[jk] += zdqdt
        pcte[jk] = (G / (paph[jk + 1] - paph[jk])) * pldfd
        zsheat += zalv * (pdmfup[jk] + pdmfdp[jk])
        zmelt += pdpmel[jk]
    prsfc = prfl
    pssfc = psfl
    paprc += zdiagw * (prfl + psfl)
    paprs = paprsm + zdiagw * psfl
    _ = (paprc, paprs, zsheat, prain, zmelt)
    return prsfc, pssfc


def _cududv(klev: int, ktopm2: int, ktype, kcbot, paph, ldcum, puen, pven,
            pvom, pvol, puu, pud, pvu, pvd, pmfu, pmfd):
    if not ldcum:
        return
    zmfuu = _zeros(klev)
    zmfdu = _zeros(klev)
    zmfuv = _zeros(klev)
    zmfdv = _zeros(klev)
    for jk in range(ktopm2, klev + 1):
        ik = jk - 1
        zmfuu[jk] = pmfu[jk] * (puu[jk] - puen[ik])
        zmfuv[jk] = pmfu[jk] * (pvu[jk] - pven[ik])
        zmfdu[jk] = pmfd[jk] * (pud[jk] - puen[ik])
        zmfdv[jk] = pmfd[jk] * (pvd[jk] - pven[ik])
    for jk in range(ktopm2, klev + 1):
        if jk > kcbot:
            zzp = (paph[klev + 1] - paph[jk]) / (paph[klev + 1] - paph[kcbot])
            if ktype == 3:
                zzp = zzp ** 2
            zmfuu[jk] = zmfuu[kcbot] * zzp
            zmfuv[jk] = zmfuv[kcbot] * zzp
            zmfdu[jk] = zmfdu[kcbot] * zzp
            zmfdv[jk] = zmfdv[kcbot] * zzp
    for jk in range(ktopm2, klev + 1):
        if jk < klev:
            zdudt = (G / (paph[jk + 1] - paph[jk])) * (zmfuu[jk + 1] - zmfuu[jk] + zmfdu[jk + 1] - zmfdu[jk])
            zdvdt = (G / (paph[jk + 1] - paph[jk])) * (zmfuv[jk + 1] - zmfuv[jk] + zmfdv[jk + 1] - zmfdv[jk])
        else:
            zdudt = -(G / (paph[jk + 1] - paph[jk])) * (zmfuu[jk] + zmfdu[jk])
            zdvdt = -(G / (paph[jk + 1] - paph[jk])) * (zmfuv[jk] + zmfdv[jk])
        pvom[jk] += zdudt
        pvol[jk] += zdvdt


def _cumastr_new(s: _TieState):
    klev = int(s.pt.shape[0] - 2)
    klevm1 = klev - 1
    ztmst = s.dt
    pqhfl = s.evap

    ptte = _zeros(klev)
    pcte = _zeros(klev)
    pvom = _zeros(klev)
    pvol = _zeros(klev)
    ztp1 = s.pt.copy()
    zqp1 = _zeros(klev)
    pum1 = s.pu.copy()
    pvm1 = s.pv.copy()
    pverv = s.pomg.copy()
    pgeo = _zeros(klev)
    pqsen = _zeros(klev)
    pqte = _zeros(klev)
    zqq = _zeros(klev)
    for k in range(1, klev + 1):
        zqp1[k] = s.pqv[k] / (1.0 + s.pqv[k])
        pgeo[k] = G * s.poz[k]
        pqsen[k] = min(0.5, _tlucua(ztp1[k]) / s.pap[k])
        pqsen[k] = pqsen[k] / (1.0 - VTMP_C1 * pqsen[k])
        pqte[k] = s.pqvf[k] + s.pqvbl[k]
        zqq[k] = pqte[k]

    st = _cuini(klev, ztp1, zqp1, pqsen, pum1, pvm1, pverv, pgeo, s.paph)
    pgeoh = st["pgeoh"]; ztenh = st["ptenh"]; zqenh = st["pqenh"]; zqsenh = st["pqsenh"]
    ptu = st["ptu"]; pqu = st["pqu"]; ztd = st["ptd"]; zqd = st["pqd"]
    zuu = st["puu"]; zvu = st["pvu"]; zud = st["pud"]; zvd = st["pvd"]
    pmfu = st["pmfu"]; pmfd = st["pmfd"]; zmfus = st["pmfus"]; zmfds = st["pmfds"]
    zmfuq = st["pmfuq"]; zmfdq = st["pmfdq"]; zmful = st["pmful"]
    zdmfup = st["pdmfup"]; zdmfdp = st["pdmfdp"]
    zdpmel = st["pdpmel"]; zlu = st["plu"]; zlude = st["plude"]; ilab = st["klab"]
    ilwmin = st["klwmin"]

    ldcum, kcbot, ilab = _cubase(klev, ztenh, zqenh, pgeoh, s.paph, ptu, pqu, zlu, pum1, pvm1, zuu, zvu)
    zdqcv = pqte[1] * (s.paph[2] - s.paph[1])
    zdqpbl = 0.0
    idtop = klev + 1
    for jk in range(2, klev + 1):
        zdqcv += pqte[jk] * (s.paph[jk + 1] - s.paph[jk])
        if jk >= kcbot:
            zdqpbl += pqte[jk] * (s.paph[jk + 1] - s.paph[jk])
    ktype = 1 if zdqcv > max(0.0, 1.1 * pqhfl * G) else 2
    ikb = kcbot
    zqumqe = pqu[ikb] + zlu[ikb] - zqenh[ikb]
    zdqmin = max(0.01 * zqenh[ikb], 1.0e-10)
    zcons2 = 1.0 / (G * ztmst)
    if zdqpbl > 0.0 and zqumqe > zdqmin and ldcum:
        zmfub = zdqpbl / (G * max(zqumqe, zdqmin))
    else:
        zmfub = 0.01
        ldcum = False
    zmfmax = (s.paph[ikb] - s.paph[ikb - 1]) * zcons2
    zmfub = min(zmfub, zmfmax)

    zhcbase = CPD * ptu[ikb] + pgeoh[ikb] + ALV * pqu[ikb]
    ictop0 = kcbot - 1
    zalvdcp = ALV / CPD
    zqalv = 1.0 / ALV
    zhhatt = _zeros(klev)
    for jk in range(klevm1, 2, -1):
        zhsat = CPD * ztenh[jk] + pgeoh[jk] + ALV * zqsenh[jk]
        zgam = C5LES * zalvdcp * zqsenh[jk] / ((1.0 - VTMP_C1 * zqsenh[jk]) * (ztenh[jk] - C4LES) ** 2)
        zzz = CPD * ztenh[jk] * 0.608
        zhhat = zhsat - (zzz + zgam * zzz) / (1.0 + zgam * zzz * zqalv) * max(zqsenh[jk] - zqenh[jk], 0.0)
        zhhatt[jk] = zhhat
        if jk < ictop0 and zhcbase > zhhat:
            ictop0 = jk
    jk = kcbot
    zhsat = CPD * ztenh[jk] + pgeoh[jk] + ALV * zqsenh[jk]
    zgam = C5LES * zalvdcp * zqsenh[jk] / ((1.0 - VTMP_C1 * zqsenh[jk]) * (ztenh[jk] - C4LES) ** 2)
    zzz = CPD * ztenh[jk] * 0.608
    zhhatt[jk] = zhsat - (zzz + zgam * zzz) / (1.0 + zgam * zzz * zqalv) * max(zqsenh[jk] - zqenh[jk], 0.0)

    if ldcum and ktype == 1:
        ihmin = kcbot
    else:
        ihmin = -1
    zhmin = 0.0
    zbi = 1.0 / (25.0 * G)
    for jk in range(klev, 0, -1):
        llo1 = ldcum and ktype == 1 and ihmin == kcbot
        if llo1 and jk < kcbot and jk >= ictop0:
            zro = RD * ztenh[jk] / (G * s.paph[jk])
            zdz = (s.paph[jk] - s.paph[jk - 1]) * zro
            zdhdz = (
                CPD * (ztp1[jk - 1] - ztp1[jk])
                + ALV * (zqp1[jk - 1] - zqp1[jk])
                + (pgeo[jk - 1] - pgeo[jk])
            ) * G / (pgeo[jk - 1] - pgeo[jk])
            zdepth = pgeoh[jk] - pgeoh[ikb]
            zfac = np.sqrt(1.0 + zdepth * zbi)
            zhmin += zdhdz * zfac * zdz
            zrh = -ALV * (zqsenh[jk] - zqenh[jk]) * zfac
            if zhmin > zrh:
                ihmin = jk
    if ldcum and ktype == 1 and ihmin < ictop0:
        ihmin = ictop0
    zentr = ENTRPEN if ktype == 1 else ENTRSCV
    if s.lndj == 1:
        zentr *= 1.1

    kctop = klevm1
    ldcum, ktype, kcbot, kctop, ictop0, icum = _cuasc_new(
        klev, ztenh, zqenh, pum1, pvm1, ztp1, zqp1, pqsen, pgeo, pgeoh,
        s.pap, s.paph, pqte, pverv, ilwmin, ldcum, zhcbase, ktype, ilab,
        ptu, pqu, zlu, zuu, zvu, pmfu, zmfub, zentr, zmfus, zmfuq, zmful,
        zlude, zdmfup, kcbot, kctop, ictop0, ztmst, ihmin, zhhatt, zqsenh
    )
    if icum == 0:
        return _finish_tiecnv(klev, s, ztp1, zqp1, zqq, ptte, pqte, pcte, pvom, pvol, 0.0, 0, ztmst)

    zpbmpt = s.paph[kcbot] - s.paph[kctop]
    if ldcum:
        ictop0 = kctop
    if ldcum and ktype == 1 and zpbmpt < ZDNOPRC:
        ktype = 2
    if ktype == 2:
        zentr = ENTRSCV * (1.1 if s.lndj == 1 else 1.0)
    zrfl = float(np.sum(zdmfup[1:klev + 1]))

    if LMFDD:
        idtop, loddraf = _cudlfs(
            klev, ztenh, zqenh, pum1, pvm1, pgeoh, s.paph, ptu, pqu, zuu, zvu,
            ldcum, kcbot, kctop, zmfub, zrfl, ztd, zqd, zud, zvd, pmfd, zmfds,
            zmfdq, zdmfdp
        )
        zrfl = _cuddraf(klev, ztenh, zqenh, pum1, pvm1, pgeoh, s.paph, zrfl, loddraf,
                        ztd, zqd, zud, zvd, pmfd, zmfds, zmfdq, zdmfdp)
    else:
        loddraf = False

    zheat = 0.0
    zcape = 0.0
    zrelh = 0.0
    zmfub1 = zmfub
    if ldcum and ktype == 1:
        ktop0 = max(12, kctop)
        ikb = kcbot
        for jk in range(2, klev + 1):
            if jk <= kcbot and jk > kctop:
                zro = s.paph[jk] / (RD * ztenh[jk])
                zdz = (s.paph[jk] - s.paph[jk - 1]) / (G * zro)
                zheat += (
                    ((ztp1[jk - 1] - ztp1[jk] + G * zdz / CPD) / ztenh[jk]
                     + 0.608 * (zqp1[jk - 1] - zqp1[jk]))
                    * (pmfu[jk] + pmfd[jk]) * G / zro
                )
                zcape += G * (
                    (ptu[jk] * (1.0 + 0.608 * pqu[jk] - zlu[jk]))
                    / (ztenh[jk] * (1.0 + 0.608 * zqenh[jk])) - 1.0
                ) * zdz
            if jk <= kcbot and jk > ktop0:
                dept = (s.paph[jk + 1] - s.paph[jk]) / (s.paph[ikb + 1] - s.paph[ktop0 + 1])
                zrelh += dept * zqp1[jk] / pqsen[jk]
        if zrelh >= CRIRH and zheat != 0.0:
            zht = max(0.0, zcape) / (ZTAU * zheat)
            zmfub1 = max(zmfub * zht, 0.01)
            zmfmax = (s.paph[ikb] - s.paph[ikb - 1]) * zcons2
            zmfub1 = min(zmfub1, zmfmax)
        else:
            zmfub1 = 0.01
            zmfub = 0.01
            ldcum = False

    if ktype != 1:
        ikb = kcbot
        zeps = CMFDEPS if pmfd[ikb] < 0.0 and loddraf else 0.0
        zqumqe = pqu[ikb] + zlu[ikb] - zeps * zqd[ikb] - (1.0 - zeps) * zqenh[ikb]
        zdqmin = max(0.01 * zqenh[ikb], 1.0e-10)
        zmfmax = (s.paph[ikb] - s.paph[ikb - 1]) * zcons2
        if zdqpbl > 0.0 and zqumqe > zdqmin and ldcum and zmfub < zmfmax:
            zmfub1 = zdqpbl / (G * max(zqumqe, zdqmin))
        else:
            zmfub1 = zmfub
        llo1 = (ktype == 2) and abs(zmfub1 - zmfub) < 0.2 * zmfub
        if not llo1:
            zmfub1 = zmfub
        zmfub1 = min(zmfub1, zmfmax)

    for jk in range(1, klev + 1):
        if ldcum:
            zfac = zmfub1 / max(zmfub, 1.0e-10)
            pmfd[jk] *= zfac
            zmfds[jk] *= zfac
            zmfdq[jk] *= zfac
            zdmfdp[jk] *= zfac
        else:
            pmfd[jk] = 0.0
            zmfds[jk] = 0.0
            zmfdq[jk] = 0.0
            zdmfdp[jk] = 0.0
    zmfub = zmfub1 if ldcum else 0.0

    ldcum, ktype, kcbot, kctop, ictop0, icum = _cuasc_new(
        klev, ztenh, zqenh, pum1, pvm1, ztp1, zqp1, pqsen, pgeo, pgeoh,
        s.pap, s.paph, pqte, pverv, ilwmin, ldcum, zhcbase, ktype, ilab,
        ptu, pqu, zlu, zuu, zvu, pmfu, zmfub, zentr, zmfus, zmfuq, zmful,
        zlude, zdmfup, kcbot, kctop, ictop0, ztmst, ihmin, zhhatt, zqsenh
    )
    ktopm2, ktype, loddraf, ldcum, prfl, psfl, prain = _cuflx(
        klev, zqp1, pqsen, ztenh, zqenh, s.paph, pgeoh, kcbot, kctop, idtop,
        ktype, loddraf, ldcum, pmfu, pmfd, zmfus, zmfds, zmfuq, zmfdq, zmful,
        zlude, zdmfup, zdmfdp, ztp1, zdpmel, ztmst, s.sig1
    )
    prsfc, pssfc = _cudtdq(
        klev, ktopm2, s.paph, ldcum, ztp1, ptte, pqte, zmfus, zmfds, zmfuq,
        zmfdq, zmful, zdmfup, zdmfdp, ztmst, zdpmel, prain, prfl, psfl, zqp1,
        pqsen, zlude, pcte
    )
    if LMFDUDV:
        _cududv(klev, ktopm2, ktype, kcbot, s.paph, ldcum, pum1, pvm1, pvom, pvol,
                zuu, zud, zvu, zvd, pmfu, pmfd)
    zprecc = max(0.0, (prsfc + pssfc) * ztmst)
    return _finish_tiecnv(klev, s, ztp1, zqp1, zqq, ptte, pqte, pcte, pvom, pvol, zprecc, ktype, ztmst)


def _finish_tiecnv(klev: int, s: _TieState, ztp1, zqp1, zqq, ptte, pqte, pcte, pvom, pvol, zprecc, ktype, ztmst):
    pqc = s.pqc.copy()
    pqi = s.pqi.copy()
    if FDBK >= 1.0e-9:
        for k in range(1, klev + 1):
            if pcte[k] > 0.0:
                ztpp1 = s.pt[k] + ptte[k] * ztmst
                if ztpp1 >= T000:
                    fliq = 1.0
                    zalf = 0.0
                elif ztpp1 <= HGFR:
                    fliq = 0.0
                    zalf = ALF
                else:
                    ztc = ztpp1 - T000
                    fliq = 0.0059 + 0.9941 * np.exp(-0.003102 * ztc * ztc)
                    zalf = ALF
                fice = 1.0 - fliq
                pqc[k] += fliq * pcte[k] * ztmst
                pqi[k] += fice * pcte[k] * ztmst
                ptte[k] -= zalf * RCPD * fliq * pcte[k]
    pt = s.pt.copy()
    pqv = s.pqv.copy()
    pu = s.pu.copy()
    pv = s.pv.copy()
    for k in range(1, klev + 1):
        pt[k] = ztp1[k] + ptte[k] * ztmst
        zqp1[k] = zqp1[k] + (pqte[k] - zqq[k]) * ztmst
        pqv[k] = zqp1[k] / (1.0 - zqp1[k])
        if LMFDUDV:
            pu[k] += pvom[k] * ztmst
            pv[k] += pvol[k] * ztmst
    return {
        "pu": pu,
        "pv": pv,
        "pt": pt,
        "pqv": pqv,
        "pqc": pqc,
        "pqi": pqi,
        "zprecc": float(zprecc),
        "ktype": int(ktype),
    }


def _prepare_tie_state(T, QV, QC, QI, P, P8W, DZ, RHO, U, V, W, QVFTEN, QVPBLTEN, QFX, XLAND, ZNU, dt):
    T = np.asarray(T, dtype=np.float64)
    klev = int(T.shape[0])
    pu = _zeros(klev); pv = _zeros(klev); pt = _zeros(klev); pqv = _zeros(klev)
    pqc = _zeros(klev); pqi = _zeros(klev); pqvf = _zeros(klev); pqvbl = _zeros(klev)
    poz = _zeros(klev); pomg = _zeros(klev); pap = _zeros(klev); paph = _zeros(klev)
    sig1 = _zeros(klev)
    zi = np.zeros(klev + 1, dtype=np.float64)
    zl = np.zeros(klev, dtype=np.float64)
    for k in range(1, klev):
        zi[k] = zi[k - 1] + DZ[k - 1]
    for k in range(1, klev):
        zl[k - 1] = 0.5 * (zi[k] + zi[k - 1])
    zl[klev - 1] = 2.0 * zi[klev - 1] - zl[klev - 2]
    for kb in range(1, klev + 1):
        kt = klev + 1 - kb
        pu[kt] = U[kb - 1]
        pv[kt] = V[kb - 1]
        pt[kt] = T[kb - 1]
        pqv[kt] = QV[kb - 1]
        pqc[kt] = QC[kb - 1]
        pqi[kt] = QI[kb - 1]
        pqvf[kt] = QVFTEN[kb - 1]
        pqvbl[kt] = QVPBLTEN[kb - 1]
        pomg[kt] = -0.5 * G * RHO[kb - 1] * (W[kb - 1] + W[kb])
        poz[kt] = zl[kb - 1]
        pap[kt] = P[kb - 1]
        sig1[kt] = ZNU[kb - 1]
    for kb in range(1, klev + 2):
        kt = klev + 2 - kb
        paph[kt] = P8W[kb - 1]
    lndj = int(abs(float(XLAND) - 2.0))
    return _TieState(pu, pv, pt, pqv, pqc, pqi, pqvf, pqvbl, poz, pomg, pap, paph, float(QFX), lndj, sig1, float(dt))


def tiedtke_column(
    T,
    QV,
    QC,
    QI,
    P,
    P8W,
    DZ,
    RHO,
    PI,
    U,
    V,
    W,
    QVFTEN,
    QVPBLTEN,
    QFX,
    XLAND,
    ZNU,
    dt,
    *,
    stepcu=5,
):
    """Run one WRF modified-Tiedtke column in savepoint orientation."""

    klev = int(np.asarray(T).shape[0])
    delt = float(dt) * int(stepcu)
    rdelt = 1.0 / delt
    s = _prepare_tie_state(T, QV, QC, QI, P, P8W, DZ, RHO, U, V, W, QVFTEN, QVPBLTEN, QFX, XLAND, ZNU, delt)
    out = _cumastr_new(s)
    rth = np.zeros(klev, dtype=np.float64)
    rqv = np.zeros(klev, dtype=np.float64)
    rqc = np.zeros(klev, dtype=np.float64)
    rqi = np.zeros(klev, dtype=np.float64)
    ru = np.zeros(klev, dtype=np.float64)
    rv = np.zeros(klev, dtype=np.float64)
    T0 = np.asarray(T, dtype=np.float64)
    QV0 = np.asarray(QV, dtype=np.float64)
    QC0 = np.asarray(QC, dtype=np.float64)
    QI0 = np.asarray(QI, dtype=np.float64)
    U0 = np.asarray(U, dtype=np.float64)
    V0 = np.asarray(V, dtype=np.float64)
    PI0 = np.asarray(PI, dtype=np.float64)
    for kb in range(1, klev + 1):
        kt = klev + 1 - kb
        rth[kb - 1] = (out["pt"][kt] - T0[kb - 1]) / PI0[kb - 1] * rdelt
        rqv[kb - 1] = (out["pqv"][kt] - QV0[kb - 1]) * rdelt
        rqc[kb - 1] = (out["pqc"][kt] - QC0[kb - 1]) * rdelt
        rqi[kb - 1] = (out["pqi"][kt] - QI0[kb - 1]) * rdelt
        ru[kb - 1] = (out["pu"][kt] - U0[kb - 1]) * rdelt
        rv[kb - 1] = (out["pv"][kt] - V0[kb - 1]) * rdelt
    raincv = out["zprecc"] / float(stepcu)
    pratec = out["zprecc"] / (float(stepcu) * float(dt))
    zeros = np.zeros_like(rth)
    return {
        "RTHCUTEN": jnp.asarray(rth),
        "RQVCUTEN": jnp.asarray(rqv),
        "RQCCUTEN": jnp.asarray(rqc),
        "RQRCUTEN": jnp.asarray(zeros),
        "RQICUTEN": jnp.asarray(rqi),
        "RQSCUTEN": jnp.asarray(zeros),
        "RUCUTEN": jnp.asarray(ru),
        "RVCUTEN": jnp.asarray(rv),
        "RAINCV": jnp.asarray(raincv),
        "PRATEC": jnp.asarray(pratec),
        "KTYPE": jnp.asarray(out["ktype"], dtype=jnp.int32),
    }


def step_tiedtke_column(
    T,
    QV,
    QC,
    QI,
    P,
    P8W,
    DZ,
    RHO,
    PI,
    U,
    V,
    W,
    QVFTEN,
    QVPBLTEN,
    QFX,
    XLAND,
    ZNU,
    dt,
    *,
    stepcu=5,
) -> PhysicsStepResult:
    """Return the frozen v0.6.0 physics interface payload for one column."""

    out = tiedtke_column(T, QV, QC, QI, P, P8W, DZ, RHO, PI, U, V, W, QVFTEN, QVPBLTEN, QFX, XLAND, ZNU, dt, stepcu=stepcu)
    state_tendencies = {
        "theta": out["RTHCUTEN"],
        "qv": out["RQVCUTEN"],
        "qc": out["RQCCUTEN"],
        "qr": out["RQRCUTEN"],
        "qi": out["RQICUTEN"],
        "qs": out["RQSCUTEN"],
    }
    diagnostics = {
        "rthcuten": out["RTHCUTEN"],
        "rqvcuten": out["RQVCUTEN"],
        "rqccuten": out["RQCCUTEN"],
        "rqrcuten": out["RQRCUTEN"],
        "rqicuten": out["RQICUTEN"],
        "rqscuten": out["RQSCUTEN"],
        "rucuten": out["RUCUTEN"],
        "rvcuten": out["RVCUTEN"],
        "raincv": out["RAINCV"],
        "pratec": out["PRATEC"],
        "ktype": out["KTYPE"],
    }
    tendency = PhysicsTendency(
        state_tendencies=state_tendencies,
        accumulator_increments={"rainc_acc": out["RAINCV"]},
    )
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus={}),
        diagnostics=PhysicsDiagnostics(cumulus=diagnostics),
    )


__all__ = ["tiedtke_column", "step_tiedtke_column"]
