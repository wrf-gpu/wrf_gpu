"""NumPy reference transcription of WRF KF_eta_PARA (cu_physics=1).

This module is the CORRECTNESS ANCHOR for the JAX port: it mirrors the Fortran
`KF_eta_PARA` control flow line-for-line using native Python control flow
(if / while / for / early-return) so it is unambiguously faithful and easy to
diff against the source. It is validated against the independent Fortran oracle
(proofs/p0_4/savepoints/*.json) to predeclared tolerances.

The JAX production port (cumulus_kf.py) reproduces this same algorithm with
lax/where control flow for GPU residency; both are checked against the oracle.

Indexing convention: Fortran arrays are 1..KX. Here we use 1-based numpy arrays
of length KX+1 (index 0 unused) so the transcription matches the source indices
EXACTLY. All math is float64 (operational release precision); the Fortran
oracle is REAL*4, so parity is to a predeclared physical tolerance, not bitwise.
"""
from __future__ import annotations

import math

import numpy as np

from gpuwrf.physics import cumulus_kf_tables as _T

# DATA-statement constants
P00 = 1.0e5
T00 = 273.16
RLF = 3.339e5
RHIC = 1.0
RHBC = 0.90
PIE = 3.141592654
TTFRZ = 268.16
TBFRZ = 248.16
C5 = 1.0723e-3
RATE = 0.03


# ----------------------- helper subroutines (1:1) ---------------------------
def _interp(p, thes):
    plutop, rdpr, rdthk = _T.PLUTOP, _T.RDPR, _T.RDTHK
    the0k, ttab, qstab = _T.THE0K, _T.TTAB, _T.QSTAB
    tp = (p - plutop) * rdpr
    qq = tp - math.floor(tp)
    iptb = int(tp) + 1                      # 1-based
    bth = (the0k[iptb] - the0k[iptb - 1]) * qq + the0k[iptb - 1]  # the0k(iptb+1),the0k(iptb)
    tth = (thes - bth) * rdthk
    pp = tth - math.floor(tth)
    ithtb = int(tth) + 1
    # ttab(ithtb,iptb) 1-based -> [ithtb-1, iptb-1]
    t00 = ttab[ithtb - 1, iptb - 1]
    t10 = ttab[ithtb, iptb - 1]
    t01 = ttab[ithtb - 1, iptb]
    t11 = ttab[ithtb, iptb]
    q00 = qstab[ithtb - 1, iptb - 1]
    q10 = qstab[ithtb, iptb - 1]
    q01 = qstab[ithtb - 1, iptb]
    q11 = qstab[ithtb, iptb]
    temp = t00 + (t10 - t00) * pp + (t01 - t00) * qq + (t00 - t10 - t01 + t11) * pp * qq
    qs = q00 + (q10 - q00) * pp + (q01 - q00) * qq + (q00 - q10 - q01 + q11) * pp * qq
    return temp, qs


def tpmix2(p, thes, tu, qu, qliq, qice, xlv1, xlv0):
    temp, qs = _interp(p, thes)
    dq = qs - qu
    qnewlq = 0.0
    if dq <= 0.0:
        qnew = qu - qs
        qu = qs
        qnewlq = qnew
    else:
        qnewlq = 0.0
        qtot = qliq + qice
        if qtot >= dq:
            qliq = qliq - dq * qliq / (qtot + 1.0e-10)
            qice = qice - dq * qice / (qtot + 1.0e-10)
            qu = qs
        else:
            rll = xlv0 - xlv1 * temp
            cpp = 1004.5 * (1.0 + 0.89 * qu)
            if qtot < 1.0e-10:
                temp = temp + rll * (dq / (1.0 + dq)) / cpp
            else:
                temp = temp + rll * ((dq - qtot) / (1.0 + dq - qtot)) / cpp
                qu = qu + qtot
                qliq = 0.0
                qice = 0.0
    tu = temp
    return tu, qu, qliq, qice, qnewlq, 0.0


def tpmix2dd(p, thes):
    ts, qs = _interp(p, thes)
    return ts, qs


def envirtht(p1, t1, q1, aliq, bliq, cliq, dliq):
    astrt = 1.0e-3
    ainc = 0.075
    alu = _T.ALU
    c1, c2 = 3374.6525, 2.5403
    t00_, p00_ = 273.16, 1.0e5
    ee = q1 * p1 / (0.622 + q1)
    a1 = ee / aliq
    tp = (a1 - astrt) / ainc
    indlu = int(tp) + 1                     # 1-based
    value = (indlu - 1) * ainc + astrt
    aintrp = (a1 - value) / ainc
    tlog = aintrp * alu[indlu] + (1.0 - aintrp) * alu[indlu - 1]   # alu(indlu+1),alu(indlu)
    tdpt = (cliq - dliq * tlog) / (bliq - tlog)
    tsat = tdpt - (0.212 + 1.571e-3 * (tdpt - t00_) - 4.36e-4 * (t1 - t00_)) * (t1 - tdpt)
    tht = t1 * (p00_ / p1) ** (0.2854 * (1.0 - 0.28 * q1))
    return tht * math.exp((c1 / tsat - c2) * q1 * (1.0 + 0.81 * q1))


def dtfrznew(tu, p, thteu, qu, qfrz, qice, aliq, bliq, cliq, dliq):
    rlc = 2.5e6 - 2369.276 * (tu - 273.16)
    rls = 2833922.0 - 259.532 * (tu - 273.16)
    rlf = rls - rlc
    cpp = 1004.5 * (1.0 + 0.89 * qu)
    a = (cliq - bliq * dliq) / ((tu - dliq) * (tu - dliq))
    dtfrz = rlf * qfrz / (cpp + rls * qu * a)
    tu = tu + dtfrz
    es = aliq * math.exp((bliq * tu - cliq) / (tu - dliq))
    qs = es * 0.622 / (p - es)
    dqevap = qs - qu
    qice = qice - dqevap
    qu = qu + dqevap
    pii = (1.0e5 / p) ** (0.2854 * (1.0 - 0.28 * qu))
    thteu = tu * pii * math.exp((3374.6525 / tu - 2.5403) * qu * (1.0 + 0.81 * qu))
    return tu, thteu, qu, qice


def condload(qliq, qice, wtw, dz, boterm, enterm, rate, qnewlq, qnewic, g):
    qtot = qliq + qice
    qnew = qnewlq + qnewic
    qest = 0.5 * (qtot + qnew)
    g1 = wtw + boterm - enterm - 2.0 * g * dz * qest / 1.5
    if g1 < 0.0:
        g1 = 0.0
    wavg = 0.5 * (math.sqrt(wtw) + math.sqrt(g1))
    conv = rate * dz / wavg
    ratio3 = qnewlq / (qnew + 1.0e-8)
    qtot = qtot + 0.6 * qnew
    oldq = qtot
    ratio4 = (0.6 * qnewlq + qliq) / (qtot + 1.0e-8)
    qtot = qtot * math.exp(-conv)
    dq = oldq - qtot
    qlqout = ratio4 * dq
    qicout = (1.0 - ratio4) * dq
    pptdrg = 0.5 * (oldq + qtot - 0.2 * qnew)
    wtw = wtw + boterm - enterm - 2.0 * g * dz * pptdrg / 1.5
    if abs(wtw) < 1.0e-4:
        wtw = 1.0e-4
    qliq = ratio4 * qtot + ratio3 * 0.4 * qnew
    qice = (1.0 - ratio4) * qtot + (1.0 - ratio3) * 0.4 * qnew
    return qliq, qice, wtw, qlqout, qicout, 0.0, 0.0


def prof5(eq):
    sqrt2p = 2.506628
    a1, a2, a3 = 0.4361836, -0.1201676, 0.9372980
    p, sigma, fe = 0.33267, 0.166666667, 0.202765151
    y = 6.0 * eq - 3.0
    ey = math.exp(y * y / (-2.0))
    e45 = math.exp(-4.5)
    t2 = 1.0 / (1.0 + p * abs(y))
    t1 = 0.500498
    c1 = a1 * t1 + a2 * t1 * t1 + a3 * t1 * t1 * t1
    c2 = a1 * t2 + a2 * t2 * t2 + a3 * t2 * t2 * t2
    if y >= 0.0:
        ee = sigma * (0.5 * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * eq * eq / 2.0
        ud = sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
    else:
        ee = sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * eq * eq / 2.0
        ud = sigma * (0.5 * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
    return ee / fe, ud / fe


# WRF model constants (passed in by the driver in WRF; fixed here for d01)
XLV0 = 3.15e6
XLV1 = 2370.0
XLS0 = 2.905e6
XLS1 = 259.532
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
CP = 1004.5
R_D = 287.0
G = 9.81
EP2 = R_D / 461.6


def _empty_result(kx, nca):
    z = np.zeros(kx, dtype=np.float64)
    return dict(DTDT=z.copy(), DQDT=z.copy(), DQCDT=z.copy(), DQRDT=z.copy(),
                DQIDT=z.copy(), DQSDT=z.copy(), RTHCUTEN=z.copy(), RQVCUTEN=z.copy(),
                RQCCUTEN=z.copy(), RQRCUTEN=z.copy(), RQICUTEN=z.copy(), RQSCUTEN=z.copy(),
                RAINCV=0.0, PRATEC=0.0, NCA=nca, CUTOP=1.0, CUBOT=float(kx + 1),
                ISHALL=2, TIMEC=0.0)


def kf_eta_para_np(t0, qv0, p0, dzq, rhoe, w0avg1d, u0, v0, dt, dx, pi_exner=None,
                   cudt=0.0, warm_rain=False, f_qi=True, f_qs=True, trigger=1):
    """Faithful flat NumPy transcription of KF_eta_PARA for ONE column.

    Inputs 0-based length KX, bottom-up. pi_exner (optional, length KX) is the
    Exner function used by the DRIVER to convert DTDT->RTHCUTEN (=DTDT/pi); if
    None it is computed as (P0/1e5)**(R/CP).
    Returns a dict (see _empty_result for keys)."""
    KX = len(t0)
    KL = KX
    KXp = KX + 3
    DXSQ = dx * dx
    aliq = SVP1 * 1000.0
    bliq = SVP2
    cliq = SVP2 * SVPT0
    dliq = SVP3
    gdry = -G / CP
    NIC = 0

    def z1():
        return np.zeros(KXp, dtype=np.float64)

    # ----- load 1-based input arrays -----
    T0 = z1(); QV0 = z1(); P0 = z1(); DZQ = z1(); RHOE = z1(); W0A = z1()
    U0 = z1(); V0 = z1()
    for k in range(1, KX + 1):
        T0[k] = t0[k-1]; QV0[k] = qv0[k-1]; P0[k] = p0[k-1]
        DZQ[k] = dzq[k-1]; RHOE[k] = rhoe[k-1]; W0A[k] = w0avg1d[k-1]
        U0[k] = u0[k-1]; V0[k] = v0[k-1]

    Q0 = z1(); QES = z1(); QL0 = z1(); QI0 = z1(); QR0 = z1(); QS0 = z1()
    RH = z1(); DILFRC = z1(); TV0 = z1(); DP = z1(); TKE = z1(); CLDHGT = z1()
    Z0 = z1(); DZA = z1()
    DTDT = z1(); DQDT = z1(); DQCDT = z1(); DQRDT = z1(); DQIDT = z1(); DQSDT = z1()

    NCA = -100.0
    ISHALL = 0
    FBFRC = 0.0
    DPMIN = 5.0e3
    ML = 0

    L5 = 1; LLFC = 1
    for k in range(1, KX + 1):
        es = aliq * math.exp((bliq * T0[k] - cliq) / (T0[k] - dliq))
        QES[k] = 0.622 * es / (P0[k] - es)
        Q0[k] = min(QES[k], QV0[k]); Q0[k] = max(0.000001, Q0[k])
        RH[k] = Q0[k] / QES[k]
        DILFRC[k] = 1.0
        TV0[k] = T0[k] * (1.0 + 0.608 * Q0[k])
        DP[k] = RHOE[k] * G * DZQ[k]
        TKE[k] = 0.0; CLDHGT[k] = 0.0
    P300 = P0[1] - 30000.0
    for k in range(1, KX + 1):
        if P0[k] >= 0.5 * P0[1]:
            L5 = k
        if P0[k] >= P300:
            LLFC = k

    Z0[1] = 0.5 * DZQ[1]
    for k in range(2, KL + 1):
        Z0[k] = Z0[k-1] + 0.5 * (DZQ[k] + DZQ[k-1])
        DZA[k-1] = Z0[k] - Z0[k-1]
    DZA[KL] = 0.0

    NCHECK = 1
    KCHECK = [0] * (KX + 2)
    KCHECK[1] = 1
    PM15 = P0[1] - 15.0e2
    for k in range(2, LLFC + 1):
        if P0[k] < PM15:
            NCHECK += 1
            KCHECK[NCHECK] = k
            PM15 = PM15 - 15.0e2

    # working arrays
    TU = z1(); TVU = z1(); QU = z1(); TZ = z1(); QD = z1()
    WU = z1(); WD = z1(); EMS = z1(); EMSD = z1()
    UMF = z1(); UER = z1(); UDR = z1(); DMF = z1(); DER = z1(); DDR = z1()
    UMF2 = z1(); UER2 = z1(); UDR2 = z1(); DMF2 = z1(); DER2 = z1(); DDR2 = z1()
    THTA0 = z1(); THETEE = z1(); THTAU = z1(); THETEU = z1(); THTAD = z1(); THETED = z1()
    QLIQ = z1(); QICE = z1(); QLQOUT = z1(); QICOUT = z1(); PPTLIQ = z1(); PPTICE = z1()
    DETLQ = z1(); DETIC = z1(); DETLQ2 = z1(); DETIC2 = z1(); RATIO2 = z1()
    DOMGDP = z1(); EXN = z1(); TVQU = z1(); EQFRC = z1(); WSPD = z1(); QDT = z1()
    FXM = z1(); THTAG = z1(); THPA = z1(); THFXOUT = z1(); THFXIN = z1()
    QPA = z1(); QFXOUT = z1(); QFXIN = z1(); QLPA = z1(); QLFXIN = z1(); QLFXOUT = z1()
    QIPA = z1(); QIFXIN = z1(); QIFXOUT = z1(); QRPA = z1(); QRFXIN = z1(); QRFXOUT = z1()
    QSPA = z1(); QSFXIN = z1(); QSFXOUT = z1()
    QLG = z1(); QIG = z1(); QRG = z1(); QSG = z1(); TG = z1(); TVG = z1(); QG = z1()
    QSD = z1(); DDILFRC = z1(); TGU = z1(); QGU = z1(); THTEEG = z1()
    OMG = np.zeros(KX + 3, dtype=np.float64)
    RAINFB = z1(); SNOWFB = z1()

    NU = 0; NUCHM = 0; NCHM = 0
    # convection-result scalars
    LC = K = KLCL = KPBL = LET = LTOP = LCL = 0
    ZLCL = TLCL = TVLCL = TVEN = VMFLCL = WLCL = RAD = ABE = TRPPT = 0.0
    DPTHMX = AU0 = ZMIX = TMIX = QMIX = PMIX = WKL = PLCL = 0.0
    IFLAG = 0
    converged = False

    # ============ USL outer loop (mirrors `usl: DO`) ============
    while True:
        NU += 1
        if NU > NCHECK:
            if ISHALL == 1:
                CHMAX = 0.0; NCHM = 0; NUCHM = 0
                for nk in range(1, NCHECK + 1):
                    nnn = KCHECK[nk]
                    if CLDHGT[nnn] > CHMAX:
                        NCHM = nnn; NUCHM = nk; CHMAX = CLDHGT[nnn]
                NU = NUCHM - 1
                FBFRC = 1.0
                continue
            else:
                ISHALL = 2
                return _empty_result(KX, NCA)
        KMIX = KCHECK[NU]
        LOW = KMIX
        LC = LOW
        NLAYRS = 0; DPTHMX = 0.0
        nk = LC - 1
        while True:
            nk += 1
            if nk > KX:
                break
            DPTHMX += DP[nk]; NLAYRS += 1
            if DPTHMX > DPMIN:
                break
        if DPTHMX < DPMIN:
            ISHALL = 2
            return _empty_result(KX, NCA)
        KPBL = LC + NLAYRS - 1
        TMIX = 0.0; QMIX = 0.0; ZMIX = 0.0; PMIX = 0.0
        for nk in range(LC, KPBL + 1):
            TMIX += DP[nk] * T0[nk]; QMIX += DP[nk] * Q0[nk]
            ZMIX += DP[nk] * Z0[nk]; PMIX += DP[nk] * P0[nk]
        TMIX /= DPTHMX; QMIX /= DPTHMX; ZMIX /= DPTHMX; PMIX /= DPTHMX
        EMIX = QMIX * PMIX / (0.622 + QMIX)
        astrt = 1.0e-3; ainc = 0.075
        a1 = EMIX / aliq
        tp = (a1 - astrt) / ainc
        indlu = int(tp) + 1
        value = (indlu - 1) * ainc + astrt
        aintrp = (a1 - value) / ainc
        tlog = aintrp * _T.ALU[indlu] + (1.0 - aintrp) * _T.ALU[indlu - 1]
        TDPT = (cliq - dliq * tlog) / (bliq - tlog)
        TLCL = TDPT - (0.212 + 1.571e-3 * (TDPT - T00) - 4.36e-4 * (TMIX - T00)) * (TMIX - TDPT)
        TLCL = min(TLCL, TMIX)
        TVLCL = TLCL * (1.0 + 0.608 * QMIX)
        ZLCL = ZMIX + (TLCL - TMIX) / gdry
        KLCL = LC
        for nk in range(LC, KL + 1):
            KLCL = nk
            if ZLCL <= Z0[nk]:
                break
        if ZLCL > Z0[KL]:
            ISHALL = 2
            return _empty_result(KX, NCA)
        K = KLCL - 1
        DLP = (ZLCL - Z0[K]) / (Z0[KLCL] - Z0[K])
        TENV = T0[K] + (T0[KLCL] - T0[K]) * DLP
        QENV = Q0[K] + (Q0[KLCL] - Q0[K]) * DLP
        TVEN = TENV * (1.0 + 0.608 * QENV)
        WKLCL = 0.02 * ZLCL / 2.0e3 if ZLCL < 2.0e3 else 0.02
        WKL = (W0A[K] + (W0A[KLCL] - W0A[K]) * DLP) * dx / 25.0e3 - WKLCL
        DTLCL = 0.0 if WKL < 0.0001 else 4.64 * WKL ** 0.33
        DTRH = 0.0
        if TLCL + DTLCL + DTRH < TENV:
            continue  # not buoyant -> next USL
        # ---- buoyant: compute updraft ----
        THETEU[K] = envirtht(PMIX, TMIX, QMIX, aliq, bliq, cliq, dliq)
        DTTOT = DTLCL + DTRH
        if DTTOT > 1.0e-4:
            GDT = 2.0 * G * DTTOT * 500.0 / TVEN
            WLCL = min(1.0 + 0.5 * math.sqrt(GDT), 3.0)
        else:
            WLCL = 1.0
        PLCL = P0[K] + (P0[KLCL] - P0[K]) * DLP
        WTW = WLCL * WLCL
        TVLCL = TLCL * (1.0 + 0.608 * QMIX)
        RHOLCL = PLCL / (R_D * TVLCL)
        LCL = KLCL
        LET = LCL
        if WKL < 0.0:
            RAD = 1000.0
        elif WKL > 0.1:
            RAD = 2000.0
        else:
            RAD = 1000.0 + 1000.0 * WKL / 0.1
        WU[K] = WLCL
        AU0 = 0.01 * DXSQ
        UMF[K] = RHOLCL * AU0
        VMFLCL = UMF[K]; UPOLD = VMFLCL; UPNEW = UPOLD
        RATIO2[K] = 0.0; UER[K] = 0.0; ABE = 0.0; TRPPT = 0.0
        TU[K] = TLCL; TVU[K] = TVLCL; QU[K] = QMIX; EQFRC[K] = 1.0
        QLIQ[K] = 0.0; QICE[K] = 0.0; QLQOUT[K] = 0.0; QICOUT[K] = 0.0
        DETLQ[K] = 0.0; DETIC[K] = 0.0; PPTLIQ[K] = 0.0; PPTICE[K] = 0.0
        IFLAG = 0; TTEMP = TTFRZ
        EE1 = 1.0; UD1 = 0.0; REI = 0.0; DILBE = 0.0
        LTOP = K
        nk = K
        broke = False
        while nk <= KL - 1:
            NK1 = nk + 1
            RATIO2[NK1] = RATIO2[nk]
            FRC1 = 0.0
            TU[NK1] = T0[NK1]; THETEU[NK1] = THETEU[nk]; QU[NK1] = QU[nk]
            QLIQ[NK1] = QLIQ[nk]; QICE[NK1] = QICE[nk]
            TU[NK1], QU[NK1], QLIQ[NK1], QICE[NK1], qnewlq, qnewic = tpmix2(
                P0[NK1], THETEU[NK1], TU[NK1], QU[NK1], QLIQ[NK1], QICE[NK1], XLV1, XLV0)
            if TU[NK1] <= TTFRZ:
                if TU[NK1] > TBFRZ:
                    if TTEMP > TTFRZ:
                        TTEMP = TTFRZ
                    FRC1 = (TTEMP - TU[NK1]) / (TTEMP - TBFRZ)
                else:
                    FRC1 = 1.0; IFLAG = 1
                TTEMP = TU[NK1]
                QFRZ = (QLIQ[NK1] + qnewlq) * FRC1
                qnewic = qnewic + qnewlq * FRC1
                qnewlq = qnewlq - qnewlq * FRC1
                QICE[NK1] = QICE[NK1] + QLIQ[NK1] * FRC1
                QLIQ[NK1] = QLIQ[NK1] - QLIQ[NK1] * FRC1
                TU[NK1], THETEU[NK1], QU[NK1], QICE[NK1] = dtfrznew(
                    TU[NK1], P0[NK1], THETEU[NK1], QU[NK1], QFRZ, QICE[NK1], aliq, bliq, cliq, dliq)
            TVU[NK1] = TU[NK1] * (1.0 + 0.608 * QU[NK1])
            if nk == K:
                BE = (TVLCL + TVU[NK1]) / (TVEN + TV0[NK1]) - 1.0
                BOTERM = 2.0 * (Z0[NK1] - ZLCL) * G * BE / 1.5
                DZZ = Z0[NK1] - ZLCL
            else:
                BE = (TVU[nk] + TVU[NK1]) / (TV0[nk] + TV0[NK1]) - 1.0
                BOTERM = 2.0 * DZA[nk] * G * BE / 1.5
                DZZ = DZA[nk]
            ENTERM = 2.0 * REI * WTW / UPOLD
            (QLIQ[NK1], QICE[NK1], WTW, QLQOUT[NK1], QICOUT[NK1], qnewlq, qnewic) = condload(
                QLIQ[NK1], QICE[NK1], WTW, DZZ, BOTERM, ENTERM, RATE, qnewlq, qnewic, G)
            if WTW < 1.0e-3:
                LTOP = nk; broke = True
                break
            else:
                WU[NK1] = math.sqrt(WTW)
            THETEE[NK1] = envirtht(P0[NK1], T0[NK1], Q0[NK1], aliq, bliq, cliq, dliq)
            REI = VMFLCL * DP[NK1] * 0.03 / RAD
            TVQU[NK1] = TU[NK1] * (1.0 + 0.608 * QU[NK1] - QLIQ[NK1] - QICE[NK1])
            if nk == K:
                DILBE = ((TVLCL + TVQU[NK1]) / (TVEN + TV0[NK1]) - 1.0) * DZZ
            else:
                DILBE = ((TVQU[nk] + TVQU[NK1]) / (TV0[nk] + TV0[NK1]) - 1.0) * DZZ
            if DILBE > 0.0:
                ABE += DILBE * G
            if TVQU[NK1] <= TV0[NK1]:
                EE2 = 0.5; UD2 = 1.0; EQFRC[NK1] = 0.0
            else:
                LET = NK1
                TTMP = TVQU[NK1]
                F1 = 0.95; F2 = 1.0 - F1
                THTTMP = F1 * THETEE[NK1] + F2 * THETEU[NK1]
                QTMP = F1 * Q0[NK1] + F2 * QU[NK1]
                TMPLIQ = F2 * QLIQ[NK1]; TMPICE = F2 * QICE[NK1]
                TTMP, QTMP, TMPLIQ, TMPICE, _a, _b = tpmix2(
                    P0[NK1], THTTMP, TTMP, QTMP, TMPLIQ, TMPICE, XLV1, XLV0)
                TU95 = TTMP * (1.0 + 0.608 * QTMP - TMPLIQ - TMPICE)
                if TU95 > TV0[NK1]:
                    EE2 = 1.0; UD2 = 0.0; EQFRC[NK1] = 1.0
                else:
                    F1 = 0.10; F2 = 1.0 - F1
                    THTTMP = F1 * THETEE[NK1] + F2 * THETEU[NK1]
                    QTMP = F1 * Q0[NK1] + F2 * QU[NK1]
                    TMPLIQ = F2 * QLIQ[NK1]; TMPICE = F2 * QICE[NK1]
                    TTMP, QTMP, TMPLIQ, TMPICE, _a, _b = tpmix2(
                        P0[NK1], THTTMP, TTMP, QTMP, TMPLIQ, TMPICE, XLV1, XLV0)
                    TU10 = TTMP * (1.0 + 0.608 * QTMP - TMPLIQ - TMPICE)
                    TVDIFF = abs(TU10 - TVQU[NK1])
                    if TVDIFF < 1.0e-3:
                        EE2 = 1.0; UD2 = 0.0; EQFRC[NK1] = 1.0
                    else:
                        EQFRC[NK1] = (TV0[NK1] - TVQU[NK1]) * F1 / (TU10 - TVQU[NK1])
                        EQFRC[NK1] = min(1.0, max(0.0, EQFRC[NK1]))
                        if EQFRC[NK1] == 1.0:
                            EE2 = 1.0; UD2 = 0.0
                        elif EQFRC[NK1] == 0.0:
                            EE2 = 0.0; UD2 = 1.0
                        else:
                            EE2, UD2 = prof5(EQFRC[NK1])
            EE2 = max(EE2, 0.5)
            UD2 = 1.5 * UD2
            UER[NK1] = 0.5 * REI * (EE1 + EE2)
            UDR[NK1] = 0.5 * REI * (UD1 + UD2)
            if UMF[nk] - UDR[NK1] < 10.0:
                if DILBE > 0.0:
                    ABE = ABE - DILBE * G
                LET = nk; LTOP = nk; broke = True
                break
            else:
                EE1 = EE2; UD1 = UD2
                UPOLD = UMF[nk] - UDR[NK1]
                UPNEW = UPOLD + UER[NK1]
                UMF[NK1] = UPNEW
                DILFRC[NK1] = UPNEW / UPOLD
                DETLQ[NK1] = QLIQ[NK1] * UDR[NK1]
                DETIC[NK1] = QICE[NK1] * UDR[NK1]
                QDT[NK1] = QU[NK1]
                QU[NK1] = (UPOLD * QU[NK1] + UER[NK1] * Q0[NK1]) / UPNEW
                THETEU[NK1] = (THETEU[NK1] * UPOLD + THETEE[NK1] * UER[NK1]) / UPNEW
                QLIQ[NK1] = QLIQ[NK1] * UPOLD / UPNEW
                QICE[NK1] = QICE[NK1] * UPOLD / UPNEW
                PPTLIQ[NK1] = QLQOUT[NK1] * UMF[nk]
                PPTICE[NK1] = QICOUT[NK1] * UMF[nk]
                TRPPT = TRPPT + PPTLIQ[NK1] + PPTICE[NK1]
                if NK1 <= KPBL:
                    UER[NK1] = UER[NK1] + VMFLCL * DP[NK1] / DPTHMX
            nk += 1
        if not broke:
            LTOP = KL
        CLDHGT[LC] = Z0[LTOP] - ZLCL
        if TLCL > 293.0:
            CHMIN = 4.0e3
        elif TLCL <= 293.0 and TLCL >= 273.0:
            CHMIN = 2.0e3 + 100.0 * (TLCL - 273.0)
        else:
            CHMIN = 2.0e3
        if LTOP <= KLCL or LTOP <= KPBL or LET + 1 <= KPBL:
            CLDHGT[LC] = 0.0
            for nk2 in range(K, LTOP + 1):
                UMF[nk2] = 0.0; UDR[nk2] = 0.0; UER[nk2] = 0.0
                DETLQ[nk2] = 0.0; DETIC[nk2] = 0.0; PPTLIQ[nk2] = 0.0; PPTICE[nk2] = 0.0
            continue
        elif CLDHGT[LC] > CHMIN and ABE > 1.0:
            ISHALL = 0
            converged = True
            break
        else:
            ISHALL = 1
            if NU == NUCHM:
                converged = True
                break
            else:
                for nk2 in range(K, LTOP + 1):
                    UMF[nk2] = 0.0; UDR[nk2] = 0.0; UER[nk2] = 0.0
                    DETLQ[nk2] = 0.0; DETIC[nk2] = 0.0; PPTLIQ[nk2] = 0.0; PPTICE[nk2] = 0.0
                continue
    # ============ end USL loop ============
    if not converged:
        ISHALL = 2
        return _empty_result(KX, NCA)

    # ---- post-USL setup (Fortran 1421+) ----
    if ISHALL == 1:
        KSTART = max(KPBL, KLCL)
        LET = KSTART

    if LET == LTOP:
        UDR[LTOP] = UMF[LTOP] + UDR[LTOP] - UER[LTOP]
        DETLQ[LTOP] = QLIQ[LTOP] * UDR[LTOP] * UPNEW / UPOLD
        DETIC[LTOP] = QICE[LTOP] * UDR[LTOP] * UPNEW / UPOLD
        UER[LTOP] = 0.0
        UMF[LTOP] = 0.0
    else:
        DPTT = 0.0
        for nj in range(LET + 1, LTOP + 1):
            DPTT += DP[nj]
        DUMFDP = UMF[LET] / DPTT
        for nk in range(LET + 1, LTOP + 1):
            if nk == LTOP:
                UDR[nk] = UMF[nk - 1]
                UER[nk] = 0.0
                DETLQ[nk] = UDR[nk] * QLIQ[nk] * DILFRC[nk]
                DETIC[nk] = UDR[nk] * QICE[nk] * DILFRC[nk]
            else:
                UMF[nk] = UMF[nk - 1] - DP[nk] * DUMFDP
                UER[nk] = UMF[nk] * (1.0 - 1.0 / DILFRC[nk])
                UDR[nk] = UMF[nk - 1] - UMF[nk] + UER[nk]
                DETLQ[nk] = UDR[nk] * QLIQ[nk] * DILFRC[nk]
                DETIC[nk] = UDR[nk] * QICE[nk] * DILFRC[nk]
            if nk >= LET + 2:
                TRPPT = TRPPT - PPTLIQ[nk] - PPTICE[nk]
                PPTLIQ[nk] = UMF[nk - 1] * QLQOUT[nk]
                PPTICE[nk] = UMF[nk - 1] * QICOUT[nk]
                TRPPT = TRPPT + PPTLIQ[nk] + PPTICE[nk]

    # init below cloud base / above cloud top
    ML = 0
    for nk in range(1, LTOP + 1):
        if T0[nk] > T00:
            ML = nk
    for nk in range(1, K + 1):
        if nk >= LC:
            if nk == LC:
                UMF[nk] = VMFLCL * DP[nk] / DPTHMX
                UER[nk] = VMFLCL * DP[nk] / DPTHMX
            elif nk <= KPBL:
                UER[nk] = VMFLCL * DP[nk] / DPTHMX
                UMF[nk] = UMF[nk - 1] + UER[nk]
            else:
                UMF[nk] = VMFLCL
                UER[nk] = 0.0
            TU[nk] = TMIX + (Z0[nk] - ZMIX) * gdry
            QU[nk] = QMIX
            WU[nk] = WLCL
        else:
            TU[nk] = 0.0; QU[nk] = 0.0; UMF[nk] = 0.0; WU[nk] = 0.0; UER[nk] = 0.0
        UDR[nk] = 0.0; QDT[nk] = 0.0; QLIQ[nk] = 0.0; QICE[nk] = 0.0
        QLQOUT[nk] = 0.0; QICOUT[nk] = 0.0; PPTLIQ[nk] = 0.0; PPTICE[nk] = 0.0
        DETLQ[nk] = 0.0; DETIC[nk] = 0.0; RATIO2[nk] = 0.0
        THETEE[nk] = envirtht(P0[nk], T0[nk], Q0[nk], aliq, bliq, cliq, dliq)
        EQFRC[nk] = 1.0

    LTOP1 = LTOP + 1
    LTOPM1 = LTOP - 1
    for nk in range(LTOP1, KX + 1):
        UMF[nk] = 0.0; UDR[nk] = 0.0; UER[nk] = 0.0; QDT[nk] = 0.0
        QLIQ[nk] = 0.0; QICE[nk] = 0.0; QLQOUT[nk] = 0.0; QICOUT[nk] = 0.0
        DETLQ[nk] = 0.0; DETIC[nk] = 0.0; PPTLIQ[nk] = 0.0; PPTICE[nk] = 0.0
        if nk > LTOP1:
            TU[nk] = 0.0; QU[nk] = 0.0; WU[nk] = 0.0
        THTA0[nk] = 0.0; THTAU[nk] = 0.0; EMS[nk] = 0.0; EMSD[nk] = 0.0
        TG[nk] = T0[nk]; QG[nk] = Q0[nk]
        QLG[nk] = 0.0; QIG[nk] = 0.0; QRG[nk] = 0.0; QSG[nk] = 0.0
        OMG[nk] = 0.0
    OMG[KX + 1] = 0.0
    for nk in range(1, LTOP + 1):
        EMS[nk] = DP[nk] * DXSQ / G
        EMSD[nk] = 1.0 / EMS[nk]
        EXN[nk] = (P00 / P0[nk]) ** (0.2854 * (1.0 - 0.28 * QDT[nk]))
        THTAU[nk] = TU[nk] * EXN[nk]
        EXN[nk] = (P00 / P0[nk]) ** (0.2854 * (1.0 - 0.28 * Q0[nk]))
        THTA0[nk] = T0[nk] * EXN[nk]
        DDILFRC[nk] = 1.0 / DILFRC[nk]
        OMG[nk] = 0.0

    # convective time scale
    WSPD[KLCL] = math.sqrt(U0[KLCL]**2 + V0[KLCL]**2)
    WSPD[L5] = math.sqrt(U0[L5]**2 + V0[L5]**2)
    WSPD[LTOP] = math.sqrt(U0[LTOP]**2 + V0[LTOP]**2)
    VCONV = 0.5 * (WSPD[KLCL] + WSPD[L5])
    TIMEC = dx / VCONV if VCONV > 0.0 else 1.0e30
    TADVEC = TIMEC
    TIMEC = max(1800.0, TIMEC)
    TIMEC = min(3600.0, TIMEC)
    if ISHALL == 1:
        TIMEC = 2400.0
    NIC = int(round(TIMEC / dt))
    TIMEC = float(NIC) * dt

    SHSIGN = 1.0 if WSPD[LTOP] > WSPD[KLCL] else -1.0
    VWS = (U0[LTOP] - U0[KLCL])**2 + (V0[LTOP] - V0[KLCL])**2
    VWS = 1.0e3 * SHSIGN * math.sqrt(VWS) / (Z0[LTOP] - Z0[LCL])
    PEF = 1.591 + VWS * (-0.639 + VWS * (9.53e-2 - VWS * 4.96e-3))
    PEF = min(max(PEF, 0.2), 0.9)
    CBH = (ZLCL - Z0[1]) * 3.281e-3
    if CBH < 3.0:
        RCBH = 0.02
    else:
        RCBH = (0.96729352 + CBH * (-0.70034167 + CBH * (0.162179896 + CBH * (
            -1.2569798e-2 + CBH * (4.2772e-4 - CBH * 5.44e-6)))))
    if CBH > 25.0:
        RCBH = 2.4
    PEFCBH = 1.0 / (1.0 + RCBH)
    PEFCBH = min(PEFCBH, 0.9)
    PEFF = 0.5 * (PEF + PEFCBH)
    PEFF2 = PEFF

    # ---- downdraft ----
    TVD = z1()
    TDER = 0.0
    LFS = 1
    LDB = 0; LDT = 0
    KSTART = 0
    if ISHALL == 1:
        LFS = 1
    else:
        KSTART = KPBL + 1
        KLFS = LET - 1
        for nk in range(KSTART + 1, KL + 1):
            DPPP = P0[KSTART] - P0[nk]
            if DPPP > 150.0e2:
                KLFS = nk
                break
        KLFS = min(KLFS, LET - 1)
        LFS = KLFS
        if (P0[KSTART] - P0[LFS]) > 50.0e2:
            THETED[LFS] = THETEE[LFS]
            QD[LFS] = Q0[LFS]
            TZ[LFS], QSS = tpmix2dd(P0[LFS], THETED[LFS])
            THTAD[LFS] = TZ[LFS] * (P00 / P0[LFS]) ** (0.2854 * (1.0 - 0.28 * QSS))
            TVD[LFS] = TZ[LFS] * (1.0 + 0.608 * QSS)
            RDD = P0[LFS] / (R_D * TVD[LFS])
            A1 = (1.0 - PEFF) * AU0
            DMF[LFS] = -A1 * RDD
            DER[LFS] = DMF[LFS]
            DDR[LFS] = 0.0
            RHBAR = RH[LFS] * DP[LFS]
            DPTT = DP[LFS]
            for nd in range(LFS - 1, KSTART - 1, -1):
                nd1 = nd + 1
                DER[nd] = DER[LFS] * EMS[nd] / EMS[LFS]
                DDR[nd] = 0.0
                DMF[nd] = DMF[nd1] + DER[nd]
                THETED[nd] = (THETED[nd1] * DMF[nd1] + THETEE[nd] * DER[nd]) / DMF[nd]
                QD[nd] = (QD[nd1] * DMF[nd1] + Q0[nd] * DER[nd]) / DMF[nd]
                DPTT += DP[nd]
                RHBAR += RH[nd] * DP[nd]
            RHBAR = RHBAR / DPTT
            DMFFRC = 2.0 * (1.0 - RHBAR)
            DPDD = 0.0
            pptmlt = 0.0
            for nk in range(KLCL, LTOP + 1):
                pptmlt += PPTICE[nk]
            if LC < ML:
                DTMELT = RLF * pptmlt / (CP * UMF[KLCL])
            else:
                DTMELT = 0.0
            LDT = min(LFS - 1, KSTART - 1)
            TZ[KSTART], QSS = tpmix2dd(P0[KSTART], THETED[KSTART])
            TZ[KSTART] = TZ[KSTART] - DTMELT
            es = aliq * math.exp((bliq * TZ[KSTART] - cliq) / (TZ[KSTART] - dliq))
            QSS = 0.622 * es / (P0[KSTART] - es)
            THETED[KSTART] = TZ[KSTART] * (1.0e5 / P0[KSTART]) ** (0.2854 * (1.0 - 0.28 * QSS)) * \
                math.exp((3374.6525 / TZ[KSTART] - 2.5403) * QSS * (1.0 + 0.81 * QSS))
            LDT = min(LFS - 1, KSTART - 1)
            LDB = 1
            for nd in range(LDT, 0, -1):
                DPDD += DP[nd]
                THETED[nd] = THETED[KSTART]
                QD[nd] = QD[KSTART]
                TZ[nd], QSS = tpmix2dd(P0[nd], THETED[nd])
                QSD[nd] = QSS
                RHH = 1.0 - 0.2 / 1000.0 * (Z0[KSTART] - Z0[nd])
                if RHH < 1.0:
                    DSSDT = (cliq - bliq * dliq) / ((TZ[nd] - dliq) * (TZ[nd] - dliq))
                    RL = XLV0 - XLV1 * TZ[nd]
                    DTMP = RL * QSS * (1.0 - RHH) / (CP + RL * RHH * QSS * DSSDT)
                    T1RH = TZ[nd] + DTMP
                    es = RHH * aliq * math.exp((bliq * T1RH - cliq) / (T1RH - dliq))
                    QSRH = 0.622 * es / (P0[nd] - es)
                    if QSRH < QD[nd]:
                        QSRH = QD[nd]
                        T1RH = TZ[nd] + (QSS - QSRH) * RL / CP
                    TZ[nd] = T1RH
                    QSS = QSRH
                    QSD[nd] = QSS
                TVD[nd] = TZ[nd] * (1.0 + 0.608 * QSD[nd])
                if TVD[nd] > TV0[nd] or nd == 1:
                    LDB = nd
                    break
            if (P0[LDB] - P0[LFS]) > 50.0e2:
                for nd in range(LDT, LDB - 1, -1):
                    nd1 = nd + 1
                    DDR[nd] = -DMF[KSTART] * DP[nd] / DPDD
                    DER[nd] = 0.0
                    DMF[nd] = DMF[nd1] + DDR[nd]
                    TDER = TDER + (QSD[nd] - QD[nd]) * DDR[nd]
                    QD[nd] = QSD[nd]
                    THTAD[nd] = TZ[nd] * (P00 / P0[nd]) ** (0.2854 * (1.0 - 0.28 * QD[nd]))

    # ---- no-downdraft vs downdraft (d_mf) ----
    CPR = 0.0; PPTFLX = 0.0; CNDTNF = 0.0; UPDINC = 1.0; AINCM2 = 100.0
    if TDER < 1.0:
        PPTFLX = TRPPT
        CPR = TRPPT
        TDER = 0.0
        CNDTNF = 0.0
        UPDINC = 1.0
        LDB = LFS
        for ndk in range(1, LTOP + 1):
            DMF[ndk] = 0.0; DER[ndk] = 0.0; DDR[ndk] = 0.0
            THTAD[ndk] = 0.0; WD[ndk] = 0.0; TZ[ndk] = 0.0; QD[ndk] = 0.0
        AINCM2 = 100.0
    else:
        DDINC = -DMFFRC * UMF[KLCL] / DMF[KSTART]
        UPDINC = 1.0
        if TDER * DDINC > TRPPT:
            DDINC = TRPPT / TDER
        TDER = TDER * DDINC
        for nk in range(LDB, LFS + 1):
            DMF[nk] = DMF[nk] * DDINC
            DER[nk] = DER[nk] * DDINC
            DDR[nk] = DDR[nk] * DDINC
        CPR = TRPPT
        PPTFLX = TRPPT - TDER
        PEFF = PPTFLX / TRPPT
        if LDB > 1:
            for nk in range(1, LDB):
                DMF[nk] = 0.0; DER[nk] = 0.0; DDR[nk] = 0.0
                WD[nk] = 0.0; TZ[nk] = 0.0; QD[nk] = 0.0; THTAD[nk] = 0.0
        for nk in range(LFS + 1, KX + 1):
            DMF[nk] = 0.0; DER[nk] = 0.0; DDR[nk] = 0.0
            WD[nk] = 0.0; TZ[nk] = 0.0; QD[nk] = 0.0; THTAD[nk] = 0.0
        for nk in range(LDT + 1, LFS):
            TZ[nk] = 0.0; QD[nk] = 0.0; THTAD[nk] = 0.0

    # ---- mass-flux limits (AINC) ----
    AINCMX = 1000.0
    LMAX = max(KLCL, LFS)
    for nk in range(LC, LMAX + 1):
        if (UER[nk] - DER[nk]) > 1.0e-3:
            AINCM1 = EMS[nk] / ((UER[nk] - DER[nk]) * TIMEC)
            AINCMX = min(AINCMX, AINCM1)
    AINC = 1.0
    if AINCMX < AINC:
        AINC = AINCMX

    TDER2 = TDER
    PPTFL2 = PPTFLX
    for nk in range(1, LTOP + 1):
        DETLQ2[nk] = DETLQ[nk]; DETIC2[nk] = DETIC[nk]
        UDR2[nk] = UDR[nk]; UER2[nk] = UER[nk]
        DDR2[nk] = DDR[nk]; DER2[nk] = DER[nk]
        UMF2[nk] = UMF[nk]; DMF2[nk] = DMF[nk]
    FABE = 1.0
    STAB = 0.95
    NOITR = 0
    ISTOP = 0

    if ISHALL == 1:
        TKEMAX = 5.0
        EVAC = 0.5 * TKEMAX * 0.1
        AINC = EVAC * DPTHMX * DXSQ / (VMFLCL * G * TIMEC)
        TDER = TDER2 * AINC
        PPTFLX = PPTFL2 * AINC
        for nk in range(1, LTOP + 1):
            UMF[nk] = UMF2[nk] * AINC; DMF[nk] = DMF2[nk] * AINC
            DETLQ[nk] = DETLQ2[nk] * AINC; DETIC[nk] = DETIC2[nk] * AINC
            UDR[nk] = UDR2[nk] * AINC; UER[nk] = UER2[nk] * AINC
            DER[nk] = DER2[nk] * AINC; DDR[nk] = DDR2[nk] * AINC

    AINCOLD = AINC
    FABEOLD = 1.0
    NSTEP = 1
    DTIME = TIMEC
    # ---- closure iteration ----
    for NCOUNT in range(1, 10 + 1):   # Fortran: iter: DO NCOUNT=1,10
        DTT = TIMEC
        for nk in range(1, LTOP + 1):
            DOMGDP[nk] = -(UER[nk] - DER[nk] - UDR[nk] - DDR[nk]) * EMSD[nk]
            if nk > 1:
                OMG[nk] = OMG[nk - 1] - DP[nk - 1] * DOMGDP[nk - 1]
                ABSOMG = abs(OMG[nk])
                ABSOMGTC = ABSOMG * TIMEC
                FRDP = 0.75 * DP[nk - 1]
                if ABSOMGTC > FRDP:
                    DTT1 = FRDP / ABSOMG
                    DTT = min(DTT, DTT1)
        for nk in range(1, LTOP + 1):
            THPA[nk] = THTA0[nk]
            QPA[nk] = Q0[nk]
            NSTEP = int(round(TIMEC / DTT + 1))
            DTIME = TIMEC / float(NSTEP)
            FXM[nk] = OMG[nk] * DXSQ / G

        for ntc in range(NSTEP):
            for nk in range(1, LTOP + 1):
                THFXIN[nk] = 0.0; THFXOUT[nk] = 0.0; QFXIN[nk] = 0.0; QFXOUT[nk] = 0.0
            for nk in range(2, LTOP + 1):
                if OMG[nk] <= 0.0:
                    THFXIN[nk] = -FXM[nk] * THPA[nk - 1]
                    QFXIN[nk] = -FXM[nk] * QPA[nk - 1]
                    THFXOUT[nk - 1] = THFXOUT[nk - 1] + THFXIN[nk]
                    QFXOUT[nk - 1] = QFXOUT[nk - 1] + QFXIN[nk]
                else:
                    THFXOUT[nk] = FXM[nk] * THPA[nk]
                    QFXOUT[nk] = FXM[nk] * QPA[nk]
                    THFXIN[nk - 1] = THFXIN[nk - 1] + THFXOUT[nk]
                    QFXIN[nk - 1] = QFXIN[nk - 1] + QFXOUT[nk]
            for nk in range(1, LTOP + 1):
                THPA[nk] = THPA[nk] + (THFXIN[nk] + UDR[nk] * THTAU[nk] + DDR[nk] * THTAD[nk]
                                       - THFXOUT[nk] - (UER[nk] - DER[nk]) * THTA0[nk]) * DTIME * EMSD[nk]
                QPA[nk] = QPA[nk] + (QFXIN[nk] + UDR[nk] * QDT[nk] + DDR[nk] * QD[nk]
                                     - QFXOUT[nk] - (UER[nk] - DER[nk]) * Q0[nk]) * DTIME * EMSD[nk]
        for nk in range(1, LTOP + 1):
            THTAG[nk] = THPA[nk]
            QG[nk] = QPA[nk]

        # moisture-borrow fixup
        for nk in range(1, LTOP + 1):
            if QG[nk] < 0.0:
                if nk == 1:
                    raise FloatingPointError("QG<0 at surface in KF (ref)")
                nk1 = nk + 1
                if nk == LTOP:
                    nk1 = KLCL
                TMA = QG[nk1] * EMS[nk1]
                TMB = QG[nk - 1] * EMS[nk - 1]
                TMM = (QG[nk] - 1.0e-9) * EMS[nk]
                BCOEFF = -TMM / ((TMA * TMA) / TMB + TMB)
                ACOEFF = BCOEFF * TMA / TMB
                TMB = TMB * (1.0 - BCOEFF)
                TMA = TMA * (1.0 - ACOEFF)
                QG[nk] = 1.0e-9
                QG[nk1] = TMA * EMSD[nk1]
                QG[nk - 1] = TMB * EMSD[nk - 1]

        TOPOMG = (UDR[LTOP] - UER[LTOP]) * DP[LTOP] * EMSD[LTOP]
        if abs(TOPOMG - OMG[LTOP]) > 1.0e-3:
            ISTOP = 1
            break

        for nk in range(1, LTOP + 1):
            EXN[nk] = (P00 / P0[nk]) ** (0.2854 * (1.0 - 0.28 * QG[nk]))
            TG[nk] = THTAG[nk] / EXN[nk]
            TVG[nk] = TG[nk] * (1.0 + 0.608 * QG[nk])

        if ISHALL == 1:
            break

        # new cloud / CAPE change
        TMIX = 0.0; QMIX = 0.0
        for nk in range(LC, KPBL + 1):
            TMIX += DP[nk] * TG[nk]
            QMIX += DP[nk] * QG[nk]
        TMIX /= DPTHMX
        QMIX /= DPTHMX
        es = aliq * math.exp((TMIX * bliq - cliq) / (TMIX - dliq))
        QSS = 0.622 * es / (PMIX - es)
        if QMIX > QSS:
            RL = XLV0 - XLV1 * TMIX
            CPM = CP * (1.0 + 0.887 * QMIX)
            DSSDT = QSS * (cliq - bliq * dliq) / ((TMIX - dliq) * (TMIX - dliq))
            DQ = (QMIX - QSS) / (1.0 + RL * DSSDT / CPM)
            TMIX = TMIX + RL / CP * DQ
            QMIX = QMIX - DQ
            TLCL = TMIX
        else:
            QMIX = max(QMIX, 0.0)
            EMIX = QMIX * PMIX / (0.622 + QMIX)
            astrt = 1.0e-3; binc = 0.075
            a1 = EMIX / aliq
            tp = (a1 - astrt) / binc
            indlu = int(tp) + 1
            value = (indlu - 1) * binc + astrt
            aintrp = (a1 - value) / binc
            tlog = aintrp * _T.ALU[indlu] + (1.0 - aintrp) * _T.ALU[indlu - 1]
            TDPT = (cliq - dliq * tlog) / (bliq - tlog)
            TLCL = TDPT - (0.212 + 1.571e-3 * (TDPT - T00) - 4.36e-4 * (TMIX - T00)) * (TMIX - TDPT)
            TLCL = min(TLCL, TMIX)
        TVLCL = TLCL * (1.0 + 0.608 * QMIX)
        ZLCL = ZMIX + (TLCL - TMIX) / gdry
        for nk in range(LC, KL + 1):
            KLCL = nk
            if ZLCL <= Z0[nk]:
                break
        K = KLCL - 1
        DLP = (ZLCL - Z0[K]) / (Z0[KLCL] - Z0[K])
        TENV = TG[K] + (TG[KLCL] - TG[K]) * DLP
        QENV = QG[K] + (QG[KLCL] - QG[K]) * DLP
        TVEN = TENV * (1.0 + 0.608 * QENV)
        PLCL = P0[K] + (P0[KLCL] - P0[K]) * DLP
        THETEU[K] = TMIX * (1.0e5 / PMIX) ** (0.2854 * (1.0 - 0.28 * QMIX)) * \
            math.exp((3374.6525 / TLCL - 2.5403) * QMIX * (1.0 + 0.81 * QMIX))
        ABEG = 0.0
        for nk in range(K, LTOPM1 + 1):
            nk1 = nk + 1
            THETEU[nk1] = THETEU[nk]
            TGU[nk1], QGU[nk1] = tpmix2dd(P0[nk1], THETEU[nk1])
            TVQU[nk1] = TGU[nk1] * (1.0 + 0.608 * QGU[nk1] - QLIQ[nk1] - QICE[nk1])
            if nk == K:
                DZZ = Z0[KLCL] - ZLCL
                DILBE = ((TVLCL + TVQU[nk1]) / (TVEN + TVG[nk1]) - 1.0) * DZZ
            else:
                DZZ = DZA[nk]
                DILBE = ((TVQU[nk] + TVQU[nk1]) / (TVG[nk] + TVG[nk1]) - 1.0) * DZZ
            if DILBE > 0.0:
                ABEG += DILBE * G
            THTEEG[nk1] = envirtht(P0[nk1], TG[nk1], QG[nk1], aliq, bliq, cliq, dliq)
            THETEU[nk1] = THETEU[nk1] * DDILFRC[nk1] + THTEEG[nk1] * (1.0 - DDILFRC[nk1])

        if NOITR == 1:
            break
        DABE = max(ABE - ABEG, 0.1 * ABE)
        FABE = ABEG / ABE
        if FABE > 1.0 and ISHALL == 0:
            ISHALL = 2
            return _empty_result(KX, NCA)
        if NCOUNT != 1:
            if abs(AINC - AINCOLD) < 0.0001:
                NOITR = 1
                AINC = AINCOLD
                continue
            DFDA = (FABE - FABEOLD) / (AINC - AINCOLD)
            if DFDA > 0.0:
                NOITR = 1
                AINC = AINCOLD
                continue
        AINCOLD = AINC
        FABEOLD = FABE
        if AINC / AINCMX > 0.999 and FABE > 1.05 - STAB:
            break
        if (FABE <= 1.05 - STAB and FABE >= 0.95 - STAB) or NCOUNT == 10:
            break
        else:
            if FABE == 0.0:
                AINC = AINC * 0.5
            else:
                if DABE < 1.0e-4:
                    NOITR = 1
                    AINC = AINCOLD
                    continue
                else:
                    AINC = AINC * STAB * ABE / DABE
            AINC = min(AINCMX, AINC)
            if AINC < 0.05:
                ISHALL = 2
                return _empty_result(KX, NCA)
            TDER = TDER2 * AINC
            PPTFLX = PPTFL2 * AINC
            for nk in range(1, LTOP + 1):
                UMF[nk] = UMF2[nk] * AINC; DMF[nk] = DMF2[nk] * AINC
                DETLQ[nk] = DETLQ2[nk] * AINC; DETIC[nk] = DETIC2[nk] * AINC
                UDR[nk] = UDR2[nk] * AINC; UER[nk] = UER2[nk] * AINC
                DER[nk] = DER2[nk] * AINC; DDR[nk] = DDR2[nk] * AINC
    # ---- end closure iteration ----

    # ---- hydrometeor tendencies (advection) ----
    if CPR > 0.0:
        FRC2 = PPTFLX / (CPR * AINC)
    else:
        FRC2 = 0.0
    QL0 = z1(); QI0 = z1(); QR0 = z1(); QS0 = z1()
    for nk in range(1, LTOP + 1):
        QLPA[nk] = QL0[nk]; QIPA[nk] = QI0[nk]
        QRPA[nk] = QR0[nk]; QSPA[nk] = QS0[nk]
        RAINFB[nk] = PPTLIQ[nk] * AINC * FBFRC * FRC2
        SNOWFB[nk] = PPTICE[nk] * AINC * FBFRC * FRC2
    for ntc in range(NSTEP):
        for nk in range(1, LTOP + 1):
            QLFXIN[nk] = 0.0; QLFXOUT[nk] = 0.0; QIFXIN[nk] = 0.0; QIFXOUT[nk] = 0.0
            QRFXIN[nk] = 0.0; QRFXOUT[nk] = 0.0; QSFXIN[nk] = 0.0; QSFXOUT[nk] = 0.0
        for nk in range(2, LTOP + 1):
            if OMG[nk] <= 0.0:
                QLFXIN[nk] = -FXM[nk] * QLPA[nk - 1]
                QIFXIN[nk] = -FXM[nk] * QIPA[nk - 1]
                QRFXIN[nk] = -FXM[nk] * QRPA[nk - 1]
                QSFXIN[nk] = -FXM[nk] * QSPA[nk - 1]
                QLFXOUT[nk - 1] += QLFXIN[nk]
                QIFXOUT[nk - 1] += QIFXIN[nk]
                QRFXOUT[nk - 1] += QRFXIN[nk]
                QSFXOUT[nk - 1] += QSFXIN[nk]
            else:
                QLFXOUT[nk] = FXM[nk] * QLPA[nk]
                QIFXOUT[nk] = FXM[nk] * QIPA[nk]
                QRFXOUT[nk] = FXM[nk] * QRPA[nk]
                QSFXOUT[nk] = FXM[nk] * QSPA[nk]
                QLFXIN[nk - 1] += QLFXOUT[nk]
                QIFXIN[nk - 1] += QIFXOUT[nk]
                QRFXIN[nk - 1] += QRFXOUT[nk]
                QSFXIN[nk - 1] += QSFXOUT[nk]
        for nk in range(1, LTOP + 1):
            QLPA[nk] = QLPA[nk] + (QLFXIN[nk] + DETLQ[nk] - QLFXOUT[nk]) * DTIME * EMSD[nk]
            QIPA[nk] = QIPA[nk] + (QIFXIN[nk] + DETIC[nk] - QIFXOUT[nk]) * DTIME * EMSD[nk]
            QRPA[nk] = QRPA[nk] + (QRFXIN[nk] - QRFXOUT[nk] + RAINFB[nk]) * DTIME * EMSD[nk]
            QSPA[nk] = QSPA[nk] + (QSFXIN[nk] - QSFXOUT[nk] + SNOWFB[nk]) * DTIME * EMSD[nk]
    for nk in range(1, LTOP + 1):
        QLG[nk] = QLPA[nk]; QIG[nk] = QIPA[nk]; QRG[nk] = QRPA[nk]; QSG[nk] = QSPA[nk]

    CNDTNF = (1.0 - EQFRC[LFS]) * (QLIQ[LFS] + QICE[LFS]) * DMF[LFS]
    PRATEC = PPTFLX * (1.0 - FBFRC) / DXSQ
    RAINCV = dt * PRATEC
    RNC = RAINCV * NIC

    # moisture budget (diagnostic; matches Fortran ERR2)
    QINIT = 0.0; QFNL = 0.0; DPT = 0.0
    for nk in range(1, LTOP + 1):
        DPT += DP[nk]
        QINIT += Q0[nk] * EMS[nk]
        QFNL += QG[nk] * EMS[nk]
        QFNL += (QLG[nk] + QIG[nk] + QRG[nk] + QSG[nk]) * EMS[nk]
    QFNL += PPTFLX * TIMEC * (1.0 - FBFRC)
    ERR2 = (QFNL - QINIT) * 100.0 / QINIT

    # feedback to resolvable scale
    if TADVEC < TIMEC:
        NIC = int(round(TADVEC / dt))
    NCA = float(NIC) * dt
    if ISHALL == 1:
        TIMEC = 2400.0
        NCA = cudt * 60.0

    for k in range(1, KX + 1):
        if warm_rain:
            CPM = CP * (1.0 + 0.887 * QG[k])
            TG[k] = TG[k] - (QIG[k] + QSG[k]) * RLF / CPM
            DQCDT[k] = (QLG[k] + QIG[k] - QL0[k] - QI0[k]) / TIMEC
            DQIDT[k] = 0.0
            DQRDT[k] = (QRG[k] + QSG[k] - QR0[k] - QS0[k]) / TIMEC
            DQSDT[k] = 0.0
        elif not f_qs:
            CPM = CP * (1.0 + 0.887 * QG[k])
            if k <= ML:
                TG[k] = TG[k] - (QIG[k] + QSG[k]) * RLF / CPM
            else:
                TG[k] = TG[k] + (QLG[k] + QRG[k]) * RLF / CPM
            DQCDT[k] = (QLG[k] + QIG[k] - QL0[k] - QI0[k]) / TIMEC
            DQIDT[k] = 0.0
            DQRDT[k] = (QRG[k] + QSG[k] - QR0[k] - QS0[k]) / TIMEC
            DQSDT[k] = 0.0
        else:  # f_qs True (mixed phase)
            DQCDT[k] = (QLG[k] - QL0[k]) / TIMEC
            DQSDT[k] = (QSG[k] - QS0[k]) / TIMEC
            DQRDT[k] = (QRG[k] - QR0[k]) / TIMEC
            if f_qi:
                DQIDT[k] = (QIG[k] - QI0[k]) / TIMEC
            else:
                DQSDT[k] = DQSDT[k] + (QIG[k] - QI0[k]) / TIMEC
        DTDT[k] = (TG[k] - T0[k]) / TIMEC
        DQDT[k] = (QG[k] - Q0[k]) / TIMEC
    PRATEC = PPTFLX * (1.0 - FBFRC) / DXSQ
    RAINCV = dt * PRATEC

    CUTOP = float(LTOP)
    CUBOT = float(LCL)

    # pack 0-based outputs + driver-applied RTHCUTEN=DTDT/pi etc.
    if pi_exner is None:
        pii = np.array([(p0[k] / 1.0e5) ** (R_D / CP) for k in range(KX)], dtype=np.float64)
    else:
        pii = np.asarray(pi_exner, dtype=np.float64)
    out = dict()
    for name, arr in (("DTDT", DTDT), ("DQDT", DQDT), ("DQCDT", DQCDT),
                      ("DQRDT", DQRDT), ("DQIDT", DQIDT), ("DQSDT", DQSDT)):
        out[name] = np.array([arr[k] for k in range(1, KX + 1)], dtype=np.float64)
    out["RTHCUTEN"] = out["DTDT"] / pii
    out["RQVCUTEN"] = out["DQDT"].copy()
    # driver: F_QR present -> RQRCUTEN=DQRDT, RQCCUTEN=DQCDT
    out["RQRCUTEN"] = out["DQRDT"].copy()
    out["RQCCUTEN"] = out["DQCDT"].copy()
    out["RQICUTEN"] = out["DQIDT"].copy()
    out["RQSCUTEN"] = out["DQSDT"].copy()
    out["RAINCV"] = RAINCV
    out["PRATEC"] = PRATEC
    out["NCA"] = NCA
    out["CUTOP"] = CUTOP
    out["CUBOT"] = CUBOT
    out["ISHALL"] = ISHALL
    out["TIMEC"] = TIMEC
    out["ERR2"] = ERR2
    return out
