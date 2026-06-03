"""Scratch: build BMJ lookup tables in NumPy (replica of BMJINIT+SPLINE)
and spot-check against a tiny Fortran probe is not available; just sanity print."""
import numpy as np

# constants
PQ0 = 379.90516
A2 = 17.2693882
A3 = 273.16
A4 = 35.86
EPSQ = 1.0e-12
ELIWV = 2.683e6
CP = 1004.5
RD = 287.0
ELOCP = ELIWV / CP
EPS = 1.0e-9

ITB = 76
JTB = 134
ITBQ = 152
JTBQ = 440
PL = 2500.0
PLQ = 70000.0
PH = 105000.0
THL = 210.0
THH = 365.0
THHQ = 325.0


def spline(nold, xold, yold):
    """Faithful replica of WRF SPLINE for natural spline, returns YNEW on xnew
    is done separately; here we return y2 (2nd deriv) given natural BCs."""
    # This mirrors only the y2 solve; evaluation done in build via direct call.
    raise NotImplementedError


def wrf_spline(nold, xold, yold, nnew, xnew):
    """Direct port of SUBROUTINE SPLINE (natural spline Y2(1)=Y2(NOLD)=0)."""
    xold = np.asarray(xold, dtype=np.float64)
    yold = np.asarray(yold, dtype=np.float64)
    xnew = np.asarray(xnew, dtype=np.float64)
    P = np.zeros(max(nold, nnew) + 2, dtype=np.float64)
    Q = np.zeros(max(nold, nnew) + 2, dtype=np.float64)
    Y2 = np.zeros(max(nold, nnew) + 2, dtype=np.float64)
    YNEW = np.zeros(nnew, dtype=np.float64)
    # 1-based helpers
    def xo(i):
        return xold[i - 1]
    def yo(i):
        return yold[i - 1]
    NOLDM1 = nold - 1
    DXL = xo(2) - xo(1)
    DXR = xo(3) - xo(2)
    DYDXL = (yo(2) - yo(1)) / DXL
    DYDXR = (yo(3) - yo(2)) / DXR
    RTDXC = 0.5 / (DXL + DXR)
    P[1] = RTDXC * (6.0 * (DYDXR - DYDXL) - DXL * Y2[1])
    Q[1] = -RTDXC * DXR
    if nold != 3:
        K = 3
        while True:
            DXL = DXR
            DYDXL = DYDXR
            DXR = xo(K + 1) - xo(K)
            DYDXR = (yo(K + 1) - yo(K)) / DXR
            DXC = DXL + DXR
            DEN = 1.0 / (DXL * Q[K - 2] + DXC + DXC)
            P[K - 1] = DEN * (6.0 * (DYDXR - DYDXL) - DXL * P[K - 2])
            Q[K - 1] = -DEN * DXR
            K = K + 1
            if not (K < nold):
                break
    K = NOLDM1
    while True:
        Y2[K] = P[K - 1] + Q[K - 1] * Y2[K + 1]
        K = K - 1
        if not (K > 1):
            break
    K = 1  # 'K' in fortran retains across K1 loop; init
    K1 = 1
    while True:
        XK = xnew[K1 - 1]
        KOLD = None
        found = False
        for K2 in range(2, nold + 1):
            if xo(K2) > XK:
                KOLD = K2 - 1
                found = True
                break
        if not found:
            YNEW[K1 - 1] = yo(nold)
            K1 += 1
            if K1 <= nnew:
                continue
            else:
                break
        # label 450
        recompute = True
        if K1 == 1:
            recompute = True
        elif K == KOLD:
            recompute = False
        if recompute:
            K = KOLD
            Y2K = Y2[K]
            Y2KP1 = Y2[K + 1]
            DX = xo(K + 1) - xo(K)
            RDX = 1.0 / DX
            AK = 0.1666667 * RDX * (Y2KP1 - Y2K)
            BK = 0.5 * Y2K
            CK = RDX * (yo(K + 1) - yo(K)) - 0.1666667 * DX * (Y2KP1 + Y2K + Y2K)
        else:
            # reuse AK,BK,CK from previous
            pass
        X = XK - xo(K)
        XSQ = X * X
        YNEW[K1 - 1] = AK * XSQ * X + BK * XSQ + CK * X + yo(K)
        K1 += 1
        if not (K1 <= nnew):
            break
    return YNEW


def build_tables():
    QS0 = np.zeros(JTB)
    SQS = np.zeros(JTB)
    THE0 = np.zeros(ITB)
    STHE = np.zeros(ITB)
    THE0Q = np.zeros(ITBQ)
    STHEQ = np.zeros(ITBQ)
    PTBL = np.zeros((ITB, JTB))  # PTBL(KP,KTH) -> index [kp, kth]
    TTBL = np.zeros((JTB, ITB))  # TTBL(KTH,KP)
    TTBLQ = np.zeros((JTBQ, ITBQ))

    KTHM = JTB
    KPM = ITB
    KTHM1 = KTHM - 1
    KPM1 = KPM - 1
    DTH = (THH - THL) / float(KTHM - 1)
    DP = (PH - PL) / float(KPM - 1)

    # ---- table 100: PTBL ----
    TH = THL - DTH
    for KTH in range(1, KTHM + 1):
        TH = TH + DTH
        P = PL - DP
        QSOLD = np.zeros(KPM + 1)
        POLD = np.zeros(KPM + 1)
        for KP in range(1, KPM + 1):
            P = P + DP
            APE = (100000.0 / P) ** (RD / CP)
            DENOM = TH - A4 * APE
            if DENOM > EPS:
                QSOLD[KP] = PQ0 / P * np.exp(A2 * (TH - A3 * APE) / DENOM)
            else:
                QSOLD[KP] = 0.0
            POLD[KP] = P
        QS0K = QSOLD[1]
        SQSK = QSOLD[KPM] - QSOLD[1]
        QSOLD[1] = 0.0
        QSOLD[KPM] = 1.0
        for KP in range(2, KPM1 + 1):
            QSOLD[KP] = (QSOLD[KP] - QS0K) / SQSK
            if (QSOLD[KP] - QSOLD[KP - 1]) < EPS:
                QSOLD[KP] = QSOLD[KP - 1] + EPS
        QS0[KTH - 1] = QS0K
        SQS[KTH - 1] = SQSK
        QSNEW = np.zeros(KPM + 1)
        QSNEW[1] = 0.0
        QSNEW[KPM] = 1.0
        DQS = 1.0 / float(KPM - 1)
        for KP in range(2, KPM1 + 1):
            QSNEW[KP] = QSNEW[KP - 1] + DQS
        PNEW = wrf_spline(KPM, QSOLD[1:KPM + 1], POLD[1:KPM + 1], KPM, QSNEW[1:KPM + 1])
        for KP in range(1, KPM + 1):
            PTBL[KP - 1, KTH - 1] = PNEW[KP - 1]

    # ---- table 200: TTBL ----
    P = PL - DP
    for KP in range(1, KPM + 1):
        P = P + DP
        TH = THL - DTH
        TOLD = np.zeros(KTHM + 1)
        THEOLD = np.zeros(KTHM + 1)
        for KTH in range(1, KTHM + 1):
            TH = TH + DTH
            APE = (1.0e5 / P) ** (RD / CP)
            DENOM = TH - A4 * APE
            if DENOM > EPS:
                QS = PQ0 / P * np.exp(A2 * (TH - A3 * APE) / DENOM)
            else:
                QS = 0.0
            TOLD[KTH] = TH / APE
            THEOLD[KTH] = TH * np.exp(ELOCP * QS / TOLD[KTH])
        THE0K = THEOLD[1]
        STHEK = THEOLD[KTHM] - THEOLD[1]
        THEOLD[1] = 0.0
        THEOLD[KTHM] = 1.0
        for KTH in range(2, KTHM1 + 1):
            THEOLD[KTH] = (THEOLD[KTH] - THE0K) / STHEK
            if (THEOLD[KTH] - THEOLD[KTH - 1]) < EPS:
                THEOLD[KTH] = THEOLD[KTH - 1] + EPS
        THE0[KP - 1] = THE0K
        STHE[KP - 1] = STHEK
        THENEW = np.zeros(KTHM + 1)
        THENEW[1] = 0.0
        THENEW[KTHM] = 1.0
        DTHE = 1.0 / float(KTHM - 1)
        for KTH in range(2, KTHM1 + 1):
            THENEW[KTH] = THENEW[KTH - 1] + DTHE
        TNEW = wrf_spline(KTHM, THEOLD[1:KTHM + 1], TOLD[1:KTHM + 1], KTHM, THENEW[1:KTHM + 1])
        for KTH in range(1, KTHM + 1):
            TTBL[KTH - 1, KP - 1] = TNEW[KTH - 1]

    # ---- table 300: TTBLQ (fine) ----
    KTHM = JTBQ
    KPM = ITBQ
    KTHM1 = KTHM - 1
    KPM1 = KPM - 1
    DTH = (THHQ - THL) / float(KTHM - 1)
    DP = (PH - PLQ) / float(KPM - 1)
    P = PLQ - DP
    for KP in range(1, KPM + 1):
        P = P + DP
        TH = THL - DTH
        TOLDQ = np.zeros(KTHM + 1)
        THEOLDQ = np.zeros(KTHM + 1)
        for KTH in range(1, KTHM + 1):
            TH = TH + DTH
            APE = (1.0e5 / P) ** (RD / CP)
            DENOM = TH - A4 * APE
            if DENOM > EPS:
                QS = PQ0 / P * np.exp(A2 * (TH - A3 * APE) / DENOM)
            else:
                QS = 0.0
            TOLDQ[KTH] = TH / APE
            THEOLDQ[KTH] = TH * np.exp(ELOCP * QS / TOLDQ[KTH])
        THE0K = THEOLDQ[1]
        STHEK = THEOLDQ[KTHM] - THEOLDQ[1]
        THEOLDQ[1] = 0.0
        THEOLDQ[KTHM] = 1.0
        for KTH in range(2, KTHM1 + 1):
            THEOLDQ[KTH] = (THEOLDQ[KTH] - THE0K) / STHEK
            if (THEOLDQ[KTH] - THEOLDQ[KTH - 1]) < EPS:
                THEOLDQ[KTH] = THEOLDQ[KTH - 1] + EPS
        THE0Q[KP - 1] = THE0K
        STHEQ[KP - 1] = STHEK
        THENEWQ = np.zeros(KTHM + 1)
        THENEWQ[1] = 0.0
        THENEWQ[KTHM] = 1.0
        DTHE = 1.0 / float(KTHM - 1)
        for KTH in range(2, KTHM1 + 1):
            THENEWQ[KTH] = THENEWQ[KTH - 1] + DTHE
        TNEWQ = wrf_spline(KTHM, THEOLDQ[1:KTHM + 1], TOLDQ[1:KTHM + 1], KTHM, THENEWQ[1:KTHM + 1])
        for KTH in range(1, KTHM + 1):
            TTBLQ[KTH - 1, KP - 1] = TNEWQ[KTH - 1]

    return dict(QS0=QS0, SQS=SQS, THE0=THE0, STHE=STHE, THE0Q=THE0Q, STHEQ=STHEQ,
                PTBL=PTBL, TTBL=TTBL, TTBLQ=TTBLQ)


if __name__ == "__main__":
    t = build_tables()
    for k, v in t.items():
        print(f"{k}: shape={v.shape} min={v.min():.6g} max={v.max():.6g} mean={v.mean():.6g}")
    print("PTBL[0,0]", t["PTBL"][0, 0], "PTBL[-1,-1]", t["PTBL"][-1, -1])
    print("TTBL[0,0]", t["TTBL"][0, 0], "TTBL[-1,-1]", t["TTBL"][-1, -1])
    print("QS0[:5]", t["QS0"][:5])
    print("THE0[:5]", t["THE0"][:5])
