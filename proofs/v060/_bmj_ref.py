"""Standalone fp64 NumPy reference port of BMJ deep+shallow for one column.

Faithful, explicit 1-based loop translation of SUBROUTINE BMJ. Used ONLY to
diff intermediate quantities against the JAX port and the WRF oracle to localize
algorithmic discrepancies. Not part of the shipped scheme.
"""
import json
import sys
import numpy as np

sys.path.insert(0, "src")
from gpuwrf.physics import cumulus_bmj as M  # noqa

QS0 = np.asarray(M._QS0)
SQS = np.asarray(M._SQS)
THE0 = np.asarray(M._THE0)
STHE = np.asarray(M._STHE)
THE0Q = np.asarray(M._THE0Q)
STHEQ = np.asarray(M._STHEQ)
PTBL = np.asarray(M._PTBL)
TTBL = np.asarray(M._TTBL)
TTBLQ = np.asarray(M._TTBLQ)

CP = M.CP; RD = M.RD; G = M.G; ELWV = M.ELWV; TFRZ = M.TFRZ; D608 = M.D608
PQ0 = M.PQ0; A2 = M.A2; A3 = M.A3; A4 = M.A4; EPSQ = M.EPSQ; ELIWV = M.ELIWV
ROW = M.ROW
EFIMN = M.EFIMN; EFMNT = M.EFMNT; EFIFC = M.EFIFC; AVGEFI = M.AVGEFI; STEFI = M.STEFI
ELEVFC = M.ELEVFC; STABS = M.STABS; STABDS = M.STABDS; DTSHAL = M.DTSHAL
TREL = M.TREL; PONE = M.PONE; PQM = M.PQM; PNO = M.PNO; PSH = M.PSH; PSHU = M.PSHU
PFRZ = M.PFRZ; RHLSC = M.RHLSC; RHHSC = M.RHHSC; EPSDN = M.EPSDN; EPSDT = M.EPSDT
EPSNTP = M.EPSNTP; EPSPR = M.EPSPR; RSFCP = M.RSFCP
DSPBSL = M.DSPBSL; DSP0SL = M.DSP0SL; DSPTSL = M.DSPTSL
DSPBSS = M.DSPBSS; DSP0SS = M.DSP0SS; DSPTSS = M.DSPTSS
SLOPBL = M.SLOPBL; SLOP0L = M.SLOP0L; SLOPTL = M.SLOPTL
SLOPBS = M.SLOPBS; SLOP0S = M.SLOP0S; SLOPTS = M.SLOPTS
SLOPST = M.SLOPST; SLOPE = M.SLOPE
ITB = M.ITB; JTB = M.JTB; ITBQ = M.ITBQ; JTBQ = M.JTBQ
PL = M.PL; PLQ = M.PLQ; PH = M.PH; THL = M.THL
RDP = M.RDP; RDPQ = M.RDPQ; RDQ = M.RDQ; RDTH = M.RDTH; RDTHE = M.RDTHE; RDTHEQ = M.RDTHEQ
CAPA = RD / CP; ELOCP = ELIWV / CP; A23M4L = A2 * (A3 - A4) * ELWV; RCP = 1.0 / CP
ITREFI_MAX = 3
DTTOP = 0.0
DTPtrigr = 0.0


def aint(x):
    return np.trunc(x)


def ttblex(itbx, jtbx, plx, prsmid, rdpx, rdthex, sthe, the0, thesp, ttbl):
    PK = prsmid
    TPK = (PK - plx) * rdpx
    QQ = TPK - aint(TPK)
    IPTB = int(aint(TPK)) + 1
    if IPTB < 1:
        IPTB = 1; QQ = 0.0
    if IPTB >= itbx:
        IPTB = itbx - 1; QQ = 0.0
    i0 = IPTB - 1
    BTHK = (the0[i0 + 1] - the0[i0]) * QQ + the0[i0]
    STHK = (sthe[i0 + 1] - sthe[i0]) * QQ + sthe[i0]
    TTHK = (thesp - BTHK) / STHK * rdthex
    PP = TTHK - aint(TTHK)
    ITHTB = int(aint(TTHK)) + 1
    if ITHTB < 1:
        ITHTB = 1; PP = 0.0
    if ITHTB >= jtbx:
        ITHTB = jtbx - 1; PP = 0.0
    j0 = ITHTB - 1
    T00K = ttbl[j0, i0]; T10K = ttbl[j0 + 1, i0]
    T01K = ttbl[j0, i0 + 1]; T11K = ttbl[j0 + 1, i0 + 1]
    return T00K + (T10K - T00K) * PP + (T01K - T00K) * QQ + (T00K - T10K - T01K + T11K) * PP * QQ


def ttblex_auto(prsmid, thesp):
    if prsmid < PLQ:
        return ttblex(ITB, JTB, PL, prsmid, RDP, RDTHE, STHE, THE0, thesp, TTBL)
    return ttblex(ITBQ, JTBQ, PLQ, prsmid, RDPQ, RDTHEQ, STHEQ, THE0Q, thesp, TTBLQ)


def run_bmj(T, Q, PRSMID, DPRS, PSFC, SM, CLDEFI, DTCNVC):
    """1-based arrays: pass length nz arrays; index via L-1. Returns dict."""
    nz = len(T)
    LMH = nz
    RDTCNVC = 1.0 / DTCNVC
    TAUK = DTCNVC / TREL
    TAUKSC = DTCNVC / (1.0 * TREL)
    DEPMIN = PSH * PSFC * RSFCP

    def Tg(L): return T[L - 1]
    def Qg(L): return Q[L - 1]
    def Pg(L): return PRSMID[L - 1]
    def DPg(L): return DPRS[L - 1]

    APE = np.array([(1.0e5 / PRSMID[k]) ** CAPA for k in range(nz)])
    def APEg(L): return APE[L - 1]

    PLMH = Pg(LMH)
    PELEVFC = PLMH * ELEVFC
    PBTmx = Pg(LMH) - PONE
    CAPEcnv = 0.0; PSPcnv = 0.0; THBTcnv = 0.0
    LBOTcnv = LMH; LTOPcnv = LMH
    CPEcnv = np.zeros(nz + 1); DTVcnv = np.zeros(nz + 1); THEScnv = np.zeros(nz + 1)

    for KB in range(LMH, 0, -1):
        PKL = Pg(KB)
        if PKL < PELEVFC:
            break
        LBOT = LMH; LTOP = LMH
        QBT = Qg(KB)
        THBT = Tg(KB) * APEg(KB)
        TTH = (THBT - THL) * RDTH
        QQ1 = TTH - aint(TTH); ITTB = int(aint(TTH)) + 1
        if ITTB < 1: ITTB = 1; QQ1 = 0.0
        elif ITTB >= JTB: ITTB = JTB - 1; QQ1 = 0.0
        it0 = ITTB - 1
        BQ = (QS0[it0 + 1] - QS0[it0]) * QQ1 + QS0[it0]
        SQ = (SQS[it0 + 1] - SQS[it0]) * QQ1 + SQS[it0]
        TQ = (QBT - BQ) / SQ * RDQ
        PP1 = TQ - aint(TQ); IQTB = int(aint(TQ)) + 1
        if IQTB < 1: IQTB = 1; PP1 = 0.0
        elif IQTB >= ITB: IQTB = ITB - 1; PP1 = 0.0
        iq0 = IQTB - 1
        P00K = PTBL[iq0, it0]; P10K = PTBL[iq0 + 1, it0]
        P01K = PTBL[iq0, it0 + 1]; P11K = PTBL[iq0 + 1, it0 + 1]
        PSP = P00K + (P10K - P00K) * PP1 + (P01K - P00K) * QQ1 + (P00K - P10K - P01K + P11K) * PP1 * QQ1
        APES = (1.0e5 / PSP) ** CAPA
        THESP = THBT * np.exp(ELOCP * QBT * APES / THBT)

        for L in range(1, LMH):
            P = Pg(L)
            if P < PSP and P >= PQM:
                LBOT = L + 1
        PBOT = Pg(LBOT)
        if PBOT >= PBTmx or LBOT >= LMH:
            for L in range(1, LMH):
                P = Pg(L)
                if P < PBTmx:
                    LBOT = L
            PBOT = Pg(LBOT)

        LTOP = LBOT; PTOP = PBOT
        THES = np.full(nz + 1, THESP)

        CPE = np.zeros(nz + 1); DTV = np.zeros(nz + 1)
        DENTPY = 0.0
        L = KB; PLO = Pg(L); TRMLO = 0.0
        CAPEtrigr = DTPtrigr / Tg(LBOT)
        broke = False
        if KB > LBOT:
            for L in range(KB - 1, LBOT, -1):
                PUP = Pg(L)
                TUP = THBT / APEg(L)
                DP = PLO - PUP
                TRMUP = (TUP * (QBT * 0.608 + 1.) - Tg(L) * (Qg(L) * 0.608 + 1.)) * 0.5 / (Tg(L) * (Qg(L) * 0.608 + 1.))
                DTV[L] = TRMLO + TRMUP
                DENTPY = DTV[L] * DP + DENTPY
                CPE[L] = DENTPY
                if DENTPY < CAPEtrigr:
                    broke = True; break
                PLO = PUP; TRMLO = TRMUP
        else:
            L = LBOT + 1
            PLO = Pg(L)
            TUP = THBT / APEg(L)
            TRMLO = (TUP * (QBT * 0.608 + 1.) - Tg(L) * (Qg(L) * 0.608 + 1.)) * 0.5 / (Tg(L) * (Qg(L) * 0.608 + 1.))

        if not broke:
            # at cloud base
            L = LBOT
            PUP = PSP; TUP = THBT / APES
            TSP = (Tg(L + 1) - Tg(L)) / (PLO - PBOT) * (PUP - PBOT) + Tg(L)
            QSP = (Qg(L + 1) - Qg(L)) / (PLO - PBOT) * (PUP - PBOT) + Qg(L)
            DP = PLO - PUP
            TRMUP = (TUP * (QBT * 0.608 + 1.) - TSP * (QSP * 0.608 + 1.)) * 0.5 / (TSP * (QSP * 0.608 + 1.))
            DTV[L] = TRMLO + TRMUP
            DENTPY = DTV[L] * DP + DENTPY
            CPE[L] = DENTPY
            DTV[L] = DTV[L] * DP
            PLO = PUP; TRMLO = TRMUP
            PUP = Pg(L)
            TUP = ttblex_auto(PUP, THES[L])
            QUP = PQ0 / PUP * np.exp(A2 * (TUP - A3) / (TUP - A4))
            QWAT = QBT - QUP
            DP = PLO - PUP
            TRMUP = (TUP * (QUP * 0.608 + 1. - QWAT) - Tg(L) * (Qg(L) * 0.608 + 1.)) * 0.5 / (Tg(L) * (Qg(L) * 0.608 + 1.))
            DENTPY = (TRMLO + TRMUP) * DP + DENTPY
            CPE[L] = DENTPY
            DTV[L] = (DTV[L] + (TRMLO + TRMUP) * DP) / (Pg(LBOT + 1) - Pg(LBOT))
            if DENTPY < CAPEtrigr:
                broke = True
            else:
                PLO = PUP; TRMLO = TRMUP
                for L in range(LBOT - 1, 0, -1):
                    PUP = Pg(L)
                    TUP = ttblex_auto(PUP, THES[L])
                    QUP = PQ0 / PUP * np.exp(A2 * (TUP - A3) / (TUP - A4))
                    QWAT = QBT - QUP
                    DP = PLO - PUP
                    TRMUP = (TUP * (QUP * 0.608 + 1. - QWAT) - Tg(L) * (Qg(L) * 0.608 + 1.)) * 0.5 / (Tg(L) * (Qg(L) * 0.608 + 1.))
                    DTV[L] = TRMLO + TRMUP
                    DENTPY = DTV[L] * DP + DENTPY
                    CPE[L] = DENTPY
                    if DENTPY < CAPEtrigr:
                        broke = True; break
                    PLO = PUP; TRMLO = TRMUP

        LTP1 = KB; CAPE = 0.0
        for L in range(KB, 0, -1):
            if CPE[L] < CAPEtrigr:
                break
            elif CPE[L] > CAPE:
                LTP1 = L; CAPE = CPE[L]
        LTOP = min(LTP1, LBOT)

        if CAPE > CAPEcnv:
            CAPEcnv = CAPE; PSPcnv = PSP; THBTcnv = THBT
            LBOTcnv = LBOT; LTOPcnv = LTOP
            CPEcnv = CPE.copy(); DTVcnv = DTV.copy(); THEScnv = THES.copy()

    if CAPEcnv > 0.:
        PSP = PSPcnv; THBT = THBTcnv; LBOT = LBOTcnv; LTOP = LTOPcnv
        PBOT = Pg(LBOT); PTOP = Pg(LTOP)
        CPE = CPEcnv; DTV = DTVcnv; THES = THEScnv

    out = dict()
    if PTOP > PBOT - PNO or LTOP > LBOT - 2 or CAPEcnv <= 0.:
        out.update(regime="nonconvective", LBOT=0, LTOP=nz, PCPCOL=0.0,
                   DTDT=np.zeros(nz), DQDT=np.zeros(nz),
                   CLDEFI=AVGEFI * SM + STEFI * (1. - SM))
        return out

    DEPTH = PBOT - PTOP
    if DEPTH >= DEPMIN:
        DEEP = True; SHALLOW = False
    else:
        DEEP = False; SHALLOW = True

    if DEEP:
        return deep(T, Q, PRSMID, DPRS, APE, PSFC, SM, CLDEFI, TAUK, RDTCNVC,
                    LBOT, LTOP, THES, DTV, PSP, THBT, nz, TREL)
    else:
        return shallow(T, Q, PRSMID, DPRS, APE, PSFC, SM, CLDEFI, TAUKSC, RDTCNVC,
                       LBOT, LTOP, PBOT, PTOP, PSP, THBT, CPE, DTV, nz)


def deep(T, Q, PRSMID, DPRS, APE, PSFC, SM, CLDEFI, TAUK, RDTCNVC,
         LBOT, LTOP, THES, DTV, PSP, THBT, nz, TRELv):
    LMH = nz
    def Pg(L): return PRSMID[L - 1]
    LB = LBOT; EFI = CLDEFI
    TK = np.zeros(nz + 1); QK = np.zeros(nz + 1); PK = np.zeros(nz + 1)
    APEK = np.zeros(nz + 1); TREFK = np.zeros(nz + 1); QREFK = np.zeros(nz + 1)
    THERK = np.zeros(nz + 1); TREF = np.zeros(nz + 1); EL = np.zeros(nz + 1)
    DIFT = np.zeros(nz + 1); DIFQ = np.zeros(nz + 1); PSK = np.zeros(nz + 1)
    APESK = np.zeros(nz + 1); THSK = np.zeros(nz + 1)
    for L in range(1, LMH + 1):
        TK[L] = T[L - 1]; TREFK[L] = T[L - 1]
        QK[L] = Q[L - 1]; QREFK[L] = Q[L - 1]
        PK[L] = PRSMID[L - 1]; APEK[L] = APE[L - 1]
        TREF[L] = ttblex_auto(PK[L], THES[L])
        THERK[L] = TREF[L] * APEK[L]

    LBM1 = LB - 1; PKB = PK[LB]; PKT = PK[LTOP]
    STABDL = (EFI - EFIMN) * SLOPST + STABDS
    EL[LB] = ELWV
    L0 = LB; PK0 = PK[LB]
    TREFKX = TREFK[LB]; THERKX = THERK[LB]; APEKXX = APEK[LB]
    THERKY = THERK[LBM1]; APEKXY = APEK[LBM1]
    hit_frz = False
    for L in range(LBM1, LTOP - 1, -1):
        if TK[L + 1] < TFRZ:
            hit_frz = True; break
        TREFKX = ((THERKY - THERKX) * STABDL + TREFKX * APEKXX) / APEKXY
        TREFK[L] = TREFKX
        EL[L] = ELWV
        APEKXX = APEKXY; THERKX = THERKY
        APEKXY = APEK[L - 1]; THERKY = THERK[L - 1]
        L0 = L; PK0 = PK[L0]
    if hit_frz:
        L0M1 = L0 - 1
        RDP0T = 1. / (PK0 - PKT)
        DTHEM = THERK[L0] - TREFK[L0] * APEK[L0]
        for L in range(LTOP, L0M1 + 1):
            TREFK[L] = (THERK[L] - (PK[L] - PKT) * DTHEM * RDP0T) / APEK[L]
            EL[L] = ELWV

    DEPWL = PKB - PK0
    DEPTHf = PFRZ * PSFC * RSFCP
    SM1 = 1. - SM; PBOTFC = 1.

    for ITREFI in range(1, ITREFI_MAX + 1):
        DSPBK = ((EFI - EFIMN) * SLOPBS + DSPBSS * PBOTFC) * SM + ((EFI - EFIMN) * SLOPBL + DSPBSL * PBOTFC) * SM1
        DSP0K = ((EFI - EFIMN) * SLOP0S + DSP0SS * PBOTFC) * SM + ((EFI - EFIMN) * SLOP0L + DSP0SL * PBOTFC) * SM1
        DSPTK = ((EFI - EFIMN) * SLOPTS + DSPTSS * PBOTFC) * SM + ((EFI - EFIMN) * SLOPTL + DSPTSL * PBOTFC) * SM1
        for L in range(LTOP, LB + 1):
            if DEPWL >= DEPTHf:
                if L < L0:
                    DSP = ((PK0 - PK[L]) * DSPTK + (PK[L] - PKT) * DSP0K) / (PK0 - PKT)
                else:
                    DSP = ((PKB - PK[L]) * DSP0K + (PK[L] - PK0) * DSPBK) / (PKB - PK0)
            else:
                DSP = DSP0K
                if L < L0:
                    DSP = ((PK0 - PK[L]) * DSPTK + (PK[L] - PKT) * DSP0K) / (PK0 - PKT)
            PSK[L] = PK[L] + DSP
            APESK[L] = (1.0e5 / PSK[L]) ** CAPA
            if PK[L] > PQM:
                THSK[L] = TREFK[L] * APEK[L]
                QREFK[L] = PQ0 / PSK[L] * np.exp(A2 * (THSK[L] - A3 * APESK[L]) / (THSK[L] - A4 * APESK[L]))
            else:
                QREFK[L] = QK[L]

        for ITER in range(1, 3):
            SUMDE = 0.; SUMDP = 0.; DHDT = 0.
            for L in range(LTOP, LB + 1):
                SUMDE = ((TK[L] - TREFK[L]) * CP + (QK[L] - QREFK[L]) * EL[L]) * DPRS[L - 1] + SUMDE
                DHDT = (QREFK[L] * A23M4L / ((TREFK[L] * APEK[L] / APESK[L]) - A4) ** 2 + CP) * DPRS[L - 1] + DHDT
                SUMDP = SUMDP + DPRS[L - 1]
            HCORR = SUMDE / (SUMDP - DPRS[LTOP - 1])
            DHDT = DHDT / (SUMDP - DPRS[LTOP - 1])
            LCOR = LTOP + 1
            LQM = 1
            for L in range(1, LB + 1):
                if PK[L] <= PQM:
                    LQM = L
            if LCOR <= LQM:
                for L in range(LCOR, LQM + 1):
                    TREFK[L] = TREFK[L] + HCORR * RCP
                LCOR = LQM + 1
            for L in range(LCOR, LB + 1):
                TREFK[L] = HCORR / DHDT + TREFK[L]
                THSKL = TREFK[L] * APEK[L]
                QREFK[L] = PQ0 / PSK[L] * np.exp(A2 * (THSKL - A3 * APESK[L]) / (THSKL - A4 * APESK[L]))

        AVRGT = 0.; PRECK = 0.; PDQD = 0.; PRWD = 0.; DSQ = 0.; DST = 0.
        SUMDP = 0.
        for L in range(LTOP, LB + 1):
            TKL = TK[L]
            DIFTL = (TREFK[L] - TKL) * TAUK
            DIFQL = (QREFK[L] - QK[L]) * TAUK
            AVRGTL = (TKL + TKL + DIFTL)
            DPOT = DPRS[L - 1] / AVRGTL
            DST = DIFTL * DPOT + DST
            DSQ = DIFQL * EL[L] * DPOT + DSQ
            AVRGT = AVRGTL * DPRS[L - 1] + AVRGT
            PRECK = DIFTL * DPRS[L - 1] + PRECK
            PDQD = (QK[L] - QREFK[L]) * DPRS[L - 1] + PDQD
            PRWD = QK[L] * DPRS[L - 1] + PRWD
            DIFT[L] = DIFTL; DIFQ[L] = DIFQL
            SUMDP = SUMDP + DPRS[L - 1]
        DST = (DST + DST) * CP
        DSQ = DSQ + DSQ
        DENTPY = DST + DSQ
        AVRGT = AVRGT / (SUMDP + SUMDP)
        DRHEAT = (PRECK * SM + max(1.e-7, PRECK) * (1. - SM)) * CP / AVRGT
        DRHEAT = max(DRHEAT, 1.e-20)
        EFI = EFIFC * DENTPY / DRHEAT
        EFI = min(EFI, 1.); EFI = max(EFI, EFIMN)

    CPRLG = CP / (ROW * G * ELWV)
    if DENTPY >= EPSNTP and PRECK > EPSPR:
        CLDEFI = EFI
        FEFI = EFMNT + SLOPE * (EFI - EFIMN)
        FEFI = (DENTPY - EPSNTP) * FEFI / DENTPY
        PRECK = PRECK * FEFI
        CUP = PRECK * CPRLG; PCPCOL = CUP
        DTDT = np.zeros(nz); DQDT = np.zeros(nz)
        for L in range(LTOP, LB + 1):
            DTDT[L - 1] = DIFT[L] * FEFI * RDTCNVC
            DQDT[L - 1] = DIFQ[L] * FEFI * RDTCNVC
        return dict(regime="deep", LBOT=LB, LTOP=LTOP, PCPCOL=PCPCOL,
                    DTDT=DTDT, DQDT=DQDT, CLDEFI=CLDEFI, DENTPY=DENTPY, PRECK=PRECK,
                    FEFI=FEFI, EFI=EFI, L0=L0, hit_frz=hit_frz,
                    TREFK=TREFK[1:LMH + 1].copy())
    else:
        return dict(regime="deep_demoted", LBOT=LB, LTOP=LTOP, PCPCOL=0.0,
                    DTDT=np.zeros(nz), DQDT=np.zeros(nz), CLDEFI=EFIMN * SM + STEFI * (1. - SM))


def shallow(*a, **k):
    return dict(regime="shallow_stub")


if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    d = json.load(open(f"proofs/v060/savepoints/bmj_case_{cid}.json"))
    sc = d["scalars"]; cols = d["columns"]
    nz = len(cols["T"])
    T = np.array(cols["T"], float); QVm = np.array(cols["QV"], float)
    P = np.array(cols["P"], float); DZ = np.array(cols["DZ"], float)
    RHO = np.array(cols["RHO"], float); PI = np.array(cols["PI"], float)
    PINT = np.array(cols["PINT"], float)
    DT = float(sc["DT"]); STEPCU = int(sc["STEPCU"]); XLAND = float(sc["XLAND"])
    SM = XLAND - 1.0
    Qspec = np.maximum(EPSQ, QVm / (1. + QVm))
    # flip top-down
    Tf = T[::-1].copy(); Qf = Qspec[::-1].copy(); Pf = P[::-1].copy()
    DPRSf = (RHO * G * DZ)[::-1].copy()
    PSFC = PINT[0]
    DTCNVC = DT * STEPCU
    res = run_bmj(Tf, Qf, Pf, DPRSf, PSFC, SM, 0.6, DTCNVC)
    print("regime", res["regime"], "LBOT", res["LBOT"], "LTOP", res["LTOP"])
    print("PCPCOL", res.get("PCPCOL"))
    RAINCV = res.get("PCPCOL", 0.0) * 1.e3 / STEPCU
    print("RAINCV ref-py", RAINCV, "oracle", sc["RAINCV"], "rel", abs(RAINCV - sc["RAINCV"]) / abs(sc["RAINCV"]))
    print("DENTPY", res.get("DENTPY"), "PRECK", res.get("PRECK"), "FEFI", res.get("FEFI"), "EFI", res.get("EFI"))
    print("L0", res.get("L0"), "hit_frz", res.get("hit_frz"))
    # tendencies back to bottom-up
    DTDTf = res["DTDT"]; DQDTf = res["DQDT"]
    DTDT = DTDTf[::-1]; DQDT = DQDTf[::-1]
    rth = DTDT / PI
    rqv = DQDT / (1. - Qspec) ** 2
    rth_ref = np.array(cols["RTHCUTEN"]); rqv_ref = np.array(cols["RQVCUTEN"])
    print("RTHCUTEN max abs err", np.abs(rth - rth_ref).max(),
          "rel", np.abs(rth - rth_ref).max() / max(np.abs(rth_ref).max(), 1e-30))
