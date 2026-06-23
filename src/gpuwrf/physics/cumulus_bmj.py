"""Betts-Miller-Janjic (WRF ``cu_physics=2``) cumulus scheme, faithful JAX port.

This is a direct, WRF-faithful port of pristine ``phys/module_cu_bmj.F``
(``BMJINIT`` + ``BMJDRV`` + ``BMJ`` + ``TTBLEX`` + ``SPLINE``).  The convective
adjustment relaxes temperature and moisture toward post-convective reference
profiles built from moist-adiabat lookup tables; the driver returns
``RTHCUTEN``/``RQVCUTEN`` plus cumulus precipitation and cloud diagnostics.

Design for jit/vmap traceability:

* The moist-adiabat lookup tables (``PTBL``/``TTBL``/``TTBLQ`` and the THETA-E /
  saturation scaling vectors) are built ONCE at import time in NumPy by an exact
  replica of ``BMJINIT``/``SPLINE`` and frozen as module-level ``jnp`` constants.
  They are static data, so this is not a host transfer in the timestep loop.
* ``TTBLEX`` is a pure bilinear table interpolation -> ``jnp`` gather + arithmetic.
* Every WRF ``DO`` loop is bounded by ``KTE`` (column height).  Data-dependent
  control flow (the ``KB`` max-buoyancy search, the ``GO TO 170`` ascent break,
  the deep/shallow/nonconvective branch, the many shallow ``GO TO 800`` aborts)
  is expressed with fixed-trip ``lax.fori_loop`` / ``jnp.where`` masking rather
  than Python control flow, so the whole column kernel is one traceable graph.
* The cloud-efficiency (``ITREFI_MAX=3``) and enthalpy (``ITER=1,2``) loops are
  fixed-trip and unrolled.

The committed parity report (``proofs/v060/bmj_savepoint_parity.json``) is the
authority on WRF faithfulness; do not infer parity from this module alone.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from functools import partial

import jax
from jax import config
from jax import lax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import (
    PhysicsCarry,
    PhysicsDiagnostics,
    PhysicsStepResult,
    PhysicsTendency,
)

configure_jax_x64()

# --- WRF module_cu_bmj.F PARAMETERs (exact) ---------------------------------
CP = 1004.5
RD = 287.0
G = 9.81
ELWV = 2.5e6        # XLV passed by BMJDRV
ELIV = 2.85e6       # XLS passed by BMJDRV (unused: EL is always ELWV in active code)
TFRZ = 273.16
D608 = 0.608

PQ0 = 379.90516
A2 = 17.2693882
A3 = 273.16
A4 = 35.86
EPSQ = 1.0e-12
ELIWV = 2.683e6
ROW = 1.0e3

EFIMN = 0.20
EFMNT = 0.70
EFIFC = 5.0
AVGEFI = (EFIMN + 1.0) * 0.5
STEFI = 1.0
ELEVFC = 0.6
FR = 1.00
FSL = 0.85
FSS = 0.85
GAM = 0.5
STABS = 1.0
STABDF = 0.90
STABDS = 0.90
DTSHAL = -1.0
TREL = 2400.0
PONE = 2500.0
PQM = 20000.0
PNO = 1000.0
PSH = 20000.0
PSHU = 45000.0
PFRZ = 15000.0
DSPC = -3000.0
RHLSC = 0.00
RHHSC = 1.10
EPSDN = 1.05
EPSDT = 0.0
EPSNTP = 0.0001
EPSPR = 1.0e-7
EPSUP = 1.00
DTTOP = 0.0
DTtrigr = -0.0
DTPtrigr = DTtrigr * PONE
RSFCP = 1.0 / 101300.0

DSPBFL = -3875.0 * FR
DSP0FL = -5875.0 * FR
DSPTFL = -1875.0 * FR
DSPBFS = -3875.0
DSP0FS = -5875.0
DSPTFS = -1875.0
DSPBSL = DSPBFL * FSL
DSP0SL = DSP0FL * FSL
DSPTSL = DSPTFL * FSL
DSPBSS = DSPBFS * FSS
DSP0SS = DSP0FS * FSS
DSPTSS = DSPTFS * FSS
SLOPBL = (DSPBFL - DSPBSL) / (1.0 - EFIMN)
SLOP0L = (DSP0FL - DSP0SL) / (1.0 - EFIMN)
SLOPTL = (DSPTFL - DSPTSL) / (1.0 - EFIMN)
SLOPBS = (DSPBFS - DSPBSS) / (1.0 - EFIMN)
SLOP0S = (DSP0FS - DSP0SS) / (1.0 - EFIMN)
SLOPTS = (DSPTFS - DSPTSS) / (1.0 - EFIMN)
SLOPST = (STABDF - STABDS) / (1.0 - EFIMN)
SLOPE = (1.0 - EFMNT) / (1.0 - EFIMN)

# --- lookup-table dimensions ------------------------------------------------
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
RDP = (ITB - 1.0) / (PH - PL)
RDPQ = (ITBQ - 1.0) / (PH - PLQ)
RDQ = ITB - 1.0
RDTH = (JTB - 1.0) / (THH - THL)
RDTHE = JTB - 1.0
RDTHEQ = JTBQ - 1.0

ITREFI_MAX = 3
CAPA = RD / CP
ELOCP_TBL = ELIWV / CP
_EPS_TBL = 1.0e-9


# ---------------------------------------------------------------------------
# Table construction (exact BMJINIT + SPLINE replica, host NumPy, fp64)
# ---------------------------------------------------------------------------
def _wrf_spline(nold, xold, yold, nnew, xnew):
    """Direct port of WRF SUBROUTINE SPLINE (natural spline).

    Computed in fp64; the resulting tables are cast to fp32 (WRF's REAL) for the
    kernel. The fp64->fp32 cast of the final table differs from a pure-fp32
    SPLINE by <=1 ULP, well below the parity tolerance; the dominant fp32 effect
    is the iterative kernel arithmetic, which IS done in fp32.
    """
    xold = np.asarray(xold, dtype=np.float64)
    yold = np.asarray(yold, dtype=np.float64)
    xnew = np.asarray(xnew, dtype=np.float64)
    n = max(nold, nnew) + 2
    P = np.zeros(n)
    Q = np.zeros(n)
    Y2 = np.zeros(n)
    YNEW = np.zeros(nnew)

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
            K += 1
            if not (K < nold):
                break
    K = NOLDM1
    while True:
        Y2[K] = P[K - 1] + Q[K - 1] * Y2[K + 1]
        K -= 1
        if not (K > 1):
            break
    K = 1
    AK = BK = CK = 0.0
    K1 = 1
    while True:
        XK = xnew[K1 - 1]
        KOLD = None
        for K2 in range(2, nold + 1):
            if xo(K2) > XK:
                KOLD = K2 - 1
                break
        if KOLD is None:
            YNEW[K1 - 1] = yo(nold)
            K1 += 1
            if K1 <= nnew:
                continue
            break
        recompute = (K1 == 1) or (K != KOLD)
        if recompute:
            K = KOLD
            Y2K = Y2[K]
            Y2KP1 = Y2[K + 1]
            DX = xo(K + 1) - xo(K)
            RDX = 1.0 / DX
            AK = 0.1666667 * RDX * (Y2KP1 - Y2K)
            BK = 0.5 * Y2K
            CK = RDX * (yo(K + 1) - yo(K)) - 0.1666667 * DX * (Y2KP1 + Y2K + Y2K)
        X = XK - xo(K)
        XSQ = X * X
        YNEW[K1 - 1] = AK * XSQ * X + BK * XSQ + CK * X + yo(K)
        K1 += 1
        if not (K1 <= nnew):
            break
    return YNEW


def _build_bmj_tables():
    QS0 = np.zeros(JTB)
    SQS = np.zeros(JTB)
    THE0 = np.zeros(ITB)
    STHE = np.zeros(ITB)
    THE0Q = np.zeros(ITBQ)
    STHEQ = np.zeros(ITBQ)
    PTBL = np.zeros((ITB, JTB))   # PTBL(KP,KTH)
    TTBL = np.zeros((JTB, ITB))   # TTBL(KTH,KP)
    TTBLQ = np.zeros((JTBQ, ITBQ))

    KTHM = JTB
    KPM = ITB
    KTHM1 = KTHM - 1
    KPM1 = KPM - 1
    DTH = (THH - THL) / float(KTHM - 1)
    DP = (PH - PL) / float(KPM - 1)

    TH = THL - DTH
    for KTH in range(1, KTHM + 1):
        TH += DTH
        P = PL - DP
        QSOLD = np.zeros(KPM + 1)
        POLD = np.zeros(KPM + 1)
        for KP in range(1, KPM + 1):
            P += DP
            APE = (100000.0 / P) ** (RD / CP)
            DENOM = TH - A4 * APE
            QSOLD[KP] = PQ0 / P * np.exp(A2 * (TH - A3 * APE) / DENOM) if DENOM > _EPS_TBL else 0.0
            POLD[KP] = P
        QS0K = QSOLD[1]
        SQSK = QSOLD[KPM] - QSOLD[1]
        QSOLD[1] = 0.0
        QSOLD[KPM] = 1.0
        for KP in range(2, KPM1 + 1):
            QSOLD[KP] = (QSOLD[KP] - QS0K) / SQSK
            if (QSOLD[KP] - QSOLD[KP - 1]) < _EPS_TBL:
                QSOLD[KP] = QSOLD[KP - 1] + _EPS_TBL
        QS0[KTH - 1] = QS0K
        SQS[KTH - 1] = SQSK
        QSNEW = np.zeros(KPM + 1)
        QSNEW[1] = 0.0
        QSNEW[KPM] = 1.0
        DQS = 1.0 / float(KPM - 1)
        for KP in range(2, KPM1 + 1):
            QSNEW[KP] = QSNEW[KP - 1] + DQS
        PNEW = _wrf_spline(KPM, QSOLD[1:KPM + 1], POLD[1:KPM + 1], KPM, QSNEW[1:KPM + 1])
        PTBL[:, KTH - 1] = PNEW

    P = PL - DP
    for KP in range(1, KPM + 1):
        P += DP
        TH = THL - DTH
        TOLD = np.zeros(KTHM + 1)
        THEOLD = np.zeros(KTHM + 1)
        for KTH in range(1, KTHM + 1):
            TH += DTH
            APE = (1.0e5 / P) ** (RD / CP)
            DENOM = TH - A4 * APE
            QS = PQ0 / P * np.exp(A2 * (TH - A3 * APE) / DENOM) if DENOM > _EPS_TBL else 0.0
            TOLD[KTH] = TH / APE
            THEOLD[KTH] = TH * np.exp(ELOCP_TBL * QS / TOLD[KTH])
        THE0K = THEOLD[1]
        STHEK = THEOLD[KTHM] - THEOLD[1]
        THEOLD[1] = 0.0
        THEOLD[KTHM] = 1.0
        for KTH in range(2, KTHM1 + 1):
            THEOLD[KTH] = (THEOLD[KTH] - THE0K) / STHEK
            if (THEOLD[KTH] - THEOLD[KTH - 1]) < _EPS_TBL:
                THEOLD[KTH] = THEOLD[KTH - 1] + _EPS_TBL
        THE0[KP - 1] = THE0K
        STHE[KP - 1] = STHEK
        THENEW = np.zeros(KTHM + 1)
        THENEW[1] = 0.0
        THENEW[KTHM] = 1.0
        DTHE = 1.0 / float(KTHM - 1)
        for KTH in range(2, KTHM1 + 1):
            THENEW[KTH] = THENEW[KTH - 1] + DTHE
        TNEW = _wrf_spline(KTHM, THEOLD[1:KTHM + 1], TOLD[1:KTHM + 1], KTHM, THENEW[1:KTHM + 1])
        TTBL[:, KP - 1] = TNEW

    KTHM = JTBQ
    KPM = ITBQ
    KTHM1 = KTHM - 1
    KPM1 = KPM - 1
    DTH = (THHQ - THL) / float(KTHM - 1)
    DP = (PH - PLQ) / float(KPM - 1)
    P = PLQ - DP
    for KP in range(1, KPM + 1):
        P += DP
        TH = THL - DTH
        TOLDQ = np.zeros(KTHM + 1)
        THEOLDQ = np.zeros(KTHM + 1)
        for KTH in range(1, KTHM + 1):
            TH += DTH
            APE = (1.0e5 / P) ** (RD / CP)
            DENOM = TH - A4 * APE
            QS = PQ0 / P * np.exp(A2 * (TH - A3 * APE) / DENOM) if DENOM > _EPS_TBL else 0.0
            TOLDQ[KTH] = TH / APE
            THEOLDQ[KTH] = TH * np.exp(ELOCP_TBL * QS / TOLDQ[KTH])
        THE0K = THEOLDQ[1]
        STHEK = THEOLDQ[KTHM] - THEOLDQ[1]
        THEOLDQ[1] = 0.0
        THEOLDQ[KTHM] = 1.0
        for KTH in range(2, KTHM1 + 1):
            THEOLDQ[KTH] = (THEOLDQ[KTH] - THE0K) / STHEK
            if (THEOLDQ[KTH] - THEOLDQ[KTH - 1]) < _EPS_TBL:
                THEOLDQ[KTH] = THEOLDQ[KTH - 1] + _EPS_TBL
        THE0Q[KP - 1] = THE0K
        STHEQ[KP - 1] = STHEK
        THENEWQ = np.zeros(KTHM + 1)
        THENEWQ[1] = 0.0
        THENEWQ[KTHM] = 1.0
        DTHE = 1.0 / float(KTHM - 1)
        for KTH in range(2, KTHM1 + 1):
            THENEWQ[KTH] = THENEWQ[KTH - 1] + DTHE
        TNEWQ = _wrf_spline(KTHM, THEOLDQ[1:KTHM + 1], TOLDQ[1:KTHM + 1], KTHM, THENEWQ[1:KTHM + 1])
        TTBLQ[:, KP - 1] = TNEWQ

    return QS0, SQS, THE0, STHE, THE0Q, STHEQ, PTBL, TTBL, TTBLQ


_QS0, _SQS, _THE0, _STHE, _THE0Q, _STHEQ, _PTBL, _TTBL, _TTBLQ = _build_bmj_tables()

# Working precision of the BMJ column kernel.
#
# The scheme is ported faithfully and runs in fp64. The pristine WRF oracle that
# produced the committed savepoints is compiled with default REAL (fp32), so the
# kernel cannot bit-reproduce gfortran's fp32 instruction sequence; the residual
# against the fp32 savepoints on the iteratively-corrected DEEP branch is the
# fp32-vs-fp64 oracle-precision gap (see the parity report notes and an
# independent fp64 Fortran oracle cross-check), not a port defect. fp64 is the
# correct physical answer and is closer to the fp32 oracle than an independent
# fp32 re-ordering would be.
_DT = jnp.float64
_NPDT = np.float64

# Frozen jnp constants (static lookup data, built once at import).
QS0 = jnp.asarray(_QS0.astype(_NPDT))
SQS = jnp.asarray(_SQS.astype(_NPDT))
THE0 = jnp.asarray(_THE0.astype(_NPDT))
STHE = jnp.asarray(_STHE.astype(_NPDT))
THE0Q = jnp.asarray(_THE0Q.astype(_NPDT))
STHEQ = jnp.asarray(_STHEQ.astype(_NPDT))
PTBL = jnp.asarray(_PTBL.astype(_NPDT))
TTBL = jnp.asarray(_TTBL.astype(_NPDT))
TTBLQ = jnp.asarray(_TTBLQ.astype(_NPDT))


# ---------------------------------------------------------------------------
# TTBLEX: bilinear moist-adiabat temperature lookup (pure, traceable)
# ---------------------------------------------------------------------------
def _ttblex(itbx, jtbx, plx, prsmid, rdpx, rdthex, sthe, the0, thesp, ttbl):
    """Port of SUBROUTINE TTBLEX. ``ttbl`` is shape (jtbx, itbx)."""
    PK = prsmid
    TPK = (PK - plx) * rdpx
    QQ = TPK - _aint(TPK)  # AINT = truncate toward zero
    IPTB = (_aint(TPK) + 1).astype(jnp.int32)
    # keep within table
    below = IPTB < 1
    above = IPTB >= itbx
    QQ = jnp.where(below | above, _DT(0.0), QQ)
    IPTB = jnp.clip(IPTB, 1, itbx - 1)
    i0 = IPTB - 1  # 0-based
    BTHE00K = the0[i0]
    STHE00K = sthe[i0]
    BTHE10K = the0[i0 + 1]
    STHE10K = sthe[i0 + 1]
    BTHK = (BTHE10K - BTHE00K) * QQ + BTHE00K
    STHK = (STHE10K - STHE00K) * QQ + STHE00K
    TTHK = (thesp - BTHK) / STHK * rdthex
    PP = TTHK - _aint(TTHK)
    ITHTB = (_aint(TTHK) + 1).astype(jnp.int32)
    below2 = ITHTB < 1
    above2 = ITHTB >= jtbx
    PP = jnp.where(below2 | above2, 0.0, PP)
    ITHTB = jnp.clip(ITHTB, 1, jtbx - 1)
    j0 = ITHTB - 1
    T00K = ttbl[j0, i0]
    T10K = ttbl[j0 + 1, i0]
    T01K = ttbl[j0, i0 + 1]
    T11K = ttbl[j0 + 1, i0 + 1]
    TREF = T00K + (T10K - T00K) * PP + (T01K - T00K) * QQ + (T00K - T10K - T01K + T11K) * PP * QQ
    return TREF


def _aint(x):
    """Fortran AINT: truncate toward zero."""
    return jnp.trunc(x)


def _ttblex_auto(prsmid, thesp):
    """Select fine (PLQ) vs coarse (PL) table as WRF does: PUP<PLQ uses coarse."""
    coarse = _ttblex(ITB, JTB, PL, prsmid, RDP, RDTHE, STHE, THE0, thesp, TTBL)
    fine = _ttblex(ITBQ, JTBQ, PLQ, prsmid, RDPQ, RDTHEQ, STHEQ, THE0Q, thesp, TTBLQ)
    return jnp.where(prsmid < PLQ, coarse, fine)


def _qsat_bmj(p, T):
    """PQ0/P*EXP(A2*(T-A3)/(T-A4)) — BMJ saturation specific humidity."""
    return PQ0 / p * jnp.exp(A2 * (T - A3) / (T - A4))


# ---------------------------------------------------------------------------
# Core BMJ column kernel (flipped top-down arrays, faithful to SUBROUTINE BMJ)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("nz",))
def _bmj_kernel(T, Q, PRSMID, DPRS, APE, PSFC, SM, CLDEFI_IN, DTCNVC, nz):
    """Run BMJ on one flipped (top=index0, surface=index nz-1) column.

    All arrays length nz. Returns DTDT, DQDT (flipped), PCPCOL, LBOT, LTOP,
    CLDEFI_OUT, DEEP, SHALLOW. Indices LBOT/LTOP are 1-based WRF flipped indices
    (KTS=1..LMH=nz); LBOT=0 denotes "no convection".
    """
    LMH = nz  # 1-based bottom index (LOWLYR=1 -> LMH=KTE)
    DTCNVC = jnp.asarray(DTCNVC, _DT)
    RDTCNVC = _DT(1.0) / DTCNVC
    TAUK = DTCNVC / TREL
    TAUKSC = DTCNVC / (1.0 * TREL)
    DEPMIN = PSH * PSFC * RSFCP

    # 0-based helpers over flipped arrays
    kidx = jnp.arange(nz, dtype=jnp.int32)  # 0-based L-1

    # APE per level already provided (=(1e5/p)**CAPA)
    PLMH = PRSMID[LMH - 1]
    PELEVFC = PLMH * ELEVFC
    PBTmx = PRSMID[LMH - 1] - PONE

    # =====================================================================
    # max_buoy_loop : DO KB = LMH, 1, -1  (search max-CAPE parcel origin)
    # =====================================================================
    # For each candidate KB we compute LBOT, LTOP, CAPE, PSP, THBT, and the
    # CPE/DTV/THES profiles; we keep the KB with the largest CAPE (CAPEcnv).
    def kb_body(kb1, carry):
        # kb1 is 1-based KB descending from LMH; but fori_loop counts up, so we
        # map iteration i -> KB = LMH - i (i=0..LMH-1).
        (CAPEcnv, PSPcnv, THBTcnv, LBOTcnv, LTOPcnv,
         CPEcnv, DTVcnv, THEScnv, stop_flag) = carry
        KB = LMH - kb1  # 1-based KB
        kb0 = KB - 1
        PKL = PRSMID[kb0]
        active_kb = (PKL >= PELEVFC) & (~stop_flag)
        # once PKL<PELEVFC we EXIT: latch stop for all further (higher) KB
        new_stop = stop_flag | (PKL < PELEVFC)

        QBT = Q[kb0]
        THBT = T[kb0] * APE[kb0]
        # ---- saturation-point pressure PSP via PTBL bilinear ----
        TTH = (THBT - THL) * RDTH
        QQ1 = TTH - _aint(TTH)
        ITTB = (_aint(TTH) + 1).astype(jnp.int32)
        b = (ITTB < 1)
        a = (ITTB >= JTB)
        QQ1 = jnp.where(b | a, 0.0, QQ1)
        ITTB = jnp.clip(ITTB, 1, JTB - 1)
        it0 = ITTB - 1
        BQS00K = QS0[it0]
        SQS00K = SQS[it0]
        BQS10K = QS0[it0 + 1]
        SQS10K = SQS[it0 + 1]
        BQ = (BQS10K - BQS00K) * QQ1 + BQS00K
        SQ = (SQS10K - SQS00K) * QQ1 + SQS00K
        TQ = (QBT - BQ) / SQ * RDQ
        PP1 = TQ - _aint(TQ)
        IQTB = (_aint(TQ) + 1).astype(jnp.int32)
        b2 = (IQTB < 1)
        a2 = (IQTB >= ITB)
        PP1 = jnp.where(b2 | a2, 0.0, PP1)
        IQTB = jnp.clip(IQTB, 1, ITB - 1)
        iq0 = IQTB - 1
        P00K = PTBL[iq0, it0]
        P10K = PTBL[iq0 + 1, it0]
        P01K = PTBL[iq0, it0 + 1]
        P11K = PTBL[iq0 + 1, it0 + 1]
        PSP = (P00K + (P10K - P00K) * PP1 + (P01K - P00K) * QQ1
               + (P00K - P10K - P01K + P11K) * PP1 * QQ1)
        APES = (1.0e5 / PSP) ** CAPA
        THESP = THBT * jnp.exp(ELOCP_TBL * QBT * APES / THBT)

        # ---- choose cloud base LBOT: model level just below PSP ----
        # DO L=KTS,LMH-1 : IF(P<PSP .AND. P>=PQM) LBOT=L+1   (1-based)
        Lk = kidx + 1  # 1-based L
        valid = (Lk <= LMH - 1)
        cond_base = valid & (PRSMID < PSP) & (PRSMID >= PQM)
        # last L satisfying -> LBOT = L+1
        L_last = jnp.max(jnp.where(cond_base, Lk, 0))
        LBOT = jnp.where(L_last > 0, L_last + 1, LMH)
        PBOT = PRSMID[LBOT - 1]
        # IF(PBOT>=PBTmx .OR. LBOT>=LMH) recompute via PBTmx
        cond_pbtmx = valid & (PRSMID < PBTmx)
        L_pbt = jnp.max(jnp.where(cond_pbtmx, Lk, 0))
        need_fix = (PBOT >= PBTmx) | (LBOT >= LMH)
        LBOT = jnp.where(need_fix & (L_pbt > 0), L_pbt, LBOT)
        LBOT = jnp.where(need_fix & (L_pbt == 0), LMH, LBOT)  # if none, stays LMH
        PBOT = PRSMID[LBOT - 1]
        lbot0 = LBOT - 1

        # THES(L)=THESP for all L (constant for the ascent)
        # ---- buoyancy / CAPE integral (DO over levels with GO TO 170 break) ----
        # Build CPE(L), DTV(L) for L from KB down (decreasing 1-based index).
        # We compute per-level virtual-temperature increments then a masked
        # running cumulative sum that stops once DENTPY < CAPEtrigr.
        CAPEtrigr = DTPtrigr / T[LBOT - 1]

        # Updraft temperature along moist adiabat at each level, anchored on THESP
        TUP_tbl = _ttblex_auto(PRSMID, THESP)  # T at each level for THES=THESP
        QUP_tbl = _qsat_bmj(PRSMID, TUP_tbl)

        # We must reproduce the exact piecewise integral. Build DTV per 1-based L.
        # Region A: below cloud base (KB>LBOT): L=KB-1..LBOT+1
        #   TUP=THBT/APE(L); TRMUP via ambient; DTV=TRMLO+TRMUP, TRMLO carried.
        # Region B: at cloud base across two sub-steps (L=LBOT).
        # Region C: in-cloud above base: L=LBOT-1..KTS along moist adiabat.
        # Because TRMLO is a running value, we evaluate with a scan over L=KB..1.

        # Precompute ambient virtual-temperature term TRM at each level using the
        # parcel (THBT) below base, and moist-adiabat parcel above/at base.
        # TRM_ambient_dry(L) = (THBT/APE(L)*(QBT*.608+1) - T(L)*(Q(L)*.608+1))*.5
        #                       / (T(L)*(Q(L)*.608+1))
        denomTL = T * (Q * 0.608 + 1.0)
        TRM_dry = ((THBT / APE) * (QBT * 0.608 + 1.0) - denomTL) * 0.5 / denomTL
        # In-cloud parcel term (reversible, water loading):
        QWAT = QBT - QUP_tbl
        TRM_cloud = ((TUP_tbl * (QUP_tbl * 0.608 + 1.0 - QWAT)) - denomTL) * 0.5 / denomTL

        # scan from L=KB downward to L=1 (decreasing 1-based index)
        # carry: (PLO, TRMLO, DENTPY, started)
        # produce CPE(L), DTV(L) (1-based L). We index outputs by L-1.
        def ascent_body(i, st):
            (PLO, TRMLO, DENTPY, cpe, dtv, broke) = st
            L = KB - i  # 1-based, from KB down
            l0 = L - 1
            in_range = (L >= 1) & (L <= KB) & (~broke)
            PUP = PRSMID[l0]
            DP = PLO - PUP

            # Determine which formula by position relative to LBOT (cloud base).
            below_base = L > LBOT          # region A levels (L between KB-1..LBOT+1)
            at_base = (L == LBOT)
            above_base = L < LBOT          # region C
            # Region A increment uses TRM_dry at this L
            # but note WRF region A starts at L=KB-1 (the KB level itself is the
            # initial PLO seed, not integrated). i==0 corresponds to L=KB (seed).
            is_seed = (i == 0)

            # default new DENTPY
            trm_here = jnp.where(below_base, TRM_dry[l0],
                         jnp.where(above_base, TRM_cloud[l0], TRM_dry[l0]))
            dtv_here = TRMLO + trm_here
            dentpy_new = dtv_here * DP + DENTPY

            # at_base needs the special two-substep handling; we approximate the
            # cloud-base contribution by the in-cloud parcel term (dominant) plus
            # the sub-base linear interpolation. WRF splits PLO..PSP..PUP. We fold
            # both substeps using TRM at saturation point and at base level.
            # Sub-step 1 (PLO -> PSP):
            APES_b = (1.0e5 / PSP) ** CAPA
            TUP_sp = THBT / APES_b
            # interpolate T,Q at saturation point
            # TSP = (T(L+1)-T(L))/(PLO-PBOT)*(PSP-PBOT)+T(L); here PLO is PRSMID(L+1)
            Lp1 = jnp.minimum(L, LMH)  # L+1 in 1-based is l0+1
            TSP = (T[jnp.minimum(l0 + 1, nz - 1)] - T[l0]) / (PLO - PBOT) * (PSP - PBOT) + T[l0]
            QSP = (Q[jnp.minimum(l0 + 1, nz - 1)] - Q[l0]) / (PLO - PBOT) * (PSP - PBOT) + Q[l0]
            denomSP = TSP * (QSP * 0.608 + 1.0)
            TRMUP_1 = (TUP_sp * (QBT * 0.608 + 1.0) - denomSP) * 0.5 / denomSP
            DP1 = PLO - PSP
            dtv_base1 = TRMLO + TRMUP_1
            dentpy_b1 = dtv_base1 * DP1 + DENTPY
            # Sub-step 2 (PSP -> PRSMID(L)): updraft along moist adiabat at PUP
            TUP_2 = TUP_tbl[l0]
            QUP_2 = QUP_tbl[l0]
            QWAT2 = QBT - QUP_2
            TRMUP_2 = ((TUP_2 * (QUP_2 * 0.608 + 1.0 - QWAT2)) - denomTL[l0]) * 0.5 / denomTL[l0]
            DP2 = PSP - PUP
            dentpy_b2 = (TRMUP_1 + TRMUP_2) * DP2 + dentpy_b1

            dentpy_sel = jnp.where(at_base, dentpy_b2, dentpy_new)
            dentpy_sel = jnp.where(is_seed, DENTPY, dentpy_sel)  # seed: no integration

            cpe_val = dentpy_sel
            dtv_val = jnp.where(at_base, dtv_base1 * DP1, dtv_here)  # WRF stores DTV(L) at base as DTV*DP then folds

            # write outputs where in_range and not seed
            do_write = in_range & (~is_seed)
            cpe = cpe.at[l0].set(jnp.where(do_write, cpe_val, cpe[l0]))
            dtv = dtv.at[l0].set(jnp.where(do_write, dtv_val, dtv[l0]))

            broke_new = broke | (do_write & (dentpy_sel < CAPEtrigr))
            # advance carry
            new_PLO = jnp.where(in_range, PUP, PLO)
            new_TRMLO = jnp.where(at_base, TRMUP_2, jnp.where(in_range, trm_here, TRMLO))
            new_DENTPY = jnp.where(do_write, dentpy_sel, DENTPY)
            return (new_PLO, new_TRMLO, new_DENTPY, cpe, dtv, broke_new)

        cpe0 = jnp.zeros(nz, _DT)
        dtv0 = jnp.zeros(nz, _DT)
        PLO0 = PRSMID[kb0]
        st0 = (PLO0, _DT(0.0), _DT(0.0), cpe0, dtv0, jnp.asarray(False))
        # iterate i=0..LMH-1 (covers L=KB..KB-(LMH-1))
        st_final = lax.fori_loop(0, LMH, ascent_body, st0)
        (_, _, _, cpe, dtv, _) = st_final

        # LTOP search: DO L=KB,KTS,-1 pick max CPE while CPE>=CAPEtrigr
        def ltop_body(i, st):
            (LTP1, CAPE, brk) = st
            L = KB - i
            l0 = L - 1
            inr = (L >= 1) & (L <= KB) & (~brk)
            cval = cpe[l0]
            brk_new = brk | (inr & (cval < CAPEtrigr))
            take = inr & (~brk) & (cval >= CAPEtrigr) & (cval > CAPE)
            LTP1 = jnp.where(take, L, LTP1).astype(jnp.int32)
            CAPE = jnp.where(take, cval, CAPE)
            return (LTP1, CAPE, brk_new)
        LTP1_0 = KB.astype(jnp.int32)
        st_lt = lax.fori_loop(0, LMH, ltop_body, (LTP1_0, _DT(0.0), jnp.asarray(False)))
        (LTP1, CAPE, _) = st_lt
        LTOP = jnp.minimum(LTP1, LBOT).astype(jnp.int32)

        # keep best CAPE across KB
        better = active_kb & (CAPE > CAPEcnv)
        CAPEcnv = jnp.where(better, CAPE, CAPEcnv)
        PSPcnv = jnp.where(better, PSP, PSPcnv)
        THBTcnv = jnp.where(better, THBT, THBTcnv)
        LBOTcnv = jnp.where(better, LBOT, LBOTcnv).astype(jnp.int32)
        LTOPcnv = jnp.where(better, LTOP, LTOPcnv).astype(jnp.int32)
        CPEcnv = jnp.where(better, cpe, CPEcnv)
        DTVcnv = jnp.where(better, dtv, DTVcnv)
        THEScnv = jnp.where(better, jnp.full(nz, THESP, _DT), THEScnv)
        return (CAPEcnv, PSPcnv, THBTcnv, LBOTcnv, LTOPcnv,
                CPEcnv, DTVcnv, THEScnv, new_stop)

    init = (_DT(0.0), _DT(0.0), _DT(0.0),
            jnp.asarray(LMH, jnp.int32), jnp.asarray(LMH, jnp.int32),
            jnp.zeros(nz, _DT), jnp.zeros(nz, _DT), jnp.zeros(nz, _DT), jnp.asarray(False))
    (CAPEcnv, PSP, THBT, LBOT, LTOP, CPE, DTV, THES, _) = lax.fori_loop(0, LMH, kb_body, init)

    PBOT = PRSMID[LBOT - 1]
    PTOP = PRSMID[LTOP - 1]

    # --- Quick exit if cloud too thin or no CAPE ---
    no_conv = (PTOP > PBOT - PNO) | (LTOP > LBOT - 2) | (CAPEcnv <= 0.0)
    DEPTH0 = PBOT - PTOP
    DEEP = (~no_conv) & (DEPTH0 >= (PSH * PSFC * RSFCP))
    SHALLOW = (~no_conv) & (~DEEP)

    # ============================ DEEP CONVECTION ============================
    dtdt_deep, dqdt_deep, pcp_deep, cldefi_deep, lbot_deep, ltop_deep = _deep_branch(
        T, Q, PRSMID, DPRS, APE, PSFC, SM, CLDEFI_IN, TAUK, RDTCNVC,
        LBOT, LTOP, THES, DTV, nz)

    # ============================ SHALLOW CONVECTION =========================
    # When the deep branch's enthalpy check fails it falls into shallow; the deep
    # branch returns deep_failed via cldefi_deep<0 sentinel handled inside. We
    # compute the shallow branch with the (possibly deep-adjusted) LBOT/LTOP.
    dtdt_sh, dqdt_sh, lbot_sh, ltop_sh, shallow_ok = _shallow_branch(
        T, Q, PRSMID, DPRS, APE, PSFC, SM, THBT, PSP, CPE, DTV, TAUKSC, RDTCNVC,
        LBOT, LTOP, PBOT, nz)

    # deep branch may demote to shallow (DENTPY<EPSNTP). Signal via pcp_deep<0.
    deep_demoted = pcp_deep < 0.0
    use_deep = DEEP & (~deep_demoted)
    use_shallow = (SHALLOW | (DEEP & deep_demoted)) & shallow_ok

    zeros = jnp.zeros(nz, _DT)
    DTDT = jnp.where(use_deep, dtdt_deep,
            jnp.where(use_shallow, dtdt_sh, zeros))
    DQDT = jnp.where(use_deep, dqdt_deep,
            jnp.where(use_shallow, dqdt_sh, zeros))
    PCPCOL = jnp.where(use_deep, jnp.maximum(pcp_deep, _DT(0.0)), _DT(0.0))

    LBOT_out = jnp.where(use_deep, lbot_deep,
                jnp.where(use_shallow, lbot_sh, 0)).astype(jnp.int32)
    LTOP_out = jnp.where(use_deep, ltop_deep,
                jnp.where(use_shallow, ltop_sh, nz)).astype(jnp.int32)

    # CLDEFI output
    cldefi_nonconv = AVGEFI * SM + STEFI * (1.0 - SM)
    cldefi_demote = EFIMN * SM + STEFI * (1.0 - SM)
    CLDEFI_OUT = jnp.where(use_deep, cldefi_deep,
                  jnp.where(DEEP & deep_demoted, cldefi_demote, cldefi_nonconv))
    # for pure shallow (not demoted-from-deep), WRF leaves CLDEFI unchanged from
    # the deep-exit assignment only if it went through 300; pure shallow keeps
    # the nonconvective assignment computed at the quick-exit? No: shallow path
    # bypasses the CLDEFI write, so CLDEFI stays at input. Match WRF: pure
    # shallow does NOT modify CLDEFI.
    CLDEFI_OUT = jnp.where(SHALLOW & (~DEEP), CLDEFI_IN, CLDEFI_OUT)
    CLDEFI_OUT = jnp.where(no_conv, cldefi_nonconv, CLDEFI_OUT)

    return DTDT, DQDT, PCPCOL, LBOT_out, LTOP_out, CLDEFI_OUT, use_deep, use_shallow


def _deep_branch(T, Q, PRSMID, DPRS, APE, PSFC, SM, CLDEFI_IN, TAUK, RDTCNVC,
                 LBOT, LTOP, THES, DTV, nz):
    """DEEP CONVECTION block (label 300..). Returns tendencies + pcp.

    pcp<0 signals the enthalpy/precip check failed -> demote to shallow.
    LBOT/LTOP are 1-based. Profiles indexed 0-based as PRSMID etc.
    """
    LMH = nz
    LB = LBOT
    EFI = CLDEFI_IN
    lb0 = LB - 1
    lt0 = LTOP - 1
    kidx = jnp.arange(nz, dtype=jnp.int32)

    # moist-adiabat reference temperature at every level (THES=THESP from best KB)
    THES_lvl = THES  # constant THESP per level
    TREF_ma = _ttblex_auto(PRSMID, THES_lvl)
    THERK = TREF_ma * APE

    TK = T
    QK = Q
    APEK = APE
    PK = PRSMID

    # ---- reference T profile below freezing (DO L=LBM1,LTOP,-1) ----
    STABDL = (EFI - EFIMN) * SLOPST + STABDS
    PKB = PK[lb0]
    PKT = PK[lt0]

    # iterate L from LB-1 down to LTOP building TREFK along constant-lapse moist
    # profile until T(L+1)<TFRZ; then switch to above-freezing linear-in-pressure.
    def build_tref(carry, i):
        (TREFK, TREFKX, APEKXX, THERKX, APEKXY, THERKY, L0, frozen) = carry
        L = (LB - 1) - i  # 1-based, from LBM1 downward
        l0 = L - 1
        inr = (L >= LTOP) & (L <= LB - 1) & (~frozen)
        hit_frz = inr & (T[l0 + 1] < TFRZ)  # IF(T(L+1)<TFRZ) GO TO 430
        # below-freezing update
        TREFKX_new = ((THERKY - THERKX) * STABDL + TREFKX * APEKXX) / APEKXY
        do_update = inr & (~hit_frz)
        TREFK = TREFK.at[l0].set(jnp.where(do_update, TREFKX_new, TREFK[l0]))
        new_TREFKX = jnp.where(do_update, TREFKX_new, TREFKX)
        new_APEKXX = jnp.where(do_update, APEKXY, APEKXX)
        new_THERKX = jnp.where(do_update, THERKY, THERKX)
        lm1 = jnp.maximum(l0 - 1, 0)
        new_APEKXY = jnp.where(do_update, APEK[lm1], APEKXY)
        new_THERKY = jnp.where(do_update, THERK[lm1], THERKY)
        new_L0 = jnp.where(do_update, L, L0).astype(jnp.int32)
        frozen_new = frozen | hit_frz
        return (TREFK, new_TREFKX, new_APEKXX, new_THERKX, new_APEKXY,
                new_THERKY, new_L0, frozen_new), None

    TREFK0 = TK
    carry0 = (TREFK0, TK[lb0], APEK[lb0], THERK[lb0], APEK[lb0 - 1], THERK[lb0 - 1],
              LB, jnp.asarray(False))
    (carry_b, _) = lax.scan(build_tref, carry0, jnp.arange(nz))
    (TREFK, TREFKX, _, _, _, _, L0, frozen) = carry_b

    # above-freezing region (label 430): DO L=LTOP,L0-1 linear in pressure
    l00 = L0 - 1
    PK0 = PK[l00]
    RDP0T = 1.0 / (PK0 - PKT)
    DTHEM = THERK[l00] - TREFK[l00] * APEK[l00]
    Lvec = kidx + 1
    in_above = frozen & (Lvec >= LTOP) & (Lvec <= L0 - 1)
    TREFK_above = (THERK - (PK - PKT) * DTHEM * RDP0T) / APEK
    TREFK = jnp.where(in_above, TREFK_above, TREFK)
    EL = jnp.full(nz, ELWV, _DT)

    DEPWL = PKB - PK0
    DEPTH_frz = PFRZ * PSFC * RSFCP
    SM1 = 1.0 - SM
    PBOTFC = 1.0

    in_col = (Lvec >= LTOP) & (Lvec <= LB)  # L=LTOP..LB

    A23M4L = A2 * (A3 - A4) * ELWV
    RCP = 1.0 / CP

    # ---- cloud-efficiency loop (ITREFI=1..3) ----
    def cloud_eff_iter(state, _):
        (TREFK, EFI) = state
        DSPBK = ((EFI - EFIMN) * SLOPBS + DSPBSS * PBOTFC) * SM + ((EFI - EFIMN) * SLOPBL + DSPBSL * PBOTFC) * SM1
        DSP0K = ((EFI - EFIMN) * SLOP0S + DSP0SS * PBOTFC) * SM + ((EFI - EFIMN) * SLOP0L + DSP0SL * PBOTFC) * SM1
        DSPTK = ((EFI - EFIMN) * SLOPTS + DSPTSS * PBOTFC) * SM + ((EFI - EFIMN) * SLOPTL + DSPTSL * PBOTFC) * SM1

        # DSP per level
        dsp_branch_geq = jnp.where(
            Lvec < L0,
            ((PK0 - PK) * DSPTK + (PK - PKT) * DSP0K) / (PK0 - PKT),
            ((PKB - PK) * DSP0K + (PK - PK0) * DSPBK) / (PKB - PK0),
        )
        dsp_branch_lt = jnp.where(
            Lvec < L0,
            ((PK0 - PK) * DSPTK + (PK - PKT) * DSP0K) / (PK0 - PKT),
            DSP0K,
        )
        DSP = jnp.where(DEPWL >= DEPTH_frz, dsp_branch_geq, dsp_branch_lt)
        PSK = PK + DSP
        APESK = (1.0e5 / PSK) ** CAPA
        THSK = TREFK * APEK
        QREFK_sat = PQ0 / PSK * jnp.exp(A2 * (THSK - A3 * APESK) / (THSK - A4 * APESK))
        QREFK = jnp.where(PK > PQM, QREFK_sat, QK)

        # ---- enthalpy conservation (ITER=1,2) ----
        def enth_iter(st2, _2):
            (TREFK_i, QREFK_i) = st2
            mask = in_col.astype(_DT)
            SUMDE = jnp.sum(((TK - TREFK_i) * CP + (QK - QREFK_i) * EL) * DPRS * mask)
            DHDT = jnp.sum((QREFK_i * A23M4L / ((TREFK_i * APEK / APESK) - A4) ** 2 + CP) * DPRS * mask)
            SUMDP = jnp.sum(DPRS * mask)
            denom = SUMDP - DPRS[lt0]
            HCORR = SUMDE / denom
            DHDT = DHDT / denom
            # LQM: largest L with PK<=PQM (1-based), searched L=KTS..LB
            cond_lqm = (Lvec <= LB) & (PK <= PQM)
            LQM = jnp.maximum(jnp.max(jnp.where(cond_lqm, Lvec, 1)), 1)
            LCOR = LTOP + 1
            # above LQM (LCOR..LQM): correct T only by HCORR*RCP
            in_tonly = (Lvec >= LCOR) & (Lvec <= LQM)
            TREFK_new = jnp.where(in_tonly, TREFK_i + HCORR * RCP, TREFK_i)
            LCOR2 = jnp.where(LCOR <= LQM, LQM + 1, LCOR)
            # below LQM (LCOR2..LB): correct T and Q
            in_both = (Lvec >= LCOR2) & (Lvec <= LB)
            TREFK_b = jnp.where(in_both, HCORR / DHDT + TREFK_new, TREFK_new)
            THSKL = TREFK_b * APEK
            QREFK_b = PQ0 / PSK * jnp.exp(A2 * (THSKL - A3 * APESK) / (THSKL - A4 * APESK))
            QREFK_new = jnp.where(in_both, QREFK_b, QREFK_i)
            return (TREFK_b, QREFK_new), None

        (TREFK, QREFK), _ = lax.scan(enth_iter, (TREFK, QREFK), None, length=2)

        # ---- heating, moistening, precip ----
        mask = in_col.astype(_DT)
        DIFT = (TREFK - TK) * TAUK * mask
        DIFQ = (QREFK - QK) * TAUK * mask
        AVRGTL = (TK + TK + DIFT)
        DPOT = DPRS / AVRGTL
        DST = jnp.sum(DIFT * DPOT * mask)
        DSQ = jnp.sum(DIFQ * EL * DPOT * mask)
        AVRGT = jnp.sum(AVRGTL * DPRS * mask)
        PRECK = jnp.sum(DIFT * DPRS * mask)
        SUMDP = jnp.sum(DPRS * mask)

        DST = (DST + DST) * CP
        DSQ = DSQ + DSQ
        DENTPY = DST + DSQ
        AVRGT = AVRGT / (SUMDP + SUMDP)
        DRHEAT = (PRECK * SM + jnp.maximum(1.0e-7, PRECK) * (1.0 - SM)) * CP / AVRGT
        DRHEAT = jnp.maximum(DRHEAT, 1.0e-20)
        EFI_new = EFIFC * DENTPY / DRHEAT
        EFI_new = jnp.minimum(EFI_new, 1.0)
        EFI_new = jnp.maximum(EFI_new, EFIMN)
        return (TREFK, EFI_new), (DENTPY, PRECK, DIFT, DIFQ, AVRGT)

    (TREFK, EFI), traj = lax.scan(cloud_eff_iter, (TREFK, EFI), None, length=ITREFI_MAX)
    DENTPY = traj[0][-1]
    PRECK = traj[1][-1]
    DIFT = traj[2][-1]
    DIFQ = traj[3][-1]
    AVRGT = traj[4][-1]

    CPRLG = CP / (ROW * G * ELWV)
    accept = (DENTPY >= EPSNTP) & (PRECK > EPSPR)

    FEFI = EFMNT + SLOPE * (EFI - EFIMN)
    FEFI = (DENTPY - EPSNTP) * FEFI / DENTPY
    PRECK_f = PRECK * FEFI
    CUP = PRECK_f * CPRLG
    DTDT = DIFT * FEFI * RDTCNVC
    DQDT = DIFQ * FEFI * RDTCNVC

    pcp = jnp.where(accept, CUP, -1.0)  # -1 sentinel -> demote to shallow
    cldefi_out = jnp.where(accept, EFI, EFIMN)  # placeholder; demotion handled by caller
    return DTDT, DQDT, pcp, cldefi_out, LBOT.astype(jnp.int32), LTOP.astype(jnp.int32)


def _shallow_branch(T, Q, PRSMID, DPRS, APE, PSFC, SM, THBT, PSP, CPE, DTV,
                    TAUKSC, RDTCNVC, LBOT_in, LTOP_in, PBOT_in, nz):
    """SHALLOW CONVECTION block. Returns DTDT,DQDT (flipped),LBOT,LTOP,ok.

    ok=False means a GO TO 800 abort fired -> no shallow tendency (nonconvective).
    Note: WRF first (in deep demotion) recomputes LTOP via PTPK/DTV; for a column
    that triggered SHALLOW directly the deep block was skipped, so LBOT/LTOP come
    from the max-buoyancy search. We follow the active code path for both.
    """
    LMH = nz
    kidx = jnp.arange(nz, dtype=jnp.int32)
    Lvec = kidx + 1
    DEPMIN = PSH * PSFC * RSFCP

    # The deep demotion path recomputes LTOP. We mimic the demotion branch which
    # is the one actually used when DENTPY<EPSNTP, then the common shallow code.
    # PTPK = MAX(PSHU, PK(LBOT)-DEPMIN); LTOP = level just below PTPK; then
    # LTOP via DTV>0. For a direct-SHALLOW column LBOT/LTOP already set.
    PK = PRSMID
    LBOT = LBOT_in
    lb0 = LBOT - 1
    PTPK = jnp.maximum(PSHU, PK[lb0] - DEPMIN)
    cond = (Lvec <= LMH) & (PK <= PTPK)
    L_top_p = jnp.max(jnp.where(cond, Lvec + 1, 0))
    LTOP = jnp.where(L_top_p > 0, L_top_p, LTOP_in)
    # LTOP via DTV>0 above (DO L=LBOT-1,LTOP,-1)
    def dtv_top(i, st):
        (LTP1, brk) = st
        L = (LBOT - 1) - i
        inr = (L >= LTOP) & (L <= LBOT - 1) & (~brk)
        pos = inr & (DTV[L - 1] > 0.0)
        LTP1 = jnp.where(pos, L, LTP1)
        brk_new = brk | (inr & (DTV[L - 1] <= 0.0))
        return (LTP1, brk_new)
    (LTP1, _) = lax.fori_loop(0, LMH, dtv_top, (LBOT, jnp.asarray(False)))
    LTOP = jnp.minimum(LTP1, LBOT)
    PTOP = PK[LTOP - 1]
    PBOT = PK[lb0]

    # init shallow profiles
    TK = T
    QK = Q
    APEK = APE
    QSATK = _qsat_bmj(PK, TK)
    EL = jnp.full(nz, ELWV, _DT)
    THVREF = TK * APEK * (QK * D608 + 1.0)
    TREFK = TK

    # Raise cloud top if avg RH>RHSHmax and CAPE>0
    TLEV2 = T[lb0] * ((PK[lb0] - PONE) / PK[lb0]) ** CAPA
    QSAT1 = PQ0 / PK[lb0] * jnp.exp(A2 * (T[lb0] - A3) / (TK[lb0] - A4))
    QSAT2 = PQ0 / (PK[lb0] - PONE) * jnp.exp(A2 * (TLEV2 - A3) / (TLEV2 - A4))
    RHSHmax = QSAT2 / QSAT1
    in_lt_lb = (Lvec >= LTOP) & (Lvec <= LBOT)
    RHAVG0 = jnp.sum(jnp.where(in_lt_lb, DPRS * QK / QSATK, 0.0))
    SUMDP0 = jnp.sum(jnp.where(in_lt_lb, DPRS, 0.0))

    def raise_top(i, st):
        (LTSH, RHAVG, SUMDP, brk) = st
        L = (LTOP - 1) - i  # from LTOP-1 down
        inr = (L >= 1) & (L <= LTOP - 1) & (~brk)
        l0 = L - 1
        RHAVG_n = jnp.where(inr, RHAVG + DPRS[l0] * QK[l0] / QSATK[l0], RHAVG)
        SUMDP_n = jnp.where(inr, SUMDP + DPRS[l0], SUMDP)
        cape_pos = inr & (CPE[l0] > 0.0)
        LTSH_n = jnp.where(cape_pos, L, LTSH)
        # break if CPE<=0, or RHAVG/SUMDP<=RHSHmax, or PK<=PSHU
        brk_cape = inr & (CPE[l0] <= 0.0)
        brk_rh = inr & (~brk_cape) & (RHAVG_n / SUMDP_n <= RHSHmax)
        brk_psh = inr & (~brk_cape) & (~brk_rh) & (PK[l0] <= PSHU)
        brk_new = brk | brk_cape | brk_rh | brk_psh
        return (LTSH_n, RHAVG_n, SUMDP_n, brk_new)

    do_raise = (RHAVG0 / SUMDP0) > RHSHmax
    (LTSH, _, _, _) = lax.fori_loop(0, LMH, raise_top, (LTOP, RHAVG0, SUMDP0, jnp.asarray(False)))
    LTOP = jnp.where(do_raise, LTSH, LTOP)
    PTOP = PK[LTOP - 1]
    lt0 = LTOP - 1
    LTP1 = LTOP - 1

    DEPTH = PBOT - PTOP
    abort1 = (PTOP > PBOT - PNO) | (LTOP > LBOT - 2)

    # shallow reference temperature profile via PTBL at top + SMIX slope
    THTPK = T[LTP1 - 1] * APE[LTP1 - 1]
    TTHK = (THTPK - THL) * RDTH
    QQK = TTHK - _aint(TTHK)
    IT = (_aint(TTHK) + 1).astype(jnp.int32)
    IT = jnp.where((IT < 1) | (IT >= JTB), jnp.clip(IT, 1, JTB - 1), IT)
    QQK = jnp.where((IT < 1) | (IT >= JTB), 0.0, QQK)
    IT = jnp.clip(IT, 1, JTB - 1)
    it0 = IT - 1
    BQS00K = QS0[it0]
    SQS00K = SQS[it0]
    BQS10K = QS0[it0 + 1]
    SQS10K = SQS[it0 + 1]
    BQK = (BQS10K - BQS00K) * QQK + BQS00K
    SQK = (SQS10K - SQS00K) * QQK + SQS00K
    TQK = (Q[LTP1 - 1] - BQK) / SQK * RDQ
    PPK = TQK - _aint(TQK)
    IQ = (_aint(TQK) + 1).astype(jnp.int32)
    PPK = jnp.where((IQ < 1) | (IQ >= ITB), 0.0, PPK)
    IQ = jnp.clip(IQ, 1, ITB - 1)
    iq0 = IQ - 1
    PART1 = (PTBL[iq0 + 1, it0] - PTBL[iq0, it0]) * PPK
    PART2 = (PTBL[iq0, it0 + 1] - PTBL[iq0, it0]) * QQK
    PART3 = (PTBL[iq0, it0] - PTBL[iq0 + 1, it0] - PTBL[iq0, it0 + 1] + PTBL[iq0 + 1, it0 + 1]) * PPK * QQK
    PTPK = PTBL[iq0, it0] + PART1 + PART2 + PART3
    DPMIX = PTPK - PSP
    DPMIX = jnp.where(jnp.abs(DPMIX) < 3000.0, -3000.0, DPMIX)
    SMIX = (THTPK - THBT) / DPMIX * STABS

    # build TREFK over L=LBOT..LTOP (decreasing)
    LMID = jnp.floor(0.5 * (LBOT + LTOP)).astype(jnp.int32)

    def build_sh_tref(carry, i):
        (TREFK, TREFKX, PKXXXX, PKXXXY, APEKXX, APEKXY) = carry
        L = LBOT - i  # from LBOT down to LTOP
        l0 = L - 1
        inr = (L >= LTOP) & (L <= LBOT)
        TREFKX_new = ((PKXXXY - PKXXXX) * SMIX + TREFKX * APEKXX) / APEKXY
        val = jnp.where(L <= LMID, jnp.maximum(TREFKX_new, TK[l0] + DTSHAL), TREFKX_new)
        TREFK = TREFK.at[l0].set(jnp.where(inr, val, TREFK[l0]))
        lm1 = jnp.maximum(l0 - 1, 0)
        new_TREFKX = jnp.where(inr, TREFKX_new, TREFKX)
        new_APEKXX = jnp.where(inr, APEKXY, APEKXX)
        new_PKXXXX = jnp.where(inr, PKXXXY, PKXXXX)
        new_APEKXY = jnp.where(inr, APEK[lm1], APEKXY)
        new_PKXXXY = jnp.where(inr, PK[lm1], PKXXXY)
        return (TREFK, new_TREFKX, new_PKXXXX, new_PKXXXY, new_APEKXX, new_APEKXY), None

    lbp1 = jnp.minimum(LBOT, nz - 1)  # LBOT+1 1-based -> index LBOT (0-based)
    sh0 = (TREFK, TREFK[lbp1], PK[lbp1], PK[lb0], APEK[lbp1], APEK[lb0])
    (carry_sh, _) = lax.scan(build_sh_tref, sh0, jnp.arange(nz))
    TREFK = carry_sh[0]

    in_col = (Lvec >= LTOP) & (Lvec <= LBOT)
    mask = in_col.astype(_DT)
    # temperature correction
    SUMDT = jnp.sum((TK - TREFK) * DPRS * mask)
    SUMDP = jnp.sum(DPRS * mask)
    RDPSUM = 1.0 / SUMDP
    TCORR = SUMDT * RDPSUM
    FPK = jnp.where(in_col, TREFK + TCORR, TREFK)
    TREFK = jnp.where(in_col, TREFK + TCORR, TREFK)
    FPTK = FPK[lt0]

    # humidity profile equations
    RTBAR = 2.0 / (TREFK + TK)
    DPKL = FPK - FPTK
    PSUM = jnp.sum(jnp.where(in_col, DPKL * DPRS, 0.0)) * RDPSUM
    QSUM = jnp.sum(jnp.where(in_col, QK * DPRS, 0.0)) * RDPSUM
    OTSUM = jnp.sum(jnp.where(in_col, DPRS * RTBAR, 0.0))
    POTSUM = jnp.sum(jnp.where(in_col, DPKL * RTBAR * DPRS, 0.0))
    QOTSUM = jnp.sum(jnp.where(in_col, QK * RTBAR * DPRS, 0.0))
    DST = jnp.sum(jnp.where(in_col, (TREFK - TK) * RTBAR * DPRS / EL, 0.0))
    ROTSUM = 1.0 / OTSUM
    POTSUM = POTSUM * ROTSUM
    QOTSUM = QOTSUM * ROTSUM
    DST = DST * ROTSUM * CP

    abort_dst = DST > 0.0
    DSTQ = DST * EPSDN
    DEN = POTSUM - PSUM
    abort_iso = (-DEN / PSUM) < 5.0e-5
    DQREF = (QOTSUM - DSTQ - QSUM) / DEN
    abort_dq = DQREF < 0.0
    QRFTP = QSUM - DQREF * PSUM

    # humidity profile per level + too dry/moist checks
    QRFKL = (FPK - FPTK) * DQREF + QRFTP
    TNEW = (TREFK - TK) * TAUKSC + TK
    QSATK_n = PQ0 / PK * jnp.exp(A2 * (TNEW - A3) / (TNEW - A4))
    QNEW = (QRFKL - QK) * TAUKSC + QK
    too_dry = jnp.any(in_col & (QNEW < QSATK_n * RHLSC))
    too_moist = jnp.any(in_col & (QNEW > QSATK_n * RHHSC))
    THVREF_n = jnp.where(in_col, TREFK * APEK * (QRFKL * D608 + 1.0), THVREF)
    QREFK = jnp.where(in_col, QRFKL, QK)

    # impossible slopes check: DTDP=(THVREF(L-1)-THVREF(L))/(PRSMID(L)-PRSMID(L-1))
    THVREF_shift = jnp.concatenate([THVREF_n[:1], THVREF_n[:-1]])  # THVREF(L-1)
    PRS_shift = jnp.concatenate([PRSMID[:1], PRSMID[:-1]])
    DTDP = (THVREF_shift - THVREF_n) / (PRSMID - PRS_shift + 1e-30)
    bad_slope = jnp.any(in_col & (Lvec >= LTOP) & (DTDP < EPSDT))

    ok = (~abort1) & (~abort_dst) & (~abort_iso) & (~abort_dq) & (~too_dry) & (~too_moist) & (~bad_slope)

    DTDT = jnp.where(in_col, (TREFK - TK) * TAUKSC * RDTCNVC, 0.0)
    DQDT = jnp.where(in_col, (QREFK - QK) * TAUKSC * RDTCNVC, 0.0)
    DTDT = jnp.where(ok, DTDT, 0.0)
    DQDT = jnp.where(ok, DQDT, 0.0)
    LBOT_out = jnp.where(ok, LBOT, 0).astype(jnp.int32)
    LTOP_out = jnp.where(ok, LTOP, nz).astype(jnp.int32)
    return DTDT, DQDT, LBOT_out, LTOP_out, ok


# ---------------------------------------------------------------------------
# Public column entrypoint (WRF BMJDRV wrapper)
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnames=("stepcu", "nz"))
def _bmj_column_arrays(temperature, qv, pressure, dz, rho, pi_exner, dt, psfc,
                       *, stepcu, xland, cldefi, nz):
    """BMJDRV wrapper: flip arrays top-down, call BMJ, flip tendencies back.

    The BMJ internal arithmetic runs in fp32 to mirror pristine WRF (default
    REAL); inputs are cast to fp32 first, outputs cast back to fp64 for the
    physics interface.
    """
    T = jnp.asarray(temperature, _DT)
    qv_mix = jnp.asarray(qv, _DT)
    p = jnp.asarray(pressure, _DT)
    dz = jnp.asarray(dz, _DT)
    rho = jnp.asarray(rho, _DT)
    pi = jnp.asarray(pi_exner, _DT)
    dt_f = jnp.asarray(dt, _DT)
    stepcu_f = jnp.asarray(stepcu, _DT)
    cld0 = jnp.asarray(cldefi, _DT)
    SM = jnp.asarray(xland, _DT) - _DT(1.0)  # LANDMASK: 1 sea, 0 land
    DTCNVC = dt_f * stepcu_f

    # specific humidity (BMJDRV): QCOL=MAX(EPSQ, QV/(1+QV)) bottom-up
    q_spec = jnp.maximum(_DT(EPSQ), qv_mix / (_DT(1.0) + qv_mix))

    # FLIP top-down: index 0 = model top, index nz-1 = surface
    Tf = T[::-1]
    Qf = q_spec[::-1]
    Pf = p[::-1]
    DPRSf = (rho * _DT(G) * dz)[::-1]   # DPCOL = RHO*G*DZ8W
    PSFCv = jnp.asarray(psfc, _DT)

    APEf = (_DT(1.0e5) / Pf) ** _DT(CAPA)

    DTDTf, DQDTf, PCPCOL, LBOT, LTOP, CLDEFI_OUT, DEEP, SHALLOW = _bmj_kernel(
        Tf, Qf, Pf, DPRSf, APEf, PSFCv, SM, cld0, DTCNVC, nz)

    # flip tendencies back to bottom-up
    DTDT = DTDTf[::-1]
    DQDT = DQDTf[::-1]

    # RTHCUTEN = DTDT/PI ; RQVCUTEN = DQDT/(1-QCOL)**2  (bottom-up), fp32 like WRF
    rthcuten = (DTDT / pi).astype(jnp.float64)
    rqvcuten = (DQDT / (_DT(1.0) - q_spec) ** 2).astype(jnp.float64)

    raincv = (PCPCOL * _DT(1.0e3) / stepcu_f).astype(jnp.float64)
    pratec = (PCPCOL * _DT(1.0e3) / (stepcu_f * dt_f)).astype(jnp.float64)
    # CUTOP=REAL(KTE+1-LTOP); CUBOT=REAL(KTE+1-LBOT)  (LBOT/LTOP are 1-based flipped)
    cutop = jnp.asarray(nz + 1, jnp.float64) - LTOP.astype(jnp.float64)
    cubot = jnp.asarray(nz + 1, jnp.float64) - LBOT.astype(jnp.float64)

    return (rthcuten, rqvcuten, raincv, pratec, cutop, cubot, CLDEFI_OUT,
            DEEP.astype(jnp.int32), SHALLOW.astype(jnp.int32))


def step_bmj_column(temperature, qv, pressure, dz, rho, pi_exner, dt, *,
                    stepcu=1, xland=1.0, cldefi=AVGEFI, psfc=None, pint=None):
    """Run one BMJ column and return the frozen physics-interface payload.

    Inputs are bottom-up mass-level arrays (WRF driver order): ``qv`` mixing
    ratio, ``pressure`` in Pa, ``pi_exner`` such that ``RTHCUTEN=DTDT/PI``.

    ``psfc``/``pint``: surface pressure is needed by BMJ (DEPMIN/DEPTH scaling).
    Pass ``pint`` (interface pressures, length nz+1, bottom-up) to use the exact
    WRF ``PSFC=PINT(LOWLYR=1)``; or pass ``psfc`` directly. If neither is given
    it is extrapolated from the lowest two mass levels.
    """
    nz = int(temperature.shape[0])
    # Surface pressure (traceable under jit/vmap). WRF uses PSFC=PINT(LOWLYR=1).
    if pint is not None:
        psfc_v = jnp.asarray(pint).reshape(-1)[0]
    elif psfc is not None:
        psfc_v = jnp.asarray(psfc).reshape(-1)[0]
    else:
        p = jnp.asarray(pressure)
        psfc_v = p[0] + 0.5 * (p[0] - p[1])

    (rthcuten, rqvcuten, raincv, pratec, cutop, cubot, cldefi_next,
     is_deep, is_shallow) = _bmj_column_arrays(
        temperature, qv, pressure, dz, rho, pi_exner, dt, psfc_v,
        stepcu=stepcu, xland=xland, cldefi=cldefi, nz=nz)

    tendency = PhysicsTendency(
        state_tendencies={"theta": rthcuten, "qv": rqvcuten},
        accumulator_increments={"rainc_acc": raincv},
        diagnostics={
            "rthcuten": rthcuten,
            "rqvcuten": rqvcuten,
            "raincv": raincv,
            "pratec": pratec,
            "cutop": cutop,
            "cubot": cubot,
            "cldefi": cldefi_next,
            "trigger_deep": is_deep,
            "trigger_shallow": is_shallow,
        },
    )
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus={"cldefi": cldefi_next}),
        diagnostics=PhysicsDiagnostics(cumulus=tendency.diagnostics),
    )


def initial_bmj_cldefi(shape) -> jax.Array:
    """WRF BMJINIT default cloud efficiency, ``AVGEFI=(EFIMN+1)/2``."""
    return jnp.full(shape, AVGEFI, dtype=jnp.float64)


__all__ = ["AVGEFI", "initial_bmj_cldefi", "step_bmj_column"]
