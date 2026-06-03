"""Kain-Fritsch-eta (WRF cu_physics=1) cumulus parameterization, JAX port.

Faithful single-column port of `KF_eta_PARA` (+ its helpers TPMIX2, TPMIX2DD,
ENVIRTHT, DTFRZNEW, CONDLOAD, PROF5) from WRF `phys/module_cu_kfeta.F`, for the
9 km parent (d01). Classic Kain-Fritsch-Chappell trigger (`trigger=1`),
mixed-phase microphysics (F_QI=F_QS=.true.), warm_rain=.false. — matching the
project's Thompson coupling.

Design
------
* Each column is processed by `_kf_column`, written to mirror the Fortran
  control flow with `jax.lax` primitives (while_loop / fori_loop / cond /
  switch) at fixed maximum trip counts (KX levels, <=10 closure iterations,
  capped advection substeps). NO masking-clamps or happy-path shortcuts: every
  branch of the Fortran (USL search, shallow vs deep, downdraft / no-downdraft,
  LET==LTOP vs linear detrainment, the moisture-borrow fixup, the three
  microphysics-feedback cases) is reproduced.
* The whole column scheme is `vmap`-ped across (i, j); there is zero host/device
  transfer inside the call.
* fp64 throughout (the operational release precision; see V0.2.0 plan P1-8a) —
  matches the float64 lookup tables and the Fortran REAL*4 oracle to within a
  predeclared, physically-meaningful tolerance.

Returned tendencies use the SAME definitions the WRF driver applies after
`KF_eta_PARA`:
    RTHCUTEN = DTDT / pi        (potential-temperature tendency, K/s)
    RQVCUTEN = DQDT
    RQCCUTEN = DQCDT ; RQRCUTEN = DQRDT ; RQICUTEN = DQIDT ; RQSCUTEN = DQSDT
plus RAINCV (mm over DT), PRATEC (mm/s), NCA (s), CUTOP, CUBOT, ISHALL.

`debug=True` (a static arg) enables snapshot asserts; XLA dead-code-eliminates
them in production (`debug=False`).
"""
from __future__ import annotations

from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsCarry, PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency
from gpuwrf.physics import cumulus_kf_tables as _T

config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------
# Constants (DATA statements + WRF model constants used by KF_eta_PARA).
# ----------------------------------------------------------------------------
_P00 = 1.0e5
_T00 = 273.16
_RLF = 3.339e5
_RHIC = 1.0
_RHBC = 0.90
_PIE = 3.141592654
_TTFRZ = 268.16
_TBFRZ = 248.16
_C5 = 1.0723e-3
_RATE = 0.03
_DPMIN = 5.0e3
_FBFRC = 0.0          # no PPT feedback into grid-resolved fields

# closure / loop caps
_MAX_CLOSURE_ITERS = 10
_MAX_NSTEP = 200      # advection-substep cap (NSTEP=NINT(TIMEC/DTT+1); bounded)


def _safe(x, eps=1e-30):
    return jnp.where(jnp.abs(x) < eps, jnp.sign(x) * eps + (x == 0) * eps, x)


def _collapse_w_to_mass_levels(w, kx):
    """Return WRF's mass-level W0 = 0.5 * (w(k) + w(k+1))."""

    w = jnp.asarray(w, jnp.float64)
    return 0.5 * (w[:kx] + w[1 : kx + 1])


# ----------------------------------------------------------------------------
# Lookup-table helpers (TPMIX2 / TPMIX2DD share the bilinear interpolation).
# ----------------------------------------------------------------------------
def _table_interp(p, thes):
    """Bilinear lookup of (temp, qs) from (THES, P). Mirrors TPMIX2/TPMIX2DD
    index arithmetic. Returns (temp, qs, qq, pp)."""
    plutop = _T.PLUTOP
    rdpr = _T.RDPR
    rdthk = _T.RDTHK
    the0k = jnp.asarray(_T.THE0K)
    ttab = jnp.asarray(_T.TTAB)
    qstab = jnp.asarray(_T.QSTAB)

    tp = (p - plutop) * rdpr
    qq = tp - jnp.floor(tp)
    iptb = (jnp.floor(tp)).astype(jnp.int32)          # 0-based table row (Fortran int(tp)+1 -> here +0)
    # Fortran: iptb=int(tp)+1 (1-based). 0-based python index = int(tp). bth uses iptb,iptb+1.
    iptb = jnp.clip(iptb, 0, _T.KFNP - 2)

    bth = (the0k[iptb + 1] - the0k[iptb]) * qq + the0k[iptb]
    tth = (thes - bth) * rdthk
    pp = tth - jnp.floor(tth)
    ithtb = (jnp.floor(tth)).astype(jnp.int32)
    ithtb = jnp.clip(ithtb, 0, _T.KFNT - 2)

    t00 = ttab[ithtb, iptb]
    t10 = ttab[ithtb + 1, iptb]
    t01 = ttab[ithtb, iptb + 1]
    t11 = ttab[ithtb + 1, iptb + 1]
    q00 = qstab[ithtb, iptb]
    q10 = qstab[ithtb + 1, iptb]
    q01 = qstab[ithtb, iptb + 1]
    q11 = qstab[ithtb + 1, iptb + 1]

    temp = t00 + (t10 - t00) * pp + (t01 - t00) * qq + (t00 - t10 - t01 + t11) * pp * qq
    qs = q00 + (q10 - q00) * pp + (q01 - q00) * qq + (q00 - q10 - q01 + q11) * pp * qq
    return temp, qs


def tpmix2(p, thes, tu, qu, qliq, qice, xlv1, xlv0):
    """Faithful TPMIX2: lookup parcel T/qs at (thes,p), then saturation
    adjustment. Returns (tu_out, qu_out, qliq_out, qice_out, qnewlq, qnewic)."""
    temp, qs = _table_interp(p, thes)
    dq = qs - qu

    # branch DQ <= 0 : qnew = qu-qs, qu=qs
    qnew_le = qu - qs
    qu_le = qs
    qliq_le = qliq
    qice_le = qice
    temp_le = temp

    # branch DQ > 0 : subsaturated -> evaporate
    qtot = qliq + qice
    rll = xlv0 - xlv1 * temp
    cpp = 1004.5 * (1.0 + 0.89 * qu)

    # sub-branch QTOT >= DQ
    qliq_a = qliq - dq * qliq / (qtot + 1.0e-10)
    qice_a = qice - dq * qice / (qtot + 1.0e-10)
    qu_a = qs
    temp_a = temp

    # sub-branch QTOT < DQ, QTOT < 1e-10
    temp_b1 = temp + rll * (dq / (1.0 + dq)) / cpp
    qu_b1 = qu
    qliq_b1 = qliq
    qice_b1 = qice

    # sub-branch QTOT < DQ, QTOT >= 1e-10
    temp_b2 = temp + rll * ((dq - qtot) / (1.0 + dq - qtot)) / cpp
    qu_b2 = qu + qtot
    qliq_b2 = 0.0 * qliq
    qice_b2 = 0.0 * qice

    cond_qtot_small = qtot < 1.0e-10
    temp_b = jnp.where(cond_qtot_small, temp_b1, temp_b2)
    qu_b = jnp.where(cond_qtot_small, qu_b1, qu_b2)
    qliq_b = jnp.where(cond_qtot_small, qliq_b1, qliq_b2)
    qice_b = jnp.where(cond_qtot_small, qice_b1, qice_b2)

    cond_qtot_ge = qtot >= dq
    temp_gt = jnp.where(cond_qtot_ge, temp_a, temp_b)
    qu_gt = jnp.where(cond_qtot_ge, qu_a, qu_b)
    qliq_gt = jnp.where(cond_qtot_ge, qliq_a, qliq_b)
    qice_gt = jnp.where(cond_qtot_ge, qice_a, qice_b)
    qnew_gt = jnp.zeros_like(qnew_le)

    le = dq <= 0.0
    temp_out = jnp.where(le, temp_le, temp_gt)
    qu_out = jnp.where(le, qu_le, qu_gt)
    qliq_out = jnp.where(le, qliq_le, qliq_gt)
    qice_out = jnp.where(le, qice_le, qice_gt)
    qnew = jnp.where(le, qnew_le, qnew_gt)

    return temp_out, qu_out, qliq_out, qice_out, qnew, jnp.zeros_like(qnew)


def tpmix2dd(p, thes):
    """Faithful TPMIX2DD: just the bilinear lookup (no saturation adjustment)."""
    ts, qs = _table_interp(p, thes)
    return ts, qs


def envirtht(p1, t1, q1, aliq, bliq, cliq, dliq):
    """Faithful ENVIRTHT: environmental equivalent potential temperature."""
    astrt = 1.0e-3
    ainc = 0.075
    alu = jnp.asarray(_T.ALU)
    c1 = 3374.6525
    c2 = 2.5403
    t00 = 273.16
    p00 = 1.0e5

    ee = q1 * p1 / (0.622 + q1)
    a1 = ee / aliq
    tp = (a1 - astrt) / ainc
    indlu = (jnp.floor(tp)).astype(jnp.int32)          # Fortran int(tp)+1 (1-based) -> 0-based int(tp)
    indlu = jnp.clip(indlu, 0, 198)
    value = (indlu) * ainc + astrt                     # (indlu-1)*ainc+astrt in Fortran 1-based == indlu0*ainc+astrt
    aintrp = (a1 - value) / ainc
    tlog = aintrp * alu[indlu + 1] + (1.0 - aintrp) * alu[indlu]

    tdpt = (cliq - dliq * tlog) / (bliq - tlog)
    tsat = tdpt - (0.212 + 1.571e-3 * (tdpt - t00) - 4.36e-4 * (t1 - t00)) * (t1 - tdpt)
    tht = t1 * (p00 / p1) ** (0.2854 * (1.0 - 0.28 * q1))
    tht1 = tht * jnp.exp((c1 / tsat - c2) * q1 * (1.0 + 0.81 * q1))
    return tht1


def dtfrznew(tu, p, thteu, qu, qfrz, qice, aliq, bliq, cliq, dliq):
    """Faithful DTFRZNEW: freezing of liquid in updraft."""
    rlc = 2.5e6 - 2369.276 * (tu - 273.16)
    rls = 2833922.0 - 259.532 * (tu - 273.16)
    rlf = rls - rlc
    cpp = 1004.5 * (1.0 + 0.89 * qu)
    a = (cliq - bliq * dliq) / ((tu - dliq) * (tu - dliq))
    dtfrz = rlf * qfrz / (cpp + rls * qu * a)
    tu = tu + dtfrz

    es = aliq * jnp.exp((bliq * tu - cliq) / (tu - dliq))
    qs = es * 0.622 / (p - es)
    dqevap = qs - qu
    qice = qice - dqevap
    qu = qu + dqevap
    pii = (1.0e5 / p) ** (0.2854 * (1.0 - 0.28 * qu))
    thteu = tu * pii * jnp.exp((3374.6525 / tu - 2.5403) * qu * (1.0 + 0.81 * qu))
    return tu, thteu, qu, qice


def condload(qliq, qice, wtw, dz, boterm, enterm, rate, qnewlq, qnewic, g):
    """Faithful CONDLOAD: precipitation fallout (Ogura & Cho 1973)."""
    qtot = qliq + qice
    qnew = qnewlq + qnewic
    qest = 0.5 * (qtot + qnew)
    g1 = wtw + boterm - enterm - 2.0 * g * dz * qest / 1.5
    g1 = jnp.where(g1 < 0.0, 0.0, g1)
    wavg = 0.5 * (jnp.sqrt(jnp.maximum(wtw, 0.0)) + jnp.sqrt(g1))
    conv = rate * dz / _safe(wavg)

    ratio3 = qnewlq / (qnew + 1.0e-8)
    qtot = qtot + 0.6 * qnew
    oldq = qtot
    ratio4 = (0.6 * qnewlq + qliq) / (qtot + 1.0e-8)
    qtot = qtot * jnp.exp(-conv)

    dq = oldq - qtot
    qlqout = ratio4 * dq
    qicout = (1.0 - ratio4) * dq

    pptdrg = 0.5 * (oldq + qtot - 0.2 * qnew)
    wtw = wtw + boterm - enterm - 2.0 * g * dz * pptdrg / 1.5
    wtw = jnp.where(jnp.abs(wtw) < 1.0e-4, 1.0e-4, wtw)

    qliq = ratio4 * qtot + ratio3 * 0.4 * qnew
    qice = (1.0 - ratio4) * qtot + (1.0 - ratio3) * 0.4 * qnew
    return qliq, qice, wtw, qlqout, qicout, 0.0 * qnewlq, 0.0 * qnewic


def prof5(eq):
    """Faithful PROF5: Gaussian-mixing entrainment/detrainment fractions.
    Returns (ee, ud)."""
    sqrt2p = 2.506628
    a1 = 0.4361836
    a2 = -0.1201676
    a3 = 0.9372980
    p = 0.33267
    sigma = 0.166666667
    fe = 0.202765151
    y = 6.0 * eq - 3.0
    ey = jnp.exp(y * y / (-2.0))
    e45 = jnp.exp(-4.5)
    t2 = 1.0 / (1.0 + p * jnp.abs(y))
    t1 = 0.500498
    c1 = a1 * t1 + a2 * t1 * t1 + a3 * t1 * t1 * t1
    c2 = a1 * t2 + a2 * t2 * t2 + a3 * t2 * t2 * t2

    ee_pos = sigma * (0.5 * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * eq * eq / 2.0
    ud_pos = sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
    ee_neg = sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * eq * eq / 2.0
    ud_neg = sigma * (0.5 * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)

    ee = jnp.where(y >= 0.0, ee_pos, ee_neg) / fe
    ud = jnp.where(y >= 0.0, ud_pos, ud_neg) / fe
    return ee, ud


# ============================================================================
# JAX single-column KF_eta_PARA (production, GPU-resident, vmappable).
#
# Faithful translation of cumulus_kf_reference.kf_eta_para_np using jax.lax
# control flow so the WHOLE column scheme traces to one XLA graph and vmaps
# across (i,j) with zero host/device transfer. The reference (validated 4/4 vs
# the Fortran oracle) is the line-for-line spec; this module reproduces the same
# algorithm with lax.while_loop (USL search, updraft, downdraft, closure) and
# lax.fori_loop (level sweeps, advection substeps), masked with jnp.where.
#
# Static (compile-time) bounds: KX (levels), MAX_USL (USL candidates) = KX,
# MAX_CLOSURE = 10, MAX_NSTEP advection substeps (bounded, =200). All match the
# Fortran's own bounds, so no behaviour is truncated for the supported d01 grid.
# ============================================================================

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
ALIQ = SVP1 * 1000.0
BLIQ = SVP2
CLIQ = SVP2 * SVPT0
DLIQ = SVP3
GDRY = -G / CP



def _empty_col(KX, nca):
    z = jnp.zeros(KX)
    return dict(DTDT=z, DQDT=z, DQCDT=z, DQRDT=z, DQIDT=z, DQSDT=z,
                RTHCUTEN=z, RQVCUTEN=z, RQCCUTEN=z, RQRCUTEN=z, RQICUTEN=z, RQSCUTEN=z,
                RAINCV=jnp.float64(0.0), PRATEC=jnp.float64(0.0), NCA=jnp.float64(nca),
                CUTOP=jnp.float64(1.0), CUBOT=jnp.float64(KX + 1), ISHALL=jnp.int32(2),
                TIMEC=jnp.float64(0.0))


def _run_updraft(NUcand, KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0,
                 P0p, W0Ap, dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu):
    """Run ONE updraft from candidate USL index NUcand (1-based into KCHECK).
    Returns a dict of outcome flags + full updraft fields. All branches are
    masked; nothing here mutates global state. Mirrors the validated reference.
    """
    N = lev.shape[0]
    KMIX = KCHECK[NUcand]
    LC = KMIX
    # 50-hPa source layer depth/KPBL
    inlayer = lev & (idx >= LC)
    dpc = jnp.cumsum(jnp.where(inlayer, DP, 0.0))
    # NLAYRS = first count where cumulative > DPMIN (starting at LC)
    reached = inlayer & (dpc > _DPMIN)
    KPBL = jnp.min(jnp.where(reached, idx, KX + 99))
    KPBL = jnp.clip(KPBL, LC, KX)
    DPTHMX = dpc[KPBL]
    valid_depth = DPTHMX >= _DPMIN
    msk = lev & (idx >= LC) & (idx <= KPBL)
    DPTHMX_safe = jnp.where(DPTHMX > 0, DPTHMX, 1.0)
    TMIX = jnp.sum(jnp.where(msk, DP * T0p, 0.0)) / DPTHMX_safe
    QMIX = jnp.sum(jnp.where(msk, DP * Q0, 0.0)) / DPTHMX_safe
    ZMIX = jnp.sum(jnp.where(msk, DP * Z0, 0.0)) / DPTHMX_safe
    PMIX = jnp.sum(jnp.where(msk, DP * P0p, 0.0)) / DPTHMX_safe
    EMIX = QMIX * PMIX / (0.622 + QMIX)
    astrt = 1.0e-3; ainc = 0.075
    a1 = EMIX / aliq
    tp = (a1 - astrt) / ainc
    indlu = jnp.clip(jnp.floor(tp).astype(jnp.int32), 0, 198)
    value = indlu * ainc + astrt
    aintrp = (a1 - value) / ainc
    tlog = aintrp * alu[indlu + 1] + (1.0 - aintrp) * alu[indlu]
    TDPT = (cliq - dliq * tlog) / (bliq - tlog)
    TLCL = TDPT - (0.212 + 1.571e-3 * (TDPT - _T00) - 4.36e-4 * (TMIX - _T00)) * (TMIX - TDPT)
    TLCL = jnp.minimum(TLCL, TMIX)
    TVLCL = TLCL * (1.0 + 0.608 * QMIX)
    ZLCL = ZMIX + (TLCL - TMIX) / GDRY
    ge = lev & (idx >= LC) & (ZLCL <= Z0)
    KLCL = jnp.clip(jnp.min(jnp.where(ge, idx, KX + 99)), LC, KL)
    off_top = ZLCL > Z0[KL]
    K = KLCL - 1
    DLP = (ZLCL - Z0[K]) / (Z0[KLCL] - Z0[K])
    TENV = T0p[K] + (T0p[KLCL] - T0p[K]) * DLP
    QENV = Q0[K] + (Q0[KLCL] - Q0[K]) * DLP
    TVEN = TENV * (1.0 + 0.608 * QENV)
    WKLCL = jnp.where(ZLCL < 2.0e3, 0.02 * ZLCL / 2.0e3, 0.02)
    WKL = (W0Ap[K] + (W0Ap[KLCL] - W0Ap[K]) * DLP) * dx / 25.0e3 - WKLCL
    DTLCL = jnp.where(WKL < 0.0001, 0.0, 4.64 * jnp.abs(WKL) ** 0.33)
    DTRH = 0.0
    buoyant = (TLCL + DTLCL + DTRH >= TENV) & valid_depth & (~off_top)

    THETEU_K = envirtht(PMIX, TMIX, QMIX, aliq, bliq, cliq, dliq)
    DTTOT = DTLCL + DTRH
    GDT = 2.0 * G * DTTOT * 500.0 / TVEN
    WLCL = jnp.where(DTTOT > 1.0e-4, jnp.minimum(1.0 + 0.5 * jnp.sqrt(jnp.abs(GDT)), 3.0), 1.0)
    PLCL = P0p[K] + (P0p[KLCL] - P0p[K]) * DLP
    TVLCL = TLCL * (1.0 + 0.608 * QMIX)
    RHOLCL = PLCL / (R_D * TVLCL)
    RAD = jnp.where(WKL < 0.0, 1000.0, jnp.where(WKL > 0.1, 2000.0, 1000.0 + 1000.0 * WKL / 0.1))
    AU0 = 0.01 * DXSQ
    VMFLCL = RHOLCL * AU0

    Zr = jnp.zeros(N)
    F = dict(UMF=Zr, UER=Zr, UDR=Zr, DETLQ=Zr, DETIC=Zr, PPTLIQ=Zr, PPTICE=Zr,
             QLIQ=Zr, QICE=Zr, QLQOUT=Zr, QICOUT=Zr, TU=Zr, TVU=Zr, QU=Zr, WU=Zr,
             THETEU=Zr, THETEE=Zr, TVQU=Zr, EQFRC=Zr, QDT=Zr, RATIO2=Zr,
             DILFRC=jnp.ones(N))
    F["WU"] = F["WU"].at[K].set(WLCL)
    F["UMF"] = F["UMF"].at[K].set(VMFLCL)
    F["TU"] = F["TU"].at[K].set(TLCL)
    F["TVU"] = F["TVU"].at[K].set(TVLCL)
    F["QU"] = F["QU"].at[K].set(QMIX)
    F["EQFRC"] = F["EQFRC"].at[K].set(1.0)
    F["THETEU"] = F["THETEU"].at[K].set(THETEU_K)

    carry = dict(F=F, WTW=WLCL * WLCL, UPOLD=VMFLCL, UPNEW=VMFLCL,
                 EE1=1.0, UD1=0.0, REI=0.0,
                 ABE=0.0, TRPPT=0.0, LET=KLCL, LTOP=K, TTEMP=_TTFRZ,
                 active=buoyant, IFLAG=0)

    def step(nk, c):
        F = dict(c["F"])
        NK1 = nk + 1
        in_range = (nk >= K) & (nk <= KL - 1)
        run = c["active"] & in_range
        WTW = c["WTW"]; UPOLD = c["UPOLD"]; REI = c["REI"]
        EE1 = c["EE1"]; UD1 = c["UD1"]; ABE = c["ABE"]; TRPPT = c["TRPPT"]
        LET = c["LET"]; TTEMP = c["TTEMP"]; IFLAG = c["IFLAG"]
        F["RATIO2"] = F["RATIO2"].at[NK1].set(jnp.where(run, F["RATIO2"][nk], F["RATIO2"][NK1]))
        tu = T0p[NK1]; thteu = F["THETEU"][nk]; qu = F["QU"][nk]
        qliq = F["QLIQ"][nk]; qice = F["QICE"][nk]
        tu, qu, qliq, qice, qnewlq, qnewic = tpmix2(P0p[NK1], thteu, tu, qu, qliq, qice, XLV1, XLV0)
        frz = tu <= _TTFRZ
        gtb = tu > _TBFRZ
        TTEMP_f = jnp.where(TTEMP > _TTFRZ, _TTFRZ, TTEMP)
        FRC1 = jnp.where(frz, jnp.where(gtb, (TTEMP_f - tu) / (TTEMP_f - _TBFRZ), 1.0), 0.0)
        IFLAG = jnp.where(run & frz & (~gtb), 1, IFLAG)
        TTEMP = jnp.where(run & frz, tu, TTEMP)
        QFRZ = (qliq + qnewlq) * FRC1
        qnewic_f = qnewic + qnewlq * FRC1
        qnewlq_f = qnewlq - qnewlq * FRC1
        qice_a = qice + qliq * FRC1
        qliq_a = qliq - qliq * FRC1
        tu2, thteu2, qu2, qice2 = dtfrznew(tu, P0p[NK1], thteu, qu, QFRZ, qice_a, aliq, bliq, cliq, dliq)
        tu = jnp.where(frz, tu2, tu)
        thteu = jnp.where(frz, thteu2, thteu)
        qu = jnp.where(frz, qu2, qu)
        qice = jnp.where(frz, qice2, jnp.where(frz, qice_a, qice))
        qliq = jnp.where(frz, qliq_a, qliq)
        qnewlq = jnp.where(frz, qnewlq_f, qnewlq)
        qnewic = jnp.where(frz, qnewic_f, qnewic)
        TVU1 = tu * (1.0 + 0.608 * qu)
        atK = nk == K
        BE = jnp.where(atK, (TVLCL + TVU1) / (TVEN + TV0[NK1]) - 1.0,
                       (F["TVU"][nk] + TVU1) / (TV0[nk] + TV0[NK1]) - 1.0)
        DZZ = jnp.where(atK, Z0[NK1] - ZLCL, DZA[nk])
        BOTERM = jnp.where(atK, 2.0 * (Z0[NK1] - ZLCL) * G * BE / 1.5, 2.0 * DZA[nk] * G * BE / 1.5)
        ENTERM = 2.0 * REI * WTW / jnp.where(UPOLD != 0, UPOLD, 1.0)
        qliq, qice, WTW2, qlqout, qicout, qnewlq, qnewic = condload(
            qliq, qice, WTW, DZZ, BOTERM, ENTERM, _RATE, qnewlq, qnewic, G)
        wtw_break = WTW2 < 1.0e-3
        F["QLQOUT"] = F["QLQOUT"].at[NK1].set(jnp.where(run, qlqout, F["QLQOUT"][NK1]))
        F["QICOUT"] = F["QICOUT"].at[NK1].set(jnp.where(run, qicout, F["QICOUT"][NK1]))

        thtee = envirtht(P0p[NK1], T0p[NK1], Q0[NK1], aliq, bliq, cliq, dliq)
        REI2 = VMFLCL * DP[NK1] * 0.03 / RAD
        TVQU1 = tu * (1.0 + 0.608 * qu - qliq - qice)
        DILBE = jnp.where(atK, ((TVLCL + TVQU1) / (TVEN + TV0[NK1]) - 1.0) * DZZ,
                          ((F["TVQU"][nk] + TVQU1) / (TV0[nk] + TV0[NK1]) - 1.0) * DZZ)
        ABE2 = jnp.where(DILBE > 0.0, ABE + DILBE * G, ABE)

        colder = TVQU1 <= TV0[NK1]
        thttmp = 0.95 * thtee + 0.05 * thteu
        qtmp = 0.95 * Q0[NK1] + 0.05 * qu
        tl = 0.05 * qliq; ti = 0.05 * qice
        t95, q95, l95, i95, _a, _b = tpmix2(P0p[NK1], thttmp, TVQU1, qtmp, tl, ti, XLV1, XLV0)
        TU95 = t95 * (1.0 + 0.608 * q95 - l95 - i95)
        warm95 = TU95 > TV0[NK1]
        thttmpb = 0.10 * thtee + 0.90 * thteu
        qtmpb = 0.10 * Q0[NK1] + 0.90 * qu
        tlb = 0.90 * qliq; tib = 0.90 * qice
        t10, q10, l10, i10, _c, _d = tpmix2(P0p[NK1], thttmpb, TVQU1, qtmpb, tlb, tib, XLV1, XLV0)
        TU10 = t10 * (1.0 + 0.608 * q10 - l10 - i10)
        TVDIFF = jnp.abs(TU10 - TVQU1)
        denom = jnp.where(jnp.abs(TU10 - TVQU1) > 0, TU10 - TVQU1, 1.0)
        eqfrc_raw = jnp.minimum(1.0, jnp.maximum(0.0, (TV0[NK1] - TVQU1) * 0.10 / denom))
        ee5, ud5 = prof5(eqfrc_raw)
        EE2 = jnp.where(colder, 0.5, jnp.where(warm95, 1.0, jnp.where(TVDIFF < 1.0e-3, 1.0,
              jnp.where(eqfrc_raw == 1.0, 1.0, jnp.where(eqfrc_raw == 0.0, 0.0, ee5)))))
        UD2 = jnp.where(colder, 1.0, jnp.where(warm95, 0.0, jnp.where(TVDIFF < 1.0e-3, 0.0,
              jnp.where(eqfrc_raw == 1.0, 0.0, jnp.where(eqfrc_raw == 0.0, 1.0, ud5)))))
        EQFRC1 = jnp.where(colder, 0.0, jnp.where(warm95, 1.0, jnp.where(TVDIFF < 1.0e-3, 1.0, eqfrc_raw)))
        LET_new = jnp.where(run & (~colder), NK1, LET)
        EE2 = jnp.maximum(EE2, 0.5)
        UD2 = 1.5 * UD2
        uer = 0.5 * REI2 * (EE1 + EE2)
        udr = 0.5 * REI2 * (UD1 + UD2)
        det_break = (F["UMF"][nk] - udr) < 10.0
        ABE3 = jnp.where(det_break & (DILBE > 0.0), ABE2 - DILBE * G, ABE2)
        LET_fin = jnp.where(det_break, nk, LET_new)
        stop = wtw_break | det_break

        # commit-on-continue updates (Fortran ELSE block), gated by run & ~stop
        cont = run & (~stop)
        UPOLD2 = F["UMF"][nk] - udr
        UPNEW = UPOLD2 + uer
        DILF = UPNEW / jnp.where(UPOLD2 != 0, UPOLD2, 1.0)
        qu_new = (UPOLD2 * qu + uer * Q0[NK1]) / jnp.where(UPNEW != 0, UPNEW, 1.0)
        thteu_new = (thteu * UPOLD2 + thtee * uer) / jnp.where(UPNEW != 0, UPNEW, 1.0)
        qliq_new = qliq * UPOLD2 / jnp.where(UPNEW != 0, UPNEW, 1.0)
        qice_new = qice * UPOLD2 / jnp.where(UPNEW != 0, UPNEW, 1.0)
        pptl = qlqout * F["UMF"][nk]
        ppti = qicout * F["UMF"][nk]
        uer_pbl = jnp.where(NK1 <= KPBL, uer + VMFLCL * DP[NK1] / DPTHMX_safe, uer)

        def setc(key, val):  # set F[key][NK1]=val only where cont
            F[key] = F[key].at[NK1].set(jnp.where(cont, val, F[key][NK1]))
        setc("UMF", UPNEW); setc("DILFRC", DILF)
        setc("DETLQ", qliq * udr); setc("DETIC", qice * udr)
        setc("QDT", qu); setc("QU", qu_new); setc("THETEU", thteu_new)
        setc("QLIQ", qliq_new); setc("QICE", qice_new)
        setc("PPTLIQ", pptl); setc("PPTICE", ppti)
        setc("UER", uer_pbl); setc("UDR", udr)
        setc("TU", tu); setc("TVU", TVU1); setc("TVQU", TVQU1)
        setc("THETEE", thtee); setc("EQFRC", EQFRC1)
        # WU set on continue (non-break) per Fortran (WU(NK1)=sqrt(WTW))
        F["WU"] = F["WU"].at[NK1].set(jnp.where(cont, jnp.sqrt(jnp.abs(WTW2)), F["WU"][NK1]))

        TRPPT2 = jnp.where(cont, TRPPT + pptl + ppti, TRPPT)
        WTWn = jnp.where(run, WTW2, WTW)
        EE1n = jnp.where(cont, EE2, EE1)
        UD1n = jnp.where(cont, UD2, UD1)
        UPOLDn = jnp.where(cont, UPOLD2, UPOLD)
        UPNEWn = jnp.where(cont, UPNEW, c["UPNEW"])
        REIn = jnp.where(cont, REI2, REI)
        ABEn = jnp.where(run, jnp.where(det_break, ABE3, ABE2), ABE)
        LETn = jnp.where(run, LET_fin, LET)
        # LTOP recorded at first stop
        first_stop = run & stop & c["active"]
        LTOPn = jnp.where(first_stop, nk, c["LTOP"])
        activen = jnp.where(run & stop, False, c["active"])
        return dict(F=F, WTW=WTWn, UPOLD=UPOLDn, UPNEW=UPNEWn, EE1=EE1n, UD1=UD1n, REI=REIn,
                    ABE=ABEn, TRPPT=TRPPT2, LET=LETn, LTOP=LTOPn, TTEMP=TTEMP,
                    active=activen, IFLAG=IFLAG)

    carry = jax.lax.fori_loop(0, KL, step, carry)
    LTOP = jnp.where(carry["active"], KL, carry["LTOP"])
    F = carry["F"]
    ABE = carry["ABE"]; TRPPT = carry["TRPPT"]; LET = carry["LET"]
    CLDHGT_LC = Z0[LTOP] - ZLCL
    CHMIN = jnp.where(TLCL > 293.0, 4.0e3, jnp.where(TLCL >= 273.0, 2.0e3 + 100.0 * (TLCL - 273.0), 2.0e3))
    no_conv = (LTOP <= KLCL) | (LTOP <= KPBL) | (LET + 1 <= KPBL)
    deep = buoyant & (~no_conv) & (CLDHGT_LC > CHMIN) & (ABE > 1.0)
    shallow = buoyant & (~no_conv) & (~deep)
    abort = (~valid_depth) | off_top
    return dict(F=F, abort=abort, buoyant=buoyant, no_conv=no_conv, deep=deep, shallow=shallow,
                CLDHGT_LC=jnp.where(no_conv | (~buoyant), 0.0, CLDHGT_LC),
                LC=LC, K=K, KLCL=KLCL, KPBL=KPBL, LET=LET, LTOP=LTOP, LCL=KLCL,
                ZLCL=ZLCL, TLCL=TLCL, TVLCL=TVLCL, TVEN=TVEN, VMFLCL=VMFLCL,
                WLCL=WLCL, RAD=RAD, ABE=ABE, TRPPT=TRPPT, DPTHMX=DPTHMX, AU0=AU0,
                ZMIX=ZMIX, TMIX=TMIX, QMIX=QMIX, PMIX=PMIX, WKL=WKL, PLCL=PLCL,
                UPOLD=carry["UPOLD"], UPNEW=carry["UPNEW"])


def _empty_candidate_state(N, KX):
    z = jnp.zeros(N, dtype=jnp.float64)
    one = jnp.asarray(1, dtype=jnp.int64)
    zero = jnp.asarray(0, dtype=jnp.int64)
    F = dict(UMF=z, UER=z, UDR=z, DETLQ=z, DETIC=z, PPTLIQ=z, PPTICE=z,
             QLIQ=z, QICE=z, QLQOUT=z, QICOUT=z, TU=z, TVU=z, QU=z, WU=z,
             THETEU=z, THETEE=z, TVQU=z, EQFRC=z, QDT=z, RATIO2=z,
             DILFRC=jnp.ones(N, dtype=jnp.float64))
    return dict(F=F, abort=jnp.array(False), buoyant=jnp.array(False),
                no_conv=jnp.array(True), deep=jnp.array(False), shallow=jnp.array(False),
                CLDHGT_LC=jnp.float64(0.0),
                LC=one, K=zero, KLCL=one, KPBL=one, LET=one, LTOP=one, LCL=one,
                ZLCL=jnp.float64(0.0), TLCL=jnp.float64(0.0), TVLCL=jnp.float64(0.0),
                TVEN=jnp.float64(0.0), VMFLCL=jnp.float64(0.0), WLCL=jnp.float64(0.0),
                RAD=jnp.float64(0.0), ABE=jnp.float64(0.0), TRPPT=jnp.float64(0.0),
                DPTHMX=jnp.float64(0.0), AU0=jnp.float64(0.0), ZMIX=jnp.float64(0.0),
                TMIX=jnp.float64(0.0), QMIX=jnp.float64(0.0), PMIX=jnp.float64(0.0),
                WKL=jnp.float64(0.0), PLCL=jnp.float64(0.0),
                UPOLD=jnp.float64(0.0), UPNEW=jnp.float64(0.0))


def _tree_where(pred, on_true, on_false):
    return jax.tree_util.tree_map(lambda a, b: jnp.where(pred, a, b), on_true, on_false)


def _search_usl(KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, W0Ap,
                dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu):
    """Fortran-faithful USL walk.

    Deep convection stops on the first triggering source layer. If only shallow
    candidates are found, scan all candidates once to find NUCHM (max cloud
    height), then rerun that candidate and stop there, matching KF_eta_PARA.
    """
    empty = _empty_candidate_state(idx.shape[0], KX)
    init = dict(nu=jnp.int32(1), phase=jnp.int32(0), have_shallow=jnp.array(False),
                best_nu=jnp.int32(0), best_height=jnp.float64(-1.0),
                selected=empty, convect=jnp.array(False), ishall=jnp.int32(2),
                done=jnp.array(False))

    def run_current(st):
        cand = _run_updraft(st["nu"], KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0,
                            P0p, W0Ap, dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu)
        phase1 = st["phase"] == 1
        deep = cand["deep"] & (~cand["abort"])
        shallow = cand["shallow"] & (~cand["abort"])
        accept_shallow = phase1 & shallow
        accept = deep | accept_shallow
        better_shallow = (st["phase"] == 0) & shallow & (
            (~st["have_shallow"]) | (cand["CLDHGT_LC"] > st["best_height"]))
        terminal_miss = phase1 & (~deep) & (~shallow)

        done = cand["abort"] | accept | terminal_miss
        return dict(
            nu=jnp.where(done, st["nu"], st["nu"] + 1),
            phase=st["phase"],
            have_shallow=st["have_shallow"] | ((st["phase"] == 0) & shallow),
            best_nu=jnp.where(better_shallow, st["nu"], st["best_nu"]),
            best_height=jnp.where(better_shallow, cand["CLDHGT_LC"], st["best_height"]),
            selected=_tree_where(accept, cand, st["selected"]),
            convect=accept,
            ishall=jnp.where(deep, jnp.int32(0), jnp.where(accept_shallow, jnp.int32(1), st["ishall"])),
            done=done,
        )

    def exhausted(st):
        rerun_shallow = (st["phase"] == 0) & st["have_shallow"]
        return dict(
            nu=jnp.where(rerun_shallow, st["best_nu"], st["nu"]),
            phase=jnp.where(rerun_shallow, jnp.int32(1), st["phase"]),
            have_shallow=st["have_shallow"],
            best_nu=st["best_nu"],
            best_height=st["best_height"],
            selected=st["selected"],
            convect=st["convect"],
            ishall=st["ishall"],
            done=~rerun_shallow,
        )

    def body(st):
        return jax.lax.cond(st["nu"] <= NCHECK, run_current, exhausted, st)

    def cond(st):
        return ~st["done"]

    out = jax.lax.while_loop(cond, body, init)
    return out["selected"], out["ishall"], out["convect"]


@partial(jax.jit, static_argnums=(10, 11, 12, 13))
def kf_eta_para(T0, QV0, P0, DZQ, RHOE, W0A, U0, V0, dt, dx,
                KX, warm_rain=False, f_qi=True, f_qs=True):
    """JAX KF_eta_PARA for ONE column (inputs length KX, 0-based, bottom-up).

    Faithful translation of the validated NumPy reference. All control flow is
    masked / lax-based; vmappable; GPU-resident."""
    N = KX + 3
    KL = KX
    DXSQ = dx * dx
    aliq, bliq, cliq, dliq = ALIQ, BLIQ, CLIQ, DLIQ
    alu = jnp.asarray(_T.ALU)

    def pad(a):
        return jnp.concatenate([jnp.zeros(1), jnp.asarray(a, jnp.float64), jnp.zeros(N - KX - 1)])
    T0p = pad(T0); QV0p = pad(QV0); P0p = pad(P0); DZQp = pad(DZQ)
    RHOEp = pad(RHOE); W0Ap = pad(W0A); U0p = pad(U0); V0p = pad(V0)
    idx = jnp.arange(N)
    lev = (idx >= 1) & (idx <= KX)

    es = aliq * jnp.exp((bliq * T0p - cliq) / jnp.where(lev, T0p - dliq, 1.0))
    QES = jnp.where(lev, 0.622 * es / (P0p - es), 0.0)
    Q0 = jnp.where(lev, jnp.maximum(0.000001, jnp.minimum(QES, QV0p)), 0.0)
    RH = jnp.where(lev, Q0 / jnp.where(QES > 0, QES, 1.0), 0.0)
    TV0 = jnp.where(lev, T0p * (1.0 + 0.608 * Q0), 0.0)
    DP = jnp.where(lev, RHOEp * G * DZQp, 0.0)
    cdz = jnp.cumsum(jnp.where(lev, DZQp, 0.0))
    Z0 = jnp.where(lev, cdz - 0.5 * DZQp, 0.0)
    DZA = jnp.zeros(N).at[1:N - 1].set(jnp.where(lev[1:N - 1], Z0[2:N] - Z0[1:N - 1], 0.0))
    P300 = P0p[1] - 30000.0
    L5 = jnp.max(jnp.where(lev & (P0p >= 0.5 * P0p[1]), idx, 0))
    LLFC = jnp.max(jnp.where(lev & (P0p >= P300), idx, 0))

    def kcheck_body(k, carry):
        kchk, ncheck, pm15 = carry
        cond = (k <= LLFC) & (P0p[k] < pm15)
        ncheck2 = jnp.where(cond, ncheck + 1, ncheck)
        kchk2 = jnp.where(cond, kchk.at[ncheck2].set(k), kchk)
        pm15_2 = jnp.where(cond, pm15 - 15.0e2, pm15)
        return (kchk2, ncheck2, pm15_2)
    KCHK0 = jnp.zeros(N, dtype=jnp.int32).at[1].set(1)
    KCHECK, NCHECK, _ = jax.lax.fori_loop(2, KX + 1, kcheck_body, (KCHK0, jnp.int32(1), P0p[1] - 15.0e2))

    S, ISHALL, convect = _search_usl(
        KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, W0Ap,
        dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu)

    NCA_none = jnp.float64(-100.0)
    # ===== closure + feedback (only if convect); else empty =====
    out_conv = _closure_feedback(S, ISHALL, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p,
                                 QES, RH, U0p, V0p, dt, dx, DXSQ, KX, KL, L5,
                                 aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs)
    empty = _empty_col(KX, -100.0)
    out = {k: jnp.where(convect, out_conv[k], empty[k]) if out_conv[k].ndim == 1
           else jnp.where(convect, out_conv[k], empty[k]) for k in empty}
    out["ISHALL"] = jnp.where(convect, ISHALL, jnp.int32(2))
    return out


def update_w0avg(w0avg, w, dt, *, stepcu=5, cudt=0.0, adapt_step_flag=False):
    """WRF ``KF_eta_CPS`` running-mean vertical velocity recurrence.

    ``w`` may be a full-level column of length ``KX+1`` or an already collapsed
    mass-level ``W0`` column of length ``KX``. ``cudt`` is WRF namelist minutes.
    """

    w0avg = jnp.asarray(w0avg, jnp.float64)
    kx = int(w0avg.shape[0])
    w_arr = jnp.asarray(w, jnp.float64)
    if w_arr.shape[0] == kx + 1:
        w0 = _collapse_w_to_mass_levels(w_arr, kx)
    elif w_arr.shape[0] == kx:
        w0 = w_arr
    else:
        raise ValueError(f"w must have length KX or KX+1, got {w_arr.shape[0]} for KX={kx}")

    if adapt_step_flag:
        window = 2.0 * max(float(cudt) * 60.0, float(dt))
        return (w0avg * (window - float(dt)) + w0 * float(dt)) / window

    tst = float(int(stepcu) * 2)
    return (w0avg * (tst - 1.0) + w0) / tst


def step_kf_column(
    T0,
    QV0,
    P0,
    DZQ,
    RHOE,
    w0avg,
    U0,
    V0,
    dt,
    dx,
    *,
    w=None,
    nca=-100.0,
    stepcu=5,
    cudt=0.0,
    adapt_step_flag=False,
    warm_rain=False,
    f_qi=True,
    f_qs=True,
):
    """Run one v0.6.0 KF column and return the frozen physics interface object.

    The savepoint replay path passes ``w=None`` because the oracle stores the
    ``W0AVG`` column after WRF's CPS recurrence. Production coupling should pass
    full-level ``w`` so this function updates the carry before the trigger gate.
    """

    w0avg_in = jnp.asarray(w0avg, jnp.float64)
    kx = int(w0avg_in.shape[0])
    w0avg_call = (
        w0avg_in
        if w is None
        else update_w0avg(w0avg_in, w, dt, stepcu=stepcu, cudt=cudt, adapt_step_flag=adapt_step_flag)
    )
    nca_in = jnp.asarray(nca, jnp.float64)

    def run(_):
        return kf_eta_para(T0, QV0, P0, DZQ, RHOE, w0avg_call, U0, V0, dt, dx, kx, warm_rain, f_qi, f_qs)

    def skip(_):
        return _empty_col(kx, nca_in)

    out = jax.lax.cond(nca_in < 0.5 * float(dt), run, skip, operand=None)
    zeros = jnp.zeros_like(out["RTHCUTEN"])
    state_tendencies = {
        "u": zeros,
        "v": zeros,
        "theta": out["RTHCUTEN"],
        "qv": out["RQVCUTEN"],
        "qc": out["RQCCUTEN"],
        "qr": out["RQRCUTEN"],
        "qi": out["RQICUTEN"],
        "qs": out["RQSCUTEN"],
    }
    cumulus_diagnostics = {
        "rucuten": zeros,
        "rvcuten": zeros,
        "rthcuten": out["RTHCUTEN"],
        "rqvcuten": out["RQVCUTEN"],
        "rqccuten": out["RQCCUTEN"],
        "rqrcuten": out["RQRCUTEN"],
        "rqicuten": out["RQICUTEN"],
        "rqscuten": out["RQSCUTEN"],
        "raincv": out["RAINCV"],
        "pratec": out["PRATEC"],
        "cutop": out["CUTOP"],
        "cubot": out["CUBOT"],
        "ishall": out["ISHALL"],
        "timec": out["TIMEC"],
    }
    tendency = PhysicsTendency(
        state_tendencies=state_tendencies,
        accumulator_increments={"rainc_acc": out["RAINCV"]},
    )
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus={"w0avg": w0avg_call, "nca": out["NCA"]}),
        diagnostics=PhysicsDiagnostics(cumulus=cumulus_diagnostics),
    )


def _closure_feedback(S, ISHALL, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p,
                      QES, RH, U0p, V0p, dt, dx, DXSQ, KX, KL, L5,
                      aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs):
    """Downdraft + mass-flux closure + advection + feedback tendencies.
    S = selected-candidate state from _run_updraft. Mirrors the validated
    reference post-USL section. Returns the output dict (length-KX tendencies)."""
    N = idx.shape[0]
    F = S["F"]
    LC = S["LC"]; K = S["K"]; KLCL = S["KLCL"]; KPBL = S["KPBL"]
    LET = S["LET"]; LTOP = S["LTOP"]; LCL = S["LCL"]
    ZLCL = S["ZLCL"]; TLCL = S["TLCL"]; TVLCL = S["TVLCL"]; TVEN = S["TVEN"]
    VMFLCL = S["VMFLCL"]; WLCL = S["WLCL"]; RAD = S["RAD"]; ABE = S["ABE"]
    TRPPT = S["TRPPT"]; DPTHMX = S["DPTHMX"]; AU0 = S["AU0"]; ZMIX = S["ZMIX"]
    TMIX = S["TMIX"]; QMIX = S["QMIX"]; PMIX = S["PMIX"]; PLCL = S["PLCL"]
    UPOLD = S["UPOLD"]; UPNEW = S["UPNEW"]
    shallow = ISHALL == 1
    is_lev = lambda k: (k >= 1) & (k <= LTOP)

    UMF = F["UMF"]; UER = F["UER"]; UDR = F["UDR"]; DETLQ = F["DETLQ"]; DETIC = F["DETIC"]
    PPTLIQ = F["PPTLIQ"]; PPTICE = F["PPTICE"]; QLIQ = F["QLIQ"]; QICE = F["QICE"]
    QLQOUT = F["QLQOUT"]; QICOUT = F["QICOUT"]; TU = F["TU"]; QU = F["QU"]; WU = F["WU"]
    THETEU = F["THETEU"]; THETEE = F["THETEE"]; QDT = F["QDT"]; DILFRC = F["DILFRC"]
    EQFRC = F["EQFRC"]

    # for shallow: KSTART, LET=KSTART
    KSTART_sh = jnp.maximum(KPBL, KLCL)
    LET = jnp.where(shallow, KSTART_sh, LET)
    # --- LET==LTOP vs linear-detrainment top ---
    let_eq = LET == LTOP
    # LET==LTOP branch
    udr_top = UMF[LTOP] + UDR[LTOP] - UER[LTOP]
    ratio_up = UPNEW / jnp.where(UPOLD != 0, UPOLD, 1.0)
    detlq_top = QLIQ[LTOP] * udr_top * ratio_up
    detic_top = QICE[LTOP] * udr_top * ratio_up
    UDR = UDR.at[LTOP].set(jnp.where(let_eq, udr_top, UDR[LTOP]))
    DETLQ = DETLQ.at[LTOP].set(jnp.where(let_eq, detlq_top, DETLQ[LTOP]))
    DETIC = DETIC.at[LTOP].set(jnp.where(let_eq, detic_top, DETIC[LTOP]))
    UER = UER.at[LTOP].set(jnp.where(let_eq, 0.0, UER[LTOP]))
    UMF = UMF.at[LTOP].set(jnp.where(let_eq, 0.0, UMF[LTOP]))

    # linear-detrainment branch (LET<LTOP): DO NK=LET+1,LTOP
    DPTT = jnp.sum(jnp.where(lev & (idx >= LET + 1) & (idx <= LTOP), DP, 0.0))
    DUMFDP = UMF[LET] / jnp.where(DPTT != 0, DPTT, 1.0)

    def lindet(nk, carry):
        UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT = carry
        do = (~let_eq) & (nk >= LET + 1) & (nk <= LTOP)
        is_top = nk == LTOP
        # non-top
        umf_nt = UMF[nk - 1] - DP[nk] * DUMFDP
        uer_nt = umf_nt * (1.0 - 1.0 / jnp.where(DILFRC[nk] != 0, DILFRC[nk], 1.0))
        udr_nt = UMF[nk - 1] - umf_nt + uer_nt
        # top
        udr_tp = UMF[nk - 1]
        umf_new = jnp.where(do, jnp.where(is_top, UMF[nk], umf_nt), UMF[nk])
        uer_new = jnp.where(do, jnp.where(is_top, 0.0, uer_nt), UER[nk])
        udr_new = jnp.where(do, jnp.where(is_top, udr_tp, udr_nt), UDR[nk])
        detlq_new = jnp.where(do, udr_new * QLIQ[nk] * DILFRC[nk], DETLQ[nk])
        detic_new = jnp.where(do, udr_new * QICE[nk] * DILFRC[nk], DETIC[nk])
        UMF = UMF.at[nk].set(umf_new); UER = UER.at[nk].set(uer_new); UDR = UDR.at[nk].set(udr_new)
        DETLQ = DETLQ.at[nk].set(detlq_new); DETIC = DETIC.at[nk].set(detic_new)
        # precip reassign for NK>=LET+2
        do2 = do & (nk >= LET + 2)
        TRPPT = jnp.where(do2, TRPPT - PPTLIQ[nk] - PPTICE[nk], TRPPT)
        pptl = UMF[nk - 1] * QLQOUT[nk]
        ppti = UMF[nk - 1] * QICOUT[nk]
        PPTLIQ = PPTLIQ.at[nk].set(jnp.where(do2, pptl, PPTLIQ[nk]))
        PPTICE = PPTICE.at[nk].set(jnp.where(do2, ppti, PPTICE[nk]))
        TRPPT = jnp.where(do2, TRPPT + pptl + ppti, TRPPT)
        return (UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT)
    UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT = jax.lax.fori_loop(
        1, KX + 1, lindet, (UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT))

    # ML = highest level with T0>T00, up to LTOP
    ML = jnp.max(jnp.where(lev & (idx <= LTOP) & (T0p > _T00), idx, 0))

    # init below cloud base (DO NK=1,K)
    def below(nk, carry):
        UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, THETEE, EQFRC = carry
        do = (nk >= 1) & (nk <= K)
        ge_lc = nk >= LC
        at_lc = nk == LC
        le_pbl = nk <= KPBL
        uer_lc = VMFLCL * DP[nk] / DPTHMX
        umf_lc = uer_lc
        umf_pbl = UMF[nk - 1] + VMFLCL * DP[nk] / DPTHMX
        uer_pblv = VMFLCL * DP[nk] / DPTHMX
        umf_top = VMFLCL
        umf_v = jnp.where(at_lc, umf_lc, jnp.where(le_pbl, umf_pbl, umf_top))
        uer_v = jnp.where(at_lc, uer_lc, jnp.where(le_pbl, uer_pblv, 0.0))
        tu_v = TMIX + (Z0[nk] - ZMIX) * GDRY
        UMF = UMF.at[nk].set(jnp.where(do, jnp.where(ge_lc, umf_v, 0.0), UMF[nk]))
        UER = UER.at[nk].set(jnp.where(do, jnp.where(ge_lc, uer_v, 0.0), UER[nk]))
        TU = TU.at[nk].set(jnp.where(do, jnp.where(ge_lc, tu_v, 0.0), TU[nk]))
        QU = QU.at[nk].set(jnp.where(do, jnp.where(ge_lc, QMIX, 0.0), QU[nk]))
        WU = WU.at[nk].set(jnp.where(do, jnp.where(ge_lc, WLCL, 0.0), WU[nk]))
        UDR = UDR.at[nk].set(jnp.where(do, 0.0, UDR[nk]))
        QLIQ = QLIQ.at[nk].set(jnp.where(do, 0.0, QLIQ[nk]))
        QICE = QICE.at[nk].set(jnp.where(do, 0.0, QICE[nk]))
        QLQOUT = QLQOUT.at[nk].set(jnp.where(do, 0.0, QLQOUT[nk]))
        QICOUT = QICOUT.at[nk].set(jnp.where(do, 0.0, QICOUT[nk]))
        PPTLIQ = PPTLIQ.at[nk].set(jnp.where(do, 0.0, PPTLIQ[nk]))
        PPTICE = PPTICE.at[nk].set(jnp.where(do, 0.0, PPTICE[nk]))
        DETLQ = DETLQ.at[nk].set(jnp.where(do, 0.0, DETLQ[nk]))
        DETIC = DETIC.at[nk].set(jnp.where(do, 0.0, DETIC[nk]))
        thtee = envirtht(P0p[nk], T0p[nk], Q0[nk], aliq, bliq, cliq, dliq)
        THETEE = THETEE.at[nk].set(jnp.where(do, thtee, THETEE[nk]))
        EQFRC = EQFRC.at[nk].set(jnp.where(do, 1.0, EQFRC[nk]))
        return (UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, THETEE, EQFRC)
    (UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE,
     DETLQ, DETIC, THETEE, EQFRC) = jax.lax.fori_loop(
        1, KX + 1, below, (UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT,
                           PPTLIQ, PPTICE, DETLQ, DETIC, THETEE, EQFRC))

    # above cloud top + EMS/EXN/THTA arrays
    above = (idx >= LTOP + 1) & (idx <= KX)
    UMF = jnp.where(above, 0.0, UMF); UER = jnp.where(above, 0.0, UER); UDR = jnp.where(above, 0.0, UDR)
    QDT = jnp.where(above, 0.0, QDT); QLIQ = jnp.where(above, 0.0, QLIQ); QICE = jnp.where(above, 0.0, QICE)
    QLQOUT = jnp.where(above, 0.0, QLQOUT); QICOUT = jnp.where(above, 0.0, QICOUT)
    DETLQ = jnp.where(above, 0.0, DETLQ); DETIC = jnp.where(above, 0.0, DETIC)
    PPTLIQ = jnp.where(above, 0.0, PPTLIQ); PPTICE = jnp.where(above, 0.0, PPTICE)
    abv2 = (idx >= LTOP + 2) & (idx <= KX)
    TU = jnp.where(abv2, 0.0, TU); QU = jnp.where(abv2, 0.0, QU); WU = jnp.where(abv2, 0.0, WU)

    inLT = lev & (idx <= LTOP)
    EMS = jnp.where(inLT, DP * DXSQ / G, 0.0)
    EMSD = jnp.where(inLT, 1.0 / jnp.where(EMS != 0, EMS, 1.0), 0.0)
    EXN_u = (_P00 / P0p) ** (0.2854 * (1.0 - 0.28 * QDT))
    THTAU = jnp.where(inLT, TU * EXN_u, 0.0)
    EXN0 = (_P00 / P0p) ** (0.2854 * (1.0 - 0.28 * Q0))
    THTA0 = jnp.where(inLT, T0p * EXN0, 0.0)
    DDILFRC = jnp.where(inLT, 1.0 / jnp.where(DILFRC != 0, DILFRC, 1.0), 0.0)
    THTAD = jnp.zeros(N)
    QD = jnp.zeros(N); TZ = jnp.zeros(N); DMF = jnp.zeros(N); DER = jnp.zeros(N); DDR = jnp.zeros(N)
    QSD = jnp.zeros(N); WD = jnp.zeros(N); TVD = jnp.zeros(N); THETED = jnp.zeros(N)

    # convective time scale
    WSk = jnp.sqrt(U0p[KLCL]**2 + V0p[KLCL]**2)
    WS5 = jnp.sqrt(U0p[L5]**2 + V0p[L5]**2)
    WSt = jnp.sqrt(U0p[LTOP]**2 + V0p[LTOP]**2)
    VCONV = 0.5 * (WSk + WS5)
    TIMEC0 = jnp.where(VCONV > 0, dx / VCONV, 1.0e30)
    TADVEC = TIMEC0
    TIMEC = jnp.minimum(3600.0, jnp.maximum(1800.0, TIMEC0))
    TIMEC = jnp.where(shallow, 2400.0, TIMEC)
    NIC = jnp.round(TIMEC / dt).astype(jnp.int32)
    TIMEC = NIC.astype(jnp.float64) * dt

    SHSIGN = jnp.where(WSt > WSk, 1.0, -1.0)
    VWS = (U0p[LTOP] - U0p[KLCL])**2 + (V0p[LTOP] - V0p[KLCL])**2
    VWS = 1.0e3 * SHSIGN * jnp.sqrt(VWS) / (Z0[LTOP] - Z0[LCL])
    PEF = 1.591 + VWS * (-0.639 + VWS * (9.53e-2 - VWS * 4.96e-3))
    PEF = jnp.minimum(0.9, jnp.maximum(0.2, PEF))
    CBH = (ZLCL - Z0[1]) * 3.281e-3
    RCBH = jnp.where(CBH < 3.0, 0.02,
           0.96729352 + CBH * (-0.70034167 + CBH * (0.162179896 + CBH * (-1.2569798e-2 + CBH * (4.2772e-4 - CBH * 5.44e-6)))))
    RCBH = jnp.where(CBH > 25.0, 2.4, RCBH)
    PEFCBH = jnp.minimum(1.0 / (1.0 + RCBH), 0.9)
    PEFF = 0.5 * (PEF + PEFCBH)

    # =================== DOWNDRAFT (deep only; shallow -> LFS=1) ===================
    KSTART = jnp.where(shallow, 1, KPBL + 1)   # only used in deep path
    # KLFS: first NK>KSTART with P0[KSTART]-P0[NK]>150hPa, else LET-1; min(.,LET-1)
    dppp = P0p[KSTART] - P0p
    klfs_cand = lev & (idx >= KSTART + 1) & (dppp > 150.0e2)
    KLFS = jnp.minimum(jnp.min(jnp.where(klfs_cand, idx, KX + 99)), LET - 1)
    KLFS = jnp.clip(KLFS, 1, KL)
    LFS = KLFS
    dd_ok = (~shallow) & ((P0p[KSTART] - P0p[LFS]) > 50.0e2)

    # LFS-level downdraft seed
    theted_lfs = THETEE[LFS]
    qd_lfs = Q0[LFS]
    tz_lfs, qss_lfs = tpmix2dd(P0p[LFS], theted_lfs)
    thtad_lfs = tz_lfs * (_P00 / P0p[LFS]) ** (0.2854 * (1.0 - 0.28 * qss_lfs))
    tvd_lfs = tz_lfs * (1.0 + 0.608 * qss_lfs)
    rdd = P0p[LFS] / (R_D * tvd_lfs)
    A1 = (1.0 - PEFF) * AU0
    dmf_lfs = -A1 * rdd
    THETED = THETED.at[LFS].set(theted_lfs)
    QD = QD.at[LFS].set(qd_lfs)
    TZ = TZ.at[LFS].set(tz_lfs)
    THTAD = THTAD.at[LFS].set(thtad_lfs)
    TVD = TVD.at[LFS].set(tvd_lfs)
    DMF = DMF.at[LFS].set(dmf_lfs)
    DER = DER.at[LFS].set(dmf_lfs)

    # descend LFS-1 .. KSTART building DMF/THETED/QD + RHBAR
    def dd_down(j, carry):
        DMF, DER, THETED, QD, rhbar, dptt = carry
        nd = LFS - j   # j=1..(LFS-KSTART)
        do = dd_ok & (nd >= KSTART) & (nd <= LFS - 1)
        nd1 = nd + 1
        der_v = DER[LFS] * EMS[nd] / jnp.where(EMS[LFS] != 0, EMS[LFS], 1.0)
        dmf_v = DMF[nd1] + der_v
        theted_v = (THETED[nd1] * DMF[nd1] + THETEE[nd] * der_v) / jnp.where(dmf_v != 0, dmf_v, 1.0)
        qd_v = (QD[nd1] * DMF[nd1] + Q0[nd] * der_v) / jnp.where(dmf_v != 0, dmf_v, 1.0)
        DER = DER.at[nd].set(jnp.where(do, der_v, DER[nd]))
        DMF = DMF.at[nd].set(jnp.where(do, dmf_v, DMF[nd]))
        THETED = THETED.at[nd].set(jnp.where(do, theted_v, THETED[nd]))
        QD = QD.at[nd].set(jnp.where(do, qd_v, QD[nd]))
        rhbar = jnp.where(do, rhbar + RH[nd] * DP[nd], rhbar)
        dptt = jnp.where(do, dptt + DP[nd], dptt)
        return (DMF, DER, THETED, QD, rhbar, dptt)
    DMF, DER, THETED, QD, RHBAR, DPTT = jax.lax.fori_loop(
        1, KX + 1, dd_down, (DMF, DER, THETED, QD, RH[LFS] * DP[LFS], DP[LFS]))
    RHBAR = RHBAR / jnp.where(DPTT != 0, DPTT, 1.0)
    DMFFRC = 2.0 * (1.0 - RHBAR)

    pptmlt = jnp.sum(jnp.where(lev & (idx >= KLCL) & (idx <= LTOP), PPTICE, 0.0))
    DTMELT = jnp.where(LC < ML, _RLF * pptmlt / (CP * jnp.where(UMF[KLCL] != 0, UMF[KLCL], 1.0)), 0.0)
    LDT = jnp.minimum(LFS - 1, KSTART - 1)
    tzks, qss_ks = tpmix2dd(P0p[KSTART], THETED[KSTART])
    tzks = tzks - DTMELT
    es_ks = aliq * jnp.exp((bliq * tzks - cliq) / (tzks - dliq))
    qss_ks = 0.622 * es_ks / (P0p[KSTART] - es_ks)
    theted_ks = tzks * (1.0e5 / P0p[KSTART]) ** (0.2854 * (1.0 - 0.28 * qss_ks)) * \
        jnp.exp((3374.6525 / tzks - 2.5403) * qss_ks * (1.0 + 0.81 * qss_ks))
    TZ = TZ.at[KSTART].set(jnp.where(dd_ok, tzks, TZ[KSTART]))
    THETED = THETED.at[KSTART].set(jnp.where(dd_ok, theted_ks, THETED[KSTART]))

    # descend LDT..1 to find LDB; specify RH decrease 20%/km
    def dd_evap(j, carry):
        TZ, QD, QSD, THTAD, TVD, ldb, found, dpdd = carry
        nd = LDT - j
        do = dd_ok & (nd >= 1) & (nd <= LDT) & (~found)
        dpdd = jnp.where(do, dpdd + DP[nd], dpdd)
        theted_nd = THETED[KSTART]
        qd_nd = QD[KSTART]
        tz_nd, qss = tpmix2dd(P0p[nd], theted_nd)
        RHH = 1.0 - 0.2 / 1000.0 * (Z0[KSTART] - Z0[nd])
        DSSDT = (cliq - bliq * dliq) / ((tz_nd - dliq) * (tz_nd - dliq))
        RL = XLV0 - XLV1 * tz_nd
        DTMP = RL * qss * (1.0 - RHH) / (CP + RL * RHH * qss * DSSDT)
        T1RH = tz_nd + DTMP
        es2 = RHH * aliq * jnp.exp((bliq * T1RH - cliq) / (T1RH - dliq))
        QSRH = 0.622 * es2 / (P0p[nd] - es2)
        belowq = QSRH < qd_nd
        QSRH2 = jnp.where(belowq, qd_nd, QSRH)
        T1RH2 = jnp.where(belowq, tz_nd + (qss - QSRH2) * RL / CP, T1RH)
        rhh_lt1 = RHH < 1.0
        tz_final = jnp.where(rhh_lt1, T1RH2, tz_nd)
        qss_final = jnp.where(rhh_lt1, QSRH2, qss)
        TZ = TZ.at[nd].set(jnp.where(do, tz_final, TZ[nd]))
        QD = QD.at[nd].set(jnp.where(do, qd_nd, QD[nd]))
        QSD = QSD.at[nd].set(jnp.where(do, qss_final, QSD[nd]))
        tvd_nd = tz_final * (1.0 + 0.608 * qss_final)
        TVD = TVD.at[nd].set(jnp.where(do, tvd_nd, TVD[nd]))
        hit = do & ((tvd_nd > TV0[nd]) | (nd == 1))
        ldb = jnp.where(hit & (~found), nd, ldb)
        found = found | hit
        return (TZ, QD, QSD, THTAD, TVD, ldb, found, dpdd)
    TZ, QD, QSD, THTAD, TVD, LDB, _found, DPDD = jax.lax.fori_loop(
        0, KX + 1, dd_evap, (TZ, QD, QSD, THTAD, TVD, jnp.asarray(1), jnp.array(False), jnp.float64(0.0)))

    dd_depth_ok = dd_ok & ((P0p[LDB] - P0p[LFS]) > 50.0e2)

    def dd_final(j, carry):
        DDR, DER, DMF, TDER, QD, THTAD = carry
        nd = LDT - j
        do = dd_depth_ok & (nd >= LDB) & (nd <= LDT)
        nd1 = nd + 1
        ddr_v = -DMF[KSTART] * DP[nd] / jnp.where(DPDD != 0, DPDD, 1.0)
        dmf_v = DMF[nd1] + ddr_v
        TDER = jnp.where(do, TDER + (QSD[nd] - QD[nd]) * ddr_v, TDER)
        qd_v = QSD[nd]
        thtad_v = TZ[nd] * (_P00 / P0p[nd]) ** (0.2854 * (1.0 - 0.28 * qd_v))
        DDR = DDR.at[nd].set(jnp.where(do, ddr_v, DDR[nd]))
        DER = DER.at[nd].set(jnp.where(do, 0.0, DER[nd]))
        DMF = DMF.at[nd].set(jnp.where(do, dmf_v, DMF[nd]))
        QD = QD.at[nd].set(jnp.where(do, qd_v, QD[nd]))
        THTAD = THTAD.at[nd].set(jnp.where(do, thtad_v, THTAD[nd]))
        return (DDR, DER, DMF, TDER, QD, THTAD)
    DDR, DER, DMF, TDER, QD, THTAD = jax.lax.fori_loop(
        0, KX + 1, dd_final, (DDR, DER, DMF, jnp.float64(0.0), QD, THTAD))

    # ---- no-downdraft vs downdraft ----
    no_dd = TDER < 1.0
    inLT_arr = (idx >= 1) & (idx <= LTOP)
    # no-dd: zero downdraft arrays, PPTFLX=TRPPT
    DMF = jnp.where(no_dd & inLT_arr, 0.0, DMF)
    DER = jnp.where(no_dd & inLT_arr, 0.0, DER)
    DDR = jnp.where(no_dd & inLT_arr, 0.0, DDR)
    THTAD = jnp.where(no_dd & inLT_arr, 0.0, THTAD)
    WD = jnp.where(no_dd & inLT_arr, 0.0, WD)
    TZ = jnp.where(no_dd & inLT_arr, 0.0, TZ)
    QD = jnp.where(no_dd & inLT_arr, 0.0, QD)
    LDB = jnp.where(no_dd, LFS, LDB)
    TDER = jnp.where(no_dd, 0.0, TDER)

    # downdraft branch scaling
    DDINC = -DMFFRC * UMF[KLCL] / jnp.where(DMF[KSTART] != 0, DMF[KSTART], 1.0)
    DDINC = jnp.where(TDER * DDINC > TRPPT, jnp.where(TDER != 0, TRPPT / TDER, DDINC), DDINC)
    TDER_dd = TDER * DDINC
    scale_dd = (~no_dd) & lev & (idx >= LDB) & (idx <= LFS)
    DMF = jnp.where(scale_dd, DMF * DDINC, DMF)
    DER = jnp.where(scale_dd, DER * DDINC, DER)
    DDR = jnp.where(scale_dd, DDR * DDINC, DDR)
    TDER = jnp.where(no_dd, TDER, TDER_dd)
    # zero out arrays below LDB and above LFS (downdraft branch)
    below_ldb = (~no_dd) & lev & (idx >= 1) & (idx <= LDB - 1) & (LDB > 1)
    above_lfs = (~no_dd) & lev & (idx >= LFS + 1) & (idx <= KX)
    DMF = jnp.where(below_ldb | above_lfs, 0.0, DMF)
    DER = jnp.where(below_ldb | above_lfs, 0.0, DER)
    DDR = jnp.where(below_ldb | above_lfs, 0.0, DDR)
    THTAD = jnp.where(below_ldb | above_lfs, 0.0, THTAD)
    QD = jnp.where(below_ldb | above_lfs, 0.0, QD)
    TZ = jnp.where(below_ldb | above_lfs, 0.0, TZ)
    mid_zero = (~no_dd) & lev & (idx >= LDT + 1) & (idx <= LFS - 1)
    TZ = jnp.where(mid_zero, 0.0, TZ)
    QD = jnp.where(mid_zero, 0.0, QD)
    THTAD = jnp.where(mid_zero, 0.0, THTAD)
    CPR = TRPPT
    PPTFLX = jnp.where(no_dd, TRPPT, TRPPT - TDER)

    # ---- AINC limits ----
    LMAX = jnp.maximum(KLCL, LFS)
    aincm_mask = lev & (idx >= LC) & (idx <= LMAX) & ((UER - DER) > 1.0e-3)
    aincm1 = jnp.where(aincm_mask, EMS / jnp.where((UER - DER) * TIMEC != 0, (UER - DER) * TIMEC, 1.0), 1000.0)
    AINCMX = jnp.minimum(1000.0, jnp.min(jnp.where(aincm_mask, aincm1, 1000.0)))
    AINC = jnp.where(AINCMX < 1.0, AINCMX, 1.0)

    # save unit-updraft/downdraft
    DETLQ2 = DETLQ; DETIC2 = DETIC; UDR2 = UDR; UER2 = UER
    DDR2 = DDR; DER2 = DER; UMF2 = UMF; DMF2 = DMF
    TDER2 = TDER; PPTFL2 = PPTFLX

    # shallow: explicit AINC (no iteration)
    TKEMAX = 5.0
    EVAC = 0.5 * TKEMAX * 0.1
    AINC_sh = EVAC * DPTHMX * DXSQ / (VMFLCL * G * TIMEC)
    AINC = jnp.where(shallow, AINC_sh, AINC)

    out = _kf_closure_iter(
        UMF, UER, UDR, DMF, DER, DDR, DETLQ, DETIC, PPTLIQ, PPTICE,
        UMF2, UER2, UDR2, DMF2, DER2, DDR2, DETLQ2, DETIC2, TDER2, PPTFL2,
        AINC, AINCMX, EMS, EMSD, DP, OMG_init(N),
        THTA0, THTAU, THTAD, QDT, QD, Q0, P0p, Z0, DZA, DXSQ, TIMEC, TADVEC, dt,
        ABE, PMIX, TMIX, QMIX, ZMIX, DPTHMX, LC, KPBL, KLCL, LET, LTOP, K, KL, KX,
        lev, idx, TV0, TVG_zero(N), DDILFRC, QLIQ, QICE, EQFRC, LFS, DMFFRC,
        shallow, CPR, PPTFLX, TRPPT, T0p, TG_init(T0p, KX, N), QES, ML, LCL,
        aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs, NIC)
    return out


def OMG_init(N):
    return jnp.zeros(N)


def TVG_zero(N):
    return jnp.zeros(N)


def TG_init(T0p, KX, N):
    return jnp.zeros(N)


def _kf_closure_iter(UMF, UER, UDR, DMF, DER, DDR, DETLQ, DETIC, PPTLIQ, PPTICE,
                     UMF2, UER2, UDR2, DMF2, DER2, DDR2, DETLQ2, DETIC2,
                     TDER2, PPTFL2, AINC, AINCMX,
                     EMS, EMSD, DP, OMG, THTA0, THTAU, THTAD, QDT, QD, Q0, P0p, Z0,
                     DZA, DXSQ, TIMEC, TADVEC, dt, ABE, PMIX, TMIX0, QMIX0, ZMIX,
                     DPTHMX, LC, KPBL, KLCL0, LET, LTOP, K0, KL, KX, lev, idx, TV0,
                     TVG_unused, DDILFRC, QLIQ, QICE, EQFRC, LFS, DMFFRC, shallow,
                     CPR, PPTFLX, TRPPT, T0p, TG_unused, QES, ML, LCL,
                     aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs, NIC):
    """Mass-flux closure iteration (<=10), advection, CAPE-removal + feedback.
    Faithful to the validated reference; fully masked/lax. Returns output dict."""
    N = idx.shape[0]
    inLT = lev & (idx <= LTOP)

    # scale by initial AINC (shallow has explicit AINC; deep starts at AINC=1 or AINCMX)
    def scale_all(ainc, UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR):
        UMF = jnp.where(inLT, UMF2 * ainc, UMF)
        DMF = jnp.where(inLT, DMF2 * ainc, DMF)
        DETLQ = jnp.where(inLT, DETLQ2 * ainc, DETLQ)
        DETIC = jnp.where(inLT, DETIC2 * ainc, DETIC)
        UDR = jnp.where(inLT, UDR2 * ainc, UDR)
        UER = jnp.where(inLT, UER2 * ainc, UER)
        DER = jnp.where(inLT, DER2 * ainc, DER)
        DDR = jnp.where(inLT, DDR2 * ainc, DDR)
        return UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR
    # shallow scales once up front
    UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR = jax.lax.cond(
        shallow,
        lambda a: scale_all(AINC, *a),
        lambda a: a,
        (UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR))
    TDER = jnp.where(shallow, TDER2 * AINC, TDER2)
    PPTFLX = jnp.where(shallow, PPTFL2 * AINC, PPTFLX)

    carry = dict(UMF=UMF, UER=UER, UDR=UDR, DMF=DMF, DER=DER, DDR=DDR, DETLQ=DETLQ,
                 DETIC=DETIC, AINC=AINC, AINCOLD=AINC, FABEOLD=1.0, NOITR=0,
                 THTAG=jnp.zeros(N), QG=jnp.zeros(N), TG=jnp.zeros(N), TVG=jnp.zeros(N),
                 OMG=jnp.zeros(N), DOMGDP=jnp.zeros(N), FXM=jnp.zeros(N),
                 NSTEP=jnp.int32(1), DTIME=TIMEC, done=jnp.array(False),
                 KLCL=KLCL0, K=K0, PPTFLX=PPTFLX, TDER=TDER, FABE=1.0)

    def adv_and_cape(c):
        UMF = c["UMF"]; UER = c["UER"]; UDR = c["UDR"]; DMF = c["DMF"]
        DER = c["DER"]; DDR = c["DDR"]
        # OMG / DTT
        DOMGDP = jnp.where(inLT, -(UER - DER - UDR - DDR) * EMSD, 0.0)
        # OMG[nk]=OMG[nk-1]-DP[nk-1]*DOMGDP[nk-1]  (cumulative)
        contrib = jnp.where(inLT, DP * DOMGDP, 0.0)
        OMG = -jnp.concatenate([jnp.zeros(1), jnp.cumsum(contrib)[:-1]])
        OMG = jnp.where(idx <= LTOP, OMG, 0.0)
        OMG = OMG.at[0].set(0.0)
        absomgtc = jnp.abs(OMG) * TIMEC
        frdp = 0.75 * jnp.concatenate([jnp.zeros(1), DP[:-1]])
        dtt1 = jnp.where((absomgtc > frdp) & (idx >= 2) & (idx <= LTOP),
                         frdp / jnp.where(jnp.abs(OMG) > 0, jnp.abs(OMG), 1.0), TIMEC)
        DTT = jnp.minimum(TIMEC, jnp.min(jnp.where((idx >= 2) & (idx <= LTOP), dtt1, TIMEC)))
        NSTEP = jnp.round(TIMEC / DTT + 1).astype(jnp.int32)
        NSTEP = jnp.clip(NSTEP, 1, _MAX_NSTEP)
        DTIME = TIMEC / NSTEP.astype(jnp.float64)
        FXM = jnp.where(inLT, OMG * DXSQ / G, 0.0)

        THPA0 = jnp.where(inLT, THTA0, 0.0)
        QPA0 = jnp.where(inLT, Q0, 0.0)

        def substep(s, sc):
            do = s < NSTEP
            THPA, QPA = sc
            # fluxes
            omg_le = OMG <= 0.0
            THFXIN = jnp.where((idx >= 2) & inLT & omg_le, -FXM * jnp.roll(THPA, 1), 0.0)
            QFXIN = jnp.where((idx >= 2) & inLT & omg_le, -FXM * jnp.roll(QPA, 1), 0.0)
            THFXOUT_up = jnp.where((idx >= 2) & inLT & (~omg_le), FXM * THPA, 0.0)
            QFXOUT_up = jnp.where((idx >= 2) & inLT & (~omg_le), FXM * QPA, 0.0)
            # accumulate into nk-1
            THFXOUT = jnp.where(omg_le, 0.0, THFXOUT_up) + jnp.roll(THFXIN, -1)
            QFXOUT = jnp.where(omg_le, 0.0, QFXOUT_up) + jnp.roll(QFXIN, -1)
            THFXIN2 = THFXIN + jnp.roll(THFXOUT_up, -1)
            QFXIN2 = QFXIN + jnp.roll(QFXOUT_up, -1)
            THFXIN_f = jnp.where(inLT, THFXIN2, 0.0)
            QFXIN_f = jnp.where(inLT, QFXIN2, 0.0)
            THFXOUT_f = jnp.where(inLT, THFXOUT, 0.0)
            QFXOUT_f = jnp.where(inLT, QFXOUT, 0.0)
            dTH = (THFXIN_f + UDR * THTAU + DDR * THTAD - THFXOUT_f - (UER - DER) * THTA0) * DTIME * EMSD
            dQ = (QFXIN_f + UDR * QDT + DDR * QD - QFXOUT_f - (UER - DER) * Q0) * DTIME * EMSD
            THPA_n = jnp.where(do & inLT, THPA + dTH, THPA)
            QPA_n = jnp.where(do & inLT, QPA + dQ, QPA)
            return (THPA_n, QPA_n)
        THPA, QPA = jax.lax.fori_loop(0, _MAX_NSTEP, substep, (THPA0, QPA0))
        THTAG = jnp.where(inLT, THPA, 0.0)
        QG = jnp.where(inLT, QPA, 0.0)
        KLCL_borrow = c["KLCL"]

        def borrow_body(nk, qg):
            needs_borrow = (nk <= LTOP) & (qg[nk] < 0.0)

            def borrow(qg_in):
                def surface_fatal(qg_surface):
                    return qg_surface.at[nk].set(jnp.nan)

                def adjacent_borrow(qg_adj):
                    nk1 = jnp.where(nk == LTOP, KLCL_borrow, nk + 1)
                    tma = qg_adj[nk1] * EMS[nk1]
                    tmb = qg_adj[nk - 1] * EMS[nk - 1]
                    tmm = (qg_adj[nk] - 1.0e-9) * EMS[nk]
                    bcoeff = -tmm / ((tma * tma) / tmb + tmb)
                    acoeff = bcoeff * tma / tmb
                    tmb = tmb * (1.0 - bcoeff)
                    tma = tma * (1.0 - acoeff)
                    qg_adj = qg_adj.at[nk].set(1.0e-9)
                    qg_adj = qg_adj.at[nk1].set(tma * EMSD[nk1])
                    qg_adj = qg_adj.at[nk - 1].set(tmb * EMSD[nk - 1])
                    return qg_adj

                return jax.lax.cond(nk == 1, surface_fatal, adjacent_borrow, qg_in)

            return jax.lax.cond(needs_borrow, borrow, lambda qg_in: qg_in, qg)

        QG = jax.lax.fori_loop(1, KX + 1, borrow_body, QG)
        EXN = (_P00 / P0p) ** (0.2854 * (1.0 - 0.28 * QG))
        TG = jnp.where(inLT, THTAG / jnp.where(EXN != 0, EXN, 1.0), 0.0)
        TVG = jnp.where(inLT, TG * (1.0 + 0.608 * QG), 0.0)
        return DOMGDP, OMG, FXM, NSTEP, DTIME, THTAG, QG, TG, TVG

    def body(c):
        DOMGDP, OMG, FXM, NSTEP, DTIME, THTAG, QG, TG, TVG = adv_and_cape(c)
        # new mixed-layer / CAPE
        msk = lev & (idx >= LC) & (idx <= KPBL)
        TMIX = jnp.sum(jnp.where(msk, DP * TG, 0.0)) / DPTHMX
        QMIX = jnp.sum(jnp.where(msk, DP * QG, 0.0)) / DPTHMX
        es = aliq * jnp.exp((TMIX * bliq - cliq) / (TMIX - dliq))
        QSS = 0.622 * es / (PMIX - es)
        sup = QMIX > QSS
        RL = XLV0 - XLV1 * TMIX
        CPM = CP * (1.0 + 0.887 * QMIX)
        DSSDT = QSS * (cliq - bliq * dliq) / ((TMIX - dliq) * (TMIX - dliq))
        DQ = (QMIX - QSS) / (1.0 + RL * DSSDT / CPM)
        TMIX_s = TMIX + RL / CP * DQ
        QMIX_s = QMIX - DQ
        TLCL_s = TMIX_s
        # subsat path
        QMIXc = jnp.maximum(QMIX, 0.0)
        EMIX = QMIXc * PMIX / (0.622 + QMIXc)
        a1 = EMIX / aliq
        tp = (a1 - 1.0e-3) / 0.075
        indlu = jnp.clip(jnp.floor(tp).astype(jnp.int32), 0, 198)
        value = indlu * 0.075 + 1.0e-3
        aintrp = (a1 - value) / 0.075
        tlog = aintrp * alu[indlu + 1] + (1.0 - aintrp) * alu[indlu]
        TDPT = (cliq - dliq * tlog) / (bliq - tlog)
        TLCL_u = TDPT - (0.212 + 1.571e-3 * (TDPT - _T00) - 4.36e-4 * (TMIX - _T00)) * (TMIX - TDPT)
        TLCL_u = jnp.minimum(TLCL_u, TMIX)
        TMIX2 = jnp.where(sup, TMIX_s, TMIX)
        QMIX2 = jnp.where(sup, QMIX_s, QMIXc)
        TLCL = jnp.where(sup, TLCL_s, TLCL_u)
        TVLCL = TLCL * (1.0 + 0.608 * QMIX2)
        ZLCL = ZMIX + (TLCL - TMIX2) / GDRY
        ge = lev & (idx >= LC) & (ZLCL <= Z0)
        KLCL = jnp.clip(jnp.min(jnp.where(ge, idx, KX + 99)), LC, KL)
        Kk = KLCL - 1
        DLP = (ZLCL - Z0[Kk]) / (Z0[KLCL] - Z0[Kk])
        TENV = TG[Kk] + (TG[KLCL] - TG[Kk]) * DLP
        QENV = QG[Kk] + (QG[KLCL] - QG[Kk]) * DLP
        TVEN = TENV * (1.0 + 0.608 * QENV)
        THETEU_K = TMIX2 * (1.0e5 / PMIX) ** (0.2854 * (1.0 - 0.28 * QMIX2)) * \
            jnp.exp((3374.6525 / TLCL - 2.5403) * QMIX2 * (1.0 + 0.81 * QMIX2))

        # ABEG via explicit scan storing TVQU per level
        def abeg_scan(carry2, nk):
            ABEG, theteu_prev, tvqu_prev, tvg_prev = carry2
            do = (nk >= Kk) & (nk <= LTOP - 1)
            nk1 = nk + 1
            theteu_nk1 = theteu_prev
            tgu, qgu = tpmix2dd(P0p[nk1], theteu_nk1)
            tvqu = tgu * (1.0 + 0.608 * qgu - QLIQ[nk1] - QICE[nk1])
            atK = nk == Kk
            dzz = jnp.where(atK, Z0[KLCL] - ZLCL, DZA[nk])
            dilbe = jnp.where(atK,
                              ((TVLCL + tvqu) / (TVEN + TVG[nk1]) - 1.0) * dzz,
                              ((tvqu_prev + tvqu) / (TVG[nk] + TVG[nk1]) - 1.0) * dzz)
            ABEG = jnp.where(do & (dilbe > 0.0), ABEG + dilbe * G, ABEG)
            thtee = envirtht(P0p[nk1], TG[nk1], QG[nk1], aliq, bliq, cliq, dliq)
            theteu_next = jnp.where(do, theteu_nk1 * DDILFRC[nk1] + thtee * (1.0 - DDILFRC[nk1]), theteu_nk1)
            return (ABEG, theteu_next, tvqu, TVG[nk1]), None
        (ABEG, _tu, _tq, _tg), _ = jax.lax.scan(
            abeg_scan, (0.0, THETEU_K, 0.0, 0.0), jnp.arange(N))

        DABE = jnp.maximum(ABE - ABEG, 0.1 * ABE)
        FABE = ABEG / jnp.where(ABE != 0, ABE, 1.0)
        STAB = 0.95
        AINC = c["AINC"]; AINCOLD = c["AINCOLD"]; FABEOLD = c["FABEOLD"]; NOITR = c["NOITR"]
        # convergence logic
        fabe_gt1 = (FABE > 1.0)  # deep -> no convection (handled by caller via tiny tendencies)
        converged = ((FABE <= 1.05 - STAB) & (FABE >= 0.95 - STAB))
        nochange = jnp.abs(AINC - AINCOLD) < 0.0001
        # next AINC
        AINC_next = jnp.where(FABE == 0.0, AINC * 0.5,
                    jnp.where(DABE < 1.0e-4, AINCOLD, AINC * STAB * ABE / jnp.where(DABE != 0, DABE, 1.0)))
        AINC_next = jnp.minimum(AINCMX, AINC_next)
        too_small = AINC_next < 0.05
        # update scaled fields with AINC_next (for next iter)
        TDERn = TDER2 * AINC_next
        PPTFLXn = PPTFL2 * AINC_next
        UMFn = jnp.where(inLT, UMF2 * AINC_next, c["UMF"])
        DMFn = jnp.where(inLT, DMF2 * AINC_next, c["DMF"])
        DETLQn = jnp.where(inLT, DETLQ2 * AINC_next, c["DETLQ"])
        DETICn = jnp.where(inLT, DETIC2 * AINC_next, c["DETIC"])
        UDRn = jnp.where(inLT, UDR2 * AINC_next, c["UDR"])
        UERn = jnp.where(inLT, UER2 * AINC_next, c["UER"])
        DERn = jnp.where(inLT, DER2 * AINC_next, c["DER"])
        DDRn = jnp.where(inLT, DDR2 * AINC_next, c["DDR"])

        stop_iter = converged | too_small | fabe_gt1 | shallow
        c2 = dict(c)
        c2.update(UMF=jnp.where(stop_iter, c["UMF"], UMFn),
                  UER=jnp.where(stop_iter, c["UER"], UERn),
                  UDR=jnp.where(stop_iter, c["UDR"], UDRn),
                  DMF=jnp.where(stop_iter, c["DMF"], DMFn),
                  DER=jnp.where(stop_iter, c["DER"], DERn),
                  DDR=jnp.where(stop_iter, c["DDR"], DDRn),
                  DETLQ=jnp.where(stop_iter, c["DETLQ"], DETLQn),
                  DETIC=jnp.where(stop_iter, c["DETIC"], DETICn),
                  AINC=jnp.where(stop_iter, AINC, AINC_next), AINCOLD=AINC,
                  FABEOLD=FABE, THTAG=THTAG, QG=QG, TG=TG, TVG=TVG, OMG=OMG,
                  DOMGDP=DOMGDP, FXM=FXM, NSTEP=NSTEP, DTIME=DTIME, KLCL=KLCL,
                  K=Kk, PPTFLX=jnp.where(stop_iter, c["PPTFLX"], PPTFLXn),
                  TDER=jnp.where(stop_iter, c["TDER"], TDERn), FABE=FABE,
                  done=stop_iter)
        # On stop, KEEP the current THTAG/QG/TG (these are the final advected fields)
        return c2

    def cond(c):
        return ~c["done"]

    # bounded iteration: at most _MAX_CLOSURE_ITERS; shallow exits after first adv
    def iter_body(i, c):
        return jax.lax.cond(c["done"], lambda x: x, body, c)
    carry = jax.lax.fori_loop(0, _MAX_CLOSURE_ITERS, iter_body, carry)
    # if shallow: we still need ONE advection pass (body did it). For shallow, body
    # ran once and set done.
    THTAG = carry["THTAG"]; QG = carry["QG"]; TG = carry["TG"]
    OMG = carry["OMG"]; FXM = carry["FXM"]; NSTEP = carry["NSTEP"]; DTIME = carry["DTIME"]
    AINC = carry["AINC"]; PPTFLX = carry["PPTFLX"]
    UMF = carry["UMF"]; UDR = carry["UDR"]; DDR = carry["DDR"]; DETLQ = carry["DETLQ"]; DETIC = carry["DETIC"]

    # ---- hydrometeor advection ----
    fbfrc = jnp.where(shallow, jnp.float64(1.0), jnp.float64(_FBFRC))
    FRC2 = jnp.where(CPR > 0.0, PPTFLX / jnp.where(CPR * AINC != 0, CPR * AINC, 1.0), 0.0)
    RAINFB = jnp.where(inLT, PPTLIQ * AINC * fbfrc * FRC2, 0.0)
    SNOWFB = jnp.where(inLT, PPTICE * AINC * fbfrc * FRC2, 0.0)
    QLPA = jnp.zeros(N); QIPA = jnp.zeros(N); QRPA = jnp.zeros(N); QSPA = jnp.zeros(N)

    def hyd_substep(s, sc):
        do = s < NSTEP
        QLPA, QIPA, QRPA, QSPA = sc
        omg_le = OMG <= 0.0
        def adv(QPA):
            FXIN = jnp.where((idx >= 2) & inLT & omg_le, -FXM * jnp.roll(QPA, 1), 0.0)
            FXOUT_up = jnp.where((idx >= 2) & inLT & (~omg_le), FXM * QPA, 0.0)
            FXOUT = jnp.where(inLT, jnp.where(omg_le, 0.0, FXOUT_up) + jnp.roll(FXIN, -1), 0.0)
            FXIN_f = jnp.where(inLT, FXIN + jnp.roll(FXOUT_up, -1), 0.0)
            return FXIN_f, FXOUT
        QLFXIN, QLFXOUT = adv(QLPA); QIFXIN, QIFXOUT = adv(QIPA)
        QRFXIN, QRFXOUT = adv(QRPA); QSFXIN, QSFXOUT = adv(QSPA)
        QLn = jnp.where(do & inLT, QLPA + (QLFXIN + DETLQ - QLFXOUT) * DTIME * EMSD, QLPA)
        QIn = jnp.where(do & inLT, QIPA + (QIFXIN + DETIC - QIFXOUT) * DTIME * EMSD, QIPA)
        QRn = jnp.where(do & inLT, QRPA + (QRFXIN - QRFXOUT + RAINFB) * DTIME * EMSD, QRPA)
        QSn = jnp.where(do & inLT, QSPA + (QSFXIN - QSFXOUT + SNOWFB) * DTIME * EMSD, QSPA)
        return (QLn, QIn, QRn, QSn)
    QLG, QIG, QRG, QSG = jax.lax.fori_loop(0, _MAX_NSTEP, hyd_substep, (QLPA, QIPA, QRPA, QSPA))

    PRATEC = PPTFLX * (1.0 - fbfrc) / DXSQ
    RAINCV = dt * PRATEC

    # feedback tendencies (mixed-phase: f_qi=f_qs=True)
    QL0 = jnp.zeros(N); QI0 = jnp.zeros(N); QR0 = jnp.zeros(N); QS0 = jnp.zeros(N)
    tend_time = jnp.where(shallow, jnp.float64(2400.0), TIMEC)
    DQCDT = jnp.where(inLT, (QLG - QL0) / tend_time, 0.0)
    DQSDT = jnp.where(inLT, (QSG - QS0) / tend_time, 0.0)
    DQRDT = jnp.where(inLT, (QRG - QR0) / tend_time, 0.0)
    DQIDT = jnp.where(inLT, (QIG - QI0) / tend_time, 0.0)
    DTDT = jnp.where(inLT, (TG - T0p) / tend_time, 0.0)
    DQDT = jnp.where(inLT, (QG - Q0) / tend_time, 0.0)

    # NCA
    NICf = jnp.where(TADVEC < TIMEC, jnp.round(TADVEC / dt).astype(jnp.int32), NIC)
    NCA = jnp.where(shallow, jnp.float64(0.0), NICf.astype(jnp.float64) * dt)
    TIMEC_out = jnp.where(shallow, jnp.float64(2400.0), TIMEC)

    pii = (P0p / 1.0e5) ** (R_D / CP)
    def slice0(a):
        return a[1:KX + 1]
    out = dict(
        DTDT=slice0(DTDT), DQDT=slice0(DQDT), DQCDT=slice0(DQCDT), DQRDT=slice0(DQRDT),
        DQIDT=slice0(DQIDT), DQSDT=slice0(DQSDT),
        RTHCUTEN=slice0(DTDT / pii), RQVCUTEN=slice0(DQDT),
        RQCCUTEN=slice0(DQCDT), RQRCUTEN=slice0(DQRDT),
        RQICUTEN=slice0(DQIDT), RQSCUTEN=slice0(DQSDT),
        RAINCV=RAINCV, PRATEC=PRATEC, NCA=NCA,
        CUTOP=LTOP.astype(jnp.float64), CUBOT=LCL.astype(jnp.float64),
        ISHALL=jnp.where(shallow, jnp.int32(1), jnp.int32(0)), TIMEC=TIMEC_out)
    return out
