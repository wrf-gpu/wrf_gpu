"""WRF REAL KF column prototype: one Pallas program per column.

Equations transcribed from physics/cumulus_kf.py at 86cf9eb0e, with
closure decisions and hydrometeor feedback matched to pristine WRF source. Numerical
profile gathers/scatters use register reductions/selects; shared WRF REAL
lookup tables use device loads. Candidate and closure loops stop per column.
The independent frozen WRF savepoints govern this default-off experiment.
"""
from __future__ import annotations
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.extend import core
from jax.interpreters import mlir
from jax._src.pallas.triton import lowering as tl
from jax._src.lib.triton import dialect as tt
from gpuwrf.physics import cumulus_kf_tables as tables

# Triton's native register gather; Pallas currently has no public wrapper.
_gather_p = core.Primitive("kf_register_gather")
_gather_p.def_impl(lambda a, ix: a[ix])
_gather_p.def_abstract_eval(lambda a, ix: a.update(shape=ix.shape))
mlir.register_lowering(_gather_p, mlir.lower_fun(lambda a, ix: a[ix], multiple_results=False))
@tl.register_lowering(_gather_p)
def _gather_lower(ctx, a, ix):
    return tt.gather(a, ix, 0, efficient_layout=False)

def _where(pred, on_true, on_false):
    # Triton select_n treats weak literals as the predicate's dtype. Make
    # branches strong without changing their WRF REAL/INTEGER precision.
    dtype = jnp.result_type(on_true, on_false)
    return jnp.where(pred, jnp.asarray(on_true, dtype), jnp.asarray(on_false, dtype))


def _nint_positive(x):
    # All KF timestep counts are positive; WRF NINT rounds half upward.
    # Pallas/Triton has floor but has no lowering for JAX round.
    return jnp.floor(x + 0.5)


def _closure_control(ncount, shallow, noitr, ainc, aincold, fab_old,
                     fab_current, abe, dabe, aincmx):
    """WRF module_cu_kfeta.F:2179-2259, in source decision order.

    A retry rolls back AINC but deliberately retains the CURRENT fluxes;
    WRF CYCLE does not run the rescaling block. The next pass exits via
    NOITR after computing its advected fields. Iteration ten never rescales.
    """
    stab = jnp.float32(0.95)
    upper = jnp.float32(1.05) - stab
    lower = jnp.float32(0.95) - stab
    bypass = shallow | (noitr == 1)
    reject_cape = (~bypass) & (fab_current > 1.0)
    delta = ainc - aincold
    nochange = jnp.abs(delta) < 0.0001
    dfda = (fab_current - fab_old) / _where(delta != 0, delta, 1.0)
    retry_early = (~bypass) & (~reject_cape) & (ncount != 1) & (nochange | (dfda > 0.0))
    history = (~bypass) & (~reject_cape) & (~retry_early)
    at_limit = (ainc / aincmx > 0.999) & (fab_current > upper)
    converged = (fab_current <= upper) & (fab_current >= lower)
    exit_history = history & (at_limit | converged | (ncount == 10))
    retry_late = history & (~exit_history) & (fab_current != 0.0) & (dabe < 1e-4)
    may_scale = history & (~exit_history) & (~retry_late)
    next_ainc = _where(fab_current == 0.0, ainc * 0.5,
                       ainc * stab * abe / _where(dabe != 0, dabe, 1.0))
    next_ainc = jnp.minimum(aincmx, next_ainc)
    reject_small = may_scale & (next_ainc < 0.05)
    rescale = may_scale & (~reject_small)
    return dict(done=bypass | reject_cape | reject_small | exit_history,
                rejected=reject_cape | reject_small, rescale=rescale,
                AINC=_where(retry_early, aincold, _where(may_scale, next_ainc, ainc)),
                AINCOLD=_where(history, ainc, aincold),
                FABEOLD=_where(history, fab_current, fab_old),
                NOITR=_where(retry_early | retry_late, jnp.int32(1), noitr))


def _roll(a, shift):
    return _gather_p.bind(a, (jnp.arange(a.shape[0], dtype=jnp.int32) - shift) % a.shape[0])

def _get(a, index):
    if isinstance(a, (tuple, list, dict)):
        return a[index]
    if not isinstance(jax.typeof(a), jax._src.state.types.AbstractRef):
        index = _where(index < 0, index + a.shape[0], index)
        index = jnp.clip(index, 0, a.shape[0] - 1)
        return jnp.sum(_where(jnp.arange(a.shape[0], dtype=jnp.int32) == index, a, 0), dtype=a.dtype)
    return a[index]

def _set(a, index, value):
    index = _where(index < 0, index + a.shape[0], index)
    return _where(jnp.arange(a.shape[0]) == index, value, a)

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
_FBFRC = 0.0
_MAX_CLOSURE_ITERS = 10
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

def _column(T0, QV0, P0, DZQ, RHOE, W0A, U0, V0, dt, dx, KX, the0_ref, ttab_ref, qstab_ref, alu_ref, warm_rain, f_qi, f_qs, zero_runtime):
    ALIQ, GDRY = _runtime_constants()
    def _table_interp(p, thes):
        tp = (p - tables.PLUTOP) * tables.RDPR
        qq = tp - jnp.floor(tp)
        iptb = jnp.clip(jnp.floor(tp).astype(jnp.int32), 0, tables.KFNP - 2)
        bth = (the0_ref[iptb + 1] - the0_ref[iptb]) * qq + the0_ref[iptb]
        tth = (thes - bth) * tables.RDTHK
        pp = tth - jnp.floor(tth)
        ithtb = jnp.clip(jnp.floor(tth).astype(jnp.int32), 0, tables.KFNT - 2)
        def interp(ref):
            t00, t10 = ref[ithtb, iptb], ref[ithtb + 1, iptb]
            t01, t11 = ref[ithtb, iptb + 1], ref[ithtb + 1, iptb + 1]
            return t00 + (t10 - t00) * pp + (t01 - t00) * qq + (t00 - t10 - t01 + t11) * pp * qq
        return interp(ttab_ref), interp(qstab_ref)


    def _safe(x, eps=1e-30):
        return _where(jnp.abs(x) < eps, jnp.sign(x) * eps + (x == 0) * eps, x)

    def tpmix2(p, thes, tu, qu, qliq, qice, xlv1, xlv0):
        """Faithful TPMIX2: lookup parcel T/qs at (thes,p), then saturation
        adjustment. Returns (tu_out, qu_out, qliq_out, qice_out, qnewlq, qnewic)."""
        temp, qs = _table_interp(p, thes)
        dq = qs - qu
        qnew_le = qu - qs
        qu_le = qs
        qliq_le = qliq
        qice_le = qice
        temp_le = temp
        qtot = qliq + qice
        rll = xlv0 - xlv1 * temp
        cpp = 1004.5 * (1.0 + 0.89 * qu)
        qliq_a = qliq - dq * qliq / (qtot + 1e-10)
        qice_a = qice - dq * qice / (qtot + 1e-10)
        qu_a = qs
        temp_a = temp
        temp_b1 = temp + rll * (dq / (1.0 + dq)) / cpp
        qu_b1 = qu
        qliq_b1 = qliq
        qice_b1 = qice
        temp_b2 = temp + rll * ((dq - qtot) / (1.0 + dq - qtot)) / cpp
        qu_b2 = qu + qtot
        qliq_b2 = 0.0 * qliq
        qice_b2 = 0.0 * qice
        cond_qtot_small = qtot < 1e-10
        temp_b = _where(cond_qtot_small, temp_b1, temp_b2)
        qu_b = _where(cond_qtot_small, qu_b1, qu_b2)
        qliq_b = _where(cond_qtot_small, qliq_b1, qliq_b2)
        qice_b = _where(cond_qtot_small, qice_b1, qice_b2)
        cond_qtot_ge = qtot >= dq
        temp_gt = _where(cond_qtot_ge, temp_a, temp_b)
        qu_gt = _where(cond_qtot_ge, qu_a, qu_b)
        qliq_gt = _where(cond_qtot_ge, qliq_a, qliq_b)
        qice_gt = _where(cond_qtot_ge, qice_a, qice_b)
        qnew_gt = jnp.zeros_like(qnew_le)
        le = dq <= 0.0
        temp_out = _where(le, temp_le, temp_gt)
        qu_out = _where(le, qu_le, qu_gt)
        qliq_out = _where(le, qliq_le, qliq_gt)
        qice_out = _where(le, qice_le, qice_gt)
        qnew = _where(le, qnew_le, qnew_gt)
        return (temp_out, qu_out, qliq_out, qice_out, qnew, jnp.zeros_like(qnew))

    def tpmix2dd(p, thes):
        """Faithful TPMIX2DD: just the bilinear lookup (no saturation adjustment)."""
        ts, qs = _table_interp(p, thes)
        return (ts, qs)

    def envirtht(p1, t1, q1, aliq, bliq, cliq, dliq):
        """Faithful ENVIRTHT: environmental equivalent potential temperature."""
        astrt = 0.001
        ainc = 0.075
        alu = alu_ref
        c1 = 3374.6525
        c2 = 2.5403
        t00 = 273.16
        p00 = 100000.0
        ee = q1 * p1 / (0.622 + q1)
        a1 = ee / aliq
        tp = (a1 - astrt) / ainc
        indlu = jnp.floor(tp).astype(jnp.int32)
        indlu = jnp.clip(indlu, 0, 198)
        value = indlu * ainc + astrt
        aintrp = (a1 - value) / ainc
        tlog = aintrp * _get(alu, indlu + 1) + (1.0 - aintrp) * _get(alu, indlu)
        tdpt = (cliq - dliq * tlog) / (bliq - tlog)
        tsat = tdpt - (0.212 + 0.001571 * (tdpt - t00) - 0.000436 * (t1 - t00)) * (t1 - tdpt)
        tht = t1 * (p00 / p1) ** (0.2854 * (1.0 - 0.28 * q1))
        tht1 = tht * jnp.exp((c1 / tsat - c2) * q1 * (1.0 + 0.81 * q1))
        return tht1

    def dtfrznew(tu, p, thteu, qu, qfrz, qice, aliq, bliq, cliq, dliq):
        """Faithful DTFRZNEW: freezing of liquid in updraft."""
        rlc = 2500000.0 - 2369.276 * (tu - 273.16)
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
        pii = (100000.0 / p) ** (0.2854 * (1.0 - 0.28 * qu))
        thteu = tu * pii * jnp.exp((3374.6525 / tu - 2.5403) * qu * (1.0 + 0.81 * qu))
        return (tu, thteu, qu, qice)

    def condload(qliq, qice, wtw, dz, boterm, enterm, rate, qnewlq, qnewic, g):
        """Faithful CONDLOAD: precipitation fallout (Ogura & Cho 1973)."""
        qtot = qliq + qice
        qnew = qnewlq + qnewic
        qest = 0.5 * (qtot + qnew)
        g1 = wtw + boterm - enterm - 2.0 * g * dz * qest / 1.5
        g1 = _where(g1 < 0.0, 0.0, g1)
        wavg = 0.5 * (jnp.sqrt(jnp.maximum(wtw, 0.0)) + jnp.sqrt(g1))
        conv = rate * dz / _safe(wavg)
        ratio3 = qnewlq / (qnew + 1e-08)
        qtot = qtot + 0.6 * qnew
        oldq = qtot
        ratio4 = (0.6 * qnewlq + qliq) / (qtot + 1e-08)
        qtot = qtot * jnp.exp(-conv)
        dq = oldq - qtot
        qlqout = ratio4 * dq
        qicout = (1.0 - ratio4) * dq
        pptdrg = 0.5 * (oldq + qtot - 0.2 * qnew)
        wtw = wtw + boterm - enterm - 2.0 * g * dz * pptdrg / 1.5
        wtw = _where(jnp.abs(wtw) < 0.0001, 0.0001, wtw)
        qliq = ratio4 * qtot + ratio3 * 0.4 * qnew
        qice = (1.0 - ratio4) * qtot + (1.0 - ratio3) * 0.4 * qnew
        return (qliq, qice, wtw, qlqout, qicout, 0.0 * qnewlq, 0.0 * qnewic)

    def prof5(eq):
        """Faithful PROF5: Gaussian-mixing entrainment/detrainment fractions.
        Returns (ee, ud)."""
        sqrt2p = 2.506628
        a1 = 0.4361836
        a2 = -0.1201676
        a3 = 0.937298
        p = 0.33267
        sigma = 0.166666667
        fe = 0.202765151
        y = 6.0 * eq - 3.0
        ey = jnp.exp(y * y / -2.0)
        e45 = jnp.exp(-4.5)
        t2 = 1.0 / (1.0 + p * jnp.abs(y))
        t1 = 0.500498
        c1 = a1 * t1 + a2 * t1 * t1 + a3 * t1 * t1 * t1
        c2 = a1 * t2 + a2 * t2 * t2 + a3 * t2 * t2 * t2
        ee_pos = sigma * (0.5 * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * eq * eq / 2.0
        ud_pos = sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
        ee_neg = sigma * (0.5 * (ey * c2 - e45 * c1) + sigma * (e45 - ey)) - e45 * eq * eq / 2.0
        ud_neg = sigma * (0.5 * (sqrt2p - e45 * c1 - ey * c2) + sigma * (e45 - ey)) - e45 * (0.5 + eq * eq / 2.0 - eq)
        ee = _where(y >= 0.0, ee_pos, ee_neg) / fe
        ud = _where(y >= 0.0, ud_pos, ud_neg) / fe
        return (ee, ud)

    def _empty_col(KX, nca, physical_kx=None):
        z = jnp.zeros(KX, dtype=jnp.float32)
        return dict(DTDT=z, DQDT=z, DQCDT=z, DQRDT=z, DQIDT=z, DQSDT=z, RTHCUTEN=z, RQVCUTEN=z, RQCCUTEN=z, RQRCUTEN=z, RQICUTEN=z, RQSCUTEN=z, RAINCV=zero_runtime, PRATEC=zero_runtime, NCA=jnp.float32(nca) + zero_runtime, CUTOP=zero_runtime + 1.0, CUBOT=zero_runtime + (KX if physical_kx is None else physical_kx) + 1.0, ISHALL=zero_runtime.astype(jnp.int32) + 2, TIMEC=zero_runtime)

    def _run_updraft(NUcand, KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, W0Ap, dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu):
        """Run ONE updraft from candidate USL index NUcand (1-based into KCHECK).
        Returns a dict of outcome flags + full updraft fields. All branches are
        masked; nothing here mutates global state. Mirrors the validated reference.
        """
        N = _get(lev.shape, 0)
        KMIX = _get(KCHECK, NUcand)
        LC = KMIX
        inlayer = lev & (idx >= LC)
        dpc = jnp.cumsum(_where(inlayer, DP, 0.0))
        reached = inlayer & (dpc > _DPMIN)
        KPBL = jnp.min(_where(reached, idx, KX + 99))
        KPBL = jnp.clip(KPBL, LC, KX)
        DPTHMX = _get(dpc, KPBL)
        valid_depth = DPTHMX >= _DPMIN
        msk = lev & (idx >= LC) & (idx <= KPBL)
        DPTHMX_safe = _where(DPTHMX > 0, DPTHMX, 1.0)
        TMIX = jnp.sum(_where(msk, DP * T0p, 0.0)) / DPTHMX_safe
        QMIX = jnp.sum(_where(msk, DP * Q0, 0.0)) / DPTHMX_safe
        ZMIX = jnp.sum(_where(msk, DP * Z0, 0.0)) / DPTHMX_safe
        PMIX = jnp.sum(_where(msk, DP * P0p, 0.0)) / DPTHMX_safe
        EMIX = QMIX * PMIX / (0.622 + QMIX)
        astrt = 0.001
        ainc = 0.075
        a1 = EMIX / aliq
        tp = (a1 - astrt) / ainc
        indlu = jnp.clip(jnp.floor(tp).astype(jnp.int32), 0, 198)
        value = indlu * ainc + astrt
        aintrp = (a1 - value) / ainc
        tlog = aintrp * _get(alu, indlu + 1) + (1.0 - aintrp) * _get(alu, indlu)
        TDPT = (cliq - dliq * tlog) / (bliq - tlog)
        TLCL = TDPT - (0.212 + 0.001571 * (TDPT - _T00) - 0.000436 * (TMIX - _T00)) * (TMIX - TDPT)
        TLCL = jnp.minimum(TLCL, TMIX)
        TVLCL = TLCL * (1.0 + 0.608 * QMIX)
        ZLCL = ZMIX + (TLCL - TMIX) / GDRY
        ge = lev & (idx >= LC) & (ZLCL <= Z0)
        KLCL = jnp.clip(jnp.min(_where(ge, idx, KX + 99)), LC, KL)
        off_top = ZLCL > _get(Z0, KL)
        K = KLCL - 1
        DLP = (ZLCL - _get(Z0, K)) / (_get(Z0, KLCL) - _get(Z0, K))
        TENV = _get(T0p, K) + (_get(T0p, KLCL) - _get(T0p, K)) * DLP
        QENV = _get(Q0, K) + (_get(Q0, KLCL) - _get(Q0, K)) * DLP
        TVEN = TENV * (1.0 + 0.608 * QENV)
        WKLCL = _where(ZLCL < 2000.0, 0.02 * ZLCL / 2000.0, 0.02)
        WKL = (_get(W0Ap, K) + (_get(W0Ap, KLCL) - _get(W0Ap, K)) * DLP) * dx / 25000.0 - WKLCL
        DTLCL = _where(WKL < 0.0001, 0.0, 4.64 * jnp.abs(WKL) ** 0.33)
        DTRH = 0.0
        buoyant = (TLCL + DTLCL + DTRH >= TENV) & valid_depth & ~off_top
        THETEU_K = envirtht(PMIX, TMIX, QMIX, aliq, bliq, cliq, dliq)
        DTTOT = DTLCL + DTRH
        GDT = 2.0 * G * DTTOT * 500.0 / TVEN
        WLCL = _where(DTTOT > 0.0001, jnp.minimum(1.0 + 0.5 * jnp.sqrt(jnp.abs(GDT)), 3.0), 1.0)
        PLCL = _get(P0p, K) + (_get(P0p, KLCL) - _get(P0p, K)) * DLP
        TVLCL = TLCL * (1.0 + 0.608 * QMIX)
        RHOLCL = PLCL / (R_D * TVLCL)
        RAD = _where(WKL < 0.0, 1000.0, _where(WKL > 0.1, 2000.0, 1000.0 + 1000.0 * WKL / 0.1))
        AU0 = 0.01 * DXSQ
        VMFLCL = RHOLCL * AU0
        Zr = jnp.zeros(N, dtype=jnp.float32)
        F = dict(UMF=Zr, UER=Zr, UDR=Zr, DETLQ=Zr, DETIC=Zr, PPTLIQ=Zr, PPTICE=Zr, QLIQ=Zr, QICE=Zr, QLQOUT=Zr, QICOUT=Zr, TU=Zr, TVU=Zr, QU=Zr, WU=Zr, THETEU=Zr, THETEE=Zr, TVQU=Zr, EQFRC=Zr, QDT=Zr, RATIO2=Zr, DILFRC=jnp.ones(N, dtype=jnp.float32))
        F['WU'] = _set(F['WU'], K, WLCL)
        F['UMF'] = _set(F['UMF'], K, VMFLCL)
        F['TU'] = _set(F['TU'], K, TLCL)
        F['TVU'] = _set(F['TVU'], K, TVLCL)
        F['QU'] = _set(F['QU'], K, QMIX)
        F['EQFRC'] = _set(F['EQFRC'], K, 1.0)
        F['THETEU'] = _set(F['THETEU'], K, THETEU_K)
        carry = dict(F=F, WTW=WLCL * WLCL, UPOLD=VMFLCL, UPNEW=VMFLCL, EE1=1.0, UD1=0.0, REI=0.0, ABE=0.0, TRPPT=0.0, LET=KLCL, LTOP=K, TTEMP=_TTFRZ, active=buoyant, IFLAG=0)

        def step(nk, c):
            F = dict(c['F'])
            NK1 = nk + 1
            in_range = (nk >= K) & (nk <= KL - 1)
            run = c['active'] & in_range
            WTW = c['WTW']
            UPOLD = c['UPOLD']
            REI = c['REI']
            EE1 = c['EE1']
            UD1 = c['UD1']
            ABE = c['ABE']
            TRPPT = c['TRPPT']
            LET = c['LET']
            TTEMP = c['TTEMP']
            IFLAG = c['IFLAG']
            F['RATIO2'] = _set(F['RATIO2'], NK1, _where(run, _get(F['RATIO2'], nk), _get(F['RATIO2'], NK1)))
            tu = _get(T0p, NK1)
            thteu = _get(F['THETEU'], nk)
            qu = _get(F['QU'], nk)
            qliq = _get(F['QLIQ'], nk)
            qice = _get(F['QICE'], nk)
            tu, qu, qliq, qice, qnewlq, qnewic = tpmix2(_get(P0p, NK1), thteu, tu, qu, qliq, qice, XLV1, XLV0)
            frz = tu <= _TTFRZ
            gtb = tu > _TBFRZ
            TTEMP_f = _where(TTEMP > _TTFRZ, _TTFRZ, TTEMP)
            FRC1 = _where(frz, _where(gtb, (TTEMP_f - tu) / (TTEMP_f - _TBFRZ), 1.0), 0.0)
            IFLAG = _where(run & frz & ~gtb, 1, IFLAG)
            TTEMP = _where(run & frz, tu, TTEMP)
            QFRZ = (qliq + qnewlq) * FRC1
            qnewic_f = qnewic + qnewlq * FRC1
            qnewlq_f = qnewlq - qnewlq * FRC1
            qice_a = qice + qliq * FRC1
            qliq_a = qliq - qliq * FRC1
            tu2, thteu2, qu2, qice2 = dtfrznew(tu, _get(P0p, NK1), thteu, qu, QFRZ, qice_a, aliq, bliq, cliq, dliq)
            tu = _where(frz, tu2, tu)
            thteu = _where(frz, thteu2, thteu)
            qu = _where(frz, qu2, qu)
            qice = _where(frz, qice2, _where(frz, qice_a, qice))
            qliq = _where(frz, qliq_a, qliq)
            qnewlq = _where(frz, qnewlq_f, qnewlq)
            qnewic = _where(frz, qnewic_f, qnewic)
            TVU1 = tu * (1.0 + 0.608 * qu)
            atK = nk == K
            BE = _where(atK, (TVLCL + TVU1) / (TVEN + _get(TV0, NK1)) - 1.0, (_get(F['TVU'], nk) + TVU1) / (_get(TV0, nk) + _get(TV0, NK1)) - 1.0)
            DZZ = _where(atK, _get(Z0, NK1) - ZLCL, _get(DZA, nk))
            BOTERM = _where(atK, 2.0 * (_get(Z0, NK1) - ZLCL) * G * BE / 1.5, 2.0 * _get(DZA, nk) * G * BE / 1.5)
            ENTERM = 2.0 * REI * WTW / _where(UPOLD != 0, UPOLD, 1.0)
            qliq, qice, WTW2, qlqout, qicout, qnewlq, qnewic = condload(qliq, qice, WTW, DZZ, BOTERM, ENTERM, _RATE, qnewlq, qnewic, G)
            wtw_break = WTW2 < 0.001
            F['QLQOUT'] = _set(F['QLQOUT'], NK1, _where(run, qlqout, _get(F['QLQOUT'], NK1)))
            F['QICOUT'] = _set(F['QICOUT'], NK1, _where(run, qicout, _get(F['QICOUT'], NK1)))
            thtee = envirtht(_get(P0p, NK1), _get(T0p, NK1), _get(Q0, NK1), aliq, bliq, cliq, dliq)
            REI2 = VMFLCL * _get(DP, NK1) * 0.03 / RAD
            TVQU1 = tu * (1.0 + 0.608 * qu - qliq - qice)
            DILBE = _where(atK, ((TVLCL + TVQU1) / (TVEN + _get(TV0, NK1)) - 1.0) * DZZ, ((_get(F['TVQU'], nk) + TVQU1) / (_get(TV0, nk) + _get(TV0, NK1)) - 1.0) * DZZ)
            ABE2 = _where(DILBE > 0.0, ABE + DILBE * G, ABE)
            colder = TVQU1 <= _get(TV0, NK1)
            thttmp = 0.95 * thtee + 0.05 * thteu
            qtmp = 0.95 * _get(Q0, NK1) + 0.05 * qu
            tl = 0.05 * qliq
            ti = 0.05 * qice
            t95, q95, l95, i95, _a, _b = tpmix2(_get(P0p, NK1), thttmp, TVQU1, qtmp, tl, ti, XLV1, XLV0)
            TU95 = t95 * (1.0 + 0.608 * q95 - l95 - i95)
            warm95 = TU95 > _get(TV0, NK1)
            thttmpb = 0.1 * thtee + 0.9 * thteu
            qtmpb = 0.1 * _get(Q0, NK1) + 0.9 * qu
            tlb = 0.9 * qliq
            tib = 0.9 * qice
            t10, q10, l10, i10, _c, _d = tpmix2(_get(P0p, NK1), thttmpb, TVQU1, qtmpb, tlb, tib, XLV1, XLV0)
            TU10 = t10 * (1.0 + 0.608 * q10 - l10 - i10)
            TVDIFF = jnp.abs(TU10 - TVQU1)
            denom = _where(jnp.abs(TU10 - TVQU1) > 0, TU10 - TVQU1, 1.0)
            eqfrc_raw = jnp.minimum(1.0, jnp.maximum(0.0, (_get(TV0, NK1) - TVQU1) * 0.1 / denom))
            ee5, ud5 = prof5(eqfrc_raw)
            EE2 = _where(colder, 0.5, _where(warm95, 1.0, _where(TVDIFF < 0.001, 1.0, _where(eqfrc_raw == 1.0, 1.0, _where(eqfrc_raw == 0.0, 0.0, ee5)))))
            UD2 = _where(colder, 1.0, _where(warm95, 0.0, _where(TVDIFF < 0.001, 0.0, _where(eqfrc_raw == 1.0, 0.0, _where(eqfrc_raw == 0.0, 1.0, ud5)))))
            EQFRC1 = _where(colder, 0.0, _where(warm95, 1.0, _where(TVDIFF < 0.001, 1.0, eqfrc_raw)))
            LET_new = _where(run & ~colder, NK1, LET)
            EE2 = jnp.maximum(EE2, 0.5)
            UD2 = 1.5 * UD2
            uer = 0.5 * REI2 * (EE1 + EE2)
            udr = 0.5 * REI2 * (UD1 + UD2)
            det_break = _get(F['UMF'], nk) - udr < 10.0
            ABE3 = _where(det_break & (DILBE > 0.0), ABE2 - DILBE * G, ABE2)
            LET_fin = _where(det_break, nk, LET_new)
            stop = wtw_break | det_break
            cont = run & ~stop
            UPOLD2 = _get(F['UMF'], nk) - udr
            UPNEW = UPOLD2 + uer
            DILF = UPNEW / _where(UPOLD2 != 0, UPOLD2, 1.0)
            qu_new = (UPOLD2 * qu + uer * _get(Q0, NK1)) / _where(UPNEW != 0, UPNEW, 1.0)
            thteu_new = (thteu * UPOLD2 + thtee * uer) / _where(UPNEW != 0, UPNEW, 1.0)
            qliq_new = qliq * UPOLD2 / _where(UPNEW != 0, UPNEW, 1.0)
            qice_new = qice * UPOLD2 / _where(UPNEW != 0, UPNEW, 1.0)
            pptl = qlqout * _get(F['UMF'], nk)
            ppti = qicout * _get(F['UMF'], nk)
            uer_pbl = _where(NK1 <= KPBL, uer + VMFLCL * _get(DP, NK1) / DPTHMX_safe, uer)

            def setc(key, val):
                F[key] = _set(F[key], NK1, _where(cont, val, _get(F[key], NK1)))
            setc('UMF', UPNEW)
            setc('DILFRC', DILF)
            setc('DETLQ', qliq * udr)
            setc('DETIC', qice * udr)
            setc('QDT', qu)
            setc('QU', qu_new)
            setc('THETEU', thteu_new)
            setc('QLIQ', qliq_new)
            setc('QICE', qice_new)
            setc('PPTLIQ', pptl)
            setc('PPTICE', ppti)
            setc('UER', uer_pbl)
            setc('UDR', udr)
            setc('TU', tu)
            setc('TVU', TVU1)
            setc('TVQU', TVQU1)
            setc('THETEE', thtee)
            setc('EQFRC', EQFRC1)
            F['WU'] = _set(F['WU'], NK1, _where(cont, jnp.sqrt(jnp.abs(WTW2)), _get(F['WU'], NK1)))
            TRPPT2 = _where(cont, TRPPT + pptl + ppti, TRPPT)
            WTWn = _where(run, WTW2, WTW)
            EE1n = _where(cont, EE2, EE1)
            UD1n = _where(cont, UD2, UD1)
            UPOLDn = _where(cont, UPOLD2, UPOLD)
            UPNEWn = _where(cont, UPNEW, c['UPNEW'])
            REIn = _where(cont, REI2, REI)
            ABEn = _where(run, _where(det_break, ABE3, ABE2), ABE)
            LETn = _where(run, LET_fin, LET)
            first_stop = run & stop & c['active']
            LTOPn = _where(first_stop, nk, c['LTOP'])
            activen = _where(run & stop, False, c['active'])
            return dict(F=F, WTW=WTWn, UPOLD=UPOLDn, UPNEW=UPNEWn, EE1=EE1n, UD1=UD1n, REI=REIn, ABE=ABEn, TRPPT=TRPPT2, LET=LETn, LTOP=LTOPn, TTEMP=TTEMP, active=activen, IFLAG=IFLAG)
        # WRF leaves the vertical sweep as soon as this parcel stops.
        # An unbuoyant source layer does not enter the updraft loop at all.
        def rise_cond(kc):
            nk, c = kc
            return (nk < KL) & c['active']

        def rise_step(kc):
            nk, c = kc
            return nk + 1, step(nk, c)

        _, carry = jax.lax.while_loop(rise_cond, rise_step, (K, carry))
        LTOP = _where(carry['active'], KL, carry['LTOP'])
        F = carry['F']
        ABE = carry['ABE']
        TRPPT = carry['TRPPT']
        LET = carry['LET']
        CLDHGT_LC = _get(Z0, LTOP) - ZLCL
        CHMIN = _where(TLCL > 293.0, 4000.0, _where(TLCL >= 273.0, 2000.0 + 100.0 * (TLCL - 273.0), 2000.0))
        no_conv = (LTOP <= KLCL) | (LTOP <= KPBL) | (LET + 1 <= KPBL)
        deep = buoyant & ~no_conv & (CLDHGT_LC > CHMIN) & (ABE > 1.0)
        shallow = buoyant & ~no_conv & ~deep
        abort = ~valid_depth | off_top
        return dict(F=F, abort=abort, buoyant=buoyant, no_conv=no_conv, deep=deep, shallow=shallow, CLDHGT_LC=_where(no_conv | ~buoyant, 0.0, CLDHGT_LC), LC=LC, K=K, KLCL=KLCL, KPBL=KPBL, LET=LET, LTOP=LTOP, LCL=KLCL, ZLCL=ZLCL, TLCL=TLCL, TVLCL=TVLCL, TVEN=TVEN, VMFLCL=VMFLCL, WLCL=WLCL, RAD=RAD, ABE=ABE, TRPPT=TRPPT, DPTHMX=DPTHMX, AU0=AU0, ZMIX=ZMIX, TMIX=TMIX, QMIX=QMIX, PMIX=PMIX, WKL=WKL, PLCL=PLCL, UPOLD=carry['UPOLD'], UPNEW=carry['UPNEW'])

    def _empty_candidate_state(N, KX):
        z = jnp.zeros(N, dtype=jnp.float32)
        one = jnp.asarray(1, dtype=jnp.int32)
        zero = jnp.asarray(0, dtype=jnp.int32)
        F = dict(UMF=z, UER=z, UDR=z, DETLQ=z, DETIC=z, PPTLIQ=z, PPTICE=z, QLIQ=z, QICE=z, QLQOUT=z, QICOUT=z, TU=z, TVU=z, QU=z, WU=z, THETEU=z, THETEE=z, TVQU=z, EQFRC=z, QDT=z, RATIO2=z, DILFRC=jnp.ones(N, dtype=jnp.float32))
        return dict(F=F, abort=jnp.array(False), buoyant=jnp.array(False), no_conv=jnp.array(True), deep=jnp.array(False), shallow=jnp.array(False), CLDHGT_LC=jnp.float32(0.0), LC=one, K=zero, KLCL=one, KPBL=one, LET=one, LTOP=one, LCL=one, ZLCL=jnp.float32(0.0), TLCL=jnp.float32(0.0), TVLCL=jnp.float32(0.0), TVEN=jnp.float32(0.0), VMFLCL=jnp.float32(0.0), WLCL=jnp.float32(0.0), RAD=jnp.float32(0.0), ABE=jnp.float32(0.0), TRPPT=jnp.float32(0.0), DPTHMX=jnp.float32(0.0), AU0=jnp.float32(0.0), ZMIX=jnp.float32(0.0), TMIX=jnp.float32(0.0), QMIX=jnp.float32(0.0), PMIX=jnp.float32(0.0), WKL=jnp.float32(0.0), PLCL=jnp.float32(0.0), UPOLD=jnp.float32(0.0), UPNEW=jnp.float32(0.0))

    def _tree_where(pred, on_true, on_false):
        return jax.tree_util.tree_map(lambda a, b: _where(pred, a, b), on_true, on_false)

    def _search_usl(KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, W0Ap, dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu):
        """Fortran-faithful USL walk.

        Deep convection stops on the first triggering source layer. If only shallow
        candidates are found, scan all candidates once to find NUCHM (max cloud
        height), then rerun that candidate and stop there, matching KF_eta_PARA.
        """
        empty = _empty_candidate_state(_get(idx.shape, 0), KX)
        init = dict(nu=jnp.int32(1), phase=jnp.int32(0), have_shallow=jnp.array(False), best_nu=jnp.int32(0), best_height=jnp.float32(-1.0), selected=empty, convect=jnp.array(False), ishall=jnp.int32(2), done=jnp.array(False))

        def run_current(st):
            cand = _run_updraft(st['nu'], KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, W0Ap, dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu)
            phase1 = st['phase'] == 1
            deep = cand['deep'] & ~cand['abort']
            shallow = cand['shallow'] & ~cand['abort']
            accept_shallow = phase1 & shallow
            accept = deep | accept_shallow
            better_shallow = (st['phase'] == 0) & shallow & (~st['have_shallow'] | (cand['CLDHGT_LC'] > st['best_height']))
            terminal_miss = phase1 & ~deep & ~shallow
            done = cand['abort'] | accept | terminal_miss
            return dict(nu=_where(done, st['nu'], st['nu'] + 1), phase=st['phase'], have_shallow=st['have_shallow'] | (st['phase'] == 0) & shallow, best_nu=_where(better_shallow, st['nu'], st['best_nu']), best_height=_where(better_shallow, cand['CLDHGT_LC'], st['best_height']), selected=_tree_where(accept, cand, st['selected']), convect=accept, ishall=_where(deep, jnp.int32(0), _where(accept_shallow, jnp.int32(1), st['ishall'])), done=done)

        def exhausted(st):
            rerun_shallow = (st['phase'] == 0) & st['have_shallow']
            return dict(nu=_where(rerun_shallow, st['best_nu'], st['nu']), phase=_where(rerun_shallow, jnp.int32(1), st['phase']), have_shallow=st['have_shallow'], best_nu=st['best_nu'], best_height=st['best_height'], selected=st['selected'], convect=st['convect'], ishall=st['ishall'], done=~rerun_shallow)

        def body(st):
            if False:
                return _tree_where(st['nu'] <= NCHECK, run_current(st), exhausted(st))
            return jax.lax.cond(st['nu'] <= NCHECK, run_current, exhausted, st)

        def cond(st):
            return ~st['done']
        out = jax.lax.while_loop(cond, body, init)
        return (out['selected'], out['ishall'], out['convect'])

    def kf_eta_para(T0, QV0, P0, DZQ, RHOE, W0A, U0, V0, dt, dx, KX, warm_rain=False, f_qi=True, f_qs=True):
        """JAX KF_eta_PARA for ONE column (inputs length KX, 0-based, bottom-up).

        Faithful translation of the validated NumPy reference. All control flow is
        masked / lax-based; vmappable; GPU-resident."""
        N = _get(T0.shape, 0)
        KL = KX
        DXSQ = dx * dx
        aliq, bliq, cliq, dliq = (ALIQ, BLIQ, CLIQ, DLIQ)
        alu = alu_ref

        def pad(a):
            return a
        T0p = pad(T0)
        QV0p = pad(QV0)
        P0p = pad(P0)
        DZQp = pad(DZQ)
        RHOEp = pad(RHOE)
        W0Ap = pad(W0A)
        U0p = pad(U0)
        V0p = pad(V0)
        idx = jnp.arange(N, dtype=jnp.int32)
        lev = (idx >= 1) & (idx <= KX)
        es = aliq * jnp.exp((bliq * T0p - cliq) / _where(lev, T0p - dliq, 1.0))
        QES = _where(lev, 0.622 * es / (P0p - es), 0.0)
        Q0 = _where(lev, jnp.maximum(1e-06, jnp.minimum(QES, QV0p)), 0.0)
        RH = _where(lev, Q0 / _where(QES > 0, QES, 1.0), 0.0)
        TV0 = _where(lev, T0p * (1.0 + 0.608 * Q0), 0.0)
        DP = _where(lev, RHOEp * G * DZQp, 0.0)
        cdz = jnp.cumsum(_where(lev, DZQp, 0.0))
        Z0 = _where(lev, cdz - 0.5 * DZQp, 0.0)
        DZA = _where(lev, _roll(Z0, -1) - Z0, 0.0)
        P300 = _get(P0p, 1) - 30000.0
        L5 = jnp.max(_where(lev & (P0p >= 0.5 * _get(P0p, 1)), idx, 0))
        LLFC = jnp.max(_where(lev & (P0p >= P300), idx, 0))

        def kcheck_body(k, carry):
            kchk, ncheck, pm15 = carry
            cond = (k <= LLFC) & (_get(P0p, k) < pm15)
            ncheck2 = _where(cond, ncheck + 1, ncheck)
            kchk2 = _where(cond, _set(kchk, ncheck2, k), kchk)
            pm15_2 = _where(cond, pm15 - 1500.0, pm15)
            return (kchk2, ncheck2, pm15_2)
        KCHK0 = _set(jnp.zeros(N, dtype=jnp.int32), 1, 1)
        KCHECK, NCHECK, _ = jax.lax.fori_loop(2, KX + 1, kcheck_body, (KCHK0, jnp.int32(1), _get(P0p, 1) - 1500.0))
        S, ISHALL, convect = _search_usl(KCHECK, NCHECK, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, W0Ap, dx, DXSQ, KX, KL, aliq, bliq, cliq, dliq, alu)
        NCA_none = jnp.float32(-100.0)

        def closure(_):
            return _closure_feedback(S, ISHALL, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, QES, RH, U0p, V0p, dt, dx, DXSQ, KX, KL, L5, aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs)
        return jax.lax.cond(convect, closure, lambda _: _empty_col(N, -100.0, KX), None)

    def _closure_feedback(S, ISHALL, lev, idx, Z0, DZA, DP, T0p, Q0, TV0, P0p, QES, RH, U0p, V0p, dt, dx, DXSQ, KX, KL, L5, aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs):
        """Downdraft + mass-flux closure + advection + feedback tendencies.
        S = selected-candidate state from _run_updraft. Mirrors the validated
        reference post-USL section. Returns the output dict (length-KX tendencies)."""
        N = _get(idx.shape, 0)
        F = S['F']
        LC = S['LC']
        K = S['K']
        KLCL = S['KLCL']
        KPBL = S['KPBL']
        LET = S['LET']
        LTOP = S['LTOP']
        LCL = S['LCL']
        ZLCL = S['ZLCL']
        TLCL = S['TLCL']
        TVLCL = S['TVLCL']
        TVEN = S['TVEN']
        VMFLCL = S['VMFLCL']
        WLCL = S['WLCL']
        RAD = S['RAD']
        ABE = S['ABE']
        TRPPT = S['TRPPT']
        DPTHMX = S['DPTHMX']
        AU0 = S['AU0']
        ZMIX = S['ZMIX']
        TMIX = S['TMIX']
        QMIX = S['QMIX']
        PMIX = S['PMIX']
        PLCL = S['PLCL']
        UPOLD = S['UPOLD']
        UPNEW = S['UPNEW']
        shallow = ISHALL == 1
        is_lev = lambda k: (k >= 1) & (k <= LTOP)
        UMF = F['UMF']
        UER = F['UER']
        UDR = F['UDR']
        DETLQ = F['DETLQ']
        DETIC = F['DETIC']
        PPTLIQ = F['PPTLIQ']
        PPTICE = F['PPTICE']
        QLIQ = F['QLIQ']
        QICE = F['QICE']
        QLQOUT = F['QLQOUT']
        QICOUT = F['QICOUT']
        TU = F['TU']
        QU = F['QU']
        WU = F['WU']
        THETEU = F['THETEU']
        THETEE = F['THETEE']
        QDT = F['QDT']
        DILFRC = F['DILFRC']
        EQFRC = F['EQFRC']
        KSTART_sh = jnp.maximum(KPBL, KLCL)
        LET = _where(shallow, KSTART_sh, LET)
        let_eq = LET == LTOP
        udr_top = _get(UMF, LTOP) + _get(UDR, LTOP) - _get(UER, LTOP)
        ratio_up = UPNEW / _where(UPOLD != 0, UPOLD, 1.0)
        detlq_top = _get(QLIQ, LTOP) * udr_top * ratio_up
        detic_top = _get(QICE, LTOP) * udr_top * ratio_up
        UDR = _set(UDR, LTOP, _where(let_eq, udr_top, _get(UDR, LTOP)))
        DETLQ = _set(DETLQ, LTOP, _where(let_eq, detlq_top, _get(DETLQ, LTOP)))
        DETIC = _set(DETIC, LTOP, _where(let_eq, detic_top, _get(DETIC, LTOP)))
        UER = _set(UER, LTOP, _where(let_eq, 0.0, _get(UER, LTOP)))
        UMF = _set(UMF, LTOP, _where(let_eq, 0.0, _get(UMF, LTOP)))
        DPTT = jnp.sum(_where(lev & (idx >= LET + 1) & (idx <= LTOP), DP, 0.0))
        DUMFDP = _get(UMF, LET) / _where(DPTT != 0, DPTT, 1.0)

        def lindet(nk, carry):
            UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT = carry
            do = ~let_eq & (nk >= LET + 1) & (nk <= LTOP)
            is_top = nk == LTOP
            umf_nt = _get(UMF, nk - 1) - _get(DP, nk) * DUMFDP
            uer_nt = umf_nt * (1.0 - 1.0 / _where(_get(DILFRC, nk) != 0, _get(DILFRC, nk), 1.0))
            udr_nt = _get(UMF, nk - 1) - umf_nt + uer_nt
            udr_tp = _get(UMF, nk - 1)
            umf_new = _where(do, _where(is_top, _get(UMF, nk), umf_nt), _get(UMF, nk))
            uer_new = _where(do, _where(is_top, 0.0, uer_nt), _get(UER, nk))
            udr_new = _where(do, _where(is_top, udr_tp, udr_nt), _get(UDR, nk))
            detlq_new = _where(do, udr_new * _get(QLIQ, nk) * _get(DILFRC, nk), _get(DETLQ, nk))
            detic_new = _where(do, udr_new * _get(QICE, nk) * _get(DILFRC, nk), _get(DETIC, nk))
            UMF = _set(UMF, nk, umf_new)
            UER = _set(UER, nk, uer_new)
            UDR = _set(UDR, nk, udr_new)
            DETLQ = _set(DETLQ, nk, detlq_new)
            DETIC = _set(DETIC, nk, detic_new)
            do2 = do & (nk >= LET + 2)
            TRPPT = _where(do2, TRPPT - _get(PPTLIQ, nk) - _get(PPTICE, nk), TRPPT)
            pptl = _get(UMF, nk - 1) * _get(QLQOUT, nk)
            ppti = _get(UMF, nk - 1) * _get(QICOUT, nk)
            PPTLIQ = _set(PPTLIQ, nk, _where(do2, pptl, _get(PPTLIQ, nk)))
            PPTICE = _set(PPTICE, nk, _where(do2, ppti, _get(PPTICE, nk)))
            TRPPT = _where(do2, TRPPT + pptl + ppti, TRPPT)
            return (UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT)
        UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT = jax.lax.fori_loop(1, KX + 1, lindet, (UMF, UER, UDR, DETLQ, DETIC, PPTLIQ, PPTICE, TRPPT))
        ML = jnp.max(_where(lev & (idx <= LTOP) & (T0p > _T00), idx, 0))

        def below(nk, carry):
            UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, THETEE, EQFRC = carry
            do = (nk >= 1) & (nk <= K)
            ge_lc = nk >= LC
            at_lc = nk == LC
            le_pbl = nk <= KPBL
            uer_lc = VMFLCL * _get(DP, nk) / DPTHMX
            umf_lc = uer_lc
            umf_pbl = _get(UMF, nk - 1) + VMFLCL * _get(DP, nk) / DPTHMX
            uer_pblv = VMFLCL * _get(DP, nk) / DPTHMX
            umf_top = VMFLCL
            umf_v = _where(at_lc, umf_lc, _where(le_pbl, umf_pbl, umf_top))
            uer_v = _where(at_lc, uer_lc, _where(le_pbl, uer_pblv, 0.0))
            tu_v = TMIX + (_get(Z0, nk) - ZMIX) * GDRY
            UMF = _set(UMF, nk, _where(do, _where(ge_lc, umf_v, 0.0), _get(UMF, nk)))
            UER = _set(UER, nk, _where(do, _where(ge_lc, uer_v, 0.0), _get(UER, nk)))
            TU = _set(TU, nk, _where(do, _where(ge_lc, tu_v, 0.0), _get(TU, nk)))
            QU = _set(QU, nk, _where(do, _where(ge_lc, QMIX, 0.0), _get(QU, nk)))
            WU = _set(WU, nk, _where(do, _where(ge_lc, WLCL, 0.0), _get(WU, nk)))
            UDR = _set(UDR, nk, _where(do, 0.0, _get(UDR, nk)))
            QLIQ = _set(QLIQ, nk, _where(do, 0.0, _get(QLIQ, nk)))
            QICE = _set(QICE, nk, _where(do, 0.0, _get(QICE, nk)))
            QLQOUT = _set(QLQOUT, nk, _where(do, 0.0, _get(QLQOUT, nk)))
            QICOUT = _set(QICOUT, nk, _where(do, 0.0, _get(QICOUT, nk)))
            PPTLIQ = _set(PPTLIQ, nk, _where(do, 0.0, _get(PPTLIQ, nk)))
            PPTICE = _set(PPTICE, nk, _where(do, 0.0, _get(PPTICE, nk)))
            DETLQ = _set(DETLQ, nk, _where(do, 0.0, _get(DETLQ, nk)))
            DETIC = _set(DETIC, nk, _where(do, 0.0, _get(DETIC, nk)))
            thtee = envirtht(_get(P0p, nk), _get(T0p, nk), _get(Q0, nk), aliq, bliq, cliq, dliq)
            THETEE = _set(THETEE, nk, _where(do, thtee, _get(THETEE, nk)))
            EQFRC = _set(EQFRC, nk, _where(do, 1.0, _get(EQFRC, nk)))
            return (UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, THETEE, EQFRC)
        UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, THETEE, EQFRC = jax.lax.fori_loop(1, KX + 1, below, (UMF, UER, UDR, TU, QU, WU, QLIQ, QICE, QLQOUT, QICOUT, PPTLIQ, PPTICE, DETLQ, DETIC, THETEE, EQFRC))
        above = (idx >= LTOP + 1) & (idx <= KX)
        UMF = _where(above, 0.0, UMF)
        UER = _where(above, 0.0, UER)
        UDR = _where(above, 0.0, UDR)
        QDT = _where(above, 0.0, QDT)
        QLIQ = _where(above, 0.0, QLIQ)
        QICE = _where(above, 0.0, QICE)
        QLQOUT = _where(above, 0.0, QLQOUT)
        QICOUT = _where(above, 0.0, QICOUT)
        DETLQ = _where(above, 0.0, DETLQ)
        DETIC = _where(above, 0.0, DETIC)
        PPTLIQ = _where(above, 0.0, PPTLIQ)
        PPTICE = _where(above, 0.0, PPTICE)
        abv2 = (idx >= LTOP + 2) & (idx <= KX)
        TU = _where(abv2, 0.0, TU)
        QU = _where(abv2, 0.0, QU)
        WU = _where(abv2, 0.0, WU)
        inLT = lev & (idx <= LTOP)
        EMS = _where(inLT, DP * DXSQ / G, 0.0)
        EMSD = _where(inLT, 1.0 / _where(EMS != 0, EMS, 1.0), 0.0)
        EXN_u = (_P00 / P0p) ** (0.2854 * (1.0 - 0.28 * QDT))
        THTAU = _where(inLT, TU * EXN_u, 0.0)
        EXN0 = (_P00 / P0p) ** (0.2854 * (1.0 - 0.28 * Q0))
        THTA0 = _where(inLT, T0p * EXN0, 0.0)
        DDILFRC = _where(inLT, 1.0 / _where(DILFRC != 0, DILFRC, 1.0), 0.0)
        THTAD = jnp.zeros(N, dtype=jnp.float32)
        QD = jnp.zeros(N, dtype=jnp.float32)
        TZ = jnp.zeros(N, dtype=jnp.float32)
        DMF = jnp.zeros(N, dtype=jnp.float32)
        DER = jnp.zeros(N, dtype=jnp.float32)
        DDR = jnp.zeros(N, dtype=jnp.float32)
        QSD = jnp.zeros(N, dtype=jnp.float32)
        WD = jnp.zeros(N, dtype=jnp.float32)
        TVD = jnp.zeros(N, dtype=jnp.float32)
        THETED = jnp.zeros(N, dtype=jnp.float32)
        WSk = jnp.sqrt(_get(U0p, KLCL) ** 2 + _get(V0p, KLCL) ** 2)
        WS5 = jnp.sqrt(_get(U0p, L5) ** 2 + _get(V0p, L5) ** 2)
        WSt = jnp.sqrt(_get(U0p, LTOP) ** 2 + _get(V0p, LTOP) ** 2)
        VCONV = 0.5 * (WSk + WS5)
        TIMEC0 = _where(VCONV > 0, dx / VCONV, 1e+30)
        TADVEC = TIMEC0
        TIMEC = jnp.minimum(3600.0, jnp.maximum(1800.0, TIMEC0))
        TIMEC = _where(shallow, 2400.0, TIMEC)
        NIC = _nint_positive(TIMEC / dt).astype(jnp.int32)
        TIMEC = NIC.astype(jnp.float32) * dt
        SHSIGN = _where(WSt > WSk, 1.0, -1.0)
        VWS = (_get(U0p, LTOP) - _get(U0p, KLCL)) ** 2 + (_get(V0p, LTOP) - _get(V0p, KLCL)) ** 2
        VWS = 1000.0 * SHSIGN * jnp.sqrt(VWS) / (_get(Z0, LTOP) - _get(Z0, LCL))
        PEF = 1.591 + VWS * (-0.639 + VWS * (0.0953 - VWS * 0.00496))
        PEF = jnp.minimum(0.9, jnp.maximum(0.2, PEF))
        CBH = (ZLCL - _get(Z0, 1)) * 0.003281
        RCBH = _where(CBH < 3.0, 0.02, 0.96729352 + CBH * (-0.70034167 + CBH * (0.162179896 + CBH * (-0.012569798 + CBH * (0.00042772 - CBH * 5.44e-06)))))
        RCBH = _where(CBH > 25.0, 2.4, RCBH)
        PEFCBH = jnp.minimum(1.0 / (1.0 + RCBH), 0.9)
        PEFF = 0.5 * (PEF + PEFCBH)
        KSTART = _where(shallow, 1, KPBL + 1)
        dppp = _get(P0p, KSTART) - P0p
        klfs_cand = lev & (idx >= KSTART + 1) & (dppp > 15000.0)
        KLFS = jnp.minimum(jnp.min(_where(klfs_cand, idx, KX + 99)), LET - 1)
        KLFS = jnp.clip(KLFS, 1, KL)
        LFS = KLFS
        dd_ok = ~shallow & (_get(P0p, KSTART) - _get(P0p, LFS) > 5000.0)
        theted_lfs = _get(THETEE, LFS)
        qd_lfs = _get(Q0, LFS)
        tz_lfs, qss_lfs = tpmix2dd(_get(P0p, LFS), theted_lfs)
        thtad_lfs = tz_lfs * (_P00 / _get(P0p, LFS)) ** (0.2854 * (1.0 - 0.28 * qss_lfs))
        tvd_lfs = tz_lfs * (1.0 + 0.608 * qss_lfs)
        rdd = _get(P0p, LFS) / (R_D * tvd_lfs)
        A1 = (1.0 - PEFF) * AU0
        dmf_lfs = -A1 * rdd
        THETED = _set(THETED, LFS, theted_lfs)
        QD = _set(QD, LFS, qd_lfs)
        TZ = _set(TZ, LFS, tz_lfs)
        THTAD = _set(THTAD, LFS, thtad_lfs)
        TVD = _set(TVD, LFS, tvd_lfs)
        DMF = _set(DMF, LFS, dmf_lfs)
        DER = _set(DER, LFS, dmf_lfs)

        def dd_down(j, carry):
            DMF, DER, THETED, QD, rhbar, dptt = carry
            nd = LFS - j
            do = dd_ok & (nd >= KSTART) & (nd <= LFS - 1)
            nd1 = nd + 1
            der_v = _get(DER, LFS) * _get(EMS, nd) / _where(_get(EMS, LFS) != 0, _get(EMS, LFS), 1.0)
            dmf_v = _get(DMF, nd1) + der_v
            theted_v = (_get(THETED, nd1) * _get(DMF, nd1) + _get(THETEE, nd) * der_v) / _where(dmf_v != 0, dmf_v, 1.0)
            qd_v = (_get(QD, nd1) * _get(DMF, nd1) + _get(Q0, nd) * der_v) / _where(dmf_v != 0, dmf_v, 1.0)
            DER = _set(DER, nd, _where(do, der_v, _get(DER, nd)))
            DMF = _set(DMF, nd, _where(do, dmf_v, _get(DMF, nd)))
            THETED = _set(THETED, nd, _where(do, theted_v, _get(THETED, nd)))
            QD = _set(QD, nd, _where(do, qd_v, _get(QD, nd)))
            rhbar = _where(do, rhbar + _get(RH, nd) * _get(DP, nd), rhbar)
            dptt = _where(do, dptt + _get(DP, nd), dptt)
            return (DMF, DER, THETED, QD, rhbar, dptt)
        DMF, DER, THETED, QD, RHBAR, DPTT = jax.lax.fori_loop(1, KX + 1, dd_down, (DMF, DER, THETED, QD, _get(RH, LFS) * _get(DP, LFS), _get(DP, LFS)))
        RHBAR = RHBAR / _where(DPTT != 0, DPTT, 1.0)
        DMFFRC = 2.0 * (1.0 - RHBAR)
        pptmlt = jnp.sum(_where(lev & (idx >= KLCL) & (idx <= LTOP), PPTICE, 0.0))
        DTMELT = _where(LC < ML, _RLF * pptmlt / (CP * _where(_get(UMF, KLCL) != 0, _get(UMF, KLCL), 1.0)), 0.0)
        LDT = jnp.minimum(LFS - 1, KSTART - 1)
        tzks, qss_ks = tpmix2dd(_get(P0p, KSTART), _get(THETED, KSTART))
        tzks = tzks - DTMELT
        es_ks = aliq * jnp.exp((bliq * tzks - cliq) / (tzks - dliq))
        qss_ks = 0.622 * es_ks / (_get(P0p, KSTART) - es_ks)
        theted_ks = tzks * (100000.0 / _get(P0p, KSTART)) ** (0.2854 * (1.0 - 0.28 * qss_ks)) * jnp.exp((3374.6525 / tzks - 2.5403) * qss_ks * (1.0 + 0.81 * qss_ks))
        TZ = _set(TZ, KSTART, _where(dd_ok, tzks, _get(TZ, KSTART)))
        THETED = _set(THETED, KSTART, _where(dd_ok, theted_ks, _get(THETED, KSTART)))

        def dd_evap(j, carry):
            TZ, QD, QSD, THTAD, TVD, ldb, found, dpdd = carry
            nd = LDT - j
            do = dd_ok & (nd >= 1) & (nd <= LDT) & ~found
            dpdd = _where(do, dpdd + _get(DP, nd), dpdd)
            theted_nd = _get(THETED, KSTART)
            qd_nd = _get(QD, KSTART)
            tz_nd, qss = tpmix2dd(_get(P0p, nd), theted_nd)
            RHH = 1.0 - 0.2 / 1000.0 * (_get(Z0, KSTART) - _get(Z0, nd))
            DSSDT = (cliq - bliq * dliq) / ((tz_nd - dliq) * (tz_nd - dliq))
            RL = XLV0 - XLV1 * tz_nd
            DTMP = RL * qss * (1.0 - RHH) / (CP + RL * RHH * qss * DSSDT)
            T1RH = tz_nd + DTMP
            es2 = RHH * aliq * jnp.exp((bliq * T1RH - cliq) / (T1RH - dliq))
            QSRH = 0.622 * es2 / (_get(P0p, nd) - es2)
            belowq = QSRH < qd_nd
            QSRH2 = _where(belowq, qd_nd, QSRH)
            T1RH2 = _where(belowq, tz_nd + (qss - QSRH2) * RL / CP, T1RH)
            rhh_lt1 = RHH < 1.0
            tz_final = _where(rhh_lt1, T1RH2, tz_nd)
            qss_final = _where(rhh_lt1, QSRH2, qss)
            TZ = _set(TZ, nd, _where(do, tz_final, _get(TZ, nd)))
            QD = _set(QD, nd, _where(do, qd_nd, _get(QD, nd)))
            QSD = _set(QSD, nd, _where(do, qss_final, _get(QSD, nd)))
            tvd_nd = tz_final * (1.0 + 0.608 * qss_final)
            TVD = _set(TVD, nd, _where(do, tvd_nd, _get(TVD, nd)))
            hit = do & ((tvd_nd > _get(TV0, nd)) | (nd == 1))
            ldb = _where(hit & ~found, nd, ldb)
            found = found | hit
            return (TZ, QD, QSD, THTAD, TVD, ldb, found, dpdd)
        TZ, QD, QSD, THTAD, TVD, LDB, _found, DPDD = jax.lax.fori_loop(0, KX + 1, dd_evap, (TZ, QD, QSD, THTAD, TVD, jnp.asarray(1, dtype=jnp.int32), jnp.array(False), jnp.float32(0.0)))
        dd_depth_ok = dd_ok & (_get(P0p, LDB) - _get(P0p, LFS) > 5000.0)

        def dd_final(j, carry):
            DDR, DER, DMF, TDER, QD, THTAD = carry
            nd = LDT - j
            do = dd_depth_ok & (nd >= LDB) & (nd <= LDT)
            nd1 = nd + 1
            ddr_v = -_get(DMF, KSTART) * _get(DP, nd) / _where(DPDD != 0, DPDD, 1.0)
            dmf_v = _get(DMF, nd1) + ddr_v
            TDER = _where(do, TDER + (_get(QSD, nd) - _get(QD, nd)) * ddr_v, TDER)
            qd_v = _get(QSD, nd)
            thtad_v = _get(TZ, nd) * (_P00 / _get(P0p, nd)) ** (0.2854 * (1.0 - 0.28 * qd_v))
            DDR = _set(DDR, nd, _where(do, ddr_v, _get(DDR, nd)))
            DER = _set(DER, nd, _where(do, 0.0, _get(DER, nd)))
            DMF = _set(DMF, nd, _where(do, dmf_v, _get(DMF, nd)))
            QD = _set(QD, nd, _where(do, qd_v, _get(QD, nd)))
            THTAD = _set(THTAD, nd, _where(do, thtad_v, _get(THTAD, nd)))
            return (DDR, DER, DMF, TDER, QD, THTAD)
        DDR, DER, DMF, TDER, QD, THTAD = jax.lax.fori_loop(0, KX + 1, dd_final, (DDR, DER, DMF, jnp.float32(0.0), QD, THTAD))
        no_dd = TDER < 1.0
        inLT_arr = (idx >= 1) & (idx <= LTOP)
        DMF = _where(no_dd & inLT_arr, 0.0, DMF)
        DER = _where(no_dd & inLT_arr, 0.0, DER)
        DDR = _where(no_dd & inLT_arr, 0.0, DDR)
        THTAD = _where(no_dd & inLT_arr, 0.0, THTAD)
        WD = _where(no_dd & inLT_arr, 0.0, WD)
        TZ = _where(no_dd & inLT_arr, 0.0, TZ)
        QD = _where(no_dd & inLT_arr, 0.0, QD)
        LDB = _where(no_dd, LFS, LDB)
        TDER = _where(no_dd, 0.0, TDER)
        DDINC = -DMFFRC * _get(UMF, KLCL) / _where(_get(DMF, KSTART) != 0, _get(DMF, KSTART), 1.0)
        DDINC = _where(TDER * DDINC > TRPPT, _where(TDER != 0, TRPPT / TDER, DDINC), DDINC)
        TDER_dd = TDER * DDINC
        scale_dd = ~no_dd & lev & (idx >= LDB) & (idx <= LFS)
        DMF = _where(scale_dd, DMF * DDINC, DMF)
        DER = _where(scale_dd, DER * DDINC, DER)
        DDR = _where(scale_dd, DDR * DDINC, DDR)
        TDER = _where(no_dd, TDER, TDER_dd)
        below_ldb = ~no_dd & lev & (idx >= 1) & (idx <= LDB - 1) & (LDB > 1)
        above_lfs = ~no_dd & lev & (idx >= LFS + 1) & (idx <= KX)
        DMF = _where(below_ldb | above_lfs, 0.0, DMF)
        DER = _where(below_ldb | above_lfs, 0.0, DER)
        DDR = _where(below_ldb | above_lfs, 0.0, DDR)
        THTAD = _where(below_ldb | above_lfs, 0.0, THTAD)
        QD = _where(below_ldb | above_lfs, 0.0, QD)
        TZ = _where(below_ldb | above_lfs, 0.0, TZ)
        mid_zero = ~no_dd & lev & (idx >= LDT + 1) & (idx <= LFS - 1)
        TZ = _where(mid_zero, 0.0, TZ)
        QD = _where(mid_zero, 0.0, QD)
        THTAD = _where(mid_zero, 0.0, THTAD)
        CPR = TRPPT
        PPTFLX = _where(no_dd, TRPPT, TRPPT - TDER)
        LMAX = jnp.maximum(KLCL, LFS)
        aincm_mask = lev & (idx >= LC) & (idx <= LMAX) & (UER - DER > 0.001)
        aincm1 = _where(aincm_mask, EMS / _where((UER - DER) * TIMEC != 0, (UER - DER) * TIMEC, 1.0), 1000.0)
        AINCMX = jnp.minimum(1000.0, jnp.min(_where(aincm_mask, aincm1, 1000.0)))
        AINC = _where(AINCMX < 1.0, AINCMX, 1.0)
        DETLQ2 = DETLQ
        DETIC2 = DETIC
        UDR2 = UDR
        UER2 = UER
        DDR2 = DDR
        DER2 = DER
        UMF2 = UMF
        DMF2 = DMF
        TDER2 = TDER
        PPTFL2 = PPTFLX
        TKEMAX = 5.0
        EVAC = 0.5 * TKEMAX * 0.1
        AINC_sh = EVAC * DPTHMX * DXSQ / (VMFLCL * G * TIMEC)
        AINC = _where(shallow, AINC_sh, AINC)
        out = _kf_closure_iter(UMF, UER, UDR, DMF, DER, DDR, DETLQ, DETIC, PPTLIQ, PPTICE, UMF2, UER2, UDR2, DMF2, DER2, DDR2, DETLQ2, DETIC2, TDER2, PPTFL2, AINC, AINCMX, EMS, EMSD, DP, OMG_init(N), THTA0, THTAU, THTAD, QDT, QD, Q0, P0p, Z0, DZA, DXSQ, TIMEC, TADVEC, dt, ABE, PMIX, TMIX, QMIX, ZMIX, DPTHMX, LC, KPBL, KLCL, LET, LTOP, K, KL, KX, lev, idx, TV0, TVG_zero(N), DDILFRC, QLIQ, QICE, EQFRC, LFS, DMFFRC, shallow, CPR, PPTFLX, TRPPT, T0p, TG_init(T0p, KX, N), QES, ML, LCL, aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs, NIC)
        return out

    def OMG_init(N):
        return jnp.zeros(N, dtype=jnp.float32)

    def TVG_zero(N):
        return jnp.zeros(N, dtype=jnp.float32)

    def TG_init(T0p, KX, N):
        return jnp.zeros(N, dtype=jnp.float32)

    def _kf_closure_iter(UMF, UER, UDR, DMF, DER, DDR, DETLQ, DETIC, PPTLIQ, PPTICE, UMF2, UER2, UDR2, DMF2, DER2, DDR2, DETLQ2, DETIC2, TDER2, PPTFL2, AINC, AINCMX, EMS, EMSD, DP, OMG, THTA0, THTAU, THTAD, QDT, QD, Q0, P0p, Z0, DZA, DXSQ, TIMEC, TADVEC, dt, ABE, PMIX, TMIX0, QMIX0, ZMIX, DPTHMX, LC, KPBL, KLCL0, LET, LTOP, K0, KL, KX, lev, idx, TV0, TVG_unused, DDILFRC, QLIQ, QICE, EQFRC, LFS, DMFFRC, shallow, CPR, PPTFLX, TRPPT, T0p, TG_unused, QES, ML, LCL, aliq, bliq, cliq, dliq, alu, warm_rain, f_qi, f_qs, NIC):
        """Mass-flux closure iteration (<=10), advection, CAPE-removal + feedback.
        Faithful to the validated reference; fully masked/lax. Returns output dict."""
        N = _get(idx.shape, 0)
        inLT = lev & (idx <= LTOP)

        def scale_all(ainc, UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR):
            UMF = _where(inLT, UMF2 * ainc, UMF)
            DMF = _where(inLT, DMF2 * ainc, DMF)
            DETLQ = _where(inLT, DETLQ2 * ainc, DETLQ)
            DETIC = _where(inLT, DETIC2 * ainc, DETIC)
            UDR = _where(inLT, UDR2 * ainc, UDR)
            UER = _where(inLT, UER2 * ainc, UER)
            DER = _where(inLT, DER2 * ainc, DER)
            DDR = _where(inLT, DDR2 * ainc, DDR)
            return (UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR)
        UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR = jax.lax.cond(shallow, lambda a: scale_all(AINC, *a), lambda a: a, (UMF, DMF, DETLQ, DETIC, UDR, UER, DER, DDR))
        TDER = _where(shallow, TDER2 * AINC, TDER2)
        PPTFLX = _where(shallow, PPTFL2 * AINC, PPTFLX)
        carry = dict(UMF=UMF, UER=UER, UDR=UDR, DMF=DMF, DER=DER, DDR=DDR, DETLQ=DETLQ, DETIC=DETIC, AINC=AINC, AINCOLD=AINC, FABEOLD=1.0, NOITR=0, THTAG=jnp.zeros(N, dtype=jnp.float32), QG=jnp.zeros(N, dtype=jnp.float32), TG=jnp.zeros(N, dtype=jnp.float32), TVG=jnp.zeros(N, dtype=jnp.float32), OMG=jnp.zeros(N, dtype=jnp.float32), DOMGDP=jnp.zeros(N, dtype=jnp.float32), FXM=jnp.zeros(N, dtype=jnp.float32), NSTEP=jnp.int32(1), DTIME=TIMEC, done=jnp.array(False), rejected=jnp.array(False), KLCL=KLCL0, K=K0, PPTFLX=PPTFLX, TDER=TDER, FABE=1.0)

        def adv_and_cape(c):
            UMF = c['UMF']
            UER = c['UER']
            UDR = c['UDR']
            DMF = c['DMF']
            DER = c['DER']
            DDR = c['DDR']
            DOMGDP = _where(inLT, -(UER - DER - UDR - DDR) * EMSD, 0.0)
            contrib = _where(inLT, DP * DOMGDP, 0.0)
            OMG = -_roll(jnp.cumsum(contrib), 1)
            OMG = _where(idx <= LTOP, OMG, 0.0)
            OMG = _set(OMG, 0, 0.0)
            absomgtc = jnp.abs(OMG) * TIMEC
            frdp = 0.75 * _set(_roll(DP, 1), 0, 0.0)
            dtt1 = _where((absomgtc > frdp) & (idx >= 2) & (idx <= LTOP), frdp / _where(jnp.abs(OMG) > 0, jnp.abs(OMG), 1.0), TIMEC)
            DTT = jnp.minimum(TIMEC, jnp.min(_where((idx >= 2) & (idx <= LTOP), dtt1, TIMEC)))
            NSTEP = _nint_positive(TIMEC / DTT + 1).astype(jnp.int32)
            DTIME = TIMEC / NSTEP.astype(jnp.float32)
            FXM = _where(inLT, OMG * DXSQ / G, 0.0)
            THPA0 = _where(inLT, THTA0, 0.0)
            QPA0 = _where(inLT, Q0, 0.0)

            def substep(s, sc):
                do = s < NSTEP
                THPA, QPA = sc
                omg_le = OMG <= 0.0
                THFXIN = _where((idx >= 2) & inLT & omg_le, -FXM * _roll(THPA, 1), 0.0)
                QFXIN = _where((idx >= 2) & inLT & omg_le, -FXM * _roll(QPA, 1), 0.0)
                THFXOUT_up = _where((idx >= 2) & inLT & ~omg_le, FXM * THPA, 0.0)
                QFXOUT_up = _where((idx >= 2) & inLT & ~omg_le, FXM * QPA, 0.0)
                THFXOUT = _where(omg_le, 0.0, THFXOUT_up) + _roll(THFXIN, -1)
                QFXOUT = _where(omg_le, 0.0, QFXOUT_up) + _roll(QFXIN, -1)
                THFXIN2 = THFXIN + _roll(THFXOUT_up, -1)
                QFXIN2 = QFXIN + _roll(QFXOUT_up, -1)
                THFXIN_f = _where(inLT, THFXIN2, 0.0)
                QFXIN_f = _where(inLT, QFXIN2, 0.0)
                THFXOUT_f = _where(inLT, THFXOUT, 0.0)
                QFXOUT_f = _where(inLT, QFXOUT, 0.0)
                dTH = (THFXIN_f + UDR * THTAU + DDR * THTAD - THFXOUT_f - (UER - DER) * THTA0) * DTIME * EMSD
                dQ = (QFXIN_f + UDR * QDT + DDR * QD - QFXOUT_f - (UER - DER) * Q0) * DTIME * EMSD
                THPA_n = _where(do & inLT, THPA + dTH, THPA)
                QPA_n = _where(do & inLT, QPA + dQ, QPA)
                return (THPA_n, QPA_n)
            THPA, QPA = jax.lax.fori_loop(0, NSTEP, substep, (THPA0, QPA0))
            THTAG = _where(inLT, THPA, 0.0)
            QG = _where(inLT, QPA, 0.0)
            KLCL_borrow = c['KLCL']

            def borrow_body(nk, qg):
                needs_borrow = (nk <= LTOP) & (_get(qg, nk) < 0.0)

                def borrow(qg_in):

                    def surface_fatal(qg_surface):
                        return _set(qg_surface, nk, jnp.nan)

                    def adjacent_borrow(qg_adj):
                        nk1 = _where(nk == LTOP, KLCL_borrow, nk + 1)
                        tma = _get(qg_adj, nk1) * _get(EMS, nk1)
                        tmb = _get(qg_adj, nk - 1) * _get(EMS, nk - 1)
                        tmm = (_get(qg_adj, nk) - 1e-09) * _get(EMS, nk)
                        bcoeff = -tmm / (tma * tma / tmb + tmb)
                        acoeff = bcoeff * tma / tmb
                        tmb = tmb * (1.0 - bcoeff)
                        tma = tma * (1.0 - acoeff)
                        qg_adj = _set(qg_adj, nk, 1e-09)
                        qg_adj = _set(qg_adj, nk1, tma * _get(EMSD, nk1))
                        qg_adj = _set(qg_adj, nk - 1, tmb * _get(EMSD, nk - 1))
                        return qg_adj
                    return jax.lax.cond(nk == 1, surface_fatal, adjacent_borrow, qg_in)
                return jax.lax.cond(needs_borrow, borrow, lambda qg_in: qg_in, qg)
            QG = jax.lax.fori_loop(1, KX + 1, borrow_body, QG)
            EXN = (_P00 / P0p) ** (0.2854 * (1.0 - 0.28 * QG))
            TG = _where(inLT, THTAG / _where(EXN != 0, EXN, 1.0), 0.0)
            TVG = _where(inLT, TG * (1.0 + 0.608 * QG), 0.0)
            return (DOMGDP, OMG, FXM, NSTEP, DTIME, THTAG, QG, TG, TVG)

        def body(c, ncount):
            DOMGDP, OMG, FXM, NSTEP, DTIME, THTAG, QG, TG, TVG = adv_and_cape(c)
            advected = dict(c)
            advected.update(THTAG=THTAG, QG=QG, TG=TG, TVG=TVG, OMG=OMG,
                            DOMGDP=DOMGDP, FXM=FXM, NSTEP=NSTEP, DTIME=DTIME)

            def shallow_exit(a):
                a = dict(a)
                a['done'] = zero_runtime == 0.0
                return a

            def deep_closure(a):
                msk = lev & (idx >= LC) & (idx <= KPBL)
                TMIX = jnp.sum(_where(msk, DP * TG, 0.0)) / DPTHMX
                QMIX = jnp.sum(_where(msk, DP * QG, 0.0)) / DPTHMX
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
                QMIXc = jnp.maximum(QMIX, 0.0)
                EMIX = QMIXc * PMIX / (0.622 + QMIXc)
                a1 = EMIX / aliq
                tp = (a1 - 0.001) / 0.075
                indlu = jnp.clip(jnp.floor(tp).astype(jnp.int32), 0, 198)
                value = indlu * 0.075 + 0.001
                aintrp = (a1 - value) / 0.075
                tlog = aintrp * _get(alu, indlu + 1) + (1.0 - aintrp) * _get(alu, indlu)
                TDPT = (cliq - dliq * tlog) / (bliq - tlog)
                TLCL_u = TDPT - (0.212 + 0.001571 * (TDPT - _T00) - 0.000436 * (TMIX - _T00)) * (TMIX - TDPT)
                TLCL_u = jnp.minimum(TLCL_u, TMIX)
                TMIX2 = _where(sup, TMIX_s, TMIX)
                QMIX2 = _where(sup, QMIX_s, QMIXc)
                TLCL = _where(sup, TLCL_s, TLCL_u)
                TVLCL = TLCL * (1.0 + 0.608 * QMIX2)
                ZLCL = ZMIX + (TLCL - TMIX2) / GDRY
                ge = lev & (idx >= LC) & (ZLCL <= Z0)
                KLCL = jnp.clip(jnp.min(_where(ge, idx, KX + 99)), LC, KL)
                Kk = KLCL - 1
                DLP = (ZLCL - _get(Z0, Kk)) / (_get(Z0, KLCL) - _get(Z0, Kk))
                TENV = _get(TG, Kk) + (_get(TG, KLCL) - _get(TG, Kk)) * DLP
                QENV = _get(QG, Kk) + (_get(QG, KLCL) - _get(QG, Kk)) * DLP
                TVEN = TENV * (1.0 + 0.608 * QENV)
                THETEU_K = TMIX2 * (100000.0 / PMIX) ** (0.2854 * (1.0 - 0.28 * QMIX2)) * jnp.exp((3374.6525 / TLCL - 2.5403) * QMIX2 * (1.0 + 0.81 * QMIX2))

                def abeg_scan(carry2, nk):
                    ABEG, theteu_prev, tvqu_prev, tvg_prev = carry2
                    do = (nk >= Kk) & (nk <= LTOP - 1)
                    nk1 = nk + 1
                    theteu_nk1 = theteu_prev
                    tgu, qgu = tpmix2dd(_get(P0p, nk1), theteu_nk1)
                    tvqu = tgu * (1.0 + 0.608 * qgu - _get(QLIQ, nk1) - _get(QICE, nk1))
                    atK = nk == Kk
                    dzz = _where(atK, _get(Z0, KLCL) - ZLCL, _get(DZA, nk))
                    dilbe = _where(atK, ((TVLCL + tvqu) / (TVEN + _get(TVG, nk1)) - 1.0) * dzz, ((tvqu_prev + tvqu) / (_get(TVG, nk) + _get(TVG, nk1)) - 1.0) * dzz)
                    ABEG = _where(do & (dilbe > 0.0), ABEG + dilbe * G, ABEG)
                    thtee = envirtht(_get(P0p, nk1), _get(TG, nk1), _get(QG, nk1), aliq, bliq, cliq, dliq)
                    theteu_next = _where(do, theteu_nk1 * _get(DDILFRC, nk1) + thtee * (1.0 - _get(DDILFRC, nk1)), theteu_nk1)
                    return ((ABEG, theteu_next, tvqu, _get(TVG, nk1)), None)
                ABEG, _tu, _tq, _tg = jax.lax.fori_loop(0, KX, lambda nk, c: _get(abeg_scan(c, nk), 0), (0.0, THETEU_K, 0.0, 0.0))
                DABE = jnp.maximum(ABE - ABEG, 0.1 * ABE)
                FABE = ABEG / _where(ABE != 0, ABE, 1.0)
                ctrl = _closure_control(ncount, shallow, c['NOITR'], c['AINC'],
                                        c['AINCOLD'], c['FABEOLD'], FABE, ABE, DABE, AINCMX)
                a = dict(a)
                for key, unit in (('UMF', UMF2), ('DMF', DMF2), ('DETLQ', DETLQ2),
                                  ('DETIC', DETIC2), ('UDR', UDR2), ('UER', UER2),
                                  ('DER', DER2), ('DDR', DDR2)):
                    a[key] = _where(ctrl['rescale'] & inLT, unit * ctrl['AINC'], a[key])
                a.update(AINC=ctrl['AINC'], AINCOLD=ctrl['AINCOLD'],
                         FABEOLD=ctrl['FABEOLD'], NOITR=ctrl['NOITR'],
                         FABE=_where(c['NOITR'] == 1, c['FABE'], FABE),
                         KLCL=KLCL, K=Kk, done=ctrl['done'], rejected=ctrl['rejected'],
                         PPTFLX=_where(ctrl['rescale'], PPTFL2 * ctrl['AINC'], a['PPTFLX']),
                         TDER=_where(ctrl['rescale'], TDER2 * ctrl['AINC'], a['TDER']))
                return a

            return jax.lax.cond(shallow, shallow_exit, deep_closure, advected)

        def cond(c):
            return ~c['done']

        def iter_cond(ic):
            i, c = ic
            return (i < _MAX_CLOSURE_ITERS) & ~c['done']

        def iter_step(ic):
            i, c = ic
            return (i + 1, body(c, i + 1))
        _, carry = jax.lax.while_loop(iter_cond, iter_step, (jnp.int32(0), carry))
        def finish(_):
            THTAG = carry['THTAG']
            QG = carry['QG']
            TG = carry['TG']
            OMG = carry['OMG']
            FXM = carry['FXM']
            NSTEP = carry['NSTEP']
            DTIME = carry['DTIME']
            AINC = carry['AINC']
            PPTFLX = carry['PPTFLX']
            UMF = carry['UMF']
            UDR = carry['UDR']
            DDR = carry['DDR']
            DETLQ = carry['DETLQ']
            DETIC = carry['DETIC']
            fbfrc = _where(shallow, jnp.float32(1.0), jnp.float32(_FBFRC))
            FRC2 = _where(CPR > 0.0, PPTFLX / _where(CPR * AINC != 0, CPR * AINC, 1.0), 0.0)
            RAINFB = _where(inLT, PPTLIQ * AINC * fbfrc * FRC2, 0.0)
            SNOWFB = _where(inLT, PPTICE * AINC * fbfrc * FRC2, 0.0)
            QLPA = jnp.zeros(N, dtype=jnp.float32)
            QIPA = jnp.zeros(N, dtype=jnp.float32)
            QRPA = jnp.zeros(N, dtype=jnp.float32)
            QSPA = jnp.zeros(N, dtype=jnp.float32)

            def hyd_substep(s, sc):
                do = s < NSTEP
                QLPA, QIPA, QRPA, QSPA = sc
                omg_le = OMG <= 0.0

                def adv(QPA):
                    FXIN = _where((idx >= 2) & inLT & omg_le, -FXM * _roll(QPA, 1), 0.0)
                    FXOUT_up = _where((idx >= 2) & inLT & ~omg_le, FXM * QPA, 0.0)
                    FXOUT = _where(inLT, _where(omg_le, 0.0, FXOUT_up) + _roll(FXIN, -1), 0.0)
                    FXIN_f = _where(inLT, FXIN + _roll(FXOUT_up, -1), 0.0)
                    return (FXIN_f, FXOUT)
                QLFXIN, QLFXOUT = adv(QLPA)
                QIFXIN, QIFXOUT = adv(QIPA)
                QRFXIN, QRFXOUT = adv(QRPA)
                QSFXIN, QSFXOUT = adv(QSPA)
                QLn = _where(do & inLT, QLPA + (QLFXIN + DETLQ - QLFXOUT) * DTIME * EMSD, QLPA)
                QIn = _where(do & inLT, QIPA + (QIFXIN + DETIC - QIFXOUT) * DTIME * EMSD, QIPA)
                QRn = _where(do & inLT, QRPA + (QRFXIN - QRFXOUT + RAINFB) * DTIME * EMSD, QRPA)
                QSn = _where(do & inLT, QSPA + (QSFXIN - QSFXOUT + SNOWFB) * DTIME * EMSD, QSPA)
                return (QLn, QIn, QRn, QSn)
            QLG, QIG, QRG, QSG = jax.lax.fori_loop(0, NSTEP, hyd_substep, (QLPA, QIPA, QRPA, QSPA))
            PRATEC = PPTFLX * (1.0 - fbfrc) / DXSQ
            RAINCV = dt * PRATEC
            QL0 = jnp.zeros(N, dtype=jnp.float32)
            QI0 = jnp.zeros(N, dtype=jnp.float32)
            QR0 = jnp.zeros(N, dtype=jnp.float32)
            QS0 = jnp.zeros(N, dtype=jnp.float32)
            tend_time = _where(shallow, jnp.float32(2400.0), TIMEC)
            CPM = CP * (1.0 + 0.887 * QG)
            if warm_rain:
                TG = TG - (QIG + QSG) * _RLF / CPM
                DQCDT = _where(inLT, (QLG + QIG - QL0 - QI0) / tend_time, 0.0)
                DQRDT = _where(inLT, (QRG + QSG - QR0 - QS0) / tend_time, 0.0)
                DQIDT = jnp.zeros(N, dtype=jnp.float32)
                DQSDT = jnp.zeros(N, dtype=jnp.float32)
            elif not f_qs:
                latent = _where(idx <= ML, -(QIG + QSG), QLG + QRG) * _RLF / CPM
                TG = TG + latent
                DQCDT = _where(inLT, (QLG + QIG - QL0 - QI0) / tend_time, 0.0)
                DQRDT = _where(inLT, (QRG + QSG - QR0 - QS0) / tend_time, 0.0)
                DQIDT = jnp.zeros(N, dtype=jnp.float32)
                DQSDT = jnp.zeros(N, dtype=jnp.float32)
            else:
                DQCDT = _where(inLT, (QLG - QL0) / tend_time, 0.0)
                DQRDT = _where(inLT, (QRG - QR0) / tend_time, 0.0)
                DQSDT = _where(inLT, (QSG - QS0) / tend_time, 0.0)
                if f_qi:
                    DQIDT = _where(inLT, (QIG - QI0) / tend_time, 0.0)
                else:
                    DQSDT = DQSDT + _where(inLT, (QIG - QI0) / tend_time, 0.0)
                    DQIDT = jnp.zeros(N, dtype=jnp.float32)
            DTDT = _where(inLT, (TG - T0p) / tend_time, 0.0)
            DQDT = _where(inLT, (QG - Q0) / tend_time, 0.0)
            NICf = _where(TADVEC < TIMEC, _nint_positive(TADVEC / dt).astype(jnp.int32), NIC)
            NCA = _where(shallow, jnp.float32(0.0), NICf.astype(jnp.float32) * dt)
            TIMEC_out = TIMEC  # WRF saves TIMEC_KF before the shallow feedback reset.
            pii = (P0p / 100000.0) ** (R_D / CP)

            def slice0(a):
                return a
            out = dict(DTDT=slice0(DTDT), DQDT=slice0(DQDT), DQCDT=slice0(DQCDT), DQRDT=slice0(DQRDT), DQIDT=slice0(DQIDT), DQSDT=slice0(DQSDT), RTHCUTEN=slice0(DTDT / pii), RQVCUTEN=slice0(DQDT), RQCCUTEN=slice0(DQCDT), RQRCUTEN=slice0(DQRDT), RQICUTEN=slice0(DQIDT), RQSCUTEN=slice0(DQSDT), RAINCV=RAINCV, PRATEC=PRATEC, NCA=NCA, CUTOP=LTOP.astype(jnp.float32), CUBOT=LCL.astype(jnp.float32), ISHALL=_where(shallow, jnp.int32(1), jnp.int32(0)), TIMEC=TIMEC_out)
            return out
        return jax.lax.cond(carry['rejected'],
                            lambda _: _empty_col(N, -100.0, KX), finish, None)
    return kf_eta_para(T0, QV0, P0, DZQ, RHOE, W0A, U0, V0, dt, dx, KX, warm_rain, f_qi, f_qs)

# Source expressions whose right-hand side contains other constants.
ALIQ = SVP1 * 1000.0
BLIQ = SVP2
CLIQ = SVP2 * SVPT0
DLIQ = SVP3
GDRY = -G / CP

# WRF module_cu_kfeta.F:720/730: the operands and results are REAL.
# Retain the old Python/fp64 values above for the default-OFF path.
_ALIQ_REAL = np.float32(np.float32(SVP1) * np.float32(1000.0))
_GDRY_REAL = np.float32(-np.float32(G) / np.float32(CP))


def _runtime_constants():
    """Native-column constants; trace-time flag is covered by the E80 env key."""
    if os.environ.get("GPUWRF_KF_REAL_CONSTANTS", "0") == "1":
        return _ALIQ_REAL, _GDRY_REAL
    return ALIQ, GDRY

_PROFILE_FIELDS = ('DTDT', 'DQDT', 'DQCDT', 'DQRDT', 'DQIDT', 'DQSDT',
                   'RTHCUTEN', 'RQVCUTEN', 'RQCCUTEN', 'RQRCUTEN', 'RQICUTEN', 'RQSCUTEN')
_SCALAR_FIELDS = ('RAINCV', 'PRATEC', 'NCA', 'CUTOP', 'CUBOT', 'ISHALL', 'TIMEC')


def kf_column(T0, QV0, P0, DZQ, RHOE, W0A, U0, V0, dt, dx,
              KX, warm_rain=False, f_qi=True, f_qs=True, *, interpret=False):
    """Single column interface, vmappable by the retained column adapter."""
    size = 1 << (KX + 2).bit_length()
    inputs = tuple(jnp.pad(jnp.asarray(a, jnp.float32), (1, size-KX-1))
                   for a in (T0, QV0, P0, DZQ, RHOE, W0A, U0, V0))
    controls = jnp.asarray([dt, dx, 0.0], jnp.float32)

    def kernel(*refs):
        profiles = [ref[...] for ref in refs[:8]]
        ctrl, the0, ttab, qstab, alu = refs[8:13]
        with jax.enable_x64(False):
            out = _column(*profiles, ctrl[0], ctrl[1], KX, the0, ttab, qstab, alu,
                          warm_rain, f_qi, f_qs, ctrl[2])
        for name, ref in zip(_PROFILE_FIELDS + _SCALAR_FIELDS, refs[13:]):
            ref[...] = out[name]

    shapes = tuple(jax.ShapeDtypeStruct((size,), jnp.float32) for _ in _PROFILE_FIELDS)
    shapes += tuple(jax.ShapeDtypeStruct((), jnp.int32 if key=='ISHALL' else jnp.float32)
                    for key in _SCALAR_FIELDS)
    out = pl.pallas_call(kernel, out_shape=shapes, grid=(), interpret=interpret,
                         compiler_params=plt.CompilerParams(num_warps=4),
                         name='kf_column_fp32')(
        *inputs, controls, jnp.asarray(tables.THE0K_R4), jnp.asarray(tables.TTAB_R4),
        jnp.asarray(tables.QSTAB_R4), jnp.asarray(tables.ALU, jnp.float32))
    fields = {key: value[1:KX+1] for key,value in zip(_PROFILE_FIELDS,out[:12])}
    fields.update(zip(_SCALAR_FIELDS,out[12:]))
    # module_cu_kfeta.F is all REAL: the native path keeps its outputs REAL
    # (BP55; the KF carry/adapter around it is REAL under the same flag).
    return {key: value if key=='ISHALL' else value.astype(jnp.float32)
            for key,value in fields.items()}
