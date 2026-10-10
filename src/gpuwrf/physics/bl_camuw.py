"""CAM-UW (Bretherton-Park UW moist turbulence) PBL, ``bl_pbl_physics=9``.

Faithful JAX port of WRF's CAM5 vertical diffusion wrapper
``phys/module_bl_camuwpbl_driver.F:camuwpbl`` (pristine WRF v4, SHA in
``proofs/v034/camuw_oracle/camuw_build_manifest.txt``) and the CAM modules it calls:

* ``module_cam_bl_eddy_diff.F``: ``compute_eddy_diff`` (nturb=5 iterations of
  ``trbintd`` -> ``caleddy`` (``exacol``/``zisocl`` CL identification + merging, SRCL,
  entrainment closure with wstar, STL) -> relaxation -> provisional ``compute_vdiff`` ->
  saturation adjustment);
* ``module_cam_bl_diffusion_solver.F``: ``compute_vdiff`` (implicit LU solver, implicit
  surface stress with residual stress ``tauresx/y``, KE-dissipation heating ``dtk``);
* ``module_cam_wv_saturation.F``: ``vqsatd``/``estblf`` (table from ``gestbl``/``gffgch``,
  computed host-side bit-identically to WRF, see ``_estbl_table``).

WRF runs this scheme in ``real(r8)`` (CAM ``shr_kind_r8``) internally: it is a WRF DOUBLE
island, so the column math is float64 here as well. WRF REAL inputs/outputs and the REAL
expressions inside the driver (``p8w(k)-p8w(k+1)``, ``z-ht``, ``rthratenlw*exner``,
``sqrt(u*u+v*v)``, ``rho*ust*ust``) are evaluated in float32 exactly like WRF; carried
``kvm3d/kvh3d/tauresx2d/tauresy2d`` are rounded to REAL between steps as WRF stores them.

Configuration fixed by WRF's ``vd_register``/``camuwpblinit``: eddy_scheme='diag_TKE',
do_tms=.false., do_molec_diff=.false., do_pseudocon_diff=.false., pcnst=5 (Q, CLDLIQ,
CLDICE, NUMLIQ, NUMICE, all diffused; not CAMMGMP), surface flux only for vapour.

Known WRF behaviours reproduced on purpose: ``cldfra`` (not ``cldfra_old_mp``) without
CAMMGMP; liquid number is diffused but WRF returns no RQNCBLTEN; the explicit surface stress
is ``-rho*ust^2*u/|u|`` (BareGround form, NaN in WRF if the lowest-level wind is exactly 0 --
guarded here only by a floor that is inactive for any |u| > 1e-30).
Divergence (documented): WRF's zisocl upward-extension loop can spin forever when a CL
reaches the model top (outer ``do while`` without the ``kt-1 > ntop_turb`` guard); the port
terminates that loop instead.
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

from gpuwrf._x64_config import configure_jax_x64

import jax
import jax.numpy as jnp
from jax import lax

configure_jax_x64()

F64 = jnp.float64
F32 = jnp.float32

# --------------------------------------------------------------------------------------------
# CAM physconst / shr_const (module_cam_shr_const_mod.F, module_cam_physconst.F), r8
# --------------------------------------------------------------------------------------------
_AVOGAD = 6.02214e26
_BOLTZ = 1.38065e-23
_MWDAIR = 28.966
_MWWV = 18.016
_RGAS = _AVOGAD * _BOLTZ
CPAIR = 1.00464e3
GRAVIT = 9.80616
RAIR = _RGAS / _MWDAIR
RH2O = _RGAS / _MWWV
ZVIR = (RH2O / RAIR) - 1.0
LATVAP = 2.501e6
LATICE = 3.337e5
LATSUB = LATVAP + LATICE
KARMAN = 0.4
EPSILO = _MWWV / _MWDAIR
TMELT = 273.15

# eddy_diff module parameters (module_cam_bl_eddy_diff.F:60-170)
NTURB = 5
CTUNL = 2.0
CLENG = 3.0
A1I = 0.2
CCRIT = 0.5
WSTAR3FACTCRIT = 0.5
A2L = 30.0
A3L = 0.8
JBUMIN = 0.001
EVHCMAX = 10.0
USTAR_MIN = 0.01
ONET = 1.0 / 3.0
QMIN_CLD = 1.0e-5
NTZERO = 1.0e-12
B1 = 5.8
TUNL = 0.085
ALPH1 = 0.5562
ALPH2 = -4.3640
ALPH3 = -34.6764
ALPH4 = -6.1272
ALPH5 = 0.6986
RICRIT = 0.19
AE = 1.0
RINC = -0.04
WPERTMIN = 1.0e-6
WFAC = 1.0
TFAC = 1.0
FAK = 8.5
RCAPMIN = 0.1
RCAPMAX = 2.0
TKEMAX = 20.0
LAMBDA = 0.5
GHMIN = -3.5334  # ricrit == 0.19 branch of caleddy
ALPH4EXS = ALPH4
# diffusion_solver
WSMIN = 1.0
KSRFMIN = 1.0e-4
TIMERES = 7200.0

# wv_saturation / esinti (tmn 173.16, tmx 375.16, trice 20, ip .true.)
_TMIN = 173.16
_TMAX = 375.16
_TTRICE = 20.0
_PCF = (5.04469588506e-01, -5.47288442819e00, -3.67471858735e-01, -8.95963532403e-03,
        -7.78053686625e-05)


def _gffgch(t: float, itype: int) -> float:
    """module_cam_gffgch.F:gffgch (Goff-Gratch), host float64 with glibc libm (== WRF)."""

    if itype < 0:
        tr = abs(float(itype))
        itype = 1
    else:
        tr = 0.0
    eswtr = 0.0
    if not (t < (TMELT - tr) and itype == 1):
        ps = 1013.246
        ts = 373.16
        e1 = 11.344 * (1.0 - t / ts)
        e2 = -3.49149 * (ts / t - 1.0)
        f1 = -7.90298 * (ts / t - 1.0)
        f2 = 5.02808 * math.log10(ts / t)
        f3 = -1.3816 * (math.pow(10.0, e1) - 1.0) / 10000000.0
        f4 = 8.1328 * (math.pow(10.0, e2) - 1.0) / 1000.0
        f5 = math.log10(ps)
        f = f1 + f2 + f3 + f4 + f5
        es = (math.pow(10.0, f)) * 100.0
        eswtr = es
        if t >= TMELT or itype == 0:
            return es
    t0 = TMELT
    term1 = 2.01889049 / (t0 / t)
    term2 = 3.56654 * math.log(t0 / t)
    term3 = 20.947031 * (t0 / t)
    es = 575.185606e10 * math.exp(-(term1 + term2 + term3))
    if t < (TMELT - tr):
        return es
    weight = min((TMELT - t) / tr, 1.0)
    return weight * es + (1.0 - weight) * eswtr


def _estbl_table() -> tuple[float, ...]:
    """gestbl table (plenest=250): entries beyond lentbl are -99999 like WRF."""

    lentbl = int(_TMAX - _TMIN + 2.000001)
    tab = []
    t = _TMIN - 1.0
    for _ in range(lentbl):
        t = t + 1.0
        tab.append(_gffgch(t, -int(_TTRICE)))
    tab.extend([-99999.0] * (250 - lentbl))
    return tuple(tab)


ESTBL = _estbl_table()


def _estblf(td):
    e = jnp.maximum(jnp.minimum(td, _TMAX), _TMIN)
    ai = jnp.floor(e - _TMIN)  # e >= tmin -> int()/aint() == floor
    i = ai.astype(jnp.int32)  # 0-based index of estbl(i+1)
    tab = jnp.asarray(ESTBL, dtype=F64)
    return (_TMIN + ai - e + 1.0) * tab[i] - (_TMIN + ai - e) * tab[i + 1]


def _vqsatd(t, p):
    """wv_saturation.F:vqsatd (icephs, ttrice=20 branch) -> es, qs, gam."""

    omeps = 1.0 - EPSILO
    es = _estblf(t)
    qs = EPSILO * es / (p - omeps * es)
    qs = jnp.minimum(1.0, qs)
    neg = qs < 0.0
    qs = jnp.where(neg, 1.0, qs)
    es = jnp.where(neg, p, es)
    trinv = 1.0 / _TTRICE
    tc = t - TMELT
    lflg = (tc >= -_TTRICE) & (tc < 0.0)
    weight = jnp.minimum(-tc * trinv, 1.0)
    hlatsb = LATVAP + weight * LATICE
    hlatvp = LATVAP - 2369.0 * tc
    hltalt = jnp.where(t < TMELT, hlatsb, hlatvp)
    pcf = _PCF
    tterm = jnp.where(lflg, pcf[0] + tc * (pcf[1] + tc * (pcf[2] + tc * (pcf[3] + tc * pcf[4]))), 0.0)
    desdt = hltalt * es / (RH2O * t * t) + tterm * trinv
    gam = hltalt * qs * p * desdt / (CPAIR * es * (p - omeps * es))
    gam = jnp.where(qs == 1.0, 0.0, gam)
    return es, qs, gam


# --------------------------------------------------------------------------------------------
# small control-flow helpers (Fortran 1-based indices are kept: index 0 of every array is pad)
# --------------------------------------------------------------------------------------------


# Optional branch census (tests only): set to a dict BEFORE tracing a fresh, NON-vmapped jit of
# camuw_column; the product path never sets it, so no callback is traced.
_CENSUS = None


def _mark(name, pred=True):
    if _CENSUS is None:
        return
    sink = _CENSUS

    def cb(p):
        sink[name] = sink.get(name, 0) + int(p)

    jax.debug.callback(cb, jnp.asarray(pred, jnp.int32))


def _like(new, old):
    """Coerce a pytree to the dtypes of ``old`` (x64 python-int promotion guard)."""

    return jax.tree_util.tree_map(lambda a, b: jnp.asarray(a, jnp.asarray(b).dtype), new, old)


def _fori_up(lo, hi, body, carry):
    """Fortran ``do k = lo, hi`` (inclusive, ascending)."""

    lo = jnp.asarray(lo, jnp.int32)
    hi = jnp.asarray(hi, jnp.int32)
    return lax.fori_loop(lo, jnp.maximum(hi + 1, lo), lambda k, c: _like(body(k, c), c), carry)


def _fori_down(hi, lo, body, carry):
    """Fortran ``do k = hi, lo, -1`` (inclusive, descending)."""

    hi = jnp.asarray(hi, jnp.int32)
    n = jnp.maximum(hi - jnp.asarray(lo, jnp.int32) + 1, 0)
    return lax.fori_loop(jnp.int32(0), n, lambda j, c: _like(body(hi - j, c), c), carry)


def _when(pred, fn, carry):
    return lax.cond(pred, lambda c: _like(fn(c), c), lambda c: c, carry)


def _cond(pred, f_true, f_false, carry):
    return lax.cond(pred, lambda c: _like(f_true(c), c), lambda c: _like(f_false(c), c), carry)


def _while(cond, body, carry):
    return lax.while_loop(cond, lambda c: _like(body(c), c), carry)


def _cubic_root_length(zi_k, lbulk, tunlramp):
    """leng = ((vk*zi)**(-cleng) + (tunlramp*lbulk)**(-cleng))**(-1/cleng) ('origin')."""

    return (jnp.power(KARMAN * zi_k, -CLENG) + jnp.power(tunlramp * lbulk, -CLENG)) ** (-1.0 / CLENG)


def _compute_cubic(a, b, c):
    """eddy_diff:compute_cubic (largest real root, floor xmin=1e-2)."""

    xmin = 1.0e-2
    qq = (a * a - 3.0 * b) / 9.0
    rr = (2.0 * (a * a * a) - 9.0 * a * b + 27.0 * c) / 54.0
    dd = rr * rr - qq * qq * qq
    safe_q = jnp.where(dd <= 0.0, qq, 1.0)
    theta = jnp.arccos(jnp.where(dd <= 0.0, rr / jnp.power(safe_q, 1.5), 0.0))
    sq = jnp.sqrt(jnp.maximum(safe_q, 0.0))
    x1 = -2.0 * sq * jnp.cos(theta / 3.0) - a / 3.0
    x2 = -2.0 * sq * jnp.cos((theta + 2.0 * 3.141592) / 3.0) - a / 3.0
    x3 = -2.0 * sq * jnp.cos((theta - 2.0 * 3.141592) / 3.0) - a / 3.0
    r1 = jnp.maximum(jnp.maximum(jnp.maximum(x1, x2), x3), xmin)
    sdd = jnp.sqrt(jnp.where(dd > 0.0, rr * rr - qq * qq * qq, 0.0))
    aa = jnp.where(rr >= 0.0, -jnp.power(sdd + rr, 1.0 / 3.0), jnp.power(jnp.maximum(sdd - rr, 0.0), 1.0 / 3.0))
    bb = jnp.where(aa == 0.0, 0.0, qq / jnp.where(aa == 0.0, 1.0, aa))
    r2 = jnp.maximum((aa + bb) - a / 3.0, xmin)
    _mark("cubic_three_real_roots", dd <= 0.0)
    _mark("cubic_calls")
    return jnp.where(dd <= 0.0, r1, r2)


def _stab_funcs_cl(ricll):
    """zisocl/ricl -> (gh, sh, sm) with alph4 (unstable-capable) branch, no max(0,...)."""

    trma = ALPH3 * ALPH4 * ricll + 2.0 * B1 * (ALPH2 - ALPH4 * ALPH5 * ricll)
    trmb = ricll * (ALPH3 + ALPH4) + 2.0 * B1 * (-ALPH5 * ricll + ALPH1)
    trmc = ricll
    det = jnp.maximum(trmb * trmb - 4.0 * trma * trmc, 0.0)
    gh = (-trmb + jnp.sqrt(det)) / 2.0 / trma
    gh = jnp.minimum(jnp.maximum(gh, -3.5334), 0.0233)
    sh = ALPH5 / (1.0 + ALPH3 * gh)
    sm = (ALPH1 + ALPH2 * gh) / (1.0 + ALPH3 * gh) / (1.0 + ALPH4 * gh)
    return gh, sh, sm


def _stab_funcs_stl(ri):
    """caleddy STL/imsi Galperin functions (alph4exs, ghmin, max(0,...))."""

    trma = ALPH3 * ALPH4EXS * ri + 2.0 * B1 * (ALPH2 - ALPH4EXS * ALPH5 * ri)
    trmb = (ALPH3 + ALPH4EXS) * ri + 2.0 * B1 * (-ALPH5 * ri + ALPH1)
    trmc = ri
    det = jnp.maximum(trmb * trmb - 4.0 * trma * trmc, 0.0)
    gh = (-trmb + jnp.sqrt(det)) / (2.0 * trma)
    gh = jnp.minimum(jnp.maximum(gh, GHMIN), 0.0233)
    sh = jnp.maximum(0.0, ALPH5 / (1.0 + ALPH3 * gh))
    sm = jnp.maximum(0.0, (ALPH1 + ALPH2 * gh) / (1.0 + ALPH3 * gh) / (1.0 + ALPH4EXS * gh))
    return gh, sh, sm


def _surf_stab(z_pver, bprod_s, tkes):
    """gg/gh/sh/sm at the surface for bflxs > 0 (zisocl)."""

    gg = 0.5 * KARMAN * z_pver * bprod_s / jnp.power(tkes, 1.5)
    gh = gg / (ALPH5 - gg * ALPH3)
    gh = jnp.minimum(jnp.maximum(gh, -3.5334), 0.0233)
    sh = ALPH5 / (1.0 + ALPH3 * gh)
    sm = (ALPH1 + ALPH2 * gh) / (1.0 + ALPH3 * gh) / (1.0 + ALPH4 * gh)
    return gh, sh, sm


# --------------------------------------------------------------------------------------------
# trbintd + sfdiag ('l' with choice 'maxi': sfuh = sflh = cld)
# --------------------------------------------------------------------------------------------


def _trbintd(pver, z, u, v, t, pmid, taux, tauy, zi, pi, cld, qv, ql, qi):
    del zi, pi
    n = pver
    rrho = RAIR * t[n] / pmid[n]
    ustar = jnp.maximum(jnp.sqrt(jnp.sqrt(taux * taux + tauy * tauy) * rrho), USTAR_MIN)
    _, _, gam = _vqsatd(t, pmid)
    qt = qv + ql + qi
    sl = CPAIR * t + GRAVIT * z - LATVAP * ql - LATSUB * qi
    slv = sl * (1.0 + ZVIR * qt)
    tsafe = jnp.where(jnp.arange(n + 1) == 0, 1.0, t)
    bfact = GRAVIT / (tsafe * (1.0 + ZVIR * qv - ql - qi))
    chu = (1.0 + ZVIR * qt) * bfact / CPAIR
    chs = ((1.0 + (1.0 + ZVIR) * gam * CPAIR * t / LATVAP) / (1.0 + gam)) * bfact / CPAIR
    cmu = ZVIR * bfact * t
    cms = LATVAP * chs - bfact * t
    # interface arrays (size pver+2): copy pver -> pver+1
    chu = jnp.concatenate([chu, chu[n:n + 1]])
    chs = jnp.concatenate([chs, chs[n:n + 1]])
    cmu = jnp.concatenate([cmu, cmu[n:n + 1]])
    cms = jnp.concatenate([cms, cms[n:n + 1]])
    # monotone slopes
    d_sl = (sl[2:] - sl[1:-1]) / (pmid[2:] - pmid[1:-1])  # index j -> k=j+1 (1..pver-1)
    d_qt = (qt[2:] - qt[1:-1]) / (pmid[2:] - pmid[1:-1])

    def _lim(a, b):
        prod = a * b
        return jnp.where(prod <= 0.0, 0.0, jnp.where(a < 0.0, jnp.maximum(a, b), jnp.minimum(a, b)))

    slslope = jnp.zeros(n + 1, F64)
    qtslope = jnp.zeros(n + 1, F64)
    slslope = slslope.at[n].set(d_sl[n - 2]).at[1].set(d_sl[0])
    qtslope = qtslope.at[n].set(d_qt[n - 2]).at[1].set(d_qt[0])
    slslope = slslope.at[2:n].set(_lim(d_sl[0:n - 2], d_sl[1:n - 1]))
    qtslope = qtslope.at[2:n].set(_lim(d_qt[0:n - 2], d_qt[1:n - 1]))
    # sfdiag
    kk = jnp.arange(n + 1)
    sfuh = jnp.where(kk >= 2, cld, 0.0)
    sflh = sfuh
    sfi = jnp.zeros(n + 2, F64)
    sfi = sfi.at[2:n + 1].set(0.5 * (sflh[1:n] + jnp.minimum(sfuh[2:n + 1], sflh[1:n])))
    sfi = sfi.at[n + 1].set(sflh[n])
    # n2, s2, ri at interfaces 2..pver
    rdz = 1.0 / (z[1:n] - z[2:n + 1])
    dsldz = (sl[1:n] - sl[2:n + 1]) * rdz
    dqtdz = (qt[1:n] - qt[2:n + 1]) * rdz
    chu_i = (chu[1:n] + chu[2:n + 1]) * 0.5
    chs_i = (chs[1:n] + chs[2:n + 1]) * 0.5
    cmu_i = (cmu[1:n] + cmu[2:n + 1]) * 0.5
    cms_i = (cms[1:n] + cms[2:n + 1]) * 0.5
    chu = chu.at[2:n + 1].set(chu_i)
    chs = chs.at[2:n + 1].set(chs_i)
    cmu = cmu.at[2:n + 1].set(cmu_i)
    cms = cms.at[2:n + 1].set(cms_i)
    sfi_i = sfi[2:n + 1]
    ch = chu_i * (1.0 - sfi_i) + chs_i * sfi_i
    cm = cmu_i * (1.0 - sfi_i) + cms_i * sfi_i
    n2_i = ch * dsldz + cm * dqtdz
    du = u[1:n] - u[2:n + 1]
    dv = v[1:n] - v[2:n + 1]
    s2_i = (du * du + dv * dv) * (rdz * rdz)
    s2_i = jnp.maximum(NTZERO, s2_i)
    ri_i = n2_i / s2_i
    pad = jnp.zeros(2, F64)
    n2 = jnp.concatenate([pad, n2_i]).at[1].set(n2_i[0])
    s2 = jnp.concatenate([pad, s2_i]).at[1].set(s2_i[0])
    ri = jnp.concatenate([pad, ri_i]).at[1].set(ri_i[0])
    return dict(ustar=ustar, rrho=rrho, s2=s2, n2=n2, ri=ri, qt=qt, sl=sl, slv=slv,
                slslope=slslope, qtslope=qtslope, chs=chs, chu=chu, cms=cms, cmu=cmu,
                sfi=sfi, sfuh=sfuh, sflh=sflh)


# --------------------------------------------------------------------------------------------
# exacol
# --------------------------------------------------------------------------------------------


def _exacol(pver, ri, bflxs):
    n = pver
    ncvmax = n
    riex = jnp.zeros(n + 2, F64).at[2:n + 1].set(ri[2:n + 1]).at[n + 1].set(0.0 - bflxs)
    kbase = jnp.zeros(ncvmax + 3, jnp.int32)
    ktop = jnp.zeros(ncvmax + 3, jnp.int32)
    init = dict(k=jnp.int32(n + 1), ncv=jnp.int32(0), inside=jnp.bool_(False), kbase=kbase, ktop=ktop)

    def cond(c):
        return (c["k"] > 2) | c["inside"]

    def body(c):
        k = c["k"]
        neg = riex[k] < 0.0

        def outside(c):
            def start(c):
                ncv = c["ncv"] + 1
                return dict(c, ncv=ncv, inside=jnp.bool_(True),
                            kbase=c["kbase"].at[ncv].set(jnp.minimum(k + 1, n + 1)))
            return _cond(neg, start, lambda c: dict(c, k=c["k"] - 1), c)

        def inside(c):
            def step(c):
                return dict(c, k=c["k"] - 1)

            def close(c):
                return dict(c, inside=jnp.bool_(False), ktop=c["ktop"].at[c["ncv"]].set(k))
            return _cond(neg & (k > 2), step, close, c)

        return _cond(c["inside"], inside, outside, c)

    out = _while(cond, body, init)
    return out["kbase"], out["ktop"], out["ncv"]


# --------------------------------------------------------------------------------------------
# zisocl
# --------------------------------------------------------------------------------------------


def _zisocl(pver, z, zi, n2, s2, bprod, sprod, bflxs, tkes, ncvfin, kbase, ktop):
    n = pver
    ncvmax = n
    tunl_cl = 0.5 * (1.0 + CTUNL) * TUNL
    clz = dict(ricl=jnp.zeros(ncvmax + 3, F64), ghcl=jnp.zeros(ncvmax + 3, F64),
               shcl=jnp.zeros(ncvmax + 3, F64), smcl=jnp.zeros(ncvmax + 3, F64),
               lbrk=jnp.zeros(ncvmax + 3, F64), wbrk=jnp.zeros(ncvmax + 3, F64),
               ebrk=jnp.zeros(ncvmax + 3, F64))
    sqtk = jnp.sqrt(tkes)

    def lz_at(k, lbulk):
        return _cubic_root_length(zi[k], lbulk, tunl_cl)

    def incr(k, lbulk, sh, sm):
        lz = lz_at(k, lbulk)
        dzinc = z[k - 1] - z[k]
        dl2n2 = lz * lz * n2[k] * dzinc
        dl2s2 = lz * lz * s2[k] * dzinc
        dwinc = -sh * dl2n2 + sm * dl2s2
        return dzinc, dl2n2, dl2s2, dwinc

    def accum(c, dzinc, dl2n2, dl2s2, dwinc):
        lint = c["lint"] + dzinc
        l2n2 = c["l2n2"] + dl2n2
        l2n2 = -jnp.minimum(-l2n2, TKEMAX * lint / (B1 * c["sh"]))
        return dict(c, lint=lint, l2n2=l2n2, l2s2=c["l2s2"] + dl2s2, wint=c["wint"] + dwinc)

    def surf_terms(sh_dummy=None):
        gh, sh, sm = _surf_stab(z[n], bprod[n + 1], tkes)
        zp = z[n]
        dl2n2_surf = -KARMAN * (zp * zp) * bprod[n + 1] / (sh * sqtk)
        dl2s2_surf = KARMAN * (zp * zp) * sprod[n + 1] / (sm * sqtk)
        dw_surf = (tkes / B1) * zp
        return gh, sh, sm, zp, dl2n2_surf, dl2s2_surf, dw_surf

    def cl_body(c):
        ncv = c["ncv"]
        kb = c["kbase"][ncv]
        kt = c["ktop"][ncv]
        lbulk = zi[kt] - zi[kb]
        surf_pos = (kb == n + 1) & (bflxs > 0.0)
        gh_s, sh_s, sm_s, zp, dl2n2_s, dl2s2_s, dw_s = surf_terms()
        ricll_s = jnp.minimum(-(sm_s / sh_s) * (bprod[n + 1] / sprod[n + 1]), RICRIT)
        dlint_surf = jnp.where(surf_pos, zp, 0.0)
        dl2n2_surf = jnp.where(surf_pos, dl2n2_s, 0.0)
        dl2s2_surf = jnp.where(surf_pos, dl2s2_s, 0.0)
        dw_surf = jnp.where(surf_pos, dw_s, 0.0)
        gh = jnp.where(surf_pos, gh_s, c["gh"])
        sh = jnp.where(surf_pos, sh_s, c["sh"])
        sm = jnp.where(surf_pos, sm_s, c["sm"])
        ricll = jnp.where(surf_pos, ricll_s, c["ricll"])
        lbulk = jnp.where((kb == n + 1) & (bflxs <= 0.0), zi[kt] - z[n], lbulk)
        c = dict(c, ncvinit=ncv, cntu=jnp.int32(0), cntd=jnp.int32(0), kb=kb, kt=kt, lbulk=lbulk,
                 lint=dlint_surf, l2n2=jnp.float64(0.0), l2s2=jnp.float64(0.0), wint=dw_surf,
                 gh=gh, sh=sh, sm=sm, ricll=ricll, extend=jnp.bool_(False))

        def interior(c):
            def acc(k, cc):
                lz = lz_at(k, cc["lbulk"])
                dzinc = z[k - 1] - z[k]
                return dict(cc, l2n2=cc["l2n2"] + lz * lz * n2[k] * dzinc,
                            l2s2=cc["l2s2"] + lz * lz * s2[k] * dzinc, lint=cc["lint"] + dzinc)
            cc = _fori_down(c["kb"] - 1, c["kt"] + 1, acc, c)
            ricll = jnp.minimum(cc["l2n2"] / jnp.maximum(cc["l2s2"], NTZERO), RICRIT)
            gh, sh, sm = _stab_funcs_cl(ricll)
            return dict(cc, ricll=ricll, gh=gh, sh=sh, sm=sm, wint=cc["wint"] - sh * cc["l2n2"] + sm * cc["l2s2"])

        def single(c):
            return dict(c, lint=dlint_surf, l2n2=dl2n2_surf, l2s2=dl2s2_surf, wint=dw_surf)

        c = _cond(c["kt"] < c["kb"] - 1, interior, single, c)
        c = dict(c, l2n2=-jnp.minimum(-c["l2n2"], TKEMAX * c["lint"] / (B1 * c["sh"])),
                 l2s2=jnp.minimum(c["l2s2"], TKEMAX * c["lint"] / (B1 * c["sm"])))
        dzinc, dl2n2, dl2s2, dwinc = incr(c["kt"], c["lbulk"], c["sh"], c["sm"])
        c = dict(c, dzinc=dzinc, dl2n2=dl2n2, dl2s2=dl2s2, dwinc=dwinc)

        # ---- upward extension ----
        def up_cond(c):
            return (-c["dl2n2"] > (-RINC * c["l2n2"] / (1.0 - RINC))) & (c["kt"] - 1 > 1)

        def up_body(c):
            c = accum(c, c["dzinc"], c["dl2n2"], c["dl2s2"], c["dwinc"])
            kt = c["kt"] - 1
            c = dict(c, kt=kt, extend=jnp.bool_(True))
            idx = c["ncv"] + c["cntu"] + 1
            ktinc = c["kbase"][idx] - 1

            def merge_up(c):
                def acc(k, cc):
                    dz, d2n, d2s, dw = incr(k, cc["lbulk"], cc["sh"], cc["sm"])
                    return accum(cc, dz, d2n, d2s, dw)
                cc = _fori_down(c["kbase"][idx] - 1, c["ktop"][idx] + 1, acc, c)
                _mark("zisocl_merge_up")
                return dict(cc, kt=c["ktop"][idx], ncvfin=cc["ncvfin"] - 1, cntu=cc["cntu"] + 1)

            c = _when(kt == ktinc, merge_up, c)
            dzinc, dl2n2, dl2s2, dwinc = incr(c["kt"], c["lbulk"], c["sh"], c["sm"])
            return dict(c, dzinc=dzinc, dl2n2=dl2n2, dl2s2=dl2s2, dwinc=dwinc)

        c = _while(up_cond, up_body, c)

        def compact_up(c):
            def mv(incv, cc):
                src = cc["ncv"] + cc["cntu"] + incv
                dst = cc["ncv"] + incv
                return dict(cc, kbase=cc["kbase"].at[dst].set(cc["kbase"][src]),
                            ktop=cc["ktop"].at[dst].set(cc["ktop"][src]))
            return _fori_up(1, c["ncvfin"] - c["ncv"], mv, c)

        c = _when(c["cntu"] > 0, compact_up, c)

        # ---- downward extension ----
        def down_all(c):
            dzinc, dl2n2, dl2s2, dwinc = incr(c["kb"], c["lbulk"], c["sh"], c["sm"])
            c = dict(c, dzinc=dzinc, dl2n2=dl2n2, dl2s2=dl2s2, dwinc=dwinc)

            def dn_cond(c):
                return (-c["dl2n2"] > (-RINC * c["l2n2"] / (1.0 - RINC))) & (c["kb"] != n + 1)

            def dn_body(c):
                c = accum(c, c["dzinc"], c["dl2n2"], c["dl2s2"], c["dwinc"])
                kb = c["kb"] + 1
                c = dict(c, kb=kb, extend=jnp.bool_(True))
                prev = jnp.maximum(c["ncv"] - 1, 0)
                kbinc = jnp.where(c["ncv"] > 1, c["ktop"][prev] + 1, 0)

                def merge_dn(c):
                    def acc(k, cc):
                        dz, d2n, d2s, dw = incr(k, cc["lbulk"], cc["sh"], cc["sm"])
                        return accum(cc, dz, d2n, d2s, dw)
                    cc = _fori_up(c["ktop"][prev] + 1, c["kbase"][prev] - 1, acc, c)
                    _mark("zisocl_merge_down")
                    return dict(cc, kb=c["kbase"][prev], ncv=cc["ncv"] - 1, ncvfin=cc["ncvfin"] - 1,
                                cntd=cc["cntd"] + 1)

                c = _when(kb == kbinc, merge_dn, c)

                def at_surface(c):
                    pos = bflxs > 0.0
                    _, _, _, zp_, d2n_s, d2s_s, dw_s_ = surf_terms()
                    dlint = jnp.where(pos, zp_, 0.0)
                    d2n = jnp.where(pos, d2n_s, 0.0)
                    d2s = jnp.where(pos, d2s_s, 0.0)
                    dw = jnp.where(pos, dw_s_, 0.0)
                    return accum(c, dlint, d2n, d2s, dw)

                def above(c):
                    dzinc, dl2n2, dl2s2, dwinc = incr(c["kb"], c["lbulk"], c["sh"], c["sm"])
                    return dict(c, dzinc=dzinc, dl2n2=dl2n2, dl2s2=dl2s2, dwinc=dwinc)

                return _cond(c["kb"] == n + 1, at_surface, above, c)

            return _while(dn_cond, dn_body, c)

        c = _when(c["kb"] != n + 1, down_all, c)

        def compact_dn(c):
            def mv(incv, cc):
                src = cc["ncvinit"] + incv
                dst = cc["ncv"] + incv
                return dict(cc, kbase=cc["kbase"].at[dst].set(cc["kbase"][src]),
                            ktop=cc["ktop"].at[dst].set(cc["ktop"][src]))
            return _fori_up(1, c["ncvfin"] - c["ncv"], mv, c)

        c = _when(c["cntd"] > 0, compact_dn, c)
        c = dict(c, wint=jnp.where(c["wint"] < 0.01, 0.01, c["wint"]))

        def redo(c):
            ncv = c["ncv"]
            kt = c["kt"]
            kb = c["kb"]
            c = dict(c, ktop=c["ktop"].at[ncv].set(kt), kbase=c["kbase"].at[ncv].set(kb))
            lbulk = zi[kt] - zi[kb]
            surf_pos = (kb == n + 1) & (bflxs > 0.0)
            gh_s, sh_s, sm_s, zp_, _, _, dw_s_ = surf_terms()
            lbulk = jnp.where((kb == n + 1) & (bflxs <= 0.0), zi[kt] - z[n], lbulk)
            lint0 = jnp.where(surf_pos, zp_, 0.0)
            wint0 = jnp.where(surf_pos, dw_s_, 0.0)
            c = dict(c, lbulk=lbulk, lint=lint0, l2n2=jnp.float64(0.0), l2s2=jnp.float64(0.0), wint=wint0,
                     gh=jnp.where(surf_pos, gh_s, c["gh"]), sh=jnp.where(surf_pos, sh_s, c["sh"]),
                     sm=jnp.where(surf_pos, sm_s, c["sm"]))

            def acc(k, cc):
                lz = lz_at(k, cc["lbulk"])
                dzinc = z[k - 1] - z[k]
                return dict(cc, lint=cc["lint"] + dzinc, l2n2=cc["l2n2"] + lz * lz * n2[k] * dzinc,
                            l2s2=cc["l2s2"] + lz * lz * s2[k] * dzinc)
            c = _fori_up(kt + 1, kb - 1, acc, c)
            ricll = jnp.minimum(c["l2n2"] / jnp.maximum(c["l2s2"], NTZERO), RICRIT)
            gh, sh, sm = _stab_funcs_cl(ricll)
            wint = jnp.maximum(c["wint"] - sh * c["l2n2"] + sm * c["l2s2"], 0.01)
            return dict(c, ricll=ricll, gh=gh, sh=sh, sm=sm, wint=wint)

        _mark("zisocl_extend", c["extend"])
        c = _when(c["extend"], redo, c)
        ncv = c["ncv"]
        ebrk = jnp.minimum(B1 * (c["wint"] / c["lint"]), TKEMAX)
        c = dict(c, lbrk=c["lbrk"].at[ncv].set(c["lint"]), wbrk=c["wbrk"].at[ncv].set(c["wint"] / c["lint"]),
                 ebrk=c["ebrk"].at[ncv].set(ebrk), ricl=c["ricl"].at[ncv].set(c["ricll"]),
                 ghcl=c["ghcl"].at[ncv].set(c["gh"]), shcl=c["shcl"].at[ncv].set(c["sh"]),
                 smcl=c["smcl"].at[ncv].set(c["sm"]), ncv=ncv + 1)
        return c

    zero = jnp.float64(0.0)
    init = dict(clz, ncv=jnp.int32(1), ncvfin=ncvfin, kbase=kbase, ktop=ktop, ncvinit=jnp.int32(1),
                cntu=jnp.int32(0), cntd=jnp.int32(0), kb=jnp.int32(0), kt=jnp.int32(0), lbulk=zero,
                lint=zero, l2n2=zero, l2s2=zero, wint=zero, gh=zero, sh=zero, sm=zero, ricll=zero,
                extend=jnp.bool_(False), dzinc=zero, dl2n2=zero, dl2s2=zero, dwinc=zero)
    c = _while(lambda c: c["ncv"] <= c["ncvfin"], cl_body, init)
    idx = jnp.arange(ncvmax + 3)
    live = (idx >= 1) & (idx <= c["ncvfin"])
    kbase = jnp.where(live, c["kbase"], 0)
    ktop = jnp.where(live, c["ktop"], 0)
    kk = jnp.arange(n + 2)
    # belongcv(k) = any CL with ktop <= k <= kbase
    belong = jnp.any(live[:, None] & (kk[None, :] >= ktop[:, None]) & (kk[None, :] <= kbase[:, None]), axis=0)
    return dict(kbase=kbase, ktop=ktop, ncvfin=c["ncvfin"], belongcv=belong,
                ricl=c["ricl"], ghcl=c["ghcl"], shcl=c["shcl"], smcl=c["smcl"],
                lbrk=c["lbrk"], wbrk=c["wbrk"], ebrk=c["ebrk"])


# --------------------------------------------------------------------------------------------
# caleddy
# --------------------------------------------------------------------------------------------


def _caleddy(pver, sl, qt, ql, slv, u, v, pi, z, zi, qflx, shflx, slslope, qtslope, chu, chs, cmu,
             cms, sfuh, sflh, n2, s2, ri, rrho, ustar, kvh_in, kvm_in, qrlw):
    n = pver
    ncvmax = n
    ni = n + 2  # interface array length (index 0 pad, 1..pver+1)
    kk = jnp.arange(ni)
    inner = (kk >= 2) & (kk <= n)
    bprod = jnp.where(inner, -kvh_in * n2_pad(n2, ni), 0.0)
    sprod = jnp.where(inner, kvm_in * n2_pad(s2, ni), 0.0)
    ch = chu[n + 1] * (1.0 - sflh[n]) + chs[n + 1] * sflh[n]
    cm = cmu[n + 1] * (1.0 - sflh[n]) + cms[n + 1] * sflh[n]
    bflxs = ch * shflx * rrho + cm * qflx * rrho
    bprod = bprod.at[n + 1].set(bflxs)
    sprod = sprod.at[n + 1].set((ustar * ustar * ustar) / (KARMAN * z[n]))
    kbase, ktop, ncvfin = _exacol(n, ri, bflxs)
    tkes = jnp.power(jnp.maximum(B1 * KARMAN * z[n] * (bprod[n + 1] + sprod[n + 1]), 1.0e-7), 2.0 / 3.0)
    tkes = jnp.minimum(tkes, TKEMAX)
    tke = jnp.zeros(ni, F64).at[n + 1].set(tkes)
    wcap = jnp.zeros(ni, F64).at[n + 1].set(tkes / B1)
    zc = _zisocl(n, z, zi, n2, s2, bprod, sprod, bflxs, tkes, ncvfin, kbase, ktop)
    has_cl = ncvfin > 0
    zclz = jnp.zeros(ncvmax + 3, F64)
    kbase = jnp.where(has_cl, zc["kbase"], kbase)
    ktop = jnp.where(has_cl, zc["ktop"], ktop)
    ncvfin = jnp.where(has_cl, zc["ncvfin"], ncvfin)
    belongcv = jnp.where(has_cl, zc["belongcv"], jnp.zeros(ni, bool))
    cl = {key: jnp.where(has_cl, zc[key], zclz) for key in ("ricl", "ghcl", "shcl", "smcl", "lbrk", "wbrk", "ebrk")}
    ncvsurf = jnp.where(has_cl & (kbase[1] == n + 1), 1, 0).astype(jnp.int32)

    # ---------------- SRCL (choice 'nonamb') ----------------
    st = dict(cl, kbase=kbase, ktop=ktop, ncvfin=ncvfin, belongcv=belongcv, ncv=jnp.int32(1),
              ncvsurf=ncvsurf)
    ncvf = ncvfin

    def srcl(k, st):
        cand = (ql[k] > QMIN_CLD) & (ql[k - 1] < QMIN_CLD) & (qrlw[k] < 0.0) & (ri[k] >= RICRIT)
        cand = cand & ~st["belongcv"][k + 1]
        chk = (1.0 - sfuh[k]) * chu[k] + sfuh[k] * chs[k]
        cmk = (1.0 - sfuh[k]) * cmu[k] + sfuh[k] * cms[k]
        n2ht_srcl = chk * slslope[k] + cmk * qtslope[k]
        cand = cand & (n2ht_srcl <= 0.0)

        def do(st):
            def wc(s):
                return (s["ncv"] <= ncvf) & ~s["found"]

            def wb(s):
                hit = s["ktop"][s["ncv"]] <= k
                return dict(s, found=hit, in_cl=jnp.where(hit, s["kbase"][s["ncv"]] > k, s["in_cl"]),
                            ncv=jnp.where(hit, s["ncv"], s["ncv"] + 1))

            s = _while(wc, wb, dict(st, found=jnp.bool_(False), in_cl=jnp.bool_(False)))
            in_cl = s["in_cl"]
            s = {key: val for key, val in s.items() if key not in ("found", "in_cl")}

            def add(s):
                _mark("srcl_added")
                _mark("srcl_surface", k >= n)
                ncvnew = s["ncvfin"] + 1
                s = dict(s, ncvfin=ncvnew, ktop=s["ktop"].at[ncvnew].set(k),
                         kbase=s["kbase"].at[ncvnew].set(k + 1),
                         belongcv=s["belongcv"].at[k].set(True).at[k + 1].set(True))
                surf = k >= n
                gg = 0.5 * KARMAN * z[n] * bprod[n + 1] / jnp.power(tkes, 1.5)
                gh = jnp.where(jnp.abs(ALPH5 - gg * ALPH3) <= 1.0e-7, GHMIN, gg / (ALPH5 - gg * ALPH3))
                gh = jnp.minimum(jnp.maximum(gh, GHMIN), 0.0233)
                shc = jnp.maximum(0.0, ALPH5 / (1.0 + ALPH3 * gh))
                smc = jnp.maximum(0.0, (ALPH1 + ALPH2 * gh) / (1.0 + ALPH3 * gh) / (1.0 + ALPH4EXS * gh))
                ricl_s = -(smc / shc) * (bprod[n + 1] / sprod[n + 1])
                # bflxs > 0 for a surface SRCL is a WRF 'stop' (Major mistake in SRCL): not reachable
                upd = {"wbrk": 0.0, "ebrk": 0.0, "lbrk": 0.0}
                for key in upd:
                    s[key] = s[key].at[ncvnew].set(0.0)
                s["ghcl"] = s["ghcl"].at[ncvnew].set(jnp.where(surf, gh, 0.0))
                s["shcl"] = s["shcl"].at[ncvnew].set(jnp.where(surf, shc, 0.0))
                s["smcl"] = s["smcl"].at[ncvnew].set(jnp.where(surf, smc, 0.0))
                s["ricl"] = s["ricl"].at[ncvnew].set(jnp.where(surf, ricl_s, 0.0))
                s["ncvsurf"] = jnp.where(surf, ncvnew, s["ncvsurf"])
                return s

            return _when(~in_cl, add, s)

        return _when(cand, do, st)

    st = _fori_down(n, 2, srcl, st)
    kbase, ktop, ncvfin, belongcv = st["kbase"], st["ktop"], st["ncvfin"], st["belongcv"]
    ncvsurf = st["ncvsurf"]

    # ---------------- per-CL closure ----------------
    kvh = jnp.zeros(ni, F64)
    kvm = jnp.zeros(ni, F64)
    leng = jnp.zeros(ni, F64)
    turbtype = jnp.zeros(ni, jnp.int32)
    sm_aw = jnp.zeros(ni, F64)
    zero = jnp.float64(0.0)
    cs = dict(st, kvh=kvh, kvm=kvm, leng=leng, tke=tke, wcap=wcap, bprod=bprod, sprod=sprod,
              turbtype=turbtype, sm_aw=sm_aw, ktblw=jnp.int32(0), web=zero, ceb=zero, wstar=zero,
              wstar1=zero)

    def cl_loop(ncv, cs):
        kt = cs["ktop"][ncv]
        kb = cs["kbase"][ncv]
        kbs = jnp.minimum(kb, n + 1)  # safe index
        surf = kb == n + 1
        lbulk = jnp.where(surf & (bflxs <= 0.0), zi[kt] - z[n], zi[kt] - zi[kb])
        ricl_n = cs["ricl"][ncv]
        shcl = cs["shcl"][ncv]
        smcl = cs["smcl"][ncv]

        def lg(k, cc):
            tunlramp = CTUNL * TUNL * (1.0 - (1.0 - 1.0 / CTUNL) * jnp.exp(jnp.minimum(0.0, ricl_n)))
            tunlramp = jnp.minimum(jnp.maximum(tunlramp, TUNL), CTUNL * TUNL)
            lk = _cubic_root_length(zi[k], lbulk, tunlramp)
            return dict(cc, leng=cc["leng"].at[k].set(lk),
                        wcap=cc["wcap"].at[k].set((lk * lk) * (-shcl * n2[k] + smcl * s2[k])))

        cs = _fori_down(jnp.minimum(kb, n), kt, lg, cs)
        leng = cs["leng"]
        kbm = jnp.maximum(kb - 1, 1)
        jbzm = z[kbm] - z[jnp.minimum(kb, n)]
        jbsl = sl[kbm] - sl[jnp.minimum(kb, n)]
        jbqt = qt[kbm] - qt[jnp.minimum(kb, n)]
        jbbu = jnp.maximum(n2[jnp.minimum(kb, n)] * jbzm, JBUMIN)
        jbu = u[kbm] - u[jnp.minimum(kb, n)]
        jbv = v[kbm] - v[jnp.minimum(kb, n)]
        chb = (1.0 - sflh[kbm]) * chu[kbs] + sflh[kbm] * chs[kbs]
        cmb = (1.0 - sflh[kbm]) * cmu[kbs] + sflh[kbm] * cms[kbs]
        n2hb = (chb * jbsl + cmb * jbqt) / jbzm
        vyb = n2hb * jbzm / jbbu
        vub = jnp.minimum(1.0, (jbu * jbu + jbv * jbv) / (jbbu * jbzm))
        jbbu = jnp.where(surf, 0.0, jbbu)
        n2hb = jnp.where(surf, 0.0, n2hb)
        vyb = jnp.where(surf, 0.0, vyb)
        vub = jnp.where(surf, 0.0, vub)
        web = jnp.where(surf, 0.0, cs["web"])
        jtzm = z[kt - 1] - z[kt]
        jtsl = sl[kt - 1] - sl[kt]
        jtqt = qt[kt - 1] - qt[kt]
        jtbu = jnp.maximum(n2[kt] * jtzm, JBUMIN)
        jtu = u[kt - 1] - u[kt]
        jtv = v[kt - 1] - v[kt]
        cht = (1.0 - sfuh[kt]) * chu[kt] + sfuh[kt] * chs[kt]
        cmt = (1.0 - sfuh[kt]) * cmu[kt] + sfuh[kt] * cms[kt]
        n2ht = (cht * jtsl + cmt * jtqt) / jtzm
        vyt = n2ht * jtzm / jtbu
        vut = jnp.minimum(1.0, (jtu * jtu + jtv * jtv) / (jtbu * jtzm))
        # evhc ('maxi')
        qleff = jnp.maximum(ql[kt - 1], ql[kt])
        jt2slv = slv[jnp.maximum(kt - 2, 1)] - slv[kt]
        jt2slv = jnp.maximum(jt2slv, JBUMIN * slv[kt - 1] / GRAVIT)
        evhc = 1.0 + A2L * A3L * LATVAP * qleff / jt2slv
        evhc = jnp.minimum(evhc, EVHCMAX)
        # radf ('maxi')
        lwp = ql[kt] * (pi[kt + 1] - pi[kt]) / GRAVIT
        od = 156.0 * lwp
        rif = od * (4.0 + od) / (6.0 * (4.0 + od) + od * od)
        radf = jnp.maximum(rif * qrlw[kt] / (pi[kt] - pi[kt + 1]) * (zi[kt] - zi[kt + 1]), 0.0)
        lwp = ql[kt - 1] * (pi[kt] - pi[kt - 1]) / GRAVIT
        od = 156.0 * lwp
        rif = od * (4.0 + od) / (6.0 * (4.0 + od) + od * od)
        radf = radf + jnp.maximum(rif * qrlw[kt - 1] / (pi[kt - 1] - pi[kt]) * (zi[kt - 1] - zi[kt]), 0.0)
        radf = jnp.maximum(radf, 0.0) * chs[kt]
        dzht = zi[kt] - z[kt]
        dzhb = z[kb - 1] - zi[kbs]
        bprod_c = cs["bprod"]

        def wsum(k, w):
            return w + bprod_c[k] * (z[k - 1] - z[k])

        wstar3 = _fori_up(kt + 1, kb - 1, wsum, radf * dzht)
        wstar3 = jnp.where(surf & (bflxs > 0.0), wstar3 + bflxs * dzhb, wstar3)
        wstar3 = jnp.maximum(2.5 * wstar3, 0.0)
        pos = wstar3 > 0.0
        cet = jnp.where(pos, A1I * evhc / (jtbu * lbulk), 0.0)
        ceb_new = A1I / (jbbu * lbulk)
        ceb = jnp.where(pos, jnp.where(surf, cs["ceb"], ceb_new), 0.0)
        w3f_s = jnp.maximum(1.0 + 2.5 * cet * n2ht * jtzm * dzht, WSTAR3FACTCRIT)
        w3f_b = jnp.maximum(1.0 + 2.5 * cet * n2ht * jtzm * dzht + 2.5 * ceb_new * n2hb * jbzm * dzhb, WSTAR3FACTCRIT)
        wstar3fact = jnp.where(pos, jnp.where(surf, w3f_s, w3f_b), 0.0)
        wstar3 = jnp.where(pos, wstar3 / jnp.where(pos, wstar3fact, 1.0), wstar3)
        leng_kb = leng[kbs]
        fact = (evhc * (-vyt + vut) * dzht + (-vyb + vub) * dzhb * leng_kb / leng[kt]) / lbulk
        # wstarent == .true.
        ebrk_n = cs["ebrk"][ncv]
        lbrk_n = cs["lbrk"][ncv]
        trmp = ebrk_n * (lbrk_n / lbulk) / 3.0 + NTZERO
        trmq = 0.5 * B1 * (leng[kt] / lbulk) * (radf * dzht + A1I * fact * wstar3)
        rmin = jnp.sqrt(trmp)
        fmin = rmin * (rmin * rmin - 3.0 * trmp) - 2.0 * trmq
        wstar = jnp.power(wstar3, ONET)
        rcrit = CCRIT * wstar
        fcrit = rcrit * (rcrit * rcrit - 3.0 * trmp) - 2.0 * trmq
        noroot = ((rmin < rcrit) & (fcrit > 0.0)) | ((rmin >= rcrit) & (fmin > 0.0))
        trma_nr = jnp.maximum(1.0 - B1 * (leng[kt] / lbulk) * A1I * fact / (CCRIT * CCRIT * CCRIT), 0.5)
        trmp = jnp.where(noroot, trmp / trma_nr, trmp)
        trmq = jnp.where(noroot, 0.5 * B1 * (leng[kt] / lbulk) * radf * dzht / trma_nr, trmq)
        qq = trmq * trmq - trmp * trmp * trmp
        sq = jnp.sqrt(jnp.maximum(qq, 0.0))
        root_a = jnp.power(trmq + sq, 1.0 / 3.0) + jnp.power(jnp.maximum(trmq - sq, 0.0), 1.0 / 3.0)
        tp_safe = jnp.where(qq >= 0.0, 1.0, trmp)
        root_b = 2.0 * jnp.sqrt(tp_safe) * jnp.cos(jnp.arccos(trmq / jnp.sqrt(tp_safe * tp_safe * tp_safe)) / 3.0)
        rootp = jnp.where(qq >= 0.0, root_a, root_b)
        _mark("cl_total")
        _mark("cl_noroot", noroot)
        _mark("cl_qq_negative", qq < 0.0)
        _mark("cl_surface_based", surf)
        _mark("cl_wstar3_positive", pos)
        wstar3 = jnp.where(noroot, (rootp / CCRIT) * (rootp / CCRIT) * (rootp / CCRIT), wstar3)
        wet = cet * wstar3
        web = jnp.where(~surf, ceb * wstar3, web)
        ebrk_v = jnp.minimum(rootp * rootp, TKEMAX)
        wbrk_v = ebrk_v / B1
        cs = dict(cs, ebrk=cs["ebrk"].at[ncv].set(ebrk_v), wbrk=cs["wbrk"].at[ncv].set(wbrk_v))
        drop = ebrk_v <= 0.0
        cs = dict(cs, belongcv=jnp.where(drop, cs["belongcv"].at[kt].set(False).at[kbs].set(False), cs["belongcv"]))

        def interior(k, cc):
            rcap = (B1 * AE + cc["wcap"][k] / wbrk_v) / (B1 * AE + 1.0)
            rcap = jnp.minimum(jnp.maximum(rcap, RCAPMIN), RCAPMAX)
            tk = jnp.minimum(ebrk_v * rcap, TKEMAX)
            kh = cc["leng"][k] * jnp.sqrt(tk) * shcl
            km = cc["leng"][k] * jnp.sqrt(tk) * smcl
            return dict(cc, tke=cc["tke"].at[k].set(tk), kvh=cc["kvh"].at[k].set(kh), kvm=cc["kvm"].at[k].set(km),
                        bprod=cc["bprod"].at[k].set(-kh * n2[k]), sprod=cc["sprod"].at[k].set(km * s2[k]),
                        turbtype=cc["turbtype"].at[k].set(2), sm_aw=cc["sm_aw"].at[k].set(smcl / ALPH1))

        cs = _fori_down(kb - 1, kt + 1, interior, cs)
        kentr = wet * jtzm
        bp_t = -kentr * n2ht + radf
        sp_t = kentr * s2[kt]
        trmp_c = -B1 * AE / (1.0 + B1 * AE)
        e15 = jnp.power(ebrk_v, 1.5)
        trmq_t = -(bp_t + sp_t) * B1 * cs["leng"][kt] / (1.0 + B1 * AE) / e15
        rc = _compute_cubic(0.0, trmp_c, trmq_t)
        rcap = jnp.minimum(jnp.maximum(rc * rc, RCAPMIN), RCAPMAX)
        tke_t = jnp.minimum(ebrk_v * rcap, TKEMAX)
        cs = dict(cs, kvh=cs["kvh"].at[kt].set(kentr), kvm=cs["kvm"].at[kt].set(kentr),
                  bprod=cs["bprod"].at[kt].set(bp_t), sprod=cs["sprod"].at[kt].set(sp_t),
                  turbtype=cs["turbtype"].at[kt].set(4), tke=cs["tke"].at[kt].set(tke_t),
                  sm_aw=cs["sm_aw"].at[kt].set(smcl / ALPH1))
        kentr_b = web * jbzm

        def base_new(cc):
            bp = -kentr_b * n2hb
            sp = kentr_b * s2[kbs]
            tq = -(bp + sp) * B1 * cc["leng"][kbs] / (1.0 + B1 * AE) / e15
            r = _compute_cubic(0.0, trmp_c, tq)
            rcp = jnp.minimum(jnp.maximum(r * r, RCAPMIN), RCAPMAX)
            return dict(cc, kvh=cc["kvh"].at[kbs].set(kentr_b), kvm=cc["kvm"].at[kbs].set(kentr_b),
                        bprod=cc["bprod"].at[kbs].set(bp), sprod=cc["sprod"].at[kbs].set(sp),
                        turbtype=cc["turbtype"].at[kbs].set(3),
                        tke=cc["tke"].at[kbs].set(jnp.minimum(ebrk_v * rcp, TKEMAX)))

        def base_merge(cc):
            dzhb5 = z[kb - 1] - zi[kbs]
            dzht5 = zi[kbs] - z[jnp.minimum(kb, n)]
            bp = (dzht5 * cc["bprod"][kbs] - dzhb5 * kentr_b * n2hb) / (dzhb5 + dzht5)
            sp = (dzht5 * cc["sprod"][kbs] + dzhb5 * kentr_b * s2[kbs]) / (dzhb5 + dzht5)
            tq = -kentr_b * (s2[kbs] - n2hb) * B1 * cc["leng"][kbs] / (1.0 + B1 * AE) / e15
            r = _compute_cubic(0.0, trmp_c, tq)
            rcp = jnp.minimum(jnp.maximum(r * r, RCAPMIN), RCAPMAX)
            tke_imsi = jnp.minimum(ebrk_v * rcp, TKEMAX)
            tk = (dzht5 * cc["tke"][kbs] + dzhb5 * tke_imsi) / (dzhb5 + dzht5)
            return dict(cc, kvh=cc["kvh"].at[kbs].add(kentr_b), kvm=cc["kvm"].at[kbs].add(kentr_b),
                        bprod=cc["bprod"].at[kbs].set(bp), sprod=cc["sprod"].at[kbs].set(sp),
                        tke=cc["tke"].at[kbs].set(jnp.minimum(tk, TKEMAX)), turbtype=cc["turbtype"].at[kbs].set(5))

        def base_surf(cc):
            rcp = (B1 * AE + cc["wcap"][kbs] / wbrk_v) / (B1 * AE + 1.0)
            rcp = jnp.minimum(jnp.maximum(rcp, RCAPMIN), RCAPMAX)
            return dict(cc, tke=cc["tke"].at[kbs].set(jnp.minimum(ebrk_v * rcp, TKEMAX)))

        branch = jnp.where(surf, 2, jnp.where(kb != cs["ktblw"], 0, 1))
        _mark("cl_base_merged_with_below", branch == 1)
        cs = lax.switch(branch, tuple((lambda f: (lambda c: _like(f(c), c)))(f) for f in (base_new, base_merge, base_surf)), cs)
        cs = dict(cs, sm_aw=cs["sm_aw"].at[kbs].set(smcl / ALPH1))
        wcap_t = (cs["bprod"][kt] + cs["sprod"][kt]) * cs["leng"][kt] / jnp.sqrt(jnp.maximum(cs["tke"][kt], 1.0e-6))
        wcap_b = (cs["bprod"][kbs] + cs["sprod"][kbs]) * cs["leng"][kbs] / jnp.sqrt(jnp.maximum(cs["tke"][kbs], 1.0e-6))
        wc = cs["wcap"].at[kt].set(wcap_t)
        wc = jnp.where(surf, wc, wc.at[kbs].set(wcap_b))
        return dict(cs, wcap=wc, ktblw=kt, web=web, ceb=ceb, wstar=wstar,
                    wstar1=jnp.where(ncv == 1, wstar, cs["wstar1"]))

    cs = _fori_up(1, ncvfin, cl_loop, cs)
    kvh, kvm, leng, tke, wcap = cs["kvh"], cs["kvm"], cs["leng"], cs["tke"], cs["wcap"]
    bprod, sprod, turbtype, sm_aw = cs["bprod"], cs["sprod"], cs["turbtype"], cs["sm_aw"]
    belongcv = cs["belongcv"]

    # ---------------- surface-based CL diagnostics ----------------
    ncvs = jnp.maximum(ncvsurf, 1)
    ktopbl_c = cs["ktop"][ncvs]
    pblh_c = zi[ktopbl_c]
    wpert_c = jnp.maximum(WFAC * jnp.sqrt(cs["ebrk"][ncvs]), WPERTMIN)
    tpert_c = jnp.maximum(jnp.abs(shflx * rrho / CPAIR) * TFAC / wpert_c, 0.0)
    qpert_c = jnp.maximum(jnp.abs(qflx * rrho) * TFAC / wpert_c, 0.0)
    has_s = ncvsurf > 0
    turbtype = jnp.where(has_s, turbtype.at[n + 1].set(jnp.where(bflxs > 0.0, 2, 3)), turbtype)
    pblh = jnp.where(has_s, pblh_c, 0.0)
    wpert = jnp.where(has_s, wpert_c, 0.0)
    tpert = jnp.where(has_s, tpert_c, 0.0)
    qpert = jnp.where(has_s, qpert_c, 0.0)
    ipbl = jnp.where(has_s, 1.0, 0.0)
    kpblh = jnp.where(has_s, (ktopbl_c - 1).astype(F64), jnp.float64(n))

    # ---------------- STL ----------------
    stl = dict(leng=leng, belongst=jnp.zeros(ni, bool), kt=cs["ktop"][jnp.maximum(ncvfin, 1)],
               kb=jnp.int32(0))
    stl["kt"] = jnp.where(ncvfin > 0, stl["kt"], jnp.int32(0))

    def stl_body(k, s):
        bst = (ri[k] < RICRIT) & ~belongcv[k]
        prev = s["belongst"][k - 1]
        s = dict(s, belongst=s["belongst"].at[k].set(bst))
        start = bst & ~prev
        end = (~bst) & prev
        s = dict(s, kt=jnp.where(start, k, s["kt"]))

        def fin(s):
            kb = k - 1
            lbulk = z[s["kt"] - 1] - z[kb]

            def lg(ks, ss):
                return dict(ss, leng=ss["leng"].at[ks].set(_cubic_root_length(zi[ks], lbulk, TUNL)))
            s = _fori_up(s["kt"], kb, lg, s)
            return dict(s, kb=kb)

        return _when(end & ~start, fin, s)

    stl = _fori_up(2, n, stl_body, stl)
    belongst = stl["belongst"].at[n + 1].set(~belongcv[n + 1])
    surf_stl = belongst[n + 1]
    kt_s = jnp.where(belongst[n], stl["kt"], n + 1)
    lbulk_s = z[kt_s - 1]
    ktopbl_s = kt_s - 1

    def stl_surf(s):
        def lg(ks, ss):
            return dict(ss, leng=ss["leng"].at[ks].set(_cubic_root_length(zi[ks], lbulk_s, TUNL)))
        return _fori_up(kt_s, n, lg, s)

    stl = _when(surf_stl, stl_surf, stl)
    leng = stl["leng"]
    turbtype = jnp.where(surf_stl, turbtype.at[n + 1].set(1), turbtype)
    pblh = jnp.where(surf_stl, z[ktopbl_s], pblh)
    wpert = jnp.where(surf_stl, 0.0, wpert)
    tpert = jnp.where(surf_stl, jnp.maximum(shflx * rrho / CPAIR * FAK / ustar, 0.0), tpert)
    qpert = jnp.where(surf_stl, jnp.maximum(qflx * rrho * FAK / ustar, 0.0), qpert)
    ipbl = jnp.where(surf_stl, 0.0, ipbl)
    kpblh = jnp.where(surf_stl, ktopbl_s.astype(F64), kpblh)

    # STL interfaces
    ri_i = n2_pad(ri, ni)
    n2_i = n2_pad(n2, ni)
    s2_i = n2_pad(s2, ni)
    _, sh, sm = _stab_funcs_stl(ri_i)
    tke_st = jnp.minimum(B1 * (leng * leng) * (-sh * n2_i + sm * s2_i), TKEMAX)
    kvh_st = leng * jnp.sqrt(tke_st) * sh
    kvm_st = leng * jnp.sqrt(tke_st) * sm
    m = belongst & inner
    turbtype = jnp.where(m, 1, turbtype)
    tke = jnp.where(m, tke_st, tke)
    wcap = jnp.where(m, tke_st / B1, wcap)
    kvh = jnp.where(m, kvh_st, kvh)
    kvm = jnp.where(m, kvm_st, kvm)
    bprod = jnp.where(m, -kvh_st * n2_i, bprod)
    sprod = jnp.where(m, kvm_st * s2_i, sprod)
    sm_aw = jnp.where(m, sm / ALPH1, sm_aw)

    # enhancement of CL-edge interfaces by local stability (turbtype 3/4/5)
    edge = inner & ((turbtype == 3) | (turbtype == 4) | (turbtype == 5))
    zm1 = jnp.concatenate([jnp.zeros(1, F64), z[:n], jnp.zeros(1, F64)])  # z(k-1) at interface k
    zk = jnp.concatenate([z, jnp.zeros(1, F64)])  # z(k)
    lbulk_e = zm1 - zk
    lbulk_e = jnp.where(edge, lbulk_e, 1.0)
    leng_imsi = _cubic_root_length(jnp.where(edge, zi, 1.0), lbulk_e, TUNL)
    tke_imsi = jnp.minimum(jnp.maximum(B1 * (leng_imsi * leng_imsi) * (-sh * n2_i + sm * s2_i), 0.0), TKEMAX)
    kvh_imsi = leng_imsi * jnp.sqrt(tke_imsi) * sh
    kvm_imsi = leng_imsi * jnp.sqrt(tke_imsi) * sm
    up = edge & (kvh < kvh_imsi)
    _mark("edge_enhanced_interfaces", jnp.sum(up))
    kvh = jnp.where(up, kvh_imsi, kvh)
    kvm = jnp.where(up, kvm_imsi, kvm)
    leng = jnp.where(up, leng_imsi, leng)
    tke = jnp.where(up, tke_imsi, tke)
    wcap = jnp.where(up, tke_imsi / B1, wcap)
    bprod = jnp.where(up, -kvh_imsi * n2_i, bprod)
    sprod = jnp.where(up, kvm_imsi * s2_i, sprod)
    sm_aw = jnp.where(up, sm / ALPH1, sm_aw)
    turbtype = jnp.where(up, 1, turbtype)

    # surface interface diagnostics (sm_aw(pver+1))
    bprod = bprod.at[n + 1].set(bflxs)
    gg = 0.5 * KARMAN * z[n] * bprod[n + 1] / jnp.power(tkes, 1.5)
    small = jnp.abs(ALPH5 - gg * ALPH3) <= 1.0e-7
    gh = jnp.where(small, jnp.where(bprod[n + 1] > 0.0, -3.5334, GHMIN), gg / (ALPH5 - gg * ALPH3))
    gh = jnp.where(bprod[n + 1] > 0.0, jnp.minimum(jnp.maximum(gh, -3.5334), 0.0233),
                   jnp.minimum(jnp.maximum(gh, GHMIN), 0.0233))
    sm_s = jnp.where(bprod[n + 1] > 0.0,
                     jnp.maximum(0.0, (ALPH1 + ALPH2 * gh) / (1.0 + ALPH3 * gh) / (1.0 + ALPH4 * gh)),
                     jnp.maximum(0.0, (ALPH1 + ALPH2 * gh) / (1.0 + ALPH3 * gh) / (1.0 + ALPH4EXS * gh)))
    sm_aw = sm_aw.at[n + 1].set(sm_s / ALPH1)
    return dict(kvh=kvh, kvm=kvm, tke=tke, bprod=bprod, sprod=sprod, turbtype=turbtype, sm_aw=sm_aw,
                pblh=pblh, tpert=tpert, qpert=qpert, wpert=wpert, ipbl=ipbl, kpblh=kpblh,
                wstar1=cs["wstar1"])


def n2_pad(a, ni):
    """Interface array of length pver+2 from an (index 0 pad, 1..pver) mid-sized array."""

    return jnp.concatenate([a, jnp.zeros(ni - a.shape[0], F64)])


# --------------------------------------------------------------------------------------------
# diffusion solver
# --------------------------------------------------------------------------------------------


def _lu_decomp(pver, ksrf, kv, tmpi, rpdel, ztodt):
    """vd_lu_decomp with ntop=1, nbot=pver, cc_top=0 -> (ca, ze, dnom), 1-based."""

    n = pver
    ca = jnp.zeros(n + 1, F64).at[1:n].set(kv[2:n + 1] * tmpi[2:n + 1] * rpdel[1:n])
    cc = jnp.zeros(n + 1, F64).at[2:n + 1].set(kv[2:n + 1] * tmpi[2:n + 1] * rpdel[2:n + 1])
    dnom_n = 1.0 / (1.0 + cc[n] + ksrf * ztodt * GRAVIT * rpdel[n])
    ze_n = cc[n] * dnom_n

    def step(ze_next, k):
        d = 1.0 / (1.0 + ca[k] + cc[k] - ca[k] * ze_next)
        return cc[k] * d, (d, cc[k] * d)

    ks = jnp.arange(n - 1, 1, -1)
    ze2, (dn, zz) = lax.scan(step, ze_n, ks)
    dnom = jnp.zeros(n + 1, F64).at[n].set(dnom_n).at[ks].set(dn)
    ze = jnp.zeros(n + 1, F64).at[n].set(ze_n).at[ks].set(zz)
    dnom = dnom.at[1].set(1.0 / (1.0 + ca[1] + 0.0 - ca[1] * ze2))
    return ca, ze, dnom


def _lu_solve(pver, q, ca, ze, dnom):
    n = pver
    zf_n = q[n] * dnom[n]

    def back(zf_next, k):
        zf = (q[k] + ca[k] * zf_next) * dnom[k]
        return zf, zf

    ks = jnp.arange(n - 1, 1, -1)
    zf2, zfs = lax.scan(back, zf_n, ks)
    zf = jnp.zeros(n + 1, F64).at[n].set(zf_n).at[ks].set(zfs)
    zf1 = (q[1] + 0.0 + ca[1] * zf2) * dnom[1]
    zf = zf.at[1].set(zf1)

    def fwd(qprev, k):
        qk = zf[k] + ze[k] * qprev
        return qk, qk

    _, qs = lax.scan(fwd, zf1, jnp.arange(2, n + 1))
    return jnp.zeros(n + 1, F64).at[1].set(zf1).at[2:n + 1].set(qs)


def _compute_vdiff(pver, pmid, pint, rpdel, t, ztodt, taux, tauy, shflx, cflx0, kvh, kvm, kvq,
                   u, v, qs, dse, tauresx, tauresy, itaures):
    """compute_vdiff, do_iss, no molecular diffusion, ksrftms=0, cgs=cgh=0 (exact no-ops)."""

    n = pver
    tint = jnp.zeros(n + 2, F64).at[1].set(t[1]).at[2:n + 1].set(0.5 * (t[2:n + 1] + t[1:n])).at[n + 1].set(t[n])
    rhoi = jnp.where(jnp.arange(n + 2) >= 1, pint / (RAIR * jnp.where(jnp.arange(n + 2) >= 1, tint, 1.0)), 0.0)
    tmpi2 = jnp.zeros(n + 2, F64).at[2:n + 1].set(
        ztodt * (GRAVIT * rhoi[2:n + 1]) ** 2 / (pmid[2:n + 1] - pmid[1:n]))
    tmp1 = ztodt * GRAVIT * rpdel[n]
    # momentum
    dinp_u = jnp.zeros(n + 2, F64).at[2:n + 1].set(u[2:n + 1] - u[1:n]).at[n + 1].set(-u[n])
    dinp_v = jnp.zeros(n + 2, F64).at[2:n + 1].set(v[2:n + 1] - v[1:n]).at[n + 1].set(-v[n])
    ws = jnp.maximum(jnp.sqrt(u[n] * u[n] + v[n] * v[n]), WSMIN)
    tau = jnp.sqrt(taux * taux + tauy * tauy)
    ksrf = jnp.maximum(tau / ws, KSRFMIN)

    def colsum(a):
        def s(acc, k):
            return acc + (1.0 / GRAVIT) * a[k] / rpdel[k], None
        return lax.scan(s, jnp.float64(0.0), jnp.arange(1, n + 1))[0]

    usum_in = colsum(u)
    vsum_in = colsum(v)
    ramda = ztodt / TIMERES
    u = u.at[n].set(u[n] + tmp1 * tauresx * ramda)
    v = v.at[n].set(v[n] + tmp1 * tauresy * ramda)
    ca, ze, dnom = _lu_decomp(n, ksrf, kvm, tmpi2, rpdel, ztodt)
    u = _lu_solve(n, u, ca, ze, dnom)
    v = _lu_solve(n, v, ca, ze, dnom)
    tautmsx = -0.0 * u[n]
    tautmsy = -0.0 * v[n]
    tauimpx = (colsum(u) - usum_in) / ztodt
    tauimpy = (colsum(v) - vsum_in) / ztodt
    if itaures == 1:
        tauresx = taux + tautmsx + tauresx - tauimpx
        tauresy = tauy + tautmsy + tauresy - tauimpy
    tmpi1 = jnp.zeros(n + 2, F64)
    tmpi1 = tmpi1.at[n + 1].set(0.5 * ztodt * GRAVIT * ((-u[n] + dinp_u[n + 1]) * tauimpx
                                                        + (-v[n] + dinp_v[n + 1]) * tauimpy))
    du = u[2:n + 1] - u[1:n]
    dv = v[2:n + 1] - v[1:n]
    tmpi1 = tmpi1.at[2:n + 1].set(0.25 * tmpi2[2:n + 1] * kvm[2:n + 1]
                                  * (du * du + dv * dv + du * dinp_u[2:n + 1] + dv * dinp_v[2:n + 1]))
    dtk = jnp.zeros(n + 1, F64).at[1:n + 1].set((tmpi1[2:n + 2] + tmpi1[1:n + 1]) * rpdel[1:n + 1])
    dse = dse + dtk
    # dry static energy (cgh == 0 -> counter-gradient term adds exactly +0)
    dse = dse.at[n].set(dse[n] + tmp1 * shflx)
    ca, ze, dnom = _lu_decomp(n, 0.0, kvh, tmpi2, rpdel, ztodt)
    dse = _lu_solve(n, dse, ca, ze, dnom)
    # constituents (kvq decomposition once; only constituent 1 has a surface flux)
    ca, ze, dnom = _lu_decomp(n, 0.0, kvq, tmpi2, rpdel, ztodt)
    out = []
    for m, q in enumerate(qs):
        if m == 0:
            q = q.at[n].set(q[n] + tmp1 * cflx0)
        out.append(_lu_solve(n, q, ca, ze, dnom))
    return u, v, out, dse, tauresx, tauresy


# --------------------------------------------------------------------------------------------
# compute_eddy_diff
# --------------------------------------------------------------------------------------------


def _compute_eddy_diff(n, t, qv, ztodt, ql, qi, s, rpdel, cldn, qrl, z, zi, pmid, pi, u, v, taux,
                       tauy, shflx, qflx, kvm_in, kvh_in, kvinit, tauresx, tauresy):
    del s
    zero_i = jnp.zeros(n + 2, F64)

    def trb(ufd, vfd, tfd, qvfd, qlfd):
        return _trbintd(n, z, ufd, vfd, tfd, pmid, taux, tauy, zi, pi, cldn, qvfd, qlfd, qi)

    def eddy(tb, qlfd, kvh, kvm):
        return _caleddy(n, tb["sl"], tb["qt"], qlfd, tb["slv"], tb["u"], tb["v"], pi, z, zi, qflx,
                        shflx, tb["slslope"], tb["qtslope"], tb["chu"], tb["chs"], tb["cmu"], tb["cms"],
                        tb["sfuh"], tb["sflh"], tb["n2"], tb["s2"], tb["ri"], tb["rrho"], tb["ustar"],
                        kvh, kvm, qrl)

    # iteration 1
    tb = trb(u, v, t, qv, ql)
    tb = dict(tb, u=u, v=v)
    qt0, sl0 = tb["qt"], tb["sl"]
    kvh = jnp.where(kvinit, zero_i, kvh_in)
    kvm = jnp.where(kvinit, zero_i, kvm_in)
    ce = eddy(tb, ql, kvh, kvm)

    def vdiff_and_adjust(kvh_out, kvm_out):
        ufd, vfd, (qtfd,), slfd, _, _ = _compute_vdiff(
            n, pmid, pi, rpdel, t, ztodt, taux, tauy, shflx, qflx, kvh_out, kvm_out, kvh_out,
            u, v, [qt0], sl0, tauresx, tauresy, 0)
        templ = (slfd - GRAVIT * z) / CPAIR
        templ_s = jnp.where(jnp.arange(n + 1) == 0, 280.0, templ)
        _, qs, _ = _vqsatd(templ_s, jnp.where(jnp.arange(n + 1) == 0, 1.0e5, pmid))
        temps = templ_s + (qtfd - qs) / (CPAIR / LATVAP + LATVAP * qs / (RAIR * (templ_s * templ_s)))
        _, qs, _ = _vqsatd(temps, jnp.where(jnp.arange(n + 1) == 0, 1.0e5, pmid))
        qlfd = jnp.maximum(qtfd - qi - qs, 0.0)
        qvfd = jnp.maximum(0.0, qtfd - qi - qlfd)
        tfd = (slfd + LATVAP * qlfd + LATSUB * qi - GRAVIT * z) / CPAIR
        tfd = jnp.where(jnp.arange(n + 1) == 0, 0.0, tfd)
        return ufd, vfd, tfd, qvfd, qlfd

    carry = dict(kvh_prev=kvh, kvm_prev=kvm, kvh_out=ce["kvh"], kvm_out=ce["kvm"], ce=ce)
    ufd, vfd, tfd, qvfd, qlfd = vdiff_and_adjust(ce["kvh"], ce["kvm"])

    def it_body(iturb, c):
        ufd, vfd, tfd, qvfd, qlfd = c["state"]
        tb = trb(ufd, vfd, tfd, qvfd, qlfd)
        tb = dict(tb, u=ufd, v=vfd)
        kvh, kvm = c["kvh_out"], c["kvm_out"]
        ce = eddy(tb, qlfd, kvh, kvm)
        relax = iturb < NTURB
        kvh_out = jnp.where(relax, LAMBDA * ce["kvh"] + (1.0 - LAMBDA) * kvh, ce["kvh"])
        kvm_out = jnp.where(relax, LAMBDA * ce["kvm"] + (1.0 - LAMBDA) * kvm, ce["kvm"])
        new_state = lax.cond(relax, lambda: vdiff_and_adjust(kvh_out, kvm_out), lambda: (ufd, vfd, tfd, qvfd, qlfd))
        return dict(c, state=new_state, kvh_out=kvh_out, kvm_out=kvm_out, ce=ce, ustar=tb["ustar"])

    tb0 = tb
    c = dict(state=(ufd, vfd, tfd, qvfd, qlfd), kvh_out=ce["kvh"], kvm_out=ce["kvm"], ce=ce, ustar=tb0["ustar"])
    c = lax.fori_loop(2, NTURB + 1, it_body, c)
    ce = c["ce"]
    return dict(ce, kvh=c["kvh_out"], kvm=c["kvm_out"], kvq=c["kvh_out"], ustar=c["ustar"])


# --------------------------------------------------------------------------------------------
# camuwpbl driver (one WRF column)
# --------------------------------------------------------------------------------------------


def camuw_column(dt, u_phy, v_phy, th_phy, rho, qv, qc, qi, qnc, qni, p_phy, p8w, z, z_at_w, t_phy,
                 cldfra, rthratenlw, exner, wsedl, hfx, qfx, ustar, ht, kvm3d, kvh3d, tauresx2d,
                 tauresy2d, first_step):
    """One WRF column (index 0 = lowest level, float32 WRF REAL in/out).

    Mid-level inputs have shape (kx,), interface inputs (kx+1,). Returns a dict of WRF REAL
    outputs: rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, rqniblten (kx,);
    kvm3d, kvh3d, tke_pbl, smaw3d, turbtype3d (kx+1,); pblh, kpbl, tpert, qpert, wpert,
    tauresx2d, tauresy2d (scalars).
    """

    del th_phy, wsedl  # th_phy unused by camuwpbl; wsedl only with id_sedfact=.true.
    kx = u_phy.shape[0]
    n = kx
    f32 = lambda a: jnp.asarray(a, F32)  # noqa: E731

    def flip_mid(a):
        return jnp.concatenate([jnp.zeros(1, F64), jnp.asarray(a, F64)[::-1]])

    def flip_int(a):
        return jnp.concatenate([jnp.zeros(1, F64), jnp.asarray(a, F64)[::-1]])

    p8w32, z32, zw32, ht32 = f32(p8w), f32(z), f32(z_at_w), f32(ht)
    dp32 = p8w32[:-1] - p8w32[1:]  # REAL difference
    dp = flip_mid(dp32)
    rpdel = jnp.where(jnp.arange(n + 1) >= 1, 1.0 / jnp.where(jnp.arange(n + 1) >= 1, dp, 1.0), 0.0)
    zm = flip_mid(z32 - ht32)
    zi = flip_int(zw32 - ht32)
    pint = flip_int(p8w32)
    pmid = flip_mid(p_phy)
    t8 = flip_mid(t_phy)
    u8 = flip_mid(u_phy)
    v8 = flip_mid(v_phy)
    phis = jnp.asarray(ht32, F64) * GRAVIT
    s8 = CPAIR * t8 + GRAVIT * zm + phis
    qrl8 = flip_mid(f32(rthratenlw) * f32(exner)) * CPAIR * dp
    qv32 = f32(qv)
    mult = 1.0 / (1.0 + jnp.asarray(qv32, F64))
    q1 = jnp.maximum(jnp.asarray(qv32, F64) * mult, 1.0e-30)
    cl_q = jnp.asarray(f32(qc), F64) * mult
    cl_i = jnp.asarray(f32(qi), F64) * mult
    cl_nc = jnp.asarray(f32(qnc), F64) * mult
    cl_ni = jnp.asarray(f32(qni), F64) * mult
    cloud = [flip_mid(q1), flip_mid(cl_q), flip_mid(cl_i), flip_mid(cl_nc), flip_mid(cl_ni)]
    exner8 = flip_mid(exner)
    cldn = flip_mid(cldfra)
    ztodt = jnp.asarray(f32(dt), F64)
    rztodt = 1.0 / ztodt
    # explicit surface stress (BareGround form), REAL arithmetic as in WRF
    u0, v0 = f32(u_phy)[0], f32(v_phy)[0]
    umean = jnp.asarray(jnp.sqrt(u0 * u0 + v0 * v0), F64)
    taufac = jnp.asarray(f32(rho)[0] * f32(ustar) * f32(ustar), F64) / jnp.maximum(umean, 1.0e-30)
    taux = -taufac * jnp.asarray(u0, F64)
    tauy = -taufac * jnp.asarray(v0, F64)
    kvh_in = flip_int(f32(kvh3d))
    kvm_in = flip_int(f32(kvm3d))
    kvh_in = jnp.where(first_step, 0.0, kvh_in)
    kvm_in = jnp.where(first_step, 0.0, kvm_in)
    tresx = jnp.where(first_step, 0.0, jnp.asarray(f32(tauresx2d), F64))
    tresy = jnp.where(first_step, 0.0, jnp.asarray(f32(tauresy2d), F64))
    shflx = jnp.asarray(f32(hfx), F64)
    cflx0 = jnp.asarray(f32(qfx), F64)

    ed = _compute_eddy_diff(n, t8, cloud[0], ztodt, cloud[1], cloud[2], s8, rpdel, cldn, qrl8, zm, zi,
                            pmid, pint, u8, v8, taux, tauy, shflx, cflx0, kvm_in, kvh_in, first_step,
                            tresx, tresy)
    u_new, v_new, q_new, s_new, tresx, tresy = _compute_vdiff(
        n, pmid, pint, rpdel, t8, ztodt, taux, tauy, shflx, cflx0, ed["kvh"], ed["kvm"], ed["kvq"],
        u8, v8, cloud, s8, tresx, tresy, 1)
    stnd = (s_new - s8) * rztodt
    ut = (u_new - u8) * rztodt
    vt = (v_new - v8) * rztodt
    qt = [(qn - qo) * rztodt for qn, qo in zip(q_new, cloud)]

    def unflip_mid(a):
        return a[1:][::-1]

    def unflip_int(a):
        return a[1:][::-1]

    multf = 1.0 + jnp.asarray(qv32, F64)
    out = dict(
        rublten=unflip_mid(ut).astype(F32),
        rvblten=unflip_mid(vt).astype(F32),
        rthblten=jnp.where(True, unflip_mid(stnd / CPAIR / jnp.where(exner8 == 0, 1.0, exner8)), 0.0).astype(F32),
        rqvblten=(unflip_mid(qt[0]) * multf * multf).astype(F32),
        rqcblten=(unflip_mid(qt[1]) * multf).astype(F32),
        rqiblten=(unflip_mid(qt[2]) * multf).astype(F32),
        rqniblten=(unflip_mid(qt[4]) * multf).astype(F32),
        kvh3d=unflip_int(ed["kvh"]).astype(F32),
        kvm3d=unflip_int(ed["kvm"]).astype(F32),
        tke_pbl=unflip_int(ed["tke"]).astype(F32),
        turbtype3d=unflip_int(ed["turbtype"].astype(F64)).astype(F32),
        smaw3d=unflip_int(ed["sm_aw"]).astype(F32),
        kpbl=(n - ed["kpblh"].astype(jnp.int32) + 1).astype(jnp.int32),
        pblh=ed["pblh"].astype(F32),
        tpert=ed["tpert"].astype(F32),
        qpert=ed["qpert"].astype(F32),
        wpert=ed["wpert"].astype(F32),
        tauresx2d=tresx.astype(F32),
        tauresy2d=tresy.astype(F32),
    )
    return out


_COLUMN_ARGS = ("u_phy", "v_phy", "th_phy", "rho", "qv", "qc", "qi", "qnc", "qni", "p_phy", "p8w", "z",
                "z_at_w", "t_phy", "cldfra", "rthratenlw", "exner", "wsedl", "hfx", "qfx", "ustar", "ht",
                "kvm3d", "kvh3d", "tauresx2d", "tauresy2d")


@partial(jax.jit, static_argnames=())
def camuw_columns(dt, first_step, **cols):
    """Batched CAM-UW over columns: every array has a leading column axis (ncol, ...)."""

    args = [cols[name] for name in _COLUMN_ARGS]

    def one(*a):
        kw = dict(zip(_COLUMN_ARGS, a))
        return camuw_column(dt, first_step=first_step, **kw)

    return jax.vmap(one)(*args)


class CamUwCarry(NamedTuple):
    """WRF CAM-UW persistent fields (REAL): KVM3D/KVH3D/TKE_PBL (nz+1, ny, nx), TAURES[XY]2D (ny, nx)."""

    kvm3d: jax.Array
    kvh3d: jax.Array
    tauresx2d: jax.Array
    tauresy2d: jax.Array
    tke_pbl: jax.Array


def initial_camuw_carry(state) -> CamUwCarry:
    """Zero carry == WRF itimestep=1 (camuwpbl zeroes kvm3d/kvh3d/taures on the first step)."""

    nz, ny, nx = state.theta.shape
    z3 = jnp.zeros((nz + 1, ny, nx), F32)
    z2 = jnp.zeros((ny, nx), F32)
    return CamUwCarry(kvm3d=z3, kvh3d=z3, tauresx2d=z2, tauresy2d=z2, tke_pbl=z3)
