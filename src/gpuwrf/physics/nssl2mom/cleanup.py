"""NSSL 2-moment state cleanup routines: ``calcnfromq`` and ``smallvalues``.

Port of WRF ``phys/module_mp_nssl_2mom.F`` (pristine, sha256 29f42e76...):

* ``calcnfromq`` (module lines 5298-5657), called by ``nssl_2mom_driver`` (line 3116) only when
  ``itimestep == 1`` with the plain argument list ``(nx,ny,nz,an,na,nor,nor,dn1)`` -> none of the
  optional inputs is present, ``mixconv = 1``, ``invertccn_flag`` absent, ``cwmasin`` absent.
* ``smallvalues`` (module lines 12054-12609), called after NUCOND (driver lines 3258-3263).

Both act on the DENSITY-SCALED driver state (indices 9..18 in #/m3 or m3/m3) and are pointwise
(no vertical coupling).  ``an`` is a stack with leading axis ``NA + 1`` (Fortran species index,
index 0 unused) and arbitrary trailing point shape; ``dn`` (air density) has the point shape.

Branches dead in the WRF-default mp=18 configuration are omitted (flag named at the site):
3-moment reflectivity moments (``lzr/lzh/lzhl = 0``), liquid-on-ice (``lsw/lhw/lhlw = 0``),
snow volume (``lvs = 0``), ``iresetmoments = 0``, ``imorrgdnglimit = 0``, coarse/nucleus CCN
(``lccnaco/lccnanu = 0``), ``invertccn = .false.``.
"""

from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from .indices import (LC, LCCN, LCCNA, LH, LHL, LI, LNC, LNH, LNHL, LNI, LNR, LNS, LR, LS, LT,
                      LV, LVH, LVHL)

F64 = jnp.float64


# ------------------------------------------------------------------ compile-time constant folding
def _lit(x, nR):
    """A Fortran default-REAL literal of the build (fp32: nearest float32)."""
    return nR(x)


def _ipow(x, n: int, nR):
    """Constant-folded ``x**n`` (gfortran folds with a correctly rounded integer power)."""
    return nR(float(Fraction(float(x)) ** n))


def _calcnfromq_params(C, nR):
    """REAL parameters of calcnfromq (module lines 5331-5341), folded per operation in REAL."""
    pi = nR(C.pi)
    xn0s, xn0r, xn0h, xn0hl = (_lit(v, nR) for v in (3.0e8, 8.0e6, 2.0e5, 4.0e4))
    xdnr, xdns, xdnh, xdnhl = (_lit(v, nR) for v in (1000., 100., 700.0, 900.0))
    one = _lit(1., nR)
    p = {}
    p['xn0s'], p['xn0r'], p['xn0h'], p['xn0hl'] = xn0s, xn0r, xn0h, xn0hl
    p['xdnh'], p['xdnhl'] = xdnh, xdnhl
    p['zhlfac'] = one / (pi * xdnhl * xn0hl)
    p['zhfac'] = one / (pi * xdnh * xn0h)
    p['zrfac'] = one / (pi * xdnr * xn0r)
    p['zsfac'] = one / (pi * xdns * xn0s)
    p['g0'] = (_lit(6.0, nR) * _lit(5.0, nR) * _lit(4.0, nR)) / (_lit(3.0, nR) * _lit(2.0, nR) * _lit(1.0, nR))
    p['xims'] = _lit(900., nR) * _lit(0.523599, nR) * _ipow(_lit(2., nR) * _lit(50.e-6, nR), 3, nR)
    p['xgms'] = xdnh * _lit(0.523599, nR) * _ipow(_lit(300.e-6, nR), 3, nR)
    p['cwmas09'] = _lit(1000., nR) * _lit(0.523599, nR) * _ipow(_lit(2., nR) * _lit(9.e-6, nR), 3, nR)
    return p


def _g1_alpha(alpha, nR):
    """(6+a)(5+a)(4+a)/((3+a)(2+a)(1+a)) evaluated in REAL (alpha is a module REAL)."""
    a = nR(alpha)
    return ((nR(6.0) + a) * (nR(5.0) + a) * (nR(4.0) + a)) / ((nR(3.0) + a) * (nR(2.0) + a) * (nR(1.0) + a))


def _set(an, il, val):
    return an.at[il].set(val)


# ------------------------------------------------------------------------------- calcnfromq
def calcnfromq(an, dn, C, prec):
    """``calcnfromq`` for the driver's cold-start call (S0 -> S1).

    Derives missing number concentrations (and graupel/hail volume) from mass with single-moment
    intercept assumptions and folds sub-threshold species back to vapour.  Returns the new ``an``.
    """
    R, nR = prec.R, prec.nR
    an = jnp.asarray(an, R)
    dn = jnp.asarray(dn, R)
    p = _calcnfromq_params(C, nR)
    cxmin = nR(C.cxmin)
    qxmin = C.qxmin
    qxi = C.qxmin_init
    qccn = nR(C.qccn)
    cwmasinv = nR(1.0) / p['cwmas09']
    zero = jnp.zeros((), R)
    # DOUBLE locals g1h/g1hl/g1r/g1s hold REAL-evaluated expressions (imurain == 1)
    g1h = np.float64(_g1_alpha(C.alphah, nR))
    g1hl = np.float64(_g1_alpha(C.alphahl, nR))
    if C.imurain == 3:
        g1r = np.float64((nR(C.rnu) + nR(2.0)) / (nR(C.rnu) + nR(1.0)))
    else:
        g1r = np.float64(_g1_alpha(C.alphar, nR))
    g1s = np.float64((nR(C.snu) + nR(2.0)) / (nR(C.snu) + nR(1.0)))
    g0 = np.float64(p['g0'])
    dnD = dn.astype(F64)

    def single_moment_n(q, zfac, xn0, g1):
        # laminv1 = (dn*q*zfac)**(0.25); n1 = laminv1*xn0; nrx = n1*g1/g0   (all DOUBLE)
        laminv1 = (dnD * q * np.float64(zfac)) ** 0.25
        n1 = laminv1 * np.float64(xn0)
        return n1 * g1 / g0

    # --- cloud droplets
    c1 = (an[LNC] <= cxmin) & (an[LC] > nR(qxi[LC]))
    newnc = jnp.minimum(qccn, an[LC] * cwmasinv) * dn
    c2 = (~c1) & ((an[LC] <= nR(qxmin[LC])) | ((an[LNC] <= cxmin) & (an[LC] <= nR(qxi[LC]))))
    # lccna > 1 and .not. invertccn: activated CCN gains the new droplets
    an = _set(an, LCCNA, jnp.where(c1, an[LCCNA] + newnc, an[LCCNA]))
    an = _set(an, LNC, jnp.where(c1, newnc, jnp.where(c2, zero, an[LNC])))
    an = _set(an, LV, jnp.where(c2, an[LV] + an[LC], an[LV]))
    an = _set(an, LC, jnp.where(c2, zero, an[LC]))

    # --- cloud ice
    c1 = (an[LNI] <= cxmin) & (an[LI] > nR(qxi[LI]))
    c2 = (~c1) & ((an[LI] <= nR(qxmin[LI])) | ((an[LNI] <= cxmin) & (an[LI] <= nR(qxi[LI]))))
    an = _set(an, LNI, jnp.where(c1, dn * an[LI] / p['xims'], jnp.where(c2, zero, an[LNI])))
    an = _set(an, LV, jnp.where(c2, an[LV] + an[LI], an[LV]))
    an = _set(an, LI, jnp.where(c2, zero, an[LI]))

    # --- rain (lzr = 0: no reflectivity moment)
    tenth_cx = nR(0.1) * cxmin
    c1 = (an[LNR] <= tenth_cx) & (an[LR] > nR(qxi[LR]))
    c2 = (~c1) & ((an[LR] <= nR(qxmin[LR])) | ((an[LNR] <= cxmin) & (an[LR] <= nR(qxi[LR]))))
    nrx = single_moment_n(an[LR].astype(F64), p['zrfac'], p['xn0r'], g1r).astype(R)
    an = _set(an, LNR, jnp.where(c1, nrx, jnp.where(c2, zero, an[LNR])))
    an = _set(an, LV, jnp.where(c2, an[LV] + an[LR], an[LV]))
    an = _set(an, LR, jnp.where(c2, zero, an[LR]))

    # --- snow
    c1 = (an[LNS] <= tenth_cx) & (an[LS] > nR(qxi[LS]))
    c2 = (~c1) & ((an[LS] <= nR(qxmin[LS])) | ((an[LNS] <= cxmin) & (an[LS] <= nR(qxi[LS]))))
    nrx = single_moment_n(an[LS].astype(F64), p['zsfac'], p['xn0s'], g1s).astype(R)
    an = _set(an, LNS, jnp.where(c1, nrx, jnp.where(c2, zero, an[LNS])))
    an = _set(an, LV, jnp.where(c2, an[LV] + an[LS], an[LV]))
    an = _set(an, LS, jnp.where(c2, zero, an[LS]))

    # --- graupel (lzh = 0)
    c1 = (an[LNH] <= tenth_cx) & (an[LH] > nR(qxi[LH]))
    c2 = (~c1) & ((an[LH] <= nR(qxmin[LH])) | ((an[LNH] <= cxmin) & (an[LH] <= nR(qxi[LH]))))
    vh = jnp.where(c1 & (an[LVH] <= 0.0), an[LH] / p['xdnh'], an[LVH])
    q = an[LH].astype(F64)
    nrx = single_moment_n(q, p['zhfac'], p['xn0h'], g1h)
    nrx2 = dnD * q / np.float64(p['xgms'])
    nrx = jnp.minimum(nrx, nrx2)
    keep = nrx > np.float64(cxmin)
    an = _set(an, LVH, jnp.where(c1 & ~keep, zero, vh))
    an = _set(an, LNH, jnp.where(c1, jnp.where(keep, nrx.astype(R), zero), an[LNH]))
    an = _set(an, LV, jnp.where(c2, an[LV] + an[LH], an[LV]))
    an = _set(an, LH, jnp.where((c1 & ~keep) | c2, zero, an[LH]))

    # --- hail (lzhl = 0)
    c1 = (an[LNHL] <= tenth_cx) & (an[LHL] > nR(qxi[LHL]))
    c2 = (~c1) & ((an[LHL] <= nR(qxmin[LHL])) | ((an[LNHL] <= cxmin) & (an[LHL] <= nR(qxi[LHL]))))
    an = _set(an, LVHL, jnp.where(c1 & (an[LVHL] <= 0.0), an[LHL] / p['xdnhl'], an[LVHL]))
    nrx = single_moment_n(an[LHL].astype(F64), p['zhlfac'], p['xn0hl'], g1hl).astype(R)
    an = _set(an, LNHL, jnp.where(c1, nrx, an[LNHL]))
    an = _set(an, LV, jnp.where(c2, an[LV] + an[LHL], an[LV]))
    an = _set(an, LHL, jnp.where(c2, zero, an[LHL]))
    return an


# ------------------------------------------------------------------------------ smallvalues
def _rimed_ice_check(an, dn, C, nR, R, lq, lnq, lvq, keep, rho_default, alpha, cnomin, is_hail):
    """Density / mean-volume / intercept checks for graupel (lq=lh) or hail (lq=lhl) on points
    that keep the species (module lines 12195-12279 hail, 12353-12434 graupel)."""
    zero = jnp.zeros((), R)
    xdnmn, xdnmx, xdn0 = nR(C.xdnmn[lq]), nR(C.xdnmx[lq]), nR(C.xdn0[lq])
    q = an[lq]
    volpos = an[lvq] > 0.0
    safe_v = jnp.where(volpos, an[lvq], nR(1.0))
    tmp = jnp.where(volpos, dn * q / safe_v, nR(rho_default))
    vol = jnp.where(keep & ~volpos, dn * q / tmp, an[lvq])
    low = tmp < xdnmn
    tmp = jnp.where(low, jnp.maximum(xdnmn, tmp), tmp)
    vol = jnp.where(keep & low, dn * q / tmp, vol)
    high = tmp > xdnmx  # lhw/lhlw <= 0: no liquid allowed on graupel/hail
    tmp = jnp.where(high, jnp.minimum(xdnmx, tmp), tmp)
    vol = jnp.where(keep & high, dn * q / tmp, vol)
    an = an.at[lvq].set(vol)
    # hwdn from the (possibly updated) volume
    volpos = an[lvq] > 0.0
    safe_v = jnp.where(volpos, an[lvq], nR(1.0))
    hwdn = jnp.where(volpos, dn * q / safe_v, xdn0)
    hwdn = jnp.maximum(xdnmn, hwdn)
    # mean-volume limits (ipconc >= 5)
    act = keep & (q > nR(C.qxmin[lq]))
    safe_n = jnp.where(act, an[lnq], nR(1.0))
    xvol = dn * q / (hwdn * safe_n)
    out = (xvol < nR(C.xvmn[lq])) | (xvol > nR(C.xvmx[lq]))
    xvol_c = jnp.minimum(nR(C.xvmx[lq]), jnp.maximum(nR(C.xvmn[lq]), xvol))
    an = an.at[lnq].set(jnp.where(act & out, dn * q / (xvol_c * hwdn), an[lnq]))
    # minimum intercept (ipconc == 5, alpha <= 0.1, no 3rd moment): static on the namelist alpha
    if nR(alpha) <= nR(0.1):  # default config: active for graupel (alphah=0), dead for hail (alphahl=1)
        pi = nR(C.pi)
        third = nR(1.) / nR(3.)
        safe_q = jnp.where(act, dn * q, nR(1.0))
        if is_hail:  # hail block recomputes the density without the xdnmn floor (line 12268)
            hwdn = dn * q / jnp.where(act, an[lvq], nR(1.0))
        tmp2 = (hwdn * an[lnq]) / safe_q
        tmpg = an[lnq] * (tmp2 * pi) ** third
        small = act & (tmpg < nR(cnomin))
        tmp3 = ((hwdn) / safe_q * pi) ** third
        an = an.at[lnq].set(jnp.where(small, (nR(cnomin) / tmp3) ** (nR(3.) / nR(4.)), an[lnq]))
    return an


def smallvalues(an, t0, dn, wn, t77, dtp, C, prec, flag_qndrop=False):
    """``smallvalues`` (S4 -> S5).  Returns ``(an, t0)``; t0 is reset to ``an[lt]*t77`` first.

    ``wn`` is unused by the default configuration (kept for the Fortran signature).
    """
    del wn
    R, nR = prec.R, prec.nR
    an = jnp.asarray(an, R)
    dn = jnp.asarray(dn, R)
    t77 = jnp.asarray(t77, R)
    zero = jnp.zeros((), R)
    t0 = an[LT] * t77
    frac = nR(1.0)
    qxmin = C.qxmin

    # zerocx (iresetmoments = 0)
    zc = {LC: (an[LNC] <= 0.0) & (not flag_qndrop)}
    for il, lnx in ((LR, LNR), (LI, LNI), (LS, LNS), (LH, LNH), (LHL, LNHL)):
        zc[il] = an[lnx] <= 0.0

    # ---- hail (lhl > 1, lzhl = 0)
    rm = (an[LHL] < frac * nR(qxmin[LHL])) | zc[LHL]
    an = an.at[LV].set(jnp.where(rm, an[LV] + an[LHL], an[LV]))
    an = an.at[LHL].set(jnp.where(rm, zero, an[LHL]))
    an = an.at[LNHL].set(jnp.where(rm, zero, an[LNHL]))
    an = an.at[LVHL].set(jnp.where(rm, zero, an[LVHL]))
    an = _rimed_ice_check(an, dn, C, nR, R, LHL, LNHL, LVHL, ~rm, C.rho_qhl, C.alphahl, C.cnohlmn, True)

    # ---- graupel (lzh = 0)
    rm = (an[LH] < frac * nR(qxmin[LH])) | zc[LH]
    an = an.at[LV].set(jnp.where(rm, an[LV] + an[LH], an[LV]))
    an = an.at[LH].set(jnp.where(rm, zero, an[LH]))
    an = an.at[LNH].set(jnp.where(rm, zero, an[LNH]))
    an = an.at[LVH].set(jnp.where(rm, zero, an[LVH]))
    an = _rimed_ice_check(an, dn, C, nR, R, LH, LNH, LVH, ~rm, C.rho_qh, C.alphah, C.cnohmn, False)
    # imorrgdnglimit = 0: no characteristic-diameter limit

    # ---- snow (both temperature branches perform the same updates; lvs = lsw = 0)
    rm = (an[LS] < frac * nR(qxmin[LS])) | zc[LS]
    an = an.at[LV].set(jnp.where(rm, an[LV] + an[LS], an[LV]))
    an = an.at[LS].set(jnp.where(rm, zero, an[LS]))
    an = an.at[LNS].set(jnp.where(rm, zero, an[LNS]))

    # ---- rain (lzr = 0)
    rm = (an[LR] < frac * nR(qxmin[LR])) | zc[LR]
    an = an.at[LV].set(jnp.where(rm, an[LV] + an[LR], an[LV]))
    an = an.at[LR].set(jnp.where(rm, zero, an[LR]))
    an = an.at[LNR].set(jnp.where(rm, zero, an[LNR]))

    # ---- cloud ice
    rm = (an[LI] <= frac * nR(qxmin[LI])) | zc[LI]
    an = an.at[LV].set(jnp.where(rm, an[LV] + an[LI], an[LV]))
    an = an.at[LI].set(jnp.where(rm, zero, an[LI]))
    an = an.at[LNI].set(jnp.where(rm, zero, an[LNI]))

    # ---- cloud droplets (irenuc = 5 -> activated CCN lccna)
    rm = (an[LC] <= frac * nR(qxmin[LC])) | zc[LC]
    an = an.at[LV].set(jnp.where(rm, an[LV] + an[LC], an[LV]))
    an = an.at[LC].set(jnp.where(rm, zero, an[LC]))
    tmp = jnp.maximum(zero, an[LNC])
    an = an.at[LCCNA].set(jnp.where(rm, jnp.maximum(zero, an[LCCNA] - tmp), an[LCCNA]))
    an = an.at[LNC].set(jnp.where(rm, zero, an[LNC]))
    an = an.at[LCCN].set(jnp.where(rm, jnp.maximum(zero, an[LCCN]), an[LCCN]))
    if C.restoreccn:  # lccna > 0: exponential restore of activated CCN outside ice/snow
        tmp = an[LI] + an[LS]
        decay = rm & (tmp < nR(qxmin[LI])) & (an[LCCNA] > 1.0)
        fac = jnp.exp(-jnp.asarray(dtp, R) / nR(C.ccntimeconst))
        an = an.at[LCCNA].set(jnp.where(decay, an[LCCNA] * fac, an[LCCNA]))
    return an, t0
