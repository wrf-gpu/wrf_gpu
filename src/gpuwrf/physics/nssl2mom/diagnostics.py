"""NSSL 2-moment diagnostics: radar reflectivity (``radardd02``) and effective radii.

Port of WRF ``phys/module_mp_nssl_2mom.F`` (pristine, sha256 29f42e76...):

* ``radardd02`` (module lines 8726-9735), driver call lines 3319-3335 (``microp = 'ZVD'``,
  ``temk = t0`` after smallvalues, ``db = dn1``).  Only the reflectivity factor feeds ``dbz``;
  the single-moment intercept diagnostics (``xcnos/xcnoh/dads/dadh``) have no effect on the
  two-moment path and are not evaluated.
* ``calc_eff_radius`` (module lines 5867-6172) with the driver initialisation and clamps
  (lines 3356-3386): only ``t1/t2/t3`` (cloud/ice/snow) are produced in the WRF-default call
  (``has_reqr/has_reqg/has_reqh`` absent -> ``f_t4/f_t5/f_t6 = .false.``).

Both see the DENSITY-SCALED state after ``smallvalues`` (S5): numbers in #/m3, volumes m3/m3.
Dead default-config branches: 3-moment Z (``lzr/lzh/lzhl = 0``), liquid on ice
(``lsw/lhw/lhlw = 0``), wet snow/graupel/hail options (``iusewetsnow = 0``,
``iusewetgraupel = 1`` without ``lhw``, ``iusewethail = 0``), snow volume (``lvs = 0``),
``idbzci = 1`` (spherical-ice reflectivity), ``ipconc = 5``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .indices import LC, LH, LHL, LI, LNC, LNH, LNHL, LNI, LNR, LNS, LR, LS, LVH, LVHL
from .mathfun import gamma_sp

F64 = jnp.float64


def _g1_alpha(alpha, nR):
    a = nR(alpha)
    return ((nR(6.0) + a) * (nR(5.0) + a) * (nR(4.0) + a)) / ((nR(3.0) + a) * (nR(2.0) + a) * (nR(1.0) + a))


def radardd02(an, dn, C, prec):
    """Equivalent radar reflectivity (dBZ) of the NSSL two-moment state (``dbz2d``)."""
    R, nR = prec.R, prec.nR
    an = jnp.asarray(an, R)
    db = jnp.asarray(dn, R)
    zero = jnp.zeros((), R)
    pi = nR(4.0) * np.arctan(nR(1.0))  # local pi = 4.0*ATan(1.)
    rwdn = nR(1000.0)
    qrmin, qsmin, qhmin, qhlmin = (nR(C.qxmin[i]) for i in (LR, LS, LH, LHL))
    cxmin = nR(C.cxmin)
    e18 = np.float64(nR(1.e18))
    # (6./(pi*1000.))**2 with the runtime REAL pi
    fac = (nR(6.) / (pi * nR(1000.))) ** 2
    facD = np.float64(fac)

    # ---- rain (ipconc = 5, lzr = 0, imurain = 1)
    g1r = np.float64(_g1_alpha(C.alphar, nR))
    rain = (an[LR] >= qrmin) & (an[LNR] > nR(1.e-3))
    nr_safe = jnp.where(rain, an[LNR], nR(1.0)).astype(F64)
    zx = g1r * ((db * an[LR]) ** 2).astype(F64) / nr_safe
    ze = e18 * zx * facD
    dtmp = jnp.where(rain, ze.astype(R), zero)

    # ---- snow (ipconc >= 4, Cox 1988 mass relation; qxw = 0)
    ksq = nR(0.189)
    qxw = nR(0.0)
    k1 = nR(1.e18) * nR(323.3226) * (nR(0.106214) * nR(0.106214))
    kden = nR(917.) * nR(917.)
    k43 = nR(np.float64(nR(1.0) + nR(C.snu)) ** np.float64(nR(4.) / nR(3.)))
    snow = (an[LS] >= qsmin) & (an[LNS] > nR(1.e-7))
    ns_safe = jnp.where(snow, an[LNS], nR(1.0))
    gtmp = (k1 * (ksq * an[LS] + (nR(1.) - ksq) * qxw) * an[LS] * db ** 2 * nR(C.gsnow73)
            / (ns_safe * kden * nR(C.gsnow1) * k43))
    dtmp = jnp.where(snow, dtmp + gtmp, dtmp)

    # ---- cloud ice (idbzci = 1: spherical ice, density 900)
    ice = (an[LI] > nR(C.qxmin[LI])) & (an[LNI] > nR(1.0))
    ni_safe = jnp.where(ice, an[LNI], nR(1.0))
    vr = db * an[LI] / (nR(900.) * ni_safe)
    cinu = nR(C.cinu)
    k2 = nR(0.224) * nR(3.6e18) * (cinu + nR(2.))
    k3 = (nR(900.) / nR(1000.)) ** 2
    dtmp = jnp.where(ice, dtmp + k2 * an[LNI] * vr ** 2 / (cinu + nR(1.)) * k3, dtmp)

    # ---- graupel (izieg = 1, ipconc >= 5, lzh = 0, lvh > 1)
    grp = (an[LH] >= qhmin) & (an[LNH] >= cxmin)
    vpos = an[LVH] > 0.0
    hwdn = jnp.where(vpos, db * an[LH] / jnp.where(vpos, an[LVH], nR(1.0)), nR(500.))
    hwdn = jnp.where(vpos, jnp.minimum(nR(900.), jnp.maximum(nR(100.), hwdn)), hwdn)
    chw = an[LNH]
    grp = grp & (chw > 0.0)
    chw_safe = jnp.where(grp, chw, nR(1.0))
    xvh = db * an[LH] / (hwdn * jnp.maximum(nR(1.0e-3), chw_safe))
    out = (xvh < nR(C.xvhmn)) | (xvh > nR(C.xvhmx))
    xvh = jnp.minimum(nR(C.xvhmx), jnp.maximum(nR(C.xvhmn), xvh))
    chw = jnp.where(out, db * an[LH] / (xvh * hwdn), chw_safe)
    qh = an[LH]
    g1h = np.float64(_g1_alpha(C.alphah, nR))
    zx = g1h * (db ** 2).astype(F64) * (nR(0.224) * qh + nR(0.776) * qxw).astype(F64) * qh.astype(F64) / chw.astype(F64)
    ze = e18 * zx * facD
    dtmp = jnp.where(grp, (dtmp.astype(F64) + ze).astype(R), dtmp)

    # ---- hail (izieg = 1, lhl > 1, ipconc >= 5, lzhl = 0, lvhl > 1)
    hl = (an[LHL] >= qhlmin) & (an[LNHL] > 0.0)
    vpos = an[LVHL] > 0.0
    hldn = jnp.where(vpos, db * an[LHL] / jnp.where(vpos, an[LVHL], nR(1.0)), nR(900.))
    hldn = jnp.where(vpos, jnp.minimum(nR(900.), jnp.maximum(nR(300.), hldn)), hldn)
    chl_safe = jnp.where(hl, an[LNHL], nR(1.0))
    xvhl = db * an[LHL] / (hldn * jnp.maximum(nR(1.0e-9), chl_safe))
    out = (xvhl < nR(C.xvhlmn)) | (xvhl > nR(C.xvhlmx))
    xvhl = jnp.minimum(nR(C.xvhlmx), jnp.maximum(nR(C.xvhlmn), xvhl))
    chl = jnp.where(out, db * an[LHL] / (xvhl * hldn), chl_safe)
    g1hl = np.float64(_g1_alpha(C.alphahl, nR))
    zx = (g1hl * (db ** 2).astype(F64) * (nR(0.224) * an[LHL] + nR(0.776) * qxw).astype(F64)
          * an[LHL].astype(F64) / chl.astype(F64))
    ze = e18 * zx * facD
    dtmp = jnp.where(hl, (dtmp.astype(F64) + ze).astype(R), dtmp)

    pos = dtmp > 0.0
    dbz = jnp.maximum(zero, nR(10.0) * jnp.log10(jnp.where(pos, dtmp, nR(1.0))))
    return jnp.where(pos, dbz, zero)


def _gsp(x, prec):
    """Compile-time REAL gamma_sp constant (init-like); concrete even inside jit."""
    import jax

    with jax.ensure_compile_time_eval():
        return prec.nR(np.asarray(gamma_sp(prec.nR(x), prec.R)))


def eff_radius(an, dn, C, prec):
    """Driver effective radii (re_cloud, re_ice, re_snow) in m, clamps of driver lines 3382-3384."""
    R, nR = prec.R, prec.nR
    an = jnp.asarray(an, R)
    rho0 = jnp.asarray(dn, R)
    cnu, cinu, snu = nR(C.cnu), nR(C.cinu), nR(C.snu)
    one = nR(1.)
    five3 = nR(5.) / nR(3.)
    gamc1, gamc2 = _gsp(nR(2.) + cnu, prec), one
    gami1, gami2 = _gsp(nR(2.) + cinu, prec), one
    gams1, gams2 = _gsp(nR(2.) + snu, prec), _gsp(one + snu, prec)
    factor_c = (one + cnu) * _gsp(one + cnu, prec) / _gsp(five3 + cnu, prec)
    factor_i = (one + cinu) * _gsp(one + cinu, prec) / _gsp(five3 + cinu, prec)
    factor_s = (one + snu) * _gsp(one + snu, prec) / _gsp(five3 + snu, prec)
    pi6 = nR(C.pi) / nR(6.)
    third = nR(1.) / nR(3.)
    half = nR(0.5)
    cxmin = nR(C.cxmin)

    def radius(il, lnx, gam1, gam2, factor, init):
        qx = jnp.maximum(an[il], nR(0.0))
        cx = jnp.maximum(an[lnx], nR(0.0))
        ok = (qx > nR(C.qxmin[il])) & (cx > cxmin)
        qs = jnp.where(ok, qx, one)
        lam = ((cx * pi6 * nR(C.xdn0[il]) * gam1) / (qs * rho0 * gam2)) ** third
        lam = jnp.where(ok, lam, one)
        return jnp.where(ok, half * factor / lam, nR(init))

    t1 = radius(LC, LNC, gamc1, gamc2, factor_c, 2.51e-6)
    t2 = radius(LI, LNI, gami1, gami2, factor_i, 10.01e-6)
    t3 = radius(LS, LNS, gams1, gams2, factor_s, 25.e-6)
    re_cloud = jnp.maximum(nR(2.51e-6), jnp.minimum(t1, nR(50.e-6)))
    re_ice = jnp.maximum(nR(10.01e-6), jnp.minimum(t2, nR(125.e-6)))
    re_snow = jnp.maximum(nR(25.e-6), jnp.minimum(t3, nR(999.e-6)))
    return re_cloud, re_ice, re_snow
