"""NSSL 2-moment sedimentation: ``sediment1d`` (module lines 4425-4859) with ``fallout1d`` (4873-4978),
``calczgr1d`` (4983-5095) and ``calcnfromz1d`` (5105-5285), default mp=18 configuration.

Configuration facts used (module defaults + nssl_2mom_init): itfall=0 (first-order upwind),
do_accurate_sedimentation=.false. (fall speeds from ONE ziegfall1d call reused for every
substep and species), vtmaxsed=70, infall=4 with linfall = (lc 0, lr 4, li 0, ls 2, lh 4, lhl 4),
irfall=4, isfall=2, imurain=1, ido(:)=1, ipconc=5 >= ipc(:), lz(:)=0, graupel/hail volume on.
Driver infdo (lines 2794-2804): infall /= 1 -> 1, Any(linfall >= 3) -> infdo = 2.

Public entry::

    sediment1d(an, t0, t7, dn, dz, dtp, C, prec, infdo=2) -> (an_new, xfall)

``an`` (NA+1, ..., nz) density-scaled stack exactly as the driver holds it (k = 0 surface),
``t0`` temperature, ``t7`` (unused: ipconc=0 only), ``dn`` air density, ``dz`` layer depth,
``dtp`` time step (python float or scalar).  ``xfall`` (NA+1, ...) holds the surface flux
accumulators indexed by FORTRAN species index (mass il, number ln(il), volume lvol(il)) as used by
the driver precipitation binding (lines 3153-3206).  Column-dependent substep counts ``ndfall`` are
handled with a ``lax.fori_loop`` to the maximum and per-column masks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from .fallspeed import ziegfall1d
from .indices import LC, LH, LHL, LI, LN, LR, LS, LVOL, NA

F64 = jnp.float64

_SPECIES = (LC, LR, LI, LS, LH, LHL)
_ZSPECIES = (LR, LH, LHL)  # linfall 3/4 (method I+II); il >= lr


def _fallout(a, vt, db1, dtz, dtptmp, dtfrac, idens):
    """fallout1d for one quantity on columns (..., nz); returns (a_new, surface_flux)."""
    q = a * vt * db1 if idens else a * vt
    qp = jnp.concatenate([q[..., 1:], jnp.zeros_like(q[..., :1])], axis=-1)
    a_new = a + dtptmp[..., None] * dtz * (qp - q)
    flux = a[..., 0] * vt[..., 0] * dtfrac
    return a_new, flux


def _xdn_vol(an, il, db, C, R):
    """Particle density used by calczgr1d/calcnfromz1d (lvol > 1: from volume, else xdn0)."""
    f = lambda v: jnp.asarray(v, R)  # noqa: E731
    rho_qx = f(C.xdn0[il])
    if il in LVOL:
        av = an[LVOL[il]]
        pos = av > 0.0
        xdn = db * an[il] / jnp.where(pos, av, 1.0)
        xdn = jnp.minimum(f(900.), jnp.maximum(f(C.hdnmn), xdn))
        xdn = jnp.where(pos, xdn, rho_qx)
    else:
        xdn = jnp.broadcast_to(rho_qx, an[il].shape)
    if il == LR:
        xdn = jnp.broadcast_to(f(1000.), an[il].shape)
    return xdn


def calczgr1d(an, db, il, C, prec):
    """Reflectivity moment of species il (imurain=1 / graupel / hail branch), REAL."""
    R = prec.R
    f = lambda v: jnp.asarray(v, R)  # noqa: E731
    ln = LN[il]
    a, n = an[il], an[ln]
    ok = (a > f(C.qxmin[il])) & (n > 1.e-15)
    xdn = _xdn_vol(an, il, db, C, R)
    nsafe = jnp.where(ok, n, 1.0)
    xv = db * a / (xdn * nsafe)
    chw = n
    oob = (xv < f(C.xvmn[il])) | (xv > f(C.xvmx[il]))
    xvc = jnp.minimum(f(C.xvmx[il]), jnp.maximum(f(C.xvmn[il]), xv))
    chw = jnp.where(oob, db * a / (xvc * xdn), chw)
    alpha = f(C.dnu[il])
    g1 = (6.0 + alpha) * (5.0 + alpha) * (4.0 + alpha) / ((3.0 + alpha) * (2.0 + alpha) * (1.0 + alpha))
    zx = g1 * db ** 2 * a * a / jnp.where(ok, chw, 1.0)
    z = zx * (f(6.) / (f(C.pi) * 1000.)) ** 2
    return jnp.where(ok, z, f(0.0))


def calcnfromz1d(an, t0z, z0, db, t1, il, C, prec):
    """Number lower bound from the sedimented reflectivity (infall = 4, Mansell 2010 I+II).

    ``t0z`` sedimented Z, ``z0`` Z before the substep, ``t1`` number sedimented with the mass-
    weighted speed.  DOUBLE locals (zx, z, nrx, g1, chw, qr) as in the Fortran.  Returns new an[ln].
    """
    R = prec.R
    f = lambda v: jnp.asarray(v, R)  # noqa: E731
    ln = LN[il]
    a, n = an[il], an[ln]
    alpha = f(C.dnu[il])
    g1 = ((6.0 + alpha) * (5.0 + alpha) * (4.0 + alpha) / ((3.0 + alpha) * (2.0 + alpha) * (1.0 + alpha))).astype(F64)
    pos = t0z > 0.0
    xdn = _xdn_vol(an, il, db, C, R)
    xv = db * a / (xdn * n)
    chw = n.astype(F64)
    oob = (xv < f(C.xvmn[il])) | (xv > f(C.xvmx[il]))
    xvc = jnp.minimum(f(C.xvmx[il]), jnp.maximum(f(C.xvmn[il]), xv))
    chw = jnp.where(oob, (db * a / (xvc * xdn)).astype(F64), chw)
    c6 = (f(6.) / (f(C.pi) * 1000.)) ** 2
    a64 = a.astype(F64)
    db2 = (db ** 2).astype(F64)
    zx = g1 * db2 * a64 * a64 / chw
    z = zx * c6.astype(F64)
    t064 = t0z.astype(F64)
    cond = (z > t064) & (z > 0.0) & (t0z > z0)
    zx2 = (t0z / c6).astype(F64)
    nrx = g1 * db2 * a64 * a64 / zx2
    n_a = jnp.maximum(jnp.minimum(nrx.astype(R), t1), n)
    n_b = jnp.maximum(t1, n)
    return jnp.where(pos, jnp.where(cond, n_a, n_b), n)


def sediment1d(an, t0, t7, dn, dz, dtp, C, prec, infdo=2):
    """Port of SUBROUTINE sediment1d (see module docstring)."""
    del t7  # ipconc = 0 only
    R = prec.R
    f = lambda v: jnp.asarray(v, R)  # noqa: E731
    an = jnp.asarray(an, R)
    t0 = jnp.asarray(t0, R)
    db1 = jnp.asarray(dn, R)
    dz = jnp.asarray(dz, R)
    dtp = f(dtp)
    db1inv = 1. / db1
    rhovtzx = jnp.sqrt(f(C.rho00) * jnp.minimum(f(1.0) / f(0.05), db1inv))
    dzinv = 1. / dz  # driver dz2dinv = 1./dz
    dtz0 = dzinv
    dtz1 = dzinv * db1inv

    an, xvt = ziegfall1d(an, t0, db1, rhovtzx, C=C, prec=prec, infdo=infdo)
    xfall = jnp.zeros((NA + 1,) + t0.shape[:-1], R)
    vtcap = f(C.vtmaxsed)

    for il in _SPECIES:
        v1 = jnp.minimum(vtcap, xvt[(il, 1)])
        v2 = jnp.minimum(vtcap, xvt[(il, 2)])
        v3 = jnp.minimum(vtcap, xvt[(il, 3)])
        vtmax = jnp.max(jnp.maximum(jnp.maximum(v1 * dzinv, v2 * dzinv), v3 * dzinv), axis=-1)
        vtmax = jnp.maximum(f(0.0), vtmax)
        c = dtp * vtmax
        nd_big = jnp.where(dtp > 20.0, jnp.maximum(2, jnp.trunc(c / 0.7).astype(jnp.int32) + 1),
                           1 + jnp.trunc(c + 0.301).astype(jnp.int32))
        ndfall = jnp.where(c < 0.7, 1, nd_big)
        ndfall = jnp.where(vtmax == 0.0, 0, ndfall)
        dtptmp = jnp.where(ndfall > 1, dtp / jnp.maximum(ndfall, 1).astype(R), dtp)
        dtfrac = dtptmp / dtp
        ln = LN[il]
        lv = LVOL.get(il)

        def body(n, carry, il=il, ln=ln, lv=lv, v1=v1, v2=v2, v3=v3, ndfall=ndfall,
                 dtptmp=dtptmp, dtfrac=dtfrac):
            an, xfall = carry
            act = n < ndfall
            actk = act[..., None]

            def upd(an, xfall, idx, vt, idens):
                dtz = dtz1 if idens else dtz0
                a_new, fl = _fallout(an[idx], vt, db1, dtz, dtptmp, dtfrac, idens)
                an = an.at[idx].set(jnp.where(actk, a_new, an[idx]))
                xfall = xfall.at[idx].set(jnp.where(act, xfall[idx] + fl, xfall[idx]))
                return an, xfall

            z = calczgr1d(an, db1, il, C, prec) if il in _ZSPECIES else None
            an, xfall = upd(an, xfall, il, v1, True)              # mass
            if lv is not None:
                an, xfall = upd(an, xfall, lv, v1, False)         # particle volume
            tmpn = an[ln]
            an, xfall = upd(an, xfall, ln, v2, False)             # number (in = 2)
            if il in _ZSPECIES:
                t2_new, _ = _fallout(z, v3, db1, dtz0, dtptmp, dtfrac, False)
                t1_new, _ = _fallout(tmpn, v1, db1, dtz0, dtptmp, dtfrac, False)
                n_new = calcnfromz1d(an, t2_new, z, db1, t1_new, il, C, prec)
                an = an.at[ln].set(jnp.where(actk, n_new, an[ln]))
            elif il == LS:  # linfall = 2, isfall /= infall: method II
                t1_new, _ = _fallout(tmpn, v1, db1, dtz0, dtptmp, dtfrac, False)
                an = an.at[ln].set(jnp.where(actk, jnp.maximum(an[ln], t1_new), an[ln]))
            return an, xfall

        nmax = jnp.max(ndfall)
        an, xfall = lax.fori_loop(0, nmax, body, (an, xfall))
    return an, xfall


__all__ = ["sediment1d", "calczgr1d", "calcnfromz1d"]
_ = (jax, LI, LC)
