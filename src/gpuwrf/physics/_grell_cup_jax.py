"""JAX port of WRF's Grell 3D ensemble (``module_cu_g3.F``, cu_physics=5) and
Grell-Devenyi ensemble (``module_cu_gd.F``, cu_physics=93) cumulus schemes.

Line-faithful re-expression of the pristine WRF Fortran: every ``cup_*``
subroutine is a traceable per-column function (``jax.vmap``-ed over columns),
written on 1-based length ``kx+1`` arrays (index 0 unused) so ``a[k]`` mirrors
Fortran ``a(i,k)``.  Control flow is re-expressed without Python data-dependent
branches:

- first-crossing ``DO ... GO TO`` searches -> masked ``argmax`` + found flag;
- the ``cup_kbcon`` GO TO 31/32 retest loop and the downdraft-origin
  ``keep_going`` loop -> ``lax.while_loop``;
- vertical recurrences (cloud MSE/moisture, mass fluxes, work functions,
  running sums) -> ``lax.scan`` in the Fortran loop order, so every running sum
  is accumulated in the same order as WRF;
- the 3 x 3 x 16 closure ensemble -> fixed-shape arrays indexed like WRF's
  ``nall`` offsets; ``ierr`` short-circuits -> ``jnp.where`` gating on the
  traced error flag (compute-all, select-on-ierr), with WRF's ``ierr``
  transitions reproduced exactly.

The arithmetic follows WRF's operation order (left-to-right Fortran evaluation)
in the dtype of the inputs: fp64 inputs reproduce the ``-fdefault-real-8``
oracle, fp32 inputs the WRF REAL build.  The G3 horizontal pieces of ``G3DRV``
(3 x 3 neighbour omega/forcing ensemble, ``cugd_avedx`` forcing average) and
``conv_grell_spread3d`` (subsidence spreading, 1-2-1 smoothing, moistening and
heating limiters) are 2-D array operations in :func:`g3drv_tile`.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax

ENS4 = 9
MAXENS = 3
MAXENS2 = 3
MAXENS3 = 16
ENSDIM = MAXENS * MAXENS2 * MAXENS3  # maxiens = 1


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _kidx(kx):
    return jnp.arange(kx + 1, dtype=jnp.int32)


def _c(v, like):
    """Constant in the working dtype (constant-only subexpressions are folded
    by gfortran in the target REAL kind; reproduce that, not Python double)."""
    return jnp.asarray(v, dtype=like.dtype)


def _first_true(mask, kx):
    """(found, first index) of a boolean mask over 0..kx."""
    found = jnp.any(mask)
    first = jnp.argmax(mask).astype(jnp.int32)
    return found, first


def _scan_up(fn, init, kx, k_lo=1):
    """Run ``fn(carry, k) -> (carry, out)`` for k = k_lo..kx; outs padded to kx+1."""
    ks = jnp.arange(k_lo, kx + 1, dtype=jnp.int32)
    carry, outs = lax.scan(fn, init, ks)
    return carry, outs


def _pad_front(outs, n):
    """Prepend n rows of zeros (to re-align scan outputs with 1-based index)."""
    z = jnp.zeros((n,) + outs.shape[1:], outs.dtype)
    return jnp.concatenate([z, outs], axis=0)


# ---------------------------------------------------------------------------
# cup_env / cup_env_clev
# ---------------------------------------------------------------------------


def cup_env(t, q, p, z1, psur, tcrit, itest, xl, cp, kx, *, he_in=None, variant="g3"):
    """cup_env (computed unconditionally; callers gate on ierr).

    Returns (z, qes, he, hes, q_out); q_out differs from q only for GD
    (``if(q>qes) q=qes``, an INTENT(INOUT) clip of the caller's array).
    """
    one = _c(1.0, t)
    ht1 = xl / cp
    ht2 = _c(2.834e6, t) / cp
    be1 = _c(0.622, t) * ht1 / _c(0.286, t)
    ae1 = be1 / _c(273.0, t) + jnp.log(_c(610.71, t))
    be2 = _c(0.622, t) * ht2 / _c(0.286, t)
    ae2 = be2 / _c(273.0, t) + jnp.log(_c(610.71, t))
    iph2 = t <= tcrit
    ae = jnp.where(iph2, ae2, ae1)
    be = jnp.where(iph2, be2, be1)
    e = jnp.exp(ae - be / t)
    qes = 0.622 * e / (100.0 * p - e)
    qes = jnp.where(qes <= 1.0e-08, _c(1.0e-08, t), qes)
    if variant == "g3":
        qes = jnp.where(qes < q, q, qes)
        q_out = q
    else:
        q_out = jnp.where(q > qes, qes, q)
    tv = t + 0.608 * q_out * t
    if itest != 2:
        logp = jnp.log(jnp.where(_kidx(kx) >= 1, p, one))
        z1v = jnp.maximum(_c(0.0, t), z1) - (logp[1] - jnp.log(psur)) * 287.0 * tv[1] / 9.81

        def step(zprev, k):
            tvbar = 0.5 * tv[k] + 0.5 * tv[k - 1]
            zk = zprev - (logp[k] - logp[k - 1]) * 287.0 * tvbar / 9.81
            return zk, zk

        _, zs = _scan_up(step, z1v, kx, k_lo=2)
        z = jnp.concatenate([jnp.zeros((1,), t.dtype), z1v[None], zs])
    else:
        z = (he_in - 1004.0 * t - 2.5e6 * q_out) / 9.81
        z = jnp.maximum(_c(1.0e-3, t), z)
    if itest == 0:
        he = 9.81 * z + 1004.0 * t + 2.5e06 * q_out
    else:
        he = he_in
    hes = 9.81 * z + 1004.0 * t + 2.5e06 * qes
    he = jnp.where(he >= hes, hes, he)
    return z, qes, he, hes, q_out


def cup_env_clev(t, qes, q, he, hes, z, p, psur, z1, xl, rv, cp, kx):
    """cup_env_clev -> (qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup)."""
    sh = lambda a: jnp.concatenate([a[:1], a[:-1]])  # a[k-1]
    qes_cup = 0.5 * (sh(qes) + qes)
    q_cup = 0.5 * (sh(q) + q)
    hes_cup = 0.5 * (sh(hes) + hes)
    he_cup = 0.5 * (sh(he) + he)
    he_cup = jnp.where(he_cup > hes_cup, hes_cup, he_cup)
    z_cup = 0.5 * (sh(z) + z)
    p_cup = 0.5 * (sh(p) + p)
    t_cup = 0.5 * (sh(t) + t)
    # k = 1
    qes_cup = qes_cup.at[1].set(qes[1])
    q_cup = q_cup.at[1].set(q[1])
    hes_cup = hes_cup.at[1].set(hes[1])
    he_cup = he_cup.at[1].set(he[1])
    z_cup = z_cup.at[1].set(0.5 * (z[1] + z1))
    p_cup = p_cup.at[1].set(0.5 * (p[1] + psur))
    t_cup = t_cup.at[1].set(t[1])
    gamma_cup = (xl / cp) * (xl / (rv * t_cup * t_cup)) * qes_cup
    return qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup


# ---------------------------------------------------------------------------
# index searches
# ---------------------------------------------------------------------------


def cup_maximi(array, ks, ke, ierr, kx):
    """MAXX: last index of the running '>=' maximum over [ks, ke]; ks if ierr/empty."""
    k = _kidx(kx)
    rng = (k >= ks) & (k <= ke)
    neg = jnp.full_like(array, -jnp.inf)
    m = jnp.max(jnp.where(rng, array, neg))
    hit = rng & (array == m)
    last = (kx - jnp.argmax(hit[::-1])).astype(jnp.int32)
    nonempty = jnp.any(rng)
    out = jnp.where(nonempty, last, jnp.int32(ks))
    return jnp.where(ierr == 0, out, jnp.asarray(ks, jnp.int32)).astype(jnp.int32)


def cup_minimi(array, ks, kend, ierr, kx):
    """KT: first index of the strict running minimum over [ks, max(ks+1,kend)]."""
    k = _kidx(kx)
    kstop = jnp.maximum(ks + 1, kend)
    rng = (k >= ks) & (k <= kstop)
    pos = jnp.full_like(array, jnp.inf)
    m = jnp.min(jnp.where(rng, array, pos))
    hit = rng & (array == m)
    first = jnp.argmax(hit).astype(jnp.int32)
    return jnp.where(ierr == 0, first, ks).astype(jnp.int32)


def cup_kbcon(cap_inc, iloop, k22, he_cup, hes_cup, ierr, kbmax, p_cup, cap_max, kx,
              variant="g3"):
    """cup_kbcon GO TO 31/32 retest loop -> (kbcon, k22, ierr)."""
    k = _kidx(kx)
    if iloop == 5:
        plus_static = _c(25.0, he_cup)
    else:
        plus_static = None

    def hetest_of(k22v):
        h = he_cup[k22v]
        if iloop == 5:
            h = jnp.maximum(h, jnp.max(jnp.where(k <= k22v, he_cup, -jnp.inf)))
        return h

    # state: kbcon, k22, ierr, done
    def cond(s):
        return ~s[3]

    def body(s):
        kbcon, k22v, ierrv, done = s
        hetest = hetest_of(k22v)
        below = hetest < hes_cup[kbcon]
        # label 31: kbcon += 1; exceeding kbmax+2 terminates
        kb_inc = kbcon + 1
        over = kb_inc > kbmax + 2
        if variant == "g3":
            ierr_over = jnp.where(iloop != 4, 3, ierrv) if iloop != 4 else ierrv
        else:
            ierr_over = 3 if iloop < 4 else ierrv
        # not below: check cap
        stop1 = (kbcon - k22v) == 1
        if iloop == 5:
            stop1 = stop1 | ((kbcon - k22v) == 0)
        pbcdif = -p_cup[kbcon] + p_cup[k22v]
        if iloop == 4:
            plus = cap_max
        elif iloop == 5:
            plus = plus_static
        else:
            plus = jnp.maximum(_c(25.0, he_cup), cap_max - float(iloop - 1) * cap_inc)
        if iloop == 5:
            pbcdif = jnp.where(cap_max > 25.0, -p_cup[kbcon] + cap_max, pbcdif)
        capped = pbcdif > plus
        # transitions
        n_kbcon = jnp.where(below, jnp.where(over, kb_inc, kb_inc),
                            jnp.where(stop1, kbcon, jnp.where(capped, k22v + 1, kbcon)))
        n_k22 = jnp.where(below, k22v, jnp.where(~stop1 & capped, k22v + 1, k22v))
        n_ierr = jnp.where(below & over, ierr_over, ierrv)
        n_done = jnp.where(below, over, stop1 | ~capped)
        return (n_kbcon.astype(jnp.int32), n_k22.astype(jnp.int32),
                jnp.asarray(n_ierr, jnp.int32), n_done)

    start = (jnp.asarray(k22, jnp.int32), jnp.asarray(k22, jnp.int32),
             jnp.asarray(ierr, jnp.int32), ierr != 0)
    kbcon, k22o, ierro, _ = lax.while_loop(cond, body, start)
    kbcon = jnp.where(ierr != 0, 1, kbcon).astype(jnp.int32)
    k22o = jnp.where(ierr != 0, k22, k22o).astype(jnp.int32)
    return kbcon, k22o, ierro


def cup_ktop(ilo, dby, kbcon, ierr, kx, variant="g3"):
    """cup_ktop -> (ktop, dby (zeroed above ktop), ierr)."""
    k = _kidx(kx)
    rng = (k >= kbcon + 1) & (k <= kx - 1)
    found, kf = _first_true(rng & (dby <= 0.0), kx)
    ok = ierr == 0
    ktop = jnp.where(ok & found, kf - 1, 1).astype(jnp.int32)
    ierr_n = ierr
    if ilo == 1:
        ierr_n = jnp.where(ok & ~found, 5, ierr_n)
    dby = jnp.where(ok & found & (k >= ktop + 1) & (k <= kx), jnp.zeros_like(dby), dby)
    if variant == "g3":
        ierr_n = jnp.where(ok & found & (kbcon == ktop), 55, ierr_n)
    return ktop, dby, jnp.asarray(ierr_n, jnp.int32)


# ---------------------------------------------------------------------------
# updraft
# ---------------------------------------------------------------------------


def cup_up_he(k22, z_cup, cd, entr, he_cup, kbcon, he, hes_cup, kx, *, shallow=False):
    """cup_up_he -> (hkb, hc, dby) (values valid where the caller's ierr == 0)."""
    k = _kidx(kx)
    hkb = he_cup[k22]
    if shallow:
        hkb = jnp.maximum(hkb, jnp.max(jnp.where((k >= 1) & (k <= k22), he_cup, -jnp.inf)))
    hc0 = jnp.where((k >= 1) & (k <= k22), he_cup, jnp.zeros_like(he_cup))
    hc0 = jnp.where((k >= k22) & (k <= kbcon), hkb, hc0)
    dby0 = jnp.where(k == kbcon, hkb - hes_cup[kbcon], jnp.zeros_like(he_cup))

    def step(hprev, kk):
        dz = z_cup[kk] - z_cup[kk - 1]
        hn = (hprev * (1.0 - 0.5 * cd[kk] * dz) + entr * dz * he[kk - 1]) / (
            1.0 + entr * dz - 0.5 * cd[kk] * dz)
        active = kk > kbcon
        hk = jnp.where(active, hn, hc0[kk])
        return hk, (hk, jnp.where(active, hk - hes_cup[kk], dby0[kk]))

    _, (hcs, dbys) = _scan_up(step, hc0[1], kx, k_lo=2)
    hc = jnp.concatenate([hc0[:2], hcs])
    dby = jnp.concatenate([dby0[:2], dbys])
    return hkb, hc, dby


def cup_up_nms(z_cup, entr, cd, kbcon, ktop, k22, kx):
    k = _kidx(kx)
    zu0 = jnp.where((k >= k22) & (k <= kbcon), jnp.ones_like(z_cup), jnp.zeros_like(z_cup))

    def step(zprev, kk):
        dz = z_cup[kk] - z_cup[kk - 1]
        zn = zprev * (1.0 + (entr - cd[kk]) * dz)
        active = (kk >= kbcon + 1) & (kk <= ktop)
        zk = jnp.where(active, zn, zu0[kk])
        return zk, zk

    _, zs = _scan_up(step, zu0[1], kx, k_lo=2)
    return jnp.concatenate([zu0[:2], zs])


def cup_up_moisture(z_cup, kbcon, ktop, cd, dby, mentr_rate, q, gamma_cup, zu, qes_cup,
                    k22, qe_cup, xl, kx, *, c0):
    """cup_up_moisture -> (qc, qrc, pw, pwav, clw_all)."""
    k = _kidx(kx)
    qc0 = jnp.where((k >= k22) & (k <= kbcon - 1), qe_cup[k22], qes_cup)
    one = _c(1.0, z_cup)
    rxl = one / xl

    def step(carry, kk):
        qprev, pwav = carry
        dz = z_cup[kk] - z_cup[kk - 1]
        qcn = (qprev * (1.0 - 0.5 * cd[kk] * dz) + mentr_rate * dz * q[kk - 1]) / (
            1.0 + mentr_rate * dz - 0.5 * cd[kk] * dz)
        qrch = qes_cup[kk] + rxl * (gamma_cup[kk] / (1.0 + gamma_cup[kk])) * dby[kk]
        clw = qcn - qrch
        qrc = (qcn - qrch) / (1.0 + c0 * dz * zu[kk])
        qrc = jnp.where(qrc < 0.0, jnp.zeros_like(qrc), qrc)
        pw = c0 * dz * qrc * zu[kk]
        qcf = qrc + qrch
        active = (kk >= kbcon) & (kk <= ktop)
        zero = jnp.zeros_like(qcn)
        qck = jnp.where(active, qcf, qc0[kk])
        pwav_n = jnp.where(active, pwav + pw, pwav)
        return (qck, pwav_n), (qck, jnp.where(active, qrc, zero), jnp.where(active, pw, zero),
                               jnp.where(active, clw, zero))

    (_, pwav), (qcs, qrcs, pws, clws) = _scan_up(step, (qc0[1], _c(0.0, z_cup)), kx, k_lo=2)
    zero2 = jnp.zeros((2,), z_cup.dtype)
    qc = jnp.concatenate([qc0[:2], qcs])
    qrc = jnp.concatenate([zero2, qrcs])
    pw = jnp.concatenate([zero2, pws])
    clw = jnp.concatenate([zero2, clws])
    return qc, qrc, pw, pwav, clw


def cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, kx):
    def step(aa, kk):
        dz = z[kk] - z[kk - 1]
        da = zu[kk] * dz * (9.81 / (1004.0 * (t_cup[kk]))) * dby[kk - 1] / (1.0 + gamma_cup[kk])
        active = (kk > kbcon) & (kk <= ktop) & ~((kk == ktop) & (da <= 0.0))
        an = aa + da
        an = jnp.where(an < 0.0, jnp.zeros_like(an), an)
        return jnp.where(active, an, aa), None

    aa, _ = _scan_up(step, _c(0.0, z), kx, k_lo=2)
    return aa


# ---------------------------------------------------------------------------
# downdraft
# ---------------------------------------------------------------------------


def cup_dd_nms(z_cup, cdd_in, entr, jmin, itest, kdet, z1, kx):
    """cup_dd_nms -> (zd, cdd)."""
    perc = _c(0.03, z_cup)
    a = 1.0 - perc
    k = _kidx(kx)
    cdd = jnp.zeros_like(z_cup) if itest == 0 else cdd_in
    if itest == 0:
        zkdet = z_cup[kdet] - z1
        dz = jnp.concatenate([z_cup[1:] - z_cup[:-1], jnp.zeros((1,), z_cup.dtype)])  # dz(ki)=z(ki+1)-z(ki)
        zkp1 = jnp.concatenate([z_cup[1:], z_cup[-1:]])
        cddv = entr + (1.0 - (a * (z_cup - z1) + perc * zkdet) / (a * (zkp1 - z1) + perc * zkdet)) / dz
        cdd = jnp.where((k >= 1) & (k <= jmin - 1) & (k <= kdet), cddv, cdd)

    def step(zprev, kk):
        dz = z_cup[kk + 1] - z_cup[kk]
        zn = zprev * (1.0 + (entr - cdd[kk]) * dz)
        active = (kk >= 1) & (kk <= jmin - 1)
        zk = jnp.where(active, zn, jnp.where(kk == jmin, _c(1.0, z_cup), _c(0.0, z_cup)))
        return zk, zk

    ks = jnp.arange(kx, 0, -1, dtype=jnp.int32)  # kx..1
    _, zs = lax.scan(step, _c(0.0, z_cup), ks)
    zd = jnp.concatenate([jnp.zeros((1,), z_cup.dtype), zs[::-1]])
    return zd, cdd


def cup_dd_he(hes_cup, z_cup, cdd, entr, jmin, he, kx):
    """cup_dd_he -> (hcd, dby) for an ierr == 0 column (hcd(1) only via recurrence)."""
    def step(hprev, kk):
        dz = z_cup[kk + 1] - z_cup[kk]
        hn = (hprev * (1.0 - 0.5 * cdd[kk] * dz) + entr * dz * he[kk]) / (
            1.0 + entr * dz - 0.5 * cdd[kk] * dz)
        active = kk <= jmin - 1
        hk = jnp.where(active, hn, hes_cup[kk])
        return hk, (hk, jnp.where(active, hk - hes_cup[kk], hk - hes_cup[kk]))

    ks = jnp.arange(kx, 0, -1, dtype=jnp.int32)
    _, (hs, ds) = lax.scan(step, _c(0.0, z_cup), ks)
    hcd = jnp.concatenate([jnp.zeros((1,), z_cup.dtype), hs[::-1]])
    dby = jnp.concatenate([jnp.zeros((1,), z_cup.dtype), ds[::-1]])
    return hcd, dby


def cup_dd_moisture(zd, hcd, hes_cup, qes_cup, q_cup, z_cup, cdd, entr, jmin, gamma_cup, q,
                    xl, kx, *, high_resolution=0):
    """cup_dd_moisture(_3d) -> (qcd, qrcd, pwd, pwev, bu) for an ierr == 0 column."""
    k = _kidx(kx)
    rxl = _c(1.0, z_cup) / xl
    zero = _c(0.0, z_cup)
    # level jmin
    dz_j = z_cup[jmin + 1] - z_cup[jmin]
    qcd_j = q_cup[jmin]
    if high_resolution == 1:
        qcd_j = 0.5 * (qes_cup[jmin] + q_cup[jmin])
    qrcd_j = qes_cup[jmin]
    pwd_j = jnp.minimum(zero, qcd_j - qrcd_j)
    pwev0 = zero + pwd_j
    dh_j = hcd[jmin] - hes_cup[jmin]
    bu0 = dz_j * dh_j

    def step(carry, kk):
        qprev, pwev, bu = carry
        dz = z_cup[kk + 1] - z_cup[kk]
        qn = (qprev * (1.0 - 0.5 * cdd[kk] * dz) + entr * dz * q[kk]) / (
            1.0 + entr * dz - 0.5 * cdd[kk] * dz)
        dh = hcd[kk] - hes_cup[kk]
        bun = bu + dz * dh
        qrcd = qes_cup[kk] + rxl * (gamma_cup[kk] / (1.0 + gamma_cup[kk])) * dh
        dqeva = qn - qrcd
        dqeva = jnp.where(dqeva > 0.0, zero, dqeva)
        pwd = zd[kk] * dqeva
        active = kk <= jmin - 1
        return ((jnp.where(active, qrcd, qprev), jnp.where(active, pwev + pwd, pwev),
                 jnp.where(active, bun, bu)),
                (jnp.where(active, qrcd, zero), jnp.where(active, pwd, zero)))

    ks = jnp.arange(kx, 0, -1, dtype=jnp.int32)
    (qlast, pwev, bu), (qrs, pws) = lax.scan(step, (qes_cup[jmin], pwev0, bu0), ks)
    qrcd = jnp.concatenate([jnp.zeros((1,), z_cup.dtype), qrs[::-1]])
    pwd = jnp.concatenate([jnp.zeros((1,), z_cup.dtype), pws[::-1]])
    qrcd = jnp.where(k == jmin, qrcd_j, qrcd)
    pwd = jnp.where(k == jmin, pwd_j, pwd)
    qcd = jnp.where((k >= 1) & (k <= jmin), qrcd, zero)  # qcd(ki) = qrcd(ki) after each level
    return qcd, qrcd, pwd, pwev, bu


def cup_dd_edt(us, vs, z, ktop, kbcon, p, pwav, pwev, edtmax, edtmin, kx, variant="g3"):
    """cup_dd_edt -> edtc (MAXENS2,) for an ierr == 0 column."""
    zero = _c(0.0, z)

    def step(carry, kk):
        vws, sdp = carry
        inr = (kk <= jnp.minimum(ktop, kx)) & (kk >= kbcon)
        dzk = z[kk + 1] - z[kk]
        term = (jnp.abs((us[kk + 1] - us[kk]) / dzk) + jnp.abs((vs[kk + 1] - vs[kk]) / dzk)) * (
            p[kk] - p[kk + 1])
        vws_n = jnp.where(inr, vws + term, vws)
        sdp_n = jnp.where(inr, sdp + p[kk] - p[kk + 1], sdp)
        return (vws_n, sdp_n), None

    ks = jnp.arange(1, kx, dtype=jnp.int32)  # kts..ktf-1
    (vws, sdp), _ = lax.scan(step, (zero, zero), ks)
    vshear = 1.0e3 * vws / sdp
    pef = (1.591 - 0.639 * vshear + 0.0953 * (vshear ** 2) - 0.00496 * (vshear ** 3))
    hi, lo = (_c(1.0, z), zero) if variant == "g3" else (edtmax, edtmin)
    pef = jnp.where(pef > hi, hi, pef)
    pef = jnp.where(pef < lo, lo, pef)
    zkbc = z[kbcon] * 3.281e-3
    poly = 0.96729352 + zkbc * (-0.70034167 + zkbc * (0.162179896 + zkbc * (
        -1.2569798e-2 + zkbc * (4.2772e-4 - zkbc * 5.44e-6))))
    prezk = jnp.where(zkbc > 3.0, poly, _c(0.02, z))
    prezk = jnp.where(zkbc > 25.0, _c(2.4, z), prezk)
    pefb = 1.0 / (1.0 + prezk)
    pefb = jnp.where(pefb > hi, hi, pefb)
    pefb = jnp.where(pefb < lo, lo, pefb)
    edt = 1.0 - 0.5 * (pefb + pef)
    einc = 0.2 * edt
    edtc = jnp.stack([edt + _c(float(kk - 2), z) * einc for kk in range(1, MAXENS2 + 1)])
    edtc = -edtc * pwav / pwev
    edtc = jnp.where(edtc > edtmax, edtmax, edtc)
    edtc = jnp.where(edtc < edtmin, edtmin, edtc)
    return edtc


# ---------------------------------------------------------------------------
# dellas
# ---------------------------------------------------------------------------


def cup_dellabot(he_cup, z_cup, p_cup, hcd, edt, zd, cdd, he, mentrd_rate, g):
    """della(1) of cup_dellabot (subs(1) = 0)."""
    dz = z_cup[2] - z_cup[1]
    dp = 100.0 * (p_cup[1] - p_cup[2])
    detdo1 = edt * zd[2] * cdd[1] * dz
    detdo2 = edt * zd[1]
    entdo = edt * zd[2] * mentrd_rate * dz
    subin = -edt * zd[2]
    return (detdo1 * 0.5 * (hcd[1] + hcd[2]) + detdo2 * hcd[1] + subin * he_cup[2]
            - entdo * he[1]) * g / dp


def cup_dellas_3d(z_cup, p_cup, hcd, edt, zd, cdd, he, mentrd_rate, zu, g, cd, hc, ktop, k22,
                  kbcon, mentr_rate, jmin, he_cup, kdet, kpbl, kx, *, shallow=False,
                  high_res=0, della1=None):
    """cup_dellas_3d -> (della, subs) on k = 1..kx (della(1) from ``della1`` for deep)."""
    zero = jnp.zeros_like(z_cup)
    k = _kidx(kx)
    kp1 = jnp.minimum(k + 1, kx)
    km1 = jnp.maximum(k - 1, 0)
    dz = z_cup[kp1] - z_cup
    zdk1 = zd[kp1]
    detdo = edt * cdd * dz * zdk1
    entdo = edt * mentrd_rate * dz * zdk1
    subin = -zdk1 * edt
    upr = (k >= kbcon) & (k < ktop)
    entup = jnp.where(upr, mentr_rate * dz * zu, zero)
    detup = jnp.where(upr, cd[kp1] * dz * zu, zero)
    subdown = -zd * edt
    entdoj = jnp.where(k == jmin, edt * zd, zero)
    is_k22m1 = k == k22 - 1
    entupk = jnp.where(is_k22m1, zu[kpbl], zero)
    subin_k22 = zu[kp1] - zdk1 * edt
    if high_res == 1:
        subin_k22 = -zdk1 * edt
    subin = jnp.where(is_k22m1, subin_k22, subin)
    detdo = jnp.where(k > kdet, zero, detdo)
    is_top = k == ktop
    detupk = jnp.where(is_top, zu[ktop], zero)
    subin = jnp.where(is_top, zero, subin)
    subdown = jnp.where(is_top, zero, subdown)
    detup = jnp.where(k < kbcon, zero, detup)
    dp = 100.0 * (p_cup[km1] - p_cup)
    hck1 = hc[kp1]
    hcdk1 = hcd[kp1]
    if high_res == 1:
        della = (detup * 0.5 * (hck1 + hc) - entup * he + (entup - detup) * he
                 + detdo * 0.5 * (hcdk1 + hcd)
                 - entdo * he
                 + subin * he_cup[kp1]
                 - subdown * he_cup
                 + detupk * (hc[ktop] - he[ktop])
                 - entdoj * he_cup[jmin]
                 - entupk * he_cup[k22] + entupk * he) * g / dp
    else:
        della = (detup * 0.5 * (hck1 + hc)
                 + detdo * 0.5 * (hcdk1 + hcd)
                 - entup * he
                 - entdo * he
                 + subin * he_cup[kp1]
                 - subdown * he_cup
                 + detupk * (hc[ktop] - he_cup[ktop])
                 - entupk * he_cup[k22]
                 - entdoj * he_cup[jmin]) * g / dp
    subs_core = (zu[kp1] * he_cup[kp1] - zu * he_cup) * g / dp
    in22 = (k >= k22) & (k < ktop)
    if high_res == 1:
        subs = jnp.where(in22, (zu[kp1] * he_cup[kp1] - zu * he_cup - (entup - detup) * he) * g / dp,
                         jnp.where(is_top, detupk * (he[ktop] - he_cup[ktop]) * g / dp,
                                   jnp.where(is_k22m1, (entupk * he - entupk * he_cup) * g / dp, zero)))
    else:
        subs = jnp.where(in22, subs_core, zero)
    active = (k >= 2) & (k <= kx - 1) & (k <= ktop)
    if shallow:
        active = active & ~(k < k22 - 1)
    della = jnp.where(active, della, zero)
    subs = jnp.where(active, subs, zero)
    if not shallow and della1 is not None:
        della = della.at[1].set(della1)
    return della, subs


# ---------------------------------------------------------------------------
# forcing / output ensembles
# ---------------------------------------------------------------------------


def _forcing_ens_g3(aa0, aa1, xaa0, mbdt2, dtime, ierr2, ierr3, axx, mconv, ktop, omeg, k22,
                    kbcon, pr_blk, edt, xland1, ichoice, high_resolution):
    """One iedt block of cup_forcing_ens_3d for an ierr == 0 column.

    pr_blk: (MAXENS, MAXENS3) slice of pr_ens; returns (xf_blk, massfln_blk, closure_n_zero).
    """
    zero = _c(0.0, aa0)
    fens4 = _c(float(ENS4), aa0)
    a_ave = zero
    for ne in range(ENS4):
        a_ave = a_ave + axx[ne]
    a_ave = jnp.maximum(zero, a_ave / fens4)
    a_ave = jnp.minimum(a_ave, aa1)
    a_ave = jnp.maximum(zero, a_ave)
    d10 = (aa1 - aa0) / dtime
    d_ave = (a_ave - aa0) / dtime
    xff0 = d_ave if high_resolution == 1 else d10
    xff = [zero] * 17  # 1-based
    xff[1] = d10
    xff[2] = d_ave
    xff[3] = d10
    xff[13] = d10
    if high_resolution == 1:
        xff[1] = xff[2] = xff[3] = xff[13] = d_ave
    x14 = zero
    for ne in range(ENS4):
        x14 = x14 - omeg[ne, k22] / (fens4 * 9.81)
    xff[14] = jnp.where(x14 < 0.0, zero, x14)
    x5 = zero
    for ne in range(ENS4):
        x5 = x5 - omeg[ne, kbcon] / (fens4 * 9.81)
    xff[5] = jnp.where(x5 < 0.0, zero, x5)
    kx = omeg.shape[1] - 1
    k = _kidx(kx)
    if high_resolution == 0:
        xomg = -omeg / 9.81  # (ENS4, kx+1)
        rng = ((k >= 2) & (k <= kbcon - 1))[None, :]
        x4 = -omeg[0, 2] / 9.81
        x4 = jnp.minimum(x4, jnp.min(jnp.where(rng, xomg, jnp.inf)))
        xff[4] = jnp.where(x4 < 0.0, zero, x4)
        x6 = -omeg[0, 2] / 9.81
        x6 = jnp.maximum(x6, jnp.max(jnp.where(rng, xomg, -jnp.inf)))
        xff[6] = jnp.where(x6 < 0.0, zero, x6)
    else:
        xff[5] = jnp.minimum(xff[5], xff[14])
        xff[4] = xff[5]
        xff[6] = xff[5]
    x7 = mconv[0]
    x8 = mconv[0]
    x9 = mconv[0]
    for ne in range(1, ENS4):
        x7 = jnp.where(mconv[ne] > x7, mconv[ne], x7)
    for ne in range(1, ENS4):
        x8 = jnp.where(mconv[ne] < x8, mconv[ne], x8)
    for ne in range(1, ENS4):
        x9 = x9 + mconv[ne]
    x9 = x9 / fens4
    xff[7], xff[8], xff[9] = x7, x8, x9
    if high_resolution == 1:
        xff[7] = xff[8] = xff[15] = x9
    else:
        xff[15] = mconv[0]
    c2400 = _c(60.0, aa0) * _c(40.0, aa0)
    xff[10] = a_ave / c2400
    xff[11] = aa1 / c2400
    xff[12] = aa1 / c2400
    if high_resolution == 1:
        xff[11] = xff[12] = xff[10]
    if ichoice == 0:
        neg = xff0 < 0.0
        for n in (1, 2, 3, 13, 10, 11, 12):
            xff[n] = jnp.where(neg, zero, xff[n])
    xk = (xaa0 - aa1) / mbdt2
    xk = jnp.where((xk <= 0.0) & (xk > -1.0e-6), _c(-1.0e-6, aa0), xk)
    xk = jnp.where((xk > 0.0) & (xk < 1.0e-6), _c(1.0e-6, aa0), xk)
    xf_rows = []
    mf_rows = []
    water_kill = (xland1 < 0.1) & ((ierr2 > 0) | (ierr3 > 0))
    for ne in range(MAXENS):
        if ne == 0:
            for n in (1, 2, 3, 10, 11, 12, 7, 8, 9, 13, 15):
                xff[n] = jnp.where(water_kill, zero, xff[n])
        xf = [zero] * 17
        xkn = xk[ne]
        pos0 = xff0 >= 0.0
        for n in (1, 2, 3, 13):
            v = jnp.where(xff[n] > 0, jnp.maximum(zero, -xff[n] / xkn) + zero, zero)
            xf[n] = jnp.where(pos0, v, zero)
        for n in (4, 5, 6, 14):
            xf[n] = jnp.maximum(zero, xff[n] + zero)
        for n in (7, 8, 9, 15):
            a1 = jnp.maximum(_c(1.0e-3, aa0), pr_blk[ne, n - 1])
            xf[n] = jnp.maximum(zero, xff[n] / a1)
        kneg = xkn < 0.0
        for n in (10, 11, 12):
            xf[n] = jnp.where(kneg, jnp.maximum(zero, -xff[n] / xkn) + zero, zero)
        if ichoice >= 1:
            pick = xf[ichoice]
            for n in range(1, 17):
                xf[n] = pick
        xf[16] = xf[1]
        mf = [jnp.maximum(zero, edt * xf[n]) for n in range(1, 17)]
        xfv = jnp.stack(xf[1:17])
        mfv = jnp.stack(mf)
        if ne == 1:
            kill = ierr2 > 0
            xfv = jnp.where(kill, jnp.zeros_like(xfv), xfv)
            mfv = jnp.where(kill, jnp.zeros_like(mfv), mfv)
        if ne == 2:
            kill = ierr3 > 0
            xfv = jnp.where(kill, jnp.zeros_like(xfv), xfv)
            mfv = jnp.where(kill, jnp.zeros_like(mfv), mfv)
        xf_rows.append(xfv)
        mf_rows.append(mfv)
    xf_blk = jnp.stack(xf_rows)
    mf_blk = jnp.stack(mf_rows)
    copy_idx = jnp.array([4, 5, 6, 14, 7, 8, 9, 15, 10, 11, 12], jnp.int32) - 1
    xf_blk = xf_blk.at[0, copy_idx].set(xf_blk[1, copy_idx])
    return xf_blk, mf_blk


def _massflx_stats(xf, ierr, variant):
    """massflx_stats on one column's flattened (ENSDIM,) ensemble.

    Returns (xt_ave, x_ave (16,), x_ave_cap (3,), stats_ok) where stats_ok is
    WRF's ``ierr == 0 .and. xt_std > 0`` gate of the APR/pr_* diagnostics.
    """
    zero = jnp.zeros((), xf.dtype)
    num = ENSDIM // MAXENS3
    num2 = ENSDIM // MAXENS
    ok = ierr == 0
    x_ave = jnp.zeros((MAXENS3,), xf.dtype)
    for kk in range(num):
        x_ave = x_ave + xf[MAXENS3 * kk:MAXENS3 * (kk + 1)]
    x_ave_cap = []
    for kcap in range(MAXENS):
        acc = zero
        for iedt in range(MAXENS2):
            for kk in range(MAXENS3):
                acc = acc + xf[MAXENS3 * kcap + iedt * MAXENS * MAXENS3 + kk]
        x_ave_cap.append(acc / _c(float(num2), xf))
    x_ave_cap = jnp.stack(x_ave_cap)
    x_ave = x_ave / _c(float(num), xf)
    xt_ave = zero
    for kk in range(MAXENS3):
        xt_ave = xt_ave + x_ave[kk]
    xt_ave = xt_ave / _c(float(MAXENS3), xf)
    xt_std = zero
    for kk in range(MAXENS3):
        xt_std = xt_std + (x_ave[kk] - xt_ave) ** 2
    xt_std = jnp.where(xt_ave > 0.0, xt_std, zero)
    stats_ok = ok & (xt_std > 0.0)
    xt_ave = jnp.where(ok, xt_ave, zero)
    return xt_ave, x_ave, x_ave_cap, stats_ok


def _apr_weights(x_ave, x_ave_cap, variant):
    """(gr, w, mc, st, as, capma, capme, capmi) closure-group means of massflx_stats."""
    if variant == "g3":
        gr = 0.25 * (x_ave[0] + x_ave[1] + x_ave[2] + x_ave[12])
        w = 0.25 * (x_ave[3] + x_ave[4] + x_ave[5] + x_ave[13])
        mc = 0.25 * (x_ave[6] + x_ave[7] + x_ave[8] + x_ave[14])
        as_ = x_ave[15]
    else:
        gr = 0.333 * (x_ave[0] + x_ave[1] + x_ave[2])
        w = 0.333 * (x_ave[3] + x_ave[4] + x_ave[5])
        mc = 0.333 * (x_ave[6] + x_ave[7] + x_ave[8])
        as_ = 0.25 * (x_ave[12] + x_ave[13] + x_ave[14] + x_ave[15])
    st = 0.333 * (x_ave[9] + x_ave[10] + x_ave[11])
    return (gr, w, mc, st, as_, x_ave_cap[0], x_ave_cap[1], x_ave_cap[2])


# ---------------------------------------------------------------------------
# jmin (downdraft origin) search
# ---------------------------------------------------------------------------


def _jmin_search(jmin0, kdet0, ktop, heso_cup, zo_cup, ierr, kx):
    """The ``keep_going`` loop of CUP_enss(_3d) -> (jmin, kdet, ierr)."""

    def outer_cond(s):
        return s[2]

    def outer_body(s):
        jmini, kdet, _keep, ierrv = s
        kdet = jnp.where(jmini - 1 < kdet, jmini - 1, kdet)
        jmini = jnp.where(jmini >= ktop - 1, ktop - 2, jmini)
        ki = jmini

        def inner(idx, c):
            jm, dh, keep, ier, broken = c
            k = ki - 1 - idx
            act = (k >= 1) & ~broken
            hcdo_k = heso_cup[jm]
            dz = zo_cup[jnp.minimum(k + 1, kx)] - zo_cup[jnp.maximum(k, 0)]
            dh_n = dh + dz * (hcdo_k - heso_cup[jnp.maximum(k, 0)])
            pos = act & (dh_n > 0.0)
            jm_n = jnp.where(pos, jm - 1, jm)
            gt3 = jm_n > 3
            keep_n = keep | (pos & gt3)
            ier_n = jnp.where(pos & ~gt3, 9, ier)
            broken_n = broken | (pos & ~gt3)
            return (jm_n, jnp.where(act, dh_n, dh), keep_n, ier_n, broken_n)

        jm, _dh, keep, ierrv, _b = lax.fori_loop(
            0, kx, inner, (jmini, _c(0.0, zo_cup), jnp.asarray(False), ierrv, jnp.asarray(False)))
        return (jm, kdet, keep, ierrv)

    start = (jnp.asarray(jmin0, jnp.int32), jnp.asarray(kdet0, jnp.int32), ierr == 0,
             jnp.asarray(ierr, jnp.int32))
    jmini, kdet, _k, ierr_n = lax.while_loop(outer_cond, outer_body, start)
    ok = ierr == 0
    jmin = jnp.where(ok, jmini, jmin0)
    kdet = jnp.where(ok, kdet, kdet0)
    ierr_n = jnp.where(ok & (jmini <= 3), 4, ierr_n)
    return jmin.astype(jnp.int32), kdet.astype(jnp.int32), ierr_n.astype(jnp.int32)


def _first_above(zc, thresh, kx, default):
    k = _kidx(kx)
    found, kf = _first_true((k >= 1) & (zc > thresh), kx)
    return jnp.where(found, kf, default).astype(jnp.int32)


def _cd_stable(cd, kstabi, kstabm, inc, cap, active, kx):
    """cd(k)=cd(k-1)+inc capped at cap for k=kstabi..kstabm-1 (sequential)."""
    def step(prev, kk):
        rng = active & (kstabm - 1 > kstabi) & (kk >= kstabi) & (kk <= kstabm - 1)
        v = prev + inc
        v = jnp.where(v > cap, cap, v)
        out = jnp.where(rng, v, cd[kk])
        return out, out

    _, outs = _scan_up(step, cd[0], kx, k_lo=1)
    return jnp.concatenate([cd[:1], outs])


# ---------------------------------------------------------------------------
# cup_axx (G3 ensemble of cloud work functions from neighbour soundings)
# ---------------------------------------------------------------------------


def _axx_member(tx, qx, p, z1, psur, kbmax, ierr, cap_max, cap_inc, entr_rate, mentr_rate,
                tcrit, xl, rv, cp, kx):
    z, qes, he, hes, qxc = cup_env(tx, qx, p, z1, psur, tcrit, 0, xl, cp, kx, variant="gd")
    # cup_axx hard-codes he/hes with 1004/2.5e6 like cup_env itest=0 (same as above)
    qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup = cup_env_clev(
        tx, qes, qxc, he, hes, z, p, psur, z1, xl, rv, cp, kx)
    ierrxx = ierr
    k22 = cup_maximi(he_cup, 3, kbmax, ierrxx, kx)
    ierrxx = jnp.where((ierrxx == 0) & (k22 >= kbmax), 2, ierrxx)
    kbcon, k22, ierrxx = cup_kbcon(cap_inc, 1, k22, he_cup, hes_cup, ierrxx, kbmax, p_cup,
                                   cap_max, kx)
    kstabm = jnp.int32(kx - 1)
    kstabi = cup_minimi(hes_cup, kbcon, kstabm, ierrxx, kx)
    cd = jnp.full((kx + 1,), 0.1, tx.dtype) * entr_rate
    cd = _cd_stable(cd, kstabi, kstabm, 1.5 * entr_rate, 10.0 * entr_rate, ierrxx == 0, kx)
    _hkb, hc, dby = cup_up_he(k22, z_cup, cd, mentr_rate, he_cup, kbcon, he, hes_cup, kx)
    ktop, dby, ierrxx = cup_ktop(1, dby, kbcon, ierrxx, kx)
    zu = cup_up_nms(z_cup, mentr_rate, cd, kbcon, ktop, k22, kx)
    aa0 = cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, kx)
    return jnp.where(ierrxx == 0, aa0, jnp.zeros_like(aa0))


# ---------------------------------------------------------------------------
# CUP_enss_3d (G3) column
# ---------------------------------------------------------------------------


def g3_cup_enss_3d_column(t, q, z1, tn, qo, po, psur, us, vs, tshall, qshall, kpbl, dhdt,
                          tx, qx, omeg, mconv, xland, gsw, dtime, xl, rv, cp, g, tscl_kf, *,
                          kx, ishallow=0, high_resolution=0, ichoice=0):
    """One column of WRF ``CUP_enss_3d`` (iens = 1).  All arrays 1-based (kx+1,).

    tx, qx, omeg: (ENS4, kx+1); mconv: (ENS4,).  Returns a dict of outputs.
    """
    D = t.dtype
    zero = _c(0.0, t)
    k = _kidx(kx)
    p = po
    tcrit = _c(258.0, t)
    xland1 = jnp.where(xland > 1.5, zero, _c(1.0, t))
    cap_max_increment = _c(25.0, t)
    radius = _c(12000.0, t)
    entr_rate = 0.2 / radius
    entr_rate3 = _c(0.2, t) / _c(200.0, t)
    mentrd_rate = zero
    mentr_rate = entr_rate
    mentr_rate3 = entr_rate3
    cd = jnp.full((kx + 1,), 0.01, D) * entr_rate
    cd3 = jnp.full((kx + 1,), 1.0, D) * entr_rate3
    edtmax = _c(1.0, t)
    edtmin = _c(0.2, t)
    depth_min = _c(500.0, t)
    cap_maxs = _c(75.0, t)
    kstabm = jnp.int32(kx - 1)
    ierr = jnp.int32(0)
    cap_max = jnp.where((gsw < 1.0) | (high_resolution == 1), _c(25.0, t), cap_maxs)
    cap_max3 = _c(25.0, t)
    zkbmax = _c(4000.0, t)
    zcutdown = _c(3000.0, t)
    z_detr = _c(1250.0, t)
    mbdt_ens = jnp.stack([(_c(float(n), t) - 3.0) * dtime * 1.0e-3 + dtime * 5.0e-03
                          for n in range(1, MAXENS + 1)])

    # environment
    z, qes, he, hes, _ = cup_env(t, q, p, z1, psur, tcrit, 0, xl, cp, kx)
    zo, qeso, heo, heso, _ = cup_env(tn, qo, po, z1, psur, tcrit, 0, xl, cp, kx)
    qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup = cup_env_clev(
        t, qes, q, he, hes, z, p, psur, z1, xl, rv, cp, kx)
    (qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup,
     tn_cup) = cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, xl, rv, cp, kx)
    kbmax = _first_above(zo_cup, zkbmax + z1, kx, 1)
    kdet = _first_above(zo_cup, z_detr + z1, kx, 0)

    k22 = cup_maximi(heo_cup, 3, kbmax, ierr, kx)
    ierr = jnp.where((ierr == 0) & (k22 >= kbmax), 2, ierr).astype(jnp.int32)
    kbcon, k22, ierr = cup_kbcon(cap_max_increment, 1, k22, heo_cup, heso_cup, ierr, kbmax,
                                 po_cup, cap_max, kx)
    kstabi = cup_minimi(heso_cup, kbcon, kstabm, ierr, kx)
    cd = _cd_stable(cd, kstabi, kstabm, 0.15 * entr_rate, 1.0 * entr_rate, ierr == 0, kx)
    hkb, hc, dby = cup_up_he(k22, z_cup, cd, mentr_rate, he_cup, kbcon, he, hes_cup, kx)
    hkbo, hco, dbyo = cup_up_he(k22, zo_cup, cd, mentr_rate, heo_cup, kbcon, heo, heso_cup, kx)
    ktop, dbyo, ierr = cup_ktop(1, dbyo, kbcon, ierr, kx)
    zktop = (zo_cup[ktop] - z1) * 0.6
    zktop = jnp.minimum(zktop + z1, zcutdown + z1)
    kzdown = jnp.where(ierr == 0, _first_above(zo_cup, zktop, kx, 0), 0).astype(jnp.int32)
    jmin = cup_minimi(heso_cup, k22, kzdown, ierr, kx)
    jmin, kdet, ierr = _jmin_search(jmin, kdet, ktop, heso_cup, zo_cup, ierr, kx)
    ierr = jnp.where((ierr == 0) & (-zo_cup[kbcon] + zo_cup[ktop] < depth_min), 6, ierr)

    zu = cup_up_nms(z_cup, mentr_rate, cd, kbcon, ktop, k22, kx)
    zuo = cup_up_nms(zo_cup, mentr_rate, cd, kbcon, ktop, k22, kx)
    zd, cdd = cup_dd_nms(z_cup, None, mentrd_rate, jmin, 0, kdet, z1, kx)
    cdd = jnp.where(ierr == 0, cdd, jnp.zeros_like(cdd))
    zdo, _ = cup_dd_nms(zo_cup, cdd, mentrd_rate, jmin, 1, kdet, z1, kx)
    hcd, _ = cup_dd_he(hes_cup, z_cup, cdd, mentrd_rate, jmin, he, kx)
    hcdo, _ = cup_dd_he(heso_cup, zo_cup, cdd, mentrd_rate, jmin, heo, kx)
    _qcd, _qrcd, pwd, pwev, _bu = cup_dd_moisture(
        zd, hcd, hes_cup, qes_cup, q_cup, z_cup, cdd, mentrd_rate, jmin, gamma_cup, q, xl, kx,
        high_resolution=high_resolution)
    _qcdo, qrcdo, pwdo, pwevo, buo = cup_dd_moisture(
        zdo, hcdo, heso_cup, qeso_cup, qo_cup, zo_cup, cdd, mentrd_rate, jmin, gammao_cup, qo,
        xl, kx, high_resolution=high_resolution)
    ierr = jnp.where((ierr == 0) & ((pwevo == 0.0) | (buo >= 0.0)), 7, ierr)
    _qc, qrc, pw, pwav, _clw = cup_up_moisture(z_cup, kbcon, ktop, cd, dby, mentr_rate, q,
                                               gamma_cup, zu, qes_cup, k22, q_cup, xl, kx,
                                               c0=0.002)
    cupclw = qrc
    qco, qrco, pwo, pwavo, _clwo = cup_up_moisture(zo_cup, kbcon, ktop, cd, dbyo, mentr_rate,
                                                   qo, gammao_cup, zuo, qeso_cup, k22, qo_cup,
                                                   xl, kx, c0=0.002)
    aa0 = cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, kx)
    aa1 = cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, kx)
    ierr = jnp.where((ierr == 0) & (aa1 == 0.0), 17, ierr).astype(jnp.int32)
    ok_deep_pre = ierr == 0

    # ---------------- shallow (ishallow_g3 == 1) ----------------
    if ishallow == 1:
        sh = _g3_shallow(t, q, qshall, tshall, po, po_cup, p_cup, z1, psur, kpbl, dhdt, z,
                         he, he_cup, hes_cup, z_cup, t_cup, kbmax, cap_max_increment,
                         cap_max3, cd3, mentr_rate3, mbdt_ens, dtime, tscl_kf, tcrit, xl, rv,
                         cp, g, kx)
    else:
        sh = None

    # ---------------- cup_axx ----------------
    axx = jax.vmap(lambda a, b: _axx_member(a, b, p, z1, psur, kbmax, ierr, cap_max,
                                            cap_max_increment, entr_rate, mentr_rate, tcrit,
                                            xl, rv, cp, kx))(tx, qx)

    edtc = cup_dd_edt(us, vs, zo, ktop, kbcon, po, pwavo, pwevo, edtmax, edtmin, kx)

    # K22x / ierr2 / ierr3 (independent of iedt): recomputed per iedt in WRF from the
    # current ierr; only ierr (deep) can change inside the loop, and ierr2/3 start
    # from it, so they are evaluated inside the scan.
    ens_shape = (MAXENS, MAXENS3)

    def iedt_body(carry, iedt):
        (ierr_c, pr_ens, xf_ens, massfln, edt_out, closure_n) = carry
        ok = ierr_c == 0
        edt_v = edtc[iedt] if high_resolution == 0 else edtc[2]
        edt_out = jnp.where(ok, edtc[2] if high_resolution == 1 else edtc[1], edt_out)
        edto = edt_v
        della1_h = cup_dellabot(heo_cup, zo_cup, po, hcdo, edto, zdo, cdd, heo, mentrd_rate, g)
        della1_q = cup_dellabot(qo_cup, zo_cup, po, qrcdo, edto, zdo, cdd, qo, mentrd_rate, g)
        dellah, dsubt = cup_dellas_3d(zo_cup, po_cup, hcdo, edto, zdo, cdd, heo, mentrd_rate,
                                      zuo, g, cd, hco, ktop, k22, kbcon, mentr_rate, jmin,
                                      heo_cup, kdet, k22, kx, high_res=high_resolution,
                                      della1=della1_h)
        dellaq, dsubq = cup_dellas_3d(zo_cup, po_cup, qrcdo, edto, zdo, cdd, qo, mentrd_rate,
                                      zuo, g, cd, qco, ktop, k22, kbcon, mentr_rate, jmin,
                                      qo_cup, kdet, k22, kx, high_res=high_resolution,
                                      della1=della1_q)
        kp1 = jnp.minimum(k + 1, kx)
        dpc = po_cup - po_cup[kp1]
        dq_top = 0.01 * zuo[ktop] * qrco[ktop] * 9.81 / dpc
        dzc = zo_cup[kp1] - zo_cup
        dq_mid = (_c(0.01, t) * _c(9.81, t)) * cd * dzc * zuo * 0.5 * (qrco + qrco[kp1]) / dpc
        dellaqc = jnp.where(k == ktop, dq_top, zero)
        dellaqc = jnp.where((k < ktop) & (k > kbcon), dq_mid, dellaqc)
        dellaqc = jnp.where((k >= 1) & (k <= kx - 1) & ok, dellaqc, zero)
        mbdt = mbdt_ens[1]
        rcp = _c(1.0, t) / cp
        xhe = (dsubt + dellah) * mbdt + heo
        xq = (dsubq + dellaq) * mbdt + qo
        dellat = rcp * (dellah - xl * dellaq)
        dsubt_c = rcp * (dsubt - xl * dsubq)
        xt = (dellat + dsubt_c) * mbdt + tn
        xq = jnp.where(xq <= 0.0, _c(1.0e-08, t), xq)
        xhe = xhe.at[kx].set(heo[kx])
        xq = xq.at[kx].set(jnp.where(qo[kx] <= 0.0, _c(1.0e-08, t), qo[kx]))
        xt = xt.at[kx].set(tn[kx])
        dellat = jnp.where(ok, dellat, zero)
        xz, xqes, xhe, xhes, _ = cup_env(xt, xq, po, z1, psur, tcrit, 2, xl, cp, kx, he_in=xhe)
        (xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, _xp_cup, xgamma_cup,
         xt_cup) = cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, psur, z1, xl, rv, cp, kx)
        _xhkb, _xhc, xdby = cup_up_he(k22, xz_cup, cd, mentr_rate, xhe_cup, kbcon, xhe,
                                      xhes_cup, kx)
        xzu = cup_up_nms(xz_cup, mentr_rate, cd, kbcon, ktop, k22, kx)
        xaa0 = cup_up_aa0(xz, xzu, xdby, xgamma_cup, xt_cup, kbcon, ktop, kx)
        # pr_ens for this iedt block (all 3 nens rows get identical sums)
        in_top = (k >= 1) & (k <= ktop)

        def pr_step(c, kk):
            a7, a8, a9, ad = c
            use = in_top[kk]
            n7 = a7 + edto * pwdo[kk] + pwo[kk]
            n8 = a8 + pwo[kk]
            n9 = a9 + 0.5 * edto * pwdo[kk] + pwo[kk]
            nd = ad + pwo[kk] + edto * pwdo[kk]
            return (jnp.where(use, n7, a7), jnp.where(use, n8, a8), jnp.where(use, n9, a9),
                    jnp.where(use, nd, ad)), None

        (a7, a8, a9, ad), _ = _scan_up(pr_step, (zero, zero, zero, zero), kx, k_lo=1)
        row = jnp.stack([ad] * 6 + [a7, a8, a9] + [ad] * 7)
        kill18 = ok & (a7 < 1.0e-6)
        row = jnp.where(row < 1.0e-4, jnp.zeros_like(row), row)
        blk_ok = ok & ~kill18
        # nens = 1 sets ierr 18 -> nens 2/3 skip (rows stay zero)
        pr_blk = jnp.where(blk_ok, jnp.broadcast_to(row, ens_shape), jnp.zeros(ens_shape, D))
        pr_ens = pr_ens.at[iedt].set(jnp.where(ok, pr_blk, pr_ens[iedt]))
        ierr_c = jnp.where(kill18, 18, ierr_c).astype(jnp.int32)
        ok = ierr_c == 0
        # K22x, ierr2, ierr3
        k22x = cup_maximi(heo_cup, 3, kbmax, ierr_c, kx)
        _kbx2, k22x2, ierr2 = cup_kbcon(cap_max_increment, 2, k22x, heo_cup, heso_cup, ierr_c,
                                        kbmax, po_cup, cap_max, kx)
        _kbx3, _k22x3, ierr3 = cup_kbcon(cap_max_increment, 3, k22x2, heo_cup, heso_cup,
                                         ierr_c, kbmax, po_cup, cap_max, kx)
        xaa0_ens = jnp.stack([xaa0] * MAXENS)
        xf_blk, mf_blk = _forcing_ens_g3(aa0, aa1, xaa0_ens, mbdt_ens[1], dtime, ierr2, ierr3,
                                         axx, mconv, ktop, omeg, k22, kbcon, pr_ens[iedt],
                                         edto, xland1, ichoice, high_resolution)
        xf_ens = xf_ens.at[iedt].set(jnp.where(ok, xf_blk, xf_ens[iedt]))
        massfln = massfln.at[iedt].set(jnp.where(ok, mf_blk, massfln[iedt]))
        clear = (ierr_c != 0) & (ierr_c != 20)
        xf_ens = jnp.where(clear, jnp.zeros_like(xf_ens), xf_ens)
        massfln = jnp.where(clear, jnp.zeros_like(massfln), massfln)
        if ichoice >= 1:
            closure_n = jnp.where(ok, zero, closure_n)
        ens = (jnp.where(ok, dsubt_c, zero), jnp.where(ok, dsubq, zero),
               jnp.where(ok, dellat, zero), jnp.where(ok, dellaq, zero),
               jnp.where(ok, dellaqc, zero), jnp.where(ok, pwo + edto * pwdo, zero))
        return (ierr_c, pr_ens, xf_ens, massfln, edt_out, closure_n), ens

    carry0 = (ierr, jnp.zeros((MAXENS2,) + ens_shape, D), jnp.zeros((MAXENS2,) + ens_shape, D),
              jnp.zeros((MAXENS2,) + ens_shape, D), zero, _c(16.0, t))
    (ierr, pr_ens, xf_ens, massfln, edt_out, closure_n), ens = lax.scan(
        iedt_body, carry0, jnp.arange(MAXENS2, dtype=jnp.int32))
    subt_ens, subq_ens, dellat_ens, dellaq_ens, dellaqc_ens, pwo_ens = ens  # (3, kx+1) each

    out = _cup_output_ens(xf_ens, ierr, dellat_ens, dellaq_ens, dellaqc_ens, subt_ens,
                          subq_ens, zuo, pwo_ens, ktop, pr_ens, massfln, closure_n, xland1, kx,
                          variant="g3")
    pre = jnp.maximum(out["pre"], zero)
    res = dict(outt=out["outt"], outq=out["outq"], outqc=out["outqc"], subt=out["subt"],
               subq=out["subq"], pre=pre, kbcon=kbcon, ktop=ktop, cupclw=cupclw,
               xf_ens=out["xf_ens"], pr_ens=pr_ens.reshape(ENSDIM), apr=out["apr"],
               edt_out=edt_out, ierr=out["ierr"])
    if sh is not None:
        outts, outqs = sh["outts"], sh["outqs"]
        if high_resolution == 1:
            both = (out["ierr"] == 0) & (sh["ierr5"] == 0) & (kbcon < sh["ktop3"] + 1)
            outts = jnp.where(both, jnp.zeros_like(outts), outts)
            outqs = jnp.where(both, jnp.zeros_like(outqs), outqs)
        res.update(outts=outts, outqs=outqs, k23=sh["k23"], kbcon3=sh["kbcon3"],
                   ktop3=sh["ktop3"], xmb3=sh["xmb3"])
    else:
        zk = jnp.zeros((kx + 1,), D)
        res.update(outts=zk, outqs=zk, k23=jnp.int32(0), kbcon3=jnp.int32(0),
                   ktop3=jnp.int32(0), xmb3=zero)
    return res


def _cup_output_ens(xf_ens, ierr, dellat, dellaq, dellaqc, subt_ens, subq_ens, zu, pw, ktop,
                    pr_ens, massfln, closure_n, xland1, kx, *, variant):
    """cup_output_ens(_3d): ensemble mean mass flux and feedbacks (iens = 1)."""
    D = zu.dtype
    zero = jnp.zeros((), D)
    ok = ierr == 0
    xf = xf_ens.reshape(ENSDIM)
    prf = pr_ens.reshape(ENSDIM)
    xf = jnp.where(ok & (prf <= 0.0), jnp.zeros_like(xf), xf)
    xmb_ave, x_ave, x_cap, ok1 = _massflx_stats(xf, ierr, variant)
    pr_ave, p_ave, p_cap, ok2 = _massflx_stats(prf, ierr, variant)
    w_x = _apr_weights(x_ave, x_cap, variant)
    w_p = _apr_weights(p_ave, p_cap, variant)
    pr_w = [jnp.where(ok1, a, zero) for a in w_x]
    apr = jnp.stack([jnp.where(ok2, b * 3600.0 * a, zero) for a, b in zip(pr_w, w_p)])
    ierr = jnp.where(ok & (xmb_ave <= 0.0), 13, ierr)
    xmb_ave = jnp.where(ok & (xmb_ave <= 0.0), zero, xmb_ave)
    tuning = zero
    xmb = jnp.maximum(0.1 * xmb_ave, xmb_ave - tuning * zero)
    clos_wei = 16.0 / jnp.maximum(_c(1.0, zu), closure_n)
    xmb = jnp.where(xland1 < 0.5, xmb * clos_wei, xmb)
    xmb = jnp.where(ok, xmb, zero)
    ierr = jnp.where(ok & (xmb == 0.0), 19, ierr)
    ierr = jnp.where(ok & (xmb > 100.0), 19, ierr)
    xfac1 = xmb
    k = _kidx(kx)
    nx = MAXENS2
    fnx = _c(float(nx), zu)
    ok_out = ierr == 0
    use = ok_out & (k <= ktop)
    dtt = zero + dellat[0]
    dtts = zero + subt_ens[0]
    dtq = zero + dellaq[0]
    dtqs = zero + subq_ens[0]
    dtqc = zero + dellaqc[0]
    dtpw = zero + pw[0]
    for n in range(1, nx):
        dtt = dtt + dellat[n]
        dtts = dtts + subt_ens[n]
        dtq = dtq + dellaq[n]
        dtqs = dtqs + subq_ens[n]
        dtqc = dtqc + dellaqc[n]
        dtpw = dtpw + pw[n]
    zk = jnp.zeros_like(zu)
    outt = jnp.where(use, xmb * dtt / fnx, zk)
    subt = jnp.where(use, xmb * dtts / fnx, zk)
    outq = jnp.where(use, xmb * dtq / fnx, zk)
    subq = jnp.where(use, xmb * dtqs / fnx, zk)
    outqc = jnp.where(use, xmb * dtqc / fnx, zk)
    prek = xmb * dtpw / fnx

    def pstep(acc, kk):
        return jnp.where(use[kk], acc + prek[kk], acc), None

    pre, _ = _scan_up(pstep, zero, kx, k_lo=1)
    xf_out = jnp.where(ok_out, xf * xfac1, xf)
    return dict(outt=outt, outq=outq, outqc=outqc, subt=subt, subq=subq, pre=pre,
                xf_ens=xf_out, apr=apr, ierr=ierr.astype(jnp.int32))


def _g3_shallow(t, q, qshall, tshall, po, po_cup, p_cup, z1, psur, kpbl, dhdt, z, he, he_cup,
                hes_cup, z_cup, t_cup, kbmax, cap_max_increment, cap_max3, cd3, mentr_rate3,
                mbdt_ens, dtime, tscl_kf, tcrit, xl, rv, cp, g, kx):
    """The ``ishallow_g3 == 1`` block of CUP_enss_3d (ierr5 chain)."""
    D = t.dtype
    zero = _c(0.0, t)
    k = _kidx(kx)
    ierr5 = jnp.int32(0)
    z3, qes3, he3, hes3, _ = cup_env(tshall, qshall, po, z1, psur, tcrit, 0, xl, cp, kx)
    (qes3_cup, q3_cup, he3_cup, hes3_cup, z3_cup, _p3, gamma3_cup,
     t3_cup) = cup_env_clev(tshall, qes3, qshall, he3, hes3, z3, po, psur, z1, xl, rv, cp, kx)
    k23 = cup_maximi(he3_cup, 1, kbmax, ierr5, kx)
    cap_max3 = jnp.where(kpbl > 5, po_cup[kpbl], cap_max3)
    ierr5 = jnp.where((ierr5 == 0) & (k23 > kbmax), 2, ierr5).astype(jnp.int32)
    k23 = jnp.where(kpbl > 5, kpbl, k23).astype(jnp.int32)
    kbcon3, k23, ierr5 = cup_kbcon(cap_max_increment, 5, k23, he3_cup, hes3_cup, ierr5, kbmax,
                                   po_cup, cap_max3, kx)
    _hkb3, hc3, dby3 = cup_up_he(k23, z3_cup, cd3, mentr_rate3, he3_cup, kbcon3, he3, hes3_cup,
                                 kx, shallow=True)
    _hkb30, _hc30, dby3_0 = cup_up_he(k23, z_cup, cd3, mentr_rate3, he_cup, kbcon3, he, hes_cup,
                                      kx, shallow=True)
    ktop3, dby3, ierr5 = cup_ktop(1, dby3, kbcon3, ierr5, kx)
    zu3 = cup_up_nms(z3_cup, mentr_rate3, cd3, kbcon3, ktop3, k23, kx)
    zu3_0 = cup_up_nms(z_cup, mentr_rate3, cd3, kbcon3, ktop3, k23, kx)
    aa3_0 = cup_up_aa0(z, zu3_0, dby3_0, gamma3_cup, t_cup, kbcon3, ktop3, kx)
    qc3, _qrc3, _pw3, _pwav3, _clw = cup_up_moisture(z3_cup, kbcon3, ktop3, cd3, dby3,
                                                     mentr_rate3, qshall, gamma3_cup, zu3,
                                                     qes3_cup, k23, q3_cup, xl, kx, c0=0.0)
    aa3 = cup_up_aa0(z3, zu3, dby3, gamma3_cup, t3_cup, kbcon3, ktop3, kx)
    zk = jnp.zeros((kx + 1,), D)
    zi = jnp.int32(0)
    dellah3, dsubt3 = cup_dellas_3d(z3_cup, po_cup, zk, zero, zk, zk, he3, zero, zu3, g, cd3,
                                    hc3, ktop3, k23, kbcon3, mentr_rate3, zi, he3_cup, zi, k23,
                                    kx, shallow=True)
    dellaq3, dsubq3 = cup_dellas_3d(z3_cup, po_cup, zk, zero, zk, zk, qshall, zero, zu3, g, cd3,
                                    qc3, ktop3, k23, kbcon3, mentr_rate3, zi, q3_cup, zi, k23,
                                    kx, shallow=True)
    mbdt_s = 1.0e-1 * mbdt_ens[0]
    rcp = _c(1.0, t) / cp
    xhe3 = (dsubt3 + dellah3) * mbdt_s + he3
    xq3 = (dsubq3 + dellaq3) * mbdt_s + qshall
    dellat3 = rcp * (dellah3 - xl * dellaq3)
    dsubt3 = rcp * (dsubt3 - xl * dsubq3)
    xt3 = (dellat3 + dsubt3) * mbdt_s + tshall
    xq3 = jnp.where(xq3 <= 0.0, _c(1.0e-08, t), xq3)
    xhe3 = xhe3.at[kx].set(he3[kx])
    xq3 = xq3.at[kx].set(jnp.where(qshall[kx] <= 0.0, _c(1.0e-08, t), qshall[kx]))
    xt3 = xt3.at[kx].set(tshall[kx])
    xz3, xqes3, xhe3, xhes3, _ = cup_env(xt3, xq3, po, z1, psur, tcrit, 2, xl, cp, kx,
                                         he_in=xhe3)
    (_xqes3c, _xq3c, xhe3_cup, xhes3_cup, xz3_cup, _xp3, gamma3_cup,
     xt3_cup) = cup_env_clev(xt3, xqes3, xq3, xhe3, xhes3, xz3, po, psur, z1, xl, rv, cp, kx)
    _xhkb3, _xhc3, xdby3 = cup_up_he(k23, xz3_cup, cd3, mentr_rate3, xhe3_cup, kbcon3, xhe3,
                                     xhes3_cup, kx, shallow=True)
    xzu3 = cup_up_nms(xz3_cup, mentr_rate3, cd3, kbcon3, ktop3, k23, kx)
    xaa3 = cup_up_aa0(xz3, xzu3, xdby3, gamma3_cup, xt3_cup, kbcon3, ktop3, kx)

    ok5 = ierr5 == 0
    xkshal = (xaa3 - aa3) / mbdt_s
    xkshal = jnp.where(xkshal >= 0.0, _c(1.0e6, t), xkshal)
    xkshal = jnp.where((xkshal > -1.0e-4) & (xkshal < 0.0), _c(-1.0e-4, t), xkshal)
    f13 = jnp.maximum(zero, -(aa3 - aa3_0) / (xkshal * dtime))
    f13 = jnp.where(aa3_0 <= 0.0, zero, f13)
    f13 = jnp.where(aa3 - aa3_0 <= 0.0, zero, f13)

    def bl_step(acc, kk):
        v = acc + 100.0 * dhdt[kk] * (p_cup[kk] - p_cup[kk + 1]) / g
        return jnp.where(kk <= kbcon3 - 1, v, acc), None

    blqe, _ = lax.scan(bl_step, zero, jnp.arange(1, kx, dtype=jnp.int32))
    trash = jnp.maximum(hc3[kbcon3] - he_cup[kbcon3], _c(1.0e1, t))
    f7 = jnp.minimum(_c(0.1, t), jnp.maximum(zero, blqe / trash))
    f7 = jnp.where(k23 < kpbl + 1, f7, zero)
    cond4 = (xkshal < -1.1e-04) & ((aa3 - aa3_0 > 0.0) | (f7 > 0))
    f4 = jnp.minimum(_c(0.1, t), jnp.maximum(zero, -aa3 / (xkshal * tscl_kf)))
    f4 = jnp.where(cond4, f4, zero)
    xmb3 = zero
    for v in (f13, f13, f13, f4, f4, f4, f7, f7, f7):
        xmb3 = xmb3 + v
    xmb3 = jnp.minimum(_c(0.1, t), xmb3 / 9.0)
    ierr5 = jnp.where(ok5 & (xmb3 == 0.0), 22, ierr5)
    ierr5 = jnp.where(ok5 & (xmb3 < 0.0), 21, ierr5).astype(jnp.int32)
    ok5 = ierr5 == 0
    xmb3 = jnp.where(ok5, xmb3, zero)
    # sanity: heating
    rng = (k >= 2) & (k <= ktop3)
    heat = 86400.0 * (dsubt3 + dellat3) * xmb3

    def hstep(acc, kk):
        return jnp.where(rng[kk], jnp.maximum(acc, heat[kk]), acc), None

    trash, _ = _scan_up(hstep, zero, kx, k_lo=1)
    xmb3 = jnp.where(trash > 150.0, xmb3 * 150.0 / trash, xmb3)

    def qstep(xm, kk):
        tend = dsubq3[kk] + dellaq3[kk]
        tr = q[kk] + tend * xm * dtime
        fac = ((1.0e-12 - q[kk]) / dtime) / (tend * xm)
        fac = jnp.minimum(_c(1.0, t), jnp.maximum(zero, fac))
        xm_n = jnp.where(rng[kk] & (tr < 1.0e-12), fac * xm, xm)
        return xm_n, None

    xmb3, _ = _scan_up(qstep, xmb3, kx, k_lo=1)
    outts = jnp.where(rng & ok5, (dsubt3 + dellat3) * xmb3, zk)
    outqs = jnp.where(rng & ok5, (dsubq3 + dellaq3) * xmb3, zk)
    k23 = jnp.where(ok5, k23, 0).astype(jnp.int32)
    kbcon3 = jnp.where(ok5, kbcon3, 0).astype(jnp.int32)
    ktop3 = jnp.where(ok5, ktop3, 0).astype(jnp.int32)
    xmb3 = jnp.where(ok5, xmb3, zero)
    return dict(outts=outts, outqs=outqs, k23=k23, kbcon3=kbcon3, ktop3=ktop3, xmb3=xmb3,
                ierr5=ierr5)


# ---------------------------------------------------------------------------
# G3DRV + conv_grell_spread3d on an (ny, nx) tile
# ---------------------------------------------------------------------------


def _to_cols1(a3):
    """(kx, ny, nx) 0-based mass levels -> (ny*nx, kx+1) 1-based, index 0 = copy of 1."""
    kx, ny, nx = a3.shape
    c = jnp.moveaxis(a3, 0, -1).reshape(ny * nx, kx)
    return jnp.concatenate([c[:, :1], c], axis=1)


def _from_cols1(c, ny, nx):
    kx = c.shape[1] - 1
    return jnp.moveaxis(c[:, 1:].reshape(ny, nx, kx), -1, 0)


def _interior_mask(ny, nx, periodic_x, periodic_y, width=4):
    """WRF ibegc..iendc / jbegc..jendc (0-based) as a boolean (ny, nx) mask."""
    i = jnp.arange(nx)
    j = jnp.arange(ny)
    mi = jnp.ones((nx,), bool) if periodic_x else (i >= width) & (i <= nx - 1 - width)
    mj = jnp.ones((ny,), bool) if periodic_y else (j >= width) & (j <= ny - 1 - width)
    return mj[:, None] & mi[None, :]


def _shift_zero(a, dj, di):
    """a[..., j+dj, i+di] with zero outside the domain (WRF non-periodic halo)."""
    pad = [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]
    ap = jnp.pad(a, pad)
    ny, nx = a.shape[-2:]
    return lax.dynamic_slice_in_dim(lax.dynamic_slice_in_dim(ap, 1 + dj, ny, axis=-2),
                                    1 + di, nx, axis=-1)


def _nine_point_mean(f):
    """G3DRV ave_f_t: 3x3 sum in WRF order / 9 (zero halo)."""
    s = None
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            v = _shift_zero(f, dj, di)
            s = v if s is None else s + v
    return s / 9.0


def g3drv_tile(*, u, v, w, t, q, p, pi, rho, p8w, rthften, rqvften, rthraten, rthblten,
               rqvblten, ht, xland, gsw, kpbl, dt, dx, cugd_avedx=1, ishallow=0, ichoice=0,
               periodic_x=False, periodic_y=False, xlv=2.5e6, cp=1004.5, g=9.81, r_v=461.6):
    """WRF ``cu_physics=5``: G3DRV followed by conv_grell_spread3d on one tile.

    3-D inputs are (kx, ny, nx) mass levels (w, p8w: (kx+1, ny, nx) staggered);
    2-D inputs (ny, nx); kpbl 1-based.  rthften is the raw advective theta
    forcing: the cumulus-driver pre-scaling (rthften+rthraten+rthblten)*pi is
    applied here.  Returns a dict of WRF-named outputs (3-D as (kx, ny, nx)).
    """
    D = t.dtype
    kx, ny, nx = t.shape
    ncol = ny * nx
    c = lambda val: jnp.asarray(val, D)
    dt = c(dt)
    dx = c(dx)
    xlv_, cp_, g_, rv_ = c(xlv), c(cp), c(g), c(r_v)
    high_res = 1 if cugd_avedx > 1 else 0
    sub_spread = c(1.0) / jnp.maximum(c(1.0), c(float(cugd_avedx * cugd_avedx - 1)))
    tscl_kf = dx / 25.0
    # module_cumulus_driver pre-scaling of the forcing (G3SCHEME).
    rthf = (rthften + rthraten + rthblten) * pi
    rqvf = rqvften + rqvblten
    if high_res:
        ave_f_t = _nine_point_mean(rthf)
        ave_f_q = _nine_point_mean(rqvf)
    t2d = t
    q2d = jnp.where(q < 1.0e-08, c(1.0e-08), q)
    tn = t2d + rthf * dt
    qo = q2d + rqvf * dt
    tshall = t2d + rthblten * pi * dt
    dhdt = cp_ * rthblten * pi + xlv_ * rqvblten
    qshall = q2d + rqvblten * dt
    if high_res:
        tn = t2d + ave_f_t * dt
        qo = q2d + ave_f_q * dt
    tn = jnp.where(tn < 200.0, t2d, tn)
    qo = jnp.where(qo < 1.0e-08, c(1.0e-08), qo)
    po = p * 0.01
    psur = p8w[0] * 0.01
    # ens4 neighbour members (clamped, not wrapped -- WRF G3DRV)
    jj = jnp.arange(ny)[:, None]
    ii = jnp.arange(nx)[None, :]
    omeg_m, tx_m, qx_m = [], [], []
    wm = w[:kx]
    for nn in (-1, 0, 1):
        jss = jnp.clip(jj + nn, 0, ny - 1)
        for n in (-1, 0, 1):
            iss = jnp.clip(ii + n, 0, nx - 1)
            omeg_m.append(-g_ * rho * wm[:, jss, iss])
            if high_res:
                txm = t2d + ave_f_t[:, jss, iss] * dt
                qxm = q2d + ave_f_q[:, jss, iss] * dt
            else:
                txm = t2d + rthf[:, jss, iss] * dt
                qxm = q2d + rqvf * dt
            tx_m.append(jnp.where(txm < 200.0, t2d, txm))
            qx_m.append(jnp.where(qxm < 1.0e-08, c(1.0e-08), qxm))
    # mconv per member
    dq = q2d[1:] - q2d[:-1]
    mconv = []
    for m in range(ENS4):
        acc = jnp.zeros((ny, nx), D)
        for kk in range(kx - 1):
            acc = acc + omeg_m[m][kk] * dq[kk] / g_
        mconv.append(jnp.where(acc < 0.0, jnp.zeros_like(acc), acc))
    cols = _to_cols1
    stack_m = lambda lst: jnp.stack([cols(a) for a in lst], axis=1)  # (ncol, 9, kx+1)
    col_fn = functools.partial(g3_cup_enss_3d_column, kx=kx, ishallow=ishallow,
                               high_resolution=high_res, ichoice=ichoice)
    res = jax.vmap(col_fn, in_axes=(0,) * 20 + (None,) * 5)(
        cols(t2d), cols(q2d), ht.reshape(ncol), cols(tn), cols(qo), cols(po),
        psur.reshape(ncol), cols(u), cols(v), cols(tshall), cols(qshall),
        jnp.asarray(kpbl, jnp.int32).reshape(ncol), cols(dhdt), stack_m(tx_m), stack_m(qx_m),
        stack_m(omeg_m), jnp.stack([m.reshape(ncol) for m in mconv], axis=1),
        xland.reshape(ncol), gsw.reshape(ncol), jnp.broadcast_to(dt, (ncol,)),
        xlv_, rv_, cp_, g_, tscl_kf)
    interior = _interior_mask(ny, nx, periodic_x, periodic_y).reshape(ncol)
    pret = res["pre"]
    cuten = jnp.where(interior & (pret > 0.0), c(1.0), c(0.0))
    cut = cuten[:, None]
    intc = interior[:, None]
    zk = jnp.zeros_like(res["outt"])
    cugd_ttens = jnp.where(intc, res["subt"] * cut * sub_spread, zk)
    cugd_qvtens = jnp.where(intc, res["subq"] * cut * sub_spread, zk)
    cugd_tten = jnp.where(intc, res["outts"] + res["outt"] * cut, zk)
    cugd_qvten = jnp.where(intc, res["outqs"] + res["outq"] * cut, zk)
    cold = cols(t2d) < 258.0
    qcc = res["outqc"] * cut
    rqccuten = jnp.where(intc & ~cold, qcc, zk)
    rqicuten = jnp.where(intc & cold, qcc, zk)
    cugd_qcten = jnp.where(intc & ~cold, qcc, zk)
    gdc = jnp.where(intc, res["cupclw"] * cut, zk)
    gdc2 = jnp.where(intc & cold, res["cupclw"] * cut, zk)
    rain = interior & (pret > 0.0)
    raincv = jnp.where(rain, pret * dt, c(0.0))
    pratec = jnp.where(rain, pret, c(0.0))
    htop = jnp.where(rain & (res["ktop"] > 1), res["ktop"].astype(D) + 0.001, c(1.0))
    hbot = jnp.where(rain & (res["kbcon"] < kx), res["kbcon"].astype(D) + 0.001, c(float(kx)))
    ktop_deep = jnp.where(interior, res["ktop"], 0)
    out = dict(
        CUGD_TTEN=cugd_tten, CUGD_QVTEN=cugd_qvten, CUGD_QCTEN=cugd_qcten,
        CUGD_TTENS=cugd_ttens, CUGD_QVTENS=cugd_qvtens, GDC=gdc, GDC2=gdc2,
        DRV_RAINCV=raincv, DRV_PRATEC=pratec, HTOP=htop, HBOT=hbot,
        KTOP_DEEP=ktop_deep,
        K22_SHALLOW=jnp.where(interior, res["k23"], 0),
        KBCON_SHALLOW=jnp.where(interior, res["kbcon3"], 0),
        KTOP_SHALLOW=jnp.where(interior, res["ktop3"], 0),
        XMB_SHALLOW=jnp.where(interior, res["xmb3"], c(0.0)),
        EDT_OUT=jnp.where(interior, res["edt_out"], c(0.0)),
        XF_ENS=jnp.where(intc, res["xf_ens"], 0.0), PR_ENS=jnp.where(intc, res["pr_ens"], 0.0),
        APR=jnp.where(intc, res["apr"], 0.0),
        IERR=res["ierr"].reshape(ny, nx),
    )
    sp = _conv_grell_spread3d(cugd_tten, cugd_qvten, cugd_ttens, cugd_qvtens, rqccuten,
                              rqicuten, raincv, pratec, cols(q), cols(pi), dt, cugd_avedx,
                              ny, nx, kx)
    out.update(sp)
    # reshape column outputs back to WRF layout
    for key in ("CUGD_TTEN", "CUGD_QVTEN", "CUGD_QCTEN", "CUGD_TTENS", "CUGD_QVTENS", "GDC",
                "GDC2", "RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN"):
        out[key] = _from_cols1(out[key], ny, nx)
    for key in ("DRV_RAINCV", "DRV_PRATEC", "HTOP", "HBOT", "KTOP_DEEP", "K22_SHALLOW",
                "KBCON_SHALLOW", "KTOP_SHALLOW", "XMB_SHALLOW", "EDT_OUT", "RAINCV", "PRATEC"):
        out[key] = out[key].reshape(ny, nx)
    out["XF_ENS"] = out["XF_ENS"].reshape(ny, nx, ENSDIM)
    out["PR_ENS"] = out["PR_ENS"].reshape(ny, nx, ENSDIM)
    out["APR"] = out["APR"].reshape(ny, nx, 8)
    return out


def _conv_grell_spread3d(cugd_tten, cugd_qvten, cugd_ttens, cugd_qvtens, rqccuten, rqicuten,
                         raincv, pratec, moist_qv, pi_phy, dt, cugd_avedx, ny, nx, kx):
    """WRF conv_grell_spread3d (single tile; smoothh = smoothv = 1). Column layout in/out."""
    D = cugd_tten.dtype
    to3 = lambda cc: cc.reshape(ny, nx, kx + 1)
    tt, tq = to3(cugd_tten), to3(cugd_qvten)
    tts, tqs = to3(cugd_ttens), to3(cugd_qvtens)
    cugd_spread = cugd_avedx // 2
    if cugd_spread > 0:
        jj = jnp.arange(ny)[:, None]
        ii = jnp.arange(nx)[None, :]
        rt, rq = tt, tq
        for nn in (-1, 0, 1):
            jdo = jnp.clip(jj + nn, 0, ny - 1)
            for kk in (-1, 0, 1):
                ido = jnp.clip(ii + kk, 0, nx - 1)
                rt = rt + 1.0 * tts[jdo, ido]
                rq = rq + 1.0 * tqs[jdo, ido]
    else:
        rt = tt + tts
        rq = tq + tqs
    # horizontal 1-2-1 in i (zero outside the domain), then in j
    def sm_i(a):
        am = jnp.pad(a, ((0, 0), (1, 1), (0, 0)))
        return 0.25 * (am[:, :-2] + 2.0 * am[:, 1:-1] + am[:, 2:])

    def sm_j(a):
        am = jnp.pad(a, ((1, 1), (0, 0), (0, 0)))
        return 0.25 * (am[:-2] + 2.0 * am[1:-1] + am[2:])

    rth = sm_j(sm_i(rt))
    rqv = sm_j(sm_i(rq))
    interior = _interior_mask(ny, nx, False, False)[:, :, None]
    zero = jnp.zeros_like(rth)
    rth = jnp.where(interior, rth, zero)
    rqv = jnp.where(interior, rqv, zero)
    qv = to3(moist_qv)
    k = _kidx(kx)
    kin = (k >= 1)[None, None, :]
    # moistening limiter
    thresh = jnp.asarray(1.0e-20, D)
    qmem1 = qv + rqv * dt
    neg = kin & (rqv < 0.0) & (qmem1 < thresh)
    ratio = ((thresh - qv) / dt) / rqv
    qmemf = jnp.min(jnp.where(neg, ratio, jnp.inf), axis=-1)
    qmemf = jnp.clip(jnp.minimum(jnp.asarray(1.0, D), qmemf), 0.0, 1.0)
    interior2 = interior[..., 0]
    qmemf = jnp.where(interior2, qmemf, jnp.asarray(1.0, D))
    f3 = qmemf[..., None]
    rqv = rqv * f3
    rth = rth * f3
    rqc = to3(rqccuten) * f3
    rqi = to3(rqicuten) * f3
    rain = raincv.reshape(ny, nx) * qmemf
    prat = pratec.reshape(ny, nx) * qmemf
    # heating limiter
    q1 = jnp.abs(rth) * 86400.0
    big = kin & (q1 > 200.0)
    qmemf2 = jnp.min(jnp.where(big, 200.0 / q1, jnp.inf), axis=-1)
    qmemf2 = jnp.maximum(jnp.asarray(0.0, D), jnp.minimum(jnp.asarray(1.0, D), qmemf2))
    qmemf2 = jnp.where(interior2, qmemf2, jnp.asarray(1.0, D))
    g3 = qmemf2[..., None]
    rain = rain * qmemf2
    prat = prat * qmemf2
    rqv = rqv * g3
    rth = rth * g3
    rqc = rqc * g3
    rqi = rqi * g3
    # vertical 1-2-1 on k = 3 .. kx-2
    def sm_k(a):
        km = jnp.concatenate([a[..., :1], a[..., :-1]], axis=-1)
        kp = jnp.concatenate([a[..., 1:], a[..., -1:]], axis=-1)
        s = 0.25 * (km + 2.0 * a + kp)
        kk = k[None, None, :]
        return jnp.where(interior & (kk >= 3) & (kk <= kx - 2), s, a)

    rth = sm_k(rth)
    rqv = sm_k(rqv)
    rth = jnp.where(interior, rth / to3(pi_phy), rth)
    flat = lambda a: a.reshape(ny * nx, kx + 1)
    return dict(RTHCUTEN=flat(rth), RQVCUTEN=flat(rqv), RQCCUTEN=flat(rqc), RQICUTEN=flat(rqi),
                RAINCV=rain.reshape(ny * nx), PRATEC=prat.reshape(ny * nx))


# ---------------------------------------------------------------------------
# Grell-Devenyi (module_cu_gd.F, cu_physics = 93)
# ---------------------------------------------------------------------------

_GD_PCRIT = (850., 800., 750., 700., 650., 600., 550., 500., 450., 400., 350., 300., 250., 200.,
             150.)
_GD_ACRIT = (.0633, .0445, .0553, .0664, .075, .1082, .1521, .2216, .3151, .3677, .41, .5255,
             .7663, 1.1686, 1.6851)
_GD_ACRITT = (.203, .515, .521, .566, .625, .665, .659, .688, .743, .813, .886, .947, 1.138,
              1.377, 1.896)


def cup_dellas_gd(z_cup, p_cup, hcd, edt, zd, cdd, he, mentrd_rate, zu, g, cd, hc, ktop, k22,
                  kbcon, mentr_rate, jmin, he_cup, kdet, kpbl, kx, *, della1):
    """GD cup_dellas (deep) -> della on k = 1..kx (della(1) from cup_dellabot)."""
    zero = jnp.zeros_like(z_cup)
    k = _kidx(kx)
    kp1 = jnp.minimum(k + 1, kx)
    km1 = jnp.maximum(k - 1, 0)
    dz = z_cup[kp1] - z_cup
    zdk1 = zd[kp1]
    detdo = edt * cdd * dz * zdk1
    entdo = edt * mentrd_rate * dz * zdk1
    subin = zu[kp1] - zdk1 * edt
    upr = (k >= kbcon) & (k < ktop)
    entup = jnp.where(upr, mentr_rate * dz * zu, zero)
    detup = jnp.where(upr, cd[kp1] * dz * zu, zero)
    subdown = (zu - zd * edt)
    entdoj = jnp.where(k == jmin, edt * zd, zero)
    entupk = jnp.where(k == k22 - 1, zu[kpbl], zero)
    detdo = jnp.where(k > kdet, zero, detdo)
    is_top = k == ktop
    detupk = jnp.where(is_top, zu[ktop], zero)
    subin = jnp.where(is_top, zero, subin)
    detup = jnp.where(k < kbcon, zero, detup)
    dp = 100.0 * (p_cup[km1] - p_cup)
    della = (subin * he_cup[kp1]
             - subdown * he_cup
             + detup * 0.5 * (hc[kp1] + hc)
             + detdo * 0.5 * (hcd[kp1] + hcd)
             - entup * he
             - entdo * he
             - entupk * he_cup[k22]
             - entdoj * he_cup[jmin]
             + detupk * hc[ktop]) * g / dp
    active = (k >= 2) & (k <= kx - 1) & (k <= ktop)
    della = jnp.where(active, della, zero)
    return della.at[1].set(della1)


def _forcing_ens_gd(aa0, aa1, xaa0, mbdt2, dtime, ierr2, ierr3, mconv, p_cup, ktop, omeg, k22,
                    kbcon, pr_blk, edt, xland1, ichoice, closure_n):
    """One iedt block of GD cup_forcing_ens for an ierr == 0 column."""
    zero = _c(0.0, aa0)
    kx = omeg.shape[0] - 1
    k = _kidx(kx)
    pk = p_cup[ktop]
    pcrit = jnp.asarray(_GD_PCRIT, aa0.dtype)
    acrit = jnp.asarray(_GD_ACRIT, aa0.dtype)
    acritt = jnp.asarray(_GD_ACRITT, aa0.dtype)
    lt = pk < pcrit  # (15,)
    idx = jnp.arange(1, 16, dtype=jnp.int32)
    kclim = jnp.max(jnp.where(lt, idx, 0))
    kclim = jnp.where(jnp.any(lt), kclim, 1)
    kclim = jnp.maximum(kclim, 1)
    kk2 = jnp.maximum(kclim - 1, 1)
    aclim1 = acrit[kclim - 1] * 1.0e3
    aclim2 = acrit[kk2 - 1] * 1.0e3
    aclim3 = acritt[kclim - 1] * 1.0e3
    aclim4 = acritt[kk2 - 1] * 1.0e3
    xff0 = (aa1 - aa0) / dtime
    xff = [zero] * 17
    xff[1] = (aa1 - aa0) / dtime
    xff[2] = 0.9 * xff[1]
    xff[3] = 1.1 * xff[1]
    xff[4] = -omeg[k22] / 9.81
    xff[5] = -omeg[kbcon] / 9.81
    x6 = -omeg[1] / 9.81
    rng = (k >= 2) & (k <= kbcon - 1)
    xff[6] = jnp.maximum(x6, jnp.max(jnp.where(rng, -omeg / 9.81, -jnp.inf)))
    xff[7] = xff[8] = xff[9] = mconv
    xff[10] = aa1 / (_c(60.0, aa0) * _c(20.0, aa0))
    xff[11] = aa1 / (_c(60.0, aa0) * _c(30.0, aa0))
    xff[12] = aa1 / (_c(60.0, aa0) * _c(40.0, aa0))
    xff[13] = jnp.maximum(zero, (aa1 - aclim1) / dtime)
    xff[14] = jnp.maximum(zero, (aa1 - aclim2) / dtime)
    xff[15] = jnp.maximum(zero, (aa1 - aclim3) / dtime)
    xff[16] = jnp.maximum(zero, (aa1 - aclim4) / dtime)
    xk = (xaa0 - aa1) / mbdt2
    xk = jnp.where((xk <= 0.0) & (xk > -1.0e-6), _c(-1.0e-6, aa0), xk)
    xk = jnp.where((xk > 0.0) & (xk < 1.0e-6), _c(1.0e-6, aa0), xk)
    water = xland1 < 0.1
    e23 = (ierr2 > 0) | (ierr3 > 0)
    xf_rows, mf_rows = [], []
    for ne in range(MAXENS):
        kill23 = water & e23
        for n in (1, 2, 3, 7, 8, 9):
            xff[n] = jnp.where(kill23, zero, xff[n])
        closure_n = jnp.where(kill23, closure_n - 1.0 - 1.0, closure_n)
        for n in (4, 5, 6, 10, 11, 12, 13, 14, 15):
            xff[n] = jnp.where(water, zero, xff[n])
        dec = 3.0 + (3.0 + 4.0 if ne == 0 else 0.0)
        closure_n = jnp.where(water, closure_n - 3.0, closure_n)
        if ne == 0:
            closure_n = jnp.where(water, closure_n - 3.0, closure_n)
            closure_n = jnp.where(water, closure_n - 4.0, closure_n)
        del dec
        xkn = xk[ne]
        xf = [zero] * 17
        pos0 = xff0 > 0.0
        for n in (1, 2, 3, 13, 14, 15, 16):
            xf[n] = jnp.where(pos0, jnp.maximum(zero, -xff[n] / xkn) + zero, zero)
        for n in (4, 5, 6):
            xf[n] = jnp.maximum(zero, xff[n] + zero)
        for n in (7, 8, 9):
            a1 = jnp.maximum(_c(1.0e-3, aa0), pr_blk[ne, n - 1])
            xf[n] = jnp.maximum(zero, xff[n] / a1)
        kneg = xkn < 0.0
        for n in (10, 11, 12):
            xf[n] = jnp.where(kneg, jnp.maximum(zero, -xff[n] / xkn) + zero, zero)
        if ichoice >= 1:
            closure_n = zero
            pick = xf[ichoice]
            for n in range(1, 17):
                xf[n] = pick
        if ichoice == 0:
            xf[14] = xf[13]
        mf = [jnp.maximum(zero, edt * xf[n]) for n in range(1, 17)]
        xfv = jnp.stack(xf[1:17])
        mfv = jnp.stack(mf)
        if ne == 1:
            xfv = jnp.where(ierr2 > 0, jnp.zeros_like(xfv), xfv)
            mfv = jnp.where(ierr2 > 0, jnp.zeros_like(mfv), mfv)
        if ne == 2:
            xfv = jnp.where(ierr3 > 0, jnp.zeros_like(xfv), xfv)
            mfv = jnp.where(ierr3 > 0, jnp.zeros_like(mfv), mfv)
        xf_rows.append(xfv)
        mf_rows.append(mfv)
    xf_blk = jnp.stack(xf_rows)
    mf_blk = jnp.stack(mf_rows)
    copy_idx = jnp.array([4, 5, 6, 7, 8, 9, 10, 11, 12], jnp.int32) - 1
    xf_blk = xf_blk.at[0, copy_idx].set(xf_blk[1, copy_idx])
    return xf_blk, mf_blk, closure_n


def gd_cup_enss_column(t, q, z1, tn, qo, po, psur, us, vs, omeg, mconv, xland, gsw, dtime, xl,
                       rv, cp, g, *, kx, ichoice=0):
    """One column of WRF GD ``CUP_enss`` + ``neg_check`` (iens = 1)."""
    D = t.dtype
    zero = _c(0.0, t)
    k = _kidx(kx)
    p = po
    tcrit = _c(258.0, t)
    xland1 = jnp.where(xland > 1.5, zero, _c(1.0, t))
    cap_max_increment = _c(25.0, t)
    radius = _c(12000.0, t)
    entr_rate = 0.2 / radius
    mentrd_rate = zero
    mentr_rate = entr_rate
    cd = jnp.full((kx + 1,), 0.1, D) * entr_rate
    edtmax = _c(0.8, t)
    edtmin = _c(0.2, t)
    depth_min = _c(500.0, t)
    kstabm = jnp.int32(kx - 1)
    ierr = jnp.int32(0)
    cap_max = jnp.where(gsw < 1.0, _c(25.0, t), _c(75.0, t))
    zkbmax = _c(4000.0, t)
    zcutdown = _c(3000.0, t)
    z_detr = _c(1250.0, t)
    mbdt_ens = jnp.stack([(_c(float(n), t) - 3.0) * dtime * 1.0e-3 + dtime * 5.0e-03
                          for n in range(1, MAXENS + 1)])
    z, qes, he, hes, q = cup_env(t, q, p, z1, psur, tcrit, 0, xl, cp, kx, variant="gd")
    zo, qeso, heo, heso, qo = cup_env(tn, qo, po, z1, psur, tcrit, 0, xl, cp, kx, variant="gd")
    qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup = cup_env_clev(
        t, qes, q, he, hes, z, p, psur, z1, xl, rv, cp, kx)
    (qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup,
     tn_cup) = cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, xl, rv, cp, kx)
    found, kf = _first_true((k >= 1) & (k <= kx - 2) & (zo_cup > zkbmax + z1), kx)
    kbmax = jnp.where(found, kf, 1).astype(jnp.int32)
    kdet = _first_above(zo_cup, z_detr + z1, kx, 0)
    k22 = cup_maximi(heo_cup, 3, kbmax, ierr, kx)
    ierr = jnp.where((ierr == 0) & (k22 >= kbmax), 2, ierr).astype(jnp.int32)
    kbcon, k22, ierr = cup_kbcon(cap_max_increment, 1, k22, heo_cup, heso_cup, ierr, kbmax,
                                 po_cup, cap_max, kx, variant="gd")
    kstabi = cup_minimi(heso_cup, kbcon, kstabm, ierr, kx)
    cd = _cd_stable(cd, kstabi, kstabm, 1.5 * entr_rate, 10.0 * entr_rate, ierr == 0, kx)
    _hkb, hc, dby = cup_up_he(k22, z_cup, cd, mentr_rate, he_cup, kbcon, he, hes_cup, kx)
    _hkbo, hco, dbyo = cup_up_he(k22, zo_cup, cd, mentr_rate, heo_cup, kbcon, heo, heso_cup, kx)
    ktop, dbyo, ierr = cup_ktop(1, dbyo, kbcon, ierr, kx, variant="gd")
    zktop = (zo_cup[ktop] - z1) * 0.6
    zktop = jnp.minimum(zktop + z1, zcutdown + z1)
    kzdown = jnp.where(ierr == 0, _first_above(zo_cup, zktop, kx, 0), 0).astype(jnp.int32)
    jmin = cup_minimi(heso_cup, k22, kzdown, ierr, kx)
    jmin, kdet, ierr = _jmin_search(jmin, kdet, ktop, heso_cup, zo_cup, ierr, kx)
    ierr = jnp.where((ierr == 0) & (-zo_cup[kbcon] + zo_cup[ktop] < depth_min), 6, ierr)
    zu = cup_up_nms(z_cup, mentr_rate, cd, kbcon, ktop, k22, kx)
    zuo = cup_up_nms(zo_cup, mentr_rate, cd, kbcon, ktop, k22, kx)
    zd, cdd = cup_dd_nms(z_cup, None, mentrd_rate, jmin, 0, kdet, z1, kx)
    cdd = jnp.where(ierr == 0, cdd, jnp.zeros_like(cdd))
    zdo, _ = cup_dd_nms(zo_cup, cdd, mentrd_rate, jmin, 1, kdet, z1, kx)
    hcd, _ = cup_dd_he(hes_cup, z_cup, cdd, mentrd_rate, jmin, he, kx)
    hcdo, _ = cup_dd_he(heso_cup, zo_cup, cdd, mentrd_rate, jmin, heo, kx)
    _qcdo, qrcdo, pwdo, pwevo, buo = cup_dd_moisture(
        zdo, hcdo, heso_cup, qeso_cup, qo_cup, zo_cup, cdd, mentrd_rate, jmin, gammao_cup, qo,
        xl, kx)
    ierr = jnp.where((ierr == 0) & ((pwevo == 0.0) | (buo >= 0.0)), 7, ierr)
    _qc, qrc, _pw, _pwav, _clw = cup_up_moisture(z_cup, kbcon, ktop, cd, dby, mentr_rate, q,
                                                 gamma_cup, zu, qes_cup, k22, q_cup, xl, kx,
                                                 c0=0.002)
    cupclw = qrc
    qco, qrco, pwo, pwavo, _clwo = cup_up_moisture(zo_cup, kbcon, ktop, cd, dbyo, mentr_rate,
                                                   qo, gammao_cup, zuo, qeso_cup, k22, qo_cup,
                                                   xl, kx, c0=0.002)
    aa0 = cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, kx)
    aa1 = cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, kx)
    ierr = jnp.where((ierr == 0) & (aa1 == 0.0), 17, ierr).astype(jnp.int32)
    edtc = cup_dd_edt(us, vs, zo, ktop, kbcon, po, pwavo, pwevo, edtmax, edtmin, kx,
                      variant="gd")
    ens_shape = (MAXENS, MAXENS3)
    scr1 = jnp.where((k >= 1) & (k <= kx - 1), qco - qrco, zero)

    def iedt_body(carry, iedt):
        ierr_c, pr_ens, xf_ens, massfln, closure_n = carry
        ok = ierr_c == 0
        edto = edtc[iedt]
        della1_h = cup_dellabot(heo_cup, zo_cup, po, hcdo, edto, zdo, cdd, heo, mentrd_rate, g)
        della1_q = cup_dellabot(qo_cup, zo_cup, po, qrcdo, edto, zdo, cdd, qo, mentrd_rate, g)
        dellah = cup_dellas_gd(zo_cup, po_cup, hcdo, edto, zdo, cdd, heo, mentrd_rate, zuo, g,
                               cd, hco, ktop, k22, kbcon, mentr_rate, jmin, heo_cup, kdet, k22,
                               kx, della1=della1_h)
        kp1 = jnp.minimum(k + 1, kx)
        dpc = po_cup - po_cup[kp1]
        dq_top = 0.01 * zuo[ktop] * qrco[ktop] * 9.81 / dpc
        dzc = zo_cup[kp1] - zo_cup
        dq_mid = (_c(0.01, t) * _c(9.81, t)) * cd * dzc * zuo * 0.5 * (qrco + qrco[kp1]) / dpc
        dellaqc = jnp.where(k == ktop, dq_top, zero)
        dellaqc = jnp.where((k < ktop) & (k > kbcon), dq_mid, dellaqc)
        dellaqc = jnp.where((k >= 1) & (k <= kx - 1) & ok, dellaqc, zero)
        dellaq = cup_dellas_gd(zo_cup, po_cup, qrcdo, edto, zdo, cdd, qo, mentrd_rate, zuo, g,
                               cd, scr1, ktop, k22, kbcon, mentr_rate, jmin, qo_cup, kdet, k22,
                               kx, della1=della1_q)
        mbdt = mbdt_ens[1]
        rcp = _c(1.0, t) / cp
        xhe = dellah * mbdt + heo
        xq = dellaq * mbdt + qo
        dellat = rcp * (dellah - xl * dellaq)
        xt = dellat * mbdt + tn
        xq = jnp.where(xq <= 0.0, _c(1.0e-08, t), xq)
        xhe = xhe.at[kx].set(heo[kx])
        xq = xq.at[kx].set(jnp.where(qo[kx] <= 0.0, _c(1.0e-08, t), qo[kx]))
        xt = xt.at[kx].set(tn[kx])
        dellat = jnp.where(ok, dellat, zero)
        xz, xqes, xhe, xhes, xq = cup_env(xt, xq, po, z1, psur, tcrit, 2, xl, cp, kx, he_in=xhe,
                                          variant="gd")
        (_xqesc, _xqc, xhe_cup, xhes_cup, xz_cup, _xpc, xgamma_cup,
         xt_cup) = cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, psur, z1, xl, rv, cp, kx)
        _xhkb, _xhc, xdby = cup_up_he(k22, xz_cup, cd, mentr_rate, xhe_cup, kbcon, xhe,
                                      xhes_cup, kx)
        xzu = cup_up_nms(xz_cup, mentr_rate, cd, kbcon, ktop, k22, kx)
        xaa0 = cup_up_aa0(xz, xzu, xdby, xgamma_cup, xt_cup, kbcon, ktop, kx)
        in_top = (k >= 1) & (k <= ktop)

        def pr_step(cc, kk):
            a7, ad = cc
            use = in_top[kk]
            n7 = a7 + pwo[kk]
            nd = ad + pwo[kk] + edto * pwdo[kk]
            return (jnp.where(use, n7, a7), jnp.where(use, nd, ad)), None

        (a7, ad), _ = _scan_up(pr_step, (zero, zero), kx, k_lo=1)
        row = jnp.stack([ad] * 6 + [a7, a7, a7] + [ad] * 7)
        kill18 = ok & (a7 < 1.0e-6)
        row = jnp.where(row < 1.0e-4, jnp.zeros_like(row), row)
        blk_ok = ok & ~kill18
        pr_blk = jnp.where(blk_ok, jnp.broadcast_to(row, ens_shape), jnp.zeros(ens_shape, D))
        pr_ens = pr_ens.at[iedt].set(jnp.where(ok, pr_blk, pr_ens[iedt]))
        ierr_c = jnp.where(kill18, 18, ierr_c).astype(jnp.int32)
        ok = ierr_c == 0
        k22x = cup_maximi(heo_cup, 3, kbmax, ierr_c, kx)
        _kbx2, k22x2, ierr2 = cup_kbcon(cap_max_increment, 2, k22x, heo_cup, heso_cup, ierr_c,
                                        kbmax, po_cup, cap_max, kx, variant="gd")
        _kbx3, _k22x3, ierr3 = cup_kbcon(cap_max_increment, 3, k22x2, heo_cup, heso_cup,
                                         ierr_c, kbmax, po_cup, cap_max, kx, variant="gd")
        xaa0_ens = jnp.stack([xaa0] * MAXENS)
        xf_blk, mf_blk, cn = _forcing_ens_gd(aa0, aa1, xaa0_ens, mbdt_ens[1], dtime, ierr2,
                                             ierr3, mconv, po_cup, ktop, omeg, k22, kbcon,
                                             pr_ens[iedt], edto, xland1, ichoice, closure_n)
        closure_n = jnp.where(ok, cn, closure_n)
        xf_ens = xf_ens.at[iedt].set(jnp.where(ok, xf_blk, xf_ens[iedt]))
        massfln = massfln.at[iedt].set(jnp.where(ok, mf_blk, massfln[iedt]))
        clear = (ierr_c != 0) & (ierr_c != 20)
        xf_ens = jnp.where(clear, jnp.zeros_like(xf_ens), xf_ens)
        massfln = jnp.where(clear, jnp.zeros_like(massfln), massfln)
        ens = (jnp.where(ok, dellat, zero), jnp.where(ok, dellaq, zero),
               jnp.where(ok, dellaqc, zero), jnp.where(ok, pwo + edto * pwdo, zero), ierr2,
               ierr3)
        return (ierr_c, pr_ens, xf_ens, massfln, closure_n), ens

    carry0 = (ierr, jnp.zeros((MAXENS2,) + ens_shape, D), jnp.zeros((MAXENS2,) + ens_shape, D),
              jnp.zeros((MAXENS2,) + ens_shape, D), _c(16.0, t))
    (ierr, pr_ens, xf_ens, massfln, closure_n), ens = lax.scan(
        iedt_body, carry0, jnp.arange(MAXENS2, dtype=jnp.int32))
    dellat_ens, dellaq_ens, dellaqc_ens, pwo_ens, ierr2s, ierr3s = ens
    ierr2, ierr3 = ierr2s[-1], ierr3s[-1]
    out = _cup_output_ens_gd(xf_ens, ierr, dellat_ens, dellaq_ens, dellaqc_ens, pwo_ens, ktop,
                             pr_ens, closure_n, xland1, ierr2, ierr3, kx)
    pre = jnp.maximum(out["pre"], zero)
    outt, outq, outqc, pre = _neg_check(dtime, q, out["outq"], out["outt"], out["outqc"], pre,
                                        kx)
    return dict(outt=outt, outq=outq, outqc=outqc, pre=pre, kbcon=kbcon, ktop=ktop,
                cupclw=cupclw, xf_ens=out["xf_ens"], pr_ens=pr_ens.reshape(ENSDIM),
                apr=out["apr"], ierr=out["ierr"])


def _cup_output_ens_gd(xf_ens, ierr, dellat, dellaq, dellaqc, pw, ktop, pr_ens, closure_n,
                       xland1, ierr2, ierr3, kx):
    D = dellat.dtype
    zero = jnp.zeros((), D)
    ok = ierr == 0
    xf = xf_ens.reshape(ENSDIM)
    prf = pr_ens.reshape(ENSDIM)
    xf = jnp.where(ok & (prf <= 0.0), jnp.zeros_like(xf), xf)
    xmb_ave, x_ave, x_cap, ok1 = _massflx_stats(xf, ierr, "gd")
    _pr_ave, p_ave, p_cap, ok2 = _massflx_stats(prf, ierr, "gd")
    w_x = _apr_weights(x_ave, x_cap, "gd")
    w_p = _apr_weights(p_ave, p_cap, "gd")
    pr_w = [jnp.where(ok1, a, zero) for a in w_x]
    apr = jnp.stack([jnp.where(ok2, b * 3600.0 * a, zero) for a, b in zip(pr_w, w_p)])
    ierr = jnp.where(ok & (xmb_ave <= 0.0), 13, ierr)
    xmb_ave = jnp.where(ok & (xmb_ave <= 0.0), zero, xmb_ave)
    xmb = jnp.maximum(0.1 * xmb_ave, xmb_ave - zero * zero)
    clos_wei = 16.0 / jnp.maximum(_c(1.0, dellat), closure_n)
    xmb = jnp.where(xland1 < 0.5, xmb * clos_wei, xmb)
    xmb = jnp.where(ok, xmb, zero)
    ierr = jnp.where(ok & (xmb == 0.0), 19, ierr)
    ierr = jnp.where(ok & (xmb > 100.0), 19, ierr)
    xfac1 = xmb_ave
    nx = MAXENS2
    fnx = _c(float(nx), dellat)
    ddtes = _c(200.0, dellat)
    ok_out = ierr == 0
    dtt = zero + dellat[0]
    dtq = zero + dellaq[0]
    dtqc = zero + dellaqc[0]
    dtpw = zero + pw[0]
    for n in range(1, nx):
        dtt = dtt + dellat[n]
        dtq = dtq + dellaq[n]
        dtqc = dtqc + dellaqc[n]
        dtpw = dtpw + pw[n]

    def kstep(carry, kk):
        xm, pre = carry
        use = ok_out & (kk <= ktop)
        outtes = dtt[kk] * xm * 86400.0 / fnx
        c1 = (outtes > 2.0 * ddtes) & (kk > 2)
        xm = jnp.where(use & c1, 2.0 * ddtes / outtes * xm, xm)
        outtes = jnp.where(c1, 1.0 * ddtes, outtes)
        c2 = outtes < -ddtes
        xm = jnp.where(use & c2, -ddtes / outtes * xm, xm)
        outtes = jnp.where(c2, -ddtes, outtes)
        c3 = (outtes > 0.5 * ddtes) & (kk <= 2)
        xm = jnp.where(use & c3, ddtes / outtes * xm, xm)
        o_t = xm * dtt[kk] / fnx
        o_q = xm * dtq[kk] / fnx
        o_c = xm * dtqc[kk] / fnx
        pre_n = jnp.where(use, pre + xm * dtpw[kk] / fnx, pre)
        return (xm, pre_n), (jnp.where(use, o_t, zero), jnp.where(use, o_q, zero),
                             jnp.where(use, o_c, zero))

    (xmb, pre), (ot, oq, oc) = _scan_up(kstep, (xmb, zero), kx, k_lo=1)
    z1 = jnp.zeros((1,), D)
    outt = jnp.concatenate([z1, ot])
    outq = jnp.concatenate([z1, oq])
    outqc = jnp.concatenate([z1, oc])
    prerate = pre * 3600.0
    k221 = ok_out & (prerate < 0.1) & ((ierr2 > 0) | (ierr3 > 0))
    pre = jnp.where(k221, zero, pre)
    ierr = jnp.where(k221, 221, ierr)
    outt = jnp.where(k221, jnp.zeros_like(outt), outt)
    outq = jnp.where(k221, jnp.zeros_like(outq), outq)
    outqc = jnp.where(k221, jnp.zeros_like(outqc), outqc)
    xf = jnp.where(k221, jnp.zeros_like(xf), xf)
    ok_fin = ierr == 0
    fac = xmb / jnp.where(ok_fin, xfac1, _c(1.0, dellat))
    xf = jnp.where(ok_fin, xf * fac, xf)
    return dict(outt=outt, outq=outq, outqc=outqc, pre=pre, xf_ens=xf, apr=apr,
                ierr=ierr.astype(jnp.int32))


def _neg_check(dt, q, outq, outt, outqc, pret, kx):
    """GD neg_check (applied to every column, gated on nothing)."""
    D = outt.dtype
    one = _c(1.0, outt)
    k = _kidx(kx)
    kin = k >= 1
    thresh = _c(200.01, outt)
    two_t = _c(2.0, outt) * thresh
    qmem = outt * 86400.0
    c_hi = kin & (qmem > two_t)
    c_lo = kin & (qmem < -thresh)
    r = jnp.minimum(jnp.min(jnp.where(c_hi, two_t / qmem, jnp.inf)),
                    jnp.min(jnp.where(c_lo, -thresh / qmem, jnp.inf)))
    qmemf = jnp.minimum(one, r)
    outq = outq * qmemf
    outt = outt * qmemf
    outqc = outqc * qmemf
    pret = pret * qmemf
    thr2 = _c(1.0e-10, outt)
    qtest = q + outq * dt
    c = kin & (jnp.abs(outq) > 0.0) & (qtest < thr2)
    r2 = jnp.min(jnp.where(c, ((thr2 - q) / dt) / outq, jnp.inf))
    qmemf2 = jnp.minimum(one, r2)
    return outt * qmemf2, outq * qmemf2, outqc * qmemf2, pret * qmemf2


def grelldrv_tile(*, u, v, w, t, q, p, pi, rho, p8w, rthften, rqvften, rthraten, rthblten,
                  rqvblten, ht, xland, gsw, dt, htop=None, hbot=None, ichoice=0,
                  periodic_x=False, periodic_y=False, xlv=2.5e6, cp=1004.5, g=9.81,
                  r_v=461.6):
    """WRF ``cu_physics=93``: GRELLDRV on one tile (same layout as :func:`g3drv_tile`).

    ``w`` is WRF's staggered ``w_2`` (kx+1 levels): GRELLDRV uses w(i,k,j) on
    k = 1..kx, i.e. the bottom face of mass level k.  ``htop``/``hbot`` are the
    running cloud-top/base diagnostics (GD does not reset them).
    """
    D = t.dtype
    kx, ny, nx = t.shape
    ncol = ny * nx
    c = lambda val: jnp.asarray(val, D)
    dt = c(dt)
    xlv_, cp_, g_, rv_ = c(xlv), c(cp), c(g), c(r_v)
    rthf = (rthften + rthraten + rthblten) * pi
    rqvf = rqvften + rqvblten
    t2d = t
    q2d = q
    omeg = -g_ * rho * w[:kx]
    tn = t2d + rthf * dt
    tn = jnp.where(tn < 200.0, t2d, tn)
    qo = q2d + rqvf * dt
    q2d = jnp.where(q2d < 1.0e-08, c(1.0e-08), q2d)
    qo = jnp.where(qo < 1.0e-08, c(1.0e-08), qo)
    po = p * 0.01
    psur = p8w[0] * 0.01
    dq = q2d[1:] - q2d[:-1]
    acc = jnp.zeros((ny, nx), D)
    for kk in range(kx - 1):
        acc = acc + omeg[kk] * dq[kk] / g_
    mconv = jnp.where(acc < 0.0, jnp.zeros_like(acc), acc)
    cols = _to_cols1
    col_fn = functools.partial(gd_cup_enss_column, kx=kx, ichoice=ichoice)
    res = jax.vmap(col_fn, in_axes=(0,) * 14 + (None,) * 4)(
        cols(t2d), cols(q2d), ht.reshape(ncol), cols(tn), cols(qo), cols(po),
        psur.reshape(ncol), cols(u), cols(v), cols(omeg), mconv.reshape(ncol),
        xland.reshape(ncol), gsw.reshape(ncol), jnp.broadcast_to(dt, (ncol,)),
        xlv_, rv_, cp_, g_)
    rows = _interior_mask(ny, nx, True, periodic_y)  # j-interior only
    interior = _interior_mask(ny, nx, periodic_x, periodic_y)
    rows = rows.reshape(ncol)
    interior = interior.reshape(ncol)
    pret = jnp.where(interior, res["pre"], c(0.0))
    rain = pret > 0.0
    cuten = jnp.where(rain, c(1.0), c(0.0))
    cut = cuten[:, None]
    rowc = rows[:, None]
    zk = jnp.zeros_like(res["outt"])
    pic = cols(pi)
    rth = jnp.where(rowc, res["outt"] * cut / pic, zk)
    rqv = jnp.where(rowc, res["outq"] * cut, zk)
    cold = cols(t2d) < 258.0
    qcc = res["outqc"] * cut
    rqc = jnp.where(rowc & ~cold, qcc, zk)
    rqi = jnp.where(rowc & cold, qcc, zk)
    gdc = jnp.where(rowc, res["cupclw"] * cut, zk)
    gdc2 = jnp.where(rowc & cold, res["cupclw"] * cut, zk)
    htop0 = (jnp.ones((ncol,), D) if htop is None else htop.reshape(ncol).astype(D))
    hbot0 = (jnp.full((ncol,), float(kx), D) if hbot is None else hbot.reshape(ncol).astype(D))
    ktop_f = res["ktop"].astype(D)
    kbcon_f = res["kbcon"].astype(D)
    htop_o = jnp.where(rows & rain & (ktop_f > htop0), ktop_f + 0.001, htop0)
    hbot_o = jnp.where(rows & rain & (kbcon_f < hbot0), kbcon_f + 0.001, hbot0)
    out = dict(
        RTHCUTEN=rth, RQVCUTEN=rqv, RQCCUTEN=rqc, RQICUTEN=rqi, GDC=gdc, GDC2=gdc2,
        RAINCV=jnp.where(rows & rain, pret * dt, c(0.0)),
        PRATEC=jnp.where(rows & rain, pret, c(0.0)),
        HTOP=htop_o, HBOT=hbot_o,
        KTOP_DEEP=jnp.where(rows, res["ktop"], 0),
        XF_ENS=res["xf_ens"], PR_ENS=res["pr_ens"], APR=res["apr"],
        IERR=res["ierr"].reshape(ny, nx),
    )
    for key in ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "GDC", "GDC2"):
        out[key] = _from_cols1(out[key], ny, nx)
    for key in ("RAINCV", "PRATEC", "HTOP", "HBOT", "KTOP_DEEP"):
        out[key] = out[key].reshape(ny, nx)
    out["DRV_RAINCV"] = out["RAINCV"]
    out["DRV_PRATEC"] = out["PRATEC"]
    out["XF_ENS"] = out["XF_ENS"].reshape(ny, nx, ENSDIM)
    out["PR_ENS"] = out["PR_ENS"].reshape(ny, nx, ENSDIM)
    out["APR"] = out["APR"].reshape(ny, nx, 8)
    return out
