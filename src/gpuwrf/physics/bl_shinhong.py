"""JIT/vmap-traceable Shin-Hong scale-aware PBL kernel.

This ports the operational single-column path of WRF ``module_bl_shinhong.F``
from the v0.90 host reference in :mod:`gpuwrf.physics.pbl_shinhong` into pure
JAX.  Profile arrays are bottom-up mass-level columns.  The batched
``shinhong_columns`` entry is the scan-adapter endpoint for
``bl_pbl_physics=11``.

Validation status: the forecast-driving path (U/V/theta/qv tendencies, EXCH_H,
PBLH/KPBL/WSTAR/DELTA) is exact to the v090 host reference at roundoff.  TKE_PBL
and EL_PBL are emitted and qke is updated, but exact diagnostic parity is not
claimed yet: worst measured residuals are TKE rel ~=0.285 and EL rel ~=0.013
against the v090 PARTIAL/fp32-sensitive reference.  TKE is non-driving for this
operational dynamics path; refine it when a faithful pristine-WRF Shin-Hong TKE
oracle is built.
"""

from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
XLV = 2.5e6
EP1 = R_V / R_D - 1.0
EP2 = R_D / R_V
KARMAN = 0.4

XKZMINM = 0.1
XKZMINH = 0.01
XKZMAX = 1000.0
RIMIN = -100.0
RLAM = 30.0
PRMIN = 0.25
PRMAX = 4.0
BRCR_UB = 0.0
BRCR_SB = 0.25
CORI = 1.0e-4
AFAC = 6.8
BFAC = 6.8
PFAC = 2.0
PFAC_Q = 2.0
PHIFAC = 8.0
SFCFRAC = 0.1
D1 = 0.02
D2 = 0.05
D3 = 0.001
H1 = 0.33333333
H2 = 0.6666667
CKZ = 0.001
ZFMIN = 1.0e-8
APHI5 = 5.0
APHI16 = 16.0
TMIN = 1.0e-2
GAMCRT = 3.0
GAMCRQ = 2.0e-3
IMVDIF = 1

EPSQ2L = 0.01
C_1 = 1.0
GAMCRE = 0.224

MLTOP = 1.0
SFCFRACN1 = 0.075
NLFRAC = 0.7
ENLFRAC = -0.4
A11 = 1.0
A12 = -1.15
EZFAC = 1.5
CPENT = -0.4
RIGSMAX = 100.0
ENTFMIN = 1.0
ENTFMAX = 5.0


def _safe_div(a, b, default=0.0):
    return jnp.where(jnp.abs(b) > 1.0e-30, a / b, default)


def _pu(d, h):
    doh = _safe_div(d, h, 0.0)
    val = (1.0 * doh**2.0 + 0.070 * doh**0.6666667) / (
        1.0 * doh**2.0 + 0.142 * doh**0.6666667 + 0.071
    )
    return jnp.clip(jnp.where(h != 0.0, val, 1.0), 0.0, 1.0)


def _pq(d, h):
    doh = _safe_div(d, h, 0.0)
    val = 0.5 * ((doh**2.0 - 0.098) / (doh**2.0 + 0.106)) + 0.5
    return jnp.clip(jnp.where(h != 0.0, val, 1.0), 0.0, 1.0)


def _pthnl(d, h):
    doh = _safe_div(d, h, 0.0)
    val = 0.243 * (
        (doh**2.0 + 0.936 * doh**0.875 - 1.110)
        / (doh**2.0 + 0.312 * doh**0.875 + 0.329)
    ) + (1.0 - 0.243)
    return jnp.clip(jnp.where(h != 0.0, val, 1.0), 0.0, 1.0)


def _pthl(d, h):
    doh = _safe_div(d, h, 0.0)
    val = 0.280 * (
        (doh**2.0 + 0.870 * jnp.sqrt(doh) - 0.913)
        / (doh**2.0 + 0.153 * jnp.sqrt(doh) + 0.278)
    ) + (1.0 - 0.280)
    return jnp.clip(jnp.where(h != 0.0, val, 1.0), 0.0, 1.0)


def _ptke(d, h):
    return _pu(d, h)


def _thomas_scan(lower, diag, upper, rhs):
    n = rhs.shape[0]
    cp0 = upper[0] / diag[0]
    dp0 = rhs[0] / diag[0]

    def fwd(carry, k):
        cp_prev, dp_prev = carry
        denom = diag[k] - lower[k] * cp_prev
        cp_k = upper[k] / denom
        dp_k = (rhs[k] - lower[k] * dp_prev) / denom
        return (cp_k, dp_k), (cp_k, dp_k)

    _, (cp_rest, dp_rest) = jax.lax.scan(fwd, (cp0, dp0), jnp.arange(1, n))
    cp = jnp.concatenate([cp0[None], cp_rest])
    dp = jnp.concatenate([dp0[None], dp_rest])

    def bwd(x_next, k):
        x_k = dp[k] - cp[k] * x_next
        return x_k, x_k

    x_last = dp[n - 1]
    _, x_rest = jax.lax.scan(bwd, x_last, jnp.arange(n - 2, -1, -1))
    return jnp.concatenate([x_rest[::-1], x_last[None]])


def _first_pbl_guess(thv, thermal, za, br, brcr, u, v):
    def step(carry, k):
        brdn_c, brup_c, stable_c, kpbl_c = carry
        spdk2 = jnp.maximum(u[k] * u[k] + v[k] * v[k], 1.0)
        brup_new = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
        active = jnp.logical_not(stable_c)
        brdn_n = jnp.where(active, brup_c, brdn_c)
        brup_n = jnp.where(active, brup_new, brup_c)
        kpbl_n = jnp.where(active, (k + 1).astype(jnp.int32), kpbl_c)
        stable_n = jnp.where(active, brup_new > brcr, stable_c)
        return (brdn_n, brup_n, stable_n, kpbl_n), None

    init = (br, br, jnp.asarray(False), jnp.asarray(1, jnp.int32))
    (brdn, brup, _stable, kpbl), _ = jax.lax.scan(
        step, init, jnp.arange(1, thv.shape[0], dtype=jnp.int32)
    )
    return kpbl, brdn, brup


def _interp_pblh(kpbl, brdn, brup, brcr, za, zq):
    brint = jnp.where(
        brdn >= brcr,
        0.0,
        jnp.where(brup <= brcr, 1.0, (brcr - brdn) / (brup - brdn)),
    )
    k0 = jnp.maximum(kpbl - 1, 1)
    hpbl = za[k0 - 1] + brint * (za[k0] - za[k0 - 1])
    kpbl = jnp.where(hpbl < zq[1], jnp.asarray(1, jnp.int32), kpbl)
    return hpbl, kpbl, kpbl > 1


def _mixlen(u, v, t, the, q, cwm, q2, z, ustar, corf, epshol, hpbl, lpbl, pblflg,
            mf, ufxpbl, vfxpbl, qfxpbl):
    nz = u.shape[0]
    kidx = jnp.arange(nz)
    kge1 = kidx >= 1
    blckdr, cn = 0.0063, 0.75
    eps1, epsl, epsru = 1.0e-12, 0.32, 1.0e-7
    el0max, el0min, elfc = 1000.0, 1.0, 0.23 * 0.5
    alph, beta, g = 0.30, 1.0 / 273.0, 9.81
    btg = beta * g
    a1, a2x = 0.659888514560862645, 0.6574209922667784586
    b1, b2 = 11.87799326209552761, 7.226971804046074028
    c1 = 0.000830955950095854396
    adnh = 9.0 * a1 * a2x * a2x * (12.0 * a1 + 3.0 * b2) * btg * btg
    adnm = 18.0 * a1 * a1 * a2x * (b2 - 3.0 * a2x) * btg
    bdnh = 3.0 * a2x * (7.0 * a1 + b2) * btg
    bdnm = 6.0 * a1 * a1
    aeqh = 9.0 * a1 * a2x * a2x * b1 * btg * btg + adnh
    aeqm = (
        3.0 * a1 * a2x * b1 * (3.0 * a2x + 3.0 * b2 * c1 + 18.0 * a1 * c1 - b2) * btg
        + adnm
    )
    requ = -aeqh / aeqm
    epsgh = 1.0e-9
    epsgm = requ * epsgh
    epsrs = 1.0e-7
    ubryl = (
        (18.0 * requ * a1 * a1 * a2x * b2 * c1 * btg + 9.0 * a1 * a2x * a2x * b2 * btg * btg)
        / (requ * adnm + adnh)
    )
    ubry = (1.0 + epsrs) * ubryl
    ubry3 = 3.0 * ubry
    aubh = 27.0 * a1 * a2x * a2x * b2 * btg * btg - adnh * ubry3
    aubm = 54.0 * a1 * a1 * a2x * b2 * c1 * btg - adnm * ubry3
    bubh = (9.0 * a1 * a2x + 3.0 * a2x * b2) * btg - bdnh * ubry3
    bubm = 18.0 * a1 * a1 * c1 - bdnm * ubry3
    cubr = 1.0 - ubry3
    rcubr = 1.0 / cubr
    elcbl = 0.77
    elocp = 2.72e6 / CP

    dth = jnp.concatenate([jnp.zeros(1, dtype=the.dtype), the[1:] - the[:-1]])
    zkm1 = jnp.concatenate([z[:1], z[:-2]])
    zkp1 = jnp.concatenate([z[2:], z[-1:]])
    rdz = 2.0 / jnp.maximum(zkp1 - zkm1, 1.0e-12)
    du = u - jnp.concatenate([u[:1], u[:-1]])
    dv = v - jnp.concatenate([v[:1], v[:-1]])
    s2l_base = (du * du + dv * dv) * rdz * rdz
    pbl_mask = jnp.logical_and(pblflg, (kidx + 1) <= lpbl)
    suk = du * rdz
    svk = dv * rdz
    s2l_pbl = (suk - ufxpbl) * suk + (svk - vfxpbl) * svk
    s2 = jnp.where(kge1, jnp.maximum(jnp.where(pbl_mask, s2l_pbl, s2l_base), epsgm), 0.0)

    tm1 = jnp.concatenate([t[:1], t[:-1]])
    them1 = jnp.concatenate([the[:1], the[:-1]])
    qm1 = jnp.concatenate([q[:1], q[:-1]])
    cwmm1 = jnp.concatenate([cwm[:1], cwm[:-1]])
    tem = 0.5 * (t + tm1)
    thm = 0.5 * (the + them1)
    a = thm * EP1
    b = (elocp / tem - 1.0 - EP1) * thm
    ghl = (
        dth * ((q + qm1 + cwm + cwmm1) * (0.5 * EP1) + 1.0)
        + (q - qm1 + cwm - cwmm1) * a
        + (cwm - cwmm1) * b
    ) * rdz
    ghl = jnp.where(pbl_mask, ghl - mf - qfxpbl * a, ghl)
    ghl = jnp.where(jnp.abs(ghl) <= epsgh, epsgh, ghl)
    gh = jnp.where(kge1, ghl, 0.0)
    en2 = gh * g / jnp.where(thm != 0.0, thm, 1.0)
    ri = jnp.where(kge1, en2 / s2, 0.0)

    s2l = s2
    discr_st = jnp.maximum((bubm * s2l + bubh * gh) ** 2 * 0.25 - ((aubm * s2l + aubh * gh) * gh) * cubr, 0.0)
    qol2st = (-0.5 * (bubm * s2l + bubh * gh) + jnp.sqrt(discr_st)) * rcubr
    elm_st = jnp.maximum(jnp.sqrt(_safe_div(q2, qol2st, 0.0)), epsl)
    aden = (adnm * s2l + adnh * gh) * gh
    bden = bdnm * s2l + bdnh * gh
    qol2un = -0.5 * bden + jnp.sqrt(jnp.maximum(bden * bden * 0.25 - aden, 0.0))
    elm_un = jnp.maximum(jnp.sqrt(_safe_div(q2, qol2un + epsru, 0.0)), epsl)
    elm_pos_limit = s2l / jnp.where(gh != 0.0, gh, 1.0) <= requ
    elm_pos = jnp.where(elm_pos_limit, epsl, elm_st)
    elm = jnp.where(kge1, jnp.where(gh >= epsgh, elm_pos, elm_un), epsl)

    q1 = jnp.where(kidx <= (lpbl - 1), jnp.sqrt(jnp.maximum(q2, 0.0)), 0.0)
    qdzl = (q1[1:] + q1[:-1]) * (z[1:nz] - z[: nz - 1])
    szq = jnp.sum((z[1:nz] + z[: nz - 1] - 2.0 * z[0]) * qdzl)
    sq = jnp.sum(qdzl)
    el0 = jnp.minimum(alph * szq * 0.5 / jnp.maximum(sq, 1.0e-30), el0max)
    el0 = jnp.maximum(el0, el0min)

    lpblm = jnp.minimum(lpbl + 1, nz)
    above = jnp.logical_and(kidx >= (lpblm - 1), kidx > 0)
    el_above = (zkp1 - zkm1) * elfc
    el = jnp.where(above, el_above, 0.0)
    rel = jnp.where(above, el / jnp.maximum(elm, 1.0e-30), 0.0)

    epshol = jnp.minimum(epshol, 0.0)
    ckp = elcbl * ((1.0 - 8.0 * epshol) ** (1.0 / 3.0))
    inside = jnp.logical_and(kidx >= 1, kidx <= (lpbl - 1))
    vkrmz = (z[kidx] - z[0]) * KARMAN
    vkrmz = jnp.where(pblflg, ckp * (z[kidx] - z[0]) * KARMAN, vkrmz)
    el_inside = vkrmz / (vkrmz / el0 + 1.0)
    rel_inside = el_inside / jnp.maximum(elm, 1.0e-30)
    el = jnp.where(jnp.logical_and(inside, lpbl > 1), el_inside, el)
    rel = jnp.where(jnp.logical_and(inside, lpbl > 1), rel_inside, rel)

    rel_l = jnp.concatenate([rel[:1], rel[:-1]])
    rel_r = jnp.concatenate([rel[1:], rel[-1:]])
    smooth = jnp.logical_and(kidx >= 2, kidx <= (lpbl - 2))
    srel = jnp.minimum(((rel_l + rel_r) * 0.5 + rel) * 0.5, rel)
    el = jnp.where(smooth, jnp.maximum(srel * elm, epsl), el)

    f = jnp.maximum(corf, eps1)
    rlambda = f / (blckdr * ustar)
    vkrmz_st = (z[kidx] - z[0]) * KARMAN
    rlb = rlambda + 1.0 / jnp.maximum(vkrmz_st, 1.0e-30)
    rln = jnp.sqrt(jnp.maximum(2.0 * en2 / jnp.maximum(q2, 1.0e-30), 0.0)) / cn
    el_stable = 1.0 / (rlb + rln)
    el = jnp.where(jnp.logical_and(kge1, en2 >= 0.0), el_stable, el)
    return s2, gh, ri, el


def _prodq2(dtturbl, ustar, s2, ri, q2, el, z, akm, akh, uxk, vxk, thxk, thvxk,
            hgamu, hgamv, hgamq, delxy, hpbl, pblflg, kpbl, mf, ufxpbl, vfxpbl, qfxpbl):
    nz = uxk.shape[0]
    kidx = jnp.arange(nz)
    kge1 = kidx >= 1
    epsq2l, c0, ceps, g = 0.01, 0.55, 16.6, 9.81
    rc02 = 2.0 / (c0 * c0)
    zkm1 = jnp.concatenate([z[:1], z[:-2]])
    zkp1 = jnp.concatenate([z[2:], z[-1:]])
    deltaz = 0.5 * (zkp1 - zkm1)
    um1 = jnp.concatenate([uxk[:1], uxk[:-1]])
    vm1 = jnp.concatenate([vxk[:1], vxk[:-1]])
    thvm1 = jnp.concatenate([thvxk[:1], thvxk[:-1]])
    thm1 = jnp.concatenate([thxk[:1], thxk[:-1]])
    suk = (uxk - um1) / jnp.maximum(deltaz, 1.0e-30)
    svk = (vxk - vm1) / jnp.maximum(deltaz, 1.0e-30)
    gthvk = (thvxk - thvm1) / jnp.maximum(deltaz, 1.0e-30)
    govrthvk = g / (0.5 * (thvxk + thvm1))
    thm = 0.5 * (thxk + thm1)
    in_pbl = jnp.logical_and(pblflg, (kidx + 1) <= kpbl)
    pru = jnp.where(in_pbl, (akm * (suk - hgamu / hpbl - ufxpbl)) * suk, akm * suk * suk)
    prv = jnp.where(in_pbl, (akm * (svk - hgamv / hpbl - vfxpbl)) * svk, akm * svk * svk)
    pr = pru + prv
    bpr_pbl = (akh * (gthvk - mf - (hgamq / hpbl + qfxpbl) * EP1 * thm)) * govrthvk
    bpr = jnp.where(in_pbl, bpr_pbl, akh * gthvk * govrthvk)
    disel = jnp.minimum(delxy, ceps * el)
    dis = q2 ** 1.5 / jnp.maximum(disel, 1.0e-30)
    q2_new = jnp.maximum(q2 + 2.0 * (pr - bpr - dis) * dtturbl, epsq2l)
    q2_new = jnp.where(kge1, q2_new, jnp.maximum(rc02 * ustar * ustar, epsq2l))
    return q2_new


def _vdifq(dtdif, q2, el, z, akhk, ptke1, hgame, hpbl, pblflg, kpbl, efxpbl):
    nz = q2.shape[0]
    kidx = jnp.arange(nz)
    c_k = 1.0
    zkm1 = jnp.concatenate([z[:1], z[:-2]])
    zkp1 = jnp.concatenate([z[2:], z[-1:]])
    z_here = z[:nz]
    z_prev = jnp.concatenate([z[:1], z[: nz - 1]])
    zak = 0.5 * (z_here + z_prev)
    zfacentk = jnp.where(kidx >= 1, (zak / hpbl) ** 3.0, 0.0)
    dtoz = jnp.where(kidx >= 2, (dtdif + dtdif) / jnp.maximum(zkp1 - zkm1, 1.0e-30), 0.0)
    akhm1 = jnp.concatenate([akhk[:1], akhk[:-1]])
    akq = c_k * (
        akhk / jnp.maximum(zkp1 - zkm1, 1.0e-30)
        + akhm1 / jnp.maximum(z[kidx] - zkm1, 1.0e-30)
    ) * ptke1
    akq = jnp.where(kidx >= 2, akq, 0.0)
    cr = -dtoz * akq
    akqs = c_k * akhk[1] / jnp.maximum(z[2] - z[0], 1.0e-30) * ptke1[1]

    cm = jnp.zeros(nz, dtype=q2.dtype).at[nz - 1].set(dtoz[nz - 1] * akq[nz - 1] + 1.0)
    rsq2 = jnp.zeros(nz, dtype=q2.dtype).at[nz - 1].set(q2[nz - 1])

    def back_body(j, carry):
        cm_c, rs_c = carry
        k = (nz - 2) - j
        cf = -dtoz[k] * akq[k + 1] / cm_c[k + 1]
        cm_k = -cr[k + 1] * cf + (akq[k + 1] + akq[k]) * dtoz[k] + 1.0
        rs_k = -rs_c[k + 1] * cf + q2[k]
        in_pbl = jnp.logical_and(pblflg, (k + 1) < kpbl)
        rs_p = (
            rs_k
            - dtoz[k] * (2.0 * hgame[k] / hpbl) * akq[k + 1] * (z[k + 1] - z[k])
            + dtoz[k] * (2.0 * hgame[k - 1] / hpbl) * akq[k] * (z[k] - z[k - 1])
            - dtoz[k] * 2.0 * efxpbl * zfacentk[k + 1]
            + dtoz[k] * 2.0 * efxpbl * zfacentk[k]
        )
        return cm_c.at[k].set(cm_k), rs_c.at[k].set(jnp.where(in_pbl, rs_p, rs_k))

    cm, rsq2 = jax.lax.fori_loop(0, jnp.maximum(nz - 3, 0), back_body, (cm, rsq2))

    dtozs = (dtdif + dtdif) / jnp.maximum(z[2] - z[0], 1.0e-30)
    cf = -dtozs * akq[2] / cm[2]
    q2_1_p = (
        dtozs * akqs * q2[0] - rsq2[2] * cf + q2[1]
        - dtozs * (2.0 * hgame[1] / hpbl) * akq[2] * (z[2] - z[1])
        + dtozs * (2.0 * hgame[0] / hpbl) * akqs * (z[1] - z[0])
        - dtozs * 2.0 * efxpbl * zfacentk[2]
        + dtozs * 2.0 * efxpbl * zfacentk[1]
    )
    q2_1_p = q2_1_p / ((akq[2] + akqs) * dtozs - cr[2] * cf + 1.0)
    q2_1_s = (
        dtozs * akqs * q2[0] - rsq2[2] * cf + q2[1]
    ) / ((akq[2] + akqs) * dtozs - cr[2] * cf + 1.0)
    q2_out = q2.at[1].set(jnp.where(jnp.logical_and(pblflg, 2 < kpbl), q2_1_p, q2_1_s))

    def fwd_body(j, arr):
        k = j + 2
        val = (-cr[k] * arr[k - 1] + rsq2[k]) / cm[k]
        return arr.at[k].set(val)

    q2_out = jax.lax.fori_loop(0, nz - 2, fwd_body, q2_out)
    return q2_out


def _shinhong_column(u, v, tx, qv, p, pdi, pi, dz, tke_in, qc, qi, *,
                     psfc, znt, ust, hfx, qfx, wspd, br, psim, psih,
                     dt, xland, u10, v10, dx, dy, corf, ctopo, shinhong_tke_diag):
    nz = u.shape[0]
    kidx = jnp.arange(nz)
    kp1 = kidx + 1
    th = tx / pi
    thv = th * (1.0 + EP1 * qv)
    rhox = psfc / (R_D * tx[0] * (1.0 + EP1 * qv[0]))
    govrth = G / th[0]
    zq = jnp.concatenate([jnp.zeros(1, dtype=dz.dtype), jnp.cumsum(dz)])
    za = 0.5 * (zq[:-1] + zq[1:])
    delp = pdi[:-1] - pdi[1:]
    dza = jnp.concatenate([za[:1], za[1:] - za[:-1]])
    xkzom = jnp.where(kidx == nz - 1, 0.0, XKZMINM)
    xkzoh = jnp.where(kidx == nz - 1, 0.0, XKZMINH)

    cont = CP / G
    conpr = BFAC * KARMAN * SFCFRAC
    dt2 = 2.0 * dt
    rdt = 1.0 / dt2
    wspd1 = jnp.sqrt(u[0] * u[0] + v[0] * v[0]) + 1.0e-9
    sflux = hfx / rhox / CP + qfx / rhox * EP1 * th[0]
    sfcflg = jnp.logical_not(br > 0.0)
    zl1 = za[0]
    q2x0 = 2.0 * tke_in

    kpbl, brdn, brup = _first_pbl_guess(thv, thv[0], za, br, BRCR_UB, u, v)
    hpbl, kpbl, pblflg = _interp_pblh(kpbl, brdn, brup, BRCR_UB, za, zq)
    zol1 = jnp.maximum(br * psim * psim / psih, RIMIN)
    zol1 = jnp.where(sfcflg, jnp.minimum(zol1, -ZFMIN), jnp.maximum(zol1, ZFMIN))
    hol1 = zol1 * hpbl / zl1 * SFCFRAC
    phim = jnp.where(sfcflg, (1.0 - APHI16 * hol1) ** (-0.25), 1.0 + APHI5 * hol1)
    phih = jnp.where(sfcflg, (1.0 - APHI16 * hol1) ** (-0.5), phim)
    bfx0 = jnp.maximum(sflux, 0.0)
    wstar3 = jnp.where(sfcflg, govrth * bfx0 * hpbl, 0.0)
    wstar = jnp.where(sfcflg, wstar3 ** H1, 0.0)
    ust3 = ust ** 3
    wscale = (ust3 + PHIFAC * KARMAN * wstar3 * 0.5) ** H1
    wscale = jnp.clip(wscale, ust / APHI5, ust * APHI16)

    conv = jnp.logical_and(sfcflg, sflux > 0.0)
    gamfac = BFAC / rhox / wscale
    hgamt_c = jnp.minimum(gamfac * hfx / CP, GAMCRT)
    hgamq_c = jnp.minimum(gamfac * qfx, GAMCRQ)
    vpert = (hgamt_c + EP1 * th[0] * hgamq_c) / BFAC * AFAC
    thermal = thv[0] + jnp.where(conv, jnp.maximum(vpert, 0.0) * jnp.minimum(za[0] / (SFCFRAC * hpbl), 1.0), 0.0)
    hgamu = jnp.where(conv, (-15.9 * ust * ust / wspd * wstar3 / (wscale ** 4)) * u[0], 0.0)
    hgamv = jnp.where(conv, (-15.9 * ust * ust / wspd * wstar3 / (wscale ** 4)) * v[0], 0.0)
    hgamq = jnp.where(conv, jnp.maximum(hgamq_c, 0.0), 0.0)
    pblflg = jnp.where(conv, pblflg, False)
    kpbl_r, brdn_r, brup_r = _first_pbl_guess(thv, thermal, za, br, BRCR_UB, u, v)
    hpbl_r, kpbl_r, pblflg_r = _interp_pblh(kpbl_r, brdn_r, brup_r, BRCR_UB, za, zq)
    hpbl = jnp.where(pblflg, hpbl_r, hpbl)
    kpbl = jnp.where(pblflg, kpbl_r, kpbl)
    pblflg = jnp.where(pblflg, pblflg_r, pblflg)

    delxy = jnp.sqrt(dx * dy)
    uwst = jnp.abs(_safe_div(ust, wstar, 0.0) - 0.5)
    csfac = jnp.where(wstar != 0.0, 0.5 * (jnp.tanh(-80.0 * uwst + 14.0) + 3.0), 1.0)
    cslen = jnp.where(pblflg, csfac * hpbl, 0.0)
    pu1 = _pu(delxy, cslen)
    pq1 = _pq(delxy, cslen)
    hgamu = jnp.where(pblflg, hgamu * pu1, hgamu)
    hgamv = jnp.where(pblflg, hgamv * pu1, hgamv)
    hgamq = jnp.where(pblflg, hgamq * pq1, hgamq)

    cond_stable_low = jnp.logical_and(jnp.logical_not(sfcflg), hpbl < zq[1])
    stable_init = jnp.logical_not(cond_stable_low)
    wspd10 = jnp.sqrt(u10 * u10 + v10 * v10)
    ross = wspd10 / (CORI * znt)
    brcr_water = jnp.minimum(0.16 * (1.0e-7 * ross) ** (-0.18), 0.3)
    brcr = jnp.where(stable_init, BRCR_UB, jnp.where((xland - 1.5) >= 0.0, brcr_water, BRCR_SB))

    def rb_step(carry, k):
        brdn_c, brup_c, stable_c, kpbl_c = carry
        active = jnp.logical_not(stable_c)
        spdk2 = jnp.maximum(u[k] * u[k] + v[k] * v[k], 1.0)
        brup_new = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2
        return (
            jnp.where(active, brup_c, brdn_c),
            jnp.where(active, brup_new, brup_c),
            jnp.where(active, brup_new > brcr, stable_c),
            jnp.where(active, (k + 1).astype(jnp.int32), kpbl_c),
        ), None

    (brdn2, brup2, _st2, kpbl2), _ = jax.lax.scan(
        rb_step, (br, br, stable_init, kpbl.astype(jnp.int32)), jnp.arange(1, nz, dtype=jnp.int32)
    )
    hpbl_b, kpbl_b, pblflg_b = _interp_pblh(kpbl2, brdn2, brup2, brcr, za, zq)
    hpbl = jnp.where(cond_stable_low, hpbl_b, hpbl)
    kpbl = jnp.where(cond_stable_low, kpbl_b, kpbl)
    # WRF updates hpbl/kpbl in this stable-boundary-layer branch, but it does not
    # re-enable pblflg after the non-convective surface branch cleared it.
    pblflg = jnp.where(jnp.logical_and(cond_stable_low, jnp.logical_not(pblflg_b)), False, pblflg)

    k_ent = jnp.clip(kpbl - 2, 0, nz - 2)
    wm3 = wstar3 + 5.0 * ust3
    wm2 = wm3 ** H2
    bfxpbl = -0.15 * thv[0] / G * wm3 / hpbl
    dthvx = jnp.maximum(thv[k_ent + 1] - thv[k_ent], TMIN)
    dthx = jnp.maximum(th[k_ent + 1] - th[k_ent], TMIN)
    dqx = jnp.minimum(qv[k_ent + 1] - qv[k_ent], 0.0)
    we = jnp.maximum(bfxpbl / dthvx, -jnp.sqrt(wm2))
    hfxpbl = we * dthx
    qfxpbl = we * dqx * pq1
    dux = u[k_ent + 1] - u[k_ent]
    dvx = v[k_ent + 1] - v[k_ent]
    ufxpbl = jnp.where(dux > TMIN, jnp.maximum(we * dux * pu1, -ust * ust),
                       jnp.where(dux < -TMIN, jnp.minimum(we * dux * pu1, ust * ust), 0.0))
    vfxpbl = jnp.where(dvx > TMIN, jnp.maximum(we * dvx * pu1, -ust * ust),
                       jnp.where(dvx < -TMIN, jnp.minimum(we * dvx * pu1, ust * ust), 0.0))
    delta = jnp.minimum(D1 * hpbl + D2 * wm2 / (govrth * D3 * hpbl), 100.0)
    deltaoh = D1 * hpbl + D2 * wm2 / (govrth * dthvx)
    deltaoh = jnp.maximum(EZFAC * deltaoh, hpbl - za[jnp.maximum(kpbl - 2, 0)] - 1.0)
    deltaoh = jnp.minimum(deltaoh, hpbl)
    rigs = jnp.where((dux != 0.0) | (dvx != 0.0), govrth * dthvx * deltaoh / (dux * dux + dvx * dvx), RIGSMAX)
    rigs = jnp.clip(rigs, RIMIN, RIGSMAX)
    cenlfrac = jnp.where((rigs > 0.0) & (jnp.abs(rigs + CPENT) <= 1.0e-6), ENTFMAX, rigs / (rigs + CPENT))
    cenlfrac = jnp.minimum(cenlfrac, ENTFMAX)
    enlfrac2 = jnp.maximum(wm3 / jnp.maximum(wstar3, 1.0e-30) * cenlfrac, ENTFMIN) * ENLFRAC
    we = jnp.where(pblflg, we, 0.0)
    qfxpbl = jnp.where(pblflg, qfxpbl, 0.0)
    ufxpbl = jnp.where(pblflg, ufxpbl, 0.0)
    vfxpbl = jnp.where(pblflg, vfxpbl, 0.0)
    wm2 = jnp.where(pblflg, wm2, 0.0)
    delta = jnp.where(pblflg, delta, 0.0)
    deltaoh = jnp.where(pblflg, deltaoh, 1.0)
    enlfrac2 = jnp.where(pblflg, enlfrac2, 0.0)

    entfacmf = jnp.where(pblflg, jnp.sqrt(((zq[1:] - hpbl) / deltaoh) ** 2), 0.0)
    entfac = jnp.where(jnp.logical_and(pblflg, kp1 >= kpbl), ((zq[1:] - hpbl) / deltaoh) ** 2, 1.0e30)
    in_pbl = kp1 < kpbl
    zfac = jnp.clip(1.0 - (zq[1:] - zl1) / jnp.maximum(hpbl - zl1, 1.0e-30), ZFMIN, 1.0)
    zfacent = (1.0 - zfac) ** 3
    wscalek = (ust3 + PHIFAC * KARMAN * wstar3 * (1.0 - zfac)) ** H1
    prfac = jnp.where(sfcflg, conpr, 0.0)
    prfac2 = jnp.where(sfcflg, 15.9 * wstar3 / ust3 / (1.0 + 4.0 * KARMAN * wstar3 / ust3), 0.0)
    prnumfac = jnp.where(sfcflg, -3.0 * jnp.maximum(zq[1:] - SFCFRAC * hpbl, 0.0) ** 2 / hpbl ** 2, 0.0)
    wscalek = jnp.where(sfcflg, wscalek, jnp.maximum(ust / (1.0 + APHI5 * zol1 * zq[1:] / zl1), 0.001))
    prnum0 = jnp.clip(phih / phim + prfac, PRMIN, PRMAX)
    xkzm_pbl = wscalek * KARMAN * zq[1:] * zfac ** PFAC
    prnum_q = 1.0 + (prnum0 - 1.0) * jnp.exp(prnumfac)
    xkzq_pbl = xkzm_pbl / prnum_q * zfac ** (PFAC_Q - PFAC)
    prnum_h = 1.0 + (prnum0 / (1.0 + prfac2 * KARMAN * SFCFRAC) - 1.0) * jnp.exp(prnumfac)
    xkzh_pbl = xkzm_pbl / prnum_h
    xkzm_pbl = jnp.minimum(xkzm_pbl + xkzom, XKZMAX)
    xkzh_pbl = jnp.minimum(xkzh_pbl + xkzoh, XKZMAX)
    xkzq_pbl = jnp.minimum(xkzq_pbl + xkzoh, XKZMAX)

    above = jnp.logical_and(kp1 >= kpbl, kidx < nz - 1)
    dza_kp1 = jnp.concatenate([dza[1:], dza[-1:]])
    u_next = jnp.concatenate([u[1:], u[-1:]])
    v_next = jnp.concatenate([v[1:], v[-1:]])
    thv_next = jnp.concatenate([thv[1:], thv[-1:]])
    th_next = jnp.concatenate([th[1:], th[-1:]])
    qc_next = jnp.concatenate([qc[1:], qc[-1:]])
    qi_next = jnp.concatenate([qi[1:], qi[-1:]])
    qv_next = jnp.concatenate([qv[1:], qv[-1:]])
    tx_next = jnp.concatenate([tx[1:], tx[-1:]])
    ss = ((u_next - u) ** 2 + (v_next - v) ** 2) / (dza_kp1 ** 2) + 1.0e-9
    ri = G / (0.5 * (thv_next + thv)) * (thv_next - thv) / (ss * dza_kp1)
    qcloud = (qc + qi > 0.01e-3) & (qc_next + qi_next > 0.01e-3)
    qmean = 0.5 * (qv + qv_next)
    tmean = 0.5 * (tx + tx_next)
    alpha = XLV * qmean / R_D / tmean
    chi = XLV * XLV * qmean / CP / R_V / tmean / tmean
    ri_cloud = (1.0 + alpha) * (ri - G * G / ss / tmean / CP * ((chi - alpha) / (1.0 + chi)))
    ri = jnp.where(qcloud, ri_cloud, ri)
    zk = KARMAN * zq[1:]
    rlamdz = jnp.minimum(jnp.minimum(jnp.maximum(0.1 * dza_kp1, RLAM), 300.0), dza_kp1)
    dk = (zk * rlamdz / (rlamdz + zk)) ** 2 * jnp.sqrt(ss)
    ri_neg = jnp.maximum(ri, RIMIN)
    sri = jnp.sqrt(jnp.maximum(-ri, 0.0))
    xkzm_un = dk * (1.0 + 8.0 * (-ri_neg) / (1.0 + 1.746 * sri))
    xkzh_un = dk * (1.0 + 8.0 * (-ri_neg) / (1.0 + 1.286 * sri))
    xkzh_st = dk / (1.0 + 5.0 * ri) ** 2
    xkzm_st = xkzh_st * jnp.minimum(1.0 + 2.1 * ri, PRMAX)
    xkzm_loc = jnp.minimum(jnp.where(ri < 0.0, xkzm_un, xkzm_st) + xkzom, XKZMAX)
    xkzh_loc = jnp.minimum(jnp.where(ri < 0.0, xkzh_un, xkzh_st) + xkzoh, XKZMAX)
    xkzm = jnp.where(in_pbl, xkzm_pbl, jnp.where(above, xkzm_loc, 0.0))
    xkzh = jnp.where(in_pbl, xkzh_pbl, jnp.where(above, xkzh_loc, 0.0))
    xkzq = jnp.where(in_pbl, xkzq_pbl, jnp.where(above, xkzh_loc, 0.0))
    xkzml = jnp.where(above, xkzm_loc, 0.0)
    xkzhl = jnp.where(above, xkzh_loc, 0.0)

    deltaoh_n = deltaoh / jnp.maximum(hpbl, 1.0e-30)
    mlfrac = MLTOP - deltaoh_n
    zfacmf0 = jnp.clip(zq[1] / jnp.maximum(hpbl, 1.0e-30), ZFMIN, 1.0)
    sfcfracn = jnp.maximum(SFCFRACN1, zfacmf0)
    snlflux0 = NLFRAC * (A11 + A12 * sfcfracn) * sflux
    amf1 = snlflux0 / sfcfracn
    amf2 = jnp.where(pblflg, -snlflux0 / jnp.maximum(mlfrac - sfcfracn, 1.0e-30), 0.0)
    bmf2 = -mlfrac * amf2
    amf3 = jnp.where((deltaoh_n == 0.0) & (enlfrac2 == 0.0), 0.0, snlflux0 * enlfrac2 / jnp.maximum(deltaoh_n, 1.0e-30))
    hfxpbl_prof = (amf3 - amf3 * mlfrac) * _pthnl(delxy, cslen)
    zfacmf = jnp.maximum(zq[1:] / jnp.maximum(hpbl, 1.0e-30), ZFMIN)
    mf_base = jnp.where(zfacmf <= sfcfracn, amf1 * zfacmf,
                        jnp.where(zfacmf <= mlfrac, amf2 * zfacmf + bmf2, 0.0))
    mf = jnp.where(jnp.logical_and(pblflg, in_pbl), (mf_base + hfxpbl_prof * jnp.exp(-entfacmf)) * _pthnl(delxy, cslen), 0.0)

    kk = jnp.arange(nz - 1)
    kp1_face = kk + 1
    dza_f = dza[1:]
    p_k = p[:-1]
    p_k1 = p[1:]
    delp_k = delp[:-1]
    delp_k1 = delp[1:]
    dtodsd = dt2 / delp_k
    dtodsu = dt2 / delp_k1
    dsig = p_k - p_k1
    rdz = 1.0 / dza_f
    ent_face = jnp.logical_and(jnp.logical_and(pblflg, kp1_face >= kpbl), entfac[:-1] < 4.6)
    dza_kpblm1 = dza[jnp.maximum(kpbl - 1, 0)]
    xkzh_ent = jnp.sqrt(jnp.maximum((-we * dza_kpblm1 * jnp.exp(-entfac[:-1])) * xkzhl[:-1], 0.0))
    xkzh_ent = jnp.minimum(jnp.maximum(xkzh_ent, xkzoh[:-1]), XKZMAX)
    xkzh_used = jnp.where(ent_face, xkzh_ent, xkzh[:-1])
    in_pbl_face = kp1_face < kpbl
    tem1_pre = dsig * xkzh[:-1] * rdz
    dsdzt = tem1_pre * (-mf[:-1] / jnp.maximum(xkzh[:-1], 1.0e-30))
    tem1 = dsig * xkzh_used * rdz
    dsdz2 = tem1 * rdz
    upper_h = -dtodsd * dsdz2
    lower_h = -dtodsu * dsdz2
    zfacdx = 0.2 * hpbl / zq[1:nz]
    pth1 = _pthl(delxy * jnp.maximum(zfacdx, 1.0), hpbl)
    upper_h = jnp.where(in_pbl_face, upper_h * pth1, upper_h)
    lower_h = jnp.where(in_pbl_face, lower_h * pth1, lower_h)
    diag_h = jnp.ones(nz).at[kk].add(-upper_h).at[kk + 1].add(-lower_h)
    rhs_h = (th - 300.0).at[0].add(hfx / cont / delp[0] * dt2)
    rhs_h = rhs_h.at[kk].add(jnp.where(in_pbl_face, dtodsd * dsdzt, 0.0))
    rhs_h = rhs_h.at[kk + 1].add(jnp.where(in_pbl_face, -dtodsu * dsdzt, 0.0))
    theta_sol = _thomas_scan(jnp.zeros(nz).at[kk + 1].set(lower_h), diag_h, jnp.zeros(nz).at[kk].set(upper_h), rhs_h)
    theta_tend = (theta_sol - th + 300.0) * rdt
    exch_h = jnp.concatenate([jnp.zeros(1, dtype=u.dtype), xkzh_used])
    xkzh_after = xkzh.at[:-1].set(xkzh_used)

    xkzq_face0 = jnp.where(kp1_face >= kpbl, xkzh_after[:-1], xkzq[:-1])
    xkzq_ent = jnp.sqrt(jnp.maximum((-we * dza_kpblm1 * jnp.exp(-entfac[:-1])) * xkzhl[:-1], 0.0))
    xkzq_ent = jnp.minimum(jnp.maximum(xkzq_ent, xkzoh[:-1]), XKZMAX)
    xkzq_used = jnp.where(ent_face, xkzq_ent, xkzq_face0)
    tem1_q_pre = dsig * xkzq_face0 * rdz
    dsdzq = tem1_q_pre * (-qfxpbl * zfacent[:-1] / jnp.maximum(xkzq_face0, 1.0e-30))
    tem1_q = dsig * xkzq_used * rdz
    upper_q = -dtodsd * tem1_q * rdz
    lower_q = -dtodsu * tem1_q * rdz
    pq1l = _pq(delxy * jnp.maximum(zfacdx, 1.0), hpbl)
    upper_q = jnp.where(in_pbl_face, upper_q * pq1l, upper_q)
    lower_q = jnp.where(in_pbl_face, lower_q * pq1l, lower_q)
    diag_q = jnp.ones(nz).at[kk].add(-upper_q).at[kk + 1].add(-lower_q)
    rhs_q = qv.at[0].add(qfx * G / delp[0] * dt2)
    rhs_q = rhs_q.at[kk].add(jnp.where(in_pbl_face, dtodsd * dsdzq, 0.0))
    rhs_q = rhs_q.at[kk + 1].add(jnp.where(in_pbl_face, -dtodsu * dsdzq, 0.0))
    qv_sol = _thomas_scan(jnp.zeros(nz).at[kk + 1].set(lower_q), diag_q, jnp.zeros(nz).at[kk].set(upper_q), rhs_q)
    qv_tend = (qv_sol - qv) * rdt
    xkzq_after = xkzq.at[:-1].set(xkzq_used)

    tflux_e = jnp.flip(jnp.cumsum(jnp.flip(theta_tend * dz)))
    qflux_e = jnp.flip(jnp.cumsum(jnp.flip(qv_tend * dz)))
    tvflux_e = tflux_e + qflux_e * EP1 * th
    q2_next = jnp.concatenate([q2x0[1:], q2x0[-1:]])
    tvflux_next = jnp.concatenate([tvflux_e[1:], tvflux_e[-1:]])
    hgame_c = C_1 * 0.2 * 2.5 * (G / thv) * wstar / jnp.maximum(0.25 * (q2_next + q2x0), 1.0e-30)
    hgame_c = jnp.minimum(hgame_c, GAMCRE)
    hgame2d = jnp.where(jnp.logical_and(pblflg, in_pbl),
                        jnp.maximum(hgame_c * 0.5 * (tvflux_e + tvflux_next) * hpbl, 0.0), 0.0)

    xkzm_ent = jnp.sqrt(jnp.maximum((xkzh_after[:-1]) * xkzml[:-1], 0.0))
    xkzm_ent = jnp.minimum(jnp.maximum(xkzm_ent, xkzom[:-1]), XKZMAX)
    xkzm_used = jnp.where(ent_face, xkzm_ent, xkzm[:-1])
    tem1_m_pre = dsig * xkzm[:-1] * rdz
    dsdzu = tem1_m_pre * (-hgamu / hpbl - ufxpbl * zfacent[:-1] / jnp.maximum(xkzm[:-1], 1.0e-30))
    dsdzv = tem1_m_pre * (-hgamv / hpbl - vfxpbl * zfacent[:-1] / jnp.maximum(xkzm[:-1], 1.0e-30))
    tem1_m = dsig * xkzm_used * rdz
    upper_m = -dtodsd * tem1_m * rdz
    lower_m = -dtodsu * tem1_m * rdz
    pu1l = _pu(delxy * jnp.maximum(zfacdx, 1.0), hpbl)
    upper_m = jnp.where(in_pbl_face, upper_m * pu1l, upper_m)
    lower_m = jnp.where(in_pbl_face, lower_m * pu1l, lower_m)
    fric = ctopo * ust * ust / wspd1 * rhox * G / delp[0] * dt2 * (wspd1 / wspd) ** 2
    diag_m = jnp.ones(nz).at[0].add(fric).at[kk].add(-upper_m).at[kk + 1].add(-lower_m)
    rhs_u = u.at[kk].add(jnp.where(in_pbl_face, dtodsd * dsdzu, 0.0))
    rhs_v = v.at[kk].add(jnp.where(in_pbl_face, dtodsd * dsdzv, 0.0))
    rhs_u = rhs_u.at[kk + 1].add(jnp.where(in_pbl_face, -dtodsu * dsdzu, 0.0))
    rhs_v = rhs_v.at[kk + 1].add(jnp.where(in_pbl_face, -dtodsu * dsdzv, 0.0))
    u_sol = _thomas_scan(jnp.zeros(nz).at[kk + 1].set(lower_m), diag_m, jnp.zeros(nz).at[kk].set(upper_m), rhs_u)
    v_sol = _thomas_scan(jnp.zeros(nz).at[kk + 1].set(lower_m), diag_m, jnp.zeros(nz).at[kk].set(upper_m), rhs_v)
    u_tend = (u_sol - u) * rdt
    v_tend = (v_sol - v) * rdt
    xkzm_after = xkzm.at[:-1].set(xkzm_used)

    akmk = jnp.zeros(nz).at[1:].set(xkzm_after[:-1])
    akhk = jnp.zeros(nz).at[1:].set(xkzh_after[:-1])
    mfk = jnp.zeros(nz).at[1:].set(mf[:-1] / jnp.maximum(xkzh_after[:-1], 1.0e-30))
    ufxpblk = jnp.zeros(nz).at[1:].set(ufxpbl * zfacent[:-1] / jnp.maximum(xkzm_after[:-1], 1.0e-30))
    vfxpblk = jnp.zeros(nz).at[1:].set(vfxpbl * zfacent[:-1] / jnp.maximum(xkzm_after[:-1], 1.0e-30))
    qfxpblk = jnp.zeros(nz).at[1:].set(qfxpbl * zfacent[:-1] / jnp.maximum(xkzq_after[:-1], 1.0e-30))
    za_safe = jnp.maximum(za[:-1], 1.0e-30)
    ptke_vals = _ptke(delxy * jnp.maximum(0.2 * hpbl / za_safe, 1.0), hpbl)
    ptke_mask = jnp.logical_and(pblflg, (jnp.arange(nz - 1) + 1) <= kpbl)
    ptke1 = jnp.ones(nz).at[1:].set(jnp.where(ptke_mask, ptke_vals, 1.0))
    dex = 0.25 * (q2x0[jnp.minimum(k_ent + 2, nz - 1)] - q2x0[k_ent])
    efxpbl = jnp.where(pblflg, we * dex, 0.0)
    hpbl_safe = jnp.maximum(hpbl, 1.0e-30)
    ufxpbl_fold = jnp.zeros(nz).at[1:].set(hgamu / hpbl_safe + ufxpblk[1:])
    vfxpbl_fold = jnp.zeros(nz).at[1:].set(hgamv / hpbl_safe + vfxpblk[1:])
    qfxpbl_fold = jnp.zeros(nz).at[1:].set(hgamq / hpbl_safe + qfxpblk[1:])
    s2k, _ghk, rik, elk = _mixlen(
        u, v, tx, th, qv, jnp.zeros_like(qv), q2x0, zq, ust, corf, hol1,
        hpbl, kpbl, pblflg, mfk, ufxpbl_fold, vfxpbl_fold, qfxpbl_fold,
    )
    q2_work = _prodq2(
        dt, ust, s2k, rik, q2x0, elk, zq, akmk, akhk, u, v, th, thv,
        hgamu, hgamv, hgamq, delxy, hpbl, pblflg, kpbl, mfk, ufxpblk, vfxpblk, qfxpblk,
    )
    q2_work = _vdifq(dt, q2_work, elk, zq, akhk, ptke1, hgame2d, hpbl, pblflg, kpbl, efxpbl)
    tke_out = 0.5 * jnp.maximum(q2_work, EPSQ2L)
    el_pbl = jnp.where(kidx != 0, elk, 0.0)
    diag_on = jnp.asarray(shinhong_tke_diag) == 1
    tke_out = jnp.where(diag_on, tke_out, tke_in)
    el_pbl = jnp.where(diag_on, el_pbl, 0.0)

    return u_tend, v_tend, theta_tend, qv_tend, exch_h, hpbl, kpbl.astype(jnp.int32), wstar, delta, tke_out, el_pbl


def shinhong_columns(
    u, v, temperature, qv, pressure, pressure_interface, exner, dz, tke,
    *, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt, xland, u10, v10,
    dx, dy, corf=1.0e-4, ctopo=1.0, shinhong_tke_diag=1, qc=None, qi=None,
):
    """Batched Shin-Hong columns over ``(ncol, nz)`` arrays."""

    ncol = u.shape[0]
    if qc is None:
        qc = jnp.zeros_like(qv)
    if qi is None:
        qi = jnp.zeros_like(qv)
    dt_b = jnp.broadcast_to(jnp.asarray(dt, jnp.float64), (ncol,))
    dx_b = jnp.broadcast_to(jnp.asarray(dx, jnp.float64), (ncol,))
    dy_b = jnp.broadcast_to(jnp.asarray(dy, jnp.float64), (ncol,))
    corf_b = jnp.broadcast_to(jnp.asarray(corf, jnp.float64), (ncol,))
    ctopo_b = jnp.broadcast_to(jnp.asarray(ctopo, jnp.float64), (ncol,))
    diag_b = jnp.broadcast_to(jnp.asarray(shinhong_tke_diag, jnp.int32), (ncol,))
    out = jax.vmap(
        lambda *a: _shinhong_column(
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10],
            psfc=a[11], znt=a[12], ust=a[13], hfx=a[14], qfx=a[15], wspd=a[16],
            br=a[17], psim=a[18], psih=a[19], dt=a[20], xland=a[21], u10=a[22],
            v10=a[23], dx=a[24], dy=a[25], corf=a[26], ctopo=a[27],
            shinhong_tke_diag=a[28],
        )
    )(
        u, v, temperature, qv, pressure, pressure_interface, exner, dz, tke, qc, qi,
        psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, dt_b, xland, u10, v10,
        dx_b, dy_b, corf_b, ctopo_b, diag_b,
    )
    u_t, v_t, th_t, qv_t, exch_h, pblh, kpbl, wstar, delta, tke_out, el_pbl = out
    return {
        "u": u_t,
        "v": v_t,
        "theta": th_t,
        "qv": qv_t,
        "exch_h": exch_h,
        "pblh": pblh,
        "kpbl": kpbl,
        "wstar": wstar,
        "delta": delta,
        "tke": tke_out,
        "el_pbl": el_pbl,
    }


__all__ = ["shinhong_columns"]
