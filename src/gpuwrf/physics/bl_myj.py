"""JIT/vmap-traceable MYJ PBL column kernel (``bl_pbl_physics=2``).

This is the OPERATIONAL twin of the host-NumPy reference column kernel in
``physics.pbl_myj`` (which is the human-readable WRF transcription used to build
the v0.6.0 savepoint-parity proof). The reference kernel uses Python ``for``
loops over host NumPy arrays and is therefore NOT ``jax.jit``/``jax.vmap``
traceable -- it cannot ride the operational device scan. This module re-expresses
the SAME WRF ``module_bl_myjpbl.F`` single-column algorithm (``MIXLEN`` ->
``PRODQ2`` -> ``DIFCOF`` -> ``VDIFQ`` -> ``VDIFH`` -> ``VDIFV``) in pure ``jnp``
with functional ``.at[].set()`` updates, so the column depth ``n`` is a static
Python int at trace time (the loops fully unroll/trace) while the batch axis is
left free for ``jax.vmap`` over ``(ny*nx)`` grid columns.

Faithfulness: every line mirrors ``physics.pbl_myj`` (and through it the
unmodified pristine WRF source). The v0.13 operational proof
(``proofs/v013/myj_janjic_oracle.py``) gates ``myj_columns`` against the SAME
fp64 WRF-source savepoints the reference kernel passes, at ~1e-13, so the
traceable port is shown WRF-faithful (NOT a JAX-vs-JAX self-compare).

Convention: profile inputs are bottom-up WRF mass-level columns of length ``n``;
work arrays use the Fortran 1-based top-down indexing (top at index 1) exactly
like the reference kernel, here realized as length-``n+2`` ``jnp`` arrays.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.physics import myj_constants as C


configure_jax_x64()


def _one_based_top_down(bottom_up: jax.Array, n: int) -> jax.Array:
    """Return a length-``n+2`` 1-based top-down work array (top at index 1)."""

    out = jnp.zeros(n + 2, dtype=jnp.float64)
    return out.at[1 : n + 1].set(bottom_up[::-1])


def _one_based_interfaces(dz_bottom_up: jax.Array, ht, n: int) -> jax.Array:
    """Fortran ``ZINT``/``ZHK`` interface heights, top-down, 1-based (len n+2)."""

    bottom = jnp.concatenate(
        [jnp.reshape(ht, (1,)).astype(jnp.float64),
         ht + jnp.cumsum(dz_bottom_up.astype(jnp.float64))]
    )
    z_top_down = bottom[::-1]
    out = jnp.zeros(n + 2, dtype=jnp.float64)
    return out.at[1 : n + 2].set(z_top_down)


def _mixlen(lmh, u, v, t, the, q, cwm, q2, z, ct):
    """Traceable transcription of WRF ``MIXLEN`` (matches pbl_myj._mixlen)."""

    n = lmh
    gm = jnp.zeros(lmh + 2, dtype=jnp.float64)
    gh = jnp.zeros(lmh + 2, dtype=jnp.float64)
    el = jnp.zeros(lmh + 2, dtype=jnp.float64)
    q1 = jnp.zeros(lmh + 2, dtype=jnp.float64)
    dth = jnp.zeros(lmh + 2, dtype=jnp.float64)
    elm = jnp.zeros(lmh + 2, dtype=jnp.float64)
    rel = jnp.zeros(lmh + 2, dtype=jnp.float64)

    # --- LPBL: first level (scanning top-down upward from LMH-1) where 2*TKE
    # floors. Reference: for k in range(lmh-1, 0, -1): first q2[k] <= EPSQ2*FH.
    floor = q2 <= (C.EPSQ2 * C.FH)
    idx = jnp.arange(lmh + 2)
    eligible = floor & (idx >= 1) & (idx <= lmh - 1)
    any_floor = jnp.any(eligible)
    # highest index k (closest to surface, largest k) that is eligible -- the
    # reference loop scans DOWN from lmh-1 and BREAKS on the first hit, i.e. the
    # LARGEST eligible k.
    rev_first = jnp.argmax(eligible[::-1].astype(jnp.int32))  # from the top end
    lpbl_hit = (lmh + 1) - rev_first
    lpbl = jnp.where(any_floor, lpbl_hit, 1)
    pblh = z[lpbl + 1] - z[lmh + 1]

    # dth[k] = the[k]-the[k+1] for k=1..lmh-1
    dth = dth.at[1:lmh].set(the[1:lmh] - the[2 : lmh + 1])

    # countergradient correction: for k=lmh-2..1 find first dth[k]>0 & dth[k+1]<=0
    cg = (dth > 0.0) & (jnp.concatenate([dth[1:], dth[-1:]]) <= 0.0)
    cg = cg & (idx >= 1) & (idx <= lmh - 2)
    any_cg = jnp.any(cg)
    # reference scans DOWN from lmh-2 and breaks on first hit => largest such k
    rev_cg = jnp.argmax(cg[::-1].astype(jnp.int32))
    kcg = (lmh + 1) - rev_cg
    dth = jnp.where(any_cg, dth.at[kcg].add(ct), dth)
    ct_out = 0.0

    # --- GM / GH (k=1..lmh-1) ---
    k = jnp.arange(1, lmh)
    rdz = 2.0 / (z[1:lmh] - z[3 : lmh + 2])
    gml = ((u[1:lmh] - u[2 : lmh + 1]) ** 2 + (v[1:lmh] - v[2 : lmh + 1]) ** 2) * rdz * rdz
    gm = gm.at[1:lmh].set(jnp.maximum(gml, C.EPSGM))

    tem = (t[1:lmh] + t[2 : lmh + 1]) * 0.5
    thm = (the[1:lmh] + the[2 : lmh + 1]) * 0.5
    a = thm * C.P608
    b = (C.ELOCP / tem - 1.0 - C.P608) * thm
    ghl = (
        dth[1:lmh] * ((q[1:lmh] + q[2 : lmh + 1] + cwm[1:lmh] + cwm[2 : lmh + 1]) * (0.5 * C.P608) + 1.0)
        + (q[1:lmh] - q[2 : lmh + 1] + cwm[1:lmh] - cwm[2 : lmh + 1]) * a
        + (cwm[1:lmh] - cwm[2 : lmh + 1]) * b
    ) * rdz
    ghl = jnp.where(jnp.abs(ghl) <= C.EPSGH, C.EPSGH, ghl)
    gh = gh.at[1:lmh].set(ghl)

    # --- ELM (mixing length from level-2.5 closure), k=1..lmh-1 ---
    gml_k = gm[1:lmh]
    ghl_k = gh[1:lmh]
    is_stable = ghl_k >= C.EPSGH
    requ_floor = is_stable & ((gml_k / ghl_k) <= C.REQU)
    # stable, above-REQU branch
    aubr = (C.AUBM * gml_k + C.AUBH * ghl_k) * ghl_k
    bubr = C.BUBM * gml_k + C.BUBH * ghl_k
    qol2st = (-0.5 * bubr + jnp.sqrt(jnp.maximum(bubr * bubr * 0.25 - aubr * C.CUBR, 0.0))) * C.RCUBR
    eloq2x_st = 1.0 / qol2st
    elm_st = jnp.maximum(jnp.sqrt(eloq2x_st * q2[1:lmh]), C.EPSL)
    # unstable branch
    aden = (C.ADNM * gml_k + C.ADNH * ghl_k) * ghl_k
    bden = C.BDNM * gml_k + C.BDNH * ghl_k
    qol2un = -0.5 * bden + jnp.sqrt(jnp.maximum(bden * bden * 0.25 - aden, 0.0))
    eloq2x_un = 1.0 / (qol2un + C.EPSRU)
    elm_un = jnp.maximum(jnp.sqrt(eloq2x_un * q2[1:lmh]), C.EPSL)
    elm_vals = jnp.where(
        requ_floor, C.EPSL, jnp.where(is_stable, elm_st, elm_un)
    )
    elm = elm.at[1:lmh].set(elm_vals)

    # LMXL: the largest k (1..lmh-1) hitting the REQU floor; default lmh.
    lmxl_elig = requ_floor & (k >= 1) & (k <= lmh - 1)
    any_lmxl = jnp.any(lmxl_elig)
    rev_lmxl = jnp.argmax(lmxl_elig[::-1].astype(jnp.int32))
    lmxl_hit = (lmh - 1) - rev_lmxl  # k index space (1..lmh-1)
    lmxl = jnp.where(any_lmxl, lmxl_hit, lmh)
    # WRF special-case: if ELM(LMH-1)==EPSL, LMXL=LMH.
    lmxl = jnp.where(elm[lmh - 1] == C.EPSL, lmh, lmxl)
    mixht = z[lmxl] - z[lmh + 1]

    # q1 over lpbl..lmh (use a mask; lpbl is traced)
    q1 = jnp.where((idx >= lpbl) & (idx <= lmh), jnp.sqrt(jnp.maximum(q2, 0.0)), q1)

    # --- EL0 (asymptotic mixing length) ---
    qdzl = (q1[1:lmh] + q1[2 : lmh + 1]) * (z[2 : lmh + 1] - z[3 : lmh + 2])
    szq = jnp.sum((z[2 : lmh + 1] + z[3 : lmh + 2] - 2.0 * z[lmh + 1]) * qdzl)
    sq = jnp.sum(qdzl)
    el0 = jnp.minimum(C.ALPH * szq * 0.5 / sq, C.EL0MAX)
    el0 = jnp.maximum(el0, C.EL0MIN)

    # --- EL profile ---
    lpblm = jnp.maximum(lpbl - 1, 1)
    # block 1: k=1..lpblm -> el = min((z[k]-z[k+2])*ELFC, elm[k])
    el_b1 = jnp.minimum((z[1:lmh] - z[3 : lmh + 2]) * C.ELFC, elm[1:lmh])
    mask_b1 = (k >= 1) & (k <= lpblm)
    el = el.at[1:lmh].set(jnp.where(mask_b1, el_b1, el[1:lmh]))
    rel = rel.at[1:lmh].set(jnp.where(mask_b1, el[1:lmh] / elm[1:lmh], rel[1:lmh]))
    # block 2: k=lpbl..lmh-1 (only when lpbl<lmh)
    vkrmz = (z[2 : lmh + 1] - z[lmh + 1]) * C.VKARMAN
    el_b2 = jnp.minimum(vkrmz / (vkrmz / el0 + 1.0), elm[1:lmh])
    mask_b2 = (k >= lpbl) & (k <= lmh - 1) & (lpbl < lmh)
    el = el.at[1:lmh].set(jnp.where(mask_b2, el_b2, el[1:lmh]))
    rel = rel.at[1:lmh].set(jnp.where(mask_b2, el[1:lmh] / elm[1:lmh], rel[1:lmh]))

    # smoothing block: k=lpbl+1..lmh-2
    rel_km1 = jnp.concatenate([rel[:1], rel[:-1]])  # rel[k-1]
    rel_kp1 = jnp.concatenate([rel[1:], rel[-1:]])  # rel[k+1]
    srel = jnp.minimum(((rel_km1 + rel_kp1) * 0.5 + rel) * 0.5, rel)
    el_sm = jnp.maximum(srel * elm, C.EPSL)
    mask_sm = (idx >= lpbl + 1) & (idx <= lmh - 2)
    el = jnp.where(mask_sm, el_sm, el)

    return gm, gh, el, pblh, lpbl, lmxl, ct_out, mixht


def _prodq2(lmh, dtturbl, ustar, gm, gh, el, q2):
    """Traceable transcription of WRF ``PRODQ2``; returns updated (el, q2)."""

    k = jnp.arange(1, lmh)
    gml = gm[1:lmh]
    ghl = gh[1:lmh]
    aequ = (C.AEQM * gml + C.AEQH * ghl) * ghl
    bequ = C.BEQM * gml + C.BEQH * ghl
    eqol2 = -0.5 * bequ + jnp.sqrt(jnp.maximum(bequ * bequ * 0.25 - aequ, 0.0))

    bad = (
        ((gml + ghl * ghl) <= C.EPSTRB)
        | ((ghl >= C.EPSGH) & ((gml / ghl) <= C.REQU))
        | (eqol2 <= C.EPS2)
    )

    anum = (C.ANMM * gml + C.ANMH * ghl) * ghl
    bnum = C.BNMM * gml + C.BNMH * ghl
    aden = (C.ADNM * gml + C.ADNH * ghl) * ghl
    bden = C.BDNM * gml + C.BDNH * ghl
    cden = 1.0
    arhs = -(anum * bden - bnum * aden) * 2.0
    brhs = -anum * 4.0
    crhs = -bnum * 2.0
    eqol2_safe = jnp.where(bad, 1.0, eqol2)
    q2_k = jnp.maximum(q2[1:lmh], C.EPS1 * C.EPS1)
    dloq1 = el[1:lmh] / jnp.sqrt(q2_k)

    eloq21 = 1.0 / eqol2_safe
    eloq11 = jnp.sqrt(jnp.maximum(eloq21, 0.0))
    eloq31 = eloq21 * eloq11
    eloq41 = eloq21 * eloq21
    eloq51 = eloq21 * eloq31
    rden1 = 1.0 / (aden * eloq41 + bden * eloq21 + cden)
    rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11) * rden1 * rden1
    eloq12 = eloq11 + (dloq1 - eloq11) * jnp.exp(rhsp1 * dtturbl)
    eloq12 = jnp.maximum(eloq12, C.EPS1)

    eloq22 = eloq12 * eloq12
    eloq32 = eloq22 * eloq12
    eloq42 = eloq22 * eloq22
    eloq52 = eloq22 * eloq32
    rden2 = 1.0 / (aden * eloq42 + bden * eloq22 + cden)
    rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + C.RB1
    rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12) * rden2 * rden2
    rhst2 = rhs2 / rhsp2
    eloq13 = eloq12 - rhst2 + (rhst2 + dloq1 - eloq12) * jnp.exp(rhsp2 * dtturbl)
    eloq13 = jnp.maximum(eloq13, C.EPS1)

    eloqn = eloq13
    good = eloqn > C.EPS1
    q2_new = jnp.where(good, el[1:lmh] * el[1:lmh] / (eloqn * eloqn), C.EPSQ2)
    q2_new = jnp.maximum(q2_new, C.EPSQ2)
    el_floor = (q2_new == C.EPSQ2)  # collapse EL to EPSL when q2 hit floor

    # combine with the "bad" mask: bad => q2=EPSQ2, el=EPSL
    q2_out_k = jnp.where(bad, C.EPSQ2, q2_new)
    el_out_k = jnp.where(bad | el_floor | (~good), C.EPSL, el[1:lmh])

    q2 = q2.at[1:lmh].set(q2_out_k)
    el = el.at[1:lmh].set(el_out_k)
    # surface boundary value
    q2 = q2.at[lmh].set(jnp.maximum(C.B1 ** (2.0 / 3.0) * ustar * ustar, C.EPSQ2))
    return el, q2


def _difcof(lmh, gm, gh, el, q2, z):
    """Traceable transcription of WRF ``DIFCOF``."""

    akm = jnp.zeros(lmh + 2, dtype=jnp.float64)
    akh = jnp.zeros(lmh + 2, dtype=jnp.float64)
    ell = el[1:lmh]
    q2_k = jnp.maximum(q2[1:lmh], C.EPSQ2)
    eloq2 = ell * ell / q2_k
    eloq4 = eloq2 * eloq2
    gml = gm[1:lmh]
    ghl = gh[1:lmh]
    aden = (C.ADNM * gml + C.ADNH * ghl) * ghl
    bden = C.BDNM * gml + C.BDNH * ghl
    cden = 1.0
    besm = C.BSMH * ghl
    besh = C.BSHM * gml + C.BSHH * ghl
    rden = 1.0 / (aden * eloq4 + bden * eloq2 + cden)
    esm = (besm * eloq2 + C.CESM) * rden
    esh = (besh * eloq2 + C.CESH) * rden
    rdz = 2.0 / (z[1:lmh] - z[3 : lmh + 2])
    q1l = jnp.sqrt(q2_k)
    elqdz = ell * q1l * rdz
    akm = akm.at[1:lmh].set(elqdz * esm)
    akh = akh.at[1:lmh].set(elqdz * esh)
    return akm, akh


def _vdifq(lmh, dtdif, q2, el, z):
    """Traceable transcription of WRF ``VDIFQ``; returns updated q2.

    A bottom-to-top forward elimination + top-to-bottom back substitution
    (tridiagonal solve). Realized with ``lax.fori_loop`` so the recurrence is
    traceable for any (static) lmh.
    """

    esqhf = 0.5 * C.ESQ
    akq = jnp.zeros(lmh + 2, dtype=jnp.float64)
    cm = jnp.zeros(lmh + 2, dtype=jnp.float64)
    cr = jnp.zeros(lmh + 2, dtype=jnp.float64)
    dtoz = jnp.zeros(lmh + 2, dtype=jnp.float64)
    rsq2 = jnp.zeros(lmh + 2, dtype=jnp.float64)

    # k=1..lmh-2
    dtoz = dtoz.at[1 : lmh - 1].set((dtdif + dtdif) / (z[1 : lmh - 1] - z[3 : lmh + 1]))
    akq = akq.at[1 : lmh - 1].set(
        jnp.sqrt(jnp.maximum((q2[1 : lmh - 1] + q2[2 : lmh]) * 0.5, 0.0))
        * (el[1 : lmh - 1] + el[2 : lmh])
        * esqhf
        / (z[2 : lmh] - z[3 : lmh + 1])
    )
    cr = cr.at[1 : lmh - 1].set(-dtoz[1 : lmh - 1] * akq[1 : lmh - 1])

    cm = cm.at[1].set(dtoz[1] * akq[1] + 1.0)
    rsq2 = rsq2.at[1].set(q2[1])

    # forward sweep k=2..lmh-2
    def fwd(k, carry):
        cm, rsq2 = carry
        cf = -dtoz[k] * akq[k - 1] / cm[k - 1]
        cm = cm.at[k].set(-cr[k - 1] * cf + (akq[k - 1] + akq[k]) * dtoz[k] + 1.0)
        rsq2 = rsq2.at[k].set(-rsq2[k - 1] * cf + q2[k])
        return cm, rsq2

    cm, rsq2 = jax.lax.fori_loop(2, lmh - 1, fwd, (cm, rsq2))

    dtozs = (dtdif + dtdif) / (z[lmh - 1] - z[lmh + 1])
    akqs = (
        jnp.sqrt(jnp.maximum((q2[lmh - 1] + q2[lmh]) * 0.5, 0.0))
        * (el[lmh - 1] + C.ELZ0)
        * esqhf
        / (z[lmh] - z[lmh + 1])
    )
    cf = -dtozs * akq[lmh - 2] / cm[lmh - 2]
    q2 = q2.at[lmh - 1].set(
        (dtozs * akqs * q2[lmh] - rsq2[lmh - 2] * cf + q2[lmh - 1])
        / ((akq[lmh - 2] + akqs) * dtozs - cr[lmh - 2] * cf + 1.0)
    )

    # back substitution k=lmh-2..1
    def back(j, q2):
        k = (lmh - 2) - j + 1  # iterate j=1.. gives k=lmh-2,...,1
        q2 = q2.at[k].set((-cr[k] * q2[k + 1] + rsq2[k]) / cm[k])
        return q2

    q2 = jax.lax.fori_loop(1, lmh - 1, back, q2)
    return q2


def _vdifh(dtdif, lmh, lpbl, sz0, rkhs, clow, cts, species, nspec, rkh, z, rho):
    """Traceable transcription of WRF ``VDIFH``; returns updated species.

    ``species`` is ``(nspec_max=8, lmh+2)``; only rows 1..nspec are mixed.
    """

    cm = jnp.zeros(lmh + 2, dtype=jnp.float64)
    cr = jnp.zeros(lmh + 2, dtype=jnp.float64)
    dtoz = jnp.zeros(lmh + 2, dtype=jnp.float64)
    nrows = species.shape[0]
    rkct = jnp.zeros((nrows, lmh + 2), dtype=jnp.float64)
    rss = jnp.zeros((nrows, lmh + 2), dtype=jnp.float64)

    idx = jnp.arange(lmh + 2)
    mrow = jnp.arange(nrows)

    # k=1..lmh-1
    dtoz = dtoz.at[1:lmh].set(dtdif / (z[1:lmh] - z[2 : lmh + 1]))
    cr = cr.at[1:lmh].set(-dtoz[1:lmh] * rkh[1:lmh])
    rkhz = rkh * (z - jnp.concatenate([z[2:], z[-2:]]))  # rkh[k]*(z[k]-z[k+2])
    # rkct[m,k] = (k>=lpbl) ? rkhz[k]*cts[m]*0.5 : 0  (m=1..nspec)
    above = (idx >= lpbl) & (idx >= 1) & (idx <= lmh - 1)
    rkct = (rkhz[None, :] * cts[:, None] * 0.5) * above[None, :].astype(jnp.float64)
    rkct = rkct * ((mrow >= 1) & (mrow <= nspec))[:, None].astype(jnp.float64)

    rhok1 = rho[1]
    cm = cm.at[1].set(dtoz[1] * rkh[1] + rhok1)
    rss = rss.at[:, 1].set(-rkct[:, 1] * dtoz[1] + species[:, 1] * rhok1)

    # forward sweep k=2..lmh-1
    def fwd(k, carry):
        cm, rss = carry
        dtozl = dtoz[k]
        cf = -dtozl * rkh[k - 1] / cm[k - 1]
        rhok = rho[k]
        cm = cm.at[k].set(-cr[k - 1] * cf + (rkh[k - 1] + rkh[k]) * dtozl + rhok)
        rss = rss.at[:, k].set(
            -rss[:, k - 1] * cf + (rkct[:, k - 1] - rkct[:, k]) * dtozl + species[:, k] * rhok
        )
        return cm, rss

    cm, rss = jax.lax.fori_loop(2, lmh, fwd, (cm, rss))

    dtozs = dtdif / (z[lmh] - z[lmh + 1])
    rkhh = rkh[lmh - 1]
    cf = -dtozs * rkhh / cm[lmh - 1]
    cmb = cr[lmh - 1] * cf
    rhok = rho[lmh]

    rkss = rkhs * clow  # (nrows,)
    cmsb = -cmb + (rkhh + rkss) * dtozs + rhok
    rssb = -rss[:, lmh - 1] * cf + rkct[:, lmh - 1] * dtozs + species[:, lmh] * rhok
    surf_val = (dtozs * rkss * sz0 + rssb) / cmsb
    # only update active species rows at the surface
    active = ((mrow >= 1) & (mrow <= nspec))
    species = species.at[:, lmh].set(jnp.where(active, surf_val, species[:, lmh]))

    # back substitution k=lmh-1..1
    def back(j, species):
        k = (lmh - 1) - j + 1
        rcml = 1.0 / cm[k]
        new = (-cr[k] * species[:, k + 1] + rss[:, k]) * rcml
        species = species.at[:, k].set(jnp.where(active, new, species[:, k]))
        return species

    species = jax.lax.fori_loop(1, lmh, back, species)
    return species


def _vdifv(lmh, dtdif, uz0, vz0, rkms, u, v, rkm, z, rho):
    """Traceable transcription of WRF ``VDIFV``; returns updated (u, v)."""

    cm = jnp.zeros(lmh + 2, dtype=jnp.float64)
    cr = jnp.zeros(lmh + 2, dtype=jnp.float64)
    dtoz = jnp.zeros(lmh + 2, dtype=jnp.float64)
    rsu = jnp.zeros(lmh + 2, dtype=jnp.float64)
    rsv = jnp.zeros(lmh + 2, dtype=jnp.float64)

    dtoz = dtoz.at[1:lmh].set(dtdif / (z[1:lmh] - z[2 : lmh + 1]))
    cr = cr.at[1:lmh].set(-dtoz[1:lmh] * rkm[1:lmh])

    rhok1 = rho[1]
    cm = cm.at[1].set(dtoz[1] * rkm[1] + rhok1)
    rsu = rsu.at[1].set(u[1] * rhok1)
    rsv = rsv.at[1].set(v[1] * rhok1)

    def fwd(k, carry):
        cm, rsu, rsv = carry
        dtozl = dtoz[k]
        cf = -dtozl * rkm[k - 1] / cm[k - 1]
        rhok = rho[k]
        cm = cm.at[k].set(-cr[k - 1] * cf + (rkm[k - 1] + rkm[k]) * dtozl + rhok)
        rsu = rsu.at[k].set(-rsu[k - 1] * cf + u[k] * rhok)
        rsv = rsv.at[k].set(-rsv[k - 1] * cf + v[k] * rhok)
        return cm, rsu, rsv

    cm, rsu, rsv = jax.lax.fori_loop(2, lmh, fwd, (cm, rsu, rsv))

    dtozs = dtdif / (z[lmh] - z[lmh + 1])
    rkmh = rkm[lmh - 1]
    cf = -dtozs * rkmh / cm[lmh - 1]
    rhok = rho[lmh]
    rcmvb = 1.0 / ((rkmh + rkms) * dtozs - cr[lmh - 1] * cf + rhok)
    dtozak = dtozs * rkms
    u = u.at[lmh].set((dtozak * uz0 - rsu[lmh - 1] * cf + u[lmh] * rhok) * rcmvb)
    v = v.at[lmh].set((dtozak * vz0 - rsv[lmh - 1] * cf + v[lmh] * rhok) * rcmvb)

    def back(j, carry):
        u, v = carry
        k = (lmh - 1) - j + 1
        rcml = 1.0 / cm[k]
        u = u.at[k].set((-cr[k] * u[k + 1] + rsu[k]) * rcml)
        v = v.at[k].set((-cr[k] * v[k + 1] + rsv[k]) * rcml)
        return u, v

    u, v = jax.lax.fori_loop(1, lmh, back, (u, v))
    return u, v


def _bottom_up_from_top(work: jax.Array, n: int) -> jax.Array:
    """Convert a 1-based top-down (len n+2) work array to a bottom-up (len n)."""

    # bottom-up index k_b (0-based) maps to Fortran kflip = n - k_b (top-down).
    kflip = n - jnp.arange(n)
    return work[kflip]


def myjpbl_column_traceable(
    *, u, v, temperature, theta, qv, qc, p_mid, p_int, exner, dz, tke,
    tsk, xland, ustar, akhs, akms, chklowq, elflx, thz0, qz0, uz0, vz0, qsfc,
    ct=0.0, snow=0.0, sice=0.0, dt=60.0, stepbl=1, ht=0.0,
) -> dict:
    """Single MYJ PBL column, fully ``jit``/``vmap``-traceable (pure ``jnp``).

    Mirrors ``physics.pbl_myj.myjpbl_column`` line-for-line. All profile inputs
    are bottom-up length-``n`` arrays; ``n`` must be a static Python int at trace
    time (it is, in the operational scan). Surface scalars are 0-d arrays/floats.
    ``tke`` is ``TKE_MYJ`` (0.5*q**2), not q**2.
    """

    f = lambda a: jnp.asarray(a, jnp.float64)
    u_b = f(u)
    n = int(u_b.shape[0])
    v_b = f(v); t_b = f(temperature); th_b = f(theta)
    qv_b = f(qv); qc_b = f(qc); p_b = f(p_mid); pint_b = f(p_int)
    exner_b = f(exner); dz_b = f(dz); tke_b = f(tke)

    lmh = n
    dtturbl = f(dt) * f(stepbl)
    rdtturbl = 1.0 / dtturbl

    u_top = _one_based_top_down(u_b, n)
    v_top = _one_based_top_down(v_b, n)
    t_top = _one_based_top_down(t_b, n)
    th_top = _one_based_top_down(th_b, n)
    qv_top_mix = _one_based_top_down(qv_b, n)
    qc_top = _one_based_top_down(qc_b, n)
    p_top = _one_based_top_down(p_b, n)
    z = _one_based_interfaces(dz_b, f(ht), n)
    cwm = qc_top
    q2 = _one_based_top_down(2.0 * tke_b, n)

    idx = jnp.arange(n + 2)
    active = (idx >= 1) & (idx <= n)
    q_top = jnp.where(active, qv_top_mix / (1.0 + qv_top_mix), 0.0)
    the = jnp.where(active, (cwm * (-C.ELOCP / jnp.where(active, t_top, 1.0)) + 1.0) * th_top, 0.0)

    gm, gh, el, pblh, lpbl, lmxl, ct_work, mixht = _mixlen(
        lmh, u_top, v_top, t_top, the, q_top, cwm, q2, z, f(ct)
    )
    el, q2 = _prodq2(lmh, dtturbl, f(ustar), gm, gh, el, q2)
    kpbl = n - lpbl + 1
    akm, akh = _difcof(lmh, gm, gh, el, q2, z)

    # EXCH_H / EXCH_M (bottom-up length n): reference writes output index k_b-1
    # for k_b=1..n-1 with kflip = n - k_b. So output index j=0..n-2 maps to
    # kflip = n - (j+1) = n-1-j; output index n-1 stays 0.
    jout = jnp.arange(n)
    kflip = (n - 1) - jout  # valid for jout=0..n-2; jout=n-1 -> kflip=0 unused
    deltaz = 0.5 * (z[kflip] - z[kflip + 2])
    exch_h = jnp.where(jout <= n - 2, akh[kflip] * deltaz, 0.0)
    exch_m = jnp.where(jout <= n - 2, akm[kflip] * deltaz, 0.0)

    q2 = _vdifq(lmh, dtturbl, q2, el, z)

    # TKE_MYJ + EL_MYJ outputs (bottom-up)
    q2 = jnp.where(active, jnp.maximum(q2, C.EPSQ2), q2)
    kflip_full = (n + 1) - jnp.arange(1, n + 1)  # bottom-up k_b=1..n -> top-down
    tke_out = 0.5 * q2[kflip_full]
    el_out = jnp.where(kflip_full < n, el[kflip_full], 0.0)

    thsk = f(tsk) * (1.0e5 / pint_b[0]) ** C.CAPA
    rho_top = jnp.where(
        active, p_top / (C.R_D * jnp.where(active, t_top, 1.0) * (1.0 + C.P608 * q_top - cwm)), 0.0
    )

    nspec = 4
    species = jnp.zeros((8, n + 2), dtype=jnp.float64)
    species = species.at[1].set(the)
    species = species.at[2].set(q_top)
    species = species.at[3].set(qc_top)
    # row 4 (qci) stays zero

    akh_dens = jnp.zeros(n + 2, dtype=jnp.float64)
    rho_kp1 = jnp.concatenate([rho_top[1:], rho_top[-1:]])
    akh_dens = akh_dens.at[1:n].set(akh[1:n] * 0.5 * (rho_top[1:n] + rho_kp1[1:n]))

    seamask = f(xland) - 1.0
    thz0_work = (1.0 - seamask) * thsk + seamask * f(thz0)
    akhs_dens = f(akhs) * rho_top[n]

    # QSFC (land/sea branch)
    qfc1 = C.XLV * f(chklowq) * akhs_dens
    qfc1 = jnp.where((f(snow) > 0.0) | (f(sice) > 0.5), qfc1 * C.RLIVWV, qfc1)
    qsfc_land = jnp.where(qfc1 > 0.0, q_top[n] + f(elflx) / qfc1, f(qsfc))
    exnsfc = (1.0e5 / pint_b[0]) ** C.CAPA
    qsfc_sea = C.PQ0SEA / pint_b[0] * jnp.exp(
        C.A2 * (thsk - C.A3 * exnsfc) / (thsk - C.A4 * exnsfc)
    )
    is_land = seamask < 0.5
    qsfc_work = jnp.where(is_land, qsfc_land, qsfc_sea)
    qz0_work = (1.0 - seamask) * qsfc_work + seamask * f(qz0)

    sz0 = jnp.zeros(8, dtype=jnp.float64)
    clow = jnp.zeros(8, dtype=jnp.float64)
    cts = jnp.zeros(8, dtype=jnp.float64)
    sz0 = sz0.at[1].set(thz0_work)
    sz0 = sz0.at[2].set(qz0_work)
    clow = clow.at[1].set(1.0)
    clow = clow.at[2].set(f(chklowq))
    cts = cts.at[1].set(ct_work)

    species = _vdifh(dtturbl, lmh, lpbl, sz0, akhs_dens, clow, cts, species, nspec, akh_dens, z, rho_top)

    the_new = species[1]
    q_new = species[2]
    qc_new = species[3]
    cwm_new = qc_new

    # tendencies (bottom-up): k_b=1..n -> kflip = n+1-k_b
    thnew = the_new[kflip_full] + cwm_new[kflip_full] * C.ELOCP / exner_b
    dtdt = (thnew - th_b) * rdtturbl
    qold = qv_b / (1.0 + qv_b)
    dqdt = (q_new[kflip_full] - qold) * rdtturbl
    rthblten = dtdt
    rqvblten = dqdt / (1.0 - q_new[kflip_full]) ** 2

    akm_dens = jnp.zeros(n + 2, dtype=jnp.float64)
    akm_dens = akm_dens.at[1:n].set(akm[1:n] * 0.5 * (rho_top[1:n] + rho_kp1[1:n]))
    uk = u_top
    vk = v_top
    akms_dens = f(akms) * rho_top[n]
    uk, vk = _vdifv(lmh, dtturbl, f(uz0), f(vz0), akms_dens, uk, vk, akm_dens, z, rho_top)

    rub = (uk[kflip_full] - u_b) * rdtturbl
    rvb = (vk[kflip_full] - v_b) * rdtturbl

    return {
        "TKE_MYJ": tke_out,
        "EXCH_H": exch_h,
        "EXCH_M": exch_m,
        "EL_MYJ": el_out,
        "RUBLTEN": rub,
        "RVBLTEN": rvb,
        "RTHBLTEN": rthblten,
        "RQVBLTEN": rqvblten,
        "PBLH": pblh,
        "KPBL": kpbl,
        "MIXHT": mixht,
        "AKH": _bottom_up_from_top(akh, n),
        "AKM": _bottom_up_from_top(akm, n),
        "QSFC": qsfc_work,
        "THZ0": thz0_work,
        "QZ0": qz0_work,
    }


def myj_columns(
    u, v, temperature, theta, qv, qc, p_mid, p_int, exner, dz, tke,
    *, tsk, xland, ustar, akhs, akms, chklowq, elflx, thz0, qz0, uz0, vz0, qsfc,
    ct=0.0, snow=0.0, sice=0.0, dt=60.0, stepbl=1, ht=0.0,
) -> dict:
    """``jax.vmap``-batched MYJ PBL over ``(ncol, nz)`` columns.

    Profile args are ``(ncol, nz)`` bottom-up; ``p_int`` is ``(ncol, nz+1)``.
    Surface coupling scalars are ``(ncol,)``. ``ct``/``snow``/``sice``/``ht``
    accept either ``(ncol,)`` arrays or scalars. Returns batched outputs.
    """

    ncol = u.shape[0]
    f1 = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float64).reshape(-1), (ncol,))

    def _one(u_c, v_c, t_c, th_c, qv_c, qc_c, pmid_c, pint_c, ex_c, dz_c, tke_c,
             tsk_c, xland_c, ust_c, akhs_c, akms_c, chk_c, elflx_c, thz0_c,
             qz0_c, uz0_c, vz0_c, qsfc_c, ct_c, snow_c, sice_c, ht_c):
        return myjpbl_column_traceable(
            u=u_c, v=v_c, temperature=t_c, theta=th_c, qv=qv_c, qc=qc_c,
            p_mid=pmid_c, p_int=pint_c, exner=ex_c, dz=dz_c, tke=tke_c,
            tsk=tsk_c, xland=xland_c, ustar=ust_c, akhs=akhs_c, akms=akms_c,
            chklowq=chk_c, elflx=elflx_c, thz0=thz0_c, qz0=qz0_c, uz0=uz0_c,
            vz0=vz0_c, qsfc=qsfc_c, ct=ct_c, snow=snow_c, sice=sice_c,
            dt=dt, stepbl=stepbl, ht=ht_c,
        )

    return jax.vmap(_one)(
        jnp.asarray(u, jnp.float64), jnp.asarray(v, jnp.float64),
        jnp.asarray(temperature, jnp.float64), jnp.asarray(theta, jnp.float64),
        jnp.asarray(qv, jnp.float64), jnp.asarray(qc, jnp.float64),
        jnp.asarray(p_mid, jnp.float64), jnp.asarray(p_int, jnp.float64),
        jnp.asarray(exner, jnp.float64), jnp.asarray(dz, jnp.float64),
        jnp.asarray(tke, jnp.float64),
        f1(tsk), f1(xland), f1(ustar), f1(akhs), f1(akms), f1(chklowq),
        f1(elflx), f1(thz0), f1(qz0), f1(uz0), f1(vz0), f1(qsfc),
        f1(ct), f1(snow), f1(sice), f1(ht),
    )


__all__ = ["myjpbl_column_traceable", "myj_columns"]
