"""JAX (jit/vmap-traceable, GPU-batchable) port of WRF Grell-Freitas cumulus.

This is a traceable re-derivation of :mod:`gpuwrf.physics._gf_reference` (the
line-faithful sequential NumPy port of pristine ``module_cu_gf_deep.F`` /
``module_cu_gf_sh.F`` / ``module_cu_gf_wrfdrv.F``). The algorithm is preserved
1:1; only the control flow is re-expressed so the entire single-column call path
``gfdrv -> cup_gf (deep) + cup_gf_sh (shallow)`` is a single ``jax.jit`` /
``jax.vmap`` traceable function with NO Python data-dependent branches and NO
host transfer in the column loop.

Conversion strategy (matches the v0.9.0 GF-vmap sprint contract, strategy A):

- Data-dependent first-crossing searches (``kbmax``/``kdet``/``pmin_lev``/``ktop``
  ``break`` loops) -> ``jnp.argmax`` over a boolean crossing mask + a "found"
  flag, exactly reproducing Fortran ``DO k ... ; IF(cond) THEN ...; EXIT``.
- The ``cup_kbcon`` ``while-True`` cap-increment search -> ``jax.lax.while_loop``
  with a bounded iteration guard (``2*(kbmax+2)`` iters is a hard upper bound on
  the GO-TO-32 retest loop, which strictly increases ``kbcon`` or ``k22``).
- The ``jmini`` downdraft-origin ``while keep_going`` search ->
  ``jax.lax.while_loop``.
- The 16-member closure ensemble (``cup_forcing_ens_3d``) -> a vectorized
  ``jnp`` array of the 16 members (no per-member Python branch).
- The beta-PDF ``math.gamma`` -> ``jnp.exp(gammaln(.))``.
- Pervasive ``ierr`` short-circuiting -> ``jnp.where`` masking (compute-all,
  select-on-``ierr``): every block runs unconditionally but its writes are gated
  on the traced ``ierr`` so the downstream state is identical to the early-return
  Fortran path. ``ierr`` itself is a traced int updated with ``jnp.where``.

Indexing: WRF arrays are Fortran ``1..KX``. We use length ``KX+1`` arrays with
index 0 unused, matching the reference 1:1 so a JAX array ``a[k]`` mirrors the
Fortran/NumPy ``a[k]``. ``ktf == kte == KX`` for a single column.

The module exposes ``gfdrv_column`` (one column) and ``gfdrv_batched`` (a
``jax.vmap`` over the leading column axis), plus ``GF_OUTPUT_KEYS``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

# --- deep-module parameters (module_cu_gf_deep.F head) ---
G = 9.81
CP = 1004.0
XLV = 2.5e6
R_V = 461.0
C1 = 0.001
FRH_THRESH = 0.9
RH_THRESH = 0.97
BETAJB = 1.5
USE_EXCESS = 1
FLUXTUNE = 1.5
PGCD = 1.0
MAXENS3 = 16

# --- shallow-module parameters (module_cu_gf_sh.F head) ---
C1_SHAL = 0.0
C0_SHAL = 0.001

F64 = jnp.float64
TINY = 2.2250738585072014e-308  # np.finfo(np.float64).tiny


def _gamma(x):
    """math.gamma via lgamma (x>0 in all GF call sites)."""
    return jnp.exp(gammaln(x))


def _idx(kx):
    """1-based index vector [0,1,2,...,kx] as an int array (index 0 unused)."""
    return jnp.arange(kx + 1)


# ---------------------------------------------------------------------------
# Vectorized helpers replacing the sequential reference functions.
# Each operates on length (KX+1) JAX arrays, kx static (Python int).
# ---------------------------------------------------------------------------


def satvap(temp2):
    """REAL FUNCTION satvap (Goff-Gratch), elementwise + branch-as-where."""
    ln10 = jnp.log(10.0)
    # ice branch (temp2-273.155 < -20)
    toot = 273.16 / temp2
    toto = 1.0 / toot
    eilog = (-9.09718 * (toot - 1.0) - 3.56654 * (jnp.log(toot) / ln10)
             + 0.876793 * (1.0 - toto) + (jnp.log(6.1071) / ln10))
    ice = 10.0 ** eilog
    # water branch
    tsot = 373.16 / temp2
    ewlog = -7.90298 * (tsot - 1.0) + 5.02808 * (jnp.log(tsot) / ln10)
    ewlog2 = ewlog - 1.3816e-07 * (10.0 ** (11.344 * (1.0 - (1.0 / tsot))) - 1.0)
    ewlog3 = ewlog2 + 0.0081328 * (10.0 ** (-3.49149 * (tsot - 1.0)) - 1.0)
    ewlog4 = ewlog3 + (jnp.log(1013.246) / ln10)
    water = 10.0 ** ewlog4
    is_ice = (temp2 - 273.155) < -20.0
    return jnp.where(is_ice, ice, water)


def cup_env(z, t, q, p, z1, psur, ierr, itest, kx):
    """cup_env. itest in {-1,0,1,2}. Returns z, qes, he, hes (length kx+1)."""
    k = _idx(kx)
    active = (k >= 1) & (k <= kx)  # ktf == kx
    e = satvap(t)
    qes = 0.622 * e / jnp.maximum(1.0e-8, (p - e))
    qes = jnp.where(qes <= 1.0e-16, 1.0e-16, qes)
    qes = jnp.where(qes < q, q, qes)
    tv = t + 0.608 * q * t

    # itest in (1,0): build z by hydrostatic integration (sequential cumsum)
    # z[1] = max(0,z1) - (log p1 - log psur)*287*tv1/9.81
    # z[k] = z[k-1] - (log pk - log p_{k-1})*287*tvbar/9.81
    tvbar = 0.5 * tv + 0.5 * jnp.roll(tv, 1)
    logp = jnp.log(jnp.where(active, p, 1.0))
    dlog = logp - jnp.roll(logp, 1)
    incr = -dlog * 287.0 * tvbar / 9.81  # for k>=2
    z1_val = jnp.maximum(0.0, z1) - (jnp.log(p[1]) - jnp.log(psur)) * 287.0 * tv[1] / 9.81
    # cumulative sum of incr over k=2..kx starting from z1_val
    incr_masked = jnp.where(k >= 2, incr, 0.0)
    z_cumsum = jnp.cumsum(jnp.where(k == 1, z1_val, incr_masked))
    z_built = jnp.where(active, z_cumsum, z)
    use_built = (itest == 1) | (itest == 0)
    z_out = jnp.where(use_built, z_built, z)

    # itest == 2: z from he (he not yet defined here; ref computes from he input)
    # The reference computes z[k]=(he-1004*t-2.5e6*q)/9.81. In all GF call sites
    # itest=-1, so itest==2 path is never taken; we keep z_out as-is for that.

    # he/hes
    he_calc = 9.81 * z_out + 1004.0 * t + 2.5e6 * q
    he = jnp.where(itest <= 0, he_calc, z * 0.0)  # itest<=0 -> compute he
    hes = 9.81 * z_out + 1004.0 * t + 2.5e6 * qes
    he = jnp.where(he >= hes, hes, he)

    # mask k range and ierr
    nm = ~active
    qes = jnp.where(nm, 0.0, qes)
    he = jnp.where(nm, 0.0, he)
    hes = jnp.where(nm, 0.0, hes)
    bad = ierr != 0
    return (jnp.where(bad, z, z_out), jnp.where(bad, 0.0, qes),
            jnp.where(bad, 0.0, he), jnp.where(bad, 0.0, hes))


def cup_env_clev(t, qes, q, he, hes, z, p, psur, z1, ierr, kx):
    """cup_env_clev -> *_cup arrays (length kx+1)."""
    k = _idx(kx)
    qm = jnp.roll(qes, 1); qpm = jnp.roll(q, 1); hesm = jnp.roll(hes, 1)
    hem = jnp.roll(he, 1); zm = jnp.roll(z, 1); pm = jnp.roll(p, 1); tm = jnp.roll(t, 1)
    qes_cup = 0.5 * (qm + qes)
    q_cup = 0.5 * (qpm + q)
    hes_cup = 0.5 * (hesm + hes)
    he_cup = 0.5 * (hem + he)
    he_cup = jnp.where(he_cup > hes_cup, hes_cup, he_cup)
    z_cup = 0.5 * (zm + z)
    p_cup = 0.5 * (pm + p)
    t_cup = 0.5 * (tm + t)
    gamma_cup = (XLV / CP) * (XLV / (R_V * t_cup * t_cup)) * qes_cup
    # k==1 overrides
    set1 = (k == 1)
    qes_cup = jnp.where(set1, qes, qes_cup)
    q_cup = jnp.where(set1, q, q_cup)
    hes_cup = jnp.where(set1, 9.81 * z1 + 1004.0 * t + 2.5e6 * qes, hes_cup)
    he_cup = jnp.where(set1, 9.81 * z1 + 1004.0 * t + 2.5e6 * q, he_cup)
    z_cup = jnp.where(set1, z1, z_cup)
    p_cup = jnp.where(set1, psur, p_cup)
    t_cup = jnp.where(set1, t, t_cup)
    gamma_cup = jnp.where(set1, XLV / CP * (XLV / (R_V * t_cup * t_cup)) * qes_cup, gamma_cup)
    # zero outside 1..kx
    active = (k >= 1) & (k <= kx)
    z0 = lambda a: jnp.where(active, a, 0.0)
    out = (z0(qes_cup), z0(q_cup), z0(he_cup), z0(hes_cup), z0(z_cup),
           z0(p_cup), z0(gamma_cup), z0(t_cup))
    bad = ierr != 0
    return tuple(jnp.where(bad, 0.0, a) for a in out)


def get_cloud_bc(array, k22, add, kx):
    """get_cloud_bc: mean of array over k22-2..k22 (order_aver=3), traced k22."""
    order_aver = 3
    local = jnp.minimum(k22, order_aver)
    k = _idx(kx)
    # sum array[k22 - i + 1] for i in 1..local  ==  array[k22], array[k22-1], array[k22-2]
    s = jnp.zeros((), F64)
    for i in range(1, order_aver + 1):
        idx = k22 - i + 1
        take = (i <= local)
        s = s + jnp.where(take, array[idx], 0.0)
    return s / local.astype(F64) + add


def cup_minimi(array, ks, kend, ierr, kx):
    """index of min of array over ks..max(ks+1,kend), traced ks/kend."""
    k = _idx(kx)
    kstop = jnp.maximum(ks + 1, kend)
    rng = (k >= ks) & (k <= kstop)
    big = jnp.where(rng, array, jnp.inf)
    # Fortran: kt=ks; x=array[ks]; for k=ks+1..kstop: if array[k]<x then kt=k.
    # i.e. strict-less updates -> argmin with ties going to first (lowest k).
    # jnp.argmin returns first min index already.
    kt = jnp.argmin(big)
    kt = jnp.where(ierr != 0, ks, kt)
    return kt.astype(jnp.int32)


def cup_maximi(array, ks, ke, ierr, kx):
    """index of max over ks..ke; Fortran uses >= so LAST max wins."""
    k = _idx(kx)
    rng = (k >= ks) & (k <= ke)
    small = jnp.where(rng, array, -jnp.inf)
    # Fortran '>=' means the last occurrence of the max is kept.
    rev = small[::-1]
    kt_rev = jnp.argmax(rev)
    kt = kx - kt_rev
    kt = jnp.where(ierr != 0, ks, kt)
    return kt.astype(jnp.int32)


def _maxloc1(arr, kx):
    """Fortran maxloc(arr(1:kx),1): FIRST index of max in 1..kx."""
    k = _idx(kx)
    rng = (k >= 1) & (k <= kx)
    small = jnp.where(rng, arr, -jnp.inf)
    return jnp.argmax(small).astype(jnp.int32)


def _argmax_range(arr, lo, hi, kx):
    """Fortran maxloc(arr(lo:hi),1)+lo-1: FIRST index of max in lo..hi."""
    k = _idx(kx)
    rng = (k >= lo) & (k <= hi)
    small = jnp.where(rng, arr, -jnp.inf)
    return jnp.argmax(small).astype(jnp.int32)


def cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, kx):
    """cloud work function aa0 (sum over kbcon..ktop, k>=2)."""
    k = _idx(kx)
    zm = jnp.roll(z, 1)
    dz = z - zm
    da = zu * dz * (9.81 / (1004.0 * t_cup)) * jnp.roll(dby, 1) / (1.0 + gamma_cup)
    rng = (k >= 2) & (k <= kx) & (k >= kbcon) & (k <= ktop)
    contrib = jnp.where(rng, jnp.maximum(0.0, da), 0.0)
    aa0 = jnp.sum(contrib)
    return jnp.where(ierr != 0, 0.0, aa0)


def cup_up_aa1bl_full(z, t, tn, q, qo, dtime, kbcon, ierr, kx):
    """cup_up_aa1bl: dz*g*(dtn+.608*dqo)/dtime, k in 2..kbcon."""
    k = _idx(kx)
    zm = jnp.roll(z, 1)
    dz = z - zm
    da = dz * 9.81 * (tn - t + 0.608 * (qo - q)) / dtime
    rng = (k >= 2) & (k <= kx) & (k <= kbcon)
    aa0 = jnp.sum(jnp.where(rng, da, 0.0))
    return jnp.where(ierr != 0, 0.0, aa0)


def _backward_first_below(zu, ml, thresh, kx):
    """Fortran: DO k=ml,1,-1; IF(zu(k)<thresh) THEN kb_adj=k+1; EXIT.

    Returns (kb_adj, found). If no k in 1..ml has zu(k)<thresh, found=False and
    kb_adj is left to the caller (Fortran leaves kb_adj unchanged).
    """
    k = _idx(kx)
    below = (zu < thresh) & (k >= 1) & (k <= ml)
    # we want the LARGEST k<=ml with below True (first hit scanning downward)
    any_below = jnp.any(below)
    # argmax on reversed gives last True -> largest index
    kk = jnp.where(below, k, -1)
    kbest = jnp.max(kk)  # largest k with below; -1 if none
    return (kbest + 1).astype(jnp.int32), any_below


def get_zu_zd_pdf_fim(p, zubeg, draft, kb, kt, kpbli, csum, beta_in,
                      kx, kts=1):
    """get_zu_zd_pdf_fim beta-PDF mass flux. ``draft`` is a static Python str.

    Mirrors the reference 1:1. kb,kt,kpbli traced ints; csum traced/scalar.
    Returns zu (length kx+1). ktf==kx, kte==kx.
    """
    ktf = kx
    kte = kx
    k = _idx(kx)
    kb_adj = jnp.maximum(kb, 2)

    if draft == "UP":
        lev_start = jnp.minimum(0.9, 0.4 + csum * 0.013)
        kb_adj = jnp.maximum(kb, 2)
        tunning = p[kt] + (p[kpbli] - p[kt]) * lev_start
        tunning = jnp.minimum(0.9, (tunning - p[kb_adj]) / (p[kt] - p[kb_adj]))
        tunning = jnp.maximum(0.2, tunning)
        beta = 1.3
    elif draft == "SH2":
        tunning = jnp.minimum(0.8, (p[kpbli] - p[kb_adj]) / (p[kt] - p[kb_adj]))
        tunning = jnp.maximum(0.2, tunning)
        beta = 2.5
    elif draft == "MID":
        kb_adj = jnp.maximum(kb, 2)
        tunning = p[kt] + (p[kb_adj] - p[kt]) * 0.9
        tunning = jnp.minimum(0.9, (tunning - p[kb_adj]) / (p[kt] - p[kb_adj]))
        tunning = jnp.maximum(0.2, tunning)
        beta = 1.3
    else:  # DOWN / DOWNM
        tunning = p[kb]
        tunning = jnp.minimum(0.9, (tunning - p[1]) / (p[kt] - p[1]))
        tunning = jnp.maximum(0.2, tunning)
        beta = 4.0

    alpha = (tunning * (beta - 2.0) + 1.0) / (1.0 - tunning)
    fzu = _gamma(alpha + beta) / (_gamma(alpha) * _gamma(beta))

    if draft in ("UP", "SH2", "MID"):
        # k in kb_adj..min(kte,kt): zu = zubeg + fzu*kratio^(a-1)*(1-kratio)^(b-1)
        denom = (p[kt] - p[kb_adj])
        kratio = (p - p[kb_adj]) / denom
        # guard pow of (negative or zero) -> only valid where 0<kratio<1 anyway;
        # masked range keeps only valid k, but pow must not produce nan inside.
        kr = jnp.clip(kratio, 1.0e-30, 1.0)
        body = zubeg + fzu * kr ** (alpha - 1.0) * jnp.clip(1.0 - kratio, 0.0, 1.0) ** (beta - 1.0)
        rng = (k >= kb_adj) & (k <= jnp.minimum(kte, kt))
        zu = jnp.where(rng, body, 0.0)
        hi = jnp.minimum(ktf, kt + 1)
        in_norm = (k >= kts) & (k <= hi)
        mx = jnp.max(jnp.where(in_norm, zu, -jnp.inf))
        zu = jnp.where(mx > 0.0, jnp.where(in_norm, zu / mx, zu), zu)
        ml = _maxloc1(zu, kte)
        kb_new, found = _backward_first_below(zu, ml, 1.0e-6, kte)
        kb_adj2 = jnp.where(found, kb_new, kb_adj)
        if draft in ("UP", "MID"):
            kb_adj2 = jnp.maximum(2, kb_adj2)
            zu = jnp.where(k < kb_adj2, 0.0, zu)
        # SH2 does NOT zero below kb_adj (matches reference)
        return zu
    else:  # DOWN
        denom = (p[kt] - p[1])
        kratio = (p - p[1]) / denom
        kr = jnp.clip(kratio, 1.0e-30, 1.0)
        body = fzu * kr ** (alpha - 1.0) * jnp.clip(1.0 - kratio, 0.0, 1.0) ** (beta - 1.0)
        rng = (k >= 2) & (k <= jnp.minimum(kt, ktf))
        zu = jnp.where(rng, body, 0.0)
        hi = jnp.minimum(ktf, kt + 1)
        in_norm = (k >= kts) & (k <= hi)
        fzu2 = jnp.max(jnp.where(in_norm, zu, -jnp.inf))
        zu = jnp.where(fzu2 > 0.0, jnp.where(in_norm, zu / fzu2, zu), zu)
        zu = jnp.where(k == 1, 0.0, zu)
        return zu


def get_lateral_massflux(zo_cup, zuo, cd_in, entr_rate_2d_in, ktop, kbcon, k22,
                         lambau, kx, with_u=True):
    """get_lateral_massflux. Both loops write output index ``j=k-1`` exactly once
    over disjoint ``j``-ranges, so there is no cross-iteration dependence and the
    whole thing vectorizes. Reindexed by ``j``: reads ``cd[j]``, ``zuo[j]``,
    ``zuo[j+1]``, ``dz=zo_cup[j+1]-zo_cup[j]``. Returns the 6 mass arrays plus the
    updated (cd, entr_rate_2d)."""
    ktf = kx
    k = _idx(kx)  # acts as j
    cd = cd_in
    entr = entr_rate_2d_in
    zup = jnp.roll(zuo, -1)       # zuo[j+1]
    zop = jnp.roll(zo_cup, -1)    # zo_cup[j+1]
    dz = zop - zo_cup             # dz at j = zo_cup[j+1]-zo_cup[j]
    mlz = _maxloc1(zuo, kx)

    # ---- loop 1: k in max(2,k22+1)..mlz  ->  j in max(1,k22)..mlz-1 ----
    j1lo = jnp.maximum(1, k22)
    rng1 = (k >= j1lo) & (k <= mlz - 1)
    det1 = cd * dz * zuo                 # up_massdetro[j] = cd[j]*dz*zuo[j]
    ent1 = zup - zuo + det1              # up_massentro[j] = zuo[j+1]-zuo[j]+det
    neg = ent1 < 0.0
    ent1f = jnp.where(neg, 0.0, ent1)
    det1f = jnp.where(neg, zuo - zup, det1)
    cd1 = jnp.where(neg & (zuo > 0.0), det1f / (dz * zuo), cd)
    entr1 = jnp.where(zuo > 0.0, ent1f / (dz * zuo), entr)

    # ---- loop 2: k in mlz+1..ktop  ->  j in mlz..ktop-1 ----
    rng2 = (k >= mlz) & (k <= ktop - 1)
    ent2 = entr * dz * zuo
    det2 = zuo + ent2 - zup
    neg2 = det2 < 0.0
    det2f = jnp.where(neg2, 0.0, det2)
    ent2f = jnp.where(neg2, zup - zuo, ent2)
    entr2 = jnp.where(neg2 & (zuo > 0.0), ent2f / (dz * zuo), entr)
    cd2 = jnp.where(zuo > 0.0, det2f / (dz * zuo), cd)

    up_massentro = jnp.where(rng1, ent1f, jnp.where(rng2, ent2f, 0.0))
    up_massdetro = jnp.where(rng1, det1f, jnp.where(rng2, det2f, 0.0))
    cd = jnp.where(rng1, cd1, jnp.where(rng2, cd2, cd))
    entr = jnp.where(rng1, entr1, jnp.where(rng2, entr2, entr))

    # up_massdetro[ktop]=zuo[ktop]; up_massentro[ktop]=0
    at_ktop = (k == ktop)
    up_massdetro = jnp.where(at_ktop, zuo, up_massdetro)
    up_massentro = jnp.where(at_ktop, 0.0, up_massentro)
    # k in ktop+1..ktf: zero cd, entr, ent, det
    above = (k >= ktop + 1) & (k <= ktf)
    cd = jnp.where(above, 0.0, cd)
    entr = jnp.where(above, 0.0, entr)
    up_massentro = jnp.where(above, 0.0, up_massentro)
    up_massdetro = jnp.where(above, 0.0, up_massdetro)

    # up_massentr/up_massdetr = entro/detro for k-1 in 1..ktf-2 (k in 2..ktf-1)
    in_emr = (k >= 1) & (k <= ktf - 2)
    up_massentr = jnp.where(in_emr, up_massentro, 0.0)
    up_massdetr = jnp.where(in_emr, up_massdetro, 0.0)

    up_massentru = jnp.zeros(kx + 1, F64)
    up_massdetru = jnp.zeros(kx + 1, F64)
    if with_u:
        up_massentru = jnp.where(in_emr, up_massentro + lambau * up_massdetro, 0.0)
        up_massdetru = jnp.where(in_emr, up_massdetro + lambau * up_massdetro, 0.0)

    return (up_massentro, up_massdetro, up_massentr, up_massdetr,
            up_massentru, up_massdetru, cd, entr)


def _hcot_profile(hkb, start_level, kbmax, entr_rate, z_cup, heo, kx):
    """Build hcot: hcot[1..start_level]=hkb; for k=start_level+1..kbmax+3:
    hcot[k]=((1-.5*er*dz)*hcot[k-1]+er*dz*heo[k-1])/(1+.5*er*dz). Linear scan."""
    k = _idx(kx)
    zm = jnp.roll(z_cup, 1)
    dz = z_cup - zm                      # dz[k]=z_cup[k]-z_cup[k-1]
    heom = jnp.roll(heo, 1)              # heo[k-1]
    a = (1.0 - 0.5 * entr_rate * dz) / (1.0 + 0.5 * entr_rate * dz)
    b = (entr_rate * dz * heom) / (1.0 + 0.5 * entr_rate * dz)
    kmax = kbmax + 3
    # scan k=1..kx; carry = previous hcot value (hcot[k-1])
    def step(prev, kk):
        in_flat = kk <= start_level
        in_rec = (kk >= start_level + 1) & (kk <= kmax)
        val = jnp.where(in_flat, hkb,
                        jnp.where(in_rec, a[kk] * prev + b[kk], 0.0))
        return val, val
    ks = jnp.arange(1, kx + 1)
    _, vals = jax.lax.scan(step, jnp.zeros((), F64), ks)
    hcot = jnp.zeros(kx + 1, F64).at[1:].set(vals)
    return hcot


def cup_kbcon(cap_inc, iloop_in, k22_in, he_cup, hes_cup, hkb_in, ierr_in, kbmax,
              p_cup, cap_max, ztexec, zqexec, z_cup, entr_rate, heo, imid, kx):
    """cup_kbcon as a bounded lax.while_loop over k22-raises. Returns
    (k22, kbcon, hkb, ierr). cap_max/imid are scalars (Python or traced)."""
    iloop = jnp.where((cap_max > 200) & (imid == 1), 5, iloop_in)
    x_add = XLV * zqexec + CP * ztexec

    def first_crossing(kbcon0, hcot):
        # Fortran 32/31 loop: from kbcon0 upward, the first k with
        # hcot[k]>=hes_cup[k] is accepted ONLY if k<=kbmax+2; otherwise the loop
        # keeps incrementing and overflow-returns the moment the (incremented)
        # kbcon exceeds kbmax+2. The first test happens at kbcon0 with no lower
        # guard, so when kbcon0 itself > kbmax+2 the overflow value is kbcon0+1.
        k = _idx(kx)
        ge_all = (hcot >= hes_cup) & (k >= kbcon0)
        found_within = jnp.any(ge_all & (k <= kbmax + 2))
        kfirst = jnp.argmax(ge_all & (k <= kbmax + 2))
        overflow_kb = jnp.maximum(kbmax + 3, kbcon0 + 1)
        kbcon = jnp.where(found_within, kfirst, overflow_kb).astype(jnp.int32)
        return kbcon, found_within

    # carry: (k22, kbcon, hkb, ierr, done)
    def cond(state):
        _k22, _kb, _hkb, _ierr, done = state
        return jnp.logical_not(done)

    def body(state):
        k22, kbcon, hkb, ierr, _done = state
        # iloop==5 starts kbcon at k22, else k22+1 (we recompute crossing from there)
        kb_start = jnp.where(iloop == 5, k22, k22 + 1).astype(jnp.int32)
        hcot = _hcot_profile(hkb, k22, kbmax, entr_rate, z_cup, heo, kx)
        kbcon, found = first_crossing(kb_start, hcot)
        overflow = jnp.logical_not(found)  # kbcon>kbmax+2
        # overflow: ierr=3 (if iloop!=4); terminate
        ierr_of = jnp.where(overflow & (iloop != 4), 3, ierr)
        # found branch decisions:
        cond_a = (kbcon - k22) == 1
        cond_b = (iloop == 5) & ((kbcon - k22) <= 2)
        pbcdif0 = -p_cup[kbcon] + p_cup[k22]
        pbcdif = jnp.where((iloop == 5) & (cap_max > 200), -p_cup[kbcon] + cap_max, pbcdif0)
        plus = jnp.maximum(25.0, cap_max - (iloop.astype(F64) - 1.0) * cap_inc)
        plus = jnp.where(iloop == 4, cap_max, plus)
        plus = jnp.where(iloop == 5, 150.0, plus)
        cond_c = pbcdif <= plus
        terminate_found = cond_a | cond_b | cond_c
        # raise-k22 path
        k22_new = k22 + 1
        hkb_new = get_cloud_bc(he_cup, k22_new, x_add, kx)
        # after raise: rebuild + check kbcon overflow (handled next iter), but
        # Fortran checks kbcon>kbmax+2 right after rebuild using kbcon=k22+1 (or
        # k22 if iloop5). We let the next iteration's first_crossing handle it;
        # however Fortran's post-rebuild overflow check uses kbcon BEFORE search.
        kb_after = jnp.where(iloop == 5, k22_new, k22_new + 1).astype(jnp.int32)
        raise_overflow = kb_after > kbmax + 2
        ierr_raise = jnp.where(raise_overflow & (iloop != 4), 3, ierr)

        # Assemble next state.
        # Case overflow (not found): terminate with ierr_of, kbcon=kbmax+3.
        # Case found & terminate_found: terminate, keep kbcon/k22/hkb/ierr.
        # Case found & not terminate: raise k22. If raise causes overflow ->
        #   terminate with ierr_raise and kbcon=kb_after; else continue.
        do_terminate = overflow | terminate_found | (jnp.logical_not(overflow) & jnp.logical_not(terminate_found) & raise_overflow)

        # next k22
        next_k22 = jnp.where(overflow | terminate_found, k22, k22_new)
        # next kbcon
        next_kbcon = jnp.where(
            overflow, kbcon,
            jnp.where(terminate_found, kbcon, kb_after)).astype(jnp.int32)
        # next hkb
        next_hkb = jnp.where(overflow | terminate_found, hkb, hkb_new)
        # next ierr
        next_ierr = jnp.where(
            overflow, ierr_of,
            jnp.where(terminate_found, ierr,
                      jnp.where(raise_overflow, ierr_raise, ierr)))
        return (next_k22.astype(jnp.int32), next_kbcon, next_hkb,
                next_ierr.astype(jnp.int32), do_terminate)

    # initial: kbcon=1; if ierr!=0 return immediately (done=True)
    init_done = ierr_in != 0
    init = (jnp.asarray(k22_in, jnp.int32), jnp.asarray(1, jnp.int32),
            jnp.asarray(hkb_in, F64), jnp.asarray(ierr_in, jnp.int32),
            jnp.asarray(init_done))
    k22, kbcon, hkb, ierr, _ = jax.lax.while_loop(cond, body, init)
    # if ierr was nonzero on entry, kbcon stays 1 (Fortran sets kbcon=1 then returns)
    return k22, kbcon, hkb, ierr


def rates_up_pdf(name, ktop_in, ierr_in, p_cup, entr_rate_2d, hkbo, heo,
                 heso_cup, z_cup, xland, kstabi, k22, kbcon_in, kpbl, csum,
                 kx):
    """rates_up_pdf. ``name`` is static ('deep'/'mid'/'shallow'). Returns
    (kbcon, ktop, ktopdby, ierr, zuo). entr_rate_2d is NOT mutated here (the
    reference only mutates zuo/kbcon/ktop/ktopdby/ierr)."""
    ktf = kx
    kte = kx
    kts = 1
    dbythresh = 1.0
    zustart = 0.1
    beta_u = jnp.maximum(0.1, 0.2 - csum * 0.01)
    k = _idx(kx)
    kbcon = jnp.maximum(kbcon_in, 2)
    bad = ierr_in != 0
    start_level = k22
    zubeg = zustart

    # zuo recurrence start_level..kbcon: zuo[k]=zuo[k-1]*(1+dz*(entr[k-1]-1e-9))
    zm = jnp.roll(z_cup, 1)
    dz = z_cup - zm
    entrm = jnp.roll(entr_rate_2d, 1)  # entr[k-1]
    factor = 1.0 + dz * (entrm - 1.0e-9)  # multiplier for k in start_level+1..kbcon

    def zstep(prev, kk):
        is_start = kk == start_level
        in_rec = (kk >= start_level + 1) & (kk <= kbcon)
        val = jnp.where(is_start, zustart,
                        jnp.where(in_rec, prev * factor[kk], 0.0))
        return val, val
    ks = jnp.arange(1, kx + 1)
    _, zvals = jax.lax.scan(zstep, jnp.zeros((), F64), ks)
    zuo = jnp.zeros(kx + 1, F64).at[1:].set(zvals)

    if name == "deep":
        # hcot recurrence start_level+1..ktf-2, dby cumulative over k>=kbcon
        a = (1.0 - 0.5 * entrm * dz) / (1.0 + 0.5 * entrm * dz)
        b = (entrm * dz * jnp.roll(heo, 1)) / (1.0 + 0.5 * entrm * dz)
        kmax = ktf - 2

        def hstep(carry, kk):
            prev_h, prev_dby = carry
            is_start = kk == start_level
            in_rec = (kk >= start_level + 1) & (kk <= kmax)
            h = jnp.where(is_start, hkbo, jnp.where(in_rec, a[kk] * prev_h + b[kk], prev_h * 0.0))
            # actually hcot at levels outside set range stays 0 except start; but
            # the recurrence only reads prev_h, so carry must hold last hcot.
            h_carry = jnp.where(is_start, hkbo, jnp.where(in_rec, a[kk] * prev_h + b[kk], prev_h))
            add = jnp.where(in_rec & (kk >= kbcon), (h_carry - heso_cup[kk]) * dz[kk], 0.0)
            dby_new = prev_dby + add
            return (h_carry, dby_new), (h_carry, dby_new)
        (_, _), (hcot_v, dby_v) = jax.lax.scan(hstep, (jnp.zeros((), F64), jnp.zeros((), F64)), ks)
        hcot = jnp.zeros(kx + 1, F64).at[1:].set(hcot_v)
        dby = jnp.zeros(kx + 1, F64).at[1:].set(dby_v)
        # dbm[k] = hcot[k]-heso_cup[k] for k in rec & k>=kbcon, else 0
        in_rec_full = (k >= start_level + 1) & (k <= kmax) & (k >= kbcon)
        dbm = jnp.where(in_rec_full, hcot - heso_cup, 0.0)

        ktopdby = _maxloc1(dby, kte)
        kklev = _maxloc1(dbm, kte)
        dby_max = jnp.max(jnp.where((k >= 1) & (k <= kte), dby, -jnp.inf))
        # forward search k in maxloc(dby)+1..ktf-2: first dby[k]<dbythresh*dby_max
        ml_dby = _maxloc1(dby, kte)
        crossing = (dby < dbythresh * dby_max) & (k >= ml_dby + 1) & (k <= ktf - 2)
        found = jnp.any(crossing)
        kfirst = jnp.argmax(crossing)
        kfinalzu = jnp.where(found, kfirst - 1, ktf - 2).astype(jnp.int32)
        # ktop = kfinalzu (then maybe 0 below)
        too_shallow = kfinalzu <= kbcon + 2
        ierr = jnp.where(bad, ierr_in, jnp.where(too_shallow, 41, ierr_in)).astype(jnp.int32)
        ktop = jnp.where(bad, ktop_in, jnp.where(too_shallow, 0, kfinalzu)).astype(jnp.int32)
        # zu only if not too_shallow and not bad
        zu_pdf = get_zu_zd_pdf_fim(p_cup, zubeg, "UP", k22, kfinalzu, kstabi, csum, beta_u, kx)
        use_pdf = (~bad) & (~too_shallow)
        zuo_out = jnp.where(use_pdf, zu_pdf, zuo)
        # if bad, return original (zuo computed from recurrence is irrelevant under bad)
        zuo_out = jnp.where(bad, zuo, zuo_out)
        kbcon_out = jnp.where(bad, kbcon_in, kbcon)  # ref returns max(kbcon,2) when not bad? ref sets kbcon=max(kbcon,2) BEFORE ierr check, so even bad returns max
        kbcon_out = kbcon  # max(kbcon_in,2) always (set before ierr check)
        ktopdby_out = jnp.where(bad, 0, ktopdby).astype(jnp.int32)
        return kbcon_out, ktop, ktopdby_out, ierr, zuo_out

    elif name == "shallow":
        too_shallow = ktop_in <= kbcon + 2
        kfinalzu = ktop_in
        ktopdby = ktop_in
        ierr = jnp.where(bad, ierr_in, jnp.where(too_shallow, 41, ierr_in)).astype(jnp.int32)
        ktop = jnp.where(bad, ktop_in, jnp.where(too_shallow, 0, ktop_in)).astype(jnp.int32)
        zu_pdf = get_zu_zd_pdf_fim(p_cup, zubeg, "SH2", k22, kfinalzu, kpbl, csum, beta_u, kx)
        use_pdf = (~bad) & (~too_shallow)
        zuo_out = jnp.where(use_pdf, zu_pdf, zuo)
        zuo_out = jnp.where(bad, zuo, zuo_out)
        return kbcon, ktop, jnp.asarray(ktopdby, jnp.int32), ierr, zuo_out

    else:  # mid (not used by cu=3)
        too_shallow = ktop_in <= kbcon + 2
        kfinalzu = ktop_in
        ktopdby = ktop_in + 1
        ierr = jnp.where(bad, ierr_in, jnp.where(too_shallow, 41, ierr_in)).astype(jnp.int32)
        ktop = jnp.where(bad, ktop_in, jnp.where(too_shallow, 0, ktop_in)).astype(jnp.int32)
        zu_pdf = get_zu_zd_pdf_fim(p_cup, zubeg, "MID", k22, kfinalzu, kbcon, csum, beta_u, kx)
        use_pdf = (~bad) & (~too_shallow)
        zuo_out = jnp.where(use_pdf, zu_pdf, zuo)
        zuo_out = jnp.where(bad, zuo, zuo_out)
        return kbcon, ktop, jnp.asarray(ktopdby, jnp.int32), ierr, zuo_out


def cup_dd_moisture(zd, hcd, hes_cup, qes_cup, q_cup, z_cup, dd_massentr,
                    dd_massdetr, jmin, ierr_in, gamma_cup, q, he, iloop, kx):
    """cup_dd_moisture. Downward scan ki=jmin-1..1 with denom-break (ierr=51)."""
    k = _idx(kx)
    bad = ierr_in != 0
    zm1 = jnp.roll(z_cup, -1)  # z_cup[k+1]
    dz0 = z_cup[jmin + 1] - z_cup[jmin]
    qcd_j = q_cup[jmin]
    dh_j = hcd[jmin] - hes_cup[jmin]
    qrcd_j = jnp.where(dh_j < 0,
                       qes_cup[jmin] + (1.0 / XLV) * (gamma_cup[jmin] / (1.0 + gamma_cup[jmin])) * dh_j,
                       qes_cup[jmin])
    pwd_j = zd[jmin] * jnp.minimum(0.0, qcd_j - qrcd_j)
    qcd_j = qrcd_j
    pwev0 = pwd_j
    bu0 = dz0 * dh_j

    # downward scan over ki = kx..1; only ki in 1..jmin-1 are active
    def step(carry, ki):
        qcd_prev, bu, pwev, broken = carry  # qcd_prev = qcd[ki+1]
        active = (ki >= 1) & (ki <= jmin - 1) & (~broken)
        dz = z_cup[ki + 1] - z_cup[ki]
        denom = zd[ki + 1] - 0.5 * dd_massdetr[ki] + dd_massentr[ki]
        will_break = active & (denom < 1.0e-8)
        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        qcd_ki = (qcd_prev * zd[ki + 1] - 0.5 * dd_massdetr[ki] * qcd_prev
                  + dd_massentr[ki] * q[ki]) / denom_safe
        dh = hcd[ki] - hes_cup[ki]
        bu_new = bu + jnp.where(active & (~will_break), dz * dh, 0.0)
        qrcd_ki = qes_cup[ki] + (1.0 / XLV) * (gamma_cup[ki] / (1.0 + gamma_cup[ki])) * dh
        dqeva = qcd_ki - qrcd_ki
        pos = dqeva > 0.0
        dqeva2 = jnp.where(pos, 0.0, dqeva)
        qrcd_ki2 = jnp.where(pos, qcd_ki, qrcd_ki)
        pwd_ki = zd[ki] * dqeva2
        qcd_final = qrcd_ki2
        use = active & (~will_break)
        pwev_new = pwev + jnp.where(use, pwd_ki, 0.0)
        broken_new = broken | will_break
        # carry qcd: if used, qcd[ki]=qcd_final; else qcd_prev passes through but
        # next lower level reads qcd[ki+1] which is the value at THIS ki's output.
        qcd_out = jnp.where(use, qcd_final, qcd_prev)
        out = (jnp.where(use, qcd_final, 0.0),
               jnp.where(use, qrcd_ki2, 0.0),
               jnp.where(use, pwd_ki, 0.0))
        return (qcd_out, bu_new, pwev_new, broken_new), out

    kis = jnp.arange(kx, 0, -1)  # kx down to 1
    init = (qcd_j, bu0, pwev0, jnp.asarray(False))
    (_, bu, pwev, broken), (qcd_v, qrcd_v, pwd_v) = jax.lax.scan(step, init, kis)
    # scan emitted in order kx..1; reverse to get index order 1..kx
    qcd_v = qcd_v[::-1]; qrcd_v = qrcd_v[::-1]; pwd_v = pwd_v[::-1]
    qcd = jnp.zeros(kx + 1, F64).at[1:].set(qcd_v)
    qrcd = jnp.zeros(kx + 1, F64).at[1:].set(qrcd_v)
    pwd = jnp.zeros(kx + 1, F64).at[1:].set(pwd_v)
    # set jmin entries
    qcd = qcd.at[jmin].set(qcd_j)
    qrcd = qrcd.at[jmin].set(qrcd_j)
    pwd = pwd.at[jmin].set(pwd_j)

    ierr = jnp.where(broken, 51, ierr_in)
    ierr = jnp.where((pwev == 0.0) & (iloop == 1), 7, ierr)
    ierr = jnp.where((bu >= 0.0) & (iloop == 1), 7, ierr)
    ierr = jnp.asarray(ierr, jnp.int32)

    # if bad on entry, return zeros
    z = jnp.zeros(kx + 1, F64)
    return (jnp.where(bad, z, qcd), jnp.where(bad, z, qrcd), jnp.where(bad, z, pwd),
            jnp.where(bad, 0.0, pwev), jnp.where(bad, 0.0, bu),
            jnp.where(bad, ierr_in, ierr).astype(jnp.int32))


def cup_up_moisture(name, ierr_in, z_cup, p_cup, kbcon, ktop, dby, xland1, q,
                    gamma_cup, zu, qes_cup, k22, qe_cup, zqexec, ccn, rho, c1d,
                    t, up_massentr, up_massdetr, kx):
    """cup_up_moisture (autoconv=1, iall=0). Upward scan k=k22+1..ktop with the
    below-LFC (k<=kbcon) and above-LFC behaviours, denom-break -> ierr=51."""
    ktf = kx
    k = _idx(kx)
    bad = ierr_in != 0
    start_level = k22
    qaver = get_cloud_bc(qe_cup, k22, 0.0, kx)
    # qc init: qc[k]=qe_cup[k] for all; qc[start_level]=qaver
    qc_init = jnp.where(k == start_level, qaver, qe_cup)
    zm = jnp.roll(z_cup, 1)
    dz = z_cup - zm
    c0_arr = jnp.where(t < 273.15, 0.004 * jnp.exp(0.07 * (t - 273.15)), 0.004)
    zum = jnp.roll(zu, 1); udm = jnp.roll(up_massdetr, 1); uem = jnp.roll(up_massentr, 1)
    qm = jnp.roll(q, 1)

    def step(carry, kk):
        qc_prev, pwav, psum, broken = carry  # qc_prev = qc[kk-1]
        below = (kk >= k22 + 1) & (kk <= kbcon)
        above = (kk >= kbcon + 1) & (kk <= ktop) & (~broken)
        c0 = c0_arr[kk]
        # below-LFC qc recurrence (no denom guard in ref)
        denom_b = (zu[kk - 1] - 0.5 * up_massdetr[kk - 1] + up_massentr[kk - 1])
        denom_b_safe = jnp.where(denom_b == 0.0, 1.0, denom_b)
        qc_b = (qc_prev * zu[kk - 1] - 0.5 * up_massdetr[kk - 1] * qc_prev
                + up_massentr[kk - 1] * q[kk - 1]) / denom_b_safe
        qrch_b = qes_cup[kk] + (1.0 / XLV) * (gamma_cup[kk] / (1.0 + gamma_cup[kk])) * dby[kk]
        qrch_b = jnp.where(kk < kbcon, qc_b, qrch_b)
        do_b = below & (qc_b > qrch_b)
        qrc_b = jnp.where(do_b, (qc_b - qrch_b) / (1.0 + c0 * dz[kk]), 0.0)
        pw_b = jnp.where(do_b, c0 * dz[kk] * qrc_b * zu[kk], 0.0)
        qc_b_final = jnp.where(do_b, qrch_b + qrc_b, qc_b)
        clw_b = jnp.where(do_b, qrc_b, 0.0)

        # above-LFC
        denom_a = zu[kk - 1] - 0.5 * up_massdetr[kk - 1] + up_massentr[kk - 1]
        will_break = above & (denom_a < 1.0e-8)
        denom_a_safe = jnp.where(denom_a == 0.0, 1.0, denom_a)
        qrch_a = qes_cup[kk] + (1.0 / XLV) * (gamma_cup[kk] / (1.0 + gamma_cup[kk])) * dby[kk]
        qc_a0 = (qc_prev * zu[kk - 1] - 0.5 * up_massdetr[kk - 1] * qc_prev
                 + up_massentr[kk - 1] * q[kk - 1]) / denom_a_safe
        qc_a0 = jnp.where(qc_a0 <= qrch_a, qrch_a, qc_a0)
        clw_a = jnp.maximum(0.0, qc_a0 - qrch_a)
        qrc_a = (qc_a0 - qrch_a) / (1.0 + (c1d[kk] + c0) * dz[kk])
        pw_a = c0 * dz[kk] * qrc_a * zu[kk]
        neg = qrc_a < 0
        qrc_a = jnp.where(neg, 0.0, qrc_a)
        pw_a = jnp.where(neg, 0.0, pw_a)
        qc_a_final = qrc_a + qrch_a
        use_a = above & (~will_break)

        # combine: which branch is active this level?
        qc_out = jnp.where(do_b, qc_b_final,
                           jnp.where(below, qc_b,  # below but qc<=qrch: qc=qc_b (no rain)
                                     jnp.where(use_a, qc_a_final, qc_prev)))
        qrc_out = jnp.where(do_b, qrc_b, jnp.where(use_a, qrc_a, 0.0))
        pw_out = jnp.where(do_b, pw_b, jnp.where(use_a, pw_a, 0.0))
        clw_out = jnp.where(do_b, clw_b, jnp.where(use_a, clw_a, 0.0))
        pwav_new = pwav + jnp.where(use_a, pw_a, 0.0)
        psum_new = psum + jnp.where(use_a, clw_a * zu[kk] * dz[kk], 0.0)
        broken_new = broken | will_break
        # carry qc_prev for next level: if this level active (below/above and not broken)
        active_any = below | use_a
        qc_carry = jnp.where(active_any, qc_out, qc_prev)
        # but if broken at this level, qc stays qc_prev and propagates
        qc_carry = jnp.where(will_break, qc_prev, qc_carry)
        emit_active = below | use_a
        return (qc_carry, pwav_new, psum_new, broken_new), (qc_out, qrc_out, pw_out, clw_out, emit_active)

    kks = jnp.arange(1, kx + 1)
    # initial qc_prev must be qc_init[k22] for the first active level (k22+1 reads qc[k22])
    init = (qc_init[start_level], jnp.zeros((), F64), jnp.zeros((), F64), jnp.asarray(False))
    (_, pwav, psum, broken), (qc_v, qrc_v, pw_v, clw_v, act_v) = jax.lax.scan(step, init, kks)
    # base qc = qc_init; overwrite where emitted active
    qc = qc_init.at[1:].set(jnp.where(act_v, qc_v, qc_init[1:]))
    qrc = jnp.zeros(kx + 1, F64).at[1:].set(jnp.where(act_v, qrc_v, 0.0))
    pw = jnp.zeros(kx + 1, F64).at[1:].set(jnp.where(act_v, pw_v, 0.0))
    clw_all = jnp.zeros(kx + 1, F64).at[1:].set(jnp.where(act_v, clw_v, 0.0))
    # final: qc[k]=qc[k]-qrc[k] for k in k22+1..ktop
    drop = (k >= k22 + 1) & (k <= ktop)
    qc = jnp.where(drop, qc - qrc, qc)

    ierr = jnp.where(broken, 51, ierr_in).astype(jnp.int32)
    z = jnp.zeros(kx + 1, F64)
    if_bad = lambda a: jnp.where(bad, z, a)
    return (if_bad(qc), if_bad(qrc), if_bad(pw), jnp.where(bad, 0.0, pwav),
            if_bad(clw_all), jnp.where(bad, 0.0, psum), jnp.where(bad, 0.0, 0.0),
            jnp.where(bad, ierr_in, ierr).astype(jnp.int32))


def cup_dd_edt(us, vs, z, ktop, kbcon, p, pwav, pw, ccn, pwev, edtmax, edtmin,
               psum2, psumh, rho, ierr_in, kx):
    """cup_dd_edt (aeroevap=1). Returns (edt, edtc)."""
    ktf = kx
    k = _idx(kx)
    bad = ierr_in != 0
    usp = jnp.roll(us, -1); vsp = jnp.roll(vs, -1); zp = jnp.roll(z, -1); pp = jnp.roll(p, -1)
    # vws sum over kk in 1..ktf-1 with kbcon<=kk<=min(ktop,ktf)
    rng = (k >= 1) & (k <= ktf - 1) & (k >= kbcon) & (k <= jnp.minimum(ktop, ktf))
    term = (jnp.abs((usp - us) / (zp - z)) + jnp.abs((vsp - vs) / (zp - z))) * (p - pp)
    vws = jnp.sum(jnp.where(rng, term, 0.0))
    rng_sdp = (k >= 1) & (k <= ktf - 1) & (k >= kbcon) & (k <= jnp.minimum(ktop, ktf))
    sdp = jnp.sum(jnp.where(rng_sdp, p - pp, 0.0))
    vshear = 1.0e3 * vws / jnp.where(sdp == 0.0, 1.0, sdp)
    pef = (1.591 - 0.639 * vshear + 0.0953 * (vshear ** 2) - 0.00496 * (vshear ** 3))
    pef = jnp.clip(pef, 0.1, 0.9)
    zkbc = z[kbcon] * 3.281e-3
    prezk_poly = (0.96729352 + zkbc * (-0.70034167 + zkbc * (0.162179896 + zkbc
                  * (-1.2569798e-2 + zkbc * (4.2772e-4 - zkbc * 5.44e-6)))))
    prezk = jnp.where(zkbc > 3.0, prezk_poly, 0.02)
    prezk = jnp.where(zkbc > 25, 2.4, prezk)
    pefb = 1.0 / (1.0 + prezk)
    pefb = jnp.clip(pefb, 0.1, 0.9)
    edt = 1.0 - 0.5 * (pefb + pef)
    einc = 0.2 * edt
    edtc = edt - einc
    edtc = -edtc * pwav / jnp.where(pwev == 0.0, 1.0, pwev)
    edtc = jnp.clip(edtc, edtmin, edtmax)
    return jnp.where(bad, 0.0, edt), jnp.where(bad, 0.0, edtc)


def cup_forcing_ens_3d(closure_n_in, xland, aa0, aa1, xaa0, mbdt, dtime, ierr_in,
                       ierr2, ierr3, mconv, p_cup, ktop, omeg, zd, k22, zu,
                       pr_ens, edt, kbcon, ichoice, imid, axx, tau_ecmwf,
                       aa1_bl, dicycle, kx):
    """cup_forcing_ens_3d (rand_clos=0). Returns (xf_ens[1..16], closure_n, xf_dicycle).

    The 16-member ensemble is a length-17 array (index 0 unused). ichoice is a
    static Python int."""
    bad = ierr_in != 0
    k = _idx(kx)
    kloc = _maxloc1(zu, kx)
    ens_adj = 1.0
    xff0 = (aa1 - aa0) / dtime
    # build xff_ens3 as a length-17 vector, index 0 unused
    xff = [0.0] * (MAXENS3 + 1)
    base = jnp.maximum(0.0, (aa1 - aa0) / dtime)
    # omega-based 4,5,6,14
    rng_o = (k >= kbcon - 1) & (k <= kbcon + 1)
    zu_pos = zu > 0.0
    den_o = jnp.maximum(0.5, (1.0 - edt * zd / jnp.where(zu == 0.0, 1.0, zu)))
    xomg_terms = jnp.where(rng_o & zu_pos, -omeg / 9.81 / den_o, 0.0)
    xomg = jnp.sum(xomg_terms)
    kk = jnp.sum(jnp.where(rng_o & zu_pos, 1, 0))
    xff4 = jnp.where(kk > 0, xomg / kk.astype(F64), 0.0)
    xff4 = BETAJB * xff4
    # mconv-based 7,8,9,15
    den_m = jnp.maximum(0.5, (1.0 - edt * zd[kbcon] / zu[kloc]))
    xff7 = mconv / den_m
    # tau-based 10,11,12,13
    xff10 = aa1 / tau_ecmwf
    xff_dicycle = jnp.where(dicycle == 1, jnp.maximum(0.0, aa1_bl / tau_ecmwf), 0.0)
    # ichoice==0 + xff0<0 zeroing of stability/tau closures + closure_n=12
    zero_stab = (ichoice == 0) & (xff0 < 0.0)
    closure_n = jnp.where(zero_stab, 12.0, closure_n_in)
    b1 = jnp.where(zero_stab, 0.0, base)   # members 1,2,3,16
    t10 = jnp.where(zero_stab, 0.0, xff10)  # members 10,11,12,13
    # xk
    xk = (xaa0 - aa1) / mbdt
    xk = jnp.where((xk <= 0.0) & (xk > -0.01 * mbdt), -0.01 * mbdt, xk)
    xk = jnp.where((xk > 0.0) & (xk < 1.0e-2), 1.0e-2, xk)
    xk_neg = xk < 0.0
    xk_safe = jnp.where(xk == 0.0, 1.0, xk)

    xf = [0.0] * (MAXENS3 + 1)
    # stability 1,2,3,16
    def stab(xff_member):
        return jnp.where(xk_neg & (xff_member > 0), jnp.maximum(0.0, -xff_member / xk_safe), 0.0)
    xf[1] = stab(b1); xf[2] = stab(b1); xf[3] = stab(b1); xf[16] = stab(b1)
    # 4,5,6,14
    xf4 = jnp.maximum(0.0, jnp.where(xff4 < 0.0, 0.0, xff4))
    xf[4] = xf4; xf[5] = xf4; xf[6] = xf4
    xf[14] = jnp.maximum(0.0, BETAJB * xff4)
    # 7,8,9 with pr_ens, 15 with pr_ens
    a7 = jnp.maximum(1.0e-5, pr_ens[7]); xf[7] = jnp.maximum(0.0, xff7 / a7)
    a8 = jnp.maximum(1.0e-5, pr_ens[8]); xf[8] = jnp.maximum(0.0, xff7 / a8)
    a9 = jnp.maximum(1.0e-5, pr_ens[9]); xf[9] = jnp.maximum(0.0, xff7 / a9)
    a15 = jnp.maximum(1.0e-3, pr_ens[15]); xf[15] = jnp.maximum(0.0, xff7 / a15)
    # tau 10,11,12,13
    def tau(t10v):
        return jnp.where(xk_neg, jnp.maximum(0.0, -t10v / xk_safe), 0.0)
    xf[10] = tau(t10); xf[11] = tau(t10); xf[12] = tau(t10); xf[13] = tau(t10)
    xf_dicycle = jnp.where(xk_neg, jnp.maximum(0.0, -xff_dicycle / xk_safe), 0.0)

    xf_ens = jnp.stack([jnp.zeros((), F64)] + [jnp.asarray(xf[n], F64) for n in range(1, MAXENS3 + 1)])
    if ichoice >= 1:
        xf_ens = jnp.where(jnp.arange(MAXENS3 + 1) >= 1, xf_ens[ichoice], xf_ens)

    z17 = jnp.zeros(MAXENS3 + 1, F64)
    return (jnp.where(bad, z17, xf_ens), jnp.where(bad, closure_n_in, closure_n),
            jnp.where(bad, 0.0, xf_dicycle))


def cup_output_ens_3d(xf_ens, ierr_in, dellat, dellaq, dellaqc, zu, pw, ktop,
                      edt, pwd, name, p_cup, pr_ens, sig, closure_n, xland1,
                      xmbm_in, xmbs_in, ichoice, imid, dicycle, xf_dicycle, kx):
    """cup_output_ens_3d (imid=0 deep). Returns (outtem,outq,outqc,pre,xmb,ierr)."""
    k = _idx(kx)
    bad = ierr_in != 0
    nens = jnp.arange(MAXENS3 + 1)
    xf2 = jnp.where((nens >= 1) & (pr_ens <= 0.0), 0.0, xf_ens)
    xmb_ave = jnp.sum(jnp.where(nens >= 1, xf2, 0.0)) / float(MAXENS3)
    xmb_ave = jnp.where(dicycle == 2, jnp.maximum(0.0, xmb_ave - jnp.maximum(jnp.maximum(0.0, xmbm_in), xmbs_in)), xmb_ave)
    xmb_ave = jnp.where(dicycle == 1, jnp.maximum(0.0, jnp.minimum(xmb_ave, xmb_ave - xf_dicycle)), xmb_ave)
    clos_wei = 16.0 / jnp.maximum(1.0, closure_n)
    xmb_ave = jnp.minimum(xmb_ave, 100.0)
    xmb = clos_wei * sig * xmb_ave
    too_small = xmb < 1.0e-16
    ierr = jnp.where(bad, ierr_in, jnp.where(too_small, 19, ierr_in)).astype(jnp.int32)

    pwtot = jnp.sum(jnp.where((k >= 1) & (k <= ktop), pw, 0.0))
    pcp = jnp.roll(p_cup, -1)  # p_cup[k+1]
    dp = 100.0 * (p_cup - pcp) / G
    dtt = dellat; dtq = dellaq
    dtpwd0 = -pwd * edt
    dtqc0 = dellaqc * dp - dtpwd0
    neg = dtqc0 < 0.0
    dtpwd = jnp.where(neg, dtpwd0 - dellaqc * dp, 0.0)
    dtqc = jnp.where(neg, 0.0, dtqc0 / jnp.where(dp == 0.0, 1.0, dp))
    rng = (k >= 1) & (k <= ktop)
    outtem = jnp.where(rng, xmb * dtt, 0.0)
    outq = jnp.where(rng, xmb * dtq, 0.0)
    outqc = jnp.where(rng, xmb * dtqc, 0.0)
    # ref: pre = pre - xmb*dtpwd (loop), then pre = -pre + xmb*pwtot
    pre_loop = jnp.sum(jnp.where(rng, -xmb * dtpwd, 0.0))
    pre = -pre_loop + xmb * pwtot

    z = jnp.zeros(kx + 1, F64)
    valid = (~bad) & (~too_small)
    return (jnp.where(valid, outtem, z), jnp.where(valid, outq, z),
            jnp.where(valid, outqc, z), jnp.where(valid, pre, 0.0),
            jnp.where(valid, xmb, 0.0), ierr)


def neg_check(name, dt, q, outq, outt, outu, outv, outqc, pret, kx):
    """neg_check (heating-rate cap). Returns (outq,outt,outu,outv,outqc,pret)."""
    k = _idx(kx)
    thresh = 148.01 if name == 'shallow' else 300.01
    names = 2.0 if name == 'shallow' else 1.0
    rng = (k >= 1) & (k <= kx)
    qmem = outt * 86400.0
    hi = (qmem > thresh) & rng
    lo = (qmem < -0.5 * thresh * names) & rng
    qmem_safe = jnp.where(qmem == 0.0, 1.0, qmem)
    f_hi = jnp.where(hi, thresh / qmem_safe, 1.0)
    f_lo = jnp.where(lo, -0.5 * names * thresh / qmem_safe, 1.0)
    qmemf = jnp.minimum(1.0, jnp.minimum(jnp.min(f_hi), jnp.min(f_lo)))
    return (outq * qmemf, outt * qmemf, outu * qmemf, outv * qmemf,
            outqc * qmemf, pret * qmemf)


# ---------------------------------------------------------------------------
# Coupled recurrence helpers (upward / downward) used by cup_gf.
# ---------------------------------------------------------------------------


def _hc_uc_vc_hco_recurrence(start_level, ktop, hkb, hkbo, he_env, heo_env,
                             hes_cup, heso_cup, u_cup, v_cup, us, vs, zu, zuo,
                             up_massentr, up_massdetr, up_massentro, up_massdetro,
                             up_massentru, up_massdetru, zo_cup, pgcon, kx):
    """Upward recurrence start_level+1..ktop for hc,uc,vc,hco + dby/dbyo/dbyt.
    Uses env he[k-1]/heo[k-1] (NOT *_cup) for entrainment, matching the
    reference (lines 1060/1071). denom-break -> ierr=51."""
    k = _idx(kx)

    def step(carry, kk):
        hc_p, uc_p, vc_p, hco_p, dbyt_p, broken = carry
        active = (kk >= start_level + 1) & (kk <= ktop) & (~broken)
        denom = zuo[kk - 1] - 0.5 * up_massdetro[kk - 1] + up_massentro[kk - 1]
        will_break = active & (denom < 1.0e-8)
        use = active & (~will_break)
        d_h = zu[kk - 1] - 0.5 * up_massdetr[kk - 1] + up_massentr[kk - 1]
        d_h = jnp.where(d_h == 0.0, 1.0, d_h)
        d_u = zu[kk - 1] - 0.5 * up_massdetru[kk - 1] + up_massentru[kk - 1]
        d_u = jnp.where(d_u == 0.0, 1.0, d_u)
        d_ho = zuo[kk - 1] - 0.5 * up_massdetro[kk - 1] + up_massentro[kk - 1]
        d_ho = jnp.where(d_ho == 0.0, 1.0, d_ho)
        hc_k = (hc_p * zu[kk - 1] - 0.5 * up_massdetr[kk - 1] * hc_p
                + up_massentr[kk - 1] * he_env[kk - 1]) / d_h
        uc_k = (uc_p * zu[kk - 1] - 0.5 * up_massdetru[kk - 1] * uc_p
                + up_massentru[kk - 1] * us[kk - 1]
                - pgcon * 0.5 * (zu[kk] + zu[kk - 1]) * (u_cup[kk] - u_cup[kk - 1])) / d_u
        vc_k = (vc_p * zu[kk - 1] - 0.5 * up_massdetru[kk - 1] * vc_p
                + up_massentru[kk - 1] * vs[kk - 1]
                - pgcon * 0.5 * (zu[kk] + zu[kk - 1]) * (v_cup[kk] - v_cup[kk - 1])) / d_u
        dby_k = hc_k - hes_cup[kk]
        hco_k = (hco_p * zuo[kk - 1] - 0.5 * up_massdetro[kk - 1] * hco_p
                 + up_massentro[kk - 1] * heo_env[kk - 1]) / d_ho
        dbyo_k = hco_k - heso_cup[kk]
        dz = zo_cup[kk + 1] - zo_cup[kk]
        dbyt_k = dbyt_p + jnp.where(use, dbyo_k * dz, 0.0)
        hc_c = jnp.where(use, hc_k, hc_p)
        uc_c = jnp.where(use, uc_k, uc_p)
        vc_c = jnp.where(use, vc_k, vc_p)
        hco_c = jnp.where(use, hco_k, hco_p)
        broken_new = broken | will_break
        emit = (jnp.where(use, hc_k, 0.0), jnp.where(use, uc_k, 0.0),
                jnp.where(use, vc_k, 0.0), jnp.where(use, dby_k, 0.0),
                jnp.where(use, hco_k, 0.0), jnp.where(use, dbyo_k, 0.0),
                dbyt_k, use)
        return (hc_c, uc_c, vc_c, hco_c, dbyt_k, broken_new), emit

    kks = jnp.arange(1, kx + 1)
    init = (hkb, u_cup[start_level], v_cup[start_level], hkbo,
            jnp.zeros((), F64), jnp.asarray(False))
    (_, _, _, _, _, broken), (hc_v, uc_v, vc_v, dby_v, hco_v, dbyo_v, dbyt_v, emit_v) = \
        jax.lax.scan(step, init, kks)
    return (hc_v, uc_v, vc_v, dby_v, hco_v, dbyo_v, dbyt_v, emit_v, broken)


# ---------------------------------------------------------------------------
# Deep-convection driver cup_gf (imid=0 path used by cu_physics=3).
# ---------------------------------------------------------------------------


def cup_gf(dicycle, ichoice, ccn, dtime, imid, kpbl, dhdt, xland, zo, t, q, z1,
           tn, qo, po, psur, us, vs, rho, hfx, qfx, dx, mconv, omeg, csum, kx):
    """JAX traceable single-column CUP_gf (deep). Returns dict matching the
    reference cup_gf outputs. imid=0 only (cu_physics=3 compile path)."""
    ktf = kx
    kte = kx
    k = _idx(kx)
    # promote scalar inputs to traced F64 so .astype / jnp ops work under jit/vmap
    xland = jnp.asarray(xland, F64)
    z1 = jnp.asarray(z1, F64)
    psur = jnp.asarray(psur, F64)
    hfx = jnp.asarray(hfx, F64)
    qfx = jnp.asarray(qfx, F64)
    dx = jnp.asarray(dx, F64)
    mconv = jnp.asarray(mconv, F64)
    csum = jnp.asarray(csum, F64)
    dtime = jnp.asarray(dtime, F64)
    ccn = jnp.asarray(ccn, F64)
    kpbl = jnp.asarray(kpbl, jnp.int32)
    imid = jnp.asarray(imid, jnp.int32)
    dicycle = jnp.asarray(dicycle, jnp.int32)
    z = zo
    flux_tun = FLUXTUNE
    pmin = 150.0
    pgcon = 0.0
    lambau = 2.0
    # zws/ztexec/zqexec
    buo_flux = (hfx / CP + 0.608 * t[1] * qfx / XLV) / rho[1]
    zws_tmp = jnp.maximum(0.0, flux_tun * 0.41 * buo_flux * zo[2] * 9.81 / t[1])
    big = zws_tmp > TINY
    zws_tmp2 = 1.2 * zws_tmp ** 0.3333
    ztexec = jnp.where(big, jnp.maximum(flux_tun * hfx / (rho[1] * jnp.where(big, zws_tmp2, 1.0) * CP), 0.0), 0.0)
    zqexec = jnp.where(big, jnp.maximum(flux_tun * qfx / XLV / (rho[1] * jnp.where(big, zws_tmp2, 1.0)), 0.0), 0.0)
    zws = jnp.maximum(0.0, flux_tun * 0.41 * buo_flux * zo[kpbl] * 9.81 / t[kpbl])
    zws = 1.2 * zws ** 0.3333 * rho[kpbl]
    cap_maxs = 75.0
    cap_max = cap_maxs
    cap_max_increment = 20.0
    xl_int = (xland + 0.0001).astype(jnp.int32)
    is_water = (xland > 1.5) | (xland < 0.5)
    xland1 = jnp.where(is_water, 0, xl_int).astype(jnp.int32)
    cap_max = jnp.where((~is_water) & (ztexec > 0.0), cap_max + 25.0, cap_max)
    cap_max = jnp.where((~is_water) & (ztexec < 0.0), cap_max - 25.0, cap_max)
    cap_max_increment = jnp.where(is_water, 20.0, cap_max_increment)
    # entr/sig
    entr_rate0 = 7.0e-5 - jnp.minimum(20.0, csum) * 3.0e-6
    entr_rate = jnp.where(xland1 == 0, 7.0e-5, entr_rate0)
    radius = 0.2 / entr_rate
    frh0 = jnp.minimum(1.0, 3.14 * radius * radius / dx / dx)
    over = frh0 > FRH_THRESH
    frh = jnp.where(over, FRH_THRESH, frh0)
    radius = jnp.where(over, jnp.sqrt(frh * dx * dx / 3.14), radius)
    entr_rate = jnp.where(over, 0.2 / radius, entr_rate)
    sig = (1.0 - frh) ** 2
    sig_thresh = (1.0 - FRH_THRESH) ** 2
    edtmax = jnp.asarray(1.0, F64); edtmin = 0.1
    depth_min = 1000.0
    kstabm = ktf - 1
    zkbmax = 4000.0; zcutdown = 4000.0; z_detr = 1000.0

    ierr = jnp.asarray(0, jnp.int32)

    # cup_env (z=zo for deep, t/q current and tn/qo forced)
    z_e, qes, he, hes = cup_env(z, t, q, po, z1, psur, ierr, -1, kx)
    zo_e, qeso, heo, heso = cup_env(zo, tn, qo, po, z1, psur, ierr, -1, kx)
    z = z_e; zo = zo_e
    qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup = \
        cup_env_clev(t, qes, q, he, hes, z, po, psur, z1, ierr, kx)
    qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup = \
        cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, ierr, kx)
    # u_cup/v_cup
    usm = jnp.roll(us, 1); vsm = jnp.roll(vs, 1)
    u_cup = jnp.where(k == 1, us[1], jnp.where((k >= 2) & (k <= ktf), 0.5 * (usm + us), 0.0))
    v_cup = jnp.where(k == 1, vs[1], jnp.where((k >= 2) & (k <= ktf), 0.5 * (vsm + vs), 0.0))

    # kbmax (first k with zo_cup>zkbmax+z1), kdet (first k with zo_cup>z_detr+z1)
    cb = (zo_cup > zkbmax + z1) & (k >= 1) & (k <= ktf)
    kbmax = jnp.where(jnp.any(cb), jnp.argmax(cb), 1).astype(jnp.int32)
    cd_ = (zo_cup > z_detr + z1) & (k >= 1) & (k <= ktf)
    kdet = jnp.where(jnp.any(cd_), jnp.argmax(cd_), 0).astype(jnp.int32)

    # k22
    k22 = _argmax_range(heo_cup, 2, kbmax + 2, kx)
    ierr = jnp.where(k22 >= kbmax, 2, ierr).astype(jnp.int32)
    k22 = jnp.where(ierr == 2, 0, k22).astype(jnp.int32)

    # hkb/hkbo
    x_add = XLV * zqexec + CP * ztexec
    k22_safe = jnp.maximum(k22, 1)
    hkb = get_cloud_bc(he_cup, k22_safe, x_add, kx)
    hkbo = get_cloud_bc(heo_cup, k22_safe, x_add, kx)

    # cup_kbcon (iloop=1)
    k22, kbcon, hkbo, ierr = cup_kbcon(cap_max_increment, 1, k22_safe, heo_cup,
                                       heso_cup, hkbo, ierr, kbmax, po_cup, cap_max,
                                       ztexec, zqexec, z_cup, entr_rate, heo, imid, kx)
    # kstabi
    kstabi = cup_minimi(heso_cup, kbcon, kstabm, ierr, kx)

    # frh check + pmin_lev
    bad = ierr != 0
    frh2 = jnp.minimum(qo_cup[kbcon] / qeso_cup[kbcon], 1.0)
    trip231 = (~bad) & (frh2 >= RH_THRESH) & (sig <= sig_thresh)
    ierr = jnp.where(trip231, 231, ierr).astype(jnp.int32)
    bad = ierr != 0
    # pmin_lev: first k>kbcon with po[kbcon]-po[k] > pmin
    pml = (po[kbcon] - po > pmin) & (k >= kbcon + 1) & (k <= ktf)
    pmin_lev = jnp.where(jnp.any(pml), jnp.argmax(pml), 0).astype(jnp.int32)
    x_add = XLV * zqexec + CP * ztexec
    hkb = jnp.where(~bad, get_cloud_bc(he_cup, jnp.maximum(k22, 1), x_add, kx), hkb)

    # kstabi<kbcon -> kbcon=1, ierr=42
    trip42 = kstabi < kbcon
    kbcon = jnp.where(trip42, 1, kbcon).astype(jnp.int32)
    ierr = jnp.where(trip42, 42, ierr).astype(jnp.int32)
    bad = ierr != 0

    # entr_rate_2d
    entr_rate_2d = jnp.where((k >= 1) & (k <= ktf), entr_rate, 0.0)
    kbcon = jnp.where(~bad, jnp.maximum(2, kbcon), kbcon).astype(jnp.int32)
    frh_arr = jnp.minimum(qo_cup / jnp.where(qeso_cup == 0.0, 1.0, qeso_cup), 1.0)
    entr_rate_2d = jnp.where((~bad) & (k >= 1) & (k <= ktf), entr_rate * (1.3 - frh_arr), entr_rate_2d)
    start_level = k22

    # rates_up_pdf deep
    kbcon, ktop, ktopdby, ierr, zuo = rates_up_pdf(
        'deep', jnp.asarray(0, jnp.int32), ierr, po_cup, entr_rate_2d, hkbo, heo,
        heso_cup, zo_cup, xland1, kstabi, k22, kbcon, kbcon, csum, kx)
    bad = ierr != 0

    # zero zuo below k22 / above ktop; zu=zuo
    below = (k < k22) & (k22 > 1)
    above = (k >= ktop + 1) & (k <= ktf)
    zuo = jnp.where(~bad, jnp.where(below | above, 0.0, zuo), zuo)
    zu = zuo

    # lateral massflux
    cd = jnp.where((k >= 1) & (k <= ktf), 1.0e-9, 0.0)
    ktop_lat = jnp.where(bad, 0, ktop).astype(jnp.int32)
    (up_massentro, up_massdetro, up_massentr, up_massdetr,
     up_massentru, up_massdetru, cd, entr_rate_2d) = get_lateral_massflux(
        zo_cup, zuo, cd, entr_rate_2d, ktop_lat, kbcon, k22, lambau, kx, with_u=True)

    # hc/uc/vc/hco recurrence
    sl = jnp.maximum(start_level, 1)
    (hc_v, uc_v, vc_v, dby_v, hco_v, dbyo_v, dbyt_v, emit_v, hc_break) = \
        _hc_uc_vc_hco_recurrence(sl, ktop, hkb, hkbo, he, heo, hes_cup, heso_cup,
                                 u_cup, v_cup, us, vs, zu, zuo, up_massentr,
                                 up_massdetr, up_massentro, up_massdetro,
                                 up_massentru, up_massdetru, zo_cup, pgcon, kx)
    # assemble hc,uc,vc,hco,dby,dbyo with the init region (k<start_level)
    hc = jnp.where(k < sl, he_cup, 0.0)
    hco = jnp.where(k < sl, heo_cup, 0.0)
    uc = jnp.where(k <= sl, u_cup, 0.0)
    vc = jnp.where(k <= sl, v_cup, 0.0)
    hc = jnp.where(k == sl, hkb, hc)
    hco = jnp.where(k == sl, hkbo, hco)
    dby = jnp.zeros(kx + 1, F64); dbyo = jnp.zeros(kx + 1, F64)
    dbyt = jnp.zeros(kx + 1, F64)
    emit = jnp.zeros(kx + 1, bool).at[1:].set(emit_v)
    hc = jnp.where(emit, jnp.zeros(kx + 1, F64).at[1:].set(hc_v), hc)
    uc = jnp.where(emit, jnp.zeros(kx + 1, F64).at[1:].set(uc_v), uc)
    vc = jnp.where(emit, jnp.zeros(kx + 1, F64).at[1:].set(vc_v), vc)
    hco = jnp.where(emit, jnp.zeros(kx + 1, F64).at[1:].set(hco_v), hco)
    dby = jnp.zeros(kx + 1, F64).at[1:].set(dby_v)
    dbyo = jnp.zeros(kx + 1, F64).at[1:].set(dbyo_v)
    dbyt = jnp.zeros(kx + 1, F64).at[1:].set(dbyt_v)
    ierr = jnp.where((~bad) & hc_break, 51, ierr).astype(jnp.int32)
    # ktopkeep: last k in kbcon-1..ktop-1 (downward) with dbyo[k]>0 -> ktop=k+1
    keep_mask = (dbyo > 0.0) & (k >= kbcon - 1) & (k <= ktop - 1)
    kk_keep = jnp.where(keep_mask, k, -1)
    kbest = jnp.max(kk_keep)
    ktopkeep = jnp.where(kbest >= 0, kbest + 1, ktop).astype(jnp.int32)
    bad2 = ierr != 0
    ktop = jnp.where(~bad2, ktopkeep, ktop).astype(jnp.int32)
    bad = ierr != 0

    # zero arrays above ktop (k in ktop+1..ktf) when ierr==0
    ab = (~bad) & (k >= ktop + 1) & (k <= ktf)
    hc = jnp.where(ab, hes_cup, hc)
    uc = jnp.where(ab, u_cup, uc)
    vc = jnp.where(ab, v_cup, vc)
    hco = jnp.where(ab, heso_cup, hco)
    dby = jnp.where(ab, 0.0, dby)
    dbyo = jnp.where(ab, 0.0, dbyo)
    zu = jnp.where(ab, 0.0, zu)
    zuo = jnp.where(ab, 0.0, zuo)
    cd = jnp.where(ab, 0.0, cd)
    entr_rate_2d = jnp.where(ab, 0.0, entr_rate_2d)
    up_massentr = jnp.where(ab, 0.0, up_massentr)
    up_massdetr = jnp.where(ab, 0.0, up_massdetr)
    up_massentro = jnp.where(ab, 0.0, up_massentro)
    up_massdetro = jnp.where(ab, 0.0, up_massdetro)

    # ktop<kbcon+2 -> ierr=5, ktop=0
    trip5 = (~bad) & (ktop < kbcon + 2)
    ierr = jnp.where(trip5, 5, ierr).astype(jnp.int32)
    ktop = jnp.where(trip5, 0, ktop).astype(jnp.int32)
    bad = ierr != 0

    # downdraft originating level kzdown
    zktop = (zo_cup[ktop] - z1) * 0.6
    zktop = jnp.minimum(zktop + z1, zcutdown + z1)
    kzc = (zo_cup > zktop) & (k >= 1) & (k <= ktf)
    kzdown0 = jnp.where(jnp.any(kzc), jnp.argmax(kzc), 0).astype(jnp.int32)
    kzdown = jnp.where(jnp.any(kzc), jnp.minimum(kzdown0, kstabi - 1), 0).astype(jnp.int32)
    jmin = cup_minimi(heso_cup, k22, kzdown, ierr, kx)

    # jmini downdraft-origin while-loop
    kdet, jmin, ierr = _jmini_search(jmin, kdet, ktop, zo_cup, heso_cup, ierr, kx)
    bad = ierr != 0
    # jmin-1<kdet -> kdet=jmin-1; depth check
    kdet = jnp.where((~bad) & (jmin - 1 < kdet), jmin - 1, kdet).astype(jnp.int32)
    trip6 = (~bad) & ((-zo_cup[kbcon] + zo_cup[ktop]) < depth_min)
    ierr = jnp.where(trip6, 6, ierr).astype(jnp.int32)
    bad = ierr != 0

    # downdraft mass flux profile
    beta_dn = jnp.maximum(0.02, 0.05 - csum * 0.0015)
    edtmax = jnp.where((imid == 0) & (xland1 == 0), jnp.maximum(0.1, 0.4 - csum * 0.015), edtmax)
    hcdo = jnp.where((k >= 1) & (k <= ktf), heso_cup, 0.0)
    ucd = jnp.where((k >= 1) & (k <= ktf), u_cup, 0.0)
    vcd = jnp.where((k >= 1) & (k <= ktf), v_cup, 0.0)
    mentrd_rate_2d = jnp.where((k >= 1) & (k <= ktf), entr_rate, 0.0)
    cdd = jnp.where((k >= 1) & (k <= jmin), 1.0e-9, 0.0)
    cdd = jnp.where(k == jmin, 0.0, cdd)
    zdo = get_zu_zd_pdf_fim(po_cup, 0.0, "DOWN", kdet, jmin, kpbl, csum, beta_dn, kx)
    # zdo[jmin]<1e-8 -> zdo[jmin]=0; jmin-=1; if zdo[jmin]<1e-8 ierr=876
    zjmin_small = zdo[jmin] < 1.0e-8
    zdo = jnp.where((k == jmin) & zjmin_small, 0.0, zdo)
    jmin = jnp.where(zjmin_small, jmin - 1, jmin).astype(jnp.int32)
    ierr = jnp.where((~bad) & zjmin_small & (zdo[jmin] < 1.0e-8), 876, ierr).astype(jnp.int32)
    bad = ierr != 0

    (dd_massentro, dd_massdetro, dd_massentru, dd_massdetru, mentrd_rate_2d,
     cdd, c1d_dn) = _downdraft_massflux(zdo, cdd, mentrd_rate_2d, zo_cup, jmin,
                                        kbcon, ktop, lambau, entr_rate, kx)
    # dbydo/bud downward recurrence for ucd/vcd/hcdo
    (ucd, vcd, hcdo, dbydo, bud) = _downdraft_uvh(jmin, zdo, dd_massdetro,
        dd_massentro, dd_massdetru, dd_massentru, ucd, vcd, hcdo, us, vs, heo,
        hco, heso_cup, zo_cup, pgcon, kx)
    ierr = jnp.where((~bad) & (bud > 0.0), 7, ierr).astype(jnp.int32)
    bad = ierr != 0
    c1d = jnp.where((~bad) & (k >= kbcon + 1) & (k <= ktop - 1), C1, 0.0)

    # downdraft moisture
    qcdo, qrcdo, pwdo, pwevo, bu, ierr = cup_dd_moisture(
        zdo, hcdo, heso_cup, qeso_cup, qo_cup, zo_cup, dd_massentro, dd_massdetro,
        jmin, ierr, gammao_cup, qo, heo, 1, kx)
    # updraft moisture
    (qco, qrco, pwo, pwavo, clw_all, psum, psumh, ierr) = cup_up_moisture(
        'deep', ierr, zo_cup, p_cup, kbcon, ktop, dbyo, xland1, qo, gammao_cup,
        zuo, qeso_cup, k22, qo_cup, zqexec, ccn, rho, c1d, tn_cup, up_massentr,
        up_massdetr, kx)
    bad = ierr != 0
    # cupclw/cnvwt
    dp12 = 100.0 * (po_cup[1] - po_cup[2])
    cupclw = jnp.where((~bad) & (k >= 2) & (k <= ktop), qrco, 0.0)
    cnvwt = jnp.where((~bad) & (k >= 2) & (k <= ktop), zuo * cupclw * G / jnp.where(dp12 == 0.0, 1.0, dp12), 0.0)

    # work functions
    aa0 = cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, kx)
    aa1 = cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, kx)
    ierr = jnp.where((~bad) & (aa1 == 0.0), 17, ierr).astype(jnp.int32)
    bad = ierr != 0

    # dicycle (iversion=1 ecmwf)
    wmean = 7.0
    tau_ecmwf = (zo_cup[ktopdby] - zo_cup[kbcon]) / wmean
    tau_ecmwf = tau_ecmwf * (1.0061 + 1.23e-2 * (dx / 1000.0))
    tau_ecmwf = jnp.where(bad, 1.0, tau_ecmwf)
    t_star = 4.0
    aa1_bl = jnp.asarray(0.0, F64)
    if True:  # dicycle==1 always for cu=3
        umean = 2.0 + jnp.sqrt(2.0 * (us[1] ** 2 + vs[1] ** 2 + us[kbcon] ** 2 + vs[kbcon] ** 2))
        tau_bl = jnp.where(xland1 == 0, (zo_cup[kbcon] - z1) / umean,
                           (zo_cup[ktopdby] - zo_cup[kbcon]) / wmean)
        aa1_bl_raw = cup_up_aa1bl_full(zo, t, tn, q, qo, dtime, kbcon, ierr, kx)
        zcheck = zo_cup[kbcon] - z1 > zo[jnp.minimum(kte, kpbl + 1)]
        aa1_bl = jnp.where((dicycle == 1) & (~bad),
                           jnp.where(zcheck, 0.0, jnp.maximum(0.0, aa1_bl_raw / t_star * tau_bl)),
                           0.0)
    axx = aa1

    # edt
    edt, edtc = cup_dd_edt(us, vs, zo, ktop, kbcon, po, pwavo, pwo, ccn, pwevo,
                           edtmax, edtmin, psum, psumh, rho, ierr, kx)
    edto = jnp.where(~bad, edtc, 0.0)

    # dellah / dellaq / dellat budgets + dellu/dellv
    (dellah, dellaq, dellat, dellaqc, dellu, dellv) = _dellas_deep(
        ierr, po_cup, zuo, zdo, hco, heo_cup, qco, qo_cup, qcdo, hcdo, qrco, pwo,
        pwdo, uc, u_cup, ucd, vc, v_cup, vcd, c1d, zo_cup, up_massdetro, edto,
        ktop, kx)
    bad = ierr != 0

    # x-profiles for static control
    mbdt = 0.1
    xhe = jnp.where((~bad) & (k >= 1) & (k <= ktf), dellah * mbdt + heo, 0.0)
    xq = jnp.where((~bad) & (k >= 1) & (k <= ktf), jnp.maximum(1.0e-16, dellaq * mbdt + qo), 0.0)
    dellat = jnp.where((~bad) & (k >= 1) & (k <= ktf), (1.0 / CP) * (dellah - XLV * dellaq), dellat)
    xt = jnp.where((~bad) & (k >= 1) & (k <= ktf), jnp.maximum(190.0, dellat * mbdt + tn), 0.0)
    # ktf-level overrides
    xhe = jnp.where((~bad) & (k == ktf), heo[ktf], xhe)
    xq = jnp.where((~bad) & (k == ktf), qo[ktf], xq)
    xt = jnp.where((~bad) & (k == ktf), tn[ktf], xt)
    xz = zo
    xz, xqes, xhe2, xhes = cup_env(xz, xt, xq, po, z1, psur, ierr, -1, kx)
    xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, _xp, _xg, xt_cup = \
        cup_env_clev(xt, xqes, xq, xhe2, xhes, xz, po, psur, z1, ierr, kx)
    # xhc recurrence start_level+1..ktop with up_massdetro/entro
    x_add = XLV * zqexec + CP * ztexec
    xhkb = get_cloud_bc(xhe_cup, jnp.maximum(k22, 1), x_add, kx)
    xzu = zuo
    xhc, xdby = _xhc_recurrence(sl, ktop, xhkb, xhe_cup, xhes_cup, xzu,
                               up_massdetro, up_massentro, xhe, kx)
    xhc = jnp.where((~bad) & (k >= ktop + 1) & (k <= ktf), xhes_cup, xhc)
    xdby = jnp.where((~bad) & (k >= ktop + 1) & (k <= ktf), 0.0, xdby)
    xaa0 = cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon, ktop, ierr, kx)
    xaa0_ens = jnp.where(~bad, xaa0, 0.0)

    # pr_ens accumulation
    pr_sum = jnp.sum(jnp.where((~bad) & (k >= 1) & (k <= ktop), pwo + edto * pwdo, 0.0))
    pr_ens = jnp.where(jnp.arange(MAXENS3 + 1) >= 1, pr_sum, 0.0)
    trip18 = (~bad) & (pr_ens[7] < 1.0e-6)
    ierr = jnp.where(trip18, 18, ierr).astype(jnp.int32)
    pr_ens = jnp.where(trip18, 0.0, pr_ens)
    pr_ens = jnp.where((jnp.arange(MAXENS3 + 1) >= 1) & (pr_ens < 1.0e-5), 0.0, pr_ens)
    bad = ierr != 0

    # large-scale forcing ierr2/ierr3
    k22x = cup_maximi(heo_cup, 2, kbmax, ierr, kx)
    _kx2, _kbx2, _h2, ierr2 = cup_kbcon(cap_max_increment, 2, k22x, heo_cup,
                                        heso_cup, hkbo, ierr, kbmax, po_cup, cap_max,
                                        ztexec, zqexec, z_cup, entr_rate, heo, imid, kx)
    _kx3, _kbx3, _h3, ierr3 = cup_kbcon(cap_max_increment, 3, k22x, heo_cup,
                                        heso_cup, hkbo, ierr, kbmax, po_cup, cap_max,
                                        ztexec, zqexec, z_cup, entr_rate, heo, imid, kx)
    ierr2 = jnp.where(bad, ierr, ierr2).astype(jnp.int32)
    ierr3 = jnp.where(bad, ierr, ierr3).astype(jnp.int32)

    # mconv recompute
    qop = jnp.roll(qo_cup, -1)
    mconv_d = jnp.sum(jnp.where((~bad) & (k >= 1) & (k <= ktop), omeg * (qop - qo_cup) / G, 0.0))

    xf_ens, closure_n, xf_dicycle = cup_forcing_ens_3d(
        16.0, xland1, aa0, aa1, xaa0_ens, mbdt, dtime, ierr, ierr2, ierr3,
        mconv_d, po_cup, ktop, omeg, zdo, k22, zuo, pr_ens, edto, kbcon, ichoice,
        imid, axx, tau_ecmwf, aa1_bl, dicycle, kx)

    outt, outq, outqc, pre, xmb, ierr = cup_output_ens_3d(
        xf_ens, ierr, dellat, dellaq, dellaqc, zuo, pwo, ktop, edto, pwdo, 'deep',
        po_cup, pr_ens, sig, closure_n, xland1, 0.0, 0.0, ichoice, imid, dicycle,
        xf_dicycle, kx)
    bad = ierr != 0

    # outu/outv + final cleanup
    outu = jnp.zeros(kx + 1, F64); outv = jnp.zeros(kx + 1, F64)
    pre_pos = (~bad) & (pre > 0.0)
    pre = jnp.where(pre_pos, jnp.maximum(pre, 0.0), pre)
    xmb_out = jnp.where(pre_pos, xmb, 0.0)
    outu = jnp.where(pre_pos & (k >= 1) & (k <= ktop), dellu * xmb, 0.0)
    outv = jnp.where(pre_pos & (k >= 1) & (k <= ktop), dellv * xmb, 0.0)
    # ierr!=0 or pre==0 -> zero outputs, ktop=0
    zero_out = bad | (pre == 0.0)
    z0 = jnp.zeros(kx + 1, F64)
    outt = jnp.where(zero_out, z0, outt)
    outq = jnp.where(zero_out, z0, outq)
    outqc = jnp.where(zero_out, z0, outqc)
    outu = jnp.where(zero_out, z0, outu)
    outv = jnp.where(zero_out, z0, outv)
    ktop_final = jnp.where(zero_out, 0, ktop).astype(jnp.int32)

    # KE dissipation heating (ierr==0)
    dp_ke = (po_cup - jnp.roll(po_cup, -1)) * 100.0
    rng_ke = (~bad) & (k >= 1) & (k <= ktop)
    dts = jnp.sum(jnp.where(rng_ke, -(outu * us + outv * vs) * dp_ke / G, 0.0))
    fpi = jnp.sum(jnp.where(rng_ke, jnp.sqrt(outu * outu + outv * outv) * dp_ke, 0.0))
    fp = jnp.where((fpi > 0.0) & rng_ke, jnp.sqrt(outu * outu + outv * outv) / jnp.where(fpi == 0.0, 1.0, fpi), 0.0)
    outt = jnp.where((fpi > 0.0) & rng_ke, outt + fp * dts * G / CP, outt)

    return {
        'outt': outt, 'outq': outq, 'outqc': outqc, 'outu': outu, 'outv': outv,
        'cupclw': cupclw, 'pre': pre, 'kbcon': kbcon, 'ktop': ktop_final,
        'k22': k22, 'ierr': ierr, 'xmb_out': xmb_out,
    }


def _jmini_search(jmin_in, kdet_in, ktop, zo_cup, heso_cup, ierr_in, kx):
    """jmini downdraft-origin while-loop (ref lines 1102-1128). Bounded
    lax.while_loop over jmini decrements. Returns (kdet, jmin, ierr)."""
    k = _idx(kx)
    bad0 = ierr_in != 0
    zp = jnp.roll(zo_cup, -1)  # zo_cup[k+1]
    dz_all = zp - zo_cup       # dz[k] = zo_cup[k+1]-zo_cup[k]

    def cond(state):
        jmini, kdet, ierr, keep_going = state
        return keep_going & (ierr == 0)

    def body(state):
        jmini, kdet, ierr, _keep = state
        kdet = jnp.where(jmini - 1 < kdet, jmini - 1, kdet).astype(jnp.int32)
        jmini = jnp.where(jmini >= ktop - 1, ktop - 2, jmini).astype(jnp.int32)
        ki = jmini
        # inner: k = ki-1 .. 1; hcdo[k]=heso_cup[jmini]; dh += dz[k]*(heso_cup[jmini]-heso_cup[k])
        # cumulative from k=ki-1 downward. find FIRST k where running dh>0.
        contrib = dz_all * (heso_cup[jmini] - heso_cup)
        # we need cumulative sum starting at k=ki-1 going DOWN to 1.
        # build masked contrib for k in 1..ki-1, then reverse-cumsum from ki-1.
        in_inner = (k >= 1) & (k <= ki - 1)
        c = jnp.where(in_inner, contrib, 0.0)
        # running sum from high k to low k: reverse cumsum
        rev = c[::-1]
        rev_cumsum = jnp.cumsum(rev)
        dh_running = rev_cumsum[::-1]  # dh_running[k] = sum_{j=k}^{ki-1} contrib[j]
        crossing = (dh_running > 0.0) & in_inner
        found = jnp.any(crossing)
        # first crossing scanning DOWN from ki-1 = largest k with crossing
        kk = jnp.where(crossing, k, -1)
        # but we want the first one encountered going downward = LARGEST k
        kcross = jnp.max(kk)
        # if found: jmini -=1; keep_going if jmini-1>5 else ierr=9
        jmini_new = jnp.where(found, jmini - 1, jmini).astype(jnp.int32)
        keep_new = found & (jmini_new > 5)
        ierr_new = jnp.where(found & (jmini_new <= 5), 9, ierr).astype(jnp.int32)
        return (jmini_new, kdet, ierr_new, keep_new)

    init = (jnp.asarray(jmin_in, jnp.int32), jnp.asarray(kdet_in, jnp.int32),
            jnp.asarray(ierr_in, jnp.int32), jnp.asarray(~bad0))
    jmini, kdet, ierr, _ = jax.lax.while_loop(cond, body, init)
    jmin = jmini
    ierr = jnp.where((~bad0) & (jmini <= 5), 4, ierr).astype(jnp.int32)
    # if bad on entry, return originals
    return (jnp.where(bad0, kdet_in, kdet).astype(jnp.int32),
            jnp.where(bad0, jmin_in, jmin).astype(jnp.int32),
            jnp.where(bad0, ierr_in, ierr).astype(jnp.int32))


def _downdraft_massflux(zdo, cdd_in, mentrd_in, zo_cup, jmin, kbcon, ktop,
                        lambau, entr_rate, kx):
    """Downdraft mass-flux profile (ref 1156-1186). Two ki-loops (each touches
    index ki once over disjoint ranges) + dd_massentru/detru. Vectorized by ki.
    Returns (dd_massentro, dd_massdetro, dd_massentru, dd_massdetru,
    mentrd_rate_2d, cdd, c1d_dn)."""
    k = _idx(kx)
    mlzd = _maxloc1(zdo, kx)
    zdp = jnp.roll(zdo, -1)       # zdo[ki+1]
    zop = jnp.roll(zo_cup, -1)
    dzo = zop - zo_cup           # dzo at ki
    cdd = cdd_in
    mentrd = mentrd_in
    dd_massdetro = jnp.zeros(kx + 1, F64)
    dd_massentro = jnp.zeros(kx + 1, F64)

    # ---- loop 1: ki in mlzd .. jmin (ref: range(jmin, mlzd-1, -1)) ----
    rng1 = (k >= mlzd) & (k <= jmin)
    det1 = cdd * dzo * zdp
    ent1 = zdo - zdp + det1
    neg1 = ent1 < 0.0
    ent1f = jnp.where(neg1, 0.0, ent1)
    det1f = jnp.where(neg1, zdp - zdo, det1)
    cdd1 = jnp.where(neg1 & (zdp > 0.0), det1f / (dzo * zdp), cdd)
    ment1 = jnp.where(zdp > 0.0, ent1f / (dzo * zdp), mentrd)
    dd_massdetro = jnp.where(rng1, det1f, dd_massdetro)
    dd_massentro = jnp.where(rng1, ent1f, dd_massentro)
    cdd = jnp.where(rng1, cdd1, cdd)
    mentrd = jnp.where(rng1, ment1, mentrd)
    # mentrd_rate_2d[1]=0 (ref line 1169)
    mentrd = jnp.where(k == 1, 0.0, mentrd)

    # ---- loop 2: ki in 1 .. mlzd-1 ----
    rng2 = (k >= 1) & (k <= mlzd - 1)
    ent2 = mentrd * dzo * zdp
    det2 = zdp + ent2 - zdo
    neg2 = det2 < 0.0
    det2f = jnp.where(neg2, 0.0, det2)
    ent2f = jnp.where(neg2, zdo - zdp, ent2)
    ment2 = jnp.where(neg2 & (zdp > 0.0), ent2f / (dzo * zdp), mentrd)
    cdd2 = jnp.where(zdp > 0.0, det2f / (dzo * zdp), cdd)
    dd_massentro = jnp.where(rng2, ent2f, dd_massentro)
    dd_massdetro = jnp.where(rng2, det2f, dd_massdetro)
    mentrd = jnp.where(rng2, ment2, mentrd)
    cdd = jnp.where(rng2, cdd2, cdd)

    # dd_massentru/detru for k-1 in 1..jmin+1 (k in 2..jmin+2) -> index k-1
    # ref: do k=2..jmin+2: dd_*[k-1]=...; reindex by j=k-1 in 1..jmin+1
    j = k
    in_u = (j >= 1) & (j <= jmin + 1)
    dd_massentru = jnp.where(in_u, dd_massentro + lambau * dd_massdetro, 0.0)
    dd_massdetru = jnp.where(in_u, dd_massdetro + lambau * dd_massdetro, 0.0)

    c1d_dn = jnp.where((k >= kbcon + 1) & (k <= ktop - 1), C1, 0.0)
    return (dd_massentro, dd_massdetro, dd_massentru, dd_massdetru, mentrd, cdd, c1d_dn)


def _downdraft_uvh(jmin, zdo, dd_massdetro, dd_massentro, dd_massdetru,
                   dd_massentru, ucd_in, vcd_in, hcdo_in, us, vs, heo, hco,
                   heso_cup, zo_cup, pgcon, kx):
    """Downward recurrence ki=jmin..1 for ucd,vcd,hcdo + dbydo + bud (ref
    1187-1201)."""
    k = _idx(kx)
    zp = jnp.roll(zo_cup, -1)
    # dbydo[jmin], bud init
    dbydo_jmin = hcdo_in[jmin] - heso_cup[jmin]
    bud0 = dbydo_jmin * (zo_cup[jmin + 1] - zo_cup[jmin])

    def step(carry, ki):
        ucd_p, vcd_p, hcdo_p, bud = carry  # *_p = value at ki+1
        active = (ki >= 1) & (ki <= jmin)
        dzo = zo_cup[ki + 1] - zo_cup[ki]
        h_entr = 0.5 * (heo[ki] + 0.5 * (hco[ki] + hco[ki + 1]))
        d_u = zdo[ki + 1] - 0.5 * dd_massdetru[ki] + dd_massentru[ki]
        d_u = jnp.where(d_u == 0.0, 1.0, d_u)
        d_h = zdo[ki + 1] - 0.5 * dd_massdetro[ki] + dd_massentro[ki]
        d_h = jnp.where(d_h == 0.0, 1.0, d_h)
        ucd_k = (ucd_p * zdo[ki + 1] - 0.5 * dd_massdetru[ki] * ucd_p
                 + dd_massentru[ki] * us[ki] - pgcon * zdo[ki + 1] * (us[ki + 1] - us[ki])) / d_u
        vcd_k = (vcd_p * zdo[ki + 1] - 0.5 * dd_massdetru[ki] * vcd_p
                 + dd_massentru[ki] * vs[ki] - pgcon * zdo[ki + 1] * (vs[ki + 1] - vs[ki])) / d_u
        hcdo_k = (hcdo_p * zdo[ki + 1] - 0.5 * dd_massdetro[ki] * hcdo_p
                  + dd_massentro[ki] * h_entr) / d_h
        dbydo_k = hcdo_k - heso_cup[ki]
        bud_new = bud + jnp.where(active, dbydo_k * dzo, 0.0)
        ucd_c = jnp.where(active, ucd_k, ucd_p)
        vcd_c = jnp.where(active, vcd_k, vcd_p)
        hcdo_c = jnp.where(active, hcdo_k, hcdo_p)
        emit = (jnp.where(active, ucd_k, 0.0), jnp.where(active, vcd_k, 0.0),
                jnp.where(active, hcdo_k, 0.0), jnp.where(active, dbydo_k, 0.0), active)
        return (ucd_c, vcd_c, hcdo_c, bud_new), emit

    kis = jnp.arange(kx, 0, -1)
    init = (ucd_in[jmin + 1], vcd_in[jmin + 1], hcdo_in[jmin + 1], bud0)
    (_, _, _, bud), (ucd_v, vcd_v, hcdo_v, dbydo_v, act_v) = jax.lax.scan(step, init, kis)
    ucd_v = ucd_v[::-1]; vcd_v = vcd_v[::-1]; hcdo_v = hcdo_v[::-1]
    dbydo_v = dbydo_v[::-1]; act_v = act_v[::-1]
    ucd = ucd_in.at[1:].set(jnp.where(act_v, ucd_v, ucd_in[1:]))
    vcd = vcd_in.at[1:].set(jnp.where(act_v, vcd_v, vcd_in[1:]))
    hcdo = hcdo_in.at[1:].set(jnp.where(act_v, hcdo_v, hcdo_in[1:]))
    dbydo = jnp.zeros(kx + 1, F64).at[1:].set(jnp.where(act_v, dbydo_v, 0.0))
    dbydo = dbydo.at[jmin].set(dbydo_jmin)
    return ucd, vcd, hcdo, dbydo, bud


def _dellas_deep(ierr_in, po_cup, zuo, zdo, hco, heo_cup, qco, qo_cup, qcdo,
                 hcdo, qrco, pwo, pwdo, uc, u_cup, ucd, vc, v_cup, vcd, c1d,
                 zo_cup, up_massdetro, edto, ktop, kx):
    """dellah/dellaq/dellat/dellaqc/dellu/dellv (ref 1248-1283). Vectorized."""
    k = _idx(kx)
    bad = ierr_in != 0
    pcp = jnp.roll(po_cup, -1)
    dp = 100.0 * (po_cup - pcp)            # dp[k] for k>=2
    dp_safe = jnp.where(dp == 0.0, 1.0, dp)
    dp1 = 100.0 * (po_cup[1] - po_cup[2])
    G_ = G

    zup = jnp.roll(zuo, -1); zdp = jnp.roll(zdo, -1)
    ucp = jnp.roll(uc, -1); ucupp = jnp.roll(u_cup, -1); ucdp = jnp.roll(ucd, -1)
    vcp = jnp.roll(vc, -1); vcupp = jnp.roll(v_cup, -1); vcdp = jnp.roll(vcd, -1)
    hcop = jnp.roll(hco, -1); heop = jnp.roll(heo_cup, -1); hcdop = jnp.roll(hcdo, -1)
    qcop = jnp.roll(qco, -1); qocp = jnp.roll(qo_cup, -1); qcdop = jnp.roll(qcdo, -1)
    qrcop = jnp.roll(qrco, -1); pwop = jnp.roll(pwo, -1); pwdop = jnp.roll(pwdo, -1)
    zom = jnp.roll(zo_cup, 1)

    rng = (k >= 2) & (k <= ktop)
    dellu = (-(zup * (ucp - ucupp) - zuo * (uc - u_cup)) * G_ / dp_safe
             + (zdp * (ucdp - ucupp) - zdo * (ucd - u_cup)) * G_ / dp_safe * edto * PGCD)
    dellv = (-(zup * (vcp - vcupp) - zuo * (vc - v_cup)) * G_ / dp_safe
             + (zdp * (vcdp - vcupp) - zdo * (vcd - v_cup)) * G_ / dp_safe * edto * PGCD)
    dellu = jnp.where(rng, dellu, 0.0)
    dellv = jnp.where(rng, dellv, 0.0)
    # k==1 entries
    dellu = jnp.where(k == 1, PGCD * (edto * zdo[2] * ucd[2] - edto * zdo[2] * u_cup[2]) * G_ / dp1, dellu)
    dellv = jnp.where(k == 1, PGCD * (edto * zdo[2] * vcd[2] - edto * zdo[2] * v_cup[2]) * G_ / dp1, dellv)

    dellah = (-(zup * (hcop - heop) - zuo * (hco - heo_cup)) * G_ / dp_safe
              + (zdp * (hcdop - heop) - zdo * (hcdo - heo_cup)) * G_ / dp_safe * edto)
    dellah = jnp.where(rng, dellah, 0.0)
    dellah = jnp.where(k == 1, (edto * zdo[2] * hcdo[2] - edto * zdo[2] * heo_cup[2]) * G_ / dp1, dellah)

    dz = zo_cup - zom
    dellaqc_lt = zuo * c1d * qrco * dz / dp_safe * G_      # k<ktop
    dellaqc_eq = up_massdetro * 0.5 * (qrcop + qrco) * G_ / dp_safe  # k==ktop
    dellaqc = jnp.where(rng & (k < ktop), dellaqc_lt,
                        jnp.where(k == ktop, dellaqc_eq, 0.0))

    g_rain = 0.5 * (pwo + pwop) * G_ / dp_safe
    e_dn = -0.5 * (pwdo + pwdop) * G_ / dp_safe * edto
    c_up = dellaqc + (zup * qrcop - zuo * qrco) * G_ / dp_safe + g_rain
    dellaq = (-(zup * (qcop - qocp) - zuo * (qco - qo_cup)) * G_ / dp_safe
              + (zdp * (qcdop - qocp) - zdo * (qcdo - qo_cup)) * G_ / dp_safe * edto
              - c_up + e_dn)
    dellaq = jnp.where(rng, dellaq, 0.0)
    # k==1
    g_rain1 = 0.5 * (pwo[1] + pwo[2]) * G_ / dp1
    e_dn1 = -0.5 * (pwdo[1] + pwdo[2]) * G_ / dp1 * edto
    dellaq1 = (edto * zdo[2] * qcdo[2] - edto * zdo[2] * qo_cup[2]) * G_ / dp1 + e_dn1 - g_rain1
    dellaq = jnp.where(k == 1, dellaq1, dellaq)

    z = jnp.zeros(kx + 1, F64)
    dellat = z
    if_bad = lambda a: jnp.where(bad, z, a)
    return (if_bad(dellah), if_bad(dellaq), dellat, if_bad(dellaqc),
            if_bad(dellu), if_bad(dellv))


def _xhc_recurrence(start_level, ktop, xhkb, xhe_cup, xhes_cup, xzu,
                    up_massdetro, up_massentro, xhe, kx):
    """xhc recurrence start_level+1..ktop (ref 1303-1309). Returns xhc,xdby."""
    k = _idx(kx)

    def step(carry, kk):
        xhc_p = carry
        active = (kk >= start_level + 1) & (kk <= ktop)
        denom = xzu[kk - 1] - 0.5 * up_massdetro[kk - 1] + up_massentro[kk - 1]
        denom = jnp.where(denom == 0.0, 1.0, denom)
        xhc_k = (xhc_p * xzu[kk - 1] - 0.5 * up_massdetro[kk - 1] * xhc_p
                 + up_massentro[kk - 1] * xhe[kk - 1]) / denom
        xhc_c = jnp.where(active, xhc_k, xhc_p)
        return xhc_c, (jnp.where(active, xhc_k, 0.0), active)

    kks = jnp.arange(1, kx + 1)
    init = xhkb  # xhc[start_level]=xhkb
    _, (xhc_v, act_v) = jax.lax.scan(step, init, kks)
    xhc = jnp.where(k < start_level, xhe_cup, 0.0)
    xhc = jnp.where(k == start_level, xhkb, xhc)
    xhc = jnp.where(jnp.zeros(kx + 1, bool).at[1:].set(act_v),
                    jnp.zeros(kx + 1, F64).at[1:].set(xhc_v), xhc)
    xdby = jnp.where(jnp.zeros(kx + 1, bool).at[1:].set(act_v), xhc - xhes_cup, 0.0)
    return xhc, xdby


def get_inversion_layers(p_cup, t_cup, z_cup, qo_cup, qeso_cup, kstart, kend,
                         ierr_in, kx):
    """get_inversion_layers (ref). Returns (k_inv2[1..2], dtempdz). Bounded
    while_loop collecting up to 5 inversion levels (local minima of |sec_deriv|)
    then compaction + 800/550-hPa selection."""
    kte = kx
    l_mid = 300.0; l_shal = 100.0
    k = _idx(kx)
    bad = ierr_in != 0
    # first/second derivatives
    tp = jnp.roll(t_cup, -1); tm = jnp.roll(t_cup, 1)
    zp = jnp.roll(z_cup, -1); zm = jnp.roll(z_cup, 1)
    first_deriv = jnp.zeros(kx + 1, F64)
    kend_p3 = kend + 3
    rng_fd = (k >= 2) & (k <= jnp.minimum(kend_p3 + 4, kte - 1))
    fd = (tp - tm) / (zp - zm)
    first_deriv = jnp.where(rng_fd, fd, 0.0)
    dtempdz = first_deriv
    fdp = jnp.roll(first_deriv, -1); fdm = jnp.roll(first_deriv, 1)
    sd_raw = jnp.abs((fdp - fdm) / (zp - zm))
    rng_sd = (k >= 3) & (k <= jnp.minimum(kend_p3 + 3, kte - 1))
    sec_deriv = jnp.where(rng_sd, sd_raw, 0.0)

    # while ilev<kend_p3: scan kk from k upward, first local-min of sec_deriv
    sdp = jnp.roll(sec_deriv, -1); sdm = jnp.roll(sec_deriv, 1)
    is_localmin = (sec_deriv < sdp) & (sec_deriv < sdm)
    kk_hi = jnp.minimum(kend_p3 + 2, kte - 1)
    ilev0 = jnp.maximum(3, kstart + 1)

    def cond(state):
        ilev, kcur, ix, kinv, cont = state
        return cont & (ilev < kend_p3)

    def body(state):
        ilev, kcur, ix, kinv, _cont = state
        # find first kk in [kcur, kk_hi] with is_localmin
        hit = is_localmin & (k >= kcur) & (k <= kk_hi)
        found = jnp.any(hit)
        kk = jnp.argmax(hit).astype(jnp.int32)
        # if found: kinv[ix]=kk; ix=min(5,ix+1); ilev=kk+1; (continue outer)
        # else: ilev advances to kk_hi+1 (no min) -> for-else break (cont=False)
        kinv2 = jnp.where(found, kinv.at[ix].set(kk), kinv)
        ix2 = jnp.where(found, jnp.minimum(5, ix + 1), ix).astype(jnp.int32)
        ilev2 = jnp.where(found, kk + 1, kk_hi + 1).astype(jnp.int32)
        cont2 = found
        return (ilev2, ilev2, ix2, kinv2, cont2)

    kinv_init = jnp.ones(kx + 1, jnp.int32)
    init = (ilev0, ilev0, jnp.asarray(1, jnp.int32), kinv_init, jnp.asarray(True))
    _, _, _, k_inv, _ = jax.lax.while_loop(cond, body, init)

    # 2nd criteria compaction (ref 1428-1444): bounded fori over kc=1..ken
    ken = _maxloc1_int_jax(k_inv, kx)

    def comp_cond(state):
        kc, kadd, kinv, go = state
        return go & (kc <= ken)

    def comp_body(state):
        kc, kadd, kinv, _go = state
        idx = kc + kadd
        stop = (idx > kte)
        kk = jnp.where(stop, 1, kinv[jnp.minimum(idx, kx)])
        stop2 = stop | (kk == 1)
        do_compact = (~stop2) & (dtempdz[kk] < dtempdz[kk - 1]) & (dtempdz[kk] < dtempdz[kk + 1])
        kadd2 = jnp.where(do_compact, kadd + 1, kadd).astype(jnp.int32)
        # inner shift kj=kc..ken
        def shift(kinv_in):
            kj_arr = jnp.arange(kx + 1)
            src = kj_arr + kadd2
            valid = (kj_arr >= kc) & (kj_arr <= ken) & (src <= kte)
            srcv = jnp.where(src <= kx, kinv_in[jnp.minimum(src, kx)], 1)
            newv = jnp.where(srcv > 1, srcv, jnp.where(srcv == 1, 1, kinv_in))
            return jnp.where(valid, newv, kinv_in)
        kinv2 = jnp.where(do_compact, shift(kinv), kinv)
        return (kc + 1, kadd2, kinv2, ~stop2)

    init_c = (jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32), k_inv, jnp.asarray(True))
    _, _, k_inv, _ = jax.lax.while_loop(comp_cond, comp_body, init_c)

    nmax = _maxloc1_int_jax(k_inv, kx)
    big = 1.0e9
    # sd[kc] = |p_cup[k_inv[kc]]-p_cup[kstart]| - l_shal for kc in 1..nmax
    kc_arr = jnp.arange(kx + 1)
    in_n = (kc_arr >= 1) & (kc_arr <= nmax)
    kinv_kc = jnp.where(in_n, k_inv, 1)
    dp = p_cup[kinv_kc] - p_cup[kstart]
    sd_shal = jnp.where(in_n, jnp.abs(dp) - l_shal, big)
    k800 = _minloc_abs_jax(sd_shal, 1, nmax, kx)
    sd_mid = jnp.where(in_n, jnp.abs(dp) - l_mid, big)
    k550 = _minloc_abs_jax(sd_mid, 1, nmax, kx)
    shal_val = k_inv[k800]
    mid_val = k_inv[k550]
    k_inv2 = jnp.full(kx + 1, -1, jnp.int32)
    k_inv2 = k_inv2.at[1].set(shal_val).at[2].set(mid_val)
    return k_inv2, jnp.where(bad, jnp.zeros(kx + 1, F64), dtempdz)


def _maxloc1_int_jax(arr, kx):
    k = _idx(kx)
    rng = (k >= 1) & (k <= kx)
    small = jnp.where(rng, arr, -2**30)
    return jnp.argmax(small).astype(jnp.int32)


def _minloc_abs_jax(arr, lo, hi, kx):
    k = _idx(kx)
    rng = (k >= lo) & (k <= hi)
    big = jnp.where(rng, jnp.abs(arr), jnp.inf)
    return jnp.argmin(big).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Shallow-convection driver cup_gf_sh.
# ---------------------------------------------------------------------------


def cup_gf_sh(zo, t, q, z1, tn, qo, po, psur, dhdt, kpbl, rho, hfx, qfx, xland,
              ichoice, tcrit, dtime, kx):
    """JAX traceable single-column CUP_gf_sh (shallow). Returns dict matching
    the reference cup_gf_sh outputs."""
    ktf = kx
    kte = kx
    k = _idx(kx)
    z1 = jnp.asarray(z1, F64); psur = jnp.asarray(psur, F64)
    hfx = jnp.asarray(hfx, F64); qfx = jnp.asarray(qfx, F64)
    xland = jnp.asarray(xland, F64); dtime = jnp.asarray(dtime, F64)
    kpbl = jnp.asarray(kpbl, jnp.int32)
    flux_tun = FLUXTUNE
    is_water = (xland > 1.5) | (xland < 0.5)
    xl_int = (xland + 0.001).astype(jnp.int32)
    xland1 = jnp.where(is_water, 0, xl_int).astype(jnp.int32)
    entr_rate = 9.0e-5
    cap_max_increment = 25.0
    z = zo
    ierr = jnp.asarray(0, jnp.int32)
    # zws/ztexec/zqexec
    buo_flux = (hfx / CP + 0.608 * t[1] * qfx / XLV) / rho[1]
    zws_t = jnp.maximum(0.0, flux_tun * 0.41 * buo_flux * zo[2] * 9.81 / t[1])
    big = zws_t > TINY
    zws_t2 = 1.2 * zws_t ** 0.3333
    ztexec = jnp.where(big, jnp.maximum(flux_tun * hfx / (rho[1] * jnp.where(big, zws_t2, 1.0) * CP), 0.0), 0.0)
    zqexec = jnp.where(big, jnp.maximum(flux_tun * qfx / XLV / (rho[1] * jnp.where(big, zws_t2, 1.0)), 0.0), 0.0)
    zws = jnp.maximum(0.0, flux_tun * 0.41 * buo_flux * zo[kpbl] * 9.81 / t[kpbl])
    zws = 1.2 * zws ** 0.3333 * rho[kpbl]
    cap_maxs = 125.0
    zkbmax = 3000.0

    z_e, qes, he, hes = cup_env(z, t, q, po, z1, psur, ierr, -1, kx)
    zo_e, qeso, heo, heso = cup_env(zo, tn, qo, po, z1, psur, ierr, -1, kx)
    z = z_e; zo = zo_e
    qes_cup, q_cup, he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup = \
        cup_env_clev(t, qes, q, he, hes, z, po, psur, z1, ierr, kx)
    qeso_cup, qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup = \
        cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, ierr, kx)

    cb = (zo_cup > zkbmax + z1) & (k >= 1) & (k <= ktf)
    kbmax = jnp.where(jnp.any(cb), jnp.argmax(cb), 1).astype(jnp.int32)
    kbmax = jnp.minimum(kbmax, ktf // 2)

    cap_max = jnp.where(kpbl > 3, po_cup[kpbl], cap_maxs)
    k22 = _argmax_range(heo_cup, 2, kbmax, kx)
    k22 = jnp.maximum(2, k22)
    ierr = jnp.where(k22 > kbmax, 2, ierr).astype(jnp.int32)
    k22 = jnp.where(ierr == 2, 0, k22).astype(jnp.int32)
    k22_safe = jnp.maximum(k22, 1)

    x_add = XLV * zqexec + CP * ztexec
    hkb = get_cloud_bc(he_cup, k22_safe, x_add, kx)
    hkbo = get_cloud_bc(heo_cup, k22_safe, x_add, kx)

    k22, kbcon, hkbo, ierr = cup_kbcon(cap_max_increment, 5, k22_safe, heo_cup,
                                       heso_cup, hkbo, ierr, kbmax, po_cup, cap_max,
                                       ztexec, zqexec, z_cup, entr_rate, heo, 0, kx)
    kstabi = cup_minimi(heso_cup, kbcon, kbmax, ierr, kx)
    k_inv_layers, dtempdz = get_inversion_layers(p_cup, t_cup, z_cup, q_cup,
                                                 qes_cup, kbcon, kstabi, ierr, kx)
    bad = ierr != 0
    entr_rate_2d = jnp.where((k >= 1) & (k <= ktf), entr_rate, 0.0)
    start_level = k22
    x_add = XLV * zqexec + CP * ztexec
    hkb = jnp.where(~bad, get_cloud_bc(he_cup, k22_safe, x_add, kx), hkb)
    ierr = jnp.where((~bad) & (kbcon > ktf - 4), 231, ierr).astype(jnp.int32)
    bad = ierr != 0
    frh = 2.0 * jnp.minimum(qo_cup / jnp.where(qeso_cup == 0.0, 1.0, qeso_cup), 1.0)
    entr_rate_2d = jnp.where((~bad) & (k >= 1) & (k <= ktf), entr_rate * (2.3 - frh), entr_rate_2d)
    cd = jnp.where((~bad) & (k >= 1) & (k <= ktf), entr_rate_2d, jnp.where((k >= 1) & (k <= ktf), 1.0 * entr_rate, 0.0))
    # ktop from inversion or pressure search
    inv1 = k_inv_layers[1]
    use_inv = (inv1 > 0) & ((po_cup[kbcon] - po_cup[jnp.maximum(inv1, 1)]) < 200.0)
    pc = (po_cup[kbcon] - po_cup > 200.0) & (k >= kbcon + 1) & (k <= ktf)
    ktop_p = jnp.where(jnp.any(pc), jnp.argmax(pc), 1).astype(jnp.int32)
    ktop = jnp.where(~bad, jnp.where(use_inv, inv1, ktop_p), 1).astype(jnp.int32)

    # rates_up_pdf shallow
    kbcon, ktop, ktopx, ierr, zuo = rates_up_pdf(
        'shallow', ktop, ierr, po_cup, entr_rate_2d, hkbo, heo, heso_cup, zo_cup,
        xland1, kstabi, k22, kbcon, kpbl, jnp.asarray(0.0, F64), kx)
    bad = ierr != 0
    # zero zuo below k22; trim ktop at zuo<1e-6; zu=zuo
    below = (k < k22) & (k22 > 1)
    zuo = jnp.where((~bad) & below, 0.0, zuo)
    mlz = _maxloc1(zuo, kx)
    trim = (zuo < 1.0e-6) & (k >= mlz) & (k <= ktop)
    ktop_trim = jnp.where(jnp.any(trim), jnp.argmax(trim) - 1, ktop).astype(jnp.int32)
    ktop = jnp.where(~bad, ktop_trim, ktop).astype(jnp.int32)
    above = (k >= ktop + 1) & (k <= ktf)
    zuo = jnp.where((~bad) & above, 0.0, zuo)
    k22 = jnp.where(~bad, jnp.maximum(2, k22), k22).astype(jnp.int32)
    zu = zuo

    # lateral massflux (with_u=False)
    ktop_lat = jnp.where(bad, 0, ktop).astype(jnp.int32)
    (up_massentro, up_massdetro, up_massentr, up_massdetr, _ue, _ud, cd, entr_rate_2d) = \
        get_lateral_massflux(zo_cup, zuo, cd, entr_rate_2d, ktop_lat, kbcon, k22, 0.0, kx, with_u=False)

    # hc/hco recurrence (shallow: dby=max(0,hc-hes_cup); no uc/vc/pgcon needed but
    # reuse the helper with pgcon=0; uc/vc unused downstream)
    sl = jnp.maximum(start_level, 1)
    u_cup0 = jnp.zeros(kx + 1, F64)
    (hc_v, _uc, _vc, dby_v, hco_v, dbyo_v, dbyt_v, emit_v, hc_break) = \
        _hc_hco_recurrence_shallow(sl, ktop, hkb, hkbo, he, heo, hes_cup, heso_cup,
                                   zu, zuo, up_massentr, up_massdetr, up_massentro,
                                   up_massdetro, zo_cup, kx)
    hc = jnp.where(k < sl, he_cup, 0.0)
    hco = jnp.where(k < sl, heo_cup, 0.0)
    hc = jnp.where(k == sl, hkb, hc)
    hco = jnp.where(k == sl, hkbo, hco)
    emit = jnp.zeros(kx + 1, bool).at[1:].set(emit_v)
    hc = jnp.where(emit, jnp.zeros(kx + 1, F64).at[1:].set(hc_v), hc)
    hco = jnp.where(emit, jnp.zeros(kx + 1, F64).at[1:].set(hco_v), hco)
    dby = jnp.zeros(kx + 1, F64).at[1:].set(dby_v)
    dbyo = jnp.zeros(kx + 1, F64).at[1:].set(dbyo_v)
    dbyt = jnp.zeros(kx + 1, F64).at[1:].set(dbyt_v)
    # ktop = ki+1 if ktop>ki+1 where ki=maxloc(dbyt)
    ki_db = _maxloc1(dbyt, kx)
    new_ktop = jnp.where((~bad) & (ktop > ki_db + 1), ki_db + 1, ktop).astype(jnp.int32)
    trimmed = (~bad) & (ktop > ki_db + 1)
    # zero zuo/zu/cd above new_ktop; up_massdetro[new_ktop]=zuo[new_ktop]; etc.
    ab2 = trimmed & (k >= new_ktop + 1) & (k <= ktf)
    zuo = jnp.where(ab2, 0.0, zuo)
    zu = jnp.where(ab2, 0.0, zu)
    cd = jnp.where(ab2, 0.0, cd)
    up_massdetro = jnp.where(trimmed & (k == new_ktop), zuo, up_massdetro)
    up_massentro = jnp.where(trimmed & (k >= new_ktop) & (k <= ktf), 0.0, up_massentro)
    up_massdetro = jnp.where(ab2, 0.0, up_massdetro)
    entr_rate_2d = jnp.where(ab2, 0.0, entr_rate_2d)
    ktop = new_ktop

    # skip_42: ierr=5 if ktop<kbcon+1 or ktop>ktf-2
    trip5 = (~bad) & ((ktop < kbcon + 1) | (ktop > ktf - 2))
    ierr = jnp.where(trip5, 5, ierr).astype(jnp.int32)
    skip42 = trip5
    bad = ierr != 0

    # qco/qrco/pwo/cnvwt updraft moisture (shallow style, ref 1633-1658)
    do_42 = (~bad) & (~skip42)
    (qco, qrco, pwo, cupclw, cnvwt) = _shallow_up_moisture(
        do_42, k22, start_level, ktop, kbcon, qo_cup, qeso_cup, gammao_cup, dbyo,
        up_massentr, up_massdetr, zuo, z_cup, po_cup, zqexec, he_cup, hes_cup,
        heso_cup, qo, kx)

    # work functions
    aa0 = cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr, kx)
    aa1 = cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop, ierr, kx)
    ierr = jnp.where((~bad) & (aa1 <= 0.0), 17, ierr).astype(jnp.int32)
    bad = ierr != 0

    # dellah/dellaq (shallow ref 1667-1679)
    (dellah, dellaq, dellat, dellaqc) = _shallow_dellas(
        ierr, po_cup, zuo, hco, heo_cup, qco, qo_cup, qrco, pwo, up_massentro,
        up_massdetro, zo_cup, k22, ktop, kx)

    # x-profiles
    mbdt = 0.5
    xhe = jnp.where((~bad) & (k >= 1) & (k <= ktf), dellah * mbdt + heo, 0.0)
    xq = jnp.where((~bad) & (k >= 1) & (k <= ktf), jnp.maximum(1.0e-16, (dellaq + dellaqc) * mbdt + qo), 0.0)
    dellat = jnp.where((~bad) & (k >= 1) & (k <= ktf), (1.0 / CP) * (dellah - XLV * dellaq), dellat)
    xt = jnp.where((~bad) & (k >= 1) & (k <= ktf), jnp.maximum(190.0, (-dellaqc * XLV / CP + dellat) * mbdt + tn), 0.0)
    xhe = jnp.where((~bad) & (k == ktf), heo[ktf], xhe)
    xq = jnp.where((~bad) & (k == ktf), qo[ktf], xq)
    xt = jnp.where((~bad) & (k == ktf), tn[ktf], xt)
    xz = zo
    xz, xqes, xhe2, xhes = cup_env(xz, xt, xq, po, z1, psur, ierr, -1, kx)
    xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup, _xp, _xg, xt_cup = \
        cup_env_clev(xt, xqes, xq, xhe2, xhes, xz, po, psur, z1, ierr, kx)
    x_add = XLV * zqexec + CP * ztexec
    xhkb = get_cloud_bc(xhe_cup, k22_safe, x_add, kx)
    xzu = zuo
    xhc, xdby = _xhc_recurrence(sl, ktop, xhkb, xhe_cup, xhes_cup, xzu,
                               up_massdetro, up_massentro, xhe, kx)
    xhc = jnp.where((~bad) & (k >= ktop + 1) & (k <= ktf), xhes_cup, xhc)
    xdby = jnp.where((~bad) & (k >= ktop + 1) & (k <= ktf), 0.0, xdby)
    xaa0 = cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon, ktop, ierr, kx)

    # shallow forcing
    xmbmax = 1.0
    xkshal = (xaa0 - aa1) / mbdt
    xkshal = jnp.where((xkshal <= 0.0) & (xkshal > -0.01 * mbdt), -0.01 * mbdt, xkshal)
    xkshal = jnp.where((xkshal > 0.0) & (xkshal < 1.0e-2), 1.0e-2, xkshal)
    xkshal_safe = jnp.where(xkshal == 0.0, 1.0, xkshal)
    xff0 = jnp.maximum(0.0, -(aa1 - aa0) / (xkshal_safe * dtime))
    xff1 = 0.03 * zws
    blqe = jnp.sum(jnp.where((k >= 1) & (k <= kpbl), 100.0 * dhdt * (po_cup - jnp.roll(po_cup, -1)) / G, 0.0))
    trash = jnp.maximum(hc[kbcon] - he_cup[kbcon], 1.0e1)
    xff2 = jnp.minimum(xmbmax, jnp.maximum(0.0, blqe / trash))
    xmb = jnp.minimum(xmbmax, (xff0 + xff1 + xff2) / 3.0)
    xff_arr = jnp.stack([xff0, xff1, xff2])
    if ichoice > 0:
        xmb = jnp.minimum(xmbmax, xff_arr[ichoice - 1])
    ierr = jnp.where((~bad) & (xmb <= 0.0), 21, ierr).astype(jnp.int32)
    bad = ierr != 0

    # output
    xmb_out = jnp.where(~bad, xmb, 0.0)
    rng = (~bad) & (k >= 2) & (k <= ktop)
    outt = jnp.where(rng, dellat * xmb, 0.0)
    outq = jnp.where(rng, dellaq * xmb, 0.0)
    outqc = jnp.where(rng, dellaqc * xmb, 0.0)
    pre = jnp.sum(jnp.where(rng, pwo * xmb, 0.0))
    pre = jnp.where(bad, 0.0, pre)
    k22_f = jnp.where(bad, 0, k22).astype(jnp.int32)
    kbcon_f = jnp.where(bad, 0, kbcon).astype(jnp.int32)
    ktop_f = jnp.where(bad, 0, ktop).astype(jnp.int32)
    return {
        'outt': outt, 'outq': outq, 'outqc': outqc, 'pre': pre,
        'kbcon': kbcon_f, 'ktop': ktop_f, 'k22': k22_f, 'ierr': ierr,
        'xmb_out': xmb_out, 'cnvwt': cnvwt, 'cupclw': cupclw, 'zuo': zuo,
    }


def _hc_hco_recurrence_shallow(start_level, ktop, hkb, hkbo, he_env, heo_env,
                               hes_cup, heso_cup, zu, zuo, up_massentr,
                               up_massdetr, up_massentro, up_massdetro, zo_cup, kx):
    """Shallow hc/hco recurrence start_level+1..ktop. dby=max(0,hc-hes_cup).
    No denom-break in the shallow path (ref does not guard). Returns same tuple
    shape as the deep helper (uc/vc are zeros)."""
    def step(carry, kk):
        hc_p, hco_p, dbyt_p = carry
        active = (kk >= start_level + 1) & (kk <= ktop)
        d_h = zu[kk - 1] - 0.5 * up_massdetr[kk - 1] + up_massentr[kk - 1]
        d_h = jnp.where(d_h == 0.0, 1.0, d_h)
        d_ho = zuo[kk - 1] - 0.5 * up_massdetro[kk - 1] + up_massentro[kk - 1]
        d_ho = jnp.where(d_ho == 0.0, 1.0, d_ho)
        hc_k = (hc_p * zu[kk - 1] - 0.5 * up_massdetr[kk - 1] * hc_p
                + up_massentr[kk - 1] * he_env[kk - 1]) / d_h
        dby_k = jnp.maximum(0.0, hc_k - hes_cup[kk])
        hco_k = (hco_p * zuo[kk - 1] - 0.5 * up_massdetro[kk - 1] * hco_p
                 + up_massentro[kk - 1] * heo_env[kk - 1]) / d_ho
        dbyo_k = hco_k - heso_cup[kk]
        dz = zo_cup[kk + 1] - zo_cup[kk]
        dbyt_k = dbyt_p + jnp.where(active, dbyo_k * dz, 0.0)
        hc_c = jnp.where(active, hc_k, hc_p)
        hco_c = jnp.where(active, hco_k, hco_p)
        emit = (jnp.where(active, hc_k, 0.0), 0.0, 0.0,
                jnp.where(active, dby_k, 0.0), jnp.where(active, hco_k, 0.0),
                jnp.where(active, dbyo_k, 0.0), dbyt_k, active)
        return (hc_c, hco_c, dbyt_k), emit

    kks = jnp.arange(1, kx + 1)
    init = (hkb, hkbo, jnp.zeros((), F64))
    _, (hc_v, uc_v, vc_v, dby_v, hco_v, dbyo_v, dbyt_v, emit_v) = \
        jax.lax.scan(step, init, kks)
    return (hc_v, uc_v, vc_v, dby_v, hco_v, dbyo_v, dbyt_v, emit_v, jnp.asarray(False))


def _shallow_up_moisture(do_42, k22, start_level, ktop, kbcon, qo_cup, qeso_cup,
                         gammao_cup, dbyo, up_massentr, up_massdetr, zuo, z_cup,
                         po_cup, zqexec, he_cup, hes_cup, heso_cup, qo, kx):
    """Shallow updraft moisture qco/qrco/pwo + cupclw/cnvwt (ref 1633-1654).
    Upward recurrence k=start_level+1..ktop."""
    k = _idx(kx)
    qaver = get_cloud_bc(qo_cup, jnp.maximum(k22, 1), 0.0, kx) + zqexec
    qco_init = jnp.where(k < start_level, qo_cup, 0.0)
    qco_init = jnp.where(k == start_level, qaver, qco_init)

    def step(carry, kk):
        qco_p = carry  # qco[kk-1]
        active = (kk >= start_level + 1) & (kk <= ktop)
        trash = qeso_cup[kk] + (1.0 / XLV) * (gammao_cup[kk] / (1.0 + gammao_cup[kk])) * dbyo[kk]
        denom = zuo[kk - 1] - 0.5 * up_massdetr[kk - 1] + up_massentr[kk - 1]
        denom = jnp.where(denom == 0.0, 1.0, denom)
        qco_k = (qco_p * (zuo[kk - 1] - 0.5 * up_massdetr[kk - 1])
                 + up_massentr[kk - 1] * qo[kk - 1]) / denom
        ge = qco_k >= trash
        dz = z_cup[kk] - z_cup[kk - 1]
        qrco_k = jnp.where(ge, (qco_k - trash) / (1.0 + (C0_SHAL + C1_SHAL) * dz), 0.0)
        pwo_k = jnp.where(ge, C0_SHAL * dz * qrco_k * zuo[kk], 0.0)
        qco_final = jnp.where(ge, trash + qrco_k, qco_k)
        qco_c = jnp.where(active, qco_final, qco_p)
        emit = (jnp.where(active, qco_final, 0.0), jnp.where(active, qrco_k, 0.0),
                jnp.where(active, pwo_k, 0.0), active)
        return qco_c, emit

    kks = jnp.arange(1, kx + 1)
    _, (qco_v, qrco_v, pwo_v, act_v) = jax.lax.scan(step, qco_init[start_level], kks)
    qco = qco_init.at[1:].set(jnp.where(act_v, qco_v, qco_init[1:]))
    qrco = jnp.zeros(kx + 1, F64).at[1:].set(jnp.where(act_v, qrco_v, 0.0))
    pwo = jnp.zeros(kx + 1, F64).at[1:].set(jnp.where(act_v, pwo_v, 0.0))
    cupclw = qrco
    # cnvwt + qco-=qrco for k in k22+1..ktop
    pcp = jnp.roll(po_cup, -1)
    dp = 100.0 * (po_cup - pcp)
    rng2 = (k >= k22 + 1) & (k <= ktop)
    cnvwt = jnp.where(rng2, zuo * cupclw * G / jnp.where(dp == 0.0, 1.0, dp), 0.0)
    qco = jnp.where(rng2, qco - qrco, qco)
    # if not do_42, return zeros
    z = jnp.zeros(kx + 1, F64)
    if_no = lambda a: jnp.where(do_42, a, z)
    return (if_no(qco), if_no(qrco), if_no(pwo), if_no(cupclw), if_no(cnvwt))


def _shallow_dellas(ierr_in, po_cup, zuo, hco, heo_cup, qco, qo_cup, qrco, pwo,
                    up_massentro, up_massdetro, zo_cup, k22, ktop, kx):
    """Shallow dellah/dellaq/dellaqc (ref 1667-1679). Vectorized k=k22..ktop."""
    k = _idx(kx)
    bad = ierr_in != 0
    pcp = jnp.roll(po_cup, -1)
    dp = 100.0 * (po_cup - pcp)
    dp_safe = jnp.where(dp == 0.0, 1.0, dp)
    zup = jnp.roll(zuo, -1)
    hcop = jnp.roll(hco, -1); heop = jnp.roll(heo_cup, -1)
    qcop = jnp.roll(qco, -1); qocp = jnp.roll(qo_cup, -1)
    qrcop = jnp.roll(qrco, -1); pwop = jnp.roll(pwo, -1)
    zop = jnp.roll(zo_cup, -1)
    rng = (k >= k22) & (k <= ktop)
    dellah = -(zup * (hcop - heop) - zuo * (hco - heo_cup)) * G / dp_safe
    dz = zop - zo_cup
    dellaqc_lt = zuo * C1_SHAL * qrco * dz / dp_safe * G
    dellaqc_eq = up_massdetro * qrco * G / dp_safe
    dellaqc = jnp.where(k < ktop, dellaqc_lt, dellaqc_eq)
    c_up = dellaqc + (zup * qrcop - zuo * qrco) * G / dp_safe
    dellaq = (-(zup * (qcop - qocp) - zuo * (qco - qo_cup)) * G / dp_safe
              - c_up - 0.5 * (pwo + pwop) * G / dp_safe)
    dellah = jnp.where(rng, dellah, 0.0)
    dellaqc = jnp.where(rng, dellaqc, 0.0)
    dellaq = jnp.where(rng, dellaq, 0.0)
    z = jnp.zeros(kx + 1, F64)
    if_bad = lambda a: jnp.where(bad, z, a)
    return (if_bad(dellah), if_bad(dellaq), z, if_bad(dellaqc))


GF_OUTPUT_KEYS = (
    'RTHCUTEN', 'RQVCUTEN', 'RQCCUTEN', 'RQICUTEN', 'RAINCV', 'PRATEC',
    'KTOP_DEEP', 'XMB_SHALLOW', 'K22_SHALLOW', 'KBCON_SHALLOW', 'KTOP_SHALLOW',
    'IERR_DEEP', 'IERR_SHALLOW',
)


def gfdrv_column(t_col, qv_col, p_col, pi_col, dz8w_col, rho_col, u_col, v_col,
                 w_col, rthblten_col, rqvblten_col, dt, dx, hfx, qfx, kpbl,
                 xland, ht, kx, ishallow_g3=1, ichoice=0):
    """JAX traceable single-column GFDRV (cu_physics=3). 1-based length-(kx+1)
    inputs (index 0 unused). Returns dict of WRF cumulus-driver outputs as
    length-(kx+1) arrays / scalars (index 0 unused for arrays)."""
    ktf = kx
    dicycle = 1
    tcrit = 258.0
    ichoice_s = 0
    k = _idx(kx)
    dt = jnp.asarray(dt, F64); dx = jnp.asarray(dx, F64)
    hfx = jnp.asarray(hfx, F64); qfx = jnp.asarray(qfx, F64)
    xland = jnp.asarray(xland, F64); ht = jnp.asarray(ht, F64)
    kpbl = jnp.asarray(kpbl, jnp.int32)

    active = (k >= 1) & (k <= ktf)
    q2d = jnp.where(qv_col < 1.0e-8, 1.0e-8, qv_col)
    q2d = jnp.where(active, q2d, 0.0)
    po = jnp.where(active, p_col * 0.01, 0.0)
    psur = p_col[1] * 0.01
    ter11 = jnp.maximum(0.0, ht)
    # zo heights (cumulative)
    dz8w = dz8w_col
    incr = 0.5 * (jnp.roll(dz8w, 1) + dz8w)  # for k>=2: 0.5*(dz[k-1]+dz[k])
    z1_val = ter11 + 0.5 * dz8w[1]
    incr_m = jnp.where(k >= 2, incr, 0.0)
    zo = jnp.cumsum(jnp.where(k == 1, z1_val, incr_m))
    zo = jnp.where(active, zo, 0.0)
    # forced sounding
    tn0 = t_col + rthblten_col * pi_col * dt
    qo0 = q2d + rqvblten_col * dt
    tshall = t_col + rthblten_col * pi_col * dt
    dhdt = CP * rthblten_col * pi_col + XLV * rqvblten_col
    qshall = q2d + rqvblten_col * dt
    tn = jnp.where(tn0 < 200.0, t_col, tn0)
    tn = jnp.where(active, tn, 0.0)
    qo = jnp.where(qo0 < 1.0e-8, 1.0e-8, qo0)
    qo = jnp.where(active, qo, 0.0)
    # omega, mconv
    omeg = jnp.where(active, -G * rho_col * w_col, 0.0)
    dq = jnp.roll(q2d, -1) - q2d  # q2d[k+1]-q2d[k]
    mconv = jnp.sum(jnp.where((k >= 1) & (k <= ktf - 1), omeg * dq / G, 0.0))
    mconv = jnp.maximum(0.0, mconv)
    csum = jnp.asarray(0.0, F64)
    ccn = 150.0

    # shallow
    sh = cup_gf_sh(zo, t_col, q2d, ter11, tshall, qshall, po, psur, dhdt, kpbl,
                   rho_col, hfx, qfx, xland, ichoice_s, tcrit, dt, kx)
    cutens = jnp.where((ishallow_g3 == 1) & (sh['xmb_out'] > 0.0), 1.0, 0.0)
    # neg_check shallow
    zoutu = jnp.zeros(kx + 1, F64)
    outqs, outts, _ou, _ov, outqcs, pre_s = neg_check(
        'shallow', dt, q2d, sh['outq'], sh['outt'], zoutu, zoutu, sh['outqc'],
        sh['pre'], kx)

    # deep
    dp_res = cup_gf(dicycle, ichoice, ccn, dt, 0, kpbl, dhdt, xland, zo, t_col,
                    q2d, ter11, tn, qo, po, psur, u_col, v_col, rho_col, hfx, qfx,
                    dx, mconv, omeg, csum, kx)
    outq_d, outt_d, outu_d, outv_d, outqc_d, pre_d = neg_check(
        'deep', dt, q2d, dp_res['outq'], dp_res['outt'], dp_res['outu'],
        dp_res['outv'], dp_res['outqc'], dp_res['pre'], kx)

    cuten = jnp.where((dp_res['ierr'] == 0) & (pre_d > 0.0), 1.0, 0.0)
    kbcon_d = jnp.where(cuten == 0.0, 0, dp_res['kbcon']).astype(jnp.int32)
    ktop_d = jnp.where(cuten == 0.0, 0, dp_res['ktop']).astype(jnp.int32)

    pi1 = jnp.where(pi_col == 0.0, 1.0, pi_col)
    rthcuten = jnp.where(active, (cutens * outts + cuten * outt_d) / pi1, 0.0)
    rqvcuten = jnp.where(active, cuten * outq_d + cutens * outqs, 0.0)
    prets = pre_s
    has_precip = (pre_d > 0.0) | (prets > 0.0)
    pratec = jnp.where(has_precip, cuten * pre_d + cutens * prets, 0.0)
    raincv = pratec * dt
    rqccuten0 = outqcs + outqc_d * cuten
    ice = t_col < 258.0
    rqicuten = jnp.where(active & ice, rqccuten0, 0.0)
    rqccuten = jnp.where(active & (~ice), rqccuten0, 0.0)

    return {
        'RTHCUTEN': rthcuten, 'RQVCUTEN': rqvcuten, 'RQCCUTEN': rqccuten,
        'RQICUTEN': rqicuten, 'RAINCV': raincv, 'PRATEC': pratec,
        'KTOP_DEEP': ktop_d, 'XMB_SHALLOW': sh['xmb_out'],
        'K22_SHALLOW': sh['k22'], 'KBCON_SHALLOW': sh['kbcon'],
        'KTOP_SHALLOW': sh['ktop'], 'IERR_DEEP': dp_res['ierr'],
        'IERR_SHALLOW': sh['ierr'],
    }


# ---------------------------------------------------------------------------
# Public jit / vmap entry points.
# ---------------------------------------------------------------------------


def _to1(arr0, kx):
    """0-based length-kx -> 1-based length-(kx+1) (index 0 = 0)."""
    return jnp.concatenate([jnp.zeros(1, F64), jnp.asarray(arr0, F64)])


@functools.partial(jax.jit, static_argnums=(11,), static_argnames=('ishallow_g3', 'ichoice'))
def gfdrv_column_jit(t1, qv1, p1, pi1, dz1, rho1, u1, v1, w1, rthbl1, rqvbl1,
                     kx, dt, dx, hfx, qfx, kpbl, xland, ht,
                     ishallow_g3=1, ichoice=0):
    """jit-compiled single column. 1-based length-(kx+1) array inputs."""
    return gfdrv_column(t1, qv1, p1, pi1, dz1, rho1, u1, v1, w1, rthbl1, rqvbl1,
                        dt, dx, hfx, qfx, kpbl, xland, ht, kx,
                        ishallow_g3=ishallow_g3, ichoice=ichoice)


@functools.partial(jax.jit, static_argnums=(11,), static_argnames=('ishallow_g3', 'ichoice'))
def gfdrv_batched(t1, qv1, p1, pi1, dz1, rho1, u1, v1, w1, rthbl1, rqvbl1,
                  kx, dt, dx, hfx, qfx, kpbl, xland, ht,
                  ishallow_g3=1, ichoice=0):
    """GPU-batched GFDRV over a leading column axis via jax.vmap.

    Array inputs have shape (NCOL, kx+1) (1-based per column). Per-column
    scalars (dt, dx, hfx, qfx, kpbl, xland, ht) have shape (NCOL,). Returns a
    dict of stacked per-column outputs. No host transfer inside the column
    loop: the entire deep+shallow column physics runs inside one vmapped jit."""
    col = lambda *a: gfdrv_column(*a, kx, ishallow_g3=ishallow_g3, ichoice=ichoice)
    # vmap over arrays (axis 0) and per-column scalars (axis 0)
    in_axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 11 arrays
               0, 0, 0, 0, 0, 0, 0)              # dt,dx,hfx,qfx,kpbl,xland,ht
    return jax.vmap(col, in_axes=in_axes)(
        t1, qv1, p1, pi1, dz1, rho1, u1, v1, w1, rthbl1, rqvbl1,
        dt, dx, hfx, qfx, kpbl, xland, ht)


def grell_freitas_column_gpu(t, qv, p, dz, rho, w, *, dt, dx, pi_exner=None,
                             u=None, v=None, rthblten=None, rqvblten=None,
                             kpbl=5, hfx=0.0, qfx=0.0, xland=1.0, ht=0.0,
                             csum=0.0):
    """Drop-in GPU (jit) replacement for ``grell_freitas_column`` (0-based KX
    inputs). Returns NumPy arrays/scalars matching the reference output dict
    keys used by the parity harness."""
    import numpy as np
    del csum
    t = np.asarray(t, np.float64)
    kx = t.shape[0]
    if pi_exner is None:
        pi_exner = (np.asarray(p, np.float64) / 1.0e5) ** (287.0 / 1004.0)
    z0 = np.zeros(kx, np.float64)
    u = z0 if u is None else u
    v = z0 if v is None else v
    rthblten = z0 if rthblten is None else rthblten
    rqvblten = z0 if rqvblten is None else rqvblten
    args = [_to1(x, kx) for x in (t, qv, p, pi_exner, dz, rho, u, v, w,
                                  rthblten, rqvblten)]
    out = gfdrv_column_jit(*args, kx, float(dt), float(dx), float(hfx),
                           float(qfx), jnp.asarray(int(kpbl), jnp.int32),
                           float(xland), float(ht))
    import numpy as _np
    def to0(a):
        return _np.asarray(a)[1:]
    rthcuten = to0(out['RTHCUTEN']); rqvcuten = to0(out['RQVCUTEN'])
    rqccuten = to0(out['RQCCUTEN']); rqicuten = to0(out['RQICUTEN'])
    # RAINCV/PRATEC are column scalars.
    raincv = float(out['RAINCV'])
    pratec = float(out['PRATEC'])
    ktop_deep = int(out['KTOP_DEEP'])
    xmb_shallow = float(out['XMB_SHALLOW'])
    return {
        'RTHCUTEN': rthcuten, 'RQVCUTEN': rqvcuten, 'RQCCUTEN': rqccuten,
        'RQICUTEN': rqicuten, 'RAINCV': raincv, 'PRATEC': pratec,
        'KTOP_DEEP': ktop_deep, 'XMB_SHALLOW': xmb_shallow,
        'K22_SHALLOW': int(out['K22_SHALLOW']), 'KBCON_SHALLOW': int(out['KBCON_SHALLOW']),
        'KTOP_SHALLOW': int(out['KTOP_SHALLOW']), 'IERR_DEEP': int(out['IERR_DEEP']),
        'IERR_SHALLOW': int(out['IERR_SHALLOW']),
        'TRIGGER_DEEP': bool(ktop_deep > 0 and raincv > 0.0),
        'TRIGGER_SHALLOW': bool(xmb_shallow > 0.0 or int(out['KTOP_SHALLOW']) > 0),
    }
