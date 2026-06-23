"""JIT/vmap-traceable MRF PBL column kernel (``bl_pbl_physics=99``).

This is a faithful, ``jax.jit``/``jax.vmap``-traceable transcription of the
single-column path of pristine WRF ``phys/module_bl_mrf.F`` (``MRF`` ->
``MRF2D`` -> ``TRIDI2``): the Hong-Pan (1996) nonlocal-K MRF PBL with the
Troen-Mahrt countergradient term, regime-dependent free-atmosphere mixing, and
the leapfrog (``DT4 = 2*dt``) implicit vertical-diffusion solve for heat,
moisture, momentum and cloud.

The scheme is the algorithmic ancestor of YSU. Unlike YSU it has no explicit
entrainment/top-down branch, uses ``BRCR = 0.5`` for the working PBL height and a
second pass with ``BRCR = 0`` for the diagnostic ``PBL0`` height, and applies the
``(WSPD1/WSPD)**2`` surface-drag correction on the momentum lower boundary.

Conventions: profile inputs are bottom-up WRF mass-level columns of length ``n``
(``u, v, t, qv, qc, p, pii, dz, z``); ``z`` is the full-level geopotential height
above sea level (the WRF ``z`` argument). Internally the kernel mirrors the
Fortran top-down 1-based work arrays (level 1 = model top) using length-``n``
``jnp`` arrays so the column depth ``n`` is a static trace-time int while the
batch axis is left free for ``jax.vmap`` over grid columns.

WRF-faithfulness is gated by the v0.13 operational oracle
(``proofs/v013/mrf_oracle.py``) against fp64 savepoints from a standalone Fortran
driver linked against the UNMODIFIED ``module_bl_mrf.F`` at ~1e-13 (NOT a
JAX-vs-JAX self-compare).
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp

configure_jax_x64()

# --- WRF model_model_constants / MRF driver-passed constants ---------------- #
G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
XLV = 2.5e6
ROVCP = R_D / CP
EP1 = R_V / R_D - 1.0
KARMAN = 0.4
P1000 = 1.0e5

# --- MRF2D PARAMETER block (module_bl_mrf.F lines 305-311) ------------------ #
RLAM = 150.0
PRMIN = 0.5
PRMAX = 4.0
XKZMIN = 0.01
XKZMAX = 1000.0
RIMIN = -100.0
BRCR = 0.5
CFAC = 7.8
PFAC = 2.0
SFCFRAC = 0.1
CKZ = 0.001
ZFMIN = 1.0e-8
APHI5 = 5.0
APHI16 = 16.0
GAMCRT = 3.0
GAMCRQ = 2.0e-3
XKA = 2.4e-5


def _flip(a: jax.Array) -> jax.Array:
    """bottom-up (k=0 surface) <-> top-down (index 0 = model top)."""
    return a[::-1]


def _tridi2(al: jax.Array, cm: jax.Array, cu: jax.Array, r1: jax.Array, r2: jax.Array):
    """Traceable WRF ``TRIDI2`` (module_bl_mrf.F:1291), bottom-up 0-based storage
    (index 0 = surface row, matching the Fortran ``KK=kme-K`` assembly).

    CRITICAL index detail: in WRF, ``TRIDI2`` declares ``CL`` with lower bound
    ``kts+1`` while the caller's ``AL`` has lower bound ``kts``. Fortran argument
    association therefore makes ``CL(K) == AL(K-1)`` -- i.e. the sub-diagonal of
    row K is the value the assembler wrote at ``AL(K-1)`` (consistent with
    ``AD(K)=1-AL(K-1)``). We reproduce that here: ``cl[k] = al[k-1]``.

    Solves two RHS systems sharing the same tridiagonal matrix:
      al = assembler sub-diagonal (al[k] belongs to row k+1; al[n-1] unused),
      cm = main diagonal, cu = super-diagonal (cu[n-1] unused).
    Returns (a1, a2) solutions.
    """
    n = cm.shape[0]
    # cl[k] (sub-diagonal of row k) = al[k-1]; cl[0] unused.
    cl = jnp.concatenate([al[:1] * 0.0, al[:-1]])
    au = jnp.zeros(n, dtype=jnp.float64)
    a1 = jnp.zeros(n, dtype=jnp.float64)
    a2 = jnp.zeros(n, dtype=jnp.float64)

    fk0 = 1.0 / cm[0]
    au = au.at[0].set(fk0 * cu[0])
    a1 = a1.at[0].set(fk0 * r1[0])
    a2 = a2.at[0].set(fk0 * r2[0])

    # forward sweep k = 1 .. n-2
    def fwd(k, carry):
        au, a1, a2 = carry
        fk = 1.0 / (cm[k] - cl[k] * au[k - 1])
        au = au.at[k].set(fk * cu[k])
        a1 = a1.at[k].set(fk * (r1[k] - cl[k] * a1[k - 1]))
        a2 = a2.at[k].set(fk * (r2[k] - cl[k] * a2[k - 1]))
        return au, a1, a2

    au, a1, a2 = jax.lax.fori_loop(1, n - 1, fwd, (au, a1, a2))

    # last row k = n-1 (no super-diagonal contribution stored)
    fk = 1.0 / (cm[n - 1] - cl[n - 1] * au[n - 2])
    a1 = a1.at[n - 1].set(fk * (r1[n - 1] - cl[n - 1] * a1[n - 2]))
    a2 = a2.at[n - 1].set(fk * (r2[n - 1] - cl[n - 1] * a2[n - 2]))

    # back substitution k = n-2 .. 0
    def bwd(j, carry):
        a1, a2 = carry
        k = n - 2 - j
        a1 = a1.at[k].set(a1[k] - au[k] * a1[k + 1])
        a2 = a2.at[k].set(a2[k] - au[k] * a2[k + 1])
        return a1, a2

    a1, a2 = jax.lax.fori_loop(0, n - 1, bwd, (a1, a2))
    return a1, a2


def _pbl_first_guess(thvx, ux, vx, za, zq, br, thermal_in, brcr_val, n):
    """WRF MRF PBL-height first-guess scan (module_bl_mrf.F:815-842).

    Arrays are top-down 0-based of length ``n`` (index 0 = model top, n-1 =
    surface ``KL``). WRF scans ``DO K=KLM,KLPBL,-1`` i.e. top-down 1-based K from
    ``n-1`` down to ``1`` (0-based ``n-2`` down to ``0``), accumulating ``brup``
    and freezing ``stable`` once ``brup > brcr``. Returns (pbl, kpbl, brint) with
    ``kpbl`` a top-down 0-based index.
    """
    thvx_kl = thvx[n - 1]

    def body(j, carry):
        brup, brdn, kpbl, stable = carry
        k = (n - 2) - j  # n-2, n-3, ..., 0  (Fortran K = KLM .. KLPBL, step -1)
        spdk2 = jnp.maximum(ux[k] ** 2 + vx[k] ** 2, 1.0)
        brup_new = (thvx[k] - thermal_in) * (G * za[k] / thvx_kl) / spdk2
        not_stable = jnp.logical_not(stable)
        brdn = jnp.where(not_stable, brup, brdn)
        brup = jnp.where(not_stable, brup_new, brup)
        kpbl = jnp.where(not_stable, k, kpbl)
        stable = jnp.where(not_stable, brup_new > brcr_val, stable)
        return brup, brdn, kpbl, stable

    # initial: brup=br, kpbl=KL surface (top-down index n-1)
    init = (br, br, jnp.array(n - 1, dtype=jnp.int32), jnp.array(False))
    brup, brdn, kpbl, stable = jax.lax.fori_loop(0, n - 1, body, init)

    # PBL = ZA(K+1) + BRINT*(ZA(K)-ZA(K+1)); K=KPBL. ZA(K+1)=level just below K.
    k = kpbl
    za_k = za[k]
    za_kp1 = za[jnp.minimum(k + 1, n - 1)]
    brint = jnp.where(
        brdn >= brcr_val,
        0.0,
        jnp.where(brup <= brcr_val, 1.0, (brcr_val - brdn) / (brup - brdn)),
    )
    pbl = za_kp1 + brint * (za_k - za_kp1)
    # IF(PBL < ZQ(KPBL+1)) KPBL=KPBL+1  (interface height of the level below K)
    zq_kp1 = zq[jnp.minimum(k + 1, n)]
    kpbl = jnp.where(pbl < zq_kp1, jnp.minimum(k + 1, n - 1), k)
    return pbl, kpbl, brint


def mrf_column(u, v, t, qv, qc, p, pii, dz, z, *, psfc, znt, ust, hfx, qfx, tsk,
               gz1oz0, wspd, br, psim, psih, xland, dt):
    """One MRF PBL column. All profile inputs are bottom-up length-``n`` arrays.

    Returns dict of bottom-up tendency columns (``rublten, rvblten, rthblten,
    rqvblten, rqcblten`` -- theta tendency already divided by exner like WRF) and
    scalar diagnostics (``pblh, kpbl`` [bottom-up 0-based], ``zol, hol, regime``).
    """
    n = u.shape[0]
    kl = n - 1  # surface in top-down 0-based storage

    # --- map bottom-up inputs to Fortran top-down 0-based storage (index 0=top) #
    ux = _flip(u)
    vx = _flip(v)
    tx_in = _flip(t)
    qx = _flip(qv)
    qcx = _flip(qc)
    p_td = _flip(p)
    dz_td = _flip(dz)      # dz8w2d top-down (Fortran reorders dz8w(NK) too)
    z_td = _flip(z)        # z2d top-down

    # PS in cb (Fortran PS=PSFCPA/1000). THX from THCON with P1000/(PL*1000) where
    # PL=P2D/1000 -> THCON = (P1000/P2D)**ROVCP.
    thx = tx_in * (P1000 / p_td) ** ROVCP
    scr3 = tx_in
    qixx = jnp.zeros(n, dtype=jnp.float64)  # flag_qi=False -> QIX=0
    tvcon = 1.0 + EP1 * qx
    thvx = thx * tvcon
    scr4 = scr3 * tvcon

    # CPM at surface level (KL): CP*(1+0.8*QX(KL))
    cpm = CP * (1.0 + 0.8 * qx[kl])

    # --- interface heights ZQ (top-down). ZQ(KLP1)=0 at the surface bottom; build
    # upward via ZQ(K)=dz8w2d(K)+ZQ(K+1). In top-down storage ZQ has n+1 entries
    # with ZQ[n]=0 (surface). We compute ZQ as length n+1.
    zq = jnp.zeros(n + 1, dtype=jnp.float64)
    # ZQ[n] = 0 (surface bottom interface); fill upward: ZQ[k] = dz_td[k] + ZQ[k+1]
    def zq_body(j, zq):
        k = (n - 1) - j  # n-1 .. 0
        return zq.at[k].set(dz_td[k] + zq[k + 1])
    zq = jax.lax.fori_loop(0, n, zq_body, zq)

    za = 0.5 * (zq[:n] + zq[1:])          # ZA(K)=0.5*(ZQ(K)+ZQ(K+1)), length n
    dzq = zq[:n] - zq[1:]                  # DZQ(K)=ZQ(K)-ZQ(K+1)
    # DZA(K)=ZA(K)-ZA(K+1) for K=1..n-1 (top-down). length n-1 valid; pad last.
    dza = jnp.zeros(n, dtype=jnp.float64)
    dza = dza.at[: n - 1].set(za[: n - 1] - za[1:])

    rhox = (psfc / 1000.0) * 1000.0 / (R_D * scr4[kl])  # PS(cb)*1000/(R*SCR4(KL))
    govrth = G / thx[kl]

    # --- first-guess PBL (BRCR=0.5), then add thermal excess, re-scan, then PBL0 #
    sfcflg = br <= 0.0  # SFCFLG true unless BR>0
    thermal0 = thvx[kl]

    pbl1, kpbl1, _ = _pbl_first_guess(thvx, ux, vx, za, zq, br, thermal0, BRCR, n)

    # HOL/HOL1, PHIM/PHIH, WSCALE
    fm = gz1oz0 - psim
    fh = gz1oz0 - psih
    hol = jnp.maximum(br * fm * fm / fh, RIMIN)
    hol = jnp.where(sfcflg, jnp.minimum(hol, -ZFMIN), jnp.maximum(hol, ZFMIN))
    zl1 = za[kl]
    hol1 = hol * pbl1 / zl1 * SFCFRAC
    hol_out = -hol * pbl1 / zl1
    phim = jnp.where(sfcflg, (1.0 - APHI16 * hol1) ** (-0.25), 1.0 + APHI5 * hol1)
    phih = jnp.where(sfcflg, (1.0 - APHI16 * hol1) ** (-0.5), phim)
    wscale = ust / phim
    wscale = jnp.minimum(wscale, ust * APHI16)
    wscale = jnp.maximum(wscale, ust / APHI5)

    # surface variables for unstable PBL: HGAMT/HGAMQ + thermal excess
    over_water = (xland - 1.5) >= 0.0
    gamfac = CFAC / rhox / wscale
    hgamt = jnp.minimum(gamfac * hfx / cpm, GAMCRT)
    hgamq = jnp.minimum(gamfac * qfx, GAMCRQ)
    hgamq = jnp.where(over_water, 0.0, hgamq)
    vpert = hgamt + EP1 * thx[kl] * hgamq
    vpert = jnp.minimum(vpert, GAMCRT)
    thermal_unstable = thermal0 + jnp.maximum(vpert, 0.0)
    hgamt = jnp.maximum(hgamt, 0.0)
    hgamq = jnp.maximum(hgamq, 0.0)

    pblflg = sfcflg  # PBLFLG set false in the else branch (BR>0)

    # WRF (887-892): IF(PBLFLG) KPBL=KL, PBL=ZQ(KL) before the enhanced re-scan;
    # then re-scan with the thermal excess (lines 896-928).
    thermal_used = jnp.where(pblflg, thermal_unstable, thermal0)
    pbl2, kpbl2, _ = _pbl_first_guess(thvx, ux, vx, za, zq, br, thermal_used, BRCR, n)
    pbl = jnp.where(pblflg, pbl2, pbl1)
    kpbl = jnp.where(pblflg, kpbl2, kpbl1)
    # IF(KPBL<=1) PBLFLG=.FALSE.  (Fortran 1-based KPBL<=1 = top-down 0-based <=0)
    pblflg = pblflg & (kpbl > 0)

    # PBL0 diagnostic with effective BRCR=0 (stable test BRUP>0.0); the PBL0 scan
    # runs only IF(PBLFLG). Otherwise PBL0/KPBL0 keep their initialization
    # (WRF lines 804-805): KPBL0=KL, PBL0=ZQ(KL) = the surface-layer top interface.
    pbl0_s, kpbl0_s, _ = _pbl_first_guess(thvx, ux, vx, za, zq, br, thermal_used, 0.0, n)
    pbl0 = jnp.where(pblflg, pbl0_s, zq[kl])
    kpbl0 = jnp.where(pblflg, kpbl0_s, jnp.array(kl, dtype=jnp.int32))

    # --- diffusion coefficients ------------------------------------------------ #
    idx = jnp.arange(n)
    # below PBL (K from kte..KLPBL i.e. all, where KPBL<K in top-down means
    # index > kpbl): nonlocal-K profile
    prnum_pbl = jnp.clip(phih / phim + CFAC * KARMAN * SFCFRAC, PRMIN, PRMAX)
    # ZFAC uses ZQ(K); below-PBL valid where idx>kpbl
    zfac = jnp.maximum(1.0 - (zq[:n] - zl1) / (pbl - zl1), ZFMIN)
    xkzo_below = CKZ * dza  # CKZ*DZA(K-1); approximate index alignment below
    # WRF uses DZA(I,K-1); in top-down 0-based that is dza[k-1]; build shifted.
    dza_km1 = jnp.concatenate([dza[:1], dza[:-1]])
    xkzo_b = CKZ * dza_km1
    xkzm_below = xkzo_b + wscale * KARMAN * zq[:n] * zfac ** PFAC
    xkzh_below = xkzm_below / prnum_pbl
    xkzm_below = jnp.clip(xkzm_below, XKZMIN, XKZMAX)
    xkzh_below = jnp.clip(xkzh_below, XKZMIN, XKZMAX)
    below_mask = idx > kpbl

    # free atmosphere (K from kts+1..kte, where K<=KPBL i.e. idx<=kpbl, idx>=1)
    ux_km1 = jnp.concatenate([ux[:1], ux[:-1]])
    vx_km1 = jnp.concatenate([vx[:1], vx[:-1]])
    thvx_km1 = jnp.concatenate([thvx[:1], thvx[:-1]])
    ss = ((ux_km1 - ux) ** 2 + (vx_km1 - vx) ** 2) / (dza_km1 ** 2) + 1.0e-9
    ri = govrth * (thvx_km1 - thvx) / (ss * dza_km1)
    # IMVDIF in-cloud branch (qc>0.01e-3 both levels) -- here qc/qi profiles are
    # typically zero in the oracle cases; transcribe faithfully anyway.
    qx_km1 = jnp.concatenate([qx[:1], qx[:-1]])
    scr3_km1 = jnp.concatenate([scr3[:1], scr3[:-1]])
    qcx_km1 = jnp.concatenate([qcx[:1], qcx[:-1]])
    qix_km1 = jnp.concatenate([qixx[:1], qixx[:-1]])
    incloud = ((qcx + qixx) > 0.01e-3) & ((qcx_km1 + qix_km1) > 0.01e-3)
    qmean = 0.5 * (qx + qx_km1)
    tmean = 0.5 * (scr3 + scr3_km1)
    alph = XLV * qmean / R_D / tmean
    chi = XLV * XLV * qmean / CP / R_V / tmean / tmean
    ri_cloud = (1.0 + alph) * (ri - G * G / ss / tmean / CP * ((chi - alph) / (1.0 + chi)))
    ri = jnp.where(incloud, ri_cloud, ri)
    zk = KARMAN * zq[:n]
    rl2 = (zk * RLAM / (RLAM + zk)) ** 2
    dk = rl2 * jnp.sqrt(ss)
    xkzo_f = CKZ * dza_km1
    sri = jnp.sqrt(jnp.maximum(-ri, 0.0))
    xkzm_unst = xkzo_f + dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.746 * sri))
    xkzh_unst = xkzo_f + dk * (1.0 + 8.0 * (-ri) / (1.0 + 1.286 * sri))
    xkzh_stab = xkzo_f + dk / (1.0 + 5.0 * ri) ** 2
    prnum_f = jnp.minimum(1.0 + 2.1 * ri, PRMAX)
    xkzm_stab = (xkzh_stab - xkzo_f) * prnum_f + xkzo_f
    unstable = ri < 0.0
    xkzm_free = jnp.where(unstable, xkzm_unst, xkzm_stab)
    xkzh_free = jnp.where(unstable, xkzh_unst, xkzh_stab)
    xkzm_free = jnp.clip(xkzm_free, XKZMIN, XKZMAX)
    xkzh_free = jnp.clip(xkzh_free, XKZMIN, XKZMAX)
    free_mask = (idx <= kpbl) & (idx >= 1)

    xkzm = jnp.where(below_mask, xkzm_below, jnp.where(free_mask, xkzm_free, 0.0))
    xkzh = jnp.where(below_mask, xkzh_below, jnp.where(free_mask, xkzh_free, 0.0))

    dt4 = 2.0 * dt
    rdt = 1.0 / dt4

    # --- assemble + solve heat/moisture tridiagonal --------------------------- #
    # Fortran builds matrices in "KK = kme - K" = bottom-up index. We assemble in
    # bottom-up 0-based arrays (index 0 = surface), then solve, mapping cleanly.
    # bottom-up views:
    scr3_bu = _flip(scr3)
    qx_bu = _flip(qx)
    qcx_bu = _flip(qcx)
    qix_bu = _flip(qixx)
    ux_bu = _flip(ux)
    vx_bu = _flip(vx)
    dz8w_bu = _flip(dz_td)
    z_bu = _flip(z_td)
    xkzh_bu = _flip(xkzh)
    xkzm_bu = _flip(xkzm)
    dza_km1_bu = _flip(dza_km1)
    pblflg_b = pblflg
    # KPBL in bottom-up 0-based = (n-1) - kpbl_topdown
    kpbl_bu = (n - 1) - kpbl

    # Per-interface coefficients indexed by bottom-up LOWER-row r (r=0..n-2):
    #   interface couples row r (Fortran KK, top-down K) and row r+1 (KK+1, K-1).
    #   DTODSD = DT4/dz8w2d(K)   = dt4 / dz(lower row r)   = dt4 / dz8w_bu[r]
    #   DTODSU = DT4/dz8w2d(K-1) = dt4 / dz(upper row r+1) = dt4 / dz8w_bu[r+1]
    #   DSIG   = -(z2d(K)-z2d(K-1)) = z(upper)-z(lower) = z_bu[r+1]-z_bu[r]
    #   RDZ    = 1/DZA(K-1) ; DZA top-down K-1 = za(K-1)-za(K) = za(upper)-za(lower)
    #            in bottom-up = za_bu[r+1]-za_bu[r]
    #   XKZH(K)/XKZM(K) at top-down level K = lower row r = xkz*_bu[r]
    za_bu = _flip(za)
    r_idx = jnp.arange(n)
    dtodsd_i = dt4 / dz8w_bu                       # at lower row r
    dtodsu_i = dt4 / jnp.concatenate([dz8w_bu[1:], dz8w_bu[-1:]])  # at upper row r+1
    dsig_i = jnp.concatenate([z_bu[1:] - z_bu[:-1], jnp.zeros(1)])
    rdz_i = 1.0 / jnp.concatenate([za_bu[1:] - za_bu[:-1], jnp.ones(1)])
    # WRF nonlocal switch: IF(PBLFLG .AND. KPBL(td) < K(td)). With K(td,1based)=n-r
    # and KPBL(td,1based)=kpbl+1, this becomes r < (n-1-kpbl) = kpbl_bu (i.e. the
    # interface lower-row r lies BELOW the PBL top).
    nonlocal_on = pblflg_b & (r_idx < kpbl_bu)

    # --- heat/moisture assembly ------------------------------------------------ #
    ad = jnp.zeros(n, dtype=jnp.float64)
    al = jnp.zeros(n, dtype=jnp.float64)
    au = jnp.zeros(n, dtype=jnp.float64)
    a1 = jnp.zeros(n, dtype=jnp.float64)
    a2 = jnp.zeros(n, dtype=jnp.float64)
    # surface row (Fortran KK=1, bottom-up r=0): AD(1)=1; A1(1)=SCR3(KL)+HFX flux;
    # A2(1)=QX(KL)+QFX flux. ZQ(KL) top-down surface interface = surface-layer dz.
    ad = ad.at[0].set(1.0)
    a1 = a1.at[0].set(scr3_bu[0] + hfx / (rhox * cpm) / zq[kl] * dt4)
    a2 = a2.at[0].set(qx_bu[0] + qfx / rhox / zq[kl] * dt4)

    def hm_body(j, carry):
        ad, al, au, a1, a2 = carry
        r = j  # bottom-up lower-row index, 0 .. n-2
        dtodsd = dtodsd_i[r]
        dtodsu = dtodsu_i[r]
        dsig = dsig_i[r]
        rdz = rdz_i[r]
        xkzh_k = xkzh_bu[r]
        nl = nonlocal_on[r]
        dsdzt = jnp.where(
            nl,
            dsig * xkzh_k * rdz * (G / CP - hgamt / pbl),
            dsig * xkzh_k * rdz * (G / CP),
        )
        dsdzq_nl = dsig * xkzh_k * rdz * (-hgamq / pbl)
        dsdz2 = dsig * xkzh_k * rdz * rdz
        au = au.at[r].set(-dtodsd * dsdz2)
        al = al.at[r].set(-dtodsu * dsdz2)
        ad = ad.at[r].add(-au[r])
        ad = ad.at[r + 1].set(1.0 - al[r])
        a1 = a1.at[r].add(dtodsd * dsdzt)
        a1 = a1.at[r + 1].set(scr3_bu[r + 1] - dtodsu * dsdzt)
        a2 = a2.at[r].set(jnp.where(nl, a2[r] + dtodsd * dsdzq_nl, a2[r]))
        a2 = a2.at[r + 1].set(jnp.where(nl, qx_bu[r + 1] - dtodsu * dsdzq_nl, qx_bu[r + 1]))
        return ad, al, au, a1, a2

    ad, al, au, a1, a2 = jax.lax.fori_loop(0, n - 1, hm_body, (ad, al, au, a1, a2))
    t_new, q_new = _tridi2(al, ad, au, a1, a2)
    ttend = (t_new - scr3_bu) * rdt
    qtend = (q_new - qx_bu) * rdt

    # --- momentum assembly ------------------------------------------------------ #
    wspd1 = jnp.sqrt(ux[kl] ** 2 + vx[kl] ** 2) + 1.0e-9
    adm = jnp.zeros(n, dtype=jnp.float64)
    alm = jnp.zeros(n, dtype=jnp.float64)
    aum = jnp.zeros(n, dtype=jnp.float64)
    a1m = jnp.zeros(n, dtype=jnp.float64)
    a2m = jnp.zeros(n, dtype=jnp.float64)
    drag = ust * ust / zq[kl] * dt4 * (wspd1 / wspd) ** 2
    adm = adm.at[0].set(1.0)
    a1m = a1m.at[0].set(ux_bu[0] - ux_bu[0] / wspd1 * drag)
    a2m = a2m.at[0].set(vx_bu[0] - vx_bu[0] / wspd1 * drag)

    def mom_body(j, carry):
        adm, alm, aum, a1m, a2m = carry
        r = j
        dtodsd = dtodsd_i[r]
        dtodsu = dtodsu_i[r]
        dsig = dsig_i[r]
        rdz = rdz_i[r]
        dsdz2 = dsig * xkzm_bu[r] * rdz * rdz
        aum = aum.at[r].set(-dtodsd * dsdz2)
        alm = alm.at[r].set(-dtodsu * dsdz2)
        adm = adm.at[r].add(-aum[r])
        adm = adm.at[r + 1].set(1.0 - alm[r])
        a1m = a1m.at[r + 1].set(ux_bu[r + 1])
        a2m = a2m.at[r + 1].set(vx_bu[r + 1])
        return adm, alm, aum, a1m, a2m

    adm, alm, aum, a1m, a2m = jax.lax.fori_loop(0, n - 1, mom_body, (adm, alm, aum, a1m, a2m))
    u_new, v_new = _tridi2(alm, adm, aum, a1m, a2m)
    utend = (u_new - ux_bu) * rdt
    vtend = (v_new - vx_bu) * rdt

    # --- cloud (qc) assembly ---------------------------------------------------- #
    adc = jnp.zeros(n, dtype=jnp.float64)
    alc = jnp.zeros(n, dtype=jnp.float64)
    auc = jnp.zeros(n, dtype=jnp.float64)
    a1c = jnp.zeros(n, dtype=jnp.float64)
    a2c = jnp.zeros(n, dtype=jnp.float64)
    adc = adc.at[0].set(1.0)
    a1c = a1c.at[0].set(qcx_bu[0])
    a2c = a2c.at[0].set(qix_bu[0])

    def cloud_body(j, carry):
        adc, alc, auc, a1c, a2c = carry
        r = j
        dtodsd = dtodsd_i[r]
        dtodsu = dtodsu_i[r]
        dsig = dsig_i[r]
        rdz = rdz_i[r]
        a1c = a1c.at[r + 1].set(qcx_bu[r + 1])
        a2c = a2c.at[r + 1].set(qix_bu[r + 1])
        dsdz2 = dsig * xkzh_bu[r] * rdz * rdz
        auc = auc.at[r].set(-dtodsd * dsdz2)
        alc = alc.at[r].set(-dtodsu * dsdz2)
        adc = adc.at[r].add(-auc[r])
        adc = adc.at[r + 1].set(1.0 - alc[r])
        return adc, alc, auc, a1c, a2c

    adc, alc, auc, a1c, a2c = jax.lax.fori_loop(0, n - 1, cloud_body, (adc, alc, auc, a1c, a2c))
    qc_new, _qi_new = _tridi2(alc, adc, auc, a1c, a2c)
    qctend = (qc_new - qcx_bu) * rdt

    # --- pack tendencies bottom-up; theta tendency = ttend / exner (WRF wrapper) #
    pii_bu = pii  # pii is bottom-up exner (input)
    rthblten = (ttend) / pii_bu
    return {
        "rublten": utend,
        "rvblten": vtend,
        "rthblten": rthblten,
        "rqvblten": qtend,
        "rqcblten": qctend,
        "pblh": pbl0,
        "kpbl": kpbl_bu,
        "zol": jnp.zeros((), dtype=jnp.float64),  # ZOL stays 0 in this code path
        "hol": hol_out,
    }


def mrf_columns(u, v, t, qv, qc, p, pii, dz, z, *, psfc, znt, ust, hfx, qfx, tsk,
                gz1oz0, wspd, br, psim, psih, xland, dt):
    """vmap-batched MRF over a leading ``(ncol,)`` axis. Profiles are
    ``(ncol, n)``; surface forcings are ``(ncol,)``. Returns batched dict."""

    def one(u, v, t, qv, qc, p, pii, dz, z,
            psfc, znt, ust, hfx, qfx, tsk, gz1oz0, wspd, br, psim, psih, xland):
        return mrf_column(
            u, v, t, qv, qc, p, pii, dz, z,
            psfc=psfc, znt=znt, ust=ust, hfx=hfx, qfx=qfx, tsk=tsk,
            gz1oz0=gz1oz0, wspd=wspd, br=br, psim=psim, psih=psih,
            xland=xland, dt=dt,
        )

    return jax.vmap(one)(
        u, v, t, qv, qc, p, pii, dz, z,
        psfc, znt, ust, hfx, qfx, tsk, gz1oz0, wspd, br, psim, psih, xland,
    )
