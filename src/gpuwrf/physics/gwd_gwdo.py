"""WRF-faithful orographic gravity-wave drag + flow-blocking (``gwd_opt=1``).

A vectorised JAX port of the WRF CCPP ``bl_gwdo_run`` kernel
(``phys/physics_mmm/bl_gwdo.F90`` in a pristine WRF v4 tree; the driver wrapper
is ``phys/module_bl_gwdo.F``). The scheme is the Kim-GWDO of Choi & Hong (2015):
the traditional upper-level wave breaking on the mountain *variance* (Alpert
1988), the enhanced lower-tropospheric breaking from mountain convexity and
asymmetry (Kim & Arakawa 1995), and the low-level flow-blocking drag with
orographic anisotropy (Hyun-Joo Choi, 2015).

References (verbatim from the WRF source header):
    Choi and Hong (2015), J. Geophys. Res.
    Hong et al. (2008), Wea. Forecasting
    Kim and Doyle (2005), Q. J. R. Meteor. Soc.
    Kim and Arakawa (1995), J. Atmos. Sci.
    Alpert et al. (1988), NWP conference
    Hong (1999), NCEP office note 424

Sub-grid orography statistics dependency
----------------------------------------
GWDO is driven by ten static 2-D fields produced by WPS GEOGRID and carried in
``wrfinput`` (see ``init/metgrid_schema.py``; Registry package ``gwd_used_1``):

    var2d  <- geo_em ``VAR``  standard deviation of subgrid orography (m)
    oc1    <- geo_em ``CON``  orographic convexity
    oa2d1..4 <- ``OA1..4``    directional asymmetry (W/S/SW/NW)
    ol2d1..4 <- ``OL1..4``    directional effective length

These are NOT prognostic ``State`` fields; like the RRTMG terrain-radiation
statics they are a per-run static bundle (:class:`GWDOStatics`) supplied to the
adapter. ``sina``/``cosa`` are the grid-rotation sines/cosines (0/1 for an
unrotated lat-lon / Mercator domain) and ``dxmeter`` is the grid spacing in m.

Faithfulness notes
------------------
* All physical constants match the WRF call (``module_pbl_driver.F``):
  ``g=9.81``, ``r_d=287``, ``cp=7*r_d/2``, ``r_v=461.6``, ``ep_1=r_v/r_d-1``,
  ``pi=3.141592653`` -- these are the WRF *physics* values (note ``g=9.81``,
  not the ``9.80665`` used by the dycore geopotential), kept so the scheme's
  internal balances reproduce WRF bit-for-bit modulo fp ordering.
* The per-column variable upper bound ``kbl(i)`` of the Fortran do-loops is
  reproduced exactly with boolean masks: every ``if (k.lt.kbl(i))`` /
  ``if (k.le.kbl(i))`` / ``if (k.ge.kbl(i))`` guard becomes a ``jnp.where`` on a
  ``(B, K)`` mask, so the vectorised reduction is identical to the scalar loop.
* The downward flow-blocking scan (``do k = kte, kpblmin, -1`` with a
  first-trigger latch) is the one genuinely sequential part; it is reproduced
  with a single reverse ``lax.scan`` over levels that latches ``kblk`` on the
  first level where ``fbdpe >= fbdke`` (matching the Fortran ``kblk.eq.0`` gate).

The kernel operates on a flat batch of columns ``B = ny*nx`` with the vertical
axis trailing (``(B, K)``), the project's column convention. ``del`` (layer
mass) and the saturation/Lindzen vertical stress recursion follow the Fortran
line-for-line; see inline ``WRF:`` line anchors.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# --- WRF physics constants (module_pbl_driver.F gwdo call) ------------------- #
G_ = 9.81
RD_ = 287.0
CP_ = 7.0 * RD_ / 2.0          # 1004.5
RV_ = 461.6
EP1_ = RV_ / RD_ - 1.0         # ~0.6077
PI_ = 3.141592653

# --- bl_gwdo_run parameters (verbatim) --------------------------------------- #
RIC = 0.25            # critical Richardson number
DW2MIN = 1.0
RIMIN = -100.0
BNV2MIN = 1.0e-5
EFMIN = 0.0
EFMAX = 10.0
XL = 4.0e4
GMAX = 1.0
VELEPS = 1.0
FRC = 1.0
CE = 0.8
CG = 0.5
KPBLMIN = 2           # Fortran 1-based; level index 2 == Python column index 1

# flow-blocking drag parameters
FRMAX = 10.0
OLMIN = 1.0e-5
ODMIN = 0.1
ODMAX = 10.0

# WRF ``data nwdir/6,7,5,8,2,3,1,4/`` (1-based wind-octant -> direction index).
# Stored 0-based: nwdir[idir0] gives the 1-based nwd used downstream.
_NWDIR = jnp.array([6, 7, 5, 8, 2, 3, 1, 4], dtype=jnp.int32)
_MDIR = 8


class GWDOStatics(NamedTuple):
    """Per-run sub-grid orography statistics on mass points, flat batch ``(B,)``.

    Built from the WPS/geo_em static fields carried in ``wrfinput`` (see
    ``metgrid_schema.py``):

        var   : std dev of subgrid orography (m)        <- geo_em ``VAR``
        oc1   : orographic convexity                    <- geo_em ``CON``
        oa1..4: directional asymmetry (W/S/SW/NW)        <- ``OA1..4``
        ol1..4: directional effective length            <- ``OL1..4``
        sina  : sin(grid rotation angle)  (0 for unrotated)
        cosa  : cos(grid rotation angle)  (1 for unrotated)
        dxmeter: grid spacing (m), per column
    """

    var: jax.Array
    oc1: jax.Array
    oa1: jax.Array
    oa2: jax.Array
    oa3: jax.Array
    oa4: jax.Array
    ol1: jax.Array
    ol2: jax.Array
    ol3: jax.Array
    ol4: jax.Array
    sina: jax.Array
    cosa: jax.Array
    dxmeter: jax.Array


class GWDOColumnState(NamedTuple):
    """Atmospheric column inputs for GWDO, flat batch ``(B, K)``.

    uproj/vproj are the projection-relative (grid) winds at mass points (m/s);
    t1 temperature (K); q1 water-vapour mixing ratio (kg/kg); prsl mid-layer
    pressure (Pa); prsi interface pressure (Pa), shape ``(B, K+1)``; prslk the
    Exner function (dimensionless); zl geopotential height of mid-layers (m).
    """

    uproj: jax.Array
    vproj: jax.Array
    t1: jax.Array
    q1: jax.Array
    prsl: jax.Array
    prsi: jax.Array
    prslk: jax.Array
    zl: jax.Array


class GWDOTendency(NamedTuple):
    """GWDO output: grid-relative wind tendencies (m/s^2) + diagnostics.

    rublten/rvblten : grid-relative u/v tendency (m/s^2), shape ``(B, K)``.
    dtaux3d/dtauy3d : grid-relative diagnosed GWD stress-tendency, ``(B, K)``.
    dusfcg/dvsfcg   : vertically-integrated GW surface stress (N/m^2), ``(B,)``.
    """

    rublten: jax.Array
    rvblten: jax.Array
    dtaux3d: jax.Array
    dtauy3d: jax.Array
    dusfcg: jax.Array
    dvsfcg: jax.Array


def gwdo_columns(
    column: GWDOColumnState,
    statics: GWDOStatics,
    deltim: float,
) -> GWDOTendency:
    """Faithful vectorised ``bl_gwdo_run``: GWD+flow-blocking tendencies.

    ``column`` profiles are ``(B, K)`` (interface pressure ``prsi`` is
    ``(B, K+1)``); ``statics`` are ``(B,)``; ``deltim`` the physics time step (s).
    Returns the grid-relative momentum tendencies (already de-rotated back from
    earth-relative, exactly as WRF's final rotation block).
    """

    dtype = column.t1.dtype
    B, K = column.t1.shape

    g_ = jnp.asarray(G_, dtype)
    cp_ = jnp.asarray(CP_, dtype)
    rd_ = jnp.asarray(RD_, dtype)
    fv_ = jnp.asarray(EP1_, dtype)
    pi_ = jnp.asarray(PI_, dtype)

    var = statics.var.astype(dtype)
    oc1 = statics.oc1.astype(dtype)
    sina = statics.sina.astype(dtype)
    cosa = statics.cosa.astype(dtype)
    dxmeter = statics.dxmeter.astype(dtype)

    uproj = column.uproj.astype(dtype)
    vproj = column.vproj.astype(dtype)
    t1 = column.t1.astype(dtype)
    q1 = column.q1.astype(dtype)
    prsl = column.prsl.astype(dtype)
    prsi = column.prsi.astype(dtype)
    prslk = column.prslk.astype(dtype)
    zl = column.zl.astype(dtype)

    # Level-index helpers. Fortran is 1-based: kts=1 (Python 0), kte=K (Python K-1),
    # kpblmin=2 (Python 1). ``k`` below is the Python 0-based level index.
    xlinv0 = jnp.asarray(1.0 / XL, dtype)

    # --- flow-blocking grid lengths (WRF:223-234) ---------------------------- #
    delx = dxmeter
    dely = dxmeter
    dxy4_1 = delx
    dxy4_2 = dely
    dxy4_3 = jnp.sqrt(delx * delx + dely * dely)
    dxy4_4 = dxy4_3
    dxy4 = jnp.stack([dxy4_1, dxy4_2, dxy4_3, dxy4_4], axis=-1)  # (B,4)
    dxy4p = jnp.stack([dxy4_2, dxy4_1, dxy4_4, dxy4_3], axis=-1)  # (B,4)
    cleff = dxmeter

    # --- per-level diagnostics (WRF:254-273) --------------------------------- #
    vtj = t1 * (1.0 + fv_ * q1)        # virtual temperature
    vtk = vtj / prslk
    rho = (1.0 / rd_) * prsl / vtj
    delp = prsi[:, :-1] - prsi[:, 1:]  # del(i,k) positive layer mass (Pa)

    # earth-relative winds (WRF:269-270)
    u1 = uproj * cosa[:, None] - vproj * sina[:, None]
    v1 = uproj * sina[:, None] + vproj * cosa[:, None]

    # --- low-level "mountain top" index kbl (WRF:276-308) -------------------- #
    zlowtop = 2.0 * var  # (B,)
    # klowtop: first k (>=1 in Python) where zl(k)-zl(0) >= zlowtop, then +1 (and
    # +1 again for the Fortran 1-based -> 0-based shift cancels: Fortran sets
    # klowtop=k+1 (1-based). Python equivalent index = (k+1)-1 = k+1-? Carefully:
    # Fortran k loops kts+1..kte i.e. 1-based 2..K -> Python 1..K-1. It sets
    # klowtop = k+1 (1-based). We keep kbl as a 1-based Fortran index and only use
    # it through ``k < kbl`` style comparisons against the *1-based* level number,
    # so build a 1-based level number array.
    klevel_1b = jnp.arange(1, K + 1, dtype=jnp.int32)[None, :]  # (1,K) 1-based
    dz_from_base = zl - zl[:, :1]                               # (B,K)
    # candidate at Fortran k (1-based) for k in 2..K: klowtop_candidate = (k+1)
    cond = (klevel_1b >= 2) & (dz_from_base >= zlowtop[:, None]) & (zlowtop[:, None] > 0.0)
    # first True along K; klowtop = (k_fortran + 1). k_fortran == klevel_1b.
    any_hit = jnp.any(cond, axis=1)
    first_k_1b = jnp.argmax(cond.astype(jnp.int32), axis=1)  # 0-based position
    first_k_fortran = first_k_1b + 1                          # -> 1-based level number
    klowtop = jnp.where(any_hit, first_k_fortran + 1, 0).astype(jnp.int32)  # (B,)

    kpblmax = K  # Fortran kte
    kbl = jnp.clip(klowtop, KPBLMIN, kpblmax)  # (B,) 1-based; min/max(klowtop,..)
    # NOTE WRF does max(min(kbl,kpblmax),kpblmin): clip(.,kpblmin,kpblmax) matches.

    kbl_b = kbl[:, None]  # (B,1) 1-based mountain-top index

    # --- low-level PBL averages (WRF:305-322) -------------------------------- #
    # delks = 1/(prsi(1)-prsi(kbl)); kbl is 1-based -> prsi index kbl-1 (0-based)
    kbl0 = kbl - 1  # 0-based interface/level index of the mountain top
    prsi_kbl = jnp.take_along_axis(prsi, kbl0[:, None], axis=1)[:, 0]  # prsi(kbl)
    prsl_kbl = jnp.take_along_axis(prsl, kbl0[:, None], axis=1)[:, 0]  # prsl(kbl)
    delks = 1.0 / (prsi[:, 0] - prsi_kbl)
    delks1 = 1.0 / (prsl[:, 0] - prsl_kbl)

    # mask: k (1-based) < kbl  <=> klevel_1b < kbl_b
    below_kbl = klevel_1b < kbl_b  # (B,K) bool
    rdelks_lev = delp * delks[:, None]  # del(k)*delks (== rcsks)
    ubar = jnp.sum(jnp.where(below_kbl, rdelks_lev * u1, 0.0), axis=1)
    vbar = jnp.sum(jnp.where(below_kbl, rdelks_lev * v1, 0.0), axis=1)
    rhobar = jnp.sum(jnp.where(below_kbl, rdelks_lev * rho, 0.0), axis=1)

    # --- low-level wind direction -> oa/ol/olp/od/dxy/dxyp (WRF:329-362) ------ #
    oa4 = jnp.stack([statics.oa1, statics.oa2, statics.oa3, statics.oa4], axis=-1).astype(dtype)
    ol4 = jnp.stack([statics.ol1, statics.ol2, statics.ol3, statics.ol4], axis=-1).astype(dtype)
    fdir = _MDIR / (2.0 * PI_)
    wdir = jnp.arctan2(ubar, vbar) + pi_
    idir = jnp.mod(jnp.round(fdir * wdir).astype(jnp.int32), _MDIR)  # 0-based -> +1 below
    # Fortran idir = mod(nint(..),mdir)+1 (1-based); nwd = nwdir(idir).
    nwd = _NWDIR[idir]  # (B,) 1-based wind-direction code (idir already 0-based index)
    m = jnp.mod(nwd - 1, 4)  # (B,) 0-based component selector
    sign = (1 - 2 * ((nwd - 1) // 4)).astype(dtype)  # (1-2*int((nwd-1)/4))
    oa = sign * jnp.take_along_axis(oa4, m[:, None], axis=1)[:, 0]
    ol = jnp.take_along_axis(ol4, m[:, None], axis=1)[:, 0]
    # ol4p(1..4) = ol4(2),ol4(1),ol4(4),ol4(3); olp = ol4p(m+1)
    ol4p = jnp.stack([ol4[:, 1], ol4[:, 0], ol4[:, 3], ol4[:, 2]], axis=-1)
    olp = jnp.take_along_axis(ol4p, m[:, None], axis=1)[:, 0]
    od = olp / jnp.maximum(ol, OLMIN)
    od = jnp.clip(od, ODMIN, ODMAX)
    dxy = jnp.take_along_axis(dxy4, m[:, None], axis=1)[:, 0]
    dxyp = jnp.take_along_axis(dxy4p, m[:, None], axis=1)[:, 0]

    # --- Richardson number usqj + bnv2 (WRF:366-378) ------------------------- #
    # k loops kts..kte-1 (Python 0..K-2); usqj/bnv2 over interfaces between k,k+1.
    ti = 2.0 / (t1[:, :-1] + t1[:, 1:])
    rdz = 1.0 / (zl[:, 1:] - zl[:, :-1])
    tem1 = u1[:, :-1] - u1[:, 1:]
    tem2 = v1[:, :-1] - v1[:, 1:]
    dw2 = tem1 * tem1 + tem2 * tem2
    shr2 = jnp.maximum(dw2, DW2MIN) * rdz * rdz
    bvf2 = g_ * (g_ / cp_ + rdz * (vtj[:, 1:] - vtj[:, :-1])) * ti
    usqj_int = jnp.maximum(bvf2 / shr2, RIMIN)             # (B,K-1)
    bnv2_int = 2.0 * g_ * rdz * (vtk[:, 1:] - vtk[:, :-1]) / (vtk[:, 1:] + vtk[:, :-1])
    # pad to (B,K): index k holds the k..k+1 interface; level K-1 (top) is unused
    # by the saturation loop's bnv2(i,k) read at k up to kte-1, and the low-level
    # weighted reduction only touches k<kbl<=K. Pad with 0 (matches the Fortran
    # arrays that were zero-initialised and only filled for k=1..kte-1).
    pad = jnp.zeros((B, 1), dtype)
    usqj = jnp.concatenate([usqj_int, pad], axis=1)  # (B,K)
    bnv2 = jnp.concatenate([bnv2_int, pad], axis=1)  # (B,K)

    # --- low-level wind magnitude (WRF:382-385) ------------------------------ #
    ulow = jnp.maximum(jnp.sqrt(ubar * ubar + vbar * vbar), 1.0)
    rulow = 1.0 / ulow

    # --- velco (component of upper wind along low-level wind) (WRF:387-396) --- #
    velco = 0.5 * ((u1[:, :-1] + u1[:, 1:]) * ubar[:, None]
                   + (v1[:, :-1] + v1[:, 1:]) * vbar[:, None])  # (B,K-1)
    velco = velco * rulow[:, None]
    velco = jnp.where((velco < VELEPS) & (velco > 0.0), VELEPS, velco)
    # pad to (B,K) for uniform indexing; top entry unused.
    velco = jnp.concatenate([velco, pad], axis=1)  # (B,K)

    # --- ldrag: no-drag conditions (WRF:400-437) ----------------------------- #
    # base layer crit-level: velco(1)<=0  (Python velco[:,0])
    ldrag = velco[:, 0] <= 0.0
    # k=kpblmin..kpblmax with k<kbl : ldrag |= velco(k)<=0
    # kpblmin is 1-based level 2 -> Python index 1. Mask: (klevel_1b>=KPBLMIN)&(klevel_1b<kbl)
    drag_band = (klevel_1b >= KPBLMIN) & (klevel_1b < kbl_b)  # (B,K)
    ldrag = ldrag | jnp.any(jnp.where(drag_band, velco <= 0.0, False), axis=1)

    # --- low-level weighted-average bnv2(1), usqj(1) (WRF:417-431) ----------- #
    wtkbj = (prsl[:, 0] - prsl[:, 1]) * delks1
    bnv2_1 = wtkbj * bnv2[:, 0]
    usqj_1 = wtkbj * usqj[:, 0]
    # k=kpblmin..kpblmax, k<kbl: rdelks=(prsl(k)-prsl(k+1))*delks1; accumulate.
    # band excludes k=1 (Python 0) which is the wtkbj seed already; band starts at
    # 1-based level 2 == Python index 1. We need prsl(k)-prsl(k+1) so build on
    # 0-based k: contributions at Python index j for j in 1..K-2 with klevel<kbl.
    prsl_diff = jnp.concatenate([prsl[:, :-1] - prsl[:, 1:], pad], axis=1)  # (B,K)
    rdelks_band = prsl_diff * delks1[:, None]
    # mask: 1-based level >= 2 (Python>=1) and < kbl
    acc_band = (klevel_1b >= KPBLMIN) & (klevel_1b < kbl_b)
    bnv2_1 = bnv2_1 + jnp.sum(jnp.where(acc_band, bnv2 * rdelks_band, 0.0), axis=1)
    usqj_1 = usqj_1 + jnp.sum(jnp.where(acc_band, usqj * rdelks_band, 0.0), axis=1)

    ldrag = ldrag | (bnv2_1 <= 0.0) | (ulow == 1.0) | (var <= 0.0)

    # set all RI low-level values to the low-level value (WRF:441-445).
    # usqj(k)=usqj(1) for kpblmin..kpblmax with k<kbl. Used later only at k<kbl
    # via the saturation loop's usqj read where k>=kbl, so this only matters for
    # the k<kbl band; rebuild usqj with the low-level value in that band.
    usqj = jnp.where(acc_band, usqj_1[:, None], usqj)
    usqj = usqj.at[:, 0].set(usqj_1)  # k=1 also holds the low-level value

    not_ldrag = ~ldrag

    # --- Froude number, base stress taub (WRF:447-477) ----------------------- #
    bnv = jnp.sqrt(jnp.maximum(bnv2_1, 0.0))
    fr = bnv * rulow * var * od
    fr = jnp.minimum(fr, FRMAX)
    xn = jnp.where(not_ldrag, ubar * rulow, 0.0)
    yn = jnp.where(not_ldrag, vbar * rulow, 0.0)

    efact = (oa + 2.0) ** (CE * fr / FRC)
    efact = jnp.clip(efact, EFMIN, EFMAX)
    coefm = (1.0 + ol) ** (oa + 1.0)
    xlinv = coefm / cleff
    tem_fr = fr * fr * oc1
    gfobnv = GMAX * tem_fr / ((tem_fr + CG) * jnp.where(bnv > 0.0, bnv, 1.0))
    taub_active = xlinv * rhobar * ulow * ulow * ulow * gfobnv * efact
    taub = jnp.where(not_ldrag, taub_active, 0.0)
    # When ldrag: xn=yn=0 (already), and Fortran leaves xlinv=1/xl, coefm unused
    # in inactive columns; keep xlinv finite for inactive columns.
    xlinv = jnp.where(not_ldrag, xlinv, xlinv0)

    # --- vertical stress structure taup (WRF:481-532) ------------------------ #
    # taup has K+1 levels (interfaces). taup(k)=taub for k<=kbl (1-based).
    # Build initial taup: 1-based level index 1..K+1 (Python 0..K).
    taup_level_1b = jnp.arange(1, K + 2, dtype=jnp.int32)[None, :]  # (1,K+1)
    taup = jnp.where(taup_level_1b <= kbl_b, taub[:, None], 0.0)    # (B,K+1)

    # The saturation recursion: k from kpblmin..kte-1 (Python 1..K-2), updates
    # taup(k+1) from taup(k). icrilv latches once true. Sequential in k.
    brvf_all = jnp.sqrt(jnp.maximum(bnv2, BNV2MIN))  # (B,K) brunt-vaisala freq

    def sat_step(carry, k):
        taup_c, icrilv_c = carry  # taup_c (B,K+1), icrilv_c (B,)
        k1b = k + 1  # 1-based Fortran level number for this Python index k
        kp1 = k + 1  # Python index of the level above (Fortran kp1 = k1b+1)
        # k>=kbl ?  (Fortran ``k .ge. kbl(i)``, both 1-based)
        ge_kbl = k1b >= kbl  # (B,)
        usqj_k = usqj[:, k]
        velco_k = velco[:, k]
        # WRF:495-497 -- inside ``if (k>=kbl)``: icrilv |= (usqj<ric) | (velco<=0).
        # The .not.icrilv test below reads this UPDATED value.
        icrilv_upd = jnp.where(ge_kbl, icrilv_c | (usqj_k < RIC) | (velco_k <= 0.0), icrilv_c)
        brvf = brvf_all[:, k]
        taup_k = taup_c[:, k]
        active = ge_kbl & not_ldrag & (~icrilv_upd) & (taup_k > 0.0)
        temv = 1.0 / jnp.where(velco_k != 0.0, velco_k, 1.0)
        tem1l = coefm / jnp.maximum(dxy, 1.0) * (rho[:, kp1] + rho[:, k]) * brvf * velco_k * 0.5
        hd = jnp.sqrt(jnp.maximum(taup_k, 0.0) / jnp.where(tem1l > 0.0, tem1l, 1.0))
        fro = brvf * hd * temv
        tem2l = jnp.sqrt(jnp.maximum(usqj_k, 0.0))
        teml = 1.0 + tem2l * fro
        rim = usqj_k * (1.0 - fro) / (teml * teml)
        # saturation hypothesis branch (WRF:520-528)
        sat = rim <= RIC
        # WRF:521 ``(oa<=0).or.(kp1>=kpblmin)`` with Fortran kp1 = k1b+1 >= 3,
        # so the ``kp1>=kpblmin`` clause is always true here -> sat_allowed == sat.
        sat_allowed = sat & ((oa <= 0.0) | ((k1b + 1) >= KPBLMIN))
        temc = 2.0 + 1.0 / jnp.where(tem2l != 0.0, tem2l, 1.0)
        hd_sat = velco_k * (2.0 * jnp.sqrt(jnp.maximum(temc, 0.0)) - temc) / jnp.where(brvf > 0.0, brvf, 1.0)
        taup_kp1_sat = tem1l * hd_sat * hd_sat
        # value to write at taup(k+1): if active: (sat -> sat branch (only when
        # sat_allowed else leave); else taup(k)). When not active, leave existing.
        existing_kp1 = taup_c[:, kp1]
        new_kp1 = jnp.where(
            active,
            jnp.where(
                sat,
                jnp.where(sat_allowed, taup_kp1_sat, existing_kp1),
                taup_k,  # rim>ric: no wavebreaking, taup(k+1)=taup(k)
            ),
            existing_kp1,
        )
        taup_new = taup_c.at[:, kp1].set(new_kp1)
        return (taup_new, icrilv_upd), None

    ks = jnp.arange(1, K - 1, dtype=jnp.int32)  # Python 1..K-2 (Fortran kpblmin..kte-1)
    icrilv0 = jnp.zeros((B,), dtype=bool)
    (taup, _), _ = jax.lax.scan(sat_step, (taup, icrilv0), ks)

    # lcap == kte == K so the WRF ``if (lcap.lt.kte)`` block (537) never fires.

    # --- flow-blocking drag (WRF:541-581) ------------------------------------ #
    # Downward scan k=kte..kpblmin (Python K-1..1), latch kblk on first
    # fbdpe>=fbdke while k<=kbl. fbdpe accumulates; fbdke is the LOCAL kinetic
    # energy at the trigger level (overwritten each iter, Fortran reads it fresh).
    zl_kbl = jnp.take_along_axis(zl, kbl0[:, None], axis=1)[:, 0]  # zl(kbl)

    def fb_step(carry, k):
        fbdpe_c, kblk_c = carry  # fbdpe_c (B,), kblk_c (B,) int (0 == not set)
        k1b = k + 1  # 1-based level number
        in_band = (kblk_c == 0) & (k1b <= kbl)  # kblk.eq.0 .and. k<=kbl
        bnv2_k = bnv2[:, k]
        zl_k = zl[:, k]
        delp_k = delp[:, k]
        rho_k = rho[:, k]
        add_pe = bnv2_k * (zl_kbl - zl_k) * delp_k / g_ / jnp.where(rho_k > 0.0, rho_k, 1.0)
        fbdpe_new = jnp.where(in_band, fbdpe_c + add_pe, fbdpe_c)
        fbdke = 0.5 * (u1[:, k] ** 2 + v1[:, k] ** 2)
        trigger = in_band & (fbdpe_new >= fbdke)
        kblk_new = jnp.where(trigger, jnp.minimum(k1b, kbl), kblk_c)
        return (fbdpe_new, kblk_new), None

    ks_down = jnp.arange(K - 1, 0, -1, dtype=jnp.int32)  # K-1 .. 1 (Fortran kte..kpblmin)
    fbdpe0 = jnp.zeros((B,), dtype)
    kblk0 = jnp.zeros((B,), dtype=jnp.int32)
    (_, kblk), _ = jax.lax.scan(fb_step, (fbdpe0, kblk0), ks_down)

    has_blk = (kblk != 0) & not_ldrag  # (B,)
    kblk0idx = jnp.maximum(kblk - 1, 0)  # 0-based; guard for kblk==0 columns
    zl_kblk = jnp.take_along_axis(zl, kblk0idx[:, None], axis=1)[:, 0]
    zblk = zl_kblk - zl[:, 0]
    fbdcd = jnp.maximum(2.0 - 1.0 / jnp.where(od != 0.0, od, 1.0), 0.0)
    taufb_kts = (0.5 * rhobar * coefm / jnp.maximum(dxmeter, 1.0) ** 2
                 * fbdcd * dxyp * olp * zblk * ulow * ulow)  # taufb(kts)
    # tautem = taufb(kts)/real(kblk-kts); kts==1 (1-based) so kblk-kts = kblk-1.
    denom_fb = jnp.maximum((kblk - 1).astype(dtype), 1.0)
    tautem = taufb_kts / denom_fb
    # taufb(k) for k=kts..kblk: taufb(1)=taufb_kts; taufb(k)=taufb(k-1)-tautem
    # => taufb(k) = taufb_kts - (k-1)*tautem for 1<=k<=kblk, else 0.
    # taup(k) += taufb(k) for ALL k (Fortran ``taup(i,:) = taup(i,:)+taufb(i,:)``),
    # but taufb is nonzero only for levels 1..kblk; taufb array has K+1 entries
    # (taufb(its:ite,kts:kte+1)) and beyond kblk it stays 0.
    taufb_level_1b = jnp.arange(1, K + 2, dtype=jnp.int32)[None, :]  # (1,K+1)
    steps = (taufb_level_1b - 1).astype(dtype)
    taufb_full = taufb_kts[:, None] - steps * tautem[:, None]
    in_fb = taufb_level_1b <= kblk[:, None]
    taufb = jnp.where(has_blk[:, None] & in_fb, taufb_full, 0.0)  # (B,K+1)
    taup = taup + taufb

    # --- (g)*d(tau)/d(p) -> dtaux/dtauy + dtfac limiter (WRF:585-626) --------- #
    taud = (taup[:, 1:] - taup[:, :-1]) * g_ / jnp.where(delp != 0.0, delp, 1.0)  # (B,K)

    # dtfac: min over k=kts..kpblmax-1 with k<=kbl of |velco/(deltim*taud)| where taud!=0.
    # kpblmax-1 (1-based) == K-1 -> Python 0..K-2. mask: klevel_1b<=kbl AND taud!=0.
    dtfac_band = (klevel_1b <= kbl_b) & (klevel_1b <= (K - 1))  # k<=kbl & k<=kpblmax-1
    ratio = jnp.abs(velco / jnp.where(deltim * taud != 0.0, deltim * taud, 1.0))
    ratio = jnp.where(dtfac_band & (taud != 0.0), ratio, jnp.inf)
    dtfac = jnp.minimum(jnp.min(ratio, axis=1), 1.0)  # WRF inits dtfac=1.0

    taud = taud * dtfac[:, None]
    dtaux2d = taud * xn[:, None]
    dtauy2d = taud * yn[:, None]
    dudt = dtaux2d
    dvdt = dtauy2d
    dusfc = jnp.sum(dtaux2d * delp, axis=1) * (-1.0 / g_)
    dvsfc = jnp.sum(dtauy2d * delp, axis=1) * (-1.0 / g_)

    # --- rotate tendencies back to grid-relative (WRF:630-641) --------------- #
    ca = cosa[:, None]
    saa = sina[:, None]
    rublten = dudt * ca + dvdt * saa
    rvblten = -dudt * saa + dvdt * ca
    dtaux3d = dtaux2d * ca + dtauy2d * saa
    dtauy3d = -dtaux2d * saa + dtauy2d * ca
    dusfcg = dusfc * cosa + dvsfc * sina
    dvsfcg = -dusfc * sina + dvsfc * cosa

    return GWDOTendency(
        rublten=rublten,
        rvblten=rvblten,
        dtaux3d=dtaux3d,
        dtauy3d=dtauy3d,
        dusfcg=dusfcg,
        dvsfcg=dvsfcg,
    )


__all__ = ["GWDOStatics", "GWDOColumnState", "GWDOTendency", "gwdo_columns"]
