"""WRF old-MM5 Monin-Obukhov surface layer (``sf_sfclay_physics=91``).

This is the v0.13 Tier-3 per-scheme lane port of WRF's
``phys/module_sf_sfclay.F`` (subroutines ``SFCLAY1D`` + ``sfclayinit``). It is
the classic MM5 surface layer: a single-pass bulk-Richardson regime classifier
(4 regimes -- stable, damped-mechanical, forced-convection, free-convection)
with regime-dependent similarity ``psim``/``psih`` functions, a tabulated
Businger-Dyer free-convection branch, and the standard surface-flux closure.

It is the predecessor of the already-operational revised-MM5 scheme
(``sf_sfclay_physics=1``, ``physics.sfclay_revised_mm5``) and writes the SAME B2
surface-flux handles (``ustar``/``theta_flux``/``qv_flux``/``tau_u``/``tau_v``/
``rhosfc``/``fltv``) that ``surface_adapter`` writes, so it drops into the
operational surface-layer scan slot.

The column kernel is a pure ``jnp`` function, ``jax.jit``/``jax.vmap``-traceable
(regime selection is vectorized via ``jnp.where``; the free-convection psi table
is a precomputed ``jnp`` array indexed with clamped integer interpolation -- no
data-dependent Python control flow). It is faithful to ``SFCLAY1D`` for the
default ARW path the generated oracle exercises: ``isfflx=1``, no SCM-forced
flux, no ``isftcflx`` water heat-transfer option, no ``iz0tlnd`` land
thermal-roughness option.

Cited to ``/home/user/src/wrf_pristine/WRF/phys/module_sf_sfclay.F``
(SFCLAY1D lines 263-951; sfclayinit lines 954-969).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np


config.update("jax_enable_x64", True)


# --- SFCLAY1D parameters (lines 280-282) ---
XKA = 2.4e-5
PRT = 1.0
SALINITY_FACTOR = 0.98

# --- thermodynamic constants WRF passes into SFCLAY (module_surface_driver.F) ---
CP = 1004.0
G = 9.81
R_D = 287.0
ROVCP = R_D / CP
XLV = 2.5e6
SVP1_KPA = 0.6112
SVP2 = 17.67
SVP3_K = 29.65
SVPT0_K = 273.15
EP1 = 0.608
EP2 = 0.622
KARMAN = 0.4
P0_PA = 100000.0      # p1000mb
CZO = 0.0185
OZO = 1.59e-5
VCONVC = 1.0          # MM5 surface-layer convective velocity coefficient

# --- Businger-Dyer free-convection psi table (sfclayinit, lines 960-967) ---
# PSIMTB(N)/PSIHTB(N) for N=0..1000 with ZOLN = -N*0.01 (unstable side only).
_TABLE_N = 1000
_zoln = -np.arange(0, _TABLE_N + 1, dtype=np.float64) * 0.01
_x = (1.0 - 16.0 * _zoln) ** 0.25
_psimtb_np = (
    2.0 * np.log(0.5 * (1.0 + _x))
    + np.log(0.5 * (1.0 + _x * _x))
    - 2.0 * np.arctan(_x)
    + 2.0 * np.arctan(1.0)
)
_y = (1.0 - 16.0 * _zoln) ** 0.5
_psihtb_np = 2.0 * np.log(0.5 * (1.0 + _y))
PSIMTB = jnp.asarray(_psimtb_np, dtype=jnp.float64)
PSIHTB = jnp.asarray(_psihtb_np, dtype=jnp.float64)


def _psitb_interp(table, zol_neg):
    """WRF tabulated psi lookup: NZOL=INT(-ZOL*100), RZOL=-ZOL*100-NZOL, linear.

    ``zol_neg`` is ZOL clamped to [-9.9999, 0]. NZOL in [0, 999]; the table has
    1001 nodes so NZOL+1 is always valid.
    """

    arg = -zol_neg * 100.0
    nzol = jnp.floor(arg).astype(jnp.int32)
    nzol = jnp.clip(nzol, 0, _TABLE_N - 1)
    rzol = arg - nzol
    lo = table[nzol]
    hi = table[nzol + 1]
    return lo + rzol * (hi - lo)


def sfclay_old_mm5_column(
    ux: jax.Array,           # u at lowest level (m/s)
    vx: jax.Array,           # v at lowest level (m/s)
    t1d: jax.Array,          # temperature at lowest level (K)
    qv1d: jax.Array,         # qv at lowest level (kg/kg)
    p1d: jax.Array,          # pressure at lowest level (Pa)
    dz8w1d: jax.Array,       # dz between full levels (m)
    psfcpa: jax.Array,       # surface pressure (Pa)
    tsk: jax.Array,          # skin temperature (K)
    xland: jax.Array,        # land mask (1 land, 2 water)
    znt: jax.Array,          # roughness length (m) -- INOUT
    ust: jax.Array,          # u* (m/s) -- INOUT
    mol: jax.Array,          # T* (K) -- INOUT (carries sign for BR clamp)
    qsfc: jax.Array,         # ground saturated mixing ratio -- INOUT
    pblh: jax.Array,         # PBL height (m)
    mavail: jax.Array,       # surface moisture availability
    lakemask: jax.Array,     # lake mask (1 lake)
    dx: jax.Array,           # grid spacing (m)
    hfx_in: jax.Array,       # HFX from previous surface call (W/m^2; VCONV land)
    qfx_in: jax.Array,       # QFX from previous surface call (kg/m^2/s)
    zol_in: jax.Array | None = None,  # ZOL on entry (regime-1 keeps it; default 0)
    *,
    isfflx: int = 1,
):
    """Faithful per-column port of WRF ``SFCLAY1D`` (default ARW path).

    Returns a dict of the updated surface-layer fields. All math is fp64 when
    inputs are fp64; nothing allocates outside the trace.
    """

    psfc = psfcpa / 1000.0                       # PSFC cb (line 396)

    # Ground potential temperature (line 405).
    tgdsa = tsk
    thgb = tsk * (P0_PA / psfcpa) ** ROVCP

    # Lowest-level theta / virtual theta (lines 430-452).
    pl = p1d / 1000.0
    scr3 = t1d
    thcon = (P0_PA * 0.001 / pl) ** ROVCP
    thx = scr3 * thcon
    qx = qv1d
    tvcon = 1.0 + EP1 * qx
    thvx = thx * tvcon
    scr4 = scr3 * tvcon

    # QSFC / QGH / CPM (lines 456-466).
    e1_g = SVP1_KPA * jnp.exp(SVP2 * (tgdsa - SVPT0_K) / (tgdsa - SVP3_K))
    is_water = (xland - 1.5) >= 0.0
    e1_g = jnp.where(
        jnp.logical_and(is_water, lakemask == 0.0), e1_g * SALINITY_FACTOR, e1_g
    )
    qsfc_new = EP2 * e1_g / (psfc - e1_g)
    qsfc = jnp.where(
        jnp.logical_or(is_water, qsfc <= 0.0), qsfc_new, qsfc
    )
    e1_a = SVP1_KPA * jnp.exp(SVP2 * (t1d - SVPT0_K) / (t1d - SVP3_K))
    qgh = EP2 * e1_a / (pl - e1_a)
    cpm = CP * (1.0 + 0.8 * qx)

    # Heights / density (lines 474-487).
    rhox = psfc * 1000.0 / (R_D * scr4)
    zqkl = dz8w1d
    za = 0.5 * zqkl
    govrth = G / thx

    # Bulk Richardson number (lines 494-534).
    gz1oz0 = jnp.log(za / znt)
    gz2oz0 = jnp.log(2.0 / znt)
    gz10oz0 = jnp.log(10.0 / znt)
    wspd0 = jnp.sqrt(ux * ux + vx * vx)
    tskv = thgb * (1.0 + EP1 * qsfc)
    dthvdz = thvx - tskv
    # Convective velocity scale (lines 512-525). WRF's land branch uses the
    # HFX/QFX values carried from the PREVIOUS surface call (zero on the first
    # step); faithful here via the explicit hfx_in/qfx_in inputs.
    fluxc = jnp.maximum(
        hfx_in / rhox / CP + EP1 * tskv * qfx_in / rhox, 0.0
    )
    vconv_land = VCONVC * (G / tgdsa * pblh * fluxc) ** 0.33
    dthvm = jnp.where(-dthvdz >= 0.0, -dthvdz, 0.0)
    vconv_water = jnp.sqrt(dthvm)
    vconv = jnp.where(xland < 1.5, vconv_land, vconv_water)
    vsgd = 0.32 * (jnp.maximum(dx / 5000.0 - 1.0, 0.0)) ** 0.33
    wspd = jnp.sqrt(wspd0 * wspd0 + vconv * vconv + vsgd * vsgd)
    wspd = jnp.maximum(wspd, 0.1)
    br = govrth * za * dthvdz / (wspd * wspd)
    br = jnp.where(mol < 0.0, jnp.minimum(br, 0.0), br)
    rmol0 = -govrth * dthvdz * za * KARMAN  # overwritten per regime below

    # --- regime classification (lines 563-695) ---
    # Regime 1: BR >= 0.2 (stable). Regime 2: 0 < BR < 0.2. Regime 3: BR == 0.
    # Regime 4: BR < 0 (free convection).
    reg1 = br >= 0.2
    reg4 = br < 0.0
    reg3 = br == 0.0
    reg2 = jnp.logical_and(jnp.logical_and(br > 0.0, br < 0.2), jnp.logical_not(reg3))

    # Regime 1 (stable).
    psim_1 = jnp.maximum(-10.0 * gz1oz0, -10.0)
    psih_1 = psim_1
    psim10_1 = jnp.maximum(10.0 / za * psim_1, -10.0)
    psih10_1 = psim10_1
    psim2_1 = jnp.maximum(2.0 / za * psim_1, -10.0)
    psih2_1 = psim2_1
    rmol_1a = br * gz1oz0
    rmol_1b = KARMAN * govrth * za * mol / (ust * ust)
    rmol_1 = jnp.where(ust < 0.01, rmol_1a, rmol_1b)
    rmol_1 = jnp.minimum(rmol_1, 9.999) / za

    # Regime 2 (damped mechanical).
    psim_2 = jnp.maximum(-5.0 * br * gz1oz0 / (1.1 - 5.0 * br), -10.0)
    psih_2 = psim_2
    psim10_2 = jnp.maximum(10.0 / za * psim_2, -10.0)
    psih10_2 = psim10_2
    psim2_2 = jnp.maximum(2.0 / za * psim_2, -10.0)
    psih2_2 = psim2_2
    zol_2_lin = br * gz1oz0 / (1.00001 - 5.0 * br)
    zol_2_nl = jnp.minimum(
        (1.89 * gz1oz0 + 44.2) * br * br + (1.18 * gz1oz0 - 1.37) * br, 9.999
    )
    zol_2 = jnp.where(zol_2_lin > 0.5, zol_2_nl, zol_2_lin)
    rmol_2 = zol_2 / za

    # Regime 3 (forced convection).
    psim_3 = jnp.zeros_like(br)
    psih_3 = psim_3
    psim10_3 = psim_3
    psih10_3 = psim_3
    psim2_3 = psim_3
    psih2_3 = psim_3
    zol_3 = jnp.where(
        ust < 0.01, br * gz1oz0, KARMAN * govrth * za * mol / (ust * ust)
    )
    rmol_3 = zol_3 / za

    # Regime 4 (free convection).
    zol_4 = jnp.where(
        ust < 0.01, br * gz1oz0, KARMAN * govrth * za * mol / (ust * ust)
    )
    zol10_4 = 10.0 / za * zol_4
    zol2_4 = 2.0 / za * zol_4
    zol_4c = jnp.clip(zol_4, -9.9999, 0.0)
    zol10_4c = jnp.clip(zol10_4, -9.9999, 0.0)
    zol2_4c = jnp.clip(zol2_4, -9.9999, 0.0)
    psim_4 = jnp.minimum(_psitb_interp(PSIMTB, zol_4c), 0.9 * gz1oz0)
    psih_4 = jnp.minimum(_psitb_interp(PSIHTB, zol_4c), 0.9 * gz1oz0)
    psim10_4 = jnp.minimum(_psitb_interp(PSIMTB, zol10_4c), 0.9 * gz10oz0)
    psih10_4 = jnp.minimum(_psitb_interp(PSIHTB, zol10_4c), 0.9 * gz10oz0)
    psim2_4 = _psitb_interp(PSIMTB, zol2_4c)
    psih2_4 = jnp.minimum(_psitb_interp(PSIHTB, zol2_4c), 0.9 * gz2oz0)
    rmol_4 = zol_4c / za

    def sel(v1, v2, v3, v4):
        return jnp.where(reg1, v1, jnp.where(reg2, v2, jnp.where(reg3, v3, v4)))

    regime = sel(
        jnp.ones_like(br), 2.0 * jnp.ones_like(br), 3.0 * jnp.ones_like(br),
        4.0 * jnp.ones_like(br),
    )
    psim = sel(psim_1, psim_2, psim_3, psim_4)
    psih = sel(psih_1, psih_2, psih_3, psih_4)
    psim10 = sel(psim10_1, psim10_2, psim10_3, psim10_4)
    psih10 = sel(psih10_1, psih10_2, psih10_3, psih10_4)
    psim2 = sel(psim2_1, psim2_2, psim2_3, psim2_4)
    psih2 = sel(psih2_1, psih2_2, psih2_3, psih2_4)
    rmol = sel(rmol_1, rmol_2, rmol_3, rmol_4)
    # ZOL is a side diagnostic WRF only assigns in regimes 2/3/4; regime 1
    # (stable) leaves it at its entry value (zol_in, default 0).
    zol_in0 = jnp.zeros_like(br) if zol_in is None else zol_in
    zol = sel(zol_in0, zol_2, zol_3, zol_4c)

    # --- frictional velocity / similarity (lines 700-826) ---
    dtg = thx - thgb
    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    psit = jnp.maximum(gz1oz0 - psih, 2.0)
    zl = jnp.where(is_water, znt, 0.01)
    psiq = jnp.log(KARMAN * ust * za / XKA + za / zl) - psih
    psit2 = gz2oz0 - psih2
    psiq2 = jnp.log(KARMAN * ust * 2.0 / XKA + 2.0 / zl) - psih2

    # V3.7 Fairall z0q/z0t over water (lines 721-736).
    visc = (1.32 + 0.009 * (scr3 - 273.15)) * 1.0e-5
    restar = ust * znt / visc
    z0t = jnp.clip((5.5e-5) * (restar ** (-0.60)), 2.0e-9, 1.0e-4)
    z0q = z0t
    psiq_w = jnp.maximum(jnp.log((za + z0q) / z0q) - psih, 2.0)
    psit_w = jnp.maximum(jnp.log((za + z0t) / z0t) - psih, 2.0)
    psiq2_w = jnp.maximum(jnp.log((2.0 + z0q) / z0q) - psih2, 2.0)
    psit2_w = jnp.maximum(jnp.log((2.0 + z0t) / z0t) - psih2, 2.0)
    psiq = jnp.where(is_water, psiq_w, psiq)
    psit = jnp.where(is_water, psit_w, psit)
    psiq2 = jnp.where(is_water, psiq2_w, psiq2)
    psit2 = jnp.where(is_water, psit2_w, psit2)

    # UST update (0.5-averaging; line 799).
    ust = 0.5 * ust + 0.5 * KARMAN * wspd / psix
    ust = jnp.where(xland < 1.5, jnp.maximum(ust, 0.1), ust)

    u10 = ux * psix10 / psix
    v10 = vx * psix10 / psix
    th2 = thgb + dtg * psit2 / psit
    q2 = qsfc + (qx - qsfc) * psiq2 / psiq
    t2 = th2 * (psfcpa / P0_PA) ** ROVCP
    mol = KARMAN * dtg / psit / PRT
    denomq = psiq
    denomq2 = psiq2
    denomt2 = psit2
    fm = psix
    fh = psit

    # --- surface fluxes (lines 834-936) ---
    # Over water, alter ZNT (lines 844-876).
    znt_water = CZO * ust * ust / G + 0.11 * 1.5e-5 / ust
    znt_water = jnp.minimum(znt_water, 2.85e-3)
    znt = jnp.where(is_water, znt_water, znt)
    zl = jnp.where(is_water, znt, 0.01)

    flqc = rhox * mavail * ust * KARMAN / denomq
    dtthx = jnp.abs(thx - thgb)
    flhc = jnp.where(
        dtthx > 1.0e-5, cpm * rhox * ust * mol / (thx - thgb), 0.0
    )

    qfx = flqc * (qsfc - qx)
    lh = XLV * qfx
    hfx = flhc * (thgb - thx)

    # isfflx=0 zeroes fluxes (line 840). Default oracle uses isfflx=1.
    if isfflx == 0:
        qfx = jnp.zeros_like(qfx)
        hfx = jnp.zeros_like(hfx)
        lh = jnp.zeros_like(lh)

    chs = ust * KARMAN / denomq
    cqs2 = ust * KARMAN / denomq2
    chs2 = ust * KARMAN / denomt2

    return {
        "ust": ust, "znt": znt, "mol": mol, "rmol": rmol, "regime": regime,
        "psim": psim, "psih": psih, "fm": fm, "fh": fh,
        "qsfc": qsfc, "qgh": qgh, "cpm": cpm,
        "hfx": hfx, "qfx": qfx, "lh": lh,
        "flhc": flhc, "flqc": flqc, "chs": chs, "chs2": chs2, "cqs2": cqs2,
        "u10": u10, "v10": v10, "th2": th2, "t2": t2, "q2": q2,
        "br": br, "wspd": wspd, "gz1oz0": gz1oz0, "zol": zol,
    }


def sfclay_old_mm5_columns(
    ux, vx, t1d, qv1d, p1d, dz8w1d, psfcpa, tsk, xland, znt, ust, mol, qsfc,
    pblh, mavail, lakemask, dx, hfx_in, qfx_in, zol_in=None, *, isfflx: int = 1,
):
    """Batched (vmap) entry over the grid columns (all inputs shape ``(ncol,)``)."""

    if zol_in is None:
        zol_in = jnp.zeros_like(jnp.asarray(ux))

    def one(*a):
        return sfclay_old_mm5_column(*a, isfflx=isfflx)

    return jax.vmap(one, in_axes=(0,) * 20)(
        ux, vx, t1d, qv1d, p1d, dz8w1d, psfcpa, tsk, xland, znt, ust, mol, qsfc,
        pblh, mavail, lakemask, dx, hfx_in, qfx_in, zol_in,
    )


__all__ = [
    "PSIMTB",
    "PSIHTB",
    "sfclay_old_mm5_column",
    "sfclay_old_mm5_columns",
]
