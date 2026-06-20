"""WRF NCEP-GFS surface layer (``sf_sfclay_physics=3``).

This is the v0.13 Tier-3 per-scheme lane port of WRF's
``phys/module_sf_gfs.F`` (``SF_GFS`` + the surface-layer subset of ``PROGTM``).
In WRF's standalone surface-layer mode the GFS land/soil/canopy physics are all
bypassed (the Fortran ``GOTO 1111`` / ``GOTO 5555`` skip the soil-moisture,
evaporation and canopy blocks -- the land arrays SMC/STC/CANOPY are not even
passed), leaving a self-contained NCEP Monin-Obukhov bulk-Richardson
exchange-coefficient solve:

* reference height ``Z1`` from the hydrostatic thickness, roughness ``Z0MAX``
  / thermal roughness ``ZTMAX`` (with the Zeng-Zhao-Dickinson 1997 ocean
  Reynolds-number correction), bulk Richardson ``RB``, and ``FM``/``FH`` log
  profiles;
* a stable-case 2-iteration Obukhov solve (``PM``/``PH`` from the
  ``sqrt(1+4*alpha*HL)`` integral form) and an unstable-case rational/log
  ``PM``/``PH``;
* exchange coefficients ``CM=CA^2/FM^2``, ``CH=CA^2/(FM*FH)``, friction velocity
  ``USTAR=sqrt(CM*WIND^2)``, and 10 m / 2 m factors;
* the ``SF_GFS`` outer wrapper then forms the WRF B2 handles ``CHS``/``CHS2``/
  ``CQS2``/``CPM``/``FLHC``/``FLQC``/``QGH``/``QSFC``/``HFX``/``QFX``/``LH``/
  ``U10``/``V10``/``PSIM``/``PSIH``/``WSPD``/``BR``/``GZ1OZ0``.

The saturation vapor pressure ``fpvs`` is reproduced FAITHFULLY: the same 7501-
node table over [180, 330] K built from the exact ``fpvsx`` Clausius-Clapeyron
form (Emanuel 1994), with the same linear table interpolation -- so QSS/QGH match
the Fortran value, not an analytic approximation.

The column kernel is a pure ``jnp`` function, ``jax.jit``/``jax.vmap``-traceable
(land/sea/stability branches are vectorized via ``jnp.where``). Faithful to the
default ARW path the generated oracle exercises: ``isfflx=1``, ``rcl=1`` (no
reduced-grid map factor), no down-draft velocity (``ddvel=0``).

Cited to ``<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_gfs.F`` (SF_GFS
9-282; PROGTM MO subset 519-753, diagnostics 1432-1457) and
``module_gfs_funcphys.F`` (fpvsx 814-882) / ``module_gfs_physcons.F`` (constants).
"""

from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp
import numpy as np


config.update("jax_enable_x64", True)


# --- GFS physical constants (module_gfs_physcons.F) ---
CON_G = 9.80665
CON_RD = 287.05
CON_RV = 461.50
CON_CP = 1004.6
CON_CVAP = 1846.0
CON_CLIQ = 4185.5
CON_CSOL = 2106.0
CON_HVAP = 2.5e6
CON_HFUS = 3.3358e5
CON_PSAT = 610.78
CON_T0C = 273.15
CON_TTP = 273.16
CON_EPS = CON_RD / CON_RV
CON_EPSM1 = CON_RD / CON_RV - 1.0
CON_FVIRT = CON_RV / CON_RD - 1.0     # RVRDM1

# --- WRF-side constants SF_GFS receives (module_surface_driver.F) ---
CP = 1004.0           # WRF cp (used in CPM/FLHC by SF_GFS, NOT the GFS con_cp)
R = 287.0             # WRF r
ROVCP = R / CP
XLV = 2.5e6
EP1 = 0.608
EP2 = 0.622
KARMAN = 0.4

# --- PROGTM parameters (module_sf_gfs.F:381-394) ---
CHARNOCK = 0.014
CA = 0.4              # von Karman (PROGTM-local)
ALPHA = 5.0
A0 = -3.975
A1 = 12.32
B1 = -7.755
B2 = 6.041
A0P = -7.941
A1P = 24.75
B1P = -8.705
B2P = 7.899
VIS = 1.4e-5


# --- fpvs: 7501-node table built from the exact fpvsx (Emanuel 1994) ---
_NXPVS = 7501
_XMIN = 180.0
_XMAX = 330.0
_XINC = (_XMAX - _XMIN) / (_NXPVS - 1)
_C2XPVS = 1.0 / _XINC
_C1XPVS = 1.0 - _XMIN * _C2XPVS

_TLIQ = CON_TTP
_TICE = CON_TTP - 20.0
_DLDTL = CON_CVAP - CON_CLIQ
_HEATL = CON_HVAP
_XPONAL = -_DLDTL / CON_RV
_XPONBL = -_DLDTL / CON_RV + _HEATL / (CON_RV * CON_TTP)
_DLDTI = CON_CVAP - CON_CSOL
_HEATI = CON_HVAP + CON_HFUS
_XPONAI = -_DLDTI / CON_RV
_XPONBI = -_DLDTI / CON_RV + _HEATI / (CON_RV * CON_TTP)


def _fpvsx_np(t):
    tr = CON_TTP / t
    pvl = CON_PSAT * (tr ** _XPONAL) * np.exp(_XPONBL * (1.0 - tr))
    pvi = CON_PSAT * (tr ** _XPONAI) * np.exp(_XPONBI * (1.0 - tr))
    w = (t - _TICE) / (_TLIQ - _TICE)
    mid = w * pvl + (1.0 - w) * pvi
    out = np.where(t >= _TLIQ, pvl, np.where(t < _TICE, pvi, mid))
    return out


# Build the table exactly as gpvs does (module_gfs_funcphys.F:gpvs).
_t_nodes = _XMIN + (np.arange(_NXPVS, dtype=np.float64)) * _XINC
_TBPVS = jnp.asarray(_fpvsx_np(_t_nodes), dtype=jnp.float64)


def fpvs(t):
    """Faithful GFS ``fpvs`` table lookup (linear interp; module_gfs_funcphys.F:fpvs)."""

    xj = jnp.minimum(jnp.maximum(_C1XPVS + _C2XPVS * t, 1.0), float(_NXPVS))
    jx = jnp.minimum(xj, _NXPVS - 1.0)
    jx_i = jnp.floor(jx).astype(jnp.int32)
    jx_i = jnp.clip(jx_i, 1, _NXPVS - 1)
    # Fortran 1-based: tbpvs(jx) + (xj-jx)*(tbpvs(jx+1)-tbpvs(jx)); jx is the
    # integer truncation. Convert to 0-based index jx_i-1.
    lo = _TBPVS[jx_i - 1]
    hi = _TBPVS[jx_i]
    return lo + (xj - jx_i) * (hi - lo)


def sf_gfs_column(
    u3d: jax.Array,          # u at lowest level (m/s)
    v3d: jax.Array,          # v at lowest level (m/s)
    t3d: jax.Array,          # temperature at lowest level (K)
    qv3d: jax.Array,         # qv at lowest level (kg/kg)
    p3d: jax.Array,          # pressure at lowest level (Pa)
    psfc: jax.Array,         # surface pressure (Pa)
    tsk: jax.Array,          # skin temperature (K)
    xland: jax.Array,        # land mask (1 land, 2 water)
    znt: jax.Array,          # roughness (m) -- INOUT (Z0RL=znt*100)
    ust: jax.Array,          # u* (m/s) -- INOUT
    *,
    isfflx: int = 1,
):
    """Faithful per-column port of WRF ``SF_GFS`` (default ARW surface-layer path).

    Returns a dict of the WRF B2 surface-layer handles. All math is fp64 when
    the inputs are fp64; nothing allocates outside the trace.
    """

    # --- SF_GFS pre-loop (module_sf_gfs.F:191-215) ---
    rcl = 1.0
    prsl1 = p3d * 0.001          # cb
    ps_cb = psfc * 0.001
    q1_in = qv3d
    slimsk = jnp.abs(xland - 2.0)   # 0=water, 1=land (sea-ice=2 not modeled here)
    tskin = tsk
    t1 = t3d
    u1 = u3d
    v1 = v3d
    ustar = ust
    z0rl = znt * 100.0
    prslki = (ps_cb / prsl1) ** ROVCP
    thgb = tskin * (100.0 / ps_cb) ** ROVCP
    thx = t1 * (100.0 / prsl1) ** ROVCP
    rho1 = prsl1 * 1000.0 / (R * t1 * (1.0 + EP1 * q1_in))
    q1 = q1_in / (1.0 + q1_in)      # specific humidity

    # --- PROGTM pre (module_sf_gfs.F:454-486) ---
    xrcl = np.sqrt(rcl)
    psurf = 1000.0 * ps_cb
    ps1 = 1000.0 * prsl1
    wind = jnp.maximum(xrcl * jnp.sqrt(u1 * u1 + v1 * v1), 1.0)
    q0 = jnp.maximum(q1, 1.0e-8)
    tsurf = tskin
    theta1 = t1 * prslki
    tv1 = t1 * (1.0 + CON_FVIRT * q0)
    thv1 = theta1 * (1.0 + CON_FVIRT * q0)
    tvs = tsurf * (1.0 + CON_FVIRT * q0)
    rho = ps1 / (CON_RD * tv1)
    qs1 = fpvs(t1)
    qs1 = CON_EPS * qs1 / (ps1 + CON_EPSM1 * qs1)
    qs1 = jnp.maximum(qs1, 1.0e-8)
    q0 = jnp.minimum(qs1, q0)
    qss = fpvs(tskin)
    qss = CON_EPS * qss / (psurf + CON_EPSM1 * qss)
    z0 = 0.01 * z0rl

    # --- Z1 + roughness (module_sf_gfs.F:544, 577-600) ---
    z1 = -CON_RD * tv1 * jnp.log(ps1 / psurf) / CON_G
    is_water = slimsk == 0.0
    ustar = jnp.where(is_water, jnp.sqrt(CON_G * z0 / CHARNOCK), ustar)
    z0max = jnp.minimum(z0, 0.1 * z1)
    restar = jnp.maximum(ustar * z0max / VIS, 1.0e-6)
    rat = jnp.minimum(2.67 * restar ** 0.25 - 2.57, 7.0)
    ztmax = jnp.where(is_water, z0max * jnp.exp(-rat), z0max)

    # --- bulk Richardson (module_sf_gfs.F:607-622) ---
    dtv = thv1 - tvs
    adtv = jnp.maximum(jnp.abs(dtv), 0.001)
    dtv = jnp.sign(dtv) * adtv
    rb = jnp.maximum(
        CON_G * dtv * z1 / (0.5 * (thv1 + tvs) * wind * wind), -5000.0
    )
    fm = jnp.log(z1 / z0max)
    fh = jnp.log(z1 / ztmax)
    hlinf = rb * fm * fm / fh
    fm10 = jnp.log((z0max + 10.0) / z0max)
    fh2 = jnp.log((ztmax + 2.0) / ztmax)

    stable = dtv >= 0.0

    # --- STABLE: first pass HL1 (lines 630-647) ---
    hl1 = hlinf
    cond_s1 = jnp.logical_and(stable, hlinf > 0.25)
    hl0inf = z0max * hlinf / z1
    hltinf = ztmax * hlinf / z1
    aa = jnp.sqrt(1.0 + 4.0 * ALPHA * hlinf)
    aa0 = jnp.sqrt(1.0 + 4.0 * ALPHA * hl0inf)
    bb = aa
    bb0 = jnp.sqrt(1.0 + 4.0 * ALPHA * hltinf)
    pm_s1 = aa0 - aa + jnp.log((aa + 1.0) / (aa0 + 1.0))
    ph_s1 = bb0 - bb + jnp.log((bb + 1.0) / (bb0 + 1.0))
    fms = fm - pm_s1
    fhs = fh - ph_s1
    hl1 = jnp.where(cond_s1, fms * fms * rb / fhs, hl1)

    # --- STABLE: second iteration (lines 651-669) ---
    hl0 = z0max * hl1 / z1
    hlt = ztmax * hl1 / z1
    aa = jnp.sqrt(1.0 + 4.0 * ALPHA * hl1)
    aa0 = jnp.sqrt(1.0 + 4.0 * ALPHA * hl0)
    bb = aa
    bb0 = jnp.sqrt(1.0 + 4.0 * ALPHA * hlt)
    pm_s = aa0 - aa + jnp.log((aa + 1.0) / (aa0 + 1.0))
    ph_s = bb0 - bb + jnp.log((bb + 1.0) / (bb0 + 1.0))
    hl110 = hl1 * 10.0 / z1
    aa110 = jnp.sqrt(1.0 + 4.0 * ALPHA * hl110)
    pm10_s = aa0 - aa110 + jnp.log((aa110 + 1.0) / (aa0 + 1.0))
    hl12 = hl1 * 2.0 / z1
    bb12 = jnp.sqrt(1.0 + 4.0 * ALPHA * hl12)
    ph2_s = bb0 - bb12 + jnp.log((bb12 + 1.0) / (bb0 + 1.0))

    # --- UNSTABLE: clamp + PM/PH (lines 681-714) ---
    olinf = z1 / hlinf
    hlinf_u = jnp.where(
        jnp.logical_and(jnp.logical_not(stable), jnp.abs(olinf) <= 50.0 * z0max),
        -z1 / (50.0 * z0max), hlinf,
    )
    # Branch a: hlinf >= -0.5 (rational form).
    hl1_ua = hlinf_u
    pm_ua = (A0 + A1 * hl1_ua) * hl1_ua / (1.0 + B1 * hl1_ua + B2 * hl1_ua * hl1_ua)
    ph_ua = (A0P + A1P * hl1_ua) * hl1_ua / (1.0 + B1P * hl1_ua + B2P * hl1_ua * hl1_ua)
    hl110_ua = hl1_ua * 10.0 / z1
    pm10_ua = (A0 + A1 * hl110_ua) * hl110_ua / (1.0 + B1 * hl110_ua + B2 * hl110_ua * hl110_ua)
    hl12_ua = hl1_ua * 2.0 / z1
    ph2_ua = (A0P + A1P * hl12_ua) * hl12_ua / (1.0 + B1P * hl12_ua + B2P * hl12_ua * hl12_ua)
    # Branch b: hlinf < -0.5 (log form).
    hl1_ub = -hlinf_u
    pm_ub = jnp.log(hl1_ub) + 2.0 * hl1_ub ** (-0.25) - 0.8776
    ph_ub = jnp.log(hl1_ub) + 0.5 * hl1_ub ** (-0.5) + 1.386
    hl110_ub = hl1_ub * 10.0 / z1
    pm10_ub = jnp.log(hl110_ub) + 2.0 * hl110_ub ** (-0.25) - 0.8776
    hl12_ub = hl1_ub * 2.0 / z1
    ph2_ub = jnp.log(hl12_ub) + 0.5 * hl12_ub ** (-0.5) + 1.386

    use_ua = hlinf_u >= -0.5
    pm_u = jnp.where(use_ua, pm_ua, pm_ub)
    ph_u = jnp.where(use_ua, ph_ua, ph_ub)
    pm10_u = jnp.where(use_ua, pm10_ua, pm10_ub)
    ph2_u = jnp.where(use_ua, ph2_ua, ph2_ub)

    pm = jnp.where(stable, pm_s, pm_u)
    ph = jnp.where(stable, ph_s, ph_u)
    pm10 = jnp.where(stable, pm10_s, pm10_u)
    ph2 = jnp.where(stable, ph2_s, ph2_u)

    # --- finish FM/FH, CM/CH, USTAR (lines 719-731) ---
    fm = fm - pm
    fh = fh - ph
    fm10 = fm10 - pm10
    fh2 = fh2 - ph2
    cm = CA * CA / (fm * fm)
    ch = CA * CA / (fm * fh)
    stress = cm * wind * wind
    ustar = jnp.sqrt(stress)

    # Update Z0/Z0RL over ocean (lines 739-750).
    z0_w = jnp.clip((CHARNOCK / CON_G) * ustar ** 2, 1.0e-7, 0.1)
    z0 = jnp.where(is_water, z0_w, z0)
    z0rl = jnp.where(is_water, 100.0 * z0, z0rl)

    # --- 10 m factor (lines 1454-1457) ---
    f10m = jnp.minimum(fm10 / fm, 1.0)
    u10m = f10m * xrcl * u1
    v10m = f10m * xrcl * v1
    # T2M/Q2M are WRF-commented in PROGTM; SF_GFS does not read them.

    # --- SF_GFS outer wrapper outputs (module_sf_gfs.F:234-276) ---
    u10 = u10m
    v10 = v10m
    br = rb
    chs = ch * wind
    chs2 = ustar * KARMAN / fh2
    cpm = CP * (1.0 + 0.8 * qv3d)
    esat = fpvs(t1)
    qgh = EP2 * esat / (1000.0 * ps_cb - esat)
    qsfc = qss
    psih = ph
    psim = pm
    wspd = wind
    znt_out = z0rl * 0.01

    flhc = cpm * rho1 * chs
    flqc = rho1 * chs
    gz1oz0 = jnp.log(z1 / (z0rl * 0.01))
    cqs2 = chs2

    # HFX/QFX/LH (isfflx; lines 258-276).
    if isfflx == 0:
        hfx = jnp.zeros_like(thgb)
        lh = jnp.zeros_like(thgb)
        qfx = jnp.zeros_like(thgb)
    else:
        hfx = flhc * (thgb - thx)
        # over land (xland-1.5>0 is WATER in WRF's xland convention 1=land/2=water)
        hfx = jnp.where(
            (xland - 1.5) < 0.0, jnp.maximum(hfx, -250.0), hfx
        )
        qfx = jnp.maximum(flqc * (qsfc - q1), 0.0)
        lh = XLV * qfx

    return {
        "ust": ustar, "znt": znt_out, "chs": chs, "chs2": chs2, "cqs2": cqs2,
        "cpm": cpm, "psim": psim, "psih": psih, "qgh": qgh, "qsfc": qsfc,
        "u10": u10, "v10": v10, "wspd": wspd, "br": br, "gz1oz0": gz1oz0,
        "flhc": flhc, "flqc": flqc, "hfx": hfx, "qfx": qfx, "lh": lh,
        "fm": fm, "fh": fh, "cm": cm, "ch": ch, "z1": z1, "rb": rb,
    }


def sf_gfs_columns(
    u3d, v3d, t3d, qv3d, p3d, psfc, tsk, xland, znt, ust, *, isfflx: int = 1,
):
    """Batched (vmap) entry over the grid columns (all inputs shape ``(ncol,)``)."""

    def one(*a):
        return sf_gfs_column(*a, isfflx=isfflx)

    return jax.vmap(one, in_axes=(0,) * 10)(
        u3d, v3d, t3d, qv3d, p3d, psfc, tsk, xland, znt, ust,
    )


__all__ = [
    "fpvs",
    "sf_gfs_column",
    "sf_gfs_columns",
]
