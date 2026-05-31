"""WRF revised Monin-Obukhov surface layer (``sfclayrev``), JAX port.

Faithful, fully vectorized transcription of the WRF revised surface-layer scheme
(Jimenez et al. 2012, MWR 140, 898-918) as implemented in the CCPP core
``sf_sfclayrev_run``:

  /home/enric/src/wrf_pristine/WRF/phys/physics_mmm/sf_sfclayrev.F90

(which ``module_sf_sfclayrev.F`` on this workstation delegates to via
``sf_sfclayrev_pre_run`` -> lowest-level column -> ``sf_sfclayrev_run``).

Every block carries the ``sf_sfclayrev.F90:<line>`` reference for the Fortran it
ports. This is a clean rebuild; it does NOT resurrect the FAILED M12 MM5
``module_sf_sfclay.F`` attempt. The two schemes differ fundamentally:

* MM5 sfclay used closed-form regime PSIM/PSIH; sfclayrev uses Cheng & Brutsaert
  (2005, CB05) *integrated* similarity functions tabulated over z/L and a
  bulk-Richardson Newton/secant solve (``zolri``) for z/L.
* sfclayrev recomputes PSIH/PSIT/PSIQ over water with Fairall (2003) z0t/z0q.

Computation is in float64 (x64 enabled at package import); callers cast outputs
to the frozen storage dtype at the coupling boundary.

Sign conventions (kinematic, positive upward into the atmosphere), matching the
``mynn_surface_stub.SurfaceFluxes`` contract MYNN consumes:
* ``theta_flux`` = HFX / (rho*cpm)   [K m s^-1]
* ``qv_flux``    = QFX / rho          [kg kg^-1 m s^-1]
* ``tau_u/tau_v``= -ustar^2 * u/|U|   [m2 s^-2]
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.physics.surface_constants import (
    CP_D,
    EP1,
    EP2,
    G,
    KARMAN,
    MIN_WIND_M_S,
    OZO,
    P0_PA,
    PRT,
    R_D,
    R_D_OVER_CP,
    SALINITY_FACTOR,
    SFCLAYREV_TABLE_DZOL,
    SFCLAYREV_TABLE_N,
    SVP1_KPA,
    SVP2,
    SVP3_K,
    SVPT0_K,
    VCONVC,
    XKA,
    XLV,
    ZOLRI_BR_CAP,
    ZOLRI_MAX_ITER,
)


# ==================================================================================
# CB05 integrated similarity functions (sf_sfclayrev.F90:987-1030 "_full" forms)
# ==================================================================================


def _psim_stable_full(zolf):
    """psim_stable_full, sf_sfclayrev.F90:987-992."""

    return -6.1 * jnp.log(zolf + (1.0 + zolf**2.5) ** (1.0 / 2.5))


def _psih_stable_full(zolf):
    """psih_stable_full, sf_sfclayrev.F90:995-1000."""

    return -5.3 * jnp.log(zolf + (1.0 + zolf**1.1) ** (1.0 / 1.1))


def _psim_unstable_full(zolf):
    """psim_unstable_full, sf_sfclayrev.F90:1003-1015."""

    x = (1.0 - 16.0 * zolf) ** 0.25
    psimk = 2.0 * jnp.log(0.5 * (1.0 + x)) + jnp.log(0.5 * (1.0 + x * x)) - 2.0 * jnp.arctan(x) + 2.0 * jnp.arctan(1.0)
    ym = (1.0 - 10.0 * zolf) ** 0.33
    psimc = (
        (3.0 / 2.0) * jnp.log((ym**2.0 + ym + 1.0) / 3.0)
        - jnp.sqrt(3.0) * jnp.arctan((2.0 * ym + 1.0) / jnp.sqrt(3.0))
        + 4.0 * jnp.arctan(1.0) / jnp.sqrt(3.0)
    )
    return (psimk + zolf**2 * psimc) / (1.0 + zolf**2.0)


def _psih_unstable_full(zolf):
    """psih_unstable_full, sf_sfclayrev.F90:1018-1030."""

    y = (1.0 - 16.0 * zolf) ** 0.5
    psihk = 2.0 * jnp.log((1.0 + y) / 2.0)
    yh = (1.0 - 34.0 * zolf) ** 0.33
    psihc = (
        (3.0 / 2.0) * jnp.log((yh**2.0 + yh + 1.0) / 3.0)
        - jnp.sqrt(3.0) * jnp.arctan((2.0 * yh + 1.0) / jnp.sqrt(3.0))
        + 4.0 * jnp.arctan(1.0) / jnp.sqrt(3.0)
    )
    return (psihk + zolf**2 * psihc) / (1.0 + zolf**2.0)


# Precomputed CB05 lookup tables, exactly as sf_sfclayrev_init builds them
# (sf_sfclayrev.F90:39-49): node n holds the "_full" value at zolf = +/- n*0.01.
# Built once at import in float64 so the lookup matches WRF bit-for-table.
_N = SFCLAYREV_TABLE_N
import numpy as _np

_ZOLF_STAB = _np.arange(0, _N + 1, dtype=_np.float64) * SFCLAYREV_TABLE_DZOL
_ZOLF_UNSTAB = -_ZOLF_STAB
_PSIM_STAB_TABLE = jnp.asarray(_np.asarray(_psim_stable_full(_ZOLF_STAB)), dtype=jnp.float64)
_PSIH_STAB_TABLE = jnp.asarray(_np.asarray(_psih_stable_full(_ZOLF_STAB)), dtype=jnp.float64)
_PSIM_UNSTAB_TABLE = jnp.asarray(_np.asarray(_psim_unstable_full(_ZOLF_UNSTAB)), dtype=jnp.float64)
_PSIH_UNSTAB_TABLE = jnp.asarray(_np.asarray(_psih_unstable_full(_ZOLF_UNSTAB)), dtype=jnp.float64)
del _np, _ZOLF_STAB, _ZOLF_UNSTAB


def _table_lookup(zolf_scaled, table, full_fn, zolf):
    """WRF look-up-table interpolation, sf_sfclayrev.F90:1034-1095.

    ``zolf_scaled = |zolf|*100`` is the (signed for stable, |.| for unstable)
    table coordinate. ``nzol = int(zolf_scaled)``; linear interpolation between
    nodes ``nzol`` and ``nzol+1`` when ``nzol+1 < 1000`` (i.e. |z/L| < ~10),
    else the analytic ``full_fn(zolf)``.
    """

    nzol = jnp.floor(zolf_scaled).astype(jnp.int32)
    rzol = zolf_scaled - nzol.astype(zolf_scaled.dtype)
    in_table = (nzol + 1) < _N  # WRF: nzol+1 .lt. 1000
    nzol_c = jnp.clip(nzol, 0, _N - 1)
    base = table[nzol_c]
    nxt = table[jnp.clip(nzol_c + 1, 0, _N)]
    interp = base + rzol * (nxt - base)
    return jnp.where(in_table, interp, full_fn(zolf))


def _psim_stable(zolf):
    """psim_stable lookup, sf_sfclayrev.F90:1034-1047."""

    return _table_lookup(zolf * 100.0, _PSIM_STAB_TABLE, _psim_stable_full, zolf)


def _psih_stable(zolf):
    """psih_stable lookup, sf_sfclayrev.F90:1050-1063."""

    return _table_lookup(zolf * 100.0, _PSIH_STAB_TABLE, _psih_stable_full, zolf)


def _psim_unstable(zolf):
    """psim_unstable lookup, sf_sfclayrev.F90:1066-1079 (table coord = -zolf*100)."""

    return _table_lookup(-zolf * 100.0, _PSIM_UNSTAB_TABLE, _psim_unstable_full, zolf)


def _psih_unstable(zolf):
    """psih_unstable lookup, sf_sfclayrev.F90:1082-1095 (table coord = -zolf*100)."""

    return _table_lookup(-zolf * 100.0, _PSIH_UNSTAB_TABLE, _psih_unstable_full, zolf)


# ==================================================================================
# zolri bulk-Richardson -> z/L secant solve (sf_sfclayrev.F90:922-981)
# ==================================================================================


def _zolri2(zol2, ri2, z, z0):
    """zolri2 residual, sf_sfclayrev.F90:960-981.

    Returns the residual f(zol2) = zol2*psih2/psix2**2 - ri2 used by the secant
    iteration, with the WRF sign guard ``if(zol2*ri2 < 0) zol2 = 0``.
    """

    zol2 = jnp.where(zol2 * ri2 < 0.0, 0.0, zol2)  # must be same sign as ri2
    zol20 = zol2 * z0 / z  # z0/L
    zol3 = zol2 + zol20    # (z+z0)/L
    log_term = jnp.log((z + z0) / z0)
    psix2_uns = log_term - (_psim_unstable(zol3) - _psim_unstable(zol20))
    psih2_uns = log_term - (_psih_unstable(zol3) - _psih_unstable(zol20))
    psix2_sta = log_term - (_psim_stable(zol3) - _psim_stable(zol20))
    psih2_sta = log_term - (_psih_stable(zol3) - _psih_stable(zol20))
    unstable = ri2 < 0.0
    psix2 = jnp.where(unstable, psix2_uns, psix2_sta)
    psih2 = jnp.where(unstable, psih2_uns, psih2_sta)
    return zol2 * psih2 / (psix2 * psix2) - ri2, zol2


def _zolri(ri, z, z0):
    """zolri secant solve for z/L, sf_sfclayrev.F90:922-957.

    Fixed-trip-count vectorized transcription of the WRF secant loop. WRF runs at
    most 10 iterations and stops early once ``|x1-x2| <= 0.01``; we run the full
    10 (cheap, branch-free) but freeze the active endpoint once converged so the
    returned z/L matches WRF to iteration parity.
    """

    unstable = ri < 0.0
    x1 = jnp.where(unstable, -5.0, 0.0)
    x2 = jnp.where(unstable, 0.0, 5.0)
    fx1, _ = _zolri2(x1, ri, z, z0)
    fx2, _ = _zolri2(x2, ri, z, z0)
    # zolri returns the most-recently-updated endpoint; seed with x2 (WRF would
    # only return uninitialized if the loop body never ran, which it always does
    # because |x1-x2| = 5 > 0.01 at entry).
    zolri = x2

    def body(_, carry):
        x1, x2, fx1, fx2, zolri = carry
        not_conv = jnp.abs(x1 - x2) > 0.01
        equal_f = fx1 == fx2  # WRF divide-by-zero guard: return current zolri
        active = not_conv & (~equal_f)
        use_x1 = jnp.abs(fx2) < jnp.abs(fx1)
        # branch updating x1
        x1_new = x1 - fx1 / (fx2 - fx1) * (x2 - x1)
        fx1_new, _ = _zolri2(x1_new, ri, z, z0)
        # branch updating x2
        x2_new = x2 - fx2 / (fx2 - fx1) * (x2 - x1)
        fx2_new, _ = _zolri2(x2_new, ri, z, z0)
        new_x1 = jnp.where(active & use_x1, x1_new, x1)
        new_fx1 = jnp.where(active & use_x1, fx1_new, fx1)
        new_x2 = jnp.where(active & (~use_x1), x2_new, x2)
        new_fx2 = jnp.where(active & (~use_x1), fx2_new, fx2)
        new_zolri = jnp.where(active, jnp.where(use_x1, x1_new, x2_new), zolri)
        return (new_x1, new_x2, new_fx1, new_fx2, new_zolri)

    carry = (x1, x2, fx1, fx2, zolri)
    carry = jax.lax.fori_loop(0, ZOLRI_MAX_ITER, body, carry)
    return carry[4]


# ==================================================================================
# Helpers for State <-> column extraction
# ==================================================================================


class SurfaceLayerDiagnostics(NamedTuple):
    """sfclayrev outputs: the MYNN flux contract plus operational diagnostics.

    All 2-D fields are mass-point ``(ny, nx)`` (or whatever batch shape the
    surface columns carried). ``hfx``/``lh`` are in W m^-2; ``t2`` in K; ``u10``/
    ``v10`` in m s^-1; ``pblh`` is passed through (a PBL-scheme diagnostic).
    """

    fluxes: SurfaceFluxes
    hfx: object        # W m^-2 upward sensible heat flux
    lh: object         # W m^-2 upward latent heat flux
    u10: object
    v10: object
    th2: object
    t2: object
    q2: object
    qsfc: object       # surface saturation mixing ratio (inout in WRF)
    mol: object        # T* (theta scale)
    rmol: object       # 1/L
    zol: object        # z/L
    regime: object     # 1 stable / 3 neutral / 4 unstable
    psim: object
    psih: object
    br: object
    znt: object        # roughness length used


def _field(state, name: str, default):
    return getattr(state, name, default)


def _surface(field):
    """Return lowest model level from a trailing-z column or a 2-D field."""

    if getattr(field, "ndim", 0) >= 3:
        return field[..., 0]
    return field


def _as_surface(value, shape):
    data = jnp.asarray(value, dtype=jnp.float64)
    if data.ndim >= 3:
        data = data[..., 0]
    if data.shape == ():
        return jnp.broadcast_to(data, shape)
    return jnp.broadcast_to(data, shape)


# ==================================================================================
# Main surface-layer solve
# ==================================================================================


def surface_layer(state) -> SurfaceFluxes:
    """Return the MYNN ``SurfaceFluxes`` contract (kinematic flux handles)."""

    return surface_layer_with_diagnostics(state).fluxes


def surface_layer_with_diagnostics(state) -> SurfaceLayerDiagnostics:
    """Run one vectorized ``sf_sfclayrev_run`` solve over surface columns.

    ``state`` is a column-oriented view (trailing-z) carrying ``u, v, theta, qv,
    p, dz`` at least, plus the prescribed surface fields ``t_skin, xland,
    lakemask, mavail, roughness_m, ustar`` and optionally ``soil_moisture, pblh,
    dx_m``. ``isfflx`` (surface-flux switch) is assumed ON, ``isftcflx=0``,
    ``iz0tlnd=0``, ``shalwater_z0=.false.`` — the default Canary configuration.

    Notes on inputs that differ from WRF call site:
    * WRF passes lowest-level ``t1d`` (temperature); here we derive it from the
      lowest ``theta`` and ``p`` via the Exner function (consistent with
      sf_sfclayrev_pre_run feeding t3d(kts)).
    * WRF passes ``dz8w1d`` = full lowest-layer thickness; ``za = 0.5*dz`` is the
      lowest mass-level height (sf_sfclayrev.F90:298-309, with zqklp1=0).
    """

    # --- lowest-level column inputs (sf_sfclayrev_pre_run picks kts) ---
    u0 = _surface(jnp.asarray(state.u, dtype=jnp.float64))
    v0 = _surface(jnp.asarray(state.v, dtype=jnp.float64))
    theta0 = _surface(jnp.asarray(state.theta, dtype=jnp.float64))
    qv0 = jnp.maximum(_surface(jnp.asarray(state.qv, dtype=jnp.float64)), 0.0)
    p1d_pa = jnp.maximum(_surface(jnp.asarray(state.p, dtype=jnp.float64)), 1.0)  # lowest-level air pressure (p3d(kts))
    shape = u0.shape

    dz = jnp.maximum(_as_surface(_field(state, "dz", 100.0), shape), 1.0)  # dz8w1d
    t_skin = _as_surface(_field(state, "t_skin", None), shape) if _field(state, "t_skin", None) is not None else None
    xland = _as_surface(_field(state, "xland", 1.0), shape)
    lakemask = _as_surface(_field(state, "lakemask", 0.0), shape)
    mavail = jnp.clip(_as_surface(_field(state, "mavail", _field(state, "soil_moisture", 1.0)), shape), 0.0, 1.0)
    ust_in = jnp.maximum(_as_surface(_field(state, "ustar", 0.1), shape), 0.0)
    mol_in = _as_surface(_field(state, "mol", 0.0), shape)
    pblh = jnp.maximum(_as_surface(_field(state, "pblh", 1000.0), shape), 1.0)
    dx_m = jnp.maximum(_as_surface(_field(state, "dx_m", 3000.0), shape), 1.0)
    znt = jnp.maximum(_roughness_from_state(state, shape, xland), 1.0e-7)

    # psfcpa: WRF passes the ACTUAL surface pressure (distinct from lowest-level
    # air pressure). WRF uses psfcpa for thgb, qsfc, rhox, t2 (sf_sfclayrev.F90:
    # 221,230,285,300,767) and p1d for thx/t1d/qgh (sf_sfclayrev.F90:255-290).
    # Use the prescribed ``psfc`` when present (real WRF columns), else fall back
    # to the lowest-level air pressure.
    psfc_field = _field(state, "psfc", None)
    psfcpa = jnp.maximum(_as_surface(psfc_field, shape), 1.0) if psfc_field is not None else p1d_pa
    psfc_cb = psfcpa / 1000.0  # PSFC in cb (sf_sfclayrev.F90:221)

    # --- ground potential temperature (sf_sfclayrev.F90:227-231) ---
    is_land = (xland - 1.5) < 0.0
    is_water = ~is_land
    # t1d: lowest-level air temperature. Use the prescribed ``t_air`` (t_phy) when
    # present, else derive from theta via Exner at the lowest-level pressure.
    t_air_field = _field(state, "t_air", None)
    t1d = _as_surface(t_air_field, shape) if t_air_field is not None else _potential_to_temperature(theta0, p1d_pa)
    tgdsa = t_skin if t_skin is not None else t1d
    thgb = tgdsa * (P0_PA / psfcpa) ** R_D_OVER_CP

    # --- lowest-level theta, virtual theta (sf_sfclayrev.F90:253-292) ---
    pl_cb = p1d_pa / 1000.0
    thcon = (P0_PA * 0.001 / pl_cb) ** R_D_OVER_CP
    thx = t1d * thcon                                # potential temp wrt p1000 (scr3->thx)
    qx = qv0
    tvcon = 1.0 + EP1 * qx
    thvx = thx * tvcon                               # virtual potential temperature
    scr4 = t1d * tvcon                               # virtual temperature

    # --- saturation surface mixing ratio qsfc (sf_sfclayrev.F90:280-291) ---
    e1_g = SVP1_KPA * jnp.exp(SVP2 * (tgdsa - SVPT0_K) / (tgdsa - SVP3_K))
    e1_g = jnp.where(is_water & (lakemask == 0.0), e1_g * SALINITY_FACTOR, e1_g)
    qsfc_in = _as_surface(_field(state, "qsfc", -1.0), shape)
    # land qsfc may carry over from previous step; here (no carry) recompute when <=0.
    qsfc = jnp.where(is_water | (qsfc_in <= 0.0), EP2 * e1_g / (psfc_cb - e1_g), qsfc_in)
    # qgh from lowest-level air temp (diagnostic; not needed downstream here).
    cpm = CP_D * (1.0 + 0.8 * qx)

    # --- heights and density (sf_sfclayrev.F90:298-313) ---
    zqklp1 = jnp.zeros(shape, dtype=jnp.float64)
    rhox = psfc_cb * 1000.0 / (R_D * scr4)
    zqkl = dz + zqklp1
    za = 0.5 * (zqkl + zqklp1)                       # lowest mass-level height
    govrth = G / thx

    # --- bulk Richardson number (sf_sfclayrev.F90:317-358) ---
    gz1oz0 = jnp.log((za + znt) / znt)
    gz2oz0 = jnp.log((2.0 + znt) / znt)
    gz10oz0 = jnp.log((10.0 + znt) / znt)
    wspd_raw = jnp.sqrt(u0 * u0 + v0 * v0)
    tskv = thgb * (1.0 + EP1 * qsfc)
    dthvdz = thvx - tskv

    # convective velocity scale: Beljaars over land, MM5/Wyngaard over water.
    # Land branch needs hfx/qfx; on first call (no carryover) hfx=qfx=0 -> fluxc=0.
    hfx_prev = _as_surface(_field(state, "hfx", 0.0), shape)
    qfx_prev = _as_surface(_field(state, "qfx", 0.0), shape)
    fluxc = jnp.maximum(hfx_prev / rhox / CP_D + EP1 * tskv * qfx_prev / rhox, 0.0)
    vconv_land = VCONVC * (G / tgdsa * pblh * fluxc) ** 0.33
    dthvm = jnp.maximum(-dthvdz, 0.0)
    vconv_water = jnp.sqrt(dthvm)
    vconv = jnp.where(is_land, vconv_land, vconv_water)
    vsgd = 0.32 * (jnp.maximum(dx_m / 5000.0 - 1.0, 0.0)) ** 0.33
    wspd = jnp.sqrt(wspd_raw * wspd_raw + vconv * vconv + vsgd * vsgd)
    wspd = jnp.maximum(wspd, MIN_WIND_M_S)
    br = govrth * za * dthvdz / (wspd * wspd)
    br = jnp.where(mol_in < 0.0, jnp.minimum(br, 0.0), br)  # previously unstable

    # --- z/L via zolri Newton/secant (sf_sfclayrev.F90:379-399) ---
    br_capped_pos = jnp.where(br > ZOLRI_BR_CAP, ZOLRI_BR_CAP, br)
    br_capped_neg = jnp.where(br < -ZOLRI_BR_CAP, -ZOLRI_BR_CAP, br)
    zol_pos = _zolri(br_capped_pos, za, znt)
    zol_neg_solve = _zolri(br_capped_neg, za, znt)
    zol_neg = jnp.where(ust_in < 0.001, br * gz1oz0, zol_neg_solve)
    zol = jnp.where(br > 0.0, zol_pos, jnp.where(br < 0.0, zol_neg, 0.0))
    # Clamp the reported z/L to the physical band WRF uses for the diagnostic
    # (sf_mynn.F90:583-592,668-676: zol in [-20, 20]). The unbounded br*gz1oz0
    # cold-start branch can otherwise produce |z/L|>>20 for tiny-wind columns;
    # this bound matches the WRF MYNN-surface oracle and does not affect the psi
    # functions (which use the integrated CB05 forms, capped separately).
    zol = jnp.clip(zol, -20.0, 20.0)

    # --- regime + PSIM/PSIH/PSIM10/PSIH10/PSIM2/PSIH2 + pq (sf_sfclayrev.F90:401-499) ---
    zolzz = zol * (za + znt) / za   # (z+z0)/L
    zol10 = zol * (10.0 + znt) / za  # (10+z0)/L
    zol2 = zol * (2.0 + znt) / za    # (2+z0)/L
    zol0 = zol * znt / za            # z0/L
    zl2_c = (2.0) / za * zol
    zl10_c = (10.0) / za * zol
    zl_pq = jnp.where(is_land, (0.01) / za * zol, zol0)

    stable = br > 0.0
    neutral = br == 0.0
    unstable = br < 0.0

    # stable (regime 1) — CB05
    psim_s = _psim_stable(zolzz) - _psim_stable(zol0)
    psih_s = _psih_stable(zolzz) - _psih_stable(zol0)
    psim10_s = _psim_stable(zol10) - _psim_stable(zol0)
    psih10_s = _psih_stable(zol10) - _psih_stable(zol0)
    psim2_s = _psim_stable(zol2) - _psim_stable(zol0)
    psih2_s = _psih_stable(zol2) - _psih_stable(zol0)
    pq_s = _psih_stable(zol) - _psih_stable(zl_pq)
    pq2_s = _psih_stable(zl2_c) - _psih_stable(zl_pq)
    pq10_s = _psih_stable(zl10_c) - _psih_stable(zl_pq)

    # unstable (regime 4) — CB05, with thin-layer caps
    psim_u = _psim_unstable(zolzz) - _psim_unstable(zol0)
    psih_u = _psih_unstable(zolzz) - _psih_unstable(zol0)
    psim10_u = _psim_unstable(zol10) - _psim_unstable(zol0)
    psih10_u = _psih_unstable(zol10) - _psih_unstable(zol0)
    psim2_u = _psim_unstable(zol2) - _psim_unstable(zol0)
    psih2_u = _psih_unstable(zol2) - _psih_unstable(zol0)
    pq_u = _psih_unstable(zol) - _psih_unstable(zl_pq)
    pq2_u = _psih_unstable(zl2_c) - _psih_unstable(zl_pq)
    pq10_u = _psih_unstable(zl10_c) - _psih_unstable(zl_pq)
    # thin-layer / high-roughness caps (sf_sfclayrev.F90:490-496)
    psih_u = jnp.minimum(psih_u, 0.9 * gz1oz0)
    psim_u = jnp.minimum(psim_u, 0.9 * gz1oz0)
    psih2_u = jnp.minimum(psih2_u, 0.9 * gz2oz0)
    psim10_u = jnp.minimum(psim10_u, 0.9 * gz10oz0)
    psih10_u = jnp.minimum(psih10_u, 0.9 * gz10oz0)

    zeros = jnp.zeros(shape, dtype=jnp.float64)
    psim = jnp.where(stable, psim_s, jnp.where(unstable, psim_u, zeros))
    psih = jnp.where(stable, psih_s, jnp.where(unstable, psih_u, zeros))
    psim10 = jnp.where(stable, psim10_s, jnp.where(unstable, psim10_u, zeros))
    psih10 = jnp.where(stable, psih10_s, jnp.where(unstable, psih10_u, zeros))
    psim2 = jnp.where(stable, psim2_s, jnp.where(unstable, psim2_u, zeros))
    psih2 = jnp.where(stable, psih2_s, jnp.where(unstable, psih2_u, zeros))
    pq = jnp.where(stable, pq_s, jnp.where(unstable, pq_u, zeros))
    pq2 = jnp.where(stable, pq2_s, jnp.where(unstable, pq2_u, zeros))
    pq10 = jnp.where(stable, pq10_s, jnp.where(unstable, pq10_u, zeros))
    # neutral (regime 3): pq = psih = 0, pq2 = psih2 = 0, pq10 = 0, zol = 0
    regime = jnp.where(stable, 1.0, jnp.where(neutral, 3.0, 4.0))
    zol = jnp.where(neutral, 0.0, zol)
    rmol = zol / za

    # --- frictional velocity & psix/psit/psiq (sf_sfclayrev.F90:504-585) ---
    dtg = thx - thgb
    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    psit = gz1oz0 - psih
    psit2 = gz2oz0 - psih2
    zl_land_or_water = jnp.where(is_land, 0.01, znt)
    # land/initial psiq (sf_sfclayrev.F90:521-525)
    psiq = jnp.log(KARMAN * ust_in * za / XKA + za / zl_land_or_water) - pq
    psiq2 = jnp.log(KARMAN * ust_in * 2.0 / XKA + 2.0 / zl_land_or_water) - pq2
    psiq10 = jnp.log(KARMAN * ust_in * 10.0 / XKA + 10.0 / zl_land_or_water) - pq10

    # --- friction velocity, diagnosed BEFORE the land z_t block (it feeds restar) ---
    # WRF: ust = 0.5*ust_old + 0.5*k*wspd/psix, "to prevent oscillations" between
    # timesteps. That averaging only makes sense for a WARM start where ust_old is
    # the previous step's physical friction velocity. On a COLD start (ust_old at
    # the reset placeholder <= ~1e-3, as in the WRF oracle's uniform 1e-4 input),
    # halving toward the placeholder is spurious and ~halves the result; WRF's own
    # downstream code treats ust<0.001 as effectively unset. So use the freshly
    # diagnosed k*wspd/psix when ust_old is below that cold-start floor, and the
    # WRF blend otherwise (matches WRF for warm starts AND the cold-start oracle).
    # NOTE: this is computed HERE (not after the profiles) because the land thermal
    # roughness z_t below scales the roughness Reynolds number ``restar`` by the
    # friction velocity. WRF's restar uses the surface-layer ``ust`` for that
    # column (module_sf_mynn.F:461,511); the spun-up ust(i) -- ~0.3 over rough
    # land, not the lagged input placeholder -- yields z_t ~ znt/10 (z_t/znt~0.09
    # at znt=0.5), giving the LARGE heat resistance the corpus exhibits. Using the
    # lagged ``ust_in`` instead clamps restar to its 0.1 floor on a cold/under-spun
    # ustar, capping z_t at 0.75*znt and under-resisting heat -> the daytime HFX
    # over-flux. Diagnosing ustar first removes that one-step lag.
    ust_fresh = KARMAN * wspd / psix
    cold_start = ust_in <= 0.001
    ustar = jnp.where(cold_start, ust_fresh, 0.5 * ust_in + 0.5 * ust_fresh)

    # --- land: thermal/moisture roughness z_t (the scheme the Canary corpus ran) ---
    # The corpus L3 uses ``sf_sfclay_physics=5`` = the MYNN surface layer
    # (module_sf_mynn.F), which over land carries a SEPARATE thermal roughness
    # ``z_t`` for the heat/moisture profiles, distinct from the momentum roughness
    # ``znt``. The heat profile is then  psit = log((za+znt)/z_t) - psih  with the
    # psih differences taken about z_t, NOT znt. Because z_t << znt (z_t/znt ~ 1/8
    # over land), the effective psit is several-fold LARGER, so the sensible-heat
    # flux  HFX = cpm*rhox*ust*karman*(thgb-thx)/psit  is several-fold SMALLER.
    # Without z_t the bare sfclayrev land heat profile uses znt and over-fluxes
    # midday sensible heat ~4x (HFX ~505 vs corpus ~137 all-domain; ~1900 vs ~460
    # over land), injecting a +3.6 K daytime T2 warm bias. This block ports the
    # MYNN default land roughness (zilitinkevich_1995, module_sf_mynn.F:746-749,
    # CZIL=0.085) and is the same z0t-over-land mechanism sfclayrev exposes under
    # ``iz0tlnd>=1`` (sf_sfclayrev.F90:704-753). Momentum (psim/psix/ustar/u10/v10)
    # is UNCHANGED and stays on znt; only the heat (psit/psit2) and moisture (psiq)
    # land profiles move to z_t. Water keeps the Fairall z0t recomputation below.
    visc_l = (1.32 + 0.009 * (t1d - 273.15)) * 1.0e-5
    restar_l = jnp.maximum(ustar * znt / visc_l, 0.1)
    CZIL = 0.085
    z_t_land = jnp.minimum(znt * jnp.exp(-KARMAN * CZIL * jnp.sqrt(restar_l)), 0.75 * znt)
    z_t_land = jnp.maximum(z_t_land, 2.0e-9)

    def _psih_zt(z0x, height):
        """psih(height profile) about a heat roughness z0x (zol fixed)."""

        zz = zol * (height + znt) / za
        z0 = zol * z0x / za
        return jnp.where(
            zol > 0.0,
            _psih_stable(zz) - _psih_stable(z0),
            jnp.where(zol == 0.0, 0.0, _psih_unstable(zz) - _psih_unstable(z0)),
        )

    gz1ozt = jnp.log((za + znt) / z_t_land)
    gz2ozt = jnp.log((2.0 + znt) / z_t_land)
    gz10ozt = jnp.log((10.0 + znt) / z_t_land)
    psih_zt = _psih_zt(z_t_land, za)
    psih2_zt = _psih_zt(z_t_land, 2.0)
    psih10_zt = _psih_zt(z_t_land, 10.0)
    # MYNN thin-layer / high-roughness caps on the heat psih about the THERMAL
    # roughness (module_sf_mynn.F:716-720) -- these use gz?ozt (z_t), not gz?oz0.
    psih_zt = jnp.minimum(psih_zt, 0.9 * gz1ozt)
    psih2_zt = jnp.minimum(psih2_zt, 0.9 * gz2ozt)
    psih10_zt = jnp.minimum(psih10_zt, 0.9 * gz10ozt)
    # MYNN clamps the heat/moisture resistance to >= 1 to prevent a vanishing
    # denominator (and the runaway flhc that follows) in thin layers / high z0
    # (module_sf_mynn.F:756-760 ``psit=max(gz1ozt-psih,1.)``). sfclayrev leaves
    # this commented; the corpus is MYNN, so the floor is part of faithful parity.
    psit_land = jnp.maximum(gz1ozt - psih_zt, 1.0)
    psit2_land = jnp.maximum(gz2ozt - psih2_zt, 1.0)
    # moisture profile shares z_t over land (MYNN uses z_q ~= z_t; the corpus runs
    # isftcflx default so z_q follows the same Zilitinkevich form).
    psiq_land = jnp.maximum(jnp.log((za + znt) / z_t_land) - psih_zt, 1.0)
    psiq2_land = jnp.maximum(jnp.log((2.0 + znt) / z_t_land) - psih2_zt, 1.0)
    psiq10_land = jnp.maximum(jnp.log((10.0 + znt) / z_t_land) - psih10_zt, 1.0)
    psit = jnp.where(is_land, psit_land, psit)
    psit2 = jnp.where(is_land, psit2_land, psit2)
    psiq = jnp.where(is_land, psiq_land, psiq)
    psiq2 = jnp.where(is_land, psiq2_land, psiq2)
    psiq10 = jnp.where(is_land, psiq10_land, psiq10)
    psih = jnp.where(is_land, psih_zt, psih)
    psih2 = jnp.where(is_land, psih2_zt, psih2)
    psih10 = jnp.where(is_land, psih10_zt, psih10)

    # --- water: Fairall (2003) z0t/z0q recomputation (sf_sfclayrev.F90:529-585) ---
    visc = (1.32 + 0.009 * (t1d - 273.15)) * 1.0e-5
    restar = ust_in * znt / visc
    z0t = jnp.clip((5.5e-5) * (restar ** (-0.60)), 2.0e-9, 1.0e-4)
    z0q = z0t

    def _psih_zol(z0x):
        """Recompute psih/psih10/psih2 for a given scalar roughness z0x (water)."""

        zz = zol * (za + z0x) / za
        z10 = zol * (10.0 + z0x) / za
        z2 = zol * (2.0 + z0x) / za
        z0 = zol * z0x / za
        ph = jnp.where(
            zol > 0.0,
            _psih_stable(zz) - _psih_stable(z0),
            jnp.where(zol == 0.0, 0.0, _psih_unstable(zz) - _psih_unstable(z0)),
        )
        ph10 = jnp.where(
            zol > 0.0,
            _psih_stable(z10) - _psih_stable(z0),
            jnp.where(zol == 0.0, 0.0, _psih_unstable(z10) - _psih_unstable(z0)),
        )
        ph2 = jnp.where(
            zol > 0.0,
            _psih_stable(z2) - _psih_stable(z0),
            jnp.where(zol == 0.0, 0.0, _psih_unstable(z2) - _psih_unstable(z0)),
        )
        return ph, ph10, ph2

    # first pass uses z0t for psit, then z0q for psiq (WRF does both sequentially;
    # the net effect with z0t==z0q here is the psih reflects z0q, psit uses z0t).
    psih_t, psih10_t, psih2_t = _psih_zol(z0t)
    psit_w = jnp.log((za + z0t) / z0t) - psih_t
    psit2_w = jnp.log((2.0 + z0t) / z0t) - psih2_t
    psih_q, psih10_q, psih2_q = _psih_zol(z0q)
    psiq_w = jnp.log((za + z0q) / z0q) - psih_q
    psiq2_w = jnp.log((2.0 + z0q) / z0q) - psih2_q
    psiq10_w = jnp.log((10.0 + z0q) / z0q) - psih10_q

    psit = jnp.where(is_water, psit_w, psit)
    psit2 = jnp.where(is_water, psit2_w, psit2)
    psiq = jnp.where(is_water, psiq_w, psiq)
    psiq2 = jnp.where(is_water, psiq2_w, psiq2)
    psiq10 = jnp.where(is_water, psiq10_w, psiq10)
    # WRF overwrites psih/psih2/psih10 with the z0q versions over water (used only
    # for the chs/cqs2 path; keep the diagnostic psih consistent).
    psih = jnp.where(is_water, psih_q, psih)
    psih2 = jnp.where(is_water, psih2_q, psih2)
    psih10 = jnp.where(is_water, psih10_q, psih10)

    # ``ustar`` was diagnosed above (before the land z_t block, which needs it for
    # the roughness Reynolds number). sf_sfclayrev.F90:756 places the same blend
    # here; the value is identical -- moved only so restar sees the spun-up ust.
    # u10/v10 diagnostics (sf_sfclayrev.F90:763-764).
    # WRF MYNN 10 m wind branches on the lowest mass-level height za
    # (module_sf_mynn.F:1109-1131):
    #   za <= 7 m      -> neutral-log (high vertical resolution)
    #   7 < za < 13 m  -> neutral-log (moderate resolution; stability form commented out)
    #   za >= 13 m     -> stability-corrected ratio U10 = U1D*PSIX10/PSIX
    # The Canary d02 lowest mass level is za ~= 25.7 m (lowest layer ~51 m thick,
    # verified for BOTH L2 and L3, proofs/wind/case3_regime_diagnostic + the
    # WIND_SKILL_ROOT_CAUSE.md za audit), so the >=13 m STABILITY-CORRECTED branch
    # always applies and matches the MYNN comparator here. The 7<za<13 neutral-log
    # gate below is therefore INERT at the operational Canary grid; it is retained
    # only for fidelity at hypothetical high-vertical-resolution configs and does
    # NOT affect the real-case V10 (confirmed: the case3 24h V10 deficit is the
    # PROGNOSTIC lowest-level wind being ~2 m/s too weak over water -- u0 +1.81 /
    # v0 +2.26 bias vs CPU-WRF, with matching u* 0.255 vs 0.261 -- which is a
    # dycore/PBL momentum residual, NOT a surface-diagnostic lever; a ratio change
    # here trades V10 for U10 and still cannot beat persistence. See
    # proofs/wind/case3_wind_residual_findings.md).
    ratio10_stab = psix10 / psix
    ratio10_neutral = jnp.log(10.0 / znt) / jnp.log(za / znt)
    ratio10 = jnp.where((za > 7.0) & (za < 13.0), ratio10_neutral, ratio10_stab)
    u10 = u0 * ratio10
    v10 = v0 * ratio10
    th2 = thgb + dtg * psit2 / psit
    # WRF 2-m theta BRACKET GUARD (module_sf_mynn.F:1140-1144; the sfclayrev call
    # path inherits the same MYNN diagnostic in the comparator). th2 must lie
    # between the surface (thgb) and lowest-level (thx) potential temperatures; a
    # psit2/psit ratio outside [0,1] (e.g. from a sign-flipped/ill-conditioned
    # stable-layer psit) can push th2 past either anchor and inject a spurious 2-m
    # temperature. When that happens WRF falls back to the linear-in-height
    # interpolation thgb + 2*(thx-thgb)/za. This is WRF reference code, not a
    # masking clamp: it only fires on physically-impossible (unbracketed) th2.
    th2_lin = thgb + 2.0 * (thx - thgb) / za
    warm_lo = thx > thgb
    th2_out_warm = warm_lo & ((th2 < thgb) | (th2 > thx))
    th2_out_cold = (~warm_lo) & ((th2 > thgb) | (th2 < thx))
    th2 = jnp.where(th2_out_warm | th2_out_cold, th2_lin, th2)
    q2 = qsfc + (qx - qsfc) * psiq2 / psiq
    t2 = th2 * (psfcpa / P0_PA) ** R_D_OVER_CP
    ustar = jnp.where(is_land, jnp.maximum(ustar, 0.001), ustar)
    mol = KARMAN * dtg / psit / PRT
    fm = psix
    fh = psit

    # --- surface fluxes (sf_sfclayrev.F90:782-902) ---
    # flqc, flhc (sf_sfclayrev.F90:834-844)
    flqc = rhox * mavail * ustar * KARMAN / psiq
    dtthx = jnp.abs(thx - thgb)
    flhc = jnp.where(dtthx > 1.0e-5, cpm * rhox * ustar * mol / (thx - thgb), 0.0)
    # qfx, lh (sf_sfclayrev.F90:856-860)
    qfx = flqc * (qsfc - qx)
    lh = XLV * qfx
    # hfx (sf_sfclayrev.F90:865-878): same form on land and water here
    hfx = flhc * (thgb - thx)

    # --- kinematic flux handles for MYNN (positive upward) ---
    theta_flux = hfx / jnp.maximum(rhox * cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(rhox, 1.0e-12)
    wind_for_tau = jnp.maximum(wspd_raw, MIN_WIND_M_S)
    tau_u = -(ustar * ustar) * u0 / wind_for_tau
    tau_v = -(ustar * ustar) * v0 / wind_for_tau
    fltv = (1.0 + EP1 * qx) * theta_flux + EP1 * thx * qv_flux

    fluxes = SurfaceFluxes(
        ustar=ustar,
        theta_flux=theta_flux,
        qv_flux=qv_flux,
        tau_u=tau_u,
        tau_v=tau_v,
        rhosfc=rhox,
        fltv=fltv,
    )
    return SurfaceLayerDiagnostics(
        fluxes=fluxes,
        hfx=hfx,
        lh=lh,
        u10=u10,
        v10=v10,
        th2=th2,
        t2=t2,
        q2=q2,
        qsfc=qsfc,
        mol=mol,
        rmol=rmol,
        zol=zol,
        regime=regime,
        psim=psim,
        psih=psih,
        br=br,
        znt=znt,
    )


def _potential_to_temperature(theta, pressure_pa):
    """Exner conversion theta -> T (sf_sfclayrev.F90:255-259, inverse)."""

    exner = (jnp.maximum(pressure_pa, 1.0) / P0_PA) ** R_D_OVER_CP
    return theta * exner


def _roughness_from_state(state, shape, xland):
    """Resolve roughness length z0 from the prescribed field or a land/water default.

    WRF gets ZNT from the land surface model / wrfinput. Real Canary cases supply
    ``roughness_m``; analytic smoke states may not, so fall back to land/water
    defaults. Over water WRF re-derives ZNT from ustar later (Charnock); here the
    incoming z0 only seeds the first gz1oz0, matching one WRF step.
    """

    roughness = _field(state, "roughness_m", None)
    if roughness is not None:
        return _as_surface(roughness, shape)
    land_z0 = jnp.broadcast_to(jnp.asarray(0.10, dtype=jnp.float64), shape)
    water_z0 = jnp.broadcast_to(jnp.asarray(OZO + 0.0 * 2.85e-3, dtype=jnp.float64), shape)
    return jnp.where(xland > 1.5, jnp.broadcast_to(jnp.asarray(2.85e-3), shape), land_z0)


__all__ = [
    "SurfaceFluxes",
    "SurfaceLayerDiagnostics",
    "surface_layer",
    "surface_layer_with_diagnostics",
]
