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
    SFCLAYREV_TABLE_DZOL,
    SFCLAYREV_TABLE_N,
    SVP1_KPA,
    SVP2,
    SVP3_K,
    SVPT0_K,
    VCONVC_MYNN,
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
# zolrib brute-force fixed-point z/L solve (module_sf_mynn.F:1984-2048)
# ==================================================================================


def _zolrib(ri, za, z0, zt, logz0, logzt, zol1_seed=None):
    """MYNN ``zolrib`` brute-force z/L solve, module_sf_mynn.F:1984-2048.

    Unlike sfclayrev's secant ``_zolri`` (which builds BOTH residual terms from the
    momentum roughness), MYNN's ``zolrib`` builds the heat residual ``psit2`` from
    the THERMAL roughness ``zt`` (log ``logzt`` and the ``psih`` difference taken
    about ``zolt = zol*zt/za``) and the momentum residual ``psix2`` from ``z0``.
    It is a fixed-point iteration on ``zolrib`` itself, NOT a secant method:

        zol20 = zol*z0/za ; zol3 = zol + zol20 ; zolt = zol*zt/za
        psit2 = max(logzt - (psih(zol3) - psih(zolt)), 1.0)
        psix2 = max(logz0 - (psim(zol3) - psim(zol20)), 1.0)
        zol_new = ri * psix2**2 / psit2

    WRF seeds ``zolold`` with +/-99999 on the first trip then takes ``zolold = zol1``
    (the first guess) on n==1, i.e. the iteration's first refinement uses ``zol1``.
    The endpoint is frozen once ``|zol_new - zol_old| <= 0.01`` (``nmax=20``). On
    non-convergence WRF falls back to ``Li_etal_2010(zolrib, ri, za/z0, z0/zt)``
    (line 2039); ``_li_etal_2010`` mirrors that here.

    Vectorized, branch-free: ``ri`` is signed per cell, so ``psih``/``psim`` pick the
    unstable form where ``ri<0`` and the stable form where ``ri>=0`` per element.
    """

    unstable = ri < 0.0

    def residual(zol_old):
        zol20 = zol_old * z0 / za
        zol3 = zol_old + zol20
        zolt = zol_old * zt / za
        psit2_u = jnp.maximum(logzt - (_psih_unstable(zol3) - _psih_unstable(zolt)), 1.0)
        psix2_u = jnp.maximum(logz0 - (_psim_unstable(zol3) - _psim_unstable(zol20)), 1.0)
        psit2_s = jnp.maximum(logzt - (_psih_stable(zol3) - _psih_stable(zolt)), 1.0)
        psix2_s = jnp.maximum(logz0 - (_psim_stable(zol3) - _psim_stable(zol20)), 1.0)
        psit2 = jnp.where(unstable, psit2_u, psit2_s)
        psix2 = jnp.where(unstable, psix2_u, psix2_s)
        return ri * psix2 * psix2 / psit2

    # n==1: zolold = zol1 (the first guess). For a WARM step (itimestep>1) WRF seeds
    # the MOL-based guess (module_sf_mynn.F:796/881); for the very first step it seeds
    # Li_etal_2010 (lines 794/879). zolrib is a fixed-point that is NOT globally
    # contractive, so the seed can select between roots -- pass the operational
    # warm-step seed (``zol1_seed``) to stay WRF-faithful; fall back to Li for the
    # cold start / when no seed is provided.
    zol1 = _li_etal_2010(ri, za / z0, z0 / zt) if zol1_seed is None else zol1_seed
    # WRF wrong-quadrant guard (module_sf_mynn.F:1998): zol1*ri<0 -> zol1=0.
    zol1 = jnp.where(zol1 * ri < 0.0, 0.0, zol1)

    def body(_, carry):
        zol_old, frozen = carry
        zol_new = residual(zol_old)
        converged = jnp.abs(zol_new - zol_old) <= 0.01
        # freeze the endpoint once converged (matches WRF early-stop)
        nxt = jnp.where(frozen, zol_old, zol_new)
        new_frozen = frozen | converged
        return (nxt, new_frozen)

    frozen0 = jnp.zeros_like(ri, dtype=bool)
    zol_conv, conv_flag = jax.lax.fori_loop(0, 20, body, (zol1, frozen0))

    # non-convergence fallback (module_sf_mynn.F:2036-2039)
    zol_fallback = _li_etal_2010(ri, za / z0, z0 / zt)
    return jnp.where(conv_flag, zol_conv, zol_fallback)


def _li_etal_2010(rib, zaz0, z0zt):
    """Li et al. (2010) closed-form z/L, module_sf_mynn.F:1831-1890.

    Robust analytic z/L matching Hogstrom (1996) (unstable) and Beljaars & Holtslag
    (1991) (stable). Used by ``zolrib`` as the first guess (n==1, itimestep<=1) and
    the non-convergence fallback.
    """

    au11, bu11, bu12 = 0.045, 0.003, 0.0059
    bu21, bu22, bu31, bu32, bu33 = -0.0828, 0.8845, 0.1739, -0.9213, -0.1057
    aw11, aw12, aw21, aw22 = 0.5738, -0.4399, -4.901, 52.50
    bw11, bw12, bw21, bw22 = -0.0539, 1.540, -0.669, -3.282
    as11, as21, bs11, bs21, bs22 = 0.7529, 14.94, 0.1569, -0.3091, -1.303

    zaz02 = jnp.clip(zaz0, 100.0, 100000.0)
    z0zt2 = jnp.clip(z0zt, 0.5, 100.0)
    alfa = jnp.log(zaz02)
    beta = jnp.log(z0zt2)

    zl_uns = au11 * alfa * rib**2 + (
        (bu11 * beta + bu12) * alfa**2
        + (bu21 * beta + bu22) * alfa
        + (bu31 * beta**2 + bu32 * beta + bu33)
    ) * rib
    zl_uns = jnp.clip(zl_uns, -15.0, 0.0)

    zl_wsta = (
        ((aw11 * beta + aw12) * alfa + (aw21 * beta + aw22)) * rib**2
        + ((bw11 * beta + bw12) * alfa + (bw21 * beta + bw22)) * rib
    )
    zl_wsta = jnp.clip(zl_wsta, 0.0, 4.0)

    zl_ssta = (as11 * alfa + as21) * rib + bs11 * alfa + bs21 * beta + bs22
    zl_ssta = jnp.clip(zl_ssta, 1.0, 20.0)

    return jnp.where(
        rib <= 0.0,
        zl_uns,
        jnp.where(rib <= 0.2, zl_wsta, zl_ssta),
    )


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

    # --- surface saturation humidity QSFC / QSFCMR (module_sf_mynn.F:522-537) ---
    # MYNN uses the Bolton(1980) SVP over water and an explicit ice formula when
    # TSK<273.15 (NO salinity factor -- that is a sfclayrev feature). It carries TWO
    # surface humidities: QSFC = specific humidity = EP2*E1/(PSFC-ep_3*E1) (used in
    # THVGB and the q2 anchor) and QSFCMR = mixing ratio = EP2*E1/(PSFC-E1) (used in
    # the QFX flux). Over land both come from the carried-over QSFC when QSFC>0.
    ep_3 = 1.0 - EP2
    e1_g = jnp.where(
        tgdsa < 273.15,
        SVP1_KPA * jnp.exp(4648.0 * (1.0 / 273.15 - 1.0 / tgdsa) - 11.64 * jnp.log(273.15 / tgdsa) + 0.02265 * (273.15 - tgdsa)),
        SVP1_KPA * jnp.exp(SVP2 * (tgdsa - SVPT0_K) / (tgdsa - SVP3_K)),
    )
    qsfc_in = _as_surface(_field(state, "qsfc", -1.0), shape)
    recompute_q = is_water | (qsfc_in <= 0.0)
    # QSFC (specific humidity): recompute over water/<=0 land, else carried value.
    qsfc = jnp.where(recompute_q, EP2 * e1_g / (psfc_cb - ep_3 * e1_g), qsfc_in)
    # QSFCMR (mixing ratio): from the recompute, else qsfc/(1-qsfc) (spec hum -> mr).
    qsfcmr = jnp.where(recompute_q, EP2 * e1_g / (psfc_cb - e1_g), qsfc_in / (1.0 - qsfc_in))
    # MYNN moist heat capacity (module_sf_mynn.F:552): CPM = CP*(1+0.84*QV1D), where
    # QV1D is the lowest-level MIXING ratio. (sfclayrev uses 0.8; this is the MYNN-SL
    # path -> 0.84.)
    cpm = CP_D * (1.0 + 0.84 * qx)

    # --- heights and density (sf_sfclayrev.F90:298-313) ---
    zqklp1 = jnp.zeros(shape, dtype=jnp.float64)
    rhox = psfc_cb * 1000.0 / (R_D * scr4)
    zqkl = dz + zqklp1
    za = 0.5 * (zqkl + zqklp1)                       # lowest mass-level height
    govrth = G / thx

    # --- bulk Richardson number (sf_sfclayrev.F90:317-358) ---
    # (gz?oz0 momentum logs are deferred until AFTER the water-z0 Charnock update,
    # because WRF computes them on the UPDATED znt -- module_sf_mynn.F:755-760.)
    wspd_raw = jnp.sqrt(u0 * u0 + v0 * v0)
    tskv = thgb * (1.0 + EP1 * qsfc)
    dthvdz = thvx - tskv

    # Convective velocity scale WSTAR + subgrid VSGD (module_sf_mynn.F:564-586).
    # MYNN uses the Beljaars (1995) convective form over BOTH land and water (NOT the
    # MM5/Wyngaard sqrt(-dthv) form), with VCONVC=1.25 and -- over land -- an
    # increased mixing height min(1.5*pblh,4000) to represent non-local mass-flux
    # transport above the PBL top. fluxc uses THVGB (=tskv) and g/TSK (=g/tgdsa).
    # On a cold start (no hfx/qfx carryover) fluxc=0 so WSTAR=0.
    hfx_prev = _as_surface(_field(state, "hfx", 0.0), shape)
    qfx_prev = _as_surface(_field(state, "qfx", 0.0), shape)
    fluxc = jnp.maximum(hfx_prev / rhox / CP_D + EP1 * tskv * qfx_prev / rhox, 0.0)
    height_land = jnp.minimum(1.5 * pblh, 4000.0)            # module_sf_mynn.F:578
    vconv_land = VCONVC_MYNN * (G / tgdsa * height_land * fluxc) ** 0.33  # :578
    vconv_water = VCONVC_MYNN * (G / tgdsa * pblh * fluxc) ** 0.33        # :574
    vconv = jnp.where(is_land, vconv_land, vconv_water)
    vsgd = 0.32 * (jnp.maximum(dx_m / 5000.0 - 1.0, 0.0)) ** 0.33
    wspd = jnp.sqrt(wspd_raw * wspd_raw + vconv * vconv + vsgd * vsgd)
    wspd = jnp.maximum(wspd, MIN_WIND_M_S)
    br = govrth * za * dthvdz / (wspd * wspd)
    # itimestep>1 bulk-Ri clamp (module_sf_mynn.F:597-600). The "if previously
    # unstable -> BR<=0" block (lines 603-605) is COMMENTED OUT in WRF; do not apply.
    br = jnp.clip(br, -4.0, 4.0)

    # ==============================================================================
    # MYNN thermal/moisture roughness z_t/z_q, computed BEFORE the z/L solve
    # (module_sf_mynn.F:671-760). WRF orders restar -> z_t/z_q -> GZ?OZt -> the
    # zolrib z/L solve, because zolrib's heat residual needs the thermal log GZ1OZt.
    # restar uses the PRIOR-STEP ust (the INTENT(INOUT) UST as it enters the column,
    # line 675/725) -- NOT a blended/look-ahead value; the in-step ust update happens
    # later (line 949), after the z/L solve (spec Mismatch 2).
    # ==============================================================================
    # MYNN kinematic viscosity, Andreas (1989) cubic in T_celsius (module_sf_mynn.F:
    # 622-623) -- NOT the sfclayrev linear form.
    tc1d = t1d - 273.15
    visc = 1.326e-5 * (1.0 + 6.542e-3 * tc1d + 8.301e-6 * tc1d ** 2 - 4.84e-9 * tc1d ** 3)

    # WATER aerodynamic z0: WRF re-derives ZNT from the COARE 3.0 Charnock relation
    # (module_sf_mynn.F:635 -> charnock_1955) every step using the current UST/WSPD,
    # then computes restar with the NEW znt (line 675). The seeded land/water z0 only
    # set the FIRST GZ1OZ0; faithfully updating water z0 here is required (otherwise
    # z0 stays at the ~2.85e-3 seed instead of the physical ~1e-4 open-ocean value).
    wsp10m = wspd * jnp.log(10.0 / 1.0e-4) / jnp.log(za / 1.0e-4)
    czc = 0.011 + 0.007 * jnp.clip((wsp10m - 10.0) / 8.0, 0.0, 1.0)   # variable Charnock
    znt_water = jnp.clip(
        czc * ust_in * ust_in / G + 0.11 * visc / jnp.maximum(ust_in, 0.05),
        1.27e-7, 2.85e-3,
    )
    znt = jnp.where(is_water, znt_water, znt)

    restar = jnp.maximum(ust_in * znt / visc, 0.1)   # module_sf_mynn.F:675/725 (NEW znt)

    # LAND: zilitinkevich_1995 default (IZ0TLND<=1, CZIL=0.085), z_q == z_t, NO lower
    # floor -- only MIN(z_t, 0.75*z0) (module_sf_mynn.F:1252-1265). Snow/ice
    # (Andreas_2002) and spp_pbl perturbations are out of scope (no-snow Canary,
    # spp_pbl=0); see spec "Out of scope".
    CZIL = 0.085
    z_t_land = jnp.minimum(znt * jnp.exp(-KARMAN * CZIL * jnp.sqrt(restar)), 0.75 * znt)
    # WATER: fairall_etal_2003 (COARE_OPT=3.0 default), z_q == z_t
    # (module_sf_mynn.F:1442-1467): Zt = 5.5e-5*restar^-0.6, clipped [2e-9, 1e-4].
    z_t_water = jnp.clip((5.5e-5) * (restar ** (-0.60)), 2.0e-9, 1.0e-4)
    z_t = jnp.where(is_land, z_t_land, z_t_water)
    z_q = z_t  # zilitinkevich land + fairall water both set z_q = z_t

    # momentum + thermal logs (module_sf_mynn.F:755-760), on the UPDATED znt; the
    # numerator is (height + ZNTstoch) for both z0 and z_t.
    gz1oz0 = jnp.log((za + znt) / znt)
    gz2oz0 = jnp.log((2.0 + znt) / znt)
    gz10oz0 = jnp.log((10.0 + znt) / znt)
    gz1ozt = jnp.log((za + znt) / z_t)
    gz2ozt = jnp.log((2.0 + znt) / z_t)
    gz10ozt = jnp.log((10.0 + znt) / z_t)

    # --- z/L via MYNN zolrib brute-force solve (module_sf_mynn.F:804/889) ---
    # zolrib's heat residual uses the THERMAL roughness z_t. The WARM-step first guess
    # (itimestep>1) is the MOL-based estimate ZA*k*g*MOL/(TH1D*max(ust^2,eps)) clamped
    # per sign (module_sf_mynn.F:796-798 stable / 881-883 unstable). BR is already
    # clamped to [-4,4] above; the extra ZOLRI_BR_CAP guard is now inert but harmless.
    br_capped = jnp.clip(br, -ZOLRI_BR_CAP, ZOLRI_BR_CAP)
    zol_guess_s = jnp.clip(za * KARMAN * G * mol_in / (thx * jnp.maximum(ust_in ** 2, 1.0e-4)), 0.0, 20.0)
    zol_guess_u = jnp.clip(za * KARMAN * G * mol_in / (thx * jnp.maximum(ust_in ** 2, 1.0e-3)), -20.0, 0.0)
    zol1_seed = jnp.where(br > 0.0, zol_guess_s, jnp.where(br < 0.0, zol_guess_u, 0.0))
    zol = _zolrib(br_capped, za, znt, z_t, gz1oz0, gz1ozt, zol1_seed=zol1_seed)
    # per-sign clamp (module_sf_mynn.F:805-806 stable -> [0,20]; 890-891 unstable ->
    # [-20,0]); neutral (br==0) -> zol=0 (module_sf_mynn.F:863).
    zol = jnp.where(
        br > 0.0,
        jnp.clip(zol, 0.0, 20.0),
        jnp.where(br < 0.0, jnp.clip(zol, -20.0, 0.0), 0.0),
    )

    # ==============================================================================
    # PSIM / PSIH / PSIH2 / PSIH10 (module_sf_mynn.F:808-935), unified land+water.
    # WRF computes the SAME structure on land and water (the only land/water
    # difference is the z_t/z_q formula above): the MOMENTUM psi (psim/psim10/psim2)
    # and the 2 m/10 m HEAT psi (psih2/psih10) are taken about the MOMENTUM baseline
    # ``zolz0``; only the lowest-level HEAT psi (psih) is taken about the THERMAL
    # baseline ``zolzt`` (spec Mismatch 3). The current GPU previously (a) solved a
    # momentum-only z/L, (b) split land off into a thermal-baseline psih2/psih10,
    # (c) ran water through a separate sfclayrev z0t recompute -- all replaced here
    # by the literal MYNN path.
    # ==============================================================================
    zolzt = zol * z_t / za            # zt/L  (module_sf_mynn.F:808/893)
    zolz0 = zol * znt / za            # z0/L  (module_sf_mynn.F:809/894)
    zolza = zol * (za + znt) / za     # (za+z0)/L (module_sf_mynn.F:810/895)
    zol10 = zol * (10.0 + znt) / za   # (10+z0)/L (module_sf_mynn.F:811/896)
    zol2 = zol * (2.0 + znt) / za     # (2+z0)/L  (module_sf_mynn.F:812/897)

    stable = br > 0.0
    neutral = br == 0.0
    unstable = br < 0.0

    # stable (regime 1/2) -- module_sf_mynn.F:823-827 (water) / 836-840 (land)
    psim_s = _psim_stable(zolza) - _psim_stable(zolz0)
    psih_s = _psih_stable(zolza) - _psih_stable(zolzt)   # THERMAL baseline (zolzt)
    psim10_s = _psim_stable(zol10) - _psim_stable(zolz0)
    psih10_s = _psih_stable(zol10) - _psih_stable(zolz0)  # MOMENTUM baseline (zolz0)
    psim2_s = _psim_stable(zol2) - _psim_stable(zolz0)
    psih2_s = _psih_stable(zol2) - _psih_stable(zolz0)    # MOMENTUM baseline (zolz0)

    # unstable (regime 4) -- module_sf_mynn.F:907-922
    psim_u = _psim_unstable(zolza) - _psim_unstable(zolz0)
    psih_u = _psih_unstable(zolza) - _psih_unstable(zolzt)   # THERMAL baseline
    psim10_u = _psim_unstable(zol10) - _psim_unstable(zolz0)
    psih10_u = _psih_unstable(zol10) - _psih_unstable(zolz0)  # MOMENTUM baseline
    psim2_u = _psim_unstable(zol2) - _psim_unstable(zolz0)
    psih2_u = _psih_unstable(zol2) - _psih_unstable(zolz0)    # MOMENTUM baseline

    zeros = jnp.zeros(shape, dtype=jnp.float64)
    psim = jnp.where(stable, psim_s, jnp.where(unstable, psim_u, zeros))
    psih = jnp.where(stable, psih_s, jnp.where(unstable, psih_u, zeros))
    psim10 = jnp.where(stable, psim10_s, jnp.where(unstable, psim10_u, zeros))
    psih10 = jnp.where(stable, psih10_s, jnp.where(unstable, psih10_u, zeros))
    psim2 = jnp.where(stable, psim2_s, jnp.where(unstable, psim2_u, zeros))
    psih2 = jnp.where(stable, psih2_s, jnp.where(unstable, psih2_u, zeros))

    # thin-layer / high-roughness caps -- module_sf_mynn.F:931-935. NOTE the WRF
    # cap is applied ONLY in the unstable (BR<0) block; the stable block has no
    # caps. The heat caps use the THERMAL logs (gz?ozt) even though psih2/psih10
    # are on the momentum baseline -- the cap and the baseline disagree on purpose
    # (spec Mismatch 3). The momentum caps use gz?oz0.
    psih_capped = jnp.minimum(psih, 0.9 * gz1ozt)          # 931: 0.9*GZ1OZt
    psim_capped = jnp.minimum(psim, 0.9 * gz1oz0)          # 932: 0.9*GZ1OZ0
    psih2_capped = jnp.minimum(psih2, 0.9 * gz2ozt)        # 933: 0.9*GZ2OZt
    psim10_capped = jnp.minimum(psim10, 0.9 * gz10oz0)     # 934: 0.9*GZ10OZ0
    psih10_capped = jnp.minimum(psih10, 0.9 * gz10ozt)     # 935: 0.9*GZ10OZt
    psih = jnp.where(unstable, psih_capped, psih)
    psim = jnp.where(unstable, psim_capped, psim)
    psih2 = jnp.where(unstable, psih2_capped, psih2)
    psim10 = jnp.where(unstable, psim10_capped, psim10)
    psih10 = jnp.where(unstable, psih10_capped, psih10)

    # Regime class (module_sf_mynn.F:783-790, 855, 875): BR>0.2 -> 1 (nighttime
    # stable), 0<BR<=0.2 -> 2 (damped mechanical turbulence), BR==0 -> 3 (neutral),
    # BR<0 -> 4 (free convection). Diagnostic only; the PSI path is identical for
    # regimes 1 and 2 (both use the stable branch).
    regime = jnp.where(br > 0.2, 1.0, jnp.where(stable, 2.0, jnp.where(neutral, 3.0, 4.0)))
    zol = jnp.where(neutral, 0.0, zol)
    rmol = zol / za

    # ==============================================================================
    # Friction velocity (module_sf_mynn.F:945-962) and thermal/moisture resistances
    # (module_sf_mynn.F:969-977). The ust update is the WRF blend with the
    # prior-step ust_in, placed AFTER the z/L solve (it does NOT feed restar/z_t;
    # restar used ust_in above -- spec Mismatch 2). Land floor is 0.005 per
    # module_sf_mynn.F:959 (no cold-start special-case: the replay path is warm).
    # ==============================================================================
    dtg = thx - thgb
    psix = gz1oz0 - psim
    psix10 = gz10oz0 - psim10
    ustar = 0.5 * ust_in + 0.5 * KARMAN * wspd / psix   # module_sf_mynn.F:949
    ustar = jnp.where(is_land, jnp.maximum(ustar, 0.005), ustar)  # 959

    # Resistances (module_sf_mynn.F:972-977). Numerators always use the thermal /
    # moisture roughness; PSIT/PSIQ subtract the thermal-baseline PSIH; PSIT2/PSIQ2
    # subtract the MOMENTUM-baseline PSIH2; PSIQ10 subtracts the momentum-baseline
    # PSIH10. The >=1 floor prevents a vanishing flux denominator (thin layers/high
    # z0). z_q == z_t so the moisture numerators reuse the thermal logs.
    psit = jnp.maximum(gz1ozt - psih, 1.0)                       # 972
    psit2 = jnp.maximum(gz2ozt - psih2, 1.0)                     # 973
    psiq = jnp.maximum(jnp.log((za + znt) / z_q) - psih, 1.0)    # 975
    psiq2 = jnp.maximum(jnp.log((2.0 + znt) / z_q) - psih2, 1.0)  # 976
    psiq10 = jnp.maximum(jnp.log((10.0 + znt) / z_q) - psih10, 1.0)  # 977

    # --- 2 m thermal/moisture weight: FAITHFUL MYNN-SL ``psit2/psit`` ---
    # WRF's MYNN-SL 2-m diagnostic is ``TH2 = THGB + DTG*PSIT2/PSIT`` (module_sf_mynn.F
    # :1138); the port uses that ratio everywhere — this is exactly module_sf_mynn.F.
    #
    # Over a Noah-MP LAND point real WRF then OVERWRITES this MYNN 2-m value with the
    # LSM diagnostic ``T2 = FVEG*T2MV + (1-FVEG)*T2MB`` (module_sf_mynn.F:1135
    # "OVERWRITTEN FOR LAND POINTS IN THE LSM"; module_surface_driver.F:3469-3473).
    # That overwrite is now done FAITHFULLY in the Noah-MP coupler from the genuine
    # Noah-MP T2MV/T2MB diagnostics (noahmp_surface_hook.overlay_noahmp_land_diagnostics
    # / proofs/v090/noahmp_t2mb_parity.json), so this surface-layer module is left as
    # the pure, savepoint-faithful module_sf_mynn.F 2-m diagnostic (the value WRF uses
    # over water and as the pre-overwrite seed over land). The earlier opt-in empirical
    # ``lsm_t2_diag`` bare-ground stand-in for the missing LSM step has been RETIRED —
    # the real T2MB/T2MV path supersedes it.
    w2m = psit2 / psit                                      # faithful MYNN-SL ratio

    # u10/v10 diagnostics (module_sf_mynn.F:1109-1131).
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
    th2 = thgb + dtg * w2m
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
    # Q2 uses the surface MIXING RATIO anchor (module_sf_mynn.F:1147), faithful MYNN
    # ``psiq2/psiq`` (the land Q2 overwrite to the Noah-MP LSM Q2 is owned by the
    # coupler, mirroring the T2 overwrite, when wired).
    q2w = psiq2 / psiq
    q2 = qsfcmr + (qx - qsfcmr) * q2w
    # MYNN 2-m mixing-ratio brackets (module_sf_mynn.F:1148-1149):
    # Q2 = MAX(Q2, MIN(QSFCMR, QV1D)) then Q2 = MIN(Q2, 1.05*QV1D). WRF reference
    # code (bracket floor + ceiling), not a masking clamp.
    q2 = jnp.maximum(q2, jnp.minimum(qsfcmr, qx))
    q2 = jnp.minimum(q2, 1.05 * qx)
    t2 = th2 * (psfcpa / P0_PA) ** R_D_OVER_CP
    # T* (theta scale) -- module_sf_mynn.F:981-983: WRF uses the VIRTUAL dtheta
    # (THV1D-THVGB) and the heat resistance PSIT. (The HFX flux below uses the
    # NON-virtual dtheta, thx-thgb, per module_sf_mynn.F:1066/1074.) MOL has no
    # wrfout truth; reported as a diagnostic.
    dtg_v = thvx - tskv
    mol = KARMAN * dtg_v / psit / PRT

    # --- surface fluxes (module_sf_mynn.F:1051-1076) ---
    # FLQC/FLHC exchange coefficients (1051-1052): direct resistance form, NOT the
    # sfclayrev mol/(thx-thgb) form. With the heat resistance PSIT this is the
    # WRF-faithful MYNN flux and is numerically robust at thx==thgb (no divide).
    flqc = rhox * mavail * ustar * KARMAN / psiq
    flhc = cpm * rhox * ustar * KARMAN / psit
    # QFX/LH (1057-1060): QFX uses the surface MIXING RATIO QSFCMR (NOT the specific
    # humidity QSFC); small negative QFX allowed, floored -0.02.
    qfx = flqc * (qsfcmr - qx)
    qfx = jnp.maximum(qfx, -0.02)
    lh = XLV * qfx
    # HFX (1066/1074): same FLHC*(THGB-TH1D) on land and water; land floored -250.
    hfx = flhc * (thgb - thx)
    hfx = jnp.where(is_land, jnp.maximum(hfx, -250.0), hfx)

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
        xland=xland,
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
