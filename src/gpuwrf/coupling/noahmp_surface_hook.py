"""Operational surface-coupling hook for prognostic Noah-MP (Sprint S6b ACTIVATE).

Thin operational wrapper around ``physics.noahmp_coupler.noahmp_surface_adapter``.
It is the drop-in REPLACEMENT for ``coupling.physics_couplers.surface_adapter`` on
the land tile: it runs the land-masked Noah-MP / sfclay blend for one physics step
and writes the blended kinematic flux HANDLES (theta_flux/qv_flux/tau_u/tau_v/
rhosfc/fltv) into ``State`` -- the SAME contract MYNN reads back via
``_surface_fluxes_from_state`` -- plus the blended t_skin / roughness_m / qsfc.

The land state EVOLVES prognostically: this hook returns ``(state', land_state')``;
the operational driver threads ``land_state'`` into the next step's carry. Ocean /
water columns keep the prescribed-SST sfclay bulk path byte-for-byte (the coupler's
``where(is_land,...)`` is the sole land/water switch).

LAYOUT BRIDGE. The operational ``State`` is leading-z ``(z, ny, nx)`` for column
fields and 2-D ``(ny, nx)`` for surface fields; the Noah-MP coupler + the revised
surface layer expect a TRAILING-z column view (the ``physics_couplers``
convention, where ``field[..., 0]`` is the lowest level). This hook builds that
trailing-z view (identical to ``physics_couplers._surface_column_view`` plus the
extra forcing fields Noah-MP reads) from the operational State, runs the coupler,
and maps the 2-D surface results back onto the State surface slots.

Radiation forcing (SOLDN/LWDN/COSZ) for Noah-MP is supplied by the operational
RRTMG diagnostics held in the carry; the land surface diagnostics (HFX/LH/TSK) are
read back from the prognostic Noah-MP fluxes by ``overlay_noahmp_land_diagnostics``
so the M9 map / gates see the prognostic land flux, not the bulk path.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing, noahmp_surface_adapter
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.physics.noahmp.types import NoahMPForcing
from gpuwrf.coupling.physics_couplers import (
    WRF_RV_OVER_RD,
    _column_dz_from_state,
    _mynn_column_uses_wrf_phy_prep,
    _surface_dz_from_state,
    _temperature_from_theta,
    _to_columns,
    _u_mass,
    _v_mass,
    _wrf_hydrostatic_pressure_from_state,
    _wrf_phy_prep_rho_from_state,
)


class _NoahMPColumnView(NamedTuple):
    """Trailing-z column view of the operational State for the Noah-MP coupler.

    Surface fields are 2-D ``(ny, nx)``; column fields are ``(ny, nx, nz)`` (last
    axis vertical) so the coupler's ``_surface(f)=f[...,0]`` reads the lowest
    level. Carries the union of fields the revised surface layer and the Noah-MP
    forcing assembler read.
    """

    # column fields (trailing-z)
    u: Any
    v: Any
    theta: Any
    qv: Any
    qc: Any
    p: Any
    dz: Any
    # surface fields (2-D)
    t_skin: Any
    soil_moisture: Any
    xland: Any
    lakemask: Any
    mavail: Any
    roughness_m: Any
    ustar: Any
    # WRF ``phy_prep`` surface-layer inputs (supplied when a grid with metrics is
    # threaded; ``None`` keeps the legacy fallback). t_air = lowest-level DRY air
    # temperature t_phy = theta_dry*(p/p0)^kappa; psfc = true surface pressure
    # (distinct from lowest-level air pressure); rho = phy_prep density (1+qv)/alt.
    # Mirrors coupling.physics_couplers._surface_column_view so the water/sfclay bulk
    # flux matches WRF instead of using the air-pressure / ideal-gas fallback.
    t_air: Any = None
    psfc: Any = None
    rho: Any = None

    def replace(self, **updates) -> "_NoahMPColumnView":
        d = self._asdict()
        d.update(updates)
        return _NoahMPColumnView(**d)


def _build_column_view(state: Any, grid: Any = None) -> _NoahMPColumnView:
    """Build the trailing-z Noah-MP column view from the operational State.

    WRF feeds the revised surface layer and noahmplsm the DRY sensible temperature
    t_phy = theta_dry*(p/p0)^kappa (module_sf_noahmpdrv.F:755), plus the ``phy_prep``
    hydrostatic pressure, true surface pressure, and density. The operational
    State.theta is the WRF MOIST potential temperature theta_m = theta_dry*(1 +
    R_v/R_d*qv) (use_theta_m=1, the dycore prognostic). Decouple theta_m -> theta_dry
    and -- when a grid with metrics is available -- supply the same phy_prep
    psfc/rho/hydrostatic-p the grid-backed ``physics_couplers._surface_column_view``
    uses, so the WATER/sfclay bulk flux (retained where Noah-MP does not run) matches
    WRF instead of using the +~4 K moist-theta air temperature and the air-pressure /
    ideal-gas fallback.
    """
    qv = _to_columns(state.qv)
    theta_dry = jnp.asarray(state.theta, jnp.float64) / (
        1.0 + WRF_RV_OVER_RD * jnp.asarray(state.qv, jnp.float64)
    )
    # t_air uses the nonhydrostatic state pressure (WRF t_phy is from p+pb), matching
    # the grid-backed view; the column ``p`` handed to the surface layer is hydrostatic.
    t_air = _to_columns(_temperature_from_theta(theta_dry, jnp.asarray(state.p, jnp.float64)))
    if _mynn_column_uses_wrf_phy_prep(grid):
        p_hyd, psfc = _wrf_hydrostatic_pressure_from_state(state, grid.metrics)
        rho = _wrf_phy_prep_rho_from_state(state, grid.metrics)
        theta = _to_columns(theta_dry)
        p = _to_columns(p_hyd)
        dz = _surface_dz_from_state(state)
        psfc_col = psfc
        rho_col = _to_columns(rho)
    else:
        # legacy fallback (no grid metrics): keep nonhydrostatic p and let the surface
        # layer derive rho; still hand it the dry t_air. psfc defaults to the
        # lowest-level air pressure -- the SAME value the prior no-psfc path used (the
        # surface layer's psfc fallback and assemble_noahmp_forcing's ``sfcprs``
        # default) -- so the grid-less proof/test callers are unchanged.
        theta = _to_columns(theta_dry)
        p = _to_columns(state.p)
        dz = _column_dz_from_state(state, None)
        psfc_col = jnp.asarray(p, jnp.float64)[..., 0]
        rho_col = None
    return _NoahMPColumnView(
        u=_to_columns(_u_mass(state)),
        v=_to_columns(_v_mass(state)),
        theta=theta,
        qv=qv,
        qc=_to_columns(state.qc),
        p=p,
        dz=dz,
        t_skin=state.t_skin,
        soil_moisture=state.soil_moisture,
        xland=state.xland,
        lakemask=state.lakemask,
        mavail=state.mavail,
        roughness_m=state.roughness_m,
        ustar=state.ustar,
        t_air=t_air,
        psfc=psfc_col,
        rho=rho_col,
    )


def _output_dtype(state: Any, field: str):
    """Write at the LIVE state field dtype (fp32-defeat fix; mirrors physics_couplers)."""
    return getattr(state, field).dtype


def _to_state_surface(state: Any, field: str, value):
    """Cast a 2-D blended field to the State surface slot's dtype/shape."""
    cur = getattr(state, field)
    v = jnp.asarray(value).astype(cur.dtype)
    return v.reshape(cur.shape) if v.shape != cur.shape and v.size == cur.size else v


def noahmp_surface_step(
    state: Any,
    land_state: NoahMPLandState,
    static: NoahMPStatic,
    dt: float,
    *,
    radiation: Any = None,
    clock: Any = None,
    energy_params: Any = None,
    rad_params: Any = None,
    first_timestep: Any = False,
    grid: Any = None,
) -> tuple[Any, NoahMPLandState]:
    """Run the Noah-MP/sfclay blend and write the blended flux handles into State.

    Returns ``(state', land_state')``. ``state'`` carries the blended kinematic
    surface-flux handles MYNN consumes, plus t_skin/roughness_m/qsfc. ``land_state'``
    is the prognostically advanced Noah-MP land carry for the next step.

    ``energy_params``/``rad_params`` are the PRE-BUILT (concrete-``nroot``) parameter
    bundles; passing them avoids re-running the frozen ``build_energy_params``
    (which concretizes ``nroot``) inside the jitted scan.
    """
    view = _build_column_view(state, grid)
    view_wb, land_out, blended = noahmp_surface_adapter(
        view, land_state, static, radiation=radiation, clock=clock, dt=float(dt),
        energy_params=energy_params, rad_params=rad_params,
        first_timestep=first_timestep,
    )
    # ``view_wb`` carries the blended 2-D t_skin/roughness_m/qsfc; the blended
    # kinematic flux handles come from ``blended``. Map both back onto the State.
    updates = {
        "ustar": _to_state_surface(state, "ustar", blended.ustar),
        "theta_flux": _to_state_surface(state, "theta_flux", blended.theta_flux),
        "qv_flux": _to_state_surface(state, "qv_flux", blended.qv_flux),
        "tau_u": _to_state_surface(state, "tau_u", blended.tau_u),
        "tau_v": _to_state_surface(state, "tau_v", blended.tau_v),
        "rhosfc": _to_state_surface(state, "rhosfc", blended.rhosfc),
        "fltv": _to_state_surface(state, "fltv", blended.fltv),
        "t_skin": _to_state_surface(state, "t_skin", view_wb.t_skin),
        "roughness_m": _to_state_surface(state, "roughness_m", view_wb.roughness_m),
    }
    state_out = state.replace(**updates)
    return state_out, land_out


def _surface_2d(field):
    a = jnp.asarray(field, dtype=jnp.float64)
    return a[..., 0] if a.ndim >= 3 else a


def overlay_noahmp_land_diagnostics(
    state: Any,
    land_state: NoahMPLandState,
    static: NoahMPStatic,
    bulk_hfx,
    bulk_lh,
    bulk_tsk,
    dt: float,
    *,
    bulk_t2=None,
    radiation: Any = None,
    clock: Any = None,
    energy_params: Any = None,
    rad_params: Any = None,
    grid: Any = None,
):
    """Overlay prognostic Noah-MP land HFX/LH/TSK (+ the LSM 2-m T2) onto the bulk.

    The M9 surface map (gates / TOST) is recomputed post-step from ``State`` via
    the bulk surface layer; over LAND it must instead report the prognostic
    Noah-MP fluxes (the standalone-replacement contract). This runs ONE Noah-MP
    column step on the CURRENT (post-step) land carry to read back HFX/LH/TSK and
    selects them where ``is_land``; ocean/water keeps the bulk diagnostic value.

    LAND T2 (v0.9.0): real WRF OVERWRITES the surface-layer (MYNN/sfclay) 2-m
    temperature with the Noah-MP LSM diagnostic ``T2 = FVEG*T2MV + (1-FVEG)*T2MB``
    over every land point (module_surface_driver.F:3469-3473). When ``bulk_t2`` is
    supplied this routes the faithful ``nm.t2`` over land (water keeps ``bulk_t2``)
    and returns a 4-tuple ``(hfx, lh, tsk, t2)``; with ``bulk_t2=None`` it returns
    the legacy ``(hfx, lh, tsk)`` (callers that have not yet wired the T2 overwrite).
    """
    view = _build_column_view(state, grid)
    forcing: NoahMPForcing = assemble_noahmp_forcing(view, static, radiation, clock, float(dt))
    _land_out, nm = noah_mp_step(
        land_state, forcing, static, float(dt),
        energy_params=energy_params, rad_params=rad_params,
    )

    xland = _surface_2d(getattr(state, "xland", jnp.ones_like(jnp.asarray(bulk_hfx, dtype=jnp.float64))))
    is_land = (xland - 1.5) < 0.0

    hfx = jnp.where(is_land, jnp.asarray(nm.hfx, dtype=jnp.float64), jnp.asarray(bulk_hfx, dtype=jnp.float64))
    lh = jnp.where(is_land, jnp.asarray(nm.lh, dtype=jnp.float64), jnp.asarray(bulk_lh, dtype=jnp.float64))
    tsk = jnp.where(is_land, jnp.asarray(nm.tsk, dtype=jnp.float64), jnp.asarray(bulk_tsk, dtype=jnp.float64))
    if bulk_t2 is None:
        return hfx, lh, tsk
    # LSM 2-m air temperature overwrite over land (the faithful resolution of the
    # operational land-T2 the MYNN-SL empirical stand-in was patching).
    t2 = jnp.where(is_land, jnp.asarray(nm.t2, dtype=jnp.float64), jnp.asarray(bulk_t2, dtype=jnp.float64))
    return hfx, lh, tsk, t2


__all__ = ["noahmp_surface_step", "overlay_noahmp_land_diagnostics"]
