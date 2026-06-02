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
    _column_dz_from_state,
    _to_columns,
    _u_mass,
    _v_mass,
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

    def replace(self, **updates) -> "_NoahMPColumnView":
        d = self._asdict()
        d.update(updates)
        return _NoahMPColumnView(**d)


def _build_column_view(state: Any) -> _NoahMPColumnView:
    """Build the trailing-z Noah-MP column view from the operational State."""
    return _NoahMPColumnView(
        u=_to_columns(_u_mass(state)),
        v=_to_columns(_v_mass(state)),
        theta=_to_columns(state.theta),
        qv=_to_columns(state.qv),
        qc=_to_columns(state.qc),
        p=_to_columns(state.p),
        dz=_column_dz_from_state(state, None),
        t_skin=state.t_skin,
        soil_moisture=state.soil_moisture,
        xland=state.xland,
        lakemask=state.lakemask,
        mavail=state.mavail,
        roughness_m=state.roughness_m,
        ustar=state.ustar,
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
) -> tuple[Any, NoahMPLandState]:
    """Run the Noah-MP/sfclay blend and write the blended flux handles into State.

    Returns ``(state', land_state')``. ``state'`` carries the blended kinematic
    surface-flux handles MYNN consumes, plus t_skin/roughness_m/qsfc. ``land_state'``
    is the prognostically advanced Noah-MP land carry for the next step.

    ``energy_params``/``rad_params`` are the PRE-BUILT (concrete-``nroot``) parameter
    bundles; passing them avoids re-running the frozen ``build_energy_params``
    (which concretizes ``nroot``) inside the jitted scan.
    """
    view = _build_column_view(state)
    view_wb, land_out, blended = noahmp_surface_adapter(
        view, land_state, static, radiation=radiation, clock=clock, dt=float(dt),
        energy_params=energy_params, rad_params=rad_params,
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
    radiation: Any = None,
    clock: Any = None,
    energy_params: Any = None,
    rad_params: Any = None,
):
    """Overlay prognostic Noah-MP land HFX/LH/TSK onto the bulk diagnostics.

    The M9 surface map (gates / TOST) is recomputed post-step from ``State`` via
    the bulk surface layer; over LAND it must instead report the prognostic
    Noah-MP fluxes (the standalone-replacement contract). This runs ONE Noah-MP
    column step on the CURRENT (post-step) land carry to read back HFX/LH/TSK and
    selects them where ``is_land``; ocean/water keeps the bulk diagnostic value.

    Returns ``(hfx, lh, tsk)`` 2-D (ny, nx). T2/Q2/U10/V10 are left to the bulk
    surface-layer diagnostic, which already uses the Noah-MP skin temperature
    (written into ``state.t_skin`` by ``noahmp_surface_step``) as its surface BC.
    """
    view = _build_column_view(state)
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
    return hfx, lh, tsk


__all__ = ["noahmp_surface_step", "overlay_noahmp_land_diagnostics"]
