"""Type-checked adapters from the coupled State pytree to M5 column kernels.

The persistent state layout stays ADR-002 SoA. These wrappers only create
transient column views with vertical as the last axis because the M5 Thompson,
MYNN, and RRTMG kernels are column-batched in that convention.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, step_mynn_pbl_column
from gpuwrf.physics.surface_layer import surface_layer
from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    density_from_pressure_temperature,
    step_thompson_column,
)


P0_PA = 100000.0
R_D_OVER_CP = 287.0 / 1004.0
GRAVITY_M_S2 = 9.80665


class _SurfaceColumnState(NamedTuple):
    """Column-oriented view consumed by `surface_layer.surface_layer`."""

    u: object
    v: object
    theta: object
    qv: object
    p: object
    dz: object
    t_skin: object
    soil_moisture: object
    xland: object
    lakemask: object
    mavail: object
    roughness_m: object
    ustar: object


class ThompsonTendencySideChannel(NamedTuple):
    """Microphysical species-tendency oracle emitted by the Thompson adapter.

    Arrays are in State orientation `(z, y, x)` except `column_water_tendency`,
    which is the vertical-mean total-water tendency on `(y, x)`.
    """

    qv: object
    qc: object
    qr: object
    qi: object
    qs: object
    qg: object
    column_water_tendency: object
    precip_out_tendency: object


def _to_columns(field):
    """Moves state vertical axis from leading `(z, y, x)` to trailing `(..., z)`."""

    return jnp.moveaxis(field, 0, -1)


def _from_columns(field):
    """Moves column-kernel vertical axis from trailing `(..., z)` to leading z."""

    return jnp.moveaxis(field, -1, 0)


def _u_mass(state: State):
    """Collocates x-face wind to mass points."""

    return 0.5 * (state.u[:, :, :-1] + state.u[:, :, 1:])


def _v_mass(state: State):
    """Collocates y-face wind to mass points."""

    return 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])


def _w_mass(state: State):
    """Collocates vertical-face wind to mass points."""

    return 0.5 * (state.w[:-1, :, :] + state.w[1:, :, :])


def _mass_to_u_face(field):
    """Maps a mass-point wind update back to periodic x faces."""

    face = 0.5 * (jnp.roll(field, 1, axis=2) + field)
    return jnp.concatenate((face, face[:, :, :1]), axis=2)


def _mass_to_v_face(field):
    """Maps a mass-point wind update back to periodic y faces."""

    face = 0.5 * (jnp.roll(field, 1, axis=1) + field)
    return jnp.concatenate((face, face[:, :1, :]), axis=1)


def _mass_to_w_face(field):
    """Maps mass-point vertical velocity back to rigid vertical faces."""

    interior = 0.5 * (field[:-1, :, :] + field[1:, :, :])
    return jnp.concatenate((field[:1, :, :], interior, field[-1:, :, :]), axis=0)


def _temperature_from_theta(theta, p):
    """Converts potential temperature to temperature for column physics."""

    exner = (jnp.maximum(p, 1.0) / P0_PA) ** R_D_OVER_CP
    return theta.astype(p.dtype) * exner


def _theta_from_temperature(T, p, dtype):
    """Converts temperature back to potential temperature at the coupling boundary."""

    exner = (jnp.maximum(p, 1.0) / P0_PA) ** R_D_OVER_CP
    return (T / jnp.maximum(exner, 1.0e-12)).astype(dtype)


def _field_dtype(field: str):
    """Returns the frozen M6 storage dtype for a State field."""

    return DEFAULT_DTYPES.dtype_for(field)


def _bottom_column(surface_field, template_columns):
    """Place a 2-D surface field into the lowest slot of a column array."""

    value = jnp.asarray(surface_field).astype(template_columns.dtype)
    return jnp.zeros_like(template_columns).at[..., 0].set(value)


def _surface_flux_column_inputs(state: State, theta_columns):
    """Return MYNN bottom-boundary columns sourced from the surface adapter."""

    return (
        _bottom_column(state.theta_flux, theta_columns),
        _bottom_column(state.qv_flux, theta_columns),
        _bottom_column(state.tau_u, theta_columns),
        _bottom_column(state.tau_v, theta_columns),
    )


def _rho_from_state(state: State):
    """Builds the density diagnostic required by column physics kernels."""

    T = _temperature_from_theta(state.theta, state.p)
    return density_from_pressure_temperature(state.p, T, state.qv)


def _column_dz_from_state(state: State, grid: GridSpec | None):
    """Returns terrain-following layer thickness from geopotential interfaces."""

    del grid
    interface_height_m = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_height_m[1:, :, :] - interface_height_m[:-1, :, :], 1.0)
    return _to_columns(dz)


def _cloud_fraction_columns(state: State):
    """Builds a bounded diagnostic cloud fraction from hydrometeor occupancy."""

    condensate = state.qc + state.qi + state.qs + state.qg
    return _to_columns(jnp.clip(condensate * 1.0e5, 0.0, 1.0))


def _thompson_column_from_state(state: State) -> ThompsonColumnState:
    """Build the column-kernel input view for Thompson microphysics."""

    T = _temperature_from_theta(state.theta, state.p)
    rho = density_from_pressure_temperature(state.p, T, state.qv)
    return ThompsonColumnState(
        _to_columns(state.qv),
        _to_columns(state.qc),
        _to_columns(state.qr),
        _to_columns(state.qi),
        _to_columns(state.qs),
        _to_columns(state.qg),
        _to_columns(state.Ni),
        _to_columns(state.Nr),
        _to_columns(T),
        _to_columns(state.p),
        _to_columns(rho),
    )


def _state_from_thompson_output(state: State, out: ThompsonColumnState) -> State:
    """Reassemble a State from Thompson column-kernel output."""

    theta = _theta_from_temperature(_from_columns(out.T), state.p, _field_dtype("theta"))
    return state.replace(
        theta=theta,
        qv=_from_columns(out.qv).astype(_field_dtype("qv")),
        qc=_from_columns(out.qc).astype(_field_dtype("qc")),
        qr=_from_columns(out.qr).astype(_field_dtype("qr")),
        qi=_from_columns(out.qi).astype(_field_dtype("qi")),
        qs=_from_columns(out.qs).astype(_field_dtype("qs")),
        qg=_from_columns(out.qg).astype(_field_dtype("qg")),
        Ni=_from_columns(out.Ni).astype(_field_dtype("Ni")),
        Nr=_from_columns(out.Nr).astype(_field_dtype("Nr")),
    )


def _thompson_tendency_side_channel(
    state: State,
    out: ThompsonColumnState,
    dt: float,
) -> ThompsonTendencySideChannel:
    """Return water-species tendencies independent of precipitation accumulators."""

    inv_dt = 1.0 / float(dt)
    tendencies = {
        field: (
            jnp.asarray(_from_columns(getattr(out, field)), dtype=jnp.float64)
            - jnp.asarray(getattr(state, field), dtype=jnp.float64)
        )
        * inv_dt
        for field in ("qv", "qc", "qr", "qi", "qs", "qg")
    }
    column_water_tendency = jnp.mean(sum(tendencies.values()), axis=0)
    precip_out_tendency = jnp.zeros_like(column_water_tendency)
    return ThompsonTendencySideChannel(
        qv=tendencies["qv"],
        qc=tendencies["qc"],
        qr=tendencies["qr"],
        qi=tendencies["qi"],
        qs=tendencies["qs"],
        qg=tendencies["qg"],
        column_water_tendency=column_water_tendency,
        precip_out_tendency=precip_out_tendency,
    )


def thompson_adapter(state: State, dt: float, *, return_tendencies: bool = False):
    """Slice state to Thompson inputs, call the kernel, and reassemble State.

    `return_tendencies=True` exposes the M6-S6 water-budget oracle while
    existing coupled-driver calls keep the original State-only API.
    """

    column = _thompson_column_from_state(state)
    out = step_thompson_column(column, dt, debug=False)
    next_state = _state_from_thompson_output(state, out)
    if return_tendencies:
        return next_state, _thompson_tendency_side_channel(state, out, dt)
    return next_state


def thompson_adapter_with_tendencies(state: State, dt: float) -> tuple[State, ThompsonTendencySideChannel]:
    """Explicit Thompson side-channel wrapper for validation call sites."""

    return thompson_adapter(state, dt, return_tendencies=True)


def mynn_adapter(state: State, dt: float, grid: GridSpec | None = None) -> State:
    """Slice state to MYNN PBL-column inputs, call the kernel, and reassemble State."""

    theta_columns = _to_columns(state.theta)
    qke_columns = _to_columns(state.qke)
    rho_columns = _to_columns(_rho_from_state(state))
    dz_columns = _column_dz_from_state(state, grid)
    theta_flux_columns, qv_flux_columns, tau_u_columns, tau_v_columns = _surface_flux_column_inputs(state, theta_columns)
    momentum_flux_columns = jnp.sqrt(tau_u_columns * tau_u_columns + tau_v_columns * tau_v_columns)
    column = MynnPBLColumnState(
        _to_columns(_u_mass(state)),
        _to_columns(_v_mass(state)),
        _to_columns(_w_mass(state)),
        theta_columns,
        _to_columns(state.qv),
        0.5 * qke_columns,
        _to_columns(state.p),
        rho_columns,
        dz_columns,
        theta_flux_columns,
        qv_flux_columns,
        momentum_flux_columns,
    )
    out = step_mynn_pbl_column(column, dt, debug=False)
    out = _apply_surface_flux_bottom_bc(out, state, dt, dz_columns, rho_columns)
    u_mass = _from_columns(out.u)
    v_mass = _from_columns(out.v)
    w_mass = _from_columns(out.w)
    return state.replace(
        u=_mass_to_u_face(u_mass).astype(_field_dtype("u")),
        v=_mass_to_v_face(v_mass).astype(_field_dtype("v")),
        w=_mass_to_w_face(w_mass).astype(_field_dtype("w")),
        theta=_from_columns(out.theta).astype(_field_dtype("theta")),
        qv=_from_columns(out.qv).astype(_field_dtype("qv")),
        qke=(2.0 * _from_columns(out.tke)).astype(_field_dtype("qke")),
    )


def _apply_surface_flux_bottom_bc(
    out: MynnPBLColumnState,
    state: State,
    dt: float,
    dz_columns,
    rho_columns,
) -> MynnPBLColumnState:
    """Apply WRF-sign surface-layer fluxes as MYNN bottom boundary tendencies.

    `theta_flux`, `qv_flux`, `tau_u`, and `tau_v` are kinematic fluxes that are
    positive upward into the atmosphere. WRF MYNN adds scalar fluxes to the
    bottom RHS with `+dt/dz*rhosfc/rho`; signed momentum flux components use the
    same positive-upward convention, so drag over fixed ground is normally
    opposite-signed to the lowest-level wind.
    """

    dz0 = jnp.maximum(dz_columns[..., 0], 1.0)
    rho0 = jnp.maximum(rho_columns[..., 0], 1.0e-4)
    rhosfc = jnp.maximum(jnp.asarray(state.rhosfc, dtype=rho0.dtype), 1.0e-4)
    scalar_scale = float(dt) / dz0 * rhosfc / rho0
    theta_increment = (scalar_scale * jnp.asarray(state.theta_flux, dtype=scalar_scale.dtype)).astype(out.theta.dtype)
    theta = out.theta.at[..., 0].add(theta_increment)
    qv_flux_floor = jnp.minimum(0.9 * out.qv[..., 0] - 1.0e-8, 0.0) / jnp.maximum(float(dt) / dz0, 1.0e-12)
    qv_flux = jnp.maximum(jnp.asarray(state.qv_flux, dtype=out.qv.dtype), qv_flux_floor)
    qv_increment = (scalar_scale * qv_flux.astype(scalar_scale.dtype)).astype(out.qv.dtype)
    qv = out.qv.at[..., 0].add(qv_increment)
    momentum_scale = scalar_scale
    u_increment = (momentum_scale * jnp.asarray(state.tau_u, dtype=momentum_scale.dtype)).astype(out.u.dtype)
    v_increment = (momentum_scale * jnp.asarray(state.tau_v, dtype=momentum_scale.dtype)).astype(out.v.dtype)
    u = out.u.at[..., 0].add(u_increment)
    v = out.v.at[..., 0].add(v_increment)
    return out.replace(u=u, v=v, theta=theta, qv=jnp.maximum(qv, 0.0))


def surface_adapter(state: State, dt: float) -> State:
    """Wrap `surface_layer(state) -> SurfaceFluxes` and store its surface handles."""

    del dt
    column_state = _SurfaceColumnState(
        u=_to_columns(_u_mass(state)),
        v=_to_columns(_v_mass(state)),
        theta=_to_columns(state.theta),
        qv=_to_columns(state.qv),
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
    flux = surface_layer(column_state)
    return state.replace(
        ustar=flux.ustar.astype(_field_dtype("ustar")),
        theta_flux=flux.theta_flux.astype(_field_dtype("theta_flux")),
        qv_flux=flux.qv_flux.astype(_field_dtype("qv_flux")),
        tau_u=flux.tau_u.astype(_field_dtype("tau_u")),
        tau_v=flux.tau_v.astype(_field_dtype("tau_v")),
        rhosfc=flux.rhosfc.astype(_field_dtype("rhosfc")),
        fltv=flux.fltv.astype(_field_dtype("fltv")),
    )


def rrtmg_adapter(state: State, dt: float, grid: GridSpec | None = None) -> State:
    """Run SW and LW RRTMG column kernels and apply their temperature tendency."""

    T = _temperature_from_theta(state.theta, state.p)
    theta_columns = _to_columns(state.theta)
    p_columns = _to_columns(state.p)
    qv_columns = _to_columns(state.qv)
    qc_columns = _to_columns(state.qc)
    qi_columns = _to_columns(state.qi)
    qs_columns = _to_columns(state.qs)
    qg_columns = _to_columns(state.qg)
    cloud_fraction = _cloud_fraction_columns(state)
    dz = _column_dz_from_state(state, grid)
    rho = _to_columns(_rho_from_state(state))
    surface_shape = state.t_skin.shape
    surface_albedo = jnp.ones(surface_shape, dtype=state.t_skin.dtype) * 0.15
    surface_emissivity = jnp.ones(surface_shape, dtype=state.t_skin.dtype) * 0.98
    coszen = jnp.ones(surface_shape, dtype=state.t_skin.dtype) * 0.50

    sw_state = RRTMGSWColumnState(
        _to_columns(T),
        p_columns,
        qv_columns,
        qc_columns,
        qi_columns,
        qs_columns,
        qg_columns,
        cloud_fraction,
        surface_albedo,
        coszen,
        dz,
        rho,
    )
    lw_state = RRTMGLWColumnState(
        _to_columns(T),
        p_columns,
        qv_columns,
        qc_columns,
        qi_columns,
        qs_columns,
        qg_columns,
        cloud_fraction,
        state.t_skin,
        surface_emissivity,
        dz,
        rho,
    )
    sw = solve_rrtmg_sw_column(sw_state, debug=False)
    lw = solve_rrtmg_lw_column(lw_state, debug=False)
    T_next = T + float(dt) * _from_columns(sw.heating_rate + lw.heating_rate)
    return state.replace(theta=_theta_from_temperature(T_next, state.p, _field_dtype("theta")))


__all__ = [
    "ThompsonTendencySideChannel",
    "mynn_adapter",
    "rrtmg_adapter",
    "surface_adapter",
    "thompson_adapter",
    "thompson_adapter_with_tendencies",
]
