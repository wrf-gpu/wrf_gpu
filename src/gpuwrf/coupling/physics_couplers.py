"""Type-checked adapters from the coupled State pytree to M5 column kernels.

The persistent state layout stays ADR-002 SoA. These wrappers only create
transient column views with vertical as the last axis because the M5 Thompson,
MYNN, and RRTMG kernels are column-batched in that convention.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
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
DEG_TO_RAD = 3.141592653589793 / 180.0
MINUTES_PER_DAY = 1440.0

# MODIFIED_IGBP_MODIS_NOAH LANDUSE.TBL summer values:
# columns ALBD (%), SFEM. Index 0 is a water fallback for legacy zero LU_INDEX
# analytic states; WRF category indices are used directly for entries 1..61.
_MODIS_NOAH_ALBEDO = jnp.asarray(
    (
        0.08,
        0.12,
        0.12,
        0.14,
        0.16,
        0.13,
        0.22,
        0.20,
        0.22,
        0.20,
        0.19,
        0.14,
        0.17,
        0.15,
        0.18,
        0.55,
        0.25,
        0.08,
        0.15,
        0.15,
        0.25,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
    ),
    dtype=jnp.float64,
)
_MODIS_NOAH_EMISSIVITY = jnp.asarray(
    (
        0.98,
        0.95,
        0.95,
        0.94,
        0.93,
        0.97,
        0.93,
        0.95,
        0.93,
        0.92,
        0.96,
        0.95,
        0.985,
        0.88,
        0.98,
        0.95,
        0.90,
        0.98,
        0.93,
        0.92,
        0.90,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.88,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
        0.97,
    ),
    dtype=jnp.float64,
)

# Existing production callers have no model-time argument yet. This fallback is
# deterministic and keeps legacy no-time call sites in the same SW magnitude
# range until a future interface sprint threads model time through the scan.
_LEGACY_RRTMG_TIME_UTC = datetime(2000, 5, 21, 17, 30, tzinfo=timezone.utc)


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


class RRTMGRadiationDiagnostics(NamedTuple):
    """Surface radiation diagnostics emitted by the RRTMG adapter inputs."""

    surface_albedo: object
    surface_emissivity: object
    coszen: object
    swdown: object
    swup: object
    glw: object
    glw_up: object


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


def _coerce_datetime_utc(time_utc) -> datetime:
    """Normalize accepted host-side time values to a timezone-aware UTC datetime."""

    if time_utc is None:
        return _LEGACY_RRTMG_TIME_UTC
    if isinstance(time_utc, datetime):
        value = time_utc
    elif isinstance(time_utc, date):
        value = datetime(time_utc.year, time_utc.month, time_utc.day)
    else:
        text = str(time_utc).strip().replace("Z", "+00:00")
        value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _time_utc_parts(time_utc) -> tuple[float, float]:
    """Return WRF-style Julian day and UTC minutes since midnight."""

    value = _coerce_datetime_utc(time_utc)
    julian = float(value.timetuple().tm_yday)
    minute = (
        float(value.hour) * 60.0
        + float(value.minute)
        + float(value.second) / 60.0
        + float(value.microsecond) / 60_000_000.0
    )
    return julian, minute


def _compute_coszen(lat, lon, time_utc):
    """Compute WRF-style cosine solar zenith from lat/lon and UTC time.

    This mirrors `module_radiation_driver.F` `radconst` + `calc_coszen`:
    23.5 degree obliquity and the same equation-of-time correction.
    """

    julian, utc_minute = _time_utc_parts(time_utc)
    lat_rad = jnp.asarray(lat, dtype=jnp.float64) * DEG_TO_RAD
    lon_deg = jnp.asarray(lon, dtype=jnp.float64)

    obecl = 23.5 * DEG_TO_RAD
    sxlong_day = jnp.where(julian >= 80.0, julian - 80.0, julian + 285.0)
    sxlong = sxlong_day * (360.0 / 365.0) * DEG_TO_RAD
    declin = jnp.arcsin(jnp.sin(obecl) * jnp.sin(sxlong))

    da = 2.0 * jnp.pi * (julian - 1.0) / 365.0
    eot = (
        0.000075
        + 0.001868 * jnp.cos(da)
        - 0.032077 * jnp.sin(da)
        - 0.014615 * jnp.cos(2.0 * da)
        - 0.04089 * jnp.sin(2.0 * da)
    ) * 229.18
    xt24 = jnp.mod(utc_minute, MINUTES_PER_DAY) + eot
    local_time_h = xt24 / 60.0 + lon_deg / 15.0
    hour_angle = 15.0 * (local_time_h - 12.0) * DEG_TO_RAD
    coszen = jnp.sin(lat_rad) * jnp.sin(declin) + jnp.cos(lat_rad) * jnp.cos(declin) * jnp.cos(hour_angle)
    return jnp.clip(coszen, -1.0, 1.0)


def _grid_lat_lon(surface_shape: tuple[int, int], grid: GridSpec | None, dtype):
    """Build a deterministic mass-grid lat/lon approximation from GridSpec metadata."""

    if grid is None:
        return (
            jnp.zeros(surface_shape, dtype=dtype),
            jnp.zeros(surface_shape, dtype=dtype),
        )
    projection = grid.projection
    ny, nx = surface_shape
    y = jnp.arange(ny, dtype=jnp.float64) - (float(ny) - 1.0) / 2.0
    x = jnp.arange(nx, dtype=jnp.float64) - (float(nx) - 1.0) / 2.0
    lat_step = float(projection.dy_m) / 111_320.0
    lon_scale = jnp.maximum(111_320.0 * jnp.cos(float(projection.lat_0) * DEG_TO_RAD), 1.0)
    lon_step = float(projection.dx_m) / lon_scale
    lat = float(projection.lat_0) + y[:, None] * lat_step
    lon = float(projection.lon_0) + x[None, :] * lon_step
    return (
        jnp.broadcast_to(lat, surface_shape).astype(dtype),
        jnp.broadcast_to(lon, surface_shape).astype(dtype),
    )


def _surface_radiation_properties(state: State):
    """Lookup MODIS land-use albedo/emissivity using `state.lu_index`."""

    lu = jnp.clip(jnp.asarray(state.lu_index, dtype=jnp.int32), 0, _MODIS_NOAH_ALBEDO.shape[0] - 1)
    albedo = jnp.take(_MODIS_NOAH_ALBEDO, lu).astype(state.t_skin.dtype)
    emissivity = jnp.take(_MODIS_NOAH_EMISSIVITY, lu).astype(state.t_skin.dtype)
    return albedo, emissivity


def _rrtmg_column_inputs(
    state: State,
    grid: GridSpec | None,
    *,
    time_utc=None,
) -> tuple[RRTMGSWColumnState, RRTMGLWColumnState, object, object, object]:
    """Build SW/LW RRTMG column states and expose shared surface fields."""

    T = _temperature_from_theta(state.theta, state.p)
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
    surface_albedo, surface_emissivity = _surface_radiation_properties(state)
    lat, lon = _grid_lat_lon(surface_shape, grid, state.t_skin.dtype)
    coszen = _compute_coszen(lat, lon, time_utc).astype(state.t_skin.dtype)

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
    return sw_state, lw_state, surface_albedo, surface_emissivity, coszen


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
    """Apply stored surface-layer fluxes as MYNN bottom boundary tendencies."""

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
    momentum_scale = float(dt) / dz0
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


def rrtmg_radiation_diagnostics(
    state: State,
    grid: GridSpec | None = None,
    *,
    time_utc=None,
) -> RRTMGRadiationDiagnostics:
    """Return surface RRTMG radiation diagnostics without changing State."""

    sw_state, lw_state, surface_albedo, surface_emissivity, coszen = _rrtmg_column_inputs(
        state,
        grid,
        time_utc=time_utc,
    )
    sw = solve_rrtmg_sw_column(sw_state, debug=False)
    lw = solve_rrtmg_lw_column(lw_state, debug=False)
    return RRTMGRadiationDiagnostics(
        surface_albedo=surface_albedo,
        surface_emissivity=surface_emissivity,
        coszen=coszen,
        swdown=sw.surface_down,
        swup=sw.surface_up,
        glw=lw.surface_down,
        glw_up=lw.surface_up,
    )


def rrtmg_adapter(state: State, dt: float, grid: GridSpec | None = None, *, time_utc=None) -> State:
    """Run SW and LW RRTMG column kernels and apply their temperature tendency."""

    T = _temperature_from_theta(state.theta, state.p)
    sw_state, lw_state, _, _, _ = _rrtmg_column_inputs(state, grid, time_utc=time_utc)
    sw = solve_rrtmg_sw_column(sw_state, debug=False)
    lw = solve_rrtmg_lw_column(lw_state, debug=False)
    T_next = T + float(dt) * _from_columns(sw.heating_rate + lw.heating_rate)
    return state.replace(theta=_theta_from_temperature(T_next, state.p, _field_dtype("theta")))


__all__ = [
    "RRTMGRadiationDiagnostics",
    "ThompsonTendencySideChannel",
    "_compute_coszen",
    "mynn_adapter",
    "rrtmg_radiation_diagnostics",
    "rrtmg_adapter",
    "surface_adapter",
    "thompson_adapter",
    "thompson_adapter_with_tendencies",
]
