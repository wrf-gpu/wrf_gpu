"""Device-side lateral-boundary forcing for the M6 d02 forecast driver."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from gpuwrf.contracts.state import State


SIDES = ("W", "E", "S", "N")
SIDE_INDEX = {name: index for index, name in enumerate(SIDES)}


@dataclass(frozen=True)
class BoundaryConfig:
    """WRF lateral-boundary control values for the pinned Gen2 d02 run."""

    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    update_cadence_s: float = 3600.0
    spec_exp: float = 0.0


DEFAULT_BOUNDARY_CONFIG = BoundaryConfig()


def apply_lateral_boundaries(
    state: State,
    lead_seconds,
    dt_s: float,
    config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
) -> State:
    """Apply specified outer boundaries plus WRF-style relaxation-zone nudging."""

    u = _apply_3d(state.u, state.u_bdy, lead_seconds, dt_s, config)
    v = _apply_3d(state.v, state.v_bdy, lead_seconds, dt_s, config)
    theta = _apply_3d(state.theta, state.theta_bdy, lead_seconds, dt_s, config)
    qv = jnp.maximum(_apply_3d(state.qv, state.qv_bdy, lead_seconds, dt_s, config), 0.0)
    ph = _apply_3d(state.ph, state.ph_bdy, lead_seconds, dt_s, config)
    mu = _apply_3d(state.mu[None, :, :], state.mu_bdy, lead_seconds, dt_s, config)[0]
    return state.replace(u=u, v=v, theta=theta, qv=qv, ph=ph, mu=mu)


def interpolate_boundary_leaf(boundary, lead_seconds, cadence_s: float = 3600.0):
    """Linearly interpolate one `(time, side, z, side_index)` boundary leaf."""

    max_index = int(boundary.shape[0]) - 1
    lead_index = jnp.asarray(lead_seconds, dtype=jnp.float64) / float(cadence_s)
    lower = jnp.clip(jnp.floor(lead_index).astype(jnp.int32), 0, max_index)
    upper = jnp.clip(lower + 1, 0, max_index)
    alpha = jnp.clip(lead_index - lower.astype(jnp.float64), 0.0, 1.0)
    lower_values = jnp.take(boundary, lower, axis=0)
    upper_values = jnp.take(boundary, upper, axis=0)
    return (lower_values * (1.0 - alpha) + upper_values * alpha).astype(boundary.dtype)


def _apply_3d(field, boundary, lead_seconds, dt_s: float, config: BoundaryConfig):
    forcing = interpolate_boundary_leaf(boundary, lead_seconds, config.update_cadence_s)
    out = field
    out = _apply_specified(out, forcing, "W", 0)
    out = _apply_specified(out, forcing, "E", 0)
    out = _apply_specified(out, forcing, "S", 0)
    out = _apply_specified(out, forcing, "N", 0)
    for offset in range(int(config.spec_zone), int(config.relax_zone)):
        out = _apply_relax(out, forcing, "W", offset, dt_s, config)
        out = _apply_relax(out, forcing, "E", offset, dt_s, config)
        out = _apply_relax(out, forcing, "S", offset, dt_s, config)
        out = _apply_relax(out, forcing, "N", offset, dt_s, config)
    out = _apply_specified(out, forcing, "W", 0)
    out = _apply_specified(out, forcing, "E", 0)
    out = _apply_specified(out, forcing, "S", 0)
    out = _apply_specified(out, forcing, "N", 0)
    return out


def _side_values(forcing, side: str, z_len: int, side_len: int):
    return forcing[SIDE_INDEX[side], :z_len, :side_len]


def _apply_specified(field, forcing, side: str, offset: int):
    z_len, y_len, x_len = field.shape
    if side == "W":
        target = _side_values(forcing, side, z_len, y_len)
        return field.at[:, :, offset].set(target)
    if side == "E":
        target = _side_values(forcing, side, z_len, y_len)
        return field.at[:, :, x_len - 1 - offset].set(target)
    if side == "S":
        target = _side_values(forcing, side, z_len, x_len)
        return field.at[:, offset, :].set(target)
    target = _side_values(forcing, side, z_len, x_len)
    return field.at[:, y_len - 1 - offset, :].set(target)


def _apply_relax(field, forcing, side: str, offset: int, dt_s: float, config: BoundaryConfig):
    z_len, y_len, x_len = field.shape
    weight_f, weight_g = _wrf_relax_weights(offset, dt_s, config)
    if side == "W":
        target = _side_values(forcing, side, z_len, y_len)
        current = field[:, :, offset]
        relaxed = _relaxed_slice(current, target, field[:, :, offset - 1], field[:, :, offset + 1], axis=1, weight_f=weight_f, weight_g=weight_g)
        start, end = offset + 1, y_len - offset - 1
        return field.at[:, start:end, offset].set(relaxed[:, start:end])
    if side == "E":
        target = _side_values(forcing, side, z_len, y_len)
        x = x_len - 1 - offset
        current = field[:, :, x]
        relaxed = _relaxed_slice(current, target, field[:, :, x + 1], field[:, :, x - 1], axis=1, weight_f=weight_f, weight_g=weight_g)
        start, end = offset + 1, y_len - offset - 1
        return field.at[:, start:end, x].set(relaxed[:, start:end])
    if side == "S":
        target = _side_values(forcing, side, z_len, x_len)
        current = field[:, offset, :]
        relaxed = _relaxed_slice(current, target, field[:, offset - 1, :], field[:, offset + 1, :], axis=1, weight_f=weight_f, weight_g=weight_g)
        start, end = offset + 1, x_len - offset - 1
        return field.at[:, offset, start:end].set(relaxed[:, start:end])
    target = _side_values(forcing, side, z_len, x_len)
    y = y_len - 1 - offset
    current = field[:, y, :]
    relaxed = _relaxed_slice(current, target, field[:, y + 1, :], field[:, y - 1, :], axis=1, weight_f=weight_f, weight_g=weight_g)
    start, end = offset + 1, x_len - offset - 1
    return field.at[:, y, start:end].set(relaxed[:, start:end])


def _relaxed_slice(current, target, normal_outer, normal_inner, *, axis: int, weight_f, weight_g):
    residual = target - current
    tangent_minus = _shift_edge(residual, 1, axis)
    tangent_plus = _shift_edge(residual, -1, axis)
    outer_residual = target - normal_outer
    inner_residual = target - normal_inner
    lap_residual = tangent_minus + tangent_plus + outer_residual + inner_residual - 4.0 * residual
    return current + weight_f * residual - weight_g * lap_residual


def _shift_edge(values, shift: int, axis: int):
    if shift == 1:
        pad = jnp.take(values, jnp.array([0]), axis=axis)
        core = jnp.take(values, jnp.arange(values.shape[axis] - 1), axis=axis)
        return jnp.concatenate((pad, core), axis=axis)
    pad = jnp.take(values, jnp.array([values.shape[axis] - 1]), axis=axis)
    core = jnp.take(values, jnp.arange(1, values.shape[axis]), axis=axis)
    return jnp.concatenate((core, pad), axis=axis)


def _wrf_relax_weights(offset: int, dt_s: float, config: BoundaryConfig) -> tuple[float, float]:
    """Return `dt * fcx` and `dt * gcx` from WRF `lbc_fcx_gcx` for one offset."""

    loop_1based = int(offset) + 1
    numerator = float(config.spec_zone + config.relax_zone - loop_1based)
    denominator = float(config.relax_zone - 1)
    linear = max(0.0, numerator / denominator)
    sponge = math.exp(-(loop_1based - (config.spec_zone + 1)) * config.spec_exp)
    fcx = 0.1 / float(dt_s) * linear * sponge
    gcx = 1.0 / float(dt_s) / 50.0 * linear * sponge
    return float(dt_s) * fcx, float(dt_s) * gcx


__all__ = [
    "BoundaryConfig",
    "DEFAULT_BOUNDARY_CONFIG",
    "SIDES",
    "apply_lateral_boundaries",
    "interpolate_boundary_leaf",
]
