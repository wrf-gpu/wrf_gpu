"""JAX longwave RRTMG-style radiation column kernel for M5-S3."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.physics.rrtmg_constants import (
    CP_AIR,
    MAX_OPTICAL_DEPTH,
    MIN_LAYER_MASS,
    MIN_OPTICAL_DEPTH,
    STEFAN_BOLTZMANN,
)
from gpuwrf.physics.rrtmg_tables import RRTMGTableBundle, RRTMG_TABLES


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class RRTMGLWColumnState:
    """Pytree for independent longwave radiation columns on mass levels."""

    __slots__ = ("T", "p", "qv", "qc", "qi", "qs", "qg", "cloud_fraction", "surface_temperature", "surface_emissivity", "dz", "rho")

    def __init__(self, T, p, qv, qc, qi, qs, qg, cloud_fraction, surface_temperature, surface_emissivity, dz, rho) -> None:
        self.T = T
        self.p = p
        self.qv = qv
        self.qc = qc
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.cloud_fraction = cloud_fraction
        self.surface_temperature = surface_temperature
        self.surface_emissivity = surface_emissivity
        self.dz = dz
        self.rho = rho

    def replace(self, **updates) -> "RRTMGLWColumnState":
        """Returns a same-layout state with named fields replaced."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        """Presents all state arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds the state after JAX transforms."""

        del aux
        return cls(*children)

    def __eq__(self, other: object) -> bool:
        """Implements array-aware equality outside JIT for tests."""

        if not isinstance(other, RRTMGLWColumnState):
            return NotImplemented
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(_leaves(self), _leaves(other), strict=True)
        )

    def __hash__(self) -> int:
        """Hashes small column states outside the physics hot path."""

        parts = []
        for leaf in _leaves(self):
            host = np.asarray(leaf)
            parts.append((tuple(host.shape), str(host.dtype), host.tobytes()))
        return hash(tuple(parts))


class RRTMGLWColumnResult(NamedTuple):
    """Longwave column outputs with bottom-to-top interface fluxes."""

    heating_rate: jnp.ndarray
    flux_down: jnp.ndarray
    flux_up: jnp.ndarray
    toa_down: jnp.ndarray
    toa_up: jnp.ndarray
    surface_down: jnp.ndarray
    surface_up: jnp.ndarray
    column_net_heating: jnp.ndarray
    surface_emission: jnp.ndarray


def _leaves(state: RRTMGLWColumnState):
    """Centralizes leaf iteration for equality and hashing."""

    return (getattr(state, name) for name in RRTMGLWColumnState.__slots__)


def _clip_state(state: RRTMGLWColumnState) -> RRTMGLWColumnState:
    """Applies radiation-safe physical bounds before optical calculations."""

    return state.replace(
        T=jnp.maximum(state.T, 120.0),
        p=jnp.maximum(state.p, 1.0),
        qv=jnp.maximum(state.qv, 0.0),
        qc=jnp.maximum(state.qc, 0.0),
        qi=jnp.maximum(state.qi, 0.0),
        qs=jnp.maximum(state.qs, 0.0),
        qg=jnp.maximum(state.qg, 0.0),
        cloud_fraction=jnp.clip(state.cloud_fraction, 0.0, 1.0),
        surface_temperature=jnp.maximum(state.surface_temperature, 120.0),
        surface_emissivity=jnp.clip(state.surface_emissivity, 0.0, 1.0),
        dz=jnp.maximum(state.dz, 1.0),
        rho=jnp.maximum(state.rho, MIN_LAYER_MASS),
    )


def _longwave_impl(state: RRTMGLWColumnState, tables: RRTMGTableBundle, debug: bool) -> RRTMGLWColumnResult:
    """Unjitted LW implementation shared by production and stripped paths."""

    state = _clip_state(state)
    layer_mass = jnp.maximum(state.rho * state.dz, MIN_LAYER_MASS)
    vapor_path = state.qv * layer_mass
    cloud_path = (state.qc + state.qi + state.qs + state.qg) * layer_mass * state.cloud_fraction

    band_index = jnp.arange(tables.lw_band_weights.shape[0])
    weights = jnp.take(tables.lw_band_weights, band_index, axis=0)
    gas_coeff = jnp.take(tables.lw_absorption_coefficients, band_index, axis=0)
    cloud_coeff = jnp.take(tables.lw_cloud_absorption, band_index, axis=0)

    pressure_scale = jnp.sqrt(jnp.maximum(state.p, 1.0) / 100000.0)
    tau = jnp.clip(vapor_path[..., None] * gas_coeff * pressure_scale[..., None] + cloud_path[..., None] * cloud_coeff, MIN_OPTICAL_DEPTH, MAX_OPTICAL_DEPTH)
    trans = jnp.exp(-tau)
    layer_source = STEFAN_BOLTZMANN * state.T[..., None] ** 4 * weights
    layer_emit = layer_source * (1.0 - trans)

    surface_emission_band = STEFAN_BOLTZMANN * state.surface_temperature[..., None] ** 4 * state.surface_emissivity[..., None] * weights
    cumulative_bottom = jnp.concatenate((jnp.ones_like(trans[..., :1, :]), jnp.cumprod(trans, axis=-2)), axis=-2)
    attenuated_surface = surface_emission_band[..., None, :] * cumulative_bottom
    lower_trans_to_ifc = cumulative_bottom[..., 1:, :]
    safe_lower = jnp.maximum(lower_trans_to_ifc, MIN_OPTICAL_DEPTH)
    emission_up = jnp.cumsum(layer_emit / safe_lower, axis=-2) * lower_trans_to_ifc
    up_band = attenuated_surface.at[..., 1:, :].add(emission_up)

    trans_top = jnp.flip(trans, axis=-2)
    emit_top = jnp.flip(layer_emit, axis=-2)
    cumulative_top = jnp.concatenate((jnp.ones_like(trans_top[..., :1, :]), jnp.cumprod(trans_top, axis=-2)), axis=-2)
    upper_trans_to_ifc = cumulative_top[..., 1:, :]
    safe_upper = jnp.maximum(upper_trans_to_ifc, MIN_OPTICAL_DEPTH)
    down_top = jnp.concatenate(
        (
            jnp.zeros_like(cumulative_top[..., :1, :]),
            jnp.cumsum(emit_top / safe_upper, axis=-2) * upper_trans_to_ifc,
        ),
        axis=-2,
    )
    down_band = jnp.flip(down_top, axis=-2)

    flux_up_model = jnp.sum(up_band, axis=-1)
    flux_down_model = jnp.sum(down_band, axis=-1)
    net_down = flux_down_model - flux_up_model
    layer_net_heating = net_down[..., 1:] - net_down[..., :-1]
    heating_rate = layer_net_heating / (layer_mass * CP_AIR)
    surface_emission = STEFAN_BOLTZMANN * state.surface_emissivity * state.surface_temperature**4
    flux_down = jnp.concatenate((flux_down_model, jnp.zeros_like(flux_down_model[..., -1:])), axis=-1)
    flux_up = jnp.concatenate((flux_up_model, flux_up_model[..., -1:]), axis=-1)

    heating_rate = assert_finite(heating_rate, "rrtmg_lw.heating_rate", enabled=debug)
    flux_down = assert_physical_bounds(flux_down, 0.0, 2000.0, "rrtmg_lw.flux_down", enabled=debug)
    flux_up = assert_physical_bounds(flux_up, 0.0, 2000.0, "rrtmg_lw.flux_up", enabled=debug)
    return RRTMGLWColumnResult(
        heating_rate=heating_rate,
        flux_down=flux_down,
        flux_up=flux_up,
        toa_down=flux_down[..., -1],
        toa_up=flux_up[..., -1],
        surface_down=flux_down[..., 0],
        surface_up=flux_up[..., 0],
        column_net_heating=jnp.sum(layer_net_heating, axis=-1),
        surface_emission=surface_emission,
    )


@partial(jax.jit, static_argnames=("debug",))
def solve_rrtmg_lw_column(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
    *,
    debug: bool = False,
) -> RRTMGLWColumnResult:
    """Computes one fused longwave column radiation call."""

    return _longwave_impl(state, tables, debug)


@jax.jit
def solve_rrtmg_lw_column_debug_stripped(
    state: RRTMGLWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGLWColumnResult:
    """Hand-stripped sibling used for the HLO debug identity proof."""

    return _longwave_impl(state, tables, False)
