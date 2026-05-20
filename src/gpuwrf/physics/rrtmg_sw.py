"""JAX shortwave RRTMG-style radiation column kernel for M5-S3."""

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
    MIN_COSZEN,
    MIN_LAYER_MASS,
    MIN_OPTICAL_DEPTH,
    SOLAR_CONSTANT,
)
from gpuwrf.physics.rrtmg_tables import RRTMGTableBundle, RRTMG_TABLES


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class RRTMGSWColumnState:
    """Pytree for independent shortwave radiation columns on mass levels."""

    __slots__ = ("T", "p", "qv", "qc", "qi", "qs", "qg", "cloud_fraction", "surface_albedo", "coszen", "dz", "rho")

    def __init__(self, T, p, qv, qc, qi, qs, qg, cloud_fraction, surface_albedo, coszen, dz, rho) -> None:
        self.T = T
        self.p = p
        self.qv = qv
        self.qc = qc
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.cloud_fraction = cloud_fraction
        self.surface_albedo = surface_albedo
        self.coszen = coszen
        self.dz = dz
        self.rho = rho

    def replace(self, **updates) -> "RRTMGSWColumnState":
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

        if not isinstance(other, RRTMGSWColumnState):
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


class RRTMGSWColumnResult(NamedTuple):
    """Shortwave column outputs with bottom-to-top interface fluxes."""

    heating_rate: jnp.ndarray
    flux_down: jnp.ndarray
    flux_up: jnp.ndarray
    toa_down: jnp.ndarray
    toa_up: jnp.ndarray
    surface_down: jnp.ndarray
    surface_up: jnp.ndarray
    column_absorbed: jnp.ndarray
    surface_absorbed: jnp.ndarray


def _leaves(state: RRTMGSWColumnState):
    """Centralizes leaf iteration for equality and hashing."""

    return (getattr(state, name) for name in RRTMGSWColumnState.__slots__)


def _clip_state(state: RRTMGSWColumnState) -> RRTMGSWColumnState:
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
        surface_albedo=jnp.clip(state.surface_albedo, 0.0, 1.0),
        coszen=jnp.maximum(state.coszen, MIN_COSZEN),
        dz=jnp.maximum(state.dz, 1.0),
        rho=jnp.maximum(state.rho, MIN_LAYER_MASS),
    )


def _expand_surface(x):
    """Expands scalar/column surface values onto the spectral band axis."""

    return x[..., None]


def _shortwave_impl(state: RRTMGSWColumnState, tables: RRTMGTableBundle, debug: bool) -> RRTMGSWColumnResult:
    """Unjitted SW implementation shared by production and stripped paths."""

    state = _clip_state(state)
    layer_mass = jnp.maximum(state.rho * state.dz, MIN_LAYER_MASS)
    vapor_path = state.qv * layer_mass
    liquid_path = (state.qc + 0.25 * state.qg) * layer_mass
    ice_path = (state.qi + state.qs + 0.75 * state.qg) * layer_mass

    band_index = jnp.arange(tables.sw_band_weights.shape[0])
    weights = jnp.take(tables.sw_band_weights, band_index, axis=0)
    gas_coeff = jnp.take(tables.sw_absorption_coefficients, band_index, axis=0)
    rayleigh_coeff = jnp.take(tables.sw_rayleigh_coefficients, band_index, axis=0)
    liquid_coeff = jnp.take(tables.sw_cloud_liquid_extinction, band_index, axis=0)
    ice_coeff = jnp.take(tables.sw_cloud_ice_extinction, band_index, axis=0)

    pressure_scale = jnp.sqrt(jnp.maximum(state.p, 1.0) / 100000.0)
    tau_gas = vapor_path[..., None] * gas_coeff
    tau_rayleigh = pressure_scale[..., None] * rayleigh_coeff
    tau_cloud = state.cloud_fraction[..., None] * (liquid_path[..., None] * liquid_coeff + ice_path[..., None] * ice_coeff)
    tau = jnp.clip(tau_gas + tau_rayleigh + tau_cloud, MIN_OPTICAL_DEPTH, MAX_OPTICAL_DEPTH)

    top_flux_band = SOLAR_CONSTANT * _expand_surface(state.coszen) * weights
    tau_top_down = jnp.flip(tau, axis=-2)
    zeros = jnp.zeros_like(tau_top_down[..., :1, :])
    optical_depth_interfaces_top_down = jnp.concatenate((zeros, jnp.cumsum(tau_top_down, axis=-2)), axis=-2)
    down_top_down = top_flux_band[..., None, :] * jnp.exp(-optical_depth_interfaces_top_down)
    down_band = jnp.flip(down_top_down, axis=-2)

    surface_down_band = down_band[..., 0, :]
    surface_up_band = _expand_surface(state.surface_albedo) * surface_down_band
    optical_depth_interfaces_bottom_up = jnp.concatenate((jnp.zeros_like(tau[..., :1, :]), jnp.cumsum(tau, axis=-2)), axis=-2)
    up_band = surface_up_band[..., None, :] * jnp.exp(-optical_depth_interfaces_bottom_up)

    flux_down = jnp.sum(down_band, axis=-1)
    flux_up = jnp.sum(up_band, axis=-1)
    net_down = flux_down - flux_up
    column_absorbed_layers = net_down[..., 1:] - net_down[..., :-1]
    heating_rate = column_absorbed_layers / (layer_mass * CP_AIR)
    surface_absorbed = jnp.sum(surface_down_band - surface_up_band, axis=-1)

    heating_rate = assert_finite(heating_rate, "rrtmg_sw.heating_rate", enabled=debug)
    flux_down = assert_physical_bounds(flux_down, 0.0, 2000.0, "rrtmg_sw.flux_down", enabled=debug)
    flux_up = assert_physical_bounds(flux_up, 0.0, 2000.0, "rrtmg_sw.flux_up", enabled=debug)
    return RRTMGSWColumnResult(
        heating_rate=heating_rate,
        flux_down=flux_down,
        flux_up=flux_up,
        toa_down=flux_down[..., -1],
        toa_up=flux_up[..., -1],
        surface_down=flux_down[..., 0],
        surface_up=flux_up[..., 0],
        column_absorbed=jnp.sum(column_absorbed_layers, axis=-1),
        surface_absorbed=surface_absorbed,
    )


@partial(jax.jit, static_argnames=("debug",))
def solve_rrtmg_sw_column(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
    *,
    debug: bool = False,
) -> RRTMGSWColumnResult:
    """Computes one fused shortwave column radiation call."""

    return _shortwave_impl(state, tables, debug)


@jax.jit
def solve_rrtmg_sw_column_debug_stripped(
    state: RRTMGSWColumnState,
    tables: RRTMGTableBundle = RRTMG_TABLES,
) -> RRTMGSWColumnResult:
    """Hand-stripped sibling used for the HLO debug identity proof."""

    return _shortwave_impl(state, tables, False)
