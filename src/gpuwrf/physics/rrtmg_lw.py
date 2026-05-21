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
    AVOGADRO,
    CH4_VMR,
    CO2_VMR,
    CP_AIR,
    DRY_AIR_MOLECULAR_WEIGHT,
    GRAVITY,
    LW_DIFFUSIVITY_A0,
    LW_DIFFUSIVITY_A1,
    LW_DIFFUSIVITY_A2,
    MAX_OPTICAL_DEPTH,
    MIN_LAYER_MASS,
    MIN_OPTICAL_DEPTH,
    N2O_VMR,
    O2_VMR,
    O3_BACKGROUND_VMR,
    STEFAN_BOLTZMANN,
    WATER_VAPOR_MOLECULAR_WEIGHT_RATIO,
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


def _pressure_interfaces(p):
    """Reconstructs WRF harness pressure interfaces from midpoint pressures."""

    nz = p.shape[-1]
    dp_bottom = jnp.maximum(10.0, p[..., 0] - p[..., 1])
    bottom = p[..., :1] + 0.5 * dp_bottom[..., None]
    middle = 0.5 * (p[..., :-1] + p[..., 1:])
    dp_top = jnp.maximum(10.0, p[..., -2] - p[..., -1])
    top = jnp.maximum(400.0, p[..., -1:] - 0.5 * dp_top[..., None])
    return jnp.concatenate((bottom, middle, top), axis=-1)


def _pressure_layer_mass(p):
    """Reconstructs WRF harness layer mass from midpoint pressure interfaces."""

    nz = p.shape[-1]
    interfaces = _pressure_interfaces(p)
    return jnp.maximum((interfaces[..., :nz] - interfaces[..., 1 : nz + 1]) / GRAVITY, MIN_LAYER_MASS)


def _nearest_pressure_coefficients(state_p, tables: RRTMGTableBundle):
    """Selects WRF reference-pressure absorption coefficients per layer."""

    ref_log = jnp.log(tables.lw_reference_pressure_pa)
    layer_log = jnp.log(jnp.maximum(state_p, 1.0))
    idx = jnp.argmin(jnp.abs(layer_log[..., None] - ref_log), axis=-1)
    gathered = jnp.take(tables.lw_absorption_coefficients, idx, axis=1)
    return jnp.moveaxis(gathered, 0, -2)


def _rrtmg_column_amounts(qv, pressure_interfaces):
    """Computes RRTMG-style scaled molecular columns and precipitable water."""

    h2ovmr = qv * WATER_VAPOR_MOLECULAR_WEIGHT_RATIO
    amm = (1.0 - h2ovmr) * DRY_AIR_MOLECULAR_WEIGHT + h2ovmr * 18.0160
    dp_mb = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) * 0.01, 1.0e-8)
    coldry = dp_mb * 1.0e3 * AVOGADRO / (1.0e2 * GRAVITY * amm * (1.0 + h2ovmr))
    colh2o = 1.0e-20 * coldry * h2ovmr
    colco2 = 1.0e-20 * coldry * CO2_VMR
    colo3 = 1.0e-20 * coldry * O3_BACKGROUND_VMR
    coln2o = 1.0e-20 * coldry * N2O_VMR
    colch4 = 1.0e-20 * coldry * CH4_VMR
    colo2 = 1.0e-20 * coldry * O2_VMR
    absorber = colh2o + 0.03 * colco2 + 0.05 * colo3 + 0.02 * coln2o + 0.02 * colch4 + 0.0001 * colo2
    dry_plus_water = coldry + coldry * h2ovmr
    pwvcm = jnp.sum(18.0160 * coldry * h2ovmr, axis=-1) / jnp.maximum(DRY_AIR_MOLECULAR_WEIGHT * jnp.sum(dry_plus_water, axis=-1), 1.0e-12)
    pwvcm = pwvcm * (1.0e3 * pressure_interfaces[..., 0] * 0.01) / (1.0e2 * GRAVITY)
    return absorber, pwvcm


def _lw_diffusivity(pwvcm):
    """Returns RRTMG LW band diffusivity secants."""

    a0 = jnp.asarray(LW_DIFFUSIVITY_A0, dtype=jnp.float64)
    a1 = jnp.asarray(LW_DIFFUSIVITY_A1, dtype=jnp.float64)
    a2 = jnp.asarray(LW_DIFFUSIVITY_A2, dtype=jnp.float64)
    secdiff = a0 + a1 * jnp.exp(a2 * pwvcm[..., None])
    variable = jnp.asarray([False, True, True, False, True, True, True, True, True, False, False, False, False, False, False, False])
    return jnp.where(variable, jnp.clip(secdiff, 1.50, 1.80), 1.66)


def _source_recurrence_down(trans, source):
    """Top-to-bottom LW source recurrence."""

    nlay = trans.shape[-3]
    rad = jnp.zeros_like(source[..., :1, :, :])
    levels = [rad]
    for idx in range(nlay - 1, -1, -1):
        rad = rad * trans[..., idx : idx + 1, :, :] + source[..., idx : idx + 1, :, :] * (1.0 - trans[..., idx : idx + 1, :, :])
        levels.append(rad)
    return jnp.concatenate(levels[::-1], axis=-3)


def _source_recurrence_up(trans, source, surface):
    """Bottom-to-top LW source recurrence with surface emission."""

    nlay = trans.shape[-3]
    rad = surface[..., None, :, :]
    levels = [rad]
    for idx in range(nlay):
        rad = rad * trans[..., idx : idx + 1, :, :] + source[..., idx : idx + 1, :, :] * (1.0 - trans[..., idx : idx + 1, :, :])
        levels.append(rad)
    return jnp.concatenate(levels, axis=-3)


def _longwave_impl(state: RRTMGLWColumnState, tables: RRTMGTableBundle, debug: bool) -> RRTMGLWColumnResult:
    """Unjitted LW implementation shared by production and stripped paths."""

    state = _clip_state(state)
    original_layers = state.p.shape[-1]
    original_interfaces = _pressure_interfaces(state.p)
    top_pressure = 0.5 * original_interfaces[..., -1:]
    pressure_interfaces = jnp.concatenate((original_interfaces, jnp.full_like(top_pressure, 1.0e-3)), axis=-1)
    p_ext = jnp.concatenate((state.p, top_pressure), axis=-1)
    t_ext = jnp.concatenate((state.T, state.T[..., -1:]), axis=-1)
    qv_ext = jnp.concatenate((state.qv, state.qv[..., -1:]), axis=-1)
    qc_ext = jnp.concatenate((state.qc, jnp.zeros_like(state.qc[..., -1:])), axis=-1)
    qi_ext = jnp.concatenate((state.qi, jnp.zeros_like(state.qi[..., -1:])), axis=-1)
    qs_ext = jnp.concatenate((state.qs, jnp.zeros_like(state.qs[..., -1:])), axis=-1)
    qg_ext = jnp.concatenate((state.qg, jnp.zeros_like(state.qg[..., -1:])), axis=-1)
    cloud_ext = jnp.concatenate((state.cloud_fraction, jnp.zeros_like(state.cloud_fraction[..., -1:])), axis=-1)
    layer_mass = _pressure_layer_mass(state.p)
    layer_mass_ext = jnp.maximum((pressure_interfaces[..., :-1] - pressure_interfaces[..., 1:]) / GRAVITY, MIN_LAYER_MASS)
    gas_column, pwvcm = _rrtmg_column_amounts(qv_ext, pressure_interfaces)
    cloud_path_g = (qc_ext + qi_ext + qs_ext + qg_ext) * layer_mass_ext * 1000.0 * cloud_ext

    weights = tables.lw_gpoint_weights
    gas_coeff = _nearest_pressure_coefficients(p_ext, tables)
    cloud_coeff = tables.lw_cloud_absorption[:, None]
    mask = tables.lw_gpoint_mask

    secdiff = _lw_diffusivity(pwvcm)
    tau = jnp.clip(gas_column[..., None, None] * jnp.maximum(gas_coeff, 0.0) + cloud_path_g[..., None, None] * cloud_coeff, 0.0, MAX_OPTICAL_DEPTH)
    tau = tau * mask
    optical_path = tau * secdiff[..., None, :, None]
    trans = jnp.exp(-jnp.minimum(jnp.maximum(optical_path, MIN_OPTICAL_DEPTH), MAX_OPTICAL_DEPTH))
    trans = jnp.where(mask > 0.0, trans, 1.0)
    layer_source = STEFAN_BOLTZMANN * t_ext[..., None, None] ** 4 * weights
    surface_blackbody = STEFAN_BOLTZMANN * state.surface_temperature[..., None, None] ** 4
    surface_emission_band = surface_blackbody * state.surface_emissivity[..., None, None] * weights

    down_band = _source_recurrence_down(trans, layer_source)
    surface_reflectance = (1.0 - state.surface_emissivity[..., None, None]) * down_band[..., 0, :, :]
    up_band = _source_recurrence_up(trans, layer_source, surface_emission_band + surface_reflectance)

    flux_up_model = jnp.sum(up_band, axis=(-1, -2))
    flux_down_model = jnp.sum(down_band, axis=(-1, -2))
    net_down = flux_down_model - flux_up_model
    layer_net_heating = net_down[..., 1 : original_layers + 1] - net_down[..., :original_layers]
    heating_rate = layer_net_heating / (layer_mass * CP_AIR)
    surface_emission = STEFAN_BOLTZMANN * state.surface_emissivity * state.surface_temperature**4
    flux_down = flux_down_model
    flux_up = flux_up_model

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
