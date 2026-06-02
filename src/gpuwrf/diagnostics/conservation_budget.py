"""WRF-shaped conservation budget diagnostics.

This module is intentionally standalone: it reduces an already-built
``State`` on a ``GridSpec`` and optional accumulated diagnostic side channels.
It does not run the scan and does not hide nonfinite or negative state values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.physics.surface_constants import CP_D, P0_PA, R_D_OVER_CP, XLV


GRAVITY_M_S2 = 9.80665
PRECIP_MM_TO_KG_M2 = 1.0

WATER_SPECIES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")
PRECIP_ACCUMULATORS: tuple[str, ...] = ("rain_acc", "snow_acc", "graupel_acc", "ice_acc")

PREDECLARED_TOLERANCES: dict[str, float] = {
    "closed_domain_dry_mass_relative_residual": 1.0e-10,
    "open_domain_lbc_corrected_dry_mass_relative_residual_24h": 1.0e-5,
    "closed_domain_total_water_relative_residual": 1.0e-8,
    "open_domain_lbc_precip_evap_corrected_water_relative_residual_24h": 1.0e-4,
    "controlled_cpu_budget_relative_residual": 1.0e-12,
    "controlled_cpu_budget_absolute_residual_kg": 1.0e-6,
}


@dataclass(frozen=True)
class GuardLimiterTerm:
    """One aggregated guard/limiter counter emitted by a scan hook."""

    count: jnp.ndarray
    sum_magnitude: jnp.ndarray
    max_magnitude: jnp.ndarray
    signed_magnitude: jnp.ndarray


@dataclass(frozen=True)
class ConservationBudget:
    """Instantaneous and accumulated conservation terms in physical units."""

    dry_mass_kg: jnp.ndarray
    dry_mass_pa_m2: jnp.ndarray
    layer_dry_mass_total_kg: jnp.ndarray
    total_water_air_kg: jnp.ndarray
    precip_accumulated_kg: jnp.ndarray
    surface_evap_accumulated_kg: jnp.ndarray
    water_storage_plus_sinks_kg: jnp.ndarray
    dry_static_energy_j: jnp.ndarray
    moist_static_energy_j: jnp.ndarray
    sensible_enthalpy_j: jnp.ndarray
    geopotential_energy_j: jnp.ndarray
    vapor_latent_energy_j: jnp.ndarray
    precip_components_kg: Mapping[str, jnp.ndarray]
    instantaneous_qfx_kg_m2_s: jnp.ndarray | None = None
    guard_limiter_terms: Mapping[str, GuardLimiterTerm] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetCorrections:
    """Run-integrated flux/source terms, positive into the model domain."""

    dry_mass_lbc_flux_kg: Any = 0.0
    water_lbc_flux_kg: Any = 0.0
    moist_static_energy_lbc_flux_j: Any = 0.0
    moist_static_energy_external_source_j: Any = 0.0


@dataclass(frozen=True)
class ConservationClosure:
    """Initial-to-final closure residuals."""

    dry_mass_residual_kg: jnp.ndarray
    dry_mass_relative_residual: jnp.ndarray
    water_residual_kg: jnp.ndarray
    water_relative_residual: jnp.ndarray
    moist_static_energy_residual_j: jnp.ndarray
    moist_static_energy_relative_residual: jnp.ndarray
    guard_limiter_total_count: jnp.ndarray
    guard_limiter_total_sum_magnitude: jnp.ndarray


def mass_cell_area_m2(grid: GridSpec) -> jnp.ndarray:
    """Return real mass-cell area dx*dy/(MAPFAC_MX*MAPFAC_MY), in m2."""

    metrics = grid.metrics
    if metrics is None:
        raise ValueError("GridSpec.metrics is required for conservation budgets")
    return (
        float(grid.projection.dx_m)
        * float(grid.projection.dy_m)
        / (jnp.asarray(metrics.msftx, dtype=jnp.float64) * jnp.asarray(metrics.msfty, dtype=jnp.float64))
    )


def layer_dry_pressure_pa(state: State, grid: GridSpec) -> jnp.ndarray:
    """Return dry pressure thickness per layer, shape (nz, ny, nx), in Pa.

    WRF hybrid-eta face coefficients define dry pressure on eta faces. The layer
    pressure thickness is the absolute face difference; for sigma/flat metrics it
    reduces to ``abs(deta) * mu_total`` and sums back to the WRF dry column mass.
    """

    metrics = grid.metrics
    if metrics is None:
        raise ValueError("GridSpec.metrics is required for conservation budgets")
    mu = jnp.asarray(state.mu_total, dtype=jnp.float64)
    dc1 = jnp.asarray(metrics.c1f[1:] - metrics.c1f[:-1], dtype=jnp.float64)
    dc2 = jnp.asarray(metrics.c2f[1:] - metrics.c2f[:-1], dtype=jnp.float64)
    return jnp.abs(dc1[:, None, None] * mu[None, :, :] + dc2[:, None, None])


def layer_dry_mass_kg(state: State, grid: GridSpec) -> jnp.ndarray:
    """Return dry-air mass per layer and cell, shape (nz, ny, nx), in kg."""

    return layer_dry_pressure_pa(state, grid) * mass_cell_area_m2(grid)[None, :, :] / GRAVITY_M_S2


def dry_column_mass_kg(state: State, grid: GridSpec) -> jnp.ndarray:
    """Return WRF dry-column mass per cell, shape (ny, nx), in kg."""

    return jnp.asarray(state.mu_total, dtype=jnp.float64) * mass_cell_area_m2(grid) / GRAVITY_M_S2


def _diagnostic_array(diagnostics: Mapping[str, Any] | None, *names: str) -> Any | None:
    if diagnostics is None:
        return None
    for name in names:
        if name in diagnostics:
            return diagnostics[name]
    return None


def _integrated_surface_term_kg(
    diagnostics: Mapping[str, Any] | None,
    area_m2: jnp.ndarray,
    *,
    scalar_names: tuple[str, ...],
    kg_m2_names: tuple[str, ...],
) -> jnp.ndarray:
    scalar = _diagnostic_array(diagnostics, *scalar_names)
    if scalar is not None:
        return jnp.asarray(scalar, dtype=jnp.float64)
    kg_m2 = _diagnostic_array(diagnostics, *kg_m2_names)
    if kg_m2 is not None:
        return jnp.sum(jnp.asarray(kg_m2, dtype=jnp.float64) * area_m2)
    return jnp.asarray(0.0, dtype=jnp.float64)


def _instantaneous_qfx(state: State) -> jnp.ndarray | None:
    if not hasattr(state, "qv_flux") or not hasattr(state, "rhosfc"):
        return None
    return jnp.asarray(state.qv_flux, dtype=jnp.float64) * jnp.asarray(state.rhosfc, dtype=jnp.float64)


def _total_water_mixing_ratio(state: State) -> jnp.ndarray:
    total = jnp.zeros_like(jnp.asarray(state.qv, dtype=jnp.float64))
    for name in WATER_SPECIES:
        total = total + jnp.asarray(getattr(state, name), dtype=jnp.float64)
    return total


def _precip_components(state: State, area_m2: jnp.ndarray) -> dict[str, jnp.ndarray]:
    return {
        name: jnp.sum(jnp.asarray(getattr(state, name), dtype=jnp.float64) * PRECIP_MM_TO_KG_M2 * area_m2)
        for name in PRECIP_ACCUMULATORS
    }


def _temperature_k(state: State) -> jnp.ndarray:
    """Return actual temperature from potential temperature and total pressure."""

    pressure = jnp.asarray(state.p_total, dtype=jnp.float64)
    theta = jnp.asarray(state.theta, dtype=jnp.float64)
    return theta * (pressure / P0_PA) ** R_D_OVER_CP


def _geopotential_mass_point(state: State) -> jnp.ndarray:
    ph = jnp.asarray(state.ph_total, dtype=jnp.float64)
    return 0.5 * (ph[:-1, :, :] + ph[1:, :, :])


def _normalise_guard_limiter_terms(diagnostics: Mapping[str, Any] | None) -> dict[str, GuardLimiterTerm]:
    raw = _diagnostic_array(diagnostics, "guard_limiter_terms", "guards", "limiters")
    if raw is None:
        return {}
    terms: dict[str, GuardLimiterTerm] = {}
    for name, payload in dict(raw).items():
        if isinstance(payload, GuardLimiterTerm):
            terms[str(name)] = payload
            continue
        if isinstance(payload, Mapping):
            terms[str(name)] = GuardLimiterTerm(
                count=jnp.asarray(payload.get("count", 0), dtype=jnp.int64),
                sum_magnitude=jnp.asarray(payload.get("sum_magnitude", payload.get("magnitude", 0.0)), dtype=jnp.float64),
                max_magnitude=jnp.asarray(payload.get("max_magnitude", 0.0), dtype=jnp.float64),
                signed_magnitude=jnp.asarray(payload.get("signed_magnitude", 0.0), dtype=jnp.float64),
            )
            continue
        terms[str(name)] = GuardLimiterTerm(
            count=jnp.asarray(payload, dtype=jnp.int64),
            sum_magnitude=jnp.asarray(0.0, dtype=jnp.float64),
            max_magnitude=jnp.asarray(0.0, dtype=jnp.float64),
            signed_magnitude=jnp.asarray(0.0, dtype=jnp.float64),
        )
    return terms


def compute_conservation_budget(
    state: State,
    grid: GridSpec,
    diagnostics: Mapping[str, Any] | None = None,
) -> ConservationBudget:
    """Compute dry-mass, water, precip, and moist-static-energy terms.

    Surface evaporation is accumulated as a positive upward QFX source and is
    subtracted in ``water_storage_plus_sinks_kg`` so the closed-budget residual is
    ``delta(column_water + precip - evap)``.
    """

    area = mass_cell_area_m2(grid)
    layer_mass = layer_dry_mass_kg(state, grid)

    dry_column_pa_m2 = jnp.sum(jnp.asarray(state.mu_total, dtype=jnp.float64) * area)
    dry_mass = jnp.sum(dry_column_mass_kg(state, grid))

    total_water_air = jnp.sum(layer_mass * _total_water_mixing_ratio(state))
    precip_by_name = _precip_components(state, area)
    precip_total = sum(precip_by_name.values(), jnp.asarray(0.0, dtype=jnp.float64))
    evap_total = _integrated_surface_term_kg(
        diagnostics,
        area,
        scalar_names=("surface_evap_accumulated_kg", "qfx_accumulated_kg", "evap_accumulated_kg"),
        kg_m2_names=("surface_evap_accumulated_kg_m2", "qfx_accumulated_kg_m2", "evap_accumulated_kg_m2"),
    )

    temperature = _temperature_k(state)
    geopotential = _geopotential_mass_point(state)
    sensible = jnp.sum(layer_mass * CP_D * temperature)
    geopotential_energy = jnp.sum(layer_mass * geopotential)
    vapor_latent = jnp.sum(layer_mass * XLV * jnp.asarray(state.qv, dtype=jnp.float64))
    dry_static = sensible + geopotential_energy
    moist_static = dry_static + vapor_latent

    return ConservationBudget(
        dry_mass_kg=dry_mass,
        dry_mass_pa_m2=dry_column_pa_m2,
        layer_dry_mass_total_kg=jnp.sum(layer_mass),
        total_water_air_kg=total_water_air,
        precip_accumulated_kg=precip_total,
        surface_evap_accumulated_kg=evap_total,
        water_storage_plus_sinks_kg=total_water_air + precip_total - evap_total,
        dry_static_energy_j=dry_static,
        moist_static_energy_j=moist_static,
        sensible_enthalpy_j=sensible,
        geopotential_energy_j=geopotential_energy,
        vapor_latent_energy_j=vapor_latent,
        precip_components_kg=precip_by_name,
        instantaneous_qfx_kg_m2_s=_instantaneous_qfx(state),
        guard_limiter_terms=_normalise_guard_limiter_terms(diagnostics),
    )


def _as_corrections(corrections: BudgetCorrections | Mapping[str, Any] | None) -> BudgetCorrections:
    if corrections is None:
        return BudgetCorrections()
    if isinstance(corrections, BudgetCorrections):
        return corrections
    return BudgetCorrections(
        dry_mass_lbc_flux_kg=corrections.get("dry_mass_lbc_flux_kg", 0.0),
        water_lbc_flux_kg=corrections.get("water_lbc_flux_kg", 0.0),
        moist_static_energy_lbc_flux_j=corrections.get("moist_static_energy_lbc_flux_j", 0.0),
        moist_static_energy_external_source_j=corrections.get("moist_static_energy_external_source_j", 0.0),
    )


def _relative_residual(residual: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(reference != 0.0, residual / jnp.abs(reference), jnp.asarray(jnp.nan, dtype=jnp.float64))


def compute_budget_closure(
    initial: ConservationBudget,
    final: ConservationBudget,
    corrections: BudgetCorrections | Mapping[str, Any] | None = None,
) -> ConservationClosure:
    """Return initial-to-final residuals after run-integrated corrections.

    Lateral-boundary flux corrections are positive into the domain. Energy is a
    moist-static-energy diagnostic residual; WRF ARW is not claimed here to be a
    closed total-energy system.
    """

    corr = _as_corrections(corrections)
    dry_residual = (
        jnp.asarray(final.dry_mass_kg, dtype=jnp.float64)
        - jnp.asarray(initial.dry_mass_kg, dtype=jnp.float64)
        - jnp.asarray(corr.dry_mass_lbc_flux_kg, dtype=jnp.float64)
    )
    water_residual = (
        jnp.asarray(final.water_storage_plus_sinks_kg, dtype=jnp.float64)
        - jnp.asarray(initial.water_storage_plus_sinks_kg, dtype=jnp.float64)
        - jnp.asarray(corr.water_lbc_flux_kg, dtype=jnp.float64)
    )
    energy_residual = (
        jnp.asarray(final.moist_static_energy_j, dtype=jnp.float64)
        - jnp.asarray(initial.moist_static_energy_j, dtype=jnp.float64)
        - jnp.asarray(corr.moist_static_energy_lbc_flux_j, dtype=jnp.float64)
        - jnp.asarray(corr.moist_static_energy_external_source_j, dtype=jnp.float64)
    )
    guard_count = sum((term.count for term in final.guard_limiter_terms.values()), jnp.asarray(0, dtype=jnp.int64))
    guard_magnitude = sum(
        (term.sum_magnitude for term in final.guard_limiter_terms.values()),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    return ConservationClosure(
        dry_mass_residual_kg=dry_residual,
        dry_mass_relative_residual=_relative_residual(dry_residual, initial.dry_mass_kg),
        water_residual_kg=water_residual,
        water_relative_residual=_relative_residual(water_residual, initial.water_storage_plus_sinks_kg),
        moist_static_energy_residual_j=energy_residual,
        moist_static_energy_relative_residual=_relative_residual(energy_residual, initial.moist_static_energy_j),
        guard_limiter_total_count=guard_count,
        guard_limiter_total_sum_magnitude=guard_magnitude,
    )


__all__ = [
    "BudgetCorrections",
    "ConservationBudget",
    "ConservationClosure",
    "GuardLimiterTerm",
    "PREDECLARED_TOLERANCES",
    "compute_budget_closure",
    "compute_conservation_budget",
    "dry_column_mass_kg",
    "layer_dry_mass_kg",
    "layer_dry_pressure_pa",
    "mass_cell_area_m2",
]
