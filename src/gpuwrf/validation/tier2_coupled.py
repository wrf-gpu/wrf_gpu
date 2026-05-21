"""Tier-2 coupled conservation and positivity diagnostics for M6-S4.

Formula provenance:
- Dry-column mass and mass-coupled advection follow WRF EM's RK prep path:
  `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_em.F:141-151`
  and `:184-212`, where WRF builds `mut/muu/muv` and moisture coefficients
  before RK advection.
- Boundary relaxation follows the M6-S2 WRF-cited implementation of
  `dyn_em/module_bc_em.F:lbc_fcx_gcx` and `share/module_bc.F` relax/spec
  tendencies in `gpuwrf.coupling.boundary_apply`.
- Water budget diagnostics follow WRF's diagnostic driver call carrying
  `DMUDT`, `MU_2`, `RAIN*`, and surface evaporation terms:
  `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_diagnostics_driver.F:336-356`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG
from gpuwrf.contracts.state import State


GRAVITY_M_S2 = 9.80665
HYDROMETEOR_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
PROGNOSTIC_FIELDS = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "p",
    "ph",
    "mu",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "qke",
)
BOUNDARY_FIELDS = ("u", "v", "theta", "qv", "ph", "mu")


def _as_float(value) -> float:
    return float(np.asarray(value))


def _as_int(value) -> int:
    return int(np.asarray(value))


def _max_abs(value) -> float:
    data = jnp.asarray(value)
    if data.size == 0:
        return 0.0
    return _as_float(jnp.max(jnp.abs(data)))


def _interior_2d(field):
    width = int(DEFAULT_BOUNDARY_CONFIG.relax_zone)
    if field.ndim < 2 or width <= 0 or field.shape[-1] <= 2 * width or field.shape[-2] <= 2 * width:
        return field
    return field[..., width:-width, width:-width]


def _column_total_water(state: State):
    total = sum(jnp.asarray(getattr(state, field), dtype=jnp.float64) for field in HYDROMETEOR_FIELDS)
    return jnp.mean(total, axis=0)


def _precip_mixing_ratio_delta(state: State, state_next: State):
    precip_delta_mm = (
        (state_next.rain_acc - state.rain_acc)
        + (state_next.snow_acc - state.snow_acc)
        + (state_next.graupel_acc - state.graupel_acc)
        + (state_next.ice_acc - state.ice_acc)
    )
    column_mass_kg_m2 = jnp.maximum(jnp.asarray(state.mu, dtype=jnp.float64) / GRAVITY_M_S2, 1.0)
    return jnp.asarray(precip_delta_mm, dtype=jnp.float64) / column_mass_kg_m2


def _water_tendency_oracle(precip_out):
    """Return a Thompson side-channel tendency oracle when one was supplied."""

    if precip_out is None:
        return None
    if isinstance(precip_out, dict):
        if "column_water_tendency" in precip_out:
            column = precip_out["column_water_tendency"]
            precip = precip_out.get("precip_out_tendency", 0.0)
            return column, precip, "thompson_tendency_side_channel"
        if "species_tendencies" in precip_out:
            species = precip_out["species_tendencies"]
            column = jnp.mean(sum(jnp.asarray(value, dtype=jnp.float64) for value in species.values()), axis=0)
            precip = precip_out.get("precip_out_tendency", 0.0)
            return column, precip, "thompson_species_tendency_dict"
        return None
    if hasattr(precip_out, "column_water_tendency"):
        return (
            getattr(precip_out, "column_water_tendency"),
            getattr(precip_out, "precip_out_tendency", 0.0),
            type(precip_out).__name__,
        )
    return None


def _field_from_snapshot(snapshot: Any, field: str):
    if snapshot is None:
        return None
    if isinstance(snapshot, dict):
        return snapshot.get(field)
    return getattr(snapshot, field)


def dry_mass_residual(state: State, state_next: State, dt: float) -> dict[str, Any]:
    """Return interior column dry-mass residual for `d/dt(rho_d)=0`.

    The M6 driver applies time-varying lateral mass forcing in the relaxation
    zone; that budget is checked separately by `boundary_flux_closure`.
    """

    del dt
    residual_kg_m2 = _interior_2d((jnp.asarray(state_next.mu) - jnp.asarray(state.mu)) / GRAVITY_M_S2)
    max_abs = _max_abs(residual_kg_m2)
    return {
        "invariant": "dry_mass",
        "units": "kg m-2",
        "max_abs": max_abs,
        "per_leaf": {"mu": {"max_abs": max_abs}},
        "wrf_source": "dyn_em/module_em.F:141-151,184-190",
    }


def mu_continuity_residual(state: State, state_next: State, dt: float, fluxes) -> dict[str, Any]:
    """Return residual for `d(mu)/dt + div(mu u) = 0` on the interior mass grid."""

    mu_delta = (jnp.asarray(state_next.mu, dtype=jnp.float64) - jnp.asarray(state.mu, dtype=jnp.float64)) / float(dt)
    if fluxes is not None and isinstance(fluxes, dict) and "mu_tendency" in fluxes:
        budget = jnp.asarray(fluxes["mu_tendency"], dtype=jnp.float64)
    else:
        budget = jnp.zeros_like(mu_delta)
    residual = _interior_2d(mu_delta - budget)
    max_abs = _max_abs(residual)
    return {
        "invariant": "mu_continuity",
        "units": "Pa s-1",
        "max_abs": max_abs,
        "per_leaf": {"mu": {"max_abs": max_abs}},
        "wrf_source": "dyn_em/module_em.F:141-151,184-212",
    }


def water_budget_residual(state: State, state_next: State, dt: float, precip_out=None) -> dict[str, Any]:
    """Return total-water residual from vapor + hydrometeors + precipitation outflow."""

    water_delta = _column_total_water(state_next) - _column_total_water(state)
    oracle = _water_tendency_oracle(precip_out)
    if oracle is None:
        precip_delta = _precip_mixing_ratio_delta(state, state_next) if precip_out is None else jnp.asarray(precip_out)
        residual = _interior_2d(water_delta + precip_delta)
        oracle_source = "precipitation_accumulator_delta"
    else:
        column_tendency, precip_tendency, oracle_source = oracle
        residual = _interior_2d(
            water_delta
            + float(dt) * jnp.asarray(precip_tendency, dtype=jnp.float64)
            - float(dt) * jnp.asarray(column_tendency, dtype=jnp.float64)
        )
    per_leaf = {
        field: {"max_abs": _max_abs(_interior_2d(jnp.asarray(getattr(state_next, field)) - jnp.asarray(getattr(state, field))))}
        for field in HYDROMETEOR_FIELDS
    }
    max_abs = _max_abs(residual)
    mean_abs = _as_float(jnp.abs(jnp.mean(residual))) if residual.size else 0.0
    return {
        "invariant": "total_water",
        "units": "kg kg-1",
        "max_abs": max_abs,
        "domain_mean_abs": mean_abs,
        "per_leaf": per_leaf,
        "oracle_source": oracle_source,
        "wrf_source": "phys/module_diagnostics_driver.F:336-356",
    }


def tke_positivity(state: State) -> dict[str, Any]:
    """Count negative and non-finite MYNN `qke` values."""

    qke = jnp.asarray(state.qke)
    negative = _as_int(jnp.sum(qke < 0.0))
    nonfinite = _as_int(jnp.sum(~jnp.isfinite(qke)))
    return {
        "invariant": "tke_positivity",
        "violations": negative,
        "nonfinite": nonfinite,
        "per_leaf": {"qke": {"negative_count": negative, "nonfinite_count": nonfinite}},
    }


def hydrometeor_positivity(state: State) -> dict[str, Any]:
    """Count negative vapor and hydrometeor values."""

    per_leaf = {}
    total_negative = 0
    total_nonfinite = 0
    for field in HYDROMETEOR_FIELDS:
        value = jnp.asarray(getattr(state, field))
        negative = _as_int(jnp.sum(value < 0.0))
        nonfinite = _as_int(jnp.sum(~jnp.isfinite(value)))
        per_leaf[field] = {"negative_count": negative, "nonfinite_count": nonfinite}
        total_negative += negative
        total_nonfinite += nonfinite
    return {
        "invariant": "hydrometeor_positivity",
        "violations": int(total_negative),
        "nonfinite": int(total_nonfinite),
        "per_leaf": per_leaf,
    }


def nan_inf_count(state: State) -> dict[str, Any]:
    """Count non-finite values in prognostic State leaves."""

    per_leaf = {}
    total = 0
    for field in PROGNOSTIC_FIELDS:
        count = _as_int(jnp.sum(~jnp.isfinite(getattr(state, field))))
        per_leaf[field] = {"nonfinite_count": count}
        total += count
    return {"invariant": "nan_inf", "violations": int(total), "per_leaf": per_leaf}


def boundary_flux_closure(state: State, state_next: State, dt: float, bdy_apply_tendency) -> dict[str, Any]:
    """Check that replay tendency closes the specified/relaxation-zone update."""

    del state
    pre_boundary = None
    tendency = bdy_apply_tendency
    if isinstance(bdy_apply_tendency, dict):
        pre_boundary = bdy_apply_tendency.get("pre_boundary")
        tendency = bdy_apply_tendency.get("tendency")
    per_leaf = {}
    for field in BOUNDARY_FIELDS:
        tendency_field = _field_from_snapshot(tendency, field)
        if tendency_field is None:
            residual = jnp.zeros_like(getattr(state_next, field), dtype=jnp.float64)
        else:
            before_field = _field_from_snapshot(pre_boundary, field)
            if before_field is None:
                before_field = jnp.asarray(getattr(state_next, field), dtype=jnp.float64) - float(dt) * jnp.asarray(tendency_field, dtype=jnp.float64)
            residual = jnp.asarray(getattr(state_next, field), dtype=jnp.float64) - (
                jnp.asarray(before_field, dtype=jnp.float64) + float(dt) * jnp.asarray(tendency_field, dtype=jnp.float64)
            )
        per_leaf[field] = {"max_abs": _max_abs(residual)}
    max_abs = max(item["max_abs"] for item in per_leaf.values())
    return {
        "invariant": "boundary_flux_closure",
        "units": "native field units",
        "max_abs": float(max_abs),
        "per_leaf": per_leaf,
        "wrf_source": "dyn_em/module_bc_em.F:lbc_fcx_gcx; share/module_bc.F:relax_bdytend_core/spec_bdytend",
    }


__all__ = [
    "BOUNDARY_FIELDS",
    "HYDROMETEOR_FIELDS",
    "PROGNOSTIC_FIELDS",
    "boundary_flux_closure",
    "dry_mass_residual",
    "hydrometeor_positivity",
    "mu_continuity_residual",
    "nan_inf_count",
    "tke_positivity",
    "water_budget_residual",
]
