from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.diagnostics.conservation_budget import (
    GRAVITY_M_S2,
    PREDECLARED_TOLERANCES,
    BudgetCorrections,
    compute_budget_closure,
    compute_conservation_budget,
    dry_column_mass_kg,
    layer_dry_mass_kg,
    mass_cell_area_m2,
)


ROOT = Path(__file__).resolve().parents[1]
PROOF_PATH = ROOT / "proofs" / "p0_7" / "conservation_budget_cpu_controlled.json"


def _grid(nx: int = 4, ny: int = 3, nz: int = 4) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -15.6, 1000.0, 1500.0, nx, ny)
    terrain = TerrainProvenance("analytic://p0_7", "p0_7", (ny, nx), "m", "native", 0.0, True)
    vertical = VerticalCoord("hybrid_eta", nz, 16000.0, eta)
    bc = BCMetadata("ideal", ("u", "v", "theta", "qv", "mu"), 0, "linear", False)
    base = GridSpec(
        projection=projection,
        terrain=terrain,
        vertical=vertical,
        bc=bc,
        eta_levels=eta,
        terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
    )
    y = jnp.arange(ny, dtype=jnp.float64)[:, None]
    x = jnp.arange(nx, dtype=jnp.float64)[None, :]
    metrics = replace(
        base.metrics,
        msftx=1.0 + 0.01 * x + 0.005 * y,
        msfty=1.0 + 0.02 * x + 0.003 * y,
    )
    return GridSpec(
        projection=projection,
        terrain=terrain,
        vertical=vertical,
        bc=bc,
        eta_levels=eta,
        terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
        metrics=metrics,
    )


def _state(grid: GridSpec, *, qv: float = 0.010, precip: bool = True) -> State:
    fields = {
        name: jnp.zeros(shape, dtype=jnp.int32 if name == "lu_index" else jnp.float64)
        for name, shape in _state_field_shapes(grid).items()
    }
    z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    y2 = jnp.arange(grid.ny, dtype=jnp.float64)[:, None]
    x2 = jnp.arange(grid.nx, dtype=jnp.float64)[None, :]
    pressure = 90000.0 - 1500.0 * z + 20.0 * y + 5.0 * x
    theta = 300.0 + 0.2 * z + 0.1 * y + 0.02 * x
    ph_faces = GRAVITY_M_S2 * (400.0 * jnp.arange(grid.nz + 1, dtype=jnp.float64)[:, None, None])
    ph_faces = ph_faces + jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64)
    mu_total = 78000.0 + 50.0 * y2 + 7.0 * x2
    mu_base = 76000.0 + jnp.zeros_like(mu_total)
    mu_pert = mu_total - mu_base
    fields.update(
        u=4.0 + jnp.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=jnp.float64),
        v=1.5 + jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=jnp.float64),
        w=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        theta=theta,
        qv=qv + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        qc=0.0010 + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        qr=0.0007 + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        qi=0.0002 + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        qs=0.0003 + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        qg=0.0001 + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        p=pressure,
        p_total=pressure,
        p_perturbation=jnp.zeros_like(pressure),
        ph=ph_faces,
        ph_total=ph_faces,
        ph_perturbation=jnp.zeros_like(ph_faces),
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=mu_pert,
        qv_flux=0.002 + jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
        rhosfc=1.1 + 0.01 * y2 + 0.002 * x2,
    )
    if precip:
        fields.update(
            rain_acc=2.0 + jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
            snow_acc=0.25 + jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
            graupel_acc=0.125 + jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
            ice_acc=0.0625 + jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
        )
    return State(**fields)


def _scalar(value) -> float:
    return float(np.asarray(value))


def test_budget_terms_use_wrf_mass_area_units_and_precip_evap():
    grid = _grid()
    state = _state(grid)
    evap_kg_m2 = jnp.full((grid.ny, grid.nx), 0.5, dtype=jnp.float64)
    diagnostics = {
        "surface_evap_accumulated_kg_m2": evap_kg_m2,
        "guard_limiter_terms": {
            "moisture_floor_qv": {
                "count": 3,
                "sum_magnitude": 2.5e-9,
                "max_magnitude": 1.5e-9,
                "signed_magnitude": 2.5e-9,
            }
        },
    }
    budget = compute_conservation_budget(state, grid, diagnostics)

    area = mass_cell_area_m2(grid)
    layer_mass = layer_dry_mass_kg(state, grid)
    dry_mass = dry_column_mass_kg(state, grid)
    expected_qtot = state.qv + state.qc + state.qr + state.qi + state.qs + state.qg
    expected_precip = jnp.sum(
        (state.rain_acc + state.snow_acc + state.graupel_acc + state.ice_acc) * area
    )
    expected_evap = jnp.sum(evap_kg_m2 * area)

    np.testing.assert_allclose(_scalar(budget.dry_mass_kg), _scalar(jnp.sum(dry_mass)), rtol=1e-13)
    np.testing.assert_allclose(
        _scalar(budget.dry_mass_pa_m2),
        _scalar(jnp.sum(state.mu_total * area)),
        rtol=1e-13,
    )
    np.testing.assert_allclose(_scalar(budget.layer_dry_mass_total_kg), _scalar(jnp.sum(layer_mass)), rtol=1e-13)
    np.testing.assert_allclose(
        _scalar(budget.total_water_air_kg),
        _scalar(jnp.sum(layer_mass * expected_qtot)),
        rtol=1e-13,
    )
    np.testing.assert_allclose(_scalar(budget.precip_accumulated_kg), _scalar(expected_precip), rtol=1e-13)
    np.testing.assert_allclose(_scalar(budget.surface_evap_accumulated_kg), _scalar(expected_evap), rtol=1e-13)
    np.testing.assert_allclose(
        _scalar(budget.water_storage_plus_sinks_kg),
        _scalar(jnp.sum(layer_mass * expected_qtot) + expected_precip - expected_evap),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        _scalar(budget.dry_static_energy_j),
        _scalar(budget.sensible_enthalpy_j + budget.geopotential_energy_j),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        _scalar(budget.moist_static_energy_j),
        _scalar(budget.dry_static_energy_j + budget.vapor_latent_energy_j),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(budget.instantaneous_qfx_kg_m2_s),
        np.asarray(state.qv_flux * state.rhosfc),
        rtol=1e-13,
    )
    assert int(np.asarray(budget.guard_limiter_terms["moisture_floor_qv"].count)) == 3


def test_budget_closure_closed_water_and_lbc_dry_mass_proof():
    grid = _grid()
    initial = _state(grid, qv=0.012, precip=False)
    dq = 0.001
    rain_acc = initial.mu_total / GRAVITY_M_S2 * dq
    final = initial.replace(qv=initial.qv - dq, rain_acc=rain_acc)

    initial_budget = compute_conservation_budget(initial, grid)
    final_budget = compute_conservation_budget(final, grid)
    closed = compute_budget_closure(initial_budget, final_budget)

    dmu = jnp.full((grid.ny, grid.nx), 2.0, dtype=jnp.float64)
    open_final = initial.replace(
        mu=initial.mu_total + dmu,
        mu_total=initial.mu_total + dmu,
        mu_perturbation=initial.mu_perturbation + dmu,
    )
    open_budget = compute_conservation_budget(open_final, grid)
    corrections = BudgetCorrections(
        dry_mass_lbc_flux_kg=open_budget.dry_mass_kg - initial_budget.dry_mass_kg,
        water_lbc_flux_kg=open_budget.water_storage_plus_sinks_kg - initial_budget.water_storage_plus_sinks_kg,
        moist_static_energy_lbc_flux_j=open_budget.moist_static_energy_j - initial_budget.moist_static_energy_j,
    )
    open_closure = compute_budget_closure(initial_budget, open_budget, corrections)

    tol_rel = PREDECLARED_TOLERANCES["controlled_cpu_budget_relative_residual"]
    tol_abs = PREDECLARED_TOLERANCES["controlled_cpu_budget_absolute_residual_kg"]
    assert abs(_scalar(closed.dry_mass_residual_kg)) <= tol_abs
    assert abs(_scalar(closed.water_residual_kg)) <= tol_abs
    assert abs(_scalar(closed.dry_mass_relative_residual)) <= tol_rel
    assert abs(_scalar(closed.water_relative_residual)) <= tol_rel
    assert abs(_scalar(open_closure.dry_mass_residual_kg)) <= tol_abs
    assert abs(_scalar(open_closure.water_residual_kg)) <= tol_abs
    assert abs(_scalar(open_closure.moist_static_energy_residual_j)) <= 1.0e-6

    proof = {
        "schema": "p0_7_conservation_budget_cpu_controlled_v1",
        "platform": jax.default_backend(),
        "grid": {
            "nx": grid.nx,
            "ny": grid.ny,
            "nz": grid.nz,
            "dx_m": grid.projection.dx_m,
            "dy_m": grid.projection.dy_m,
            "map_factors": "nonunit deterministic MAPFAC_MX/MAPFAC_MY",
        },
        "predeclared_tolerances": PREDECLARED_TOLERANCES,
        "closed_control": {
            "dry_mass_residual_kg": _scalar(closed.dry_mass_residual_kg),
            "dry_mass_relative_residual": _scalar(closed.dry_mass_relative_residual),
            "water_residual_kg": _scalar(closed.water_residual_kg),
            "water_relative_residual": _scalar(closed.water_relative_residual),
            "passes_predeclared_tolerance": bool(
                abs(_scalar(closed.dry_mass_relative_residual)) <= tol_rel
                and abs(_scalar(closed.water_relative_residual)) <= tol_rel
            ),
        },
        "open_lbc_corrected_control": {
            "dry_mass_residual_kg": _scalar(open_closure.dry_mass_residual_kg),
            "dry_mass_relative_residual": _scalar(open_closure.dry_mass_relative_residual),
            "water_residual_kg": _scalar(open_closure.water_residual_kg),
            "water_relative_residual": _scalar(open_closure.water_relative_residual),
            "moist_static_energy_residual_j": _scalar(open_closure.moist_static_energy_residual_j),
            "passes_predeclared_tolerance": bool(
                abs(_scalar(open_closure.dry_mass_relative_residual)) <= tol_rel
                and abs(_scalar(open_closure.water_relative_residual)) <= tol_rel
            ),
        },
        "budget_terms_sample": {
            "dry_mass_kg": _scalar(initial_budget.dry_mass_kg),
            "layer_dry_mass_total_kg": _scalar(initial_budget.layer_dry_mass_total_kg),
            "total_water_air_kg": _scalar(initial_budget.total_water_air_kg),
            "moist_static_energy_j": _scalar(initial_budget.moist_static_energy_j),
        },
    }
    PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROOF_PATH.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
