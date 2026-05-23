from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics import acoustic_wrf
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.validation.analytic_oracles.vertical_linear_acoustic import (
    GRAVITY_M_S2,
    vertical_acoustic_mode,
)


N_LEVELS = 16
COLUMN_HEIGHT_M = 10_000.0
THETA_BASE_K = 300.0
BRUNT_N_INV_S = 0.01
WAVELENGTH_M = 2.0 * COLUMN_HEIGHT_M
INITIAL_W_M_S = 1.0


def _column_grid() -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, 1000.0, 1000.0, 1, 1)
    terrain = TerrainProvenance(
        source_path="analytic://m6x-vertical-acoustic-flat-column",
        sha256="analytic-m6x",
        shape=(projection.ny, projection.nx),
        units="m",
        projection_transform="flat-column",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta_levels = jnp.linspace(1.0, 0.0, N_LEVELS + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", N_LEVELS, 5000.0, eta_levels)
    bc = BCMetadata(
        source="ideal",
        fields=("w", "ph", "theta"),
        update_cadence_h=1,
        interpolation="linear",
        restart_compatible=True,
    )
    terrain_height = jnp.zeros(terrain.shape, dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


def _broadcast_face(profile: np.ndarray, grid: GridSpec) -> jnp.ndarray:
    return jnp.broadcast_to(jnp.asarray(profile, dtype=jnp.float64)[:, None, None], (grid.nz + 1, grid.ny, grid.nx))


def _broadcast_mass(profile: np.ndarray, grid: GridSpec) -> jnp.ndarray:
    return jnp.broadcast_to(jnp.asarray(profile, dtype=jnp.float64)[:, None, None], (grid.nz, grid.ny, grid.nx))


def _state_and_base_from_mode(mode: dict, time_index: int, grid: GridSpec) -> tuple[State, BaseState]:
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}
    z_faces = jnp.asarray(mode["z_w_m"], dtype=jnp.float64)
    pb = jnp.ones((grid.nz, grid.ny, grid.nx), dtype=jnp.float64) * 90_000.0
    phb = jnp.broadcast_to((GRAVITY_M_S2 * z_faces)[:, None, None], (grid.nz + 1, grid.ny, grid.nx))
    mub = jnp.ones((grid.ny, grid.nx), dtype=jnp.float64) * 90_000.0
    theta_base = jnp.ones((grid.nz, grid.ny, grid.nx), dtype=jnp.float64) * THETA_BASE_K
    theta_perturbation = _broadcast_mass(mode["theta_perturbation"][time_index], grid)
    ph_perturbation = _broadcast_face(mode["ph_perturbation"][time_index], grid)

    arrays["w"] = _broadcast_face(mode["w"][time_index], grid)
    arrays["theta"] = theta_base + theta_perturbation
    arrays["p"] = pb
    arrays["p_total"] = pb
    arrays["p_perturbation"] = jnp.zeros_like(pb)
    arrays["ph"] = phb + ph_perturbation
    arrays["ph_total"] = phb + ph_perturbation
    arrays["ph_perturbation"] = ph_perturbation
    arrays["mu"] = mub
    arrays["mu_total"] = mub
    arrays["mu_perturbation"] = jnp.zeros_like(mub)

    state = State(**arrays)
    base = BaseState(pb=pb, phb=phb, mub=mub, t0=theta_base, theta_base=theta_base)
    return state, base


def _advance_current_operator(
    state: State,
    base: BaseState,
    grid: GridSpec,
    dt_s: float,
    *,
    n_substeps: int = 64,
) -> State:
    metrics = flat_metrics_for_grid(grid)
    config = AcousticConfig(
        n_substeps=n_substeps,
        dx_m=grid.projection.dx_m,
        dy_m=grid.projection.dy_m,
        non_hydrostatic=False,
        mu_continuity=False,
    )
    next_state, _ = acoustic_wrf.run_acoustic_scan(
        state,
        state.p_perturbation,
        metrics,
        config,
        float(dt_s),
        base,
    )
    return next_state


def _modal_amplitude(field: jnp.ndarray, mode_profile: np.ndarray) -> float:
    profile = np.asarray(field[:, 0, 0], dtype=np.float64)
    basis = np.asarray(mode_profile, dtype=np.float64)
    return float(np.dot(profile, basis) / np.dot(basis, basis))


def _oracle_mode(times_s: np.ndarray) -> dict:
    return vertical_acoustic_mode(
        n_levels=N_LEVELS,
        column_height_m=COLUMN_HEIGHT_M,
        theta_base_K=THETA_BASE_K,
        brunt_vaisala_N_inv_s=BRUNT_N_INV_S,
        wavelength_m=WAVELENGTH_M,
        initial_amplitude_w_m_s=INITIAL_W_M_S,
        times_s=times_s,
    )


def test_linear_acoustic_period_matches_dispersion_relation():
    grid = _column_grid()
    mode0 = _oracle_mode(np.asarray([0.0], dtype=np.float64))
    period_s = float(mode0["period_s"])
    phase_times = np.asarray([0.0, 0.25 * period_s, 0.5 * period_s, period_s], dtype=np.float64)
    analytic = _oracle_mode(phase_times)
    state, base = _state_and_base_from_mode(analytic, 0, grid)

    quarter = _advance_current_operator(state, base, grid, phase_times[1])
    half = _advance_current_operator(state, base, grid, phase_times[2])
    full = _advance_current_operator(state, base, grid, phase_times[3])

    initial_profile = analytic["w"][0]
    quarter_amp = _modal_amplitude(quarter.w, initial_profile)
    half_amp = _modal_amplitude(half.w, initial_profile)
    full_amp = _modal_amplitude(full.w, initial_profile)
    expected = [
        _modal_amplitude(_broadcast_face(analytic["w"][1], grid), initial_profile),
        _modal_amplitude(_broadcast_face(analytic["w"][2], grid), initial_profile),
        _modal_amplitude(_broadcast_face(analytic["w"][3], grid), initial_profile),
    ]
    tolerance = 0.02 * INITIAL_W_M_S

    assert abs(quarter_amp - expected[0]) < tolerance
    assert abs(half_amp - expected[1]) < tolerance
    assert abs(full_amp - expected[2]) < tolerance


def test_no_drift_in_hydrostatic_rest_state():
    grid = _column_grid()
    rest = _oracle_mode(np.asarray([0.0], dtype=np.float64))
    state, base = _state_and_base_from_mode(rest, 0, grid)
    state = state.replace(
        w=jnp.zeros_like(state.w),
        theta=base.theta_base,
        ph_total=base.phb,
        ph_perturbation=jnp.zeros_like(state.ph_perturbation),
    )
    ph_initial = state.ph_perturbation

    final = _advance_current_operator(state, base, grid, 10.0, n_substeps=1000)

    assert float(jnp.max(jnp.abs(final.w))) < 1.0e-12
    assert float(jnp.max(jnp.abs(final.ph_perturbation - ph_initial))) < 1.0e-12
    assert hasattr(acoustic_wrf, "vertical_acoustic_update"), (
        "no operator-level vertical_acoustic_update is exposed; hydrostatic-rest "
        "invariance cannot yet be certified for the WRF advance_w path"
    )


def test_amplitude_decay_within_2pct_of_analytic():
    grid = _column_grid()
    mode0 = _oracle_mode(np.asarray([0.0], dtype=np.float64))
    half_period_s = 0.5 * float(mode0["period_s"])
    analytic = _oracle_mode(np.asarray([0.0, half_period_s], dtype=np.float64))
    state, base = _state_and_base_from_mode(analytic, 0, grid)

    final = _advance_current_operator(state, base, grid, half_period_s)

    initial_profile = analytic["w"][0]
    actual_signed_amplitude = _modal_amplitude(final.w, initial_profile)
    expected_signed_amplitude = _modal_amplitude(_broadcast_face(analytic["w"][1], grid), initial_profile)
    analytic_decay = np.exp(-float(analytic["decay_rate_inv_s"]) * half_period_s)
    assert analytic_decay == pytest.approx(1.0)
    assert abs(actual_signed_amplitude - expected_signed_amplitude) < 0.02 * INITIAL_W_M_S
