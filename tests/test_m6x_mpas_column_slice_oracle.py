from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import NONHYDROSTATIC_BUOYANCY_SCALE, vertical_acoustic_update
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.validation.mpas_oracles import mpas_column_slice


N_LEVELS = 16
COLUMN_HEIGHT_M = 10_000.0
DT_ACOUSTIC_S = 1.0
N_SUBSTEPS = 40
EPS_SM = 0.1
THETA_BASE_K = 300.0
GRAVITY_M_S2 = 9.80665
FIXTURE_PATH = Path("data/fixtures/mpas_column_slice/warm_bubble_2km.npz")
REQUIRED_KEYS = {
    "t",
    "w",
    "theta_perturbation",
    "ph_perturbation",
    "mu_perturbation",
    "rho_perturbation",
}


def _assert_contract_shapes(result: dict, n_levels: int, n_substeps: int) -> None:
    assert set(result) == REQUIRED_KEYS
    assert result["t"].shape == (n_substeps + 1,)
    assert result["w"].shape == (n_substeps + 1, n_levels + 1)
    assert result["theta_perturbation"].shape == (n_substeps + 1, n_levels)
    assert result["ph_perturbation"].shape == (n_substeps + 1, n_levels + 1)
    assert result["mu_perturbation"].shape == (n_substeps + 1,)
    assert result["rho_perturbation"].shape == (n_substeps + 1, n_levels)
    for value in result.values():
        assert np.all(np.isfinite(value))


def _column_grid() -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, 1000.0, 1000.0, 1, 1)
    terrain = TerrainProvenance(
        source_path="analytic://m6x-mpas-column-slice",
        sha256="analytic-mpas-column-slice",
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
    return GridSpec(projection, terrain, vertical, bc, eta_levels, jnp.zeros((1, 1), dtype=jnp.float64))


def _state_and_base_from_slice_initial(slice_result: dict, grid: GridSpec) -> tuple[State, BaseState]:
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}
    z_faces = jnp.linspace(0.0, COLUMN_HEIGHT_M, N_LEVELS + 1, dtype=jnp.float64)
    theta_base = jnp.ones((grid.nz, grid.ny, grid.nx), dtype=jnp.float64) * THETA_BASE_K
    pb = jnp.ones((grid.nz, grid.ny, grid.nx), dtype=jnp.float64) * 90_000.0
    phb = jnp.broadcast_to((GRAVITY_M_S2 * z_faces)[:, None, None], (grid.nz + 1, grid.ny, grid.nx))
    mub = jnp.ones((grid.ny, grid.nx), dtype=jnp.float64) * 90_000.0
    theta_perturbation = jnp.asarray(slice_result["theta_perturbation"][0], dtype=jnp.float64)[:, None, None]
    ph_perturbation = jnp.asarray(slice_result["ph_perturbation"][0], dtype=jnp.float64)[:, None, None]

    arrays["w"] = jnp.asarray(slice_result["w"][0], dtype=jnp.float64)[:, None, None]
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
    return State(**arrays), BaseState(pb=pb, phb=phb, mub=mub, t0=theta_base, theta_base=theta_base)


def _c2_vertical_trajectory(slice_result: dict) -> np.ndarray:
    grid = _column_grid()
    metrics = flat_metrics_for_grid(grid)
    state, base = _state_and_base_from_slice_initial(slice_result, grid)
    trajectory = [np.asarray(state.w[:, 0, 0], dtype=np.float64)]
    for _ in range(N_SUBSTEPS):
        state = vertical_acoustic_update(
            state,
            base,
            metrics,
            dt=DT_ACOUSTIC_S,
            epssm=EPS_SM,
            top_lid=True,
            pressure_scale=0.0,
            buoyancy_scale=NONHYDROSTATIC_BUOYANCY_SCALE,
        )
        trajectory.append(np.asarray(state.w[:, 0, 0], dtype=np.float64))
    return np.asarray(trajectory, dtype=np.float64)


def test_slice_runs_warm_bubble_scenario():
    result = mpas_column_slice(
        scenario="warm_bubble_2km",
        n_levels=N_LEVELS,
        column_height_m=COLUMN_HEIGHT_M,
        dt_acoustic_s=DT_ACOUSTIC_S,
        n_substeps=N_SUBSTEPS,
        epssm=EPS_SM,
    )

    _assert_contract_shapes(result, N_LEVELS, N_SUBSTEPS)
    assert float(np.max(result["w"])) > 1.0


def test_slice_runs_stratified_rest_scenario():
    result = mpas_column_slice(
        scenario="stratified_rest",
        n_levels=N_LEVELS,
        column_height_m=COLUMN_HEIGHT_M,
        dt_acoustic_s=DT_ACOUSTIC_S,
        n_substeps=N_SUBSTEPS,
        epssm=EPS_SM,
    )

    _assert_contract_shapes(result, N_LEVELS, N_SUBSTEPS)
    assert float(np.max(np.abs(result["w"]))) < 1.0e-10
    assert float(np.max(np.abs(result["theta_perturbation"]))) < 1.0e-10
    assert float(np.max(np.abs(result["rho_perturbation"]))) < 1.0e-10


def test_adr023_operator_matches_slice_within_tolerance():
    slice_result = mpas_column_slice(
        scenario="warm_bubble_2km",
        n_levels=N_LEVELS,
        column_height_m=COLUMN_HEIGHT_M,
        dt_acoustic_s=DT_ACOUSTIC_S,
        n_substeps=N_SUBSTEPS,
        epssm=EPS_SM,
    )
    c2_w = _c2_vertical_trajectory(slice_result)

    slice_peak = float(np.max(np.abs(slice_result["w"])))
    c2_peak = float(np.max(np.abs(c2_w)))
    peak_amplitude_error = abs(c2_peak - slice_peak) / slice_peak
    trajectory_rmse = float(np.sqrt(np.mean((c2_w - slice_result["w"]) ** 2)) / slice_peak)

    assert c2_peak > 1.0
    assert peak_amplitude_error <= 0.20
    assert trajectory_rmse < 0.80


def test_warm_bubble_fixture_replays_generated_slice():
    assert FIXTURE_PATH.exists()
    generated = mpas_column_slice(
        scenario="warm_bubble_2km",
        n_levels=N_LEVELS,
        column_height_m=COLUMN_HEIGHT_M,
        dt_acoustic_s=DT_ACOUSTIC_S,
        n_substeps=N_SUBSTEPS,
        epssm=EPS_SM,
    )

    with np.load(FIXTURE_PATH) as fixture:
        assert set(fixture.files) == REQUIRED_KEYS
        for key in REQUIRED_KEYS:
            assert np.allclose(fixture[key], generated[key], rtol=0.0, atol=0.0)
