from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics import acoustic_wrf
from gpuwrf.dynamics.acoustic_wrf import (
    AcousticConfig,
    AcousticScanCarry,
    MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    POST_SOLVE_REPLACEMENT_ORDER,
    acoustic_substep_carry,
    initialize_acoustic_carry,
)
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.validation.mpas_oracles import mpas_column_slice


N_LEVELS = 16
COLUMN_HEIGHT_M = 10_000.0
DT_ACOUSTIC_S = 1.0
N_SUBSTEPS = 40
THETA_BASE_K = 300.0
GRAVITY_M_S2 = 9.80665


def _column_grid(nz: int = N_LEVELS) -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, 1000.0, 1000.0, 1, 1)
    terrain = TerrainProvenance(
        source_path="analytic://m6x-adr023-production-grade",
        sha256="analytic-m6x-adr023-production-grade",
        shape=(1, 1),
        units="m",
        projection_transform="flat-column",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta_levels = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta_levels)
    bc = BCMetadata("ideal", ("w", "ph", "theta"), 1, "linear", True)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, jnp.zeros((1, 1), dtype=jnp.float64))


def _state_and_base_from_slice_initial(slice_result: dict, grid: GridSpec) -> tuple[State, BaseState]:
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}
    z_faces = jnp.linspace(0.0, COLUMN_HEIGHT_M, grid.nz + 1, dtype=jnp.float64)
    theta_base = jnp.ones((grid.nz, grid.ny, grid.nx), dtype=jnp.float64) * THETA_BASE_K
    pb = jnp.ones((grid.nz, grid.ny, grid.nx), dtype=jnp.float64) * 90_000.0
    phb = jnp.broadcast_to((GRAVITY_M_S2 * z_faces)[:, None, None], (grid.nz + 1, grid.ny, grid.nx))
    mub = jnp.ones((grid.ny, grid.nx), dtype=jnp.float64) * 90_000.0
    arrays["w"] = jnp.asarray(slice_result["w"][0], dtype=jnp.float64)[:, None, None]
    arrays["theta"] = theta_base + jnp.asarray(slice_result["theta_perturbation"][0], dtype=jnp.float64)[:, None, None]
    arrays["p"] = pb
    arrays["p_total"] = pb
    arrays["p_perturbation"] = jnp.zeros_like(pb)
    arrays["ph"] = phb + jnp.asarray(slice_result["ph_perturbation"][0], dtype=jnp.float64)[:, None, None]
    arrays["ph_total"] = arrays["ph"]
    arrays["ph_perturbation"] = jnp.asarray(slice_result["ph_perturbation"][0], dtype=jnp.float64)[:, None, None]
    arrays["mu"] = mub
    arrays["mu_total"] = mub
    arrays["mu_perturbation"] = jnp.zeros_like(mub)
    return State(**arrays), BaseState(pb=pb, phb=phb, mub=mub, t0=theta_base, theta_base=theta_base)


def _c2_mpas_trajectory(epssm: float) -> tuple[np.ndarray, dict]:
    slice_result = mpas_column_slice(
        scenario="warm_bubble_2km",
        n_levels=N_LEVELS,
        column_height_m=COLUMN_HEIGHT_M,
        dt_acoustic_s=DT_ACOUSTIC_S,
        n_substeps=N_SUBSTEPS,
        epssm=epssm,
    )
    grid = _column_grid()
    metrics = flat_metrics_for_grid(grid)
    state, base = _state_and_base_from_slice_initial(slice_result, grid)
    config = AcousticConfig(n_substeps=1, non_hydrostatic=True, mu_continuity=True, epssm=epssm)
    carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, config)
    trajectory = [np.asarray(state.w[:, 0, 0], dtype=np.float64)]
    for _ in range(N_SUBSTEPS):
        carry = acoustic_substep_carry(
            carry,
            metrics,
            config,
            DT_ACOUSTIC_S,
            base,
        )
        trajectory.append(np.asarray(carry.state.w[:, 0, 0], dtype=np.float64))
    return np.asarray(trajectory, dtype=np.float64), slice_result


def _trajectory_rmse_fraction(c2_w: np.ndarray, slice_result: dict) -> float:
    slice_peak = float(np.max(np.abs(slice_result["w"])))
    return float(np.sqrt(np.mean((c2_w - slice_result["w"]) ** 2)) / slice_peak)


def test_mpas_slice_trajectory_rmse_under_production_target():
    c2_w, slice_result = _c2_mpas_trajectory(0.1)

    assert _trajectory_rmse_fraction(c2_w, slice_result) < 0.15


def test_epssm_sweep_keeps_mpas_slice_rung_below_target():
    rmse = {}
    for epssm in (0.0, 0.1, 0.3):
        c2_w, slice_result = _c2_mpas_trajectory(epssm)
        rmse[epssm] = _trajectory_rmse_fraction(c2_w, slice_result)

    assert rmse[0.0] < 0.15
    assert rmse[0.1] < 0.15
    assert rmse[0.3] < 0.15
    assert AcousticConfig().epssm == 0.1


def test_nonhydrostatic_mu_continuity_runs_inside_scan_body():
    grid = _column_grid(nz=6)
    metrics = flat_metrics_for_grid(grid)
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}
    theta_base = jnp.ones((grid.nz, 1, 1), dtype=jnp.float64) * THETA_BASE_K
    pb = jnp.ones_like(theta_base) * 90_000.0
    phb = jnp.linspace(0.0, 6000.0 * GRAVITY_M_S2, grid.nz + 1, dtype=jnp.float64)[:, None, None]
    mub = jnp.ones((1, 1), dtype=jnp.float64) * 90_000.0
    arrays["theta"] = theta_base
    arrays["p"] = pb
    arrays["p_total"] = pb
    arrays["p_perturbation"] = jnp.zeros_like(pb)
    arrays["ph"] = phb
    arrays["ph_total"] = phb
    arrays["ph_perturbation"] = jnp.zeros_like(phb)
    arrays["mu"] = mub
    arrays["mu_total"] = mub
    arrays["mu_perturbation"] = jnp.zeros_like(mub)
    arrays["u"] = arrays["u"].at[:, :, 1].set(2.0)
    state = State(**arrays)
    base = BaseState(pb=pb, phb=phb, mub=mub, t0=theta_base, theta_base=theta_base)
    config = AcousticConfig(n_substeps=1, non_hydrostatic=True, mu_continuity=True)
    carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, config)

    updated = acoustic_substep_carry(carry, metrics, config, 0.25, base)

    assert float(jnp.max(jnp.abs(updated.state.mu_perturbation))) > 0.0


def test_post_solve_order_and_carry_boundaries_are_explicit():
    assert POST_SOLVE_REPLACEMENT_ORDER == (
        "w",
        "theta",
        "ph_perturbation",
        "mu_perturbation",
        "p_perturbation",
        "al",
        "alt",
    )
    assert AcousticScanCarry.__slots__ == ("state", "previous_pressure", "al", "alt", "cqu", "cqv")
    assert MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE == 0.38
    assert "Post-solve replacement order" in (acoustic_wrf.vertical_acoustic_update.__doc__ or "")
    assert "Per-substep locals" in (AcousticScanCarry.__doc__ or "")
