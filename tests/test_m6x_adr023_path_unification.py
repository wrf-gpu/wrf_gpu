from __future__ import annotations

import hashlib
import inspect

import numpy as np
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics import acoustic_wrf
from gpuwrf.dynamics.acoustic_wrf import (
    AcousticConfig,
    MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    acoustic_substep_carry,
    initialize_acoustic_carry,
    vertical_acoustic_update,
)
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.validation.mpas_oracles import mpas_column_slice


N_LEVELS = 16
COLUMN_HEIGHT_M = 10_000.0
DT_ACOUSTIC_S = 1.0
THETA_BASE_K = 300.0
GRAVITY_M_S2 = 9.80665


def _column_grid(nz: int = N_LEVELS) -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, 1000.0, 1000.0, 1, 1)
    terrain = TerrainProvenance(
        source_path="analytic://m6x-adr023-path-unification",
        sha256="analytic-m6x-adr023-path-unification",
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


def _initial_column_state(epssm: float = 0.1) -> tuple[State, BaseState, object]:
    slice_result = mpas_column_slice(
        scenario="warm_bubble_2km",
        n_levels=N_LEVELS,
        column_height_m=COLUMN_HEIGHT_M,
        dt_acoustic_s=DT_ACOUSTIC_S,
        n_substeps=1,
        epssm=epssm,
    )
    grid = _column_grid()
    state, base = _state_and_base_from_slice_initial(slice_result, grid)
    return state, base, flat_metrics_for_grid(grid)


def _public_scan_one_step(epssm: float) -> State:
    state, base, metrics = _initial_column_state(epssm)
    config = AcousticConfig(n_substeps=1, non_hydrostatic=True, mu_continuity=True, epssm=epssm)
    carry = initialize_acoustic_carry(state, state.p_perturbation, metrics, base, config)
    return acoustic_substep_carry(carry, metrics, config, DT_ACOUSTIC_S, base).state


def _state_vector(state: State) -> np.ndarray:
    parts = (
        np.asarray(state.w).ravel(),
        np.asarray(state.theta).ravel(),
        np.asarray(state.ph_perturbation).ravel(),
        np.asarray(state.p_perturbation).ravel(),
    )
    return np.concatenate(parts)


def test_no_separate_wrf_buoyancy_column_update_is_exported():
    source = inspect.getsource(acoustic_wrf.vertical_acoustic_update)

    assert not hasattr(acoustic_wrf, "_wrf_buoyancy_column_update")
    assert not hasattr(acoustic_wrf, "NONHYDROSTATIC_BUOYANCY_SCALE")
    assert "NONHYDROSTATIC_UPDRAFT_DRAG" not in inspect.getsource(acoustic_wrf)
    assert "_mpas_recurrence_vertical_update" in source
    assert "float(pressure_scale) <= 0.0" in source


def test_pressure_scale_zero_and_public_negative_path_are_identical():
    state, base, metrics = _initial_column_state(0.1)
    direct = vertical_acoustic_update(
        state,
        base,
        metrics,
        dt=DT_ACOUSTIC_S,
        epssm=0.1,
        top_lid=True,
        pressure_scale=0.0,
        buoyancy_scale=MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    )
    public = vertical_acoustic_update(
        state,
        base,
        metrics,
        dt=DT_ACOUSTIC_S,
        epssm=0.1,
        top_lid=True,
        pressure_scale=-1.0,
        buoyancy_scale=MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    )

    assert np.allclose(_state_vector(direct), _state_vector(public), rtol=0.0, atol=0.0)


def test_public_scan_epssm_changes_the_nonhydrostatic_update():
    outputs = {epssm: _state_vector(_public_scan_one_step(epssm)) for epssm in (0.0, 0.1, 0.3)}

    assert np.max(np.abs(outputs[0.0] - outputs[0.1])) > 1.0e-6
    assert np.max(np.abs(outputs[0.1] - outputs[0.3])) > 1.0e-6


def test_direct_and_public_recurrence_source_hash_is_shared():
    source = inspect.getsource(acoustic_wrf._mpas_recurrence_vertical_update)
    kernel_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    state, base, metrics = _initial_column_state(0.1)
    direct = vertical_acoustic_update(
        state,
        base,
        metrics,
        dt=DT_ACOUSTIC_S,
        epssm=0.1,
        top_lid=True,
        pressure_scale=0.0,
        buoyancy_scale=MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE,
    )
    public = _public_scan_one_step(0.1)

    assert kernel_hash == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert np.max(np.abs(np.asarray(direct.w - public.w))) < 1.0e-9
