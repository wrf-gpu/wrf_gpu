from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, Tendencies
from gpuwrf.coupling.boundary_apply import BoundaryConfig
from gpuwrf.integration.d02_replay import ReplayConfig, run_replay_scan
from gpuwrf.profiling.transfer_audit import block_until_ready


GRAVITY_M_S2 = 9.80665
R_DRY_AIR = 287.0
P0_PA = 100_000.0
T0_K = 300.0


def _has_gpu() -> bool:
    return any(device.platform == "gpu" for device in jax.devices())


def _pressure_at_height(z_m: np.ndarray) -> np.ndarray:
    return P0_PA * np.exp(-z_m / 8400.0)


def _theta_from_pressure(pressure_pa: np.ndarray) -> np.ndarray:
    return T0_K * (P0_PA / pressure_pa) ** (R_DRY_AIR / 1004.0)


def _grid(nx: int = 6, ny: int = 6, nz: int = 4, dx_m: float = 1000.0, dz_m: float = 500.0) -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, dx_m, dx_m, nx, ny)
    terrain = TerrainProvenance(
        source_path="synthetic://m6x-d02-replay-hang-debug",
        sha256="synthetic-m6x-d02-replay-hang-debug",
        shape=(ny, nx),
        units="m",
        projection_transform="cartesian",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", nz, float(_pressure_at_height(np.asarray([nz * dz_m]))[0]), eta)
    bc = BCMetadata("ideal", ("u", "v", "w", "theta", "qv", "p", "pb", "ph", "mu"), 0, "linear", False)
    return GridSpec(projection, terrain, vertical, bc, eta, jnp.zeros((ny, nx), dtype=jnp.float64))


def _boundary_3d(field: np.ndarray, boundary_side: int) -> jax.Array:
    values = np.zeros((1, 4, field.shape[0], boundary_side), dtype=field.dtype)
    values[0, 0, :, : field.shape[1]] = field[:, :, 0]
    values[0, 1, :, : field.shape[1]] = field[:, :, -1]
    values[0, 2, :, : field.shape[2]] = field[:, 0, :]
    values[0, 3, :, : field.shape[2]] = field[:, -1, :]
    return jnp.asarray(values)


def _boundary_mu(mu: np.ndarray, boundary_side: int) -> jax.Array:
    values = np.zeros((1, 4, 1, boundary_side), dtype=mu.dtype)
    values[0, 0, 0, : mu.shape[0]] = mu[:, 0]
    values[0, 1, 0, : mu.shape[0]] = mu[:, -1]
    values[0, 2, 0, : mu.shape[1]] = mu[0, :]
    values[0, 3, 0, : mu.shape[1]] = mu[-1, :]
    return jnp.asarray(values)


def _synthetic_case() -> tuple[State, BaseState, Tendencies, GridSpec]:
    grid = _grid()
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    z_face_1d = np.arange(nz + 1, dtype=np.float64) * 500.0
    z_mass_1d = 0.5 * (z_face_1d[:-1] + z_face_1d[1:])
    z_face = np.broadcast_to(z_face_1d[:, None, None], (nz + 1, ny, nx))
    pb = np.broadcast_to(_pressure_at_height(z_mass_1d)[:, None, None], (nz, ny, nx)).copy()
    phb = (GRAVITY_M_S2 * z_face).copy()
    mub = np.full((ny, nx), P0_PA - float(grid.vertical.top_pressure_pa), dtype=np.float64)
    theta_base = np.broadcast_to(_theta_from_pressure(_pressure_at_height(z_mass_1d))[:, None, None], (nz, ny, nx)).copy()
    theta = theta_base.copy()
    qv = np.full((nz, ny, nx), 1.0e-3, dtype=np.float64)
    u = np.zeros((nz, ny, nx + 1), dtype=np.float64)
    v = np.zeros((nz, ny + 1, nx), dtype=np.float64)
    boundary_side = max(nx + 1, ny + 1)

    state = State.zeros(grid).replace(
        u=jnp.asarray(u),
        v=jnp.asarray(v),
        theta=jnp.asarray(theta),
        qv=jnp.asarray(qv),
        p_total=jnp.asarray(pb),
        p_perturbation=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        ph_total=jnp.asarray(phb),
        ph_perturbation=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        mu_total=jnp.asarray(mub),
        mu_perturbation=jnp.zeros((ny, nx), dtype=jnp.float64),
        t_skin=jnp.full((ny, nx), T0_K, dtype=jnp.float64),
        xland=jnp.ones((ny, nx), dtype=jnp.float32),
        mavail=jnp.ones((ny, nx), dtype=jnp.float32),
        roughness_m=jnp.full((ny, nx), 0.1, dtype=jnp.float64),
        rhosfc=jnp.full((ny, nx), P0_PA / (R_DRY_AIR * T0_K), dtype=jnp.float64),
        u_bdy=_boundary_3d(u, boundary_side),
        v_bdy=_boundary_3d(v, boundary_side),
        theta_bdy=_boundary_3d(theta, boundary_side),
        qv_bdy=_boundary_3d(qv, boundary_side),
        ph_bdy=_boundary_3d(phb, boundary_side),
        mu_bdy=_boundary_mu(mub, boundary_side),
    )
    base = BaseState(
        pb=jnp.asarray(pb),
        phb=jnp.asarray(phb),
        mub=jnp.asarray(mub),
        t0=jnp.asarray(theta_base),
        theta_base=jnp.asarray(theta_base),
    )
    return state, base, Tendencies.zeros(grid), grid


def test_synthetic_replay_scan_returns_within_60_seconds():
    if not _has_gpu():
        pytest.skip("d02 replay hang smoke requires a visible JAX GPU")

    state, base, tendencies, grid = _synthetic_case()
    config = ReplayConfig(
        duration_s=0.25,
        dt_s=0.25,
        n_acoustic=1,
        radiation_cadence_steps=999,
        final_radiation=False,
        boundary_config=BoundaryConfig(spec_bdy_width=1, spec_zone=1, relax_zone=1, update_cadence_s=3600.0),
    )

    start = time.perf_counter()
    result = run_replay_scan(state, state.p_perturbation, tendencies, grid, grid.metrics, base, config)
    block_until_ready(result)
    elapsed = time.perf_counter() - start

    final_state, _previous_pressure, diagnostics = result
    assert elapsed < 60.0
    assert bool(jnp.all(jnp.isfinite(final_state.theta)))
    assert int(diagnostics.finite_after_sanitize.shape[0]) == 1
