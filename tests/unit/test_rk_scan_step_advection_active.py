from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import OperationalNamelist, _rk_scan_step
from gpuwrf.runtime.operational_state import initial_operational_carry


def _grid(nx: int = 8, ny: int = 6, nz: int = 5) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    return GridSpec(
        projection=Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, nx, ny),
        terrain=TerrainProvenance("analytic://unit", "unit", (ny, nx), "m", "native", 0.0, True),
        vertical=VerticalCoord("hybrid_eta", nz, 16000.0, eta),
        bc=BCMetadata("ideal", ("u", "v", "theta"), 0, "linear", False),
        eta_levels=eta,
        terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
    )


def _state(grid: GridSpec) -> State:
    fields = {
        name: jnp.zeros(shape, dtype=jnp.int32 if name == "lu_index" else jnp.float64)
        for name, shape in _state_field_shapes(grid).items()
    }
    z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    theta = 300.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x / float(grid.nx)) + 0.05 * y + 0.01 * z
    p_base = 90000.0 - 1000.0 * z + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64)
    mu_pert = 500.0 + jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64)
    mu_base = 85000.0 + jnp.zeros_like(mu_pert)
    fields.update(
        u=3.0 + jnp.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=jnp.float64),
        v=1.0 + jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=jnp.float64),
        w=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        theta=theta,
        qv=0.004 + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        p=p_base,
        p_total=p_base,
        p_perturbation=jnp.zeros_like(p_base),
        ph=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        ph_total=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        ph_perturbation=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        mu=mu_base + mu_pert,
        mu_total=mu_base + mu_pert,
        mu_perturbation=mu_pert,
    )
    return State(**fields)


def _zero_tendencies(grid: GridSpec) -> Tendencies:
    return Tendencies(
        u=jnp.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=jnp.float64),
        v=jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=jnp.float64),
        w=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        theta=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        qv=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        p=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        ph=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        mu=jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
    )


def test_rk_scan_step_advection_changes_theta_when_gradient_and_wind_are_nonzero():
    grid = _grid()
    state = _state(grid)
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=_zero_tendencies(grid),
        metrics=grid.metrics,
        dt_s=1.0,
        acoustic_substeps=1,
        run_physics=False,
        run_boundary=False,
        use_vertical_solver=False,
        disable_guards=True,
    )

    out = _rk_scan_step(initial_operational_carry(state), namelist).state
    jax.block_until_ready(out.theta)

    assert float(jnp.max(jnp.abs(out.theta - state.theta))) > 1.0e-8
