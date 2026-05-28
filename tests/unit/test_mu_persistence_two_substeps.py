from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import OperationalNamelist, _operational_acoustic_substep_core, _with_save_family
from gpuwrf.runtime.operational_state import initial_operational_carry


def _grid(nx: int = 5, ny: int = 5, nz: int = 4) -> GridSpec:
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
    p_base = 90000.0 - 1000.0 * z + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64)
    mu_pert = 750.0 + 2.0 * jnp.arange(grid.ny, dtype=jnp.float64)[:, None] + jnp.zeros((grid.ny, grid.nx))
    mu_base = 85000.0 + jnp.zeros_like(mu_pert)
    fields.update(
        theta=300.0 + 0.01 * z + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
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


def test_mu_save_preserves_nonzero_perturbation_across_two_zero_tendency_substeps():
    grid = _grid()
    state = _state(grid)
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=_zero_tendencies(grid),
        metrics=grid.metrics,
        dt_s=2.0,
        acoustic_substeps=2,
        run_physics=False,
        run_boundary=False,
        use_vertical_solver=True,
        disable_guards=True,
    )
    carry = _with_save_family(initial_operational_carry(state).replace(state=state), state)
    initial_mu_save = carry.mu_save

    carry = _operational_acoustic_substep_core(carry, namelist, 1.0)
    carry = _operational_acoustic_substep_core(carry, namelist, 1.0)
    jax.block_until_ready(carry.mu_save)

    assert float(jnp.max(jnp.abs(carry.mu_save - initial_mu_save))) <= 1.0e-10
    assert float(jnp.max(jnp.abs(carry.state.mu_perturbation - initial_mu_save))) <= 1.0e-10
