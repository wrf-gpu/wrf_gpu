from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.coupling.physics_couplers import (
    mynn_adapter_with_source_leaves,
    surface_adapter,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist, _physics_step_forcing
from gpuwrf.runtime.operational_state import initial_operational_carry

jax.config.update("jax_enable_x64", True)


def _grid(ny: int = 3, nx: int = 3, nz: int = 8) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="v014-dry-source-leaf-test",
        sha256="v014-dry-source-leaf-test",
        shape=(ny, nx),
        units="m",
        projection_transform="native-wrf-lambert",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny,
        nx=nx,
        nz=nz,
        eta_levels=eta,
        top_pressure_pa=5000.0,
        provenance="v014-dry-source-leaf-test-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, terrain_height, metrics=metrics)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    return Tendencies(
        z((nz, ny, nx + 1)),
        z((nz, ny + 1, nx)),
        z((nz + 1, ny, nx)),
        z((nz, ny, nx)),
        z((nz, ny, nx)),
        z((nz, ny, nx)),
        z((nz + 1, ny, nx)),
        z((ny, nx)),
    )


def _state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {
        name: jnp.zeros(shape, dtype=jnp.float64)
        for name, shape in _state_field_shapes(grid).items()
    }
    p = jnp.linspace(95000.0, 25000.0, nz, dtype=jnp.float64)[:, None, None]
    p = jnp.broadcast_to(p, (nz, ny, nx))
    ph = jnp.linspace(0.0, 12000.0 * 9.80665, nz + 1, dtype=jnp.float64)[:, None, None]
    ph = jnp.broadcast_to(ph, (nz + 1, ny, nx))
    fields.update(
        u=jnp.full((nz, ny, nx + 1), 6.0, dtype=jnp.float64),
        v=jnp.full((nz, ny + 1, nx), -2.0, dtype=jnp.float64),
        theta=jnp.linspace(292.0, 310.0, nz, dtype=jnp.float64)[:, None, None]
        + jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        qv=jnp.full((nz, ny, nx), 8.0e-3, dtype=jnp.float64),
        p=p,
        p_total=p,
        p_perturbation=p,
        ph=ph,
        ph_total=ph,
        ph_perturbation=ph,
        mu=jnp.full((ny, nx), 90000.0, dtype=jnp.float64),
        mu_total=jnp.full((ny, nx), 90000.0, dtype=jnp.float64),
        mu_perturbation=jnp.full((ny, nx), 90000.0, dtype=jnp.float64),
        qke=jnp.full((nz, ny, nx), 0.2, dtype=jnp.float64),
        t_skin=jnp.full((ny, nx), 302.0, dtype=jnp.float64),
        xland=jnp.full((ny, nx), 1.0, dtype=jnp.float64),
        mavail=jnp.full((ny, nx), 0.7, dtype=jnp.float64),
        roughness_m=jnp.full((ny, nx), 0.1, dtype=jnp.float64),
        lu_index=jnp.ones((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def test_source_leaf_mode_mass_couples_held_rthraten_and_mynn_rthblten() -> None:
    grid = _grid()
    state = _state(grid)
    base = OperationalNamelist.from_grid(
        grid,
        dt_s=10.0,
        tendencies=_cpu_tendencies(grid),
        metrics=grid.metrics,
        force_fp64=True,
    )
    namelist = dataclasses.replace(
        base,
        mp_physics=0,
        bl_pbl_physics=5,
        sf_sfclay_physics=5,
        cu_physics=0,
        ra_sw_physics=0,
        ra_lw_physics=0,
        rad_rk_tendf=1,
    )
    held_rthraten = jnp.full_like(state.theta, 2.5e-4)
    carry = initial_operational_carry(state).replace(rthraten=held_rthraten)

    surface_state = surface_adapter(state, float(namelist.dt_s))
    mynn = mynn_adapter_with_source_leaves(surface_state, float(namelist.dt_s), grid)
    mass_h = (
        namelist.metrics.c1h[:, None, None] * mynn.state.mu_total[None, :, :]
        + namelist.metrics.c2h[:, None, None]
    )
    expected = mass_h * (held_rthraten + mynn.rthblten)

    forcing = _physics_step_forcing(carry, namelist, 0.0, run_radiation=False)

    np.testing.assert_allclose(np.asarray(forcing.dry_tendencies.t_tendf), np.asarray(expected))
    assert float(jnp.max(jnp.abs(mynn.rthblten))) > 0.0
    np.testing.assert_allclose(np.asarray(forcing.state.theta), np.asarray(surface_state.theta))
