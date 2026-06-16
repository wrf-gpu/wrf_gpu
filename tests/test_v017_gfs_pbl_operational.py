"""v0.17 RC GFS PBL (bl_pbl_physics=3) operational wiring tests.

The RC scope is deliberately narrow: GFS is operational and scan-wired; the
other v0.17 PBL scaffolds from the source branch stay out of this release.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.physics.bl_gfs import gfs_columns
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _physics_step_forcing,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

TIME_UTC = "2024-06-01T12:00:00Z"


def test_gfs_columns_is_jit_traceable_and_finite() -> None:
    n = 16
    z_mid = np.linspace(20.0, 8000.0, n)
    dz = np.full(n, 500.0)
    arr = lambda v: jnp.asarray(np.stack([v, v]), jnp.float64)  # noqa: E731
    sc = lambda v: jnp.asarray([v, v], jnp.float64)  # noqa: E731
    u = np.full(n, 5.0)
    v = np.full(n, 2.0)
    t = np.linspace(295.0, 250.0, n)
    qv = np.full(n, 5.0e-3)
    p = np.linspace(95000.0, 30000.0, n)
    pii = (p / 1.0e5) ** (287.0 / 1004.5)

    out = jax.jit(gfs_columns)(
        arr(u),
        arr(v),
        arr(t),
        arr(qv),
        jnp.zeros((2, n), jnp.float64),
        arr(p),
        arr(pii),
        arr(dz),
        arr(z_mid),
        psfc=sc(100000.0),
        ust=sc(0.4),
        hfx=sc(200.0),
        qfx=sc(1.0e-4),
        tsk=sc(300.0),
        gz1oz0=sc(np.log(20.0 / 0.1)),
        psim=sc(0.0),
        psih=sc(0.0),
        wspd=sc(np.hypot(5.0, 2.0)),
        br=sc(-0.05),
        dt=60.0,
    )

    assert out["rthblten"].shape == (2, n)
    for key in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten"):
        assert bool(np.all(np.isfinite(np.asarray(out[key]))))


def _grid(ny: int = 3, nx: int = 3, nz: int = 8) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="gfs-wire-test",
        sha256="gfs-wire-test",
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
        provenance="gfs-wire-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in _state_field_shapes(grid).items()}
    p = jnp.broadcast_to(jnp.linspace(95000.0, 20000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 12000.0 * 9.80665, nz + 1)[:, None, None], (nz + 1, ny, nx))
    fields.update(
        theta=jnp.full((nz, ny, nx), 295.0),
        p=p,
        ph=ph,
        mu=jnp.full((ny, nx), 90000.0),
        qv=jnp.full((nz, ny, nx), 5.0e-3),
        qc=jnp.full((nz, ny, nx), 1.0e-4),
        qke=jnp.full((nz, ny, nx), 0.5),
        u=jnp.full((nz, ny, nx + 1), 5.0),
        v=jnp.full((nz, ny + 1, nx), 2.0),
        t_skin=jnp.full((ny, nx), 298.0),
        xland=jnp.full((ny, nx), 1.0),
        mavail=jnp.full((ny, nx), 0.5),
        roughness_m=jnp.full((ny, nx), 0.1),
        ustar=jnp.full((ny, nx), 0.3),
        lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


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


def _namelist(grid: GridSpec, **overrides) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(grid, dt_s=10.0, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(base, time_utc=TIME_UTC, run_physics=True, **overrides)


def test_gfs_resolves_in_operational_suite() -> None:
    grid = _grid()
    nml = _namelist(grid, bl_pbl_physics=3, sf_sfclay_physics=1, use_noahmp=False)

    suite = _resolve_operational_suite(nml)

    assert suite.pbl.option == 3
    assert suite.pbl.gpu_runnable is True
    assert suite.surface_layer.option == 1


def test_operational_step_routes_gfs_and_changes_state() -> None:
    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, bl_pbl_physics=3, sf_sfclay_physics=1, use_noahmp=False)
    carry = initial_operational_carry(state)

    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state

    assert np.all(np.isfinite(np.asarray(after.theta)))
    assert np.all(np.isfinite(np.asarray(after.u)))
    assert not np.allclose(np.asarray(after.theta), np.asarray(state.theta))
    assert not np.allclose(np.asarray(after.u), np.asarray(state.u))
