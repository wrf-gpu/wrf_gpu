"""v0.22 CAM-UW PBL operational wiring.

The CAM-UW kernel is proof-limited to an idealized/source-present gate until a
full pristine-WRF CAM savepoint exists. These tests cover the landed behavior:
traceable finite column output and scan routing for ``bl_pbl_physics=9``.
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
from gpuwrf.physics.bl_camuw import camuw_columns
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _physics_step_forcing,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

TIME_UTC = "2024-06-01T12:00:00Z"


def test_camuw_columns_is_jit_traceable_finite_and_plausible() -> None:
    n = 14
    z_mid = np.linspace(35.0, 7000.0, n)
    dz = np.full(n, 500.0)
    p = np.linspace(95500.0, 36000.0, n)
    pii = (p / 1.0e5) ** (287.0 / 1004.0)
    t = np.linspace(296.0, 254.0, n)
    theta = t / pii
    qv = np.maximum(0.011 * np.exp(-z_mid / 2300.0), 3.0e-5)
    qc = np.zeros(n)
    qi = np.zeros(n)
    qc[2:5] = 4.0e-5
    u = np.linspace(7.0, 17.0, n)
    v = np.linspace(1.0, 5.0, n)

    A = lambda x: jnp.asarray(np.stack([x, x]), jnp.float64)
    sc = lambda x: jnp.asarray([x, x], jnp.float64)
    out = jax.jit(camuw_columns)(
        A(u),
        A(v),
        A(t),
        A(theta),
        A(qv),
        A(qc),
        A(qi),
        A(p),
        A(pii),
        A(dz),
        A(z_mid),
        jnp.full((2, n), 0.03, jnp.float64),
        hfx=sc(180.0),
        qfx=sc(1.2e-4),
        ust=sc(0.42),
        wspd=sc(float(np.hypot(u[0], v[0]))),
        dt=60.0,
    )
    for key in ("u", "v", "theta", "qv", "qc", "qi", "tke", "kvh", "kvm", "pblh", "smaw"):
        assert bool(np.all(np.isfinite(np.asarray(out[key])))), key
    assert out["theta"].shape == (2, n)
    assert bool(np.all(np.asarray(out["pblh"]) > 0.0))
    assert bool(np.all(np.asarray(out["pblh"]) < z_mid[-1]))
    assert float(np.max(np.asarray(out["kvm"]))) > 0.05
    assert float(np.max(np.asarray(out["tke"]))) > 0.03
    assert float(np.asarray(out["u"])[0, 0]) < 0.0
    assert float(np.asarray(out["theta"])[0, 0]) > 0.0


def _grid(ny: int = 3, nx: int = 3, nz: int = 8) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="camuw-wire-test",
        sha256="camuw-wire-test",
        shape=(ny, nx),
        units="m",
        projection_transform="native-wrf-lambert",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="camuw-wire-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    p = jnp.broadcast_to(jnp.linspace(95500.0, 30000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 10000.0 * 9.80665, nz + 1)[:, None, None], (nz + 1, ny, nx))
    qv_profile = jnp.maximum(0.011 * jnp.exp(-jnp.linspace(0.0, 7000.0, nz) / 2500.0), 5.0e-5)
    fields.update(
        theta=jnp.broadcast_to(jnp.linspace(296.0, 315.0, nz)[:, None, None], (nz, ny, nx)),
        p_total=p,
        ph_total=ph,
        mu_total=jnp.full((ny, nx), 90000.0),
        qv=jnp.broadcast_to(qv_profile[:, None, None], (nz, ny, nx)),
        qc=jnp.full((nz, ny, nx), 2.0e-5),
        qi=jnp.zeros((nz, ny, nx)),
        qke=jnp.full((nz, ny, nx), 0.03),
        u=jnp.broadcast_to(jnp.linspace(7.0, 15.0, nz)[:, None, None], (nz, ny, nx + 1)),
        v=jnp.broadcast_to(jnp.linspace(1.0, 4.0, nz)[:, None, None], (nz, ny + 1, nx)),
        t_skin=jnp.full((ny, nx), 300.0),
        xland=jnp.full((ny, nx), 1.0),
        mavail=jnp.full((ny, nx), 0.7),
        roughness_m=jnp.full((ny, nx), 0.08),
        ustar=jnp.full((ny, nx), 0.35),
        lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)
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


def _namelist(grid: GridSpec, **over) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(grid, dt_s=10.0, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(base, time_utc=TIME_UTC, run_physics=True, **over)


def test_operational_step_routes_camuw_and_changes_state() -> None:
    grid = _grid()
    state = _state(grid)
    nml = _namelist(
        grid,
        mp_physics=0,
        bl_pbl_physics=9,
        sf_sfclay_physics=1,
        cu_physics=0,
        use_noahmp=False,
    )
    suite = _resolve_operational_suite(nml)
    assert suite.pbl.option == 9
    assert suite.pbl.gpu_runnable is True
    assert suite.surface_layer.option == 1

    forcing = _physics_step_forcing(initial_operational_carry(state), nml, 0.0, run_radiation=False)
    after = forcing.state
    for leaf in ("theta", "qv", "qc", "qi", "u", "v", "qke"):
        assert np.all(np.isfinite(np.asarray(getattr(after, leaf)))), leaf
    assert float(np.max(np.abs(np.asarray(after.theta) - np.asarray(state.theta)))) > 1.0e-5
    assert float(np.max(np.abs(np.asarray(after.u) - np.asarray(state.u)))) > 1.0e-5
    assert float(np.max(np.abs(np.asarray(after.qke) - np.asarray(state.qke)))) > 1.0e-5
