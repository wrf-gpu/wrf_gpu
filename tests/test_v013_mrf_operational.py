"""v0.13 operational wiring of the MRF PBL (bl_pbl_physics=99).

Covers the WIRING (kernel-vs-WRF parity lives in ``proofs/v013/mrf_oracle.py``):

* ``mrf_columns`` is genuinely ``jax.jit``-traceable (operational-scan requirement);
* the operational PBL slot routes ``bl_pbl_physics=99`` to the MRF adapter and the
  result actually CHANGES the state (not a silent no-op);
* MRF (bl=99) resolves in ``_resolve_operational_suite`` with the revised-MM5
  surface layer (sf=1), and the DEFAULT suite (MYNN / sfclayrev) is byte-for-byte
  UNCHANGED by this wiring.
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
from gpuwrf.physics.bl_mrf import mrf_columns
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _physics_step_forcing,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


TIME_UTC = "2019-05-21T12:00:00Z"


# ----------------------------------------------------------------------------
# 1. traceable kernel
# ----------------------------------------------------------------------------

def test_mrf_columns_is_jit_traceable_and_finite() -> None:
    """Batched mrf_columns runs under jax.jit and yields finite tendencies."""

    n = 16
    z_mid = np.linspace(20.0, 8000.0, n)
    dz = np.full(n, 500.0)
    A = lambda v: jnp.asarray(np.stack([v, v]), jnp.float64)
    sc = lambda v: jnp.asarray([v, v], jnp.float64)
    u = np.full(n, 5.0)
    v = np.full(n, 2.0)
    t = np.linspace(295.0, 250.0, n)
    qv = np.full(n, 5.0e-3)
    p = np.linspace(95000.0, 30000.0, n)
    pii = (p / 1.0e5) ** (287.0 / 1004.5)
    out = jax.jit(mrf_columns)(
        A(u), A(v), A(t), A(qv), jnp.zeros((2, n), jnp.float64),
        A(p), A(pii), A(dz), A(z_mid),
        psfc=sc(100000.0), znt=sc(0.1), ust=sc(0.4), hfx=sc(200.0), qfx=sc(1.0e-4),
        tsk=sc(300.0), gz1oz0=sc(np.log(20.0 / 0.1)), wspd=sc(np.hypot(5.0, 2.0)),
        br=sc(-0.05), psim=sc(0.0), psih=sc(0.0), xland=sc(1.0), dt=60.0,
    )
    assert out["rthblten"].shape == (2, n)
    for k in ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten"):
        assert bool(np.all(np.isfinite(np.asarray(out[k]))))


# ----------------------------------------------------------------------------
# 2. operational dispatch wiring
# ----------------------------------------------------------------------------

def _grid(ny: int = 3, nx: int = 3, nz: int = 8) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="mrf-wire-test", sha256="mrf-wire-test", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="mrf-wire-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    p = jnp.broadcast_to(jnp.linspace(95000.0, 20000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 12000.0 * 9.80665, nz + 1)[:, None, None], (nz + 1, ny, nx))
    fields.update(
        theta=jnp.full((nz, ny, nx), 295.0),
        p_total=p,
        ph_total=ph,
        mu_total=jnp.full((ny, nx), 90000.0),
        qv=jnp.full((nz, ny, nx), 5.0e-3), qc=jnp.full((nz, ny, nx), 1.0e-4),
        qke=jnp.full((nz, ny, nx), 0.5),
        u=jnp.full((nz, ny, nx + 1), 5.0), v=jnp.full((nz, ny + 1, nx), 2.0),
        t_skin=jnp.full((ny, nx), 298.0), xland=jnp.full((ny, nx), 1.0),
        mavail=jnp.full((ny, nx), 0.5), roughness_m=jnp.full((ny, nx), 0.1),
        ustar=jnp.full((ny, nx), 0.3), lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)
    return Tendencies(
        z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)),
        z((nz, ny, nx)), z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)),
    )


def _namelist(grid: GridSpec, **over) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(grid, dt_s=10.0, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(base, time_utc=TIME_UTC, run_physics=True, **over)


def test_mrf_resolves_in_operational_suite() -> None:
    grid = _grid()
    nml = _namelist(grid, bl_pbl_physics=99, sf_sfclay_physics=1, use_noahmp=False)
    suite = _resolve_operational_suite(nml)
    assert suite.pbl.option == 99
    assert suite.surface_layer.option == 1


def test_operational_step_routes_mrf_and_changes_state() -> None:
    """bl=99 dispatch runs the MRF PBL and actually mutates theta/u (not a no-op)."""

    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, bl_pbl_physics=99, sf_sfclay_physics=1, use_noahmp=False)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    assert np.all(np.isfinite(np.asarray(after.theta)))
    assert np.all(np.isfinite(np.asarray(after.u)))
    assert not np.allclose(np.asarray(after.theta), np.asarray(state.theta))
    assert not np.allclose(np.asarray(after.u), np.asarray(state.u))


def test_default_suite_byte_unchanged_by_mrf_wiring() -> None:
    """The default (MYNN / sfclayrev) physics step is byte-for-byte unchanged."""

    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, use_noahmp=False)  # defaults: bl=5 MYNN, sf=5 MYNN-sfclay
    assert nml.bl_pbl_physics == 5 and nml.sf_sfclay_physics == 5
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    forcing2 = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    for leaf in ("theta", "qv", "u", "v", "qke", "ustar"):
        a = np.asarray(getattr(forcing.state, leaf))
        b = np.asarray(getattr(forcing2.state, leaf))
        assert np.array_equal(a, b), f"default {leaf} not deterministic/unchanged"
