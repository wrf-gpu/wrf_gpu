"""CAM-UW (bl_pbl_physics=9) operational wiring (v0.3.4).

The v023 F3 verdict (reference-only, scaffold RED) compared against a defective oracle (no
WRF esinti -> zero saturation table). The faithful r8 port is parity-gated in
tests/test_v034_camuw_oracle_parity.py; these tests lock the operational wiring:
namelist acceptance, the surface-flux pairing rule, routing through the PBL slot with the
OperationalCarry.camuw_pbl (WRF KVM3D/KVH3D/TAURES) carry, and the PBLH hand-off.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (  # noqa: E402
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes  # noqa: E402
from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection  # noqa: E402
from gpuwrf.io.namelist_check import validate_namelist, validate_operational_namelist  # noqa: E402
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    OperationalNamelist,
    _initial_carry_for_run,
    _physics_step_forcing,
    _resolve_operational_suite,
)

TIME_UTC = "2024-06-01T12:00:00Z"


def _grid(ny: int = 3, nx: int = 3, nz: int = 10) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="camuw-wire-test", sha256="camuw-wire-test", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0,
                                 provenance="camuw-wire-flat")
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    p = jnp.broadcast_to(jnp.linspace(97000.0, 25000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 10000.0 * 9.81, nz + 1)[:, None, None], (nz + 1, ny, nx))
    zc = jnp.linspace(0.0, 10000.0, nz)[:, None, None]
    theta = jnp.broadcast_to(300.0 + 0.004 * zc - 0.6 * (zc < 900.0), (nz, ny, nx))  # unstable ML
    fields.update(
        theta=theta, p_total=p, ph_total=ph, mu_total=jnp.full((ny, nx), 90000.0),
        qv=jnp.full((nz, ny, nx), 6.0e-3), qc=jnp.full((nz, ny, nx), 1.0e-5),
        u=jnp.broadcast_to(3.0 + 0.002 * zc, (nz, ny, nx + 1)) * 1.0,
        v=jnp.full((nz, ny + 1, nx), 1.0),
        t_skin=jnp.full((ny, nx), 305.0), xland=jnp.full((ny, nx), 1.0),
        mavail=jnp.full((ny, nx), 0.5), roughness_m=jnp.full((ny, nx), 0.1),
        ustar=jnp.full((ny, nx), 0.3), lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    return Tendencies(z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)), z((nz, ny, nx)),
                      z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)))


def _namelist(grid: GridSpec, **over) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(grid, dt_s=10.0, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(base, time_utc=TIME_UTC, run_physics=True, **over)


def test_camuw_is_implemented_with_honest_qualification_note() -> None:
    support = classify_scheme("bl_pbl_physics", 9)
    assert support.status is SupportStatus.IMPLEMENTED
    assert "CPU-oracle-qualified" in support.reason and "GPU" in support.reason
    cfg = {"physics": {"bl_pbl_physics": [9], "sf_sfclay_physics": [5]}}
    validate_namelist(cfg)
    validate_operational_namelist(cfg)


def test_camuw_suite_requires_wrf_surface_flux_producer() -> None:
    grid = _grid()
    ok = _resolve_operational_suite(_namelist(grid, mp_physics=0, bl_pbl_physics=9, sf_sfclay_physics=5,
                                              cu_physics=0, use_noahmp=False))
    assert ok.pbl.option == 9 and ok.pbl.gpu_runnable is True
    with pytest.raises(UnsupportedSchemeSelection, match="CAM-UW"):
        _resolve_operational_suite(_namelist(grid, mp_physics=0, bl_pbl_physics=9, sf_sfclay_physics=1,
                                             cu_physics=0, use_noahmp=False))


def test_operational_step_routes_camuw_threads_carry_and_pblh() -> None:
    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, mp_physics=0, bl_pbl_physics=9, sf_sfclay_physics=5, cu_physics=0, use_noahmp=False)
    carry = _initial_carry_for_run(state, nml)
    assert carry.camuw_pbl is not None
    assert float(jnp.max(jnp.abs(carry.camuw_pbl.kvh3d))) == 0.0  # itimestep=1 == zero carry
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    nxt = forcing.carry if hasattr(forcing, "carry") else forcing
    camuw = getattr(nxt, "camuw_pbl", None)
    assert camuw is not None
    for leaf in ("theta", "u", "v", "qv", "qc"):
        assert np.all(np.isfinite(np.asarray(getattr(after, leaf)))), leaf
    assert not np.allclose(np.asarray(after.theta), np.asarray(state.theta))
    assert not np.allclose(np.asarray(after.u), np.asarray(state.u))
    kvh = np.asarray(camuw.kvh3d)
    assert kvh.shape == (grid.nz + 1, grid.ny, grid.nx) and kvh.dtype == np.float32
    assert np.max(kvh) > 1.0  # convective mixed layer -> UW CL eddy diffusivity
    assert np.all(kvh[0] == 0.0) and np.all(kvh[-1] == 0.0)  # WRF: surface/top interfaces never set
    assert np.all(np.asarray(after.pblh) > 0.0)  # WRF PBLH2D hand-off (read by SFCLAY_mynn)
    assert np.any(np.asarray(camuw.tauresx2d) != 0.0)
