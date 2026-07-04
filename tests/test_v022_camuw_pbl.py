"""F3 CAM-UW PBL reference-only / fail-closed wiring.

F3 built the standalone WRF-Fortran CAM-UW column oracle and proved the old
JAX CAM-UW scaffold RED against it. These tests lock the honest scope decision:
``bl_pbl_physics=9`` remains namelist-accepted for oracle comparison, but the
operational scan and endpoint stubs must fail closed with the F3 reason.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import Tendencies
from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection
from gpuwrf.coupling.scan_adapters import camuw_pbl_adapter
from gpuwrf.io.namelist_check import (
    NotOperationallyWiredError,
    validate_namelist,
    validate_operational_namelist,
)
from gpuwrf.physics.bl_camuw import (
    CAMUW_REFERENCE_ONLY_REASON,
    CamUwReferenceOnlyError,
    camuw_columns,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist, _resolve_operational_suite

TIME_UTC = "2024-06-01T12:00:00Z"


def test_camuw_column_endpoint_fails_closed() -> None:
    arr = jnp.ones((1, 2), dtype=jnp.float64)
    scal = jnp.ones((1,), dtype=jnp.float64)

    with pytest.raises(CamUwReferenceOnlyError) as excinfo:
        camuw_columns(
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            arr,
            hfx=scal,
            qfx=scal,
            ust=scal,
            wspd=scal,
            dt=60.0,
        )

    message = str(excinfo.value)
    assert message == CAMUW_REFERENCE_ONLY_REASON
    assert "REFERENCE_ONLY" in message
    assert "pblh max_abs=1384.6212005615234 m" in message
    assert "separate milestone" in message


def test_camuw_scan_adapter_stub_fails_closed() -> None:
    with pytest.raises(CamUwReferenceOnlyError, match="REFERENCE_ONLY"):
        camuw_pbl_adapter(object(), 60.0, grid=None)  # type: ignore[arg-type]


def test_camuw_namelist_accepts_reference_only_but_operational_rejects() -> None:
    cfg = {"physics": {"bl_pbl_physics": [9], "sf_sfclay_physics": [1]}}

    validate_namelist(cfg)

    with pytest.raises(NotOperationallyWiredError) as excinfo:
        validate_operational_namelist(cfg)

    message = str(excinfo.value)
    assert "bl_pbl_physics=9" in message
    assert "CAM-UW" in message
    assert "REFERENCE-ONLY" in message
    assert "NOT operationally wired" in message
    assert "bl_pbl_physics=0/1/2/3/5/7/8/11/12/99" in message


def _grid(ny: int = 3, nx: int = 3, nz: int = 8) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="camuw-reference-only-test",
        sha256="camuw-reference-only-test",
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
        provenance="camuw-reference-only-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


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


def test_operational_suite_rejects_camuw_before_compute() -> None:
    grid = _grid()
    nml = _namelist(
        grid,
        mp_physics=0,
        bl_pbl_physics=9,
        sf_sfclay_physics=1,
        cu_physics=0,
        use_noahmp=False,
    )

    with pytest.raises(UnsupportedSchemeSelection) as excinfo:
        _resolve_operational_suite(nml)

    message = str(excinfo.value)
    assert "bl_pbl_physics=9" in message
    assert "CAM-UW is F3 REFERENCE_ONLY" in message
    assert "pblh max_abs=1384.6212005615234 m" in message
    assert "bl_pbl_physics in {0,1,2,3,5,7,8,11,12,99}" in message
