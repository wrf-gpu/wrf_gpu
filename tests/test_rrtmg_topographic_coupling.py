from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import BCMetadata, DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.physics_couplers import (
    _compute_coszen,
    _compute_solar_geometry,
    build_radiation_static_from_wrf_fields,
    rrtmg_radiation_diagnostics,
    wrf_radiation_slope_aspect_from_terrain,
)


def _grid(*, ny: int = 4, nx: int = 4, nz: int = 4, terrain=None) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    terrain_height = (
        jnp.zeros((ny, nx), dtype=jnp.float64)
        if terrain is None
        else jnp.asarray(terrain, dtype=jnp.float64)
    )
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="unit-test",
        sha256="unit-test",
        shape=(ny, nx),
        units="m",
        projection_transform="native-wrf-lambert",
        max_elevation_m=float(np.asarray(terrain_height).max()),
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
        provenance="unit-test-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, terrain_height, metrics=metrics)


def _state(grid: GridSpec) -> State:
    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    p = jnp.linspace(90000.0, 60000.0, nz, dtype=jnp.float64)[:, None, None]
    p = jnp.broadcast_to(p, (nz, ny, nx))
    ph = jnp.linspace(0.0, 4500.0 * 9.80665, nz + 1, dtype=jnp.float64)[:, None, None]
    ph = jnp.broadcast_to(ph, (nz + 1, ny, nx))
    mu = jnp.full((ny, nx), 85000.0, dtype=jnp.float64)
    fields.update(
        theta=jnp.full((nz, ny, nx), 300.0, dtype=jnp.float64),
        qv=jnp.full((nz, ny, nx), 0.006, dtype=jnp.float64),
        p=p,
        p_total=p,
        ph=ph,
        ph_total=ph,
        mu=mu,
        mu_total=mu,
        t_skin=jnp.full((ny, nx), 290.0, dtype=jnp.float64),
        xland=jnp.ones((ny, nx), dtype=jnp.float64),
        lu_index=jnp.full((ny, nx), 13, dtype=jnp.int32),
    )
    return State(**fields)


def test_solar_geometry_coszen_matches_legacy_helper():
    lat = jnp.asarray([[28.0, 28.2], [28.4, 28.6]], dtype=jnp.float64)
    lon = jnp.asarray([[-16.7, -16.5], [-16.3, -16.1]], dtype=jnp.float64)
    init = datetime(2026, 5, 21, 18, tzinfo=timezone.utc)
    geometry = _compute_solar_geometry(lat, lon, init, lead_seconds=18 * 3600.0)
    legacy = _compute_coszen(lat, lon, init, lead_seconds=18 * 3600.0)
    np.testing.assert_allclose(np.asarray(geometry.coszen), np.asarray(legacy), rtol=0, atol=0)


def test_wrf_slope_aspect_matches_start_em_ramp_formula():
    terrain = jnp.asarray(
        [
            [0.0, 300.0, 600.0],
            [0.0, 300.0, 600.0],
            [0.0, 300.0, 600.0],
        ],
        dtype=jnp.float64,
    )
    slope, aspect = wrf_radiation_slope_aspect_from_terrain(
        terrain,
        dx_m=3000.0,
        dy_m=3000.0,
    )
    np.testing.assert_allclose(float(slope[1, 1]), np.arctan(0.1), rtol=0, atol=1.0e-14)
    np.testing.assert_allclose(float(aspect[1, 1]), 1.5 * np.pi, rtol=0, atol=1.0e-14)


def test_rrtmg_diagnostics_use_land_albedo_and_topographic_swnorm():
    terrain = jnp.asarray(
        [
            [0.0, 500.0, 1000.0, 1500.0],
            [0.0, 500.0, 1000.0, 1500.0],
            [0.0, 500.0, 1000.0, 1500.0],
            [0.0, 500.0, 1000.0, 1500.0],
        ],
        dtype=jnp.float64,
    )
    grid = _grid(terrain=terrain)
    ny, nx = grid.ny, grid.nx
    xlat = jnp.full((ny, nx), 28.3, dtype=jnp.float64)
    xlong = jnp.full((ny, nx), -16.4, dtype=jnp.float64)
    radiation_static = build_radiation_static_from_wrf_fields(
        xlat,
        xlong,
        terrain,
        dx_m=grid.projection.dx_m,
        dy_m=grid.projection.dy_m,
        msftx=grid.metrics.msftx,
        msfty=grid.metrics.msfty,
        sina=grid.metrics.sina,
        cosa=grid.metrics.cosa,
    )
    land = SimpleNamespace(
        albedo=jnp.full((ny, nx), 0.61, dtype=jnp.float64),
        emiss=jnp.full((ny, nx), 0.93, dtype=jnp.float64),
    )
    diag = rrtmg_radiation_diagnostics(
        _state(grid),
        grid,
        time_utc=datetime(2026, 5, 22, 12, tzinfo=timezone.utc),
        lead_seconds=0.0,
        radiation_static=radiation_static,
        topo_shading=0,
        slope_rad=1,
        land_state=land,
    )
    np.testing.assert_allclose(np.asarray(diag.surface_albedo), 0.61)
    np.testing.assert_allclose(np.asarray(diag.surface_emissivity), 0.93)
    assert np.all(np.isfinite(np.asarray(diag.swnorm)))
    assert not np.allclose(np.asarray(diag.swnorm), np.asarray(diag.swdown))
    assert np.all(np.asarray(diag.shadow_mask) == 0)
