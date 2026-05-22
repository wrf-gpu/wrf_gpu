from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.dynamics.hybrid_eta import face_level_pressure, mass_level_pressure, pressure_thickness
from gpuwrf.dynamics.metrics import flat_metrics_for_grid, load_wrfinput_metrics


WRFINPUT_D02 = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z/wrfinput_d02"
)


def test_hybrid_pressure_matches_analytic_oracle():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    mu = jnp.full((grid.ny, grid.nx), 90000.0, dtype=jnp.float64)

    pressure = mass_level_pressure(mu, metrics)
    expected = metrics.c3h[:, None, None] * mu[None, :, :] + metrics.c4h[:, None, None] + metrics.p_top

    assert pressure.shape == (grid.nz, grid.ny, grid.nx)
    assert jnp.max(jnp.abs(pressure - expected)) == 0.0
    assert jnp.all(pressure_thickness(mu, metrics) >= 0.0)


def test_hybrid_helpers_jaxpr_has_no_host_callback():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    mu = jnp.full((grid.ny, grid.nx), 90000.0, dtype=jnp.float64)

    jaxpr = str(jax.make_jaxpr(face_level_pressure)(mu, metrics)).lower()

    assert "host_callback" not in jaxpr
    assert "io_callback" not in jaxpr
    assert "pure_callback" not in jaxpr


def test_wrfinput_hybrid_coefficients_load_when_fixture_available():
    if not WRFINPUT_D02.exists():
        return

    metrics = load_wrfinput_metrics(WRFINPUT_D02)
    mu = jnp.full(metrics.msftx.shape, 85000.0, dtype=jnp.float64)
    pressure = mass_level_pressure(mu, metrics)

    assert pressure.shape == (44, 66, 159)
    assert face_level_pressure(mu, metrics).shape == (45, 66, 159)
    assert float(jnp.min(pressure)) > 0.0
