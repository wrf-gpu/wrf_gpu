from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.dynamics.metrics import (
    flat_metrics_for_grid,
    load_wrfinput_metrics,
    mass_metric_area,
    metric_minmax,
    u_metric_ratio,
    v_metric_ratio,
)


WRFINPUT_D02 = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z/wrfinput_d02"
)


def test_flat_metrics_shapes_staggering_and_identity_values():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)

    assert metrics.msftx.shape == (grid.ny, grid.nx)
    assert metrics.msfux.shape == (grid.ny, grid.nx + 1)
    assert metrics.msfvx.shape == (grid.ny + 1, grid.nx)
    assert metrics.dzdx.shape == (grid.ny, grid.nx)
    assert metrics.dzdy.shape == (grid.ny, grid.nx)
    assert metrics.dzdx_u.shape == (grid.ny, grid.nx + 1)
    assert metrics.dzdy_v.shape == (grid.ny + 1, grid.nx)
    assert metrics.c1h.shape == (grid.nz,)
    assert metrics.c1f.shape == (grid.nz + 1,)
    assert metrics.cf1.shape == ()
    assert metrics.cf2.shape == ()
    assert metrics.cf3.shape == ()
    assert metrics.fnm.shape == (grid.nz,)
    assert metrics.fnp.shape == (grid.nz,)
    assert metrics.msftx.dtype == jnp.float64
    assert metrics.fnm.dtype == jnp.float64
    assert jnp.allclose(metrics.dzdx, 0.0)
    assert jnp.allclose(metrics.dzdy, 0.0)
    assert jnp.allclose(metrics.fnm, 0.5)
    assert jnp.allclose(metrics.fnp, 0.5)
    assert jnp.allclose(jnp.asarray([metrics.cf1, metrics.cf2, metrics.cf3]), jnp.asarray([2.0, -1.5, 0.5]))
    assert jnp.allclose(mass_metric_area(metrics), 1.0)
    assert jnp.allclose(u_metric_ratio(metrics), 1.0)
    assert jnp.allclose(v_metric_ratio(metrics), 1.0)


def test_metric_helpers_jaxpr_has_no_host_callback():
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)

    jaxpr = str(jax.make_jaxpr(metric_minmax)(metrics)).lower()

    assert "host_callback" not in jaxpr
    assert "io_callback" not in jaxpr
    assert "pure_callback" not in jaxpr


def test_wrfinput_metric_loader_shapes_when_fixture_available():
    if not WRFINPUT_D02.exists():
        return

    metrics = load_wrfinput_metrics(WRFINPUT_D02)

    assert metrics.msftx.shape == (66, 159)
    assert metrics.msfux.shape == (66, 160)
    assert metrics.msfvx.shape == (67, 159)
    assert metrics.dzdx.shape == (66, 159)
    assert metrics.dzdy.shape == (66, 159)
    assert metrics.dzdx_u.shape == (66, 160)
    assert metrics.dzdy_v.shape == (67, 159)
    assert metrics.c1h.shape == (44,)
    assert metrics.c1f.shape == (45,)
    assert metrics.cf1.shape == ()
    assert metrics.cf2.shape == ()
    assert metrics.cf3.shape == ()
    assert metrics.fnm.shape == (44,)
    assert metrics.fnp.shape == (44,)
    assert metrics.p_top.shape == ()
    assert float(jnp.min(metric_minmax(metrics))) > 0.0

    with Dataset(WRFINPUT_D02) as dataset:
        assert float(metrics.cf1) == float(np.asarray(dataset.variables["CF1"][0], dtype=np.float64))
        assert float(metrics.cf2) == float(np.asarray(dataset.variables["CF2"][0], dtype=np.float64))
        assert float(metrics.cf3) == float(np.asarray(dataset.variables["CF3"][0], dtype=np.float64))
        assert jnp.allclose(metrics.fnm, jnp.asarray(np.asarray(dataset.variables["FNM"][0], dtype=np.float64)))
        assert jnp.allclose(metrics.fnp, jnp.asarray(np.asarray(dataset.variables["FNP"][0], dtype=np.float64)))
