from __future__ import annotations

import os
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.integration.nested_pipeline import _make_namelist


def _tiny_grid() -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, 4, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, 2, 2)
    terrain = TerrainProvenance(
        "analytic://v020-acoustic-substeps",
        "v020-acoustic-substeps",
        (2, 2),
        "m",
        "flat",
        0.0,
        True,
    )
    vertical = VerticalCoord("hybrid_eta", 3, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    return GridSpec(
        projection,
        terrain,
        vertical,
        bc,
        eta,
        jnp.zeros((2, 2), dtype=jnp.float64),
    )


def _nested_namelist_acoustic_substeps() -> int:
    grid = _tiny_grid()
    namelist = _make_namelist(
        grid=grid,
        tendencies=object(),
        metrics=grid.metrics,
        dt_s=18.0,
        parent_dt_s=None,
        run_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        radiation_static=None,
        cu_physics=0,
    )
    return int(namelist.acoustic_substeps)


def test_nested_pipeline_defaults_to_ten_acoustic_substeps(monkeypatch):
    monkeypatch.delenv("GPUWRF_ACOUSTIC_SUBSTEPS", raising=False)
    assert _nested_namelist_acoustic_substeps() == 10


def test_nested_pipeline_accepts_acoustic_substeps_env_override(monkeypatch):
    monkeypatch.setenv("GPUWRF_ACOUSTIC_SUBSTEPS", "7")
    assert _nested_namelist_acoustic_substeps() == 7
