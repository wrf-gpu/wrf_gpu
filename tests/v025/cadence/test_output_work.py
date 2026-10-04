"""Output-work wiring checks; flux sentinels do not assert WRF fidelity."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import importlib.util

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics
from gpuwrf.diagnostics import census
from gpuwrf.integration import daily_pipeline, nested_pipeline
from gpuwrf.runtime import operational_mode as op

spec = importlib.util.spec_from_file_location(
    "output_work_fixture", Path(__file__).resolve().parents[2] / "test_rrtm_lw_operational_wiring.py"
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.mark.parametrize("held,subset,expected", [
    (True, None, 0), (False, None, 1),
    # A T2-only full diagnostic fallback still invokes radiation. SW-only
    # selected output computes its complete family even with surface SW held.
    (False, ("T2",), 1), (False, ("SWDOWN",), 1),
])
def test_noahmp_output_returns_actual_solver_work(monkeypatch, held, subset, expected):
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = fixture._state(grid)
    nml = dataclasses.replace(fixture._namelist(grid), use_noahmp=True)
    values = [jnp.full((2, 2), float(i + 1)) for i in range(15)]
    values[10] = jnp.zeros((2, 2), jnp.int32)
    rad = RRTMGRadiationDiagnostics(*values)
    z = jnp.zeros((2, 2))
    calls = []
    def solve(*args, **kwargs):
        calls.append(1)
        return rad
    monkeypatch.setattr(op, "rrtmg_radiation_diagnostics", solve)
    monkeypatch.setattr(op, "surface_layer_diagnostics", lambda *a, **k: SimpleNamespace(
        hfx=z, lh=z, t2=z, pblh=z, u10=z, v10=z))
    monkeypatch.setattr(op, "overlay_noahmp_land_diagnostics", lambda *a, **k: (z, z, z, z))
    monkeypatch.setattr(op, "_lane_noahmp_static", lambda *a: None)
    op.compute_m9_selected_diagnostics.clear_cache()
    try:
        for _ in range(2):  # A cached JIT invocation must still return one solve.
            work = []
            result = nested_pipeline._noahmp_surface_diagnostics_for_output(
                state, nml, datetime(2026, 7, 26, tzinfo=timezone.utc),
                lead_seconds=54., noahmp_land=None, noahmp_rad=(z, z, z),
                radiation_diagnostics=rad if held else None, variable_subset=subset,
                output_radiation_work=work)
            assert result is not None
            assert sum(work) == expected
        assert bool(calls) == bool(expected)
    finally:
        op.compute_m9_selected_diagnostics.clear_cache()


def test_daily_legacy_fallback_returns_actual_solver_work(monkeypatch):
    # The generic writer helper must forward the count actually returned by M9.
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = fixture._state(grid)
    nml = fixture._namelist(grid)
    def diagnostics(*args, **kwargs):
        assert kwargs["_with_radiation_count"] is True
        return SimpleNamespace(t2=jnp.ones((2, 2))), 2
    monkeypatch.setattr(daily_pipeline, "compute_m9_diagnostics", diagnostics)
    work = []
    assert daily_pipeline._surface_diagnostics_for_output(
        state, nml, datetime(2026, 7, 26, tzinfo=timezone.utc),
        lead_seconds=0., variable_subset=("T2",), output_radiation_work=work)
    assert work == [2]


def test_disabled_segment_has_no_transfer_or_io(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled census touched transfer/IO")
    monkeypatch.setattr(jax, "device_get", forbidden)
    writer = SimpleNamespace(census_io_ledger=SimpleNamespace(snapshot=forbidden))
    census.write_segment(tmp_path, {"d01": SimpleNamespace(census=None)},
                         {}, {}, writer, {})
    assert list(tmp_path.iterdir()) == []
