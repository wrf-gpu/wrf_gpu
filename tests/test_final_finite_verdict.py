"""The bool-only final verdict avoids a full-state copy and still fails closed."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts import state as state_module
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.integration import daily_pipeline as daily
from gpuwrf.integration import nested_pipeline as nested
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
from test_m7_netcdf_writer import synthetic_case, writer_authority


@pytest.fixture
def real_state(monkeypatch):
    monkeypatch.setattr(state_module, "_gpu_device", lambda: jax.devices("cpu")[0])
    grid = GridSpec.canary_3km_template()
    # Include active GWDO and conditional physics rather than only default slots.
    return State(**{name: jnp.zeros(shape, DEFAULT_DTYPES.dtype_for(name))
                    for name, shape in _state_field_shapes(
                        grid, include_all_conditional=True, gwd_opt=1).items()})


def test_finite_state_uses_device_guard_without_host_statistics(real_state, monkeypatch):
    expected = daily.finite_summary(real_state)
    assert expected["all_finite"]
    def no_host_copy(_state):
        raise AssertionError("final verdict downloaded the entire finite State")
    monkeypatch.setattr(daily, "finite_summary", no_host_copy)
    got = nested._finite_stats_host(real_state)
    assert got["all_finite"] is True
    assert got["field_count"] == expected["field_count"]
    assert got["fields"] == {}


def test_every_inexact_state_field_still_rejects_nan(real_state):
    names = [name for name in State.__slots__
             if isinstance(getattr(real_state, name), jax.Array)
             and np.issubdtype(getattr(real_state, name).dtype, np.inexact)]
    assert {"theta", "p_total", "qv", "qke", "el_pbl", "dtaux3d", "dtauy3d"} <= set(names)
    for name in names:
        value = getattr(real_state, name)
        poisoned = value.at[(0,) * value.ndim].set(jnp.nan)
        state = real_state.replace(**{name: poisoned})
        got = nested._finite_stats_host(state)
        assert got["all_finite"] is False, name
        assert got["fields"][name]["nonfinite_count"] == 1, name
        assert got["fields"][name]["finite"] is False, name


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_diagnostic_keeps_full_failure_report(real_state, bad):
    value = real_state.el_pbl.at[0, 0, 0].set(bad)
    state = real_state.replace(el_pbl=value)
    assert nested._finite_stats_host(state) == daily.finite_summary(state)


def test_device_error_uses_original_host_fallback(real_state, monkeypatch):
    expected = daily.finite_summary(real_state)
    def fail(_arrays):
        raise RuntimeError("device finite reduction unavailable")
    monkeypatch.setattr(daily, "_all_finite_device", fail)
    assert nested._finite_stats_host(real_state) == expected


def test_host_only_and_empty_fields_keep_original_semantics():
    for state in (SimpleNamespace(), SimpleNamespace(empty=np.empty((0,), np.float32),
                  integer=np.array([1, 2]), optional=None),
                  SimpleNamespace(bad=np.array([np.nan, 2.0], np.float32))):
        got, expected = nested._finite_stats_host(state), daily.finite_summary(state)
        assert got["all_finite"] == expected["all_finite"]
        assert got["field_count"] == expected["field_count"]
        assert got["fields"].keys() == expected["fields"].keys()


def test_final_guard_preserves_full_history_bytes_and_live_state(tmp_path):
    source, grid, namelist = synthetic_case()
    state = SimpleNamespace(**{name: jnp.asarray(value) for name, value in vars(source).items()})
    before = {name: np.asarray(value).tobytes() for name, value in vars(state).items()}
    authority = writer_authority(grid)
    kwargs = dict(domain="d02", domain_authority=authority,
                  valid_time="2026-02-28T03:00:00Z", lead_hours=3,
                  run_start="2026-02-28T00:00:00Z", full_variable_set=True)
    first, second = tmp_path / "before.nc", tmp_path / "after.nc"
    write_wrfout_netcdf(state, grid, namelist, first, **kwargs)
    assert nested._finite_stats_host(state)["all_finite"] is True
    write_wrfout_netcdf(state, grid, namelist, second, **kwargs)
    assert first.read_bytes() == second.read_bytes()
    assert {name: np.asarray(value).tobytes() for name, value in vars(state).items()} == before
