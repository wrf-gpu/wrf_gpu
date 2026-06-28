from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.io.wrfout_writer import MANDATORY_WRFOUT_COORDINATES, prepare_wrfout_payload
from gpuwrf.runtime import finite_state_guard
from test_m7_netcdf_writer import synthetic_case


def _jaxify_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(
            **{name: _jaxify_arrays(item) for name, item in vars(value).items()}
        )
    if isinstance(value, dict):
        return {name: _jaxify_arrays(item) for name, item in value.items()}
    return value


def test_prepare_wrfout_payload_batches_device_get(monkeypatch, tmp_path):
    state, grid, namelist = synthetic_case()
    state = _jaxify_arrays(state)
    grid = _jaxify_arrays(grid)

    calls: list[Any] = []
    real_device_get = jax.device_get

    def counting_device_get(value):
        calls.append(value)
        return real_device_get(value)

    monkeypatch.setattr(jax, "device_get", counting_device_get)

    prepared = prepare_wrfout_payload(
        state,
        grid,
        namelist,
        tmp_path / "wrfout_d01_2026-05-25_19:00:00",
        valid_time=datetime(2026, 5, 25, 19),
        lead_hours=1.0,
        run_start=datetime(2026, 5, 25, 18),
    )

    assert prepared.fields["U"].shape == (3, 4, 6)
    assert len(calls) == 1
    assert isinstance(calls[0], tuple)
    assert len(calls[0]) >= 20


def test_finite_guard_batches_jax_success_path(monkeypatch):
    theta = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4))
    state = SimpleNamespace(
        u=theta + 1.0,
        theta=theta,
        qv=theta * 0.0 + 0.01,
    )

    calls: list[tuple[Any, ...]] = []
    real_all_finite_flags = finite_state_guard._jax_all_finite_flags

    def counting_all_finite_flags(values):
        calls.append(tuple(values))
        return real_all_finite_flags(values)

    monkeypatch.setattr(
        finite_state_guard, "_jax_all_finite_flags", counting_all_finite_flags
    )

    finite_state_guard.assert_state_finite_at_boundary(
        state, domain="d01", step=12, sim_time_s=216.0
    )

    assert len(calls) == 1
    assert len(calls[0]) == 3


def test_prepare_subset_limits_payload_before_device_get(monkeypatch, tmp_path):
    state, grid, namelist = synthetic_case()
    state = _jaxify_arrays(state)
    grid = _jaxify_arrays(grid)

    calls: list[Any] = []
    real_device_get = jax.device_get

    def counting_device_get(value):
        calls.append(value)
        return real_device_get(value)

    monkeypatch.setattr(jax, "device_get", counting_device_get)

    prepared = prepare_wrfout_payload(
        state,
        grid,
        namelist,
        tmp_path / "wrfout_d01_2026-05-25_19:00:00",
        valid_time=datetime(2026, 5, 25, 19),
        lead_hours=1.0,
        run_start=datetime(2026, 5, 25, 18),
        variable_subset=("T2", "U10"),
        include_mandatory_coords=True,
    )

    allowed = {"T2", "U10"} | (set(MANDATORY_WRFOUT_COORDINATES) - {"Times", "XTIME"})
    assert set(prepared.fields) <= allowed
    assert {"T2", "U10"} <= set(prepared.fields)
    assert {"U", "V", "W", "P", "PB", "QVAPOR"}.isdisjoint(prepared.fields)
    assert len(calls) == 1
    assert isinstance(calls[0], tuple)
    assert len(calls[0]) < 12
