"""v0.15 Stream-A: device-side finite guard must stay fail-closed.

`finite_guard_summary` replaces the per-hour full-state host pull with one
on-device all-finite reduce. These tests pin its contract:
  * clean state -> fast path, `all_finite=True`, NO per-field host pull;
  * any non-finite device leaf -> falls back to the FULL host `finite_summary`
    (identical failure payload: per-field entries present);
  * any non-finite host (numpy) leaf -> same fallback;
  * integer leaves never trip the guard.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import finite_guard_summary, finite_summary


class _SlotState:
    __slots__ = ("a", "b", "ints")

    def __init__(self, a, b, ints):
        self.a = a
        self.b = b
        self.ints = ints


def _clean_state() -> _SlotState:
    return _SlotState(
        a=jnp.ones((4, 5), dtype=jnp.float64),
        b=np.linspace(0.0, 1.0, 7),
        ints=np.arange(5, dtype=np.int32),
    )


def test_clean_state_fast_path() -> None:
    summary = finite_guard_summary(_clean_state())
    assert summary["all_finite"] is True
    # fast path returns no per-field map (success consumers only read all_finite)
    assert summary["fields"] == {}
    assert summary["field_count"] == 3


def test_device_nan_falls_back_to_full_summary() -> None:
    state = _clean_state()
    state.a = state.a.at[1, 2].set(jnp.nan)
    summary = finite_guard_summary(state)
    assert summary["all_finite"] is False
    # full host summary payload (fail-closed report identical to finite_summary)
    assert summary["fields"]["a"]["nonfinite_count"] == 1
    assert summary == finite_summary(state)


def test_host_inf_falls_back_to_full_summary() -> None:
    state = _clean_state()
    state.b = np.array([0.0, np.inf, 1.0])
    summary = finite_guard_summary(state)
    assert summary["all_finite"] is False
    assert summary == finite_summary(state)


def test_integer_leaves_never_trip() -> None:
    state = _clean_state()
    state.ints = np.array([np.iinfo(np.int32).max, -1], dtype=np.int32)
    assert finite_guard_summary(state)["all_finite"] is True
