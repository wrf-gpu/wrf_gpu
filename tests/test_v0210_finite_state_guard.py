"""v0.21 A1: fail-fast finite detector at output/chunk boundaries."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime.finite_state_guard import (
    NonFiniteStateError,
    assert_state_finite_at_boundary,
    finite_check_enabled,
    first_nonfinite_state_location,
)


class _StateLike:
    __slots__ = ("theta", "qv", "u", "xland")

    def __init__(self, *, theta, qv=None, u=None, xland=None):
        self.theta = theta
        self.qv = jnp.ones_like(theta) if qv is None else qv
        self.u = jnp.ones_like(theta) if u is None else u
        self.xland = np.asarray([np.nan]) if xland is None else xland


def test_default_env_enables_finite_check(monkeypatch) -> None:
    monkeypatch.delenv("GPUWRF_FINITE_CHECK", raising=False)
    assert finite_check_enabled() is True
    monkeypatch.setenv("GPUWRF_FINITE_CHECK", "0")
    assert finite_check_enabled() is False


def test_finite_state_passes_and_values_are_bit_identical(monkeypatch) -> None:
    monkeypatch.delenv("GPUWRF_FINITE_CHECK", raising=False)
    theta = jnp.arange(24, dtype=jnp.float32).reshape((2, 3, 4))
    state = _StateLike(theta=theta)
    before = np.asarray(state.theta).copy()

    assert_state_finite_at_boundary(state, domain="d01", step=12, sim_time_s=216.0)

    assert np.array_equal(np.asarray(state.theta), before)


def test_nan_raises_typed_error_with_field_level_domain_and_step() -> None:
    theta = jnp.ones((3, 4, 5), dtype=jnp.float32)
    state = _StateLike(theta=theta.at[2, 1, 3].set(jnp.nan))

    with pytest.raises(NonFiniteStateError) as raised:
        assert_state_finite_at_boundary(state, domain="d02", step=123, sim_time_s=369.0)

    err = raised.value
    assert err.field == "theta"
    assert err.domain == "d02"
    assert err.level == 2
    assert err.step == 123
    assert err.index == (2, 1, 3)
    assert err.sim_time_s == pytest.approx(369.0)
    assert "field=theta" in str(err)
    assert "domain=d02" in str(err)
    assert "level=2" in str(err)
    assert "step=123" in str(err)


def test_inf_reports_first_available_prognostic_field() -> None:
    theta = jnp.ones((3, 4, 5), dtype=jnp.float32)
    qv = theta.at[1, 0, 4].set(jnp.inf)
    state = _StateLike(theta=theta, qv=qv)

    loc = first_nonfinite_state_location(state, domain="d03", step=9)

    assert loc is not None
    assert loc.field == "qv"
    assert loc.level == 1
    assert loc.index == (1, 0, 4)


def test_opt_out_env_disables_guard(monkeypatch) -> None:
    monkeypatch.setenv("GPUWRF_FINITE_CHECK", "0")
    theta = jnp.ones((3, 4, 5), dtype=jnp.float32).at[0, 0, 0].set(jnp.nan)
    state = _StateLike(theta=theta)

    assert_state_finite_at_boundary(
        state, domain="d02", step=1, environ={"GPUWRF_FINITE_CHECK": "0"}
    )

