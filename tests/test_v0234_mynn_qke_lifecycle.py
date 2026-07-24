"""WRF-source lifecycle gates for first-call MYNN QKE initialization."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import physics_couplers as couplers


@dataclass(frozen=True)
class _State:
    qke: object

    def replace(self, **updates):
        return _State(updates.get("qke", self.qke))


def _install_seed(monkeypatch, value=7.0):
    calls = []

    def seed(state, grid):
        calls.append((state, grid))
        return jnp.full_like(state.qke, value)

    monkeypatch.setattr(couplers, "mynn_coldstart_qke_from_state", seed)
    return calls


def test_fresh_first_call_reinitializes_even_when_qke_exceeds_cycling_threshold(
    monkeypatch,
):
    calls = _install_seed(monkeypatch)
    state = _State(jnp.full((3, 2), 25.0))

    result = couplers._mynn_state_with_first_call_qke(
        state, None, True, restart=False
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(np.asarray(result.qke), np.full((3, 2), 7.0))


def test_nonfirst_call_preserves_qke(monkeypatch):
    calls = _install_seed(monkeypatch)
    state = _State(jnp.full((3, 2), 0.0))

    result = couplers._mynn_state_with_first_call_qke(state, None, False)

    assert not calls
    assert result is state


def test_restart_skips_initialization(monkeypatch):
    calls = _install_seed(monkeypatch)
    state = _State(jnp.full((3, 2), 0.0))

    result = couplers._mynn_state_with_first_call_qke(
        state, None, True, restart=True
    )

    assert not calls
    assert result is state


def test_traced_fresh_first_call_uses_same_unconditional_branch(monkeypatch):
    calls = _install_seed(monkeypatch)
    state = _State(jnp.full((3, 2), 25.0))

    result = couplers._mynn_state_with_first_call_qke(
        state, None, jnp.asarray(True), restart=False
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(np.asarray(result.qke), np.full((3, 2), 7.0))


def test_lifecycle_authority_must_be_static_python_bool(monkeypatch):
    _install_seed(monkeypatch)
    state = _State(jnp.full((3, 2), 25.0))

    with np.testing.assert_raises_regex(TypeError, "lifecycle flag"):
        couplers._mynn_state_with_first_call_qke(
            state, None, True, restart=jnp.asarray(False)
        )
