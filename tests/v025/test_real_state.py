"""Guards on the REAL production-state snapshot used by A5/A6 (contract §10).

§10 forbids synthetic arrays. These tests do not re-download or re-run anything;
they check the two properties that decide whether the map means anything:

1. the loader refuses to run once a GPU backend has been initialised, because
   §13 makes that a coordination violation rather than a convenience; and
2. the RK stage pair is not degenerate. A pair whose two members are the same
   array makes every WRF small-step work prime identically zero, and a
   cancellation map built on it would report perfect fp32 safety for exactly the
   subtractions the v0.25 rewrite is worried about. That failure is silent -- the
   harness runs, every operator returns, every error is zero -- so it needs a
   test rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v025 import real_state  # noqa: E402


def test_cpu_only_env_defaults_are_set_at_import():
    """Importing the module must have pinned the CPU platform before jax loads."""

    import os

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_assert_cpu_only_passes_on_a_cpu_process():
    result = real_state.assert_cpu_only()
    assert result["platforms"] == ["cpu"]


def test_stage_pair_fields_cover_the_prognostic_carry():
    """Every field whose difference feeds a WRF work prime must be in the list."""

    required = {
        "u", "v", "w", "theta", "qv",
        "p_total", "p_perturbation",
        "ph_total", "ph_perturbation",
        "mu_total", "mu_perturbation",
    }
    assert required <= set(real_state.STAGE_PAIR_FIELDS)


class _FakeState:
    """Minimal stand-in exposing the leaves and ``replace`` the helper uses."""

    def __init__(self, **fields):
        self._fields = dict(fields)
        for name, value in fields.items():
            setattr(self, name, value)

    def replace(self, **updates):
        merged = dict(self._fields)
        merged.update(updates)
        return _FakeState(**merged)


class _FakeSnapshot:
    def __init__(self, state, tag, valid_time="2026-07-26_00:00:00"):
        self.state = state
        self.provenance = {"wrfout": tag, "valid_times": [valid_time]}


def _pair(scale: float, *, prev_time="2026-07-26_00:00:00", now_time="2026-07-26_01:00:00"):
    shape = (3, 4, 5)
    base = np.linspace(1.0, 2.0, int(np.prod(shape))).reshape(shape)
    previous = _FakeSnapshot(
        _FakeState(**{name: base for name in real_state.STAGE_PAIR_FIELDS}), "prev", prev_time,
    )
    current = _FakeSnapshot(
        _FakeState(**{name: base * scale for name in real_state.STAGE_PAIR_FIELDS}), "now", now_time,
    )
    return previous, current


def test_stage_pair_produces_a_nonzero_increment():
    """The whole point: reference and advanced must actually differ."""

    previous, current = _pair(1.01)
    advanced, meta = real_state.stage_advanced_state(previous, current, 54.0)
    delta = np.asarray(advanced.u) - np.asarray(current.state.u)
    assert np.abs(delta).max() > 0.0
    assert meta["dt_s"] == 54.0
    assert "u" in meta["fields"]


def test_stage_pair_increment_is_scaled_to_one_timestep():
    """dt/3600 of the hourly difference, not the hourly difference itself."""

    previous, current = _pair(2.0)
    advanced, _ = real_state.stage_advanced_state(previous, current, 54.0)
    hourly = np.asarray(current.state.u) - np.asarray(previous.state.u)
    applied = np.asarray(advanced.u) - np.asarray(current.state.u)
    np.testing.assert_allclose(applied, hourly * (54.0 / 3600.0), rtol=1e-12)


def test_stage_pair_records_both_source_files():
    previous, current = _pair(1.05)
    _, meta = real_state.stage_advanced_state(previous, current, 54.0)
    assert meta["previous_wrfout"] == "prev"
    assert meta["current_wrfout"] == "now"
    assert "vacuous" in meta["why"]


def test_identical_snapshots_give_a_degenerate_pair():
    """Documents the trap: same state in, zero increment out.

    This is not a bug in the helper -- it is what happens when a caller passes
    the same snapshot twice, and it is exactly the condition the loader's
    two-history requirement exists to prevent. Pinning it keeps the failure mode
    visible if someone later 'simplifies' the caller.
    """

    previous, current = _pair(1.0)
    advanced, _ = real_state.stage_advanced_state(previous, current, 54.0)
    delta = np.asarray(advanced.theta) - np.asarray(current.state.theta)
    assert np.abs(delta).max() == 0.0


def test_theta_offset_matches_the_production_writer():
    """State.theta = THM + 300; reading T instead would be a silent DRY branch."""

    assert real_state.THETA_OFFSET_K == 300.0


@pytest.mark.parametrize("name", ["VAR", "OA1", "OL4", "C3F", "RDNW", "THM", "MUB"])
def test_required_wrfout_variables_are_requested(name):
    """Metrics, hybrid coefficients and GWD statics must all be pulled."""

    assert name in real_state.WRFOUT_VARIABLES
