from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG
from gpuwrf.coupling.driver import (
    DEFAULT_DT_S,
    DEFAULT_RADIATION_CADENCE_STEPS,
    MAX_LIFTED_DYCORE_DT_S,
    run_forecast_segment,
    sanitize_state_with_stats,
    validate_lifted_coupled_dt,
)


def test_driver_no_longer_contains_m6_s2_one_second_cap():
    source = Path("src/gpuwrf/coupling/driver.py").read_text(encoding="utf-8")

    assert "dycore_dt_s = min(float(dt_s), 1.0)" not in source
    assert "dycore_dt_s = validate_lifted_coupled_dt(dt_s)" in source
    assert DEFAULT_DT_S == 10.0
    assert MAX_LIFTED_DYCORE_DT_S == 12.0
    assert DEFAULT_RADIATION_CADENCE_STEPS == 60


def test_lifted_path_rejects_legacy_60s_coupled_dt_instead_of_capping():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)

    assert validate_lifted_coupled_dt(10.0) == 10.0
    with pytest.raises(ValueError, match="exceeds the M6-S5 Path-B dycore limit"):
        validate_lifted_coupled_dt(60.0)
    with pytest.raises(ValueError, match="exceeds the M6-S5 Path-B dycore limit"):
        run_forecast_segment(
            state,
            tendencies,
            grid,
            60.0,
            1,
            start_step=0,
            total_steps=1,
            n_acoustic=1,
            radiation_cadence_steps=10,
            final_radiation=False,
            boundary_config=DEFAULT_BOUNDARY_CONFIG,
        )


def test_sanitize_stats_count_nonfinite_and_clip_changes():
    grid = GridSpec.canary_3km_template()
    previous = State.zeros(grid).replace(theta=jnp.ones((10, 8, 8)) * 300.0, p=jnp.ones((10, 8, 8)) * 90000.0)
    candidate = previous.replace(
        theta=jnp.ones_like(previous.theta) * 100.0,
        p=previous.p.at[0, 0, 0].set(jnp.nan),
    )

    sanitized, stats = sanitize_state_with_stats(candidate, previous)

    assert float(jnp.min(sanitized.theta)) == 150.0
    assert float(sanitized.p[0, 0, 0]) == 90000.0
    assert int(stats.clip_count) >= previous.theta.size
    assert int(stats.nonfinite_count) >= 1
    assert int(stats.changed_count) == int(stats.clip_count + stats.nonfinite_count)
