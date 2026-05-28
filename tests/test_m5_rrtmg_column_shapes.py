from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column
from gpuwrf.validation.tier1_rrtmg import load_lw_fixture_state, load_sw_fixture_state


def test_rrtmg_sw_step_preserves_column_shapes_and_fp64_dtype():
    state, _ = load_sw_fixture_state()
    out = solve_rrtmg_sw_column(state, debug=False)
    assert out.heating_rate.shape == state.T.shape
    assert out.flux_down.shape[-1] == state.T.shape[-1] + 2
    assert out.flux_up.shape == out.flux_down.shape
    assert out.heating_rate.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(out.heating_rate)))


def test_rrtmg_lw_step_preserves_column_shapes_and_fp64_dtype():
    state, _ = load_lw_fixture_state()
    out = solve_rrtmg_lw_column(state, debug=False)
    assert out.heating_rate.shape == state.T.shape
    assert out.flux_down.shape[-1] == state.T.shape[-1] + 2
    assert out.flux_up.shape == out.flux_down.shape
    assert out.heating_rate.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(out.heating_rate)))


def test_rrtmg_hlo_diff_artifacts_are_raw_diffs_when_present():
    for path in (
        Path("artifacts/m5/hlo_dump/rrtmg_sw_debug_vs_stripped.diff"),
        Path("artifacts/m5/hlo_dump/rrtmg_lw_debug_vs_stripped.diff"),
    ):
        if path.exists():
            text = path.read_text(encoding="utf-8")
            assert text == "" or text.startswith("--- production\n+++ stripped\n")
