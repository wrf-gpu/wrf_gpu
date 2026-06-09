from __future__ import annotations

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State
from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.runtime.operational_mode import (
    _rk_scan_step,
    _rk_scan_step_with_pre_halo_capture,
    run_forecast_operational,
    run_forecast_operational_segmented,
    run_forecast_operational_single_scan,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


def _fixture():
    setup = build_warm_bubble_setup(require_gpu=False)
    namelist = dataclasses.replace(
        setup.namelist,
        run_physics=False,
        run_boundary=False,
        const_nu_m2_s=0.0,
        diff_6th_opt=0,
        km_opt=0,
        dt_s=0.1,
        acoustic_substeps=1,
        disable_guards=True,
    )
    return initial_operational_carry(setup.state), namelist


def _assert_same_tree(left, right) -> None:
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for lhs, rhs in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def test_pre_halo_capture_is_default_off_and_preserves_normal_rk_return() -> None:
    carry, namelist = _fixture()
    lead_seconds = jnp.asarray(0.0, dtype=jnp.float64)

    normal = _rk_scan_step(carry, namelist, lead_seconds=lead_seconds)
    captured = _rk_scan_step_with_pre_halo_capture(carry, namelist, lead_seconds=lead_seconds)
    jax.block_until_ready(captured.carry.state.theta)

    assert isinstance(normal, OperationalCarry)
    assert isinstance(captured.carry, OperationalCarry)
    assert isinstance(captured.pre_halo_state, State)
    _assert_same_tree(normal, captured.carry)

    for leaf in jax.tree_util.tree_leaves(captured.pre_halo_state):
        arr = np.asarray(leaf)
        if np.issubdtype(arr.dtype, np.floating):
            assert np.all(np.isfinite(arr))


def test_normal_forecast_public_signatures_do_not_expose_capture() -> None:
    expected = ["state", "namelist", "hours"]
    for fn in (
        run_forecast_operational,
        run_forecast_operational_segmented,
        run_forecast_operational_single_scan,
    ):
        params = list(inspect.signature(fn).parameters)
        assert params[:3] == expected
        assert all("capture" not in name and "pre_halo" not in name for name in params)
