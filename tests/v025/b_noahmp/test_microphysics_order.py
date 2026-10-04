"""Operational call-order regressions; mocks are NOT WRF fidelity evidence."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry


def _fixture():
    path = Path(__file__).resolve().parents[2] / "test_v013_operational_smoke.py"
    spec = importlib.util.spec_from_file_location("order_smoke_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grid = module._grid(nz=8)
    state = module._base_state(grid).replace(Nr=jnp.ones((8, 4, 4)))
    nml = module._namelist(
        grid, dt_s=6.0, mp_physics=8, bl_pbl_physics=0,
        sf_sfclay_physics=0, cu_physics=0, ra_sw_physics=0,
        ra_lw_physics=0, run_boundary=False, disable_guards=True)
    return initial_operational_carry(state), nml


@pytest.mark.parametrize("enabled", [False, True, None])
def test_two_step_dispatch_uses_post_transport_operands(monkeypatch, enabled):
    """Expose the subtraction defect and check the actual operational seam."""
    if enabled is None:
        monkeypatch.delenv("GPUWRF_MICROPHYSICS_WRF_ORDER", raising=False)
    else:
        monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER", str(int(enabled)))
    active_order = enabled is not False
    events = []

    def microphysics(state, dt):
        events.append(("mp", np.asarray(state.Nr).copy()))
        return state.replace(Nr=state.Nr * 0.5)

    def dynamics(carry, nml, **kw):
        events.append(("rk", np.asarray(carry.state.Nr).copy()))
        return carry.replace(state=carry.state.replace(Nr=carry.state.Nr * 0.25))

    monkeypatch.setattr(op, "thompson_adapter", microphysics)
    monkeypatch.setattr(op, "_rk_scan_step", dynamics)
    carry, nml = _fixture()
    for step in (1, 2):
        before = np.asarray(carry.state.Nr).copy()
        carry, _ = op._physics_boundary_step_with_limiter_diagnostics(
            carry, nml, jnp.int32(step), run_radiation=False)
        if active_order:
            np.testing.assert_array_equal(events[-1][1], before * 0.25)
            np.testing.assert_array_equal(np.asarray(carry.state.Nr), before * 0.125)
        else:
            np.testing.assert_array_equal(events[-2][1], before)
            np.testing.assert_array_equal(np.asarray(carry.state.Nr), before * -0.25)
    assert [name for name, _ in events] == (
        ["rk", "mp", "rk", "mp"] if active_order else ["mp", "rk", "mp", "rk"])


def test_post_microphysics_disabled_with_physics(monkeypatch):
    carry, nml = _fixture()
    monkeypatch.setattr(op, "thompson_adapter", lambda *a: pytest.fail("unexpected MP"))
    assert op._apply_post_rk_microphysics(carry.state, replace(nml, run_physics=False)) is carry.state
