"""GPUWRF_DYN_GLUE_FUSED part parsing and the nested-only large-step u/v kernel dispatch (uvn).

Fidelity of the kernel itself is the pristine REAL4 gate in probe_dyn_real.py (uv family:
horizontal_pressure_gradient + coriolis + curvature, lid on/off, cq mutant). These tests pin
the dispatch: parts parse, 'uvn' enables the fused kernel on NESTED children only (the root
keeps the XLA path: the fused kernel flips the root step layout), 'pin' adds the row-major
operand constraint, and the default (unset) path is untouched.
"""
import inspect

import pytest

from gpuwrf.dynamics.core import rk_addtend_dry as rk
from gpuwrf.kernels import dyn_real_fp32 as real
from gpuwrf.runtime import operational_mode as op


@pytest.mark.parametrize("value,parts", [
    ("0", set()), ("", set()), ("1", {"mom", "uv"}), ("uvn_pin", {"uvn", "pin"}),
    ("mom_omega_rhsph_uv2", {"mom", "omega", "rhsph", "uv2"}),
])
def test_glue_parts(monkeypatch, value, parts):
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", value)
    assert real.glue_parts() == frozenset(parts)


@pytest.mark.parametrize("value,nested_only", [("uvn_pin", True), ("uvn", True), ("uv", False), ("uv_uvn", False), ("0", False)])
def test_nested_only_predicate(monkeypatch, value, nested_only):
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", value)
    assert rk.large_step_uv_nested_only() is nested_only


def test_pin_rows_only_with_pin_part(monkeypatch):
    import jax.numpy as jnp
    x = jnp.ones((2, 3, 4), jnp.float32)
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "uvn")
    assert real.pin_rows((x,))[0] is x
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "uvn_pin")
    assert real.pin_rows((x,))[0] is not x


def test_augment_gates_fused_uv_on_nested_children():
    source = inspect.getsource(op._augment_large_step_tendencies)
    assert "large_step_uv_fused_enabled(u_t, v_t, base_state) and (" in source
    assert "_acoustic_lateral_bc_flags(namelist)[2] or not large_step_uv_nested_only()" in source
