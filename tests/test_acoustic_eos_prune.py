"""Pressure operand lifetime and cache controls; physics is gated separately."""
import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gpuwrf.dynamics.core.acoustic import AcousticCoreState
from gpuwrf.kernels import dyn_acoustic_fp32 as da
from gpuwrf.runtime import aot_cheap_key as key


@pytest.mark.parametrize("flag, enabled", [(None, False), ("0", False), ("1", True)])
def test_pressure_prune_is_explicit_default_off(flag, enabled):
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    if flag is None:
        env.pop("GPUWRF_ACOUSTIC_EOS_PRUNE", None)
    else:
        env["GPUWRF_ACOUSTIC_EOS_PRUNE"] = flag
    code = "from gpuwrf.kernels.dyn_acoustic_fp32 import _EOS_PRUNE; print(int(_EOS_PRUNE))"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, check=True, timeout=30)
    assert result.stdout.strip() == str(int(enabled))


def test_pressure_flag_resolved_value_is_in_executable_key(monkeypatch):
    monkeypatch.setattr(da, "_EOS_PRUNE", False)
    original = key.module_const_env_hash()
    monkeypatch.setattr(da, "_EOS_PRUNE", True)
    assert key.module_const_env_hash() != original


def test_prune_retains_every_actual_pressure_read_and_alias_target():
    # The read set comes from the kernel body, independently of its selector.
    reads = set()
    for node in ast.walk(ast.parse(inspect.getsource(da._pressure_kernel))):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "r" and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            reads.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "l3" and node.args
                and isinstance(node.args[0], ast.Constant)):
            reads.add(node.args[0].value)
    assert reads - {"smdiv"} <= set(da._EOS_INPUTS)
    assert set(da._POINTWISE_ALIASES["b_core_pressure_fp32"]) <= set(da._EOS_INPUTS)


@pytest.mark.parametrize("enabled", [False, True])
def test_real_acoustic_caller_filters_only_pressure_operands(monkeypatch, enabled):
    # Isolate the product caller from the numerical stages; this is a protocol
    # regression test, not a WRF-fidelity or runtime-performance claim.
    shape = jax.ShapeDtypeStruct((4, 3, 5), np.float32)
    values = {name: shape for name in AcousticCoreState.__annotations__}
    state = SimpleNamespace(**values)
    state.to_dict = lambda: values
    state.replace = lambda **updates: state
    cfg = SimpleNamespace(dt=1., dx=1., dy=1., epssm=.1)
    calls = {}

    def call(kernel, inputs, outputs, grid, *, name, **kwargs):
        calls[name] = inputs
        return {n: values.get(n, shape) for n in outputs}

    monkeypatch.setattr(da, "_EOS_PRUNE", enabled)
    monkeypatch.setattr(da, "_check", lambda *a, **kw: None)
    monkeypatch.setattr(da, "_call", call)
    monkeypatch.setattr(da, "_w_phase", lambda s, *a, **kw: s)
    da.acoustic_substep_fp32(state, coefficients={}, cfg=cfg)
    assert "t_2ave" in calls["b_core_mass_fp32"]
    assert "u" in calls["b_core_uv_fp32"]
    pressure = calls["b_core_pressure_fp32"]
    if enabled:
        assert "t_2ave" not in pressure and "u" not in pressure
        assert set(pressure) == set(da._EOS_INPUTS) | {"smdiv"}
    else:
        assert set(pressure) == set(values) | {"smdiv"}
