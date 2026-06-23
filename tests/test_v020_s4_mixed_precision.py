from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.runtime.operational_mode import _advance_chunk, _initial_carry_for_run


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _assert_s4_mixed_dtypes(carry) -> None:
    assert carry.base_state is not None
    state = carry.state
    for name in ("p_perturbation", "ph_perturbation", "mu_perturbation", "w"):
        assert getattr(state, name).dtype == jnp.float32, name
    for name in ("p_total", "ph_total", "mu_total"):
        assert getattr(state, name).dtype == jnp.float64, name
    for name in ("pb", "phb", "mub"):
        assert getattr(carry.base_state, name).dtype == jnp.float64, name


def test_s4_mixed_mode_downcasts_only_authorized_acoustic_carry():
    setup = build_warm_bubble_setup(require_gpu=False)
    nml = replace(setup.namelist, acoustic_precision_mode="mixed_perturb_fp32_v020")

    carry = _initial_carry_for_run(setup.state, nml)

    _assert_s4_mixed_dtypes(carry)
    assert carry.state.theta.dtype == jnp.float64
    assert carry.state.u.dtype == jnp.float64
    assert carry.state.v.dtype == jnp.float64


def test_s4_mixed_mode_stays_mixed_after_one_hot_step():
    setup = build_warm_bubble_setup(require_gpu=False)
    nml = replace(setup.namelist, acoustic_precision_mode="mixed_perturb_fp32_v020")
    carry = _initial_carry_for_run(setup.state, nml)
    cadence = max(1, int(nml.radiation_cadence_steps))

    out = _advance_chunk(carry, nml, jnp.asarray(1, dtype=jnp.int32), n_steps=1, cadence=cadence)

    _assert_s4_mixed_dtypes(out)


def test_global_fp32_audit_mode_does_not_force_jax_x64_on_import():
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "GPUWRF_FP32_MODE": "global_fp32",
        "JAX_ENABLE_X64": "false",
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": str(_repo_root() / "src"),
    })
    code = (
        "import jax; "
        "import gpuwrf; "
        "import gpuwrf.runtime.operational_mode; "
        "print(jax.config.read('jax_enable_x64'))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().splitlines()[-1] == "False"
