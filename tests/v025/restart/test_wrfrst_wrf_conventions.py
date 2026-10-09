"""WRF inspection conventions from an original P0227 CPU-WRF restart.

This tests the export surface and exact GPU checkpoint, not native WRF resume.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
from gpuwrf.runtime.operational_state import initial_operational_carry

FIXTURE = Path(__file__).with_name("fixtures") / "p0227_tau48_d03_restart.npz"
# Measured CPU self-consistency of THM_2/qv -> dry T on this frozen REAL slab.
# Two ulps of full theta near 300 K; never a trajectory/fidelity tolerance.
CPU_DRY_BRIDGE_ATOL_K = 2.0**-14
CURRENT_LEAVES = {"U_2": "u", "V_2": "v", "W_2": "w",
                  "PH_2": "ph_perturbation", "MU_2": "mu_perturbation"}


def finite_equal(expected, actual, *, atol=0.0, exact=False):
    expected, actual = np.asarray(expected), np.asarray(actual)
    assert np.isfinite(expected).all() and np.isfinite(actual).all(), "nonfinite restart comparison"
    assert expected.shape == actual.shape
    if exact:
        assert expected.dtype == actual.dtype
        assert expected.tobytes() == actual.tobytes(), "exact GPU restart bytes changed"
    else:
        assert np.max(np.abs(expected - actual)) <= atol


def real_carry(dtype):
    grid = GridSpec.canary_3km_template()
    with np.load(FIXTURE) as source:
        cpu = {name: source[name].copy() for name in source.files}
    state = State(**{name: jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(grid).items()})
    state = state.ensure_conditional_leaves(mp_physics=8)
    actual = {name: getattr(state, name) for name in State.__slots__}
    for name, leaf in CURRENT_LEAVES.items():
        actual[leaf] = jnp.asarray(cpu[name], dtype=dtype)
    actual["theta"] = jnp.asarray(cpu["THM_2"].astype(dtype) + dtype(300.0))
    actual["qv"] = jnp.asarray(cpu["QVAPOR"], dtype=dtype)
    state = State.tree_unflatten(None, tuple(actual[name] for name in State.__slots__))
    return grid, initial_operational_carry(state), cpu


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("use_theta_m", (0, 1))
def test_real_cpu_restart_temperature_slots_and_exact_gpu_payload(tmp_path, dtype, use_theta_m):
    receipt = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == receipt["fixture_sha256"]
    grid, carry, cpu = real_carry(dtype)
    for name, value in cpu.items():
        assert hashlib.sha256(value.tobytes()).hexdigest() == receipt["fields"][name]["array_sha256"]
        assert np.isfinite(value).all()
    assert np.any(cpu["LANDMASK"] == 0) and np.any(cpu["LANDMASK"] == 1)
    assert np.mean(cpu["THM_2"] - cpu["T"]) > 3.0, "fixture must expose moist/dry error"
    path = tmp_path / "gpu_checkpoint.nc"
    write_wrfrst_carry(carry, grid, {"use_theta_m": use_theta_m}, path,
                      valid_time="2026-03-02_00:00:00", run_start="2026-02-28_00:00:00",
                      step_index=3200)
    with Dataset(path) as ds:
        assert "THM_2" in ds.variables, "missing current potential-temperature slot"
        finite_equal(cpu["T"], ds["T"][0], atol=CPU_DRY_BRIDGE_ATOL_K)
        finite_equal(cpu["THM_2"] if use_theta_m else cpu["T"], ds["THM_2"][0],
                     atol=CPU_DRY_BRIDGE_ATOL_K)
        assert ds.USE_THETA_M == use_theta_m
        assert "WRF-COMPATIBLE" not in ds.TITLE
        assert ds.GPUWRF_RESTART_SCOPE.startswith("GPUWRF_ONLY:")
        for name, leaf in CURRENT_LEAVES.items():
            assert name in ds.variables, f"missing current restart slot {name}"
            spec = receipt["fields"][name]
            assert list(ds[name].dimensions) == spec["dimensions"]
            for attr in ("stagger", "units", "MemoryOrder"):
                assert getattr(ds[name], attr) == spec[attr]
            finite_equal(np.asarray(getattr(carry.state, leaf)), ds[name][0], exact=True)
            assert name in json.loads(ds.GPUWRF_STANDARD_RESTART_VARIABLES)
            assert name.removesuffix("_2") + "_1" not in ds.variables
        assert list(ds["T"].dimensions) == receipt["fields"]["T"]["dimensions"]
        # Inspection fields cannot replace the exact authoritative payload.
    with Dataset(path, "r+") as ds:
        for name in ("T", "THM_2", *CURRENT_LEAVES):
            ds[name][:] = np.float32(-123.0)
    restored, _ = read_wrfrst_carry(path)
    assert jax.tree.structure(carry) == jax.tree.structure(restored)
    for expected, actual in zip(jax.tree.leaves(carry), jax.tree.leaves(restored), strict=True):
        finite_equal(expected, actual, exact=True)


@pytest.mark.parametrize("role", ("candidate", "reference", "both"))
@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("exact", (False, True))
def test_restart_comparison_rejects_nonfinite(role, bad, exact):
    expected = np.array([300.0, 301.0], dtype=np.float32)
    actual = expected.copy()
    if role != "reference":
        actual[-1] = bad
    if role != "candidate":
        expected[-1] = bad
    with pytest.raises(AssertionError, match="nonfinite restart comparison"):
        finite_equal(expected, actual, exact=exact)
