"""BP55 C1 (review-s2small): the product initial carry seeds KF held rates/W0AVG/NCA REAL under the native flag."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import pytest

ROOT = Path(__import__("gpuwrf").__file__).resolve().parents[2]


@pytest.mark.parametrize("flag", ("0", "1"))
def test_initial_carry_kf_leaves_follow_the_native_flag(monkeypatch, flag):
    from gpuwrf.runtime import operational_mode as op

    import jax
    from gpuwrf.contracts import state as state_contract

    monkeypatch.setenv("GPUWRF_KF_COLUMN_FP32", flag)
    # CPU placement for State.zeros inside the initializer (E126); shapes/dtypes unchanged.
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
    spec = importlib.util.spec_from_file_location("kf_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    grid = fixture._grid(ny=2, nx=2, nz=8)
    nml = replace(op.OperationalNamelist.from_grid(grid, force_fp64=True), cu_physics=1)
    carry = op._initial_carry_for_run(fixture._state(grid), nml)
    leaves = (*carry.cumulus_carry, *(carry.cumulus_tendencies or ()))
    want = jnp.float32 if flag == "1" else jnp.float64
    assert carry.cumulus_carry[0].dtype == carry.cumulus_carry[1].dtype == want
    if flag == "1":
        assert carry.cumulus_tendencies is not None and all(leaf.dtype == jnp.float32 for leaf in leaves), \
            [leaf.dtype for leaf in leaves]
