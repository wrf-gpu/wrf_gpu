"""fid-q2 RE03 G5: the held nest ``o3rad`` survives a restart (WRF Registry o3rad IO 'r'; CPU).

The product RestartStore serializes every OperationalCarry field (io.restart v2+ extra_fields), so the
forced nest ozone resumes exactly.  The WRF-NetCDF carry writer has no O3RAD variable yet: it must refuse a
populated o3rad loudly instead of dropping it.  A run/resume flag mismatch is refused by the restart identity
(GPUWRF_NEST_O3_FROM_PARENT is part of the trace-environment hash).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.io.restart import _CARRY_FIELD_ORDER, read_restart, write_restart
from gpuwrf.io.wrfrst_netcdf import UNSUPPORTED_CARRY_FIELDS, write_wrfrst_carry
from gpuwrf.nesting.nest_o3 import NEST_O3_FROM_PARENT_ENV
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry


@pytest.fixture
def carry_with_o3():
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.asarray(np.arange(np.prod(shape)).reshape(shape) / 991 + 1,
                                       dtype=DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(grid).items()})
    o3 = jnp.asarray(np.linspace(2.0e-8, 9.0e-6, state.theta.size, dtype=np.float32).reshape(state.theta.shape))
    carry = initial_operational_carry(state).replace(o3rad=o3)
    namelist = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=54, acoustic_substeps=4)
    return grid, namelist, carry


def test_pickle_restart_roundtrips_o3rad_bitwise(carry_with_o3, tmp_path):
    grid, namelist, carry = carry_with_o3
    assert "o3rad" in _CARRY_FIELD_ORDER
    path = tmp_path / "carry.pkl"
    write_restart(carry, namelist, grid, 7, path)
    restored, _, _, step = read_restart(path)
    assert step == 7
    got = np.asarray(restored.o3rad)
    assert got.dtype == np.float32 and got.tobytes() == np.asarray(carry.o3rad).tobytes()
    assert jax.tree.structure(restored) == jax.tree.structure(carry)
    # deletion-sensitive: a carry without the leaf restores None, not a silently re-seeded field
    write_restart(carry.replace(o3rad=None), namelist, grid, 7, path)
    assert read_restart(path)[0].o3rad is None


def test_netcdf_carry_writer_refuses_populated_o3rad(carry_with_o3, tmp_path):
    grid, _namelist, carry = carry_with_o3
    assert "o3rad" in UNSUPPORTED_CARRY_FIELDS
    with pytest.raises(ValueError, match="o3rad"):
        write_wrfrst_carry(carry, grid, {}, tmp_path / "carry.nc", valid_time="2026-07-25_18:00:54",
                           run_start="2026-07-25_18:00:00", step_index=1)


def test_restart_identity_splits_on_the_flag(monkeypatch):
    from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash

    monkeypatch.delenv(NEST_O3_FROM_PARENT_ENV, raising=False)
    off = global_trace_env_hash()
    monkeypatch.setenv(NEST_O3_FROM_PARENT_ENV, "1")
    assert global_trace_env_hash() != off
    assert os.environ[NEST_O3_FROM_PARENT_ENV] == "1"
