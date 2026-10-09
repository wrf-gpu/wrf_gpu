"""Held ozone persistence: exact software checkpoint evidence, no forecast claim."""
import json

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
from gpuwrf.runtime.restart_store import RestartStore


def carry_fixture(dtype):
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(grid).items()})
    state = state.ensure_conditional_leaves(mp_physics=8)
    shape = (grid.nz, grid.ny, grid.nx)
    ozone = None if dtype is None else jnp.asarray(
        np.linspace(.2e-6, 8.3e-6, np.prod(shape), dtype=dtype).reshape(shape))
    return grid, initial_operational_carry(state).replace(o3rad=ozone)


def ozone_exact(expected, actual):
    if expected is None:
        assert actual is None
        return
    assert actual is not None, 'held ozone was dropped'
    expected, actual = np.asarray(expected), np.asarray(actual)
    assert np.isfinite(expected).all() and np.isfinite(actual).all(), 'nonfinite held ozone'
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    assert expected.tobytes() == actual.tobytes(), 'held ozone bytes changed'


@pytest.mark.parametrize('dtype', (np.float32, np.float64, None))
@pytest.mark.parametrize('kind', ('wrfrst', 'restart_store'))
def test_nontrivial_held_ozone_roundtrip(tmp_path, dtype, kind):
    grid, carry = carry_fixture(dtype)
    if dtype is not None:
        assert np.unique(np.asarray(carry.o3rad)).size > 1
    if kind == 'wrfrst':
        path = tmp_path / 'ozone.nc'
        write_wrfrst_carry(carry, grid, {}, path, valid_time='2026-07-25_18:09:54',
                          run_start='2026-07-25_18:00:00', step_index=11)
        with Dataset(path) as ds:
            if dtype is not None:
                assert ds['GPUWRF_O3RAD_LEAF_000'].dtype == np.dtype(dtype)
            assert json.loads(ds.GPUWRF_OPTIONAL_CARRY_KIND)['o3rad'] == ('none' if dtype is None else 'pytree')
        restored, _ = read_wrfrst_carry(path)
    else:
        store = RestartStore(tmp_path / 'store', 10**8, 2, 0)
        generation, _ = store.save({'d01': carry}, {'d01': 11}, {'d01': 54.}, {})
        snapshot, _ = store.read(generation)
        restored = snapshot['carries']['d01']
    ozone_exact(carry.o3rad, restored.o3rad)
    assert jax.tree.structure(carry) == jax.tree.structure(restored)


def test_wrfrst_missing_ozone_payload_fails_closed(tmp_path):
    grid, carry = carry_fixture(np.float32)
    path = tmp_path / 'old_ozone.nc'
    write_wrfrst_carry(carry, grid, {}, path, valid_time='2026-07-25_18:09:54',
                      run_start='2026-07-25_18:00:00', step_index=11)
    with Dataset(path, 'r+') as ds:
        for attr in ('GPUWRF_OPTIONAL_CARRY_FIELD_ORDER', 'GPUWRF_OPTIONAL_CARRY_KIND'):
            payload = json.loads(ds.getncattr(attr))
            payload.pop('o3rad')
            ds.setncattr(attr, json.dumps(payload))
    with pytest.raises(ValueError, match='missing groups.*o3rad'):
        read_wrfrst_carry(path)


@pytest.mark.parametrize('role', ('candidate', 'reference', 'both'))
@pytest.mark.parametrize('bad', (np.nan, np.inf, -np.inf))
def test_ozone_comparison_rejects_nonfinite(role, bad):
    expected = np.array([.2e-6, 8.3e-6], dtype=np.float32)
    actual = expected.copy()
    if role != 'reference':
        actual[-1] = bad
    if role != 'candidate':
        expected[-1] = bad
    with pytest.raises(AssertionError, match='nonfinite held ozone'):
        ozone_exact(expected, actual)
