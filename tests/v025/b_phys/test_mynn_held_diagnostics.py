"""MYNN history producers, their held-State seam, and required restart schema."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES, MYNN_DIAGNOSTIC_LEAVES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.io.restart import read_restart, write_restart
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
from gpuwrf.io.wrfout_writer import _full_source_value
from gpuwrf.physics import mynn_pbl as P
from gpuwrf.runtime import operational_mode as O
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore


def fixture():
    base = GridSpec.canary_3km_template()
    grid = replace(base, projection=replace(base.projection, nx=3, ny=2),
                   terrain=replace(base.terrain, shape=(2, 3)),
                   terrain_height=jnp.zeros((2, 3)), metrics=None)
    state = State(**{name: jnp.ones(shape, DEFAULT_DTYPES.dtype_for(name))
                     for name, shape in _state_field_shapes(grid).items()
                     if name not in MYNN_DIAGNOSTIC_LEAVES})
    nl = O.OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics,
                              dt_s=54., acoustic_substeps=4)
    return grid, nl, state


def oracle_module():
    path = Path(__file__).resolve().parents[2] / 'test_mynn_edmf_oracle.py'
    spec = importlib.util.spec_from_file_location('bp70_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('path', ['legacy', 'native', 'sums'])
def test_plume_history_matches_pristine_wrf(path, monkeypatch):
    oracle = oracle_module()
    monkeypatch.setenv('GPUWRF_EDMF_FUSED_PLUME', '0')
    if path != 'legacy':
        from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
        monkeypatch.setattr(__import__('gpuwrf.physics.mynn_edmf', fromlist=['']),
                            'dmp_mf_columns', dmp_mf_columns_native)
        monkeypatch.setenv('GPUWRF_MYNN_PLUME_SUMS', '1' if path == 'sums' else '0')
    column = json.loads(Path(oracle.COL).read_text())
    got = oracle._dmp_result(column)
    ref = json.loads((Path(__file__).parent / 'fixtures/mynn_history_wrf.json').read_text())
    for name, expected in ref['expected'].items():
        actual = float(got[name][0])
        assert np.isfinite(actual)
        assert abs(actual - expected) <= ref['tolerance'] * abs(expected), (path, name, actual, expected)
    # Existing frozen transport gate rides with the diagnostic check.
    oracle.test_dmp_mf_matches_wrf_oracle(column)


@pytest.mark.parametrize('native', [False, True])
def test_column_retains_the_one_existing_plume_call(native, monkeypatch):
    oracle = oracle_module()
    column = json.loads(Path(oracle.COL).read_text())
    state, surface, _ = oracle._build_state(column)
    mf = oracle._dmp_result(column)
    mf.update(maxmf=jnp.array([-1.25]), maxwidth=jnp.array([321.]), ztop_plume=jnp.array([456.]))
    if native:
        mf = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), mf)
    calls = []
    def plume(*args, **kwargs):
        calls.append(1)
        return mf
    monkeypatch.setattr(P, '_edmf_arrays_from_state', plume)
    from gpuwrf.kernels import phys_mynn_columns as native_module
    monkeypatch.setattr(native_module, '_native_edmf_arrays', plume)
    monkeypatch.setenv('GPUWRF_MYNN_FP32_COLUMNS', '1' if native else '0')
    P.step_mynn_pbl_column_with_pblh.clear_cache()
    out, _ = P.step_mynn_pbl_column_with_pblh(state, column['config']['delt'],
                                            surface=surface, edmf=True, dx=1000.)
    assert calls == [1]
    for name in ('maxmf', 'maxwidth', 'ztop_plume'):
        np.testing.assert_array_equal(np.asarray(getattr(out, name)), np.asarray(mf[name], dtype=getattr(out, name).dtype))
    assert np.isfinite(np.asarray(out.el)).all() and np.asarray(out.el).max() > 0
    P.step_mynn_pbl_column_with_pblh.clear_cache()


def test_adapter_rk_hold_and_writer_consume_actual_mynn_outputs(monkeypatch):
    grid, _, state = fixture()
    ny, nx = state.t_skin.shape
    nz = state.theta.shape[0]
    volume = jnp.arange(ny * nx * nz, dtype=jnp.float32).reshape(ny, nx, nz) + 2.
    surface = jnp.arange(ny * nx, dtype=jnp.float32).reshape(ny, nx) + 3.
    calls = []
    def column_step(column, *args, **kwargs):
        calls.append(1)
        return column.replace(el=volume.reshape(ny * nx, nz), maxmf=-surface.ravel(),
                              maxwidth=(surface * 100).ravel(), ztop_plume=(surface * 50).ravel()), jnp.zeros(ny * nx)
    monkeypatch.setattr(C, 'step_mynn_pbl_column_with_pblh', column_step)
    produced = C.mynn_adapter_with_source_leaves(state, 54., None, restart=True).state
    expected = dict(el_pbl=volume.transpose(2, 0, 1), maxmf=-surface,
                    maxwidth=surface * 100, ztop_plume=surface * 50)
    # RK's non-dry path must replace these diagnostics, not add a state delta.
    dynamics = state.replace(**{name: jnp.full_like(getattr(state, name), 900.) for name in expected})
    held = jax.jit(O._apply_physics_non_dry_updates)(dynamics, state, produced)
    for name, value in expected.items():
        np.testing.assert_array_equal(np.asarray(getattr(held, name)), value)
        assert getattr(held, name).dtype == jnp.float32
        writer_value = _full_source_value(name.upper(), value.shape, dtype=np.float32,
                                         diagnostics=None, state=held, land_state=None,
                                         grid=grid, namelist=None)
        np.testing.assert_array_equal(writer_value, value)
    assert calls == [1]  # writer reads the retained leaves without a solve


def test_wrf_t0_zero_and_force_fp64_retains_real_diagnostics():
    _, _, state = fixture()
    wide = O._enforce_operational_precision(state, force_fp64=True)
    for name in MYNN_DIAGNOSTIC_LEAVES:
        leaf = getattr(wide, name)
        assert leaf.dtype == jnp.float32 and not np.asarray(leaf).any()
        assert leaf.shape == (state.theta.shape if name == 'el_pbl' else state.t_skin.shape)


@pytest.mark.parametrize('format', ['pickle', 'netcdf', 'store'])
@pytest.mark.parametrize('old_schema', [False, True])
def test_held_mynn_restart_roundtrip_or_e78_refusal(format, old_schema, tmp_path):
    grid, nl, state = fixture()
    state = state.replace(**{name: jnp.arange(getattr(state, name).size, dtype=jnp.float32)
                             .reshape(getattr(state, name).shape) + i + .25
                             for i, name in enumerate(MYNN_DIAGNOSTIC_LEAVES)})
    carry = initial_operational_carry(state)
    def strip(payload):
        payload['state_field_order'] = [name for name in payload['state_field_order'] if name not in MYNN_DIAGNOSTIC_LEAVES]
        for name in MYNN_DIAGNOSTIC_LEAVES:
            payload['state_fields'].pop(name)
    if format == 'pickle':
        path = tmp_path / 'carry.pkl'
        write_restart(carry, nl, grid, 11, path)
        if old_schema:
            data = pickle.loads(path.read_bytes()); strip(data['carry']); path.write_bytes(pickle.dumps(data))
        read = lambda: read_restart(path)[0].state
    elif format == 'netcdf':
        path = tmp_path / 'carry.nc'
        write_wrfrst_carry(carry, grid, {}, path, valid_time='2026-07-25_18:09:54',
                          run_start='2026-07-25_18:00:00', step_index=11)
        if old_schema:
            with netCDF4.Dataset(path, 'r+') as nc:
                order = json.loads(nc.GPUWRF_STATE_FIELD_ORDER)
                nc.GPUWRF_STATE_FIELD_ORDER = json.dumps([name for name in order if name not in MYNN_DIAGNOSTIC_LEAVES])
        read = lambda: read_wrfrst_carry(path)[0].state
    else:
        store = RestartStore(tmp_path / 'store', 10**8, 2, 0)
        path, _ = store.save({'d01': carry}, {'d01': 11}, {'d01': 54.}, {})
        if old_schema:
            data = pickle.loads((path / 'snapshot.pkl').read_bytes()); strip(data['carries']['d01'])
            blob = pickle.dumps(data); (path / 'snapshot.pkl').write_bytes(blob)
            manifest = json.loads((path / 'manifest.json').read_text())
            manifest.update(payload_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            (path / 'manifest.json').write_text(json.dumps(manifest))
        def read():
            domain = store.read(path, device=not old_schema)[0]['carries']['d01']
            return domain if old_schema else domain.state
    if old_schema:
        with pytest.raises(ValueError, match='missing non-conditional leaves'):
            read()
    else:
        restored = read()
        for name in MYNN_DIAGNOSTIC_LEAVES:
            left, right = np.asarray(getattr(state, name)), np.asarray(getattr(restored, name))
            assert (left.dtype, left.shape, left.tobytes()) == (right.dtype, right.shape, right.tobytes())
