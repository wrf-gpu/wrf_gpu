"""CPU-WRF import ABI against actual native X4 carry subsets (not fidelity)."""
import gzip
import hashlib
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.validation.wrf_restart_hydration import hydrate_wrfrst, load_restart_templates

FIXTURE = Path(__file__).with_name('fixtures') / 'shadow_dtype_native_cpu.pkl.gz'


@pytest.fixture(autouse=True)
def real_import_x64():
    with jax.enable_x64(True):
        yield


def assert_all_leaves(expected, actual, *, exact=False):
    assert jax.tree.structure(expected) == jax.tree.structure(actual), 'carry treedef changed'
    for index, (left, right) in enumerate(zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True)):
        left, right = np.asarray(left), np.asarray(right)
        assert np.isfinite(left).all() and np.isfinite(right).all(), 'nonfinite carry operand'
        assert left.shape == right.shape, f'carry shape changed at leaf {index}'
        assert left.dtype == right.dtype, f'carry dtype changed at leaf {index}: {left.dtype} / {right.dtype}'
        if exact:
            assert left.tobytes() == right.tobytes(), f'carry bytes changed at leaf {index}'


@pytest.fixture
def real_records():
    receipt = json.loads(FIXTURE.with_suffix('').with_suffix('.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == receipt['fixture_sha256']
    with gzip.open(FIXTURE, 'rb') as stream:
        records = pickle.load(stream)  # Trusted repository fixture, not external input.
    assert set(records) == {'d01', 'd02', 'd03'}
    return records


def namelist_for(cpu):
    return OperationalNamelist(grid=None, tendencies=None, metrics=None,
        dt_s=cpu.dt_s, mp_physics=cpu.physics['MP_PHYSICS'],
        cu_physics=cpu.physics['CU_PHYSICS'], ra_lw_physics=cpu.physics['RA_LW_PHYSICS'],
        ra_sw_physics=cpu.physics['RA_SW_PHYSICS'], radiation_cadence_steps=cpu.radiation_steps,
        cumulus_cadence_steps=int(cpu.values['STEPCU']))


def capture_import_seeds(monkeypatch, records):
    # Exercise the real saved-radiation hook. Geometry/model stepping is outside
    # this ABI gate; the loader returns the real native reference with that seed.
    def loader(_config, names):
        result = {}
        for name in names:
            native, cpu, _grid = records[name]
            _held, diagnostics = pipeline.noahmp_initial_rad(
                native.state, namelist_for(cpu), _with_diagnostics=True)
            result[name] = native.replace(radiation_diagnostics=diagnostics)
        return result
    monkeypatch.setattr(pipeline, '_load_domains', loader)
    return load_restart_templates(None, {name: row[1] for name, row in records.items()})


@pytest.mark.parametrize('domain', ('d01', 'd02', 'd03'))
def test_saved_radiation_carry_matches_all_native_leaf_types_before_hydration(monkeypatch, real_records, domain):
    seeds = capture_import_seeds(monkeypatch, real_records)
    native, _cpu, _grid = real_records[domain]
    assert native.radiation_diagnostics.shadow_mask.dtype == np.int32
    assert_all_leaves(native, seeds[domain])


@pytest.mark.parametrize('domain', ('d01', 'd02', 'd03'))
@pytest.mark.parametrize('bad_legacy_mask', (False, True))
@pytest.mark.parametrize('storage', ('wrfrst', 'restart_store'))
def test_hydrated_carry_all_native_types_and_exact_serializers(
        monkeypatch, real_records, domain, bad_legacy_mask, storage, tmp_path):
    seeds = capture_import_seeds(monkeypatch, real_records)
    native, cpu, grid = real_records[domain]
    seeded = seeds[domain]
    if bad_legacy_mask:
        # The genuine pre-fix seed type: verifies the reset independently even
        # when the corrected first producer would otherwise conceal its deletion.
        seeded = seeded.replace(radiation_diagnostics=seeded.radiation_diagnostics._replace(
            shadow_mask=jnp.asarray(seeded.radiation_diagnostics.shadow_mask, dtype=jnp.float32)))
    imported, _phase, _audit = hydrate_wrfrst(seeded, cpu, namelist_for(cpu))
    assert_all_leaves(native, imported)
    if storage == 'wrfrst':
        path = tmp_path / 'import.nc'
        write_wrfrst_carry(imported, grid, namelist_for(cpu), path, valid_time=cpu.stamp,
                          run_start=cpu.epoch, step_index=cpu.own_steps)
        restored, _ = read_wrfrst_carry(path)
    else:
        store = RestartStore(tmp_path / 'store', 10**8, 2, 0)
        generation, _ = store.save({domain: imported}, {domain: cpu.own_steps},
                                  {domain: cpu.dt_s}, {})
        snapshot, _ = store.read(generation)
        restored = snapshot['carries'][domain]
    assert_all_leaves(imported, restored, exact=True)


@pytest.mark.parametrize('role', ('candidate', 'reference', 'both'))
@pytest.mark.parametrize('bad', (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize('exact', (False, True))
def test_all_leaf_comparison_rejects_nonfinite(role, bad, exact):
    reference = {'held': np.array([1.0, 2.0], dtype=np.float32)}
    candidate = {'held': reference['held'].copy()}
    if role != 'reference':
        candidate['held'][-1] = bad
    if role != 'candidate':
        reference['held'][-1] = bad
    with pytest.raises(AssertionError, match='nonfinite carry operand'):
        assert_all_leaves(reference, candidate, exact=exact)
