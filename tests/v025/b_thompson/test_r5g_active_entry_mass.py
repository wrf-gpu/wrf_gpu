"""qg-only controls on R5G backgrounds expose an active spectral mass floor.

These are counterfactual operator inputs, not captured GPU pre-MP savepoints.
Pristine WRF supplies the reference; all thermodynamic/rain/geometry operands
are the retained real columns. Reverting the fix must fail the same gates.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from gpuwrf.physics import thompson_column as tc

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    'r5g_active_frozen_gate', HERE / 'test_mixed_phase_wrf_oracle.py')
mixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mixed)


@pytest.fixture(scope='module')
def captured_controls():
    manifest = json.loads((HERE / 'fixtures/r5g_active_entry_mass.json').read_text())
    path = Path(os.environ.get('GPUWRF_R5G_ACTIVE_FIXTURE', manifest['artifact']))
    if not path.is_file():
        pytest.skip('private R5G active-mass oracle absent; see manifest for producer')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
    z = dict(np.load(path, allow_pickle=False))
    z['names'] = np.array([f'r5g_active_{i}' for i in range(manifest['columns'])])
    z['dt'] = np.full(manifest['columns'], 6., np.float32)
    mixed._validate_fixture(z)
    assert z['in_qg'].shape == (11, 44)
    parent_manifest = json.loads((HERE / 'fixtures/r5g_captured_thresholds.json').read_text())
    parent_path = Path(os.environ.get('GPUWRF_R5G_FIXTURE', parent_manifest['artifact']))
    if not parent_path.is_file():
        pytest.skip('private R5G parent capture absent; provenance cannot be qualified')
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == manifest['parent_artifact_sha256']
    parent = dict(np.load(parent_path, allow_pickle=False))
    for i, row in enumerate(manifest['witnesses']):
        k, p = row['k'], row['captured_index']
        assert int(parent['own_step'][p]) == row['captured_step']
        assert parent['columns'][p].tolist() == [row['y'], row['x']]
        for key in z:
            if key.startswith(('in_', 'column_')):
                expected = parent[key][p].copy()
                if key in ('in_qg', 'column_qg'):
                    expected[k] = np.float32(row['qg_seed'])
                assert z[key][i].dtype == expected.dtype
                assert z[key][i].tobytes() == expected.tobytes(), key
        assert z['in_qg'][i, k] > np.float32(1.e-12)
        assert z['in_qg'][i, k] * z['column_rho'][i, k] < np.float32(1.e-12)
        assert z['wrf_qg'][i, k] > np.float32(1.e-12)
        assert z['baseline_qg'][i, k] == 0.
    return z


@pytest.fixture(scope='module')
def native(captured_controls):
    variant = os.environ.get('GPUWRF_R5G_ACTIVE_MUTANT', 'none')
    assert variant in ('none', 'entry_floor')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('GPUWRF_THOMPSON_WRF_CONSTANTS', '1')
        if variant == 'entry_floor':
            # Literal reversion of the two original callers' mass expression.
            # Execute in tc's globals so the native float32 clone sees jnp/R1.
            namespace = dict(vars(tc))
            exec('def _graupel_working_mass(qg, rho):\n'
                 '    return jnp.maximum(qg * rho, R1)\n', namespace)
            mp.setattr(tc, '_graupel_working_mass', namespace['_graupel_working_mass'])
        jax.clear_caches()
        try:
            yield mixed._run(captured_controls, np.arange(11))
        finally:
            jax.clear_caches()


def test_r5g_active_entry_zero_topology_matches_pristine(captured_controls, native):
    np.testing.assert_array_equal(native['qg'] > 0, captured_controls['wrf_qg'] > 0)


def test_r5g_active_entry_passes_frozen_increment_gate(captured_controls, native):
    z = captured_controls
    errors = mixed._excess(native, z, np.arange(11), mixed._scales(z))
    assert not [(key, value) for key, value in errors.items() if value > mixed.TOL[key[1]]]
