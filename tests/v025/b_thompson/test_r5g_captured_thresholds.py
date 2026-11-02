"""Original-WRF R1 rules on retained real R5G trace columns (CPU native path).

The private fixture is external; its manifest and SHA are committed. This is
one-step operator parity, not reconstruction of the hidden GPU substeps.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import textwrap

import jax
import numpy as np
import pytest

from gpuwrf.physics import thompson_column as tc

HERE=Path(__file__).parent
SPEC=importlib.util.spec_from_file_location('r5g_frozen_mixed_gate',HERE/'test_mixed_phase_wrf_oracle.py')
mixed=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(mixed)


@pytest.fixture(scope='module')
def captured():
    manifest=json.loads((HERE/'fixtures/r5g_captured_thresholds.json').read_text())
    path=Path(os.environ.get('GPUWRF_R5G_FIXTURE',manifest['artifact']))
    if not path.is_file():
        pytest.skip('private R5G oracle absent; reproduce via proofs/v46_graupel scripts')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    z=dict(np.load(path,allow_pickle=False))
    z['names']=np.array([f'r5g_s{s}_y{y}_x{x}' for s,(y,x) in zip(z['own_step'],z['columns'])])
    z['dt']=np.full(len(z['names']),6.,np.float32)
    mixed._validate_fixture(z)
    assert z['in_qg'].shape==(124,44)
    assert np.count_nonzero(z['in_qg']>0)==222
    assert np.count_nonzero((z['in_qg']>0)&(z['wrf_qg']==0))==17
    return z,manifest


@pytest.fixture(scope='module')
def native(captured):
    z,_=captured
    variant=os.environ.get('GPUWRF_R5G_TEST_MUTANT','none')
    assert variant in ('none','mass','final')
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('GPUWRF_THOMPSON_WRF_CONSTANTS','1')
        if variant!='none':
            name='_wrf_mixed_phase_sources' if variant=='mass' else '_finish'
            old=('qgten = jnp.where(xrg > R1, qgten, -state.qg * odts)' if variant=='mass'
                 else 'qg = jnp.where(state.qg <= R1, 0.0, jnp.maximum(state.qg, 0.0))')
            new='qgten = qgten' if variant=='mass' else 'qg = jnp.maximum(state.qg, 0.0)'
            source=textwrap.dedent(inspect.getsource(getattr(tc,name)))
            assert source.count(old)==1
            namespace=dict(vars(tc))
            exec(compile(source.replace(old,new),tc.__file__,'exec'),namespace)
            mp.setattr(tc,name,namespace[name])
        jax.clear_caches()
        try:
            yield mixed._run(z,np.arange(len(z['names'])))
        finally:
            jax.clear_caches()


def test_r5g_captured_zero_topology_matches_pristine(captured,native):
    z,_=captured
    np.testing.assert_array_equal(native['qg']>0,z['wrf_qg']>0)


def test_r5g_captured_columns_pass_frozen_increment_gate(captured,native):
    z,_=captured
    errors=mixed._excess(native,z,np.arange(len(z['names'])),mixed._scales(z))
    assert not [(key,value) for key,value in errors.items() if value>mixed.TOL[key[1]]]


def test_r5g_pristine_mass_cut_is_not_merely_the_final_qg_cut(captured,native):
    z,manifest=captured
    for row in manifest['density_cut_witnesses']:
        i,k=row['index'],row['k']
        # Independent pristine trace: residual mixing ratio exceeds R1, but
        # density-weighted residual mass does not. The upstream mass rule acts.
        assert row['raw_rqg']/row['rho']>np.float32(1e-12)
        assert 0<row['raw_rqg']<=np.float32(1e-12)
        assert z['in_qg'][i,k]>np.float32(1e-12)
        assert native['qg'][i,k]==z['wrf_qg'][i,k]==0
