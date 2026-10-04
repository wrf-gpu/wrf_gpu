"""Runtime checkpoint optional GWDO order and active held-schema refusal (E78)."""
from dataclasses import replace
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES, GWDO_DIAGNOSTIC_LEAVES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.runtime import checkpoint as cp
from gpuwrf.runtime.operational_mode import OperationalNamelist

READERS = (cp.read_checkpoint, cp.read_checkpoint_with_runtime_state, cp.read_checkpoint_with_land_state)


@pytest.fixture
def inactive():
    grid = GridSpec.canary_3km_template()
    fields = {name: jnp.full(shape, i/16, DEFAULT_DTYPES.dtype_for(name))
              for i,(name,shape) in enumerate(_state_field_shapes(grid).items(),1)}
    state = State(**fields)
    nl = OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,dt_s=18,
                             acoustic_substeps=3,gwd_opt=0)
    return state,nl,grid


def active_state(state,grid):
    shapes = _state_field_shapes(grid,gwd_opt=1)
    return state.replace(**{name: jnp.arange(np.prod(shapes[name]),dtype=jnp.float32)
                           .reshape(shapes[name])/32+i/8
                           for i,name in enumerate(GWDO_DIAGNOSTIC_LEAVES,1)})


@pytest.mark.parametrize('reader',READERS,ids=('plain','runtime','land'))
def test_inactive_gwdo_none_roundtrip_and_legacy_marker(inactive,tmp_path,reader):
    state,nl,grid=inactive
    path=tmp_path/'inactive.pkl'
    cp.write_checkpoint(state,nl,grid,17,path)
    for legacy in (False,True):
        if legacy:
            payload=pickle.loads(path.read_bytes())
            payload.pop('gwdo_checkpoint_schema_version')
            path.write_bytes(pickle.dumps(payload))
        restored,_,_,step,*_=reader(path)
        assert step==17 and restored.active_field_names()==state.active_field_names()
        assert all(getattr(restored,name) is None for name in GWDO_DIAGNOSTIC_LEAVES)
        for name in state.active_field_names():
            actual,expected=np.asarray(getattr(restored,name)),np.asarray(getattr(state,name))
            assert actual.dtype==expected.dtype and actual.shape==expected.shape,name
            np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('reader',READERS,ids=('plain','runtime','land'))
def test_active_gwdo_roundtrip_and_pre_schema_refused(inactive,tmp_path,reader):
    state,nl,grid=inactive
    state=active_state(state,grid);nl=replace(nl,gwd_opt=1)
    path=tmp_path/'active.pkl'
    cp.write_checkpoint(state,nl,grid,23,path)
    restored=reader(path)[0]
    for name in GWDO_DIAGNOSTIC_LEAVES:
        np.testing.assert_array_equal(getattr(restored,name),getattr(state,name))
        assert getattr(restored,name).dtype==jnp.float32
    payload=pickle.loads(path.read_bytes())
    payload.pop('gwdo_checkpoint_schema_version')
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError,match='GWDO.*E78'):
        reader(path)


@pytest.mark.parametrize('missing',GWDO_DIAGNOSTIC_LEAVES)
def test_active_gwdo_missing_held_leaf_refused(inactive,tmp_path,missing):
    state,nl,grid=inactive
    path=tmp_path/'partial.pkl'
    cp.write_checkpoint(active_state(state,grid),replace(nl,gwd_opt=1),grid,23,path)
    payload=pickle.loads(path.read_bytes())
    payload['state_fields'].pop(missing)
    payload['state_field_order'].remove(missing)
    payload['state_field_count']-=1
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError,match='active GWDO.*E78'):
        cp.read_checkpoint(path)


def test_active_gwdo_write_refusal_preserves_previous_checkpoint(inactive,tmp_path):
    state,nl,grid=inactive
    path=tmp_path/'preserve.pkl'
    cp.write_checkpoint(state,nl,grid,17,path)
    previous=path.read_bytes()
    with pytest.raises(ValueError,match='active GWDO.*E78'):
        cp.write_checkpoint(state,replace(nl,gwd_opt=1),grid,23,path)
    assert path.read_bytes()==previous
