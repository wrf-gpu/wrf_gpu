"""WRF SCALE_AWARE final fraction bound, actual night/onset PBLH columns."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from gpuwrf.runtime import aot_cheap_key as K

DATA=json.loads((Path(__file__).parent/'fixtures/mynn_psig_clamp_wrf.json').read_text())


@pytest.fixture(autouse=True)
def clear_flag_traces():
    jax.clear_caches()
    yield
    jax.clear_caches()


def gate(got,ref,inputs):
    for side,fields in [('candidate',{'psig':got}),('reference',{'psig':ref}),('input',inputs)]:
        for n,v in fields.items():assert np.isfinite(v).all(),f'{side} {n} nonfinite'
    np.testing.assert_allclose(got,ref,rtol=0,atol=1e-6)
    assert np.all((got>=0)&(got<=1))


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:f"tau{r['tau']}")
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64],ids=['native','retained'])
def test_original_real4_fraction_bound(monkeypatch,row,dtype):
    monkeypatch.setattr(P,'_MYNN_PSIG_CLAMP',True)
    h=jnp.asarray(row['pblh'],dtype);dx=jnp.asarray(row['dx'],dtype)
    with jax.enable_x64(dtype==jnp.float64):got=np.asarray(jax.jit(P._scale_aware_psig_bl)(dx,h))
    ref=np.asarray(row['WRF_psig_bl'],np.float32)
    gate(got,ref,{'dx':np.asarray(dx),'pblh':np.asarray(h)})
    if dtype==jnp.float32:assert np.all(abs(got-ref)<=8*abs(np.spacing(ref)))


def test_off_returns_original_unbounded_fraction(monkeypatch):
    monkeypatch.setattr(P,'_MYNN_PSIG_CLAMP',False)
    row=DATA['rows'][0]
    with jax.enable_x64(False):got=np.asarray(jax.jit(P._scale_aware_psig_bl)(jnp.float32(row['dx']),jnp.asarray(row['pblh'],jnp.float32)))
    ref=np.asarray(row['old_port_psig_bl'],np.float32)
    assert np.isfinite(got).all() and np.isfinite(ref).all()
    np.testing.assert_array_equal(got,ref)
    assert np.any(got>1), 'real onset witness lost'


def test_imported_flag_is_resolved_in_key(monkeypatch):
    assert ('gpuwrf.physics.mynn_pbl','_MYNN_PSIG_CLAMP') in K.IMPORT_TIME_ENV_CONSTANTS
    monkeypatch.setattr(P,'_MYNN_PSIG_CLAMP',False);key0=K.module_const_env_hash()
    monkeypatch.setenv('GPUWRF_MYNN_PSIG_CLAMP','1');assert K.module_const_env_hash()==key0
    monkeypatch.setattr(P,'_MYNN_PSIG_CLAMP',True);assert K.module_const_env_hash()!=key0


@pytest.mark.parametrize('side',['candidate','reference','pblh','dx'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_gate_rejects_nonfinite_before_bounding(side,bad):
    got=np.ones(4);ref=np.ones(4);inp={'pblh':np.ones(4)*800,'dx':np.ones(1)*1000}
    (got if side=='candidate' else ref if side=='reference' else inp[side]).flat[0]=bad
    with pytest.raises(AssertionError,match='nonfinite'):gate(got,ref,inp)
