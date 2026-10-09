import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from gpuwrf.physics import rrtmg_constants as RC,rrtmg_lw as LW,rrtmg_sw as SW
DATA=json.loads((Path(__file__).parent/'fixtures/rrtmg_cloud_gravity_wrf.json').read_text())
def finite_operands(values):
 for name,x in values.items():assert np.isfinite(np.asarray(x)).all(),name+' nonfinite'
@pytest.fixture(autouse=True)
def clear(monkeypatch):
 if jax.default_backend()=='cpu':
  original=pl.pallas_call
  monkeypatch.setattr(pl,'pallas_call',lambda *args,**kwargs:original(*args,**dict(kwargs,interpret=True)))
 jax.clear_caches()
 yield
 jax.clear_caches()
def test_original_cloud_model_g_split(monkeypatch):
 rows=DATA['records'];v=np.asarray([[r[n] for n in ['q','p0','p1']] for r in rows],np.float32);ref=np.asarray([r['reference_bits'] for r in rows],np.int32).view(np.float32)[:,None]
 finite_operands({'q':v[:,0],'p0':v[:,1],'p1':v[:,2],'reference':ref})
 q=jnp.asarray(v[:,:1]);p=jnp.asarray(v[:,1:]);m=(p[:,:1]-p[:,1:])/jnp.float32(9.8066)
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True)
 with jax.enable_x64(False):got=np.asarray(jax.jit(RC.rrtmg_cloud_water_path)(q,m,p))
 assert np.isfinite(got).all();np.testing.assert_array_equal(got.view(np.uint32),ref.view(np.uint32))
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_cloud_off_and_legacy_unchanged(monkeypatch,dtype):
 v=np.asarray([[r[n] for n in ['q','p0','p1']] for r in DATA['records']],np.float32);q=jnp.asarray(v[:,:1],dtype);p=jnp.asarray(v[:,1:],dtype);m=(p[:,:1]-p[:,1:])/RC.GRAVITY
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',False);got=RC.rrtmg_cloud_water_path(q,m,p);ref=q*m*1000.
 for x in [q,p,m,got,ref]:assert np.isfinite(np.asarray(x)).all()
 np.testing.assert_array_equal(got,ref)
 if dtype==jnp.float64:
  monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True);np.testing.assert_array_equal(RC.rrtmg_cloud_water_path(q,m,p),ref)

@pytest.mark.parametrize('family',['lw','sw','sw_intermediate'])
def test_actual_solver_cloud_paths_use_original_interfaces(monkeypatch,family):
 from gpuwrf.validation import tier1_rrtmg as T
 module=LW if family=='lw' else SW;state,_=getattr(T,'load_'+('lw' if family=='lw' else 'sw')+'_fixture_state')();state=jax.tree.map(lambda x:jnp.asarray(x,jnp.float32),state)
 for x in jax.tree.leaves(state):assert np.isfinite(np.asarray(x)).all()
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True)
 class Captured(Exception):pass
 class AfterCloud(Exception):pass
 original=module.rrtmg_cloud_water_path
 calls=[]
 def capture(q,m,p,**kw):
  for x in [q,m,p]:assert x is not None and np.isfinite(np.asarray(x)).all()
  assert kw['entry_dtype']==jnp.float32 and p.shape[-1]==q.shape[-1]+1
  calls.append(1)
  if family=='lw':raise Captured
  return original(q,m,p,**kw)
 monkeypatch.setattr(module,'rrtmg_cloud_water_path',capture)
 if family!='lw':
  def after_cloud(*args,**kwargs):raise AfterCloud
  monkeypatch.setattr(module,'_sw_setcoef',after_cloud)
 with jax.disable_jit(),jax.enable_x64(False),pytest.raises(Captured if family=='lw' else AfterCloud):
  if family=='lw':module._lw_solver_base(state,module.RRTMG_TABLES,build_taumol=False,cloud_global=True)
  elif family=='sw':module._shortwave_impl(state,module.RRTMG_TABLES,False)
  else:module.compute_rrtmg_sw_intermediates(state)
 assert len(calls)>=(1 if family=='lw' else 3), 'cloud mass helper missing before gas-coefficient stage'

@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('field',['q','p0','p1','reference'])
def test_original_cloud_gate_rejects_nonfinite(bad,field):
 row=DATA['records'][1];values={n:np.asarray([row[n]],np.float32) for n in ['q','p0','p1']};values['reference']=np.asarray([row['reference_bits']],np.int32).view(np.float32);values[field][0]=bad
 with pytest.raises(AssertionError,match='nonfinite'):
  finite_operands(values)
