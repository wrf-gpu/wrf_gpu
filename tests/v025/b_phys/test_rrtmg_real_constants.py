"""BP105: original RRTMG REAL constants and heating arithmetic."""
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import rrtmg_constants as RC,rrtmg_lw as LW,rrtmg_sw as SW
DATA=json.loads((Path(__file__).parent/'fixtures/rrtmg_real_constants_wrf.json').read_text())
def finite_reference(reference):
 for name,entry in reference['constants'].items():
  assert np.isfinite(entry['value']),f'reference {name} value nonfinite'
  assert np.isfinite(np.asarray(entry['bits'],np.uint32).view(np.float32)),f'reference {name} bits nonfinite'
 for name in ['operands','heat_lw','heat_sw']:
  assert np.isfinite(np.asarray(reference[name])).all(),f'reference {name} nonfinite'
@pytest.fixture(autouse=True)
def clear():
 finite_reference(DATA)
 jax.clear_caches()
 yield
 jax.clear_caches()
@pytest.mark.parametrize('name',['LW_BPADE','GRAVITY','CP_AIR'])
def test_real_constant_original_bits_and_double(monkeypatch,name):
 old=getattr(RC,name);monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True)
 value=RC.rrtmg_constant(name,jnp.float32);assert np.isfinite(value)
 assert np.float32(value).view(np.uint32)==DATA['constants'][name]['bits']
 assert RC.rrtmg_constant(name,jnp.float64)==old
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',False);assert RC.rrtmg_constant(name,jnp.float32)==old
@pytest.mark.parametrize('shortwave',[False,True])
def test_heating_original_real_order(monkeypatch,shortwave):
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True)
 v=np.asarray(DATA['operands'],np.float32);p=jnp.asarray(v[:,:2]);delta=jnp.asarray(v[:,2:3]);mass=(p[:,:1]-p[:,1:])/np.float32(DATA['constants']['GRAVITY']['value'])
 ref=np.asarray(DATA['heat_sw' if shortwave else 'heat_lw'],np.float32)[:,None]
 for x in [v,ref]:assert np.isfinite(x).all()
 out=jax.jit(lambda d,m,pp:RC.rrtmg_heating_rate(d,m,pp,shortwave=shortwave))(delta,mass,p)
 assert np.isfinite(np.asarray(out)).all()
 np.testing.assert_array_equal(np.asarray(out).view(np.uint32),ref.view(np.uint32))
def test_actual_lw_lookup_consumes_original_bpade(monkeypatch):
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True);original=LW.rrtmg_constant;seen=[]
 def get(name,dtype,legacy=None):
  v=original(name,dtype,legacy)
  if name=='LW_BPADE':assert np.float32(v).view(np.uint32)==DATA['constants']['LW_BPADE']['bits'];seen.append(name)
  return v
 monkeypatch.setattr(LW,'rrtmg_constant',get)
 with jax.enable_x64(False):out=LW._lw_tfn_factor(jnp.array([.01,.1,1.,10.],jnp.float32))
 assert np.isfinite(np.asarray(out)).all();assert seen
def test_pressure_mass_both_families_use_original_gravity(monkeypatch):
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True)
 for mod in [LW,SW]:
  original=mod.rrtmg_constant;seen=[]
  def get(name,dtype,legacy=None):
   v=original(name,dtype,legacy)
   if name=='GRAVITY':assert np.float32(v).view(np.uint32)==DATA['constants']['GRAVITY']['bits'];seen.append(name)
   return v
  monkeypatch.setattr(mod,'rrtmg_constant',get)
  out=mod._pressure_layer_mass(jnp.array([[90000.,70000.,50000.]],jnp.float32));assert np.isfinite(np.asarray(out)).all();assert seen

@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64],ids=['native','retained'])
def test_sw_setcoef_constant_selection_precedes_real_island(monkeypatch,dtype):
 from gpuwrf.validation import tier1_rrtmg as T
 from gpuwrf.physics.rrtmg_tables import RRTMG_TABLES
 state,reference=T.load_sw_fixture_state()
 for x in [*jax.tree.leaves(state),*reference.values()]:assert np.isfinite(np.asarray(x)).all()
 state=jax.tree.map(lambda x:jnp.asarray(x,dtype),state);interfaces=SW._pressure_interfaces(state.p)
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True);original=SW.rrtmg_constant;seen=[]
 def get(name,precision,legacy=None):
  assert name=='GRAVITY' and precision==dtype
  value=original(name,precision,legacy);assert np.isfinite(value)
  expected=DATA['constants']['GRAVITY']['bits'] if dtype==jnp.float32 else np.float32(RC.GRAVITY).view(np.uint32)
  assert np.float32(value).view(np.uint32)==expected;seen.append(name);return value
 monkeypatch.setattr(SW,'rrtmg_constant',get)
 out=SW._sw_setcoef(state.qv,state.p,state.T,interfaces,RRTMG_TABLES)
 for x in jax.tree.leaves(out):assert np.isfinite(np.asarray(x)).all()
 assert seen
@pytest.mark.parametrize('shortwave',[False,True])
def test_double_and_off_heat_preserved(monkeypatch,shortwave):
 v=np.asarray(DATA['operands'],np.float64);d=jnp.asarray(v[:,2:3]);p=jnp.asarray(v[:,:2]);m=(p[:,:1]-p[:,1:])/RC.GRAVITY
 with jax.enable_x64(True):
  monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',False);a=RC.rrtmg_heating_rate(d,m,p,shortwave=shortwave)
  monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True);b=RC.rrtmg_heating_rate(d,m,p,shortwave=shortwave)
 assert np.isfinite(np.asarray(a)).all() and np.isfinite(np.asarray(b)).all();np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('shortwave',[False,True])
def test_real_flux_retained_pressure_keeps_legacy_heating(monkeypatch,shortwave):
 v=np.asarray(DATA['operands'],np.float64)
 with jax.enable_x64(True):
  d=jnp.asarray(v[:,2:3],jnp.float32);p=jnp.asarray(v[:,:2],jnp.float64);m=(p[:,:1]-p[:,1:])/RC.GRAVITY
  monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',False);a=jax.jit(lambda d,m,p:RC.rrtmg_heating_rate(d,m,p,shortwave=shortwave))(d,m,p)
  jax.clear_caches();monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True);b=jax.jit(lambda d,m,p:RC.rrtmg_heating_rate(d,m,p,shortwave=shortwave))(d,m,p)
 for x in [d,p,m,a,b]:assert np.isfinite(np.asarray(x)).all()
 assert a.dtype==b.dtype==jnp.float64;np.testing.assert_array_equal(a,b)
def test_resolved_real_constants_key(monkeypatch):
 from gpuwrf.runtime.aot_cheap_key import module_const_env_hash
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',False);old=module_const_env_hash()
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True);assert module_const_env_hash()!=old

def test_import_time_registry_has_no_unresolved_entries():
 import importlib
 from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS
 for module,attr in IMPORT_TIME_ENV_CONSTANTS:assert hasattr(importlib.import_module(module),attr),(module,attr)

@pytest.mark.parametrize('entry',['lw_band_fluxes','lw_band_flux_sums','lw_band_flux_sums_mp_re'])
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64],ids=['native','retained'])
def test_fused_entry_passes_original_bpade(monkeypatch,entry,dtype):
 from types import SimpleNamespace
 from gpuwrf.kernels import rad_lw_transfer as K
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True)
 z=jnp.zeros((1,2,16),dtype);state=SimpleNamespace(surface_emissivity=jnp.ones(1,dtype))
 class Captured(Exception):pass
 def intercept(*args,**kw):
  def call(*operands):
   for operand in operands:assert np.isfinite(np.asarray(operand)).all()
   const=operands[-1] if entry=='lw_band_fluxes' else operands[-3]
   expected=DATA['constants']['LW_BPADE']['bits'] if dtype==jnp.float32 else np.float32(RC.LW_BPADE).view(np.uint32)
   assert np.asarray(const[5]).view(np.uint32)==expected
   raise Captured
  return call
 monkeypatch.setattr(K.pl,'pallas_call',intercept)
 shared=(jnp.ones(1),jnp.float32(1),jnp.ones(16),jnp.zeros((1,2)),jnp.zeros((1,3)),jnp.zeros(1),jnp.zeros((1,2),bool),True)
 with pytest.raises(Captured):
  if entry=='lw_band_fluxes':K.lw_band_fluxes(state,z,z,z,z,*shared)
  else:
   cloud=SimpleNamespace(liquid=jnp.ones((16,)),ice=jnp.ones((16,)),snow=jnp.ones((16,)))
   getattr(K,entry)(state,z,z,(),0,*shared,gpoint_counts=[16]*16,cloud=cloud)

@pytest.mark.parametrize('name',['lw','sw'])
def test_full_solver_uses_original_heating_seam(monkeypatch,name):
 from gpuwrf.validation import tier1_rrtmg as T
 module=LW if name=='lw' else SW
 if name=='lw':monkeypatch.setattr(module,'_FUSED_TRANSFER',False)
 else:monkeypatch.setattr(module,'_FUSED_QUADRATURE',False)
 monkeypatch.setattr(RC,'_RRTMG_REAL_CONSTANTS',True)
 initial,reference=getattr(T,'load_'+name+'_fixture_state')()
 for operand in [*jax.tree.leaves(initial),*reference.values()]:assert np.isfinite(np.asarray(operand)).all()
 initial=jax.tree.map(lambda x:jnp.asarray(x,jnp.float32),initial)
 class Captured(Exception):pass
 def capture(delta,mass,interfaces,**kw):
  for operand in [delta,mass,interfaces]:assert operand is not None and np.isfinite(np.asarray(operand)).all()
  assert delta.dtype==jnp.float32
  assert bool(kw.get('shortwave',False))==(name=='sw')
  raise Captured
 monkeypatch.setattr(module,'rrtmg_heating_rate',capture)
 with jax.disable_jit(),jax.enable_x64(False),pytest.raises(Captured):
  getattr(module,'solve_rrtmg_'+name+'_column')(initial)

@pytest.mark.parametrize('name',list(DATA['constants'])+['operands','heat_lw','heat_sw'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_reference_nonfinite_rejected(name,bad):
 import copy
 reference=copy.deepcopy(DATA)
 if name in reference['constants']:reference['constants'][name]['value']=float(bad)
 else:np.asarray(reference[name]);reference[name][0]=[float(bad)]*len(reference[name][0]) if name=='operands' else float(bad)
 with pytest.raises(AssertionError,match='reference .* nonfinite'):finite_reference(reference)

@pytest.mark.parametrize('name',DATA['constants'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_reference_bits_nonfinite_rejected(name,bad):
 import copy
 reference=copy.deepcopy(DATA)
 reference['constants'][name]['bits']=int(np.float32(bad).view(np.uint32))
 with pytest.raises(AssertionError,match='reference .* nonfinite'):finite_reference(reference)
