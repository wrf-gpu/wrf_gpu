"""Original WRF CALL-day solar clock; midpoint belongs only to hour angle."""
import copy,importlib,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.coupling import physics_couplers as C
DATA=json.loads((Path(__file__).parent/'fixtures/solar_julian_wrf.json').read_text())
FIELDS=['declination_rad','coszen','hour_angle_rad','solcon']

def validate(rows):
 for r in rows:
  for name in ['lat','lon','julian_base','minute_base','lead','call_julian','midpoint']:
   assert np.isfinite(r[name]),f'input {name} nonfinite'
  ref=np.asarray(r['reference_bits'],np.int32).view(np.float32)
  assert np.isfinite(ref).all(),'reference nonfinite'
  if r['reference_eot'] is not None:assert np.isfinite(r['reference_eot']),'reference eot nonfinite'

def compare(got,ref,julian,expected_julian):
 for name,values in [('candidate',got),('reference',ref),('julian',julian),('reference julian',expected_julian)]:assert np.isfinite(values).all(),name+' nonfinite'
 np.testing.assert_array_equal(julian.view(np.uint32),expected_julian.view(np.uint32))
 for i in [2,3]:np.testing.assert_array_equal(got[:,i].view(np.uint32),ref[:,i].view(np.uint32),err_msg=FIELDS[i])
 ulp=abs(got[:,0].view(np.uint32).astype(np.int64)-ref[:,0].view(np.uint32).astype(np.int64))
 assert np.max(ulp)<=1,'declination >1ULP'
 assert np.max(abs(got[:,1].astype(float)-ref[:,1].astype(float)))<=np.spacing(np.float32(1)),'COSZEN >1ULP(1)'

@pytest.fixture(autouse=True)
def clear():
 validate(DATA['rows']);jax.clear_caches()
 yield
 jax.clear_caches()

def test_original_real_solar_504_columns(monkeypatch):
 rows=DATA['rows'];validate(rows);monkeypatch.setattr(C,'_SOLAR_JULIAN_WRF',True)
 args=[jnp.asarray([r[n] for r in rows],jnp.float32) for n in ['lat','lon','julian_base','minute_base','lead','midpoint']]
 def run(lat,lon,julian,minute,lead,midpoint):
  clock=(julian,minute);g=C._compute_solar_geometry(lat,lon,None,lead,clock_base=clock,solar_lead_seconds=lead+midpoint)
  return jnp.stack([g.declination_rad,g.coszen,g.hour_angle_rad,C._solcon_for_time(None,lead,clock_base=clock)],axis=-1),C._wrf_solar_julian_real(julian,minute,lead)
 with jax.enable_x64(False):out,jul=jax.jit(run)(*args)
 ref=np.asarray([r['reference_bits'] for r in rows],np.int32).view(np.float32)
 compare(np.asarray(out),ref,np.asarray(jul),np.asarray([r['call_julian'] for r in rows],np.float32))

def test_both_geometry_consumers_and_midpoint_date(monkeypatch):
 monkeypatch.setattr(C,'_SOLAR_JULIAN_WRF',True)
 row=next(r for r in DATA['rows'] if r['case']=='0227day3');clock=(jnp.float32(row['julian_base']),jnp.float32(row['minute_base']));lat=jnp.float32(row['lat']);lon=jnp.float32(row['lon']);lead=jnp.float32(row['lead'])
 with jax.enable_x64(False):
  a=C._compute_coszen(lat,lon,None,lead,clock_base=clock);b=C._compute_solar_geometry(lat,lon,None,lead,clock_base=clock)
  midpoint=C._compute_solar_geometry(lat,lon,None,lead,clock_base=clock,solar_lead_seconds=lead+900)
 for x in jax.tree.leaves((a,b,midpoint)):assert np.isfinite(np.asarray(x)).all()
 np.testing.assert_array_equal(a,b.coszen);np.testing.assert_array_equal(b.declination_rad,midpoint.declination_rad)
 assert b.hour_angle_rad!=midpoint.hour_angle_rad

def test_flag_registered_and_changes_module_key(monkeypatch):
 from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS,module_const_env_hash
 for module,attr in IMPORT_TIME_ENV_CONSTANTS:assert hasattr(importlib.import_module(module),attr),(module,attr)
 monkeypatch.setattr(C,'_SOLAR_JULIAN_WRF',False);off=module_const_env_hash()
 monkeypatch.setattr(C,'_SOLAR_JULIAN_WRF',True);assert module_const_env_hash()!=off

@pytest.mark.parametrize('field',['lat','lon','julian_base','minute_base','lead','call_julian','midpoint','reference_bits','reference_eot'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_every_nonfinite_operand_rejected(field,bad):
 rows=copy.deepcopy(DATA['rows'][:1])
 if field=='reference_bits':rows[0][field][0]=int(np.float32(bad).view(np.int32))
 else:rows[0][field]=float(bad)
 with pytest.raises(AssertionError,match='nonfinite'):validate(rows)

@pytest.mark.parametrize('role',['candidate','reference','julian','reference julian'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_complete_comparison_rejects_nonfinite_before_bits(role,bad):
 rows=DATA['rows'][:1];ref=np.asarray([rows[0]['reference_bits']],np.int32).view(np.float32);got=ref.copy();jul=np.asarray([rows[0]['call_julian']],np.float32);jref=jul.copy();values={'candidate':got,'reference':ref,'julian':jul,'reference julian':jref};values[role].flat[0]=bad
 with pytest.raises(AssertionError,match='nonfinite'):compare(got,ref,jul,jref)
