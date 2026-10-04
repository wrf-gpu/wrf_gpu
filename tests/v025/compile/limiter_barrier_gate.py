"""Direct-Python CPU/GPU byte gate for the PD limiter compile barrier.

Run outside pytest's v025 CPU conftest. Requires real-history fixtures built by
b_core/build_savepoints.py plus the corresponding CPU-WRF history. Regression
against the unchanged limiter arithmetic; this is not a WRF fidelity oracle.
"""
import argparse,importlib.util,sys,json,time,statistics,os
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from netCDF4 import Dataset
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--fixture',required=True);p.add_argument('--history',required=True);p.add_argument('--species',type=int,default=6);a=p.parse_args()
assert jax.devices()[0].platform==('gpu' if a.gpu else 'cpu'),jax.devices()
if a.gpu: assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
import gpuwrf.dynamics.flux_advection as candidate
import types,ast,hashlib
barrier='        ph_low, flux_out = jax.lax.optimization_barrier((ph_low, flux_out))\n'
candidate_source=Path(candidate.__file__).read_text()
assert candidate_source.count(barrier)==1
baseline_source=candidate_source.replace(barrier,'')
baseline=types.ModuleType('flux_baseline');sys.modules[baseline.__name__]=baseline
exec(compile(baseline_source,candidate.__file__,'exec'),baseline.__dict__)

z=np.load(a.fixture)
history=a.history
with Dataset(history) as ds:
 qs=tuple(jnp.asarray(ds.variables[k][0],dtype=jnp.float64) for k in ['QVAPOR','QCLOUD','QRAIN','QICE','QSNOW','QGRAUP'][:a.species])
inputs=qs+tuple(jnp.asarray(z[k]) for k in ['ref11_ru_m','ref11_rv_m','ref11_ww_m','state_msftx','state_mut','state_c1h','state_c2h','state_rdnw','state_fnm','state_fnp'])
records={}
values={}
for name,module in [('baseline',baseline),('candidate',candidate)]:
 def run(*args):
  fields=args[:a.species];ru,rv,rom,msf,mu,c1,c2,rdzw,fzm,fzp=args[a.species:]
  vel=module.CoupledVelocities(ru=ru[:,:,:-1],rv=rv[:,:-1,:],rom=rom,msftx=msf,specified=True,ru_full=ru,rv_full=rv)
  return tuple(module.advect_scalar_flux_limited(q,q,vel,scalar_adv_opt=1,mut=mu,mu_old=mu,c1=c1,c2=c2,rdx=1/9000.,rdy=1/9000.,rdzw=rdzw,fzm=fzm,fzp=fzp,dt=54.) for q in fields)
 started=time.perf_counter();lowered=jax.jit(run).lower(*inputs);compiled=lowered.compile();ct=time.perf_counter()-started
 output=compiled(*inputs);jax.block_until_ready(output);values[name]=tuple(np.asarray(x) for x in output)
 for _ in range(3):jax.block_until_ready(compiled(*inputs))
 samples=[]
 for _ in range(11 if a.gpu else 1):
  started=time.perf_counter()
  for _ in range(8 if a.gpu else 1):output=compiled(*inputs)
  jax.block_until_ready(output);samples.append((time.perf_counter()-started)/(8 if a.gpu else 1))
 records[name]={'compile_seconds':ct,'median_s':statistics.median(samples),'samples_s':samples,'finite':all(np.isfinite(x).all() for x in values[name])}
 (Path(a.out).parent/(Path(a.out).stem+'_'+name+'.hlo')).write_text(compiled.as_text())
records['comparison']={'bitwise':all(x.tobytes()==y.tobytes() for x,y in zip(values['baseline'],values['candidate'])),'bit_mismatch_cells':sum(int(np.count_nonzero(x.view(np.uint64)!=y.view(np.uint64))) for x,y in zip(values['baseline'],values['candidate'])),'max_abs':max(float(np.max(np.abs(x-y))) for x,y in zip(values['baseline'],values['candidate']))}
records.update(platform=jax.devices()[0].platform,species=a.species,case='real PROD CPU-WRF d01 h01:00:18 moisture + SAME history-derived d01 RK3 acoustic ref11 mean fluxes + stage mass/metrics; specified boundaries; regression only; not WRF fidelity',xla_flags=os.environ.get('XLA_FLAGS'))
# Count actually active limiter predicates in a separate untimed compilation.
import inspect
node=next(n for n in ast.parse(baseline_source).body if isinstance(n,ast.FunctionDef) and n.name=='advect_scalar_flux_limited')
source='\n'.join(baseline_source.splitlines()[node.lineno-1:node.end_lineno])
source=source.replace('        scale = jnp.where(', '        PD_COUNTS.append(jnp.count_nonzero(flux_out > ph_low))\n        scale = jnp.where(',1)
baseline.PD_COUNTS=[]
exec(source,baseline.__dict__)
def active(*args):
 fields=args[:a.species];ru,rv,rom,msf,mu,c1,c2,rdzw,fzm,fzp=args[a.species:]
 baseline.PD_COUNTS.clear()
 vel=baseline.CoupledVelocities(ru=ru[:,:,:-1],rv=rv[:,:-1,:],rom=rom,msftx=msf,specified=True,ru_full=ru,rv_full=rv)
 for q in fields:
  baseline.advect_scalar_flux_limited(q,q,vel,scalar_adv_opt=1,mut=mu,mu_old=mu,c1=c1,c2=c2,rdx=1/9000.,rdy=1/9000.,rdzw=rdzw,fzm=fzm,fzp=fzp,dt=54.)
 return tuple(baseline.PD_COUNTS)
records['active_limiter_cells']=[int(np.asarray(x)) for x in jax.jit(active)(*inputs)]
baseline.PD_COUNTS.clear()
records['flux_max_abs']=[float(np.max(np.abs(np.asarray(x)))) for x in inputs[a.species:a.species+3]]
records['fixture']=a.fixture
records['source_sha256']=hashlib.sha256(candidate_source.encode()).hexdigest()
records['history']=history
records['nonzero_input_cells']=[int(np.count_nonzero(np.asarray(q))) for q in qs]
records['cache_status']=__import__('gpuwrf')._JAX_CACHE_STATUS
np.savez(Path(a.out).with_suffix('.npz'),**{name+'_'+str(i):v for name,vs in values.items() for i,v in enumerate(vs)})
Path(a.out).write_text(json.dumps(records,indent=2));print(json.dumps(records),flush=True)
assert all(v['finite'] for k,v in records.items() if k in ['baseline','candidate'])
assert sum(records['active_limiter_cells'])>0,'inactive limiter fixture'
assert records['comparison']['bitwise'],records['comparison']
