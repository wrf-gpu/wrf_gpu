"""Real PROD two-domain one-root CPU forecast + actual writer at lead0 and54s.

Not WRF fidelity or GPU admission. Full real physics/coupling/force-down, no
sentinel stepping. ZNT-only selected history bounds disk; snapshots are retained
as ZNT NPZ and hashes after every emitted file is read and validated.
"""
from pathlib import Path
import os,sys,json,time,hashlib,resource,traceback
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--case',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
tree=args.source;out=args.output;out.mkdir(exist_ok=False)
assert os.environ['JAX_PLATFORMS']=='cpu';assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path[:0]=[str(tree/'src'),str(tree/'proofs/o1_nlbind')]
started=time.monotonic();rec={'scope':__doc__,'phase':'imports','frames':[],'model_steps':'actual d01own1/d02own3','source':str(tree),'GPU':False,'cache':'NONEpersistent;own fix1_pycache/TMPDIR'};rc=1

def save():
 rec.update(wall_s=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(rec['phase'],round(rec['wall_s'],1),flush=True)
try:
 import jax
 import numpy as np
 from jax.experimental import pallas as pl
 from jax._src.pallas import pallas_call as pc
 original=pc.pallas_call
 def interpreted(*args,**kwargs):return original(*args,**{**kwargs,'interpret':True})
 pc.pallas_call=pl.pallas_call=interpreted
 import cpu_safe
 cpu_safe.install()
 from gpuwrf.contracts import state as sc
 sc._gpu_device=lambda:jax.devices('cpu')[0]
 from gpuwrf.backends import pallas as facade
 facade.cuda_native=lambda:False
 from gpuwrf.integration import nested_pipeline as p
 from gpuwrf.runtime import domain_tree as dt
 from gpuwrf.io.netcdf_lock import Dataset
 assert jax.devices()[0].platform=='cpu'
 case=args.case;names=('d01','d02')
 cfg=p.NestedPipelineConfig(case,out/'history',out/'proof',hours=3,max_dom=2,aot_prefetch=False)
 rec['phase']='load';save()
 hierarchy,bundles,meta,start,dts,carries=p._load_domains(cfg,names)
 rec.update(dt_s=dts,grids={n:list(c.state.theta.shape) for n,c in carries.items()},input_namelist_sha256=hashlib.sha256((case/'namelist.input').read_bytes()).hexdigest())
 cfg.output_dir.mkdir()
 writer=p._PerDomainWrfoutWriter(output_dir=cfg.output_dir,input_dir=case,run_start=start,bundles=bundles,output_cadence_steps={'d01':1,'d02':3},dt_by_domain=dts,initial_lands={n:jax.device_get(c.noahmp_land) for n,c in carries.items()})
 writer._variable_subset=('ZNT',);writer._full_variable_set=False
 maps={}
 def emit(name,step,carry):
  expected=np.asarray(carry.state.roughness_m,np.float32).copy()
  if step:
   land=np.asarray(carry.state.xland)<1.5
   assert expected[land].tobytes()==np.asarray(carry.noahmp_land.znt,np.float32)[land].tobytes(),name
  writer(name,step,carry)
  path=Path(writer.written[name][-1])
  with Dataset(path) as ds:
   assert 'ZNT' in ds.variables
   actual=np.asarray(ds['ZNT'][0]);assert actual.tobytes()==expected.tobytes()
   assert np.isfinite(actual).all()
   rec['frames'].append({'domain':name,'step':step,'lead_s':step*dts[name],'variables':list(ds.variables),'ZNT_shape':list(actual.shape),'ZNT_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'all_values_match_current_carry':True,'land_matches_current_Noah_z0wrf':bool(step),'path':str(path),'bytes':path.stat().st_size})
   maps[name+'_step'+str(step)]=actual.copy()
  save()
 emit.wants_carry=True  # public output callback contract: keep full OperationalCarry
 for name in names:emit(name,0,carries[name])
 nested=meta['nesting'];domain=dt.DomainTree.from_domains(hierarchy,bundles,feedback_enabled=nested['feedback'],feedback_spec_zone=nested['spec_zone'],feedback_smooth_option=nested['smooth_option'])
 rec['phase']='real_root';save()
 result=dt.run_operational_domain_tree(domain,root_steps=1,carries=carries,initial_own_steps={'d01':0,'d02':0},output=emit,output_cadence_steps={'d01':1,'d02':3},output_alarm_steps={'d01':(1,),'d02':(3,)},max_event_tail=0)
 jax.block_until_ready(result.carries)
 assert result.own_steps=={'d01':1,'d02':3}
 assert len(rec['frames'])==4 and {(x['domain'],x['step']) for x in rec['frames']}=={('d01',0),('d02',0),('d01',1),('d02',3)}
 np.savez_compressed(out/'ZNT_frames.npz',**maps)
 rec['phase']='PASS';rc=0
except BaseException as e:rec.update(phase='FAIL',exception=repr(e),traceback=traceback.format_exc())
finally:
 save();sys.stdout.flush();sys.stderr.flush();os._exit(rc)
