"""CPU-only real-PROD raw CUDA-target HLO and boundary byte capture.

Run the two immutable sources through the same import alias in fresh processes;
Pallas embeds absolute source paths in raw backend configurations. No HLO text
normalization, GPU compilation, or execution is performed.
"""
import argparse,os,sys,json,hashlib,time,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True)
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']==''
assert os.environ['GPUWRF_BOUNDARY_FP32']=='0'
sys.path.insert(0,str(a.source/'src'))
sys.path.insert(0,str(a.source/'tests/v025/b_core'))
from prod_inputs import prod_domains
import jax,jax.numpy as jnp,numpy as np,jaxlib._jax as x
assert all(d.platform=='cpu' for d in jax.devices())
from jax.experimental import topologies
from gpuwrf.runtime import operational_mode as op,domain_tree as dt,aot_executable as ax
from gpuwrf.coupling import boundary_apply as bd
from gpuwrf.physics.noahmp.precision import real_tree
record={'source':str(a.source),'resolved_source':str(a.source.resolve()),'platform':'cpu','BOUNDARY_FP32':'0','flags':{k:v for k,v in os.environ.items() if k.startswith(('GPUWRF_DYN_','GPUWRF_THOMPSON_','GPUWRF_MYNN_','GPUWRF_NOAHMP_','GPUWRF_KF_','GPUWRF_RRTMG_','GPUWRF_MCICA_','GPUWRF_EDMF_'))},'domains':{}}
record['boundary_source_sha256']=hashlib.sha256(Path(bd.__file__).read_bytes()).hexdigest()
try:
 target_path=Path('<USER_HOME>/wrf_gpu2_lanes/compile/phase2/R3_joint/default_import/captured_rtx_target.textproto');target=target_path.read_text();record['target_sha256']=hashlib.sha256(target.encode()).hexdigest()
 topo=topologies.get_topology_desc(platform='cuda',target_config=target,topology='1x1x1');assert isinstance(topo.devices[0].client,x.CompileOnlyPyClient)
 sharding=jax.sharding.SingleDeviceSharding(topo.devices[0]);record['lower_client']='CompileOnlyPyClient/cuda; runtimeCPU CUDAhidden; no compile/execute'
 h,b,meta,start,dts,c=prod_domains()
 tree=dt.DomainTree.from_domains(h,b,feedback_enabled=False)
 def abstract(v):
  return jax.ShapeDtypeStruct(v.shape,v.dtype,weak_type=bool(getattr(v,'weak_type',False)),sharding=sharding) if hasattr(v,'shape') and hasattr(v,'dtype') else v
 def digest(value):
  out={}
  for path,v in jax.tree.flatten_with_path(value)[0]:
   arr=np.asarray(v);out[jax.tree_util.keystr(path)]={'shape':list(arr.shape),'dtype':str(arr.dtype),'SHA':hashlib.sha256(arr.tobytes()).hexdigest()}
  return out
 for domain in ['d01','d02']:
  carry=c[domain].replace(noahmp_land=real_tree(c[domain].noahmp_land));nl=tree.domains[domain].namelist;clock=op.build_clock_base(nl)
  concrete=(carry,nl,jnp.asarray(1,jnp.int32),clock);args=jax.tree.map(abstract,concrete)
  t=time.perf_counter();lower=op._advance_chunk_fori.lower(*args,n_steps=1 if domain=='d01' else 3,cadence=int(nl.radiation_cadence_steps))
  hlo=lower.compiler_ir('hlo').as_hlo_text();(a.out/(domain+'.hlo')).write_text(hlo)
  entry={'HLO_SHA':hashlib.sha256(hlo.encode()).hexdigest(),'hlo_bytes':len(hlo),'lower_s':time.perf_counter()-t,'root_shape':list(carry.state.theta.shape),'output_hashes':{}}
  record['domains'][domain]=entry
  print(domain,'lower',entry['HLO_SHA'],entry['lower_s'],flush=True);(a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
  # Run only the boundary operators on CPU; native GPU kernels outside this
  # family are not executed or emulated by the byte-identity gate.
  with np.load('<USER_HOME>/wrf_gpu2_lanes/b-diff/boundary_fixtures/'+domain+'.npz') as z:data={n:jnp.asarray(z[n]) for n in z.files}
  from gpuwrf.contracts.state import State
  from gpuwrf.contracts.grid import DycoreMetrics
  metrics=DycoreMetrics(**{n:data['metric_'+n] for n in DycoreMetrics._array_names()})
  for variant in (['physical'] if domain=='d01' else ['physical','coupled']):
   fields={n:data.get('state_'+n,data.get(variant+'_'+n)) for n in State.__slots__};s=State.tree_unflatten(None,[fields[n] for n in State.__slots__])
   cfg=bd.BoundaryConfig(update_cadence_s=21600. if domain=='d01' else 54.,force_geopotential=domain=='d01',nested_frozen_wrf_boundary_bundle=variant=='coupled',normal_bdy_relax_strength=1.)
   dt=54. if domain=='d01' else 18.
   def call(state,m):
    dry=bd.specified_relax_dry_tendencies(state,dt,m,dt,cfg,include_nested_w=domain=='d02',coupled_boundary_leaves=variant=='coupled')
    result={'dry':tuple(getattr(dry,n) for n in ['ru','rv','t','ph','mu','w']),
      'lateral':bd.apply_lateral_boundaries(state,dt,dt,cfg,m),
      'spec_only':bd.apply_lateral_boundaries(state,dt,dt,cfg,m,dry_spec_only=True)}
    if variant=='coupled':result['scalars']=bd.nested_scalar_boundary_tendencies(state,dt,m,dt,cfg)
    for name,field in [('u',state.u),('v',state.v),('theta',state.theta),('w',state.w),('ph',state.ph_perturbation),('mu',state.mu_perturbation[None])]:
     leaf=getattr(state,name+'_bdy');result['interpolate_'+name]=bd.interpolate_boundary_leaf(leaf,dt,cfg.update_cadence_s);result['tend_'+name]=bd.specified_boundary_tendency(leaf,dt,cfg.update_cadence_s,z_len=field.shape[0],y_len=field.shape[1],x_len=field.shape[2],dtype=field.dtype,config=cfg)
    return result
   outputs=jax.jit(call)(s,metrics);jax.block_until_ready(outputs);entry['output_hashes'][variant]=digest(outputs);print(domain,variant,'outputs',len(entry['output_hashes'][variant]),flush=True)
  (a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');jax.clear_caches()
 record['success']=True
except BaseException as e:
 record['success']=False;record['error']=str(e);record['traceback']=traceback.format_exc();traceback.print_exc()
finally:(a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
if not record['success']:raise SystemExit(1)
