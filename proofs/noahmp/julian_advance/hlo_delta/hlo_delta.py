# Lane copy of compile/v032/raw_wn3_identity.py (read-only origin) + op histograms, raw HLO gz, --rev/--fused.
from pathlib import Path
import argparse,os,sys,time,json,hashlib,resource,subprocess
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--domains',default='d01,d02');p.add_argument('--out',type=Path,required=True);p.add_argument('--rev',required=True);p.add_argument('--fused',action='store_true');a=p.parse_args()
import gzip,collections,re
def ophist(raw):
 return dict(collections.Counter(m.group(1) for m in re.finditer(r'=\s*\S+\s+([a-z][a-z0-9\-]*)\(',raw)))
def save(tag,raw):
 (a.out/(tag.replace('/','_')+'.hlo.gz')).write_bytes(gzip.compress(raw.encode()))
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ.update(GPUWRF_FAST_DEFAULTS='1',GPUWRF_JAX_CACHE='0',JAX_ENABLE_COMPILATION_CACHE='false',JAX_ENABLE_X64='true',JAX_PALLAS_USE_MOSAIC_GPU='false',GPUWRF_WRF_ROOT='<DATA_ROOT>/wrf_gpu2/v025/tenerife_b4/wrf_root')
# SAME alias/source coordinates for base and candidate; opt out of new path
# canonicalization so this arm isolates only the dormant-scheme import change.
os.environ['JAX_HLO_SOURCE_FILE_CANONICALIZATION_REGEX']=''
sys.path.insert(0,str(a.source/'src'))
import jax,numpy as np
from jax._src.pallas import pallas_call as pc
import jax.experimental.pallas as pl
native=[False];original_pc=pc.pallas_call

def selected_pc(*args,**kwargs):
 kwargs['interpret']=not native[0];return original_pc(*args,**kwargs)
pc.pallas_call=selected_pc;pl.pallas_call=selected_pc
backend=jax.default_backend;jax.default_backend=lambda:'gpu' if native[0] else backend()
import gpuwrf.contracts.state as st
st._gpu_device=lambda:jax.devices('cpu')[0]
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime import operational_mode as op,domain_tree as dt,aot_cheap_key as ck
from jax.experimental import topologies
assert all(d.platform=='cpu' for d in jax.devices())
a.out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
names=tuple(a.domains.split(','));loaded=_load_domains(NestedPipelineConfig(a.case,a.out/'unused',a.out/'proof',hours=3,max_dom=len(names)),names)
hierarchy,bundles,metadata,run_start,dts,carries=loaded
tree=dt.DomainTree.from_domains(hierarchy,bundles,feedback_enabled=False)
record={'source':str(a.source.resolve()),'revision':a.rev,'julian_flag':os.environ.get('GPUWRF_NOAHMP_JULIAN_ADVANCE','unset'),'case':str(a.case),'physical_backend':'cpu','scope':'real per-tree initialized carries; virtual CUDA lower only; no native compile, execution or fidelity claim','init_s':time.monotonic()-start,'domains':{}}
target=Path('<USER_HOME>/wrf_gpu2_lanes/compile/phase2/R3_joint/default_import/captured_rtx_target.textproto').read_text();topo=topologies.get_topology_desc(platform='cuda',target_config=target,topology='1x1x1');sharding=jax.sharding.SingleDeviceSharding(topo.devices[0])
def abstract(v):
 return jax.ShapeDtypeStruct(v.shape,v.dtype,weak_type=bool(getattr(v,'weak_type',False)),sharding=sharding) if hasattr(v,'shape') and hasattr(v,'dtype') else v
for name in (names[:1] if a.fused else names):
 carry=carries[name];nl=bundles[name].namelist
 if name!=names[0]:
  edge=next(e for parent in names for e in tree.children(parent) if e.child==name)
  carry=dt._operational_force(edge,carries[edge.parent],carry);jax.block_until_ready(carry);carries[name]=carry;jax.clear_caches()
 # The init seam seeds the same two-time boundary signature used by the product.
 # No stale pickle crosses trees; all metadata/scalars are rebuilt here.
 native[0]=True
 import jax.numpy as jnp
 args=jax.tree.map(abstract,(carry,nl,jnp.asarray(1,jnp.int32),op.build_clock_base(nl)))
 kwargs={'n_steps':1 if name==names[0] else 3,'cadence':int(nl.radiation_cadence_steps)}
 t=time.monotonic();low=op._advance_chunk_fori.lower(*args,**kwargs);raw=low.compiler_ir('hlo').as_hlo_text()
 components={'fn_identity':ck.fn_identity_hash(op._advance_chunk_fori),'program_config':ck.program_config_hash(),'source_fingerprint':ck.source_fingerprint_hash(),'static_config':ck.static_config_hash(nl),'carry_aval':ck.carry_aval_hash(args,kwargs),'trace_env':ck.global_trace_env_hash(),'module_constants':ck.module_const_env_hash()}
 save(name,raw);record['domains'][name]={'ops':ophist(raw),'raw_lower_hlo_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_lower_hlo_bytes':len(raw),'lower_s':time.monotonic()-t,'components':components,'flags':kwargs,'carry_aval':[(list(v.shape),str(v.dtype)) for v in jax.tree.leaves(carry) if hasattr(v,'shape')]}
 (a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(name,record['domains'][name]['raw_lower_hlo_sha256'],record['domains'][name]['lower_s'],flush=True)
 native[0]=False;del low,raw;jax.clear_caches()
# Include the ACTUAL shipped fused d02+d03 cascade, not only per-domain steps.
if not a.fused:
 (a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');os._exit(0)
for child in names[1:]:
 edge=next(e for parent in names for e in tree.children(parent) if e.child==child)
 carries[child]=dt._operational_force(edge,carries[edge.parent],carries[child]);jax.block_until_ready(carries[child]);jax.clear_caches()
native[0]=True
fused=dt._operational_fused_cascade_factory(tree)('d02')
assert fused is not None, 'WN3 shipped fused cascade unavailable'
import inspect
captured=inspect.getclosurevars(fused).nonlocals
fused_jit=captured['fused_jit'];aux=captured['fused_aux']
args=jax.tree.map(abstract,captured['_args'](carries['d02'],(carries['d03'],),1,(1,)))
t=time.monotonic();low=fused_jit.lower(*args);raw=low.compiler_ir('hlo').as_hlo_text()
components={'fn_identity':ck.fn_identity_hash(fused_jit),'program_config':ck.program_config_hash(),'source_fingerprint':ck.source_fingerprint_hash(),'static_config':ck.static_config_hash(aux),'carry_aval':ck.carry_aval_hash(args,{}),'trace_env':ck.global_trace_env_hash(),'module_constants':ck.module_const_env_hash()}
save('fused/d02',raw);record['domains']['fused/d02']={'ops':ophist(raw),'raw_lower_hlo_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_lower_hlo_bytes':len(raw),'lower_s':time.monotonic()-t,'components':components}
print('fused/d02',record['domains']['fused/d02']['raw_lower_hlo_sha256'],record['domains']['fused/d02']['lower_s'],flush=True)
record['host_vmhwm_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
(a.out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');os._exit(0)
