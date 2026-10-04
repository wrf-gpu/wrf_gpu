"""Real-column KF component lower/timing probe (no forecast speed claim)."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs',required=True,type=Path)
parser.add_argument('--source-tree',type=Path,default=Path(__file__).resolve().parents[3])
parser.add_argument('--domain',choices=('d01','d02'),required=True)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--native',type=int,choices=(0,1),required=True)
parser.add_argument('--lower',action='store_true')
parser.add_argument('--profile',action='store_true')
parser.add_argument('--repeats',type=int,default=9)
args=parser.parse_args()
sys.path.insert(0,str(args.source_tree/'src'))
os.environ['GPUWRF_KF_RESIDENT_TABLES']='1'
os.environ['GPUWRF_KF_COLUMN_FP32']=str(args.native)
if not args.lower and os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1':
    raise RuntimeError('GPU probe requires the lane lock')
os.environ.setdefault('JAX_PALLAS_USE_MOSAIC_GPU','false')
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics.cumulus_kf import step_kf_column

if not args.lower:
    assert jax.devices()[0].platform == 'gpu', 'GPU probe must not run on CPU'

meta=json.loads((args.inputs/f'{args.domain}.json').read_text())
with np.load(args.inputs/f'{args.domain}.npz') as data:
    keys=('T0','QV0','P0','DZQ','RHOE','w0avg','U0','V0','w','nca')
    inputs=tuple(jnp.asarray(data[k]) for k in keys)

def one(T0,QV0,P0,DZQ,RHOE,w0avg,U0,V0,w,nca):
    result=step_kf_column(T0,QV0,P0,DZQ,RHOE,w0avg,U0,V0,meta['dt'],meta['dx'],
                          w=None if meta.get('w0avg_after_cps', False) else w,nca=nca)
    return (result.tendency.state_tendencies,
            result.tendency.accumulator_increments,result.carry.cumulus,
            result.diagnostics.cumulus)

fn=jax.jit(jax.vmap(one))
args.output.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
lowered=fn.lower(*inputs)
record=dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_tree=str(args.source_tree),
            source_sha256={name:hashlib.sha256((args.source_tree/'src/gpuwrf'/name).read_bytes()).hexdigest() for name in ('physics/cumulus_kf.py','kernels/phys_kf_column.py')},metadata=meta,native=args.native,backend=jax.default_backend(),
            inputs_sha256=hashlib.sha256((args.inputs/f'{args.domain}.npz').read_bytes()).hexdigest(),
            cache=os.environ.get('GPUWRF_JAX_CACHE_DIR','disabled'),
            scope='isolated real-column component, fresh cloud carry; d02 scalability only')
if args.lower:
    text=lowered.as_text(dialect='hlo')
    (args.output/'pre.hlo.txt').write_text(text)
    record['replicated_table_shapes']=dict(
        f64=len(re.findall(r'f64\[\d+,250,220\]',text)),
        f32=len(re.findall(r'f32\[\d+,250,220\]',text)))
    record['lower_s']=time.perf_counter()-started
else:
    compiled=lowered.compile()
    memory=compiled.memory_analysis()
    memory_fields=('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
                   'alias_size_in_bytes','generated_code_size_in_bytes',
                   'host_argument_size_in_bytes','host_output_size_in_bytes',
                   'host_temp_size_in_bytes','host_alias_size_in_bytes')
    record['compiled_memory']={key:int(getattr(memory,key)) for key in memory_fields
                               if hasattr(memory,key)}
    record['input_interfaces']={key:dict(shape=list(value.shape),dtype=str(value.dtype))
                                for key,value in zip(keys,inputs)}
    optimized=compiled.as_text()
    (args.output/'optimized.hlo.txt').write_text(optimized)
    blob=bytes(compiled.runtime_executable().serialize())
    (args.output/'component.xlaexec').write_bytes(blob)
    record['aot']=dict(path=str(args.output/'component.xlaexec'),bytes=len(blob),
                       sha256=hashlib.sha256(blob).hexdigest(),
                       optimized_hlo_sha256=hashlib.sha256(optimized.encode()).hexdigest(),
                       scope='native PJRT program bytes for offline memory/options audit')
    jax.block_until_ready(compiled(*inputs))
    record['compile_and_first_call_s']=time.perf_counter()-started
    if args.profile:
        cuda=ctypes.CDLL(ctypes.util.find_library('cudart') or 'libcudart.so')
        jax.block_until_ready(inputs)
        assert cuda.cudaProfilerStart()==0
        with jax.profiler.TraceAnnotation(f'BKF:{args.domain}:native{args.native}'):
            result=compiled(*inputs)
            jax.block_until_ready(result)
        assert cuda.cudaProfilerStop()==0
    else:
        samples=[]
        for repeat in range(args.repeats):
            start=time.perf_counter()
            result=compiled(*inputs)
            jax.block_until_ready(result)
            samples.append((time.perf_counter()-start)*1000)
        record['wall_ms_samples']=samples
        record['wall_ms_median']=statistics.median(samples)
        record['device_nonfinite_output_count']=int(jax.device_get(
            sum(jnp.sum(~jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(result))))
    record['device_nonfinite_output_count']=int(jax.device_get(
        sum(jnp.sum(~jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(result))))
    record['output_interfaces']=[dict(shape=list(value.shape),dtype=str(value.dtype))
                                 for value in jax.tree_util.tree_leaves(result)]
    arrays=jax.device_get(dict(result[3], nca=result[2]['nca']))
    np.savez(args.output/'outputs.npz', **arrays)
    record['actual_device']=str(jax.devices()[0])
    record['ishall_counts']={str(int(value)):int(count) for value,count in zip(*np.unique(arrays['ishall'],return_counts=True))}
    record['environment']={key:os.environ.get(key,'') for key in ('JAX_PLATFORMS','XLA_FLAGS','GPUWRF_KF_COLUMN_FP32','GPUWRF_KF_RESIDENT_TABLES','GPUWRF_JAX_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR','JAX_PALLAS_USE_MOSAIC_GPU','GPUWRF_XLA_AUTOTUNE_CACHE_DIR','GPUWRF_XLA_PARALLEL_COMPILE')}
(args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='metadata'}))
