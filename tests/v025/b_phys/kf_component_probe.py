"""Real-column KF component lower/timing probe (no forecast speed claim)."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs',required=True,type=Path)
parser.add_argument('--domain',choices=('d01','d02'),required=True)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--resident',type=int,choices=(0,1),required=True)
parser.add_argument('--lower',action='store_true')
parser.add_argument('--profile',action='store_true')
parser.add_argument('--repeats',type=int,default=9)
args=parser.parse_args()
os.environ['GPUWRF_KF_RESIDENT_TABLES']=str(args.resident)
if not args.lower and os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1':
    raise RuntimeError('GPU probe requires the lane lock')
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
                          w=w,nca=nca)
    return (result.tendency.state_tendencies,
            result.tendency.accumulator_increments,result.carry.cumulus)

fn=jax.jit(jax.vmap(one))
args.output.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
lowered=fn.lower(*inputs)
record=dict(metadata=meta,resident=args.resident,backend=jax.default_backend(),
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
    jax.block_until_ready(compiled(*inputs))
    record['compile_and_first_call_s']=time.perf_counter()-started
    if args.profile:
        cuda=ctypes.CDLL('libcudart.so.12')
        jax.block_until_ready(inputs)
        assert cuda.cudaProfilerStart()==0
        with jax.profiler.TraceAnnotation(f'BP_KF:{args.domain}:resident{args.resident}'):
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
(args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='metadata'}))
