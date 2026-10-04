"""Real day/night MYNN plume probe, plus independent WRF GPU oracle."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs',required=True,type=Path)
parser.add_argument('--domain',choices=('d01','d02'),required=True)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--native',type=int,choices=(0,1),required=True)
parser.add_argument('--profile',action='store_true')
parser.add_argument('--oracle',action='store_true')
args=parser.parse_args()
if os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1':
    raise RuntimeError('requires GPU lock')
os.environ['GPUWRF_EDMF_FUSED_PLUME']='0'
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_edmf as E
from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
assert jax.devices()[0].platform == 'gpu'
if jax.default_backend() not in ('gpu','cuda'):
    raise RuntimeError('requires real GPU')
args.output.mkdir(parents=True,exist_ok=True)
if args.oracle:
    path=Path(__file__).resolve().parents[2]/'test_mynn_edmf_oracle.py'
    spec=importlib.util.spec_from_file_location('b_phys_gpu_mynn_oracle',path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    column=json.loads(Path(oracle.COL).read_text())
    E.dmp_mf_columns=lambda *a,**kw:dmp_mf_columns_native(*a,interpret=False,**kw)
    oracle.test_dmp_mf_matches_wrf_oracle(column)
    (args.output/'oracle.json').write_text(json.dumps(dict(pass_wrf=True,backend=jax.default_backend(),tolerance=oracle.TOL_REL))+'\n')
    print('GPU WRF plume oracle PASS')
    raise SystemExit(0)
meta=json.loads((args.inputs/f'{args.domain}.json').read_text())
with np.load(args.inputs/f'{args.domain}.npz') as data:
    kwargs={k:jnp.asarray(data[k]) for k in data.files}
def run(**kw):
    return dmp_mf_columns_native(interpret=False,**kw) if args.native else E.dmp_mf_columns(**kw)
fn=jax.jit(run)
start=time.perf_counter()
compiled=fn.lower(**kwargs).compile()
jax.block_until_ready(compiled(**kwargs))
record=dict(native=args.native,backend=jax.default_backend(),shape=meta['shape'],
            inputs_sha256=hashlib.sha256((args.inputs/f'{args.domain}.npz').read_bytes()).hexdigest(),
            compile_first_s=time.perf_counter()-start,cache=os.environ.get('GPUWRF_JAX_CACHE_DIR'),
            dt=meta['dt'],dx=meta['dx'])
if args.profile:
    cuda=ctypes.CDLL('libcudart.so.12')
    assert cuda.cudaProfilerStart()==0
    with jax.profiler.TraceAnnotation(f'BP_MYNN:{args.domain}:native{args.native}'):
        result=compiled(**kwargs)
        jax.block_until_ready(result)
    assert cuda.cudaProfilerStop()==0
else:
    samples=[]
    for _ in range(9):
        start=time.perf_counter()
        result=compiled(**kwargs)
        jax.block_until_ready(result)
        samples.append((time.perf_counter()-start)*1000)
    record['wall_ms_samples']=samples
    record['wall_ms_median']=statistics.median(samples)
    record['nonfinite_outputs']=int(jax.device_get(sum(jnp.sum(~jnp.isfinite(x)) for x in result.values())))
    record['active_columns']=int(jax.device_get(jnp.sum(result['active'])))
(args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
