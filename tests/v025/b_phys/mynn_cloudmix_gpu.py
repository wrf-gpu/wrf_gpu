"""Asserted-GPU whole-MYNN original-object mean gates on full PROD shapes."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',required=True,type=Path)
parser.add_argument('--period',required=True,choices=('night','day'))
parser.add_argument('--domain',required=True,choices=('d01','d02'))
parser.add_argument('--native',required=True,type=int,choices=(0,1))
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--profile',action='store_true')
parser.add_argument('--input-prefix',default='columns_p0')
parser.add_argument('--oracle-dir',default='BP29')
parser.add_argument('--cold-oracle-dir')
args=parser.parse_args()
os.environ['GPUWRF_MYNN_CLOUDMIX']='1'
os.environ['GPUWRF_MYNN_FP32_COLUMNS']=str(args.native)
import jax
import jax.numpy as jnp
from gpuwrf.physics import mynn_pbl as P
from mynn_wrf_gates import inputs,compare
assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
assert jax.devices()[0].platform=='gpu'
state,flux,meta=inputs(args.root,args.period,args.domain,selected=False,dtype=jnp.float32 if args.native else jnp.float64,source_forcing=True,input_prefix=args.input_prefix)
def step(state,flux):
 return P.step_mynn_pbl_column_with_pblh(state,meta['dt'],surface=flux,edmf=True,dx=meta['dx'])
lowered=jax.jit(step).lower(state,flux)
compiled=lowered.compile()
jax.block_until_ready(compiled(state,flux))
if args.profile:
 cuda=ctypes.CDLL('libcudart.so.12')
 assert cuda.cudaProfilerStart()==0
 result=compiled(state,flux);jax.block_until_ready(result)
 assert cuda.cudaProfilerStop()==0
else:
 result=compiled(state,flux);jax.block_until_ready(result)
receipt=compare(args.root,args.period,args.domain,state,result,full_shape=True,oracle_dir=args.oracle_dir)
receipt.update(native=args.native,backend=jax.default_backend(),device_platform=jax.devices()[0].platform,
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               shape=meta['shape'],dt=meta['dt'],dx=meta['dx'],
               forcing='WRF REAL hfx/qfx round trip and MYNN virtual flux; pristine gas constants',
               input_prefix=args.input_prefix,oracle_dir=args.oracle_dir,
               f64_hlo_lines=lowered.compiler_ir(dialect='hlo').as_hlo_text().count('= f64['),
               output_dtypes=sorted({str(v.dtype) for v in jax.tree_util.tree_leaves(result)}))
if args.cold_oracle_dir:
 import numpy as np
 indices=np.asarray(meta['columns'])
 sample=lambda v:v[indices] if v.ndim else v
 cold_state=jax.tree.map(sample,state)
 cold_flux=jax.tree.map(sample,flux)
 with jax.enable_x64(not bool(args.native)):
  cold=jax.jit(lambda s,f:P.mynn_coldstart_init_columns(s,f.ustar,meta['dx'],f.xland))
  qke=np.asarray(cold(cold_state,cold_flux)[0])
 lines=(args.root/args.cold_oracle_dir/f'{args.period}_{args.domain}.output').read_text().splitlines()
 truth=np.asarray([[float(v) for v in lines[2+45*i:46+45*i]] for i in range(len(indices))])
 delta=np.abs(qke-truth)
 median=float(np.median(delta/np.maximum(np.abs(truth),1e-8)))
 receipt['cold_initializer']=dict(relative_median=median,frozen_limit=1e-4,
   pass_frozen=median<=1e-4,max_abs=float(delta.max()),
   scope='original WRF subroutines, driver mapping, deterministic rmol0; historical median gate')
 assert receipt['cold_initializer']['pass_frozen'],receipt['cold_initializer']
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
assert receipt['mean_gate_pass'] and receipt['nonfinite']==0,receipt
if args.native:assert receipt['output_dtypes']==['float32'] and receipt['f64_hlo_lines']==0,receipt
