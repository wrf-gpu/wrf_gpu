"""Profile the actual d01 Noah caller, including its mixed interface shapes."""
import argparse
import ctypes
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--barrier',choices=['0','1'],required=True)
    p.add_argument('--transpose-consumer',action='store_true',
                   help='One bounded attempt to reproduce the root transpose fusion')
    args=p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():return 125
    os.environ['GPUWRF_NOAHMP_NATIVE_REAL']='1'
    os.environ['GPUWRF_MYNN_FP32_COLUMNS']='1'
    os.environ['GPUWRF_NOAHMP_ITERATION_BARRIER']=args.barrier
    sys.path.insert(0,str(args.repo/'src'))
    import gpuwrf,jax,nvtx
    from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
    assert jax.devices()[0].platform=='gpu'
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
    args.out.mkdir(parents=True,exist_ok=True)
    with args.inputs.open('rb') as f:view,land,static,rad,clock,ep,rp,dt=pickle.load(f)
    rad=dict(soldn=rad.soldn,lwdn=rad.lwdn,cosz=rad.cosz)
    clock=dict(julian=clock.julian,yearlen=clock.yearlen)
    view,land,static,rad,clock,ep,rp=jax.tree.map(
        lambda x:jax.device_put(x) if hasattr(x,'dtype') else x,
        (view,land,static,rad,clock,ep,rp))
    def call(v,l):
        result=noahmp_surface_adapter(v,l,static,radiation=rad,clock=clock,
                       dt=dt,energy_params=ep,rad_params=rp)
        if args.transpose_consumer:
            import jax.numpy as jnp
            consumer=jax.tree.map(
                lambda x:jnp.transpose(x.astype(jnp.float64))
                if hasattr(x,'dtype') and x.ndim==2 and jnp.issubdtype(x.dtype,jnp.floating)
                else x,result)
            return result,consumer
        return result
    fn=jax.jit(call)
    t0=time.perf_counter();compiled=fn.lower(view,land).compile();setup=time.perf_counter()-t0
    result=compiled(view,land);jax.block_until_ready(result)
    host=jax.device_get(result)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(host)
               if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.floating))
    np.savez(args.out/'outputs.npz',**{f'leaf_{i}':x for i,x in enumerate(jax.tree.leaves(host))})
    for i,m in enumerate(compiled.runtime_executable().hlo_modules()):
        (args.out/f'module_{i}.optimized.hlo').write_text(m.to_string())
    times=[]
    for _ in range(5):
        t0=time.perf_counter();jax.block_until_ready(compiled(view,land));times.append(time.perf_counter()-t0)
    receipt=dict(backend='gpu',device=str(jax.devices()[0]),barrier=args.barrier,
                 shape=list(land.tv.shape),dt=dt,setup_s=setup,wall_s=times,finite=True,
                 input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                 xla_flags=os.environ.get('XLA_FLAGS'),scope='Noah d01 caller only; actual IT07 pack initial state')
    receipt['transpose_consumer']=args.transpose_consumer
    (args.out/'warm.json').write_text(json.dumps(receipt,indent=2)+'\n')
    cudart=ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
    assert cudart.cudaProfilerStart()==0
    try:
        for i in range(3):
            with nvtx.annotate(f'NOAHMP:{args.barrier}:caller_d01:{i}'):
                jax.block_until_ready(compiled(view,land))
    finally:cudart.cudaProfilerStop()
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
