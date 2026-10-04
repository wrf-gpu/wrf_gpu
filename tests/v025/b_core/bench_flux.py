"""Real-history WRF face-stencil gate and fp64/JAX-fp32/Pallas bake-off."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from flux_oracle import evaluate
from gpuwrf.dynamics.flux_advection import specified_flux_faces
from gpuwrf.kernels.dyn_flux_fp32 import specified_flux_faces_fp32

BASE=Path("<USER_HOME>/wrf_gpu2_lanes/b-core")


def inputs(domain,axis,upstream,crop):
    path=BASE/"phase1/fixtures"/(domain+".npz")
    with np.load(path) as data:
        field=data["state_theta_1"]
        nz,ny,nx=field.shape
        direction="u" if axis==2 else "v"
        wind=data["state_"+direction+"_1"][...,:ny,:nx]
        mass=data["state_mu"+direction][...,:ny,:nx]
        c1,c2=data["state_c1h"],data["state_c2h"]
        metric=data["state_msfuy" if axis==2 else "state_msfvx"][...,:ny,:nx]
        velocity=(c1[:,None,None]*mass+c2[:,None,None])*wind/metric
        if upstream:field=wind
        if crop:field,velocity=field[:,:8,:16],velocity[:,:8,:16]
        # This is WRF REAL input rounding before arithmetic, for all arms.
        return field.astype(np.float32),velocity.astype(np.float32),str(path)


def error(ref,got,fast,off):
    ref=np.asarray(ref,np.float64);got=np.asarray(got,np.float64)
    d=got-ref;spread=np.asarray(fast,np.float64)-np.asarray(off,np.float64)
    rms=float(np.sqrt(np.mean(d*d)));mx=float(np.max(np.abs(d)))
    radius=2e-5*max(1.,float(np.sqrt(np.mean(ref*ref))))
    maxradius=2e-4*max(1.,float(np.max(np.abs(ref))))
    crms=float(np.sqrt(np.mean(spread*spread)));cmax=float(np.max(np.abs(spread)))
    return dict(rms=rms,max_abs=mx,frozen_rms_limit=radius,frozen_max_limit=maxradius,
        compiler_rms=crms,compiler_max=cmax,tier_p_rms_limit=radius+2*crms,
        tier_p_max_limit=maxradius+2*cmax,finite=bool(np.isfinite(got).all()),
        frozen_passed=bool(rms<=radius and mx<=maxradius),
        passed=bool(np.isfinite(got).all() and rms<=radius+2*crms and mx<=maxradius+2*cmax))


def run(args):
    records=[]
    for domain in ("d01","d02"):
        for axis in (1,2):
            for upstream in (False,True):
                q,v,path=inputs(domain,axis,upstream,args.interpret)
                ref=evaluate(args.oracles/"off"/"faces.so",q,v,axis,upstream)
                fast=evaluate(args.oracles/"fast"/"faces.so",q,v,axis,upstream)
                for mode in (("jax32","pallas32") if args.interpret else ("jax64","jax32","pallas32")):
                    dtype=jnp.float64 if mode=="jax64" else jnp.float32
                    a,b=jnp.asarray(q,dtype),jnp.asarray(v,dtype)
                    function=(lambda a,b:specified_flux_faces_fp32(a,b,axis,upstream=upstream,interpret=args.interpret)) if mode=="pallas32" else (lambda a,b:specified_flux_faces(a,b,axis,upstream=upstream))
                    fn=jax.jit(function)
                    started=time.perf_counter();got=fn(a,b);jax.block_until_ready(got)
                    verdict=error(ref,got,fast,ref)
                    result=dict(domain=domain,axis=axis,upstream=upstream,mode=mode,
                        shape=list(q.shape),input_path=path,errors=verdict,
                        first_s=time.perf_counter()-started,backend=jax.devices()[0].platform,
                        device=jax.devices()[0].device_kind)
                    records.append(result)
                    assert verdict["passed"],result
                    if not args.interpret:
                        compiled=fn.lower(a,b).compile()
                        hlo=str(fn.lower(a,b).compiler_ir(dialect="stablehlo"))
                        result.update(f64_tokens=hlo.count("f64"),hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())
                        for _ in range(3):jax.block_until_ready(compiled(a,b))
                        if args.profile:
                            runtime=ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
                            assert runtime.cudaProfilerStart()==0
                        samples=[]
                        with jax.profiler.TraceAnnotation(f"BFLUX:{domain}:{axis}:{int(upstream)}:{mode}"):
                            for _ in range(11 if args.profile else 31):
                                start=time.perf_counter()
                                for _ in range(32):output=compiled(a,b)
                                jax.block_until_ready(output);samples.append((time.perf_counter()-start)/32)
                        if args.profile:runtime.cudaProfilerStop()
                        result.update(samples_s=samples,median_s=statistics.median(samples))
                    print(json.dumps({k:v for k,v in result.items() if k not in ("samples_s",)}),flush=True)
    args.out.write_text(json.dumps(records,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--interpret",action="store_true")
    p.add_argument("--profile",action="store_true")
    p.add_argument("--oracles",type=Path,default=BASE/"phase2b/BC29/oracles")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    stat=Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")",1)[1].split()
    a.out.with_suffix(".pid.json").write_text(json.dumps(dict(pid=os.getpid(),starttime=stat[19]))+"\n")
    if not a.interpret:
        assert os.environ.get("GPUWRF_GPU_LOCK_HELD")=="1"
        assert jax.devices()[0].platform=="gpu",jax.devices()
    run(a)
