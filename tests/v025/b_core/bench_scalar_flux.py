"""Complete original WRF scalar tendency vs fp64/fp32 JAX and native faces."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

from scalar_oracle import build,evaluate
from bench_flux import error
from gpuwrf.dynamics.flux_advection import CoupledVelocities,advect_scalar_flux
from gpuwrf.kernels.dyn_flux_fp32 import specified_flux_faces_fp32,advect_scalar_flux_fp32

BASE=Path("<USER_HOME>/wrf_gpu2_lanes/b-core")


def inputs(domain,crop=False):
    with np.load(BASE/"phase1/fixtures"/(domain+".npz")) as d:
        q=d["state_theta_1"].astype(np.float32)
        nz,ny,nx=q.shape
        c1=d["state_c1h"].astype(np.float32);c2=d["state_c2h"].astype(np.float32)
        def coupled(direction,metric):
            mass=d["state_mu"+direction].astype(np.float32)
            wind=d["state_"+direction+"_1"].astype(np.float32)
            return (c1[:,None,None]*mass+c2[:,None,None])*wind/d["state_"+metric].astype(np.float32)
        ru=coupled("u","msfuy");rv=coupled("v","msfvx")
        map=d["state_msftx"].astype(np.float32);dnw=d["state_dnw"].astype(np.float32)
        rdx=np.float32(1/(9000. if domain=="d01" else 3000.))
        div=dnw[:,None,None]*map[None]*(rdx*(ru[:,:,1:]-ru[:,:,:-1])+rdx*(rv[:,1:]-rv[:,:-1]))
        dm=np.zeros((ny,nx),np.float32)
        for k in range(nz):dm=dm+div[k]
        rom=np.zeros((nz+1,ny,nx),np.float32)
        for k in range(1,nz):rom[k]=rom[k-1]-dnw[k-1]*c1[k-1]*dm-div[k-1]
        arrays=dict(q=q,ru=ru[:,:,:nx],rv=rv[:,:ny],rom=rom,mu=d["state_muts"].astype(np.float32),map=map,
            fzm=d["state_fnm"].astype(np.float32),fzp=d["state_fnp"].astype(np.float32),rdzw=d["state_rdnw"].astype(np.float32),c1=c1,c2=c2)
    assert np.count_nonzero(rom)>0
    if crop:
        for key,value in arrays.items():
            if value.ndim>=2:arrays[key]=value[..., :8,:16]
    return arrays,float(rdx)


def function(mode,rdx,interpret=False):
    if mode=="fused32":
        return jax.jit(lambda q,ru,rv,rom,mu,map,fzm,fzp,rdzw,c1,c2:advect_scalar_flux_fp32(q,ru,rv,rom,map,rdzw,fzm,fzp,rdx,rdx,interpret=interpret))
    op=advect_scalar_flux
    if mode=="faces32":
        globals={**op.__globals__,"specified_flux_faces":lambda q,v,axis:specified_flux_faces_fp32(q,v,axis,interpret=interpret)}
        op=types.FunctionType(op.__code__,globals,"native_faces_scalar",op.__defaults__)
        op.__kwdefaults__=advect_scalar_flux.__kwdefaults__
    def call(q,ru,rv,rom,mu,map,fzm,fzp,rdzw,c1,c2):
        vel=CoupledVelocities(ru=ru,rv=rv,rom=rom,msftx=map,specified=True)
        return op(q,vel,mut=mu,c1=c1,rdx=rdx,rdy=rdx,rdzw=rdzw,fzm=fzm,fzp=fzp)
    return jax.jit(call)


def run(args):
    if Path("/tmp/wrf_gpu2_quiet").exists():raise SystemExit(125)
    records=[];args.out.parent.mkdir(parents=True,exist_ok=True)
    for contraction in ("off","fast"):
        if not (args.oracles/contraction/"scalar.so").exists():build(args.oracles/contraction,contraction)
    for domain in ("d01","d02"):
        if Path("/tmp/wrf_gpu2_quiet").exists():raise SystemExit(125)
        arrays,rdx=inputs(domain,args.cpu)
        ref=evaluate(args.oracles/"off/scalar.so",arrays,rdx,domain=="d02")
        fast=evaluate(args.oracles/"fast/scalar.so",arrays,rdx,domain=="d02")
        for mode in args.modes:
            dtype=jnp.float64 if mode=="jax64" else jnp.float32
            values=[jnp.asarray(arrays[key],dtype) for key in ("q","ru","rv","rom","mu","map","fzm","fzp","rdzw","c1","c2")]
            fn=function(mode,rdx,args.cpu);started=time.perf_counter();got=fn(*values);jax.block_until_ready(got)
            result=dict(domain=domain,mode=mode,shape=list(arrays["q"].shape),backend=jax.devices()[0].platform,device=jax.devices()[0].device_kind,
                errors=error(ref,got,fast,ref),first_s=time.perf_counter()-started,rom_nonzero=int(np.count_nonzero(arrays["rom"])))
            records.append(result);args.out.write_text(json.dumps(records,indent=2)+"\n")
            assert result["errors"]["passed"],result
            if not args.cpu:
                compiled=fn.lower(*values).compile();hlo=compiled.as_text()
                (args.out.parent/(domain+"_"+mode+".hlo.txt")).write_text(hlo)
                result.update(f64_tokens=hlo.count("f64"),hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())
                for _ in range(3):jax.block_until_ready(compiled(*values))
                runtime=ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so");assert runtime.cudaProfilerStart()==0
                samples=[]
                for _ in range(11):
                    started=time.perf_counter()
                    for _ in range(32):output=compiled(*values)
                    jax.block_until_ready(output);samples.append((time.perf_counter()-started)/32)
                assert runtime.cudaProfilerStop()==0
                result.update(samples_s=samples,median_s=statistics.median(samples))
                args.out.write_text(json.dumps(records,indent=2)+"\n")
            print(json.dumps({k:v for k,v in result.items() if k!="samples_s"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cpu",action="store_true");p.add_argument("--out",type=Path,required=True)
    p.add_argument("--modes",nargs="+",default=["jax64","jax32","faces32"])
    p.add_argument("--oracles",type=Path,default=BASE/"phase2b/BC30/oracles");args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stat=Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")",1)[1].split()
    args.out.with_suffix(".pid.json").write_text(json.dumps(dict(pid=os.getpid(),starttime=stat[19]))+"\n")
    assert jax.devices()[0].platform==("cpu" if args.cpu else "gpu"),jax.devices()
    if not args.cpu:assert os.environ.get("GPUWRF_GPU_LOCK_HELD")=="1"
    run(args)
