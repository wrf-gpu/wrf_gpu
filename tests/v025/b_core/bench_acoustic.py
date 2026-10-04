"""Real-derived stage oracle and warm acoustic benchmark (run under GPU lock)."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("JAX_ENABLE_X64","true")
os.environ["JAX_PALLAS_USE_MOSAIC_GPU"]="false"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig,AcousticCoreState  # noqa: E402
from gpuwrf.kernels.dyn_acoustic_fp32 import acoustic_substep_fp32,calc_coef_fp32,_w_phase,evolving_payload  # noqa: E402

UPDATED=("u","v","w","ph","mu","muts","muave","mudf","ww","theta",
         "theta_coupled_work","theta_ave","t_2ave","p","al","pm1","ru_m","rv_m","ww_m")


def fixture(root,domain,dtype=jnp.float32):
    data=np.load(root/(domain+".npz"))
    meta=json.loads((root/(domain+".json")).read_text())
    state=AcousticCoreState(**{key[6:]:jnp.asarray(data[key],dtype) for key in data.files if key.startswith("state_")})
    if state.mu_work is None:
        # Initial fixtures predate the explicit leaf: derive ONCE in fp64
        # outside the step, before rounding, never from fp32 totals.
        state=state.replace(mu_work=jnp.asarray(data["state_muts"]-data["state_mut"],dtype))
    return state,AcousticCoreConfig(**meta["config"]),data,meta


def compare(ref,got):
    ref=np.asarray(ref,dtype=np.float64);got=np.asarray(got,dtype=np.float64)
    err=got-ref
    rms=float(np.sqrt(np.mean(err**2)))
    maxabs=float(np.max(np.abs(err)))
    rms_ref=float(np.sqrt(np.mean(ref**2)))
    max_ref=float(np.max(np.abs(ref)))
    return dict(rms=rms,max_abs=maxabs,rms_ref=rms_ref,max_ref=max_ref,
                rms_limit=2e-5*max(1.,rms_ref),max_limit=2e-4*max(1.,max_ref),
                finite=bool(np.isfinite(got).all()),
                passed=bool(np.isfinite(got).all() and rms<=2e-5*max(1.,rms_ref) and maxabs<=2e-4*max(1.,max_ref)))


def verify(root,domain,interpret=False):
    state,cfg,data,meta=fixture(root,domain)
    coef_fn=jax.jit(lambda s:calc_coef_fp32(s,cfg,interpret=interpret))
    started=time.monotonic()
    coef=coef_fn(state)
    jax.block_until_ready(coef)
    print(domain,"coef ready",round(time.monotonic()-started,2),flush=True)
    result={"domain":domain,"shape":list(state.theta.shape),"fixture_sha256":meta["payload_sha256"],
            "coefficient_errors":{name:compare(data[name],value) for name,value in coef.items()},"steps":{}}
    fn=jax.jit(lambda s:acoustic_substep_fp32(s,coefficients=coef,cfg=cfg,interpret=interpret,return_guard_events=True))
    current=state
    for step in range(1,12):
        current,guard=fn(current)
        if step in (1,11):
            jax.block_until_ready(current)
            result["steps"][str(step)]={name:compare(data[f"ref{step}_"+name],getattr(current,name)) for name in UPDATED}
            result.setdefault("guard_events",{})[str(step)]=int(np.asarray(guard).sum())
            failures={name:val for name,val in result["steps"][str(step)].items() if not val["passed"]}
            print(domain,"step",step,"failures",json.dumps(failures),flush=True)
    result["passed"]=all(v["passed"] for v in result["coefficient_errors"].values()) and all(
             v["passed"] for step in result["steps"].values() for v in step.values()) and all(
             count==0 for count in result["guard_events"].values())
    result["compile_and_verify_s"]=time.monotonic()-started
    return result


def profile_range(start):
    cudart=ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    fn=cudart.cudaProfilerStart if start else cudart.cudaProfilerStop
    rc=fn()
    if rc:
        raise RuntimeError(f"cudaProfiler API failed: {rc}")


def bench(root,domain,profile=False,reps=31,batch=32,family="substep",payload="full"):
    state,cfg,data,meta=fixture(root,domain)
    cfg=replace(cfg,dt=cfg.dt_full/4.)
    coef=jax.jit(lambda s:calc_coef_fp32(s,cfg))(state)
    def substep(s):
        out=acoustic_substep_fp32(s,coefficients=coef,cfg=cfg)
        return evolving_payload(out) if payload=="evolving" else out
    fn=jax.jit(substep if family=="substep" else
               (lambda s:_w_phase(s,coef,cfg,interpret=False)) if family=="w" else
               (lambda s:calc_coef_fp32(s,cfg)))
    compiled=fn.lower(state).compile()
    lowered=fn.lower(state).compiler_ir(dialect="stablehlo")
    hlo=str(lowered)
    for _ in range(3):
        jax.block_until_ready(compiled(state))
    samples=[]
    if profile:
        profile_range(True)
    for _ in range(reps):
        start=time.perf_counter()
        # Independent repeated substeps exclude input preparation and host transfers.
        for _ in range(batch):
            result=compiled(state)
        jax.block_until_ready(result)
        samples.append((time.perf_counter()-start)/batch)
    if profile:
        profile_range(False)
    return dict(domain=domain,family=family,payload=payload,shape=list(state.theta.shape),dts=cfg.dt,
                fixture_sha256=meta["payload_sha256"],warm_dispatch_s=samples,
                median_dispatch_s=statistics.median(samples),
                min_dispatch_s=min(samples),reps=reps,batch=batch,
                stablehlo_f64_tokens=hlo.count("f64"),
                stablehlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",type=Path,default=Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/fixtures"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("verify","bench","profile"),default="verify")
    p.add_argument("--domain",choices=("d01","d02","both"),default="both")
    p.add_argument("--family",choices=("substep","w","coef"),default="substep")
    p.add_argument("--payload",choices=("full","evolving"),default="full")
    p.add_argument("--reps",type=int,default=31)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--interpret",action="store_true")
    args=p.parse_args()
    if not args.interpret and (not os.environ.get("GPUWRF_GPU_LOCK_HELD") or jax.devices()[0].platform!="gpu"):
        raise RuntimeError("device harness requires the b-core GPU lock")
    results=[]
    for domain in (("d01","d02") if args.domain=="both" else (args.domain,)):
        result=(verify(args.fixtures,domain,args.interpret) if args.mode=="verify" else
                bench(args.fixtures,domain,args.mode=="profile",args.reps,args.batch,args.family,args.payload))
        results.append(result)
        print(json.dumps({key:value for key,value in result.items() if key not in ("steps","warm_dispatch_s","coefficient_errors")}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,indent=2)+"\n")
    return 0 if args.mode!="verify" or all(r["passed"] for r in results) else 1


if __name__=="__main__":
    raise SystemExit(main())
