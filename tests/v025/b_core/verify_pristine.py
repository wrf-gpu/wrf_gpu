"""Independent binary32 Fortran savepoint checks for all acoustic phases."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

from bench_acoustic import fixture,compare
from pristine_oracle import Oracle,build
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.kernels.dyn_acoustic_fp32 import acoustic_substep_fp32,calc_coef_fp32,evolving_payload,state_from_payload

ROOT=Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1")
FIELDS={"u":"u","v":"v","w":"w","ph":"ph","mu_work":"mu","muts":"muts",
        "muave":"muave","mudf":"mudf","ww":"ww","theta_coupled_work":"t",
        "t_2ave":"t_2ave","p":"p","al":"al","pm1":"pm1"}


def run(domain,*,crop=False,interpret=False,steps=1,dts=None,payload="full"):
    state,cfg,_,meta=fixture(ROOT/"fixtures",domain)
    if crop:
        ny,nx=8,16
        oldny,oldnx=state.theta.shape[1:]
        updates={}
        for name,value in state.to_dict().items():
            if value is None:continue
            if value.ndim>=2:
                updates[name]=value[..., :ny+int(value.shape[-2]==oldny+1),
                                     :nx+int(value.shape[-1]==oldnx+1)]
        state=state.replace(**updates)
    # Pristine advance_w receives stage rw_tend after W damping. The routine
    # does not apply the legacy additional per-substep vertical-CFL guard.
    cfg=replace(cfg,w_damping=0,dt=cfg.dt_full/4. if dts is None else dts)
    lib=ROOT/"pristine_corrected/reference/oracle.so"
    if not lib.exists():build(lib.parent)
    fastlib=ROOT/"pristine_corrected/fast/oracle.so"
    offlib=ROOT/"pristine_corrected/off/oracle.so"
    for path,contract in ((fastlib,"fast"),(offlib,"off")):
        if not path.exists():build(path.parent,contraction=contract,native=True)
    oracle=Oracle(lib,state,cfg)
    fast=Oracle(fastlib,state,cfg)
    nativeoff=Oracle(offlib,state,cfg)
    oracle.run(1)
    fast.run(1);nativeoff.run(1)
    co=jax.jit(lambda s:calc_coef_fp32(s,cfg,interpret=interpret))(state)
    coefficients={name:compare(oracle.field(name)[1:],value[1:]) for name,value in co.items()}
    def substep(s):
        result,events=acoustic_substep_fp32(s,coefficients=co,cfg=cfg,interpret=interpret,return_guard_events=True)
        return (evolving_payload(result) if payload=="evolving" else result),events
    step_fn=jax.jit(substep)
    result=dict(domain=domain,payload=payload,shape=list(state.theta.shape),dts=cfg.dt,fixture_sha256=meta["payload_sha256"],
                oracle="unchanged pristine Fortran binary32 routines",coefficients=coefficients,steps={})
    for step in range(1,steps+1):
        for phase in range(2,6):
            oracle.run(phase);fast.run(phase);nativeoff.run(phase)
        updated,events=step_fn(state)
        state=state_from_payload(state,updated) if payload=="evolving" else updated
        errors={name:compare(oracle.field(ref),getattr(state,name)) for name,ref in FIELDS.items()}
        for name,ref in FIELDS.items():
            delta=fast.field(ref).astype(np.float64)-nativeoff.field(ref).astype(np.float64)
            compiler_rms=float(np.sqrt(np.mean(delta*delta)))
            compiler_max=float(np.max(np.abs(delta)))
            e=errors[name]
            e.update(frozen_passed=e["passed"],compiler_rms=compiler_rms,compiler_max=compiler_max,
                     tier_p_rms_limit=e["rms_limit"]+2.*compiler_rms,
                     tier_p_max_limit=e["max_limit"]+2.*compiler_max)
            e["passed"]=bool(e["finite"] and e["rms"]<=e["tier_p_rms_limit"] and e["max_abs"]<=e["tier_p_max_limit"])
        result["steps"][str(step)]=errors
        result.setdefault("guard_events",{})[str(step)]=int(np.asarray(events).sum())
        print(domain,"step",step,json.dumps({k:v for k,v in errors.items() if not v["passed"]}),flush=True)
    result["passed"]=all(v["passed"] for v in coefficients.values()) and all(
        v["passed"] for errors in result["steps"].values() for v in errors.values()) and all(
        v==0 for v in result["guard_events"].values())
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--domain",choices=("d01","d02"),default="d01")
    p.add_argument("--crop",action="store_true")
    p.add_argument("--interpret",action="store_true")
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--wrf-stages",action="store_true")
    p.add_argument("--sound-steps",type=int,default=4)
    p.add_argument("--payload",choices=("full","evolving"),default="full")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if not a.interpret:
        if not os.environ.get("GPUWRF_GPU_LOCK_HELD") or jax.devices()[0].platform != "gpu":
            raise RuntimeError("device oracle requires GPU lock and GPU backend")
        print("asserted device backend:", jax.devices()[0].platform,
              jax.devices()[0].device_kind, flush=True)
    if a.wrf_stages:
        full_dt=54. if a.domain=="d01" else 18.
        if a.sound_steps<2:raise ValueError("sound steps must be >=2")
        trips=(1,max(1,a.sound_steps//2),a.sound_steps)
        stages=[run(a.domain,crop=a.crop,interpret=a.interpret,steps=n,dts=dt,payload=a.payload)
                for n,dt in ((trips[0],full_dt/3.),(trips[1],full_dt/a.sound_steps),(trips[2],full_dt/a.sound_steps))]
        result=dict(domain=a.domain,protocol=f"resolved WRF RK trip counts {trips[0]}/{trips[1]}/{trips[2]}; separate stage savepoint arms",
                    stages=stages,passed=all(stage["passed"] for stage in stages))
    else:
        result=run(a.domain,crop=a.crop,interpret=a.interpret,steps=a.steps,payload=a.payload)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("steps","coefficients","stages")}),flush=True)
    raise SystemExit(0 if result["passed"] else 1)
