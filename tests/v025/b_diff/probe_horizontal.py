"""Real-history horizontal diffusion inputs, oracle, and profiler replay."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import subprocess
import types

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts/v025"),str(ROOT/"tests/v025/b_core")]
if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode")+1]=="fixture":
    import cpu_guard  # Must precede JAX for the real-history loader.
os.environ.setdefault("JAX_ENABLE_X64","true")
os.environ.setdefault("JAX_PALLAS_USE_MOSAIC_GPU","false")
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.dynamics import explicit_diffusion as diff
from gpuwrf.kernels.dyn_diff_fp32 import horizontal_diffusion_fp32
from pristine_diffusion import build,run,MAPS
from bench_acoustic import compare

NAMES=("u","v","w","theta","base","kh","mut","c1h","c2h","c1f","c2f",*MAPS)
DATA=Path("<USER_HOME>/wrf_gpu2_lanes/b-diff/fixtures")


def fixture(domain):
    from real_state import load_real_snapshot
    source=Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/"+("fastbind_r1" if domain=="d01" else "legacybind_r1"))
    snap=load_real_snapshot(source,domain=domain)
    state,m=snap.state,snap.metrics
    dx=9000. if domain=="d01" else 3000.
    def coefficient(s,metric):
        zx,zy,rdzw=diff.wrf_nonperiodic_diffusion_metrics(s.ph_total,dx_m=dx,dy_m=dx)
        deform=diff.horizontal_deformation_2d(s.u,s.v,dx_m=dx,dy_m=dx,
            **{n:getattr(metric,n) for n in (*MAPS,"fnm","fnp","cf1","cf2","cf3","dn","dnw")},
            zx=zx,zy=zy,rdzw=rdzw)
        kh,_=diff.smag2d_horizontal_km(*deform,dx_m=dx,dy_m=dx,msftx=metric.msftx,msfty=metric.msfty)
        return kh.at[:,0,:].set(0).at[:,-1,:].set(0).at[:,:,0].set(0).at[:,:,-1].set(0)
    kh=jax.jit(coefficient)(state,m)
    values=(state.u,state.v,state.w,state.theta,snap.base_state.theta_base,kh,state.mu_total,
            m.c1h,m.c2h,m.c1f,m.c2f,*[getattr(m,n) for n in MAPS])
    DATA.mkdir(parents=True,exist_ok=True)
    path=DATA/(domain+".npz")
    np.savez(path,**{n:np.asarray(v) for n,v in zip(NAMES,values)})
    metadata=dict(source_provenance=snap.provenance,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        work="real CPU-WRF PROD history; retained compute_diff_metrics/cal_deform_and_div/smag2d; operator oracle uses SAME frozen kh",shape=list(state.theta.shape))
    (DATA/(domain+".json")).write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def call(args,dx,arm,interpret=False,scalar_mass=None):
    u,v,w,t,base,kh,mut,c1h,c2h,c1f,c2f,tx,ty,ux,uy,vx,vy=args
    mh=c1h[:,None,None]*mut[None]+c2h[:,None,None]
    mf=c1f[:,None,None]*mut[None]+c2f[:,None,None]
    if arm=="native":
        maps=(tx,ty,ux,uy,vx,vy)
        def kernel(q,k,m,name):
            return horizontal_diffusion_fp32(q,k,m,*maps,name=name,dx=dx,dy=dx,interpret=interpret)
        return (kernel(u,kh,mh,"u"),kernel(v,kh,mh,"v"),kernel(w,kh,mf,"w"),
                kernel(t-base,jnp.float32(3)*kh,mh if scalar_mass is None else scalar_mass,"m"))
    kw=dict(c1h=c1h,c2h=c2h,c1f=c1f,c2f=c2f,msftx=tx,msfty=ty,msfux=ux,msfuy=uy,msfvx=vx,msfvy=vy,dx_m=dx,dy_m=dx)
    return (*diff.wrf_nested_horizontal_diffusion_momentum_tendency(u,v,w,kh,mut,**kw),
        diff.horizontal_diffusion_coord_scalar_tendency(t,3.*kh,mh,dx_m=dx,dy_m=dx,base_3d=base,
            msftx=tx,msfty=ty,msfux=ux,msfuy=uy,msfvx=vx,msfvy=vy,nonperiodic_owned=True))


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("fixture","lower","cpuverify","verify","bench","profile"),required=True)
    p.add_argument("--domain",choices=("d01","d02"),default="d01")
    p.add_argument("--arm",choices=("native","legacy","jax32"),default="native")
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    cpu=a.mode in ("fixture","lower","cpuverify")
    assert jax.devices()[0].platform==("cpu" if cpu else "gpu"),jax.devices()
    if a.mode=="fixture":
        a.out.write_text(json.dumps(fixture(a.domain),indent=2)+"\n");return
    dtype=jnp.float64 if a.arm=="legacy" else jnp.float32
    path=DATA/(a.domain+".npz")
    with np.load(path) as z:args=tuple(jnp.asarray(z[n],dtype) for n in NAMES)
    if a.mode=="cpuverify":
        args=tuple(v[:4+int(i==2),:16+int(i==1),:16+int(i==0)] if i<6 else
                   v[:16,:16] if i==6 else v[:4+int(i in (9,10))] if i<11 else
                   v[:16+int(i in (15,16)),:16+int(i in (13,14))] for i,v in enumerate(args))
    dx=9000. if a.domain=="d01" else 3000.
    result=dict(platform=jax.devices()[0].platform,domain=a.domain,arm=a.arm,fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        cache={k:os.environ.get(k) for k in ("JAX_COMPILATION_CACHE_DIR","GPUWRF_XLA_AUTOTUNE_CACHE_DIR")},
        work=dict(dx=dx,dy=dx,fields="u/v/w/theta",diff_opt=1,km_opt=4,cadence="RK1 frozen four fields component replay"))
    if a.mode=="lower":
        from jax._src.pallas.triton import lowering
        jp=jax.make_jaxpr(lambda xs:call(xs,dx,"native"))(args).jaxpr
        ir=[str(lowering.lower_jaxpr_to_triton_module(e.params['jaxpr'],e.params['grid_mapping'],'cuda',120).module) for e in jp.eqns if e.primitive.name=="pallas_call"]
        assert len(ir)==4;result.update(lowered=4,f64_in_ir=any("f64" in s for s in ir))
        a.out.with_suffix(".ttir").write_text("\n".join(ir))
    else:
        fn=jax.jit(lambda xs:call(xs,dx,a.arm,interpret=cpu))
        start=time.monotonic();values=fn(args);jax.block_until_ready(values);result["compile_s"]=time.monotonic()-start
        hlo=fn.lower(args).compiler_ir(dialect="hlo").as_hlo_text();result["f64_hlo_lines"]=sum("f64[" in l for l in hlo.splitlines())
        if a.mode in ("verify","cpuverify"):
            oracle=build(a.out.parent/"pristine")
            maps={n:np.asarray(v) for n,v in zip(MAPS,args[11:])}
            cases={}
            for i,name in enumerate(("u","v","w","m")):
                c1,c2=(args[9],args[10]) if name=="w" else (args[7],args[8])
                ref=run(oracle,np.asarray(args[i]),np.asarray(args[6]),np.asarray(c1),np.asarray(c2),maps,
                        name=name,dt=1./dx,factor=1./dx,monotonic=True,
                        kh=np.asarray(args[5])*(np.float32(3) if name=="m" else np.float32(1)),
                        base=np.asarray(args[4]) if name=="m" else None)
                cases[name]=compare(ref,values[i])
            result.update(cases=cases,passed=all(v["passed"] for v in cases.values()))
            if not cpu:
                original=types.ModuleType("pre_b_diff")
                exec(compile(subprocess.check_output(["git","show","59ccd974c:src/gpuwrf/dynamics/explicit_diffusion.py"],text=True),"pre_b_diff.py","exec"),original.__dict__)
                xs64=tuple(x.astype(jnp.float64) for x in args)
                new=jax.jit(lambda xs:call(xs,dx,"legacy"))(xs64)
                current=diff;globals()["diff"]=original
                old=jax.jit(lambda xs:call(xs,dx,"legacy"))(xs64)
                globals()["diff"]=current
                result["default_off_bit_exact"]=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(new,old))
                jax.clear_caches();diff._NATIVE_DIFFUSION_FP32=True
                adapter=jax.jit(lambda xs:call(xs,dx,"legacy"))(xs64)
                diff._NATIVE_DIFFUSION_FP32=False;jax.clear_caches()
                # The scalar interface receives an ALREADY assembled mass.
                # Compare the same input bytes: legacy ABI computes this in
                # f64 then the adapter rounds it, while rebuilding from f32
                # coefficients here would compare a different input operand.
                same_mass=(xs64[7][:,None,None]*xs64[6][None]+xs64[8][:,None,None]).astype(jnp.float32)
                pure=jax.jit(lambda xs, mass:call(xs,dx,"native",scalar_mass=mass))(args,same_mass)
                result["adapter_bit_exact"]=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(adapter,pure))
                result["adapter_field_bit_exact"]={n:np.array_equal(np.asarray(x),np.asarray(y)) for n,x,y in zip(("u","v","w","theta"),adapter,pure)}
                result["passed"] &= result["default_off_bit_exact"] and result["adapter_bit_exact"]
        else:
            for _ in range(5):jax.block_until_ready(fn(args))
            if a.mode=="profile":
                cudart=ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so");assert cudart.cudaProfilerStart()==0
            samples=[]
            for _ in range(10):
                start=time.perf_counter()
                for _ in range(20):values=fn(args)
                jax.block_until_ready(values);samples.append((time.perf_counter()-start)/20)
            if a.mode=="profile":assert cudart.cudaProfilerStop()==0
            result.update(dispatches=200,warm_us=1e6*statistics.median(samples),samples_s=samples)
    a.out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)
    if result.get("passed") is False:raise SystemExit(1)


if __name__=="__main__":main()
