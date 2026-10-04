"""CPU compiler preflight / asserted-GPU WRF oracle and diffusion benchmark."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/v025/b_core")]
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PALLAS_USE_MOSAIC_GPU", "false")
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.kernels.dyn_diff_fp32 import sixth_order_fp32
from gpuwrf.dynamics import explicit_diffusion as diff
from bench_acoustic import compare
from pristine_diffusion import build, run, MAPS

FIXTURES = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/fixtures")


def inputs(domain, dtype):
    path = FIXTURES / (domain + ".npz")
    with np.load(path) as z:
        names = ("u_1", "v_1", "w_save", "theta_1")
        fields = tuple(jnp.asarray(z["state_"+n], dtype) for n in names)
        rest = tuple(jnp.asarray(z["state_"+n], dtype) for n in
                     ("mut", "c1h", "c2h", "c1f", "c2f", *MAPS))
    return fields+rest, hashlib.sha256(path.read_bytes()).hexdigest()


def call(args, dt, arm, monotonic=True, interpret=False):
    u, v, w, t, mut, c1h, c2h, c1f, c2f, tx, ty, ux, uy, vx, vy = args
    if arm == "native":
        def kernel(q, c1, c2, mx, my, name):
            return sixth_order_fp32(q, mut, c1, c2, mx, my, name=name, dt=dt,
                factor=.12, monotonic=monotonic, interpret=interpret)
        return (kernel(u,c1h,c2h,ux,uy,"u"), kernel(v,c1h,c2h,vx,vy,"v"),
                kernel(w,c1f,c2f,tx,ty,"w"), kernel(t,c1h,c2h,tx,ty,"m"))
    return (*diff.wrf_sixth_order_uvw_tendf(u,v,w,mut,c1h=c1h,c2h=c2h,
        c1f=c1f,c2f=c2f,msfux=ux,msfuy=uy,msfvx=vx,msfvy=vy,msftx=tx,
        msfty=ty,dt=dt,diff_6th_factor=.12,monotonic=monotonic),
        diff.wrf_sixth_order_scalar_tendf(t,mut,c1=c1h,c2=c2h,msftx=tx,
        msfty=ty,dt=dt,diff_6th_factor=.12,monotonic=monotonic,specified_or_nested=True))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("lower","cpuverify","verify","bench","profile"),required=True)
    p.add_argument("--domain",choices=("d01","d02"),default="d01")
    p.add_argument("--arm",choices=("native","legacy","jax32"),default="native")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True)
    platform=jax.devices()[0].platform
    cpu=a.mode in ("lower","cpuverify")
    assert platform == ("cpu" if cpu else "gpu"),jax.devices()
    dtype=jnp.float64 if a.arm=="legacy" else jnp.float32
    args, sha=inputs(a.domain,dtype)
    if a.mode=="cpuverify":
        args=tuple(v[:4+int(i==2),:16+int(i==1),:16+int(i==0)] if i<4 else
                   v[:16,:16] if i==4 else v[:4+int(i in (7,8))] if i<9 else
                   v[:16+int(i in (13,14)),:16+int(i in (11,12))]
                   for i,v in enumerate(args))
    dt=54. if a.domain=="d01" else 18.
    result=dict(platform=platform,device=str(jax.devices()[0]),domain=a.domain,
                arm=a.arm,fixture_sha256=sha,work=dict(dt=dt,factor=.12,opt=2,slopeopt=0,
                cadence="RK1 frozen; four fields incl scalar; component replay"),
                cache={key:os.environ.get(key) for key in ("JAX_COMPILATION_CACHE_DIR","GPUWRF_XLA_AUTOTUNE_CACHE_DIR")})
    if a.mode=="lower":
        from jax._src.pallas.triton import lowering
        jp=jax.make_jaxpr(lambda xs:call(xs,dt,"native"))(args).jaxpr
        ir=[]
        for e in jp.eqns:
            if e.primitive.name=="pallas_call":
                ir.append(str(lowering.lower_jaxpr_to_triton_module(e.params['jaxpr'],e.params['grid_mapping'],'cuda',120).module))
        assert len(ir)==4
        result.update(lowered=4, f64_in_ir=any("f64" in s for s in ir))
        a.out.with_suffix(".ttir").write_text("\n".join(ir))
    else:
        fn=jax.jit(lambda xs:call(xs,dt,a.arm,interpret=cpu))
        start=time.monotonic(); values=fn(args);jax.block_until_ready(values)
        result["compile_s"]=time.monotonic()-start
        hlo=fn.lower(args).compiler_ir(dialect="hlo").as_hlo_text()
        result["f64_hlo_lines"]=sum("f64[" in l for l in hlo.splitlines())
        if a.mode in ("verify","cpuverify"):
            oracle=build(a.out.parent/"pristine")
            maps={n:np.asarray(v) for n,v in zip(MAPS,args[9:])}
            cases={}
            for monotonic in (True,False):
                values=jax.jit(lambda xs:call(xs,dt,"native",monotonic,interpret=cpu))(args)
                for i,name in enumerate(("u","v","w","m")):
                    c1,c2=(args[7],args[8]) if name=="w" else (args[5],args[6])
                    ref=run(oracle,np.asarray(args[i]),np.asarray(args[4]),np.asarray(c1),np.asarray(c2),maps,
                            name=name,dt=dt,factor=.12,monotonic=monotonic)
                    cases[name+"_"+str(monotonic)]=compare(ref,values[i])
            result.update(cases=cases,passed=all(c["passed"] for c in cases.values()))
            if not cpu:
                # Frozen pre-lane source: same GPU arrays, exact legacy return.
                old=types.ModuleType("pre_b_diff")
                source=subprocess.check_output(["git","show","59ccd974c:src/gpuwrf/dynamics/explicit_diffusion.py"],text=True)
                exec(compile(source,"pre_b_diff.py","exec"),old.__dict__)
                current=diff
                xs64=tuple(x.astype(jnp.float64) for x in args)
                new_values=jax.jit(lambda xs:call(xs,dt,"legacy"))(xs64)
                globals()["diff"]=old
                old_values=jax.jit(lambda xs:call(xs,dt,"legacy"))(xs64)
                globals()["diff"]=current
                result["default_off_bit_exact"]=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(new_values,old_values))
                jax.clear_caches()
                diff._NATIVE_DIFFUSION_FP32=True
                adapter_values=jax.jit(lambda xs:call(xs,dt,"legacy"))(xs64)
                diff._NATIVE_DIFFUSION_FP32=False
                jax.clear_caches()
                pure_values=jax.jit(lambda xs:call(xs,dt,"native"))(args)
                result["adapter_bit_exact"]=all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(adapter_values,pure_values))
                result["passed"] &= result["default_off_bit_exact"] and result["adapter_bit_exact"]
        else:
            for _ in range(5):jax.block_until_ready(fn(args))
            if a.mode=="profile":
                cudart=ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
                assert cudart.cudaProfilerStart()==0
            samples=[]
            for _ in range(10):
                start=time.perf_counter()
                for _ in range(20):values=fn(args)
                jax.block_until_ready(values)
                samples.append((time.perf_counter()-start)/20)
            if a.mode=="profile":assert cudart.cudaProfilerStop()==0
            result.update(dispatches=200,warm_us=1e6*statistics.median(samples),samples_s=samples)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    if result.get("passed") is False:raise SystemExit(1)


if __name__=="__main__":main()
