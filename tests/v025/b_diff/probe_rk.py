"""Real-stage WRF RK/prep/finish fidelity and per-family replay measurements."""
import argparse
import ctypes
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts/v025"),str(ROOT/"tests/v025/b_core")]
mode=sys.argv[sys.argv.index("--mode")+1] if "--mode" in sys.argv else ""
arm=sys.argv[sys.argv.index("--arm")+1] if "--arm" in sys.argv else "native"
os.environ["GPUWRF_DYN_RK_FP32"]="1" if arm=="native" else "0"
os.environ.setdefault("JAX_ENABLE_X64","true")
if mode=="fixture":import cpu_guard
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.contracts.state import State,BaseState,Tendencies
from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.dynamics.core import rk_addtend_dry as rkmod
from gpuwrf.dynamics.core import small_step_prep as prepmod
from gpuwrf.dynamics.core import small_step_finish as finishmod
from gpuwrf.kernels.dyn_rk_fp32 import real_state,real_base,real_metrics,diagnose_pressure_fp32
from gpuwrf.kernels import dyn_rk_fp32 as native_kernels
from pristine_rk import build,Oracle,VECTORS,FIELDS2
from bench_acoustic import compare

CASE_DT,CASE_DX=54.,9000.
DATA=Path("<USER_HOME>/wrf_gpu2_lanes/b-diff/rk_fixtures")
STATE_FIELDS=("u","v","w","theta","p_total","p_perturbation","ph_total",
              "ph_perturbation","mu_total","mu_perturbation","qv","qc","qr","qi","qs","qg")
PREP_OUTPUT=("u_work","v_work","w_work","theta_work","ph_work","mu_work","muts",
             "muus","muvs","u_save","v_save","w_save","t_save","ph_save","mu_save","c2a")
FINISH_OUTPUT=("u","v","w","theta","ph_perturbation","mu_perturbation")


def fixture(domain):
    from real_state import load_real_snapshot,stage_advanced_state
    source=Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/"+("fastbind_r1" if domain=="d01" else "legacybind_r1"))
    histories=sorted(source.glob("wrfout_"+domain+"_*"))
    before=load_real_snapshot(source,domain=domain,wrfout_name=histories[0].name)
    snap=load_real_snapshot(source,domain=domain,wrfout_name=histories[1].name)
    dt=54. if domain=="d01" else 18.
    advanced,pair=stage_advanced_state(before,snap,dt)
    arrays={prefix+n:np.asarray(getattr(s,n)) for prefix,s in (("state_",advanced),("ref_",snap.state)) for n in STATE_FIELDS}
    arrays.update({"base_"+n:np.asarray(getattr(snap.base_state,n)) for n in BaseState.__slots__})
    arrays.update({"metric_"+n:np.asarray(getattr(snap.metrics,n)) for n in DycoreMetrics._array_names()})
    DATA.mkdir(parents=True,exist_ok=True);path=DATA/(domain+".npz");np.savez(path,**arrays)
    manifest=dict(provenance=snap.provenance,pair=pair,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        shape=list(snap.state.theta.shape),dtype="lossless fp64 container of WRF REAL histories; stage interpolation fp64 before native rounding")
    (DATA/(domain+".json")).write_text(json.dumps(manifest,indent=2)+"\n");return manifest


def load(domain,dtype,crop=False):
    with np.load(DATA/(domain+".npz")) as z:
        data={k:z[k] for k in z.files}
    if crop:
        for key,value in data.items():
            if value.ndim==3:
                data[key]=value[:4+int("ph" in key or key.endswith("_w")),:16+int(key.endswith("_v")),:16+int(key.endswith("_u"))]
            elif value.ndim==2:
                data[key]=value[:16+int("msfv" in key),:16+int("msfu" in key)]
            elif value.ndim==1:
                data[key]=value[:4+int(key.endswith(("c1f","c2f","c3f","c4f")))]
    data={k:jnp.asarray(v,dtype) for k,v in data.items()}
    def state(prefix):
        return State.tree_unflatten(None,[data.get(prefix+n) for n in State.__slots__])
    state_cur,state_ref=state("state_"),state("ref_")
    base=BaseState(**{n:data["base_"+n] for n in BaseState.__slots__})
    metrics=DycoreMetrics(**{n:data["metric_"+n] for n in DycoreMetrics._array_names()},precision="fp32" if dtype==jnp.float32 else "fp64")
    return state_cur,state_ref,base,metrics


def prep(inputs,stage):
    s,ref,b,m=inputs
    return prepmod.small_step_prep_wrf(s,stage,CASE_DT,metrics=m,reference_state=s if stage==1 else ref,base_state=b)


def tendencies(inputs):
    s,ref,b,m=inputs
    # Real hourly tendency structure, coupled with real WRF dry column masses.
    dt=jnp.asarray(CASE_DT,s.u.dtype)
    mh=m.c1h[:,None,None]*s.mu_total[None]+m.c2h[:,None,None]
    mf=m.c1f[:,None,None]*s.mu_total[None]+m.c2f[:,None,None]
    muu=prepmod._u_face_average_2d(s.mu_total)
    muv=prepmod._v_face_average_2d(s.mu_total)
    mu=m.c1h[:,None,None]*muu[None]+m.c2h[:,None,None]
    mv=m.c1h[:,None,None]*muv[None]+m.c2h[:,None,None]
    t=Tendencies((s.u-ref.u)/dt*mu,(s.v-ref.v)/dt*mv,(s.w-ref.w)/dt*mf,
                 (s.theta-ref.theta)/dt*mh,(s.qv-ref.qv)/dt,(s.p_perturbation-ref.p_perturbation)/dt,
                 (s.ph_perturbation-ref.ph_perturbation)/dt,(s.mu_perturbation-ref.mu_perturbation)/dt)
    f=rkmod.DryPhysicsTendencies(ru_tendf=t.u,rv_tendf=t.v,rw_tendf=t.w,ph_tendf=t.ph,
        t_tendf=t.theta,mu_tendf=t.mu,h_diabatic=(s.theta-ref.theta)/dt,
        u_save=(s.u-ref.u)/dt,v_save=(s.v-ref.v)/dt,w_save=(s.w-ref.w)/dt,
        ph_save=t.ph,t_save=t.theta)
    return t,f


def work_out(p):
    # Replay the prepared nonzero real work fields. The physical-mu adapter
    # carries MU_save+MU_work; WRF finish receives MU_work and adds MU_save.
    return types.SimpleNamespace(u=p.u_work,v=p.v_work,w=p.w_work,theta_coupled_work=p.theta_work,
        ph=p.ph_work,mu=p.mu_save+p.mu_work,p=jnp.zeros_like(p.al),muts=p.muts,ww=p.ww_save)


def call(inputs,family,stage):
    s,ref,b,m=inputs
    if family=="prep":
        p=prep(inputs,stage);return tuple(getattr(p,n) for n in PREP_OUTPUT)
    if family=="finish":
        p=prep(inputs,stage)
        result=finishmod.small_step_finish_wrf(p,work_out(p))
        return tuple(getattr(result,n) for n in FINISH_OUTPUT)
    if family=="rk":
        t,f=tendencies(inputs)
        result=rkmod.rk_addtend_dry(t,f,rk_step=stage,metrics=m,mut=s.mu_total)
        return tuple(getattr(result,n) for n in ("u","v","w","theta","ph","mu"))
    if family=="pgf":return rkmod.large_step_horizontal_pgf(s,m,dx_m=CASE_DX,dy_m=CASE_DX,hypsometric_opt=2,base_state=b)
    if family=="coriolis":return rkmod.large_step_coriolis(s,m)
    if family=="curvature":return rkmod.large_step_horizontal_curvature(s,m,dx_m=CASE_DX,dy_m=CASE_DX)
    if family=="eos":return prepmod.diagnose_pressure_al_alt(s,b,m,hypsometric_opt=stage if stage<3 else 2)
    raise ValueError(family)


def oracle_inputs(oracle,inputs):
    s,ref,b,m=inputs
    for n in VECTORS:
        if hasattr(m,n):oracle.set(n,getattr(m,n))
    for n in FIELDS2:
        if hasattr(m,n):oracle.set(n,getattr(m,n))
    oracle.set("msfvx_inv",1./np.asarray(m.msfvx))
    for n in ("rdx","rdy"):oracle.set(n,1./CASE_DX)
    for n in ("cf1","cf2","cf3"):oracle.set(n,getattr(m,n))
    oracle.set("cfn",(.5*m.dnw[-1]+m.dn[-1])/m.dn[-1]);oracle.set("cfn1",-.5*m.dnw[-1]/m.dn[-1])
    oracle.set("dts",CASE_DT)
    oracle.set("p0",100000.);oracle.set("t0",300.);oracle.set("ptop",m.p_top)


def verify(inputs,path,family,stage):
    s,ref,b,m=inputs
    oracle=Oracle(path,s.theta.shape);oracle_inputs(oracle,inputs)
    got=jax.jit(lambda xs:call(xs,family,stage))(inputs)
    jax.block_until_ready(got)
    print("operator ready",family,stage,flush=True)
    if family in ("prep","finish"):
        p=prep(inputs,stage)
        for n in ("muu","muv","muus","muvs","mut","muts","mu_save"):
            oracle.set(n,getattr(p,n))
        oracle.set("mub",b.mub)
        chosen=s if stage==1 else ref
        for n in ("u","v","w"):
            oracle.set(n+"_1",getattr(chosen,n));oracle.set(n+"_2",getattr(s,n))
            oracle.set(n+"_save",getattr(p,n+"_save"))
        oracle.set("t_1",chosen.theta-jnp.float32(300));oracle.set("t_2",s.theta-jnp.float32(300))
        oracle.set("ph_1",chosen.ph_perturbation);oracle.set("ph_2",s.ph_perturbation)
        oracle.set("mu_1",chosen.mu_perturbation);oracle.set("mu_2",s.mu_perturbation)
        oracle.set("pb",p.pb);oracle.set("p",prepmod.diagnose_pressure_al_alt(s,b,m)[0]);oracle.set("alt",p.alt)
        oracle.set("ww",jnp.zeros_like(s.w))
        if family=="prep":
            oracle.run("calc_mu_uv",stage)
            oracle.run("small_step_prep",stage)
            fields=("u_2","v_2","w_2","t_2","ph_2","mu_2","muts","muus","muvs",
                    "u_save","v_save","w_save","t_save","ph_save","mu_save","c2a")
        else:
            for n,v in (("u_2",p.u_work),("v_2",p.v_work),("w_2",p.w_work),("t_2",p.theta_work),
                        ("ph_2",p.ph_work),("mu_2",p.mu_work),("t_save",p.t_save),("ph_save",p.ph_save),
                        ("ww1",p.ww_save),("h_diabatic",jnp.zeros_like(p.theta_work))):oracle.set(n,v)
            oracle.run("calc_mu_uv_1",stage)
            oracle.run("small_step_finish",stage)
            fields=("u_2","v_2","w_2","t_2","ph_2","mu_2")
        refs=[oracle.field(n,v.shape) for n,v in zip(fields,got)]
        if family=="finish":refs[3]+=np.float32(300)
    elif family=="rk":
        t,f=tendencies(inputs)
        for n,attr in (("ru_tend","u"),("rv_tend","v"),("rw_tend","w"),("t_tend","theta"),("ph_tend","ph"),("mu_tend","mu")):oracle.set(n,getattr(t,attr))
        for n in f.__dataclass_fields__:
            if getattr(f,n) is not None:oracle.set(n,getattr(f,n))
        oracle.set("mut",s.mu_total);oracle.run("rk_addtend_dry",stage)
        fields=("ru_tend","rv_tend","rw_tend","t_tend","ph_tend","mu_tend")
        refs=[oracle.field(n,v.shape) for n,v in zip(fields,got)]
    elif family=="eos":
        option=stage if stage<3 else 2
        muts=b.mub+s.mu_perturbation
        if option==1:
            alb=-(b.phb[1:]-b.phb[:-1])/(m.dnw[:,None,None]*(m.c1h[:,None,None]*b.mub[None]+m.c2h[:,None,None]))
        else:
            upper=m.c3f[1:,None,None]*b.mub[None]+m.c4f[1:,None,None]+m.p_top
            lower=m.c3f[:-1,None,None]*b.mub[None]+m.c4f[:-1,None,None]+m.p_top
            half=m.c3h[:,None,None]*b.mub[None]+m.c4h[:,None,None]+m.p_top
            alb=(b.phb[1:]-b.phb[:-1])/half/jnp.log(lower/upper)
        for n,v in (("alb",alb),("mu_2",s.mu_perturbation),("muts",muts),
                    ("ph_2",s.ph_perturbation),("phb",b.phb),("pb",b.pb),("t_2",s.theta-jnp.float32(300))):oracle.set(n,v)
        for n in ("qv","qc","qr","qi","qs","qg"):oracle.set(n,getattr(s,n))
        oracle.run("calc_p_rho_phi",option)
        fields=("p","al","alt")
        refs=[oracle.field("p",got[0].shape),oracle.field("al",got[1].shape),
              oracle.field("al",got[2].shape)+np.asarray(alb)]
    elif family=="pgf":
        ph,p_abs,al,alt,php=rkmod._absolute_diagnostics(s,m,hypsometric_opt=2,base_state=b)
        for n,v in (("ph_2",ph),("p",p_abs),("al",al),("alt",alt),("php",php),("pb",b.pb),("mu_2",s.mu_perturbation),
                    ("muu",prepmod._u_face_average_2d(s.mu_total)),("muv",prepmod._v_face_average_2d(s.mu_total))):oracle.set(n,v)
        cqu,cqv=rkmod.moisture_coupling_factors(s);oracle.set("cqu",cqu);oracle.set("cqv",cqv)
        oracle.run("horizontal_pressure_gradient",stage)
        fields=("ru_tend","rv_tend");refs=[oracle.field(n,v.shape) for n,v in zip(fields,got)]
    else:
        p=prep(inputs,stage)
        for n,v in (("u_2",s.u),("v_2",s.v),("w_2",s.w),
                    ("ru",s.u*(m.c1h[:,None,None]*p.muu[None]+m.c2h[:,None,None])/m.msfuy[None]),
                    ("rv",s.v*(m.c1h[:,None,None]*p.muv[None]+m.c2h[:,None,None])/m.msfvx[None]),
                    ("rw",s.w*(m.c1f[:,None,None]*s.mu_total[None]+m.c2f[:,None,None])/m.msfty[None])):oracle.set(n,v)
        oracle.run(family,stage)
        fields=("ru_tend","rv_tend");refs=[oracle.field(n,v.shape) for n,v in zip(fields,got)]
    # PGF/Coriolis/curvature are overwritten at specified boundary faces by
    # solve_em's spec_bdy_dry. Freeze the source-owned comparison window.
    if family in ("pgf","coriolis","curvature"):
        refs=[refs[0][:,:,1:-1],refs[1][:,1:-1,:]]
        got=(got[0][:,:,1:-1],got[1][:,1:-1,:])
    return {n:compare(r,g) for n,r,g in zip(fields,refs,got)}


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("fixture","lower","cpuverify","verify","identity","hlo","bench","profile"),required=True)
    p.add_argument("--domain",choices=("d01","d02"),default="d01")
    p.add_argument("--arm",choices=("native","legacy"),default="native")
    p.add_argument("--family",choices=("prep","finish","rk","pgf","coriolis","curvature","eos"),default="prep")
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    print("setup",a.mode,a.family,a.domain,flush=True)
    cpu=a.mode in ("fixture","lower","cpuverify","hlo")
    prepmod._PALLAS_INTERPRET = a.mode in ("cpuverify", "hlo")
    native_kernels._EOS_INTERPRET = a.mode in ("cpuverify", "hlo")
    if a.mode=="cpuverify":
        # CPU reference emulates the explicit PTX RN boundaries. This hook
        # never runs on GPU and does not substantiate native HLO precision.
        def cpu_rn(op,x,y,*,interpret):
            x,y=x.astype(jnp.float64),y.astype(jnp.float64)
            value={"add":lambda:x+y,"sub":lambda:x-y,"mul":lambda:x*y,"div":lambda:x/y}[op]()
            return value.astype(jnp.float32)
        native_kernels._rn=cpu_rn
    assert jax.devices()[0].platform==("cpu" if cpu else "gpu"),jax.devices()
    if a.mode=="fixture":a.out.write_text(json.dumps(fixture(a.domain),indent=2)+"\n");return
    global CASE_DT,CASE_DX
    CASE_DT,CASE_DX=(54.,9000.) if a.domain=="d01" else (18.,3000.)
    xs=load(a.domain,jnp.float32 if a.arm=="native" else jnp.float64,crop=a.mode=="cpuverify")
    print("inputs ready",a.domain,flush=True)
    result=dict(platform=jax.devices()[0].platform,domain=a.domain,arm=a.arm,family=a.family,
        fixture_sha256=hashlib.sha256((DATA/(a.domain+".npz")).read_bytes()).hexdigest(),
        cache={k:os.environ.get(k) for k in ("JAX_COMPILATION_CACHE_DIR","GPUWRF_XLA_AUTOTUNE_CACHE_DIR")},
        work="real-derived nonzero stage; owned operator outputs; default source metrics/REAL; no full forecast claim")
    if a.mode=="lower":
        from jax._src.pallas.triton import lowering
        jp=jax.make_jaxpr(lambda x:call(x,"prep",2))(xs).jaxpr
        ir=[str(lowering.lower_jaxpr_to_triton_module(e.params['jaxpr'],e.params['grid_mapping'],'cuda',120).module)
            for e in jp.eqns if e.primitive.name=='pallas_call']
        assert len(ir)==6
        result.update(lowered=6,f64_in_ir=any('f64' in s for s in ir))
        a.out.with_suffix('.ttir').write_text('\n'.join(ir))
    elif a.mode in ("verify","cpuverify"):
        path=build(a.out.parent/"pristine")
        cases={str(stage):verify(xs,path,a.family,stage) for stage in (1,2,3)}
        result.update(cases=cases,passed=all(v["passed"] for row in cases.values() for v in row.values()))
    elif a.mode=="identity":
        assert a.arm=="legacy"
        families=("prep","finish","rk","pgf","coriolis","curvature")
        new={(f,stage):jax.jit(lambda x,f=f,stage=stage:call(x,f,stage))(xs) for f in families for stage in (1,2,3)}
        modules={}
        for name in ("rk_addtend_dry","small_step_prep","small_step_finish"):
            mod=types.ModuleType("baseline_"+name);sys.modules[mod.__name__]=mod
            source=subprocess.check_output(["git","show","59ccd974c:src/gpuwrf/dynamics/core/"+name+".py"],text=True)
            exec(compile(source,mod.__name__+".py","exec"),mod.__dict__);modules[name]=mod
        globals().update(rkmod=modules['rk_addtend_dry'],prepmod=modules['small_step_prep'],finishmod=modules['small_step_finish'])
        exact={f+"_"+str(stage):all(np.array_equal(np.asarray(v),np.asarray(w)) for v,w in zip(value,jax.jit(lambda x:call(x,f,stage))(xs))) for (f,stage),value in new.items()}
        result.update(default_off_bit_exact=exact,passed=all(exact.values()))
    else:
        fn=jax.jit(lambda x:call(x,a.family,2));start=time.monotonic();values=fn(xs);jax.block_until_ready(values)
        result['compile_s']=time.monotonic()-start
        hlo=fn.lower(xs).compiler_ir(dialect='hlo').as_hlo_text();result['f64_hlo_lines']=sum('f64[' in l for l in hlo.splitlines())
        if a.mode!='hlo':
            for _ in range(5):jax.block_until_ready(fn(xs))
            if a.mode=='profile':
                cudart=ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so');assert cudart.cudaProfilerStart()==0
            samples=[]
            for _ in range(10):
                start=time.perf_counter()
                for _ in range(20):values=fn(xs)
                jax.block_until_ready(values);samples.append((time.perf_counter()-start)/20)
            if a.mode=='profile':assert cudart.cudaProfilerStop()==0
            result.update(dispatches=200,warm_us=1e6*statistics.median(samples),samples_s=samples)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result.get('passed') is False:raise SystemExit(1)


if __name__=='__main__':main()
