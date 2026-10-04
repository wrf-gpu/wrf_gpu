"""GPUWRF_DYN_REAL_ALL operator gates against pristine WRF binary32 routines.

eos: the routed acoustic_wrf.diagnose_pressure_al_alt (finished-stage grid%p
refresh) under the flag vs pristine calc_p_rho_phi (hypsometric 1/2), and
bitwise vs the validated native REAL kernel it dispatches to.
p_rho0: the step-0 calc_p_rho seed (_calc_al_p) in REAL vs pristine calc_p_rho
on b-core's real acoustic fixture (oracle pm1 = the undamped step-0 pressure);
the f64 island and a dropped-geopotential mutant are reported beside it.
ops: stage omega (calc_ww_cp), calc_cq, pg_buoy_w (moist) and rhs_ph (order 5, specified, top lid)
in REAL vs pristine pristine_dyn_real.py on identical operands (pg_buoy_w gets the oracle cqw, rhs_ph
the oracle ww); one mutant per operator must fail.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"tests/v025/b_diff"),str(ROOT/"tests/v025/b_core")]
os.environ["GPUWRF_DYN_FP32"]=os.environ["GPUWRF_DYN_RK_FP32"]=os.environ["GPUWRF_DYN_CARRY_FP32"]="1"
import probe_rk
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.dynamics import acoustic_wrf
from gpuwrf.dynamics.core import calc_p_rho
from bench_acoustic import compare,fixture
import pristine_oracle
import pristine_dyn_real

BCORE=Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/fixtures")


def flag(value):
    os.environ["GPUWRF_DYN_REAL_ALL"]=value


def eos(xs,out,stage):
    flag("1")
    routed=jax.jit(lambda x:acoustic_wrf.diagnose_pressure_al_alt(x[0],x[2],x[3],hypsometric_opt=stage))(xs)
    native=jax.jit(lambda x:probe_rk.diagnose_pressure_fp32(x[0],x[2],x[3],hypsometric_opt=stage))(xs)
    exact=all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(routed,native))
    keep=probe_rk.prepmod
    probe_rk.prepmod=type("routed",(),{"diagnose_pressure_al_alt":staticmethod(acoustic_wrf.diagnose_pressure_al_alt)})
    try:errors=probe_rk.verify(xs,probe_rk.build(out/"pristine_rk"),"eos",stage)
    finally:probe_rk.prepmod=keep
    flag("0")
    return dict(bitwise_native=exact,dtypes=[str(a.dtype) for a in routed],errors=errors,
                passed=exact and all(a.dtype==jnp.float32 for a in routed) and all(v["passed"] for v in errors.values()))


def p_rho0(domain,out,crop):
    state,cfg,_,meta=fixture(BCORE,domain)
    if crop:
        state=state.replace(**{n:v[...,:9 if v.shape[-2]==state.theta.shape[1]+1 else 8,
                                   :17 if v.shape[-1]==state.theta.shape[2]+1 else 16]
                               for n,v in state.to_dict().items() if v is not None and v.ndim>=2})
    lib=out/"pristine_acoustic/oracle.so"
    if not lib.exists():pristine_oracle.build(lib.parent)
    oracle=pristine_oracle.Oracle(lib,state,cfg);oracle.run(pristine_oracle.FUNCTIONS.index("calc_p_rho")+1)
    refs=dict(al=oracle.field("al"),p=oracle.field("pm1"))
    def seed(s,geo=1.):
        return calc_p_rho._calc_al_p(mu_work=s.mu_work,muts_total=s.muts,ph_work=s.ph*geo,theta_work=s.theta_coupled_work,
            theta_1=s.theta_1,c2a=s.c2a,alt=s.alt,c1h=s.c1h,c2h=s.c2h,rdnw=s.rdnw,t0=300.)
    arms={}
    for name,value,geo in (("real","1",1.),("island_f64","0",1.),("mutant_no_geopotential","1",0.)):
        flag(value)
        al,p=jax.jit(lambda s,geo=geo:seed(s,geo))(state)
        arms[name]=dict(dtype=str(p.dtype),errors={"al":compare(refs["al"],al),"p":compare(refs["p"],p)})
        arms[name]["passed"]=all(v["passed"] for v in arms[name]["errors"].values())
    flag("0")
    return dict(fixture_sha256=meta["payload_sha256"],shape=list(state.theta.shape),arms=arms,
                passed=arms["real"]["passed"] and arms["real"]["dtype"]=="float32" and not arms["mutant_no_geopotential"]["passed"])


def ops(xs,out):
    from gpuwrf.dynamics.flux_advection import stage_omega_specified
    from gpuwrf.dynamics.core.advance_w import moist_cqw_calc_face,pg_buoy_w_moist
    from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf
    s,_,b,m=[jax.tree.map(lambda a:jnp.asarray(a,jnp.float32) if hasattr(a,"dtype") and jnp.issubdtype(a.dtype,jnp.floating) else a,x) for x in xs]
    nz,ny,nx=s.theta.shape;rdx=1./probe_rk.CASE_DX
    f32=lambda a:np.asarray(a,np.float32)
    mub,mup=f32(b.mub),f32(s.mu_perturbation)
    mut=f32(mub+mup)
    pad_x=np.pad(mut,((0,0),(1,1)),mode="edge");pad_y=np.pad(mut,((1,1),(0,0)),mode="edge")
    muu=f32(.5*(pad_x[:,1:]+pad_x[:,:-1]));muv=f32(.5*(pad_y[1:]+pad_y[:-1]))
    dn=f32(m.dn)[-1];dnw=f32(m.dnw)[-1];cfn=np.float32((np.float32(.5)*dnw+dn)/dn);cfn1=np.float32(-np.float32(.5)*dnw/dn)
    o=pristine_dyn_real.Oracle(pristine_dyn_real.build(out/"pristine_dyn"),(nz,ny,nx))
    for n,v in (("u",s.u),("v",s.v),("w",s.w),("ph",s.ph_perturbation),("phb",b.phb),("p",s.p_perturbation),
                ("mu",mup),("mub",mub),("mut",mut),("muu",muu),("muv",muv),("msfvx_inv",1./f32(m.msfvx)),
                ("qv",s.qv),("qc",s.qc),("qr",s.qr),("qi",s.qi),("qs",s.qs),("qg",s.qg),("rdx",rdx),("rdy",rdx),
                ("cfn",cfn),("cfn1",cfn1),("g",9.81)):o.set(n,v)
    for n in ("msftx","msfty","msfux","msfuy","msfvx","msfvy","c1h","c2h","c1f","c2f","fnm","fnp","rdnw","rdn","dnw"):o.set(n,getattr(m,n))
    face=(nz+1,ny,nx)
    o.run("calc_ww_cp");ww=o.field("ww",face)
    o.run("calc_cq");cqw=o.field("cqw",face)
    o.set("rw_tend",np.zeros(face,np.float32));o.run("pg_buoy_w");rw,cq1=o.field("rw_tend",face),o.field("cqw",face)
    o.set("ph_tend",np.zeros(face,np.float32));o.run("rhs_ph");ph_tend=o.field("ph_tend",face)
    def ours(mutant):
        omega=stage_omega_specified(s.u,s.v,s.mu_total,c1h=m.c1h,c2h=m.c2h,dnw=m.dnw,rdx=rdx,rdy=rdx,
            msfuy=m.msfuy,msfvx=m.msfvx if not mutant else jnp.ones_like(m.msfvx),msftx=m.msftx)
        qtot=s.qv+s.qc+s.qr+s.qi+s.qs+s.qg
        cq=moist_cqw_calc_face(qtot if not mutant else jnp.roll(qtot,1,axis=0))
        rwt,c1=pg_buoy_w_moist(s.p_perturbation,s.mu_perturbation,b.mub,jnp.asarray(cqw) if not mutant else jnp.zeros_like(cq),
            c1f=m.c1f,c2f=m.c2f,rdnw=m.rdnw,rdn=m.rdn,msfty=m.msfty,gravity=9.81)
        pht=rhs_ph_wrf(u=s.u,v=s.v,ww=jnp.asarray(ww),ph=s.ph_perturbation,phb=b.phb,w=s.w,mut=jnp.asarray(mut),
            muu=jnp.asarray(muu),muv=jnp.asarray(muv),c1f=m.c1f,c2f=m.c2f,fnm=m.fnm,fnp=m.fnp,rdnw=m.rdnw,rdx=rdx,rdy=rdx,
            msfty=m.msfty,non_hydrostatic=True,gravity=9.81,advective_order=5 if not mutant else 2,specified=True,
            msfux=m.msfux,msfvy=m.msfvy,cfn=float(cfn),cfn1=float(cfn1),top_lid=True)
        return dict(ww=omega,cqw=cq,rw_tend=rwt,cq1=c1,ph_tend=pht)
    refs=dict(ww=ww,cqw=cqw,rw_tend=rw,cq1=cq1,ph_tend=ph_tend)
    window={"ww":np.s_[1:nz],"cqw":np.s_[1:nz],"rw_tend":np.s_[1:nz+1],"cq1":np.s_[1:nz],"ph_tend":np.s_[1:nz]}  # top face: port zeroes ph_tend(kde) under the rigid lid (advance_w ignores it)
    arms={}
    for name,value,mutant in (("real","1",False),("island_f64","0",False),("mutants","1",True)):
        flag(value);got=jax.jit(lambda mutant=mutant:ours(mutant))()
        arms[name]={k:dict(compare(refs[k][window[k]],np.asarray(v)[window[k]]),dtype=str(v.dtype)) for k,v in got.items()}
    # Association mutant: REAL operands but rhs_ph differencing the rounded total ph+phb.
    import gpuwrf.dynamics.core.rhs_ph as rhs_mod
    keep=rhs_mod.dyn_real_enabled;rhs_mod.dyn_real_enabled=lambda:False
    try:
        flag("1");pht=jax.jit(lambda:ours(False)["ph_tend"])()
    finally:rhs_mod.dyn_real_enabled=keep
    arms["real_total_assoc"]={"ph_tend":dict(compare(refs["ph_tend"][window["ph_tend"]],np.asarray(pht)[window["ph_tend"]]),dtype=str(pht.dtype))}
    flag("0")
    passed=all(e["passed"] and e["dtype"]=="float32" for e in arms["real"].values())
    sensitive={k:not arms["mutants"][k]["passed"] for k in ("ww","cqw","rw_tend","ph_tend")}
    return dict(arms=arms,mutant_fails=sensitive,passed=passed and all(sensitive.values()))


def uv(xs,out,top_lid):
    """GPUWRF_DYN_GLUE_FUSED large_step_uv_fused vs pristine pgf+coriolis+curvature into one ru/rv_tend."""
    rkmod=probe_rk.rkmod
    s,ref,b,m=xs
    o=probe_rk.Oracle(probe_rk.build(out/("pristine_rk_lid" if top_lid else "pristine_rk"),top_lid=top_lid),s.theta.shape)
    probe_rk.oracle_inputs(o,xs)
    ph,p_abs,al,alt,php=jax.jit(lambda x:rkmod._absolute_diagnostics(x[0],x[3],hypsometric_opt=2,base_state=x[2]))(xs)
    pp=probe_rk.prep(xs,2);cqu,cqv=rkmod.moisture_coupling_factors(s)
    c=lambda a,mu,msf:a*(m.c1h[:,None,None]*mu[None]+m.c2h[:,None,None])/msf[None]
    for n,v in (("ph_2",ph),("p",p_abs),("al",al),("alt",alt),("php",php),("pb",b.pb),("mu_2",s.mu_perturbation),
                ("muu",probe_rk.prepmod._u_face_average_2d(s.mu_total)),("muv",probe_rk.prepmod._v_face_average_2d(s.mu_total)),
                ("cqu",cqu),("cqv",cqv),("u_2",s.u),("v_2",s.v),("w_2",s.w),("ru",c(s.u,pp.muu,m.msfuy)),("rv",c(s.v,pp.muv,m.msfvx)),
                ("rw",s.w*(m.c1f[:,None,None]*s.mu_total[None]+m.c2f[:,None,None])/m.msfty[None])):o.set(n,v)
    for name in ("horizontal_pressure_gradient","coriolis","curvature"):o.run(name,2)
    nz,ny,nx=s.theta.shape
    refs=(o.field("ru_tend",(nz,ny,nx+1))[:,:,1:-1],o.field("rv_tend",(nz,ny+1,nx))[:,1:-1,:])
    zu,zv=jnp.zeros_like(s.u),jnp.zeros_like(s.v)
    def fused(x):
        return rkmod.large_step_uv_fused(x[0],x[3],zu,zv,dx_m=probe_rk.CASE_DX,dy_m=probe_rk.CASE_DX,top_lid=top_lid,hypsometric_opt=2,base_state=x[2])
    def unfused(x):
        st,_,bb,mm=x
        pu,pv=rkmod.large_step_horizontal_pgf(st,mm,dx_m=probe_rk.CASE_DX,dy_m=probe_rk.CASE_DX,top_lid=top_lid,hypsometric_opt=2,base_state=bb)
        cu,cv=rkmod.large_step_coriolis(st,mm,specified=True)
        ku,kv=rkmod.large_step_horizontal_curvature(st,mm,dx_m=probe_rk.CASE_DX,dy_m=probe_rk.CASE_DX,specified=True)
        return ((zu+pu)+cu)+ku,((zv+pv)+cv)+kv
    flag("0");os.environ["GPUWRF_DYN_GLUE_FUSED"]="1"
    arms={"fused":jax.jit(fused)(xs),"unfused":jax.jit(unfused)(xs)}
    keep=rkmod.moisture_coupling_factors;rkmod.moisture_coupling_factors=lambda st:(jnp.ones_like(st.u),jnp.ones_like(st.v))
    try:arms["mutant_cq1"]=jax.jit(lambda x:fused(x))(xs)  # fresh trace (jit caches on the function object)
    finally:rkmod.moisture_coupling_factors=keep
    os.environ["GPUWRF_DYN_GLUE_FUSED"]="0"
    win=lambda a:(np.asarray(a[0])[:,:,1:-1],np.asarray(a[1])[:,1:-1,:])
    res={n:{"ru":compare(refs[0],win(v)[0]),"rv":compare(refs[1],win(v)[1])} for n,v in arms.items()}
    du=np.asarray(arms["fused"][0],np.float64)-np.asarray(arms["unfused"][0]);dv=np.asarray(arms["fused"][1],np.float64)-np.asarray(arms["unfused"][1])
    res["fused_vs_unfused"]=dict(u_max=float(np.abs(du).max()),v_max=float(np.abs(dv).max()),
        u_ref_max=float(np.abs(np.asarray(arms["unfused"][0])).max()),v_ref_max=float(np.abs(np.asarray(arms["unfused"][1])).max()))
    res["passed"]=all(e["passed"] for e in res["fused"].values()) and not all(e["passed"] for e in res["mutant_cq1"].values())
    return res


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("cpuverify","verify"),required=True)
    p.add_argument("--domain",choices=("d01","d02"),default="d01");p.add_argument("--arm",default="native")
    p.add_argument("--out",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    cpu=a.mode=="cpuverify"
    assert jax.devices()[0].platform==("cpu" if cpu else "gpu"),jax.devices()
    probe_rk.native_kernels._EOS_INTERPRET=cpu
    probe_rk.prepmod._PALLAS_INTERPRET=cpu
    if cpu:
        def cpu_rn(op,x,y,*,interpret):
            x,y=x.astype(jnp.float64),y.astype(jnp.float64)
            return {"add":lambda:x+y,"sub":lambda:x-y,"mul":lambda:x*y,"div":lambda:x/y}[op]().astype(jnp.float32)
        probe_rk.native_kernels._rn=cpu_rn
    probe_rk.CASE_DT,probe_rk.CASE_DX=(54.,9000.) if a.domain=="d01" else (18.,3000.)
    xs=probe_rk.load(a.domain,jnp.float32,crop=cpu)
    result=dict(platform=jax.devices()[0].platform,domain=a.domain,mode=a.mode,
        rk_fixture_sha256=hashlib.sha256((probe_rk.DATA/(a.domain+".npz")).read_bytes()).hexdigest(),
        eos={str(stage):eos(xs,a.out,stage) for stage in (1,2)},p_rho0=p_rho0(a.domain,a.out,cpu),ops=ops(xs,a.out),
        uv={str(lid):uv(xs,a.out,lid) for lid in (True,False)})
    result["passed"]=(all(v["passed"] for v in result["eos"].values()) and result["p_rho0"]["passed"] and result["ops"]["passed"]
                      and all(v["passed"] for v in result["uv"].values()))
    (a.out/(a.domain+".json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:(v if k!="eos" else {s:e["passed"] for s,e in v.items()}) for k,v in result.items() if k not in ("p_rho0","ops")}),
          json.dumps({arm:{f:(e["rms"],e["max_abs"],e["passed"]) for f,e in errs.items()} for arm,errs in result["ops"]["arms"].items()}),
          json.dumps({n:{f:(e["rms"],e["max_abs"]) for f,e in arm["errors"].items()} for n,arm in result["p_rho0"]["arms"].items()}),flush=True)
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
