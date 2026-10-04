"""Real-field momentum advection caller audit; retains every miss."""
import json
import ctypes
import hashlib
import os
import statistics
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from bench_scalar_flux import inputs,BASE
from bench_flux import error
from momentum_oracle import build,evaluate
from gpuwrf.dynamics.flux_advection import CoupledVelocities,advect_u_flux,advect_v_flux,advect_w_flux

out=BASE/"phase2b/BC31";records=[]
import argparse
p=argparse.ArgumentParser();p.add_argument("--corrected",action="store_true")
p.add_argument("--gpu",action="store_true");p.add_argument("--full",action="store_true")
p.add_argument("--both-lids",action="store_true")
p.add_argument("--operator-source",type=Path)
p.add_argument("--out",type=Path);p.add_argument("--compare-default",action="store_true");args=p.parse_args()
if Path("/tmp/wrf_gpu2_quiet").exists():raise SystemExit(125)
if args.out:
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stat=Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")",1)[1].split()
    args.out.with_suffix(".pid.json").write_text(json.dumps(dict(pid=os.getpid(),starttime=stat[19]))+"\n")
assert jax.devices()[0].platform==("gpu" if args.gpu else "cpu"),jax.devices()
if args.gpu:assert os.environ.get("GPUWRF_GPU_LOCK_HELD")=="1"
if args.operator_source:
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location("bcore_audit_flux",args.operator_source)
    audited=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audited;spec.loader.exec_module(audited)
    CoupledVelocities=audited.CoupledVelocities
    advect_u_flux,advect_v_flux,advect_w_flux=audited.advect_u_flux,audited.advect_v_flux,audited.advect_w_flux
if args.compare_default:
    assert os.environ.get("GPUWRF_DYN_ADVECTION_FP32", "0") == "0"
    import importlib.util,sys
    source=BASE/"phase2b/BC32/source/src/gpuwrf/dynamics/flux_advection.py"
    spec=importlib.util.spec_from_file_location("bcore_legacy_flux",source)
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
for mode in ("off","fast"):
    if not (out/"oracles"/mode/"momentum.so").exists():build(out/"oracles"/mode,mode)
for domain in ("d01","d02"):
    if Path("/tmp/wrf_gpu2_quiet").exists():raise SystemExit(125)
    arrays,rdx=inputs(domain,not args.full)
    nz,ny,nx=arrays["q"].shape
    with np.load(BASE/"phase1/fixtures"/(domain+".npz")) as d:
        arrays.update({key:d["state_"+var].astype(np.float32)[...,:ny,:nx] for key,var in (("mux","msfux"),("muy","msfuy"),("mvx","msfvx"),("mvy","msfvy"))})
        arrays["mux"]=d["state_msfux"].astype(np.float32)[:ny,:nx+1];arrays["muy"]=d["state_msfuy"].astype(np.float32)[:ny,:nx+1]
        arrays["mvx"]=d["state_msfvx"].astype(np.float32)[:ny+1,:nx];arrays["mvy"]=d["state_msfvy"].astype(np.float32)[:ny+1,:nx]
        for direction,metric,key in (("u","msfuy","ru_full"),("v","msfvx","rv_full")):
            mass=d["state_mu"+direction].astype(np.float32);wind=d["state_"+direction+"_1"].astype(np.float32)
            coupled=(arrays["c1"][:,None,None]*mass+arrays["c2"][:,None,None])*wind/d["state_"+metric].astype(np.float32)
            arrays[key]=coupled[:,:ny+(direction=="v"),:nx+(direction=="u")]
        fields=[d["state_w_save" if field=="w" else "state_"+field+"_1"].astype(np.float32)[:,:ny+(field=="v"),:nx+(field=="u")] for field in ("u","v","w")]
        rdn=d["state_rdn"].astype(np.float32)
        c1f=d["state_c1f"].astype(np.float32);c2f=d["state_c2f"].astype(np.float32)
    for kind,field in enumerate(fields):
        a=dict(arrays);a["rdz"]=rdn if kind==2 else arrays["rdzw"]
        if kind==2:a.update(c1=c1f,c2=c2f)
        reference=evaluate(out/"oracles/off/momentum.so",kind,a,field,rdx,domain=="d02")
        fast=evaluate(out/"oracles/fast/momentum.so",kind,a,field,rdx,domain=="d02")
        for dtype in (jnp.float64,jnp.float32):
            values={key:jnp.asarray(value,dtype) for key,value in a.items()}
            vel=CoupledVelocities(ru=values["ru"],rv=values["rv"],rom=values["rom"],msftx=values["map"],msfux=values["mux"],msfvy=values["mvy"],msfvx=values["mvx"],specified=True,ru_full=values["ru_full"],rv_full=values["rv_full"])
            for lid in ((False,True) if kind==2 and args.both_lids else
                        ((False,) if args.corrected or kind!=2 else (False,True))):
                if kind==2:call=lambda q:advect_w_flux(q,vel,rdx=rdx,rdy=rdx,rdn=values["rdz"],fzm=values["fzm"],fzp=values["fzp"],top_lid=lid,wrf_top_extrapolation=args.corrected)
                else:
                    fn=(advect_u_flux,advect_v_flux)[kind]
                    call=lambda q:fn(q,vel,rdx=rdx,rdy=rdx,rdzw=values["rdz"],fzm=values["fzm"],fzp=values["fzp"])
                if args.gpu:
                    def dynamic(q,a):
                        dv=CoupledVelocities(ru=a["ru"],rv=a["rv"],rom=a["rom"],msftx=a["map"],msfux=a["mux"],msfvy=a["mvy"],msfvx=a["mvx"],specified=True,ru_full=a["ru_full"],rv_full=a["rv_full"])
                        if kind==2:
                            return advect_w_flux(q,dv,rdx=rdx,rdy=rdx,rdn=a["rdz"],fzm=a["fzm"],fzp=a["fzp"],top_lid=lid,wrf_top_extrapolation=args.corrected)
                        op=(advect_u_flux,advect_v_flux)[kind]
                        return op(q,dv,rdx=rdx,rdy=rdx,rdzw=a["rdz"],fzm=a["fzm"],fzp=a["fzp"])
                    jitted=jax.jit(dynamic)
                    input_field=jnp.asarray(field,dtype);call_args=(input_field,values)
                else:
                    jitted=jax.jit(call);input_field=jnp.asarray(field,dtype);call_args=(input_field,)
                got=np.asarray(jitted(*call_args))
                verdict=error(reference,got,fast,reference)
                item=dict(domain=domain,field=("u","v","w")[kind],dtype=str(dtype),top_lid=lid,wrf_top_extrapolation=args.corrected,backend=jax.devices()[0].platform,shape=list(field.shape),mode=np.dtype(dtype).name,device=jax.devices()[0].device_kind,errors=verdict)
                if args.operator_source:
                    item.update(operator_source=str(args.operator_source),operator_sha256=hashlib.sha256(args.operator_source.read_bytes()).hexdigest())
                diff=np.abs(got-reference);loc=np.unravel_index(np.argmax(diff),diff.shape)
                item.update(worst_index=[int(v) for v in loc],worst_reference=float(reference[loc]),worst_got=float(got[loc]))
                records.append(item)
                if not verdict["passed"]:
                    np.savez((args.out.parent if args.out else out)/(domain+f"_{kind}_{np.dtype(dtype).name}_lid{int(lid)}_corrected{int(args.corrected)}_RED.npz"),reference=reference,candidate=got,fast=fast)
                if args.compare_default:
                    if kind==2:
                        oldcall=lambda q:legacy.advect_w_flux(q,vel,rdx=rdx,rdy=rdx,rdn=values["rdz"],fzm=values["fzm"],fzp=values["fzp"],top_lid=lid)
                    else:
                        oldfn=(legacy.advect_u_flux,legacy.advect_v_flux)[kind]
                        oldcall=lambda q:oldfn(q,vel,rdx=rdx,rdy=rdx,rdzw=values["rdz"],fzm=values["fzm"],fzp=values["fzp"])
                    previous=np.asarray(jax.jit(oldcall)(input_field))
                    item["default_off_bitwise"]=bool(previous.tobytes()==got.tobytes())
                    assert item["default_off_bitwise"],item
                if args.gpu:
                    assert verdict["passed"],item
                    compiled=jitted.lower(*call_args).compile();hlo=compiled.as_text()
                    artifact=args.out.parent/(domain+"_"+item["field"]+"_"+item["mode"]+"_lid"+str(int(lid))+".hlo.txt")
                    artifact.write_text(hlo)
                    item.update(f64_tokens=hlo.count("f64"),hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())
                    for _ in range(3):jax.block_until_ready(compiled(*call_args))
                    runtime=ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so");assert runtime.cudaProfilerStart()==0
                    samples=[]
                    for _ in range(11):
                        start=time.perf_counter()
                        for _ in range(32):output=compiled(*call_args)
                        jax.block_until_ready(output);samples.append((time.perf_counter()-start)/32)
                    assert runtime.cudaProfilerStop()==0
                    item.update(samples_s=samples,median_s=statistics.median(samples))
                if args.out:args.out.write_text(json.dumps(records,indent=2)+"\n")
                print(json.dumps({k:v for k,v in item.items() if k!="samples_s"}),flush=True)
(args.out or (out/("cpu_corrected.json" if args.corrected else "cpu.json"))).write_text(json.dumps(records,indent=2)+"\n")
print("PASS",sum(r["errors"]["passed"] for r in records),"of",len(records),flush=True)
