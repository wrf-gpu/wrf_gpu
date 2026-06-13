"""Single-size probe to find the largest grid that fits under the async CUDA
allocator (the BFC allocator fragments and OOMs on the one ~9.5 GiB op even
though peak is well under the ~30 GiB ceiling).  Reuses the tiler from the main
bench.  Tries a descending set of (fy,fx) until one fits; reports peak VRAM."""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path
import jax, jax.numpy as jnp
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational
sys.path.insert(0, str(Path("proofs/perf/v015/km_bench")))
from grid_scaling_bench import _tile_state, _tile_namelist, _block, _peak_gib, _reset_peak

CANDIDATES = [(3,3),(3,4),(4,4),(4,5),(5,5)]  # 94k,126k,168k,210k,262k cols

def run_one(base_state, base_nl, ny0, nx0, fy, fx, dt_s):
    ny,nx=fy*ny0,fx*nx0; ncol=ny*nx
    nl=_tile_namelist(base_nl,ny0,nx0,fy,fx)
    h1,h2=0.05,0.15; dt=dt_s
    n1=int(round(h1*3600/dt)); n2=int(round(h2*3600/dt))
    def fs():
        st=_tile_state(base_state,ny0,nx0,fy,fx)
        st=jax.tree_util.tree_map(lambda x:(x+0) if hasattr(x,'shape') else x, st)
        _block(st); return st
    _reset_peak()
    t=run_forecast_operational(fs(),nl,h1); _block(t)
    t0=time.perf_counter(); o=run_forecast_operational(fs(),nl,h1); _block(o); w1=time.perf_counter()-t0
    run_forecast_operational(fs(),nl,h2)  # compile h2
    t0=time.perf_counter(); o=run_forecast_operational(fs(),nl,h2); _block(o); w2=time.perf_counter()-t0
    ps=(w2-w1)/(n2-n1)*1000.0
    return {'ny':ny,'nx':nx,'ncol':ncol,'warmed_ms_per_step':ps,
            'ms_per_forecast_hour':ps*3600/dt,'peak_vram_gib':_peak_gib(),'ran_ok':True,'oom':False}

def main():
    cfg=DailyPipelineConfig(hours=1,dt_s=10.0,acoustic_substeps=10)
    case,_=_build_real_case(cfg)
    bnl,bst=case.namelist,case.state; dt=float(bnl.dt_s)
    ny0,nx0=int(case.grid.ny),int(case.grid.nx)
    allocator=__import__("os").environ.get("XLA_PYTHON_CLIENT_ALLOCATOR","(default bfc)")
    print(f"[probe] base {ny0}x{nx0} allocator={allocator}",flush=True)
    recs=[]
    for fy,fx in CANDIDATES:
        ny,nx=fy*ny0,fx*nx0
        print(f"[probe] {ny}x{nx} ncol={ny*nx} ...",flush=True)
        try:
            r=run_one(bst,bnl,ny0,nx0,fy,fx,dt)
            print(f"  OK ms/step={r['warmed_ms_per_step']:.1f} peak={r['peak_vram_gib']:.2f}G",flush=True)
        except Exception as e:
            is_oom="RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower()
            r={'ny':ny,'nx':nx,'ncol':ny*nx,'ran_ok':False,'oom':bool(is_oom),
               'error':f"{type(e).__name__}: {e}"[:300],'peak_vram_gib':_peak_gib()}
            print(f"  FAIL oom={is_oom} :: {str(e)[:140]}",flush=True)
            recs.append(r)
            if is_oom: break
            continue
        recs.append(r)
    out={'scope':'largest-grid probe (allocator='+allocator+')','allocator':allocator,'records':recs}
    Path("proofs/perf/v015/km_bench/largest_probe.json").write_text(json.dumps(out,indent=2)+"\n")
    print("wrote proofs/perf/v015/km_bench/largest_probe.json",flush=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
