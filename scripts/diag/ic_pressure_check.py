import os, sys
os.environ.setdefault("JAX_ENABLE_X64","true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
os.environ.setdefault("OMP_NUM_THREADS","4")
pass
sys.path.insert(0,"/home/enric/src/wrf_gpu2/src")
import numpy as np
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, resolve_run_dir
from pathlib import Path
# Build the d03 case (IC only, no stepping) and inspect state.p[0] vs corpus
sys.argv=["x"]
sys.path.insert(0,"/home/enric/src/wrf_gpu2/scripts"); from d03_replay import build_l3_d03_daily_case
cfg=DailyPipelineConfig(run_id="20260521_18z_l3_24h_20260522T133443Z",hours=24,
    output_dir=Path("/tmp/icp"),proof_dir=Path("/tmp/icp"),
    run_root=Path("/mnt/data/canairy_meteo/runs/wrf_l3"),score=False,domain="d03",
    dt_s=3.0,acoustic_substeps=10,radiation_cadence_steps=600)
case,run_dir=build_l3_d03_daily_case(cfg)
st=case.state
p0=np.asarray(st.p[0]); pb0=np.asarray(st.p_total[0]-st.p_perturbation[0]) if hasattr(st,'p_perturbation') else None
print("IC state.p[0] mean:", float(np.mean(p0)))
print("IC state.p_total[0] mean:", float(np.mean(np.asarray(st.p_total[0]))))
ppert=np.asarray(st.p_perturbation[0]); pbase=np.asarray((st.p_total-st.p_perturbation)[0])
print("IC p_perturbation[0] mean:", float(np.mean(ppert)), " p_base[0] mean:", float(np.mean(pbase)))
# corpus t=0 (the IC source)
from netCDF4 import Dataset
c=Dataset("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfout_d03_2026-05-21_18:00:00")
def f(ds,n):
    v=ds.variables[n]; d=v[0] if v.dimensions and v.dimensions[0]=='Time' else v[:]
    return np.asarray(np.ma.filled(d,np.nan),dtype=np.float64)
Pc=f(c,'P'); PBc=f(c,'PB')
print("CORPUS t0 P+PB[0] mean:", float(np.nanmean((Pc+PBc)[0])), " Ppert[0]:", float(np.nanmean(Pc[0])), " PB[0]:", float(np.nanmean(PBc[0])))
print("IC pressure bias vs corpus t0:", float(np.mean(np.asarray(st.p_total[0]))) - float(np.nanmean((Pc+PBc)[0])))
