#!/usr/bin/env python3
"""HV02 analysis: per-step mean-theta drift between precision arms (R release, Q f64 carry, P f64 carry+dycore) and
vs original CPU-WRF (RD11 Swiss, THM = moist theta' at 00/01 h). usage: hv02_analyze.py LANE_DIR OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np
import netCDF4 as nc

D = Path(sys.argv[1])
CPU = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
arms = {a: np.load(D / f"hv02_{a}.npz") for a in ("R", "Q", "P") if (D / f"hv02_{a}.npz").exists()}
out = {"arms": list(arms)}


def wrf(h, var):
    with nc.Dataset(CPU / f"wrfout_d01_2023-01-15_{h:02d}:00:00") as d:
        return np.asarray(d.variables[var][0], np.float64)


thm = {h: wrf(h, "THM") + 300.0 for h in (0, 1, 2)}
qvw = {h: wrf(h, "QVAPOR") for h in (0, 1, 2)}
thm0 = thm[0]
nz = arms["R"]["s0_theta_mean"].shape[0]
w_mean = lambda a: a[:nz].reshape(nz, -1).mean(1)
out["wrf_init_minus_port_s0_K"] = {a: float(np.abs(w_mean(thm0) - z["s0_theta_mean"]).max()) for a, z in arms.items()}
last = max(int(k.split("_")[0][1:]) for k in arms["R"].files if k.endswith("_theta_mean"))
out["nsteps"] = last
for a, z in arms.items():
    th = np.stack([z[f"s{i}_theta_mean"] for i in range(last + 1)])  # (steps+1, nz)
    for h in (1, 2):
        s_h = 200 * h
        if s_h <= last:
            out[f"{a}_theta_vs_wrf_{h}h_mK_k0_15"] = ((th[s_h] - w_mean(thm[h])) * 1e3)[:16].round(3).tolist()
            out[f"{a}_qv_vs_wrf_{h}h_1e7_k0_9"] = ((z[f"s{s_h}_qv_mean"] - w_mean(qvw[h])) * 1e7)[:10].round(2).tolist()
    out[f"{a}_theta_dtype"] = str(z["s0_theta_dtype"])
for a in ("Q", "P"):
    if a not in arms:
        continue
    d = np.stack([arms["R"][f"s{i}_theta_mean"] - arms[a][f"s{i}_theta_mean"] for i in range(last + 1)])
    out[f"R-{a}_mK_at_steps"] = {s: (d[s] * 1e3)[:12].round(3).tolist() for s in (1, 10, 50, 100, 200, 300, last) if s <= last}
    # per-step slope of the k0-4 mean difference (least squares over all steps)
    y = d[:, :5].mean(1)
    out[f"R-{a}_k0_4_slope_K_per_step"] = float(np.polyfit(np.arange(last + 1), y, 1)[0])
    full = f"s{last}_theta"
    if full in arms["R"].files and full in arms[a].files:
        diff = arms["R"][full] - arms[a][full]
        out[f"R-{a}_full_k0_4_rms_mK"] = float(np.sqrt((diff[:5] ** 2).mean()) * 1e3)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
