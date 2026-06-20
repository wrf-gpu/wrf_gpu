"""RAINNC chaos falsifier: CPU-WRF internal variability vs the GPU-CPU residual.

Member: identical binary/inputs/ranks, EXCEPT one fp32-ulp-scale perturbation
(T[k=20,j=64,i=64] += 1e-3 K in wrfinput_d01). Pooled rmse over all cells and
all 72 hourly leads (the same pooling the Grid-Delta Atlas RAINNC gate uses:
sum_sq over every paired finite cell across every lead).
"""
import netCDF4 as nc, numpy as np, glob, os, json, sys

TRUTH = '<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu'
PERT = '<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_72h_cpu_pert1/run_cpu'

fields = ["RAINNC", "T2", "U10", "V10", "PSFC", "QVAPOR", "T", "U", "V", "W"]
acc = {f: [0.0, 0] for f in fields}
per_lead_rainnc = []
tf = sorted(glob.glob(TRUTH + '/wrfout_d01_*'))
n_pairs = 0
for f in tf[1:]:  # skip t0 (identical)
    pf = PERT + '/' + os.path.basename(f)
    if not os.path.exists(pf):
        continue
    a = nc.Dataset(f); b = nc.Dataset(pf)
    for name in fields:
        if name not in a.variables or name not in b.variables:
            continue
        x = np.asarray(a.variables[name][0], np.float64)
        y = np.asarray(b.variables[name][0], np.float64)
        d2 = (y - x) ** 2
        acc[name][0] += d2.sum(); acc[name][1] += d2.size
        if name == "RAINNC":
            per_lead_rainnc.append((os.path.basename(f)[-19:], float(np.sqrt(d2.mean())), float(np.abs(y - x).max())))
    a.close(); b.close()
    n_pairs += 1

out = {"paired_leads": n_pairs,
       "perturbation": "T[0,20,64,64] += 1e-3 K in wrfinput_d01 (single point, ~6e-5 relative)",
       "pooled_rmse": {k: (np.sqrt(v[0] / v[1]) if v[1] else None) for k, v in acc.items()}}
print(json.dumps(out, indent=1, default=float))
print("\nRAINNC per-lead rmse/max (every 6h):")
for t, r, m in per_lead_rainnc[::6]:
    print(f"  {t}  rmse {r:8.4f}  max {m:8.3f}")
if per_lead_rainnc:
    print(f"  {per_lead_rainnc[-1][0]}  rmse {per_lead_rainnc[-1][1]:8.4f}  max {per_lead_rainnc[-1][2]:8.3f}")
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
