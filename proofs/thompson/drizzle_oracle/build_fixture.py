"""LL01 (v0.3.3): freeze a REAL-drizzle subset of the pristine mp_gt_driver one-step oracle for the Thompson full column.

Source: WN3 0227 CPU-WRF history state (<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run, d03 dt 6 s /
d02 dt 18 s, tau 18/24/30/36), columns selected in LL01 (seeded, land>500 m / land<=500 m / sea x drizzle / onset / rain),
pristine one step from the lane copy of the TH08 driver (module_mp_thompson.F, gfortran; hgt3d padded because
thompson_init reads hgt(its+1,k,jts+1) at :473-476). Per (domain, area, kind) group the 2 columns with the largest WRF
cloud->rain conversion (-column dqc) and the column with the largest WRF rain evaporation (column dqv) are kept.
usage: build_fixture.py <columns.npz> <columns.json> <oracle.npz> <fixture.npz>
"""
import json, sys
import numpy as np
z, meta, o = dict(np.load(sys.argv[1])), json.load(open(sys.argv[2]))["columns"], dict(np.load(sys.argv[3]))
rho, dz = o["port_rho"].astype(np.float64), z["in_dz8w"].astype(np.float64)
col = lambda v: ((o[f"wrf_{v}"].astype(np.float64) - z[f"in_{v}"]) * rho * dz).sum(1)  # noqa: E731
conv, evap = -col("qc"), col("qv")
groups = {}
for i, m in enumerate(meta):
    groups.setdefault((m["dom"], m["cat"], m["kind"]), []).append(i)
keep = []
for key in sorted(groups):
    idx = np.array(groups[key])
    pick = list(idx[np.argsort(conv[idx])[-2:]]) + [idx[np.argmax(evap[idx])]]
    keep += [i for i in dict.fromkeys(int(i) for i in pick) if i not in keep]
keep = np.array(sorted(keep))
names = np.array(["%s_t%d_%s_%s_y%d_x%d" % (meta[i]["dom"], meta[i]["tau"], meta[i]["cat"], meta[i]["kind"], meta[i]["y"], meta[i]["x"])
                  for i in keep])
ins = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
outs = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "rainncv", "snowncv", "graupelncv")
np.savez_compressed(sys.argv[4], names=names, dt=z["dt"][keep],
                    **{f"in_{v}": z[f"in_{v}"][keep] for v in ins}, **{f"wrf_{v}": o[f"wrf_{v}"][keep] for v in outs})
print(len(keep), "columns:", dict(zip(*np.unique(z["dt"][keep], return_counts=True))))
