"""B55: REAL mixed-phase columns (CPU-WRF history states) for the pristine-vs-port Thompson one-step oracle.
Kinds (any level): rime = T<0 C & qc>1e-5 & qi+qs+qg>1e-6 (riming, cloud freezing); bigg = T<0 C & qr>1e-6 (rain
freezing); rci = bigg & qi>1e-7 (rain-ice collection). Caller-faithful inputs as LL01 build_columns.py.
usage: build_mixed.py <out.npz> <out.json>"""
import glob, json, sys
import numpy as np
from netCDF4 import Dataset
SW = "<USER_HOME>/wrf_gpu2_lanes/release-docs/SWISS_NEST_REF"
CASES = [("wn3_0227", "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run", d, t)
         for d in ("d02", "d03") for t in ("2026-03-02_18:00:00", "2026-03-03_00:00:00")]
CASES += [("swiss_20230115", f"{SW}/SN04/2023-01-15_00z/cpu", d, t) for d in ("d01", "d02")
          for t in ("2023-01-15_06:00:00", "2023-01-15_12:00:00", "2023-01-15_18:00:00")]
CASES += [("swiss_20241124", f"{SW}/SN04/2024-11-24_18z/cpu", d, t) for d in ("d01", "d02") for t in ("2024-11-25_06:00:00",)]
NPICK = {"rime": 24, "bigg": 16, "rci": 16}
rng = np.random.default_rng(20261005)
R_D, CP = np.float32(287.0), np.float32(7.0 * 287.0 / 2.0)
KEYS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
cols, meta = {k: [] for k in KEYS}, []
for case, run, dom, t in CASES:
    fs = glob.glob(f"{run}/wrfout_{dom}_{t}")
    if not fs:
        print("missing", case, dom, t); continue
    d = Dataset(fs[0]); dt = float(d.DT)
    g = lambda v: np.asarray(d[v][0], np.float32)  # noqa: E731
    qv, qc, qr, qi, qs, qg = (g(v) for v in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"))
    ni, nr, T, P, PB, W, PH, PHB = (g(v) for v in ("QNICE", "QNRAIN", "T", "P", "PB", "W", "PH", "PHB"))
    th = T + np.float32(300.0); p = P + PB; pii = (p / np.float32(1.0e5)) ** (R_D / CP)
    z = (PH + PHB) / np.float32(9.81); dz = z[1:] - z[:-1]
    temp = th * pii; cold = temp < 273.15
    kinds = {"rime": (cold & (qc > 1e-5) & (qi + qs + qg > 1e-6)).any(0), "bigg": (cold & (qr > 1e-6)).any(0),
             "rci": (cold & (qr > 1e-6) & (qi > 1e-7)).any(0)}
    for kind, m in kinds.items():
        yx = np.argwhere(m)
        if len(yx) == 0:
            continue
        for y, x in yx[rng.choice(len(yx), min(NPICK[kind], len(yx)), replace=False)]:
            for k, a in zip(KEYS, (qv, qc, qr, qi, qs, qg, ni, nr, th, pii, p, W[:-1], dz)):
                cols[k].append(a[:, y, x])
            meta.append(dict(case=case, dom=dom, time=t, kind=kind, y=int(y), x=int(x), dt=dt,
                             qc_cold=float(qc[:, y, x][cold[:, y, x]].max(initial=0)), qr_cold=float(qr[:, y, x][cold[:, y, x]].max(initial=0)),
                             ice=float((qi + qs + qg)[:, y, x].max())))
nlev = {len(v) for v in cols["qv"]} if False else {a.shape[0] for a in cols["qv"]}
assert len(nlev) == 1, nlev
np.savez_compressed(sys.argv[1], **{f"in_{k}": np.stack(v).astype(np.float32) for k, v in cols.items()},
                    dt=np.array([m["dt"] for m in meta], np.float32))
json.dump(dict(columns=meta), open(sys.argv[2], "w"), indent=0)
from collections import Counter
print(len(meta), Counter((m["case"], m["dom"], m["kind"], m["dt"]) for m in meta))
