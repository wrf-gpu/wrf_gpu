"""LL01: REAL WN3 0227 CPU-WRF drizzle/stratocumulus columns -> Thompson mp_gt_driver inputs (caller-faithful).

WRF phy_prep / microphysics_driver wiring: th = T + 300 (dry theta; wrfout T, USE_THETA_M=1 history writes dry T — checked
against THM below), p = P + PB, pii = (p/1e5)**(R_d/cp) with R_d 287, cp 7*R_d/2, w1d(k) = W(k) (BOTTOM w-face,
MPT :1224), dz8w = ((PH+PHB)(k+1) - (PH+PHB)(k)) / 9.81. Inputs are the CPU-WRF history state (REAL, end of a step).
usage: build_columns.py <out.npz> <out.json>
"""
import json, sys
import numpy as np
from netCDF4 import Dataset
RUN = "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run"
TIMES = {18: "2026-02-28_18:00:00", 24: "2026-03-01_00:00:00", 30: "2026-03-01_06:00:00", 36: "2026-03-01_12:00:00"}
DT = {"d03": 6.0, "d02": 18.0}
NPICK = {"drizzle": 32, "onset": 8, "rain": 12}
rng = np.random.default_rng(20261005)
R_D, CP = np.float32(287.0), np.float32(7.0 * 287.0 / 2.0)
cols, meta, thm_err = {k: [] for k in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")}, [], []
for dom in ("d03", "d02"):
    for tau, t in TIMES.items():
        d = Dataset(f"{RUN}/wrfout_{dom}_{t}")
        g = lambda v: np.asarray(d[v][0], np.float32)
        qv, qc, qr, qi, qs, qg = (g(v) for v in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"))
        ni, nr, T, P, PB, W, PH, PHB = (g(v) for v in ("QNICE", "QNRAIN", "T", "P", "PB", "W", "PH", "PHB"))
        thm = g("THM")
        th = T + np.float32(300.0)
        thm_err.append(float(np.abs((th * (1 + np.float32(461.6 / 287.0) * qv) - 300) - thm).max()))
        p = P + PB
        pii = (p / np.float32(1.0e5)) ** (R_D / CP)
        z = (PH + PHB) / np.float32(9.81)
        dz = z[1:] - z[:-1]
        hgt, land = g("HGT"), g("LANDMASK") > 0.5
        qcm, qrm = qc.max(0), qr.max(0)
        cloud = (qcm >= 1e-4) & (qcm <= 1e-3)
        for cat, area in (("land>500", land & (hgt > 500)), ("land<=500", land & (hgt <= 500)), ("sea", ~land)):
            for kind, m in (("drizzle", cloud & (qrm > 1e-8) & (qrm <= 1e-5)), ("onset", cloud & (qrm <= 1e-8)),
                            ("rain", (qcm >= 1e-4) & (qrm > 1e-5))):
                yx = np.argwhere(area & m)
                if len(yx) == 0:
                    continue
                pick = yx[rng.choice(len(yx), min(NPICK[kind], len(yx)), replace=False)]
                for y, x in pick:
                    for k, a in (("qv", qv), ("qc", qc), ("qr", qr), ("qi", qi), ("qs", qs), ("qg", qg), ("ni", ni),
                                 ("nr", nr), ("th", th), ("pii", pii), ("p", p), ("w", W[:-1]), ("dz8w", dz)):
                        cols[k].append(a[:, y, x])
                    meta.append(dict(dom=dom, tau=tau, cat=cat, kind=kind, y=int(y), x=int(x), hgt=float(hgt[y, x]),
                                     qc_max=float(qcm[y, x]), qr_max=float(qrm[y, x]),
                                     ice_max=float((qi + qs + qg)[:, y, x].max()), dt=DT[dom]))
print("theta check |(T+300)(1+Rv/Rd qv)-300 - THM| max", max(thm_err))
np.savez_compressed(sys.argv[1], **{f"in_{k}": np.stack(v).astype(np.float32) for k, v in cols.items()},
                    dt=np.array([m["dt"] for m in meta], np.float32))
json.dump(dict(source=RUN, times=TIMES, columns=meta), open(sys.argv[2], "w"), indent=0)
from collections import Counter
print(len(meta), Counter((m["dom"], m["cat"], m["kind"]) for m in meta))
