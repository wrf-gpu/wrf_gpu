#!/usr/bin/env python3
"""HV01: deterministic early-lead GPU-vs-CPU-WRF signal on WN3 0227 d03 (CPU only, numpy/netCDF4).

Arms: G = GPU v0.4.0 (F20 archive, 24 h), C = original CPU-WRF 2x2 (server, read-only),
C2 = CPU-WRF 4x1 decomposition twin (w43-twin0227). Yardstick: |C2-C| is pure CPU round-off/decomposition
noise; a coherent G-C beyond it at leads where fields are still correlated points at a deterministic operator
difference. Output JSON to argv[1].
"""
import json, sys
import numpy as np
import netCDF4 as nc

RUNS = {
    "G": "<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/WN3-0227/gpu",
    "C": "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run",
    "C2": "<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu",
}
DOM = sys.argv[2] if len(sys.argv) > 2 else "d03"
LEADS = range(0, 25)


def fname(run, lead):
    import glob
    hh = lead % 24
    day = 28 + lead // 24
    pat = f"{RUNS[run]}/wrfout_{DOM}_2026-02-{day:02d}_{hh:02d}:00:*" if day <= 28 else \
        f"{RUNS[run]}/wrfout_{DOM}_2026-03-{day-28:02d}_{hh:02d}:00:*"
    g = sorted(glob.glob(pat))
    return g[0] if g else None


def load(run, lead):
    f = fname(run, lead)
    with nc.Dataset(f) as d:
        out = {}
        for v in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNRAIN", "T", "W", "QC_BL"):
            if v in d.variables:
                out[v] = np.asarray(d.variables[v][0], np.float64)
        for v in ("RAINNC", "MU", "MUB", "LANDMASK", "HGT", "PSFC", "T2", "Q2", "HFX", "LH", "U10", "V10", "PBLH"):
            if v in d.variables:
                out[v] = np.asarray(d.variables[v][0], np.float64)
        ph = np.asarray(d.variables["PH"][0], np.float64) + np.asarray(d.variables["PHB"][0], np.float64)
        out["DZ"] = (ph[1:] - ph[:-1]) / 9.81
        p = np.asarray(d.variables["P"][0], np.float64) + np.asarray(d.variables["PB"][0], np.float64)
        out["RHO"] = p / (287.0 * (out["T"] + 300.0) * (p / 1e5) ** 0.2857 * (1 + 0.608 * out["QVAPOR"]))
    return out


def colint(x, s):
    return (x * s["RHO"] * s["DZ"]).sum(0)


def stats(a, b, mask):
    d = (a - b)[mask]
    return {"mean": float(d.mean()), "rms": float(np.sqrt((d ** 2).mean())),
            "corr": float(np.corrcoef(a[mask], b[mask])[0, 1]) if a[mask].std() > 0 and b[mask].std() > 0 else None}


res = {"domain": DOM, "leads": {}}
prev = {}
for L in LEADS:
    S = {r: load(r, L) for r in RUNS}
    land = S["C"]["LANDMASK"] > 0.5
    ny, nx = land.shape
    ring = np.minimum.reduce(np.meshgrid(np.arange(nx), np.arange(ny))) if False else None
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ringd = np.minimum(np.minimum(jj, ny - 1 - jj), np.minimum(ii, nx - 1 - ii))
    interior = ringd >= 10
    masks = {"all": np.ones_like(land), "sea_int": (~land) & interior, "land_int": land & interior}
    row = {}
    for r in RUNS:
        s = S[r]
        s["CWV"] = colint(s["QVAPOR"], s)
        s["CLW"] = colint(s["QCLOUD"], s)
        s["CRW"] = colint(s["QRAIN"], s)
        s["CIW"] = colint(s["QICE"] + s["QSNOW"] + s["QGRAUP"], s)
        s["TH1_5"] = s["T"][:5].mean(0)
        s["WMAX"] = s["W"].max(0)
        s["HR"] = s["RAINNC"] - (prev[r] if r in prev else 0.0)
    for name, m in masks.items():
        row[name] = {}
        for v in ("RAINNC", "HR", "CWV", "CLW", "CRW", "CIW", "TH1_5", "WMAX", "PSFC", "HFX", "LH", "T2", "Q2"):
            if v not in S["C"]:
                continue
            row[name][v] = {
                "C_mean": float(S["C"][v][m].mean()),
                "G-C": stats(S["G"][v], S["C"][v], m),
                "C2-C": stats(S["C2"][v], S["C"][v], m),
            }
        for thr in (1.0, 5.0, 10.0):
            row[name][f"n_RAINNC_ge{thr}"] = {r: int((S[r]["RAINNC"][m] >= thr).sum()) for r in RUNS}
    # profile means (sea interior) of QCLOUD/QRAIN/QVAPOR/T
    m = masks["sea_int"]
    row["prof_sea_int"] = {v: {r: S[r][v][:, m].mean(1).tolist() for r in RUNS} for v in ("QCLOUD", "QRAIN", "QVAPOR", "T")}
    res["leads"][L] = row
    prev = {r: S[r]["RAINNC"] for r in RUNS}
    print(L, {k: (round(row[k]["HR"]["G-C"]["mean"], 5), round(row[k]["HR"]["C2-C"]["mean"], 5),
                  round(row[k]["HR"]["C_mean"], 4)) for k in masks}, flush=True)
json.dump(res, open(sys.argv[1], "w"))
