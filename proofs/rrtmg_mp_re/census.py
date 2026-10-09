"""RE01 census: real WN3 0227 CPU-WRF d02/d03 cloudy columns for the RRTMG has_req=1 oracle.

Categories (water paths in g/m2 from resolved hydrometeors): liq_thin_ocean (5<LWP<40, IWP<1, SWP<1, ocean),
liq_thick (LWP>=100), liq_land (LWP>5 over land), ice (IWP>5, LWP<1), snow (SWP>5), mixed (LWP>5 & (IWP+SWP)>5),
sgs_only (CLDFRA_BL>0.001 with qc<1e-6 in >=2 layers, GPU QC_BL operand; tau<=24, inside the 24 h GPU run), clear (max CLDFRA==0).
Picks up to 3 per (domain, tau, category): max-path, median-path, and a third distinct cell; >=6 cells from the edge.
"""
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
GPU = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/W9/finalb_r3/20260227_18z_a1/gpu_24h")
TAUS = {12: "2026-02-28_12:00:00", 18: "2026-02-28_18:00:00", 24: "2026-03-01_00:00:00",
        30: "2026-03-01_06:00:00", 36: "2026-03-01_12:00:00"}  # leads from START_DATE 2026-02-28_00Z (GMT 0, JULDAY 59)
G = 9.81


def paths(ds):
    p = np.asarray(ds["P"][0], np.float64) + np.asarray(ds["PB"][0], np.float64)
    phw = (np.asarray(ds["PH"][0], np.float64) + np.asarray(ds["PHB"][0], np.float64)) / G
    dz = phw[1:] - phw[:-1]
    thm = np.asarray(ds["THM"][0], np.float64) + 300.0
    alt = (287.0 / 1e5) * thm * (p / 1e5) ** (-(1004.5 - 287.0) / 1004.5)
    qv = np.asarray(ds["QVAPOR"][0], np.float64)
    rho = (1.0 + qv) / alt
    out = {}
    for name, var in (("LWP", "QCLOUD"), ("IWP", "QICE"), ("SWP", "QSNOW"), ("RWP", "QRAIN"), ("GWP", "QGRAUP")):
        out[name] = (np.asarray(ds[var][0], np.float64) * rho * dz).sum(0) * 1e3
    return out


def main():
    picks, stats = [], {}
    for dom in ("d02", "d03"):
        for tau, stamp in TAUS.items():
            with Dataset(CPU / f"wrfout_{dom}_{stamp}") as c:
                wp = paths(c)
                land = np.asarray(c["XLAND"][0]) < 1.5
                cf = np.asarray(c["CLDFRA"][0]).max(0)
                qc = np.asarray(c["QCLOUD"][0])
                ny, nx = cf.shape
            jj, ii = np.mgrid[0:ny, 0:nx]
            inner = (jj >= 6) & (ii >= 6) & (jj < ny - 6) & (ii < nx - 6)
            cats = {
                "liq_thin_ocean": (wp["LWP"] > 5) & (wp["LWP"] < 40) & (wp["IWP"] < 1) & (wp["SWP"] < 1) & ~land,
                "liq_thick": (wp["LWP"] >= 100),
                "liq_land": (wp["LWP"] > 5) & land,
                "ice": (wp["IWP"] > 5) & (wp["LWP"] < 1),
                "snow": (wp["SWP"] > 5),
                "mixed": (wp["LWP"] > 5) & ((wp["IWP"] + wp["SWP"]) > 5),
                "clear": cf <= 0.0,
            }
            g = GPU / f"wrfout_{dom}_{stamp}"
            if g.is_file():
                with Dataset(g) as gd:
                    cfbl = np.asarray(gd["CLDFRA_BL"][0]); qcbl = np.asarray(gd["QC_BL"][0])
                sgs_layers = ((cfbl > 0.001) & (qc < 1e-6) & (qcbl > 0)).sum(0)
                cats["sgs_only"] = sgs_layers >= 2
                key_sgs = sgs_layers
            for cat, mask in cats.items():
                mask = mask & inner
                n = int(mask.sum())
                stats[f"{dom}_t{tau}_{cat}"] = n
                if n == 0:
                    continue
                score = {"liq_thin_ocean": wp["LWP"], "liq_thick": wp["LWP"], "liq_land": wp["LWP"], "ice": wp["IWP"],
                         "snow": wp["SWP"], "mixed": wp["LWP"] + wp["IWP"] + wp["SWP"], "clear": -wp["LWP"],
                         "sgs_only": key_sgs if cat == "sgs_only" else None}[cat]
                cand = np.flatnonzero(mask.ravel())
                order = cand[np.argsort(score.ravel()[cand])]
                chosen = {int(order[-1]), int(order[len(order) // 2]), int(order[len(order) // 4])}
                for flat in sorted(chosen):
                    j, i = divmod(flat, nx)
                    picks.append({"domain": dom, "tau": tau, "stamp": stamp, "category": cat, "j": int(j), "i": int(i),
                                  "LWP": float(wp["LWP"][j, i]), "IWP": float(wp["IWP"][j, i]), "SWP": float(wp["SWP"][j, i]),
                                  "land": bool(land[j, i])})
    Path("census.json").write_text(json.dumps({"stats": stats, "picks": picks}, indent=1))
    print(len(picks), "picks")
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
