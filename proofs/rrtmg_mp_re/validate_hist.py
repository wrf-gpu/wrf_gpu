"""RE01 independent check: which caller wiring reproduces CPU-WRF's own radiation history best?

The CPU history at tau holds the radiation call at tau-30 min (COSZEN verified to 8.7e-5), the oracle arms the call at tau
on the history state -> 30 min of state evolution separates them (noise), the radius wiring is systematic (signal).
LW: LWDNB (=GLW), LWUPT (=OLR). SW (coszen>0.15 both calls): transmissivity SWDNB/SWDNT and TOA albedo SWUPT/SWDNT.
"""
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
fx = np.load("re01_fixture.npz")
meta = json.loads(Path("re01_columns.json").read_text())["columns"]
hist = {k: [] for k in ("LWDNB", "LWUPT", "SWDNB", "SWDNT", "SWUPT", "COSZEN")}
for m in meta:
    with Dataset(CPU / f"wrfout_{m['domain']}_{m['stamp']}") as d:
        for k in hist:
            hist[k].append(float(d[k][0, m["j"], m["i"]]))
hist = {k: np.asarray(v) for k, v in hist.items()}
cat = fx["category"]
cloudy = cat != "clear"
day = (hist["COSZEN"] > 0.15) & (fx["col_coszen"] > 0.15)
out = {}
for arm in ("A", "B", "C", "D", "E"):
    r = {}
    for name, a, h, mask in (
        ("LWDNB", fx[f"{arm}_lwdnb"], hist["LWDNB"], cloudy),
        ("LWUPT", fx[f"{arm}_lwupt"], hist["LWUPT"], cloudy),
        ("SW_transmissivity", fx[f"{arm}_swdnb"] / np.maximum(fx[f"{arm}_swdnt"], 1e-3), hist["SWDNB"] / np.maximum(hist["SWDNT"], 1e-3), cloudy & day),
        ("SW_toa_albedo", fx[f"{arm}_swupt"] / np.maximum(fx[f"{arm}_swdnt"], 1e-3), hist["SWUPT"] / np.maximum(hist["SWDNT"], 1e-3), cloudy & day),
    ):
        d = (a - h)[mask]
        r[name] = {"n": int(mask.sum()), "bias": float(d.mean()), "rms": float(np.sqrt((d ** 2).mean())),
                   "median_abs": float(np.median(np.abs(d)))}
    # category breakdown for the marine thin-cloud case (mass-opus F3 regime)
    for c in ("liq_thin_ocean", "liq_land", "ice", "snow"):
        m = (cat == c)
        md = m & day
        r[f"{c}_LWDNB_rms"] = float(np.sqrt(((fx[f"{arm}_lwdnb"] - hist["LWDNB"])[m] ** 2).mean()))
        if md.any():
            t_a = fx[f"{arm}_swdnb"] / np.maximum(fx[f"{arm}_swdnt"], 1e-3)
            t_h = hist["SWDNB"] / np.maximum(hist["SWDNT"], 1e-3)
            r[f"{c}_SWtrans_rms_day"] = float(np.sqrt(((t_a - t_h)[md] ** 2).mean()))
            r[f"{c}_SWtrans_bias_day"] = float((t_a - t_h)[md].mean())
            r[f"{c}_n_day"] = int(md.sum())
    out[arm] = r
Path("re01_vs_cpu_history.json").write_text(json.dumps(out, indent=1))
for arm, r in out.items():
    print(arm, " ".join(f"{k}: rms {v['rms']:.4g} bias {v['bias']:+.4g} (n={v['n']})" for k, v in r.items() if isinstance(v, dict)))
    print("   ", " ".join(f"{k}={v:.4g}" for k, v in r.items() if not isinstance(v, dict)))
