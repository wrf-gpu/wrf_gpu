"""CAM01: real CPU-WRF WN3 0227 radiation-call operands for the pristine CAM oracle (cam01_driver.F90).

Columns = the RE01 census picks (proofs/rrtmg_mp_re/census.py: d02/d03 x tau 12-36 x liq_thin_ocean/liq_thick/liq_land/ice/
snow/mixed/sgs_only/clear, 186 columns, day 12Z/18Z and night 00Z/06Z). Column reconstruction = RE01 extract.column()
(WRF phy_prep: t_phy, pi_phy, rho, dz8w; radiation P/P8W = hydrostatic p_hyd/p_hyd_w, first_rk_step_part1:280).
CAM-specific operands: CLDFRA = history CLDFRA, z = mass-level height (unused by CAM outputs: only aerosol_indirect, which
has no outputs), qc/qi = resolved QCLOUD/QICE (no QC_BL merge; operands only need to be real-structured), shalf = ZNU.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rrtmg_mp_re"))
from extract import CPU, G, column  # noqa: E402

CENSUS = Path("<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE01/census.json")
RADT, DT, YR = 30.0, 18.0, 2026
# Augmented real-structured columns (E154/E102: the 186 real picks reach at most 2 max-overlap regions): real T/p/qv/surface
# of a base pick, synthetic CLDFRA + condensate patterns (WRF k = 0 bottom) that force 3-5 regions, tied fractions,
# cloud in the lowest and the topmost layer, and a single thin layer. qc where T > 258 K else qi (+ qs in pattern B).
AUG_PATTERNS = {
    "A_five_regions": {2: 0.3, 3: 0.3, 4: 0.3, 8: 1.0, 9: 1.0, 10: 1.0, 15: 0.6, 16: 0.6, 17: 0.8, 18: 0.45, 25: 0.2,
                       26: 0.2, 33: 0.9},
    "B_surface_top_ties": {0: 0.5, 1: 0.5, 2: 0.5, 12: 0.7, 13: 0.25, 14: 0.7, 20: 1.0, 43: 0.15},
    "C_alternating": {k: (0.35 if k % 4 == 0 else 0.0) for k in range(0, 40)},
    "D_single_thin": {6: 0.05},
}
AUG_BASES = 6  # bases per pattern (alternating day / night picks)


def main(out_bin, out_json):
    picks = json.loads(CENSUS.read_text())["picks"]
    cols, meta = [], []
    znu = p_top = gmt = None
    for pk in picks:
        with Dataset(CPU / f"wrfout_{pk['domain']}_{pk['stamp']}") as c:
            col = column(c, pk["j"], pk["i"])
            zw = (np.asarray(c["PH"][0, :, pk["j"], pk["i"]], np.float64)
                  + np.asarray(c["PHB"][0, :, pk["j"], pk["i"]], np.float64)) / G
            col["z"] = 0.5 * (zw[1:] + zw[:-1])
            julday, g = int(c.getncattr("JULDAY")), float(c.getncattr("GMT"))
            pt = float(np.asarray(c["P_TOP"][0]))
            zn = np.asarray(c["ZNU"][0], np.float32)
            start = datetime.strptime(c.getncattr("START_DATE"), "%Y-%m-%d_%H:%M:%S")
        if znu is None:
            znu, p_top, gmt = zn, pt, g
        assert np.array_equal(zn, znu) and pt == p_top and g == gmt
        hours = (datetime.strptime(pk["stamp"], "%Y-%m-%d_%H:%M:%S") - start).total_seconds() / 3600.0
        assert hours == float(pk["tau"]), (pk, hours)
        julian = (julday - 1) + (gmt + hours) / 24.0
        cols.append(col)
        meta.append({**pk, "julday": julday, "julian": julian, "coszen_hist": col["coszen"], "xland": col["xland"],
                     "max_cldfra": float(np.max(col["cldfra"])), "n_cloud_layers": int((col["cldfra"] > 0).sum())})
    base_ids = [i for i, m in enumerate(meta) if m["category"] in ("clear", "liq_land", "ice", "mixed")]
    day = [i for i in base_ids if meta[i]["tau"] in (12, 36)]
    night = [i for i in base_ids if meta[i]["tau"] in (24, 30)]
    for pi_, (pname, pat) in enumerate(AUG_PATTERNS.items()):
        for b in range(AUG_BASES):
            src = (day if b % 2 == 0 else night)[(3 * pi_ + b) % len(day if b % 2 == 0 else night)]
            col = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in cols[src].items()}
            cf = np.zeros_like(col["cldfra"]); qc = np.zeros_like(col["qc"]); qi = np.zeros_like(col["qi"])
            qs = np.zeros_like(col["qs"])
            for k, frac in pat.items():
                cf[k] = frac
                if frac > 0:
                    if col["t"][k] > 258.0:
                        qc[k] = 2.0e-4 * (1.0 + 0.1 * b)
                    else:
                        qi[k] = 4.0e-5 * (1.0 + 0.1 * b)
                        if pname.startswith("B"):
                            qs[k] = 2.0e-5
            col.update(cldfra=cf, qc=qc, qi=qi, qs=qs)
            cols.append(col)
            meta.append({**meta[src], "category": f"aug_{pname}", "aug_base_index": src,
                         "max_cldfra": float(cf.max()), "n_cloud_layers": int((cf > 0).sum())})
    nz = cols[0]["p"].size
    ncol = len(cols)
    f4 = lambda a: np.asarray(a, np.float32)
    with open(out_bin, "wb") as f:
        np.asarray([ncol, nz], np.int32).tofile(f)
        f4([p_top, gmt, RADT, DT]).tofile(f)
        np.asarray([YR], np.int32).tofile(f)
        f4(znu).tofile(f)
        f4([m["julian"] for m in meta]).tofile(f)
        np.asarray([m["julday"] for m in meta], np.int32).tofile(f)
        for name in ("tsk", "emiss", "albedo", "xland", "xlat", "xlong", "snow", "xice"):
            f4([c[name] for c in cols]).tofile(f)
        f4([m["tau"] for m in meta]).tofile(f)
        for name in ("t", "p_hyd", "pi", "rho", "z", "qv", "qc", "qr", "qi", "qs", "qg", "cldfra"):
            f4([c[name] for c in cols]).ravel(order="F").tofile(f)
        for name in ("p_hyd_w", "dz8w"):
            f4([c[name] for c in cols]).ravel(order="F").tofile(f)
    Path(out_json).write_text(json.dumps({"ncol": ncol, "nz": nz, "p_top": p_top, "gmt": gmt, "radt": RADT, "dt": DT,
                                          "yr": YR, "znu": [float(x) for x in znu], "columns": meta}, indent=1))
    print(ncol, "columns", nz, "levels")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
