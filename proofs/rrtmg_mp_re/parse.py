"""RE01: parse the pristine RRTMG has_req oracle output -> fixture NPZ + miswire summary JSON.

Arms: A = WRF truth (has_reqc/i/s=1, Thompson radii, cldovrlp=2), B = has_req=0, C = has_req=1 + constant 10/30/75 um,
D = C with cldovrlp=1 (port today), E = A with cldovrlp=1.
Frozen tier-1 RRTMG bounds (src/gpuwrf/validation/tier1_rrtmg.py): flux 1.0 W/m2 abs + 0.05 rel, heating 1e-4 abs + 0.05 rel.
"""
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("A", "B", "C", "D", "E")
SCAL = ("glw", "olr", "lwcf", "lwupt", "lwuptc", "lwdnt", "lwdntc", "lwupb", "lwupbc", "lwdnb", "lwdnbc",
        "gsw", "swcf", "swupt", "swuptc", "swdnt", "swdntc", "swupb", "swupbc", "swdnb", "swdnbc",
        "swddir", "swddni", "swddif", "swdownc", "coszr")
INPUT_SCAL = ("tsk", "emiss", "albedo", "coszen_hist", "xland", "xlat", "xlong", "snow", "xice", "lat_d01")
INPUT_3D = ("t", "p", "pi", "th", "rho", "qv", "qc", "qi", "ni", "qs", "qr", "qg", "qc_rad", "qi_rad", "cldfra", "p_hyd",
            "p_d01")
INPUT_W = ("dz8w", "p_hyd_w", "t8w")


def read_input(path):
    raw = Path(path).read_bytes()
    ncol, nz = np.frombuffer(raw, np.int32, 2)
    off = 8
    def take(dtype, n):
        nonlocal off
        a = np.frombuffer(raw, dtype, n, off).copy(); off += a.nbytes; return a
    p_top, gmt = take(np.float32, 2)
    out = {"p_top": p_top, "gmt": gmt, "julian": take(np.float32, ncol), "julday": take(np.int32, ncol)}
    for n in INPUT_SCAL:
        out[n] = take(np.float32, ncol)
    for n in INPUT_3D:
        out[n] = take(np.float32, ncol * nz).reshape(nz, ncol).T
    for n in INPUT_W:
        out[n] = take(np.float32, ncol * (nz + 1)).reshape(nz + 1, ncol).T
    out["tau"] = take(np.float32, ncol)
    assert off == len(raw), (off, len(raw))
    return int(ncol), int(nz), out


def read_output(path, ncol, nz):
    raw = np.frombuffer(Path(path).read_bytes(), np.float32)
    assert tuple(raw[:2].view(np.int32)) == (ncol, nz)
    pos = 2
    def take(n):
        nonlocal pos
        a = raw[pos:pos + n]; pos += n; return a
    col = {k: [] for k in ("solcon", "declin", "coszen", "o3", "re_cloud", "re_ice", "re_snow")}
    arm = {a: {k: [] for k in ("hlw", "hlwc", "hsw", "hswc", "lwup", "lwupc", "lwdn", "lwdnc",
                               "swup", "swupc", "swdn", "swdnc", *SCAL)} for a in ARMS}
    for _ in range(ncol):
        s = take(3)
        col["solcon"].append(s[0]); col["declin"].append(s[1]); col["coszen"].append(s[2])
        for k in ("o3", "re_cloud", "re_ice", "re_snow"):
            col[k].append(take(nz))
        for a in ARMS:
            for k in ("hlw", "hlwc", "hsw", "hswc"):
                arm[a][k].append(take(nz))
            for k in ("lwup", "lwupc", "lwdn", "lwdnc", "swup", "swupc", "swdn", "swdnc"):
                arm[a][k].append(take(nz + 2))
            sc = take(len(SCAL))
            for k, v in zip(SCAL, sc):
                arm[a][k].append(v)
    assert pos == raw.size, (pos, raw.size)
    col = {k: np.asarray(v, np.float32) for k, v in col.items()}
    arm = {a: {k: np.asarray(v, np.float32) for k, v in d.items()} for a, d in arm.items()}
    return col, arm


def excess(ref, got, abs_tol, rel_tol):
    """max over elements of |got-ref| / (abs_tol + rel_tol*|ref|) — >1 means outside the frozen bound."""
    return np.abs(got.astype(np.float64) - ref) / (abs_tol + rel_tol * np.abs(ref.astype(np.float64)))


def main(inp, outp, meta_json, npz_out, summary_out):
    ncol, nz, x = read_input(inp)
    col, arm = read_output(outp, ncol, nz)
    meta = json.loads(Path(meta_json).read_text())["columns"]
    cats = np.asarray([m["category"] for m in meta])
    fx = {f"in_{k}": np.asarray(v) for k, v in x.items()}
    fx.update({f"col_{k}": v for k, v in col.items()})
    for a in ARMS:
        fx.update({f"{a}_{k}": v for k, v in arm[a].items()})
    fx["category"] = cats
    fx["domain"] = np.asarray([m["domain"] for m in meta])
    fx["ji"] = np.asarray([[m["j"], m["i"]] for m in meta], np.int32)
    np.savez_compressed(npz_out, **fx)

    A = arm["A"]
    summ = {"ncol": ncol, "nz": nz, "categories": {c: int((cats == c).sum()) for c in np.unique(cats)},
            "sanity": {}, "arms": {}}
    clear = cats == "clear"
    summ["sanity"] = {
        "solcon_range": [float(col["solcon"].min()), float(col["solcon"].max())],
        "coszen_range": [float(col["coszen"].min()), float(col["coszen"].max())],
        "coszen_vs_hist_max_abs": float(np.abs(col["coszen"] - x["coszen_hist"]).max()),
        "o3_vmr_range": [float(col["o3"].min()), float(col["o3"].max())],
        "re_cloud_um_range_where_qc": [float(col["re_cloud"][x["qc"] > 1e-6].min() * 1e6),
                                       float(col["re_cloud"][x["qc"] > 1e-6].max() * 1e6)],
        "re_ice_um_range_where_qi": [float(col["re_ice"][x["qi"] > 1e-8].min() * 1e6) if (x["qi"] > 1e-8).any() else None,
                                     float(col["re_ice"][x["qi"] > 1e-8].max() * 1e6) if (x["qi"] > 1e-8).any() else None],
        "re_snow_um_range_where_qs": [float(col["re_snow"][x["qs"] > 1e-8].min() * 1e6) if (x["qs"] > 1e-8).any() else None,
                                      float(col["re_snow"][x["qs"] > 1e-8].max() * 1e6) if (x["qs"] > 1e-8).any() else None],
        "clear_cols_A_eq_B_eq_C_glw": bool(np.array_equal(A["glw"][clear], arm["B"]["glw"][clear])
                                           and np.array_equal(A["glw"][clear], arm["C"]["glw"][clear])),
        "cloudy_levels_with_cldfra0": int(((x["qc_rad"] > 1e-6) & (x["cldfra"] <= 0)).sum()),
        "cldfra_levels_with_re_bg_liquid_fallback": int(((x["cldfra"] > 0) & (col["re_cloud"] * 1e6 <= 2.5)).sum()),
        "finite": bool(all(np.isfinite(v).all() for d in arm.values() for v in d.values())),
    }
    flux_keys = ("glw", "olr", "lwupt", "lwdnb", "lwupb", "gsw", "swdnb", "swupt", "swupb", "swddir", "swddif")
    for a in ("B", "C", "D", "E"):
        d = arm[a]
        res = {}
        for c in [*np.unique(cats), "ALL"]:
            m = np.ones(ncol, bool) if c == "ALL" else cats == c
            e = {}
            for k in flux_keys:
                diff = d[k][m].astype(np.float64) - A[k][m]
                e[k] = {"max_abs": float(np.abs(diff).max()), "mean": float(diff.mean()),
                        "max_bound_ratio": float(excess(A[k][m], d[k][m], 1.0, 0.05).max())}
            for k in ("hlw", "hsw"):
                diff = d[k][m].astype(np.float64) - A[k][m]
                e[k] = {"max_abs_K_per_day": float(np.abs(diff).max() * 86400),
                        "max_bound_ratio": float(excess(A[k][m], d[k][m], 1e-4, 0.05).max())}
            fails = sum(int(excess(A[k][m], d[k][m], 1.0, 0.05).max() > 1) for k in flux_keys)
            fails += sum(int(excess(A[k][m], d[k][m], 1e-4, 0.05).max() > 1) for k in ("hlw", "hsw"))
            ncols_fail = int(((np.stack([excess(A[k][m], d[k][m], 1.0, 0.05) for k in flux_keys]).max(0) > 1)
                              | (excess(A["hlw"][m], d["hlw"][m], 1e-4, 0.05).max(1) > 1)
                              | (excess(A["hsw"][m], d["hsw"][m], 1e-4, 0.05).max(1) > 1)).sum())
            e["n_quantities_outside_frozen_bounds"] = fails
            e["n_columns_outside_frozen_bounds"] = ncols_fail
            e["n_columns"] = int(m.sum())
            res[c] = e
        summ["arms"][f"{a}_minus_A"] = res
    Path(summary_out).write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["sanity"], indent=1))
    for a in ("B", "C", "D", "E"):
        print(f"== {a} - A")
        for c, e in summ["arms"][f"{a}_minus_A"].items():
            print(f"{c:16s} n={e['n_columns']:3d} fail_cols={e['n_columns_outside_frozen_bounds']:3d}"
                  f" GLW max {e['glw']['max_abs']:7.2f} mean {e['glw']['mean']:+7.2f}"
                  f" | SWDOWN max {e['swdnb']['max_abs']:7.2f} mean {e['swdnb']['mean']:+7.2f}"
                  f" | OLR max {e['olr']['max_abs']:6.2f} | hLW {e['hlw']['max_abs_K_per_day']:6.2f} K/d"
                  f" hSW {e['hsw']['max_abs_K_per_day']:6.2f} K/d")


if __name__ == "__main__":
    main(*sys.argv[1:6])
