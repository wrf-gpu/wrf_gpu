"""NF05: real Swiss snow columns in savepoint-column format (state = CPU-WRF t0, forcing = CPU-WRF h1 lowest level)."""
import importlib.util, json, sys
import numpy as np
from netCDF4 import Dataset
W = "<USER_HOME>/src/wrf_gpu2_wt/b-thompson"
spec = importlib.util.spec_from_file_location("bns", W + "/proofs/noahmp/build_noahmp_savepoints.py")
bns = importlib.util.module_from_spec(spec); spec.loader.exec_module(bns)
R = "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/"
d0, d1 = Dataset(R + "wrfout_d01_2023-01-15_00:00:00"), Dataset(R + "wrfout_d01_2023-01-15_01:00:00")
v0 = lambda n: np.asarray(d0[n][0], float)  # noqa: E731
v1 = lambda n: np.asarray(d1[n][0], float)  # noqa: E731
isn0, iv, il, land = v0("ISNOW"), v0("IVGTYP").astype(int), v0("ISLTYP").astype(int), v0("LANDMASK") > 0.5
ok = land & (iv != 15) & (iv != 17)
liq1 = (v1("SNLIQ") > 0).any(axis=0)
k = (3 + isn0).astype(int).clip(0, 2); top0 = np.take_along_axis(v0("TSNO"), k[None], 0)[0]
rng = np.random.default_rng(15)
def pick(mask, n):
    idx = np.argwhere(mask); return [tuple(x) for x in idx[rng.choice(len(idx), size=min(n, len(idx)), replace=False)]]
cells = ([("melt", c) for c in pick(ok & (isn0 < 0) & liq1, 6)] + [("cold", c) for c in pick(ok & (isn0 < 0) & ~liq1 & (top0 < 268.0), 4)]
         + [("nosnow", c) for c in pick(ok & (isn0 == 0) & (v0("SNOW") == 0), 2)])
dzs = [float(x) for x in d0["DZS"][0]]; zsoil = list(-np.cumsum(dzs))
cols = []
for tag, (i, j) in cells:
    g = lambda n: float(v0(n)[i, j])  # noqa: E731
    lay = lambda n: [float(x) for x in v0(n)[:, i, j]]  # noqa: E731
    f = bns._column_forcing(d1, i, j)
    state = {"stc": lay("TSNO") + lay("TSLB"), "smc": lay("SMOIS"), "sh2o": lay("SH2O"),
             "tv": g("TV"), "tg": g("TG"), "tah": g("TAH"), "eah": g("EAH"), "canliq": g("CANLIQ"), "canice": g("CANICE"),
             "fwet": g("FWET"), "qsfc": f["q2"], "snowh": g("SNOWH"), "sneqv": g("SNOW"), "sneqvo": g("SNEQVO"),
             "albold": g("ALBOLD"), "tauss": g("TAUSS"), "isnow": int(g("ISNOW")), "zsnso": lay("ZSNSO"),
             "snice": lay("SNICE"), "snliq": lay("SNLIQ"), "cm": g("CM"), "ch": g("CH"), "smcwtd": g("SMCWTD")}
    cols.append({"name": f"{tag}_{i}_{j}", "case": tag, "vegtyp": int(iv[i, j]), "isltyp": int(il[i, j]),
                 "lat_rad": float(np.deg2rad(v0("XLAT")[i, j])), "julian": 14.0, "yearlen": 365, "dt": 18.0, "dx": 3000.0,
                 "zsoil": [float(z) for z in zsoil], "dzs": dzs, "shdfac": g("VEGFRA") / 100.0, "shdmax": g("SHDMAX") / 100.0,
                 "tbot": g("TMN"), "lai0": g("LAI"), "sai0": g("XSAI"), "forcing": f, "state_in": state})
json.dump({"dataset": d0.MMINLU, "source": R, "columns": cols}, open(sys.argv[1], "w"), indent=1)
for c in cols:
    s = c["state_in"]; print(c["name"], "veg", c["vegtyp"], "isnow", s["isnow"], "sneqv", round(s["sneqv"], 2), "tsno", [round(x, 2) for x in s["stc"][:3]],
                               "sfctmp", round(c["forcing"]["sfctmp"], 2), "lwdn", round(c["forcing"]["lwdn"], 1), "snice+snliq", round(sum(s["snice"]) + sum(s["snliq"]), 2))
