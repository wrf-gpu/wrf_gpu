"""NF05: real rainy LAND columns (CPU-WRF state at h0, lowest-level forcing at h1, precip = hourly accumulation deltas h0->h1)."""
import importlib.util, json, sys
import numpy as np
from netCDF4 import Dataset
out, case, dom, h0, h1, julian, dt, dx = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])
W = "<USER_HOME>/src/wrf_gpu2_wt/b-thompson"
spec = importlib.util.spec_from_file_location("bns", W + "/proofs/noahmp/build_noahmp_savepoints.py")
bns = importlib.util.module_from_spec(spec); spec.loader.exec_module(bns)
d0, d1 = Dataset(f"{case}/wrfout_{dom}_{h0}"), Dataset(f"{case}/wrfout_{dom}_{h1}")
v0 = lambda n: np.asarray(d0[n][0], float)  # noqa: E731
v1 = lambda n: np.asarray(d1[n][0], float)  # noqa: E731
acc = lambda n: (v1(n) - v0(n)) / 3600.0 if n in d0.variables else np.zeros_like(v0("RAINNC"))  # noqa: E731
rnc, rc, snc, gnc, hnc = acc("RAINNC"), acc("RAINC"), acc("SNOWNC"), acc("GRAUPELNC"), acc("HAILNC")
iv, il = v0("IVGTYP").astype(int), v0("ISLTYP").astype(int)
ok = (v0("LANDMASK") > 0.5) & ~np.isin(iv, (15, 17)) & (v0("ISNOW") == 0)
rng = np.random.default_rng(7); tag = sys.argv[9] if len(sys.argv) > 9 else dom
def pick(mask, n, key=None):
    idx = np.argwhere(mask)
    if len(idx) == 0: return []
    if key is not None: idx = idx[np.argsort(-key[mask])][:n]; return [tuple(x) for x in idx]
    return [tuple(x) for x in idx[rng.choice(len(idx), size=min(n, len(idx)), replace=False)]]
rain = rnc + rc
cells = [("rain", c) for c in pick(ok & (rain > 0), 4, rain)]
cells += [("grpl", c) for c in pick(ok & (gnc + snc + hnc > 0) & (rain > 0), 3, gnc + snc + hnc) if ("rain", c) not in cells]
cells += [("coldtv", c) for c in pick(ok & (rain > 0) & (v0("TV") < 275.0), 3, -v0("TV")) if all(c != x[1] for x in cells)]
dzs = [float(x) for x in d0["DZS"][0]]; zsoil = list(-np.cumsum(dzs))
cols = []
for kind, (i, j) in cells:
    g = lambda n: float(v0(n)[i, j])  # noqa: E731
    lay = lambda n: [float(x) for x in v0(n)[:, i, j]]  # noqa: E731
    f = bns._column_forcing(d1, i, j)
    f.update(prcpconv=float(rc[i, j]), prcpnonc=float(rnc[i, j]), prcpsnow=float(snc[i, j]), prcpgrpl=float(gnc[i, j]), prcphail=float(hnc[i, j]))
    state = {"stc": lay("TSNO") + lay("TSLB"), "smc": lay("SMOIS"), "sh2o": lay("SH2O"),
             "tv": g("TV"), "tg": g("TG"), "tah": g("TAH"), "eah": g("EAH"), "canliq": g("CANLIQ"), "canice": g("CANICE"),
             "fwet": g("FWET"), "qsfc": f["q2"], "snowh": g("SNOWH"), "sneqv": g("SNOW"), "sneqvo": g("SNEQVO"),
             "albold": g("ALBOLD"), "tauss": g("TAUSS"), "isnow": int(g("ISNOW")), "zsnso": lay("ZSNSO"),
             "snice": lay("SNICE"), "snliq": lay("SNLIQ"), "cm": g("CM"), "ch": g("CH"), "smcwtd": g("SMCWTD")}
    cols.append({"name": f"{tag}_{kind}_{i}_{j}", "case": tag, "vegtyp": int(iv[i, j]), "isltyp": int(il[i, j]),
                 "lat_rad": float(np.deg2rad(v0("XLAT")[i, j])), "julian": julian, "yearlen": 365, "dt": dt, "dx": dx,
                 "zsoil": [float(z) for z in zsoil], "dzs": dzs, "shdfac": g("VEGFRA") / 100.0, "shdmax": g("SHDMAX") / 100.0,
                 "tbot": g("TMN"), "lai0": g("LAI"), "sai0": g("XSAI"), "forcing": f, "state_in": state})
json.dump({"dataset": d0.MMINLU, "source": f"{case} {dom} {h0}->{h1}", "columns": cols}, open(out, "w"), indent=1)
for c in cols:
    f = c["forcing"]; print(c["name"], "veg", c["vegtyp"], "rain %.2e conv %.2e snow %.2e grpl %.2e" % (f["prcpnonc"], f["prcpconv"], f["prcpsnow"], f["prcpgrpl"]), "tv %.2f sfctmp %.2f canliq %.3f" % (c["state_in"]["tv"], f["sfctmp"], c["state_in"]["canliq"]))
