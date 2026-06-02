"""Build WRF Noah-classic (sf_surface_physics=2) savepoint fixtures (v0.6.0 lane 14).

EXTERNAL ORACLE — NOT a self-compare. Mirrors proofs/noahmp/build_noahmp_savepoints.py:

  1. extracts representative Canary LAND columns across regimes (daytime / nighttime
     vegetated + bare + a Teide snow column + a synthetic wet-soil precip column)
     from the corpus d03 wrfout/wrfinput;
  2. writes the column forcing + prognostic land state into the Fortran driver's
     ``noahclassic_columns.in``;
  3. runs ``noahclassic_offline_driver.exe`` — which links the COMPILED pristine WRF
     ``module_sf_noahlsm.o`` and calls the exact ``SFLX`` orchestrator on each column
     under the FROZEN WRF-coupled option set (LOCAL=F, UA_PHYS=F, OPT_THCND=1, ICE=0);
  4. parses the per-column output into proofs/v060/savepoints_noahclassic.json — each
     savepoint carries the full per-column INPUT snapshot + the WRF-computed OUTPUT
     (TSK/TSLB/SMOIS/SH2O/HFX/QFX/GRDFLX/...), the parity gate's oracle.

Run (CPU only; cores 0-3):
    cd proofs/v060/oracle && ./build_driver.sh          # one-time Fortran build
    taskset -c 0-3 python3 proofs/v060/build_noahclassic_savepoints.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"netCDF4 required: {exc}")

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "oracle"
RUN = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
DAY = "wrfout_d03_2026-05-22_12:00:00"
NIGHT = "wrfout_d03_2026-05-22_03:00:00"
DRIVER = ORACLE / "noahclassic_offline_driver.exe"

NSOIL = 4
G = 9.80665
DATASET = "MODIFIED_IGBP_MODIS_NOAH"
SLDPTH = [0.10, 0.30, 0.60, 1.00]      # DZS (layer thicknesses, >0)
# ZSOIL = cumulative -SLDPTH, exactly as SFLX computes it (NOT the ZS midpoints).
ZSOIL = [-0.10, -0.40, -1.00, -2.00]   # interface depths (cumulative, <0)
DT = 90.0                              # representative physics step
DX = 1000.0                            # d03 grid spacing
JULIAN = 141.5 + 0.5
YEARLEN = 365
SLOPETYP = 1
SOILCOLOR = 4
ISURBAN = 13
ISICE = 15
ISWATER = 17

# Representative Canary land columns (IGBP/MODIS veg categories present in d03).
#   veg 5 Mixed Forests, 9 Savannas, 10 Grasslands, 12 Croplands, 16 Barren.
VEG_CELLS = {5: (44, 23), 9: (32, 54), 10: (31, 50), 12: (48, 41), 16: (30, 36)}
TEIDE_CELL = (34, 34)                  # max-terrain barren column (snow regime)


def _v(d, name):
    return np.asarray(d.variables[name][0], dtype=np.float64)


def _column_forcing(d, i, j):
    T = _v(d, "T"); P = _v(d, "P"); PB = _v(d, "PB")
    PH = _v(d, "PH"); PHB = _v(d, "PHB"); QV = _v(d, "QVAPOR")
    U = _v(d, "U"); V = _v(d, "V")
    theta = T[0, i, j] + 300.0
    p = P[0, i, j] + PB[0, i, j]
    rovcp = 287.0 / 1004.0
    sfctmp = theta * (p / 1.0e5) ** rovcp
    qv = max(float(QV[0, i, j]), 1e-8)
    dz = float(((PH + PHB)[1, i, j] - (PH + PHB)[0, i, j]) / G)
    uu = 0.5 * (U[0, i, j] + U[0, i, j + 1])
    vv = 0.5 * (V[0, i, j] + V[0, i + 1, j])
    return {
        "sfctmp": float(sfctmp), "sfcprs": float(p), "psfc": float(_v(d, "PSFC")[i, j]),
        "uu": float(uu), "vv": float(vv), "q2k": qv, "qc": 0.0,
        "soldn": float(_v(d, "SWDOWN")[i, j]), "glw": float(_v(d, "GLW")[i, j]),
        "zlvl": 0.5 * dz, "dz8w": dz,
    }


def _column_state(d, i, j, *, snow=False, wet=False):
    tslb = _v(d, "TSLB")[:, i, j]
    smois = _v(d, "SMOIS")[:, i, j]
    sh2o = _v(d, "SH2O")[:, i, j]
    tsk = float(_v(d, "TSK")[i, j])
    canwat = float(_v(d, "CANWAT")[i, j]) if "CANWAT" in d.variables else 0.0
    snow_mm = float(_v(d, "SNOW")[i, j]) if "SNOW" in d.variables else 0.0
    snowh = float(_v(d, "SNOWH")[i, j]) if "SNOWH" in d.variables else 0.0
    sncovr = float(_v(d, "SNOWC")[i, j]) if "SNOWC" in d.variables else 0.0
    albbck = float(_v(d, "ALBBCK")[i, j]) if "ALBBCK" in d.variables else 0.18
    emiss = float(_v(d, "EMISS")[i, j]) if "EMISS" in d.variables else 0.98
    snoalb = float(_v(d, "SNOALB")[i, j]) if "SNOALB" in d.variables else 0.70
    z0brd = 0.1
    rainbl = 0.0
    smc = [float(x) for x in smois]
    swc = [float(x) for x in sh2o]
    if snow:
        snow_mm = 75.0
        snowh = 0.30
        sncovr = 0.0   # SFLX recomputes SNCOVR from SNEQV/SNUP
        albbck = 0.18
        snoalb = 0.75
    if wet:
        rainbl = 5.0   # 5 mm over the step -> nonzero infiltration path
        smc = [min(x + 0.05, 0.45) for x in smc]
        swc = [min(x + 0.05, 0.45) for x in swc]
    return {
        "stc": [float(x) for x in tslb], "smc": smc, "swc": swc,
        "t1": tsk, "chk": 0.02, "cmk": 0.02,
        "snow_mm": snow_mm, "snowhk": snowh, "sncovr": sncovr,
        "snotime1": 0.0, "ribb": 0.0,
        "albbck": albbck, "z0brd": z0brd, "emiss": emiss,
        "snoalb": snoalb, "canwat": canwat, "rainbl": rainbl,
    }


def _write_columns(cols: list[dict]) -> None:
    L = []
    L.append(DATASET)
    L.append(f"{len(cols)} {SLOPETYP} {SOILCOLOR} {DT} {ISURBAN} {ISICE} {ISWATER}")
    L.append(" ".join(f"{z}" for z in SLDPTH))
    for c in cols:
        f, s = c["forcing"], c["state"]
        L.append(f"{c['vegtyp']} {c['soiltyp']}")
        L.append(f"{c['lat']} {JULIAN} {YEARLEN} 0.0 {DX} {f['dz8w']} {f['zlvl']}")
        L.append(f"{c['shdfac']} {c['shmin']} {c['shmax']} {c['tbot']}")
        L.append(" ".join(str(x) for x in [f["sfctmp"], f["sfcprs"], f["psfc"],
                  f["uu"], f["vv"], f["q2k"], f["qc"], f["soldn"], f["glw"]]))
        L.append(" ".join(str(x) for x in [s["rainbl"], 0.0, s["snoalb"]]))
        L.append(" ".join(str(x) for x in [s["t1"], s["chk"], s["cmk"]]))
        L.append(" ".join(str(x) for x in [s["snow_mm"], s["snowhk"], s["sncovr"],
                  s["snotime1"], s["ribb"]]))
        L.append(" ".join(str(x) for x in [s["albbck"], s["z0brd"], s["emiss"]]))
        L.append(" ".join(str(x) for x in s["stc"]))
        L.append(" ".join(str(x) for x in s["smc"]))
        L.append(" ".join(str(x) for x in s["swc"]))
    (ORACLE / "noahclassic_columns.in").write_text("\n".join(L) + "\n")


def _parse_savepoints() -> list[dict]:
    text = (ORACLE / "noahclassic_savepoints.out").read_text().splitlines()
    cols, cur = [], None

    def nums(line):
        out = []
        for tok in line.split()[1:]:
            try:
                out.append(float(tok))
            except ValueError:
                pass
        return out

    for ln in text:
        if ln.startswith("COL"):
            cur = {}
        elif ln.startswith("ENDCOL"):
            cols.append(cur); cur = None
        elif cur is None:
            continue
        elif ln.startswith("CAT "):
            v = nums(ln); cur["vegtyp"], cur["soiltyp"] = int(v[0]), int(v[1])
        elif ln.startswith("FORCING3"):
            cur["forcing3"] = dict(zip(["ffrozp", "zlvl", "lwdn"], nums(ln)))
        elif ln.startswith("FORCING2"):
            cur["forcing2"] = dict(zip(["th2", "q2sat", "dqsdt2", "prcp", "solnet"], nums(ln)))
        elif ln.startswith("FORCING "):
            cur["forcing"] = dict(zip(["sfctmp", "sfcprs", "q2k", "soldn", "glw", "uu", "vv"], nums(ln)))
        elif ln.startswith("CHCM_IN"):
            cur["chcm_in"] = dict(zip(["chk", "cmk", "ribb"], nums(ln)))
        elif ln.startswith("T1_IN"):
            cur["t1_in"] = nums(ln)[0]
        elif ln.startswith("STC_IN"):
            cur["stc_in"] = nums(ln)
        elif ln.startswith("SMC_IN"):
            cur["smc_in"] = nums(ln)
        elif ln.startswith("SH2O_IN"):
            cur["sh2o_in"] = nums(ln)
        elif ln.startswith("SNOW_IN"):
            cur["snow_in"] = dict(zip(["sneqv", "snowh", "sncovr", "cmc"], nums(ln)))
        elif ln.startswith("T1_OUT"):
            cur["t1_out"] = nums(ln)[0]
        elif ln.startswith("STC_OUT"):
            cur["stc_out"] = nums(ln)
        elif ln.startswith("SMC_OUT"):
            cur["smc_out"] = nums(ln)
        elif ln.startswith("SH2O_OUT"):
            cur["sh2o_out"] = nums(ln)
        elif ln.startswith("SNOW_OUT"):
            cur["snow_out"] = dict(zip(["sneqv", "snowh", "sncovr", "cmc"], nums(ln)))
        elif ln.startswith("FLUX "):
            cur["flux"] = dict(zip(["hfx", "qfx", "lh", "grdflx", "etp"], nums(ln)))
        elif ln.startswith("DIAG "):
            cur["diag"] = dict(zip(["albedo", "emiss", "z0", "q1", "snomlt"], nums(ln)))
        elif ln.startswith("EVAP "):
            cur["evap"] = dict(zip(["edir", "ec", "ett", "esnow", "dew"], nums(ln)))
        elif ln.startswith("FLXX "):
            cur["flxx"] = dict(zip(["flx1", "flx2", "flx3", "beta"], nums(ln)))
        elif ln.startswith("RUNOFF "):
            cur["runoff"] = dict(zip(["runoff1", "runoff2", "runoff3"], nums(ln)))
        elif ln.startswith("PARM "):
            v = nums(ln)
            cur["parm"] = dict(zip(["smcmax", "smcwlt", "smcref", "smcdry", "nroot"], v))
        elif ln.startswith("RP1 "):
            cur.setdefault("redprm", {}).update(dict(zip(
                ["bexp", "dksat", "dwsat", "psisat", "quartz", "f1"], nums(ln))))
        elif ln.startswith("RP2 "):
            cur.setdefault("redprm", {}).update(dict(zip(
                ["smcmax", "smcwlt", "smcref", "smcdry", "kdt", "frzx"], nums(ln))))
        elif ln.startswith("RP3 "):
            cur.setdefault("redprm", {}).update(dict(zip(
                ["slope", "snup", "salp", "czil", "sbeta", "csoil"], nums(ln))))
        elif ln.startswith("RP4 "):
            cur.setdefault("redprm", {}).update(dict(zip(
                ["fxexp", "zbot", "cfactr", "cmcmax", "rsmax", "topt"], nums(ln))))
        elif ln.startswith("RP5 "):
            cur.setdefault("redprm", {}).update(dict(zip(
                ["rgl", "hs", "rsmin", "lvcoef", "shdfac"], nums(ln))))
        elif ln.startswith("RPV "):
            cur.setdefault("redprm", {}).update(dict(zip(
                ["laimin", "laimax", "emissmin", "emissmax", "albedomin", "albedomax"], nums(ln))))
        elif ln.startswith("RPZ "):
            cur.setdefault("redprm", {}).update(dict(zip(["z0min", "z0max"], nums(ln))))
        elif ln.startswith("RPNROOT"):
            cur.setdefault("redprm", {})["nroot"] = int(nums(ln)[0])
        elif ln.startswith("RPRTDIS"):
            cur.setdefault("redprm", {})["rtdis"] = nums(ln)
        elif ln.startswith("RESOLVED"):
            cur.setdefault("redprm", {}).update(dict(zip(
                ["alb", "embrd", "xlai", "z0brd"], nums(ln))))
    return cols


def build_columns() -> list[dict]:
    cols = []
    for tag, fname in [("daytime", DAY), ("nighttime", NIGHT)]:
        d = Dataset(str(RUN / fname))
        lat = _v(d, "XLAT"); shdmax = _v(d, "SHDMAX"); shdmin = _v(d, "SHDMIN")
        vegfra = _v(d, "VEGFRA"); tmn = _v(d, "TMN")
        iv = _v(d, "IVGTYP").astype(int); il = _v(d, "ISLTYP").astype(int)
        for cat, (i, j) in VEG_CELLS.items():
            cols.append({
                "name": f"{tag}_veg{cat}", "case": tag, "vegtyp": int(iv[i, j]),
                "soiltyp": int(il[i, j]), "lat": float(np.deg2rad(lat[i, j])),
                "shdfac": float(vegfra[i, j]) / 100.0, "shmin": float(shdmin[i, j]) / 100.0,
                "shmax": float(shdmax[i, j]) / 100.0, "tbot": float(tmn[i, j]),
                "forcing": _column_forcing(d, i, j), "state": _column_state(d, i, j),
            })
        # a wet-soil precip column (daytime grassland with synthetic rain)
        if tag == "daytime":
            i, j = VEG_CELLS[10]
            cols.append({
                "name": "daytime_wetprecip_veg10", "case": "wetprecip",
                "vegtyp": int(iv[i, j]), "soiltyp": int(il[i, j]),
                "lat": float(np.deg2rad(lat[i, j])), "shdfac": float(vegfra[i, j]) / 100.0,
                "shmin": float(shdmin[i, j]) / 100.0, "shmax": float(shdmax[i, j]) / 100.0,
                "tbot": float(tmn[i, j]),
                "forcing": _column_forcing(d, i, j), "state": _column_state(d, i, j, wet=True),
            })
        d.close()
    # Teide snow column (daytime forcing, synthesized snowpack)
    d = Dataset(str(RUN / DAY))
    i, j = TEIDE_CELL
    lat = _v(d, "XLAT"); shdmax = _v(d, "SHDMAX"); shdmin = _v(d, "SHDMIN")
    vegfra = _v(d, "VEGFRA"); tmn = _v(d, "TMN")
    iv = _v(d, "IVGTYP").astype(int); il = _v(d, "ISLTYP").astype(int)
    cols.append({
        "name": "teide_snow", "case": "snow", "vegtyp": int(iv[i, j]),
        "soiltyp": int(il[i, j]), "lat": float(np.deg2rad(lat[i, j])),
        "shdfac": float(vegfra[i, j]) / 100.0, "shmin": float(shdmin[i, j]) / 100.0,
        "shmax": float(shdmax[i, j]) / 100.0, "tbot": float(tmn[i, j]),
        "forcing": _column_forcing(d, i, j), "state": _column_state(d, i, j, snow=True),
    })
    d.close()
    return cols


def main() -> None:
    if not DRIVER.exists():
        raise SystemExit(f"driver not built: {DRIVER} (run oracle/build_driver.sh first)")
    for tbl in ("VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        link = ORACLE / tbl
        if not link.exists():
            link.symlink_to(Path("/home/enric/src/wrf_pristine/WRF/run") / tbl)

    meta = build_columns()
    _write_columns(meta)
    res = subprocess.run([str(DRIVER)], cwd=ORACLE, capture_output=True, text=True)
    if "NOAHCLASSIC_OFFLINE_OK" not in res.stdout:
        raise SystemExit(f"driver failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}")
    parsed = _parse_savepoints()
    assert len(parsed) == len(meta), f"{len(parsed)} parsed != {len(meta)} columns"

    columns = []
    for m, p in zip(meta, parsed):
        columns.append({
            "name": m["name"], "case": m["case"],
            "vegtyp": m["vegtyp"], "soiltyp": m["soiltyp"],
            "lat_rad": m["lat"], "shdfac": m["shdfac"], "shmin": m["shmin"],
            "shmax": m["shmax"], "tbot": m["tbot"], "dt": DT, "dx": DX,
            "slopetyp": SLOPETYP, "isurban": ISURBAN, "isice": ISICE, "iswater": ISWATER,
            "sldpth": SLDPTH, "zsoil": ZSOIL,
            "forcing": m["forcing"], "state_in": m["state"], "wrf": p,
        })

    header = {
        "proof": "noahclassic-savepoints (v0.6.0 lane 14, sf_surface_physics=2)",
        "kind": ("external oracle: compiled pristine WRF module_sf_noahlsm.o SFLX "
                 "called on real Canary d03 land columns; per-column input snapshot "
                 "+ WRF-computed output (NOT a self-compare)"),
        "wrf_source": "/home/enric/src/wrf_pristine/WRF/phys/module_sf_noahlsm.F",
        "corpus": str(RUN), "dataset": DATASET,
        "scope_options": {"local": False, "ua_phys": False, "rdlai2d": False,
                          "usemonalb": False, "opt_thcnd": 1, "fasdas": 0,
                          "ice": 0, "slopetyp": SLOPETYP},
        "nsoil": NSOIL, "ncolumns": len(columns),
        "columns_index": [c["name"] for c in columns],
    }
    out = {**header, "columns": columns}
    (HERE / "savepoints_noahclassic.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "verdict": "NOAHCLASSIC_SAVEPOINTS_BUILT",
        "ncolumns": len(columns),
        "regimes": sorted({c["case"] for c in columns}),
        "sample_daytime_grassland_HFX": columns[2]["wrf"]["flux"]["hfx"],
        "sample_daytime_grassland_TSK": columns[2]["wrf"]["t1_out"],
        "sample_daytime_grassland_GRDFLX": columns[2]["wrf"]["flux"]["grdflx"],
        "sample_snow_SNEQV_out": columns[-1]["wrf"]["snow_out"]["sneqv"],
    }, indent=2))


if __name__ == "__main__":
    main()
