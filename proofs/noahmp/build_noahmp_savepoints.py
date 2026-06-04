"""Build WRF Noah-MP per-component savepoint fixtures (v0.2.0 P0-3, Sprint 0b).

EXTERNAL ORACLE — NOT a self-compare. This harness mirrors
``proofs/v010_validation/sfclay_mynn_full_parity.py``:

  1. extracts representative Canary LAND columns (daytime + nighttime per veg
     category + a Teide snow column) from the corpus d03 wrfout/wrfinput;
  2. writes the column forcing + cold-start prognostic land state into the
     Fortran driver's ``noahmp_columns.in``;
  3. runs ``noahmp_offline_driver.exe`` — which links the COMPILED pristine WRF
     ``module_sf_noahmplsm.o`` and calls the exact ``NOAHMP_SFLX`` orchestrator
     on each column under the FROZEN active-option set (dveg=4, opt_run=3
     Schaake, opt_sfc=1, opt_alb=2, opt_tbot=2, opt_stc=1, ...);
  4. parses the driver's per-column output and writes the canonical per-component
     savepoint JSON files the component sprints validate against:
       proofs/noahmp/savepoints_energy.json     (S1 — the HFX-fix component)
       proofs/noahmp/savepoints_soil_thermo.json (S2 — semi-implicit STC)
       proofs/noahmp/savepoints_phenology.json   (S5 — table LAI/SAI/FVEG)
       proofs/noahmp/savepoints_snow.json         (S3 — snow water/aging)
       proofs/noahmp/savepoints_water.json        (S4 — Schaake soil hydrology)
     plus a combined proofs/noahmp/savepoints_all.json index.

Each savepoint carries the full per-column INPUT snapshot + the WRF-computed
OUTPUT, so a component sprint can feed the input through its JAX port and assert
field-wise agreement against the WRF output with NO dependency on the other
components (the upstream inputs are frozen fixtures).

Run (CPU only; cores 0-3):
    cd proofs/noahmp && ./build_driver.sh          # one-time Fortran build
    taskset -c 0-3 python3 build_noahmp_savepoints.py
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
RUN = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
DAY = "wrfout_d03_2026-05-22_12:00:00"
NIGHT = "wrfout_d03_2026-05-22_03:00:00"
DRIVER = HERE / "noahmp_offline_driver.exe"

NSOIL, NSNOW = 4, 2  # ISNOW snow layers used in the cold/Teide columns (<=NSNOW=3)
G = 9.80665
DATASET = "MODIFIED_IGBP_MODIS_NOAH"
ZSOIL = [-0.05, -0.25, -0.70, -1.50]   # corpus ZS (interface depths, <0)
DT = 90.0      # physics step used for the savepoint (representative long step)
DX = 1000.0    # d03 grid spacing
JULIAN = 141.5 + 0.5  # JULDAY 141 (2026-05-21) + ~midday fraction (corpus)
YEARLEN = 365
SLOPETYP = 1
SOILCOLOR = 4   # default WRF soil color (drv uses ISC=4)

# Representative Canary land columns (cell indices verified in the d03 wrfout):
#   veg 5  Mixed Forests, 9 Savannas, 10 Grasslands, 13 Urban->land, 16 Barren.
VEG_CELLS = {5: (44, 23), 9: (32, 54), 10: (31, 50), 13: (48, 41), 16: (30, 36)}
TEIDE_CELL = (34, 34)  # max-terrain barren column (HGT ~3462 m)


def _v(d, name):
    return np.asarray(d.variables[name][0], dtype=np.float64)


def _column_forcing(d, i, j):
    """Lowest-level atmospheric + radiation + precip forcing at cell (i, j)."""
    T = _v(d, "T"); P = _v(d, "P"); PB = _v(d, "PB")
    PH = _v(d, "PH"); PHB = _v(d, "PHB"); QV = _v(d, "QVAPOR")
    U = _v(d, "U"); V = _v(d, "V")
    theta = T[0, i, j] + 300.0
    p = P[0, i, j] + PB[0, i, j]
    rovcp = 287.0 / 1004.0
    sfctmp = theta * (p / 1.0e5) ** rovcp     # air T at lowest level
    qv = max(float(QV[0, i, j]), 1e-6)
    dz = float(((PH + PHB)[1, i, j] - (PH + PHB)[0, i, j]) / G)
    uu = 0.5 * (U[0, i, j] + U[0, i, j + 1])
    vv = 0.5 * (V[0, i, j] + V[0, i + 1, j])
    return {
        "sfctmp": float(sfctmp), "sfcprs": float(p), "psfc": float(_v(d, "PSFC")[i, j]),
        "uu": float(uu), "vv": float(vv), "q2": qv, "qc": 0.0,
        "soldn": float(_v(d, "SWDOWN")[i, j]), "lwdn": float(_v(d, "GLW")[i, j]),
        "cosz": max(float(_v(d, "COSZEN")[i, j]), 0.0),
        "zlvl": 0.5 * dz, "dz8w": dz,
        "prcpconv": 0.0, "prcpnonc": 0.0, "prcpsnow": 0.0,
        "prcpgrpl": 0.0, "prcphail": 0.0,
    }


def _column_state(d, i, j, *, snow=False):
    """Cold-start prognostic land state at cell (i, j) from the corpus soil state."""
    tslb = _v(d, "TSLB")[:, i, j]
    smois = _v(d, "SMOIS")[:, i, j]
    sh2o = _v(d, "SH2O")[:, i, j]
    tsk = float(_v(d, "TSK")[i, j])
    sfc = _v(d, "T2")[i, j]
    # snow+soil temperature column STC(-NSNOW+1:NSOIL): snow slots 0 unless `snow`.
    stc = [0.0] * 3 + [float(x) for x in tslb]           # len 7 (NSNOW=3 + NSOIL=4)
    isnow = 0
    snowh = 0.0
    sneqv = 0.0
    zsnso = [0.0, 0.0, 0.0] + list(ZSOIL)                # len 7
    snice = [0.0, 0.0, 0.0]
    snliq = [0.0, 0.0, 0.0]
    albold = 0.65
    if snow:
        # single active snow layer (ISNOW=-1): a faithful WRF column will advance
        # it; the answer is the WRF Fortran's, so this is a real oracle column.
        isnow = -1
        snowh = 0.30
        sneqv = 75.0       # SWE [mm]
        stc[2] = 268.0     # snow-layer temperature (slot index -> -1)
        snice[2] = 70.0
        snliq[2] = 5.0
        zsnso = [0.0, 0.0, -snowh] + list(ZSOIL)
        albold = 0.75
    return {
        "stc": stc, "smc": [float(x) for x in smois], "sh2o": [float(x) for x in sh2o],
        "tv": float(sfc), "tg": tsk, "tah": float(sfc), "eah": 1500.0,
        "canliq": 0.0, "canice": 0.0, "fwet": 0.0, "qsfc": 0.012,
        "snowh": snowh, "sneqv": sneqv, "sneqvo": sneqv, "albold": albold,
        "tauss": 0.0, "isnow": isnow, "zsnso": zsnso, "snice": snice, "snliq": snliq,
        "cm": 0.001, "ch": 0.001, "smcwtd": float(smois[-1]),
    }


def _write_columns(cols: list[dict]) -> None:
    L = []
    L.append(DATASET)
    L.append(f"{len(cols)} {SLOPETYP} {SOILCOLOR} {DT}")
    L.append(" ".join(f"{z}" for z in ZSOIL))
    for c in cols:
        f, s = c["forcing"], c["state"]
        L.append(f"{c['vegtyp']} {c['isltyp']}")
        L.append(f"{c['lat']} {JULIAN} {YEARLEN} {f['cosz']} {DX} {f['dz8w']} {f['zlvl']}")
        L.append(f"{c['shdfac']} {c['shdmax']} {c['tbot']}")
        L.append(" ".join(str(x) for x in [f["sfctmp"], f["sfcprs"], f["psfc"],
                  f["uu"], f["vv"], f["q2"], f["qc"], f["soldn"], f["lwdn"]]))
        L.append(" ".join(str(x) for x in [f["prcpconv"], f["prcpnonc"],
                  f["prcpsnow"], f["prcpgrpl"], f["prcphail"]]))
        L.append(" ".join(str(x) for x in s["stc"]))
        L.append(" ".join(str(x) for x in s["smc"]))
        L.append(" ".join(str(x) for x in s["sh2o"]))
        L.append(" ".join(str(x) for x in [s["tv"], s["tg"], s["tah"], s["eah"],
                  s["canliq"], s["canice"], s["fwet"], s["qsfc"]]))
        L.append(" ".join(str(x) for x in [c["lai0"], c["sai0"], s["snowh"],
                  s["sneqv"], s["sneqvo"], s["albold"], s["tauss"]]))
        L.append(f"{s['isnow']}")
        L.append(" ".join(str(x) for x in s["zsnso"]))
        L.append(" ".join(str(x) for x in s["snice"]))
        L.append(" ".join(str(x) for x in s["snliq"]))
        L.append(" ".join(str(x) for x in [s["cm"], s["ch"], s["smcwtd"]]))
    (HERE / "noahmp_columns.in").write_text("\n".join(L) + "\n")


def _parse_savepoints() -> list[dict]:
    """Parse noahmp_savepoints.out into one dict per column."""
    text = (HERE / "noahmp_savepoints.out").read_text().splitlines()
    cols, cur = [], None

    def nums(line: str) -> list[float]:
        out = []
        for tok in line.split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
        return out

    for ln in text:
        if ln.startswith("COL "):
            cur = {}
        elif ln.startswith("ENDCOL"):
            cols.append(cur); cur = None
        elif cur is None:
            continue
        elif ln.startswith("CAT "):
            v = nums(ln); cur["vegtyp"], cur["isltyp"] = int(v[0]), int(v[1])
        elif ln.startswith("FORCING2"):
            v = nums(ln)
            cur["cosz"], cur["julian"], cur["shdfac"], cur["shdmax"], cur["tbot"] = v[:5]
        elif ln.startswith("FORCING "):
            cur["forcing"] = nums(ln)
        elif ln.startswith("PHEN_OUT"):
            v = nums(ln); cur["phen_out"] = {"lai": v[0], "sai": v[1], "fveg": v[2]}
        elif ln.startswith("PHEN_IN"):
            v = nums(ln); cur["phen_in"] = {"lai": v[0], "sai": v[1]}
        elif ln.startswith("ENERGY_IN"):
            v = nums(ln)
            cur["energy_in"] = {"tv": v[0], "tg": v[1], "tah": v[2], "eah": v[3]}
        elif ln.startswith("ENERGY_OUT"):
            v = nums(ln)
            cur["energy_out"] = dict(zip(
                ["fsh", "fcev", "fgev", "fctr", "ssoil", "fira", "trad",
                 "emissi", "z0wrf", "chv", "chb", "sav", "sag"], v))
        elif ln.startswith("ENERGY_STATE"):
            v = nums(ln)
            cur["energy_state"] = dict(zip(
                ["tv", "tg", "tah", "eah", "albedo", "fsno", "fsa"], v))
        elif ln.startswith("T2DIAG"):
            v = nums(ln)
            cur["t2diag"] = dict(zip(["t2mv", "t2mb", "t2m", "q2v", "q2b"], v))
        elif ln.startswith("ET "):
            v = nums(ln)
            cur["et"] = dict(zip(["ecan", "etran", "edir", "qsnow", "qmelt"], v))
        elif ln.startswith("STC_IN"):
            cur["stc_in"] = nums(ln)
        elif ln.startswith("STC_OUT"):
            cur["stc_out"] = nums(ln)
        elif ln.startswith("HCPCT"):
            cur["hcpct"] = nums(ln)
        elif ln.startswith("SNOW_ISNOW"):
            v = nums(ln); cur["isnow_in"], cur["isnow_out"] = int(v[0]), int(v[1])
        elif ln.startswith("SNOW_IN"):
            v = nums(ln)
            cur["snow_in"] = dict(zip(["snowh", "sneqv", "sneqvo", "albold"], v))
        elif ln.startswith("SNOW_OUT"):
            v = nums(ln)
            cur["snow_out"] = dict(zip(["snowh", "sneqv", "qsnbot", "fsno", "albold"], v))
        elif ln.startswith("ZSNSO_OUT"):
            cur["zsnso_out"] = nums(ln)
        elif ln.startswith("SNICE_OUT"):
            cur["snice_out"] = nums(ln)
        elif ln.startswith("SNLIQ_OUT"):
            cur["snliq_out"] = nums(ln)
        elif ln.startswith("SMC_IN"):
            cur["smc_in"] = nums(ln)
        elif ln.startswith("SMC_OUT"):
            cur["smc_out"] = nums(ln)
        elif ln.startswith("SH2O_IN"):
            cur["sh2o_in"] = nums(ln)
        elif ln.startswith("SH2O_OUT"):
            cur["sh2o_out"] = nums(ln)
        elif ln.startswith("WATER_OUT"):
            v = nums(ln)
            cur["water_out"] = dict(zip(
                ["runsrf", "runsub", "smcwtd", "canliq", "canice"], v))
        elif ln.startswith("DRIVER"):
            v = nums(ln)
            cur["driver"] = dict(zip(["hfx", "lh", "qfx", "grdflx", "tsk"], v))
    return cols


def build_columns() -> list[dict]:
    cols = []
    for tag, fname in [("daytime", DAY), ("nighttime", NIGHT)]:
        d = Dataset(str(RUN / fname))
        lat = _v(d, "XLAT"); shdmax = _v(d, "SHDMAX"); vegfra = _v(d, "VEGFRA")
        tmn = _v(d, "TMN"); iv = _v(d, "IVGTYP").astype(int); il = _v(d, "ISLTYP").astype(int)
        for cat, (i, j) in VEG_CELLS.items():
            shdfac = float(vegfra[i, j]) / 100.0   # SHDFAC = VEGFRA/100 (dveg=4 FVEG)
            cols.append({
                "name": f"{tag}_veg{cat}", "case": tag, "vegtyp": int(iv[i, j]),
                "isltyp": int(il[i, j]), "lat": float(np.deg2rad(lat[i, j])),
                "shdfac": shdfac, "shdmax": float(shdmax[i, j]) / 100.0,
                "tbot": float(tmn[i, j]), "lai0": 0.0, "sai0": 0.0,
                "forcing": _column_forcing(d, i, j), "state": _column_state(d, i, j),
            })
        d.close()
    # Teide snow column (daytime forcing, synthesized snowpack -> exercises S2/S3)
    d = Dataset(str(RUN / DAY))
    i, j = TEIDE_CELL
    lat = _v(d, "XLAT"); shdmax = _v(d, "SHDMAX"); vegfra = _v(d, "VEGFRA")
    tmn = _v(d, "TMN"); iv = _v(d, "IVGTYP").astype(int); il = _v(d, "ISLTYP").astype(int)
    cols.append({
        "name": "teide_snow", "case": "snow", "vegtyp": int(iv[i, j]),
        "isltyp": int(il[i, j]), "lat": float(np.deg2rad(lat[i, j])),
        "shdfac": float(vegfra[i, j]) / 100.0, "shdmax": float(shdmax[i, j]) / 100.0,
        "tbot": float(tmn[i, j]), "lai0": 0.0, "sai0": 0.0,
        "forcing": _column_forcing(d, i, j), "state": _column_state(d, i, j, snow=True),
    })
    d.close()
    return cols


def main() -> None:
    if not DRIVER.exists():
        raise SystemExit(f"driver not built: {DRIVER} (run build_driver.sh first)")
    for tbl in ("MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        link = HERE / tbl
        if not link.exists():
            link.symlink_to(Path("/home/enric/src/wrf_pristine/WRF/run") / tbl)

    meta = build_columns()
    _write_columns(meta)
    res = subprocess.run([str(DRIVER)], cwd=HERE, capture_output=True, text=True)
    if "NOAHMP_OFFLINE_OK" not in res.stdout:
        raise SystemExit(f"driver failed:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}")
    parsed = _parse_savepoints()
    assert len(parsed) == len(meta), f"{len(parsed)} parsed != {len(meta)} columns"

    columns = []
    for m, p in zip(meta, parsed):
        columns.append({
            "name": m["name"], "case": m["case"],
            "vegtyp": m["vegtyp"], "isltyp": m["isltyp"],
            "lat_rad": m["lat"], "shdfac": m["shdfac"], "shdmax": m["shdmax"],
            "tbot": m["tbot"], "dt": DT, "dx": DX, "julian": JULIAN,
            "yearlen": YEARLEN, "zsoil": ZSOIL,
            "forcing": m["forcing"], "state_in": m["state"], "wrf": p,
        })

    header = {
        "proof": "noahmp-component-savepoints (v0.2.0 P0-3, Sprint 0b)",
        "kind": ("external oracle: compiled pristine WRF module_sf_noahmplsm.o "
                 "NOAHMP_SFLX called on real Canary d03 land columns; per-column "
                 "input snapshot + WRF-computed output (NOT a self-compare)"),
        "wrf_source": "/home/enric/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F",
        "corpus": str(RUN), "dataset": DATASET,
        "scope_options": {"dveg": 4, "opt_crs": 1, "opt_btr": 1, "opt_run": 3,
                          "opt_sfc": 1, "opt_frz": 1, "opt_inf": 1, "opt_rad": 3,
                          "opt_alb": 2, "opt_snf": 1, "opt_tbot": 2, "opt_stc": 1},
        "nsoil": NSOIL, "nsnow": 3, "ncolumns": len(columns),
        "columns_index": [c["name"] for c in columns],
    }

    # combined index + per-component slices the sprints load as fixtures
    _dump("savepoints_all.json", {**header, "columns": columns})
    _dump("savepoints_phenology.json", {**header, "component": "S5 phenology",
          "columns": [_slice(c, ["phen_in", "phen_out"]) for c in _proj(columns)]})
    _dump("savepoints_energy.json", {**header, "component": "S1 energy (HFX fix)",
          "columns": [_slice(c, ["energy_in", "energy_out", "energy_state",
                                 "t2diag", "et", "phen_out", "driver"])
                      for c in _proj(columns)]})
    _dump("savepoints_soil_thermo.json", {**header, "component": "S2 soil_thermo",
          "columns": [_slice(c, ["stc_in", "stc_out", "hcpct"]) for c in _proj(columns)]})
    _dump("savepoints_snow.json", {**header, "component": "S3 snow",
          "columns": [_slice(c, ["isnow_in", "isnow_out", "snow_in", "snow_out",
                                 "zsnso_out", "snice_out", "snliq_out"])
                      for c in _proj(columns)]})
    _dump("savepoints_water.json", {**header, "component": "S4 water (Schaake)",
          "columns": [_slice(c, ["smc_in", "smc_out", "sh2o_in", "sh2o_out",
                                 "water_out", "et"]) for c in _proj(columns)]})

    print(json.dumps({
        "verdict": "NOAHMP_SAVEPOINTS_BUILT",
        "ncolumns": len(columns),
        "components": ["energy", "soil_thermo", "phenology", "snow", "water"],
        "sample_daytime_grassland_HFX": columns[2]["wrf"]["driver"]["hfx"],
        "sample_daytime_grassland_TAH": columns[2]["wrf"]["energy_state"]["tah"],
        "sample_daytime_grassland_TRAD": columns[2]["wrf"]["energy_out"]["trad"],
    }, indent=2))


def _proj(columns):
    """Expose name/case/forcing/state alongside the WRF dict for per-component files."""
    for c in columns:
        yield {
            "name": c["name"], "case": c["case"], "vegtyp": c["vegtyp"],
            "isltyp": c["isltyp"], "shdfac": c["shdfac"], "julian": c["julian"],
            "forcing": c["forcing"], "state_in": c["state_in"], "wrf": c["wrf"],
        }


def _slice(c, keys):
    out = {k: c[k] for k in ("name", "case", "vegtyp", "isltyp", "shdfac",
                             "julian", "forcing", "state_in")}
    out["wrf"] = {k: c["wrf"][k] for k in keys if k in c["wrf"]}
    return out


def _dump(fname, obj):
    (HERE / fname).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
