"""Frozen-column pristine NOAHMP_SFLX savepoints (review-b47 A2): WN3 0227 d03 06Z land cells with TSK < TFRZ.

Rebuild: copy proofs/noahmp/build_driver.sh + noahmp_offline_driver.F90 next to this script, run
build_driver.sh (links the compiled pristine WRF Noah-MP objects), then
``JAX_PLATFORMS=cpu PYTHONPATH=src python <dir>/build_frozen_savepoints.py <repo> fixtures/savepoints_energy_frozen.json``.
"""
import importlib.util, json, subprocess, sys
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bns", REPO / "proofs/noahmp/build_noahmp_savepoints.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
b.HERE = HERE; b.DRIVER = HERE / "noahmp_offline_driver.exe"
b.RUN = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
b.JULIAN = 58.25       # WRF: (JULDAY 59 - 1) + 06/24
WRFOUT = "wrfout_d03_2026-02-28_06:00:00"
CELLS = {"frozen_barren16": (46, 45), "frozen_mixedforest5": (50, 54), "frozen_savanna9": (53, 60), "frozen_grass10": (47, 47)}
for tbl in ("MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
    if not (HERE / tbl).exists():
        (HERE / tbl).symlink_to(b.WRF_PRISTINE_ROOT / "run" / tbl)
d = Dataset(str(b.RUN / WRFOUT))
lat, shdmax, vegfra, tmn = b._v(d, "XLAT"), b._v(d, "SHDMAX"), b._v(d, "VEGFRA"), b._v(d, "TMN")
iv, il = b._v(d, "IVGTYP").astype(int), b._v(d, "ISLTYP").astype(int)
meta = []
for name, (i, j) in CELLS.items():
    meta.append({"name": name, "case": "frozen", "vegtyp": int(iv[i, j]), "isltyp": int(il[i, j]),
                 "lat": float(np.deg2rad(lat[i, j])), "shdfac": float(vegfra[i, j]) / 100.0,
                 "shdmax": float(shdmax[i, j]) / 100.0, "tbot": float(tmn[i, j]), "lai0": 0.0, "sai0": 0.0,
                 "forcing": b._column_forcing(d, i, j), "state": b._column_state(d, i, j)})
d.close()
b._write_columns(meta)
res = subprocess.run([str(b.DRIVER)], cwd=HERE, capture_output=True, text=True)
assert "NOAHMP_OFFLINE_OK" in res.stdout, (res.stdout[-2000:], res.stderr[-2000:])
parsed = b._parse_savepoints(); assert len(parsed) == len(meta)
cols = [{"name": m["name"], "case": m["case"], "vegtyp": m["vegtyp"], "isltyp": m["isltyp"], "lat_rad": m["lat"],
         "shdfac": m["shdfac"], "shdmax": m["shdmax"], "tbot": m["tbot"], "dt": b.DT, "dx": b.DX, "julian": b.JULIAN,
         "yearlen": b.YEARLEN, "zsoil": b.ZSOIL, "forcing": m["forcing"], "state_in": m["state"], "wrf": p}
        for m, p in zip(meta, parsed)]
header = {"proof": "noahmp frozen-column savepoints (review-b47 A2)",
          "kind": "external oracle: compiled pristine WRF module_sf_noahmplsm.o NOAHMP_SFLX on real WN3 0227 d03 frozen land columns (TSK < TFRZ, 06Z)",
          "corpus": str(b.RUN / WRFOUT), "dataset": b.DATASET, "julian": b.JULIAN, "ncolumns": len(cols),
          "columns_index": [c["name"] for c in cols]}
energy = {**header, "component": "S1 energy (frozen)",
          "columns": [b._slice(c, ["energy_in", "energy_out", "energy_state", "t2diag", "et", "phen_out", "driver"]) for c in b._proj(cols)]}
OUT.write_text(json.dumps(energy, indent=2, sort_keys=True) + "\n")
for c in energy["columns"]:
    w = c["wrf"]; print(c["name"], "veg", c["vegtyp"], "TG/TV in", round(c["state_in"]["tg"], 1), round(c["state_in"]["tv"], 1), "t2diag", w["t2diag"], "tg out", w["energy_state"].get("tg"), "tv out", w["energy_state"].get("tv"))
