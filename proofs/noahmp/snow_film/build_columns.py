"""NF12: real frost/rain land columns for the Noah-MP snow-film reset (module_sf_noahmplsm.F:1067-1070) and the
QSNOWXY/QRAINXY ground rates (PRECIP_HEAT QSNOW/QRAIN, :1548; driver module_sf_noahmpdrv.F:1258).

* teide_0120: WN3 0120 d02 Teide cells, CPU-WRF state at 2026-01-23_18, 6 h: canopy frost from 20Z, ICEDRIP
  unloading -> QSNOW > 0 with no precipitation, the per-step film is zeroed by :1067 (CPU-WRF SNOW == 0).
* rain_0227: WN3 0227 d03 cells (NF08), CPU-WRF state at 2026-02-28_12, 16 h: afternoon rain (QRAIN), night frost.
Forcing = lowest level of the hourly CPU-WRF frames (albedo_schedule.step_forcing), precip = hourly deltas.

usage: build_columns.py <fixture.json> <records.npz> <driver.exe> <workdir>
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

W = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ahb", W / "proofs/noahmp/albedo_hold/build_columns.py")
ahb = importlib.util.module_from_spec(spec); spec.loader.exec_module(ahb)
bns, sched = ahb.bns, ahb.sched

CASES = (  # name, CPU-WRF history dir, domain, start frame, hours, cells (j, i), julian, dt, dx
    ("teide_0120", "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260120_18z_a1/run/run", "d02", "2026-01-23_18", 6,
     [(51, 106), (51, 107), (51, 108), (51, 109), (52, 107), (69, 66)], 22.0, 18.0, 3000.0),
    ("rain_0227", "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run", "d03", "2026-02-28_12", 16,
     [(51, 51), (51, 52), (52, 50), (53, 53), (53, 55)], 58.0, 6.0, 1000.0),
)
EXTRA = {"qsnow": 9, "qrain": 10, "sneqv": 11, "snowh": 12}   # noahmp_albedo.out columns (0-based)


def columns(case, cpu, dom, start, hours, cells, julian, dt, dx):
    keyed = {f[11:24]: f for f in os.listdir(cpu) if f.startswith(f"wrfout_{dom}_")}
    keys = sorted(k for k in keyed if k >= start)[:hours + 1]
    frames = [Dataset(f"{cpu}/{keyed[k]}") for k in keys]
    d0 = frames[0]
    v0 = lambda n: np.asarray(d0[n][0], float)  # noqa: E731
    iv, il = v0("IVGTYP").astype(int), v0("ISLTYP").astype(int)
    dzs = [float(x) for x in d0["DZS"][0]]
    out = []
    for i_, (j, i) in enumerate(cells):
        assert v0("LANDMASK")[j, i] > 0.5 and iv[j, i] not in (15, 17), (case, j, i)
        g = lambda n: float(v0(n)[j, i])  # noqa: E731
        lay = lambda n: [float(x) for x in v0(n)[:, j, i]]  # noqa: E731
        hourly = []
        for h, d in enumerate(frames):
            f = bns._column_forcing(d, j, i)
            if h + 1 < len(frames):
                acc = lambda n: (float(frames[h + 1][n][0][j, i]) - float(d[n][0][j, i])) / 3600.0  # noqa: E731
                f.update(prcpconv=acc("RAINC"), prcpnonc=acc("RAINNC"))
            hourly.append([float(f[k]) for k in sched.FIELDS])
        f0 = bns._column_forcing(d0, j, i)
        state = {"stc": lay("TSNO") + lay("TSLB"), "smc": lay("SMOIS"), "sh2o": lay("SH2O"),
                 "tv": g("TV"), "tg": g("TG"), "tah": g("TAH"), "eah": g("EAH"), "canliq": g("CANLIQ"), "canice": g("CANICE"),
                 "fwet": g("FWET"), "qsfc": f0["q2"], "snowh": g("SNOWH"), "sneqv": g("SNOW"), "sneqvo": g("SNEQVO"),
                 "albold": g("ALBOLD"), "tauss": g("TAUSS"), "isnow": int(g("ISNOW")), "zsnso": lay("ZSNSO"),
                 "snice": lay("SNICE"), "snliq": lay("SNLIQ"), "cm": g("CM"), "ch": g("CH"), "smcwtd": g("SMCWTD")}
        out.append({"name": f"{case}_{j}_{i}", "case": case, "vegtyp": int(iv[j, i]), "isltyp": int(il[j, i]),
                    "lat_rad": float(np.deg2rad(v0("XLAT")[j, i])), "julian": julian, "yearlen": 365, "dt": dt, "dx": dx,
                    "zsoil": [float(z) for z in -np.cumsum(dzs)], "dzs": dzs, "shdfac": g("VEGFRA") / 100.0,
                    "shdmax": g("SHDMAX") / 100.0, "tbot": g("TMN"), "lai0": g("LAI"), "sai0": g("XSAI"),
                    "zlvl": f0["zlvl"], "dz8w": f0["dz8w"], "albedo0": g("ALBEDO"), "hourly": hourly,
                    "forcing": {k: hourly[0][n] for n, k in enumerate(sched.FIELDS)} | {"zlvl": f0["zlvl"]}, "state_in": state,
                    "cpu_hourly": {v: [float(np.asarray(d[v][0])[j, i]) for d in frames] for v in ("QSNOWXY", "QRAINXY", "SNOW", "CANICE")}})
    for d in frames:
        d.close()
    return out


def main():
    fixture, records, exe, work = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    fx, rec = {}, {}
    for case, cpu, dom, start, hours, cells, julian, dt, dx in CASES:
        cols = columns(case, cpu, dom, start, hours, cells, julian, dt, dx)
        nsteps = int(round(hours * 3600 / dt))
        salb, alb, tauss, albold = ahb.run_driver(cols, nsteps, exe, work / case)
        raw = np.loadtxt(work / case / "noahmp_albedo.out", usecols=(1, 2, *EXTRA.values()), dtype=np.float64)
        fx[case] = {"source": f"{cpu} {dom} {start} +{hours} h", "nsteps": nsteps, "columns": cols}
        rec[f"{case}_salb"], rec[f"{case}_albedo"], rec[f"{case}_tauss"], rec[f"{case}_albold"] = salb, alb, tauss, albold
        for n, key in enumerate(EXTRA):
            a = np.zeros((len(cols), nsteps), np.float32)
            a[raw[:, 0].astype(int) - 1, raw[:, 1].astype(int) - 1] = raw[:, 2 + n]
            rec[f"{case}_{key}"] = a
        print(case, nsteps, "steps", [c["name"] for c in cols], "max qsnow %.2e qrain %.2e" % (rec[f"{case}_qsnow"].max(), rec[f"{case}_qrain"].max()),
              "sneqv>0 steps", int((rec[f"{case}_sneqv"] > 0).sum()), "max sneqv %.2e" % rec[f"{case}_sneqv"].max(),
              "steps with qsnow>0 and sneqv==0", int(((rec[f"{case}_qsnow"] > 0) & (rec[f"{case}_sneqv"] == 0)).sum()))
    Path(fixture).write_text(json.dumps(fx, indent=0))
    np.savez_compressed(records, **rec)


if __name__ == "__main__":
    main()
