"""NF10: real land columns for the Noah-MP ALBEDO hold (module_sf_noahmpdrv.F:1230-1232) and snow-age oracle
(SNOW_AGE/SNOWALB_CLASS once per step in ALBEDO, module_sf_noahmplsm.F:2925-2945).

Swiss d01 (snow), WN3 0227 d01 and PROD d01: state = CPU-WRF lead-zero history (post-NOAHMP_INIT), ALBEDO entering
step 1 = CPU-WRF lead-zero ALBEDO (landuse_init), forcing = lowest level of the 25 hourly CPU-WRF frames (00Z..24Z,
linear in time; precip = hourly accumulation deltas). 24 h: night -> sunrise -> day -> sunset -> night.

usage: build_columns.py <fixture.json> <records.npz> <driver.exe> <workdir>
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

W = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("bns", W / "proofs/noahmp/build_noahmp_savepoints.py")
bns = importlib.util.module_from_spec(spec); spec.loader.exec_module(bns)
spec = importlib.util.spec_from_file_location("sched", W / "tests/v025/b_noahmp/albedo_schedule.py")
sched = importlib.util.module_from_spec(spec); spec.loader.exec_module(sched)

D5 = "<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case"
CASES = (  # name, CPU-WRF history dir, julian (day of year - 1), dt, dx
    ("swiss", "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu", 14.0, 18.0, 3000.0),
    ("wn3_0227", "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run", 58.0, 54.0, 9000.0),
    ("prod", D5, 206.0, 54.0, 9000.0),
)


def columns(case, cpu, julian, dt, dx):
    files = sorted(f for f in os.listdir(cpu) if f.startswith("wrfout_d01_"))[:25]
    frames = [Dataset(f"{cpu}/{f}") for f in files]
    d0 = frames[0]
    v0 = lambda n: np.asarray(d0[n][0], float)  # noqa: E731
    iv, il = v0("IVGTYP").astype(int), v0("ISLTYP").astype(int)
    ok = (v0("LANDMASK") > 0.5) & ~np.isin(iv, (15, 17))
    rng = np.random.default_rng(10)

    def pick(mask, n):
        idx = np.argwhere(mask)
        return [tuple(int(v) for v in x) for x in idx[rng.choice(len(idx), size=min(n, len(idx)), replace=False)]]
    if case == "swiss":
        cells = [("snow", c) for c in pick(ok & (v0("ISNOW") < 0), 4)] + [("nosnow", c) for c in pick(ok & (v0("SNOW") == 0), 2)]
    else:
        cells = [("land", c) for c in pick(ok, 5)]
    dzs = [float(x) for x in d0["DZS"][0]]
    out = []
    for kind, (i, j) in cells:
        g = lambda n: float(v0(n)[i, j])  # noqa: E731
        lay = lambda n: [float(x) for x in v0(n)[:, i, j]]  # noqa: E731
        hourly = []
        for h, d in enumerate(frames):
            f = bns._column_forcing(d, i, j)
            if h + 1 < len(frames):
                acc = lambda n: (float(frames[h + 1][n][0][i, j]) - float(d[n][0][i, j])) / 3600.0  # noqa: E731
                f.update(prcpconv=acc("RAINC"), prcpnonc=acc("RAINNC"))
            hourly.append([float(f[k]) for k in sched.FIELDS])
        f0 = bns._column_forcing(d0, i, j)
        state = {"stc": lay("TSNO") + lay("TSLB"), "smc": lay("SMOIS"), "sh2o": lay("SH2O"),
                 "tv": g("TV"), "tg": g("TG"), "tah": g("TAH"), "eah": g("EAH"), "canliq": g("CANLIQ"), "canice": g("CANICE"),
                 "fwet": g("FWET"), "qsfc": f0["q2"], "snowh": g("SNOWH"), "sneqv": g("SNOW"), "sneqvo": g("SNEQVO"),
                 "albold": g("ALBOLD"), "tauss": g("TAUSS"), "isnow": int(g("ISNOW")), "zsnso": lay("ZSNSO"),
                 "snice": lay("SNICE"), "snliq": lay("SNLIQ"), "cm": g("CM"), "ch": g("CH"), "smcwtd": g("SMCWTD")}
        out.append({"name": f"{case}_{kind}_{i}_{j}", "case": case, "vegtyp": int(iv[i, j]), "isltyp": int(il[i, j]),
                    "lat_rad": float(np.deg2rad(v0("XLAT")[i, j])), "julian": julian, "yearlen": 365, "dt": dt, "dx": dx,
                    "zsoil": [float(z) for z in -np.cumsum(dzs)], "dzs": dzs, "shdfac": g("VEGFRA") / 100.0,
                    "shdmax": g("SHDMAX") / 100.0, "tbot": g("TMN"), "lai0": g("LAI"), "sai0": g("XSAI"),
                    "zlvl": f0["zlvl"], "dz8w": f0["dz8w"], "albedo0": g("ALBEDO"), "hourly": hourly,
                    "forcing": {k: hourly[0][n] for n, k in enumerate(sched.FIELDS)} | {"zlvl": f0["zlvl"]}, "state_in": state})
    for d in frames:
        d.close()
    return out


def run_driver(cols, nsteps, exe, work):
    """noahmp_columns.in + noahmp_sched.in for the lane driver; returns (SALB, ALBEDO) per column and step."""
    work.mkdir(parents=True, exist_ok=True)
    for t in ("MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        shutil.copy(Path(exe).parent / t, work / t)
    c0 = cols[0]
    L = [bns.DATASET, f"{len(cols)} {bns.SLOPETYP} {bns.SOILCOLOR} {c0['dt']}", " ".join(str(z) for z in c0["zsoil"])]
    S = []
    for c in cols:
        f, s = c["forcing"], c["state_in"]
        L += [f"{c['vegtyp']} {c['isltyp']}",
              f"{c['lat_rad']} {c['julian']} {c['yearlen']} {f['cosz']} {c['dx']} {c['dz8w']} {c['zlvl']}",
              f"{c['shdfac']} {c['shdmax']} {c['tbot']}",
              " ".join(str(f[k]) for k in ("sfctmp", "sfcprs", "psfc", "uu", "vv", "q2", "qc", "soldn", "lwdn")),
              " ".join(str(f[k]) for k in ("prcpconv", "prcpnonc", "prcpsnow", "prcpgrpl", "prcphail")),
              " ".join(str(x) for x in s["stc"]), " ".join(str(x) for x in s["smc"]), " ".join(str(x) for x in s["sh2o"]),
              " ".join(str(s[k]) for k in ("tv", "tg", "tah", "eah", "canliq", "canice", "fwet", "qsfc")),
              " ".join(str(x) for x in (c["lai0"], c["sai0"], s["snowh"], s["sneqv"], s["sneqvo"], s["albold"], s["tauss"])),
              f"{s['isnow']}", " ".join(str(x) for x in s["zsnso"]), " ".join(str(x) for x in s["snice"]),
              " ".join(str(x) for x in s["snliq"]), " ".join(str(s[k]) for k in ("cm", "ch", "smcwtd"))]
        S.append("%.9e" % np.float32(c["albedo0"]))
        S += [" ".join("%.9e" % v for v in row) for row in sched.step_forcing(c["hourly"], c["dt"], nsteps)]
    (work / "noahmp_columns.in").write_text("\n".join(L) + "\n")
    (work / "noahmp_sched.in").write_text("\n".join(S) + "\n")
    env = dict(os.environ, NOAHMP_TABLES=str(work), NOAHMP_WIRE_SOIL="1", NOAHMP_WIRE_FICEOLD="1",
               NOAHMP_NSTEPS=str(nsteps), NOAHMP_SCHED="1")
    subprocess.run([str(exe)], cwd=work, env=env, check=True, capture_output=True, timeout=3000)
    rec = np.loadtxt(work / "noahmp_albedo.out", usecols=(1, 2, 3, 4, 7, 8), dtype=np.float64)
    out = np.zeros((4, len(cols), nsteps), np.float32)   # SALB, ALBEDO, TAUSS, ALBOLD after each step
    out[:, rec[:, 0].astype(int) - 1, rec[:, 1].astype(int) - 1] = rec[:, 2:].T
    return out


def main():
    fixture, records, exe, work = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    fx, rec = {}, {}
    for case, cpu, julian, dt, dx in CASES:
        cols = columns(case, cpu, julian, dt, dx)
        nsteps = int(round(24 * 3600 / dt))
        salb, alb, tauss, albold = run_driver(cols, nsteps, exe, work / case)
        fx[case] = {"source": cpu, "nsteps": nsteps, "columns": cols}
        rec[f"{case}_salb"], rec[f"{case}_albedo"], rec[f"{case}_tauss"], rec[f"{case}_albold"] = salb, alb, tauss, albold
        night = salb < -999
        print(case, nsteps, "steps", [c["name"] for c in cols], "night steps/col", night.sum(1).tolist(),
              "sentinel in ALBEDO", int((alb < -999).sum()), "max TAUSS", float(tauss.max()))
    with Dataset(f"{CASES[0][1]}/wrfout_d01_2023-01-15_00:00:00") as d0:   # Swiss lead-zero ALBEDO (seed oracle)
        rec["swiss_t0_albedo"] = np.asarray(d0["ALBEDO"][0], np.float32)
    Path(fixture).write_text(json.dumps(fx, indent=0))
    np.savez_compressed(records, **rec)


if __name__ == "__main__":
    main()
