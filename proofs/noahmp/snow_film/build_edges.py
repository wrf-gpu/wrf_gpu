"""NF12 review-b (2): pack-edge columns for the NOAHMP_SFLX snow-film reset (module_sf_noahmplsm.F:1067-1070).

Real-structured (E95/E186): real WN3 0227 d03 land states and hourly CPU-WRF forcing (snow_film.build_columns), with
synthetic weather where the real case has none:
* onset_0227: SFCTMP 271 K (Jordan FPICE = 1) and non-convective precipitation ramped per hour 1e-7 .. 3e-4 mm/s at
  dt 6 s. WRF zeroes the film every step while one step's SNOWH stays <= 1e-6 m (light snowfall never accumulates),
  then the pack starts (0 -> snow transition).
* melt_0227: the real afternoon forcing from 2026-02-28_12 with a thin no-layer pack (SNEQV 2 mm, SNOWH 0.02 m,
  ISNOW 0) on a melting-point thermal state (TSLB(1) 273.16 K, TG 273 K) that melts/sublimates out through the 1e-6 thresholds (snow -> 0 transition).

usage: build_edges.py <fixture.json> <records.npz> <driver.exe> <workdir>
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("sfb", W / "proofs/noahmp/snow_film/build_columns.py")
sfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sfb)
ahb, sched = sfb.ahb, sfb.sched
F = sched.FIELDS
RAMP = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)   # mm/s, one per hour


def base():
    case = [c for c in sfb.CASES if c[0] == "rain_0227"][0]
    return sfb.columns(*case[:4], len(RAMP), *case[5:])


def onset(cols):
    out = []
    for c in cols:
        c = copy.deepcopy(c)
        c["name"] = c["name"].replace("rain_0227", "onset_0227")
        for h, row in enumerate(c["hourly"]):
            row[F.index("sfctmp")] = 271.0
            row[F.index("prcpconv")] = 0.0
            row[F.index("prcpnonc")] = RAMP[min(h, len(RAMP) - 1)]
        c["forcing"] = {k: c["hourly"][0][n] for n, k in enumerate(F)} | {"zlvl": c["forcing"]["zlvl"]}
        out.append(c)
    return out


def melt(cols):
    out = []
    for c in cols:
        c = copy.deepcopy(c)
        c["name"] = c["name"].replace("rain_0227", "melt_0227")
        s = c["state_in"]
        # Thermal state at the melting point: on ~285 K soil WRF's no-layer PHASECHANGE melts the pack in one step from
        # the top-soil excess heat; at TFRZ it melts out gradually under the real afternoon fluxes.
        s.update(sneqv=2.0, snowh=0.02, sneqvo=2.0, isnow=0, snice=[0.0, 0.0, 0.0], snliq=[0.0, 0.0, 0.0],
                 stc=[273.0, 273.0, 273.0, 273.16, 274.0, 276.0, 280.0], tg=273.0, tv=273.5, tah=273.5)
        out.append(c)
    return out


def main():
    fixture, records, exe, work = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    cols = base()
    fx, rec = {}, {}
    for case, cs, hours in (("onset_0227", onset(cols), len(RAMP)), ("melt_0227", melt(cols), 6)):
        nsteps = int(round(hours * 3600 / cs[0]["dt"]))
        ahb.run_driver(cs, nsteps, exe, work / case)
        raw = np.loadtxt(work / case / "noahmp_albedo.out", usecols=(1, 2, *sfb.EXTRA.values()), dtype=np.float64)
        fx[case] = {"source": f"real-structured: WN3 0227 d03 states, {case}", "nsteps": nsteps, "columns": cs}
        for n, key in enumerate(sfb.EXTRA):
            a = np.zeros((len(cs), nsteps), np.float32)
            a[raw[:, 0].astype(int) - 1, raw[:, 1].astype(int) - 1] = raw[:, 2 + n]
            rec[f"{case}_{key}"] = a
        sw, qs = rec[f"{case}_sneqv"], rec[f"{case}_qsnow"]
        z = np.concatenate([np.array([[c["state_in"]["sneqv"]] for c in cs]) == 0, sw == 0], 1)
        print(case, nsteps, "steps | film-reset steps (qsnow>0, SNEQV 0)", int(((qs > 0) & (sw == 0)).sum()),
              "| 0->snow", int((z[:, :-1] & ~z[:, 1:]).sum()), "| snow->0", int((~z[:, :-1] & z[:, 1:]).sum()),
              "| max SNEQV %.3e" % sw.max())
    Path(fixture).write_text(json.dumps(fx, indent=0))
    np.savez_compressed(records, **rec)


if __name__ == "__main__":
    main()
