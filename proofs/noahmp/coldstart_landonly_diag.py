#!/usr/bin/env python3
"""Cold-start DIAGNOSIS - standalone land-only overnight integration.

Drives the GPU Noah-MP land carry with WRF's OWN forcing (extracted hourly from
the corpus wrfout sequence: lowest-level T/Q/U/V/P + SWDOWN/GLW/COSZEN it used)
with NO atmosphere feedback, integrates overnight at the real dt, and compares
TG/TV/TAH/TSK/HFX/GRDFLX against WRF's wrfout land state hour-by-hour.

Init from WRF's actual t=0 wrfout land state (== our NOAHMP_INIT reconstruction,
verified byte-identical), so any divergence is purely the integrated land scheme.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netCDF4 as nc

from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.physics.noahmp.types import NoahMPForcing

P0 = 1.0e5
RDGAS = 287.04
CP = 1004.64
RCP = RDGAS / CP
GRAV = 9.80616


def _sq(a):
    return np.squeeze(np.asarray(a, dtype=np.float64))


def load_forcing_from_wrfout(path):
    ds = nc.Dataset(path)
    def g(v):
        return _sq(ds.variables[v][:])
    th = g("T")[0] + 300.0
    p = g("P")[0] + g("PB")[0]
    qv = g("QVAPOR")[0]
    U = g("U")[0]; V = g("V")[0]
    u = 0.5 * (U[:, :-1] + U[:, 1:])
    v = 0.5 * (V[:-1, :] + V[1:, :])
    psfc = g("PSFC")
    swdown = g("SWDOWN"); glw = g("GLW"); cosz = g("COSZEN")
    ph = g("PH"); phb = g("PHB")
    z_w = (ph + phb) / GRAV
    hgt = g("HGT")
    zlvl = 0.5 * (z_w[0] + z_w[1]) - hgt
    sfctmp = th * (p / P0) ** RCP
    tg = g("TG"); tv = g("TV"); tah = g("TAH"); tsk = g("TSK")
    hfx = g("HFX"); lh = g("LH"); grdflx = g("GRDFLX")
    t2 = g("T2")
    ds.close()
    return dict(sfctmp=sfctmp, sfcprs=p, psfc=psfc, uu=u, vv=v, qair=np.maximum(qv, 0.0),
                qc=np.zeros_like(qv), soldn=np.maximum(swdown, 0.0), lwdn=glw, cosz=np.maximum(cosz, 0.0),
                zlvl=zlvl, tg=tg, tv=tv, tah=tah, tsk=tsk, hfx=hfx, lh=lh, grdflx=grdflx, t2=t2)


def make_forcing(fd, julian, yearlen):
    z = jnp.zeros_like(jnp.asarray(fd["sfctmp"]))
    return NoahMPForcing(
        sfctmp=jnp.asarray(fd["sfctmp"]), sfcprs=jnp.asarray(fd["sfcprs"]),
        psfc=jnp.asarray(fd["psfc"]), uu=jnp.asarray(fd["uu"]), vv=jnp.asarray(fd["vv"]),
        qair=jnp.asarray(fd["qair"]), qc=jnp.asarray(fd["qc"]),
        soldn=jnp.asarray(fd["soldn"]), lwdn=jnp.asarray(fd["lwdn"]),
        prcpconv=z, prcpnonc=z, prcpsnow=z, prcpgrpl=z, prcphail=z,
        cosz=jnp.asarray(fd["cosz"]), zlvl=jnp.asarray(fd["zlvl"]),
        julian=jnp.asarray(float(julian)), yearlen=jnp.asarray(float(yearlen)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--domain", default="d03")
    ap.add_argument("--init", default="2026-05-21_18:00:00")
    ap.add_argument("--hours", type=int, default=12)
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument(
        "--table-dir",
        default=os.environ.get(
            "WRF_PRISTINE_ROOT_RUN",
            str(ROOT.parent / "wrf_pristine" / "WRF" / "run"),
        ),
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    dom = args.domain
    t0 = datetime.strptime(args.init, "%Y-%m-%d_%H:%M:%S")

    ls, st, meta = build_noahmp_land_state(rd, dom, table_dir=args.table_dir)
    en_p, rad_p, nroot = build_noahmp_params(st)
    land = np.asarray(st.xland) < 1.5

    @jax.jit
    def step(ls, forcing):
        ls2, fl = noah_mp_step(ls, forcing, st, args.dt,
                               energy_params=en_p, rad_params=rad_p)
        return ls2, fl

    julian = t0.timetuple().tm_yday + t0.hour / 24.0
    nsub = int(round(3600.0 / args.dt))

    def landstat(name, gpu, wrf):
        g = np.asarray(gpu); w = np.asarray(wrf)
        d = (g - w)[land]
        d = d[np.isfinite(d)]
        return {"bias": float(d.mean()), "rmse": float(np.sqrt((d**2).mean())),
                "mae": float(np.abs(d).mean()), "gpu_mean": float(g[land].mean()),
                "wrf_mean": float(w[land].mean())}

    rows = []
    fl = None
    for h in range(args.hours):
        fpath = rd / f"wrfout_{dom}_{(t0 + timedelta(hours=h)).strftime('%Y-%m-%d_%H:%M:%S')}"
        fd = load_forcing_from_wrfout(str(fpath))
        forcing = make_forcing(fd, julian + h / 24.0, 365.0)
        for _ in range(nsub):
            ls, fl = step(ls, forcing)
        jax.block_until_ready((ls.tg, ls.tv))
        vpath = rd / f"wrfout_{dom}_{(t0 + timedelta(hours=h+1)).strftime('%Y-%m-%d_%H:%M:%S')}"
        vd = load_forcing_from_wrfout(str(vpath))
        row = {
            "lead_h": h + 1,
            "valid": (t0 + timedelta(hours=h+1)).isoformat(),
            "TG": landstat("TG", ls.tg, vd["tg"]),
            "TV": landstat("TV", ls.tv, vd["tv"]),
            "TAH": landstat("TAH", ls.tah, vd["tah"]),
            "TSK": landstat("TSK", fl.tsk, vd["tsk"]),
            "HFX": landstat("HFX", fl.hfx, vd["hfx"]),
            "GRDFLX": landstat("GRDFLX", fl.grdflx, vd["grdflx"]),
            "all_finite": bool(np.all(np.isfinite(np.asarray(ls.tg))) and
                               np.all(np.isfinite(np.asarray(ls.tv)))),
        }
        rows.append(row)
        print(f"h={h+1:2d} TG bias={row['TG']['bias']:+.3f} TV bias={row['TV']['bias']:+.3f} "
              f"TSK bias={row['TSK']['bias']:+.3f} GRDFLX bias={row['GRDFLX']['bias']:+.2f} "
              f"(gpu {row['GRDFLX']['gpu_mean']:+.1f} wrf {row['GRDFLX']['wrf_mean']:+.1f}) "
              f"HFX bias={row['HFX']['bias']:+.2f}", flush=True)

    out = {
        "proof": "cold-start land-only overnight diagnosis (WRF-forced, no atm feedback)",
        "run_dir": str(rd), "domain": dom, "init": args.init,
        "dt_s": args.dt, "hours": args.hours, "n_land": int(land.sum()),
        "per_lead": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
