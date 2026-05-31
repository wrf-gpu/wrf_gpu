"""GPU wind-skill LOCALIZATION harness.

Runs the REAL operational segmented GPU forecast for case2 (0509 18z) L2 at the
requested lead(s), then dumps the FULL wind diagnostic chain so the V10/U10 error
can be localized SPATIALLY (land/water/coast), TEMPORALLY (across leads), and
PHYSICALLY (lowest-level wind error vs 10 m diagnostic-ratio error vs surface
drag).

For each lead it materializes from the post-forecast State:
  * the SCORED diagnostics:   U10, V10, T2  (compute_m9_diagnostics path)
  * the lowest-mass-level wind the 10 m diagnostic reads from: u0, v0, wspd
  * the surface-layer internals: ustar, zol, regime, psim, br, znt, the 10 m
    reconstruction ratio (u10/u0), Cd = (k/psix10)^2, tau_u, tau_v
and scores each vs the CPU-WRF wrfout_d02 truth at the valid time, split by
land / water / coast masks and reported as full fields (saved to .npz) plus a
compact JSON of the masked RMSE/bias decomposition.

This is the EXECUTE block. It LAUNCHES A GPU FORECAST. Share the box politely:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OMP_NUM_THREADS=2 taskset -c 0-3

USAGE
  PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OMP_NUM_THREADS=2 \
    taskset -c 0-3 python proofs/wind/gpu_wind_localize.py \
      --leads 24 48 --out proofs/wind/gpu_wind_localize.json \
      --npz proofs/wind/gpu_wind_localize_fields.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

L2_RUN_ID = "20260509_18z_l2_72h_20260511T190519Z"
L2_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l2")
DOMAIN = "d02"
INIT = datetime(2026, 5, 9, 18, 0, 0, tzinfo=timezone.utc)
FIELDS = ("U10", "V10", "T2")
STATIC = ("XLAND",)


def masks_from_xland(xland):
    import numpy as np
    xland = np.asarray(xland, dtype=np.float64)
    land = xland < 1.5          # WRF: 1=land, 2=water
    water = ~land
    coast = np.zeros_like(land)
    for ax in (0, 1):
        for sh in (1, -1):
            coast |= (land != np.roll(land, sh, axis=ax))
    coast[0, :] = coast[-1, :] = coast[:, 0] = coast[:, -1] = False
    return {"all": np.ones_like(land), "land": land, "water": water, "coast": coast}


def masked_stats(diff, m):
    import numpy as np
    d = np.asarray(diff, dtype=np.float64)[m]
    if d.size == 0:
        return None
    return {
        "rmse": float(np.sqrt(np.mean(d**2))),
        "bias": float(np.mean(d)),
        "mae": float(np.mean(np.abs(d))),
        "n": int(d.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=int, nargs="+", default=[24, 48])
    ap.add_argument("--segment-steps", type=int, default=180)
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--out", type=Path, default=Path("proofs/wind/gpu_wind_localize.json"))
    ap.add_argument("--npz", type=Path, default=Path("proofs/wind/gpu_wind_localize_fields.npz"))
    # Optional case override (defaults preserve the original case2 0509 L2 behavior).
    ap.add_argument("--run-id", type=str, default=L2_RUN_ID)
    ap.add_argument("--run-root", type=Path, default=L2_RUN_ROOT)
    ap.add_argument("--init", type=str, default=None,
                    help="init UTC 'YYYY-MM-DD_HH:MM:SS'; default = case2 0509 18z")
    ap.add_argument("--case-label", type=str, default="case2_0509_18z_L2")
    args = ap.parse_args()

    global INIT
    if args.init is not None:
        INIT = datetime.strptime(args.init, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    from gpuwrf.runtime.operational_mode import (
        compute_m9_diagnostics,
        run_forecast_operational_segmented,
    )
    from gpuwrf.coupling.physics_couplers import (
        _surface_column_view, _u_mass, _v_mass,
    )
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

    cfg = DailyPipelineConfig(
        run_id=args.run_id, run_root=args.run_root, domain=DOMAIN,
        dt_s=args.dt_s, acoustic_substeps=args.acoustic_substeps,
        radiation_cadence_steps=args.radiation_cadence_steps,
    )
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=args.radiation_cadence_steps, time_utc=case.run_start,
    )

    # static masks from the corpus init wrfout
    init_static = read_wrfout_file(
        run_dir / f"wrfout_{DOMAIN}_{INIT:%Y-%m-%d_%H:%M:%S}", fields=STATIC)["fields"]
    masks = masks_from_xland(init_static["XLAND"])

    per_lead: list[dict[str, Any]] = []
    npz_store: dict[str, Any] = {"XLAND": np.asarray(init_static["XLAND"])}
    for lead_h in args.leads:
        t0 = time.time()
        final_state = run_forecast_operational_segmented(
            case.state, nl, float(lead_h), segment_steps=args.segment_steps)
        jax.block_until_ready(final_state.theta)
        diags = compute_m9_diagnostics(final_state, nl, float(lead_h) * 3600.0)

        # surface-layer internals on the SAME final state, same column view as the
        # scored path (surface_layer_diagnostics uses _surface_column_view too).
        sl = surface_layer_with_diagnostics(_surface_column_view(final_state))
        # lowest-mass-level wind that the 10 m diagnostic reconstructs from.
        u0 = _u_mass(final_state)[0]          # (ny,nx) mass-point lowest level
        v0 = _v_mass(final_state)[0]
        wspd0 = jnp.sqrt(u0 * u0 + v0 * v0)

        get = lambda x: np.asarray(jax.device_get(x))
        gpu = {"U10": get(diags.u10), "V10": get(diags.v10), "T2": get(diags.t2)}
        internals = {
            "u0": get(u0), "v0": get(v0), "wspd0": get(wspd0),
            "ustar": get(sl.fluxes.ustar), "zol": get(sl.zol), "regime": get(sl.regime),
            "psim": get(sl.psim), "br": get(sl.br), "znt": get(sl.znt),
            "tau_u": get(sl.fluxes.tau_u), "tau_v": get(sl.fluxes.tau_v),
            # 10 m reconstruction ratio actually used (u10/u0), guarding |u0|~0
            "ratio10_u": get(jnp.where(jnp.abs(u0) > 0.05, sl.u10 / jnp.where(u0 == 0, 1.0, u0), jnp.nan)),
            "ratio10_v": get(jnp.where(jnp.abs(v0) > 0.05, sl.v10 / jnp.where(v0 == 0, 1.0, v0), jnp.nan)),
        }

        valid = INIT + timedelta(hours=lead_h)
        wrfout = run_dir / f"wrfout_{DOMAIN}_{valid:%Y-%m-%d_%H:%M:%S}"
        wrf = read_wrfout_file(wrfout, fields=FIELDS)["fields"]
        wrf = {f: np.asarray(wrf[f], dtype=np.float64) for f in FIELDS}

        # store full fields for offline plots / deeper analysis
        for f in FIELDS:
            npz_store[f"gpu_{f}_{lead_h}h"] = gpu[f]
            npz_store[f"wrf_{f}_{lead_h}h"] = wrf[f]
        for k, v in internals.items():
            npz_store[f"{k}_{lead_h}h"] = v

        # masked error decomposition
        row: dict[str, Any] = {"lead_h": lead_h, "wall_s": round(time.time() - t0, 1)}
        for f in FIELDS:
            diff = np.asarray(gpu[f], dtype=np.float64) - wrf[f]
            row[f] = {mname: masked_stats(diff, m) for mname, m in masks.items()}
        # surface internals masked means (water vs land) to attribute the drag
        def mmean(arr, m):
            a = np.asarray(arr, dtype=np.float64)[m]
            a = a[np.isfinite(a)]
            return None if a.size == 0 else float(np.mean(a))
        row["surface_internals_mean"] = {
            mname: {k: mmean(internals[k], m) for k in
                    ("u0", "v0", "wspd0", "ustar", "zol", "regime", "br", "znt",
                     "ratio10_u", "ratio10_v")}
            for mname, m in masks.items()
        }
        # GPU lowest-level wind vs WRF 10 m (proxy: is the error already in u0/v0,
        # before the diagnostic?). WRF has no diagnostic-free lowest wind in the
        # 2-D file, so compare GPU u0 magnitude to GPU u10 to see the ratio's role.
        row["wspd_check"] = {
            mname: {
                "gpu_wspd0_mean": mmean(internals["wspd0"], m),
                "gpu_wspd10_mean": mmean(np.hypot(gpu["U10"], gpu["V10"]), m),
                "wrf_wspd10_mean": mmean(np.hypot(wrf["U10"], wrf["V10"]), m),
            }
            for mname, m in masks.items()
        }
        per_lead.append(row)
        print(f"[lead {lead_h}h done in {row['wall_s']}s] "
              f"V10 all-RMSE={row['V10']['all']['rmse']:.3f} "
              f"water-RMSE={row['V10']['water']['rmse']:.3f} "
              f"land-RMSE={row['V10']['land']['rmse']:.3f}", flush=True)

    payload = {
        "_doc": "GPU wind localization: real operational segmented forecast for "
                "case2 (0509 18z) L2, V10/U10/T2 error decomposed by land/water/"
                "coast mask + surface-layer internals (u0,v0,ustar,zol,regime,Cd-"
                "proxy,10m-ratio). Truth = corpus CPU-WRF wrfout_d02.",
        "case": args.case_label,
        "init_utc": INIT.isoformat(),
        "config": {"dt_s": args.dt_s, "acoustic_substeps": args.acoustic_substeps,
                   "segment_steps": args.segment_steps,
                   "radiation_cadence_steps": args.radiation_cadence_steps},
        "grid": {k: int(v.sum()) for k, v in masks.items()},
        "per_lead": per_lead,
        "npz_file": str(args.npz),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(args.npz, **npz_store)
    print(f"\nwrote {args.out}\nwrote {args.npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
