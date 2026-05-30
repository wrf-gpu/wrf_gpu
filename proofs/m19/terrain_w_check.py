"""Empirical decider for the bbfa269 surface-w BC contradiction.

Runs the FIXED GPU operational forecast (case1 = 20260528_18z, L2 d02) to a
lead where CPU-WRF wrfout W exists, dumps the GPU full-level staggered W, reads
the matching CPU-WRF W + HGT, and compares the vertical-velocity field over the
HIGHEST-TERRAIN columns vs ocean columns.

VERDICT LOGIC
-------------
* If GPU mountain-wave W over terrain is the same O(0.1-10 m/s) scale as
  CPU-WRF (correlated, ratio ~O(1)) -> the bbfa269 fix is CORRECT; agy's 1e5x
  silencing claim is REFUTED.
* If GPU W over terrain is ~1e5x (or even >=10x) SMALLER than CPU-WRF W
  (mountains produce ~no vertical motion) -> agy is CORRECT; mountain waves
  are silenced.

Read-only on src.  Writes only its own JSON proof + stdout.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-h", type=int, default=12)
    ap.add_argument("--run-id", default="20260528_18z_l2_72h_20260529T002423Z")
    ap.add_argument("--run-root", default="/mnt/data/canairy_meteo/runs/wrf_l2")
    ap.add_argument("--domain", default="d02")
    ap.add_argument("--segment-steps", type=int, default=180)
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--out", default="proofs/m19/terrain_w_check_result.json")
    ap.add_argument("--top-n-terrain", type=int, default=30)
    args = ap.parse_args()

    import jax
    from netCDF4 import Dataset

    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented

    cfg = DailyPipelineConfig(
        run_id=args.run_id,
        run_root=Path(args.run_root),
        domain=args.domain,
        dt_s=args.dt_s,
        acoustic_substeps=args.acoustic_substeps,
        radiation_cadence_steps=args.radiation_cadence_steps,
    )
    case, run_dir = _build_real_case(cfg)
    time_utc = case.run_start
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=True,
        radiation_cadence_steps=args.radiation_cadence_steps,
        time_utc=time_utc,
    )

    print(f"[terrain_w] running GPU forecast to +{args.lead_h}h ...", flush=True)
    final_state = run_forecast_operational_segmented(
        case.state, nl, float(args.lead_h), segment_steps=args.segment_steps
    )
    jax.block_until_ready(final_state.w)
    gpu_w = np.asarray(jax.device_get(final_state.w))  # (nz+1, ny, nx)
    print(f"[terrain_w] GPU W shape {gpu_w.shape} "
          f"range [{gpu_w.min():.4g}, {gpu_w.max():.4g}] finite={np.isfinite(gpu_w).all()}",
          flush=True)

    # --- CPU-WRF truth (W + HGT) at the same valid time ---
    valid = time_utc + timedelta(hours=args.lead_h)
    wrfout = run_dir / f"wrfout_{args.domain}_{valid:%Y-%m-%d_%H:%M:%S}"
    if not wrfout.is_file():
        raise FileNotFoundError(f"no CPU-WRF truth at {wrfout}")
    with Dataset(wrfout, "r") as ds:
        wrf_w = np.asarray(ds.variables["W"][0])      # (bottom_top_stag, sn, we)
        hgt = np.asarray(ds.variables["HGT"][0])       # (sn, we)
    print(f"[terrain_w] CPU-WRF W shape {wrf_w.shape} "
          f"range [{wrf_w.min():.4g}, {wrf_w.max():.4g}]", flush=True)
    print(f"[terrain_w] HGT range [{hgt.min():.1f}, {hgt.max():.1f}] m", flush=True)

    # Align shapes (GPU may carry an extra halo? expect exact match).
    if gpu_w.shape != wrf_w.shape:
        nz = min(gpu_w.shape[0], wrf_w.shape[0])
        ny = min(gpu_w.shape[1], wrf_w.shape[1])
        nx = min(gpu_w.shape[2], wrf_w.shape[2])
        gpu_w = gpu_w[:nz, :ny, :nx]
        wrf_w = wrf_w[:nz, :ny, :nx]
        hgt = hgt[:ny, :nx]
        print(f"[terrain_w] shapes differed -> cropped to {gpu_w.shape}", flush=True)

    nz, ny, nx = gpu_w.shape

    # --- terrain vs ocean column masks ---
    flat_hgt = hgt.ravel()
    order = np.argsort(flat_hgt)[::-1]
    top_idx = order[: args.top_n_terrain]
    terrain_mask = np.zeros((ny, nx), dtype=bool)
    terrain_mask.ravel()[top_idx] = True
    ocean_mask = hgt < 1.0

    def col_stats(w3d, mask2d):
        colmax = np.max(np.abs(w3d), axis=0)  # (ny, nx)
        sel = colmax[mask2d]
        return {
            "n_cols": int(mask2d.sum()),
            "max_abs_w": float(sel.max()) if sel.size else 0.0,
            "mean_abs_w": float(sel.mean()) if sel.size else 0.0,
            "p95_abs_w": float(np.percentile(sel, 95)) if sel.size else 0.0,
        }

    gpu_terr = col_stats(gpu_w, terrain_mask)
    wrf_terr = col_stats(wrf_w, terrain_mask)
    gpu_ocean = col_stats(gpu_w, ocean_mask)
    wrf_ocean = col_stats(wrf_w, ocean_mask)

    def rmse_3d(mask2d):
        m3 = np.broadcast_to(mask2d, (nz, ny, nx))
        d = (gpu_w - wrf_w)[m3]
        return float(np.sqrt(np.mean(d**2))) if d.size else 0.0

    terr_rmse = rmse_3d(terrain_mask)
    ocean_rmse = rmse_3d(ocean_mask)

    def face_stats(k, mask2d):
        g = gpu_w[k][mask2d]
        t = wrf_w[k][mask2d]
        return {
            "gpu_max_abs": float(np.abs(g).max()) if g.size else 0.0,
            "wrf_max_abs": float(np.abs(t).max()) if t.size else 0.0,
            "gpu_mean_abs": float(np.abs(g).mean()) if g.size else 0.0,
            "wrf_mean_abs": float(np.abs(t).mean()) if t.size else 0.0,
        }

    surf_terr = face_stats(0, terrain_mask)

    ratio = (wrf_terr["max_abs_w"] / gpu_terr["max_abs_w"]) if gpu_terr["max_abs_w"] > 0 else float("inf")

    g_cm = np.max(np.abs(gpu_w), axis=0)[terrain_mask]
    w_cm = np.max(np.abs(wrf_w), axis=0)[terrain_mask]
    if g_cm.size > 2 and g_cm.std() > 0 and w_cm.std() > 0:
        corr = float(np.corrcoef(g_cm, w_cm)[0, 1])
    else:
        corr = float("nan")

    pk = np.unravel_index(np.argmax(hgt), hgt.shape)
    peak_profile = {
        "j": int(pk[0]), "i": int(pk[1]), "hgt_m": float(hgt[pk]),
        "gpu_w_col": [float(x) for x in gpu_w[:, pk[0], pk[1]]],
        "wrf_w_col": [float(x) for x in wrf_w[:, pk[0], pk[1]]],
    }

    result = {
        "lead_h": args.lead_h,
        "valid_utc": valid.isoformat(),
        "wrfout": str(wrfout),
        "shapes": {"gpu_w": list(gpu_w.shape), "wrf_w": list(wrf_w.shape)},
        "gpu_w_global_range": [float(gpu_w.min()), float(gpu_w.max())],
        "wrf_w_global_range": [float(wrf_w.min()), float(wrf_w.max())],
        "gpu_w_finite": bool(np.isfinite(gpu_w).all()),
        "hgt_range_m": [float(hgt.min()), float(hgt.max())],
        "terrain_cells": {"gpu": gpu_terr, "wrf": wrf_terr},
        "ocean_cells": {"gpu": gpu_ocean, "wrf": wrf_ocean},
        "terrain_w_rmse": terr_rmse,
        "ocean_w_rmse": ocean_rmse,
        "surface_face_terrain": surf_terr,
        "scale_ratio_wrf_over_gpu_terrain_maxw": ratio,
        "terrain_colmax_correlation": corr,
        "peak_column_profile": peak_profile,
    }

    if gpu_terr["max_abs_w"] < 1e-3 and wrf_terr["max_abs_w"] > 0.1:
        verdict = "BUG_CONFIRMED"
    elif ratio >= 10.0:
        verdict = "BUG_CONFIRMED"
    elif 0.1 <= ratio <= 10.0:
        verdict = "FIX_CORRECT"
    else:
        verdict = "FIX_CORRECT_BUT_GPU_LARGER"
    result["verdict"] = verdict

    print("\n===== TERRAIN-W VERDICT =====", flush=True)
    print(f"GPU W global range:  [{gpu_w.min():.4g}, {gpu_w.max():.4g}]", flush=True)
    print(f"WRF W global range:  [{wrf_w.min():.4g}, {wrf_w.max():.4g}]", flush=True)
    print(f"Terrain max|W|:  GPU={gpu_terr['max_abs_w']:.4g}  WRF={wrf_terr['max_abs_w']:.4g}  "
          f"ratio(WRF/GPU)={ratio:.3g}", flush=True)
    print(f"Terrain mean|W|: GPU={gpu_terr['mean_abs_w']:.4g}  WRF={wrf_terr['mean_abs_w']:.4g}", flush=True)
    print(f"Ocean  max|W|:   GPU={gpu_ocean['max_abs_w']:.4g}  WRF={wrf_ocean['max_abs_w']:.4g}", flush=True)
    print(f"Terrain W RMSE={terr_rmse:.4g}   Ocean W RMSE={ocean_rmse:.4g}", flush=True)
    print(f"Terrain colmax corr={corr:.3f}", flush=True)
    print(f"Peak col HGT={peak_profile['hgt_m']:.0f}m  "
          f"GPU max|w_col|={max(abs(x) for x in peak_profile['gpu_w_col']):.4g}  "
          f"WRF max|w_col|={max(abs(x) for x in peak_profile['wrf_w_col']):.4g}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)

    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
