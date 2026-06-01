#!/usr/bin/env python
"""DIAGNOSIS: d02 3 km T2 warm bias on the CURRENT (post-P1-4a) code.

Runs the PRODUCTION d02 operational pipeline (``_build_real_case`` ->
``execute_daily_pipeline``, domain=d02, ``force_geopotential=True``) so the GPU
writes real wrfouts (T2/U10/V10/Q2/PBLH + P/PB/PH/PHB/PSFC/T/HFX/LH/TSK/GLW/
SWDOWN), then scores them against the corpus CPU-WRF wrfout_d02 truth and runs
the SAME Exner pressure-vs-theta decomposition + surface-energy-budget table the
d03 RCA used (``scripts/diag/d03_psfc_t2_check.py`` /
``.agent/reviews/2026-06-01-gpt-t2bias-energy-budget.md``).

This is DIAGNOSIS ONLY: no production ``src/`` edits.  CPU pinned 0-3.

PHASE 1 (GPU):  python d02_t2bias_diagnosis.py run  --run-id <id> [--hours 24]
PHASE 2 (CPU):  python d02_t2bias_diagnosis.py score --gpu-dir <dir> --run-id <id>

The two phases are split so the heavy GPU run commits its wrfouts before the
(cheap, re-runnable) CPU scoring/decomposition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from netCDF4 import Dataset

L3_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
L2_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l2")
OUTPUT_ROOT = Path("/tmp/v010_d02_diag_runs")
PROOF_DIR = ROOT / "proofs" / "v010_validation"

KAPPA = 287.0 / 1004.0
P0 = 1.0e5

TENERIFE_BOX = {"lat_min": 27.9, "lat_max": 28.7, "lon_min": -17.0, "lon_max": -16.0}


# ---------------------------------------------------------------------------
# PHASE 1 -- GPU forecast (production d02 path) writing real wrfouts.
# ---------------------------------------------------------------------------
def phase_run(args: argparse.Namespace) -> int:
    from gpuwrf.integration.daily_pipeline import (
        DailyPipelineConfig,
        _build_real_case,
        execute_daily_pipeline,
        resolve_run_dir,
        write_json,
    )

    run_root = Path(args.run_root)
    run_dir = resolve_run_dir(args.run_id, run_root)
    tag = f"_{args.tag}" if args.tag else ""
    output_dir = Path(args.output_root) / f"d02_{args.run_id}{tag}"
    config = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=PROOF_DIR,
        run_root=run_root,
        score=False,
        domain="d02",
        dt_s=float(args.dt_s),
        acoustic_substeps=int(args.acoustic_substeps),
        radiation_cadence_steps=int(args.radiation_cadence_steps),
    )
    print(f"=== d02 production pipeline: run_id={args.run_id} hours={args.hours} -> {output_dir} ===", flush=True)
    payload = execute_daily_pipeline(config, case_builder=_build_real_case)
    payload["orchestration_cpu_affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    write_json(PROOF_DIR / f"pipeline_run_d02_diag{tag}.json", payload)
    print(json.dumps({
        "verdict": payload.get("verdict"),
        "all_finite": (payload.get("all_finite_check") or {}).get("all_finite"),
        "n_wrfout": len(payload.get("wrfout_files", [])),
        "output_dir": str(output_dir),
        "wall_clock_total_s": payload.get("wall_clock_total_s"),
    }, indent=2))
    return 0 if payload.get("verdict") != "PIPELINE_BLOCKED" else 2


# ---------------------------------------------------------------------------
# PHASE 2 -- CPU scoring + Exner / energy-budget decomposition.
# ---------------------------------------------------------------------------
def _read(ds: Dataset, name: str) -> np.ndarray:
    v = ds.variables[name]
    data = v[0] if v.dimensions and v.dimensions[0] == "Time" else v[:]
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def _has(ds: Dataset, name: str) -> bool:
    return name in ds.variables


def _t_actual_bottom(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lowest-model-level actual T, surface pressure (P+PB)[0], theta[0]+300."""
    P = _read(ds, "P")
    PB = _read(ds, "PB")
    theta = _read(ds, "T") + 300.0
    p_full = (P + PB)[0]
    th0 = theta[0]
    t_act = th0 * (np.maximum(p_full, 1.0) / P0) ** KAPPA
    return t_act, p_full, th0


def _stats(g: np.ndarray, t: np.ndarray, mask: np.ndarray | None = None) -> dict:
    g = np.asarray(g, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if mask is not None:
        g = g[mask]
        t = t[mask]
    d = g - t
    return {
        "bias": float(np.nanmean(d)),
        "rmse": float(np.sqrt(np.nanmean(d * d))),
        "mae": float(np.nanmean(np.abs(d))),
        "gpu_mean": float(np.nanmean(g)),
        "wrf_mean": float(np.nanmean(t)),
        "n": int(np.isfinite(d).sum()),
    }


def phase_score(args: argparse.Namespace) -> int:
    gpu_dir = Path(args.gpu_dir)
    run_dir = (Path(args.run_root) / args.run_id)
    gpu_files = sorted(gpu_dir.glob("wrfout_d02_*"))
    if not gpu_files:
        print(f"NO GPU wrfout_d02 in {gpu_dir}")
        return 1

    # ----- masks from the first corpus file -----
    first_corpus = run_dir / gpu_files[0].name
    with Dataset(first_corpus) as c0:
        landmask = _read(c0, "LANDMASK") > 0.5
        lat = _read(c0, "XLAT")
        lon = _read(c0, "XLONG")
    sea = ~landmask
    # exclude a 1-cell outer ring from land/sea decomposition (boundary forcing).
    interior = np.zeros_like(landmask)
    interior[1:-1, 1:-1] = True
    land_i = landmask & interior
    sea_i = sea & interior
    tene = ((lat >= TENERIFE_BOX["lat_min"]) & (lat <= TENERIFE_BOX["lat_max"]) &
            (lon >= TENERIFE_BOX["lon_min"]) & (lon <= TENERIFE_BOX["lon_max"]))

    SURF = ("T2", "U10", "V10", "Q2", "PBLH")
    BUDGET = ("HFX", "LH", "GLW", "SWDOWN", "TSK")

    per_lead: list[dict] = []
    # accumulators for land/sea-day/night aggregate energy budget
    agg = {k: {f: [] for f in ("T2", "TSK", "HFX", "LH", "GLW", "SWDOWN", "PBLH", "Q2",
                                "psfc", "p_only", "th_only", "t_act", "th0")}
           for k in ("land_all", "land_day", "land_night", "sea_all", "sea_day", "sea_night")}

    print(f"=== d02 T2-bias decomposition: {len(gpu_files)} leads (GPU={gpu_dir.name}) ===")
    hdr = f"{'lead':>21} {'T2_b':>7} {'T2_rmse':>8} {'U10_b':>7} {'V10_b':>7} {'Q2_b':>9} {'PBLH_b':>7} {'PSFC_b_Pa':>10} {'Ponly_K':>8} {'THonly_K':>9} {'TSK_b':>7}"
    print(hdr)

    for gp in gpu_files:
        cp = run_dir / gp.name
        if not cp.is_file():
            continue
        g = Dataset(gp)
        c = Dataset(cp)
        rec: dict[str, Any] = {"lead": gp.name.replace("wrfout_d02_", "")}

        # surface field RMSE/bias (full domain)
        for f in SURF:
            if _has(g, f) and _has(c, f):
                rec[f] = _stats(_read(g, f), _read(c, f))
        # tenerife T2
        if _has(g, "T2") and _has(c, "T2"):
            rec["T2_tenerife"] = _stats(_read(g, "T2"), _read(c, "T2"), tene)

        # Exner pressure-vs-theta decomposition at lowest model level
        gt_act, gp_full, gth0 = _t_actual_bottom(g)
        ct_act, cp_full, cth0 = _t_actual_bottom(c)
        psfc_b = gp_full - cp_full
        # pressure-only: hold gpu theta, swap pressure to corpus
        p_only = gth0 * (np.maximum(gp_full, 1.0) / P0) ** KAPPA - gth0 * (np.maximum(cp_full, 1.0) / P0) ** KAPPA
        # theta-only: hold corpus pressure, swap theta
        th_only = gth0 * (np.maximum(cp_full, 1.0) / P0) ** KAPPA - cth0 * (np.maximum(cp_full, 1.0) / P0) ** KAPPA
        rec["psfc_bias_mean"] = float(np.nanmean(psfc_b))
        rec["t_actbottom_bias_mean"] = float(np.nanmean(gt_act - ct_act))
        rec["pressure_only_K_mean"] = float(np.nanmean(p_only))
        rec["theta_only_K_mean"] = float(np.nanmean(th_only))
        rec["theta0_bias_mean"] = float(np.nanmean(gth0 - cth0))

        # budget fields full-domain
        for f in BUDGET:
            if _has(g, f) and _has(c, f):
                rec[f] = _stats(_read(g, f), _read(c, f))

        # accumulate land/sea day/night
        sw_corpus = _read(c, "SWDOWN") if _has(c, "SWDOWN") else np.zeros_like(landmask, dtype=float)
        day = sw_corpus > 50.0
        night = sw_corpus <= 10.0
        regions = {
            "land_all": land_i, "land_day": land_i & day, "land_night": land_i & night,
            "sea_all": sea_i, "sea_day": sea_i & day, "sea_night": sea_i & night,
        }
        gfields = {f: (_read(g, f) if _has(g, f) else None) for f in
                   ("T2", "TSK", "HFX", "LH", "GLW", "SWDOWN", "PBLH", "Q2")}
        cfields = {f: (_read(c, f) if _has(c, f) else None) for f in gfields}
        for rk, rmask in regions.items():
            if rmask.sum() == 0:
                continue
            for f in gfields:
                if gfields[f] is not None and cfields[f] is not None:
                    agg[rk][f].append(float(np.nanmean((gfields[f] - cfields[f])[rmask])))
            agg[rk]["psfc"].append(float(np.nanmean(psfc_b[rmask])))
            agg[rk]["p_only"].append(float(np.nanmean(p_only[rmask])))
            agg[rk]["th_only"].append(float(np.nanmean(th_only[rmask])))
            agg[rk]["t_act"].append(float(np.nanmean((gt_act - ct_act)[rmask])))
            agg[rk]["th0"].append(float(np.nanmean((gth0 - cth0)[rmask])))

        per_lead.append(rec)
        t2 = rec.get("T2", {})
        q2 = rec.get("Q2", {})
        pblh = rec.get("PBLH", {})
        u10 = rec.get("U10", {})
        v10 = rec.get("V10", {})
        tsk = rec.get("TSK", {})
        print(f"{rec['lead']:>21} {t2.get('bias',float('nan')):>+7.3f} {t2.get('rmse',float('nan')):>8.3f} "
              f"{u10.get('bias',float('nan')):>+7.3f} {v10.get('bias',float('nan')):>+7.3f} "
              f"{q2.get('bias',float('nan')):>+9.6f} {pblh.get('bias',float('nan')):>+7.1f} "
              f"{rec['psfc_bias_mean']:>+10.1f} {rec['pressure_only_K_mean']:>+8.3f} "
              f"{rec['theta_only_K_mean']:>+9.3f} {tsk.get('bias',float('nan')):>+7.3f}")
        g.close()
        c.close()

    # aggregate energy budget table
    def _m(lst: list[float]) -> float:
        return float(np.nanmean(lst)) if lst else float("nan")

    budget_table = {}
    print("\n=== aggregate land/sea day/night (GPU - corpus) ===")
    print(f"{'region':>11} {'T2_K':>7} {'TSK_K':>7} {'HFX':>8} {'LH':>9} {'GLW':>7} {'SW':>8} {'PBLH':>7} {'Q2':>10} {'PSFC_Pa':>9} {'Ponly_K':>8} {'THonly_K':>9}")
    for rk in ("land_all", "land_day", "land_night", "sea_all", "sea_day", "sea_night"):
        a = agg[rk]
        row = {
            "T2": _m(a["T2"]), "TSK": _m(a["TSK"]), "HFX": _m(a["HFX"]), "LH": _m(a["LH"]),
            "GLW": _m(a["GLW"]), "SWDOWN": _m(a["SWDOWN"]), "PBLH": _m(a["PBLH"]),
            "Q2": _m(a["Q2"]), "psfc": _m(a["psfc"]), "p_only": _m(a["p_only"]),
            "th_only": _m(a["th_only"]), "t_act": _m(a["t_act"]), "th0": _m(a["th0"]),
        }
        budget_table[rk] = row
        print(f"{rk:>11} {row['T2']:>+7.3f} {row['TSK']:>+7.3f} {row['HFX']:>+8.1f} {row['LH']:>+9.1f} "
              f"{row['GLW']:>+7.1f} {row['SWDOWN']:>+8.1f} {row['PBLH']:>+7.1f} {row['Q2']:>+10.6f} "
              f"{row['psfc']:>+9.1f} {row['p_only']:>+8.3f} {row['th_only']:>+9.3f}")

    # mean over leads of headline metrics
    def _mean_lead(field: str, key: str) -> float:
        vals = [r[field][key] for r in per_lead if field in r and key in r[field]]
        return float(np.nanmean(vals)) if vals else float("nan")

    summary = {
        "schema": "D02T2BiasDiagnosis",
        "run_id": args.run_id,
        "gpu_dir": str(gpu_dir),
        "n_leads": len(per_lead),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mean_over_leads": {
            f: {"bias": _mean_lead(f, "bias"), "rmse": _mean_lead(f, "rmse")}
            for f in SURF
        },
        "mean_psfc_bias_Pa": float(np.nanmean([r["psfc_bias_mean"] for r in per_lead])),
        "mean_pressure_only_K": float(np.nanmean([r["pressure_only_K_mean"] for r in per_lead])),
        "mean_theta_only_K": float(np.nanmean([r["theta_only_K_mean"] for r in per_lead])),
        "mean_theta0_bias_K": float(np.nanmean([r["theta0_bias_mean"] for r in per_lead])),
        "final_lead_T2": per_lead[-1].get("T2") if per_lead else None,
        "budget_land_sea_day_night": budget_table,
        "per_lead": per_lead,
    }
    out = Path(args.out) if args.out else (PROOF_DIR / f"d02_t2bias_diag_{args.run_id}.json")
    out.write_text(json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o) + "\n")
    print(f"\nMEAN-over-leads: T2 bias={summary['mean_over_leads']['T2']['bias']:+.3f} rmse={summary['mean_over_leads']['T2']['rmse']:.3f}  "
          f"PSFC bias={summary['mean_psfc_bias_Pa']:+.1f} Pa  Ponly={summary['mean_pressure_only_K']:+.3f}K  THonly={summary['mean_theta_only_K']:+.3f}K")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="PHASE 1: GPU d02 production forecast -> wrfouts")
    pr.add_argument("--run-id", required=True)
    pr.add_argument("--run-root", default=str(L3_RUN_ROOT))
    pr.add_argument("--hours", type=int, default=24)
    pr.add_argument("--dt-s", type=float, default=10.0)
    pr.add_argument("--acoustic-substeps", type=int, default=10)
    pr.add_argument("--radiation-cadence-steps", type=int, default=180)
    pr.add_argument("--output-root", default=str(OUTPUT_ROOT))
    pr.add_argument("--tag", default="")
    pr.set_defaults(func=phase_run)

    sc = sub.add_parser("score", help="PHASE 2: CPU score + Exner/budget decomposition")
    sc.add_argument("--gpu-dir", required=True)
    sc.add_argument("--run-id", required=True)
    sc.add_argument("--run-root", default=str(L3_RUN_ROOT))
    sc.add_argument("--out", default="")
    sc.set_defaults(func=phase_score)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
