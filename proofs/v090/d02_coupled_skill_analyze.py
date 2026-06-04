#!/usr/bin/env python
"""PART-1 analyzer: d02 coupled multi-hour SKILL vs CPU-WRF wrfout.

Reads the GPU d02-replay output (gated-fp32) per-hour wrfout and the matching
CPU-WRF corpus wrfout (the backfilled L2 d02 reference) and computes, PER LEAD
TIME, the gridded RMSE + bias for T2 / HFX / U10 / V10 / PBLH against the CPU
truth, plus a persistence (t=0 hold) baseline so the net skill is honest.

This is post-processing only: NO new GPU run. It consumes the wrfout already
written by scripts/m7_l2_d02_replay.py --gated-fp32 and the CPU reference run.

Usage:
  taskset -c 0-3 python3 proofs/v090/d02_coupled_skill_analyze.py \
      --gpu-output-dir /tmp/vburst_d02/l2_d02_<run_id> \
      --cpu-ref-dir /tmp/vburst_runs/<run_id> \
      --out proofs/v090/d02_coupled_skill.json \
      [--precision gated_fp32]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

# Operational acceptance bars. T2/U10/V10 are the binding v0.1.0/v0.2.0 Tier-4
# bars. HFX/PBLH are reported for context (no hard published bar; informational
# operational margins below are conservative engineering bands).
RMSE_BARS = {"T2": 3.0, "U10": 7.5, "V10": 7.5, "HFX": 120.0, "PBLH": 400.0}
FIELDS = ("T2", "HFX", "U10", "V10", "PBLH")


def _pin() -> None:
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0, 1, 2, 3})
        except OSError:
            pass


def _read(path: Path, fields) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with Dataset(path, "r") as ds:
        for name in fields:
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            data = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
            out[name] = np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = (a - b).ravel()
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(diff[finite] ** 2)))


def _bias(a: np.ndarray, b: np.ndarray) -> float:
    diff = (a - b).ravel()
    finite = np.isfinite(diff)
    if not finite.any():
        return float("nan")
    return float(np.mean(diff[finite]))


def _valid_time(path: Path) -> str:
    # wrfout_d02_2026-05-07_19:00:00
    stem = path.name.split("wrfout_d02_")[-1]
    return stem.replace("_", "T", 1)


def main(argv=None) -> int:
    _pin()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-output-dir", required=True, type=Path)
    ap.add_argument("--cpu-ref-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--precision", default="gated_fp32")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--wall-clock-json", default=None,
                    help="optional pipeline_run/wall_clock json to fold in command-to-finish wall")
    args = ap.parse_args(argv)

    gpu_files = sorted(Path(args.gpu_output_dir).glob("wrfout_d02_*"), key=lambda p: p.name)
    if not gpu_files:
        raise FileNotFoundError(f"no GPU wrfout_d02_* in {args.gpu_output_dir}")
    ref_dir = Path(args.cpu_ref_dir)

    # persistence baseline: t=0 CPU truth held at every lead
    t0_ref = _read(ref_dir / gpu_files[0].name, FIELDS)

    per_lead = []
    agg_sq = {f: [] for f in FIELDS}
    agg_bias = {f: [] for f in FIELDS}
    failures = []
    nonfinite_leads = 0

    for gf in gpu_files:
        ref_path = ref_dir / gf.name
        if not ref_path.is_file():
            failures.append(f"missing CPU truth for {gf.name}")
            continue
        gpu = _read(gf, FIELDS)
        ref = _read(ref_path, FIELDS)
        rec = {"valid_time_utc": _valid_time(gf), "fields": {}}
        any_nonfinite = False
        for f in FIELDS:
            if f not in gpu or f not in ref:
                continue
            g, r = gpu[f], ref[f]
            if g.shape != r.shape:
                failures.append(f"{f} shape {g.shape} vs {r.shape} at {gf.name}")
                continue
            if not np.isfinite(g).all():
                any_nonfinite = True
            rmse = _rmse(g, r)
            bias = _bias(g, r)
            pers = t0_ref.get(f)
            pers_rmse = _rmse(pers, r) if pers is not None and pers.shape == r.shape else None
            beats = bool(pers_rmse is not None and np.isfinite(rmse) and rmse <= pers_rmse)
            bar = RMSE_BARS.get(f)
            within = bool(np.isfinite(rmse) and (bar is None or rmse <= bar))
            agg_sq[f].append(rmse)
            agg_bias[f].append(bias)
            rec["fields"][f] = {
                "rmse": rmse,
                "bias": bias,
                "persistence_rmse": pers_rmse,
                "beats_persistence": beats,
                "skill_score": (float(1.0 - rmse / pers_rmse) if pers_rmse not in (None, 0.0) else None),
                "operational_bar": bar,
                "within_bar": within,
                "units": "K" if f == "T2" else ("W m-2" if f == "HFX" else ("m" if f == "PBLH" else "m s-1")),
            }
        if any_nonfinite:
            nonfinite_leads += 1
        per_lead.append(rec)

    final = per_lead[-1] if per_lead else {"fields": {}}
    # binding bars: T2/U10/V10 at every lead within bar
    binding = ("T2", "U10", "V10")
    binding_all_leads = all(
        rec["fields"].get(f, {}).get("within_bar", False)
        for rec in per_lead for f in binding if f in rec["fields"]
    ) and bool(per_lead)
    final_within = all(final["fields"].get(f, {}).get("within_bar", False) for f in binding if f in final["fields"])

    field_summary = {}
    for f in FIELDS:
        vals = [v for v in agg_sq[f] if np.isfinite(v)]
        biases = [v for v in agg_bias[f] if np.isfinite(v)]
        field_summary[f] = {
            "mean_rmse_over_leads": float(np.mean(vals)) if vals else None,
            "max_rmse_over_leads": float(np.max(vals)) if vals else None,
            "final_lead_rmse": final["fields"].get(f, {}).get("rmse"),
            "final_lead_bias": final["fields"].get(f, {}).get("bias"),
            "mean_bias_over_leads": float(np.mean(biases)) if biases else None,
            "operational_bar": RMSE_BARS.get(f),
            "all_leads_within_bar": bool(all(v <= RMSE_BARS[f] for v in vals)) if (vals and f in RMSE_BARS) else None,
        }

    wall = None
    if args.wall_clock_json and Path(args.wall_clock_json).is_file():
        wj = json.load(open(args.wall_clock_json))
        wall = {
            "wall_clock_total_s": wj.get("wall_clock_total_s"),
            "wall_clock_per_forecast_hour_s": wj.get("wall_clock_per_forecast_hour_s"),
            "wall_clock_per_hour_s": wj.get("wall_clock_per_hour_s"),
            "device": wj.get("device"),
            "hours": wj.get("hours"),
            "source": str(args.wall_clock_json),
        }

    status = "PASS" if (binding_all_leads and nonfinite_leads == 0 and not failures) else "FAIL"
    payload = {
        "schema": "V090D02CoupledSkill",
        "schema_version": 1,
        "status": status,
        "precision_mode": args.precision,
        "precision_policy": "ADR-007 gated-fp32: theta/u/v/qv fp32; mu/p/ph/w + acoustic/pressure accumulators fp64 (operational SHIP mode)" if args.precision == "gated_fp32" else "full fp64",
        "run_id": args.run_id or ref_dir.name,
        "gpu_output_dir": str(args.gpu_output_dir),
        "cpu_reference_dir": str(ref_dir),
        "reference_provenance": "CPU-WRF v4.7.1 28-rank backfilled L2 d02 wrfout (wrf_l2_backfill_output) + matching corpus namelist",
        "lead_count": len(per_lead),
        "nonfinite_leads": nonfinite_leads,
        "all_leads_finite": nonfinite_leads == 0,
        "binding_fields": list(binding),
        "binding_all_leads_within_bar": binding_all_leads,
        "final_lead_binding_within_bar": final_within,
        "operational_bars": RMSE_BARS,
        "field_summary": field_summary,
        "per_lead": per_lead,
        "wall_clock": wall,
        "failures": failures,
        "notes": [
            "RMSE/bias = gridded GPU-d02(gated-fp32) vs CPU-WRF d02 truth at each output hour.",
            "bias = mean(GPU - CPU); positive = GPU warmer/stronger.",
            "T2/U10/V10 bars are the binding v0.1.0/v0.2.0 Tier-4 operational gate. HFX/PBLH bars are informational engineering bands (no published hard bar).",
            "persistence baseline holds the t=0 CPU truth at every lead; beats_persistence = GPU RMSE <= persistence RMSE.",
        ],
        "generated": datetime.utcnow().isoformat() + "Z",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "lead_count": len(per_lead),
        "all_leads_finite": nonfinite_leads == 0,
        "binding_all_leads_within_bar": binding_all_leads,
        "field_summary": {f: {"mean_rmse": field_summary[f]["mean_rmse_over_leads"],
                              "final_rmse": field_summary[f]["final_lead_rmse"],
                              "mean_bias": field_summary[f]["mean_bias_over_leads"]} for f in FIELDS},
    }, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
