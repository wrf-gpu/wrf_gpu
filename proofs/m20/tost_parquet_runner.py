#!/usr/bin/env python3
"""v0.2.0 PARQUET-grounded TOST GPU runner (ADR-029).

Runs the GPU replay forecast for each runnable (case, domain) unit, emits a
full-grid wrfout NetCDF per lead hour (T2/U10/V10/XLAT/XLONG), then scores it
against the 31-case CPU-WRF point-shadow PARQUET corpus via
``tost_parquet_scorer`` (which samples the GPU grid at the parquet's
``nearest_grid_iy/ix`` cells -- the identical nearest-grid-cell rule).

REUSE-ONLY.  Launches NO new CPU-WRF runs.  The GPU forecast is the SAME
carry-advance loop the v0.1.0 d02 validator + the S6b/ensemble TOST use
(gpuwrf.integration.daily_pipeline._build_real_case + operational_mode
_advance_chunk).  The consolidated L1 rad-time + P1-5 Thompson fixes are in the
operational dycore/physics, so the emit carries those automatically.

ONE GPU CONSUMER.  Advances ONE forecast carry per unit to its max parquet lead,
snapshotting at every integer lead hour the parquet covers.  Memory bounded to
one segment (block_until_ready between segments).

INIT-CAP (honest): the replay forecast needs the t=0 wrfout snapshot + parent
hourly wrfout boundary history.  That survives only for 3 d02 cases (0509, 0521,
0530) + a few partial d03; the runnable manifest lists exactly those.  See the
FINDINGS report for the n-cap discussion (parquet truth is n=31, GPU-side n is
init-capped until CPU-WRF is re-run from the retained met_em forcing).

USAGE (manager, sequenced on the FREE GPU, fp64):
  # dry plan (no GPU):
  JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 python3 \
    proofs/m20/tost_parquet_runner.py --plan

  # smoke (2 d02 cases) -> emit + parquet-TOST:
  PYTHONPATH=src OMP_NUM_THREADS=4 JAX_ENABLE_X64=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 taskset -c 0-3 \
    python3 proofs/m20/tost_parquet_runner.py --execute \
      --units 20260509_18z__d02 20260530_18z__d02 \
      --out-dir proofs/m20/tost_run/parquet_smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # THIS worktree's repo root
WT_SRC = str(ROOT / "src")
sys.path.insert(0, WT_SRC)
sys.path.insert(0, str(HERE))

# RUNNABLE replay units: case_id (as it appears in the parquet) + the wrf run dir
# that retains the t=0 wrfout + boundary history.  domain == d02 grid (the L2
# 3km replay; the parquet d02 nest).  max_lead from the retained frame count.
RUNNABLE_UNITS = {
    "20260509_18z__d02": {
        "case_id": "20260509_18z", "domain": "d02",
        "run_root": "<DATA_ROOT>/canairy_meteo/runs/wrf_l2",
        "run_id": "20260509_18z_l2_72h_20260511T190519Z",
        "init_utc": "2026-05-09T18:00:00+00:00", "max_lead_h": 72,
        "wrfout_frames": 73, "grid": [66, 120], "parquet_workdir_match": True,
    },
    "20260530_18z__d02": {
        "case_id": "20260530_18z", "domain": "d02",
        "run_root": "<DATA_ROOT>/canairy_meteo/runs/wrf_l2",
        "run_id": "20260530_18z_l2_72h_20260531T161057Z",
        "init_utc": "2026-05-30T18:00:00+00:00", "max_lead_h": 72,
        "wrfout_frames": 73, "grid": [66, 159], "parquet_workdir_match": True,
    },
    # 0521: parquet case_id is the l2rerun variant; the wrf_l2 dir that retains
    # wrfout is the non-rerun 72h run (only 20 frames -> ~19 h coverage).  The
    # parquet workdir does NOT match the retained wrf_l2 dir, so the (iy,ix)
    # alignment is NOT guaranteed -> EXCLUDED from the clean campaign by default
    # (kept here for completeness / manager override only).
    "20260521_18z_l2rerun__d02": {
        "case_id": "20260521_18z_l2rerun", "domain": "d02",
        "run_root": "<DATA_ROOT>/canairy_meteo/runs/wrf_l2",
        "run_id": "20260521_18z_l2_72h_20260522T133443Z",
        "init_utc": "2026-05-21T18:00:00+00:00", "max_lead_h": 19,
        "wrfout_frames": 20, "grid": [66, 159], "parquet_workdir_match": False,
        "caveat": "parquet workdir != retained wrf_l2 dir; iy/ix alignment "
                  "unverified -- DO NOT include without grid re-check",
    },
}
DEFAULT_CAMPAIGN_UNITS = ["20260509_18z__d02", "20260530_18z__d02"]


def run_gpu_unit(spec: dict, out_dir: Path, parquet_leads: list[int], *,
                 dt_s: float, acoustic_substeps: int,
                 radiation_cadence_steps: int, segment_steps: int) -> dict:
    """Advance ONE GPU forecast carry, emitting a full-grid wrfout at each lead
    in ``parquet_leads`` (intersected with [1, max_lead_h]).  Returns timing."""
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import (
        _build_real_case, DailyPipelineConfig)
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk, _commit_to_operational_device,
        _enforce_operational_precision, compute_m9_diagnostics)
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    from tost_parquet_scorer import _emit  # local helper (defined below)

    cfg = DailyPipelineConfig(
        run_id=spec["run_id"], run_root=Path(spec["run_root"]),
        domain=spec["domain"], dt_s=dt_s,
        acoustic_substeps=acoustic_substeps,
        radiation_cadence_steps=radiation_cadence_steps)
    case, run_dir = _build_real_case(cfg)
    time_utc = case.run_start
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=radiation_cadence_steps, time_utc=time_utc)

    init_path = Path(spec["run_root"]) / spec["run_id"] / (
        f"wrfout_{spec['domain']}_{time_utc:%Y-%m-%d_%H:%M:%S}")
    ll = read_wrfout_file(init_path, fields=("XLAT", "XLONG"))["fields"]
    lat = np.asarray(ll["XLAT"], dtype=np.float64)
    lon = np.asarray(ll["XLONG"], dtype=np.float64)

    max_lead = int(spec["max_lead_h"])
    leads = sorted(h for h in parquet_leads if 1 <= h <= max_lead)
    if not leads:
        return {"emitted": 0, "leads": [], "note": "no parquet lead in range"}

    dt = float(nl.dt_s)
    cadence = int(nl.radiation_cadence_steps)
    seg = int(segment_steps) if segment_steps else cadence
    lead_steps = {h: int(round(h * 3600.0 / dt)) for h in leads}

    carry = _commit_to_operational_device(initial_operational_carry(
        _enforce_operational_precision(case.state,
                                       force_fp64=bool(nl.force_fp64))))
    timings: dict[int, float] = {}
    t0 = time.time()
    start = 1
    for h in leads:
        target = lead_steps[h]
        while start <= target:
            n = min(seg, target - start + 1)
            carry = _advance_chunk(carry, nl,
                                   jnp.asarray(start, dtype=jnp.int32),
                                   n_steps=int(n), cadence=cadence)
            jax.block_until_ready(carry.state.theta)
            start += n
        diags = compute_m9_diagnostics(carry.state, nl, float(h) * 3600.0)
        valid = time_utc + timedelta(hours=h)
        _emit(out_dir, spec["domain"], valid, lat, lon,
              np.asarray(jax.device_get(diags.t2), dtype=np.float64),
              np.asarray(jax.device_get(diags.u10), dtype=np.float64),
              np.asarray(jax.device_get(diags.v10), dtype=np.float64))
        timings[h] = round(time.time() - t0, 1)
    return {"emitted": len(leads), "leads": leads,
            "cpu_run_dir": str(run_dir), "init_utc": time_utc.isoformat(),
            "max_lead_h": max_lead, "grid": list(lat.shape),
            "timings_cumulative_s": timings}


def main(argv=None) -> int:
    from tost_parquet_scorer import (load_cpu_truth, score_case,
                                     aggregate_tost, MARGINS, MIN_PAIRS_PER_BLOCK,
                                     LEAD_BLOCKS, SCORE_VARS, PARQUET)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", nargs="+", default=None,
                    help="subset of RUNNABLE_UNITS keys")
    ap.add_argument("--out-dir", type=Path,
                    default=HERE / "tost_run" / "parquet_smoke")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--segment-steps", type=int, default=180)
    a = ap.parse_args(argv)

    units = a.units if a.units else DEFAULT_CAMPAIGN_UNITS
    cpu_truth = load_cpu_truth(PARQUET)

    # leads the parquet actually covers per (case, domain)
    def parquet_leads(case_id: str, domain: str) -> list[int]:
        sub = cpu_truth[(cpu_truth["case_id"] == case_id) &
                        (cpu_truth["domain"] == domain)]
        return sorted(int(h) for h in sub["lead_hour"].unique() if h >= 1)

    if a.plan or not a.execute:
        plan = {"mode": "DRY-PLAN (no GPU)", "parquet": str(PARQUET),
                "margins_frozen_adr029": MARGINS,
                "min_pairs_per_block": MIN_PAIRS_PER_BLOCK,
                "lead_blocks": LEAD_BLOCKS,
                "units": []}
        for uid in units:
            spec = RUNNABLE_UNITS[uid]
            pl = parquet_leads(spec["case_id"], spec["domain"])
            scoreable = [h for h in pl if 1 <= h <= spec["max_lead_h"]]
            plan["units"].append({
                "unit_id": uid, "case_id": spec["case_id"],
                "domain": spec["domain"], "init": spec["init_utc"],
                "max_lead_h": spec["max_lead_h"],
                "parquet_leads_covered": len(pl),
                "scoreable_leads": len(scoreable),
                "parquet_stations": int(cpu_truth[
                    (cpu_truth["case_id"] == spec["case_id"]) &
                    (cpu_truth["domain"] == spec["domain"])]
                    ["station_id"].nunique()),
                "parquet_workdir_match": spec.get("parquet_workdir_match"),
                "caveat": spec.get("caveat"),
            })
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "parquet_run_plan.json").write_text(
            json.dumps(plan, indent=2, default=str) + "\n")
        print(json.dumps(plan, indent=2, default=str))
        return 0

    score_kwargs = dict(dt_s=a.dt_s, acoustic_substeps=a.acoustic_substeps,
                        radiation_cadence_steps=a.radiation_cadence_steps,
                        segment_steps=a.segment_steps)
    emit_root = a.out_dir / "gpu_wrfout_parquet"
    t0 = time.time()
    case_scores = []
    per_unit_meta = []
    for uid in units:
        spec = RUNNABLE_UNITS[uid]
        pl = parquet_leads(spec["case_id"], spec["domain"])
        print(f"=== GPU forecast {uid} ({spec['init_utc']}, "
              f"max_lead={spec['max_lead_h']}h, {len(pl)} parquet leads) ===",
              flush=True)
        emit_dir = emit_root / uid
        meta = run_gpu_unit(spec, emit_dir, pl, **score_kwargs)
        per_unit_meta.append({"unit_id": uid, **{k: v for k, v in meta.items()}})
        if meta["emitted"] == 0:
            print(f"    {uid}: nothing emitted — skipped", flush=True)
            continue
        cs = score_case(spec["case_id"], spec["domain"], emit_dir, cpu_truth)
        (a.out_dir / f"paired_parquet_{uid}.json").write_text(
            json.dumps(cs, indent=2, default=str) + "\n")
        case_scores.append(cs)
        agg = aggregate_tost(case_scores)
        (a.out_dir / "tost_parquet.json").write_text(json.dumps({
            "schema": "V020ParquetTostCampaign", "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "truth_source": str(PARQUET),
            "truth_kind": "31-case CPU-WRF point-shadow parquet "
                          "(nearest_grid_cell), 278 stations, 4 nests, 0% NaN",
            "units_run": units, "per_unit_meta": per_unit_meta,
            "margins_frozen_adr029": MARGINS, "tost_aggregate": agg,
            "verdict": ("EQUIVALENT" if agg["all_variables_equivalent"]
                        else "NOT_EQUIVALENT_OR_UNDERPOWERED"),
        }, indent=2, default=str) + "\n")
        print(f"    {uid}: pairs={cs['total_complete_pairs']} "
              f"n_so_far={agg['n_cases']}", flush=True)

    print(f"DONE -> {a.out_dir / 'tost_parquet.json'} "
          f"wall={round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
