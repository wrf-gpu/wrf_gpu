#!/usr/bin/env python3
"""M20/M21 paired-TOST ENSEMBLE RUNNER (ADR-029) — CPU-orchestrated, GPU-sequenced.

WHAT THIS IS
------------
The end-to-end harness that turns the corpus + GPU model into a paired-TOST
equivalence verdict on T2/U10/V10 vs CPU-WRF. It is the runnable join of three
pieces that already exist but were never wired together:

  1. GPU forecast  — the SAME carry-advance loop the v0.1.0 d02 validator uses
     (gpuwrf.integration.daily_pipeline._build_real_case + operational_mode
     _advance_chunk), proven at proofs/v010_validation/v010_d02_validate.py.
  2. GPU wrfout emit — writes the GPU T2/U10/V10 (+ XLAT/XLONG) at each scoreable
     lead into a minimal NetCDF ``wrfout_d02_<valid>`` file, so the frozen M7
     station interpolator can read it with NO model-code change.
  3. Station-paired TOST — proofs/m20/paired_tost_scorer.py forms CPU/GPU/obs
     COMPLETE PAIRS on the same station x valid_time mask, computes per-case
     paired delta RMSE_GPU-RMSE_CPU, and runs the ADR-029 predeclared-margin TOST.

REUSE-ONLY. Launches NO new WRF runs. CPU-WRF truth = the corpus wrfout_d02
hourly history. GPU init = each run's t=0 wrfout_d02 snapshot (the replay case).

GPU SEQUENCING (manager-run). This script is the ONLY GPU consumer; it advances
ONE forecast carry per (case, level) to the maximum scoreable lead, snapshotting
GPU surface fields at every AEMET-overlapping lead hour. Memory stays bounded to
one segment (block_until_ready between segments). The manager runs it AFTER the
HFX+precip physics fixes land, sequenced on the GPU.

USAGE
-----
  # dry plan (CPU only — what WOULD run, no GPU touched):
  PYTHONPATH=src taskset -c 0-3 python proofs/m20/tost_ensemble_runner.py \
      --manifest proofs/m20/tost_corpus_manifest.json --plan

  # full GPU run + paired-TOST aggregate (manager, sequenced on GPU):
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
    python proofs/m20/tost_ensemble_runner.py \
      --manifest proofs/m20/tost_corpus_manifest.json \
      --out-dir proofs/m20/tost_run \
      --execute

The manifest schema is identical to proofs/v010_validation/v010_cases_manifest.json
(``cases[*].levels[L2|L3]`` each with run_root/run_id/run_dir/domain/
gpu_init_source/max_lead_h). Default manifest = that v0.1.0 manifest (3 MAM days);
swap in tost_corpus_manifest.json once the corpus is backfilled to n>=15.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROOF = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "proofs/v010_validation/v010_cases_manifest.json"

SCORE_VARS = ("T2", "U10", "V10")
# AEMET hourly obs begin 2026-05-11 09z (station_join_manifest.json). A lead hour
# is scoreable only if its valid time has hourly obs; the paired scorer enforces
# the >=30-pair-per-block exclusion, but we avoid emitting frames with zero obs.
OBS_START_UTC = datetime(2026, 5, 11, 9, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# 1+2. GPU forecast for one (case, level) -> emit GPU wrfout NetCDF per lead.
# ---------------------------------------------------------------------------
def _emit_gpu_wrfout(out_dir: Path, domain: str, valid: datetime,
                     lat: np.ndarray, lon: np.ndarray,
                     t2: np.ndarray, u10: np.ndarray, v10: np.ndarray) -> Path:
    """Write a minimal WRF-style NetCDF the frozen M7 interpolator can read.

    The scorer parses the valid time from the FILENAME and reads XLAT/XLONG/T2/
    U10/V10 as 2D fields. We write exactly those, with a leading Time axis so the
    loader's Time-squeeze path is exercised (matches real wrfout layout)."""
    from netCDF4 import Dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"wrfout_{domain}_{valid:%Y-%m-%d_%H:%M:%S}"
    ny, nx = np.asarray(t2).shape
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.Times = valid.strftime("%Y-%m-%d_%H:%M:%S")

        def _v(name, arr):
            var = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            var[0, :, :] = np.asarray(arr, dtype=np.float32)

        _v("XLAT", lat); _v("XLONG", lon)
        _v("T2", t2); _v("U10", u10); _v("V10", v10)
    return path


def run_gpu_case_level(level_spec: dict, out_dir: Path, *, dt_s: float,
                       acoustic_substeps: int, radiation_cadence_steps: int,
                       segment_steps: int) -> dict:
    """Advance ONE GPU forecast carry to max scoreable lead, emitting a GPU wrfout
    NetCDF at every integer lead hour that has obs coverage. Returns timing meta."""
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk, _commit_to_operational_device,
        _enforce_operational_precision, compute_m9_diagnostics,
    )
    from gpuwrf.runtime.operational_state import initial_operational_carry

    cfg = DailyPipelineConfig(
        run_id=level_spec["run_id"], run_root=Path(level_spec["run_root"]),
        domain=level_spec["domain"], dt_s=dt_s,
        acoustic_substeps=acoustic_substeps,
        radiation_cadence_steps=radiation_cadence_steps,
    )
    case, run_dir = _build_real_case(cfg)
    time_utc = case.run_start
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=radiation_cadence_steps, time_utc=time_utc,
    )
    init_path = Path(level_spec["gpu_init_source"])
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    ll = read_wrfout_file(init_path, fields=("XLAT", "XLONG"))["fields"]
    lat = np.asarray(ll["XLAT"], dtype=np.float64)
    lon = np.asarray(ll["XLONG"], dtype=np.float64)

    max_lead = int(level_spec["max_lead_h"])
    # Score every integer lead hour whose valid time has obs AND whose CPU-WRF
    # truth frame exists on disk (the paired scorer also requires the CPU frame).
    leads = []
    for h in range(1, max_lead + 1):
        valid = time_utc + timedelta(hours=h)
        if valid < OBS_START_UTC:
            continue
        cpu_frame = run_dir / f"wrfout_{level_spec['domain']}_{valid:%Y-%m-%d_%H:%M:%S}"
        if cpu_frame.is_file():
            leads.append(h)
    if not leads:
        return {"emitted": 0, "leads": [], "note": "no obs-covered lead with CPU truth",
                "out_dir": str(out_dir)}

    dt = float(nl.dt_s)
    cadence = int(nl.radiation_cadence_steps)
    seg = int(segment_steps) if segment_steps else cadence
    lead_steps = {h: int(round(h * 3600.0 / dt)) for h in leads}

    carry = _commit_to_operational_device(initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))))
    timings: dict[int, float] = {}
    t0 = time.time()
    start = 1
    for h in leads:
        target = lead_steps[h]
        while start <= target:
            n = min(seg, target - start + 1)
            carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                                   n_steps=int(n), cadence=cadence)
            jax.block_until_ready(carry.state.theta)
            start += n
        diags = compute_m9_diagnostics(carry.state, nl, float(h) * 3600.0)
        valid = time_utc + timedelta(hours=h)
        _emit_gpu_wrfout(
            out_dir, level_spec["domain"], valid, lat, lon,
            np.asarray(jax.device_get(diags.t2), dtype=np.float64),
            np.asarray(jax.device_get(diags.u10), dtype=np.float64),
            np.asarray(jax.device_get(diags.v10), dtype=np.float64),
        )
        timings[h] = round(time.time() - t0, 1)
    return {"emitted": len(leads), "leads": leads,
            "cpu_run_dir": str(run_dir), "out_dir": str(out_dir),
            "init_utc": time_utc.isoformat(), "max_lead_h": max_lead,
            "timings_cumulative_s": timings}


# ---------------------------------------------------------------------------
# 3. Per-case station-paired scoring (reuse proofs/m20/paired_tost_scorer.py).
# ---------------------------------------------------------------------------
def score_case_level(case_id: str, cpu_run_dir: Path, gpu_out_dir: Path,
                     domain: str, init: datetime, fh: int) -> dict:
    from paired_tost_scorer import paired_score  # local module (same dir)
    from gpuwrf.validation.forecast_vs_obs import DEFAULT_AEMET_ROOT
    return paired_score(case_id, cpu_run_dir, gpu_out_dir, domain, init, fh,
                        DEFAULT_AEMET_ROOT)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def load_cases(manifest_path: Path) -> list[dict]:
    return json.loads(manifest_path.read_text())["cases"]


def expand_units(cases: list[dict]) -> list[dict]:
    """Flatten cases x levels into (case_id, level) scoring units."""
    units = []
    for c in cases:
        for level_name, spec in c["levels"].items():
            init = datetime.fromisoformat(c["init_utc"].replace("Z", "+00:00"))
            units.append({
                "unit_id": f"{c['case_id']}_{level_name}",
                "case_id": c["case_id"], "level": level_name,
                "init_utc": init.isoformat(), "spec": spec,
                "fh": int(spec["max_lead_h"]),
            })
    return units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=PROOF / "tost_run")
    ap.add_argument("--cases", nargs="+", default=None, help="subset of case ids")
    ap.add_argument("--levels", nargs="+", default=None, help="subset of levels, e.g. L2 L3")
    ap.add_argument("--execute", action="store_true", help="LAUNCH GPU forecasts + score")
    ap.add_argument("--plan", action="store_true", help="print the dry plan and exit")
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--segment-steps", type=int, default=180)
    a = ap.parse_args(argv)

    cases = load_cases(a.manifest)
    if a.cases:
        cases = [c for c in cases if c["case_id"] in a.cases]
    units = expand_units(cases)
    if a.levels:
        units = [u for u in units if u["level"] in a.levels]

    a.out_dir.mkdir(parents=True, exist_ok=True)

    if a.plan or not a.execute:
        plan = {
            "mode": "DRY-PLAN (no GPU touched)",
            "manifest": str(a.manifest),
            "n_cases": len(cases),
            "n_scoring_units": len(units),
            "units": [{"unit_id": u["unit_id"], "init_utc": u["init_utc"],
                       "fh": u["fh"], "level": u["level"],
                       "cpu_run_dir": u["spec"]["run_dir"],
                       "init_exists": Path(u["spec"]["gpu_init_source"]).is_file()}
                      for u in units],
            "note": "Pass --execute to run GPU forecasts + paired TOST. "
                    "n_scoring_units is the TOST n (per variable) IF every unit "
                    "yields >=1 obs-covered lead block with >=30 pairs.",
        }
        print(json.dumps(plan, indent=2))
        (a.out_dir / "tost_run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        return 0

    from paired_tost_scorer import aggregate_tost  # noqa: E402

    score_kwargs = dict(dt_s=a.dt_s, acoustic_substeps=a.acoustic_substeps,
                        radiation_cadence_steps=a.radiation_cadence_steps,
                        segment_steps=a.segment_steps)
    t0 = time.time()
    case_scores: list[dict] = []
    per_unit_meta: list[dict] = []
    for u in units:
        print(f"=== GPU forecast {u['unit_id']} ({u['init_utc']}, fh={u['fh']}) ===",
              flush=True)
        gpu_out = a.out_dir / "gpu_wrfout" / u["unit_id"]
        meta = run_gpu_case_level(u["spec"], gpu_out, **score_kwargs)
        per_unit_meta.append({"unit_id": u["unit_id"], **{k: v for k, v in meta.items()
                                                          if k != "out_dir"}})
        if meta["emitted"] == 0:
            print(f"    {u['unit_id']}: no scoreable lead — skipped", flush=True)
            continue
        init = datetime.fromisoformat(u["init_utc"])
        cs = score_case_level(u["unit_id"], Path(meta["cpu_run_dir"]), gpu_out,
                              u["spec"]["domain"], init, u["fh"])
        (a.out_dir / f"paired_score_{u['unit_id']}.json").write_text(
            json.dumps(cs, indent=2, default=str) + "\n")
        case_scores.append(cs)
        # incremental aggregate so a late crash still leaves a partial verdict.
        agg = aggregate_tost(case_scores)
        (a.out_dir / "tost_aggregate.json").write_text(
            json.dumps(agg, indent=2, default=str) + "\n")
        print(f"    {u['unit_id']}: pairs={cs['total_complete_pairs']} "
              f"n_so_far={agg['n_cases']}", flush=True)

    agg = aggregate_tost(case_scores)
    payload = {
        "schema": "M21TostCampaignResult", "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(a.manifest),
        "n_scoring_units_attempted": len(units),
        "n_scoring_units_scored": len(case_scores),
        "per_unit_meta": per_unit_meta,
        "tost_aggregate": agg,
        "wall_s": round(time.time() - t0, 1),
        "verdict": ("EQUIVALENT" if agg["all_variables_equivalent"]
                    else "NOT_EQUIVALENT_OR_UNDERPOWERED"),
        "honesty": ("TOST n per variable = "
                    f"{agg['n_cases']}; ADR-029 requires n>=15 (target n~=27/30) "
                    "for the declared 10% MDE. A single-season corpus cannot claim "
                    "SEASONAL equivalence without stats-reviewer sign-off."),
    }
    out = a.out_dir / "tost_campaign_result.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {out}\nverdict={payload['verdict']} n={agg['n_cases']} "
          f"wall={payload['wall_s']}s")
    return 0 if agg["all_variables_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
