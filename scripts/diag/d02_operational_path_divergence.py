#!/usr/bin/env python
"""DIAGNOSIS: localize the d02 continuous-vs-per-hour operational path divergence.

The v0.1.0 D02_VALIDATED proof used the CONTINUOUS `_advance_chunk` harness
(single carry across the whole run, global step index, `time_utc=run_start`,
`disable_guards=True`).  The production operational product (the wrfouts a user
gets, via `daily_pipeline` -> `run_forecast_operational(state, nl, 1.0)` once per
forecast hour) RE-SEEDS the operational carry every hour AND restarts the global
step index at 1, with `time_utc=None` and `disable_guards=False`.  The SAME
model/HEAD/case scored two ways gives ~2x-different T2 (1.88 vs 3.59 @6h) with a
+2.6 kPa surface-pressure error on the per-hour path.

This script runs the SAME case3 IC forward and isolates EACH difference, all on
ONE compiled `_advance_chunk` program (the per-hour path is emulated EXACTLY by
re-seeding the carry + restarting the step index at each hour boundary; radiation
gating is bit-identical -- cadence 180, 360 steps/hour -> fires at the same global
steps either way).  For each config we report state surface pressure (p[0]) and
theta[0] bias vs the CPU-WRF corpus truth at each hour, so we can say which path
matches truth and WHERE the divergence enters.

Configs (all advance the identical case3 IC, hours configurable):
  CONT    = continuous carry, global step, time_utc=run_start, guards OFF   (= VALIDATED harness)
  PERHOUR = re-seed carry + step->1 each hour, time_utc=None, guards ON     (= production product)
  PH_CLOCK= per-hour reset BUT time_utc=run_start + GLOBAL lead             (isolate the clock)
  PH_GUARDOFF = per-hour reset, time_utc=None, guards OFF                   (isolate guards)
  CONT_NOCLOCK= continuous carry BUT time_utc=None + per-hour-reset lead    (isolate clock on cont path)

CPU-pinned orchestration; ONE short GPU job.  No production src/ edits.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
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

import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
from netCDF4 import Dataset

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3")
CORPUS = RUN_ROOT / RUN_ID
T0 = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)
DT = 10.0
CADENCE = 180
SEG = 180  # one radiation interval per segment


def corpus_fields(hour: int):
    valid = T0 + timedelta(hours=hour)
    p = CORPUS / f"wrfout_d02_{valid.strftime('%Y-%m-%d_%H:%M:%S')}"
    with Dataset(p) as d:
        P = np.asarray(d.variables["P"][0], float)
        PB = np.asarray(d.variables["PB"][0], float)
        T = np.asarray(d.variables["T"][0], float)  # theta perturbation (K), +300 base
        T2 = np.asarray(d.variables["T2"][0], float)
        SW = np.asarray(d.variables["SWDOWN"][0], float) if "SWDOWN" in d.variables else None
    psfc = (P + PB)[0]
    return psfc, (T + 300.0)[0], T2, SW


def device_copy(state):
    return jax.tree_util.tree_map(lambda x: jnp.array(np.asarray(jax.device_get(x))), state)


def advance_config(state0, nl_base, hours, *, reseed_per_hour, clock_threaded, guards_off):
    """Advance `state0` `hours` hours under one config; snapshot p[0]/theta[0] each hour."""
    steps_per_hour = int(round(3600.0 / DT))  # 360
    nl = dataclasses.replace(
        nl_base,
        run_physics=True,
        run_boundary=True,
        disable_guards=bool(guards_off),
        radiation_cadence_steps=CADENCE,
        time_utc=(T0 if clock_threaded else None),
    )
    carry = initial_operational_carry(_enforce_operational_precision(state0, force_fp64=bool(nl.force_fp64)))
    snaps = {}
    global_start = 1  # continuous global step index
    for hour in range(1, hours + 1):
        if reseed_per_hour:
            # EXACT emulation of run_forecast_operational(state, nl, 1.0): re-seed the
            # operational carry from the current state AND restart the step index at 1.
            carry = initial_operational_carry(carry.state)
            local_start = 1
            target = steps_per_hour
            while local_start <= target:
                n = min(SEG, target - local_start + 1)
                carry = _advance_chunk(carry, nl, jnp.asarray(local_start, dtype=jnp.int32),
                                       n_steps=int(n), cadence=CADENCE)
                jax.block_until_ready(carry.state.theta)
                local_start += n
        else:
            # Continuous carry, global step index runs 1..hours*steps_per_hour.
            target = hour * steps_per_hour
            while global_start <= target:
                n = min(SEG, target - global_start + 1)
                carry = _advance_chunk(carry, nl, jnp.asarray(global_start, dtype=jnp.int32),
                                       n_steps=int(n), cadence=CADENCE)
                jax.block_until_ready(carry.state.theta)
                global_start += n
        snaps[hour] = (
            np.asarray(jax.device_get(carry.state.p[0]), float),
            np.asarray(jax.device_get(carry.state.theta[0]), float),
        )
    return snaps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--configs", type=str, nargs="+",
                    default=["CONT", "PERHOUR", "PH_CLOCK", "PH_GUARDOFF"])
    ap.add_argument("--out", type=Path, default=ROOT / "proofs" / "v010_validation" / "path_divergence_case3.json")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(run_id=RUN_ID, run_root=RUN_ROOT, domain="d02",
                              dt_s=DT, acoustic_substeps=10, radiation_cadence_steps=CADENCE)
    case, _ = _build_real_case(cfg)
    nl_base = case.namelist
    state0 = case.state

    CONFIGS = {
        # name:            reseed,  clock,  guards_off
        "CONT":           dict(reseed_per_hour=False, clock_threaded=True,  guards_off=True),
        "PERHOUR":        dict(reseed_per_hour=True,  clock_threaded=False, guards_off=False),
        "PH_CLOCK":       dict(reseed_per_hour=True,  clock_threaded=True,  guards_off=False),
        "PH_GUARDOFF":    dict(reseed_per_hour=True,  clock_threaded=False, guards_off=True),
        "CONT_NOCLOCK":   dict(reseed_per_hour=False, clock_threaded=False, guards_off=True),
    }

    results = {}
    for name in args.configs:
        kw = CONFIGS[name]
        s0 = device_copy(state0)
        snaps = advance_config(s0, nl_base, args.hours, **kw)
        results[name] = snaps
        print(f"[done] config {name}")

    # corpus + report
    hours = sorted(next(iter(results.values())).keys())
    report = {"case": RUN_ID, "dt_s": DT, "cadence": CADENCE, "hours": list(hours), "configs": {}}
    print("\n=== d02 case3 operational-path divergence (state vs CPU-WRF corpus) ===")
    header = f"{'cfg':<14}{'hr':>3}{'psfc_bias_Pa':>14}{'theta0_bias_K':>15}{'T2-ish_K(*)':>13}"
    for name in args.configs:
        report["configs"][name] = {}
        for hour in hours:
            psfc, th0 = results[name][hour]
            cps, cth0, cT2, _ = corpus_fields(hour)
            pbias = float(np.nanmean(psfc - cps))
            thbias = float(np.nanmean(th0 - cth0))
            report["configs"][name][str(hour)] = {
                "psfc_bias_Pa": pbias,
                "psfc_max_abs_Pa": float(np.nanmax(np.abs(psfc - cps))),
                "theta0_bias_K": thbias,
                "theta0_max_abs_K": float(np.nanmax(np.abs(th0 - cth0))),
            }
        print()
        print(header)
        for hour in hours:
            r = report["configs"][name][str(hour)]
            print(f"{name:<14}{hour:>3}{r['psfc_bias_Pa']:>14.1f}{r['theta0_bias_K']:>15.3f}{'':>13}")

    # explicit pairwise divergence CONT vs PERHOUR
    if "CONT" in results and "PERHOUR" in results:
        print("\n=== CONT vs PERHOUR pairwise STATE divergence (first hour it appears) ===")
        for hour in hours:
            pA, thA = results["PERHOUR"][hour]
            pB, thB = results["CONT"][hour]
            print(f"hour {hour:>2}: psfc(PERHOUR-CONT) mean {np.nanmean(pA-pB):+9.1f} Pa  "
                  f"max {np.nanmax(np.abs(pA-pB)):8.1f}   theta0 mean {np.nanmean(thA-thB):+7.3f} K  "
                  f"max {np.nanmax(np.abs(thA-thB)):6.3f}")

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
