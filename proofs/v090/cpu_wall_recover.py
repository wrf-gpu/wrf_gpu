#!/usr/bin/env python
"""Recover 28-rank CPU-WRF real wall-clock from EXISTING run artifacts.

NO new CPU runs. READ-ONLY analysis of finished L2 (9/3 km nest) and L3
(5-domain incl. 1 km) WRF run dirs under /mnt/data/canairy_meteo/runs.

Two independent wall-clock derivations per run:
  (1) file-mtime span  = mtime(rsl.out.0000) - mtime(freezeH2O.dat)
      freezeH2O.dat is loaded by Thompson MP init immediately before WRF starts
      time-stepping; rsl.out.0000 is the rank-0 log, closed at SUCCESS COMPLETE.
      This is the TRUE command-to-finish wall-clock for the whole nest.
  (2) per-step 'Timing for main on domain N: X elapsed seconds' lines, per domain.
      The MEDIAN per-step (robust to contention spikes) x steps/fc-hr gives the
      domain's OWN-solver clean-compute cost, isolated by domain id. This is the
      ONLY honest per-domain figure for a single-domain GPU comparison (the GPU
      port runs one domain, not the whole nest).

Honesty: these CPU runs were produced on a CONTENDED workstation (concurrent GPU
+ backfill). On contended days method (2)-sum EXCEEDS method (1)-span because the
per-step wall timer counts descheduled time. We therefore use the per-step MEDIAN
(not sum) for the per-domain cost, and select the FASTEST complete (least-
contended) run for the conservative speedup denominator (a slow contended CPU run
would INFLATE the speedup -- we refuse that).

Resource: pin to cores 0-3; never touch cores 4-31 (live CPU-WRF backfill).
NEVER read the live backfill staging dir (it is being written).

Usage:
  taskset -c 0-3 python3 proofs/v090/cpu_wall_recover.py [--json OUT.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

TIMING_MAIN = re.compile(r"Timing for main:.*on domain\s+(\d+):\s+([0-9.]+)\s+elapsed")
TIMING_WRITE = re.compile(r"Timing for Writing.*for domain\s+(\d+):\s+([0-9.]+)\s+elapsed")

# Per-level inner timestep (s) by domain id and forecast length (h).
L2_DT = {1: 18, 2: 6}
L3_DT = {1: 18, 2: 6, 3: 2, 4: 2, 5: 2}


def _pin_cpus() -> list[int] | None:
    if not hasattr(os, "sched_setaffinity"):
        return None
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass
    return sorted(os.sched_getaffinity(0))


def analyze_run(run_dir: str, dt_by_dom: dict[int, int], fc_hours: int) -> dict | None:
    rsl = os.path.join(run_dir, "rsl.out.0000")
    fz = os.path.join(run_dir, "freezeH2O.dat")
    if not os.path.exists(rsl):
        return None
    main_by_dom: dict[int, list[float]] = {}
    write_by_dom: dict[int, list[float]] = {}
    success = False
    with open(rsl, "r", errors="replace") as fh:
        for line in fh:
            if "SUCCESS COMPLETE WRF" in line:
                success = True
            m = TIMING_MAIN.search(line)
            if m:
                main_by_dom.setdefault(int(m.group(1)), []).append(float(m.group(2)))
                continue
            w = TIMING_WRITE.search(line)
            if w:
                write_by_dom.setdefault(int(w.group(1)), []).append(float(w.group(2)))

    span = None
    if os.path.exists(fz):
        span = round(os.path.getmtime(rsl) - os.path.getmtime(fz), 1)

    per_domain: dict[str, dict] = {}
    for d, vals in sorted(main_by_dom.items()):
        clean = vals[1:] if len(vals) > 1 else vals  # drop hour-1 compile/IO spike
        dt = dt_by_dom.get(d)
        steps_per_fc_hr = (3600.0 / dt) if dt else None
        med = statistics.median(clean)
        per_domain[f"d{d:02d}"] = {
            "domain_id": d,
            "dt_s": dt,
            "n_steps": len(vals),
            "median_s_per_step": round(med, 5),
            "s_per_fc_hr_median": round(med * steps_per_fc_hr, 1) if steps_per_fc_hr else None,
            "total_compute_s": round(sum(vals), 1),
            "total_compute_s_per_fc_hr": round(sum(vals) / fc_hours, 1),
        }
    return {
        "run_id": os.path.basename(run_dir.rstrip("/")),
        "run_dir": run_dir,
        "success_complete": success,
        "forecast_length_h": fc_hours,
        "full_nest_wall_s_file_mtime_span": span,
        "full_nest_wall_human": None if span is None else f"{int(span // 3600)}h{int((span % 3600) // 60)}m",
        "full_nest_s_per_fc_hr_file_mtime_CONTEXT_ONLY": round(span / fc_hours, 1) if span else None,
        "per_domain": per_domain,
        "n_write_lines": int(sum(len(v) for v in write_by_dom.values())),
    }


def scan_level(level: str, dt_by_dom: dict[int, int], fc_hours: int, limit: int | None = None) -> list[dict]:
    base = f"/mnt/data/canairy_meteo/runs/{level}"
    out: list[dict] = []
    for rd in sorted(glob.glob(f"{base}/*/")):
        rd = rd.rstrip("/")
        res = analyze_run(rd, dt_by_dom, fc_hours)
        if res:
            out.append(res)
    # rank complete runs by full-nest wall (least contended first)
    complete = [r for r in out if r["success_complete"] and r["full_nest_wall_s_file_mtime_span"]]
    complete.sort(key=lambda r: r["full_nest_wall_s_file_mtime_span"])
    if limit:
        complete = complete[:limit]
    return complete


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="write full result JSON here")
    ap.add_argument("--top", type=int, default=4, help="report N least-contended complete runs per level")
    args = ap.parse_args(argv)
    affinity = _pin_cpus()

    l2 = scan_level("wrf_l2", L2_DT, 72, limit=args.top)
    l3 = scan_level("wrf_l3", L3_DT, 24, limit=args.top)

    def denom(runs: list[dict], dom_key: str) -> dict:
        meds = [r["per_domain"][dom_key]["s_per_fc_hr_median"] for r in runs if dom_key in r["per_domain"]]
        return {
            "conservative_low_s_per_fc_hr": round(min(meds), 1) if meds else None,
            "midpoint_s_per_fc_hr": round(statistics.median(meds), 1) if meds else None,
            "realistic_high_s_per_fc_hr": round(max(meds), 1) if meds else None,
        }

    payload = {
        "schema": "GpuwrfV090CpuWallRecover",
        "schema_version": 1,
        "orchestration_cpu_affinity": affinity,
        "no_new_cpu_runs": True,
        "nested_9_3km_d02_denominator": denom(l2, "d02"),
        "single_1km_d03_denominator": denom(l3, "d03"),
        "l2_least_contended_complete_runs": l2,
        "l3_least_contended_complete_runs": l3,
    }
    text = json.dumps(payload, indent=2)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
