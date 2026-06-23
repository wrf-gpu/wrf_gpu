#!/usr/bin/env python3
"""v020_run_metrics.py — reduce one forecast run (log + VRAM CSV + optional nsys
stats) into the per-arm metric JSON the G0 verdict consumes.

Pure CPU. Reads:
  --wall-s / --steps      : forecast wall seconds + advance count -> ms_per_step
                            (steps can be read from the proof JSON or passed directly)
  --vram-csv FILE         : nvidia-smi samples -> peak_vram_mib
  --stats-prefix PREFIX   : nsys CSVs -> dominant_kernel_ms (top kernel total time)
  --fit BOOL              : did the run complete without OOM
  --transient-fp32-frac F : (fp32 arm only) share of the transient that is fp32-able,
                            estimated by the driver from the dtype/liveness audit; pass
                            -1 if unknown (verdict treats NaN conservatively)

Emits one arm dict matching the G0 manifest schema (fp64 OR fp32 block).

CPU-dry-runnable: every input is optional; missing inputs yield null fields (the
verdict reducer treats nulls/NaN conservatively, never crashes).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path


def peak_vram_mib(csv_path: str | None) -> float | None:
    if not csv_path or not Path(csv_path).is_file():
        return None
    peak = 0.0
    with open(csv_path, newline="") as fh:
        for row in csv.reader(fh):
            for cell in row:
                m = re.search(r"(\d+(?:\.\d+)?)\s*MiB", cell)
                if m:
                    peak = max(peak, float(m.group(1)))
    return peak or None


def dominant_kernel_ms(stats_prefix: str | None) -> float | None:
    if not stats_prefix:
        return None
    cands = sorted(glob.glob(f"{stats_prefix}*cuda_gpu_kern_sum*.csv"))
    if not cands:
        return None
    best = 0.0
    with open(cands[0], newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return None
    # locate a time column header
    header = None
    hi = 0
    for i, r in enumerate(rows[:5]):
        if any("time" in c.lower() for c in r):
            header = [c.strip().lower() for c in r]; hi = i; break
    if header is None:
        return None
    tcol = next((j for j, c in enumerate(header) if "total time" in c), None)
    if tcol is None:
        tcol = next((j for j, c in enumerate(header) if "time" in c), None)
    if tcol is None:
        return None
    for r in rows[hi + 1:]:
        if tcol < len(r):
            try:
                best = max(best, float(r[tcol].replace(",", "")))
            except ValueError:
                pass
    return (best / 1e6) if best else None  # ns -> ms


def steps_from_proof(proof_dir: str | None) -> int | None:
    if not proof_dir:
        return None
    for p in glob.glob(f"{proof_dir}/**/*.json", recursive=True) + glob.glob(f"{proof_dir}/*.json"):
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        for key in ("advance", "advances", "n_steps", "steps", "total_steps"):
            v = _deep_get(d, key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return None


def _deep_get(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _deep_get(v, key)
            if r is not None:
                return r
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wall-s", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--proof-dir", default=None, help="proof dir to read step count from")
    ap.add_argument("--vram-csv", default=None)
    ap.add_argument("--stats-prefix", default=None)
    ap.add_argument("--fit", default="true", choices=["true", "false"])
    ap.add_argument("--transient-fp32-frac", type=float, default=None,
                    help="fp32 arm only; -1 or omit if unknown")
    ap.add_argument("--bottleneck-moved", default="false", choices=["true", "false"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    steps = args.steps or steps_from_proof(args.proof_dir)
    ms_per_step = None
    if args.wall_s and steps:
        ms_per_step = (args.wall_s * 1000.0) / steps

    arm = {
        "ms_per_step": ms_per_step,
        "wall_s": args.wall_s,
        "steps": steps,
        "peak_vram_mib": peak_vram_mib(args.vram_csv),
        "fit": (args.fit == "true"),
        "dominant_kernel_ms": dominant_kernel_ms(args.stats_prefix),
    }
    if args.transient_fp32_frac is not None and args.transient_fp32_frac >= 0:
        arm["transient_fp32_fraction"] = args.transient_fp32_frac
    if args.bottleneck_moved == "true":
        arm["bottleneck_moved"] = True

    payload = json.dumps(arm, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
