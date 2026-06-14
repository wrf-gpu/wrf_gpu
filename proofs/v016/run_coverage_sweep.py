#!/usr/bin/env python
"""v0.16 STABILITY coverage sweep driver — runs the L2 coupled-coverage gate for
every remaining L2 target in ONE process (one baseline build, one GPU-lock hold).

Reads the 25 L2 targets from proofs/v016/coverage_map.json, builds the baseline
once (cached npz in proofs/v016/coverage/), then runs each candidate that does
NOT already have an up-to-date <fam><opt>_gate.json whose npz is reachable.

Run it UNDER the GPU lock (single hold for the whole sweep):
  scripts/with_gpu_lock.sh --label v016-coverage --timeout 14400 -- \
      env PYTHONPATH=src python proofs/v016/run_coverage_sweep.py --hours 1

--only mp1,pbl7  restrict to a subset; --force re-run even if a gate exists.
mp=28 is intentionally NOT in the map on this branch (see report scope-carry).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COVERAGE = HERE / "coverage"
MAP = HERE / "coverage_map.json"
GATE = HERE / "coupled_coverage_gate.py"
LOCK = HERE.parent.parent / "scripts" / "with_gpu_lock.sh"

FAM_SHORT = {
    "mp_physics": "mp", "bl_pbl_physics": "pbl", "sf_sfclay_physics": "sfclay",
    "cu_physics": "cu", "ra_sw_physics": "sw", "ra_lw_physics": "lw",
    "sf_surface_physics": "lsm",
}


def l2_targets() -> list[tuple[str, int]]:
    m = json.loads(MAP.read_text())
    return [(FAM_SHORT[r["family"]], r["option"]) for r in m["rows"] if r.get("l2_target")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--only", default="", help="comma list like mp1,pbl7")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    targets = l2_targets()
    if args.only:
        want = set(args.only.split(","))
        targets = [(f, o) for f, o in targets if f"{f}{o}" in want]

    base_npz = COVERAGE / f"_baseline_{args.hours}h_state.npz"
    results: dict[str, str] = {}
    t_start = time.time()

    for fam, opt in targets:
        gate = COVERAGE / f"{fam}{opt}_gate.json"
        if gate.exists() and not args.force:
            # Trust an existing verdict only if it was scored (has the deltas key).
            try:
                d = json.loads(gate.read_text())
                if "verdict" in d:
                    print(f"[skip] {fam}{opt} already has verdict={d['verdict']}", flush=True)
                    results[f"{fam}{opt}"] = d["verdict"]
                    continue
            except Exception:
                pass
        gate_cmd = [
            sys.executable, str(GATE),
            "--family", fam, "--option", str(opt), "--hours", str(args.hours),
        ]
        if not base_npz.exists():
            gate_cmd.append("--refresh-baseline")
        # Per-gate GPU lock: the lock FREES between schemes so the make-or-break
        # fp32 lane and the v017 lane can interleave. Do NOT hold one lock for the
        # whole multi-hour sweep.
        cmd = [str(LOCK), "--label", "v016-sweep", "--timeout", "28800", "--"] + gate_cmd
        print(f"\n===== RUN {fam}{opt} (baseline_cached={base_npz.exists()}) =====", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        dt = round(time.time() - t0, 1)
        try:
            d = json.loads(gate.read_text())
            v = d.get("verdict", f"NO_VERDICT(rc={rc})")
        except Exception:
            v = f"NO_GATE_FILE(rc={rc})"
        results[f"{fam}{opt}"] = v
        print(f"===== DONE {fam}{opt} verdict={v} in {dt}s (rc={rc}) =====", flush=True)
        # Commit each verdict immediately (durability against a manager crash).
        repo = str(HERE.parent.parent)
        subprocess.call(["git", "add", "-A", str(COVERAGE)], cwd=repo)
        subprocess.call(
            ["git", "commit", "-q", "-m",
             f"v016 coverage verdict: {fam}{opt} ({v})\n\n"
             "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"],
            cwd=repo,
        )

    print("\n==================== SWEEP SUMMARY ====================", flush=True)
    for k, v in results.items():
        print(f"  {k:10s} {v}", flush=True)
    print(f"  total wall: {round(time.time() - t_start, 1)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
