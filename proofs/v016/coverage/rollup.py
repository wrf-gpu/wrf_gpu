#!/usr/bin/env python
"""v0.16 STABILITY — coverage roll-up: aggregate all L2 gate verdicts.

Joins the recomputable coverage map (proofs/v016/coverage_map.json, the L2-target
set) with the per-scheme gate verdicts (coverage/<fam><opt>_gate.json) and reports
which L2 targets are GREEN, which are still PENDING, and an overall verdict.  CPU.

Run:  PYTHONPATH=src python proofs/v016/coverage/rollup.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V016 = HERE.parent

# family token used in gate filenames <token><opt>_gate.json -> registry family key
TOKEN_TO_FAMILY = {
    "mp": "mp_physics", "pbl": "bl_pbl_physics", "sfclay": "sf_sfclay_physics",
    "cu": "cu_physics", "sw": "ra_sw_physics", "lw": "ra_lw_physics",
    "lsm": "sf_surface_physics",
}
FAMILY_TO_TOKEN = {v: k for k, v in TOKEN_TO_FAMILY.items()}


def main() -> int:
    cov = json.loads((V016 / "coverage_map.json").read_text())
    l2_targets = [(r["family"], r["option"]) for r in cov["rows"] if r["l2_target"]]

    gates = {}
    for f in sorted(HERE.glob("*_gate.json")):
        d = json.loads(f.read_text())
        # SCOPED_CARRY records never ran the forecast -> no field deltas; tolerate.
        fd = d.get("field_deltas_vs_baseline") or {}
        gates[(TOKEN_TO_FAMILY[d["family"]], int(d["option"]))] = {
            "verdict": d["verdict"], "all_finite": d.get("all_finite"),
            "peak_vram_gib": d.get("peak_vram_gib"),
            "worst_dyn_over_limit": fd.get("worst_dynamics_rmse_over_limit"),
            "file": f.name,
        }

    green, review, pending, fail, carry = [], [], [], [], []
    for fam, opt in l2_targets:
        g = gates.get((fam, opt))
        token = FAMILY_TO_TOKEN.get(fam, fam)
        tag = f"{token}{opt}"
        if g is None:
            pending.append(tag)
        elif g["verdict"] == "PASS":
            green.append(tag)
        elif g["verdict"] == "REVIEW":
            review.append(tag)
        elif g["verdict"] in ("SCOPED_CARRY", "CARRY"):
            carry.append(tag)
        else:
            fail.append(tag)

    # ALL_GREEN_OR_CARRIED = every L2 target is either coupled-green or an
    # explicit, documented scope-carry; no silent pending and no unexplained FAIL.
    if pending or fail or review:
        overall = "FAIL" if fail else "IN_PROGRESS"
    elif carry:
        overall = "ALL_GREEN_OR_CARRIED"
    else:
        overall = "ALL_GREEN"
    payload = {
        "schema": "V016CoverageRollup",
        "l2_target_count": len(l2_targets),
        "tested_green": sorted(green),
        "tested_review": sorted(review),
        "tested_fail": sorted(fail),
        "scoped_carry": sorted(carry),
        "pending": sorted(pending),
        "n_green": len(green), "n_review": len(review), "n_fail": len(fail),
        "n_carry": len(carry), "n_pending": len(pending),
        "gate_details": {f"{FAMILY_TO_TOKEN.get(k[0],k[0])}{k[1]}": v for k, v in sorted(gates.items())},
        "overall": overall,
    }
    out = HERE / "coverage_rollup.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "gate_details"}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
