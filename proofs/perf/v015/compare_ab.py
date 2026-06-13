#!/usr/bin/env python
"""Compare two probe_ab_identity runs: bit-identity verdict + wall delta.

Usage: python compare_ab.py ab_v014_base.json ab_streamA.json
Writes ab_compare_<a>_vs_<b>.json next to the inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
    A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
    mismatches = []
    hours = sorted(set(A["hashes"]) & set(B["hashes"]))
    for h in hours:
        ha, hb = A["hashes"][h], B["hashes"][h]
        for leaf in sorted(set(ha) | set(hb)):
            if ha.get(leaf) != hb.get(leaf):
                mismatches.append({"hour": h, "leaf": leaf, "a": ha.get(leaf), "b": hb.get(leaf)})
    out = {
        "schema": "V015ABCompare",
        "a": A["tag"],
        "b": B["tag"],
        "hours_compared": hours,
        "leaves_per_hour": {h: len(A["hashes"][h]) for h in hours},
        "bit_identical": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches_first20": mismatches[:20],
        "wall_a_per_hour_s": A["per_hour_wall_s"],
        "wall_b_per_hour_s": B["per_hour_wall_s"],
        "steady_a_ms_per_step": A.get("steady_ms_per_step"),
        "steady_b_ms_per_step": B.get("steady_ms_per_step"),
        "host_overhead_a": A.get("host_overhead"),
        "host_overhead_b": B.get("host_overhead"),
    }
    path = HERE / f"ab_compare_{A['tag']}_vs_{B['tag']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if "mismatches_first20" != k}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
