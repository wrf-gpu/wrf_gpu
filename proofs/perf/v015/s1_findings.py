#!/usr/bin/env python
"""v0.15 S1 — collate the host-removal A/B matrix into one findings artifact.

Reads the ab_s1_*.json runs (+ optional nsys summaries + tiered gate) and
writes s1_host_removal_findings.json with walls, speedups vs the v0.14
baseline, hash-identity classes, and env provenance per variant.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_TAG = "s1_base"


def load(tag):
    p = HERE / f"ab_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def hash_mismatches(a, b):
    if not a or not b:
        return None
    tot = sum(len(v) for v in a["hashes"].values())
    mism = sum(
        1
        for h in a["hashes"]
        for k in a["hashes"][h]
        if b["hashes"].get(h, {}).get(k) != a["hashes"][h][k]
    )
    return {"mismatch": mism, "total": tot}


def main() -> int:
    base = load(BASE_TAG)
    base_ms = base["steady_ms_per_step"] if base else None
    rows = {}
    for p in sorted(HERE.glob("ab_s1_*.json")):
        tag = p.stem[3:]
        d = load(tag)
        if not d:
            continue
        row = {
            "steady_ms_per_step": d["steady_ms_per_step"],
            "per_hour_wall_s": d["per_hour_wall_s"],
            "speedup_vs_s1_base": round(base_ms / d["steady_ms_per_step"], 3) if base_ms else None,
            "env": {k: v for k, v in (d.get("env") or {}).items() if v},
        }
        hm = hash_mismatches(base, d)
        if hm is not None and tag != BASE_TAG:
            row["leaf_hash_vs_base"] = hm
        rows[tag] = row

    extras = {}
    for name in ("nsys_summary.json", "cond_niter_oracle.json", "tiered_gate_flat16_cbwhile.json",
                 "tiered_gate_combined.json"):
        p = HERE / name
        if p.exists():
            extras[name] = json.loads(p.read_text())

    cpu_ms = 200.5  # 24-rank CPU-WRF mainloop ms/step on this case (run_h36/cpu_timing.json)
    payload = {
        "schema": "V015S1HostRemovalFindings",
        "case": "Switzerland d01 reinit-h36, 128x128x44, dt=18s, force_fp64, RTX 5090",
        "cpu_denominator_ms_per_step": cpu_ms,
        "variants": rows,
        "speedups_vs_cpu": {
            t: round(cpu_ms / r["steady_ms_per_step"], 2) for t, r in rows.items()
        },
        "attached": sorted(extras.keys()),
    }
    out = HERE / "s1_host_removal_findings.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
