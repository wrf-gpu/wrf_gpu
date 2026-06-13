#!/usr/bin/env python
"""Aggregate an nsys nvtx_gpu_proj_sum CSV into model-phase GPU time.

Usage: python analyze_nvtx_phases.py <nvtx_csv> <steps> <out_json>
Leaf thunks only (container `while.*` ranges skipped to avoid double counting).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def main() -> int:
    path, steps, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    rows = []
    with open(path) as f:
        rdr = csv.reader(f)
        hdr = None
        for r in rdr:
            if not r:
                continue
            if hdr is None:
                if any("Range" in c for c in r):
                    hdr = r
                continue
            rows.append(dict(zip(hdr, r)))
    durcol = next(c for c in hdr if "Total" in c and "Proj" in c)

    def fnum(x):
        try:
            return float(str(x).replace(",", ""))
        except Exception:
            return 0.0

    cats: dict[str, float] = {}
    total = 0.0
    for r in rows:
        name = r.get("Range", "")
        if not name.startswith("TSL:Thunk") or re.search(r"hlo_op=while\.", name):
            continue
        d = fnum(r.get(durcol))
        total += d
        if "step_mynn_pbl_column" in name:
            cat = "MYNN_PBL(EDMF)"
        elif "thompson" in name.lower():
            cat = "Thompson"
        elif "rrtmg" in name.lower() or "radiation" in name.lower():
            cat = "RRTMG"
        elif "noahmp" in name.lower() or "sfclay" in name.lower() or "surface" in name.lower():
            cat = "Surface"
        elif re.search(r"closed_call/while/body", name):
            cat = "dycore_inner_scans"
        else:
            cat = "step_body_other"
        cats[cat] = cats.get(cat, 0.0) + d

    payload = {
        "schema": "V015NvtxPhases",
        "source": str(path),
        "steps": steps,
        "phases_ms": {k: round(v / 1e6, 1) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
        "phases_ms_per_step": {k: round(v / 1e6 / steps, 2) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
        "total_ms": round(total / 1e6, 1),
        "total_ms_per_step": round(total / 1e6 / steps, 2),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
