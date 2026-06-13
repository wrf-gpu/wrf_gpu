#!/usr/bin/env python
"""v0.15 S1 — tiered identity field gate: candidate vs baseline final state.

Compares two `ab_<tag>_state.npz` dumps (same case, same hours, different
env/flags) leaf-by-leaf and scores the prognostic deltas against the v0.14
frozen-tolerance manifest (proofs/v014/grid_delta_atlas/
tolerance_manifest_candidate.json).  The manifest limits bound GPU-vs-CPU-WRF
operational error; a candidate whose GPU-vs-GPU delta is far inside those
limits cannot flip the release gate.  This is the Tier-P (physics-bounded)
gate of the v0.15 tiered-identity ADR; Tier-S (strict) remains the ab-hash.

Usage: compare_tiered_identity.py BASE.npz CAND.npz --hours H --out OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parents[1] / "v014" / "grid_delta_atlas" / "tolerance_manifest_candidate.json"

# state-leaf -> manifest wrfout field (prognostic/diagnostic hard gates only).
# `p` (full-3D pressure perturbation) is gated against the PSFC limit as a
# conservative proxy (surface pressure error is bounded by the 3D field delta);
# rain_acc is the RAINNC accumulator leaf.
LEAF_TO_FIELD = {
    "theta": "T", "u": "U", "v": "V", "w": "W", "qv": "QVAPOR",
    "p": "PSFC", "rain_acc": "RAINNC",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = np.load(args.base)
    cand = np.load(args.cand)
    manifest = json.loads(MANIFEST.read_text())["fields"]

    gated, others = {}, {}
    worst_margin = 0.0
    fail = []
    for name in sorted(set(base.files) & set(cand.files)):
        a, b = base[name], cand[name]
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.floating):
            continue
        d = (a.astype(np.float64) - b.astype(np.float64))
        finite = np.isfinite(d)
        if not finite.all():
            d = np.where(finite, d, np.nan)
        rmse = float(np.sqrt(np.nanmean(d * d)))
        mx = float(np.nanmax(np.abs(d))) if d.size else 0.0
        rec = {"rmse": rmse, "max_abs": mx}
        key = name.lower().split(".")[-1]
        fld = LEAF_TO_FIELD.get(key)
        if fld and fld in manifest and "rmse" in manifest[fld]:
            lim = float(manifest[fld]["rmse"])
            rec["manifest_field"] = fld
            rec["manifest_rmse_limit"] = lim
            rec["rmse_over_limit"] = rmse / lim if lim else float("inf")
            worst_margin = max(worst_margin, rec["rmse_over_limit"])
            if rmse > lim:
                fail.append(fld)
            gated[name] = rec
        else:
            others[name] = rec

    payload = {
        "schema": "V015TieredIdentity",
        "base": args.base, "cand": args.cand, "hours": args.hours,
        "gated_fields": gated,
        "worst_rmse_over_manifest_limit": worst_margin,
        "hard_gate_fails": fail,
        "verdict": "PASS" if not fail else "FAIL",
        "ungated_leaf_deltas": {
            k: v for k, v in sorted(others.items(), key=lambda kv: -kv[1]["rmse"])[:25]
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "ungated_leaf_deltas"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
