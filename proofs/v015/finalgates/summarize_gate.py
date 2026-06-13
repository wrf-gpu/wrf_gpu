#!/usr/bin/env python3
"""Summarize a v0.15 final-gate region: per-field overall rmse vs frozen limit,
h72 value, X/10 hard-gate verdict, stability. Reads the grid_compare.json.

Usage: summarize_gate.py <grid_compare.json> <region_label>
"""
import json
import sys

HARD = ["T", "U", "V", "W", "QVAPOR", "T2", "U10", "V10", "PSFC", "RAINNC"]


def main():
    path, label = sys.argv[1], sys.argv[2]
    d = json.load(open(path))
    fs = d["field_summaries"]
    print(f"=== {label} ===")
    print(f"paired_files={d['pairing'].get('paired_file_count')} "
          f"top_verdict={d.get('summaries',{}).get('verdict') or d.get('verdict')}")
    passes = 0
    rows = []
    worst_ratio = 0.0
    worst_field = None
    for f in HARD:
        v = fs.get(f)
        if not v:
            rows.append((f, "MISSING", None, None, None, False))
            continue
        ov = v.get("overall", {})
        tol = v.get("tolerance_result", {})
        rmse = ov.get("rmse")
        spec = tol.get("spec", {})
        limit = spec.get("rmse")
        passed = tol.get("pass", False)
        bylead = v.get("by_lead", [])
        h72 = bylead[-1]["rmse"] if isinstance(bylead, list) and bylead else None
        fpf = ov.get("finite_pair_fraction")
        if passed:
            passes += 1
        ratio = (rmse / limit) if (rmse is not None and limit) else None
        if ratio is not None and ratio > worst_ratio:
            worst_ratio, worst_field = ratio, f
        rows.append((f, "PASS" if passed else "FAIL", rmse, limit, h72, fpf))
    print(f"{'field':8} {'verdict':7} {'overall_rmse':>14} {'limit':>10} "
          f"{'h72_rmse':>14} {'ratio':>7} finite")
    for (f, vd, rmse, limit, h72, fpf) in rows:
        rr = (rmse / limit) if (rmse is not None and limit) else None
        print(f"{f:8} {vd:7} "
              f"{(f'{rmse:.6g}' if rmse is not None else 'n/a'):>14} "
              f"{(f'{limit:.6g}' if limit is not None else 'n/a'):>10} "
              f"{(f'{h72:.6g}' if h72 is not None else 'n/a'):>14} "
              f"{(f'{rr:.3f}' if rr is not None else 'n/a'):>7} "
              f"{fpf}")
    print(f"\nHARD-GATE VERDICT: {passes}/{len(HARD)} within frozen tolerance")
    if worst_field:
        print(f"WORST hard field: {worst_field} ratio={worst_ratio:.3f}x its limit")
    # finiteness / stability
    nonfinite = [f for f, v in fs.items()
                 if v.get("overall", {}).get("finite_pair_fraction", 1.0) not in (None, 1.0)]
    print(f"hard fields with finite_pair_fraction<1: "
          f"{[f for f in HARD if (fs.get(f,{}).get('overall',{}).get('finite_pair_fraction') not in (None,1.0))] or 'none'}")


if __name__ == "__main__":
    main()
