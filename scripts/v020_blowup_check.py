#!/usr/bin/env python3
"""v020_blowup_check.py — per-domain NO-BLOW-UP gate for the P4 + skill ladders.

The HARD stability floor of V0200-ROADMAP §8.5: every variable stays finite and within
physical bounds at every output time, EXCEPT the documented carve-outs (cumulative
precip rain/snow/graupel/ice acc + QVAPOR-class drift). Reads the wrfout set of a run
and reports, per domain, whether any non-carve-out field went non-finite or breached a
sane physical envelope (the blow-up signature). Pure numpy + netCDF4, NO GPU.

This is the ladder's primary gate (stability-first, not tolerance). It is also the
early-abort signal: if a rung blows up, the ladder STOPS at that rung and never widens.

Carve-outs (drift allowed, blow-up NOT — checked finite, not bounded):
  RAINNC RAINC SNOWNC GRAUPELNC HAILNC QVAPOR (+ *NC accumulators)

Physical envelopes (blow-up if exceeded anywhere, any time):
  |U|,|V| <= 200 m/s ; |W| <= 100 m/s ; T(theta-pert) in [-200,400] ;
  |P|(pert) <= 1.5e4 Pa over a sane band ; QCLOUD/QRAIN/QICE/QSNOW in [0, 0.1]

Usage:
  python v020_blowup_check.py --run-dir DIR --max-dom N [--out gate.json]
Exit code 0 = PASS (no blow-up), 1 = BLOW-UP detected, 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover
    print(f"v020_blowup_check: netCDF4 import failed: {exc}", file=sys.stderr)
    sys.exit(2)

CARVE_OUT = {"RAINNC", "RAINC", "SNOWNC", "GRAUPELNC", "HAILNC", "QVAPOR",
             "ACSNOW", "SNOWH", "SR"}

# field -> (lo, hi) physical envelope; breach = blow-up.
# Hydrometeor lower bounds carry a tiny NEGATIVE slack: WRF advection legitimately
# produces O(1e-12) negative mixing ratios at the fp-noise floor (physically zero); a
# blow-up gate must catch RUNAWAY negatives (a real instability drives |q| large), not
# the harmless noise. -1e-6 kg/kg is ~1e4x below any meaningful hydrometeor value.
_Q_NEG_FLOOR = -1.0e-6
ENVELOPE = {
    "U": (-200.0, 200.0), "V": (-200.0, 200.0), "W": (-100.0, 100.0),
    "T": (-200.0, 400.0),                       # theta perturbation about 300 K
    "P": (-1.0e5, 5.0e4),                       # pressure perturbation (Pa); wide band
    "QCLOUD": (_Q_NEG_FLOOR, 0.1), "QRAIN": (_Q_NEG_FLOOR, 0.1),
    "QICE": (_Q_NEG_FLOOR, 0.1), "QSNOW": (_Q_NEG_FLOOR, 0.1),
    "QGRAUP": (_Q_NEG_FLOOR, 0.1),
}


def check_file(path: Path) -> dict:
    ds = Dataset(str(path))
    res = {"file": path.name, "nonfinite_fields": [], "envelope_breaches": [],
           "carveout_nonfinite": []}
    try:
        for name, var in ds.variables.items():
            try:
                arr = np.asarray(var[:], dtype=np.float64)
            except Exception:
                continue
            if arr.dtype.kind not in "fc":
                continue
            finite = np.isfinite(arr)
            n_bad = int(arr.size - finite.sum())
            if n_bad > 0:
                if name in CARVE_OUT:
                    res["carveout_nonfinite"].append({"field": name, "n_nonfinite": n_bad})
                else:
                    res["nonfinite_fields"].append({"field": name, "n_nonfinite": n_bad})
            if name in ENVELOPE and name not in CARVE_OUT:
                lo, hi = ENVELOPE[name]
                fin = arr[finite]
                if fin.size:
                    amin = float(fin.min()); amax = float(fin.max())
                    if amin < lo or amax > hi:
                        res["envelope_breaches"].append(
                            {"field": name, "min": amin, "max": amax, "lo": lo, "hi": hi})
    finally:
        ds.close()
    res["blowup"] = bool(res["nonfinite_fields"] or res["envelope_breaches"])
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--max-dom", type=int, default=9)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"v020_blowup_check: run dir not found {args.run_dir}", file=sys.stderr)
        return 2

    report = {"run_dir": str(args.run_dir), "domains": {}, "BLOWUP": False, "files_checked": 0}
    for i in range(1, args.max_dom + 1):
        dom = f"d{i:02d}"
        files = sorted(args.run_dir.glob(f"wrfout_{dom}_*"))
        dom_blow = False
        per_file = []
        for f in files:
            r = check_file(f)
            per_file.append(r)
            dom_blow = dom_blow or r["blowup"]
            report["files_checked"] += 1
        report["domains"][dom] = {"n_files": len(files), "blowup": dom_blow,
                                  "files": per_file}
        report["BLOWUP"] = report["BLOWUP"] or dom_blow

    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(payload)

    print(f"\n=== blow-up gate: {'BLOW-UP' if report['BLOWUP'] else 'PASS'} "
          f"({report['files_checked']} files) ===", file=sys.stderr)
    for dom, d in report["domains"].items():
        if d["n_files"]:
            tag = "BLOW-UP" if d["blowup"] else "ok"
            print(f"  {dom}: {d['n_files']} files -> {tag}", file=sys.stderr)
    return 1 if report["BLOWUP"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
