#!/usr/bin/env python3
"""R6 G4: bitwise comparison of two saved forecast final states (npz dumps).

Mechanical gate: for every array leaf present in BOTH dumps, shapes and dtypes
must match and max abs diff must be EXACTLY 0.0 on every element (bitwise).
Writes a JSON verdict; exit 0 iff bitwise-identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def compare(path_a: Path, path_b: Path, out: Path) -> bool:
    a = np.load(path_a)
    b = np.load(path_b)
    keys_a, keys_b = set(a.files), set(b.files)
    verdict: dict = {
        "schema": "wrf_gpu2.v025.r6.bitwise_compare.v1",
        "a": str(path_a), "b": str(path_b),
        "only_in_a": sorted(keys_a - keys_b),
        "only_in_b": sorted(keys_b - keys_a),
        "fields": {},
        "bitwise_identical": False,
    }
    ok = not (keys_a - keys_b) and not (keys_b - keys_a)
    for key in sorted(keys_a & keys_b):
        x, y = a[key], b[key]
        row: dict = {"shape_a": list(x.shape), "shape_b": list(y.shape),
                     "dtype_a": str(x.dtype), "dtype_b": str(y.dtype)}
        if x.shape != y.shape or x.dtype != y.dtype:
            row["identical"] = False
            row["reason"] = "shape_or_dtype_mismatch"
            ok = False
        else:
            equal = bool(np.array_equal(x, y))  # exact, NaN != NaN
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64)) if x.dtype.kind == "f" else None
            row.update({
                "identical": equal,
                "max_abs_diff": float(np.max(diff)) if diff is not None and diff.size else (0.0 if equal else float("inf")),
                "n_different_elements": int(np.count_nonzero(x != y)),
            })
            ok = ok and equal
        verdict["fields"][key] = row
    verdict["bitwise_identical"] = ok
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(f"bitwise_identical={ok} fields={len(verdict['fields'])} -> {out}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: r6_compare_states.py A.npz B.npz OUT.json", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if compare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])) else 1)
