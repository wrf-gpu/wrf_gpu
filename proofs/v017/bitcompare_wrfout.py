#!/usr/bin/env python3
"""Strict bitwise comparison of two wrfout directories (data arrays).

For every wrfout file present in BOTH dirs, compare every numeric variable's raw
array with np.array_equal (exact).  Reports any variable whose values differ and
the global max abs diff.  Used to prove the v0.17 host-sync fix produces
byte-identical wrfout to the legacy block_between path (block_until_ready is a
host wait outside every jax.jit -> no dispatched op changes).  CPU only.

Usage: bitcompare_wrfout.py <dirA> <dirB> [domain_glob]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def main():
    a_dir = Path(sys.argv[1])
    b_dir = Path(sys.argv[2])
    glob = sys.argv[3] if len(sys.argv) > 3 else "wrfout_*"

    a_files = {p.name: p for p in a_dir.glob(glob)}
    b_files = {p.name: p for p in b_dir.glob(glob)}
    common = sorted(set(a_files) & set(b_files))
    if not common:
        print(f"NO COMMON FILES matching {glob} in {a_dir} and {b_dir}")
        return 2

    total_vars = 0
    diff_vars = 0
    global_max = 0.0
    worst = None
    per_file = []
    for name in common:
        da = Dataset(a_files[name])
        db = Dataset(b_files[name])
        file_diffs = 0
        file_max = 0.0
        for vname in da.variables:
            if vname not in db.variables:
                continue
            va = da.variables[vname][:]
            vb = db.variables[vname][:]
            if not np.issubdtype(np.asarray(va).dtype, np.number):
                continue
            total_vars += 1
            arr_a = np.asarray(va)
            arr_b = np.asarray(vb)
            if arr_a.shape != arr_b.shape:
                file_diffs += 1
                diff_vars += 1
                continue
            if not np.array_equal(arr_a, arr_b):
                d = float(np.nanmax(np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))))
                file_diffs += 1
                diff_vars += 1
                file_max = max(file_max, d)
                if d > global_max:
                    global_max = d
                    worst = (name, vname, d)
        da.close()
        db.close()
        per_file.append((name, file_diffs, file_max))

    print(f"files_compared={len(common)} total_numeric_vars={total_vars} differing_vars={diff_vars}")
    print(f"global_max_abs_diff={global_max:.6e}")
    if worst:
        print(f"worst_var: file={worst[0]} var={worst[1]} maxabsdiff={worst[2]:.6e}")
    for name, fd, fm in per_file:
        flag = "OK" if fd == 0 else f"DIFF({fd} vars, max={fm:.3e})"
        print(f"  {name}: {flag}")
    if diff_vars == 0:
        print("VERDICT: BITWISE-IDENTICAL (all numeric variables exactly equal)")
        return 0
    print("VERDICT: NOT bitwise-identical")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
