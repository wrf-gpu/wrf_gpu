#!/usr/bin/env python3
"""v020_wrfout_byte_compare.py -- bitwise wrfout comparison across two A/B arms.

Proves the v0.20.0 low-hanging-fruit levers keep the nest fp64_default output
identical. For each wrfout_d0N file present in BOTH arm output dirs, every data
variable is compared bit-for-bit (exact equality, NaN-aware). The host-only
levers (async output, host-RAM event fold, sync-cadence) must be BYTE-identical;
the cuda_async allocator is numerics-free (same ops/order -> expected byte-
identical too, but reported as max-abs-diff if XLA re-autotunes a fusion).

Usage:
  python scripts/v020_wrfout_byte_compare.py DIR_A DIR_B [--label-a A --label-b B]

Exit 0 if every compared variable is bit-identical; 1 otherwise (still prints the
full report either way).
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import netCDF4  # noqa: F401  (registers the netcdf backend)


def find_wrfout(d: str) -> dict[str, str]:
    """basename -> path for every wrfout_d0* file under d (recursively)."""
    out: dict[str, str] = {}
    for p in glob.glob(os.path.join(d, "**", "wrfout_d0*"), recursive=True):
        if os.path.isfile(p):
            out[os.path.basename(p)] = p
    return out


def compare_file(pa: str, pb: str) -> tuple[int, int, float, list[str]]:
    """Return (n_vars, n_identical, max_abs_diff, differing_var_names)."""
    from netCDF4 import Dataset
    da, db = Dataset(pa), Dataset(pb)
    try:
        common = [v for v in da.variables if v in db.variables]
        n_ident = 0
        max_diff = 0.0
        differing: list[str] = []
        for v in common:
            a = np.asarray(da.variables[v][:])
            b = np.asarray(db.variables[v][:])
            if a.shape != b.shape:
                differing.append(f"{v}(shape {a.shape}!={b.shape})")
                continue
            if a.dtype.kind in "SU" or b.dtype.kind in "SU":
                # char/string vars (e.g. Times) -- exact equality only
                if np.array_equal(a, b):
                    n_ident += 1
                else:
                    differing.append(f"{v}(char)")
                continue
            af = a.astype(np.float64, copy=False)
            bf = b.astype(np.float64, copy=False)
            both_nan = np.isnan(af) & np.isnan(bf)
            eq = (af == bf) | both_nan
            if eq.all():
                n_ident += 1
            else:
                d = np.abs(af[~eq] - bf[~eq])
                d = d[np.isfinite(d)]
                md = float(d.max()) if d.size else float("inf")
                max_diff = max(max_diff, md)
                differing.append(f"{v}(maxΔ={md:.3e})")
        return len(common), n_ident, max_diff, differing
    finally:
        da.close()
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    fa, fb = find_wrfout(args.dir_a), find_wrfout(args.dir_b)
    common = sorted(set(fa) & set(fb))
    print(f"# byte-compare {args.label_a} vs {args.label_b}")
    print(f"#   A={args.dir_a}")
    print(f"#   B={args.dir_b}")
    if not common:
        print(f"  NO common wrfout files (A has {len(fa)}, B has {len(fb)}) -- cannot compare")
        return 1
    all_identical = True
    tot_vars = tot_ident = 0
    worst = 0.0
    for name in common:
        nv, ni, md, diff = compare_file(fa[name], fb[name])
        tot_vars += nv
        tot_ident += ni
        worst = max(worst, md)
        ok = ni == nv
        all_identical = all_identical and ok
        tag = "BIT-IDENTICAL" if ok else f"DIFFERS maxΔ={md:.3e}"
        print(f"  {name}: {ni}/{nv} vars identical  [{tag}]")
        if diff:
            print(f"      differing: {', '.join(diff[:12])}{' ...' if len(diff) > 12 else ''}")
    print(f"# SUMMARY: {tot_ident}/{tot_vars} vars bit-identical across {len(common)} domain files; "
          f"worst maxΔ={worst:.3e}; verdict={'BYTE-IDENTICAL' if all_identical else 'NOT byte-identical'}")
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())
