#!/usr/bin/env python
"""Report the numeric magnitude of per-leaf deltas between two state dumps.

Distinguishes fp64-ULP XLA-codegen noise (max |rel| ~ 1e-13..1e-11) from an
algorithmic bug (large deltas). Inputs are the *_state.npz files written by
probe_ab_identity.py --dump-state.

Usage: delta_magnitude.py A_state.npz B_state.npz [label]
"""
import sys
import numpy as np

a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else "A-vs-B"
keys = sorted(set(a.files) & set(b.files))
rows = []
worst_abs = 0.0
worst_rel = 0.0
for k in keys:
    x = np.asarray(a[k], dtype=np.float64); y = np.asarray(b[k], dtype=np.float64)
    if x.shape != y.shape:
        rows.append((k, "shape-mismatch", x.shape, y.shape)); continue
    d = np.abs(x - y)
    mab = float(d.max()) if d.size else 0.0
    scale = np.maximum(np.abs(x), np.abs(y))
    rel = np.where(scale > 0, d / np.where(scale > 0, scale, 1.0), 0.0)
    mrel = float(rel.max()) if rel.size else 0.0
    n_diff = int((d > 0).sum())
    if mab > 0:
        rows.append((k, mab, mrel, n_diff, int(x.size)))
        worst_abs = max(worst_abs, mab); worst_rel = max(worst_rel, mrel)

print(f"=== {label}: {len(rows)} leaves with nonzero delta (of {len(keys)} compared) ===")
for r in sorted(rows, key=lambda t: -(t[1] if isinstance(t[1], float) else 0)):
    if r[1] == "shape-mismatch":
        print(f"  {r[0]:18s} SHAPE {r[2]} vs {r[3]}")
    else:
        k, mab, mrel, nd, sz = r
        print(f"  {k:18s} max|abs|={mab:.3e}  max|rel|={mrel:.3e}  diff_cells={nd}/{sz}")
print(f"\nWORST max|abs|={worst_abs:.3e}   WORST max|rel|={worst_rel:.3e}")
verdict = ("FP64-ULP-NOISE (XLA codegen; Tier-P-equivalent, non-algorithmic)"
           if worst_rel < 1e-9 else
           "LARGE -> ALGORITHMIC DIFFERENCE (real bug)")
print(f"VERDICT: {verdict}")
