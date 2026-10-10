"""CAM01: compact committed fixture = one column per (census category, tau) + one day/one night per augmented pattern.

python3 compact.py cam01_fixture.npz cam01_columns.json ../../data/fixtures/cam01-compact-v1.npz
Drops arrays the tests never read (r8l_abstot/absnxt duplicate lwi_*, r8h_abstot/absnxt are the held REAL widened).
"""
import hashlib
import json
import sys

import numpy as np

DROP = {"r8l_abstot", "r8l_absnxt", "r8h_abstot", "r8h_absnxt", "r8h_emstot"}


def main(src, cols_json, dst):
    f = np.load(src)
    cols = json.load(open(cols_json))["columns"]
    seen, keep = set(), []
    for i, c in enumerate(cols):
        key = (c["category"], c["tau"] if not c["category"].startswith("aug_") else c["tau"] in (12, 36))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    keep = np.asarray(keep)
    ncol = len(cols)
    out = {}
    for k in f.files:
        if k in DROP:
            continue
        a = f[k]
        out[k] = a[keep] if a.ndim >= 1 and a.shape[0] == ncol and not k.startswith(("pin", "m_hybi", "in_znu")) else a
    out["columns_index"] = keep
    out["columns_category"] = np.asarray([cols[i]["category"] for i in keep])
    np.savez_compressed(dst, **out)
    print(len(keep), "columns ->", dst, hashlib.sha256(open(dst, "rb").read()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:4])
