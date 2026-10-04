"""Lossless deflate of one case's GPU history files, with a per-file value-identity check (CPU).

Usage: compress_case.py CASE_DIR [--workers 4]
For each CASE_DIR/wrfout/wrfout_d0*: nccopy -d1 -s -> tmp; every variable np.array_equal (NaN-aware) and
global/variable attributes equal -> replace original; else keep the original and record the mismatch.
Receipt: CASE_DIR/compress_receipt.json (original + compressed sha256/bytes per file).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from netCDF4 import Dataset


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def identical(a: Path, b: Path) -> list[str]:
    bad = []
    with Dataset(a) as x, Dataset(b) as y:
        if {k: str(x.getncattr(k)) for k in x.ncattrs()} != {k: str(y.getncattr(k)) for k in y.ncattrs()}:
            bad.append("<global attrs>")
        if set(x.variables) != set(y.variables) or dict(x.dimensions.items()).keys() != dict(y.dimensions.items()).keys():
            bad.append("<variable/dimension set>")
        for name in x.variables:
            if name not in y.variables:
                continue
            vx, vy = x[name], y[name]
            if vx.dtype != vy.dtype or vx.shape != vy.shape or vx.dimensions != vy.dimensions:
                bad.append(name); continue
            if {k: str(vx.getncattr(k)) for k in vx.ncattrs()} != {k: str(vy.getncattr(k)) for k in vy.ncattrs()}:
                bad.append(name + ":attrs"); continue
            ax, ay = np.ma.getdata(vx[:]), np.ma.getdata(vy[:])
            kind = np.asarray(ax).dtype.kind
            if not np.array_equal(ax, ay, equal_nan=kind in "fc"):
                bad.append(name)
    return bad


def one(path_s: str) -> dict:
    p = Path(path_s)
    tmp = p.with_name(p.name + ".deflate.tmp")
    rec = {"file": p.name, "orig_bytes": p.stat().st_size, "orig_sha256": sha256(p)}
    subprocess.run(["nccopy", "-d1", "-s", str(p), str(tmp)], check=True)
    bad = identical(p, tmp)
    rec.update(comp_bytes=tmp.stat().st_size, comp_sha256=sha256(tmp), value_identical=not bad, mismatches=bad)
    if bad:
        tmp.unlink()
    else:
        os.replace(tmp, p)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert str(a.case_dir.resolve()).startswith("<USER_HOME>/wrf_gpu2_lanes/wn3/W6/twins24_p3/"), "twins outputs only"
    files = sorted(str(p) for p in (a.case_dir / "wrfout").glob("wrfout_d0*") if not p.name.endswith(".tmp"))
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(one, files))
    out = {"tool": "nccopy -d1 -s", "files": recs, "all_value_identical": all(r["value_identical"] for r in recs),
           "orig_total_gb": sum(r["orig_bytes"] for r in recs) / 1e9, "comp_total_gb": sum(r["comp_bytes"] for r in recs) / 1e9}
    (a.case_dir / "compress_receipt.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "files"} | {"n": len(recs)}))


if __name__ == "__main__":
    main()
