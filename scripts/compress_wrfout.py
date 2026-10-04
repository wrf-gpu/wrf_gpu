"""Lossless rolling deflate of finished wrfout frames under a run_parallel_cases.sh output root (CPU only).

Usage: compress_wrfout.py OUT_ROOT [--workers 2] [--poll 20]
A frame of OUT_ROOT/<case>/wrfout is finished when a newer frame of the same domain exists, when the case's
receipt.json exists, or when OUT_ROOT/.cases_done exists (the launcher writes it last; the final pass then exits).
Each file: nccopy -d1 -s -> tmp, every variable and attribute compared (NaN-aware), the original is replaced only
when identical. Append-only, resumable receipt: OUT_ROOT/<case>/compress_receipt.jsonl.
Not for checkpointed runs: the restart journal authenticates the original history bytes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

# CPU only, never a GPU backend; every netCDF-C/HDF5 access goes through the product's process-wide lock.
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # this tree's gpuwrf (standalone runs too)
from gpuwrf.io.netcdf_lock import Dataset  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mismatches(a: Path, b: Path) -> list[str]:
    bad = []
    attrs = lambda v: {k: str(v.getncattr(k)) for k in v.ncattrs()}  # noqa: E731
    with Dataset(a) as x, Dataset(b) as y:
        if attrs(x) != attrs(y):
            bad.append("<global attrs>")
        if set(x.variables) != set(y.variables) or {k: len(d) for k, d in x.dimensions.items()} != {k: len(d) for k, d in y.dimensions.items()}:
            bad.append("<variable/dimension set>")
        for name in set(x.variables) & set(y.variables):
            vx, vy = x[name], y[name]
            if vx.dtype != vy.dtype or vx.shape != vy.shape or vx.dimensions != vy.dimensions or attrs(vx) != attrs(vy):
                bad.append(name)
                continue
            ax, ay = np.ma.getdata(vx[:]), np.ma.getdata(vy[:])
            if not np.array_equal(ax, ay, equal_nan=np.asarray(ax).dtype.kind in "fc"):
                bad.append(name)
    return sorted(bad)


def compress_one(path_s: str) -> dict:
    p = Path(path_s)
    tmp = p.with_name(p.name + ".deflate.tmp")
    rec = {"file": p.name, "orig_bytes": p.stat().st_size, "orig_sha256": sha256(p), "utc": time.strftime("%H:%M:%S", time.gmtime())}
    subprocess.run(["nccopy", "-d1", "-s", str(p), str(tmp)], check=True)
    bad = mismatches(p, tmp)
    rec.update(comp_bytes=tmp.stat().st_size, comp_sha256=sha256(tmp), value_identical=not bad, mismatches=bad)
    if bad:
        tmp.unlink()
    else:
        os.replace(tmp, p)
    return rec


def finished(root: Path, done: set[str]) -> list[Path]:
    over = (root / ".cases_done").exists()
    out = []
    for wrfout in sorted(root.glob("*/wrfout")):
        case_over = over or (wrfout.parent / "receipt.json").exists()
        for dom in sorted({p.name[:10] for p in wrfout.glob("wrfout_d*")}):
            frames = sorted(p for p in wrfout.glob(dom + "_*") if not p.name.endswith(".tmp"))
            out += [f for f in (frames if case_over else frames[:-1]) if str(f) not in done]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()
    root = a.root.resolve()
    done: set[str] = set()
    for receipt in root.glob("*/compress_receipt.jsonl"):
        done |= {str(receipt.parent / "wrfout" / json.loads(l)["file"]) for l in receipt.read_text().splitlines()}
    failures = 0
    with ProcessPoolExecutor(a.workers) as ex:
        while True:
            over = (root / ".cases_done").exists()
            todo = finished(root, done)
            for path, rec in zip(todo, ex.map(compress_one, [str(p) for p in todo])):
                with (path.parent.parent / "compress_receipt.jsonl").open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                done.add(str(path))
                failures += not rec["value_identical"]
            if over and not finished(root, done):
                break
            time.sleep(a.poll)
    print(json.dumps({"root": str(root), "files": len(done), "kept_uncompressed_mismatch": failures}))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
