"""Rolling lossless compression of completed GPU history frames during a long WN3 arm (CPU, lane cores).

Usage: rolling_compress.py ARM_DIR [--workers 2] [--poll 20]
A frame is eligible when a newer frame of the same domain exists, or when ARM_DIR/end.json exists (run over).
Each file: nccopy -d1 -s -> tmp, NaN-aware per-variable + attribute identity check, replace only if identical.
Receipt (append-only, resumable): ARM_DIR/<case>/compress_receipt.jsonl. Exits after the final pass post end.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from netCDF4 import Dataset

ROOT_PREFIX = "<USER_HOME>/wrf_gpu2_lanes/wn3/W7/"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def mismatches(a: Path, b: Path) -> list[str]:
    bad = []
    with Dataset(a) as x, Dataset(b) as y:
        if {k: str(x.getncattr(k)) for k in x.ncattrs()} != {k: str(y.getncattr(k)) for k in y.ncattrs()}:
            bad.append("<global attrs>")
        if set(x.variables) != set(y.variables):
            bad.append("<variable set>")
        for name in x.variables:
            if name not in y.variables:
                continue
            vx, vy = x[name], y[name]
            if vx.dtype != vy.dtype or vx.shape != vy.shape or vx.dimensions != vy.dimensions:
                bad.append(name); continue
            if {k: str(vx.getncattr(k)) for k in vx.ncattrs()} != {k: str(vy.getncattr(k)) for k in vy.ncattrs()}:
                bad.append(name + ":attrs"); continue
            ax, ay = np.ma.getdata(vx[:]), np.ma.getdata(vy[:])
            if not np.array_equal(ax, ay, equal_nan=np.asarray(ax).dtype.kind in "fc"):
                bad.append(name)
    return bad


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


def eligible(arm: Path, done: set[str]) -> list[Path]:
    over = (arm / "end.json").exists()
    out = []
    for wrfout in sorted(arm.glob("2026*/wrfout")):
        for dom in ("d01", "d02", "d03"):
            frames = sorted(wrfout.glob(f"wrfout_{dom}_*"))
            frames = [f for f in frames if not f.name.endswith(".tmp")]
            ready = frames if over else frames[:-1]
            out += [f for f in ready if str(f) not in done]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--poll", type=float, default=20.0)
    a = ap.parse_args()
    assert str(a.arm.resolve()).startswith(ROOT_PREFIX), "W7 arms only"
    done: set[str] = set()
    for rj in a.arm.glob("2026*/compress_receipt.jsonl"):
        for line in rj.read_text().splitlines():
            r = json.loads(line)
            done.add(str(rj.parent / "wrfout" / r["file"]))
    with ProcessPoolExecutor(a.workers) as ex:
        while True:
            over = (a.arm / "end.json").exists()
            todo = eligible(a.arm, done)
            for path, rec in zip(todo, ex.map(compress_one, [str(p) for p in todo])):
                with (path.parent.parent / "compress_receipt.jsonl").open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                done.add(str(path))
            if over and not eligible(a.arm, done):
                break
            time.sleep(a.poll)
    print(json.dumps({"arm": str(a.arm), "compressed": len(done)}))


if __name__ == "__main__":
    main()
