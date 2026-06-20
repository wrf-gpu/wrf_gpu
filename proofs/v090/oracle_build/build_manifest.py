"""Build manifest.json for a v090 physics-oracle subdir from sidecar + .f64 files.

The Fortran oracle dumper writes the sidecar HEADER (dims_ni_nk_nj, grid_id,
itimestep) but NOT per-field FIELD lines, so we reconstruct the manifest from the
.f64 files on disk + the sidecar dims, matching the schema that
gpuwrf.validation.tier1_thompson._load_f64_manifest / proofs/b2 readers consume:
each field entry has {scheme, tag, name, file, rank, shape, bytes, md5, min, max,
mean, nan}.  3D fields reshape C-order to (nj, nk, ni); 2D to (nj, ni).

Usage:
  python3 build_manifest.py <oracle_subdir> --scheme thompson [--source-run ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PHYS = dict(mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5,
            sf_surface_physics=4, ra_lw_physics=4, ra_sw_physics=4)


def parse_sidecar(p: Path):
    d = {}
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("dims_ni_nk_nj"):
            _, ni, nk, nj = ln.split()
            d["ni"], d["nk"], d["nj"] = int(ni), int(nk), int(nj)
        elif ln.startswith("grid_id"):
            d["grid_id"] = int(ln.split()[1])
        elif ln.startswith("itimestep"):
            d["itimestep"] = int(ln.split()[1])
    return d


def stats(a):
    f = a[np.isfinite(a)]
    return dict(min=float(np.min(f)), max=float(np.max(f)), mean=float(np.mean(f)),
                nan=int(np.sum(~np.isfinite(a))), shape=list(a.shape))


def build(subdir: Path, scheme: str, source_run: str, dims=None, itimestep=None) -> dict:
    """``dims`` = (ni, nk, nj) fallback when the Fortran sidecar header is empty
    (FORMATTED-unit buffering can leave the sidecar 0 bytes if read before WRF
    exits). The .f64 files are STREAM (unbuffered) so they are always complete."""
    man = dict(subdir=subdir.name, source_run=source_run, physics_options=PHYS,
               byte_order="big-endian", dtype="float64",
               reshape_order="C: 3D->(nj,nk,ni), 2D->(nj,ni)",
               grid_id=1, itimestep=itimestep, fields=[])
    for tag in ("in", "out"):
        sidecar = subdir / f"{scheme}_{tag}.sidecar.txt"
        files = sorted(subdir.glob(f"{scheme}_{tag}__*.f64"))
        if not files:
            continue
        if sidecar.exists() and sidecar.stat().st_size > 0:
            sc = parse_sidecar(sidecar)
            man["grid_id"], man["itimestep"] = sc["grid_id"], sc.get("itimestep", itimestep)
            ni, nk, nj = sc["ni"], sc["nk"], sc["nj"]
        elif dims is not None:
            ni, nk, nj = dims
        else:
            raise ValueError(f"{sidecar}: empty/missing and no --dims fallback given")
        for fpath in sorted(subdir.glob(f"{scheme}_{tag}__*.f64")):
            name = fpath.name.split("__", 1)[1][:-4]
            raw = np.fromfile(fpath, dtype=">f8")
            n3 = nj * nk * ni
            n2 = nj * ni
            if raw.size == n3:
                rank, shape = 3, (nj, nk, ni)
            elif raw.size == n2:
                rank, shape = 2, (nj, ni)
            else:
                raise ValueError(f"{fpath}: size {raw.size} != 3D {n3} or 2D {n2}")
            arr = raw.reshape(shape, order="C")
            md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
            man["fields"].append(dict(
                scheme=scheme, tag=tag, name=name, units="", stagger="", desc="",
                rank=rank, file=fpath.name, bytes=fpath.stat().st_size, md5=md5,
                **stats(arr)))
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subdir")
    ap.add_argument("--scheme", required=True)
    ap.add_argument("--source-run", default="<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260428_18z_l3_24h_20260525T221139Z")
    args = ap.parse_args()
    sd = Path(args.subdir)
    man = build(sd, args.scheme, args.source_run)
    (sd / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {sd/'manifest.json'} with {len(man['fields'])} fields, itimestep={man['itimestep']}")
