"""Assemble the final CU_SCALESAS oracle column set and pack the savepoints.

base set   : gen_inputs.py default (seed 20261010: 320 synthetic regimes + 64 real PROD D5 columns)
rare paths : columns selected from a 12000-column random search (seed 777, regimes
             lowbase/capped/dry_mid/generic/tropical/cold) by the census-instrumented COPY of the
             pristine module (debug writes only; its outputs are byte-identical to the pristine build)

Usage: python make_savepoints.py <base_dir> <search_dir> <out_work_dir>
Then: build_and_run.sh <out_work_dir> <out_work_dir>/final_cfg_*.txt ; python make_savepoints.py --pack ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collect import parse  # noqa: E402
from gen_inputs import CONFIGS, FIELDS, write_case  # noqa: E402

PATH_QUOTAS = {  # census path -> max columns taken from the search set
    "kill_jmin_aa1": 10, "kill_wc_ddaa1": 10, "kill_closure": 10, "kill_cina": 10,
    "restore_rn<=0": 12, "ktconn_active": 20, "ktconn": 6, "htop_km": 6, "evapcap": 8,
    "active_nonQE": 8, "active_QE": 8,
}


def census(prefix):
    sys.path.insert(0, "<USER_HOME>/wrf_gpu2_lanes/o1-sas/tools")
    from census import load  # lane tool (instrumented-oracle parser)

    return load(str(prefix))[0]


def select(search_dir: Path):
    chosen = []
    for cfg in CONFIGS:
        paths = census(search_dir / f"s_{cfg}")
        for name, quota in PATH_QUOTAS.items():
            idx = np.where(paths[name])[0]
            for i in idx[:quota]:
                if int(i) not in chosen:
                    chosen.append(int(i))
    return sorted(chosen)


def assemble(base_dir: Path, search_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rare = select(search_dir)
    for name, cfg in CONFIGS.items():
        b = np.load(base_dir / f"{name}_inputs.npz")
        s = np.load(search_dir / f"s_{name}_inputs.npz")
        cols = []
        for src, idx in ((b, range(len(b["XLAND"]))), (s, rare)):
            for i in idx:
                cols.append({f: src[f][i] for f in FIELDS + ("P8W", "W")})
        xland = np.concatenate([b["XLAND"], s["XLAND"][rare]])
        dx2d = np.concatenate([b["DX2D"], s["DX2D"][rare]])
        kinds = list(b["KIND"]) + [f"search_{k}" for k in s["KIND"][rare]]
        write_case(out, f"final_{name}", cols, xland, dx2d, cfg, kinds)
    (out / "rare_indices.json").write_text(json.dumps({"search_seed": 777, "indices": rare}))
    print(f"assembled {len(b['XLAND'])} base + {len(rare)} rare columns")


def pack(work: Path, dest: Path, census_dir: Path):
    dest.mkdir(parents=True, exist_ok=True)
    meta = {"wrf_module": "phys/module_cu_scalesas.F (WRF 4.7.1, unmodified)", "configs": {}}
    src_sha = (work / "build_r4" / "sources.sha256").read_text().split()[0]
    meta["module_cu_scalesas_sha256"] = src_sha
    first = True
    for name, cfg in CONFIGS.items():
        inp = np.load(work / f"final_{name}_inputs.npz")
        if first:
            np.savez_compressed(dest / "columns_inputs.npz",
                                **{k: inp[k] for k in FIELDS + ("P8W", "W", "XLAND", "DX2D", "KIND")})
            first = False
        for prec in ("r4", "r8"):
            o = parse(work / f"final_{name}_{prec}.out")
            dt = np.float32 if prec == "r4" else np.float64
            np.savez_compressed(dest / f"{name}_oracle_{prec}.npz", **{k: v.astype(dt) for k, v in o.items()})
        paths = census(census_dir / f"final_{name}")
        meta["configs"][name] = {
            "DT": float(inp["DT"]), "STEPCU": int(inp["STEPCU"]), "ITIMESTEP": int(inp["ITIMESTEP"]),
            "DY": float(inp["DY"]), "PGCON": float(inp["PGCON"]),
            "census_r4": {k: int(v.sum()) for k, v in paths.items()},
        }
    for f in sorted(dest.glob("*.npz")):
        meta.setdefault("sha256", {})[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (dest / "MANIFEST.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(json.dumps(meta["configs"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("c")
    args = ap.parse_args()
    if args.pack:
        pack(Path(args.a), Path(args.b), Path(args.c))
    else:
        assemble(Path(args.a), Path(args.b), Path(args.c))
