#!/usr/bin/env python3
"""Audit every Swiss history frame with the frozen strict scorer and snow mask.

This adds an explicit per-frame census beside the scorer's three-frame degeneracy
sample. It does not change the scorer, its result, or any physical tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

from netCDF4 import Dataset
import numpy as np


def read(dataset, name):
    return np.ma.asarray(dataset[name][0], dtype=np.float64).filled(np.nan)


def metrics(cpu, gpu, mask):
    c, g = cpu[mask], gpu[mask]
    if not c.size or not np.isfinite(c).all() or not np.isfinite(g).all():
        raise ValueError("snow diagnostic needs nonempty, finite original CPU/GPU cells")
    delta = g - c
    return {"n": int(c.size), "cpu_mean": float(c.mean()), "gpu_mean": float(g.mean()),
            "rmse": float(np.sqrt(np.mean(delta * delta))), "bias": float(delta.mean()),
            "max_abs": float(np.abs(delta).max()), "cpu_abs_max": float(np.abs(c).max()),
            "cpu_large_sentinel_count": int((np.abs(c) > 1e30).sum()),
            "gpu_cpu_mean_ratio": float(g.mean() / c.mean()) if c.mean() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--gpu-receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    cpu_receipt = json.loads(args.cpu_receipt.read_text())
    gpu_receipt = json.loads(args.gpu_receipt.read_text())
    if (not cpu_receipt.get("wrf_success") or not cpu_receipt.get("complete_24h")
            or not gpu_receipt.get("complete_24h") or gpu_receipt.get("rc") != 0
            or gpu_receipt.get("device", {}).get("platform") != "gpu"):
        raise ValueError("audit needs successful original CPU-WRF and actual GPU receipts")
    inputs = {name: hashlib.sha256((args.cpu_dir / name).read_bytes()).hexdigest()
              for name in ("namelist.input", "wrfinput_d01", "wrfbdy_d01")}
    if inputs != gpu_receipt["inputs_sha256"]:
        raise ValueError("GPU inputs differ from the original CPU-WRF inputs")
    cpu = sorted(args.cpu_dir.glob("wrfout_d01_*"))
    gpu = sorted(args.gpu_dir.glob("wrfout_d01_*"))
    if len(cpu) != 25 or [p.name for p in cpu] != [p.name for p in gpu]:
        raise ValueError("Swiss audit needs all 25 identically named original CPU/GPU frames")
    scorer_path = args.repo / "scripts/wn3_score.py"
    spec = importlib.util.spec_from_file_location("swiss_frozen_score", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    with Dataset(cpu[0]) as c:
        snow = read(c, "SNOWH") > 0
        soil16 = read(c, "ISLTYP") == 16
    masks = {"initial_snow": snow, "initial_snow_nonice_soil": snow & ~soil16,
             "ice_soil16": snow & soil16}
    scorer_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    revision = gpu_receipt["git_head"]
    audit = {"model_commit": revision, "scorer_commit": scorer_revision,
             "inputs_sha256": inputs, "scorer_sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
             "scope": "unmodified frozen output_integrity called separately on every frame", "rows": []}
    tauss = {"model_commit": revision, "scope": "original CPU-WRF, fixed initial SNOWH>0 masks; no TAUSS tolerance invented",
             "initial_snow_cells": int(snow.sum()), "snow_ice_soil16_cells": int((snow & soil16).sum()), "rows": []}
    fields = {}
    albedo = []
    for h, (cp, gp) in enumerate(zip(cpu, gpu)):
        result = scorer.output_integrity([cp], [gp])
        audit["rows"].append({"lead_h": h, "file": gp.name, "result": result})
        for field, events in result["degenerate_fields"].items():
            fields.setdefault(field, []).extend(events)
        with Dataset(cp) as c, Dataset(gp) as g:
            ct, gt = read(c, "TAUSS"), read(g, "TAUSS")
            tauss["rows"].append({"lead_h": h, "file": gp.name,
                                 "groups": {k: metrics(ct, gt, mask) for k, mask in masks.items()}})
            ca, ga = read(c, "ALBEDO"), read(g, "ALBEDO")
            albedo.append({"lead_h": h, "file": gp.name, "gpu_sentinel_cells": int((ga <= -999).sum()),
                           "cpu_sentinel_cells": int((ca <= -999).sum()), **metrics(ca, ga, np.ones(ca.shape, bool))})
    summary = {name: {"frames": len(events), "first": events[0]["file"], "last": events[-1]["file"],
                     "cpu_abs_max": max(abs(v) for e in events for v in e["cpu_range"]),
                     "gpu_values": sorted({e["gpu_value"] for e in events}, key=lambda x: (x is None, x or 0))}
               for name, events in sorted(fields.items())}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [("integrity_every_frame.json", audit), ("degenerate_every_frame_summary.json", summary),
                        ("tauss_original_cpu.json", tauss), ("albedo_every_frame.json", albedo)]:
        (args.out_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"frames": 25, "strict_frames_passing": sum(r["result"]["pass"] for r in audit["rows"]),
                      "degenerate_fields": len(fields), "albedo_sentinel_cells": sum(r["gpu_sentinel_cells"] for r in albedo),
                      "snow_cells": int(snow.sum())}))


if __name__ == "__main__":
    main()
