#!/usr/bin/env python
"""v0.15 kernel probe — condense nsys stats CSVs into per-step launch/gap facts.

Reads proofs/perf/v015/nsys_steady50_{cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_sum,
nvtx_gpu_proj_sum}.csv (whatever exists) and writes nsys_summary.json.
Steps profiled: 3 x 50 (two warm calls + STEADY50 region).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "nsys_summary.json"


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        # nsys prepends comment lines sometimes; sniff the header row
        rows = [r for r in csv.reader(f) if r]
    if not rows:
        return []
    hdr = None
    out = []
    for r in rows:
        if hdr is None:
            if any("Time" in c or "Name" in c for c in r):
                hdr = r
            continue
        out.append(dict(zip(hdr, r)))
    return out


def fnum(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0


def main() -> int:
    summary = {"schema": "V015NsysSummary", "steps_total": 150}

    api = read_csv(HERE / "nsys_steady50_cuda_api_sum.csv")
    if api:
        launches = [r for r in api if "LaunchKernel" in r.get("Name", "")]
        total_api_ms = sum(fnum(r.get("Total Time (ns)")) for r in api) / 1e6
        launch_ms = sum(fnum(r.get("Total Time (ns)")) for r in launches) / 1e6
        launch_n = sum(int(fnum(r.get("Num Calls"))) for r in launches)
        summary["cuda_api"] = {
            "total_api_ms": round(total_api_ms, 1),
            "launch_calls": launch_n,
            "launch_total_ms": round(launch_ms, 1),
            "launch_mean_us": round(launch_ms * 1000.0 / launch_n, 2) if launch_n else None,
            "launches_per_step_est": round(launch_n / 150.0, 1),
            "top": [
                {
                    "name": r.get("Name"),
                    "calls": int(fnum(r.get("Num Calls"))),
                    "total_ms": round(fnum(r.get("Total Time (ns)")) / 1e6, 1),
                }
                for r in sorted(api, key=lambda r: -fnum(r.get("Total Time (ns)")))[:10]
            ],
        }

    kern = read_csv(HERE / "nsys_steady50_cuda_gpu_kern_sum.csv")
    if kern:
        total_ms = sum(fnum(r.get("Total Time (ns)")) for r in kern) / 1e6
        total_n = sum(int(fnum(r.get("Instances"))) for r in kern)
        summary["gpu_kernels"] = {
            "distinct": len(kern),
            "instances": total_n,
            "instances_per_step_est": round(total_n / 150.0, 1),
            "total_kernel_ms": round(total_ms, 1),
            "mean_kernel_us": round(total_ms * 1000.0 / total_n, 2) if total_n else None,
            "top": [
                {
                    "name": (r.get("Name") or "")[:110],
                    "instances": int(fnum(r.get("Instances"))),
                    "total_ms": round(fnum(r.get("Total Time (ns)")) / 1e6, 1),
                    "mean_us": round(fnum(r.get("Avg (ns)")) / 1e3, 1),
                }
                for r in sorted(kern, key=lambda r: -fnum(r.get("Total Time (ns)")))[:25]
            ],
        }

    gsum = read_csv(HERE / "nsys_steady50_cuda_gpu_sum.csv")
    if gsum:
        summary["gpu_sum_top"] = [
            {
                "name": (r.get("Name") or r.get("Operation") or "")[:110],
                "total_ms": round(fnum(r.get("Total Time (ns)")) / 1e6, 1),
                "category": r.get("Category"),
            }
            for r in sorted(gsum, key=lambda r: -fnum(r.get("Total Time (ns)")))[:12]
        ]

    nvtx = read_csv(HERE / "nsys_steady50_nvtx_gpu_proj_sum.csv")
    if nvtx:
        summary["nvtx_gpu_projection"] = [
            {
                "range": r.get("Range") or r.get("Name"),
                "proj_total_ms": round(fnum(r.get("Total Proj Time (ns)") or r.get("Projected Time (ns)") or r.get("Total Time (ns)")) / 1e6, 1),
            }
            for r in nvtx[:8]
        ]

    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
