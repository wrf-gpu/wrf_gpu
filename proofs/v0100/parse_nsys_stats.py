"""Parse v0.10.0 Phase 0 nsys CSV exports into compact JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARTIFACT_KERNEL_MARKERS = (
    "RedzoneAllocator",
    "DelayKernel",
    "xla_fp_comparison",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(line for line in handle if line.strip())
        return list(reader)


def _num(row: dict[str, str], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            text = str(row[name]).replace(",", "")
            try:
                return float(text)
            except ValueError:
                continue
    return 0.0


def _name(row: dict[str, str]) -> str:
    for key in ("Name", "Operation", "Mem Operation", "Name:Event Type"):
        if key in row:
            return row[key]
    return ""


def _count(row: dict[str, str]) -> int:
    return int(_num(row, "Instances", "Total", "Calls", "Count", "Num Calls"))


def _sum_counts(rows: list[dict[str, str]], needle: str) -> int:
    needle_l = needle.lower()
    return sum(_count(row) for row in rows if needle_l in _name(row).lower())


def _top(rows: list[dict[str, str]], limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for row in rows[:limit]:
        out.append(
            {
                "name": _name(row),
                "instances": _count(row),
                "total_time_pct": _num(row, "Time (%)", "Total Time (%)"),
                "total_time_ns": _num(row, "Total Time (ns)", "Total Time"),
                "avg_ns": _num(row, "Avg (ns)", "Average Time (ns)", "Avg"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver-json", type=Path, required=True)
    parser.add_argument("--stats-prefix", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    driver = json.loads(args.driver_json.read_text(encoding="utf-8"))
    steps = int(driver["profile_steps"])
    prefix = str(args.stats_prefix)
    kern = _read_csv(Path(prefix + "_cuda_gpu_kern_sum.csv"))
    gpu = _read_csv(Path(prefix + "_cuda_gpu_sum.csv"))
    api = _read_csv(Path(prefix + "_cuda_api_sum.csv"))
    nvtx = _read_csv(Path(prefix + "_nvtx_sum.csv"))

    loop_elementwise = sum(_count(row) for row in kern if _name(row).startswith("loop_"))
    d2d = sum(
        _count(row)
        for row in gpu
        if "device-to-device" in _name(row).lower() or "[CUDA memcpy Device-to-Device]" in _name(row)
    )
    h2d = sum(
        _count(row)
        for row in gpu
        if "host-to-device" in _name(row).lower() or "[CUDA memcpy Host-to-Device]" in _name(row)
    )
    d2h = sum(
        _count(row)
        for row in gpu
        if "device-to-host" in _name(row).lower() or "[CUDA memcpy Device-to-Host]" in _name(row)
    )
    artifact_counts = {
        marker: _sum_counts(kern, marker)
        for marker in ARTIFACT_KERNEL_MARKERS
    }
    artifact_total = int(sum(artifact_counts.values()))

    payload = {
        "schema": "V0100Phase0NsysSummary",
        "schema_version": 1,
        "status": "PASS",
        "driver": driver,
        "steps": steps,
        "counts_total": {
            "kernel_instances": int(sum(_count(row) for row in kern)),
            "loop_elementwise_instances": int(loop_elementwise),
            "d2d_memcpy_instances": int(d2d),
            "h2d_memcpy_instances": int(h2d),
            "d2h_memcpy_instances": int(d2h),
            "autotune_artifact_kernel_instances": artifact_total,
            "autotune_artifact_by_marker": artifact_counts,
        },
        "counts_per_step": {
            "kernel_instances": float(sum(_count(row) for row in kern)) / steps,
            "loop_elementwise_instances": float(loop_elementwise) / steps,
            "d2d_memcpy_instances": float(d2d) / steps,
            "h2d_memcpy_instances": float(h2d) / steps,
            "d2h_memcpy_instances": float(d2h) / steps,
            "autotune_artifact_kernel_instances": float(artifact_total) / steps,
        },
        "top_kernels": _top(kern),
        "top_gpu_memops": _top(gpu),
        "top_cuda_api": _top(api),
        "nvtx": _top(nvtx),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts_per_step"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
