#!/usr/bin/env python
"""Extract the fair d02 CPU denominator from the pinned Gen2 WRF backfill."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.gen2_accessor import DEFAULT_M6_GEN2_RUN_DIR, Gen2Run


RUN_PATH = DEFAULT_M6_GEN2_RUN_DIR
WRF_COMPILE_LOG = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/compile.log")
OUTPUT = ROOT / "artifacts" / "m6" / "cpu_denominator_v2.json"
TIMING_RE = re.compile(r"Timing for main: time .* on domain\s+(?P<domain>\d+):\s+(?P<seconds>[0-9.]+) elapsed seconds")


def _timings(run_path: Path) -> dict[int, list[float]]:
    source = run_path / "rsl.error.0000"
    timings: dict[int, list[float]] = {}
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING_RE.search(line)
        if match is None:
            continue
        domain = int(match.group("domain"))
        timings.setdefault(domain, []).append(float(match.group("seconds")))
    if not timings:
        raise RuntimeError(f"no WRF timing lines found in {source}")
    return timings


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower))


def _compile_metadata() -> dict[str, Any]:
    flags = [
        "-O3",
        "-acc",
        "-gpu=cc120,fastmath",
        "-w",
        "-Mfree",
        "-byteswapio",
        "-Mrecursive",
        "-r4",
        "-i4",
    ]
    nvhpc_version = "26.3"
    if WRF_COMPILE_LOG.exists():
        text = WRF_COMPILE_LOG.read_text(encoding="utf-8", errors="replace")
        version_match = re.search(r"nvfortran\)\s+(?P<version>\d+\.\d+)", text)
        if version_match is not None:
            nvhpc_version = version_match.group("version")
    return {
        "nvhpc_version": nvhpc_version,
        "compile_flags": flags,
        "compile_log": str(WRF_COMPILE_LOG),
        "compile_precision_note": "compile.log records -r4 default real; this artifact preserves that evidence for M6-S5.",
    }


def _domain_dt_seconds(run: Gen2Run) -> dict[int, float]:
    domains = run.namelist["domains"]
    base_dt = float(domains["time_step"] if "time_step" in domains else run.namelist["domains"].get("time_step", 18))
    if "time_step" not in domains:
        base_dt = float(run.namelist.get("domains", {}).get("time_step", 18))
    ratios = domains.get("parent_grid_ratio", [1])
    parents = domains.get("parent_id", [1])
    if not isinstance(ratios, list):
        ratios = [ratios]
    if not isinstance(parents, list):
        parents = [parents]
    dt_by_domain = {1: base_dt}
    for idx in range(2, len(ratios) + 1):
        parent = int(parents[idx - 1])
        dt_by_domain[idx] = dt_by_domain[parent] / float(ratios[idx - 1])
    return dt_by_domain


def _work_weights(run: Gen2Run) -> dict[int, float]:
    dt = _domain_dt_seconds(run)
    total_seconds = 24.0 * 3600.0
    weights = {}
    for domain in run.domains:
        number = int(domain[1:])
        grid = run.grid(domain)
        points = float(grid.mass_nx * grid.mass_ny * grid.mass_nz)
        steps = total_seconds / dt[number]
        weights[number] = points * steps
    return weights


def build_denominator(run_path: Path = RUN_PATH) -> dict[str, Any]:
    run = Gen2Run(run_path)
    timings = _timings(run.path)
    domain1 = timings.get(1)
    if not domain1:
        raise RuntimeError("domain 1 timing lines are required for total nested-run wall time")
    total_wall_s = float(sum(domain1))
    weights = _work_weights(run)
    d02_fraction = weights[2] / sum(weights.values())
    d02_wall_s = total_wall_s * d02_fraction
    d02_steps = (24.0 * 3600.0) / _domain_dt_seconds(run)[2]
    timing_summary = {
        str(domain): {
            "count": len(values),
            "sum_s": float(sum(values)),
            "mean_s": float(statistics.fmean(values)),
            "p50_s": float(statistics.median(values)),
            "p95_s": _percentile(values, 95.0),
        }
        for domain, values in sorted(timings.items())
    }
    metadata = _compile_metadata()
    return {
        "run_id": run.run_id,
        "hardware": "same workstation as RTX 5090 (28-rank WRF)",
        "nvhpc_version": metadata["nvhpc_version"],
        "compile_flags": metadata["compile_flags"],
        "compile_log": metadata["compile_log"],
        "compile_precision_note": metadata["compile_precision_note"],
        "wall_time_per_step_ms": float(d02_wall_s / d02_steps * 1000.0),
        "wall_time_total_24h_s": total_wall_s,
        "wall_time_d02_attributable_s": float(d02_wall_s),
        "attribution_policy": "d02 fraction of total nested run by domain ratio of grid points x timestep ratio",
        "attribution_fraction_d02": float(d02_fraction),
        "fp_precision": "FP32 default real observed (-r4); review contract expected FP64, so M6-S5 must account for this evidence",
        "domain_count": len(run.domains),
        "nest_ratio": [int(run.grid(domain).dx_m) for domain in run.domains],
        "domain_work_weights": {str(domain): value for domain, value in sorted(weights.items())},
        "raw_timing_summary": timing_summary,
        "source_logs": [str(run.path / "rsl.error.0000"), str(run.path / "namelist.input"), str(WRF_COMPILE_LOG)],
    }


def main() -> int:
    artifact = build_denominator()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUTPUT), "wall_time_d02_attributable_s": artifact["wall_time_d02_attributable_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
