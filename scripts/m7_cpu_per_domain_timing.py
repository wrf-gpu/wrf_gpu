#!/usr/bin/env python
"""Extract honest per-domain WRF timing denominators for the M7 speedup claim."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.paths import reference_path  # noqa: E402


SPRINT_DIR = ROOT / "proofs" / "generated" / "2026-05-27-m7-honest-speedup-skill-diff"
DEFAULT_NAMELIST_GLOB = str(reference_path("runs", "wrf_l3", "20260521_18z_l3_24h_*", "namelist.output"))
DEFAULT_PIPELINE_RUN = ROOT / "proofs" / "2026-05-27-m7-daily-pipeline-integration__pipeline_run_20260521.json"
DEFAULT_CPU_OUTPUT = SPRINT_DIR / "cpu_per_domain_wall_clock.json"
DEFAULT_SPEEDUP_OUTPUT = SPRINT_DIR / "honest_speedup_table.json"
DEFAULT_GPU_WALL_S = 324.77563990700037

TIMING_RE = re.compile(
    r"Timing for main:\s+time\s+"
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})\s+"
    r"on domain\s+(?P<domain>\d+):\s+"
    r"(?P<elapsed>[0-9]+(?:\.[0-9]+)?)\s+elapsed seconds"
)


@dataclass(frozen=True)
class TimingRecord:
    timestamp: datetime
    domain: int
    elapsed_s: float

    @property
    def dedupe_key(self) -> tuple[str, int, float]:
        return (self.timestamp.isoformat(), self.domain, round(self.elapsed_s, 9))


def parse_timing_line(line: str) -> TimingRecord | None:
    """Parse one WRF ``Timing for main`` line."""

    match = TIMING_RE.search(line)
    if match is None:
        return None
    return TimingRecord(
        timestamp=datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc),
        domain=int(match.group("domain")),
        elapsed_s=float(match.group("elapsed")),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def timing_files_for_namelist(namelist_path: Path) -> list[Path]:
    """Return the requested namelist plus sibling WRF rank-0 timing logs."""

    parent = namelist_path.parent
    candidates = [namelist_path, parent / "rsl.error.0000", parent / "rsl.out.0000"]
    return [path for path in candidates if path.exists()]


def parse_timing_files(paths: Iterable[Path]) -> dict[str, Any]:
    """Parse timing records from files and de-duplicate mirrored rsl records."""

    unique: dict[tuple[str, int, float], dict[str, Any]] = {}
    raw_count = 0
    source_counts: dict[str, int] = {}
    for path in paths:
        local_count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                record = parse_timing_line(line)
                if record is None:
                    continue
                raw_count += 1
                local_count += 1
                entry = unique.setdefault(
                    record.dedupe_key,
                    {
                        "timestamp": record.timestamp,
                        "domain": record.domain,
                        "elapsed_s": record.elapsed_s,
                        "sources": [],
                    },
                )
                entry["sources"].append({"path": str(path), "line": line_no})
        source_counts[str(path)] = local_count
    return {
        "raw_record_count": raw_count,
        "unique_record_count": len(unique),
        "duplicate_record_count": raw_count - len(unique),
        "source_record_counts": source_counts,
        "records": list(unique.values()),
    }


def _median_positive_delta_s(stamps: Sequence[datetime]) -> float | None:
    unique = sorted(set(stamps))
    deltas = [
        (right - left).total_seconds()
        for left, right in zip(unique[:-1], unique[1:])
        if (right - left).total_seconds() > 0
    ]
    if not deltas:
        return None
    return float(statistics.median(deltas))


def summarize_domains(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_domain.setdefault(int(record["domain"]), []).append(record)

    summaries: list[dict[str, Any]] = []
    for domain, domain_records in sorted(by_domain.items()):
        elapsed = [float(record["elapsed_s"]) for record in domain_records]
        stamps = [record["timestamp"] for record in domain_records]
        step_s = _median_positive_delta_s(stamps)
        expected_24h_steps = int(round(24 * 3600 / step_s)) if step_s else None
        first = min(stamps)
        last = max(stamps)
        summaries.append(
            {
                "domain": f"d{domain:02d}",
                "domain_id": domain,
                "step_count": len(domain_records),
                "total_wall_s": float(sum(elapsed)),
                "mean_per_step_s": float(statistics.fmean(elapsed)),
                "median_per_step_s": float(statistics.median(elapsed)),
                "min_per_step_s": float(min(elapsed)),
                "max_per_step_s": float(max(elapsed)),
                "first_model_time": first.isoformat(),
                "last_model_time": last.isoformat(),
                "coverage_s": float((last - first).total_seconds()),
                "median_model_step_s": step_s,
                "expected_24h_steps": expected_24h_steps,
                "complete_24h_timing": bool(expected_24h_steps is not None and len(domain_records) >= expected_24h_steps),
            }
        )
    return summaries


def _run_id_from_namelist(path: Path) -> str:
    return path.parent.name


def build_cpu_wall_clock_payload(namelist_paths: Sequence[Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for namelist in sorted(namelist_paths):
        timing_files = timing_files_for_namelist(namelist)
        parsed = parse_timing_files(timing_files)
        domains = summarize_domains(parsed["records"])
        d02 = next((domain for domain in domains if domain["domain_id"] == 2), None)
        status = "PASS" if d02 and d02["complete_24h_timing"] else "INCOMPLETE_TIMING"
        runs.append(
            {
                "run_id": _run_id_from_namelist(namelist),
                "run_path": str(namelist.parent),
                "status": status,
                "requested_namelist": str(namelist),
                "timing_files_scanned": [str(path) for path in timing_files],
                "source_record_counts": parsed["source_record_counts"],
                "raw_record_count": parsed["raw_record_count"],
                "unique_record_count": parsed["unique_record_count"],
                "duplicate_record_count": parsed["duplicate_record_count"],
                "namelist_timing_record_count": parsed["source_record_counts"].get(str(namelist), 0),
                "domains": domains,
            }
        )

    selected = select_speedup_run(runs)
    return {
        "schema": "M7CpuPerDomainWallClock",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "requested_namelist_paths": [str(path) for path in sorted(namelist_paths)],
        "timing_source_note": (
            "The requested namelist.output files contain zero Timing for main records on this workstation. "
            "The parser scans each namelist plus sibling rsl.error.0000/rsl.out.0000 files, then de-duplicates "
            "mirrored records before summing per-domain wall time."
        ),
        "selected_run_id": None if selected is None else selected["run_id"],
        "selected_run_path": None if selected is None else selected["run_path"],
        "selected_run": selected,
        "runs": runs,
    }


def select_speedup_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select the complete 24h run with the largest d02 timing coverage."""

    if not runs:
        return None

    def score(run: Mapping[str, Any]) -> tuple[int, float, str]:
        d02 = next((domain for domain in run.get("domains", []) if domain.get("domain_id") == 2), None)
        if not d02:
            return (0, 0.0, str(run.get("run_id", "")))
        complete = 1 if d02.get("complete_24h_timing") else 0
        return (complete, float(d02.get("step_count", 0)), str(run.get("run_id", "")))

    return max(runs, key=score)


def _domain_totals(selected_run: Mapping[str, Any]) -> dict[int, float]:
    return {int(domain["domain_id"]): float(domain["total_wall_s"]) for domain in selected_run.get("domains", [])}


def load_gpu_wall_s(pipeline_run_path: Path, fallback: float) -> float:
    if not pipeline_run_path.exists():
        return fallback
    payload = json.loads(pipeline_run_path.read_text(encoding="utf-8"))
    value = payload.get("wall_clock_total_s")
    return float(value) if value is not None else fallback


def _speedup_row(
    *,
    comparison_id: str,
    gpu_wall_s: float,
    cpu_wall_s: float,
    included_domains: Sequence[int],
    whats_being_compared: str,
    fairness_verdict: str,
    source_run_id: str,
) -> dict[str, Any]:
    return {
        "comparison_id": comparison_id,
        "gpu_wall_s": float(gpu_wall_s),
        "cpu_wall_s": float(cpu_wall_s),
        "ratio": float(cpu_wall_s / gpu_wall_s) if gpu_wall_s > 0 else None,
        "included_cpu_domains": [f"d{domain:02d}" for domain in included_domains],
        "whats_being_compared": whats_being_compared,
        "fairness_verdict": fairness_verdict,
        "source_run_id": source_run_id,
    }


def build_speedup_payload(cpu_payload: Mapping[str, Any], *, gpu_wall_s: float) -> dict[str, Any]:
    selected = cpu_payload.get("selected_run")
    if selected is None:
        raise ValueError("no timing run available for speedup table")
    totals = _domain_totals(selected)
    missing = [domain for domain in (1, 2, 3, 4, 5) if domain not in totals]
    if missing:
        raise ValueError(f"selected run is missing domains: {missing}")

    rows = [
        _speedup_row(
            comparison_id="cpu_full_nest_5_domain_aggregate_24h",
            gpu_wall_s=gpu_wall_s,
            cpu_wall_s=sum(totals[domain] for domain in (1, 2, 3, 4, 5)),
            included_domains=(1, 2, 3, 4, 5),
            whats_being_compared=(
                "GPU d02 24h end-to-end wall-clock versus de-duplicated aggregate sum of WRF per-domain "
                "Timing for main records across d01-d05."
            ),
            fairness_verdict="CONSERVATIVE_BUT_NOT_APPLES_TO_APPLES_FULL_NEST_AGGREGATE",
            source_run_id=str(selected["run_id"]),
        ),
        _speedup_row(
            comparison_id="cpu_d02_only_24h",
            gpu_wall_s=gpu_wall_s,
            cpu_wall_s=totals[2],
            included_domains=(2,),
            whats_being_compared="GPU d02 24h end-to-end wall-clock versus CPU d02-only cumulative WRF timing.",
            fairness_verdict="BEST_APPLES_TO_APPLES_DOMAIN_DENOMINATOR_AND_LOWEST_DEFENSIBLE_RATIO",
            source_run_id=str(selected["run_id"]),
        ),
        _speedup_row(
            comparison_id="cpu_d01_plus_d02_minimum_physical_subset_24h",
            gpu_wall_s=gpu_wall_s,
            cpu_wall_s=totals[1] + totals[2],
            included_domains=(1, 2),
            whats_being_compared=(
                "GPU d02 24h end-to-end wall-clock versus CPU d01+d02 timing, the minimum nested-domain "
                "subset that accounts for d02 boundary forcing context."
            ),
            fairness_verdict="PHYSICALLY_CONSERVATIVE_SUBSET_BUT_CPU_DENOMINATOR_INCLUDES_D01_WORK_GPU_DID_NOT_RUN",
            source_run_id=str(selected["run_id"]),
        ),
        _speedup_row(
            comparison_id="cpu_d01_only_24h",
            gpu_wall_s=gpu_wall_s,
            cpu_wall_s=totals[1],
            included_domains=(1,),
            whats_being_compared="GPU d02 24h end-to-end wall-clock versus CPU d01-only cumulative WRF timing.",
            fairness_verdict="CONTEXT_ONLY_NOT_A_D02_COMPARISON",
            source_run_id=str(selected["run_id"]),
        ),
    ]
    return {
        "schema": "M7HonestSpeedupTable",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_wall_source": str(DEFAULT_PIPELINE_RUN),
        "gpu_d02_24h_end_to_end_wall_s": float(gpu_wall_s),
        "selected_cpu_run_id": str(selected["run_id"]),
        "selected_cpu_run_path": str(selected["run_path"]),
        "timing_precision": "de-duplicated Timing for main records from sibling rsl logs; namelist.output had zero timing records",
        "rows": rows,
    }


def _resolve_namelists(pattern: str) -> list[Path]:
    paths = [Path(path) for path in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"no namelist files matched: {pattern}")
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namelist-glob", default=DEFAULT_NAMELIST_GLOB)
    parser.add_argument("--pipeline-run", type=Path, default=DEFAULT_PIPELINE_RUN)
    parser.add_argument("--gpu-wall-s", type=float, default=None)
    parser.add_argument("--cpu-output", type=Path, default=DEFAULT_CPU_OUTPUT)
    parser.add_argument("--speedup-output", type=Path, default=DEFAULT_SPEEDUP_OUTPUT)
    args = parser.parse_args()

    namelists = _resolve_namelists(args.namelist_glob)
    cpu_payload = build_cpu_wall_clock_payload(namelists)
    write_json(args.cpu_output, cpu_payload)
    gpu_wall = float(args.gpu_wall_s) if args.gpu_wall_s is not None else load_gpu_wall_s(args.pipeline_run, DEFAULT_GPU_WALL_S)
    speedup_payload = build_speedup_payload(cpu_payload, gpu_wall_s=gpu_wall)
    speedup_payload["gpu_wall_source"] = str(args.pipeline_run)
    write_json(args.speedup_output, speedup_payload)

    d02_row = next(row for row in speedup_payload["rows"] if row["comparison_id"] == "cpu_d02_only_24h")
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_cpu_run_id": speedup_payload["selected_cpu_run_id"],
                "d02_only_speedup": d02_row["ratio"],
                "cpu_output": str(args.cpu_output),
                "speedup_output": str(args.speedup_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
