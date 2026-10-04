#!/usr/bin/env python3
"""Analyze captured v0.23.4 prepared-runtime A/B artifacts without using a GPU."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resource_summary(arm_dir: Path) -> dict[str, int]:
    rss: list[int] = []
    swap: list[int] = []
    available: list[int] = []
    with (arm_dir / "process_resources.tsv").open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        for row in rows:
            rss.append(int(row["vmrss_kib"]))
            swap.append(int(row["vmswap_kib"]))
            available.append(int(row["memavailable_kib"]))
    vram: list[int] = []
    with (arm_dir / "nvidia_smi.csv").open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                vram.append(int(row[1].strip()))
    return {
        "peak_rss_kib": max(rss, default=0),
        "peak_swap_kib": max(swap, default=0),
        "minimum_host_available_kib": min(available, default=0),
        "peak_vram_mib": max(vram, default=0),
    }


def _valid_time(path: str) -> datetime:
    name = Path(path).name
    marker = "wrfout_d09_"
    if marker not in name:
        raise ValueError(f"not a d09 wrfout path: {path}")
    return datetime.strptime(name.split(marker, 1)[1], "%Y-%m-%d_%H:%M:%S")


def _timeline(arm_dir: Path) -> dict[str, Any]:
    frames: list[tuple[datetime, float, str]] = []
    with (arm_dir / "wrfout_mtimes.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            path, mtime, _size = line.rstrip("\n").split("\t")
            if "wrfout_d09_" in path:
                frames.append((_valid_time(path), float(mtime), path))
    frames.sort()
    ordinary: list[float] = []
    post_hour: list[float] = []
    intervals: list[dict[str, Any]] = []
    for previous, current in zip(frames, frames[1:], strict=False):
        gap = current[1] - previous[1]
        kind = "post_hour" if previous[0].minute == 0 else "ordinary"
        (post_hour if kind == "post_hour" else ordinary).append(gap)
        intervals.append(
            {
                "from_valid": previous[0].isoformat(),
                "to_valid": current[0].isoformat(),
                "wall_gap_s": gap,
                "kind": kind,
            }
        )
    wall = _read_json(arm_dir / "wall.json")
    start_epoch = float(wall["start_epoch_ns"]) / 1.0e9
    last_frame_wall = max((frame[1] for frame in frames), default=start_epoch) - start_epoch
    return {
        "d09_frame_count": len(frames),
        "command_wall_s": float(wall["command_wall_s"]),
        "command_to_last_d09_frame_s": last_frame_wall,
        "ordinary_group_gaps_s": ordinary,
        "ordinary_group_median_s": statistics.median(ordinary) if ordinary else None,
        "post_hour_group_gaps_s": post_hour,
        "post_hour_group_median_s": statistics.median(post_hour) if post_hour else None,
        "intervals": intervals,
    }


def _sha_inventory(arm_dir: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    with (arm_dir / "wrfout_sha256.txt").open(encoding="utf-8") as handle:
        for line in handle:
            digest, path = line.rstrip("\n").split(maxsplit=1)
            inventory[path] = digest
    return inventory


def _ratio_gain(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else float("-inf")


def analyze(output_root: Path, manifest: Path) -> dict[str, Any]:
    selection = _read_json(manifest)["prepared_runtime_ab_prearm"]
    arms: dict[str, dict[str, Any]] = {}
    for label in ("a", "b"):
        arm_dir = output_root / f"arm_{label}"
        proof = _read_json(arm_dir / "proof" / "nested_pipeline_run.json")
        arms[label.upper()] = {
            "proof_verdict": proof.get("verdict"),
            "all_domains_finite": proof.get("all_domains_finite"),
            "all_outputs_present": proof.get("all_outputs_present"),
            "load_count": proof.get("metadata", {}).get("nested_aot", {}).get("load_count"),
            "prepared_runtime_reuse": proof.get("metadata", {}).get("nested_runtime", {}).get("prepared_runtime_reuse"),
            "timeline": _timeline(arm_dir),
            "resources": _resource_summary(arm_dir),
            "wrfout_sha256": _sha_inventory(arm_dir),
        }

    a = arms["A"]
    b = arms["B"]
    a_timeline = a["timeline"]
    b_timeline = b["timeline"]
    command_gain = _ratio_gain(
        float(a_timeline["command_wall_s"]),
        float(b_timeline["command_wall_s"]),
    )
    last_frame_gain = _ratio_gain(
        float(a_timeline["command_to_last_d09_frame_s"]),
        float(b_timeline["command_to_last_d09_frame_s"]),
    )
    a_post = a_timeline["post_hour_group_median_s"]
    b_post = b_timeline["post_hour_group_median_s"]
    post_gain = _ratio_gain(float(a_post), float(b_post)) if a_post and b_post else float("-inf")
    a_ordinary = a_timeline["ordinary_group_median_s"]
    b_ordinary = b_timeline["ordinary_group_median_s"]
    ordinary_regression = (
        (float(b_ordinary) - float(a_ordinary)) / float(a_ordinary)
        if a_ordinary and b_ordinary
        else float("inf")
    )
    variable_identity_path = output_root / "wrfout_variable_identity.txt"
    variables_exact = (
        variable_identity_path.is_file()
        and "verdict=BYTE-IDENTICAL" in variable_identity_path.read_text(encoding="utf-8")
    )
    file_bytes_identical = a["wrfout_sha256"] == b["wrfout_sha256"]
    gates = {
        "arm_a_green": a["proof_verdict"] == "PIPELINE_GREEN" and a["all_domains_finite"] is True and a["all_outputs_present"] is True,
        "arm_b_green": b["proof_verdict"] == "PIPELINE_GREEN" and b["all_domains_finite"] is True and b["all_outputs_present"] is True,
        "arm_controls_exact": a["prepared_runtime_reuse"] is False and b["prepared_runtime_reuse"] is True,
        "load_count_27_to_9": a["load_count"] == 27 and b["load_count"] == 9,
        "wrfout_file_bytes_identical_or_variables_exact": file_bytes_identical or variables_exact,
        "command_wall_gain_at_least_20pct": command_gain >= 0.20,
        "post_hour_gap_at_most_180s": b_post is not None and float(b_post) <= 180.0,
        "post_hour_gain_at_least_25pct": post_gain >= 0.25,
        "ordinary_group_regression_at_most_5pct": ordinary_regression <= 0.05,
        "candidate_vram_delta_at_most_512mib": b["resources"]["peak_vram_mib"] - a["resources"]["peak_vram_mib"] <= 512,
        "candidate_swap_delta_at_most_256mib": b["resources"]["peak_swap_kib"] - a["resources"]["peak_swap_kib"] <= 262144,
    }
    automated_green = all(gates.values())
    return {
        "schema": "wrf_gpu.v0234.prepared_runtime_ab.analysis.v1",
        "selection_status": selection["selection_status"],
        "arms": arms,
        "metrics": {
            "command_wall_gain_fraction": command_gain,
            "command_to_last_frame_gain_fraction_diagnostic": last_frame_gain,
            "post_hour_gain_fraction": post_gain,
            "ordinary_group_regression_fraction": ordinary_regression,
            "wrfout_file_bytes_identical": file_bytes_identical,
            "wrfout_variables_exact": variables_exact,
        },
        "automated_gates": gates,
        "automated_verdict": "GREEN_PENDING_MANUAL_PROFILE_REVIEW" if automated_green else "REJECT_OR_INVALID",
        "manual_gates_remaining": [
            "Nsight in-loop transfer and synchronization-window audit",
            "segment-by-segment RSS/VRAM growth and post-warmup swap audit",
            "required variable/range/grid/D0/pair/native7/thin QA review",
            "named NetCDF metadata-only difference review if file bytes differ",
            "preemption/production-load validity and profiler completeness review",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.output_root, args.manifest)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload["automated_verdict"] != "REJECT_OR_INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
