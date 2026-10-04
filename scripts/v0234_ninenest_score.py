#!/usr/bin/env python3
"""CPU-only aggregate scorer for the established v0.19 nine-domain field gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring


REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpt-ninenest-replay")
COMPARATOR = REPO / "scripts/compare_wrfout_grid.py"
TOLERANCES = REPO / "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json"
HARNESS_CORRECTION = REPO / ".agent/patches/2026-07-22-v0234-ninenest-terminal-harness-correction.md"
DOMAINS = tuple(f"d{i:02d}" for i in range(1, 10))
HARD_FIELDS = {"T2", "U10", "V10", "PSFC", "RAINNC", "T", "U", "V", "W", "QVAPOR"}
BASE_FIELDS = {"HGT", "PB", "PHB", "MUB"}


def expected_names() -> dict[str, list[str]]:
    result = {}
    for domain in DOMAINS:
        times = ("18:20:06", "18:40:12", "19:00:00") if domain == "d01" else ("18:20:00", "18:40:00", "19:00:00")
        result[domain] = [f"wrfout_{domain}_2026-04-28_{stamp}" for stamp in times]
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "canonical_payload_sha256"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_times(ds: Dataset) -> str:
    arr = np.asarray(ds.variables["Times"][:])
    converted = chartostring(arr)
    return str(np.asarray(converted).ravel()[0])


def finite_inventory(path: Path, expected_time: str) -> dict[str, object]:
    numeric_count = 0
    numeric_values = 0
    failures = []
    with Dataset(path, "r") as ds:
        observed_time = read_times(ds)
        if observed_time != expected_time:
            failures.append({"field": "Times", "observed": observed_time, "expected": expected_time})
        for name, variable in ds.variables.items():
            if np.dtype(variable.dtype).kind not in {"b", "i", "u", "f", "c"}:
                continue
            numeric_count += 1
            raw = variable[:]
            arr = np.asarray(np.ma.filled(raw, np.nan))
            numeric_values += int(arr.size)
            finite = np.isfinite(arr)
            if not bool(np.all(finite)):
                failures.append({"field": name, "nonfinite": int(arr.size - np.count_nonzero(finite)), "values": int(arr.size)})
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "numeric_field_count": numeric_count,
        "numeric_value_count": numeric_values,
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pipeline-proof", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected = expected_names()
    actual_names = sorted(path.name for path in args.gpu_dir.glob("wrfout_d??_*") if path.is_file())
    flat_expected = sorted(name for names in expected.values() for name in names)
    inventory = []
    for domain, names in expected.items():
        for name in names:
            path = args.gpu_dir / name
            expected_time = name.split(f"wrfout_{domain}_", 1)[1]
            if path.is_file():
                inventory.append(finite_inventory(path, expected_time))
            else:
                inventory.append({"path": str(path), "pass": False, "failures": [{"missing": True}]})

    manifest = json.loads(TOLERANCES.read_text())
    manifest_fields = set(manifest["fields"])
    report_only_fields = {
        name for name, spec in manifest["fields"].items() if spec.get("gate") == "critical_report_only"
    }
    enforced_manifest_fields = manifest_fields - report_only_fields
    static_fields = {name for name, spec in manifest["fields"].items() if spec.get("gate") == "static_exactness"}
    domain_results = {}
    for domain in DOMAINS:
        out_json = args.out_dir / f"grid_compare_{domain}.json"
        out_md = args.out_dir / f"grid_compare_{domain}.md"
        command = [
            sys.executable,
            str(COMPARATOR),
            "--cpu-dir", str(args.cpu_dir),
            "--gpu-dir", str(args.gpu_dir),
            "--domain", domain,
            "--init", "2026-04-28T18:00:00Z",
            "--min-lead", "0",
            "--max-lead", "1",
            "--tolerance-json", str(TOLERANCES),
            "--progress", "25",
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0 or not out_json.is_file():
            domain_results[domain] = {
                "pass": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
            continue
        report = json.loads(out_json.read_text())
        fields = report["field_summaries"]
        summaries = report["summaries"]
        observed_pairs = [Path(row["gpu_file"]).name for row in report["pairing"]["pairs"]]
        required_results = {
            name: fields.get(name, {}).get("tolerance_result", {}).get("pass")
            for name in sorted(enforced_manifest_fields)
        }
        base_max = {name: fields.get(name, {}).get("overall", {}).get("max_abs") for name in sorted(BASE_FIELDS)}
        checks = {
            "comparator_verdict_pass": summaries.get("verdict") == "PASS",
            "zero_tolerance_failures": summaries.get("tolerance_failure_count") == 0,
            "paired_file_count_three": report["pairing"].get("paired_file_count") == 3,
            "paired_names_exact": observed_pairs == expected[domain],
            "compared_numeric_fields_102": summaries.get("comparable_field_count") == 102,
            "all_manifest_fields_present": manifest_fields.issubset(fields),
            "all_enforced_manifest_field_gates_pass": all(value is True for value in required_results.values()),
            "all_report_only_fields_present": report_only_fields.issubset(fields),
            "all_hard_fields_present": HARD_FIELDS.issubset(fields),
            "all_static_fields_present": static_fields.issubset(fields),
            "base_fields_at_most_0_2": all(value is not None and math.isfinite(float(value)) and float(value) <= 0.2 for value in base_max.values()),
        }
        domain_results[domain] = {
            "pass": all(checks.values()),
            "checks": checks,
            "comparator_json": str(out_json),
            "comparator_json_sha256": sha256(out_json),
            "comparator_md": str(out_md),
            "comparator_md_sha256": sha256(out_md),
            "summaries": summaries,
            "base_state_max_abs": base_max,
            "hard_field_metrics": {name: fields[name]["overall"] for name in sorted(HARD_FIELDS) if name in fields},
        }

    pipeline = json.loads(args.pipeline_proof.read_text())
    preflight = json.loads(args.preflight.read_text())
    event_counts = pipeline.get("hierarchy", {}).get("event_counts", {})
    per_domain = pipeline.get("per_domain", {})
    domain_meta = pipeline.get("metadata", {}).get("domains", {})
    pipeline_checks = {
        "verdict_green": pipeline.get("verdict") == "PIPELINE_GREEN",
        "hours_one": pipeline.get("hours") == 1,
        "max_dom_nine": pipeline.get("max_dom") == 9,
        "domains_exact": pipeline.get("domains") == list(DOMAINS),
        "all_domains_finite": pipeline.get("all_domains_finite") is True,
        "all_outputs_present": pipeline.get("all_outputs_present") is True,
        "event_advance_5000_exact_hour": event_counts.get("advance") == 5000,
        "event_force_4400_exact_hour": event_counts.get("force") == 4400,
        "event_output_27": event_counts.get("output") == 27,
        "event_feedback_zero": event_counts.get("feedback", 0) == 0,
        "own_steps_exact": per_domain.get("d01", {}).get("own_steps") == 200 and per_domain.get("d02", {}).get("own_steps") == 600 and all(per_domain.get(domain, {}).get("own_steps") == 1800 for domain in DOMAINS[2:]),
        "three_outputs_each": all(per_domain.get(domain, {}).get("wrfout_count") == 3 and per_domain.get(domain, {}).get("expected_wrfout_count") == 3 for domain in DOMAINS),
        "twenty_minute_cadence_each": all(per_domain.get(domain, {}).get("history_interval_min") == 20 for domain in DOMAINS),
        "d01_boundary_default_false": domain_meta.get("d01", {}).get("namelist", {}).get("nested_frozen_wrf_boundary_bundle") is False,
        "all_child_boundary_defaults_true": all(domain_meta.get(domain, {}).get("namelist", {}).get("nested_frozen_wrf_boundary_bundle") is True for domain in DOMAINS[1:]),
        "fused_default_path": pipeline.get("metadata", {}).get("parallel_compile", {}).get("nested_precompile", {}).get("source") == "skip:fused-default",
        "aot_enabled": pipeline.get("metadata", {}).get("nested_aot", {}).get("enabled") is True,
        "prepared_runtime_reuse": pipeline.get("metadata", {}).get("nested_runtime", {}).get("prepared_runtime_reuse") is True,
        "established_nested_radiation_cadence": domain_meta.get("d01", {}).get("namelist", {}).get("radiation_cadence_steps") == 100 and domain_meta.get("d02", {}).get("namelist", {}).get("radiation_cadence_steps") == 300 and all(domain_meta.get(domain, {}).get("namelist", {}).get("radiation_cadence_steps") == 900 for domain in DOMAINS[2:]),
        "cadvariant_gwd_zero": all(domain_meta.get(domain, {}).get("namelist", {}).get("gwd_opt") == 0 for domain in DOMAINS),
    }
    checks = {
        "static_preflight_pass": preflight.get("verdict") == "STATIC_PREFLIGHT_PASS",
        "gpu_inventory_names_exact": actual_names == flat_expected,
        "gpu_inventory_count_27": len(actual_names) == 27,
        "all_gpu_numeric_values_finite": all(row.get("pass") is True for row in inventory),
        "pipeline_all_pass": all(pipeline_checks.values()),
        "all_nine_comparators_pass": len(domain_results) == 9 and all(result.get("pass") is True for result in domain_results.values()),
    }
    verdict = "NINE_NEST_ACCEPTANCE_CRITERION_5_GREEN" if all(checks.values()) else "NINE_NEST_ACCEPTANCE_CRITERION_5_RED"
    payload = {
        "schema": "wrfgpu2.v0234.ninenest-terminal-proof.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": checks,
        "pipeline_checks": pipeline_checks,
        "gpu_output_inventory": inventory,
        "domain_results": domain_results,
        "authorities": {
            "pipeline_proof": str(args.pipeline_proof),
            "pipeline_proof_sha256": sha256(args.pipeline_proof),
            "preflight": str(args.preflight),
            "preflight_sha256": sha256(args.preflight),
            "comparator": str(COMPARATOR),
            "comparator_sha256": sha256(COMPARATOR),
            "tolerance_manifest": str(TOLERANCES),
            "tolerance_manifest_sha256": sha256(TOLERANCES),
            "harness_authority_correction": str(HARNESS_CORRECTION),
            "harness_authority_correction_sha256": sha256(HARNESS_CORRECTION),
            "cpu_truth_dir": str(args.cpu_dir),
            "gpu_candidate_dir": str(args.gpu_dir),
        },
        "cpu_only_scoring": True,
        "gpu_used_by_scorer": False,
    }
    payload["canonical_payload_sha256"] = canonical(payload)
    proof_path = args.out_dir / "proof.json"
    proof_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "proof": str(proof_path), "canonical_payload_sha256": payload["canonical_payload_sha256"]}, indent=2))
    return 0 if verdict.endswith("_GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
