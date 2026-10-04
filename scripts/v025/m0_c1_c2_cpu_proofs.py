#!/usr/bin/env python3
"""Reusable CPU-only proof fixtures for the M0 C1/C2 closure.

The complete census fixture is deliberately adversarial-proof data, not device
evidence.  It proves that the frozen arithmetic can reach ``OK`` when every
input exists and that each missing/mixed input blocks independently.  The W1b
helper drives the real post-lock exporter over the existing incomplete report
and must remain ``BLOCKED``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
W1B_REPORT = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/nsys_baseline.nsys-rep")
RUN_ID = "cpu-complete-census-fixture"
SOURCE_SHA256 = "b" * 64
WORKLOAD_SHA256 = "c" * 64


def _event(name: str, start: float, duration: float, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "start_ns": float(start),
        "duration_ns": float(duration),
        "end_ns": float(start + duration),
        **extra,
    }


def complete_census_inputs() -> dict[str, Any]:
    """Return one complete, internally independent §9 arithmetic fixture."""

    import m0_postlock_census as postlock
    import nsys_export

    nvtx_rows = [
        _event(postlock.RANGE_NAME, 0, 10_000),
        _event(f"{postlock.STEP_RANGE_PREFIX} step=0", 100, 2_300),
        _event(f"{postlock.STEP_RANGE_PREFIX} step=1", 2_500, 2_300),
        _event(f"{postlock.STEP_RANGE_PREFIX} step=2", 4_900, 2_300),
        _event(f"{postlock.STEP_RANGE_PREFIX} step=3", 7_300, 2_600),
    ]
    specifications = (
        (0, "rrtmg_lw_kernel", 600),
        (0, "kf_cumulus_kernel", 300),
        (0, "thompson_microphysics_kernel", 200),
        (1, "kf_cumulus_kernel", 400),
        (1, "thompson_microphysics_kernel", 300),
        (1, "mynn_pbl_kernel", 200),
        (1, "advect_u_flux_kernel", 100),
        (2, "kf_cumulus_kernel", 350),
        (2, "thompson_microphysics_kernel", 250),
        (2, "mynn_pbl_kernel", 180),
        (2, "advect_u_flux_kernel", 120),
        (3, "kf_cumulus_kernel", 300),
        (3, "thompson_microphysics_kernel", 200),
        (3, "mynn_pbl_kernel", 150),
        (3, "advect_u_flux_kernel", 100),
    )
    cursors = {0: 150.0, 1: 2_550.0, 2: 4_950.0, 3: 7_350.0}
    cuda_rows: list[dict[str, Any]] = []
    for step, name, duration in specifications:
        cuda_rows.append(_event(name, cursors[step], duration))
        cursors[step] += duration + 20.0
    cuda_rows.append(
        _event(
            "[CUDA memcpy Device-to-Device]",
            9_500,
            10,
            source_memory_kind="Device",
            destination_memory_kind="Device",
            bytes=4096,
        )
    )
    summary: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"launches": 0, "device_time_ns": 0.0}
    )
    for row in cuda_rows:
        if "memcpy" in row["name"].lower():
            continue
        summary[row["name"]]["launches"] += 1
        summary[row["name"]]["device_time_ns"] += row["duration_ns"]
    kernel_summary = [
        {"name": name, **values}
        for name, values in sorted(summary.items())
    ]

    required = list(nsys_export.REQUIRED)
    exports = {
        name: {
            "report": name,
            "status": "OK",
            "run_id": RUN_ID,
            "source_rep_sha256": SOURCE_SHA256,
            "source_rep_bytes": 1234,
            "path": f"/cpu-complete-fixture/{name}.csv",
            "sha256": f"{index + 1:x}" * 64,
            "data_rows": max(1, len(cuda_rows)),
            "returncode": 0,
            "private_sqlite_sha256": "9" * 64,
        }
        for index, name in enumerate(nsys_export.REPORTS)
    }
    export = {
        "schema": "wrf_gpu2.v025.m0.nsys_export.v2",
        "status": "OK",
        "run_id": RUN_ID,
        "source_rep": "/cpu-complete-fixture/capture.nsys-rep",
        "source_rep_sha256": SOURCE_SHA256,
        "source_rep_bytes": 1234,
        "private_sqlite_export":
            "/cpu-complete-fixture/private_export.sqlite",
        "private_sqlite_bytes": 4321,
        "private_sqlite_sha256": "9" * 64,
        "forced_sqlite_export_count": 1,
        "sqlite_report_read_count": len(exports) - 1,
        "required": required,
        "required_unusable": [],
        "exports": exports,
        "manifest_sha256": "d" * 64,
        "fixture_complete": True,
    }
    exact_child = {
        "schema": "wrf_gpu2.v025.m0.exact_executable_boundary.v1",
        "status": "OK",
        "run_id": RUN_ID,
        "allocator": {
            "peak_bytes_in_use": 2 << 30,
            "peak_bytes_reserved": 3 << 30,
        },
        "case": {
            "case_metadata": {
                "namelist": {
                    "dt_s": 900.0,
                    "radiation_cadence_steps": 4,
                }
            }
        },
        "call": {
            "hours": 1.0,
            "stablehlo": {
                "dtype_token_counts": {"f32": 100, "f64": 3},
                "convert_operation_count": 2,
            },
        },
        "timing": {
            "readiness_seconds": 5.0,
            "integration_seconds": 10.0,
        },
    }
    residency = {
        "schema": "wrf_gpu2.v025.m0.lock_owner_total_residency.v1",
        "status": "PASS",
        "run_id": RUN_ID,
        "sampling_quality": {"status": "PASS"},
        "peak_baseline_subtracted_bytes": 3 << 30,
    }
    host_rss = {
        "schema": "wrf_gpu2.v025.m0.process_tree_rss.v1",
        "status": "PASS",
        "run_id": RUN_ID,
        "peak_process_tree_rss_bytes": 4 << 30,
    }
    identity = {
        "workload_identity_sha256": WORKLOAD_SHA256,
        "integration_scope_sha256": "e" * 64,
        "event_mix_sha256": "f" * 64,
        "timing_region": "integration-only",
    }
    matched_profiler = {
        "schema": "wrf_gpu2.v025.m0.profiler_matched_pair.v1",
        "profiled": {
            **identity,
            "run_id": "profiled-fixture",
            "seconds": 11.0,
            "instrumentation": "nsys",
        },
        "unprofiled": {
            **identity,
            "run_id": "clean-fixture",
            "seconds": 10.0,
            "instrumentation": "none",
        },
        "order": ["profiled-fixture", "clean-fixture"],
    }
    clean_exact_child = {
        "status": "OK",
        "run_id": "clean-fixture",
        "timing": {"integration_seconds": 10.0},
    }
    reference_totals = {
        "physics.cumulus": 1350.0,
        "physics.microphysics": 950.0,
        "physics.radiation": 600.0,
        "physics.pbl": 530.0,
        "dycore.advection": 320.0,
    }
    production_reference = {
        "case_role": "matched_short_two_domain_production",
        "run_id": "production-reference-fixture",
        "source_rep": {
            "path": "/cpu-complete-fixture/reference.nsys-rep",
            "sha256": "a" * 64,
        },
        "workload_identity_sha256": WORKLOAD_SHA256,
        "non_nesting_families": reference_totals,
        "rank_families": [
            "physics.cumulus",
            "physics.microphysics",
            "physics.radiation",
            "physics.pbl",
            "dycore.advection",
        ],
        "active_d01_schemes": ["mp=8", "ra=4", "pbl=5", "cu=1"],
        "cadence_events": ["ordinary", "radiation", "cumulus"],
    }
    candidate_coverage = {
        "run_id": RUN_ID,
        "capture_run_id": RUN_ID,
        "source_rep_sha256": SOURCE_SHA256,
        "workload_identity_sha256": WORKLOAD_SHA256,
        "executed_schemes": ["mp=8", "ra=4", "pbl=5", "cu=1"],
        "executed_cadence_events": [
            "ordinary",
            "radiation",
            "cumulus",
        ],
    }
    return {
        "export": export,
        "nvtx_rows": nvtx_rows,
        "cuda_rows": cuda_rows,
        "kernel_summary": kernel_summary,
        "run_id": RUN_ID,
        "exact_child": exact_child,
        "release_binding": {
            "schema": postlock.RELEASE_SCHEMA,
            "run_id": RUN_ID,
            "release_before_analysis": True,
        },
        "production_derived": True,
        "residency": residency,
        "host_rss": host_rss,
        "matched_profiler": matched_profiler,
        "clean_exact_child": clean_exact_child,
        "production_reference": production_reference,
        "candidate_coverage": candidate_coverage,
        "cold_readiness_seconds": 20.0,
        "hlo_evidence_override": {
            "fixture_complete": True,
            "status": "PASS",
            "operator_count": 49,
            "dtype_and_convert_census": {"status": "PASS"},
            "compiled_memory_analysis_gate": {
                "status": "PASS",
                "required_fields": [
                    "temp_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                ],
            },
            "peak_live_buffers": "MISSING",
        },
    }


def complete_census_proof() -> dict[str, Any]:
    """Build the complete fixture and independently recheck its arithmetic."""

    import m0_postlock_census as postlock
    import run_gpu_arm

    inputs = complete_census_inputs()
    census = postlock.build_census(**inputs)
    postlock.validate_census(census)
    kernels = [
        row
        for row in inputs["cuda_rows"]
        if "memcpy" not in row["name"].lower()
    ]
    expected_launches = len(kernels)
    expected_time = sum(float(row["duration_ns"]) for row in kernels)
    expected_families: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"launches": 0.0, "device_time_ns": 0.0}
    )
    for row in kernels:
        family = run_gpu_arm.attribute_kernel(row["name"])
        expected_families[family]["launches"] += 1
        expected_families[family]["device_time_ns"] += row["duration_ns"]
    observed_families = census["family_census"]
    arithmetic = {
        "top_level_ok": census["status"] == "OK",
        "launches_exact": (
            census["device_time_attribution"]["total_launches"]
            == expected_launches
        ),
        "device_time_exact": math.isclose(
            census["device_time_attribution"]["total_device_time_ns"],
            expected_time,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "families_exact": all(
            observed_families[name]["launches"] == int(values["launches"])
            and math.isclose(
                observed_families[name]["device_time_ns"],
                values["device_time_ns"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for name, values in expected_families.items()
        ),
        "four_steps_exact": (
            census["step_census"]["derived_steps"] == 4
            and sum(
                step["kernel_launches"]
                for step in census["step_census"]["steps"]
            )
            == expected_launches
        ),
        "ordinary_and_radiation_present": (
            census["step_census"]["classes"]["ordinary"]["steps"] == 3
            and census["step_census"]["classes"]["radiation"]["steps"] == 1
        ),
        "zero_timestep_host_device_transfers": (
            census["transfer_audit"]["host_device_in_loop"] == 0
        ),
    }
    if not all(arithmetic.values()):
        raise RuntimeError(f"complete census arithmetic failed: {arithmetic}")
    return {
        "status": "PASS",
        "fixture_is_device_evidence": False,
        "arithmetic": arithmetic,
        "expected_launches": expected_launches,
        "expected_device_time_ns": expected_time,
        "census": census,
    }


def census_attack_matrix() -> dict[str, Any]:
    """Attack every independently required C2 gate; all must fail closed."""

    import m0_postlock_census as postlock

    attacks: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("export_blocked", lambda x: x["export"].update(status="BLOCKED")),
        (
            "required_table_empty",
            lambda x: x["export"]["exports"]["cuda_gpu_trace"].update(
                status="EMPTY", data_rows=0
            ),
        ),
        (
            "export_record_wrong_run",
            lambda x: x["export"]["exports"]["nvtx_pushpop_trace"].update(
                run_id="stale-run"
            ),
        ),
        (
            "export_record_wrong_source",
            lambda x: x["export"]["exports"]["cuda_gpu_trace"].update(
                source_rep_sha256="a" * 64
            ),
        ),
        (
            "release_wrong_run",
            lambda x: x["release_binding"].update(run_id="stale-run"),
        ),
        (
            "exact_child_wrong_run",
            lambda x: x["exact_child"].update(run_id="stale-run"),
        ),
        (
            "candidate_stale_source",
            lambda x: x["candidate_coverage"].update(
                source_rep_sha256="a" * 64
            ),
        ),
        ("missing_nvtx", lambda x: x.update(nvtx_rows=[])),
        (
            "wrong_integration_range",
            lambda x: x["nvtx_rows"][0].update(name="WRONG_RANGE"),
        ),
        ("empty_cuda_trace", lambda x: x.update(cuda_rows=[])),
        (
            "aggregate_transfer_substitution",
            lambda x: x.update(
                cuda_rows=[
                    {
                        "name": "[CUDA memcpy HtoD]",
                        "instances": 4,
                        "seconds": 1.0,
                    }
                ]
            ),
        ),
        (
            "host_device_copy_in_loop",
            lambda x: x["cuda_rows"].append(
                _event(
                    "[CUDA memcpy HtoD]",
                    9_600,
                    5,
                    source_memory_kind="Host",
                    destination_memory_kind="Device",
                )
            ),
        ),
        (
            "launch_attribution_below_95",
            lambda x: x["cuda_rows"].extend(
                _event(f"unknown_launch_{index}", 9_000 + index, 0.001)
                for index in range(100)
            ),
        ),
        (
            "time_attribution_below_95",
            lambda x: (
                x["cuda_rows"].extend(
                    _event(
                        f"advect_known_extra_{index}",
                        8_500 + index,
                        0.001,
                    )
                    for index in range(20)
                ),
                x["cuda_rows"].append(
                    _event("unknown_heavy_kernel", 8_900, 1_000)
                ),
            ),
        ),
        (
            "missing_radiation_cadence",
            lambda x: x["cuda_rows"][0].update(name="advect_only_kernel"),
        ),
        (
            "kernel_summary_understates_trace",
            lambda x: x.update(kernel_summary=[
                {"name": "advect", "launches": 1, "device_time_ns": 1}
            ]),
        ),
        ("missing_profiler_pair", lambda x: x.update(matched_profiler=None)),
        ("missing_clean_w3", lambda x: x.update(clean_exact_child=None)),
        ("missing_host_rss", lambda x: x.update(host_rss=None)),
        ("missing_residency", lambda x: x.update(residency=None)),
        (
            "missing_compiled_memory_analysis",
            lambda x: x["hlo_evidence_override"].update(
                compiled_memory_analysis_gate={"status": "MISSING"}
            ),
        ),
        (
            "production_workload_mismatch",
            lambda x: x["production_reference"].update(
                workload_identity_sha256="d" * 64
            ),
        ),
        (
            "synthetic_production_scope",
            lambda x: x.update(production_derived=False),
        ),
    ]
    results = []
    for name, mutate in attacks:
        inputs = complete_census_inputs()
        try:
            mutate(inputs)
            census = postlock.build_census(**inputs)
            rejected = census.get("status") == "BLOCKED"
            detail = {
                "status": census.get("status"),
                "gates_not_pass": census.get("gates_not_pass"),
            }
        except Exception as exc:  # malformed evidence is also a closed gate
            rejected = True
            detail = {"exception": f"{type(exc).__name__}: {exc}"}
        results.append(
            {"name": name, "rejected": rejected, "detail": detail}
        )
    return {
        "status": "PASS" if all(item["rejected"] for item in results) else "FAIL",
        "rejected": sum(item["rejected"] for item in results),
        "total": len(results),
        "attacks": results,
    }


def _artifact(path: Path) -> dict[str, Any]:
    import m0_postlock_census as postlock

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": postlock.sha256_file(path),
    }


def run_incomplete_w1b_postlock(
    *,
    report_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Drive the real CPU-only W2 post-lock path over the incomplete W1b report."""

    import m0_postlock_census as postlock

    if not report_path.is_file():
        raise RuntimeError(f"incomplete W1b report is absent: {report_path}")
    run_root.mkdir(parents=True, exist_ok=False)
    run_id = "cpu-w1b-postlock-path"
    exact_path = run_root / "exact.json"
    allocator_path = run_root / "allocator.json"
    residency_path = run_root / "residency.json"
    manager_path = run_root / "w2.window.json"
    release_path = run_root / "w2.lock-release.json"
    output_path = run_root / "w2.post.json"
    exact = {
        "schema": "wrf_gpu2.v025.m0.exact_executable_boundary.v1",
        "status": "OK",
        "run_id": run_id,
        "instrumentation": {
            "mode": "profiled",
            "nvtx_range": postlock.RANGE_NAME,
        },
        "allocator": {"peak_bytes_in_use": 1, "peak_bytes_reserved": 1},
        "call": {"hours": 1.0},
        "case": {"case_metadata": {"namelist": {"dt_s": 10.0}}},
        "timing": {"readiness_seconds": 1.0, "integration_seconds": 1.0},
    }
    residency = {
        "schema": "wrf_gpu2.v025.m0.lock_owner_total_residency.v1",
        "status": "PASS",
        "run_id": run_id,
        "sampling_quality": {"status": "PASS"},
    }
    postlock._atomic_json_no_replace(exact_path, exact)
    postlock._atomic_json_no_replace(
        allocator_path,
        {"schema": "cpu-only-incomplete-fixture", "run_id": run_id},
    )
    postlock._atomic_json_no_replace(residency_path, residency)
    artifacts = {
        "exact_boundary": _artifact(exact_path),
        "source_rep": _artifact(report_path),
        "allocator": _artifact(allocator_path),
        "lock_owner_total_residency": _artifact(residency_path),
    }
    manager = {
        "schema": "wrf_gpu2.v025.m0.manager_window.v1",
        "status": "OK",
        "window": "W2",
        "run_id": run_id,
        "stages": [
            {
                "name": "profiled_cached_readiness_and_integration",
                "status": "OK",
                "host_rss_path": None,
            },
            {
                "name": "profiled_artifact_capture_integrity",
                "status": "PASS",
                "postlock_export_and_census_status": "MISSING",
                "artifacts": artifacts,
            },
        ],
    }
    postlock._atomic_json_no_replace(manager_path, manager)
    returned_ns = time.monotonic_ns()
    release = postlock.build_lock_release_proof(
        window_id="W2",
        run_id=run_id,
        wrapper_command=["CPU_DRY_LOCK_WRAPPER_RETURN_FIXTURE"],
        wrapper_started_monotonic_ns=returned_ns - 2,
        wrapper_returned_monotonic_ns=returned_ns - 1,
        wrapper_returncode=0,
        window_result_path=manager_path,
        output_path=release_path,
        wrapper_started_at_utc=datetime.now(timezone.utc).isoformat(),
        wrapper_returned_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    census = postlock.analyze_w2(
        w2_result_path=manager_path,
        release_path=release_path,
        output_path=output_path,
        export_root=run_root / "exports",
    )
    if census.get("status") != "BLOCKED":
        raise RuntimeError("incomplete W1b report produced a false census pass")
    return {
        "status": "PASS",
        "path_status": "BLOCKED_AS_REQUIRED",
        "device_action": False,
        "source_report": _artifact(report_path),
        "manager_result": _artifact(manager_path),
        "release_proof": {
            **release,
            "path": str(release_path),
            "sha256": postlock.sha256_file(release_path),
        },
        "analysis": {
            "path": str(output_path),
            "sha256": postlock.sha256_file(output_path),
            "status": census["status"],
            "gates_not_pass": census["gates_not_pass"],
            "export": census["export"],
            "export_status": census["export"]["status"],
            "required_unusable": census["export"]["required_unusable"],
            "parse_errors": census["parse_errors"],
            "release_before_analysis":
                census["release_provenance"]["release_before_analysis"],
            "accelerator_modules_imported":
                census["analysis_process"]["accelerator_modules_imported"],
            "lock_environment_present":
                census["analysis_process"]["lock_environment_present"],
        },
        "run_root": str(run_root),
    }


def write_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    import m0_postlock_census as postlock

    postlock._atomic_json_no_replace(path, payload)


__all__ = [
    "RUN_ID",
    "SOURCE_SHA256",
    "W1B_REPORT",
    "census_attack_matrix",
    "complete_census_inputs",
    "complete_census_proof",
    "run_incomplete_w1b_postlock",
    "write_json_no_replace",
]
