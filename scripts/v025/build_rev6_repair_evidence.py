#!/usr/bin/env python3
"""Build the CPU-only proof object for the P5-REV6 replacement repair."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v025"))

# Defense in depth.  This generator performs only parsing, subprocess lifecycle
# tests, and the dry stub path.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import baseline_census as bc  # noqa: E402
import h1_discriminator as h1  # noqa: E402
import nsys_export as nex  # noqa: E402
import parse_profiler as pp  # noqa: E402
import step1_driver as drv  # noqa: E402
import step1_dryrun as dry  # noqa: E402

LIVE_W1B_REP = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/nsys_baseline.nsys-rep")


def _sha(path: Path) -> str:
    return drv.sha256_file(path)


def _rho_mutation() -> dict[str, Any]:
    names = [f"f{i}" for i in range(6)]
    reference_values = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    candidate_values = [4.0, 5.0, 6.0, 3.0, 2.0, 1.0]
    reference = {
        "case_role": "matched_short_two_domain_production",
        "run_id": "proof-reference",
        "source_rep": {"path": "proof", "sha256": "a" * 64},
        "workload_identity_sha256": "c" * 64,
        "non_nesting_families": dict(zip(names, reference_values)),
        "rank_families": names,
        "active_d01_schemes": ["all-active-d01"],
        "cadence_events": ["ordinary", "radiation"],
    }
    candidate = {
        name: {"device_time_share": value}
        for name, value in zip(names, candidate_values)
    }
    coverage = {
        "run_id": "proof-candidate",
        "capture_run_id": "proof-candidate",
        "source_rep_sha256": "b" * 64,
        "workload_identity_sha256": "c" * 64,
        "executed_schemes": ["all-active-d01"],
        "executed_cadence_events": ["ordinary", "radiation"],
    }
    gate = bc.production_representativeness_gate(candidate, reference, coverage)
    return {
        "input_rho_between_old_and_frozen_bars": gate["spearman"],
        "frozen_minimum": gate["min_spearman"],
        "status": gate["status"],
        "passed": gate["status"] == "FAILED" and 0.70 < gate["spearman"] < 0.80,
    }


def _transfer_mutation() -> dict[str, Any]:
    aggregate = [{"name": "[CUDA memcpy HtoD]", "instances": 7, "seconds": 1.0}]
    scope = {
        "schema": "wrf_gpu2.v025.m0.integration_scope.v1",
        "status": "OK",
        "run_id": "proof-candidate",
        "capture_run_id": "proof-candidate",
        "source_rep_sha256": "b" * 64,
        "production_derived": True,
        "mechanically_verified": True,
        "boundary_kind": "integration",
        "boundary_source": "nvtx_pushpop_trace",
        "start_ns": 0,
        "end_ns": 10,
    }
    gate = bc.transfer_gate(aggregate, scope)
    live_rows = pp.parse_cuda_gpu_trace(
        "Start (ns),Duration (ns),SrcMemKd,DstMemKd,Bytes (MB),Name\n"
        "5,1,Host,Device,1,[CUDA memcpy HtoD]\n"
    )
    live_gate = bc.transfer_gate(live_rows, scope)
    return {
        "aggregate_htod_status": gate["status"],
        "aggregate_false_pass_closed": gate["status"] != "OK",
        "timestamped_htod_status": live_gate["status"],
        "timestamped_copy_failed_gate": live_gate["status"] == "FAILED",
    }


def _live_existing_artifact_path() -> dict[str, Any]:
    """Exercise the production exporter/parser against the retained W1b report."""
    if not LIVE_W1B_REP.is_file():
        return {
            "status": "MISSING",
            "path": str(LIVE_W1B_REP),
            "passed": False,
            "reason": "retained W1b nsys report is absent",
        }
    with tempfile.TemporaryDirectory(prefix="v025-rev6-live-export-") as temporary:
        output = Path(temporary)
        exported = nex.export_all(
            LIVE_W1B_REP,
            output,
            run_id="retained-w1b-read-only",
        )
        trace_record = exported["exports"]["cuda_gpu_trace"]
        trace_path = output / nex.REPORTS["cuda_gpu_trace"]
        rows = (
            pp.parse_cuda_gpu_trace(trace_path.read_text())
            if trace_record["status"] == "OK"
            else []
        )
        statuses = {
            name: {
                "status": record["status"],
                "data_rows": record.get("data_rows"),
                "sha256": record.get("sha256"),
            }
            for name, record in exported["exports"].items()
        }
        return {
            "status": "OK" if rows else "FAILED",
            "path": str(LIVE_W1B_REP),
            "source_rep_sha256": exported["source_rep_sha256"],
            "export_overall_status": exported["status"],
            "required_unusable": exported["required_unusable"],
            "reports": statuses,
            "timestamped_rows_parsed": len(rows),
            "first_row": rows[0] if rows else None,
            "passed": (
                trace_record["status"] == "OK"
                and len(rows) > 0
                and all(
                    row.get("start_ns") is not None
                    and row.get("duration_ns") is not None
                    for row in rows
                )
                # This retained failed capture has no kernels. The complete
                # exporter must remain BLOCKED rather than laundering it.
                and exported["status"] == "BLOCKED"
                and "cuda_gpu_kern_sum" in exported["required_unusable"]
            ),
            "device_action": "NONE; nsys stats read an existing report/SQLite only",
        }


def _dry_path() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v025-rev6-proof-") as temporary:
        root = Path(temporary)
        proofs = root / "proofs"
        original_proofs = drv.PROOFS
        original_env = os.environ.get("GPUWRF_STEP1_PROOFS")
        drv.PROOFS = proofs
        os.environ["GPUWRF_STEP1_PROOFS"] = str(proofs)
        try:
            record = dry.run(root / "runs", run_id="dryrun-rev6-proof")
        finally:
            drv.PROOFS = original_proofs
            if original_env is None:
                os.environ.pop("GPUWRF_STEP1_PROOFS", None)
            else:
                os.environ["GPUWRF_STEP1_PROOFS"] = original_env

        analysis = json.loads((proofs / "step1_attribution_dryrun.json").read_text())
        run_root = Path(record["run_directory"])
        capture = json.loads((run_root / "capture_outcome.json").read_text())
        artifacts = json.loads((run_root / "capture_artifacts.json").read_text())
        return {
            "execution_status": record["execution_status"],
            "gate_status": record["gate_status"],
            "analysis_returncode": record["analysis"]["returncode"],
            "wrapper_returncode": record["lock_wrapper"]["returncode"],
            "wrapper_log_sha256": record["lock_wrapper"]["log"]["sha256"],
            "capture_log_sha256": capture["capture_log"]["sha256"],
            "capture_run_id": capture["run_id"],
            "manifest_run_id": artifacts["run_id"],
            "hlo_candidates_manifested": len(artifacts["hlo_candidates"]),
            "analysis_gate_status": analysis["gate_status"],
            "census_status": analysis["baseline_census"]["status"],
            "census_gates_not_ok": analysis["baseline_census"]["gates_not_ok"],
            "h1_measurement_kind": analysis["H1"]["measurement_kind"],
            "h1_product_compile_gate_eligible": analysis["H1"][
                "product_compile_gate_eligible"
            ],
            "production_representativeness_status": analysis["baseline_census"][
                "production_representativeness"
            ]["status"],
            "a6_role": analysis["baseline_census"][
                "a6_static_proxy_rank_diagnostic"
            ]["status"],
            "transfer_status": analysis["baseline_census"]["transfer_audit"]["status"],
            "passed": (
                record["execution_status"] == "COMPLETED"
                and record["gate_status"] == "PARTIAL"
                and record["analysis"]["returncode"] != 0
                and set(analysis["baseline_census"]["gates_not_ok"])
                == {
                    "production_representativeness",
                    "transfer_audit",
                    "profiler_perturbation",
                    "vram",
                }
            ),
        }


def build() -> dict[str, Any]:
    source_files = [
        REPO / "scripts/v025/build_rev6_repair_evidence.py",
        REPO / "scripts/v025/build_step1_mechanism_evidence.py",
        REPO / "scripts/v025/step1_driver.py",
        REPO / "scripts/v025/step1_dryrun.py",
        REPO / "scripts/v025/step1_stub.py",
        REPO / "scripts/v025/nsys_export.py",
        REPO / "scripts/v025/baseline_census.py",
        REPO / "scripts/v025/h1_discriminator.py",
        REPO / "scripts/v025/nvtx_exclusive.py",
        REPO / "scripts/v025/parse_profiler.py",
        REPO / "tests/v025/test_h1_discriminator.py",
        REPO / "tests/v025/test_nsys_export_and_census.py",
        REPO / "tests/v025/test_step1_boundary.py",
        REPO / "tests/v025/test_step1_driver.py",
    ]
    outer = drv._group_runner(["true"], timeout_s=5.0)
    h1_upper = h1.verdict(
        140.0,
        completed=True,
        t_off_basis=h1.UPPER_BOUND_BASIS,
        measurement_kind=h1.MEASUREMENT_UPPER_BOUND,
    )
    production_diff = subprocess.run(
        ["git", "diff", "--exit-code", "a1fa459c", "--", "src/gpuwrf"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    dry_path = _dry_path()
    transfer = _transfer_mutation()
    rho = _rho_mutation()
    live_existing_artifact = _live_existing_artifact_path()
    checks = {
        "outer_rc_zero_preserved": outer.returncode == 0,
        "outer_log_hashed": len(outer.log_sha256) == 64,
        "aggregate_transfer_false_pass_closed": transfer["aggregate_false_pass_closed"],
        "timestamped_transfer_detected": transfer["timestamped_copy_failed_gate"],
        "live_existing_artifact_export_and_parser": live_existing_artifact["passed"],
        "frozen_rho_bar_enforced": rho["passed"],
        "h1_upper_bound_typed": (
            h1_upper["measurement_kind"] == "upper_bound"
            and h1_upper["product_compile_gate_eligible"] is False
        ),
        "dry_real_runner_path": dry_path["passed"],
        "production_source_unchanged": production_diff.returncode == 0,
    }
    return {
        "schema": "wrf_gpu2.v025.m0.rev6_repair_cpu_evidence.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": {
            "JAX_PLATFORMS": os.environ["JAX_PLATFORMS"],
            "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
            "device_actions": "NONE",
        },
        "status": "CPU_REPAIR_GREEN_REQUIRED_GPU_EVIDENCE_MISSING"
        if all(checks.values()) else "CPU_REPAIR_FAILED",
        "checks": checks,
        "outer_runner": {
            "returncode": outer.returncode,
            "log_sha256": outer.log_sha256,
        },
        "transfer_mutations": transfer,
        "live_existing_artifact": live_existing_artifact,
        "production_rho_mutation": rho,
        "h1_upper_bound": h1_upper,
        "dry_path": dry_path,
        "profiler_matched_pair_plan": drv.profiler_matched_pair_plan(),
        "vram_measurement_plan": drv.vram_measurement_plan(),
        "gates_deliberately_missing": {
            "real_production_integration_scope": (
                "requires an instrumentation boundary in the forecast process or an "
                "approved equivalent; src/gpuwrf is frozen in M0"
            ),
            "matched_two_domain_production_reference": (
                "no successful matched production CUDA trace exists"
            ),
            "profiler_perturbation": (
                "sum of frozen caps is 1640 s, above the 1500 s window"
            ),
            "vram": (
                "forecast-process allocator sidecar and lock-owner sampler are not wired"
            ),
        },
        "production_source_diff": {
            "command": "git diff --exit-code a1fa459c -- src/gpuwrf",
            "returncode": production_diff.returncode,
            "stdout": production_diff.stdout,
            "stderr": production_diff.stderr,
        },
        "source_hashes": {
            str(path.relative_to(REPO)): _sha(path) for path in source_files
        },
        "reproduce": (
            "taskset -c 0-3 env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "NUMEXPR_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false "
            "PYTHONPATH=src python scripts/v025/build_rev6_repair_evidence.py"
        ),
    }


def main() -> int:
    output = REPO / "proofs/v025/m0/rev6_repair_cpu_evidence.json"
    obj = build()
    output.parent.mkdir(parents=True, exist_ok=True)
    drv._atomic_json(output, obj)
    print(json.dumps({"path": str(output), "status": obj["status"]}, indent=2))
    return 0 if obj["status"].startswith("CPU_REPAIR_GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
