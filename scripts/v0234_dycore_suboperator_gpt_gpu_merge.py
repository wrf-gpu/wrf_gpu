"""CPU-only terminal merge for the single v0234 dycore ladder GPU arm.

The script authenticates the retained runtime and CPU proofs, validates every
emitted savepoint against the runtime manifest, and evaluates the frozen
step-1 ladder in source order.  It stops at the first systematic compatible
span.  It imports neither JAX nor gpuwrf and performs no GPU operation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_dycore_suboperator_gpt_cpu_analysis as cpu  # noqa: E402
from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-gpu-arm"
OUTPUT = SPRINT / "gpu-cpu-ladder-analysis.json"
CPU_PROOF = (
    REPO
    / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation/proof.json"
)
CPU_ANALYSIS = (
    REPO
    / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation/"
    "cpu-ladder-analysis.json"
)
CPU_AUDIT = (
    REPO
    / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/"
    "dycore-suboperator-runner-cpu-proof.json"
)
LAUNCHER = (
    REPO
    / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi/"
    "dycore-suboperator-ladder-exact-launch-command.sh"
)
LINEAGE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
RUN_DIR = LINEAGE / "nested_stage_omega_transport_470e6111_dycore_suboperator_ladder1"
RUNTIME_PROOF = RUN_DIR / "dycore-suboperator-ladder-terminal-proof.json"
SAVEPOINT_DIR = RUN_DIR / "savepoints"
RUN_PIN = RUN_DIR / "autotune-results.pb"

TERMINAL_CPU_COMMIT = "453c9fa5d9461dece7ba50de722c1f35c7976931"
LAUNCHER_SHA256 = "70029a23b5c6cf3572e5a79147eec908b78c349a59009627f470b73064e16f23"
LOCK_WRAPPER_SHA256 = "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
RUNTIME_SCHEMA = "gpuwrf.v0234.dycore-suboperator-ladder-short-arm.v1"
RUNTIME_VERDICT = "DYCORE_SUBOPERATOR_LADDER_SHORT_ARM_COMPLETE"
SYSTEMATIC_FACTOR = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object, *, omit: str | None = None) -> str:
    if omit is not None and isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != omit}
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_self_hashed(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    expected = payload.get("proof_sha256")
    actual = canonical_digest(payload, omit="proof_sha256")
    if expected != actual:
        raise RuntimeError(
            f"self-hash mismatch {path}: expected={expected} actual={actual}"
        )
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "canonical_self_hash": actual,
        "bytes": path.stat().st_size,
    }


def load_gpu_rung(step: int, rung: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for component, field in rung["fields"].items():
        path = SAVEPOINT_DIR / f"step{step:06d}_{rung['name']}__{field}.npy"
        arrays[component] = np.load(path, allow_pickle=False)
    return arrays


def classification(metrics: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    value = float(metrics["rmse"])
    low = float(envelope["wrf_envelope_min_rmse"])
    high = float(envelope["wrf_envelope_max_rmse"])
    threshold = SYSTEMATIC_FACTOR * high
    outside = value < low or value > high
    systematic = outside and value > threshold
    return {
        "gpu_rmse": value,
        "wrf_literal_envelope_min_rmse": low,
        "wrf_literal_envelope_max_rmse": high,
        "gpu_to_wrf_member_max_ratio": None if high == 0.0 else value / high,
        "wrf_member_max_is_exact_zero": high == 0.0,
        "systematic_factor": SYSTEMATIC_FACTOR,
        "systematic_threshold": threshold,
        "outside_literal_wrf_envelope": outside,
        "systematic": systematic,
    }


def compare_rung(
    name: str,
    gpu_state: dict[str, np.ndarray],
    control_state: dict[str, np.ndarray],
    cpu_row: dict[str, Any],
    hgt: np.ndarray,
) -> dict[str, Any]:
    difference = cpu.subtract(gpu_state, control_state)
    metrics = cpu.metrics_with_bands(difference, hgt)
    return {
        "name": name,
        "representation": cpu_row["representation"],
        "source_boundary": cpu_row["source_boundary"],
        "fields": cpu_row["fields"],
        "gpu_state": cpu.state_identity(gpu_state),
        "control_state": cpu.state_identity(control_state),
        "gpu_vs_control": metrics,
        "wrf_envelope": cpu_row["envelope"],
        "classification": classification(metrics, cpu_row["envelope"]),
    }


def validate_manifest(runtime: dict[str, Any]) -> dict[str, Any]:
    manifest = runtime["savepoint_manifest"]
    rows = manifest["rows"]
    if manifest["count"] != 180 or manifest["expected_count"] != 180:
        raise RuntimeError(f"unexpected savepoint count: {manifest['count']}")
    rows_digest = canonical_digest(rows)
    if rows_digest != manifest["rows_sha256"]:
        raise RuntimeError("runtime savepoint rows hash mismatch")
    seen: set[str] = set()
    steps: dict[int, int] = {}
    for row in rows:
        path = Path(row["path"])
        if path.parent != SAVEPOINT_DIR.resolve():
            raise RuntimeError(f"savepoint escaped namespace: {path}")
        if str(path) in seen:
            raise RuntimeError(f"duplicate savepoint: {path}")
        seen.add(str(path))
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"savepoint file hash mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
            raise RuntimeError(f"savepoint metadata mismatch: {path}")
        steps[int(row["step"])] = steps.get(int(row["step"]), 0) + 1
    actual = sorted(str(path.resolve()) for path in SAVEPOINT_DIR.glob("*.npy"))
    if actual != sorted(seen):
        raise RuntimeError("runtime manifest does not exactly cover savepoint directory")
    if steps != {step: 20 for step in range(1, 10)}:
        raise RuntimeError(f"unexpected per-step savepoint count: {steps}")
    return {
        "count": len(rows),
        "rows_sha256": rows_digest,
        "all_file_hashes_valid": True,
        "exact_directory_coverage": True,
        "per_step_counts": {str(step): count for step, count in sorted(steps.items())},
    }


def main() -> None:
    cpu_proof, cpu_proof_row = read_self_hashed(CPU_PROOF)
    cpu_analysis, cpu_analysis_row = read_self_hashed(CPU_ANALYSIS)
    cpu_audit, cpu_audit_row = read_self_hashed(CPU_AUDIT)
    runtime, runtime_row = read_self_hashed(RUNTIME_PROOF)
    if runtime["schema"] != RUNTIME_SCHEMA or runtime["verdict"] != RUNTIME_VERDICT:
        raise RuntimeError("runtime proof schema/verdict mismatch")
    if not runtime["gpu_released_at_exit"]:
        raise RuntimeError("runtime did not promise GPU release at exit")
    if Path("/tmp/wrf_gpu2_gpu.lock.holder").exists():
        raise RuntimeError("canonical GPU holder remains after launcher exit")
    if sha256_file(LAUNCHER) != LAUNCHER_SHA256:
        raise RuntimeError("exact launcher bytes changed")
    live_lock = runtime["authority"]["live_lock"]
    lock_source = runtime["authority"]["lock_source"]
    if live_lock["intent"] != "production-preemptible":
        raise RuntimeError("runtime lock intent mismatch")
    if lock_source["wrapper"]["sha256"] != LOCK_WRAPPER_SHA256:
        raise RuntimeError("runtime canonical lock wrapper mismatch")
    if runtime["authority"]["cpu_affinity"] != [13, 14, 15, 29, 30, 31]:
        raise RuntimeError("runtime CPU affinity mismatch")
    if len(runtime["health_rows"]) != 9 or not all(
        row["passed"] and row["complete_carry_nonfinite_count"] == 0
        for row in runtime["health_rows"]
    ):
        raise RuntimeError("runtime health rows are not all finite/green")
    if not runtime["production_hlo_identity"]["identical"]:
        raise RuntimeError("production HLO identity failed")

    manifest = validate_manifest(runtime)
    hgt = cpu.load_hgt()
    control_ranks = reassemble.load_ranks(cpu.RUNS / "control/momsp_dumps")
    step_key = "step1"
    cpu_step = cpu_analysis["steps"][step_key]
    rung_specs = {row["name"]: row for row in cpu.RUNG_CHAIN}
    control = {
        name: cpu.load_rung(control_ranks, 1, rung_specs[name])
        for name in ("sp3_tendf", "l1_rk1_tend")
    }
    gpu = {
        name: load_gpu_rung(1, rung_specs[name])
        for name in ("sp3_tendf", "l1_rk1_tend")
    }
    rung_rows = {
        name: compare_rung(
            name, gpu[name], control[name], cpu_step["rungs"][name], hgt,
        )
        for name in ("sp3_tendf", "l1_rk1_tend")
    }

    auxiliary_spec = cpu.AUXILIARY_RUNGS[0]
    control_aux = cpu.load_rung(control_ranks, 1, auxiliary_spec)
    gpu_aux = load_gpu_rung(1, auxiliary_spec)
    auxiliary_row = compare_rung(
        auxiliary_spec["name"], gpu_aux, control_aux,
        cpu_step["auxiliary_rungs"][auxiliary_spec["name"]], hgt,
    )
    auxiliary_row["included_in_primary_chain"] = False

    span_name = "sp3_to_l1_tendency_build"
    cpu_span = cpu_step["spans"][span_name]
    gpu_update = cpu.subtract(gpu["l1_rk1_tend"], gpu["sp3_tendf"])
    control_update = cpu.subtract(control["l1_rk1_tend"], control["sp3_tendf"])
    update_difference = cpu.subtract(gpu_update, control_update)
    span_metrics = cpu.metrics_with_bands(update_difference, hgt)
    span_classification = classification(span_metrics, cpu_span["envelope"])
    if not cpu_span["dimensionally_compatible_update_subtraction"]:
        raise RuntimeError("first frozen span unexpectedly cross-representation")
    if not span_classification["systematic"]:
        raise RuntimeError("first compatible source span was not decisive")

    evaluation_trace = [
        {
            "order": 0,
            "kind": "incoming_rung",
            "name": "sp3_tendf",
            "systematic": rung_rows["sp3_tendf"]["classification"]["systematic"],
        },
        {
            "order": 1,
            "kind": "outgoing_rung",
            "name": "l1_rk1_tend",
            "systematic": rung_rows["l1_rk1_tend"]["classification"]["systematic"],
        },
        {
            "order": 2,
            "kind": "compatible_source_bound_span",
            "name": span_name,
            "systematic": span_classification["systematic"],
            "terminal_stop": True,
        },
    ]

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.dycore-suboperator-gpt-gpu-arm.analysis.v1",
        "verdict": "DECISIVE_SP3_TO_L1_TENDENCY_BUILD_SYSTEMATIC",
        "step": 1,
        "rungs_evaluated_in_order": rung_rows,
        "auxiliary_rungs": {auxiliary_spec["name"]: auxiliary_row},
        "spans_evaluated_in_order": {
            span_name: {
                "before_rung": "sp3_tendf",
                "after_rung": "l1_rk1_tend",
                "before_representation": cpu_span["before_representation"],
                "after_representation": cpu_span["after_representation"],
                "dimensionally_compatible_update_subtraction": True,
                "source_boundary": (
                    "module_first_rk_step_part2.F after update_phy_ten to solve_em.F "
                    "after relax_bdy_dry/rk_addtend_dry/spec_bdy_dry"
                ),
                "gpu_update": cpu.state_identity(gpu_update),
                "control_update": cpu.state_identity(control_update),
                "gpu_vs_control_update_difference": span_metrics,
                "wrf_envelope": cpu_span["envelope"],
                "classification": span_classification,
            }
        },
        "evaluation_trace": evaluation_trace,
        "terminal_decision": {
            "earliest_decisive_named_source_bound_span": span_name,
            "source_localization": (
                "the first compatible merged-tendency build span is systematic; "
                "the SP3 input is already systematic, and this span adds a much "
                "larger systematic response before the L1 boundary"
            ),
            "not_proven": (
                "this bracket does not distinguish relax_bdy_dry, rk_addtend_dry, "
                "and spec_bdy_dry internally and does not authorize a correction"
            ),
            "analysis_stopped_after_first_decisive_span": True,
            "later_rungs_or_spans_analyzed": False,
            "gpu_followup_authorized": False,
        },
        "metric_definition": {
            "rung": "GPU rung state - control WRF rung state",
            "span": "(GPU after - GPU before) - (control WRF after - control WRF before)",
            "combined_rmse": "cell-count-weighted across U- and V-staggered components",
            "bands": cpu_analysis["metric_definition"]["bands"],
            "systematic": (
                "outside literal six-member WRF RMSE envelope and GPU RMSE > "
                "2.0 * WRF member maximum RMSE"
            ),
        },
        "authority": {
            "terminal_cpu_commit": TERMINAL_CPU_COMMIT,
            "terminal_cpu_proof": cpu_proof_row,
            "cpu_ladder_analysis": cpu_analysis_row,
            "cpu_runner_audit": cpu_audit_row,
            "runtime_proof": runtime_row,
            "runtime_runner_head": runtime["authority"]["candidate"]["runner_head"],
            "runtime_model_tree": runtime["model_tree"],
            "production_hlo_identity": runtime["production_hlo_identity"],
            "exact_launcher": {
                "path": str(LAUNCHER.resolve()),
                "sha256": sha256_file(LAUNCHER),
            },
            "canonical_lock": {
                "live_lock": live_lock,
                "source": lock_source,
                "holder_absent_after_exit": True,
            },
            "runtime_health": {
                "row_count": len(runtime["health_rows"]),
                "all_passed": True,
                "all_complete_carry_finite": True,
            },
            "savepoint_manifest": manifest,
            "autotune_run_pin": {
                "path": str(RUN_PIN.resolve()),
                "sha256": sha256_file(RUN_PIN),
                "staged_pin_absent_after_move": not (
                    LINEAGE
                    / ".v0234-dycore-suboperator-autotune-v1/"
                    "dycore-suboperator-autotune-results.pb"
                ).exists(),
            },
            "cpu_proof_verdict": cpu_proof["verdict"],
            "cpu_analysis_verdict": cpu_analysis["verdict"],
            "cpu_audit_verdict": cpu_audit["verdict"],
            "gpu_queries_during_cpu_merge": 0,
            "gpu_commands_during_cpu_merge": 0,
            "jax_imported_during_cpu_merge": "jax" in sys.modules,
            "gpuwrf_imported_during_cpu_merge": any(
                name == "gpuwrf" or name.startswith("gpuwrf.") for name in sys.modules
            ),
        },
    }
    if payload["authority"]["jax_imported_during_cpu_merge"]:
        raise RuntimeError("JAX imported during CPU-only merge")
    if payload["authority"]["gpuwrf_imported_during_cpu_merge"]:
        raise RuntimeError("gpuwrf imported during CPU-only merge")
    cpu.write_self_hashed(OUTPUT, payload)
    written = json.loads(OUTPUT.read_text())
    print(json.dumps({
        "wrote": str(OUTPUT),
        "verdict": written["verdict"],
        "proof_sha256": written["proof_sha256"],
        "first_span_rmse": span_metrics["rmse"],
        "first_span_member_max_rmse": cpu_span["envelope"]["wrf_envelope_max_rmse"],
        "ratio": span_classification["gpu_to_wrf_member_max_ratio"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
