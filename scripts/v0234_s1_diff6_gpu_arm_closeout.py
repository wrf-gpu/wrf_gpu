"""Authenticate and seal the one-run v0234 S1 diff6 GPU-arm result.

This closeout is deliberately standard-library-only.  It must not import JAX or
gpuwrf, acquire a GPU lock, query a device, or execute the model.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-diff6-gpt-gpu-arm"
OUTPUT = SPRINT / "proof.json"
ZERO_SELF = "0" * 64

RUN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
)
AUTH_EXTERNAL = Path("/tmp/v0234-s1-diff6-gpu-authorization.json")
AUTH_ARCHIVE = SPRINT / "chief-0-3-authorization.json"
LOCK_HOLDER = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2/state/holder.json"
)
AUTOTUNE_STAGE = RUN.parent / ".v0234-s1-diff6-validation-autotune-v1"

CANDIDATE = "2e5fa008563cfe56c3c766b21352effc7179c4bf"
FIX = "e65ce784bec85ea4f000e94ba572420c595ecb68"
RUNNER_HEAD = "35af0b0a3b1a90ef9a5938f067f2da6f2c67c7bb"
NAMESPACE = "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
AUTH_FILE_SHA = "c7999d5398ab5a84458e795c59342d6999479fa0b24bb22743822165a1d34ca7"

EXPECTED_SELF_HASHES = {
    "authorization": "ff44e73e14ede339c43a71fd47d166e36d1f064de70327b88e33012f2ad1905a",
    "step1_gate": "e61a083b77ae75b5580178b51cc5654d086b5fde387f992c9f2f4f4f7683a5b8",
    "ordinary_hlo": "612f2552416e19318d234f86571916a0ab843e4de9c726b60aca0c72d52011b5",
    "capture_hlo": "43bd2071d29d73b872e5565329a2ab5ca79becbc8e87db3e86f11a8cb40fa05f",
    "terminal": "6e8ed620932459a6d663feafb8a64cdd3bf5cbd3aaa65a6e878dcdb1e70c677e",
    "release_receipt": "35963b7f1a5c485daac70421c5c362e19540e97ac6a253c2c021f484049a630c",
    "cpu_audit": "671bdf8b690c3e5155b371524584e17a4c826b76f09e1c2acb469062744c9ca5",
}
PRODUCTION_HASHES = {
    "src/gpuwrf/dynamics/explicit_diffusion.py": (
        "58e677d4e7748057f4b3546380296f01e99a58fb161b76971343a46aa9792b8c"
    ),
    "src/gpuwrf/runtime/operational_mode.py": (
        "9480c992b46df6a76f0f492baf8098d7945507d1c9e6c44c002bfbd0a4eceaff"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(REPO), *args), check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def plain_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
    }


def authenticated_json(
    path: Path, expected_self: str, *, field: str = "proof_sha256",
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    unsigned = {key: value for key, value in payload.items() if key != field}
    observed = canonical(unsigned)
    if payload.get(field) != expected_self or observed != expected_self:
        raise RuntimeError(f"canonical self-hash mismatch: {path}")
    return payload, plain_file(path) | {
        "self_hash_field": field,
        "canonical_self_hash": observed,
        "authenticated": True,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(RUN.is_dir() and not RUN.is_symlink(), "run namespace missing or symlink")
    require(not LOCK_HOLDER.exists(), "canonical GPU lock is still held")
    require(
        AUTOTUNE_STAGE.is_dir() and not any(AUTOTUNE_STAGE.iterdir()),
        "autotune staging did not finalize to an empty directory",
    )

    auth, auth_row = authenticated_json(
        AUTH_ARCHIVE, EXPECTED_SELF_HASHES["authorization"],
    )
    require(auth_row["file_sha256"] == AUTH_FILE_SHA, "authorization archive bytes drifted")
    require(
        AUTH_EXTERNAL.is_file() and sha256_file(AUTH_EXTERNAL) == AUTH_FILE_SHA,
        "executed external authorization bytes are unavailable or changed",
    )
    require(
        auth == json.loads(AUTH_EXTERNAL.read_text()),
        "authorization archive differs from executed external object",
    )
    require(
        auth["decision"] == "RELEASE_GPU"
        and auth["issuer"] == "chief-0:3"
        and auth["intent"] == "production-preemptible"
        and auth["one_arm"] is True
        and auth["candidate_head"] == RUNNER_HEAD
        and auth["namespace"] == NAMESPACE,
        "authorization authority fields are not exact",
    )

    gate, gate_row = authenticated_json(
        RUN / "step1-s1-residual-gate.json", EXPECTED_SELF_HASHES["step1_gate"],
    )
    ordinary, ordinary_row = authenticated_json(
        RUN / "ordinary-one-step-fixed-reference-lowered-hlo.json",
        EXPECTED_SELF_HASHES["ordinary_hlo"],
    )
    capture, capture_row = authenticated_json(
        RUN / "dycore-suboperator-capture-lowered-hlo.json",
        EXPECTED_SELF_HASHES["capture_hlo"],
    )
    terminal, terminal_row = authenticated_json(
        RUN / "s1-diff6-validation-terminal-proof.json",
        EXPECTED_SELF_HASHES["terminal"],
    )
    receipt, receipt_row = authenticated_json(
        RUN / "gpu-lock-release-receipt.json",
        EXPECTED_SELF_HASHES["release_receipt"],
    )
    cpu_audit, cpu_audit_row = authenticated_json(
        SPRINT / "s1-diff6-runner-cpu-proof.json",
        EXPECTED_SELF_HASHES["cpu_audit"],
    )

    require(terminal["verdict"] == "S1_DIFF6_ARM_GREEN", "arm is not green")
    require(terminal["namespace"] == NAMESPACE, "terminal namespace drifted")
    require(
        receipt["terminal_proofs"] == [{
            "path": str((RUN / "s1-diff6-validation-terminal-proof.json").resolve()),
            "file_sha256": terminal_row["file_sha256"],
        }],
        "release receipt does not bind the terminal proof",
    )
    require(
        receipt["authorization_file_sha256"] == AUTH_FILE_SHA
        and receipt["wrapper_returned"] is True
        and receipt["file_descriptor_lease_released_before_receipt"] is True
        and receipt["model_returncode"] == 0
        and receipt["gpu_queries_run_by_receipt"] == 0,
        "release receipt is not a clean immediate release",
    )
    require(
        terminal["gpu_released_at_exit"] is True
        and terminal["release_gate_unchanged"] is True
        and terminal["tolerance_changed"] is False,
        "terminal release/tolerance invariants failed",
    )

    limits = {"nonspec": 4.0, "relax_rows_1_4": 9.5, "interior_ge_5": 1.4}
    require(
        gate["passed"] is True
        and gate["step"] == 1
        and gate["dispatched_steps_at_gate"] == 1
        and gate["first_red"] is None
        and gate["step2_dispatch_admitted"] is True
        and gate["step2_withheld_on_red"] is True
        and gate["fresh_savepoints_authenticated"] == 20,
        "step-1 stop-first-red protocol failed",
    )
    for name, limit in limits.items():
        row = gate["rows"][name]
        require(
            row["finite"] is True
            and row["passed"] is True
            and row["threshold_le"] == limit
            and row["value"] <= limit,
            f"step-1 threshold failed: {name}",
        )

    require(
        terminal["window"]["model_dispatches"] == 9
        and terminal["window"]["sampled_steps_count"] == 9
        and terminal["window"]["sampled_d03_first_last"] == [1, 9]
        and terminal["window"]["own_steps"] == {"d01": 1, "d02": 3, "d03": 9}
        and terminal["capture_compile"]["compile_calls"] == 1
        and terminal["capture_compile"]["dispatch_calls"] == 9,
        "one-arm dispatch schedule changed",
    )
    health = terminal["health_rows"]
    require(
        [row["step"] for row in health] == list(range(1, 10))
        and all(
            row["passed"] is True
            and not row["violations"]
            and row["complete_carry_nonfinite_count"] == 0
            and row["ni_nonfinite_count"] == 0
            for row in health
        ),
        "per-step finite/health evidence failed",
    )

    manifest = terminal["savepoint_manifest"]
    rows = manifest["rows"]
    require(
        len(rows) == manifest["count"] == manifest["expected_count"] == 180,
        "savepoint count is not 180",
    )
    require(canonical(rows) == manifest["rows_sha256"], "savepoint row manifest drifted")
    actual_savepoints = sorted((RUN / "savepoints").glob("*.npy"))
    require(len(actual_savepoints) == 180, "filesystem savepoint count is not 180")
    expected_names = set()
    for row in rows:
        path = Path(row["path"])
        require(path.parent == RUN / "savepoints", "savepoint escaped namespace")
        require(path.is_file() and sha256_file(path) == row["file_sha256"], f"savepoint drift: {path}")
        require(row["step"] in range(1, 10), "savepoint step outside arm")
        require(re.fullmatch(r"step\d{6}_.+__[a-z_]+\.npy", path.name) is not None, "savepoint name malformed")
        expected_names.add(path.name)
    require(expected_names == {path.name for path in actual_savepoints}, "savepoint inventory mismatch")

    require(
        ordinary["namespace"] == NAMESPACE
        and ordinary["reference_kind"] == "fresh-fixed-candidate"
        and ordinary["production_fix_commit"] == FIX
        and ordinary["reviewed_candidate_head"] == CANDIDATE
        and ordinary["runner_head"] == RUNNER_HEAD
        and ordinary["compile_calls_before_artifact"] == 0
        and ordinary["dispatch_calls_before_artifact"] == 0
        and ordinary["policy_evaluated_before_artifact"] is False
        and ordinary["obsolete_prefix_hlo_used"] is False
        and ordinary["carry_interface"]["leaf_count"] == 106,
        "fresh ordinary HLO reference protocol failed",
    )
    for name, hlo in (("ordinary", ordinary), ("capture", capture)):
        extraction = hlo["extraction"]
        require(
            extraction["extraction_complete"] is True
            and extraction["targets"] == ["cusparse_gtsv2_ffi"]
            and extraction["custom_call_syntax_occurrence_count"]
            == extraction["parsed_custom_call_count"]
            and extraction["parsed_offsets"] == extraction["syntax_offsets"],
            f"{name} HLO extraction/policy input failed",
        )
    policy = terminal["production_hlo_policy"]
    require(
        policy["passed"] is True
        and policy["custom_call_targets"] == ["cusparse_gtsv2_ffi"]
        and not policy["forbidden_custom_call_targets"]
        and not policy["unknown_custom_call_targets"]
        and not policy["forbidden_text_tokens"]
        and policy["unparsed_custom_call_count"] == 0,
        "production HLO policy failed",
    )

    production_sources: dict[str, str] = {}
    for relative, expected in PRODUCTION_HASHES.items():
        observed = sha256_file(REPO / relative)
        require(observed == expected, f"production source changed: {relative}")
        production_sources[relative] = observed
    require(ordinary["production_source_sha256"] == production_sources, "HLO source pins drifted")
    require(
        not git("diff", "--name-only", CANDIDATE, "HEAD", "--", "src/gpuwrf"),
        "review/arm preparation changed production sources",
    )
    require(
        terminal["authority"]["candidate"]["approved_runner_head"] == RUNNER_HEAD
        and terminal["authority"]["candidate"]["launcher_sha256"]
        == "7dbba83a00c61a6fb4ca9f5c0fa410b3bbbddb82d3c243706091ffa7e7c9050c"
        and terminal["authority"]["candidate"]["profile_source_sha256"]
        == "a1bbd22610f0993c5990880c5e7f6f0ac9b0b4058a46149d9606639e195fb5c8"
        and terminal["authority"]["candidate"]["runner_source_sha256"]
        == "b34498345da5513fc3508ee8febb3e1cdd6756f5167f34b8d7340ffcb5d6fbdc",
        "executed runner/tool pins drifted",
    )
    require(
        cpu_audit["verdict"] == "READY_FOR_AUTHORIZED_SINGLE_S1_DIFF6_GPU_ARM"
        and cpu_audit["static_audit"]["passed"] is True
        and cpu_audit["exact_launch_audit"]["passed"] is True
        and cpu_audit["gpu_commands_run"] == 0
        and cpu_audit["gpu_queries_run"] == 0
        and cpu_audit["jax_imported"] is False,
        "prelaunch CPU audit authority failed",
    )

    review_path = REPO / ".agent/sprints/2026-07-18-v0234-s1-diff6-gpt-review-arm-prep/proof.json"
    review = json.loads(review_path.read_text())
    review_body = {
        key: value for key, value in review.items()
        if key not in {"canonical_payload_sha256", "normalized_whole_file_self_sha256"}
    }
    require(
        review["terminal_verdict"] == "GPT_REVIEW_ACCEPT_READY_FOR_AUTH"
        and canonical(review_body) == review["canonical_payload_sha256"]
        == "0baacc303b5a0bd9e5470d312dad5d5cdd71fc3e1e731c685741dc7b3a2b24ba",
        "independent review proof authority failed",
    )
    review_normalized = dict(review)
    review_normalized["normalized_whole_file_self_sha256"] = ZERO_SELF
    require(
        hashlib.sha256((json.dumps(
            review_normalized, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode()).hexdigest()
        == review["normalized_whole_file_self_sha256"]
        == "61f5279193f31fed0ffa0b0b2975bb91cc0c74401bcea482b0ebf03ec261eb8f",
        "independent review normalized self-hash failed",
    )

    autotune = plain_file(RUN / "autotune-results.pb")
    require(
        autotune["file_sha256"] == "65c564c2355ae3db1db49878fa6575e5622a532f98a527c594b41f027c797c3a",
        "fresh autotune dump drifted",
    )

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.s1-diff6-gpt-gpu-arm-closeout.v1",
        "terminal_verdict": "S1_DIFF6_ARM_GREEN",
        "review_verdict": "GPT_REVIEW_ACCEPT_READY_FOR_AUTH",
        "owner": "GPT-5.6-sol-xhigh",
        "closeout_head": git("rev-parse", "HEAD"),
        "candidate": {
            "reviewed_terminal": CANDIDATE,
            "production_fix": FIX,
            "executed_runner_head": RUNNER_HEAD,
            "production_source_sha256": production_sources,
            "reviewer_production_delta_files": [],
        },
        "contract": plain_file(SPRINT / "CONTRACT.md") | {
            "commit": "116db7fb",
            "state_transition": "external chief authorization received, consumed once, arm terminal",
        },
        "launcher": plain_file(SPRINT / "s1-diff6-exact-launch-command.sh") | {
            "invocations": 1,
            "retries": 0,
        },
        "authorization": auth_row | {
            "executed_external_path": str(AUTH_EXTERNAL),
            "executed_external_file_sha256": sha256_file(AUTH_EXTERNAL),
            "issuer": auth["issuer"],
            "intent": auth["intent"],
            "one_arm": auth["one_arm"],
            "consumed": True,
        },
        "prelaunch_authority": {
            "independent_review": plain_file(review_path) | {
                "canonical_payload_sha256": review["canonical_payload_sha256"],
                "normalized_whole_file_self_sha256": review[
                    "normalized_whole_file_self_sha256"
                ],
            },
            "cpu_arm_audit": cpu_audit_row,
        },
        "external_artifacts": {
            "run_directory": str(RUN),
            "step1_gate": gate_row,
            "fresh_ordinary_hlo": ordinary_row,
            "capture_hlo": capture_row,
            "terminal_proof": terminal_row,
            "release_receipt": receipt_row,
            "fresh_autotune_dump": autotune,
        },
        "step1_scientific_gate": {
            "passed": True,
            "dispatched_steps_at_gate": gate["dispatched_steps_at_gate"],
            "fresh_savepoints_authenticated": gate["fresh_savepoints_authenticated"],
            "thresholds_le": limits,
            "observed_rmse": {
                name: gate["rows"][name]["value"] for name in limits
            },
            "first_red": None,
            "step2_admitted_only_after_green": True,
        },
        "fresh_hlo_reference": {
            "retained_before_policy_compile_dispatch": True,
            "reference_kind": ordinary["reference_kind"],
            "carry_leaf_count": ordinary["carry_interface"]["leaf_count"],
            "stablehlo_bytes": ordinary["extraction"]["stablehlo_bytes"],
            "stablehlo_sha256": ordinary["extraction"]["stablehlo_sha256"],
            "custom_call_occurrences": ordinary["extraction"]["parsed_custom_call_count"],
            "custom_call_targets": ordinary["extraction"]["targets"],
            "policy_passed": policy["passed"],
            "forbidden_or_unknown_targets": [],
            "forbidden_tokens": [],
            "obsolete_prefix_used": False,
        },
        "runtime_result": {
            "started_utc": terminal["started_utc"],
            "finished_utc": terminal["finished_utc"],
            "namespace": NAMESPACE,
            "model_dispatches": terminal["window"]["model_dispatches"],
            "own_steps": terminal["window"]["own_steps"],
            "health_steps": len(health),
            "all_health_rows_passed": True,
            "savepoint_count": len(rows),
            "savepoint_rows_sha256": manifest["rows_sha256"],
            "all_savepoints_rehashed": True,
            "frame_pair_counts": terminal["frame_pair_counts"],
        },
        "lock_and_release": {
            "lock_commit": receipt["lock_commit"],
            "label": receipt["lock_label"],
            "intent": auth["intent"],
            "timeout": 0,
            "wrapper_sha256": receipt["lock_wrapper_sha256"],
            "wrapper_returncode": receipt["model_returncode"],
            "file_descriptor_lease_released": True,
            "holder_absent_after_exit": True,
            "receipt_canonical_self_hash": receipt["proof_sha256"],
            "gpu_queries_by_launcher_or_receipt": 0,
        },
        "scope_disposition": {
            "scientific_class": "PARTIAL",
            "immediate_diff6_prediction_validated": True,
            "s1_closed": False,
            "additional_gpu_arm_authorized": False,
            "v10_authorized": False,
            "ni_authorized": False,
            "full_18h_authorized": False,
            "performance_authorized": False,
            "production_release_authorized": False,
        },
        "tests": [
            {"scope": "prelaunch focused/adversarial CPU audit", "result": "43 passed"},
            {"scope": "prelaunch broad boundary/source/deformation CPU regression", "result": "85 passed"},
            {"scope": "prelaunch preceding ladder/stage/conditioning CPU regression", "result": "34 passed"},
            {"scope": "prelaunch isolated Kimi discriminator", "result": "11 passed"},
            {"scope": "prelaunch theta/scalar sixth-order CPU regression", "result": "22 passed"},
            {"scope": "post-run canonical closeout authentication", "result": "PASS"},
        ],
        "unresolved_risks": [
            "PARTIAL ring-1 relaxation and secondary interior S1 floors remain unexplained.",
            "Nine-step green validates only the immediate correction prediction, not S1 closure.",
            "No second arm or V10/Ni/18h/performance/release scope follows from this result.",
        ],
        "gpu_attestation": {
            "authorized_launcher_invocations": 1,
            "gpu_model_processes": 1,
            "gpu_queries": 0,
            "profilers": 0,
            "retries": 0,
            "second_namespaces": 0,
            "lock_released": True,
            "post_release_closeout_cpu_only": True,
        },
        "canonical_payload_sha256": None,
        "normalized_whole_file_self_sha256": ZERO_SELF,
    }
    body = {
        key: value for key, value in payload.items()
        if key not in {"canonical_payload_sha256", "normalized_whole_file_self_sha256"}
    }
    payload["canonical_payload_sha256"] = canonical(body)
    normalized = (json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode()
    payload["normalized_whole_file_self_sha256"] = hashlib.sha256(normalized).hexdigest()
    OUTPUT.write_text(json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n")
    print(json.dumps({
        "verdict": payload["terminal_verdict"],
        "proof": str(OUTPUT),
        "canonical_payload_sha256": payload["canonical_payload_sha256"],
        "normalized_whole_file_self_sha256": payload[
            "normalized_whole_file_self_sha256"
        ],
        "file_sha256": sha256_file(OUTPUT),
        "jax_imported": "jax" in sys.modules,
        "gpuwrf_imported": any(name.startswith("gpuwrf") for name in sys.modules),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
