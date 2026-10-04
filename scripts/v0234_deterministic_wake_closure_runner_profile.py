"""Pre-import bindings for the v0234 deterministic full-tree replay pair."""

from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from scripts import v0234_deterministic_wake_admission as admission


KIMI_COMMIT = "ceec296946b40185d21a3c1958830f1530711d11"
MODEL_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
REFERENCE_NAMESPACE = (
    "nested_stage_omega_transport_470e6111_deterministic_wake_reference1"
)
PINNED_NAMESPACE = (
    "nested_stage_omega_transport_470e6111_deterministic_wake_pinned1"
)
MODES = {
    "reference": {
        "namespace": REFERENCE_NAMESPACE,
        "lock_label": "v0234-deterministic-fulltree-reference",
        "audit": "reference-runner-cpu-proof-v2.json",
        "dump_pin": "reference-autotune-results.pb",
    },
    "pinned": {
        "namespace": PINNED_NAMESPACE,
        "lock_label": "v0234-deterministic-fulltree-pinned",
        "audit": "pinned-runner-cpu-proof.json",
        "dump_pin": "pinned-autotune-results.pb",
    },
}


def _authenticate_pin_manifest(
    runner: Any, *, required: bool,
) -> dict[str, Any]:
    path_raw = os.environ.get("GPUWRF_DETERMINISTIC_AUTOTUNE_PIN_MANIFEST", "")
    file_hash = os.environ.get(
        "GPUWRF_DETERMINISTIC_AUTOTUNE_PIN_MANIFEST_SHA256", ""
    )
    if not required:
        if path_raw or file_hash:
            raise runner.RunnerGateError(
                "AUTOTUNE_PIN_MANIFEST_UNEXPECTED", "reference arm must create it later",
            )
        return {"required": False, "present": False}
    if not path_raw or len(file_hash) != 64:
        raise runner.RunnerGateError(
            "AUTOTUNE_PIN_MANIFEST_ENV", "pinned arm requires path and SHA-256",
        )
    path = Path(path_raw).resolve()
    expected_path = (
        runner.REPO_ROOT
        / ".agent/sprints/2026-07-18-v0234-deterministic-wake-closure-gpt/"
        "reference-pin-manifest.json"
    ).resolve()
    if path != expected_path:
        raise runner.RunnerGateError(
            "AUTOTUNE_PIN_MANIFEST_PATH", f"expected={expected_path} actual={path}",
        )
    payload, row = runner.read_authenticated_json(
        path, file_hash, "AUTOTUNE_PIN_MANIFEST_FILE",
    )
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = runner.canonical_digest(unsigned)
    pin = payload.get("autotune_pin") or {}
    reference = payload.get("reference_run") or {}
    pin_path = Path(str(pin.get("path", ""))).resolve()
    expected_pin = (runner.LINEAGE_WORK_DIR / REFERENCE_NAMESPACE / "autotune-results.pb").resolve()
    if (
        payload.get("schema")
        != "gpuwrf.v0234.deterministic-reference-pin-manifest.v1"
        or payload.get("verdict") != "REFERENCE_AUTOTUNE_PIN_AUTHENTICATED"
        or embedded != observed
        or payload.get("model_tree") != MODEL_TREE
        or reference.get("run_dir")
        != str((runner.LINEAGE_WORK_DIR / REFERENCE_NAMESPACE).resolve())
        or reference.get("terminal_verdict")
        != "FULL_18H_LATE_NI_COMPLETE_KNOWN_V10_RED_REMAINS"
        or reference.get("terminal_output_counts")
        != {"d01": 19, "d02": 19, "d03": 55}
        or pin_path != expected_pin
        or not pin_path.is_file()
        or pin_path.is_symlink()
        or pin.get("bytes") != pin_path.stat().st_size
        or pin.get("file_sha256") != runner.sha256_file(pin_path)
        or pin.get("file_sha256")
        != os.environ.get("GPUWRF_DETERMINISTIC_AUTOTUNE_PIN_SHA256")
        or payload.get("complete_reference_process") is not True
        or payload.get("release_gate_green") is not False
    ):
        raise runner.RunnerGateError(
            "AUTOTUNE_PIN_MANIFEST_SEMANTICS",
            repr({"embedded": embedded, "observed": observed, "pin": pin}),
        )
    return {
        **row,
        "required": True,
        "canonical_payload_sha256": observed,
        "pin": pin,
        "reference_run": reference,
    }


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_DETERMINISTIC_WAKE_CLOSURE_PROFILE_APPLIED", False):
        return

    from scripts.v0234_stage_omega_transport_runner_profile import (
        apply_profile as apply_stage_omega_profile,
    )

    apply_stage_omega_profile(runner)
    prior_final_authority = runner.assert_final_candidate_proof_authority
    prior_validate_environment = runner.validate_preimport_environment
    prior_static_source_audit = runner.static_source_audit
    baseline_static_audit = prior_static_source_audit()
    if baseline_static_audit.get("passed") is not True:
        raise runner.RunnerGateError(
            "BASELINE_STATIC_AUDIT", repr(baseline_static_audit),
        )
    mode = os.environ.get("GPUWRF_DETERMINISTIC_AUTOTUNE_MODE", "")
    if mode not in MODES:
        raise runner.RunnerGateError(
            "DETERMINISTIC_AUTOTUNE_MODE", f"expected one of {sorted(MODES)} got={mode!r}",
        )
    binding = MODES[mode]
    sprint = runner.REPO_ROOT / (
        ".agent/sprints/2026-07-18-v0234-deterministic-wake-closure-gpt"
    )
    launcher = sprint / "deterministic-fulltree-exact-launch-command.sh"
    autotune_stage = (
        runner.LINEAGE_WORK_DIR / ".v0234-deterministic-wake-autotune-v1"
    ).resolve()

    runner.SCHEMA = "gpuwrf.v0234.deterministic-wake-fulltree.v1"
    runner.CPU_PROOF_SCHEMA = "gpuwrf.v0234.deterministic-wake-runner-cpu-audit.v1"
    runner.AUDIT_ADMISSION = "READY_FOR_DETERMINISTIC_WAKE_FULLTREE_GPU_REPLAY"
    runner.FULL_REPLAY_NAMESPACE = binding["namespace"]
    runner.LOCK_LABEL = binding["lock_label"]
    runner.LAUNCH_COMMAND = launcher
    runner.RUNNER_CPU_AUDIT = sprint / binding["audit"]
    runner.REQUIRE_KNOWN_1500_V10_RECORD = True
    runner.REQUIRE_TOOLING_CRITIC_ACCEPT = False
    runner.TOOLING_CRITIC_AUTHORITY_HOOK = None
    runner.ALLOW_DIAGNOSTIC_TERMINAL_WAKE_RED = True
    runner.DIAGNOSTIC_TERMINAL_ALLOWED_FIELDS = frozenset({"V", "V10"})
    runner.CPU_FOCUSED_TEST_ARGS = (
        "tests/test_v0234_deterministic_wake_closure.py",
    )
    runner.INFRASTRUCTURE_GPUWRF_ENV = set(runner.INFRASTRUCTURE_GPUWRF_ENV) | {
        "GPUWRF_DETERMINISTIC_WAKE_CLOSURE",
        "GPUWRF_DETERMINISTIC_AUTOTUNE_MODE",
        "GPUWRF_DETERMINISTIC_AUTOTUNE_PIN",
        "GPUWRF_DETERMINISTIC_AUTOTUNE_PIN_SHA256",
        "GPUWRF_DETERMINISTIC_AUTOTUNE_PIN_MANIFEST",
        "GPUWRF_DETERMINISTIC_AUTOTUNE_PIN_MANIFEST_SHA256",
    }
    runner.FORBIDDEN_NON_GPUWRF_ENV = tuple(
        name for name in runner.FORBIDDEN_NON_GPUWRF_ENV if name != "XLA_FLAGS"
    )
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        **runner.CANDIDATE_CLEAN_AUTHORITY,
        "deterministic_wake_contract_file_sha256": admission.CONTRACT_SHA256,
        "kimi_terminal_commit": KIMI_COMMIT,
        "kimi_proof_file_sha256": admission.KIMI_PROOF_FILE_SHA256,
        "kimi_proof_canonical_sha256": admission.KIMI_PROOF_CANONICAL_SHA256,
        "kimi_retained_file_sha256": admission.KIMI_RETAINED_FILE_SHA256,
        "kimi_retained_canonical_sha256": admission.KIMI_RETAINED_CANONICAL_SHA256,
        "frozen_model_tree": MODEL_TREE,
        "admission_isolation_only": True,
        "release_gate_unchanged": True,
        "autotune_modes": sorted(MODES),
    }

    def authenticate_diagnostic_observation() -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return admission.authenticate_authority()
        except Exception as exc:
            raise runner.RunnerGateError(
                "DETERMINISTIC_WAKE_AUTHORITY", f"{type(exc).__name__}: {exc}",
            ) from exc

    def classify_diagnostic_observation(
        decisive: Mapping[str, Any],
        *,
        candidate_sha256: str,
        authority: Mapping[str, Any],
        candidate_path: Path | None = None,
        cpu_path: Path | None = None,
        runtime: Any | None = None,
    ) -> dict[str, Any]:
        return admission.classify_known_wake_observation(
            decisive,
            candidate_sha256=candidate_sha256,
            authority=authority,
            candidate_path=candidate_path,
            cpu_path=cpu_path,
            runtime=runtime,
        )

    def assert_deterministic_final_authority() -> dict[str, Any]:
        prior = prior_final_authority()
        diagnostic, diagnostic_row = authenticate_diagnostic_observation()
        pin_manifest = _authenticate_pin_manifest(
            runner, required=(mode == "pinned"),
        )
        return {
            **prior,
            "deterministic_wake_contract": {
                "path": str(admission.CONTRACT.resolve()),
                "file_sha256": admission.CONTRACT_SHA256,
            },
            "kimi_terminal_proof": diagnostic_row,
            "diagnostic_admission_authority_sha256": diagnostic["authority_sha256"],
            "diagnostic_admission_isolation_only": True,
            "deterministic_autotune_mode": mode,
            "autotune_pin_manifest": pin_manifest,
            "model_tree": MODEL_TREE,
        }

    def assert_deterministic_source_authority(
        environment: Mapping[str, str], *, require_clean: bool,
    ) -> dict[str, Any]:
        head = runner._git(runner.REPO_ROOT, "rev-parse", "HEAD")
        approved = environment.get("GPUWRF_NESTED_BUNDLE_RUNNER_SHA")
        dirty = runner._git(runner.REPO_ROOT, "status", "--porcelain")
        if approved and approved != head:
            raise runner.RunnerGateError(
                "RUNNER_HEAD", f"approved={approved} head={head}",
            )
        if require_clean and dirty:
            raise runner.RunnerGateError("WORKTREE_DIRTY", dirty)
        if subprocess.run(
            ("git", "-C", str(runner.REPO_ROOT), "merge-base", "--is-ancestor", KIMI_COMMIT, head),
            check=False,
        ).returncode != 0:
            raise runner.RunnerGateError("KIMI_ANCESTRY", f"{KIMI_COMMIT} !<= {head}")
        model_tree = runner._git(runner.REPO_ROOT, "rev-parse", "HEAD:src/gpuwrf")
        model_delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", KIMI_COMMIT, "HEAD", "--", "src/gpuwrf",
        )
        if model_tree != MODEL_TREE or model_delta:
            raise runner.RunnerGateError(
                "DETERMINISTIC_MODEL_CHANGED", repr({"tree": model_tree, "delta": model_delta}),
            )
        delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", KIMI_COMMIT, "HEAD",
        ).splitlines()
        allowed_exact = {
            "scripts/v0234_nested_frozen_wrf_boundary_window.py",
            "scripts/v0234_deterministic_wake_admission.py",
            "scripts/v0234_deterministic_wake_closure_runner_profile.py",
            "scripts/v0234_deterministic_wake_pin_manifest.py",
            "scripts/v0234_deterministic_fulltree_equality.py",
            "scripts/v0234_deterministic_wake_terminal_proof.py",
            "scripts/v0234_deterministic_wake_rca.py",
            "tests/test_v0234_deterministic_wake_closure.py",
        }
        allowed_prefix = (
            ".agent/sprints/2026-07-18-v0234-deterministic-wake-closure-gpt/"
        )
        unexpected = sorted(
            path for path in delta
            if path not in allowed_exact and not path.startswith(allowed_prefix)
        )
        if unexpected:
            raise runner.RunnerGateError("RUNNER_ONLY_DELTA", repr(unexpected))
        source_rows = {
            relative: runner._require_sha(
                runner.REPO_ROOT / relative, expected, "ACCEPTED_SOURCE_HASH",
            )
            for relative, expected in runner.ACCEPTED_SOURCE_HASHES.items()
        }
        return {
            "runner_head": head,
            "approved_runner_head": approved,
            "kimi_commit": KIMI_COMMIT,
            "kimi_is_ancestor": True,
            "accepted_model_tree": model_tree,
            "accepted_model_diff_empty": True,
            "runner_only_delta": delta,
            "unexpected_runner_delta": unexpected,
            "accepted_sources": source_rows,
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
            "admission_source_sha256": runner.sha256_file(
                Path(str(admission.__file__)).resolve()
            ),
            "worktree_clean": not bool(dirty),
        }

    def validate_deterministic_environment(
        environment: Mapping[str, str],
        *,
        require_runner_audit: bool,
        require_tooling_critic: bool = False,
    ) -> dict[str, Any]:
        base = prior_validate_environment(
            environment,
            require_runner_audit=require_runner_audit,
            require_tooling_critic=False,
        )
        if environment.get("GPUWRF_DETERMINISTIC_WAKE_CLOSURE") != "1":
            raise runner.RunnerGateError("DETERMINISTIC_WAKE_ENV", "expected literal 1")
        if environment.get("GPUWRF_DETERMINISTIC_AUTOTUNE_MODE") != mode:
            raise runner.RunnerGateError("DETERMINISTIC_AUTOTUNE_MODE_DRIFT", mode)
        run_dir = (runner.LINEAGE_WORK_DIR / binding["namespace"]).resolve()
        output_pin = (autotune_stage / binding["dump_pin"]).resolve()
        pin_env = Path(environment.get("GPUWRF_DETERMINISTIC_AUTOTUNE_PIN", "")).resolve()
        if pin_env != output_pin:
            raise runner.RunnerGateError(
                "DETERMINISTIC_AUTOTUNE_PIN_PATH",
                f"expected={output_pin} actual={pin_env}",
            )
        if mode == "reference":
            expected_flags = f"--xla_gpu_dump_autotune_results_to={output_pin}"
            if output_pin.exists() or output_pin.is_symlink():
                raise runner.RunnerGateError("REFERENCE_PIN_PREEXISTS", str(output_pin))
            pin_manifest = _authenticate_pin_manifest(runner, required=False)
        else:
            reference_pin = (
                runner.LINEAGE_WORK_DIR / REFERENCE_NAMESPACE / "autotune-results.pb"
            ).resolve()
            expected_flags = (
                f"--xla_gpu_load_autotune_results_from={reference_pin} "
                f"--xla_gpu_dump_autotune_results_to={output_pin}"
            )
            if output_pin.exists() or output_pin.is_symlink():
                raise runner.RunnerGateError("PINNED_OUTPUT_PIN_PREEXISTS", str(output_pin))
            pin_manifest = _authenticate_pin_manifest(runner, required=True)
        if environment.get("XLA_FLAGS") != expected_flags:
            raise runner.RunnerGateError(
                "DETERMINISTIC_XLA_FLAGS",
                f"expected={expected_flags!r} actual={environment.get('XLA_FLAGS')!r}",
            )
        return {
            **base,
            "deterministic_autotune": {
                "mode": mode,
                "xla_flags": expected_flags,
                "output_pin": str(output_pin),
                "pin_manifest": pin_manifest,
                "persistent_compilation_cache_used": False,
            },
        }

    def audit_deterministic_launcher(path: Path) -> dict[str, Any]:
        text = path.read_text()
        required = (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'MODE="${1:?usage: $0 reference|pinned}"',
            "GPUWRF_DETERMINISTIC_WAKE_CLOSURE=1",
            "GPUWRF_DETERMINISTIC_AUTOTUNE_MODE=\"$MODE\"",
            "AUTOTUNE_STAGE=",
            'mkdir -p "$AUTOTUNE_STAGE"',
            'mv -- "$DUMP_PIN" "$PIN"',
            "--xla_gpu_dump_autotune_results_to=$DUMP_PIN",
            "--xla_gpu_load_autotune_results_from=$REFERENCE_PIN",
            REFERENCE_NAMESPACE,
            PINNED_NAMESPACE,
            "v0234-deterministic-fulltree-reference",
            "v0234-deterministic-fulltree-pinned",
            "/usr/bin/env -i",
            "JAX_ENABLE_COMPILATION_CACHE=false",
            "GPUWRF_JAX_CACHE=0",
            "GPUWRF_JAX_CACHE_LOCK=0",
            str(runner.LOCK_WRAPPER),
            "--intent production-preemptible",
            "/usr/bin/taskset -c 12-15",
            "-m scripts.v0234_nested_frozen_wrf_boundary_window",
            "--direct-terminal",
            "--record-known-1500-v10-red",
        )
        forbidden = tuple(
            token for token in (
                "nvidia-smi", "rocm-smi", "JAX_COMPILATION_CACHE_DIR=",
                "GPUWRF_JAX_CACHE_DIR=", "GPUWRF_CACHE=", "--cpu-dry-run",
                "--parent-join-resume", "GPUWRF_TOLERANCE", "GPUWRF_SANITIZER",
            ) if token in text
        )
        missing = [token for token in required if token not in text]
        passed = bool(
            not missing
            and not forbidden
            and text.count("/scripts/with_gpu_lock.sh") == 1
            and text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1
            and text.count('mv -- "$DUMP_PIN" "$PIN"') == 1
        )
        return {
            "passed": passed,
            "path": str(path.resolve()),
            "sha256": runner.sha256_file(path),
            "missing_required_tokens": missing,
            "forbidden_tokens": list(forbidden),
            "lock_commit_bound_by_preimport_authority": runner.LOCK_COMMIT,
            "one_model_process": True,
            "known_1500_v10_record_required": True,
            "tooling_critic_accept_required": False,
            "deterministic_dump_then_pin_pair": True,
            "fresh_namespace_compatible_staged_dump": True,
            "shell_syntax_checked_separately": True,
        }

    def static_deterministic_source_audit() -> dict[str, Any]:
        payload = prior_static_source_audit()
        allowed_drift = {"passed", "known_1500_v10_record_fail_closed"}
        drift = {
            key: {"baseline": baseline_static_audit.get(key), "current": value}
            for key, value in payload.items()
            if key not in allowed_drift and baseline_static_audit.get(key) != value
        }
        wrapper_source = inspect.getsource(classify_diagnostic_observation)
        admission_source = inspect.getsource(admission.classify_known_wake_observation)
        deterministic_checks = {
            "delegates_to_frozen_admission": (
                "admission.classify_known_wake_observation(" in wrapper_source
            ),
            "requires_candidate_cpu_runtime_fingerprint_inputs": all(
                token in admission_source
                for token in ("candidate_path", "cpu_path", "runtime")
            ),
            "metric_and_spatial_policy_both_required": (
                'metric["passed"] and fingerprint.get("policy", {}).get("passed")'
                in admission_source
            ),
            "release_gate_never_green": all(
                token in admission_source
                for token in (
                    '"waiver_or_reclassification": False',
                    '"tolerance_changed": False',
                    '"release_gate_green": False',
                    '"release_blocker_remains": True',
                    '"isolation_decision_only": True',
                )
            ),
            "legacy_invariants_unchanged_except_classifier_binding": not drift,
            "legacy_classifier_is_only_expected_failure": (
                payload.get("known_1500_v10_record_fail_closed") is False
            ),
        }
        deterministic_passed = all(deterministic_checks.values())
        return {
            **payload,
            "passed": deterministic_passed,
            "known_1500_v10_record_fail_closed": deterministic_passed,
            "legacy_static_baseline_sha256": runner.canonical_digest(
                baseline_static_audit
            ),
            "deterministic_wake_classifier": {
                "passed": deterministic_passed,
                "checks": deterministic_checks,
                "unexpected_legacy_audit_drift": drift,
                "admission_source_sha256": runner.sha256_file(
                    Path(str(admission.__file__)).resolve()
                ),
            },
        }

    runner.authenticate_known_1500_v10_observation = authenticate_diagnostic_observation
    runner.classify_exact_known_1500_v10_red = classify_diagnostic_observation
    runner.assert_final_candidate_proof_authority = assert_deterministic_final_authority
    runner.assert_candidate_source_authority = assert_deterministic_source_authority
    runner.validate_preimport_environment = validate_deterministic_environment
    runner.audit_exact_launch_command = audit_deterministic_launcher
    runner.static_source_audit = static_deterministic_source_audit
    runner._DETERMINISTIC_WAKE_CLOSURE_PROFILE_APPLIED = True
