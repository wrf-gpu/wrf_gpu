"""Pre-import contract profile for the single v0234 S1 diff6 GPU arm.

This module is deliberately standard-library-only at import time.  It layers
the accepted RK1 ladder capture on the fixed two-file production tree, binds a
fresh ordinary-program StableHLO audit, and inserts the frozen S1 residual gate
after d03 step 1 and before the loop can dispatch step 2.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
SPRINT_NAME = "2026-07-18-v0234-s1-diff6-gpt-gpu-arm"
SPRINT = REPO / ".agent/sprints" / SPRINT_NAME
CONTRACT = SPRINT / "CONTRACT.md"
CONTRACT_SHA256 = "43c0658250b94141e82cf72dd15915bdf2cf77e91e92c00ccd4fc19616ea380e"
LAUNCHER = SPRINT / "s1-diff6-exact-launch-command.sh"
CPU_AUDIT = SPRINT / "s1-diff6-runner-cpu-proof.json"

CANDIDATE_HEAD = "2e5fa008563cfe56c3c766b21352effc7179c4bf"
FIX_COMMIT = "e65ce784bec85ea4f000e94ba572420c595ecb68"
FABLE_PARENT = "5193db05cd67cfc79292b0206ca43b1d3c68b2e8"
MODEL_HASHES = {
    "src/gpuwrf/dynamics/explicit_diffusion.py": (
        "58e677d4e7748057f4b3546380296f01e99a58fb161b76971343a46aa9792b8c"
    ),
    "src/gpuwrf/runtime/operational_mode.py": (
        "9480c992b46df6a76f0f492baf8098d7945507d1c9e6c44c002bfbd0a4eceaff"
    ),
}

NAMESPACE = "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
LOCK_LABEL = "v0234-s1-diff6-validation"
AUTHORIZATION_PATH = Path("/tmp/v0234-s1-diff6-gpu-authorization.json")
AUTHORIZATION_SCHEMA = "gpuwrf.v0234.s1-diff6-gpu-authorization.v1"
AUTHORIZATION_FIELDS = frozenset({
    "schema", "decision", "issuer", "intent", "one_arm", "sprint",
    "namespace", "lock_label", "candidate_head", "expires_utc", "nonce",
    "proof_sha256",
})
THRESHOLDS = {
    "nonspec": 4.0,
    "relax_rows_1_4": 9.5,
    "interior_ge_5": 1.4,
}

RETAINED_PIN_SHA256 = (
    "edd3b1271dbc59d2998cfc25a062c54ed23cce54e4bd740bbafe625f9bcb222f"
)
AUTOTUNE_STAGE_NAME = ".v0234-s1-diff6-validation-autotune-v1"
DUMP_PIN_NAME = "s1-diff6-validation-autotune-results.pb"
TASKSET_TOKEN = "/usr/bin/taskset -c 13-15,29-31"

FROZEN_ATTRIBUTION = (
    REPO / ".agent/sprints/2026-07-18-v0234-s1-dyn-attribution-fable5/"
    "s1-attribution-analysis.json"
)
FROZEN_ATTRIBUTION_SELF = (
    "12025ab254f560d4746d25211a5f7b3db45f65c8b8af0bd72b1c4d3cc4066190"
)
FROZEN_RUNTIME_PROOF = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_dycore_suboperator_ladder1/"
    "dycore-suboperator-ladder-terminal-proof.json"
)
FROZEN_RUNTIME_SELF = (
    "f0d0b526ed4769f83b8dd2403343186197c5f439f8ad838ee9733e5bfd170e4d"
)
FROZEN_SAVEPOINT_ROWS = (
    "bc974c1848cb625a1dc3bc4edd879dbf6fb4a6a9a3cbc195b25af2865f5e9503"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object, *, omit: str | None = None) -> str:
    if omit is not None and isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != omit}
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_authorization(
    path: Path = AUTHORIZATION_PATH,
    *,
    expected_head: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Authenticate the one-run chief object without importing JAX/gpuwrf."""

    if path.resolve() != AUTHORIZATION_PATH.resolve():
        raise ValueError(f"authorization path must be {AUTHORIZATION_PATH}")
    if not path.is_file() or path.is_symlink():
        raise ValueError("authorization absent, non-file, or symlink")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != AUTHORIZATION_FIELDS:
        raise ValueError("authorization fields are not exact")
    embedded = payload.get("proof_sha256")
    observed = _canonical_digest(payload, omit="proof_sha256")
    if not isinstance(embedded, str) or embedded != observed:
        raise ValueError("authorization canonical self-hash mismatch")
    expected = {
        "schema": AUTHORIZATION_SCHEMA,
        "decision": "RELEASE_GPU",
        "issuer": "chief-0:3",
        "intent": "production-preemptible",
        "one_arm": True,
        "sprint": SPRINT_NAME,
        "namespace": NAMESPACE,
        "lock_label": LOCK_LABEL,
        "candidate_head": expected_head,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"authorization mismatch: {key}")
    nonce = payload.get("nonce")
    if not isinstance(nonce, str) or not nonce.strip():
        raise ValueError("authorization nonce is empty")
    expires_text = payload.get("expires_utc")
    if not isinstance(expires_text, str):
        raise ValueError("authorization expiry is absent")
    try:
        expires = datetime.fromisoformat(expires_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("authorization expiry is malformed") from exc
    if expires.tzinfo is None:
        raise ValueError("authorization expiry is not timezone-aware")
    observed_now = now or datetime.now(timezone.utc)
    if expires <= observed_now:
        raise ValueError("authorization expired")
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "proof_sha256": observed,
        "issuer": payload["issuer"],
        "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
        "expires_utc": expires.isoformat(),
        "candidate_head": expected_head,
        "validated_before_runtime_import": True,
    }


def classify_step1_metrics(metrics: Mapping[str, float]) -> dict[str, Any]:
    """Apply the preregistered inclusive thresholds and expose first red."""

    ordered = ("nonspec", "relax_rows_1_4", "interior_ge_5")
    rows: dict[str, Any] = {}
    first_red: str | None = None
    for name in ordered:
        value = float(metrics[name])
        finite = math.isfinite(value)
        passed = bool(finite and value <= THRESHOLDS[name])
        rows[name] = {
            "value": value,
            "threshold_le": THRESHOLDS[name],
            "finite": finite,
            "passed": passed,
        }
        if first_red is None and not passed:
            first_red = name
    return {
        "passed": first_red is None,
        "first_red": first_red,
        "rows": rows,
        "step2_dispatch_admitted": first_red is None,
    }


def audit_exact_launcher(path: Path = LAUNCHER) -> dict[str, Any]:
    text = path.read_text()
    required = (
        "#!/usr/bin/env bash", "set -euo pipefail", str(AUTHORIZATION_PATH),
        "--verify-authorization", NAMESPACE, LOCK_LABEL,
        "GPUWRF_S1_DIFF6_VALIDATION=1", "GPUWRF_DYCORE_SUBOPERATOR_LADDER=1",
        "GPUWRF_S1_DIFF6_AUTHORIZATION_SHA256=", RETAINED_PIN_SHA256,
        "--xla_gpu_load_autotune_results_from=",
        "--xla_gpu_dump_autotune_results_to=", "/usr/bin/env -i",
        "--timeout 0", "--intent production-preemptible", TASKSET_TOKEN,
        "-m scripts.v0234_nested_frozen_wrf_boundary_window",
        "--seal-release-receipt", "--returncode", "--authorization-sha",
    )
    forbidden = tuple(token for token in (
        "nvidia-smi", "rocm-smi", "--direct-terminal", "--parent-join-resume",
        "--record-known-1500-v10-red", "GPUWRF_TOLERANCE", "GPUWRF_SANITIZER",
    ) if token in text)
    missing = [token for token in required if token not in text]
    auth_index = text.find("--verify-authorization")
    freshness_index = text.find('test ! -e "$RUN_DIR"')
    lock_index = text.find("/scripts/with_gpu_lock.sh")
    release_index = text.find("--seal-release-receipt")
    green_finalize_index = text.find('if [ "$LOCK_RC" -eq 0 ]; then')
    passed = bool(
        not missing and not forbidden
        and text.count("/scripts/with_gpu_lock.sh") == 1
        and text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1
        and 0 <= auth_index < freshness_index < lock_index
        and lock_index < release_index < green_finalize_index
    )
    return {
        "passed": passed,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "missing_required_tokens": missing,
        "forbidden_tokens": list(forbidden),
        "authorization_before_freshness_before_lock": (
            0 <= auth_index < freshness_index < lock_index
        ),
        "canonical_lock_invocations": text.count("/scripts/with_gpu_lock.sh"),
        "model_processes": text.count(
            "-m scripts.v0234_nested_frozen_wrf_boundary_window"
        ),
        "post_wrapper_finalization_only_on_zero": (
            'if [ "$LOCK_RC" -eq 0 ]; then' in text
        ),
        "release_receipt_before_green_finalization": (
            lock_index < release_index < green_finalize_index
        ),
    }


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_S1_DIFF6_GPU_ARM_PROFILE_APPLIED", False):
        return

    # These are read while the historical ladder profile builds its closures.
    runner.LADDER_AUTOTUNE_STAGE_NAME = AUTOTUNE_STAGE_NAME
    runner.LADDER_DUMP_PIN_NAME = DUMP_PIN_NAME
    runner.LADDER_CPU_AUDIT_SKIP_STAGE = True
    from scripts.v0234_dycore_suboperator_kimi_gpu_ladder import (
        apply_profile as apply_ladder_profile,
    )

    apply_ladder_profile(runner)
    prior_validate_environment = runner.validate_preimport_environment

    autotune_stage = (runner.LINEAGE_WORK_DIR / AUTOTUNE_STAGE_NAME).resolve()
    dump_pin = (autotune_stage / DUMP_PIN_NAME).resolve()
    runner.SCHEMA = "gpuwrf.v0234.s1-diff6-validation-arm.v1"
    runner.CPU_PROOF_SCHEMA = "gpuwrf.v0234.s1-diff6-runner-cpu-audit.v1"
    runner.AUDIT_ADMISSION = "READY_FOR_AUTHORIZED_SINGLE_S1_DIFF6_GPU_ARM"
    runner.REPAIRED_PRODUCTION_NAMESPACE = NAMESPACE
    runner.LOCK_LABEL = LOCK_LABEL
    runner.LAUNCH_COMMAND = LAUNCHER
    runner.RUNNER_CPU_AUDIT = CPU_AUDIT
    runner.CPU_FOCUSED_TEST_ARGS = (
        "tests/test_v0234_s1_diff6_gpu_arm.py",
        "tests/test_v0234_s1_dyn_attribution_fable5.py",
        "tests/test_v0234_dycore_suboperator_kimi.py",
    )
    runner.LADDER_SUCCESS_VERDICT = "S1_DIFF6_ARM_GREEN"
    runner.LADDER_MODEL_BEFORE = FABLE_PARENT
    runner.LADDER_MODEL_DELTA_FILES = sorted(MODEL_HASHES)
    runner.INFRASTRUCTURE_GPUWRF_ENV = set(runner.INFRASTRUCTURE_GPUWRF_ENV) | {
        "GPUWRF_S1_DIFF6_VALIDATION",
        "GPUWRF_S1_DIFF6_AUTHORIZATION",
        "GPUWRF_S1_DIFF6_AUTHORIZATION_SHA256",
    }
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        "reviewed_candidate_head": CANDIDATE_HEAD,
        "production_fix_commit": FIX_COMMIT,
        "production_parent": FABLE_PARENT,
        "production_delta_files": sorted(MODEL_HASHES),
        "production_source_sha256": MODEL_HASHES,
        "arm_contract_sha256": CONTRACT_SHA256,
        "fresh_hlo_reference_required": True,
        "step1_thresholds": THRESHOLDS,
        "one_root_step_d03_steps": list(range(1, 10)),
        "gpu_authorization_required": True,
        "release_gate_unchanged": True,
        "tolerance_changed": False,
    }

    def assert_s1_source_authority(
        environment: Mapping[str, str], *, require_clean: bool,
    ) -> dict[str, Any]:
        head = runner._git(runner.REPO_ROOT, "rev-parse", "HEAD")
        approved = environment.get("GPUWRF_NESTED_BUNDLE_RUNNER_SHA")
        if approved and approved != head:
            raise runner.RunnerGateError(
                "RUNNER_HEAD", f"approved={approved} head={head}",
            )
        dirty = runner._git(runner.REPO_ROOT, "status", "--porcelain")
        if require_clean and dirty:
            raise runner.RunnerGateError("WORKTREE_DIRTY", dirty)
        if subprocess.run(
            ("git", "-C", str(runner.REPO_ROOT), "merge-base", "--is-ancestor",
             CANDIDATE_HEAD, head), check=False,
        ).returncode != 0:
            raise runner.RunnerGateError("CANDIDATE_ANCESTRY", head)
        model_delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", FABLE_PARENT, "HEAD",
            "--", "src/gpuwrf",
        ).splitlines()
        if model_delta != sorted(MODEL_HASHES):
            raise runner.RunnerGateError(
                "S1_DIFF6_MODEL_DELTA",
                repr({"expected": sorted(MODEL_HASHES), "actual": model_delta}),
            )
        accepted_sources: dict[str, Any] = {}
        for relative, expected in runner.ACCEPTED_SOURCE_HASHES.items():
            if relative not in MODEL_HASHES:
                accepted_sources[relative] = runner._require_sha(
                    runner.REPO_ROOT / relative, expected, "ACCEPTED_SOURCE_HASH",
                )
        for relative, expected in MODEL_HASHES.items():
            accepted_sources[relative] = runner._require_sha(
                runner.REPO_ROOT / relative, expected, "S1_DIFF6_SOURCE_HASH",
            )
        return {
            "runner_head": head,
            "approved_runner_head": approved,
            "reviewed_candidate_head": CANDIDATE_HEAD,
            "production_fix_commit": FIX_COMMIT,
            "candidate_is_ancestor": True,
            "model_delta_files": model_delta,
            "accepted_sources": accepted_sources,
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
            "launcher_sha256": runner.sha256_file(LAUNCHER),
            "contract_sha256": runner.sha256_file(CONTRACT),
            "worktree_clean": not bool(dirty),
        }

    def validate_s1_environment(
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
        if environment.get("GPUWRF_S1_DIFF6_VALIDATION") != "1":
            raise runner.RunnerGateError("S1_DIFF6_ENV", "expected literal 1")
        authorization: dict[str, Any]
        if require_runner_audit:
            auth_path = Path(environment.get("GPUWRF_S1_DIFF6_AUTHORIZATION", ""))
            try:
                authorization = validate_authorization(
                    auth_path, expected_head=runner._git(runner.REPO_ROOT, "rev-parse", "HEAD"),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise runner.RunnerGateError("S1_DIFF6_AUTHORIZATION", str(exc)) from exc
            if environment.get("GPUWRF_S1_DIFF6_AUTHORIZATION_SHA256") != authorization[
                "file_sha256"
            ]:
                raise runner.RunnerGateError(
                    "S1_DIFF6_AUTHORIZATION_FILE_HASH", authorization["file_sha256"],
                )
        else:
            authorization = {
                "required": False,
                "reason": "CPU dry audit cannot create or consume GPU authorization",
            }
        return {
            **base,
            "s1_diff6_authorization": authorization,
            "fresh_autotune_dump": str(dump_pin),
        }

    def static_s1_audit() -> dict[str, Any]:
        profile_source = Path(__file__).read_text()
        profile_tree = ast.parse(profile_source)
        top_imports: list[str] = []
        for node in profile_tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_imports.append(node.module or "")
        ladder_source = (
            runner.REPO_ROOT / "scripts/v0234_dycore_suboperator_kimi_gpu_ladder.py"
        ).read_text()
        runtime_source = ladder_source[ladder_source.index("def _ladder_runtime_main("):]
        window_source = ladder_source[
            ladder_source.index("def _run_ladder_window("):
            ladder_source.index("def _ladder_runtime_main(")
        ]
        entry_source = runner.RUNNER_SOURCE.read_text()
        launch = audit_exact_launcher(LAUNCHER)
        actual_model_delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", FABLE_PARENT, "HEAD",
            "--", "src/gpuwrf",
        ).splitlines()
        checks = {
            "contract_hash_exact": runner.sha256_file(CONTRACT) == CONTRACT_SHA256,
            "model_delta_exact": actual_model_delta == sorted(MODEL_HASHES),
            "model_hashes_exact": all(
                runner.sha256_file(runner.REPO_ROOT / rel) == digest
                for rel, digest in MODEL_HASHES.items()
            ),
            "profile_top_has_no_runtime_import": not any(
                name.split(".")[0] in {"jax", "gpuwrf", "numpy", "netCDF4"}
                for name in top_imports
            ),
            "fresh_hlo_retained_before_policy": (
                runtime_source.index("fresh_hlo_retainer(")
                < runtime_source.index("evaluate_stablehlo_policy(production_stablehlo)")
            ),
            "ordinary_program_never_compiled_or_dispatched": (
                "production_lowered.compile(" not in runtime_source
                and "production_lowered(" not in runtime_source
            ),
            "step1_savepoints_before_science_gate": (
                window_source.index("_write_step_savepoints(")
                < window_source.index("after_step_gate(")
            ),
            "red_gate_precedes_next_loop_dispatch": (
                "next_dispatch_withheld" in window_source
                and "S1_DIFF6_THRESHOLD_RED" in window_source
            ),
            "s1_profile_selected_first": (
                entry_source.index('GPUWRF_S1_DIFF6_VALIDATION')
                < entry_source.index('GPUWRF_DYCORE_SUBOPERATOR_LADDER')
            ),
            "launcher_exact": launch["passed"],
            "one_root_nine_d03_steps": (
                runner.schedule_clock_oracle()["own_steps"]
                == {"d01": 1, "d02": 3, "d03": 9}
            ),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "model_delta_files": actual_model_delta,
            "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
            "ladder_source_sha256": runner.sha256_file(
                runner.REPO_ROOT / "scripts/v0234_dycore_suboperator_kimi_gpu_ladder.py"
            ),
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "launcher": launch,
        }

    def retain_fresh_production_hlo(
        *, path: Path, stablehlo: str, carry: Any, runtime: Any,
        authority: Mapping[str, Any],
    ) -> dict[str, Any]:
        signature = runner.carry_interface_signature(carry, runtime)
        if signature.get("leaf_count") != 106:
            raise runner.RunnerGateError(
                "S1_DIFF6_HLO_LEAF_COUNT", repr(signature.get("leaf_count")),
            )
        extraction = runner.extract_stablehlo_custom_calls(stablehlo)
        artifact = {
            "schema": "gpuwrf.v0234.s1-diff6-fresh-ordinary-hlo.v1",
            "reference_kind": "fresh-fixed-candidate",
            "namespace": NAMESPACE,
            "runner_head": authority["candidate"]["runner_head"],
            "reviewed_candidate_head": CANDIDATE_HEAD,
            "production_fix_commit": FIX_COMMIT,
            "production_source_sha256": MODEL_HASHES,
            "retained_autotune_input_sha256": RETAINED_PIN_SHA256,
            "carry_interface": signature,
            "extraction": extraction,
            "compile_calls_before_artifact": 0,
            "dispatch_calls_before_artifact": 0,
            "policy_evaluated_before_artifact": False,
            "obsolete_prefix_hlo_used": False,
        }
        artifact["proof_sha256"] = runner.canonical_digest(artifact)
        runner.atomic_write_json(path, artifact)
        return {
            "path": str(path.resolve()),
            "file_sha256": runner.sha256_file(path),
            "proof_sha256": artifact["proof_sha256"],
            "extraction": extraction,
            "carry_leaf_count": signature["leaf_count"],
            "fresh_before_policy_compile_dispatch": True,
        }

    def after_step_gate(
        *, native_step: int, step_rows: tuple[dict[str, Any], ...],
        run_dir: Path, savepoint_dir: Path,
    ) -> dict[str, Any] | None:
        if native_step != 1:
            return None
        if len(step_rows) != 20:
            raise runner.RunnerGateError(
                "S1_DIFF6_STEP1_INVENTORY", f"expected=20 actual={len(step_rows)}",
            )
        frozen_runtime = json.loads(FROZEN_RUNTIME_PROOF.read_text())
        if (
            frozen_runtime.get("proof_sha256") != FROZEN_RUNTIME_SELF
            or runner.canonical_digest({
                key: value for key, value in frozen_runtime.items()
                if key != "proof_sha256"
            }) != FROZEN_RUNTIME_SELF
            or runner.canonical_digest(frozen_runtime["savepoint_manifest"]["rows"])
            != FROZEN_SAVEPOINT_ROWS
        ):
            raise runner.RunnerGateError(
                "S1_DIFF6_FROZEN_RUNTIME_AUTHORITY", str(FROZEN_RUNTIME_PROOF),
            )
        expected_shapes = {
            (row["tag"], row["field"]): row["shape"]
            for row in frozen_runtime["savepoint_manifest"]["rows"]
            if row["step"] == 1
        }
        fresh_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for row in step_rows:
            key = (row["tag"], row["field"])
            path = Path(row["path"])
            if (
                key in fresh_rows
                or key not in expected_shapes
                or row["shape"] != expected_shapes[key]
                or path.parent != savepoint_dir.resolve()
                or path.is_symlink()
                or runner.sha256_file(path) != row["file_sha256"]
            ):
                raise runner.RunnerGateError(
                    "S1_DIFF6_STEP1_SAVEPOINT_AUTHORITY", repr(row),
                )
            fresh_rows[key] = row
        if set(fresh_rows) != set(expected_shapes):
            raise runner.RunnerGateError(
                "S1_DIFF6_STEP1_SAVEPOINT_KEYS", repr(sorted(fresh_rows)),
            )

        attribution = json.loads(FROZEN_ATTRIBUTION.read_text())
        if (
            attribution.get("proof_sha256") != FROZEN_ATTRIBUTION_SELF
            or runner.canonical_digest({
                key: value for key, value in attribution.items()
                if key != "proof_sha256"
            }) != FROZEN_ATTRIBUTION_SELF
        ):
            raise runner.RunnerGateError(
                "S1_DIFF6_ATTRIBUTION_AUTHORITY", str(FROZEN_ATTRIBUTION),
            )

        # Runtime imports are intentionally deferred until after the live step-1
        # arrays have been retained and authenticated.
        from scripts import v0234_dycore_internal_split_kimi as split
        from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble
        from scripts import v0234_s1_dyn_attribution_fable5 as s1

        np = sys.modules["numpy"]
        load = lambda tag, field: np.load(  # noqa: E731
            fresh_rows[(tag, field)]["path"], allow_pickle=False,
        )
        gpu = {
            "ru_tend": load("l1_rk1_tend", "ru_tend"),
            "rv_tend": load("l1_rk1_tend", "rv_tend"),
            "ru_tendf": load("sp3_tendf", "ru_tendf"),
            "rv_tendf": load("sp3_tendf", "rv_tendf"),
            "u_save": load("l1_rk1_relax", "u_save"),
            "v_save": load("l1_rk1_relax", "v_save"),
        }
        ranks = reassemble.load_ranks(split.RUNS / "control/momsp_dumps")
        control = {
            "ru_tend": reassemble.reassemble3d("l1_rk1_tend__ru_tend", 1, ranks),
            "rv_tend": reassemble.reassemble3d("l1_rk1_tend__rv_tend", 1, ranks),
            "ru_tendf": reassemble.reassemble3d("sp3_tendf__ru_tendf", 1, ranks),
            "rv_tendf": reassemble.reassemble3d("sp3_tendf__rv_tendf", 1, ranks),
            "u_save": reassemble.reassemble3d("l1_rk1_tend__u_save", 1, ranks),
            "v_save": reassemble.reassemble3d("l1_rk1_tend__v_save", 1, ranks),
        }
        const = split.load_constants()
        residual = s1.s1_residual(gpu, control, const)
        bands = split.per_band_rmse(residual)
        nonspec = split.combined_rmse(residual, s1.nonspec_masks(residual))
        metrics = {
            "nonspec": nonspec,
            "relax_rows_1_4": bands["relax_rows_1_4"],
            "interior_ge_5": bands["interior_ge_5"],
        }
        gate = classify_step1_metrics(metrics)
        gate.update({
            "schema": "gpuwrf.v0234.s1-diff6-step1-gate.v1",
            "step": 1,
            "namespace": NAMESPACE,
            "frozen_attribution_self_hash": FROZEN_ATTRIBUTION_SELF,
            "frozen_runtime_self_hash": FROZEN_RUNTIME_SELF,
            "frozen_savepoint_rows_hash": FROZEN_SAVEPOINT_ROWS,
            "fresh_savepoint_rows_hash": runner.canonical_digest(list(step_rows)),
            "fresh_savepoints_authenticated": 20,
            "dispatched_steps_at_gate": 1,
            "step2_withheld_on_red": True,
            "artifact_path": str(
                (run_dir / "step1-s1-residual-gate.json").resolve()
            ),
        })
        gate["proof_sha256"] = runner.canonical_digest(gate)
        gate_path = run_dir / "step1-s1-residual-gate.json"
        runner.atomic_write_json(gate_path, gate)
        return gate

    def cpu_dry_run(args: Any, authority: Mapping[str, Any]) -> int:
        if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
            raise runner.RunnerGateError("CPU_DRY_IMPORT_PRE", "runtime already imported")
        schedule = runner.schedule_clock_oracle()
        static = static_s1_audit()
        launch = audit_exact_launcher(LAUNCHER)
        if not schedule["passed"] or not static["passed"] or not launch["passed"]:
            raise runner.RunnerGateError(
                "CPU_STATIC_AUDIT",
                repr({"schedule": schedule, "static": static, "launch": launch}),
            )
        focused = {
            "command": [sys.executable, "-m", "pytest", "-q", *runner.CPU_FOCUSED_TEST_ARGS],
            "returncode": 0,
            "stdout_sha256": None,
            "stdout_tail": None,
        }
        if args.run_focused_tests:
            completed = subprocess.run(
                focused["command"], cwd=runner.REPO_ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"},
            )
            normalized = runner.normalize_pytest_output(completed.stdout)
            focused.update({
                "returncode": completed.returncode,
                "stdout_sha256": runner.sha256_text(normalized),
                "stdout_tail": normalized[-4000:],
                "stdout_normalization": "pytest elapsed seconds replaced with <elapsed>s",
            })
            if completed.returncode:
                raise runner.RunnerGateError("FOCUSED_TESTS", completed.stdout[-4000:])
        if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
            raise runner.RunnerGateError("CPU_DRY_IMPORT_POST", "runtime imported")
        proof = {
            "schema": runner.CPU_PROOF_SCHEMA,
            "verdict": runner.AUDIT_ADMISSION,
            "candidate_commit": runner.CANDIDATE_COMMIT,
            "candidate_tree": runner.CANDIDATE_TREE,
            "reviewed_candidate_head": CANDIDATE_HEAD,
            "runner_head_at_audit": authority["candidate"]["runner_head"],
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
            "candidate_clean_authority": runner.CANDIDATE_CLEAN_AUTHORITY,
            "authority": authority,
            "schedule_clock_oracle": schedule,
            "static_audit": static,
            "exact_launch_audit": launch,
            "focused_tests": focused,
            "gpu_commands_run": 0,
            "gpu_queries_run": 0,
            "gpu_lock_acquired_or_verified": False,
            "model_imported_or_executed": False,
            "jax_imported": False,
            "fresh_hlo_reference_required_at_authorized_runtime": True,
            "step1_thresholds": THRESHOLDS,
            "authorization_consumed": False,
            "cpu_dry_run": True,
            "deterministic_payload": True,
        }
        proof["proof_sha256"] = runner.canonical_digest(proof)
        runner.atomic_write_json(args.proof_output, proof)
        print(json.dumps({
            "verdict": proof["verdict"],
            "proof": str(args.proof_output),
            "proof_sha256": proof["proof_sha256"],
            "gpu_commands_run": 0,
            "jax_imported": False,
        }, sort_keys=True), flush=True)
        return 0

    def retain_terminal_outcome(
        args: Any,
        *,
        code: str,
        detail: str,
        unexpected: bool,
    ) -> int:
        dispatched = len(list((args.run_dir / "savepoints").glob("step*_*.npy"))) // 20
        failure_path = args.run_dir / "failure/failure-proof.json"
        step1_gate_path = args.run_dir / "step1-s1-residual-gate.json"

        def artifact_row(path: Path) -> dict[str, Any] | None:
            if not path.is_file() or path.is_symlink():
                return None
            payload = json.loads(path.read_text())
            embedded = payload.get("proof_sha256")
            observed = runner.canonical_digest({
                key: value for key, value in payload.items()
                if key != "proof_sha256"
            })
            return {
                "path": str(path.resolve()),
                "file_sha256": runner.sha256_file(path),
                "proof_sha256": embedded,
                "self_hash_valid": embedded == observed,
            }

        threshold_red = code == "S1_DIFF6_THRESHOLD_RED"
        proof = {
            "schema": runner.SCHEMA,
            "verdict": (
                "S1_DIFF6_FIX_FALSIFIED" if threshold_red
                else "S1_DIFF6_ARM_BLOCKED"
            ),
            "namespace": NAMESPACE,
            "failure_code": code,
            "detail": detail,
            "unexpected_exception": unexpected,
            "model_dispatches": dispatched,
            "step2_withheld": bool(threshold_red and dispatched == 1),
            "failure_capture": artifact_row(failure_path),
            "step1_scientific_gate": artifact_row(step1_gate_path),
            "gpu_release": {
                "mechanism": "canonical lock-v2 file-descriptor wrapper",
                "receipt_path": str(
                    (args.run_dir / "gpu-lock-release-receipt.json").resolve()
                ),
                "receipt_sealed_by_launcher_after_wrapper_return": True,
            },
            "retry_authorized": False,
        }
        proof["proof_sha256"] = runner.canonical_digest(proof)
        runner.atomic_write_json(args.proof_output, proof)
        print(json.dumps({
            "verdict": proof["verdict"],
            "failure_code": code,
            "model_dispatches": dispatched,
            "proof": str(args.proof_output),
            "proof_sha256": proof["proof_sha256"],
        }, sort_keys=True), flush=True)
        return 3

    def window_falsified_handler(args: Any, exc: Any) -> int:
        return retain_terminal_outcome(
            args, code=exc.code, detail=exc.detail, unexpected=False,
        )

    def runtime_blocker_handler(args: Any, exc: Any, unexpected: bool) -> int:
        return retain_terminal_outcome(
            args,
            code=(
                f"UNEXPECTED_{type(exc).__name__}"
                if unexpected else str(exc.code)
            ),
            detail=(
                f"{type(exc).__name__}: {exc}"
                if unexpected else str(exc.detail)
            ),
            unexpected=unexpected,
        )

    runner.assert_candidate_source_authority = assert_s1_source_authority
    runner.validate_preimport_environment = validate_s1_environment
    runner.audit_exact_launch_command = audit_exact_launcher
    runner.static_source_audit = static_s1_audit
    runner.RETAIN_FRESH_PRODUCTION_HLO = retain_fresh_production_hlo
    runner.AFTER_LADDER_STEP_GATE = after_step_gate
    runner.PROFILE_WINDOW_FALSIFIED_HANDLER = window_falsified_handler
    runner.PROFILE_RUNTIME_BLOCKER_HANDLER = runtime_blocker_handler
    runner._cpu_dry_run = cpu_dry_run
    runner._S1_DIFF6_GPU_ARM_PROFILE_APPLIED = True


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-authorization", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--seal-release-receipt", type=Path)
    parser.add_argument("--returncode", type=int)
    parser.add_argument("--authorization-sha")
    args = parser.parse_args(argv)
    verify_mode = args.verify_authorization is not None
    receipt_mode = args.seal_release_receipt is not None
    if verify_mode == receipt_mode:
        parser.error("select exactly one authorization-verify or release-receipt mode")
    if verify_mode:
        if not args.expected_head:
            parser.error("--expected-head is required for authorization verification")
        row = validate_authorization(
            args.verify_authorization, expected_head=args.expected_head,
        )
        print(row["file_sha256"])
        return 0
    if args.returncode is None or not args.authorization_sha:
        parser.error("--returncode and --authorization-sha are required for receipt")
    run_dir = args.seal_release_receipt.resolve()
    if not run_dir.is_dir() or run_dir.is_symlink() or run_dir.name != NAMESPACE:
        raise ValueError("release receipt run directory is not the exact namespace")
    receipt = run_dir / "gpu-lock-release-receipt.json"
    if receipt.exists() or receipt.is_symlink():
        raise ValueError("release receipt already exists")
    terminal_rows = []
    for path in sorted(run_dir.glob("*terminal-proof.json")):
        terminal_rows.append({
            "path": str(path.resolve()),
            "file_sha256": _sha256_file(path),
        })
    payload = {
        "schema": "gpuwrf.v0234.s1-diff6-lock-release-receipt.v1",
        "namespace": NAMESPACE,
        "lock_label": LOCK_LABEL,
        "lock_commit": "8152309aff1e85e1052d44d549a5a5409e710bdd",
        "lock_wrapper_sha256": (
            "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
        ),
        "wrapper_returned": True,
        "file_descriptor_lease_released_before_receipt": True,
        "model_returncode": args.returncode,
        "authorization_file_sha256": args.authorization_sha,
        "terminal_proofs": terminal_rows,
        "gpu_queries_run_by_receipt": 0,
    }
    payload["proof_sha256"] = _canonical_digest(payload)
    temporary = receipt.with_name(receipt.name + ".part")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, receipt)
    print(payload["proof_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
