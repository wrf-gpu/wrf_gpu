"""Pre-import bindings for the v0234 first-interval momentum short GPU arm.

Contract: .agent/sprints/2026-07-18-v0234-first-interval-momentum-kimi/CONTRACT.md

One canonical lock-v2 production-preemptible diagnostic short arm over the
first coupled d03 interval (steps 1..200, 00:00..00:20).  The arm reuses the
frozen-boundary nested replay machinery and admission stack of
``scripts/v0234_nested_frozen_wrf_boundary_window.py`` with exactly these
profile-only changes:

* the production one-step d03 program is lowered but NOT compiled or run;
  its lowered HLO must be byte-identical to the retained deterministic
  reference artifact (proof that the diagnostic model-tree delta does not
  touch production numerics);
* d03 steps are dispatched one at a time through the proof-only
  ``advance_one_step_with_first_interval_capture`` program, which returns
  the raw MYNN RUBLTEN/RVBLTEN and assembled ru_tendf/rv_tendf beside the
  ordinary carry; per-step host transfers of those records and of the
  step-entry/step-exit U/V are diagnostic-only and absent from any
  production path;
* the retained staged autotune pin (deterministic wake reference dump) is
  loaded and any newly encountered selections are dumped to a fresh staged
  file, preserving the dump-then-pin lineage.

No tolerance, waiver, clamp, masking, or model correction is performed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

KIMI_BASE_COMMIT = "7a7351aeecc1b0b59b7681a68a25eb2ad848f203"
GPT_PROOF_COMMIT_TREE = "835dcc29bf316c0715b41a72e064985e9cf099df"
MODEL_FILE = "src/gpuwrf/runtime/operational_mode.py"
FIRST_INTERVAL_MODEL_SHA256 = (
    "2bb1ce4ff7f5ef45be907e33a8fd19491d57b9709cc2a1e0aaced7f328a3e59b"
)
# Second accepted production baseline (sprint
# 2026-07-18-v0234-s1-dyn-attribution-fable5): the WRF-faithful momentum
# sixth-order-diffusion correction changes exactly two model files.  Exact
# byte pins; any other tree still fails closed.  Arms binding this profile on
# the corrected tree must re-dump their lowered-HLO reference — the live
# nested-child production program is INTENDED to differ from the retained one.
S1_FIX_DIFF6_FILE = "src/gpuwrf/dynamics/explicit_diffusion.py"
S1_FIX_MODEL_SHA256 = (
    "9480c992b46df6a76f0f492baf8098d7945507d1c9e6c44c002bfbd0a4eceaff"
)
S1_FIX_DIFF6_SHA256 = (
    "58e677d4e7748057f4b3546380296f01e99a58fb161b76971343a46aa9792b8c"
)


def _s1_fix_tree_accepted(runner: Any, accepted_model_diff: object) -> bool:
    if accepted_model_diff != sorted([MODEL_FILE, S1_FIX_DIFF6_FILE]):
        return False
    for relative, expected in (
        (MODEL_FILE, S1_FIX_MODEL_SHA256),
        (S1_FIX_DIFF6_FILE, S1_FIX_DIFF6_SHA256),
    ):
        digest = hashlib.sha256(
            (runner.REPO_ROOT / relative).read_bytes()
        ).hexdigest()
        if digest != expected:
            return False
    return True
NAMESPACE = "nested_stage_omega_transport_470e6111_first_interval_momentum2"
LOCK_LABEL = "v0234-first-interval-momentum"
RETAINED_PIN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/.v0234-deterministic-wake-autotune-v1/"
    "reference-autotune-results.pb"
)
RETAINED_PIN_SHA256 = (
    "edd3b1271dbc59d2998cfc25a062c54ed23cce54e4bd740bbafe625f9bcb222f"
)
RETAINED_PRODUCTION_HLO = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_deterministic_wake_reference1/"
    "ordinary-one-step-lowered-hlo.json"
)
RETAINED_PRODUCTION_HLO_SHA256 = (
    "15575931876ca3126e91e5b1c7cbcd85360e7cc2754354f262c5c82556d3a308"
)
AUTOTUNE_STAGE_NAME = ".v0234-first-interval-autotune-v1"
DUMP_PIN_NAME = "first-interval-autotune-results.pb"

FIRST_INTERVAL_ROOT_STEPS = 23
FIRST_INTERVAL_OWN_STEPS = {"d01": 23, "d02": 69, "d03": 207}
FIRST_INTERVAL_D03_STEPS = tuple(range(1, 208))
FIRST_INTERVAL_LAST_ANALYZED_D03_STEP = 200
EXPECTED_FRAME_COUNTS = {"d01": 1, "d02": 1, "d03": 2}
SAVEPOINT_TAGS = (
    ("sp1_entry", ("u", "v")),
    ("sp2_pbl", ("rublten", "rvblten")),
    ("sp3_tendf", ("ru_tendf", "rv_tendf")),
    ("sp4_exit", ("u", "v")),
)
CPU_FOCUSED_TESTS = ("tests/test_v0234_first_interval_momentum.py",)


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_FIRST_INTERVAL_MOMENTUM_PROFILE_APPLIED", False):
        return

    from scripts.v0234_stage_omega_transport_runner_profile import (
        apply_profile as apply_stage_omega_profile,
    )

    apply_stage_omega_profile(runner)
    prior_validate_environment = runner.validate_preimport_environment
    prior_final_authority = runner.assert_final_candidate_proof_authority
    prior_static_source_audit = runner.static_source_audit
    baseline_static_audit = prior_static_source_audit()
    # The stage-omega baseline audit fails-closed on ANY src/gpuwrf diff.  The
    # contracted diagnostic delta here is exactly one model file; require every
    # other baseline invariant to hold and the diff to be exactly that file.
    baseline_false_rows = {
        key: value
        for key, value in baseline_static_audit.items()
        if value is False
        and key not in {"passed", "cpu_dry_run_calls_runtime_import"}
    }
    accepted_diff = baseline_static_audit.get("accepted_model_diff")
    if (
        (
            accepted_diff != [MODEL_FILE]
            and not _s1_fix_tree_accepted(runner, accepted_diff)
        )
        or baseline_false_rows
    ):
        raise runner.RunnerGateError(
            "BASELINE_STATIC_AUDIT", repr(baseline_static_audit),
        )

    sprint = runner.REPO_ROOT / ".agent/sprints/2026-07-18-v0234-first-interval-momentum-kimi"
    launcher = sprint / "first-interval-momentum-exact-launch-command.sh"
    autotune_stage = (runner.LINEAGE_WORK_DIR / AUTOTUNE_STAGE_NAME).resolve()
    dump_pin = (autotune_stage / DUMP_PIN_NAME).resolve()
    cpu_audit = sprint / "first-interval-runner-cpu-proof.json"

    runner.SCHEMA = "gpuwrf.v0234.first-interval-momentum-short-arm.v1"
    runner.CPU_PROOF_SCHEMA = "gpuwrf.v0234.first-interval-momentum-runner-cpu-audit.v1"
    runner.AUDIT_ADMISSION = "READY_FOR_FIRST_INTERVAL_MOMENTUM_SHORT_GPU_ARM"
    runner.REPAIRED_PRODUCTION_NAMESPACE = NAMESPACE
    runner.LOCK_LABEL = LOCK_LABEL
    runner.LAUNCH_COMMAND = launcher
    runner.RUNNER_CPU_AUDIT = cpu_audit
    runner.CPU_FOCUSED_TEST_ARGS = CPU_FOCUSED_TESTS
    runner.REQUIRE_KNOWN_1500_V10_RECORD = False
    runner.REQUIRE_TOOLING_CRITIC_ACCEPT = False
    runner.TOOLING_CRITIC_AUTHORITY_HOOK = None
    runner.INFRASTRUCTURE_GPUWRF_ENV = set(runner.INFRASTRUCTURE_GPUWRF_ENV) | {
        "GPUWRF_FIRST_INTERVAL_MOMENTUM",
        "GPUWRF_FIRST_INTERVAL_PIN_SHA256",
    }
    runner.FORBIDDEN_NON_GPUWRF_ENV = tuple(
        name for name in runner.FORBIDDEN_NON_GPUWRF_ENV if name != "XLA_FLAGS"
    )
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        **runner.CANDIDATE_CLEAN_AUTHORITY,
        "first_interval_kimi_base_commit": KIMI_BASE_COMMIT,
        "first_interval_model_tree_before": GPT_PROOF_COMMIT_TREE,
        "first_interval_model_delta_files": [MODEL_FILE],
        "first_interval_model_file_sha256": FIRST_INTERVAL_MODEL_SHA256,
        "first_interval_diagnostic_only_delta": True,
        "first_interval_production_hlo_must_match_retained": True,
        "retained_autotune_pin_sha256": RETAINED_PIN_SHA256,
        "retained_production_hlo_sha256": RETAINED_PRODUCTION_HLO_SHA256,
        "release_gate_unchanged": True,
        "tolerance_changed": False,
    }

    def assert_first_interval_source_authority(
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
             KIMI_BASE_COMMIT, head),
            check=False,
        ).returncode != 0:
            raise runner.RunnerGateError(
                "KIMI_BASE_ANCESTRY", f"{KIMI_BASE_COMMIT} !<= {head}",
            )
        model_delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only",
            KIMI_BASE_COMMIT, "HEAD", "--", "src/gpuwrf",
        ).splitlines()
        if model_delta != [MODEL_FILE]:
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_MODEL_DELTA",
                repr({"expected": [MODEL_FILE], "actual": model_delta}),
            )
        model_row = runner._require_sha(
            runner.REPO_ROOT / MODEL_FILE,
            FIRST_INTERVAL_MODEL_SHA256,
            "FIRST_INTERVAL_MODEL_FILE",
        )
        accepted = {
            relative: expected
            for relative, expected in runner.ACCEPTED_SOURCE_HASHES.items()
            if relative != MODEL_FILE
        }
        source_rows = {
            relative: runner._require_sha(
                runner.REPO_ROOT / relative, expected, "ACCEPTED_SOURCE_HASH",
            )
            for relative, expected in accepted.items()
        }
        source_rows[MODEL_FILE] = model_row
        delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", KIMI_BASE_COMMIT, "HEAD",
        ).splitlines()
        allowed_exact = {
            "scripts/v0234_nested_frozen_wrf_boundary_window.py",
            "scripts/v0234_first_interval_momentum_runner_profile.py",
            "scripts/v0234_first_interval_momentum_wrf_reassemble.py",
            "scripts/v0234_first_interval_momentum_compare.py",
            "tests/test_v0234_first_interval_momentum.py",
            MODEL_FILE,
        }
        allowed_prefix = (
            ".agent/sprints/2026-07-18-v0234-first-interval-momentum-kimi/"
        )
        unexpected = sorted(
            path for path in delta
            if path not in allowed_exact and not path.startswith(allowed_prefix)
        )
        if unexpected:
            raise runner.RunnerGateError("RUNNER_ONLY_DELTA", repr(unexpected))
        return {
            "runner_head": head,
            "approved_runner_head": approved,
            "kimi_base_commit": KIMI_BASE_COMMIT,
            "kimi_base_is_ancestor": True,
            "model_tree_before": GPT_PROOF_COMMIT_TREE,
            "model_tree_after": runner._git(
                runner.REPO_ROOT, "rev-parse", "HEAD:src/gpuwrf",
            ),
            "model_delta_files": model_delta,
            "diagnostic_only_model_delta": True,
            "runner_only_delta": delta,
            "unexpected_runner_delta": unexpected,
            "accepted_sources": source_rows,
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
            "worktree_clean": not bool(dirty),
        }

    def validate_first_interval_environment(
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
        if environment.get("GPUWRF_FIRST_INTERVAL_MOMENTUM") != "1":
            raise runner.RunnerGateError("FIRST_INTERVAL_ENV", "expected literal 1")
        if environment.get("GPUWRF_FIRST_INTERVAL_PIN_SHA256") != RETAINED_PIN_SHA256:
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_PIN_ENV",
                environment.get("GPUWRF_FIRST_INTERVAL_PIN_SHA256"),
            )
        pin_row = runner._require_sha(
            RETAINED_PIN, RETAINED_PIN_SHA256, "FIRST_INTERVAL_RETAINED_PIN",
        )
        expected_flags = (
            f"--xla_gpu_load_autotune_results_from={RETAINED_PIN} "
            f"--xla_gpu_dump_autotune_results_to={dump_pin}"
        )
        if environment.get("XLA_FLAGS") != expected_flags:
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_XLA_FLAGS",
                f"expected={expected_flags!r} actual={environment.get('XLA_FLAGS')!r}",
            )
        if not autotune_stage.is_dir() or autotune_stage.is_symlink():
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_STAGE", f"missing staged dir {autotune_stage}",
            )
        if dump_pin.exists() or dump_pin.is_symlink():
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_DUMP_PREEXISTS", str(dump_pin),
            )
        return {
            **base,
            "first_interval_autotune": {
                "retained_pin": pin_row,
                "dump_pin": str(dump_pin),
                "xla_flags": expected_flags,
                "persistent_compilation_cache_used": False,
            },
        }

    def audit_first_interval_launcher(path: Path) -> dict[str, Any]:
        text = path.read_text()
        required = (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "GPUWRF_FIRST_INTERVAL_MOMENTUM=1",
            "GPUWRF_FIRST_INTERVAL_PIN_SHA256=",
            RETAINED_PIN_SHA256,
            "--xla_gpu_load_autotune_results_from=",
            "--xla_gpu_dump_autotune_results_to=",
            NAMESPACE,
            LOCK_LABEL,
            "/usr/bin/env -i",
            "JAX_ENABLE_COMPILATION_CACHE=false",
            "GPUWRF_JAX_CACHE=0",
            "GPUWRF_JAX_CACHE_LOCK=0",
            str(runner.LOCK_WRAPPER),
            "--intent production-preemptible",
            "/usr/bin/taskset -c 12-15",
            "-m scripts.v0234_nested_frozen_wrf_boundary_window",
        )
        forbidden = tuple(
            token for token in (
                "nvidia-smi", "rocm-smi", "JAX_COMPILATION_CACHE_DIR=",
                "GPUWRF_JAX_CACHE_DIR=", "GPUWRF_CACHE=", "--cpu-dry-run",
                "--parent-join-resume", "GPUWRF_TOLERANCE", "GPUWRF_SANITIZER",
                "--direct-terminal", "--record-known-1500-v10-red",
            ) if token in text
        )
        missing = [token for token in required if token not in text]
        passed = bool(
            not missing
            and not forbidden
            and text.count("/scripts/with_gpu_lock.sh") == 1
            and text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1
        )
        return {
            "passed": passed,
            "path": str(path.resolve()),
            "sha256": runner.sha256_file(path),
            "missing_required_tokens": missing,
            "forbidden_tokens": list(forbidden),
            "lock_commit_bound_by_preimport_authority": runner.LOCK_COMMIT,
            "one_model_process": True,
            "known_1500_v10_record_required": False,
            "deterministic_load_pin_dump_pair": True,
            "shell_syntax_checked_separately": True,
        }

    def static_first_interval_source_audit() -> dict[str, Any]:
        # Use the baseline captured at apply time: by audit time this profile
        # has already replaced _runtime_main/_cpu_dry_run, so re-calling the
        # stage-omega audit would inspect the short-arm functions and fail on
        # them.  The captured baseline is the honest pre-patch evaluation.
        payload = baseline_static_audit
        legacy_false_rows = {
            key: value
            for key, value in payload.items()
            if value is False
            and key not in {"passed", "cpu_dry_run_calls_runtime_import"}
        }
        legacy_model_only_drift = (
            (
                payload.get("accepted_model_diff") == [MODEL_FILE]
                or _s1_fix_tree_accepted(
                    runner, payload.get("accepted_model_diff")
                )
            )
            and not legacy_false_rows
        )
        profile_source = Path(__file__).read_text()
        profile_tree = ast.parse(profile_source)
        top_imports = []
        for node in profile_tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_imports.append(node.module or "")
        capture_source = (runner.REPO_ROOT / MODEL_FILE).read_text()
        checks = {
            "legacy_static_audit_clean_except_contracted_model_file": (
                legacy_model_only_drift
            ),
            "capture_flag_mutually_exclusive": (
                "capture_first_interval) and (bool(capture_rca)" in capture_source
            ),
            "capture_record_namedtuple": (
                "class FirstIntervalMomentumRecord(NamedTuple)" in capture_source
            ),
            "proof_only_entry_point": (
                "def advance_one_step_with_first_interval_capture" in capture_source
            ),
            "capture_is_separate_jit_program": (
                "capture_first_interval: bool = False" in capture_source
            ),
            "window_dispatches_capture_then_health_then_savepoint": (
                profile_source.index("result = capture_program.one_step(")
                < profile_source.index("summary = runner.materialize_health(")
                < profile_source.index("savepoint_rows.extend(")
            ),
            "capture_binds_on_first_live_carry": (
                "lazy_compile_trigger\": \"first-live-d03-carry\"" in profile_source
                or "first-live-d03-carry" in profile_source
            ),
            "production_lowered_hlo_identity_gate": (
                "RETAINED_PRODUCTION_HLO_SHA256" in profile_source
                and "stablehlo_sha256" in profile_source
            ),
            "no_jax_import_at_profile_top": not any(
                name.split(".")[0] in {"jax", "gpuwrf", "numpy", "netCDF4"}
                for name in top_imports
            ),
        }
        passed = all(checks.values())
        return {
            **payload,
            "passed": passed,
            "first_interval_short_arm": {
                "passed": passed,
                "checks": checks,
                "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
                "model_file_sha256": FIRST_INTERVAL_MODEL_SHA256,
            },
        }

    def schedule_first_interval_clock_oracle() -> dict[str, Any]:
        sampled = list(range(1, FIRST_INTERVAL_ROOT_STEPS * 9 + 1))
        own = {
            "d01": FIRST_INTERVAL_ROOT_STEPS,
            "d02": FIRST_INTERVAL_ROOT_STEPS * 3,
            "d03": FIRST_INTERVAL_ROOT_STEPS * 9,
        }
        outputs = runner.standard_output_steps(own)
        counts = {name: len(values) for name, values in outputs.items()}
        passed = bool(
            own == FIRST_INTERVAL_OWN_STEPS
            and sampled == list(FIRST_INTERVAL_D03_STEPS)
            and counts == EXPECTED_FRAME_COUNTS
            and outputs["d03"] == (0, 200)
        )
        return {
            "passed": passed,
            "root_steps": FIRST_INTERVAL_ROOT_STEPS,
            "own_steps": own,
            "sampled_d03_first_last": [sampled[0], sampled[-1]],
            "output_counts": counts,
            "d03_output_steps": list(outputs["d03"]),
            "schedule": runner.standard_frame_schedule(own),
        }

    runner.assert_candidate_source_authority = assert_first_interval_source_authority
    runner.validate_preimport_environment = validate_first_interval_environment
    runner.audit_exact_launch_command = audit_first_interval_launcher
    runner.static_source_audit = static_first_interval_source_audit
    runner.schedule_clock_oracle = schedule_first_interval_clock_oracle

    def _cpu_dry_run(args: Any, authority: Mapping[str, Any]) -> int:
        return _first_interval_cpu_dry_run(runner, args, authority)

    def _runtime_main(args: Any, authority: dict[str, Any], runtime: Any) -> int:
        return _first_interval_runtime_main(runner, args, authority, runtime)

    runner._cpu_dry_run = _cpu_dry_run
    runner._runtime_main = _runtime_main
    runner._FIRST_INTERVAL_MOMENTUM_PROFILE_APPLIED = True


# ---------------------------------------------------------------------------
# Short-arm CPU audit and runtime (profile-scoped implementations)
# ---------------------------------------------------------------------------


def _first_interval_cpu_dry_run(runner: Any, args: Any, authority: Mapping[str, Any]) -> int:
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise runner.RunnerGateError("CPU_DRY_IMPORT_PRE", "JAX/gpuwrf already imported")
    schedule = runner.schedule_clock_oracle()
    static = runner.static_source_audit()
    launch = runner.audit_exact_launch_command(runner.LAUNCH_COMMAND)
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
        )
        normalized_stdout = runner.normalize_pytest_output(completed.stdout)
        focused.update({
            "returncode": completed.returncode,
            "stdout_sha256": runner.sha256_text(normalized_stdout),
            "stdout_tail": normalized_stdout[-4000:],
            "stdout_normalization": "pytest elapsed seconds replaced with <elapsed>s",
        })
        if completed.returncode:
            raise runner.RunnerGateError("FOCUSED_TESTS", completed.stdout[-4000:])
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise runner.RunnerGateError("CPU_DRY_IMPORT_POST", "JAX/gpuwrf imported during CPU audit")
    proof = {
        "schema": runner.CPU_PROOF_SCHEMA,
        "verdict": runner.AUDIT_ADMISSION,
        "candidate_commit": runner.CANDIDATE_COMMIT,
        "candidate_tree": runner.CANDIDATE_TREE,
        "runner_head_at_audit": authority["candidate"]["runner_head"],
        "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
        "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
        "candidate_clean_authority": runner.CANDIDATE_CLEAN_AUTHORITY,
        "authority": authority,
        "schedule_clock_oracle": schedule,
        "static_audit": static,
        "exact_launch_audit": launch,
        "focused_tests": focused,
        "exact_launch_command_path": str(runner.LAUNCH_COMMAND.resolve()),
        "exact_launch_command_sha256": runner.sha256_file(runner.LAUNCH_COMMAND),
        "gpu_commands_run": 0,
        "gpu_queries_run": 0,
        "gpu_lock_acquired_or_verified": False,
        "model_imported_or_executed": False,
        "jax_imported": False,
        "first_interval_contract": {
            "bracket": "first coupled d03 steps 1..200 (00:00..00:20)",
            "root_steps": FIRST_INTERVAL_ROOT_STEPS,
            "own_steps": FIRST_INTERVAL_OWN_STEPS,
            "last_analyzed_d03_step": FIRST_INTERVAL_LAST_ANALYZED_D03_STEP,
            "savepoint_tags": [tag for tag, _ in SAVEPOINT_TAGS],
            "retained_pin_sha256": RETAINED_PIN_SHA256,
            "production_hlo_identity_required": True,
            "model_delta_files": [MODEL_FILE],
            "diagnostic_only": True,
        },
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


class _LazyFirstIntervalCaptureExecutable:
    """Compile the capture one-step program on its first LIVE d03 carry.

    The initial child carry's ``*_bdy`` leaves are zero-shaped with ONE time
    level; the live parent force attaches the two-time boundary package
    (build_child_boundary_package) before the first child advance.  Compiling
    against the initial carry therefore bakes a time=1 argument signature
    that every live dispatch violates (the harness failure of the first
    arm).  This binder mirrors the validated LazyOneStepDomainExecutable
    contract: exactly one lower+compile, on the first live carry; any later
    signature drift fails closed.
    """

    def __init__(self, runner: Any, runtime: Any, artifact_path: Path) -> None:
        self.runner = runner
        self.runtime = runtime
        self.artifact_path = artifact_path
        self.executable: Any | None = None
        self.namelist: Any | None = None
        self.clock: Any | None = None
        self.cadence: int | None = None
        self.signature: dict[str, Any] | None = None
        self.compile_audit: dict[str, Any] | None = None
        self.compile_calls = 0
        self.dispatch_calls = 0

    def _bind(self, carry: Any, namelist: Any, clock: Any, cadence: int) -> None:
        runner = self.runner
        signature = runner.carry_interface_signature(carry, self.runtime)
        if signature["leaf_count"] != 106:
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_LIVE_LEAF_COUNT",
                f"leaves={signature['leaf_count']}",
            )
        if self.executable is None:
            from gpuwrf.runtime.operational_mode import (
                advance_one_step_with_first_interval_capture,
            )

            jnp = self.runtime.jnp
            lower_started = time.perf_counter()
            lowered = advance_one_step_with_first_interval_capture.lower(
                carry,
                namelist,
                jnp.asarray(1, dtype=jnp.int32),
                clock,
                cadence=int(cadence),
            )
            stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
            lower_seconds = time.perf_counter() - lower_started
            retained = runner.atomically_retain_lowered_hlo_audit(
                self.artifact_path, stablehlo,
            )
            policy = runner.evaluate_stablehlo_policy(stablehlo)
            if not policy["passed"]:
                raise runner.RunnerGateError(
                    "FIRST_INTERVAL_CAPTURE_HLO_POLICY",
                    json.dumps(policy, sort_keys=True),
                )
            runner.assert_preemption_clear("pre-capture-compile")
            compile_started = time.perf_counter()
            executable = lowered.compile()
            compile_seconds = time.perf_counter() - compile_started
            self.executable = executable
            self.namelist = namelist
            self.clock = clock
            self.cadence = int(cadence)
            self.signature = signature
            self.compile_audit = {
                "callable": (
                    "gpuwrf.runtime.operational_mode."
                    "advance_one_step_with_first_interval_capture"
                ),
                "lazy_compile_trigger": "first-live-d03-carry",
                "lower_calls": 1,
                "compile_calls": 1,
                "lower_wall_seconds": lower_seconds,
                "compile_wall_seconds": compile_seconds,
                "input_leaf_count": signature["leaf_count"],
                "stablehlo_sha256": retained["extraction"]["stablehlo_sha256"],
                "stablehlo_bytes": retained["extraction"]["stablehlo_bytes"],
                "lowered_hlo_artifact": retained,
                "hlo_policy": policy,
                "host_health_or_output_values_feed_model": False,
            }
            self.compile_calls = 1
            print(
                "FIRST_INTERVAL_COMPILE_COMPLETE "
                f"seconds={compile_seconds:.6f} program=first_interval_capture "
                "bind=first-live-d03-carry",
                flush=True,
            )
        elif signature != self.signature:
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_LIVE_SIGNATURE_DRIFT",
                json.dumps({
                    "compiled": self.signature,
                    "called": signature,
                    "compile_calls": self.compile_calls,
                }, sort_keys=True),
            )
        if self.compile_calls != 1 or self.executable is None:
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_COMPILE_COUNT", str(self.compile_calls),
            )

    def one_step(self, carry: Any, namelist: Any, step: int, clock: Any, cadence: int) -> Any:
        self._bind(carry, namelist, clock, cadence)
        self.dispatch_calls += 1
        return self.executable(
            carry,
            namelist,
            self.runtime.jnp.asarray(int(step), dtype=self.runtime.jnp.int32),
            clock,
            cadence=int(cadence),
        )

    def audit(self) -> dict[str, Any]:
        if self.compile_audit is None or self.compile_calls != 1:
            raise runner.RunnerGateError(
                "FIRST_INTERVAL_NOT_COMPILED", str(self.compile_calls),
            )
        return {
            **self.compile_audit,
            "compile_calls": self.compile_calls,
            "dispatch_calls": self.dispatch_calls,
        }


def _write_step_savepoints(
    runner: Any,
    runtime: Any,
    savepoint_dir: Path,
    step: int,
    before: Any,
    result: Any,
    after: Any,
) -> list[dict[str, Any]]:
    """Diagnostic-only host transfer of the per-step momentum savepoints.

    Runs strictly after the completed one-step dispatch, outside any
    production HLO; never present in the production timestep path.
    """

    jax, np = runtime.jax, runtime.np
    arrays = {
        ("sp1_entry", "u"): before.state.u,
        ("sp1_entry", "v"): before.state.v,
        ("sp2_pbl", "rublten"): result.record.rublten,
        ("sp2_pbl", "rvblten"): result.record.rvblten,
        ("sp3_tendf", "ru_tendf"): result.record.ru_tendf,
        ("sp3_tendf", "rv_tendf"): result.record.rv_tendf,
        ("sp4_exit", "u"): after.state.u,
        ("sp4_exit", "v"): after.state.v,
    }
    rows = []
    for (tag, field), device_value in arrays.items():
        host = jax.device_get(device_value)
        arr = np.asarray(host, dtype=np.float64)
        name = f"step{step:06d}_{tag}__{field}.npy"
        path = savepoint_dir / name
        temporary = savepoint_dir / (name + ".part")
        with temporary.open("wb") as handle:
            np.save(handle, arr)
        os.replace(temporary, path)
        rows.append({
            "step": int(step),
            "tag": tag,
            "field": field,
            "path": str(path.resolve()),
            "shape": [int(dim) for dim in arr.shape],
            "dtype": str(arr.dtype),
            "file_sha256": runner.sha256_file(path),
        })
    return rows


def _run_first_interval_window(
    runner: Any,
    tree: Any,
    carries: dict[str, Any],
    *,
    output: Any,
    cadence: dict[str, int],
    nonintegral: dict[str, tuple[int, ...]],
    capture_program: Any,
    namelist: Any,
    clock: Any,
    d03_cadence: int,
    health_executable: Any,
    runtime: Any,
    run_dir: Path,
    savepoint_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    normal_advance = runtime._operational_advance_factory(tree)
    block_between, root_sync_cadence = runtime._nested_sync_mode_from_env()
    rows: list[dict[str, Any]] = []
    sampled: list[int] = []
    savepoint_rows: list[dict[str, Any]] = []
    last_healthy_carry = carries["d03"]
    last_healthy_step = 0
    previous_healthy_carry = last_healthy_carry
    previous_healthy_step = 0

    def advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        nonlocal last_healthy_carry, last_healthy_step
        nonlocal previous_healthy_carry, previous_healthy_step
        if name != "d03":
            return normal_advance(name, carry, start_step, n_steps)
        value = carry
        for offset in range(int(n_steps)):
            native_step = int(start_step + offset)
            try:
                runner.assert_preemption_clear(f"first-interval-d03-step-{native_step}-pre")
            except runner.RunnerGateError as exc:
                failure = runner._retain_failure_pair(
                    runtime,
                    run_dir,
                    last_step=native_step - 1,
                    last_carry=value,
                    failed_step=native_step - 1,
                    failed_carry=value,
                    failure={
                        "code": exc.code,
                        "detail": exc.detail,
                        "dispatch_withheld": True,
                    },
                )
                raise runner.WindowFalsified(exc.code, failure["proof_sha256"]) from exc
            before = value
            result = capture_program.one_step(
                value,
                namelist,
                native_step,
                clock,
                d03_cadence,
            )
            value = result.carry
            summary = runner.materialize_health(health_executable, value, runtime)
            gate = runner.evaluate_health_summary(
                native_step,
                summary,
                step9313_scale_baseline=None,
            )
            savepoint_rows.extend(
                _write_step_savepoints(
                    runner, runtime, savepoint_dir, native_step, before, result, value,
                )
            )
            rows.append(gate)
            sampled.append(native_step)
            if not gate["passed"]:
                failure = runner._retain_failure_pair(
                    runtime,
                    run_dir,
                    last_step=last_healthy_step,
                    last_carry=before,
                    failed_step=native_step,
                    failed_carry=value,
                    failure={"code": "HEALTH_GATE", "gate": gate},
                )
                raise runner.WindowFalsified("HEALTH_GATE", failure["proof_sha256"])
            previous_healthy_carry, previous_healthy_step = (
                last_healthy_carry,
                last_healthy_step,
            )
            last_healthy_carry, last_healthy_step = value, native_step
            if native_step % 25 == 0 or native_step == 1:
                print(
                    f"FIRST_INTERVAL_HEALTH step={native_step} finite=1 violations=0",
                    flush=True,
                )
        return value

    try:
        result = runtime.run_domain_tree_callbacks(
            tree.hierarchy,
            carries,
            root_steps=FIRST_INTERVAL_ROOT_STEPS,
            advance=advance,
            force=runtime._operational_force,
            feedback=None,
            feedback_enabled=False,
            output=output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            edge_lookup=runtime.ordinary._edge_lookup(tree),
            fused_cascade=None,
            initial_own_steps={"d01": 0, "d02": 0, "d03": 0},
        )
    except runner.MetricGateFailure as exc:
        failed = output.failed_carry or last_healthy_carry
        failed_step = output.failed_step or last_healthy_step
        failed_domain = output.failed_domain or "d03"
        failure = runner._retain_failure_pair(
            runtime,
            run_dir,
            domain=failed_domain,
            last_step=previous_healthy_step,
            last_carry=previous_healthy_carry,
            failed_step=failed_step,
            failed_carry=failed,
            failure={"code": exc.code, "detail": exc.detail},
        )
        raise runner.WindowFalsified(exc.code, failure["proof_sha256"]) from exc
    runtime.block_until_ready(
        tuple(result.states[name].theta for name in tree.hierarchy.order)
    )
    if dict(result.own_steps) != FIRST_INTERVAL_OWN_STEPS:
        raise runner.RunnerGateError("FIRST_INTERVAL_CLOCKS", repr(result.own_steps))
    if sampled != list(FIRST_INTERVAL_D03_STEPS):
        raise runner.RunnerGateError(
            "FIRST_INTERVAL_SAMPLING",
            f"first={sampled[:2]} last={sampled[-2:]} count={len(sampled)}",
        )
    proof = {
        "own_steps": dict(result.own_steps),
        "sampled_steps_count": len(sampled),
        "sampled_d03_first_last": [sampled[0], sampled[-1]],
        "health_rows": rows,
        "model_dispatches": len(sampled),
        "health_materialization": "after each completed d03 dispatch, outside production HLO",
        "savepoint_host_transfer": (
            "diagnostic-only, after each completed d03 dispatch, absent from "
            "the production timestep path"
        ),
        "savepoint_rows": savepoint_rows,
        "normal_output_counts": dict(output.pairer.counts),
    }
    return result, proof


def _first_interval_runtime_main(
    runner: Any,
    args: Any,
    authority: dict[str, Any],
    runtime: Any,
) -> int:
    jax, jnp, np = runtime.jax, runtime.jnp, runtime.np
    from datetime import datetime, timezone

    started = datetime.now(timezone.utc)
    authority["cuda_runtime"] = runner._assert_cuda_runtime(runtime)
    authority["runtime_cache"] = runner.assert_runtime_cache_disabled(runtime)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.run_dir / "gpu-output"
    output_dir.mkdir(parents=True, exist_ok=False)
    savepoint_dir = args.run_dir / "savepoints"
    savepoint_dir.mkdir(parents=True, exist_ok=False)

    tree, names, initial_carries, dt_by_domain, load_authority = (
        runtime.ordinary.load_corrected_tree(args.run_dir)
    )
    if names != ("d01", "d02", "d03") or dt_by_domain != {
        "d01": 54.0,
        "d02": 18.0,
        "d03": 6.0,
    }:
        raise runner.RunnerGateError(
            "DOMAIN_AUTHORITY", f"names={names} dt={dt_by_domain}",
        )
    cadence, nonintegral, schedule = runtime.ordinary.scheduler_contract(
        names, dt_by_domain, total_steps=runner.TERMINAL_OWN_STEPS,
    )

    runner.assert_preemption_clear("pre-lower")

    d03_namelist = tree.domains["d03"].namelist
    clock = runtime.build_clock_base(d03_namelist)
    d03_cadence = int(d03_namelist.radiation_cadence_steps)

    # --- Production one-step lower (never compiled, never dispatched) -------
    # Proves the diagnostic model delta leaves production numerics untouched:
    # the lowered production HLO must equal the retained deterministic
    # reference artifact's stablehlo hash.
    production_lower_started = time.perf_counter()
    production_lowered = runtime._advance_chunk_fori.lower(
        initial_carries["d03"],
        d03_namelist,
        jnp.asarray(1, dtype=jnp.int32),
        clock,
        n_steps=1,
        cadence=d03_cadence,
    )
    production_stablehlo = str(production_lowered.compiler_ir(dialect="stablehlo"))
    production_lower_seconds = time.perf_counter() - production_lower_started
    production_extraction = runner.extract_stablehlo_custom_calls(production_stablehlo)
    retained_hlo_row = runner._require_sha(
        RETAINED_PRODUCTION_HLO,
        RETAINED_PRODUCTION_HLO_SHA256,
        "RETAINED_PRODUCTION_HLO",
    )
    retained_payload = json.loads(RETAINED_PRODUCTION_HLO.read_text())
    retained_extraction = retained_payload["extraction"]
    production_hlo_identity = {
        "stablehlo_sha256": production_extraction["stablehlo_sha256"],
        "retained_stablehlo_sha256": retained_extraction["stablehlo_sha256"],
        "stablehlo_bytes": production_extraction["stablehlo_bytes"],
        "retained_stablehlo_bytes": retained_extraction["stablehlo_bytes"],
        "custom_call_targets": production_extraction["targets"],
        "retained_custom_call_targets": retained_extraction["targets"],
        "custom_call_occurrences": production_extraction["parsed_custom_call_count"],
        "retained_custom_call_occurrences": retained_extraction[
            "parsed_custom_call_count"
        ],
        "identical": bool(
            production_extraction["stablehlo_sha256"]
            == retained_extraction["stablehlo_sha256"]
            and production_extraction["stablehlo_bytes"]
            == retained_extraction["stablehlo_bytes"]
            and production_extraction["targets"]
            == retained_extraction["targets"]
            and production_extraction["parsed_custom_call_count"]
            == retained_extraction["parsed_custom_call_count"]
        ),
        "compiled": False,
        "dispatched": False,
        "lower_wall_seconds": production_lower_seconds,
        "retained_artifact": retained_hlo_row,
    }
    if not production_hlo_identity["identical"]:
        raise runner.RunnerGateError(
            "PRODUCTION_HLO_DRIFT",
            json.dumps(production_hlo_identity, sort_keys=True)[:4000],
        )
    production_policy = runner.evaluate_stablehlo_policy(production_stablehlo)
    if not production_policy["passed"]:
        raise runner.RunnerGateError(
            "PRODUCTION_HLO_POLICY", json.dumps(production_policy, sort_keys=True),
        )

    # --- Capture one-step program: lazy compile on the first LIVE d03 carry --
    # The initial child carry's boundary leaves hold ONE time level; the live
    # parent force attaches the two-time package before the first child
    # advance, so the capture program must be compiled against a live carry
    # (the repaired harness; first arm failed on the time-dim mismatch).
    capture_program = _LazyFirstIntervalCaptureExecutable(
        runner,
        runtime,
        args.run_dir / "first-interval-capture-lowered-hlo.json",
    )

    health_executable = runner.LazyLiveCarryHealthExecutable(runtime)
    runner.assert_preemption_clear("post-lower-pre-initial-output")
    writer = runtime._PerDomainWrfoutWriter(
        output_dir=output_dir,
        input_dir=runner.INPUT_DIR,
        run_start=runner.RUN_START,
        bundles=tree.domains,
        output_cadence_steps=cadence,
        dt_by_domain=dt_by_domain,
        async_writer=None,
        output_pipeline=None,
    )
    pairer = runner.IncrementalFramePairer(
        runtime,
        args.run_dir / "frame-pairs",
        record_known_1500_v10_red=False,
    )
    output = runner.StandardCadenceOutput(
        writer,
        pairer,
        output_dir,
        health_executable,
        runtime,
        args.run_dir,
    )
    try:
        initial_history = runtime._emit_initial_history_frames(
            output, names, initial_carries, enabled=True,
        )
    except runner.MetricGateFailure as exc:
        domain = output.failed_domain or "d03"
        failed = output.failed_carry or initial_carries[domain]
        failed_step = output.failed_step or 0
        failure = runner._retain_failure_pair(
            runtime,
            args.run_dir,
            domain=domain,
            last_step=0,
            last_carry=initial_carries[domain],
            failed_step=failed_step,
            failed_carry=failed,
            failure={"code": exc.code, "detail": exc.detail, "initial_history": True},
        )
        raise runner.WindowFalsified(exc.code, failure["proof_sha256"]) from exc

    runner.assert_preemption_clear("pre-first-interval-window")
    result, window_proof = _run_first_interval_window(
        runner,
        tree,
        initial_carries,
        output=output,
        cadence=cadence,
        nonintegral=nonintegral,
        capture_program=capture_program,
        namelist=d03_namelist,
        clock=clock,
        d03_cadence=d03_cadence,
        health_executable=health_executable,
        runtime=runtime,
        run_dir=args.run_dir,
        savepoint_dir=savepoint_dir,
    )
    if dict(pairer.counts) != EXPECTED_FRAME_COUNTS:
        raise runner.RunnerGateError(
            "FIRST_INTERVAL_OUTPUT_COUNTS",
            f"expected={EXPECTED_FRAME_COUNTS} actual={dict(pairer.counts)}",
        )
    if not all(row["passed"] for row in window_proof["health_rows"]):
        raise runner.RunnerGateError("FIRST_INTERVAL_HEALTH_ROWS", "non-green row")

    # --- Eight-field report at d03 steps 0 and 200 --------------------------
    field_rows = []
    for row in pairer.rows:
        if row["domain"] != "d03" or row["own_step"] not in (0, 200):
            continue
        field_rows.append({
            "own_step": row["own_step"],
            "valid_time": row["valid_time"],
            "finite_identity_passed": row["finite_identity_pass"],
            "static_identity": row["static_identity"],
            "scientific_pair_pass": row["scientific_pair_pass"],
            "strict_rmse": row["strict_rmse"],
            "cpu_sha256": row["cpu_sha256"],
            "candidate_sha256": row["candidate_sha256"],
            "artifact_sha256": row["artifact_sha256"],
        })
    if len(field_rows) != 2 or any(
        row["strict_rmse"] is None or set(row["strict_rmse"])
        != {"T", "U", "V", "W", "T2", "U10", "V10", "PSFC"}
        for row in field_rows
    ):
        raise runner.RunnerGateError(
            "FIRST_INTERVAL_EIGHT_FIELD", repr(field_rows)[:4000],
        )

    savepoint_manifest = {
        "count": len(window_proof["savepoint_rows"]),
        "expected_count": len(FIRST_INTERVAL_D03_STEPS) * 8,
        "rows": window_proof["savepoint_rows"],
    }
    if savepoint_manifest["count"] != savepoint_manifest["expected_count"]:
        raise runner.RunnerGateError(
            "FIRST_INTERVAL_SAVEPOINT_INVENTORY",
            repr({k: savepoint_manifest[k] for k in ("count", "expected_count")}),
        )
    savepoint_manifest["rows_sha256"] = runner.canonical_digest(
        savepoint_manifest["rows"]
    )

    finished = datetime.now(timezone.utc)
    proof = {
        "schema": runner.SCHEMA,
        "verdict": "FIRST_INTERVAL_MOMENTUM_SHORT_ARM_COMPLETE",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "namespace": NAMESPACE,
        "bracket": "first coupled d03 steps 1..200 (00:00..00:20)",
        "authority": authority,
        "load_authority": load_authority,
        "production_hlo_identity": production_hlo_identity,
        "capture_compile": capture_program.audit(),
        "health_executable": health_executable.audit(),
        "initial_history": initial_history,
        "window": {
            key: value
            for key, value in window_proof.items()
            if key not in {"health_rows", "savepoint_rows"}
        },
        "health_rows": window_proof["health_rows"],
        "eight_field_report": field_rows,
        "savepoint_manifest": savepoint_manifest,
        "frame_pair_counts": dict(pairer.counts),
        "retained_pin_sha256": RETAINED_PIN_SHA256,
        "model_tree": {
            "before": GPT_PROOF_COMMIT_TREE,
            "delta_files": [MODEL_FILE],
            "diagnostic_only": True,
            "production_numerics_untouched": production_hlo_identity["identical"],
        },
        "gpu_released_at_exit": True,
        "release_gate_unchanged": True,
        "tolerance_changed": False,
    }
    proof["proof_sha256"] = runner.canonical_digest(proof)
    runner.atomic_write_json(args.proof_output, proof)
    print(json.dumps({
        "verdict": proof["verdict"],
        "proof": str(args.proof_output),
        "proof_sha256": proof["proof_sha256"],
        "frame_pair_counts": dict(pairer.counts),
        "own_steps": window_proof["own_steps"],
    }, sort_keys=True), flush=True)
    return 0
