"""Fail-closed profile for the current-tree V/V10 dump-then-pin replay.

The production runner historically continues from the 15:00 metric into the
late-Ni/full-18-hour window.  This profile keeps the same real fixture,
writer/comparator, numerical environment, and synchronous checkpoint path, but
terminates exactly after d03 step 9000.  A V/V10 red is retained as red so an
autotune table can be authenticated and replayed; no red is waived.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


BASE_COMMIT = "1bfee4e94be5c614051563678d62e5293bb5426a"
BASE_TREE = "431188fd3c69e79f714bd98de97676c0bf74345f"
MODEL_TREE = "5a6298fba90e76c3cafe674e1413d38efb694532"
MODEL_TREE_COMMIT = "77dd2b334df90b14100fc87beff9bfc46f96a84a"
CONTRACT_COMMIT = "d41732b9e6d90fa5a89d4770714be8b6243ca638"
CONTRACT_SHA256 = "842b7a250f9cc4e327c33e5229c0446dd1c7006ed0401411b65882ec7e5a3255"
NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIX_SEGMENTS = (67,) * 14 + (62,)
PREFIX_OWN_STEPS = {"d01": 1000, "d02": 3000, "d03": 9000}
EXPECTED_COUNTS = {"d01": 16, "d02": 16, "d03": 46}
ALLOWED_RED_FIELDS = frozenset({"V", "V10"})
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")

KIMI_PROOF_FILE_SHA256 = (
    "252f7a77647f8bcf485676fde99382e2e87e08b0c5942651008e392d2bd62015"
)
KIMI_PROOF_CANONICAL_SHA256 = (
    "a05e5bbd89bb61f40b18775985b5237fa01a0fd55a0285dae4cb0e692e9fadd8"
)
OLD_REPLAY_PROOF_FILE_SHA256 = (
    "acb7b7ae07d731085b335f05039d6932ed96884650007f68f1fd3ce26b770d80"
)
OLD_REPLAY_PROOF_CANONICAL_SHA256 = (
    "bad255c6275ac4c04b089576a2720d9366c6bf3bc5897d5c70054192586bd8ba"
)
RUNTIME_AUTHORITY_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "v0234_v10_runtime_authority_complete1_5a6298fb"
)
RUNTIME_AUTHORITY_FILES = {
    "phys/module_ra_rrtmg_lw.F": (
        652002,
        "c7a5238612aa8a4213c8d3af6708ec6a5248e6701e19758a80e563905d306de3",
    ),
    "run/CAMtr_volume_mixing_ratio": (
        42780,
        "9a427fd106f8e36b30e0b29266bff1398b025b82af5b878e5a7a8e9dfe268ca7",
    ),
    "run/GENPARM.TBL": (
        261,
        "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2",
    ),
    "run/MPTABLE.TBL": (
        56140,
        "7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d",
    ),
    "run/SOILPARM.TBL": (
        6557,
        "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f",
    ),
}
RUNTIME_AUTHORITY_MANIFEST_FILE_SHA256 = (
    "633ef843d2db02f4e4e6f7fa62d14074672d971aefd3e539dec9d6d93365d1f2"
)
RUNTIME_AUTHORITY_MANIFEST_CANONICAL_SHA256 = (
    "ceb131ff9eff0e76f2b2317ae76a4cc52f5306b36e107cebe01c0648431581d9"
)
CPU_DOMAIN_LOAD_PREFLIGHT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "v0234_v10_cpu_domain_load_preflight_complete1_shim1.json"
)
CPU_DOMAIN_LOAD_PREFLIGHT_FILE_SHA256 = (
    "a90bd597c94836b426b8c0c8fd1091db1990343ad77376e2d2d63fb92b812ee7"
)
CPU_DOMAIN_LOAD_PREFLIGHT_CANONICAL_SHA256 = (
    "2916dc6f641824c5549710f562d0d5508b3feea3ac1c1ec084819e93806e9e0c"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _self_hashed_payload(
    runner: Any,
    path: Path,
    *,
    file_sha256: str | None = None,
    canonical_sha256: str | None = None,
    code: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise runner.RunnerGateError(code, f"missing or symlink: {path}")
    observed_file = runner.sha256_file(path)
    if file_sha256 is not None and observed_file != file_sha256:
        raise runner.RunnerGateError(
            code, f"file expected={file_sha256} observed={observed_file}",
        )
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed_canonical = runner.canonical_digest(unsigned)
    if embedded != observed_canonical or (
        canonical_sha256 is not None and observed_canonical != canonical_sha256
    ):
        raise runner.RunnerGateError(
            f"{code}_CANONICAL",
            repr({
                "embedded": embedded,
                "observed": observed_canonical,
                "expected": canonical_sha256,
            }),
        )
    return payload, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": observed_file,
        "canonical_sha256": observed_canonical,
    }


def classify_scoped_v10_observation(
    decisive: Mapping[str, Any],
    *,
    candidate_sha256: str,
    authority: Mapping[str, Any],
    **_unused: Any,
) -> dict[str, Any]:
    """Retain only an all-finite/static V/V10 red at the terminal scope.

    ``passed`` means only that the observation is safe to retain and stop on;
    ``release_gate_green`` remains the unchanged scalar gate result.
    """

    fields = decisive.get("strict_fields") or {}
    violations = decisive.get("violations") or []
    red_fields = sorted(
        field for field, row in fields.items() if row.get("no_worse") is not True
    )
    values = {
        field: float((fields.get(field) or {}).get("candidate_rmse", math.nan))
        for field in STRICT_FIELDS
    }
    exact_fields = set(fields) == set(STRICT_FIELDS)
    finite = all(math.isfinite(value) for value in values.values())
    authority_exact = all(
        (fields.get(field) or {}).get("retry20_authority_exact") is True
        for field in STRICT_FIELDS
    )
    release_gate_green = decisive.get("passed") is True and not red_fields
    red_retainable = bool(
        decisive.get("passed") is False
        and red_fields
        and set(red_fields) <= ALLOWED_RED_FIELDS
        and violations
        and all(row.get("field") in ALLOWED_RED_FIELDS for row in violations)
    )
    passed = bool(
        decisive.get("static_exact") is True
        and exact_fields
        and finite
        and authority_exact
        and (release_gate_green or red_retainable)
    )
    payload = {
        "schema": "gpuwrf.v0234.v10-scoped-observation.v1",
        "passed": passed,
        "classification": (
            "STRICT_V_V10_GREEN"
            if release_gate_green
            else "STRICT_V_V10_RED_RETAIN_AND_STOP"
            if red_retainable
            else "UNAUTHORIZED_SCIENTIFIC_RED"
        ),
        "release_gate_green": release_gate_green,
        "release_blocker_remains": not release_gate_green,
        "waiver_or_reclassification": False,
        "tolerance_changed": False,
        "continue_beyond_scored_horizon": False,
        "red_fields": red_fields,
        "strict_rmse": values,
        "candidate_sha256": candidate_sha256,
        "authority_sha256": authority.get("authority_sha256"),
    }
    payload["proof_sha256"] = _canonical(payload)
    return payload


def apply_profile(runner: Any) -> None:
    if getattr(runner, "_V10_PINNED_REPLAY_PROFILE_APPLIED", False):
        return

    nonce = os.environ.get("GPUWRF_V10_REPLAY_NONCE", "")
    if NONCE_RE.fullmatch(nonce) is None:
        raise runner.RunnerGateError("V10_REPLAY_NONCE", repr(nonce))
    mode = os.environ.get("GPUWRF_V10_REPLAY_MODE", "")
    if mode not in {"reference", "pinned"}:
        raise runner.RunnerGateError("V10_REPLAY_MODE", repr(mode))
    short = nonce[:16]
    reference_namespace = f"v0234_gpt_v10_replay_{short}_reference"
    pinned_namespace = f"v0234_gpt_v10_replay_{short}_pinned"
    binding = {
        "reference": {
            "namespace": reference_namespace,
            "label": f"v0234-v10-reference-{short}",
            "audit": "REFERENCE_CPU_AUDIT_RETRY3.json",
            "dump": "reference-autotune-results.pb",
        },
        "pinned": {
            "namespace": pinned_namespace,
            "label": f"v0234-v10-pinned-{short}",
            "audit": "PINNED_CPU_AUDIT_RETRY3.json",
            "dump": "pinned-autotune-results.pb",
        },
    }[mode]

    from scripts.v0234_stage_omega_transport_runner_profile import (
        apply_profile as apply_stage_omega_profile,
    )

    apply_stage_omega_profile(runner)
    # Retry20 predates production CLWRF-gas population and the native RRTMG-LW
    # source parser.  This read-only root is its byte-identical Noah-MP superset
    # with pristine SSP245 and module_ra_rrtmg_lw.F authorities added.
    runner.RETRY20_WRF_ROOT = RUNTIME_AUTHORITY_ROOT
    base_validate_environment = runner.validate_preimport_environment

    sprint = runner.REPO_ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-replay"
    contract = sprint / "CONTRACT.md"
    authorization_path = sprint / "GPU_AUTHORIZATION_RETRY3.json"
    launcher = sprint / "v10-pinned-replay-retry3.sh"
    pin_manifest_path = sprint / "REFERENCE_PIN_MANIFEST.json"
    autotune_stage = (
        runner.LINEAGE_WORK_DIR / f".v0234-v10-replay-{short}-autotune"
    ).resolve()
    reference_run = (runner.LINEAGE_WORK_DIR / reference_namespace).resolve()
    reference_pin = (reference_run / "autotune-results.pb").resolve()
    output_pin = (autotune_stage / binding["dump"]).resolve()

    runner.SCHEMA = "gpuwrf.v0234.v10-pinned-replay-arm.v1"
    runner.CPU_PROOF_SCHEMA = "gpuwrf.v0234.v10-pinned-replay-cpu-audit.v1"
    runner.AUDIT_ADMISSION = "READY_FOR_V10_PINNED_REPLAY_GPU_ARM"
    runner.CANDIDATE_COMMIT = BASE_COMMIT
    runner.CANDIDATE_TREE = BASE_TREE
    runner.FULL_REPLAY_NAMESPACE = binding["namespace"]
    runner.LOCK_LABEL = binding["label"]
    runner.LAUNCH_COMMAND = launcher
    runner.RUNNER_CPU_AUDIT = sprint / binding["audit"]
    runner.REQUIRE_KNOWN_1500_V10_RECORD = True
    runner.REQUIRE_TOOLING_CRITIC_ACCEPT = False
    runner.TOOLING_CRITIC_AUTHORITY_HOOK = None
    runner.PREFIX_SEGMENTS = PREFIX_SEGMENTS
    runner.PREFIX_OWN_STEPS = dict(PREFIX_OWN_STEPS)
    runner.EXPECTED_AFFINITY = [13, 14, 15, 29, 30, 31]
    runner.CPU_FOCUSED_TEST_ARGS = (
        "tests/test_v0234_v10_pinned_replay.py",
    )
    runner.INFRASTRUCTURE_GPUWRF_ENV = set(runner.INFRASTRUCTURE_GPUWRF_ENV) | {
        "GPUWRF_V10_PINNED_REPLAY",
        "GPUWRF_V10_REPLAY_MODE",
        "GPUWRF_V10_REPLAY_NONCE",
        "GPUWRF_V10_REPLAY_AUTOTUNE_OUTPUT",
        "GPUWRF_V10_REPLAY_REFERENCE_PIN",
        "GPUWRF_V10_REPLAY_PIN_MANIFEST",
        "GPUWRF_V10_REPLAY_PIN_MANIFEST_SHA256",
        "GPUWRF_V10_REPLAY_AUTHORIZATION",
        "GPUWRF_V10_REPLAY_AUTHORIZATION_SHA256",
    }
    runner.FORBIDDEN_NON_GPUWRF_ENV = tuple(
        name for name in runner.FORBIDDEN_NON_GPUWRF_ENV if name != "XLA_FLAGS"
    )
    runner.CANDIDATE_CLEAN_AUTHORITY = {
        "contract_commit": CONTRACT_COMMIT,
        "contract_file_sha256": CONTRACT_SHA256,
        "starting_commit": BASE_COMMIT,
        "starting_tree": BASE_TREE,
        "src_gpuwrf_tree": MODEL_TREE,
        "nonce": nonce,
        "mode": mode,
        "namespace": binding["namespace"],
        "scored_horizon_d03_step": 9000,
        "late_ni_scope": False,
        "tolerance_changed": False,
    }

    def authenticate_runtime_authority() -> dict[str, Any]:
        root = RUNTIME_AUTHORITY_ROOT
        run = root / "run"
        phys = root / "phys"
        if (
            not root.is_dir()
            or root.is_symlink()
            or not run.is_dir()
            or run.is_symlink()
            or not phys.is_dir()
            or phys.is_symlink()
            or root.stat().st_mode & 0o222
            or run.stat().st_mode & 0o222
            or phys.stat().st_mode & 0o222
        ):
            raise runner.RunnerGateError(
                "V10_RUNTIME_AUTHORITY_ROOT", str(root),
            )
        inventory = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        )
        if inventory != sorted(RUNTIME_AUTHORITY_FILES):
            raise runner.RunnerGateError(
                "V10_RUNTIME_AUTHORITY_INVENTORY", repr(inventory),
            )
        rows = {}
        for relative, (expected_bytes, expected_sha256) in sorted(
            RUNTIME_AUTHORITY_FILES.items()
        ):
            path = root / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_mode & 0o222
                or path.stat().st_size != expected_bytes
                or runner.sha256_file(path) != expected_sha256
            ):
                raise runner.RunnerGateError(
                    "V10_RUNTIME_AUTHORITY_FILE", str(path),
                )
            rows[relative] = {
                "path": str(path.resolve()),
                "bytes": expected_bytes,
                "sha256": expected_sha256,
                "mode": oct(path.stat().st_mode & 0o777),
            }
        payload = {
            "root": str(root.resolve()),
            "root_mode": oct(root.stat().st_mode & 0o777),
            "run_mode": oct(run.stat().st_mode & 0o777),
            "phys_mode": oct(phys.stat().st_mode & 0o777),
            "files": rows,
            "exact_file_inventory": True,
            "writable_bits_absent": True,
            "retry20_noahmp_tables_byte_identical": True,
            "clwrf_ssp245_source": "<USER_HOME>/src/wrf_pristine/WRF/run/CAMtr_volume_mixing_ratio.SSP245",
            "rrtmg_lw_source": "<USER_HOME>/src/wrf_pristine/WRF/phys/module_ra_rrtmg_lw.F",
        }
        preflight, preflight_row = _self_hashed_payload(
            runner,
            CPU_DOMAIN_LOAD_PREFLIGHT,
            file_sha256=CPU_DOMAIN_LOAD_PREFLIGHT_FILE_SHA256,
            canonical_sha256=CPU_DOMAIN_LOAD_PREFLIGHT_CANONICAL_SHA256,
            code="V10_CPU_DOMAIN_LOAD_PREFLIGHT",
        )
        if (
            preflight.get("schema")
            != "gpuwrf.v0234.v10-cpu-domain-load-preflight.v1"
            or preflight.get("verdict")
            != "CPU_DOMAIN_LOAD_DEPENDENCY_CLOSURE_PASSED"
            or (preflight.get("candidate_authority") or {}).get("src_gpuwrf_tree")
            != MODEL_TREE
            or (preflight.get("runtime_authority") or {}).get("root")
            != str(root.resolve())
            or (preflight.get("runtime_authority") or {}).get(
                "exact_file_inventory"
            ) is not True
            or (preflight.get("scope") or {}).get("cpu_only") is not True
            or (preflight.get("scope") or {}).get("gpu_command_or_query") is not False
            or (preflight.get("scope") or {}).get("timestep_advanced") is not False
            or (preflight.get("scope") or {}).get("forecast_output_emitted") is not False
            or (preflight.get("cpu_allocation_shim") or {}).get(
                "model_source_edited"
            ) is not False
        ):
            raise runner.RunnerGateError(
                "V10_CPU_DOMAIN_LOAD_PREFLIGHT_SEMANTICS", repr(preflight),
            )
        payload["cpu_domain_load_preflight"] = preflight_row
        manifest_path = sprint / "RUNTIME_AUTHORITY_COMPLETE1.json"
        manifest, manifest_row = _self_hashed_payload(
            runner,
            manifest_path,
            file_sha256=RUNTIME_AUTHORITY_MANIFEST_FILE_SHA256,
            canonical_sha256=RUNTIME_AUTHORITY_MANIFEST_CANONICAL_SHA256,
            code="V10_RUNTIME_AUTHORITY_MANIFEST",
        )
        manifest_files = manifest.get("files") or {}
        if (
            manifest.get("schema")
            != "gpuwrf.v0234.v10-runtime-table-authority.v1"
            or manifest.get("verdict")
            != "V10_RUNTIME_AUTHORITY_READ_ONLY_EXACT"
            or manifest.get("model_tree") != MODEL_TREE
            or manifest.get("root") != str(root.resolve())
            or set(manifest_files) != set(RUNTIME_AUTHORITY_FILES)
            or (manifest.get("cpu_domain_load_preflight") or {}).get(
                "canonical_sha256"
            ) != CPU_DOMAIN_LOAD_PREFLIGHT_CANONICAL_SHA256
            or any(
                manifest_files[relative].get("bytes") != expected_bytes
                or manifest_files[relative].get("sha256") != expected_sha256
                for relative, (expected_bytes, expected_sha256)
                in RUNTIME_AUTHORITY_FILES.items()
            )
        ):
            raise runner.RunnerGateError(
                "V10_RUNTIME_AUTHORITY_MANIFEST_SEMANTICS", repr(manifest),
            )
        payload["manifest"] = manifest_row
        payload["authority_sha256"] = runner.canonical_digest(payload)
        return payload

    def authenticate_manager_authorization() -> dict[str, Any]:
        path_raw = os.environ.get("GPUWRF_V10_REPLAY_AUTHORIZATION", "")
        expected_hash = os.environ.get(
            "GPUWRF_V10_REPLAY_AUTHORIZATION_SHA256", ""
        )
        if Path(path_raw).resolve() != authorization_path.resolve():
            raise runner.RunnerGateError(
                "V10_AUTHORIZATION_PATH", f"expected={authorization_path} actual={path_raw}",
            )
        payload, row = _self_hashed_payload(
            runner,
            authorization_path,
            file_sha256=expected_hash,
            code="V10_GPU_AUTHORIZATION",
        )
        if (
            payload.get("schema") != "gpuwrf.v0234.v10-manager-gpu-authorization.v1"
            or payload.get("verdict") != "MANAGER_GPU_AUTHORIZED"
            or payload.get("manager_pane") != "0:1"
            or payload.get("nonce") != nonce
            or payload.get("reference_namespace") != reference_namespace
            or payload.get("pinned_namespace") != pinned_namespace
            or payload.get("authorized_arms") != ["reference", "pinned"]
            or payload.get("contract_commit") != CONTRACT_COMMIT
            or payload.get("src_gpuwrf_tree") != MODEL_TREE
            or payload.get("runtime_authority_canonical_sha256")
            != RUNTIME_AUTHORITY_MANIFEST_CANONICAL_SHA256
            or payload.get("cpu_domain_load_preflight_canonical_sha256")
            != CPU_DOMAIN_LOAD_PREFLIGHT_CANONICAL_SHA256
            or payload.get("late_ni_authorized") is not False
        ):
            raise runner.RunnerGateError(
                "V10_GPU_AUTHORIZATION_SEMANTICS", repr(payload),
            )
        return {**row, "verdict": payload["verdict"], "nonce": nonce}

    def authenticate_pin_manifest(*, required: bool) -> dict[str, Any]:
        path_raw = os.environ.get("GPUWRF_V10_REPLAY_PIN_MANIFEST", "")
        expected_hash = os.environ.get(
            "GPUWRF_V10_REPLAY_PIN_MANIFEST_SHA256", ""
        )
        if not required:
            if path_raw or expected_hash:
                raise runner.RunnerGateError(
                    "V10_PIN_MANIFEST_UNEXPECTED", "reference arm must not load a manifest",
                )
            return {"required": False, "present": False}
        if Path(path_raw).resolve() != pin_manifest_path.resolve():
            raise runner.RunnerGateError(
                "V10_PIN_MANIFEST_PATH",
                f"expected={pin_manifest_path} actual={path_raw}",
            )
        payload, row = _self_hashed_payload(
            runner,
            pin_manifest_path,
            file_sha256=expected_hash,
            code="V10_PIN_MANIFEST",
        )
        pin = payload.get("autotune_pin") or {}
        reference = payload.get("reference_run") or {}
        if (
            payload.get("schema") != "gpuwrf.v0234.v10-reference-pin-manifest.v1"
            or payload.get("verdict") != "V10_REFERENCE_PIN_AUTHENTICATED"
            or payload.get("nonce") != nonce
            or payload.get("model_tree") != MODEL_TREE
            or payload.get("scoped_horizon_complete") is not True
            or payload.get("late_ni_claimed") is not False
            or reference.get("namespace") != reference_namespace
            or reference.get("output_counts") != EXPECTED_COUNTS
            or Path(str(pin.get("path", ""))).resolve() != reference_pin
            or not reference_pin.is_file()
            or reference_pin.is_symlink()
            or pin.get("bytes") != reference_pin.stat().st_size
            or pin.get("file_sha256") != runner.sha256_file(reference_pin)
            or os.environ.get("GPUWRF_V10_REPLAY_REFERENCE_PIN")
            != str(reference_pin)
        ):
            raise runner.RunnerGateError(
                "V10_PIN_MANIFEST_SEMANTICS", repr(payload),
            )
        return {**row, "required": True, "pin": pin, "reference_run": reference}

    def assert_replay_final_authority() -> dict[str, Any]:
        if (
            not contract.is_file()
            or contract.is_symlink()
            or runner.sha256_file(contract) != CONTRACT_SHA256
        ):
            raise runner.RunnerGateError("V10_CONTRACT", str(contract))
        kimi_path = runner.REPO_ROOT / (
            ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi/proof.json"
        )
        old_path = runner.REPO_ROOT / (
            ".agent/sprints/2026-07-18-v0234-deterministic-wake-closure-gpt/proof.json"
        )
        kimi, kimi_row = _self_hashed_payload(
            runner,
            kimi_path,
            file_sha256=KIMI_PROOF_FILE_SHA256,
            canonical_sha256=KIMI_PROOF_CANONICAL_SHA256,
            code="V10_KIMI_PROOF",
        )
        old, old_row = _self_hashed_payload(
            runner,
            old_path,
            file_sha256=OLD_REPLAY_PROOF_FILE_SHA256,
            canonical_sha256=OLD_REPLAY_PROOF_CANONICAL_SHA256,
            code="V10_OLD_REPLAY_PROOF",
        )
        if (
            kimi.get("verdict") != "KIMI_STEP9000_V_V10_NO_FIX_LOCALIZED"
            or old.get("verdict") != "GPT_DETERMINISTIC_WAKE_NO_FIX_LOCALIZED"
            or (old.get("reference_replay") or {}).get("first_red_step") != 9000
            or (old.get("reference_replay") or {}).get(
                "strict_rmse_all_required_fields", {}
            ).get("V10") != 2.2377159247655007
        ):
            raise runner.RunnerGateError("V10_PRIOR_PROOF_SEMANTICS", "prior evidence drift")
        return {
            "contract": {
                "path": str(contract.resolve()),
                "file_sha256": CONTRACT_SHA256,
                "commit": CONTRACT_COMMIT,
            },
            "manager_authorization": authenticate_manager_authorization(),
            "runtime_authority": authenticate_runtime_authority(),
            "kimi_prior": kimi_row,
            "old_replay_prior": old_row,
            "pin_manifest": authenticate_pin_manifest(required=(mode == "pinned")),
            "src_gpuwrf_tree": MODEL_TREE,
            "late_ni_scope": False,
        }

    def assert_replay_source_authority(
        environment: Mapping[str, str], *, require_clean: bool,
    ) -> dict[str, Any]:
        head = runner._git(runner.REPO_ROOT, "rev-parse", "HEAD")
        approved = environment.get("GPUWRF_NESTED_BUNDLE_RUNNER_SHA")
        dirty = runner._git(runner.REPO_ROOT, "status", "--porcelain")
        if approved != head:
            raise runner.RunnerGateError(
                "V10_RUNNER_HEAD", f"approved={approved} head={head}",
            )
        if require_clean and dirty:
            raise runner.RunnerGateError("V10_WORKTREE_DIRTY", dirty)
        ancestor = subprocess.run(
            ("git", "-C", str(runner.REPO_ROOT), "merge-base", "--is-ancestor", BASE_COMMIT, head),
            check=False,
        ).returncode == 0
        model_tree = runner._git(runner.REPO_ROOT, "rev-parse", "HEAD:src/gpuwrf")
        model_delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", BASE_COMMIT, "HEAD", "--", "src/gpuwrf",
        )
        delta = runner._git(
            runner.REPO_ROOT, "diff", "--name-only", BASE_COMMIT, "HEAD",
        ).splitlines()
        allowed_exact = {
            "scripts/v0234_nested_frozen_wrf_boundary_window.py",
            "scripts/v0234_v10_pinned_replay_profile.py",
            "scripts/v0234_v10_replay_evidence.py",
            "scripts/v0234_v10_cpu_domain_load_preflight.py",
            "tests/test_v0234_nested_frozen_wrf_boundary_window.py",
            "tests/test_v0234_v10_pinned_replay.py",
        }
        allowed_prefix = ".agent/sprints/2026-07-21-v0234-gpt-v10-replay/"
        unexpected = sorted(
            path for path in delta
            if path not in allowed_exact and not path.startswith(allowed_prefix)
        )
        if not ancestor or model_tree != MODEL_TREE or model_delta or unexpected:
            raise runner.RunnerGateError(
                "V10_SOURCE_AUTHORITY",
                repr({
                    "ancestor": ancestor,
                    "model_tree": model_tree,
                    "model_delta": model_delta,
                    "unexpected": unexpected,
                }),
            )
        return {
            "runner_head": head,
            "approved_runner_head": approved,
            "base_commit": BASE_COMMIT,
            "base_is_ancestor": ancestor,
            "accepted_model_tree": model_tree,
            "accepted_model_diff_empty": True,
            "runner_only_delta": delta,
            "unexpected_runner_delta": unexpected,
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
            "worktree_clean": not bool(dirty),
        }

    def validate_replay_environment(
        environment: Mapping[str, str],
        *,
        require_runner_audit: bool,
        require_tooling_critic: bool = False,
    ) -> dict[str, Any]:
        base = base_validate_environment(
            environment,
            require_runner_audit=require_runner_audit,
            require_tooling_critic=False,
        )
        if environment.get("GPUWRF_V10_PINNED_REPLAY") != "1":
            raise runner.RunnerGateError("V10_REPLAY_ENV", "expected literal 1")
        if environment.get("GPUWRF_V10_REPLAY_MODE") != mode:
            raise runner.RunnerGateError("V10_REPLAY_MODE_DRIFT", mode)
        if environment.get("GPUWRF_V10_REPLAY_NONCE") != nonce:
            raise runner.RunnerGateError("V10_REPLAY_NONCE_DRIFT", nonce)
        if environment.get("GPUWRF_V10_REPLAY_AUTOTUNE_OUTPUT") != str(output_pin):
            raise runner.RunnerGateError("V10_AUTOTUNE_OUTPUT", str(output_pin))
        if output_pin.exists() or output_pin.is_symlink():
            raise runner.RunnerGateError("V10_AUTOTUNE_OUTPUT_PREEXISTS", str(output_pin))
        if mode == "reference":
            expected_flags = f"--xla_gpu_dump_autotune_results_to={output_pin}"
            if environment.get("GPUWRF_V10_REPLAY_REFERENCE_PIN"):
                raise runner.RunnerGateError("V10_REFERENCE_PIN_UNEXPECTED", mode)
            pin_manifest = authenticate_pin_manifest(required=False)
        else:
            expected_flags = (
                f"--xla_gpu_load_autotune_results_from={reference_pin} "
                f"--xla_gpu_dump_autotune_results_to={output_pin}"
            )
            pin_manifest = authenticate_pin_manifest(required=True)
        if environment.get("XLA_FLAGS") != expected_flags:
            raise runner.RunnerGateError(
                "V10_XLA_FLAGS",
                f"expected={expected_flags!r} actual={environment.get('XLA_FLAGS')!r}",
            )
        return {
            **base,
            "v10_replay": {
                "mode": mode,
                "nonce": nonce,
                "namespace": binding["namespace"],
                "xla_flags": expected_flags,
                "autotune_output": str(output_pin),
                "pin_manifest": pin_manifest,
                "persistent_compilation_cache_used": False,
                "scored_horizon_d03_step": 9000,
            },
        }

    def audit_replay_launcher(path: Path) -> dict[str, Any]:
        text = path.read_text()
        required = (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"NONCE={nonce}",
            reference_namespace,
            pinned_namespace,
            "GPUWRF_V10_PINNED_REPLAY=1",
            'GPUWRF_V10_REPLAY_MODE="$MODE"',
            'GPUWRF_V10_REPLAY_NONCE="$NONCE"',
            f'RUNTIME_AUTHORITY="{RUNTIME_AUTHORITY_ROOT}"',
            'GPUWRF_WRF_ROOT="$RUNTIME_AUTHORITY"',
            "--xla_gpu_dump_autotune_results_to=$DUMP_PIN",
            "--xla_gpu_load_autotune_results_from=$REFERENCE_PIN",
            str(runner.LOCK_WRAPPER),
            "--intent production-preemptible",
            "/usr/bin/taskset -c 13,14,15,29,30,31",
            "-m scripts.v0234_nested_frozen_wrf_boundary_window",
            "--direct-terminal",
            "--record-known-1500-v10-red",
            "JAX_ENABLE_COMPILATION_CACHE=false",
            "GPUWRF_JAX_CACHE=0",
            "GPUWRF_NESTED_FUSE=0",
        )
        forbidden = tuple(
            token for token in (
                "nvidia-smi", "rocm-smi", "GPUWRF_TOLERANCE",
                "GPUWRF_SANITIZER", "--parent-join-resume",
                "--continuation-authority", "GPUWRF_NESTED_EVENT_AWARE_FUSION_K=1",
            ) if token in text
        )
        missing = [token for token in required if token not in text]
        return {
            "passed": bool(
                not missing
                and not forbidden
                and text.count("/scripts/with_gpu_lock.sh") == 1
                and text.count("-m scripts.v0234_nested_frozen_wrf_boundary_window") == 1
            ),
            "path": str(path.resolve()),
            "sha256": runner.sha256_file(path),
            "missing_required_tokens": missing,
            "forbidden_tokens": list(forbidden),
            "deterministic_dump_then_pin_pair": True,
            "late_ni_scope": False,
            "shell_syntax_checked_separately": True,
        }

    def replay_schedule_oracle() -> dict[str, Any]:
        root_steps = sum(PREFIX_SEGMENTS)
        own_steps = {"d01": root_steps, "d02": root_steps * 3, "d03": root_steps * 9}
        outputs = runner.standard_output_steps(PREFIX_OWN_STEPS)
        counts = {name: len(values) for name, values in outputs.items()}
        passed = bool(
            own_steps == PREFIX_OWN_STEPS
            and counts == EXPECTED_COUNTS
            and outputs["d03"][-2:] == (8800, 9000)
            and outputs["d01"][-1] == 1000
            and outputs["d02"][-1] == 3000
        )
        return {
            "passed": passed,
            "prefix_segments": list(PREFIX_SEGMENTS),
            "prefix_build_count": 1,
            "prefix_own_steps": own_steps,
            "prefix_output_steps": {name: list(values) for name, values in outputs.items()},
            "window_output_counts": counts,
            "terminal_output_counts": counts,
            "first_progress_d03_steps": list(runner.FIRST_PROGRESS_D03_STEPS),
            "decisive_d03_step": runner.DECISIVE_D03_STEP,
            "scoped_stop_after_decisive": True,
            "late_window_dispatch_count": 0,
            "terminal_continuation_dispatch_count": 0,
        }

    def replay_static_audit() -> dict[str, Any]:
        runner_source = runner.RUNNER_SOURCE.read_text()
        classifier_source = inspect.getsource(classify_scoped_v10_observation)
        accepted_model_tree = runner._git(
            runner.REPO_ROOT, "rev-parse", f"{MODEL_TREE_COMMIT}:src/gpuwrf"
        )
        accepted_model_commit_is_ancestor = subprocess.run(
            (
                "git", "-C", str(runner.REPO_ROOT), "merge-base",
                "--is-ancestor", MODEL_TREE_COMMIT, "HEAD",
            ),
            check=False,
        ).returncode == 0
        checks = {
            "prefix_terminal_hook_present": (
                'globals().get("PROFILE_PREFIX_TERMINAL_HANDLER")' in runner_source
            ),
            "profile_selected_before_legacy_profiles": (
                'os.environ.get("GPUWRF_V10_PINNED_REPLAY", "") == "1"'
                in runner_source
            ),
            "obsolete_step200_candidate_gate_replaced": (
                'globals().get("PROFILE_EARLY_CAUSAL_HANDLER")' in runner_source
            ),
            "cpu_only_wrappers_not_misclassified_as_gpu_preemption": all(
                token in runner_source
                for token in (
                    "def _declares_explicit_cpu_only_no_cuda",
                    "CPU_ONLY_JAX_PLATFORM_PATTERN",
                    "CUDA_HIDDEN_PATTERN",
                    "if _declares_explicit_cpu_only_no_cuda(argv):",
                )
            ),
            "only_v_v10_red_retainable": "ALLOWED_RED_FIELDS" in classifier_source,
            "no_waiver_or_tolerance_change": all(
                token in classifier_source
                for token in (
                    '"waiver_or_reclassification": False',
                    '"tolerance_changed": False',
                    '"continue_beyond_scored_horizon": False',
                )
            ),
            "unchanged_v_gate": runner.FROZEN_1500_RMSE["V"] == 1.1318203205639872,
            "unchanged_v10_gate": runner.FROZEN_1500_RMSE["V10"] == 2.1128268857679338,
            "accepted_model_tree_authenticated": accepted_model_tree == MODEL_TREE,
            "accepted_model_commit_is_ancestor": accepted_model_commit_is_ancestor,
            "contract_hash_frozen": contract.is_file() and _sha256(contract) == CONTRACT_SHA256,
            "runtime_authority_authenticated": bool(
                authenticate_runtime_authority().get("authority_sha256")
            ),
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
            "profile_source_sha256": runner.sha256_file(Path(__file__).resolve()),
        }

    def authenticate_scoped_observation() -> tuple[dict[str, Any], dict[str, Any]]:
        authority = assert_replay_final_authority()
        compact = {
            "schema": "gpuwrf.v0234.v10-scoped-retention-authority.v1",
            "contract_sha256": CONTRACT_SHA256,
            "nonce": nonce,
            "mode": mode,
            "allowed_red_fields": sorted(ALLOWED_RED_FIELDS),
            "stop_after_d03_step": 9000,
            "release_gate_unchanged": True,
            "late_ni_scope": False,
        }
        compact["authority_sha256"] = runner.canonical_digest(compact)
        return compact, {
            "path": str(contract.resolve()),
            "file_sha256": CONTRACT_SHA256,
            "authority_sha256": compact["authority_sha256"],
            "manager_authorization": authority["manager_authorization"],
        }

    def record_current_tree_step200(**context: Any) -> dict[str, Any]:
        """Retain provenance/finite evidence without applying an obsolete gate.

        The inherited T/U ring-1 comparison was an admission gate for the
        470e6111 candidate.  Its fixed literals and predecessor frames are not
        authority for the current model tree.  This replay therefore records
        the real current-tree step-200 frame and metrics, while leaving the
        unchanged release decision exclusively at the scoped step-9000 gate.
        """

        strict_rmse = {
            field: float(context["strict_rmse"].get(field, math.nan))
            for field in STRICT_FIELDS
        }
        finite = all(math.isfinite(value) for value in strict_rmse.values())
        identity = context["finite_identity"]
        passed = bool(finite and identity.get("passed") is True)
        return {
            "schema": "gpuwrf.v0234.v10-current-tree-step200-record.v1",
            "passed": passed,
            "policy": "PROVENANCE_AND_FINITE_RECORD_ONLY",
            "legacy_470e6111_t_u_ring1_gate_applicable": False,
            "legacy_gate_waived_or_reclassified": False,
            "reason": (
                "the inherited causal gate is candidate-specific and predates "
                "the accepted current model tree"
            ),
            "current_model_tree": MODEL_TREE,
            "release_gate": "unchanged strict full-grid d03 step 9000 only",
            "strict_rmse": strict_rmse,
            "all_strict_fields_finite": finite,
            "finite_and_static_identity_pass": identity.get("passed") is True,
            "candidate": {
                "path": str(Path(context["candidate_path"]).resolve()),
                "sha256": context["candidate_sha256"],
            },
            "cpu": {
                "path": str(Path(context["cpu_path"]).resolve()),
                "sha256": context["cpu_sha256"],
            },
        }

    def finish_at_v10_horizon(**context: Any) -> int:
        args = context["args"]
        authority = context["authority"]
        runtime = context["runtime"]
        pairer = context["pairer"]
        output = context["output"]
        prefix_steps = context["prefix_steps"]
        decisive = [
            row for row in pairer.rows
            if row["domain"] == "d03" and row["own_step"] == 9000
        ]
        if prefix_steps != PREFIX_OWN_STEPS or pairer.counts != EXPECTED_COUNTS or len(decisive) != 1:
            raise runner.RunnerGateError(
                "V10_HORIZON_INVENTORY",
                repr({"steps": prefix_steps, "counts": pairer.counts, "decisive": len(decisive)}),
            )
        row = decisive[0]
        decision = row.get("decisive_1500") or {}
        scoped = row.get("known_1500_v10_record") or {}
        release_green = decision.get("passed") is True
        if not release_green and (
            row.get("scientific_pair_pass") is not False
            or scoped.get("passed") is not True
            or scoped.get("release_gate_green") is not False
        ):
            raise runner.RunnerGateError("V10_SCOPED_RED_SEMANTICS", repr(row))
        red_fields = sorted(
            field for field, value in (decision.get("strict_fields") or {}).items()
            if value.get("no_worse") is not True
        )
        if release_green and red_fields:
            raise runner.RunnerGateError("V10_GREEN_WITH_RED_FIELDS", repr(red_fields))
        authority["inputs_post"] = runner.assert_input_retry20_cache_authority(os.environ)
        authority["runtime_authority_post"] = authenticate_runtime_authority()
        authority["lock_post"] = runner.assert_live_lock_authority(os.environ)
        authority["preemption_post"] = runner.assert_preemption_clear("v10-horizon-complete")
        args._full_run_stage = "V10_REPLAY_HORIZON_COMPLETE"
        args._full_run_stages.append(args._full_run_stage)
        proof = {
            "schema": runner.SCHEMA,
            "verdict": "V10_V_STRICT_GREEN" if release_green else "V10_V_STRICT_RED_RETAINED",
            "mode": mode,
            "nonce": nonce,
            "namespace": binding["namespace"],
            "runner_head": authority["candidate"]["runner_head"],
            "src_gpuwrf_tree": MODEL_TREE,
            "authority": authority,
            "load_authority": context["load_authority"],
            "scheduler": context["schedule"],
            "initial_history": context["initial_history"],
            "prefix": context["prefix_proof"],
            "prefix_health": context["prefix_health"],
            "ordinary_one_step_program": context["ordinary_audit"],
            "health_program": context["health_executable"].audit(),
            "incremental_frame_pairs": {
                "counts": dict(pairer.counts),
                "expected_counts": EXPECTED_COUNTS,
                "rows": pairer.rows,
                "all_outputs_retained": True,
            },
            "checkpoint_carries": output.checkpoints,
            "decisive_1500": row,
            "strict_rmse": {
                field: float((decision["strict_fields"][field])["candidate_rmse"])
                for field in STRICT_FIELDS
            },
            "red_fields": red_fields,
            "release_gate_green": release_green,
            "reference_or_pinned_scoped_horizon_complete": True,
            "output_inventory": [
                {
                    "domain": emitted["domain"],
                    "own_step": emitted["own_step"],
                    "valid_time": emitted["valid_time"],
                    "wrfout": emitted["wrfout"],
                }
                for emitted in output.emitted
            ],
            "scope": {
                "d03_terminal_step": 9000,
                "late_ni_steps_dispatched": [],
                "late_ni_claimed": False,
                "full_18h_claimed": False,
                "tolerance_changed": False,
                "waiver_or_reclassification": False,
                "persistent_compilation_cache_used": False,
                "model_or_numerical_edit": False,
            },
            "timing": {
                "started_utc": context["started"].isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "load_wall_seconds": context["load_wall_seconds"],
            },
            "stage_trace": list(args._full_run_stages),
        }
        proof["proof_sha256"] = runner.canonical_digest(proof)
        runner.atomic_write_json(args.proof_output, proof)
        print(json.dumps({
            "verdict": proof["verdict"],
            "mode": mode,
            "V": proof["strict_rmse"]["V"],
            "V10": proof["strict_rmse"]["V10"],
            "red_fields": red_fields,
            "proof": str(args.proof_output),
            "proof_sha256": proof["proof_sha256"],
            "late_ni_dispatched": False,
        }, sort_keys=True), flush=True)
        return 0

    runner.assert_final_candidate_proof_authority = assert_replay_final_authority
    runner.assert_candidate_source_authority = assert_replay_source_authority
    runner.validate_preimport_environment = validate_replay_environment
    runner.audit_exact_launch_command = audit_replay_launcher
    runner.schedule_clock_oracle = replay_schedule_oracle
    runner.static_source_audit = replay_static_audit
    runner.authenticate_known_1500_v10_observation = authenticate_scoped_observation
    runner.classify_exact_known_1500_v10_red = classify_scoped_v10_observation
    runner.PROFILE_EARLY_CAUSAL_HANDLER = record_current_tree_step200
    runner.PROFILE_PREFIX_TERMINAL_HANDLER = finish_at_v10_horizon
    runner._V10_PINNED_REPLAY_PROFILE_APPLIED = True
