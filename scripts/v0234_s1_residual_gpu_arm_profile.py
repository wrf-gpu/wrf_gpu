"""Pre-import profile for the frozen v0234 S1 residual-closure GPU arm."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt"
CONTRACT = SPRINT / "GPU-ARM-CONTRACT.md"
CONTRACT_SHA256 = "9f2a04b914a95295d1e7ae690a4fc96a5af019ab315f9c1c6d93a3b403517cf2"
LAUNCHER = SPRINT / "s1-residual-exact-launch-command.sh"
CPU_AUDIT = SPRINT / "s1-residual-runner-cpu-proof.json"
CANDIDATE_HEAD = "d4b03077f2e8858163d1fbdf99c090fab902e0ac"
MODEL_PARENT = "e65ce784bec85ea4f000e94ba572420c595ecb68"
MODEL_HASHES = {
    "src/gpuwrf/dynamics/core/rk_addtend_dry.py": "15a9eccc3f66e25176f6b23fca205b7d997c04c82d15aad2b5ae52cc183f2c34",
    "src/gpuwrf/dynamics/explicit_diffusion.py": "f6eaa78d08ee25dd7d2977721521bf30a21fae72a751d9184d2a408459c4b11e",
    "src/gpuwrf/runtime/operational_mode.py": "68efe76b9d1f91860e9a49a6e6c9ab9e573986dd11a47677fa161badb3a79282",
}
NAMESPACE = "nested_stage_omega_transport_470e6111_s1_residual_closure_gpt1"
LOCK_LABEL = "v0234-s1-residual-closure"
AUTHORIZATION_PATH = Path("/tmp/v0234-s1-residual-gpu-authorization.json")
AUTHORIZATION_SCHEMA = "gpuwrf.v0234.s1-residual-gpu-authorization.v1"
SPRINT_NAME = "2026-07-18-v0234-s1-residual-closure-gpt"
PREDICTED = {
    "nonspec": 0.7955252696331928,
    "relax_rows_1_4": 0.7826322158654425,
    "interior_ge_5": 0.7978517274341745,
}
ABSOLUTE_BAND = 0.01


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _canonical(value: object, *, omit: str | None = None) -> str:
    if omit is not None and isinstance(value, dict): value = {k: v for k, v in value.items() if k != omit}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_authorization(path: Path = AUTHORIZATION_PATH, *, expected_head: str, now: datetime | None = None) -> dict[str, Any]:
    if path.resolve() != AUTHORIZATION_PATH.resolve() or not path.is_file() or path.is_symlink():
        raise ValueError("authorization absent, non-file, symlink, or wrong path")
    payload = json.loads(path.read_text())
    required = {"schema", "decision", "issuer", "intent", "one_arm", "sprint", "namespace", "lock_label", "candidate_head", "expires_utc", "nonce", "proof_sha256"}
    if set(payload) != required or payload.get("proof_sha256") != _canonical(payload, omit="proof_sha256"):
        raise ValueError("authorization fields or self-hash invalid")
    expected = {
        "schema": AUTHORIZATION_SCHEMA, "decision": "RELEASE_GPU",
        "issuer": "manager-0:1", "intent": "production-preemptible", "one_arm": True,
        "sprint": SPRINT_NAME, "namespace": NAMESPACE, "lock_label": LOCK_LABEL,
        "candidate_head": expected_head,
    }
    for key, value in expected.items():
        if payload.get(key) != value: raise ValueError(f"authorization mismatch: {key}")
    if not isinstance(payload.get("nonce"), str) or not payload["nonce"].strip(): raise ValueError("authorization nonce empty")
    expiry = datetime.fromisoformat(str(payload["expires_utc"]).replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= (now or datetime.now(timezone.utc)): raise ValueError("authorization expired")
    return {"path": str(path.resolve()), "file_sha256": _sha256(path), "proof_sha256": payload["proof_sha256"], "issuer": payload["issuer"], "expires_utc": expiry.isoformat(), "candidate_head": expected_head, "validated_before_runtime_import": True}


def classify_step1_metrics(metrics: Mapping[str, float]) -> dict[str, Any]:
    rows: dict[str, Any] = {}; first_red = None
    for name in ("nonspec", "relax_rows_1_4", "interior_ge_5"):
        value = float(metrics[name]); delta = abs(value - PREDICTED[name]); passed = math.isfinite(value) and delta <= ABSOLUTE_BAND
        rows[name] = {"value": value, "predicted": PREDICTED[name], "absolute_delta": delta, "absolute_delta_le": ABSOLUTE_BAND, "finite": math.isfinite(value), "passed": passed}
        if first_red is None and not passed: first_red = name
    return {"passed": first_red is None, "first_red": first_red, "rows": rows, "step2_dispatch_admitted": first_red is None}


def audit_exact_launcher(path: Path = LAUNCHER) -> dict[str, Any]:
    text = path.read_text()
    required = ("#!/usr/bin/env bash", "set -euo pipefail", str(AUTHORIZATION_PATH), "--verify-authorization", NAMESPACE, LOCK_LABEL, "GPUWRF_S1_RESIDUAL_VALIDATION=1", "GPUWRF_S1_DIFF6_VALIDATION=1", "/scripts/with_gpu_lock.sh", "--timeout 0", "--intent production-preemptible", "/usr/bin/taskset -c 13-15,29-31", "-m scripts.v0234_nested_frozen_wrf_boundary_window", "-m scripts.v0234_s1_residual_gpu_arm_closeout", "--seal-release-receipt", "/usr/bin/env -i")
    missing = [token for token in required if token not in text]
    forbidden = [token for token in ("nvidia-smi", "rocm-smi", "GPUWRF_TOLERANCE", "GPUWRF_SANITIZER") if token in text]
    auth = text.find("--verify-authorization"); fresh = text.find('test ! -e "$RUN_DIR"'); lock = text.find("/scripts/with_gpu_lock.sh"); closeout = text.find("-m scripts.v0234_s1_residual_gpu_arm_closeout"); receipt = text.find("--seal-release-receipt")
    passed = not missing and not forbidden and text.count("/scripts/with_gpu_lock.sh") == 1 and 0 <= auth < fresh < lock < closeout < receipt
    return {"passed": passed, "missing": missing, "forbidden": forbidden, "authorization_before_freshness_before_lock": 0 <= auth < fresh < lock, "closeout_after_lock_before_receipt": lock < closeout < receipt, "file_sha256": _sha256(path)}


def _configure_base() -> Any:
    from scripts import v0234_s1_diff6_gpu_arm_profile as base
    base.SPRINT_NAME = SPRINT_NAME; base.SPRINT = SPRINT; base.CONTRACT = CONTRACT; base.CONTRACT_SHA256 = CONTRACT_SHA256
    base.LAUNCHER = LAUNCHER; base.CPU_AUDIT = CPU_AUDIT; base.CANDIDATE_HEAD = CANDIDATE_HEAD
    base.FIX_COMMIT = CANDIDATE_HEAD; base.FABLE_PARENT = MODEL_PARENT; base.MODEL_HASHES = MODEL_HASHES
    base.NAMESPACE = NAMESPACE; base.LOCK_LABEL = LOCK_LABEL; base.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    base.AUTHORIZATION_SCHEMA = AUTHORIZATION_SCHEMA; base.THRESHOLDS = {name: value + ABSOLUTE_BAND for name, value in PREDICTED.items()}
    base.AUTOTUNE_STAGE_NAME = ".v0234-s1-residual-validation-autotune-v1"; base.DUMP_PIN_NAME = "s1-residual-validation-autotune-results.pb"
    base.FROZEN_ATTRIBUTION = SPRINT / "candidate-cpu-proof.json"; base.FROZEN_ATTRIBUTION_SELF = "500db72270e5bd0d5eceb5c37a807cc530d41d38bc716f1c71c754906ea6e8b9"
    base.FROZEN_RUNTIME_PROOF = Path("<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/corrected_ni_rca_max_22c2bd7a/nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1/s1-diff6-validation-terminal-proof.json")
    base.FROZEN_RUNTIME_SELF = "6e8ed620932459a6d663feafb8a64cdd3bf5cbd3aaa65a6e878dcdb1e70c677e"; base.FROZEN_SAVEPOINT_ROWS = "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a"
    base.validate_authorization = validate_authorization; base.classify_step1_metrics = classify_step1_metrics; base.audit_exact_launcher = audit_exact_launcher
    return base


def apply_profile(runner: Any) -> None:
    base = _configure_base(); base.apply_profile(runner)
    runner.SCHEMA = "gpuwrf.v0234.s1-residual-validation-arm.v1"
    runner.CPU_PROOF_SCHEMA = "gpuwrf.v0234.s1-residual-runner-cpu-audit.v1"
    runner.AUDIT_ADMISSION = "READY_FOR_AUTHORIZED_SINGLE_S1_RESIDUAL_GPU_ARM"
    runner.LADDER_SUCCESS_VERDICT = "S1_RESIDUAL_ARM_RAW_GREEN_PENDING_CLOSEOUT"
    runner.CPU_FOCUSED_TEST_ARGS = ("tests/test_v0234_s1_residual_gpu_arm.py", "tests/test_v0234_s1_residual_source_operators.py")
    runner.INFRASTRUCTURE_GPUWRF_ENV = set(runner.INFRASTRUCTURE_GPUWRF_ENV) | {"GPUWRF_S1_RESIDUAL_VALIDATION"}
    runner.CANDIDATE_CLEAN_AUTHORITY.update({"source_candidate_commit": CANDIDATE_HEAD, "pgf_model_edit": False, "raw_predictions": PREDICTED, "absolute_band": ABSOLUTE_BAND, "post_lock_intrinsic_closeout_required": True})


def _main(argv: list[str] | None = None) -> int:
    return int(_configure_base()._main(argv))


if __name__ == "__main__": raise SystemExit(_main())
