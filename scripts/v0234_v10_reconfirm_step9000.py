#!/usr/bin/env python3
"""Pinned Tenerife Step0-to-d03-step9000 V10 re-confirmation.

This runner is deliberately stdlib-only until it has authenticated the model,
fixture, determinism pin, manager authorization, fresh namespace, and inherited
GPU lock.  It reuses the accepted H5 scoring implementation and cold-Step0
admission proof, but stops exactly at d03 step 9000.  It never dispatches a
post-terminal model step and never reruns the CPU or pre-fix GPU baseline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

from scripts import v0234_h5_terminal_step9000 as h5


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-reconfirm"
LOCK_LABEL = "v0234-gpt-v10-reconfirm"
ORIGINAL_MODEL_COMMIT = "3b81fb5b093639e70c12cce87d602c45b326b18b"
CANDIDATE_MODEL_COMMIT = "72fb78a87fda04519a241f5b00fdf2c1712063d7"
CANDIDATE_SRC_TREE = "7169637f197d558738d232c9fbc006d7ca0902bf"
CANDIDATE_SOURCE_SHA256 = {
    "src/gpuwrf/integration/nested_pipeline.py": (
        "77b6a690647ec4b0b615f425338ca41f3e594c29566eeafc0f9e884e1f0e98c4"
    ),
    "src/gpuwrf/physics/thompson_aero_column.py": (
        "b45e56e36c008bdef240e9c1036c91ddd0118effeff97b5fbabb520fd2c14608"
    ),
    "src/gpuwrf/physics/thompson_column.py": (
        "b49a9ceacb9dd032bdb3d2d9e2f05c0be0e5d73f826ddc79fce052311b0a31a8"
    ),
    "src/gpuwrf/runtime/operational_mode.py": (
        "d5ce22236188c4264ab13e3108f163830d9c5bce1560b0090db1c927e8e1a19d"
    ),
}
EXPECTED_SOURCE_DELTA = (
    "src/gpuwrf/physics/thompson_aero_column.py",
    "src/gpuwrf/physics/thompson_column.py",
)

# Base union of every manager-issued V10/late-Ni nonce found before this sprint,
# plus this sprint's consumed re-confirmation arms below. Length is intentionally
# 32--128, because one historical manager nonce has 59 hex characters.
BASE_CONSUMED_NONCES = frozenset(
    {
        "ad0e60072a2a0df1ceacf0f5593e3834f2d0da10086f46316375f8f17aa",
        "28185746394c972f906031a9f8b28510faef6b94b903010fb25bdba2c26ee5ad",
        "30b5cf89ff6027c40d5b441442f194308ac1e6f3239d486bcace39c60ba8c9dc",
        "736b7b414bc6ba311c306365777ff173a62a0ada68e91f525e2c5dc1f860cf6d",
        "7576309de98eaa105c3a8e070e9500df35b57d492fc38f35817fcfc3594c6c3e",
        "8520924a5f97acc1ea8337bb16ea860903b8436e2d30b10b6efca9d54f55552c",
        "c17fca1e201ae106cf8ad77463686823c82226d59c7dea01e741c0be0ba7f6bc",
        "ca1df30181823579e35b5c3dbadfa09782cfde6104efe3793937ed9d38e02f2a",
        "d414b0b7ad7064fdb81bfc1d743f44d68f9e68ea811f8d2aa0ca1a6237244cac",
        "de550240614a6fe3f833cac1d489dad93b9d7e91cba196fb62490443ced76d72",
        "df242a850d6ebab12529978934df494857c46002dda2ad8bb10d2e3c84ee6796",
        "ed8f781ae62e33cef6bd7618599e8c84dd907633ba1c39a3176b9a61d6b0c3f5",
        "eef7df75c5a6b2731615a81e5ca868caa97d7bfa3b3769a6c79e809d5f2011fd",
        "f1e9d810260e2eaa92fed5eba3a10a3268604c3996efbd705d5cc023cf1bcce8",
        "fbb2716e855e4fe572d1bbef538f8b26ce5f4a106a4ed8f7b44b41400e852687",
    }
)
RECONFIRM_CONSUMED_NONCES = frozenset(
    {
        "262f0a774ca423607720e5d503633b1b86cdc0171782d4c8f7baec3bae07b82e",
        "2f71bb631e4ebdac061281a1c2b1a114d08a5a8dedfb985a34338c9974d02731",
        "99a8e7d0b499a18012384fc0011f8973970f144a91099ba225c353104a7d54f3",
        "21b93124fb9cfe0c83cf9ff81e3550f9b1d5c073519ef14a9bcd07bbd5815a54",
    }
)
CONSUMED_NONCES = BASE_CONSUMED_NONCES | RECONFIRM_CONSUMED_NONCES
NONCE_PATTERN = re.compile(r"[0-9a-f]{32,128}")
NONCE_AUDIT = SPRINT / "CONSUMED_NONCE_AUDIT.json"
NONCE_AUDIT_ADDENDUM = SPRINT / "CONSUMED_NONCE_AUDIT_RETRY4_ADDENDUM.json"
NONCE_AUDIT_ADDENDUM_CANONICAL_SHA256 = (
    "057ca5480ec45b34f6b7e6389b31e3bcc0a9f1bc27215ba7c95b187e008d544e"
)
CPU_PREFLIGHT = SPRINT / "GPU_STEP9000_CPU_PREFLIGHT_RETRY5.json"
COMPUTE_PID_FILTER = SPRINT / "v10-compute-pid-filter.awk"
COMPUTE_PID_FILTER_SHA256 = (
    "5bc22a89b3ee1b563cb56cb2254cf74ab6fac9748145ad8b72bc566f2701b6b7"
)

ICE_GPU_PROOF = (
    ROOT
    / ".agent/sprints/2026-07-21-v0234-opus-late-ni-fix/"
    "OPUS_LATE_NI_GPU_CONFIRM_RESULT.json"
)
ICE_GPU_PROOF_FILE_SHA256 = (
    "97576542e5fb3aaeff994cabae8e2049d50967f9a97dee95d8a6246424dfaaca"
)
ICE_GPU_PROOF_CANONICAL_SHA256 = (
    "f8de27065353e855801abc9bd064ac051d01a41fcef368ac5c5ff4b65cfd4c34"
)
ICE_BASELINE_PROOF = (
    ROOT
    / ".agent/sprints/2026-07-21-v0234-opus-late-ni-fix/BASELINE_IMPACT.json"
)
ICE_BASELINE_PROOF_FILE_SHA256 = (
    "9c8b7d3c8bf0fcb9f2670cba37c6853072a7d0404087329c948504819157e326"
)
ICE_BASELINE_PROOF_CANONICAL_SHA256 = (
    "d7a55c1b67b945a6114c089cb4e06f41d7372398efa1164336e32d3c6f9ed81c"
)

LINEAGE = h5.LINEAGE
INPUT_DIR = h5.INPUT_DIR
RUNTIME_AUTHORITY = h5.RUNTIME_AUTHORITY
AUTOTUNE_PIN = h5.AUTOTUNE_PIN
EXPECTED_AFFINITY = h5.EXPECTED_AFFINITY
ROOT_STEP_END = 1000
D03_STEP_END = 9000
EXPECTED_OUTPUT_COUNTS = {"d01": 16, "d02": 16, "d03": 46}
OUTPUT_CADENCE = {"d01": 67, "d02": 200, "d03": 200}

ORIGINAL_H5_RMSE = {
    "PSFC": 12.701964738122888,
    "T": 0.3535465029687384,
    "T2": 0.7596390497352711,
    "U": 0.8784029261656536,
    "U10": 1.5744613630957462,
    "V": 0.9283496901915557,
    "V10": 1.9423363525731614,
    "W": 0.1516568929748467,
}
ORIGINAL_H5_CORRELATIONS = {
    "U10": 0.8628009208688981,
    "V10": 0.8138990847660432,
}
MATERIALITY_SCALE = {
    "fields": ["V", "V10"],
    "absolute_rmse_m_s": 0.01,
    "relative_percent": 1.0,
    "rule": (
        "materially unchanged only when both V and V10 movements are within "
        "both absolute and relative bounds"
    ),
}

# Empirical planning split from the accepted pinned H5 arms.  Compilation is
# lazily interleaved, so this is an honest estimate rather than fake precision.
EXPECTED_TIMING = {
    "cold_compile_and_first_step200_envelope_seconds": 1304.7110441131517,
    "observed_additional_lazy_compile_seconds": 193.462344627,
    "post_step200_through_step9000_model_seconds": 5036.752290769014,
    "domain_load_and_terminal_scoring_overhead_seconds": 485.959838848561,
    "expected_total_seconds": 6827.423173730727,
    "hard_model_process_timeout_seconds": 7800,
    "interpretation": (
        "approximately 25 minutes compilation (cold plus observed lazy compile), "
        "81 minutes integration/output measurement, and 8 minutes load/scoring; "
        "compile and measurement are not separately instrumented by the production runner"
    ),
}

# The general Nightly state also covers CPU-only publication work, so it is not
# by itself evidence of GPU contention.  A manager authorization may classify
# it as informational only when it is bound to the run nonce and relays direct
# 0:3 confirmation that gpu_chunks_enabled=false.  Real preemption sentinels,
# the inherited exclusive lock, and the in-lock compute-PID query remain hard
# fail-closed gates.
REAL_PREEMPT_PATHS = tuple(h5.PREEMPT_PATHS) + (
    Path("/tmp/PREEMPT_CPU"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_CPU"),
)
PREEMPTION_POLICY_BASE = {
    "authority": "manager-0:1-relaying-direct-0:3",
    "exception_scope": (
        "single explicitly authorized arm; expires at terminal lock release"
    ),
    "coarse_nightly_active_json_role": (
        "informational-only-under-explicit-cpu-only-nightly-authorization"
    ),
    "direct_0_3_gpu_clearance": (
        "Nightly-Compute and 9-Nest are CPU-only; "
        "gpu_chunks_enabled=false; no GPU use planned tonight"
    ),
    "nightly_gpu_chunks_enabled": False,
    "required_real_gpu_contention_gates": [
        "all configured PREEMPT sentinels absent",
        "inherited exclusive GPU lock intact at every segment",
        "in-lock prelaunch nvidia-smi compute-PID query clear",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _canonical_without(payload: Mapping[str, Any], field: str) -> str:
    clean = dict(payload)
    clean.pop(field, None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(
            (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def namespace_for_nonce(nonce: str) -> str:
    if NONCE_PATTERN.fullmatch(nonce) is None:
        raise RuntimeError("manager nonce must be 32--128 lowercase hex characters")
    if nonce in CONSUMED_NONCES:
        raise RuntimeError(f"manager nonce is already consumed: {nonce}")
    return f"v0234_gpt_v10_reconfirm_{nonce[:16]}_step9000"


def _expected_preemption_policy(nonce: str) -> dict[str, Any]:
    return {**PREEMPTION_POLICY_BASE, "authorization_nonce": nonce}


def _read_nonce_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    if not NONCE_AUDIT.is_file() or NONCE_AUDIT.is_symlink():
        raise RuntimeError("consumed-nonce audit is absent, non-file, or symlinked")
    base_payload = json.loads(NONCE_AUDIT.read_text())
    base_canonical = _canonical(base_payload)
    base_observed = {row.get("nonce") for row in base_payload.get("entries", [])}
    if (
        base_payload.get("schema")
        != "gpuwrf.v0234.v10-reconfirm-consumed-nonce-audit.v1"
        or base_payload.get("verdict")
        != "ALL_2026_07_21_V10_LATE_NI_NONCES_SEALED"
        or base_payload.get("proof_sha256") != base_canonical
        or base_observed != set(BASE_CONSUMED_NONCES)
        or len(base_payload.get("entries", [])) != len(BASE_CONSUMED_NONCES)
        or base_payload.get("s2_boundary_nonce_count") != 0
    ):
        raise RuntimeError("base consumed-nonce audit semantics or canonical hash changed")
    if not NONCE_AUDIT_ADDENDUM.is_file() or NONCE_AUDIT_ADDENDUM.is_symlink():
        raise RuntimeError("consumed-nonce addendum is absent, non-file, or symlinked")
    addendum = json.loads(NONCE_AUDIT_ADDENDUM.read_text())
    addendum_canonical = _canonical(addendum)
    added_observed = {
        row.get("nonce") for row in addendum.get("added_entries", [])
    }
    expected_base = {
        "canonical_sha256": base_canonical,
        "file_sha256": _sha256(NONCE_AUDIT),
        "path": (
            ".agent/sprints/2026-07-21-v0234-gpt-v10-reconfirm/"
            "CONSUMED_NONCE_AUDIT.json"
        ),
        "unique_consumed_nonce_count": len(BASE_CONSUMED_NONCES),
    }
    if (
        addendum.get("schema")
        != "gpuwrf.v0234.v10-reconfirm-consumed-nonce-audit-addendum.v1"
        or addendum.get("verdict")
        != "BASE_AUDIT_PLUS_RECONFIRM_ARM_NONCES_SEALED"
        or addendum.get("proof_sha256") != addendum_canonical
        or addendum_canonical != NONCE_AUDIT_ADDENDUM_CANONICAL_SHA256
        or addendum.get("base_audit") != expected_base
        or added_observed != set(RECONFIRM_CONSUMED_NONCES)
        or len(addendum.get("added_entries", []))
        != len(RECONFIRM_CONSUMED_NONCES)
        or addendum.get("added_nonce_count") != len(RECONFIRM_CONSUMED_NONCES)
        or addendum.get("union_unique_consumed_nonce_count")
        != len(CONSUMED_NONCES)
    ):
        raise RuntimeError("consumed-nonce addendum semantics or canonical hash changed")
    return {"base": base_payload, "addendum": addendum}, {
        "path": str(NONCE_AUDIT_ADDENDUM.resolve()),
        "file_sha256": _sha256(NONCE_AUDIT_ADDENDUM),
        "canonical_sha256": addendum_canonical,
        "base_path": str(NONCE_AUDIT.resolve()),
        "base_file_sha256": _sha256(NONCE_AUDIT),
        "base_canonical_sha256": base_canonical,
        "consumed_nonce_count": len(CONSUMED_NONCES),
    }


def _assert_exact_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} authority changed: {path}")


def _assert_model_and_fixture_authority(*, require_clean: bool) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    if require_clean and dirty:
        raise RuntimeError(f"worktree must be clean: {dirty}")
    for commit, label in (
        (ORIGINAL_MODEL_COMMIT, "original H5"),
        (CANDIDATE_MODEL_COMMIT, "ice-fixed candidate"),
    ):
        if subprocess.run(
            ("git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, head),
            check=False,
        ).returncode != 0:
            raise RuntimeError(f"{label} commit is not an ancestor of runner HEAD")
    src_tree = _git("rev-parse", "HEAD:src/gpuwrf")
    if src_tree != CANDIDATE_SRC_TREE:
        raise RuntimeError(f"candidate src/gpuwrf tree changed: {src_tree}")
    source_delta = tuple(
        line
        for line in _git(
            "diff",
            "--name-only",
            f"{ORIGINAL_MODEL_COMMIT}..{CANDIDATE_MODEL_COMMIT}",
            "--",
            "src/gpuwrf",
        ).splitlines()
        if line
    )
    if source_delta != EXPECTED_SOURCE_DELTA:
        raise RuntimeError(f"source delta from accepted H5 changed: {source_delta}")
    source_rows = {}
    for relative, expected in CANDIDATE_SOURCE_SHA256.items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"candidate source changed: {relative}: {actual}")
        source_rows[relative] = actual

    input_rows = {}
    for relative, expected in h5.INPUT_SHA256.items():
        path = INPUT_DIR / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"fixture input changed: {relative}: {actual}")
        input_rows[relative] = actual
    exact_runtime_files = sorted(
        str(path.relative_to(RUNTIME_AUTHORITY))
        for path in RUNTIME_AUTHORITY.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if exact_runtime_files != sorted(h5.RUNTIME_AUTHORITY_FILES):
        raise RuntimeError(f"runtime authority inventory changed: {exact_runtime_files}")
    runtime_rows = {}
    for relative, (expected_bytes, expected_sha) in h5.RUNTIME_AUTHORITY_FILES.items():
        path = RUNTIME_AUTHORITY / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or _sha256(path) != expected_sha
        ):
            raise RuntimeError(f"runtime authority changed: {relative}")
        runtime_rows[relative] = {"bytes": expected_bytes, "sha256": expected_sha}

    for path, expected, label in (
        (h5.CPU_STEP9000, h5.CPU_STEP9000_SHA256, "CPU step9000"),
        (h5.REFERENCE_PAIR, h5.REFERENCE_PAIR_SHA256, "original H5 frame pair"),
        (h5.CASE_METADATA, h5.CASE_METADATA_SHA256, "terminal case metadata"),
        (AUTOTUNE_PIN, h5.AUTOTUNE_PIN_SHA256, "determinism pin"),
        (h5.RETAINED_STEP0, h5.RETAINED_STEP0_SHA256, "retained Step0"),
        (h5.C1_PROOF, h5.C1_PROOF_SHA256, "C1 candidate-off proof"),
        (
            h5.COLD_STEP0_ANCHOR_PROOF,
            h5.COLD_STEP0_ANCHOR_PROOF_SHA256,
            "cold Step0 anchor proof",
        ),
        (h5.QKE_BACKEND_PROOF, h5.QKE_BACKEND_PROOF_SHA256, "QKE backend proof"),
        (
            h5.QKE_CPU_ORACLE_PROOF,
            h5.QKE_CPU_ORACLE_PROOF_SHA256,
            "QKE CPU oracle proof",
        ),
        (h5.QKE_CPU_ORACLE, h5.QKE_CPU_ORACLE_SHA256, "QKE CPU oracle array"),
        (ICE_GPU_PROOF, ICE_GPU_PROOF_FILE_SHA256, "late-Ni GPU confirmation"),
        (
            ICE_BASELINE_PROOF,
            ICE_BASELINE_PROOF_FILE_SHA256,
            "ice-balance baseline impact",
        ),
        (COMPUTE_PID_FILTER, COMPUTE_PID_FILTER_SHA256, "compute-PID filter"),
    ):
        _assert_exact_file(path, expected, label)

    c1 = json.loads(h5.C1_PROOF.read_text())
    cold = json.loads(h5.COLD_STEP0_ANCHOR_PROOF.read_text())
    qke_backend = json.loads(h5.QKE_BACKEND_PROOF.read_text())
    qke_oracle = json.loads(h5.QKE_CPU_ORACLE_PROOF.read_text())
    if (
        c1.get("canonical_payload_sha256") != h5.C1_CANONICAL_SHA256
        or c1.get("candidate_off_identity_pass") is not True
        or cold.get("proof_sha256") != h5.COLD_STEP0_ANCHOR_CANONICAL_SHA256
        or _canonical(cold) != h5.COLD_STEP0_ANCHOR_CANONICAL_SHA256
        or (cold.get("cold_state_identity") or {}).get("all_exact") is not True
        or (cold.get("cold_state_identity") or {}).get("array_count") != 60
        or qke_backend.get("proof_sha256") != h5.QKE_BACKEND_CANONICAL_SHA256
        or _canonical(qke_backend) != h5.QKE_BACKEND_CANONICAL_SHA256
        or qke_oracle.get("proof_sha256") != h5.QKE_CPU_ORACLE_CANONICAL_SHA256
        or _canonical(qke_oracle) != h5.QKE_CPU_ORACLE_CANONICAL_SHA256
    ):
        raise RuntimeError("inherited H5 admission proof semantics changed")

    ice_gpu = json.loads(ICE_GPU_PROOF.read_text())
    ice_baseline = json.loads(ICE_BASELINE_PROOF.read_text())
    if (
        ice_gpu.get("canonical_sha256") != ICE_GPU_PROOF_CANONICAL_SHA256
        or _canonical_without(ice_gpu, "canonical_sha256")
        != ICE_GPU_PROOF_CANONICAL_SHA256
        or ice_gpu.get("verdict") != "OPUS_LATE_NI_GPU_CONFIRM_GREEN"
        or ice_gpu.get("fix_present_in_tree") is not True
        or ice_gpu.get("output_all_leaves_finite") is not True
        or ice_gpu.get("gpu_actions") != 1
        or ice_baseline.get("canonical_sha256")
        != ICE_BASELINE_PROOF_CANONICAL_SHA256
        or _canonical_without(ice_baseline, "canonical_sha256")
        != ICE_BASELINE_PROOF_CANONICAL_SHA256
        or ice_baseline.get("fraction_of_ice_cells_changed")
        != 0.981376200744952
    ):
        raise RuntimeError("ice-fix proof semantics changed")

    return {
        "runner_head": head,
        "worktree_clean": not bool(dirty),
        "original_h5_model_commit": ORIGINAL_MODEL_COMMIT,
        "candidate_model_commit": CANDIDATE_MODEL_COMMIT,
        "candidate_src_gpuwrf_tree": src_tree,
        "source_delta_from_original_h5": list(source_delta),
        "candidate_source_sha256": source_rows,
        "fixture_input_sha256": input_rows,
        "runtime_authority": {
            "root": str(RUNTIME_AUTHORITY.resolve()),
            "files": runtime_rows,
        },
        "cpu_step9000_sha256": h5.CPU_STEP9000_SHA256,
        "original_h5_frame_pair_sha256": h5.REFERENCE_PAIR_SHA256,
        "case_metadata_sha256": h5.CASE_METADATA_SHA256,
        "autotune_pin_sha256": h5.AUTOTUNE_PIN_SHA256,
        "retained_step0_sha256": h5.RETAINED_STEP0_SHA256,
        "inherited_h5_proofs": {
            "c1_file_sha256": h5.C1_PROOF_SHA256,
            "cold_step0_file_sha256": h5.COLD_STEP0_ANCHOR_PROOF_SHA256,
            "qke_backend_file_sha256": h5.QKE_BACKEND_PROOF_SHA256,
            "qke_cpu_oracle_file_sha256": h5.QKE_CPU_ORACLE_PROOF_SHA256,
        },
        "ice_fix_proofs": {
            "gpu_confirmation_file_sha256": ICE_GPU_PROOF_FILE_SHA256,
            "gpu_confirmation_canonical_sha256": ICE_GPU_PROOF_CANONICAL_SHA256,
            "baseline_impact_file_sha256": ICE_BASELINE_PROOF_FILE_SHA256,
            "baseline_impact_canonical_sha256": ICE_BASELINE_PROOF_CANONICAL_SHA256,
        },
        "compute_pid_filter_sha256": COMPUTE_PID_FILTER_SHA256,
    }


def _expected_schedules() -> dict[str, tuple[int, ...]]:
    return {
        "d01": tuple((hour * 3600 + 53) // 54 for hour in range(1, 16)),
        "d02": tuple(range(200, 3001, 200)),
        "d03": tuple(range(200, 9001, 200)),
    }


def _read_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    if not CPU_PREFLIGHT.is_file() or CPU_PREFLIGHT.is_symlink():
        raise RuntimeError("CPU preflight is absent, non-file, or symlinked")
    payload = json.loads(CPU_PREFLIGHT.read_text())
    canonical = _canonical(payload)
    if (
        payload.get("schema") != "gpuwrf.v0234.v10-reconfirm-cpu-preflight.v1"
        or payload.get("verdict") != "V10_RECONFIRM_STEP9000_RUNNER_READY"
        or payload.get("proof_sha256") != canonical
        or payload.get("runner_source_sha256") != _sha256(Path(__file__).resolve())
        or payload.get("model_steps") != 0
        or payload.get("gpu_commands") != 0
        or payload.get("gpu_queries") != 0
    ):
        raise RuntimeError("CPU preflight semantics or canonical hash changed")
    return payload, {
        "path": str(CPU_PREFLIGHT.resolve()),
        "file_sha256": _sha256(CPU_PREFLIGHT),
        "canonical_sha256": canonical,
    }


def _validate_authorization_payload(payload: Mapping[str, Any]) -> str:
    nonce = payload.get("nonce")
    if not isinstance(nonce, str):
        raise RuntimeError("manager authorization has no string nonce")
    namespace = namespace_for_nonce(nonce)
    _, audit_row = _read_nonce_audit()
    _, preflight_row = _read_preflight()
    required = {
        "schema": "gpuwrf.v0234.v10-reconfirm-manager-authorization.v1",
        "verdict": "MANAGER_GPU_AUTHORIZED",
        "manager_pane": "0:1",
        "nonce": nonce,
        "namespace": namespace,
        "lock_label": LOCK_LABEL,
        "candidate_model_commit": CANDIDATE_MODEL_COMMIT,
        "candidate_src_gpuwrf_tree": CANDIDATE_SRC_TREE,
        "source_delta_from_original_h5": list(EXPECTED_SOURCE_DELTA),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "cpu_preflight_canonical_sha256": preflight_row["canonical_sha256"],
        "consumed_nonce_audit_canonical_sha256": audit_row["canonical_sha256"],
        "authorized_arm_count": 1,
        "authorized_model_processes": 1,
        "root_step_upper_bound": ROOT_STEP_END,
        "d03_step_upper_bound": D03_STEP_END,
        "full_output_counts": EXPECTED_OUTPUT_COUNTS,
        "d03_steps_beyond_9000_authorized": False,
        "stop_exactly_at_step9000": True,
        "baseline_rerun_authorized": False,
        "existing_cpu_reference_only": True,
        "reservation_seconds": EXPECTED_TIMING["hard_model_process_timeout_seconds"],
        "compile_measurement_estimate": EXPECTED_TIMING,
        "authorized_prelaunch_compute_process_query": True,
        "compute_pid_filter_sha256": COMPUTE_PID_FILTER_SHA256,
        "preemption_policy": _expected_preemption_policy(nonce),
    }
    canonical = _canonical(payload)
    if payload.get("proof_sha256") != canonical or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError(
            f"manager authorization mismatch: embedded={payload.get('proof_sha256')} "
            f"canonical={canonical}"
        )
    gate = payload.get("scientific_gate") or {}
    if (
        gate.get("evaluated_at_d03_step") != D03_STEP_END
        or gate.get("strict_rmse_ceilings") != h5.REFERENCE_RMSE
        or gate.get("finite_strict_fields_required") is not True
        or gate.get("static_identity_required") != list(h5.STATIC_FIELDS)
        or gate.get("materiality_scale") != MATERIALITY_SCALE
    ):
        raise RuntimeError("manager scientific gate drifted")
    autotune = payload.get("autotune_authority") or {}
    if (
        autotune.get("path") != str(AUTOTUNE_PIN)
        or autotune.get("sha256") != h5.AUTOTUNE_PIN_SHA256
        or "read-only" not in str(autotune.get("policy", ""))
    ):
        raise RuntimeError("manager autotune authority drifted")
    if not isinstance(payload.get("manager_decision"), str) or not payload[
        "manager_decision"
    ].strip():
        raise RuntimeError("manager decision text is absent")
    return canonical


def _assert_real_gpu_preemption_clear(
    label: str, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    nonce = authorization.get("nonce")
    if not isinstance(nonce, str) or authorization.get(
        "preemption_policy"
    ) != _expected_preemption_policy(nonce):
        raise RuntimeError(f"{label}: CPU-only Nightly authority is absent or drifted")
    present = [
        str(path)
        for path in REAL_PREEMPT_PATHS
        if path.exists() or path.is_symlink()
    ]
    if present:
        raise RuntimeError(f"{label}: preemption sentinel present: {present}")
    lock = _assert_locked_gpu_environment(nonce)["lock"]
    nightly = None
    nightly_file_sha256 = None
    if h5.NIGHTLY_ACTIVE.exists():
        nightly_file_sha256 = _sha256(h5.NIGHTLY_ACTIVE)
        nightly = json.loads(h5.NIGHTLY_ACTIVE.read_text())
    return {
        "label": label,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "preemption_sentinels_absent": True,
        "nightly_active": None if nightly is None else nightly.get("active"),
        "nightly_status": None if nightly is None else nightly.get("status"),
        "nightly_file_sha256": nightly_file_sha256,
        "coarse_nightly_status_informational": bool(
            nightly is not None and nightly.get("active") is True
        ),
        "direct_gpu_authority": "0:3",
        "nightly_gpu_chunks_enabled": False,
        "real_gpu_contention_clear": True,
        "lock": lock,
    }


def _read_authorization(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.parent.resolve() != SPRINT.resolve():
        raise RuntimeError("manager authorization must be inside this sprint")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("manager authorization is absent, non-file, or symlinked")
    payload = json.loads(path.read_text())
    nonce = payload.get("nonce", "")
    if path.name != f"GPU_STEP9000_AUTHORIZATION_{str(nonce)[:16]}.json":
        raise RuntimeError("manager authorization filename/nonce mismatch")
    canonical = _validate_authorization_payload(payload)
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
        "canonical_sha256": canonical,
    }


def _assert_fresh_namespace(run_dir: Path) -> None:
    if run_dir.exists() or run_dir.is_symlink():
        raise RuntimeError(f"authorized namespace is not fresh: {run_dir}")


def _assert_locked_gpu_environment(nonce: str) -> dict[str, Any]:
    required = {
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "true",
        "CUDA_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "GPUWRF_ALLOCATOR": "cuda_async",
        "GPUWRF_FINITE_CHECK": "1",
        "GPUWRF_NESTED_FUSE": "0",
        "GPUWRF_NESTED_DEFUSE_COMPILE": "0",
        "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
        "GPUWRF_NESTED_AOT": "0",
        "GPUWRF_AOT_VERIFY": "0",
        "GPUWRF_NESTED_ASYNC_OUTPUT": "0",
        "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
        "GPUWRF_FULL_WRFOUT": "1",
        "GPUWRF_TRAINING_OUTPUT_SUBSET": "0",
        "GPUWRF_BATCH_ENSEMBLE": "1",
        "GPUWRF_NESTED_SYNC_MODE": "root",
        "GPUWRF_BITWISE": "1",
        "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
        "GPUWRF_JAX_CACHE": "0",
        "GPUWRF_JAX_CACHE_LOCK": "0",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
        "GPUWRF_WRF_ROOT": str(RUNTIME_AUTHORITY),
        "GPUWRF_V10_RECONFIRM_NONCE": nonce,
        "GPUWRF_V10_COMPUTE_PID_PREFLIGHT": "clear",
        "XLA_FLAGS": f"--xla_gpu_load_autotune_results_from={AUTOTUNE_PIN}",
    }
    actual = {name: os.environ.get(name) for name in required}
    if actual != required:
        raise RuntimeError(f"locked GPU environment mismatch: {actual!r}")
    affinity = sorted(os.sched_getaffinity(0))
    if affinity != EXPECTED_AFFINITY:
        raise RuntimeError(f"CPU affinity mismatch: {affinity}")
    lock_env = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": "9",
        "GPUWRF_GPU_LOCK_FILE": "/tmp/wrf_gpu2_gpu.lock",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": "/tmp/wrf_gpu2_gpu.lock.holder",
        "GPUWRF_GPU_LOCK_LABEL": LOCK_LABEL,
    }
    if any(os.environ.get(name) != value for name, value in lock_env.items()):
        raise RuntimeError("live flock environment mismatch")
    token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN", "")
    if not token:
        raise RuntimeError("live flock token absent")
    compute_pid_query_sha256 = os.environ.get(
        "GPUWRF_V10_COMPUTE_PID_QUERY_SHA256", ""
    )
    if re.fullmatch(r"[0-9a-f]{64}", compute_pid_query_sha256) is None:
        raise RuntimeError("in-lock compute-PID query evidence is absent or malformed")
    lock_stat = Path(lock_env["GPUWRF_GPU_LOCK_FILE"]).stat()
    fd_stat = os.fstat(9)
    if (lock_stat.st_dev, lock_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino):
        raise RuntimeError("fd9 is not the shared GPU lock file")
    holder = Path(lock_env["GPUWRF_GPU_LOCK_HOLDER_FILE"]).read_text()
    if f"holder={LOCK_LABEL} " not in holder or f"token={token} " not in holder:
        raise RuntimeError("live holder sidecar differs from inherited lease")
    return {
        "environment": actual,
        "cpu_affinity": affinity,
        "lock": {
            "file": lock_env["GPUWRF_GPU_LOCK_FILE"],
            "holder_file": lock_env["GPUWRF_GPU_LOCK_HOLDER_FILE"],
            "label": LOCK_LABEL,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "fd9_inode_verified": True,
            "prelaunch_compute_pid_query": "clear",
            "prelaunch_compute_pid_query_sha256": compute_pid_query_sha256,
        },
    }


def _terminal_comparison(score: Mapping[str, Any]) -> dict[str, Any]:
    candidate = score["decision"]["strict_rmse"]
    fields = {}
    for field in sorted(ORIGINAL_H5_RMSE):
        old = ORIGINAL_H5_RMSE[field]
        new = float(candidate[field])
        fields[field] = {
            "original_ice_unfixed_h5_rmse": old,
            "ice_fixed_candidate_rmse": new,
            "absolute_movement": new - old,
            "movement_percent": 100.0 * (new / old - 1.0),
            "frozen_ceiling": h5.REFERENCE_RMSE[field],
            "margin_to_frozen_ceiling": h5.REFERENCE_RMSE[field] - new,
            "inside_frozen_ceiling": new <= h5.REFERENCE_RMSE[field],
        }
    correlations = {}
    for field, old in ORIGINAL_H5_CORRELATIONS.items():
        new = float(score["d03_spatial_pearson_correlation"][field])
        correlations[field] = {
            "original_ice_unfixed_h5": old,
            "ice_fixed_candidate": new,
            "absolute_movement": new - old,
            "movement_percent": 100.0 * (new / old - 1.0),
        }
    materially_unchanged = all(
        abs(fields[field]["absolute_movement"])
        <= MATERIALITY_SCALE["absolute_rmse_m_s"]
        and abs(fields[field]["movement_percent"])
        <= MATERIALITY_SCALE["relative_percent"]
        for field in MATERIALITY_SCALE["fields"]
    )
    gate_passed = bool(score["decision"]["passed"])
    if not gate_passed:
        verdict = "V10_REGRESSES_AND_FAILS_GATE"
    elif materially_unchanged:
        verdict = "V10_MATERIALLY_UNCHANGED_AFTER_ICE_FIX"
    elif fields["V10"]["absolute_movement"] > 0.0:
        verdict = "V10_REGRESSES_BUT_REMAINS_WITHIN_GATE"
    else:
        verdict = "V10_REMAINS_GREEN_AFTER_ICE_FIX"
    return {
        "verdict": verdict,
        "all_frozen_fields_inside_gate": gate_passed,
        "fields": fields,
        "correlations": correlations,
        "materiality_scale": MATERIALITY_SCALE,
        "V_and_V10_materially_unchanged": materially_unchanged,
    }


def _cpu_preflight(output: Path) -> int:
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise RuntimeError("CPU preflight imported JAX/gpuwrf")
    authority = _assert_model_and_fixture_authority(require_clean=False)
    _, audit_row = _read_nonce_audit()
    schedules = _expected_schedules()
    counts = {name: len(values) + 1 for name, values in schedules.items()}
    if counts != EXPECTED_OUTPUT_COUNTS or schedules["d01"][-1] != ROOT_STEP_END:
        raise RuntimeError(f"terminal output schedule drifted: {schedules}")
    proof = {
        "schema": "gpuwrf.v0234.v10-reconfirm-cpu-preflight.v1",
        "verdict": "V10_RECONFIRM_STEP9000_RUNNER_READY",
        "authority": authority,
        "consumed_nonce_audit": audit_row,
        "authorization_boundary": {
            "fresh_manager_authorization_required": True,
            "consumed_nonces_rejected": sorted(CONSUMED_NONCES),
            "fresh_namespace_derived_only_from_manager_nonce": True,
            "one_candidate_process": True,
            "baseline_rerun": False,
            "step_beyond_9000": False,
            "full_output_counts": EXPECTED_OUTPUT_COUNTS,
        },
        "post_step0_alarm_schedules": {
            name: list(values) for name, values in schedules.items()
        },
        "strict_rmse_ceilings": h5.REFERENCE_RMSE,
        "original_ice_unfixed_h5_rmse": ORIGINAL_H5_RMSE,
        "original_ice_unfixed_h5_correlations": ORIGINAL_H5_CORRELATIONS,
        "materiality_scale": MATERIALITY_SCALE,
        "compile_measurement_estimate": EXPECTED_TIMING,
        "preemption_mechanism": {
            "coarse_nightly_active_json_is_not_standalone_gpu_contention": True,
            "authorization_policy_required": PREEMPTION_POLICY_BASE,
            "real_gpu_gates_remain_fail_closed": True,
            "compute_pid_filter_sha256": COMPUTE_PID_FILTER_SHA256,
        },
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "authorization_files_read": 0,
        "model_steps": 0,
        "gpu_commands": 0,
        "gpu_queries": 0,
        "jax_or_gpuwrf_imported": False,
    }
    proof["proof_sha256"] = _canonical(proof)
    _atomic_json(output, proof)
    print(json.dumps({"verdict": proof["verdict"], "proof_sha256": proof["proof_sha256"]}))
    return 0


def _gpu_run(authorization_path: Path, run_dir: Path) -> int:
    started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    authorization, authorization_row = _read_authorization(authorization_path)
    expected_run_dir = LINEAGE / authorization["namespace"]
    if run_dir.resolve() != expected_run_dir.resolve():
        raise RuntimeError(f"run-dir mismatch: expected={expected_run_dir} actual={run_dir}")
    authority = _assert_model_and_fixture_authority(require_clean=True)
    _assert_fresh_namespace(run_dir)
    environment = _assert_locked_gpu_environment(authorization["nonce"])
    preemption_pre = _assert_real_gpu_preemption_clear(
        "v10-reconfirm-step9000-pre", authorization
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        import jax
        import numpy as np
        from netCDF4 import Dataset

        if jax.default_backend() != "gpu" or any(
            device.platform != "gpu" for device in jax.devices()
        ):
            raise RuntimeError(f"CUDA backend unavailable: {jax.devices()!r}")
        import gpuwrf
        from gpuwrf.integration.nested_pipeline import (
            NestedPipelineConfig,
            _PerDomainWrfoutWriter,
            _load_domains,
            _nested_sync_mode_from_env,
            assert_state_finite_at_boundary,
            domain_names_for,
        )
        from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree

        gpuwrf_path = Path(gpuwrf.__file__).resolve()
        if ROOT not in gpuwrf_path.parents:
            raise RuntimeError(f"gpuwrf imported outside candidate worktree: {gpuwrf_path}")
        output_dir = run_dir / "gpu-output"
        output_dir.mkdir()
        config = NestedPipelineConfig(
            input_dir=INPUT_DIR,
            output_dir=run_dir / "unused-output",
            proof_dir=run_dir / "unused-pipeline-proof",
            hours=18,
            max_dom=3,
            feedback=False,
            emit_initial_history=True,
        )
        load_started = time.perf_counter()
        names = domain_names_for(3)
        hierarchy, bundles, metadata, run_start, dt_by_domain, carries = _load_domains(
            config, names
        )
        load_seconds = time.perf_counter() - load_started
        if names != ("d01", "d02", "d03") or dt_by_domain != {
            "d01": 54.0,
            "d02": 18.0,
            "d03": 6.0,
        }:
            raise RuntimeError(f"domain hierarchy changed: {names}/{dt_by_domain}")
        loaded_options = {
            name: [
                int(bundles[name].namelist.moist_adv_opt),
                int(bundles[name].namelist.scalar_adv_opt),
            ]
            for name in names
        }
        if loaded_options != {name: [1, 1] for name in names}:
            raise RuntimeError(f"candidate loader did not resolve 1/1: {loaded_options}")

        with h5.RETAINED_STEP0.open("rb") as stream:
            retained = pickle.load(stream)
        cold = h5._state_arrays(np, jax.device_get(carries["d03"]).state)
        retained_arrays = h5._state_arrays(np, retained.state)
        cold_anchor_proof = json.loads(h5.COLD_STEP0_ANCHOR_PROOF.read_text())
        qke_cpu_oracle = np.load(h5.QKE_CPU_ORACLE, allow_pickle=False)
        if h5._array_sha256(np, qke_cpu_oracle) != h5.QKE_CPU_ARRAY_SHA256:
            raise RuntimeError("loaded QKE CPU oracle content changed")
        cold_anchor = h5._validate_cold_step0_anchor(
            np, cold, retained_arrays, cold_anchor_proof, qke_cpu_oracle
        )

        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
        schedules = _expected_schedules()
        if {name: len(values) + 1 for name, values in schedules.items()} != EXPECTED_OUTPUT_COUNTS:
            raise RuntimeError(f"full-cadence schedule drifted: {schedules}")
        writer = _PerDomainWrfoutWriter(
            output_dir=output_dir,
            input_dir=INPUT_DIR,
            run_start=run_start,
            bundles=bundles,
            output_cadence_steps=OUTPUT_CADENCE,
            dt_by_domain=dt_by_domain,
        )
        initial_outputs = [writer(name, 0, carries[name]) for name in names]
        block_between, root_sync_cadence = _nested_sync_mode_from_env()
        model_started = time.perf_counter()
        own_steps = {name: 0 for name in names}
        segments = []
        while own_steps["d01"] < ROOT_STEP_END:
            segment_start = int(own_steps["d01"])
            segment_steps = min(OUTPUT_CADENCE["d01"], ROOT_STEP_END - segment_start)
            segment_preemption = _assert_real_gpu_preemption_clear(
                f"v10-reconfirm-before-root-{segment_start}", authorization
            )
            segment_wall_started = time.perf_counter()
            result = run_operational_domain_tree(
                tree,
                root_steps=segment_steps,
                feedback_enabled=False,
                output=writer,
                output_cadence_steps=OUTPUT_CADENCE,
                output_alarm_steps={"d01": schedules["d01"]},
                block_between=block_between,
                root_sync_cadence=root_sync_cadence,
                carries=carries,
                initial_own_steps=own_steps,
            )
            jax.block_until_ready(tuple(state.theta for state in result.states.values()))
            carries = result.carries
            own_steps = {name: int(result.own_steps[name]) for name in names}
            for name, state in result.states.items():
                assert_state_finite_at_boundary(
                    state,
                    domain=name,
                    step=own_steps[name],
                    sim_time_s=own_steps[name] * dt_by_domain[name],
                )
            segments.append(
                {
                    "root_start": segment_start,
                    "root_steps": segment_steps,
                    "own_steps_after": dict(own_steps),
                    "wall_seconds": time.perf_counter() - segment_wall_started,
                    "preemption": segment_preemption,
                }
            )

        model_seconds = time.perf_counter() - model_started
        if own_steps != {"d01": 1000, "d02": 3000, "d03": 9000}:
            raise RuntimeError(f"step9000 terminal clock drifted: {own_steps}")
        candidate_frame = output_dir / "wrfout_d03_2025-03-01_15:00:00"
        if not candidate_frame.is_file() or candidate_frame.is_symlink():
            raise RuntimeError(f"step9000 frame missing: {candidate_frame}")
        score = h5._score_frame(np, Dataset, candidate_frame)
        comparison = _terminal_comparison(score)
        score_finished_utc = datetime.now(timezone.utc).isoformat()

        counts = {name: len(writer.written[name]) for name in names}
        if counts != EXPECTED_OUTPUT_COUNTS:
            raise RuntimeError(f"full-cadence output counts drifted: {counts}")
        case_metadata = json.loads(h5.CASE_METADATA.read_text())
        expected_variables = case_metadata["planned_full_output"]["variables"]
        if len(expected_variables) != 375 or len(set(expected_variables)) != 375:
            raise RuntimeError("sealed 375-variable inventory changed")
        frame_manifest = []
        for name in names:
            for raw_path in writer.written[name]:
                path = Path(raw_path)
                if not path.is_file() or path.is_symlink():
                    raise RuntimeError(f"missing or symlink output: {path}")
                with Dataset(str(path)) as frame:
                    observed_variables = list(frame.variables)
                if observed_variables != expected_variables:
                    raise RuntimeError(f"full variable inventory drifted: {path}")
                frame_manifest.append(
                    {
                        "domain": name,
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "variable_count": len(observed_variables),
                    }
                )
        preemption_post = _assert_real_gpu_preemption_clear(
            "v10-reconfirm-step9000-post", authorization
        )
        lock_post = _assert_locked_gpu_environment(authorization["nonce"])["lock"]
        proof = {
            "schema": "gpuwrf.v0234.v10-reconfirm-step9000-gpu-result.v1",
            "verdict": comparison["verdict"],
            "authorization": authorization_row,
            "authorization_nonce": authorization["nonce"],
            "namespace": str(run_dir.resolve()),
            "authority": authority,
            "environment": environment,
            "cuda_runtime": {
                "jax_version": jax.__version__,
                "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
            },
            "fixture": {
                "input_dir": str(INPUT_DIR.resolve()),
                "run_start": run_start.isoformat(),
                "terminal": "2025-03-01T15:00:00+00:00",
                "domains": list(names),
                "dt_by_domain": dt_by_domain,
                "loaded_moist_scalar_options": loaded_options,
                "metadata_scalar_options": {
                    name: {
                        "moist_adv_opt": metadata["domains"][name]["namelist"][
                            "moist_adv_opt"
                        ],
                        "scalar_adv_opt": metadata["domains"][name]["namelist"][
                            "scalar_adv_opt"
                        ],
                    }
                    for name in names
                },
                "cold_step0_anchor": cold_anchor,
            },
            "score": score,
            "terminal_comparison": comparison,
            "full_case_through_step9000": {
                "initial_outputs": initial_outputs,
                "post_step0_alarm_schedules": {
                    name: list(values) for name, values in schedules.items()
                },
                "output_counts": counts,
                "expected_output_counts": EXPECTED_OUTPUT_COUNTS,
                "final_own_steps": own_steps,
                "variable_count_each_frame": 375,
                "frame_count": len(frame_manifest),
                "frame_manifest": frame_manifest,
            },
            "bounded_execution": {
                "candidate_processes": 1,
                "baseline_processes": 0,
                "root_step_upper_bound": ROOT_STEP_END,
                "d03_step_upper_bound": D03_STEP_END,
                "d03_steps_beyond_9000_dispatched": 0,
                "segments": segments,
                "autotune_pin_read_only": str(AUTOTUNE_PIN.resolve()),
                "autotune_pin_sha256": h5.AUTOTUNE_PIN_SHA256,
            },
            "preemption": {"pre": preemption_pre, "post": preemption_post},
            "lock_post": lock_post,
            "timing": {
                "started_utc": started.isoformat(),
                "score_finished_utc": score_finished_utc,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "domain_load_seconds": load_seconds,
                "model_through_step9000_seconds": model_seconds,
                "wall_seconds": time.perf_counter() - wall_started,
                "compile_measurement_estimate_from_prior_arms": EXPECTED_TIMING,
                "actual_compile_measurement_separately_instrumented": False,
            },
            "gpu_accounting": {
                "lock_acquisitions": 1,
                "candidate_model_processes": 1,
                "baseline_model_processes": 0,
                "gpu_queries_inside_runner": 0,
                "wrapper_acquire_gpu_summary_query": 1,
                "in_lock_prelaunch_compute_pid_query": 1,
            },
            "scientific_falsification": not bool(score["decision"]["passed"]),
        }
        proof["proof_sha256"] = _canonical(proof)
        result_path = run_dir / "v10-reconfirm-step9000-result.json"
        _atomic_json(result_path, proof)
        decision = score["decision"]
        print(
            json.dumps(
                {
                    "milestone": "ICE_FIXED_STEP9000_SCORED_AND_STOPPED",
                    "verdict": proof["verdict"],
                    "V": decision["strict_rmse"]["V"],
                    "V10": decision["strict_rmse"]["V10"],
                    "correlations": score["d03_spatial_pearson_correlation"],
                    "movement": {
                        "V": comparison["fields"]["V"],
                        "V10": comparison["fields"]["V10"],
                    },
                    "proof": str(result_path.resolve()),
                    "proof_sha256": proof["proof_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if score["decision"]["passed"] else 3
    except BaseException as exc:
        blocker = {
            "schema": "gpuwrf.v0234.v10-reconfirm-step9000-gpu-blocker.v1",
            "verdict": "V10_RECONFIRMATION_HARNESS_OR_RUNTIME_BLOCKED",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "authorization": authorization_row,
            "namespace": str(run_dir.resolve()),
            "candidate_model_commit": CANDIDATE_MODEL_COMMIT,
            "started_utc": started.isoformat(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - wall_started,
            "scientific_falsification": False,
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(run_dir / "v10-reconfirm-step9000-blocker.json", blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--proof-output", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.cpu_preflight:
        if args.proof_output is None or args.authorization is not None or args.run_dir is not None:
            parser.error(
                "--cpu-preflight requires --proof-output and forbids authorization/run-dir"
            )
        return _cpu_preflight(args.proof_output.resolve())
    if args.authorization is None or args.run_dir is None or args.proof_output is not None:
        parser.error("GPU mode requires authorization/run-dir and forbids proof-output")
    return _gpu_run(args.authorization.resolve(), args.run_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
