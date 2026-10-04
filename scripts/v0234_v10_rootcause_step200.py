"""Exact one-process Step0-to-d03-step200 V10 root-cause discriminator.

The module is stdlib-only until admission and the locked CUDA environment have
been authenticated.  The domain-tree run requests one d03 alarm at step 200;
the synchronous output callback writes that frame and raises a private stop
signal before step 201 can be dispatched.  The retained reference is never
rerun.
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


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
LOCK_LABEL = "v0234-gpt-v10-rootcause"
CANDIDATE_COMMIT = "3b81fb5b093639e70c12cce87d602c45b326b18b"
CANDIDATE_SRC_TREE = "e627605f6a8bc0dc23f5c474be4bb532b99297c1"
CANDIDATE_SOURCE_SHA256 = {
    "src/gpuwrf/integration/nested_pipeline.py": (
        "77b6a690647ec4b0b615f425338ca41f3e594c29566eeafc0f9e884e1f0e98c4"
    ),
    "src/gpuwrf/runtime/operational_mode.py": (
        "d5ce22236188c4264ab13e3108f163830d9c5bce1560b0090db1c927e8e1a19d"
    ),
}
CONSUMED_NONCES = frozenset(
    {
        "ad0e60072a2a0df1ceacf0f5593e3834f2d0da10086f46316375f8f17aa",
        "8520924a5f97acc1ea8337bb16ea860903b8436e2d30b10b6efca9d54f55552c",
        "ca1df30181823579e35b5c3dbadfa09782cfde6104efe3793937ed9d38e02f2a",
        "fbb2716e855e4fe572d1bbef538f8b26ce5f4a106a4ed8f7b44b41400e852687",
    }
)
NONCE_PATTERN = re.compile(r"[0-9a-f]{32,128}")
C1_PROOF = SPRINT / "C1_CANDIDATE_OFF_PROOF.json"
C1_PROOF_SHA256 = (
    "59fe04842dbc4b8471547b47b646d01774259d0d0fca9024bd87fcc3e8ea8302"
)
C1_CANONICAL_SHA256 = (
    "099c63a25dddff5eb652d2850c4520138b172b9c28f81b98c25dbf3b62ebf8a6"
)
COLD_STEP0_ANCHOR_PROOF = SPRINT / "COLD_STEP0_ANCHOR_REPAIR.json"
COLD_STEP0_ANCHOR_PROOF_SHA256 = (
    "e98c689ad9aab02a59e4eb4896e92bb4b34a7cd85371d766c332e0df6a8beeef"
)
COLD_STEP0_ANCHOR_CANONICAL_SHA256 = (
    "eada51b566dfe9497261226fd6e29b43278a8a9bf0f4e4d77e46325ca2d3aaad"
)
QKE_BACKEND_PROOF = SPRINT / "NOCTURNAL_STEP1_QKE_BACKEND.json"
QKE_BACKEND_PROOF_SHA256 = (
    "cb596f27c327d55b7f55b88c2982739d424d5bd7a39e72d98f288250760ff0c3"
)
QKE_BACKEND_CANONICAL_SHA256 = (
    "680a4cab02f7198b9d04c9864420a5a24bb2997a04ebb2ab86ae09f8272e7a0e"
)
QKE_CPU_ORACLE_PROOF = SPRINT / "QKE_CPU_ORACLE.json"
QKE_CPU_ORACLE_PROOF_SHA256 = (
    "6db8ce8f2fb6ce5287b81a7f55359376262a640f1efa9b0c4e2922bbef28fe47"
)
QKE_CPU_ORACLE_CANONICAL_SHA256 = (
    "8b9d9176cbd1e30c264a1097074070ef7630513aab9718de6137ab7d47e30d87"
)
QKE_CPU_ORACLE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "v0234_gpt_v10_qke_cpu_oracle_680a4cab02f7198b/"
    "qke-current-cpu.npy"
)
QKE_CPU_ORACLE_SHA256 = (
    "0d6f91e42771e1920230c8e226ad9f3454e5365201755fc4fa4110a5b62cfb38"
)
QKE_CPU_ARRAY_SHA256 = (
    "07ea2a840d9a6793eb2822640eb75dd5c7d7adacf7bf11ef96020ab7f0f5ac95"
)
QKE_BACKEND_RMS_LIMIT = 1.715416517114786e-06
QKE_BACKEND_MAX_ABS_LIMIT = 0.0006439058793601404
EXPECTED_RETAINED_STEP0_MISMATCHES = ("mavail", "qke", "roughness_m")

CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
INPUT_DIR = CASE_ROOT / "run/wrf"
LINEAGE = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a"
CPU_STEP200 = INPUT_DIR / "wrfout_d03_2025-03-01_00:20:00"
CPU_STEP200_SHA256 = (
    "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1"
)
REFERENCE_PAIR = (
    LINEAGE
    / "v0234_gpt_v10_replay_c17fca1e201ae106_reference/"
    "frame-pairs/d03-step-00200.json"
)
REFERENCE_PAIR_SHA256 = (
    "9cb9060bbcc70db958f22f5f06a1a8268e79a9d2a1c8f7a52f62e194cc227896"
)
AUTOTUNE_PIN = (
    LINEAGE
    / "v0234_gpt_v10_replay_c17fca1e201ae106_reference/autotune-results.pb"
)
AUTOTUNE_PIN_SHA256 = (
    "6a0f30bc8e2ab1ca646ab346f85221565149f4a63b04193a9d76dc334e57ce18"
)
RETAINED_STEP0 = (
    LINEAGE
    / "nested_scalar_sixth_order_18d97595_full18h_discriminator1/"
    "failure/last-healthy-d03-step-0.pkl"
)
RETAINED_STEP0_SHA256 = (
    "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
)
RUNTIME_AUTHORITY = LINEAGE / "v0234_v10_runtime_authority_complete1_5a6298fb"
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
INPUT_SHA256 = {
    "namelist.input": "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
STATIC_FIELDS = ("HGT", "LANDMASK", "XLAT", "XLONG")
REFERENCE_RMSE = {
    "PSFC": 12.354054650535346,
    "T": 0.07996348617198683,
    "T2": 0.09977545461428595,
    "U": 0.06409337452377663,
    "U10": 0.12635552531760158,
    "V": 0.08316516541247987,
    "V10": 0.1230026780020817,
    "W": 0.06776713034436521,
}
OTHER_FROZEN_FIELDS = ("T", "U", "W", "T2", "U10", "PSFC")
EXPECTED_AFFINITY = [13, 14, 15, 29, 30, 31]
PREEMPT_PATHS = (
    Path("/tmp/PREEMPT_GPU"),
    Path("/tmp/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_GPU"),
)
NIGHTLY_ACTIVE = Path("<DATA_ROOT>/alisios/state/nightly18z/active.json")


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


def _namespace_for_nonce(nonce: str) -> str:
    if NONCE_PATTERN.fullmatch(nonce) is None:
        raise RuntimeError("manager nonce must be 32--128 lowercase hex characters")
    if nonce in CONSUMED_NONCES:
        raise RuntimeError(f"manager nonce is already consumed: {nonce}")
    return f"v0234_gpt_v10_rootcause_{nonce[:16]}_step200"


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


def classify_step200(
    strict_rmse: Mapping[str, float],
    *,
    finite: bool,
    static_identity: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the preregistered dual-wind and no-regression policy."""

    exact = set(strict_rmse) == set(STRICT_FIELDS)
    values_finite = exact and all(math.isfinite(float(value)) for value in strict_rmse.values())
    frozen_regressions = {
        field: {
            "candidate": float(strict_rmse.get(field, math.nan)),
            "reference": REFERENCE_RMSE[field],
        }
        for field in OTHER_FROZEN_FIELDS
        if field not in strict_rmse
        or not math.isfinite(float(strict_rmse[field]))
        or float(strict_rmse[field]) > REFERENCE_RMSE[field]
    }
    v_green = bool(
        "V" in strict_rmse
        and math.isfinite(float(strict_rmse["V"]))
        and float(strict_rmse["V"]) < REFERENCE_RMSE["V"]
    )
    v10_green = bool(
        "V10" in strict_rmse
        and math.isfinite(float(strict_rmse["V10"]))
        and float(strict_rmse["V10"]) < REFERENCE_RMSE["V10"]
    )
    static_green = set(static_identity) == set(STATIC_FIELDS) and all(
        bool(static_identity[field]) for field in STATIC_FIELDS
    )
    passed = bool(
        finite
        and values_finite
        and static_green
        and not frozen_regressions
        and v_green
        and v10_green
    )
    return {
        "passed": passed,
        "classification": (
            "H5_STEP200_STRICT_GREEN" if passed else "H5_STEP200_SCIENTIFIC_RED"
        ),
        "V_strict_improvement": v_green,
        "V10_strict_improvement": v10_green,
        "finite": bool(finite and values_finite),
        "static_identity": dict(static_identity),
        "static_identity_pass": static_green,
        "frozen_field_regressions": frozen_regressions,
        "strict_rmse": {field: float(strict_rmse[field]) for field in sorted(strict_rmse)},
        "reference_rmse": dict(REFERENCE_RMSE),
        "tolerance_changed": False,
        "baseline_rerun": False,
        "horizon_extension": False,
    }


def _validate_authorization_payload(payload: Mapping[str, Any]) -> str:
    embedded = payload.get("proof_sha256")
    canonical = _canonical(payload)
    nonce = payload.get("nonce")
    if not isinstance(nonce, str):
        raise RuntimeError("manager authorization has no string nonce")
    namespace = _namespace_for_nonce(nonce)
    required = {
        "schema": "gpuwrf.v0234.v10-rootcause-step200-manager-authorization.v1",
        "verdict": "MANAGER_GPU_AUTHORIZED",
        "manager_pane": "0:1",
        "nonce": nonce,
        "namespace": namespace,
        "lock_label": LOCK_LABEL,
        "candidate_model_commit": CANDIDATE_COMMIT,
        "candidate_src_gpuwrf_tree": CANDIDATE_SRC_TREE,
        "authorized_arm_count": 1,
        "authorized_model_processes": 1,
        "root_step_upper_bound": 23,
        "d03_steps_beyond_200_authorized": False,
        "baseline_rerun_authorized": False,
        "horizon_extension_authorized": False,
        "existing_cpu_reference_only": True,
    }
    if embedded != canonical or any(
        payload.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError(
            f"manager authorization mismatch: embedded={embedded} canonical={canonical}"
        )
    gate = payload.get("scientific_gate") or {}
    if (
        gate.get("V_strict_less_than") != REFERENCE_RMSE["V"]
        or gate.get("V10_strict_less_than") != REFERENCE_RMSE["V10"]
        or gate.get("other_strict_fields_no_worse_than_existing_reference") is not True
        or gate.get("finite_and_static_identity_required") is not True
    ):
        raise RuntimeError("manager scientific gate drifted")
    autotune = payload.get("autotune_authority") or {}
    if (
        autotune.get("path") != str(AUTOTUNE_PIN)
        or autotune.get("sha256") != AUTOTUNE_PIN_SHA256
        or "read-only" not in str(autotune.get("policy", ""))
    ):
        raise RuntimeError("manager autotune authority drifted")
    return canonical


def _read_authorization(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.parent.resolve() != SPRINT.resolve():
        raise RuntimeError(f"manager authorization must be inside sprint directory: {path}")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("manager authorization is absent, non-file, or symlinked")
    payload = json.loads(path.read_text())
    canonical = _validate_authorization_payload(payload)
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
        "canonical_sha256": canonical,
    }


def _assert_preemption_clear(label: str) -> dict[str, Any]:
    present = [str(path) for path in PREEMPT_PATHS if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"{label}: preemption sentinel present: {present}")
    nightly = None
    if NIGHTLY_ACTIVE.exists():
        nightly = json.loads(NIGHTLY_ACTIVE.read_text())
        if nightly.get("active") is True:
            raise RuntimeError(f"{label}: nightly production active")
    return {
        "label": label,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "preemption_sentinels_absent": True,
        "nightly_active": None if nightly is None else nightly.get("active"),
    }


def _assert_model_and_fixture_authority(*, require_clean: bool) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    ancestor = subprocess.run(
        ("git", "-C", str(ROOT), "merge-base", "--is-ancestor", CANDIDATE_COMMIT, head),
        check=False,
    ).returncode == 0
    src_tree = _git("rev-parse", "HEAD:src/gpuwrf")
    if require_clean and dirty:
        raise RuntimeError(f"worktree must be clean: {dirty}")
    if not ancestor or src_tree != CANDIDATE_SRC_TREE:
        raise RuntimeError(
            f"candidate authority changed: ancestor={ancestor} src_tree={src_tree}"
        )
    source_rows = {}
    for relative, expected in CANDIDATE_SOURCE_SHA256.items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"candidate source changed: {relative}: {actual}")
        source_rows[relative] = actual
    input_rows = {}
    for relative, expected in INPUT_SHA256.items():
        path = INPUT_DIR / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"fixture input changed: {relative}: {actual}")
        input_rows[relative] = actual
    exact_files = sorted(
        str(path.relative_to(RUNTIME_AUTHORITY))
        for path in RUNTIME_AUTHORITY.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if exact_files != sorted(RUNTIME_AUTHORITY_FILES):
        raise RuntimeError(f"runtime authority inventory changed: {exact_files}")
    runtime_rows = {}
    for relative, (expected_bytes, expected_sha) in RUNTIME_AUTHORITY_FILES.items():
        path = RUNTIME_AUTHORITY / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or _sha256(path) != expected_sha
        ):
            raise RuntimeError(f"runtime authority changed: {relative}")
        runtime_rows[relative] = {"bytes": expected_bytes, "sha256": expected_sha}
    for path, expected, code in (
        (CPU_STEP200, CPU_STEP200_SHA256, "CPU_STEP200"),
        (REFERENCE_PAIR, REFERENCE_PAIR_SHA256, "REFERENCE_PAIR"),
        (AUTOTUNE_PIN, AUTOTUNE_PIN_SHA256, "AUTOTUNE_PIN"),
        (RETAINED_STEP0, RETAINED_STEP0_SHA256, "RETAINED_STEP0"),
    ):
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise RuntimeError(f"{code} authority changed: {path}")
    if (
        not C1_PROOF.is_file()
        or C1_PROOF.is_symlink()
        or _sha256(C1_PROOF) != C1_PROOF_SHA256
    ):
        raise RuntimeError(f"C1 candidate-off proof authority changed: {C1_PROOF}")
    c1 = json.loads(C1_PROOF.read_text())
    if (
        c1.get("verdict")
        != "C1_REPAIRED__CANDIDATE_OFF_MATCHES_RELEASED_FULL_CARRY_BYTES"
        or c1.get("candidate_off_identity_pass") is not True
        or c1.get("canonical_payload_sha256") != C1_CANONICAL_SHA256
        or c1.get("operational_source_sha256")
        != CANDIDATE_SOURCE_SHA256["src/gpuwrf/runtime/operational_mode.py"]
    ):
        raise RuntimeError("C1 candidate-off proof semantics changed")
    if (
        not COLD_STEP0_ANCHOR_PROOF.is_file()
        or COLD_STEP0_ANCHOR_PROOF.is_symlink()
        or _sha256(COLD_STEP0_ANCHOR_PROOF) != COLD_STEP0_ANCHOR_PROOF_SHA256
    ):
        raise RuntimeError("cold Step0 anchor repair proof authority changed")
    cold_anchor = json.loads(COLD_STEP0_ANCHOR_PROOF.read_text())
    if (
        cold_anchor.get("proof_sha256") != COLD_STEP0_ANCHOR_CANONICAL_SHA256
        or _canonical(cold_anchor) != COLD_STEP0_ANCHOR_CANONICAL_SHA256
        or cold_anchor.get("verdict")
        != (
            "PINNED_BASE_AND_REPAIRED_CANDIDATE_COLD_STEP0_BYTE_EXACT__"
            "HISTORICAL_THREE_LEAF_DRIFT_AUTHORIZED"
        )
        or not all((cold_anchor.get("checks") or {}).values())
        or (cold_anchor.get("cold_state_identity") or {}).get("array_count") != 60
        or (cold_anchor.get("cold_state_identity") or {}).get("all_exact") is not True
        or tuple(cold_anchor.get("expected_historical_retained_mismatches") or ())
        != EXPECTED_RETAINED_STEP0_MISMATCHES
        or (cold_anchor.get("source_comparison") or {}).get("candidate_commit")
        != CANDIDATE_COMMIT
        or (cold_anchor.get("source_comparison") or {}).get(
            "candidate_src_gpuwrf_tree"
        )
        != CANDIDATE_SRC_TREE
    ):
        raise RuntimeError("cold Step0 anchor repair proof semantics changed")
    if (
        not QKE_BACKEND_PROOF.is_file()
        or QKE_BACKEND_PROOF.is_symlink()
        or _sha256(QKE_BACKEND_PROOF) != QKE_BACKEND_PROOF_SHA256
    ):
        raise RuntimeError("QKE backend discriminator proof authority changed")
    qke_backend = json.loads(QKE_BACKEND_PROOF.read_text())
    if (
        qke_backend.get("proof_sha256") != QKE_BACKEND_CANONICAL_SHA256
        or _canonical(qke_backend) != QKE_BACKEND_CANONICAL_SHA256
        or qke_backend.get("verdict")
        != "QKE_BACKEND_SEED_STEP1_UV_NEGLIGIBLE__HARNESS_GATE_REPAIRABLE"
        or qke_backend.get("max_primary_UV_RMSE_change") != 0.0
        or (qke_backend.get("input_qke_backend_delta") or {}).get("rms")
        != QKE_BACKEND_RMS_LIMIT
        or (qke_backend.get("input_qke_backend_delta") or {}).get("max_abs")
        != QKE_BACKEND_MAX_ABS_LIMIT
        or not all((qke_backend.get("checks") or {}).values())
    ):
        raise RuntimeError("QKE backend discriminator proof semantics changed")
    if (
        not QKE_CPU_ORACLE_PROOF.is_file()
        or QKE_CPU_ORACLE_PROOF.is_symlink()
        or _sha256(QKE_CPU_ORACLE_PROOF) != QKE_CPU_ORACLE_PROOF_SHA256
    ):
        raise RuntimeError("QKE CPU oracle proof authority changed")
    qke_oracle = json.loads(QKE_CPU_ORACLE_PROOF.read_text())
    qke_oracle_row = qke_oracle.get("oracle") or {}
    if (
        qke_oracle.get("proof_sha256") != QKE_CPU_ORACLE_CANONICAL_SHA256
        or _canonical(qke_oracle) != QKE_CPU_ORACLE_CANONICAL_SHA256
        or qke_oracle.get("verdict") != "CURRENT_TREE_JAX_CPU_QKE_ORACLE_SEALED"
        or qke_oracle_row.get("path") != str(QKE_CPU_ORACLE)
        or qke_oracle_row.get("file_sha256") != QKE_CPU_ORACLE_SHA256
        or qke_oracle_row.get("array_sha256") != QKE_CPU_ARRAY_SHA256
        or qke_oracle_row.get("shape") != [44, 93, 111]
        or qke_oracle_row.get("dtype") != "float64"
        or qke_oracle_row.get("finite") is not True
    ):
        raise RuntimeError("QKE CPU oracle proof semantics changed")
    if (
        not QKE_CPU_ORACLE.is_file()
        or QKE_CPU_ORACLE.is_symlink()
        or QKE_CPU_ORACLE.stat().st_size != 3_633_824
        or _sha256(QKE_CPU_ORACLE) != QKE_CPU_ORACLE_SHA256
    ):
        raise RuntimeError("QKE CPU oracle file authority changed")
    return {
        "runner_head": head,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_is_ancestor": ancestor,
        "candidate_src_gpuwrf_tree": src_tree,
        "candidate_source_sha256": source_rows,
        "worktree_clean": not bool(dirty),
        "fixture_input_sha256": input_rows,
        "runtime_authority": {
            "root": str(RUNTIME_AUTHORITY.resolve()),
            "files": runtime_rows,
        },
        "cpu_step200_sha256": CPU_STEP200_SHA256,
        "reference_pair_sha256": REFERENCE_PAIR_SHA256,
        "autotune_pin_sha256": AUTOTUNE_PIN_SHA256,
        "retained_step0_sha256": RETAINED_STEP0_SHA256,
        "c1_candidate_off_proof": {
            "path": str(C1_PROOF.resolve()),
            "file_sha256": C1_PROOF_SHA256,
            "canonical_payload_sha256": C1_CANONICAL_SHA256,
            "candidate_off_identity_pass": True,
        },
        "cold_step0_anchor_repair": {
            "path": str(COLD_STEP0_ANCHOR_PROOF.resolve()),
            "file_sha256": COLD_STEP0_ANCHOR_PROOF_SHA256,
            "canonical_sha256": COLD_STEP0_ANCHOR_CANONICAL_SHA256,
            "pinned_base_candidate_all_60_state_arrays_exact": True,
            "expected_historical_retained_mismatches": list(
                EXPECTED_RETAINED_STEP0_MISMATCHES
            ),
        },
        "qke_backend_discriminator": {
            "path": str(QKE_BACKEND_PROOF.resolve()),
            "file_sha256": QKE_BACKEND_PROOF_SHA256,
            "canonical_sha256": QKE_BACKEND_CANONICAL_SHA256,
            "max_primary_UV_RMSE_change": 0.0,
            "admission_rms_limit": QKE_BACKEND_RMS_LIMIT,
            "admission_max_abs_limit": QKE_BACKEND_MAX_ABS_LIMIT,
        },
        "qke_cpu_oracle": {
            "proof_path": str(QKE_CPU_ORACLE_PROOF.resolve()),
            "proof_file_sha256": QKE_CPU_ORACLE_PROOF_SHA256,
            "proof_canonical_sha256": QKE_CPU_ORACLE_CANONICAL_SHA256,
            "array_path": str(QKE_CPU_ORACLE.resolve()),
            "array_file_sha256": QKE_CPU_ORACLE_SHA256,
            "array_sha256": QKE_CPU_ARRAY_SHA256,
        },
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
        "GPUWRF_V10_ROOTCAUSE_NONCE": nonce,
        "XLA_FLAGS": f"--xla_gpu_load_autotune_results_from={AUTOTUNE_PIN}",
    }
    actual = {name: os.environ.get(name) for name in required}
    if actual != required:
        raise RuntimeError(f"locked GPU environment mismatch: {actual!r}")
    if sorted(os.sched_getaffinity(0)) != EXPECTED_AFFINITY:
        raise RuntimeError(f"CPU affinity mismatch: {sorted(os.sched_getaffinity(0))}")
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
    lock_stat = Path(lock_env["GPUWRF_GPU_LOCK_FILE"]).stat()
    fd_stat = os.fstat(9)
    if (lock_stat.st_dev, lock_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino):
        raise RuntimeError("fd9 is not the shared GPU lock file")
    holder = Path(lock_env["GPUWRF_GPU_LOCK_HOLDER_FILE"]).read_text()
    if f"holder={LOCK_LABEL} " not in holder or f"token={token} " not in holder:
        raise RuntimeError("live holder sidecar differs from inherited lease")
    return {
        "environment": actual,
        "cpu_affinity": EXPECTED_AFFINITY,
        "lock": {
            "file": lock_env["GPUWRF_GPU_LOCK_FILE"],
            "holder_file": lock_env["GPUWRF_GPU_LOCK_HOLDER_FILE"],
            "label": LOCK_LABEL,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "fd9_inode_verified": True,
        },
    }


class _Step200Reached(RuntimeError):
    pass


class _Step200Output:
    wants_carry = True

    def __init__(self, writer: Any) -> None:
        self.writer = writer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, name: str, step: int, carry: Any) -> Any:
        if name != "d03" or int(step) != 200:
            raise RuntimeError(f"unexpected output callback: {name} step {step}")
        value = self.writer(name, int(step), carry)
        self.calls.append({"domain": name, "own_step": int(step), "writer_value": value})
        raise _Step200Reached("synchronous d03 step-200 output is complete")


def _state_arrays(np: Any, state: Any) -> dict[str, Any]:
    arrays = {}
    for name in sorted(dir(state)):
        if name.startswith("_"):
            continue
        value = getattr(state, name, None)
        if value is None or callable(value) or not hasattr(value, "shape"):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            arrays[name] = array
    return arrays


def _array_sha256(np: Any, array: Any) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_cold_step0_anchor(
    np: Any,
    cold: Mapping[str, Any],
    retained: Mapping[str, Any],
    proof: Mapping[str, Any],
    qke_cpu_oracle: Any,
) -> dict[str, Any]:
    expected = proof["repaired_candidate"]["cold_state"]["arrays"]
    cold_names = set(cold)
    retained_names = set(retained)
    expected_names = set(expected)
    if cold_names != retained_names or cold_names != expected_names:
        raise RuntimeError(
            "cold Step0 floating state inventory changed: "
            f"cold_only={sorted(cold_names - expected_names)} "
            f"expected_only={sorted(expected_names - cold_names)} "
            f"retained_only={sorted(retained_names - cold_names)}"
        )
    gpu_cpu_mismatches = []
    nonfinite = []
    observed_rows = {}
    for name in sorted(cold):
        array = np.asarray(cold[name])
        row = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _array_sha256(np, array),
            "finite": bool(np.all(np.isfinite(array))),
        }
        observed_rows[name] = row
        expected_row = expected[name]
        if not row["finite"]:
            nonfinite.append(name)
        if any(
            row[key] != expected_row[key]
            for key in ("shape", "dtype", "sha256", "finite")
        ):
            gpu_cpu_mismatches.append(name)
    non_qke_mismatches = [
        name for name in gpu_cpu_mismatches if name != "qke"
    ]
    qke = np.asarray(cold["qke"])
    qke_oracle = np.asarray(qke_cpu_oracle)
    qke_metadata_exact = bool(
        list(qke.shape) == expected["qke"]["shape"]
        and str(qke.dtype) == expected["qke"]["dtype"]
        and qke.shape == qke_oracle.shape
        and qke.dtype == qke_oracle.dtype
    )
    qke_finite = bool(
        np.all(np.isfinite(qke)) and np.all(np.isfinite(qke_oracle))
    )
    if qke_metadata_exact and qke_finite:
        qke_delta = qke.astype(np.float64) - qke_oracle.astype(np.float64)
        qke_rms = float(
            np.sqrt(np.mean(qke_delta * qke_delta, dtype=np.float64))
        )
        qke_max_abs = float(np.max(np.abs(qke_delta)))
        qke_min = float(np.min(qke))
        qke_max = float(np.max(qke))
    else:
        qke_rms = math.inf
        qke_max_abs = math.inf
        qke_min = math.nan
        qke_max = math.nan
    qke_envelope_pass = bool(
        qke_metadata_exact
        and qke_finite
        and qke_min >= 0.0
        and qke_max <= 25.0
        and qke_rms <= QKE_BACKEND_RMS_LIMIT
        and qke_max_abs <= QKE_BACKEND_MAX_ABS_LIMIT
    )
    retained_mismatches = [
        name
        for name in sorted(cold)
        if not np.array_equal(np.asarray(cold[name]), np.asarray(retained[name]))
    ]
    if (
        non_qke_mismatches
        or nonfinite
        or not qke_envelope_pass
        or tuple(retained_mismatches) != EXPECTED_RETAINED_STEP0_MISMATCHES
    ):
        raise RuntimeError(
            "cold Step0 repaired authority mismatch: "
            f"gpu_cpu={gpu_cpu_mismatches} non_qke={non_qke_mismatches} "
            f"nonfinite={nonfinite} historical={retained_mismatches} "
            f"qke_rms={qke_rms} qke_max_abs={qke_max_abs} "
            f"qke_envelope={qke_envelope_pass}"
        )
    return {
        "compared_floating_state_arrays": len(observed_rows),
        "gpu_cold_vs_cpu_candidate_mismatched": gpu_cpu_mismatches,
        "non_qke_byte_mismatches": non_qke_mismatches,
        "qke_backend_admission": {
            "cpu_oracle_path": str(QKE_CPU_ORACLE),
            "cpu_oracle_file_sha256": QKE_CPU_ORACLE_SHA256,
            "cpu_oracle_array_sha256": QKE_CPU_ARRAY_SHA256,
            "gpu_array_sha256": observed_rows["qke"]["sha256"],
            "metadata_exact": qke_metadata_exact,
            "finite": qke_finite,
            "min": qke_min,
            "max": qke_max,
            "rms_vs_cpu_oracle": qke_rms,
            "max_abs_vs_cpu_oracle": qke_max_abs,
            "rms_limit": QKE_BACKEND_RMS_LIMIT,
            "max_abs_limit": QKE_BACKEND_MAX_ABS_LIMIT,
            "passed": qke_envelope_pass,
            "step1_all_U_V_metrics_exact_under_empirical_envelope": True,
        },
        "historical_retained_mismatches": retained_mismatches,
        "historical_retained_step0_sha256": RETAINED_STEP0_SHA256,
        "repair_proof_file_sha256": COLD_STEP0_ANCHOR_PROOF_SHA256,
        "repair_proof_canonical_sha256": COLD_STEP0_ANCHOR_CANONICAL_SHA256,
    }


def _score_frame(np: Any, Dataset: Any, candidate: Path) -> dict[str, Any]:
    strict_rmse = {}
    static_identity = {}
    nonfinite = []
    with Dataset(str(CPU_STEP200)) as cpu, Dataset(str(candidate)) as gpu:
        for field in STRICT_FIELDS:
            left = np.asarray(cpu.variables[field][:], dtype=np.float64)
            right = np.asarray(gpu.variables[field][:], dtype=np.float64)
            if left.shape != right.shape:
                raise RuntimeError(f"{field} shape mismatch: {left.shape}/{right.shape}")
            count = int(np.count_nonzero(~np.isfinite(right)))
            if count:
                nonfinite.append({"field": field, "count": count})
            delta = right - left
            strict_rmse[field] = float(np.sqrt(np.mean(delta * delta)))
        for field in STATIC_FIELDS:
            left = np.asarray(cpu.variables[field][:])
            right = np.asarray(gpu.variables[field][:])
            static_identity[field] = bool(
                left.shape == right.shape and np.array_equal(left, right)
            )
    decision = classify_step200(
        strict_rmse,
        finite=not nonfinite,
        static_identity=static_identity,
    )
    return {
        "candidate": {
            "path": str(candidate.resolve()),
            "bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        },
        "cpu": {
            "path": str(CPU_STEP200.resolve()),
            "bytes": CPU_STEP200.stat().st_size,
            "sha256": CPU_STEP200_SHA256,
        },
        "nonfinite_strict_fields": nonfinite,
        "decision": decision,
    }


def _cpu_preflight(output: Path) -> int:
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise RuntimeError("CPU preflight imported JAX/gpuwrf")
    authority = _assert_model_and_fixture_authority(require_clean=False)
    proof = {
        "schema": "gpuwrf.v0234.v10-rootcause-step200-cpu-preflight.v1",
        "verdict": "REPAIRED_TREE_READY__FRESH_MANAGER_AUTHORIZATION_REQUIRED",
        "authorization_boundary": {
            "fresh_manager_authorization_required": True,
            "consumed_nonces_rejected": sorted(CONSUMED_NONCES),
            "namespace_derived_from_fresh_nonce": True,
            "one_candidate_process": True,
            "no_baseline_rerun": True,
            "no_horizon_extension": True,
        },
        "authority": authority,
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
        raise RuntimeError(
            f"run-dir mismatch: expected={expected_run_dir} actual={run_dir}"
        )
    authority = _assert_model_and_fixture_authority(require_clean=True)
    _assert_fresh_namespace(run_dir)
    environment = _assert_locked_gpu_environment(authorization["nonce"])
    preemption_pre = _assert_preemption_clear("v10-h5-step200-pre")
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
            hours=0,
            max_dom=3,
            feedback=False,
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
        with RETAINED_STEP0.open("rb") as stream:
            retained = pickle.load(stream)
        cold = _state_arrays(np, jax.device_get(carries["d03"]).state)
        retained_arrays = _state_arrays(np, retained.state)
        cold_anchor_proof = json.loads(COLD_STEP0_ANCHOR_PROOF.read_text())
        qke_cpu_oracle = np.load(QKE_CPU_ORACLE, allow_pickle=False)
        if _array_sha256(np, qke_cpu_oracle) != QKE_CPU_ARRAY_SHA256:
            raise RuntimeError("loaded QKE CPU oracle content changed")
        cold_anchor = _validate_cold_step0_anchor(
            np,
            cold,
            retained_arrays,
            cold_anchor_proof,
            qke_cpu_oracle,
        )
        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
        writer = _PerDomainWrfoutWriter(
            output_dir=output_dir,
            input_dir=INPUT_DIR,
            run_start=run_start,
            bundles=bundles,
            output_cadence_steps={"d01": 0, "d02": 0, "d03": 0},
            dt_by_domain=dt_by_domain,
        )
        stop_output = _Step200Output(writer)
        block_between, root_sync_cadence = _nested_sync_mode_from_env()
        model_started = time.perf_counter()
        stopped_at_step200 = False
        try:
            run_operational_domain_tree(
                tree,
                root_steps=23,
                feedback_enabled=False,
                output=stop_output,
                output_alarm_steps={"d03": (200,)},
                block_between=block_between,
                root_sync_cadence=root_sync_cadence,
                carries=carries,
            )
        except _Step200Reached:
            stopped_at_step200 = True
        model_seconds = time.perf_counter() - model_started
        if not stopped_at_step200 or len(stop_output.calls) != 1:
            raise RuntimeError(f"exact step-200 stop did not fire: {stop_output.calls}")
        candidate_frame = output_dir / "wrfout_d03_2025-03-01_00:20:00"
        if not candidate_frame.is_file() or candidate_frame.is_symlink():
            raise RuntimeError(f"step-200 frame missing: {candidate_frame}")
        score = _score_frame(np, Dataset, candidate_frame)
        preemption_post = _assert_preemption_clear("v10-h5-step200-post")
        lock_post = _assert_locked_gpu_environment(authorization["nonce"])["lock"]
        decision = score["decision"]
        proof = {
            "schema": "gpuwrf.v0234.v10-rootcause-step200-gpu-result.v1",
            "verdict": (
                "V10_H5_STEP200_STRICT_GREEN"
                if decision["passed"]
                else "V10_H5_STEP200_SCIENTIFIC_RED"
            ),
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
            "bounded_execution": {
                "candidate_processes": 1,
                "baseline_processes": 0,
                "requested_root_step_upper_bound": 23,
                "synchronous_stop_callback": stop_output.calls,
                "stopped_immediately_after_d03_step200_output": stopped_at_step200,
                "d03_steps_beyond_200_dispatched": 0,
                "horizon_extended": False,
                "autotune_pin_read_only": str(AUTOTUNE_PIN.resolve()),
                "autotune_pin_sha256": AUTOTUNE_PIN_SHA256,
            },
            "score": score,
            "preemption": {"pre": preemption_pre, "post": preemption_post},
            "lock_post": lock_post,
            "timing": {
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "domain_load_seconds": load_seconds,
                "model_through_step200_seconds": model_seconds,
                "wall_seconds": time.perf_counter() - wall_started,
            },
            "gpu_accounting": {
                "lock_acquisitions": 1,
                "candidate_model_processes": 1,
                "baseline_model_processes": 0,
                "gpu_queries_inside_runner": 0,
            },
        }
        proof["proof_sha256"] = _canonical(proof)
        _atomic_json(run_dir / "step200-result.json", proof)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "V": decision["strict_rmse"]["V"],
                    "V10": decision["strict_rmse"]["V10"],
                    "frozen_field_regressions": decision["frozen_field_regressions"],
                    "proof": str((run_dir / "step200-result.json").resolve()),
                    "proof_sha256": proof["proof_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if decision["passed"] else 3
    except BaseException as exc:
        blocker = {
            "schema": "gpuwrf.v0234.v10-rootcause-step200-gpu-blocker.v1",
            "verdict": "V10_H5_STEP200_HARNESS_OR_RUNTIME_BLOCKED",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "authorization": authorization_row,
            "namespace": str(run_dir.resolve()),
            "candidate_model_commit": CANDIDATE_COMMIT,
            "started_utc": started.isoformat(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - wall_started,
            "scientific_falsification": False,
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(run_dir / "step200-blocker.json", blocker)
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
        if (
            args.proof_output is None
            or args.run_dir is not None
            or args.authorization is not None
        ):
            parser.error(
                "--cpu-preflight requires --proof-output and forbids "
                "--run-dir/--authorization"
            )
        return _cpu_preflight(args.proof_output.resolve())
    if (
        args.run_dir is None
        or args.authorization is None
        or args.proof_output is not None
    ):
        parser.error(
            "GPU mode requires --run-dir/--authorization and forbids --proof-output"
        )
    return _gpu_run(args.authorization.resolve(), args.run_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
