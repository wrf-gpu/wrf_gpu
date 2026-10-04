"""One-process H5 terminal gate plus complete Tenerife paired-case replay.

The module is stdlib-only until authority and the locked CUDA environment have
been authenticated.  It emits the canonical 19/19/55 full-output cadence from
Step0 through +18 h.  No step beyond d03 step 9000 is dispatched until its
frame has been synchronously scored against retained CPU WRF: RED ends the arm;
GREEN is sealed, then the identical carries continue to d03 step 10800 solely
for paired-case completeness.  The retained reference is never rerun.
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
PROOF_SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
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
        "c17fca1e201ae106cf8ad77463686823c82226d59c7dea01e741c0be0ba7f6bc",
    }
)
NONCE_PATTERN = re.compile(r"[0-9a-f]{32,128}")
C1_PROOF = PROOF_SPRINT / "C1_CANDIDATE_OFF_PROOF.json"
C1_PROOF_SHA256 = (
    "59fe04842dbc4b8471547b47b646d01774259d0d0fca9024bd87fcc3e8ea8302"
)
C1_CANONICAL_SHA256 = (
    "099c63a25dddff5eb652d2850c4520138b172b9c28f81b98c25dbf3b62ebf8a6"
)
COLD_STEP0_ANCHOR_PROOF = PROOF_SPRINT / "COLD_STEP0_ANCHOR_REPAIR.json"
COLD_STEP0_ANCHOR_PROOF_SHA256 = (
    "e98c689ad9aab02a59e4eb4896e92bb4b34a7cd85371d766c332e0df6a8beeef"
)
COLD_STEP0_ANCHOR_CANONICAL_SHA256 = (
    "eada51b566dfe9497261226fd6e29b43278a8a9bf0f4e4d77e46325ca2d3aaad"
)
QKE_BACKEND_PROOF = PROOF_SPRINT / "NOCTURNAL_STEP1_QKE_BACKEND.json"
QKE_BACKEND_PROOF_SHA256 = (
    "cb596f27c327d55b7f55b88c2982739d424d5bd7a39e72d98f288250760ff0c3"
)
QKE_BACKEND_CANONICAL_SHA256 = (
    "680a4cab02f7198b9d04c9864420a5a24bb2997a04ebb2ab86ae09f8272e7a0e"
)
QKE_CPU_ORACLE_PROOF = PROOF_SPRINT / "QKE_CPU_ORACLE.json"
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
CPU_STEP9000 = INPUT_DIR / "wrfout_d03_2025-03-01_15:00:00"
CPU_STEP9000_SHA256 = (
    "1ea00bd68bfa4098abd14d49e031389bae53bd0c921c80b2af264cd9dcbe2f0c"
)
REFERENCE_PAIR = (
    LINEAGE
    / "v0234_gpt_v10_replay_c17fca1e201ae106_reference/"
    "frame-pairs/d03-step-09000.json"
)
REFERENCE_PAIR_SHA256 = (
    "ba489cbd8e74d897b142643b558368d008f9c01e10a00b91bf6006621c72f6f4"
)
CASE_METADATA = PROOF_SPRINT / "TERMINAL_CASE_METADATA.json"
CASE_METADATA_SHA256 = (
    "085caaa97b3bdd17c0b9e86ad59eefe454ee31ae4501df68908a3d383cda86da"
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
WATER_SPECIES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
REFERENCE_RMSE = {
    "PSFC": 20.22747532736003,
    "T": 0.5421680888949643,
    "T2": 1.352612988238872,
    "U": 1.1323567330094144,
    "U10": 1.9737860008971808,
    "V": 1.1318203205639872,
    "V10": 2.1128268857679338,
    "W": 0.18633111790197587,
}
TERMINAL_NW_V10_REFERENCE_RMSE = 1.4920505837370035
TERMINAL_QKE_REFERENCE_RMSE = 1.0174668940252842
EXPECTED_OUTPUT_COUNTS = {"d01": 19, "d02": 19, "d03": 55}
ROOT_STEP_GATE = 1000
ROOT_STEP_END = 1200
D03_STEP_GATE = 9000
D03_STEP_END = 10800
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
    return f"v0234_gpt_v10_rootcause_{nonce[:16]}_full18h"


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


def classify_step9000(
    strict_rmse: Mapping[str, float],
    *,
    finite: bool,
    static_identity: Mapping[str, bool],
    dry_mass_rmse: float,
    column_water_rmse: float,
) -> dict[str, Any]:
    """Apply the frozen original terminal ceilings at exact d03 step 9000."""

    exact = set(strict_rmse) == set(STRICT_FIELDS)
    values_finite = exact and all(math.isfinite(float(value)) for value in strict_rmse.values())
    frozen_regressions = {
        field: {
            "candidate": float(strict_rmse.get(field, math.nan)),
            "reference": REFERENCE_RMSE[field],
        }
        for field in STRICT_FIELDS
        if field not in strict_rmse
        or not math.isfinite(float(strict_rmse[field]))
        or float(strict_rmse[field]) > REFERENCE_RMSE[field]
    }
    v_green = bool(
        "V" in strict_rmse
        and math.isfinite(float(strict_rmse["V"]))
        and float(strict_rmse["V"]) <= REFERENCE_RMSE["V"]
    )
    v10_green = bool(
        "V10" in strict_rmse
        and math.isfinite(float(strict_rmse["V10"]))
        and float(strict_rmse["V10"]) <= REFERENCE_RMSE["V10"]
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
        "constituent_watch": {
            "dry_mass_rmse": float(dry_mass_rmse),
            "column_water_rmse": float(column_water_rmse),
            "gating": False,
        },
        "classification": (
            "H5_STEP9000_STRICT_GREEN" if passed else "H5_STEP9000_SCIENTIFIC_RED"
        ),
        "V_at_or_below_frozen_ceiling": v_green,
        "V10_at_or_below_frozen_ceiling": v10_green,
        "finite": bool(finite and values_finite),
        "static_identity": dict(static_identity),
        "static_identity_pass": static_green,
        "frozen_field_regressions": frozen_regressions,
        "strict_rmse": {field: float(strict_rmse[field]) for field in sorted(strict_rmse)},
        "reference_rmse": dict(REFERENCE_RMSE),
        "tolerance_changed": False,
        "baseline_rerun": False,
        "gate_evaluated_only_at_d03_step": D03_STEP_GATE,
        "post_gate_horizon_changes_gate": False,
    }


def _validate_authorization_payload(payload: Mapping[str, Any]) -> str:
    embedded = payload.get("proof_sha256")
    canonical = _canonical(payload)
    nonce = payload.get("nonce")
    if not isinstance(nonce, str):
        raise RuntimeError("manager authorization has no string nonce")
    namespace = _namespace_for_nonce(nonce)
    required = {
        "schema": "gpuwrf.v0234.h5-terminal-full18h-manager-authorization.v1",
        "verdict": "MANAGER_GPU_AUTHORIZED",
        "manager_pane": "0:1",
        "nonce": nonce,
        "namespace": namespace,
        "lock_label": LOCK_LABEL,
        "candidate_model_commit": CANDIDATE_COMMIT,
        "candidate_src_gpuwrf_tree": CANDIDATE_SRC_TREE,
        "authorized_arm_count": 1,
        "authorized_model_processes": 1,
        "root_step_upper_bound": ROOT_STEP_END,
        "d03_step_upper_bound": D03_STEP_END,
        "full_output_counts": EXPECTED_OUTPUT_COUNTS,
        "d03_steps_beyond_9000_authorized": True,
        "continue_beyond_step9000_only_if_green": True,
        "baseline_rerun_authorized": False,
        "full_18h_completion_authorized": True,
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
        gate.get("V_less_than_or_equal") != REFERENCE_RMSE["V"]
        or gate.get("V10_less_than_or_equal") != REFERENCE_RMSE["V10"]
        or gate.get("all_other_fields_less_than_or_equal_frozen_ceilings") is not True
        or gate.get("finite_and_static_identity_required") is not True
        or gate.get("evaluated_at_d03_step") != D03_STEP_GATE
        or gate.get("reevaluated_at_step10800") is not False
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
        (CPU_STEP9000, CPU_STEP9000_SHA256, "CPU_STEP9000"),
        (REFERENCE_PAIR, REFERENCE_PAIR_SHA256, "REFERENCE_PAIR"),
        (CASE_METADATA, CASE_METADATA_SHA256, "CASE_METADATA"),
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
        "cpu_step9000_sha256": CPU_STEP9000_SHA256,
        "reference_pair_sha256": REFERENCE_PAIR_SHA256,
        "case_metadata_sha256": CASE_METADATA_SHA256,
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
        "GPUWRF_V10_TERMINAL_NONCE": nonce,
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


def _expected_schedules() -> dict[str, tuple[int, ...]]:
    return {
        "d01": tuple((hour * 3600 + 53) // 54 for hour in range(1, 19)),
        "d02": tuple(range(200, 3601, 200)),
        "d03": tuple(range(200, D03_STEP_END + 1, 200)),
    }


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


def _constituents(np: Any, ds: Any) -> tuple[Any, Any]:
    """Dry hydrostatic surface pressure and column water substance.

    PSFC is the sum of exactly these two, so gating them individually closes the
    error-cancellation loophole a raw PSFC RMSE leaves open.
    """

    read = lambda name: np.asarray(ds.variables[name][:], dtype=np.float64)
    mu_d = (read("MU") + read("MUB"))[0]
    dry = mu_d + float(read("P_TOP")[0])
    dnw = np.diff(np.asarray(ds.variables["ZNW"][:], dtype=np.float64)[0])
    total_q = None
    for name in WATER_SPECIES:
        if name in ds.variables:
            q = read(name)[0]
            total_q = q if total_q is None else total_q + q
    water = -(total_q * dnw[:, None, None]).sum(axis=0) * mu_d / 9.81
    return dry, water


def _score_frame(np: Any, Dataset: Any, candidate: Path) -> dict[str, Any]:
    strict_rmse = {}
    static_identity = {}
    nonfinite = []
    correlations = {}
    watch = {}
    with Dataset(str(CPU_STEP9000)) as cpu, Dataset(str(candidate)) as gpu:
        dry_cpu, water_cpu = _constituents(np, cpu)
        dry_gpu, water_gpu = _constituents(np, gpu)
        dry_rmse = float(np.sqrt(np.mean((dry_gpu - dry_cpu) ** 2)))
        water_rmse = float(np.sqrt(np.mean((water_gpu - water_cpu) ** 2)))
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
            if field in {"U10", "V10"}:
                correlations[field] = float(
                    np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1]
                )
        for field in STATIC_FIELDS:
            left = np.asarray(cpu.variables[field][:])
            right = np.asarray(gpu.variables[field][:])
            static_identity[field] = bool(
                left.shape == right.shape and np.array_equal(left, right)
            )
        qke_cpu = np.asarray(cpu.variables["QKE"][:], dtype=np.float64)
        qke_gpu = np.asarray(gpu.variables["QKE"][:], dtype=np.float64)
        qke_rmse = float(np.sqrt(np.mean((qke_gpu - qke_cpu) ** 2)))
        lat = np.asarray(cpu.variables["XLAT"][:], dtype=np.float64).squeeze()
        lon = np.asarray(cpu.variables["XLONG"][:], dtype=np.float64).squeeze()
        nw = (lat >= float(np.median(lat))) & (lon < float(np.median(lon)))
        v10_cpu = np.asarray(cpu.variables["V10"][:], dtype=np.float64).squeeze()
        v10_gpu = np.asarray(gpu.variables["V10"][:], dtype=np.float64).squeeze()
        nw_v10_rmse = float(np.sqrt(np.mean((v10_gpu[nw] - v10_cpu[nw]) ** 2)))
        watch = {
            "NW_quadrant_V10": {
                "candidate_rmse": nw_v10_rmse,
                "accepted_pre_fix_reference_rmse": TERMINAL_NW_V10_REFERENCE_RMSE,
                "change_pct": 100.0
                * (nw_v10_rmse / TERMINAL_NW_V10_REFERENCE_RMSE - 1.0),
                "gating": False,
                "cell_count": int(np.count_nonzero(nw)),
            },
            "QKE": {
                "candidate_rmse": qke_rmse,
                "accepted_pre_fix_reference_rmse": TERMINAL_QKE_REFERENCE_RMSE,
                "change_pct": 100.0
                * (qke_rmse / TERMINAL_QKE_REFERENCE_RMSE - 1.0),
                "gating": False,
            },
        }
    decision = classify_step9000(
        strict_rmse,
        finite=not nonfinite,
        static_identity=static_identity,
        dry_mass_rmse=dry_rmse,
        column_water_rmse=water_rmse,
    )
    return {
        "candidate": {
            "path": str(candidate.resolve()),
            "bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        },
        "cpu": {
            "path": str(CPU_STEP9000.resolve()),
            "bytes": CPU_STEP9000.stat().st_size,
            "sha256": CPU_STEP9000_SHA256,
        },
        "nonfinite_strict_fields": nonfinite,
        "d03_spatial_pearson_correlation": correlations,
        "terminal_watch_items": watch,
        "decision": decision,
    }


def _cpu_preflight(output: Path) -> int:
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise RuntimeError("CPU preflight imported JAX/gpuwrf")
    authority = _assert_model_and_fixture_authority(require_clean=False)
    proof = {
        "schema": "gpuwrf.v0234.h5-terminal-full18h-cpu-preflight.v1",
        "verdict": "H5_TERMINAL_FULL18H_RUNNER_READY",
        "authorization_boundary": {
            "fresh_manager_authorization_required": True,
            "consumed_nonces_rejected": sorted(CONSUMED_NONCES),
            "namespace_derived_from_fresh_nonce": True,
            "one_candidate_process": True,
            "no_baseline_rerun": True,
            "step9000_red_stops_before_any_later_dispatch": True,
            "step9000_green_only_continuation_to_step10800": True,
            "full_output_counts": EXPECTED_OUTPUT_COUNTS,
        },
        "exact_post_step0_alarm_schedules": {
            name: list(values) for name, values in _expected_schedules().items()
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
    preemption_pre = _assert_preemption_clear("v10-h5-step9000-pre")
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
        schedules = _expected_schedules()
        output_cadence = {"d01": 67, "d02": 200, "d03": 200}
        if {name: len(values) + 1 for name, values in schedules.items()} != EXPECTED_OUTPUT_COUNTS:
            raise RuntimeError(f"full-cadence schedule drifted: {schedules}")
        writer = _PerDomainWrfoutWriter(
            output_dir=output_dir,
            input_dir=INPUT_DIR,
            run_start=run_start,
            bundles=bundles,
            output_cadence_steps=output_cadence,
            dt_by_domain=dt_by_domain,
        )
        initial_outputs = [writer(name, 0, carries[name]) for name in names]
        block_between, root_sync_cadence = _nested_sync_mode_from_env()
        model_started = time.perf_counter()
        own_steps = {name: 0 for name in names}
        segments = []
        score = None
        gate_finished_utc = None
        model_through_gate_seconds = None
        while own_steps["d01"] < ROOT_STEP_END:
            target = ROOT_STEP_GATE if own_steps["d01"] < ROOT_STEP_GATE else ROOT_STEP_END
            segment_start = int(own_steps["d01"])
            segment_steps = min(output_cadence["d01"], target - segment_start)
            _assert_preemption_clear(f"v10-h5-full18h-before-root-{segment_start}")
            result = run_operational_domain_tree(
                tree,
                root_steps=segment_steps,
                feedback_enabled=False,
                output=writer,
                output_cadence_steps=output_cadence,
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
                }
            )
            if own_steps["d01"] == ROOT_STEP_GATE:
                if own_steps != {"d01": 1000, "d02": 3000, "d03": D03_STEP_GATE}:
                    raise RuntimeError(f"step9000 gate clock drifted: {own_steps}")
                candidate_frame = output_dir / "wrfout_d03_2025-03-01_15:00:00"
                if not candidate_frame.is_file() or candidate_frame.is_symlink():
                    raise RuntimeError(f"step-9000 frame missing: {candidate_frame}")
                score = _score_frame(np, Dataset, candidate_frame)
                gate_finished_utc = datetime.now(timezone.utc).isoformat()
                model_through_gate_seconds = time.perf_counter() - model_started
                gate_proof = {
                    "schema": "gpuwrf.v0234.h5-terminal-step9000-gpu-result.v2",
                    "verdict": (
                        "V10_H5_STEP9000_STRICT_GREEN"
                        if score["decision"]["passed"]
                        else "V10_H5_STEP9000_SCIENTIFIC_RED"
                    ),
                    "authorization": authorization_row,
                    "authorization_nonce": authorization["nonce"],
                    "namespace": str(run_dir.resolve()),
                    "authority": authority,
                    "own_steps_at_score": dict(own_steps),
                    "d03_steps_beyond_9000_dispatched_before_score": 0,
                    "score": score,
                    "timing": {
                        "started_utc": started.isoformat(),
                        "gate_finished_utc": gate_finished_utc,
                        "domain_load_seconds": load_seconds,
                        "model_through_step9000_seconds": model_through_gate_seconds,
                        "wall_through_gate_seconds": time.perf_counter() - wall_started,
                    },
                }
                gate_proof["proof_sha256"] = _canonical(gate_proof)
                _atomic_json(run_dir / "step9000-result.json", gate_proof)
                decision = score["decision"]
                print(
                    json.dumps(
                        {
                            "milestone": "STEP9000_SCORED_BEFORE_LATER_DISPATCH",
                            "verdict": gate_proof["verdict"],
                            "V": decision["strict_rmse"]["V"],
                            "V10": decision["strict_rmse"]["V10"],
                            "U10_correlation": score["d03_spatial_pearson_correlation"]["U10"],
                            "V10_correlation": score["d03_spatial_pearson_correlation"]["V10"],
                            "watch": score["terminal_watch_items"],
                            "proof": str((run_dir / "step9000-result.json").resolve()),
                            "proof_sha256": gate_proof["proof_sha256"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if not decision["passed"]:
                    return 3

        if own_steps != {"d01": ROOT_STEP_END, "d02": 3600, "d03": D03_STEP_END}:
            raise RuntimeError(f"full18h terminal clock drifted: {own_steps}")
        if score is None or not score["decision"]["passed"]:
            raise RuntimeError("full18h continuation occurred without sealed green gate")
        counts = {name: len(writer.written[name]) for name in names}
        if counts != EXPECTED_OUTPUT_COUNTS:
            raise RuntimeError(f"full-cadence output counts drifted: {counts}")
        case_metadata = json.loads(CASE_METADATA.read_text())
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
        preemption_post = _assert_preemption_clear("v10-h5-full18h-post")
        lock_post = _assert_locked_gpu_environment(authorization["nonce"])["lock"]
        proof = {
            "schema": "gpuwrf.v0234.h5-terminal-full18h-gpu-result.v1",
            "verdict": "V10_H5_STEP9000_STRICT_GREEN__FULL18H_19_19_55_COMPLETE",
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
                "end": "2025-03-01T18:00:00+00:00",
                "domains": list(names),
                "dt_by_domain": dt_by_domain,
                "loaded_moist_scalar_options": loaded_options,
                "metadata_scalar_options": {
                    name: {
                        "moist_adv_opt": metadata["domains"][name]["namelist"]["moist_adv_opt"],
                        "scalar_adv_opt": metadata["domains"][name]["namelist"]["scalar_adv_opt"],
                    }
                    for name in names
                },
                "cold_step0_anchor": cold_anchor,
            },
            "step9000_gate": {
                "proof_path": str((run_dir / "step9000-result.json").resolve()),
                "proof_file_sha256": _sha256(run_dir / "step9000-result.json"),
                "score": score,
                "evaluated_before_any_later_dispatch": True,
                "reevaluated_at_step10800": False,
            },
            "full_case": {
                "initial_outputs": initial_outputs,
                "post_step0_alarm_schedules": {name: list(values) for name, values in schedules.items()},
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
                "segments": segments,
                "green_only_post_gate_continuation": True,
                "autotune_pin_read_only": str(AUTOTUNE_PIN.resolve()),
                "autotune_pin_sha256": AUTOTUNE_PIN_SHA256,
            },
            "preemption": {"pre": preemption_pre, "post": preemption_post},
            "lock_post": lock_post,
            "timing": {
                "started_utc": started.isoformat(),
                "step9000_finished_utc": gate_finished_utc,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "domain_load_seconds": load_seconds,
                "model_through_step9000_seconds": model_through_gate_seconds,
                "model_through_step10800_seconds": time.perf_counter() - model_started,
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
        _atomic_json(run_dir / "full18h-result.json", proof)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "V": score["decision"]["strict_rmse"]["V"],
                    "V10": score["decision"]["strict_rmse"]["V10"],
                    "correlations": score["d03_spatial_pearson_correlation"],
                    "output_counts": counts,
                    "proof": str((run_dir / "full18h-result.json").resolve()),
                    "proof_sha256": proof["proof_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        blocker = {
            "schema": "gpuwrf.v0234.h5-terminal-step9000-gpu-blocker.v1",
            "verdict": "V10_H5_STEP9000_HARNESS_OR_RUNTIME_BLOCKED",
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
        _atomic_json(run_dir / "step9000-blocker.json", blocker)
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
