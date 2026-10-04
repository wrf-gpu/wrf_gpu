"""Fail-closed v0234 diagnostic admission for the known 15:00 wake red.

This module does not change a tolerance and never declares V or V10 green.  It
authenticates Kimi's retained realization envelope, then admits one observation
only when finite/static identity is already green, no strict field outside
V/V10 is red, and the retained southwest-ocean wake fingerprint is reproduced.
The admitted record remains a release blocker and exists solely to isolate the
late-Ni window in the same fresh full-tree process.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SPRINT = REPO_ROOT / ".agent/sprints/2026-07-18-v0234-deterministic-wake-closure-gpt"
CONTRACT = SPRINT / "CONTRACT.md"
CONTRACT_SHA256 = "d26ad781c615f6fdcd2310d65d933693ebf60bad10924746cab1eb2c62f9dcae"

KIMI_SPRINT = REPO_ROOT / ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi"
KIMI_PROOF = KIMI_SPRINT / "proof.json"
KIMI_PROOF_FILE_SHA256 = "252f7a77647f8bcf485676fde99382e2e87e08b0c5942651008e392d2bd62015"
KIMI_PROOF_CANONICAL_SHA256 = "a05e5bbd89bb61f40b18775985b5237fa01a0fd55a0285dae4cb0e692e9fadd8"
KIMI_RETAINED = KIMI_SPRINT / "retained-evidence-manifest.json"
KIMI_RETAINED_FILE_SHA256 = "344efd6c14ef78d7825afee59a30d29bb8164a291ceac7b1ba5a663785a21ddc"
KIMI_RETAINED_CANONICAL_SHA256 = "7e12fda919926b538d783a31ca2ce5a9735d1adaefd5424eaa21f29e4ab84edc"

CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
LINEAGE_ROOT = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a"
RETAINED_A = LINEAGE_ROOT / (
    "nested_stage_omega_transport_470e6111_full18h_toolingrepair2/"
    "output/wrfout_d03_2025-03-01_15:00:00"
)
RETAINED_B = LINEAGE_ROOT / (
    "nested_stage_omega_transport_470e6111_post_fable_corner_window_"
    "gpt56_resource_retry1/gpu-output/wrfout_d03_2025-03-01_15:00:00"
)
CPU_WRF = CASE_ROOT / "run/wrf/wrfout_d03_2025-03-01_15:00:00"
RETAINED_A_SHA256 = "f287d01cbe8948d85ee719a18ff3d36edc68013654c8f3ecddd66c795a07e989"
RETAINED_B_SHA256 = "6a1dda387493da7275388abfbaa63b974d7ab42ba072625acc9fc8e022d58a5b"
CPU_WRF_SHA256 = "1ea00bd68bfa4098abd14d49e031389bae53bd0c921c80b2af264cd9dcbe2f0c"

STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
FROZEN_V = 1.1318203205639872
V_REALIZATION_SPREAD = 0.00507377303290113
V_ENVELOPE = (FROZEN_V - V_REALIZATION_SPREAD, FROZEN_V + V_REALIZATION_SPREAD)
V10_ENVELOPE = (2.2044028706153287, 2.233988686142588)

# Frozen before the new observation.  These bounds describe the two
# authenticated retained realizations; they are not fitted after this sprint's
# run and are deliberately narrower than a generic Tenerife-region test.
WAKE_BOX = (slice(10, 51), slice(5, 51))
WAKE_CORRELATION_MIN = 0.95
V_LOW_TO_UPPER_RMS_RATIO_MIN = 4.0
V10_MAX_ERROR_Y = (35, 47)
V10_MAX_ERROR_X = (13, 23)
V_MAX_ERROR_K = (0, 3)
V_MAX_ERROR_Y = (35, 43)
V_MAX_ERROR_X = (14, 24)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _authenticate_self_hashed_json(
    path: Path, file_sha256: str, canonical_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != file_sha256:
        raise RuntimeError(f"authenticated evidence file changed: {path}")
    payload = json.loads(path.read_text())
    unsigned = dict(payload)
    embedded = unsigned.pop("proof_sha256", None)
    observed = canonical_digest(unsigned)
    if embedded != canonical_sha256 or observed != canonical_sha256:
        raise RuntimeError(f"authenticated evidence payload changed: {path}")
    return payload


def authenticate_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate every metric and frame used by the admission."""

    if (
        not CONTRACT.is_file()
        or CONTRACT.is_symlink()
        or sha256_file(CONTRACT) != CONTRACT_SHA256
    ):
        raise RuntimeError("deterministic wake-closure contract changed")
    proof = _authenticate_self_hashed_json(
        KIMI_PROOF, KIMI_PROOF_FILE_SHA256, KIMI_PROOF_CANONICAL_SHA256,
    )
    retained = _authenticate_self_hashed_json(
        KIMI_RETAINED,
        KIMI_RETAINED_FILE_SHA256,
        KIMI_RETAINED_CANONICAL_SHA256,
    )
    rows = retained.get("trajectory", {}).get("step9000", {})
    reconstruction = proof.get("reconstruction", {}).get("v_v10_mechanism_verdict", {})
    if (
        proof.get("verdict") != "KIMI_STEP9000_V_V10_NO_FIX_LOCALIZED"
        or proof.get("immutable_code", {}).get("src_gpuwrf_tree_after")
        != "835dcc29bf316c0715b41a72e064985e9cf099df"
        or proof.get("discriminators_run", {}).get("locked_gpu_discriminator", {}).get(
            "verdict"
        ) != "NONDETERMINISM_REPRODUCED__AUTOTUNE_MECHANISM__PIN_VALIDATED"
        or reconstruction.get("v10_is_known_admitted_blocker") is not True
        or reconstruction.get("v_red_is_noise_level") is not True
        or rows.get("V_frozen_ceiling") != FROZEN_V
        or rows.get("V_realization_spread") != V_REALIZATION_SPREAD
        or rows.get("V10_toolingrepair2") != V10_ENVELOPE[0]
        or rows.get("V10_resource_retry1") != V10_ENVELOPE[1]
        or rows.get("V10_red_robust_to_realization_noise") is not True
    ):
        raise RuntimeError("Kimi admission semantics changed")

    frames = {
        "retained_A": (RETAINED_A, RETAINED_A_SHA256),
        "retained_B": (RETAINED_B, RETAINED_B_SHA256),
        "cpu_wrf": (CPU_WRF, CPU_WRF_SHA256),
    }
    frame_rows: dict[str, dict[str, Any]] = {}
    for role, (path, expected) in frames.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"admission frame missing or symlinked: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"admission frame changed: {path}")
        frame_rows[role] = {
            "path": str(path.resolve()),
            "file_sha256": observed,
            "bytes": path.stat().st_size,
        }

    authority = {
        "schema": "gpuwrf.v0234.deterministic-wake-admission-authority.v1",
        "contract": {
            "path": str(CONTRACT.resolve()),
            "file_sha256": CONTRACT_SHA256,
        },
        "kimi_proof": {
            "path": str(KIMI_PROOF.resolve()),
            "file_sha256": KIMI_PROOF_FILE_SHA256,
            "canonical_sha256": KIMI_PROOF_CANONICAL_SHA256,
        },
        "kimi_retained_evidence": {
            "path": str(KIMI_RETAINED.resolve()),
            "file_sha256": KIMI_RETAINED_FILE_SHA256,
            "canonical_sha256": KIMI_RETAINED_CANONICAL_SHA256,
        },
        "model_tree": "835dcc29bf316c0715b41a72e064985e9cf099df",
        "metric_envelope": {
            "V": {"minimum": V_ENVELOPE[0], "maximum": V_ENVELOPE[1]},
            "V10": {"minimum": V10_ENVELOPE[0], "maximum": V10_ENVELOPE[1]},
        },
        "spatial_policy": {
            "wake_box": "y10..50,x5..50",
            "retained_error_correlation_minimum": WAKE_CORRELATION_MIN,
            "cpu_best_shift": {"dy": -1, "dx": 6},
            "retained_best_shift": {"dy": 0, "dx": 0},
            "v_low_to_upper_rms_ratio_minimum": V_LOW_TO_UPPER_RMS_RATIO_MIN,
            "v10_max_error_y_inclusive": list(V10_MAX_ERROR_Y),
            "v10_max_error_x_inclusive": list(V10_MAX_ERROR_X),
            "v_max_error_k_inclusive": list(V_MAX_ERROR_K),
            "v_max_error_y_inclusive": list(V_MAX_ERROR_Y),
            "v_max_error_x_inclusive": list(V_MAX_ERROR_X),
        },
        "frames": frame_rows,
        "release_gate_green": False,
        "isolation_only": True,
    }
    authority["authority_sha256"] = canonical_digest(authority)
    row = {
        "authority_sha256": authority["authority_sha256"],
        "kimi_proof_file_sha256": KIMI_PROOF_FILE_SHA256,
        "kimi_proof_canonical_sha256": KIMI_PROOF_CANONICAL_SHA256,
        "retained_evidence_file_sha256": KIMI_RETAINED_FILE_SHA256,
        "retained_evidence_canonical_sha256": KIMI_RETAINED_CANONICAL_SHA256,
        "authenticated_frame_sha256": {
            role: value["file_sha256"] for role, value in frame_rows.items()
        },
    }
    return authority, row


def _best_shift(np: Any, value: Any, reference: Any) -> dict[str, Any]:
    by, bx = WAKE_BOX[0].start, WAKE_BOX[1].start
    ey, ex = WAKE_BOX[0].stop, WAKE_BOX[1].stop
    best: dict[str, Any] | None = None
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            shifted = np.roll(np.roll(reference, dy, 0), dx, 1)
            ys = slice(max(by, by + dy), min(ey, ey + min(0, dy)))
            xs = slice(max(bx, bx + dx), min(ex, ex + min(0, dx)))
            rmse = float(np.sqrt(np.mean((value[ys, xs] - shifted[ys, xs]) ** 2)))
            if best is None or rmse < best["rmse"]:
                best = {"rmse": rmse, "dy": dy, "dx": dx}
    if best is None:
        raise RuntimeError("empty wake shift search")
    return best


def _in_range(value: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= int(value) <= bounds[1]


def evaluate_spatial_policy(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate predeclared spatial gates without inspecting a new file."""

    v10_location = fingerprint.get("v10_max_error_location") or []
    v_location = fingerprint.get("v_max_error_location") or []
    correlations = fingerprint.get("wake_error_correlation") or {}
    shifts = fingerprint.get("best_shifts") or {}
    checks = {
        "all_arrays_finite": fingerprint.get("all_arrays_finite") is True,
        "v10_max_in_southwest_wake": bool(
            len(v10_location) == 2
            and _in_range(v10_location[0], V10_MAX_ERROR_Y)
            and _in_range(v10_location[1], V10_MAX_ERROR_X)
        ),
        "v_max_low_level_southwest_wake": bool(
            len(v_location) == 3
            and _in_range(v_location[0], V_MAX_ERROR_K)
            and _in_range(v_location[1], V_MAX_ERROR_Y)
            and _in_range(v_location[2], V_MAX_ERROR_X)
        ),
        "wake_error_correlates_with_A": bool(
            float(correlations.get("retained_A", -math.inf))
            >= WAKE_CORRELATION_MIN
        ),
        "wake_error_correlates_with_B": bool(
            float(correlations.get("retained_B", -math.inf))
            >= WAKE_CORRELATION_MIN
        ),
        "same_cpu_wake_shift": bool(
            (shifts.get("candidate_vs_cpu") or {}).get("dy") == -1
            and (shifts.get("candidate_vs_cpu") or {}).get("dx") == 6
        ),
        "same_realization_location_A": bool(
            (shifts.get("candidate_vs_A") or {}).get("dy") == 0
            and (shifts.get("candidate_vs_A") or {}).get("dx") == 0
        ),
        "same_realization_location_B": bool(
            (shifts.get("candidate_vs_B") or {}).get("dy") == 0
            and (shifts.get("candidate_vs_B") or {}).get("dx") == 0
        ),
        "v_error_low_level_concentrated": bool(
            float(fingerprint.get("v_low_to_upper_rms_ratio", -math.inf))
            >= V_LOW_TO_UPPER_RMS_RATIO_MIN
        ),
        "v_error_interior_exceeds_ring1": bool(
            float(fingerprint.get("v_ring1_rms", math.inf))
            < float(fingerprint.get("v_interior_rms", -math.inf))
        ),
        "documented_wake_cell_has_negative_candidate_flow": bool(
            float(fingerprint.get("candidate_v10_y39x19", math.inf)) < -8.0
            and float(fingerprint.get("cpu_v10_y39x19", -math.inf)) > 1.0
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def spatial_fingerprint(
    runtime: SimpleNamespace,
    *,
    candidate_path: Path,
    cpu_path: Path,
    candidate_sha256: str,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    np = runtime.np
    Dataset = runtime.Dataset
    frames = authority["frames"]
    expected_cpu = Path(frames["cpu_wrf"]["path"])
    if cpu_path.resolve() != expected_cpu.resolve() or sha256_file(cpu_path) != CPU_WRF_SHA256:
        raise RuntimeError("CPU WRF frame authority changed")
    if sha256_file(candidate_path) != candidate_sha256:
        raise RuntimeError("candidate changed between pairing and fingerprint")
    for role, expected in (("retained_A", RETAINED_A_SHA256), ("retained_B", RETAINED_B_SHA256)):
        path = Path(frames[role]["path"])
        if sha256_file(path) != expected:
            raise RuntimeError(f"retained wake frame changed: {role}")

    def read(path: Path) -> dict[str, Any]:
        with Dataset(path) as dataset:
            return {
                "V": np.asarray(dataset.variables["V"][0], dtype=np.float64),
                "V10": np.asarray(dataset.variables["V10"][0], dtype=np.float64),
            }

    candidate = read(candidate_path)
    cpu = read(cpu_path)
    retained_a = read(Path(frames["retained_A"]["path"]))
    retained_b = read(Path(frames["retained_B"]["path"]))
    arrays = [*candidate.values(), *cpu.values(), *retained_a.values(), *retained_b.values()]
    finite = all(bool(np.isfinite(value).all()) for value in arrays)

    error10 = candidate["V10"] - cpu["V10"]
    error_v = candidate["V"] - cpu["V"]
    error_a = (retained_a["V10"] - cpu["V10"])[WAKE_BOX].ravel()
    error_b = (retained_b["V10"] - cpu["V10"])[WAKE_BOX].ravel()
    error_candidate = error10[WAKE_BOX].ravel()

    nz, ny, nx = error_v.shape
    y, x = np.ogrid[:ny, :nx]
    distance = np.minimum.reduce(
        [
            np.broadcast_to(y, (ny, nx)),
            np.broadcast_to(x, (ny, nx)),
            np.broadcast_to(ny - 1 - y, (ny, nx)),
            np.broadcast_to(nx - 1 - x, (ny, nx)),
        ]
    )
    ring1 = np.broadcast_to(distance == 1, error_v.shape)
    interior = np.broadcast_to(distance >= 5, error_v.shape)
    low3 = float(np.sqrt(np.mean(error_v[:3] ** 2)))
    upper14 = float(np.sqrt(np.mean(error_v[14:] ** 2)))
    fingerprint = {
        "all_arrays_finite": finite,
        "candidate_frame_sha256": candidate_sha256,
        "cpu_frame_sha256": CPU_WRF_SHA256,
        "retained_frame_sha256": {
            "A": RETAINED_A_SHA256,
            "B": RETAINED_B_SHA256,
        },
        "v10_max_error_location": [
            int(value) for value in np.unravel_index(np.argmax(np.abs(error10)), error10.shape)
        ],
        "v_max_error_location": [
            int(value) for value in np.unravel_index(np.argmax(np.abs(error_v)), error_v.shape)
        ],
        "wake_error_correlation": {
            "retained_A": float(np.corrcoef(error_candidate, error_a)[0, 1]),
            "retained_B": float(np.corrcoef(error_candidate, error_b)[0, 1]),
        },
        "best_shifts": {
            "candidate_vs_cpu": _best_shift(np, candidate["V10"], cpu["V10"]),
            "candidate_vs_A": _best_shift(np, candidate["V10"], retained_a["V10"]),
            "candidate_vs_B": _best_shift(np, candidate["V10"], retained_b["V10"]),
        },
        "v_low3_rms": low3,
        "v_upper14plus_rms": upper14,
        "v_low_to_upper_rms_ratio": low3 / max(upper14, np.finfo(np.float64).tiny),
        "v_ring1_rms": float(np.sqrt(np.mean(error_v[ring1] ** 2))),
        "v_interior_rms": float(np.sqrt(np.mean(error_v[interior] ** 2))),
        "candidate_v10_y39x19": float(candidate["V10"][39, 19]),
        "cpu_v10_y39x19": float(cpu["V10"][39, 19]),
    }
    policy = evaluate_spatial_policy(fingerprint)
    fingerprint["policy"] = policy
    fingerprint["fingerprint_sha256"] = canonical_digest(fingerprint)
    return fingerprint


def metric_signature(decisive: Mapping[str, Any]) -> dict[str, Any]:
    fields = decisive.get("strict_fields") or {}
    violations = decisive.get("violations") or []
    red_fields = sorted(
        field for field, row in fields.items() if row.get("no_worse") is not True
    )
    values = {
        field: float((fields.get(field) or {}).get("candidate_rmse", math.nan))
        for field in STRICT_FIELDS
    }
    checks = {
        "static_exact": decisive.get("static_exact") is True,
        "all_strict_fields_reported": set(fields) == set(STRICT_FIELDS),
        "retry20_authority_exact": bool(fields) and all(
            row.get("retry20_authority_exact") is True for row in fields.values()
        ),
        "only_v_v10_may_be_red": set(red_fields) <= {"V", "V10"},
        "v10_is_still_red": "V10" in red_fields,
        "violations_only_v_v10": all(
            row.get("field") in {"V", "V10"} for row in violations
        ),
        "V_inside_authenticated_envelope": bool(
            math.isfinite(values["V"]) and V_ENVELOPE[0] <= values["V"] <= V_ENVELOPE[1]
        ),
        "V10_inside_authenticated_range": bool(
            math.isfinite(values["V10"])
            and V10_ENVELOPE[0] <= values["V10"] <= V10_ENVELOPE[1]
        ),
        "all_metric_values_finite": all(math.isfinite(value) for value in values.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "strict_rmse": values,
        "red_fields": red_fields,
        "violations": violations,
        "metric_payload_sha256": canonical_digest(dict(decisive)),
    }


def classify_known_wake_observation(
    decisive: Mapping[str, Any],
    *,
    candidate_sha256: str,
    authority: Mapping[str, Any],
    candidate_path: Path | None,
    cpu_path: Path | None,
    runtime: SimpleNamespace | None,
) -> dict[str, Any]:
    metric = metric_signature(decisive)
    fingerprint: dict[str, Any]
    try:
        if candidate_path is None or cpu_path is None or runtime is None:
            raise RuntimeError("candidate/cpu/runtime fingerprint inputs are required")
        fingerprint = spatial_fingerprint(
            runtime,
            candidate_path=candidate_path,
            cpu_path=cpu_path,
            candidate_sha256=candidate_sha256,
            authority=authority,
        )
    except Exception as exc:  # fail closed while retaining the observation
        fingerprint = {
            "policy": {"passed": False, "checks": {}},
            "error": f"{type(exc).__name__}: {exc}",
        }
    passed = bool(metric["passed"] and fingerprint.get("policy", {}).get("passed"))
    return {
        "schema": "gpuwrf.v0234.deterministic-known-wake-admission.v1",
        "passed": passed,
        "classification": (
            "KNOWN_WAKE_RELEASE_BLOCKER_METRIC_AND_FINGERPRINT_MATCH"
            if passed
            else "NEW_SCIENTIFIC_RED_OR_CHANGED_WAKE_SIGNATURE"
        ),
        "record_and_continue_for_late_ni_only": passed,
        "finite_and_static_identity_required_upstream": True,
        "metric_signature": metric,
        "spatial_fingerprint": fingerprint,
        "candidate_frame_sha256": candidate_sha256,
        "authority_sha256": authority.get("authority_sha256"),
        "waiver_or_reclassification": False,
        "tolerance_changed": False,
        "release_gate_green": False,
        "release_blocker_remains": True,
        "isolation_decision_only": True,
    }
