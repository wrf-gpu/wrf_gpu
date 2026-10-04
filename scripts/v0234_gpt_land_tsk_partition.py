#!/usr/bin/env python3
"""Repartition the QAIR-corrected v0234 SP2 CPU capture.

This backend-dark wrapper binds the unchanged frozen five-category comparator
to the one-invocation CPU archive, validates the complete paired-delta chain,
and makes the sprint's acceptance decision from strict improvement of both U
and V RMS against the immediately preceding authenticated capture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
PRIOR_SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
CAPTURE_ROOT = SCRATCH / "qml-fixed-cpu-adapter/qml-v1"
CAPTURE_ARCHIVE = CAPTURE_ROOT / "single-authority-capture.npz"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
PRIOR_CAPTURE = STAGED / "surface-fixed-cpu-adapter/tskin-v1/single-authority-capture.npz"
WRF_ROOT = STAGED / "wrf-dumps"
DELTA = SCRATCH / "qml-source-delta-v1.npz"
RAW_OUTPUT = SCRATCH / "partition/qml-v1-raw.json"

CAPTURE_PREFLIGHT = SPRINT / "qml-capture-preflight.json"
CAPTURE_PROOF = SPRINT / "qml-capture-proof.json"
PROVENANCE_PROOF = SPRINT / "tsk-provenance-proof.json"
PRIOR_PARTITION = PRIOR_SPRINT / "surface-fixed-partition-proof.json"
CAPTURE_PROFILE = REPO / "scripts/v0234_gpt_land_tsk_cpu_capture.py"
FROZEN_DRIVER = REPO / "scripts/v0234_fable_sp2_postfix_partition.py"
FROZEN_COMPARATOR = REPO / "scripts/v0234_gpt_operand_attribution.py"
AUDIT_READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"

EXPECTED = {
    "capture_archive": "2bbb1b184db796b5f5faa02e4040283ba962b2239960ec8319a10817fbfe883d",
    "capture_manifest": "7d0d8b3113f46d6f2ab8cbdcb03df85561031bcc59aa41380d45e5a7d0dfac4e",
    "capture_manifest_canonical": "3f200a5ef09b7ee2315379290dbdf62332370f17361f04bf7af4c2ffd5da4050",
    "capture_proof": "70e4738ebae38b4f9e3069b52adf0b118393d019cfa1f5d78fee0362e7ab3dbf",
    "capture_proof_canonical": "f53c6d43bd2dd8b2aede3401a210eab365ddc7a96cc961d9459b823eacd38abf",
    "capture_preflight": "d438039d7b9ce7be357e93e628967555dc283ae631b40fe54063e1dbdcbc1fb2",
    "capture_preflight_canonical": "6e29b4cdcaee94585a5fa108bbac2c8af0041367e6d9b4abc37d75600cf1b8ec",
    "provenance_proof": "140905f7b8998ab4b5a39fa78e2cfc1213e18baccc21479aabad052e2cda26c0",
    "provenance_canonical": "c75c4b7ce65b64bd5fc5b2da431fce4de720fe7f6e02a407fe31ce8492cbc5a2",
    "delta": "85dce69d3a9d6a33d699d852ad4779de53a322d96dcd738f7e9a128ae6d8a0a0",
    "prior_capture": "880ac4d84258d3f8fccc7af3aab86ede89511a05c4be4d8aa9549958fc6fd94c",
    "prior_partition": "375f915393bc92791648756094144f86e0adc30885968e572e0282c66350f6b0",
    "capture_profile": "62db3a090f8a34589f0dd0d94d7eb8231f5094216c56df59ce68efae1871bd17",
    "capture_head": "88e17d878c2d606bbf3a7f89467c979a56b8c938",
    "frozen_driver": "e7a2aebba9fb6460d3863a8a1e03ec9240621ea19a496bc429b23416c35d3b4c",
    "frozen_comparator": "247a1dca01b7aeb35af1adcfd3eb1dc459472c04275edc7f5b2cb3f380df3244",
    "audit_reader": "49bc7b9761e2da1e807000bbbc101d05039a585b8b00f4b31c3b404b66d79d2a",
    "wrf_tree": "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}
BASELINE = {
    "u_rms": 3.9935109120127676e-7,
    "v_rms": 7.372784327594862e-7,
}
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")


class PartitionFailure(RuntimeError):
    """Fail-closed evidence, source, or comparator failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    body = {
        key: item for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def checked_json(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise PartitionFailure(f"EVIDENCE_HASH:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_without_self(value) != value.get("canonical_payload_sha256"):
        raise PartitionFailure(f"EVIDENCE_CANONICAL:{path}")
    return value


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PartitionFailure(f"IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_delta(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape or not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise PartitionFailure(f"DELTA_INPUT:{lhs.shape}:{rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    return {
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "exact_fraction": float(np.mean(lhs == rhs)),
        "logical_c_bitpayload_sha256": hashlib.sha256(
            np.ascontiguousarray(delta).tobytes()
        ).hexdigest(),
    }


def tracked_wrapper_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{relative}"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    if disk != head:
        raise PartitionFailure("WRAPPER_NOT_HEAD")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", f"HEAD:{relative}"],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip(),
    }


def validate_chain() -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if tracked:
        raise PartitionFailure(f"TRACKED_WORKTREE_NOT_CLEAN:{tracked}")
    for label, path in (
        ("capture_archive", CAPTURE_ARCHIVE),
        ("prior_capture", PRIOR_CAPTURE),
        ("delta", DELTA),
        ("capture_profile", CAPTURE_PROFILE),
        ("frozen_driver", FROZEN_DRIVER),
        ("frozen_comparator", FROZEN_COMPARATOR),
        ("audit_reader", AUDIT_READER),
        ("prior_partition", PRIOR_PARTITION),
    ):
        if path.is_symlink() or not path.is_file():
            raise PartitionFailure(f"INPUT_NOT_REGULAR:{path}")
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise PartitionFailure(f"INPUT_HASH:{label}:{actual}")

    manifest = checked_json(CAPTURE_MANIFEST, EXPECTED["capture_manifest"])
    capture_proof = checked_json(CAPTURE_PROOF, EXPECTED["capture_proof"])
    preflight = checked_json(CAPTURE_PREFLIGHT, EXPECTED["capture_preflight"])
    provenance = checked_json(PROVENANCE_PROOF, EXPECTED["provenance_proof"])
    prior_partition = checked_json(PRIOR_PARTITION, EXPECTED["prior_partition"])
    qml_record = preflight.get("qml_tsk_correction", {})
    authority = manifest.get("authority", {})
    if not (
        manifest.get("canonical_payload_sha256") == EXPECTED["capture_manifest_canonical"]
        and manifest.get("passed") is True
        and manifest.get("status") == "SINGLE_AUTHORITY_CAPTURE_COMPLETE"
        and manifest.get("single_adapter_invocation") is True
        and manifest.get("single_namespace") is True
        and manifest.get("production_operands_and_tendencies_same_invocation") is True
        and authority.get("adapter_invocations") == 1
        and authority.get("gpu_actions") == 0
        and authority.get("wrf_or_mpi_executions") == 0
        and manifest.get("git", {}).get("head") == EXPECTED["capture_head"]
        and manifest.get("archive", {}).get("sha256") == EXPECTED["capture_archive"]
        and manifest.get("input_grid", {}).get("wrfinput_d03_sha256") == EXPECTED["wrfinput"]
        and capture_proof.get("canonical_payload_sha256") == EXPECTED["capture_proof_canonical"]
        and capture_proof.get("passed") is True
        and capture_proof.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and capture_proof.get("manifest", {}).get("sha256") == EXPECTED["capture_manifest"]
        and capture_proof.get("git", {}).get("head") == EXPECTED["capture_head"]
        and preflight.get("canonical_payload_sha256") == EXPECTED["capture_preflight_canonical"]
        and preflight.get("passed") is True
        and preflight.get("git", {}).get("head") == EXPECTED["capture_head"]
        and preflight.get("profile", {}).get("sha256") == EXPECTED["capture_profile"]
        and preflight.get("grid_metric_authority", {}).get("analytic_flat_fallback_forbidden") is True
        and qml_record.get("proof_sha256") == EXPECTED["provenance_proof"]
        and qml_record.get("artifact_sha256") == EXPECTED["delta"]
        and qml_record.get("changed_state_handles") == list(DELTA_FIELDS)
        and qml_record.get("water_delta_bitwise_zero") is True
        and qml_record.get("synthetic_wrf_reference") is False
        and qml_record.get("empirical_tuning") is False
        and provenance.get("canonical_payload_sha256") == EXPECTED["provenance_canonical"]
        and provenance.get("passed") is True
        and provenance.get("verdict") == "WRF_QML_SPECIFIC_HUMIDITY_TSK_CANDIDATE_PROVEN"
        and provenance.get("delta_artifact", {}).get("sha256") == EXPECTED["delta"]
        and prior_partition.get("passed") is True
    ):
        raise PartitionFailure("AUTHORITY_CHAIN")

    with np.load(DELTA, allow_pickle=False) as delta, np.load(
        CAPTURE_ARCHIVE, allow_pickle=False,
    ) as current, np.load(PRIOR_CAPTURE, allow_pickle=False) as previous:
        closure: dict[str, Any] = {}
        archive_names = {
            "theta_flux": "entry_state_theta_flux",
            "qv_flux": "entry_state_qv_flux",
            "fltv": "entry_state_fltv",
            "t_skin": "surface_terms_flux_t_skin",
        }
        for field, archive_name in archive_names.items():
            declared = np.asarray(delta[f"{field}_delta"], dtype=np.float64)
            expected = (
                np.asarray(previous[archive_name], dtype=np.float64) + declared
            )
            # Validate the actual transform expression directly. Subtracting the
            # two archive values and then comparing that recovered delta is not
            # bitwise reversible at the ~1e-18 level because IEEE addition rounds.
            closure[field] = exact_delta(current[archive_name], expected)
            if closure[field]["max_abs"] != 0.0:
                raise PartitionFailure(f"PAIRED_DELTA_CLOSURE:{field}:{closure[field]}")
        unchanged = {
            name: exact_delta(current[f"entry_state_{name}"], previous[f"entry_state_{name}"])
            for name in ("ustar", "rhosfc")
        }
        if any(value["max_abs"] != 0.0 for value in unchanged.values()):
            raise PartitionFailure(f"UNCHANGED_SURFACE_HANDLES:{unchanged}")
    return {
        "capture_manifest_canonical_sha256": manifest["canonical_payload_sha256"],
        "capture_proof_canonical_sha256": capture_proof["canonical_payload_sha256"],
        "capture_preflight_canonical_sha256": preflight["canonical_payload_sha256"],
        "provenance_proof_canonical_sha256": provenance["canonical_payload_sha256"],
        "paired_delta_entry_exact_closure": closure,
        "unchanged_prior_surface_handles": unchanged,
        "prior_tau_handles": "source-proven in the prior correction but not consumed/captured by MYNN",
        "prior_partition_sha256": EXPECTED["prior_partition"],
        "wrapper": tracked_wrapper_record(),
    }


def configure_frozen_driver(driver: Any, comparator: Any) -> None:
    comparator.WRF_ROOT = WRF_ROOT

    def staged_reader() -> tuple[Any, dict[str, Any]]:
        reader_module = import_file("v0234_qml_partition_reader", AUDIT_READER)
        return reader_module.IndependentWrfDump(WRF_ROOT), {
            "path": str(AUDIT_READER),
            "sha256": EXPECTED["audit_reader"],
            "staged_root": str(WRF_ROOT),
        }

    comparator.import_dump_reader = staged_reader
    driver.load_gpt_module = lambda: comparator
    driver.POSTFIX_ROOT = CAPTURE_ROOT
    driver.POSTFIX_ARCHIVE = CAPTURE_ARCHIVE
    driver.POSTFIX_MANIFEST = CAPTURE_MANIFEST
    driver.EXPECTED_POSTFIX_ARCHIVE = EXPECTED["capture_archive"]
    driver.EXPECTED_WRF_TREE = EXPECTED["wrf_tree"]


def run_frozen_driver(driver: Any) -> int:
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if RAW_OUTPUT.exists() or RAW_OUTPUT.is_symlink():
        raise PartitionFailure(f"RAW_OUTPUT_NOT_FRESH:{RAW_OUTPUT}")
    original_argv = sys.argv
    try:
        sys.argv = [str(FROZEN_DRIVER), "--output", str(RAW_OUTPUT)]
        return int(driver.main())
    finally:
        sys.argv = original_argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if output.exists() or output.is_symlink():
            raise PartitionFailure(f"OUTPUT_NOT_FRESH:{output}")
        chain = validate_chain()
        driver = import_file("v0234_qml_partition_driver", FROZEN_DRIVER)
        comparator = import_file("v0234_qml_partition_comparator", FROZEN_COMPARATOR)
        configure_frozen_driver(driver, comparator)
        if run_frozen_driver(driver) != 0:
            raise PartitionFailure("FROZEN_DRIVER_FAILED")
        raw_sha = sha256_file(RAW_OUTPUT)
        raw = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))
        if canonical_without_self(raw) != raw.get("canonical_payload_sha256"):
            raise PartitionFailure("RAW_COMPARATOR_CANONICAL")

        u_rms = float(raw["components"]["u"]["authentic_postfix_adapter_vs_pristine_wrf"]["rms"])
        v_rms = float(raw["components"]["v"]["authentic_postfix_adapter_vs_pristine_wrf"]["rms"])
        strict_u = u_rms < BASELINE["u_rms"]
        strict_v = v_rms < BASELINE["v_rms"]
        accepted = strict_u and strict_v

        proof = dict(raw)
        proof["schema"] = "wrfgpu2-v0234-gpt-land-tsk-partition-v1"
        proof["passed"] = True
        proof["verdict"] = (
            "QML_TSK_FIX_ACCEPTED_STRICT_UV_IMPROVEMENT"
            if accepted else "QML_TSK_FIX_REJECTED_SP2_GATE"
        )
        proof["candidate_acceptance"] = {
            "accepted": accepted,
            "immediately_preceding_authenticated": BASELINE,
            "candidate": {"u_rms": u_rms, "v_rms": v_rms},
            "strict_u_improvement": strict_u,
            "strict_v_improvement": strict_v,
            "u_ratio_candidate_over_baseline": u_rms / BASELINE["u_rms"],
            "v_ratio_candidate_over_baseline": v_rms / BASELINE["v_rms"],
            "policy": "both U and V RMS must be strictly lower",
        }
        proof["qml_tsk_fix"] = {
            "changed_adapter_entry_handles": list(DELTA_FIELDS),
            "first_source_divergence": "Q_ML=QV3D/(1+QV3D)",
            "provenance_proof_sha256": EXPECTED["provenance_proof"],
            "delta_artifact_sha256": EXPECTED["delta"],
            "capture_archive_sha256": EXPECTED["capture_archive"],
            "capture_manifest_sha256": EXPECTED["capture_manifest"],
            **chain,
        }
        proof["frozen_comparator"] = {
            "driver_sha256": EXPECTED["frozen_driver"],
            "comparator_sha256": EXPECTED["frozen_comparator"],
            "audit_reader_sha256": EXPECTED["audit_reader"],
            "raw_output_path": str(RAW_OUTPUT),
            "raw_output_sha256": raw_sha,
            "raw_canonical_payload_sha256": raw["canonical_payload_sha256"],
            "arithmetic_reimplemented_by_wrapper": False,
        }
        comparator.atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "accepted": accepted,
            "verdict": proof["verdict"],
            "output": str(output),
            "baseline": BASELINE,
            "candidate": {"u_rms": u_rms, "v_rms": v_rms},
            "mass_flux_activity": proof.get("mass_flux_activity"),
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
