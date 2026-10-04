#!/usr/bin/env python3
"""Frozen SP2 partition for the final accepted QML + top-buffer endpoint."""

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
CAPTURE_ROOT = (
    SCRATCH
    / "qml-rrtmg-final-accepted-fixed-cpu-adapter/qml-buffer-v2-final"
)
CAPTURE_ARCHIVE = CAPTURE_ROOT / "single-authority-capture.npz"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
CAPTURE_PREFLIGHT = SPRINT / "final-accepted-capture-preflight.json"
CAPTURE_PROOF = SPRINT / "final-accepted-capture-proof.json"
CAPTURE_PROFILE = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_cpu_capture.py"
QML_PROOF = SPRINT / "tsk-provenance-proof.json"
QML_DELTA = SCRATCH / "qml-source-delta-v1.npz"
BUFFER_PROOF = SPRINT / "rrtmg-top-buffer-tsk-proof.json"
BUFFER_DELTA = SCRATCH / "rrtmg-buffer-surface-delta-v1.npz"
PRIOR_CAPTURE = STAGED / "surface-fixed-cpu-adapter/tskin-v1/single-authority-capture.npz"
QML_CAPTURE = SCRATCH / "qml-fixed-cpu-adapter/qml-v1/single-authority-capture.npz"
PREVIOUS_COMBINED_CAPTURE = (
    SCRATCH / "qml-rrtmg-fixed-cpu-adapter/qml-rrtmg-v1/single-authority-capture.npz"
)
PRIOR_PARTITION = PRIOR_SPRINT / "surface-fixed-partition-proof.json"
QML_PARTITION = SPRINT / "qml-partition-proof.json"
PREVIOUS_COMBINED_PARTITION = SPRINT / "rrtmg-partition-proof.json"
WRF_ROOT = STAGED / "wrf-dumps"
RAW_OUTPUT = SCRATCH / "partition/qml-rrtmg-final-accepted-v2-raw.json"
FROZEN_DRIVER = REPO / "scripts/v0234_fable_sp2_postfix_partition.py"
FROZEN_COMPARATOR = REPO / "scripts/v0234_gpt_operand_attribution.py"
AUDIT_READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"

EXPECTED = {
    "capture_archive": "081c8528a1ca46e568669f0086fc4f0b86d70b939b60036096ab7576f919e034",
    "capture_manifest": "6206a9e4e7bcdc056387bde35862251e7b5a3ee52e29b68ef967b117d8aef6e1",
    "capture_manifest_canonical": "ef699756ebbd8ee88384448458b6662b425bb817ff21142d71b57d0b46555133",
    "capture_preflight": "147159aac53cb4f520f38e7da856a27ebb08e5d6e3e33510155852d8067a6214",
    "capture_preflight_canonical": "25bb2f89be19f2beb6f0491bf0848f306d7ac09e17015c8232f6f76547a44423",
    "capture_proof": "3b019b6e3542c1f54a94c4575dd8fc874729d335daf0b0775538ec29504c7d4d",
    "capture_proof_canonical": "22af121883addb1136f6041eb42a5279818719c4d31f26bb038e52bb760dcda4",
    "capture_profile": "ff6d684c6389cf4fe69c22c1e17cf8115e630047e3536ec561dfa5583957a0a2",
    "capture_head": "b4a33d3c230c56cd328fa89d9ff1319f6fa8d310",
    "qml_proof": "140905f7b8998ab4b5a39fa78e2cfc1213e18baccc21479aabad052e2cda26c0",
    "qml_proof_canonical": "c75c4b7ce65b64bd5fc5b2da431fce4de720fe7f6e02a407fe31ce8492cbc5a2",
    "qml_delta": "85dce69d3a9d6a33d699d852ad4779de53a322d96dcd738f7e9a128ae6d8a0a0",
    "buffer_proof": "af78f7a3eba095c2a69b688eb8c3f441b57975a3b500c8802f2f92cc0312917a",
    "buffer_proof_canonical": "ec2eeef215b9faaa356a5e858212993e4959f45dee6677986464371600499879",
    "buffer_delta": "f50ac95bb2e22d5e52a12f4fa4634140dd6df0a273a7183b58f167291cb98d0c",
    "prior_capture": "880ac4d84258d3f8fccc7af3aab86ede89511a05c4be4d8aa9549958fc6fd94c",
    "qml_capture": "2bbb1b184db796b5f5faa02e4040283ba962b2239960ec8319a10817fbfe883d",
    "previous_combined_capture": "081c8528a1ca46e568669f0086fc4f0b86d70b939b60036096ab7576f919e034",
    "prior_partition": "375f915393bc92791648756094144f86e0adc30885968e572e0282c66350f6b0",
    "qml_partition": "dcb620a04eea595169240e0b2cb19d33c0c2e12e381991100071db2a5977e4e5",
    "previous_combined_partition": "4982196f666e3c4d128e67d07f6d6c7668b8da01273b64523446bfa74d0dde2a",
    "previous_combined_partition_canonical": "92c7789490f1d35438fe9118a709010ceb581f4241238c8631e27dd11231b5ce",
    "frozen_driver": "e7a2aebba9fb6460d3863a8a1e03ec9240621ea19a496bc429b23416c35d3b4c",
    "frozen_comparator": "247a1dca01b7aeb35af1adcfd3eb1dc459472c04275edc7f5b2cb3f380df3244",
    "audit_reader": "49bc7b9761e2da1e807000bbbc101d05039a585b8b00f4b31c3b404b66d79d2a",
    "wrf_tree": "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}
BASELINE = {"u_rms": 3.9935109120127676e-7, "v_rms": 7.372784327594862e-7}
PREVIOUS_COMBINED_ENDPOINT = {
    "u_rms": 3.993371444169241e-7,
    "v_rms": 7.373171847791068e-7,
}
NOISE_BOUND = 1.0e-4
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")


PARTITION_PATH = REPO / "scripts/v0234_gpt_land_tsk_partition.py"
SPEC = importlib.util.spec_from_file_location("v0234_qml_partition", PARTITION_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("QML partition helper unavailable")
qml_partition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qml_partition)


class PartitionFailure(RuntimeError):
    """Fail-closed combined evidence or comparator error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "canonical_payload_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def checked_json(path: Path, expected_sha: str, expected_canonical: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise PartitionFailure(f"EVIDENCE_HASH:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = canonical_without_self(value)
    if canonical != value.get("canonical_payload_sha256"):
        raise PartitionFailure(f"EVIDENCE_CANONICAL:{path}")
    if expected_canonical is not None and canonical != expected_canonical:
        raise PartitionFailure(f"EVIDENCE_PIN:{path}")
    return value


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


def wrapper_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if disk != head:
        raise PartitionFailure("WRAPPER_NOT_HEAD")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", f"HEAD:{relative}"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip(),
    }


def validate_chain() -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if tracked:
        raise PartitionFailure(f"TRACKED_WORKTREE_NOT_CLEAN:{tracked}")
    files = {
        "capture_archive": CAPTURE_ARCHIVE,
        "capture_profile": CAPTURE_PROFILE,
        "qml_delta": QML_DELTA,
        "buffer_delta": BUFFER_DELTA,
        "prior_capture": PRIOR_CAPTURE,
        "qml_capture": QML_CAPTURE,
        "previous_combined_capture": PREVIOUS_COMBINED_CAPTURE,
        "prior_partition": PRIOR_PARTITION,
        "qml_partition": QML_PARTITION,
        "previous_combined_partition": PREVIOUS_COMBINED_PARTITION,
        "frozen_driver": FROZEN_DRIVER,
        "frozen_comparator": FROZEN_COMPARATOR,
        "audit_reader": AUDIT_READER,
    }
    for label, path in files.items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != EXPECTED[label]:
            raise PartitionFailure(f"INPUT_HASH:{label}:{path}")

    manifest = checked_json(
        CAPTURE_MANIFEST,
        EXPECTED["capture_manifest"],
        EXPECTED["capture_manifest_canonical"],
    )
    preflight = checked_json(
        CAPTURE_PREFLIGHT,
        EXPECTED["capture_preflight"],
        EXPECTED["capture_preflight_canonical"],
    )
    capture_proof = checked_json(
        CAPTURE_PROOF, EXPECTED["capture_proof"], EXPECTED["capture_proof_canonical"]
    )
    qml_proof = checked_json(
        QML_PROOF, EXPECTED["qml_proof"], EXPECTED["qml_proof_canonical"]
    )
    buffer_proof = checked_json(
        BUFFER_PROOF, EXPECTED["buffer_proof"], EXPECTED["buffer_proof_canonical"]
    )
    previous_combined_partition = checked_json(
        PREVIOUS_COMBINED_PARTITION,
        EXPECTED["previous_combined_partition"],
        EXPECTED["previous_combined_partition_canonical"],
    )
    qml_record = preflight.get("qml_tsk_correction", {})
    buffer_record = preflight.get("rrtmg_lw_top_buffer_correction", {})
    authority = manifest.get("authority", {})
    if not (
        manifest.get("passed") is True
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
        and preflight.get("passed") is True
        and preflight.get("git", {}).get("head") == EXPECTED["capture_head"]
        and preflight.get("combined_profile", {}).get("sha256") == EXPECTED["capture_profile"]
        and capture_proof.get("passed") is True
        and capture_proof.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and capture_proof.get("manifest", {}).get("sha256") == EXPECTED["capture_manifest"]
        and capture_proof.get("git", {}).get("head") == EXPECTED["capture_head"]
        and qml_record.get("proof_sha256") == EXPECTED["qml_proof"]
        and qml_record.get("artifact_sha256") == EXPECTED["qml_delta"]
        and buffer_record.get("proof_sha256") == EXPECTED["buffer_proof"]
        and buffer_record.get("artifact_sha256") == EXPECTED["buffer_delta"]
        and qml_proof.get("passed") is True
        and buffer_proof.get("passed") is True
        and buffer_proof.get("tsk_parity", {}).get("land_rms_and_max_strictly_improve") is True
        and previous_combined_partition.get("passed") is True
        and previous_combined_partition.get("gate_split_localization", {}).get("localized") is True
    ):
        raise PartitionFailure("AUTHORITY_CHAIN")

    archive_names = {
        "theta_flux": "entry_state_theta_flux",
        "qv_flux": "entry_state_qv_flux",
        "fltv": "entry_state_fltv",
        "t_skin": "surface_terms_flux_t_skin",
    }
    closure: dict[str, Any] = {}
    activity_identity: dict[str, Any] = {}
    with (
        np.load(QML_DELTA, allow_pickle=False) as qml_delta,
        np.load(BUFFER_DELTA, allow_pickle=False) as buffer_delta,
        np.load(CAPTURE_ARCHIVE, allow_pickle=False) as current,
        np.load(PRIOR_CAPTURE, allow_pickle=False) as previous,
        np.load(QML_CAPTURE, allow_pickle=False) as qml_capture,
        np.load(PREVIOUS_COMBINED_CAPTURE, allow_pickle=False) as previous_combined,
    ):
        for field, archive_name in archive_names.items():
            expected = (
                np.asarray(previous[archive_name], dtype=np.float64)
                + np.asarray(qml_delta[f"{field}_delta"], dtype=np.float64)
                + np.asarray(buffer_delta[f"{field}_delta"], dtype=np.float64)
            )
            closure[field] = exact_delta(current[archive_name], expected)
            if closure[field]["max_abs"] != 0.0:
                raise PartitionFailure(f"COMPOSED_DELTA_CLOSURE:{field}")
        for name in ("ustar", "rhosfc"):
            record = exact_delta(current[f"entry_state_{name}"], previous[f"entry_state_{name}"])
            if record["max_abs"] != 0.0:
                raise PartitionFailure(f"UNCHANGED_HANDLE:{name}")
        for field in ("s_aw", "s_awu", "s_awv"):
            current_mask = np.asarray(current[f"mass_flux_{field}"]) != 0.0
            previous_mask = (
                np.asarray(previous_combined[f"mass_flux_{field}"]) != 0.0
            )
            qml_mask = np.asarray(qml_capture[f"mass_flux_{field}"]) != 0.0
            record = exact_delta(
                current_mask.astype(np.uint8), previous_mask.astype(np.uint8)
            )
            qml_record = exact_delta(
                current_mask.astype(np.uint8), qml_mask.astype(np.uint8)
            )
            record["active_count"] = int(np.sum(current_mask))
            record["mask_sha256"] = hashlib.sha256(
                np.ascontiguousarray(current_mask).tobytes()
            ).hexdigest()
            if record["max_abs"] != 0.0:
                raise PartitionFailure(f"MASS_FLUX_ACTIVITY_MOVED:{field}")
            if qml_record["max_abs"] != 0.0:
                raise PartitionFailure(f"MASS_FLUX_ACTIVITY_MOVED_VS_QML:{field}")
            record["vs_qml_capture"] = qml_record
            activity_identity[field] = record
    return {
        "composed_entry_exact_closure": closure,
        "mass_flux_activity_masks_vs_previous_combined_and_qml": activity_identity,
        "wrapper": wrapper_record(),
    }


def configure_frozen_driver() -> Any:
    driver = qml_partition.import_file("v0234_rrtmg_partition_driver", FROZEN_DRIVER)
    comparator = qml_partition.import_file(
        "v0234_rrtmg_partition_comparator", FROZEN_COMPARATOR
    )
    comparator.WRF_ROOT = WRF_ROOT

    def staged_reader() -> tuple[Any, dict[str, Any]]:
        reader = qml_partition.import_file("v0234_rrtmg_partition_reader", AUDIT_READER)
        return reader.IndependentWrfDump(WRF_ROOT), {
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
    return driver, comparator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if output.exists() or output.is_symlink() or RAW_OUTPUT.exists() or RAW_OUTPUT.is_symlink():
            raise PartitionFailure("OUTPUT_NOT_FRESH")
        chain = validate_chain()
        driver, comparator = configure_frozen_driver()
        RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        original_argv = sys.argv
        try:
            sys.argv = [str(FROZEN_DRIVER), "--output", str(RAW_OUTPUT)]
            status = int(driver.main())
        finally:
            sys.argv = original_argv
        if status != 0:
            raise PartitionFailure("FROZEN_DRIVER_FAILED")
        raw_sha = sha256_file(RAW_OUTPUT)
        raw = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))
        if canonical_without_self(raw) != raw.get("canonical_payload_sha256"):
            raise PartitionFailure("RAW_COMPARATOR_CANONICAL")

        u_rms = float(raw["components"]["u"]["authentic_postfix_adapter_vs_pristine_wrf"]["rms"])
        v_rms = float(raw["components"]["v"]["authentic_postfix_adapter_vs_pristine_wrf"]["rms"])
        u_ratio = u_rms / BASELINE["u_rms"]
        v_ratio = v_rms / BASELINE["v_rms"]
        u_pass = u_ratio <= 1.0 + NOISE_BOUND
        v_pass = v_ratio <= 1.0 + NOISE_BOUND
        accepted = bool(u_pass and v_pass)
        mass_activity = raw["mass_flux_activity"]
        port_only_counts = {
            field: int(mass_activity[field]["columns_port_only"])
            for field in ("s_aw", "s_awu", "s_awv")
        }
        localized = bool(
            all(count == 4526 for count in port_only_counts.values())
            and all(
                record["max_abs"] == 0.0
                for record in chain[
                    "mass_flux_activity_masks_vs_previous_combined_and_qml"
                ].values()
            )
        )

        proof = dict(raw)
        proof["schema"] = "wrfgpu2-v0234-gpt-land-tsk-final-accepted-partition-v1"
        proof["passed"] = accepted and localized
        proof["verdict"] = (
            "QML_RRTMG_TOP_BUFFER_FINAL_ACCEPTED_NOISE_GATE_GATE_SPLIT_LOCALIZED"
            if proof["passed"]
            else "LAND_TSK_FIXES_FAIL_TERMINAL_GATE"
        )
        proof["candidate_acceptance"] = {
            "accepted": accepted,
            "e653bdbf_baseline": BASELINE,
            "immediately_preceding_qml_rrtmg_endpoint": PREVIOUS_COMBINED_ENDPOINT,
            "candidate": {"u_rms": u_rms, "v_rms": v_rms},
            "relative_noise_bound": NOISE_BOUND,
            "u_ratio_candidate_over_baseline": u_ratio,
            "v_ratio_candidate_over_baseline": v_ratio,
            "u_relative_change_vs_baseline": u_ratio - 1.0,
            "v_relative_change_vs_baseline": v_ratio - 1.0,
            "u_within_noise_bound": u_pass,
            "v_within_noise_bound": v_pass,
            "u_ratio_candidate_over_previous_combined": (
                u_rms / PREVIOUS_COMBINED_ENDPOINT["u_rms"]
            ),
            "v_ratio_candidate_over_previous_combined": (
                v_rms / PREVIOUS_COMBINED_ENDPOINT["v_rms"]
            ),
            "policy": "each endpoint may worsen by at most 1e-4 relative to e653bdbf",
        }
        proof["tsk_fix_chain"] = {
            "qml": {
                "proof_sha256": EXPECTED["qml_proof"],
                "delta_sha256": EXPECTED["qml_delta"],
            },
            "rrtmg_lw_top_buffer": {
                "proof_sha256": EXPECTED["buffer_proof"],
                "delta_sha256": EXPECTED["buffer_delta"],
            },
            "capture_archive_sha256": EXPECTED["capture_archive"],
            **chain,
        }
        proof["gate_split_localization"] = {
            "localized": localized,
            "port_only_columns": port_only_counts,
            "activity_masks_bitwise_identical_to_previous_combined_and_qml": True,
            "interpretation": (
                "All three tested land-surface corrections leave all MYNN mass-flux "
                "activity masks bitwise unchanged; the 4,526-column split is "
                "therefore outside the corrected TSK/Noah-MP forcing path and "
                "remains localized to the MYNN mass-flux activation family."
            ),
            "combined_vector_mass_flux_shapley": raw["combined_vector_shapley"]["mass_flux"],
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
        print(
            json.dumps(
                {
                    "passed": proof["passed"],
                    "verdict": proof["verdict"],
                    "candidate": {"u_rms": u_rms, "v_rms": v_rms},
                    "ratios": {"u": u_ratio, "v": v_ratio},
                    "port_only_counts": port_only_counts,
                    "mass_flux_shapley_fraction": raw["combined_vector_shapley"][
                        "mass_flux"
                    ]["signed_projection_fraction_of_authentic_error_sse"],
                    "output": str(output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if proof["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - partition proof must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
