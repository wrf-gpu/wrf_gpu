#!/usr/bin/env python3
"""Frozen SP2 partition for the QML + static-top + exact O3RAD endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
PRIOR_SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
BASE_SCRIPT = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_partition.py"
CAPTURE_ROOT = SCRATCH / "qml-rrtmg-o3-fixed-cpu-adapter/o3-qml-buffer-v1"
CAPTURE_ARCHIVE = CAPTURE_ROOT / "single-authority-capture.npz"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
CAPTURE_PREFLIGHT = SPRINT / "o3-capture-preflight.json"
CAPTURE_PROOF = SPRINT / "o3-capture-proof.json"
CAPTURE_PROFILE = REPO / "scripts/v0234_gpt_land_tsk_o3_cpu_capture.py"
QML_PROOF = SPRINT / "tsk-provenance-proof.json"
QML_DELTA = SCRATCH / "qml-source-delta-v1.npz"
BUFFER_PROOF = SPRINT / "rrtmg-top-buffer-tsk-proof.json"
BUFFER_DELTA = SCRATCH / "rrtmg-buffer-surface-delta-v1.npz"
O3_TSK_PROOF = SPRINT / "rrtmg-o3-tsk-proof.json"
O3_REAL_WRF_PROOF = SPRINT / "rrtmg-o3-real-wrf-proof.json"
O3_DELTA = SCRATCH / "rrtmg-o3-surface-delta-v3.npz"
PRIOR_CAPTURE = STAGED / "surface-fixed-cpu-adapter/tskin-v1/single-authority-capture.npz"
QML_CAPTURE = SCRATCH / "qml-fixed-cpu-adapter/qml-v1/single-authority-capture.npz"
PREVIOUS_CAPTURE = (
    SCRATCH
    / "qml-rrtmg-final-accepted-fixed-cpu-adapter/qml-buffer-v2-final"
    / "single-authority-capture.npz"
)
PREVIOUS_PARTITION = SPRINT / "final-accepted-partition-proof.json"
BASE_OUTPUT = SCRATCH / "partition/o3-qml-buffer-v1-base-proof.json"
RAW_OUTPUT = SCRATCH / "partition/o3-qml-buffer-v1-raw.json"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")

EXPECTED = {
    "capture_archive": "5cb2ab598fabcac92fecec7560d649d5c13de92608641c70792cd71f70742fc9",
    "capture_manifest": "c92f69f72ac87cabdcfafda539291dde4f162aa242794f7839be50e1d913cf98",
    "capture_manifest_canonical": "2b18f5ff5636dc8bc8e705116ce3862844279f9455a4d3ffe0af2fea95203b0c",
    "capture_preflight": "a4700d93f4c9f4673014e2de8251424cb454daeabcd40bfcce7cf4af6c8c539b",
    "capture_preflight_canonical": "139f9cd0827a3c7af12b07303f1ef4a301e13386ae70dba3d919e058f060e7dd",
    "capture_proof": "f1ffec002f1e2f2e0b709eb43b4e832373fc3cf61008e46f092121ebb28f0054",
    "capture_proof_canonical": "c5c59cb8e6b2058f3a7ab6e3d8060a4ea5703a127bf158160a68e1142e6aa041",
    "capture_profile": "d7d67bb008b0dab00b9f60188ac16b8462a7999d20de9dfb7ddfd21175523774",
    "capture_head": "e85cd01293a674aecab5902891ae92ab29ddbca1",
    "qml_proof": "140905f7b8998ab4b5a39fa78e2cfc1213e18baccc21479aabad052e2cda26c0",
    "qml_proof_canonical": "c75c4b7ce65b64bd5fc5b2da431fce4de720fe7f6e02a407fe31ce8492cbc5a2",
    "qml_delta": "85dce69d3a9d6a33d699d852ad4779de53a322d96dcd738f7e9a128ae6d8a0a0",
    "buffer_proof": "af78f7a3eba095c2a69b688eb8c3f441b57975a3b500c8802f2f92cc0312917a",
    "buffer_proof_canonical": "ec2eeef215b9faaa356a5e858212993e4959f45dee6677986464371600499879",
    "buffer_delta": "f50ac95bb2e22d5e52a12f4fa4634140dd6df0a273a7183b58f167291cb98d0c",
    "o3_tsk_proof": "3505800030964c8706b631d9ed009a1194696854f4471cebe3798a1986eef6ac",
    "o3_tsk_proof_canonical": "411413c2262710dbb5fb65fe0ba45de91882d4bbc326947d5e27310bc146b7bd",
    "o3_real_wrf_proof": "333d716259c1d012b7db6d7092717611d3c708113ddf247683161872d753e4d9",
    "o3_real_wrf_proof_canonical": "62b84106b003c92bd5ded1eb0290912c1d12f50bbebf7e9ce7c2f15cb534d545",
    "o3_delta": "812e3a2fba5872b456efbb773aaa736c27dd78fbfa611f7dfb69c36a820a9de9",
    "prior_capture": "880ac4d84258d3f8fccc7af3aab86ede89511a05c4be4d8aa9549958fc6fd94c",
    "qml_capture": "2bbb1b184db796b5f5faa02e4040283ba962b2239960ec8319a10817fbfe883d",
    "previous_capture": "081c8528a1ca46e568669f0086fc4f0b86d70b939b60036096ab7576f919e034",
    "previous_partition": "f4ae01b8564c286d471b33e0b98647edbaef793ab04a03c58a2d2927ff092717",
    "previous_partition_canonical": "d4d235d29f6dfbc22fe3a2c7a645ebd0e08533aea8f7bf8131bb38f20b11c79b",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"partition helper unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("v0234_o3_partition_base", BASE_SCRIPT)


class O3PartitionFailure(RuntimeError):
    """Fail-closed evidence or frozen-comparator failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(record: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in record.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _checked(path: Path, digest: str, canonical: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
        raise O3PartitionFailure(f"EVIDENCE_HASH:{path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    observed = _canonical(record)
    if observed != record.get("canonical_payload_sha256"):
        raise O3PartitionFailure(f"EVIDENCE_CANONICAL:{path}")
    if canonical is not None and observed != canonical:
        raise O3PartitionFailure(f"EVIDENCE_PIN:{path}")
    return record


def _tracked_wrapper() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if disk != head:
        raise O3PartitionFailure("WRAPPER_NOT_HEAD")
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


def _configure_base() -> None:
    base.CAPTURE_ROOT = CAPTURE_ROOT
    base.CAPTURE_ARCHIVE = CAPTURE_ARCHIVE
    base.CAPTURE_MANIFEST = CAPTURE_MANIFEST
    base.CAPTURE_PREFLIGHT = CAPTURE_PREFLIGHT
    base.CAPTURE_PROOF = CAPTURE_PROOF
    base.CAPTURE_PROFILE = CAPTURE_PROFILE
    base.PREVIOUS_COMBINED_CAPTURE = PREVIOUS_CAPTURE
    base.PREVIOUS_COMBINED_PARTITION = PREVIOUS_PARTITION
    base.RAW_OUTPUT = RAW_OUTPUT
    base.EXPECTED.update(
        {
            "capture_archive": EXPECTED["capture_archive"],
            "capture_manifest": EXPECTED["capture_manifest"],
            "capture_manifest_canonical": EXPECTED["capture_manifest_canonical"],
            "capture_preflight": EXPECTED["capture_preflight"],
            "capture_preflight_canonical": EXPECTED["capture_preflight_canonical"],
            "capture_proof": EXPECTED["capture_proof"],
            "capture_proof_canonical": EXPECTED["capture_proof_canonical"],
            "capture_profile": EXPECTED["capture_profile"],
            "capture_head": EXPECTED["capture_head"],
            "previous_combined_capture": EXPECTED["previous_capture"],
            "previous_combined_partition": EXPECTED["previous_partition"],
            "previous_combined_partition_canonical": EXPECTED[
                "previous_partition_canonical"
            ],
        }
    )


def _validate_chain() -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if tracked:
        raise O3PartitionFailure(f"TRACKED_WORKTREE_NOT_CLEAN:{tracked}")
    files = {
        "capture_archive": CAPTURE_ARCHIVE,
        "capture_profile": CAPTURE_PROFILE,
        "qml_delta": QML_DELTA,
        "buffer_delta": BUFFER_DELTA,
        "o3_delta": O3_DELTA,
        "prior_capture": PRIOR_CAPTURE,
        "qml_capture": QML_CAPTURE,
        "previous_capture": PREVIOUS_CAPTURE,
    }
    for label, path in files.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != EXPECTED[label]:
            raise O3PartitionFailure(f"INPUT_HASH:{label}:{path}")

    manifest = _checked(
        CAPTURE_MANIFEST,
        EXPECTED["capture_manifest"],
        EXPECTED["capture_manifest_canonical"],
    )
    preflight = _checked(
        CAPTURE_PREFLIGHT,
        EXPECTED["capture_preflight"],
        EXPECTED["capture_preflight_canonical"],
    )
    capture_proof = _checked(
        CAPTURE_PROOF,
        EXPECTED["capture_proof"],
        EXPECTED["capture_proof_canonical"],
    )
    qml = _checked(
        QML_PROOF, EXPECTED["qml_proof"], EXPECTED["qml_proof_canonical"]
    )
    buffer = _checked(
        BUFFER_PROOF,
        EXPECTED["buffer_proof"],
        EXPECTED["buffer_proof_canonical"],
    )
    o3_tsk = _checked(
        O3_TSK_PROOF,
        EXPECTED["o3_tsk_proof"],
        EXPECTED["o3_tsk_proof_canonical"],
    )
    o3_real = _checked(
        O3_REAL_WRF_PROOF,
        EXPECTED["o3_real_wrf_proof"],
        EXPECTED["o3_real_wrf_proof_canonical"],
    )
    previous = _checked(
        PREVIOUS_PARTITION,
        EXPECTED["previous_partition"],
        EXPECTED["previous_partition_canonical"],
    )
    authority = manifest.get("authority", {})
    o3_record = preflight.get("wrf_o3rad_correction", {})
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
        and manifest.get("input_grid", {}).get("wrfinput_d03_sha256")
        == EXPECTED["wrfinput"]
        and preflight.get("passed") is True
        and preflight.get("git", {}).get("head") == EXPECTED["capture_head"]
        and preflight.get("o3_capture_profile", {}).get("sha256")
        == EXPECTED["capture_profile"]
        and capture_proof.get("passed") is True
        and capture_proof.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and capture_proof.get("manifest", {}).get("sha256")
        == EXPECTED["capture_manifest"]
        and capture_proof.get("git", {}).get("head") == EXPECTED["capture_head"]
        and o3_record.get("tsk_proof_sha256") == EXPECTED["o3_tsk_proof"]
        and o3_record.get("real_wrf_proof_sha256")
        == EXPECTED["o3_real_wrf_proof"]
        and o3_record.get("delta_sha256") == EXPECTED["o3_delta"]
        and o3_record.get("changed_state_handles") == list(DELTA_FIELDS)
        and o3_record.get("source_authorized") is True
        and o3_record.get("real_wrf_glw_strictly_improves") is True
        and o3_record.get("real_wrf_swdnb_strictly_improves") is True
        and o3_record.get("land_tsk_strictly_improves") is True
        and o3_record.get("water_delta_bitwise_zero") is True
        and qml.get("passed") is True
        and buffer.get("passed") is True
        and o3_tsk.get("passed") is True
        and o3_real.get("status") == "PASS"
        and previous.get("passed") is True
        and previous.get("gate_split_localization", {}).get("localized") is True
    ):
        raise O3PartitionFailure("AUTHORITY_CHAIN")

    archive_names = {
        "theta_flux": "entry_state_theta_flux",
        "qv_flux": "entry_state_qv_flux",
        "fltv": "entry_state_fltv",
        "t_skin": "surface_terms_flux_t_skin",
    }
    closure: dict[str, Any] = {}
    activity: dict[str, Any] = {}
    with (
        np.load(QML_DELTA, allow_pickle=False) as qml_delta,
        np.load(BUFFER_DELTA, allow_pickle=False) as buffer_delta,
        np.load(O3_DELTA, allow_pickle=False) as o3_delta,
        np.load(CAPTURE_ARCHIVE, allow_pickle=False) as current,
        np.load(PRIOR_CAPTURE, allow_pickle=False) as prior,
        np.load(QML_CAPTURE, allow_pickle=False) as qml_capture,
        np.load(PREVIOUS_CAPTURE, allow_pickle=False) as previous_capture,
    ):
        for field, archive_name in archive_names.items():
            expected = (
                np.asarray(prior[archive_name], dtype=np.float64)
                + np.asarray(qml_delta[f"{field}_delta"], dtype=np.float64)
                + np.asarray(buffer_delta[f"{field}_delta"], dtype=np.float64)
                + np.asarray(o3_delta[f"{field}_delta"], dtype=np.float64)
            )
            closure[field] = base.exact_delta(current[archive_name], expected)
            if closure[field]["max_abs"] != 0.0:
                raise O3PartitionFailure(f"COMPOSED_DELTA_CLOSURE:{field}")
        for name in ("ustar", "rhosfc"):
            unchanged = base.exact_delta(
                current[f"entry_state_{name}"], prior[f"entry_state_{name}"]
            )
            if unchanged["max_abs"] != 0.0:
                raise O3PartitionFailure(f"UNCHANGED_HANDLE:{name}")
        for field in ("s_aw", "s_awu", "s_awv"):
            current_mask = np.asarray(current[f"mass_flux_{field}"]) != 0.0
            previous_mask = (
                np.asarray(previous_capture[f"mass_flux_{field}"]) != 0.0
            )
            qml_mask = np.asarray(qml_capture[f"mass_flux_{field}"]) != 0.0
            record = base.exact_delta(
                current_mask.astype(np.uint8), previous_mask.astype(np.uint8)
            )
            record["vs_qml_capture"] = base.exact_delta(
                current_mask.astype(np.uint8), qml_mask.astype(np.uint8)
            )
            record["active_count"] = int(np.sum(current_mask))
            record["mask_sha256"] = hashlib.sha256(
                np.ascontiguousarray(current_mask).tobytes()
            ).hexdigest()
            activity[field] = record
    return {
        "composed_entry_exact_closure": closure,
        "mass_flux_activity_masks_vs_previous_combined_and_qml": activity,
        "wrf_o3rad": {
            "tsk_proof_sha256": EXPECTED["o3_tsk_proof"],
            "real_wrf_proof_sha256": EXPECTED["o3_real_wrf_proof"],
            "delta_sha256": EXPECTED["o3_delta"],
            "land_tsk_rms": o3_tsk["tsk_parity"][
                "candidate_qml_plus_static_top_plus_o3rad"
            ]["land"]["rms"],
            "land_tsk_max_abs": o3_tsk["tsk_parity"][
                "candidate_qml_plus_static_top_plus_o3rad"
            ]["land"]["max_abs"],
            "water_bitwise_invariant": True,
        },
        "wrapper": _tracked_wrapper(),
    }


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise O3PartitionFailure("OUTPUT_NOT_FRESH")
    record.pop("canonical_payload_sha256", None)
    record["canonical_payload_sha256"] = _canonical(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if output.exists() or output.is_symlink() or BASE_OUTPUT.exists() or BASE_OUTPUT.is_symlink():
            raise O3PartitionFailure("OUTPUT_NOT_FRESH")
        _configure_base()
        base.validate_chain = _validate_chain
        original_argv = sys.argv
        try:
            sys.argv = [str(BASE_SCRIPT), "--output", str(BASE_OUTPUT)]
            status = int(base.main())
        finally:
            sys.argv = original_argv
        if not BASE_OUTPUT.is_file():
            raise O3PartitionFailure("BASE_PARTITION_OUTPUT_MISSING")
        proof = json.loads(BASE_OUTPUT.read_text(encoding="utf-8"))
        proof["schema"] = "wrfgpu2-v0234-qml-static-top-o3-partition-v1"
        proof["verdict"] = (
            "WRF_O3RAD_ACCEPTED_SP2_NOISE_GATE_GATE_SPLIT_LOCALIZED"
            if proof.get("passed") is True
            else "WRF_O3RAD_REJECTED_SP2_OR_GATE_SPLIT_GATE"
        )
        proof["candidate_acceptance"]["composition"] = (
            "QML + static LW top pressure + exact WRF O3RAD on LW and SW"
        )
        proof["candidate_acceptance"]["production_retention"] = bool(
            proof.get("passed") is True
        )
        proof["tsk_fix_chain"]["wrf_o3rad"] = {
            "tsk_proof_sha256": EXPECTED["o3_tsk_proof"],
            "real_wrf_proof_sha256": EXPECTED["o3_real_wrf_proof"],
            "delta_sha256": EXPECTED["o3_delta"],
            "capture_archive_sha256": EXPECTED["capture_archive"],
        }
        proof["gate_split_localization"]["interpretation"] = (
            "Exact QML, static-top, and O3RAD corrections leave all three MYNN "
            "mass-flux activity masks bitwise unchanged from the accepted pre-O3 "
            "capture. The 4,526-column split is therefore outside the corrected "
            "TSK/Noah-MP forcing path and remains localized to MYNN activation."
        )
        proof["o3_partition_wrapper"] = {
            "base_output_path": str(BASE_OUTPUT),
            "base_output_sha256": _sha256(BASE_OUTPUT),
            "base_canonical_payload_sha256": proof.get("canonical_payload_sha256"),
            "arithmetic_reimplemented": False,
        }
        _atomic_json(output, proof)
        candidate = proof["candidate_acceptance"]["candidate"]
        ratios = {
            "u": proof["candidate_acceptance"]["u_ratio_candidate_over_baseline"],
            "v": proof["candidate_acceptance"]["v_ratio_candidate_over_baseline"],
        }
        print(
            json.dumps(
                {
                    "passed": proof["passed"],
                    "verdict": proof["verdict"],
                    "candidate": candidate,
                    "ratios": ratios,
                    "port_only_counts": proof["gate_split_localization"][
                        "port_only_columns"
                    ],
                    "output": str(output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if proof.get("passed") is True and status == 0 else 1
    except Exception as exc:  # noqa: BLE001 - partition must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
