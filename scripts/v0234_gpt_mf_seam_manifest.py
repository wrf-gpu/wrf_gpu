#!/usr/bin/env python3
"""Build the canonical v0234 radiation/MYNN seam proof manifest."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-mf-seam-closure"
BASE = "6045ff02f263f52614f9f21de19550615b847af0"
CAPTURE_HEAD = "2694609c73e02480adda8f1d665b085ef7e35238"
PROMPT = Path("/tmp/v0234_gpt_mf_seam_prompt.txt")
RESOLUTION = (
    REPO
    / ".agent/sprints/2026-07-20-v0234-opus-sp2-batch-resolution/RESOLUTION.md"
)
RESOLUTION_PROOF = (
    REPO
    / ".agent/sprints/2026-07-20-v0234-opus-sp2-batch-resolution/"
    "resolution-audit-proof.json"
)
CAPTURE_ARCHIVE = Path(
    "/tmp/v0234_gpt_mf_seam_capture/endpoint-v2/"
    "single-authority-capture.npz"
)
CAPTURE_MANIFEST = CAPTURE_ARCHIVE.parent / "manifest.json"
RADIATION_DELTA = Path("/tmp/v0234_gpt_mf_seam_combined_radiation_delta.npz")

ARTIFACTS = {
    "contract": SPRINT / "CONTRACT.md",
    "manager_question": SPRINT / "GPT_MANAGER_QUESTION.md",
    "combined_radiation_tsk": SPRINT / "combined-radiation-tsk-proof.json",
    "combined_capture_preflight": SPRINT / "combined-capture-preflight.json",
    "combined_capture": SPRINT / "combined-capture-proof.json",
    "channel_raw": SPRINT / "channel-decomposed-sp2-raw.json",
    "channel_gate": SPRINT / "channel-decomposed-sp2-gate.json",
    "activation_localization": SPRINT / "activation-localization-parity-proof.json",
    "radiation_readmission": SPRINT / "radiation-readmission-proof.json",
    "drift": SPRINT / "drift-proof.json",
    "focused_junit": SPRINT / "focused-tests.xml",
    "exploratory_capture_preflight": SPRINT / "mf-seam-capture-preflight.json",
    "exploratory_capture": SPRINT / "mf-seam-capture-proof.json",
}
EXPECTED = {
    PROMPT: "4a3c171681026f3250314abd32c907603c388d4c1ed72a94d61b8cce2b843d43",
    RESOLUTION: "ce243564abf92e8ea091e28aeb9e79401f25477426290cf82c5df9e8e02ffba3",
    RESOLUTION_PROOF: "c7f0fdbeaac8e0776a899195307210a284e9f0fa53a6d7cc3031a876460d4c3e",
    ARTIFACTS["contract"]: "39e41311e5d96d8c459148e3f1fa03444d32c9f60af33f887bad627618c683e2",
    ARTIFACTS["manager_question"]: "5375a0e851f6786a84861d82fe5c2592da1b1776119eff8794723f067f341365",
    ARTIFACTS["combined_radiation_tsk"]: "6cbc55d2c0f9018f90110abb963e3fe75e7d307b80f873cba967ddc12d5236a2",
    ARTIFACTS["combined_capture_preflight"]: "0880fe708914c9e4ad1187b4a02d6b6a7ba9bcaef01da3425551534a3aa3f24e",
    ARTIFACTS["combined_capture"]: "e2b905573884f2839c3858a75494b50870874c1a60f4d1bacdff2bd4128ca11b",
    ARTIFACTS["channel_raw"]: "43d43673706c76758f52036a0d950c874d6b529aa786be3e58be3b2bb9d1d9bb",
    ARTIFACTS["channel_gate"]: "e73bbcf104575676ea3482838577e87a1914954249919fbe167e642a10168c25",
    ARTIFACTS["activation_localization"]: "b3e261eabd2bb3e9f90ae69ad634c626f05d58ca2240e0902fb1c01eeaf57bce",
    ARTIFACTS["radiation_readmission"]: "c3a530e21b798cab8ecccfa308227093b203c5a0183d316d9c7beedc4596ec50",
    ARTIFACTS["drift"]: "12aa0dc1bd9b9939a7cfd66e58a660b7667e9be57bc37440086f7c4deee3a233",
    ARTIFACTS["focused_junit"]: "df0134daab94f89c20179cb1730e02939482d98e8c596dc116193071fdc3d422",
    ARTIFACTS["exploratory_capture_preflight"]: "eb9b27c80dea0d2e86eaba578b9647b86832ed46f68921708eb6bd5b541f621a",
    ARTIFACTS["exploratory_capture"]: "0f74b6155d593e66a98abc3186af3d6cd99d8c1ff02e139a81c70720b1c7b667",
    CAPTURE_ARCHIVE: "882bd72120011dbe76fa82d52379bf29a5b42f0e0a7fceb4811bd9a2a4e68e11",
    CAPTURE_MANIFEST: "db2b3949af18cc0ebf0c762235509fd3548e218375a31c3cb35a875fb8365b8a",
    RADIATION_DELTA: "1c2a5beb17960cc3c5a07bff9b0b71b6f21a233175dad0be9e14b5e8d0e356e8",
}
PRODUCTION = {
    REPO / "src/gpuwrf/coupling/physics_couplers.py":
        "84440c116f2f8309fa76390fdeea15ad55c58bb05c8bf783af79895411c537a2",
    REPO / "src/gpuwrf/physics/mynn_edmf.py":
        "2aeeffae999369c9c334f2db05595ac6044a2d9feb8c78bda549b0f89fa34ff0",
    REPO / "src/gpuwrf/physics/mynn_pbl.py":
        "bad38f5bd6d7cb82e57f6c24016152f00688559251acb48dbab232cdb9b82b68",
    REPO / "tests/test_mynn_edmf_oracle.py":
        "b60cf092da475dcacc19d37de84802fbdbb016d321f83c09fc05a8a5b586ca7c",
    REPO / "tests/test_v014_dry_source_leaf_wiring.py":
        "f21be51b469471922906521dd11989d361c7b8d47ec90606e49df3b60beeb58a",
}


class ManifestFailure(RuntimeError):
    """Fail-closed manifest input or semantic failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ManifestFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ManifestFailure(f"TEMP_NOT_FRESH:{temporary}")
    with temporary.open("xb") as stream:
        stream.write(
            (
                json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode()
        )
        stream.flush()
    os.link(temporary, path)
    temporary.unlink()


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr.decode() if binary else result.stderr
        raise ManifestFailure(f"GIT:{' '.join(args)}:{stderr.strip()}")
    return result.stdout if binary else result.stdout.strip()


def load_canonical(name: str) -> dict[str, Any]:
    path = ARTIFACTS[name]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManifestFailure(f"JSON_OBJECT:{name}")
    if canonical(value) != value.get("canonical_payload_sha256"):
        raise ManifestFailure(f"CANONICAL:{name}")
    return value


def validate_inputs(approved_head: str) -> dict[str, Any]:
    if git("rev-parse", "HEAD") != approved_head:
        raise ManifestFailure("APPROVED_HEAD_DRIFT")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ManifestFailure("TRACKED_WORKTREE_NOT_CLEAN")
    for path, expected in {**EXPECTED, **PRODUCTION}.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ManifestFailure(f"HASH:{path}:{observed}")
    relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    disk = Path(__file__).resolve().read_bytes()
    if disk != git("show", f"HEAD:{relative}", binary=True):
        raise ManifestFailure("SCRIPT_NOT_HEAD")

    proofs = {
        name: load_canonical(name)
        for name in (
            "combined_radiation_tsk",
            "combined_capture_preflight",
            "combined_capture",
            "channel_raw",
            "channel_gate",
            "activation_localization",
            "radiation_readmission",
            "drift",
            "exploratory_capture_preflight",
            "exploratory_capture",
        )
    }
    resolution = json.loads(RESOLUTION_PROOF.read_text(encoding="utf-8"))
    if canonical(resolution) != resolution.get("canonical_payload_sha256"):
        raise ManifestFailure("RESOLUTION_CANONICAL")
    if resolution.get("canonical_payload_sha256") != (
        "f4502bc37da90c5387448c3f4b70dbe4bf85da5cd4483bc53423e4ec444fb1c9"
    ):
        raise ManifestFailure("RESOLUTION_ROOT")

    tsk = proofs["combined_radiation_tsk"]
    capture_preflight = proofs["combined_capture_preflight"]
    capture = proofs["combined_capture"]
    channel = proofs["channel_gate"]
    activation = proofs["activation_localization"]
    radiation = proofs["radiation_readmission"]
    drift = proofs["drift"]
    if not (
        tsk.get("passed") is True
        and tsk.get("verdict") == "WRF_PRODUCTION_STRICTLY_IMPROVES_D03_TSK"
        and tsk["tsk_parity"]["land_rms_and_max_strictly_improve"] is True
        and tsk["tsk_parity"]["water_bitwise_invariant"] is True
        and capture_preflight.get("passed") is True
        and capture_preflight.get("backend_imported") is False
        and capture.get("passed") is True
        and capture.get("authority", {}).get("adapter_invocations") == 1
        and capture.get("authority", {}).get("gpu_actions") == 0
        and capture.get("authority", {}).get("wrf_or_mpi_executions") == 0
        and channel.get("passed") is True
        and activation.get("passed") is True
        and radiation.get("passed") is True
        and drift.get("audit_complete") is True
        and drift.get("passed") is False
        and drift.get("ship_ready") is False
    ):
        raise ManifestFailure("GATE_SEMANTICS")

    activity = channel["channel_decomposed_gate"]["activation_mask_primary"]
    if not all(
        activity[field]["baseline_port_only_columns"] == 4526
        and activity[field]["final_port_only_columns"] == 67
        and activity[field]["final_wrf_only_columns"] == 0
        for field in ("s_aw", "s_awu", "s_awv")
    ):
        raise ManifestFailure("ACTIVATION_METRICS")

    junit_root = ET.parse(ARTIFACTS["focused_junit"]).getroot()
    suites = list(junit_root.findall("testsuite"))
    tests = sum(int(suite.attrib["tests"]) for suite in suites)
    failures = sum(int(suite.attrib["failures"]) for suite in suites)
    errors = sum(int(suite.attrib["errors"]) for suite in suites)
    skipped = sum(int(suite.attrib["skipped"]) for suite in suites)
    if (tests, failures, errors, skipped) != (40, 0, 0, 0):
        raise ManifestFailure(f"JUNIT:{tests}:{failures}:{errors}:{skipped}")
    return {
        "proofs": proofs,
        "resolution": resolution,
        "junit": {
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "time_seconds": sum(float(suite.attrib["time"]) for suite in suites),
        },
        "script": {
            "path": relative,
            "sha256": hashlib.sha256(disk).hexdigest(),
            "git_blob": git("rev-parse", f"HEAD:{relative}"),
        },
    }


def artifact_record(name: str, role: str, status: str) -> dict[str, Any]:
    path = ARTIFACTS[name]
    record: dict[str, Any] = {
        "path": str(path.relative_to(REPO)),
        "sha256": EXPECTED[path],
        "role": role,
        "status": status,
    }
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        record["canonical_payload_sha256"] = value.get("canonical_payload_sha256")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = validate_inputs(args.approved_head)
        proofs = inputs["proofs"]
        channel = proofs["channel_gate"]["channel_decomposed_gate"]
        tsk = proofs["combined_radiation_tsk"]["tsk_parity"]
        activation = proofs["activation_localization"]
        manifest: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-gpt-mf-seam-proof-manifest-v1",
            "verdict": "SCIENCE_ENDPOINT_SEALED_DRIFT_GATE_BLOCKED",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "passed": False,
            "science_endpoint_passed": True,
            "sprint_work_terminal": True,
            "ship_ready": False,
            "acceptance": {
                "source_fidelity": True,
                "real_wrf_component_ab": True,
                "combined_tsk_parity": True,
                "fresh_single_authority_cpu_capture": True,
                "channel_decomposed_sp2": True,
                "activation_localization_and_frozen_bound": True,
                "focused_tests": True,
                "24h_72h_drift": False,
            },
            "authority": {
                "prompt": {"path": str(PROMPT), "sha256": EXPECTED[PROMPT]},
                "contract": {
                    "path": str(ARTIFACTS["contract"].relative_to(REPO)),
                    "sha256": EXPECTED[ARTIFACTS["contract"]],
                    "commit": "3415ef5a4d9280bab5274d512eea5a55a256b5f3",
                },
                "opus_resolution": {
                    "commit": BASE,
                    "path": str(RESOLUTION.relative_to(REPO)),
                    "sha256": EXPECTED[RESOLUTION],
                    "canonical_proof_sha256": inputs["resolution"][
                        "canonical_payload_sha256"
                    ],
                },
                "approved_head": args.approved_head,
                "branch": git("branch", "--show-current"),
                "manifest_script": inputs["script"],
                "gpu_actions_this_sprint": 0,
                "wrf_or_mpi_executions_this_sprint": 0,
            },
            "implementation": {
                "base": BASE,
                "production_fix_commit": "6ea21733710f4fbafe6997d82b0cabb3244d577a",
                "fresh_capture_head": CAPTURE_HEAD,
                "production_and_test_sha256": {
                    str(path.relative_to(REPO)): digest
                    for path, digest in PRODUCTION.items()
                },
                "changed_files": git("diff", "--name-only", f"{BASE}..HEAD").splitlines(),
                "commits": git(
                    "log", "--format=%H %s", "--reverse", f"{BASE}..HEAD"
                ).splitlines(),
            },
            "headline_metrics": {
                "land_tsk_rms_k": {
                    "baseline": tsk["accepted_qml_plus_top_buffer"]["land"]["rms"],
                    "candidate": tsk["candidate_plus_wrf_cam_ghg"]["land"]["rms"],
                    "candidate_max_abs": tsk["candidate_plus_wrf_cam_ghg"]["land"]["max_abs"],
                    "water_bitwise_invariant": True,
                },
                "activation_masks": channel["activation_mask_primary"],
                "sp2_adapter_error": channel["target_adapter_error"],
                "mass_flux_shapley": channel["target_mass_flux_channel"],
                "protected_channel_shapley_rms": channel["protected_channel_shapley_rms"],
                "residual_bound": activation["frozen_67_column_bound"],
                "drift": proofs["drift"]["observed_metrics"],
            },
            "proof_objects": {
                "combined_radiation_tsk": artifact_record(
                    "combined_radiation_tsk", "fresh paired LW/NoahMP TSK composition", "load-bearing"
                ),
                "radiation_readmission": artifact_record(
                    "radiation_readmission", "hash-pinned real-WRF component A/B aggregate", "load-bearing"
                ),
                "combined_capture_preflight": artifact_record(
                    "combined_capture_preflight", "backend-dark capture preflight", "load-bearing"
                ),
                "combined_capture": artifact_record(
                    "combined_capture", "one-shot CPU production-adapter capture", "load-bearing"
                ),
                "channel_raw": artifact_record(
                    "channel_raw", "frozen five-channel SP2 partition", "load-bearing"
                ),
                "channel_gate": artifact_record(
                    "channel_gate", "ratified channel acceptance gate", "load-bearing"
                ),
                "activation_localization": artifact_record(
                    "activation_localization", "direct-kernel predicate audit and 67-column bound", "load-bearing"
                ),
                "focused_junit": artifact_record(
                    "focused_junit", "source and regression tests", "load-bearing"
                ),
                "drift": artifact_record(
                    "drift", "fail-closed 24/72 h trajectory availability", "blocking"
                ),
                "manager_question": artifact_record(
                    "manager_question", "unresolved authority request", "blocking-context"
                ),
            },
            "external_payloads": {
                "combined_radiation_delta": {
                    "path": str(RADIATION_DELTA),
                    "sha256": EXPECTED[RADIATION_DELTA],
                },
                "capture_manifest": {
                    "path": str(CAPTURE_MANIFEST),
                    "sha256": EXPECTED[CAPTURE_MANIFEST],
                },
                "capture_archive": {
                    "path": str(CAPTURE_ARCHIVE),
                    "sha256": EXPECTED[CAPTURE_ARCHIVE],
                    "bytes": CAPTURE_ARCHIVE.stat().st_size,
                },
            },
            "superseded_exploration": {
                "status": "retained for audit, not load-bearing",
                "preflight": artifact_record(
                    "exploratory_capture_preflight", "endpoint-v1 preflight", "superseded"
                ),
                "capture": artifact_record(
                    "exploratory_capture", "endpoint-v1 O3-only capture", "superseded"
                ),
            },
            "focused_tests": inputs["junit"],
            "unresolved_risks": [
                "The 67 captured port-only MYNN columns are a non-roundoff observational bound; exact WRF DMP plume-state operands are not retained.",
                "CLWRF gas increment alone regresses LW GLW over the dynamic-interface arm; only the manager-ratified full composition is admitted.",
                "No exact 24/72 h candidate/control trajectories exist, so T2 and 10 m-wind drift metrics are null and this endpoint is not ship-ready.",
            ],
            "next_decision": (
                "Authorize or supply the exact 24/72 h candidate/control trajectory pair; "
                "the resulting measured drift proof must supersede the blocking artifact."
            ),
        }
        atomic_json(args.output.resolve(), manifest)
        print(json.dumps({
            "output": str(args.output.resolve()),
            "verdict": manifest["verdict"],
            "science_endpoint_passed": True,
            "ship_ready": False,
            "canonical_payload_sha256": canonical(manifest),
        }, sort_keys=True))
        return 0
    except (ManifestFailure, OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL:{error}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
