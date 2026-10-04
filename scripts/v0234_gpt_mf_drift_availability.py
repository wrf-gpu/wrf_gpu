#!/usr/bin/env python3
"""Fail-closed availability audit for the v0234 24/72 h drift gate.

The sprint requires T2 and 10 m-wind RMSE at 24 h and 72 h, while its same
authority forbids both GPU and WRF/MPI execution.  This audit does not turn
missing trajectories into a pass.  It records the constraint conflict,
searches retained manifests for this endpoint's provenance, and emits a
canonical NOT-SHIP-READY proof when no authorized candidate/control pair is
available.
"""

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


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-mf-seam-closure"
PROMPT = Path("/tmp/v0234_gpt_mf_seam_prompt.txt")
CONTRACT = SPRINT / "CONTRACT.md"
QUESTION = SPRINT / "GPT_MANAGER_QUESTION.md"
CAPTURE_MANIFEST = Path(
    "/tmp/v0234_gpt_mf_seam_capture/endpoint-v2/manifest.json"
)
EXPECTED = {
    PROMPT: "4a3c171681026f3250314abd32c907603c388d4c1ed72a94d61b8cce2b843d43",
    CONTRACT: "39e41311e5d96d8c459148e3f1fa03444d32c9f60af33f887bad627618c683e2",
    CAPTURE_MANIFEST: "db2b3949af18cc0ebf0c762235509fd3548e218375a31c3cb35a875fb8365b8a",
}
ENDPOINT_COMMITS = (
    "6ea21733",
    "2694609c",
    "v0234-massflux-seam-closure",
)
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class DriftAuditFailure(RuntimeError):
    """Fail-closed authority or inventory error."""


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
        raise DriftAuditFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise DriftAuditFailure(f"TEMP_NOT_FRESH:{temporary}")
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
        raise DriftAuditFailure(f"GIT:{' '.join(args)}:{stderr.strip()}")
    return result.stdout if binary else result.stdout.strip()


def require_token(text: str, token: str) -> None:
    if token not in text:
        raise DriftAuditFailure(f"AUTHORITY_TOKEN:{token}")


def resource_and_authority_gate(approved_head: str) -> dict[str, Any]:
    if Path("/tmp/PREEMPT_CPU").exists():
        raise DriftAuditFailure("PREEMPT_CPU")
    if git("rev-parse", "HEAD") != approved_head:
        raise DriftAuditFailure("APPROVED_HEAD_DRIFT")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise DriftAuditFailure("TRACKED_WORKTREE_NOT_CLEAN")
    if set(os.sched_getaffinity(0)) != ALLOWED_CPUS:
        raise DriftAuditFailure(f"CPUSET:{sorted(os.sched_getaffinity(0))}")
    environment = {key: os.environ.get(key) for key in THREAD_ENV}
    if environment != THREAD_ENV:
        raise DriftAuditFailure(f"THREAD_ENV:{environment}")
    for path, expected in EXPECTED.items():
        observed = sha256_file(path)
        if observed != expected:
            raise DriftAuditFailure(f"HASH:{path}:{observed}")
    relative = Path(__file__).resolve().relative_to(REPO).as_posix()
    disk = Path(__file__).resolve().read_bytes()
    if disk != git("show", f"HEAD:{relative}", binary=True):
        raise DriftAuditFailure("SCRIPT_NOT_HEAD")

    prompt = PROMPT.read_text(encoding="utf-8")
    contract = CONTRACT.read_text(encoding="utf-8")
    require_token(prompt, "drift-check requirement")
    require_token(contract, "24 h / 72 h drift comparison")
    require_token(contract, "T2")
    require_token(contract, "10 m")
    require_token(prompt, "NO GPU")
    require_token(prompt, "No WRF/MPI execution")
    require_token(contract, "No GPU action or query")
    require_token(contract, "No WRF or MPI execution")
    return {
        "approved_head": approved_head,
        "branch": git("branch", "--show-current"),
        "cpuset": sorted(ALLOWED_CPUS),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "thread_environment": environment,
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
        "script": {
            "path": relative,
            "sha256": hashlib.sha256(disk).hexdigest(),
            "git_blob": git("rev-parse", f"HEAD:{relative}"),
        },
        "authority_hashes": {str(path): digest for path, digest in EXPECTED.items()},
        "manager_question": {
            "path": str(QUESTION.relative_to(REPO)),
            "sha256": sha256_file(QUESTION),
            "response_recorded": "Manager answer" in QUESTION.read_text(encoding="utf-8"),
        },
    }


def retained_manifest_search(root: Path) -> list[str]:
    pattern = "|".join(ENDPOINT_COMMITS)
    result = subprocess.run(
        [
            "rg", "-l", pattern, str(root),
            "-g", "*.json", "-g", "*.md", "-g", "*.txt", "-g", "*.log",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        raise DriftAuditFailure(f"RG:{root}:{result.stderr.strip()}")
    return sorted(line for line in result.stdout.splitlines() if line)


def inventory() -> dict[str, Any]:
    mnt_matches = retained_manifest_search(Path("<DATA_ROOT>/wrf_gpu2"))
    # Avoid unrelated systemd-private roots under /tmp, which are intentionally
    # unreadable.  Every project scratch namespace follows the v0234* prefix.
    tmp_roots = sorted(
        path for path in Path("/tmp").glob("v0234*")
        if not path.is_symlink() and (path.is_file() or path.is_dir())
    )
    tmp_matches = sorted({
        match
        for root in tmp_roots
        for match in retained_manifest_search(root)
    })
    expected_tmp = {
        "/tmp/v0234_gpt_mf_seam_capture/endpoint-v1/manifest.json",
        "/tmp/v0234_gpt_mf_seam_capture/endpoint-v2/manifest.json",
        "/tmp/v0234_gpt_mf_seam_capture/endpoint-v2/run-start.json",
        "/tmp/v0234_gpt_mf_seam_channel_raw.json",
        "/tmp/v0234_gpt_mf_seam_prompt.txt",
    }
    if mnt_matches or set(tmp_matches) != expected_tmp:
        raise DriftAuditFailure(
            f"ENDPOINT_PROVENANCE_INVENTORY:{mnt_matches}:{tmp_matches}"
        )
    capture_wrfouts = sorted(
        str(path)
        for path in Path("/tmp/v0234_gpt_mf_seam_capture").rglob("wrfout*")
        if path.is_file()
    )
    if capture_wrfouts:
        raise DriftAuditFailure(f"UNEXPECTED_CAPTURE_WRFOUT:{capture_wrfouts}")
    manifest = json.loads(CAPTURE_MANIFEST.read_text(encoding="utf-8"))
    if not (
        manifest.get("passed") is True
        and manifest.get("authority", {}).get("adapter_invocations") == 1
        and manifest.get("authority", {}).get("wrf_or_mpi_executions") == 0
        and manifest.get("authority", {}).get("gpu_actions") == 0
    ):
        raise DriftAuditFailure("CAPTURE_MANIFEST_SEMANTICS")
    return {
        "search_patterns": list(ENDPOINT_COMMITS),
        "search_roots": ["<DATA_ROOT>/wrf_gpu2", "/tmp/v0234*"],
        "tmp_project_roots_searched": len(tmp_roots),
        "retained_mnt_manifest_matches": mnt_matches,
        "retained_tmp_matches": tmp_matches,
        "tmp_matches_are_step1_capture_or_comparator_only": True,
        "wrfout_files_in_endpoint_capture_namespaces": capture_wrfouts,
        "candidate_control_pair_available": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        authority = resource_and_authority_gate(args.approved_head)
        retained = inventory()
        proof = {
            "schema": "wrfgpu2-v0234-gpt-radiation-drift-availability-v1",
            "verdict": "DRIFT_GATE_BLOCKED_NO_AUTHORIZED_TRAJECTORY_PAIR",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "audit_complete": True,
            "passed": False,
            "ship_ready": False,
            "execution": authority,
            "required_gate": {
                "leads_hours": [24, 72],
                "metrics": ["T2_RMSE_K", "10m_wind_speed_RMSE_m_s-1"],
                "candidate": "exact re-admitted radiation + MYNN-veto endpoint",
                "control": "manager-authorized accepted pre-readmission endpoint",
            },
            "observed_metrics": {
                "24h": {"T2_RMSE_K": None, "10m_wind_speed_RMSE_m_s-1": None},
                "72h": {"T2_RMSE_K": None, "10m_wind_speed_RMSE_m_s-1": None},
            },
            "retained_evidence_inventory": retained,
            "blocking_conflict": {
                "required": "Generate or receive exact 24 h and 72 h candidate/control trajectories.",
                "forbidden": "This sprint may perform neither a GPU action/query nor WRF/MPI execution.",
                "cpu_only_alternative": (
                    "No sealed 24/72 h candidate trajectory exists, and a new full CPU JAX "
                    "forecast pair was neither pre-authorized nor feasible within the bounded "
                    "CPU release. A step-1 capture cannot substitute for long-horizon drift."
                ),
            },
            "non_claims": [
                "No drift metric was fabricated or inferred from step-1 TSK parity.",
                "The green component and channel gates do not waive this drift gate.",
                "Production re-admission is not ship-ready until this proof is superseded by measured trajectories.",
            ],
            "next_authorized_action": (
                "Manager supplies an exact sealed candidate/control trajectory pair, or "
                "opens a separately authorized forecast run with the required GPU/WRF authority."
            ),
        }
        atomic_json(output, proof)
        print(json.dumps({
            "output": str(output),
            "verdict": proof["verdict"],
            "ship_ready": False,
            "canonical_payload_sha256": canonical(proof),
        }, sort_keys=True))
        return 0
    except (DriftAuditFailure, OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL:{error}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
