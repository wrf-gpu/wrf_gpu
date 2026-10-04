#!/usr/bin/env python3
"""Build the canonical proof manifest for the v0234 land-TSK sprint."""

from __future__ import annotations

import argparse
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
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
BASELINE = "e653bdbf"
TERMINAL_PROOF = SPRINT / "terminal-audit-proof.json"
TERMINAL_PROOF_SHA256 = (
    "1e93ea283c07487e5d4f6b84a8ef8a5433f615a2635ed5f6564855de8d77827e"
)
TERMINAL_CANONICAL = (
    "98ddfe1117309b5dd97df9db1b9859070f2ad7719780a493e457c2d7acff4424"
)
JUNIT = SPRINT / "focused-tests.xml"
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class ManifestFailure(RuntimeError):
    """The closeout evidence graph is incomplete or has drifted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload(record: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in record.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        text=not binary,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        error = result.stderr.decode() if binary else result.stderr
        raise ManifestFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def tracked_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(REPO).as_posix()
    data = path.read_bytes()
    if data != git("show", f"HEAD:{relative}", binary=True):
        raise ManifestFailure(f"NOT_HEAD:{relative}")
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "git_blob": git("rev-parse", f"HEAD:{relative}"),
    }


def resource_gate() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    threads = {name: os.environ.get(name) for name in THREAD_ENV}
    if affinity != ALLOWED_CPUS:
        raise ManifestFailure(f"CPUSET:{sorted(affinity)}")
    if threads != THREAD_ENV:
        raise ManifestFailure(f"THREAD_ENV:{threads}")
    if Path("/tmp/PREEMPT_CPU").exists():
        raise ManifestFailure("PREEMPT_CPU")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": threads,
        "preempt_cpu_absent": True,
    }


def validate_terminal() -> dict[str, Any]:
    if sha256_file(TERMINAL_PROOF) != TERMINAL_PROOF_SHA256:
        raise ManifestFailure("TERMINAL_PROOF_HASH")
    proof = json.loads(TERMINAL_PROOF.read_text(encoding="utf-8"))
    if not (
        canonical_payload(proof) == TERMINAL_CANONICAL
        and proof.get("canonical_payload_sha256") == TERMINAL_CANONICAL
        and proof.get("passed") is True
        and proof.get("verdict")
        == "SOURCE_FAITHFUL_RADIATION_TSK_BOUND_SP2_GATE_CONFLICT_LOCALIZED"
        and proof["terminal_conditions"][
            "land_tsk_closed_or_floor_justified"
        ]
        is False
        and proof["terminal_conditions"][
            "final_sp2_within_e653bdbf_relative_noise_bound_1e_4"
        ]
        is True
        and proof["terminal_conditions"][
            "gate_split_addressed_or_falsifiably_localized"
        ]
        is True
    ):
        raise ManifestFailure("TERMINAL_PROOF_GATE")
    return proof


def validate_junit() -> dict[str, Any]:
    root = ET.parse(JUNIT).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if totals != {"tests": 28, "failures": 0, "errors": 0, "skipped": 0}:
        raise ManifestFailure(f"JUNIT:{totals}")
    return {
        "path": JUNIT.relative_to(REPO).as_posix(),
        "sha256": sha256_file(JUNIT),
        "size": JUNIT.stat().st_size,
        **totals,
    }


def sprint_artifacts(output: Path) -> dict[str, Any]:
    excluded = {output.name, "WORKER_REPORT.md"}
    records: dict[str, Any] = {}
    for path in sorted(SPRINT.iterdir()):
        if path.name in excluded:
            continue
        if path.is_symlink() or not path.is_file():
            raise ManifestFailure(f"SPRINT_ARTIFACT_TYPE:{path}")
        record: dict[str, Any] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            record["schema"] = value.get("schema")
            declared = value.get("canonical_payload_sha256")
            if declared is not None:
                observed = canonical_payload(value)
                if observed != declared:
                    raise ManifestFailure(f"SPRINT_CANONICAL:{path.name}")
                record["canonical_payload_sha256"] = observed
        records[path.name] = record
    return records


def tracked_changes() -> dict[str, Any]:
    names = git("diff", "--name-only", f"{BASELINE}..HEAD").splitlines()
    records: dict[str, Any] = {}
    for relative in names:
        if not relative or relative.startswith(
            ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance/"
        ):
            continue
        path = REPO / relative
        if path.is_symlink() or not path.is_file():
            raise ManifestFailure(f"TRACKED_CHANGE_TYPE:{relative}")
        records[relative] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "git_blob": git("rev-parse", f"HEAD:{relative}"),
        }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if output.exists() or output.is_symlink():
            raise ManifestFailure(f"OUTPUT_NOT_FRESH:{output}")
        status = git("status", "--porcelain")
        if status:
            raise ManifestFailure(f"WORKTREE_NOT_CLEAN:{status}")
        terminal = validate_terminal()
        artifacts = sprint_artifacts(output)
        junit = validate_junit()
        if artifacts.get("focused-tests.xml", {}).get("sha256") != junit["sha256"]:
            raise ManifestFailure("JUNIT_ARTIFACT_DISAGREEMENT")
        manifest = {
            "schema": "wrfgpu2-v0234-gpt-land-tsk-proof-manifest-v1",
            "verdict": terminal["verdict"],
            "passed": True,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "branch": git("branch", "--show-current"),
            "head_at_manifest": git("rev-parse", "HEAD"),
            "tree_at_manifest": git("rev-parse", "HEAD^{tree}"),
            "baseline": BASELINE,
            "generator": tracked_record(Path(__file__).resolve()),
            "resource_gate": resource_gate(),
            "terminal_conditions": terminal["terminal_conditions"],
            "final_endpoint": terminal["final_production_endpoint"],
            "gate_split_localization": terminal["gate_split_localization"],
            "focused_tests": junit,
            "sprint_artifacts": artifacts,
            "tracked_changes_outside_sprint": tracked_changes(),
            "actions": {
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
                "final_fresh_capture_adapter_invocations": 1,
                "terminal_audit_adapter_invocations": 0,
            },
            "limitations": {
                "land_tsk_closed": False,
                "representation_floor_claimed": False,
                "manager_amended_all_conditions_met": False,
                "strongest_falsifiable_localization_claimed": True,
                "terminal_opus_critic_pending": True,
            },
        }
        manifest["canonical_payload_sha256"] = canonical_payload(manifest)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "passed": True,
                    "output": str(output),
                    "artifact_count": len(artifacts),
                    "tracked_change_count": len(
                        manifest["tracked_changes_outside_sprint"]
                    ),
                    "canonical_payload_sha256": manifest[
                        "canonical_payload_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - closeout must fail closed
        print(f"FAIL_CLOSED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
