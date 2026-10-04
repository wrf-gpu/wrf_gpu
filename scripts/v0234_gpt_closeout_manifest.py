#!/usr/bin/env python3
"""Seal the v0234 GPT single-authority proof graph.

This backend-dark closeout validates the terminal objects and hashes the
immutable roots that close their nested evidence graphs.  It never imports
gpuwrf/JAX, invokes an adapter, or executes WRF/MPI.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
PROOFS = {
    "phase_a_adversarial_audit": SPRINT / "phase-a-adversarial-audit.json",
    "capture_preflight": SPRINT / "single-authority-capture-preflight.json",
    "single_authority_capture": SPRINT / "single-authority-capture-proof.json",
    "qke_lifecycle_fix": SPRINT / "qke-lifecycle-fix-proof.json",
    "operand_term_attribution": SPRINT / "operand-term-attribution.json",
}
JUNIT = SPRINT / "qke-lifecycle-pytest.xml"
CAPTURE_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_gpt_single_authority_attribution/"
    "cpu-production-adapter-authority-v1"
)
INPUT_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
    "capture/authentic-ac6712170cbe5084"
)
SEALED_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2"
)
WRF_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/"
    "evidence-dumps-fresh-d03-runtime"
)
IMMUTABLE_FILES = {
    "capture_archive": (
        CAPTURE_ROOT / "single-authority-capture.npz",
        "bc576333ba54d26db7272c8692b327b56f477e6183a6c02d596bf0e6685cd2fe",
    ),
    "capture_manifest": (
        CAPTURE_ROOT / "manifest.json",
        "7835e8fecce89fb84d1dafaf316cb8ea0f448acb241bae789095886406b23e7a",
    ),
    "input_state_manifest": (
        INPUT_ROOT / "manifest.json",
        "295760b17cba2fde8e931188caab76fbf0564b7e9229784e8f739270f75309e6",
    ),
    "namelist_receipt": (
        SEALED_ROOT / "control/capture-input-manifest.json",
        "be3ee7df849fed756a9b6c0432b8c7ace7ccf0c411fb8630f4252bcecf7c2f40",
    ),
    "sealed_namelist": (
        SEALED_ROOT / "capture-inputs/namelist.input",
        "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    ),
    "wrf_mynn_source": (
        SEALED_ROOT / "source/instrumented/phys/MYNN-EDMF/module_bl_mynnedmf.F90",
        "294fd2f5afa650c50ddf8f2cc1bdc444d5d579c3ac42e38f7a56d8027a3775d3",
    ),
    "wrf_mynn_driver_source": (
        SEALED_ROOT
        / "source/instrumented/phys/MYNN-EDMF/WRF/module_bl_mynnedmf_driver.F90",
        "feb04ed5b24c1a469fc9c44d8138a2f050b5ca18347b328e4295020fd6abc3fe",
    ),
}
EXPECTED_WRF_TREE = "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class CloseoutFailure(RuntimeError):
    """Fail-closed closeout failure."""


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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CloseoutFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def regular_hash(path: Path) -> str:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise CloseoutFailure(f"NOT_REGULAR:{path}")
    return sha256_file(path)


def resource_gate() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    observed_threads = {key: os.environ.get(key) for key in THREAD_ENV}
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    if affinity != ALLOWED_CPUS:
        raise CloseoutFailure(f"CPUSET:{sorted(affinity)}")
    if observed_threads != THREAD_ENV:
        raise CloseoutFailure(f"THREAD_ENV:{observed_threads}")
    if nice < 15 or ioprio < 0 or int(ioprio) >> 13 != 3:
        raise CloseoutFailure(f"PRIORITY:{nice}:{ioprio}")
    if Path("/tmp/PREEMPT_CPU").exists() or Path("/tmp/PREEMPT_CPU").is_symlink():
        raise CloseoutFailure("PREEMPT_CPU")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": observed_threads,
        "nice": nice,
        "ionice_class": int(ioprio) >> 13,
        "preempt_cpu_absent": True,
    }


def read_proof(name: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    digest = regular_hash(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    canonical = canonical_without_self(value)
    if canonical != value.get("canonical_payload_sha256"):
        raise CloseoutFailure(f"CANONICAL:{name}")
    return value, {
        "path": str(path),
        "sha256": digest,
        "size": path.stat().st_size,
        "schema": value.get("schema"),
        "canonical_payload_sha256": canonical,
    }


def validate_junit() -> dict[str, Any]:
    digest = regular_hash(JUNIT)
    root = ET.parse(JUNIT).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals != {"tests": 14, "failures": 0, "errors": 0, "skipped": 0}:
        raise CloseoutFailure(f"JUNIT:{totals}")
    return {"path": str(JUNIT), "sha256": digest, "size": JUNIT.stat().st_size, **totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        print(f"REFUSE:OUTPUT_NOT_FRESH:{output}", file=sys.stderr)
        return 74
    try:
        status = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"], check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
        if status:
            raise CloseoutFailure(f"WORKTREE:{status}")
        head = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

        values: dict[str, dict[str, Any]] = {}
        records: dict[str, dict[str, Any]] = {}
        for name, path in PROOFS.items():
            values[name], records[name] = read_proof(name, path)
        phase = values["phase_a_adversarial_audit"]
        preflight = values["capture_preflight"]
        capture = values["single_authority_capture"]
        fix = values["qke_lifecycle_fix"]
        attribution = values["operand_term_attribution"]
        if not (
            phase.get("terminal", {}).get("phase_b_authorized") is True
            and preflight.get("passed") is True
            and capture.get("passed") is True
            and capture.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
            and fix.get("passed") is True
            and fix.get("terminal_eligible") is True
            and fix.get("status") == "SOURCE_AUTHORIZED_FRESH_QKE_INIT_FIX_PROVEN"
            and attribution.get("passes") is True
            and attribution.get("terminal_verdict")
            == "SINGLE_AUTHORITY_SOURCE_LOCALIZED_FIX_PROVEN"
            and attribution.get("fix_authorization", {}).get("authorized") is True
        ):
            raise CloseoutFailure("TERMINAL_GRAPH_STATUS")

        immutable: dict[str, Any] = {}
        for name, (path, expected) in IMMUTABLE_FILES.items():
            actual = regular_hash(path)
            if actual != expected:
                raise CloseoutFailure(f"IMMUTABLE_HASH:{name}:{actual}")
            immutable[name] = {
                "path": str(path), "sha256": actual, "size": path.stat().st_size,
            }
        tree = {}
        for path in sorted(WRF_ROOT.rglob("*")):
            if path.is_symlink():
                raise CloseoutFailure(f"WRF_TREE_SYMLINK:{path}")
            if path.is_file():
                tree[path.relative_to(WRF_ROOT).as_posix()] = sha256_file(path)
        tree_sha = hashlib.sha256(json.dumps(
            tree, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if len(tree) != 462 or tree_sha != EXPECTED_WRF_TREE:
            raise CloseoutFailure(f"WRF_TREE:{len(tree)}:{tree_sha}")

        proof = {
            "schema": "wrfgpu2-v0234-gpt-single-authority-proof-manifest-v1",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "terminal_verdict": "SINGLE_AUTHORITY_SOURCE_LOCALIZED_FIX_PROVEN",
            "passed": True,
            "git_head": head,
            "generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": regular_hash(Path(__file__).resolve()),
            },
            "resource_gate": resource_gate(),
            "actions": {
                "single_authority_capture_adapter_invocations": 1,
                "additional_production_adapter_invocations_after_capture": 0,
                "gpu_actions": 0,
                "wrf_or_mpi_executions": 0,
            },
            "proof_objects": records,
            "focused_test_output": validate_junit(),
            "immutable_roots": immutable,
            "wrf_dump_tree": {
                "path": str(WRF_ROOT), "file_count": len(tree), "sha256": tree_sha,
            },
            "coverage": (
                "Every required local proof object is whole-file and canonical-"
                "payload hashed. Immutable capture/receipt/source roots are whole-"
                "file hashed; their nested manifests bind array/state inventories. "
                "The pristine WRF dump is closed by a deterministic 462-file tree hash."
            ),
            "limitations": {
                "qke_lifecycle_fix_proven": True,
                "downstream_full_sp2_closure_claimed": False,
                "reason": (
                    "A second production adapter run is forbidden; remaining captured "
                    "mass-flux and mixing divergences require a new authority."
                ),
            },
        }
        atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "output": str(output),
            "terminal_verdict": proof["terminal_verdict"],
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True))
        return 0
    except (
        CloseoutFailure, OSError, ValueError, TypeError, KeyError,
        json.JSONDecodeError, subprocess.CalledProcessError, ET.ParseError,
    ) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
