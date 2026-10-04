#!/usr/bin/env python3
"""Fail-closed terminal manifest for the authorized post-fix SP2 validation.

This backend-dark program seals the failed first arm, the corrected single arm,
the same-authority CPU/reference chain, and the frozen exact-comparator result.
It performs no GPU query, model invocation, WRF execution, or MPI execution.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
REPORT = SPRINT / "WORKER_REPORT.md"
JUNIT = SPRINT / "postfix-harness-pytest.xml"

ACCEPTED_COMPONENT_HEAD = "672f55c0e6ac5ede82c288cdcaed63fd71a5e9a6"
FAILED_NONCE = "2e1ad51a081af14542d1326672c4306e5fbdd849259e4bd01fcb9ae753532d8e"
NONCE = "a73aef522fbdca9c88ec1c50464db6e51819f47aa00cc9493a1fe2616cd2ebfe"
PREFIX = NONCE[:16]
CONSUMED_PREFIXES = (
    "0b18530a1dc9cac2",
    "ac6712170cbe5084",
    FAILED_NONCE[:16],
)
AUTHORITY_ROOT = Path(f"<DATA_ROOT>/wrf_gpu2/v0234_gpt_postfix_sp2_{PREFIX}")
CAPTURE_ROOT = AUTHORITY_ROOT / f"capture/authentic-{PREFIX}"
CPU_ROOT = AUTHORITY_ROOT / "cpu-adapter/single-authority-v1"
REFERENCE_ROOT = AUTHORITY_ROOT / "cpu-reference"
COMPARATOR_ROOT = AUTHORITY_ROOT / "comparator"
WRF_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/"
    "evidence-dumps-fresh-d03-runtime"
)

PRE_FIX_RMS = {
    "rublten": 0.0002538443572720412,
    "rvblten": 0.00016647474452487508,
}
POST_FIX_RMS = {
    "rublten": 0.00025202622402235,
    "rvblten": 0.00016144535534717142,
}
EXPECTED_CAPTURE_TREE_SHA256 = (
    "4476c4405f653bea5fc8cc36281c76d4290980dfb6bc7b8175c17941da384673"
)
EXPECTED_WRF_TREE_SHA256 = (
    "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
)

LOCAL_PROOFS = {
    "initial_authority_reservation": (
        SPRINT / "postfix-authority-reservation.json",
        "a16a737e1362eb1f4643eace045193afc25819ec490a17cc472ec35e227b0e05",
    ),
    "initial_gpu_preflight": (
        SPRINT / "postfix-gpu-capture-preflight.json",
        "6c9eff30cf0cd8d240cd542211a024144aaa16e8d4aab8349dcb91b941aa8d6f",
    ),
    "initial_comparator_preflight": (
        SPRINT / "postfix-comparator-preflight.json",
        "3dbde6819fd9473900628aa251847a37a4a491ccaae470c428d33b8be470c764",
    ),
    "initial_failed_gpu_receipt": (
        SPRINT / "postfix-gpu-execution-receipt.json",
        "a2c47cfbee904045ad65ec68d1fcdeccbc05e657715243f516054172b35e6f1c",
    ),
    "corrected_authority_reservation": (
        SPRINT / "postfix-corrected-authority-reservation.json",
        "804e358e3b7f1bce25846b2a72f71c9aee43a1f269af4857086e7bbfe3910312",
    ),
    "corrected_gpu_preflight": (
        SPRINT / "postfix-corrected-gpu-capture-preflight.json",
        "d2b0bd46bba291b817f1fdb05d07a71c4340254749399d721bdf136374b52d51",
    ),
    "corrected_comparator_preflight": (
        SPRINT / "postfix-corrected-comparator-preflight.json",
        "e10b3e62e37d9621df68423e26eb1c646181fb54b6009d8c1a4986b4c05fb715",
    ),
    "corrected_gpu_receipt": (
        SPRINT / "postfix-corrected-gpu-execution-receipt.json",
        "e5e601cbecbe3366d100596227bdfd011d5ecabb43db1222044d2eb3bfb1ea0f",
    ),
    "corrected_gpu_validation": (
        SPRINT / "postfix-corrected-gpu-capture-validation.json",
        "a61369e1474da6284e6608553394c0ff76a5fb7e72b6d83331ff27cb941748bb",
    ),
    "corrected_cpu_preflight": (
        SPRINT / "postfix-corrected-cpu-adapter-preflight.json",
        "4242822d4d35522ccc10f0d366bad2c505bc4bba51c5201625ef778d4d5bef99",
    ),
    "corrected_cpu_capture": (
        SPRINT / "postfix-corrected-cpu-adapter-proof.json",
        "cf958e8b291073cb11a8bae0d8042fc9e877b8bb3ffe6c7327deeb805f0832a0",
    ),
    "corrected_reference_materialization": (
        SPRINT / "postfix-corrected-reference-materialization.json",
        "25bfbde0aff126472b009958b747ad65fd70d28b10206e9817f107fc89ea7891",
    ),
    "corrected_reference_validation": (
        SPRINT / "postfix-corrected-reference-validation.json",
        "af0cba7a318b0bc18b3cfb38863027be28e1a80fbe9e5952fdd8c757b3e4864f",
    ),
    "corrected_exact_comparison": (
        SPRINT / "postfix-corrected-exact-comparison-result.json",
        "e9530aefc507bde9da7f9dee6f343f718b6fe9d8a578a6a39b64b6c622d09321",
    ),
    "accepted_component_manifest": (
        SPRINT / "proof-manifest.json",
        "a2ecb613a63560a6c53f73ce628ab77f66bbd5b81edd96a3739f9ebae9e6661d",
    ),
}

EXTERNAL_FILES = {
    "gpu_capture_manifest": (
        CAPTURE_ROOT / "manifest.json",
        "ae584d25d0e8fc56afedc4a1745f9109d52fecf76635007324180f40bb709873",
    ),
    "cpu_capture_manifest": (
        CPU_ROOT / "manifest.json",
        "4722c6faf5df0bf09675ff3d9d51d61f6d81b2793c7c4ee9ea496f474fe9afe9",
    ),
    "cpu_capture_archive": (
        CPU_ROOT / "single-authority-capture.npz",
        "0f778d51658148fa2d482493fd8fae8a15c5862c2052e6d5942e839e16a850e0",
    ),
    "postfix_reference_archive": (
        REFERENCE_ROOT / "postfix-sp2-reference-28.npz",
        "c52938c21ac3bf348232978a01ea28d9ef9a8ced784dadc715857f044bf8d920",
    ),
    "postfix_reference_manifest": (
        REFERENCE_ROOT / "postfix-sp2-reference-manifest.json",
        "5a64cc7a0bbd24840cc4c87384608591b440fe7dc09aafe2ae9df7f935d10ee0",
    ),
    "postfix_reference_validation": (
        REFERENCE_ROOT / "postfix-sp2-reference-validation.json",
        "af0cba7a318b0bc18b3cfb38863027be28e1a80fbe9e5952fdd8c757b3e4864f",
    ),
    "comparator_run_adapter": (
        COMPARATOR_ROOT / "run-proof-adapter.json",
        "b7b4bbd7bdbe047223115604585fe0a6da0d8f8543ba8748e720a5c1c77f91fa",
    ),
    "comparator_reference_adapter": (
        COMPARATOR_ROOT / "reference-adapter.json",
        "37deb879a98cc6e0891b4b4f04c55293d8bd8d277510ae4fc1b8a9be122aca58",
    ),
    "scientific_comparison": (
        COMPARATOR_ROOT / "scientific-comparison.json",
        "a7edae693dfe3758e393165237f8a9b63048d9e26e5249329dd5ba60e69f3839",
    ),
    "comparison_terminal": (
        COMPARATOR_ROOT / "comparison-terminal.json",
        "bba37c4ed25c2621152a8b665f10952aff61a9114c286c82beff33624f3aad32",
    ),
    "authentic_pre_fix_archive": (
        Path(
            "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
            "cpu-reference/pbl-sp2-reference-28.npz"
        ),
        "a6416b7245d26f23f0df398dd6a3a926a1749cba2069dc3d0ea39c98bc3d2566",
    ),
    "authentic_pre_fix_manifest": (
        Path(
            "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
            "cpu-reference/pbl-sp2-reference-manifest.json"
        ),
        "7063326d9df10d2e78a7804d6d2a9d1bae666bbd7be4a62e51bf656670f81933",
    ),
    "authentic_pre_fix_validation": (
        Path(
            "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
            "cpu-reference/pbl-sp2-reference-validation.json"
        ),
        "acebb09e607df32e0e4d81d83930068ed5e9b7c0c1ddcc599154703fea03fbe3",
    ),
}

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
    """Fail-closed terminal-proof failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    return canonical_sha256({
        key: item for key, item in value.items()
        if key != "canonical_payload_sha256"
    })


def require_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise CloseoutFailure(f"NOT_REGULAR:{path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise CloseoutFailure(f"HASH:{path}:{actual}")
    return {"path": str(path), "sha256": actual, "size": path.stat().st_size}


def load_canonical(
    path: Path, expected: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = require_file(path, expected)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CloseoutFailure(f"JSON_OBJECT:{path}")
    canonical = canonical_without_self(value)
    if value.get("canonical_payload_sha256") != canonical:
        raise CloseoutFailure(f"CANONICAL:{path}")
    record["canonical_payload_sha256"] = canonical
    record["schema"] = value.get("schema")
    return value, record


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


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise CloseoutFailure(f"GIT:{' '.join(args)}:{result.stderr}")
    return result.stdout.strip()


def resource_gate() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    observed_threads = {key: os.environ.get(key) for key in THREAD_ENV}
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    if affinity != ALLOWED_CPUS or observed_threads != THREAD_ENV:
        raise CloseoutFailure(f"CPU_RESOURCE:{sorted(affinity)}:{observed_threads}")
    if nice < 15 or ioprio < 0 or int(ioprio) >> 13 != 3:
        raise CloseoutFailure(f"CPU_PRIORITY:{nice}:{ioprio}")
    if Path("/tmp/PREEMPT_CPU").exists() or Path("/tmp/PREEMPT_CPU").is_symlink():
        raise CloseoutFailure("PREEMPT_CPU")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": observed_threads,
        "nice": nice,
        "ionice_class": int(ioprio) >> 13,
        "preempt_cpu_absent": True,
    }


def tree_inventory(root: Path) -> dict[str, Any]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise CloseoutFailure(f"TREE_ROOT:{root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CloseoutFailure(f"TREE_SYMLINK:{path}")
        if path.is_file():
            records.append({
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {
        "file_count": len(records),
        "files": records,
        "tree_sha256": canonical_sha256(records),
    }


def wrf_tree_inventory(root: Path) -> dict[str, Any]:
    """Reproduce the sealed WRF dump's historical path-to-hash encoding."""

    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise CloseoutFailure(f"WRF_TREE_ROOT:{root}")
    records: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CloseoutFailure(f"WRF_TREE_SYMLINK:{path}")
        if path.is_file():
            records[path.relative_to(root).as_posix()] = sha256_file(path)
    return {
        "file_count": len(records),
        "tree_sha256": canonical_sha256(records),
    }


def validate_junit() -> dict[str, Any]:
    record = require_file(JUNIT)
    root = ET.parse(JUNIT).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals != {"tests": 20, "failures": 0, "errors": 0, "skipped": 0}:
        raise CloseoutFailure(f"JUNIT:{totals}")
    return {**record, **totals}


def close_enough_exact(observed: Any, expected: float) -> bool:
    try:
        return math.isfinite(float(observed)) and float(observed) == expected
    except (TypeError, ValueError):
        return False


def validate_graph(values: Mapping[str, Mapping[str, Any]]) -> None:
    failed = values["initial_failed_gpu_receipt"]
    corrected = values["corrected_gpu_receipt"]
    reservation = values["corrected_authority_reservation"]
    gpu = values["corrected_gpu_validation"]
    cpu = values["corrected_cpu_capture"]
    materialized = values["corrected_reference_materialization"]
    reference = values["corrected_reference_validation"]
    result = values["corrected_exact_comparison"]
    component = values["accepted_component_manifest"]

    failed_status = failed.get("output_status", {}).get("value", {})
    if not (
        failed.get("nonce") == FAILED_NONCE
        and failed.get("nonce_consumed") is True
        and failed.get("lock_invocation_count") == 1
        and failed.get("returncode") == 75
        and failed.get("payload_or_wrapper_returncode") == 75
        and failed.get("retry_performed") is False
        and failed.get("alternate_arm_performed") is False
        and failed.get("gpu_query_performed") is False
        and failed.get("lock_release", {}).get("acquired") is True
        and failed.get("lock_release", {}).get("own_holder_absent") is True
        and failed_status.get("status") == "CAPTURE_FAILED_NONCE_CONSUMED"
        and failed_status.get("error") == "MANIFEST_NONCE"
    ):
        raise CloseoutFailure("INITIAL_HARNESS_FAILURE_RECEIPT")

    command = corrected.get("lock_command", [])
    intent_index = command.index("--intent") if "--intent" in command else -1
    if not (
        reservation.get("nonce") == NONCE
        and reservation.get("nonce_prefix") == PREFIX
        and reservation.get("single_use") is True
        and reservation.get("consumed_prefixes_forbidden") == list(CONSUMED_PREFIXES)
        and reservation.get("gpu_arm_limit") == 1
        and reservation.get("retry_permitted") is False
        and reservation.get("gpu_query_permitted") is False
        and corrected.get("nonce") == NONCE
        and corrected.get("nonce_consumed") is True
        and corrected.get("lock_invocation_count") == 1
        and corrected.get("returncode") == 0
        and corrected.get("payload_or_wrapper_returncode") == 0
        and corrected.get("retry_performed") is False
        and corrected.get("alternate_arm_performed") is False
        and corrected.get("gpu_query_performed") is False
        and intent_index >= 0
        and command[intent_index + 1] == "production-preemptible"
        and command.count("--intent") == 1
        and command.count("--execute") == 1
        and corrected.get("lock_release", {}).get("acquired") is True
        and corrected.get("lock_release", {}).get("release_line_count") == 1
        and corrected.get("lock_release", {}).get("own_holder_absent") is True
    ):
        raise CloseoutFailure("CORRECTED_SINGLE_ARM_RECEIPT")

    if not (
        gpu.get("valid") is True
        and gpu.get("nonce") == NONCE
        and gpu.get("gpu_arm_count") == 1
        and gpu.get("retry_performed") is False
        and gpu.get("gpu_query_performed") is False
        and gpu.get("slot_count") == 71
        and gpu.get("array_count") == 57
        and gpu.get("none_count") == 14
        and gpu.get("auxiliary_count") == 4
        and gpu.get("manifest_file_sha256")
        == EXTERNAL_FILES["gpu_capture_manifest"][1]
        and gpu.get("output_tree", {}).get("tree_sha256")
        == EXPECTED_CAPTURE_TREE_SHA256
    ):
        raise CloseoutFailure("GPU_CAPTURE_STATUS")

    authority = cpu.get("authority", {})
    consistency = cpu.get("self_consistency", {})
    if not (
        cpu.get("passed") is True
        and cpu.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and authority.get("backend") == "cpu"
        and authority.get("adapter_invocations") == 1
        and authority.get("initialization_invocations") == 1
        and authority.get("gpu_actions") == 0
        and authority.get("wrf_or_mpi_executions") == 0
        and authority.get("adapter_entry", "").endswith(
            "first_timestep=True,restart=False)"
        )
        and all(
            consistency.get(component_name, {})
            .get("solve_x_vs_mean_output", {}).get("max_abs") == 0.0
            and consistency.get(component_name, {})
            .get("solve_implied_vs_adapter_tendency", {}).get("max_abs") == 0.0
            for component_name in ("u", "v")
        )
    ):
        raise CloseoutFailure("CPU_CAPTURE_STATUS")

    if not (
        materialized.get("verdict") == "AUTHENTIC_POSTFIX_28_OF_28"
        and materialized.get("production_adapter_invocations") == 1
        and materialized.get("gpu_actions") == 0
        and reference.get("valid") is True
        and reference.get("array_count") == 28
        and reference.get("production_adapter_invocations") == 1
    ):
        raise CloseoutFailure("REFERENCE_STATUS")

    if not (
        result.get("passed") is True
        and result.get("terminal_verdict")
        == "POSTFIX_SP2_VALIDATED_IMPROVED_BOTH"
        and result.get("exact_comparator_verdict") == "SOURCE_LOCALIZED_LOWER_BC"
        and component.get("terminal_verdict")
        == "SINGLE_AUTHORITY_SOURCE_LOCALIZED_FIX_PROVEN"
        and component.get("passed") is True
    ):
        raise CloseoutFailure("TERMINAL_STATUS")
    for field in PRE_FIX_RMS:
        item = result.get("improvement", {}).get(field, {})
        if not (
            close_enough_exact(result.get("post_fix_rms", {}).get(field), POST_FIX_RMS[field])
            and close_enough_exact(item.get("pre_fix_rms"), PRE_FIX_RMS[field])
            and close_enough_exact(item.get("post_fix_rms"), POST_FIX_RMS[field])
            and item.get("strictly_improved") is True
            and POST_FIX_RMS[field] < PRE_FIX_RMS[field]
        ):
            raise CloseoutFailure(f"RMS:{field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        print(f"REFUSE:OUTPUT_NOT_FRESH:{output}", file=sys.stderr)
        return 74
    try:
        if git_text("status", "--porcelain"):
            raise CloseoutFailure("WORKTREE_NOT_CLEAN")
        head = git_text("rev-parse", "HEAD")
        if subprocess.run(
            [
                "git", "-C", str(REPO), "diff", "--quiet",
                f"{ACCEPTED_COMPONENT_HEAD}..{head}", "--", "src/gpuwrf",
            ],
            check=False,
        ).returncode:
            raise CloseoutFailure("PRODUCTION_SOURCE_DRIFT")

        values: dict[str, dict[str, Any]] = {}
        proof_records: dict[str, dict[str, Any]] = {}
        for name, (path, digest) in LOCAL_PROOFS.items():
            values[name], proof_records[name] = load_canonical(path, digest)
        validate_graph(values)

        external_records: dict[str, dict[str, Any]] = {}
        for name, (path, digest) in EXTERNAL_FILES.items():
            external_records[name] = require_file(path, digest)
            if path.suffix == ".json":
                _, canonical = load_canonical(path, digest)
                external_records[name].update({
                    "canonical_payload_sha256": canonical[
                        "canonical_payload_sha256"
                    ],
                    "schema": canonical["schema"],
                })

        captured_tree = tree_inventory(CAPTURE_ROOT)
        recorded_tree = values["corrected_gpu_validation"].get("output_tree")
        if captured_tree != recorded_tree or (
            captured_tree["tree_sha256"] != EXPECTED_CAPTURE_TREE_SHA256
        ):
            raise CloseoutFailure("CAPTURE_TREE_DRIFT")

        wrf_tree = wrf_tree_inventory(WRF_ROOT)
        if (
            wrf_tree["file_count"] != 462
            or wrf_tree["tree_sha256"] != EXPECTED_WRF_TREE_SHA256
        ):
            raise CloseoutFailure("WRF_TREE_DRIFT")

        report_record = require_file(REPORT)
        report_text = REPORT.read_text(encoding="utf-8")
        if "Terminal: `POSTFIX_SP2_VALIDATED_IMPROVED_BOTH`" not in report_text:
            raise CloseoutFailure("REPORT_TERMINAL")

        terminal, _ = load_canonical(
            EXTERNAL_FILES["comparison_terminal"][0],
            EXTERNAL_FILES["comparison_terminal"][1],
        )
        if not (
            terminal.get("passed") is True
            and terminal.get("terminal_verdict")
            == "POSTFIX_SP2_VALIDATED_IMPROVED_BOTH"
            and terminal.get("strictly_improved_both") is True
            and terminal.get("gpu_arm_count") == 1
            and terminal.get("gpu_retry_count") == 0
            and terminal.get("gpu_query_count") == 0
            and terminal.get("production_cpu_adapter_invocations") == 1
            and terminal.get("tolerance_changes") == 0
        ):
            raise CloseoutFailure("EXTERNAL_TERMINAL")

        proof = {
            "schema": "wrfgpu2-v0234-gpt-postfix-sp2-closeout-v1",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "terminal_verdict": "POSTFIX_SP2_VALIDATED_IMPROVED_BOTH",
            "exact_comparator_verdict": "SOURCE_LOCALIZED_LOWER_BC",
            "passed": True,
            "git_head": head,
            "accepted_component_head": ACCEPTED_COMPONENT_HEAD,
            "production_source_unchanged_since_acceptance": True,
            "generator": require_file(Path(__file__).resolve()),
            "resource_gate": resource_gate(),
            "scope": {
                "fixture": "d03 step-1",
                "corrected_nonce": NONCE,
                "corrected_namespace": str(AUTHORITY_ROOT),
                "failed_nonce_consumed": FAILED_NONCE,
                "previously_consumed_prefixes_forbidden": list(
                    CONSUMED_PREFIXES
                ),
                "corrected_gpu_arms": 1,
                "gpu_retries": 0,
                "gpu_queries": 0,
                "alternate_arms": 0,
                "production_cpu_adapter_invocations": 1,
                "wrf_or_mpi_executions": 0,
                "tolerance_changes": 0,
            },
            "initial_arm": {
                "classification": "AUTHORITY_OR_HARNESS_FAILURE",
                "cause": "stale Python default expected_nonce binding",
                "accepted_scientific_result_produced": False,
                "nonce_consumed": True,
                "receipt": proof_records["initial_failed_gpu_receipt"],
            },
            "corrected_arm": {
                "intent": "production-preemptible",
                "lock_acquired_and_released": True,
                "capture_slots": 71,
                "seam_trace_invocations": 1,
                "pbl_invocations": 0,
                "capture_tree": {
                    "path": str(CAPTURE_ROOT),
                    "file_count": captured_tree["file_count"],
                    "sha256": captured_tree["tree_sha256"],
                },
            },
            "same_authority_cpu": {
                "backend": "cpu",
                "fresh_start_restart": False,
                "production_adapter_invocations": 1,
                "production_exact_internal_closure": True,
                "reference_array_count": 28,
                "mix_sm_provenance": (
                    "source_instrumented_diagnostic_replay_unreturned_sm_only"
                ),
                "diagnostic_replay_is_production_replacement": False,
            },
            "comparison": {
                "authentic_pre_fix_rms": PRE_FIX_RMS,
                "post_fix_rms": POST_FIX_RMS,
                "improvement": values["corrected_exact_comparison"]["improvement"],
                "strictly_improved_both": True,
            },
            "proof_objects": proof_records,
            "external_objects": external_records,
            "focused_test_output": validate_junit(),
            "worker_report": report_record,
            "wrf_dump_tree": {
                "path": str(WRF_ROOT),
                "file_count": wrf_tree["file_count"],
                "sha256": wrf_tree["tree_sha256"],
            },
            "limitations": {
                "full_sp2_parity_claimed": False,
                "remaining_error_is_nonzero": True,
                "rho_entry_divergence_out_of_scope": True,
                "cpu_aot_machine_feature_warnings_observed": True,
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
        json.JSONDecodeError, subprocess.SubprocessError, ET.ParseError,
    ) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
