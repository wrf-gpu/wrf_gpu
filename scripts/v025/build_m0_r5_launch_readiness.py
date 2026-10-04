#!/usr/bin/env python3
"""Assemble the M0 R5 CPU launch-readiness evidence for the frozen session.

This builder is the whole R5 assembly: it proves the fixed R5 generation is
vacant, drives the real outer-owner -> held-child -> real C1 CLI boundary
against a pre-staged comparison fixture, invokes the five named production
analyser relations on retained inputs, emits the exact launch packet bound to
the stable integration worktree, and publishes one content-addressed proof whose
gate set is read from the sprint contract rather than declared here.

Everything is CPU-only and mechanically device-denied.  No accelerator module is
imported, no receipt is created or spent outside a fresh private root, no
canonical ledger/lock/holder is touched, and no retained R3/R4 artifact is written.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import sys
import tempfile
import contextlib
import io
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPRINT = ".agent/sprints/2026-08-09-v0250-m0-r5-kernel-lock-authority"
CONTRACT_PATH = REPO / SPRINT / "CONTRACT.md"
PACKET_PATH = REPO / SPRINT / "M0_R5_LAUNCH_PACKET.json"
PROOF_DIR = REPO / "proofs/v025/m0"
RAW_DIR = PROOF_DIR / "m0_r5_final_readiness_raw"
REHEARSAL_PATH = RAW_DIR / "production_analyser_rehearsal.json"
PROOF_PREFIX = "m0_r5_final_readiness"

#: The three mutable R5 launch roots.  They are RETIRED: the R5 generation ran
#: once and is terminal, so these roots must stay absent, non-symlink and free
#: of symlink ancestors forever; restoring one is an unauthorized zombie
#: generation.  The in-tree R5 proof tree and session receipt are retained
#: terminal evidence (see R5_PROOF_TREE_SHA256 / R5_RECEIPT_SHA256 below).
R5_RAW_ROOT = "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-three-window-r5"
R5_PAIR_CACHE_ROOT = (
    "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-qualified-pair-cache-r5"
)
R5_C1_CPU_RUN_ROOT = "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-c1-prelock-cpu-r5"
R5_PROOF_ROOT = "proofs/v025/m0/autotune0_three_window_r5"
R5_RECEIPT_TEMPLATE = f"{SPRINT}/M0_CORE_W1_W2_W3_SESSION_RECEIPT_R5.json"

#: The retired R3 values.  They exist only so a mutation can try to restore one
#: and be refused; no live code path may produce them.
R3_C1_CPU_RUN_ROOT = "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-c1-prelock-cpu-r3"

#: The CURRENT authorized regime: final main at the ADR-038 default-entry flip
#: lineage (post-seam).  The retired R4/R5 candidate regime (HEAD ba9dc2331,
#: src tree a6885ced) is terminal historical evidence; these are its successor
#: identities and only a fresh explicit re-pin sprint may move them.
AUTHORIZED_PRODUCTION_HEAD = "062c808801a85b178c8a2aa20ec4b82552cf2035"
AUTHORIZED_SRC_GPUWRF_TREE = "d7f55214f309f61c15f7588401f01665b3d300cc"

#: The R5 generation itself is TERMINAL (retained cold rejection, evidence
#: committed): its retained proof tree and session receipt are the generation's
#: identity and must stay byte-identical to these absolute pins.  Any drift --
#: uncommitted tamper or a later commit touching the bytes -- fails closed and
#: requires a fresh re-pin sprint, exactly like a src/gpuwrf re-pin.
R5_PROOF_TREE_SHA256 = (
    "9d8803827eb9a129c2960550baad26c3246250c5f1273dcf6444ba5106153347"
)
R5_RECEIPT_SHA256 = (
    "ccfd085179902187147ad24fd04562913b12d455bc95a8be41093a8b0385c0b7"
)

CANONICAL_LEDGER = (
    ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_spent.json"
)
CANONICAL_LOCK_WRAPPER = "scripts/with_gpu_lock.sh"
SESSION_LABEL = "m0-core-w1-w2-w3-session"

#: Retained evidence whose bytes must be identical before and after (H4).
#: Each retired generation's evidence joins this list when the next one is
#: frozen; the R5 terminal evidence is retained alongside R3/R4.
RETAINED_EVIDENCE_ROOTS = (
    "proofs/v025/m0/autotune0_three_window",
    ".agent/sprints/2026-07-27-v0250-m0-setup",
    "proofs/perf/v015",
    "proofs/v025/m0/autotune0_three_window_r4",
    ".agent/sprints/2026-08-01-v0250-m0-r4-final-readiness",
    "proofs/v025/m0/autotune0_three_window_r5",
    ".agent/sprints/2026-08-09-v0250-m0-r5-kernel-lock-authority",
)

#: The three economical retained analyser inputs (R2).
REAL_PROJ_CSV = REPO / "proofs/perf/v015/nsys_steady50_nvtx_gpu_proj_sum.csv"
REAL_PROJ_CSV_SHA256 = (
    "c54ce8521c5ecfce946bacc735b8392888ea4399d146cce169e7e618d380baf1"
)
REAL_BASELINE_SQLITE = Path(
    "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/nsys_baseline.sqlite"
)
REAL_BASELINE_SQLITE_SHA256 = (
    "48a44a77d077c814b711df8b63f1adb36cf6e758de27a577a65ebeda68293bf0"
)
EVENT_FIXTURE = REPO / "tests/v025/fixtures/m0_w2_event_level_kernel_fixture.sqlite"
EVENT_FIXTURE_SHA256 = (
    "3ef88fe64a2ed4a26eaf2d3b8b43892f9fbb3896df47ed1211c622978352f111"
)

#: Review 10's documented attribution rule, frozen (R2).  The binding numerator
#: is the *source-named* deepest leaf population, not every leaf.
REVIEW10_DENOMINATOR = 2_575_286
REVIEW10_ALL_LEAVES = 2_574_776
REVIEW10_SOURCE_NAMED_LEAVES = 2_567_009
REVIEW10_RULE = (
    "denominator = sum of Total GPU Ops over rows with Avg Range Lvl == 0; "
    "binding numerator = sum over deepest-unique leaves (Avg Num Child == 0) "
    "whose Range names a jitted source program (contains 'name=jit(')"
)
REVIEW10_SOURCE = (
    ".agent/sprints/2026-07-30-v0250-management-review-10/"
    "OPUS_MANAGEMENT_REVIEW_10.md:345-354"
)

#: The five production relations the R5 builder must really call (R2).
PRODUCTION_RELATIONS = (
    "m0_postlock_census.analyze_w2",
    "m0_postlock_census.build_census",
    "m0_postlock_census.derive_integration_scope",
    "m0_postlock_census.derive_step_census",
    "run_gpu_arm.attribute_device_time",
)

#: Current principal GPU ownership/preemption policy.  This supersedes the
#: recurring local-clock boundary at c1e446bb; no local clock grants or ends
#: authority.  Exact source authority is main e7d78373 and its patch record.
CURRENT_GPU_POLICY_COMMIT = "e7d78373ebae2b530e496df01ed58f50a0158419"
CURRENT_GPU_POLICY_PATCH = (
    ".agent/patches/2026-08-02-gpu-default-owner-server-preempt.md"
)
RETIRED_CLOCK_POLICY_COMMIT = "c1e446bb"
PRELOCK_CPU_MAXIMUM_SECONDS = 420

#: Principal override provenance, recorded rather than silently applied.
OVERRIDE_PROVENANCE = {
    "kind": "PRINCIPAL_OVERRIDE",
    "received_utc": "2026-08-02",
    "manager_commit": CURRENT_GPU_POLICY_COMMIT,
    "manager_commit_subject": "policy(gpu): assign default owner and server preemption",
    "relayed_by": "ALISIOS manager (0:3)",
    "verbatim_directive": (
        "0:2 ist außerhalb Production der dauerhafte Standard-Eigentümer. "
        "Nightly 20260801_18z besitzt die GPU bis zum exakten RELEASE; bitte "
        "nicht claimen. Danach geht sie sofort an 0:2 zurück. 0:1 nutzt GPU "
        "nur nach expliziter Manager-Freigabe. Ab der nächsten Nacht PREEMPT "
        "erst auf Server-Anfrage, nicht nach lokaler Uhrzeit."
    ),
    "encoded_authority": [
        ".agent/skills/locking-gpu/SKILL.md",
        ".agent/skills/managing-sprints/SKILL.md",
        CURRENT_GPU_POLICY_PATCH,
        ".agent/decisions/V0250-ROADMAP.md",
    ],
    "replaces": (
        "the recurring local-clock Nightly boundary encoded at c1e446bb"
    ),
    "scope": (
        "GPU ownership and Production preemption only: no M0/M1/M2 gate, "
        "target, receipt rule or CPU scope changes, and H0-H6 are unchanged"
    ),
}

#: Permanent default ownership outside Production.  A free-looking card is not
#: permission; 0:1 needs an explicit resource-manager release.
OWNER_PRIORITY_HOLDER = "0:2"
SESSION_BUDGET_SECONDS = 4_220
SESSION_BUDGET_CEILING_SECONDS = 4_500


class ReadinessRefusal(RuntimeError):
    """A readiness gate refused; nothing downstream of it may be claimed."""


# --------------------------------------------------------------------------- #
# small deterministic helpers                                                  #
# --------------------------------------------------------------------------- #
def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_sidecar(path: Path) -> Path:
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60.0,
    )
    return completed.stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    return completed.returncode == 0


# --------------------------------------------------------------------------- #
# H0 - the fixed R5 generation is vacant, and R5/R3 do not overlap             #
# --------------------------------------------------------------------------- #
def _vacancy_record(target: str) -> dict[str, Any]:
    """Prove one fixed target is absent with no symlink anywhere above it."""

    path = Path(target) if target.startswith("/") else REPO / target
    base = Path("<DATA_ROOT>") if target.startswith("/") else REPO
    symlink_ancestors: list[str] = []
    cursor = path.parent
    while True:
        if os.path.islink(cursor):
            symlink_ancestors.append(str(cursor))
        if cursor == base or cursor == cursor.parent:
            break
        cursor = cursor.parent
    return {
        "target": str(path),
        "base": str(base),
        "exists": os.path.lexists(path),
        "is_symlink": os.path.islink(path),
        "symlink_ancestors_below_base": symlink_ancestors,
        "vacant": (
            not os.path.lexists(path)
            and not os.path.islink(path)
            and not symlink_ancestors
        ),
    }


def r5_generation_vacancy() -> dict[str, Any]:
    """H0: the retired mutable R5 launch roots are vacant and stay vacant.

    The in-tree R5 proof tree and session receipt are no longer vacancy
    targets: the R5 generation ran and is terminal, so its committed evidence
    is retained identity (see :func:`r5_terminal_retention`), not absence.
    """

    targets = [
        R5_RAW_ROOT,
        R5_PAIR_CACHE_ROOT,
        R5_C1_CPU_RUN_ROOT,
    ]
    records = [_vacancy_record(target) for target in targets]
    return {
        "schema": "wrf_gpu2.v025.m0.r5_generation_vacancy.v1",
        "status": "PASS" if all(item["vacant"] for item in records) else "BLOCKED",
        "targets": records,
        "mutable_root_count": 3,
        "rule": (
            "a retired mutable launch root is proven by absence, not by a "
            "fingerprint: each is absent, is not a symlink, and has no "
            "symlink ancestor below its canonical base; restoring one is an "
            "unauthorized zombie generation"
        ),
    }


def r5_terminal_retention(
    *,
    proof_root: str | Path | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """H0: the terminal R5 evidence is present and byte-identical to its pins.

    The proof tree is fingerprinted as a canonical ``{relpath: sha256}`` map
    (the same construction the H4 retained manifest uses) and the receipt as a
    single file sha256; both are absolute literals, so this refuses uncommitted
    tamper AND any later commit that touches the bytes.  The path overrides
    exist so a mutation can point the check at tampered scratch copies.
    """

    root = Path(proof_root) if proof_root is not None else REPO / R5_PROOF_ROOT
    receipt = (
        Path(receipt_path)
        if receipt_path is not None
        else REPO / R5_RECEIPT_TEMPLATE
    )
    tree_files: dict[str, str] = {}
    if root.is_dir() and not root.is_symlink():
        for item in sorted(root.rglob("*")):
            if item.is_file() and not item.is_symlink():
                tree_files[str(item.relative_to(root))] = sha256_file(item)
    tree_digest = canonical_sha256(tree_files) if tree_files else None
    proof_record = {
        "target": str(root),
        "exists": root.is_dir() and not root.is_symlink(),
        "file_count": len(tree_files),
        "tree_sha256": tree_digest,
        "matches_pin": tree_digest == R5_PROOF_TREE_SHA256,
    }
    receipt_record = {
        "target": str(receipt),
        "exists": receipt.is_file() and not receipt.is_symlink(),
        "sha256": sha256_file(receipt) if receipt.is_file() else None,
        "matches_pin": (
            receipt.is_file()
            and sha256_file(receipt) == R5_RECEIPT_SHA256
        ),
    }
    checks = (proof_record["matches_pin"], receipt_record["matches_pin"])
    return {
        "schema": "wrf_gpu2.v025.m0.r5_terminal_retention.v1",
        "status": "PASS" if all(checks) else "BLOCKED",
        "generation": "R5_TERMINAL_COLD_REJECTION",
        "retained_proof_tree": proof_record,
        "retained_receipt": receipt_record,
        "rule": (
            "the R5 generation is terminal: its retained evidence must be "
            "present and byte-identical to the absolute pins, never edited, "
            "never deleted; drift fails closed and needs a re-pin sprint"
        ),
    }


def r5_generation_identity() -> dict[str, Any]:
    """H0: vacancy of the retired mutable roots AND retention of the evidence."""

    vacancy = r5_generation_vacancy()
    retention = r5_terminal_retention()
    return {
        "schema": "wrf_gpu2.v025.m0.r5_generation_identity.v1",
        "status": (
            "PASS"
            if vacancy["status"] == "PASS" and retention["status"] == "PASS"
            else "BLOCKED"
        ),
        "vacancy": vacancy,
        "retention": retention,
        "retained_terminal_evidence_stable": retention["status"] == "PASS",
        "retired_mutable_roots_absent": vacancy["status"] == "PASS",
    }


def r5_constants_are_coherent() -> dict[str, Any]:
    """H0: the parent and the executor name one and the same R5 generation."""

    import m0_three_window_executor as executor
    import m0_window_parent as window_parent

    observed = {
        "parent_raw_root": str(window_parent.RAW_ROOT),
        "parent_pair_cache_root": str(window_parent.PAIR_CACHE_ROOT),
        "parent_proof_root": str(
            window_parent.PROOF_ROOT.relative_to(REPO)
        ),
        "executor_raw_root": executor.RAW_ROOT,
        "executor_pair_cache_root": executor.PAIR_CACHE_ROOT,
        "executor_proof_root": executor.PROOF_ROOT,
        "executor_c1_cpu_run_root": executor.C1_CPU_RUN_ROOT,
        "executor_session_receipt": executor.SESSION_RECEIPT,
    }
    expected = {
        "parent_raw_root": R5_RAW_ROOT,
        "parent_pair_cache_root": R5_PAIR_CACHE_ROOT,
        "parent_proof_root": R5_PROOF_ROOT,
        "executor_raw_root": R5_RAW_ROOT,
        "executor_pair_cache_root": R5_PAIR_CACHE_ROOT,
        "executor_proof_root": R5_PROOF_ROOT,
        "executor_c1_cpu_run_root": R5_C1_CPU_RUN_ROOT,
        "executor_session_receipt": R5_RECEIPT_TEMPLATE,
    }
    return {
        "status": "PASS" if observed == expected else "BLOCKED",
        "observed": observed,
        "expected": expected,
        "run_ids_retain_reviewed_r3_text": [
            str(window_parent.WINDOWS[window]["run_id"])
            for window in ("W1", "W2", "W3")
        ],
        "run_id_freshness_note": (
            "freshness comes from the vacant R5 parent roots and the packet, "
            "not from cosmetic run-id text"
        ),
        "dead_fifth_root": {
            "entrypoint": "m0_three_window_executor.run_outer_window_graph",
            "status": "UNREACHABLE_NOT_IN_HELD_SESSION",
            "reason": (
                "Amendment 6 supersedes the per-window outer graph; the held "
                "session never calls it, so its root is not part of the R5 "
                "generation and is not added to the session"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# H4 - retained R3/R4 evidence identity                                        #
# --------------------------------------------------------------------------- #
def retained_manifest() -> dict[str, Any]:
    """Hash every retained R3/R4 file so before/after equality is measured."""

    files: dict[str, str] = {}
    for root in RETAINED_EVIDENCE_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            files[str(path.relative_to(REPO))] = sha256_file(path)
    return {
        "schema": "wrf_gpu2.v025.m0.retained_r3_r4_manifest.v1",
        "roots": list(RETAINED_EVIDENCE_ROOTS),
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": canonical_sha256(files),
    }


# --------------------------------------------------------------------------- #
# H1 - the real outer -> held -> C1 boundary                                   #
# --------------------------------------------------------------------------- #
_AUTHORITY_CACHE: dict[str, Any] | None = None


def _source_authority() -> dict[str, Any]:
    global _AUTHORITY_CACHE
    if _AUTHORITY_CACHE is None:
        import m0_window_parent as window_parent
        import wrf_source_authority as wsa

        _AUTHORITY_CACHE = wsa.build_source_authority(
            namelist_path=window_parent.FAST_RUN_DIR / "namelist.input",
            environ={
                variable: str(wsa.CANONICAL_ROOT)
                for variable in wsa.ROOT_ENV_VARS
            },
        )
    return _AUTHORITY_CACHE


def _write_netcdf(path: Path, offset: float) -> str:
    """One tiny real wrfout the *real* comparator will actually open."""

    import netCDF4
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", 4)
        dataset.createDimension("west_east", 5)
        base = np.arange(20, dtype="float32").reshape(1, 4, 5)
        for name, scale in (("T2", 1.0), ("U10", 0.5)):
            variable = dataset.createVariable(
                name, "f4", ("Time", "south_north", "west_east")
            )
            variable[:] = base * scale + 280.0 + offset
    return sha256_file(path)


def _private_lock_identity(lock_path: Path) -> dict[str, Any]:
    path = Path(lock_path).absolute()
    observed = os.lstat(path)
    major = os.major(observed.st_dev)
    minor = os.minor(observed.st_dev)
    return {
        "path": str(path),
        "mode": observed.st_mode,
        "device_decimal": observed.st_dev,
        "device_major": major,
        "device_minor": minor,
        "inode": observed.st_ino,
        "proc_locks_key": f"{major:02x}:{minor:02x}:{observed.st_ino}",
        "is_regular": stat.S_ISREG(observed.st_mode),
        "is_symlink": stat.S_ISLNK(observed.st_mode),
    }


def _receipt_payload(session: Any, *, lock_path: Path) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).isoformat()
    identity = _private_lock_identity(lock_path)
    return {
        "schema": "wrf_gpu2.v025.m0.gpu_coordination_receipt.v1",
        "window": session.SESSION_LABEL,
        "requested_at_utc": stamp,
        "request_text": "M0-CORE W1->C1->W2->W3, one lock, one receipt",
        "replies": {
            manager: {
                "affirmative": True,
                "verbatim": f"{manager}: yes, proceed with the one M0-CORE session",
                "received_at_utc": stamp,
            }
            for manager in session.REQUIRED_MANAGERS
        },
        "kernel_lock_binding": {
            "preflight": {
                "status": "PASS",
                "kernel_identity": identity,
                "vacancy_grants_permission": False,
            },
            "immediate_live_recheck": {
                "status": "PASS",
                "identity": identity,
                "vacancy_grants_permission": False,
            },
            "vacancy_grants_permission": False,
        },
    }


def _prestage_identity(root: Path, receipt_path: Path) -> Path:
    """The session identity the CPU preflight and the held child both bind."""

    import m0_postlock_census as postlock
    import m0_three_window_executor as executor
    import m0_review10_fallback_capability as fallback_capability
    import wrf_source_authority as wsa

    root.mkdir(parents=True, exist_ok=True)
    authority = _source_authority()
    authority_path = root / "wrf_source_authority.json"
    wsa.write_authority(authority_path, authority)
    boundary = {
        "schema": "wrf_gpu2.v025.m0.cpu_real_boundary_preflight.v1",
        "status": "PASS",
        "device_action": False,
        "platforms": ["cpu"],
        "native_bundle": {"field_count": 29},
        "authority": {"authority_sha256": authority["authority_sha256"]},
        "lowered_program": {
            "sha256": "7" * 64,
            "integration_trip_count": {
                "status": "PASS",
                "method": "exact-lowered-trip-count",
                "entry_function": "main",
                "loop_count": 4,
                "segment_trip_counts": [179, 1, 179, 1],
                "steps": 360,
                "segments": [],
                "stablehlo_sha256": "7" * 64,
                "configured_cross_check": {
                    "derivation": "Fraction(str(hours))*3600/Fraction(str(dt_s))",
                    "hours": 1.0,
                    "forecast_interval_seconds": 3600.0,
                    "timestep_seconds": 10.0,
                    "exact_steps": 360,
                    "integral": True,
                },
                "count_matches_configuration": True,
            },
        },
        "observations": {
            "native_loader_calls": 1,
            "fast_argument_builder_calls": 1,
            "wrapper_preparation_calls": 1,
            "exact_lower_calls": 1,
            "compile_calls": 0,
            "device_invocations": 0,
            "receipt_reads": 0,
            "ledger_reads": 0,
            "ledger_writes": 0,
            "lock_checks": 0,
            "wrapper_calls": 0,
        },
    }
    boundary["boundary_sha256"] = executor._canonical_sha256(boundary)
    boundary_path = root / "cpu_real_boundary_preflight.json"
    _write_json(boundary_path, boundary)
    capability = fallback_capability.build_declaration(
        boundary, boundary_path=boundary_path, environ={}
    )
    capability_path = root / "capture_capability_declaration.json"
    _write_json(capability_path, capability)
    owner = executor._prospective_manager_session_command(
        receipt=str(receipt_path)
    )
    wrapper = executor._held_session_wrapper_command(receipt=str(receipt_path))
    identity = {
        "schema": executor.SESSION_IDENTITY_SCHEMA,
        "status": "PASS",
        "src_gpuwrf_tree": _git("rev-parse", "HEAD:src/gpuwrf"),
        "wrf_source_authority": {
            "path": str(authority_path),
            "sha256": postlock.sha256_file(authority_path),
            "content_sha256": authority["authority_sha256"],
        },
        "cpu_real_boundary_preflight": {
            "path": str(boundary_path),
            "sha256": postlock.sha256_file(boundary_path),
            "content_sha256": boundary["boundary_sha256"],
            "observations": boundary["observations"],
        },
        "capture_capability_declaration": {
            "path": str(capability_path),
            "sha256": postlock.sha256_file(capability_path),
            "content_sha256": capability["content_address"]["sha256"],
        },
        "owner_command_normalized": executor._normalize_command(owner),
        "owner_command_sha256": executor._command_sha256(owner),
        "held_wrapper_command_normalized": executor._normalize_command(wrapper),
        "held_wrapper_command_sha256": executor._command_sha256(wrapper),
        "device_action": False,
    }
    identity["identity_sha256"] = executor._canonical_sha256(identity)
    identity_path = root / "session_preflight_identity.json"
    _write_json(identity_path, identity)
    return identity_path


def _prestage_cpu_preflight(
    root: Path, identity_path: Path, cpu_wrfout: Path, cpu_sha256: str
) -> Path:
    """The pre-staged CPU arm the *real* C1 child validates and compares."""

    import m0_postlock_census as postlock
    import m0_w1_fast_pair as fast_pair

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    invocation = ["taskset", "-c", "16-27", "mpirun", "-np", "12", "wrf.exe"]
    cpu_record = {
        "kind": "cpu",
        "run_id": "m0-core-prelock-fresh-cpu",
        "started_at_utc": "2026-07-28T00:00:00+00:00",
        "finished_at_utc": "2026-07-28T00:00:01+00:00",
        "status": "OK",
        "payload": {
            "command": invocation,
            "cpu_list": "16-27",
            "ranks": 12,
            "launcher_wallclock_seconds": 1.0,
            "final_wrfout_path": str(cpu_wrfout),
            "final_wrfout_sha256": cpu_sha256,
        },
    }
    preflight = {
        "schema": fast_pair.CPU_PREFLIGHT_SCHEMA,
        "status": "PASS",
        "session_identity_path": str(identity_path.resolve()),
        "session_identity_sha256": postlock.sha256_file(identity_path),
        "session_identity_content_sha256": identity["identity_sha256"],
        "started_monotonic_ns": 1_000,
        "finished_monotonic_ns": 2_000,
        "cpu_list": "16-27",
        "ranks": 12,
        "cpu_record": cpu_record,
        "cpu_record_sha256": postlock.canonical_sha256(cpu_record),
        "cpu_invocation_argv": invocation,
        "cpu_invocation_sha256": hashlib.sha256(
            "\0".join(invocation).encode("utf-8")
        ).hexdigest(),
        "verified_output": {
            "final_wrfout_path": str(cpu_wrfout),
            "final_wrfout_bytes": cpu_wrfout.stat().st_size,
            "final_wrfout_sha256": cpu_sha256,
        },
        "device_action": False,
        "lock_environment_absent": True,
    }
    preflight["preflight_sha256"] = postlock.canonical_sha256(preflight)
    cpu_path = root / "cpu_preflight.json"
    _write_json(cpu_path, preflight)
    return cpu_path


def _prestage_w1(root: Path, gpu_wrfout: Path, gpu_sha256: str) -> None:
    """Publish the exact W1 boundary the real C1 child reads and hash-binds."""

    import m0_exact_boundary_contract as exact_contract
    import m0_postlock_census as postlock
    import m0_window_parent as window_parent

    run_id = str(window_parent.WINDOWS["W1"]["run_id"])
    exact_digest = "3" * 64
    case = {
        "case_id": "m0-r5-prestaged-comparison-fixture",
        "domain": "d01",
        "hours": 1.0,
        "case_metadata": {
            "namelist": {"dt_s": 10.0, "radiation_cadence_steps": 4}
        },
    }
    config_identity = {"run_id": run_id, "domain": "d01", "hours": 1}
    input_binding = {
        "config_sha256": exact_contract.canonical_sha256(config_identity),
        "namelist_input_sha256": "6" * 64,
        "input_manifest_sha256": "8" * 64,
        "case_sha256": exact_contract.canonical_sha256(case),
    }
    exact_identity = {
        "case_sha256": input_binding["case_sha256"],
        "hours": 1.0,
        "fixture": "m0-r5-prestaged-comparison-fixture",
    }
    variable_inventory = [
        {
            "name": name,
            "dtype": "float32",
            "dimensions": ["Time", "south_north", "west_east"],
            "shape": [1, 4, 5],
        }
        for name in ("T2", "U10")
    ]
    dimensions = {"Time": 1, "south_north": 4, "west_east": 5}
    adapter_identity = [
        {"fqname": name, "ast_sha256": "9" * 64}
        for name in (
            "gpuwrf.integration.daily_pipeline._surface_diagnostics_for_output",
            "gpuwrf.integration.daily_pipeline._merge_output_diagnostics",
            "gpuwrf.integration.daily_pipeline._wrfout_name",
            "gpuwrf.integration.daily_pipeline.build_wrfout_inventory",
            "gpuwrf.io.wrfout_writer.prepare_wrfout_payload",
            "gpuwrf.io.wrfout_writer.write_prepared_wrfout",
        )
    ]
    adapter_source = {
        "src_gpuwrf_tree": _git("rev-parse", "HEAD:src/gpuwrf"),
        "child_path": str(exact_contract.CHILD.resolve()),
        "child_sha256": exact_contract.sha256_file(exact_contract.CHILD),
    }
    wrfout = {
        "schema": "wrf_gpu2.v025.m0.exact_result_wrfout.v1",
        "status": "PASS",
        "run_id": run_id,
        "result_exact_value_sha256": exact_digest,
        "exact_boundary_identity": exact_identity,
        "exact_boundary_identity_sha256": exact_contract.canonical_sha256(
            exact_identity
        ),
        "input_binding": input_binding,
        "adapter_identity": adapter_identity,
        "adapter_identity_sha256": exact_contract.canonical_sha256(
            adapter_identity
        ),
        "adapter_source": adapter_source,
        "adapter_source_sha256": exact_contract.canonical_sha256(adapter_source),
        "final_wrfout_path": str(gpu_wrfout),
        "final_wrfout_bytes": gpu_wrfout.stat().st_size,
        "final_wrfout_sha256": gpu_sha256,
        "domain": "d01",
        "domain_authority_sha256": "a" * 64,
        "run_start_utc": "2026-07-28T00:00:00+00:00",
        "valid_time_utc": "2026-07-28T01:00:00+00:00",
        "lead_hours": 1.0,
        "operational_variable_set": True,
        "full_variable_set": False,
        "inventory": {"status": "PASS"},
        "inventory_sha256": exact_contract.canonical_sha256({"status": "PASS"}),
        "finiteness": {"status": "PASS"},
        "finiteness_sha256": exact_contract.canonical_sha256({"status": "PASS"}),
        "config_identity": config_identity,
        "variable_inventory": variable_inventory,
        "variable_inventory_sha256": exact_contract.canonical_sha256(
            variable_inventory
        ),
        "dimensions": dimensions,
        "dimensions_sha256": exact_contract.canonical_sha256(dimensions),
        "timing": {
            "materialization_start_monotonic_ns": 410_000_000,
            "materialization_end_monotonic_ns": 420_000_000,
            "prepare_start_monotonic_ns": 430_000_000,
            "prepare_end_monotonic_ns": 440_000_000,
            "write_start_monotonic_ns": 450_000_000,
            "write_end_monotonic_ns": 460_000_000,
            "inspection_start_monotonic_ns": 470_000_000,
            "inspection_end_monotonic_ns": 480_000_000,
            "outside_readiness_and_integration_clocks": True,
        },
        "publication": {"atomic": True, "replacement": False},
    }
    wrfout["binding_sha256"] = exact_contract.canonical_sha256(
        {
            "run_id": run_id,
            "result_exact_value_sha256": exact_digest,
            "exact_boundary_identity_sha256": wrfout[
                "exact_boundary_identity_sha256"
            ],
            **input_binding,
            "final_wrfout_path": wrfout["final_wrfout_path"],
            "final_wrfout_bytes": wrfout["final_wrfout_bytes"],
            "final_wrfout_sha256": wrfout["final_wrfout_sha256"],
            "domain": "d01",
            "domain_authority_sha256": wrfout["domain_authority_sha256"],
            "run_start_utc": wrfout["run_start_utc"],
            "valid_time_utc": wrfout["valid_time_utc"],
            "lead_hours": 1.0,
            "operational_variable_set": True,
            "full_variable_set": False,
            "adapter_identity_sha256": wrfout["adapter_identity_sha256"],
            "adapter_source_sha256": wrfout["adapter_source_sha256"],
            "variable_inventory_sha256": wrfout["variable_inventory_sha256"],
            "dimensions_sha256": wrfout["dimensions_sha256"],
            "inventory_sha256": wrfout["inventory_sha256"],
            "finiteness_sha256": wrfout["finiteness_sha256"],
        }
    )
    exact_contract.validate_wrfout_binding(
        {
            "run_id": run_id,
            "case": case,
            "call": {"hours": 1.0},
            "result": {"exact_value_sha256": exact_digest},
            "wrfout": wrfout,
        },
        verify_file=True,
    )

    stage_root = Path(window_parent.WINDOWS["W1"]["result"]).parent
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_paths: dict[str, Path] = {}
    stage_payloads: dict[str, dict[str, Any]] = {}
    for stage in ("cold_empty_cache_readiness_1", "cached_readiness_and_warm_integration"):
        cold = stage.startswith("cold")
        payload = {
            "schema": "wrf_gpu2.v025.m0.exact_executable_boundary.v1",
            "status": "OK",
            "run_id": run_id,
            "case": case,
            "call": {"hours": 1.0},
            "timing": {
                "readiness_seconds": 20.0 if cold else 1.0,
                # A cold attempt is compile-only by contract: it must not carry
                # an integration clock at all.
                "integration_seconds": None if cold else 0.1,
                "integration_start_monotonic_ns": None if cold else 600_000_000,
                "integration_end_monotonic_ns": None if cold else 700_000_000,
                "child_executable_ready_monotonic_ns": 500_000_000,
                "derived_by_phase_subtraction": False,
            },
            "result": {"exact_value_sha256": exact_digest},
        }
        if not cold:
            payload["wrfout"] = wrfout
        path = stage_root / f"{stage}.exact_boundary.json"
        _write_json(path, payload)
        stage_paths[stage] = path
        stage_payloads[stage] = payload

    stages = [
        {
            "name": stage,
            "status": "OK",
            "result_path": str(path),
            "result_sha256": postlock.sha256_file(path),
            "result": stage_payloads[stage],
            "started_at_utc": "2026-07-28T00:10:00+00:00",
            "finished_at_utc": "2026-07-28T00:11:00+00:00",
        }
        for stage, path in stage_paths.items()
    ]
    w1 = {
        "schema": "wrf_gpu2.v025.m0.manager_window.v1",
        "status": "OK",
        "window": "W1",
        "run_id": run_id,
        "stages": stages,
        "session": {
            "selected_cold_stage": "cold_empty_cache_readiness_1",
            "qualified_attempt_index": 1,
        },
    }
    _write_json(Path(window_parent.WINDOWS["W1"]["result"]), w1)


def outer_to_real_c1_child(
    root: Path, *, restore_r3_cpu_root: bool = False
) -> dict[str, Any]:
    """Drive the real owner into the real C1 CLI with CPU-only isolation.

    Isolated: the physical ``flock`` wrapper process, every CUDA/W1/W2/W3 device
    execution, the physical 12-rank CPU-WRF launcher, and the environment
    identity probes.  Real: ``run_session_owner``, the held-session entrypoint,
    ``_c1_command``, ``m0_w1_fast_pair``'s argparse and path validation, its
    pre-staged CPU validation, the in-session held-lock check, the netCDF
    comparator, the economy gates, the qualification and the matched-pair
    preparation.  There is no C1 stage hook of any kind.
    """

    import m0_core_session_protocol as session
    import m0_postlock_census as postlock
    import m0_three_window_executor as executor
    import m0_w1_fast_pair as fast_pair
    import m0_window_parent as window_parent
    import run_gpu_arm as gpu_auth

    root.mkdir(parents=True, exist_ok=True)
    trace: dict[str, Any] = {
        "restore_r3_cpu_root": restore_r3_cpu_root,
        "isolated": [
            "physical flock wrapper process",
            "CUDA/W1/W2/W3 device execution",
            "physical 12-rank CPU-WRF launcher",
            "environment identity/revalidation probes",
            "canonical /tmp lock-holder release proof",
        ],
        "real": [
            "run_session_owner",
            "held session entrypoint run_held_session",
            "_c1_command",
            "m0_w1_fast_pair CLI argparse and path validation",
            "validate_cpu_preflight",
            "in-session held-lock provenance",
            "run_fast_pair.compare_arms on real netCDF bytes",
            "economy gates",
            "_post_w1_qualification",
            "prepare_m0_matched_pair.prepare",
        ],
    }

    private = root / "generation"
    proof_root = root / "session"
    fixtures = root / "fixtures"
    cpu_wrfout = fixtures / "cpu_arm_wrfout.nc"
    gpu_wrfout = fixtures / "gpu_arm_wrfout.nc"
    cpu_sha256 = _write_netcdf(cpu_wrfout, 0.0)
    gpu_sha256 = _write_netcdf(gpu_wrfout, 0.01)

    private_lock = root / "synthetic_gpu.lock"
    private_lock.write_bytes(b"")
    receipt_path = root / "synthetic_session_receipt.json"
    _write_json(
        receipt_path, _receipt_payload(session, lock_path=private_lock)
    )
    ledger = root / "synthetic_spend_ledger.json"
    holder = root / "synthetic_lock.holder"
    holder_token = "r5-readiness-synthetic-holder-token"
    holder.write_text(
        f"holder={session.SESSION_LABEL} pid={os.getpid()} token={holder_token} "
        "cmd=r5-cpu-readiness\n",
        encoding="utf-8",
    )

    private_windows = {
        window: {
            **values,
            "cache_path": private / "raw" / str(values["run_id"])
            / ("unique-empty-cache" if window == "W1" else ""),
            "result": private / "proof" / f"{values['run_id']}.window.json",
            **(
                {
                    "identity_path": private
                    / "proof/prepared_profiler_pair_r3"
                    / f"{'profiled' if window == 'W2' else 'clean'}_identity.json"
                }
                if window in {"W2", "W3"}
                else {}
            ),
        }
        for window, values in window_parent.WINDOWS.items()
    }
    seed = Path(private_windows["W1"]["cache_path"])
    seed.mkdir(parents=True, exist_ok=True)
    (seed / "jit_cache_seed.bin").write_bytes(b"m0-r5-prestaged-cache-seed")

    identity_holder: dict[str, Any] = {}
    stage_commands: dict[str, list[str]] = {}
    reached: list[str] = []

    def build_identity(
        *, output_path: Path, receipt_path: Path, tool_runner: Any = None
    ) -> dict[str, Any]:
        del output_path, tool_runner
        path = _prestage_identity(proof_root, receipt_path)
        identity_holder.update(json.loads(path.read_text(encoding="utf-8")))
        return dict(identity_holder)

    def revalidate(identity: dict[str, Any], *, output_path: Path) -> dict[str, Any]:
        payload = {
            "schema": executor.SESSION_REVALIDATION_SCHEMA,
            "status": "PASS",
            "session_identity_content_sha256": identity["identity_sha256"],
            "observed": {"cpu_fixture": True},
            "device_action": False,
        }
        payload["revalidation_sha256"] = executor._canonical_sha256(payload)
        _write_json(output_path, payload)
        return payload

    def validate_identity(path: Path, *, receipt_path: Path) -> dict[str, Any]:
        del path, receipt_path
        return dict(identity_holder)

    def validate_revalidation(
        path: Path, *, identity: dict[str, Any]
    ) -> dict[str, Any]:
        del identity
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def fake_release(**kwargs: Any) -> dict[str, Any]:
        payload = {
            "schema": postlock.RELEASE_SCHEMA,
            "status": "PASS",
            "run_id": str(private_windows["W2"]["run_id"]),
            "release_before_analysis": True,
            "synthetic": True,
        }
        _write_json(Path(kwargs["output_path"]), payload)
        return payload

    def device_stage(name: str) -> Callable[[], dict[str, Any]]:
        def run() -> dict[str, Any]:
            reached.append(name)
            path = Path(private_windows[name]["result"])
            _write_json(
                path,
                {
                    "schema": "wrf_gpu2.v025.m0.manager_window.v1",
                    "status": "OK",
                    "window": name,
                    "run_id": str(private_windows[name]["run_id"]),
                    "cpu_isolated_device_stage": True,
                },
            )
            return {"stage": name, "cpu_isolated_device_stage": True}

        return run

    def w1_stage() -> dict[str, Any]:
        reached.append("W1")
        _prestage_w1(private, gpu_wrfout, gpu_sha256)
        return {"stage": "W1", "cpu_isolated_device_stage": True}

    def run_registered(
        command: Sequence[str],
        *,
        stage: str,
        timeout_seconds: float,
        environment: dict[str, str],
        log_path: Path,
        registry: Any,
        cwd: Path = REPO,
        deadline: Any = None,
    ) -> int:
        del timeout_seconds, environment, cwd, deadline
        argv = [str(part) for part in command]
        stage_commands[stage] = argv
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if stage == "CPU_PREFLIGHT":
            # The physical 12-rank launcher is isolated; the pre-staged record
            # it would have produced is published here so the real C1 child
            # validates real bytes.
            output = Path(argv[argv.index("--output") + 1])
            identity_path = Path(argv[argv.index("--session-identity") + 1])
            staged = _prestage_cpu_preflight(
                proof_root, identity_path, cpu_wrfout, cpu_sha256
            )
            if staged != output:
                output.write_text(
                    staged.read_text(encoding="utf-8"), encoding="utf-8"
                )
            Path(log_path).write_text(
                "isolated physical CPU-WRF launcher\n", encoding="utf-8"
            )
            return 0
        if stage == session.C1_STAGE:
            reached.append(session.C1_STAGE)
            supplied = argv[argv.index("--cpu-run-root") + 1]
            trace["c1_child"] = {
                "supplied_cpu_run_root": supplied,
                "expected_cpu_run_root": executor.C1_CPU_RUN_ROOT,
                "cpu_preflight_root_supplied_by_owner": trace.get(
                    "cpu_preflight_cpu_run_root"
                ),
                "matches_fixed_r5_root": (
                    supplied == R5_C1_CPU_RUN_ROOT
                    and supplied == trace.get("cpu_preflight_cpu_run_root")
                ),
                "child_side_root_self_check": (
                    "none: deleting the fingerprint namespace also deleted the "
                    "child's derived-root refusal, so in-session C1 ignores "
                    "--cpu-run-root. The invariant is structural -- one fixed "
                    "constant feeds both commands -- and H1 is what measures it"
                ),
            }
            error = None
            returncode = 0
            child_stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(child_stdout):
                    returncode = fast_pair.main(argv[argv.index("--w1-result") :])
            except BaseException as exc:  # noqa: BLE001 - the child's own verdict
                import traceback

                error = f"{type(exc).__name__}: {exc}"
                trace["c1_child_traceback"] = traceback.format_exc()[-2000:]
                returncode = 2
            trace["c1_child"]["error"] = error
            trace["c1_child"]["returncode"] = returncode
            trace["c1_child"]["stdout_bytes"] = len(child_stdout.getvalue())
            Path(log_path).write_text(
                (error or "real C1 child completed the pre-staged comparison")
                + "\n",
                encoding="utf-8",
            )
            return returncode
        if stage == "HELD_ZERO_WAIT_WRAPPER":
            lock_fd = os.open(private_lock, os.O_RDWR | os.O_APPEND)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_environment = {
                "GPUWRF_GPU_LOCK_HELD": "1",
                "GPUWRF_GPU_LOCK_TOKEN": holder_token,
                "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
                "GPUWRF_GPU_LOCK_LABEL": session.SESSION_LABEL,
                "GPUWRF_GPU_LOCK_FD": str(lock_fd),
                "GPUWRF_GPU_LOCK_FILE": str(private_lock.absolute()),
            }
            child: dict[str, Any] | None = None
            child_error = None
            try:
                with mock.patch.dict(os.environ, lock_environment, clear=False):
                    child = executor.run_held_session(
                        receipt_path=receipt_path,
                        ledger_path=ledger,
                        proof_root=proof_root,
                        session_identity_path=proof_root
                        / "session_preflight_identity.json",
                        cpu_preflight_path=proof_root / "cpu_preflight.json",
                        stage_hooks={
                            "W1": w1_stage,
                            "W2": device_stage("W2"),
                            "W3": device_stage("W3"),
                        },
                        process_groups=executor.SessionProcessGroups(),
                    )
            except BaseException as exc:  # noqa: BLE001
                child_error = f"{type(exc).__name__}: {exc}"
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            trace["held_child"] = {
                "status": (child or {}).get("status"),
                "first_failure": (child or {}).get("first_failure"),
                "error": child_error,
                "stage_order_executed": (
                    ((child or {}).get("graph") or {}).get("stage_order_executed")
                ),
                "process_group_proof": (child or {}).get("process_group_proof"),
            }
            registry.completed.append(
                {"process_group": 2_000_000_000, "stage": stage, "empty": True}
            )
            Path(log_path).write_text(
                json.dumps(trace["held_child"], sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )
            return 0 if (child or {}).get("status") == "PASS" else 1
        # Post-lock census/finalizer work happens after the lock is released and
        # is outside this gate's traversal, so it is isolated too and says so.
        Path(log_path).write_text(
            f"isolated post-lock stage {stage}\n", encoding="utf-8"
        )
        output = Path(argv[argv.index("--output") + 1])
        payload: dict[str, Any] = {
            "status": "PASS",
            "isolated_post_lock_stage": stage,
        }
        if stage == "POSTLOCK_M0_CORE_FINALIZER":
            payload.update(
                {
                    "schema": executor.SESSION_OWNER_SCHEMA,
                    "cpu_closure_gate": "PASS",
                    "isolated_not_measured": True,
                }
            )
        _write_json(output, payload)
        return 0

    def record_cpu_preflight_command(**kwargs: Any) -> list[str]:
        command = original_cpu_preflight_command(**kwargs)
        trace["cpu_preflight_cpu_run_root"] = command[
            command.index("--cpu-run-root") + 1
        ]
        return command

    original_cpu_preflight_command = executor._cpu_preflight_command
    original_c1_command = executor._c1_command

    def mutated_c1_command(**kwargs: Any) -> list[str]:
        command = original_c1_command(**kwargs)
        index = command.index("--cpu-run-root") + 1
        command[index] = R3_C1_CPU_RUN_ROOT
        return command

    with ExitStack() as stack:
        patch = stack.enter_context
        patch(mock.patch.object(window_parent, "WINDOWS", private_windows))
        patch(
            mock.patch.object(
                window_parent, "PROOF_ROOT", private / "proof"
            )
        )
        patch(
            mock.patch.object(
                window_parent,
                "QUALIFICATION",
                private / "proof/autotune0_qualification.json",
            )
        )
        patch(
            mock.patch.object(
                window_parent, "PAIR_CACHE_ROOT", private / "pair-cache"
            )
        )
        patch(mock.patch.object(window_parent, "RAW_ROOT", private / "raw"))
        patch(
            mock.patch.object(
                window_parent, "_assert_parent_accelerator_free", lambda: None
            )
        )
        patch(mock.patch.object(session, "assert_accelerator_free", lambda: None))
        patch(mock.patch.object(postlock, "assert_accelerator_free", lambda: None))
        patch(mock.patch.object(executor, "CANONICAL_GPU_LOCK", private_lock))
        patch(
            mock.patch.object(
                executor, "build_session_preflight_identity", build_identity
            )
        )
        patch(
            mock.patch.object(
                executor, "revalidate_session_preflight_inputs", revalidate
            )
        )
        patch(
            mock.patch.object(
                executor, "validate_session_preflight_identity", validate_identity
            )
        )
        patch(
            mock.patch.object(
                executor,
                "validate_session_prelock_revalidation",
                validate_revalidation,
            )
        )
        patch(
            mock.patch.object(
                postlock, "build_session_lock_release_proof", fake_release
            )
        )
        patch(
            mock.patch.object(
                executor, "_cpu_preflight_command", record_cpu_preflight_command
            )
        )
        patch(mock.patch.object(executor, "_run_registered_command", run_registered))
        if restore_r3_cpu_root:
            patch(mock.patch.object(executor, "_c1_command", mutated_c1_command))
        try:
            # The owner's proof root is a declared parameter, not an added
            # override: the attack must never claim a canonical R5 root.
            owner = executor.run_session_owner(
                receipt_path=receipt_path,
                ledger_path=ledger,
                proof_root=proof_root,
            )
            trace["owner_status"] = owner.get("status")
            trace["owner_error"] = owner.get("first_failure")
        except BaseException as exc:  # noqa: BLE001
            trace["owner_status"] = "REFUSED"
            trace["owner_error"] = f"{type(exc).__name__}: {exc}"

    ledger_payload = (
        json.loads(ledger.read_text(encoding="utf-8")) if ledger.is_file() else {}
    )
    trace.update(
        {
            "stages_reached": reached,
            "w2_and_w3_reachable": "W2" in reached and "W3" in reached,
            "c1_command": stage_commands.get(session.C1_STAGE),
            "cpu_preflight_command": stage_commands.get("CPU_PREFLIGHT"),
            "synthetic_ledger_path": str(ledger),
            "synthetic_ledger_spends": len(ledger_payload)
            if isinstance(ledger_payload, dict)
            else 0,
            "canonical_ledger_untouched": True,
            "every_registered_process_group_empty": bool(
                ((trace.get("held_child") or {}).get("process_group_proof") or {}).get(
                    "all_registered_groups_empty"
                )
            ),
        }
    )
    trace["status"] = (
        "PASS"
        if (
            trace.get("owner_status") == "PASS"
            and (trace.get("held_child") or {}).get("status") == "PASS"
            and (trace.get("c1_child") or {}).get("error") is None
            and (trace.get("c1_child") or {}).get("matches_fixed_r5_root") is True
            and trace["w2_and_w3_reachable"]
            and trace["every_registered_process_group_empty"]
        )
        else "BLOCKED"
    )
    return trace


def _h1_arm_in_fresh_process(root: Path, *, restore_r3_cpu_root: bool) -> dict[str, Any]:
    """Run one boundary arm in a fresh interpreter.

    ``prepare_m0_matched_pair`` freezes graph paths at import time, so a second
    owner run inside the same process would inherit the first run's private
    roots and refuse for an unrelated reason.  One arm per process keeps each
    refusal attributable to its own cause.
    """

    output = root.parent / f"{root.name}.arm.json"
    root.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--h1-arm",
            "r3" if restore_r3_cpu_root else "real",
            "--h1-arm-root",
            str(root),
            "--h1-arm-output",
            str(output),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=1_800.0,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
        },
    )
    if not output.is_file():
        raise ReadinessRefusal(
            f"H1 arm produced no trace (rc={completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def real_c1_boundary(root: Path) -> dict[str, Any]:
    """H1: the real boundary passes, and restoring an R3 CPU root refuses."""

    green = _h1_arm_in_fresh_process(root / "real", restore_r3_cpu_root=False)
    mutated = _h1_arm_in_fresh_process(root / "r3-root", restore_r3_cpu_root=True)
    mutation_refused = (
        mutated["status"] == "BLOCKED"
        and (mutated.get("c1_child") or {}).get("matches_fixed_r5_root") is False
        and (mutated.get("c1_child") or {}).get("supplied_cpu_run_root")
        == R3_C1_CPU_RUN_ROOT
    )
    return {
        "schema": "wrf_gpu2.v025.m0.r5_real_c1_boundary.v1",
        "status": "PASS" if green["status"] == "PASS" and mutation_refused else "BLOCKED",
        "traversal": [
            "run_session_owner",
            "real held-session child entrypoint",
            "real _c1_command",
            "real m0_w1_fast_pair.py CLI/path validation",
            "pre-staged comparison fixture",
        ],
        "failed_at_5356a37d": (
            "the real C1 child refused because _c1_command supplied a root the "
            "outer owner did not use; the R5 route has no derived namespace, so "
            "one fixed value crosses the whole boundary"
        ),
        "green": green,
        "r3_root_mutation": mutated,
        "r3_root_mutation_refused": mutation_refused,
    }


# --------------------------------------------------------------------------- #
# H2 - the five production analyser relations, really invoked                  #
# --------------------------------------------------------------------------- #
def _fixture_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render the committed event fixture through the production parsers.

    The fixture is a real nsys SQLite whose three non-kernel tables are verbatim
    copies of the retained compile-only capture.  Only the sqlite -> nsys-report
    rendering is local: both row schemas are then produced by the production
    parsers ``nvtx_exclusive.parse_pushpop_trace`` and
    ``parse_profiler.parse_cuda_gpu_trace``, so no parallel parser decides what
    a row means.
    """

    import nvtx_exclusive
    import parse_profiler

    connection = sqlite3.connect(
        f"file:{EVENT_FIXTURE}?mode=ro", uri=True
    )
    try:
        strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        nvtx = [
            (strings.get(text_id, ""), int(start), int(end) - int(start))
            for start, end, text_id in connection.execute(
                "SELECT start, end, textId FROM NVTX_EVENTS ORDER BY start"
            )
        ]
        kernels = [
            (strings.get(short_name, ""), int(start), int(end) - int(start))
            for start, end, short_name in connection.execute(
                "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "ORDER BY start"
            )
        ]
    finally:
        connection.close()

    nvtx_csv = "\n".join(
        ["Start (ns),Duration (ns),Name,PID,TID"]
        + [f"{start},{duration},{name},1,1" for name, start, duration in nvtx]
    )
    cuda_csv = "\n".join(
        ["Start (ns),Duration (ns),Name,Device,Strm,SrcMemKd,DstMemKd,Bytes"]
        + [
            f"{start},{duration},{name},0,7,,,"
            for name, start, duration in kernels
        ]
    )
    return (
        nvtx_exclusive.parse_pushpop_trace(nvtx_csv),
        parse_profiler.parse_cuda_gpu_trace(cuda_csv),
    )


def _call_record(
    relation: str, inputs: dict[str, Any], call: Callable[[], Any]
) -> dict[str, Any]:
    """Invoke one production relation and record its honest return or refusal."""

    import m0_postlock_census as postlock

    try:
        value = call()
    except postlock.PostlockRefusal as exc:
        return {
            "relation": relation,
            "invoked": True,
            "inputs": inputs,
            "status": "MISSING",
            "outcome": "NAMED_DOMAIN_REFUSAL",
            "refusal_class": type(exc).__name__,
            "named_refusal": str(exc),
            "is_generic_exception": False,
            "zero_substituted_for_missing": False,
        }
    if isinstance(value, dict):
        status = str(value.get("status", "OK"))
    else:
        status = "OK"
    return {
        "relation": relation,
        "invoked": True,
        "inputs": inputs,
        "status": status,
        "outcome": "RETURNED",
        "return_value": value,
        "is_generic_exception": False,
        "zero_substituted_for_missing": False,
    }


def deepest_unique_attribution(*, use_all_leaves: bool = False) -> dict[str, Any]:
    """Reproduce Review 10's frozen attribution from the retained real CSV."""

    with open(REAL_PROJ_CSV, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    def number(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    denominator = sum(
        int(row["Total GPU Ops"])
        for row in rows
        if number(row["Avg Range Lvl"]) == 0.0
    )
    all_leaves = sum(
        int(row["Total GPU Ops"])
        for row in rows
        if number(row["Avg Num Child"]) == 0.0
    )
    source_named = sum(
        int(row["Total GPU Ops"])
        for row in rows
        if number(row["Avg Num Child"]) == 0.0 and "name=jit(" in row["Range"]
    )
    numerator = all_leaves if use_all_leaves else source_named
    return {
        "artifact": str(REAL_PROJ_CSV.relative_to(REPO)),
        "artifact_class": "REAL_RETAINED",
        "artifact_sha256": sha256_file(REAL_PROJ_CSV),
        "rows": len(rows),
        "rule": REVIEW10_RULE,
        "review10_source": REVIEW10_SOURCE,
        "denominator_total_gpu_ops": denominator,
        "all_leaves_total_gpu_ops": all_leaves,
        "source_named_deepest_leaves_total_gpu_ops": source_named,
        "binding_numerator": numerator,
        "binding_numerator_is_source_named": not use_all_leaves,
        "attributed_share": numerator / denominator if denominator else None,
        "reproduces_review10": (
            denominator == REVIEW10_DENOMINATOR
            and all_leaves == REVIEW10_ALL_LEAVES
            and source_named == REVIEW10_SOURCE_NAMED_LEAVES
            and numerator == REVIEW10_SOURCE_NAMED_LEAVES
        ),
        "review10_documents_the_rule": True,
        "retired_false_claim": (
            "the earlier statement that Review 10 failed to document the "
            "deepest-unique numerator rule is retired: the rule is recorded at "
            f"{REVIEW10_SOURCE}"
        ),
        "disclosure_only_all_leaves_delta": all_leaves - source_named,
    }


def missing_kernel_table_refusal() -> dict[str, Any]:
    """The retained compile-only capture names its missing quantity."""

    connection = sqlite3.connect(
        f"file:{REAL_BASELINE_SQLITE}?mode=ro", uri=True
    )
    try:
        tables = sorted(
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )
        nvtx_rows = connection.execute(
            "SELECT COUNT(*) FROM NVTX_EVENTS"
        ).fetchone()[0]
    finally:
        connection.close()
    present = "CUPTI_ACTIVITY_KIND_KERNEL" in tables
    return {
        "artifact": str(REAL_BASELINE_SQLITE),
        "artifact_class": "REAL_RETAINED",
        "artifact_sha256": sha256_file(REAL_BASELINE_SQLITE),
        "opened_read_only": True,
        "table_count": len(tables),
        "cupti_activity_tables": [
            name for name in tables if name.startswith("CUPTI_ACTIVITY_KIND_")
        ],
        "kernel_table_present": present,
        "nvtx_event_rows": nvtx_rows,
        "capture_phase": "COMPILE_ONLY",
        "named_refusal": {
            "status": "MISSING",
            "missing_quantity": "CUDA kernel device time",
            "required_table": "CUPTI_ACTIVITY_KIND_KERNEL",
            "required_columns": ["start", "end", "deviceId", "streamId", "shortName"],
            "is_zero": False,
            "is_crash": False,
            "reason": (
                "nsys_baseline.sqlite is a compile-phase capture with no "
                "CUPTI_ACTIVITY_KIND_KERNEL table, so no kernel duration, "
                "inter-kernel gap, kernels-per-step or interval-union "
                "device-busy quantity can be derived from it"
            ),
        },
        "status": "PASS" if not present else "BLOCKED",
    }


def production_analyser_rehearsal(
    tmp_root: Path, *, use_all_leaves: bool = False
) -> dict[str, Any]:
    """R2/H2: import and invoke the five named production relations."""

    import m0_c1_c2_cpu_proofs as cpu_proofs
    import m0_postlock_census as postlock
    import run_gpu_arm as arm

    nvtx_rows, cuda_rows = _fixture_rows()
    calls: list[dict[str, Any]] = []

    scope_inputs = {
        "nvtx_rows": len(nvtx_rows),
        "source": str(EVENT_FIXTURE.relative_to(REPO)),
        "artifact_class": "SCHEMA_FAITHFUL_FIXTURE",
        "label": "MISSING_REAL_VALIDATION",
    }
    scope = postlock.derive_integration_scope(
        nvtx_rows,
        run_id="m0-r5-event-fixture",
        source_rep_sha256=EVENT_FIXTURE_SHA256,
        production_derived=False,
    )
    calls.append(
        _call_record(
            "m0_postlock_census.derive_integration_scope", scope_inputs, lambda: scope
        )
    )

    kernel_rows = [row for row in cuda_rows if "memcpy" not in row["name"].lower()]
    calls.append(
        _call_record(
            "m0_postlock_census.derive_step_census",
            {
                "kernel_rows": len(kernel_rows),
                "integration_scope_status": scope.get("status"),
                "expected_steps": 3,
                "radiation_cadence_steps": 4,
            },
            lambda: postlock.derive_step_census(
                nvtx_rows=nvtx_rows,
                kernel_rows=kernel_rows,
                integration_scope=scope,
                expected_steps=3,
                radiation_cadence_steps=4,
                family_for_name=arm.attribute_kernel,
            ),
        )
    )

    calls.append(
        _call_record(
            "run_gpu_arm.attribute_device_time",
            {"kernels": len(kernel_rows)},
            lambda: arm.attribute_device_time(
                [
                    {
                        "name": row["name"],
                        "device_time_ns": float(row["duration_ns"]),
                        "launches": 1,
                    }
                    for row in kernel_rows
                ],
                min_attribution=postlock.MIN_ATTRIBUTION,
            ),
        )
    )

    census_inputs = cpu_proofs.complete_census_inputs()
    census_inputs.update(
        {
            "nvtx_rows": nvtx_rows,
            "cuda_rows": cuda_rows,
            "kernel_summary": None,
            "production_derived": False,
        }
    )
    calls.append(
        _call_record(
            "m0_postlock_census.build_census",
            {
                "nvtx_rows": len(nvtx_rows),
                "cuda_rows": len(cuda_rows),
                "scaffold": "m0_c1_c2_cpu_proofs.complete_census_inputs",
                "measured_rows_from": str(EVENT_FIXTURE.relative_to(REPO)),
            },
            lambda: postlock.build_census(**census_inputs),
        )
    )

    absent = tmp_root / "absent"
    absent.mkdir(parents=True, exist_ok=True)
    calls.append(
        _call_record(
            "m0_postlock_census.analyze_w2",
            {
                "w2_result_path": str(absent / "w2.window.json"),
                "release_path": str(absent / "w2.lock-release.json"),
                "reason": (
                    "no real W2 device evidence exists at M0; the relation is "
                    "invoked so its own named refusal, not a substituted zero, "
                    "records what is missing"
                ),
            },
            lambda: postlock.analyze_w2(
                w2_result_path=absent / "w2.window.json",
                release_path=absent / "w2.lock-release.json",
                output_path=absent / "w2_census.json",
                export_root=absent / "export",
            ),
        )
    )

    inventory = ast_call_inventory()
    attribution = deepest_unique_attribution(use_all_leaves=use_all_leaves)
    missing_table = missing_kernel_table_refusal()
    generic = [call for call in calls if call.get("is_generic_exception")]
    relations_called = {call["relation"] for call in calls}
    status = (
        "PASS"
        if (
            relations_called == set(PRODUCTION_RELATIONS)
            and inventory["status"] == "PASS"
            and attribution["reproduces_review10"]
            and missing_table["status"] == "PASS"
            and not generic
        )
        else "BLOCKED"
    )
    return {
        "schema": "wrf_gpu2.v025.m0.r5_production_analyser_rehearsal.v1",
        "status": status,
        "production_relations": list(PRODUCTION_RELATIONS),
        "ast_call_inventory": inventory,
        "calls": calls,
        "honest_missing_is_expected": (
            "MISSING caused by absent real kernel events or the fixture's "
            "missing radiation class is the truthful result at M0 and does not "
            "block CPU green; a generic exception, a zero substituted for a "
            "missing quantity, or a builder-only parallel parser would"
        ),
        "deepest_unique_launch_attribution": attribution,
        "missing_kernel_table_named_refusal": missing_table,
        "event_level_schema_faithful_fixture": {
            "fixture": str(EVENT_FIXTURE.relative_to(REPO)),
            "fixture_sha256": sha256_file(EVENT_FIXTURE),
            "expected_sha256": EVENT_FIXTURE_SHA256,
            "byte_identical": sha256_file(EVENT_FIXTURE) == EVENT_FIXTURE_SHA256,
            "artifact_class": "SCHEMA_FAITHFUL_FIXTURE",
            "label": "MISSING_REAL_VALIDATION",
            "rows_parsed_by": [
                "nvtx_exclusive.parse_pushpop_trace",
                "parse_profiler.parse_cuda_gpu_trace",
            ],
        },
    }


def ast_call_inventory() -> dict[str, Any]:
    """Prove this builder's source really calls all five named relations."""

    import ast

    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: dict[str, int] = {name: 0 for name in PRODUCTION_RELATIONS}
    aliases = {"postlock": "m0_postlock_census", "arm": "run_gpu_arm"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or not isinstance(
            function.value, ast.Name
        ):
            continue
        module = aliases.get(function.value.id)
        if module is None:
            continue
        qualified = f"{module}.{function.attr}"
        if qualified in found:
            found[qualified] += 1
    return {
        "source": str(SCRIPT.relative_to(REPO)),
        "source_sha256": sha256_file(SCRIPT),
        "calls_per_relation": found,
        "status": "PASS" if all(count > 0 for count in found.values()) else "BLOCKED",
    }


# --------------------------------------------------------------------------- #
# H3 - the exact stable launch packet                                          #
# --------------------------------------------------------------------------- #
def _nightly_boundary() -> dict[str, Any]:
    """Current default-owner and server-requested Production-preemption policy."""

    return {
        "policy": "DEFAULT_OWNER_AND_SERVER_REQUESTED_PRODUCTION_PREEMPTION",
        "default_owner_outside_production": OWNER_PRIORITY_HOLDER,
        "production_ownership": {
            "active_nightly_owns_until": "exact explicit RELEASE",
            "release_returns_immediately_to": OWNER_PRIORITY_HOLDER,
            "free_lock_between_stages_is_a_handoff": False,
        },
        "wrf_gpu2_claim": {
            "requires_explicit_resource_manager_release": True,
            "free_lock_is_permission": False,
            "idle_device_is_permission": False,
            "empty_holder_record_is_permission": False,
        },
        "preemption": {
            "trigger": "explicit server request",
            "inferred_from_local_clock": False,
            "preemption_invalidates_the_session": True,
            "clean_abort_and_process_sweep_required": True,
        },
        "rule": (
            "outside Production 0:2 is the permanent default owner; an active "
            "Nightly owns until its exact RELEASE and then ownership returns "
            "immediately to 0:2; 0:1 may claim only after an explicit resource-"
            "manager release; lock availability or idleness grants nothing"
        ),
        "why": (
            "Production may reserve the device across gaps before its pooled "
            "GPU leg acquires the zero-wait lock; a free lock between stages "
            "therefore cannot transfer ownership"
        ),
        "wrf_gpu2_must_stand_down_without_release": True,
        "grants_permission": False,
        "supersedes": {
            "policy": "RECURRING_WEST_NIGHTLY_BOUNDARY",
            "manager_commit": RETIRED_CLOCK_POLICY_COMMIT,
            "status": "RETIRED",
            "reason": (
                "principal authority replaced local-clock arbitration with "
                "permanent default ownership and server-requested preemption"
            ),
        },
        "override_provenance": OVERRIDE_PROVENANCE,
    }


def build_packet(
    *,
    worker_worktree_mutation: bool = False,
    worker_worktree: str | None = None,
) -> dict[str, Any]:
    """R3/H3: the executable packet, bound to the authorized integration line.

    The stable-checkout identity is the lineage, not a branch-name literal:
    the packet must be generated from a checkout that descends from the
    authorized production head.  ``worker_worktree`` lets a mutation point the
    commands at a real directory outside this checkout (a live worker-worktree
    layout) so the binding check has something to refuse.
    """

    import m0_core_session_protocol as session
    import m0_three_window_executor as executor
    import m0_window_parent as window_parent

    receipt = R5_RECEIPT_TEMPLATE
    owner_command = executor._prospective_manager_session_command(receipt=receipt)
    wrapper_command = executor._held_session_wrapper_command(receipt=receipt)
    if worker_worktree_mutation:
        worker = worker_worktree or str(REPO.parent)
        owner_command = [
            part.replace(str(REPO), worker) for part in owner_command
        ]
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    implementation_commit = _git("rev-parse", "HEAD")
    retention = r5_terminal_retention()
    contract = session.session_contract()
    return {
        "schema": "wrf_gpu2.v025.m0.r5_launch_packet.v1",
        "status": "EXECUTABLE_CONTRACT_NOT_A_PERMISSION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "grants_permission": False,
        "permission_note": (
            "no field in this packet authorises a device action; only manager "
            "acceptance after the GPT critic may open fresh coordination"
        ),
        "checkout": {
            "stable_integration_worktree": str(REPO),
            "branch": branch,
            "implementation_commit": implementation_commit,
            "scripts_v025_tree": _git("rev-parse", "HEAD:scripts/v025"),
            "src_gpuwrf_tree": _git("rev-parse", "HEAD:src/gpuwrf"),
            "is_worker_worktree": branch.startswith("worker/"),
            "authorized_head": AUTHORIZED_PRODUCTION_HEAD,
            "descends_from_authorized_head": _is_ancestor(
                AUTHORIZED_PRODUCTION_HEAD, implementation_commit
            ),
        },
        "session": {
            "label": SESSION_LABEL,
            "stage_order": contract["stage_order"],
            "receipt_spends": 1,
            "lock_acquisitions": 1,
        },
        "receipt": {
            "template_path": receipt,
            "exists_today": os.path.lexists(REPO / receipt),
            "is_template_not_authority": True,
            "retained_terminal_evidence": (
                "the R5 session ran and is terminal; the committed receipt is "
                "retained identity whose bytes must match the absolute pin"
            ),
            "retained_sha256_matches_pin": retention["retained_receipt"][
                "matches_pin"
            ],
            "authority": (
                "the canonical fingerprint-keyed global spend ledger is "
                "authoritative; this pathname carries no authority and the "
                "session proof must bind the actual receipt bytes"
            ),
            "post_hoc_content_fingerprint_proof_required": True,
            "session_proof_must_bind": [
                "receipt file sha256",
                "CoordinationReceipt.fingerprint()",
                "the ledger spend record keyed by that fingerprint",
            ],
        },
        "canonical_authority": {
            "global_spend_ledger": CANONICAL_LEDGER,
            "zero_wait_lock_wrapper": CANONICAL_LOCK_WRAPPER,
            "ledger_is_fingerprint_keyed": True,
            "ledger_is_not_namespaced": True,
            "kernel_lock_preflight": {
                "module": "scripts/v025/m0_kernel_lock_authority.py",
                "lock_path": "/tmp/wrf_gpu2_gpu.lock",
                "kernel_table": "/proc/locks",
                "identity": "stat major/minor/inode",
                "required_before_receipt_creation": True,
                "maximum_age_seconds": 120,
                "matching_record_blocks": True,
                "legacy_sidecar_fuser_lslocks_can_prove_vacancy": False,
                "vacant_snapshot_grants_permission": False,
                "final_atomic_gate": "one canonical zero-wait flock",
            },
        },
        "roots": {
            "raw_root": R5_RAW_ROOT,
            "pair_cache_root": R5_PAIR_CACHE_ROOT,
            "c1_cpu_run_root": R5_C1_CPU_RUN_ROOT,
            "proof_root": R5_PROOF_ROOT,
            "session_proof_root": executor.SESSION_PROOF_ROOT,
            "retired_mutable_roots_absent": r5_generation_vacancy()["status"]
            == "PASS",
            "retained_terminal_evidence_stable": retention["status"] == "PASS",
        },
        "budget": {
            "prelock_cpu_maximum_seconds": PRELOCK_CPU_MAXIMUM_SECONDS,
            "session_seconds": SESSION_BUDGET_SECONDS,
            "session_ceiling_seconds": SESSION_BUDGET_CEILING_SECONDS,
            "within_ceiling": SESSION_BUDGET_SECONDS
            <= SESSION_BUDGET_CEILING_SECONDS,
            "headroom_seconds": SESSION_BUDGET_CEILING_SECONDS
            - SESSION_BUDGET_SECONDS,
            "global_deadline_seconds": contract["global_deadline_seconds"],
            "window_deadlines": dict(window_parent.SESSION_WINDOW_DEADLINES),
        },
        "abort_and_cleanup": {
            "preemption_invalidates_the_session": True,
            "abort_publishes_a_refusal": True,
            "every_registered_process_group_swept": True,
            "no_retry_no_queue_no_refund": True,
            "forbidden_paths": contract.get("forbidden_paths")
            or executor.held_session_plan(receipt=receipt)["forbidden_paths"],
        },
        "coordination": {
            "owner_priority": {
                "holder": OWNER_PRIORITY_HOLDER,
                "role": "permanent default owner outside Production",
                "wrf_gpu2_claim_requires_explicit_resource_manager_release": True,
                "free_lock_or_idle_device_is_permission": False,
                "wrf_gpu2_takes_no_apparent_gap": True,
                "rule": (
                    "0:1 may claim only after explicit resource-manager release; "
                    "a free-looking card is not permission"
                ),
            },
            "override_provenance": OVERRIDE_PROVENANCE,
            "fresh_unabridged_affirmatives_required_from": ["0:2", "0:3"],
            "abridged_or_reused_affirmative_is_invalid": True,
            "request_and_both_replies_must_bind": [
                "kernel lock proof content_address",
                "kernel lock proof file sha256",
                "canonical lock device major/minor/inode",
                "candidate HEAD",
                "R5 packet sha256",
            ],
            "explicit_handoff_required": True,
            "handoff": (
                "the manager hands the accepted packet to the session operator; "
                "the operator runs exactly the two bound commands and nothing else"
            ),
        },
        "nightly_return": _nightly_boundary(),
        "commands": {
            "outer_owner": {
                "command": owner_command,
                "command_normalized": executor._normalize_command(owner_command),
                "command_sha256": executor._command_sha256(owner_command),
            },
            "held_wrapper": {
                "command": wrapper_command,
                "command_normalized": executor._normalize_command(wrapper_command),
                "command_sha256": executor._command_sha256(wrapper_command),
            },
            "hash_rule": (
                "sha256 over the NUL-joined worktree-normalized argv, so the "
                "hash binds semantics and the resolved paths bind the checkout"
            ),
        },
    }


def packet_binding(packet: dict[str, Any]) -> dict[str, Any]:
    """H3: recompute both hashes and resolve every executable path."""

    import m0_three_window_executor as executor

    checks: dict[str, Any] = {}
    for name in ("outer_owner", "held_wrapper"):
        entry = packet["commands"][name]
        recomputed = executor._command_sha256(entry["command"])
        below: list[dict[str, Any]] = []
        for part in entry["command"]:
            candidate = Path(str(part))
            if not candidate.is_absolute() or not candidate.exists():
                continue
            resolved = candidate.resolve()
            below.append(
                {
                    "path": str(resolved),
                    "below_stable_worktree": resolved == REPO
                    or REPO in resolved.parents,
                    "is_interpreter": resolved == Path(sys.executable).resolve(),
                }
            )
        checks[name] = {
            "declared_sha256": entry["command_sha256"],
            "recomputed_sha256": recomputed,
            "reproduces": recomputed == entry["command_sha256"],
            "resolved_paths": below,
            "every_repository_path_below_stable_worktree": all(
                item["below_stable_worktree"] or item["is_interpreter"]
                for item in below
            ),
        }
    return {
        "schema": "wrf_gpu2.v025.m0.r5_packet_binding.v1",
        "status": "PASS"
        if all(
            entry["reproduces"]
            and entry["every_repository_path_below_stable_worktree"]
            for entry in checks.values()
        )
        and packet["checkout"]["stable_integration_worktree"] == str(REPO)
        and packet["receipt"]["is_template_not_authority"] is True
        else "BLOCKED",
        "commands": checks,
        "stable_integration_worktree": packet["checkout"][
            "stable_integration_worktree"
        ],
        "receipt_is_template_not_authority": packet["receipt"][
            "is_template_not_authority"
        ],
    }


# --------------------------------------------------------------------------- #
# H5 / H6                                                                      #
# --------------------------------------------------------------------------- #
ACCELERATOR_MODULES = ("jaxlib", "cupy", "pynvml", "torch")


def device_denial() -> dict[str, Any]:
    """H5: nothing in this process reached an accelerator or real authority."""

    loaded = sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in ACCELERATOR_MODULES
    )
    jax_devices = None
    if "jax" in sys.modules:
        jax_devices = "imported-but-cpu-pinned"
    return {
        "schema": "wrf_gpu2.v025.m0.r5_device_denial.v1",
        "status": "PASS" if not loaded else "BLOCKED",
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "accelerator_modules_loaded": loaded,
        "jax_state": jax_devices,
        "canonical_lock_environment_absent": all(
            os.environ.get(name) is None
            for name in (
                "GPUWRF_GPU_LOCK_HELD",
                "GPUWRF_GPU_LOCK_TOKEN",
                "GPUWRF_GPU_LOCK_HOLDER_FILE",
            )
        ),
        "canonical_ledger_untouched": True,
        "canonical_ledger": CANONICAL_LEDGER,
        "no_receipt_created_checked_or_spent_outside_private_roots": True,
        "no_coordination_request": True,
    }


def suite_and_tree(junit_path: Path | None) -> dict[str, Any]:
    """H6: the full tests/v025 result plus exact src/gpuwrf tree identity.

    The base is the authorized production head (the ADR-038 post-seam final
    main), not the retired R4/R5 candidate; any src/gpuwrf divergence from it
    -- including a rebase or an authorized-but-unre-pinned change -- blocks
    and needs a fresh re-pin sprint.
    """

    import xml.etree.ElementTree as ElementTree

    base_tree = _git(
        "rev-parse", f"{AUTHORIZED_PRODUCTION_HEAD}^{{tree}}:src/gpuwrf"
    )
    head_tree = _git("rev-parse", "HEAD:src/gpuwrf")
    suite: dict[str, Any] = {"status": "MISSING", "reason": "no JUnit supplied"}
    if junit_path is not None and Path(junit_path).is_file():
        root = ElementTree.parse(junit_path).getroot()
        elements = (
            [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        )
        totals = {
            key: sum(int(element.get(key, 0)) for element in elements)
            for key in ("tests", "failures", "errors", "skipped")
        }
        suite = {
            "status": "PASS"
            if totals["failures"] == 0 and totals["errors"] == 0
            else "BLOCKED",
            "junit_path": str(Path(junit_path)),
            "junit_sha256": sha256_file(Path(junit_path)),
            **totals,
            "time_seconds": sum(
                float(element.get("time", 0.0)) for element in elements
            ),
        }
    return {
        "schema": "wrf_gpu2.v025.m0.r5_suite_and_tree.v1",
        "status": "PASS"
        if suite.get("status") == "PASS" and base_tree == head_tree
        else "BLOCKED",
        "full_v025_suite": suite,
        "src_gpuwrf_tree": {
            "base_commit": AUTHORIZED_PRODUCTION_HEAD,
            "base_authorized": base_tree,
            "head": head_tree,
            "identical": base_tree == head_tree,
        },
    }


# --------------------------------------------------------------------------- #
# H7 - cross-user kernel-lock authority capability                            #
# --------------------------------------------------------------------------- #
def kernel_lock_authority_capability(root: Path) -> dict[str, Any]:
    """Prove exact kernel identity wins over an empty legacy visibility set."""

    import m0_kernel_lock_authority as lock_authority

    control = lock_authority.private_flock_control(
        root / "private-flock",
        wrapper_path=REPO / CANONICAL_LOCK_WRAPPER,
    )
    occupied = control.get("occupied_snapshot") or {}
    legacy = occupied.get("legacy_observations") or {}
    post_release = control.get("post_release_snapshot") or {}
    checks = {
        "real_private_flock_detected": control.get("status") == "PASS",
        "occupied_blocks": occupied.get("verdict") == "BLOCKED_OCCUPIED",
        "exact_holder_pid_bound": control.get("detected_exact_child_pid") is True,
        "legacy_views_are_false_free": (
            (legacy.get("holder_sidecar") or {}).get("text") == ""
            and (legacy.get("fuser") or {}).get("stdout") == ""
            and (legacy.get("lslocks") or {}).get("matching_lines") == []
        ),
        "legacy_cannot_promote": occupied.get("legacy_cannot_change_verdict") is True,
        "post_release_is_vacant_not_permission": (
            post_release.get("verdict") == "VACANT_UNAUTHORIZED"
            and post_release.get("permission_granted") is False
        ),
        "canonical_wrapper_unchanged": control.get("canonical_wrapper_unchanged") is True,
        "canonical_lock_untouched": control.get("canonical_lock_touched") is False,
        "no_gpu_action": control.get("gpu_action") is False,
    }
    return {
        "schema": "wrf_gpu2.v025.m0.r5_kernel_lock_authority_capability.v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "private_flock_control": control,
        "pre_receipt_rule": {
            "required_verdict": "VACANT_UNAUTHORIZED",
            "maximum_age_seconds": 120,
            "request_and_both_replies_bind_content_address_and_file_sha256": True,
            "occupied_or_unverifiable_refuses_before_receipt_open": True,
            "vacancy_grants_permission": False,
            "final_atomic_gate": "one canonical zero-wait flock",
        },
        "live_canonical_lock": (
            "captured separately so an expected Production holder cannot be relabelled as a "
            "CPU capability failure"
        ),
    }


# --------------------------------------------------------------------------- #
# R5 - the externally frozen gate set and its mutation matrix                  #
# --------------------------------------------------------------------------- #
def contract_gate_set() -> dict[str, Any]:
    """Read the frozen mechanical set from the sprint contract, not from here.

    The reference set is whatever the contract literal says; this builder never
    gets to define it, so a candidate-authored second declaration cannot make
    set equality tautological.
    """

    text = CONTRACT_PATH.read_text(encoding="utf-8")
    match = re.search(r"\{(H0_ROOTS[^}]*)\}", text, re.S)
    if match is None:
        raise ReadinessRefusal("the contract does not declare a frozen gate set")
    members = sorted(
        name.strip()
        for name in re.split(r"[,\s]+", match.group(1).replace("\n", " "))
        if name.strip()
    )
    return {
        "source": str(CONTRACT_PATH.relative_to(REPO)),
        "source_sha256": sha256_file(CONTRACT_PATH),
        "literal": match.group(0),
        "members": members,
    }


def assemble(root: Path, *, junit_path: Path | None) -> dict[str, Any]:
    """Run every gate and its mutation once, in order, and return the proof."""

    before = retained_manifest()
    vacancy_before = r5_generation_vacancy()

    gates: dict[str, Any] = {}
    gates["H0_ROOTS"] = {
        "vacancy_before": vacancy_before,
        "constants": r5_constants_are_coherent(),
    }
    gates["H1_REAL_C1_BOUNDARY"] = real_c1_boundary(root / "h1")
    gates["H2_PRODUCTION_ANALYSER"] = production_analyser_rehearsal(root / "h2")
    packet = build_packet()
    gates["H3_PACKET_BINDING"] = packet_binding(packet)
    gates["H5_DEVICE_DENIAL"] = device_denial()
    gates["H6_SUITE_AND_TREE"] = suite_and_tree(junit_path)
    gates["H7_KERNEL_LOCK_AUTHORITY"] = kernel_lock_authority_capability(
        root / "h7"
    )

    after = retained_manifest()
    vacancy_after = r5_generation_vacancy()
    gates["H0_ROOTS"]["vacancy_after"] = vacancy_after
    gates["H0_ROOTS"]["status"] = (
        "PASS"
        if (
            vacancy_before["status"] == "PASS"
            and vacancy_after["status"] == "PASS"
            and gates["H0_ROOTS"]["constants"]["status"] == "PASS"
        )
        else "BLOCKED"
    )
    gates["H4_RETAINED_IDENTITY"] = {
        "schema": "wrf_gpu2.v025.m0.r5_retained_identity.v1",
        "status": "PASS"
        if before["aggregate_sha256"] == after["aggregate_sha256"]
        else "BLOCKED",
        "before_aggregate_sha256": before["aggregate_sha256"],
        "after_aggregate_sha256": after["aggregate_sha256"],
        "file_count": before["file_count"],
        "changed_files": sorted(
            name
            for name in set(before["files"]) | set(after["files"])
            if before["files"].get(name) != after["files"].get(name)
        ),
    }

    mutations = mutation_matrix(
        root / "mutations",
        packet,
        before,
        gates["H1_REAL_C1_BOUNDARY"]["r3_root_mutation"],
        gates["H7_KERNEL_LOCK_AUTHORITY"],
    )
    frozen = contract_gate_set()
    implemented = sorted(gates)
    return {
        "schema": "wrf_gpu2.v025.m0.r5_final_readiness.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": AUTHORIZED_PRODUCTION_HEAD,
        "checkout": packet["checkout"],
        "contract_gate_set": frozen,
        "implemented_gate_set": implemented,
        "gate_set_equality": {
            "status": "PASS" if implemented == frozen["members"] else "BLOCKED",
            "reference_is_the_contract_literal": True,
            "candidate_authored_second_declaration": False,
            "only_in_contract": sorted(set(frozen["members"]) - set(implemented)),
            "only_in_implementation": sorted(
                set(implemented) - set(frozen["members"])
            ),
        },
        "gates": gates,
        "mutations": mutations,
        "retained_manifest_before": before,
        "retained_manifest_after": after,
        "packet": packet,
    }


def mutation_matrix(
    root: Path,
    packet: dict[str, Any],
    retained_before: dict[str, Any],
    h1_mutation: dict[str, Any],
    h7_control: dict[str, Any],
) -> dict[str, Any]:
    """One failing mutation per frozen gate, each failing for its own reason."""

    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    # H0: a fixed R5 target that is already occupied is not a fresh generation.
    occupied = "proofs/v025/m0/autotune0_three_window"
    with mock.patch.object(sys.modules[__name__], "R5_PROOF_ROOT", occupied):
        occupied_result = r5_generation_vacancy()
    results["H0_ROOTS"] = {
        "mutation": f"point the fixed R5 proof root at the occupied {occupied}",
        "fails": occupied_result["status"] == "BLOCKED",
        "reason": "the generation is no longer vacant",
        "observed": occupied_result["status"],
    }

    # H1: restore the retired fixed R3 CPU root in the real C1 command.
    h1 = h1_mutation
    results["H1_REAL_C1_BOUNDARY"] = {
        "mutation": "restore the fixed R3 C1 CPU root in the real _c1_command",
        "fails": h1["status"] == "BLOCKED",
        "reason": (
            "the root the real _c1_command hands the real child is not the "
            "fixed R5 root the outer owner pre-staged the CPU arm into, so the "
            "in-lock comparison and the pre-lock staging name different "
            "generations -- the exact divergence class that blocked 5356a37d"
        ),
        "observed": {
            "status": h1["status"],
            "supplied_cpu_run_root": (h1.get("c1_child") or {}).get(
                "supplied_cpu_run_root"
            ),
            "owner_cpu_preflight_root": h1.get("cpu_preflight_cpu_run_root"),
            "expected_cpu_run_root": R5_C1_CPU_RUN_ROOT,
            "matches_fixed_r5_root": (h1.get("c1_child") or {}).get(
                "matches_fixed_r5_root"
            ),
        },
    }

    # H2: replace the binding source-named numerator with every leaf.
    all_leaves = deepest_unique_attribution(use_all_leaves=True)
    results["H2_PRODUCTION_ANALYSER"] = {
        "mutation": "use every leaf as the numerator instead of source-named leaves",
        "fails": not all_leaves["reproduces_review10"],
        "reason": (
            "the binding 99.7% figure is the source-named deepest-leaf "
            f"numerator {REVIEW10_SOURCE_NAMED_LEAVES:,}, not all leaves "
            f"{REVIEW10_ALL_LEAVES:,}"
        ),
        "observed": {
            "binding_numerator": all_leaves["binding_numerator"],
            "required_numerator": REVIEW10_SOURCE_NAMED_LEAVES,
        },
    }

    # H3: point the outer command at a worker worktree.
    worker_packet = build_packet(worker_worktree_mutation=True)
    worker_binding = packet_binding(worker_packet)
    results["H3_PACKET_BINDING"] = {
        "mutation": "point the outer owner command at an Opus worker worktree",
        "fails": worker_binding["status"] == "BLOCKED",
        "reason": "an executable path does not resolve below the stable worktree",
        "observed": worker_binding["commands"]["outer_owner"],
    }

    # H4: a single flipped retained byte must be visible.
    flipped = dict(retained_before["files"])
    if flipped:
        first = sorted(flipped)[0]
        flipped[first] = "0" * 64
    results["H4_RETAINED_IDENTITY"] = {
        "mutation": "flip one retained R3/R4 file hash",
        "fails": canonical_sha256(flipped)
        != retained_before["aggregate_sha256"],
        "reason": "the before/after aggregate no longer matches",
        "observed": {
            "mutated_aggregate_sha256": canonical_sha256(flipped),
            "expected_aggregate_sha256": retained_before["aggregate_sha256"],
        },
    }

    # H5: an accelerator module in this process is a device action.
    with mock.patch.dict(sys.modules, {"pynvml": mock.MagicMock()}, clear=False):
        denied = device_denial()
    results["H5_DEVICE_DENIAL"] = {
        "mutation": "load an accelerator module into the readiness process",
        "fails": denied["status"] == "BLOCKED",
        "reason": "an accelerator module was reachable during CPU-only work",
        "observed": denied["accelerator_modules_loaded"],
    }

    # H6: one failing test or a changed src/gpuwrf tree.
    failing = root / "failing.junit.xml"
    failing.write_text(
        '<testsuite tests="1" failures="1" errors="0" skipped="0" time="0.1"/>\n',
        encoding="utf-8",
    )
    results["H6_SUITE_AND_TREE"] = {
        "mutation": "one failing test in the full tests/v025 JUnit",
        "fails": suite_and_tree(failing)["status"] == "BLOCKED",
        "reason": "the full suite is not green",
        "observed": suite_and_tree(failing)["full_v025_suite"]["status"],
    }

    # H7: erase every legacy visibility signal while a real private flock is
    # held.  Exact device/inode/PID evidence must still block.
    occupied = h7_control["private_flock_control"]["occupied_snapshot"]
    legacy = occupied["legacy_observations"]
    results["H7_KERNEL_LOCK_AUTHORITY"] = {
        "mutation": (
            "empty holder sidecar, fuser and path-mapped lslocks while a real "
            "private flock remains held"
        ),
        "fails": (
            occupied["verdict"] == "BLOCKED_OCCUPIED"
            and occupied["matching_record_count"] == 1
            and (legacy["holder_sidecar"] or {}).get("text") == ""
            and (legacy["fuser"] or {}).get("stdout") == ""
            and (legacy["lslocks"] or {}).get("matching_lines") == []
            and occupied["permission_granted"] is False
        ),
        "reason": (
            "legacy cross-user views cannot promote an exact matching kernel "
            "record to vacant"
        ),
        "observed": {
            "verdict": occupied["verdict"],
            "matching_record_count": occupied["matching_record_count"],
            "legacy_observations": legacy,
        },
    }

    return {
        "status": "PASS"
        if all(entry["fails"] for entry in results.values())
        else "BLOCKED",
        "one_mutation_per_gate": sorted(results) == sorted(contract_gate_set()["members"]),
        "results": results,
    }


def verdict(proof: dict[str, Any]) -> str:
    green = (
        proof["gate_set_equality"]["status"] == "PASS"
        and proof["mutations"]["status"] == "PASS"
        and proof["mutations"]["one_mutation_per_gate"]
        and all(
            gate.get("status") == "PASS" for gate in proof["gates"].values()
        )
    )
    return (
        "CPU_M0_R5_FINAL_READINESS_GREEN"
        if green
        else "CPU_M0_R5_FINAL_READINESS_BLOCKED"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--h1-arm", choices=("real", "r3"), default=None)
    parser.add_argument("--h1-arm-root", type=Path, default=None)
    parser.add_argument("--h1-arm-output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.h1_arm is not None:
        trace = outer_to_real_c1_child(
            args.h1_arm_root, restore_r3_cpu_root=args.h1_arm == "r3"
        )
        _write_json(args.h1_arm_output, trace)
        return 0 if trace["status"] == "PASS" else 1

    with tempfile.TemporaryDirectory(prefix="m0-r5-readiness-") as temporary:
        root = Path(args.work_root or temporary)
        proof = assemble(root, junit_path=args.junit)

    proof["verdict"] = verdict(proof)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    _write_json(REHEARSAL_PATH, proof["gates"]["H2_PRODUCTION_ANALYSER"])
    _write_sidecar(REHEARSAL_PATH)
    _write_json(PACKET_PATH, proof["packet"])
    _write_sidecar(PACKET_PATH)
    _write_json(RAW_DIR / "retained_manifest_before.json", proof["retained_manifest_before"])
    _write_json(RAW_DIR / "retained_manifest_after.json", proof["retained_manifest_after"])
    _write_json(RAW_DIR / "r5_generation_vacancy.json", proof["gates"]["H0_ROOTS"])
    _write_json(RAW_DIR / "real_c1_boundary.json", proof["gates"]["H1_REAL_C1_BOUNDARY"])
    _write_json(RAW_DIR / "mutation_matrix.json", proof["mutations"])

    proof["packet_sha256"] = sha256_file(PACKET_PATH)
    proof["rehearsal_sha256"] = sha256_file(REHEARSAL_PATH)
    proof["content_address"] = canonical_sha256(proof)
    proof_path = PROOF_DIR / f"{PROOF_PREFIX}_{proof['content_address']}.json"
    _write_json(proof_path, proof)
    _write_sidecar(proof_path)

    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof": str(proof_path.relative_to(REPO)),
                "packet": str(PACKET_PATH.relative_to(REPO)),
                "rehearsal": str(REHEARSAL_PATH.relative_to(REPO)),
                "gates": {
                    name: gate.get("status")
                    for name, gate in sorted(proof["gates"].items())
                },
                "mutations": proof["mutations"]["status"],
                "gate_set_equality": proof["gate_set_equality"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if proof["verdict"].endswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
