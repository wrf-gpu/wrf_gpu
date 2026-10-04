#!/usr/bin/env python3
"""CPU-only orchestration for corrected full-buffer pipelined validation.

This module never imports JAX or gpuwrf.  ``audit`` and ``command`` only inspect
or materialize authority artifacts.  ``run`` remains unavailable until later
independent-review and manager commits supply an exact one-use runtime blob.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
from datetime import datetime, timedelta, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset, chartostring


BASE_SHA = "bb2ffe33dbff5e04246dc6903bf6aa26e5827163"
REPAIR_PARENT_SHA = "3e0f184dbcf151a0616ecdc086b9823545852375"
FINAL_REPAIR_PARENT_SHA = "af3ed6181c1924bf649bd60a15a8ccc8e100bc6c"
MONITOR_REPAIR_PARENT_SHA = "cc26affc46d826c2fe59a40c4bb0df56c418e533"
CRITIC_COMMIT = "e84e7ec7dc8dff9ce61b9faf27067c4988914968"
CRITIC_REPORT_SHA256 = "273aeb349c419a658dc19fa56d84a540c00e8d66c7746be74e9f22f1edaa30ae"
FINAL_REREVIEW_COMMIT = "57ed4d31c7b7ea1ed30c255bb1f156304bc98d54"
FINAL_REREVIEW_REPORT_SHA256 = "f832e48d5c3e162b9f3862d158600fd085499cbdbb5be1fe65941bd3d2ab5dbe"
MONITOR_REREVIEW_COMMIT = "278f68218bb22b1f6e98ddc098973494ea8e25cc"
MONITOR_REREVIEW_REPORT_SHA256 = "8f2650bdde14602461491629fed354beab18fa268b1526abd0289071f905b3b8"
CACHE_REREVIEW_COMMIT = "be80785c7d372ee0d94d8323e7b46c5a83de65a9"
CACHE_REREVIEW_REPORT_SHA256 = "7d61920d64ebb5b8b11a12678121a84e4fc9aaab3dd1c115934dba3a15d0196f"
AUTHORITY_V2_REREVIEW_COMMIT = "d46f93462a8836c51fb2de1ad5b31e3518e281cb"
AUTHORITY_V2_REREVIEW_REPORT_SHA256 = "ecf88dffc2fb1c12f5099dd96a14415792eac0db2814bebcf4f1d93a2243efa3"
COMBINED_TERMINAL_REREVIEW_COMMIT = "4284403b4a76ee7ed93a297f8f18a99d5af67b2a"
COMBINED_TERMINAL_REREVIEW_REPORT_SHA256 = "8ff2fdc338e4acd5591a68940c63540e2a2f44ef4095aaf1333be67a41de8b37"
PRELAUNCH_ENV_ORDER_REREVIEW_COMMIT = "3f1d335b96a443f825443066b8198b159ffd312b"
PRELAUNCH_ENV_ORDER_REREVIEW_REPORT_SHA256 = "a0384f22260e4315f7033332ddb1ddd460d8a562ad524322cc1894b8cf1253c0"
PRELAUNCH_ENV_ORDER_SMOKE_SHA256 = "34c515efcf74e72f77aad347a37d24387f323578530e284b1ced16dfdf88b0ee"
CACHE_REPAIR_PARENT_SHA = "bb806475920668ace8793454e96faa7184d6cad5"
PRELAUNCH_ENV_ORDER_PARENT_SHA = "2d89276aff3137853a60a28241b0ffa1325ef681"
PRELAUNCH_SYSTEMD_ENV_PARENT_SHA = "8ac8a695ab62320b3cf2b39a89e0126ecfb87b15"
PRETIMESTEP_SOURCE_AUTHORITY_PARENT_SHA = "27793ff5d50d9d0ecfdef2ca08478e92e5958685"
PRETIMESTEP_SOURCE_AUTHORITY_REPAIR_PARENT_SHA = "4f0d93023c307d12448f830481518e12ac45d571"
PRETIMESTEP_SOURCE_AUTHORITY_REVIEW_COMMIT = "e9777ce9ff548e38d960c7bbe0a05ddfd709f8fb"
PRETIMESTEP_SOURCE_AUTHORITY_REVIEW_SHA256 = "9ca423dd6ff92e1609ceb275a7107efac4fd9d9ccd4783c7c51b852a68d45dee"
PRETIMESTEP_FAILURE_MANAGER_COMMIT = "50fafe7d1f0e3681f95bcf8516a057823809c3f0"
PRETIMESTEP_FAILURE_JSON_SHA256 = "4a4b4a6ebe672c0b0d79eea1e8e5f052da5516e8ce08af9cc6b2b930dcf10201"
PRETIMESTEP_FAILURE_LOG_SHA256 = "9e05b9b285a95cf4c906fe390a8f3c9185f1b80ad4405f394143e5905944cda7"
PRETIMESTEP_FAILURE_JSON_RELATIVE = (
    ".agent/sprints/2026-07-12-v0234-corrected-validation/"
    "pre-timestep-source-materialization-failure.json"
)
PRETIMESTEP_FAILURE_LOG_RELATIVE = (
    ".agent/sprints/2026-07-12-v0234-corrected-validation/"
    "pre-timestep-source-materialization-failure.log"
)
CASE_ID = "20250228_18z"
GRID_ID = "tenerife_operational_v2_fullbuffer_111x93"
GRID_SHA256 = "8ec38f8e1a90d70d390ad533a9c3cf3546aca242f37a93847c7441b925e7c688"
AIFS_SHA256 = "5256342931d4db2853426b05a3e10e67c3fac37e969387041ada6ec83c551fe4"
LAUNCH_PLAN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/launch_plan.json"
)
LAUNCH_PLAN_SHA256 = "782ddbcd55d062e7d3f1f8346e2e83b36ded7de2a9cb0a8651e38235957b667b"
CPU_CASE_ROOT = LAUNCH_PLAN.parent
CPU_RUN_ROOT = CPU_CASE_ROOT / "run"
CANONICAL_V2_CONTRACT = CPU_CASE_ROOT / "canonical_authority_contract_v2.json"
CANONICAL_V2_CONTRACT_SHA256 = "f7f7f634de1ffb583c04d80edfa2cd5e39996c6fd32f7f4cca403d26361c1850"
CANONICAL_BINDING = CPU_CASE_ROOT / "canonical_run_binding.json"
CANONICAL_BINDING_SHA256 = "5b20a3c1843afe93162bf31406fe8422acac6bf3aca1bf722de5cb06330ce075"
FORCING_PREFLIGHT = CPU_CASE_ROOT / "forcing_preflight_v2/manifest.json"
FORCING_PREFLIGHT_SHA256 = "4a9aaa929842c99ec1d9da021b70ec567d1d7c9d8e5357ecc94fc1447295959b"
PRE_WRF_SEAL_SHA256 = "bb4deff4245df75d922c790239d8271b99d352bd1a403e20b89bbf6200b4a73e"
LAUNCH_TIME_CHECKSUMS = CPU_CASE_ROOT / "provenance/checksums_launch_time.sha256"
LAUNCH_TIME_CHECKSUMS_SHA256 = "a278bba06b2a6fdd41f4ae2aa0458dfa19e87529381abd41c70b4b0ca537f7ee"
CURRENT_CHECKSUMS = CPU_CASE_ROOT / "provenance/checksums.sha256"
CURRENT_CHECKSUMS_SHA256 = "1e4dd529aee6e2e7d869ef44f756641e7270a96be115303ab21c237ceae537b0"
CPU_INPUT_DIR = CPU_RUN_ROOT / "wrf"
CPU_MANIFEST = CPU_RUN_ROOT / "cpu_oracle_manifest.json"
CPU_PAIR_INDEX = CPU_RUN_ROOT / "cpu_oracle_pair_index.jsonl"
CPU_THIN_DIR = CPU_RUN_ROOT / "thin"
FROZEN_GEO_PATH = CPU_RUN_ROOT.parent / "static/geo_em.d03.nc"
EARLY_MARKER = CPU_RUN_ROOT / "EARLY_GPU_READY.json"
EARLY_REVOKED = CPU_RUN_ROOT / "EARLY_GPU_READY_REVOKED.json"
EARLY_INVALIDATED = CPU_RUN_ROOT / "EARLY_GPU_READY.invalidated.json"
INPUT_SEAL = CPU_RUN_ROOT / "input_seal.json"
RESOURCE_PROOF = CPU_RUN_ROOT / "EARLY_GPU_READY_RESOURCE_PROOF.json"
TERMINAL_CONTRACT = CPU_CASE_ROOT / "terminal_cpu_authority_contract_v1.json"
TERMINAL_CONTRACT_SHA256 = "26b16e21782400308ec4d58175907796bf8aaddb01062bb06be8b87a2a88848a"
TERMINAL_PROCESS_STATUS = CPU_RUN_ROOT / "terminal_process_status.json"
TERMINAL_PROCESS_STATUS_SHA256 = "c902d696eb2866c1bc5b9e5faf2e4459e4bb80b5b7d0ed76940ab286a85bb221"
TERMINAL_QA_LOG = CPU_RUN_ROOT / "logs/final_qa_corrected_scheduler.log"
TERMINAL_QA_LOG_SHA256 = "ee2a66490c6a82104b53b32c4badfd1a117fa97b484f40ef531e25acb2781192"
TERMINAL_MANIFEST_SHA256 = "c8f087d7d6ca2e0b8d2e959429eb81c4584d3d1f94ef944723774b347ba831ad"
TERMINAL_PAIR_INDEX_SHA256 = "24ab3a22b37901f2a774de21051f8cbec622c3cbc8224ffe45f4de775b997e0d"
TERMINAL_SUPERSESSION = CPU_CASE_ROOT / "canonical_marker_emission_plan_v2.SUPERSEDED_TERMINAL.json"
TERMINAL_SUPERSESSION_SHA256 = "6817a449260b76f4acd27a6b72f2eebabb0ddbe851b23143e9529a7048d3662b"

EARLY_AUTHORITY_REPO = Path("<USER_HOME>/src/wrf_downscale")
EARLY_AUTHORITY_COMMIT = "74b3a8089d95260752da6f2d519f7db77cce4162"
EARLY_AUTHORITY_SOURCE = "scripts/emit_tenerife_fullbuffer_early_gpu_ready.py"
EARLY_AUTHORITY_SOURCE_SHA256 = "ea685861dcde6a9225708e52a9dcacdcf9653ea51af8a418893380a298250a07"
EARLY_AUTHORITY_REPORT_COMMIT = "78a7eed31c9ff1171c897e2af30c26d7bc16f2ce"
EARLY_AUTHORITY_REPORT = EARLY_AUTHORITY_REPO / "reports/tenerife_cpu_oracle_canonical_authority_v2_20260712.md"
EARLY_AUTHORITY_REPORT_SHA256 = "737dd007e396a340c3b93797a1d402ba095f38c8ac614e67ab39584621206111"
CPU_QA_SOURCE = "scripts/verify_tenerife_fullbuffer_cpu_oracle.py"
CPU_QA_SOURCE_SHA256 = "61ce7bca102e877daabdb54f91c2740961ebbacafcff4df91b6446c0998e0f50"
TERMINAL_REPORT = EARLY_AUTHORITY_REPO / "reports/tenerife_terminal_cpu_authority_v1_20260712.md"
TERMINAL_REPORT_COMMIT = "ce4f711758c51ae3b2d89e3f662d0a75410cc811"
TERMINAL_REPORT_SHA256 = "7482d59647c38faadb3a015501972d29e83ff0c0e9f8d6b3ba19aabda40439ca"
TERMINAL_SCHEMA_SOURCE = EARLY_AUTHORITY_REPO / "manifests/terminal_cpu_authority_schema_v1.json"
TERMINAL_SCHEMA_COMMIT = "1271d3d318cb1033e528a675e147749157f72931"
TERMINAL_SCHEMA_SOURCE_SHA256 = "946b1ffe24ff74609acaa852948aea43bec0f3d687570c34f328949470ec0836"
TERMINAL_VALIDATOR_SOURCE = EARLY_AUTHORITY_REPO / "scripts/validate_tenerife_terminal_cpu_authority.py"
TERMINAL_VALIDATOR_SHA256 = "148149fa1861fd040fc064ca1cbb7a88896b757931c8b46829966b41933d9693"
TERMINAL_TEST_SOURCE = EARLY_AUTHORITY_REPO / "tests/test_tenerife_terminal_cpu_authority.py"
TERMINAL_TEST_SOURCE_SHA256 = "b4e35aadfa0684399cec3a7b25f4fc84f9c7ba3bc8359f4f8a8290734342b076"
TERMINAL_VERIFIER_COMMIT = "6fdefdc5f98a355f84294c0c86372a48061734a4"
TERMINAL_VERIFIER_SHA256 = "4db2449b38b71d59f857e6122412bc4d9636a8840d2bee5a362d276d8df6ded9"

MANAGER_REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v024-real1km-perf")
MANAGER_POLICY_RELATIVE = ".agent/decisions/V0234-CORRECTED-IDENTITY-POLICY.md"
MANAGER_POLICY = MANAGER_REPO / MANAGER_POLICY_RELATIVE
MANAGER_POLICY_COMMIT = "898201ade3cc63af99ae4d216a6bb0563bc0aa51"
MANAGER_POLICY_SHA256 = "c991f2c83d814eb0efb1d99071b39a0b2e636e92b70bd90268e066af60612df2"
RELEASE_POLICY = Path("proofs/v014/switzerland_validation_plan.md")
RELEASE_POLICY_SHA256 = "60d68c04346d56c96d38be6507b6b6e801771ccde8734b7fd814e1ba258cb3c6"
REVIEW_REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-corrected-validation-review")
CODE_REVIEW_ACCEPTANCE_REF = "refs/v0234/corrected-validation/final-code-review-accepted"
CODE_REVIEW_ACCEPTANCE_RELATIVE = (
    ".agent/sprints/2026-07-12-v0234-corrected-validation/"
    "code-rereview-acceptance.json"
)
RUNTIME_ACCEPTANCE_PREFIX = (
    ".agent/sprints/2026-07-12-v0234-corrected-validation/runtime-acceptance-"
)
MANAGER_RUNTIME_ACCEPTANCE_REF = "refs/v0234/corrected-validation/runtime-accepted-once"
RUNTIME_AUTHORITY_ROOT = CPU_RUN_ROOT.parent / "runtime_authority"

MARKER_MAX_AGE_SECONDS = 120.0
RESOURCE_MAX_AGE_SECONDS = 120.0
PREEMPT_MAX_AGE_SECONDS = 120.0
QA_MAX_AGE_SECONDS = 180.0
DERIVED_PROOF_MAX_AGE_SECONDS = 60.0
RUNTIME_ACCEPTANCE_MAX_AGE_SECONDS = 60.0
RUNTIME_ACCEPTANCE_MAX_LIFETIME_SECONDS = 120.0

WRF_SOURCE_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")
TABLE_HASHES = {
    "MPTABLE.TBL": "7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d",
    "SOILPARM.TBL": "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f",
    "GENPARM.TBL": "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2",
}
WRF_DEPENDENCY_MANIFEST = {
    "run/MPTABLE.TBL": {
        "sha256": TABLE_HASHES["MPTABLE.TBL"], "size": 56140,
        "resolved_relative_path": "phys/noahmp/parameters/MPTABLE.TBL",
        "provenance": "sf_surface_physics=4:gpuwrf.physics.noahmp.tables.load_noahmp_parameters",
    },
    "run/SOILPARM.TBL": {
        "sha256": TABLE_HASHES["SOILPARM.TBL"], "size": 6557,
        "resolved_relative_path": "run/SOILPARM.TBL",
        "provenance": "sf_surface_physics=4:gpuwrf.physics.noahmp.tables.load_noahmp_parameters",
    },
    "run/GENPARM.TBL": {
        "sha256": TABLE_HASHES["GENPARM.TBL"], "size": 261,
        "resolved_relative_path": "run/GENPARM.TBL",
        "provenance": "sf_surface_physics=4:gpuwrf.physics.noahmp.tables.load_noahmp_parameters",
    },
    "phys/module_ra_rrtmg_lw.F": {
        "sha256": "c7a5238612aa8a4213c8d3af6708ec6a5248e6701e19758a80e563905d306de3",
        "size": 652002, "resolved_relative_path": "phys/module_ra_rrtmg_lw.F",
        "provenance": "ra_lw_physics=4:gpuwrf.physics.rrtmg_lw._native_lw_tables",
    },
}
WRF_LOADER_SOURCE_AUTHORITY = {
    "src/gpuwrf/physics/noahmp/tables.py": "70cb7ca5456b291a5651962f923ffaa58ebdc8ef216e2e9a558600b9d7a230f5",
    "src/gpuwrf/physics/rrtmg_lw.py": "74add582b3526d14d619d6a15ee899ed79e961e2cf15d6cd7664c59cf45ab94d",
    "scripts/extract_rrtmg_tables.py": "90fa7a2137fc818154f722a3a937339f397314c16cc79f6cf10ec95f51c6c4da",
}
WRF_SCHEME_DEPENDENCIES = {
    "sf_surface_physics=4": (
        "run/MPTABLE.TBL", "run/SOILPARM.TBL", "run/GENPARM.TBL",
    ),
    "ra_lw_physics=4": ("phys/module_ra_rrtmg_lw.F",),
    "ra_sw_physics=4": (),
}

LOCK_ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2")
LOCK_COMMIT = "8152309aff1e85e1052d44d549a5a5409e710bdd"
LOCK_WRAPPER = LOCK_ROOT / "scripts/with_gpu_lock.sh"
LOCK_WRAPPER_SHA256 = "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
LOCK_INTENT = "production-preemptible"

RUN_REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0233-corrected-fullbuffer-run-bb2ffe33")
PYTHON_BIN = Path("<USER_HOME>/miniconda3/bin/python")
CPU_LANE = tuple(range(12))
VALIDATION_LANE = (12, 13, 14, 15)
PRTERUN_ARGV0 = "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/prterun"
PRTE_RESOLVED = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/prte"
PRTE_SHA256 = "8e61af4375677a82d3dff0ba1fd96b6bc45778de0fe0c86b477c2421a08b73ea"
PRTERUN_ARGV = (PRTERUN_ARGV0, "--use-hwthread-cpus", "--bind-to", "none", "-np", "12", "./wrf.exe")
EXPECTED_START = datetime(2025, 3, 1, 0, 0, tzinfo=timezone.utc)
EXPECTED_TIMES = tuple(EXPECTED_START + timedelta(minutes=20 * i) for i in range(55))
EXPECTED_PARENT_TIMES = tuple(EXPECTED_START + timedelta(hours=i) for i in range(19))
TERMINAL_D01_TIMES = tuple(
    EXPECTED_START + timedelta(hours=i, seconds=(18 * i) % 54) for i in range(19)
)
EARLY_REQUIRED_TIMES = EXPECTED_TIMES[1:3]
EARLY_REQUIRED_STAMPS = tuple(value.strftime("%Y-%m-%d_%H:%M:%S") for value in EARLY_REQUIRED_TIMES)

INPUT_NAMES = (
    "namelist.input",
    "wrfbdy_d01",
    "wrfinput_d01",
    "wrfinput_d02",
    "wrfinput_d03",
)
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
STRICT_RMSE_LIMITS = {
    "T": 1.5,
    "U": 1.8,
    "V": 1.8,
    "W": 0.3,
    "T2": 1.5,
    "U10": 1.5,
    "V10": 1.5,
    "PSFC": 120.0,
}
STATIC_GATE_FIELDS = ("XLAT", "XLONG", "HGT", "LANDMASK")
REPORT_ONLY_FIELDS = ("QVAPOR", "RAINC", "RAINNC")
FRAME_REQUIRED_FIELDS = ("Times",) + STRICT_FIELDS + STATIC_GATE_FIELDS + REPORT_ONLY_FIELDS
CPU_PARENT_REQUIRED = {
    "Times", "T2", "U10", "V10", "PSFC", "Q2", "TSK", "U", "V", "W",
    "T", "QVAPOR", "PH", "PHB", "HGT",
}
CPU_CHILD_REQUIRED = CPU_PARENT_REQUIRED | {"XLAT", "XLONG", "LANDMASK", "CLDFRA"}
CPU_THIN_REQUIRED = {
    "Times", "XLAT", "XLONG", "LANDMASK", "HGT", "U10", "V10", "T2", "Q2",
    "TSK", "PSFC", "TCC", "CLD_LOW", "CLD_MID", "CLD_HIGH",
}
PHYSICAL_BOUNDS = {
    "T": (-200.0, 200.0),
    "U": (-250.0, 250.0),
    "V": (-250.0, 250.0),
    "W": (-100.0, 100.0),
    "T2": (250.0, 330.0),
    "U10": (-60.0, 60.0),
    "V10": (-60.0, 60.0),
    "PSFC": (55000.0, 104000.0),
}
PREEMPT_PATHS = (
    Path("/tmp/PREEMPT_GPU"),
    Path("/tmp/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION"),
)

EARLY_SCHEMA = "gpuwrf.v0234.corrected-fullbuffer-early-ready.v1"
TERMINAL_PROOF_SCHEMA = "gpuwrf.v0234.corrected-fullbuffer-terminal-cpu-authority.v1"
TABLE_SCHEMA = "gpuwrf.v0234.private-wrf-initialization-dependencies.v3"
WRF_DEPENDENCY_SAME_UID_RESIDUAL = (
    "read-only mode prevents accidental writes but the owning UID can chmod; "
    "post-command restat+rehash remains mandatory"
)
COMMAND_SCHEMA = "gpuwrf.v0234.corrected-fullbuffer-command.v1"
PAIR_SCHEMA = "gpuwrf.v0234.incremental-identity-pairs.v1"
CODE_REVIEW_SCHEMA = "gpuwrf.v0234.corrected-validation-code-review.v2"
RUNTIME_ACCEPTANCE_SCHEMA = "gpuwrf.v0234.corrected-validation-runtime-accept.v2"
NONCE_BURN_SCHEMA = "gpuwrf.v0234.corrected-validation-nonce-burn.v2"
UPSTREAM_MARKER_SCHEMA = "tenerife_early_gpu_ready_v1"
UPSTREAM_SEAL_SCHEMA = "tenerife_fullbuffer_cpu_oracle_input_seal_v1"
UPSTREAM_RESOURCE_SCHEMA = "tenerife_early_gpu_resource_proof_v1"
UPSTREAM_QA_SCHEMA = "tenerife_early_gpu_raw_qa_v1"
UPSTREAM_REVOKE_SCHEMA = "tenerife_early_gpu_ready_revoke_v1"


class GateError(RuntimeError):
    """Named fail-closed gate error."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_json_finite(value: object, *, path: str = "$") -> None:
    """Reject a nested NaN/Inf before it can be hashed or published as JSON."""
    if isinstance(value, (float, np.floating)):
        observed = float(value)
        if not math.isfinite(observed):
            raise GateError("JSON_NONFINITE", f"{path}={observed!r}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_json_finite(key, path=f"{path}.<key>")
            key_path = f"{path}.{key}" if isinstance(key, str) else f"{path}[{key!r}]"
            require_json_finite(item, path=key_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_json_finite(item, path=f"{path}[{index}]")


def canonical_json(value: object) -> bytes:
    require_json_finite(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def authority_digest(value: object) -> str:
    require_json_finite(value)
    return sha256_bytes(canonical_json(value))


def atomic_write_json(path: Path, payload: object) -> None:
    require_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise GateError("STALE_TEMP", str(temporary))
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _strict_json_bytes(raw: bytes, path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateError("INVALID_JSON", f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError("INVALID_JSON", f"{path}: root must be an object")
    return value


def load_exact_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateError("INVALID_JSON", f"{path}: {exc}") from exc
    return _strict_json_bytes(raw, path)


def stable_json_authority(
    path: Path,
    *,
    interval_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one atomically published JSON object and reject in-place replacement."""
    first_stat = _regular_file_stat(path)
    first_bytes = path.read_bytes()
    payload = _strict_json_bytes(first_bytes, path)
    sleep_fn(interval_seconds)
    second_stat = _regular_file_stat(path)
    second_bytes = path.read_bytes()
    identity = (
        first_stat.st_dev, first_stat.st_ino, first_stat.st_size,
        first_stat.st_mtime_ns, first_stat.st_ctime_ns, stat.S_IMODE(first_stat.st_mode),
    )
    second_identity = (
        second_stat.st_dev, second_stat.st_ino, second_stat.st_size,
        second_stat.st_mtime_ns, second_stat.st_ctime_ns, stat.S_IMODE(second_stat.st_mode),
    )
    if identity != second_identity or first_bytes != second_bytes:
        raise GateError("ATOMIC_AUTHORITY_CHANGED", str(path))
    return payload, {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_bytes(first_bytes),
        "bytes": len(first_bytes),
        "device": first_stat.st_dev,
        "inode": first_stat.st_ino,
        "mtime_ns": first_stat.st_mtime_ns,
        "ctime_ns": first_stat.st_ctime_ns,
        "mode": stat.S_IMODE(first_stat.st_mode),
    }


def parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateError("INVALID_TIME", value) from exc
    if parsed.tzinfo is None:
        raise GateError("NAIVE_TIME", value)
    return parsed.astimezone(timezone.utc)


def validate_launch_plan(path: Path = LAUNCH_PLAN) -> dict[str, Any]:
    _regular_file_stat(path)
    if path.resolve(strict=True) != LAUNCH_PLAN.resolve(strict=True):
        raise GateError("WRONG_LAUNCH_PLAN", str(path))
    digest = sha256_file(path)
    if digest != LAUNCH_PLAN_SHA256:
        raise GateError("LAUNCH_PLAN_HASH", digest)
    plan = load_exact_json(path)
    exact = {
        "schema": "tenerife_fullbuffer_cpu_oracle_launch_plan_v1",
        "status": "running_canonical_early_gpu_authority_pending_wrfgpu_rereview",
        "case_id": CASE_ID,
        "grid_id": GRID_ID,
        "grid_sha256": GRID_SHA256,
        "max_dom": 3,
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise GateError("LAUNCH_PLAN_FIELD", f"{key}={plan.get(key)!r}")
    if (
        plan.get("issue_time_utc") != "2025-02-28T18:00:00Z"
        or plan.get("valid_start_utc") != "2025-03-01T00:00:00Z"
        or plan.get("valid_end_utc") != "2025-03-01T18:00:00Z"
        or plan.get("forcing", {}).get("sha256") != AIFS_SHA256
        or plan.get("resources")
        != {
            "engine": "cpu_wrf_dmpar",
            "mpi_ranks": 12,
            "cpuset": "0-11",
            "nice": 5,
            "gpu": False,
            "estimated_wall_minutes": [75, 105],
            "conservative_upper_minutes": 120,
        }
        or plan.get("outputs", {}).get("workspace") != str(CPU_RUN_ROOT)
        or plan.get("outputs", {}).get("expected_raw_frames")
        != {"d01": 19, "d02": 19, "d03": 55}
        or plan.get("outputs", {}).get("expected_thin_frames") != {"d03": 55}
        or plan.get("gpu_identity_consumer")
        != {
            "owner": "wrf_gpu pane 0:1",
            "domain": "d03",
            "cadence_minutes": 20,
            "matched_frame_count": 55,
            "promotion_requires_cpu_qa_pass": True,
        }
        or plan.get("early_gpu_ready", {}).get("first_groups")
        != {
            "domain": "d03",
            "regular_valid_times": list(EARLY_REQUIRED_STAMPS),
            "initialization_00_00_counted": False,
            "required_raw_qa": [
                "closed", "readable_netcdf",
                "T_U_V_W_T2_U10_V10_PSFC_finite", "surface_ranges_green",
            ],
        }
        or plan.get("early_gpu_ready", {}).get("resource_contract")
        != {
            "cpu_oracle_cpuset": "0-11",
            "cpu_oracle_wrf_ranks": 12,
            "cpu_auxiliary_free": "12-15",
            "authoritative_gpu_lock": "/tmp/wrf_gpu2_gpu.lock",
            "gpu_compute_processes": 0,
            "minimum_gpu_free_mib": 24000,
            "maximum_gpu_utilization_percent": 20,
            "baseline_gpu_processes_recorded": True,
            "nightly_and_preempt_inactive": True,
        }
        or plan.get("early_gpu_ready", {}).get("full_cpu_verdict_at_emit") != "pending_55_of_55"
        or plan.get("early_gpu_ready", {}).get("status") != "canonical_bound_pending_fresh_marker"
        or plan.get("early_gpu_ready", {}).get("marker") != str(EARLY_MARKER)
        or plan.get("early_gpu_ready", {}).get("input_seal") != str(INPUT_SEAL)
        or plan.get("early_gpu_ready", {}).get("resource_proof") != str(RESOURCE_PROOF)
        or plan.get("early_gpu_ready", {}).get("revoke_action")
        != "atomic invalidation plus STOP_FAIL_CLOSED_AND_RELEASE_LOCK callbacks; owned CPU yields on production preemption"
        or plan.get("lineage", {}).get("wrf_downscale_early_gpu_gate_commit") != "3972f13dce7654eaa5dc40d882e3a88016028503"
        or plan.get("lineage", {}).get("wrf_downscale_canonical_authority_commit") != EARLY_AUTHORITY_COMMIT
        or plan.get("lineage", {}).get("wrf_downscale_emitter_commit") != EARLY_AUTHORITY_COMMIT
    ):
        raise GateError("LAUNCH_PLAN_SEMANTICS", "frozen plan fields changed")
    return {"path": str(path), "sha256": digest, "payload": plan}


def _strip_namelist_comments(text: str) -> str:
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def namelist_values(text: str, key: str) -> list[str]:
    clean = _strip_namelist_comments(text)
    pattern = re.compile(
        rf"\b{re.escape(key)}\s*=\s*(.*?)(?=\b[A-Za-z_]\w*\s*=|/)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(clean)
    if not match:
        raise GateError("NAMELIST_KEY", key)
    return [part.strip().strip("'\"") for part in match.group(1).split(",") if part.strip()]


def _namelist_ints(text: str, key: str) -> list[int]:
    try:
        return [int(float(value.replace("d", "e").replace("D", "e"))) for value in namelist_values(text, key)]
    except ValueError as exc:
        raise GateError("NAMELIST_VALUE", key) from exc


def validate_corrected_namelist(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    max_dom = _namelist_ints(text, "max_dom")[0]
    run_seconds = (
        _namelist_ints(text, "run_days")[0] * 86400
        + _namelist_ints(text, "run_hours")[0] * 3600
        + _namelist_ints(text, "run_minutes")[0] * 60
        + _namelist_ints(text, "run_seconds")[0]
    )
    e_we = _namelist_ints(text, "e_we")
    e_sn = _namelist_ints(text, "e_sn")
    i_start = _namelist_ints(text, "i_parent_start")
    j_start = _namelist_ints(text, "j_parent_start")
    ratios = _namelist_ints(text, "parent_grid_ratio")
    sf_surface = _namelist_ints(text, "sf_surface_physics")
    ra_lw = _namelist_ints(text, "ra_lw_physics")
    ra_sw = _namelist_ints(text, "ra_sw_physics")
    if (
        max_dom != 3
        or run_seconds != 18 * 3600
        or e_we[:3][-1] != 112
        or e_sn[:3][-1] != 94
        or i_start[:3][-1] != 92
        or j_start[:3][-1] != 36
        or ratios[:3][-1] != 3
        or sf_surface[:3] != [4, 4, 4]
        or ra_lw[:3] != [4, 4, 4]
        or ra_sw[:3] != [4, 4, 4]
    ):
        raise GateError(
            "WRONG_GEOMETRY",
            f"max_dom={max_dom} seconds={run_seconds} e={e_we[:3]}x{e_sn[:3]} "
            f"start={i_start[:3]}/{j_start[:3]} ratio={ratios[:3]} "
            f"sf={sf_surface[:3]} lw={ra_lw[:3]} sw={ra_sw[:3]}",
        )
    return {
        "max_dom": max_dom,
        "duration_seconds": run_seconds,
        "d03_e_we_e_sn": [e_we[2], e_sn[2]],
        "d03_mass_shape_yx": [e_sn[2] - 1, e_we[2] - 1],
        "d03_parent_start_ij": [i_start[2], j_start[2]],
        "d03_parent_grid_ratio": ratios[2],
        "physics_dependency_selection": {
            "sf_surface_physics": sf_surface[:3],
            "ra_lw_physics": ra_lw[:3],
            "ra_sw_physics": ra_sw[:3],
        },
    }


def required_wrf_initialization_dependencies(selection: Mapping[str, Any]) -> tuple[str, ...]:
    expected_selection = {
        "sf_surface_physics": [4, 4, 4],
        "ra_lw_physics": [4, 4, 4],
        "ra_sw_physics": [4, 4, 4],
    }
    if selection != expected_selection:
        raise GateError("WRF_DEPENDENCY_NAMELIST_SELECTION", repr(selection))
    selected = tuple(sorted({
        relative
        for key, values in expected_selection.items()
        for relative in WRF_SCHEME_DEPENDENCIES[f"{key}={values[0]}"]
    }))
    if set(selected) != set(WRF_DEPENDENCY_MANIFEST):
        raise GateError("WRF_DEPENDENCY_MANIFEST_COMPLETENESS", repr(selected))
    return selected


def _regular_file_stat(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise GateError("MISSING_FILE", str(path)) from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise GateError("NOT_REGULAR", str(path))
    return info


def file_authority(path: Path) -> dict[str, Any]:
    info = _regular_file_stat(path)
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "size": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def verify_pinned_file_metadata(
    authority: Mapping[str, Any], code: str, *, required_mode: int | None = None,
) -> dict[str, Any]:
    path = Path(str(authority.get("path", "")))
    info = _regular_file_stat(path)
    observed = {
        "path": str(path.resolve(strict=True)), "device": info.st_dev, "inode": info.st_ino,
        "size": info.st_size, "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
        "mode": stat.S_IMODE(info.st_mode),
    }
    for key in ("path", "device", "inode", "size", "mtime_ns", "ctime_ns"):
        if key in authority and authority.get(key) != observed[key]:
            raise GateError(code, f"{path}:{key}")
    if required_mode is not None and observed["mode"] != required_mode:
        raise GateError(code, f"{path}:mode={oct(observed['mode'])}")
    if "mode" in authority and authority.get("mode") != observed["mode"]:
        raise GateError(code, f"{path}:mode")
    return observed


def source_pinned_metadata(authority: Mapping[str, Any], code: str) -> dict[str, Any]:
    return verify_pinned_file_metadata({
        "path": authority.get("source_path"), "device": authority.get("source_device"),
        "inode": authority.get("source_inode"), "size": authority.get("source_size"),
        "mtime_ns": authority.get("source_mtime_ns"), "ctime_ns": authority.get("source_ctime_ns"),
    }, code)


def load_frozen_geometry() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = FROZEN_GEO_PATH
    authority = file_authority(path)
    if authority["sha256"] != GRID_SHA256:
        raise GateError("FROZEN_GEO_HASH", authority["sha256"])
    source_names = {
        "XLAT": "XLAT_M", "XLONG": "XLONG_M",
        "HGT": "HGT_M", "LANDMASK": "LANDMASK",
    }
    with Dataset(path, "r") as dataset:
        missing = sorted(set(source_names.values()) - set(dataset.variables))
        if missing:
            raise GateError("FROZEN_GEO_FIELDS", repr(missing))
        arrays = {field: np.asarray(dataset.variables[source][0]).copy() for field, source in source_names.items()}
    if file_authority(path) != authority:
        raise GateError("FROZEN_GEO_CHANGED_DURING_READ", str(path))
    return arrays, authority


def _static_plane(dataset: Dataset, field: str) -> np.ndarray:
    values = np.asarray(dataset.variables[field][:])
    if values.ndim >= 3 and values.shape[0] == 1:
        values = values[0]
    return values


def require_exact_frozen_geometry(dataset: Dataset, geo: Mapping[str, np.ndarray], label: str) -> None:
    for field in STATIC_GATE_FIELDS:
        if field not in dataset.variables:
            raise GateError("FROZEN_GEOMETRY_FIELD", f"{label}:{field}")
        actual = _static_plane(dataset, field)
        expected = geo[field]
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise GateError("FROZEN_GEOMETRY_EXACT", f"{label}:{field}")


def stable_file_authorities(
    paths: Sequence[Path],
    *,
    interval_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, dict[str, Any]]:
    first = {path.name: file_authority(path) for path in paths}
    sleep_fn(interval_seconds)
    second = {path.name: file_authority(path) for path in paths}
    if first != second:
        raise GateError("UNSTABLE_FILE", "file authority changed across interval")
    return second


def _read_resolved_regular(
    path: Path, *, source_root: Path, expected_resolved_relative: str,
) -> tuple[bytes, Path, os.stat_result, os.stat_result, str | None]:
    try:
        canonical_before = path.lstat()
        link_target = os.readlink(path) if stat.S_ISLNK(canonical_before.st_mode) else None
        if not (stat.S_ISREG(canonical_before.st_mode) or stat.S_ISLNK(canonical_before.st_mode)):
            raise GateError("WRF_SOURCE_PATH_TYPE", str(path))
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GateError("WRF_SOURCE_RESOLVE", str(path)) from exc
    expected_resolved = source_root / expected_resolved_relative
    if resolved != expected_resolved or not resolved.is_relative_to(source_root):
        raise GateError("WRF_SOURCE_PATH_SUBSTITUTION", f"{path}->{resolved}")
    resolved_info = resolved.lstat()
    if resolved.is_symlink() or not stat.S_ISREG(resolved_info.st_mode):
        raise GateError("WRF_SOURCE_NOT_REGULAR", str(resolved))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(resolved, flags)
    try:
        before = os.fstat(fd)
        if (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        ) != (
            resolved_info.st_dev, resolved_info.st_ino, resolved_info.st_size,
            resolved_info.st_mtime_ns, resolved_info.st_ctime_ns,
        ):
            raise GateError("WRF_SOURCE_CHANGED_BEFORE_READ", str(resolved))
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        canonical_after = path.lstat()
        resolved_after = path.resolve(strict=True)
    except OSError as exc:
        raise GateError("WRF_SOURCE_CHANGED_DURING_READ", str(path)) from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or any(getattr(canonical_before, field) != getattr(canonical_after, field) for field in stable_fields)
        or resolved_after != resolved
        or (os.readlink(path) if path.is_symlink() else None) != link_target
    ):
        raise GateError("WRF_SOURCE_CHANGED_DURING_READ", str(resolved))
    return b"".join(chunks), resolved, after, canonical_after, link_target


_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    _RENAMEAT2.restype = ctypes.c_int
RENAME_NOREPLACE = 1


def _rename_noreplace(dir_fd: int, source_name: str, destination_name: str) -> None:
    """Publish one dirfd-relative name atomically without replacing a collision."""
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    ctypes.set_errno(0)
    result = _RENAMEAT2(
        dir_fd, os.fsencode(source_name), dir_fd, os.fsencode(destination_name),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination_name)
    raise OSError(error, os.strerror(error), destination_name)


def _directory_binding(path: Path, code: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise GateError(code, str(path)) from exc
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or path.resolve() != path:
        raise GateError(code, str(path))
    if info.st_uid != os.geteuid():
        raise GateError(code, f"owner={info.st_uid} expected={os.geteuid()} path={path}")
    return {
        "path": str(path), "device": info.st_dev, "inode": info.st_ino,
        "uid": info.st_uid, "gid": info.st_gid, "mode": stat.S_IMODE(info.st_mode),
        "type": "directory",
    }


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_durable_snapshot_file(path: Path, data: bytes) -> os.stat_result:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise GateError("WRF_DEPENDENCY_SHORT_WRITE", str(path))
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        return os.fstat(fd)
    finally:
        os.close(fd)


def _expected_snapshot_root(expected_work_dir: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    work_dir = Path(expected_work_dir)
    if not work_dir.is_absolute():
        raise GateError("WRF_DEPENDENCY_EXPECTED_WORK_DIR", str(work_dir))
    work_binding = _directory_binding(work_dir, "WRF_DEPENDENCY_EXPECTED_WORK_DIR")
    authority_dir = work_dir / "authority"
    authority_binding = _directory_binding(authority_dir, "WRF_DEPENDENCY_AUTHORITY_DIR")
    return authority_dir / "wrf_root", work_binding, authority_binding


def materialize_table_snapshot(source_root: Path, work_dir: Path) -> dict[str, Any]:
    if source_root != WRF_SOURCE_ROOT or source_root.is_symlink() or source_root.resolve() != WRF_SOURCE_ROOT:
        raise GateError("WRONG_WRF_ROOT", str(source_root))
    for relative, digest in WRF_LOADER_SOURCE_AUTHORITY.items():
        if sha256_file(RUN_REPO / relative) != digest:
            raise GateError("WRF_LOADER_SOURCE_MUTATION", relative)
    namelist = validate_corrected_namelist(CPU_INPUT_DIR / "namelist.input")
    selection = namelist["physics_dependency_selection"]
    required_wrf_initialization_dependencies(selection)
    work_dir = Path(work_dir)
    if not work_dir.is_absolute() or (work_dir.exists() and (work_dir.is_symlink() or not work_dir.is_dir())):
        raise GateError("WRF_DEPENDENCY_EXPECTED_WORK_DIR", str(work_dir))
    authority_dir = work_dir / "authority"
    authority_dir.mkdir(parents=True, exist_ok=True)
    target_root, work_binding, authority_dir_binding = _expected_snapshot_root(work_dir)
    temporary_root = authority_dir / f".wrf_root.tmp-{os.getpid()}"
    if target_root.exists() or temporary_root.exists():
        raise GateError("TABLE_SNAPSHOT_EXISTS", str(target_root))
    temporary_root.mkdir()
    inventory: dict[str, Any] = {}
    try:
        for relative, expected in WRF_DEPENDENCY_MANIFEST.items():
            source = source_root / relative
            data, resolved, source_info, canonical_info, link_target = _read_resolved_regular(
                source, source_root=source_root,
                expected_resolved_relative=expected["resolved_relative_path"],
            )
            digest = sha256_bytes(data)
            if digest != expected["sha256"] or len(data) != expected["size"]:
                raise GateError("WRF_DEPENDENCY_HASH", f"{relative}: {digest}/{len(data)}")
            snapshot = temporary_root / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snap_info = _write_durable_snapshot_file(snapshot, data)
            inventory[relative] = {
                "relative_path": relative,
                "canonical_source_path": str(source),
                "resolved_source_path": str(resolved),
                "source_kind": "symlink" if link_target is not None else "regular",
                "source_link_target": link_target,
                "source_canonical_device": canonical_info.st_dev,
                "source_canonical_inode": canonical_info.st_ino,
                "source_canonical_mtime_ns": canonical_info.st_mtime_ns,
                "source_canonical_ctime_ns": canonical_info.st_ctime_ns,
                "source_device": source_info.st_dev,
                "source_inode": source_info.st_ino,
                "source_mtime_ns": source_info.st_mtime_ns,
                "source_ctime_ns": source_info.st_ctime_ns,
                "snapshot_path": str(target_root / relative),
                "snapshot_device": snap_info.st_dev,
                "snapshot_inode": snap_info.st_ino,
                "snapshot_mtime_ns": snap_info.st_mtime_ns,
                "snapshot_ctime_ns": snap_info.st_ctime_ns,
                "size": len(data),
                "mode": stat.S_IMODE(snap_info.st_mode),
                "sha256": digest,
                "provenance": expected["provenance"],
            }
        for directory in sorted(
            (path for path in temporary_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts), reverse=True,
        ):
            os.chmod(directory, 0o555)
            _fsync_directory(directory)
        os.chmod(temporary_root, 0o555)
        _fsync_directory(temporary_root)
        authority_fd = os.open(
            authority_dir,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            try:
                _rename_noreplace(authority_fd, temporary_root.name, target_root.name)
            except FileExistsError as exc:
                raise GateError("WRF_DEPENDENCY_PUBLISH_COLLISION", str(target_root)) from exc
            os.fsync(authority_fd)
        finally:
            os.close(authority_fd)
    except BaseException:
        if temporary_root.exists():
            for directory in [temporary_root, *temporary_root.rglob("*")]:
                if directory.is_dir() and not directory.is_symlink():
                    os.chmod(directory, 0o755)
            shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    result = {
        "schema": TABLE_SCHEMA,
        "source_root": str(source_root),
        "snapshot_root": str(target_root),
        "work_dir_authority": work_binding,
        "authority_dir_authority": authority_dir_binding,
        "namelist_path": str(CPU_INPUT_DIR / "namelist.input"),
        "namelist_sha256": sha256_file(CPU_INPUT_DIR / "namelist.input"),
        "namelist_dependency_selection": selection,
        "loader_source_authority": dict(WRF_LOADER_SOURCE_AUTHORITY),
        "inventory": inventory,
        "inventory_completeness": "exact selected max_dom3 initialization closure; no ambient fallback",
        "same_uid_chmod_residual": WRF_DEPENDENCY_SAME_UID_RESIDUAL,
    }
    result["authority_sha256"] = authority_digest(result)
    verify_table_snapshot(result, expected_work_dir=work_dir)
    return result


def verify_table_snapshot(
    authority: Mapping[str, Any], *, expected_work_dir: Path,
) -> dict[str, Any]:
    if authority.get("schema") != TABLE_SCHEMA:
        raise GateError("TABLE_SCHEMA", str(authority.get("schema")))
    if set(authority) != {
        "schema", "source_root", "snapshot_root", "work_dir_authority",
        "authority_dir_authority", "namelist_path", "namelist_sha256",
        "namelist_dependency_selection", "loader_source_authority", "inventory",
        "inventory_completeness", "same_uid_chmod_residual", "authority_sha256",
    }:
        raise GateError("WRF_DEPENDENCY_AUTHORITY_SCHEMA", repr(sorted(authority)))
    unsigned = dict(authority)
    recorded_digest = unsigned.pop("authority_sha256", None)
    if authority_digest(unsigned) != recorded_digest:
        raise GateError("TABLE_AUTHORITY_DIGEST", "semantic authority changed")
    source_root = Path(str(authority["source_root"]))
    root = Path(str(authority["snapshot_root"]))
    expected_root, work_binding, authority_dir_binding = _expected_snapshot_root(expected_work_dir)
    namelist_path = CPU_INPUT_DIR / "namelist.input"
    selection = validate_corrected_namelist(namelist_path)["physics_dependency_selection"]
    required_wrf_initialization_dependencies(selection)
    if (
        source_root != WRF_SOURCE_ROOT or source_root.is_symlink() or source_root.resolve() != WRF_SOURCE_ROOT
        or root != expected_root
        or authority["work_dir_authority"] != work_binding
        or authority["authority_dir_authority"] != authority_dir_binding
        or authority["namelist_path"] != str(namelist_path)
        or authority["namelist_sha256"] != sha256_file(namelist_path)
        or authority["namelist_dependency_selection"] != selection
        or authority["loader_source_authority"] != WRF_LOADER_SOURCE_AUTHORITY
        or authority["inventory_completeness"]
        != "exact selected max_dom3 initialization closure; no ambient fallback"
        or authority["same_uid_chmod_residual"] != WRF_DEPENDENCY_SAME_UID_RESIDUAL
    ):
        raise GateError("WRF_DEPENDENCY_AUTHORITY_BINDING", "root/namelist/loader selection")
    for relative, digest in WRF_LOADER_SOURCE_AUTHORITY.items():
        if sha256_file(RUN_REPO / relative) != digest:
            raise GateError("WRF_LOADER_SOURCE_MUTATION", relative)
    if root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.lstat().st_mode) != 0o555:
        raise GateError("WRF_DEPENDENCY_ROOT", str(root))
    expected_entries = {"run", "phys", *WRF_DEPENDENCY_MANIFEST}
    actual_entries: set[str] = set()
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*dirnames, *filenames]:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise GateError("WRF_DEPENDENCY_TARGET_SYMLINK", relative)
            actual_entries.add(relative)
        for name in dirnames:
            directory = current_path / name
            if stat.S_IMODE(directory.lstat().st_mode) != 0o555:
                raise GateError("WRF_DEPENDENCY_DIRECTORY_MODE", str(directory))
    if actual_entries != expected_entries:
        raise GateError("WRF_DEPENDENCY_EXTRA_OR_MISSING", repr(sorted(actual_entries)))
    inventory = authority["inventory"]
    if not isinstance(inventory, dict) or set(inventory) != set(WRF_DEPENDENCY_MANIFEST):
        raise GateError("WRF_DEPENDENCY_INVENTORY", repr(inventory))
    verified: dict[str, Any] = {}
    for relative, expected in WRF_DEPENDENCY_MANIFEST.items():
        detail = inventory.get(relative)
        expected_detail_keys = {
            "relative_path", "canonical_source_path", "resolved_source_path", "source_kind",
            "source_link_target", "source_canonical_device", "source_canonical_inode",
            "source_canonical_mtime_ns", "source_canonical_ctime_ns", "source_device",
            "source_inode", "source_mtime_ns", "source_ctime_ns", "snapshot_path",
            "snapshot_device", "snapshot_inode", "snapshot_mtime_ns", "snapshot_ctime_ns",
            "size", "mode", "sha256", "provenance",
        }
        if not isinstance(detail, dict) or set(detail) != expected_detail_keys:
            raise GateError("WRF_DEPENDENCY_DETAIL_SCHEMA", relative)
        source = source_root / relative
        source_data, resolved, source_info, canonical_info, link_target = _read_resolved_regular(
            source, source_root=source_root,
            expected_resolved_relative=expected["resolved_relative_path"],
        )
        path = root / relative
        info = _regular_file_stat(path)
        digest = sha256_file(path)
        if (
            detail["relative_path"] != relative
            or detail["canonical_source_path"] != str(source)
            or detail["resolved_source_path"] != str(resolved)
            or detail["source_kind"] != ("symlink" if link_target is not None else "regular")
            or detail["source_link_target"] != link_target
            or detail["source_canonical_device"] != canonical_info.st_dev
            or detail["source_canonical_inode"] != canonical_info.st_ino
            or detail["source_canonical_mtime_ns"] != canonical_info.st_mtime_ns
            or detail["source_canonical_ctime_ns"] != canonical_info.st_ctime_ns
            or detail["source_device"] != source_info.st_dev
            or detail["source_inode"] != source_info.st_ino
            or detail["source_mtime_ns"] != source_info.st_mtime_ns
            or detail["source_ctime_ns"] != source_info.st_ctime_ns
            or sha256_bytes(source_data) != expected["sha256"]
            or len(source_data) != expected["size"]
            or detail["snapshot_path"] != str(path)
            or detail["snapshot_device"] != info.st_dev
            or detail["snapshot_inode"] != info.st_ino
            or detail["snapshot_mtime_ns"] != info.st_mtime_ns
            or detail["snapshot_ctime_ns"] != info.st_ctime_ns
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) == (source_info.st_dev, source_info.st_ino)
            or digest != expected["sha256"]
            or detail.get("sha256") != expected["sha256"]
            or detail.get("size") != info.st_size
            or detail.get("mode") != 0o444
            or stat.S_IMODE(info.st_mode) != 0o444
            or detail["provenance"] != expected["provenance"]
        ):
            raise GateError("WRF_DEPENDENCY_MUTATION", relative)
        verified[relative] = {"sha256": digest, "size": info.st_size, "mode": 0o444}
    return verified


def cpu_initialization_dependency_smoke(
    authority: Mapping[str, Any], *, expected_work_dir: Path,
) -> dict[str, Any]:
    """Exercise real CPU-only table/source parsers without importing a JAX backend."""
    verify_table_snapshot(authority, expected_work_dir=expected_work_dir)
    code = r'''
import hashlib, importlib.util, json, os, sys, types
from pathlib import Path
import numpy as np

class Config:
    @staticmethod
    def update(*_args, **_kwargs):
        return None

jax = types.ModuleType("jax")
jax.__path__ = []
jax.config = Config()
jax.numpy = np
jax.lax = types.ModuleType("jax.lax")
jax.tree_util = types.SimpleNamespace(register_pytree_node_class=lambda cls: cls)
jax.jit = lambda function=None, **_kwargs: (lambda fn: fn) if function is None else function
sys.modules["jax"] = jax
sys.modules["jax.numpy"] = np
sys.modules["jax.lax"] = jax.lax
candidate_src = Path(os.environ["CANDIDATE_SRC"])
gpuwrf = types.ModuleType("gpuwrf")
gpuwrf.__path__ = [str(candidate_src / "gpuwrf")]
physics = types.ModuleType("gpuwrf.physics")
physics.__path__ = [str(candidate_src / "gpuwrf" / "physics")]
noahmp = types.ModuleType("gpuwrf.physics.noahmp")
noahmp.__path__ = [str(candidate_src / "gpuwrf" / "physics" / "noahmp")]
x64 = types.ModuleType("gpuwrf._x64_config")
x64.configure_jax_x64 = lambda: True
config_pkg = types.ModuleType("gpuwrf.config")
config_pkg.__path__ = []
paths = types.ModuleType("gpuwrf.config.paths")
paths.wrf_root = lambda: Path(os.environ["GPUWRF_WRF_ROOT"])
paths.wrf_run_dir = lambda: paths.wrf_root() / "run"
debug_pkg = types.ModuleType("gpuwrf.debug")
debug_pkg.__path__ = []
debug_asserts = types.ModuleType("gpuwrf.debug.asserts")
debug_asserts.assert_finite = lambda value, *_args, **_kwargs: value
debug_asserts.assert_physical_bounds = lambda value, *_args, **_kwargs: value
sys.modules.update({
    "gpuwrf": gpuwrf, "gpuwrf.physics": physics,
    "gpuwrf.physics.noahmp": noahmp, "gpuwrf._x64_config": x64,
    "gpuwrf.config": config_pkg, "gpuwrf.config.paths": paths,
    "gpuwrf.debug": debug_pkg, "gpuwrf.debug.asserts": debug_asserts,
})
tables_path = os.path.join(os.environ["CANDIDATE_SRC"], "gpuwrf", "physics", "noahmp", "tables.py")
tables_spec = importlib.util.spec_from_file_location("gpuwrf.physics.noahmp.tables", tables_path)
tables = importlib.util.module_from_spec(tables_spec)
sys.modules[tables_spec.name] = tables
tables_spec.loader.exec_module(tables)
params = tables.load_noahmp_parameters()
lw_path = candidate_src / "gpuwrf" / "physics" / "rrtmg_lw.py"
lw_spec = importlib.util.spec_from_file_location("gpuwrf.physics.rrtmg_lw", lw_path)
lw_module = importlib.util.module_from_spec(lw_spec)
sys.modules[lw_spec.name] = lw_module
lw_spec.loader.exec_module(lw_module)
native_lw = lw_module._native_lw_tables()
lw_source = Path(os.environ["GPUWRF_WRF_ROOT"]) / "phys" / "module_ra_rrtmg_lw.F"
real_jax = [name for name, value in sys.modules.items() if name.startswith("jaxlib") or getattr(value, "__file__", "") and "/jax/" in str(getattr(value, "__file__", ""))]
if real_jax:
    raise RuntimeError("real JAX backend imported: " + repr(real_jax))
print(json.dumps({
    "noahmp_parameter_fields": len(params._fields),
    "native_lw_chi_mls_shape": list(native_lw.chi_mls.shape),
    "native_lw_absa_shape": list(native_lw.absa.shape),
    "lw_source_sha256": hashlib.sha256(lw_source.read_bytes()).hexdigest(),
    "production_loader": "gpuwrf.physics.rrtmg_lw._native_lw_tables",
    "real_jax_backend_imported": False,
}, sort_keys=True))
'''
    environment = {
        "HOME": "<USER_HOME>",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "GPUWRF_WRF_ROOT": str(authority["snapshot_root"]),
        "CANDIDATE_ROOT": str(RUN_REPO),
        "CANDIDATE_SRC": str(RUN_REPO / "src"),
        "PYTHONNOUSERSITE": "1",
    }
    result = subprocess.run(
        ["/usr/bin/taskset", "-c", "12-15", str(PYTHON_BIN), "-c", code],
        env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise GateError("WRF_DEPENDENCY_CPU_SMOKE", result.stderr[-2000:])
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise GateError("WRF_DEPENDENCY_CPU_SMOKE_OUTPUT", result.stdout[-2000:]) from exc
    expected = {
        "noahmp_parameter_fields": 72,
        "native_lw_chi_mls_shape": [7, 59],
        "native_lw_absa_shape": [16, 585, 16],
        "lw_source_sha256": WRF_DEPENDENCY_MANIFEST["phys/module_ra_rrtmg_lw.F"]["sha256"],
        "production_loader": "gpuwrf.physics.rrtmg_lw._native_lw_tables",
        "real_jax_backend_imported": False,
    }
    if observed != expected:
        raise GateError("WRF_DEPENDENCY_CPU_SMOKE_SEMANTICS", repr(observed))
    return observed


def parse_wrfout_time(path: Path) -> datetime:
    match = re.fullmatch(r"wrfout_d03_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})", path.name)
    if not match:
        raise GateError("FRAME_NAME", path.name)
    return datetime.strptime(match.group(1), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)


def _decoded_times(dataset: Dataset) -> str:
    raw = np.asarray(dataset.variables["Times"][0])
    try:
        return str(chartostring(raw).item()).strip()
    except Exception:
        return b"".join(raw.astype("S1").tolist()).decode("ascii").strip()


def _variable_array(dataset: Dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    raw = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
    return np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)


def qa_d03_frame(path: Path, expected_time: datetime) -> dict[str, Any]:
    _regular_file_stat(path)
    with Dataset(path, "r") as dataset:
        missing = sorted(set(FRAME_REQUIRED_FIELDS) - set(dataset.variables))
        if missing:
            raise GateError("FRAME_FIELDS", f"{path}: {missing}")
        expected_stamp = expected_time.strftime("%Y-%m-%d_%H:%M:%S")
        if _decoded_times(dataset) != expected_stamp:
            raise GateError("FRAME_TIME", f"{path}: {_decoded_times(dataset)!r}")
        if (
            int(getattr(dataset, "GRID_ID", -1)) != 3
            or int(getattr(dataset, "WEST-EAST_GRID_DIMENSION", -1)) != 112
            or int(getattr(dataset, "SOUTH-NORTH_GRID_DIMENSION", -1)) != 94
        ):
            raise GateError("FRAME_GRID", str(path))
        ranges: dict[str, list[float]] = {}
        for name in FRAME_REQUIRED_FIELDS:
            if name == "Times":
                continue
            values = _variable_array(dataset, name)
            if not np.isfinite(values).all():
                raise GateError("FRAME_NONFINITE", f"{path}:{name}")
            lower = float(values.min())
            upper = float(values.max())
            ranges[name] = [lower, upper]
            if name in PHYSICAL_BOUNDS:
                allowed = PHYSICAL_BOUNDS[name]
                if lower < allowed[0] or upper > allowed[1]:
                    raise GateError("FRAME_UNPHYSICAL", f"{path}:{name}={lower, upper}")
        if _variable_array(dataset, "XLAT").shape[-2:] != (93, 111):
            raise GateError("FRAME_MASS_SHAPE", str(path))
    return {
        "path": str(path.resolve(strict=True)),
        "valid_time": expected_time.isoformat(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "ranges": ranges,
        "required_fields": list(FRAME_REQUIRED_FIELDS),
    }


def discover_d03_frames(directory: Path) -> dict[datetime, Path]:
    return discover_domain_frames(directory, "d03")


def discover_domain_frames(directory: Path, domain: str) -> dict[datetime, Path]:
    if domain not in {"d01", "d02", "d03"}:
        raise GateError("FRAME_DOMAIN", domain)
    result: dict[datetime, Path] = {}
    if not directory.is_dir():
        return result
    pattern = re.compile(rf"wrfout_{domain}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}:\d{{2}}:\d{{2}})")
    for path in sorted(directory.glob(f"wrfout_{domain}_*")):
        if path.is_symlink() or not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        valid = datetime.strptime(match.group(1), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        if valid in result:
            raise GateError("DUPLICATE_FRAME_TIME", valid.isoformat())
        result[valid] = path
    return result


def strict_domain_frame_inventory(directory: Path, domain: str) -> dict[datetime, Path]:
    candidates = sorted(directory.glob(f"wrfout_{domain}_*")) if directory.is_dir() else []
    frames = discover_domain_frames(directory, domain)
    if len(candidates) != len(frames):
        raise GateError(
            "FRAME_INVENTORY_NONCANONICAL",
            f"{domain}: candidates={len(candidates)} canonical={len(frames)}",
        )
    return frames


def _exact_fields(value: Mapping[str, Any], expected: Mapping[str, Any], code: str) -> None:
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise GateError(code, f"{key}={value.get(key)!r}, expected {wanted!r}")


def validate_upstream_early_authority() -> dict[str, Any]:
    commit = subprocess.check_output(
        ("git", "-C", str(EARLY_AUTHORITY_REPO), "rev-parse", EARLY_AUTHORITY_COMMIT),
        text=True,
    ).strip()
    source = subprocess.check_output(
        ("git", "-C", str(EARLY_AUTHORITY_REPO), "show", f"{commit}:{EARLY_AUTHORITY_SOURCE}"),
    )
    report_relative = str(EARLY_AUTHORITY_REPORT.relative_to(EARLY_AUTHORITY_REPO))
    report_commit = subprocess.check_output(
        ("git", "-C", str(EARLY_AUTHORITY_REPO), "rev-parse", EARLY_AUTHORITY_REPORT_COMMIT),
        text=True,
    ).strip()
    committed_report = subprocess.check_output(
        ("git", "-C", str(EARLY_AUTHORITY_REPO), "show", f"{report_commit}:{report_relative}"),
    )
    report_hash = sha256_file(EARLY_AUTHORITY_REPORT)
    if (
        commit != EARLY_AUTHORITY_COMMIT
        or report_commit != EARLY_AUTHORITY_REPORT_COMMIT
        or sha256_bytes(source) != EARLY_AUTHORITY_SOURCE_SHA256
        or sha256_bytes(committed_report) != EARLY_AUTHORITY_REPORT_SHA256
        or report_hash != EARLY_AUTHORITY_REPORT_SHA256
    ):
        raise GateError("EARLY_AUTHORITY_SOURCE", "wrf_downscale authority changed")
    return {
        "repo": str(EARLY_AUTHORITY_REPO),
        "commit": commit,
        "emitter": EARLY_AUTHORITY_SOURCE,
        "emitter_sha256": sha256_bytes(source),
        "report": str(EARLY_AUTHORITY_REPORT),
        "report_commit": report_commit,
        "report_sha256": report_hash,
    }


def _validate_binding_file_record(
    record: Mapping[str, Any], actual: Path, code: str, *, expected_record_path: Path,
) -> dict[str, Any]:
    if set(record) != {
        "bytes", "device", "inode", "mode", "mtime_ns", "path", "resolved_path", "sha256",
    }:
        raise GateError(code, repr(sorted(record)))
    expected_text = str(expected_record_path)
    if record.get("path") != expected_text or record.get("resolved_path") != expected_text:
        raise GateError(code, f"bound role differs: {record.get('path')!r}")
    info = _regular_file_stat(actual)
    digest = sha256_file(actual)
    if (
        info.st_size != record["bytes"] or info.st_dev != record["device"]
        or info.st_ino != record["inode"] or stat.S_IMODE(info.st_mode) != record["mode"]
        or info.st_mtime_ns != record["mtime_ns"] or digest != record["sha256"]
    ):
        raise GateError(code, str(actual))
    return file_authority(actual)


def _validate_compatibility_symlink(binding: Mapping[str, Any], compatibility: Path) -> None:
    if (
        binding.get("compatibility_symlink") != {"path": str(compatibility), "target": "../run"}
        or not compatibility.is_symlink() or os.readlink(compatibility) != "../run"
        or compatibility.resolve(strict=True) != CPU_RUN_ROOT
    ):
        raise GateError("CANONICAL_COMPATIBILITY_SYMLINK", str(compatibility))


def validate_prterun_authority() -> dict[str, Any]:
    argv0 = Path(PRTERUN_ARGV0)
    if not argv0.is_symlink() or os.readlink(argv0) != "prte":
        raise GateError("PRTERUN_ARGV0", str(argv0))
    resolved = argv0.resolve(strict=True)
    if str(resolved) != PRTE_RESOLVED or sha256_file(resolved) != PRTE_SHA256:
        raise GateError("PRTERUN_BINARY", str(resolved))
    return {
        "argv": list(PRTERUN_ARGV), "argv0": PRTERUN_ARGV0,
        "resolved": PRTE_RESOLVED, "sha256": PRTE_SHA256,
    }


def validate_canonical_authority_v2() -> dict[str, Any]:
    contract, contract_artifact = stable_json_authority(
        CANONICAL_V2_CONTRACT, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    if contract_artifact["sha256"] != CANONICAL_V2_CONTRACT_SHA256:
        raise GateError("CANONICAL_V2_CONTRACT_HASH", contract_artifact["sha256"])
    if set(contract) != {
        "schema", "status", "created_utc", "canonical_root", "launch_plan", "emitter",
        "report", "canonical_promotion", "forcing_preflight", "pre_wrf_seal", "live_open_fd",
        "required_decision", "marker_exact_keys", "gpu_authority",
    }:
        raise GateError("CANONICAL_V2_CONTRACT_SCHEMA", repr(sorted(contract)))
    _exact_fields(contract, {
        "schema": "tenerife_cpu_oracle_canonical_authority_contract_v2",
        "status": "gpu_blocked_pending_WRFGPU_seal_decision_and_fresh_rereview",
        "canonical_root": str(CPU_RUN_ROOT), "gpu_authority": "NOT_AUTHORIZED",
    }, "CANONICAL_V2_CONTRACT_FIELD")
    expected_marker_keys = [
        "schema", "status", "emitted_utc", "case_id", "grid_id", "grid_sha256",
        "input_seal_path", "input_hashes", "provenance", "cpu_oracle",
        "complete_regular_d03_groups", "initialization_group_counted",
        "resource_proof_path", "resource_proof", "full_cpu_verdict", "gpu_contract",
        "callback_targets",
    ]
    if contract.get("marker_exact_keys") != expected_marker_keys:
        raise GateError("CANONICAL_V2_MARKER_KEYS", repr(contract.get("marker_exact_keys")))
    if contract.get("required_decision") != (
        "accept original truthful pre-WRF seal plus canonical promotion binding, "
        "or wait CPU terminal for fresh open_by_pids=[] seal"
    ):
        raise GateError("CANONICAL_V2_REQUIRED_DECISION", repr(contract.get("required_decision")))
    if CANONICAL_V2_CONTRACT_SHA256 == "d7b266bf490e72b8c35977995defd4e6357bc98b5e85467fba4a26ebf30ad877":
        raise GateError("CANONICAL_V2_SUPERSEDED_CONTRACT", CANONICAL_V2_CONTRACT_SHA256)
    root_info = CPU_RUN_ROOT.lstat()
    if CPU_RUN_ROOT.is_symlink() or not stat.S_ISDIR(root_info.st_mode) or CPU_RUN_ROOT.resolve() != CPU_RUN_ROOT:
        raise GateError("CANONICAL_ROOT_DIRECTORY", str(CPU_RUN_ROOT))

    launch = validate_launch_plan(LAUNCH_PLAN)
    launch_contract = contract.get("launch_plan") or {}
    if launch_contract != {"path": str(LAUNCH_PLAN), "sha256": LAUNCH_PLAN_SHA256, "payload": launch["payload"]}:
        raise GateError("CANONICAL_LAUNCH_BINDING", repr(launch_contract))
    upstream = validate_upstream_early_authority()
    if contract.get("emitter") != {
        "commit": EARLY_AUTHORITY_COMMIT,
        "path": str(EARLY_AUTHORITY_REPO / EARLY_AUTHORITY_SOURCE),
        "sha256": EARLY_AUTHORITY_SOURCE_SHA256,
    } or contract.get("report") != {
        "path": str(EARLY_AUTHORITY_REPORT), "sha256": EARLY_AUTHORITY_REPORT_SHA256,
        "commit": EARLY_AUTHORITY_REPORT_COMMIT,
    }:
        raise GateError("CANONICAL_SOURCE_BINDING", "emitter/report differs")

    binding, binding_artifact = stable_json_authority(
        CANONICAL_BINDING, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    if binding_artifact["sha256"] != CANONICAL_BINDING_SHA256:
        raise GateError("CANONICAL_BINDING_HASH", binding_artifact["sha256"])
    if set(binding) != {
        "schema", "status", "bound_utc", "case_id", "grid_id", "repaired_attempt_id",
        "canonical_path", "canonical_resolved_path", "canonical_is_regular_directory",
        "compatibility_symlink", "failed_attempt_archive", "failed_attempt_manifest",
        "archived_attempt_local_authority", "sealed_inputs", "input_seal",
        "input_seal_checksums_sha256", "input_seal_launch_plan_sha256",
        "launch_runtime", "early_groups", "atomicity", "gpu_authority",
    }:
        raise GateError("CANONICAL_BINDING_SCHEMA", repr(sorted(binding)))
    _exact_fields(binding, {
        "schema": "tenerife_cpu_oracle_canonical_run_binding_v1",
        "status": "canonical_bound_repaired_attempt_running", "case_id": CASE_ID,
        "grid_id": GRID_ID, "canonical_is_regular_directory": True,
        "canonical_path": str(CPU_RUN_ROOT), "canonical_resolved_path": str(CPU_RUN_ROOT),
        "failed_attempt_archive": str(CPU_CASE_ROOT / "attempts/attempt_failed_pre_wrf_missing_forcing"),
        "gpu_authority": "not_authorized_pending_fresh_WRFGPU_review_and_new_canonical_marker",
    }, "CANONICAL_BINDING_FIELD")
    if contract.get("canonical_promotion") != {"path": str(CANONICAL_BINDING), "sha256": CANONICAL_BINDING_SHA256}:
        raise GateError("CANONICAL_PROMOTION_ROLE", repr(contract.get("canonical_promotion")))
    compatibility = CPU_CASE_ROOT / "attempts/attempt_repaired_v2"
    _validate_compatibility_symlink(binding, compatibility)
    failed = CPU_CASE_ROOT / "attempts/attempt_failed_pre_wrf_missing_forcing"
    if failed.is_symlink() or not failed.is_dir():
        raise GateError("CANONICAL_FAILED_ATTEMPT", str(failed))
    _validate_binding_file_record(
        binding["failed_attempt_manifest"], failed / "attempt_failed_pre_wrf_missing_forcing.json",
        "CANONICAL_FAILED_ATTEMPT_MANIFEST",
        expected_record_path=failed / "attempt_failed_pre_wrf_missing_forcing.json",
    )
    for name in ("EARLY_GPU_READY_REVOKED.json", "EARLY_GPU_READY.invalidated.json"):
        if (CPU_RUN_ROOT / name).exists() or (CPU_RUN_ROOT / name).is_symlink():
            raise GateError("CANONICAL_ROOT_REVOKED_HISTORY", name)
    for record in binding.get("archived_attempt_local_authority", []):
        if set(record) != {
            "archived_path", "bytes", "device", "inode", "mode", "mtime_ns",
            "path", "resolved_path", "sha256",
        }:
            raise GateError("CANONICAL_ARCHIVED_AUTHORITY", repr(sorted(record)))
        archived_record = {key: value for key, value in record.items() if key != "archived_path"}
        _validate_binding_file_record(
            archived_record, Path(record["archived_path"]), "CANONICAL_ARCHIVED_AUTHORITY",
            expected_record_path=compatibility / Path(record["archived_path"]).name,
        )

    sealed: dict[str, Any] = {}
    if not isinstance(binding.get("sealed_inputs"), list) or len(binding["sealed_inputs"]) != 5:
        raise GateError("CANONICAL_SEALED_INPUTS", repr(binding.get("sealed_inputs")))
    for record in binding["sealed_inputs"]:
        name = Path(str(record.get("path"))).name
        if name not in INPUT_NAMES or name in sealed:
            raise GateError("CANONICAL_SEALED_INPUT_NAME", name)
        expected_compat = compatibility / "wrf" / name
        if record.get("path") != str(expected_compat):
            raise GateError("CANONICAL_SEALED_INPUT_ROLE", str(record.get("path")))
        sealed[name] = _validate_binding_file_record(
            record, CPU_INPUT_DIR / name, "CANONICAL_SEALED_INPUT_FILE",
            expected_record_path=expected_compat,
        )
        if sealed[name]["mode"] != 0o444:
            raise GateError("CANONICAL_SEALED_INPUT_MODE", name)
    live_fd = contract.get("live_open_fd")
    if not isinstance(live_fd, dict) or set(live_fd) != {
        "pid", "fd", "flags_octal", "access", "flock", "device", "inode",
        "mode", "bytes", "sha256",
    }:
        raise GateError("CANONICAL_LIVE_FD_SCHEMA", repr(live_fd))
    wrfbdy = sealed["wrfbdy_d01"]
    _exact_fields(live_fd, {
        "fd": 37, "flags_octal": "0100000", "access": "read_only",
        "flock": "ADVISORY_READ", "device": wrfbdy["device"],
        "inode": wrfbdy["inode"], "mode": "0444", "bytes": wrfbdy["size"],
        "sha256": wrfbdy["sha256"],
    }, "CANONICAL_LIVE_FD_FIELD")
    if type(live_fd.get("pid")) is not int or live_fd["pid"] <= 1:
        raise GateError("CANONICAL_LIVE_FD_PID", repr(live_fd.get("pid")))
    seal_payload = load_exact_json(INPUT_SEAL)
    pre_seal = contract.get("pre_wrf_seal") or {}
    if set(pre_seal) != {
        "path", "sha256", "payload", "canonical_resolution_pass", "launch_time_checksums",
    } or pre_seal.get("path") != str(INPUT_SEAL) or pre_seal.get("sha256") != PRE_WRF_SEAL_SHA256 or pre_seal.get("canonical_resolution_pass") is not True:
        raise GateError("CANONICAL_PRE_WRF_SEAL_ROLE", repr(pre_seal))
    if sha256_file(INPUT_SEAL) != PRE_WRF_SEAL_SHA256 or pre_seal.get("payload") != seal_payload:
        raise GateError("CANONICAL_PRE_WRF_SEAL", str(INPUT_SEAL))
    seal_provenance = seal_payload.get("provenance") if isinstance(seal_payload, dict) else None
    if not isinstance(seal_provenance, dict):
        raise GateError("CANONICAL_PRE_WRF_SEAL_PROVENANCE", repr(seal_provenance))
    if (
        binding.get("input_seal_checksums_sha256") != LAUNCH_TIME_CHECKSUMS_SHA256
        or binding.get("input_seal_checksums_sha256") != seal_provenance.get("checksums_sha256")
        or binding.get("input_seal_checksums_sha256") == CURRENT_CHECKSUMS_SHA256
    ):
        raise GateError(
            "CANONICAL_BINDING_CHECKSUM_ROLE",
            repr(binding.get("input_seal_checksums_sha256")),
        )
    if (
        binding.get("input_seal_launch_plan_sha256")
        != seal_provenance.get("launch_plan_sha256")
        or binding.get("input_seal_launch_plan_sha256") == LAUNCH_PLAN_SHA256
        or seal_provenance.get("launch_plan") == launch["payload"]
    ):
        raise GateError(
            "CANONICAL_BINDING_LAUNCH_PLAN_ROLE",
            repr(binding.get("input_seal_launch_plan_sha256")),
        )
    _validate_binding_file_record(
        binding["input_seal"], INPUT_SEAL, "CANONICAL_PRE_WRF_SEAL_RECORD",
        expected_record_path=compatibility / "input_seal.json",
    )
    launch_time = pre_seal.get("launch_time_checksums")
    if launch_time != {"path": str(LAUNCH_TIME_CHECKSUMS), "sha256": LAUNCH_TIME_CHECKSUMS_SHA256}:
        raise GateError("CANONICAL_LAUNCH_TIME_CHECKSUM_ROLE", repr(launch_time))
    if sha256_file(LAUNCH_TIME_CHECKSUMS) != LAUNCH_TIME_CHECKSUMS_SHA256:
        raise GateError("CANONICAL_LAUNCH_TIME_CHECKSUM_HASH", str(LAUNCH_TIME_CHECKSUMS))
    if sha256_file(CURRENT_CHECKSUMS) != CURRENT_CHECKSUMS_SHA256:
        raise GateError("CANONICAL_CURRENT_CHECKSUM_HASH", str(CURRENT_CHECKSUMS))
    if LAUNCH_TIME_CHECKSUMS.resolve() == CURRENT_CHECKSUMS.resolve() or LAUNCH_TIME_CHECKSUMS_SHA256 == CURRENT_CHECKSUMS_SHA256:
        raise GateError("CANONICAL_CHECKSUM_ROLE_COLLAPSE", "historical and current roles must differ")

    forcing, forcing_artifact = stable_json_authority(
        FORCING_PREFLIGHT, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    if forcing_artifact["sha256"] != FORCING_PREFLIGHT_SHA256 or contract.get("forcing_preflight") != {
        "path": str(FORCING_PREFLIGHT), "sha256": FORCING_PREFLIGHT_SHA256,
    }:
        raise GateError("CANONICAL_FORCING_MANIFEST", str(FORCING_PREFLIGHT))
    _exact_fields(forcing, {
        "schema": "tenerife_fullbuffer_wps_forcing_preflight_v2", "status": "pass",
        "case_id": CASE_ID, "boundary_interval_seconds": 21600,
        "mandatory_met_fields": ["TT", "UU", "VV", "SPECHUMD", "DEWPT", "PSFC", "SKINTEMP"],
    }, "CANONICAL_FORCING_FIELD")
    if forcing.get("source", {}).get("sha256") != AIFS_SHA256 or forcing.get("metgrid_qc", {}).get("file_count") != 12:
        raise GateError("CANONICAL_FORCING_LINEAGE", "source or metgrid inventory differs")

    runtime_path = CPU_RUN_ROOT / "launch_runtime.json"
    runtime = load_exact_json(runtime_path)
    _validate_binding_file_record(
        binding["launch_runtime"], runtime_path, "CANONICAL_LAUNCH_RUNTIME",
        expected_record_path=compatibility / "launch_runtime.json",
    )
    _exact_fields(runtime, {
        "status": "running", "stage": "wrf", "attempt_id": "attempt_repaired_v2",
        "ranks": 12, "cpuset": "0-11",
    }, "CANONICAL_LAUNCH_RUNTIME_FIELD")
    if (
        type(runtime.get("wrf_pid")) is not int or runtime["wrf_pid"] <= 1
        or type(runtime.get("wrf_start_ticks")) is not int
        or runtime["wrf_start_ticks"] <= 0
    ):
        raise GateError(
            "CANONICAL_LAUNCH_RUNTIME_IDENTITY",
            "PID/start-ticks must be positive integers in the authenticated launch_runtime record",
        )
    live_fd_rank = validate_contract_live_fd_rank(live_fd, runtime, sealed)
    prterun = validate_prterun_authority()
    result = {
        "contract": contract, "contract_artifact": contract_artifact,
        "binding": binding, "binding_artifact": binding_artifact,
        "launch_plan": launch, "forcing_artifact": forcing_artifact,
        "upstream": upstream, "sealed_inputs": sealed,
        "historical_provenance": {
            "launch_time_checksums": launch_time,
            "historical_launch_plan_sha256": binding["input_seal_launch_plan_sha256"],
            "current_checksums": {"path": str(CURRENT_CHECKSUMS), "sha256": CURRENT_CHECKSUMS_SHA256},
            "current_launch_plan": {"path": str(LAUNCH_PLAN), "sha256": LAUNCH_PLAN_SHA256},
        },
        "cpu_runtime": runtime, "live_fd_rank": live_fd_rank, "prterun": prterun,
    }
    result["authority_sha256"] = authority_digest(result)
    return result


def validate_identity_policy_authority() -> dict[str, Any]:
    manager_commit = subprocess.check_output(
        ("git", "-C", str(MANAGER_REPO), "rev-parse", MANAGER_POLICY_COMMIT), text=True,
    ).strip()
    manager_bytes = subprocess.check_output(
        ("git", "-C", str(MANAGER_REPO), "show", f"{manager_commit}:{MANAGER_POLICY_RELATIVE}"),
    )
    release_path = Path(__file__).resolve().parents[1] / RELEASE_POLICY
    release_bytes = subprocess.check_output(
        ("git", "show", f"{BASE_SHA}:{RELEASE_POLICY}"), cwd=Path(__file__).resolve().parents[1],
    )
    if (
        manager_commit != MANAGER_POLICY_COMMIT
        or sha256_bytes(manager_bytes) != MANAGER_POLICY_SHA256
        or sha256_file(MANAGER_POLICY) != MANAGER_POLICY_SHA256
        or sha256_bytes(release_bytes) != RELEASE_POLICY_SHA256
        or sha256_file(release_path) != RELEASE_POLICY_SHA256
    ):
        raise GateError("IDENTITY_POLICY_AUTHORITY", "manager or release policy changed")
    policy = {
        "manager_repo": str(MANAGER_REPO),
        "manager_commit": manager_commit,
        "manager_path": MANAGER_POLICY_RELATIVE,
        "manager_sha256": MANAGER_POLICY_SHA256,
        "release_candidate": BASE_SHA,
        "release_path": str(RELEASE_POLICY),
        "release_sha256": RELEASE_POLICY_SHA256,
        "aggregation": "pooled_rmse_across_all_55_exact_same_valid_d03_frames",
        "strict_fields": list(STRICT_FIELDS),
        "strict_rmse_limits": STRICT_RMSE_LIMITS,
        "owner_report_only_fields": list(REPORT_ONLY_FIELDS),
        "owner_report_only_status": "OWNER_DIRECTIVE_REPORT_ONLY",
        "untoleranced_status": "UNTOLERANCED_REPORT_ONLY",
        "static_exact_fields": list(STATIC_GATE_FIELDS),
    }
    policy["authority_sha256"] = authority_digest(policy)
    return policy


def _parse_cpu_set(value: str) -> set[int]:
    result: set[int] = set()
    for token in value.split(","):
        if not token:
            continue
        if "-" in token:
            lower, upper = (int(item) for item in token.split("-", 1))
            result.update(range(lower, upper + 1))
        else:
            result.add(int(token))
    return result


def validate_cpu_identity_payload(
    cpu: Mapping[str, Any], expected_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if set(cpu) != {"pid", "start_ticks", "state", "cpuset", "command", "descendants", "observed_wrf_rank_count"}:
        raise GateError("CPU_IDENTITY_SCHEMA", repr(sorted(cpu)))
    expected_runtime = expected_runtime or validate_canonical_authority_v2()["cpu_runtime"]
    exact_command = " ".join(PRTERUN_ARGV)
    if (
        not isinstance(cpu.get("pid"), int)
        or cpu["pid"] != expected_runtime.get("wrf_pid")
        or not isinstance(cpu.get("start_ticks"), int)
        or cpu["start_ticks"] != expected_runtime.get("wrf_start_ticks")
        or cpu.get("cpuset") != "0-11"
        or str(cpu.get("state", "")).startswith("Z")
        or cpu.get("command") != exact_command
        or cpu.get("observed_wrf_rank_count") != 12
    ):
        raise GateError("CPU_IDENTITY", repr(cpu))
    descendants = cpu.get("descendants")
    if not isinstance(descendants, list) or len(descendants) != 12:
        raise GateError("CPU_DESCENDANTS", repr(descendants))
    wrf_pids: set[int] = set()
    all_pids: set[int] = set()
    for row in descendants:
        if not isinstance(row, dict) or set(row) != {"pid", "cpuset", "command"}:
            raise GateError("CPU_DESCENDANT_SCHEMA", repr(row))
        pid = row.get("pid")
        if not isinstance(pid, int) or pid <= 1 or pid in all_pids:
            raise GateError("CPU_DESCENDANT_PID", repr(row))
        all_pids.add(pid)
        try:
            cpus = _parse_cpu_set(str(row.get("cpuset")))
        except ValueError as exc:
            raise GateError("CPU_DESCENDANT_CPUSET", repr(row)) from exc
        if not cpus or not cpus.issubset(set(CPU_LANE)):
            raise GateError("CPU_DESCENDANT_CPUSET", repr(row))
        if row.get("command") != "./wrf.exe":
            raise GateError("CPU_WRF_RANK_COMMAND", repr(row))
        if row.get("command") == "./wrf.exe":
            wrf_pids.add(pid)
    if len(wrf_pids) != 12:
        raise GateError("CPU_WRF_RANKS", f"expected exact 12, got {len(wrf_pids)}")
    return dict(cpu)


def _proc_stat_identity(pid: int) -> tuple[int, int]:
    text = Path(f"/proc/{pid}/stat").read_text()
    close = text.rfind(")")
    if close < 0:
        raise ValueError("missing stat comm terminator")
    fields = text[close + 2 :].split()
    return int(fields[19]), int(fields[1])


def _is_descendant_of(pid: int, root_pid: int) -> tuple[bool, list[int]]:
    ancestry: list[int] = []
    current = pid
    for _ in range(128):
        if current == root_pid:
            return True, ancestry
        if current <= 1 or current in ancestry:
            return False, ancestry
        ancestry.append(current)
        _start, current = _proc_stat_identity(current)
    return False, ancestry


def _live_cpu_proc(pid: int) -> dict[str, Any]:
    start_ticks, parent_pid = _proc_stat_identity(pid)
    status = {
        key: value.strip()
        for key, value in (
            line.split(":", 1)
            for line in Path(f"/proc/{pid}/status").read_text().splitlines()
            if ":" in line
        )
    }
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    return {
        "start_ticks": start_ticks, "parent_pid": parent_pid,
        "status": status, "command": command,
    }


def validate_contract_live_fd_rank(
    live_fd: Mapping[str, Any],
    runtime: Mapping[str, Any],
    sealed_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the contract holder to the authenticated live WRF process tree."""
    holder_pid = live_fd.get("pid")
    root_pid = runtime.get("wrf_pid")
    if type(holder_pid) is not int or holder_pid <= 1 or type(root_pid) is not int or root_pid <= 1:
        raise GateError("CANONICAL_LIVE_FD_PID", repr((holder_pid, root_pid)))
    try:
        root = _live_cpu_proc(root_pid)
        holder = _live_cpu_proc(holder_pid)
        descendant, ancestry = _is_descendant_of(holder_pid, root_pid)
        root_cpus = _parse_cpu_set(root["status"].get("Cpus_allowed_list", ""))
        holder_cpus = _parse_cpu_set(holder["status"].get("Cpus_allowed_list", ""))
    except (OSError, UnicodeDecodeError, ValueError, IndexError, TypeError) as exc:
        raise GateError("CANONICAL_LIVE_FD_RANK", repr(holder_pid)) from exc
    if (
        root["start_ticks"] != runtime.get("wrf_start_ticks")
        or root["command"].strip() != " ".join(PRTERUN_ARGV)
        or root_cpus != set(CPU_LANE)
        or holder["command"].strip() != "./wrf.exe"
        or not holder_cpus or not holder_cpus.issubset(set(CPU_LANE))
        or holder["start_ticks"] <= 0
        or not descendant
    ):
        raise GateError("CANONICAL_LIVE_FD_RANK", repr(holder_pid))
    wrfbdy = sealed_inputs.get("wrfbdy_d01")
    if not isinstance(wrfbdy, Mapping):
        raise GateError("CANONICAL_LIVE_FD_DESCRIPTOR", "missing wrfbdy authority")
    observation = _stable_relevant_fd_observation(
        Path(f"/proc/{holder_pid}"), Path(f"/proc/{holder_pid}/fd/{live_fd['fd']}"),
        {(wrfbdy["device"], wrfbdy["inode"]): "wrfbdy_d01"},
        {str(Path(str(wrfbdy["path"])).resolve(strict=True)): "wrfbdy_d01"},
    )
    if observation is None:
        raise GateError("CANONICAL_LIVE_FD_DESCRIPTOR", repr((holder_pid, live_fd["fd"])))
    validate_input_fd_observations({"wrfbdy_d01": wrfbdy}, [observation])
    if (
        observation["pid"] != live_fd["pid"]
        or observation["process_start_ticks"] != holder["start_ticks"]
        or observation["fd"] != live_fd["fd"]
        or observation["flags"] != int(live_fd["flags_octal"], 8)
        or observation["access"] != live_fd["access"]
        or observation["device"] != live_fd["device"]
        or observation["inode"] != live_fd["inode"]
        or observation["path_identity_match"] is not True
    ):
        raise GateError("CANONICAL_LIVE_FD_DESCRIPTOR", repr(observation))
    return {
        "pid": holder_pid, "start_ticks": holder["start_ticks"],
        "root_pid": root_pid, "root_start_ticks": root["start_ticks"],
        "ancestry_to_authenticated_root": ancestry,
        "command": "./wrf.exe", "cpuset": holder["status"].get("Cpus_allowed_list"),
        "fd_observation": observation,
    }


def validate_live_prterun_executable(pid: int) -> str:
    try:
        resolved = Path(f"/proc/{pid}/exe").resolve(strict=True)
    except OSError as exc:
        raise GateError("CPU_PRTERUN_EXECUTABLE", str(pid)) from exc
    if str(resolved) != PRTE_RESOLVED:
        raise GateError("CPU_PRTERUN_EXECUTABLE", str(resolved))
    return str(resolved)


def verify_live_cpu_identity(
    cpu: Mapping[str, Any], expected_live: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recheck root and every rank PID start tick, affinity, and ancestry."""
    canonical = validate_canonical_authority_v2()
    validated = validate_cpu_identity_payload(cpu, canonical["cpu_runtime"])
    pid = validated["pid"]
    try:
        root = _live_cpu_proc(pid)
    except (OSError, ValueError, IndexError) as exc:
        raise GateError("CPU_PID_NOT_LIVE", str(pid)) from exc
    if (
        root["start_ticks"] != validated["start_ticks"]
        or _parse_cpu_set(root["status"].get("Cpus_allowed_list", "")) != set(CPU_LANE)
        or root["command"].strip() != " ".join(PRTERUN_ARGV)
        or validate_live_prterun_executable(pid) != PRTE_RESOLVED
    ):
        raise GateError("CPU_PID_IDENTITY_CHANGED", str(pid))
    expected_ranks = {
        row["pid"] for row in validated["descendants"] if "wrf.exe" in row["command"]
    }
    live_ranks: list[dict[str, Any]] = []
    for rank_pid in expected_ranks:
        try:
            rank = _live_cpu_proc(rank_pid)
            descendant, ancestry = _is_descendant_of(rank_pid, pid)
        except (OSError, UnicodeDecodeError, ValueError, IndexError) as exc:
            raise GateError("CPU_RANK_NOT_LIVE", str(rank_pid)) from exc
        if (
            rank["command"].strip() != "./wrf.exe"
            or not _parse_cpu_set(rank["status"].get("Cpus_allowed_list", "")).issubset(set(CPU_LANE))
            or not descendant
        ):
            raise GateError("CPU_RANK_IDENTITY_CHANGED", str(rank_pid))
        live_ranks.append({
            "pid": rank_pid,
            "start_ticks": rank["start_ticks"],
            "parent_pid": rank["parent_pid"],
            "ancestry_to_root": ancestry,
            "cpuset": rank["status"].get("Cpus_allowed_list"),
        })
    live_ranks.sort(key=lambda row: row["pid"])
    if {row["pid"] for row in live_ranks} != expected_ranks or len(live_ranks) != 12:
        raise GateError("CPU_WRF_RANKS_LIVE", repr(live_ranks))
    result = {
        "pid": pid,
        "start_ticks": root["start_ticks"],
        "root_cpuset": "0-11",
        "ranks": live_ranks,
    }
    if expected_live is not None and result != expected_live:
        raise GateError("CPU_LIVE_IDENTITY_CHANGED", "rank PID/start/ancestry changed")
    return result


def validate_input_fd_observations(
    input_authority: Mapping[str, Mapping[str, Any]], observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(input_authority, Mapping) or not isinstance(observations, Sequence):
        raise GateError("INPUT_FD_SCHEMA", "authority/observations container")
    by_identity: dict[tuple[int, int], str] = {}
    for name, value in input_authority.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise GateError("INPUT_FD_AUTHORITY_SCHEMA", repr((name, value)))
        device, inode = value.get("device"), value.get("inode")
        if (
            type(device) is not int or device <= 0
            or type(inode) is not int or inode <= 0
            or (device, inode) in by_identity
        ):
            raise GateError("INPUT_FD_AUTHORITY_IDENTITY", repr((name, device, inode)))
        by_identity[(device, inode)] = name
    holders: dict[str, list[dict[str, Any]]] = {name: [] for name in input_authority}
    for row in observations:
        if not isinstance(row, Mapping) or set(row) != {
            "pid", "process_start_ticks", "fd", "flags", "access", "device", "inode",
            "path_identity_match",
        }:
            raise GateError("INPUT_FD_SCHEMA", repr(row))
        pid, start_ticks, fd = row["pid"], row["process_start_ticks"], row["fd"]
        flags, device, inode = row["flags"], row["device"], row["inode"]
        if (
            type(pid) is not int or pid <= 1
            or type(start_ticks) is not int or start_ticks <= 0
            or type(fd) is not int or fd < 0
            or type(flags) is not int or flags < 0
            or type(device) is not int or device <= 0
            or type(inode) is not int or inode <= 0
            or type(row["access"]) is not str
            or type(row["path_identity_match"]) is not bool
        ):
            raise GateError("INPUT_FD_TYPE_OR_RANGE", repr(row))
        identity = (device, inode)
        name = by_identity.get(identity)
        if row["path_identity_match"] is not True:
            raise GateError("INPUT_FD_INODE_MISMATCH", repr(row))
        if name is None:
            raise GateError("INPUT_FD_INODE_MISMATCH", repr(row))
        access_bits = flags & os.O_ACCMODE
        derived_access = (
            "read_only" if access_bits == os.O_RDONLY
            else "write_only" if access_bits == os.O_WRONLY
            else "read_write"
        )
        if row["access"] != derived_access or derived_access != "read_only":
            raise GateError("INPUT_FD_WRITABLE", repr(row))
        holders[name].append(dict(row))
    return {
        "targets": {
            name: {
                "path": value["path"], "device": value["device"], "inode": value["inode"],
                "mode": value["mode"], "sha256": value["sha256"],
                "holders": sorted(holders[name], key=lambda row: (row["pid"], row["fd"])),
            }
            for name, value in input_authority.items()
        },
        "seal_time_open_by_pids": {name: [] for name in input_authority},
        "current_policy": "read_only_exact_inode_holders_only",
    }


def _read_fd_flags(fdinfo_path: Path) -> int:
    text = fdinfo_path.read_text()
    try:
        flag_text = next(
            line.split(":", 1)[1].strip()
            for line in text.splitlines() if line.startswith("flags:")
        )
        flags = int(flag_text, 8)
    except (StopIteration, ValueError, IndexError) as exc:
        raise GateError("INPUT_FD_FLAGS", str(fdinfo_path)) from exc
    if flags < 0:
        raise GateError("INPUT_FD_FLAGS", str(fdinfo_path))
    return flags


def _fd_target_snapshot(process: Path, descriptor: Path) -> dict[str, Any]:
    if not process.name.isdigit() or not descriptor.name.isdigit():
        raise GateError("INPUT_FD_PROC_IDENTITY", f"{process}/{descriptor.name}")
    pid, fd = int(process.name), int(descriptor.name)
    start_ticks, _parent = _proc_stat_identity(pid)
    info = descriptor.stat()
    target = os.readlink(descriptor)
    return {
        "pid": pid, "process_start_ticks": start_ticks, "fd": fd,
        "device": info.st_dev, "inode": info.st_ino, "target": target,
    }


def _fd_snapshot(process: Path, descriptor: Path) -> dict[str, Any]:
    snapshot = _fd_target_snapshot(process, descriptor)
    snapshot["flags"] = _read_fd_flags(process / "fdinfo" / descriptor.name)
    return snapshot


def _stable_relevant_fd_observation(
    process: Path,
    descriptor: Path,
    identities: Mapping[tuple[int, int], str],
    canonical_paths: Mapping[str, str],
) -> dict[str, Any] | None:
    try:
        anchor = _fd_target_snapshot(process, descriptor)
    except (OSError, ValueError, IndexError, GateError):
        return None
    identity = (anchor["device"], anchor["inode"])
    path_name = canonical_paths.get(anchor["target"])
    if identity not in identities and path_name is None:
        return None
    try:
        first = _fd_snapshot(process, descriptor)
        second = _fd_snapshot(process, descriptor)
    except (OSError, ValueError, IndexError, GateError) as exc:
        raise GateError(
            "INPUT_FD_CHANGED_DURING_SCAN", f"{process.name}/{descriptor.name}: inaccessible",
        ) from exc
    if any(anchor[key] != first[key] for key in anchor) or first != second:
        raise GateError(
            "INPUT_FD_CHANGED_DURING_SCAN", f"{process.name}/{descriptor.name}: reused or raced",
        )
    path_identity_match = path_name is None or identities.get(identity) == path_name
    access_bits = first["flags"] & os.O_ACCMODE
    access = (
        "read_only" if access_bits == os.O_RDONLY
        else "write_only" if access_bits == os.O_WRONLY
        else "read_write"
    )
    return {
        "pid": first["pid"], "process_start_ticks": first["process_start_ticks"],
        "fd": first["fd"], "flags": first["flags"], "access": access,
        "device": first["device"], "inode": first["inode"],
        "path_identity_match": path_identity_match,
    }


def scan_current_input_open_fds(input_authority: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    # Validate the target authority even when there are no open holders.
    validate_input_fd_observations(input_authority, [])
    identities = {(value["device"], value["inode"]): name for name, value in input_authority.items()}
    canonical_paths = {str(Path(value["path"]).resolve(strict=True)): name for name, value in input_authority.items()}
    observations: list[dict[str, Any]] = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            observation = _stable_relevant_fd_observation(
                process, descriptor, identities, canonical_paths,
            )
            if observation is not None:
                observations.append(observation)
    return validate_input_fd_observations(input_authority, observations)


def validate_input_seal(
    marker: Mapping[str, Any],
    seal_path: Path,
    *,
    interval_seconds: float,
    sleep_fn: Callable[[float], None],
    canonical_authority: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical = canonical_authority or validate_canonical_authority_v2()
    seal, seal_artifact = stable_json_authority(
        seal_path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    if seal_path.resolve(strict=True) != INPUT_SEAL.resolve(strict=True):
        raise GateError("INPUT_SEAL_CANONICAL_PATH", str(seal_path))
    if set(seal) != {
        "schema", "status", "sealed_utc", "case_id", "grid_id", "grid_sha256",
        "immutability", "files", "provenance",
    }:
        raise GateError("INPUT_SEAL_SCHEMA", repr(sorted(seal)))
    _exact_fields(seal, {
        "schema": UPSTREAM_SEAL_SCHEMA,
        "status": "sealed_immutable_monitored",
        "case_id": CASE_ID,
        "grid_id": GRID_ID,
        "grid_sha256": GRID_SHA256,
        "immutability": "mode_0444_plus_continuous_sha256_monitor_until_cpu_exit",
    }, "INPUT_SEAL_FIELD")
    parse_utc(str(seal.get("sealed_utc")))
    if str(marker.get("input_seal_path")) != str(INPUT_SEAL) or seal_path != INPUT_SEAL:
        raise GateError("INPUT_SEAL_PATH", str(marker.get("input_seal_path")))
    if marker.get("provenance") != seal.get("provenance"):
        raise GateError("INPUT_PROVENANCE_SUBSTITUTION", "marker and seal differ")
    records = seal.get("files")
    if not isinstance(records, list) or len(records) != len(INPUT_NAMES):
        raise GateError("INPUT_SEAL_COUNT", repr(records))
    input_authority: dict[str, Any] = {}
    input_paths: list[Path] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "path", "sha256", "bytes", "mtime_ns", "mode", "open_by_pids"}:
            raise GateError("INPUT_RECORD_SCHEMA", repr(record))
        name = record.get("name")
        if name not in INPUT_NAMES or name in input_authority:
            raise GateError("INPUT_RECORD_NAME", repr(name))
        path = Path(str(record.get("path")))
        expected_path = CPU_INPUT_DIR / str(name)
        observed = _regular_file_stat(path)
        _regular_file_stat(expected_path)
        expected_historical_path = CPU_CASE_ROOT / "attempts/attempt_repaired_v2/wrf" / str(name)
        if str(path) != str(expected_historical_path) or path.resolve(strict=True) != expected_path:
            raise GateError("INPUT_RECORD_PATH", str(path))
        if stat.S_IMODE(observed.st_mode) != 0o444 or record.get("mode") != "-r--r--r--" or record.get("open_by_pids") != []:
            raise GateError("INPUT_IMMUTABILITY", repr(record))
        digest = sha256_file(path)
        if digest != record.get("sha256") or observed.st_size != record.get("bytes") or observed.st_mtime_ns != record.get("mtime_ns"):
            raise GateError("INPUT_MUTATION", str(path))
        input_authority[str(name)] = {
            "path": str(path.resolve(strict=True)), "sha256": digest,
            "size": observed.st_size, "device": observed.st_dev, "inode": observed.st_ino,
            "mtime_ns": observed.st_mtime_ns, "ctime_ns": observed.st_ctime_ns,
            "mode": 0o444,
        }
        input_paths.append(path)
    stable_inputs = stable_file_authorities(
        input_paths, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    for name, detail in input_authority.items():
        stable = stable_inputs[name]
        if (
            stable["sha256"] != detail["sha256"]
            or stable["size"] != detail["size"]
            or stable["mtime_ns"] != detail["mtime_ns"]
            or stable["ctime_ns"] != detail["ctime_ns"]
            or stable["device"] != detail["device"]
            or stable["inode"] != detail["inode"]
            or stable["mode"] != 0o444
        ):
            raise GateError("INPUT_STABILITY_BINDING", name)
    if set(input_authority) != set(INPUT_NAMES) or marker.get("input_hashes") != {name: input_authority[name]["sha256"] for name in INPUT_NAMES}:
        raise GateError("INPUT_HASH_BINDING", repr(marker.get("input_hashes")))
    provenance = seal.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "checksums_path", "checksums_sha256", "launch_plan_path",
        "launch_plan_sha256", "launch_plan",
    }:
        raise GateError("PROVENANCE_SCHEMA", repr(provenance))
    historical = canonical["contract"]["pre_wrf_seal"]["payload"]["provenance"]
    if provenance != historical:
        raise GateError("PROVENANCE_HISTORICAL_PAYLOAD", "seal provenance differs from v2 contract")
    _exact_fields(provenance, {
        "launch_plan_path": str(LAUNCH_PLAN),
        "launch_plan_sha256": canonical["binding"]["input_seal_launch_plan_sha256"],
        "checksums_path": str(CURRENT_CHECKSUMS),
        "checksums_sha256": LAUNCH_TIME_CHECKSUMS_SHA256,
    }, "PROVENANCE_HISTORICAL_FIELD")
    if provenance["launch_plan_sha256"] == LAUNCH_PLAN_SHA256 or provenance["launch_plan"] == canonical["launch_plan"]["payload"]:
        raise GateError("PROVENANCE_FALSE_CURRENT_EQUIVALENCE", "historical launch plan must remain distinct")
    if sha256_file(LAUNCH_TIME_CHECKSUMS) != provenance["checksums_sha256"]:
        raise GateError("PROVENANCE_LAUNCH_TIME_CHECKSUMS", str(LAUNCH_TIME_CHECKSUMS))
    if sha256_file(CURRENT_CHECKSUMS) != CURRENT_CHECKSUMS_SHA256:
        raise GateError("PROVENANCE_CURRENT_CHECKSUMS", str(CURRENT_CHECKSUMS))
    live_fd_evidence = scan_current_input_open_fds(input_authority)
    for name, detail in input_authority.items():
        detail["seal_time_open_by_pids"] = []
        detail["current_open_fd_evidence"] = live_fd_evidence["targets"][name]["holders"]
    return seal, seal_artifact, input_authority


def validate_resource_proof(
    marker: Mapping[str, Any],
    resource_path: Path,
    *,
    interval_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resource, artifact = stable_json_authority(
        resource_path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    if resource_path != RESOURCE_PROOF:
        raise GateError("RESOURCE_CANONICAL_PATH", str(resource_path))
    if set(resource) != {
        "schema", "status", "checked_utc", "cpu_oracle", "cpu_auxiliary",
        "gpu_lock", "gpu_device", "preemption",
    }:
        raise GateError("RESOURCE_SCHEMA", repr(sorted(resource)))
    _exact_fields(resource, {"schema": UPSTREAM_RESOURCE_SCHEMA, "status": "pass"}, "RESOURCE_FIELD")
    if str(marker.get("resource_proof_path")) != str(RESOURCE_PROOF) or marker.get("resource_proof") != resource:
        raise GateError("RESOURCE_PROOF_BINDING", "embedded and file proof differ")
    cpu = validate_cpu_identity_payload(resource.get("cpu_oracle") or {})
    if marker.get("cpu_oracle") != cpu:
        raise GateError("RESOURCE_CPU_BINDING", "CPU identity differs")
    auxiliary = resource.get("cpu_auxiliary") or {}
    if set(auxiliary) != {"cpuset", "free", "proof_method", "compute_blockers"}:
        raise GateError("AUXILIARY_CPU_SCHEMA", repr(sorted(auxiliary)))
    _exact_fields(auxiliary, {"cpuset": "12-15", "free": True, "compute_blockers": []}, "AUXILIARY_CPU_FIELD")
    lock = resource.get("gpu_lock") or {}
    if set(lock) != {"path", "free", "authority", "device", "inode", "holder_sidecar_at_probe"}:
        raise GateError("GPU_LOCK_RESOURCE_SCHEMA", repr(sorted(lock)))
    _exact_fields(lock, {
        "path": "/tmp/wrf_gpu2_gpu.lock", "free": True,
        "authority": "kernel_flock_exclusive_nonblocking_probe",
    }, "GPU_LOCK_RESOURCE_FIELD")
    if not isinstance(lock.get("device"), int) or not isinstance(lock.get("inode"), int):
        raise GateError("GPU_LOCK_IDENTITY", repr(lock))
    device = resource.get("gpu_device") or {}
    if set(device) != {
        "compute_query", "gpu_query", "gpu", "baseline_compute_processes",
        "unexpected_compute_processes", "minimum_free_mib",
        "maximum_utilization_percent", "free",
    }:
        raise GateError("GPU_DEVICE_SCHEMA", repr(sorted(device)))
    _exact_fields(device, {
        "free": True, "unexpected_compute_processes": [],
        "minimum_free_mib": 24000, "maximum_utilization_percent": 20,
        "compute_query": [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        "gpu_query": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    }, "GPU_DEVICE_FIELD")
    gpu = device.get("gpu") or {}
    if set(gpu) != {
        "index", "name", "memory_total_mib", "memory_used_mib",
        "memory_free_mib", "utilization_percent",
    }:
        raise GateError("GPU_OBJECT_SCHEMA", repr(sorted(gpu)))
    if (
        not isinstance(gpu.get("index"), int)
        or not isinstance(gpu.get("name"), str) or not gpu["name"]
        or any(not isinstance(gpu.get(key), int) for key in (
            "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_percent",
        ))
        or gpu["memory_total_mib"] <= 0
        or gpu["memory_used_mib"] < 0
        or gpu["memory_free_mib"] < 24000
        or gpu["utilization_percent"] > 20
    ):
        raise GateError("GPU_DEVICE_NUMBERS", repr(gpu))
    preempt = resource.get("preemption") or {}
    if set(preempt) != {
        "checked_utc", "sentinels_checked", "sentinels_present",
        "nightly_active_path", "nightly_active",
    }:
        raise GateError("PREEMPT_RESOURCE_SCHEMA", repr(sorted(preempt)))
    _exact_fields(preempt, {
        "sentinels_checked": [
            "/tmp/PREEMPT_PRODUCTION", "/tmp/PREEMPT_GPU",
            "<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION",
            "<DATA_ROOT>/alisios/state/PREEMPT_GPU",
        ],
        "sentinels_present": [],
        "nightly_active_path": "<DATA_ROOT>/alisios/state/nightly18z/active.json",
        "nightly_active": False,
    }, "PREEMPT_RESOURCE_FIELD")
    parse_utc(str(resource.get("checked_utc")))
    parse_utc(str(preempt.get("checked_utc")))
    return resource, artifact


def canonical_gpu_baseline_identity(resource: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = resource.get("gpu_device", {}).get("baseline_compute_processes")
    if not isinstance(rows, list):
        raise GateError("GPU_BASELINE_AUTHORITY", repr(rows))
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("baseline_allowed") is not True:
            raise GateError("GPU_BASELINE_AUTHORITY", repr(row))
        pid = int(row.get("pid", -1))
        name = str(row.get("process_name", ""))
        identity = (pid, name)
        if pid <= 1 or not name or identity in seen:
            raise GateError("GPU_BASELINE_AUTHORITY", repr(row))
        seen.add(identity)
        result.append({"pid": pid, "process_name": name, **live_process_binding(pid)})
    return sorted(result, key=lambda item: (item["pid"], item["process_name"]))


def live_process_binding(pid: int) -> dict[str, Any]:
    try:
        start_ticks, _parent = _proc_stat_identity(pid)
        argv0 = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="strict")
        exe_path = Path(f"/proc/{pid}/exe").resolve(strict=True)
        exe_info = exe_path.stat()
    except (OSError, UnicodeDecodeError, ValueError, IndexError) as exc:
        raise GateError("GPU_BASELINE_PID_NOT_LIVE", str(pid)) from exc
    return {
        "start_ticks": start_ticks, "argv0": argv0, "exe_path": str(exe_path),
        "exe_device": exe_info.st_dev, "exe_inode": exe_info.st_ino,
    }


def validate_early_groups(
    marker: Mapping[str, Any],
    *,
    interval_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = marker.get("complete_regular_d03_groups")
    if not isinstance(groups, list) or [row.get("valid_time") for row in groups if isinstance(row, dict)] != list(EARLY_REQUIRED_STAMPS):
        raise GateError("EARLY_GROUP_TIMES", repr(groups))
    qa_rows: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    for group, valid in zip(groups, EARLY_REQUIRED_TIMES, strict=True):
        if set(group) != {"valid_time", "raw_path", "qa_evidence", "bytes"}:
            raise GateError("EARLY_GROUP_SCHEMA", repr(group))
        raw_path = Path(str(group["raw_path"]))
        expected_path = CPU_INPUT_DIR / f"wrfout_d03_{group['valid_time']}"
        observed = _regular_file_stat(raw_path)
        _regular_file_stat(expected_path)
        if raw_path.resolve(strict=True) != expected_path.resolve(strict=True):
            raise GateError("EARLY_GROUP_PATH", str(raw_path))
        qa_path = Path(str(group["qa_evidence"]))
        evidence, evidence_artifact = stable_json_authority(
            qa_path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
        )
        if set(evidence) != {
            "schema", "checked_utc", "path", "bytes", "mtime_ns", "closed",
            "readable", "status", "valid_time", "variables", "ranges",
        }:
            raise GateError("EARLY_QA_SCHEMA", repr(sorted(evidence)))
        _exact_fields(evidence, {
            "schema": UPSTREAM_QA_SCHEMA, "status": "pass",
            "valid_time": group["valid_time"], "path": str(raw_path),
            "bytes": group["bytes"], "closed": True, "readable": True,
            "variables": list(STRICT_FIELDS),
        }, "EARLY_QA_FIELD")
        if observed.st_size != group["bytes"] or observed.st_mtime_ns != evidence.get("mtime_ns"):
            raise GateError("EARLY_FRAME_STAT", str(raw_path))
        frame_stability = stable_file_authorities(
            [raw_path], interval_seconds=interval_seconds, sleep_fn=sleep_fn,
        )[raw_path.name]
        independent = qa_d03_frame(raw_path, valid)
        ranges = evidence.get("ranges")
        if not isinstance(ranges, dict) or set(ranges) != set(STRICT_FIELDS):
            raise GateError("EARLY_QA_RANGES", repr(ranges))
        for field, bounds in ranges.items():
            if not isinstance(bounds, list) or len(bounds) != 2 or not all(isinstance(number, (int, float)) and math.isfinite(number) for number in bounds):
                raise GateError("EARLY_QA_RANGE_VALUE", f"{field}={bounds!r}")
            if [float(number) for number in bounds] != independent["ranges"][field]:
                raise GateError("EARLY_QA_RANGE_SUBSTITUTION", f"{field}={bounds!r}")
        parse_utc(str(evidence.get("checked_utc")))
        qa_rows.append({"upstream": evidence, "artifact": evidence_artifact, "independent": independent})
        stability[group["valid_time"]] = frame_stability
    return qa_rows, stability


def reject_revoked_authority(marker_path: Path) -> None:
    run_root = marker_path.parent
    revoked = run_root / EARLY_REVOKED.name
    invalidated = run_root / EARLY_INVALIDATED.name
    if revoked.exists():
        payload = load_exact_json(revoked)
        _exact_fields(payload, {
            "schema": UPSTREAM_REVOKE_SCHEMA,
            "status": "revoked_gpu_stop_required",
            "gpu_action": "STOP_FAIL_CLOSED_AND_RELEASE_LOCK",
        }, "EARLY_REVOKED_FIELD")
        raise GateError("EARLY_REVOKED_STOP_AND_RELEASE", str(revoked))
    if invalidated.exists():
        raise GateError("EARLY_INVALIDATED_STOP_AND_RELEASE", str(invalidated))


def require_current_no_preemption() -> dict[str, Any]:
    present = [str(path) for path in PREEMPT_PATHS if path.exists()]
    present.extend(
        str(path) for path in (
            Path("<DATA_ROOT>/alisios/state/PREEMPT_GPU"),
        ) if path.exists()
    )
    active_path = Path("<DATA_ROOT>/alisios/state/nightly18z/active.json")
    active = load_exact_json(active_path) if active_path.exists() else None
    if present or (active and active.get("active") is True):
        raise GateError(
            "CURRENT_PREEMPT_STOP_AND_RELEASE",
            f"sentinels={sorted(present)} nightly_active={bool(active and active.get('active') is True)}",
        )
    return {
        "sentinels_present": [],
        "nightly_active": False,
        "nightly_active_path": str(active_path),
    }


def require_fresh_time(
    label: str, observed: datetime, now: datetime, maximum_age_seconds: float,
) -> None:
    age = (now - observed).total_seconds()
    if age < 0.0 or age > maximum_age_seconds:
        raise GateError("STALE_AUTHORITY", f"{label}: age_seconds={age:.3f} max={maximum_age_seconds}")


def build_early_ready_proof(
    *,
    marker_path: Path,
    work_dir: Path,
    interval_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    live_identity_checker: Callable[[Mapping[str, Any]], Mapping[str, Any]] = verify_live_cpu_identity,
    gpu_baseline_checker: Callable[[Mapping[str, Any]], list[dict[str, Any]]] = canonical_gpu_baseline_identity,
    existing_table_authority: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if marker_path != EARLY_MARKER:
        raise GateError("EARLY_MARKER_CANONICAL_PATH", str(marker_path))
    reject_revoked_authority(marker_path)
    current_preemption = require_current_no_preemption()
    marker, marker_artifact = stable_json_authority(
        marker_path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    if set(marker) != {
        "schema", "status", "emitted_utc", "case_id", "grid_id", "grid_sha256",
        "input_seal_path", "input_hashes", "provenance", "cpu_oracle",
        "complete_regular_d03_groups", "initialization_group_counted",
        "resource_proof_path", "resource_proof", "full_cpu_verdict", "gpu_contract",
        "callback_targets",
    }:
        raise GateError("EARLY_MARKER_SCHEMA", repr(sorted(marker)))
    _exact_fields(marker, {
        "schema": UPSTREAM_MARKER_SCHEMA, "status": "EARLY_GPU_READY",
        "case_id": CASE_ID, "grid_id": GRID_ID, "grid_sha256": GRID_SHA256,
        "initialization_group_counted": False,
        "full_cpu_verdict": "pending_55_of_55",
        "gpu_contract": "consumer_must_acquire_authoritative_gpu_lock; final CPU QA and plot gates remain mandatory",
        "callback_targets": ["0:fable-mgr.0", "0:gpt-mgr.0"],
    }, "EARLY_MARKER_FIELD")
    emitted = parse_utc(str(marker.get("emitted_utc")))
    require_fresh_time("marker.emitted_utc", emitted, now, MARKER_MAX_AGE_SECONDS)
    canonical_v2 = validate_canonical_authority_v2()
    authority = canonical_v2["upstream"]
    launch = canonical_v2["launch_plan"]
    seal_path = Path(str(marker["input_seal_path"]))
    seal, seal_artifact, inputs = validate_input_seal(
        marker, seal_path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
        canonical_authority=canonical_v2,
    )
    namelist = validate_corrected_namelist(CPU_INPUT_DIR / "namelist.input")
    resource_path = Path(str(marker["resource_proof_path"]))
    resource, resource_artifact = validate_resource_proof(
        marker, resource_path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    resource_checked = parse_utc(str(resource["checked_utc"]))
    preempt_checked = parse_utc(str(resource["preemption"]["checked_utc"]))
    require_fresh_time("resource.checked_utc", resource_checked, now, RESOURCE_MAX_AGE_SECONDS)
    require_fresh_time("preemption.checked_utc", preempt_checked, now, PREEMPT_MAX_AGE_SECONDS)
    live_identity = dict(live_identity_checker(marker["cpu_oracle"]))
    gpu_baseline_identity = gpu_baseline_checker(resource)
    groups, frame_stability = validate_early_groups(
        marker, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    qa_checked = [parse_utc(str(row["upstream"]["checked_utc"])) for row in groups]
    for index, checked in enumerate(qa_checked):
        require_fresh_time(f"qa[{index}].checked_utc", checked, now, QA_MAX_AGE_SECONDS)
    sealed = parse_utc(str(seal["sealed_utc"]))
    if not (
        sealed <= min(qa_checked)
        and max(qa_checked) <= resource_checked
        and resource_checked <= preempt_checked
        and preempt_checked <= emitted
    ):
        raise GateError(
            "AUTHORITY_TIME_ORDER",
            f"seal={sealed} qa={qa_checked} resource={resource_checked} "
            f"preempt={preempt_checked} marker={emitted}",
        )
    if existing_table_authority is None:
        table_authority = materialize_table_snapshot(WRF_SOURCE_ROOT, work_dir)
    else:
        table_authority = dict(existing_table_authority)
        verify_table_snapshot(table_authority, expected_work_dir=work_dir)
    identity_policy = validate_identity_policy_authority()
    binding = {
        "schema": EARLY_SCHEMA,
        "status": "EARLY_TWO_FRAME_READY",
        "case_id": CASE_ID,
        "grid_id": GRID_ID,
        "grid_sha256": GRID_SHA256,
        "candidate_sha": BASE_SHA,
        "created_utc": now.isoformat(),
        "stability_interval_seconds": interval_seconds,
        "upstream_authority": authority,
        "canonical_authority_v2": canonical_v2,
        "upstream_marker_artifact": marker_artifact,
        "upstream_marker": marker,
        "input_seal": {"artifact": seal_artifact, "payload": seal},
        "resource_proof": {"artifact": resource_artifact, "payload": resource},
        "current_preemption_recheck": current_preemption,
        "launch_plan": launch,
        "cpu_input_dir": str(CPU_INPUT_DIR.resolve(strict=True)),
        "cpu_live_identity": live_identity,
        "gpu_baseline_identity": gpu_baseline_identity,
        "input_authority": inputs,
        "namelist_authority": namelist,
        "early_frame_stability": frame_stability,
        "early_frame_qa": groups,
        "table_authority": table_authority,
        "identity_policy": identity_policy,
        "revocation_action": "STOP_FAIL_CLOSED_AND_RELEASE_LOCK",
        "final_verdict": "FINAL_PENDING_CPU_55_OF_55",
        "gpu_command_executed": False,
    }
    canonical = {
        "candidate_sha": BASE_SHA,
        "marker": marker_artifact,
        "resource": resource_artifact,
        "seal": seal_artifact,
        "qa": [row["artifact"] for row in groups],
        "input_authority": inputs,
        "cpu_live_identity": live_identity,
        "gpu_baseline_identity": gpu_baseline_identity,
        "launch_plan_sha256": launch["sha256"],
        "canonical_v2_contract_sha256": canonical_v2["contract_artifact"]["sha256"],
        "canonical_binding_sha256": canonical_v2["binding_artifact"]["sha256"],
        "table_authority_sha256": table_authority["authority_sha256"],
        "identity_policy_sha256": identity_policy["authority_sha256"],
        "freshness_limits_seconds": {
            "marker": MARKER_MAX_AGE_SECONDS,
            "resource": RESOURCE_MAX_AGE_SECONDS,
            "preemption": PREEMPT_MAX_AGE_SECONDS,
            "qa": QA_MAX_AGE_SECONDS,
            "derived_proof": DERIVED_PROOF_MAX_AGE_SECONDS,
        },
    }
    binding["canonical_authority_sha256"] = authority_digest(canonical)
    binding["authority_sha256"] = authority_digest(binding)
    return binding


def verify_early_ready_proof(
    path: Path,
    *,
    expected_work_dir: Path,
    now: datetime | None = None,
    interval_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    live_identity_checker: Callable[[Mapping[str, Any]], Mapping[str, Any]] = verify_live_cpu_identity,
    gpu_baseline_checker: Callable[[Mapping[str, Any]], list[dict[str, Any]]] = canonical_gpu_baseline_identity,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    proof = load_exact_json(path)
    digest = proof.pop("authority_sha256", None)
    if authority_digest(proof) != digest:
        raise GateError("EARLY_PROOF_DIGEST", "proof changed")
    proof["authority_sha256"] = digest
    exact = {
        "schema": EARLY_SCHEMA,
        "status": "EARLY_TWO_FRAME_READY",
        "case_id": CASE_ID,
        "grid_id": GRID_ID,
        "grid_sha256": GRID_SHA256,
        "candidate_sha": BASE_SHA,
        "final_verdict": "FINAL_PENDING_CPU_55_OF_55",
        "gpu_command_executed": False,
    }
    for key, expected in exact.items():
        if proof.get(key) != expected:
            raise GateError("EARLY_PROOF_FIELD", f"{key}={proof.get(key)!r}")
    created = parse_utc(str(proof.get("created_utc")))
    require_fresh_time("derived_proof.created_utc", created, now, DERIVED_PROOF_MAX_AGE_SECONDS)
    if proof.get("upstream_marker_artifact", {}).get("path") != str(EARLY_MARKER.resolve(strict=True)):
        raise GateError("EARLY_PROOF_CANONICAL_MARKER", repr(proof.get("upstream_marker_artifact")))
    rebuilt = build_early_ready_proof(
        marker_path=EARLY_MARKER,
        work_dir=expected_work_dir,
        interval_seconds=interval_seconds,
        sleep_fn=sleep_fn,
        live_identity_checker=live_identity_checker,
        gpu_baseline_checker=gpu_baseline_checker,
        existing_table_authority=proof["table_authority"],
        now=now,
    )
    if rebuilt["canonical_authority_sha256"] != proof.get("canonical_authority_sha256"):
        raise GateError("EARLY_PROOF_NOT_CANONICAL", "canonical reconstruction differs")
    return rebuilt


def _terminal_stamps() -> dict[str, list[str]]:
    return {
        "d01": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in TERMINAL_D01_TIMES],
        "d02": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in EXPECTED_PARENT_TIMES],
        "d03": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in EXPECTED_TIMES],
    }


def _validate_terminal_source_artifact(
    record: Mapping[str, Any], path: Path, commit: str, digest: str, label: str,
) -> dict[str, Any]:
    if record != {"path": str(path), "sha256": digest, "git_commit": commit}:
        raise GateError("TERMINAL_SOURCE_ROLE", label)
    relative = str(path.relative_to(EARLY_AUTHORITY_REPO))
    resolved, blob = _git_blob(EARLY_AUTHORITY_REPO, commit, relative)
    if resolved != commit or sha256_bytes(blob) != digest or sha256_file(path) != digest:
        raise GateError("TERMINAL_SOURCE_AUTHORITY", label)
    return {"path": str(path), "commit": commit, "sha256": digest}


def validate_terminal_contract_payload(contract: Mapping[str, Any]) -> None:
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema", "status", "generated_utc", "case_id", "grid_id", "grid_sha256",
        "canonical_root", "purpose", "source_authority", "terminal_cpu_process",
        "corrected_cpu_qa", "immutable_input_lineage", "parent_schedule_contract",
        "terminal_consumer_requirements", "prohibited", "early_marker_plan_supersession",
        "requested_reviewer_verdict", "gpu_authority",
    }:
        raise GateError("TERMINAL_CONTRACT_SCHEMA", repr(sorted(contract) if isinstance(contract, Mapping) else contract))
    _exact_fields(contract, {
        "schema": "tenerife_terminal_cpu_authority_contract_v1",
        "status": "immutable_terminal_cpu_truth_pending_WRFGPU_review",
        "case_id": CASE_ID, "grid_id": GRID_ID, "grid_sha256": GRID_SHA256,
        "canonical_root": str(CPU_RUN_ROOT),
        "purpose": "Bind immutable terminal CPU truth for independent WRFGPU review of at most the same single corrected-fullbuffer GPU identity arm; this contract grants no GPU authority.",
        "requested_reviewer_verdict": "TERMINAL_CPU_AUTHORITY_ACCEPT or exact changes required",
        "gpu_authority": "NOT_AUTHORIZED_UNTIL_WRFGPU_REVIEW_AND_RUNTIME_ACCEPTANCE",
    }, "TERMINAL_CONTRACT_FIELD")
    parse_utc(str(contract["generated_utc"]))
    if contract.get("source_authority") != {
        "report": {"path": str(TERMINAL_REPORT), "sha256": TERMINAL_REPORT_SHA256, "git_commit": TERMINAL_REPORT_COMMIT},
        "schema": {"path": str(TERMINAL_SCHEMA_SOURCE), "sha256": TERMINAL_SCHEMA_SOURCE_SHA256, "git_commit": TERMINAL_SCHEMA_COMMIT},
        "validator": {"path": str(TERMINAL_VALIDATOR_SOURCE), "sha256": TERMINAL_VALIDATOR_SHA256, "git_commit": TERMINAL_SCHEMA_COMMIT},
        "verifier": {"path": str(EARLY_AUTHORITY_REPO / CPU_QA_SOURCE), "sha256": TERMINAL_VERIFIER_SHA256, "git_commit": TERMINAL_VERIFIER_COMMIT},
        "adversarial_tests": {"path": str(TERMINAL_TEST_SOURCE), "sha256": TERMINAL_TEST_SOURCE_SHA256, "git_commit": TERMINAL_SCHEMA_COMMIT},
    }:
        raise GateError("TERMINAL_SOURCE_ROLE", repr(contract.get("source_authority")))
    process = contract.get("terminal_cpu_process")
    if not isinstance(process, Mapping) or set(process) != {
        "artifact", "wrf_status", "wrf_exit_code", "wrapper_status", "wrapper_exit_code",
        "terminal_all_processes_absent", "live_cpu_requirement", "early_marker_claim",
    }:
        raise GateError("TERMINAL_PROCESS_SCHEMA", repr(process))
    _exact_fields(process, {
        "artifact": {"path": str(TERMINAL_PROCESS_STATUS), "sha256": TERMINAL_PROCESS_STATUS_SHA256},
        "wrf_status": "success", "wrf_exit_code": 0,
        "wrapper_status": "failed_final_qa", "wrapper_exit_code": 1,
        "terminal_all_processes_absent": True, "live_cpu_requirement": False,
        "early_marker_claim": "none_terminal_contract_rejects_marker",
    }, "TERMINAL_PROCESS_FIELD")
    qa = contract.get("corrected_cpu_qa")
    if not isinstance(qa, Mapping) or set(qa) != {
        "status", "log", "manifest", "pair_index", "raw_frame_counts",
        "thin_frame_count", "pair_rows", "pair_counts", "artifact_hash_recompute",
    }:
        raise GateError("TERMINAL_QA_SCHEMA", repr(qa))
    _exact_fields(qa, {
        "status": "pass_posthoc_scheduler_corrected",
        "log": {"path": str(TERMINAL_QA_LOG), "sha256": TERMINAL_QA_LOG_SHA256},
        "manifest": {"path": str(CPU_MANIFEST), "sha256": TERMINAL_MANIFEST_SHA256},
        "pair_index": {"path": str(CPU_PAIR_INDEX), "sha256": TERMINAL_PAIR_INDEX_SHA256},
        "raw_frame_counts": {"d01": 19, "d02": 19, "d03": 55},
        "thin_frame_count": 55, "pair_rows": 129,
        "pair_counts": {"aifs_to_1km_physical_target": 55, "nine_to_three": 19, "three_to_one": 55},
        "artifact_hash_recompute": {"status": "pass", "raw": 93, "thin": 55, "total": 148},
    }, "TERMINAL_QA_FIELD")
    lineage = contract.get("immutable_input_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != {
        "canonical_binding", "input_seal", "sealed_inputs", "forcing_preflight",
        "aifs_forcing_sha256", "frozen_geo_sha256",
    }:
        raise GateError("TERMINAL_LINEAGE_SCHEMA", repr(lineage))
    _exact_fields(lineage, {
        "canonical_binding": {"path": str(CANONICAL_BINDING), "sha256": CANONICAL_BINDING_SHA256},
        "input_seal": {"path": str(INPUT_SEAL), "sha256": PRE_WRF_SEAL_SHA256},
        "forcing_preflight": {"path": str(FORCING_PREFLIGHT), "sha256": FORCING_PREFLIGHT_SHA256},
        "aifs_forcing_sha256": AIFS_SHA256, "frozen_geo_sha256": GRID_SHA256,
    }, "TERMINAL_LINEAGE_FIELD")
    expected_inputs = [
        ("namelist.input", "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838", 2043),
        ("wrfbdy_d01", "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec", 8198766),
        ("wrfinput_d01", "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756", 9674911),
        ("wrfinput_d02", "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964", 26672072),
        ("wrfinput_d03", "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a", 10563652),
    ]
    if lineage.get("sealed_inputs") != [
        {"name": name, "canonical_path": str(CPU_INPUT_DIR / name), "sha256": digest, "bytes": size, "mode": "-r--r--r--"}
        for name, digest, size in expected_inputs
    ]:
        raise GateError("TERMINAL_SEALED_INPUT_ROLE", repr(lineage.get("sealed_inputs")))
    schedule = contract.get("parent_schedule_contract")
    if not isinstance(schedule, Mapping) or set(schedule) != {
        "source_namelist_sha256", "base_time_step_seconds", "parent_time_step_ratio",
        "history_interval_minutes", "d01", "d02", "d03",
        "d01_to_d02_offsets_seconds", "pair_policy",
    }:
        raise GateError("TERMINAL_SCHEDULE_SCHEMA", repr(schedule))
    stamps = _terminal_stamps()
    _exact_fields(schedule, {
        "source_namelist_sha256": lineage["sealed_inputs"][0]["sha256"] if isinstance(lineage.get("sealed_inputs"), list) and lineage["sealed_inputs"] else None,
        "base_time_step_seconds": 54, "parent_time_step_ratio": [1, 3, 3],
        "history_interval_minutes": [60, 60, 20],
        "d01": stamps["d01"], "d02": stamps["d02"], "d03": stamps["d03"],
        "d01_to_d02_offsets_seconds": [(18 * i) % 54 for i in range(19)],
        "pair_policy": "Use exact produced NetCDF Times and filenames: 7 exact d01/d02 pairs, 6 +18-second pairs, and 6 +36-second pairs; never rename or normalize timestamps.",
    }, "TERMINAL_SCHEDULE_FIELD")
    if contract.get("terminal_consumer_requirements") != [
        "Validate contract, process, manifest, pair index, seal, binding, forcing, schema, report, validator, and verifier hashes before use.",
        "Require WRF success, terminal process absence, corrected physical/finite QA, and exact d01/d02/d03 scheduler lists.",
        "Remove live CPU PID and early-marker freshness requirements only for this exact terminal authority mode.",
        "Rehash every selected CPU raw frame before GPU comparison and retain the authoritative GPU lock and production preemption gates.",
        "Retain final finite/physical, pair, cache, plot, and independent review gates; at most one GPU arm may be considered.",
        "Reject any canonical EARLY_GPU_READY, revoked, or invalidated marker for this terminal run.",
    ] or contract.get("prohibited") != [
        "no gpu launch by wrf_downscale", "no new cpu wrf run",
        "no live PID or early-marker authority claim", "no old input seal or canonical binding rewrite",
        "no output filename rename or timestamp normalization",
        "no training eligibility promotion from this contract",
    ]:
        raise GateError("TERMINAL_POLICY_LIST", "requirements/prohibited")
    if contract.get("early_marker_plan_supersession") != {
        "path": str(TERMINAL_SUPERSESSION), "sha256": TERMINAL_SUPERSESSION_SHA256,
    }:
        raise GateError("TERMINAL_SUPERSESSION_ROLE", repr(contract.get("early_marker_plan_supersession")))


def validate_terminal_process_evidence(contract: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_file(TERMINAL_PROCESS_STATUS) != TERMINAL_PROCESS_STATUS_SHA256:
        raise GateError("TERMINAL_PROCESS_HASH", str(TERMINAL_PROCESS_STATUS))
    payload = load_exact_json(TERMINAL_PROCESS_STATUS)
    _exact_fields(payload, {
        "schema": "tenerife_terminal_cpu_process_status_v1",
        "status": "wrf_complete_wrapper_failed_obsolete_timestamp_harness",
        "case_id": CASE_ID, "grid_id": GRID_ID, "gpu_authority": "NOT_AUTHORIZED",
    }, "TERMINAL_PROCESS_EVIDENCE")
    _exact_fields(payload.get("terminal_process_probe") or {}, {
        "wrapper_proc_absent": True, "wrf_root_proc_absent": True,
        "old_watcher_proc_absent": True, "all_12_rank_procs_absent": True,
    }, "TERMINAL_PROCESS_ABSENCE")
    execution = payload.get("wrf_execution") or {}
    wrapper = payload.get("wrapper_terminal") or {}
    if execution.get("status") != "success" or execution.get("exit_code") != 0:
        raise GateError("TERMINAL_WRF_RC0", repr(execution))
    if wrapper.get("payload", {}).get("status") != "failed_final_qa" or wrapper.get("exit_code") != 1:
        raise GateError("TERMINAL_WRAPPER_TRUTH", repr(wrapper))
    for artifact in (
        execution.get("rsl_error"), execution.get("rsl_out"),
        {"path": wrapper.get("path"), "sha256": wrapper.get("sha256")},
        payload.get("obsolete_qa_failure"),
        {"path": payload.get("corrected_posthoc_qa", {}).get("path"), "sha256": payload.get("corrected_posthoc_qa", {}).get("sha256")},
        {"path": payload.get("early_marker", {}).get("evidence_path"), "sha256": payload.get("early_marker", {}).get("evidence_sha256")},
    ):
        if not isinstance(artifact, Mapping) or sha256_file(Path(str(artifact.get("path")))) != artifact.get("sha256"):
            raise GateError("TERMINAL_PROCESS_ARTIFACT", repr(artifact))
    launch = payload.get("launch_identity") or {}
    for key in ("wrapper_pid", "wrf_root_pid", "old_watcher_pid"):
        pid = launch.get(key)
        if type(pid) is not int or pid <= 1 or Path(f"/proc/{pid}").exists():
            raise GateError("TERMINAL_PROCESS_STILL_LIVE", f"{key}={pid}")
    if payload.get("early_marker", {}).get("status") != "not_emitted_and_must_not_be_emitted_for_terminal_run":
        raise GateError("TERMINAL_MARKER_CLAIM", repr(payload.get("early_marker")))
    return {"payload": payload, "artifact": file_authority(TERMINAL_PROCESS_STATUS)}


def validate_terminal_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    lineage = contract["immutable_input_lineage"]
    binding = load_exact_json(CANONICAL_BINDING)
    seal = load_exact_json(INPUT_SEAL)
    if sha256_file(CANONICAL_BINDING) != CANONICAL_BINDING_SHA256 or sha256_file(INPUT_SEAL) != PRE_WRF_SEAL_SHA256:
        raise GateError("TERMINAL_INPUT_LINEAGE_HASH", "binding/seal")
    provenance = seal.get("provenance") or {}
    if (
        binding.get("input_seal_checksums_sha256") != LAUNCH_TIME_CHECKSUMS_SHA256
        or binding.get("input_seal_checksums_sha256") != provenance.get("checksums_sha256")
        or binding.get("input_seal_checksums_sha256") == CURRENT_CHECKSUMS_SHA256
        or binding.get("input_seal_launch_plan_sha256") != provenance.get("launch_plan_sha256")
        or binding.get("input_seal_launch_plan_sha256") == LAUNCH_PLAN_SHA256
    ):
        raise GateError("TERMINAL_HISTORICAL_ROLE", "binding/seal provenance")
    declared = lineage.get("sealed_inputs")
    if not isinstance(declared, list) or [row.get("name") for row in declared if isinstance(row, Mapping)] != list(INPUT_NAMES):
        raise GateError("TERMINAL_INPUT_SET", repr(declared))
    sealed_by_name = {row.get("name"): row for row in seal.get("files", []) if isinstance(row, Mapping)}
    inputs: dict[str, Any] = {}
    for row in declared:
        if set(row) != {"name", "canonical_path", "sha256", "bytes", "mode"}:
            raise GateError("TERMINAL_INPUT_SCHEMA", repr(row))
        name = row["name"]
        path = CPU_INPUT_DIR / name
        if row["canonical_path"] != str(path) or path.is_symlink():
            raise GateError("TERMINAL_INPUT_PATH", repr(row))
        info = _regular_file_stat(path)
        digest = sha256_file(path)
        if (
            stat.S_IMODE(info.st_mode) != 0o444 or row["mode"] != "-r--r--r--"
            or info.st_size != row["bytes"] or digest != row["sha256"]
            or sealed_by_name.get(name, {}).get("sha256") != digest
        ):
            raise GateError("TERMINAL_INPUT_AUTHORITY", name)
        inputs[name] = file_authority(path)
    forcing = load_exact_json(FORCING_PREFLIGHT)
    if sha256_file(FORCING_PREFLIGHT) != FORCING_PREFLIGHT_SHA256 or forcing.get("source", {}).get("sha256") != AIFS_SHA256:
        raise GateError("TERMINAL_FORCING_AUTHORITY", str(FORCING_PREFLIGHT))
    fd_evidence = scan_current_input_open_fds(inputs)
    for name in inputs:
        inputs[name]["current_open_fd_evidence"] = fd_evidence["targets"][name]["holders"]
    return inputs


def _require_terminal_geometry(dataset: Dataset, geo: Mapping[str, np.ndarray], label: str) -> None:
    for field in ("XLAT", "XLONG", "LANDMASK"):
        actual = _static_plane(dataset, field)
        source = geo[field]
        if actual.shape != source.shape or not np.allclose(actual, source, rtol=0.0, atol=5e-5):
            raise GateError("TERMINAL_CPU_GEOMETRY", f"{label}:{field}")
    hgt = np.asarray(_static_plane(dataset, "HGT"), dtype=np.float64)
    source_hgt = np.asarray(geo["HGT"], dtype=np.float64)
    if hgt.shape != source_hgt.shape or not np.isfinite(hgt).all() or hgt.min() < -0.01 or hgt.max() > 5000.0:
        raise GateError("TERMINAL_CPU_GEOMETRY", f"{label}:HGT")
    if min(hgt.shape) > 20 and not np.allclose(
        hgt[10:-10, 10:-10], source_hgt[10:-10, 10:-10], rtol=0.0, atol=5e-5,
    ):
        raise GateError("TERMINAL_CPU_GEOMETRY", f"{label}:HGT_INTERIOR")


def _expected_terminal_pair_rows(
    raw: Mapping[str, Mapping[str, Path]],
    raw_hashes: Mapping[str, Mapping[str, str]],
    thin: Mapping[str, Path],
    thin_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    stamps = _terminal_stamps()
    rows: list[dict[str, Any]] = []
    for index, target_stamp in enumerate(stamps["d02"]):
        input_stamp = stamps["d01"][index]
        offset = int((
            datetime.strptime(input_stamp, "%Y-%m-%d_%H:%M:%S")
            - datetime.strptime(target_stamp, "%Y-%m-%d_%H:%M:%S")
        ).total_seconds())
        rows.append({
            "ladder": "nine_to_three",
            "relation": "exact" if offset == 0 else "wrf_scheduler_aligned_offset",
            "valid_time": target_stamp, "input_valid_time": input_stamp,
            "target_valid_time": target_stamp, "scheduler_offset_seconds": offset,
            "input_raw": str(raw["d01"][input_stamp]),
            "input_sha256": raw_hashes["d01"][input_stamp],
            "target_raw": str(raw["d02"][target_stamp]),
            "target_sha256": raw_hashes["d02"][target_stamp],
        })
    parent = [datetime.strptime(value, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc) for value in stamps["d02"]]
    for index, stamp in enumerate(stamps["d03"]):
        valid = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        common = {
            "valid_time": stamp, "target_raw": str(raw["d03"][stamp]),
            "target_raw_sha256": raw_hashes["d03"][stamp],
            "target_thin": str(thin[stamp]), "target_thin_sha256": thin_hashes[stamp],
        }
        rows.append({
            "ladder": "aifs_to_1km", "relation": "physical_target_only_se_cache_pending",
            "forcing_frame_index": index, **common,
        })
        if stamp in raw["d02"]:
            rows.append({"ladder": "three_to_one", "relation": "exact", "input_raw": str(raw["d02"][stamp]), **common})
        else:
            before = max(value for value in parent if value < valid).strftime("%Y-%m-%d_%H:%M:%S")
            after = min(value for value in parent if value > valid).strftime("%Y-%m-%d_%H:%M:%S")
            rows.append({
                "ladder": "three_to_one", "relation": "bracketed",
                "input_raw_before": str(raw["d02"][before]),
                "input_raw_after": str(raw["d02"][after]),
                "weight_after": (valid.minute % 60) / 60.0, **common,
            })
    return rows


def validate_terminal_cpu_manifest(
    contract: Mapping[str, Any],
    *,
    interval_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    input_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest, manifest_artifact = stable_json_authority(
        CPU_MANIFEST, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    if manifest_artifact["sha256"] != TERMINAL_MANIFEST_SHA256 or set(manifest) != {
        "schema", "status", "qa_status", "case_id", "grid_id", "grid_sha256",
        "engine", "raw_frame_counts", "raw_frame_schedules", "scheduler_contract",
        "child_geometry_qc", "raw_artifacts", "thin_frame_count", "thin_artifacts",
        "pair_counts", "pair_index", "pair_index_rows", "surface_ranges",
        "all_required_raw_and_thin_fields_finite", "grid_contract", "forcing_path",
        "forcing_sha256", "gpu_identity_contract", "wrapper_terminal_status",
        "eligibility", "generated_utc",
    }:
        raise GateError("TERMINAL_MANIFEST_SCHEMA", repr(sorted(manifest)))
    stamps = _terminal_stamps()
    _exact_fields(manifest, {
        "schema": "tenerife_fullbuffer_cpu_oracle_v1", "status": "complete", "qa_status": "pass",
        "case_id": CASE_ID, "grid_id": GRID_ID, "grid_sha256": GRID_SHA256,
        "engine": "cpu_wrf_max_dom3", "raw_frame_counts": {"d01": 19, "d02": 19, "d03": 55},
        "raw_frame_schedules": stamps, "thin_frame_count": 55,
        "pair_counts": {"nine_to_three": 19, "aifs_to_1km_physical_target": 55, "three_to_one": 55},
        "pair_index": str(CPU_PAIR_INDEX), "pair_index_rows": 129,
        "all_required_raw_and_thin_fields_finite": True,
        "grid_contract": {
            "lat_lon_landmask_match_frozen_geo_all_frames": True,
            "hgt_matches_frozen_geo_outside_nested_boundary_margin": True,
            "nested_hgt_boundary_margin_cells": 10,
        },
        "forcing_path": "<DATA_ROOT>/alisios/forcing/historical/aifs/20250228_18z/aifs_pure_wps_20250228_18z.grib2",
        "forcing_sha256": AIFS_SHA256,
        "eligibility": {
            "aifs_to_1km": "pending_se_cache_and_model_qa",
            "three_to_one": "pending_training_cache_and_cv",
            "nine_to_three": "pending_training_policy_for_d01_scheduler_offsets_max_36_seconds",
        },
    }, "TERMINAL_MANIFEST_FIELD")
    parse_utc(str(manifest["generated_utc"]))
    gpu_contract = manifest.get("gpu_identity_contract")
    if not isinstance(gpu_contract, Mapping) or set(gpu_contract) != {
        "consumer", "matched_raw_domain", "valid_times", "exclude_fields",
    } or gpu_contract.get("consumer") != "wrf_gpu 0:1" or gpu_contract.get("matched_raw_domain") != "d03" or gpu_contract.get("valid_times") != stamps["d03"] or gpu_contract.get("exclude_fields") != ["QVAPOR", "RAINNC"]:
        raise GateError("TERMINAL_GPU_IDENTITY_CONTRACT", repr(gpu_contract))
    scheduler = manifest.get("scheduler_contract") or {}
    if (
        scheduler.get("status") != "pass" or scheduler.get("source") != str(CPU_INPUT_DIR / "namelist.input")
        or scheduler.get("source_sha256") != contract["parent_schedule_contract"]["source_namelist_sha256"]
        or scheduler.get("base_time_step_seconds") != 54
        or scheduler.get("parent_time_step_ratio") != [1, 3, 3]
        or scheduler.get("history_interval_minutes") != [60, 60, 20]
        or scheduler.get("frames_per_outfile") != [1, 1, 1]
        or scheduler.get("spec_bdy_width") != 5
        or scheduler.get("nested_hgt_boundary_margin_cells") != 10
    ):
        raise GateError("TERMINAL_MANIFEST_SCHEDULER", repr(scheduler))
    for domain in ("d01", "d02", "d03"):
        rows = scheduler.get("domains", {}).get(domain, {}).get("rows")
        if not isinstance(rows, list) or [row.get("actual_output_time") for row in rows] != stamps[domain]:
            raise GateError("TERMINAL_MANIFEST_SCHEDULE", domain)
    raw: dict[str, dict[str, Path]] = {}
    raw_hashes: dict[str, dict[str, str]] = {}
    raw_qa: dict[str, dict[str, Any]] = {}
    geo, geo_authority = load_frozen_geometry()
    aggregate: dict[str, list[float]] = {}
    declared_raw = manifest.get("raw_artifacts")
    if not isinstance(declared_raw, Mapping) or set(declared_raw) != {"d01", "d02", "d03"}:
        raise GateError("TERMINAL_RAW_ARTIFACTS", repr(declared_raw))
    for domain in ("d01", "d02", "d03"):
        frames = strict_domain_frame_inventory(CPU_INPUT_DIR, domain)
        raw[domain] = {valid.strftime("%Y-%m-%d_%H:%M:%S"): path for valid, path in frames.items()}
        if list(sorted(raw[domain])) != stamps[domain]:
            raise GateError("TERMINAL_RAW_INVENTORY", domain)
        rows = declared_raw[domain]
        if not isinstance(rows, list) or [row.get("valid_time") for row in rows] != stamps[domain]:
            raise GateError("TERMINAL_RAW_DECLARATION", domain)
        raw_hashes[domain], raw_qa[domain] = {}, {}
        for row in rows:
            stamp = row["valid_time"]
            path = raw[domain][stamp]
            if set(row) != {"valid_time", "path", "bytes", "sha256"} or row["path"] != str(path):
                raise GateError("TERMINAL_RAW_ROLE", repr(row))
            qa = _cpu_frame_qa(
                path, stamp, CPU_CHILD_REQUIRED if domain == "d03" else CPU_PARENT_REQUIRED,
                None,
            )
            if qa["authority"]["sha256"] != row["sha256"] or qa["authority"]["size"] != row["bytes"]:
                raise GateError("TERMINAL_RAW_HASH", str(path))
            if domain == "d03":
                with Dataset(path, "r") as dataset:
                    _require_terminal_geometry(dataset, geo, f"raw:{stamp}")
                for field, bounds in qa["surface_ranges"].items():
                    current = aggregate.setdefault(field, [float("inf"), float("-inf")])
                    current[0], current[1] = min(current[0], bounds[0]), max(current[1], bounds[1])
            raw_hashes[domain][stamp] = row["sha256"]
            raw_qa[domain][stamp] = qa
    if manifest.get("surface_ranges") != aggregate:
        raise GateError("TERMINAL_MANIFEST_RANGES", repr(manifest.get("surface_ranges")))
    thin: dict[str, Path] = {}
    thin_hashes: dict[str, str] = {}
    thin_qa: dict[str, Any] = {}
    thin_rows = manifest.get("thin_artifacts")
    if not isinstance(thin_rows, list) or [row.get("valid_time") for row in thin_rows] != stamps["d03"]:
        raise GateError("TERMINAL_THIN_DECLARATION", repr(thin_rows))
    if set(CPU_THIN_DIR.glob("wrfout_d03_*.thin.nc")) != {
        CPU_THIN_DIR / f"wrfout_d03_{stamp}.thin.nc" for stamp in stamps["d03"]
    }:
        raise GateError("TERMINAL_THIN_INVENTORY", str(CPU_THIN_DIR))
    for row in thin_rows:
        stamp = row["valid_time"]
        path = CPU_THIN_DIR / f"wrfout_d03_{stamp}.thin.nc"
        if set(row) != {"valid_time", "path", "bytes", "sha256"} or row["path"] != str(path):
            raise GateError("TERMINAL_THIN_ROLE", repr(row))
        authority = file_authority(path)
        if authority["sha256"] != row["sha256"] or authority["size"] != row["bytes"]:
            raise GateError("TERMINAL_THIN_HASH", str(path))
        with Dataset(path, "r") as dataset:
            if sorted(CPU_THIN_REQUIRED - set(dataset.variables)) or _decoded_times(dataset) != stamp:
                raise GateError("TERMINAL_THIN_QA", str(path))
            if getattr(dataset, "grid_id", None) != GRID_ID or getattr(dataset, "geo_em_sha256", None) != GRID_SHA256 or getattr(dataset, "source_raw", None) != str(raw["d03"][stamp]):
                raise GateError("TERMINAL_THIN_AUTHORITY", str(path))
            _require_terminal_geometry(dataset, geo, f"thin:{stamp}")
            for field in CPU_THIN_REQUIRED - {"Times"}:
                values = np.asarray(np.ma.filled(dataset.variables[field][:], np.nan))
                if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                    raise GateError("TERMINAL_THIN_NONFINITE", f"{path}:{field}")
        thin[stamp], thin_hashes[stamp], thin_qa[stamp] = path, row["sha256"], authority
    pair_rows = _strict_json_lines(CPU_PAIR_INDEX)
    if sha256_file(CPU_PAIR_INDEX) != TERMINAL_PAIR_INDEX_SHA256 or pair_rows != _expected_terminal_pair_rows(raw, raw_hashes, thin, thin_hashes):
        raise GateError("TERMINAL_PAIR_INDEX", str(CPU_PAIR_INDEX))
    producer_commit, producer_source = _git_blob(EARLY_AUTHORITY_REPO, TERMINAL_VERIFIER_COMMIT, CPU_QA_SOURCE)
    if producer_commit != TERMINAL_VERIFIER_COMMIT or sha256_bytes(producer_source) != TERMINAL_VERIFIER_SHA256:
        raise GateError("TERMINAL_QA_PRODUCER", producer_commit)
    recomputed = {
        "producer": {"commit": producer_commit, "path": CPU_QA_SOURCE, "sha256": TERMINAL_VERIFIER_SHA256},
        "manifest_artifact": manifest_artifact, "raw_counts": {key: len(value) for key, value in raw.items()},
        "raw_qa": raw_qa, "thin_count": len(thin_qa), "thin_qa": thin_qa,
        "pair_index": file_authority(CPU_PAIR_INDEX), "pair_index_rows": len(pair_rows),
        "pair_counts": manifest["pair_counts"], "geo_authority": geo_authority,
        "identity_policy": validate_identity_policy_authority(),
        "terminal_input_authority": dict(input_authority or validate_terminal_inputs(contract)),
        "producer_declared_legacy_exclude_fields": gpu_contract["exclude_fields"],
        "effective_owner_report_only_fields": list(REPORT_ONLY_FIELDS),
    }
    recomputed["authority_sha256"] = authority_digest(recomputed)
    return {"manifest": manifest, "recomputed": recomputed}


def validate_terminal_contract_authority(
    path: Path = TERMINAL_CONTRACT,
    *,
    interval_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if path != TERMINAL_CONTRACT or path.resolve(strict=True) != TERMINAL_CONTRACT:
        raise GateError("TERMINAL_CONTRACT_PATH", str(path))
    info = _regular_file_stat(path)
    if stat.S_IMODE(info.st_mode) != 0o444:
        raise GateError("TERMINAL_CONTRACT_MODE", oct(stat.S_IMODE(info.st_mode)))
    contract, artifact = stable_json_authority(
        path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    if artifact["sha256"] != TERMINAL_CONTRACT_SHA256:
        raise GateError("TERMINAL_CONTRACT_HASH", artifact["sha256"])
    validate_terminal_contract_payload(contract)
    source = contract["source_authority"]
    source_authority = {
        "report": _validate_terminal_source_artifact(
            source["report"], TERMINAL_REPORT, TERMINAL_REPORT_COMMIT,
            TERMINAL_REPORT_SHA256, "report",
        ),
        "schema": _validate_terminal_source_artifact(
            source["schema"], TERMINAL_SCHEMA_SOURCE, TERMINAL_SCHEMA_COMMIT,
            TERMINAL_SCHEMA_SOURCE_SHA256, "schema",
        ),
        "validator": _validate_terminal_source_artifact(
            source["validator"], TERMINAL_VALIDATOR_SOURCE, TERMINAL_SCHEMA_COMMIT,
            TERMINAL_VALIDATOR_SHA256, "validator",
        ),
        "verifier": _validate_terminal_source_artifact(
            source["verifier"], EARLY_AUTHORITY_REPO / CPU_QA_SOURCE,
            TERMINAL_VERIFIER_COMMIT, TERMINAL_VERIFIER_SHA256, "verifier",
        ),
        "adversarial_tests": _validate_terminal_source_artifact(
            source["adversarial_tests"], TERMINAL_TEST_SOURCE, TERMINAL_SCHEMA_COMMIT,
            TERMINAL_TEST_SOURCE_SHA256, "adversarial_tests",
        ),
    }
    for expected_path, expected_hash, code in (
        (TERMINAL_QA_LOG, TERMINAL_QA_LOG_SHA256, "TERMINAL_QA_LOG"),
        (CPU_MANIFEST, TERMINAL_MANIFEST_SHA256, "TERMINAL_MANIFEST_HASH"),
        (CPU_PAIR_INDEX, TERMINAL_PAIR_INDEX_SHA256, "TERMINAL_PAIR_HASH"),
        (TERMINAL_SUPERSESSION, TERMINAL_SUPERSESSION_SHA256, "TERMINAL_SUPERSESSION_HASH"),
    ):
        if sha256_file(expected_path) != expected_hash:
            raise GateError(code, str(expected_path))
    if CPU_RUN_ROOT.is_symlink() or not CPU_RUN_ROOT.is_dir() or CPU_RUN_ROOT.resolve() != CPU_RUN_ROOT:
        raise GateError("TERMINAL_CANONICAL_ROOT", str(CPU_RUN_ROOT))
    for marker in (EARLY_MARKER, EARLY_REVOKED, EARLY_INVALIDATED):
        if marker.exists() or marker.is_symlink():
            raise GateError("TERMINAL_REJECTS_EARLY_MARKER", str(marker))
    process = validate_terminal_process_evidence(contract)
    inputs = validate_terminal_inputs(contract)
    terminal_cpu = validate_terminal_cpu_manifest(
        contract, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
        input_authority=inputs,
    )
    result = {
        "contract": contract, "artifact": artifact, "source_authority": source_authority,
        "process": process, "input_authority": inputs, "terminal_cpu": terminal_cpu,
        "gpu_authority": "NOT_AUTHORIZED",
        "live_cpu_requirement": False, "early_marker_claim": False,
    }
    result["authority_sha256"] = authority_digest(result)
    return result


def build_terminal_cpu_proof(
    *,
    contract_path: Path,
    work_dir: Path,
    interval_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    existing_table_authority: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    require_current_no_preemption()
    authority = validate_terminal_contract_authority(
        contract_path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    if existing_table_authority is None:
        table_authority = materialize_table_snapshot(WRF_SOURCE_ROOT, work_dir)
    else:
        table_authority = dict(existing_table_authority)
        verify_table_snapshot(table_authority, expected_work_dir=work_dir)
    identity_policy = validate_identity_policy_authority()
    proof = {
        "schema": TERMINAL_PROOF_SCHEMA, "status": "TERMINAL_CPU_AUTHORITY_READY",
        "case_id": CASE_ID, "grid_id": GRID_ID, "grid_sha256": GRID_SHA256,
        "candidate_sha": BASE_SHA, "created_utc": now.isoformat(),
        "stability_interval_seconds": interval_seconds,
        "terminal_authority": authority,
        "cpu_input_dir": str(CPU_INPUT_DIR.resolve(strict=True)),
        "input_authority": authority["input_authority"],
        "terminal_cpu": authority["terminal_cpu"],
        "table_authority": table_authority, "identity_policy": identity_policy,
        "gpu_authority": "NOT_AUTHORIZED_PENDING_COMBINED_REVIEW_AND_RUNTIME_ACCEPTANCE",
        "live_cpu_requirement": False, "early_marker_requirement": False,
        "final_verdict": "FINAL_PENDING_GPU_55_OF_55", "gpu_command_executed": False,
    }
    canonical = {
        "candidate_sha": BASE_SHA, "terminal_contract": authority["artifact"],
        "terminal_process": authority["process"]["artifact"],
        "manifest": authority["terminal_cpu"]["recomputed"]["manifest_artifact"],
        "pair_index": authority["terminal_cpu"]["recomputed"]["pair_index"],
        "input_authority": authority["input_authority"],
        "table_authority_sha256": table_authority["authority_sha256"],
        "identity_policy_sha256": identity_policy["authority_sha256"],
        "live_cpu_requirement": False, "early_marker_requirement": False,
    }
    proof["canonical_authority_sha256"] = authority_digest(canonical)
    proof["authority_sha256"] = authority_digest(proof)
    return proof


def verify_terminal_cpu_proof(
    path: Path,
    *,
    expected_work_dir: Path,
    interval_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> dict[str, Any]:
    proof = load_exact_json(path)
    digest = proof.pop("authority_sha256", None)
    if authority_digest(proof) != digest:
        raise GateError("TERMINAL_PROOF_DIGEST", "proof changed")
    proof["authority_sha256"] = digest
    _exact_fields(proof, {
        "schema": TERMINAL_PROOF_SCHEMA, "status": "TERMINAL_CPU_AUTHORITY_READY",
        "case_id": CASE_ID, "grid_id": GRID_ID, "grid_sha256": GRID_SHA256,
        "candidate_sha": BASE_SHA, "live_cpu_requirement": False,
        "early_marker_requirement": False, "final_verdict": "FINAL_PENDING_GPU_55_OF_55",
        "gpu_command_executed": False,
    }, "TERMINAL_PROOF_FIELD")
    if proof.get("terminal_authority", {}).get("artifact", {}).get("path") != str(TERMINAL_CONTRACT):
        raise GateError("TERMINAL_PROOF_CONTRACT", repr(proof.get("terminal_authority")))
    rebuilt = build_terminal_cpu_proof(
        contract_path=TERMINAL_CONTRACT,
        work_dir=expected_work_dir,
        interval_seconds=interval_seconds, sleep_fn=sleep_fn,
        existing_table_authority=proof["table_authority"], now=now,
    )
    if rebuilt["canonical_authority_sha256"] != proof.get("canonical_authority_sha256"):
        raise GateError("TERMINAL_PROOF_NOT_CANONICAL", "canonical reconstruction differs")
    return rebuilt


def validate_lock_v2() -> dict[str, Any]:
    commit = subprocess.check_output(
        ("git", "-C", str(LOCK_ROOT), "rev-parse", "HEAD"), text=True,
    ).strip()
    wrapper_hash = sha256_file(LOCK_WRAPPER)
    if commit != LOCK_COMMIT or wrapper_hash != LOCK_WRAPPER_SHA256:
        raise GateError("LOCK_V2_AUTHORITY", f"commit={commit} wrapper={wrapper_hash}")
    return {
        "root": str(LOCK_ROOT),
        "commit": commit,
        "wrapper": str(LOCK_WRAPPER),
        "wrapper_sha256": wrapper_hash,
        "intent": LOCK_INTENT,
    }


def validate_candidate_worktree(path: Path) -> dict[str, Any]:
    head = subprocess.check_output(("git", "-C", str(path), "rev-parse", "HEAD"), text=True).strip()
    clean = not subprocess.check_output(("git", "-C", str(path), "status", "--porcelain"), text=True).strip()
    symbolic = subprocess.run(
        ("git", "-C", str(path), "symbolic-ref", "-q", "HEAD"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    common = Path(
        subprocess.check_output(
            ("git", "-C", str(path), "rev-parse", "--git-common-dir"), text=True,
        ).strip()
    )
    if not common.is_absolute():
        common = (path / common).resolve()
    if head != BASE_SHA or not clean or symbolic.returncode == 0 or common == (path / ".git").resolve():
        raise GateError(
            "CANDIDATE_WORKTREE",
            f"head={head} clean={clean} detached={symbolic.returncode != 0} common={common}",
        )
    return {"path": str(path.resolve()), "head": head, "clean": clean, "detached": True}


def validate_live_cpu_lane() -> dict[str, Any]:
    compute = re.compile(r"wrf\.exe|real\.exe|mpirun|mpiexec|gpuwrf|\bjax\b|\bxla\b", re.I)
    blockers: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        pid = int(entry.name)
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if not compute.search(command):
                continue
            status = {
                key: value.strip()
                for key, value in (
                    line.split(":", 1) for line in (entry / "status").read_text().splitlines() if ":" in line
                )
            }
            cpus = _parse_cpu_set(status.get("Cpus_allowed_list", ""))
        except (OSError, ValueError):
            continue
        if cpus & set(VALIDATION_LANE):
            blockers.append({"pid": pid, "cpuset": sorted(cpus), "command": command[:500]})
    if blockers:
        raise GateError("VALIDATION_LANE_BUSY", repr(blockers))
    return {"cpuset": "12-15", "free": True, "compute_blockers": []}


def validate_live_gpu_device(
    expected: Mapping[str, Any], expected_live_identity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    processes = subprocess.run(
        ("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if processes.returncode != 0:
        raise GateError("GPU_DEVICE_QUERY", processes.stdout[-1000:])
    rows = [line.strip() for line in processes.stdout.splitlines() if line.strip() and "No running" not in line]
    current: list[dict[str, Any]] = []
    for row in rows:
        fields = [value.strip() for value in row.rsplit(",", 2)]
        if len(fields) != 3:
            raise GateError("GPU_DEVICE_QUERY", repr(row))
        try:
            current.append({
                "pid": int(fields[0]), "process_name": fields[1],
                "used_memory_mib": int(fields[2]),
            })
        except ValueError as exc:
            raise GateError("GPU_DEVICE_QUERY", repr(row)) from exc
    expected_rows = expected.get("baseline_compute_processes")
    if not isinstance(expected_rows, list) or expected.get("unexpected_compute_processes") != []:
        raise GateError("GPU_BASELINE_AUTHORITY", repr(expected))
    expected_identity: set[tuple[int, str]] = set()
    expected_live = {
        (int(item["pid"]), str(item["process_name"])): dict(item)
        for item in expected_live_identity
    }
    for item in expected_rows:
        if not isinstance(item, dict) or item.get("baseline_allowed") is not True:
            raise GateError("GPU_BASELINE_AUTHORITY", repr(item))
        identity = (int(item.get("pid", -1)), str(item.get("process_name", "")))
        if identity[0] <= 1 or not identity[1] or identity in expected_identity:
            raise GateError("GPU_BASELINE_AUTHORITY", repr(item))
        expected_identity.add(identity)
    if set(expected_live) != expected_identity:
        raise GateError("GPU_BASELINE_START_AUTHORITY", repr(expected_live_identity))
    current_identity = {(item["pid"], item["process_name"]) for item in current}
    if current_identity != expected_identity:
        raise GateError(
            "GPU_BASELINE_IDENTITY_CHANGED",
            f"expected={sorted(expected_identity)} current={sorted(current_identity)}",
        )
    for identity, bound in expected_live.items():
        current_process = live_process_binding(identity[0])
        if current_process["start_ticks"] != bound["start_ticks"]:
            raise GateError("GPU_BASELINE_PID_REUSED", repr(identity))
        if current_process != {
            key: bound[key]
            for key in ("start_ticks", "argv0", "exe_path", "exe_device", "exe_inode")
        }:
            raise GateError("GPU_BASELINE_COMMAND_CHANGED", repr(identity))
    canonical_gpu = expected.get("gpu")
    if not isinstance(canonical_gpu, dict):
        raise GateError("GPU_OBJECT_SCHEMA", repr(canonical_gpu))
    device = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if device.returncode != 0 or not device.stdout.strip():
        raise GateError("GPU_DEVICE_QUERY", device.stdout[-1000:])
    fields = [value.strip() for value in device.stdout.splitlines()[0].split(",")]
    if len(fields) != 6:
        raise GateError("GPU_DEVICE_QUERY", repr(fields))
    try:
        index, name = int(fields[0]), fields[1]
        total_mib, used_mib, free_mib, utilization = map(int, fields[2:])
    except ValueError as exc:
        raise GateError("GPU_DEVICE_QUERY", repr(fields)) from exc
    if (
        index != canonical_gpu.get("index")
        or name != canonical_gpu.get("name")
        or total_mib != canonical_gpu.get("memory_total_mib")
    ):
        raise GateError(
            "GPU_DEVICE_IDENTITY_CHANGED",
            f"canonical={canonical_gpu!r} live=index={index!r},name={name!r},total={total_mib!r}",
        )
    if free_mib < 24000 or utilization > 20:
        raise GateError("GPU_DEVICE_NOT_FREE", f"free_mib={free_mib} utilization={utilization}")
    return {
        "free": True, "index": index, "name": name,
        "memory_total_mib": total_mib, "memory_used_mib": used_mib,
        "memory_free_mib": free_mib, "utilization_percent": utilization,
        "baseline_compute_processes": current, "unexpected_compute_processes": [],
    }


def validate_live_lock_available() -> dict[str, Any]:
    lock_authority = validate_lock_v2()
    lock_path = Path("/tmp/wrf_gpu2_gpu.lock")
    info = _regular_file_stat(lock_path)
    fd = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    acquired = False
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise GateError("GPU_LOCK_REPLACED", str(lock_path))
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise GateError("GPU_LOCK_BUSY", str(lock_path)) from exc
        current = lock_path.stat()
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise GateError("GPU_LOCK_REPLACED", str(lock_path))
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    return {
        "free": True, "path": str(lock_path), "device": info.st_dev, "inode": info.st_ino,
        "lock_v2": lock_authority,
    }


def immediate_pre_spawn_checks(
    proof_path: Path,
    *,
    expected_work_dir: Path,
    now: datetime | None = None,
    interval_seconds: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    rebuilt = verify_early_ready_proof(
        proof_path, expected_work_dir=expected_work_dir,
        now=now, interval_seconds=interval_seconds,
    )
    verify_live_cpu_identity(
        rebuilt["upstream_marker"]["cpu_oracle"], rebuilt["cpu_live_identity"],
    )
    live_lock = validate_live_lock_available()
    canonical_lock = rebuilt["resource_proof"]["payload"]["gpu_lock"]
    if (
        live_lock.get("device") != canonical_lock.get("device")
        or live_lock.get("inode") != canonical_lock.get("inode")
    ):
        raise GateError("GPU_LOCK_IDENTITY_CHANGED", f"canonical={canonical_lock} live={live_lock}")
    resources = {
        "checked_utc": now.isoformat(),
        "cpu_identity": rebuilt["cpu_live_identity"],
        "cpu_lane": validate_live_cpu_lane(),
        "gpu_device": validate_live_gpu_device(
            rebuilt["resource_proof"]["payload"]["gpu_device"],
            rebuilt["gpu_baseline_identity"],
        ),
        "gpu_lock": live_lock,
        "preemption": require_current_no_preemption(),
        "input_authority_sha256": authority_digest(rebuilt["input_authority"]),
        "table_authority_sha256": rebuilt["table_authority"]["authority_sha256"],
        "canonical_authority_sha256": rebuilt["canonical_authority_sha256"],
    }
    resources["pre_spawn_sha256"] = authority_digest(resources)
    return rebuilt, resources


def validate_terminal_gpu_runtime_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"gpu", "gpu_lock", "baseline_compute_processes"}:
        raise GateError("TERMINAL_GPU_RUNTIME_SCHEMA", repr(value))
    _exact_fields(value.get("gpu") or {}, {
        "index": 0, "name": "NVIDIA GeForce RTX 5090", "memory_total_mib": 32607,
    }, "TERMINAL_GPU_RUNTIME_DEVICE")
    lock = value.get("gpu_lock") or {}
    if set(lock) != {"path", "device", "inode"} or lock.get("path") != "/tmp/wrf_gpu2_gpu.lock" or type(lock.get("device")) is not int or lock["device"] <= 0 or type(lock.get("inode")) is not int or lock["inode"] <= 0:
        raise GateError("TERMINAL_GPU_RUNTIME_LOCK", repr(lock))
    rows = value.get("baseline_compute_processes")
    if not isinstance(rows, list):
        raise GateError("TERMINAL_GPU_RUNTIME_BASELINE", repr(rows))
    identities: set[tuple[int, str]] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "pid", "process_name", "start_ticks", "argv0", "exe_path", "exe_device", "exe_inode",
        }:
            raise GateError("TERMINAL_GPU_RUNTIME_BASELINE", repr(row))
        pid, name = row["pid"], row["process_name"]
        if (
            type(pid) is not int or pid <= 1 or type(name) is not str or not name
            or type(row["start_ticks"]) is not int or row["start_ticks"] <= 0
            or type(row["argv0"]) is not str or not row["argv0"]
            or type(row["exe_path"]) is not str or not row["exe_path"]
            or type(row["exe_device"]) is not int or row["exe_device"] <= 0
            or type(row["exe_inode"]) is not int or row["exe_inode"] <= 0
            or (pid, name) in identities
        ):
            raise GateError("TERMINAL_GPU_RUNTIME_BASELINE", repr(row))
        identities.add((pid, name))
        normalized.append(dict(row))
    if normalized != sorted(normalized, key=lambda row: (row["pid"], row["process_name"])):
        raise GateError("TERMINAL_GPU_RUNTIME_BASELINE_ORDER", repr(rows))
    return {"gpu": dict(value["gpu"]), "gpu_lock": dict(lock), "baseline_compute_processes": normalized}


def immediate_terminal_pre_spawn_checks(
    proof_path: Path,
    gpu_runtime_authority: Mapping[str, Any],
    *,
    expected_work_dir: Path,
    now: datetime | None = None,
    interval_seconds: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    rebuilt = verify_terminal_cpu_proof(
        proof_path, expected_work_dir=expected_work_dir,
        now=now, interval_seconds=interval_seconds,
    )
    manager_gpu = validate_terminal_gpu_runtime_authority(gpu_runtime_authority)
    live_lock = validate_live_lock_available()
    if any(live_lock.get(key) != manager_gpu["gpu_lock"].get(key) for key in ("path", "device", "inode")):
        raise GateError("GPU_LOCK_IDENTITY_CHANGED", f"manager={manager_gpu['gpu_lock']} live={live_lock}")
    expected_device = {
        "gpu": manager_gpu["gpu"],
        "baseline_compute_processes": [
            {"pid": row["pid"], "process_name": row["process_name"], "baseline_allowed": True}
            for row in manager_gpu["baseline_compute_processes"]
        ],
        "unexpected_compute_processes": [],
    }
    resources = {
        "checked_utc": now.isoformat(), "cpu_authority_mode": "terminal_no_live_pid",
        "cpu_lane": validate_live_cpu_lane(),
        "gpu_device": validate_live_gpu_device(
            expected_device, manager_gpu["baseline_compute_processes"],
        ),
        "gpu_lock": live_lock, "preemption": require_current_no_preemption(),
        "input_authority_sha256": authority_digest(rebuilt["input_authority"]),
        "table_authority_sha256": rebuilt["table_authority"]["authority_sha256"],
        "canonical_authority_sha256": rebuilt["canonical_authority_sha256"],
    }
    resources["pre_spawn_sha256"] = authority_digest(resources)
    return rebuilt, resources


PRE_SYSTEMD_COMMAND_ENV_KEYS = (
    "HOME", "PATH", "PYTHONPATH", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_PREALLOCATE", "GPUWRF_ALLOCATOR", "GPUWRF_FINITE_CHECK",
    "GPUWRF_NESTED_FUSE", "GPUWRF_NESTED_DEFUSE_COMPILE",
    "GPUWRF_NESTED_PARALLEL_COMPILE", "GPUWRF_NESTED_AOT", "GPUWRF_AOT_VERIFY",
    "GPUWRF_NESTED_ASYNC_OUTPUT", "GPUWRF_NEST_OUTPUT_PIPELINE",
    "GPUWRF_FULL_WRFOUT", "GPUWRF_TRAINING_OUTPUT_SUBSET", "GPUWRF_BATCH_ENSEMBLE",
    "GPUWRF_NESTED_SYNC_MODE", "GPUWRF_JAX_CACHE", "GPUWRF_JAX_CACHE_LOCK",
    "GPUWRF_CACHE", "GPUWRF_JAX_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR",
    "GPUWRF_WRF_ROOT", "GPUWRF_CORRECTED_VALIDATION_CASE",
)
SYSTEMD_USER_ENVIRONMENT = (
    ("XDG_RUNTIME_DIR", "/run/user/1000"),
    ("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus"),
)
COMMAND_ENV_KEYS = (
    *PRE_SYSTEMD_COMMAND_ENV_KEYS[:2],
    *(key for key, _value in SYSTEMD_USER_ENVIRONMENT),
    *PRE_SYSTEMD_COMMAND_ENV_KEYS[2:],
)


def _command_environment(work_dir: Path, table_root: Path) -> dict[str, str]:
    cache = work_dir / "cache" / f"{BASE_SHA[:12]}-{GRID_SHA256[:12]}"
    return {
        "HOME": "<USER_HOME>",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "PYTHONPATH": str(RUN_REPO / "src"),
        "OMP_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "true",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "GPUWRF_ALLOCATOR": "cuda_async",
        "GPUWRF_FINITE_CHECK": "1",
        "GPUWRF_NESTED_FUSE": "0",
        "GPUWRF_NESTED_DEFUSE_COMPILE": "1",
        "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
        "GPUWRF_NESTED_AOT": "1",
        "GPUWRF_AOT_VERIFY": "0",
        "GPUWRF_NESTED_ASYNC_OUTPUT": "1",
        "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
        "GPUWRF_FULL_WRFOUT": "1",
        "GPUWRF_TRAINING_OUTPUT_SUBSET": "0",
        "GPUWRF_BATCH_ENSEMBLE": "1",
        "GPUWRF_NESTED_SYNC_MODE": "root",
        "GPUWRF_JAX_CACHE": "1",
        "GPUWRF_JAX_CACHE_LOCK": "1",
        "GPUWRF_CACHE": str(cache),
        "GPUWRF_JAX_CACHE_DIR": str(cache / "jit"),
        "JAX_COMPILATION_CACHE_DIR": str(cache / "jit"),
        "GPUWRF_WRF_ROOT": str(table_root),
        "GPUWRF_CORRECTED_VALIDATION_CASE": CASE_ID,
    }


def _lock_wrapped_argv(
    environment: Mapping[str, str], payload_argv: Sequence[str], *, lock_wrapper: Path,
) -> list[str]:
    if tuple(environment) != COMMAND_ENV_KEYS:
        raise GateError("COMMAND_ENV_ALLOWLIST", repr(tuple(environment)))
    if (
        tuple((key, environment.get(key)) for key, _value in SYSTEMD_USER_ENVIRONMENT)
        != SYSTEMD_USER_ENVIRONMENT
        or set(COMMAND_ENV_KEYS) - set(PRE_SYSTEMD_COMMAND_ENV_KEYS)
        != {key for key, _value in SYSTEMD_USER_ENVIRONMENT}
        or len(COMMAND_ENV_KEYS) != len(PRE_SYSTEMD_COMMAND_ENV_KEYS) + 2
    ):
        raise GateError("COMMAND_SYSTEMD_USER_ENV_AUTHORITY", repr(environment))
    if any(
        not isinstance(value, str) or "\x00" in value or key.startswith("GPUWRF_GPU_LOCK_")
        for key, value in environment.items()
    ):
        raise GateError("COMMAND_ENV_VALUE", "invalid or pre-attested lock environment")
    payload = list(payload_argv)
    if not payload or "/usr/bin/env" in payload or "-i" in payload:
        raise GateError("COMMAND_POST_LOCK_ENV_CLEAR", repr(payload))
    return [
        "/usr/bin/env", "-i",
        *[f"{key}={environment[key]}" for key in COMMAND_ENV_KEYS],
        str(lock_wrapper),
        "--timeout", "0",
        "--label", "v0234-corrected-fullbuffer-20250228",
        "--intent", LOCK_INTENT,
        "--",
        *payload,
    ]


def _gpuwrf_payload_argv(output_dir: Path, proof_dir: Path, scratch_dir: Path) -> list[str]:
    return [
        "/usr/bin/taskset", "-c", "12-15",
        str(PYTHON_BIN), "-m", "gpuwrf", "run",
        "--namelist", str(CPU_INPUT_DIR / "namelist.input"),
        "--input-dir", str(CPU_INPUT_DIR),
        "--output-dir", str(output_dir),
        "--proof-dir", str(proof_dir),
        "--scratch-dir", str(scratch_dir),
        "--max-dom", "3",
        "--hours", "18",
    ]


def validate_exact_command_argv(command: Mapping[str, Any]) -> dict[str, Any]:
    environment = command.get("environment")
    paths = command.get("paths")
    argv = command.get("argv")
    if not isinstance(environment, dict) or not isinstance(paths, dict) or not isinstance(argv, list):
        raise GateError("COMMAND_EXECUTION_FORM", "environment/paths mappings and argv list required")
    expected_environment = _command_environment(
        Path(str(paths.get("work", ""))), Path(str(environment.get("GPUWRF_WRF_ROOT", ""))),
    )
    if environment != expected_environment or tuple(environment) != COMMAND_ENV_KEYS:
        raise GateError("COMMAND_ENV_ALLOWLIST", repr(environment))
    payload = _gpuwrf_payload_argv(
        Path(str(paths.get("output", ""))),
        Path(str(paths.get("proof", ""))),
        Path(str(paths.get("scratch", ""))),
    )
    expected = _lock_wrapped_argv(environment, payload, lock_wrapper=LOCK_WRAPPER)
    if argv != expected:
        raise GateError("COMMAND_ARGV_ORDER", repr(argv))
    separator = argv.index("--")
    return {
        "env_i_index": 0,
        "lock_wrapper_index": argv.index(str(LOCK_WRAPPER)),
        "separator_index": separator,
        "payload": argv[separator + 1:],
        "lock_environment_injected_after_sanitization": True,
    }


def build_exact_command(
    *,
    early_proof: Mapping[str, Any],
    work_dir: Path,
    candidate_authority: Mapping[str, Any],
    lock_authority: Mapping[str, Any],
) -> dict[str, Any]:
    authority_status = early_proof.get("status")
    if authority_status not in {"EARLY_TWO_FRAME_READY", "TERMINAL_CPU_AUTHORITY_READY"}:
        raise GateError("CPU_AUTHORITY_NOT_READY", str(authority_status))
    if candidate_authority != {
        "path": str(RUN_REPO), "head": BASE_SHA, "clean": True, "detached": True,
    }:
        raise GateError("CANDIDATE_AUTHORITY", repr(candidate_authority))
    if (
        lock_authority.get("commit") != LOCK_COMMIT
        or lock_authority.get("wrapper_sha256") != LOCK_WRAPPER_SHA256
        or lock_authority.get("intent") != LOCK_INTENT
    ):
        raise GateError("LOCK_AUTHORITY", repr(lock_authority))
    table_root = Path(str(early_proof["table_authority"]["snapshot_root"]))
    verify_table_snapshot(early_proof["table_authority"], expected_work_dir=work_dir)
    environment = _command_environment(work_dir, table_root)
    output_dir = work_dir / "gpu-output"
    proof_dir = work_dir / "gpu-proof"
    scratch_dir = work_dir / "scratch"
    pair_state = work_dir / "incremental-pairs.json"
    pair_snapshots = work_dir / "pair-snapshots"
    final_record = work_dir / "final-or-pending.json"
    plot_path = work_dir / "identity-numbers-first.jpg"
    for path in (
        output_dir, proof_dir, scratch_dir, Path(environment["GPUWRF_CACHE"]),
        pair_state, pair_snapshots, final_record, plot_path,
    ):
        if path.exists():
            raise GateError("RUN_PATH_EXISTS", str(path))
    argv = _lock_wrapped_argv(
        environment, _gpuwrf_payload_argv(output_dir, proof_dir, scratch_dir),
        lock_wrapper=LOCK_WRAPPER,
    )
    payload = {
        "schema": COMMAND_SCHEMA,
        "status": "REVIEW_REQUIRED_NOT_EXECUTABLE",
        "candidate": dict(candidate_authority),
        "lock_v2": dict(lock_authority),
        "cpuset": "12-15",
        "environment": environment,
        "argv": argv,
        "shell": False,
        "one_case_only": True,
        "no_warmup_compile": True,
        "paths": {
            "work": str(work_dir.resolve()),
            "output": str(output_dir.resolve()),
            "proof": str(proof_dir.resolve()),
            "scratch": str(scratch_dir.resolve()),
            "cache": str(Path(environment["GPUWRF_CACHE"]).resolve()),
            "pair_state": str(pair_state.resolve()),
            "pair_snapshots": str(pair_snapshots.resolve()),
            "final_record": str(final_record.resolve()),
            "identity_plot": str(plot_path.resolve()),
        },
        "input_authority_sha256": authority_digest(early_proof["input_authority"]),
        "table_authority_sha256": early_proof["table_authority"]["authority_sha256"],
        "cpu_authority_mode": "terminal" if authority_status == "TERMINAL_CPU_AUTHORITY_READY" else "early_live",
        "cpu_authority_sha256": early_proof["canonical_authority_sha256"],
        "identity_policy_sha256": early_proof["identity_policy"]["authority_sha256"],
    }
    payload["argv_authority"] = validate_exact_command_argv(payload)
    payload["command_sha256"] = authority_digest(payload)
    return payload


def _git_blob(repo: Path, commit: str, relative: str) -> tuple[str, bytes]:
    try:
        resolved = subprocess.check_output(
            ("git", "-C", str(repo), "rev-parse", f"{commit}^{{commit}}"),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        blob = subprocess.check_output(
            ("git", "-C", str(repo), "show", f"{resolved}:{relative}"),
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise GateError("GIT_BLOB_AUTHORITY", f"{repo}@{commit}:{relative}") from exc
    return resolved, blob


def validate_pretimestep_failure_evidence() -> dict[str, Any]:
    json_commit, raw_json = _git_blob(
        MANAGER_REPO, PRETIMESTEP_FAILURE_MANAGER_COMMIT, PRETIMESTEP_FAILURE_JSON_RELATIVE,
    )
    log_commit, raw_log = _git_blob(
        MANAGER_REPO, PRETIMESTEP_FAILURE_MANAGER_COMMIT, PRETIMESTEP_FAILURE_LOG_RELATIVE,
    )
    if (
        json_commit != PRETIMESTEP_FAILURE_MANAGER_COMMIT
        or log_commit != PRETIMESTEP_FAILURE_MANAGER_COMMIT
        or sha256_bytes(raw_json) != PRETIMESTEP_FAILURE_JSON_SHA256
        or sha256_bytes(raw_log) != PRETIMESTEP_FAILURE_LOG_SHA256
    ):
        raise GateError("PRETIMESTEP_FAILURE_EVIDENCE_HASH", PRETIMESTEP_FAILURE_MANAGER_COMMIT)
    payload = _strict_json_bytes(raw_json, Path(PRETIMESTEP_FAILURE_JSON_RELATIVE))
    failure = payload.get("failure") or {}
    runtime = payload.get("immutable_runtime") or {}
    observations = payload.get("observations") or {}
    gap = payload.get("runtime_source_authority_gap") or {}
    canonical = gap.get("canonical_source_file") or {}
    if (
        payload.get("schema") != "gpuwrf.v0234.corrected-validation-pre-timestep-failure.v1"
        or payload.get("status") != "ARCHIVED_FAIL_CLOSED"
        or payload.get("classification") != "PRE_TIMESTEP_HARNESS_SOURCE_MATERIALIZATION_FAILURE"
        or failure.get("exception") != "FileNotFoundError"
        or not str(failure.get("missing_path", "")).endswith("/phys/module_ra_rrtmg_lw.F")
        or failure.get("stage") != "nested initialization before forecast timestep 1"
        or runtime.get("tooling_commit") != PRETIMESTEP_SOURCE_AUTHORITY_PARENT_SHA
        or observations.get("gpu_output_frame_count") != 0
        or observations.get("gpu_proof_file_count") != 0
        or observations.get("launcher_log_sha256") != PRETIMESTEP_FAILURE_LOG_SHA256
        or canonical.get("path") != str(WRF_SOURCE_ROOT / "phys/module_ra_rrtmg_lw.F")
        or canonical.get("sha256")
        != WRF_DEPENDENCY_MANIFEST["phys/module_ra_rrtmg_lw.F"]["sha256"]
        or canonical.get("size") != WRF_DEPENDENCY_MANIFEST["phys/module_ra_rrtmg_lw.F"]["size"]
        or payload.get("scientific_verdict", {}).get("physics_result") is not False
    ):
        raise GateError("PRETIMESTEP_FAILURE_EVIDENCE_SEMANTICS", repr(payload))
    return {
        "manager_commit": json_commit,
        "json_path": PRETIMESTEP_FAILURE_JSON_RELATIVE,
        "json_sha256": PRETIMESTEP_FAILURE_JSON_SHA256,
        "log_path": PRETIMESTEP_FAILURE_LOG_RELATIVE,
        "log_sha256": PRETIMESTEP_FAILURE_LOG_SHA256,
        "classification": payload["classification"],
        "observed_missing_relative": "phys/module_ra_rrtmg_lw.F",
        "pre_timestep": True,
        "gpu_output_frames": 0,
        "scientific_result": False,
        "prior_workdir_immutable": True,
        "runtime_ref_consumed_no_reuse": True,
    }


def _git_ref_commit(repo: Path, ref: str) -> str:
    try:
        return subprocess.check_output(
            ("git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise GateError("INDEPENDENT_AUTHORITY_REF", f"{repo}:{ref}") from exc


def _git_blob_at_ref(repo: Path, ref: str, relative: str) -> tuple[str, bytes]:
    commit = _git_ref_commit(repo, ref)
    return _git_blob(repo, commit, relative)


def validate_code_review_acceptance() -> dict[str, Any]:
    resolved, raw = _git_blob_at_ref(
        REVIEW_REPO, CODE_REVIEW_ACCEPTANCE_REF, CODE_REVIEW_ACCEPTANCE_RELATIVE,
    )
    payload = _strict_json_bytes(raw, Path(CODE_REVIEW_ACCEPTANCE_RELATIVE))
    if set(payload) != {
        "schema", "verdict", "reviewer_role", "tooling_commit",
        "manager_policy_sha256", "release_policy_sha256",
        "runtime_marker_status", "reviewed_utc",
    }:
        raise GateError("CODE_REVIEW_SCHEMA", repr(sorted(payload)))
    _exact_fields(payload, {
        "schema": CODE_REVIEW_SCHEMA,
        "verdict": "ACCEPT",
        "reviewer_role": "independent_gpt56_sol_xhigh_critic",
        "manager_policy_sha256": MANAGER_POLICY_SHA256,
        "release_policy_sha256": RELEASE_POLICY_SHA256,
        "runtime_marker_status": "MAY_BE_PENDING_CODE_REVIEW_ONLY",
    }, "CODE_REVIEW_FIELD")
    tooling_commit = str(payload.get("tooling_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", tooling_commit):
        raise GateError("CODE_REVIEW_TOOLING_COMMIT", repr(tooling_commit))
    parse_utc(str(payload["reviewed_utc"]))
    return {
        "repo": str(REVIEW_REPO), "commit": resolved,
        "ref": CODE_REVIEW_ACCEPTANCE_REF,
        "path": CODE_REVIEW_ACCEPTANCE_RELATIVE, "blob_sha256": sha256_bytes(raw),
        "payload": payload,
    }


def validate_executing_tooling(accepted_commit: str) -> dict[str, Any]:
    """Bind the actual standalone orchestration script to the reviewed tree.

    The script imports no repository-local Python modules, so it is the complete
    tracked execution surface.  Repository identity comes from ``__file__``;
    the caller's CWD is irrelevant.
    """
    script = Path(__file__).resolve(strict=True)
    repo = script.parents[1]
    try:
        top = Path(subprocess.check_output(
            ("git", "-C", str(repo), "rev-parse", "--show-toplevel"), text=True,
        ).strip()).resolve(strict=True)
        head = subprocess.check_output(
            ("git", "-C", str(repo), "rev-parse", "HEAD"), text=True,
        ).strip()
        dirty = subprocess.check_output(
            ("git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"), text=True,
        ).strip()
        symbolic = subprocess.run(
            ("git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GateError("EXECUTING_TOOLING_REPO", str(repo)) from exc
    if top != repo or head != accepted_commit or dirty or symbolic.returncode == 0:
        raise GateError(
            "EXECUTING_TOOLING_STATE",
            f"repo={repo} top={top} head={head} accepted={accepted_commit} "
            f"clean={not bool(dirty)} detached={symbolic.returncode != 0}",
        )
    relative = str(script.relative_to(repo))
    resolved, blob = _git_blob(repo, accepted_commit, relative)
    _regular_file_stat(script)
    current = script.read_bytes()
    if resolved != accepted_commit or current != blob:
        raise GateError("EXECUTING_TOOLING_BLOB", relative)
    return {
        "repo": str(repo), "commit": head, "clean": True, "detached": True,
        "tracked_execution_blobs": {relative: sha256_bytes(blob)},
        "threat_model": "cooperative same-UID authority refs; kernel-enforced process isolation is not claimed",
    }


def runtime_nonce_boundary_authority() -> dict[str, Any]:
    path = RUNTIME_AUTHORITY_ROOT
    try:
        info = path.lstat()
    except OSError as exc:
        raise GateError("NONCE_AUTHORITY_DIRECTORY", str(path)) from exc
    if (
        path.is_symlink() or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_nlink < 2
    ):
        raise GateError("NONCE_AUTHORITY_DIRECTORY", repr({
            "path": str(path), "uid": info.st_uid, "mode": oct(stat.S_IMODE(info.st_mode)),
            "nlink": info.st_nlink,
        }))
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        current = path.stat()
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino) or (
            current.st_dev, current.st_ino
        ) != (opened.st_dev, opened.st_ino):
            raise GateError("NONCE_AUTHORITY_REPLACED", str(path))
    finally:
        os.close(fd)
    return {
        "path": str(path.resolve(strict=True)), "device": info.st_dev, "inode": info.st_ino,
        "uid": info.st_uid, "mode": 0o700, "nlink_minimum": 2,
        "mechanism": "nonce_specific_O_EXCL_burn_record",
    }


def _runtime_acceptance_expected(
    *,
    early: Mapping[str, Any],
    command: Mapping[str, Any],
    tooling_commit: str,
    code_review_commit: str,
    nonce: str,
    gpu_runtime_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    common = {
        "schema": RUNTIME_ACCEPTANCE_SCHEMA,
        "verdict": "ACCEPT_ONCE",
        "issuer_role": "manager_runtime_authority",
        "code_review_commit": code_review_commit,
        "tooling_commit": tooling_commit,
        "candidate_sha": BASE_SHA,
        "grid_sha256": GRID_SHA256,
        "manager_policy_sha256": MANAGER_POLICY_SHA256,
        "release_policy_sha256": RELEASE_POLICY_SHA256,
        "nonce": nonce,
        "nonce_authority": runtime_nonce_boundary_authority(),
        "canonical_authority_sha256": early["canonical_authority_sha256"],
        "command_sha256": command["command_sha256"],
        "paths": command["paths"],
    }
    if early.get("status") == "TERMINAL_CPU_AUTHORITY_READY":
        terminal = early["terminal_authority"]
        normalized_gpu = validate_terminal_gpu_runtime_authority(gpu_runtime_authority or {})
        return {
            **common,
            "authority_mode": "terminal_cpu_no_live_pid_no_marker",
            "terminal_contract": {
                "path": str(TERMINAL_CONTRACT), "sha256": TERMINAL_CONTRACT_SHA256,
            },
            "terminal_process": terminal["process"]["artifact"],
            "manifest": terminal["terminal_cpu"]["recomputed"]["manifest_artifact"],
            "seal": {"path": str(INPUT_SEAL), "sha256": PRE_WRF_SEAL_SHA256},
            "input_hashes": {
                name: early["input_authority"][name]["sha256"] for name in INPUT_NAMES
            },
            "gpu_runtime_authority": normalized_gpu,
        }
    return {
        **common,
        "marker": {
            "path": str(EARLY_MARKER),
            "sha256": early["upstream_marker_artifact"]["sha256"],
            "emitted_utc": early["upstream_marker"]["emitted_utc"],
        },
        "resource": {
            "path": str(RESOURCE_PROOF),
            "sha256": early["resource_proof"]["artifact"]["sha256"],
            "checked_utc": early["resource_proof"]["payload"]["checked_utc"],
        },
        "seal": {
            "path": str(INPUT_SEAL),
            "sha256": early["input_seal"]["artifact"]["sha256"],
            "sealed_utc": early["input_seal"]["payload"]["sealed_utc"],
        },
        "input_hashes": early["upstream_marker"]["input_hashes"],
        "qa_artifacts": [
            {
                "path": row["artifact"]["path"], "sha256": row["artifact"]["sha256"],
                "checked_utc": row["upstream"]["checked_utc"],
            }
            for row in early["early_frame_qa"]
        ],
        "cpu_live_identity": early["cpu_live_identity"],
        "gpu_baseline_identity": early["gpu_baseline_identity"],
    }


def validate_runtime_acceptance(
    *,
    relative_path: str,
    command: Mapping[str, Any],
    early: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    review = validate_code_review_acceptance()
    tooling_commit = review["payload"]["tooling_commit"]
    executing = validate_executing_tooling(tooling_commit)
    resolved, raw = _git_blob_at_ref(
        MANAGER_REPO, MANAGER_RUNTIME_ACCEPTANCE_REF, relative_path,
    )
    policy_at_commit = subprocess.check_output(
        ("git", "-C", str(MANAGER_REPO), "show", f"{resolved}:{MANAGER_POLICY_RELATIVE}"),
    )
    if sha256_bytes(policy_at_commit) != MANAGER_POLICY_SHA256:
        raise GateError("MANAGER_POLICY_AT_ACCEPTANCE", resolved)
    payload = _strict_json_bytes(raw, Path(relative_path))
    common_keys = {
        "schema", "verdict", "issuer_role", "code_review_commit", "tooling_commit",
        "candidate_sha", "grid_sha256", "manager_policy_sha256", "release_policy_sha256",
        "issued_utc", "expires_utc", "nonce", "nonce_authority",
        "canonical_authority_sha256", "command_sha256", "paths",
    }
    terminal_mode = early.get("status") == "TERMINAL_CPU_AUTHORITY_READY"
    expected_keys = common_keys | (
        {"authority_mode", "terminal_contract", "terminal_process", "manifest", "seal", "input_hashes", "gpu_runtime_authority"}
        if terminal_mode else
        {"marker", "resource", "seal", "input_hashes", "qa_artifacts", "cpu_live_identity", "gpu_baseline_identity"}
    )
    if set(payload) != expected_keys:
        raise GateError("RUNTIME_ACCEPTANCE_SCHEMA", repr(sorted(payload)))
    nonce = str(payload.get("nonce", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise GateError("RUNTIME_NONCE", repr(nonce))
    expected_path = f"{RUNTIME_ACCEPTANCE_PREFIX}{nonce}.json"
    if relative_path != expected_path:
        raise GateError("RUNTIME_ACCEPTANCE_PATH", relative_path)
    expected = _runtime_acceptance_expected(
        early=early, command=command, tooling_commit=tooling_commit,
        code_review_commit=review["commit"], nonce=nonce,
        gpu_runtime_authority=payload.get("gpu_runtime_authority") if terminal_mode else None,
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise GateError("RUNTIME_ACCEPTANCE_FIELD", f"{key} differs")
    issued = parse_utc(str(payload["issued_utc"]))
    expires = parse_utc(str(payload["expires_utc"]))
    authority_time = parse_utc(str(
        early["terminal_authority"]["contract"]["generated_utc"]
        if terminal_mode else early["upstream_marker"]["emitted_utc"]
    ))
    if (
        issued < authority_time
        or expires <= issued
        or (expires - issued).total_seconds() > RUNTIME_ACCEPTANCE_MAX_LIFETIME_SECONDS
        or now < issued
        or now >= expires
    ):
        raise GateError("RUNTIME_ACCEPTANCE_TIME_ORDER", f"authority={authority_time} issued={issued} expires={expires} now={now}")
    require_fresh_time("runtime_acceptance.issued_utc", issued, now, RUNTIME_ACCEPTANCE_MAX_AGE_SECONDS)
    commit_epoch = int(subprocess.check_output(
        ("git", "-C", str(MANAGER_REPO), "show", "-s", "--format=%ct", resolved), text=True,
    ).strip())
    commit_time = datetime.fromtimestamp(commit_epoch, tz=timezone.utc)
    if commit_time < issued or (commit_time - issued).total_seconds() > 600:
        raise GateError("RUNTIME_ACCEPTANCE_COMMIT_TIME", f"issued={issued} commit={commit_time}")
    return {
        "repo": str(MANAGER_REPO), "commit": resolved, "path": relative_path,
        "ref": MANAGER_RUNTIME_ACCEPTANCE_REF,
        "blob_sha256": sha256_bytes(raw), "payload": payload, "code_review": review,
        "executing_tooling": executing,
    }


def consume_runtime_nonce(
    acceptance: Mapping[str, Any], pre_spawn: Mapping[str, Any], *, now: datetime,
) -> dict[str, Any]:
    payload = acceptance["payload"]
    boundary = runtime_nonce_boundary_authority()
    if payload.get("nonce_authority") != boundary:
        raise GateError("NONCE_AUTHORITY_BINDING", repr(payload.get("nonce_authority")))
    nonce = str(payload["nonce"])
    name = f"consumed-{nonce}.json"
    directory_fd = os.open(
        RUNTIME_AUTHORITY_ROOT,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    entry = {
        "schema": NONCE_BURN_SCHEMA,
        "nonce": nonce,
        "consumed_utc": now.isoformat(),
        "manager_commit": acceptance["commit"],
        "manager_ref": MANAGER_RUNTIME_ACCEPTANCE_REF,
        "acceptance_blob_sha256": acceptance["blob_sha256"],
        "command_sha256": payload["command_sha256"],
        "canonical_authority_sha256": payload["canonical_authority_sha256"],
        "pre_spawn_sha256": pre_spawn["pre_spawn_sha256"],
        "status": "BURNED_BEFORE_SPAWN_NO_REUSE",
    }
    encoded = (json.dumps(entry, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = -1
    try:
        try:
            fd = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o400, dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise GateError("RUNTIME_NONCE_REUSED", nonce) from exc
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise GateError("NONCE_BURN_WRITE", nonce)
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o400)
        opened = os.fstat(fd)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o400 or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise GateError("NONCE_BURN_IDENTITY", nonce)
        os.fsync(directory_fd)
        artifact = {
            "path": str(RUNTIME_AUTHORITY_ROOT / name), "device": opened.st_dev,
            "inode": opened.st_ino, "uid": opened.st_uid, "mode": 0o400,
            "nlink": 1, "sha256": sha256_bytes(encoded), "size": len(encoded),
        }
        return {"entry": entry, "artifact": artifact, "boundary": boundary}
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(directory_fd)


def verify_runtime_nonce_burn(
    burn: Mapping[str, Any], acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    entry = burn.get("entry") or {}
    payload = acceptance.get("payload") or {}
    if (
        entry.get("manager_commit") != acceptance.get("commit")
        or entry.get("acceptance_blob_sha256") != acceptance.get("blob_sha256")
        or entry.get("nonce") != payload.get("nonce")
        or entry.get("command_sha256") != payload.get("command_sha256")
        or entry.get("canonical_authority_sha256") != payload.get("canonical_authority_sha256")
    ):
        raise GateError("NONCE_BURN_ACCEPTANCE_CHANGED", repr(entry))
    if runtime_nonce_boundary_authority() != burn.get("boundary"):
        raise GateError("NONCE_AUTHORITY_REPLACED", str(RUNTIME_AUTHORITY_ROOT))
    artifact = burn.get("artifact") or {}
    path = Path(str(artifact.get("path", "")))
    if path.parent != RUNTIME_AUTHORITY_ROOT or path.name != f"consumed-{entry['nonce']}.json":
        raise GateError("NONCE_BURN_PATH", str(path))
    info = _regular_file_stat(path)
    observed = {
        "path": str(path), "device": info.st_dev, "inode": info.st_ino,
        "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode), "nlink": info.st_nlink,
        "sha256": sha256_file(path), "size": info.st_size,
    }
    if observed != artifact or load_exact_json(path) != entry:
        raise GateError("NONCE_BURN_TAMPERED", str(path))
    return observed


def immutable_file_snapshot(source: Path, target: Path) -> dict[str, Any]:
    if target.exists():
        raise GateError("PAIR_SNAPSHOT_EXISTS", str(target))
    source_lstat = _regular_file_stat(source)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    target_fd = -1
    digest = hashlib.sha256()
    try:
        opened_before = os.fstat(source_fd)
        if (opened_before.st_dev, opened_before.st_ino) != (source_lstat.st_dev, source_lstat.st_ino):
            raise GateError("PAIR_SOURCE_REPLACED", str(source))
        target_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        while True:
            chunk = os.read(source_fd, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                view = view[os.write(target_fd, view):]
        os.fsync(target_fd)
        os.fchmod(target_fd, 0o444)
        target_info = os.fstat(target_fd)
        opened_after = os.fstat(source_fd)
        current = _regular_file_stat(source)
        before_identity = (
            opened_before.st_dev, opened_before.st_ino, opened_before.st_size,
            opened_before.st_mtime_ns, opened_before.st_ctime_ns,
        )
        after_identity = (
            opened_after.st_dev, opened_after.st_ino, opened_after.st_size,
            opened_after.st_mtime_ns, opened_after.st_ctime_ns,
        )
        path_identity = (
            current.st_dev, current.st_ino, current.st_size,
            current.st_mtime_ns, current.st_ctime_ns,
        )
        if before_identity != after_identity or after_identity != path_identity:
            raise GateError("PAIR_SOURCE_CHANGED_DURING_COPY", str(source))
        if target_info.st_size != opened_before.st_size:
            raise GateError("PAIR_SNAPSHOT_SIZE", str(target))
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(source_fd)
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    snapshot = file_authority(target)
    if snapshot["sha256"] != digest.hexdigest() or snapshot["mode"] != 0o444:
        raise GateError("PAIR_SNAPSHOT_VERIFY", str(target))
    return {
        "source_path": str(source.resolve(strict=True)),
        "source_device": source_lstat.st_dev,
        "source_inode": source_lstat.st_ino,
        "source_size": source_lstat.st_size,
        "source_mtime_ns": source_lstat.st_mtime_ns,
        "source_ctime_ns": source_lstat.st_ctime_ns,
        "source_sha256": digest.hexdigest(),
        "snapshot": snapshot,
    }


def _json_float_input_identity(value: float) -> dict[str, Any]:
    observed = float(value)
    if math.isfinite(observed):
        return {"finite": True, "value": observed}
    if math.isnan(observed):
        label = "NAN"
    elif observed > 0.0:
        label = "POSITIVE_INFINITY"
    else:
        label = "NEGATIVE_INFINITY"
    return {"finite": False, "value": label}


def relative_rmse_fields(rmse: float, reference_rms: float) -> dict[str, Any]:
    """Represent informational relative RMSE without denominator substitution."""
    rmse_value = float(rmse)
    reference_value = float(reference_rms)
    inputs = {
        "rmse": _json_float_input_identity(rmse_value),
        "reference_rms": _json_float_input_identity(reference_value),
    }
    result: dict[str, Any] = {
        "reference_rms": reference_value if math.isfinite(reference_value) else None,
        "relative_rmse": None,
        "relative_rmse_status": "OVERFLOW_OR_UNBOUNDED",
        "relative_rmse_inputs": inputs,
    }
    if (
        not math.isfinite(rmse_value)
        or not math.isfinite(reference_value)
        or rmse_value < 0.0
        or reference_value < 0.0
    ):
        return result
    if reference_value == 0.0:
        if rmse_value == 0.0:
            result["relative_rmse"] = 0.0
            result["relative_rmse_status"] = "EXACT_ZERO_REFERENCE"
        else:
            result["relative_rmse_status"] = "UNDEFINED_ZERO_REFERENCE"
        return result
    try:
        ratio = rmse_value / reference_value
    except OverflowError:
        return result
    if math.isfinite(ratio):
        result["relative_rmse"] = ratio
        result["relative_rmse_status"] = "FINITE"
    return result


def _metric(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, Any]:
    if cpu.shape != gpu.shape:
        raise GateError("PAIR_SHAPE", f"cpu={cpu.shape} gpu={gpu.shape}")
    if not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise GateError("PAIR_NONFINITE", "paired arrays must be finite")
    delta = gpu.astype(np.float64) - cpu.astype(np.float64)
    flat_cpu = cpu.astype(np.float64).ravel()
    flat_gpu = gpu.astype(np.float64).ravel()
    rmse = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(flat_cpu * flat_cpu)))
    if flat_cpu.size >= 2 and np.std(flat_cpu) > 0.0 and np.std(flat_gpu) > 0.0:
        correlation: float | None = float(np.corrcoef(flat_cpu, flat_gpu)[0, 1])
    else:
        correlation = None
    sufficient = {
        "n": int(delta.size),
        "sum_delta": float(delta.sum()),
        "sum_abs_delta": float(np.abs(delta).sum()),
        "sum_sq_delta": float(np.sum(delta * delta)),
        "sum_sq_cpu": float(np.sum(flat_cpu * flat_cpu)),
        "sum_cpu": float(flat_cpu.sum()),
        "sum_gpu": float(flat_gpu.sum()),
        "sum_sq_gpu": float(np.sum(flat_gpu * flat_gpu)),
        "sum_cpu_gpu": float(np.sum(flat_cpu * flat_gpu)),
    }
    return {
        "n": int(delta.size),
        "bias": float(delta.mean()),
        "mae": float(np.abs(delta).mean()),
        "rmse": rmse,
        **relative_rmse_fields(rmse, reference_rms),
        "max_abs": float(np.abs(delta).max(initial=0.0)),
        "correlation": correlation,
        "sufficient": sufficient,
    }


def compare_pair(
    cpu_path: Path,
    gpu_path: Path,
    valid: datetime,
    *,
    cpu_snapshot: Mapping[str, Any] | None = None,
    gpu_snapshot: Mapping[str, Any] | None = None,
    frozen_geometry_policy: str = "exact",
) -> dict[str, Any]:
    if frozen_geometry_policy not in {"exact", "terminal_nested_boundary_v1"}:
        raise GateError("PAIR_FROZEN_GEOMETRY_POLICY", frozen_geometry_policy)
    cpu_time_path = Path(str(cpu_snapshot["source_path"])) if cpu_snapshot else cpu_path
    gpu_time_path = Path(str(gpu_snapshot["source_path"])) if gpu_snapshot else gpu_path
    if parse_wrfout_time(cpu_time_path) != valid or parse_wrfout_time(gpu_time_path) != valid:
        raise GateError("UNMATCHED_VALID_TIME", valid.isoformat())
    frozen_geo, frozen_geo_authority = load_frozen_geometry()
    with Dataset(cpu_path) as cpu, Dataset(gpu_path) as gpu:
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        if _decoded_times(cpu) != stamp or _decoded_times(gpu) != stamp:
            raise GateError("PAIR_TIME_METADATA", stamp)
        cpu_fields = set(cpu.variables)
        gpu_fields = set(gpu.variables)
        if frozen_geometry_policy == "terminal_nested_boundary_v1":
            _require_terminal_geometry(cpu, frozen_geo, f"cpu:{valid.isoformat()}")
            _require_terminal_geometry(gpu, frozen_geo, f"gpu:{valid.isoformat()}")
        else:
            require_exact_frozen_geometry(cpu, frozen_geo, f"cpu:{valid.isoformat()}")
            require_exact_frozen_geometry(gpu, frozen_geo, f"gpu:{valid.isoformat()}")
        common = sorted(cpu_fields & gpu_fields)
        cpu_only = sorted(cpu_fields - gpu_fields)
        gpu_only = sorted(gpu_fields - cpu_fields)
        compatible: list[str] = []
        incompatible: list[dict[str, Any]] = []
        for field in common:
            cpu_var = cpu.variables[field]
            gpu_var = gpu.variables[field]
            if field == "Times":
                continue
            if (
                not np.issubdtype(cpu_var.dtype, np.number)
                or not np.issubdtype(gpu_var.dtype, np.number)
                or cpu_var.shape != gpu_var.shape
                or cpu_var.dimensions != gpu_var.dimensions
            ):
                incompatible.append({
                    "field": field, "cpu_shape": list(cpu_var.shape),
                    "gpu_shape": list(gpu_var.shape), "cpu_dtype": str(cpu_var.dtype),
                    "gpu_dtype": str(gpu_var.dtype),
                    "cpu_dimensions": list(cpu_var.dimensions),
                    "gpu_dimensions": list(gpu_var.dimensions),
                })
                continue
            compatible.append(field)
        missing_strict_cpu = sorted(set(STRICT_FIELDS) - cpu_fields)
        missing_strict_gpu = sorted(set(STRICT_FIELDS) - gpu_fields)
        incompatible_names = {row["field"] for row in incompatible}
        if missing_strict_cpu or missing_strict_gpu or incompatible_names & set(STRICT_FIELDS):
            raise GateError(
                "PAIR_STRICT_FIELDS",
                f"cpu={missing_strict_cpu} gpu={missing_strict_gpu} incompatible={sorted(incompatible_names)}",
            )
        for field in STATIC_GATE_FIELDS:
            if field not in cpu_fields or field not in gpu_fields or field in incompatible_names:
                raise GateError("PAIR_STATIC_INVENTORY", field)
        metrics: dict[str, Any] = {}
        field_policy: dict[str, str] = {}
        for field in compatible:
            values = _metric(_variable_array(cpu, field), _variable_array(gpu, field))
            if field in STRICT_RMSE_LIMITS:
                values["gate"] = "STRICT_POOLED_RMSE"
                values["limit"] = STRICT_RMSE_LIMITS[field]
                values["per_frame_pass_diagnostic"] = values["rmse"] <= values["limit"]
                values["pass"] = None
                field_policy[field] = "STRICT_POOLED_RMSE"
            elif field in STATIC_GATE_FIELDS:
                values["gate"] = "STATIC_EXACT"
                values["limit"] = 0.0
                values["pass"] = values["max_abs"] == 0.0
                field_policy[field] = "STATIC_EXACT"
            elif field in REPORT_ONLY_FIELDS:
                values["gate"] = "OWNER_DIRECTIVE_REPORT_ONLY"
                values["limit"] = None
                values["pass"] = None
                field_policy[field] = "OWNER_DIRECTIVE_REPORT_ONLY"
            else:
                values["gate"] = "UNTOLERANCED_REPORT_ONLY"
                values["limit"] = None
                values["pass"] = None
                field_policy[field] = "UNTOLERANCED_REPORT_ONLY"
            metrics[field] = values
    return {
        "valid_time": valid.isoformat(),
        "cpu_snapshot_path": str(cpu_path.resolve(strict=True)),
        "gpu_snapshot_path": str(gpu_path.resolve(strict=True)),
        "cpu_sha256": sha256_file(cpu_path),
        "gpu_sha256": sha256_file(gpu_path),
        "frozen_geo_sha256": frozen_geo_authority["sha256"],
        "frozen_geometry_policy": frozen_geometry_policy,
        "cpu_snapshot_authority": dict(cpu_snapshot or {}),
        "gpu_snapshot_authority": dict(gpu_snapshot or {}),
        "inventory": {
            "cpu_fields": sorted(cpu_fields), "gpu_fields": sorted(gpu_fields),
            "common_compatible_numeric_fields": compatible,
            "common_incompatible_fields": incompatible,
            "cpu_only_fields": cpu_only, "gpu_only_fields": gpu_only,
        },
        "field_policy": field_policy,
        "metrics": metrics,
        "per_frame_static_pass": all(
            field in metrics and metrics[field]["pass"] is True for field in STATIC_GATE_FIELDS
        ),
        "strict_final_status": "PENDING_POOLED_55_FRAME_AGGREGATION",
    }


def validate_pair_state_integrity(state: Mapping[str, Any], *, recompute: bool) -> dict[str, Any]:
    unsigned = dict(state)
    recorded_digest = unsigned.pop("authority_sha256", None)
    if authority_digest(unsigned) != recorded_digest:
        raise GateError("PAIR_STATE_DIGEST", "state changed")
    _exact_fields(state, {
        "schema": PAIR_SCHEMA,
        "case_id": CASE_ID,
        "grid_id": GRID_ID,
        "strict_fields": list(STRICT_FIELDS),
        "static_gate_fields": list(STATIC_GATE_FIELDS),
        "report_only_fields": list(REPORT_ONLY_FIELDS),
    }, "PAIR_STATE_FIELD")
    frozen_geometry_policy = state.get("frozen_geometry_policy")
    if frozen_geometry_policy not in {"exact", "terminal_nested_boundary_v1"}:
        raise GateError("PAIR_FROZEN_GEOMETRY_POLICY", repr(frozen_geometry_policy))
    policy = validate_identity_policy_authority()
    if state.get("identity_policy") != policy:
        raise GateError("PAIR_POLICY_AUTHORITY", "manager/release policy differs")
    rows = state.get("pairs")
    if not isinstance(rows, list) or state.get("matched_count") != len(rows):
        raise GateError("PAIR_STATE_COUNT", repr(state.get("matched_count")))
    seen: set[datetime] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise GateError("PAIR_ROW_SCHEMA", repr(row))
        valid = parse_utc(str(row.get("valid_time")))
        if valid not in EXPECTED_TIMES or valid in seen:
            raise GateError("PAIR_ROW_TIME", valid.isoformat())
        seen.add(valid)
        cpu_path = Path(str(row.get("cpu_snapshot_path")))
        gpu_path = Path(str(row.get("gpu_snapshot_path")))
        for key, path in (("cpu_snapshot_authority", cpu_path), ("gpu_snapshot_authority", gpu_path)):
            authority = row.get(key) or {}
            snapshot = authority.get("snapshot") or {}
            verify_pinned_file_metadata(snapshot, "PAIR_SNAPSHOT_AUTHORITY", required_mode=0o444)
            if str(path.resolve(strict=True)) != snapshot.get("path"):
                raise GateError("PAIR_SNAPSHOT_AUTHORITY", f"{valid.isoformat()}:{key}:path")
        if recompute and (
            sha256_file(cpu_path) != row.get("cpu_sha256")
            or sha256_file(gpu_path) != row.get("gpu_sha256")
        ):
            raise GateError("PAIRED_FRAME_MUTATED", valid.isoformat())
        if recompute and compare_pair(
            cpu_path, gpu_path, valid,
            cpu_snapshot=row["cpu_snapshot_authority"],
            gpu_snapshot=row["gpu_snapshot_authority"],
            frozen_geometry_policy=frozen_geometry_policy,
        ) != row:
            raise GateError("PAIR_METRIC_SUBSTITUTION", valid.isoformat())
    if [row["valid_time"] for row in rows] != sorted(row["valid_time"] for row in rows):
        raise GateError("PAIR_ROW_ORDER", "pairs not sorted")
    pooled, inventory = pooled_metrics(rows)
    if state.get("pooled_metrics") != pooled or state.get("complete_field_inventory") != inventory:
        raise GateError("PAIR_POOLED_SUBSTITUTION", "pooled metrics or field inventory differ")
    return dict(state)


def pooled_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = sorted({field for row in rows for field in row.get("metrics", {})})
    pooled: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    for field in fields:
        present = [row for row in rows if field in row.get("metrics", {})]
        metrics = [row["metrics"][field] for row in present]
        sufficient = [metric["sufficient"] for metric in metrics]
        n = sum(item["n"] for item in sufficient)
        sum_delta = sum(item["sum_delta"] for item in sufficient)
        sum_abs = sum(item["sum_abs_delta"] for item in sufficient)
        sum_sq_delta = sum(item["sum_sq_delta"] for item in sufficient)
        sum_sq_cpu = sum(item["sum_sq_cpu"] for item in sufficient)
        sum_cpu = sum(item["sum_cpu"] for item in sufficient)
        sum_gpu = sum(item["sum_gpu"] for item in sufficient)
        sum_sq_gpu = sum(item["sum_sq_gpu"] for item in sufficient)
        sum_cpu_gpu = sum(item["sum_cpu_gpu"] for item in sufficient)
        rmse = math.sqrt(sum_sq_delta / n)
        reference_rms = math.sqrt(sum_sq_cpu / n)
        variance_cpu = max(0.0, sum_sq_cpu - sum_cpu * sum_cpu / n)
        variance_gpu = max(0.0, sum_sq_gpu - sum_gpu * sum_gpu / n)
        denominator = math.sqrt(variance_cpu * variance_gpu)
        correlation = None if denominator == 0.0 else (sum_cpu_gpu - sum_cpu * sum_gpu / n) / denominator
        status = present[0]["field_policy"][field]
        if any(row["field_policy"][field] != status for row in present):
            raise GateError("PAIR_FIELD_POLICY_DRIFT", field)
        result = {
            "n": n,
            "frame_count": len(present),
            "bias": sum_delta / n,
            "mae": sum_abs / n,
            "pooled_rmse": rmse,
            **relative_rmse_fields(rmse, reference_rms),
            "max_abs": max(metric["max_abs"] for metric in metrics),
            "correlation": correlation,
            "policy": status,
            "threshold": STRICT_RMSE_LIMITS.get(field),
        }
        if status == "STRICT_POOLED_RMSE":
            result["pass"] = len(present) == 55 and rmse <= STRICT_RMSE_LIMITS[field]
        elif status == "STATIC_EXACT":
            result["pass"] = all(metric["max_abs"] == 0.0 for metric in metrics)
        else:
            result["pass"] = None
        pooled[field] = result
        inventory[field] = {
            "policy": status,
            "present_valid_times": [row["valid_time"] for row in present],
            "missing_valid_times": [row["valid_time"] for row in rows if field not in row.get("metrics", {})],
        }
    all_cpu = sorted({field for row in rows for field in row["inventory"]["cpu_fields"]})
    all_gpu = sorted({field for row in rows for field in row["inventory"]["gpu_fields"]})
    every_field = sorted(set(all_cpu) | set(all_gpu))
    complete_fields: dict[str, Any] = {}
    for field in every_field:
        if field in STRICT_FIELDS:
            policy = "STRICT_POOLED_RMSE"
        elif field in STATIC_GATE_FIELDS:
            policy = "STATIC_EXACT"
        elif field in REPORT_ONLY_FIELDS:
            policy = "OWNER_DIRECTIVE_REPORT_ONLY"
        else:
            policy = "UNTOLERANCED_REPORT_ONLY"
        complete_fields[field] = {
            "policy": policy,
            "cpu_present_valid_times": [row["valid_time"] for row in rows if field in row["inventory"]["cpu_fields"]],
            "gpu_present_valid_times": [row["valid_time"] for row in rows if field in row["inventory"]["gpu_fields"]],
            "compatible_metric_valid_times": inventory.get(field, {}).get("present_valid_times", []),
            "missing_cpu_valid_times": [row["valid_time"] for row in rows if field not in row["inventory"]["cpu_fields"]],
            "missing_gpu_valid_times": [row["valid_time"] for row in rows if field not in row["inventory"]["gpu_fields"]],
        }
    complete = {
        "fields": complete_fields,
        "all_cpu_fields": all_cpu,
        "all_gpu_fields": all_gpu,
        "cpu_only_fields_union": sorted(set(all_cpu) - set(all_gpu)),
        "gpu_only_fields_union": sorted(set(all_gpu) - set(all_cpu)),
        "per_frame_incompatible": {
            row["valid_time"]: row["inventory"]["common_incompatible_fields"]
            for row in rows if row["inventory"]["common_incompatible_fields"]
        },
    }
    return pooled, complete


def incremental_pair(
    cpu_dir: Path,
    gpu_dir: Path,
    state_path: Path,
    *,
    terminal_cache: dict[str, Any] | None = None,
    terminal_mode: bool = False,
) -> dict[str, Any]:
    frozen_geometry_policy = "terminal_nested_boundary_v1" if terminal_mode else "exact"
    cpu = discover_d03_frames(cpu_dir)
    gpu = discover_d03_frames(gpu_dir)
    common = sorted(set(cpu) & set(gpu))
    unexpected = sorted(set(common) - set(EXPECTED_TIMES))
    if unexpected:
        raise GateError("PAIR_UNEXPECTED_TIME", unexpected[0].isoformat())
    if state_path.exists():
        _regular_file_stat(state_path)
        state = load_exact_json(state_path)
        validate_pair_state_integrity(state, recompute=False)
        if state.get("frozen_geometry_policy") != frozen_geometry_policy:
            raise GateError("PAIR_FROZEN_GEOMETRY_POLICY", "state/mode mismatch")
    else:
        state = {
            "schema": PAIR_SCHEMA,
            "case_id": CASE_ID,
            "grid_id": GRID_ID,
            "strict_fields": list(STRICT_FIELDS),
            "static_gate_fields": list(STATIC_GATE_FIELDS),
            "report_only_fields": list(REPORT_ONLY_FIELDS),
            "frozen_geometry_policy": frozen_geometry_policy,
            "identity_policy": validate_identity_policy_authority(),
            "closure_observations": {},
            "pairs": [],
        }
    existing = {parse_utc(row["valid_time"]): row for row in state["pairs"]}
    if sorted(existing) != list(EXPECTED_TIMES[:len(existing)]):
        raise GateError("PAIR_NONCONTIGUOUS_HISTORY", repr(sorted(existing)))
    for valid, row in existing.items():
        for side, discovered in (("cpu", cpu), ("gpu", gpu)):
            if valid not in discovered:
                raise GateError("PAIR_SOURCE_REMOVED_AFTER_SNAPSHOT", f"{side}:{valid.isoformat()}")
            authority = row[f"{side}_snapshot_authority"]
            try:
                source_pinned_metadata(authority, "PAIR_SOURCE_CHANGED_AFTER_SNAPSHOT")
            except GateError as exc:
                raise GateError("PAIR_SOURCE_CHANGED_AFTER_SNAPSHOT", f"{side}:{valid.isoformat()}") from exc
            if str(discovered[valid].resolve(strict=True)) != authority.get("source_path"):
                raise GateError("PAIR_SOURCE_CHANGED_AFTER_SNAPSHOT", f"{side}:{valid.isoformat()}:path")
    prior_observations = state.get("closure_observations")
    if not isinstance(prior_observations, dict):
        raise GateError("PAIR_CLOSURE_OBSERVATIONS", repr(prior_observations))
    frontier = EXPECTED_TIMES[len(existing)] if len(existing) < len(EXPECTED_TIMES) else None
    frontier_candidates: set[datetime] = set()
    if frontier is not None:
        frontier_candidates.add(frontier)
        index = EXPECTED_TIMES.index(frontier)
        if index + 1 < len(EXPECTED_TIMES):
            frontier_candidates.add(EXPECTED_TIMES[index + 1])
    current_observations: dict[str, Any] = {}
    stable_closed: set[datetime] = set()
    for valid in sorted(frontier_candidates & set(common)):
        key = valid.isoformat()
        detail: dict[str, Any] = {}
        for side, frames in (("cpu", cpu), ("gpu", gpu)):
            qa = qa_d03_frame(frames[valid], valid)
            authority = file_authority(frames[valid])
            if qa["sha256"] != authority["sha256"]:
                raise GateError("PAIR_CLOSURE_CHANGED_DURING_QA", f"{side}:{key}")
            detail[side] = {"authority": authority, "qa": qa}
        current_observations[key] = detail
        if prior_observations.get(key) == detail:
            stable_closed.add(valid)
    state["closure_observations"] = current_observations
    closed: list[datetime] = []
    if frontier is not None and frontier in common:
        valid = frontier
        index = EXPECTED_TIMES.index(valid)
        if index < len(EXPECTED_TIMES) - 1:
            successor = EXPECTED_TIMES[index + 1]
            if successor in stable_closed:
                closed.append(valid)
        elif CPU_MANIFEST.exists():
            if terminal_cache is None:
                cpu_terminal = validate_cpu_terminal_manifest(CPU_MANIFEST)
                if gpu_terminal_inventory_ready(gpu_dir, terminal_mode=terminal_mode):
                    validate_gpu_completion_frames(gpu_dir, terminal_mode=terminal_mode)
                    closed.append(valid)
            elif "cpu" in terminal_cache and gpu_terminal_inventory_ready(gpu_dir, terminal_mode=terminal_mode):
                verify_cached_terminal_metadata(terminal_cache["cpu"], "CPU_TERMINAL_CACHE_CHANGED")
                if "gpu" not in terminal_cache:
                    terminal_cache["gpu"] = validate_gpu_completion_frames(gpu_dir, terminal_mode=terminal_mode)
                else:
                    verify_cached_terminal_metadata(terminal_cache["gpu"], "GPU_TERMINAL_CACHE_CHANGED")
                closed.append(valid)
    for valid in closed:
        if valid not in EXPECTED_TIMES:
            raise GateError("PAIR_UNEXPECTED_TIME", valid.isoformat())
        if valid not in existing:
            stamp = valid.strftime("%Y%m%dT%H%M%S")
            snapshot_dir = state_path.parent / "pair-snapshots" / stamp
            if snapshot_dir.exists():
                raise GateError("PAIR_SNAPSHOT_ORPHAN", str(snapshot_dir))
            try:
                cpu_authority = immutable_file_snapshot(cpu[valid], snapshot_dir / "cpu.nc")
                gpu_authority = immutable_file_snapshot(gpu[valid], snapshot_dir / "gpu.nc")
                os.chmod(snapshot_dir, 0o555)
                row = compare_pair(
                    snapshot_dir / "cpu.nc", snapshot_dir / "gpu.nc", valid,
                    cpu_snapshot=cpu_authority, gpu_snapshot=gpu_authority,
                    frozen_geometry_policy=frozen_geometry_policy,
                )
            except BaseException:
                if snapshot_dir.exists():
                    os.chmod(snapshot_dir, 0o755)
                    for child in snapshot_dir.iterdir():
                        child.chmod(0o644)
                    shutil.rmtree(snapshot_dir)
                raise
            state["pairs"].append(row)
    state["pairs"] = sorted(state["pairs"], key=lambda row: row["valid_time"])
    state["matched_count"] = len(state["pairs"])
    state["cpu_only_times"] = [value.isoformat() for value in sorted(set(cpu) - set(gpu))]
    state["gpu_only_times"] = [value.isoformat() for value in sorted(set(gpu) - set(cpu))]
    state["pooled_metrics"], state["complete_field_inventory"] = pooled_metrics(state["pairs"])
    state["final_status"] = "FINAL_PENDING" if len(state["pairs"]) < 55 else "PAIRS_COMPLETE_PENDING_CPU_MANIFEST"
    unsigned = dict(state)
    unsigned.pop("authority_sha256", None)
    state["authority_sha256"] = authority_digest(unsigned)
    atomic_write_json(state_path, state)
    return state


def cross_bind_terminal_pair_sources(
    state: Mapping[str, Any], cpu_terminal: Mapping[str, Any], gpu_completion: Mapping[str, Any],
    gpu_dir: Path,
) -> dict[str, Any]:
    cpu_rows = cpu_terminal["recomputed"]["raw_qa"]["d03"]
    gpu_rows = gpu_completion["frame_authority"]["d03"]
    bound: dict[str, Any] = {}
    for row in state["pairs"]:
        valid = parse_utc(row["valid_time"])
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        name = f"wrfout_d03_{stamp}"
        expected = {
            "cpu": cpu_rows[stamp]["authority"],
            "gpu": gpu_rows[stamp],
        }
        for side, terminal in expected.items():
            source = row[f"{side}_snapshot_authority"]
            wanted_path = CPU_INPUT_DIR / name if side == "cpu" else gpu_dir / name
            current = _regular_file_stat(wanted_path)
            if (
                source.get("source_path") != str(wanted_path.resolve(strict=True))
                or source.get("source_device") != current.st_dev
                or source.get("source_inode") != current.st_ino
                or source.get("source_size") != current.st_size
                or source.get("source_mtime_ns") != current.st_mtime_ns
                or source.get("source_ctime_ns") != current.st_ctime_ns
                or source.get("source_sha256") != terminal.get("sha256")
                or row.get(f"{side}_sha256") != terminal.get("sha256")
                or sha256_file(wanted_path) != terminal.get("sha256")
            ):
                raise GateError("PAIR_TERMINAL_SOURCE_BINDING", f"{side}:{stamp}")
        if row.get("frozen_geo_sha256") != GRID_SHA256 or row.get("per_frame_static_pass") is not True:
            raise GateError("PAIR_TERMINAL_GEOMETRY", stamp)
        bound[stamp] = {
            "cpu_sha256": expected["cpu"]["sha256"],
            "gpu_sha256": expected["gpu"]["sha256"],
            "frozen_geo_sha256": GRID_SHA256,
        }
    if len(bound) != 55:
        raise GateError("PAIR_TERMINAL_SOURCE_COUNT", str(len(bound)))
    return bound


def _cpu_frame_qa(
    path: Path,
    stamp: str,
    required: set[str],
    geo: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    authority = file_authority(path)
    with Dataset(path, "r") as dataset:
        missing = sorted(required - set(dataset.variables))
        if missing or _decoded_times(dataset) != stamp:
            raise GateError("CPU_RAW_QA", f"{path}: missing={missing} time={_decoded_times(dataset)!r}")
        ranges: dict[str, list[float]] = {}
        for name in required - {"Times"}:
            variable = dataset.variables[name]
            if not np.issubdtype(variable.dtype, np.number):
                continue
            values = np.asarray(np.ma.filled(variable[:], np.nan))
            if not np.isfinite(values).all():
                raise GateError("CPU_RAW_NONFINITE", f"{path}:{name}")
            low, high = float(values.min()), float(values.max())
            if name in {"T2", "U10", "V10", "PSFC"}:
                allowed = PHYSICAL_BOUNDS[name]
                if low < allowed[0] or high > allowed[1]:
                    raise GateError("CPU_RAW_UNPHYSICAL", f"{path}:{name}={low, high}")
                ranges[name] = [low, high]
            elif name == "Q2" and (low < 0.0 or high > 0.04):
                raise GateError("CPU_RAW_UNPHYSICAL", f"{path}:{name}={low, high}")
            elif name == "TSK" and (low < 250.0 or high > 340.0):
                raise GateError("CPU_RAW_UNPHYSICAL", f"{path}:{name}={low, high}")
            if name in {"Q2", "TSK"}:
                ranges[name] = [low, high]
        if geo is not None:
            for name in STATIC_GATE_FIELDS:
                actual = _static_plane(dataset, name)
                source = geo[name]
                if actual.shape != source.shape or not np.array_equal(actual, source):
                    raise GateError("CPU_RAW_GEOMETRY", f"{path}:{name}")
    if file_authority(path) != authority:
        raise GateError("CPU_RAW_CHANGED_DURING_QA", str(path))
    return {"authority": authority, "surface_ranges": ranges}


def _strict_json_lines(path: Path) -> list[dict[str, Any]]:
    before = file_authority(path)
    raw = path.read_bytes()
    if file_authority(path) != before:
        raise GateError("PAIR_INDEX_CHANGED_DURING_READ", str(path))
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise GateError("PAIR_INDEX_EMPTY_LINE", f"{path}:{index}")
        rows.append(_strict_json_bytes(line, Path(f"{path}:{index}")))
    return rows


def _expected_cpu_pair_rows(raw: Mapping[str, Mapping[str, Path]], thin: Mapping[str, Path]) -> list[dict[str, Any]]:
    parent_times = [value.strftime("%Y-%m-%d_%H:%M:%S") for value in EXPECTED_PARENT_TIMES]
    child_times = [value.strftime("%Y-%m-%d_%H:%M:%S") for value in EXPECTED_TIMES]
    rows: list[dict[str, Any]] = []
    for stamp in parent_times:
        rows.append({
            "ladder": "nine_to_three", "relation": "exact", "valid_time": stamp,
            "input_raw": str(raw["d01"][stamp]), "target_raw": str(raw["d02"][stamp]),
        })
    for index, stamp in enumerate(child_times):
        valid = EXPECTED_TIMES[index]
        common = {
            "valid_time": stamp, "target_raw": str(raw["d03"][stamp]),
            "target_thin": str(thin[stamp]),
        }
        rows.append({
            "ladder": "aifs_to_1km", "relation": "physical_target_only_se_cache_pending",
            "forcing_frame_index": index, **common,
        })
        if stamp in raw["d02"]:
            rows.append({
                "ladder": "three_to_one", "relation": "exact",
                "input_raw": str(raw["d02"][stamp]), **common,
            })
        else:
            before = max(value for value in EXPECTED_PARENT_TIMES if value < valid)
            after = min(value for value in EXPECTED_PARENT_TIMES if value > valid)
            rows.append({
                "ladder": "three_to_one", "relation": "bracketed",
                "input_raw_before": str(raw["d02"][before.strftime("%Y-%m-%d_%H:%M:%S")]),
                "input_raw_after": str(raw["d02"][after.strftime("%Y-%m-%d_%H:%M:%S")]),
                "weight_after": (valid.minute % 60) / 60.0, **common,
            })
    return rows


def validate_cpu_terminal_manifest(
    path: Path = CPU_MANIFEST,
    *,
    interval_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if path.resolve() != CPU_MANIFEST.resolve():
        raise GateError("CPU_MANIFEST_CANONICAL_PATH", str(path))
    reject_revoked_authority(EARLY_MARKER)
    terminal_marker = load_exact_json(EARLY_MARKER)
    _exact_fields(terminal_marker, {
        "schema": UPSTREAM_MARKER_SCHEMA, "status": "EARLY_GPU_READY",
        "case_id": CASE_ID, "grid_id": GRID_ID, "grid_sha256": GRID_SHA256,
    }, "CPU_TERMINAL_MARKER")
    _seal, _seal_artifact, terminal_inputs = validate_input_seal(
        terminal_marker, INPUT_SEAL,
        interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    manifest, manifest_artifact = stable_json_authority(
        path, interval_seconds=interval_seconds, sleep_fn=sleep_fn,
    )
    if set(manifest) != {
        "schema", "status", "qa_status", "case_id", "grid_id", "grid_sha256",
        "engine", "raw_frame_counts", "thin_frame_count", "pair_counts", "pair_index",
        "pair_index_rows", "surface_ranges", "all_required_fields_finite",
        "all_grid_fields_match_frozen_geo", "forcing_path", "forcing_sha256",
        "gpu_identity_contract", "eligibility", "generated_utc",
    }:
        raise GateError("CPU_MANIFEST_SCHEMA", repr(sorted(manifest)))
    _exact_fields(manifest, {
        "schema": "tenerife_fullbuffer_cpu_oracle_v1", "status": "complete",
        "qa_status": "pass", "case_id": CASE_ID, "grid_id": GRID_ID,
        "grid_sha256": GRID_SHA256, "engine": "cpu_wrf_max_dom3",
        "raw_frame_counts": {"d01": 19, "d02": 19, "d03": 55},
        "thin_frame_count": 55,
        "pair_counts": {"nine_to_three": 19, "aifs_to_1km_physical_target": 55, "three_to_one": 55},
        "pair_index": str(CPU_PAIR_INDEX), "pair_index_rows": 129,
        "all_required_fields_finite": True, "all_grid_fields_match_frozen_geo": True,
        "forcing_path": "<DATA_ROOT>/alisios/forcing/historical/aifs/20250228_18z/aifs_pure_wps_20250228_18z.grib2",
        "forcing_sha256": AIFS_SHA256,
        "eligibility": {
            "aifs_to_1km": "pending_se_cache_and_model_qa",
            "three_to_one": "pending_training_cache_and_cv",
            "nine_to_three": "unchanged_parent_pair_evidence",
        },
    }, "CPU_MANIFEST_FIELD")
    parse_utc(str(manifest["generated_utc"]))
    expected_stamps = {
        "d01": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in EXPECTED_PARENT_TIMES],
        "d02": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in EXPECTED_PARENT_TIMES],
        "d03": [value.strftime("%Y-%m-%d_%H:%M:%S") for value in EXPECTED_TIMES],
    }
    gpu_contract = manifest.get("gpu_identity_contract")
    if not isinstance(gpu_contract, dict) or set(gpu_contract) != {
        "consumer", "matched_raw_domain", "valid_times", "exclude_fields",
    } or any((
        gpu_contract.get("consumer") != "wrf_gpu 0:1",
        gpu_contract.get("matched_raw_domain") != "d03",
        gpu_contract.get("valid_times") != expected_stamps["d03"],
        gpu_contract.get("exclude_fields") not in (
            ["QVAPOR", "RAINNC"], list(REPORT_ONLY_FIELDS),
        ),
    )):
        raise GateError("CPU_MANIFEST_GPU_CONTRACT", repr(gpu_contract))
    raw_by_stamp: dict[str, dict[str, Path]] = {}
    for domain, stamps in expected_stamps.items():
        frames = strict_domain_frame_inventory(CPU_INPUT_DIR, domain)
        mapped = {valid.strftime("%Y-%m-%d_%H:%M:%S"): frame for valid, frame in frames.items()}
        if sorted(mapped) != stamps:
            raise GateError("CPU_RAW_INVENTORY", f"{domain}: observed={len(mapped)}")
        raw_by_stamp[domain] = mapped
    geo, geo_authority = load_frozen_geometry()
    raw_qa: dict[str, dict[str, Any]] = {}
    aggregate_ranges: dict[str, list[float]] = {}
    for domain, stamps in expected_stamps.items():
        raw_qa[domain] = {}
        for stamp in stamps:
            qa = _cpu_frame_qa(
                raw_by_stamp[domain][stamp], stamp,
                CPU_CHILD_REQUIRED if domain == "d03" else CPU_PARENT_REQUIRED,
                geo if domain == "d03" else None,
            )
            raw_qa[domain][stamp] = qa
            if domain == "d03":
                for field, bounds in qa["surface_ranges"].items():
                    current = aggregate_ranges.setdefault(field, [float("inf"), float("-inf")])
                    current[0], current[1] = min(current[0], bounds[0]), max(current[1], bounds[1])
    if manifest.get("surface_ranges") != aggregate_ranges:
        raise GateError("CPU_MANIFEST_RANGE_SUBSTITUTION", repr(manifest.get("surface_ranges")))
    thin: dict[str, Path] = {}
    thin_qa: dict[str, Any] = {}
    expected_thin_paths = {
        CPU_THIN_DIR / f"wrfout_d03_{stamp}.thin.nc" for stamp in expected_stamps["d03"]
    }
    observed_thin_paths = set(CPU_THIN_DIR.glob("wrfout_d03_*.thin.nc")) if CPU_THIN_DIR.is_dir() else set()
    if observed_thin_paths != expected_thin_paths:
        raise GateError("CPU_THIN_INVENTORY", f"expected=55 observed={len(observed_thin_paths)}")
    for stamp in expected_stamps["d03"]:
        thin_path = CPU_THIN_DIR / f"wrfout_d03_{stamp}.thin.nc"
        authority = file_authority(thin_path)
        with Dataset(thin_path, "r") as thin_ds, Dataset(raw_by_stamp["d03"][stamp], "r") as raw_ds:
            missing = sorted(CPU_THIN_REQUIRED - set(thin_ds.variables))
            if missing or _decoded_times(thin_ds) != stamp:
                raise GateError("CPU_THIN_QA", f"{thin_path}: missing={missing}")
            if getattr(thin_ds, "grid_id", None) != GRID_ID or getattr(thin_ds, "geo_em_sha256", None) != GRID_SHA256:
                raise GateError("CPU_THIN_AUTHORITY", str(thin_path))
            if getattr(thin_ds, "source_raw", None) != str(raw_by_stamp["d03"][stamp]):
                raise GateError("CPU_THIN_SOURCE", str(thin_path))
            require_exact_frozen_geometry(thin_ds, geo, f"thin:{stamp}")
            for field in CPU_THIN_REQUIRED - {"Times"}:
                values = np.asarray(np.ma.filled(thin_ds.variables[field][:], np.nan))
                if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                    raise GateError("CPU_THIN_NONFINITE", f"{thin_path}:{field}")
            for field in ("XLAT", "XLONG", "LANDMASK", "HGT", "U10", "V10", "T2", "Q2", "TSK", "PSFC"):
                raw_values = np.asarray(raw_ds.variables[field][:]).squeeze()
                thin_values = np.asarray(thin_ds.variables[field][:]).squeeze()
                if raw_values.shape != thin_values.shape or not np.array_equal(raw_values, thin_values):
                    raise GateError("CPU_RAW_THIN_MISMATCH", f"{stamp}:{field}")
        if file_authority(thin_path) != authority:
            raise GateError("CPU_THIN_CHANGED_DURING_QA", str(thin_path))
        thin[stamp] = thin_path
        thin_qa[stamp] = authority
    pair_rows = _strict_json_lines(CPU_PAIR_INDEX)
    expected_rows = _expected_cpu_pair_rows(raw_by_stamp, thin)
    if pair_rows != expected_rows:
        raise GateError("CPU_PAIR_INDEX_SUBSTITUTION", "canonical pair rows differ")
    qa_commit, qa_source = _git_blob(EARLY_AUTHORITY_REPO, EARLY_AUTHORITY_COMMIT, CPU_QA_SOURCE)
    if qa_commit != EARLY_AUTHORITY_COMMIT or sha256_bytes(qa_source) != CPU_QA_SOURCE_SHA256:
        raise GateError("CPU_QA_PRODUCER_AUTHORITY", qa_commit)
    identity_policy = validate_identity_policy_authority()
    recomputed = {
        "producer": {"commit": qa_commit, "path": CPU_QA_SOURCE, "sha256": CPU_QA_SOURCE_SHA256},
        "manifest_artifact": manifest_artifact,
        "raw_counts": {domain: len(rows) for domain, rows in raw_qa.items()},
        "raw_qa": raw_qa,
        "thin_count": len(thin_qa), "thin_qa": thin_qa,
        "pair_index": file_authority(CPU_PAIR_INDEX), "pair_index_rows": len(pair_rows),
        "pair_counts": manifest["pair_counts"], "geo_authority": geo_authority,
        "identity_policy": identity_policy,
        "terminal_input_authority": terminal_inputs,
        "producer_declared_legacy_exclude_fields": gpu_contract["exclude_fields"],
        "effective_owner_report_only_fields": list(REPORT_ONLY_FIELDS),
    }
    recomputed["authority_sha256"] = authority_digest(recomputed)
    return {"manifest": manifest, "recomputed": recomputed}


def _identity_plot(state: Mapping[str, Any], target: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized: dict[str, float] = {}
    for field in STRICT_FIELDS:
        normalized[field] = float(state["pooled_metrics"][field]["pooled_rmse"]) / STRICT_RMSE_LIMITS[field]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    fields = list(STRICT_FIELDS)
    values = [normalized[field] for field in fields]
    colors = ["#1a9850" if value <= 1.0 else "#d73027" for value in values]
    bars = axis.bar(fields, values, color=colors)
    axis.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
    axis.set_ylabel("pooled RMSE / frozen released limit")
    axis.set_title(f"{GRID_ID} — 55 matched d03 frames")
    axis.set_ylim(0.0, max(1.15, max(values, default=0.0) * 1.15))
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160, format="jpeg", metadata={"Creator": "gpuwrf-v0234-corrected-validation"})
    plt.close(fig)
    return str(target.resolve())


def final_verdict(
    cpu_manifest_path: Path,
    pair_state_path: Path,
    plot_path: Path,
    gpu_dir: Path,
    *,
    terminal_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if cpu_manifest_path.resolve() != CPU_MANIFEST.resolve():
        raise GateError("CPU_MANIFEST_CANONICAL_PATH", str(cpu_manifest_path))
    expected_gpu_dir = pair_state_path.parent / "gpu-output"
    expected_plot = pair_state_path.parent / "identity-numbers-first.jpg"
    if (
        pair_state_path.name != "incremental-pairs.json"
        or gpu_dir.resolve() != expected_gpu_dir.resolve()
        or plot_path.resolve() != expected_plot.resolve()
    ):
        raise GateError("FINAL_CANONICAL_PATHS", f"pairs={pair_state_path} gpu={gpu_dir} plot={plot_path}")
    if not cpu_manifest_path.exists():
        return {"verdict": "FINAL_PENDING", "reason": "cpu_manifest_absent"}
    if not pair_state_path.exists():
        return {"verdict": "FINAL_PENDING", "reason": "pair_state_absent"}
    state = load_exact_json(pair_state_path)
    validate_pair_state_integrity(state, recompute=True)
    if state.get("schema") != PAIR_SCHEMA or state.get("matched_count") != 55:
        return {"verdict": "FINAL_PENDING", "reason": "matched_frames_not_55"}
    if terminal_cache is not None and "cpu" in terminal_cache:
        verify_cached_terminal_metadata(terminal_cache["cpu"], "CPU_TERMINAL_CACHE_CHANGED")
        cpu_terminal = terminal_cache["cpu"]
        verify_cached_terminal_manifest_sha(cpu_terminal, cpu_manifest_path)
    else:
        cpu_terminal = validate_cpu_terminal_manifest(cpu_manifest_path)
        if terminal_cache is not None:
            terminal_cache["cpu"] = cpu_terminal
    if terminal_cache is not None and "gpu" in terminal_cache:
        verify_cached_terminal_metadata(terminal_cache["gpu"], "GPU_TERMINAL_CACHE_CHANGED")
        gpu_completion = terminal_cache["gpu"]
    else:
        gpu_completion = validate_gpu_completion_frames(gpu_dir)
        if terminal_cache is not None:
            terminal_cache["gpu"] = gpu_completion
    expected_iso = [value.isoformat() for value in EXPECTED_TIMES]
    if [row["valid_time"] for row in state["pairs"]] != expected_iso:
        raise GateError("FINAL_PAIR_TIMES", "unmatched or reordered valid times")
    terminal_pair_sources = cross_bind_terminal_pair_sources(
        state, cpu_terminal, gpu_completion, gpu_dir,
    )
    failures = []
    for field in STRICT_FIELDS:
        metric = state["pooled_metrics"].get(field)
        if not metric or metric["frame_count"] != 55 or metric["pass"] is not True:
            failures.append({"field": field, "gate": "STRICT_POOLED_RMSE"})
    for field in STATIC_GATE_FIELDS:
        metric = state["pooled_metrics"].get(field)
        if metric is None or metric["frame_count"] != 55 or metric["pass"] is not True:
            failures.append({"field": field, "gate": "STATIC_EXACT"})
    if failures:
        return {"verdict": "FAIL_IDENTITY", "failures": failures}
    plot = _identity_plot(state, plot_path)
    return {
        "verdict": "PASS",
        "cpu_manifest_sha256": sha256_file(cpu_manifest_path),
        "matched_frames": 55,
        "strict_fields": list(STRICT_FIELDS),
        "owner_report_only_fields": list(REPORT_ONLY_FIELDS),
        "pooled_metrics": state["pooled_metrics"],
        "complete_field_inventory": state["complete_field_inventory"],
        "identity_policy": state["identity_policy"],
        "identity_plot_jpeg": plot,
        "telegram_delivery_owner": "manager",
        "cpu_terminal_authority": cpu_terminal,
        "gpu_completion_authority": gpu_completion,
        "terminal_pair_source_authority": terminal_pair_sources,
    }


def monitor_once(
    *,
    early_proof: Mapping[str, Any],
    expected_work_dir: Path,
    cpu_manifest_path: Path,
    terminal_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reject_revoked_authority(EARLY_MARKER)
    require_current_no_preemption()
    if sha256_file(EARLY_MARKER) != early_proof["upstream_marker_artifact"]["sha256"]:
        raise GateError("MARKER_MUTATED_STOP_AND_RELEASE", str(EARLY_MARKER))
    if sha256_file(RESOURCE_PROOF) != early_proof["resource_proof"]["artifact"]["sha256"]:
        raise GateError("RESOURCE_MUTATED_STOP_AND_RELEASE", str(RESOURCE_PROOF))
    if sha256_file(INPUT_SEAL) != early_proof["input_seal"]["artifact"]["sha256"]:
        raise GateError("SEAL_MUTATED_STOP_AND_RELEASE", str(INPUT_SEAL))
    verify_table_snapshot(
        early_proof["table_authority"], expected_work_dir=expected_work_dir,
    )
    current_fd_evidence = scan_current_input_open_fds(early_proof["input_authority"])
    for detail in early_proof["input_authority"].values():
        verify_pinned_file_metadata(
            detail, "INPUT_AUTHORITY_CHANGED_DURING_RUN", required_mode=0o444,
        )
        if sha256_file(Path(detail["path"])) != detail["sha256"]:
            raise GateError("INPUT_MUTATED_DURING_RUN", detail["path"])
    for frame in early_proof["early_frame_qa"]:
        qa_artifact = frame["artifact"]
        if sha256_file(Path(qa_artifact["path"])) != qa_artifact["sha256"]:
            raise GateError("EARLY_QA_MUTATED_DURING_RUN", qa_artifact["path"])
        independent = frame["independent"]
        path = Path(independent["path"])
        if not path.is_file() or sha256_file(path) != independent["sha256"]:
            raise GateError("EARLY_FRAME_MUTATED_DURING_RUN", str(path))
    if cpu_manifest_path.resolve() != CPU_MANIFEST.resolve():
        raise GateError("CPU_MANIFEST_CANONICAL_PATH", str(cpu_manifest_path))
    if cpu_manifest_path.exists():
        if terminal_cache is not None and "cpu" in terminal_cache:
            verify_cached_terminal_metadata(terminal_cache["cpu"], "CPU_TERMINAL_CACHE_CHANGED")
            terminal = terminal_cache["cpu"]
        else:
            terminal = validate_cpu_terminal_manifest(cpu_manifest_path)
            if terminal_cache is not None:
                terminal_cache["cpu"] = terminal
        return {
            "status": "CONTINUE_CPU_TERMINAL_AUTHENTICATED",
            "cpu_terminal": terminal, "current_input_fd_evidence": current_fd_evidence,
        }
    verify_live_cpu_identity(
        early_proof["upstream_marker"]["cpu_oracle"], early_proof["cpu_live_identity"],
    )
    return {"status": "CONTINUE_CPU_LIVE", "current_input_fd_evidence": current_fd_evidence}


def active_monitor_poll(
    *,
    early_proof: Mapping[str, Any],
    expected_work_dir: Path,
    cpu_manifest_path: Path,
    cpu_dir: Path,
    gpu_dir: Path,
    pair_state_path: Path,
    terminal_cache: dict[str, Any],
) -> dict[str, Any]:
    """Run safety authority before the bounded pairing frontier on every poll."""
    safety = monitor_once(
        early_proof=early_proof, cpu_manifest_path=cpu_manifest_path,
        expected_work_dir=expected_work_dir, terminal_cache=terminal_cache,
    )
    pairs = incremental_pair(
        cpu_dir, gpu_dir, pair_state_path, terminal_cache=terminal_cache,
    )
    return {"safety": safety, "pairs": pairs}


def terminal_monitor_once(
    *,
    terminal_proof: Mapping[str, Any],
    expected_work_dir: Path,
    terminal_cache: dict[str, Any],
) -> dict[str, Any]:
    require_current_no_preemption()
    for marker in (EARLY_MARKER, EARLY_REVOKED, EARLY_INVALIDATED):
        if marker.exists() or marker.is_symlink():
            raise GateError("TERMINAL_REJECTS_EARLY_MARKER", str(marker))
    verify_table_snapshot(
        terminal_proof["table_authority"], expected_work_dir=expected_work_dir,
    )
    terminal = terminal_proof["terminal_authority"]
    for artifact in (
        terminal["artifact"], terminal["process"]["artifact"],
        terminal["terminal_cpu"]["recomputed"]["manifest_artifact"],
        terminal["terminal_cpu"]["recomputed"]["pair_index"],
    ):
        verify_pinned_file_metadata(artifact, "TERMINAL_AUTHORITY_CHANGED_DURING_RUN")
    current_fd_evidence = scan_current_input_open_fds(terminal_proof["input_authority"])
    for detail in terminal_proof["input_authority"].values():
        verify_pinned_file_metadata(
            detail, "INPUT_AUTHORITY_CHANGED_DURING_RUN", required_mode=0o444,
        )
        if sha256_file(Path(detail["path"])) != detail["sha256"]:
            raise GateError("INPUT_MUTATED_DURING_RUN", detail["path"])
    if "cpu" not in terminal_cache:
        terminal_cache["cpu"] = terminal_proof["terminal_cpu"]
    verify_cached_terminal_metadata(terminal_cache["cpu"], "CPU_TERMINAL_CACHE_CHANGED")
    verify_cached_terminal_manifest_sha(terminal_cache["cpu"], CPU_MANIFEST)
    return {
        "status": "CONTINUE_TERMINAL_CPU_AUTHENTICATED",
        "cpu_terminal": terminal_cache["cpu"],
        "current_input_fd_evidence": current_fd_evidence,
    }


def terminal_active_monitor_poll(
    *,
    terminal_proof: Mapping[str, Any],
    expected_work_dir: Path,
    gpu_dir: Path,
    pair_state_path: Path,
    terminal_cache: dict[str, Any],
) -> dict[str, Any]:
    safety = terminal_monitor_once(
        terminal_proof=terminal_proof, expected_work_dir=expected_work_dir,
        terminal_cache=terminal_cache,
    )
    pairs = incremental_pair(
        CPU_INPUT_DIR, gpu_dir, pair_state_path,
        terminal_cache=terminal_cache, terminal_mode=True,
    )
    return {"safety": safety, "pairs": pairs}


def verify_cached_terminal_metadata(payload: object, code: str) -> dict[str, Any]:
    """Cheaply re-pin every cached file identity without NetCDF or SHA reads."""
    checked: dict[str, Any] = {}
    referenced: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if "path" in value and ("size" in value or "bytes" in value):
                path = str(value["path"])
                referenced.add(path)
            if all(key in value for key in (
                "path", "device", "inode", "mtime_ns", "ctime_ns", "mode",
            )) and ("size" in value or "bytes" in value):
                path = str(value["path"])
                if path not in checked:
                    authority = {
                        "path": path, "device": value["device"], "inode": value["inode"],
                        "size": value.get("size", value.get("bytes")), "mtime_ns": value["mtime_ns"],
                        "ctime_ns": value["ctime_ns"], "mode": value["mode"],
                    }
                    if not isinstance(value["mode"], int):
                        raise GateError(code, f"{path}:non-numeric mode authority")
                    checked[path] = verify_pinned_file_metadata(authority, code)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(payload)
    missing = sorted(referenced - set(checked))
    if missing:
        raise GateError(code, f"cached terminal files lack complete metadata: {missing}")
    if not checked:
        raise GateError(code, "cached terminal authority contains no pinned files")
    return {"checked_files": len(checked), "paths": sorted(checked)}


def verify_cached_terminal_manifest_sha(
    cpu_terminal: Mapping[str, Any], canonical_manifest: Path,
) -> dict[str, Any]:
    artifact = cpu_terminal.get("recomputed", {}).get("manifest_artifact")
    if not isinstance(artifact, Mapping):
        raise GateError("CPU_TERMINAL_CACHE_MANIFEST", "missing cached manifest artifact")
    if artifact.get("path") != str(canonical_manifest.resolve(strict=True)):
        raise GateError("CPU_TERMINAL_CACHE_MANIFEST", "canonical path differs")
    digest = sha256_file(canonical_manifest)
    if digest != artifact.get("sha256"):
        raise GateError("CPU_TERMINAL_CACHE_MANIFEST_SHA", digest)
    return {"path": artifact["path"], "sha256": digest}


def _gpu_completion_schedule(terminal_mode: bool) -> dict[str, tuple[datetime, ...]]:
    return {
        "d01": TERMINAL_D01_TIMES if terminal_mode else EXPECTED_PARENT_TIMES,
        "d02": EXPECTED_PARENT_TIMES,
        "d03": EXPECTED_TIMES,
    }


def gpu_terminal_inventory_ready(directory: Path, *, terminal_mode: bool = False) -> bool:
    expected = {domain: set(times) for domain, times in _gpu_completion_schedule(terminal_mode).items()}
    for domain, wanted in expected.items():
        if set(strict_domain_frame_inventory(directory, domain)) != wanted:
            return False
    return True


def validate_gpu_completion_frames(directory: Path, *, terminal_mode: bool = False) -> dict[str, Any]:
    inventories: dict[str, dict[datetime, Path]] = {
        domain: strict_domain_frame_inventory(directory, domain) for domain in ("d01", "d02", "d03")
    }
    expected = _gpu_completion_schedule(terminal_mode)
    for domain, wanted in expected.items():
        if set(inventories[domain]) != set(wanted):
            raise GateError(
                "GPU_FRAME_SET",
                f"{domain}: expected={len(wanted)} observed={len(inventories[domain])} "
                f"missing={len(set(wanted) - set(inventories[domain]))}",
            )
    qa: dict[str, list[dict[str, Any]]] = {}
    frame_authority: dict[str, dict[str, dict[str, Any]]] = {}
    for domain, wanted in expected.items():
        rows: list[dict[str, Any]] = []
        frame_authority[domain] = {}
        for valid in wanted:
            path = inventories[domain][valid]
            if domain == "d03":
                row = qa_d03_frame(path, valid)
                authority = file_authority(path)
                if row["sha256"] != authority["sha256"]:
                    raise GateError("GPU_FRAME_CHANGED_DURING_QA", str(path))
                rows.append(row)
                frame_authority[domain][valid.strftime("%Y-%m-%d_%H:%M:%S")] = authority
                continue
            _regular_file_stat(path)
            with Dataset(path, "r") as dataset:
                missing = sorted(set(FRAME_REQUIRED_FIELDS) - set(dataset.variables))
                if missing or _decoded_times(dataset) != valid.strftime("%Y-%m-%d_%H:%M:%S"):
                    raise GateError("GPU_PARENT_FRAME_QA", f"{path}: missing={missing}")
                ranges: dict[str, list[float]] = {}
                for field in FRAME_REQUIRED_FIELDS:
                    if field == "Times":
                        continue
                    values = _variable_array(dataset, field)
                    if not np.isfinite(values).all():
                        raise GateError("GPU_PARENT_FRAME_NONFINITE", f"{path}:{field}")
                    low, high = float(values.min()), float(values.max())
                    if field in PHYSICAL_BOUNDS and (
                        low < PHYSICAL_BOUNDS[field][0] or high > PHYSICAL_BOUNDS[field][1]
                    ):
                        raise GateError("GPU_PARENT_FRAME_UNPHYSICAL", f"{path}:{field}={low, high}")
                    ranges[field] = [low, high]
            rows.append({
                "path": str(path.resolve(strict=True)), "valid_time": valid.isoformat(),
                "sha256": sha256_file(path), "size": path.stat().st_size, "ranges": ranges,
            })
            frame_authority[domain][valid.strftime("%Y-%m-%d_%H:%M:%S")] = file_authority(path)
        qa[domain] = rows
    return {
        "counts": {domain: len(rows) for domain, rows in qa.items()},
        "expected_counts": {"d01": 19, "d02": 19, "d03": 55},
        "valid_times": {domain: [valid.isoformat() for valid in expected[domain]] for domain in expected},
        "frame_sha256": {
            domain: {Path(item["path"]).name: item["sha256"] for item in rows}
            for domain, rows in qa.items()
        },
        "frame_authority": frame_authority,
    }


def run_reviewed_command(
    command: Mapping[str, Any],
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    monitor: Callable[[], None] | None = None,
    poll_seconds: float = 5.0,
) -> int:
    if command.get("shell") is not False or not isinstance(command.get("argv"), list):
        raise GateError("COMMAND_EXECUTION_FORM", "argv list and shell=false required")
    validate_exact_command_argv(command)
    process = popen_factory(command["argv"], shell=False, start_new_session=True)
    while process.poll() is None:
        try:
            if monitor is not None:
                monitor()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            raise
        time.sleep(poll_seconds)
    return int(process.returncode)


def assert_test_affinity() -> None:
    affinity = set(os.sched_getaffinity(0))
    if affinity != set(VALIDATION_LANE):
        raise GateError("TEST_CPUSET", f"expected 12-15, got {sorted(affinity)}")


def generate_cpu_proof(output: Path) -> dict[str, Any]:
    assert_test_affinity()
    root = Path(__file__).resolve().parents[1]
    contract, contract_artifact = stable_json_authority(
        CANONICAL_V2_CONTRACT, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    binding, binding_artifact = stable_json_authority(
        CANONICAL_BINDING, interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    if (
        contract_artifact["sha256"] != CANONICAL_V2_CONTRACT_SHA256
        or binding_artifact["sha256"] != CANONICAL_BINDING_SHA256
    ):
        raise GateError("CPU_PROOF_STATIC_AUTHORITY", "contract or binding changed")
    retired_runtime_path = CPU_RUN_ROOT / "launch_runtime.json"
    retired_runtime = load_exact_json(retired_runtime_path)
    not_emitted_path = CPU_RUN_ROOT / "EARLY_GPU_READY_NOT_EMITTED.json"
    not_emitted = load_exact_json(not_emitted_path)
    if (
        retired_runtime.get("status") != "failed_final_qa"
        or retired_runtime.get("authority") != "revoked_until_early_gpu_ready_marker"
        or not_emitted.get("status") != "cpu_exited_without_early_gpu_ready"
        or EARLY_MARKER.exists() or EARLY_MARKER.is_symlink()
    ):
        raise GateError("CPU_PROOF_LIVE_AUTHORITY_STATE", "expected expired/no-marker state")
    static_prterun = validate_prterun_authority()
    terminal_authority = validate_terminal_contract_authority(
        interval_seconds=0.0, sleep_fn=lambda _: None,
    )
    test_path = root / "tests/test_v0234_corrected_fullbuffer_gate.py"
    collect = subprocess.run(
        (sys.executable, "-m", "pytest", "--collect-only", "-q", str(test_path)),
        cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if collect.returncode:
        raise GateError("TEST_COLLECT", collect.stdout)
    nodes = sorted(
        line.strip() for line in collect.stdout.splitlines()
        if line.startswith("tests/test_v0234_corrected_fullbuffer_gate.py::")
    )
    run = subprocess.run(
        (sys.executable, "-m", "pytest", "-q", str(test_path)),
        cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if run.returncode:
        raise GateError("CPU_TESTS", run.stdout)
    tree = ast.parse(Path(__file__).read_text())
    forbidden_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            forbidden_imports.extend(alias.name for alias in node.names if alias.name in {"jax", "gpuwrf"} or alias.name.startswith(("jax.", "gpuwrf.")))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module in {"jax", "gpuwrf"} or node.module.startswith(("jax.", "gpuwrf.")):
                forbidden_imports.append(node.module)
    if forbidden_imports:
        raise GateError("FORBIDDEN_IMPORT", repr(forbidden_imports))
    source = Path(__file__).read_text()
    if str(root / "scripts/with_gpu_lock.sh") in source:
        raise GateError("LEGACY_LOCK", "repo-local lock referenced")
    pretimestep_failure = validate_pretimestep_failure_evidence()
    proof = {
        "schema": "gpuwrf.v0234.corrected-pipeline-cpu-proof.v1",
        "verdict": "FOCUSED_CPU_REPAIR_AND_ADVERSARIAL_GATES_PASS",
        "base_sha": BASE_SHA,
        "repair_parent_sha": REPAIR_PARENT_SHA,
        "final_repair_parent_sha": FINAL_REPAIR_PARENT_SHA,
        "monitor_repair_parent_sha": MONITOR_REPAIR_PARENT_SHA,
        "cache_repair_parent_sha": CACHE_REPAIR_PARENT_SHA,
        "prelaunch_env_order_parent_sha": PRELAUNCH_ENV_ORDER_PARENT_SHA,
        "prelaunch_systemd_env_parent_sha": PRELAUNCH_SYSTEMD_ENV_PARENT_SHA,
        "pretimestep_source_authority_parent_sha": PRETIMESTEP_SOURCE_AUTHORITY_PARENT_SHA,
        "pretimestep_source_authority_repair_parent_sha": PRETIMESTEP_SOURCE_AUTHORITY_REPAIR_PARENT_SHA,
        "pretimestep_failure_evidence": pretimestep_failure,
        "critic_authority": {
            "commit": CRITIC_COMMIT,
            "report_sha256": CRITIC_REPORT_SHA256,
            "final_rereview_commit": FINAL_REREVIEW_COMMIT,
            "final_rereview_report_sha256": FINAL_REREVIEW_REPORT_SHA256,
            "monitor_rereview_commit": MONITOR_REREVIEW_COMMIT,
            "monitor_rereview_report_sha256": MONITOR_REREVIEW_REPORT_SHA256,
            "cache_rereview_commit": CACHE_REREVIEW_COMMIT,
            "cache_rereview_report_sha256": CACHE_REREVIEW_REPORT_SHA256,
            "authority_v2_rereview_commit": AUTHORITY_V2_REREVIEW_COMMIT,
            "authority_v2_rereview_report_sha256": AUTHORITY_V2_REREVIEW_REPORT_SHA256,
            "combined_terminal_rereview_commit": COMBINED_TERMINAL_REREVIEW_COMMIT,
            "combined_terminal_rereview_report_sha256": COMBINED_TERMINAL_REREVIEW_REPORT_SHA256,
            "prelaunch_env_order_rereview_commit": PRELAUNCH_ENV_ORDER_REREVIEW_COMMIT,
            "prelaunch_env_order_rereview_report_sha256": PRELAUNCH_ENV_ORDER_REREVIEW_REPORT_SHA256,
            "prelaunch_env_order_failed_smoke_sha256": PRELAUNCH_ENV_ORDER_SMOKE_SHA256,
            "pretimestep_source_authority_review_commit": PRETIMESTEP_SOURCE_AUTHORITY_REVIEW_COMMIT,
            "pretimestep_source_authority_review_sha256": PRETIMESTEP_SOURCE_AUTHORITY_REVIEW_SHA256,
        },
        "case_id": CASE_ID,
        "grid_id": GRID_ID,
        "grid_sha256": GRID_SHA256,
        "launch_plan_sha256": LAUNCH_PLAN_SHA256,
        "aifs_sha256": AIFS_SHA256,
        "early_gpu_ready_authority": {
            "wrf_downscale_commit": EARLY_AUTHORITY_COMMIT,
            "emitter_sha256": EARLY_AUTHORITY_SOURCE_SHA256,
            "report_commit": EARLY_AUTHORITY_REPORT_COMMIT,
            "report_sha256": EARLY_AUTHORITY_REPORT_SHA256,
            "marker_schema": UPSTREAM_MARKER_SCHEMA,
            "marker_path": str(EARLY_MARKER),
            "regular_d03_groups": list(EARLY_REQUIRED_STAMPS),
            "initialization_00_00_counted": False,
            "full_cpu_verdict_at_emit": "pending_55_of_55",
            "revocation_action": "STOP_FAIL_CLOSED_AND_RELEASE_LOCK",
            "real_marker_consumed_in_this_cpu_sprint": False,
            "live_probe_executed_in_this_cpu_sprint": False,
            "reason": "CPU exited before authority publication; no marker was minted and live authority is expired",
        },
        "canonical_authority_v2": {
            "contract_path": str(CANONICAL_V2_CONTRACT),
            "contract_sha256": contract_artifact["sha256"],
            "superseded_contract_sha256": "d7b266bf490e72b8c35977995defd4e6357bc98b5e85467fba4a26ebf30ad877",
            "binding_path": str(CANONICAL_BINDING),
            "binding_sha256": binding_artifact["sha256"],
            "forcing_preflight_sha256": FORCING_PREFLIGHT_SHA256,
            "pre_wrf_seal_sha256": PRE_WRF_SEAL_SHA256,
            "launch_time_checksums": {
                "path": str(LAUNCH_TIME_CHECKSUMS), "sha256": LAUNCH_TIME_CHECKSUMS_SHA256,
            },
            "current_checksums": {
                "path": str(CURRENT_CHECKSUMS), "sha256": CURRENT_CHECKSUMS_SHA256,
            },
            "current_launch_plan_sha256": LAUNCH_PLAN_SHA256,
            "historical_launch_plan_sha256": binding["input_seal_launch_plan_sha256"],
            "historical_and_current_roles_distinct": True,
            "live_root_pid_and_start_ticks_source_required": "hash-and-inode-authenticated canonical launch_runtime.json",
            "prterun": static_prterun,
            "current_fd_policy": "read_only_exact_inode_holders_only",
            "live_authority_status": "LIVE_AUTHORITY_EXPIRED",
            "real_live_authority_proof_generated": False,
            "runtime_authority_usable": False,
            "expired_runtime_evidence": file_authority(retired_runtime_path),
            "not_emitted_evidence": file_authority(not_emitted_path),
            "terminal_contract_adaptation_in_scope": True,
        },
        "wrf_table_sha256": TABLE_HASHES,
        "wrf_initialization_dependency_authority": {
            "schema": TABLE_SCHEMA,
            "source_root": str(WRF_SOURCE_ROOT),
            "namelist_sha256": sha256_file(CPU_INPUT_DIR / "namelist.input"),
            "namelist_dependency_selection": validate_corrected_namelist(
                CPU_INPUT_DIR / "namelist.input",
            )["physics_dependency_selection"],
            "scheme_dependency_map": WRF_SCHEME_DEPENDENCIES,
            "inventory": WRF_DEPENDENCY_MANIFEST,
            "loader_source_authority": WRF_LOADER_SOURCE_AUTHORITY,
            "private_regular_0444_exact_tree": True,
            "no_mutable_cwd_or_ambient_fallback": True,
            "cpu_loader_smoke_test": (
                "test_real_maxdom3_initialization_dependency_inventory_and_cpu_loader_smoke"
            ),
            "cpu_loader_smoke_real_jax_backend_imported": False,
            "cpu_loader_smoke_production_lw_entry": (
                "gpuwrf.physics.rrtmg_lw._native_lw_tables"
            ),
            "ra_sw_physics_4_wrf_root_runtime_dependencies": [],
            "ra_sw_prebuilt_npz_bound_by_candidate_commit": BASE_SHA,
            "externally_supplied_work_dir_required": True,
            "snapshot_root_equation": "WORK_DIR/authority/wrf_root",
            "work_and_authority_dir_identity_owner_type_mode_rechecked": True,
            "fully_rehashed_relocated_root_rejected": True,
            "publication": "renameat2_RENAME_NOREPLACE_same_dirfd",
            "durability_order": (
                "file_data_fsync_then_fchmod0444_then_file_fsync; "
                "child_dirs_and_temp_root_fsync; noreplace_publish; authority_dir_fsync"
            ),
            "warning_free_bottom_up_adversary_cleanup": True,
        },
        "identity_policy": validate_identity_policy_authority(),
        "freshness_limits_seconds": {
            "marker": MARKER_MAX_AGE_SECONDS,
            "resource": RESOURCE_MAX_AGE_SECONDS,
            "preemption": PREEMPT_MAX_AGE_SECONDS,
            "qa": QA_MAX_AGE_SECONDS,
            "derived_proof": DERIVED_PROOF_MAX_AGE_SECONDS,
            "runtime_acceptance": RUNTIME_ACCEPTANCE_MAX_AGE_SECONDS,
            "runtime_acceptance_lifetime": RUNTIME_ACCEPTANCE_MAX_LIFETIME_SECONDS,
        },
        "runtime_authority": {
            "code_review_schema": CODE_REVIEW_SCHEMA,
            "runtime_acceptance_schema": RUNTIME_ACCEPTANCE_SCHEMA,
            "nonce_burn_schema": NONCE_BURN_SCHEMA,
            "code_review_fixed_ref": CODE_REVIEW_ACCEPTANCE_REF,
            "manager_runtime_fixed_ref": MANAGER_RUNTIME_ACCEPTANCE_REF,
            "acceptance_must_be_fixed_ref_tracked_blob": True,
            "executing_repo_derived_from___file__": True,
            "executing_script_blob_clean_detached_exact": True,
            "nonce_consumed_before_spawn": True,
            "post_burn_acceptance_and_live_preflight_revalidated": True,
            "nonce_specific_O_EXCL_replay_rejected": True,
            "same_uid_residual": (
                "same Unix UID can mutate refs or delete authority files outside a cooperative run; "
                "no impossible kernel isolation claim is made"
            ),
        },
        "pair_closure_authority": {
            "incremental_rule": "lag_one_closed_on_both_producers",
            "last_frame_rule": "both_producers_terminal",
            "final_source_sha256_cross_binding": True,
            "source_mutation_after_pairing_fails": True,
            "active_poll_expensive_scope": "unpaired_frontier_plus_successor_only",
            "paired_source_poll": "pinned_metadata_no_sha",
            "immutable_snapshot_poll": "pinned_0444_metadata_no_sha",
            "terminal_authority_cache": "full_once_then_complete_path_device_inode_size_mtime_ctime_mode_pins",
            "stable_json_authority_fields": [
                "path", "sha256", "bytes", "device", "inode", "mtime_ns", "ctime_ns", "mode",
            ],
            "final_cached_manifest_sha_recheck": True,
            "safety_before_pairing_every_poll": True,
            "terminal_gpu_completion_schedules": {
                "d01": _terminal_stamps()["d01"],
                "d02": _terminal_stamps()["d02"],
                "d03": _terminal_stamps()["d03"],
            },
            "timestamp_normalization_forbidden": True,
        },
        "frozen_geometry_authority": {
            "path": str(FROZEN_GEO_PATH), "sha256": GRID_SHA256,
            "fields": list(STATIC_GATE_FIELDS),
            "terminal_cpu_to_frozen_geo": (
                "XLAT_XLONG_LANDMASK_rtol0_atol5e-5; "
                "HGT_finite_physical_and_rtol0_atol5e-5_outside_10_cell_boundary"
            ),
            "cpu_vs_gpu_pair_static_gate": "bit_exact_XLAT_XLONG_HGT_LANDMASK_all_55_pairs",
        },
        "gpu_device_authority": {
            "exact_producer_query_schema": True,
            "live_identity_fields": ["index", "name", "memory_total_mib"],
            "dynamic_fields": ["memory_free_mib", "utilization_percent"],
        },
        "terminal_cpu_authority": {
            "contract_path": str(TERMINAL_CONTRACT),
            "contract_sha256": terminal_authority["artifact"]["sha256"],
            "status": "TERMINAL_CPU_AUTHORITY_READY_PENDING_COMBINED_REVIEW",
            "producer_commit": TERMINAL_VERIFIER_COMMIT,
            "producer_source": CPU_QA_SOURCE,
            "producer_source_sha256": TERMINAL_VERIFIER_SHA256,
            "report_commit": TERMINAL_REPORT_COMMIT,
            "report_sha256": TERMINAL_REPORT_SHA256,
            "schema_commit": TERMINAL_SCHEMA_COMMIT,
            "schema_sha256": TERMINAL_SCHEMA_SOURCE_SHA256,
            "canonical_manifest": str(CPU_MANIFEST),
            "manifest_sha256": TERMINAL_MANIFEST_SHA256,
            "pair_index_sha256": TERMINAL_PAIR_INDEX_SHA256,
            "raw_counts": {"d01": 19, "d02": 19, "d03": 55},
            "thin_d03": 55,
            "pair_rows": 129,
            "d01_schedule": _terminal_stamps()["d01"],
            "d02_schedule": _terminal_stamps()["d02"],
            "d03_schedule": _terminal_stamps()["d03"],
            "live_cpu_requirement": False,
            "early_marker_requirement": False,
            "gpu_authority": "NOT_AUTHORIZED_PENDING_COMBINED_REVIEW_AND_RUNTIME_ACCEPTANCE",
            "recomputed_not_declarative": True,
        },
        "lock_v2": {
            "commit": LOCK_COMMIT,
            "wrapper_sha256": LOCK_WRAPPER_SHA256,
            "intent": LOCK_INTENT,
            "command_environment_order": (
                "sanitized_env_then_exact_allowlist_then_lock_v2_then_taskset_python_payload"
            ),
            "payload_preserves_injected_GPUWRF_GPU_LOCK_variables": True,
            "post_lock_env_clear_forbidden": True,
            "systemd_user_environment": dict(SYSTEMD_USER_ENVIRONMENT),
            "systemd_user_environment_position": "after_HOME_PATH_before_lock_v2",
            "sole_added_ambient_capabilities": [key for key, _value in SYSTEMD_USER_ENVIRONMENT],
            "arbitrary_host_environment_inherited": False,
        },
        "test_cpuset": "12-15",
        "test_count": len(nodes),
        "tests": nodes,
        "no_jax_or_gpuwrf_import": True,
        "no_gpu_process": True,
        "no_model_compile": True,
        "no_wrf_run": True,
        "no_queue_mutation": True,
        "run_mode_review_acceptance_present": False,
        "source_sha256": {
            "ownership.md": sha256_file(root / "ownership.md"),
            "scripts/v0234_corrected_fullbuffer_gate.py": sha256_file(Path(__file__)),
            "tests/test_v0234_corrected_fullbuffer_gate.py": sha256_file(test_path),
        },
        "all_gates_passed": True,
    }
    atomic_write_json(output, proof)
    return proof


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work-dir", type=Path, required=True)
    common.add_argument("--marker", type=Path, default=EARLY_MARKER)
    common.add_argument("--stability-seconds", type=float, default=5.0)
    audit = sub.add_parser("audit", parents=[common])
    audit.add_argument("--early-proof", type=Path)
    command = sub.add_parser("command", parents=[common])
    command.add_argument("--early-proof", type=Path, required=True)
    command.add_argument("--command-json", type=Path)
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--early-proof", type=Path, required=True)
    run.add_argument("--runtime-acceptance-path", required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--max-frame-stall-seconds", type=float, default=1800.0)
    terminal_common = argparse.ArgumentParser(add_help=False)
    terminal_common.add_argument("--work-dir", type=Path, required=True)
    terminal_common.add_argument("--terminal-contract", type=Path, default=TERMINAL_CONTRACT)
    terminal_common.add_argument("--stability-seconds", type=float, default=5.0)
    terminal_audit = sub.add_parser("terminal-audit", parents=[terminal_common])
    terminal_audit.add_argument("--terminal-proof", type=Path)
    terminal_command = sub.add_parser("terminal-command", parents=[terminal_common])
    terminal_command.add_argument("--terminal-proof", type=Path, required=True)
    terminal_command.add_argument("--command-json", type=Path)
    terminal_run = sub.add_parser("terminal-run", parents=[terminal_common])
    terminal_run.add_argument("--terminal-proof", type=Path, required=True)
    terminal_run.add_argument("--runtime-acceptance-path", required=True)
    terminal_run.add_argument("--execute", action="store_true")
    terminal_run.add_argument("--max-frame-stall-seconds", type=float, default=1800.0)
    pair = sub.add_parser("pair")
    pair.add_argument("--cpu-dir", type=Path, required=True)
    pair.add_argument("--gpu-dir", type=Path, required=True)
    pair.add_argument("--state", type=Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("--pair-state", type=Path, required=True)
    final.add_argument("--gpu-dir", type=Path, required=True)
    final.add_argument("--plot", type=Path, required=True)
    proof = sub.add_parser("proof")
    proof.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "proof":
        proof = generate_cpu_proof(args.output)
        print(json.dumps({"verdict": proof["verdict"], "output": str(args.output)}, sort_keys=True))
        return 0
    if args.mode == "pair":
        result = incremental_pair(args.cpu_dir, args.gpu_dir, args.state)
        print(json.dumps({"matched_count": result["matched_count"], "status": result["final_status"]}, sort_keys=True))
        return 0
    if args.mode == "final":
        print(json.dumps(final_verdict(CPU_MANIFEST, args.pair_state, args.plot, args.gpu_dir), sort_keys=True))
        return 0
    if args.mode == "terminal-audit":
        terminal = build_terminal_cpu_proof(
            contract_path=args.terminal_contract, work_dir=args.work_dir,
            interval_seconds=args.stability_seconds,
        )
        output = args.terminal_proof or (args.work_dir / "terminal-cpu-authority.json")
        atomic_write_json(output, terminal)
        print(json.dumps({"status": terminal["status"], "proof": str(output)}, sort_keys=True))
        return 0
    if args.mode in {"terminal-command", "terminal-run"}:
        terminal = verify_terminal_cpu_proof(
            args.terminal_proof, expected_work_dir=args.work_dir,
            interval_seconds=args.stability_seconds,
        )
        candidate = validate_candidate_worktree(RUN_REPO)
        lock = validate_lock_v2()
        command = build_exact_command(
            early_proof=terminal, work_dir=args.work_dir,
            candidate_authority=candidate, lock_authority=lock,
        )
        if args.mode == "terminal-command":
            output = args.command_json or (args.work_dir / "review-command.json")
            atomic_write_json(output, command)
            print(json.dumps(command, indent=2, sort_keys=True))
            return 0
        if not args.execute:
            raise GateError("RUN_NOT_EXPLICIT", "--execute is required")
        acceptance = validate_runtime_acceptance(
            relative_path=args.runtime_acceptance_path,
            command=command, early=terminal, now=datetime.now(timezone.utc),
        )
        terminal, pre_spawn = immediate_terminal_pre_spawn_checks(
            args.terminal_proof, acceptance["payload"]["gpu_runtime_authority"],
            expected_work_dir=args.work_dir,
        )
        candidate = validate_candidate_worktree(RUN_REPO)
        lock = validate_lock_v2()
        command = build_exact_command(
            early_proof=terminal, work_dir=args.work_dir,
            candidate_authority=candidate, lock_authority=lock,
        )
        acceptance = validate_runtime_acceptance(
            relative_path=args.runtime_acceptance_path,
            command=command, early=terminal, now=datetime.now(timezone.utc),
        )
        burn = consume_runtime_nonce(acceptance, pre_spawn, now=datetime.now(timezone.utc))
        terminal, _post_burn = immediate_terminal_pre_spawn_checks(
            args.terminal_proof, acceptance["payload"]["gpu_runtime_authority"],
            expected_work_dir=args.work_dir,
        )
        candidate = validate_candidate_worktree(RUN_REPO)
        lock = validate_lock_v2()
        command = build_exact_command(
            early_proof=terminal, work_dir=args.work_dir,
            candidate_authority=candidate, lock_authority=lock,
        )
        acceptance = validate_runtime_acceptance(
            relative_path=args.runtime_acceptance_path,
            command=command, early=terminal, now=datetime.now(timezone.utc),
        )
        verify_runtime_nonce_burn(burn, acceptance)
        last_pair_count = -1
        last_progress = time.monotonic()
        terminal_cache: dict[str, Any] = {"cpu": terminal["terminal_cpu"]}

        def terminal_monitor() -> None:
            nonlocal last_pair_count, last_progress
            poll = terminal_active_monitor_poll(
                terminal_proof=terminal, expected_work_dir=args.work_dir,
                gpu_dir=args.work_dir / "gpu-output",
                pair_state_path=args.work_dir / "incremental-pairs.json",
                terminal_cache=terminal_cache,
            )
            pairs = poll["pairs"]
            if pairs["matched_count"] > last_pair_count:
                last_pair_count = pairs["matched_count"]
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > args.max_frame_stall_seconds:
                raise GateError("FRAME_STALL", f"no new matched frame for {args.max_frame_stall_seconds}s")

        returncode = run_reviewed_command(command, monitor=terminal_monitor)
        verify_table_snapshot(
            terminal["table_authority"], expected_work_dir=args.work_dir,
        )
        for detail in terminal["input_authority"].values():
            if sha256_file(Path(detail["path"])) != detail["sha256"]:
                raise GateError("POST_RUN_INPUT_MUTATION", detail["path"])
        if returncode != 0:
            raise GateError("GPU_COMMAND_FAILED", str(returncode))
        terminal_cache["gpu"] = validate_gpu_completion_frames(
            args.work_dir / "gpu-output", terminal_mode=True,
        )
        incremental_pair(
            CPU_INPUT_DIR, args.work_dir / "gpu-output",
            args.work_dir / "incremental-pairs.json",
            terminal_cache=terminal_cache, terminal_mode=True,
        )
        result = final_verdict(
            CPU_MANIFEST, args.work_dir / "incremental-pairs.json",
            args.work_dir / "identity-numbers-first.jpg", args.work_dir / "gpu-output",
            terminal_cache=terminal_cache,
        )
        atomic_write_json(args.work_dir / "final-or-pending.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.mode in {"audit", "command", "run"}:
        raise GateError(
            "EARLY_AUTHORITY_SUPERSEDED_BY_TERMINAL_CONTRACT",
            "use terminal-audit/terminal-command/terminal-run with the exact terminal contract",
        )
    if args.mode == "audit":
        proof = build_early_ready_proof(
            marker_path=args.marker,
            work_dir=args.work_dir,
            interval_seconds=args.stability_seconds,
        )
        output = args.early_proof or (args.work_dir / "early-ready.json")
        atomic_write_json(output, proof)
        print(json.dumps({"status": proof["status"], "proof": str(output)}, sort_keys=True))
        return 0
    early = verify_early_ready_proof(args.early_proof, expected_work_dir=args.work_dir)
    candidate = validate_candidate_worktree(RUN_REPO)
    lock = validate_lock_v2()
    command = build_exact_command(
        early_proof=early, work_dir=args.work_dir,
        candidate_authority=candidate, lock_authority=lock,
    )
    if args.mode == "command":
        output = args.command_json or (args.work_dir / "review-command.json")
        atomic_write_json(output, command)
        print(json.dumps(command, indent=2, sort_keys=True))
        return 0
    if not args.execute:
        raise GateError("RUN_NOT_EXPLICIT", "--execute is required")
    acceptance = validate_runtime_acceptance(
        relative_path=args.runtime_acceptance_path,
        command=command, early=early,
        now=datetime.now(timezone.utc),
    )
    last_pair_count = -1
    last_progress = time.monotonic()
    terminal_cache: dict[str, Any] = {}

    def monitor() -> None:
        nonlocal last_pair_count, last_progress
        poll = active_monitor_poll(
            early_proof=early, expected_work_dir=args.work_dir,
            cpu_manifest_path=CPU_MANIFEST,
            cpu_dir=CPU_INPUT_DIR, gpu_dir=args.work_dir / "gpu-output",
            pair_state_path=args.work_dir / "incremental-pairs.json",
            terminal_cache=terminal_cache,
        )
        pairs = poll["pairs"]
        if pairs["matched_count"] > last_pair_count:
            last_pair_count = pairs["matched_count"]
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress > args.max_frame_stall_seconds:
            raise GateError("FRAME_STALL", f"no new matched frame for {args.max_frame_stall_seconds}s")

    early, pre_spawn = immediate_pre_spawn_checks(
        args.early_proof, expected_work_dir=args.work_dir,
    )
    candidate = validate_candidate_worktree(RUN_REPO)
    lock = validate_lock_v2()
    command = build_exact_command(
        early_proof=early, work_dir=args.work_dir,
        candidate_authority=candidate, lock_authority=lock,
    )
    acceptance = validate_runtime_acceptance(
        relative_path=args.runtime_acceptance_path,
        command=command, early=early,
        now=datetime.now(timezone.utc),
    )
    burn = consume_runtime_nonce(acceptance, pre_spawn, now=datetime.now(timezone.utc))
    # The burn is irreversible before the final live check.  Expiry and every
    # live authority are then revalidated immediately before Popen.
    early, _post_burn_pre_spawn = immediate_pre_spawn_checks(
        args.early_proof, expected_work_dir=args.work_dir,
    )
    candidate = validate_candidate_worktree(RUN_REPO)
    lock = validate_lock_v2()
    command = build_exact_command(
        early_proof=early, work_dir=args.work_dir,
        candidate_authority=candidate, lock_authority=lock,
    )
    acceptance = validate_runtime_acceptance(
        relative_path=args.runtime_acceptance_path,
        command=command, early=early,
        now=datetime.now(timezone.utc),
    )
    verify_runtime_nonce_burn(burn, acceptance)
    returncode = run_reviewed_command(command, monitor=monitor)
    verify_table_snapshot(early["table_authority"], expected_work_dir=args.work_dir)
    for detail in early["input_authority"].values():
        if sha256_file(Path(detail["path"])) != detail["sha256"]:
            raise GateError("POST_RUN_INPUT_MUTATION", detail["path"])
    if returncode != 0:
        raise GateError("GPU_COMMAND_FAILED", str(returncode))
    if "cpu" not in terminal_cache:
        terminal_cache["cpu"] = validate_cpu_terminal_manifest(CPU_MANIFEST)
    else:
        verify_cached_terminal_metadata(terminal_cache["cpu"], "CPU_TERMINAL_CACHE_CHANGED")
    if "gpu" not in terminal_cache:
        terminal_cache["gpu"] = validate_gpu_completion_frames(args.work_dir / "gpu-output")
    else:
        verify_cached_terminal_metadata(terminal_cache["gpu"], "GPU_TERMINAL_CACHE_CHANGED")
    incremental_pair(
        CPU_INPUT_DIR,
        args.work_dir / "gpu-output",
        args.work_dir / "incremental-pairs.json",
        terminal_cache=terminal_cache,
    )
    result = final_verdict(
        CPU_MANIFEST,
        args.work_dir / "incremental-pairs.json",
        args.work_dir / "identity-numbers-first.jpg",
        args.work_dir / "gpu-output",
        terminal_cache=terminal_cache,
    )
    atomic_write_json(args.work_dir / "final-or-pending.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(75) from exc
