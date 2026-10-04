#!/usr/bin/env python3
"""Fail-closed v0234 SP2 long-trajectory validation support.

This tool owns no model implementation.  It inventories the retained Canary
authority, launches exactly one already-locked native live-nested trajectory,
and scores unmasked all-cell T2/U10/V10/wind-speed metrics against CPU WRF.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset, chartostring


AUTHORIZED_CPUS = frozenset({13, 14, 15, 29, 30, 31})
EXPECTED_RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
EXPECTED_INIT = datetime(2026, 5, 1, 18, tzinfo=timezone.utc)
EXPECTED_GRID_YX = (66, 159)
REQUIRED_TRUTH_FIELDS = ("T2", "U10", "V10")
FROZEN_LIMITS = {"T2": 1.5, "U10": 1.5, "V10": 1.5}
DERIVED_WSPD_CEILING = math.sqrt(1.5**2 + 1.5**2)
SUPPORTED_FORECAST_HOURS = (24, 72)
NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
LEASE_RE = re.compile(r"^[0-9a-f]{64}$")
WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")
ACCEPTED_CACHE_SEED = {
    "path": "<DATA_ROOT>/wrf_gpu2/v0234_boundary_default_confirm_e9e820d1b9297613/cache",
    "file_count": 7055,
    "total_bytes": 1667499920,
    "inventory_sha256": "afbe0ed86598e04a39a47127c3ae6d727ec86baf0b47e424eb5d3325100e81f3",
}
ACCEPTED_POST_DELEGATION_FILES = {
    "src/gpuwrf/integration/nested_pipeline.py": "9a27b7f71aa575569fa1b2c03638cbb955936e8664f86fec7154ddd39f58dc89",
    "src/gpuwrf/runtime/domain_tree.py": "47eb3974fcf98db1be436a7af5c625e7bea7129f6129a6a16df31dfe665eae72",
    "tests/test_aot_executable.py": "3eac29f4071ca74eceec240e9db257b7bcf9b3ee35ddd36e4e2ac55552a12aa7",
    "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py": "24aebc71a257b87232c7c1a3dd34d11bae7606be69df3782e020182cd3b1663e",
    "tests/test_v0234_prepared_runtime_ab_tooling.py": "116e77b8383de9cc9610856fd9f1b5afff9b82aa0f992b7499a393bc1b34e9d9",
    "tests/test_v024_nested_runtime_reuse.py": "71d5850d13ea625f0b4abc6287b0ceb3621b8715f3d9d9f69cd07627da2ca38d",
}
ACCEPTED_POST_DELEGATION_COMMITS = {
    "prepared_runtime_reuse": "215d302dfc1493601cf1d21604b3ae10cf396298",
    "prepared_runtime_ab": "54e8051a31fa65d08d27810dd00a85a33a3a8a7c",
    "prepared_runtime_opus_accept": "92bddcafc472a8d34c907112df7ef351712cdbca",
    "frozen_boundary_default_fix": "8192d1b6fda7f247f4dc34ee056b39e0efb30ad9",
}
LOCK_AUTHORITY = {
    "worktree": "<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2",
    "commit": "8152309aff1e85e1052d44d549a5a5409e710bdd",
    "wrapper": "<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2/scripts/with_gpu_lock.sh",
    "wrapper_sha256": "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a",
    "implementation": "<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2/scripts/gpu_lock_v2.py",
    "implementation_sha256": "ec911f565ee9d2c58dee81d8d1e17411ed677be500a57e73ded5f793e2c4187d",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _cpu_policy() -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    affinity_set = set(affinity)
    required_env = {
        "OMP_NUM_THREADS": "1",
        "OMP_THREAD_LIMIT": "1",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    observed_env = {key: os.environ.get(key) for key in required_env}
    problems = []
    if not affinity_set or not affinity_set.issubset(AUTHORIZED_CPUS):
        problems.append(
            f"affinity {affinity} is not a nonempty subset of {sorted(AUTHORIZED_CPUS)}"
        )
    for key, expected in required_env.items():
        if observed_env[key] != expected:
            problems.append(f"{key}={observed_env[key]!r}, expected {expected!r}")
    return {
        "authorized_cpus": sorted(AUTHORIZED_CPUS),
        "observed_affinity": affinity,
        "required_environment": required_env,
        "observed_environment": observed_env,
        "ok": not problems,
        "problems": problems,
    }


def _read_times(dataset: Dataset) -> list[str]:
    if "Times" not in dataset.variables:
        return []
    values = chartostring(dataset.variables["Times"][:])
    return [str(value) for value in np.asarray(values).reshape(-1)]


def _read_var(dataset: Dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    raw = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
    return np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)


def _valid_time(path: Path) -> datetime:
    match = WRFOUT_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a canonical wrfout name: {path}")
    return datetime.strptime(match.group(2), "%Y-%m-%d_%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def _namelist_checks(path: Path) -> dict[str, Any]:
    text = path.read_text()
    patterns = {
        "run_hours_72": r"\brun_hours\s*=\s*72\s*,",
        "max_dom_2": r"\bmax_dom\s*=\s*2\s*,",
        "feedback_zero": r"\bfeedback\s*=\s*0\s*,",
        "history_hourly": r"\bhistory_interval\s*=\s*60\s*,\s*60\s*,",
        "noahmp_both_domains": r"\bsf_surface_physics\s*=\s*4\s*,\s*4\s*,",
        "mynn_pbl_both_domains": r"\bbl_pbl_physics\s*=\s*5\s*,\s*5\s*,",
        "mynn_surface_both_domains": r"\bsf_sfclay_physics\s*=\s*5\s*,\s*5\s*,",
        "rrtmg_lw_both_domains": r"\bra_lw_physics\s*=\s*4\s*,\s*4\s*,",
        "rrtmg_sw_both_domains": r"\bra_sw_physics\s*=\s*4\s*,\s*4\s*,",
        "radiation_cadence_30min_both_domains": r"\bradt\s*=\s*30\s*,\s*30\s*,",
        "gwd_enabled": r"\bgwd_opt\s*=\s*1\s*,",
    }
    checks = {name: re.search(pattern, text) is not None for name, pattern in patterns.items()}
    return {"path": str(path), "sha256": sha256_file(path), "checks": checks, "ok": all(checks.values())}


def _truth_inventory(truth_dir: Path) -> dict[str, Any]:
    files = sorted(truth_dir.glob("wrfout_d02_*"), key=_valid_time)
    expected_times = [EXPECTED_INIT + timedelta(hours=hour) for hour in range(73)]
    observed_times = [_valid_time(path) for path in files]
    problems: list[str] = []
    if observed_times != expected_times:
        problems.append(
            f"expected exact d02 h0..h72 inventory (73 frames), got {len(files)} frames"
        )

    field_finite_counts = {name: 0 for name in REQUIRED_TRUTH_FIELDS}
    field_total_counts = {name: 0 for name in REQUIRED_TRUTH_FIELDS}
    shapes: dict[str, list[int]] = {}
    internal_time_mismatches: list[str] = []
    for path in files:
        with Dataset(path, "r") as dataset:
            times = _read_times(dataset)
            expected_label = _valid_time(path).strftime("%Y-%m-%d_%H:%M:%S")
            if times != [expected_label]:
                internal_time_mismatches.append(
                    f"{path.name}: Times={times!r}, expected {[expected_label]!r}"
                )
            for name in REQUIRED_TRUTH_FIELDS:
                if name not in dataset.variables:
                    problems.append(f"{path.name}: missing {name}")
                    continue
                array = _read_var(dataset, name)
                shapes.setdefault(name, list(array.shape))
                if array.shape != EXPECTED_GRID_YX:
                    problems.append(
                        f"{path.name}: {name} shape={array.shape}, expected {EXPECTED_GRID_YX}"
                    )
                field_finite_counts[name] += int(np.isfinite(array).sum())
                field_total_counts[name] += int(array.size)
    if internal_time_mismatches:
        problems.extend(internal_time_mismatches[:8])
    finite_fractions = {
        name: (
            field_finite_counts[name] / field_total_counts[name]
            if field_total_counts[name]
            else 0.0
        )
        for name in REQUIRED_TRUTH_FIELDS
    }
    if any(value != 1.0 for value in finite_fractions.values()):
        problems.append(f"nonfinite retained truth: {finite_fractions}")

    endpoint_hashes = {}
    by_time = dict(zip(observed_times, files, strict=True)) if len(observed_times) == len(set(observed_times)) else {}
    for hour in (0, 24, 72):
        valid = EXPECTED_INIT + timedelta(hours=hour)
        path = by_time.get(valid)
        if path is not None:
            endpoint_hashes[f"h{hour}"] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "path": str(truth_dir),
        "frame_count": len(files),
        "first_valid_utc": observed_times[0].isoformat() if observed_times else None,
        "last_valid_utc": observed_times[-1].isoformat() if observed_times else None,
        "exact_h0_through_h72": observed_times == expected_times,
        "required_field_shapes_yx": shapes,
        "finite_fractions": finite_fractions,
        "endpoint_hashes": endpoint_hashes,
        "problems": problems,
        "ok": not problems,
    }


def _input_inventory(input_dir: Path) -> dict[str, Any]:
    required = ("namelist.input", "wrfinput_d01", "wrfinput_d02", "wrfbdy_d01")
    problems = [f"missing {name}" for name in required if not (input_dir / name).is_file()]
    files: dict[str, Any] = {}
    for name in required:
        path = input_dir / name
        if path.is_file():
            files[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    domain_times = {}
    for domain in ("d01", "d02"):
        path = input_dir / f"wrfinput_{domain}"
        if not path.is_file():
            continue
        with Dataset(path, "r") as dataset:
            domain_times[domain] = _read_times(dataset)
            for field in REQUIRED_TRUTH_FIELDS:
                if field not in dataset.variables:
                    problems.append(f"wrfinput_{domain}: missing {field}")
    boundary_times: list[str] = []
    boundary = input_dir / "wrfbdy_d01"
    if boundary.is_file():
        with Dataset(boundary, "r") as dataset:
            boundary_times = _read_times(dataset)
    # WRF wrfbdy stores one base+tendency record per forcing segment, not an
    # extra endpoint record.  For this 72 h / 6 h fixture the 12 records start
    # at h0..h66; the h66 _BT tendency advances the final [66,72) segment.  The
    # production loader materializes that endpoint explicitly (d02_replay.py,
    # ``records = ... + [(ntimes - 1, interval_s)]``).
    expected_boundary = [
        (EXPECTED_INIT + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
        for hour in range(0, 72, 6)
    ]
    if boundary_times != expected_boundary:
        problems.append(
            f"wrfbdy_d01 does not provide exact 6-hour segment starts h0..h66: {boundary_times}"
        )
    expected_init_label = EXPECTED_INIT.strftime("%Y-%m-%d_%H:%M:%S")
    for domain, times in domain_times.items():
        if times != [expected_init_label]:
            problems.append(f"wrfinput_{domain}: Times={times}, expected {[expected_init_label]}")
    return {
        "path": str(input_dir),
        "files": files,
        "wrfinput_times": domain_times,
        "wrfbdy_d01_times": boundary_times,
        "wrfbdy_exact_segment_starts_h0_through_h66": boundary_times == expected_boundary,
        "wrfbdy_final_tendency_segment_end_h": 72 if boundary_times == expected_boundary else None,
        "wrfbdy_endpoint_semantics": (
            "each record k supplies base at k*interval plus _BT tendency over "
            "[k*interval,(k+1)*interval); production loader synthesizes h72 from h66"
        ),
        "problems": problems,
        "ok": not problems,
    }


def _fix_chain(repo: Path) -> dict[str, Any]:
    manifest_path = repo / ".agent/sprints/2026-07-20-v0234-gpt-mf-seam-closure/PROOF_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["implementation"]["production_and_test_sha256"]
    observed = {relative: sha256_file(repo / relative) for relative in expected}
    hash_match = {relative: observed[relative] == digest for relative, digest in expected.items()}
    delegated = "4d1a32f7cacd60194998ff743d88d5ad829efa7b"
    production_fix = manifest["implementation"]["production_fix_commit"]
    head = _git("rev-parse", "HEAD", cwd=repo)
    ancestry = {}
    required_commits = {
        "delegated_terminal": delegated,
        "production_fix": production_fix,
        **ACCEPTED_POST_DELEGATION_COMMITS,
    }
    for label, commit in required_commits.items():
        rc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head], cwd=repo, check=False
        ).returncode
        ancestry[label] = rc == 0
    model_test_diff = _git(
        "diff",
        "--name-only",
        delegated,
        "--",
        "src/gpuwrf",
        "tests",
        cwd=repo,
    ).splitlines()
    owned_validation_test = "tests/test_v0234_gpt_drift_validation.py"
    accepted_observed = {
        relative: sha256_file(repo / relative)
        for relative in ACCEPTED_POST_DELEGATION_FILES
    }
    accepted_hash_match = {
        relative: accepted_observed[relative] == digest
        for relative, digest in ACCEPTED_POST_DELEGATION_FILES.items()
    }
    allowed_diff = set(ACCEPTED_POST_DELEGATION_FILES) | {owned_validation_test}
    disallowed_model_test_diff = [
        path for path in model_test_diff if path not in allowed_diff
    ]
    no_model_test_diff = not disallowed_model_test_diff
    dirty_model_test = _git("status", "--short", "--", "src/gpuwrf", "tests", cwd=repo).splitlines()
    return {
        "head": head,
        "delegated_terminal": delegated,
        "predecessor_manifest": str(manifest_path),
        "predecessor_manifest_sha256": sha256_file(manifest_path),
        "expected_production_and_test_sha256": expected,
        "observed_production_and_test_sha256": observed,
        "hash_match": hash_match,
        "ancestry": ancestry,
        "accepted_post_delegation_expected_sha256": ACCEPTED_POST_DELEGATION_FILES,
        "accepted_post_delegation_observed_sha256": accepted_observed,
        "accepted_post_delegation_hash_match": accepted_hash_match,
        "model_or_test_diff_since_delegated_terminal": model_test_diff,
        "accepted_model_test_allowlist": sorted(allowed_diff),
        "disallowed_model_or_test_diff": disallowed_model_test_diff,
        "no_disallowed_model_or_test_diff_since_delegated_terminal": no_model_test_diff,
        "dirty_model_or_test_tree": dirty_model_test,
        "ok": (
            all(hash_match.values())
            and all(ancestry.values())
            and all(accepted_hash_match.values())
            and no_model_test_diff
            and not dirty_model_test
        ),
    }


def _lock_authority() -> dict[str, Any]:
    root = Path(LOCK_AUTHORITY["worktree"])
    wrapper = Path(LOCK_AUTHORITY["wrapper"])
    implementation = Path(LOCK_AUTHORITY["implementation"])
    observed = {
        **LOCK_AUTHORITY,
        "observed_commit": _git("rev-parse", "HEAD", cwd=root),
        "observed_wrapper_sha256": sha256_file(wrapper),
        "observed_implementation_sha256": sha256_file(implementation),
    }
    observed["ok"] = (
        observed["observed_commit"] == LOCK_AUTHORITY["commit"]
        and observed["observed_wrapper_sha256"] == LOCK_AUTHORITY["wrapper_sha256"]
        and observed["observed_implementation_sha256"]
        == LOCK_AUTHORITY["implementation_sha256"]
    )
    return observed


def _wrf_root_inventory(wrf_root: Path) -> dict[str, Any]:
    required = (
        "run/MPTABLE.TBL",
        "run/SOILPARM.TBL",
        "run/GENPARM.TBL",
        "run/RRTMG_LW_DATA",
        "run/RRTMG_SW_DATA",
        "run/CAMtr_volume_mixing_ratio",
    )
    files = {}
    problems = []
    for relative in required:
        path = wrf_root / relative
        if not path.is_file():
            problems.append(f"missing runtime authority {path}")
            continue
        files[relative] = {
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "path": str(wrf_root),
        "files": files,
        "problems": problems,
        "ok": not problems,
    }


def _cache_inventory(root: Path) -> dict[str, Any]:
    problems: list[str] = []
    if not root.is_dir():
        return {
            "path": str(root),
            "problems": [f"cache seed is not a directory: {root}"],
            "ok": False,
        }
    inventory_digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest = sha256_file(path)
        relative = "./" + path.relative_to(root).as_posix()
        inventory_digest.update(f"{digest}  {relative}\n".encode())
        file_count += 1
        total_bytes += path.stat().st_size
    observed = {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "inventory_sha256": inventory_digest.hexdigest(),
    }
    for key in ("file_count", "total_bytes", "inventory_sha256"):
        if observed[key] != ACCEPTED_CACHE_SEED[key]:
            problems.append(
                f"cache seed {key}={observed[key]!r}, expected {ACCEPTED_CACHE_SEED[key]!r}"
            )
    return {
        "path": str(root),
        "expected": ACCEPTED_CACHE_SEED,
        "observed": observed,
        "problems": problems,
        "ok": not problems,
    }


def _validate_cache_destination(cache_dir: Path, nonce: str, *, require_absent: bool) -> dict[str, Any]:
    problems = []
    if not cache_dir.is_absolute():
        problems.append("cache destination must be absolute")
    if cache_dir.parent != Path("<DATA_ROOT>/wrf_gpu2"):
        problems.append("cache destination parent must be exactly <DATA_ROOT>/wrf_gpu2")
    if nonce[:16] not in cache_dir.name:
        problems.append("cache destination name does not carry the nonce prefix")
    if require_absent and cache_dir.exists():
        problems.append("cache destination already exists")
    free_bytes = shutil.disk_usage("<DATA_ROOT>").free
    if free_bytes < 10 * 1024**3:
        problems.append(f"<DATA_ROOT> has only {free_bytes} free bytes; require at least 10 GiB")
    return {
        "path": str(cache_dir),
        "nonce_prefix_in_path": nonce[:16],
        "exists": cache_dir.exists(),
        "required_absent": require_absent,
        "filesystem_free_bytes": free_bytes,
        "minimum_free_bytes": 10 * 1024**3,
        "problems": problems,
        "ok": not problems,
    }


def _production_environment(repo: Path, wrf_root: Path, cache_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    # Start from a clean model/runtime configuration so an inherited experimental
    # knob cannot silently alter physics or fragment the AOT key.  Preserve only
    # the canonical lock-v2 descriptors, which the child must independently
    # attest and which the AOT key explicitly treats as inert bookkeeping.
    for key in tuple(env):
        if key.startswith("GPUWRF_") and not key.startswith("GPUWRF_GPU_LOCK_"):
            env.pop(key)
    env.pop("JAX_PLATFORMS", None)
    env.update(
        {
            "PYTHONPATH": str(repo / "src"),
            "PYTHONUNBUFFERED": "1",
            "JAX_ENABLE_X64": "true",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": (
                "--xla_gpu_per_fusion_autotune_cache_dir=<USER_HOME>/.cache/gpuwrf/autotune "
                "--xla_gpu_force_compilation_parallelism=8"
            ),
            "GPUWRF_WRF_ROOT": str(wrf_root),
            "GPUWRF_JAX_CACHE_DIR": str(cache_dir),
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            "GPUWRF_NESTED_AOT": "1",
            "GPUWRF_NESTED_FUSE": "0",
            "GPUWRF_NESTED_DEFUSE_COMPILE": "0",
            "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
            "GPUWRF_NESTED_SYNC_MODE": "root",
            "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
            "GPUWRF_PREPARED_RUNTIME_REUSE": "1",
            "GPUWRF_FINITE_CHECK": "1",
        }
    )
    return env


def _cli_dry_run(
    repo: Path,
    input_dir: Path,
    namespace: Path,
    wrf_root: Path,
    cache_dir: Path,
    hours: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "gpuwrf.cli",
        "run",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(namespace / "gpu_output" / f"canary_l2_{EXPECTED_RUN_ID}"),
        "--scratch-dir",
        str(namespace / "scratch"),
        "--proof-dir",
        str(namespace / "pipeline_proofs"),
        "--max-dom",
        "2",
        "--hours",
        str(hours),
        "--dry-run",
    ]
    env = _production_environment(repo, wrf_root, cache_dir)
    # Static CPU inspection must not materialize or mutate the fresh GPU-arm
    # cache destination merely by initializing JAX's cache machinery.
    env.pop("GPUWRF_JAX_CACHE_DIR", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        plan = json.loads(completed.stdout)
    except json.JSONDecodeError:
        plan = {}
    expected = {
        "dry_run": True,
        "run_type": "nested_live",
        "init_mode": "standalone_native_init_nested",
        "effective_max_dom": 2,
        "effective_hours": hours,
        "feedback": False,
    }
    plan_checks = {key: plan.get(key) == value for key, value in expected.items()}
    problems = []
    if completed.returncode != 0:
        problems.append(f"CLI dry-run rc={completed.returncode}")
    if not all(plan_checks.values()):
        problems.append(f"CLI dry-run effective plan mismatch: {plan_checks}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "effective_plan": plan,
        "expected_plan_subset": expected,
        "plan_checks": plan_checks,
        "problems": problems,
        "ok": not problems,
    }


def _validate_namespace(namespace: Path, nonce: str, *, require_absent: bool) -> dict[str, Any]:
    problems = []
    if NONCE_RE.fullmatch(nonce) is None:
        problems.append("nonce must be exactly 64 lowercase hexadecimal characters")
    if not namespace.is_absolute():
        problems.append("namespace must be absolute")
    if namespace.parent != Path("<DATA_ROOT>/wrf_gpu2"):
        problems.append("namespace parent must be exactly <DATA_ROOT>/wrf_gpu2")
    if nonce[:16] not in namespace.name:
        problems.append("namespace name does not carry the nonce prefix")
    if require_absent and namespace.exists():
        problems.append("namespace already exists")
    return {
        "path": str(namespace),
        "nonce": nonce,
        "nonce_prefix_in_path": nonce[:16],
        "exists": namespace.exists(),
        "required_absent": require_absent,
        "problems": problems,
        "ok": not problems,
    }


def command_preflight(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    namespace = args.namespace
    checks = {
        "cpu_policy": _cpu_policy(),
        "namespace": _validate_namespace(namespace, args.nonce, require_absent=True),
        "cache_destination": _validate_cache_destination(
            args.cache_dir, args.nonce, require_absent=True
        ),
        "accepted_cache_seed": _cache_inventory(Path(ACCEPTED_CACHE_SEED["path"])),
        "fix_chain": _fix_chain(repo),
        "lock_v2_authority": _lock_authority(),
        "wrf_runtime_authority": _wrf_root_inventory(args.wrf_root),
        "namelist": _namelist_checks(args.input_dir / "namelist.input"),
        "input_authority": _input_inventory(args.input_dir),
        "cpu_wrf_truth": _truth_inventory(args.truth_dir),
        "cli_dry_run": _cli_dry_run(
            repo,
            args.input_dir,
            namespace,
            args.wrf_root,
            args.cache_dir,
            args.hours,
        ),
    }
    problems = [name for name, check in checks.items() if not check.get("ok", False)]
    payload = {
        "schema": "wrfgpu2-v0234-sp2-drift-preflight-v1",
        "generated_utc": utc_now(),
        "candidate_repo": str(repo),
        "run_id": EXPECTED_RUN_ID,
        "init_utc": EXPECTED_INIT.isoformat(),
        "forecast_hours": args.hours,
        "namespace": str(namespace),
        "nonce": args.nonce,
        "cache_destination": str(args.cache_dir),
        "runner": (
            f"python -m gpuwrf.cli run --max-dom 2 --hours {args.hours} "
            "(native live-nested; full hourly output)"
        ),
        "radiation_cadence_minutes": 30,
        "output_cadence_minutes": 60,
        "candidate_uses_cpu_wrf_history": False,
        "checks": checks,
        "problems": problems,
        "verdict": "STATIC_PREFLIGHT_PASS" if not problems else "STATIC_PREFLIGHT_FAIL",
    }
    write_json(args.out, payload)
    print(json.dumps({"verdict": payload["verdict"], "problems": problems, "out": str(args.out)}))
    return 0 if not problems else 1


def _lock_attestation() -> dict[str, Any]:
    held = os.environ.get("GPUWRF_GPU_LOCK_HELD")
    token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN", "")
    label = os.environ.get("GPUWRF_GPU_LOCK_LABEL", "")
    raw_fd = os.environ.get("GPUWRF_GPU_LOCK_FD", "")
    holder_path = Path(os.environ.get("GPUWRF_GPU_LOCK_HOLDER_FILE", ""))
    problems = []
    if held != "1":
        problems.append("GPUWRF_GPU_LOCK_HELD is not 1")
    if LEASE_RE.fullmatch(token) is None:
        problems.append("GPUWRF_GPU_LOCK_TOKEN is not a canonical lease id")
    if not holder_path.is_file():
        problems.append(f"holder sidecar missing: {holder_path}")
        sidecar = {}
    else:
        sidecar = json.loads(holder_path.read_text())
        if sidecar.get("lease_id") != token:
            problems.append("holder sidecar lease_id does not match payload token")
        if sidecar.get("intent") != "production-preemptible":
            problems.append("holder sidecar intent is not production-preemptible")
        if sidecar.get("label") != label:
            problems.append("holder sidecar label does not match payload label")
    try:
        lock_fd = int(raw_fd)
        fd_stat = os.fstat(lock_fd)
        fd_target = os.readlink(f"/proc/self/fd/{lock_fd}")
    except (TypeError, ValueError, OSError) as exc:
        lock_fd = -1
        fd_stat = None
        fd_target = None
        problems.append(f"inherited lock fd is unavailable: {type(exc).__name__}: {exc}")
    if fd_stat is not None and sidecar:
        sidecar_lock = sidecar.get("lock", {})
        if int(sidecar_lock.get("device", -1)) != int(fd_stat.st_dev):
            problems.append("inherited lock fd device does not match holder sidecar")
        if int(sidecar_lock.get("inode", -1)) != int(fd_stat.st_ino):
            problems.append("inherited lock fd inode does not match holder sidecar")
    return {
        "held": held,
        "lease_id": token,
        "label": label,
        "holder_path": str(holder_path),
        "holder_sidecar": sidecar,
        "inherited_lock_fd": lock_fd,
        "inherited_lock_fd_target": fd_target,
        "inherited_lock_fd_device": int(fd_stat.st_dev) if fd_stat is not None else None,
        "inherited_lock_fd_inode": int(fd_stat.st_ino) if fd_stat is not None else None,
        "problems": problems,
        "ok": not problems,
    }


def command_arm(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    cpu = _cpu_policy()
    namespace = _validate_namespace(args.namespace, args.nonce, require_absent=True)
    cache_destination = _validate_cache_destination(
        args.cache_dir, args.nonce, require_absent=True
    )
    cache_seed = _cache_inventory(Path(ACCEPTED_CACHE_SEED["path"]))
    fix_chain = _fix_chain(repo)
    lock = _lock_attestation()
    failed = [
        name
        for name, check in (
            ("cpu_policy", cpu),
            ("namespace", namespace),
            ("cache_destination", cache_destination),
            ("accepted_cache_seed", cache_seed),
            ("fix_chain", fix_chain),
            ("lock", lock),
        )
        if not check["ok"]
    ]
    if failed:
        raise SystemExit(f"fail-closed arm refusal: {failed}")

    # Reflink-copy the immutable accepted current-key cache into a fresh sibling
    # namespace.  The seed itself is never exposed to JAX writes.
    os.mkdir(args.cache_dir, mode=0o755)
    subprocess.run(
        [
            "cp",
            "--reflink=auto",
            "-a",
            f"{ACCEPTED_CACHE_SEED['path']}/.",
            str(args.cache_dir),
        ],
        check=True,
    )
    copied_cache = _cache_inventory(args.cache_dir)
    if not copied_cache["ok"]:
        raise SystemExit("fail-closed arm refusal: copied cache inventory mismatch")

    # Atomic fresh-namespace creation happens only after the canonical lease has
    # been attested.  FileExistsError is terminal; this tool never reuses a root.
    os.mkdir(args.namespace, mode=0o755)
    output_dir = args.namespace / "gpu_output" / f"canary_l2_{EXPECTED_RUN_ID}"
    proof_dir = args.namespace / "pipeline_proofs"
    scratch_dir = args.namespace / "scratch"
    output_dir.mkdir(parents=True)
    proof_dir.mkdir()
    scratch_dir.mkdir()

    command = [
        sys.executable,
        "-m",
        "gpuwrf.cli",
        "run",
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(output_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--proof-dir",
        str(proof_dir),
        "--max-dom",
        "2",
        "--hours",
        str(args.hours),
    ]
    env = _production_environment(repo, args.wrf_root, args.cache_dir)
    if env.get("CUDA_VISIBLE_DEVICES") == "":
        raise SystemExit("fail-closed arm refusal: CUDA_VISIBLE_DEVICES disables the GPU")

    descriptor = {
        "schema": "wrfgpu2-v0234-sp2-single-gpu-arm-v2",
        "created_utc": utc_now(),
        "namespace": str(args.namespace),
        "nonce": args.nonce,
        "forecast_hours": args.hours,
        "output_enabled": True,
        "output_cadence_minutes": 60,
        "radiation_cadence_minutes": 30,
        "candidate_repo": str(repo),
        "candidate_head": _git("rev-parse", "HEAD", cwd=repo),
        "input_dir": str(args.input_dir),
        "cpu_truth_dir": str(args.truth_dir),
        "output_dir": str(output_dir),
        "proof_dir": str(proof_dir),
        "scratch_dir": str(scratch_dir),
        "cache_seed": cache_seed,
        "cache_destination": copied_cache,
        "command": command,
        "working_directory": str(repo),
        "environment": {
            key: env.get(key)
            for key in (
                "PYTHONPATH",
                "JAX_ENABLE_X64",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_FLAGS",
                "GPUWRF_WRF_ROOT",
                "GPUWRF_JAX_CACHE_DIR",
                "JAX_COMPILATION_CACHE_DIR",
                "GPUWRF_NESTED_AOT",
                "GPUWRF_NESTED_FUSE",
                "GPUWRF_NESTED_DEFUSE_COMPILE",
                "GPUWRF_NESTED_PARALLEL_COMPILE",
                "GPUWRF_NESTED_SYNC_MODE",
                "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE",
                "GPUWRF_NEST_OUTPUT_PIPELINE",
                "GPUWRF_NESTED_ASYNC_OUTPUT",
                "GPUWRF_PREPARED_RUNTIME_REUSE",
                "GPUWRF_FINITE_CHECK",
                "GPUWRF_TRAINING_OUTPUT_SUBSET",
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OMP_DYNAMIC",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "cpu_policy": cpu,
        "fix_chain": fix_chain,
        "lock_attestation": lock,
        "lock_authority": _lock_authority(),
        "one_scientific_arm_ordinal": 1,
    }
    write_json(args.namespace / "ARM_DESCRIPTOR.json", descriptor)

    combined_log = args.namespace / "GPU_TRAJECTORY.log"
    started = datetime.now(timezone.utc)
    with combined_log.open("w", encoding="utf-8", buffering=1) as log:
        # The production CLI independently attests the canonical lease before it
        # touches CUDA.  Preserve the exact inherited lock-v2 descriptor through
        # this validation supervisor and the CLI's allocator re-exec.
        inherited_lock_fd = int(lock["inherited_lock_fd"])
        os.set_inheritable(inherited_lock_fd, True)
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            pass_fds=(inherited_lock_fd,),
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        returncode = process.wait()
    result = {
        "schema": "wrfgpu2-v0234-sp2-single-gpu-arm-result-v2",
        "started_utc": started.isoformat(),
        "finished_utc": utc_now(),
        "returncode": returncode,
        "forecast_hours": args.hours,
        "command": command,
        "combined_log": str(combined_log),
        "combined_log_sha256": sha256_file(combined_log),
        "pipeline_proof": str(proof_dir / "nested_pipeline_run.json"),
        "output_dir": str(output_dir),
        "lock_lease_id": lock["lease_id"],
        "lock_intent": lock["holder_sidecar"].get("intent"),
        "lock_arm_count": 1,
        "model_trajectory_started": (proof_dir / "nested_pipeline_run.json").is_file()
        or any(output_dir.glob("wrfout_*")),
        "science_trajectory_count": int(
            (proof_dir / "nested_pipeline_run.json").is_file()
            or any(output_dir.glob("wrfout_*"))
        ),
        "retry_count": 0,
    }
    write_json(args.namespace / "ARM_RESULT.json", result)
    print(json.dumps({"arm_returncode": returncode, "result": str(args.namespace / "ARM_RESULT.json")}))
    return returncode


def command_freedom(args: argparse.Namespace) -> int:
    descriptor = json.loads(args.arm_descriptor.read_text())
    lock = descriptor["lock_attestation"]
    sidecar_before = Path(lock["holder_path"]).exists()
    lock_path = Path(lock["holder_sidecar"]["lock"]["path"])
    scope = str(lock["holder_sidecar"]["payload"]["unit"])
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    sidecar_after = Path(lock["holder_path"]).exists()
    systemctl = subprocess.run(
        ["systemctl", "--user", "show", scope, "--property=LoadState,ActiveState,SubState,ControlGroup"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    properties = {}
    for line in systemctl.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    active = properties.get("ActiveState") == "active"
    ok = acquired and not sidecar_before and not sidecar_after and not active
    payload = {
        "schema": "wrfgpu2-v0234-lock-v2-freedom-proof-v1",
        "generated_utc": utc_now(),
        "method": (
            "CPU-only nonblocking fcntl acquisition of canonical lock inode plus "
            "holder-sidecar absence and exact systemd scope state; no CUDA query"
        ),
        "failed_arm_descriptor": str(args.arm_descriptor),
        "lease_id": lock["lease_id"],
        "intent": lock["holder_sidecar"].get("intent"),
        "lock_path": str(lock_path),
        "lock_nonblocking_acquired_then_released": acquired,
        "holder_sidecar_before": sidecar_before,
        "holder_sidecar_after": sidecar_after,
        "scope": scope,
        "systemctl_returncode": systemctl.returncode,
        "systemctl_properties": properties,
        "systemctl_stderr": systemctl.stderr,
        "scope_active": active,
        "gpu_or_cuda_queried": False,
        "ok": ok,
        "verdict": "GPU_LOCK_V2_FREED" if ok else "GPU_LOCK_V2_FREEDOM_NOT_PROVEN",
    }
    write_json(args.out, payload)
    print(json.dumps({"verdict": payload["verdict"], "out": str(args.out)}))
    return 0 if ok else 1


def _paired_stats(gpu: np.ndarray, cpu: np.ndarray) -> dict[str, Any]:
    if gpu.shape != cpu.shape:
        return {"shape_match": False, "gpu_shape": list(gpu.shape), "cpu_shape": list(cpu.shape)}
    finite = np.isfinite(gpu) & np.isfinite(cpu)
    total = int(gpu.size)
    paired = int(finite.sum())
    if not paired:
        return {
            "shape_match": True,
            "total_cells": total,
            "finite_pair_count": 0,
            "finite_pair_fraction": 0.0,
        }
    diff = gpu[finite] - cpu[finite]
    return {
        "shape_match": True,
        "shape": list(gpu.shape),
        "total_cells": total,
        "finite_pair_count": paired,
        "finite_pair_fraction": paired / total,
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_signed_drift": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _worst_cell(
    gpu: np.ndarray, cpu: np.ndarray, lat: np.ndarray | None, lon: np.ndarray | None
) -> dict[str, Any] | None:
    if gpu.shape != cpu.shape:
        return None
    diff = gpu - cpu
    finite = np.isfinite(diff)
    if not finite.any():
        return None
    scored = np.where(finite, np.abs(diff), -np.inf)
    index = np.unravel_index(int(np.argmax(scored)), scored.shape)
    row: dict[str, Any] = {
        "index": [int(value) for value in index],
        "gpu": float(gpu[index]),
        "cpu": float(cpu[index]),
        "signed_difference": float(diff[index]),
        "absolute_difference": float(abs(diff[index])),
    }
    if lat is not None and lon is not None and lat.shape == gpu.shape and lon.shape == gpu.shape:
        row["latitude"] = float(lat[index])
        row["longitude"] = float(lon[index])
    return row


def _discover_domain(root: Path, domain: str) -> dict[datetime, Path]:
    discovered: dict[datetime, Path] = {}
    for path in sorted(root.glob(f"wrfout_{domain}_*")):
        if path.is_file():
            discovered[_valid_time(path)] = path
    return discovered


def _required_exact_leads(hours: int) -> tuple[int, ...]:
    if hours == 24:
        return (24,)
    if hours == 72:
        return (24, 72)
    raise ValueError(f"unsupported forecast hours: {hours}")


def command_score(args: argparse.Namespace) -> int:
    cpu_map = _discover_domain(args.truth_dir, "d02")
    gpu_map = _discover_domain(args.candidate_dir, "d02")
    common = sorted(set(cpu_map) & set(gpu_map))
    problems: list[str] = []
    timeline = []
    exact_rows: dict[str, Any] = {}
    endpoint_hashes: dict[str, Any] = {}
    required_exact_leads = _required_exact_leads(args.hours)
    for valid in common:
        lead = int(round((valid - EXPECTED_INIT).total_seconds() / 3600.0))
        if lead < 1 or lead > args.hours:
            continue
        with Dataset(cpu_map[valid], "r") as cpu_ds, Dataset(gpu_map[valid], "r") as gpu_ds:
            cpu_times = _read_times(cpu_ds)
            gpu_times = _read_times(gpu_ds)
            expected_label = valid.strftime("%Y-%m-%d_%H:%M:%S")
            if cpu_times != [expected_label] or gpu_times != [expected_label]:
                problems.append(
                    f"h{lead}: internal Times mismatch cpu={cpu_times} gpu={gpu_times} expected={expected_label}"
                )
            arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for field in REQUIRED_TRUTH_FIELDS:
                if field not in cpu_ds.variables or field not in gpu_ds.variables:
                    problems.append(f"h{lead}: missing paired field {field}")
                    continue
                arrays[field] = (_read_var(gpu_ds, field), _read_var(cpu_ds, field))
            if set(arrays) != set(REQUIRED_TRUTH_FIELDS):
                continue
            metrics = {field: _paired_stats(*arrays[field]) for field in REQUIRED_TRUTH_FIELDS}
            gpu_speed = np.hypot(arrays["U10"][0], arrays["V10"][0])
            cpu_speed = np.hypot(arrays["U10"][1], arrays["V10"][1])
            metrics["WSPD10"] = _paired_stats(gpu_speed, cpu_speed)
            binding_pass = all(
                metrics[field].get("finite_pair_fraction") == 1.0
                and metrics[field].get("rmse", math.inf) <= FROZEN_LIMITS[field]
                for field in REQUIRED_TRUTH_FIELDS
            )
            speed_finite = metrics["WSPD10"].get("finite_pair_fraction") == 1.0
            row = {
                "lead_h": lead,
                "valid_utc": valid.isoformat(),
                "metrics": metrics,
                "binding_frozen_gate_pass": binding_pass,
                "wind_speed_derived_ceiling_pass": speed_finite
                and metrics["WSPD10"].get("rmse", math.inf) <= DERIVED_WSPD_CEILING,
            }
            timeline.append(row)
            if lead in required_exact_leads:
                lat = _read_var(cpu_ds, "XLAT") if "XLAT" in cpu_ds.variables else None
                lon = _read_var(cpu_ds, "XLONG") if "XLONG" in cpu_ds.variables else None
                worst = {
                    field: _worst_cell(*arrays[field], lat, lon)
                    for field in REQUIRED_TRUTH_FIELDS
                }
                worst["WSPD10"] = _worst_cell(gpu_speed, cpu_speed, lat, lon)
                row["worst_cells"] = worst
                exact_rows[f"h{lead}"] = row
                endpoint_hashes[f"h{lead}"] = {
                    "cpu": {
                        "path": str(cpu_map[valid]),
                        "sha256": sha256_file(cpu_map[valid]),
                    },
                    "gpu": {
                        "path": str(gpu_map[valid]),
                        "sha256": sha256_file(gpu_map[valid]),
                    },
                }

    observed_leads = [row["lead_h"] for row in timeline]
    expected_leads = list(range(1, args.hours + 1))
    if observed_leads != expected_leads:
        problems.append(
            f"candidate/truth pairing is not exact h1..h{args.hours}: "
            f"observed {observed_leads[:4]}..."
            f"{observed_leads[-4:] if observed_leads else []} "
            f"({len(observed_leads)} leads)"
        )
    required_labels = tuple(f"h{lead}" for lead in required_exact_leads)
    for required in required_labels:
        if required not in exact_rows:
            problems.append(f"missing mandatory exact score {required}")
    first_failure = next(
        (row for row in timeline if not row["binding_frozen_gate_pass"]), None
    )
    exact_pass = all(
        exact_rows.get(label, {}).get("binding_frozen_gate_pass", False)
        for label in required_labels
    )
    verdict = (
        ("PASS__SP2_24H_DRIFT_GATE_MET" if args.hours == 24 else "PASS__SP2_CHAIN_DRIFT_GATE_MET")
        if not problems and exact_pass
        else (
            "FAIL__SP2_24H_DRIFT_GATE_MISSED"
            if args.hours == 24
            else "FAIL__SP2_CHAIN_DRIFT_GATE_MISSED"
        )
        if exact_rows and not exact_pass
        else "BLOCKED__INVALID_PAIRED_TRAJECTORY"
    )
    payload = {
        "schema": "wrfgpu2-v0234-sp2-drift-score-v2",
        "generated_utc": utc_now(),
        "verdict": verdict,
        "forecast_hours": args.hours,
        "definition": {
            "population": "every d02 cell; no crop, mask, clamp, or tolerance weakening",
            "difference_sign": "GPU candidate minus CPU WRF truth",
            "drift": "mean signed difference at the exact lead",
            "wind_speed": "sqrt(U10**2 + V10**2) independently on candidate and truth",
        },
        "candidate_dir": str(args.candidate_dir),
        "truth_dir": str(args.truth_dir),
        "init_utc": EXPECTED_INIT.isoformat(),
        "paired_leads_h": observed_leads,
        "frozen_binding_limits": {
            "T2_RMSE_K": 1.5,
            "U10_RMSE_m_s-1": 1.5,
            "V10_RMSE_m_s-1": 1.5,
            "finite_pair_fraction": 1.0,
        },
        "wind_speed_policy": {
            "separately_frozen_scalar_limit_exists": False,
            "derived_consistency_ceiling_m_s-1": DERIVED_WSPD_CEILING,
            "derived_not_binding_policy": True,
        },
        "exact_requested_leads": exact_rows,
        "metrics_by_lead": timeline,
        "first_binding_gate_failure": first_failure,
        "endpoint_file_hashes": endpoint_hashes,
        "problems": problems,
    }
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "paired_leads": len(observed_leads),
                "h24": exact_rows.get("h24", {}).get("metrics"),
                "h72": exact_rows.get("h72", {}).get("metrics"),
                "out": str(args.out),
            },
            default=str,
        )
    )
    return 0 if verdict.startswith("PASS__") else 1


def _iter_files(paths: Iterable[Path], excluded: Path) -> Iterable[Path]:
    seen = set()
    for supplied in paths:
        if supplied.is_file():
            candidates = [supplied]
        elif supplied.is_dir():
            candidates = sorted(path for path in supplied.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(supplied)
        for path in candidates:
            resolved = path.resolve()
            if resolved == excluded.resolve() or resolved in seen:
                continue
            seen.add(resolved)
            yield path


def command_manifest(args: argparse.Namespace) -> int:
    rows = []
    for path in _iter_files(args.path, args.out):
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "wrfgpu2-v0234-sp2-drift-proof-manifest-v1",
        "generated_utc": utc_now(),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    write_json(args.out, payload)
    print(json.dumps({"file_count": len(rows), "total_bytes": payload["total_bytes"], "out": str(args.out)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--repo", type=Path, required=True)
    preflight.add_argument("--input-dir", type=Path, required=True)
    preflight.add_argument("--truth-dir", type=Path, required=True)
    preflight.add_argument("--wrf-root", type=Path, required=True)
    preflight.add_argument("--namespace", type=Path, required=True)
    preflight.add_argument("--cache-dir", type=Path, required=True)
    preflight.add_argument("--nonce", required=True)
    preflight.add_argument("--hours", type=int, choices=SUPPORTED_FORECAST_HOURS, default=72)
    preflight.add_argument("--out", type=Path, required=True)
    preflight.set_defaults(func=command_preflight)

    arm = sub.add_parser("arm")
    arm.add_argument("--repo", type=Path, required=True)
    arm.add_argument("--input-dir", type=Path, required=True)
    arm.add_argument("--truth-dir", type=Path, required=True)
    arm.add_argument("--wrf-root", type=Path, required=True)
    arm.add_argument("--namespace", type=Path, required=True)
    arm.add_argument("--cache-dir", type=Path, required=True)
    arm.add_argument("--nonce", required=True)
    arm.add_argument("--hours", type=int, choices=SUPPORTED_FORECAST_HOURS, default=72)
    arm.set_defaults(func=command_arm)

    score = sub.add_parser("score")
    score.add_argument("--truth-dir", type=Path, required=True)
    score.add_argument("--candidate-dir", type=Path, required=True)
    score.add_argument("--hours", type=int, choices=SUPPORTED_FORECAST_HOURS, default=72)
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(func=command_score)

    freedom = sub.add_parser("freedom")
    freedom.add_argument("--arm-descriptor", type=Path, required=True)
    freedom.add_argument("--out", type=Path, required=True)
    freedom.set_defaults(func=command_freedom)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--path", type=Path, action="append", required=True)
    manifest.add_argument("--out", type=Path, required=True)
    manifest.set_defaults(func=command_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
