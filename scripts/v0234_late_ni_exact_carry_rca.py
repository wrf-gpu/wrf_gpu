#!/usr/bin/env python3
"""Exact accepted-trajectory acquisition and first-red late-Ni RCA.

The module stays stdlib-only until the locked CUDA environment and manager
authorization have been authenticated. GPU mode has two separately authorized
arms:

* ``smoke``: one real d03 Step0 step, ordinary versus callback-free recorder;
* ``exact``: one accepted Step0 trajectory, bounded health after the sealed
  step-9000 gate, stop at the first red dispatch, then exact-input local replay.

No production model operation is changed. Recorder evidence is decoded only
after complete output-carry byte identity passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-late-ni-rootcause"
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
INPUT_DIR = CASE_ROOT / "run/wrf"
LINEAGE = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a"
RUNTIME_AUTHORITY = LINEAGE / "v0234_v10_runtime_authority_complete1_5a6298fb"
AUTOTUNE_PIN = (
    LINEAGE
    / "v0234_gpt_v10_replay_c17fca1e201ae106_reference/autotune-results.pb"
)
AUTOTUNE_PIN_SHA256 = "6a0f30bc8e2ab1ca646ab346f85221565149f4a63b04193a9d76dc334e57ce18"
CANDIDATE_COMMIT = "3b81fb5b093639e70c12cce87d602c45b326b18b"
CANDIDATE_SRC_TREE = "e627605f6a8bc0dc23f5c474be4bb532b99297c1"
INPUT_SHA256 = {
    "namelist.input": "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}
RUNTIME_AUTHORITY_FILES = {
    "phys/module_ra_rrtmg_lw.F": (
        652002,
        "c7a5238612aa8a4213c8d3af6708ec6a5248e6701e19758a80e563905d306de3",
    ),
    "run/CAMtr_volume_mixing_ratio": (
        42780,
        "9a427fd106f8e36b30e0b29266bff1398b025b82af5b878e5a7a8e9dfe268ca7",
    ),
    "run/GENPARM.TBL": (
        261,
        "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2",
    ),
    "run/MPTABLE.TBL": (
        56140,
        "7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d",
    ),
    "run/SOILPARM.TBL": (
        6557,
        "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f",
    ),
}
CONSUMED_NONCES = frozenset(
    {
        "ad0e60072a2a0df1ceacf0f5593e3834f2d0da10086f46316375f8f17aa",
        "8520924a5f97acc1ea8337bb16ea860903b8436e2d30b10b6efca9d54f55552c",
        "ca1df30181823579e35b5c3dbadfa09782cfde6104efe3793937ed9d38e02f2a",
        "fbb2716e855e4fe572d1bbef538f8b26ce5f4a106a4ed8f7b44b41400e852687",
        "c17fca1e201ae106cf8ad77463686823c82226d59c7dea01e741c0be0ba7f6bc",
        "30b5cf89ff6027c40d5b441442f194308ac1e6f3239d486bcace39c60ba8c9dc",
    }
)
LOCK_LABEL = "v0234-gpt-late-ni-rootcause"
NONCE_PATTERN = re.compile(r"[0-9a-f]{32,128}")
EXPECTED_AFFINITY = [13, 14, 15, 29, 30, 31]
GATE_STEPS = {"d01": 1000, "d02": 3000, "d03": 9000}
EXACT_LIMIT_STEPS = {"d01": 1156, "d02": 3468, "d03": 10404}
EXPECTED_FAILURE = {"domain": "d03", "field": "Ni", "python_index": [0, 1, 1]}
PREEMPT_PATHS = (
    Path("/tmp/PREEMPT_GPU"),
    Path("/tmp/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_GPU"),
)
NIGHTLY_ACTIVE = Path("<DATA_ROOT>/alisios/state/nightly18z/active.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(
            clean, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        pickle.dump(value, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def namespace_for(nonce: str, arm: str) -> str:
    if NONCE_PATTERN.fullmatch(nonce) is None:
        raise RuntimeError("manager nonce must be 32--128 lowercase hex characters")
    if arm not in {"smoke", "exact"}:
        raise ValueError(arm)
    if nonce in CONSUMED_NONCES:
        raise RuntimeError("manager nonce was consumed by an earlier GPU arm")
    return f"v0234_gpt_late_ni_{nonce[:16]}_{arm}"


def segment_lengths(start: int, stop: int, *, cadence: int = 67) -> tuple[int, ...]:
    if not 0 <= int(start) <= int(stop):
        raise ValueError((start, stop))
    values: list[int] = []
    current = int(start)
    while current < int(stop):
        step = min(int(cadence), int(stop) - current)
        values.append(step)
        current += step
    return tuple(values)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(ROOT), *args), text=True
    ).strip()


def _authorization_schema(arm: str) -> str:
    return f"gpuwrf.v0234.late-ni-{arm}-manager-authorization.v1"


def validate_authorization(payload: Mapping[str, Any], *, arm: str) -> str:
    nonce = payload.get("nonce")
    if not isinstance(nonce, str):
        raise RuntimeError("manager authorization has no nonce")
    runner_commit = payload.get("runner_commit")
    if (
        not isinstance(runner_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", runner_commit) is None
    ):
        raise RuntimeError("manager authorization has no valid runner commit")
    if subprocess.run(
        (
            "git", "-C", str(ROOT), "merge-base", "--is-ancestor",
            runner_commit, "HEAD",
        ),
        check=False,
    ).returncode != 0:
        raise RuntimeError("authorized runner commit is not an ancestor of HEAD")
    committed_runner = subprocess.check_output(
        (
            "git", "-C", str(ROOT), "show",
            f"{runner_commit}:scripts/v0234_late_ni_exact_carry_rca.py",
        )
    )
    committed_runner_sha256 = hashlib.sha256(committed_runner).hexdigest()
    required: dict[str, Any] = {
        "schema": _authorization_schema(arm),
        "verdict": "MANAGER_GPU_AUTHORIZED",
        "manager_pane": "0:1",
        "arm": arm,
        "nonce": nonce,
        "namespace": namespace_for(nonce, arm),
        "lock_label": LOCK_LABEL,
        "authorized_model_processes": 1,
        "baseline_processes_authorized": 0,
        "runner_commit": runner_commit,
        "runner_sha256": committed_runner_sha256,
        "candidate_model_commit": CANDIDATE_COMMIT,
        "candidate_src_gpuwrf_tree": CANDIDATE_SRC_TREE,
        "output_materialization": False,
        "existing_cpu_reference_only": True,
    }
    if arm == "smoke":
        required.update(
            {
                "domain": "d03",
                "native_step_upper_bound": 1,
                "ordinary_native_step_evaluations": 1,
                "recorder_native_step_evaluations": 1,
                "total_native_step_evaluations": 2,
                "ordinary_recorder_identity_required": True,
            }
        )
    else:
        required.update(
            {
                "root_step_upper_bound": EXACT_LIMIT_STEPS["d01"],
                "d03_step_upper_bound": EXACT_LIMIT_STEPS["d03"],
                "monitor_after_d03_step": GATE_STEPS["d03"],
                "stop_at_first_red": True,
                "exact_carry_retention": True,
                "ordinary_prefix_bisection": True,
                "ordinary_replay_step_upper_bound": 3,
                "ordinary_replay_byte_identity_required": True,
                "recorder_payload_authorized": False,
                "proof_health_transfer_after_every_dispatch": True,
            }
        )
    canonical = canonical_digest(payload)
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if payload.get("proof_sha256") != canonical or mismatches:
        raise RuntimeError(
            f"manager authorization mismatch: canonical={canonical} "
            f"embedded={payload.get('proof_sha256')} mismatches={mismatches}"
        )
    current_runner_sha256 = sha256_file(Path(__file__).resolve())
    if current_runner_sha256 != committed_runner_sha256:
        raise RuntimeError(
            "working runner bytes differ from manager-authorized committed bytes"
        )
    return canonical


def read_authorization(path: Path, *, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.parent.resolve() != SPRINT.resolve():
        raise RuntimeError("authorization must live in this sprint directory")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("authorization is missing or symlinked")
    payload = json.loads(path.read_text())
    canonical = validate_authorization(payload, arm=arm)
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "canonical_sha256": canonical,
    }


def assert_preemption_clear(label: str) -> dict[str, Any]:
    present = [str(path) for path in PREEMPT_PATHS if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"{label}: preemption sentinel present: {present}")
    nightly = None
    if NIGHTLY_ACTIVE.exists():
        nightly = json.loads(NIGHTLY_ACTIVE.read_text())
        if nightly.get("active") is True:
            raise RuntimeError(f"{label}: nightly production active")
    return {
        "label": label,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "preemption_sentinels_absent": True,
        "nightly_active": None if nightly is None else nightly.get("active"),
    }


def validate_locked_environment(nonce: str) -> dict[str, Any]:
    required = {
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "true",
        "CUDA_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "GPUWRF_ALLOCATOR": "cuda_async",
        "GPUWRF_FINITE_CHECK": "1",
        "GPUWRF_NESTED_FUSE": "0",
        "GPUWRF_NESTED_DEFUSE_COMPILE": "0",
        "GPUWRF_NESTED_PARALLEL_COMPILE": "0",
        "GPUWRF_NESTED_AOT": "0",
        "GPUWRF_AOT_VERIFY": "0",
        "GPUWRF_NESTED_ASYNC_OUTPUT": "0",
        "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
        "GPUWRF_FULL_WRFOUT": "1",
        "GPUWRF_TRAINING_OUTPUT_SUBSET": "0",
        "GPUWRF_BATCH_ENSEMBLE": "1",
        "GPUWRF_NESTED_SYNC_MODE": "root",
        "GPUWRF_BITWISE": "1",
        "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
        "GPUWRF_JAX_CACHE": "0",
        "GPUWRF_JAX_CACHE_LOCK": "0",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
        "GPUWRF_WRF_ROOT": str(RUNTIME_AUTHORITY),
        "GPUWRF_LATE_NI_NONCE": nonce,
        "XLA_FLAGS": f"--xla_gpu_load_autotune_results_from={AUTOTUNE_PIN}",
    }
    actual = {name: os.environ.get(name) for name in required}
    if actual != required:
        raise RuntimeError(f"locked CUDA environment mismatch: {actual!r}")
    if sorted(os.sched_getaffinity(0)) != EXPECTED_AFFINITY:
        raise RuntimeError(f"CPU affinity mismatch: {sorted(os.sched_getaffinity(0))}")
    lock_required = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": "9",
        "GPUWRF_GPU_LOCK_FILE": "/tmp/wrf_gpu2_gpu.lock",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": "/tmp/wrf_gpu2_gpu.lock.holder",
        "GPUWRF_GPU_LOCK_LABEL": LOCK_LABEL,
    }
    if any(os.environ.get(name) != value for name, value in lock_required.items()):
        raise RuntimeError("live GPU flock environment mismatch")
    token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN", "")
    if not token:
        raise RuntimeError("GPU lock token is absent")
    lock_stat = Path(lock_required["GPUWRF_GPU_LOCK_FILE"]).stat()
    fd_stat = os.fstat(9)
    if (lock_stat.st_dev, lock_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino):
        raise RuntimeError("fd9 is not the shared GPU lock")
    holder = Path(lock_required["GPUWRF_GPU_LOCK_HOLDER_FILE"]).read_text()
    if f"holder={LOCK_LABEL} " not in holder or f"token={token} " not in holder:
        raise RuntimeError("GPU lock holder sidecar differs from inherited lease")
    return {
        "required_environment": actual,
        "cpu_affinity": EXPECTED_AFFINITY,
        "lock_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "fd9_inode_verified": True,
    }


def assert_source_authority(*, require_clean: bool) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    ancestor = subprocess.run(
        ("git", "-C", str(ROOT), "merge-base", "--is-ancestor", CANDIDATE_COMMIT, head),
        check=False,
    ).returncode == 0
    src_tree = _git("rev-parse", "HEAD:src/gpuwrf")
    if require_clean and dirty:
        raise RuntimeError(f"GPU launch requires a clean worktree: {dirty}")
    if not ancestor or src_tree != CANDIDATE_SRC_TREE:
        raise RuntimeError(
            f"accepted model authority changed: ancestor={ancestor} src_tree={src_tree}"
        )
    input_rows = {}
    for relative, expected in INPUT_SHA256.items():
        path = INPUT_DIR / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"fixture authority unavailable: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"fixture authority changed: {relative}: {actual}")
        input_rows[relative] = actual
    inventory = sorted(
        str(path.relative_to(RUNTIME_AUTHORITY))
        for path in RUNTIME_AUTHORITY.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if inventory != sorted(RUNTIME_AUTHORITY_FILES):
        raise RuntimeError(f"runtime authority inventory changed: {inventory}")
    runtime_rows = {}
    for relative, (expected_bytes, expected_sha) in RUNTIME_AUTHORITY_FILES.items():
        path = RUNTIME_AUTHORITY / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != expected_sha
        ):
            raise RuntimeError(f"runtime authority changed: {relative}")
        runtime_rows[relative] = {
            "bytes": expected_bytes,
            "sha256": expected_sha,
        }
    if not AUTOTUNE_PIN.is_file() or AUTOTUNE_PIN.is_symlink():
        raise RuntimeError("autotune authority is unavailable")
    if sha256_file(AUTOTUNE_PIN) != AUTOTUNE_PIN_SHA256:
        raise RuntimeError("autotune authority changed")
    return {
        "head": head,
        "candidate_model_commit": CANDIDATE_COMMIT,
        "candidate_is_ancestor": ancestor,
        "candidate_src_gpuwrf_tree": src_tree,
        "worktree_clean": not bool(dirty),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "autotune_pin_sha256": AUTOTUNE_PIN_SHA256,
        "fixture_input_sha256": input_rows,
        "runtime_authority": {
            "root": str(RUNTIME_AUTHORITY.resolve()),
            "files": runtime_rows,
        },
    }


def cpu_preflight(output: Path) -> int:
    if "jax" in sys.modules or any(name.startswith("gpuwrf") for name in sys.modules):
        raise RuntimeError("CPU preflight imported JAX/gpuwrf")
    authority = assert_source_authority(require_clean=False)
    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.late-ni-exact-carry-cpu-preflight.v1",
        "verdict": "LATE_NI_EXACT_CARRY_RUNNER_READY",
        "authority": authority,
        "arms": {
            "smoke": {
                "fresh_manager_authorization_required": True,
                "model_processes": 1,
                "domain": "d03",
                "step_upper_bound": 1,
                "ordinary_native_step_evaluations": 1,
                "recorder_native_step_evaluations": 1,
                "total_native_step_evaluations": 2,
                "ordinary_recorder_identity_required": True,
            },
            "exact": {
                "separate_fresh_manager_authorization_required": True,
                "model_processes": 1,
                "gate_steps": GATE_STEPS,
                "hard_limits": EXACT_LIMIT_STEPS,
                "stop_first_red": True,
                "ordinary_prefix_bisection": True,
                "ordinary_replay_step_upper_bound": 3,
                "ordinary_replay_byte_identity_required": True,
                "recorder_payload_authorized": False,
                "proof_health_transfer_after_every_dispatch": True,
            },
        },
        "segment_schedule_to_gate": list(segment_lengths(0, GATE_STEPS["d01"])),
        "segment_schedule_after_gate": list(
            segment_lengths(GATE_STEPS["d01"], EXACT_LIMIT_STEPS["d01"])
        ),
        "gpu_commands": 0,
        "gpu_queries": 0,
        "authorization_files_read": 0,
        "jax_or_gpuwrf_imported": False,
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_json(output, proof)
    print(json.dumps({"verdict": proof["verdict"], "proof_sha256": proof["proof_sha256"]}))
    return 0


class FirstRed(RuntimeError):
    """Internal stop signal retaining the exact device input/output carries."""


def _gpu_run(authorization_path: Path, run_dir: Path, *, arm: str) -> int:
    started = datetime.now(timezone.utc)
    wall_started = time.perf_counter()
    authorization, authorization_row = read_authorization(authorization_path, arm=arm)
    expected = LINEAGE / authorization["namespace"]
    if run_dir.resolve() != expected.resolve():
        raise RuntimeError(f"run-dir mismatch: expected={expected} actual={run_dir}")
    if run_dir.exists() or run_dir.is_symlink():
        raise RuntimeError(f"fresh namespace required: {run_dir}")
    source_authority = assert_source_authority(require_clean=True)
    environment = validate_locked_environment(authorization["nonce"])
    preemption_pre = assert_preemption_clear(f"late-ni-{arm}-startup")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        import dataclasses
        import jax
        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "gpu" or any(
            device.platform != "gpu" for device in jax.devices()
        ):
            raise RuntimeError(f"CUDA backend unavailable: {jax.devices()!r}")

        import gpuwrf
        from gpuwrf.integration.nested_pipeline import (
            NestedPipelineConfig,
            _load_domains,
            _nested_sync_mode_from_env,
            _output_alarm_steps_by_domain,
            _output_cadence_steps_by_domain,
            domain_names_for,
        )
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.runtime.domain_tree import (
            DomainTree,
            _operational_force,
            _prepare_operational_domain_tree_runtime,
            run_operational_domain_tree,
        )
        from gpuwrf.runtime.operational_mode import (
            RCA_BOUNDARY_FIELDS,
            _rca_array_health,
            advance_chunk_with_corrected_ni_rca,
            build_clock_base,
        )
        from gpuwrf.runtime.finite_state_guard import (
            PROGNOSTIC_STATE_FIELDS,
            _finite_check_candidates,
            assert_state_finite_at_boundary,
        )
        from scripts import v0234_corrected_ni_rca_gpu as rca

        gpuwrf_path = Path(gpuwrf.__file__).resolve()
        if ROOT not in gpuwrf_path.parents:
            raise RuntimeError(f"gpuwrf imported outside launch worktree: {gpuwrf_path}")

        config = NestedPipelineConfig(
            input_dir=INPUT_DIR,
            output_dir=run_dir / "unused-output",
            proof_dir=run_dir / "unused-pipeline-proof",
            hours=18,
            max_dom=3,
            feedback=False,
            emit_initial_history=True,
        )
        names = domain_names_for(3)
        hierarchy, bundles, metadata, run_start, dt_by_domain, carries = _load_domains(
            config, names
        )
        if names != ("d01", "d02", "d03") or dt_by_domain != {
            "d01": 54.0,
            "d02": 18.0,
            "d03": 6.0,
        }:
            raise RuntimeError(f"domain hierarchy changed: {names}/{dt_by_domain}")
        options = {
            name: [
                int(bundles[name].namelist.moist_adv_opt),
                int(bundles[name].namelist.scalar_adv_opt),
            ]
            for name in names
        }
        if options != {name: [1, 1] for name in names}:
            raise RuntimeError(f"accepted 1/1 scalar options changed: {options}")
        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
        runtime = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
        normal_advance = runtime.advance

        scratch_fields = (
            "t_2ave", "ww", "mudf", "muave", "muts", "ph_tend",
            "u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save",
            "ww_save", "rthraten",
        )
        scalar_boundary_fields = (
            "qv_bdy", "qc_bdy", "qr_bdy", "qi_bdy", "qs_bdy", "qg_bdy",
            "Ni_bdy", "Nr_bdy",
        )
        requested_boundary_fields = tuple(
            dict.fromkeys(RCA_BOUNDARY_FIELDS + scalar_boundary_fields)
        )
        state_fields_by_domain = {
            name: tuple(
                field
                for field, _value in _finite_check_candidates(
                    carries[name].state, PROGNOSTIC_STATE_FIELDS
                )
            )
            for name in names
        }
        if len(set(state_fields_by_domain.values())) != 1:
            raise RuntimeError(
                f"active finite-guard fields differ by domain: {state_fields_by_domain}"
            )
        state_fields = state_fields_by_domain["d03"]
        boundary_fields_by_domain = {
            name: tuple(
                field
                for field in requested_boundary_fields
                if getattr(carries[name].state, field, None) is not None
                and np.issubdtype(
                    np.dtype(getattr(carries[name].state, field).dtype),
                    np.inexact,
                )
            )
            for name in names
        }
        absent_health_fields = {
            "prognostic": sorted(set(PROGNOSTIC_STATE_FIELDS) - set(state_fields)),
            "boundary_by_domain": {
                name: sorted(
                    set(requested_boundary_fields)
                    - set(boundary_fields_by_domain[name])
                )
                for name in names
            },
        }
        health_names_by_domain = {
            name: (
                tuple(f"state.{field}" for field in state_fields)
                + tuple(f"carry.{field}" for field in scratch_fields)
                + tuple(
                    f"boundary.{field}"
                    for field in boundary_fields_by_domain[name]
                )
            )
            for name in names
        }

        def build_carry_health(active_boundary_fields):
            @jax.jit
            def carry_health(carry):
                values = (
                    tuple(getattr(carry.state, name) for name in state_fields)
                    + tuple(getattr(carry, name) for name in scratch_fields)
                    + tuple(
                        getattr(carry.state, name)
                        for name in active_boundary_fields
                    )
                )
                return jnp.stack(
                    tuple(_rca_array_health(value) for value in values), axis=0
                )

            return carry_health

        carry_health_by_domain = {
            name: build_carry_health(boundary_fields_by_domain[name])
            for name in names
        }

        def health_host(carry, domain: str) -> np.ndarray:
            return np.asarray(
                jax.device_get(carry_health_by_domain[domain](carry)),
                dtype=np.float64,
            )

        def decode_health(
            values: np.ndarray, carry, domain: str
        ) -> dict[str, Any]:
            rows = []
            first_bad = None
            active_boundary_fields = boundary_fields_by_domain[domain]
            arrays = (
                tuple(getattr(carry.state, name) for name in state_fields)
                + tuple(getattr(carry, name) for name in scratch_fields)
                + tuple(
                    getattr(carry.state, name)
                    for name in active_boundary_fields
                )
            )
            for name, row, array in zip(
                health_names_by_domain[domain], values, arrays, strict=True
            ):
                count = int(row[0])
                flat = int(row[1])
                index = None
                if flat >= 0:
                    index = [
                        int(value)
                        for value in np.unravel_index(flat, tuple(int(d) for d in array.shape))
                    ]
                item = {
                    "field": name,
                    "nonfinite_count": count,
                    "first_nonfinite_python_index": index,
                    "max_abs_finite": rca.json_number(row[2]),
                    "min_finite": rca.json_number(row[5]),
                    "max_finite": rca.json_number(row[7]),
                }
                rows.append(item)
                if count and first_bad is None:
                    first_bad = item
            return {"first_bad": first_bad, "fields": rows}

        def manifest(value) -> dict[str, Any]:
            return rca.host_tree_manifest(value)

        def compare(left, right) -> dict[str, Any]:
            return rca.compare_manifests(left, right)

        if arm == "smoke":
            assert_preemption_clear("late-ni-smoke-pre-ordinary")
            d03_edge = next(
                edge for edge in tree.children("d02") if edge.child == "d03"
            )
            input_carry = _operational_force(
                d03_edge, carries["d02"], carries["d03"]
            )
            boundary_time_levels = {
                name: int(getattr(input_carry.state, name).shape[0])
                for name in RCA_BOUNDARY_FIELDS
            }
            if set(boundary_time_levels.values()) != {2}:
                raise RuntimeError(
                    "smoke did not build a live two-time boundary package: "
                    f"{boundary_time_levels}"
                )
            ordinary = normal_advance("d03", input_carry, 1, 1)
            jax.block_until_ready(ordinary.state.theta)
            ordinary_host = jax.device_get(ordinary)
            ordinary_manifest = manifest(ordinary_host)
            ordinary_health = decode_health(
                health_host(ordinary, "d03"), ordinary, "d03"
            )

            namelist = bundles["d03"].namelist
            clock = build_clock_base(namelist)
            lowered = advance_chunk_with_corrected_ni_rca.lower(
                input_carry,
                namelist,
                jnp.asarray(1, dtype=jnp.int32),
                clock,
                n_steps=1,
                cadence=int(namelist.radiation_cadence_steps),
            )
            stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
            forbidden = [
                token
                for token in (
                    "xla_python_cpu_callback", "host_callback", "io_callback",
                    "pure_callback", "outside_compilation",
                )
                if token in stablehlo.lower()
            ]
            executable = lowered.compile()
            observed = executable(
                input_carry,
                namelist,
                jnp.asarray(1, dtype=jnp.int32),
                clock,
            )
            jax.block_until_ready(observed.carry.state.theta)
            recorder_host = jax.device_get(observed.carry)
            recorder_manifest = manifest(recorder_host)
            identity = compare(ordinary_manifest, recorder_manifest)
            recorder_health = decode_health(
                health_host(observed.carry, "d03"), observed.carry, "d03"
            )
            passed = bool(
                identity["all_leaf_output_bytes_equal"]
                and not forbidden
                and ordinary_health["first_bad"] is None
                and recorder_health["first_bad"] is None
            )
            proof = {
                "schema": "gpuwrf.v0234.late-ni-recorder-smoke-result.v1",
                "verdict": (
                    "LATE_NI_RECORDER_SMOKE_GREEN"
                    if passed
                    else "LATE_NI_RECORDER_SMOKE_RED"
                ),
                "authorization": authorization_row,
                "source_authority": source_authority,
                "environment": environment,
                "preemption_pre": preemption_pre,
                "fixture": {
                    "run_start": run_start.isoformat(),
                    "dt_by_domain": dt_by_domain,
                    "loaded_options": options,
                    "domain": "d03",
                    "native_step": 1,
                    "boundary_package_source": (
                        "real initial d02/d03 fixture via production _operational_force"
                    ),
                    "boundary_time_levels": boundary_time_levels,
                    "active_health_fields": {
                        "prognostic": list(state_fields),
                        "boundary_by_domain": {
                            name: list(fields)
                            for name, fields in boundary_fields_by_domain.items()
                        },
                    },
                    "absent_optional_health_fields": absent_health_fields,
                },
                "program": {
                    "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
                    "stablehlo_bytes": len(stablehlo.encode()),
                    "forbidden_callback_tokens": forbidden,
                    "callback_free": not forbidden,
                },
                "ordinary_manifest": ordinary_manifest,
                "recorder_manifest": recorder_manifest,
                "identity": identity,
                "ordinary_output_health": ordinary_health,
                "recorder_output_health": recorder_health,
                "preemption_post": assert_preemption_clear("late-ni-smoke-post"),
                "wall_seconds": time.perf_counter() - wall_started,
            }
            proof["proof_sha256"] = canonical_digest(proof)
            atomic_json(run_dir / "smoke-result.json", proof)
            print(json.dumps({
                "verdict": proof["verdict"],
                "proof": str((run_dir / "smoke-result.json").resolve()),
                "proof_sha256": proof["proof_sha256"],
                "identity": identity["all_leaf_output_bytes_equal"],
            }, sort_keys=True))
            return 0 if passed else 3

        class ScheduleOnlyOutput:
            wants_carry = True

            def __init__(self):
                self.calls: list[tuple[str, int]] = []

            def __call__(self, name: str, step: int, _carry: Any):
                self.calls.append((name, int(step)))
                return name, int(step)

        class Monitor:
            def __init__(self):
                self.enabled = False
                self.tail: list[dict[str, Any]] = []
                self.red: dict[str, Any] | None = None
                self.last_output: dict[str, Any] = {}

            def advance(self, name: str, carry, start_step: int, n_steps: int):
                if self.enabled:
                    input_decoded = decode_health(
                        health_host(carry, name), carry, name
                    )
                    if input_decoded["first_bad"] is not None:
                        edge = next(
                            (
                                edge
                                for parent in tree.edges.values()
                                for edge in parent
                                if edge.child == name
                            ),
                            None,
                        )
                        if edge is None or name not in self.last_output:
                            raise RuntimeError(
                                f"red advance input has no preceding force: {name}"
                            )
                        pre_force = self.last_output[name]
                        pre_force_health = decode_health(
                            health_host(pre_force, name), pre_force, name
                        )
                        if pre_force_health["first_bad"] is not None:
                            raise RuntimeError(
                                f"last recorded {name} output was already red"
                            )
                        parent_carry = self.last_output[edge.parent]
                        force_row = {
                            "event_kind": "nest_force",
                            "edge": f"{edge.parent}->{name}",
                            "before_native_step": int(start_step),
                            "first_bad": input_decoded["first_bad"],
                        }
                        self.tail.append(force_row)
                        self.tail = self.tail[-32:]
                        self.red = {
                            "event_kind": "nest_force",
                            "domain": name,
                            "parent_domain": edge.parent,
                            "native_start_step": int(start_step),
                            "native_end_step": int(start_step - 1),
                            "n_steps": 0,
                            "input": pre_force,
                            "output": carry,
                            "parent": parent_carry,
                            "input_health": pre_force_health,
                            "output_health": input_decoded,
                            "monitor_tail": list(self.tail),
                        }
                        raise FirstRed(
                            f"first red nest force {edge.parent}->{name} "
                            f"before step {start_step}: {input_decoded['first_bad']}"
                        )
                output = normal_advance(name, carry, start_step, n_steps)
                if not self.enabled:
                    return output
                values = health_host(output, name)
                decoded = decode_health(values, output, name)
                selected = {
                    row["field"]: {
                        key: row[key]
                        for key in ("nonfinite_count", "max_abs_finite", "min_finite", "max_finite")
                    }
                    for row in decoded["fields"]
                    if row["field"] in {
                        "state.Ni", "state.Nr", "state.qi", "state.qv",
                        "state.mu_total", "state.p_total", "state.w",
                        "carry.ww", "carry.muts", "carry.mudf", "boundary.Ni_bdy",
                    }
                }
                self.tail.append({
                    "domain": name,
                    "native_start_step": int(start_step),
                    "native_end_step": int(start_step + n_steps - 1),
                    "selected": selected,
                    "first_bad": decoded["first_bad"],
                })
                self.tail = self.tail[-32:]
                if decoded["first_bad"] is not None:
                    input_values = health_host(carry, name)
                    self.red = {
                        "event_kind": "advance",
                        "domain": name,
                        "native_start_step": int(start_step),
                        "native_end_step": int(start_step + n_steps - 1),
                        "n_steps": int(n_steps),
                        "input": carry,
                        "output": output,
                        "input_health": decode_health(
                            input_values, carry, name
                        ),
                        "output_health": decoded,
                        "monitor_tail": list(self.tail),
                    }
                    raise FirstRed(
                        f"first red {name} {start_step}..{start_step + n_steps - 1}: "
                        f"{decoded['first_bad']}"
                    )
                self.last_output[name] = output
                return output

        monitor = Monitor()
        monitored_runtime = dataclasses.replace(
            runtime, advance=monitor.advance, fused_cascade=None
        )
        output = ScheduleOnlyOutput()
        gen2_run = Gen2Run(INPUT_DIR)
        output_cadence, history_minutes = _output_cadence_steps_by_domain(
            gen2_run, names, dt_by_domain
        )
        schedules, nonintegral_alarms = _output_alarm_steps_by_domain(
            names,
            dt_by_domain,
            history_minutes,
            EXACT_LIMIT_STEPS,
        )
        if output_cadence != {"d01": 67, "d02": 200, "d03": 200}:
            raise RuntimeError(f"accepted output cadence changed: {output_cadence}")
        block_between, root_sync_cadence = _nested_sync_mode_from_env()
        own_steps = {name: 0 for name in names}
        segments: list[dict[str, Any]] = []

        def run_segment(steps: int, *, monitored: bool) -> None:
            nonlocal carries, own_steps
            start = int(own_steps["d01"])
            result = run_operational_domain_tree(
                tree,
                root_steps=int(steps),
                feedback_enabled=False,
                output=output,
                output_cadence_steps=output_cadence,
                output_alarm_steps=nonintegral_alarms,
                block_between=block_between,
                root_sync_cadence=root_sync_cadence,
                carries=carries,
                initial_own_steps=own_steps,
                prepared_runtime=(monitored_runtime if monitored else runtime),
            )
            jax.block_until_ready(tuple(state.theta for state in result.states.values()))
            carries = result.carries
            own_steps = {name: int(result.own_steps[name]) for name in names}
            for name, state in result.states.items():
                assert_state_finite_at_boundary(
                    state,
                    domain=name,
                    step=own_steps[name],
                    sim_time_s=own_steps[name] * dt_by_domain[name],
                )
            segments.append({
                "root_start": start,
                "root_steps": int(steps),
                "own_steps_after": dict(own_steps),
                "monitored": bool(monitored),
            })
            print(
                f"LATE_NI_PREFIX monitored={int(monitored)} own_steps={own_steps}",
                flush=True,
            )

        for steps in segment_lengths(0, GATE_STEPS["d01"]):
            assert_preemption_clear(f"late-ni-prefix-{own_steps['d01']}")
            run_segment(steps, monitored=False)
        if own_steps != GATE_STEPS:
            raise RuntimeError(f"step9000 gate clocks changed: {own_steps}")
        gate_health = {
            name: decode_health(
                health_host(carries[name], name), carries[name], name
            )
            for name in names
        }
        gate_bad = {
            name: row["first_bad"]
            for name, row in gate_health.items()
            if row["first_bad"] is not None
        }
        if gate_bad:
            raise RuntimeError(f"promoted carry already red at sealed step9000 gate: {gate_bad}")
        monitor.enabled = True
        monitor.last_output = dict(carries)
        caught = None
        try:
            for steps in segment_lengths(GATE_STEPS["d01"], EXACT_LIMIT_STEPS["d01"]):
                assert_preemption_clear(f"late-ni-monitored-{own_steps['d01']}")
                run_segment(steps, monitored=True)
        except FirstRed as exc:
            caught = str(exc)

        if monitor.red is None:
            proof = {
                "schema": "gpuwrf.v0234.late-ni-exact-first-red-result.v1",
                "verdict": "LATE_NI_EXACT_EVENT_NOT_REPRODUCED",
                "authorization": authorization_row,
                "source_authority": source_authority,
                "environment": environment,
                "preemption_pre": preemption_pre,
                "fixture": {
                    "run_start": run_start.isoformat(),
                    "dt_by_domain": dt_by_domain,
                    "loaded_options": options,
                    "history_interval_minutes": history_minutes,
                    "health_field_inventory": {
                        "prognostic": list(state_fields),
                        "boundary_by_domain": {
                            name: list(fields)
                            for name, fields in boundary_fields_by_domain.items()
                        },
                        "absent_optional": absent_health_fields,
                    },
                    "output_alarm_schedules": {
                        name: list(values) for name, values in schedules.items()
                    },
                },
                "gate_steps": GATE_STEPS,
                "gate_health": gate_health,
                "final_own_steps": own_steps,
                "hard_limits": EXACT_LIMIT_STEPS,
                "segments": segments,
                "monitor_tail": monitor.tail,
                "output_schedule_calls": [list(value) for value in output.calls],
                "wall_seconds": time.perf_counter() - wall_started,
            }
            proof["proof_sha256"] = canonical_digest(proof)
            atomic_json(run_dir / "exact-result.json", proof)
            print(json.dumps({
                "verdict": proof["verdict"],
                "proof": str((run_dir / "exact-result.json").resolve()),
                "proof_sha256": proof["proof_sha256"],
            }, sort_keys=True))
            return 3

        red = monitor.red
        event_kind = str(red["event_kind"])
        domain = red["domain"]
        start_step = int(red["native_start_step"])
        n_steps = int(red["n_steps"])
        end_step = int(red["native_end_step"])
        input_carry = red.pop("input")
        ordinary_output = red.pop("output")
        parent_carry = red.pop("parent", None)
        assert_preemption_clear("late-ni-event-local-pre")
        input_host = jax.device_get(input_carry)
        ordinary_host = jax.device_get(ordinary_output)
        input_path = run_dir / (
            f"last-green-{event_kind}-input-{domain}-step-{start_step - 1}.pkl"
        )
        output_path = run_dir / (
            f"first-red-{event_kind}-output-{domain}-step-{end_step}.pkl"
        )
        atomic_pickle(input_path, input_host)
        atomic_pickle(output_path, ordinary_host)
        input_manifest = manifest(input_host)
        ordinary_manifest = manifest(ordinary_host)

        retained = {
            "input": {
                "path": str(input_path.resolve()),
                "bytes": input_path.stat().st_size,
                "file_sha256": sha256_file(input_path),
                "manifest": input_manifest,
            },
            "ordinary_output": {
                "path": str(output_path.resolve()),
                "bytes": output_path.stat().st_size,
                "file_sha256": sha256_file(output_path),
                "manifest": ordinary_manifest,
            },
        }

        if event_kind == "nest_force":
            if parent_carry is None:
                raise RuntimeError("nest-force event did not retain its parent carry")
            parent_domain = str(red["parent_domain"])
            edge = next(
                edge
                for edge in tree.children(parent_domain)
                if edge.child == domain
            )
            parent_host = jax.device_get(parent_carry)
            parent_path = run_dir / (
                f"nest-force-parent-{parent_domain}-before-{domain}-step-"
                f"{start_step}.pkl"
            )
            atomic_pickle(parent_path, parent_host)
            replay_output = _operational_force(edge, parent_carry, input_carry)
            jax.block_until_ready(replay_output.state.theta)
            replay_host = jax.device_get(replay_output)
            replay_manifest = manifest(replay_host)
            comparison = compare(ordinary_manifest, replay_manifest)
            retained["parent"] = {
                "path": str(parent_path.resolve()),
                "bytes": parent_path.stat().st_size,
                "file_sha256": sha256_file(parent_path),
                "manifest": manifest(parent_host),
            }
            admissible = bool(comparison["all_leaf_output_bytes_equal"])
            verdict = (
                "LATE_NI_EXACT_FIRST_RED_LOCALIZED"
                if admissible
                else "LATE_NI_FORCE_REPLAY_IDENTITY_BLOCKED"
            )
            proof = {
                "schema": "gpuwrf.v0234.late-ni-exact-first-red-result.v1",
                "verdict": verdict,
                "authorization": authorization_row,
                "source_authority": source_authority,
                "environment": environment,
                "preemption_pre": preemption_pre,
                "fixture": {
                    "run_start": run_start.isoformat(),
                    "dt_by_domain": dt_by_domain,
                    "loaded_options": options,
                    "history_interval_minutes": history_minutes,
                    "health_field_inventory": {
                        "prognostic": list(state_fields),
                        "boundary_by_domain": {
                            name: list(fields)
                            for name, fields in boundary_fields_by_domain.items()
                        },
                        "absent_optional": absent_health_fields,
                    },
                    "output_alarm_schedules": {
                        name: list(values) for name, values in schedules.items()
                    },
                },
                "gate": {
                    "steps": GATE_STEPS,
                    "health": gate_health,
                    "V10_rescored": False,
                },
                "ordinary_first_red": {
                    "event_kind": event_kind,
                    "exception": caught,
                    "parent_domain": parent_domain,
                    "domain": domain,
                    "before_native_step": start_step,
                    "input_health": red["input_health"],
                    "output_health": red["output_health"],
                    "monitor_tail": red["monitor_tail"],
                    "expected_terminal_guard": EXPECTED_FAILURE,
                },
                "nest_force_replay": {
                    "source_operation": "_operational_force/build_child_boundary_package",
                    "manifest": replay_manifest,
                    "comparison_to_ordinary": comparison,
                    "admissible": admissible,
                },
                "recorder": None,
                "retained_exact_carries": retained,
                "segments": segments,
                "output_schedule_calls": [list(value) for value in output.calls],
                "hard_limits": EXACT_LIMIT_STEPS,
                "preemption_post": assert_preemption_clear("late-ni-exact-post"),
                "timing": {
                    "started_utc": started.isoformat(),
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "wall_seconds": time.perf_counter() - wall_started,
                },
            }
            proof["proof_sha256"] = canonical_digest(proof)
            atomic_json(run_dir / "exact-result.json", proof)
            print(json.dumps({
                "verdict": verdict,
                "proof": str((run_dir / "exact-result.json").resolve()),
                "proof_sha256": proof["proof_sha256"],
                "event_kind": event_kind,
                "edge": f"{parent_domain}->{domain}",
                "force_replay_identity": comparison["all_leaf_output_bytes_equal"],
                "first_bad": red["output_health"]["first_bad"],
            }, sort_keys=True, allow_nan=False), flush=True)
            return 0 if admissible else 4
        if event_kind != "advance":
            raise RuntimeError(f"unsupported first-red event kind: {event_kind}")
        if not 1 <= n_steps <= 3:
            raise RuntimeError(
                "ordinary first-red replay exceeds authorized native-step bound: "
                f"{n_steps}"
            )

        # Replay cumulative prefixes from the exact same dispatch input through
        # the accepted production entry.  The complete prefix is the same call
        # as the original red dispatch and must match every output byte.  Shorter
        # prefixes use the same traced-count executable and select the first red
        # native step without adding any record to the model HLO.
        prefix_values = []
        prefix_health = []
        for prefix_length in range(1, n_steps + 1):
            prefix_value = normal_advance(
                domain, input_carry, start_step, prefix_length
            )
            jax.block_until_ready(prefix_value.state.theta)
            prefix_values.append(prefix_value)
            prefix_health.append({
                "prefix_length": prefix_length,
                "native_start_step": start_step,
                "native_end_step": start_step + prefix_length - 1,
                "health": decode_health(
                    health_host(prefix_value, domain), prefix_value, domain
                ),
            })

        full_prefix_host = jax.device_get(prefix_values[-1])
        full_prefix_manifest = manifest(full_prefix_host)
        full_prefix_comparison = compare(
            ordinary_manifest, full_prefix_manifest
        )
        first_prefix_bad = next(
            (
                row
                for row in prefix_health
                if row["health"]["first_bad"] is not None
            ),
            None,
        )
        identity_admissible = bool(
            full_prefix_comparison["all_leaf_output_bytes_equal"]
        )
        admissible = bool(identity_admissible and first_prefix_bad is not None)
        if not identity_admissible:
            verdict = "LATE_NI_ORDINARY_REPLAY_IDENTITY_BLOCKED"
        elif first_prefix_bad is None:
            verdict = "LATE_NI_ORDINARY_PREFIX_HEALTH_INCONSISTENT"
        else:
            verdict = "LATE_NI_EXACT_FIRST_RED_DISPATCH_LOCALIZED"

        def persist_prefix_carry(label: str, value, native_step: int):
            host = jax.device_get(value)
            path = run_dir / f"{label}-{domain}-step-{native_step}.pkl"
            atomic_pickle(path, host)
            return {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "manifest": manifest(host),
            }

        if first_prefix_bad is not None:
            first_length = int(first_prefix_bad["prefix_length"])
            first_red_step = int(first_prefix_bad["native_end_step"])
            if first_length == 1:
                retained["ordinary_prefix_last_green"] = {
                    **retained["input"],
                    "alias_of": "input",
                }
            else:
                retained["ordinary_prefix_last_green"] = persist_prefix_carry(
                    "ordinary-prefix-last-green",
                    prefix_values[first_length - 2],
                    first_red_step - 1,
                )
            if first_length == n_steps and identity_admissible:
                retained["ordinary_prefix_first_red"] = {
                    **retained["ordinary_output"],
                    "alias_of": "ordinary_output",
                }
            else:
                retained["ordinary_prefix_first_red"] = persist_prefix_carry(
                    "ordinary-prefix-first-red",
                    prefix_values[first_length - 1],
                    first_red_step,
                )

        proof = {
            "schema": "gpuwrf.v0234.late-ni-exact-first-red-result.v1",
            "verdict": verdict,
            "authorization": authorization_row,
            "source_authority": source_authority,
            "environment": environment,
            "preemption_pre": preemption_pre,
            "fixture": {
                "run_start": run_start.isoformat(),
                "dt_by_domain": dt_by_domain,
                "loaded_options": options,
                "history_interval_minutes": history_minutes,
                "health_field_inventory": {
                    "prognostic": list(state_fields),
                    "boundary_by_domain": {
                        name: list(fields)
                        for name, fields in boundary_fields_by_domain.items()
                    },
                    "absent_optional": absent_health_fields,
                },
                "output_alarm_schedules": {
                    name: list(values) for name, values in schedules.items()
                },
                "metadata_options": {
                    name: {
                        "moist_adv_opt": metadata["domains"][name]["namelist"]["moist_adv_opt"],
                        "scalar_adv_opt": metadata["domains"][name]["namelist"]["scalar_adv_opt"],
                    }
                    for name in names
                },
            },
            "gate": {
                "steps": GATE_STEPS,
                "health": gate_health,
                "V10_rescored": False,
            },
            "ordinary_first_red": {
                "event_kind": event_kind,
                "exception": caught,
                "domain": domain,
                "native_start_step": start_step,
                "native_end_step": end_step,
                "n_steps": n_steps,
                "input_health": red["input_health"],
                "output_health": red["output_health"],
                "monitor_tail": red["monitor_tail"],
                "expected_terminal_guard": EXPECTED_FAILURE,
            },
            "ordinary_prefix_replay": {
                "callable": "prepared production runtime.advance",
                "method": "cumulative prefixes from the exact dispatch input",
                "recorder_payload_authorized": False,
                "health_by_prefix": prefix_health,
                "first_bad_prefix": first_prefix_bad,
                "full_prefix_manifest": full_prefix_manifest,
                "full_prefix_comparison_to_observed_dispatch": (
                    full_prefix_comparison
                ),
                "admissible": admissible,
            },
            "recorder": {
                "authorized": False,
                "executed": False,
                "reason": "authorized smoke proved RCA capture output-changing",
            },
            "retained_exact_carries": retained,
            "segments": segments,
            "output_schedule_calls": [list(value) for value in output.calls],
            "hard_limits": EXACT_LIMIT_STEPS,
            "preemption_post": assert_preemption_clear("late-ni-exact-post"),
            "timing": {
                "started_utc": started.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.perf_counter() - wall_started,
            },
        }
        proof["proof_sha256"] = canonical_digest(proof)
        atomic_json(run_dir / "exact-result.json", proof)
        print(json.dumps({
            "verdict": verdict,
            "proof": str((run_dir / "exact-result.json").resolve()),
            "proof_sha256": proof["proof_sha256"],
            "ordinary_full_prefix_identity": identity_admissible,
            "first_bad_prefix": first_prefix_bad,
            "recorder_executed": False,
        }, sort_keys=True, allow_nan=False), flush=True)
        return 0 if admissible else 4
    except BaseException as exc:
        blocker = {
            "schema": f"gpuwrf.v0234.late-ni-{arm}-blocker.v1",
            "verdict": "LATE_NI_GPU_HARNESS_OR_RUNTIME_BLOCKED",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "authorization": authorization_row,
            "source_authority": source_authority,
            "namespace": str(run_dir.resolve()),
            "started_utc": started.isoformat(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - wall_started,
            "scientific_falsification": False,
        }
        blocker["proof_sha256"] = canonical_digest(blocker)
        atomic_json(run_dir / f"{arm}-blocker.json", blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--proof-output", type=Path)
    parser.add_argument("--arm", choices=("smoke", "exact"))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.cpu_preflight:
        if args.proof_output is None or any(
            value is not None for value in (args.arm, args.authorization, args.run_dir)
        ):
            parser.error("--cpu-preflight requires only --proof-output")
        return cpu_preflight(args.proof_output.resolve())
    if args.proof_output is not None or any(
        value is None for value in (args.arm, args.authorization, args.run_dir)
    ):
        parser.error("GPU mode requires --arm/--authorization/--run-dir")
    return _gpu_run(args.authorization.resolve(), args.run_dir.resolve(), arm=args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
