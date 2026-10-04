#!/usr/bin/env python3
"""One-prefix, one-recorder-compile corrected-d03 Ni RCA replay.

The runner is intentionally proof-only.  It authenticates the corrected terminal
inputs and Retry20 provenance, recreates the exact live three-domain scheduler to
d03 step 9198, and replays steps 9199..9405 from the same resident carry through
three arms:

* canonical production chunking;
* recorder-OFF with the d03 child split into one-step calls;
* recorder-ON with the same one-step split and bounded on-device summaries.

All materialization happens after a replay boundary.  No callback or host transfer
is present in the timestep program.  If either carry-byte identity comparison
fails, the recording evidence is marked inadmissible and cannot authorize a fix.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


REQUIRED_BACKEND = "cuda"
REQUIRED_PREIMPORT_ENV = {
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
    "GPUWRF_BATCH_ENSEMBLE": "1",
    "GPUWRF_NESTED_SYNC_MODE": "root",
    "GPUWRF_BITWISE": "1",
    "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
    "GPUWRF_JAX_CACHE": "1",
    "GPUWRF_JAX_CACHE_LOCK": "1",
}


def validate_preimport_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    actual = {name: environment.get(name) for name in REQUIRED_PREIMPORT_ENV}
    if actual != REQUIRED_PREIMPORT_ENV:
        raise RuntimeError(
            "pre-import runtime environment differs from the frozen RCA lane: "
            f"expected={REQUIRED_PREIMPORT_ENV!r} actual={actual!r}"
        )
    cache = environment.get("GPUWRF_JAX_CACHE_DIR")
    compilation_cache = environment.get("JAX_COMPILATION_CACHE_DIR")
    if not cache or cache != compilation_cache:
        raise RuntimeError("JAX cache paths must be equal, explicit, and non-empty")
    wrf_root = environment.get("GPUWRF_WRF_ROOT")
    if not wrf_root:
        raise RuntimeError("GPUWRF_WRF_ROOT must bind the retained authority snapshot")
    return {
        "required_backend": REQUIRED_BACKEND,
        "validated_before_jax_import": True,
        "required_environment": actual,
        "cache_dir": cache,
        "wrf_root": wrf_root,
    }


if __name__ == "__main__":
    _PREIMPORT_AUTHORITY = validate_preimport_environment(os.environ)


import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.nested_pipeline import (
    NestedPipelineConfig,
    _load_domains,
    _nested_sync_mode_from_env,
    _output_alarm_steps_by_domain,
    _output_cadence_steps_by_domain,
    domain_names_for,
)
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.profiling.transfer_audit import block_until_ready, count_transfer_bytes
from gpuwrf.runtime.domain_tree import (
    DomainTree,
    _operational_advance_factory,
    _operational_force,
    run_domain_tree_callbacks,
    run_operational_domain_tree,
)
from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
from gpuwrf.runtime.operational_mode import (
    RCA_ACOUSTIC_FIELDS,
    RCA_BOUNDARY_FIELDS,
    RCA_HEALTH_METRICS,
    RCA_STATE_FIELDS,
    RCA_STATE_PHASES,
    RCA_TARGET_K,
    RCA_TARGET_X,
    RCA_TARGET_Y,
    advance_chunk_with_corrected_ni_rca,
    build_clock_base,
)


SCHEMA = "gpuwrf.v0234.corrected-ni-rca-short-replay.v1"
NUMERICAL_BASE_SHA = "bb2ffe33dbff5e04246dc6903bf6aa26e5827163"
RUNTIME_PARENT_SHA = "16774bed70b26d465e3aa87e91060e6a17de7f92"
TERMINAL_CONTRACT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "terminal_cpu_authority_contract_v1.json"
)
TERMINAL_CONTRACT_SHA256 = "26b16e21782400308ec4d58175907796bf8aaddb01062bb06be8b87a2a88848a"
INPUT_DIR = TERMINAL_CONTRACT.parent / "run/wrf"
INPUT_HASHES = {
    "namelist.input": "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}
RETRY20_ROOT = TERMINAL_CONTRACT.parent / "gpu_validation_retry20_relative_rmse_3ee02c19"
RETRY20_ATTESTATION = RETRY20_ROOT / "gpu-proof/retry20-runtime-source-attestation.json"
RETRY20_FINAL_ACCEPT = RETRY20_ROOT / "retry20-final-accept.json"
RETRY20_CACHE_AUTHORITY = RETRY20_ROOT / "retry20-cache-source-authority.json"
EXPECTED_RETRY20_RUNTIME_SHA = RUNTIME_PARENT_SHA
EXPECTED_RETRY20_COUNTS = {"d01": 16, "d02": 16, "d03": 47}
EXPECTED_LAST_D03 = "wrfout_d03_2025-03-01_15:20:00"
LOCK_ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-gpu-lock-v2")
LOCK_COMMIT = "8152309aff1e85e1052d44d549a5a5409e710bdd"
LOCK_WRAPPER = LOCK_ROOT / "scripts/with_gpu_lock.sh"
LOCK_VERIFIER = LOCK_ROOT / "scripts/gpu_lock_v2.py"
LOCK_WRAPPER_SHA256 = "c75b3a4eda17e94df921986e1c15fcc71e182e01d51b6fe877e4c52533077e1a"
LOCK_VERIFIER_SHA256 = "ec911f565ee9d2c58dee81d8d1e17411ed677be500a57e73ded5f793e2c4187d"
PREEMPT_PATHS = (
    Path("/tmp/PREEMPT_GPU"),
    Path("/tmp/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_PRODUCTION"),
    Path("<DATA_ROOT>/alisios/state/PREEMPT_GPU"),
)
NIGHTLY_ACTIVE = Path("<DATA_ROOT>/alisios/state/nightly18z/active.json")
PREFIX_ROOT_STEPS = 1022
PREFIX_OWN_STEPS = {"d01": 1022, "d02": 3066, "d03": 9198}
REPLAY_ROOT_STEPS = 23
REPLAY_OWN_STEPS = {"d01": 1045, "d02": 3135, "d03": 9405}
PROMPT_PATH = Path("/tmp/v0234_corrected_ni_rca_max_prompt.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
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


def json_number(value: Any) -> int | float | str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Infinity" if number > 0 else "-Infinity"
    integer = int(number)
    if float(integer) == number:
        return integer
    return number


def assert_candidate_authority() -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    approved = os.environ.get("GPUWRF_CORRECTED_NI_RCA_APPROVED_SHA")
    if approved != head:
        raise RuntimeError(
            "GPUWRF_CORRECTED_NI_RCA_APPROVED_SHA must equal the exact committed HEAD"
        )
    if git_output("status", "--porcelain"):
        raise RuntimeError("RCA replay requires a clean committed worktree at launch")
    for ancestor in (NUMERICAL_BASE_SHA, RUNTIME_PARENT_SHA):
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", ancestor, head), check=True,
        )
    source_rows = []
    for relative in git_output("ls-files", "src/gpuwrf").splitlines():
        source_rows.append((relative, sha256_file(Path(relative))))
    return {
        "head": head,
        "approved_sha": approved,
        "numerical_base_sha": NUMERICAL_BASE_SHA,
        "runtime_parent_sha": RUNTIME_PARENT_SHA,
        "tracked_source_manifest_sha256": canonical_digest(source_rows),
        "worktree_clean": True,
    }


def assert_input_authority() -> dict[str, Any]:
    contract_digest = sha256_file(TERMINAL_CONTRACT)
    if contract_digest != TERMINAL_CONTRACT_SHA256:
        raise RuntimeError("terminal CPU authority contract digest changed")
    rows: dict[str, Any] = {}
    for name, expected in INPUT_HASHES.items():
        path = INPUT_DIR / name
        stat_result = path.stat()
        digest = sha256_file(path)
        if digest != expected or stat_result.st_mode & 0o222:
            raise RuntimeError(f"corrected input authority mismatch: {path}")
        rows[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "bytes": stat_result.st_size,
            "mode": oct(stat_result.st_mode & 0o777),
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
        }
    return {
        "terminal_contract": {
            "path": str(TERMINAL_CONTRACT), "sha256": contract_digest,
        },
        "inputs": rows,
        "authority_sha256": canonical_digest(rows),
    }


def _retry20_output_inventory() -> dict[str, Any]:
    output = RETRY20_ROOT / "gpu-output"
    inventory: dict[str, Any] = {}
    for domain in ("d01", "d02", "d03"):
        paths = sorted(output.glob(f"wrfout_{domain}_*"))
        if len(paths) != EXPECTED_RETRY20_COUNTS[domain]:
            raise RuntimeError(f"Retry20 retained {domain} frame count changed")
        inventory[domain] = {
            "count": len(paths),
            "first": paths[0].name,
            "last": paths[-1].name,
            "last_sha256": sha256_file(paths[-1]),
        }
    if inventory["d03"]["last"] != EXPECTED_LAST_D03:
        raise RuntimeError("Retry20 last retained d03 frame changed")
    return inventory


def assert_retry20_authority() -> dict[str, Any]:
    attestation = json.loads(RETRY20_ATTESTATION.read_text())
    final_accept = json.loads(RETRY20_FINAL_ACCEPT.read_text())
    cache_authority = json.loads(RETRY20_CACHE_AUTHORITY.read_text())
    runtime = attestation.get("runtime_source") or {}
    runtime_sha = runtime.get("commit") or runtime.get("head") or runtime.get("sha")
    if runtime_sha != EXPECTED_RETRY20_RUNTIME_SHA:
        raise RuntimeError(f"Retry20 runtime source changed: {runtime_sha!r}")
    if final_accept.get("runtime_source_commit") != EXPECTED_RETRY20_RUNTIME_SHA:
        raise RuntimeError("Retry20 final acceptance runtime binding changed")
    source_child = Path(str(cache_authority.get("source_keyed_child", {}).get("path", "")))
    cache_dir = Path(str(cache_authority.get("aot", {}).get("jax_cache_dir", "")))
    if not source_child.is_dir() or not cache_dir.is_dir():
        raise RuntimeError("Retry20 retained cache authority is unavailable")
    artifacts = {}
    for path in (RETRY20_ATTESTATION, RETRY20_FINAL_ACCEPT, RETRY20_CACHE_AUTHORITY):
        artifacts[path.name] = {
            "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
        }
    prompt = {
        "path": str(PROMPT_PATH),
        "sha256": sha256_file(PROMPT_PATH),
        "bytes": PROMPT_PATH.stat().st_size,
    }
    return {
        "artifacts": artifacts,
        "runtime_source_commit": runtime_sha,
        "numerical_candidate_sha": attestation.get("numerical_candidate_sha"),
        "runtime_delta_from_numerical": attestation.get("runtime_delta_from_numerical"),
        "retained_outputs": _retry20_output_inventory(),
        "source_keyed_cache_child": str(source_child),
        "source_jax_cache": str(cache_dir),
        "manager_failure_contract": {
            "prompt": prompt,
            "d03_failure": {
                "sim_time_s": 56400,
                "native_step": 9400,
                "field": "Ni",
                "level": 1,
                "first_index": [1, 48, 78],
                "model_rc": 1,
                "launcher_status": 75,
            },
        },
    }


def assert_preemption_clear(label: str) -> dict[str, Any]:
    present = [str(path) for path in PREEMPT_PATHS if path.exists() or path.is_symlink()]
    if present:
        raise RuntimeError(f"{label}: production preemption sentinel present: {present}")
    nightly: dict[str, Any] | None = None
    if NIGHTLY_ACTIVE.exists():
        nightly = json.loads(NIGHTLY_ACTIVE.read_text())
        if nightly.get("active") is True:
            raise RuntimeError(f"{label}: nightly production is active")
    return {
        "label": label,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "sentinels_absent": [str(path) for path in PREEMPT_PATHS],
        "nightly_active": None if nightly is None else nightly.get("active"),
        "nightly_status": None if nightly is None else nightly.get("status"),
    }


def assert_lock_authority() -> dict[str, Any]:
    commit = subprocess.check_output(
        ("git", "-C", str(LOCK_ROOT), "rev-parse", "HEAD"), text=True,
    ).strip()
    wrapper_digest = sha256_file(LOCK_WRAPPER)
    verifier_digest = sha256_file(LOCK_VERIFIER)
    if (
        commit != LOCK_COMMIT
        or wrapper_digest != LOCK_WRAPPER_SHA256
        or verifier_digest != LOCK_VERIFIER_SHA256
    ):
        raise RuntimeError("accepted GPU lock-v2 authority changed")
    required = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": "9",
        "GPUWRF_GPU_LOCK_FILE": "/tmp/wrf_gpu2_gpu.lock",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": "/tmp/wrf_gpu2_gpu.lock.holder",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"live GPU lock environment mismatch: {name}")
    token = os.environ.get("GPUWRF_GPU_LOCK_TOKEN", "")
    label = os.environ.get("GPUWRF_GPU_LOCK_LABEL", "")
    if not token or not label:
        raise RuntimeError("live GPU lock token/label missing")
    spec = importlib.util.spec_from_file_location("v0234_gpu_lock_v2", LOCK_VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load accepted GPU lock-v2 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock_path = Path(required["GPUWRF_GPU_LOCK_FILE"])
    lease = module._read_sidecar(lock_path)
    module._validate_schema(dict(lease))
    if lease.get("intent") != "production-preemptible":
        raise RuntimeError("live GPU lease is not production-preemptible")
    if lease.get("lease_id") != token or lease.get("label") != label:
        raise RuntimeError("live GPU lease differs from payload environment")
    module._validate_lock_lease_fd(lock_path, 9, lease)
    probe_fd = module._open_gpu_lock(lock_path)
    try:
        module._validate_live_lease(lock_path, probe_fd, expected=lease)
    finally:
        os.close(probe_fd)
    return {
        "authority_root": str(LOCK_ROOT),
        "commit": commit,
        "wrapper_sha256": wrapper_digest,
        "verifier_sha256": verifier_digest,
        "live_verified": True,
        "intent": lease["intent"],
        "schema": lease["schema"],
        "version": lease["version"],
        "label": label,
        "lease_id_sha256": sha256_text(token),
    }


def assert_cuda_runtime() -> dict[str, Any]:
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"RCA lane is not CUDA-only: {devices!r}")
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
    }


class _ScheduleOnlyOutput:
    """Preserve production output splits without materializing device state."""

    wants_carry = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def __call__(self, name: str, step: int, _carry: Any) -> tuple[str, int]:
        self.calls.append((name, int(step)))
        return name, int(step)


def scheduler_contract(
    names: tuple[str, ...],
    dt_by_domain: dict[str, float],
    *,
    total_steps: dict[str, int],
) -> tuple[dict[str, int], dict[str, tuple[int, ...]], dict[str, Any]]:
    run = Gen2Run(INPUT_DIR)
    cadence, minutes = _output_cadence_steps_by_domain(run, names, dt_by_domain)
    schedules, nonintegral = _output_alarm_steps_by_domain(
        names, dt_by_domain, minutes, total_steps,
    )
    return cadence, nonintegral, {
        "dt_by_domain": dt_by_domain,
        "history_interval_minutes": minutes,
        "output_cadence_steps": cadence,
        "output_alarm_schedule": {name: list(values) for name, values in schedules.items()},
        "nonintegral_output_alarms": {
            name: list(values) for name, values in nonintegral.items()
        },
    }


def load_corrected_tree(work_dir: Path) -> tuple[
    DomainTree,
    tuple[str, ...],
    dict[str, Any],
    dict[str, float],
    dict[str, Any],
]:
    names = domain_names_for(3)
    runtime_root = work_dir / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=False)
    config = NestedPipelineConfig(
        input_dir=INPUT_DIR,
        output_dir=runtime_root / "unused-output",
        proof_dir=runtime_root / "unused-pipeline-proof",
        hours=0,
        max_dom=3,
        feedback=False,
    )
    hierarchy, bundles, load_meta, run_start, dt_by_domain, carries = _load_domains(
        config, names,
    )
    if run_start.isoformat() != "2025-03-01T00:00:00+00:00":
        raise RuntimeError(f"corrected input start changed: {run_start.isoformat()}")
    if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise RuntimeError(f"corrected timestep hierarchy changed: {dt_by_domain!r}")
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    return tree, names, carries, dt_by_domain, {
        "run_start": run_start.isoformat(),
        "load_metadata": load_meta,
        "domain_order": list(hierarchy.order),
        "nests": [edge.__dict__ for edge in hierarchy.nests],
    }


def _event_summary(result: Any, output: _ScheduleOnlyOutput) -> dict[str, Any]:
    counts = Counter(event[0] for event in result.events)
    forces = Counter(
        f"{event[1]}->{event[2]}"
        for event in result.events
        if event and event[0] == "force"
    )
    return {
        "own_steps": dict(result.own_steps),
        "event_counts": dict(sorted(counts.items())),
        "force_counts": dict(sorted(forces.items())),
        "schedule_only_output_calls": [list(value) for value in output.calls],
        "last_event": list(result.events[-1]) if result.events else None,
    }


def run_prefix(
    tree: DomainTree,
    names: tuple[str, ...],
    initial_carries: dict[str, Any],
    dt_by_domain: dict[str, float],
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any], dict[str, int], dict[str, tuple[int, ...]]]:
    cadence, nonintegral, schedule = scheduler_contract(
        names, dt_by_domain, total_steps=PREFIX_OWN_STEPS,
    )
    block_between, root_sync_cadence = _nested_sync_mode_from_env()
    output = _ScheduleOnlyOutput()
    carries = initial_carries
    own_steps = {name: 0 for name in names}
    aggregate_events: Counter[str] = Counter()
    aggregate_forces: Counter[str] = Counter()
    segments = (67,) * 15 + (17,)
    segment_rows: list[dict[str, Any]] = []
    for index, steps in enumerate(segments):
        assert_preemption_clear(f"prefix-segment-{index}-pre")
        result = run_operational_domain_tree(
            tree,
            root_steps=steps,
            feedback_enabled=False,
            output=output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            carries=carries,
            initial_own_steps=own_steps,
        )
        block_until_ready(tuple(result.states[name].theta for name in names))
        for name in names:
            step = int(result.own_steps[name])
            assert_state_finite_at_boundary(
                result.states[name],
                domain=name,
                step=step,
                sim_time_s=step * float(dt_by_domain[name]),
            )
        segment_events = Counter(event[0] for event in result.events)
        segment_forces = Counter(
            f"{event[1]}->{event[2]}"
            for event in result.events
            if event and event[0] == "force"
        )
        aggregate_events.update(segment_events)
        aggregate_forces.update(segment_forces)
        carries = result.carries
        own_steps = dict(result.own_steps)
        segment_rows.append({
            "index": index,
            "root_steps": steps,
            "own_steps": own_steps,
            "event_counts": dict(sorted(segment_events.items())),
        })
        print(
            f"RCA_PREFIX segment={index + 1}/{len(segments)} own_steps={own_steps}",
            flush=True,
        )
    if own_steps != PREFIX_OWN_STEPS:
        raise RuntimeError(f"prefix stopped at wrong clocks: {own_steps!r}")
    return carries, own_steps, {
        "segments": segment_rows,
        "event_counts": dict(sorted(aggregate_events.items())),
        "force_counts": dict(sorted(aggregate_forces.items())),
        "schedule_only_output_calls": [list(value) for value in output.calls],
        "scheduler": schedule,
        "block_between": block_between,
        "root_sync_cadence": root_sync_cadence,
        "finite_at_boundary": True,
    }, cadence, nonintegral


def _edge_lookup(tree: DomainTree):
    edges = {
        (edge.parent, edge.child): edge
        for values in tree.edges.values()
        for edge in values
    }

    def lookup(spec):
        return edges[(spec.parent, spec.child)]

    return lookup


def run_replay(
    tree: DomainTree,
    prefix_carries: dict[str, Any],
    prefix_steps: dict[str, int],
    *,
    cadence: dict[str, int],
    nonintegral: dict[str, tuple[int, ...]],
    mode: str,
    records: list[Any] | None = None,
    recorder_executable: Any | None = None,
    recorder_step_arrays: Mapping[int, Any] | None = None,
) -> Any:
    if mode not in {"canonical", "off", "on"}:
        raise ValueError(mode)
    output = _ScheduleOnlyOutput()
    block_between, root_sync_cadence = _nested_sync_mode_from_env()
    if mode == "canonical":
        result = run_operational_domain_tree(
            tree,
            root_steps=REPLAY_ROOT_STEPS,
            feedback_enabled=False,
            output=output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            carries=prefix_carries,
            initial_own_steps=prefix_steps,
        )
    else:
        normal_advance = _operational_advance_factory(tree)
        d03_namelist = tree.domains["d03"].namelist
        d03_clock = build_clock_base(d03_namelist)

        def advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
            if name != "d03":
                return normal_advance(name, carry, start_step, n_steps)
            value = carry
            for offset in range(int(n_steps)):
                step_number = int(start_step + offset)
                native_step = (
                    recorder_step_arrays[step_number]
                    if recorder_step_arrays is not None
                    else jnp.asarray(step_number, dtype=jnp.int32)
                )
                if mode == "off":
                    value = normal_advance(name, value, step_number, 1)
                else:
                    if recorder_executable is None:
                        raise RuntimeError("recorder executable was not precompiled")
                    observed = recorder_executable(
                        value,
                        d03_namelist,
                        native_step,
                        d03_clock,
                    )
                    value = observed.carry
                    assert records is not None
                    records.append(observed.records)
            return value

        result = run_domain_tree_callbacks(
            tree.hierarchy,
            prefix_carries,
            root_steps=REPLAY_ROOT_STEPS,
            advance=advance,
            force=_operational_force,
            feedback=None,
            feedback_enabled=False,
            output=output,
            output_cadence_steps=cadence,
            output_alarm_steps=nonintegral,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            edge_lookup=_edge_lookup(tree),
            fused_cascade=None,
            initial_own_steps=prefix_steps,
        )
    block_until_ready(tuple(result.states[name].theta for name in tree.hierarchy.order))
    if dict(result.own_steps) != REPLAY_OWN_STEPS:
        raise RuntimeError(f"{mode} replay stopped at wrong clocks: {result.own_steps!r}")
    return result, _event_summary(result, output)


def host_tree_manifest(value: Any) -> dict[str, Any]:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(value)
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    floating_nonfinite = 0
    for path, leaf in path_leaves:
        array = np.asarray(jax.device_get(leaf))
        contiguous = np.ascontiguousarray(array)
        raw = contiguous.tobytes(order="C")
        total_bytes += len(raw)
        nonfinite = 0
        if np.issubdtype(contiguous.dtype, np.floating):
            nonfinite = int(np.count_nonzero(~np.isfinite(contiguous)))
            floating_nonfinite += nonfinite
        rows.append({
            "path": jax.tree_util.keystr(path),
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "nonfinite_count": nonfinite,
        })
    identity_rows = [
        {key: row[key] for key in ("path", "shape", "dtype", "bytes", "sha256")}
        for row in rows
    ]
    return {
        "treedef": str(treedef),
        "leaf_count": len(rows),
        "total_bytes": total_bytes,
        "floating_nonfinite_count": floating_nonfinite,
        "leaves": rows,
        "manifest_sha256": canonical_digest({
            "treedef": str(treedef), "leaves": identity_rows,
        }),
    }


def compare_manifests(canonical: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    canonical_rows = {row["path"]: row for row in canonical["leaves"]}
    candidate_rows = {row["path"]: row for row in candidate["leaves"]}
    differences: list[dict[str, Any]] = []
    for path in sorted(set(canonical_rows) | set(candidate_rows)):
        left = canonical_rows.get(path)
        right = candidate_rows.get(path)
        if left != right:
            differences.append({"path": path, "canonical": left, "candidate": right})
    equal = (
        canonical.get("treedef") == candidate.get("treedef")
        and canonical.get("manifest_sha256") == candidate.get("manifest_sha256")
        and not differences
    )
    return {
        "all_leaf_output_bytes_equal": equal,
        "canonical_manifest_sha256": canonical.get("manifest_sha256"),
        "candidate_manifest_sha256": candidate.get("manifest_sha256"),
        "different_leaf_count": len(differences),
        "first_differences": differences[:20],
    }


def compile_and_audit_recorder(
    tree: DomainTree, prefix_carries: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    namelist = tree.domains["d03"].namelist
    clock = build_clock_base(namelist)
    lowered = advance_chunk_with_corrected_ni_rca.lower(
        prefix_carries["d03"],
        namelist,
        jnp.asarray(PREFIX_OWN_STEPS["d03"] + 1, dtype=jnp.int32),
        clock,
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    callback_tokens = tuple(
        token
        for token in (
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
        )
        if token in stablehlo.lower()
    )
    compiled = lowered.compile()
    return {
        "lowered_once": True,
        "compiled_once": True,
        "n_steps_per_dispatch": 1,
        "one_static_executable_reused_for_all_207_d03_steps": True,
        "stablehlo_sha256": sha256_text(stablehlo),
        "stablehlo_bytes": len(stablehlo.encode()),
        "callback_tokens_found": list(callback_tokens),
        "callback_free": not callback_tokens,
        "record_schema": {
            "health_metrics": list(RCA_HEALTH_METRICS),
            "acoustic_fields": list(RCA_ACOUSTIC_FIELDS),
            "state_fields": list(RCA_STATE_FIELDS),
            "state_phases": list(RCA_STATE_PHASES),
            "boundary_fields": list(RCA_BOUNDARY_FIELDS),
            "target_python_index": [RCA_TARGET_K, RCA_TARGET_Y, RCA_TARGET_X],
            "target_wrf_index": [RCA_TARGET_K + 1, RCA_TARGET_Y + 1, RCA_TARGET_X + 1],
        },
    }, compiled


def state_shape_map(state: Any) -> dict[str, list[int]]:
    return {
        name: list(np.shape(getattr(state, name)))
        for name in RCA_STATE_FIELDS
    }


def boundary_shape_map(state: Any) -> dict[str, list[int]]:
    return {
        name: list(np.shape(getattr(state, name)))
        for name in RCA_BOUNDARY_FIELDS
    }


def acoustic_shape_map(state: Any) -> dict[str, list[int]]:
    mapping = {
        "pre_uv_u": state.u,
        "pre_uv_v": state.v,
        "large_u_tend": state.u,
        "large_v_tend": state.v,
        "pre_p": state.p_total,
        "pre_al": state.theta,
        "pre_ph": state.ph_total,
        "small_dpx": state.u,
        "small_dpy": state.v,
        "uv_u": state.u,
        "uv_v": state.v,
        "raw_dvdxi": state.theta,
        "raw_dmdt": state.mu_total,
        "raw_mu_tendency": state.mu_total,
        "mu_scale": state.mu_total,
        "limited_dvdxi": state.theta,
        "limited_dmdt": state.mu_total,
        "output_mudf": state.mu_total,
        "new_muts": state.mu_total,
        "post_w": state.w,
        "post_ph": state.ph_total,
        "post_p": state.p_total,
        "post_al": state.theta,
        "post_theta": state.theta,
        "post_u": state.u,
        "post_v": state.v,
        "post_mu": state.mu_total,
        "post_ww": state.w,
    }
    return {name: list(np.shape(mapping[name])) for name in RCA_ACOUSTIC_FIELDS}


def materialize_records(records: list[Any]) -> dict[str, np.ndarray]:
    if len(records) != REPLAY_OWN_STEPS["d03"] - PREFIX_OWN_STEPS["d03"]:
        raise RuntimeError(f"wrong recorder step count: {len(records)}")
    host = [jax.device_get(record) for record in records]
    return {
        "acoustic_health": np.concatenate(
            [np.asarray(record.acoustic_health) for record in host], axis=0,
        ),
        "acoustic_target": np.concatenate(
            [np.asarray(record.acoustic_target) for record in host], axis=0,
        ),
        "state_health": np.concatenate(
            [np.asarray(record.state_health) for record in host], axis=0,
        ),
        "state_target": np.concatenate(
            [np.asarray(record.state_target) for record in host], axis=0,
        ),
        "boundary_health": np.concatenate(
            [np.asarray(record.boundary_health) for record in host], axis=0,
        ),
    }


def decode_flat_index(flat_index: int, shape: list[int]) -> dict[str, Any]:
    if flat_index < 0:
        return {"flat_index": flat_index, "python_index": None, "wrf_index": None}
    python_index = tuple(int(value) for value in np.unravel_index(flat_index, tuple(shape)))
    return {
        "flat_index": flat_index,
        "python_index": list(python_index),
        "wrf_index": [value + 1 for value in python_index],
    }


def health_detail(
    health: np.ndarray,
    *,
    shape: list[int],
) -> dict[str, Any]:
    return {
        "nonfinite_count": int(health[0]),
        "first_nonfinite": decode_flat_index(int(health[1]), shape),
        "max_abs_finite": json_number(health[2]),
        "max_abs": decode_flat_index(int(health[3]), shape),
        "value_at_max_abs": json_number(health[4]),
        "min_finite": json_number(health[5]),
        "min_finite_location": decode_flat_index(int(health[6]), shape),
        "max_finite": json_number(health[7]),
    }


def acoustic_substep_label(index: int) -> dict[str, int]:
    if index == 0:
        return {"rk_stage": 1, "acoustic_substep": 1}
    if index <= 5:
        return {"rk_stage": 2, "acoustic_substep": index}
    return {"rk_stage": 3, "acoustic_substep": index - 5}


def _acoustic_operator(field: str) -> str:
    if field in {
        "pre_uv_u", "pre_uv_v", "large_u_tend", "large_v_tend",
        "pre_p", "pre_al", "pre_ph",
    }:
        return "advance_uv_inputs"
    if field in {"small_dpx", "small_dpy", "uv_u", "uv_v"}:
        return "post_advance_uv"
    if field in {
        "raw_dvdxi", "raw_dmdt", "raw_mu_tendency", "mu_scale",
        "limited_dvdxi", "limited_dmdt", "output_mudf", "new_muts",
    }:
        return "advance_mu_t"
    return "post_acoustic_substep"


def decode_records(
    arrays: Mapping[str, np.ndarray],
    *,
    state_shapes: Mapping[str, list[int]],
    boundary_shapes: Mapping[str, list[int]],
    acoustic_shapes: Mapping[str, list[int]],
    domain: str = "d03",
    dt_s: float = 6.0,
    prefix_step: int | None = None,
    replay_step: int | None = None,
) -> dict[str, Any]:
    acoustic_health = arrays["acoustic_health"]
    acoustic_target = arrays["acoustic_target"]
    state_health = arrays["state_health"]
    state_target = arrays["state_target"]
    boundary_health = arrays["boundary_health"]
    start_step = int(PREFIX_OWN_STEPS[domain] if prefix_step is None else prefix_step)
    end_step = int(REPLAY_OWN_STEPS[domain] if replay_step is None else replay_step)
    record_steps = end_step - start_step
    if record_steps <= 0:
        raise RuntimeError(
            f"invalid recorder window: domain={domain} "
            f"prefix={start_step} replay={end_step}"
        )
    expected = {
        "acoustic_health": (record_steps, 16, len(RCA_ACOUSTIC_FIELDS), 8),
        "acoustic_target": (record_steps, 16, len(RCA_ACOUSTIC_FIELDS)),
        "state_health": (
            record_steps, len(RCA_STATE_PHASES), len(RCA_STATE_FIELDS), 8
        ),
        "state_target": (
            record_steps, len(RCA_STATE_PHASES), len(RCA_STATE_FIELDS)
        ),
        "boundary_health": (record_steps, len(RCA_BOUNDARY_FIELDS), 8),
    }
    actual = {name: tuple(value.shape) for name, value in arrays.items()}
    if actual != expected:
        raise RuntimeError(f"recorder schema mismatch: expected={expected!r} actual={actual!r}")

    state_first: list[dict[str, Any]] = []
    acoustic_first: list[dict[str, Any]] = []
    boundary_first: list[dict[str, Any]] = []
    for step_offset in range(record_steps):
        native_step = start_step + step_offset + 1
        for phase_index, phase in enumerate(RCA_STATE_PHASES):
            for field_index, field in enumerate(RCA_STATE_FIELDS):
                health = state_health[step_offset, phase_index, field_index]
                if health[0] > 0:
                    state_first.append({
                        "native_step": native_step,
                        "sim_time_s": native_step * float(dt_s),
                        "phase": phase,
                        "field": field,
                        "target_value": json_number(
                            state_target[step_offset, phase_index, field_index]
                        ),
                        "health": health_detail(health, shape=state_shapes[field]),
                    })
        for substep_index in range(16):
            for field_index, field in enumerate(RCA_ACOUSTIC_FIELDS):
                health = acoustic_health[step_offset, substep_index, field_index]
                if health[0] > 0:
                    acoustic_first.append({
                        "native_step": native_step,
                        "sim_time_s": native_step * float(dt_s),
                        **acoustic_substep_label(substep_index),
                        "operator_phase": _acoustic_operator(field),
                        "field": field,
                        "target_value": json_number(
                            acoustic_target[step_offset, substep_index, field_index]
                        ),
                        "health": health_detail(health, shape=acoustic_shapes[field]),
                    })
        for field_index, field in enumerate(RCA_BOUNDARY_FIELDS):
            health = boundary_health[step_offset, field_index]
            if health[0] > 0:
                boundary_first.append({
                    "native_step": native_step,
                        "sim_time_s": native_step * float(dt_s),
                    "field": field,
                    "health": health_detail(health, shape=boundary_shapes[field]),
                })

    phase_rank = {name: index for index, name in enumerate(RCA_STATE_PHASES)}
    operator_rank = {
        "advance_uv_inputs": 0,
        "post_advance_uv": 1,
        "advance_mu_t": 2,
        "post_acoustic_substep": 3,
    }
    state_first.sort(key=lambda row: (
        row["native_step"], phase_rank[row["phase"]], RCA_STATE_FIELDS.index(row["field"]),
    ))
    acoustic_first.sort(key=lambda row: (
        row["native_step"], row["rk_stage"], row["acoustic_substep"],
        operator_rank[row["operator_phase"]], RCA_ACOUSTIC_FIELDS.index(row["field"]),
    ))
    boundary_first.sort(key=lambda row: (
        row["native_step"], RCA_BOUNDARY_FIELDS.index(row["field"]),
    ))

    timeline_candidates: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for row in state_first:
        phase = phase_rank[row["phase"]]
        # Entry and physics occur before acoustic; later phases occur after it.
        within = phase if phase <= 1 else 100 + phase
        timeline_candidates.append(((row["native_step"], within, 0, 0), row))
    for row in acoustic_first:
        linear_substep = (
            0 if row["rk_stage"] == 1
            else row["acoustic_substep"] if row["rk_stage"] == 2
            else 5 + row["acoustic_substep"]
        )
        timeline_candidates.append((
            (
                row["native_step"], 10 + linear_substep,
                operator_rank[row["operator_phase"]],
                RCA_ACOUSTIC_FIELDS.index(row["field"]),
            ),
            row,
        ))
    earliest = min(timeline_candidates, key=lambda item: item[0])[1] if timeline_candidates else None

    scale_index = RCA_ACOUSTIC_FIELDS.index("mu_scale")
    limiter: dict[str, Any] | None = None
    for step_offset in range(record_steps):
        for substep_index in range(16):
            health = acoustic_health[step_offset, substep_index, scale_index]
            if health[5] < 1.0:
                limiter = {
                    "native_step": start_step + step_offset + 1,
                    **acoustic_substep_label(substep_index),
                    "minimum_scale": json_number(health[5]),
                    "location": decode_flat_index(
                        int(health[6]), acoustic_shapes["mu_scale"],
                    ),
                    "target_scale": json_number(
                        acoustic_target[step_offset, substep_index, scale_index]
                    ),
                }
                break
        if limiter is not None:
            break

    raw_index = RCA_ACOUSTIC_FIELDS.index("raw_mu_tendency")
    output_index = RCA_ACOUSTIC_FIELDS.index("output_mudf")
    closure = (
        acoustic_target[:, :, raw_index] * acoustic_target[:, :, scale_index]
        - acoustic_target[:, :, output_index]
    )
    finite_closure = closure[np.isfinite(closure)]
    closure_max = float(np.max(np.abs(finite_closure))) if finite_closure.size else math.nan

    default_center = (
        9400 if prefix_step is None and replay_step is None else end_step
    )
    center_step = default_center if earliest is None else int(earliest["native_step"])
    first_window = max(start_step + 1, center_step - 3)
    last_window = min(end_step, center_step + 3)
    selected_state_fields = (
        "Ni", "Nr", "qc", "qr", "qi", "qs", "qg", "theta", "qv",
        "p_total", "ph_total", "mu_total", "u", "v", "w",
    )
    state_window = []
    for native_step in range(first_window, last_window + 1):
        offset = native_step - start_step - 1
        phases: dict[str, Any] = {}
        for phase_index, phase in enumerate(RCA_STATE_PHASES):
            phases[phase] = {
                field: json_number(
                    state_target[offset, phase_index, RCA_STATE_FIELDS.index(field)]
                )
                for field in selected_state_fields
            }
        state_window.append({"native_step": native_step, "phases": phases})

    acoustic_step = min(max(center_step, start_step + 1), end_step)
    offset = acoustic_step - start_step - 1
    acoustic_window = []
    for substep_index in range(16):
        acoustic_window.append({
            **acoustic_substep_label(substep_index),
            "values": {
                field: json_number(
                    acoustic_target[offset, substep_index, RCA_ACOUSTIC_FIELDS.index(field)]
                )
                for field in RCA_ACOUSTIC_FIELDS
            },
        })

    return {
        "array_shapes": {name: list(shape) for name, shape in actual.items()},
        "first_nonfinite_event": earliest,
        "first_state_nonfinite_by_timeline": state_first[:40],
        "first_acoustic_nonfinite_by_timeline": acoustic_first[:40],
        "first_boundary_nonfinite_by_timeline": boundary_first[:40],
        "first_mu_limiter_activation": limiter,
        "target_mass_primitive_closure": {
            "equation": "raw_mu_tendency * mu_scale == output_mudf",
            "max_abs_error": json_number(closure_max),
            "sample_count": int(finite_closure.size),
        },
        "target_state_window": state_window,
        "target_acoustic_step": acoustic_step,
        "target_acoustic_substeps": acoustic_window,
    }


def terminal_nonfinite_summary(states: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain, state in states.items():
        fields: list[dict[str, Any]] = []
        for name in RCA_STATE_FIELDS:
            array = np.asarray(jax.device_get(getattr(state, name)))
            mask = ~np.isfinite(array)
            count = int(np.count_nonzero(mask))
            if count:
                first = tuple(int(value) for value in np.argwhere(mask)[0])
                fields.append({
                    "field": name,
                    "nonfinite_count": count,
                    "first_python_index": list(first),
                    "first_wrf_index": [value + 1 for value in first],
                })
        result[domain] = fields
    return result


def _trace_inventory(trace_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in trace_dir.rglob("*") if item.is_file()):
        rows.append({
            "path": str(path.relative_to(trace_dir)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    parser.add_argument(
        "--v10-proof",
        type=Path,
        default=Path(
            ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max/"
            "v10-spatial-causal-proof.json"
        ),
    )
    args = parser.parse_args(argv)
    work_dir = args.work_dir.resolve()
    proof_output = args.proof_output.resolve()
    v10_proof = args.v10_proof.resolve()
    if proof_output.exists() or proof_output.is_symlink():
        raise FileExistsError(proof_output)
    work_dir.mkdir(parents=True, exist_ok=True)
    if (work_dir / "runtime").exists():
        raise FileExistsError(work_dir / "runtime")
    cache_dir = Path(os.environ["GPUWRF_JAX_CACHE_DIR"]).resolve()
    if work_dir not in cache_dir.parents or not cache_dir.is_dir():
        raise RuntimeError("RCA cache must be a preseeded directory under --work-dir")
    expected_wrf_root = (RETRY20_ROOT / "authority/wrf_root").resolve()
    if Path(os.environ["GPUWRF_WRF_ROOT"]).resolve() != expected_wrf_root:
        raise RuntimeError("GPUWRF_WRF_ROOT is not the retained Retry20 authority snapshot")
    if not v10_proof.is_file():
        raise FileNotFoundError(v10_proof)

    started = datetime.now(timezone.utc)
    authority = {
        "preimport": _PREIMPORT_AUTHORITY,
        "candidate": assert_candidate_authority(),
        "inputs_pre": assert_input_authority(),
        "retry20": assert_retry20_authority(),
        "gpu_lock_pre": assert_lock_authority(),
        "preemption_pre": assert_preemption_clear("startup"),
        "cuda_runtime": assert_cuda_runtime(),
        "process": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "work_dir": str(work_dir),
            "cache_dir": str(cache_dir),
            "wrf_root": str(expected_wrf_root),
        },
    }
    if authority["process"]["cpu_affinity"] != [12, 13, 14, 15]:
        raise RuntimeError(
            f"RCA runner must be taskset to CPUs 12-15: {authority['process']['cpu_affinity']}"
        )

    tree, names, initial_carries, dt_by_domain, load_authority = load_corrected_tree(
        work_dir,
    )
    prefix_carries, prefix_steps, prefix_proof, prefix_cadence, _prefix_nonintegral = run_prefix(
        tree, names, initial_carries, dt_by_domain,
    )
    block_until_ready(prefix_carries)
    state_shapes = state_shape_map(prefix_carries["d03"].state)
    boundary_shapes = boundary_shape_map(prefix_carries["d03"].state)
    acoustic_shapes = acoustic_shape_map(prefix_carries["d03"].state)
    replay_cadence, replay_nonintegral, replay_schedule = scheduler_contract(
        names, dt_by_domain, total_steps=REPLAY_OWN_STEPS,
    )
    if replay_cadence != prefix_cadence:
        raise RuntimeError("prefix and replay scheduler cadence differ")

    assert_preemption_clear("canonical-pre")
    canonical_result, canonical_scheduler = run_replay(
        tree,
        prefix_carries,
        prefix_steps,
        cadence=replay_cadence,
        nonintegral=replay_nonintegral,
        mode="canonical",
    )
    canonical_manifest = host_tree_manifest(canonical_result.carries)
    canonical_nonfinite = terminal_nonfinite_summary(canonical_result.states)
    print(
        "RCA_REPLAY canonical "
        f"manifest={canonical_manifest['manifest_sha256']} "
        f"nonfinite={canonical_manifest['floating_nonfinite_count']}",
        flush=True,
    )
    del canonical_result
    gc.collect()

    assert_preemption_clear("recorder-off-pre")
    off_result, off_scheduler = run_replay(
        tree,
        prefix_carries,
        prefix_steps,
        cadence=replay_cadence,
        nonintegral=replay_nonintegral,
        mode="off",
    )
    off_manifest = host_tree_manifest(off_result.carries)
    off_nonfinite = terminal_nonfinite_summary(off_result.states)
    canonical_vs_off = compare_manifests(canonical_manifest, off_manifest)
    print(
        "RCA_REPLAY off "
        f"identity={canonical_vs_off['all_leaf_output_bytes_equal']} "
        f"manifest={off_manifest['manifest_sha256']}",
        flush=True,
    )
    del off_result
    gc.collect()

    assert_preemption_clear("recorder-compile-pre")
    recorder_program, recorder_executable = compile_and_audit_recorder(
        tree, prefix_carries,
    )
    if not recorder_program["callback_free"]:
        raise RuntimeError(f"recorder StableHLO contains callback tokens: {recorder_program}")
    step_arrays = {
        step: jax.device_put(np.asarray(step, dtype=np.int32))
        for step in range(PREFIX_OWN_STEPS["d03"] + 1, REPLAY_OWN_STEPS["d03"] + 1)
    }
    block_until_ready(tuple(step_arrays.values()))
    print(
        "RCA_RECORDER compiled "
        f"stablehlo={recorder_program['stablehlo_sha256']}",
        flush=True,
    )

    assert_preemption_clear("recorder-on-pre")
    trace_dir = work_dir / "runtime/recorder-trace"
    records: list[Any] = []
    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        on_result, on_scheduler = run_replay(
            tree,
            prefix_carries,
            prefix_steps,
            cadence=replay_cadence,
            nonintegral=replay_nonintegral,
            mode="on",
            records=records,
            recorder_executable=recorder_executable,
            recorder_step_arrays=step_arrays,
        )
        block_until_ready(on_result)
    h2d, d2h, transfer_files = count_transfer_bytes(trace_dir)
    transfer_audit = {
        "scope": (
            "precompiled recorder-ON nested replay d03 steps 9199..9405; "
            "all record/carry materialization occurs after profiler trace"
        ),
        "host_to_device_bytes": int(h2d),
        "device_to_host_bytes": int(d2h),
        "matched_trace_files": [str(Path(path).relative_to(trace_dir)) for path in transfer_files],
        "trace_inventory": _trace_inventory(trace_dir),
        "precompiled_before_trace": True,
        "step_scalars_device_resident_before_trace": True,
    }
    record_arrays = materialize_records(records)
    decoded = decode_records(
        record_arrays,
        state_shapes=state_shapes,
        boundary_shapes=boundary_shapes,
        acoustic_shapes=acoustic_shapes,
    )
    on_manifest = host_tree_manifest(on_result.carries)
    on_nonfinite = terminal_nonfinite_summary(on_result.states)
    canonical_vs_on = compare_manifests(canonical_manifest, on_manifest)
    print(
        "RCA_REPLAY on "
        f"identity={canonical_vs_on['all_leaf_output_bytes_equal']} "
        f"manifest={on_manifest['manifest_sha256']} "
        f"h2d={h2d} d2h={d2h}",
        flush=True,
    )

    d03_canonical_bad = canonical_nonfinite.get("d03", [])
    canonical_ni = next(
        (row for row in d03_canonical_bad if row["field"] == "Ni"), None,
    )
    failure_geometry_reproduced = bool(
        canonical_ni
        and canonical_ni["first_python_index"] == [1, 48, 78]
    )
    scheduler_identity = canonical_scheduler == off_scheduler == on_scheduler
    recorder_admissible = bool(
        canonical_vs_off["all_leaf_output_bytes_equal"]
        and canonical_vs_on["all_leaf_output_bytes_equal"]
        and scheduler_identity
        and recorder_program["callback_free"]
        and h2d == 0
        and d2h == 0
        and failure_geometry_reproduced
    )
    first_event = decoded.get("first_nonfinite_event")
    causal_gate = {
        "canonical_failure_geometry_reproduced": failure_geometry_reproduced,
        "canonical_terminal_Ni": canonical_ni,
        "canonical_vs_unit_chunk_off_identity": canonical_vs_off[
            "all_leaf_output_bytes_equal"
        ],
        "canonical_vs_recorder_on_identity": canonical_vs_on[
            "all_leaf_output_bytes_equal"
        ],
        "scheduler_identity": scheduler_identity,
        "callback_free": recorder_program["callback_free"],
        "zero_transfer_trace": h2d == 0 and d2h == 0,
        "recorder_admissible": recorder_admissible,
        "first_event_usable_for_causality": first_event if recorder_admissible else None,
        "inadmissible_first_event_for_debug_only": None if recorder_admissible else first_event,
    }

    inputs_post = assert_input_authority()
    if inputs_post["authority_sha256"] != authority["inputs_pre"]["authority_sha256"]:
        raise RuntimeError("corrected inputs changed during RCA replay")
    authority["inputs_post"] = inputs_post
    authority["gpu_lock_post"] = assert_lock_authority()
    authority["preemption_post"] = assert_preemption_clear("completion")
    authority["v10_spatial_causal_proof"] = {
        "path": str(v10_proof),
        "sha256": sha256_file(v10_proof),
        "bytes": v10_proof.stat().st_size,
    }
    authority["finished_utc"] = datetime.now(timezone.utc).isoformat()
    authority["wall_seconds"] = (
        datetime.now(timezone.utc) - started
    ).total_seconds()

    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "SHORT_REPLAY_COMPLETE",
        "authority": authority,
        "load_authority": load_authority,
        "geometry": {
            "corrected_d03_shape": state_shapes["Ni"],
            "target_python_index": [RCA_TARGET_K, RCA_TARGET_Y, RCA_TARGET_X],
            "target_wrf_index": [RCA_TARGET_K + 1, RCA_TARGET_Y + 1, RCA_TARGET_X + 1],
            "lat": 28.297913,
            "lon": -16.303406,
            "LANDMASK": 0,
            "HGT_m": 0,
            "classification": "interior_offshore_east_northeast_not_boundary_edge",
        },
        "prefix": prefix_proof,
        "replay_scheduler": replay_schedule,
        "canonical": {
            "scheduler": canonical_scheduler,
            "carry_manifest": canonical_manifest,
            "terminal_nonfinite": canonical_nonfinite,
        },
        "recorder_off": {
            "scheduler": off_scheduler,
            "carry_manifest": off_manifest,
            "terminal_nonfinite": off_nonfinite,
            "comparison_to_canonical": canonical_vs_off,
        },
        "recorder_program": recorder_program,
        "recorder_on": {
            "scheduler": on_scheduler,
            "carry_manifest": on_manifest,
            "terminal_nonfinite": on_nonfinite,
            "comparison_to_canonical": canonical_vs_on,
            "transfer_audit": transfer_audit,
            "decoded_records": decoded,
        },
        "causal_gate": causal_gate,
        "candidate_fix": None,
        "full_18h_run_performed": False,
        "full_18h_reason": "reserved for manager/critic after a causally authorized candidate",
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(proof_output, proof)
    print(
        json.dumps({
            "proof": str(proof_output),
            "proof_sha256": proof["proof_sha256"],
            "recorder_admissible": recorder_admissible,
            "failure_geometry_reproduced": failure_geometry_reproduced,
            "first_event": first_event,
        }, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
