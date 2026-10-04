#!/usr/bin/env python3
"""Exact-production ordinary-dispatch checkpoint localization for corrected d03.

This continuation deliberately has no recorder lane.  It builds one corrected
three-domain prefix to d03 native step 9198, then reuses the ordinary production
one-step path for d03 steps 9199..9405.  A separate compiled health reduction is
called only at host-visible boundaries between completed ordinary dispatches.
The external observations are admissible only when the final complete carry is
byte-identical to the frozen ordinary replay manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
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
            "pre-import runtime environment differs from the frozen ordinary lane: "
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
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.runtime.domain_tree import (
    DomainTree,
    _operational_advance_factory,
    _operational_force,
    run_domain_tree_callbacks,
    run_operational_domain_tree,
)
from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
from gpuwrf.runtime.operational_mode import _advance_chunk_fori, build_clock_base


SCHEMA = "gpuwrf.v0234.corrected-ni-ordinary-bisection.v1"
CONTINUATION_PARENT_SHA = "fc0290da83d478acc710e32a7165e51fe2daf4f1"
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
PRIOR_RCA_DIR = Path(".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max")
PRIOR_RECORDER_PROOF = PRIOR_RCA_DIR / "recorder-proof.json"
PRIOR_RECORDER_PROOF_SHA256 = "cd0ca04df4f96e1f896c97fde2a68dd5d7b97dcaa84c07e2c5baa4209046cbc5"
PRIOR_AUTHORITY_PROOF = PRIOR_RCA_DIR / "authority-proof.json"
PRIOR_AUTHORITY_PROOF_SHA256 = "4132b6e467b6313d4ae80171b1817119478fc3d0a8cfebc0244d0b893f407497"
PRIOR_V10_PROOF = PRIOR_RCA_DIR / "v10-spatial-causal-proof.json"
PRIOR_V10_PROOF_SHA256 = "d269513748c85006a7a95b87f778e4c442041c812b2d76dee85ff20309599e68"
EXPECTED_TERMINAL_MANIFEST = "2dcfc195baaf3e59ba539f6701fdb82b9a134a0461cfc68dec3a429e38433565"
EXPECTED_RETRY20_COUNTS = {"d01": 16, "d02": 16, "d03": 47}
EXPECTED_LAST_D03 = "wrfout_d03_2025-03-01_15:20:00"
LINEAGE_WORK_DIR = TERMINAL_CONTRACT.parent / "corrected_ni_rca_max_22c2bd7a"
LINEAGE_CACHE = LINEAGE_WORK_DIR / "cache/bb2ffe33dbff-8ec38f8e1a90/jit"
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
PREFIX_OWN_STEPS = {"d01": 1022, "d02": 3066, "d03": 9198}
REPLAY_OWN_STEPS = {"d01": 1045, "d02": 3135, "d03": 9405}
REPLAY_D03_STEPS = tuple(range(9199, 9406))


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


def atomic_write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        pickle.dump(payload, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def assert_candidate_authority() -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    approved = os.environ.get("GPUWRF_CORRECTED_NI_BISECTION_APPROVED_SHA")
    if approved != head:
        raise RuntimeError(
            "GPUWRF_CORRECTED_NI_BISECTION_APPROVED_SHA must equal committed HEAD"
        )
    if git_output("status", "--porcelain"):
        raise RuntimeError("ordinary bisection requires a clean worktree")
    for ancestor in (CONTINUATION_PARENT_SHA, NUMERICAL_BASE_SHA, RUNTIME_PARENT_SHA):
        subprocess.run(("git", "merge-base", "--is-ancestor", ancestor, head), check=True)
    source_rows = [
        (relative, sha256_file(Path(relative)))
        for relative in git_output("ls-files", "src/gpuwrf").splitlines()
    ]
    return {
        "head": head,
        "approved_sha": approved,
        "continuation_parent_sha": CONTINUATION_PARENT_SHA,
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
            "path": str(TERMINAL_CONTRACT),
            "sha256": contract_digest,
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


def assert_prior_authority() -> dict[str, Any]:
    if sha256_file(PRIOR_RECORDER_PROOF) != PRIOR_RECORDER_PROOF_SHA256:
        raise RuntimeError("prior recorder attempt proof changed")
    if sha256_file(PRIOR_AUTHORITY_PROOF) != PRIOR_AUTHORITY_PROOF_SHA256:
        raise RuntimeError("prior authority proof changed")
    if sha256_file(PRIOR_V10_PROOF) != PRIOR_V10_PROOF_SHA256:
        raise RuntimeError("prior V10 spatial-causal proof changed")
    prior = json.loads(PRIOR_RECORDER_PROOF.read_text())
    gate = prior["causal_gate"]
    if not gate["canonical_vs_unit_chunk_off_identity"]:
        raise RuntimeError("prior ordinary unit-step identity gate is not green")
    if prior["recorder_off"]["carry_manifest"]["manifest_sha256"] != EXPECTED_TERMINAL_MANIFEST:
        raise RuntimeError("prior ordinary terminal manifest changed")
    v10 = json.loads(PRIOR_V10_PROOF.read_text())
    v10_expected = {
        "common_root_proved": False,
        "valid_time": "2025-03-01T15:00:00+00:00",
        "rmse": 2.1128268857679338,
        "land_rmse": 2.2264724301076377,
        "sea_rmse": 2.0839931788025168,
        "max_abs": 11.358115434646606,
        "max_location": {
            "landmask": 0,
            "lat": 28.216018676757812,
            "lon": -16.90625,
            "x": 19,
            "y": 39,
        },
        "ni_cell_delta": -0.7763886451721191,
    }
    v10_actual = {
        "common_root_proved": v10["bounded_gates"]["common_ni_v10_root_proved"],
        "valid_time": v10["last_frame"]["valid_time"],
        "rmse": v10["last_frame"]["v10"]["rmse"],
        "land_rmse": v10["last_frame"]["v10"]["land_rmse"],
        "sea_rmse": v10["last_frame"]["v10"]["sea_rmse"],
        "max_abs": v10["last_frame"]["v10"]["max_abs"],
        "max_location": v10["last_frame"]["v10"]["max_location"],
        "ni_cell_delta": v10["last_frame"]["v10"]["ni_cell_delta"],
    }
    if v10_actual != v10_expected:
        raise RuntimeError("prior bounded Ni/V10 separation evidence changed")
    attestation = json.loads(RETRY20_ATTESTATION.read_text())
    final_accept = json.loads(RETRY20_FINAL_ACCEPT.read_text())
    cache_authority = json.loads(RETRY20_CACHE_AUTHORITY.read_text())
    runtime = attestation.get("runtime_source") or {}
    runtime_sha = runtime.get("commit") or runtime.get("head") or runtime.get("sha")
    if runtime_sha != RUNTIME_PARENT_SHA:
        raise RuntimeError("Retry20 runtime source changed")
    if final_accept.get("runtime_source_commit") != RUNTIME_PARENT_SHA:
        raise RuntimeError("Retry20 final acceptance runtime binding changed")
    if not LINEAGE_WORK_DIR.is_dir() or not LINEAGE_CACHE.is_dir():
        raise RuntimeError("existing RCA workdir/cache lineage is unavailable")
    if Path(os.environ["GPUWRF_JAX_CACHE_DIR"]).resolve() != LINEAGE_CACHE.resolve():
        raise RuntimeError("bisection must reuse the existing RCA JAX cache")
    return {
        "retry20_artifacts": {
            path.name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (RETRY20_ATTESTATION, RETRY20_FINAL_ACCEPT, RETRY20_CACHE_AUTHORITY)
        },
        "retry20_runtime_source_commit": runtime_sha,
        "retry20_outputs": _retry20_output_inventory(),
        "prior_recorder_proof": {
            "path": str(PRIOR_RECORDER_PROOF.resolve()),
            "sha256": PRIOR_RECORDER_PROOF_SHA256,
            "ordinary_unit_step_identity": True,
            "ordinary_terminal_manifest": EXPECTED_TERMINAL_MANIFEST,
        },
        "prior_authority_proof": {
            "path": str(PRIOR_AUTHORITY_PROOF.resolve()),
            "sha256": PRIOR_AUTHORITY_PROOF_SHA256,
        },
        "prior_v10_spatial_causal_proof": {
            "path": str(PRIOR_V10_PROOF.resolve()),
            "sha256": PRIOR_V10_PROOF_SHA256,
            "bounded_evidence": v10_actual,
            "causal_verdict": (
                "Ni onset and broad V10 drift remain separate mechanisms unless "
                "a temporal and operator-causal link is proved"
            ),
        },
        "lineage_work_dir": str(LINEAGE_WORK_DIR.resolve()),
        "lineage_cache": str(LINEAGE_CACHE.resolve()),
        "cache_authority_payload_sha256": canonical_digest(cache_authority),
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
    spec = importlib.util.spec_from_file_location("v0234_bisection_gpu_lock_v2", LOCK_VERIFIER)
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
        raise RuntimeError(f"ordinary lane is not CUDA-only: {devices!r}")
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
    }


class _ScheduleOnlyOutput:
    """Preserve production output-alarm splits without materializing state."""

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


def load_corrected_tree(run_dir: Path) -> tuple[
    DomainTree,
    tuple[str, ...],
    dict[str, Any],
    dict[str, float],
    dict[str, Any],
]:
    names = domain_names_for(3)
    runtime_root = run_dir / "runtime"
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


def run_prefix(
    tree: DomainTree,
    names: tuple[str, ...],
    initial_carries: dict[str, Any],
    dt_by_domain: dict[str, float],
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any], dict[str, int]]:
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
            f"BISECTION_PREFIX segment={index + 1}/{len(segments)} own_steps={own_steps}",
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
        "finite_state_guard_at_boundary": True,
    }, cadence


def _edge_lookup(tree: DomainTree):
    edges = {
        (edge.parent, edge.child): edge
        for values in tree.edges.values()
        for edge in values
    }

    def lookup(spec):
        return edges[(spec.parent, spec.child)]

    return lookup


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


def _subtree_leaf_labels(prefix: str, value: Any) -> list[str]:
    labels: list[str] = []
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(value)
    for path, _leaf in path_leaves:
        suffix = jax.tree_util.keystr(path)
        labels.append(prefix if not suffix else f"{prefix}{suffix}")
    return labels


def carry_leaf_labels(carry: Any) -> list[str]:
    """Return stable semantic labels in exact OperationalCarry pytree order."""

    labels: list[str] = []
    if not is_dataclass(carry):
        raise TypeError("expected dataclass OperationalCarry")
    for carry_name in carry.__dataclass_fields__:
        value = getattr(carry, carry_name)
        if carry_name == "state":
            for state_name in value.__slots__:
                labels.extend(
                    _subtree_leaf_labels(f"state.{state_name}", getattr(value, state_name))
                )
        else:
            labels.extend(_subtree_leaf_labels(carry_name, value))
    leaves = jax.tree_util.tree_leaves(carry)
    if len(labels) != len(leaves):
        raise RuntimeError(f"semantic labels do not match carry leaves: {len(labels)} != {len(leaves)}")
    return labels


def carry_leaf_specs(carry: Any) -> list[dict[str, Any]]:
    labels = carry_leaf_labels(carry)
    leaves = jax.tree_util.tree_leaves(carry)
    return [
        {
            "leaf_index": index,
            "name": name,
            "shape": list(leaf.shape),
            "dtype": str(leaf.dtype),
        }
        for index, (name, leaf) in enumerate(zip(labels, leaves))
    ]


def carry_nonfinite_counts(carry: Any) -> jax.Array:
    counts = []
    for leaf in jax.tree_util.tree_leaves(carry):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            counts.append(jnp.count_nonzero(~jnp.isfinite(leaf)))
        else:
            counts.append(jnp.asarray(0, dtype=jnp.int32))
    return jnp.stack(counts).astype(jnp.int64)


def build_health_executable(prefix_carry: Any) -> tuple[dict[str, Any], Any]:
    health_jit = jax.jit(carry_nonfinite_counts)
    lowered = health_jit.lower(prefix_carry)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    callback_tokens = [
        token
        for token in (
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
        )
        if token in stablehlo.lower()
    ]
    executable = lowered.compile()
    return {
        "separate_from_production_hlo": True,
        "lowered_once": True,
        "compiled_once": True,
        "stablehlo_sha256": sha256_text(stablehlo),
        "stablehlo_bytes": len(stablehlo.encode()),
        "callback_tokens_found": callback_tokens,
        "callback_free": not callback_tokens,
        "output": "one int64 nonfinite count per d03 OperationalCarry array leaf",
    }, executable


def audit_ordinary_one_step(tree: DomainTree, prefix_carry: Any) -> dict[str, Any]:
    namelist = tree.domains["d03"].namelist
    clock = build_clock_base(namelist)
    lowered = _advance_chunk_fori.lower(
        prefix_carry,
        namelist,
        jnp.asarray(9199, dtype=jnp.int32),
        clock,
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    forbidden_tokens = [
        token
        for token in (
            "corrected_ni_rca",
            "rca_state",
            "rca_acoustic",
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
        )
        if token in stablehlo.lower()
    ]
    input_treedef = jax.tree_util.tree_structure(prefix_carry)
    output_treedef = jax.tree_util.tree_structure(lowered.out_info)
    input_avals = [
        {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
        for leaf in jax.tree_util.tree_leaves(prefix_carry)
    ]
    output_avals = [
        {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
        for leaf in jax.tree_util.tree_leaves(lowered.out_info)
    ]
    structure_identity = input_treedef == output_treedef and input_avals == output_avals
    executable = lowered.compile()
    compiled_output_treedef = executable.out_tree
    if compiled_output_treedef != output_treedef:
        raise RuntimeError("compiled ordinary executable output tree changed after lowering")
    return {
        "callable": "gpuwrf.runtime.operational_mode._advance_chunk_fori",
        "n_steps": 1,
        "cadence": int(namelist.radiation_cadence_steps),
        "lowered_once": True,
        "compiled_once": True,
        "stablehlo_sha256": sha256_text(stablehlo),
        "stablehlo_bytes": len(stablehlo.encode()),
        "forbidden_tokens_found": forbidden_tokens,
        "ordinary_only": not forbidden_tokens and structure_identity,
        "input_treedef": str(input_treedef),
        "output_treedef": str(output_treedef),
        "compiled_output_treedef": str(compiled_output_treedef),
        "input_leaf_count": len(input_avals),
        "output_leaf_count": len(output_avals),
        "input_output_structure_and_avals_identical": structure_identity,
        "input_avals": input_avals,
        "output_avals": output_avals,
    }


def audit_source_separation() -> dict[str, Any]:
    source = inspect.getsource(run_ordinary_checkpoint_replay)
    dispatch = "value = normal_advance(name, value, native_step, 1)"
    health = "post_counts = materialize_health(health_executable, value)"
    dispatch_offset = source.find(dispatch)
    health_offset = source.find(health)
    recorder_tokens = [
        token
        for token in (
            "advance_chunk_with_corrected_ni_rca",
            "_physics_boundary_step_with_rca",
            "recorder_executable",
        )
        if token in source
    ]
    ordered = 0 <= dispatch_offset < health_offset
    if not ordered or recorder_tokens:
        raise RuntimeError("ordinary replay source separation audit failed")
    return {
        "function": "run_ordinary_checkpoint_replay",
        "source_sha256": sha256_text(source),
        "ordinary_dispatch_precedes_post_health_materialization": ordered,
        "health_result_not_passed_to_dispatch": "normal_advance(name, value, native_step, 1)" in source,
        "recorder_tokens_found": recorder_tokens,
        "source_separated": ordered and not recorder_tokens,
    }


def materialize_health(executable: Any, carry: Any) -> np.ndarray:
    counts = np.asarray(jax.device_get(executable(carry)), dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise RuntimeError("invalid carry health vector")
    return counts


def _nonzero_health_rows(
    counts: np.ndarray,
    baseline: np.ndarray,
    leaf_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for spec, count, base in zip(leaf_specs, counts, baseline):
        if int(count) != int(base):
            rows.append({
                "leaf_index": spec["leaf_index"],
                "name": spec["name"],
                "baseline_nonfinite": int(base),
                "nonfinite_count": int(count),
            })
    return rows


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
        if np.issubdtype(contiguous.dtype, np.inexact):
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
            "treedef": str(treedef),
            "leaves": identity_rows,
        }),
    }


def compare_manifests(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_rows = {row["path"]: row for row in left["leaves"]}
    right_rows = {row["path"]: row for row in right["leaves"]}
    differences = []
    for path in sorted(set(left_rows) | set(right_rows)):
        before = left_rows.get(path)
        after = right_rows.get(path)
        if before != after:
            differences.append({"path": path, "left": before, "right": after})
    equal = (
        left.get("treedef") == right.get("treedef")
        and left.get("manifest_sha256") == right.get("manifest_sha256")
        and not differences
    )
    return {
        "all_leaf_bytes_equal": equal,
        "left_manifest_sha256": left.get("manifest_sha256"),
        "right_manifest_sha256": right.get("manifest_sha256"),
        "different_leaf_count": len(differences),
        "first_differences": differences[:20],
    }


def named_nonfinite_inventory(carry: Any) -> list[dict[str, Any]]:
    labels = carry_leaf_labels(carry)
    leaves = jax.tree_util.tree_leaves(carry)
    rows = []
    for leaf_index, (name, leaf) in enumerate(zip(labels, leaves)):
        array = np.asarray(leaf)
        if not np.issubdtype(array.dtype, np.inexact):
            continue
        mask = ~np.isfinite(array)
        count = int(np.count_nonzero(mask))
        if count:
            first = tuple(int(value) for value in np.argwhere(mask)[0])
            rows.append({
                "leaf_index": leaf_index,
                "name": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "nonfinite_count": count,
                "first_python_index": list(first),
                "first_wrf_index": [value + 1 for value in first],
            })
    return rows


def retain_checkpoint(
    checkpoint_dir: Path,
    *,
    role: str,
    native_step: int,
    host_carry: Any,
) -> dict[str, Any]:
    path = checkpoint_dir / f"{role}-d03-step-{native_step}.pkl"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    manifest_before = host_tree_manifest(host_carry)
    atomic_write_pickle(path, host_carry)
    file_digest = sha256_file(path)
    with path.open("rb") as handle:
        reread = pickle.load(handle)
    manifest_after = host_tree_manifest(reread)
    comparison = compare_manifests(manifest_before, manifest_after)
    if not comparison["all_leaf_bytes_equal"]:
        raise RuntimeError(f"checkpoint re-read identity failed: {role}")
    return {
        "role": role,
        "native_step": native_step,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": file_digest,
        "manifest": manifest_before,
        "reread_identity": comparison,
        "named_nonfinite_inventory": named_nonfinite_inventory(reread),
    }


def run_ordinary_checkpoint_replay(
    tree: DomainTree,
    prefix_carries: dict[str, Any],
    prefix_steps: dict[str, int],
    *,
    cadence: dict[str, int],
    nonintegral: dict[str, tuple[int, ...]],
    health_executable: Any,
    leaf_specs: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any], dict[str, Any], Any, Any]:
    normal_advance = _operational_advance_factory(tree)
    baseline_counts = materialize_health(health_executable, prefix_carries["d03"])
    baseline_total = int(np.sum(baseline_counts, dtype=np.int64))
    if baseline_total != 0:
        raise RuntimeError(f"prefix d03 carry is already nonfinite: {baseline_total}")

    output = _ScheduleOnlyOutput()
    block_between, root_sync_cadence = _nested_sync_mode_from_env()
    rows: list[dict[str, Any]] = []
    first_bad: dict[str, Any] | None = None
    last_finite_host: Any | None = None
    first_bad_host: Any | None = None
    last_post_counts = baseline_counts

    def advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        nonlocal first_bad, last_finite_host, first_bad_host
        nonlocal last_post_counts
        if name != "d03":
            return normal_advance(name, carry, start_step, n_steps)

        value = carry
        entry_counts = materialize_health(health_executable, value)
        force_changed = not np.array_equal(entry_counts, last_post_counts)
        for offset in range(int(n_steps)):
            native_step = int(start_step + offset)
            assert_preemption_clear(f"ordinary-d03-step-{native_step}-pre")
            before = value
            pre_counts = entry_counts if offset == 0 else last_post_counts
            pre_total = int(np.sum(pre_counts, dtype=np.int64))
            value = normal_advance(name, value, native_step, 1)
            post_counts = materialize_health(health_executable, value)
            post_total = int(np.sum(post_counts, dtype=np.int64))
            row = {
                "native_step": native_step,
                "sim_time_s": native_step * 6,
                "pre_nonfinite_count": pre_total,
                "post_nonfinite_count": post_total,
                "callback_entry_after_parent_force": bool(offset == 0),
                "parent_force_changed_health": bool(offset == 0 and force_changed),
                "pre_changes_from_prefix": _nonzero_health_rows(
                    pre_counts, baseline_counts, leaf_specs,
                ),
                "post_changes_from_prefix": _nonzero_health_rows(
                    post_counts, baseline_counts, leaf_specs,
                ),
            }
            rows.append(row)
            if first_bad is None and post_total > baseline_total:
                last_finite_host = jax.device_get(before)
                first_bad_host = jax.device_get(value)
                first_bad = {
                    "native_step": native_step,
                    "sim_time_s": native_step * 6,
                    "pre_nonfinite_count": pre_total,
                    "post_nonfinite_count": post_total,
                    "parent_force_changed_health": bool(offset == 0 and force_changed),
                    "introduced_by_completed_ordinary_dispatch": pre_total == baseline_total,
                    "post_changes_from_prefix": row["post_changes_from_prefix"],
                }
                print(
                    "BISECTION_TRANSITION "
                    f"step={native_step} pre={pre_total} post={post_total} "
                    f"force_changed={bool(offset == 0 and force_changed)}",
                    flush=True,
                )
            elif native_step % 10 == 0 or native_step == REPLAY_D03_STEPS[-1]:
                print(
                    f"BISECTION_STEP step={native_step} nonfinite={post_total}",
                    flush=True,
                )
            last_post_counts = post_counts
        return value

    result = run_domain_tree_callbacks(
        tree.hierarchy,
        prefix_carries,
        root_steps=23,
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
        raise RuntimeError(f"ordinary replay stopped at wrong clocks: {result.own_steps!r}")
    sampled = [row["native_step"] for row in rows]
    exact_sampling = sampled == list(REPLAY_D03_STEPS)
    if not exact_sampling:
        raise RuntimeError("ordinary replay did not sample each d03 step exactly once")
    if first_bad is None or last_finite_host is None or first_bad_host is None:
        raise RuntimeError("ordinary replay found no unique finite-to-nonfinite transition")
    first_offset = int(first_bad["native_step"]) - REPLAY_D03_STEPS[0]
    before_rows = rows[:first_offset]
    unique_transition = bool(
        first_bad["pre_nonfinite_count"] == baseline_total
        and all(row["post_nonfinite_count"] == baseline_total for row in before_rows)
        and rows[first_offset]["post_nonfinite_count"] > baseline_total
    )
    replay_proof = {
        "baseline_native_step": PREFIX_OWN_STEPS["d03"],
        "baseline_nonfinite_count": baseline_total,
        "sampled_steps": len(rows),
        "expected_sampled_steps": len(REPLAY_D03_STEPS),
        "exact_step_sampling": exact_sampling,
        "unique_transition": unique_transition,
        "first_bad": first_bad,
        "health_rows": rows,
        "scheduler": _event_summary(result, output),
        "health_materialization_position": (
            "separate executable at callback entry after parent force and after each "
            "completed ordinary d03 dispatch; never inside ordinary production HLO"
        ),
    }
    return result, _event_summary(result, output), replay_proof, last_finite_host, first_bad_host


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    proof_output = args.proof_output.resolve()
    if run_dir == LINEAGE_WORK_DIR.resolve() or LINEAGE_WORK_DIR.resolve() not in run_dir.parents:
        raise RuntimeError("--run-dir must be a fresh child of the existing RCA workdir lineage")
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(run_dir)
    if proof_output.exists() or proof_output.is_symlink():
        raise FileExistsError(proof_output)
    if Path(os.environ["GPUWRF_WRF_ROOT"]).resolve() != (RETRY20_ROOT / "authority/wrf_root").resolve():
        raise RuntimeError("GPUWRF_WRF_ROOT is not the retained Retry20 authority snapshot")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)

    authority = {
        "preimport": _PREIMPORT_AUTHORITY,
        "candidate": assert_candidate_authority(),
        "inputs_pre": assert_input_authority(),
        "prior": assert_prior_authority(),
        "gpu_lock_pre": assert_lock_authority(),
        "preemption_pre": assert_preemption_clear("startup"),
        "cuda_runtime": assert_cuda_runtime(),
        "process": {
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "run_dir": str(run_dir),
            "cache_dir": str(LINEAGE_CACHE.resolve()),
        },
    }
    if authority["process"]["cpu_affinity"] != [12, 13, 14, 15]:
        raise RuntimeError(
            f"ordinary bisection must be taskset to CPUs 12-15: "
            f"{authority['process']['cpu_affinity']}"
        )

    tree, names, initial_carries, dt_by_domain, load_authority = load_corrected_tree(run_dir)
    prefix_carries, prefix_steps, prefix_proof, prefix_cadence = run_prefix(
        tree, names, initial_carries, dt_by_domain,
    )
    block_until_ready(prefix_carries)
    leaf_specs = carry_leaf_specs(prefix_carries["d03"])
    ordinary_program = audit_ordinary_one_step(tree, prefix_carries["d03"])
    if not ordinary_program["ordinary_only"]:
        raise RuntimeError(f"ordinary one-step StableHLO contains forbidden tokens: {ordinary_program}")
    health_program, health_executable = build_health_executable(prefix_carries["d03"])
    if not health_program["callback_free"]:
        raise RuntimeError(f"health executable contains callback tokens: {health_program}")
    source_separation = audit_source_separation()

    replay_cadence, replay_nonintegral, replay_schedule = scheduler_contract(
        names, dt_by_domain, total_steps=REPLAY_OWN_STEPS,
    )
    if replay_cadence != prefix_cadence:
        raise RuntimeError("prefix and replay scheduler cadence differ")
    assert_preemption_clear("ordinary-replay-pre")
    result, replay_scheduler, replay_proof, last_finite_host, first_bad_host = (
        run_ordinary_checkpoint_replay(
            tree,
            prefix_carries,
            prefix_steps,
            cadence=replay_cadence,
            nonintegral=replay_nonintegral,
            health_executable=health_executable,
            leaf_specs=leaf_specs,
        )
    )

    terminal_manifest = host_tree_manifest(result.carries)
    terminal_identity = terminal_manifest["manifest_sha256"] == EXPECTED_TERMINAL_MANIFEST
    print(
        "BISECTION_TERMINAL "
        f"manifest={terminal_manifest['manifest_sha256']} identity={terminal_identity}",
        flush=True,
    )

    transition_step = int(replay_proof["first_bad"]["native_step"])
    checkpoint_dir = run_dir / "checkpoints"
    last_finite_checkpoint = retain_checkpoint(
        checkpoint_dir,
        role="last-finite-input-to-first-bad",
        native_step=transition_step - 1,
        host_carry=last_finite_host,
    )
    first_bad_checkpoint = retain_checkpoint(
        checkpoint_dir,
        role="first-bad-output",
        native_step=transition_step,
        host_carry=first_bad_host,
    )
    checkpoint_digest = canonical_digest({
        "last_finite": {
            "path": last_finite_checkpoint["path"],
            "file_sha256": last_finite_checkpoint["file_sha256"],
            "manifest_sha256": last_finite_checkpoint["manifest"]["manifest_sha256"],
        },
        "first_bad": {
            "path": first_bad_checkpoint["path"],
            "file_sha256": first_bad_checkpoint["file_sha256"],
            "manifest_sha256": first_bad_checkpoint["manifest"]["manifest_sha256"],
        },
    })

    inputs_post = assert_input_authority()
    if inputs_post["authority_sha256"] != authority["inputs_pre"]["authority_sha256"]:
        raise RuntimeError("corrected inputs changed during ordinary replay")
    authority["inputs_post"] = inputs_post
    authority["gpu_lock_post"] = assert_lock_authority()
    authority["preemption_post"] = assert_preemption_clear("completion")
    authority["finished_utc"] = datetime.now(timezone.utc).isoformat()
    authority["wall_seconds"] = (datetime.now(timezone.utc) - started).total_seconds()

    checkpoint_identity = bool(
        last_finite_checkpoint["reread_identity"]["all_leaf_bytes_equal"]
        and first_bad_checkpoint["reread_identity"]["all_leaf_bytes_equal"]
    )
    checkpoint_health = bool(
        last_finite_checkpoint["manifest"]["floating_nonfinite_count"] == 0
        and not last_finite_checkpoint["named_nonfinite_inventory"]
        and first_bad_checkpoint["manifest"]["floating_nonfinite_count"] > 0
        and first_bad_checkpoint["named_nonfinite_inventory"]
    )
    localized = bool(
        replay_proof["exact_step_sampling"]
        and replay_proof["unique_transition"]
        and replay_proof["first_bad"]["introduced_by_completed_ordinary_dispatch"]
        and terminal_identity
        and checkpoint_identity
        and checkpoint_health
        and ordinary_program["ordinary_only"]
        and health_program["callback_free"]
        and source_separation["source_separated"]
    )
    verdict = "ORDINARY_BISECTION_LOCALIZED" if localized else "ORDINARY_BISECTION_BLOCKED"
    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "verdict": verdict,
        "authority": authority,
        "load_authority": load_authority,
        "geometry": {
            "target_python_index": [1, 48, 78],
            "target_wrf_index": [2, 49, 79],
            "lat": 28.297913,
            "lon": -16.303406,
            "LANDMASK": 0,
            "HGT_m": 0,
            "classification": "interior_offshore_east_northeast_not_boundary_edge",
        },
        "prefix": prefix_proof,
        "replay_schedule": replay_schedule,
        "ordinary_program": ordinary_program,
        "health_program": health_program,
        "source_separation": source_separation,
        "d03_leaf_specs": leaf_specs,
        "ordinary_replay": replay_proof,
        "replay_scheduler": replay_scheduler,
        "terminal_identity": {
            "required_manifest_sha256": EXPECTED_TERMINAL_MANIFEST,
            "actual_manifest_sha256": terminal_manifest["manifest_sha256"],
            "all_leaf_bytes_equal_to_frozen_ordinary": terminal_identity,
            "manifest": terminal_manifest,
        },
        "retained_carries": {
            "aggregate_digest": checkpoint_digest,
            "last_finite": last_finite_checkpoint,
            "first_bad": first_bad_checkpoint,
            "reread_identity": checkpoint_identity,
            "last_finite_and_first_bad_health_proved": checkpoint_health,
        },
        "mechanism_separation": {
            "bounded_v10_evidence": authority["prior"]["prior_v10_spatial_causal_proof"],
            "common_root_proved": False,
            "causal_gate": (
                "This ordinary-dispatch experiment localizes only the Ni-track "
                "transition. It supplies no temporal or operator-causal linkage to "
                "the spatially broad V10 drift, so the mechanisms remain separate."
            ),
            "v10_gate": "OPEN_SEPARATE_TRACK_NO_FIX_AUTHORITY",
        },
        "scope_gates": {
            "recorder_on_used": False,
            "phase_tap_executed": False,
            "model_or_numerical_edit": False,
            "full_18h_run_performed": False,
        },
        "localization_gate": localized,
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(proof_output, proof)
    print(
        json.dumps({
            "proof": str(proof_output),
            "proof_sha256": proof["proof_sha256"],
            "verdict": verdict,
            "first_bad": replay_proof["first_bad"],
            "terminal_manifest": terminal_manifest["manifest_sha256"],
            "checkpoint_digest": checkpoint_digest,
        }, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0 if localized else 3


if __name__ == "__main__":
    raise SystemExit(main())
