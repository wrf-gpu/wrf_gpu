#!/usr/bin/env python3
"""Bounded CPU-only dependency-closure preflight for the V10 replay.

This utility deliberately calls only ``load_corrected_tree``.  It does not
advance a timestep, construct an output writer, or touch a GPU.  Its retained
proof establishes that the read-only WRF runtime authority is sufficient to
load all three corrected Tenerife domains before a fresh GPU namespace is
requested.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import time
import traceback
from typing import Any, Mapping


SCHEMA = "gpuwrf.v0234.v10-cpu-domain-load-preflight.v1"
MODEL_TREE = "5a6298fba90e76c3cafe674e1413d38efb694532"
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_AUTHORITY_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "v0234_v10_runtime_authority_complete1_5a6298fb"
)
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
REQUIRED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": str(RUNTIME_AUTHORITY_ROOT),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    ).hexdigest()


def atomic_write_self_hashed(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    final = dict(payload)
    final["proof_sha256"] = canonical_digest(final)
    encoded = (json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    observed = {name: environment.get(name) for name in REQUIRED_ENVIRONMENT}
    if observed != REQUIRED_ENVIRONMENT:
        raise RuntimeError(
            f"CPU preflight environment mismatch: expected={REQUIRED_ENVIRONMENT!r} "
            f"observed={observed!r}"
        )
    if environment.get("XLA_FLAGS"):
        raise RuntimeError("CPU preflight forbids XLA_FLAGS and autotune load/dump flags")
    return {
        "validated_before_jax_import": True,
        "environment": observed,
        "xla_flags_absent": True,
    }


def validate_runtime_authority() -> dict[str, Any]:
    root = RUNTIME_AUTHORITY_ROOT
    required_directories = (root, root / "run", root / "phys")
    for path in required_directories:
        if not path.is_dir() or path.is_symlink() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"runtime authority directory is not read-only/exact: {path}")
    inventory = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if inventory != sorted(RUNTIME_AUTHORITY_FILES):
        raise RuntimeError(f"runtime authority inventory mismatch: {inventory!r}")
    rows: dict[str, Any] = {}
    for relative, (expected_bytes, expected_sha256) in sorted(
        RUNTIME_AUTHORITY_FILES.items()
    ):
        path = root / relative
        observed_sha256 = sha256_file(path) if path.is_file() else None
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_mode & 0o222
            or path.stat().st_size != expected_bytes
            or observed_sha256 != expected_sha256
        ):
            raise RuntimeError(f"runtime authority file mismatch: {path}")
        rows[relative] = {
            "path": str(path.resolve()),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
            "mode": stat.filemode(path.stat().st_mode),
        }
    return {
        "root": str(root.resolve()),
        "directories": {
            str(path.relative_to(root) or "."): stat.filemode(path.stat().st_mode)
            for path in required_directories
        },
        "files": rows,
        "exact_file_inventory": True,
        "writable_bits_absent": True,
    }


def candidate_authority() -> dict[str, Any]:
    head = subprocess.check_output(
        ("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"), text=True,
    ).strip()
    model_tree = subprocess.check_output(
        ("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD:src/gpuwrf"), text=True,
    ).strip()
    if model_tree != MODEL_TREE:
        raise RuntimeError(f"model tree changed: expected={MODEL_TREE} observed={model_tree}")
    return {
        "runner_head": head,
        "src_gpuwrf_tree": model_tree,
        "model_tree_exact": True,
    }


def _leaf_summary(carries: Mapping[str, Any], jax: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for domain, carry in sorted(carries.items()):
        leaves = jax.tree_util.tree_leaves(carry)
        rows[domain] = {
            "leaf_count": len(leaves),
            "array_leaf_count": sum(
                hasattr(leaf, "shape") and hasattr(leaf, "dtype") for leaf in leaves
            ),
            "shape_dtype_metadata_available": all(
                (hasattr(leaf, "shape") and hasattr(leaf, "dtype"))
                or isinstance(leaf, (bool, int, float, str, type(None)))
                for leaf in leaves
            ),
        }
    return rows


def run_preflight(run_dir: Path) -> dict[str, Any]:
    environment = validate_environment(os.environ)
    runtime_authority = validate_runtime_authority()
    candidate = candidate_authority()
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(run_dir)
    if not run_dir.parent.is_dir():
        raise FileNotFoundError(run_dir.parent)
    run_dir.mkdir()

    import jax
    from gpuwrf.contracts import state as state_module
    from scripts.v0234_corrected_ni_ordinary_bisection import load_corrected_tree

    devices = jax.devices()
    if jax.default_backend() != "cpu" or not devices or any(
        device.platform != "cpu" for device in devices
    ):
        raise RuntimeError(f"CPU-only backend requirement failed: {devices!r}")

    # Production State constructors deliberately fail closed without a GPU.
    # This preflight needs their allocation phase only so the unchanged host-side
    # domain/table loaders can execute.  Redirect the single centralized device
    # selector to the already authenticated CPU device; no model source is edited,
    # and no timestep or numerical comparison is attempted under this shim.
    original_device_selector = inspect.getsource(state_module._gpu_device)
    if (
        "device.platform == \"gpu\"" not in original_device_selector
        or "State.zeros requires a GPU device" not in original_device_selector
    ):
        raise RuntimeError("State._gpu_device source changed; CPU allocation shim refused")
    cpu_device = devices[0]

    def _preflight_cpu_device() -> Any:
        return cpu_device

    state_module._gpu_device = _preflight_cpu_device
    allocation_shim = {
        "target": "gpuwrf.contracts.state._gpu_device",
        "purpose": "allocation-only host dependency-closure preflight",
        "original_source_sha256": hashlib.sha256(
            original_device_selector.encode()
        ).hexdigest(),
        "replacement_device": str(cpu_device),
        "process_local_only": True,
        "model_source_edited": False,
        "numerical_or_timestep_claimed": False,
    }

    started = time.monotonic()
    _tree, names, carries, dt_by_domain, metadata = load_corrected_tree(run_dir)
    elapsed = time.monotonic() - started
    if tuple(names) != ("d01", "d02", "d03"):
        raise RuntimeError(f"domain order changed: {names!r}")
    if set(carries) != set(names):
        raise RuntimeError(f"carry inventory changed: {sorted(carries)!r}")
    if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
    forbidden_outputs = (
        run_dir / "runtime" / "unused-output",
        run_dir / "runtime" / "unused-pipeline-proof",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden_outputs):
        raise RuntimeError(f"loader unexpectedly emitted output: {forbidden_outputs!r}")

    load_metadata = metadata.get("load_metadata") or {}
    return {
        "schema": SCHEMA,
        "verdict": "CPU_DOMAIN_LOAD_DEPENDENCY_CLOSURE_PASSED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "operation": "load_corrected_tree",
            "cpu_only": True,
            "gpu_command_or_query": False,
            "timestep_advanced": False,
            "forecast_output_emitted": False,
            "hard_timeout_seconds_external": 1200,
        },
        "environment_authority": environment,
        "candidate_authority": candidate,
        "runtime_authority": runtime_authority,
        "cpu_allocation_shim": allocation_shim,
        "runtime": {
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
            "default_backend": jax.default_backend(),
            "devices": [str(device) for device in devices],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
            "ionice": subprocess.check_output(
                ("ionice", "-p", str(os.getpid())), text=True,
            ).strip(),
        },
        "load": {
            "run_dir": str(run_dir.resolve()),
            "elapsed_seconds": elapsed,
            "domain_order": list(names),
            "dt_by_domain": dt_by_domain,
            "run_start": metadata.get("run_start"),
            "nests": metadata.get("nests"),
            "load_metadata_keys": sorted(load_metadata),
            "carry_metadata": _leaf_summary(carries, jax),
            "forbidden_output_paths_absent": [str(path) for path in forbidden_outputs],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    args = parser.parse_args()
    if args.proof_output.exists() or args.proof_output.is_symlink():
        raise FileExistsError(args.proof_output)
    try:
        payload = run_preflight(args.run_dir)
        rc = 0
    except BaseException as exc:
        payload = {
            "schema": SCHEMA,
            "verdict": "CPU_DOMAIN_LOAD_DEPENDENCY_CLOSURE_FAILED",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {
                "operation": "load_corrected_tree",
                "cpu_only_requested": True,
                "timestep_advanced": False,
                "forecast_output_requested": False,
                "hard_timeout_seconds_external": 1200,
            },
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        rc = 1
    atomic_write_self_hashed(args.proof_output, payload)
    print(json.dumps({
        "verdict": payload["verdict"],
        "proof_output": str(args.proof_output.resolve()),
        "proof_sha256": json.loads(args.proof_output.read_text())["proof_sha256"],
    }, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
