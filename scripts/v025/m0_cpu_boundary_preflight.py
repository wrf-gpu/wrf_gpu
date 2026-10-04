#!/usr/bin/env python3
"""Execute the real FAST argument/lower boundary in a fresh CPU-only child.

This program is intentionally separate from the later CPU-WRF comparator.  It
binds the WRF authority before importing JAX or gpuwrf, executes the native
RRTMG-LW parser, builds the exact production arguments, and lowers the exact
production JIT.  It never compiles or invokes the lowered program.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import resource
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

SCHEMA = "wrf_gpu2.v025.m0.cpu_real_boundary_preflight.v1"


class CpuBoundaryRefusal(RuntimeError):
    """The real CPU boundary was not observed exactly."""


def validate_device_free_environment(environ: dict[str, str]) -> None:
    if environ.get("JAX_PLATFORMS") != "cpu":
        raise CpuBoundaryRefusal("JAX_PLATFORMS must equal cpu")
    if environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CpuBoundaryRefusal("CUDA_VISIBLE_DEVICES must be empty")
    forbidden_lock = sorted(
        key for key in environ if key.startswith("GPUWRF_GPU_LOCK_")
    )
    if forbidden_lock:
        raise CpuBoundaryRefusal(
            f"lock environment leaked into CPU preflight: {forbidden_lock}"
        )


def _callable_identity(function: Callable[..., Any]) -> dict[str, Any]:
    unwrapped = inspect.unwrap(function)
    source = Path(inspect.getsourcefile(unwrapped) or "").resolve()
    if not source.is_file():
        raise CpuBoundaryRefusal(f"callable source is absent: {function!r}")
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "wrapper_type": f"{type(function).__module__}.{type(function).__qualname__}",
        "unwrapped_module": unwrapped.__module__,
        "unwrapped_qualname": unwrapped.__qualname__,
        "source_path": str(source),
        "source_sha256": _sha256_file(source),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _bundle_identity(jax: Any, bundle: Any) -> dict[str, Any]:
    leaves, treedef = jax.tree_util.tree_flatten(bundle)
    fields: list[str] = []
    if dataclasses.is_dataclass(bundle):
        fields = [field.name for field in dataclasses.fields(bundle)]
    elif hasattr(bundle, "_fields"):
        fields = list(bundle._fields)
    leaf_records = [
        {
            "shape": [int(item) for item in getattr(leaf, "shape", ())],
            "dtype": str(getattr(leaf, "dtype", type(leaf).__name__)),
        }
        for leaf in leaves
    ]
    structural = {
        "type": f"{type(bundle).__module__}.{type(bundle).__qualname__}",
        "fields": fields,
        "field_count": len(fields),
        "leaf_count": len(leaves),
        "treedef": str(treedef),
        "leaves": leaf_records,
    }
    structural["sha256"] = hashlib.sha256(
        json.dumps(
            structural, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return structural


def _configured_trip_count(*, hours: float, dt_s: float) -> dict[str, Any]:
    """Derive the exact configured count without a rounded/assumed denominator."""

    hours_fraction = Fraction(str(float(hours)))
    timestep_fraction = Fraction(str(float(dt_s)))
    if hours_fraction <= 0 or timestep_fraction <= 0:
        raise CpuBoundaryRefusal(
            f"forecast interval and timestep must be positive: {hours=}, {dt_s=}"
        )
    forecast_seconds = hours_fraction * 3600
    quotient = forecast_seconds / timestep_fraction
    if quotient.denominator != 1:
        raise CpuBoundaryRefusal(
            "configured forecast interval is not an exact integer number of "
            f"timesteps: seconds={forecast_seconds}, dt_s={timestep_fraction}"
        )
    return {
        "derivation": "Fraction(str(hours))*3600/Fraction(str(dt_s))",
        "hours": float(hours),
        "forecast_interval_seconds": float(forecast_seconds),
        "timestep_seconds": float(timestep_fraction),
        "exact_steps": int(quotient),
        "integral": True,
    }


def execute(
    *, authority_path: Path, run_dir: Path, hours: float = 1.0
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    rss_before = _rss_bytes()

    # This must be the first project import: it refuses if JAX/gpuwrf appeared.
    import wrf_source_authority as wsa

    authority = wsa.assert_child_binding()
    validate_device_free_environment(dict(os.environ))

    import cpu_guard  # noqa: F401
    import jax
    import m0_exact_boundary_child as exact
    from gpuwrf.physics import rrtmg_lw

    platforms = sorted({device.platform for device in jax.devices()})
    if platforms != ["cpu"]:
        raise CpuBoundaryRefusal(f"non-CPU backend observed: {platforms}")

    observations = {
        "native_loader_calls": 0,
        "fast_argument_builder_calls": 0,
        "wrapper_preparation_calls": 0,
        "exact_lower_calls": 0,
        "compile_calls": 0,
        "device_invocations": 0,
        "receipt_reads": 0,
        "ledger_reads": 0,
        "ledger_writes": 0,
        "lock_checks": 0,
        "wrapper_calls": 0,
    }

    loader = rrtmg_lw._native_lw_tables
    loader_identity = _callable_identity(loader)
    bundle = loader()
    observations["native_loader_calls"] += 1
    bundle_identity = _bundle_identity(jax, bundle)

    raw, case, verified = exact.load_fast_call_arguments(
        run_dir=run_dir,
        hours=hours,
        cpu_device_adapter=True,
    )
    observations["fast_argument_builder_calls"] += 1
    prepared, preparation = exact.prepare_exact_call(raw, verified)
    observations["wrapper_preparation_calls"] += 1
    jit = verified["functions"][exact.JIT_FUNCTION]
    lowered = jit.lower(*prepared)
    observations["exact_lower_calls"] += 1
    lowered_identity = exact._stablehlo_identity(lowered)
    configured_count = _configured_trip_count(
        hours=float(prepared[2]),
        dt_s=float(prepared[1].dt_s),
    )
    lowered_count = lowered_identity.get("integration_trip_count") or {}
    if lowered_count.get("method") != "exact-lowered-trip-count":
        raise CpuBoundaryRefusal(
            "lowered program did not expose exact-lowered-trip-count"
        )
    if lowered_count.get("steps") != configured_count["exact_steps"]:
        raise CpuBoundaryRefusal(
            "exact lowered integration count differs from configured "
            "forecast-interval/timestep count: "
            f"{lowered_count.get('steps')} != {configured_count['exact_steps']}"
        )
    lowered_count["configured_cross_check"] = configured_count
    lowered_count["count_matches_configuration"] = True

    required = {
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
    }
    if observations != required:
        raise CpuBoundaryRefusal(
            f"real-boundary observation mismatch: {observations!r}"
        )
    lw_file = next(
        item
        for item in authority["files"]
        if item["relpath"] == "phys/module_ra_rrtmg_lw.F"
    )
    if bundle_identity["field_count"] != 29:
        raise CpuBoundaryRefusal(
            "native RRTMG-LW bundle does not have the expected 29 fields"
        )

    ended_ns = time.monotonic_ns()
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "device_action": False,
        "platforms": platforms,
        "authority": {
            "path": str(Path(authority_path).resolve()),
            "authority_sha256": authority["authority_sha256"],
            "root": authority["root"],
            "lw_file_binding": lw_file,
        },
        "callables": {
            "native_loader": loader_identity,
            "argument_builder": _callable_identity(
                exact.load_fast_call_arguments
            ),
            "argument_preparation": _callable_identity(
                exact.prepare_exact_call
            ),
            "exact_jit": verified["identities"][exact.JIT_FUNCTION],
        },
        "native_bundle": bundle_identity,
        "exact_case": {
            "run_dir": str(Path(run_dir).resolve()),
            "hours": float(hours),
            "case_sha256": exact.canonical_sha256(case),
            "wrapper_preparation_sequence": preparation,
            "argument_structure": exact._leaf_semantics(jax, prepared),
            "configured_trip_count": configured_count,
        },
        "lowered_program": lowered_identity,
        "observations": observations,
        "timing": {
            "clock": "time.monotonic_ns",
            "start_ns": started_ns,
            "end_ns": ended_ns,
            "wall_seconds": (ended_ns - started_ns) / 1e9,
            "outside_future_gpu_clocks": True,
        },
        "peak_rss": {
            "source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "before_bytes": rss_before,
            "peak_bytes": _rss_bytes(),
        },
    }
    payload["boundary_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = execute(
            authority_path=args.authority, run_dir=args.run_dir
        )
    except Exception as exc:  # noqa: BLE001 - terminal refusal transcript
        payload = {
            "schema": SCHEMA,
            "status": "REFUSED",
            "device_action": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
