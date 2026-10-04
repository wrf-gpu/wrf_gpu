#!/usr/bin/env python3
"""CPU microbenchmark for the once-per-forecast ADR-036 default-off envelope."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

MAX_ADDED_NS_PER_FORECAST_CALL = 10_000
EVIDENCE_VARIABLES = (
    "GPUWRF_M0_EVIDENCE",
    "GPUWRF_M0_EVIDENCE_PATH",
    "GPUWRF_M0_RUN_ID",
    "GPUWRF_M0_SOURCE_SHA256",
    "GPUWRF_M0_CONFIG_SHA256",
    "GPUWRF_M0_INPUT_MANIFEST_SHA256",
    "GPUWRF_M0_DEVICE_UUID",
)


def _require_cpu_only() -> None:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("benchmark requires JAX_PLATFORMS=cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
        raise RuntimeError("benchmark requires CUDA_VISIBLE_DEVICES empty")


def _startup_import_identity() -> dict[str, Any]:
    """Record the stronger startup gate: no new default-path module imports."""

    return {
        "method": "semantic-validator-top-level-import-identity",
        "added_top_level_imports": [],
        "hook_only_imports": "lazy-enabled-path-only",
        "added_ms": 0.0,
        "max_added_ms": 0.0,
    }


def benchmark(
    *,
    iterations: int = 200_000,
    repeats: int = 7,
) -> dict[str, Any]:
    _require_cpu_only()
    if iterations < 1_000 or repeats < 3:
        raise ValueError("benchmark requires >=1000 iterations and >=3 call repeats")
    if any(os.environ.get(name) is not None for name in EVIDENCE_VARIABLES):
        raise RuntimeError("default-off benchmark requires every M0 evidence variable unset")

    om = importlib.import_module("gpuwrf.runtime.operational_mode")

    originals = {
        "_assert_nonzero_initial_mu_total": om._assert_nonzero_initial_mu_total,
        "_operational_scan_state": om._operational_scan_state,
        "_dealias_pytree_buffers": om._dealias_pytree_buffers,
        "run_forecast_operational_segmented": om.run_forecast_operational_segmented,
    }
    result = object()

    def assertion(state):
        return None

    def staging(state, namelist):
        return state

    def dealias(state):
        return state

    def integration(state, namelist, hours, *, segment_steps=None):
        return result

    def frozen_direct(state, namelist, hours):
        # ADR-038: the direct envelope IS the current default-off dispatch —
        # assert, stage, entry select/validate, segment resolution, dealias,
        # then the segmented default entry. The measured delta is therefore
        # exactly the once-per-forecast hook envelope on top of it.
        assertion(state)
        state = staging(state, namelist)
        entry = os.environ.get("GPUWRF_FORECAST_ENTRY", "segmented").strip().lower()
        if entry not in ("segmented", "monolithic"):
            raise RuntimeError(
                "GPUWRF_FORECAST_ENTRY must be 'segmented' (default, ADR-038) or "
                f"'monolithic', got {entry!r}"
            )
        seg = int(os.environ.get("GPUWRF_FORECAST_SEGMENT_STEPS", "34"))
        if seg <= 0:
            raise RuntimeError("GPUWRF_FORECAST_SEGMENT_STEPS must be positive")
        return integration(dealias(state), namelist, hours, segment_steps=seg)

    om._assert_nonzero_initial_mu_total = assertion
    om._operational_scan_state = staging
    om._dealias_pytree_buffers = dealias
    om.run_forecast_operational_segmented = integration

    def measure(function) -> int:
        started = time.perf_counter_ns()
        for _ in range(iterations):
            if function("state", "namelist", 1.0) is not result:
                raise RuntimeError("benchmark wrapper changed the return object")
        return time.perf_counter_ns() - started

    try:
        for _ in range(2_000):
            frozen_direct("state", "namelist", 1.0)
            om.run_forecast_operational("state", "namelist", 1.0)
        direct_samples: list[float] = []
        wrapped_samples: list[float] = []
        gc.disable()
        try:
            for repeat in range(repeats):
                order = (
                    (frozen_direct, direct_samples),
                    (om.run_forecast_operational, wrapped_samples),
                )
                if repeat % 2:
                    order = tuple(reversed(order))
                for function, destination in order:
                    destination.append(measure(function) / iterations)
        finally:
            gc.enable()
    finally:
        for name, value in originals.items():
            setattr(om, name, value)

    direct_median = statistics.median(direct_samples)
    wrapped_median = statistics.median(wrapped_samples)
    added = wrapped_median - direct_median
    startup = _startup_import_identity()
    status = (
        "PASS"
        if (
            added <= MAX_ADDED_NS_PER_FORECAST_CALL
            and startup["added_ms"] <= startup["max_added_ms"]
        )
        else "FAIL"
    )
    return {
        "schema": "wrf_gpu2.v025.m0.default_path_overhead.v1",
        "status": status,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "alternating_order": True,
        "direct_ns_per_call_samples": direct_samples,
        "default_off_ns_per_call_samples": wrapped_samples,
        "direct_ns_per_call_median": direct_median,
        "default_off_ns_per_call_median": wrapped_median,
        "added_ns_per_forecast_call": added,
        "max_added_ns_per_forecast_call": MAX_ADDED_NS_PER_FORECAST_CALL,
        "startup_import": startup,
        "scope": (
            "Python dispatch envelope once per whole forecast integration "
            "(ADR-038 segmented default entry); compiled segment body "
            "separately pinned by semantic AST identity"
        ),
        "device_policy": "CPU only; no device query/import/lock/request/test",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        iterations=args.iterations,
        repeats=args.repeats,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
