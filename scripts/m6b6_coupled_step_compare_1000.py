#!/usr/bin/env python
"""Run the M9.A 1000-step pure-dycore savepoint parity extension."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from gpuwrf.validation.comparator_common import DEFAULT_GEN2_WRFOUT


DEFAULT_OUTPUT = ROOT / "proofs/m9/savepoint_parity_1000.json"
DEFAULT_SAVEPOINT_ROOT = ROOT / ".agent/sprints/2026-05-28-m9a-trace-harness/savepoints_1000"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else str(scalar)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _device_summary() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "default_backend": jax.default_backend(),
        "visible_devices": [str(device) for device in devices],
        "gpu_devices": [str(device) for device in devices if device.platform == "gpu"],
    }


def _field_summary(m6b5: Any, results: list[dict[str, Any]], final_step: dict[str, Any]) -> tuple[dict[str, Any], float, int | None]:
    per_field: dict[str, Any] = {}
    first_divergence: int | None = None
    final_max = 0.0
    for field in m6b5.COMPARE_FIELDS:
        step_max = 0.0
        step_at_max = None
        final_stats = final_step["fields"][field]
        final_delta = float(final_stats["max_abs_delta"])
        final_max = max(final_max, final_delta)
        for row in results:
            stats = row["fields"][field]
            delta = float(stats["max_abs_delta"])
            if delta >= step_max:
                step_max = delta
                step_at_max = int(row["step"])
            if first_divergence is None and not bool(stats["passed"]):
                first_divergence = int(row["step"])
        per_field[field] = {
            "passed_all_steps": all(bool(row["fields"][field]["passed"]) for row in results),
            "max_abs_diff_over_depth": step_max,
            "step_of_max_abs_diff": step_at_max,
            "max_abs_diff_at_final_step": final_delta,
            "rmse_at_final_step": None,
            "argmax_diff_idx_at_final_step": final_stats["location"],
            "tolerance": final_stats["tolerance"],
            "expected_shape": final_stats["expected_shape"],
            "actual_shape": final_stats["actual_shape"],
            "units": final_stats["units"],
        }
    return per_field, final_max, first_divergence


def _compare_depth(m6b5: Any, tier: str, depth: int, savepoint_root: Path) -> dict[str, Any]:
    result = m6b5.compare_tier(tier, depth, savepoint_root)
    results = result["results"]
    final_step = results[-1]
    per_field, final_max, first_divergence = _field_summary(m6b5, results, final_step)
    return {
        "depth": int(depth),
        "status": "PASS" if bool(result["passed"]) else "FAIL",
        "first_divergence_step": first_divergence,
        "max_abs_diff_at_final_step": final_max,
        "per_field_summary": per_field,
        "tier": tier,
        "operator": "dycore_step",
        "source_script_pattern": "scripts/m6b5_dycore_step_compare.py",
        "source_wrfout": str(m6b5.SOURCE_WRFOUT),
        "savepoint_root": str(savepoint_root),
        "manifest": result["manifest"],
        "outcome": result["outcome"],
        "tolerance_ladder_path": result["tolerance_ladder_path"],
        "transfer_audit": result["transfer_audit"],
        "device_summary": _device_summary(),
    }


def _gpu_unavailable_payload(depth: int, source_wrfout: Path, start: float) -> dict[str, Any]:
    return {
        "depth": int(depth),
        "status": "FAIL",
        "first_divergence_step": None,
        "max_abs_diff_at_final_step": None,
        "per_field_summary": {},
        "wall_clock_seconds": time.perf_counter() - start,
        "commit": _git_commit(),
        "blocked_reason": "No JAX GPU backend is visible; 1000-step M9.A run is GPU-required by sprint contract.",
        "source_wrfout": str(source_wrfout),
        "device_summary": _device_summary(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("column", "patch16", "golden"), default="column")
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=None, help="Alias for --depth.")
    parser.add_argument("--source-wrfout", type=Path, default=DEFAULT_GEN2_WRFOUT)
    parser.add_argument("--savepoint-root", type=Path, default=DEFAULT_SAVEPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    depth = int(args.steps if args.steps is not None else args.depth)
    start = time.perf_counter()
    if not [device for device in jax.devices() if device.platform == "gpu"]:
        payload = _gpu_unavailable_payload(depth, args.source_wrfout, start)
        _write_json(args.output, payload)
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
        return 0

    from scripts import m6b5_dycore_step_compare as m6b5

    m6b5._set_source_wrfout(args.source_wrfout)
    payload = _compare_depth(m6b5, args.tier, depth, args.savepoint_root)
    payload["wall_clock_seconds"] = time.perf_counter() - start
    payload["commit"] = _git_commit()
    _write_json(args.output, payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
