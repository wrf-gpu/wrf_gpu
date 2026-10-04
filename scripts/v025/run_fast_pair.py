#!/usr/bin/env python3
"""Fail-closed FAST-v025 paired-run harness (contract §5.2, §5.4).

The recurring-pair rule is the thing that keeps the whole rewrite honest: every
use of FAST-v025 must launch a **fresh CPU-WRF arm and a fresh GPU arm in the
same invocation** and emit a comparison for that invocation. A cached CPU result
may accelerate diagnostics but can never satisfy the rule.

This module makes that mechanical rather than procedural. ``run_fast_pair``
returns a result only if every one of the following happened *in this call*:

* a fresh CPU arm launched, with its own run id and its own directory;
* a fresh GPU arm launched, with its own run id;
* the two timed arms did not overlap in wallclock;
* a field/parity comparison was produced from those two specific arms.

Any missing piece raises ``PairIncompleteError``. There is deliberately no
``--skip-cpu``, no ``--reuse``, and no partial-success return path: the only way
to get a result object is to have produced the whole pair.

The GPU arm is injected as a callable. Phase A builds and tests this harness
with a stub arm and **never launches GPU work**; Phase B injects the real one
after dual-manager coordination and the canonical lock wrapper.
"""

from __future__ import annotations

import json
import hashlib
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_cpu_arm import run_arm as run_cpu_arm  # noqa: E402


class PairIncompleteError(RuntimeError):
    """Raised when a pair invocation did not produce every required component."""


@dataclass
class ArmRecord:
    """One timed arm of a pair."""

    kind: str  # "cpu" | "gpu"
    run_id: str
    started_at_utc: str
    finished_at_utc: str
    status: str
    payload: dict = field(default_factory=dict)

    def interval(self) -> tuple[datetime, datetime]:
        return (
            datetime.fromisoformat(self.started_at_utc),
            datetime.fromisoformat(self.finished_at_utc),
        )


class GpuArm(Protocol):
    """A GPU arm: runs FAST-v025 on the port and returns an ``ArmRecord``."""

    def __call__(self, *, run_id: str, run_root: Path) -> ArmRecord: ...


def _overlaps(a: ArmRecord, b: ArmRecord) -> bool:
    a0, a1 = a.interval()
    b0, b1 = b.interval()
    return a0 < b1 and b0 < a1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_output(record: ArmRecord) -> dict:
    """Bind an arm to the immutable wrfout the comparator actually opens."""

    nested = record.payload.get("wrfout")
    nested = nested if isinstance(nested, dict) else {}
    path_text = (
        record.payload.get("final_wrfout_path")
        or nested.get("final_wrfout_path")
    )
    expected_hash = (
        record.payload.get("final_wrfout_sha256")
        or nested.get("final_wrfout_sha256")
    )
    if not isinstance(path_text, str) or not path_text:
        raise PairIncompleteError(f"{record.kind} arm has no final wrfout path")
    path = Path(path_text)
    if path.is_symlink() or not path.is_file():
        raise PairIncompleteError(
            f"{record.kind} arm wrfout is missing/not a regular file: {path}"
        )
    observed_hash = _sha256_file(path)
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or observed_hash != expected_hash
    ):
        raise PairIncompleteError(
            f"{record.kind} arm wrfout hash does not match the arm record"
        )
    output = {
        "final_wrfout_path": str(path),
        "final_wrfout_bytes": path.stat().st_size,
        "final_wrfout_sha256": observed_hash,
    }
    if record.kind == "gpu":
        binding = (
            record.payload.get("wrfout_binding_sha256")
            or nested.get("binding_sha256")
        )
        result_hash = (
            record.payload.get("result_exact_value_sha256")
            or nested.get("result_exact_value_sha256")
        )
        exact_run_id = (
            record.payload.get("exact_run_id")
            or nested.get("run_id")
        )
        if (
            not isinstance(binding, str)
            or len(binding) != 64
            or not isinstance(result_hash, str)
            or len(result_hash) != 64
            or exact_run_id != record.run_id
        ):
            raise PairIncompleteError(
                "GPU wrfout lacks the exact run/result/output binding"
            )
        output.update(
            {
                "wrfout_binding_sha256": binding,
                "result_exact_value_sha256": result_hash,
                "exact_run_id": exact_run_id,
            }
        )
    return output


def compare_arms(cpu: ArmRecord, gpu: ArmRecord) -> dict:
    """Field/parity comparison for THIS invocation's two arms.

    Contract §5.4 requires a candidate-vs-fresh-CPU field metric every time the
    pair runs. The comparison is keyed to both run ids so a result can never be
    silently paired with a different arm than the one that produced it.
    """
    cpu_output = _verified_output(cpu)
    gpu_output = _verified_output(gpu)
    cpu_out = cpu_output["final_wrfout_path"]
    gpu_out = gpu_output["final_wrfout_path"]

    import netCDF4
    import numpy as np

    metrics: dict[str, dict] = {}
    shape_mismatches: dict[str, dict] = {}
    nonfloat_shared: list[str] = []
    empty_shared: list[str] = []
    with netCDF4.Dataset(cpu_out) as ds_cpu, netCDF4.Dataset(gpu_out) as ds_gpu:
        cpu_names = set(ds_cpu.variables)
        gpu_names = set(ds_gpu.variables)
        shared = sorted(cpu_names & gpu_names)
        for name in shared:
            a = np.asarray(ds_cpu.variables[name][:])
            b = np.asarray(ds_gpu.variables[name][:])
            if a.dtype.kind != "f" or b.dtype.kind != "f":
                nonfloat_shared.append(name)
                continue
            if a.shape != b.shape:
                shape_mismatches[name] = {
                    "cpu_shape": list(a.shape),
                    "candidate_shape": list(b.shape),
                }
                continue
            if a.size == 0:
                empty_shared.append(name)
                continue
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            denom = np.maximum(np.abs(a.astype(np.float64)), 1e-30)
            metrics[name] = {
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "max_abs": float(diff.max()),
                "max_rel": float((diff / denom).max()),
                "cpu_nonfinite": int((~np.isfinite(a)).sum()),
                "gpu_nonfinite": int((~np.isfinite(b)).sum()),
            }
    if not metrics:
        raise PairIncompleteError("no comparable float fields between the two arms")
    total_nonfinite = sum(
        m["cpu_nonfinite"] + m["gpu_nonfinite"] for m in metrics.values()
    )
    if total_nonfinite:
        raise PairIncompleteError(
            f"paired outputs contain {total_nonfinite} non-finite values"
        )
    return {
        "cpu_run_id": cpu.run_id,
        "gpu_run_id": gpu.run_id,
        "variables_compared": len(metrics),
        "per_variable": metrics,
        "worst_rmse_variable": max(metrics, key=lambda k: metrics[k]["rmse"]),
        "total_nonfinite": total_nonfinite,
        "coverage": {
            "cpu_variable_count": len(cpu_names),
            "candidate_variable_count": len(gpu_names),
            "shared_variable_count": len(shared),
            "comparable_float_variable_count": len(metrics),
            "cpu_only_variables": sorted(cpu_names - gpu_names),
            "candidate_only_variables": sorted(gpu_names - cpu_names),
            "nonfloat_shared_variables": nonfloat_shared,
            "empty_shared_variables": empty_shared,
            "shape_mismatches": shape_mismatches,
            "all_shared_float_shapes_match": not shape_mismatches,
        },
        "output_binding": {
            "cpu": cpu_output,
            "gpu": gpu_output,
        },
    }


def cpu_arm_runner(
    *,
    run_id: str,
    run_root: Path,
    bind_to_core: bool = True,
    cpu_list: str = "16-27",
) -> ArmRecord:
    """The real CPU arm: one fresh 12-rank FAST-v025 run."""
    flags = (
        ["--use-hwthread-cpus", "--bind-to", "core"]
        if bind_to_core
        else ["--use-hwthread-cpus", "--bind-to", "none"]
    )
    payload = run_cpu_arm(
        run_root=run_root,
        label=f"pair_cpu_{run_id}",
        mpi_flags=flags,
        max_dom=1,
        cpu_list=cpu_list,
    )
    if payload.get("status") == "OK":
        final_path = Path(payload["run_dir"]) / payload["final_wrfout"]
        payload["final_wrfout_path"] = str(final_path)
        if payload.get("final_wrfout_sha256") != _sha256_file(final_path):
            payload["status"] = "FAILED"
            payload["error"] = "CPU arm final wrfout hash changed after run"
    return ArmRecord(
        kind="cpu",
        run_id=run_id,
        started_at_utc=payload["started_at_utc"],
        finished_at_utc=payload["finished_at_utc"],
        status=payload.get("status", "FAILED"),
        payload=payload,
    )


def finalize_pair_records(
    *,
    cpu_record: ArmRecord,
    gpu_record: ArmRecord,
    pair_id: str,
    comparator: Callable[[ArmRecord, ArmRecord], dict] = compare_arms,
) -> dict:
    """Finalize two fresh records, including GPU-first W1 orchestration."""

    if cpu_record.status != "OK":
        raise PairIncompleteError(
            f"CPU arm did not complete (status={cpu_record.status}); "
            "a pair without a fresh CPU arm is invalid by contract §5.2"
        )
    if gpu_record.status != "OK":
        raise PairIncompleteError(
            f"GPU arm did not complete (status={gpu_record.status})"
        )
    if _overlaps(cpu_record, gpu_record):
        raise PairIncompleteError(
            "CPU and GPU timed arms overlapped in wallclock; contract §5.2 forbids it"
        )
    if cpu_record.run_id == gpu_record.run_id:
        raise PairIncompleteError("both arms reported the same run id")

    cpu_output = _verified_output(cpu_record)
    gpu_output = _verified_output(gpu_record)
    comparison = comparator(cpu_record, gpu_record)
    if comparison.get("cpu_run_id") != cpu_record.run_id or comparison.get(
        "gpu_run_id"
    ) != gpu_record.run_id:
        raise PairIncompleteError(
            "comparison is not keyed to this invocation's two arms"
        )
    variables = comparison.get("variables_compared")
    nonfinite = comparison.get("total_nonfinite")
    if (
        isinstance(variables, bool)
        or not isinstance(variables, int)
        or variables <= 0
        or isinstance(nonfinite, bool)
        or not isinstance(nonfinite, int)
        or nonfinite != 0
    ):
        raise PairIncompleteError(
            "comparison has no finite field-parity result"
        )
    return {
        "schema": "wrf_gpu2.v025.m0.fast_pair.v2",
        "pair_id": pair_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "cpu_arm": {
            "run_id": cpu_record.run_id,
            "started_at_utc": cpu_record.started_at_utc,
            "finished_at_utc": cpu_record.finished_at_utc,
            "status": cpu_record.status,
            "timing_classes": cpu_record.payload.get("timing_classes"),
            "content_digest": cpu_record.payload.get("content_digest"),
            "run_dir": cpu_record.payload.get("run_dir"),
            **cpu_output,
        },
        "gpu_arm": {
            "run_id": gpu_record.run_id,
            "started_at_utc": gpu_record.started_at_utc,
            "finished_at_utc": gpu_record.finished_at_utc,
            "status": gpu_record.status,
            "timing_classes": gpu_record.payload.get("timing_classes"),
            "content_digest": gpu_record.payload.get("content_digest"),
            **gpu_output,
        },
        "arms_non_overlapping": True,
        "comparison": comparison,
    }


def run_fast_pair(
    *,
    run_root: Path,
    gpu_arm: GpuArm,
    cpu_arm: Callable[..., ArmRecord] = cpu_arm_runner,
    pair_id: str | None = None,
    comparator: Callable[[ArmRecord, ArmRecord], dict] = compare_arms,
) -> dict:
    """Run one complete FAST-v025 pair. Fails closed on any missing component.

    The arms run strictly one after the other: a GPU arm timed while a 12-rank
    CPU job is saturating the box is not a measurement of either.
    """
    pair_id = pair_id or uuid.uuid4().hex[:12]
    run_root.mkdir(parents=True, exist_ok=True)

    cpu_record = cpu_arm(run_id=f"{pair_id}_cpu", run_root=run_root)
    if cpu_record.status != "OK":
        raise PairIncompleteError(
            f"CPU arm did not complete (status={cpu_record.status}); "
            "a pair without a fresh CPU arm is invalid by contract §5.2"
        )
    gpu_record = gpu_arm(run_id=f"{pair_id}_gpu", run_root=run_root)
    return finalize_pair_records(
        cpu_record=cpu_record,
        gpu_record=gpu_record,
        pair_id=pair_id,
        comparator=comparator,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.parse_args()
    raise SystemExit(
        "run_fast_pair has no CLI GPU arm yet: the real GPU arm is injected in "
        "Phase B after dual-manager coordination and scripts/with_gpu_lock.sh. "
        "Phase A tests this harness with a stub arm (tests/v025/test_fast_pair.py) "
        "and launches no GPU work."
    )


if __name__ == "__main__":
    raise SystemExit(main())
