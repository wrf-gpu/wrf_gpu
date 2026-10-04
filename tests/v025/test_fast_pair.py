"""The fail-closed guarantees of the FAST-v025 paired-run harness.

These tests exercise every way a pair can be incomplete, using stub arms. No
GPU work is launched: contract §13 forbids touching the GPU before the CPU-first
checkpoint exists, and the point of these tests is precisely that the harness
refuses to produce a result without a real pair.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v025"))

from run_fast_pair import (  # noqa: E402
    ArmRecord,
    PairIncompleteError,
    run_fast_pair,
)

T0 = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


def make_arm(
    kind: str,
    run_id: str,
    *,
    status: str = "OK",
    offset_s: int = 0,
    duration_s: int = 60,
    wrfout: str | None = "AUTO",
) -> ArmRecord:
    start = T0 + timedelta(seconds=offset_s)
    return ArmRecord(
        kind=kind,
        run_id=run_id,
        started_at_utc=start.isoformat(),
        finished_at_utc=(start + timedelta(seconds=duration_s)).isoformat(),
        status=status,
        payload={"final_wrfout_path": wrfout} if wrfout else {},
    )


def stub_comparator(cpu: ArmRecord, gpu: ArmRecord) -> dict:
    return {
        "cpu_run_id": cpu.run_id,
        "gpu_run_id": gpu.run_id,
        "variables_compared": 3,
        "per_variable": {},
        "worst_rmse_variable": "T",
        "total_nonfinite": 0,
    }


def ok_pair(tmp_path, **overrides):
    cpu = overrides.pop("cpu", make_arm("cpu", "p_cpu", offset_s=0, duration_s=60))
    gpu = overrides.pop("gpu", make_arm("gpu", "p_gpu", offset_s=120, duration_s=60))

    for arm in (cpu, gpu):
        if arm.payload.get("final_wrfout_path") != "AUTO":
            continue
        path = tmp_path / f"{arm.kind}-{arm.run_id}.nc"
        content = f"immutable fixture for {arm.kind} {arm.run_id}\n".encode()
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        arm.payload.update(
            {
                "final_wrfout_path": str(path),
                "final_wrfout_sha256": digest,
            }
        )
        if arm.kind == "gpu":
            arm.payload.update(
                {
                    "wrfout_binding_sha256": "b" * 64,
                    "result_exact_value_sha256": "c" * 64,
                    "exact_run_id": arm.run_id,
                }
            )
    return run_fast_pair(
        run_root=tmp_path,
        cpu_arm=lambda **kw: cpu,
        gpu_arm=lambda **kw: gpu,
        comparator=overrides.pop("comparator", stub_comparator),
        pair_id="testpair",
        **overrides,
    )


def test_complete_pair_returns_a_result(tmp_path):
    result = ok_pair(tmp_path)
    assert result["complete"] is True
    assert result["pair_id"] == "testpair"
    assert result["cpu_arm"]["run_id"] == "p_cpu"
    assert result["gpu_arm"]["run_id"] == "p_gpu"
    assert result["arms_non_overlapping"] is True
    assert result["comparison"]["variables_compared"] == 3


def test_failed_cpu_arm_fails_closed(tmp_path):
    """The headline rule: no fresh CPU arm, no result."""
    with pytest.raises(PairIncompleteError, match="fresh CPU arm"):
        ok_pair(tmp_path, cpu=make_arm("cpu", "p_cpu", status="FAILED"))


def test_failed_gpu_arm_fails_closed(tmp_path):
    with pytest.raises(PairIncompleteError, match="GPU arm did not complete"):
        ok_pair(tmp_path, gpu=make_arm("gpu", "p_gpu", status="FAILED", offset_s=120))


def test_overlapping_timed_arms_are_rejected(tmp_path):
    """A GPU arm timed while 12 MPI ranks saturate the box measures neither."""
    with pytest.raises(PairIncompleteError, match="overlapped"):
        ok_pair(
            tmp_path,
            cpu=make_arm("cpu", "p_cpu", offset_s=0, duration_s=100),
            gpu=make_arm("gpu", "p_gpu", offset_s=50, duration_s=100),
        )


def test_adjacent_non_overlapping_arms_are_accepted(tmp_path):
    """Back-to-back is fine; only genuine overlap is not."""
    result = ok_pair(
        tmp_path,
        cpu=make_arm("cpu", "p_cpu", offset_s=0, duration_s=60),
        gpu=make_arm("gpu", "p_gpu", offset_s=60, duration_s=60),
    )
    assert result["complete"] is True


def test_identical_run_ids_are_rejected(tmp_path):
    with pytest.raises(PairIncompleteError, match="same run id"):
        ok_pair(
            tmp_path,
            cpu=make_arm("cpu", "same"),
            gpu=make_arm("gpu", "same", offset_s=120),
        )


def test_comparison_keyed_to_other_arms_is_rejected(tmp_path):
    """A cached comparison from a previous pair must not pass as this one's."""

    def stale(cpu, gpu):
        return {
            "cpu_run_id": "SOME_OLD_RUN",
            "gpu_run_id": gpu.run_id,
            "variables_compared": 3,
        }

    with pytest.raises(PairIncompleteError, match="not keyed to this invocation"):
        ok_pair(tmp_path, comparator=stale)


@pytest.mark.parametrize(
    "variables,nonfinite",
    [(0, 0), (3, 1), (True, 0), (3, False)],
)
def test_comparison_requires_positive_fields_and_zero_nonfinite(
    tmp_path, variables, nonfinite
):
    def incomplete(cpu, gpu):
        return {
            "cpu_run_id": cpu.run_id,
            "gpu_run_id": gpu.run_id,
            "variables_compared": variables,
            "total_nonfinite": nonfinite,
        }

    with pytest.raises(PairIncompleteError, match="finite field-parity"):
        ok_pair(tmp_path, comparator=incomplete)


def test_missing_wrfout_blocks_the_comparison(tmp_path):
    """A pair with no field output cannot emit a parity result, so it fails."""
    from run_fast_pair import compare_arms

    with pytest.raises(PairIncompleteError, match="gpu arm has no final wrfout"):
        ok_pair(
            tmp_path,
            gpu=make_arm("gpu", "p_gpu", offset_s=120, wrfout=None),
            comparator=compare_arms,
        )


def test_no_partial_success_path_exists(tmp_path):
    """There must be no way to get a dict back without the full pair.

    Every failure mode above raises; none returns a partially-populated result.
    This test pins the absence of a ``complete: False`` return, which is what a
    caller would otherwise be tempted to treat as "good enough".
    """
    for kwargs in (
        {"cpu": make_arm("cpu", "c", status="FAILED")},
        {"gpu": make_arm("gpu", "g", status="FAILED", offset_s=120)},
        {"cpu": make_arm("cpu", "x"), "gpu": make_arm("gpu", "x", offset_s=120)},
    ):
        with pytest.raises(PairIncompleteError):
            ok_pair(tmp_path, **kwargs)
