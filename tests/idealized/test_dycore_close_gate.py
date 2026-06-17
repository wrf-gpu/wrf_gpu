"""Dycore CLOSE GATE — assert the idealized verdicts are PASS (not just runnable).

Sprint U (P0-5).  GPT pre-close P0 finding: the committed idealized tests assert
only ``verdict in {PASS, FAIL}``, so a regression to FAIL would still be green.
These tests assert ``verdict == "PASS"`` and archive the proof JSON, so a dycore
regression FAILS CI instead of being false-green.

The cases run through the unified operational dycore (``_physics_boundary_step``,
the same step ``run_forecast_operational`` calls; bitwise-verified in Sprint U
P0-1).  They skip only when no JAX GPU backend is visible (CPU-only dev box);
they must NOT be silently skipped in the GPU CI close gate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import jax

from gpuwrf.ic_generators.idealized import (
    run_density_current_case,
    run_warm_bubble_case,
)


CLOSE_GATE_PROOF_DIR = Path("proofs/sprintU/close_gate")


def _gpu_available() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def _archive(result, name: str, proof_dir: Path) -> Path:
    # Archive under a caller-supplied (pytest tmp) dir, NOT the committed
    # proofs/sprintU/close_gate tree. The PASS verdict below is the close-gate
    # signal; rerunning the gate must not re-dirty the canonical committed proofs
    # with fp/plot-byte regeneration noise (the worktree must stay clean on a GPU
    # suite run). The committed close-gate proofs remain the canonical record.
    proof_dir.mkdir(parents=True, exist_ok=True)
    dst = proof_dir / f"{name}_verdict.json"
    shutil.copyfile(result.proof_json, dst)
    return dst


@pytest.mark.close_gate
def test_warm_bubble_close_gate_passes(tmp_path) -> None:
    if not _gpu_available():
        pytest.skip("close gate requires a visible JAX GPU backend")
    result = run_warm_bubble_case(proof_dir=tmp_path / "f2", require_gpu=True)
    assert result.status == "RAN_TO_COMPLETION", result.status
    archived = _archive(result, "warm_bubble", tmp_path)
    payload = json.loads(archived.read_text())
    # CLOSE GATE: must be PASS, not merely runnable.
    assert result.verdict == "PASS", (
        f"warm bubble REGRESSED to {result.verdict}: "
        f"{ {k: v.get('passed') for k, v in result.checks.items()} }"
    )
    assert payload["verdict"] == "PASS"


@pytest.mark.close_gate
def test_density_current_close_gate_passes(tmp_path) -> None:
    if not _gpu_available():
        pytest.skip("close gate requires a visible JAX GPU backend")
    result = run_density_current_case(proof_dir=tmp_path / "f2", require_gpu=True)
    assert result.status == "RAN_TO_COMPLETION", result.status
    archived = _archive(result, "density_current", tmp_path)
    payload = json.loads(archived.read_text())
    # CLOSE GATE: must be PASS, not merely runnable.
    assert result.verdict == "PASS", (
        f"density current REGRESSED to {result.verdict}: "
        f"{ {k: v.get('passed') for k, v in result.checks.items()} }"
    )
    assert payload["verdict"] == "PASS"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "close_gate"]))
