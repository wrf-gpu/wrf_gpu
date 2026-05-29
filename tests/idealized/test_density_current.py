from __future__ import annotations

from pathlib import Path

import pytest

from gpuwrf.ic_generators.idealized import run_density_current_case


PROOF_DIR = Path("proofs/f2")


@pytest.mark.close_gate
def test_density_current_runs_or_writes_blocked_proof() -> None:
    result = run_density_current_case(proof_dir=PROOF_DIR, require_gpu=True)

    assert result.proof_json.exists()
    assert result.proof_markdown.exists()
    assert result.checks

    if result.status == "BLOCKED_GPU_UNAVAILABLE":
        pytest.skip("density-current dycore run requires a visible JAX GPU backend")

    assert result.status == "RAN_TO_COMPLETION"
    # CLOSE GATE (Sprint U P0-5): the dycore must PASS, not merely run.  A
    # regression to FAIL fails CI instead of being false-green.
    assert result.verdict == "PASS", (
        f"density current regressed to {result.verdict}: "
        f"{ {k: v.get('passed') for k, v in result.checks.items()} }"
    )
    assert {"theta_prime_min_900s", "max_abs_w_900s", "front_position_900s", "relative_mass_drift"} <= set(
        result.checks
    )
    assert result.plot_paths
    assert all(path.exists() for path in result.plot_paths)
