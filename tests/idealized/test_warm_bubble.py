from __future__ import annotations

from pathlib import Path

import pytest

from gpuwrf.ic_generators.idealized import run_warm_bubble_case


PROOF_DIR = Path("proofs/f2")


@pytest.mark.close_gate
def test_warm_bubble_runs_or_writes_blocked_proof() -> None:
    result = run_warm_bubble_case(proof_dir=PROOF_DIR, require_gpu=True)

    assert result.proof_json.exists()
    assert result.proof_markdown.exists()
    assert result.checks

    if result.status == "BLOCKED_GPU_UNAVAILABLE":
        pytest.skip("warm-bubble dycore run requires a visible JAX GPU backend")

    assert result.status == "RAN_TO_COMPLETION"
    # CLOSE GATE (Sprint U P0-5): the dycore must PASS, not merely run.  A
    # regression to FAIL fails CI instead of being false-green.
    assert result.verdict == "PASS", (
        f"warm bubble regressed to {result.verdict}: "
        f"{ {k: v.get('passed') for k, v in result.checks.items()} }"
    )
    assert {"theta_prime_max_500s", "max_abs_w_500s", "thermal_rise_500s", "relative_mass_drift"} <= set(
        result.checks
    )
    assert result.plot_paths
    assert all(path.exists() for path in result.plot_paths)
