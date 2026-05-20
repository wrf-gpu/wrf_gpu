from __future__ import annotations

from pathlib import Path

from gpuwrf.validation.tier1_thompson import run_tier1


def test_m5_thompson_tier1_parity_passes():
    record = run_tier1()
    assert record["pass"] is True
    assert record["tolerances_met"] is True
    assert record["scenarios_tested"] >= 3


def test_fixture_oracle_uses_wrf_tendency_ledger_not_kernel_sequence():
    script = Path("scripts/m5_generate_thompson_fixture.py").read_text(encoding="utf-8")
    kernel = Path("src/gpuwrf/physics/thompson_column.py").read_text(encoding="utf-8")
    assert "Path-B-strict oracle: WRF-style tendency ledger" in script
    assert "qvten = -clap / dt" in script
    assert "qvten" not in kernel


def test_attempt1_compact_relaxation_terms_are_absent():
    combined = (
        Path("scripts/m5_generate_thompson_fixture.py").read_text(encoding="utf-8")
        + Path("src/gpuwrf/physics/thompson_column.py").read_text(encoding="utf-8")
    )
    rejected_terms = (
        "autoconv_source",
        "1.0 - np.exp(-dt / 900.0)",
        "1.0 - jnp.exp(-float(dt) / 900.0)",
        "warm_fraction =",
        "rain_freeze_fraction",
        "0.25 * supersat",
        "0.25 * subsat",
    )
    for term in rejected_terms:
        assert term not in combined
