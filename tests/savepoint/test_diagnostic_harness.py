"""Pytest entry point for the comprehensive diagnostic harness.

This is a thin wrapper around :mod:`gpuwrf.diagnostics.comprehensive_harness`
that asserts the harness itself functions end-to-end on a very short forecast
window. The point is to keep CI cheap (CPU, < 60s) while verifying that the
single ``diagnostic_report.json`` artifact is produced with the expected
schema.

The full 24h Canary GPU run is invoked manually via
``scripts/run_diagnostic_harness.py``; this test only verifies the harness
mechanics, not the operational fidelity of the forecast.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CANARY_RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")


def _skip_if_no_canary_data() -> None:
    if not CANARY_RUN_DIR.is_dir():
        pytest.skip(f"canary run dir not present: {CANARY_RUN_DIR}")


def _skip_if_no_gpu() -> None:
    """``State.zeros`` requires a JAX GPU backend; skip cleanly in CPU-only CI."""

    import jax

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("diagnostic harness requires JAX GPU backend; CPU-only environment detected")


def _run_harness_short(hours: float) -> dict:
    """Run the harness on a few steps and return the report dict.

    ``hours`` should be small (e.g. 60/3600 = 1 step) for fast CI; the goal
    here is schema validation, not operational validation.
    """

    _skip_if_no_canary_data()
    _skip_if_no_gpu()

    from gpuwrf.diagnostics.comprehensive_harness import (
        DIAGNOSTIC_FIELD_INDEX,
        DIAGNOSTIC_INVARIANTS,
        DIAGNOSTIC_OPERATORS,
        build_diagnostic_report,
        initial_diagnostic_accumulator,
        run_diagnostic_forecast,
    )
    from gpuwrf.integration.d02_replay import build_replay_case
    from gpuwrf.runtime.operational_mode import OperationalNamelist, _steps_for_hours
    import jax

    case = build_replay_case(CANARY_RUN_DIR, domain="d02")
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,  # disable radiation in CI smoke
        use_vertical_solver=True,
    )
    steps_total = _steps_for_hours(hours, float(namelist.dt_s))
    accumulator = initial_diagnostic_accumulator(steps_total)
    final_state, final_acc = run_diagnostic_forecast(
        state, namelist, accumulator, float(hours), diagnostic_on=True
    )
    jax.block_until_ready(final_state)

    report = build_diagnostic_report(
        accumulator=final_acc,
        namelist=namelist,
        steps_total=steps_total,
        run_config={
            "case_run_dir": str(CANARY_RUN_DIR),
            "domain": "d02",
            "hours": float(hours),
            "dt_s": 10.0,
            "steps_total": int(steps_total),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "rk_order": int(namelist.rk_order),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "diagnostic_on": True,
            "disable_guards": bool(namelist.disable_guards),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "platform": "cpu",
        },
        wrf_anchor_payload=None,
        commit="TEST",
        generated_utc=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        wall_seconds_total=0.0,
        wall_seconds_diagnostic_overhead=None,
    )
    return report.to_dict()


@pytest.fixture(scope="module")
def harness_report() -> dict:
    return _run_harness_short(hours=60.0 / 3600.0)  # 1 minute = 6 steps


def test_diagnostic_report_schema_top_level(harness_report: dict) -> None:
    assert harness_report["schema_version"] == "diagnostic-harness-1.0"
    assert harness_report["verdict"] == "DIAGNOSIS_PRODUCED"
    for key in (
        "headline_diagnosis",
        "first_failure_attribution",
        "operator_attribution_24h",
        "internal_consistency_24h",
        "wrf_anchor_comparison",
        "coupling_chain_audit",
        "next_sprint_recommendations",
        "run_config",
    ):
        assert key in harness_report, f"missing top-level key: {key}"


def test_diagnostic_report_operator_block(harness_report: dict) -> None:
    operators = harness_report["operator_attribution_24h"]
    expected = {
        "dycore_rk3",
        "dynamics_guards",
        "microphysics_thompson",
        "surface_layer",
        "mynn_pbl",
        "rrtmg",
        "lateral_boundary",
        "boundary_guards",
    }
    assert set(operators.keys()) == expected
    for name, info in operators.items():
        assert info["verdict"] in {"ACTIVE", "INACTIVE", "MISSING", "NOISY_ZERO", "PASSIVE_OK"}, (
            f"operator {name} has unknown verdict {info['verdict']}"
        )
        assert "mean_abs_delta_per_step" in info
        assert "max_abs_delta_per_step" in info


def test_diagnostic_report_invariants_block(harness_report: dict) -> None:
    invariants = harness_report["internal_consistency_24h"]
    expected = {
        "all_state_finite",
        "qv_nonnegative",
        "qc_nonnegative",
        "qr_nonnegative",
        "qi_nonnegative",
        "qs_nonnegative",
        "qg_nonnegative",
        "theta_in_bounds",
        "wind_in_bounds",
        "mu_nonnegative",
    }
    assert set(invariants.keys()) == expected
    for name, info in invariants.items():
        assert "violated" in info
        assert "first_violation_step" in info
        assert "violation_count" in info


def test_diagnostic_report_coupling_chains(harness_report: dict) -> None:
    chains = harness_report["coupling_chain_audit"]
    for chain_name, info in chains.items():
        assert info["verdict"] in {"ACTIVE", "INACTIVE", "BROKEN"}, (
            f"chain {chain_name} has unknown verdict {info['verdict']}"
        )
        assert "evidence" in info


def test_diagnostic_report_serializable(harness_report: dict, tmp_path: Path) -> None:
    out = tmp_path / "diagnostic_report.json"
    out.write_text(json.dumps(harness_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["verdict"] == "DIAGNOSIS_PRODUCED"
