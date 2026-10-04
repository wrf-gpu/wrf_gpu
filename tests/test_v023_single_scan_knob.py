"""Forecast dispatch after 4cf38ef5d retired GPUWRF_SINGLE_SCAN (ADR-038)."""

from __future__ import annotations

from gpuwrf.runtime import operational_mode as om


def test_run_forecast_operational_dispatches_to_selected_launcher(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(om, "_assert_nonzero_initial_mu_total", lambda state: None)
    monkeypatch.setattr(om, "_operational_scan_state", lambda state, namelist: state)
    monkeypatch.setattr(om, "_dealias_pytree_buffers", lambda state: ("dealias", state))
    monkeypatch.delenv(om._M0_EVIDENCE_FLAG, raising=False)

    def default_launcher(state, namelist, hours, *, segment_steps):
        calls.append(("segmented", state, namelist, hours, segment_steps))
        return "segmented"

    def monolithic_launcher(state, namelist, hours):
        calls.append(("monolithic", state, namelist, hours))
        return "monolithic"

    monkeypatch.setattr(om, "run_forecast_operational_segmented", default_launcher)
    monkeypatch.setattr(om, "_run_forecast_operational_jit", monolithic_launcher)

    monkeypatch.delenv("GPUWRF_FORECAST_ENTRY", raising=False)
    monkeypatch.delenv("GPUWRF_FORECAST_SEGMENT_STEPS", raising=False)
    assert om.run_forecast_operational("state", "namelist", 1.0) == "segmented"
    monkeypatch.setenv("GPUWRF_FORECAST_SEGMENT_STEPS", "7")
    assert om.run_forecast_operational("state", "namelist", 2.0) == "segmented"
    monkeypatch.setenv("GPUWRF_FORECAST_ENTRY", "monolithic")
    assert om.run_forecast_operational("state", "namelist", 3.0) == "monolithic"
    assert calls == [
        ("segmented", ("dealias", "state"), "namelist", 1.0, 34),
        ("segmented", ("dealias", "state"), "namelist", 2.0, 7),
        ("monolithic", ("dealias", "state"), "namelist", 3.0),
    ]
