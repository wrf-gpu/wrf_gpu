from __future__ import annotations

from gpuwrf.runtime import operational_mode as om


def test_single_scan_knob_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("GPUWRF_SINGLE_SCAN", raising=False)
    assert om._single_scan_forecast_enabled() is False
    monkeypatch.setenv("GPUWRF_SINGLE_SCAN", "1")
    assert om._single_scan_forecast_enabled() is True
    monkeypatch.setenv("GPUWRF_SINGLE_SCAN", "true")
    assert om._single_scan_forecast_enabled() is True
    monkeypatch.setenv("GPUWRF_SINGLE_SCAN", "0")
    assert om._single_scan_forecast_enabled() is False


def test_run_forecast_operational_dispatches_to_selected_launcher(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(om, "_assert_nonzero_initial_mu_total", lambda state: None)
    monkeypatch.setattr(om, "_operational_scan_state", lambda state, namelist: ("scan", state))
    monkeypatch.setattr(om, "_dealias_pytree_buffers", lambda state: ("dealias", state))

    def default_launcher(state, namelist, hours):
        calls.append("default")
        return ("default", state, namelist, hours)

    def single_launcher(state, namelist, hours):
        calls.append("single")
        return ("single", state, namelist, hours)

    monkeypatch.setattr(om, "_run_forecast_operational_jit", default_launcher)
    monkeypatch.setattr(om, "run_forecast_operational_single_scan", single_launcher)

    monkeypatch.delenv("GPUWRF_SINGLE_SCAN", raising=False)
    assert om.run_forecast_operational("state", "namelist", 1.0)[0] == "default"
    monkeypatch.setenv("GPUWRF_SINGLE_SCAN", "1")
    assert om.run_forecast_operational("state", "namelist", 2.0)[0] == "single"
    assert calls == ["default", "single"]
