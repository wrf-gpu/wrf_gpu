"""Release history defaults: actual nested selection and emitted NetCDF inventories."""
from datetime import datetime, timezone

import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.config.history_output import full_wrfout_variables_enabled
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _full_wrfout_variables_enabled
from gpuwrf.integration.nested_pipeline import _PerDomainWrfoutWriter, _resolve_full_wrfout_variables
from gpuwrf.io.wrfout_writer import FULL_WRFOUT_VARIABLES
from test_initial_history_wrf_t0 import _writer


@pytest.mark.parametrize("canonical,alias,expected", [
    (None, None, True), ("", "", True), ("1", None, True), ("0", None, False),
    ("false", None, False), ("OFF", None, False), (" no ", None, False),
    (None, "0", False), (None, "1", True), ("0", "1", False), ("1", "0", True),
])
def test_full_history_resolution_and_precedence(monkeypatch, canonical, alias, expected):
    for name, value in (("GPUWRF_FULL_WRFOUT_VARIABLES", canonical), ("GPUWRF_FULL_WRFOUT", alias)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert full_wrfout_variables_enabled() is expected
    assert _resolve_full_wrfout_variables() is expected
    assert _full_wrfout_variables_enabled(DailyPipelineConfig()) is expected


@pytest.mark.parametrize("explicit,setting,expected", [
    (None, None, True), (False, None, False), (True, None, True),
    (True, "0", False), (False, "1", False), (True, "1", True),
])
def test_daily_api_choice_respects_resident_producer_opt_out(monkeypatch, explicit, setting, expected):
    monkeypatch.delenv("GPUWRF_FULL_WRFOUT", raising=False)
    if setting is None:
        monkeypatch.delenv("GPUWRF_FULL_WRFOUT_VARIABLES", raising=False)
    else:
        monkeypatch.setenv("GPUWRF_FULL_WRFOUT_VARIABLES", setting)
    assert _full_wrfout_variables_enabled(DailyPipelineConfig(full_wrfout_variables=explicit)) is expected


def test_default_full_and_explicit_reduced_retain_identical_common_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUWRF_FULL_WRFOUT", raising=False)
    monkeypatch.delenv("GPUWRF_TRAINING_OUTPUT_SUBSET", raising=False)
    snapshots = {}
    for setting in (None, "0"):
        if setting is None:
            monkeypatch.delenv("GPUWRF_FULL_WRFOUT_VARIABLES", raising=False)
        else:
            monkeypatch.setenv("GPUWRF_FULL_WRFOUT_VARIABLES", setting)
        root = tmp_path / ("full" if setting is None else "reduced")
        root.mkdir()
        # Exercise the real constructor's selection, then the existing host callback fixture.
        selected = _PerDomainWrfoutWriter(output_dir=root, input_dir=root,
            run_start=datetime(2026, 2, 28, tzinfo=timezone.utc), bundles={},
            output_cadence_steps={}, dt_by_domain={})
        writer, state = _writer(root, lambda *a, **k: None)
        writer._full_variable_set = selected._full_variable_set
        result = writer("d01", 0, state)
        with Dataset(result["wrfout"]) as ds:
            snapshots[setting] = {name: np.asarray(var[:]) for name, var in ds.variables.items()}
    full, reduced = snapshots[None], snapshots["0"]
    assert set(FULL_WRFOUT_VARIABLES) <= set(full)
    assert len(full) >= 375 and len(reduced) < 375
    assert set(reduced) < set(full)
    for name, value in reduced.items():
        np.testing.assert_array_equal(full[name], value, err_msg=name)
