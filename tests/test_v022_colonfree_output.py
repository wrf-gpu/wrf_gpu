from datetime import datetime, timezone
from pathlib import Path

from gpuwrf.integration.daily_pipeline import _wrfout_name
from gpuwrf.integration.nested_pipeline import _wrfout_path
from gpuwrf.io.data_inventory import parse_wrfout_valid_time, wrfout_name


VALID_TIME = datetime(2024, 8, 6, 13, 0, 0, tzinfo=timezone.utc)


def test_wrfout_name_default_keeps_wrf_colons(monkeypatch):
    monkeypatch.delenv("GPUWRF_COLONFREE_OUTPUT", raising=False)

    name = wrfout_name("d02", VALID_TIME)

    assert name == "wrfout_d02_2024-08-06_13:00:00"
    assert _wrfout_name(VALID_TIME, "d02") == name
    assert _wrfout_path(Path("/tmp/out"), "d02", VALID_TIME).name == name
    assert parse_wrfout_valid_time(Path(name)) == VALID_TIME


def test_wrfout_name_env_flag_uses_dash_time(monkeypatch):
    monkeypatch.setenv("GPUWRF_COLONFREE_OUTPUT", "1")

    name = wrfout_name("d02", VALID_TIME)

    assert name == "wrfout_d02_2024-08-06_13-00-00"
    assert _wrfout_name(VALID_TIME, "d02") == name
    assert _wrfout_path(Path("/tmp/out"), "d02", VALID_TIME).name == name
    assert parse_wrfout_valid_time(Path(name)) == VALID_TIME
