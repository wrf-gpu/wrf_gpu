from __future__ import annotations

from pathlib import Path

import pytest

from gpuwrf.io.gen2_accessor import DEFAULT_M6_GEN2_RUN_DIR


pytestmark = pytest.mark.skipif(
    not Path("/mnt/data/canairy_meteo").exists(),
    reason="Gen2 read-only tree /mnt/data/canairy_meteo not mounted",
)


def test_default_m6_gen2_run_dir_has_d02_wrfouts() -> None:
    assert DEFAULT_M6_GEN2_RUN_DIR.name == "20260521_18z_l3_24h_20260522T133443Z"
    d02_files = sorted(DEFAULT_M6_GEN2_RUN_DIR.glob("wrfout_d02_*"))
    assert d02_files, f"{DEFAULT_M6_GEN2_RUN_DIR} has no wrfout_d02 history"
