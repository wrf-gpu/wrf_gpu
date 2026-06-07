from __future__ import annotations

from pathlib import Path

import pytest

from gpuwrf.io.gen2_accessor import DEFAULT_M6_GEN2_RUN_DIR


# NOTE: this test reads DEFAULT_M6_GEN2_RUN_DIR (a specific Gen2 corpus run dir),
# so the skip must gate on THAT path, not just on the /mnt corpus root being
# mounted -- otherwise the run dir can be absent/relocated while /mnt exists and
# the test fails with "no wrfout_d02 history" instead of skipping.
pytestmark = pytest.mark.skipif(
    not DEFAULT_M6_GEN2_RUN_DIR.exists(),
    reason=f"Gen2 corpus run directory unavailable (not vendored): {DEFAULT_M6_GEN2_RUN_DIR}",
)


def test_default_m6_gen2_run_dir_has_d02_wrfouts() -> None:
    assert DEFAULT_M6_GEN2_RUN_DIR.name == "20260521_18z_l3_24h_20260522T133443Z"
    d02_files = sorted(DEFAULT_M6_GEN2_RUN_DIR.glob("wrfout_d02_*"))
    assert d02_files, f"{DEFAULT_M6_GEN2_RUN_DIR} has no wrfout_d02 history"
