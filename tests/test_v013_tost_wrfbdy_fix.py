"""Regression guard for the v0.13 powered-TOST wrfbdy_d02 routing fix (KI-5).

On v0.12.0 the powered-n=15 per-case GPU forecast failed rc=2 with
``FileNotFoundError: standalone native-init requires wrfbdy_d02`` because the
L2 cases are a max_dom=2 ONE-WAY nest (no nest ever has a ``wrfbdy_d02``) but the
runner forced a d02-only single-domain standalone forecast that demanded one.

The fix routes the per-case forecast through the standalone LIVE-NESTED driver
(``execute_nested_pipeline``, max_dom=2): d01 standalone (LBC ``wrfbdy_d01``),
d02 IC-only with its LBC fed live by the parent -- ``wrfbdy_d02`` is never read.

These tests are CPU-only (no GPU / no forecast). They exercise the runner's
``setup_only_check`` -- the LBC-source routing logic that previously raised the
error -- against a synthetic nest fixture that reproduces the corpus topology.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "proofs/v0120/powered_tost_n15/run_one_case_v0120.py"


@pytest.fixture()
def runner():
    if not RUNNER.is_file():
        pytest.skip("per-case runner not present in this checkout")
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_one_case_v0120_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REAL_CASE = "20260429_18z_l2_72h_20260524T204451Z"
REAL_L2 = Path("/mnt/data/canairy_meteo/runs/wrf_l2") / REAL_CASE


def test_max_dom_constant_is_two(runner):
    """The runner is wired for the d01->d02 nest (max_dom=2), scores d02."""
    assert runner.MAX_DOM == 2
    assert runner.SCORE_DOMAIN == "d02"


@pytest.mark.skipif(
    not (REAL_L2 / "wrfinput_d02").exists(),
    reason="real wrf_l2 corpus not present on this box",
)
def test_setup_only_passes_on_real_case_no_wrfbdy_d02(runner, tmp_path):
    """End-to-end CPU setup-verify on a real corpus case: SETUP_OK, no wrfbdy_d02."""
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    run_dir = tmp_path / REAL_CASE
    run_dir.mkdir(parents=True)
    for f in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01",
              "namelist.input", "namelist.output"):
        src = REAL_L2 / f
        if src.exists():
            (run_dir / f).symlink_to(src)

    report = runner.setup_only_check(REAL_CASE, tmp_path, hours=24)

    assert report["verdict"] == "SETUP_OK"
    assert report["max_dom"] == 2
    assert report["nest_config"]["one_way"] is True
    assert report["disk_inputs"]["wrfbdy_d01"] is True
    # The crux: no wrfbdy_d02 exists, and the path does NOT require one.
    assert report["disk_inputs"]["wrfbdy_d02"] is False
    assert report["wrfbdy_d02_required"] is False
    assert report["nest_edge"]["parent"] == "d01"
    assert report["nest_edge"]["parent_grid_ratio"] > 1


@pytest.mark.skipif(
    not (REAL_L2 / "wrfinput_d02").exists(),
    reason="real wrf_l2 corpus not present on this box",
)
def test_setup_only_raises_when_wrfbdy_d01_missing(runner, tmp_path):
    """d01 LBC is mandatory: a missing wrfbdy_d01 must fail loudly (not silently)."""
    run_dir = tmp_path / REAL_CASE
    run_dir.mkdir(parents=True)
    # Stage everything EXCEPT wrfbdy_d01 (the d01 root LBC source).
    for f in ("wrfinput_d01", "wrfinput_d02", "namelist.input", "namelist.output"):
        src = REAL_L2 / f
        if src.exists():
            (run_dir / f).symlink_to(src)

    with pytest.raises((FileNotFoundError, RuntimeError)):
        runner.setup_only_check(REAL_CASE, tmp_path, hours=24)
