"""Fail-closed wrfbdy coverage for native runs (single-domain + nested, incl. --extend-run).

Without the check ``interpolate_boundary_leaf`` clips its time index and a run past the
last boundary record silently holds the last values (rv-restart A1). CPU, no model work.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gpuwrf.io.wrfbdy_coverage import BoundaryCoverageError, require_wrfbdy_coverage, wrfbdy_coverage

SWISS = Path(__file__).resolve().parents[3] / "examples/switzerland_d01"  # 8 x 3 h records -> 24 h


@pytest.fixture(autouse=True)
def _explicit_allocator(monkeypatch):
    # The driver/CLI setdefault the allocator env; keep it test-local (scratch env: tests/conftest.py).
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")


def _case(tmp_path, interval_seconds=None):
    case = tmp_path / "case"
    case.mkdir()
    text = (SWISS / "namelist.input").read_text()
    if interval_seconds is not None:
        assert " interval_seconds                    = 10800," in text
        text = text.replace(" interval_seconds                    = 10800,",
                            f" interval_seconds                    = {interval_seconds},")
    (case / "namelist.input").write_text(text)
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        (case / name).symlink_to(SWISS / name)
    return case


def test_swiss_coverage_matches_wrf_boundary_metadata():
    info = wrfbdy_coverage(SWISS)
    assert (info["records"], info["interval_seconds"], info["coverage_seconds"]) == (8, 10800.0, 86400.0)
    assert (info["run_start"], info["last_boundary_time"]) == ("2023-01-15_00:00:00", "2023-01-16_00:00:00")
    assert require_wrfbdy_coverage(SWISS, 24)["coverage_seconds"] == 86400.0
    with pytest.raises(BoundaryCoverageError, match=r"25 h exceeds .*wrfbdy_d01: 24 h .*2023-01-16_00:00:00"):
        require_wrfbdy_coverage(SWISS, 25)
    assert wrfbdy_coverage(SWISS.parent / "missing-case") is None  # loader refuses a missing wrfbdy itself


@pytest.mark.parametrize("interval, hours_ok, hours_bad", [(21600, 24, 25), (3600, 8, 9)])
def test_coverage_is_the_tighter_of_leaf_span_and_wrf_metadata(tmp_path, interval, hours_ok, hours_bad):
    # 6 h namelist interval: the leaf axis would span 48 h but WRF's boundary data ends at 24 h.
    # 1 h namelist interval: the leaf axis spans only 8 records x 1 h, the clamp would start at 8 h.
    case = _case(tmp_path, interval)
    assert require_wrfbdy_coverage(case, hours_ok)["coverage_seconds"] == hours_ok * 3600.0
    with pytest.raises(BoundaryCoverageError):
        require_wrfbdy_coverage(case, hours_bad)


def _stub_cli(monkeypatch):
    from gpuwrf.integration import daily_pipeline as daily
    from gpuwrf.integration import nested_pipeline as shared
    calls = []
    monkeypatch.setattr(shared, "execute_nested_pipeline",
                        lambda config: calls.append(config) or {"verdict": "PIPELINE_GREEN", "wrfout_files": [], "metadata": {}})
    monkeypatch.setattr(daily, "execute_daily_pipeline", lambda config: pytest.fail("native run fell back to daily"))
    return calls


def test_cli_refuses_uncovered_end_before_any_model_work(tmp_path, monkeypatch, capsys):
    from gpuwrf import cli
    calls = _stub_cli(monkeypatch)
    base = ["run", "--input-dir", str(SWISS), "--output-dir", str(tmp_path / "out"),
            "--scratch-dir", str(tmp_path / "scratch")]
    assert cli.main([*base, "--hours", "25"]) == 2
    assert "exceeds the lateral boundary coverage of wrfbdy_d01" in capsys.readouterr().err
    assert cli.main([*base, "--hours", "30", "--resume-checkpoint", str(tmp_path / "gen"), "--extend-run"]) == 2
    assert "exceeds the lateral boundary coverage" in capsys.readouterr().err
    assert cli.main([*base, "--hours", "25", "--dry-run"]) == 2
    assert cli.main([*base, "--hours", "25", "--max-dom", "2", "--dry-run"]) == 2  # nested root d01
    assert "exceeds the lateral boundary coverage" in capsys.readouterr().err
    assert calls == []
    assert cli.main([*base, "--hours", "24"]) == 0 and calls[-1].hours == 24
    capsys.readouterr()
    assert cli.main([*base, "--hours", "24", "--max-dom", "2", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["run_type"] == "nested_live"


@pytest.mark.parametrize("max_dom", [1, 2])
def test_driver_refuses_uncovered_end_before_loading(tmp_path, monkeypatch, max_dom):
    """Library callers fail closed too (also a resumed --extend-run), before any domain load."""
    import gpuwrf.integration.nested_pipeline as pipeline

    class Reached(Exception):
        pass

    def loader(*a, **k):
        raise Reached("loader reached")

    monkeypatch.setattr(pipeline, "_load_domains", loader)
    config = pipeline.NestedPipelineConfig(SWISS, tmp_path / "out", tmp_path / "proof", 25, max_dom,
                                           emit_initial_history=max_dom == 1)
    with pytest.raises(BoundaryCoverageError, match="exceeds the lateral boundary coverage"):
        pipeline.execute_nested_pipeline(config)
    extended = pipeline.NestedPipelineConfig(SWISS, tmp_path / "out", tmp_path / "proof", 25, max_dom,
                                             resume_checkpoint=tmp_path / "gen", extend_end_time=True)
    with pytest.raises(BoundaryCoverageError):
        pipeline.execute_nested_pipeline(extended)
    with pytest.raises(Reached):
        pipeline.execute_nested_pipeline(pipeline.NestedPipelineConfig(SWISS, tmp_path / "out", tmp_path / "proof", 24, max_dom))
    assert not os.path.lexists(tmp_path / "out" / "wrfout_d01_2023-01-15_00:00:00")
