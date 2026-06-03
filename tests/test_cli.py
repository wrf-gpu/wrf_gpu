"""Tests for the public ``gpuwrf`` CLI (no GPU / no forecast required).

These exercise only the cheap, fail-closed surface of ``gpuwrf run``: argument
validation, the namelist registry gate, and the standalone dimension comparator.
The heavy forecast path (``execute_daily_pipeline``) is intentionally not invoked
here -- that is the GPU clean-clone gate, run separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuwrf.cli import build_parser, compare_wrfout_dimensions, main


def _write_namelist(path: Path, mp_physics: int = 8) -> None:
    path.write_text(f"&physics\n mp_physics = {mp_physics},\n/\n")


def test_build_parser_has_run_subcommand() -> None:
    parser = build_parser()
    # --help is the only thing a naive user can rely on; ensure 'run' parses.
    args = parser.parse_args(
        [
            "run",
            "--namelist", "nl",
            "--input-dir", "in",
            "--output-dir", "out",
        ]
    )
    assert args.command == "run"
    assert args.hours == 1
    assert args.domain == "d02"
    assert args.score is False


def test_main_no_command_returns_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2


def test_run_missing_input_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "run",
            "--namelist", str(tmp_path / "namelist.input"),
            "--input-dir", str(tmp_path / "does_not_exist"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "--input-dir does not exist" in capsys.readouterr().err


def test_run_namelist_must_match_input_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = tmp_path / "case"
    case.mkdir()
    _write_namelist(case / "namelist.input")
    other = tmp_path / "other.input"
    _write_namelist(other)
    rc = main(
        [
            "run",
            "--namelist", str(other),
            "--input-dir", str(case),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "must be <input-dir>/namelist.input" in capsys.readouterr().err


def test_run_hours_must_be_positive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = tmp_path / "case"
    case.mkdir()
    _write_namelist(case / "namelist.input")
    rc = main(
        [
            "run",
            "--namelist", str(case / "namelist.input"),
            "--input-dir", str(case),
            "--output-dir", str(tmp_path / "out"),
            "--hours", "0",
        ]
    )
    assert rc == 2
    assert "--hours must be a positive integer" in capsys.readouterr().err


def test_run_unsupported_namelist_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = tmp_path / "case"
    case.mkdir()
    _write_namelist(case / "namelist.input", mp_physics=99)
    rc = main(
        [
            "run",
            "--namelist", str(case / "namelist.input"),
            "--input-dir", str(case),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "Unsupported namelist" in err
    assert "mp_physics" in err


# --------------------------------------------------------------------------- #
# Dimension comparator                                                        #
# --------------------------------------------------------------------------- #
def _write_nc(path: Path, dims: dict[str, int]) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path, "w") as ds:
        for name, length in dims.items():
            ds.createDimension(name, length)


def test_compare_dimensions_no_output() -> None:
    result = compare_wrfout_dimensions([], "/tmp")
    assert result["status"] == "NO_OUTPUT"


def test_compare_dimensions_pass(tmp_path: Path) -> None:
    dims = {"west_east": 10, "south_north": 8, "bottom_top": 5}
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    name = "wrfout_d02_2026-05-21_19:00:00"
    _write_nc(gen_dir / name, dims)
    _write_nc(ref_dir / name, dims)
    result = compare_wrfout_dimensions([gen_dir / name], ref_dir)
    assert result["status"] == "PASS"
    assert result["file_count"] == 1
    assert result["files"][0]["pass"] is True


def test_compare_dimensions_mismatch_fails(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    name = "wrfout_d02_2026-05-21_19:00:00"
    _write_nc(gen_dir / name, {"west_east": 10, "south_north": 8})
    _write_nc(ref_dir / name, {"west_east": 12, "south_north": 8})
    result = compare_wrfout_dimensions([gen_dir / name], ref_dir)
    assert result["status"] == "FAIL"
    assert result["files"][0]["mismatches"]


def test_compare_dimensions_missing_reference(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    gen_dir.mkdir()
    ref_dir.mkdir()
    name = "wrfout_d02_2026-05-21_19:00:00"
    _write_nc(gen_dir / name, {"west_east": 10})
    result = compare_wrfout_dimensions([gen_dir / name], ref_dir)
    assert result["status"] == "FAIL"
    assert result["files"][0]["status"] == "MISSING_REFERENCE"
