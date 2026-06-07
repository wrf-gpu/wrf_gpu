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


def test_run_namelist_defaults_to_input_dir(tmp_path: Path) -> None:
    """v0.12.0: --namelist is optional and defaults to <input-dir>/namelist.input."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--input-dir", "in",
            "--output-dir", "out",
        ]
    )
    # Not supplied on the command line -> resolved inside _cmd_run to the case's own.
    assert args.namelist is None


def test_run_missing_default_namelist_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A case dir with no namelist.input and no --namelist fails closed (no traceback)."""
    case = tmp_path / "case"
    case.mkdir()  # deliberately no namelist.input
    rc = main(
        [
            "run",
            "--input-dir", str(case),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert "--namelist file not found" in capsys.readouterr().err


def test_run_scratch_dir_flag_parses(tmp_path: Path) -> None:
    """v0.12.0: --scratch-dir is accepted (disk-backed scratch override)."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--input-dir", "in",
            "--output-dir", "out",
            "--scratch-dir", str(tmp_path / "scratch"),
        ]
    )
    assert args.scratch_dir == tmp_path / "scratch"


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


@pytest.mark.parametrize(
    "section, alt_substring",
    [
        ("&physics\n ra_lw_physics = 1,\n/\n", "ra_lw_physics=4"),
        ("&physics\n ra_sw_physics = 1,\n/\n", "ra_sw_physics=4"),
    ],
)
def test_run_rejects_reference_only_radiation_pre_jax(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    section: str,
    alt_substring: str,
) -> None:
    """``gpuwrf run`` with a parity-proven-but-not-wired radiation scheme
    (RRTM/Dudhia) fails closed BEFORE any JAX import, naming RRTMG (=4) -- so the
    operational run never silently substitutes RRTMG for the requested scheme."""

    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(section)
    rc = main(
        [
            "run",
            "--namelist", str(case / "namelist.input"),
            "--input-dir", str(case),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2  # fail-closed before the heavy pipeline (exit 2, not a crash)
    err = capsys.readouterr().err
    assert "SILENTLY" in err
    assert "NOT operationally wired" in err
    assert alt_substring in err


def test_run_accepts_implemented_radiation_at_validation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operationally-wired RRTMG (=4) suite passes the pre-JAX validation gate.

    To keep this test cheap (no JAX), a fake ``daily_pipeline`` module is injected
    so the run gets PAST validation into the pipeline step without importing JAX;
    the sentinel proves validation did not reject the implemented suite.
    """

    import sys
    import types

    sentinel = "PIPELINE_REACHED"

    fake = types.ModuleType("gpuwrf.integration.daily_pipeline")

    class _Config:  # accepts the kwargs DailyPipelineConfig is called with
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    def _detect_init_mode(config: object) -> str:
        return "standalone_native_init"

    def _execute(config: object) -> dict:
        raise RuntimeError(sentinel)

    fake.DailyPipelineConfig = _Config
    fake.detect_init_mode = _detect_init_mode
    fake.execute_daily_pipeline = _execute
    monkeypatch.setitem(sys.modules, "gpuwrf.integration.daily_pipeline", fake)

    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(
        "&physics\n mp_physics = 8,\n ra_lw_physics = 4,\n ra_sw_physics = 4,\n/\n"
    )
    with pytest.raises(RuntimeError) as excinfo:
        main(
            [
                "run",
                "--namelist", str(case / "namelist.input"),
                "--input-dir", str(case),
                "--output-dir", str(tmp_path / "out"),
            ]
        )
    # Reached the pipeline => validation accepted the implemented RRTMG suite.
    assert str(excinfo.value) == sentinel
    err = capsys.readouterr().err
    assert "SILENTLY" not in err
    assert "NOT operationally wired" not in err
    assert "Unsupported namelist" not in err


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
