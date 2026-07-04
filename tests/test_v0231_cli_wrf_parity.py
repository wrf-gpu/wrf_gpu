"""v0.23.1 WRF-parity CLI usability tests (CPU only; no GPU, no JAX forecast).

These cover the argument-resolution and offline-subcommand surface added so a WRF
user can drive ``gpuwrf`` the way ``wrf.exe`` reads ``namelist.input``:

* ``--hours`` defaults from ``&time_control`` (explicit flag always wins);
* ``--domain`` defaults to ``d01`` for single-domain runs (WRF's root domain);
* ``--domains-from-namelist`` resolves ``--max-dom`` from ``&domains max_dom``
  and conflicts with an explicit ``--max-dom``;
* ``--dry-run`` prints the effective plan and exits WITHOUT importing the heavy
  JAX/GPU forecast pipeline;
* ``gpuwrf namelist-support`` prints the scheme-support registry offline;
* the run payload carries the new effective-values metadata.

The heavy forecast path (``execute_daily_pipeline``) is never invoked; where a
run needs to reach the pipeline step, a fake module is injected (the same
technique as ``tests/test_cli.py``) so no JAX is imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gpuwrf.cli import (
    _namelist_forecast_hours,
    _namelist_support_registry,
    build_parser,
    main,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"

# A minimal RRTMG/Thompson namelist: every selected scheme is operationally
# wired AND needs no $GPUWRF_WRF_ROOT tables, so validation passes cleanly and
# no WRF_ROOT preflight note fires.
_PHYSICS = (
    "&physics\n mp_physics = 8,\n ra_lw_physics = 4,\n ra_sw_physics = 4,\n/\n"
)


def _write_case(
    tmp_path: Path,
    *,
    time_control: str = "",
    max_dom: int | None = None,
) -> Path:
    case = tmp_path / "case"
    case.mkdir(parents=True, exist_ok=True)
    text = ""
    if time_control:
        text += time_control
    if max_dom is not None:
        text += f"&domains\n max_dom = {max_dom},\n/\n"
    text += _PHYSICS
    (case / "namelist.input").write_text(text)
    return case


def _dry_run_plan(
    case: Path, out: Path, extra: list[str], capsys: pytest.CaptureFixture[str]
) -> dict:
    """Invoke ``gpuwrf run --dry-run`` in-process and return the parsed plan."""
    rc = main(
        [
            "run",
            "--input-dir",
            str(case),
            "--output-dir",
            str(out),
            "--dry-run",
            *extra,
        ]
    )
    assert rc == 0
    return json.loads(capsys.readouterr().out)


def _subprocess_cli(args: list[str], code: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet against a FRESH interpreter (proves import isolation)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


# --------------------------------------------------------------------------- #
# 1) --hours defaults from &time_control                                       #
# --------------------------------------------------------------------------- #
def test_hours_default_from_namelist_run_hours_and_days(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(
        tmp_path,
        time_control="&time_control\n run_days = 1,\n run_hours = 6,\n/\n",
    )
    plan = _dry_run_plan(case, tmp_path / "out", [], capsys)
    assert plan["effective_hours"] == 30  # 1*24 + 6
    assert plan["override_sources"]["hours"] == "namelist"


def test_hours_default_from_namelist_minutes_floor_never_below_one(
    tmp_path: Path,
) -> None:
    # 90 minutes -> 1.5 h -> floored to 1 (never below 1); 20 minutes -> 1.
    case = _write_case(
        tmp_path, time_control="&time_control\n run_minutes = 90,\n/\n"
    )
    assert _namelist_forecast_hours(case / "namelist.input") == 1
    case2 = _write_case(
        tmp_path / "b", time_control="&time_control\n run_minutes = 20,\n/\n"
    )
    assert _namelist_forecast_hours(case2 / "namelist.input") == 1


def test_hours_explicit_flag_overrides_namelist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(
        tmp_path, time_control="&time_control\n run_hours = 12,\n/\n"
    )
    plan = _dry_run_plan(case, tmp_path / "out", ["--hours", "2"], capsys)
    assert plan["effective_hours"] == 2
    assert plan["override_sources"]["hours"] == "flag"


def test_hours_fallback_to_one_when_namelist_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(tmp_path)  # no &time_control at all
    plan = _dry_run_plan(case, tmp_path / "out", [], capsys)
    assert plan["effective_hours"] == 1
    assert plan["override_sources"]["hours"] == "default"


def test_hours_helper_returns_none_when_no_duration(tmp_path: Path) -> None:
    case = _write_case(
        tmp_path, time_control="&time_control\n run_hours = 0,\n run_days = 0,\n/\n"
    )
    assert _namelist_forecast_hours(case / "namelist.input") is None


# --------------------------------------------------------------------------- #
# 2) --domain defaults to d01 for single-domain runs                          #
# --------------------------------------------------------------------------- #
def test_domain_defaults_to_d01_when_omitted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(tmp_path)
    plan = _dry_run_plan(case, tmp_path / "out", [], capsys)
    assert plan["effective_domain"] == "d01"
    assert plan["override_sources"]["domain"] == "default"


def test_domain_respected_when_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(tmp_path)
    plan = _dry_run_plan(case, tmp_path / "out", ["--domain", "d02"], capsys)
    assert plan["effective_domain"] == "d02"
    assert plan["override_sources"]["domain"] == "flag"


def test_parser_new_defaults_are_none() -> None:
    """The new resolution logic keys on None defaults for --hours/--domain."""
    args = build_parser().parse_args(
        ["run", "--input-dir", "in", "--output-dir", "out"]
    )
    assert args.hours is None
    assert args.domain is None
    assert args.max_dom is None
    assert args.domains_from_namelist is False
    assert args.dry_run is False


# --------------------------------------------------------------------------- #
# 3) --domains-from-namelist reads max_dom; conflicts with --max-dom          #
# --------------------------------------------------------------------------- #
def test_domains_from_namelist_reads_max_dom(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(tmp_path, max_dom=3)
    plan = _dry_run_plan(case, tmp_path / "out", ["--domains-from-namelist"], capsys)
    assert plan["effective_max_dom"] == 3
    assert plan["override_sources"]["max_dom"] == "namelist"
    assert plan["run_type"] == "nested_live"
    assert plan["init_mode"] == "standalone_native_init_nested"


def test_max_dom_default_is_one_even_with_multidomain_namelist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Backward-compat: a multi-domain namelist still runs SINGLE domain by
    default (max_dom stays 1) unless the operator opts in."""
    case = _write_case(tmp_path, max_dom=3)
    plan = _dry_run_plan(case, tmp_path / "out", [], capsys)
    assert plan["effective_max_dom"] == 1
    assert plan["namelist_max_dom"] == 3
    assert plan["override_sources"]["max_dom"] == "default"


def test_domains_from_namelist_conflicts_with_max_dom(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(tmp_path, max_dom=3)
    rc = main(
        [
            "run",
            "--input-dir",
            str(case),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-dom",
            "2",
            "--domains-from-namelist",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_explicit_max_dom_still_wins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(tmp_path, max_dom=3)
    plan = _dry_run_plan(case, tmp_path / "out", ["--max-dom", "2"], capsys)
    assert plan["effective_max_dom"] == 2
    assert plan["override_sources"]["max_dom"] == "flag"


# --------------------------------------------------------------------------- #
# 4) --dry-run prints the plan and does NOT run compute                       #
# --------------------------------------------------------------------------- #
def test_dry_run_prints_plan_with_all_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _write_case(
        tmp_path, time_control="&time_control\n run_hours = 4,\n/\n"
    )
    plan = _dry_run_plan(case, tmp_path / "out", [], capsys)
    for key in (
        "namelist_path",
        "namelist_max_dom",
        "effective_max_dom",
        "effective_domain",
        "effective_hours",
        "override_sources",
        "init_mode",
        "run_type",
    ):
        assert key in plan, f"missing {key} in dry-run plan"
    assert plan["dry_run"] is True
    assert plan["namelist_path"].endswith("namelist.input")


def test_dry_run_does_not_import_forecast_pipeline(tmp_path: Path) -> None:
    """--dry-run must be import-light: no JAX/GPU forecast pipeline module."""
    case = _write_case(tmp_path)
    out = tmp_path / "out"
    code = (
        "import sys\n"
        "from gpuwrf.cli import main\n"
        f"rc = main(['run','--input-dir',{str(case)!r},"
        f"'--output-dir',{str(out)!r},'--dry-run'])\n"
        "assert 'gpuwrf.integration.daily_pipeline' not in sys.modules, 'daily imported'\n"
        "assert 'gpuwrf.integration.nested_pipeline' not in sys.modules, 'nested imported'\n"
        "sys.exit(rc)\n"
    )
    res = _subprocess_cli([], code)
    assert res.returncode == 0, res.stderr
    plan = json.loads(res.stdout)
    assert plan["dry_run"] is True


# --------------------------------------------------------------------------- #
# 5) gpuwrf namelist-support prints without importing JAX                      #
# --------------------------------------------------------------------------- #
def test_namelist_support_prints_readable_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["namelist-support"])
    assert rc == 0
    out = capsys.readouterr().out
    # Every required physics key is present, with the status buckets.
    for key in (
        "mp_physics",
        "cu_physics",
        "bl_pbl_physics",
        "sf_sfclay_physics",
        "sf_surface_physics",
        "ra_lw_physics",
        "ra_sw_physics",
    ):
        assert key in out
    assert "operational" in out
    assert "reference-only" in out
    assert "fail-closed" in out


def test_namelist_support_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["namelist-support", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "GpuwrfNamelistSupport"
    mp = payload["keys"]["mp_physics"]["options"]
    # Thompson (mp=8) is operational; a well-known reference-only/fail-closed
    # code is present too (registry is non-trivial).
    by_code = {opt["code"]: opt for opt in mp}
    assert by_code[8]["status"] == "implemented"
    assert by_code[8]["status_label"] == "operational"


def test_namelist_support_registry_helper_is_offline() -> None:
    """The registry builder pulls only the physics catalog, not JAX."""
    reg = _namelist_support_registry()
    assert set(reg["keys"]) >= {"mp_physics", "ra_lw_physics", "sf_surface_physics"}


def test_namelist_support_does_not_import_forecast_pipeline() -> None:
    code = (
        "import sys\n"
        "from gpuwrf.cli import main\n"
        "rc = main(['namelist-support'])\n"
        "assert 'gpuwrf.integration.daily_pipeline' not in sys.modules, 'daily imported'\n"
        "assert 'gpuwrf.integration.nested_pipeline' not in sys.modules, 'nested imported'\n"
        "sys.exit(rc)\n"
    )
    res = _subprocess_cli([], code)
    assert res.returncode == 0, res.stderr
    assert "mp_physics" in res.stdout


# --------------------------------------------------------------------------- #
# 6) The run payload carries the new effective-values metadata                 #
# --------------------------------------------------------------------------- #
def _install_fake_daily_pipeline(
    monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    fake = types.ModuleType("gpuwrf.integration.daily_pipeline")

    class _Config:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    fake.DailyPipelineConfig = _Config
    fake.detect_init_mode = lambda config: "standalone_native_init"
    fake.execute_daily_pipeline = lambda config: dict(payload)
    monkeypatch.setitem(
        sys.modules, "gpuwrf.integration.daily_pipeline", fake
    )


def test_run_payload_contains_effective_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual (single-domain) run payload is augmented with the effective
    values + override sources -- verified via a fake pipeline (no JAX)."""
    _install_fake_daily_pipeline(
        monkeypatch, {"verdict": "PIPELINE_GREEN", "wrfout_files": []}
    )
    case = _write_case(
        tmp_path, time_control="&time_control\n run_hours = 5,\n/\n"
    )
    rc = main(
        [
            "run",
            "--input-dir",
            str(case),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["effective_hours"] == 5
    assert payload["effective_domain"] == "d01"
    assert payload["effective_max_dom"] == 1
    assert payload["namelist_max_dom"] == 1
    assert payload["namelist_path"].endswith("namelist.input")
    assert payload["override_sources"] == {
        "hours": "namelist",
        "domain": "default",
        "max_dom": "default",
    }
