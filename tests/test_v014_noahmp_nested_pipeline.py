"""CPU unit gates for the v0.14 nested Noah-MP land-surface activation.

The v0.14 Canary h24 residual review proved the standalone live-nested pipeline
never enabled an LSM: ``nested_pipeline._make_namelist`` omitted ``use_noahmp``
/ ``sf_surface_physics``, so the land tile stayed on the prescribed bulk path
and land TSK was FROZEN for the whole run on both domains
(proofs/v014/canary_h24_residual_adjudication.md).  These tests pin the fix's
CPU-provable wiring (no GPU, no forecast):

* per-domain ``sf_surface_physics`` resolution (max-dom list / scalar / default);
* the FAIL-CLOSED rejection of unsupported land options (no silent bulk fallback);
* the WRF Noah-MP clock (0-based FRACTIONAL julian + leap-aware yearlen);
* the domain-tree ``wants_carry`` output opt-in (the wrfout writer must see the
  evolved Noah-MP carry, not just the post-step ``State``);
* on a box with the real corpus: both Canary L2 domains resolve to Noah-MP.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from gpuwrf.contracts.grid import DomainHierarchy
from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
from gpuwrf.integration.nested_pipeline import (
    _PerDomainWrfoutWriter,
    _SUPPORTED_NESTED_LAND_OPTIONS,
    _domain_sf_surface_physics,
    _nested_async_output_from_env,
    _wrf_julian_yearlen,
)
from gpuwrf.runtime.domain_tree import run_domain_tree_callbacks


REAL_CANARY_L2 = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z")


class _StubRun:
    """Minimal namelist-only stand-in for ``Gen2Run`` (no disk / device access)."""

    def __init__(self, physics: dict):
        self.namelist = {"physics": physics}


# ---------------------------------------------------------------------------
# Per-domain sf_surface_physics resolution
# ---------------------------------------------------------------------------

def test_supported_options_are_bulk_and_noahmp_only():
    assert _SUPPORTED_NESTED_LAND_OPTIONS == (0, 4)


def test_noahmp_option_resolves_per_domain_from_max_dom_list():
    run = _StubRun({"sf_surface_physics": [4, 4]})
    assert _domain_sf_surface_physics(run, "d01") == 4
    assert _domain_sf_surface_physics(run, "d02") == 4


def test_scalar_option_applies_to_every_domain():
    run = _StubRun({"sf_surface_physics": 4})
    assert _domain_sf_surface_physics(run, "d01") == 4
    assert _domain_sf_surface_physics(run, "d03") == 4


def test_mixed_per_domain_options_resolve_independently():
    run = _StubRun({"sf_surface_physics": [4, 0]})
    assert _domain_sf_surface_physics(run, "d01") == 4
    assert _domain_sf_surface_physics(run, "d02") == 0


def test_missing_option_defaults_to_bulk_zero():
    assert _domain_sf_surface_physics(_StubRun({}), "d01") == 0


@pytest.mark.parametrize("option", [1, 2, 3, 5, 7, 8])
def test_unsupported_land_option_fails_closed(option):
    """No silent bulk fallback: anything but 0/4 must raise with the reason."""
    run = _StubRun({"sf_surface_physics": [option, option]})
    with pytest.raises(ValueError, match="not wired"):
        _domain_sf_surface_physics(run, "d01")
    # The error must name the frozen-land hazard so the rejection is actionable.
    with pytest.raises(ValueError, match="frozen"):
        _domain_sf_surface_physics(run, "d02")


# ---------------------------------------------------------------------------
# WRF Noah-MP clock
# ---------------------------------------------------------------------------

def test_wrf_julian_is_zero_based_fractional_day_of_year():
    # 2026-05-01 18:00 UTC: tm_yday = 121 -> WRF julian = 120 + 18/24 = 120.75
    julian, yearlen = _wrf_julian_yearlen(
        datetime(2026, 5, 1, 18, 0, 0, tzinfo=timezone.utc)
    )
    assert julian == pytest.approx(120.75)
    assert yearlen == 365.0


def test_wrf_yearlen_honours_leap_years():
    julian, yearlen = _wrf_julian_yearlen(
        datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    )
    assert julian == pytest.approx(60.0)  # 2024-03-01 is yday 61 -> 0-based 60
    assert yearlen == 366.0


# ---------------------------------------------------------------------------
# Domain-tree wants_carry output opt-in
# ---------------------------------------------------------------------------

def _single_domain_run(output):
    hierarchy = DomainHierarchy.from_edges(("d01",), (), max_dom=5)
    carry = SimpleNamespace(state="post-step-state", noahmp_land="evolved-land")
    return run_domain_tree_callbacks(
        hierarchy,
        {"d01": carry},
        root_steps=1,
        advance=lambda name, c, start, n: c,
        output=output,
        output_cadence_steps={"d01": 1},
        block_between=False,
    )


def test_output_callback_default_receives_state_only():
    received = []

    def output(name, step, payload):
        received.append(payload)

    _single_domain_run(output)
    assert received == ["post-step-state"]


def test_output_callback_with_wants_carry_receives_full_carry():
    received = []

    def output(name, step, payload):
        received.append(payload)

    output.wants_carry = True
    _single_domain_run(output)
    (payload,) = received
    assert payload.state == "post-step-state"
    assert payload.noahmp_land == "evolved-land"


def test_nested_wrfout_writer_opts_into_the_carry():
    """The nested writer must see carry.noahmp_land for the land diagnostics."""
    assert _PerDomainWrfoutWriter.wants_carry is True


# ---------------------------------------------------------------------------
# Real Canary L2 corpus (skipped where the corpus is not mounted)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REAL_CANARY_L2 / "namelist.input").exists(),
    reason="real wrf_l2 corpus not present on this box",
)
def test_real_canary_case_selects_noahmp_on_both_domains():
    from gpuwrf.io.gen2_accessor import Gen2Run

    run = Gen2Run(REAL_CANARY_L2)
    assert _domain_sf_surface_physics(run, "d01") == 4
    assert _domain_sf_surface_physics(run, "d02") == 4


# ---------------------------------------------------------------------------
# v0.20 async history output lever: env default + byte-identity of the writer split
# ---------------------------------------------------------------------------

def test_nested_async_output_default_is_on(monkeypatch):
    monkeypatch.delenv("GPUWRF_NESTED_ASYNC_OUTPUT", raising=False)
    assert _nested_async_output_from_env() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "FALSE", "Off"])
def test_nested_async_output_disable_tokens(monkeypatch, value):
    monkeypatch.setenv("GPUWRF_NESTED_ASYNC_OUTPUT", value)
    assert _nested_async_output_from_env() is False


@pytest.mark.parametrize("value", ["1", "true", "on", "yes", "", "garbage"])
def test_nested_async_output_enable_tokens(monkeypatch, value):
    monkeypatch.setenv("GPUWRF_NESTED_ASYNC_OUTPUT", value)
    assert _nested_async_output_from_env() is True


def _make_nested_writer(*, async_writer):
    """Build a _PerDomainWrfoutWriter without its Gen2Run-dependent __init__.

    The async-output split lives entirely in __call__; this stubs only the host
    attributes that split reads (no corpus, no device, no Gen2Run static latlon).
    """
    from gpuwrf.integration.daily_pipeline import (
        _merge_output_diagnostics,
        _surface_diagnostics_for_output,
        finite_summary,
    )

    writer = object.__new__(_PerDomainWrfoutWriter)
    writer._surface_diagnostics_for_output = _surface_diagnostics_for_output
    writer._merge_output_diagnostics = _merge_output_diagnostics
    writer._finite_summary = finite_summary
    writer.writer_diagnostics = {}
    writer.writer_static_latlon_metadata = {}
    writer._async_writer = async_writer
    writer._variable_subset = None  # full byte-identical default output
    writer._full_variable_set = False
    writer.written = {"d01": []}
    return writer


def _drive_nested_writer(writer, *, output_dir, dt_s, run_start):
    """Drive one __call__ on the nested writer with a synthetic plain-numpy case."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_m7_netcdf_writer import synthetic_case  # type: ignore

    state, grid, namelist = synthetic_case()
    bundle = SimpleNamespace(namelist=namelist, grid=grid)
    writer.output_dir = output_dir
    writer.run_start = run_start
    writer.bundles = {"d01": bundle}
    writer.dt_by_domain = {"d01": dt_s}
    return writer("d01", own_step=120, carry=state)


def test_nested_async_output_byte_identical_to_sync(tmp_path):
    """The async-submit branch writes byte-identical wrfout vs the sync branch.

    Drives the REAL _PerDomainWrfoutWriter.__call__ split: once with
    async_writer=None (synchronous write on the step thread) and once with an
    AsyncWrfoutWriter (device->host pull on the step thread, NetCDF write on the
    background thread). The two wrfout files must be bit-identical -- the lever
    changes only the wall-clock timing of the write, not its content.
    """
    from netCDF4 import Dataset

    run_start = datetime(2026, 5, 25, 18, tzinfo=timezone.utc)
    dt_s = 30.0

    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    sync_writer = _make_nested_writer(async_writer=None)
    sync_result = _drive_nested_writer(
        sync_writer, output_dir=sync_dir, dt_s=dt_s, run_start=run_start
    )

    async_dir = tmp_path / "async"
    async_dir.mkdir()
    aw = AsyncWrfoutWriter(max_pending=2)
    async_writer = _make_nested_writer(async_writer=aw)
    async_result = _drive_nested_writer(
        async_writer, output_dir=async_dir, dt_s=dt_s, run_start=run_start
    )
    # The split records the path at submit time (before the write lands); join
    # flushes the background write, mirroring execute_nested_pipeline.
    aw.join()

    # Same recorded output path basename + same summary payload.
    assert Path(sync_result["wrfout"]).name == Path(async_result["wrfout"]).name
    assert sync_result["all_finite"] == async_result["all_finite"]
    assert sync_writer.written["d01"] and async_writer.written["d01"]

    p_sync = Path(sync_result["wrfout"])
    p_async = Path(async_result["wrfout"])
    assert p_sync.exists() and p_async.exists()
    assert p_sync.read_bytes() == p_async.read_bytes()

    with Dataset(p_sync) as ds_a, Dataset(p_async) as ds_b:
        assert sorted(ds_a.variables) == sorted(ds_b.variables)
        for name in ds_a.variables:
            a = np.asarray(ds_a.variables[name][:])
            b = np.asarray(ds_b.variables[name][:])
            assert a.shape == b.shape, f"{name} shape differs"
            if np.issubdtype(a.dtype, np.floating):
                assert np.array_equal(a, b, equal_nan=True), f"{name} bytes differ"
            else:
                assert np.array_equal(a, b), f"{name} bytes differ"
