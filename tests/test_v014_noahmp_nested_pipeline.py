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

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from gpuwrf.contracts.grid import DomainHierarchy
from gpuwrf.integration.nested_pipeline import (
    _PerDomainWrfoutWriter,
    _SUPPORTED_NESTED_LAND_OPTIONS,
    _domain_sf_surface_physics,
    _wrf_julian_yearlen,
)
from gpuwrf.runtime.domain_tree import run_domain_tree_callbacks


REAL_CANARY_L2 = Path("/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z")


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
