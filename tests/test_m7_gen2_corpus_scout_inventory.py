"""Regression baseline for the M7 Gen2 corpus scout findings.

Locks in three facts that the scout's `recommendation.md` depends on:

1. The d02 modal/pinned mass shape is `(66, 159)`.
2. There are exactly three runs across `wrf_l[23]/` whose d02 history is
   24-h hourly complete and on the pinned grid.
3. Zero of those three fall inside the harness's default tier4 eligibility
   window (cycle <= ``DEFAULT_ENDING_CYCLE``, matching ``RUN_DIR_RE_L3``).

The test is skipped when the Gen2 read-only tree is not present (CI safety).
It uses ``netCDF4.Dataset`` for header-only reads per the sprint contract;
it never iterates variable arrays.

When the corpus grows past the current scout snapshot (the desired outcome
after Option D in ``recommendation.md`` lands), this test is expected to
fail and must be updated alongside the cycle-window bump.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from gpuwrf.paths import reference_path

GEN2_ROOTS = (
    reference_path("runs", "wrf_l3"),
    reference_path("runs", "wrf_l2"),
)
WRFOUT_RE = re.compile(
    r"^wrfout_(?P<dom>d0[1-5])_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$"
)
RUN_DIR_RE_L3 = re.compile(
    r"^(?P<cycle>\d{8}_\d{2}z)_l3_24h_(?P<created>\d{8}T\d{6}Z)$"
)
PINNED_END_CYCLE = "20260520_18z"
PINNED_HELDOUT = "20260519_18z"
PINNED_GRID_YX = (66, 159)

# Scout snapshot 2026-05-27.
EXPECTED_PINNED_GRID_COMPLETE_RUN_IDS = {
    "20260521_18z_l3_24h_20260522T133443Z",
    "20260524_18z_l3_24h_20260525T225640Z",
    "20260524_18z_l2_72h_20260525T225640Z",
}


def _gen2_available() -> bool:
    return all(root.exists() for root in GEN2_ROOTS)


pytestmark = pytest.mark.skipif(
    not _gen2_available(),
    reason="Gen2 read-only tree configured by WRF_GPU_REFERENCE_ROOT is not mounted",
)


def _domain_24h_hourly_complete(run_dir: Path, domain: str = "d02") -> tuple[bool, tuple[int, int] | None]:
    """Header-only check: does ``run_dir`` carry a 24-h hourly ``domain`` history?

    Returns ``(is_complete, mass_shape_yx)``. Shape is read from the first wrfout's
    header dimensions only; we never iterate variable arrays.
    """

    netCDF4 = pytest.importorskip("netCDF4")

    files = sorted(run_dir.glob(f"wrfout_{domain}_*"))
    files = [path for path in files if WRFOUT_RE.match(path.name)]
    if len(files) < 25:
        return False, None
    stamps = []
    for path in files:
        match = WRFOUT_RE.match(path.name)
        assert match is not None
        stamps.append(match.group("stamp"))
    stamps.sort()
    span_hours = _hours_between(stamps[0], stamps[-1])
    if span_hours < 24:
        return False, None
    if len({stamp for stamp in stamps[:25]}) < 25:
        return False, None
    with netCDF4.Dataset(files[0], "r") as dataset:
        dims = {name: int(len(dim)) for name, dim in dataset.dimensions.items()}
    shape = dims.get("south_north"), dims.get("west_east")
    if None in shape:
        return False, None
    return True, (int(shape[0]), int(shape[1]))


def _hours_between(first_stamp: str, last_stamp: str) -> int:
    from datetime import datetime

    fmt = "%Y-%m-%d_%H:%M:%S"
    delta = datetime.strptime(last_stamp, fmt) - datetime.strptime(first_stamp, fmt)
    return int(delta.total_seconds() // 3600)


def _walk_pinned_grid_complete_runs() -> list[tuple[str, str, tuple[int, int]]]:
    """Return ``[(run_id, root_label, shape), ...]`` for every pinned-grid-complete run."""

    found: list[tuple[str, str, tuple[int, int]]] = []
    for root in GEN2_ROOTS:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            is_complete, shape = _domain_24h_hourly_complete(child, domain="d02")
            if not is_complete or shape != PINNED_GRID_YX:
                continue
            found.append((child.name, root.name, shape))
    return found


def test_pinned_grid_complete_runs_match_scout_snapshot() -> None:
    """Exactly the three runs the scout identified are pinned-grid-complete on disk."""

    found = _walk_pinned_grid_complete_runs()
    run_ids = {entry[0] for entry in found}
    assert run_ids == EXPECTED_PINNED_GRID_COMPLETE_RUN_IDS, (
        f"pinned-grid-complete d02 24h runs drifted; "
        f"expected {sorted(EXPECTED_PINNED_GRID_COMPLETE_RUN_IDS)!r}, "
        f"got {sorted(run_ids)!r}"
    )


def test_pinned_d02_modal_mass_shape_is_66_by_159() -> None:
    """All complete pinned-grid d02 24h runs share mass shape (66, 159)."""

    found = _walk_pinned_grid_complete_runs()
    assert found, "expected at least one pinned-grid-complete d02 24h run on disk"
    shapes = {entry[2] for entry in found}
    assert shapes == {PINNED_GRID_YX}, f"unexpected d02 mass shapes: {shapes!r}"


def test_tier4_default_window_has_zero_eligible_pinned_complete_members() -> None:
    """Default DEFAULT_ENDING_CYCLE (20260520_18z) excludes every surviving pinned-grid-complete L3 24h member.

    This locks in the BLOCKED_CORPUS verdict against the harness's defaults.
    When Option D in ``recommendation.md`` grows the corpus, this assertion will
    start failing and the cycle window in ``src/gpuwrf/validation/tier4_probtest.py``
    must be bumped at the same time.
    """

    found = _walk_pinned_grid_complete_runs()
    eligible: list[str] = []
    for run_id, _root_label, _shape in found:
        match = RUN_DIR_RE_L3.match(run_id)
        if match is None:
            continue  # wrong directory pattern (e.g. l2_72h)
        cycle = match.group("cycle")
        if cycle > PINNED_END_CYCLE:
            continue
        if cycle == PINNED_HELDOUT:
            continue
        eligible.append(run_id)
    assert eligible == [], (
        "tier4 eligibility window unexpectedly contains pinned-grid-complete members "
        f"{eligible!r}; bump DEFAULT_ENDING_CYCLE in tier4_probtest.py and update this test."
    )


def test_default_m6_gen2_run_dir_is_now_wrfout_empty() -> None:
    """Latent breakage flagged in tester-report Gap 2: the M6 reference cycle dir was stripped.

    The constant ``gen2_accessor.DEFAULT_M6_GEN2_RUN_DIR`` names a dir whose
    ``wrfout_d02_*`` history is no longer on disk; any code path that walks
    that constant for actual variable data will fail. This test pins the
    breakage so the follow-up sprint can't forget about it.
    """

    reference = reference_path("runs", "wrf_l3", "20260520_18z_l3_24h_20260521T045847Z")
    if not reference.exists():
        pytest.skip("reference cycle dir not mounted")
    d02_files = list(reference.glob("wrfout_d02_*"))
    assert d02_files == [], (
        "M6 reference cycle wrfout_d02 history is back on disk — Gap 2 resolved; "
        "delete this regression marker."
    )
