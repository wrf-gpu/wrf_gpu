"""Shared comparator helpers for M6B savepoint parity scripts.

Extracted from `scripts/m6b0r_jax_vs_wrf_compare.py`,
`scripts/m6b1_advance_mu_t_compare.py`, and
`scripts/m6b2_tridiag_solve_compare.py` by sprint
`2026-05-25-m6b-ladder-hygiene-cleanup` (Stage 4).

Goals:
- Single tolerance formula (`field_tolerance`) — was triplicated.
- Single per-field report dict (`field_compare`) — was duplicated.
- A canonical CLI argument parser (`build_compare_argparser`) so future
  comparator scripts default to the same flags (`--tier`, `--steps`,
  `--source-wrfout`, `--savepoint-root`, `--output`).

No external state; pure-numpy helpers. Comparator scripts should call these
instead of redefining `_threshold` / `_field_compare` locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def field_tolerance(entry: dict[str, Any], expected: np.ndarray) -> float:
    """Compute the per-field pass tolerance from a tolerance-ladder entry.

    Mirrors the historical triplicated `_threshold` helper exactly:
    `tol = max(abs, rel * max(|expected|_max, 1))`.

    Parameters
    ----------
    entry : dict
        One field entry from `tolerance_ladder.json` (must contain `abs` and
        `rel` keys; `None` is treated as 0.0).
    expected : np.ndarray
        The expected (WRF) values, used to compute a robust scale.
    """

    abs_tol = float(entry["abs"]) if entry.get("abs") is not None else 0.0
    rel_tol = float(entry["rel"]) if entry.get("rel") is not None else 0.0
    scale = float(np.nanmax(np.abs(expected))) if expected.size else 1.0
    return max(abs_tol, rel_tol * max(scale, 1.0))


def field_compare(
    name: str,
    got: np.ndarray,
    expected: np.ndarray,
    ladder: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical per-field comparator report dict.

    Mirrors the historical duplicated `_field_compare` helpers from M6B1/M6B2
    so existing comparator JSON payloads remain byte-identical when scripts
    are migrated to import from this module.
    """

    got_arr = np.asarray(got)
    exp_arr = np.asarray(expected)
    common_shape = tuple(min(a, b) for a, b in zip(exp_arr.shape, got_arr.shape))
    slices = tuple(slice(0, dim) for dim in common_shape)
    delta = got_arr[slices] - exp_arr[slices]
    if delta.size == 0:
        max_abs = float("nan")
        location = tuple(0 for _ in delta.shape)
    else:
        max_abs = float(np.nanmax(np.abs(delta)))
        flat_index = int(np.nanargmax(np.abs(delta)))
        location = np.unravel_index(flat_index, delta.shape)
    entry = dict(ladder["fields"][name])
    tol = field_tolerance(entry, exp_arr[slices])
    passed = bool(
        exp_arr.shape == got_arr.shape
        and np.isfinite(max_abs)
        and max_abs <= tol
    )
    return {
        "max_abs_delta": max_abs,
        "tolerance": tol,
        "passed": passed,
        "location": [int(item) for item in location],
        "expected_shape": list(exp_arr.shape),
        "actual_shape": list(got_arr.shape),
        "units": entry["units"],
        "dtype": entry["dtype"],
        "abs_threshold": entry["abs"],
        "rel_threshold": entry["rel"],
        "ulp_threshold": entry["ulp"],
    }


# Default Gen2 canary path; comparator scripts override via --source-wrfout.
DEFAULT_GEN2_WRFOUT = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l3/"
    "20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_2026-05-22_00:00:00"
)


def build_compare_argparser(
    *,
    operator_choices: tuple[str, ...] | None = None,
    tier_choices: tuple[str, ...] = ("column", "patch16", "golden", "all"),
    default_steps: int = 1,
    default_source_wrfout: Path | None = DEFAULT_GEN2_WRFOUT,
    default_output: Path | None = None,
    default_savepoint_root: Path | None = None,
) -> argparse.ArgumentParser:
    """Return a comparator CLI parser with the standard M6B flag set.

    Individual scripts can extend the returned parser with more flags. Keeping
    the canonical set in one place prevents the M6B0-R/M6B1/M6B2/M6B3 drift
    flagged by the ladder-audit Part 4.
    """

    parser = argparse.ArgumentParser()
    if operator_choices is not None:
        parser.add_argument("--operator", choices=operator_choices, required=True)
    parser.add_argument("--tier", choices=tier_choices, required=True)
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument(
        "--source-wrfout",
        type=Path,
        default=default_source_wrfout,
        help="Path to canary wrfout slice (overrides hardcoded Gen2 default).",
    )
    parser.add_argument(
        "--savepoint-root",
        type=Path,
        default=default_savepoint_root,
        help="Output directory for HDF5 savepoints (per-tier subdirs).",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    return parser
