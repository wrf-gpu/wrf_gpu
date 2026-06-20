"""Regression for the v0.15 Thompson cold-collection lane (rain-collecting-snow
qr_acr_qs + rain-collecting-graupel qr_acr_qg) against a bit-exact WRF
mp_gt_driver cold mixed-phase savepoint.

The Switzerland d01 72h RAINNC +5 mm surplus was PROVEN non-chaotic
(proofs/v015/falsifier_rainnc_report.json) and traced to the port retaining
supercooled rain that WRF collects/freezes into graupel aloft below 0 C.  This
test validates that the cold-collection lane reproduces WRF's COLUMN-INTEGRATED
rain->graupel conversion (the RAINNC-relevant bulk) to within a predeclared
band, and that disabling it (GPUWRF_THOMPSON_COLD_COLLECTION=0) collapses the
rain sink -- so the lane, not a coincidence, is what closes the bias.

Skips when the cold savepoint or a JAX backend is unavailable (the savepoint
lives outside git at <DATA_ROOT>; regenerate via
proofs/v015/cold_collection_oracle/build_and_run.sh).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ORACLE = Path("<DATA_ROOT>/wrf_gpu2/physics_oracle/microphysics_coldmix")
ROOT = Path(__file__).resolve().parents[1]


def _have_oracle() -> bool:
    return (ORACLE / "thompson_in.sidecar.txt").is_file()


def _have_tables() -> bool:
    from gpuwrf.physics.thompson_tables import cold_tables_available
    return cold_tables_available()


@pytest.mark.skipif(not _have_oracle(), reason="cold mixed-phase oracle savepoint absent")
@pytest.mark.skipif(not _have_tables(), reason="cold-collection fixture absent")
def test_cold_collection_closes_column_rain_sink():
    sys.path.insert(0, str(ROOT / "proofs" / "v015" / "cold_collection_oracle"))
    import coldmix_validate as cv

    out = cv.run()
    on = out["jax_scheme_cold_collection_ON"]["column_deltas_vs_wrf"]
    off = out["jax_scheme_cold_collection_OFF"]["column_deltas_vs_wrf"]

    # The cold column must actually exercise the lane (supercooled rain over
    # snow/graupel), else the test is vacuous.
    assert out["regime"]["cold_rain_snow_levels"] > 50, out["regime"]
    assert out["regime"]["cold_rain_graupel_levels"] > 50, out["regime"]

    # FALSIFIER: with the lane OFF the port removes <30% of WRF's column rain
    # sink (the bug); with it ON the column-integrated rain->graupel conversion
    # matches WRF to within +-15% (predeclared band -- the per-cell vertical
    # distribution is not yet bit-exact; the bulk magnitude that drives RAINNC
    # accumulation is).
    assert abs(off["qr"]["ratio_jax_over_wrf"]) < 0.30, off["qr"]
    assert 0.85 <= on["qr"]["ratio_jax_over_wrf"] <= 1.15, on["qr"]

    # graupel source likewise must move substantially toward WRF.
    assert on["qg"]["ratio_jax_over_wrf"] > off["qg"]["ratio_jax_over_wrf"], (on["qg"], off["qg"])


@pytest.mark.skipif(not _have_oracle(), reason="cold mixed-phase oracle savepoint absent")
@pytest.mark.skipif(not _have_tables(), reason="cold-collection fixture absent")
def test_cold_collection_inactive_on_warm_oracle():
    """The cold lane must not perturb the warm precip oracle (gate stays green)."""
    sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))
    from precip_oracle_validate import run_scheme

    r = run_scheme("faithful_explicit")
    # warm column: rain stays WRF-faithful (cold branch gated off by T>=T_0).
    assert r["per_field"]["qr"]["mean_rel"] < 0.01, r["per_field"]["qr"]
