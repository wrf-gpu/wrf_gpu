from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.metrics import load_wrfinput_metrics
from gpuwrf.dynamics.vertical_implicit_solver import build_epssm_column_coefficients
from gpuwrf.validation.savepoint_io import read_savepoint


ROOT = Path(__file__).resolve().parents[1]
SAVEPOINT_ROOTS = (
    ROOT / "<development-history-not-included-in-public-repo>" / "2026-05-24-m6b0r-real-fortran-emission" / "savepoints",
)


def _savepoint_path(tier: str) -> Path:
    for root in SAVEPOINT_ROOTS:
        path = root / tier / "calc_coef_w_post_step001.h5"
        if path.exists():
            return path
    pytest.skip("M6B0-R HDF5 savepoints are not available in this workspace")


@pytest.mark.parametrize("tier", ("column", "patch16", "golden"))
def test_m6b0r_wrf_calc_coef_w_matches_savepoint_tiers(tier: str):
    savepoint = read_savepoint(_savepoint_path(tier))
    metrics = load_wrfinput_metrics(savepoint.metadata.source_path)

    # WRF source: module_small_step_em.F:624-649. The source run has TOP_LID=F,
    # so line 620 leaves lid_flag=1 for the stored top coefficient row.
    actual = calc_coef_w_wrf_coefficients(
        jnp.asarray(savepoint.arrays["mut"]),
        metrics,
        dt=savepoint.metadata.dt_seconds,
        epssm=0.1,
        top_lid=False,
    )
    for name, got in zip(("a", "alpha", "gamma"), actual, strict=True):
        expected = np.asarray(savepoint.arrays[name])
        max_abs = float(np.nanmax(np.abs(np.asarray(got) - expected)))
        assert max_abs <= 1.0e-6, f"{tier} {name} max_abs_delta={max_abs}"


def test_m6b0r_fix_improves_over_legacy_dz_theta_builder():
    savepoint = read_savepoint(_savepoint_path("golden"))
    legacy_coeffs = build_epssm_column_coefficients(
        jnp.asarray(savepoint.arrays["theta"]),
        jnp.asarray(savepoint.arrays["dz_m"]),
        dt=savepoint.metadata.dt_seconds,
        epssm=0.1,
    )
    _cofrz, _cofwr, _cofwz, _coftz, _cofwt, _rdzw, tri_a, tri_b, tri_c = legacy_coeffs
    legacy = {"a": np.asarray(tri_a), "alpha": 1.0 / np.asarray(tri_b), "gamma": np.asarray(tri_c) / np.asarray(tri_b)}

    deltas = {
        name: float(np.nanmax(np.abs(np.asarray(legacy[name]) - np.asarray(savepoint.arrays[name]))))
        for name in ("a", "alpha", "gamma")
    }
    assert deltas["a"] > 1.0
    assert deltas["alpha"] > 1.0e-2
    assert deltas["gamma"] > 1.0e-2
