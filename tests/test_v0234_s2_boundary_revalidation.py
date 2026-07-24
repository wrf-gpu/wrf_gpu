"""Focused guards for the v0.23.4 S2 revalidation proof itself."""

from __future__ import annotations

import subprocess

import numpy as np

from scripts.v0234_gpt_s2_boundary_revalidation import (
    _distance_to_edge,
    _wrf_relax,
    source_interaction_audit,
)


def test_correct_field_target_sign_closes_literal_wrf_relax() -> None:
    rng = np.random.default_rng(23421)
    field = rng.normal(size=(2, 10, 11))
    target = rng.normal(size=field.shape)
    d_field = rng.normal(scale=0.01, size=field.shape)
    d_target = rng.normal(scale=0.01, size=field.shape)

    base = _wrf_relax(field, target, dt=6.0)
    direct_field = _wrf_relax(field + d_field, target, dt=6.0) - base
    direct_target = _wrf_relax(field, target + d_target, dt=6.0) - base
    direct_both = (
        _wrf_relax(field + d_field, target + d_target, dt=6.0) - base
    )

    # relax(dF, 0) already has the field-side sign.  Negating this expression
    # was the historical helper's bookkeeping error.
    field_formula = _wrf_relax(d_field, np.zeros_like(d_field), dt=6.0)
    target_formula = _wrf_relax(np.zeros_like(d_target), d_target, dt=6.0)
    np.testing.assert_allclose(direct_field, field_formula, rtol=0, atol=1e-15)
    np.testing.assert_allclose(direct_target, target_formula, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        direct_both, direct_field + direct_target, rtol=0, atol=1e-15
    )


def test_literal_wrf_relax_support_is_only_rows_one_to_three() -> None:
    field = np.zeros((2, 12, 13), dtype=np.float64)
    target = np.ones_like(field)
    tendency = _wrf_relax(field, target, dt=6.0)
    distance = _distance_to_edge(tendency.shape)
    assert np.count_nonzero(tendency[distance == 0]) == 0
    assert np.count_nonzero(tendency[distance == 4]) == 0
    assert np.count_nonzero(tendency[distance >= 4]) == 0
    assert np.count_nonzero(tendency[(distance >= 1) & (distance <= 3)]) > 0


def test_interaction_surface_retains_operator_and_wrapper() -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    audit = source_interaction_audit(head)
    assert audit["gate"] is True
    assert audit["boundary_operator"]["whole_file_identical"] is True
    assert all(
        row["identical"]
        for row in audit["operational_wrapper_functions"].values()
    )
    assert all(
        row["identical"]
        for row in audit["operational_fold_and_freeze_nodes"].values()
    )
    assert audit["pipeline_default_promotion"]["current_explicit_default_one"]
