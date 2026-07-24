from __future__ import annotations

import numpy as np

from scripts import v0234_1500_scientific_rca_source_oracle as oracle


def test_endpoint_pin_matches_only_at_stage_end() -> None:
    start = np.asarray([1.0, -2.0])
    tendency = np.asarray([0.5, 0.25])
    wrf = oracle.wrf_spec_walk(start, tendency, 0.2, 5)
    pinned = oracle.endpoint_pin_walk(start, tendency, 0.2, 5)
    np.testing.assert_array_equal(wrf[0], pinned[0])
    np.testing.assert_allclose(wrf[-1], pinned[-1], rtol=0.0, atol=5.0e-16)
    assert not np.array_equal(wrf[1:-1], pinned[1:-1])


def test_wrf_ph_equation_closes_coupled_conservation() -> None:
    lhs, rhs = oracle.coupled_ph_after_update(
        ph_work=np.asarray([2.0, -3.0]),
        ph_save=np.asarray([80.0, 120.0]),
        field_tend=np.asarray([5.0, -2.0]),
        mu_tend=np.asarray([0.2, -0.3]),
        muts=np.asarray([9000.0, 10000.0]),
        c1f=np.asarray([0.4, 0.8]),
        c2f=np.asarray([2.0, 3.0]),
        dts=0.6,
    )
    np.testing.assert_allclose(lhs, rhs, rtol=2.0e-16, atol=2.0e-11)


def test_zero_mass_tendency_reduces_to_additive_coupled_ph() -> None:
    result = oracle.wrf_spec_ph_update(
        ph_work=np.asarray([2.0]),
        ph_save=np.asarray([100.0]),
        field_tend=np.asarray([12.0]),
        mu_tend=np.asarray([0.0]),
        muts=np.asarray([8000.0]),
        c1f=np.asarray([0.5]),
        c2f=np.asarray([4.0]),
        dts=0.25,
    )
    np.testing.assert_allclose(result, [2.0 + 3.0 / 4004.0], rtol=0.0, atol=1.0e-15)


def test_pinned_sources_close_discriminator() -> None:
    proof = oracle.build_source_proof()
    assert proof["verdict"] == "SOURCE_DISCREPANCY_PROVEN"
    assert all(proof["source_checks"].values())
    assert proof["conservation_oracle"]["passed"] is True
