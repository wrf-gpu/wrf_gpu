from __future__ import annotations

import numpy as np

from scripts import v0234_nested_boundary_spatial_causal as spatial


def _coverage(masks: dict[str, np.ndarray]) -> np.ndarray:
    return sum(mask.astype(np.int8) for mask in masks.values())


def test_distance_topology_and_footprint_masks_are_exact_partitions() -> None:
    distances = spatial.horizontal_distances((93, 111))
    assert np.all(_coverage(spatial.distance_bin_masks(distances["nearest"])) == 1)
    topology = spatial.topology_masks(distances)
    assert np.all(_coverage(topology) == 1)
    assert int(topology["corner"].sum()) == 4 * 4 * 4
    footprint = spatial.forcing_footprint_masks(distances["nearest"])
    assert np.all(_coverage(footprint) == 1)
    assert not np.any(
        footprint["direct_spec_relax_rings_0_3"]
        & footprint["reserved_package_buffer_ring_4"]
    )


def test_staggered_land_sea_has_explicit_coast_faces() -> None:
    mass = np.zeros((3, 4), dtype=np.float64)
    mass[:, 2:] = 1.0
    mass_masks = spatial.surface_type_masks(mass, (3, 4))
    u_masks = spatial.surface_type_masks(mass, (3, 5))
    v_masks = spatial.surface_type_masks(mass, (4, 4))
    assert np.all(_coverage(mass_masks) == 1)
    assert np.all(_coverage(u_masks) == 1)
    assert np.all(_coverage(v_masks) == 1)
    assert int(u_masks["mixed_coast"].sum()) == 3
    assert int(v_masks["mixed_coast"].sum()) == 0


def test_metrics_and_regression_classifier_distinguish_propagated_interior() -> None:
    retry = np.zeros((2, 20, 20), dtype=np.float64)
    candidate = np.zeros_like(retry)
    distance = spatial.horizontal_distances((20, 20))["nearest"]
    candidate[:, distance > spatial.DIRECT_FORCING_MAX_DISTANCE] = 2.0
    result = spatial.classify_regression(candidate, retry)
    assert result["verdict"] == "propagated_interior"
    assert result["direct_spec_relax_excess_sum_sq"] == 0.0
    assert result["outside_direct_fraction_of_positive_excess"] == 1.0
    summary = spatial.metrics(candidate)
    assert summary["n"] == candidate.size
    assert summary["sum_sq"] > 0.0


def test_real_authenticated_spatial_proof_matches_frozen_rmse() -> None:
    proof = spatial.build_proof()
    assert proof["causal_verdict"]["spatial_class"] == "propagated_interior_after_boundary_operator_change"
    assert proof["causal_verdict"]["boundary_local_rejected"] is True
    for field in spatial.REGRESSED_FIELDS:
        actual = proof["fields"][field]["candidate_minus_cpu"]["overall"]["rmse"]
        assert np.isclose(
            actual,
            spatial.EXPECTED_RMSE["candidate_minus_cpu"][field],
            rtol=0.0,
            atol=5.0e-15,
        )
