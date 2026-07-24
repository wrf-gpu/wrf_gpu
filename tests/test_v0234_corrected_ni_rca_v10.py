from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.v0234_corrected_ni_rca_v10 import (
    destagger_lowest,
    earth_relative,
    haversine_km,
    vector_scale_decomposition,
)


def test_destagger_lowest_uses_wrf_mass_points() -> None:
    u = np.arange(2 * 3 * 5, dtype=np.float64).reshape(2, 3, 5)
    v = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    u0, v0 = destagger_lowest(u, v)
    np.testing.assert_array_equal(u0, 0.5 * (u[0, :, :-1] + u[0, :, 1:]))
    np.testing.assert_array_equal(v0, 0.5 * (v[0, :-1, :] + v[0, 1:, :]))
    assert u0.shape == v0.shape == (3, 4)


def test_vector_scale_decomposition_closes_scalar_surface_case() -> None:
    cpu_u0 = np.array([[3.0, 4.0]])
    cpu_v0 = np.array([[4.0, 3.0]])
    gpu_u0 = cpu_u0 + np.array([[0.2, -0.1]])
    gpu_v0 = cpu_v0 + np.array([[-0.3, 0.4]])
    cpu_ratio = 0.8
    gpu_ratio = 0.9
    result = vector_scale_decomposition(
        cpu_ratio * cpu_u0,
        cpu_ratio * cpu_v0,
        gpu_ratio * gpu_u0,
        gpu_ratio * gpu_v0,
        cpu_u0,
        cpu_v0,
        gpu_u0,
        gpu_v0,
    )
    np.testing.assert_allclose(
        result["total"],
        result["dynamics"] + result["surface_ratio"] + result["turning_and_cross_residual"],
        rtol=0.0,
        atol=2.0e-16,
    )


def test_rotation_and_distance_oracles() -> None:
    u = np.array([[1.0]])
    v = np.array([[0.0]])
    east, north = earth_relative(u, v, np.array([[1.0]]), np.array([[0.0]]))
    np.testing.assert_array_equal(east, np.array([[0.0]]))
    np.testing.assert_array_equal(north, np.array([[1.0]]))
    assert haversine_km(28.0, -16.0, 28.0, -16.0) == 0.0
    assert 95.0 < haversine_km(28.0, -16.0, 29.0, -16.0) < 120.0


def test_bound_spatial_evidence_requires_separate_ni_and_v10_gates() -> None:
    proof_path = Path(
        ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max/"
        "v10-spatial-causal-proof.json"
    )
    proof = json.loads(proof_path.read_text())
    ni = proof["ni_spatial_binding"]
    last = proof["last_frame"]["v10"]

    assert ni["jax_mass_index"] == {"y": 48, "x": 78}
    assert ni["lat"] == 28.297913
    assert ni["lon"] == -16.303406
    assert ni["landmask"] == 0 and ni["hgt_m"] == 0.0
    assert "not boundary or edge" in ni["classification"]
    assert last["rmse"] == 2.1128268857679338
    assert last["land_rmse"] == 2.2264724301076377
    assert last["sea_rmse"] == 2.0839931788025168
    assert last["max_abs"] == 11.358115434646606
    assert last["max_location"] == {
        "y": 39,
        "x": 19,
        "lat": 28.216018676757812,
        "lon": -16.90625,
        "landmask": 0,
    }
    assert last["ni_cell_delta"] == -0.7763886451721191
    assert proof["cross_frame"]["last_v10_max_distance_from_ni_km"] == 59.7430710652915
    assert proof["bounded_gates"]["common_ni_v10_root_proved"] is False
    assert proof["verdict"]["fix_authority"] == "NONE_FROM_RETAINED_OUTPUTS"
