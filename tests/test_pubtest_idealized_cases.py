from __future__ import annotations

import numpy as np

from gpuwrf.fixtures.idealized_cases import build_density_current, build_schaer_mountain_wave, build_warmbubble


def test_warmbubble_builder_is_finite_symmetric_and_reference_scaled() -> None:
    case = build_warmbubble(nx=101, nz=41, dx_m=100.0, dz_m=100.0)
    theta_prime = case.arrays["theta_perturbation_k"]

    assert case.case_id == "idealized-warmbubble-bryan-fritsch-2002"
    assert theta_prime.shape == (41, 101)
    assert np.all(np.isfinite(theta_prime))
    assert np.isclose(float(theta_prime.max()), 2.0)
    assert np.allclose(theta_prime, theta_prime[:, ::-1])
    assert np.all(case.arrays["density_kg_m3"] > 0.0)


def test_density_current_builder_matches_straka_cold_block_extrema() -> None:
    case = build_density_current(nx=65, nz=65, dx_m=100.0, dz_m=100.0)
    theta_prime = case.arrays["theta_perturbation_k"]

    assert case.case_id == "idealized-density-current-straka-1993"
    assert theta_prime.shape == (65, 65)
    assert np.all(np.isfinite(theta_prime))
    assert np.isclose(float(theta_prime.min()), -15.0)
    assert float(theta_prime.max()) == 0.0
    assert case.reference["published_targets"]["integration_s"] == 900


def test_schaer_builder_has_sinusoidal_terrain_and_linear_surface_w() -> None:
    case = build_schaer_mountain_wave(nx=101, nz=21, dx_m=250.0, dz_m=250.0, domain_half_width_m=12500.0)
    terrain = case.arrays["terrain_m"]
    w_surface = case.arrays["w_surface_linear_m_s"]

    assert case.case_id == "idealized-mountain-wave-schaer-2002"
    assert terrain.shape == (101,)
    assert np.all(np.isfinite(terrain))
    assert np.all(terrain >= 0.0)
    assert np.isclose(float(terrain.max()), 250.0)
    assert np.all(np.isfinite(w_surface))
    assert np.isclose(float(w_surface[terrain.argmax()]), 0.0, atol=1.0e-12)
