from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gpuwrf.runtime.operational_mode import _positive_definite_theta_increment_limiter


def test_theta_positive_definite_limiter_counts_first_cell_and_conserves_mass() -> None:
    origin = jnp.full((1, 2, 3), 300.0, dtype=jnp.float64)
    candidate = origin.at[0, 0, 1].set(-10.0).at[0, 1, 2].set(610.0)
    mass = jnp.ones_like(origin)

    limited, diagnostics = _positive_definite_theta_increment_limiter(candidate, origin, mass)

    limited_np = np.asarray(limited)
    assert float(limited_np.min()) >= 0.0
    assert float(limited_np.max()) <= 500.0
    assert int(np.asarray(diagnostics["theta_limited_cell_count"])) == 2
    np.testing.assert_array_equal(np.asarray(diagnostics["theta_first_limited_cell_xyz"]), np.array([1, 0, 0]))
    assert abs(float(np.asarray(diagnostics["theta_mass_residual"]))) < 1.0e-9
    assert abs(float(np.asarray(diagnostics["theta_mass_before"])) - float(np.asarray(diagnostics["theta_mass_after"]))) < 1.0e-9


def test_theta_positive_definite_limiter_records_noop_diagnostics() -> None:
    origin = jnp.full((1, 2, 3), 300.0, dtype=jnp.float64)
    candidate = origin + 2.0
    mass = jnp.ones_like(origin)

    limited, diagnostics = _positive_definite_theta_increment_limiter(candidate, origin, mass)

    np.testing.assert_allclose(np.asarray(limited), np.asarray(candidate))
    assert int(np.asarray(diagnostics["theta_limited_cell_count"])) == 0
    np.testing.assert_array_equal(np.asarray(diagnostics["theta_first_limited_cell_xyz"]), np.array([-1, -1, -1]))
    assert abs(float(np.asarray(diagnostics["theta_mass_residual"]))) < 1.0e-9


def test_theta_positive_definite_limiter_honors_monotonic_bounds() -> None:
    origin = jnp.asarray([[[290.0, 300.0, 310.0]]], dtype=jnp.float64)
    candidate = origin.at[0, 0, 1].set(100.0)
    mass = jnp.ones_like(origin)
    lower = jnp.asarray([[[290.0, 290.0, 300.0]]], dtype=jnp.float64)
    upper = jnp.asarray([[[300.0, 310.0, 310.0]]], dtype=jnp.float64)

    limited, diagnostics = _positive_definite_theta_increment_limiter(
        candidate,
        origin,
        mass,
        lower_bound=lower,
        upper_bound=upper,
    )

    assert float(np.asarray(limited)[0, 0, 1]) >= 290.0
    assert int(np.asarray(diagnostics["theta_limited_cell_count"])) == 1
    np.testing.assert_array_equal(np.asarray(diagnostics["theta_first_limited_cell_xyz"]), np.array([1, 0, 0]))
