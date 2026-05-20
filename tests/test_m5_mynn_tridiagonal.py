from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.tridiagonal_solver import solve_tridiagonal, solve_tridiagonal_thomas_reference


def test_tridiagonal_solver_matches_dense_reference():
    a = np.array([[0.0, -0.2, -0.1, -0.3]], dtype=np.float64)
    b = np.array([[1.2, 1.5, 1.4, 1.3]], dtype=np.float64)
    c = np.array([[-0.1, -0.2, -0.2, 0.0]], dtype=np.float64)
    d = np.array([[1.0, 2.0, 1.5, 0.5]], dtype=np.float64)
    matrix = np.array(
        [
            [b[0, 0], c[0, 0], 0.0, 0.0],
            [a[0, 1], b[0, 1], c[0, 1], 0.0],
            [0.0, a[0, 2], b[0, 2], c[0, 2]],
            [0.0, 0.0, a[0, 3], b[0, 3]],
        ],
        dtype=np.float64,
    )
    expected = np.linalg.solve(matrix, d[0])
    out = solve_tridiagonal(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d))
    assert np.allclose(np.asarray(out)[0], expected, rtol=1e-12, atol=1e-12)


def test_thomas_reference_matches_xla_primitive():
    a = jnp.asarray([[0.0, -0.1, -0.1, -0.1]], dtype=jnp.float64)
    b = jnp.asarray([[1.1, 1.2, 1.2, 1.1]], dtype=jnp.float64)
    c = jnp.asarray([[-0.1, -0.1, -0.1, 0.0]], dtype=jnp.float64)
    d = jnp.asarray([[0.5, 0.7, 0.9, 1.1]], dtype=jnp.float64)
    fast = solve_tridiagonal(a, b, c, d)
    thomas = solve_tridiagonal_thomas_reference(a, b, c, d)
    assert np.allclose(np.asarray(fast), np.asarray(thomas), rtol=1e-12, atol=1e-12)
