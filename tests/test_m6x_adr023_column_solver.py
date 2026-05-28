from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import _calc_coef_w, vertical_acoustic_update
from gpuwrf.dynamics.metrics import flat_metrics_for_grid
from gpuwrf.dynamics.vertical_implicit_solver import solve_tridiagonal


def _grid(nz: int = 6) -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, 500.0, 500.0, 1, 1)
    terrain = TerrainProvenance(
        source_path="analytic://adr023-column-solver",
        sha256="analytic-adr023",
        shape=(1, 1),
        units="m",
        projection_transform="flat-column",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", ("w", "ph", "theta"), 0, "linear", True)
    return GridSpec(projection, terrain, vertical, bc, eta, jnp.zeros((1, 1), dtype=jnp.float64))


def _state_and_base(grid: GridSpec) -> tuple[State, BaseState]:
    arrays = {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}
    z = jnp.linspace(0.0, 6000.0, grid.nz + 1, dtype=jnp.float64)
    phb = 9.80665 * z[:, None, None]
    theta_base = jnp.ones((grid.nz, 1, 1), dtype=jnp.float64) * 300.0
    pb = jnp.ones((grid.nz, 1, 1), dtype=jnp.float64) * 90_000.0
    mub = jnp.ones((1, 1), dtype=jnp.float64) * 85_000.0
    arrays["theta"] = theta_base
    arrays["p"] = pb
    arrays["p_total"] = pb
    arrays["p_perturbation"] = jnp.zeros_like(pb)
    arrays["ph"] = phb
    arrays["ph_total"] = phb
    arrays["ph_perturbation"] = jnp.zeros_like(phb)
    arrays["mu"] = mub
    arrays["mu_total"] = mub
    arrays["mu_perturbation"] = jnp.zeros_like(mub)
    return State(**arrays), BaseState(pb=pb, phb=phb, mub=mub, t0=theta_base, theta_base=theta_base)


def test_thomas_solver_matches_dense_spd_reference():
    a = np.array([0.0, -0.2, -0.15, -0.1, -0.05], dtype=np.float64)[:, None]
    b = np.array([1.4, 1.6, 1.5, 1.45, 1.3], dtype=np.float64)[:, None]
    c = np.array([-0.1, -0.05, -0.2, -0.1, 0.0], dtype=np.float64)[:, None]
    rhs = np.array([1.0, 0.5, -0.25, 2.0, 1.5], dtype=np.float64)[:, None]
    matrix = np.diag(b[:, 0]) + np.diag(c[:-1, 0], k=1) + np.diag(a[1:, 0], k=-1)
    expected = np.linalg.solve(matrix, rhs[:, 0])

    actual = solve_tridiagonal(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(rhs))

    assert np.allclose(np.asarray(actual)[:, 0], expected, rtol=1.0e-12, atol=1.0e-12)


def test_top_lid_boundary_rows_are_honored():
    grid = _grid()
    state, base = _state_and_base(grid)
    metrics = flat_metrics_for_grid(grid)

    a_lid, b_lid, c_lid = _calc_coef_w(state, base, metrics, dt=0.5, top_lid=True)
    a_open, b_open, c_open = _calc_coef_w(state, base, metrics, dt=0.5, top_lid=False)

    assert float(a_lid[-1, 0, 0]) == 0.0
    assert float(b_lid[-1, 0, 0]) == 1.0
    assert float(c_lid[-1, 0, 0]) == 0.0
    assert float(a_open[-1, 0, 0]) == -1.0
    assert float(b_open[-1, 0, 0]) == 1.0
    assert float(c_open[-1, 0, 0]) == 0.0


def test_top_lid_update_enforces_top_w_only_when_requested():
    grid = _grid()
    state, base = _state_and_base(grid)
    metrics = flat_metrics_for_grid(grid)
    w = jnp.zeros_like(state.w).at[-2, 0, 0].set(2.0)
    state = state.replace(w=w)

    closed = vertical_acoustic_update(state, base, metrics, dt=0.1, top_lid=True)
    open_top = vertical_acoustic_update(state, base, metrics, dt=0.1, top_lid=False)

    assert float(closed.w[-1, 0, 0]) == 0.0
    assert abs(float(open_top.w[-1, 0, 0] - open_top.w[-2, 0, 0])) < 1.0e-10


def test_zero_rhs_invariance():
    a = jnp.asarray([[0.0], [-0.1], [-0.1], [-0.1]], dtype=jnp.float64)
    b = jnp.asarray([[1.0], [1.2], [1.2], [1.0]], dtype=jnp.float64)
    c = jnp.asarray([[-0.1], [-0.1], [-0.1], [0.0]], dtype=jnp.float64)
    rhs = jnp.zeros((4, 1), dtype=jnp.float64)

    actual = solve_tridiagonal(a, b, c, rhs)

    assert np.allclose(np.asarray(actual), 0.0, rtol=0.0, atol=0.0)
