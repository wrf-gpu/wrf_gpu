"""Root lateral boundary moist/scalar floor: WRF has none (review-f2b RFB02).

Pristine WRF applies moist/scalar boundaries only as tendencies
(relax_bdy_scalar / spec_bdy_scalar -> rk_update_scalar) with no positivity
clamp.  The port's released max(., 0) acts on the FULL field; strict
(guards-disabled) runs must therefore pass ``positivity_floor=False``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries


NZ, NY, NX = 6, 20, 24


def _leaf(z_len, value, dtype=jnp.float32):
    # (time, side, bdy_width, z, side_len), current State layout.
    return jnp.full((2, 4, 5, z_len, max(NX + 1, NY + 1)), value, dtype)


def _state():
    qv = jnp.full((NZ, NY, NX), 0.001, jnp.float32).at[2, NY // 2, NX // 2].set(-1e-6)
    qc = jnp.zeros((NZ, NY, NX), jnp.float32).at[3, NY // 2 + 1, NX // 2].set(-2e-7)
    f64 = jnp.float64
    fields = dict(
        theta=jnp.zeros((NZ, NY, NX), jnp.float32), qv=qv, qc=qc,
        u=jnp.zeros((NZ, NY, NX + 1), jnp.float32), v=jnp.zeros((NZ, NY + 1, NX), jnp.float32),
        w=jnp.zeros((NZ + 1, NY, NX), f64),
        p_total=jnp.ones((NZ, NY, NX), f64) * 1000.0, p_perturbation=jnp.zeros((NZ, NY, NX), f64),
        ph_total=jnp.ones((NZ + 1, NY, NX), f64) * 100.0, ph_perturbation=jnp.zeros((NZ + 1, NY, NX), f64),
        mu_total=jnp.ones((NY, NX), f64) * 1000.0, mu_perturbation=jnp.zeros((NY, NX), f64),
        theta_bdy=_leaf(NZ, 10.0), qv_bdy=_leaf(NZ, 0.002), qc_bdy=_leaf(NZ, 0.0),
        u_bdy=_leaf(NZ, 3.0), v_bdy=_leaf(NZ, 4.0), w_bdy=_leaf(NZ + 1, 7.0, f64),
        p_bdy=_leaf(NZ, 8.0, f64), pb_bdy=_leaf(NZ, 900.0, f64), ph_bdy=_leaf(NZ + 1, 5.0, f64),
        phb_bdy=_leaf(NZ + 1, 95.0, f64), mu_bdy=_leaf(1, 6.0, f64), mub_bdy=_leaf(1, 990.0, f64),
    )
    state = State.tree_unflatten(None, [fields.get(name) for name in State.__slots__])
    return state, (2, NY // 2, NX // 2), (3, NY // 2 + 1, NX // 2)


@pytest.mark.parametrize("dry_spec_only", [False, True])
def test_unfloored_root_boundary_keeps_interior_negatives_exact(dry_spec_only):
    state, qv_cell, qc_cell = _state()
    free = apply_lateral_boundaries(state, 0.0, 60.0, dry_spec_only=dry_spec_only, positivity_floor=False)
    floored = apply_lateral_boundaries(state, 0.0, 60.0, dry_spec_only=dry_spec_only)
    # Interior cells (beyond the 5-wide band) are untouched bit-for-bit without the floor.
    inner = (slice(None), slice(5, -5), slice(5, -5))
    np.testing.assert_array_equal(np.asarray(free.qv)[inner], np.asarray(state.qv)[inner])
    np.testing.assert_array_equal(np.asarray(free.qc)[inner], np.asarray(state.qc)[inner])
    assert float(free.qv[qv_cell]) == pytest.approx(-1e-6) and float(free.qc[qc_cell]) == pytest.approx(-2e-7)
    # The released default still floors the FULL field (documents the non-WRF repair).
    assert float(floored.qv[qv_cell]) == 0.0 and float(floored.qc[qc_cell]) == 0.0
    # Boundary values (non-negative here) agree between the two paths.
    band = np.ones(state.qv.shape, bool)
    band[inner] = False
    np.testing.assert_array_equal(np.asarray(free.qv)[band], np.asarray(floored.qv)[band])


def test_root_step_passes_strict_flag_to_boundary_floor():
    """Wiring: the operational end-of-step boundary call ties the floor to guards."""
    source = Path(__file__).resolve().parents[1]/"src/gpuwrf/runtime/operational_mode.py"
    calls = [node for node in ast.walk(ast.parse(source.read_text()))
             if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "apply_lateral_boundaries"]
    keywords = [kw for call in calls for kw in call.keywords if kw.arg == "positivity_floor"]
    assert len(keywords) == 1
    assert ast.unparse(keywords[0].value) == "not bool(namelist.disable_guards)"
