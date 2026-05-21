from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import FP32_GATED, FP64, PRECISION_MATRIX
from gpuwrf.contracts.state import State


def test_precision_matrix_covers_every_state_leaf_dtype():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)

    for field in State.__slots__:
        assert field in PRECISION_MATRIX
        expected_dtype, _gate_required = PRECISION_MATRIX[field]
        assert getattr(state, field).dtype == expected_dtype


def test_precision_matrix_gate_flags_match_adr007_boundary_classes():
    gated = {
        "u",
        "v",
        "theta",
        "qv",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "Ni",
        "Nr",
        "Ns",
        "Ng",
        "qke",
        "xland",
        "lakemask",
        "mavail",
        "u_bdy",
        "v_bdy",
        "theta_bdy",
        "qv_bdy",
    }
    locked = set(State.__slots__) - gated

    for field in gated:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == FP32_GATED
        assert gate_required is True

    for field in locked:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == FP64
        assert dtype == jnp.float64
        assert gate_required is False
