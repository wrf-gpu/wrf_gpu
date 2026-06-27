from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import FP32_GATED, FP64, INT32, PRECISION_MATRIX
from gpuwrf.contracts.state import State


def test_precision_matrix_covers_every_state_leaf_dtype():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)

    for field in state.active_field_names():
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
        # qke moved gated->FP64-locked (qke-fp64-fix sprint 2026-06-04): the
        # MYNN TKE budget overflows fp32 at 1km (d03 NONFINITE after hour 1).
        # qke is OUTSIDE the conserved mass/pressure path, so it joins the
        # FP64-locked class. See contracts/precision.py for the rationale.
        "xland",
        "lakemask",
        "mavail",
        "u_bdy",
        "v_bdy",
        "theta_bdy",
        "qv_bdy",
        "qc_bdy",
        "qr_bdy",
        "qi_bdy",
        "qs_bdy",
        "qg_bdy",
        "Ni_bdy",
        "Nr_bdy",
        # v0.6.0 S0 additive WDM6 number concentrations (FP32_GATED like the
        # other hydrometeor number species). Added here to keep this contract
        # test in sync with the matrix (they were appended to PRECISION_MATRIX
        # but never to this set).
        "Nc",
        "Nn",
        # v0.15 MYNN SGS-cloud diagnostics (FP32_GATED like qc); the prognostic
        # total-water variance qsq stays FP64-locked with the qke TKE family.
        "qc_bl",
        "qi_bl",
        "cldfra_bl",
        # v0.17 ADR-032 graupel/hail substrate (FP32_GATED, same class as the
        # qg/Ng hydrometeor + number species they extend).
        "qh",
        "Nh",
        "qvolg",
        "qvolh",
        # v0.16 additive aerosol-aware Thompson (mp=28) aerosol numbers
        # (FP32_GATED, same class as Nc/Nn).
        "nwfa",
        "nifa",
    }
    integer_static = {"lu_index"}
    locked = set(State.__slots__) - gated - integer_static

    for field in gated:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == FP32_GATED
        assert gate_required is True

    for field in locked:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == FP64
        assert dtype == jnp.float64
        assert gate_required is False

    for field in integer_static:
        dtype, gate_required = PRECISION_MATRIX[field]
        assert dtype == INT32
        assert dtype == jnp.int32
        assert gate_required is False
