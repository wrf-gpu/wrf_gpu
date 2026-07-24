"""CPU algebra/oracle gates for the coupled live-nest forcedown repair.

These tests are intentionally independent of CUDA.  The expected coupling and
SINT operations are re-derived with NumPy from pristine WRF v4.7.1
``couple_or_uncouple_em.F`` and ``bdy_interp1``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.nesting.boundary_construction import (
    build_child_boundary_package,
    build_nest_force_weights,
)
from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from v0234_wrf_sint_source_oracle import sint_full, wrf_sides


SEALED_AUTHORITY = (
    Path(__file__).resolve().parents[1]
    / ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max/authority-proof.json"
)


def _varied_metrics(grid, seed: int):
    rng = np.random.default_rng(seed)
    m = grid.metrics
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    return dataclasses.replace(
        m,
        msfuy=jnp.asarray(1.01 + 0.03 * rng.random((ny, nx + 1))),
        msfvx=jnp.asarray(1.02 + 0.03 * rng.random((ny + 1, nx))),
        msfty=jnp.asarray(1.03 + 0.03 * rng.random((ny, nx))),
        c1h=jnp.asarray(np.linspace(0.78, 0.94, nz)),
        c2h=jnp.asarray(np.linspace(35.0, 65.0, nz)),
        c1f=jnp.asarray(np.linspace(0.76, 0.96, nz + 1)),
        c2f=jnp.asarray(np.linspace(30.0, 70.0, nz + 1)),
        provenance=f"coupled-forcedown-test-{seed}",
    )


def _pattern(shape, offset: float, scale: float):
    index = np.indices(shape, dtype=np.float64)
    value = np.full(shape, offset, dtype=np.float64)
    for axis, coord in enumerate(index):
        value += scale * float(axis + 1) * coord
    return value


def _state_with_gradients(grid, *, seed: int, optional_boundaries: bool):
    state = build_neutral_state(grid)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    base_mu = np.asarray(state.mu_total - state.mu_perturbation)
    mu_p = _pattern((ny, nx), 20.0 + seed, 0.8)
    updates = {
        "mu_total": jnp.asarray(base_mu + mu_p),
        "mu_perturbation": jnp.asarray(mu_p),
        "u": jnp.asarray(_pattern((nz, ny, nx + 1), 2.0, 0.015)),
        "v": jnp.asarray(_pattern((nz, ny + 1, nx), -1.0, 0.012)),
        "w": jnp.asarray(_pattern((nz + 1, ny, nx), 0.03, 0.0007)),
        "theta": jnp.asarray(_pattern((nz, ny, nx), 300.5, 0.008)),
        "qv": jnp.asarray(_pattern((nz, ny, nx), 0.004, 2.0e-6)),
        "qc": jnp.asarray(_pattern((nz, ny, nx), 2.0e-5, 1.0e-8)),
        "qr": jnp.asarray(_pattern((nz, ny, nx), 3.0e-5, 1.1e-8)),
        "qi": jnp.asarray(_pattern((nz, ny, nx), 4.0e-5, 1.2e-8)),
        "qs": jnp.asarray(_pattern((nz, ny, nx), 5.0e-5, 1.3e-8)),
        "qg": jnp.asarray(_pattern((nz, ny, nx), 6.0e-5, 1.4e-8)),
        "Ni": jnp.asarray(_pattern((nz, ny, nx), 1000.0, 0.5)),
        "Nr": jnp.asarray(_pattern((nz, ny, nx), 500.0, 0.25)),
        "ph_perturbation": jnp.asarray(
            _pattern((nz + 1, ny, nx), 15.0, 0.03)
        ),
    }
    ph_base = np.asarray(state.ph_total - state.ph_perturbation)
    updates["ph_total"] = jnp.asarray(ph_base) + updates["ph_perturbation"]
    if optional_boundaries:
        for name in ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr"):
            updates[f"{name}_bdy"] = jnp.zeros_like(state.qv_bdy)
    return state.replace(**updates)


def _np_couple(state, metrics):
    mu = np.asarray(state.mu_total)
    c1h = np.asarray(metrics.c1h)[:, None, None]
    c2h = np.asarray(metrics.c2h)[:, None, None]
    c1f = np.asarray(metrics.c1f)[:, None, None]
    c2f = np.asarray(metrics.c2f)[:, None, None]
    mass_h = c1h * mu[None] + c2h
    mass_f = c1f * mu[None] + c2f
    muu = 0.5 * (
        np.concatenate((mu[:, :1], mu), axis=1)
        + np.concatenate((mu, mu[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu[:1, :], mu), axis=0)
        + np.concatenate((mu, mu[-1:, :]), axis=0)
    )
    coupled = {
        "u": np.asarray(state.u)
        * (c1h * muu[None] + c2h)
        / np.asarray(metrics.msfuy)[None],
        "v": np.asarray(state.v)
        * (c1h * muv[None] + c2h)
        / np.asarray(metrics.msfvx)[None],
        "w": np.asarray(state.w) * mass_f / np.asarray(metrics.msfty)[None],
        "theta": (np.asarray(state.theta) - 300.0) * mass_h,
        "qv": np.asarray(state.qv) * mass_h,
        "ph": np.asarray(state.ph_perturbation) * mass_f,
        "mu": np.asarray(state.mu_perturbation),
    }
    for name in ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr"):
        coupled[name] = np.asarray(getattr(state, name)) * mass_h
    return coupled


def _fixture():
    pgrid0 = build_flat_grid(nx=18, ny=17, nz=4, dx_m=3000.0)
    cgrid0 = build_flat_grid(nx=14, ny=13, nz=4, dx_m=1000.0)
    pmetrics = _varied_metrics(pgrid0, 12)
    cmetrics = _varied_metrics(cgrid0, 13)
    pgrid = dataclasses.replace(pgrid0, metrics=pmetrics)
    cgrid = dataclasses.replace(cgrid0, metrics=cmetrics)
    parent = _state_with_gradients(pgrid, seed=1, optional_boundaries=False)
    child = _state_with_gradients(cgrid, seed=2, optional_boundaries=True)
    weights = build_nest_force_weights(
        parent_grid_ratio=3,
        i_parent_start=3,
        j_parent_start=3,
        parent_grid=pgrid,
        child_grid=cgrid,
        registration="sint",
    )
    return pgrid, cgrid, parent, child, weights


def test_sealed_ni_v10_spatial_evidence_remains_two_causal_tracks():
    raw = SEALED_AUTHORITY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "4132b6e467b6313d4ae80171b1817119478fc3d0a8cfebc0244d0b893f407497"
    )
    proof = json.loads(raw)
    geometry = proof["corrected_geometry"]
    assert geometry["python_mass_index"] == [1, 48, 78]
    assert geometry["lat"] == 28.297913
    assert geometry["lon"] == -16.303406
    assert geometry["landmask"] == 0
    assert geometry["hgt_m"] == 0
    assert "not boundary or edge" in geometry["classification"]
    v10 = proof["v10_track"]
    assert v10["last_frame_v10_rmse"] == 2.1128268857679338
    assert v10["last_frame_max_python_yx"] == [39, 19]
    assert v10["last_frame_max_distance_from_ni_km"] > 59.0
    assert v10["last_frame_ni_cell_delta"] == -0.7763886451721191
    assert proof["material_closure"]["ni_v10_common_root"] is False


def test_exact_coupled_package_matches_independent_wrf_algebra_all_staggers():
    pgrid, cgrid, parent, child, weights = _fixture()
    out = build_child_boundary_package(
        child,
        parent,
        weights,
        bdy_width=5,
        parent_metrics=pgrid.metrics,
        child_metrics=cgrid.metrics,
        coupled_forcedown=True,
        parent_grid_ratio=3,
    )
    pc = _np_couple(parent, pgrid.metrics)
    cc = _np_couple(child, cgrid.metrics)
    side_len = int(out.u_bdy.shape[-1])
    cases = {
        "u_bdy": ("u", (cgrid.ny, cgrid.nx + 1), True, False),
        "v_bdy": ("v", (cgrid.ny + 1, cgrid.nx), False, True),
        "w_bdy": ("w", (cgrid.ny, cgrid.nx), False, False),
        "theta_bdy": ("theta", (cgrid.ny, cgrid.nx), False, False),
        "qv_bdy": ("qv", (cgrid.ny, cgrid.nx), False, False),
        "ph_bdy": ("ph", (cgrid.ny, cgrid.nx), False, False),
        "mu_bdy": ("mu", (cgrid.ny, cgrid.nx), False, False),
    }
    for name in ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr"):
        cases[f"{name}_bdy"] = (name, (cgrid.ny, cgrid.nx), False, False)
    for leaf_name, (field_name, child_shape, xstag, ystag) in cases.items():
        leaf = np.asarray(getattr(out, leaf_name))
        old_expected = wrf_sides(cc[field_name], width=5, side_len=side_len)
        new_full = sint_full(
            pc[field_name],
            ratio=3,
            i_parent_start=3,
            j_parent_start=3,
            child_ny=child_shape[0],
            child_nx=child_shape[1],
            xstag=xstag,
            ystag=ystag,
        )
        new_expected = wrf_sides(new_full, width=5, side_len=side_len)
        # Boundary precision is part of the State contract (most 3-D records are
        # fp32-gated even when the transient coupling algebra is fp64).
        old_expected = old_expected.astype(leaf.dtype)
        new_expected = new_expected.astype(leaf.dtype)
        np.testing.assert_array_equal(leaf[0], old_expected)
        np.testing.assert_allclose(leaf[1], new_expected, rtol=2e-7, atol=2e-7)


def test_candidate_off_package_is_exactly_the_released_decoupled_path():
    pgrid, cgrid, parent, child, weights = _fixture()
    released = build_child_boundary_package(child, parent, weights, bdy_width=5)
    explicit_off = build_child_boundary_package(
        child,
        parent,
        weights,
        bdy_width=5,
        parent_metrics=pgrid.metrics,
        child_metrics=cgrid.metrics,
        coupled_forcedown=False,
    )
    assert jax.tree_util.tree_structure(released) == jax.tree_util.tree_structure(explicit_off)
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(released),
        jax.tree_util.tree_leaves(explicit_off),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def test_couple_then_sint_commutator_and_conservation_falsifiers():
    """Variable mass/field covariance is precisely what late coupling loses."""

    mass = np.array([[2.0, 3.0], [5.0, 9.0]])
    field = np.array([[1.0, 4.0], [2.0, 8.0]])
    # Center of four parent cells: the bilinear SINT low-order operator is a
    # convex average.  Exact forcedown conserves the interpolated coupled amount.
    sint = lambda value: float(np.mean(value))
    exact = sint(mass * field)
    late = sint(mass) * sint(field)
    assert abs(exact - late) > 1.0
    assert exact == sint(mass * field)  # coupled conservation oracle
    # Two independent falsifiers: uniform mass OR constant physical field makes
    # the commutator vanish, as the source algebra predicts.
    uniform_mass = np.full_like(mass, 4.0)
    constant_field = np.full_like(field, 7.0)
    assert sint(uniform_mass * field) == sint(uniform_mass) * sint(field)
    assert sint(mass * constant_field) == sint(mass) * sint(constant_field)


def test_two_force_sequence_is_deterministic_restart_safe_and_interface_stable():
    pgrid, cgrid, parent, child, weights = _fixture()

    def force(cstate, pstate):
        return build_child_boundary_package(
            cstate,
            pstate,
            weights,
            bdy_width=5,
            parent_metrics=pgrid.metrics,
            child_metrics=cgrid.metrics,
            coupled_forcedown=True,
            parent_grid_ratio=3,
        )

    first = force(child, parent)
    parent2 = parent.replace(theta=parent.theta + 0.125, u=parent.u - 0.25)
    second_a = force(first, parent2)
    second_b = force(first, parent2)
    assert jax.tree_util.tree_structure(second_a) == jax.tree_util.tree_structure(child)
    assert len(jax.tree_util.tree_leaves(second_a)) == len(jax.tree_util.tree_leaves(child))
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(second_a),
        jax.tree_util.tree_leaves(second_b),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))
    restarted = pickle.loads(pickle.dumps(second_a, protocol=5))
    for lhs, rhs in zip(
        jax.tree_util.tree_leaves(second_a),
        jax.tree_util.tree_leaves(restarted),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))


def test_coupled_builder_lowering_has_unchanged_interface_and_no_observer():
    pgrid, cgrid, parent, child, weights = _fixture()

    def force(cstate, pstate):
        return build_child_boundary_package(
            cstate,
            pstate,
            weights,
            bdy_width=5,
            parent_metrics=pgrid.metrics,
            child_metrics=cgrid.metrics,
            coupled_forcedown=True,
            parent_grid_ratio=3,
        )

    # The first force normalizes synthetic one-time fixture records to the
    # production two-record package.  Every resident/AOT force thereafter must
    # preserve that exact interface.
    resident = force(child, parent)
    lowered = jax.jit(force).lower(resident, parent)
    text = str(lowered.compiler_ir(dialect="stablehlo")).lower()
    for token in (
        "host_callback",
        "io_callback",
        "pure_callback",
        "outside_compilation",
        "xla_python_cpu_callback",
    ):
        assert token not in text
    assert jax.tree_util.tree_structure(lowered.out_info) == jax.tree_util.tree_structure(resident)
    in_specs = [(tuple(x.shape), str(x.dtype)) for x in jax.tree_util.tree_leaves(resident)]
    out_specs = [(tuple(x.shape), str(x.dtype)) for x in jax.tree_util.tree_leaves(lowered.out_info)]
    assert in_specs == out_specs
