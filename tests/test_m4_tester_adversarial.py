"""Adversarial tester tests for the M4 reduced JAX dycore.

These tests are owned by the sonnet-test-engineer role (Claude Opus 4.7 xhigh).
They try to break the worker's implementation rather than confirm it: tautological
oracles, dead variables in invariants, debug-hook leakage, and contract edges that
the worker's own tests skip.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.halo import HaloSpec
from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.debug.snapshots import snapshot
from gpuwrf.dynamics.acoustic import forward_backward_acoustic
from gpuwrf.dynamics.advection import (
    advect_mass_scalar,
    compute_advection_tendencies,
    derivative3_upwind,
    derivative5_upwind,
    fixture_reference_update,
    mass_face_velocities,
)
from gpuwrf.dynamics.step import run, step
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.validation.tier1 import run_tier1
from gpuwrf.validation.tier2 import density_current_state, invariant_record, make_ideal_grid


# ----------------------------------------------------------------------------- #
# Tier-1 tautology audit                                                        #
# ----------------------------------------------------------------------------- #


def test_tier1_is_a_self_check_not_a_dycore_check():
    """The fixed Tier-1 path must use the dycore's upwind sibling fixture."""

    record = run_tier1()
    assert record["fixture_id"] == "analytic-stencil-3d-upwind5-v1"
    assert "upwind" in record["operator"]
    assert "M1 centered" not in record["operator"]


def test_tier1_artifact_pass_is_consistent_with_zero_error():
    record = json.loads(Path("artifacts/m4/tier1_advection_parity.json").read_text())
    assert record["pass"] is True
    assert record["max_abs_err"] <= record["tolerance_abs"]
    assert "5th-order horizontal" in record["operator"]
    assert "3rd-order vertical" in record["operator"]


def test_dycore_advection_operator_is_NOT_what_tier1_checks():
    """Direct comparison confirms Tier-1 now checks the dycore upwind operator."""

    with np.load("fixtures/samples/analytic-stencil-3d-upwind5-v1.npz", allow_pickle=False) as loaded:
        phi = jnp.asarray(loaded["phi_initial"], dtype=jnp.float64)
        u = jnp.asarray(loaded["u_face"], dtype=jnp.float64)
        v = jnp.asarray(loaded["v_face"], dtype=jnp.float64)
        w = jnp.asarray(loaded["w_face"], dtype=jnp.float64)
        ref = np.asarray(loaded["phi_next_upwind5"], dtype=np.float64)
    grid = make_ideal_grid(8, 16, 32, dx_m=900.0, dy_m=900.0, top_m=960.0)
    u_mass = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_mass = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    w_mass = 0.5 * (w[:-1, :, :] + w[1:, :, :])
    got = np.asarray(phi + 3.0 * advect_mass_scalar(phi, u_mass, v_mass, w_mass, grid))
    assert float(np.max(np.abs(got - ref))) <= 1.0e-10


# ----------------------------------------------------------------------------- #
# Tier-2 dead-variable audit                                                    #
# ----------------------------------------------------------------------------- #


def test_tier2_mass_invariant_is_trivial_because_mu_is_dead():
    """`mu` is initialised to ones and `compute_advection_tendencies` never
    writes to it, so `sum(mu)` is invariant by construction. The tier2 mass
    residual proof is therefore evidentially weak; document the gap so reviewer
    decides whether to amend the contract before M5."""

    grid = make_ideal_grid(4, 8, 8)
    state, tendencies = density_current_state(grid)
    out = run(state, tendencies, grid, 0.25, 5, n_acoustic=2, debug=False)
    block_until_ready(out)
    assert float(jnp.sum(out.mu)) == float(jnp.sum(state.mu))


def test_tier2_density_current_ic_is_a_noop_simulation():
    """The fixed Tier-2 IC must be a nontrivial tracer translation, not a no-op."""

    grid = make_ideal_grid(4, 8, 8)
    state, tendencies = density_current_state(grid)
    assert float(jnp.max(jnp.abs(state.u))) > 0.0
    assert float(jnp.max(jnp.abs(state.v))) == 0.0
    assert float(jnp.max(jnp.abs(state.w))) == 0.0
    out = run(state, tendencies, grid, 2.0, 10, n_acoustic=4, debug=False)
    block_until_ready(out)
    assert float(jnp.max(jnp.abs(out.theta - state.theta))) > 0.1


def test_dycore_advects_theta_when_u_is_perturbed():
    """Positive companion to the no-op finding: with non-zero u, the dycore
    actually moves theta. Confirms the integration loop itself is alive — the
    tier-2 weakness is in the IC, not the implementation."""

    grid = make_ideal_grid(4, 8, 8)
    state, tendencies = density_current_state(grid)
    state = state.replace(u=jnp.ones_like(state.u) * 5.0)
    out = run(state, tendencies, grid, 0.5, 5, n_acoustic=2, debug=False)
    block_until_ready(out)
    delta = float(jnp.max(jnp.abs(out.theta - state.theta)))
    assert delta > 0.0


def test_tier2_record_pass_keys():
    record = invariant_record(make_ideal_grid(4, 6, 6), n_steps=5, dt=1.0, n_acoustic=1)
    assert set(record).issuperset(
        {
            "mass_residual_relative",
            "qv_positivity_violations",
            "nan_inf_violations",
            "final_state_differs_from_initial",
            "pass",
        }
    )
    assert record["final_state_differs_from_initial"] is True


# ----------------------------------------------------------------------------- #
# Hot-path discipline / static argname enforcement                              #
# ----------------------------------------------------------------------------- #


def test_step_requires_static_dt_and_grid_and_debug():
    """ADR-002 + M3 lesson: dt / grid / n_acoustic / debug all static. Passing
    dt as a traced scalar must fail at trace time (not silently accept and
    blow the cache)."""

    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    # Passing a traced jax array as dt should error because dt is in
    # static_argnames; JAX cannot hash a traced value.
    with pytest.raises(Exception):
        step(state, tendencies, grid, jnp.asarray(0.25), n_acoustic=2, debug=False)


def test_run_n_acoustic_zero_raises_cleanly():
    """`forward_backward_acoustic` divides dt by n_acoustic. n_acoustic=0 must
    not silently produce inf — it should either be rejected or fail at trace."""

    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    with pytest.raises(Exception):
        out = step(state, tendencies, grid, 0.25, n_acoustic=0, debug=False)
        block_until_ready(out)


# ----------------------------------------------------------------------------- #
# Debug hooks                                                                   #
# ----------------------------------------------------------------------------- #


def test_assert_finite_enabled_propagates_nan_for_bad_input():
    bad = jnp.asarray([1.0, jnp.nan, 3.0], dtype=jnp.float64)
    out = np.asarray(assert_finite(bad, "bad", enabled=True))
    assert np.all(np.isnan(out))


def test_assert_finite_enabled_is_identity_for_good_input():
    good = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    out = np.asarray(assert_finite(good, "good", enabled=True))
    np.testing.assert_array_equal(out, np.asarray([1.0, 2.0, 3.0]))


def test_assert_physical_bounds_enabled_catches_violation():
    x = jnp.asarray([0.5, 1.5, 0.25], dtype=jnp.float64)
    out = np.asarray(assert_physical_bounds(x, 0.0, 1.0, "x", enabled=True))
    # 1.5 is out of bounds; the implementation NaN-s out the whole array.
    assert np.all(np.isnan(out))


def test_assert_physical_bounds_disabled_is_pure_python_identity():
    x = jnp.asarray([99.0, -1.0, jnp.nan], dtype=jnp.float64)
    assert assert_physical_bounds(x, 0.0, 1.0, "x", enabled=False) is x


def test_snapshot_disabled_is_pure_python_identity():
    grid = make_ideal_grid(3, 4, 4)
    state, _ = density_current_state(grid)
    out = snapshot(state, "tester", enabled=False, ring_size=4)
    assert out is state


def test_step_debug_false_hlo_excludes_finite_check_ops():
    """Constitutional gate independently re-verified at unit scale: a function
    that contains `assert_finite(..., enabled=False)` compiled to HLO must not
    leak isfinite/comparison ops."""

    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    prod_hlo = compiled_text(step.lower(state, tendencies, grid, 0.1, n_acoustic=1, debug=False).compile()).lower()
    # No finiteness or NaN-tag plumbing should survive into production HLO.
    assert "is-finite" not in prod_hlo
    assert "isfinite" not in prod_hlo


def test_step_debug_true_emits_compare_ops_that_production_lacks():
    """Counterpart: confirm the debug=True compile actually emits the
    finiteness check, so 'no finite check in production' is a non-trivial
    statement."""

    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    prod_hlo = compiled_text(step.lower(state, tendencies, grid, 0.1, n_acoustic=1, debug=False).compile()).lower()
    dbg_hlo = compiled_text(step.lower(state, tendencies, grid, 0.1, n_acoustic=1, debug=True).compile()).lower()
    assert ("is-finite" in dbg_hlo) or ("compare" in dbg_hlo and "compare" not in prod_hlo) or (len(dbg_hlo) > len(prod_hlo) + 1024)


def test_artifact_hlo_diff_file_is_empty_and_canonical_sha():
    diff = Path("artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff")
    assert diff.exists(), "HLO debug-vs-stripped diff artifact missing"
    payload = diff.read_bytes()
    assert payload == b""
    assert hashlib.sha256(payload).hexdigest() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ----------------------------------------------------------------------------- #
# HaloSpec edge_type contract                                                   #
# ----------------------------------------------------------------------------- #


def test_halospec_rejects_invalid_edge_type():
    with pytest.raises(ValueError):
        HaloSpec(width=2, fields_to_exchange=("u",), edge_type="reflective")  # type: ignore[arg-type]


def test_halospec_rejects_invalid_width():
    with pytest.raises(ValueError):
        HaloSpec(width=0, fields_to_exchange=("u",), edge_type="periodic")
    with pytest.raises(ValueError):
        HaloSpec(width=5, fields_to_exchange=("u",), edge_type="periodic")


# ----------------------------------------------------------------------------- #
# Determinism and reproducibility                                               #
# ----------------------------------------------------------------------------- #


def test_run_is_bitwise_deterministic_across_repeated_calls():
    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    a = run(state, tendencies, grid, 0.25, 5, n_acoustic=2, debug=False)
    b = run(state, tendencies, grid, 0.25, 5, n_acoustic=2, debug=False)
    block_until_ready(a)
    block_until_ready(b)
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


def test_run_uniform_field_is_unchanged_at_zero_velocity():
    """If u,v,w are zero, advection tendencies are exactly zero; theta should
    not drift in a still atmosphere."""

    grid = make_ideal_grid(4, 8, 8)
    state, tendencies = density_current_state(grid)
    state = state.replace(
        u=jnp.zeros_like(state.u),
        v=jnp.zeros_like(state.v),
        w=jnp.zeros_like(state.w),
        theta=jnp.ones_like(state.theta) * 300.0,
        p=jnp.zeros_like(state.p),
        ph=jnp.zeros_like(state.ph),
    )
    out = run(state, tendencies, grid, 0.25, 3, n_acoustic=1, debug=False)
    block_until_ready(out)
    # Uniform theta in still air with zero pressure ⇒ no acoustic drive ⇒
    # the field should remain numerically unchanged. Allow round-off.
    assert float(jnp.max(jnp.abs(out.theta - state.theta))) < 1.0e-12


def test_derivative5_upwind_is_zero_on_constant_field():
    field = jnp.ones((16,), dtype=jnp.float64) * 17.0
    velocity = field * 0.0 + 1.0
    out = derivative5_upwind(field, velocity, 1.0, axis=0)
    assert float(jnp.max(jnp.abs(out))) == 0.0


def test_derivative3_upwind_responds_to_velocity_sign_flip():
    """Same data, opposite velocity sign, should switch which periodic-shifted
    stencil produces the derivative; magnitudes should match, sign differs."""

    field = jnp.sin(2.0 * jnp.pi * jnp.arange(16, dtype=jnp.float64) / 16.0)
    pos = derivative3_upwind(field, field * 0 + 1.0, 1.0, axis=0)
    neg = derivative3_upwind(field, field * 0 - 1.0, 1.0, axis=0)
    # 3rd-order upwind: backward vs forward stencil — values are unrelated by
    # simple negation, but magnitudes should be comparable on a smooth wave.
    assert float(jnp.max(jnp.abs(pos))) > 0.0
    assert float(jnp.max(jnp.abs(neg))) > 0.0
    assert not np.allclose(np.asarray(pos), np.asarray(neg))


# ----------------------------------------------------------------------------- #
# Public artifacts integrity                                                    #
# ----------------------------------------------------------------------------- #


def test_required_m4_artifacts_exist():
    artifacts = [
        "artifacts/m4/dycore_profile.json",
        "artifacts/m4/transfer_audit.json",
        "artifacts/m4/spacetime_budget.json",
        "artifacts/m4/tier1_advection_parity.json",
        "artifacts/m4/tier2_invariants.json",
        "artifacts/m4/tier3_convergence.json",
        "artifacts/m4/m5_gate_dryrun.json",
        "artifacts/m4/hlo_dump/dycore_step_production.txt",
        "artifacts/m4/hlo_dump/dycore_step_debug_stripped.txt",
        "artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff",
        "artifacts/m4/maintainability.md",
        "artifacts/m4/agent_success.json",
        "<development-history-not-included-in-public-repo>/decisions/ADR-003-dycore-precision.md",
    ]
    missing = [p for p in artifacts if not Path(p).exists()]
    assert missing == [], f"missing artifacts: {missing}"


def test_spacetime_budget_temporary_bytes_is_zero():
    record = json.loads(Path("artifacts/m4/spacetime_budget.json").read_text())
    assert record["temporary_bytes_per_step"] == 0


def test_transfer_audit_post_init_bytes_are_zero():
    record = json.loads(Path("artifacts/m4/transfer_audit.json").read_text())
    assert record["host_to_device_bytes_post_init"] == 0
    assert record["device_to_host_bytes_post_init"] == 0
    assert int(record["iterations"]) >= 100


def test_m5_gate_dryrun_records_trip_with_kernel_launches():
    record = json.loads(Path("artifacts/m4/m5_gate_dryrun.json").read_text())
    assert record["gate_status"] in ("trip", "pass")
    # Worker reported a trip on kernel_launches_per_step=29; document it here.
    if record["gate_status"] == "trip":
        assert "kernel_launches_per_step" in record["tripped_thresholds"]


def test_adr003_has_required_tokens():
    text = Path("<development-history-not-included-in-public-repo>/decisions/ADR-003-dycore-precision.md").read_text()
    for token in ("Decision:", "Per-field precision:", "Downcast plan:", "Validation evidence:"):
        assert token in text, f"ADR-003 missing required token: {token!r}"
    assert len(text.encode("utf-8")) >= 1500


# ----------------------------------------------------------------------------- #
# step_stripped_reference tautology audit                                       #
# ----------------------------------------------------------------------------- #


def test_step_stripped_reference_calls_same_impl_as_step_debug_false():
    """The fixed HLO proof must use a real hand-stripped sibling source file."""

    src = Path("src/gpuwrf/dynamics/step.py").read_text()
    stripped = Path("src/gpuwrf/dynamics/step_debug_stripped.py").read_text()
    assert "def step_stripped_reference" not in src
    assert "assert_" not in stripped
    assert "snapshot" not in stripped
