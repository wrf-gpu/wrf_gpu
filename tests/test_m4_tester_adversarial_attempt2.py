"""Attempt-2 adversarial tester tests for the M4 reduced JAX dycore.

These tests are owned by the sonnet-test-engineer role and probe the *specific*
blocker fixes the worker applied after attempt-1 reviewer Reject. Each test
targets one of the seven fixes the contract amendment required, plus several
new edge cases not previously covered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from gpuwrf.dynamics.advection import (
    advect_mass_scalar,
    compute_advection_tendencies,
)
from gpuwrf.dynamics.rk3 import rk3_step
from gpuwrf.dynamics.step import run, step
from gpuwrf.dynamics.step_debug_stripped import (
    run_debug_stripped,
    step_debug_stripped,
)
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.validation.tier1 import run_tier1
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid
from gpuwrf.validation.tier3 import convergence_record


# --------------------------------------------------------------------------- #
# Blocker #1 (rk3 constant tendency) — broader spot checks                    #
# --------------------------------------------------------------------------- #


def _zero_velocity_state(grid):
    """Helper: state with all velocities zero and constants elsewhere."""

    state, tendencies = density_current_state(grid)
    state = state.replace(
        u=jnp.zeros_like(state.u),
        v=jnp.zeros_like(state.v),
        w=jnp.zeros_like(state.w),
        theta=jnp.ones_like(state.theta) * 300.0,
        qv=jnp.ones_like(state.qv) * 1.0e-3,
        p=jnp.zeros_like(state.p),
        ph=jnp.zeros_like(state.ph),
        mu=jnp.ones_like(state.mu),
    )
    tendencies = tendencies.replace(
        u=jnp.zeros_like(tendencies.u),
        v=jnp.zeros_like(tendencies.v),
        w=jnp.zeros_like(tendencies.w),
        theta=jnp.zeros_like(tendencies.theta),
        qv=jnp.zeros_like(tendencies.qv),
        p=jnp.zeros_like(tendencies.p),
        ph=jnp.zeros_like(tendencies.ph),
        mu=jnp.zeros_like(tendencies.mu),
    )
    return state, tendencies


@pytest.mark.parametrize("dt,const", [(2.0, 1.0), (6.0, 0.7), (10.0, -1.25), (0.5, 11.0)])
def test_rk3_constant_theta_tendency_integrates_to_dt_times_tendency(dt, const):
    """Attempt-1 reproduced delta=11 for dt=6, const=1; attempt-2 must be ~dt*const."""

    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = _zero_velocity_state(grid)
    tendencies = tendencies.replace(theta=jnp.ones_like(tendencies.theta) * const)
    out = step(state, tendencies, grid, dt, n_acoustic=1, debug=False)
    block_until_ready(out)
    expected = dt * const
    residual = float(jnp.max(jnp.abs(out.theta - state.theta - expected)))
    assert residual <= 1.0e-12, f"dt={dt} const={const} expected delta {expected}, residual {residual}"


def test_rk3_multistep_run_constant_theta_tendency_integrates_linearly():
    """After N steps of constant theta tendency, theta should grow by N*dt*tendency."""

    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = _zero_velocity_state(grid)
    tendencies = tendencies.replace(theta=jnp.ones_like(tendencies.theta) * 0.5)
    n_steps = 7
    dt = 2.0
    out = run(state, tendencies, grid, dt, n_steps, n_acoustic=1, debug=False)
    block_until_ready(out)
    expected = float(n_steps) * dt * 0.5
    residual = float(jnp.max(jnp.abs(out.theta - state.theta - expected)))
    assert residual <= 1.0e-10


# --------------------------------------------------------------------------- #
# Blocker #2 (tier-1 dycore upwind oracle) — sibling fixture integrity        #
# --------------------------------------------------------------------------- #


def test_upwind5_sibling_fixture_checksum_matches_manifest():
    """Manifest's recorded sha256 must equal the on-disk sample file digest."""

    manifest_path = Path("fixtures/manifests/analytic-stencil-3d-upwind5-v1.yaml")
    sample_path = Path("fixtures/samples/analytic-stencil-3d-upwind5-v1.npz")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    advertised = manifest["files"][0]["checksum_sha256"]
    actual = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    assert advertised == actual


def test_upwind5_sibling_phi_next_matches_dycore_kernel_zero_error():
    """The committed phi_next_upwind5 must be byte-equal to what advect_mass_scalar produces."""

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
    candidate = np.asarray(phi + 3.0 * advect_mass_scalar(phi, u_mass, v_mass, w_mass, grid))
    assert float(np.max(np.abs(candidate - ref))) <= 1.0e-12


def test_tier1_artifact_advertises_dycore_operator_not_m1_reference():
    """The tier-1 artifact must self-identify as dycore upwind, not the M1 centred reference."""

    record = json.loads(Path("artifacts/m4/tier1_advection_parity.json").read_text())
    assert record["fixture_id"] == "analytic-stencil-3d-upwind5-v1"
    op = record["operator"].lower()
    assert "upwind" in op
    assert "5th-order" in op
    assert "3rd-order" in op
    assert "centered" not in op and "centred" not in op


# --------------------------------------------------------------------------- #
# Blocker #3 (tier-2 nontrivial trajectory)                                   #
# --------------------------------------------------------------------------- #


def test_tier2_ic_has_nonzero_u():
    """The fixed tier-2 IC must have a non-zero advection velocity somewhere."""

    grid = make_ideal_grid(4, 8, 8)
    state, _ = density_current_state(grid)
    assert float(jnp.max(jnp.abs(state.u))) > 0.0


def test_tier2_artifact_records_final_state_differs_from_initial():
    """The fixed tier-2 artifact must record final_state_differs_from_initial=True."""

    record = json.loads(Path("artifacts/m4/tier2_invariants.json").read_text())
    assert record["final_state_differs_from_initial"] is True
    assert record["max_theta_delta"] > 0.1
    assert record["mass_residual_relative"] <= 1.0e-10
    assert record["qv_positivity_violations"] == 0
    assert record["nan_inf_violations"] == 0
    assert record["pass"] is True


def test_tier2_qv_violation_counter_accumulates_over_trajectory():
    """If qv goes briefly negative mid-trajectory, the count must be nonzero even if final qv is fine."""

    # Manufacture a "bad" run: large negative qv tendency pulse, then back to zero.
    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    # Make qv slightly positive, then drive it strongly negative for several steps.
    state = state.replace(qv=jnp.ones_like(state.qv) * 1.0e-3)
    tendencies = tendencies.replace(qv=jnp.ones_like(tendencies.qv) * -1.0)
    from gpuwrf.validation.tier2 import _trajectory_with_checks

    final, qv_bad, finite_bad = _trajectory_with_checks(state, tendencies, grid, 1.0, 3, 1)
    block_until_ready(final)
    # qv started at 1e-3 and is being driven by -1 per second for 3 steps:
    # qv must go negative.
    assert int(qv_bad) > 0


# --------------------------------------------------------------------------- #
# Blocker #4 (tier-3 uses public run())                                       #
# --------------------------------------------------------------------------- #


def test_tier3_source_uses_public_run_and_not_centered_helper():
    """tier3.py must call the public `run` and not import or use `ddx4_centered`."""

    src = Path("src/gpuwrf/validation/tier3.py").read_text()
    assert "from gpuwrf.dynamics.step import run" in src
    assert "ddx4_centered" not in src
    assert "advect_mass_scalar(" not in src  # don't bypass through a private kernel


def test_tier3_convergence_record_observed_order_meets_expected_minus_half():
    """Re-run convergence; observed_order must be >= expected - 0.5."""

    record = convergence_record()
    assert record["observed_order"] >= record["expected_order"] - 0.5
    # Three resolution levels must be present.
    assert len(record["errors_per_level"]) == 3
    # Errors should monotonically decrease as resolution refines.
    errs = [lvl["l2_error"] for lvl in record["errors_per_level"]]
    assert errs[0] > errs[1] > errs[2]


# --------------------------------------------------------------------------- #
# Major #5 (literal hand-stripped sibling)                                    #
# --------------------------------------------------------------------------- #


def test_step_debug_stripped_is_a_separate_source_file():
    """step_debug_stripped.py must exist as a separate, hand-edited file."""

    stripped_path = Path("src/gpuwrf/dynamics/step_debug_stripped.py")
    assert stripped_path.exists()
    src = stripped_path.read_text()
    # No debug-hook calls in source.
    assert "assert_finite" not in src
    assert "assert_physical_bounds" not in src
    assert "snapshot(" not in src
    assert "from gpuwrf.debug" not in src


def test_step_debug_stripped_matches_step_debug_false_bitwise():
    """The stripped sibling must produce bitwise-identical output to step(..., debug=False)."""

    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    a = step(state, tendencies, grid, 0.25, n_acoustic=2, debug=False)
    b = step_debug_stripped(state, tendencies, grid, 0.25, n_acoustic=2)
    block_until_ready(a)
    block_until_ready(b)
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


def test_run_debug_stripped_matches_run_debug_false_bitwise():
    """The stripped loop sibling must also match `run(..., debug=False)` bitwise across N steps."""

    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    a = run(state, tendencies, grid, 0.25, 5, n_acoustic=2, debug=False)
    b = run_debug_stripped(state, tendencies, grid, 0.25, 5, n_acoustic=2)
    block_until_ready(a)
    block_until_ready(b)
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


# --------------------------------------------------------------------------- #
# Major #5.1 (m4_hlo_diff.py typo fix)                                        #
# --------------------------------------------------------------------------- #


def test_m4_hlo_diff_writes_stripped_hlo_to_stripped_artifact():
    """The fixed m4_hlo_diff.py must pass `stripped` (not `prod`) to the stripped artifact write."""

    src = Path("scripts/m4_hlo_diff.py").read_text()
    # Find the write for the stripped file and confirm the variable is `stripped`.
    target_lines = [line for line in src.splitlines() if 'dycore_step_debug_stripped.txt"' in line and "write_hlo" in line]
    assert target_lines, "stripped-HLO write line not found in m4_hlo_diff.py"
    for line in target_lines:
        assert ", stripped," in line, f"stripped artifact still receives prod variable: {line}"


def test_production_and_stripped_hlo_artifacts_are_distinct_files():
    """Attempt-1 typo wrote prod text to both files; on-disk stripped artifact must reference the stripped HLO module."""

    prod_path = Path("artifacts/m4/hlo_dump/dycore_step_production.txt")
    stripped_path = Path("artifacts/m4/hlo_dump/dycore_step_debug_stripped.txt")
    diff_path = Path("artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff")
    assert prod_path.exists() and stripped_path.exists()
    assert diff_path.read_bytes() == b""
    prod = prod_path.read_text()
    stripped = stripped_path.read_text()
    assert "HloModule" in prod and "HloModule" in stripped
    # On-disk stripped artifact must self-identify as the stripped entrypoint
    # (attempt-1 wrote `HloModule jit_step` to both files because of a
    # variable-name typo in m4_hlo_diff.py).
    assert "jit_step_debug_stripped" in stripped or "step_debug_stripped" in stripped
    assert "jit_step_debug_stripped" not in prod


# --------------------------------------------------------------------------- #
# Minor #6 (m5 gate dry-run null metrics)                                     #
# --------------------------------------------------------------------------- #


def test_m5_gate_dryrun_unknown_metrics_are_json_null_not_zero():
    """JSON null distinguishes "unknown" from "measured zero" — must not be zero."""

    record = json.loads(Path("artifacts/m4/m5_gate_dryrun.json").read_text())
    assert record["local_memory_bytes_per_kernel"] is None
    assert record["registers_per_kernel"] is None
    assert "null" in record["rationale"].lower() or "unavailable" in record["rationale"].lower()


def test_m5_gate_dryrun_rationale_documents_unavailable_metrics():
    """Rationale must explicitly state metrics are unavailable, per AC #6.fix."""

    record = json.loads(Path("artifacts/m4/m5_gate_dryrun.json").read_text())
    rat = record["rationale"].lower()
    assert "unavailable" in rat or "pending" in rat or "ncu" in rat


# --------------------------------------------------------------------------- #
# Hot-path discipline — re-verify on attempt 2                                #
# --------------------------------------------------------------------------- #


def _compile_step_hlo_at_unit_scale_debug_false():
    """Helper compiled outside the test function so its name does not poison HLO source-info."""

    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    return compiled_text(
        step.lower(state, tendencies, grid, 0.1, n_acoustic=1, debug=False).compile()
    ).lower()


def test_production_hlo_has_no_debug_branch_ops():
    """Constitutional gate at the dycore scale: production HLO has zero debug ops."""

    prod = _compile_step_hlo_at_unit_scale_debug_false()
    forbidden = ("is-finite", "debug.callback", "io_callback", "checkify")
    for token in forbidden:
        assert token not in prod, f"production HLO leaked {token!r}"
    # The HLO source-info embeds the calling Python frame's qualname, so
    # "isfinite" can appear in HLO text through that channel even when no
    # op of that name is present. To assert the absence of finiteness *ops*
    # we look for the canonical XLA spelling `is-finite` (with hyphen).


def test_production_hlo_size_does_not_explode_with_grid():
    """Trivial sanity guard: production HLO at unit scale should be reasonably small."""

    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    prod = compiled_text(
        step.lower(state, tendencies, grid, 0.1, n_acoustic=1, debug=False).compile()
    )
    # Empirical: attempt-2 ~ 30-60 KB lowered. Guard against an order-of-magnitude regression.
    assert len(prod) < 5_000_000


# --------------------------------------------------------------------------- #
# Vertical advection — no top/bottom wrap                                     #
# --------------------------------------------------------------------------- #


def test_vertical_upwind_uses_no_wrap_at_lid_and_floor():
    """A spike at z=0 must not appear at z=top after one upward-velocity step."""

    grid = make_ideal_grid(8, 4, 4)
    state, tendencies = density_current_state(grid)
    field = jnp.zeros_like(state.theta)
    spike = field.at[0, :, :].set(100.0)
    state = state.replace(
        theta=300.0 + spike,
        u=jnp.zeros_like(state.u),
        v=jnp.zeros_like(state.v),
        w=jnp.ones_like(state.w) * 1.0,
    )
    out = step(state, tendencies, grid, 0.05, n_acoustic=1, debug=False)
    block_until_ready(out)
    # Lid (z=last) cell must NOT have received the spike via wrap-around.
    top_minus_baseline = float(jnp.max(jnp.abs(out.theta[-1, :, :] - 300.0)))
    assert top_minus_baseline < 50.0, f"vertical wrap detected: top cell delta {top_minus_baseline}"


# --------------------------------------------------------------------------- #
# Determinism across rk3 and acoustic substep counts                          #
# --------------------------------------------------------------------------- #


def test_step_is_deterministic_across_n_acoustic_values():
    """Two different n_acoustic values give two compiled programs — both must be deterministic."""

    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    a1 = step(state, tendencies, grid, 0.1, n_acoustic=1, debug=False)
    a2 = step(state, tendencies, grid, 0.1, n_acoustic=1, debug=False)
    b1 = step(state, tendencies, grid, 0.1, n_acoustic=4, debug=False)
    b2 = step(state, tendencies, grid, 0.1, n_acoustic=4, debug=False)
    block_until_ready(a1); block_until_ready(a2); block_until_ready(b1); block_until_ready(b2)
    np.testing.assert_array_equal(np.asarray(a1.theta), np.asarray(a2.theta))
    np.testing.assert_array_equal(np.asarray(b1.theta), np.asarray(b2.theta))


# --------------------------------------------------------------------------- #
# Sprint scope discipline                                                     #
# --------------------------------------------------------------------------- #


def test_step_debug_stripped_is_not_exported_in_dynamics_init():
    """Contract AC #2.4 says the stripped sibling must NOT be exported from __init__.py."""

    init_src = Path("src/gpuwrf/dynamics/__init__.py").read_text()
    assert "step_debug_stripped" not in init_src
