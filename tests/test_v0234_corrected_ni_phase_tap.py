from __future__ import annotations

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.core.acoustic import (
    PHASE_TAP_SCRATCH_FIELDS,
    PHASE_TAP_SUMMARY_METRICS,
    _phase_tap_array_summary,
    acoustic_substep_core,
)
from gpuwrf.runtime.operational_mode import (
    _advance_chunk_fori,
    _rk_scan_step,
    advance_one_step_with_corrected_ni_phase_tap,
)
from scripts import v0234_corrected_ni_phase_tap as tap


def test_bounded_summary_has_exact_metrics_and_nonfinite_locations() -> None:
    value = jnp.asarray([3.0, -4.0, jnp.nan, jnp.inf, -2.0], dtype=jnp.float64)
    got = np.asarray(_phase_tap_array_summary(value), dtype=np.float64)

    np.testing.assert_array_equal(got, np.asarray([2.0, 2.0, 4.0, 1.0]))
    assert PHASE_TAP_SCRATCH_FIELDS == ("t_2ave", "ww", "mudf", "muave", "muts")
    assert PHASE_TAP_SUMMARY_METRICS == (
        "nonfinite_count",
        "first_nonfinite_flat_index",
        "max_abs_finite",
        "max_abs_finite_flat_index",
    )
    assert 2 * len(PHASE_TAP_SCRATCH_FIELDS) * len(PHASE_TAP_SUMMARY_METRICS) == 40


def test_all_finite_summary_uses_minus_one_first_bad() -> None:
    got = np.asarray(
        _phase_tap_array_summary(jnp.asarray([-1.0, 7.0, 2.0], dtype=jnp.float64))
    )
    np.testing.assert_array_equal(got, np.asarray([0.0, -1.0, 7.0, 1.0]))


def test_two_boundaries_are_after_normal_paths_and_before_pressure() -> None:
    source = inspect.getsource(acoustic_substep_core)
    positions = [
        source.index("state_for_w = _maybe_exchange_sharded_acoustic_halos(state_for_w)"),
        source.index("post_advance_mu_t_summary ="),
        source.index("ph_next = spec_bdyupdate_ph_inloop("),
        source.index("w_solved = _specified_w_zero_grad_work("),
        source.index(
            "state_for_pressure = _maybe_exchange_sharded_acoustic_halos(state_for_pressure)"
        ),
        source.index("post_advance_w_summary ="),
        source.index("p_rho = calc_p_rho_step("),
    ]
    assert positions == sorted(positions)
    interval = source[positions[0]:positions[-1]]
    assert "observe_mass_primitive=True" not in interval
    assert "capture_phase_tap: bool = False" in source


def test_tap_targets_only_rk3_first_substep_and_public_call_is_one_step() -> None:
    rk_source = inspect.getsource(_rk_scan_step)
    assert (
        "carry = advance_stage(carry, stages[0])\n"
        "        carry = advance_stage(carry, stages[1])\n"
        "        return advance_stage(carry, stages[2], capture_stage_phase_tap=True)"
    ) in rk_source
    public_source = inspect.getsource(advance_one_step_with_corrected_ni_phase_tap)
    assert public_source.count("_physics_boundary_step_with_phase_tap(") == 1
    assert "lax.scan" not in public_source
    assert "lax.fori_loop" not in public_source


def test_ordinary_production_entry_has_no_phase_tap_lane() -> None:
    source = inspect.getsource(_advance_chunk_fori)
    assert "phase_tap" not in source
    assert "advance_one_step_with_corrected_ni_phase_tap" not in source


def test_runner_gates_summary_decode_after_complete_carry_identity() -> None:
    source = inspect.getsource(tap.main)
    identity = source.index("identity = ordinary.compare_manifests")
    gate = source.index("if exact_identity:")
    decode = source.index("decode_admissible_summary(summary_host)")
    discard = source.index('"status": "DISCARDED_UNREAD"')
    assert identity < gate < decode < discard
    assert source.count("executable(") == 1
    assert '"ordinary_reference_model_calls": 0' in source
    assert '"tap_model_calls": model_calls' in source


def test_contract_freezes_a_arm_and_identity_manifest() -> None:
    contract = Path(
        ".agent/sprints/2026-07-13-v0234-corrected-ni-phase-tap/sprint-contract.md"
    ).read_text()
    assert "The retained ordinary step-9314 output is the A arm" in contract
    assert tap.INPUT_FILE_SHA256 in contract
    assert tap.REFERENCE_MANIFEST_SHA256 in contract
    assert "DISCARDED_UNREAD" in contract


def test_phase_tap_result_schema_is_two_summary_leaves() -> None:
    # This is a tree/schema gate only; no model fixture or forecast compile runs.
    from gpuwrf.dynamics.core.acoustic import AcousticPhaseTapSummary

    summary = AcousticPhaseTapSummary(
        jnp.zeros((5, 4), dtype=jnp.float64),
        jnp.zeros((5, 4), dtype=jnp.float64),
    )
    leaves = jax.tree_util.tree_leaves(summary)
    assert len(leaves) == 2
    assert [tuple(leaf.shape) for leaf in leaves] == [(5, 4), (5, 4)]
