"""CPU proof for the MANAGER-ACCEPTED B2 bounded terminal criterion.

The prior harness gated on strict terminal bit identity, which is stricter than
(i) the project's non-bitwise validation philosophy and (ii) the model's OWN
eager-vs-eager XLA-recompilation floor. The accepted replacement: exact ordering
+ finite + fused(K=1)-vs-eager(K=0) terminal difference within the in-run
recompile floor. This module proves the pure gate logic and that, on the corrected
finite all-7 fixture, the real fused-vs-eager difference sits BELOW that floor.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import pytest

from scripts.v0234_event_aware_fusion_probe import _bounded_terminal_verdict


def test_bounded_gate_accepts_reassociation_scale_difference():
    # Real measured values (B2_FUSED_EAGER_DISCRIMINATOR.json + eager-floor probe).
    v = _bounded_terminal_verdict(
        fused_vs_eager_state=8.881784197001252e-11,
        fused_vs_eager_carry=1.0693911463022232e-06,
        eager_floor_state=1.0186340659856796e-10,
        eager_floor_carry=1.132219040300697e-06,
    )
    assert v["passed"] is True
    assert v["state_ok"] and v["carry_ok"]
    # fused-vs-eager is actually BELOW the floor (ratio < 1), not merely within 4x.
    assert v["state_ratio_to_floor"] < 1.0
    assert v["carry_ratio_to_floor"] < 1.0


def test_bounded_gate_rejects_algorithmic_bug_scale_difference():
    # A real scheduling/boundary bug lands orders of magnitude above the floor.
    v = _bounded_terminal_verdict(
        fused_vs_eager_state=1.0e-2,
        fused_vs_eager_carry=5.0e-3,
        eager_floor_state=1.0e-10,
        eager_floor_carry=1.0e-6,
    )
    assert v["passed"] is False
    assert v["state_ok"] is False


def test_bounded_gate_fails_closed_on_degenerate_zero_floor():
    # Zero floor + nonzero diff must NOT pass (no divergence slips through).
    assert _bounded_terminal_verdict(
        fused_vs_eager_state=1e-12,
        fused_vs_eager_carry=0.0,
        eager_floor_state=0.0,
        eager_floor_carry=0.0,
    )["passed"] is False
    # Zero floor + zero diff is bit-identical -> pass.
    assert _bounded_terminal_verdict(
        fused_vs_eager_state=0.0,
        fused_vs_eager_carry=0.0,
        eager_floor_state=0.0,
        eager_floor_carry=0.0,
    )["passed"] is True


def test_bounded_gate_respects_factor_boundary():
    # Exactly at 4x floor passes; just above fails.
    assert _bounded_terminal_verdict(
        fused_vs_eager_state=4.0e-10, fused_vs_eager_carry=0.0,
        eager_floor_state=1.0e-10, eager_floor_carry=1.0,
    )["state_ok"] is True
    assert _bounded_terminal_verdict(
        fused_vs_eager_state=4.0001e-10, fused_vs_eager_carry=0.0,
        eager_floor_state=1.0e-10, eager_floor_carry=1.0,
    )["state_ok"] is False


@pytest.mark.slow
def test_real_fixture_fused_vs_eager_is_within_recompile_floor():
    """Integration: on the corrected finite all-7 fixture, the real fused-vs-eager
    terminal difference is bounded by the model's own in-run recompile floor.

    Mirrors the probe's terminal decomposition exactly (control=K0 on the
    fused-compiled runtime; candidate=K1; floor_ref=K0 on a fresh runtime whose
    fused program is never compiled)."""
    import jax
    import numpy as np

    from scripts.v0234_event_aware_fusion_probe import (
        SYNTHETIC_LEAF_ALARM_STEP,
        _build_all7_minigrid,
        _tree_maxabs,
    )
    from dataclasses import replace as dataclass_replace

    from gpuwrf.runtime.domain_tree import (
        _prepare_operational_domain_tree_runtime,
        run_operational_domain_tree,
    )
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run

    tree = _build_all7_minigrid()
    carries = {
        n: _initial_carry_for_run(b.state, b.namelist)
        for n, b in tree.domains.items()
    }
    names = tuple(tree.hierarchy.order)
    init = {n: 0 for n in names}
    alarms = {n: (SYNTHETIC_LEAF_ALARM_STEP,) for n in names[2:]}

    def out_cb(name, step, _s):
        return (name, int(step))

    def run(rt, k):
        r = run_operational_domain_tree(
            tree, root_steps=3, feedback_enabled=False, output=out_cb,
            output_alarm_steps=alarms, block_between=False, root_sync_cadence=1,
            carries=carries, initial_own_steps=init, max_event_tail=None,
            prepared_runtime=rt, event_aware_fusion_k=k,
        )
        jax.block_until_ready(tuple(s.theta for s in r.states.values()))
        return r

    # Matches the corrected probe: control/candidate on the fused-present runtime;
    # the eager floor reference toggles the fused pathway OFF (fused_cascade=None).
    runtime = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    runtime.fused_cascade("d02")  # compile fused (matches probe)
    control = run(runtime, 0)
    candidate = run(runtime, 1)
    floor_ref = run(dataclass_replace(runtime, fused_cascade=None), 0)

    # ordering exact + candidate genuinely fuses 8/9 + floor stays eager on same schedule
    assert control.events == candidate.events
    assert control.outputs == candidate.outputs
    assert control.own_steps == candidate.own_steps
    assert candidate.cascade_counts == {"fused:d02": 8, "event_fallback:d02": 1}
    assert not floor_ref.cascade_counts
    assert floor_ref.own_steps == control.own_steps

    v = _bounded_terminal_verdict(
        fused_vs_eager_state=_tree_maxabs(jax, np, control.states, candidate.states),
        fused_vs_eager_carry=_tree_maxabs(jax, np, control.carries, candidate.carries),
        eager_floor_state=_tree_maxabs(jax, np, control.states, floor_ref.states),
        eager_floor_carry=_tree_maxabs(jax, np, control.carries, floor_ref.carries),
    )
    assert v["passed"] is True, v
    assert v["eager_floor_state_maxabs"] > 0.0  # floor is genuinely nonzero
    assert v["state_ratio_to_floor"] <= 4.0
