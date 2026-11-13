"""Child stochastic history at original-WRF phase on EVERY aligned frame (GPU arm 11.11 G12, FIX5).

Original WRF writes a child's history inside integrate(child), after the parent's next solve and the
parent->child SINT force-down (frame/module_integrate.F:375,393-430). When a child history time is also a
parent step start (D15: every 3 h with dt 54 s), the forced child patterns in the frame are therefore one
parent step ahead of the carry at the end of the previous child subcycle. The GPU arm measured exactly that
on d02/d03 at 03..21 h (rel 0.10-0.71) with every other lead at <= 1e-6.

This fixture runs the real scheduler (run_domain_tree_callbacks) with stochastic-only advance/force
callbacks and the real writer view, and compares every frame with an independent emulation of WRF's
integrate order (history before solve, force-down before each child integrate, final frame after the last
step without another force).
"""
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.integration.nested_pipeline import _PerDomainWrfoutWriter, _StochasticHistoryPhase
from gpuwrf.nesting.stochastic import force_stochastic
from gpuwrf.physics.stochastic_driver import advance_stochastic
from gpuwrf.runtime.domain_tree import run_domain_tree_callbacks
from test_stochastic_nest_leadzero import GRIDS, Carry, _leaves, _setup

ROOT_STEPS = 6
CADENCE = {"d01": 3, "d02": 2, "d03": 4}     # d02: 6, 12 aligned with d01; d03: 12, 24, 48 with d02, 36 with d01


def _callbacks(configs, weights):
    def step(name, stoch):
        g = GRIDS[name]
        return advance_stochastic(stoch, configs[name], dx=g.dx, dy=g.dy)

    def advance(name, carry, start, chunk):
        stoch = carry.stochastic
        for _ in range(int(chunk)):
            stoch = step(name, stoch)
        return carry.replace(stochastic=stoch)

    def force(edge, parent, child):
        return child.replace(stochastic=force_stochastic(child.stochastic, parent.stochastic,
                                                         weights[edge.spec.child], 3))
    return step, advance, force


def _wrf_reference(configs, carries, weights, step):
    """Independent original-WRF integrate order (no scheduler, no view)."""
    out, own, frames = dict(carries), {d: 0 for d in GRIDS}, {}
    kids = {"d01": ("d02",), "d02": ("d03",), "d03": ()}
    final = {"d01": ROOT_STEPS, "d02": ROOT_STEPS * 3, "d03": ROOT_STEPS * 9}

    def integrate(name, n):
        for _ in range(n):
            s = own[name]
            if 0 < s < final[name] and s % CADENCE[name] == 0:
                frames[(name, s)] = out[name].stochastic            # med_before_solve_io: before this step
            out[name] = out[name].replace(stochastic=step(name, out[name].stochastic))
            own[name] += 1
            for kid in kids[name]:
                out[kid] = out[kid].replace(stochastic=force_stochastic(out[kid].stochastic, out[name].stochastic,
                                                                        weights[kid], 3))
                integrate(kid, 3)

    integrate("d01", ROOT_STEPS)
    for d in GRIDS:                                                   # med_last_solve_io at the stop time
        if own[d] % CADENCE[d] == 0:
            frames[(d, own[d])] = out[d].stochastic
    return frames


class _Writer:
    wants_carry = True
    history_view = _PerDomainWrfoutWriter.history_view

    def __init__(self, phase):
        self.stochastic_phase = phase
        self.frames = {}

    def __call__(self, name, step, carry):
        self.frames[(name, step)] = carry.stochastic
        return (name, step)


def _run(phase):
    configs, carries, bundles, tree, w12, w23 = _setup()
    weights = {"d02": w12, "d03": w23}
    step, advance, force = _callbacks(configs, weights)
    hierarchy = DomainHierarchy.from_edges(("d01", "d02", "d03"), (DomainNest("d01", "d02", 3, 5, 4),
                                                                    DomainNest("d02", "d03", 3, 3, 3)), max_dom=3)
    writer = _Writer(_StochasticHistoryPhase(tree, bundles, final_root_step=ROOT_STEPS) if phase else None)
    result = run_domain_tree_callbacks(hierarchy, dict(carries), root_steps=ROOT_STEPS, advance=advance, force=force,
                                       output=writer, output_cadence_steps=CADENCE, block_between=False)
    return configs, carries, weights, step, writer, result


def _close(a, b):
    for x, y in zip(_leaves(a), _leaves(b)):
        scale = float(np.max(np.abs(y))) if y.size else 0.
        np.testing.assert_allclose(x, y, rtol=8e-6, atol=64 * np.finfo(np.float32).eps * scale)


def _rel(a, b):
    x, y = (np.asarray(a.patterns[2].field, np.float64), np.asarray(b.patterns[2].field, np.float64))
    return float(np.max(np.abs(x - y)) / np.max(np.abs(y)))


def test_every_child_frame_matches_the_original_wrf_integrate_order():
    configs, carries, weights, step, writer, result = _run(phase=True)
    want = _wrf_reference(configs, carries, weights, step)
    assert set(writer.frames) == set(want)
    aligned = {("d02", 6), ("d02", 12), ("d03", 36)}                  # child time == d01 step start (not final)
    assert aligned <= set(want) and ("d02", 18) in want and ("d03", 52) in want
    for key in sorted(want):
        _close(writer.frames[key], want[key])
    # The numerical carries are untouched by the view: rerun without it gives identical end carries.
    _, _, _, _, plain, plain_result = _run(phase=False)
    for d in GRIDS:
        for a, b in zip(_leaves(result.carries[d].stochastic), _leaves(plain_result.carries[d].stochastic)):
            np.testing.assert_array_equal(a, b)
    # Without the view the aligned frames carry the previous parent step (the measured G12 failure) ...
    for key in aligned:
        assert _rel(plain.frames[key], want[key]) > 1e-3, key
    # ... and every other frame (including the final d02 frame and d03 frames aligned only with d02) is unchanged.
    for key in set(want) - aligned:
        for a, b in zip(_leaves(plain.frames[key]), _leaves(writer.frames[key])):
            np.testing.assert_array_equal(a, b)


def test_view_refuses_a_parent_at_another_instant_and_ignores_non_stochastic_carries():
    configs, carries, bundles, tree, _, _ = _setup()
    phase = _StochasticHistoryPhase(tree, bundles, final_root_step=ROOT_STEPS)
    assert phase.final_step == {"d01": 6, "d02": 18, "d03": 54}
    with pytest.raises(ValueError, match="d02: history view at step 6 but d01 is at step 1"):
        phase.view("d02", 6, carries, {"d01": 1, "d02": 6, "d03": 0})
    assert phase.view("d02", 5, carries, {"d01": 1, "d02": 5}) is carries["d02"]      # inside a parent step
    assert phase.view("d02", 18, carries, {"d01": 6, "d02": 18}) is carries["d02"]    # final frame
    assert phase.view("d01", 3, carries, {"d01": 3}) is carries["d01"]                # root
    plain = {d: Carry(None, d) for d in GRIDS}
    assert phase.view("d02", 6, plain, {"d01": 2, "d02": 6}) is plain["d02"]
    writer = _Writer(None)
    assert writer.history_view("d02", 6, carries, {}) is carries["d02"]                # OFF: identity
