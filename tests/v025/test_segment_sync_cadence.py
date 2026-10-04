"""GPUWRF_SEGMENT_SYNC_EVERY sync-cadence knob (sprint 2026-09-18-v0250-sync-pipelining).

CPU-focused unit coverage of the segmented host loop's sync cadence: the knob
must defer ``jax.block_until_ready`` to every K-th segment (final segment always
syncs), must reject K <= 0, and must leave outputs bitwise identical (pure
scheduling change). GPU end-to-end bitwise proof lives in the sprint folder
A/B sweep; this file only pins the scheduling semantics and validation.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented

# 10 steps at dt=0.1s, segment_steps=4 -> segments of (4, 4, 2) = 3 segments.
_HOURS = 1.0 / 3600.0
_SEGMENT_STEPS = 4
_SEGMENTS = 3
# syncs per run by K: K=1 -> {1,2,3}; K=2 -> {2,3}; K=3 -> {3(final)}; K=4 -> {3(final)}
_EXPECTED_SYNCS = {1: 3, 2: 2, 3: 1, 4: 1}


def _fixture():
    setup = build_warm_bubble_setup(require_gpu=False)
    namelist = dataclasses.replace(
        setup.namelist,
        run_physics=False,
        run_boundary=False,
        const_nu_m2_s=0.0,
        diff_6th_opt=0,
        km_opt=0,
        dt_s=0.1,
        acoustic_substeps=1,
        disable_guards=True,
    )
    return setup.state, namelist


def _run(state, namelist, sync_every: int, block_calls: list[int]) -> object:
    real_block = jax.block_until_ready

    def spy(tree, *args, **kwargs):
        block_calls.append(1)
        return real_block(tree, *args, **kwargs)

    jax.block_until_ready = spy  # monkeypatch.setattr on the module attr also works
    try:
        return run_forecast_operational_segmented(
            state, namelist, _HOURS, segment_steps=_SEGMENT_STEPS
        )
    finally:
        jax.block_until_ready = real_block


def _leaves(state) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)]


@pytest.mark.parametrize("sync_every", sorted(_EXPECTED_SYNCS))
def test_sync_cadence_defers_blocks_and_keeps_outputs_identical(
    monkeypatch, sync_every: int
) -> None:
    monkeypatch.setenv("GPUWRF_SEGMENT_SYNC_EVERY", str(sync_every))
    state, namelist = _fixture()
    block_calls: list[int] = []
    out = _run(state, namelist, sync_every, block_calls)
    jax.block_until_ready(out.theta)
    # The loop-level blocks are the ONLY blocking calls on this entry path.
    assert len(block_calls) == _EXPECTED_SYNCS[sync_every]
    for leaf in _leaves(out):
        assert np.all(np.isfinite(leaf))


def test_sync_cadence_outputs_bitwise_identical_across_k() -> None:
    reference = None
    for sync_every in sorted(_EXPECTED_SYNCS):
        state, namelist = _fixture()
        out = run_forecast_operational_segmented(
            state, namelist, _HOURS, segment_steps=_SEGMENT_STEPS
        )
        jax.block_until_ready(out.theta)
        leaves = _leaves(out)
        if reference is None:
            reference = leaves
        else:
            assert len(leaves) == len(reference)
            for got, want in zip(leaves, reference):
                assert got.tobytes() == want.tobytes()  # bitwise, not just allclose


@pytest.mark.parametrize("bad", ["0", "-1", "-3"])
def test_sync_cadence_rejects_nonpositive_k(monkeypatch, bad: str) -> None:
    monkeypatch.setenv("GPUWRF_SEGMENT_SYNC_EVERY", bad)
    state, namelist = _fixture()
    with pytest.raises(ValueError, match="GPUWRF_SEGMENT_SYNC_EVERY"):
        run_forecast_operational_segmented(
            state, namelist, _HOURS, segment_steps=_SEGMENT_STEPS
        )


def test_sync_cadence_default_is_every_segment(monkeypatch) -> None:
    monkeypatch.delenv("GPUWRF_SEGMENT_SYNC_EVERY", raising=False)
    state, namelist = _fixture()
    block_calls: list[int] = []
    out = _run(state, namelist, 1, block_calls)
    jax.block_until_ready(out.theta)
    assert len(block_calls) == _SEGMENTS
