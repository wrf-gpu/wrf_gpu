"""physconv sprint (2026-09-18) focused gates: C1 BouLac first-crossing reformulation.

Mechanical re-verification, no tolerances:
  * the new argmax first-crossing mask is VALUE-IDENTICAL to the reverted
    cumsum(astype(int32)) == 1 rank on adversarial mask populations
    (integer/bool machinery only -- fp operands untouched);
  * the committed census JSONs show the C1 census delta and the bitwise G1
    compares for the shipped arms (same econ_census.py instrument both sides).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SPRINT = REPO / ".agent/sprints/2026-09-17-v0250-physics-convert-elimination"
NZ = 44


def _reference_up_first(up_cross):
    """The reverted formulation (commit fe53d6d16^): rank by int32 cumsum."""
    return up_cross & (jnp.cumsum(up_cross.astype(jnp.int32), axis=-1) == 1)


def _reference_do_first(do_cross):
    return do_cross & (
        jnp.cumsum(do_cross[..., ::-1].astype(jnp.int32), axis=-1)[..., ::-1] == 1
    )


def _candidate_up_first(up_cross):
    j_idx = jnp.arange(NZ)
    return up_cross & (j_idx[None, None, :] == jnp.argmax(up_cross, axis=-1, keepdims=True))


def _candidate_do_first(do_cross):
    j_idx = jnp.arange(NZ)
    return do_cross & (
        j_idx[None, None, :] == (NZ - 1 - jnp.argmax(do_cross[..., ::-1], axis=-1, keepdims=True))
    )


@pytest.mark.parametrize("seed,population", [(0, 0.02), (1, 0.2), (2, 0.5), (3, 0.9)])
def test_first_crossing_mask_value_identity_random(seed, population):
    rng = np.random.default_rng(seed)
    cross = jnp.asarray(rng.random((13, NZ, NZ)) < population)
    for reference, candidate in (
        (_reference_up_first, _candidate_up_first),
        (_reference_do_first, _candidate_do_first),
    ):
        assert np.array_equal(np.asarray(reference(cross)), np.asarray(candidate(cross)))


def test_first_crossing_mask_value_identity_edge_cases():
    all_false = jnp.zeros((5, NZ, NZ), dtype=bool)
    all_true = jnp.ones((5, NZ, NZ), dtype=bool)
    single_mid = jnp.zeros((5, NZ, NZ), dtype=bool).at[:, 7, 7].set(True)
    first_col = jnp.zeros((5, NZ, NZ), dtype=bool).at[:, :, 0].set(True)
    last_col = jnp.zeros((5, NZ, NZ), dtype=bool).at[:, :, -1].set(True)
    two = jnp.zeros((5, NZ, NZ), dtype=bool).at[:, 3, 5].set(True).at[:, 3, 40].set(True)
    for cross in (all_false, all_true, single_mid, first_col, last_col, two):
        for reference, candidate in (
            (_reference_up_first, _candidate_up_first),
            (_reference_do_first, _candidate_do_first),
        ):
            assert np.array_equal(np.asarray(reference(cross)), np.asarray(candidate(cross)))


def test_boulac_length_dense_shapes_and_finiteness():
    """Synthetic physical column inputs: finite lb1/lb2, preserved shapes."""
    from gpuwrf.physics.mynn_pbl import _boulac_length_dense

    rng = np.random.default_rng(11)
    columns = 97
    zw = np.cumsum(rng.uniform(20.0, 90.0, (columns, NZ)), axis=-1)
    dz = rng.uniform(20.0, 90.0, (columns, NZ))
    qtke = rng.uniform(0.0, 1.5, (columns, NZ))
    theta = 290.0 + rng.uniform(-2.0, 2.0, (columns, NZ))
    lb1, lb2 = _boulac_length_dense(
        jnp.asarray(zw), jnp.asarray(dz), jnp.asarray(qtke), jnp.asarray(theta)
    )
    assert lb1.shape == lb2.shape == (columns, NZ)
    assert bool(np.all(np.isfinite(np.asarray(lb1))))
    assert bool(np.all(np.isfinite(np.asarray(lb2))))
    assert bool(np.all(np.asarray(lb1) >= 0.0)) and bool(np.all(np.asarray(lb2) >= 0.0))


def _load(name):
    return json.loads((SPRINT / "census" / name).read_text())


def test_census_c1_reduces_in_loop_converts_on_production_arm():
    pre = _load("pre_production_chunk.json")["program"]["converts_in_loop"]
    post = _load("c1_applied_production_chunk.json")["program"]["converts_in_loop"]
    assert post["in_loop"] < pre["in_loop"]
    assert post["in_loop_converted_elements"] < pre["in_loop_converted_elements"]


def test_census_c1_keeps_rk_anchor_unchanged():
    for name in ("rk_step_r3", "rk_step_r3_mixed"):
        pre = _load(f"pre_{name}.json")["program"]
        post = _load(f"c1_applied_{name}.json")["program"]
        assert post["static_launch_proxy"] == pre["static_launch_proxy"]
        assert (
            post["converts_in_loop"]["total"] == pre["converts_in_loop"]["total"]
        )


def test_census_c1_bitwise_compares_green():
    for name in (
        "compare_c1_rk_step_r3",
        "compare_c1_rk_step_r3_mixed",
        "compare_c1_production_chunk",
        "compare_c1_production_chunk_mixed",
    ):
        report = _load(f"{name}.json")
        assert report["all_bitwise_identical"] is True, name
        assert report["max_abs_diff_overall"] == 0.0, name
