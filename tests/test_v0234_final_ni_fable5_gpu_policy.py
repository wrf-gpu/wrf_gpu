"""Adversarial CPU gates for the bounded GPU policy postprocessor."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from scripts import v0234_final_ni_fable5_gpu_policy_proof as policy


def test_pooled_rmse_is_not_maximum_per_frame_rmse() -> None:
    count = 10_000
    frame_rmse = (2.143740471999841, 0.1, 0.1)
    sum_squares = [value * value * count for value in frame_rmse]
    pooled = policy.pooled_rmse(sum_squares, [count, count, count])
    assert math.isclose(
        pooled,
        math.sqrt(sum(value * value for value in frame_rmse) / 3.0),
        rel_tol=0.0,
        abs_tol=2.0e-15,
    )
    assert pooled < 1.5 < frame_rmse[0]


def test_pooled_rmse_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        policy.pooled_rmse([], [])
    with pytest.raises(ValueError):
        policy.pooled_rmse([1.0], [0])
    with pytest.raises(ValueError):
        policy.pooled_rmse([float("nan")], [1])
    with pytest.raises(ValueError):
        policy.pooled_rmse([1.0], [1, 1])


def test_raw_red_parser_cannot_hide_a_post_dispatch_gate() -> None:
    assert policy.raw_red_coordinates(
        ["00:00:T2:rmse=2.143740471999841"]
    ) == (("00:00", "T2"),)
    assert policy.raw_red_coordinates(
        ["00:00:T2:rmse=2.14", "00:20:V:rmse=2.0"]
    ) == (("00:00", "T2"), ("00:20", "V"))
    with pytest.raises(ValueError):
        policy.raw_red_coordinates(["T2"])


def test_frozen_policy_and_known_initial_false_red_are_authenticated() -> None:
    assert hashlib.sha256(policy.MANAGER_POLICY.read_bytes()).hexdigest() == policy.MANAGER_POLICY_SHA
    assert hashlib.sha256(policy.FROZEN_PAIR_STATE.read_bytes()).hexdigest() == policy.FROZEN_PAIR_STATE_SHA
    frozen = json.loads(policy.FROZEN_PAIR_STATE.read_text())
    identity = frozen["identity_policy"]
    assert identity["aggregation"].startswith("pooled_rmse_across_all_")
    assert tuple(identity["strict_fields"]) == policy.STRICT_FIELDS
    assert identity["strict_rmse_limits"] == {
        "PSFC": 120.0,
        "T": 1.5,
        "T2": 1.5,
        "U": 1.8,
        "U10": 1.5,
        "V": 1.8,
        "V10": 1.5,
        "W": 0.3,
    }

    initial = json.loads(policy.INITIAL_FRAME_PROOF.read_text())
    assert policy.canonical_hash(initial) == policy.INITIAL_FRAME_PROOF_SHA
    assert initial["proof_sha256"] == policy.INITIAL_FRAME_PROOF_SHA
    assert initial["passed"] is True
    assert initial["candidate"]["sha256"] == policy.INITIAL_GPU_FRAME_SHA
    assert initial["d03_full_pair"]["strict_rmse"]["T2"] > 1.5
