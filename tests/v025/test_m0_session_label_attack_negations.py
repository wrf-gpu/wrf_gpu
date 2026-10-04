"""Green successors for the eleven retired GPT exploit reproductions.

Each collected case is the safe negation of one historical failing assertion.
All subprocess attacks use real authority consumers, synthetic holders that
agree with the label under test, and fresh pytest-owned roots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"
TESTS = REPO / "tests" / "v025"
sys.path[:0] = [str(SCRIPTS), str(TESTS)]

import gpu_window_registry as registry  # noqa: E402
import m0_authority_guard_inventory as guard_inventory  # noqa: E402
import m0_exact_boundary_child as child  # noqa: E402
import m0_window_parent as window_parent  # noqa: E402
import test_m0_session_label_authority as authority_suite  # noqa: E402


SESSION_LABEL = registry.SESSION_LABEL
HELD_MODE = registry.HELD_LAUNCH_MODE
LEGACY_MODE = registry.LEGACY_LAUNCH_MODE
STAGE_LABELS = {
    stage: str(configuration["window_label"])
    for stage, configuration in registry.SESSION_STAGE_IDENTITIES.items()
}


def _assert_structured_unconsumed_refusal(outcome):
    payload = outcome["payload"]
    assert outcome["returncode"] == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["handoff_consumed"] is False
    assert outcome["handoff_remaining"] is True
    assert outcome["handoff_spent"] is False
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert payload["device_touched"] is False


WRONG_LABEL_CASES = (
    ("W1", STAGE_LABELS["W2"]),
    ("W2", STAGE_LABELS["W3"]),
    ("W3", STAGE_LABELS["W1"]),
    ("W1", "baseline-census"),
)


@pytest.mark.parametrize(
    ("stage_identity", "wrong_registered_label"),
    WRONG_LABEL_CASES,
    ids=[
        f"{stage_identity}-{wrong_registered_label}"
        for stage_identity, wrong_registered_label in WRONG_LABEL_CASES
    ],
)
def test_wrong_registered_parent_label_is_refused_by_the_real_child_relation(
    tmp_path, stage_identity, wrong_registered_label
):
    """Negate four historical wrong-label crossings with a neutral lock check."""

    root = tmp_path / f"wrong-{stage_identity}-{wrong_registered_label}"
    root.mkdir()
    outcome = authority_suite._drive_child(
        root,
        launch_mode=HELD_MODE,
        stage_identity=stage_identity,
        session_label=wrong_registered_label,
        holder_label=wrong_registered_label,
        exported_label=wrong_registered_label,
    )
    _assert_structured_unconsumed_refusal(outcome)
    assert "allow coordination label" in outcome["payload"]["reason"]
    assert "exported canonical-lock label differs" not in outcome["payload"]["reason"]


def _stage_mismatch(payload):
    payload["stage"] = "a_different_stage"
    return payload


def _drop_parent_endpoint(environment):
    environment.pop("GPUWRF_M0_PARENT_LAUNCH_NS")


def _empty_receipt_fingerprint(payload):
    payload["receipt_fingerprint"] = "   "
    return payload


FORMERLY_OMITTED_GUARDS = (
    (
        "handoff_stage_mismatch",
        {"handoff_edit": _stage_mismatch},
        "stage_name_matches_handoff",
        "authorization handoff stage=",
    ),
    (
        "missing_parent_launch_endpoint",
        {"environment_edit": _drop_parent_endpoint},
        "parent_launch_endpoint_is_integer",
        "parent launch monotonic endpoint is missing/invalid",
    ),
    (
        "empty_receipt_fingerprint",
        {"handoff_edit": _empty_receipt_fingerprint},
        "receipt_fingerprint_nonempty",
        "authorization receipt fingerprint is empty",
    ),
    (
        "symlink_handoff",
        {"handoff_shape": "symlink"},
        "handoff_is_not_a_symlink",
        "authorization handoff must not be a symlink",
    ),
)


@pytest.mark.parametrize(
    ("attack", "kwargs", "guard", "reason"),
    FORMERLY_OMITTED_GUARDS,
    ids=[case[0] for case in FORMERLY_OMITTED_GUARDS],
)
def test_formerly_omitted_guard_is_derived_and_refuses_on_the_real_child_cli(
    tmp_path, attack, kwargs, guard, reason
):
    """Negate four historical inventory-omission assertions dynamically."""

    root = tmp_path / attack
    root.mkdir()
    outcome = authority_suite._drive_child(root, **kwargs)
    _assert_structured_unconsumed_refusal(outcome)
    assert reason in outcome["payload"]["reason"]
    declared = set(child.SESSION_AUTHORITY_HARD_GATES)
    derived = set(guard_inventory.derive_pre_device_guard_inventory(SCRIPTS))
    assert guard in declared
    assert guard in derived


def test_inventory_is_mechanical_35_by_35_not_the_retired_16_by_25(tmp_path):
    """Negate the historical self-declared inventory assertion."""

    declared = set(child.SESSION_AUTHORITY_HARD_GATES)
    derived = set(guard_inventory.derive_pre_device_guard_inventory(SCRIPTS))
    mutated = {
        guard for _name, _kwargs, guard, _reason in authority_suite.MUTATIONS
    }
    assert len(declared) == 35
    assert len(derived) == 35
    assert len(mutated) == 35
    assert declared == derived == mutated

    root = tmp_path / "launch-mode-disagreement"
    root.mkdir()
    outcome = authority_suite._drive_child(
        root,
        launch_mode=HELD_MODE,
        cli_launch_mode=LEGACY_MODE,
    )
    _assert_structured_unconsumed_refusal(outcome)
    assert "allow coordination label" in outcome["payload"]["reason"]


def test_pallas_cross_stage_label_is_refused_before_native_execution(tmp_path):
    """Negate the historical Pallas cross-stage-label acceptance."""

    wrong_label = STAGE_LABELS["W1"]
    root = tmp_path / "pallas-cross-stage"
    root.mkdir()
    outcome = authority_suite._drive_pallas_native_consumer(
        root,
        label=wrong_label,
        stage_identity="W3",
    )
    _assert_structured_unconsumed_refusal(outcome)
    assert "allow coordination label" in outcome["payload"]["reason"]


def test_parent_and_real_child_both_enforce_the_context_join(tmp_path):
    """Negate the historical missing-context-join assertion behaviorally."""

    wrong_label = "pallas-sm120"
    authorization = {
        "window": "M0_CORE_SESSION",
        "label": wrong_label,
    }
    with pytest.raises(window_parent.WindowRefusal, match="allow coordination label"):
        window_parent.authorized_launch_context(
            authorization,
            stage_identity="W1",
            run_id=authority_suite._stage_run_id("W1"),
        )

    root = tmp_path / "child-context-join"
    root.mkdir()
    outcome = authority_suite._drive_child(
        root,
        launch_mode=HELD_MODE,
        stage_identity="W1",
        session_label=wrong_label,
        holder_label=wrong_label,
        exported_label=wrong_label,
    )
    _assert_structured_unconsumed_refusal(outcome)
    assert "allow coordination label" in outcome["payload"]["reason"]
    assert "exported canonical-lock label differs" not in outcome["payload"]["reason"]
