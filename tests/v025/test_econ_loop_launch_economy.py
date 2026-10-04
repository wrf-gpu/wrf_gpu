"""Gates for the 2026-09-17 loop-launch-economy sprint (worker/glm/loop-launch-economy).

Contract: .agent/sprints/2026-09-17-v0250-loop-launch-economy/CONTRACT.md

The sprint's terminal state (see WORKER_REPORT.md in the sprint folder):

* the documented Round-3 loop-body leftover (65 sites / 494,216 elements per
  acoustic substep) does NOT exist on this tree -- the fresh census measures
  ZERO in-loop conversions on the RK-step fixture in BOTH precision modes, so
  G3 versus the documented baseline is met with the population absent;
* the shipped tree is byte-identical to the pre-change source: the one
  candidate edit (step-scope specified-relax hoist, commit ``4d189d6b2``) cut
  -98 launches / -72 converts but VIOLATED the fp64 bitwise gate (max-abs
  10.47, fusion-context-dependent FMA contraction) and was reverted
  (``8f71b5563``). The violation is pinned here as a regression guard for the
  M2 track.

All tests re-verify committed artifacts mechanically (no tolerances, NaN
payload mismatches fail). Artifacts: ``proofs/v025/econ/`` produced by the
committed harness ``.agent/sprints/2026-09-17-v0250-loop-launch-economy/
econ_census.py`` (one tool both sides).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROOFS = REPO / "proofs" / "v025" / "econ"
SPRINT = REPO / ".agent" / "sprints" / "2026-09-17-v0250-loop-launch-economy"

#: Documented Round-3 leftover the sprint was dispatched to cut (CONTRACT G3).
DOCUMENTED_BASELINE_SITES = 65
DOCUMENTED_BASELINE_ELEMENTS_PER_SUBSTEP = 494_216

ARMS = (
    ("rk_step_r3", "rk_step_r3"),
    ("production_chunk", "production_chunk"),
)
MODES = ("", "_mixed")


def _census(label: str) -> dict:
    path = PROOFS / f"{label}.json"
    if not path.is_file():
        pytest.skip(f"census artifact not committed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _compare(name: str) -> dict:
    path = PROOFS / name
    if not path.is_file():
        pytest.skip(f"comparison artifact not committed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# instrument anchor: the tool reproduces the documented fp64 CONTROL           #
# --------------------------------------------------------------------------- #
def test_census_instrument_anchors_to_documented_fp64_control():
    """pre launch proxy must equal the documented fp64 CONTROL (3510)."""

    pre = _census("pre_rk_step_r3")["program"]["static_launch_proxy"]
    assert pre == 3510, f"census launch proxy {pre} != documented fp64 CONTROL 3510"


def test_documented_round3_leftover_is_absent_on_this_tree():
    """G3 vs the documented baseline: the 65/494,216 population is gone.

    Both precision modes must show ZERO conversions inside any while body of
    the RK-step fixture, on both measured trees, i.e. a 100% reduction of the
    documented 65 sites / 494,216 elements per acoustic substep.
    """

    for label in ("pre_rk_step_r3", "post_rk_step_r3",
                  "pre_rk_step_r3_mixed", "post_rk_step_r3_mixed"):
        loop = _census(label)["program"]["converts_in_loop"]
        assert loop["in_loop"] == 0, (
            f"{label}: {loop['in_loop']} in-loop converts reappeared "
            f"(documented leftover was {DOCUMENTED_BASELINE_SITES} sites)"
        )
        assert loop["in_loop_converted_elements"] == 0, (
            f"{label}: {loop['in_loop_converted_elements']} in-loop converted "
            f"elements reappeared (documented leftover was "
            f"{DOCUMENTED_BASELINE_ELEMENTS_PER_SUBSTEP} per substep)"
        )


# --------------------------------------------------------------------------- #
# G1: committed bitwise comparisons (shipped tree vs pre-change tree)          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "report",
    [
        "compare_fp64_anchor.json",
        "compare_fp64_production.json",
        "compare_mixed_anchor.json",
        "compare_mixed_production.json",
    ],
)
def test_pre_post_outputs_are_bitwise_identical(report: str):
    payload = _compare(report)
    assert payload["only_left"] == [] and payload["only_right"] == []
    for name, field in payload["fields"].items():
        assert field["nan_mismatch"] is False, f"{report}:{name} NaN mismatch"
        assert field["bitwise"] is True, (
            f"{report}:{name} differs (max_abs_diff={field['max_abs_diff']!r})"
        )
    assert payload["all_bitwise_identical"] is True


# --------------------------------------------------------------------------- #
# G2: launch proxy non-increase on every measured arm                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("prefix, arm", ARMS)
def test_launch_proxy_does_not_increase(mode: str, prefix: str, arm: str):
    pre = _census(f"pre_{prefix}{mode}")["program"]["static_launch_proxy"]
    post = _census(f"post_{prefix}{mode}")["program"]["static_launch_proxy"]
    assert post <= pre, f"{prefix}{mode}: launch proxy regressed {pre} -> {post}"


# --------------------------------------------------------------------------- #
# the reverted E1 hoist is pinned as a negative result                         #
# --------------------------------------------------------------------------- #
def test_e1_hoist_violation_is_recorded_and_fails_bitwise():
    """The step-scope relax hoist must stay reverted.

    Commit ``4d189d6b2`` (reverted by ``8f71b5563``) traced the step-constant
    specified relax bundle once instead of once per RK stage: -98 launch proxy,
    -72 program converts, but the fp64 RK-step outputs drifted (max-abs 10.47;
    XLA FMA contraction is fusion-context-dependent). The committed comparison
    must keep recording that violation so the M2 fused-module track inherits
    the constraint instead of rediscovering it.
    """

    payload = _compare("compare_e1_applied_anchor_VIOLATION.json")
    assert payload["all_bitwise_identical"] is False
    assert payload["max_abs_diff_overall"] > 0.0


def test_worker_tree_does_not_contain_the_reverted_hoist():
    from gpuwrf.runtime import operational_mode
    import inspect

    source = inspect.getsource(operational_mode._rk_scan_step)
    step_scope = source.split("def advance_stage", 1)[0]
    assert "specified_frozen_relax" not in step_scope and "specified_frozen_relax" not in source
    assert step_scope.count("_specified_bdy_relax(") == 0  # hoist removed


# --------------------------------------------------------------------------- #
# honest boundary: the production chunk's residual population is reported      #
# --------------------------------------------------------------------------- #
def test_production_chunk_in_loop_population_is_reported_not_claimed():
    """The physics-owned in-loop conversions exist and are handed over."""

    post = _census("post_production_chunk")["program"]["converts_in_loop"]
    assert post["in_loop"] > 0
    proposals = SPRINT / "PATCH_PROPOSALS.md"
    text = proposals.read_text(encoding="utf-8") if proposals.is_file() else ""
    for owner in ("rrtmg", "mynn", "physics_couplers"):
        assert owner in text.lower(), (
            f"PATCH_PROPOSALS.md must report the out-of-scope in-loop owner {owner}"
        )
