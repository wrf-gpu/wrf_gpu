"""H1's rule must be sound in BOTH directions, not just the one I checked.

Manager review on main `6a497c0e`: the one-sided lower bound establishes a
positive minimum removable share but cannot infer a negative upper bound. Two of
my branches were unsound and are now INCONCLUSIVE:

* a lower bound below 30% does not show the true share is below 30%;
* an autotune-OFF timeout does not refute H1 while `T_on` has no upper bound.

The counterexamples that make those unsound are pinned here as tests, so the
branches cannot quietly come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import h1_discriminator as h1  # noqa: E402


# --------------------------------------------------------------------------- #
# the unsound branches, as concrete counterexamples                            #
# --------------------------------------------------------------------------- #
def test_a_small_lower_bound_does_not_refute_h1():
    """T_off=500 gives a 16.9% bound, but the true share could be 90%."""
    result = h1.verdict(500.0, completed=True, t_off_basis=h1.REQUIRED_BASIS)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["decisive"] is False

    bound = result["removable_lower_bound"]
    assert bound < h1.BAND_PARTIAL
    true_share_if_t_on_were_5000 = 1.0 - 500.0 / 5000.0
    assert true_share_if_t_on_were_5000 > h1.BAND_SUPPORTED, (
        "the counterexample: the same measurement is consistent with H1 being "
        "strongly SUPPORTED, so a small bound cannot mean NOT_SUPPORTED"
    )


def test_an_autotune_off_timeout_does_not_refute_h1():
    """T_off > 600 s says nothing about the ratio while T_on is unbounded above."""
    result = h1.verdict(None, completed=False, t_off_basis=h1.REQUIRED_BASIS)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["decisive"] is False
    assert "NOT a refutation" in result["reason"]


def test_no_negative_verdict_is_reachable_without_an_exact_t_on():
    """Sweep the whole space of cheap-arm outcomes; none may come back negative."""
    outcomes = [h1.verdict(t, completed=True, t_off_basis=h1.REQUIRED_BASIS) for t in
                (1.0, 60.0, 180.0, 181.0, 420.0, 421.0, 500.0, 599.0, 5000.0)]
    outcomes.append(h1.verdict(None, completed=False, t_off_basis=h1.REQUIRED_BASIS))
    assert all(o["verdict"] != "NOT_SUPPORTED" for o in outcomes)
    assert all(o["verdict"] in {"SUPPORTED", "PARTIAL_SUPPORT", "INCONCLUSIVE"}
               for o in outcomes)


def test_the_retracted_branches_are_named_in_the_output():
    """A reader of the artifact must see that this case used to be scored wrongly."""
    assert "wrongly reported this case as NOT_SUPPORTED" in \
        h1.verdict(500.0, completed=True, t_off_basis=h1.REQUIRED_BASIS)["reason"]
    assert "wrongly called it one" in h1.verdict(None, completed=False, t_off_basis=h1.REQUIRED_BASIS)["reason"]


# --------------------------------------------------------------------------- #
# the sound direction still works                                              #
# --------------------------------------------------------------------------- #
def test_the_lower_bound_holds_for_every_admissible_true_value():
    t_off = 150.0
    reported = h1.removable_lower_bound(t_off, h1.T_ON_LOWER_BOUND_S)
    for t_on in (601.4, 700.0, 1200.0, 5000.0, 1e6):
        assert reported <= 1.0 - t_off / t_on + 1e-12


def test_the_lower_bound_is_tight_at_the_endpoint():
    assert h1.removable_lower_bound(150.0, 601.4) == pytest.approx(1.0 - 150.0 / 601.4)


@pytest.mark.parametrize("t_off,expected", [
    (60.0, "SUPPORTED"),
    (180.0, "SUPPORTED"),
    (181.0, "PARTIAL_SUPPORT"),
    (420.0, "PARTIAL_SUPPORT"),
    (421.0, "INCONCLUSIVE"),
])
def test_the_positive_bands_are_applied_mechanically(t_off, expected):
    assert h1.verdict(t_off, completed=True, t_off_basis=h1.REQUIRED_BASIS)["verdict"] == expected


def test_the_band_edges_are_derived_not_typed_in():
    assert h1.T_OFF_SUPPORTED_MAX_S == pytest.approx(0.30 * 601.4)
    assert h1.T_OFF_PARTIAL_MAX_S == pytest.approx(0.70 * 601.4)


# --------------------------------------------------------------------------- #
# what an exact T_on unlocks                                                   #
# --------------------------------------------------------------------------- #
def test_an_exact_t_on_makes_a_negative_verdict_reachable():
    """Both arms completed: 590/700 leaves 15.7% removable, a real negative."""
    result = h1.verdict(590.0, completed=True, t_on_s=700.0, t_off_basis=h1.REQUIRED_BASIS)
    assert result["verdict"] == "NOT_SUPPORTED"
    assert result["decisive"] is True
    assert result["removable_exact"] == pytest.approx(1.0 - 590.0 / 700.0)
    assert "real negative rather than a silent bound" in result["reason"]


def test_a_timeout_plus_an_exact_t_on_can_refute():
    """A timed-out arm bounds T_off BELOW, which with an exact T_on bounds the share ABOVE."""
    t_on = h1.T_OFF_BUDGET_S + 100.0
    result = h1.verdict(None, completed=False, t_on_s=t_on, t_off_basis=h1.REQUIRED_BASIS)
    assert result["verdict"] == "NOT_SUPPORTED"
    assert result["decisive"] is True
    assert result["removable_upper_bound"] == pytest.approx(
        1.0 - h1.T_OFF_BUDGET_S / t_on)


def test_a_timeout_with_a_large_t_on_is_still_inconclusive():
    """A big enough T_on leaves the upper bound above the bar, concluding nothing."""
    result = h1.verdict(None, completed=False, t_on_s=5000.0, t_off_basis=h1.REQUIRED_BASIS)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["decisive"] is False


def test_an_exact_t_on_can_also_confirm():
    result = h1.verdict(100.0, completed=True, t_on_s=1000.0, t_off_basis=h1.REQUIRED_BASIS)
    assert result["verdict"] == "SUPPORTED"
    assert result["removable_exact"] == pytest.approx(0.9)


def test_the_upper_bound_needs_the_opposite_inputs_to_the_lower_bound():
    """Guard the direction of each bound; swapping them is the original error."""
    assert h1.removable_lower_bound(100.0, 1000.0) == pytest.approx(0.9)
    assert h1.removable_upper_bound(100.0, 1000.0) == pytest.approx(0.9)
    # decreasing in T_off, increasing in T_on
    assert h1.removable_lower_bound(200.0, 1000.0) < h1.removable_lower_bound(100.0, 1000.0)
    assert h1.removable_lower_bound(100.0, 2000.0) > h1.removable_lower_bound(100.0, 1000.0)


# --------------------------------------------------------------------------- #
# the limitation is stated, not hidden                                         #
# --------------------------------------------------------------------------- #
def test_the_asymmetry_is_declared_in_every_result():
    for result in (h1.verdict(60.0, completed=True, t_off_basis=h1.REQUIRED_BASIS), h1.verdict(None, completed=False, t_off_basis=h1.REQUIRED_BASIS)):
        assert "only be supported, never refuted" in result["asymmetry"]


def test_inconclusive_results_say_what_would_decide_them():
    for result in (h1.verdict(500.0, completed=True, t_off_basis=h1.REQUIRED_BASIS), h1.verdict(None, completed=False, t_off_basis=h1.REQUIRED_BASIS)):
        assert "exact T_on" in result["what_would_decide_it"]


def test_the_not_supported_band_advertises_its_requirement():
    assert "exact T_on" in h1.verdict(60.0, completed=True, t_off_basis=h1.REQUIRED_BASIS)["bands"]["NOT_SUPPORTED"]


def test_a_completed_arm_must_report_a_time():
    with pytest.raises(ValueError):
        h1.verdict(None, completed=True, t_off_basis=h1.REQUIRED_BASIS)


# --------------------------------------------------------------------------- #
# the clock gate (manager review, main 8d27c1b3)                               #
# --------------------------------------------------------------------------- #
def test_a_missing_basis_blocks():
    assert h1.verdict(100.0, completed=True)["verdict"] == "BLOCKED"


@pytest.mark.parametrize("basis", [
    "nvtx-span", "first-range-to-last-range", "subprocess-wall-time", "", None,
])
def test_any_other_basis_blocks(basis):
    result = h1.verdict(100.0, completed=True, t_off_basis=basis)
    assert result["verdict"] == "BLOCKED"
    assert result["required_basis"] == h1.REQUIRED_BASIS
    assert result["t_on_basis"] == "process-launch-to-kill"


def test_the_block_explains_which_direction_the_error_would_have_gone():
    """Dropping startup shrinks T_off, which INFLATES the removable share."""
    reason = h1.verdict(100.0, completed=True, t_off_basis="nvtx-span")["reason"]
    assert "inflates the apparent removable share" in reason


def test_the_two_clocks_are_named_in_the_module():
    assert h1.REQUIRED_BASIS == "process-launch-to-executable-readiness"
    assert h1.UPPER_BOUND_BASIS == "nsys-session-origin-to-last-compile-range-end"
    assert h1.T_ON_BASIS == "process-launch-to-kill"


def test_the_gate_is_checked_before_anything_else():
    """A wrong basis blocks even for inputs that would otherwise raise."""
    assert h1.verdict(None, completed=True, t_off_basis="nvtx-span")["verdict"] == "BLOCKED"


def test_nsys_session_clock_is_accepted_only_as_an_upper_bound():
    result = h1.verdict(
        140.0,
        completed=True,
        t_off_basis=h1.UPPER_BOUND_BASIS,
        measurement_kind=h1.MEASUREMENT_UPPER_BOUND,
    )
    assert result["verdict"] == "SUPPORTED"
    assert result["measurement_kind"] == "upper_bound"
    assert result["t_off_upper_bound_seconds"] == 140.0
    assert result["product_compile_gate_eligible"] is False
    assert result["exact_executable_readiness_identified"] is False


def test_upper_bound_clock_cannot_be_mislabeled_exact():
    result = h1.verdict(
        140.0, completed=True, t_off_basis=h1.UPPER_BOUND_BASIS
    )
    assert result["verdict"] == "BLOCKED"
    assert result["required_basis"] == h1.REQUIRED_BASIS


def test_unknown_measurement_kind_blocks():
    result = h1.verdict(
        140.0, completed=True, t_off_basis=h1.UPPER_BOUND_BASIS,
        measurement_kind="approximately_exact",
    )
    assert result["verdict"] == "BLOCKED"
    assert result["product_compile_gate_eligible"] is False
