"""Proof objects must agree with the raw log about what actually touched the GPU.

Manager evidence correction, before M0 handoff. `fast_cold_compile_gate.json` and
`W1_ATTEMPT_REPORT.md` both said `nsys` and `ncu` "did not run". The raw
`w1b.log` shows Stage 2 `nsys` DID start after Stage 1's `EARLY_STOP`, attached,
and reached forecast initialisation before a separate kill. "No usable nsys
capture" was true; "nsys did not run" was false.

That is the most dangerous kind of error in this sprint's artifacts: a claim that
UNDERSTATES a GPU touch. The gates exist so neighbouring agents can trust what
this worker says it did with a shared device.

These tests are the durable version of that correction. They hold the committed
proof object against the raw log, so the two cannot drift again — and they are
deliberately phrased so that a *future* window with the fixed driver (where nsys
genuinely never starts) also passes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "proofs/v025/m0/fast_cold_compile_gate.json"
LOG = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/logs/w1b.log")
REPORT = REPO / ".agent/sprints/2026-07-27-v0250-m0-setup/W1_ATTEMPT_REPORT.md"

pytestmark = pytest.mark.skipif(not GATE.exists(), reason="no cold-compile gate object")


def _log() -> str:
    return LOG.read_text(errors="replace") if LOG.exists() else ""


def _gate() -> dict:
    return json.loads(GATE.read_text())


# --------------------------------------------------------------------------- #
# the correction itself                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not LOG.exists(), reason="raw w1b log not on disk")
def test_if_the_log_shows_a_stage_starting_the_proof_object_admits_it():
    """The load-bearing check: a started GPU stage must not be reported as not run."""
    log = _log()
    gate = _gate()
    nsys_started_in_log = "--- stage 2: nsys ---" in log
    claimed = gate["window"].get("stage_2_nsys_STARTED_unintentionally")
    if nsys_started_in_log:
        assert claimed is True, (
            "the raw log shows Stage 2 nsys started, but the proof object does not say so"
        )
    else:
        assert claimed in (False, None)


def test_no_usable_capture_is_stated_separately_from_did_not_run():
    """These are different claims and must not be collapsed into one."""
    gate = _gate()["window"]
    assert gate["nsys_captured"] is False
    correction = gate.get("evidence_correction", {})
    assert "did not run' is FALSE" in correction.get("what_was_wrong", "") or \
           "FALSE" in correction.get("what_was_wrong", "")


def test_unusability_is_scoped_and_verified_not_blanket():
    """'Unusable' must name WHAT it is unusable for, and rest on a measurement.

    This assertion previously demanded `usable is False` — a blanket claim. The
    capture is unusable for the §9 CUDA kernel census and perfectly usable for
    NVTX compile analysis (107,412 ranges, 50 modules). A blanket "unusable"
    would have discarded compile evidence already paid for with a coordinated
    window, then justified asking two managers for a new one to re-obtain data
    already on disk.
    """
    artifact = _gate()["window"]["evidence_correction"]["artifact_left_behind"]
    assert artifact["unusable_for"], "the object does not say what it is unusable FOR"
    assert "kernel census" in artifact["unusable_for"]
    assert artifact["usable_for"], "the object does not say what it IS usable for"
    assert "nvtx" in artifact["usable_for"].lower()
    assert "does not contain CUDA kernel data" in artifact["usable_verified_by_measurement"]


def test_the_nvtx_evidence_object_refuses_to_equate_the_two_compile_numbers():
    """346.360 s and 4.382 s must not be conflated, and the object must say so."""
    path = REPO / "proofs/v025/m0/nvtx_compile_evidence.json"
    if not path.exists():
        pytest.skip("nvtx evidence not built")
    node = json.loads(path.read_text())["the_346s_vs_4.382s_question"]
    assert node["MUST_NOT_BE_EQUATED"] is True
    assert node["status"] == "UNRESOLVED"
    # v1 of this object claimed the warm-cache explanation was FALSIFIED, using the
    # DEPRECATED cache resolver. The retraction must stay on record, not vanish.
    assert "WITHDRAWN" in node["retracted_falsification"]


def test_autotune_presence_is_not_reported_as_time_dominance():
    """A retracted claim must stay visible, not be silently deleted.

    v1 asserted an autotune-VOLUME signal supporting H1 while also saying the
    bearing on H1 was NONE — a contradiction inside one object. v2 separates
    presence (evidenced) from time dominance (unmeasured). Detailed coverage
    lives in test_nvtx_evidence.py; this is the cross-file guard.
    """
    path = REPO / "proofs/v025/m0/nvtx_compile_evidence.json"
    if not path.exists():
        pytest.skip("nvtx evidence not built")
    node = json.loads(path.read_text())["bearing_on_pre_registered_H1"]
    assert node["autotune_time_dominance"] == "NOT MEASURED"
    assert "presence is not dominance" in node["verdict"]


def test_the_receipt_is_recorded_as_spent_not_refunded():
    """Stopping early does not refund coordination."""
    status = _gate()["window"]["receipt_status"]
    assert status["spent"] is True
    assert "ABORTED RATHER THAN SPENT" in status["correction"]


CORRECTION_MARKERS = (
    "false", "wrong", "replaced", "superseded", "corrected", "conflated", "previous version",
)


def test_the_superseded_framing_is_only_ever_quoted_while_being_corrected():
    """The phrase may appear — but only inside a correction, never as a claim.

    A blunt "this string must not appear" fails the moment the report quotes the
    wrong claim in order to retract it, which is exactly what an honest
    correction does. What must hold is that every occurrence sits next to a
    marker retracting it.
    """
    text = REPORT.read_text()
    lowered = text.lower()
    needle = "aborted rather than spent"
    start = 0
    occurrences = 0
    while (index := lowered.find(needle, start)) != -1:
        occurrences += 1
        context = lowered[max(0, index - 400): index + 400]
        assert any(marker in context for marker in CORRECTION_MARKERS), (
            f"occurrence at char {index} asserts the superseded framing without retracting it"
        )
        start = index + len(needle)
    assert occurrences, "the correction should quote the framing it retracts"
    # Whitespace-normalised: the report wraps, so "remains **SPENT**" can span a
    # newline. Searching raw text for a phrase that crosses a line break has now
    # produced a false failure three times in this suite.
    flat = re.sub(r"\s+", " ", text)
    assert "remains **SPENT**" in flat or "remains SPENT" in flat


# --------------------------------------------------------------------------- #
# the attribution that was in hand and missed                                  #
# --------------------------------------------------------------------------- #
def test_the_module_level_compile_attribution_is_recorded():
    node = _gate().get("compile_attribution_found_in_the_stage_1_log")
    assert node, "the 346 s module attribution is not in the proof object"
    assert node["module"] == "jit__run_forecast_operational_jit"
    assert 340.0 < node["compile_seconds"] < 350.0
    assert 0.5 < node["share_of_the_600s_bar"] < 0.65


@pytest.mark.skipif(not LOG.exists(), reason="raw w1b log not on disk")
def test_the_attribution_is_actually_in_the_raw_evidence():
    """Do not let a number into a proof object unless the raw log carries it."""
    arm = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/w1b_gpu_arm.json")
    if not arm.is_file():
        pytest.skip("arm json not on disk")
    detail = json.loads(arm.read_text())["arm"]["payload"].get("detail", "")
    assert "jit__run_forecast_operational_jit" in detail
    match = re.search(r"took (\d+)m([\d.]+)s", detail)
    assert match, "the slow_operation_alarm duration is not in the raw log"
    seconds = int(match.group(1)) * 60 + float(match.group(2))
    recorded = _gate()["compile_attribution_found_in_the_stage_1_log"]["compile_seconds"]
    assert abs(seconds - recorded) < 1.0, (
        f"proof object records {recorded} s but the raw log says {seconds} s"
    )


# --------------------------------------------------------------------------- #
# the failure result is untouched by the correction                            #
# --------------------------------------------------------------------------- #
def test_the_gate_verdict_is_unchanged_by_the_evidence_correction():
    """Correcting what ran must not soften what failed."""
    result = _gate()["result"]
    assert result["verdict"] == "FAIL"
    assert result["killed_at_seconds"] == 601.4
    assert result["is_lower_bound"] is True
    assert "bar is not moved" in _gate()["bar_not_moved"].lower() or \
           "not moved" in _gate()["bar_not_moved"]
