"""M6b D2H warmed re-capture acceptance tests.

These tests assert facts about ``proof_nsys_transfers_inside_loop.json``,
the parsed Nsight trace summary the warmed re-capture orchestrator
produces (``scripts/m6b_d2h_warmed_recapture.py``). They do not run a
GPU job themselves; they validate the recorded proof object so the
M6b RETRY codex can read a single boolean ``inside_loop_d2h_clean``
from this sprint's deliverables.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-d2h-warmed-recapture"
WARMED = SPRINT / "proof_nsys_transfers_inside_loop.json"
UNWARMED = SPRINT / "proof_nsys_transfers_m6b_unwarmed_baseline.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_warmed_capture_artifacts_present():
    """The orchestrator must produce all four canonical proof objects."""

    assert (SPRINT / "proof_warmed.nsys-rep").exists(), (
        "missing the warmed Nsight trace; rerun "
        "scripts/m6b_d2h_warmed_recapture.py under nsys"
    )
    assert (SPRINT / "proof_warmed_call_log.json").exists()
    assert (SPRINT / "proof_warmed_trace_summary.txt").exists()
    assert WARMED.exists(), (
        f"{WARMED} not found; run "
        f"'python scripts/m6b_d2h_warmed_recapture.py --parse-rep "
        f".agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_warmed.nsys-rep'"
    )


def test_warmed_orchestrator_followed_warmup_protocol():
    """The script logged at least one untimed warmup outside the profile window."""

    call_log = _load(SPRINT / "proof_warmed_call_log.json")
    assert call_log["warmed_protocol"]["warmups_outside_profile_window"] >= 1
    assert call_log["warmed_protocol"]["cuda_profiler_range_used"] is True
    assert call_log["warmed_protocol"]["warmup_and_profiled_signature_identical"] is True
    # Confirm the profiled call really was a JIT cache hit (sub-second wall
    # time) and not silently a re-compile.
    profiled_s = float(call_log["wall_time_s"]["profiled_call"])
    warmup_s = float(call_log["wall_time_s"]["warmup_call_includes_compile"])
    assert profiled_s < 1.0, (
        f"profiled call took {profiled_s:.3f}s — that is not a JIT cache "
        "hit; the warmup protocol failed to populate the cache for the "
        "exact (state, namelist, hours) signature."
    )
    assert warmup_s > profiled_s * 10, (
        "first warmup did not dominate the profiled call wall-time; the "
        "compilation may not have happened in the warmup window."
    )


def test_warmed_capture_zero_host_to_device_transfers():
    """No H2D inside the warmed window — verified separately from D2H.

    The D2H grep verdict (sprint 2026-05-25-m6b-d2h-grep) showed H2D = 0
    in the unwarmed capture as well. With the warmup protocol applied the
    invariant must continue to hold.
    """

    warmed = _load(WARMED)
    assert warmed["h2d_total"] == 0, (
        f"H2D=={warmed['h2d_total']} inside warmed profile window; "
        f"this would be a fresh constitutional violation"
    )


def test_warmed_capture_pre_kernel_d2h_dropped_vs_unwarmed_baseline():
    """The warmup protocol must reduce XLA per-call argument-staging D2Hs.

    The D2H grep verdict predicted that pre-kernel D2Hs (XLA constant
    staging captured because cudaProfilerStart fired before the warmup
    call) would vanish under proper warmup discipline. This test
    enforces that the warmup at least *halves* the pre-kernel count
    compared to the M6b unwarmed baseline.
    """

    warmed = _load(WARMED)
    unwarmed = _load(UNWARMED)
    assert unwarmed["d2h_pre_kernel"] >= warmed["d2h_pre_kernel"], (
        "warmed pre-kernel D2H not lower than unwarmed baseline — the "
        "warmup protocol failed; revisit the orchestrator"
    )
    # The grep memo says ~50 pre-kernel transfers come from first-graph
    # constant staging. After the warmup these should be largely gone;
    # require at least a 2x reduction so we have a non-degenerate signal.
    assert warmed["d2h_pre_kernel"] * 2 <= unwarmed["d2h_pre_kernel"], (
        f"warmed pre-kernel D2H = {warmed['d2h_pre_kernel']}, unwarmed = "
        f"{unwarmed['d2h_pre_kernel']}; warmup did not cut at least 50% "
        "of XLA first-call staging — protocol regression"
    )


def test_warmed_capture_inter_kernel_d2h_summary_recorded():
    """Inter-kernel D2H = the genuine inside-timestep-loop transfer count.

    Per the sprint contract: if this number is non-zero we route to a
    fix sprint (it documents real residual D2Hs inside the compiled
    scan body). The contract does not require this number to be zero
    today; it only requires the warmed re-capture to *honestly record*
    it for the M6b RETRY codex.
    """

    warmed = _load(WARMED)
    assert "d2h_inter_kernel" in warmed
    assert "inter_kernel_d2h_clusters_by_prev_kernel" in warmed
    assert isinstance(warmed["d2h_inter_kernel"], int)
    # If inter-kernel is non-zero we must have at least one cluster
    # attributed to a previous kernel, so the next fix sprint has a
    # localisation target.
    if warmed["d2h_inter_kernel"] > 0:
        clusters = warmed["inter_kernel_d2h_clusters_by_prev_kernel"]
        assert clusters, (
            "inter-kernel D2H>0 but no per-kernel cluster localisation "
            "recorded; the parser must attribute each transfer to its "
            "preceding kernel for the fix sprint"
        )
        attributed = sum(c["count"] for c in clusters)
        assert attributed == warmed["d2h_inter_kernel"], (
            f"cluster total {attributed} != inter-kernel D2H "
            f"{warmed['d2h_inter_kernel']}; parser bug"
        )


def test_warmed_recapture_does_not_touch_operational_sources():
    """Constitutional non-goal: this sprint may not edit operational sources."""

    operational = (ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py").read_text(encoding="utf-8")
    state = (ROOT / "src" / "gpuwrf" / "runtime" / "operational_state.py").read_text(encoding="utf-8")
    for source in (operational, state):
        for token in (
            "device_get",
            "host_callback",
            "pure_callback",
            "io_callback",
            "sanitize_state",
            "snapshot(",
            "gpuwrf.dynamics.acoustic_loop",
            "gpuwrf.dynamics.dycore_step",
            "gpuwrf.dynamics.coupled_step",
        ):
            assert token not in source, (
                f"forbidden token {token!r} appeared in operational source; "
                "this sprint is read-only with respect to operational code"
            )
