"""M6b D2H warmed re-capture v2 acceptance tests.

These tests assert facts about the v2 warmed re-capture summary
``<development-history-not-included-in-public-repo>/2026-05-25-m6b-d2h-warmed-recapture-v2/proof_nsys_transfers_inside_loop.json``
that this sprint produces. They do not run a GPU job themselves; they
validate the recorded proof object so the M6 close gate can read a
single boolean ``inside_loop_d2h_clean`` from this sprint's deliverables.

The v2 sprint differs from the prior warmed re-capture (which recorded
``d2h_inter_kernel=20``) because the post-reframe + RK1 + radiation-cadence
lift in ``operational_mode.py`` removed both dynamic control-flow
emitters (RK ``lax.switch`` and radiation-cadence ``lax.cond``) located
by the inside-loop-fix bisection. Per ADR-027 the constitutional
invariant is ``d2h_inter_kernel == 0``, not ``d2h_total == 0``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / "<development-history-not-included-in-public-repo>" / "2026-05-25-m6b-d2h-warmed-recapture-v2"
WARMED = SPRINT / "proof_nsys_transfers_inside_loop.json"

pytestmark = pytest.mark.skipif(not SPRINT.exists(), reason="development-history proof not included in public repo")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_warmed_capture_artifacts_present():
    """The v2 orchestrator must produce the canonical proof objects."""

    assert (SPRINT / "proof_warmed.nsys-rep").exists(), (
        "missing the v2 warmed Nsight trace; rerun the orchestrator "
        "with GPUWRF_D2H_SPRINT_DIR pointing to the v2 sprint dir"
    )
    assert (SPRINT / "proof_warmed_call_log.json").exists()
    assert (SPRINT / "proof_warmed_trace_summary.txt").exists()
    assert WARMED.exists(), (
        f"{WARMED} not found; run "
        f"'python scripts/m6b_d2h_warmed_recapture.py --parse-rep "
        f"{SPRINT}/proof_warmed.nsys-rep' with "
        f"GPUWRF_D2H_SPRINT_DIR={SPRINT}"
    )


def test_v2_warmed_orchestrator_followed_warmup_protocol():
    """The script logged at least three untimed warmups outside the profile window."""

    call_log = _load(SPRINT / "proof_warmed_call_log.json")
    assert call_log["warmed_protocol"]["warmups_outside_profile_window"] >= 3
    assert call_log["warmed_protocol"]["cuda_profiler_range_used"] is True
    assert call_log["warmed_protocol"]["warmup_and_profiled_signature_identical"] is True
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


def test_v2_warmed_capture_zero_host_to_device_transfers():
    """H2D inside the warmed window must be 0 (constitutional invariant)."""

    warmed = _load(WARMED)
    assert warmed["h2d_total"] == 0, (
        f"H2D=={warmed['h2d_total']} inside warmed profile window; "
        f"this would be a fresh constitutional violation"
    )


def test_v2_warmed_capture_inter_kernel_d2h_is_zero():
    """ADR-027 constitutional invariant: d2h_inter_kernel == 0.

    After the RK1 + radiation-cadence lift in operational_mode.py, the
    warmed Nsight capture must report zero D2H transfers interleaved
    with compute kernels.
    """

    warmed = _load(WARMED)
    assert warmed["d2h_inter_kernel"] == 0, (
        f"d2h_inter_kernel = {warmed['d2h_inter_kernel']} (>0). The "
        f"constitutional invariant per ADR-027 is violated. Inspect "
        f"{warmed['inter_kernel_d2h_clusters_by_prev_kernel']} for "
        "per-kernel localisation."
    )
    assert warmed["inter_kernel_d2h_clusters_by_prev_kernel"] == [], (
        "inter-kernel cluster list non-empty even though count is zero "
        "— parser bug"
    )


def test_v2_warmed_recapture_does_not_touch_operational_sources():
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
            "gpuwrf.dynamics.acoustic_loop",
            "gpuwrf.dynamics.dycore_step",
            "gpuwrf.dynamics.coupled_step",
        ):
            assert token not in source, (
                f"forbidden token {token!r} appeared in operational source; "
                "this sprint is read-only with respect to operational code"
            )
