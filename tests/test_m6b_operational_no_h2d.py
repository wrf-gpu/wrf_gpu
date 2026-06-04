from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-honest-1h-canary"


def test_operational_mode_source_still_has_no_host_callbacks_or_sanitizer():
    source = (ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py").read_text(encoding="utf-8")
    # d47fe0f moved M9 diagnostics into the output-cadence-only `_m9_snapshot`
    # helper. Keep rejecting any other old snapshot path.
    source_without_m9_snapshot = source.replace("_m9_snapshot(", "")
    forbidden = (
        "device_get",
        "host_callback",
        "pure_callback",
        "io_callback",
        "sanitize_state",
        "gpuwrf.dynamics.acoustic_loop",
        "gpuwrf.dynamics.dycore_step",
        "gpuwrf.dynamics.coupled_step",
    )
    for token in forbidden:
        assert token not in source
    assert "snapshot(" not in source_without_m9_snapshot


def test_m6b_nsys_transfer_audit_records_profiled_transfer_gate_status():
    proof = json.loads((SPRINT / "proof_nsys_transfers_inside_loop.json").read_text(encoding="utf-8"))

    if proof["status"] == "PASS":
        assert proof["cudaMemcpyHtoD_inside_profiled_loop"] == 0
        assert proof["cudaMemcpyDtoH_inside_profiled_loop"] == 0
    else:
        assert proof["status"] == "FAIL"
        assert (
            proof["cudaMemcpyHtoD_inside_profiled_loop"] > 0
            or proof["cudaMemcpyDtoH_inside_profiled_loop"] > 0
        )
    assert proof["full_1h_verified"] is False
    assert "blocked" in proof["scope_note"]
