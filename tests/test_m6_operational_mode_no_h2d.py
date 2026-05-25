from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_operational_source_has_no_host_transfer_or_sanitizer_calls():
    source = (ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py").read_text(encoding="utf-8")

    forbidden = ("device_get", "host_callback", "pure_callback", "io_callback", "sanitize_state", "snapshot(")
    for token in forbidden:
        assert token not in source


def test_nsight_solver_loop_has_zero_h2d_d2h_memcopies():
    path = (
        ROOT
        / ".agent"
        / "sprints"
        / "2026-05-25-m6-perf-design"
        / "artifacts"
        / "proof_solver_bakeoff_nsight_loop_mem_cuda_gpu_mem_size_sum.json"
    )
    assert path.exists()
    rows = json.loads(path.read_text(encoding="utf-8"))
    operations = {row["Operation"]: row for row in rows}

    assert "[CUDA memcpy Host-to-Device]" not in operations
    assert "[CUDA memcpy Device-to-Host]" not in operations


def test_jax_trace_solver_loop_transfer_audit_is_zero():
    path = ROOT / ".agent" / "sprints" / "2026-05-25-m6-perf-design" / "artifacts" / "proof_solver_bakeoff.json"
    proof = json.loads(path.read_text(encoding="utf-8"))

    assert proof["transfer_audit"]["host_to_device_bytes"] == 0
    assert proof["transfer_audit"]["device_to_host_bytes"] == 0
