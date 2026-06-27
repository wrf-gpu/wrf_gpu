from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

from gpuwrf.validation.two_way_feedback_gate import run_gate


def test_v022_two_way_feedback_cpu_gate(tmp_path):
    payload = run_gate(output=tmp_path / "gate.json")

    assert payload["verdict"] == "PASS"
    assert payload["finite"] == {"parent": True, "child": True}
    assert payload["conservation"]["conserved"] is True
    assert payload["conservation"]["rel_residual"] <= 1.0e-12
    assert payload["scheduler"]["event_counts"]["feedback"] == 2
    assert payload["scheduler"]["own_steps"] == {"d01": 2, "d02": 6}
    assert payload["parent_reflects_child"]["outside_overlap_max_abs_change"] <= 1.0e-12
    assert payload["parent_reflects_child"]["overlap_max_abs_error_vs_copyfcn_then_sm121"] <= 1.0e-12
    assert payload["parent_reflects_child"]["boundary_ring_mean_abs_delta_from_initial_parent"] > 1.0
