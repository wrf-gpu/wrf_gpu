"""Focused integrity gates for the late-Ni retained-output proof."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v0234_late_ni_retained_forensics.py"
PROOF = (
    ROOT
    / ".agent/sprints/2026-07-21-v0234-gpt-late-ni-rootcause"
    / "RETAINED_PRECURSOR_PROOF.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("late_ni_forensics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_array_stats_counts_nonfinite_nonzero_and_thresholds() -> None:
    module = _module()
    stats = module.array_stats(
        np.asarray([0.0, -2.0, 3.0, np.inf, np.nan]), thresholds=(1.0, 2.0)
    )
    assert stats["finite_count"] == 3
    assert stats["nonfinite_count"] == 2
    assert stats["nonzero_count"] == 4
    assert stats["finite_min"] == -2.0
    assert stats["finite_max"] == 3.0
    assert stats["abs_count_above"] == {"1": 2, "2": 1}


def test_current_source_topology_keeps_failure_cell_outside_diff6() -> None:
    topology = _module().source_topology()
    assert all(topology["assertions"].values())
    assert "relaxation ring 1" in topology["failure_cell_y1_x1_semantics"]
    assert "sixth-order scalar diffusion is zero" in topology["failure_cell_y1_x1_semantics"]


def test_retained_proof_is_canonical_and_does_not_overclaim() -> None:
    module = _module()
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    assert proof["canonical_sha256"] == module.canonical_sha256(proof)
    assert proof["verdict"] == "LATE_NI_RETAINED_PRECURSOR_NARROWED_EXACT_CARRY_REQUIRED"
    assert proof["failure"]["guarded_ni_onset_interval_d03_steps"] == "(10206,10400]"
    assert all(proof["d03_ice_zero_checks"].values())
    for reconstruction in proof["d02_to_d03_ni_reconstruction"]["reconstructions"].values():
        assert reconstruction["child_full"]["nonfinite_count"] == 0
        assert reconstruction["boundary_strips"]["nonfinite_count"] == 0
    assert "not an exact carry replay" in proof["d02_to_d03_ni_reconstruction"]["retained_output_precision_caveat"]
    assert proof["method"]["gpu_actions"] == 0
