from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-standalone-vs-comparator-bisect"


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_m6b_standalone_and_comparator_feed_identical_step1_inputs():
    proof = _load("proof_input_signatures.json")

    assert proof["status"] == "PASS"
    assert proof["matches"] == {
        "state": True,
        "namelist": True,
        "carry_initialization": True,
    }


def test_m6b_standalone_step1_output_matches_comparator_bitwise():
    proof = _load("proof_standalone_step1_matches.json")

    assert proof["status"] == "PASS"
    assert proof["steps"] == 1
    assert proof["final_max_abs_delta"] == 0.0
    assert proof["all_leaves_finite"] is True
