"""v0.18 CU-family honesty checks."""

from __future__ import annotations

import json
from pathlib import Path
from _historical_artifacts import require_historical


def test_v018_tail_cu_schemes_are_not_accepted_without_oracles() -> None:
    from gpuwrf.contracts.physics_registry import ACCEPTED_CU_PHYSICS

    assert not ({7, 10, 11} & set(ACCEPTED_CU_PHYSICS))


def test_v018_cu_family_status_has_no_scoped_oracle_gaps() -> None:
    path = Path(__file__).resolve().parents[1] / "proofs/v018/cu_family_status.json"
    require_historical(path)
    report = json.loads(path.read_text())

    assert report["step1_honesty_gate_met"] is True
    assert report["full_v018_cu_ship_gate_met"] is True
    assert report["full_ship_gate_blockers"] == []
    assert report["tail_relevance_assessment"]["status"] == "RELEVANT_NOT_PROVEN_IRRELEVANT"
    for code in ("7", "10", "11"):
        assert report["schemes"][code]["status"] == "REFERENCE_ONLY_WITH_REAL_ORACLE"
        assert report["schemes"][code]["savepoints"]["nontrivial"] is True
    for code in ("5", "93"):
        assert report["schemes"][code]["status"] == "REFERENCE_ONLY_WITH_REAL_ORACLE_RED_JAX"
        assert report["schemes"][code]["savepoints"]["nontrivial"] is True
    assert report["accepted_matrix"]["tail_without_oracle_accepted"] == []
