from __future__ import annotations

import numpy as np

from scripts import v0234_1500_scientific_rca_trajectory as rca


def test_material_rule_requires_two_consecutive_frames() -> None:
    assert rca._first_sustained([0.0, 2.0, 0.5, 2.0, 2.1], 1.0, 2) == 3
    assert rca._first_sustained([0.0, 2.0, 0.5], 1.0, 2) is None


def test_compact_spatial_partitions_are_exhaustive_for_all_staggers() -> None:
    land = np.zeros((93, 111), dtype=np.float64)
    for shape in ((93, 111), (93, 112), (94, 111)):
        result = rca.compact_decomposition(np.ones((2, *shape)), land)
        assert sum(item["n"] for item in result["distance"].values()) == 2 * shape[0] * shape[1]
        assert sum(item["n"] for item in result["topology"].values()) == 2 * shape[0] * shape[1]
        assert sum(item["n"] for item in result["surface"].values()) == 2 * shape[0] * shape[1]
        assert 0.0 < result["direct_0_3_energy_fraction"] < 1.0


def test_discovered_frame_set_and_pair_proofs_are_exact() -> None:
    rows = rca.discover_frames()
    assert len(rows) == 46
    assert rows[0]["own_step"] == 0
    assert rows[-1]["own_step"] == 9000
    assert rows[-1]["pair_proof_sha256"] == "f5786ec0705f752a271b74190ca901df4b5b2cd2641198a5941032f3cebe5002"


def test_canonical_hash_excludes_only_its_own_field() -> None:
    payload = {"schema": "x", "value": [1, 2, 3]}
    digest = rca.canonical_hash(payload)
    payload["proof_sha256"] = digest
    assert rca.canonical_hash(payload) == digest
