"""Binding C1 gate for the scalar candidate's released 0/0 control."""

from __future__ import annotations

from scripts import v0234_c1_candidate_off_probe as probe


def test_nested_scalar_options_zero_match_precandidate_full_carry_bytes() -> None:
    result = probe.run_probe()
    assert result["candidate_off_identity_pass"] is True
    assert result["input_leaf_count"] == result["output_leaf_count"] == 71
    assert result["options"] == {"moist_adv_opt": 0, "scalar_adv_opt": 0}
    assert result["nested_frozen_wrf_boundary_bundle"] is True
    assert result["broken_control_reproduced_as_nonidentity"] is True
    assert (
        result["verdict"]
        == "C1_REPAIRED__CANDIDATE_OFF_MATCHES_RELEASED_FULL_CARRY_BYTES"
    )
    assert len(result["canonical_payload_sha256"]) == 64
