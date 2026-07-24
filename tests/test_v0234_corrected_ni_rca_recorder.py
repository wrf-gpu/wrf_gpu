from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.runtime.operational_mode import (
    RCA_ACOUSTIC_FIELDS,
    RCA_BOUNDARY_FIELDS,
    RCA_HEALTH_METRICS,
    RCA_STATE_FIELDS,
    RCA_STATE_PHASES,
    RCA_TARGET_K,
    RCA_TARGET_X,
    RCA_TARGET_Y,
    _rca_array_health,
    _rca_target_value,
)
from scripts.v0234_corrected_ni_rca_gpu import decode_records


def test_rca_schema_binds_corrected_ni_cell_and_separate_phases() -> None:
    assert (RCA_TARGET_K, RCA_TARGET_Y, RCA_TARGET_X) == (1, 48, 78)
    assert len(RCA_HEALTH_METRICS) == 8
    assert RCA_STATE_PHASES[:2] == ("step_entry", "physics_output")
    assert RCA_STATE_PHASES[-2:] == ("post_boundary", "post_precision")
    assert {"Ni", "u", "v", "p_total", "ph_total", "mu_total"} <= set(RCA_STATE_FIELDS)
    assert {"raw_dvdxi", "raw_dmdt", "raw_mu_tendency", "mu_scale"} <= set(
        RCA_ACOUSTIC_FIELDS
    )
    assert {"u_bdy", "v_bdy", "ph_bdy", "mu_bdy"} <= set(RCA_BOUNDARY_FIELDS)


def test_rca_health_reports_exact_first_bad_and_extrema_locations() -> None:
    value = jnp.asarray([[3.0, -7.0, jnp.nan], [jnp.inf, 2.0, -1.0]], dtype=jnp.float64)
    health = np.asarray(_rca_array_health(value))

    assert health[0] == 2
    assert health[1] == 2
    assert health[2] == 7.0
    assert health[3] == 1
    assert health[4] == -7.0
    assert health[5] == -7.0
    assert health[6] == 1
    assert health[7] == 3.0


def test_rca_target_uses_corrected_indices_and_clamps_only_tiny_unit_grids() -> None:
    full = jnp.arange(3 * 60 * 90, dtype=jnp.float64).reshape(3, 60, 90)
    expected = np.asarray(full)[RCA_TARGET_K, RCA_TARGET_Y, RCA_TARGET_X]
    assert float(_rca_target_value(full)) == float(expected)

    tiny = jnp.arange(2 * 2 * 2, dtype=jnp.float64).reshape(2, 2, 2)
    assert float(_rca_target_value(tiny)) == float(np.asarray(tiny)[1, 1, 1])


def test_rca_health_jaxpr_has_no_host_callback() -> None:
    text = str(jax.make_jaxpr(_rca_array_health)(jnp.ones((3, 4), dtype=jnp.float64))).lower()
    for token in ("callback", "outside_call", "host_compute", "io_callback", "pure_callback"):
        assert token not in text


def test_decoder_orders_physics_output_before_acoustic_at_retry20_step() -> None:
    state_health = np.zeros((207, len(RCA_STATE_PHASES), len(RCA_STATE_FIELDS), 8))
    state_target = np.zeros((207, len(RCA_STATE_PHASES), len(RCA_STATE_FIELDS)))
    acoustic_health = np.zeros((207, 16, len(RCA_ACOUSTIC_FIELDS), 8))
    acoustic_target = np.zeros((207, 16, len(RCA_ACOUSTIC_FIELDS)))
    boundary_health = np.zeros((207, len(RCA_BOUNDARY_FIELDS), 8))
    acoustic_health[:, :, RCA_ACOUSTIC_FIELDS.index("mu_scale"), 5] = 1.0

    step_offset = 9400 - 9199
    ni_index = RCA_STATE_FIELDS.index("Ni")
    phase_index = RCA_STATE_PHASES.index("physics_output")
    shape = (3, 60, 90)
    flat = int(np.ravel_multi_index((1, 48, 78), shape))
    state_health[step_offset, phase_index, ni_index, 0] = 1
    state_health[step_offset, phase_index, ni_index, 1] = flat
    state_target[step_offset, phase_index, ni_index] = np.nan

    state_shapes = {name: list(shape) for name in RCA_STATE_FIELDS}
    acoustic_shapes = {name: list(shape) for name in RCA_ACOUSTIC_FIELDS}
    boundary_shapes = {name: [2, 4, 5, 3, 60] for name in RCA_BOUNDARY_FIELDS}
    decoded = decode_records(
        {
            "acoustic_health": acoustic_health,
            "acoustic_target": acoustic_target,
            "state_health": state_health,
            "state_target": state_target,
            "boundary_health": boundary_health,
        },
        state_shapes=state_shapes,
        acoustic_shapes=acoustic_shapes,
        boundary_shapes=boundary_shapes,
    )

    event = decoded["first_nonfinite_event"]
    assert event["native_step"] == 9400
    assert event["phase"] == "physics_output"
    assert event["field"] == "Ni"
    assert event["health"]["first_nonfinite"]["python_index"] == [1, 48, 78]


def test_decoder_accepts_bounded_event_local_window() -> None:
    steps = 3
    state_health = np.zeros(
        (steps, len(RCA_STATE_PHASES), len(RCA_STATE_FIELDS), 8)
    )
    state_target = np.zeros(
        (steps, len(RCA_STATE_PHASES), len(RCA_STATE_FIELDS))
    )
    acoustic_health = np.zeros(
        (steps, 16, len(RCA_ACOUSTIC_FIELDS), 8)
    )
    acoustic_target = np.zeros(
        (steps, 16, len(RCA_ACOUSTIC_FIELDS))
    )
    boundary_health = np.zeros(
        (steps, len(RCA_BOUNDARY_FIELDS), 8)
    )
    acoustic_health[:, :, RCA_ACOUSTIC_FIELDS.index("mu_scale"), 5] = 1.0
    ni_index = RCA_STATE_FIELDS.index("Ni")
    phase_index = RCA_STATE_PHASES.index("post_rk_pre_non_dry")
    state_health[1, phase_index, ni_index, 0] = 1
    state_health[1, phase_index, ni_index, 1] = 0
    state_target[1, phase_index, ni_index] = np.nan
    shape = (3, 4, 5)
    decoded = decode_records(
        {
            "acoustic_health": acoustic_health,
            "acoustic_target": acoustic_target,
            "state_health": state_health,
            "state_target": state_target,
            "boundary_health": boundary_health,
        },
        state_shapes={name: list(shape) for name in RCA_STATE_FIELDS},
        acoustic_shapes={name: list(shape) for name in RCA_ACOUSTIC_FIELDS},
        boundary_shapes={
            name: [2, 4, 5, 3, 6] for name in RCA_BOUNDARY_FIELDS
        },
        domain="d03",
        dt_s=6.0,
        prefix_step=10398,
        replay_step=10401,
    )
    event = decoded["first_nonfinite_event"]
    assert event["native_step"] == 10400
    assert event["sim_time_s"] == 62400.0
    assert event["phase"] == "post_rk_pre_non_dry"


def test_gpu_attempt_proof_fails_closed_without_promoting_debug_event() -> None:
    proof_path = Path(
        ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max/"
        "recorder-proof.json"
    )
    proof = json.loads(proof_path.read_text())
    gate = proof["causal_gate"]

    assert gate["canonical_failure_geometry_reproduced"] is True
    assert gate["canonical_terminal_Ni"] == {
        "field": "Ni",
        "first_python_index": [1, 48, 78],
        "first_wrf_index": [2, 49, 79],
        "nonfinite_count": 233,
    }
    assert gate["canonical_vs_unit_chunk_off_identity"] is True
    assert gate["canonical_vs_recorder_on_identity"] is False
    assert gate["zero_transfer_trace"] is False
    assert gate["recorder_admissible"] is False
    assert gate["first_event_usable_for_causality"] is None
    assert gate["inadmissible_first_event_for_debug_only"]["field"] == "post_w"
    assert proof["recorder_on"]["comparison_to_canonical"]["different_leaf_count"] == 57
    assert proof["recorder_on"]["transfer_audit"]["host_to_device_bytes"] > 0
    assert proof["recorder_on"]["transfer_audit"]["device_to_host_bytes"] > 0
    assert proof["candidate_fix"] is None
    assert proof["full_18h_run_performed"] is False
