from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State
from gpuwrf.runtime.operational_state import initial_operational_carry
from scripts import v0234_corrected_ni_ordinary_bisection as bisection


SPRINT_DIR = Path(
    ".agent/sprints/2026-07-13-v0234-corrected-ni-ordinary-bisection"
)


def _tiny_carry(*, ni: float = 0.0):
    values = {}
    for name, parameter in inspect.signature(State).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            values[name] = jnp.asarray(
                [ni if name == "Ni" else 0.0], dtype=jnp.float64,
            )
    return initial_operational_carry(State(**values))


def test_runner_imports_only_the_ordinary_production_step() -> None:
    path = Path("scripts/v0234_corrected_ni_ordinary_bisection.py")
    module = ast.parse(path.read_text())
    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        and node.module == "gpuwrf.runtime.operational_mode"
        for alias in node.names
    }

    assert imported == {"_advance_chunk_fori", "build_clock_base"}
    assert "advance_chunk_with_corrected_ni_rca" not in imported
    audit = bisection.audit_source_separation()
    assert audit["ordinary_dispatch_precedes_post_health_materialization"] is True
    assert audit["health_result_not_passed_to_dispatch"] is True
    assert audit["recorder_tokens_found"] == []
    assert audit["source_separated"] is True


def test_health_covers_every_carry_leaf_and_names_corrected_ni() -> None:
    carry = _tiny_carry(ni=float("nan"))
    labels = bisection.carry_leaf_labels(carry)
    counts = np.asarray(bisection.carry_nonfinite_counts(carry))
    inventory = bisection.named_nonfinite_inventory(carry)

    assert len(labels) == len(counts)
    assert labels.count("state.Ni") == 1
    assert int(np.sum(counts)) == 1
    assert inventory == [
        {
            "leaf_index": labels.index("state.Ni"),
            "name": "state.Ni",
            "shape": [1],
            "dtype": "float64",
            "nonfinite_count": 1,
            "first_python_index": [0],
            "first_wrf_index": [1],
        }
    ]


def test_checkpoint_primitive_roundtrips_all_leaf_bytes(tmp_path: Path) -> None:
    value = {
        "float": np.asarray([1.0, np.nan], dtype=np.float64),
        "integer": np.asarray([3, 4], dtype=np.int32),
    }
    before = bisection.host_tree_manifest(value)
    path = tmp_path / "carry.pkl"
    bisection.atomic_write_pickle(path, value)
    with path.open("rb") as handle:
        reread = pickle.load(handle)
    after = bisection.host_tree_manifest(reread)

    assert before["floating_nonfinite_count"] == 1
    assert bisection.compare_manifests(before, after)["all_leaf_bytes_equal"] is True
    assert bisection.sha256_file(path)


def test_frozen_terminal_identity_and_bounded_separate_track_evidence() -> None:
    prior = json.loads(bisection.PRIOR_RECORDER_PROOF.read_text())
    v10 = json.loads(bisection.PRIOR_V10_PROOF.read_text())
    last = v10["last_frame"]["v10"]

    assert prior["recorder_off"]["carry_manifest"]["manifest_sha256"] == (
        "2dcfc195baaf3e59ba539f6701fdb82b9a134a0461cfc68dec3a429e38433565"
    )
    assert v10["bounded_gates"]["common_ni_v10_root_proved"] is False
    assert last["rmse"] == 2.1128268857679338
    assert last["land_rmse"] == 2.2264724301076377
    assert last["sea_rmse"] == 2.0839931788025168
    assert last["max_abs"] == 11.358115434646606
    assert last["max_location"] == {
        "landmask": 0,
        "lat": 28.216018676757812,
        "lon": -16.90625,
        "x": 19,
        "y": 39,
    }
    assert last["ni_cell_delta"] == -0.7763886451721191
    contract = (SPRINT_DIR / "sprint-contract.md").read_text()
    assert "separate causal tracks" in contract
    assert bisection.PRIOR_V10_PROOF_SHA256 == (
        "d269513748c85006a7a95b87f778e4c442041c812b2d76dee85ff20309599e68"
    )
    assert bisection.sha256_file(bisection.PRIOR_V10_PROOF) == (
        bisection.PRIOR_V10_PROOF_SHA256
    )


def test_preimport_authority_requires_exact_production_lane() -> None:
    environment = dict(bisection.REQUIRED_PREIMPORT_ENV)
    environment.update({
        "GPUWRF_JAX_CACHE_DIR": "/cache",
        "JAX_COMPILATION_CACHE_DIR": "/cache",
        "GPUWRF_WRF_ROOT": "/authority",
    })
    proof = bisection.validate_preimport_environment(environment)
    assert proof["validated_before_jax_import"] is True

    environment["GPUWRF_BITWISE"] = "0"
    try:
        bisection.validate_preimport_environment(environment)
    except RuntimeError as error:
        assert "frozen ordinary lane" in str(error)
    else:
        raise AssertionError("non-production environment must fail closed")


def test_gpu_proof_localizes_ordinary_step_and_keeps_v10_separate() -> None:
    proof_path = SPRINT_DIR / "ordinary-bisection-proof.json"
    proof = json.loads(proof_path.read_text())
    payload_digest = proof.pop("proof_sha256")

    assert bisection.sha256_file(proof_path) == (
        "237e6e59e3f080a28434b72e8e149eb85f2d032139c7f8d4524caee04ff5087b"
    )
    assert payload_digest == bisection.canonical_digest(proof) == (
        "9e206a1c3b02f3e252a0c06c7884a592e07fdf7fb402c690dbde12cd2568ef5a"
    )
    assert proof["verdict"] == "ORDINARY_BISECTION_LOCALIZED"
    assert proof["localization_gate"] is True
    replay = proof["ordinary_replay"]
    assert replay["exact_step_sampling"] is True
    assert replay["sampled_steps"] == replay["expected_sampled_steps"] == 207
    assert replay["unique_transition"] is True
    assert replay["first_bad"] == {
        "introduced_by_completed_ordinary_dispatch": True,
        "native_step": 9314,
        "parent_force_changed_health": False,
        "post_changes_from_prefix": [
            {
                "baseline_nonfinite": 0, "leaf_index": 57,
                "name": "t_2ave", "nonfinite_count": 4840,
            },
            {
                "baseline_nonfinite": 0, "leaf_index": 58,
                "name": "ww", "nonfinite_count": 4730,
            },
            {
                "baseline_nonfinite": 0, "leaf_index": 59,
                "name": "mudf", "nonfinite_count": 110,
            },
            {
                "baseline_nonfinite": 0, "leaf_index": 60,
                "name": "muave", "nonfinite_count": 110,
            },
            {
                "baseline_nonfinite": 0, "leaf_index": 61,
                "name": "muts", "nonfinite_count": 110,
            },
        ],
        "post_nonfinite_count": 9900,
        "pre_nonfinite_count": 0,
        "sim_time_s": 55884,
    }
    rows = {row["native_step"]: row for row in replay["health_rows"]}
    assert rows[9313]["post_nonfinite_count"] == 0
    assert rows[9314]["post_nonfinite_count"] == 9900
    assert next(
        row for row in rows[9315]["post_changes_from_prefix"]
        if row["name"] == "state.Ni"
    )["nonfinite_count"] == 119

    terminal = proof["terminal_identity"]
    assert terminal["all_leaf_bytes_equal_to_frozen_ordinary"] is True
    assert terminal["actual_manifest_sha256"] == terminal["required_manifest_sha256"] == (
        "2dcfc195baaf3e59ba539f6701fdb82b9a134a0461cfc68dec3a429e38433565"
    )
    retained = proof["retained_carries"]
    assert retained["reread_identity"] is True
    assert retained["last_finite_and_first_bad_health_proved"] is True
    assert retained["last_finite"]["manifest"]["floating_nonfinite_count"] == 0
    assert retained["first_bad"]["manifest"]["floating_nonfinite_count"] == 9900
    assert proof["ordinary_program"]["ordinary_only"] is True
    assert proof["health_program"]["separate_from_production_hlo"] is True
    assert proof["health_program"]["callback_free"] is True
    assert proof["scope_gates"] == {
        "full_18h_run_performed": False,
        "model_or_numerical_edit": False,
        "phase_tap_executed": False,
        "recorder_on_used": False,
    }
    assert proof["mechanism_separation"]["common_root_proved"] is False
    assert proof["mechanism_separation"]["v10_gate"] == (
        "OPEN_SEPARATE_TRACK_NO_FIX_AUTHORITY"
    )


def test_checkpoint_mask_analysis_stops_before_operator_claim() -> None:
    path = SPRINT_DIR / "checkpoint-mask-analysis.json"
    analysis = json.loads(path.read_text())
    assert bisection.sha256_file(path) == (
        "3d18986579bce6d91f54b596874fed9b72dda973e557741a41cfcf97a66b09e6"
    )
    transition = analysis["transition"]
    assert transition["native_step"] == 9314
    assert transition["pre_nonfinite_count"] == 0
    assert transition["post_nonfinite_count"] == 9900
    assert transition["state_nonfinite_count_at_first_bad_output"] == 0
    assert transition["state_Ni_nonfinite_count_after_next_dispatch_9315"] == 119
    assert analysis["shared_spatial_mask"]["count"] == 110
    assert analysis["shared_spatial_mask"]["low_y_high_x"][0] == {
        "y": 1, "x_first": 100, "x_last": 109,
    }
    assert analysis["shared_spatial_mask"]["high_y_low_x"][-1] == {
        "y": 91, "x_first": 1, "x_last": 10,
    }
    bounds = analysis["causal_bounds"]
    assert "no individual boundary" in bounds["not_proved"]
    assert "no temporal and operator-causal link" in bounds["V10_role"]
    design = (SPRINT_DIR / "phase-tap-design.md").read_text()
    assert "not executed" in design.lower()
    assert "cannot preserve byte identity by construction" in " ".join(design.split())
