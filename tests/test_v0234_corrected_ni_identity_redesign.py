from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v0234_corrected_ni_identity_redesign.py"


def _module():
    spec = importlib.util.spec_from_file_location("v0234_corrected_ni_identity_redesign", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_redesign_has_no_jax_or_gpuwrf_import() -> None:
    tree = ast.parse(SCRIPT.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "jax" or name.startswith("jax.") for name in imports)
    assert not any(name == "gpuwrf" or name.startswith("gpuwrf.") for name in imports)


def test_independent_algebraic_oracles() -> None:
    module = _module()
    boundary = module.boundary_response_oracle()
    assert boundary["pristine_wrf_frozen_tendency_response"] == 0.1
    assert np.isclose(
        boundary["released_moving_residual_strength20"]["response"],
        0.8926258176,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.isclose(
        boundary["candidate_moving_residual_strength1"]["response"],
        0.09561792499119559,
        rtol=0.0,
        atol=1.0e-15,
    )

    continuity = module.continuity_oracles()
    nullspace = continuity["many_to_one_nullspace"]
    assert nullspace["same_weighted_divergence"]
    assert nullspace["different_vertical_profile"]
    ieee = continuity["ieee_nonfinite_scale_zero"]
    assert ieee["inf_times_zero_is_nan"]
    assert ieee["nan_times_zero_is_nan"]


def test_authenticated_redesign_reproduces_bounded_no_go() -> None:
    module = _module()
    proof = module.build_proof()

    assert proof["status"] == "COMPLETE"
    assert proof["verdict"] == "NO_FIX_LOCALIZED"
    assert proof["scope"] == {
        "gpu_calls": 0,
        "model_calls": 0,
        "jax_imported": False,
        "tap_science": "DISCARDED_UNREAD",
        "model_or_numerical_edits": 0,
        "full_18h_run": False,
        "agents_launched": 0,
    }
    unsigned = dict(proof)
    digest = unsigned.pop("proof_sha256")
    assert module.canonical_digest(unsigned) == digest

    authority = proof["authority"]
    assert authority["launch"] == {
        "commit": module.LAUNCH_COMMIT,
        "parent": module.LAUNCH_PARENT,
        "tree": module.LAUNCH_TREE,
        "commit_object_present": True,
    }
    assert authority["terminal_manifest"]["actual"] == module.TERMINAL_MANIFEST
    assert authority["compiler"]["ordinary"]["stablehlo_sha256"] == (
        "24035858ab6c6f555d670a09491ac2737367741916f9edc51a00e5ec427dbfe3"
    )

    phase = proof["observer_failure_explanation"]["phase_tap"]
    assert phase["tap_science"] == "DISCARDED_UNREAD"
    assert phase["different_leaf_count"] == 22
    assert phase["all_differing_leaf_nonfinite_counts_unchanged"]
    names = {row["name"] for row in phase["named_differing_leaves"]}
    assert {"state.u", "state.Ni", "t_2ave", "ww", "mudf", "muave", "muts", "ph_tend", "ww_save"} <= names

    localization = proof["ordinary_localization"]
    assert localization["two_cone_geometry"]["union_cells"] == 110
    assert localization["two_cone_geometry"]["exactly_matches_all_five_projected_nonfinite_masks"]
    assert localization["rk3_stage_entry_saved_by_completed_step_9314"]["u_save"]["max_abs_finite"] > 5.0e8
    assert localization["state_finite_is_not_health"]["p_total_max_abs"] > 1.0e300

    no_go = proof["identifiability_no_go"]
    assert not no_go["unique_operator_identified"]
    assert not no_go["general_fix_authorized"]
    assert proof["shortest_decisive_experiment"]["B"]["only_static_difference"] == (
        "GPUWRF_NORMAL_BDY_RELAX_STRENGTH=1.0 before import"
    )
    assert not proof["mechanism_separation"]["common_ni_v10_root_proved"]
