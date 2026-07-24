"""Integrity gates for the exact-carry Thompson sedimentation proof."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v0234_late_ni_thompson_sedimentation_rca.py"
PROOF = (
    ROOT
    / ".agent/sprints/2026-07-21-v0234-gpt-late-ni-rootcause"
    / "THOMPSON_SEDIMENTATION_ROOT_CAUSE.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("late_ni_thompson_rca", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_literal_ice_balance_caps_diameter_not_mass() -> None:
    module = _module()
    qi = np.asarray([4.881511619437367e-6])
    ni = np.asarray([0.0])
    rho = np.asarray([0.582343])
    balanced = module.wrf_pre_sed_ice_number_balance(qi, ni, rho)
    assert balanced[0] > 0.0
    assert np.isclose(
        module.ice_diameter_um(qi[0], balanced[0], rho[0]), 300.0, atol=1.0e-10
    )
    assert qi[0] == 4.881511619437367e-6


def test_source_authority_proves_balance_order_and_nstep_discrepancy() -> None:
    authority = _module().source_authority()
    assert all(authority["assertions"].values())
    assert authority["accepted_src_gpuwrf_tree"] == (
        "e627605f6a8bc0dc23f5c474be4bb532b99297c1"
    )


def test_sealed_proof_is_canonical_and_causal() -> None:
    module = _module()
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    assert proof["canonical_sha256"] == module.canonical_sha256(proof)
    assert proof["verdict"] == (
        "LATE_NI_ROOT_CAUSE_THOMPSON_PRESED_ICE_BALANCE_OMITTED"
    )
    assert proof["gpu_actions"] == 0
    assert proof["model_edit_performed"] is False
    assert proof["public_thompson_reproduction"]["relative_error"] < 1.0e-12
    assert proof["causal_chain"]["raw_wrf_nstep"] == 911
    assert proof["causal_chain"]["port_static_nstep_cap"] == 16
    assert proof["causal_chain"]["balanced_raw_wrf_nstep"] == 1
    assert proof["first_catastrophic_operation"][
        "direct_ice_scan_equals_full_sedimentation_qi_bytes"
    ]
    assert proof["first_catastrophic_operation"][
        "direct_ice_scan_equals_full_sedimentation_Ni_bytes"
    ]
    assert proof["first_catastrophic_operation"]["direct_ice_scan_capped_nstep"] == 16
    assert proof["first_catastrophic_operation"]["qi_after_max"] > 1.0e9
    assert proof["first_catastrophic_operation"]["mass_creation_factor"] > 1.0e13
    assert abs(
        proof["source_balanced_counterfactual"]["ice_column_mass_error_kg_m2"]
    ) < 1.0e-15
    assert proof["handoff"]["fix_status"] == "UNIMPLEMENTED_IN_THIS_SPRINT"
