from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-real-ic-bisection"


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_real_ic_bisection_localizes_first_stage():
    proof = _load("proof_first_diverging_stage.json")

    assert proof["status"] == "LOCALIZED"
    assert proof["first_diverging_stage"] == "rk1_acoustic_substep_1"
    assert proof["rk_stage"] == 1
    assert proof["acoustic_substep"] == 1
    assert proof["rk1_fix_invoked"] is True
    assert proof["max_abs_delta"] > proof["threshold"]


def test_real_ic_operator_proof_names_committed_advance_mu_t_defect():
    proof = _load("proof_first_diverging_operator.json")

    assert proof["status"] == "LOCALIZED"
    assert proof["largest_delta_operator"] == "advance_mu_t_committed_outputs"
    assert "discards its prognostic mu/theta/mudf outputs" in proof["named_defect"]
    assert proof["defect_location"]["file"] == "src/gpuwrf/runtime/operational_mode.py"
    assert proof["defect_location"]["mu_new_line"] is not None
    assert "module_small_step_em.F:1102-1108" in proof["wrf_source_citation"]["mu_commit"]
    assert "module_small_step_em.F:1141-1171" in proof["wrf_source_citation"]["theta_commit"]


def test_rk1_invocation_marker_and_debug_gate_are_present():
    invocation = (SPRINT / "proof_rk1_fix_invocation.txt").read_text(encoding="utf-8")
    source = (ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py").read_text(encoding="utf-8")

    assert "status: PASS" in invocation
    assert "GPUWRF_M6B_RK1_ACOUSTIC_LOOP_ENTER substeps=1" in invocation
    assert "run_forecast_operational_debug" in source
    assert "debug=False" in source
    assert "jax.debug.print" in source
