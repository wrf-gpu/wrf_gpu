from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6-perf-design-acceptance"


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_operational_mode_does_not_import_validation_only_helpers():
    source = (ROOT / "src/gpuwrf/runtime/operational_mode.py").read_text(encoding="utf-8")
    forbidden = (
        "gpuwrf.dynamics.acoustic_loop",
        "gpuwrf.dynamics.dycore_step",
        "gpuwrf.dynamics.coupled_step",
        "from gpuwrf.dynamics.acoustic_loop",
        "from gpuwrf.dynamics.dycore_step",
        "from gpuwrf.dynamics.coupled_step",
        "sanitize_state",
    )
    for token in forbidden:
        assert token not in source


def test_cpu_wrf_baseline_uses_28_ranks_and_reserved_cores():
    proof = _load("proof_cpu_wrf_baseline.json")
    assert proof["status"] == "PASS"
    assert proof["cpu_cores"] == "4-31"
    assert proof["mpi_ranks"] == 28
    assert proof["wall_time_s"] > 0


def test_operational_run_tier4_speedup_and_transfer_gates_pass():
    op = _load("proof_operational_run.json")
    rmse = _load("proof_tier4_rmse.json")
    speedup = _load("proof_speedup.json")
    transfers = _load("proof_nsys_transfers_inside_loop.json")

    assert op["status"] == "PASS"
    assert rmse["status"] == "PASS"
    assert speedup["status"] == "PASS"
    assert speedup["speedup"] >= 1.2
    assert transfers["cudaMemcpyHtoD_inside_loop"] == 0
    assert transfers["cudaMemcpyDtoH_inside_loop"] == 0


def test_solver_bakeoff_v2_has_cusparse_reference():
    proof = _load("proof_solver_bakeoff_v2.json")
    assert proof["status"] == "PASS"
    cusparse = proof["references"]["cusparse_gtsv2_strided_batch"]
    assert cusparse["status"] == "measured"
    assert cusparse["wall_ms_per_call_median"] > 0
    assert cusparse["residual_l2_relative"] < 1.0e-10


def test_adr_026_promoted_to_proposed():
    proposed = ROOT / ".agent/decisions/ADR-026-operational-mode-design-PROPOSED.md"
    draft = ROOT / ".agent/decisions/ADR-026-operational-mode-design-DRAFT.md"
    assert proposed.exists()
    assert not draft.exists()
    text = proposed.read_text(encoding="utf-8")
    assert "Status: **PROPOSED**" in text
    assert "proof_tier4_rmse.json" in text
    assert "proof_solver_bakeoff_v2.json" in text
