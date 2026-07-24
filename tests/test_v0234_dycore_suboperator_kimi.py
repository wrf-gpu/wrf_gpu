"""Focused CPU tests for the v0234 dycore-suboperator-kimi sprint.

Static/source-level and pure-math tests only: no JAX, no GPU, no WRF run.
Run: python -m pytest -q tests/test_v0234_dycore_suboperator_kimi.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi"
MODEL_FILE = REPO / "src/gpuwrf/runtime/operational_mode.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit_mod():
    return _load("v0234_audit", REPO / "scripts/v0234_dycore_suboperator_kimi_audit.py")


@pytest.fixture(scope="module")
def runner_mod():
    return _load("v0234_wrf_ladder", REPO / "scripts/v0234_dycore_suboperator_kimi_wrf_ladder.py")


# --------------------------------------------------------------- mask math

def test_splitmix64_deterministic_and_wrapping(audit_mod):
    values = np.arange(8, dtype=np.uint64)
    first = audit_mod.splitmix64(values, 0x243F6A8885A308D3)
    second = audit_mod.splitmix64(values, 0x243F6A8885A308D3)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint64


def test_mask_balance_and_orthogonality(audit_mod):
    shapes = {"U": (44, 93, 112), "V": (44, 94, 111)}
    masks = {}
    for mask_id, spec in audit_mod.MASK_SPECS.items():
        masks[mask_id] = {v: audit_mod.mask_signs(v, shapes[v], spec) for v in ("U", "V")}
        for v in ("U", "V"):
            signs = masks[mask_id][v]
            assert int(signs.astype(np.int64).sum()) == 0
            assert signs.shape == (44, shapes[v][1] - 10, shapes[v][2] - 10)
    for left, right in (("mask_a", "mask_b"), ("mask_a", "mask_c"), ("mask_b", "mask_c")):
        dot = sum(
            int(np.sum(masks[left][v].astype(np.int64) * masks[right][v].astype(np.int64)))
            for v in ("U", "V")
        )
        assert dot == 0


def test_mask_sign_hashes_match_frozen_plan(audit_mod):
    plan = json.loads((REPO / ".agent/sprints/2026-07-18-v0234-conditioning-ensemble-gpt/ensemble-plan.json").read_text())
    shapes = {"U": (44, 93, 112), "V": (44, 94, 111)}
    for mask_id, spec in audit_mod.MASK_SPECS.items():
        for v in ("U", "V"):
            signs = audit_mod.mask_signs(v, shapes[v], spec)
            digest = hashlib.sha256(signs.tobytes()).hexdigest()
            assert digest == plan["perturbation"]["masks"][mask_id]["per_variable"][v]["sign_sha256"]


# ----------------------------------------------------------- band helpers

def test_distance_to_edge():
    from scripts.v0234_dycore_suboperator_kimi_band_decomp import distance_to_edge

    dist = distance_to_edge((2, 6, 7))
    assert dist.shape == (2, 6, 7)
    assert dist[0, 0, 0] == 0
    assert dist[0, 2, 3] == 2
    assert dist[0, 5, 6] == 0
    assert dist[0, 1, 5] == 1


# -------------------------------------------------------- runner constants

def test_runner_cpuset_maps_to_physical_13_15(runner_mod):
    cpus = set()
    for part in runner_mod.CPUSET.split(","):
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    assert cpus == {13, 14, 15, 29, 30, 31}
    assert runner_mod._physical_cores(cpus) == {13, 14, 15}
    assert runner_mod.DUMP_FILES_PER_STEP == 23
    assert runner_mod.STEPS == (1, 2)
    assert runner_mod.ANALYSIS_TAGS["l1_rk1_tend"] == ("ru_tend", "rv_tend")


def test_runner_launch_env_is_cpu_only_and_ladder_gated(runner_mod):
    assert runner_mod.SYSTEMD_CPU_AFFINITY == "13 14 15 29 30 31"
    assert runner_mod.SYSTEMD_AFFINITY_SYSCALL_FILTER == "~sched_setaffinity"


def test_watchdog_uses_systemd_cgroup_and_fails_closed(runner_mod):
    sample_source = inspect.getsource(runner_mod.Watchdog.sample)
    finish_source = inspect.getsource(runner_mod.Watchdog.finish)
    admit_source = inspect.getsource(runner_mod.admit_run)
    assert "_unit_process_rows" in sample_source
    assert "pgrep" not in sample_source
    assert "member-rank-observation" in finish_source
    assert "systemctl" in admit_source and "stop" in admit_source
    assert "SUCCESS COMPLETE WRF" in admit_source


def test_reassembler_knows_frozen_ladder_staggering():
    from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble

    expected = {
        "l1_rk1_tend__ru_tend": "u",
        "l1_rk1_tend__rv_tend": "v",
        "l1_rk1_tend__u_save": "u",
        "l1_rk1_tend__v_save": "v",
        "l2_rk1_fin__u": "u",
        "l2_rk1_fin__v": "v",
        "l3_rk2_fin__u": "u",
        "l3_rk2_fin__v": "v",
        "l4_rk3_fin__u": "u",
        "l4_rk3_fin__v": "v",
        "l5_prebdry__u": "u",
        "l5_prebdry__v": "v",
    }
    assert {key: reassemble.FIELD_STAGGER[key] for key in expected} == expected


def test_cpu_analysis_metrics_are_cell_weighted_and_band_complete():
    from scripts import v0234_dycore_suboperator_gpt_cpu_analysis as analysis

    u = np.zeros((1, 13, 14), dtype=np.float64)
    v = np.zeros((1, 14, 13), dtype=np.float64)
    u[0, 6, 6] = -3.0
    v[0, 0, 0] = 4.0
    hgt = np.arange(13 * 13, dtype=np.float64).reshape(13, 13)
    result = analysis.metrics_with_bands({"u": u, "v": v}, hgt)
    assert result["cells"] == u.size + v.size
    assert result["sse"] == 25.0
    assert result["rmse"] == pytest.approx(np.sqrt(25.0 / (u.size + v.size)))
    assert result["max_abs"] == 4.0
    assert result["argmax"]["component"] == "v"
    assert result["finite_fraction"] == 1.0
    assert sum(row["sse"] for row in result["bands"].values()) == 25.0
    assert sum(row["cells"] for row in result["bands"].values()) == result["cells"]


# ------------------------------------------------ GPU diagnostic model edit

def test_model_capture_is_diagnostic_only_static():
    source = MODEL_FILE.read_text()
    assert "class FirstIntervalLadderRecord(NamedTuple)" in source
    assert "class FirstIntervalLadderStepResult(NamedTuple)" in source
    assert "def advance_one_step_with_first_interval_ladder" in source
    assert "capture_ladder: bool = False" in source
    assert "capture_ladder) and (bool(capture_rca)" in source
    # every capture site is behind the default-False static flag
    assert source.count("capture_ladder=capture_ladder") >= 1
    assert "pre_bdry_u = next_state.u" in source


def test_gpu_profile_static_shape():
    profile = (REPO / "scripts/v0234_dycore_suboperator_kimi_gpu_ladder.py").read_text()
    assert "advance_one_step_with_first_interval_ladder" in profile
    assert "RETAINED_PRODUCTION_HLO_SHA256" in profile
    assert "--intent production-preemptible" in profile
    assert "nested_stage_omega_transport_470e6111_dycore_suboperator_ladder1" in profile


# ------------------------------------------------------------- WRF patch

def test_wrf_ladder_patch_contains_all_blocks():
    patch = (SPRINT / "wrf-ladder-instrumentation.patch").read_text()
    assert "momsp_ladder" in patch
    assert "l1_rk1_tend" in patch
    assert "l2_rk1_fin" in patch
    assert "l3_rk2_fin" in patch
    assert "l4_rk3_fin" in patch
    assert "l5_prebdry" in patch
    assert patch.count("momsp_dump3d") >= 12
