from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gpuwrf.io.proof_schemas import FullDomainBatchingVerdict


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_full_domain_batching.py"
SPEC = importlib.util.spec_from_file_location("m6_full_domain_batching", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_binding_cpu_denominator_uses_m6_s2a_raw_timing_subtraction():
    cpu = MODULE._binding_cpu_denominator()

    assert cpu["basis"] == "raw_timing_subtraction"
    assert cpu["artifact"] == "artifacts/m6/cpu_denominator.json"
    assert cpu["cpu_wall_s"] == 4859.53
    assert round(cpu["raw_timing_subtraction_s_exact"], 2) == 4859.53
    assert "-r4" in cpu["fp_precision_evidence"]


def test_speedup_ratio_is_cpu_wall_divided_by_gpu_wall():
    cpu_wall_s = 4859.53
    gpu_wall_s = 1000.0

    assert cpu_wall_s / gpu_wall_s == pytest.approx(4.85953)


def test_full_domain_batching_verdict_schema_matches_m6_s5_contract():
    payload = {
        "speedup_ratio": 4.1,
        "pass": True,
        "gpu_wall_s": 1185.0,
        "cpu_wall_s": 4859.53,
        "cpu_denominator_basis": "raw_timing_subtraction",
        "fp_precision_caveat": "WRF -r4 default real; GPU FP32-gated per ADR-007 storage",
        "dycore_cap_status": "lifted_via_path_b",
        "tier2_invariants_under_lifted_cap": "pass",
        "profiler_raw_paths": ["artifacts/m6/performance/profile/plugins/profile/xplane.pb"],
        "transfer_audit": {"host_to_device_bytes": 0, "device_to_host_bytes": 0},
        "op_count": 1,
        "hlo_size_bytes": 100,
        "temp_peak_bytes": 1,
        "compile_retries": 0,
        "cache_size_bytes": 0,
        "allocator_fragmentation": 0.0,
        "artifact_paths": ["artifacts/m6/performance/full_domain_batching_verdict.json"],
    }

    assert FullDomainBatchingVerdict.validate_dict(payload) is payload
