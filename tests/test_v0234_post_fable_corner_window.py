from __future__ import annotations

import inspect

from scripts import v0234_final_holistic_late_window_corner_discriminator as arm_s


def _stats_for(step: int) -> dict[str, float | int]:
    return {
        "nonfinite_total": 0,
        "corner_nonfinite": 0,
        **arm_s.CPU_PREFIX_REFERENCES[step],
    }


def test_retained_cpu_prefixes_authenticate_at_all_three_checkpoints() -> None:
    proof = arm_s._authenticate_cpu_prefixes()
    assert set(proof["references"]) == {9016, 9042, 9075}
    assert proof["logs"]["first16"]["last_step"] == 9016
    assert proof["logs"]["first42"]["last_step"] == 9042
    assert proof["logs"]["first75"]["last_step"] == 9075


def test_cpu_prefix_comparison_accepts_reference_and_rejects_structure_or_growth() -> None:
    for step in (9016, 9042, 9075):
        comparison = arm_s._compare_cpu_prefix(step, _stats_for(step))
        assert comparison is not None and comparison["passed"] is True

    structural = _stats_for(9016)
    structural["nonfinite_total"] = 1
    assert arm_s._compare_cpu_prefix(9016, structural)["passed"] is False

    growth = _stats_for(9042)
    growth["corner_mu_maxabs"] += arm_s.CPU_PREFIX_MAX_ABS_DELTA["corner_mu_maxabs"] + 0.01
    rejected = arm_s._compare_cpu_prefix(9042, growth)
    assert rejected["passed"] is False
    assert "corner_mu_maxabs" in rejected["violations"]


def test_cuda_route_is_cache_dark_and_production_preemptible_lock_bound() -> None:
    assert arm_s.CUDA_CORE_ENV["JAX_PLATFORMS"] == "cuda"
    assert arm_s.CUDA_CORE_ENV["CUDA_VISIBLE_DEVICES"] == "0"
    assert arm_s.CUDA_CORE_ENV["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert arm_s.CUDA_CORE_ENV["GPUWRF_JAX_CACHE"] == "0"
    assert arm_s.CUDA_CORE_ENV["GPUWRF_JAX_CACHE_LOCK"] == "0"
    assert arm_s.LOCK_COMMIT == "8152309aff1e85e1052d44d549a5a5409e710bdd"


def test_cuda_device_query_follows_live_lock_authentication_and_scalar_metrics() -> None:
    source = inspect.getsource(arm_s.main)
    lock_at = source.index("lock_authority = ordinary.assert_lock_authority()")
    first_cuda_query_at = source.index("# This is the first CUDA availability query")
    assert lock_at < first_cuda_query_at
    assert "stats = host_scalar_stats(carry)" in source
    assert '"hot_loop_full_carry_transfers": 0 if backend == "cuda"' in source

