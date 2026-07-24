from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import v0234_nested_frozen_wrf_boundary_window as runner


def exact_environment(*, audit: bool = False) -> dict[str, str]:
    env = dict(runner.REQUIRED_PREIMPORT_ENV)
    env.update({
        "GPUWRF_WRF_ROOT": str(runner.RETRY20_WRF_ROOT),
        "GPUWRF_NESTED_BUNDLE_APPROVED_SHA": runner.CANDIDATE_COMMIT,
        "GPUWRF_FINAL_NI_OWNER_OVERRIDE": str(runner.OWNER_OVERRIDE),
    })
    if audit:
        env.update({
            "GPUWRF_NESTED_BUNDLE_RUNNER_SHA": "runner-sha",
            "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT": "/tmp/audit.json",
            "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT_SHA256": "a" * 64,
        })
    return env


def healthy_summary() -> dict[str, object]:
    return {
        "nonfinite_by_leaf": [0] * 106,
        "scale_maxabs": [10.0] * len(runner.ALL_SCALE_FIELDS),
        "positive_min": [100.0, 1000.0],
        "wind_maxabs": [20.0, 25.0, 5.0],
        "corner_mu_pert_maxabs": 500.0,
        "theta_max": 400.0,
        "ni_nonfinite_count": 0,
    }


class FakeHealthLeaf:
    def __init__(self, shape: tuple[int, ...], dtype: str = "float64") -> None:
        self.shape = shape
        self.dtype = dtype


class FakeHealthCarry:
    def __init__(self, boundary_time_records: int, *, leaf_count: int = 106) -> None:
        self.leaves = [FakeHealthLeaf((2, 2)) for _ in range(leaf_count)]
        for index in runner.BOUNDARY_TARGET_TRANSITION_LEAF_INDICES:
            if index < leaf_count:
                self.leaves[index] = FakeHealthLeaf((boundary_time_records, 4, 5, 44, 112))


def fake_health_runtime() -> SimpleNamespace:
    tree_util = SimpleNamespace(
        tree_leaves=lambda carry: carry.leaves,
        tree_structure=lambda _carry: "Fake106LeafCarry",
    )
    return SimpleNamespace(jax=SimpleNamespace(tree_util=tree_util))


def test_module_import_is_stdlib_only_in_fresh_interpreter() -> None:
    code = (
        "import sys; "
        "import scripts.v0234_nested_frozen_wrf_boundary_window; "
        "assert 'jax' not in sys.modules; "
        "assert not any(n.startswith('gpuwrf') for n in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=runner.REPO_ROOT,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_exact_environment_passes_without_runner_audit() -> None:
    result = runner.validate_preimport_environment(
        exact_environment(), require_runner_audit=False,
    )
    assert result["validated_before_jax_import"] is True
    assert result["unexpected_gpuwrf"] == []
    assert result["persistent_cache"]["enabled"] is False


@pytest.mark.parametrize("name", runner.FORBIDDEN_PERSISTENT_CACHE_ENV)
def test_each_persistent_cache_path_or_override_fails(name: str) -> None:
    env = exact_environment()
    env[name] = "/tmp/untrusted-cache"
    with pytest.raises(runner.RunnerGateError, match="ENV_PERSISTENT_CACHE_FORBIDDEN"):
        runner.validate_preimport_environment(env, require_runner_audit=False)


def test_runtime_cache_must_be_disabled() -> None:
    runtime = SimpleNamespace(
        compile_cache_runtime=SimpleNamespace(CACHE_STATUS={
            "source": "disabled-by-GPUWRF_JAX_CACHE",
            "enabled": False,
            "dir": None,
        }),
        jax=SimpleNamespace(config=SimpleNamespace(jax_enable_compilation_cache=False)),
    )
    assert runner.assert_runtime_cache_disabled(runtime)["persistent"] is False
    runtime.jax.config.jax_enable_compilation_cache = True
    with pytest.raises(runner.RunnerGateError, match="RUNTIME_PERSISTENT_CACHE"):
        runner.assert_runtime_cache_disabled(runtime)


def test_final_candidate_and_owner_override_authenticate() -> None:
    authority = runner.assert_final_candidate_proof_authority()
    assert authority["owner_override_verdict"] == "OWNER_OVERRIDE_FULL18H_AUTHORIZED"
    assert authority["critic_cancelled_by_owner"] is True
    assert authority["terminal_candidate"]["canonical_payload_sha256"] == (
        runner.TERMINAL_CANDIDATE_PAYLOAD_SHA256
    )
    assert authority["cpu_candidate"]["canonical_payload_sha256"] == (
        runner.CPU_CANDIDATE_PAYLOAD_SHA256
    )
    assert authority["gpu_policy"]["canonical_payload_sha256"] == (
        runner.GPU_POLICY_PAYLOAD_SHA256
    )
    assert authority["corrected_cpu_arm"]["canonical_payload_sha256"] == (
        runner.CORRECTED_CPU_ARM_PAYLOAD_SHA256
    )
    assert authority["falsified_cpu_arm"]["canonical_payload_sha256"] == (
        runner.FALSIFIED_CPU_ARM_PAYLOAD_SHA256
    )
    assert authority["nested_advection_amendment"]["canonical_payload_sha256"] == (
        runner.NESTED_ADV_AMENDMENT_PAYLOAD_SHA256
    )
    assert authority["nested_advection_offline_proof"]["canonical_payload_sha256"] == (
        runner.NESTED_ADV_PROOF_PAYLOAD_SHA256
    )
    assert authority["nested_advection_partial_wind_commit"] == (
        runner.PARTIAL_WIND_CANDIDATE_COMMIT
    )
    assert authority["theta_amendment"]["canonical_payload_sha256"] == (
        runner.THETA_AMENDMENT_PAYLOAD_SHA256
    )
    assert authority["theta_source_proof"]["canonical_payload_sha256"] == (
        runner.THETA_SOURCE_PROOF_PAYLOAD_SHA256
    )
    assert authority["theta_candidate_commit"] == runner.THETA_CANDIDATE_COMMIT
    assert authority["diffopt_amendment"]["sha256"] == runner.DIFFOPT_AMENDMENT_SHA256
    assert authority["diffopt_cpu_ab"]["canonical_payload_sha256"] == (
        runner.DIFFOPT_CPU_AB_PAYLOAD_SHA256
    )
    assert authority["diffopt_candidate"]["canonical_payload_sha256"] == (
        runner.DIFFOPT_CANDIDATE_PAYLOAD_SHA256
    )
    assert authority["diffopt_parent_candidate_commit"] == runner.DIFFOPT_PARENT_COMMIT
    assert authority["scalar_amendment"]["canonical_payload_sha256"] == (
        runner.SCALAR_AMENDMENT_PAYLOAD_SHA256
    )
    assert authority["scalar_oracle"]["canonical_payload_sha256"] == (
        runner.SCALAR_ORACLE_PAYLOAD_SHA256
    )
    assert authority["scalar_cpu_ab"]["canonical_payload_sha256"] == (
        runner.SCALAR_CPU_AB_PAYLOAD_SHA256
    )
    assert authority["scalar_candidate"]["canonical_payload_sha256"] == (
        runner.SCALAR_CANDIDATE_PAYLOAD_SHA256
    )
    assert authority["scalar_candidate_commit"] == runner.SCALAR_CANDIDATE_COMMIT
    assert authority["t_source_amendment"]["canonical_payload_sha256"] == (
        runner.T_SOURCE_AMENDMENT_PAYLOAD_SHA256
    )
    assert authority["t_source_cpu_ab"]["canonical_payload_sha256"] == (
        runner.T_SOURCE_CPU_AB_PAYLOAD_SHA256
    )
    assert authority["t_source_candidate"]["canonical_payload_sha256"] == (
        runner.T_SOURCE_CANDIDATE_PAYLOAD_SHA256
    )
    assert authority["t_source_candidate_commit"] == runner.CANDIDATE_COMMIT
    assert authority["earliest_causal_gate"] == "d03 step 200 / 00:20"
    assert authority["model_lineage_green"] is True
    assert authority["component_bindings_green"] is True
    assert authority["full_18h_authorized"] is True
    assert authority["additional_review_or_manager_authority_required"] is False
    assert "jax" not in sys.modules
    assert not any(name.startswith("gpuwrf") for name in sys.modules)


def test_early_causal_ring1_gate_requires_u_and_t_improvement(monkeypatch) -> None:
    import numpy as np

    shape = (3, 7, 8)
    cpu = {name: np.zeros(shape) for name in ("T", "U")}
    retry = {name: np.zeros(shape) for name in ("T", "U")}
    prior = {name: np.full(shape, 4.0) for name in ("T", "U")}
    partial = {name: np.full(shape, 2.0) for name in ("T", "U")}
    scalar = {name: np.full(shape, 3.0) for name in ("T", "U")}
    improved = {name: np.full(shape, 1.0) for name in ("T", "U")}
    worsened_t = {"T": np.full(shape, 3.0), "U": np.full(shape, 1.0)}
    retained_wind_gain = {"T": np.full(shape, 1.0), "U": np.full(shape, 3.0)}
    lost_wind_gain = {"T": np.full(shape, 1.0), "U": np.full(shape, 5.0)}
    monkeypatch.setattr(runner, "EARLY_CAUSAL_RING1_BASELINE", {
        name: {
            "prior_vs_cpu": 4.0,
            "prior_vs_retry20": 4.0,
            "partial_vs_cpu": 2.0,
            "partial_vs_retry20": 2.0,
            "scalar_vs_cpu": 3.0,
            "scalar_vs_retry20": 3.0,
        }
        for name in ("T", "U")
    })

    green = runner.early_causal_ring1_metrics(
        np, current=improved, prior=prior, partial=partial, scalar=scalar,
        retry20=retry, cpu=cpu,
    )
    assert green["passed"] is True
    assert all(row["passed"] for row in green["fields"].values())

    red = runner.early_causal_ring1_metrics(
        np, current=worsened_t, prior=prior, partial=partial, scalar=scalar,
        retry20=retry, cpu=cpu,
    )
    assert red["passed"] is False
    assert red["fields"]["T"]["passed"] is False
    assert red["fields"]["U"]["passed"] is True

    retained = runner.early_causal_ring1_metrics(
        np,
        current=retained_wind_gain,
        prior=prior,
        partial=partial,
        scalar=scalar,
        retry20=retry,
        cpu=cpu,
    )
    assert retained["passed"] is True
    assert retained["fields"]["U"]["passed"] is True

    wind_red = runner.early_causal_ring1_metrics(
        np,
        current=lost_wind_gain,
        prior=prior,
        partial=partial,
        scalar=scalar,
        retry20=retry,
        cpu=cpu,
    )
    assert wind_red["passed"] is False
    assert wind_red["fields"]["T"]["passed"] is True
    assert wind_red["fields"]["U"]["passed"] is False
    assert "remain better than immutable 4484" in wind_red["u_policy"]


def test_real_early_causal_baselines_match_sha_bound_recompute() -> None:
    import numpy as np
    from netCDF4 import Dataset

    authorities = (
        (runner.EARLY_CAUSAL_PRIOR, runner.EARLY_CAUSAL_PRIOR_SHA256),
        (runner.EARLY_CAUSAL_PARTIAL_WIND, runner.EARLY_CAUSAL_PARTIAL_WIND_SHA256),
        (runner.EARLY_CAUSAL_SCALAR_PARENT, runner.EARLY_CAUSAL_SCALAR_PARENT_SHA256),
        (runner.EARLY_CAUSAL_RETRY20, runner.EARLY_CAUSAL_RETRY20_SHA256),
        (runner.EARLY_CAUSAL_CPU, runner.EARLY_CAUSAL_CPU_SHA256),
    )
    for path, expected in authorities:
        assert runner.sha256_file(path) == expected

    with (
        Dataset(runner.EARLY_CAUSAL_PRIOR) as prior_ds,
        Dataset(runner.EARLY_CAUSAL_PARTIAL_WIND) as partial_ds,
        Dataset(runner.EARLY_CAUSAL_SCALAR_PARENT) as scalar_ds,
        Dataset(runner.EARLY_CAUSAL_RETRY20) as retry_ds,
        Dataset(runner.EARLY_CAUSAL_CPU) as cpu_ds,
    ):
        arrays = {
            role: {
                field: dataset.variables[field][0]
                for field in ("T", "U")
            }
            for role, dataset in (
                ("prior", prior_ds),
                ("partial", partial_ds),
                ("scalar", scalar_ds),
                ("retry20", retry_ds),
                ("cpu", cpu_ds),
            )
        }
        result = runner.early_causal_ring1_metrics(
            np,
            current=arrays["scalar"],
            prior=arrays["prior"],
            partial=arrays["partial"],
            scalar=arrays["scalar"],
            retry20=arrays["retry20"],
            cpu=arrays["cpu"],
        )

    assert all(
        row["gates"]["baseline_authenticated"]
        for row in result["fields"].values()
    )
    recomputed = result["fields"]["U"]["scalar_vs_retry20"]
    assert recomputed == 0.19252577923290928
    assert recomputed == runner.EARLY_CAUSAL_RING1_BASELINE["U"][
        "scalar_vs_retry20"
    ]


def test_early_causal_cpu_authority_is_byte_bound_not_path_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"same authenticated CPU frame bytes"
    alias = tmp_path / "manifest-selected-cpu.nc"
    alias.write_bytes(payload)
    monkeypatch.setattr(
        runner, "EARLY_CAUSAL_CPU_SHA256", hashlib.sha256(payload).hexdigest(),
    )

    row = runner.authenticate_early_causal_cpu_path(alias)
    assert row["path"] == str(alias.resolve())
    assert row["sha256"] == hashlib.sha256(payload).hexdigest()

    alias.write_bytes(payload + b" changed")
    with pytest.raises(runner.RunnerGateError, match="EARLY_CAUSAL_SELECTED_CPU"):
        runner.authenticate_early_causal_cpu_path(alias)


def test_pytest_output_normalization_removes_only_elapsed_noise() -> None:
    left = runner.normalize_pytest_output("62 passed in 1.42s\n")
    right = runner.normalize_pytest_output("62 passed in 9.01s\n")
    assert left == right == "62 passed in <elapsed>s\n"


@pytest.mark.parametrize("name", sorted(runner.REQUIRED_PREIMPORT_ENV))
def test_each_required_environment_mismatch_fails(name: str) -> None:
    env = exact_environment()
    env[name] = "wrong"
    with pytest.raises(runner.RunnerGateError, match="ENV_FROZEN_LANE"):
        runner.validate_preimport_environment(env, require_runner_audit=False)


@pytest.mark.parametrize(
    "name",
    [
        "GPUWRF_NORMAL_BDY_RELAX_STRENGTH",
        "GPUWRF_CORRECTED_NI_RCA",
        "GPUWRF_PHASE_TAP_SUMMARY",
        "GPUWRF_SANITIZER",
        "GPUWRF_TOLERANCE",
        "GPUWRF_ACOUSTIC_PRECISION_MODE",
    ],
)
def test_each_forbidden_override_fails(name: str) -> None:
    env = exact_environment()
    env[name] = "1"
    with pytest.raises(runner.RunnerGateError, match="ENV_UNAPPROVED_GPUWRF"):
        runner.validate_preimport_environment(env, require_runner_audit=False)


@pytest.mark.parametrize("name", sorted(runner.FORBIDDEN_NON_GPUWRF_ENV))
def test_each_forbidden_non_gpuwrf_override_fails(name: str) -> None:
    env = exact_environment()
    env[name] = "1"
    with pytest.raises(runner.RunnerGateError, match="ENV_FORBIDDEN_OVERRIDE"):
        runner.validate_preimport_environment(env, require_runner_audit=False)


def test_runner_audit_environment_is_required_for_real_lane() -> None:
    with pytest.raises(runner.RunnerGateError, match="ENV_RUNNER_AUDIT"):
        runner.validate_preimport_environment(exact_environment(), require_runner_audit=True)
    assert runner.validate_preimport_environment(
        exact_environment(audit=True), require_runner_audit=True,
    )["runner_audit_required"] is True


def test_failed_preflight_cannot_reach_import_hook() -> None:
    called = []

    def preflight() -> None:
        raise runner.RunnerGateError("SENTINEL", "expected")

    def importer() -> None:
        called.append(True)

    with pytest.raises(runner.RunnerGateError, match="SENTINEL"):
        runner.execute_after_preflight(preflight, importer)
    assert called == []


def test_schedule_clock_oracle_is_exact() -> None:
    proof = runner.schedule_clock_oracle()
    assert proof["passed"] is True
    assert proof["prefix_build_count"] == 1
    assert proof["prefix_own_steps"] == runner.PREFIX_OWN_STEPS
    assert proof["window_own_steps"] == runner.WINDOW_OWN_STEPS
    assert proof["d03_dispatch_steps"] == list(range(9199, 9406))
    assert proof["window_output_counts"] == {"d01": 16, "d02": 16, "d03": 48}
    assert proof["terminal_output_counts"] == {"d01": 19, "d02": 19, "d03": 55}
    assert proof["first_progress_d03_steps"] == [0, 200]
    assert proof["decisive_d03_step"] == 9000
    assert proof["second_prefix_or_rebuild"] is False


def test_parent_join_schedule_is_exact_mid_subcycle_and_reaches_terminal() -> None:
    join = runner.schedule_clock_oracle()["parent_join"]
    assert join["segments"] == [67] * 14 + [40]
    assert join["parent_only_own_steps"] == {"d01": 978, "d02": 2934}
    assert join["retained_d03_step"] == 8800
    assert join["d03_subcycle_position_at_join"] == 1
    assert join["catchup_d03_steps"] == [8801, 8802]
    assert join["aligned_own_steps"] == {"d01": 978, "d02": 2934, "d03": 8802}
    assert join["stage_9000_own_steps"] == {"d01": 1000, "d02": 3000, "d03": 9000}
    assert join["stage_9405_own_steps"] == runner.WINDOW_OWN_STEPS
    assert join["terminal_own_steps"] == runner.TERMINAL_OWN_STEPS
    assert join["retained_parent_output_counts"] == {"d01": 15, "d02": 15}


def test_lazy_parent_program_compiles_once_loops_one_step_and_rejects_shape_drift(
    tmp_path: Path,
) -> None:
    calls: list[int] = []
    builds: list[tuple[str, int]] = []
    runtime = fake_health_runtime()
    runtime.jnp = SimpleNamespace(asarray=lambda value, dtype=None: int(value), int32="int32")

    def executable(carry: object, _namelist: object, step: int, _clock: object, **kwargs: object) -> object:
        assert kwargs == {"n_steps": 1, "cadence": 7}
        calls.append(step)
        return carry

    def builder(
        _tree: object,
        domain: str,
        _carry: object,
        _runtime: object,
        *,
        artifact_path: Path,
        start_step: int,
    ) -> tuple[object, object, object, int, dict[str, object]]:
        builds.append((domain, start_step))
        assert artifact_path == tmp_path / "d02.json"
        return executable, object(), object(), 7, {
            "lower_calls": 1,
            "compile_calls": 1,
            "compile_wall_seconds": 0.0,
        }

    program = runner.LazyOneStepDomainExecutable(
        object(), "d02", runtime, tmp_path / "d02.json", builder=builder,
    )
    carry = FakeHealthCarry(2)
    assert program.advance(carry, 2932, 3) is carry
    assert builds == [("d02", 2932)]
    assert calls == [2932, 2933, 2934]
    assert program.audit()["compile_calls"] == 1
    assert program.audit()["dispatch_calls"] == 3
    with pytest.raises(runner.RunnerGateError, match="ORDINARY_LIVE_SIGNATURE_DRIFT"):
        program.one_step(FakeHealthCarry(3), 2935)
    assert builds == [("d02", 2932)]


def _write_exact_netcdf(path: Path, *, value: float = 1.0, title: str = "parent") -> None:
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("west_east", 2)
        dataset.setncattr("TITLE", title)
        variable = dataset.createVariable("T", "f8", ("Time", "west_east"), zlib=False)
        variable.setncattr("units", "K")
        variable[:] = np.asarray([[value, 2.0]], dtype=np.float64)


def test_parent_netcdf_comparator_is_value_and_metadata_bit_exact(tmp_path: Path) -> None:
    import numpy as np
    from netCDF4 import Dataset

    runtime = SimpleNamespace(np=np, Dataset=Dataset)
    reference = tmp_path / "reference.nc"
    exact = tmp_path / "exact.nc"
    data_bad = tmp_path / "data-bad.nc"
    metadata_bad = tmp_path / "metadata-bad.nc"
    _write_exact_netcdf(reference)
    _write_exact_netcdf(exact)
    _write_exact_netcdf(data_bad, value=1.0 + 2.0 ** -40)
    _write_exact_netcdf(metadata_bad, title="changed")
    good = runner.compare_netcdf_semantic_bits(reference, exact, runtime)
    assert good["passed"] is True
    assert good["all_variable_values_bit_exact"] is True
    assert good["all_dimensions_attributes_and_variable_metadata_exact"] is True
    data = runner.compare_netcdf_semantic_bits(reference, data_bad, runtime)
    assert data["passed"] is False
    assert any(row["category"] == "data" for row in data["first_differences"])
    metadata = runner.compare_netcdf_semantic_bits(reference, metadata_bad, runtime)
    assert metadata["passed"] is False
    assert any(row["category"] == "metadata" for row in metadata["first_differences"])


def test_join_target_comparison_uses_only_record_one_and_fails_on_target_change() -> None:
    import numpy as np

    def carry(old: float, target: float) -> object:
        state = SimpleNamespace(**{
            field: np.asarray([[old, old + 1.0], [target, target + 1.0]], dtype=np.float64)
            for field in runner.BOUNDARY_TARGET_TRANSITION_FIELDS
        })
        return SimpleNamespace(state=state)

    runtime = SimpleNamespace(
        np=np,
        jax=SimpleNamespace(device_get=lambda value: value),
    )
    edge = SimpleNamespace(parent_grid_ratio=3)
    retained = carry(10.0, 20.0)
    rebuilt_different_old = carry(-999.0, 20.0)
    verdict = runner.compare_join_boundary_target_records(
        retained, rebuilt_different_old, runtime, edge=edge,
    )
    assert verdict["passed"] is True
    assert verdict["cadence"]["d03_subcycle_position"] == 1
    assert verdict["rebuilt_carry_used_for_continuation"] is False
    bad = runner.compare_join_boundary_target_records(
        retained, carry(-999.0, 20.5), runtime, edge=edge,
    )
    assert bad["passed"] is False
    assert all(row["bit_exact"] is False for row in bad["rows"])


def test_boundary_target_transition_is_exact_source_backed_state_leaf_block() -> None:
    assert runner.BOUNDARY_TARGET_TRANSITION_LEAF_INDICES == tuple(range(38, 49))
    assert runner.BOUNDARY_TARGET_TRANSITION_FIELDS == (
        "u_bdy", "v_bdy", "theta_bdy", "qv_bdy", "ph_bdy", "mu_bdy",
        "w_bdy", "p_bdy", "pb_bdy", "phb_bdy", "mub_bdy",
    )
    source = (runner.REPO_ROOT / "src/gpuwrf/nesting/boundary_construction.py").read_text()
    assert "return jnp.stack([old, new], axis=0)" in source
    for name in runner.BOUNDARY_TARGET_TRANSITION_FIELDS:
        assert f"{name}=two_time(" in source


def test_lazy_health_skips_placeholder_and_compiles_once_on_live_shape() -> None:
    runtime = fake_health_runtime()
    placeholder = FakeHealthCarry(1)
    live = FakeHealthCarry(2)
    calls: list[dict[str, object]] = []
    compiled = object()

    def builder(carry: FakeHealthCarry, _runtime: SimpleNamespace) -> tuple[object, dict[str, object]]:
        calls.append(runner.health_carry_signature(carry, runtime))
        return compiled, {
            "compile_wall_seconds": 0.25,
            "lower_calls": 1,
            "compile_calls": 1,
            "bounded_outputs": {"nonfinite_by_leaf": 106},
        }

    program = runner.LazyLiveCarryHealthExecutable(runtime, builder=builder)
    assert program.compile_calls == 0
    assert runner.health_carry_signature(placeholder, runtime)["leaves"][38]["shape"][0] == 1
    assert program.executable_for(live) is compiled
    assert calls[0]["leaves"][38]["shape"][0] == 2
    assert program.executable_for(FakeHealthCarry(2)) is compiled
    assert len(calls) == 1
    assert program.audit()["initial_placeholder_carry_was_not_compiled"] is True
    assert program.audit()["signature_recompile_allowed"] is False


def test_lazy_health_rejects_post_compile_shape_drift_without_recompile() -> None:
    runtime = fake_health_runtime()
    calls = []

    def builder(_carry: object, _runtime: object) -> tuple[object, dict[str, object]]:
        calls.append(True)
        return object(), {
            "compile_wall_seconds": 0.0,
            "lower_calls": 1,
            "compile_calls": 1,
        }

    program = runner.LazyLiveCarryHealthExecutable(runtime, builder=builder)
    program.executable_for(FakeHealthCarry(2))
    with pytest.raises(runner.RunnerGateError, match="HEALTH_LIVE_SIGNATURE_DRIFT"):
        program.executable_for(FakeHealthCarry(3))
    assert calls == [True]
    assert program.compile_calls == 1


def test_lazy_health_rejects_incomplete_carry_before_compile() -> None:
    runtime = fake_health_runtime()
    called = []

    def builder(_carry: object, _runtime: object) -> tuple[object, dict[str, object]]:
        called.append(True)
        return object(), {}

    program = runner.LazyLiveCarryHealthExecutable(runtime, builder=builder)
    with pytest.raises(runner.RunnerGateError, match="HEALTH_LIVE_LEAF_COUNT"):
        program.executable_for(FakeHealthCarry(2, leaf_count=105))
    assert called == []


def test_lazy_health_rejects_builder_that_does_not_prove_one_compile() -> None:
    runtime = fake_health_runtime()
    program = runner.LazyLiveCarryHealthExecutable(
        runtime,
        builder=lambda _carry, _runtime: (
            object(), {"lower_calls": 1, "compile_calls": 0},
        ),
    )
    with pytest.raises(runner.RunnerGateError, match="HEALTH_COMPILE_AUDIT"):
        program.executable_for(FakeHealthCarry(2))
    assert program.compile_calls == 0


def test_standard_frame_schedule_matches_terminal_cpu_authority() -> None:
    manifest = json.loads(runner.CPU_MANIFEST.read_text())
    assert runner.standard_frame_schedule(runner.TERMINAL_OWN_STEPS) == manifest["raw_frame_schedules"]
    index, authority = runner.load_cpu_frame_index()
    assert len(index) == 93
    assert authority["sha256"] == runner.CPU_MANIFEST_SHA256
    assert ("d03", "2025-03-01_00:00:00") in index
    assert ("d03", "2025-03-01_18:00:00") in index


def test_real_rc3_continuation_authority_is_immutable_and_parent_blocked() -> None:
    authority = runner.assert_rc3_continuation_source_authority()
    assert authority["failure_proof"]["sha256"] == runner.RC3_FAILURE_PROOF_SHA256
    assert authority["failure_proof"]["canonical_payload_sha256"] == (
        runner.RC3_FAILURE_PROOF_PAYLOAD_SHA256
    )
    assert authority["step"] == 8800
    assert authority["valid_time"] == "2025-03-01T14:40:00+00:00"
    assert authority["leaf_count"] == 106
    assert authority["floating_nonfinite_count"] == 0
    assert authority["raw_output_counts"] == {"d01": 15, "d02": 15, "d03": 45}
    assert authority["frame_pair_counts"] == {"d01": 15, "d02": 15, "d03": 44}
    assert authority["missing_parent_carries"] == ["d01", "d02"]
    assert authority["carry_candidates"]["d03"]
    assert authority["continuation_ready"] is False
    assert [row["leading_time_records"] for row in authority["boundary_target_transition"]["rows"]] == [2] * 11
    assert {row["field"] for row in authority["boundary_target_transition"]["rows"]} == set(
        runner.BOUNDARY_TARGET_TRANSITION_FIELDS
    )
    blocker_codes = {row["code"] for row in authority["blockers"]}
    assert "PARENT_OPERATIONAL_CARRIES_NOT_RETAINED" in blocker_codes
    assert "TERMINAL_PARENT_OUTPUTS_UNPRODUCIBLE_WITH_D03_ONLY_CARRY" in blocker_codes
    assert "jax" not in sys.modules
    assert not any(name.startswith("gpuwrf") for name in sys.modules)


def test_exact_launch_command_binds_lock_candidate_audit_and_scope(tmp_path: Path) -> None:
    launch = runner.LAUNCH_COMMAND
    audit = runner.audit_exact_launch_command(launch)
    assert audit["passed"] is True
    assert audit["one_model_process"] is True
    text = launch.read_text()
    assert runner.FULL_REPLAY_NAMESPACE in text
    assert "--direct-terminal" in text
    assert "--parent-join-resume" not in text
    assert "--hold-seconds" not in text
    assert "--continuation-authority" not in text
    assert "GPUWRF_NESTED_BUNDLE_CRITIC_ACCEPTANCE" not in text
    assert f"GPUWRF_FINAL_NI_OWNER_OVERRIDE={runner.OWNER_OVERRIDE}" in text
    assert "GPUWRF_JAX_CACHE=0" in text
    assert "GPUWRF_JAX_CACHE_LOCK=0" in text
    assert "JAX_ENABLE_COMPILATION_CACHE=false" in text
    assert "GPUWRF_JAX_CACHE_DIR=" not in text
    assert "JAX_COMPILATION_CACHE_DIR=" not in text
    assert "nvidia-smi" not in text
    mutated = tmp_path / "launch.txt"
    mutated.write_text(launch.read_text().replace("--intent production-preemptible", "--intent debug"))
    assert runner.audit_exact_launch_command(mutated)["passed"] is False


def test_same_object_carry_continuity_and_rejection() -> None:
    carries = {"d01": object(), "d02": object(), "d03": object()}
    proof = runner.assert_same_carry_continuity(carries, carries)
    assert proof["same_object_per_domain"] is True
    changed = dict(carries)
    changed["d03"] = object()
    with pytest.raises(runner.RunnerGateError, match="CARRY_CONTINUITY"):
        runner.assert_same_carry_continuity(carries, changed)


@pytest.mark.parametrize("step", [8800, 9000, 9198])
def test_prefix_complete_carry_gates(step: int) -> None:
    assert runner.evaluate_health_summary(
        step, healthy_summary(), step9313_scale_baseline=None,
    )["passed"] is True
    bad = healthy_summary()
    bad["nonfinite_by_leaf"] = [0] * 105 + [1]
    verdict = runner.evaluate_health_summary(step, bad, step9313_scale_baseline=None)
    assert verdict["passed"] is False
    assert verdict["complete_carry_nonfinite_count"] == 1


@pytest.mark.parametrize("step", [9214, 9260, 9313])
def test_last_100_theta_ceiling_gate(step: int) -> None:
    good = healthy_summary()
    good["theta_max"] = 999.999
    assert runner.evaluate_health_summary(step, good, step9313_scale_baseline=None)["passed"]
    bad = healthy_summary()
    bad["theta_max"] = 1000.0
    verdict = runner.evaluate_health_summary(step, bad, step9313_scale_baseline=None)
    assert not verdict["passed"]
    assert any(row["gate"] == "theta_below_existing_limiter_ceiling" for row in verdict["violations"])


@pytest.mark.parametrize(
    ("key", "value", "gate"),
    [
        ("corner_mu_pert_maxabs", 2000.0, "corner_mu_pert_abs_lt_2000"),
        ("wind_maxabs", [80.0, 1.0, 1.0], "u_abs_lt_80"),
        ("wind_maxabs", [1.0, 80.0, 1.0], "v_abs_lt_80"),
        ("wind_maxabs", [1.0, 1.0, 30.0], "w_abs_lt_30"),
    ],
)
def test_9313_physical_bounds_fail_at_boundary(key: str, value: object, gate: str) -> None:
    summary = healthy_summary()
    summary[key] = value
    verdict = runner.evaluate_health_summary(9313, summary, step9313_scale_baseline=None)
    assert not verdict["passed"]
    assert any(row["gate"] == gate for row in verdict["violations"])


def test_9314_scale_amplification_and_positivity_gates() -> None:
    baseline = [10.0] * len(runner.ALL_SCALE_FIELDS)
    assert runner.evaluate_health_summary(
        9314, healthy_summary(), step9313_scale_baseline=baseline,
    )["passed"]
    missing = runner.evaluate_health_summary(
        9314, healthy_summary(), step9313_scale_baseline=None,
    )
    assert not missing["passed"]
    absolute = healthy_summary()
    absolute["scale_maxabs"] = [10.0] * len(runner.ALL_SCALE_FIELDS)
    absolute["scale_maxabs"][0] = runner.ABSOLUTE_SCALE_CEILING + 1.0
    assert not runner.evaluate_health_summary(9314, absolute, step9313_scale_baseline=baseline)["passed"]
    amplification = healthy_summary()
    amplification["scale_maxabs"] = [10.0] * len(runner.ALL_SCALE_FIELDS)
    amplification["scale_maxabs"][0] = 1001.0
    tiny_baseline = [1.0] * len(runner.ALL_SCALE_FIELDS)
    assert not runner.evaluate_health_summary(9314, amplification, step9313_scale_baseline=tiny_baseline)["passed"]
    nonpositive = healthy_summary()
    nonpositive["positive_min"] = [0.0, 1.0]
    assert not runner.evaluate_health_summary(9314, nonpositive, step9313_scale_baseline=baseline)["passed"]


def test_9315_ni_and_complete_carry_finiteness() -> None:
    assert runner.evaluate_health_summary(
        9315, healthy_summary(), step9313_scale_baseline=[10.0] * len(runner.ALL_SCALE_FIELDS),
    )["passed"]
    bad = healthy_summary()
    bad["ni_nonfinite_count"] = 1
    verdict = runner.evaluate_health_summary(9315, bad, step9313_scale_baseline=None)
    assert not verdict["passed"]
    assert any(row["gate"] == "state_Ni_finite" for row in verdict["violations"])


def test_9405_reuses_scale_and_positivity_gates() -> None:
    baseline = [10.0] * len(runner.ALL_SCALE_FIELDS)
    assert runner.evaluate_health_summary(
        9405, healthy_summary(), step9313_scale_baseline=baseline,
    )["passed"]
    bad = healthy_summary()
    bad["positive_min"] = [1.0, -1.0]
    assert not runner.evaluate_health_summary(9405, bad, step9313_scale_baseline=baseline)["passed"]


def metric_payload(value: float, *, static: bool = True) -> dict[str, object]:
    return {
        "metrics": {field: {"rmse": value} for field in runner.STRICT_FIELDS},
        "per_frame_static_pass": static,
    }


def test_cpu_metric_comparator_is_fieldwise_no_worse() -> None:
    baseline = {
        "metrics": {
            field: {"rmse": value} for field, value in runner.FROZEN_1500_RMSE.items()
        },
        "per_frame_static_pass": True,
    }
    candidate = {
        "metrics": {
            field: {"rmse": value} for field, value in runner.FROZEN_1500_RMSE.items()
        },
        "per_frame_static_pass": True,
    }
    verdict = runner.compare_no_worse_metrics(candidate, baseline)
    assert verdict["passed"] is True
    worse = json.loads(json.dumps(candidate))
    worse["metrics"]["V10"]["rmse"] = runner.FROZEN_1500_RMSE["V10"] + 1e-12
    verdict = runner.compare_no_worse_metrics(worse, baseline)
    assert verdict["passed"] is False
    assert verdict["v10_causal_use"] is False
    changed_baseline = json.loads(json.dumps(baseline))
    changed_baseline["metrics"]["T"]["rmse"] += 1e-12
    assert not runner.compare_no_worse_metrics(candidate, changed_baseline)["passed"]
    static_bad = json.loads(json.dumps(candidate))
    static_bad["per_frame_static_pass"] = False
    assert not runner.compare_no_worse_metrics(
        static_bad, baseline,
    )["passed"]


@pytest.mark.parametrize(
    "stamp",
    ["2025-03-01T15:00:00+00:00", "2025-03-01T15:20:00+00:00"],
)
def test_real_retry20_cpu_comparator_and_frozen_geometry_authority(stamp: str) -> None:
    from scripts import v0234_corrected_fullbuffer_gate as gate

    pairs = json.loads(runner.RETRY20_PAIRS.read_text())
    valid = datetime.fromisoformat(stamp)
    if stamp.endswith("15:00:00+00:00"):
        retained = next(row for row in pairs["pairs"] if row["valid_time"] == stamp)
        cpu = Path(retained["cpu_snapshot_path"])
        gpu = Path(retained["gpu_snapshot_path"])
        cpu_snapshot = retained["cpu_snapshot_authority"]
        gpu_snapshot = retained["gpu_snapshot_authority"]
    else:
        closure = pairs["closure_observations"][stamp]
        cpu = Path(closure["cpu"]["authority"]["path"])
        gpu = Path(closure["gpu"]["authority"]["path"])
        retained = None
        cpu_snapshot = None
        gpu_snapshot = None
    recomputed = gate.compare_pair(
        cpu, gpu, valid,
        cpu_snapshot=cpu_snapshot,
        gpu_snapshot=gpu_snapshot,
        frozen_geometry_policy="terminal_nested_boundary_v1",
    )
    assert recomputed["per_frame_static_pass"] is True
    assert set(runner.STRICT_FIELDS) <= set(recomputed["metrics"])
    if retained is not None:
        for field in runner.STRICT_FIELDS:
            assert recomputed["metrics"][field]["rmse"] == retained["metrics"][field]["rmse"]
        assert runner.compare_no_worse_metrics(recomputed, retained)["passed"] is True
    assert "jax" not in sys.modules


def test_standard_output_materializes_and_pairs_every_alarm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner, "assert_preemption_clear", lambda label: {"label": label}
    )
    class Writer:
        def __call__(self, name: str, step: int, carry: object) -> dict[str, object]:
            valid = runner.RUN_START + timedelta(seconds=step * runner.DT_SECONDS[name])
            path = tmp_path / f"wrfout_{name}_{valid.strftime('%Y-%m-%d_%H:%M:%S')}"
            path.write_bytes(b"frame")
            return {"wrfout": str(path), "domain": name, "step": step}

    class Pairer:
        def __init__(self) -> None:
            self.calls = []
            self.counts = {"d01": 0, "d02": 0, "d03": 0}

        def pair(self, name: str, step: int, path: Path) -> dict[str, object]:
            self.calls.append((name, step, path))
            self.counts[name] += 1
            return {"finite_identity_pass": True}

    pairer = Pairer()
    output = runner.StandardCadenceOutput(
        Writer(), pairer, tmp_path, object(), SimpleNamespace(), tmp_path,
    )
    carry = object()
    for name, step in (("d01", 0), ("d02", 0), ("d03", 0), ("d03", 200)):
        output(name, step, carry)
    assert [(row["domain"], row["own_step"]) for row in output.emitted] == [
        ("d01", 0), ("d02", 0), ("d03", 0), ("d03", 200),
    ]
    assert len(pairer.calls) == 4
    assert pairer.counts == {"d01": 1, "d02": 1, "d03": 2}
    with pytest.raises(runner.MetricGateFailure, match="DUPLICATE_OUTPUT_ALARM"):
        output("d03", 200, carry)


def test_step9000_writer_retain_health_pair_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    events = []
    monkeypatch.setattr(
        runner, "assert_preemption_clear", lambda label: {"label": label}
    )

    class Writer:
        def __call__(self, name: str, step: int, carry: object) -> dict[str, object]:
            events.append("writer")
            valid = runner.RUN_START + timedelta(seconds=step * runner.DT_SECONDS[name])
            path = tmp_path / f"wrfout_{name}_{valid.strftime('%Y-%m-%d_%H:%M:%S')}"
            path.write_bytes(b"closed")
            return {"wrfout": str(path)}

    class Pairer:
        counts = {"d01": 0, "d02": 0, "d03": 0}

        def pair(self, name: str, step: int, path: Path) -> dict[str, object]:
            events.append("pair")
            return {"finite_identity_pass": True}

    def retain(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("retain")
        return {"reread_identity": {"all_leaf_bytes_equal": True}}

    def health(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("health")
        return healthy_summary()

    monkeypatch.setattr(runner, "_retain_complete_carry", retain)
    monkeypatch.setattr(runner, "materialize_health", health)
    output = runner.StandardCadenceOutput(
        Writer(), Pairer(), tmp_path, object(), SimpleNamespace(), tmp_path,
    )
    result = output("d03", 9000, object())
    assert events == ["writer", "retain", "health", "pair"]
    assert result["checkpoint"]["health"]["passed"] is True


def test_preempt_sentinel_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sentinel = tmp_path / "PREEMPT"
    sentinel.write_text("1")
    monkeypatch.setattr(runner, "PREEMPT_PATHS", (sentinel,))
    monkeypatch.setattr(runner, "HOLD_PATHS", ())
    monkeypatch.setattr(runner, "NIGHTLY_ACTIVE", tmp_path / "missing-nightly")
    monkeypatch.setattr(runner, "GPU_LOCK_HOLDER", tmp_path / "missing-holder")
    monkeypatch.setattr(runner, "_active_production_gpu_processes", lambda: [])
    with pytest.raises(runner.RunnerGateError, match="PREEMPT"):
        runner.assert_preemption_clear("test")
    sentinel.unlink()
    assert runner.assert_preemption_clear("test")["label"] == "test"


def test_waiting_source_without_preempt_hold_gpu_or_lock_is_admitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    nightly = tmp_path / "active.json"
    nightly.write_text(json.dumps({
        "active": True,
        "run_id": "20260716_00z",
        "status": "waiting_source",
    }))
    monkeypatch.setattr(runner, "PREEMPT_PATHS", ())
    monkeypatch.setattr(runner, "HOLD_PATHS", ())
    monkeypatch.setattr(runner, "NIGHTLY_ACTIVE", nightly)
    monkeypatch.setattr(runner, "GPU_LOCK_HOLDER", tmp_path / "missing-holder")
    monkeypatch.setattr(runner, "_active_production_gpu_processes", lambda: [])

    result = runner.assert_preemption_clear("test")
    assert result["nightly_active"] is True
    assert result["nightly_status"] == "waiting_source"
    assert result["nightly_non_gpu_admitted"] is True
    assert result["production_gpu_processes"] == []
    assert result["foreign_gpu_lock_holder"] is False


@pytest.mark.parametrize(
    ("argv", "expected"),
    (
        (
            [
                "/bin/bash",
                "-c",
                "from concurrent.futures import ThreadPoolExecutor; "
                "from alisios.pipeline.nightly_profiles import wn2_remote_preflight",
            ],
            False,
        ),
        (["/usr/local/cuda/bin/ncu", "--set", "full"], True),
        (["/usr/local/cuda/bin/nsys", "profile", "python"], True),
        (["python", "-m", "gpuwrf.runtime.operational_mode"], True),
        (["/bin/bash", "/tmp/gpuwrf-production-run.sh"], True),
    ),
)
def test_production_gpu_process_markers_are_identifier_boundary_aware(
    argv: list[str], expected: bool,
) -> None:
    assert runner._has_production_gpu_process_marker(argv) is expected


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        (
            "env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "
            "python -c 'from gpuwrf.runtime.domain_tree import run_domain_tree'",
            True,
        ),
        (
            "env JAX_PLATFORM_NAME='cpu' CUDA_VISIBLE_DEVICES=\"\" "
            "python -m gpuwrf.runtime.operational_mode",
            True,
        ),
        (
            "env JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 "
            "python -m gpuwrf.runtime.operational_mode",
            False,
        ),
        (
            "env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 "
            "python -m gpuwrf.runtime.operational_mode",
            False,
        ),
        (
            "env CUDA_VISIBLE_DEVICES= python -m gpuwrf.runtime.operational_mode",
            False,
        ),
    ),
)
def test_explicit_cpu_only_process_requires_cpu_platform_and_hidden_cuda(
    command: str, expected: bool,
) -> None:
    assert runner._declares_explicit_cpu_only_no_cuda(
        ["/bin/bash", "-c", command]
    ) is expected


def test_explicit_hold_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    hold = tmp_path / "HOLD_GPU"
    hold.write_text("HOLD")
    monkeypatch.setattr(runner, "PREEMPT_PATHS", ())
    monkeypatch.setattr(runner, "HOLD_PATHS", (hold,))
    with pytest.raises(runner.RunnerGateError, match="HOLD"):
        runner.assert_preemption_clear("test")


def test_live_production_gpu_process_fails_closed_even_while_waiting_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    nightly = tmp_path / "active.json"
    nightly.write_text(json.dumps({"active": True, "status": "waiting_source"}))
    monkeypatch.setattr(runner, "PREEMPT_PATHS", ())
    monkeypatch.setattr(runner, "HOLD_PATHS", ())
    monkeypatch.setattr(runner, "NIGHTLY_ACTIVE", nightly)
    monkeypatch.setattr(runner, "GPU_LOCK_HOLDER", tmp_path / "missing-holder")
    monkeypatch.setattr(
        runner,
        "_active_production_gpu_processes",
        lambda: [{"pid": 77, "argv": ["python", "gpuwrf-production"]}],
    )
    with pytest.raises(runner.RunnerGateError, match="PREEMPT_PRODUCTION_GPU_ACTIVE"):
        runner.assert_preemption_clear("test")


def test_foreign_lock_holder_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    holder = tmp_path / "holder"
    holder.write_text("foreign")
    monkeypatch.setattr(runner, "PREEMPT_PATHS", ())
    monkeypatch.setattr(runner, "HOLD_PATHS", ())
    monkeypatch.setattr(runner, "NIGHTLY_ACTIVE", tmp_path / "missing-nightly")
    monkeypatch.setattr(runner, "GPU_LOCK_HOLDER", holder)
    monkeypatch.setattr(runner, "_active_production_gpu_processes", lambda: [])
    for name in (
        "GPUWRF_GPU_LOCK_HELD",
        "GPUWRF_GPU_LOCK_HOLDER_FILE",
        "GPUWRF_GPU_LOCK_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(runner.RunnerGateError, match="PREEMPT_PRODUCTION_GPU_LOCKED"):
        runner.assert_preemption_clear("test")


class FakeOrdinary:
    @staticmethod
    def host_tree_manifest(value: object) -> dict[str, object]:
        raw = pickle.dumps(value, protocol=5)
        return {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "leaf_count": 1,
            "floating_nonfinite_count": 0,
            "leaves": [],
            "treedef": "fake",
        }

    @staticmethod
    def compare_manifests(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
        equal = left["manifest_sha256"] == right["manifest_sha256"]
        return {"all_leaf_bytes_equal": equal, "different_leaf_count": 0 if equal else 1}


def test_atomic_failure_retention_authenticates_both_carries(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        jax=SimpleNamespace(device_get=lambda value: value),
        ordinary=FakeOrdinary,
    )
    proof = runner._retain_failure_pair(
        runtime,
        tmp_path,
        last_step=9313,
        last_carry={"value": 1},
        failed_step=9314,
        failed_carry={"value": 2},
        failure={"code": "TEST"},
    )
    assert proof["status"] == "ATOMIC_FAILURE_CAPTURED"
    assert proof["last_healthy"]["reread_identity"]["all_leaf_bytes_equal"]
    assert proof["first_failed"]["reread_identity"]["all_leaf_bytes_equal"]
    assert (tmp_path / "failure/failure-proof.json").is_file()


def test_hold_refuses_without_fresh_authority(tmp_path: Path) -> None:
    called = []
    carries = {"d03": object()}
    result = runner.hold_for_fresh_continuation(
        carries,
        authority_path=tmp_path / "authority.json",
        hold_seconds=0.0,
        runner_commit="a" * 40,
        window_proof_sha256="b" * 64,
        continuation=lambda value: called.append(value),
    )
    assert result["decision"] == "HOLD_EXPIRED_NO_CONTINUATION"
    assert called == []
    assert (tmp_path / "continuation-request.json").is_file()


def test_hold_accepts_only_fresh_exact_authority_and_reuses_same_object(tmp_path: Path) -> None:
    authority_path = tmp_path / "authority.json"
    request_path = tmp_path / "continuation-request.json"
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    carries = {"d03": object()}
    seen = []

    def sleep_fn(_seconds: float) -> None:
        request = json.loads(request_path.read_text())
        payload = {
            "schema": "gpuwrf.v0234.nested-frozen-wrf-boundary-continuation-authorization.v1",
            "decision": "CONTINUE_SAME_PROCESS_TO_18H",
            "nonce": request["nonce"],
            "runner_commit": "a" * 40,
            "window_proof_sha256": "b" * 64,
            "reuse_live_carries": True,
            "reuse_compiled_d03_one_step": True,
            "no_second_prefix": True,
            "terminal_d03_step": 10800,
            "issued_utc": now.isoformat(),
            "expires_utc": (now + timedelta(seconds=30)).isoformat(),
        }
        authority_path.write_text(json.dumps(payload))

    def continuation(value: object) -> object:
        seen.append(value is carries)
        return value

    result = runner.hold_for_fresh_continuation(
        carries,
        authority_path=authority_path,
        hold_seconds=0.05,
        runner_commit="a" * 40,
        window_proof_sha256="b" * 64,
        continuation=continuation,
        sleep_fn=sleep_fn,
        now_fn=lambda: now,
    )
    assert result["decision"] == "AUTHORIZED_CONTINUATION_EXECUTED"
    assert seen == [True]


def test_continuation_authority_rejects_stale_or_wrong_binding() -> None:
    now = datetime.now(timezone.utc)
    payload = {
        "schema": "gpuwrf.v0234.nested-frozen-wrf-boundary-continuation-authorization.v1",
        "decision": "CONTINUE_SAME_PROCESS_TO_18H",
        "nonce": "nonce",
        "runner_commit": "a" * 40,
        "window_proof_sha256": "b" * 64,
        "reuse_live_carries": True,
        "reuse_compiled_d03_one_step": True,
        "no_second_prefix": True,
        "terminal_d03_step": 10800,
        "issued_utc": (now - timedelta(hours=1)).isoformat(),
        "expires_utc": (now - timedelta(minutes=30)).isoformat(),
    }
    with pytest.raises(runner.RunnerGateError, match="CONTINUATION_FRESHNESS"):
        runner.validate_continuation_authority(
            payload,
            nonce="nonce",
            runner_commit="a" * 40,
            window_proof_sha256="b" * 64,
            now=now,
        )
    payload["issued_utc"] = (now - timedelta(seconds=1)).isoformat()
    payload["expires_utc"] = (now + timedelta(seconds=30)).isoformat()
    payload["no_second_prefix"] = False
    with pytest.raises(runner.RunnerGateError, match="CONTINUATION_BINDING"):
        runner.validate_continuation_authority(
            payload,
            nonce="nonce",
            runner_commit="a" * 40,
            window_proof_sha256="b" * 64,
            now=now,
        )


def test_runner_audit_file_semantics(tmp_path: Path) -> None:
    payload = {
        "schema": runner.CPU_PROOF_SCHEMA,
        "verdict": runner.AUDIT_ADMISSION,
        "candidate_commit": runner.CANDIDATE_COMMIT,
        "candidate_tree": runner.CANDIDATE_TREE,
        "runner_head_at_audit": subprocess.check_output(
            ["git", "-C", str(runner.REPO_ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "runner_source_sha256": runner.sha256_file(runner.RUNNER_SOURCE),
        "candidate_clean_authority": runner.CANDIDATE_CLEAN_AUTHORITY,
        "deterministic_payload": True,
        "gpu_commands_run": 0,
        "jax_imported": False,
        "focused_tests": {"returncode": 0, "stdout_sha256": "c" * 64},
        "static_audit": {"passed": True},
        "exact_launch_audit": {"passed": True},
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload))
    env = {
        "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT": str(path),
        "GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT_SHA256": runner.sha256_file(path),
    }
    assert runner.validate_runner_audit(env)["verdict"] == runner.AUDIT_ADMISSION
    payload["gpu_commands_run"] = 1
    path.write_text(json.dumps(payload))
    env["GPUWRF_NESTED_BUNDLE_RUNNER_AUDIT_SHA256"] = runner.sha256_file(path)
    with pytest.raises(runner.RunnerGateError, match="RUNNER_AUDIT_SEMANTICS"):
        runner.validate_runner_audit(env)


def test_authenticated_json_is_single_read_regular_and_rejects_symlink(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text('{"ok": true}\n')
    digest = runner.sha256_file(path)
    payload, row = runner.read_authenticated_json(path, digest, "TEST_AUTHORITY")
    assert payload == {"ok": True}
    assert row["single_read_stable"] is True
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(runner.RunnerGateError, match="symlink rejected"):
        runner.read_authenticated_json(link, digest, "TEST_AUTHORITY")


def test_static_source_audit_binds_dispatch_order_and_no_model_diff() -> None:
    audit = runner.static_source_audit()
    assert audit["passed"] is True
    assert audit["forbidden_top_level_imports"] == []
    assert audit["preempt_dispatch_health_ordered"] is True
    assert audit["normal_writer_checkpoint_health_pair_ordered"] is True
    assert audit["explicit_compile_counts_exact"] is True
    assert audit["incremental_pairing_uses_host_files_only"] is True
    assert audit["lowered_hlo_artifact_precedes_gate"] is True
    assert audit["lower_only_diagnostic_never_compiles"] is True
    assert audit["lazy_health_binds_first_live_shape_exactly_once"] is True
    assert audit["parent_prejoin_zero_d03_dispatch"] is True
    assert audit["parent_one_compile_per_domain"] is True
    assert audit["parent_mixed_integral_scheduler_projection"] is True
    assert audit["parent_real_schema_identity_rehearsal"] is True
    assert audit["parent_blocker_retains_stage_and_traceback"] is True
    assert audit["parent_continuation_health_after_dispatch"] is True
    assert audit["parent_join_target_record_fail_closed"] is True
    assert audit["terminal_union_uses_immutable_snapshots"] is True
    assert audit["terminal_cpu_uses_marker_free_canonical_contract"] is True
    assert audit["domain_specific_exact_interface_audit"] is True
    assert audit["direct_full_terminal_19_19_55_flow"] is True
    assert audit["explicit_lead_zero_normal_history"] is True
    assert audit["continuation_retains_normal_output_callback"] is True
    assert audit["accepted_model_diff"] == []
    assert audit["gpu_query_command_tokens"] == []


def test_custom_call_extraction_is_exact_bounded_and_hashed(tmp_path: Path) -> None:
    stablehlo = (
        'module {\n  %0 = stablehlo.custom_call @Sharding(%arg0) '
        '{backend_config = "device=0"} : (tensor<1xf32>) -> tensor<1xf32>\n'
        '  %1 = stablehlo.custom_call @"xla.gpu.device_internal"(%0) '
        ': (tensor<1xf32>) -> tensor<1xf32>\n}\n'
    )
    extracted = runner.extract_stablehlo_custom_calls(stablehlo)
    assert extracted["extraction_complete"] is True
    assert extracted["custom_call_syntax_occurrence_count"] == 2
    assert extracted["targets"] == ["Sharding", "xla.gpu.device_internal"]
    assert all(row["snippet"] in stablehlo for row in extracted["occurrences"])
    assert all(runner.sha256_text(row["snippet"]) == row["snippet_sha256"] for row in extracted["occurrences"])
    artifact_path = tmp_path / "lowered.json"
    retained = runner.atomically_retain_lowered_hlo_audit(artifact_path, stablehlo)
    payload = json.loads(artifact_path.read_text())
    assert retained["file_sha256"] == runner.sha256_file(artifact_path)
    assert payload["compile_calls_before_artifact"] == 0
    assert payload["dispatch_calls_before_artifact"] == 0
    assert payload["extraction"]["stablehlo_sha256"] == runner.sha256_text(stablehlo)


def test_exact_allowed_device_target_passes_but_unknown_and_malformed_fail_closed() -> None:
    allowed = 'module { %0 = stablehlo.custom_call @Sharding() : () -> tensor<i32> }'
    verdict = runner.evaluate_stablehlo_policy(
        allowed, allowed_custom_targets=frozenset({"Sharding"}),
    )
    assert verdict["passed"] is True
    assert verdict["custom_call_targets"] == ["Sharding"]
    unknown = runner.evaluate_stablehlo_policy(
        allowed, allowed_custom_targets=frozenset({"DifferentTarget"}),
    )
    assert unknown["passed"] is False
    assert unknown["unknown_custom_call_targets"] == ["Sharding"]
    case_changed = runner.evaluate_stablehlo_policy(
        allowed.replace("@Sharding", "@sharding"),
        allowed_custom_targets=frozenset({"Sharding"}),
    )
    assert case_changed["passed"] is False
    malformed = runner.evaluate_stablehlo_policy(
        "module { stablehlo.custom_call %arg0 : tensor<i32> }",
        allowed_custom_targets=frozenset(),
    )
    assert malformed["passed"] is False
    assert malformed["unparsed_custom_call_count"] == 1


def test_production_policy_allows_only_exact_cuda_tridiagonal_device_target() -> None:
    target = "cusparse_gtsv2_ffi"
    stablehlo = (
        "module { %0 = stablehlo.custom_call @cusparse_gtsv2_ffi(%arg0) "
        '{mhlo.frontend_attributes = {num_batch_dims = "1"}} '
        ": (tensor<45x10494xf64>) -> tensor<45x10494xf64> }"
    )
    assert runner.ALLOWED_DEVICE_CUSTOM_CALL_TARGETS == frozenset({target})
    verdict = runner.evaluate_stablehlo_policy(stablehlo)
    assert verdict["passed"] is True
    assert verdict["custom_call_targets"] == [target]
    assert verdict["unknown_custom_call_targets"] == []


@pytest.mark.parametrize(
    "near_match",
    [
        "cusparse_gtsv2",
        "cusparse_gtsv2_ffi_extra",
        "hipsparse_gtsv2_ffi",
        "xla.gpu.cusparse_gtsv2_ffi",
        "CUSPARSE_GTSV2_FFI",
    ],
)
def test_production_policy_rejects_every_near_match_of_device_target(near_match: str) -> None:
    stablehlo = (
        f"module {{ %0 = stablehlo.custom_call @{near_match}() : () -> tensor<i32> }}"
    )
    verdict = runner.evaluate_stablehlo_policy(stablehlo)
    assert verdict["passed"] is False
    assert verdict["unknown_custom_call_targets"] == [near_match]


def test_allowed_device_target_does_not_mask_a_host_callback_sibling() -> None:
    stablehlo = (
        "module {\n"
        "  %0 = stablehlo.custom_call @cusparse_gtsv2_ffi() : () -> tensor<i32>\n"
        "  %1 = stablehlo.custom_call @xla_python_cpu_callback() : () -> tensor<i32>\n"
        "}\n"
    )
    verdict = runner.evaluate_stablehlo_policy(stablehlo)
    assert verdict["passed"] is False
    assert verdict["custom_call_targets"] == [
        "cusparse_gtsv2_ffi", "xla_python_cpu_callback",
    ]
    assert verdict["forbidden_custom_call_targets"] == ["xla_python_cpu_callback"]
    assert verdict["unknown_custom_call_targets"] == ["xla_python_cpu_callback"]


@pytest.mark.parametrize("token", runner.FORBIDDEN_HLO_TEXT_TOKENS)
def test_every_forbidden_hlo_text_token_fails(token: str) -> None:
    verdict = runner.evaluate_stablehlo_policy(
        f"module {{ // {token}\n }}", allowed_custom_targets=frozenset(),
    )
    assert verdict["passed"] is False
    assert token in verdict["forbidden_text_tokens"]


@pytest.mark.parametrize(
    "target",
    [
        "xla_python_cpu_callback",
        "gpu_host_bridge",
        "io_callback",
        "pure_callback",
        "debug_callback",
        "outside_compilation",
        "device_send",
        "device_recv",
    ],
)
def test_forbidden_custom_target_fails_even_if_explicitly_allowlisted(target: str) -> None:
    stablehlo = f"module {{ %0 = stablehlo.custom_call @{target}() : () -> tensor<i32> }}"
    verdict = runner.evaluate_stablehlo_policy(
        stablehlo, allowed_custom_targets=frozenset({target}),
    )
    assert verdict["passed"] is False
    assert target in verdict["forbidden_custom_call_targets"]


def test_model_hlo_policy_is_target_aware_and_keeps_callback_denials() -> None:
    source = runner.RUNNER_SOURCE.read_text()
    assert "ALLOWED_DEVICE_CUSTOM_CALL_TARGETS" in source
    assert '"cusparse_gtsv2_ffi"' in source
    assert "unknown_custom_call_targets" in source
    assert '"outside_compilation"' in source
    assert '"host_callback"' in source
    assert 'domain_leaf_contract = 106 if domain == "d03" else None' in source
    assert "input_tree == output_tree and input_avals == output_avals" in source


def test_parent_scheduler_projection_handles_real_mixed_integral_schema() -> None:
    schedule = {
        "nonintegral_output_alarms": {"d01": [67, 134]},
        "output_cadence_steps": {"d01": 67, "d02": 200, "d03": 200},
    }
    runtime = SimpleNamespace(ordinary=SimpleNamespace(
        scheduler_contract=lambda *_args, **_kwargs: (
            {"d01": 67, "d02": 200, "d03": 200},
            {"d01": (67, 134)},
            schedule,
        ),
    ))
    cadence, nonintegral, observed = runner.parent_scheduler_contract(
        runtime, {"d01": 54.0, "d02": 18.0, "d03": 6.0},
    )
    assert cadence == {"d01": 67, "d02": 200}
    assert nonintegral == {"d01": (67, 134)}
    assert "d02" not in nonintegral
    assert observed is schedule


def test_parent_blocker_retains_stage_trace_and_exact_traceback(tmp_path: Path) -> None:
    args = SimpleNamespace(
        run_dir=tmp_path,
        _parent_join_stage="PARENT_RECON_SCHEDULER_PROJECTION",
        _parent_join_stages=[
            "PRE_RUNTIME", "DOMAIN_LOAD", "PARENT_RECON_SCHEDULER_PROJECTION",
        ],
    )
    proof = runner._retain_parent_join_blocker(
        args,
        "TEST_LINE",
        "KeyError: d02",
        traceback_text=(
            "Traceback (most recent call last):\n"
            "  File \"runner.py\", line 3228, in run_parent_only_reconstitution\n"
            "KeyError: 'd02'\n"
        ),
    )
    retained = json.loads((tmp_path / "parent-join-blocker.json").read_text())
    assert retained == proof
    assert retained["stage"] == "PARENT_RECON_SCHEDULER_PROJECTION"
    assert retained["stage_trace"][-1] == retained["stage"]
    assert "line 3228" in retained["traceback"]
    assert retained["proof_sha256"] == runner.canonical_digest({
        key: value for key, value in retained.items() if key != "proof_sha256"
    })


def test_real_schema_parent_cpu_rehearsal(tmp_path: Path) -> None:
    rehearsal = tmp_path / "real-schema"
    code = (
        "from pathlib import Path; "
        "from scripts import v0234_nested_frozen_wrf_boundary_window as r; "
        "p=r.run_real_schema_parent_cpu_rehearsal(Path('" + str(rehearsal) + "')); "
        "assert p['verdict']=='PARENT_REAL_SCHEMA_REHEARSAL_GREEN'"
    )
    env = os.environ.copy()
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    env.pop("GPUWRF_JAX_CACHE_DIR", None)
    env.update({
        "PYTHONPATH": f"{runner.REPO_ROOT}:{runner.REPO_ROOT / 'src'}",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "GPUWRF_JAX_CACHE": "0",
        "GPUWRF_WRF_ROOT": str(runner.RETRY20_WRF_ROOT),
        "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
        "GPUWRF_NESTED_FUSE": "0",
        "GPUWRF_NESTED_AOT": "0",
        "GPUWRF_NESTED_ASYNC_OUTPUT": "0",
        "GPUWRF_NEST_OUTPUT_PIPELINE": "0",
        "GPUWRF_ADVANCE_CHUNK_LOOP": "fori",
    })
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=runner.REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    proof = json.loads((rehearsal / "real-schema-rehearsal.json").read_text())
    assert proof["domains"] == {
        "loaded": ["d01", "d02", "d03"],
        "hierarchy": ["d01", "d02"],
        "carries": ["d01", "d02"],
        "bundles": ["d01", "d02"],
        "writer": ["d01", "d02"],
        "scheduler_output": ["d01", "d02"],
        "leaf_counts": {"d01": 115, "d02": 106},
    }
    assert proof["edge_lookup"] == {
        "parent": "d01", "child": "d02", "ratio": 3, "weights_present": True,
    }
    scheduler = proof["scheduler"]
    assert scheduler["cadence"] == {"d01": 67, "d02": 200}
    assert scheduler["nonintegral_override_domains"] == ["d01"]
    assert [row["domain"] for row in scheduler["initial_history"]] == ["d01", "d02"]
    assert scheduler["initial_history_calls"] == [["d01", 0], ["d02", 0]]
    assert scheduler["initial_history_counts"] == {"d01": 1, "d02": 1}
    assert scheduler["writer_counts"] == {"d01": 1, "d02": 1}
    assert scheduler["first_recursive_own_steps"] == {"d01": 1, "d02": 3}
    assert scheduler["event_counts"] == {"advance": 2, "force": 1}
    assert scheduler["output_calls_after_first_recursion"] == [["d01", 0], ["d02", 0]]
    assert scheduler["output_counts_after_first_recursion"] == {"d01": 1, "d02": 1}
    assert proof["ordinary_model_lowers"] == 0
    assert proof["ordinary_model_compiles"] == 0
    assert proof["ordinary_model_dispatches"] == 0
    assert proof["cpu_state_constructor_override_restored"] is True
