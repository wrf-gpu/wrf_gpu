"""CPU-only identity, boundary, and mutation gates for ADR-036.

ADR-038 re-alignment (2026-09-18): the evidence hook binds the CURRENT product
default entry ``run_forecast_operational_segmented`` (seg=34, env override
``GPUWRF_FORECAST_SEGMENT_STEPS``), so every identity/overhead stub pins THAT
entry — not the retired monolithic ``_run_forecast_operational_jit`` escape
hatch.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import benchmark_m0_default_path as overhead_benchmark  # noqa: E402
import validate_m0_instrumentation_diff as diff_validator  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402


EVIDENCE_ENV = (
    "GPUWRF_M0_EVIDENCE",
    "GPUWRF_M0_EVIDENCE_PATH",
    "GPUWRF_M0_RUN_ID",
    "GPUWRF_M0_SOURCE_SHA256",
    "GPUWRF_M0_CONFIG_SHA256",
    "GPUWRF_M0_INPUT_MANIFEST_SHA256",
    "GPUWRF_M0_DEVICE_UUID",
)


@pytest.fixture(autouse=True)
def clean_evidence_environment(monkeypatch):
    for name in EVIDENCE_ENV:
        monkeypatch.delenv(name, raising=False)


def _enabled(monkeypatch, path: Path) -> None:
    values = {
        "GPUWRF_M0_EVIDENCE": "1",
        "GPUWRF_M0_EVIDENCE_PATH": str(path),
        "GPUWRF_M0_RUN_ID": "m0-hook-test-0001",
        "GPUWRF_M0_SOURCE_SHA256": "1" * 64,
        "GPUWRF_M0_CONFIG_SHA256": "2" * 64,
        "GPUWRF_M0_INPUT_MANIFEST_SHA256": "3" * 64,
        "GPUWRF_M0_DEVICE_UUID": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _patch_outer_wrapper(monkeypatch, events: list[str], result) -> list:
    """Stub the ADR-038 default entry; returns the segment_steps it received."""

    monkeypatch.setattr(om, "_assert_nonzero_initial_mu_total", lambda state: None)
    monkeypatch.setattr(
        om, "_operational_scan_state", lambda state, namelist: ("staged", state)
    )
    segments: list = []

    def dealias(state):
        events.append("dealias")
        return ("unique", state)

    def integrate(state, namelist, hours, *, segment_steps=None):
        events.append("integration")
        segments.append(segment_steps)
        return result

    monkeypatch.setattr(om, "_dealias_pytree_buffers", dealias)
    monkeypatch.setattr(om, "run_forecast_operational_segmented", integrate)
    return segments


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_default_off_is_the_original_direct_call_and_emits_nothing(
    tmp_path, monkeypatch
):
    events: list[str] = []
    result = {"theta": [1.0, 2.0], "dtype": "float32"}
    segments = _patch_outer_wrapper(monkeypatch, events, result)
    monkeypatch.setattr(
        om.jax.profiler,
        "TraceAnnotation",
        lambda *_args, **_kwargs: pytest.fail("default path created a range"),
    )
    monkeypatch.setattr(
        om,
        "_m0_evidence_allocator_stats",
        lambda _result: pytest.fail("default path queried allocator stats"),
    )

    sidecar = tmp_path / "forecast_allocator.json"
    actual = om.run_forecast_operational("state", "namelist", 1.0)

    assert actual is result
    assert events == ["dealias", "integration"]
    # ADR-038: default-off IS the segmented default entry at the product
    # default segment length (seg=34, no env override in this test).
    assert segments == [34]
    assert not sidecar.exists()


def test_enabled_and_default_outputs_have_identical_numerical_digest(
    tmp_path, monkeypatch
):
    result = {"theta": [1.25, -2.5, 8.0], "dtype": "float32"}
    default_events: list[str] = []
    default_segments = _patch_outer_wrapper(monkeypatch, default_events, result)
    default = om.run_forecast_operational("state", "namelist", 1.0)

    events: list[str] = []
    enabled_segments = _patch_outer_wrapper(monkeypatch, events, result)
    _enabled(monkeypatch, tmp_path / "forecast_allocator.json")

    @contextmanager
    def trace(name):
        assert name == "GPUWRF_M0_FORECAST_INTEGRATION"
        events.append("range_enter")
        yield
        events.append("range_exit")

    monkeypatch.setattr(om.jax.profiler, "TraceAnnotation", trace)
    monkeypatch.setattr(
        om.jax, "block_until_ready", lambda value: events.append("synchronize")
    )

    def allocator(value):
        assert value is result
        events.append("allocator_read")
        return {
            "peak_bytes_in_use": 1024,
            "peak_bytes_reserved": 2048,
            "device_platform": "cpu-test-double",
            "device_local_ordinal": 0,
        }

    monkeypatch.setattr(om, "_m0_evidence_allocator_stats", allocator)
    enabled = om.run_forecast_operational("state", "namelist", 1.0)

    assert _digest(enabled) == _digest(default)
    assert enabled is default
    # ADR-038: both paths ran the identical default entry at seg=34, so the
    # digests match by construction, not by threshold.
    assert default_segments == enabled_segments == [34]
    assert events == [
        "dealias",
        "range_enter",
        "integration",
        "synchronize",
        "range_exit",
        "allocator_read",
    ]


def test_enabled_sidecar_is_identity_bound_and_range_enclosed(tmp_path, monkeypatch):
    events: list[str] = []
    result = {"x": 1}
    _patch_outer_wrapper(monkeypatch, events, result)
    sidecar = tmp_path / "forecast_allocator.json"
    _enabled(monkeypatch, sidecar)

    @contextmanager
    def trace(_name):
        yield

    monkeypatch.setattr(om.jax.profiler, "TraceAnnotation", trace)
    monkeypatch.setattr(om.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        om,
        "_m0_evidence_allocator_stats",
        lambda value: {
            "peak_bytes_in_use": 10,
            "peak_bytes_reserved": 12,
            "device_platform": "cpu-test-double",
            "device_local_ordinal": 0,
        },
    )
    om.run_forecast_operational("state", "namelist", 1.0)
    payload = json.loads(sidecar.read_text())

    assert payload["schema"] == "wrf_gpu2.v025.m0.forecast_allocator.v1"
    assert payload["emitter_process_role"] == "forecast_process"
    assert payload["forecast_pid"] == os.getpid()
    assert payload["run_id"] == "m0-hook-test-0001"
    assert payload["source_sha256"] == "1" * 64
    assert payload["config_sha256"] == "2" * 64
    assert payload["input_manifest_sha256"] == "3" * 64
    assert payload["instrumentation"]["enabled"] is True
    assert payload["instrumentation"]["default_when_unset"].startswith(
        "original-direct-call"
    )
    assert payload["measurement_start_ns"] < payload["measurement_end_ns"]
    assert payload["peak_bytes_reserved"] >= payload["peak_bytes_in_use"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"GPUWRF_M0_EVIDENCE": "0"}, "unset"),
        (
            {
                "GPUWRF_M0_EVIDENCE": "1",
                "GPUWRF_M0_EVIDENCE_PATH": "/tmp/not-enough-identity.json",
            },
            "requires",
        ),
    ],
)
def test_invalid_or_partial_opt_in_fails_before_integration(
    tmp_path, monkeypatch, mutation, message
):
    calls: list[str] = []
    _patch_outer_wrapper(monkeypatch, calls, {"x": 1})
    for name, value in mutation.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=message):
        om.run_forecast_operational("state", "namelist", 1.0)
    assert "integration" not in calls


def test_existing_sidecar_fails_before_integration(tmp_path, monkeypatch):
    sidecar = tmp_path / "forecast_allocator.json"
    sidecar.write_text("stale\n")
    _enabled(monkeypatch, sidecar)
    calls: list[str] = []
    _patch_outer_wrapper(monkeypatch, calls, {"x": 1})

    with pytest.raises(RuntimeError, match="overwrite"):
        om.run_forecast_operational("state", "namelist", 1.0)
    assert sidecar.read_text() == "stale\n"
    assert "integration" not in calls


def test_enabled_path_enforces_the_same_segment_guard_as_default_off(
    tmp_path, monkeypatch
):
    # ADR-038: the hook's seg resolution must fail closed exactly like the
    # default-off dispatch -- a non-positive GPUWRF_FORECAST_SEGMENT_STEPS is
    # rejected before the integration range opens, on BOTH paths.
    sidecar = tmp_path / "forecast_allocator.json"
    _enabled(monkeypatch, sidecar)
    monkeypatch.setenv("GPUWRF_FORECAST_SEGMENT_STEPS", "0")
    calls: list[str] = []
    _patch_outer_wrapper(monkeypatch, calls, {"x": 1})

    with pytest.raises(RuntimeError, match="must be positive"):
        om.run_forecast_operational("state", "namelist", 1.0)
    assert "integration" not in calls
    assert not sidecar.exists()


def test_bad_allocator_stats_fail_after_sync_without_publishing(
    tmp_path, monkeypatch
):
    events: list[str] = []
    _patch_outer_wrapper(monkeypatch, events, {"x": 1})
    sidecar = tmp_path / "forecast_allocator.json"
    _enabled(monkeypatch, sidecar)

    @contextmanager
    def trace(_name):
        yield

    monkeypatch.setattr(om.jax.profiler, "TraceAnnotation", trace)
    monkeypatch.setattr(om.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        om,
        "_m0_evidence_allocator_stats",
        lambda value: (_ for _ in ()).throw(RuntimeError("missing allocator peaks")),
    )

    with pytest.raises(RuntimeError, match="missing allocator"):
        om.run_forecast_operational("state", "namelist", 1.0)
    assert "integration" in events
    assert not sidecar.exists()


def test_atomic_publish_never_replaces_a_racing_sidecar(tmp_path, monkeypatch):
    target = tmp_path / "sidecar.json"
    real_link = os.link

    def race(source, destination):
        Path(destination).write_text("winner\n")
        return real_link(source, destination)

    monkeypatch.setattr(om.os, "link", race)
    with pytest.raises(FileExistsError):
        om._m0_evidence_atomic_json(target, {"complete": True})
    assert target.read_text() == "winner\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_allocator_stats_use_result_device_without_backend_enumeration(monkeypatch):
    calls: list[str] = []

    class Device:
        platform = "cpu-test-double"
        id = 7

        def memory_stats(self):
            calls.append("memory_stats")
            return {"peak_bytes_in_use": 12, "peak_bytes_reserved": 16}

    class Leaf:
        device = Device()

    monkeypatch.setattr(om.jax.tree_util, "tree_leaves", lambda result: [Leaf()])
    monkeypatch.setattr(
        om.jax,
        "devices",
        lambda *args, **kwargs: pytest.fail("must not enumerate another backend/device"),
    )
    stats = om._m0_evidence_allocator_stats(object())
    assert calls == ["memory_stats"]
    assert stats["peak_bytes_reserved"] == 16
    assert stats["device_local_ordinal"] == 7


def test_allocator_stats_reject_non_integer_byte_counters(monkeypatch):
    class Device:
        def memory_stats(self):
            return {
                "peak_bytes_in_use": 12.0,
                "peak_bytes_reserved": 16,
            }

    class Leaf:
        device = Device()

    monkeypatch.setattr(om.jax.tree_util, "tree_leaves", lambda result: [Leaf()])
    with pytest.raises(RuntimeError, match="peak_bytes_in_use"):
        om._m0_evidence_allocator_stats(object())


def test_semantic_validator_pins_default_hlo_and_rejects_numerical_mutation():
    path = REPO / "src/gpuwrf/runtime/operational_mode.py"
    base = diff_validator._git_file(
        diff_validator.BASE_PRODUCTION_COMMIT, str(path.relative_to(REPO))
    )
    # This validator is an archived ADR-036 authority gate. Exercise its
    # pinned candidate, rather than asking later model work to match that tree.
    current = diff_validator._git_file(
        diff_validator.CANDIDATE_PRODUCTION_COMMIT, str(path.relative_to(REPO))
    )
    result = diff_validator.validate_sources(base, current)
    assert result["status"] == "PASS"
    assert len(result["default_off_jitted_body_ast_sha256"]) == 64

    mutated = current.replace(
        "steps = _steps_for_hours(hours, float(namelist.dt_s))",
        "steps = 1 + _steps_for_hours(hours, float(namelist.dt_s))",
        1,
    )
    with pytest.raises(diff_validator.DiffValidationError, match="jitted|pre-existing"):
        diff_validator.validate_sources(base, mutated)


def test_semantic_validator_rejects_hot_loop_callback_mutation():
    path = REPO / "src/gpuwrf/runtime/operational_mode.py"
    base = diff_validator._git_file(
        diff_validator.BASE_PRODUCTION_COMMIT, str(path.relative_to(REPO))
    )
    current = diff_validator._git_file(
        diff_validator.CANDIDATE_PRODUCTION_COMMIT, str(path.relative_to(REPO))
    )
    mutated = current.replace(
        'evidence = _m0_evidence_config_from_env()',
        'jax.debug.callback(lambda: None)\n    evidence = _m0_evidence_config_from_env()',
        1,
    )
    assert ast.parse(mutated)
    with pytest.raises(diff_validator.DiffValidationError, match="callback"):
        diff_validator.validate_sources(base, mutated)


@pytest.mark.parametrize(
    "replacement",
    [
        # ADR-038 anchors: the enabled helper's segmented default-entry call.
        "prepared_state, namelist, hours + 1, segment_steps=seg",
        "state, namelist, hours, segment_steps=seg",
    ],
)
def test_semantic_validator_rejects_enabled_numerical_argument_mutation(replacement):
    path = REPO / "src/gpuwrf/runtime/operational_mode.py"
    base = diff_validator._git_file(
        diff_validator.BASE_PRODUCTION_COMMIT, str(path.relative_to(REPO))
    )
    current = diff_validator._git_file(
        diff_validator.CANDIDATE_PRODUCTION_COMMIT, str(path.relative_to(REPO))
    )
    mutated = current.replace(
        "prepared_state, namelist, hours, segment_steps=seg",
        replacement,
        1,
    )
    assert mutated != current
    with pytest.raises(
        diff_validator.DiffValidationError,
        match="integration arguments|range",
    ):
        diff_validator.validate_sources(base, mutated)


def test_default_envelope_meets_once_per_forecast_overhead_gate():
    result = overhead_benchmark.benchmark(iterations=10_000, repeats=3)
    assert result["status"] == "PASS"
    assert (
        result["added_ns_per_forecast_call"]
        <= result["max_added_ns_per_forecast_call"]
    )
    assert (
        result["startup_import"]["added_ms"]
        <= result["startup_import"]["max_added_ms"]
    )


def test_benchmark_cli_resolves_product_source_without_external_pythonpath(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    for name in EVIDENCE_ENV:
        env.pop(name, None)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/v025/benchmark_m0_default_path.py"),
            "--iterations",
            "1000",
            "--repeats",
            "3",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PASS"
