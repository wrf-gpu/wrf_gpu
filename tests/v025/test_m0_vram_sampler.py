"""CPU fakes for the external, non-JAX M0 residency sampler."""
from __future__ import annotations

import ast
import io
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import m0_vram_sampler as sampler  # noqa: E402


RUN_ID = "m0-sampler-test-0001"


@pytest.fixture(autouse=True)
def isolate_lock_owner_from_test_process_jax(monkeypatch):
    """The real lock owner is a JAX-free process; pytest itself is not."""

    monkeypatch.setattr(sampler, "_loaded_jax_modules", lambda: ())


def _identity() -> sampler.EvidenceIdentity:
    return sampler.EvidenceIdentity(
        run_id=RUN_ID,
        source_sha256="1" * 64,
        config_sha256="2" * 64,
        input_manifest_sha256="3" * 64,
        device_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        workload_identity_sha256="4" * 64,
        integration_scope_sha256="5" * 64,
        event_mix_sha256="6" * 64,
    )


def _identity_file(path: Path, **updates) -> Path:
    payload = {"schema": sampler.IDENTITY_SCHEMA, **_identity().binding(), **updates}
    path.write_text(json.dumps(payload))
    return path


def _allocator(path: Path, *, identity=None, pid=4242, **updates) -> dict:
    identity = identity or _identity()
    now = time.monotonic_ns()
    payload = {
        "schema": sampler.ALLOCATOR_SCHEMA,
        "emitter_process_role": "forecast_process",
        "instrumentation": {
            "enabled": True,
            "opt_in_environment": "GPUWRF_M0_EVIDENCE=1",
            "range_name": "GPUWRF_M0_FORECAST_INTEGRATION",
            "default_when_unset": "original-direct-call-no-range-no-sidecar",
            "range_scope": "full-forecast-integration",
            "final_synchronization": "jax.block_until_ready(result)",
            "allocator_read": "after-range-on-result-device",
        },
        "run_id": identity.run_id,
        "forecast_pid": pid,
        "source_sha256": identity.source_sha256,
        "config_sha256": identity.config_sha256,
        "input_manifest_sha256": identity.input_manifest_sha256,
        "device_uuid": identity.device_uuid,
        "measurement_start_ns": now,
        "measurement_end_ns": now + 1,
        "measurement_start_utc": "2026-07-28T00:00:00+00:00",
        "measurement_end_utc": "2026-07-28T00:00:01+00:00",
        "peak_bytes_in_use": 100,
        "peak_bytes_reserved": 120,
        "device_platform": "cpu-test-double",
        "device_local_ordinal": 0,
        **updates,
    }
    path.write_text(json.dumps(payload))
    return payload


def _exact_allocator(path: Path, *, identity=None, pid=4242, **updates) -> dict:
    identity = identity or _identity()
    now = time.monotonic_ns()
    payload = {
        "schema": sampler.EXACT_ALLOCATOR_SCHEMA,
        "run_id": identity.run_id,
        "pid": pid,
        "range_name": "GPUWRF_M0_FORECAST_INTEGRATION",
        "measurement_start_ns": now,
        "measurement_end_ns": now + 1,
        "result_sha256": "7" * 64,
        "peak_bytes_in_use": 100,
        "peak_bytes_reserved": 120,
        "device_platform": "gpu",
        "device_local_ordinal": 0,
        **updates,
    }
    path.write_text(json.dumps(payload))
    return payload


class FakeBackend:
    commands = {
        "stream_total": ["fake-telemetry", "--persistent"],
        "stream_processes": ["fake-process-telemetry", "--persistent"],
    }

    def __init__(
        self,
        *,
        baseline=100,
        baseline_processes=None,
        totals=(110, 180, 150),
        process_rows=((4242, 80),),
    ):
        self.baseline_value = baseline
        self.baseline_processes = baseline_processes or []
        self.totals = totals
        self.process_rows = process_rows
        self.stopped = False

    def baseline(self):
        return self.baseline_value, list(self.baseline_processes)

    def start(self, total_callback, process_callback):
        for value in self.totals:
            total_callback(time.monotonic_ns(), value)
        for pid, value in self.process_rows:
            process_callback(time.monotonic_ns(), pid, value)

    def stop(self):
        self.stopped = True


def _instance(tmp_path: Path, backend) -> sampler.LockOwnerResidencySampler:
    return sampler.LockOwnerResidencySampler(
        identity=_identity(),
        allocator_sidecar=tmp_path / "forecast_allocator.json",
        output_sidecar=tmp_path / "lock_owner_total_residency.json",
        stderr_path=tmp_path / "sampler.log",
        cadence_ms=100,
        backend=backend,
    )


def test_identity_loader_requires_every_hash_and_exact_run_id(tmp_path):
    path = _identity_file(tmp_path / "identity.json")
    loaded = sampler.load_identity(path, expected_run_id=RUN_ID)
    assert loaded == _identity()

    _identity_file(path, source_sha256="short")
    with pytest.raises(sampler.ResidencyEvidenceError, match="source_sha256"):
        sampler.load_identity(path, expected_run_id=RUN_ID)

    _identity_file(path, run_id="different-run-0001")
    with pytest.raises(sampler.ResidencyEvidenceError, match="does not match"):
        sampler.load_identity(path, expected_run_id=RUN_ID)

    _identity_file(path, unrecognised="value")
    with pytest.raises(sampler.ResidencyEvidenceError, match="unrecognised"):
        sampler.load_identity(path)


def test_identity_is_derived_from_exact_source_config_and_input_bytes(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "model.py").write_text("answer = 1\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "model.pyc").write_bytes(b"ignored-runtime-bytecode")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "namelist.input").write_text("&time_control\n/\n")
    (run_dir / "wrfout_d01_0000").write_bytes(b"initial")
    (run_dir / "wrfout_d01_0100").write_bytes(b"final")

    identity = sampler.derive_identity(
        run_id=RUN_ID,
        device_uuid=_identity().device_uuid,
        source_root=source,
        run_dir=run_dir,
    )
    sampler.validate_identity_against_workload(
        identity, source_root=source, run_dir=run_dir
    )

    (source / "model.py").write_text("answer = 2\n")
    with pytest.raises(sampler.ResidencyEvidenceError, match="source_sha256|workload"):
        sampler.validate_identity_against_workload(
            identity, source_root=source, run_dir=run_dir
        )

    (source / "model.py").write_text("answer = 1\n")
    (run_dir / "wrfout_d01_0100").write_bytes(b"mutated")
    with pytest.raises(sampler.ResidencyEvidenceError, match="input_manifest"):
        sampler.validate_identity_against_workload(
            identity, source_root=source, run_dir=run_dir
        )


def test_hook_environment_is_explicit_and_refuses_stale_target(tmp_path):
    target = tmp_path / "forecast_allocator.json"
    env = sampler.hook_environment(_identity(), allocator_sidecar=target)
    assert env["GPUWRF_M0_EVIDENCE"] == "1"
    assert env["GPUWRF_M0_RUN_ID"] == RUN_ID
    assert env["GPUWRF_M0_DEVICE_UUID"].startswith("GPU-")

    target.write_text("stale")
    with pytest.raises(sampler.ResidencyEvidenceError, match="already exists"):
        sampler.hook_environment(_identity(), allocator_sidecar=target)


def test_sampler_binds_allocator_and_emits_total_product_metric(tmp_path):
    backend = FakeBackend()
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _allocator(tmp_path / "forecast_allocator.json")
    payload = instance.finish()

    assert backend.stopped is True
    assert payload["schema"] == sampler.SAMPLER_SCHEMA
    assert payload["sampler_process_role"] == "lock_owner_parent"
    assert payload["jax_imported_by_sampler"] is False
    assert payload["jax_import_check"]["status"] == "PASS"
    assert payload["orphan_control"].startswith("linux-prctl")
    assert payload["forecast_pid"] == 4242
    assert payload["baseline_absolute_bytes"] == 100
    assert payload["peak_absolute_bytes"] == 180
    assert payload["peak_baseline_subtracted_bytes"] == 80
    assert payload["observed_process_tree_pids"] == [4242]
    assert payload["unexpected_competing_contexts"] == []
    assert payload["samples"] == 3
    assert payload["sampling_quality"]["baseline_counted_as_stream_sample"] is False
    assert payload["sampling_quality"]["observed_miss_fraction"] <= 0.05
    assert payload["sampling_limitations"]
    assert json.loads(
        (tmp_path / "lock_owner_total_residency.json").read_text()
    ) == payload


def test_sampler_accepts_exact_boundary_allocator_without_enabling_product_hook(
    tmp_path,
):
    backend = FakeBackend()
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _exact_allocator(tmp_path / "forecast_allocator.json")
    payload = instance.finish()

    assert payload["forecast_pid"] == 4242
    assert payload["peak_baseline_subtracted_bytes"] == 80
    assert payload["unexpected_competing_contexts"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"run_id": "different-run-0001"}, "run_id"),
        ({"range_name": "WRONG"}, "range identity"),
        ({"result_sha256": "short"}, "result digest"),
        ({"measurement_end_ns": 1}, "numeric fields are invalid"),
        ({"peak_bytes_in_use": "100"}, "exact integers"),
        ({"device_platform": "cpu"}, "GPU platform"),
        ({"unrecognised": True}, "unrecognised"),
    ],
)
def test_exact_boundary_allocator_mutations_fail_closed(
    tmp_path, mutation, message
):
    instance = _instance(tmp_path, FakeBackend())
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _exact_allocator(tmp_path / "forecast_allocator.json", **mutation)
    with pytest.raises(sampler.ResidencyEvidenceError, match=message):
        instance.finish()


def test_sampler_aggregates_long_series_in_constant_ram(tmp_path):
    backend = FakeBackend(totals=tuple(100 + index % 50 for index in range(20_000)))
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _allocator(tmp_path / "forecast_allocator.json")
    payload = instance.finish()
    assert payload["samples"] == 20_000
    assert payload["peak_absolute_bytes"] == 149
    assert not hasattr(instance, "_totals")
    assert payload["sampler_ram_scaling"].startswith("O(unique process IDs)")


def test_repeated_process_rows_are_classified_once_per_unique_pid(
    tmp_path, monkeypatch
):
    classifications: list[int] = []
    monkeypatch.setattr(
        sampler,
        "_pid_in_group_or_tree",
        lambda pid, **kwargs: classifications.append(pid) is None,
    )
    backend = FakeBackend(process_rows=((4242, 80),) * 20_000)
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    assert classifications == [4242]


def test_proc_parent_parser_recognises_real_linux_ppid():
    assert sampler._pid_in_group_or_tree(
        os.getpid(),
        root_pid=os.getppid(),
        process_group_id=1 << 29,
    )


def test_prelaunch_compute_context_fails_before_stream_start(tmp_path):
    backend = FakeBackend(baseline_processes=[(9999, 10)])
    instance = _instance(tmp_path, backend)
    with pytest.raises(sampler.ResidencyEvidenceError, match="before forecast launch"):
        instance.start()
    assert backend.stopped is False
    assert not (tmp_path / "lock_owner_total_residency.json").exists()


def test_loaded_jax_is_rejected_before_baseline_or_stream(tmp_path, monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(sampler, "_loaded_jax_modules", lambda: ("jax", "jaxlib"))
    instance = _instance(tmp_path, backend)

    with pytest.raises(sampler.ResidencyEvidenceError, match="already loaded"):
        instance.start()

    assert instance._started_ns is None
    assert backend.stopped is False


def test_double_start_and_double_attach_are_rejected(tmp_path):
    instance = _instance(tmp_path, FakeBackend())
    instance.start()
    with pytest.raises(sampler.ResidencyEvidenceError, match="started twice"):
        instance.start()

    instance.attach(root_pid=4000, process_group_id=4000)
    with pytest.raises(sampler.ResidencyEvidenceError, match="attached twice"):
        instance.attach(root_pid=4001, process_group_id=4001)
    instance.abort()


def test_reader_error_is_rejected_at_finish(tmp_path):
    backend = FakeBackend()
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _allocator(tmp_path / "forecast_allocator.json")
    backend._reader_errors = ["injected stream reader failure"]

    with pytest.raises(sampler.ResidencyEvidenceError, match="reader failed"):
        instance.finish()


def test_sampler_rejects_non_enclosed_monotonic_allocator_interval(tmp_path):
    instance = _instance(tmp_path, FakeBackend())
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    assert instance._started_ns is not None
    _allocator(
        tmp_path / "forecast_allocator.json",
        measurement_start_ns=instance._started_ns - 1,
        measurement_end_ns=instance._started_ns + 1,
    )

    with pytest.raises(sampler.ResidencyEvidenceError, match="does not enclose"):
        instance.finish()


def test_finish_accepts_small_miss_rate_and_rejects_above_five_percent(tmp_path):
    accepted_backend = FakeBackend(totals=tuple(range(100, 200)))
    accepted = _instance(tmp_path, accepted_backend)
    accepted.start()
    accepted.attach(root_pid=4000, process_group_id=4000)
    accepted._stream_started_ns = time.monotonic_ns() - 10_200_000_000
    _allocator(tmp_path / "forecast_allocator.json")
    payload = accepted.finish()
    assert 0.0 < payload["sampling_quality"]["observed_miss_fraction"] <= 0.05

    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    rejected = _instance(
        rejected_root,
        FakeBackend(totals=tuple(range(100, 200))),
    )
    rejected.start()
    rejected.attach(root_pid=4000, process_group_id=4000)
    rejected._stream_started_ns = time.monotonic_ns() - 20_000_000_000
    _allocator(rejected_root / "forecast_allocator.json")
    with pytest.raises(sampler.ResidencyEvidenceError, match="fraction=.*limit"):
        rejected.finish()


def _comparison_mutant(function, old: str, new: str):
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1
    namespace = {
        "MAX_SAMPLING_MISS_FRACTION": sampler.MAX_SAMPLING_MISS_FRACTION,
        "_sampling_miss_fraction": sampler._sampling_miss_fraction,
    }
    exec(source.replace(old, new, 1), namespace)
    return namespace[function.__name__]


def test_inverted_sampling_coverage_comparison_is_killed():
    mutant = _comparison_mutant(
        sampler._sampling_coverage_is_acceptable,
        "<= MAX_SAMPLING_MISS_FRACTION",
        "> MAX_SAMPLING_MISS_FRACTION",
    )
    assert sampler._sampling_coverage_is_acceptable(95, 5) is True
    assert sampler._sampling_coverage_is_acceptable(94, 6) is False
    assert mutant(95, 5) is False
    assert mutant(94, 6) is True


def test_inverted_monotonic_enclosure_comparison_is_killed():
    mutant = _comparison_mutant(
        sampler._monotonic_interval_encloses,
        "outer_start_ns <= inner_start_ns < inner_end_ns <= outer_end_ns",
        "outer_start_ns >= inner_start_ns > inner_end_ns >= outer_end_ns",
    )
    assert sampler._monotonic_interval_encloses(1, 2, 3, 4) is True
    assert sampler._monotonic_interval_encloses(2, 1, 3, 4) is False
    assert mutant(1, 2, 3, 4) is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_sha256": "9" * 64}, "identity mismatch"),
        ({"peak_bytes_reserved": 1}, "numeric fields are invalid"),
        ({"peak_bytes_in_use": "100"}, "exact integers"),
        ({"unrecognised": "value"}, "unrecognised"),
        (
            {"measurement_start_utc": "2026-07-28T01:00:00+01:00"},
            "not UTC",
        ),
        ({"schema": "wrong"}, "schema"),
    ],
)
def test_allocator_mutations_fail_closed(tmp_path, mutation, message):
    backend = FakeBackend()
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _allocator(tmp_path / "forecast_allocator.json", **mutation)

    with pytest.raises(sampler.ResidencyEvidenceError, match=message):
        instance.finish()
    assert not (tmp_path / "lock_owner_total_residency.json").exists()


def test_missing_allocator_or_unobserved_forecast_pid_fails_closed(tmp_path):
    backend = FakeBackend(process_rows=())
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    with pytest.raises(sampler.ResidencyEvidenceError, match="missing"):
        instance.finish()

    other = tmp_path / "other"
    other.mkdir()
    instance = _instance(other, FakeBackend(process_rows=((9999, 10),)))
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _allocator(other / "forecast_allocator.json")
    with pytest.raises(sampler.ResidencyEvidenceError, match="never observed"):
        instance.finish()


def test_unexpected_competing_context_fails_closed(tmp_path, monkeypatch):
    backend = FakeBackend(process_rows=((4242, 80), (9999, 40)))
    instance = _instance(tmp_path, backend)
    instance.start()
    instance.attach(root_pid=4000, process_group_id=4000)
    _allocator(tmp_path / "forecast_allocator.json")
    monkeypatch.setattr(
        sampler,
        "_pid_in_group_or_tree",
        lambda pid, **kwargs: pid == 4242,
    )
    with pytest.raises(sampler.ResidencyEvidenceError, match="competing"):
        instance.finish()
    assert not (tmp_path / "lock_owner_total_residency.json").exists()


def test_sampler_refuses_to_replace_prior_evidence(tmp_path):
    target = tmp_path / "lock_owner_total_residency.json"
    target.write_text("prior\n")
    with pytest.raises(sampler.ResidencyEvidenceError, match="already exists"):
        sampler.LockOwnerResidencySampler(
            identity=_identity(),
            allocator_sidecar=tmp_path / "forecast_allocator.json",
            output_sidecar=target,
            stderr_path=tmp_path / "sampler.log",
            backend=FakeBackend(),
        )
    assert target.read_text() == "prior\n"


def test_sampler_module_has_no_jax_import_or_device_initialisation():
    source = (REPO / "scripts/v025/m0_vram_sampler.py").read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name == "jax" or name.startswith("jax.") for name in imported)
    assert "import pynvml" not in source
    assert "jax.devices" not in source
    assert "preexec_fn=_terminate_if_parent_dies" in source


def test_cpu_manifest_builder_contains_no_device_query():
    source = (REPO / "scripts/v025/build_manifests.py").read_text()
    assert "nvidia-smi" not in source
    assert '"nvidia_smi_query": None' in source


def test_nvidia_smi_parsers_bind_the_selected_physical_uuid():
    uuid = _identity().device_uuid
    assert sampler._parse_global_row(f"{uuid}, 12", uuid) == 12 << 20
    assert sampler._parse_compute_row(f"42, {uuid}, 3", uuid) == (42, 3 << 20)
    with pytest.raises(sampler.ResidencyEvidenceError, match="unexpected"):
        sampler._parse_global_row("GPU-other, 12", uuid)


def test_no_compute_process_message_is_an_empty_baseline_not_a_parse_error(
    tmp_path, monkeypatch
):
    backend = sampler.NvidiaSmiStreamingBackend(
        _identity().device_uuid,
        cadence_ms=100,
        stderr_path=tmp_path / "stderr.log",
    )
    monkeypatch.setattr(
        sampler.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "No running processes found\n", ""
        ),
    )
    assert backend._run_once(["fake"], allow_empty=True) == []


def test_persistent_stream_eof_before_stop_is_fail_closed(tmp_path):
    backend = sampler.NvidiaSmiStreamingBackend(
        _identity().device_uuid,
        cadence_ms=100,
        stderr_path=tmp_path / "stderr.log",
    )

    class Process:
        stdout = io.StringIO(f"{_identity().device_uuid}, 12\n")

    backend._read_total(Process(), lambda *_args: None)
    assert "ended before sampler stop" in "; ".join(backend._reader_errors)
