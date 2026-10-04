"""Machine-schema and cross-file mutation tests for M0 evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import m0_vram_sampler as mvs  # noqa: E402
import validate_m0_evidence as validator  # noqa: E402


def _bundle(root: Path) -> tuple[Path, Path, Path]:
    binding = {
        "run_id": "schema-test-run-0001",
        "source_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_manifest_sha256": "3" * 64,
        "device_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    }
    extended = {
        **binding,
        "workload_identity_sha256": "4" * 64,
        "integration_scope_sha256": "5" * 64,
        "event_mix_sha256": "6" * 64,
    }
    identity = root / "identity.json"
    allocator = root / "allocator.json"
    residency = root / "residency.json"
    identity.write_text(json.dumps({"schema": mvs.IDENTITY_SCHEMA, **extended}))
    allocator.write_text(json.dumps({
        "schema": mvs.ALLOCATOR_SCHEMA,
        "emitter_process_role": "forecast_process",
        "instrumentation": {
            "enabled": True,
            "opt_in_environment": "GPUWRF_M0_EVIDENCE=1",
            "default_when_unset": "original-direct-call-no-range-no-sidecar",
            "range_name": "GPUWRF_M0_FORECAST_INTEGRATION",
            "range_scope": "full-forecast-integration",
            "final_synchronization": "jax.block_until_ready(result)",
            "allocator_read": "after-range-on-result-device",
        },
        **binding,
        "forecast_pid": 4242,
        "measurement_start_ns": 1000,
        "measurement_end_ns": 9000,
        "measurement_start_utc": "2026-07-28T00:00:01+00:00",
        "measurement_end_utc": "2026-07-28T00:00:09+00:00",
        "peak_bytes_in_use": 50,
        "peak_bytes_reserved": 70,
        "device_platform": "cuda",
        "device_local_ordinal": 0,
    }))
    residency.write_text(json.dumps({
        "schema": mvs.SAMPLER_SCHEMA,
        "sampler_process_role": "lock_owner_parent",
        "sampler_backend": "external-persistent-nvidia-smi",
        "jax_imported_by_sampler": False,
        "jax_import_check": {
            "method": "sys.modules-prefix-scan-before-baseline-or-stream",
            "status": "PASS",
            "loaded_modules": [],
        },
        "orphan_control": "linux-prctl-pdeathsig-sigterm-plus-parent-stop",
        **extended,
        "forecast_pid": 4242,
        "capture_root_pid": 4000,
        "capture_process_group_id": 4000,
        "baseline_absolute_bytes": 100,
        "peak_absolute_bytes": 180,
        "peak_baseline_subtracted_bytes": 80,
        "sample_cadence_ms": 100,
        "samples": 20,
        "sampling_misses": 0,
        "sampling_quality": {
            "baseline_counted_as_stream_sample": False,
            "denominator": "stream_samples_plus_inferred_or_malformed_misses",
            "observed_miss_fraction": 0.0,
            "maximum_miss_fraction": 0.05,
            "status": "PASS",
        },
        "sampling_limitations": list(mvs.SAMPLING_LIMITATIONS),
        "malformed_samples": 0,
        "observed_process_tree_pids": [4000, 4242],
        "unexpected_competing_contexts": [],
        "measurement_start_ns": 500,
        "measurement_end_ns": 10000,
        "measurement_start_utc": "2026-07-28T00:00:00+00:00",
        "measurement_end_utc": "2026-07-28T00:00:10+00:00",
        "command_sha256": {
            "baseline_total": "7" * 64,
            "baseline_processes": "8" * 64,
            "stream_total": "9" * 64,
            "stream_processes": "a" * 64,
        },
        "product_metric": "baseline-subtracted peak total selected-device residency",
        "allocator_role": "decomposition-cross-check-only",
        "sampler_ram_scaling": "O(unique process IDs); sample values aggregated online",
    }))
    return identity, allocator, residency


def test_complete_bundle_passes_and_hashes_every_artifact(tmp_path):
    identity, allocator, residency = _bundle(tmp_path)
    result = validator.validate_bundle(
        identity_path=identity,
        allocator_path=allocator,
        residency_path=residency,
    )
    assert result["status"] == "PASS"
    assert result["product_peak_resident_bytes"] == 80
    assert set(result["artifact_sha256"]) == {
        "schema", "identity", "allocator", "residency"
    }
    assert all(len(value) == 64 for value in result["artifact_sha256"].values())


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    [
        ("allocator", lambda value: value.update({"extra": 1}), "unrecognised"),
        ("allocator", lambda value: value.update({"source_sha256": "9" * 64}), "mismatch"),
        (
            "allocator",
            lambda value: value["instrumentation"].update({"range_scope": "one-step"}),
            "provenance",
        ),
        (
            "allocator",
            lambda value: value.update({"peak_bytes_in_use": "50"}),
            "exact integers",
        ),
        (
            "residency",
            lambda value: value.update({"peak_baseline_subtracted_bytes": 79}),
            "arithmetic",
        ),
        (
            "residency",
            lambda value: value.update(
                {"measurement_start_utc": "2026-07-28T01:00:00+01:00"}
            ),
            "not UTC",
        ),
        (
            "residency",
            lambda value: value.update({"unexpected_competing_contexts": [9999]}),
            "competing",
        ),
        (
            "residency",
            lambda value: value.update({"observed_process_tree_pids": [4000]}),
            "process-tree",
        ),
        (
            "residency",
            lambda value: value["command_sha256"].pop("baseline_total"),
            "command hashes",
        ),
    ],
)
def test_schema_mutations_fail_closed(tmp_path, target, mutation, message):
    identity, allocator, residency = _bundle(tmp_path)
    path = allocator if target == "allocator" else residency
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value))
    with pytest.raises(validator.EvidenceSchemaError, match=message):
        validator.validate_bundle(
            identity_path=identity,
            allocator_path=allocator,
            residency_path=residency,
        )


def test_machine_schema_has_exact_frozen_object_ids():
    schema = json.loads(
        (REPO / "scripts/v025/m0_evidence_v1.schema.json").read_text()
    )
    text = json.dumps(schema, sort_keys=True)
    assert schema["$id"] == "wrf_gpu2.v025.m0.evidence_bundle.v1"
    for object_id in (
        mvs.IDENTITY_SCHEMA,
        mvs.ALLOCATOR_SCHEMA,
        mvs.SAMPLER_SCHEMA,
    ):
        assert object_id in text
