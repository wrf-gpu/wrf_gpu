#!/usr/bin/env python3
"""Validate one identity-bound ADR-036 allocator/residency sidecar bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import baseline_census as bc  # noqa: E402
import m0_vram_sampler as mvs  # noqa: E402


DEFAULT_SCHEMA = SCRIPT_DIR / "m0_evidence_v1.schema.json"


class EvidenceSchemaError(RuntimeError):
    """A bundle violates the frozen machine schema or cross-file binding."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceSchemaError(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceSchemaError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceSchemaError(f"JSON artifact is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_keys(
    payload: dict[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    label: str,
) -> None:
    missing = required - set(payload)
    extra = set(payload) - allowed
    if missing:
        raise EvidenceSchemaError(f"{label} missing fields: {sorted(missing)}")
    if extra:
        raise EvidenceSchemaError(f"{label} has unrecognised fields: {sorted(extra)}")


def _validate_total(
    total: dict[str, Any],
    identity: mvs.EvidenceIdentity,
    *,
    forecast_pid: int,
) -> None:
    binding = set(identity.binding())
    required = {
        "schema",
        "sampler_process_role",
        "sampler_backend",
        "jax_imported_by_sampler",
        "jax_import_check",
        "orphan_control",
        *binding,
        "forecast_pid",
        "capture_root_pid",
        "capture_process_group_id",
        "baseline_absolute_bytes",
        "peak_absolute_bytes",
        "peak_baseline_subtracted_bytes",
        "sample_cadence_ms",
        "samples",
        "sampling_misses",
        "sampling_quality",
        "sampling_limitations",
        "malformed_samples",
        "observed_process_tree_pids",
        "unexpected_competing_contexts",
        "measurement_start_ns",
        "measurement_end_ns",
        "measurement_start_utc",
        "measurement_end_utc",
        "command_sha256",
        "product_metric",
        "allocator_role",
        "sampler_ram_scaling",
    }
    _require_keys(total, required=required, allowed=required, label="residency sidecar")
    if total["schema"] != mvs.SAMPLER_SCHEMA:
        raise EvidenceSchemaError("residency sidecar schema mismatch")
    for key, expected in identity.binding().items():
        if total[key] != expected:
            raise EvidenceSchemaError(f"residency sidecar identity mismatch: {key}")
    constants = {
        "sampler_process_role": "lock_owner_parent",
        "sampler_backend": "external-persistent-nvidia-smi",
        "jax_imported_by_sampler": False,
        "orphan_control": "linux-prctl-pdeathsig-sigterm-plus-parent-stop",
        "product_metric": "baseline-subtracted peak total selected-device residency",
        "allocator_role": "decomposition-cross-check-only",
        "sampler_ram_scaling": (
            "O(unique process IDs); sample values aggregated online"
        ),
    }
    for key, expected in constants.items():
        if total[key] != expected:
            raise EvidenceSchemaError(f"residency sidecar constant mismatch: {key}")
    if total["forecast_pid"] != forecast_pid:
        raise EvidenceSchemaError("residency/allocator forecast PID mismatch")

    integer_fields = (
        "forecast_pid",
        "capture_root_pid",
        "capture_process_group_id",
        "baseline_absolute_bytes",
        "peak_absolute_bytes",
        "peak_baseline_subtracted_bytes",
        "sample_cadence_ms",
        "samples",
        "sampling_misses",
        "malformed_samples",
        "measurement_start_ns",
        "measurement_end_ns",
    )
    if any(
        isinstance(total[field], bool) or not isinstance(total[field], int)
        for field in integer_fields
    ):
        raise EvidenceSchemaError("residency sidecar integer field has wrong type")
    if any(total[field] < 0 for field in integer_fields):
        raise EvidenceSchemaError("residency sidecar has a negative integer field")
    if (
        total["forecast_pid"] <= 0
        or total["capture_root_pid"] <= 0
        or total["capture_process_group_id"] <= 0
        or total["sample_cadence_ms"] <= 0
        or total["samples"] <= 0
        or total["measurement_start_ns"] <= 0
        or total["measurement_end_ns"] <= total["measurement_start_ns"]
    ):
        raise EvidenceSchemaError("residency sidecar integer bounds are invalid")
    if not mvs._sampling_coverage_is_acceptable(
        total["samples"], total["sampling_misses"]
    ):
        raise EvidenceSchemaError("residency sidecar sampling coverage is invalid")
    if total["malformed_samples"] > total["sampling_misses"]:
        raise EvidenceSchemaError("malformed sample count exceeds total misses")
    if total["jax_import_check"] != {
        "method": "sys.modules-prefix-scan-before-baseline-or-stream",
        "status": "PASS",
        "loaded_modules": [],
    }:
        raise EvidenceSchemaError("sampler JAX import check is invalid")
    if total["sampling_limitations"] != list(mvs.SAMPLING_LIMITATIONS):
        raise EvidenceSchemaError("sampler limitations are missing or changed")
    quality = total["sampling_quality"]
    expected_fraction = mvs._sampling_miss_fraction(
        total["samples"], total["sampling_misses"]
    )
    if (
        not isinstance(quality, dict)
        or set(quality)
        != {
            "baseline_counted_as_stream_sample",
            "denominator",
            "observed_miss_fraction",
            "maximum_miss_fraction",
            "status",
        }
        or quality["baseline_counted_as_stream_sample"] is not False
        or quality["denominator"]
        != "stream_samples_plus_inferred_or_malformed_misses"
        or isinstance(quality["observed_miss_fraction"], bool)
        or not isinstance(quality["observed_miss_fraction"], (int, float))
        or not math.isclose(
            float(quality["observed_miss_fraction"]),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or quality["maximum_miss_fraction"] != mvs.MAX_SAMPLING_MISS_FRACTION
        or quality["status"] != "PASS"
    ):
        raise EvidenceSchemaError("sampler sampling-quality record is invalid")
    if total["peak_absolute_bytes"] < total["baseline_absolute_bytes"]:
        raise EvidenceSchemaError("absolute peak is below baseline")
    if total["peak_baseline_subtracted_bytes"] != (
        total["peak_absolute_bytes"] - total["baseline_absolute_bytes"]
    ):
        raise EvidenceSchemaError("baseline-subtracted peak arithmetic mismatch")

    observed = total["observed_process_tree_pids"]
    if (
        not isinstance(observed, list)
        or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in observed)
        or len(observed) != len(set(observed))
        or forecast_pid not in observed
    ):
        raise EvidenceSchemaError("observed process-tree PID set is invalid")
    if total["unexpected_competing_contexts"] != []:
        raise EvidenceSchemaError("unexpected competing contexts are present")
    command_hashes = total["command_sha256"]
    required_command_hashes = {
        "baseline_total",
        "baseline_processes",
        "stream_total",
        "stream_processes",
    }
    if (
        not isinstance(command_hashes, dict)
        or set(command_hashes) != required_command_hashes
        or any(
            not isinstance(value, str) or not mvs.SHA256_PATTERN.fullmatch(value)
            for value in command_hashes.values()
        )
    ):
        raise EvidenceSchemaError("sampler command hashes are invalid")
    utc_values: dict[str, datetime] = {}
    for field in ("measurement_start_utc", "measurement_end_utc"):
        try:
            stamp = datetime.fromisoformat(total[field])
        except (TypeError, ValueError) as exc:
            raise EvidenceSchemaError(f"invalid residency timestamp: {field}") from exc
        if stamp.tzinfo is None:
            raise EvidenceSchemaError(f"residency timestamp lacks timezone: {field}")
        if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            raise EvidenceSchemaError(f"residency timestamp is not UTC: {field}")
        utc_values[field] = stamp
    if utc_values["measurement_end_utc"] <= utc_values["measurement_start_utc"]:
        raise EvidenceSchemaError("residency UTC interval is invalid")


def validate_bundle(
    *,
    identity_path: Path,
    allocator_path: Path,
    residency_path: Path,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    schema = _load_json(schema_path)
    if schema.get("$id") != "wrf_gpu2.v025.m0.evidence_bundle.v1":
        raise EvidenceSchemaError("machine schema ID mismatch")
    schema_text = json.dumps(schema, sort_keys=True)
    for required_schema in (
        mvs.IDENTITY_SCHEMA,
        mvs.ALLOCATOR_SCHEMA,
        mvs.SAMPLER_SCHEMA,
    ):
        if required_schema not in schema_text:
            raise EvidenceSchemaError(
                f"machine schema omits required object: {required_schema}"
            )

    identity = mvs.load_identity(identity_path)
    allocator = _load_json(allocator_path)
    try:
        mvs.validate_allocator_sidecar(allocator, identity)
    except mvs.ResidencyEvidenceError as exc:
        raise EvidenceSchemaError(str(exc)) from exc
    _require_keys(
        allocator,
        required={
            "schema",
            "emitter_process_role",
            "instrumentation",
            "run_id",
            "forecast_pid",
            "source_sha256",
            "config_sha256",
            "input_manifest_sha256",
            "device_uuid",
            "measurement_start_ns",
            "measurement_end_ns",
            "measurement_start_utc",
            "measurement_end_utc",
            "peak_bytes_in_use",
            "peak_bytes_reserved",
            "device_platform",
            "device_local_ordinal",
        },
        allowed={
            "schema",
            "emitter_process_role",
            "instrumentation",
            "run_id",
            "forecast_pid",
            "source_sha256",
            "config_sha256",
            "input_manifest_sha256",
            "device_uuid",
            "measurement_start_ns",
            "measurement_end_ns",
            "measurement_start_utc",
            "measurement_end_utc",
            "peak_bytes_in_use",
            "peak_bytes_reserved",
            "device_platform",
            "device_local_ordinal",
        },
        label="allocator sidecar",
    )
    total = _load_json(residency_path)
    _validate_total(total, identity, forecast_pid=int(allocator["forecast_pid"]))
    allocator_start_utc = datetime.fromisoformat(allocator["measurement_start_utc"])
    allocator_end_utc = datetime.fromisoformat(allocator["measurement_end_utc"])
    total_start_utc = datetime.fromisoformat(total["measurement_start_utc"])
    total_end_utc = datetime.fromisoformat(total["measurement_end_utc"])
    if not (
        total_start_utc
        <= allocator_start_utc
        < allocator_end_utc
        <= total_end_utc
    ):
        raise EvidenceSchemaError(
            "residency UTC interval does not enclose allocator UTC interval"
        )

    gate = bc.vram_gate(
        {
            "schema": "wrf_gpu2.v025.m0.vram_evidence.v1",
            "forecast_allocator": allocator,
            "lock_owner_total_residency": total,
        }
    )
    if gate.get("status") != "OK":
        raise EvidenceSchemaError(
            f"baseline census rejected the bundle: {gate.get('reason')}"
        )
    return {
        "schema": "wrf_gpu2.v025.m0.evidence_bundle_validation.v1",
        "status": "PASS",
        "run_id": identity.run_id,
        "forecast_pid": allocator["forecast_pid"],
        "product_peak_resident_bytes": gate["peak_resident_bytes"],
        "allocator_peak_bytes_in_use": gate["allocator_peak_bytes_in_use"],
        "allocator_peak_bytes_reserved": gate["allocator_peak_bytes_reserved"],
        "artifact_sha256": {
            "schema": _sha256(schema_path),
            "identity": _sha256(identity_path),
            "allocator": _sha256(allocator_path),
            "residency": _sha256(residency_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--allocator", required=True, type=Path)
    parser.add_argument("--residency", required=True, type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_bundle(
            identity_path=args.identity,
            allocator_path=args.allocator,
            residency_path=args.residency,
            schema_path=args.schema,
        )
    except (EvidenceSchemaError, mvs.ResidencyEvidenceError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, indent=2))
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
