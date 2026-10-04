#!/usr/bin/env python3
"""Fail-closed pre-spend declaration for the Review-10 M0 fallback.

The declaration answers one narrow question: can the already-frozen W2 capture
mechanically expose the quantities named in the fallback contract?  It does not
run a device, inspect a receipt, acquire a lock, or claim that lowering proves a
device compile/link.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA = "wrf_gpu2.v025.m0.review10_fallback_capability.v1"

AVAILABLE_FIELDS = (
    "exact_kernels_per_step_denominator",
    "integration_clipped_gpu_busy",
    "event_level_family_join",
    "kernel_duration_distribution",
    "inter_kernel_gap_distribution",
    "launch_latency_distribution",
    "transfer_audit",
    "profiler_perturbation",
    "peak_vram",
)
MISSING_FIELDS = (
    "achieved_dram_gb_per_s",
    "occupancy_and_stall_counters",
    "run_physics_false",
)
FALLBACK_INVARIANT_NAMES = (
    "raw_capture_hash_bound_before_postlock_analysis",
    "postcapture_analysis_after_lock_release",
    "postcapture_failure_preserves_raw_capture_and_readonly_sqlite",
    "insufficient_w2_w3_time_stops_before_w2",
    "no_extra_device_arm_after_w2_hash_and_w3_stop",
)
HARD_GATE_MUTATIONS = {
    "R1": (
        "unset_wrf_roots",
        "split_wrf_roots",
        "mutated_authority_content_hash",
        "mutated_authoritative_source_file",
        "product_import_before_authority",
        "non_cpu_preflight_platform",
        "visible_cuda_device_in_preflight",
        "lock_environment_leaks_into_preflight",
        "native_loader_not_executed_once",
        "exact_lower_not_executed_once",
        "compile_or_device_call_during_preflight",
        "authority_drift_before_spend",
        "authority_drift_at_stage_entry",
    ),
    "R2": (
        "one_character_token",
        "strict_prefix_token",
        "strict_suffix_token",
        "token_only_inside_cmd",
        "wrong_holder_label",
        "missing_exported_label",
        "mutated_registry",
        "stale_spend_ledger",
    ),
    "C0": (
        "missing_lowered_trip_count",
        "lowered_configured_count_mismatch",
        "wrong_denominator_method",
        "stablehlo_hash_mismatch",
        "duplicate_production_integration_range",
        "available_set_drift",
        "missing_set_drift",
        "integration_clip_wrong_emitter",
        "raw_capture_hash_binding_removed",
        "postlock_order_removed",
        "postcapture_readonly_preservation_removed",
        "w2_w3_reservation_removed",
        "extra_device_arm_after_w3",
    ),
}
EXPECTED_BOUNDARY_OBSERVATIONS = {
    "native_loader_calls": 1,
    "fast_argument_builder_calls": 1,
    "wrapper_preparation_calls": 1,
    "exact_lower_calls": 1,
    "compile_calls": 0,
    "device_invocations": 0,
    "receipt_reads": 0,
    "ledger_reads": 0,
    "ledger_writes": 0,
    "lock_checks": 0,
    "wrapper_calls": 0,
}


class CapabilityRefusal(RuntimeError):
    """The capture cannot support the declared fallback without guessing."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(path: Path, required_tokens: tuple[str, ...]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    text = resolved.read_text(encoding="utf-8")
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise CapabilityRefusal(
            f"capture source {resolved} lacks required tokens: {missing}"
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "required_tokens": list(required_tokens),
        "all_required_tokens_present": True,
    }


def observe_fallback_invariants() -> dict[str, dict[str, Any]]:
    """Bind the five frozen no-extra-spend/no-data-loss invariants to source."""

    import m0_core_session_protocol as core

    expected_order = (
        "W1_CACHED_AND_CORRECTNESS",
        core.C1_STAGE,
        "W2_PROFILED_CAPTURE",
        "W3_CLEAN_MATCHED_ARM",
    )
    if tuple(core.POST_COLD_STAGE_ORDER) != expected_order:
        raise CapabilityRefusal("held graph order differs from W1→C1→W2→W3")
    if core.HELD_RESERVATION_SECONDS != 4_220.0:
        raise CapabilityRefusal("held reservation is not exactly 4,220 seconds")
    if core.GLOBAL_DEADLINE_SECONDS != 4_500.0:
        raise CapabilityRefusal("held deadline is not exactly 4,500 seconds")
    if core.W2_W3_REQUIRED_SECONDS != 2_000.0:
        raise CapabilityRefusal("combined W2/W3 reservation is not 2,000 seconds")

    executor = _source_identity(
        SCRIPT_DIR / "m0_three_window_executor.py",
        (
            'if stage == "W2_PROFILED_CAPTURE":',
            '"W2_AND_W3_REMAINING_BUDGET"',
            "core_session.W2_W3_REQUIRED_SECONDS",
            "build_session_lock_release_proof(",
            'stage = "POSTLOCK_W3_PAIR"',
            'stage = "POSTLOCK_W2_EXPORT_CENSUS"',
        ),
    )
    postlock = _source_identity(
        SCRIPT_DIR / "m0_postlock_census.py",
        (
            'expected_source_sha256=artifacts["source_rep"]["sha256"]',
            "analyze_w2(",
        ),
    )
    export = _source_identity(
        SCRIPT_DIR / "nsys_export.py",
        (
            '"--force-export=true"',
            '"READ_EXISTING"',
            '"private_sqlite_sha256"',
            "expected_source_sha256",
        ),
    )
    sources = {
        "executor": executor,
        "postlock": postlock,
        "export": export,
    }
    return {
        "raw_capture_hash_bound_before_postlock_analysis": {
            "status": "PASS",
            "mechanism": (
                "analyze_w2 passes the W2 manager-bound .nsys-rep SHA-256 as "
                "expected_source_sha256 before export"
            ),
            "sources": ["postlock", "export"],
        },
        "postcapture_analysis_after_lock_release": {
            "status": "PASS",
            "mechanism": (
                "the outer owner creates the session lock-release proof before "
                "either POSTLOCK analysis stage"
            ),
            "sources": ["executor"],
        },
        "postcapture_failure_preserves_raw_capture_and_readonly_sqlite": {
            "status": "PASS",
            "mechanism": (
                "one private SQLite is created after release; subsequent reports "
                "use READ_EXISTING and bind both raw-report and SQLite hashes"
            ),
            "sources": ["export", "postlock"],
        },
        "insufficient_w2_w3_time_stops_before_w2": {
            "status": "PASS",
            "mechanism": (
                "the live graph requires the combined 2,000-second W2+W3 "
                "reservation before invoking W2"
            ),
            "sources": ["executor"],
        },
        "no_extra_device_arm_after_w2_hash_and_w3_stop": {
            "status": "PASS",
            "mechanism": (
                "the immutable held graph ends at W3; all subsequent stages are "
                "post-lock CPU analysis of the hash-bound W2 capture"
            ),
            "sources": ["executor", "postlock"],
        },
        "_source_identities": sources,
    }


def _validate_trip_count(boundary: Mapping[str, Any]) -> dict[str, Any]:
    lowered = boundary.get("lowered_program")
    if not isinstance(lowered, Mapping):
        raise CapabilityRefusal("real CPU boundary has no lowered program")
    trip = lowered.get("integration_trip_count")
    if not isinstance(trip, Mapping):
        raise CapabilityRefusal("exact lowered integration trip count is absent")
    configured = trip.get("configured_cross_check")
    if not isinstance(configured, Mapping):
        raise CapabilityRefusal("configured trip-count cross-check is absent")
    steps = trip.get("steps")
    configured_steps = configured.get("exact_steps")
    if (
        trip.get("status") != "PASS"
        or trip.get("method") != "exact-lowered-trip-count"
        or trip.get("count_matches_configuration") is not True
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
        or steps != configured_steps
        or trip.get("stablehlo_sha256") != lowered.get("sha256")
    ):
        raise CapabilityRefusal(
            "exact lowered trip count is missing, mislabeled, or disagrees "
            "with the exact configured interval/timestep derivation"
        )
    return json.loads(json.dumps(dict(trip), sort_keys=True))


def validate_real_boundary(
    boundary: Mapping[str, Any],
    *,
    expected_authority_sha256: str | None = None,
) -> dict[str, Any]:
    authority = boundary.get("authority")
    if (
        boundary.get("schema")
        != "wrf_gpu2.v025.m0.cpu_real_boundary_preflight.v1"
        or boundary.get("status") != "PASS"
        or boundary.get("device_action") is not False
        or boundary.get("platforms") != ["cpu"]
        or boundary.get("observations") != EXPECTED_BOUNDARY_OBSERVATIONS
        or (boundary.get("native_bundle") or {}).get("field_count") != 29
        or not isinstance(authority, Mapping)
        or (
            expected_authority_sha256 is not None
            and authority.get("authority_sha256")
            != expected_authority_sha256
        )
    ):
        raise CapabilityRefusal(
            "real CPU boundary schema, platform, native loader, observations, "
            "or source-authority binding is invalid"
        )
    _validate_trip_count(boundary)
    return json.loads(json.dumps(dict(boundary), sort_keys=True))


def build_declaration(
    boundary: Mapping[str, Any],
    *,
    boundary_path: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a signed/content-addressed declaration without device authority."""

    environ = os.environ if environ is None else environ
    validate_real_boundary(boundary)
    if str(environ.get("GPUWRF_M0_EVIDENCE", "")).strip():
        raise CapabilityRefusal(
            "GPUWRF_M0_EVIDENCE must remain unset; a production range would "
            "duplicate and invalidate the harness integration clip"
        )
    trip = _validate_trip_count(boundary)
    child = _source_identity(
        SCRIPT_DIR / "m0_exact_boundary_child.py",
        (
            'RANGE_NAME = "GPUWRF_M0_FORECAST_INTEGRATION"',
            "with jax.profiler.TraceAnnotation(RANGE_NAME):",
            "jax.block_until_ready(result)",
        ),
    )
    parent = _source_identity(
        SCRIPT_DIR / "m0_window_parent.py",
        (
            '"GPUWRF_M0_EVIDENCE",',
            "*EVIDENCE_ENV,",
            "step1.nsys_command(inner, out_root=stage_root)",
        ),
    )
    invariants = observe_fallback_invariants()
    source_identities = invariants.pop("_source_identities")

    providers = {
        "exact_kernels_per_step_denominator": (
            "same exact FAST StableHLO plus W2 integration-clipped kernel events"
        ),
        "integration_clipped_gpu_busy": (
            "W2 NVTX/CUDA projection for GPUWRF_M0_FORECAST_INTEGRATION"
        ),
        "event_level_family_join": "W2 CUDA-kernel event/name join",
        "kernel_duration_distribution": "W2 kernel event durations",
        "inter_kernel_gap_distribution": "W2 integration-clipped event gaps",
        "launch_latency_distribution": "W2 runtime-to-kernel correlation",
        "transfer_audit": "W2 CUDA memcpy/memset event audit",
        "profiler_perturbation": "W2 profiled / W3 clean matched pair",
        "peak_vram": "W2 same-process allocator plus lock-owner sampler",
    }
    available = {
        name: {"status": "AVAILABLE", "provider": providers[name]}
        for name in AVAILABLE_FIELDS
    }
    missing_reasons = {
        "achieved_dram_gb_per_s": (
            "the frozen nsys capture has no achieved-DRAM-throughput counter"
        ),
        "occupancy_and_stall_counters": (
            "the frozen capture has no Nsight Compute occupancy/stall counters"
        ),
        "run_physics_false": (
            "the frozen FAST call has run_physics=True and no second device arm "
            "is authorized"
        ),
    }
    missing = {
        name: {"status": "MISSING", "reason": missing_reasons[name]}
        for name in MISSING_FIELDS
    }
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "device_action": False,
        "coordination_action": False,
        "boundary": {
            "path": str(Path(boundary_path).resolve()),
            "file_sha256": sha256_file(Path(boundary_path)),
            "content_sha256": boundary.get("boundary_sha256"),
            "stablehlo_sha256": (
                boundary.get("lowered_program") or {}
            ).get("sha256"),
        },
        "denominator": trip,
        "integration_clip": {
            "status": "AVAILABLE",
            "range_name": "GPUWRF_M0_FORECAST_INTEGRATION",
            "emitter": "scripts/v025/m0_exact_boundary_child.py",
            "emitter_symbol": "run_boundary",
            "emitter_source": child,
            "production_operational_mode_emitter": False,
            "GPUWRF_M0_EVIDENCE": "UNSET",
            "parent_strips_production_evidence_environment": True,
            "parent_source": parent,
        },
        "available": available,
        "missing": missing,
        "fallback_invariants": invariants,
        "invariant_source_identities": source_identities,
        "accepted_residual": (
            "CPU lower() proves exact argument/tracing reachability but cannot "
            "prove later device compile/link or measured device values"
        ),
    }
    signed_sha256 = canonical_sha256(payload)
    payload["content_address"] = {
        "algorithm": "sha256",
        "sha256": signed_sha256,
    }
    payload["signature"] = {
        "scheme": "sha256-canonical-json-content-signature",
        "signed_sha256": signed_sha256,
    }
    validate_declaration(payload)
    return payload


def validate_declaration(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"content_address", "signature"}
    }
    digest = canonical_sha256(unsigned)
    available = payload.get("available")
    missing = payload.get("missing")
    invariants = payload.get("fallback_invariants")
    expected_invariants = observe_fallback_invariants()
    expected_invariants.pop("_source_identities")
    clip = payload.get("integration_clip")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("device_action") is not False
        or payload.get("coordination_action") is not False
        or not isinstance(available, Mapping)
        or set(available) != set(AVAILABLE_FIELDS)
        or any(
            not isinstance(record, Mapping)
            or record.get("status") != "AVAILABLE"
            for record in available.values()
        )
        or not isinstance(missing, Mapping)
        or set(missing) != set(MISSING_FIELDS)
        or any(
            not isinstance(record, Mapping)
            or record.get("status") != "MISSING"
            for record in missing.values()
        )
        or not isinstance(invariants, Mapping)
        or set(invariants) != set(FALLBACK_INVARIANT_NAMES)
        or any(
            not isinstance(record, Mapping)
            or record.get("status") != "PASS"
            for record in invariants.values()
        )
        or dict(invariants) != expected_invariants
        or not isinstance(clip, Mapping)
        or clip.get("range_name") != "GPUWRF_M0_FORECAST_INTEGRATION"
        or clip.get("emitter")
        != "scripts/v025/m0_exact_boundary_child.py"
        or clip.get("emitter_symbol") != "run_boundary"
        or clip.get("production_operational_mode_emitter") is not False
        or clip.get("GPUWRF_M0_EVIDENCE") != "UNSET"
        or clip.get("parent_strips_production_evidence_environment")
        is not True
        or (payload.get("content_address") or {}).get("sha256") != digest
        or (payload.get("signature") or {}).get("signed_sha256") != digest
    ):
        raise CapabilityRefusal(
            "fallback capability declaration schema, exact sets, invariant "
            "statuses, or content signature is invalid"
        )
    _validate_trip_count(
        {"lowered_program": {
            "sha256": (payload.get("boundary") or {}).get("stablehlo_sha256"),
            "integration_trip_count": payload.get("denominator"),
        }}
    )
    return json.loads(json.dumps(dict(payload), sort_keys=True))


__all__ = [
    "AVAILABLE_FIELDS",
    "CapabilityRefusal",
    "FALLBACK_INVARIANT_NAMES",
    "HARD_GATE_MUTATIONS",
    "MISSING_FIELDS",
    "SCHEMA",
    "build_declaration",
    "canonical_sha256",
    "observe_fallback_invariants",
    "validate_real_boundary",
    "validate_declaration",
]
