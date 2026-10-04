#!/usr/bin/env python3
"""The §9 `baseline-census`, fail-closed (CPU-only parsing).

Manager review on main `c227cfc7` returned the previous version. It was not the
frozen census; it was a weaker parallel one, and every weakness pointed the same
way -- towards reporting `OK`:

* it checked attributed **device time** only. §9 caps unknown at 5% of **both**
  launches and device time. A census that attributes 99% of nanoseconds while
  losing half the launches has not attributed itself.
* it had no top-family rank qualification against the static proxy.
* its transfer summary could not tell initialisation from timestep-loop copies,
  and an **empty** transfer input returned `OK` -- absence of evidence scored as
  evidence of absence, on the one gate §7 exists to enforce.
* it classified "profiler perturbation" from `TSL:` projected device time. That
  is application/XLA work, not profiler overhead. The number was meaningless.
* it reported the largest single byte row as "peak VRAM". That is the biggest
  allocation *event*, not peak *residency*.

The repair reuses `run_gpu_arm.attribute_device_time` -- the frozen two-share
taxonomy that `validate_kernel_census.py` already validates -- rather than
carrying a second, divergent implementation. Everything that cannot be measured
from the artifacts in hand is `MISSING`, and any missing required sub-gate makes
the whole census `BLOCKED`. `OK` here has to mean the census was actually taken.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v025"))

# NOTE: deliberately does NOT import `cpu_guard`. This is a library imported by
# the coordinated GPU entry point `step1_driver`; pinning the platform here would
# make a legitimately authorised window refuse itself. It imports no accelerator
# package, so there is nothing to pin.
import run_gpu_arm as arm  # noqa: E402

A6_CENSUS = REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json"

MIN_ATTRIBUTION = 0.95          # §9, applied to BOTH launches and device time
MIN_PRODUCTION_FAMILY_COVERAGE = 0.95
MIN_TOP_FAMILY_SPEARMAN = 0.80  # §5.3, matched two-domain production reference
TOP_FAMILIES = 6

#: Sub-gates that must be present AND passing for the census to be OK.
REQUIRED_GATES = ("device_time_attribution", "production_representativeness",
                  "transfer_audit", "profiler_perturbation", "vram")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEVICE_UUID = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)


def _missing(what: str, needs: str) -> dict[str, Any]:
    return {"status": "MISSING", "reason": what, "needs": needs}


# --------------------------------------------------------------------------- #
# static proxy, per family                                                     #
# --------------------------------------------------------------------------- #
def static_proxy_shares(census_path: Path = A6_CENSUS) -> dict[str, float]:
    """Per-family share of the A6 static launch-count proxy."""
    census = json.loads(census_path.read_text())
    per_family: dict[str, float] = defaultdict(float)
    for operator in census["operators"]:
        proxy = operator.get("static_launch_proxy")
        value = proxy.get("launches") if isinstance(proxy, dict) else proxy
        per_family[operator["family"]] += float(value or 0)
    total = sum(per_family.values())
    return {family: value / total for family, value in per_family.items()} if total else {}


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation without scipy. Ties get average ranks."""
    if len(a) != len(b) or len(a) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2.0 + 1.0
            for position in range(index, stop + 1):
                out[order[position]] = average
            index = stop + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den = (sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)) ** 0.5
    return (num / den) if den else None


def a6_static_proxy_diagnostic(families: dict[str, Any],
                               proxy: dict[str, float]) -> dict[str, Any]:
    """Compare device time with A6, explicitly as a non-gating diagnostic."""
    measured = {name: entry.get("device_time_share") or 0.0
                for name, entry in families.items() if name != "unknown"}
    common = [f for f in measured if f in proxy]
    if len(common) < 3:
        return _missing(
            f"only {len(common)} families are common to the trace and the static proxy",
            "at least 3 common families for a rank correlation")
    common.sort(key=lambda f: -measured[f])
    top = common[:TOP_FAMILIES]
    rho = _spearman([measured[f] for f in top], [proxy[f] for f in top])
    if rho is None:
        return _missing("rank correlation undefined", "at least 3 ranked families")
    return {
        "status": "DIAGNOSTIC",
        "spearman": rho,
        "top_families": top,
        "measured_shares": {f: measured[f] for f in top},
        "proxy_shares": {f: proxy[f] for f in top},
        "eligible_for_fast_representativeness_gate": False,
        "meaning": (
            "A6 counts launches on the CPU backend; this trace measures device time. "
            "This is retained as a separately named diagnostic and MUST NOT substitute "
            "for the frozen matched two-domain production rho>=0.80 gate."
        ),
    }


def production_representativeness_gate(
    families: dict[str, Any],
    reference: dict[str, Any] | None,
    candidate_coverage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Frozen §5.3 coverage/rank/scheme/cadence gate.

    The reference must be a hash-bound matched short two-domain production
    trace.  A6 is intentionally not accepted here.
    """
    if not reference:
        return _missing(
            "no matched two-domain production reference was supplied",
            "a hash-bound reference trace with a pre-registered non-nesting family set, "
            "top-family shares, active d01 schemes, and cadence events",
        )
    if reference.get("dry_run_stub") is True or (
        candidate_coverage and candidate_coverage.get("dry_run_stub") is True
    ):
        return _missing(
            "the matched reference or candidate coverage is a dry-run fixture",
            "hash-bound evidence from a real FAST-v025 capture and a real matched "
            "two-domain production capture",
        )
    if reference.get("case_role") != "matched_short_two_domain_production":
        return _missing(
            "the supplied reference is not labelled matched_short_two_domain_production",
            "the frozen production reference, not A6 or another candidate trace",
        )
    source = reference.get("source_rep") or {}
    if not _SHA256.fullmatch(str(source.get("sha256", ""))):
        return _missing(
            "the production reference is not source-report hash bound",
            "source_rep.sha256 with the full 64-hex digest",
        )
    reference_workload = reference.get("workload_identity_sha256")
    if not _SHA256.fullmatch(str(reference_workload or "")):
        return _missing(
            "the production reference has no workload identity hash",
            "workload_identity_sha256 binding the case, configuration, inputs, and event window",
        )

    raw_reference_families = reference.get("non_nesting_families")
    if not isinstance(raw_reference_families, dict) or not raw_reference_families:
        return _missing(
            "the production reference has no pre-registered non-nesting family set",
            "non_nesting_families mapping family -> device_time_share",
        )
    reference_shares: dict[str, float] = {}
    for family, value in raw_reference_families.items():
        share = value.get("device_time_share") if isinstance(value, dict) else value
        try:
            share = float(share)
        except (TypeError, ValueError):
            return _missing(
                f"production family {family!r} has no numeric device-time share",
                "a complete reference share for every pre-registered family",
            )
        if not math.isfinite(share) or share < 0:
            return _missing(
                f"production family {family!r} has an invalid device-time share",
                "finite non-negative family shares",
            )
        reference_shares[str(family)] = share

    rank_families = reference.get("rank_families")
    if not isinstance(rank_families, list) or len(rank_families) < 3:
        return _missing(
            "the production reference has no pre-registered rank family set",
            "rank_families with at least three family names",
        )
    if len(set(rank_families)) != len(rank_families) or any(
        family not in reference_shares for family in rank_families
    ):
        return _missing(
            "rank_families is duplicated or not a subset of non_nesting_families",
            "one unique, fully referenced ranking set",
        )

    candidate_shares = {
        name: float(entry.get("device_time_share") or 0.0)
        for name, entry in families.items()
        if name != "unknown"
    }
    present = [
        family for family in reference_shares
        if candidate_shares.get(family, 0.0) > 0.0
    ]
    missing_families = sorted(set(reference_shares) - set(present))
    family_coverage = len(present) / len(reference_shares)
    rho = _spearman(
        [candidate_shares.get(family, 0.0) for family in rank_families],
        [reference_shares[family] for family in rank_families],
    )

    if not candidate_coverage:
        return _missing(
            "no executed scheme/cadence manifest was supplied for the candidate",
            "a run/hash-bound production-derived integration manifest",
        )
    if candidate_coverage.get("run_id") != candidate_coverage.get("capture_run_id"):
        return _missing(
            "candidate scheme/cadence evidence is not bound to its capture run ID",
            "matching run_id and capture_run_id",
        )
    if not _SHA256.fullmatch(str(candidate_coverage.get("source_rep_sha256", ""))):
        return _missing(
            "candidate scheme/cadence evidence is not bound to its source report",
            "the full source .nsys-rep SHA-256",
        )
    candidate_workload = candidate_coverage.get("workload_identity_sha256")
    if not _SHA256.fullmatch(str(candidate_workload or "")):
        return _missing(
            "candidate scheme/cadence evidence has no workload identity hash",
            "workload_identity_sha256 binding the case, configuration, inputs, and event window",
        )
    if candidate_workload != reference_workload:
        return _missing(
            "candidate and production reference are not bound to the same workload identity",
            "the same workload identity in matching workload_identity_sha256 values",
        )

    required_schemes = set(reference.get("active_d01_schemes") or [])
    required_events = set(reference.get("cadence_events") or [])
    if not required_schemes or not required_events:
        return _missing(
            "the production reference omits active d01 schemes or cadence events",
            "non-empty active_d01_schemes and cadence_events",
        )
    executed_schemes = set(candidate_coverage.get("executed_schemes") or [])
    executed_events = set(candidate_coverage.get("executed_cadence_events") or [])
    missing_schemes = sorted(required_schemes - executed_schemes)
    missing_events = sorted(required_events - executed_events)

    checks = {
        "family_coverage": family_coverage >= MIN_PRODUCTION_FAMILY_COVERAGE,
        "top_family_rank": rho is not None and rho >= MIN_TOP_FAMILY_SPEARMAN,
        "active_d01_schemes": not missing_schemes,
        "cadence_events": not missing_events,
    }
    return {
        "status": "OK" if all(checks.values()) else "FAILED",
        "checks": checks,
        "family_coverage": family_coverage,
        "min_family_coverage": MIN_PRODUCTION_FAMILY_COVERAGE,
        "missing_reference_families": missing_families,
        "spearman": rho,
        "min_spearman": MIN_TOP_FAMILY_SPEARMAN,
        "rank_families": rank_families,
        "candidate_rank_shares": {
            family: candidate_shares.get(family, 0.0) for family in rank_families
        },
        "reference_rank_shares": {
            family: reference_shares[family] for family in rank_families
        },
        "missing_schemes": missing_schemes,
        "missing_cadence_events": missing_events,
        "reference_binding": {
            "run_id": reference.get("run_id"),
            "source_rep": source,
            "workload_identity_sha256": reference_workload,
        },
        "candidate_binding": {
            "run_id": candidate_coverage.get("run_id"),
            "source_rep_sha256": candidate_coverage.get("source_rep_sha256"),
            "workload_identity_sha256": candidate_workload,
        },
        "reference_kind": "matched two-domain production trace",
        "a6_used_for_this_gate": False,
    }


# --------------------------------------------------------------------------- #
# transfers, scoped to the timestep loop                                       #
# --------------------------------------------------------------------------- #
def transfer_gate(mem_rows: list[dict[str, Any]] | None,
                  integration_scope: dict[str, Any] | None) -> dict[str, Any]:
    """§7/§9: ZERO host/device transfers inside the timestep loop.

    An aggregate whole-process memcpy summary cannot answer this: initialisation
    legitimately copies inputs to the device. Without a timestep window the gate
    is MISSING, and an empty input is MISSING too -- "no rows" is equally
    consistent with "no transfers" and "the report was never exported", and the
    previous version scored that as a pass.
    """
    if mem_rows is None:
        return _missing("no timestamped CUDA trace was exported",
                        "cuda_gpu_trace from the real .nsys-rep")
    if not mem_rows:
        return _missing(
            "the timestamped CUDA trace is empty",
            "positive evidence from a non-empty trace or an explicit "
            "profiler zero-event certificate",
        )

    unplaceable = [row for row in mem_rows
                   if row.get("start_ns") is None or row.get("duration_ns") is None]
    if unplaceable:
        return _missing(
            f"{len(unplaceable)} CUDA rows have no timestamps; aggregate summaries are ineligible",
            "cuda_gpu_trace rows with Start (ns) and Duration (ns)",
        )
    if not integration_scope or integration_scope.get("status") != "OK":
        return _missing(
            "no mechanically verified production integration scope is available",
            "a production-derived integration/timestep boundary; stub-only timestep markers "
            "and whole-process ranges are ineligible",
        )
    if integration_scope.get("dry_run_stub") is True:
        return _missing(
            "the available integration scope is a dry-run fixture",
            "an NVTX range derived from the real captured forecast process",
        )
    if not (
        integration_scope.get("schema") == "wrf_gpu2.v025.m0.integration_scope.v1"
        and integration_scope.get("production_derived") is True
        and integration_scope.get("mechanically_verified") is True
        and integration_scope.get("boundary_kind") in {"integration", "timestep"}
        and integration_scope.get("boundary_source") == "nvtx_pushpop_trace"
    ):
        return _missing(
            "the supplied scope is not a mechanically verified production integration boundary",
            "the integration_scope.v1 schema, production_derived=true, "
            "mechanically_verified=true, and an nvtx_pushpop_trace boundary",
        )
    if (
        integration_scope.get("run_id") != integration_scope.get("capture_run_id")
        or not _SHA256.fullmatch(
            str(integration_scope.get("source_rep_sha256", ""))
        )
    ):
        return _missing(
            "the integration scope is not bound to its capture run and source report",
            "matching run_id/capture_run_id and the source .nsys-rep SHA-256",
        )
    try:
        start = float(integration_scope["start_ns"])
        end = float(integration_scope["end_ns"])
    except (KeyError, TypeError, ValueError):
        return _missing("the integration scope has no numeric bounds", "start_ns and end_ns")
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        return _missing("the integration scope bounds are invalid", "finite 0 <= start < end")

    in_loop = [r for r in mem_rows
               if float(r["start_ns"]) <= end
               and float(r["start_ns"]) + float(r["duration_ns"]) >= start]

    def crosses_host_device(row: dict[str, Any]) -> bool:
        src = str(row.get("source_memory_kind") or "").lower()
        dst = str(row.get("destination_memory_kind") or "").lower()
        name = str(row.get("name") or "").lower()
        return (
            ("host" in src and "device" in dst)
            or ("device" in src and "host" in dst)
            or "htod" in name
            or "dtoh" in name
            or "host-to-device" in name
            or "device-to-host" in name
        )

    host_device = [r for r in in_loop
                   if crosses_host_device(r)]
    return {
        "status": "OK" if not host_device else "FAILED",
        "integration_scope": integration_scope,
        "rows_in_loop": len(in_loop),
        "host_device_in_loop": len(host_device),
        "offending": [r.get("name") for r in host_device][:10],
        "rule": "§7 forbids host/device transfer inside the timestep loop without an ADR",
    }


# --------------------------------------------------------------------------- #
# profiler perturbation: a MATCHED pair or nothing                             #
# --------------------------------------------------------------------------- #
def profiler_gate(matched: dict[str, Any] | None) -> dict[str, Any]:
    """Needs profiled and unprofiled wall times for the SAME workload.

    The previous version derived a "class" from the share of projected device
    time sitting under `TSL:` ranges. That is application and XLA work, not
    profiler cost -- the number could not have measured overhead even in
    principle. There is no substitute for a matched pair.
    """
    if not matched:
        return _missing(
            "no matched profiled/unprofiled timing was supplied",
            "the same workload timed with and without nsys, so overhead is a difference "
            "rather than an inference")
    if matched.get("schema") != "wrf_gpu2.v025.m0.profiler_matched_pair.v1":
        return _missing(
            "the matched pair has no recognised identity schema",
            "wrf_gpu2.v025.m0.profiler_matched_pair.v1",
        )
    profiled_arm = matched.get("profiled") or {}
    unprofiled_arm = matched.get("unprofiled") or {}
    required_binding = (
        "workload_identity_sha256", "integration_scope_sha256", "event_mix_sha256"
    )
    if any(
        not _SHA256.fullmatch(str(arm.get(key, "")))
        for arm in (profiled_arm, unprofiled_arm)
        for key in required_binding
    ):
        return _missing(
            "the matched arms are not hash-bound to workload, integration scope, and event mix",
            ", ".join(required_binding),
        )
    mismatched = [
        key for key in required_binding
        if profiled_arm.get(key) != unprofiled_arm.get(key)
    ]
    if mismatched:
        return _missing(
            f"the profiler arms differ in {mismatched}",
            "identical workload, integration scope, and cadence/event mix",
        )
    if (
        profiled_arm.get("timing_region") != "integration-only"
        or unprofiled_arm.get("timing_region") != "integration-only"
    ):
        return _missing(
            "the timing region is not integration-only in both arms",
            "the same synchronized integration region, excluding compile and I/O",
        )
    if profiled_arm.get("instrumentation") != "nsys" or unprofiled_arm.get(
        "instrumentation"
    ) != "none":
        return _missing(
            "the instrumentation labels do not identify nsys versus bare execution",
            "profiled.instrumentation=nsys and unprofiled.instrumentation=none",
        )
    profiled_run_id = profiled_arm.get("run_id")
    unprofiled_run_id = unprofiled_arm.get("run_id")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (profiled_run_id, unprofiled_run_id)
    ):
        return _missing("one or both profiler arms lack a run ID",
                        "two fresh named processes")
    if profiled_run_id == unprofiled_run_id:
        return _missing("the two arms reuse one run ID", "two fresh distinct processes")
    order = matched.get("order")
    if order != [profiled_arm.get("run_id"), unprofiled_arm.get("run_id")]:
        return _missing(
            "the pair is not mechanically profiled-first",
            "order=[profiled.run_id, unprofiled.run_id]",
        )
    try:
        profiled = float(profiled_arm["seconds"])
        unprofiled = float(unprofiled_arm["seconds"])
    except (KeyError, TypeError, ValueError):
        return _missing("the matched pair is incomplete",
                        "both arms with numeric seconds > 0")
    if not (
        math.isfinite(profiled) and math.isfinite(unprofiled)
        and profiled > 0 and unprofiled > 0
    ):
        return _missing("the matched pair has non-finite/non-positive time",
                        "finite seconds > 0 in both arms")
    overhead = (profiled - unprofiled) / unprofiled
    return {
        "status": "OK",
        "overhead_fraction": overhead,
        "profiled_seconds": profiled,
        "unprofiled_seconds": unprofiled,
        "order": order,
        "identity": {key: profiled_arm[key] for key in required_binding},
        "class": "LOW" if overhead <= 0.10 else "HIGH",
        "why_it_matters": ("a device-time census read off a heavily perturbed run is a census of "
                           "the instrumented run, not of production"),
    }


# --------------------------------------------------------------------------- #
# VRAM: total residency or nothing                                             #
# --------------------------------------------------------------------------- #
def vram_gate(residency: dict[str, Any] | None) -> dict[str, Any]:
    """Validate paired forecast-allocator and lock-owner residency sidecars."""
    if not residency:
        return _missing(
            "no total-residency measurement is available",
            "forecast-process JAX allocator counters PLUS a lock-owner total-residency "
            "series. The capture parent must never initialize a second JAX allocator.",
        )
    if residency.get("schema") != "wrf_gpu2.v025.m0.vram_evidence.v1":
        return _missing("unrecognised VRAM evidence schema",
                        "wrf_gpu2.v025.m0.vram_evidence.v1")
    allocator = residency.get("forecast_allocator") or {}
    total = residency.get("lock_owner_total_residency") or {}
    if allocator.get("schema") != "wrf_gpu2.v025.m0.forecast_allocator.v1":
        return _missing(
            "unrecognised forecast allocator schema",
            "wrf_gpu2.v025.m0.forecast_allocator.v1",
        )
    if total.get("schema") != "wrf_gpu2.v025.m0.lock_owner_total_residency.v1":
        return _missing(
            "unrecognised lock-owner sampler schema",
            "wrf_gpu2.v025.m0.lock_owner_total_residency.v1",
        )
    if allocator.get("emitter_process_role") != "forecast_process":
        return _missing(
            "allocator counters were not emitted by the forecast process",
            "peak_bytes_in_use and peak_bytes_reserved from the process that owns JAX",
        )
    if total.get("sampler_process_role") != "lock_owner_parent":
        return _missing(
            "total residency was not sampled by the lock-owning parent",
            "an external sampler that does not initialize JAX",
        )
    if total.get("jax_imported_by_sampler") is not False:
        return {
            "status": "FAILED",
            "reason": "lock-owner sampler did not prove that it avoided JAX import",
        }
    if total.get("jax_import_check") != {
        "method": "sys.modules-prefix-scan-before-baseline-or-stream",
        "status": "PASS",
        "loaded_modules": [],
    }:
        return {
            "status": "FAILED",
            "reason": "lock-owner sampler lacks a runtime pre-stream JAX import check",
        }
    if (
        total.get("orphan_control")
        != "linux-prctl-pdeathsig-sigterm-plus-parent-stop"
    ):
        return {
            "status": "FAILED",
            "reason": "external sampler lacks parent-death/orphan control provenance",
        }
    instrumentation = allocator.get("instrumentation") or {}
    if (
        instrumentation.get("enabled") is not True
        or instrumentation.get("range_name") != "GPUWRF_M0_FORECAST_INTEGRATION"
        or instrumentation.get("default_when_unset")
        != "original-direct-call-no-range-no-sidecar"
    ):
        return {
            "status": "FAILED",
            "reason": "forecast allocator sidecar lacks ADR-036 opt-in/default-off provenance",
        }
    binding_keys = (
        "run_id",
        "forecast_pid",
        "source_sha256",
        "config_sha256",
        "input_manifest_sha256",
        "device_uuid",
    )
    if any(allocator.get(key) != total.get(key) for key in binding_keys):
        return _missing(
            "allocator and total-residency sidecars do not bind to the same run/process/identity",
            ", ".join(binding_keys),
        )
    if not isinstance(allocator.get("run_id"), str) or not allocator["run_id"].strip():
        return _missing("VRAM sidecars have no run ID", "one non-empty run ID")
    try:
        forecast_pid = int(allocator["forecast_pid"])
    except (KeyError, TypeError, ValueError):
        return _missing("VRAM sidecars have no forecast PID", "a positive forecast PID")
    if forecast_pid <= 0:
        return {"status": "FAILED", "reason": "VRAM evidence has an invalid forecast PID"}
    if not all(
        _SHA256.fullmatch(str(allocator.get(key, "")))
        for key in ("source_sha256", "config_sha256", "input_manifest_sha256")
    ):
        return _missing("VRAM sidecars have invalid identity hashes", "full SHA-256 values")
    if not _DEVICE_UUID.fullmatch(str(allocator.get("device_uuid", ""))):
        return _missing("VRAM sidecars have invalid device identity", "one full GPU UUID")
    if total.get("unexpected_competing_contexts") not in ([], 0):
        return {
            "status": "FAILED",
            "reason": "an unexpected competing context was observed during the residency series",
            "unexpected_competing_contexts": total.get("unexpected_competing_contexts"),
        }
    try:
        peak_in_use = float(allocator["peak_bytes_in_use"])
        peak_reserved = float(allocator["peak_bytes_reserved"])
        peak_absolute = float(total["peak_absolute_bytes"])
        peak_baseline_subtracted = float(total["peak_baseline_subtracted_bytes"])
        baseline_absolute = float(total["baseline_absolute_bytes"])
        cadence_ms = float(total["sample_cadence_ms"])
        samples = int(total["samples"])
        misses = int(total["sampling_misses"])
        allocator_start_ns = float(allocator["measurement_start_ns"])
        allocator_end_ns = float(allocator["measurement_end_ns"])
        sampler_start_ns = float(total["measurement_start_ns"])
        sampler_end_ns = float(total["measurement_end_ns"])
    except (KeyError, TypeError, ValueError):
        return _missing("VRAM sidecars are incomplete", "all allocator and sampler fields")
    numbers = (
        peak_in_use, peak_reserved, peak_absolute, peak_baseline_subtracted,
        baseline_absolute,
        cadence_ms, allocator_start_ns, allocator_end_ns,
        sampler_start_ns, sampler_end_ns,
    )
    if any(not math.isfinite(value) or value < 0 for value in numbers):
        return {"status": "FAILED", "reason": "VRAM evidence contains invalid numeric values"}
    if cadence_ms <= 0:
        return {"status": "FAILED", "reason": "VRAM sampling cadence must be positive"}
    if peak_reserved < peak_in_use:
        return {
            "status": "FAILED",
            "reason": "allocator peak_bytes_reserved is below peak_bytes_in_use",
        }
    if peak_baseline_subtracted > peak_absolute:
        return {
            "status": "FAILED",
            "reason": "baseline-subtracted residency exceeds absolute residency",
        }
    expected_subtracted = max(0.0, peak_absolute - baseline_absolute)
    if peak_baseline_subtracted != expected_subtracted:
        return {
            "status": "FAILED",
            "reason": "baseline-subtracted residency is inconsistent with absolute/baseline peaks",
        }
    if not (
        sampler_start_ns <= allocator_start_ns
        < allocator_end_ns <= sampler_end_ns
    ):
        return {
            "status": "FAILED",
            "reason": (
                "lock-owner sampling timestamps do not enclose the forecast allocator "
                "measurement interval"
            ),
        }
    miss_fraction = (
        misses / (samples + misses)
        if samples > 0 and misses >= 0
        else math.inf
    )
    quality = total.get("sampling_quality")
    limitations = total.get("sampling_limitations")
    if (
        samples <= 0
        or misses < 0
        or miss_fraction > 0.05
        or not isinstance(quality, dict)
        or quality.get("baseline_counted_as_stream_sample") is not False
        or quality.get("denominator")
        != "stream_samples_plus_inferred_or_malformed_misses"
        or quality.get("maximum_miss_fraction") != 0.05
        or quality.get("status") != "PASS"
        or quality.get("observed_miss_fraction") != miss_fraction
        or not isinstance(limitations, list)
        or len(limitations) < 3
        or not any("shorter than sample_cadence_ms" in str(item) for item in limitations)
    ):
        return {"status": "FAILED", "reason": "VRAM sampling coverage is invalid"}
    observed_pids = total.get("observed_process_tree_pids")
    if (
        not isinstance(observed_pids, list)
        or forecast_pid not in observed_pids
    ):
        return {
            "status": "FAILED",
            "reason": "forecast PID was not observed in the sampled child process tree",
        }
    utc_intervals = []
    for sidecar in (allocator, total):
        interval = []
        for field in ("measurement_start_utc", "measurement_end_utc"):
            try:
                stamp = datetime.fromisoformat(str(sidecar[field]))
            except (KeyError, TypeError, ValueError):
                return _missing(
                    "VRAM sidecars lack valid UTC enclosure",
                    "timezone-bearing measurement_start_utc and measurement_end_utc",
                )
            if stamp.tzinfo is None:
                return _missing(
                    "VRAM sidecars lack timezone-bearing UTC enclosure",
                    "timezone-bearing measurement_start_utc and measurement_end_utc",
                )
            if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
                return {
                    "status": "FAILED",
                    "reason": "VRAM sidecar timestamp is timezone-bearing but not UTC",
                }
            interval.append(stamp)
        if interval[1] <= interval[0]:
            return {
                "status": "FAILED",
                "reason": "VRAM sidecar UTC interval is invalid",
            }
        utc_intervals.append(interval)
    allocator_utc, sampler_utc = utc_intervals
    if not (
        sampler_utc[0]
        <= allocator_utc[0]
        < allocator_utc[1]
        <= sampler_utc[1]
    ):
        return {
            "status": "FAILED",
            "reason": "lock-owner UTC timestamps do not enclose forecast allocator interval",
        }
    return {
        "status": "OK",
        "peak_resident_bytes": peak_baseline_subtracted,
        "peak_resident_mib": peak_baseline_subtracted / (1 << 20),
        "peak_absolute_bytes": peak_absolute,
        "baseline_absolute_bytes": baseline_absolute,
        "allocator_peak_bytes_in_use": peak_in_use,
        "allocator_peak_bytes_reserved": peak_reserved,
        "sample_cadence_ms": cadence_ms,
        "samples": samples,
        "sampling_misses": misses,
        "sampling_miss_fraction": miss_fraction,
        "sampling_limitations": limitations,
        "measurement_interval_ns": {
            "forecast_allocator": [allocator_start_ns, allocator_end_ns],
            "lock_owner_sampler": [sampler_start_ns, sampler_end_ns],
        },
        "binding": {key: allocator.get(key) for key in binding_keys},
        "product_metric": "lock-owner baseline-subtracted total device residency",
    }


# --------------------------------------------------------------------------- #
# build                                                                        #
# --------------------------------------------------------------------------- #
def build(*, kernels: list[dict[str, Any]] | None,
          mem_rows: list[dict[str, Any]] | None = None,
          integration_scope: dict[str, Any] | None = None,
          production_reference: dict[str, Any] | None = None,
          candidate_coverage: dict[str, Any] | None = None,
          matched_profiler: dict[str, Any] | None = None,
          residency: dict[str, Any] | None = None,
          census_path: Path = A6_CENSUS) -> dict[str, Any]:
    census: dict[str, Any] = {"schema": "wrf_gpu2.v025.m0.baseline_census.v2"}

    if not kernels:
        census["device_time_attribution"] = _missing(
            "no CUDA kernel rows were exported",
            "cuda_gpu_kern_sum from the real .nsys-rep with at least one kernel")
        census["production_representativeness"] = _missing("no attribution", "kernel rows")
        census["a6_static_proxy_rank_diagnostic"] = _missing("no attribution", "kernel rows")
    else:
        attribution = arm.attribute_device_time(kernels, min_attribution=MIN_ATTRIBUTION)
        attribution["status"] = "OK" if attribution["meets_attribution_bar"] else "FAILED"
        if attribution["status"] == "FAILED":
            attribution["reason"] = (
                f"unknown is {attribution['unknown_launch_share']:.1%} of launches and "
                f"{attribution['unknown_device_time_share']:.1%} of device time; §9 caps BOTH "
                f"at {1 - MIN_ATTRIBUTION:.0%}")
        census["device_time_attribution"] = attribution
        if attribution["status"] == "OK":
            census["production_representativeness"] = production_representativeness_gate(
                attribution["families"], production_reference, candidate_coverage
            )
            census["a6_static_proxy_rank_diagnostic"] = a6_static_proxy_diagnostic(
                attribution["families"], static_proxy_shares(census_path)
            )
        else:
            census["production_representativeness"] = _missing(
                "attribution did not meet the §9 bar",
                "a census that attributes itself before representativeness is scored",
            )
            census["a6_static_proxy_rank_diagnostic"] = _missing(
                "attribution did not meet the §9 bar", "attributed kernel rows"
            )

    census["transfer_audit"] = transfer_gate(mem_rows, integration_scope)
    census["profiler_perturbation"] = profiler_gate(matched_profiler)
    census["vram"] = vram_gate(residency)

    failed = [g for g in REQUIRED_GATES if census[g].get("status") != "OK"]
    census["required_gates"] = list(REQUIRED_GATES)
    census["gates_not_ok"] = failed
    census["status"] = "OK" if not failed else "BLOCKED"
    if failed:
        census["reason"] = (
            f"{len(failed)} of {len(REQUIRED_GATES)} required sub-gates are not OK: "
            f"{', '.join(failed)}. The census does not report OK on a subset -- a missing "
            f"sub-gate is an unmeasured one, not a passing one.")
    return census


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels-csv", type=Path, required=True)
    parser.add_argument("--mem-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import parse_profiler as pp
    kernels = pp.parse_nsys_kernel_summary(args.kernels_csv.read_text(errors="replace")) \
        if args.kernels_csv.is_file() else None
    mem = pp.parse_generic_sum(args.mem_csv.read_text(errors="replace")) \
        if args.mem_csv and args.mem_csv.is_file() else None

    census = build(kernels=kernels, mem_rows=mem)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(census, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {args.out}: {census['status']}")
    return 0 if census["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
