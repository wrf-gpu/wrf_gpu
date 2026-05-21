"""Machine-readable proof-object schemas for M6 sprint artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, ClassVar


JsonType = str


@dataclass(frozen=True)
class FieldRule:
    """One JSON-schema field rule with a short human-facing description."""

    json_type: JsonType | tuple[JsonType, ...]
    description: str

    def as_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"description": self.description}
        schema["type"] = list(self.json_type) if isinstance(self.json_type, tuple) else self.json_type
        return schema


class ProofObjectSchema:
    """Small dataclass-backed schema helper used instead of adding pydantic."""

    schema_name: ClassVar[str]
    description: ClassVar[str]
    required: ClassVar[dict[str, FieldRule]]
    optional: ClassVar[dict[str, FieldRule]] = {}

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        properties = {name: rule.as_json_schema() for name, rule in {**cls.required, **cls.optional}.items()}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": cls.schema_name,
            "description": cls.description,
            "type": "object",
            "required": list(cls.required),
            "properties": properties,
            "additionalProperties": True,
        }

    @classmethod
    def validate_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError(f"{cls.schema_name} must be a JSON object")
        for name, rule in cls.required.items():
            if name not in data:
                raise ValueError(f"{cls.schema_name} missing required field {name!r}")
            _assert_json_type(cls.schema_name, name, data[name], rule.json_type)
        for name, rule in cls.optional.items():
            if name in data:
                _assert_json_type(cls.schema_name, name, data[name], rule.json_type)
        return data

    @classmethod
    def validate_file(cls, path: str | Path) -> dict[str, Any]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.validate_dict(data)


def _assert_json_type(schema_name: str, field: str, value: Any, json_type: JsonType | tuple[JsonType, ...]) -> None:
    allowed = (json_type,) if isinstance(json_type, str) else json_type
    if any(_matches_json_type(value, item) for item in allowed):
        return
    raise TypeError(f"{schema_name}.{field} expected {allowed}, got {type(value).__name__}")


def _matches_json_type(value: Any, json_type: JsonType) -> bool:
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "null":
        return value is None
    raise ValueError(f"unsupported schema type {json_type!r}")


class CoupledDummyCarry(ProofObjectSchema):
    """M6-S1 proof that the coupled dummy carry stays device-resident."""

    schema_name = "CoupledDummyCarry"
    description = "M6-S1 coupled dummy carry transfer and launch proof."
    required = {
        "domain": FieldRule("array", "Domain dimensions as [nx, ny, nz]."),
        "steps": FieldRule("integer", "Number of dummy coupled steps."),
        "host_to_device_bytes_post_init": FieldRule("integer", "H2D bytes after initialization."),
        "device_to_host_bytes_post_init": FieldRule("integer", "D2H bytes after initialization."),
        "temporary_bytes_per_step": FieldRule("integer", "Temporary allocation bytes per step."),
        "wall_time_per_step_ms": FieldRule("number", "Median wall time per step in ms."),
        "kernel_launches_per_step": FieldRule("integer", "HLO-derived launches per step."),
        "hlo_bytes": FieldRule("integer", "Compiled HLO text size in bytes."),
    }
    optional = {
        "trace_dir": FieldRule("string", "Profiler trace directory."),
        "trace_transfer_event_files": FieldRule("array", "Raw transfer-audit files."),
    }


class SpacetimeBudget(ProofObjectSchema):
    """M6-S1 proof object summarizing per-kernel cost and transfer budget."""

    schema_name = "SpacetimeBudget"
    description = "M6 spacetime budget and transfer accounting artifact."
    required = {
        "benchmark": FieldRule("string", "Benchmark identifier."),
        "backend": FieldRule("string", "Execution backend."),
        "case": FieldRule("string", "Case identifier."),
        "host_device_transfer_bytes": FieldRule("integer", "Total post-init host/device transfer bytes."),
        "temporary_bytes_per_step": FieldRule("integer", "Temporary allocation bytes per step."),
        "total_per_step_ms": FieldRule("number", "Total per-step wall time."),
        "per_kernel": FieldRule("object", "Per-kernel cost records."),
        "artifact_paths": FieldRule("array", "Related proof-object paths."),
    }


class ForecastSmoke(ProofObjectSchema):
    """M6-S2 short forecast smoke proof."""

    schema_name = "ForecastSmoke"
    description = "Short coupled forecast smoke result with transfer and boundary metadata."
    required = {
        "run_id": FieldRule("string", "Forecast run identifier."),
        "domain": FieldRule("string", "WRF/GPU domain ID."),
        "lead_hours": FieldRule("number", "Forecast lead length in hours."),
        "status": FieldRule("string", "PASS/FAIL/BLOCKED status."),
        "boundary_artifact": FieldRule("string", "Boundary replay fixture or source path."),
        "host_device_transfer_bytes_post_init": FieldRule("integer", "Post-init transfer bytes."),
        "artifact_paths": FieldRule("array", "Related proof-object paths."),
    }


class Forecast24h(ProofObjectSchema):
    """M6-S2 24-hour d02 forecast proof."""

    schema_name = "Forecast24h"
    description = "Full 24h d02 coupled forecast summary and correctness envelope."
    required = {
        "run_id": FieldRule("string", "Forecast run identifier."),
        "domain": FieldRule("string", "Domain ID."),
        "lead_hours": FieldRule("number", "Forecast lead length."),
        "status": FieldRule("string", "PASS/FAIL/BLOCKED status."),
        "boundary_artifact": FieldRule("string", "Boundary replay fixture path."),
        "output_manifest": FieldRule("string", "Forecast output manifest path."),
        "artifact_paths": FieldRule("array", "Related proof-object paths."),
    }


class Tier2CoupledInvariants(ProofObjectSchema):
    """M6-S4 invariant proof with source/sink/boundary accounting."""

    schema_name = "Tier2CoupledInvariants"
    description = "Tier-2 coupled conservation and positivity proof."
    required = {
        "run_id": FieldRule("string", "Validation run identifier."),
        "domain": FieldRule("string", "WRF/GPU domain ID."),
        "status": FieldRule("string", "PASS/FAIL/BLOCKED status."),
        "budgets": FieldRule("object", "Dry-mass, water, positivity, and energy budget records."),
        "per_step": FieldRule("array", "Per-step per-leaf residual table."),
        "thresholds": FieldRule("object", "Binding AC6 thresholds and pass/fail state."),
        "boundary_terms": FieldRule("object", "Boundary-flux terms used by the independent oracle."),
        "artifact_paths": FieldRule("array", "Raw and summary proof paths."),
    }
    optional = {
        "sanitize_policy": FieldRule("object", "PRE-sanitize tap or sanitize-OFF policy evidence."),
        "gen2_pin": FieldRule("object", "Pinned Gen2 run path and history inventory."),
    }


class Tier3DriftEnvelope(ProofObjectSchema):
    """M6-S6 timestep/drift envelope proof."""

    schema_name = "Tier3DriftEnvelope"
    description = "Tier-3 controlled short-run convergence and drift envelope."
    required = {
        "run_id": FieldRule("string", "Validation run identifier."),
        "domain": FieldRule("string", "WRF/GPU domain ID for the pinned drift comparison."),
        "status": FieldRule("string", "GREEN/PARTIAL/BLOCKED/FAIL status."),
        "base_dt_s": FieldRule("number", "Base timestep in seconds."),
        "refined_dt_s": FieldRule("number", "Refined timestep in seconds."),
        "further_refined_dt_s": FieldRule("number", "Second refinement timestep in seconds."),
        "lead_hours": FieldRule("array", "Lead times evaluated by the envelope."),
        "variables": FieldRule("array", "Variables evaluated by the envelope."),
        "boundary_mode": FieldRule("object", "Boundary forcing mode and provenance."),
        "forcing_mode": FieldRule("object", "Physics/forcing cadence used by reduced and pinned runs."),
        "regridding": FieldRule("object", "Regridding and staggering policy."),
        "norm_definitions": FieldRule("object", "Definitions for max_abs/rmse/mean_abs norms."),
        "envelope_derivation": FieldRule("object", "Controlled dt-refinement and CPU/analytic reference derivation."),
        "envelope": FieldRule("object", "Per-variable per-lead dt-sensitivity envelope."),
        "gpu_drift": FieldRule("object", "Per-variable per-lead GPU-vs-reference drift."),
        "per_variable_status": FieldRule("object", "GREEN/PARTIAL/FAIL/BLOCKED status by variable and lead."),
        "artifact_paths": FieldRule("array", "Raw and summary proof paths."),
    }
    optional = {
        "thompson_water_budget_oracle": FieldRule("object", "F-min-1 independent Thompson water-budget side-channel proof."),
        "wrfbdy_boundary_oracle": FieldRule("object", "F-min-2 wrfbdy decoder comparison proof."),
    }


class Tier4ProbtestTolerances(ProofObjectSchema):
    """M6-S7 statistical tolerance proof."""

    schema_name = "Tier4ProbtestTolerances"
    description = "Tier-4 probtest-style tolerance freeze artifact."
    required = {
        "run_id": FieldRule("string", "Tolerance-generation run identifier."),
        "status": FieldRule("string", "PASS/FAIL/BLOCKED status."),
        "member_manifest": FieldRule("string", "Ensemble or historical-member manifest path."),
        "tolerances": FieldRule("object", "Per-variable and per-lead tolerances."),
        "freeze_time_utc": FieldRule("string", "Time when tolerances were frozen."),
        "artifact_paths": FieldRule("array", "Raw and summary proof paths."),
    }


class Gen2Comparison(ProofObjectSchema):
    """M6-S8 operational GPU-vs-Gen2 comparison proof."""

    schema_name = "Gen2Comparison"
    description = "Operational comparison against pinned Gen2 WRF backfill truth."
    required = {
        "run_id": FieldRule("string", "Pinned Gen2 run identifier."),
        "domain": FieldRule("string", "Domain ID."),
        "status": FieldRule("string", "GREEN/PARTIAL/BLOCKED/FAIL status."),
        "variables": FieldRule("object", "Per-variable RMSE/bias/lead records."),
        "binding_gate": FieldRule("string", "Operational gate policy."),
        "artifact_paths": FieldRule("array", "Raw and summary proof paths."),
    }


class FullDomainBatchingVerdict(ProofObjectSchema):
    """M6-S5 ADR-007 full-domain performance verdict proof."""

    schema_name = "FullDomainBatchingVerdict"
    description = "Full-domain batching speedup verdict against fair CPU denominator."
    required = {
        "benchmark": FieldRule("string", "Benchmark identifier."),
        "backend": FieldRule("string", "Execution backend."),
        "hardware": FieldRule("string", "Hardware description."),
        "case": FieldRule("string", "Case identifier."),
        "wall_time_s": FieldRule("number", "Measured wall time."),
        "host_device_transfer_bytes": FieldRule("integer", "Total transfer bytes."),
        "cpu_denominator_artifact": FieldRule("string", "Fair CPU denominator JSON path."),
        "verdict": FieldRule("string", "PASS/FAIL/BLOCKED verdict."),
        "artifact_paths": FieldRule("array", "Raw and summary proof paths."),
    }
    optional = {
        "kernel_launches": FieldRule(("integer", "null"), "Kernel launch count, null if profiler unavailable."),
        "occupancy_pct": FieldRule(("number", "null"), "Profiler occupancy, null if unavailable."),
        "registers_per_thread": FieldRule(("integer", "null"), "Register count, null if unavailable."),
        "local_memory_bytes": FieldRule(("integer", "null"), "Local-memory bytes, null if unavailable."),
    }


class SurfaceLayerArtifact(ProofObjectSchema):
    """M6-S3 surface-layer, land-state, and operational-delta proof object."""

    schema_name = "SurfaceLayerArtifact"
    description = "M6-S3 surface-layer proof object covering land state, radiation feasibility, and deltas."
    required = {
        "artifact_type": FieldRule("string", "Surface-layer proof subtype."),
        "status": FieldRule("string", "PASS/PARTIAL/BLOCKED/FAIL status."),
        "run_id": FieldRule("string", "Pinned Gen2 run identifier."),
        "domain": FieldRule("string", "Domain ID."),
        "artifact_paths": FieldRule("array", "Related proof-object paths."),
    }
    optional = {
        "variables": FieldRule("object", "Variable inventory or per-variable validation payload."),
        "operational_delta": FieldRule("object", "Per-variable and per-lead operational delta metrics."),
        "prerequisites": FieldRule("object", "F-S3 prerequisite evidence."),
    }


class MilestoneCloseoutM6(ProofObjectSchema):
    """M6 closeout proof index."""

    schema_name = "MilestoneCloseoutM6"
    description = "M6 manager closeout proof index and final dispatch decision state."
    required = {
        "milestone": FieldRule("string", "Milestone identifier."),
        "status": FieldRule("string", "GREEN/PARTIAL/BLOCKED/FAIL status."),
        "proof_index": FieldRule("object", "Artifact paths grouped by sprint."),
        "blocking_risks": FieldRule("array", "Remaining blocking risks."),
        "next_decision": FieldRule("string", "Manager/human decision needed next."),
    }


SCHEMA_REGISTRY: dict[str, type[ProofObjectSchema]] = {
    "coupled_dummy_carry": CoupledDummyCarry,
    "coupled_dummy_carry.json": CoupledDummyCarry,
    "spacetime_budget": SpacetimeBudget,
    "spacetime_budget.json": SpacetimeBudget,
    "forecast_smoke": ForecastSmoke,
    "forecast_6h_summary": ForecastSmoke,
    "forecast_24h": Forecast24h,
    "forecast_24h_summary": Forecast24h,
    "tier2_coupled_invariants": Tier2CoupledInvariants,
    "tier3_drift_envelope": Tier3DriftEnvelope,
    "tsc_envelope": Tier3DriftEnvelope,
    "tier4_probtest_tolerances": Tier4ProbtestTolerances,
    "probtest_tolerances": Tier4ProbtestTolerances,
    "gen2_comparison": Gen2Comparison,
    "full_domain_batching_verdict": FullDomainBatchingVerdict,
    "surface_layer_artifact": SurfaceLayerArtifact,
    "radiation_conditioning_feasibility": SurfaceLayerArtifact,
    "surface_operational_delta": SurfaceLayerArtifact,
    "land_state_manifest": SurfaceLayerArtifact,
    "milestone_closeout_m6": MilestoneCloseoutM6,
}


def schema_for_artifact(path: str | Path) -> type[ProofObjectSchema]:
    name = Path(path).name
    stem = Path(path).stem
    if name in SCHEMA_REGISTRY:
        return SCHEMA_REGISTRY[name]
    if stem in SCHEMA_REGISTRY:
        return SCHEMA_REGISTRY[stem]
    raise KeyError(f"no M6 proof schema registered for {path}")


def validate_artifact(path: str | Path) -> dict[str, Any]:
    schema = schema_for_artifact(path)
    return schema.validate_file(path)


__all__ = [
    "CoupledDummyCarry",
    "Forecast24h",
    "ForecastSmoke",
    "FullDomainBatchingVerdict",
    "Gen2Comparison",
    "MilestoneCloseoutM6",
    "SCHEMA_REGISTRY",
    "SpacetimeBudget",
    "SurfaceLayerArtifact",
    "Tier2CoupledInvariants",
    "Tier3DriftEnvelope",
    "Tier4ProbtestTolerances",
    "schema_for_artifact",
    "validate_artifact",
]
