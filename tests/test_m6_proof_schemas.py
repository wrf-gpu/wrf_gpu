from __future__ import annotations

from pathlib import Path

from gpuwrf.io.proof_schemas import (
    CoupledDummyCarry,
    Forecast24h,
    ForecastSmoke,
    FullDomainBatchingVerdict,
    Gen2Comparison,
    MilestoneCloseoutM6,
    SCHEMA_REGISTRY,
    SpacetimeBudget,
    Tier2CoupledInvariants,
    Tier3DriftEnvelope,
    Tier4ProbtestTolerances,
    validate_artifact,
)


def test_existing_m6_artifacts_validate_against_registry():
    validate_artifact(Path("artifacts/m6/coupled_dummy_carry.json"))
    validate_artifact(Path("artifacts/m6/spacetime_budget.json"))


def test_all_m6_schema_classes_have_machine_readable_json_schema():
    classes = [
        CoupledDummyCarry,
        SpacetimeBudget,
        ForecastSmoke,
        Forecast24h,
        Tier2CoupledInvariants,
        Tier3DriftEnvelope,
        Tier4ProbtestTolerances,
        Gen2Comparison,
        FullDomainBatchingVerdict,
        MilestoneCloseoutM6,
    ]
    for schema in classes:
        json_schema = schema.json_schema()
        assert json_schema["type"] == "object"
        assert json_schema["required"]
        assert json_schema["properties"]


def test_schema_registry_exposes_expected_artifact_aliases():
    for key in (
        "coupled_dummy_carry",
        "spacetime_budget",
        "forecast_6h_summary",
        "forecast_24h_summary",
        "tier2_coupled_invariants",
        "tsc_envelope",
        "probtest_tolerances",
        "gen2_comparison",
        "full_domain_batching_verdict",
        "milestone_closeout_m6",
    ):
        assert key in SCHEMA_REGISTRY
