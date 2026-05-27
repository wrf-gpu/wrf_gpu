#!/usr/bin/env python
"""Run M7 one-hour physics ON/OFF bracket forecasts and compare to CPU WRF."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpuwrf.integration import daily_pipeline as pipeline  # noqa: E402
from gpuwrf.io.data_inventory import parse_wrfout_valid_time  # noqa: E402
from m7_rca_hour_by_hour import (  # noqa: E402
    DEFAULT_CPU_ROOT,
    DEFAULT_FIELDS,
    SPRINT_DIR,
    compare_field,
    write_json,
    _json_default,
)


DEFAULT_OUTPUT = SPRINT_DIR / "physics_on_off_bracket.json"
DEFAULT_ARTIFACT_DIR = Path("/tmp/m7_rca_artifacts/physics_bracket")
DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T072630Z"


def _pin_orchestration_cpus() -> list[int] | None:
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpus = {0, 1, 2, 3}
    try:
        os.sched_setaffinity(0, cpus)
    except OSError:
        pass
    return sorted(os.sched_getaffinity(0))


def _case_builder(run_physics: bool):
    def build(config: pipeline.DailyPipelineConfig):
        case, run_dir = pipeline._build_real_case(config)  # Diagnostic sprint; production code stays untouched.
        namelist = replace(case.namelist, run_physics=bool(run_physics))
        metadata = dict(case.metadata)
        namelist_metadata = dict(metadata.get("namelist", {}))
        namelist_metadata["run_physics"] = bool(run_physics)
        metadata["namelist"] = namelist_metadata
        metadata["rca_case"] = "physics_on" if run_physics else "physics_off"
        return replace(case, namelist=namelist, metadata=metadata), run_dir

    return build


def _cpu_file_for_forecast(forecast_wrfout: str | Path, cpu_root: str | Path) -> Path:
    valid_time = parse_wrfout_valid_time(Path(forecast_wrfout))
    target_name = f"wrfout_d02_{valid_time:%Y-%m-%d_%H:%M:%S}"
    target = Path(cpu_root) / target_name
    if not target.is_file():
        raise FileNotFoundError(f"missing CPU reference for {forecast_wrfout}: {target}")
    return target


def _compare_to_cpu(forecast_wrfout: str | Path, cpu_root: str | Path, fields: Sequence[str]) -> dict[str, Any]:
    cpu_path = _cpu_file_for_forecast(forecast_wrfout, cpu_root)
    return {
        "gpu_path": str(forecast_wrfout),
        "cpu_path": str(cpu_path),
        "valid_time_utc": parse_wrfout_valid_time(Path(forecast_wrfout)).isoformat(),
        "fields": {field: compare_field(forecast_wrfout, cpu_path, field) for field in fields},
    }


def _run_case(
    *,
    label: str,
    run_physics: bool,
    run_id: str,
    run_root: Path,
    output_dir: Path,
    proof_dir: Path,
    cpu_root: Path,
    fields: Sequence[str],
) -> dict[str, Any]:
    config = pipeline.DailyPipelineConfig(
        run_id=run_id,
        hours=1,
        output_dir=output_dir,
        proof_dir=proof_dir,
        run_root=run_root,
        score=False,
        radiation_cadence_steps=999999,
    )
    payload = pipeline.execute_daily_pipeline(config, case_builder=_case_builder(run_physics))
    wrfout_files = [Path(path) for path in payload.get("wrfout_files", [])]
    comparison = None
    if wrfout_files:
        comparison = _compare_to_cpu(wrfout_files[-1], cpu_root, fields)
    return {
        "label": label,
        "run_physics": bool(run_physics),
        "radiation_cadence_steps": int(config.radiation_cadence_steps),
        "pipeline_payload": payload,
        "comparison_to_cpu": comparison,
    }


def _field_delta(case_a: Mapping[str, Any], case_b: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    a_fields = ((case_a.get("comparison_to_cpu") or {}).get("fields") or {})
    b_fields = ((case_b.get("comparison_to_cpu") or {}).get("fields") or {})
    deltas: dict[str, Any] = {}
    for field in fields:
        a = a_fields.get(field, {})
        b = b_fields.get(field, {})
        if a.get("status") != "OK" or b.get("status") != "OK":
            deltas[field] = {"status": "MISSING_OR_INVALID", "physics_on_status": a.get("status"), "physics_off_status": b.get("status")}
            continue
        deltas[field] = {
            "status": "OK",
            "physics_on_max_abs_diff": a.get("max_abs_diff"),
            "physics_off_max_abs_diff": b.get("max_abs_diff"),
            "physics_on_mean_diff": a.get("mean_diff"),
            "physics_off_mean_diff": b.get("mean_diff"),
            "off_minus_on_max_abs_diff": float(b["max_abs_diff"]) - float(a["max_abs_diff"]),
            "off_minus_on_abs_mean_diff": abs(float(b["mean_diff"])) - abs(float(a["mean_diff"])),
        }
    return deltas


def build_payload(
    *,
    run_id: str,
    run_root: Path,
    cpu_root: Path,
    artifact_dir: Path,
    fields: Sequence[str],
) -> dict[str, Any]:
    affinity = _pin_orchestration_cpus()
    on_case = _run_case(
        label="physics_on_radiation_effectively_off",
        run_physics=True,
        run_id=run_id,
        run_root=run_root,
        output_dir=artifact_dir / "physics_on_radiation_off",
        proof_dir=artifact_dir / "proof_physics_on_radiation_off",
        cpu_root=cpu_root,
        fields=fields,
    )
    off_case = _run_case(
        label="physics_off_dynamics_only",
        run_physics=False,
        run_id=run_id,
        run_root=run_root,
        output_dir=artifact_dir / "physics_off",
        proof_dir=artifact_dir / "proof_physics_off",
        cpu_root=cpu_root,
        fields=fields,
    )
    deltas = _field_delta(on_case, off_case, fields)
    better_off = [
        field
        for field, stats in deltas.items()
        if stats.get("status") == "OK"
        and stats.get("physics_off_max_abs_diff") is not None
        and stats.get("physics_on_max_abs_diff") is not None
        and float(stats["physics_off_max_abs_diff"]) < float(stats["physics_on_max_abs_diff"])
    ]
    verdict = "PHYSICS_BRACKET_COMPLETE"
    if on_case["pipeline_payload"].get("verdict") == "PIPELINE_BLOCKED" or off_case["pipeline_payload"].get("verdict") == "PIPELINE_BLOCKED":
        verdict = "PHYSICS_BRACKET_BLOCKED"
    return {
        "schema": "M7SkillRegressionPhysicsOnOffBracket",
        "schema_version": 1,
        "verdict": verdict,
        "run_id": run_id,
        "run_root": str(run_root),
        "cpu_root": str(cpu_root),
        "artifact_dir": str(artifact_dir),
        "cpu_affinity": affinity,
        "fields": list(fields),
        "cases": [on_case, off_case],
        "field_delta_summary": deltas,
        "fields_where_physics_off_reduced_max_abs_diff": better_off,
        "interpretation": (
            "If physics-off is materially closer to CPU than physics-on, physics coupling is implicated. "
            "If both are similarly wrong at lead 1, dycore, boundary, or initialization is implicated."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=pipeline.RUN_ROOT)
    parser.add_argument("--cpu-root", type=Path, default=DEFAULT_CPU_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        run_id=args.run_id,
        run_root=args.run_root,
        cpu_root=args.cpu_root,
        artifact_dir=args.artifact_dir,
        fields=args.fields,
    )
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": payload["verdict"],
                "artifact_dir": str(args.artifact_dir),
                "fields_where_physics_off_reduced_max_abs_diff": payload["fields_where_physics_off_reduced_max_abs_diff"],
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0 if payload["verdict"] == "PHYSICS_BRACKET_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
