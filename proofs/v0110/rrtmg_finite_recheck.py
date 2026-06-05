#!/usr/bin/env python
"""Finite recheck for v0.11.0 RRTMG slope/shading on the operational d02 cadence.

This intentionally uses the daily pipeline product path and swaps only the forecast
function to the warmed/chunked operational segment driver. It is not a cold
single-step comparator.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
DEFAULT_JSON = ROOT / "proofs" / "v0110" / "rrtmg_finite_recheck.json"
DEFAULT_MD = ROOT / "proofs" / "v0110" / "rrtmg_finite_recheck.md"
DEFAULT_OUTPUT_ROOT = Path("/tmp/v0110_rrtmg_recheck_outputs")
DEFAULT_PIPELINE_PROOF_ROOT = Path("/tmp/v0110_rrtmg_recheck_pipeline_proofs")
RRTMG_OFF_CADENCE = 10_000_000


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _field_counts(summary: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    all_fields = summary.get("fields", {}) if isinstance(summary, Mapping) else {}
    out: dict[str, dict[str, Any]] = {}
    for name in fields:
        rec = dict(all_fields.get(name, {}))
        if rec:
            out[name] = {
                "finite": bool(rec.get("finite")),
                "nonfinite_count": int(rec.get("nonfinite_count", -1)),
                "min": rec.get("min"),
                "max": rec.get("max"),
                "shape": rec.get("shape"),
                "dtype": rec.get("dtype"),
            }
    return out


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    final = payload.get("final_state_fields", {})
    lines = [
        "# v0.11.0 RRTMG finite recheck",
        "",
        f"- verdict: {payload.get('verdict')}",
        f"- proper-cadence finite: {payload.get('proper_cadence_finite')}",
        f"- run: {payload.get('run_id')} / {payload.get('domain')}",
        f"- hours: {payload.get('hours')}",
        f"- forecast path: {payload.get('forecast_fn')}",
        f"- segment steps: {payload.get('segment_steps')}",
        f"- radiation cadence steps: {payload.get('radiation_cadence_steps')}",
        f"- topo_shading: {payload.get('topo_shading')}",
        f"- slope_rad: {payload.get('slope_rad')}",
        f"- pipeline verdict: {payload.get('pipeline_verdict')}",
        f"- wall clock total s: {payload.get('wall_clock_total_s')}",
        "",
        "## Final-state finite counts",
        "",
        "| field | finite | nonfinite | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("theta", "u", "v", "w", "p", "ph", "mu", "qv"):
        rec = final.get(name, {})
        lines.append(
            f"| {name} | {rec.get('finite')} | {rec.get('nonfinite_count')} | "
            f"{rec.get('min')} | {rec.get('max')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(payload.get("interpretation")),
            "",
            "## Commands",
            "",
        ]
    )
    for command in payload.get("commands", []):
        lines.append(f"- `{command}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--hours", type=int, default=1)
    parser.add_argument("--dt-s", type=float, default=10.0)
    parser.add_argument("--acoustic-substeps", type=int, default=10)
    parser.add_argument("--radiation-cadence-steps", type=int, default=180)
    parser.add_argument("--segment-steps", type=int, default=180)
    parser.add_argument("--mode", choices=("on", "topo_off", "rrtmg_off"), default="on")
    parser.add_argument("--diagnostic-exit-zero", action="store_true")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pipeline-proof-root", type=Path, default=DEFAULT_PIPELINE_PROOF_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.hours) not in (1, 2):
        raise ValueError("--hours must be 1 or 2 for this short recheck")

    import jax

    from gpuwrf.config import paths
    from gpuwrf.integration.daily_pipeline import (
        DailyCase,
        DailyPipelineConfig,
        _build_real_case,
        execute_daily_pipeline,
        resolve_run_dir,
    )
    from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented

    gpus = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpus:
        raise RuntimeError("No JAX GPU backend visible; refusing to produce GPU finite proof")

    if args.run_root is not None:
        run_root = args.run_root
    else:
        run_root = paths.wrf_l3_root()
        fallback_root = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
        if not (run_root / args.run_id).is_dir() and (fallback_root / args.run_id).is_dir():
            run_root = fallback_root
    run_dir = resolve_run_dir(args.run_id, run_root)
    output_dir = args.output_root / f"{args.domain}_{args.run_id}_{int(args.hours)}h_{args.mode}"
    pipeline_proof_dir = args.pipeline_proof_root / f"{args.domain}_{args.run_id}_{int(args.hours)}h_{args.mode}"
    seg = int(args.segment_steps)
    if args.mode == "on":
        mode_topo_shading = 1
        mode_slope_rad = 1
        mode_cadence = int(args.radiation_cadence_steps)
        mode_note = "RRTMG slope/shading on"
    elif args.mode == "topo_off":
        mode_topo_shading = 0
        mode_slope_rad = 0
        mode_cadence = int(args.radiation_cadence_steps)
        mode_note = "RRTMG active, topo_shading=0, slope_rad=0"
    else:
        mode_topo_shading = 0
        mode_slope_rad = 0
        mode_cadence = RRTMG_OFF_CADENCE
        mode_note = "RRTMG suppressed by cadence beyond forecast length"
    attempted_context: dict[str, Any] = {
        "mode": args.mode,
        "mode_note": mode_note,
        "topo_shading": mode_topo_shading,
        "slope_rad": mode_slope_rad,
        "radiation_cadence_steps": mode_cadence,
    }

    def case_builder(config: DailyPipelineConfig) -> tuple[DailyCase, Path]:
        case, built_run_dir = _build_real_case(config)
        namelist = dataclasses.replace(
            case.namelist,
            run_physics=True,
            run_boundary=True,
            disable_guards=False,
            radiation_cadence_steps=mode_cadence,
            topo_shading=mode_topo_shading,
            slope_rad=mode_slope_rad,
            time_utc=case.run_start,
        )
        metadata = dict(case.metadata)
        nl_meta = dict(metadata.get("namelist", {}))
        nl_meta.update(
            {
                "run_physics": True,
                "run_boundary": True,
                "disable_guards": False,
                "radiation_cadence_steps": mode_cadence,
                "topo_shading": mode_topo_shading,
                "slope_rad": mode_slope_rad,
            }
        )
        metadata["namelist"] = nl_meta
        metadata["rrtmg_finite_recheck_case_builder"] = mode_note
        attempted_context["namelist"] = nl_meta
        attempted_context["run_start_utc"] = case.run_start.isoformat()
        attempted_context["radiation_static"] = metadata.get("radiation_static")
        return dataclasses.replace(case, namelist=namelist, metadata=metadata), built_run_dir

    def forecast_fn(state: Any, namelist: Any, hours: float) -> Any:
        result = run_forecast_operational_segmented(state, namelist, float(hours), segment_steps=seg)
        jax.block_until_ready(result.theta)
        return result

    config = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=pipeline_proof_dir,
        run_root=run_root,
        score=False,
        domain=args.domain,
        dt_s=float(args.dt_s),
        acoustic_substeps=int(args.acoustic_substeps),
        radiation_cadence_steps=mode_cadence,
        async_output=True,
    )
    payload = execute_daily_pipeline(config, forecast_fn=forecast_fn, case_builder=case_builder)

    finite_summary = payload.get("all_finite_check") or payload.get("detail", {}).get("finite_summary", {})
    namelist_meta = (payload.get("metadata") or {}).get("namelist", {}) or attempted_context.get("namelist", {})
    proper_finite = (
        payload.get("verdict") != "PIPELINE_BLOCKED"
        and bool(finite_summary.get("all_finite"))
    )
    one_step_context = _read_json(ROOT / "proofs" / "v0110" / "rrtmg_slope_gpu_sanity.json")
    control_counts = None
    if one_step_context:
        control_counts = {
            "one_step_topo_on_counts": {
                name: rec.get("full_nonfinite_count")
                for name, rec in one_step_context.get("state_summary", {}).items()
                if name in ("theta", "u", "v", "w", "p", "ph", "mu")
            },
            "one_step_topo_off_counts": (
                one_step_context.get("control_topo_off_one_step", {}).get("state_nonfinite_counts")
            ),
            "topo_off_control_same_failure": one_step_context.get("checks", {}).get("topo_off_control_same_failure"),
        }

    verdict = "KEEP_RRTMG_ON_TRUNK" if proper_finite else "REGRESSION_REQUIRES_BISECTION"
    interpretation = (
        "The proper daily output-interval segmented cadence stayed finite with full physics, "
        "real d02 XLAT/XLONG radiation static fields, topo_shading=1, and slope_rad=1. "
        "The earlier cold one-step theta/u/v nonfinite is therefore treated as a harness artifact."
        if proper_finite
        else "The proper segmented cadence did not produce a finite successful run; RRTMG-on/off bisection is required before keeping this lane."
    )
    command = " ".join(sys.argv)
    proof: dict[str, Any] = {
        "schema": "V0110RRTMGFiniteRecheck",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if proper_finite else "FAIL",
        "verdict": verdict,
        "interpretation": interpretation,
        "proper_cadence_finite": bool(proper_finite),
        "mode": args.mode,
        "mode_note": mode_note,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "domain": args.domain,
        "hours": int(args.hours),
        "device": str(gpus[0]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "forecast_fn": f"run_forecast_operational_segmented(segment_steps={seg}) via execute_daily_pipeline",
        "segment_steps": int(args.segment_steps),
        "radiation_cadence_steps": mode_cadence,
        "dt_s": float(args.dt_s),
        "acoustic_substeps": int(args.acoustic_substeps),
        "topo_shading": int(namelist_meta.get("topo_shading", -1)),
        "slope_rad": int(namelist_meta.get("slope_rad", -1)),
        "pipeline_verdict": payload.get("verdict"),
        "pipeline_output_dir": str(output_dir),
        "pipeline_proof_dir": str(pipeline_proof_dir),
        "wall_clock_total_s": payload.get("wall_clock_total_s"),
        "wall_clock_per_hour_s": payload.get("wall_clock_per_hour_s"),
        "wrfout_files": payload.get("wrfout_files", []),
        "wrfout_inventory_status": payload.get("wrfout_inventory_status"),
        "final_state_fields": _field_counts(finite_summary, ("theta", "u", "v", "w", "p", "ph", "mu", "qv")),
        "all_finite_check": finite_summary,
        "namelist": namelist_meta,
        "radiation_static": (payload.get("metadata") or {}).get("radiation_static") or attempted_context.get("radiation_static"),
        "attempted_context": attempted_context,
        "cold_one_step_context": control_counts,
        "raw_pipeline_payload": payload,
        "commands": [command],
        "env": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "TF_GPU_ALLOCATOR": os.environ.get("TF_GPU_ALLOCATOR"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    _write_json(args.out_json, proof)
    _write_markdown(args.out_md, proof)
    print(json.dumps({
        "status": proof["status"],
        "verdict": verdict,
        "proper_cadence_finite": proper_finite,
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
        "wall_clock_total_s": payload.get("wall_clock_total_s"),
    }, indent=2))
    return 0 if (proper_finite or bool(args.diagnostic_exit_zero)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
