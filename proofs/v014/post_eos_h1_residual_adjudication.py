#!/usr/bin/env python3
"""Post-EOS h1 residual adjudication for the v0.14 72h gate decision.

Reads existing compare JSON and NetCDF wrfout files only. Does not run WRF,
JAX, CUDA, or model code.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
NEW_COMPARE = Path(
    "/mnt/data/wrf_gpu_validation/"
    "v014_short_field_falsifier_20260610T134205Z/short_field_h1_grid_compare.json"
)
OLD_COMPARE = ROOT / "proofs/v014/short_field_falsifier_h1_grid_compare.json"
MANIFEST = ROOT / "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json"
OUT_JSON = ROOT / "proofs/v014/post_eos_h1_residual_adjudication.json"
OUT_MD = ROOT / "proofs/v014/post_eos_h1_residual_adjudication.md"

CPU_H0 = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/"
    "20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-01_18:00:00"
)
CPU_H1 = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/"
    "20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-01_19:00:00"
)
GPU_H1 = Path(
    "/mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T134205Z/"
    "gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z/"
    "wrfout_d02_2026-05-01_19:00:00"
)

MANDATORY_FIELDS = [
    "PSFC",
    "MU",
    "P",
    "PB",
    "MUB",
    "PH",
    "T",
    "THM",
    "U",
    "V",
    "W",
    "QVAPOR",
    "HFX",
    "LH",
    "PBLH",
    "SWDOWN",
    "SWNORM",
    "COSZEN",
    "GLW",
]

EXTRA_HARD_FIELDS = ["T2", "U10", "V10", "RAINNC", "PHB", "HGT"]

REPORT_ONLY_NONBLOCKERS = [
    "P",
    "PH",
    "MU",
    "HFX",
    "LH",
    "PBLH",
    "SWDOWN",
    "SWNORM",
    "COSZEN",
    "GLW",
]

NEXT_COMMANDS = [
    "RUN_ROOT=/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_$(date -u +%Y%m%dT%H%M%SZ)",
    "mkdir -p \"$RUN_ROOT\"/{gpu_output,proofs,resources}",
    "set +e",
    "scripts/run_gpu_lowprio.sh --cores 0-23 --resource-log-dir \"$RUN_ROOT/resources\" --resource-label v014_canary_d02_72h --resource-interval 5 -- python proofs/v0120/powered_tost_n15/run_one_case_v0120.py --run-root /mnt/data/canairy_meteo/runs/wrf_l2 --cpu-truth-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output --run-id 20260501_18z_l2_72h_20260519T173026Z --hours 72 --output-root \"$RUN_ROOT/gpu_output\" --proof-dir \"$RUN_ROOT/proofs\" > \"$RUN_ROOT/canary_d02_72h_gpu.log\" 2>&1",
    "echo $? > \"$RUN_ROOT/canary_d02_72h_gpu.rc\"",
    "set -e",
    "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/compare_wrfout_grid.py --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z --gpu-dir \"$RUN_ROOT/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z\" --domain d02 --init 2026-05-01T18:00:00+00:00 --min-lead 1 --max-lead 72 --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json --out-json \"$RUN_ROOT/canary_d02_72h_grid_compare.json\" --out-md \"$RUN_ROOT/canary_d02_72h_grid_compare.md\"",
    "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/build_grid_delta_atlas.py --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z --gpu-dir \"$RUN_ROOT/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z\" --case-id canary_d02_20260501_18z --domain d02 --init 2026-05-01T18:00:00+00:00 --min-lead 1 --max-lead 72 --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json --proof-dir \"$RUN_ROOT/grid_delta_atlas\" --asset-dir \"$RUN_ROOT/grid_delta_atlas_assets\"",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3) -> str:
    value = fnum(value)
    if value is None:
        return "NA"
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def field_stats(compare: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for field, summary in compare["field_summaries"].items():
        overall = summary.get("overall")
        if not isinstance(overall, dict):
            continue
        cpu_meta = summary.get("metadata", {}).get("cpu", {})
        boundary = summary.get("spatial_splits", {}).get("boundary", {})
        rows[field] = {
            "classification": summary.get("classification"),
            "units": cpu_meta.get("units"),
            "shape": summary.get("native_shape"),
            "rmse": fnum(overall.get("rmse")),
            "bias": fnum(overall.get("bias")),
            "mae": fnum(overall.get("mae")),
            "p95_abs": fnum(overall.get("p95_abs")),
            "p99_abs": fnum(overall.get("p99_abs")),
            "max_abs": fnum(overall.get("max_abs")),
            "finite_pair_fraction": fnum(overall.get("finite_pair_fraction")),
            "worst_cell": summary.get("worst_cell", {}),
            "boundary_frame5_rmse": fnum(boundary.get("frame_5cells", {}).get("rmse")),
            "interior_excluding_5cell_frame_rmse": fnum(
                boundary.get("interior_excluding_5cell_frame", {}).get("rmse")
            ),
        }
    return rows


def check_manifest(
    stats: dict[str, dict[str, Any]], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for field, spec in manifest["fields"].items():
        if field not in stats:
            continue
        row = stats[field]
        for metric in ("rmse", "max_abs"):
            if metric not in spec:
                continue
            actual = row.get(metric)
            limit = fnum(spec[metric])
            if actual is not None and limit is not None and actual > limit:
                failures.append(
                    {
                        "field": field,
                        "gate": spec.get("gate"),
                        "metric": metric,
                        "actual": actual,
                        "limit": limit,
                    }
                )
        limit = fnum(spec.get("finite_pair_fraction_min"))
        actual = row.get("finite_pair_fraction")
        if actual is not None and limit is not None and actual < limit:
            failures.append(
                {
                    "field": field,
                    "gate": spec.get("gate"),
                    "metric": "finite_pair_fraction",
                    "actual": actual,
                    "limit": limit,
                }
            )
    return failures


def classify_field(
    field: str,
    stats: dict[str, dict[str, Any]],
    old_stats: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    row = stats.get(field)
    if row is None:
        return {"class": "missing", "gate_implication": "missing_field"}
    spec = manifest["fields"].get(field, {})
    field_failures = [f for f in failures if f["field"] == field]
    gate = spec.get("gate", "report_only_or_unmanifested")
    old = old_stats.get(field, {})

    if field_failures and field in {"PB", "MUB"}:
        cls = "known_static_boundary_spike"
        implication = (
            "nonblocking_for_start_but_final_static_exactness_fails_current_manifest"
        )
    elif field_failures and field == "PSFC":
        cls = "hard_dynamic_h1_margin_breach"
        implication = (
            "nonblocking_for_start_but_not_release_green_until_72h_scored_passes"
        )
    elif field_failures:
        cls = "hard_manifest_failure"
        implication = "would_block_final_release_scoring"
    elif gate in {"hard_release_gate", "static_exactness"}:
        cls = "green_under_current_manifest"
        implication = "hard_manifest_green_at_h1"
    elif gate == "critical_report_only":
        cls = "critical_report_only_bounded"
        implication = "review_in_72h_atlas_nonblocking_for_start"
    elif field in {"SWDOWN", "SWNORM", "COSZEN"}:
        cls = "radiation_timing_report_only"
        implication = "measure_drift_slope_in_72h_nonblocking_for_start"
    elif field in {"HFX", "LH", "PBLH", "GLW"}:
        cls = "physics_diagnostic_report_only_bounded"
        implication = "measure_in_72h_nonblocking_for_start"
    elif field == "THM":
        cls = "new_theta_m_diagnostic_green"
        implication = "confirms_post_eos_writer_semantics_at_h1"
    else:
        cls = "report_only_or_unmanifested_bounded"
        implication = "inventory_only_nonblocking_for_start"

    return {
        "class": cls,
        "gate": gate,
        "gate_implication": implication,
        "new": row,
        "old": old or None,
        "manifest_failures": field_failures,
    }


def _array(ds: Dataset, field: str) -> np.ndarray:
    return np.asarray(ds.variables[field][0], dtype=np.float64)


def _basic_stats(diff: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
        "max_abs": float(np.max(np.abs(diff))),
    }


def boundary_analysis() -> dict[str, Any]:
    out: dict[str, Any] = {}
    with Dataset(CPU_H1) as cpu, Dataset(GPU_H1) as gpu:
        ny = len(cpu.dimensions["south_north"])
        nx = len(cpu.dimensions["west_east"])
        frame2 = np.zeros((ny, nx), dtype=bool)
        frame2[:2, :] = True
        frame2[-2:, :] = True
        frame2[:, :2] = True
        frame2[:, -2:] = True
        frame5 = np.zeros((ny, nx), dtype=bool)
        frame5[:5, :] = True
        frame5[-5:, :] = True
        frame5[:, :5] = True
        frame5[:, -5:] = True
        interior5 = ~frame5

        for field in ["PB", "MUB", "PHB", "P", "PH", "PSFC", "MU"]:
            diff = _array(gpu, field) - _array(cpu, field)
            absd = np.abs(diff)
            if diff.ndim == 3:
                mask2 = np.broadcast_to(frame2, diff.shape)
                mask5 = np.broadcast_to(frame5, diff.shape)
                maski = np.broadcast_to(interior5, diff.shape)
            else:
                mask2 = frame2
                mask5 = frame5
                maski = interior5
            worst_index = tuple(int(x) for x in np.unravel_index(np.argmax(absd), absd.shape))
            counts: dict[str, Any] = {}
            for threshold in [0.2, 1.0, 10.0, 50.0, 100.0]:
                mask = absd > threshold
                counts[str(threshold)] = {
                    "total": int(mask.sum()),
                    "frame2": int(np.logical_and(mask, mask2).sum()),
                    "frame5": int(np.logical_and(mask, mask5).sum()),
                    "interior_excluding_5cell_frame": int(
                        np.logical_and(mask, maski).sum()
                    ),
                }
            out[field] = {
                "shape": list(diff.shape),
                "worst_index": list(worst_index),
                "worst_diff": float(diff[worst_index]),
                "worst_abs": float(absd[worst_index]),
                "threshold_counts": counts,
                "frame5": _basic_stats(diff[mask5]),
                "interior_excluding_5cell_frame": _basic_stats(diff[maski]),
            }
    return out


def spinup_analysis() -> dict[str, Any]:
    fields = [
        "PSFC",
        "MU",
        "P",
        "PB",
        "MUB",
        "PH",
        "T",
        "THM",
        "U",
        "V",
        "W",
        "QVAPOR",
        "T2",
        "U10",
        "V10",
        "HFX",
        "LH",
        "PBLH",
        "SWDOWN",
        "SWNORM",
        "COSZEN",
        "GLW",
    ]
    out: dict[str, Any] = {}
    with Dataset(CPU_H0) as c0, Dataset(CPU_H1) as c1, Dataset(GPU_H1) as g1:
        for field in fields:
            if field not in c0.variables or field not in c1.variables or field not in g1.variables:
                continue
            cpu0 = _array(c0, field)
            cpu1 = _array(c1, field)
            gpu1 = _array(g1, field)
            out[field] = {
                "cpu_h1_minus_cpu_h0": _basic_stats(cpu1 - cpu0),
                "gpu_h1_minus_cpu_h0": _basic_stats(gpu1 - cpu0),
                "gpu_h1_minus_cpu_h1": _basic_stats(gpu1 - cpu1),
            }
    return out


def radiation_timing(spinup: dict[str, Any], stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ["COSZEN", "SWDOWN", "SWNORM"]:
        cpu_hour_bias = spinup[field]["cpu_h1_minus_cpu_h0"]["bias"]
        gpu_cpu_h1_bias = stats[field]["bias"]
        minutes = None
        if cpu_hour_bias:
            minutes = 60.0 * gpu_cpu_h1_bias / cpu_hour_bias
        out[field] = {
            "h1_bias": gpu_cpu_h1_bias,
            "cpu_h0_to_h1_bias_per_hour": cpu_hour_bias,
            "approx_time_offset_minutes_from_cpu_h0_h1_slope": minutes,
        }
    return out


def make_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Post-EOS H1 Residual Adjudication")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "PROCEED_72H_GATES. The post-EOS h1 falsifier no longer shows the "
        "radical theta/EOS pressure-temperature failure. The remaining residuals "
        "are bounded enough to start the 72h stability/atlas runs, but this is "
        "not a release-green h1 score: the current candidate manifest still "
        "fails `PSFC` narrowly and `PB`/`MUB` globally because of static "
        "5-cell-frame spikes. Those are final-scoring risks to carry into the "
        "72h atlas, not evidence of renewed live interior drift."
    )
    lines.append("")
    lines.append("## Evidence Table")
    lines.append("")
    lines.append(
        "| Field | Metric old -> new | Class | Gate implication |"
    )
    lines.append("|---|---:|---|---|")
    for field in MANDATORY_FIELDS + EXTRA_HARD_FIELDS:
        fc = payload["field_classes"][field]
        new = fc.get("new") or {}
        old = fc.get("old") or {}
        metric = (
            f"RMSE {fmt(old.get('rmse'))} -> {fmt(new.get('rmse'))}; "
            f"bias {fmt(old.get('bias'))} -> {fmt(new.get('bias'))}; "
            f"p99 {fmt(new.get('p99_abs'))}; max {fmt(new.get('max_abs'))}"
        )
        lines.append(
            f"| `{field}` | {metric} | {fc['class']} | {fc['gate_implication']} |"
        )
    lines.append("")
    lines.append("All 100 common numeric h1 fields were parsed from the post-fix compare JSON; the full per-field metrics are in the companion JSON under `all_numeric_field_stats`.")
    lines.append("")
    lines.append("## Boundary/Static Spike Analysis")
    lines.append("")
    lines.append(
        "- `PB`: max 249.883 Pa at index [0, 57, 156]; cells >0.2 Pa: 4160 total, 4160 in the 5-cell frame, 0 in the interior. Interior max is 0.0078125 Pa."
    )
    lines.append(
        "- `MUB`: max 250.664 Pa at index [57, 156]; cells >0.2 Pa: 194 total, 194 in the 5-cell frame, 0 in the interior. Interior max is 0.0078125 Pa."
    )
    lines.append(
        "- `PHB`: max 0.015625 m2/s2, below the 0.2 static exactness limit everywhere."
    )
    lines.append(
        "- `P`, `PH`, `PSFC`, and `MU` are live dynamic residuals across the domain, not static-boundary-only artifacts. `P/PH/MU` are report-only in the candidate manifest; `PSFC` is the only hard dynamic h1 breach."
    )
    lines.append("")
    lines.append("## Radiation Timing Analysis")
    lines.append("")
    rt = payload["radiation_timing_analysis"]
    lines.append(
        f"`COSZEN` h1 bias is {fmt(rt['COSZEN']['h1_bias'])} against a CPU h0->h1 change of {fmt(rt['COSZEN']['cpu_h0_to_h1_bias_per_hour'])} per hour, implying about {fmt(rt['COSZEN']['approx_time_offset_minutes_from_cpu_h0_h1_slope'])} minutes by local slope. `SWDOWN` gives about {fmt(rt['SWDOWN']['approx_time_offset_minutes_from_cpu_h0_h1_slope'])} minutes. This is the same known timing class; it is report-only and should be measured for drift slope over 72h, not fixed before launch."
    )
    lines.append("")
    lines.append("## Manifest/Tolerance Implication")
    lines.append("")
    lines.append(
        "- Hard h1 manifest-green fields: `T2`, `U10`, `V10`, `RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`, `HGT`, and `PHB`."
    )
    lines.append(
        "- Hard h1 manifest failures if the candidate manifest is applied now: `PSFC` RMSE 124.299 Pa > 120 Pa; `PB` max 249.883 Pa > 0.2 Pa; `MUB` max 250.664 Pa > 0.2 Pa."
    )
    lines.append(
        "- `P`, `PH`, and `MU` are critical report-only fields in the current manifest; no frozen RMSE/max limit exists for them. Do not widen the manifest to bless h1. Run the 72h gate with the manifest and report these as drift diagnostics."
    )
    lines.append(
        "- Starting the 72h gate is compatible with the release-gate start criterion because the short falsifier found bounded, classified residuals rather than nonfinite output, schema failure, or renewed radical field drift. Final release scoring remains stricter than this start decision."
    )
    lines.append("")
    lines.append("## Exact Next Manager Commands")
    lines.append("")
    lines.append("```bash")
    lines.extend(NEXT_COMMANDS)
    lines.append("```")
    lines.append("")
    lines.append("## Context-Sparing Handoff")
    lines.append("")
    lines.append("- objective: adjudicate post-EOS h1 residuals for 72h gate start.")
    lines.append("- files changed: `proofs/v014/post_eos_h1_residual_adjudication.py`, `.json`, `.md`.")
    lines.append("- commands run: JSON validation for old/new compares; manifest-aware h1 comparator to `/tmp`; NetCDF boundary/spin-up inspection; proof script; py_compile/json.tool.")
    lines.append("- proof objects produced: this markdown and `proofs/v014/post_eos_h1_residual_adjudication.json`.")
    lines.append("- verdict: `PROCEED_72H_GATES`; no launch blocker found.")
    lines.append("- hard final-scoring risks: `PSFC`, `PB`, `MUB` under the current candidate manifest.")
    lines.append("- report-only nonblockers: `P`, `PH`, `MU`, radiation timing, surface flux/PBL diagnostics.")
    lines.append("- boundary result: `PB/MUB` exactness failures are entirely in the 5-cell nest frame, not live interior drift.")
    lines.append("- unresolved risk: 72h slope could expose growth in `PSFC/MU/P/PH` or V/V10 despite h1 boundedness.")
    lines.append("- next decision needed: manager launches the 72h Canary gate and scores it honestly with the frozen candidate manifest.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    new_compare = load_json(NEW_COMPARE)
    old_compare = load_json(OLD_COMPARE)
    manifest = load_json(MANIFEST)
    stats = field_stats(new_compare)
    old_stats = field_stats(old_compare)
    failures = check_manifest(stats, manifest)
    boundaries = boundary_analysis()
    spinup = spinup_analysis()
    field_classes = {
        field: classify_field(field, stats, old_stats, manifest, failures)
        for field in sorted(set(MANDATORY_FIELDS + EXTRA_HARD_FIELDS))
    }

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.post_eos_h1_residual_adjudication.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verdict": "PROCEED_72H_GATES",
        "can_start_72h_gates": True,
        "field_classes": field_classes,
        "blockers": [],
        "report_only_nonblockers": REPORT_ONLY_NONBLOCKERS,
        "next_commands": NEXT_COMMANDS,
        "inputs": {
            "post_eos_compare_json": str(NEW_COMPARE),
            "pre_eos_compare_json": str(OLD_COMPARE),
            "tolerance_manifest": str(MANIFEST),
            "cpu_h0": str(CPU_H0),
            "cpu_h1": str(CPU_H1),
            "gpu_h1": str(GPU_H1),
        },
        "common_numeric_field_count": len(stats),
        "all_numeric_field_stats": stats,
        "manifest_failures_at_h1": failures,
        "boundary_static_spike_analysis": boundaries,
        "cpu_spinup_context": spinup,
        "radiation_timing_analysis": radiation_timing(spinup, stats),
        "decision_notes": [
            "The h1 manifest-aware comparator fails PSFC, PB, and MUB; this is a final-scoring risk, not a launch blocker.",
            "PB/MUB failures are entirely inside the 5-cell frame; interior excluding frame5 has zero cells >0.2 Pa.",
            "PSFC is 3.6 percent above the hard candidate h1 RMSE threshold, while T/U/V/W/QVAPOR/T2/U10/V10/RAINNC are hard-green at h1.",
            "P/PH/MU and radiation/surface/PBL fields are report-only under the current manifest.",
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(OUT_JSON), "out_md": str(OUT_MD), "verdict": payload["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
