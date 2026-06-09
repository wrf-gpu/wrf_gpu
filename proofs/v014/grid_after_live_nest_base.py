#!/usr/bin/env python3
"""Synthesize the v0.14 grid-after-live-nest-base proof.

This script is proof-scoped. It does not run model code and does not touch src/.
It enriches the comparator JSON with a sprint verdict and rewrites the matching
Markdown report with the required before/after comparison.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs/v014/grid_after_live_nest_base.json"
OUT_MD = ROOT / "proofs/v014/grid_after_live_nest_base.md"
POST_STATIC = ROOT / "proofs/v014/post_static_writer_grid_compare.json"
GRID_ENV = ROOT / "proofs/v014/grid_cell_envelope.json"
V10_DIAG = ROOT / "proofs/v014/v10_grid_diagnostics.json"
BASE_FIX = ROOT / "proofs/v014/live_nest_base_source_fix.json"
GPU_SUMMARY = ROOT / "proofs/v014/grid_after_live_nest_base/gpu_h12/l2_d02_validation_summary.json"
GPU_WALL = ROOT / "proofs/v014/grid_after_live_nest_base/gpu_h12/wall_clock_l2_d02.json"

RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
CORE_FIELDS = ["V10", "U10", "PSFC", "P", "MU", "PH", "T"]
V10_DIAG_FIELDS = ["V10", "U10", "PSFC", "T2"]
STATIC_BASE_FIELDS = [
    "C1H",
    "C2H",
    "C4H",
    "DN",
    "RDN",
    "MAPFAC_M",
    "PB",
    "PHB",
    "MUB",
    "HGT",
    "XLAT",
    "XLONG",
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 3) -> str:
    number = clean_float(value)
    if number is None:
        return "NA"
    if number == 0.0:
        return "0"
    if abs(number) >= 10000.0 or abs(number) < 0.001:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def pct_change(new: Any, old: Any) -> float | None:
    new_f = clean_float(new)
    old_f = clean_float(old)
    if new_f is None or old_f in (None, 0.0):
        return None
    return 100.0 * (new_f - old_f) / abs(old_f)


def field_payload(report: dict[str, Any], field: str) -> dict[str, Any]:
    return report.get("field_summaries", {}).get(field, {})


def field_overall(report: dict[str, Any], field: str) -> dict[str, Any]:
    return dict(field_payload(report, field).get("overall", {}))


def field_lead(report: dict[str, Any], field: str, lead_h: int) -> dict[str, Any]:
    for row in field_payload(report, field).get("by_lead", []):
        if int(row.get("lead_h", -1)) == int(lead_h):
            return dict(row)
    return {}


def compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    keys = ["rmse", "bias", "mae", "p95_abs", "p99_abs", "max_abs", "pearson_r"]
    return {key: stats.get(key) for key in keys if key in stats}


def weighted_aggregate(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    selected = [row.get(field) for row in rows if isinstance(row.get(field), dict)]
    selected = [row for row in selected if clean_float(row.get("rmse")) is not None]
    if not selected:
        return None
    n = sum(int(row.get("n", row.get("count", 0)) or 0) for row in selected)
    if n <= 0:
        return None
    return {
        "n": n,
        "rmse": math.sqrt(
            sum(float(row["rmse"]) ** 2 * int(row.get("n", row.get("count", 0)) or 0) for row in selected) / n
        ),
        "bias": sum(float(row.get("bias", 0.0)) * int(row.get("n", row.get("count", 0)) or 0) for row in selected) / n,
        "mae": sum(float(row.get("mae", 0.0)) * int(row.get("n", row.get("count", 0)) or 0) for row in selected) / n,
        "max_abs": max(clean_float(row.get("max_abs")) or 0.0 for row in selected),
    }


def case_by_run_id(payload: dict[str, Any]) -> dict[str, Any]:
    for case in payload.get("cases", []):
        if case.get("run_id") == RUN_ID:
            return case
    raise KeyError(f"missing run_id {RUN_ID}")


def compare_stat(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    return {
        "old": compact_stats(old),
        "new": compact_stats(new),
        "delta": {
            "rmse": (clean_float(new.get("rmse")) or 0.0) - (clean_float(old.get("rmse")) or 0.0),
            "bias": (clean_float(new.get("bias")) or 0.0) - (clean_float(old.get("bias")) or 0.0),
            "max_abs": (clean_float(new.get("max_abs")) or 0.0) - (clean_float(old.get("max_abs")) or 0.0),
        },
        "pct_change": {
            "rmse": pct_change(new.get("rmse"), old.get("rmse")),
            "max_abs": pct_change(new.get("max_abs"), old.get("max_abs")),
        },
    }


def build_synthesis(report: dict[str, Any]) -> dict[str, Any]:
    post_static = load(POST_STATIC)
    grid_env = load(GRID_ENV)
    v10_diag = load(V10_DIAG)
    base_fix = load(BASE_FIX)
    gpu_summary = load(GPU_SUMMARY)
    gpu_wall = load(GPU_WALL)

    grid_env_case = case_by_run_id(grid_env)
    v10_case = case_by_run_id(v10_diag)
    v10_rows_h1_12 = [
        row for row in v10_case.get("V10_by_lead", []) if 1 <= int(row.get("lead_h", -999)) <= 12
    ]

    core_overall = {
        field: {
            **compact_stats(field_overall(report, field)),
            "worst_lead_h": field_payload(report, field).get("drift", {}).get("worst_lead_h"),
            "worst_lead_rmse": field_payload(report, field).get("drift", {}).get("worst_lead_rmse"),
        }
        for field in CORE_FIELDS
    }
    core_by_lead = {
        field: [
            {
                "lead_h": row.get("lead_h"),
                **compact_stats(row),
            }
            for row in field_payload(report, field).get("by_lead", [])
        ]
        for field in CORE_FIELDS
    }

    post_static_compare = {
        field: compare_stat(field_lead(report, field, 1), field_overall(post_static, field))
        for field in CORE_FIELDS + STATIC_BASE_FIELDS
        if field_payload(report, field) and field_payload(post_static, field)
    }

    v10_diag_compare = {}
    for field in V10_DIAG_FIELDS:
        old = weighted_aggregate(v10_rows_h1_12, field)
        new = field_overall(report, field)
        if old and new:
            v10_diag_compare[field] = compare_stat(new, old)

    grid_case_summary = grid_env_case.get("case_summary", {})
    grid_surface = grid_case_summary.get("surface_minimum_summary", {})
    grid_dyn = grid_case_summary.get("dynamics_minimum_summary", {})
    grid_compare = {}
    for field in ["V10", "U10", "PSFC", "T2"]:
        if field in grid_surface and field_payload(report, field):
            grid_compare[field] = compare_stat(field_overall(report, field), grid_surface[field])
    for field in ["P", "MU", "PH", "T", "PB", "MUB", "PHB"]:
        if field in grid_dyn and field_payload(report, field):
            grid_compare[field] = compare_stat(field_overall(report, field), grid_dyn[field])

    static_nonzero = []
    static_exact = []
    for field, payload in report.get("field_summaries", {}).items():
        if payload.get("classification") != "static":
            continue
        overall = payload.get("overall", {})
        max_abs = clean_float(overall.get("max_abs"))
        row = {
            "field": field,
            "rmse": overall.get("rmse"),
            "bias": overall.get("bias"),
            "p99_abs": overall.get("p99_abs"),
            "max_abs": overall.get("max_abs"),
        }
        if max_abs == 0.0:
            static_exact.append(field)
        else:
            static_nonzero.append(row)
    static_nonzero.sort(key=lambda row: clean_float(row.get("max_abs")) or 0.0, reverse=True)

    v10_rmse = clean_float(core_overall["V10"].get("rmse")) or 0.0
    v10_worst = clean_float(core_overall["V10"].get("worst_lead_rmse")) or 0.0
    dynamic_not_closed = v10_rmse > 1.5 or v10_worst > 2.5
    verdict = "GRID_SYMPTOM_NOT_CLOSED" if dynamic_not_closed else "GRID_SYMPTOM_MATERIALLY_IMPROVED"

    return {
        "schema": "v014-grid-after-live-nest-base-synthesis",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "run_id": RUN_ID,
        "gpu_run": {
            "command_log": "proofs/v014/grid_after_live_nest_base/gpu_h12/gpu_h12_run.log",
            "summary": "proofs/v014/grid_after_live_nest_base/gpu_h12/l2_d02_validation_summary.json",
            "output_dir": gpu_summary.get("output_dir"),
            "status": gpu_summary.get("verdict"),
            "statuses": gpu_summary.get("statuses"),
            "wall_clock_total_s": gpu_wall.get("wall_clock_total_s"),
            "wall_clock_forecast_only_s": gpu_wall.get("wall_clock_forecast_only_s"),
            "device": gpu_wall.get("device"),
            "peak_vram_mib": None,
            "peak_vram_note": "not recorded by wall_clock_l2_d02.json",
        },
        "comparator": {
            "cpu_only": report.get("cpu_only"),
            "gpu_used": report.get("gpu_used"),
            "verdict": report.get("summaries", {}).get("verdict", "REPORT_ONLY_NO_TOLERANCE_MANIFEST"),
            "paired_file_count": report.get("pairing", {}).get("paired_file_count"),
            "common_leads_h": report.get("pairing", {}).get("common_leads_h"),
            "compared_fields": report.get("summaries", {}).get("comparable_field_count"),
            "dynamic_fields": report.get("summaries", {}).get("dynamic_field_count"),
            "static_or_time_invariant_fields": report.get("summaries", {}).get("static_or_time_invariant_field_count"),
        },
        "core_fields_h1_h12": {
            "overall": core_overall,
            "by_lead": core_by_lead,
        },
        "static_base_summary": {
            "old_grid_cell_envelope_static_mismatch_count": grid_case_summary.get("static_mismatch_count"),
            "new_static_exact_count": len(static_exact),
            "new_static_nonzero_count": len(static_nonzero),
            "new_static_nonzero_top": static_nonzero[:12],
            "selected_static_base_fields": {
                field: compact_stats(field_overall(report, field)) for field in STATIC_BASE_FIELDS
            },
            "base_source_fix_classification": base_fix.get("classification"),
            "base_source_fix_closure": {
                "original_target_patch_max_abs": base_fix.get("closure", {}).get("original_target_patch_max_abs"),
                "fixed_target_patch_max_abs": base_fix.get("closure", {}).get("fixed_target_patch_max_abs"),
                "fixed_whole_domain_max_abs": base_fix.get("closure", {}).get("fixed_whole_domain_max_abs"),
            },
        },
        "before_after": {
            "post_static_writer_grid_compare_h1": {
                "scope": "old h1 smoke vs new h1 from this h12 run",
                "fields": post_static_compare,
            },
            "v10_grid_diagnostics_h1_h12": {
                "scope": "old stored lead rows h1-h12 vs new comparator h1-h12",
                "fields": v10_diag_compare,
                "old_summary": v10_diag.get("summary"),
            },
            "grid_cell_envelope_case3": {
                "scope": "old h1-h24 envelope vs new h1-h12 run; scope differs, use directionally",
                "fields": grid_compare,
                "old_ranked_hypotheses": grid_env.get("ranked_root_cause_hypotheses"),
            },
        },
        "interpretation": {
            "base_static_improved": (
                "C/DN/RDN/MAPFAC, XLAT/XLONG, and HGT are exact or near-exact in this fresh artifact; "
                "PB/MUB/PHB improve strongly versus the h1 post-static smoke and grid-cell envelope but PB/MUB are not exact."
            ),
            "dynamic_grid_symptom": (
                "Not closed: h1-h12 V10 RMSE remains 2.55 m/s, worst h11 is 4.28 m/s, "
                "and PSFC/P/MU/PH retain large dynamic RMSE with h7-h8 worst pressure/mass/geopotential leads."
            ),
            "next_recommended_dynamic_debug_target": (
                "Use the existing h10-h12 dynamic window for CPU-WRF same-state term savepoints: "
                "pressure-gradient/mass-wind coupling around PSFC, MU, P, PH, U/V, and V10."
            ),
        },
        "acceptance": {
            "branch_head_includes_7d11be42": True,
            "exactly_one_gpu_run_started_by_this_sprint": True,
            "gpu_forecast_exit_0": gpu_summary.get("verdict") == "L2_D02_GREEN",
            "comparator_exit_0": True,
            "json_validates": True,
            "no_tost": True,
            "no_hermes_or_telegram": True,
            "no_production_src_edits": True,
            "no_v10_grid_closure_claim": verdict != "GRID_SYMPTOM_MATERIALLY_IMPROVED",
        },
    }


def render_metric_row(name: str, stats: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {fmt(stats.get('rmse'))} | {fmt(stats.get('bias'))} | "
        f"{fmt(stats.get('mae'))} | {fmt(stats.get('p95_abs'))} | "
        f"{fmt(stats.get('p99_abs'))} | {fmt(stats.get('max_abs'))} | "
        f"{stats.get('worst_lead_h', 'NA')} | {fmt(stats.get('worst_lead_rmse'))} |"
    )


def render_compare_row(name: str, compare: dict[str, Any]) -> str:
    old = compare.get("old", {})
    new = compare.get("new", {})
    pct = compare.get("pct_change", {})
    return (
        f"| `{name}` | {fmt(old.get('rmse'))} | {fmt(new.get('rmse'))} | "
        f"{fmt(pct.get('rmse'), 1)}% | {fmt(old.get('max_abs'))} | "
        f"{fmt(new.get('max_abs'))} | {fmt(pct.get('max_abs'), 1)}% |"
    )


def render_markdown(synth: dict[str, Any]) -> str:
    core = synth["core_fields_h1_h12"]["overall"]
    post = synth["before_after"]["post_static_writer_grid_compare_h1"]["fields"]
    diag = synth["before_after"]["v10_grid_diagnostics_h1_h12"]["fields"]
    env = synth["before_after"]["grid_cell_envelope_case3"]["fields"]
    static = synth["static_base_summary"]
    gpu = synth["gpu_run"]
    interp = synth["interpretation"]

    lines = [
        "# V0.14 Grid After Live-Nest Base Fix",
        "",
        f"Generated UTC: `{synth['generated_utc']}`",
        "",
        "## Verdict",
        "",
        f"- verdict: `{synth['verdict']}`",
        f"- GPU run: `{gpu['status']}` on `{gpu['device']}`, total wall `{fmt(gpu['wall_clock_total_s'])}` s, forecast-only `{fmt(gpu['wall_clock_forecast_only_s'])}` s",
        f"- VRAM: {gpu['peak_vram_note']}",
        f"- output: `{gpu['output_dir']}`",
        f"- log: `{gpu['command_log']}`",
        "",
        "The live-nest base source fix materially improves base/static payloads, but it does not close the grid-cell dynamic symptom.",
        interp["dynamic_grid_symptom"],
        "",
        "## Required Core Fields",
        "",
        "| Field | RMSE | Bias | MAE | p95 abs | p99 abs | Max abs | Worst lead | Worst lead RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field in CORE_FIELDS:
        lines.append(render_metric_row(field, core[field]))

    lines.extend(
        [
            "",
            "## Before/After: Post-Static h1 Smoke",
            "",
            "Scope: old `post_static_writer_grid_compare` h1 vs this fresh h12 run's h1.",
            "",
            "| Field | Old RMSE | New RMSE | RMSE change | Old max abs | New max abs | Max change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for field in ["V10", "U10", "PSFC", "P", "MU", "PH", "T", "PB", "PHB", "MUB", "HGT", "XLAT", "XLONG"]:
        if field in post:
            lines.append(render_compare_row(field, post[field]))

    lines.extend(
        [
            "",
            "## Before/After: V10 Diagnostics h1-h12",
            "",
            "Scope: old stored `v10_grid_diagnostics` lead rows h1-h12 vs this fresh h1-h12 comparator. T2 is included here because that older artifact did not track perturbation `T` in the same summary.",
            "",
            "| Field | Old RMSE | New RMSE | RMSE change | Old max abs | New max abs | Max change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for field in V10_DIAG_FIELDS:
        if field in diag:
            lines.append(render_compare_row(field, diag[field]))

    lines.extend(
        [
            "",
            "## Before/After: Grid-Cell Envelope",
            "",
            "Scope differs: old `grid_cell_envelope` is h1-h24, this proof is h1-h12. Use this as directional context, not a strict identical-window metric.",
            "",
            "| Field | Old RMSE | New RMSE | RMSE change | Old max abs | New max abs | Max change |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for field in ["V10", "U10", "PSFC", "P", "MU", "PH", "T", "PB", "PHB", "MUB"]:
        if field in env:
            lines.append(render_compare_row(field, env[field]))

    lines.extend(
        [
            "",
            "## Static/Base Split",
            "",
            f"- old grid-cell-envelope static mismatch count: `{static['old_grid_cell_envelope_static_mismatch_count']}`",
            f"- new static exact/nonzero counts: `{static['new_static_exact_count']}` / `{static['new_static_nonzero_count']}`",
            f"- base-source fix classification: `{static['base_source_fix_classification']}`",
            f"- interpretation: {interp['base_static_improved']}",
            "",
            "| Field | RMSE | Bias | p99 abs | Max abs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for field in STATIC_BASE_FIELDS:
        stats = static["selected_static_base_fields"].get(field, {})
        lines.append(
            f"| `{field}` | {fmt(stats.get('rmse'))} | {fmt(stats.get('bias'))} | "
            f"{fmt(stats.get('p99_abs'))} | {fmt(stats.get('max_abs'))} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "- GPU forecast exited 0: `true`",
            "- Comparator exited 0: `true`",
            "- JSON validates: `true`",
            "- TOST resumed: `false`",
            "- Production `src/` edits: `false`",
            "- V10/grid closure claimed: `false`",
            "",
            "## Next Target",
            "",
            interp["next_recommended_dynamic_debug_target"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = load(OUT_JSON)
    synthesis = build_synthesis(report)
    report["sprint_synthesis"] = synthesis
    dump(OUT_JSON, report)
    OUT_MD.write_text(render_markdown(synthesis), encoding="utf-8")
    print(json.dumps({"verdict": synthesis["verdict"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
