#!/usr/bin/env python3
"""V0.14 Noah-MP nested-pipeline GPU h1-h4 validation proof.

Offline/CPU-only checker for the short manager gate after the nested pipeline
starts using prognostic Noah-MP land carries. It reads an already completed
h1-h4 GPU run plus CPU-WRF truth and writes compact JSON/Markdown evidence:

- land-masked TSK/HFX/T2/PBLH bias and RMSE by lead;
- all-field comparator highlights from ``canary_d02_h4_grid_compare.json``;
- optional RMSE deltas against the frozen-land baseline;
- resource CSV summary for the GPU short run.

The hard acceptance gates mirror the sprint contract:

- h2-h4 land-mean TSK absolute bias <= 2 K;
- h2-h4 land-mean HFX absolute bias <= 40 W m-2;
- GPU run return code is 0 and paired wrfouts exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DOMAIN = "d02"
INIT = datetime(2026, 5, 1, 18, tzinfo=UTC)
DEFAULT_CPU_DIR = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
DEFAULT_BASELINE_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu_validation/v014_canary_d02_72h_moistcqw_20260610T171818Z"
)
FIELDS = (
    "PSFC",
    "MU",
    "P",
    "PH",
    "W",
    "T",
    "QVAPOR",
    "U",
    "V",
    "U10",
    "V10",
    "T2",
    "TSK",
    "HFX",
    "LH",
    "PBLH",
    "GLW",
    "SWDOWN",
    "SWNORM",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _read_run_root() -> Path:
    env = os.environ.get("GPUWRF_NOAHMP_H4_RUN_ROOT")
    if env:
        return Path(env)
    marker = Path("/tmp/v014_latest_noahmp_h4_run_root")
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())
    raise SystemExit(
        "run root not supplied; pass --run-root or set GPUWRF_NOAHMP_H4_RUN_ROOT"
    )


def _lead_path(base: Path, lead_h: int) -> Path:
    valid = INIT + timedelta(hours=lead_h)
    return base / f"wrfout_{DOMAIN}_{valid:%Y-%m-%d_%H:%M:%S}"


def _read_field(path: Path, name: str) -> np.ndarray:
    with Dataset(path) as nc:
        if name not in nc.variables:
            raise KeyError(f"{name} missing in {path}")
        var = nc.variables[name]
        raw = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
        return np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)


def _land_mask(cpu_path: Path) -> np.ndarray:
    with Dataset(cpu_path) as nc:
        if "LANDMASK" in nc.variables:
            raw = nc.variables["LANDMASK"][0]
            return np.asarray(raw, dtype=np.float64) > 0.5
        if "XLAND" in nc.variables:
            raw = nc.variables["XLAND"][0]
            return np.asarray(raw, dtype=np.float64) < 1.5
    raise KeyError(f"neither LANDMASK nor XLAND present in {cpu_path}")


def _stats(diff: np.ndarray) -> dict[str, float | int | None]:
    flat = np.asarray(diff, dtype=np.float64).ravel()
    valid = np.isfinite(flat)
    if not np.any(valid):
        return {
            "n": int(flat.size),
            "finite": 0,
            "bias": None,
            "rmse": None,
            "mae": None,
            "p99_abs": None,
            "max_abs": None,
        }
    x = flat[valid]
    return {
        "n": int(flat.size),
        "finite": int(x.size),
        "bias": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "mae": float(np.mean(np.abs(x))),
        "p99_abs": float(np.percentile(np.abs(x), 99)),
        "max_abs": float(np.max(np.abs(x))),
    }


def _masked_diff_stats(cpu_path: Path, gpu_path: Path, field: str, mask: np.ndarray) -> dict[str, Any]:
    cpu = _read_field(cpu_path, field)
    gpu = _read_field(gpu_path, field)
    if cpu.shape != gpu.shape:
        raise ValueError(f"{field} shape mismatch: CPU {cpu.shape} GPU {gpu.shape}")
    if cpu.ndim != 2:
        raise ValueError(f"{field} expected 2-D surface field, got {cpu.shape}")
    if mask.shape != cpu.shape:
        raise ValueError(f"mask shape {mask.shape} does not match {field} shape {cpu.shape}")
    return _stats((gpu - cpu)[mask])


def _land_metrics(cpu_dir: Path, gpu_dir: Path, leads: range) -> dict[str, Any]:
    by_lead: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for lead_h in leads:
        cpu_path = _lead_path(cpu_dir, lead_h)
        gpu_path = _lead_path(gpu_dir, lead_h)
        mask = _land_mask(cpu_path)
        row: dict[str, Any] = {
            "cpu_file": str(cpu_path),
            "gpu_file": str(gpu_path),
            "land_cell_count": int(np.sum(mask)),
        }
        for field in ("TSK", "HFX", "T2", "PBLH"):
            row[field] = _masked_diff_stats(cpu_path, gpu_path, field, mask)
        if lead_h >= 2:
            tsk_bias = abs(float(row["TSK"]["bias"]))
            hfx_bias = abs(float(row["HFX"]["bias"]))
            if tsk_bias > 2.0:
                failures.append(
                    {
                        "lead_h": lead_h,
                        "field": "TSK",
                        "metric": "land_abs_bias",
                        "value": tsk_bias,
                        "limit": 2.0,
                    }
                )
            if hfx_bias > 40.0:
                failures.append(
                    {
                        "lead_h": lead_h,
                        "field": "HFX",
                        "metric": "land_abs_bias",
                        "value": hfx_bias,
                        "limit": 40.0,
                    }
                )
        by_lead[f"h{lead_h}"] = row
    return {
        "thresholds": {
            "TSK_h2_h4_land_abs_bias_K": 2.0,
            "HFX_h2_h4_land_abs_bias_W_m2": 40.0,
        },
        "failures": failures,
        "pass": not failures,
        "by_lead": by_lead,
    }


def _compare_field_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = {}
    for field in FIELDS:
        summary = data.get("field_summaries", {}).get(field)
        if not summary:
            continue
        overall = summary.get("overall") or {}
        drift = summary.get("drift") or {}
        fields[field] = {
            "rmse": overall.get("rmse"),
            "bias": overall.get("bias"),
            "p99_abs": overall.get("p99_abs"),
            "max_abs": overall.get("max_abs"),
            "rmse_slope_per_hour": drift.get("rmse_slope_per_hour"),
            "bias_slope_per_hour": drift.get("bias_slope_per_hour"),
            "worst_lead_h": drift.get("worst_lead_h"),
            "tolerance_pass": (summary.get("tolerance_result") or {}).get("pass"),
        }
    return {
        "available": True,
        "path": str(path),
        "verdict": data.get("summaries", {}).get("verdict"),
        "tolerance_failure_count": data.get("summaries", {}).get("tolerance_failure_count"),
        "tolerance_failures": data.get("summaries", {}).get("tolerance_failures", [])[:12],
        "top_dynamic_fields_by_rmse": data.get("summaries", {}).get(
            "top_dynamic_fields_by_rmse", []
        )[:12],
        "fields": fields,
    }


def _field_deltas(new_fields: dict[str, Any], old_fields: dict[str, Any]) -> dict[str, Any]:
    if not new_fields.get("available") or not old_fields.get("available"):
        return {"available": False}
    out: dict[str, Any] = {"available": True, "fields": {}}
    for field in FIELDS:
        new = new_fields.get("fields", {}).get(field)
        old = old_fields.get("fields", {}).get(field)
        if not new or not old:
            continue
        if new.get("rmse") is None or old.get("rmse") is None:
            continue
        out["fields"][field] = {
            "old_rmse": old["rmse"],
            "new_rmse": new["rmse"],
            "rmse_delta": float(new["rmse"]) - float(old["rmse"]),
            "old_bias": old.get("bias"),
            "new_bias": new.get("bias"),
            "old_rmse_slope_per_hour": old.get("rmse_slope_per_hour"),
            "new_rmse_slope_per_hour": new.get("rmse_slope_per_hour"),
        }
    return out


def _resource_summary(run_root: Path) -> dict[str, Any]:
    resources = run_root / "resources"
    gpu_csvs = sorted(resources.glob("*gpu_usage.csv"))
    process_csvs = sorted(resources.glob("*process_usage.csv"))
    system_csvs = sorted(resources.glob("*system_memory.csv"))
    out: dict[str, Any] = {
        "available": bool(gpu_csvs or process_csvs or system_csvs),
        "gpu_csv": str(gpu_csvs[0]) if gpu_csvs else None,
        "process_csv": str(process_csvs[0]) if process_csvs else None,
        "system_csv": str(system_csvs[0]) if system_csvs else None,
    }
    if gpu_csvs:
        rows = list(csv.DictReader(gpu_csvs[0].open(newline="", encoding="utf-8")))
        mem = [float(r["memory_used_mib"]) for r in rows if r.get("memory_used_mib")]
        util = [float(r["utilization_gpu_pct"]) for r in rows if r.get("utilization_gpu_pct")]
        power = [float(r["power_draw_w"]) for r in rows if r.get("power_draw_w")]
        out.update(
            {
                "gpu_samples": len(rows),
                "peak_gpu_memory_used_mib": max(mem) if mem else None,
                "avg_gpu_memory_used_mib": float(np.mean(mem)) if mem else None,
                "avg_gpu_utilization_pct": float(np.mean(util)) if util else None,
                "max_gpu_power_w": max(power) if power else None,
            }
        )
    if process_csvs:
        rows = list(csv.DictReader(process_csvs[0].open(newline="", encoding="utf-8")))
        rss = [float(r["rss_kib"]) / 1024.0 for r in rows if r.get("rss_kib")]
        out.update(
            {
                "process_samples": len(rows),
                "peak_process_rss_mib": max(rss) if rss else None,
            }
        )
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _write_md(out: dict[str, Any], path: Path) -> None:
    land = out["land_gate"]
    res = out["resource_summary"]
    lines = [
        "# V0.14 Noah-MP Nested GPU h1-h4 Validation",
        "",
        f"Verdict: `{out['verdict']}`",
        "",
        f"- run root: `{out['run_root']}`",
        f"- GPU rc: `{out['gpu_rc']}`",
        f"- compare JSON: `{out['compare_summary'].get('path')}`",
        f"- frozen-land baseline compare: `{out['baseline_compare_summary'].get('path')}`",
        f"- peak GPU memory: `{_fmt(res.get('peak_gpu_memory_used_mib'), 0)} MiB`",
        f"- peak process RSS: `{_fmt(res.get('peak_process_rss_mib'), 0)} MiB`",
        "",
        "## Land Gate",
        "",
        f"- pass: `{land['pass']}`",
        f"- failures: `{len(land['failures'])}`",
        "",
        "| Lead | TSK bias K | TSK RMSE K | HFX bias W/m2 | HFX RMSE W/m2 | T2 bias K | PBLH RMSE m |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in land["by_lead"].items():
        lines.append(
            f"| {key[1:]} | {_fmt(row['TSK']['bias'])} | {_fmt(row['TSK']['rmse'])} | "
            f"{_fmt(row['HFX']['bias'])} | {_fmt(row['HFX']['rmse'])} | "
            f"{_fmt(row['T2']['bias'])} | {_fmt(row['PBLH']['rmse'])} |"
        )
    lines += [
        "",
        "## Comparator Highlights",
        "",
        f"- comparator verdict: `{out['compare_summary'].get('verdict')}`",
        f"- tolerance failure count: `{out['compare_summary'].get('tolerance_failure_count')}`",
        "",
        "| Field | RMSE | Bias | p99 abs | RMSE slope/h | Worst lead |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, row in out["compare_summary"].get("fields", {}).items():
        lines.append(
            f"| `{field}` | {_fmt(row.get('rmse'))} | {_fmt(row.get('bias'))} | "
            f"{_fmt(row.get('p99_abs'))} | {_fmt(row.get('rmse_slope_per_hour'))} | "
            f"{row.get('worst_lead_h')} |"
        )
    deltas = out.get("rmse_delta_vs_frozen_land_baseline", {})
    if deltas.get("available"):
        lines += [
            "",
            "## RMSE Delta Vs Frozen-Land Baseline",
            "",
            "Negative delta means the Noah-MP nested run moved closer to CPU-WRF.",
            "",
            "| Field | Old RMSE | New RMSE | Delta | Old bias | New bias |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for field, row in deltas["fields"].items():
            lines.append(
                f"| `{field}` | {_fmt(row['old_rmse'])} | {_fmt(row['new_rmse'])} | "
                f"{_fmt(row['rmse_delta'])} | {_fmt(row.get('old_bias'))} | "
                f"{_fmt(row.get('new_bias'))} |"
            )
    if land["failures"]:
        lines += ["", "## Gate Failures", "", "```json", json.dumps(land["failures"], indent=2), "```"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--cpu-dir", type=Path, default=DEFAULT_CPU_DIR)
    ap.add_argument("--baseline-run-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    ap.add_argument("--out-json", type=Path, default=Path(__file__).with_suffix(".json"))
    ap.add_argument("--out-md", type=Path, default=Path(__file__).with_suffix(".md"))
    args = ap.parse_args(argv)

    run_root = args.run_root or _read_run_root()
    gpu_dir = run_root / "gpu_output" / f"l2_d02_{RUN_ID}"
    baseline_gpu_dir = args.baseline_run_root / "gpu_output" / f"l2_d02_{RUN_ID}"
    gpu_rc_path = run_root / "gpu_h4.rc"
    gpu_rc = int(gpu_rc_path.read_text(encoding="utf-8").strip()) if gpu_rc_path.exists() else None

    missing = [
        str(_lead_path(gpu_dir, h))
        for h in range(1, 5)
        if not _lead_path(gpu_dir, h).exists()
    ]
    compare_summary = _compare_field_summary(run_root / "canary_d02_h4_grid_compare.json")
    baseline_compare = _compare_field_summary(
        args.baseline_run_root / "canary_d02_h04_grid_compare.json"
    )
    if not baseline_compare.get("available"):
        baseline_compare = _compare_field_summary(
            args.baseline_run_root / "canary_d02_h4_grid_compare.json"
        )
    land_gate = _land_metrics(args.cpu_dir, gpu_dir, range(1, 5)) if not missing else {
        "thresholds": {
            "TSK_h2_h4_land_abs_bias_K": 2.0,
            "HFX_h2_h4_land_abs_bias_W_m2": 40.0,
        },
        "failures": [{"reason": "missing_gpu_wrfouts", "files": missing}],
        "pass": False,
        "by_lead": {},
    }
    verdict = (
        "NOAHMP_NESTED_GPU_H4_ACCEPT"
        if gpu_rc == 0 and not missing and land_gate["pass"]
        else "NOAHMP_NESTED_GPU_H4_REVIEW"
    )
    out = {
        "schema": "V014NoahmpNestedGpuH4Validation",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "verdict": verdict,
        "run_root": str(run_root),
        "run_id": RUN_ID,
        "domain": DOMAIN,
        "cpu_dir": str(args.cpu_dir),
        "gpu_dir": str(gpu_dir),
        "baseline_run_root": str(args.baseline_run_root),
        "baseline_gpu_dir": str(baseline_gpu_dir),
        "gpu_rc": gpu_rc,
        "missing_gpu_wrfouts": missing,
        "land_gate": land_gate,
        "compare_summary": compare_summary,
        "baseline_compare_summary": baseline_compare,
        "rmse_delta_vs_frozen_land_baseline": _field_deltas(compare_summary, baseline_compare),
        "resource_summary": _resource_summary(run_root),
        "cpu_only": True,
        "gpu_used_by_this_script": False,
    }
    args.out_json.write_text(json.dumps(out, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_md(out, args.out_md)
    print(json.dumps({"verdict": verdict, "json": str(args.out_json), "md": str(args.out_md)}, indent=2))
    return 0 if verdict == "NOAHMP_NESTED_GPU_H4_ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
