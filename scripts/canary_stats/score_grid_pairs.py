#!/usr/bin/env python3
"""Compute gridded paired CPU-vs-GPUWRF statistics on existing wrfout files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_SCORE_VARS,
    list_wrfout_files,
    parse_iso_time,
    read_var,
    safe_corr,
    summarize_deltas,
    trim_boundary,
    write_csv,
    write_json,
)


def _lead(valid: datetime, init: datetime) -> int:
    return int(round((valid - init).total_seconds() / 3600.0))


def _weighted_aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_var: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "OK":
            by_var[row["variable"]].append(row)
    out = {}
    for var, recs in by_var.items():
        n_total = sum(int(r["n"]) for r in recs)
        if n_total == 0:
            out[var] = {"status": "NO_DATA", "n": 0}
            continue
        mse = sum((float(r["rmse"]) ** 2) * int(r["n"]) for r in recs) / n_total
        bias = sum(float(r["bias"]) * int(r["n"]) for r in recs) / n_total
        mae = sum(float(r["mae"]) * int(r["n"]) for r in recs) / n_total
        corr_vals = [float(r["corr"]) for r in recs if r.get("corr") is not None]
        out[var] = {
            "status": "OK",
            "n": int(n_total),
            "frame_count": len(recs),
            "rmse": float(np.sqrt(mse)),
            "bias": float(bias),
            "mae": float(mae),
            "mean_frame_corr": float(np.mean(corr_vals)) if corr_vals else None,
        }
    return out


def score_pair(
    *,
    cpu_dir: Path,
    gpu_dir: Path,
    domain: str,
    init: datetime,
    variables: tuple[str, ...],
    max_lead: int | None,
    spinup_hours: int,
    boundary_width: int,
) -> dict[str, Any]:
    cpu = {t: p for t, p in list_wrfout_files(cpu_dir, domain)}
    gpu = {t: p for t, p in list_wrfout_files(gpu_dir, domain)}
    common = sorted(set(cpu) & set(gpu))
    rows = []
    for valid in common:
        lead = _lead(valid, init)
        if lead <= spinup_hours:
            continue
        if max_lead is not None and lead > max_lead:
            continue
        for var in variables:
            row: dict[str, Any] = {
                "valid_time_utc": valid.isoformat(),
                "lead_hour": lead,
                "variable": var,
                "cpu_file": str(cpu[valid]),
                "gpu_file": str(gpu[valid]),
            }
            try:
                c = trim_boundary(read_var(cpu[valid], var), boundary_width)
                g = trim_boundary(read_var(gpu[valid], var), boundary_width)
            except Exception as exc:
                row.update({"status": "READ_ERROR", "reason": f"{type(exc).__name__}: {exc}"})
                rows.append(row)
                continue
            if c.shape != g.shape:
                row.update({"status": "SHAPE_MISMATCH", "cpu_shape": list(c.shape), "gpu_shape": list(g.shape)})
                rows.append(row)
                continue
            mask = np.isfinite(c) & np.isfinite(g)
            if not mask.any():
                row.update({"status": "NO_FINITE_PAIRS", "cpu_shape": list(c.shape), "gpu_shape": list(g.shape)})
                rows.append(row)
                continue
            cv = c[mask].ravel()
            gv = g[mask].ravel()
            delta = gv - cv
            stats = summarize_deltas(delta)
            row.update(stats)
            row["corr"] = safe_corr(gv, cv)
            row["cpu_shape"] = list(c.shape)
            row["gpu_shape"] = list(g.shape)
            row["finite_fraction"] = float(mask.mean())
            rows.append(row)
    aggregate = _weighted_aggregate(rows)
    return {
        "schema": "CanaryGridPairStats",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_dir": str(cpu_dir),
        "gpu_dir": str(gpu_dir),
        "domain": domain,
        "init_time_utc": init.isoformat(),
        "variables": list(variables),
        "max_lead": max_lead,
        "spinup_hours": spinup_hours,
        "boundary_width": boundary_width,
        "common_frame_count_before_filters": len(common),
        "scored_rows": rows,
        "aggregate": aggregate,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-dir", type=Path, required=True)
    ap.add_argument("--gpu-dir", type=Path, required=True)
    ap.add_argument("--domain", default="d02")
    ap.add_argument("--init", required=True)
    ap.add_argument("--max-lead", type=int, default=None)
    ap.add_argument("--spinup-hours", type=int, default=0)
    ap.add_argument("--boundary-width", type=int, default=0)
    ap.add_argument("--vars", nargs="+", default=list(DEFAULT_SCORE_VARS))
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path)
    args = ap.parse_args(argv)
    init = parse_iso_time(args.init)
    if init is None:
        raise SystemExit(f"invalid --init: {args.init}")
    result = score_pair(
        cpu_dir=args.cpu_dir,
        gpu_dir=args.gpu_dir,
        domain=args.domain,
        init=init,
        variables=tuple(args.vars),
        max_lead=args.max_lead,
        spinup_hours=args.spinup_hours,
        boundary_width=args.boundary_width,
    )
    write_json(args.out_json, result)
    if args.out_csv:
        write_csv(
            args.out_csv,
            result["scored_rows"],
            [
                "valid_time_utc",
                "lead_hour",
                "variable",
                "status",
                "n",
                "rmse",
                "bias",
                "mae",
                "p95_abs",
                "p99_abs",
                "max_abs",
                "corr",
                "finite_fraction",
                "cpu_shape",
                "gpu_shape",
                "reason",
                "cpu_file",
                "gpu_file",
            ],
        )
    ok = {k: v.get("rmse") for k, v in result["aggregate"].items()}
    print({"common_frames": result["common_frame_count_before_filters"], "aggregate_rmse": ok})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
