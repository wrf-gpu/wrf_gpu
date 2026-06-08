#!/usr/bin/env python3
"""
v0.12.0 Powered n=15 TOST campaign — GPU orchestrator + scorer.

Runs 15 GPU d02 24h fp64 forecasts (init from wrf_l2/<RUN_ID>),
scores each against the CPU-WRF truth in wrf_l2_backfill_output/<RUN_ID>,
and writes per-case proof JSONs + the final TOST + cell-level stats report.

MANAGER: run ONE instance at a time on the GPU (flock wrapper enforces serial).

Usage (from this worktree root):
    /tmp/wrf_gpu_run_lowprio.sh taskset -c 0-3 \\
        env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \\
        python proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py [--resume]

Options:
    --resume        Skip cases that already have a case_<RUN_ID>.json proof
    --skip-gpu      Score from existing GPU wrfouts (no forecast; for re-scoring)
    --case <ID>     Run only this single case (debug)
    --dry-run       Prepare merged root + print plan; no GPU forecasts, no scoring

Root fix: ROOT is resolved from __file__ so this script always imports
gpuwrf from THIS worktree's src/, not from /home/enric/src/wrf_gpu2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# ── Path bootstrap — MUST come before any gpuwrf import ────────────────────────
ROOT = Path(__file__).resolve().parents[3]   # .../worktrees/v0120-tostprep
SRC = ROOT / "src"
for _p in [str(SRC), str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from gpuwrf.io.data_inventory import parse_wrfout_valid_time  # noqa: E402
from proofs.m20.paired_tost_scorer import (                   # noqa: E402
    MARGINS, SCORE_VARS, aggregate_tost, paired_score,
)

# ── Data paths (all env-overridable so the manager can repoint without edits) ───
L2_INIT_ROOT  = Path(os.environ.get(
    "GPUWRF_L2_INIT_ROOT", "/mnt/data/canairy_meteo/runs/wrf_l2"))
L2_CPU_ROOT   = Path(os.environ.get(
    "GPUWRF_L2_CPU_ROOT", "/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output"))
AEMET_ROOT    = Path(os.environ.get(
    "GPUWRF_AEMET_ROOT",
    "/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations",
))
MERGED_RUN_ROOT = Path(os.environ.get(
    "GPUWRF_TOST_MERGED_ROOT", "/tmp/v0120_merged_run_root"))
GPU_RUNS_ROOT   = Path(os.environ.get(
    "GPUWRF_TOST_GPU_RUNS_ROOT", "/tmp/v0120_powered_tost_runs"))
PROOF_DIR       = ROOT / "proofs/v0120/powered_tost_n15"
# Only-if-present GPU lock wrapper. If the wrapper is absent (fresh checkout /
# tester box) we launch the per-case runner DIRECTLY — the runner does not need
# the lock for correctness, only for serialising against other GPU lanes. NEVER
# wrap the orchestrator itself in a second lock wrapper (the inner per-case call
# would deadlock on the same flock the outer wrapper already holds).
_WRAP_ENV = os.environ.get("GPUWRF_GPU_LOCK_WRAPPER", "/tmp/wrf_gpu_run_lowprio.sh")
GPU_LOCK_WRAPPER = _WRAP_ENV if (_WRAP_ENV and Path(_WRAP_ENV).is_file()) else None

FIELDS = ("T2", "U10", "V10")
FORECAST_HOURS = 24

# ── 15 cases (matched to wrf_l2_backfill_output dirs) ─────────────────────────
CASE_IDS = [
    "20260429_18z_l2_72h_20260524T204451Z",
    "20260430_18z_l2_72h_20260520T191306Z",
    "20260501_18z_l2_72h_20260519T173026Z",
    "20260502_18z_l2_72h_20260520T103946Z",
    "20260503_18z_l2_72h_20260518T205545Z",
    "20260504_18z_l2_72h_20260515T061907Z",
    "20260505_18z_l2_72h_20260518T074056Z",
    "20260506_18z_l2_72h_20260513T222831Z",
    "20260507_18z_l2_72h_20260513T124307Z",
    "20260508_18z_l2_72h_20260512T161222Z",
    "20260510_18z_l2_72h_20260511T124717Z",
    "20260511_18z_l2_72h_20260512T045528Z",
    "20260512_18z_l2_72h_20260513T014823Z",
    "20260513_18z_l2_72h_20260514T054102Z",
    "20260530_18z_l2_72h_20260531T161057Z",
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def json_default(v):
    if isinstance(v, Path):        return str(v)
    if isinstance(v, datetime):    return v.isoformat()
    if isinstance(v, np.ndarray):  return v.tolist()
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating): return float(v) if np.isfinite(v) else None
    if isinstance(v, float):       return v if math.isfinite(v) else None
    return str(v)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
                    encoding="utf-8")


def parse_init_time(run_id: str) -> datetime:
    """Parse '20260429_18z_...' → 2026-04-29T18:00:00+00:00"""
    parts = run_id.split("_")
    ds = parts[0]
    hr = int(parts[1].replace("z", ""))
    return datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]), hr, 0, 0, tzinfo=timezone.utc)


def wrfout_map(run_dir: Path, domain: str = "d02") -> dict[datetime, Path]:
    mapped: dict[datetime, Path] = {}
    for path in sorted(run_dir.glob(f"wrfout_{domain}_*")):
        if not path.is_file():
            continue
        try:
            vt = parse_wrfout_valid_time(path)
            if vt.tzinfo is None:
                vt = vt.replace(tzinfo=timezone.utc)
            else:
                vt = vt.astimezone(timezone.utc)
            mapped[vt] = path
        except Exception:
            pass
    return mapped


def read_field(path: Path, name: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        var = ds.variables[name]
        data = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def check_finite_all_fields(path: Path, fields=FIELDS + ("U", "V")) -> tuple[bool, list[str]]:
    not_finite = []
    with Dataset(path, "r") as ds:
        for f in fields:
            if f not in ds.variables:
                continue
            var = ds.variables[f]
            arr = np.asarray(
                np.ma.filled(
                    var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:],
                    np.nan,
                ),
                dtype=float,
            )
            if not np.all(np.isfinite(arr)):
                not_finite.append(f)
    return len(not_finite) == 0, not_finite


# ── Merged-run-root preparation ────────────────────────────────────────────────

def prepare_merged_run_root() -> Path:
    """
    Create per-case dirs that Gen2Run / execute_daily_pipeline can use:
      - init files (wrfinput_d01/d02, wrfbdy_d01, namelist.*) ← wrf_l2
      - d01 wrfout history (73 files)                          ← wrf_l2_backfill_output
      - d02 wrfout t=0 snapshot (first file)                   ← wrf_l2_backfill_output
    All files are created as symlinks; existing symlinks are reused.
    """
    MERGED_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for run_id in CASE_IDS:
        merged_dir = MERGED_RUN_ROOT / run_id
        merged_dir.mkdir(parents=True, exist_ok=True)
        init_dir     = L2_INIT_ROOT / run_id
        backfill_dir = L2_CPU_ROOT  / run_id

        # Init files
        for fname in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01",
                      "namelist.input", "namelist.output"):
            src = init_dir / fname
            if src.is_file():
                dest = merged_dir / fname
                if not dest.exists():
                    dest.symlink_to(src)

        # d01 wrfout history (boundary forcing for nested run)
        for src in sorted(backfill_dir.glob("wrfout_d01_*")):
            if src.is_file():
                dest = merged_dir / src.name
                if not dest.exists():
                    dest.symlink_to(src)

        # d02 wrfout t=0 (metrics_source for build_replay_case)
        d02_sorted = sorted(backfill_dir.glob("wrfout_d02_*"))
        if d02_sorted:
            src = d02_sorted[0]
            dest = merged_dir / src.name
            if not dest.exists():
                dest.symlink_to(src)

    return MERGED_RUN_ROOT


# ── Per-case GPU forecast ──────────────────────────────────────────────────────

def run_gpu_forecast(run_id: str, gpu_out_dir: Path, proof_subdir: Path) -> dict:
    gpu_out_dir.mkdir(parents=True, exist_ok=True)
    proof_subdir.mkdir(parents=True, exist_ok=True)

    # The per-case runner is in THIS worktree (uses same ROOT-relative path)
    runner = ROOT / "proofs/v0120/powered_tost_n15/run_one_case_v0120.py"

    # Prefix with the GPU lock wrapper ONLY if it exists on this box. The
    # orchestrator must NOT itself be wrapped in a lock wrapper (double-wrap
    # deadlock); the lock lives at the per-case granularity only.
    prefix = [GPU_LOCK_WRAPPER] if GPU_LOCK_WRAPPER else []
    cmd = [
        *prefix,
        "taskset", "-c", "0-3",
        "env",
        f"PYTHONPATH={SRC}",
        "JAX_ENABLE_X64=true",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "python",
        str(runner),
        "--run-root",       str(MERGED_RUN_ROOT),
        "--cpu-truth-root", str(L2_CPU_ROOT),
        "--run-id",         run_id,
        "--hours",          str(FORECAST_HOURS),
        "--output-root",    str(GPU_RUNS_ROOT),
        "--proof-dir",      str(proof_subdir),
    ]
    print(f"\n[GPU] Starting forecast: {run_id}", flush=True)
    print(f"[GPU] Command: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    rc = result.returncode
    # On rc=2 (L2_D02_BLOCKED) read the per-case summary's blocked_reason so the
    # campaign log + exclusion record carry the real GPU-forecast failure cause.
    blocked_reason = None
    if rc != 0:
        summary_path = proof_subdir / "l2_d02_validation_summary.json"
        try:
            blocked_reason = json.loads(summary_path.read_text()).get("blocked_reason")
        except Exception:
            blocked_reason = None
    print(f"[GPU] Finished: {run_id}  rc={rc}  elapsed={elapsed:.1f}s"
          + (f"  blocked_reason={blocked_reason}" if blocked_reason else ""), flush=True)
    return {"run_id": run_id, "returncode": rc, "elapsed_s": elapsed,
            "gpu_out_dir": str(gpu_out_dir), "blocked_reason": blocked_reason}


# ── GPU output validation ──────────────────────────────────────────────────────

def validate_gpu_output(gpu_out_dir: Path, run_id: str) -> dict:
    gpu_files = sorted(gpu_out_dir.glob("wrfout_d02_*"))
    n_files = len(gpu_files)
    expected = FORECAST_HOURS + 1
    nonfinite_hours: list[str] = []
    dtype_fp64 = False

    for i, path in enumerate(gpu_files):
        all_fin, bad_fields = check_finite_all_fields(path)
        if not all_fin:
            nonfinite_hours.append(f"{path.name}:{bad_fields}")
        if i == 0:
            with Dataset(path, "r") as ds:
                if "T2" in ds.variables:
                    dtype_fp64 = "float64" in str(ds.variables["T2"].dtype)

    valid = (n_files >= expected) and len(nonfinite_hours) == 0
    return {
        "run_id": run_id,
        "valid": valid,
        "n_files": n_files,
        "expected_files": expected,
        "nonfinite_hour_files": nonfinite_hours,
        "dtype_fp64_first_file": dtype_fp64,
    }


# ── Cell-level gridded scoring (GPU vs CPU-WRF) ────────────────────────────────

def score_cell_level(gpu_dir: Path, cpu_dir: Path, run_id: str, init_time: datetime) -> dict:
    CELL_TOL = {"T2": 2.0, "U10": 2.5, "V10": 2.5}
    lead_blocks = {"0-6h": (0, 6), "6-12h": (6, 12), "12-24h": (12, 24)}

    gpu_map = wrfout_map(gpu_dir)
    cpu_map = wrfout_map(cpu_dir)
    common = sorted(
        t for t in (set(gpu_map) & set(cpu_map))
        if 0 < (t - init_time).total_seconds() / 3600 <= FORECAST_HOURS
    )

    per_field: dict[str, dict] = {
        f: {"dsq": [], "dabs": [], "d": [],
            "px": 0., "py": 0., "pxy": 0., "px2": 0., "py2": 0., "pn": 0,
            "n": 0}
        for f in FIELDS
    }
    per_block: dict[str, dict[str, dict]] = {
        f: {blk: {"dsq": [], "dabs": [], "d": []} for blk in lead_blocks}
        for f in FIELDS
    }

    for t in common:
        lead_h = (t - init_time).total_seconds() / 3600
        for f in FIELDS:
            g = read_field(gpu_map[t], f)
            c = read_field(cpu_map[t], f)
            mask = np.isfinite(g) & np.isfinite(c)
            if not mask.any():
                continue
            gv, cv = g[mask].ravel(), c[mask].ravel()
            dv = gv - cv
            n = len(dv)
            acc = per_field[f]
            acc["dsq"].extend(dv ** 2)
            acc["dabs"].extend(np.abs(dv))
            acc["d"].extend(dv)
            acc["n"] += n
            acc["px"]  += float(np.sum(gv));  acc["py"]  += float(np.sum(cv))
            acc["pxy"] += float(np.sum(gv * cv))
            acc["px2"] += float(np.sum(gv ** 2)); acc["py2"] += float(np.sum(cv ** 2))
            acc["pn"]  += n
            for blk, (lo, hi) in lead_blocks.items():
                if lo < lead_h <= hi:
                    per_block[f][blk]["dsq"].extend(dv ** 2)
                    per_block[f][blk]["dabs"].extend(np.abs(dv))
                    per_block[f][blk]["d"].extend(dv)

    def summarize(dsq, dabs, d, n, tol):
        if n == 0:
            return {"n_cells": 0, "status": "NO_DATA"}
        dsqa = np.asarray(dsq); dabsa = np.asarray(dabs); da = np.asarray(d)
        return {
            "n_cells": int(n),
            "rmse":  float(np.sqrt(np.mean(dsqa))),
            "bias":  float(np.mean(da)),
            "mae":   float(np.mean(dabsa)),
            "p95":   float(np.percentile(dabsa, 95)),
            "p99":   float(np.percentile(dabsa, 99)),
            "max":   float(np.max(dabsa)),
            "frac_within_tol": float(np.mean(dabsa < tol)),
        }

    def pearson_r(acc):
        n = acc["pn"]
        if n < 2:
            return None
        sx, sy, sxy = acc["px"], acc["py"], acc["pxy"]
        sx2, sy2 = acc["px2"], acc["py2"]
        num = n * sxy - sx * sy
        den = math.sqrt(max((n * sx2 - sx**2) * (n * sy2 - sy**2), 0.0))
        return float(num / den) if den > 1e-30 else None

    field_stats = {}
    for f in FIELDS:
        acc = per_field[f]
        s = summarize(acc["dsq"], acc["dabs"], acc["d"], acc["n"], CELL_TOL[f])
        s["pearson_r"] = pearson_r(acc)
        s["by_lead_block"] = {
            blk: summarize(
                per_block[f][blk]["dsq"], per_block[f][blk]["dabs"],
                per_block[f][blk]["d"], len(per_block[f][blk]["d"]), CELL_TOL[f]
            )
            for blk in lead_blocks
        }
        field_stats[f] = s

    return {
        "run_id": run_id,
        "init_time_utc": init_time.isoformat(),
        "n_common_lead_hours": len(common),
        "cell_tol": CELL_TOL,
        "field_stats": field_stats,
    }


# ── Per-case scoring: TOST + cell ─────────────────────────────────────────────

def score_one_case(run_id: str, gpu_dir: Path, cpu_dir: Path,
                   init_time: datetime, validation: dict) -> dict:
    tost_pairs = paired_score(
        case_id=run_id,
        cpu_dir=cpu_dir,
        gpu_dir=gpu_dir,
        domain="d02",
        init=init_time,
        fh=FORECAST_HOURS,
        aemet_root=AEMET_ROOT,
    )
    cell = score_cell_level(gpu_dir, cpu_dir, run_id, init_time)
    return {
        "schema": "PoweredTOSTCaseResult",
        "schema_version": 1,
        "run_id": run_id,
        "init_time_utc": init_time.isoformat(),
        "gpu_dir": str(gpu_dir),
        "cpu_dir": str(cpu_dir),
        "gpu_validation": validation,
        "tost_pairs": tost_pairs,
        "cell_level": cell,
    }


# ── Pool cell stats across cases ───────────────────────────────────────────────

def pool_cell_stats(included_cases: list[str]) -> dict:
    lead_blocks = {"0-6h", "6-12h", "12-24h"}
    CELL_TOL = {"T2": 2.0, "U10": 2.5, "V10": 2.5}

    pooled = {f: {"mse_pairs": [], "mae_pairs": [], "bias_pairs": [],
                  "r_vals": [], "frac_vals": [],
                  "by_block": {blk: {"mse_pairs": [], "mae_pairs": [], "bias_pairs": []}
                               for blk in lead_blocks}}
              for f in FIELDS}

    for run_id in included_cases:
        cp = PROOF_DIR / f"case_{run_id}.json"
        if not cp.is_file():
            continue
        cd = json.loads(cp.read_text())
        for f in FIELDS:
            fs = cd.get("cell_level", {}).get("field_stats", {}).get(f, {})
            n = fs.get("n_cells", 0)
            if n == 0:
                continue
            pooled[f]["mse_pairs"].append((fs["rmse"] ** 2, n))
            pooled[f]["mae_pairs"].append((fs["mae"], n))
            pooled[f]["bias_pairs"].append((fs["bias"], n))
            r = fs.get("pearson_r")
            if r is not None:
                pooled[f]["r_vals"].append(r)
            frac = fs.get("frac_within_tol")
            if frac is not None:
                pooled[f]["frac_vals"].append(frac)
            for blk in lead_blocks:
                bs = fs.get("by_lead_block", {}).get(blk, {})
                bn = bs.get("n_cells", 0)
                if bn > 0:
                    pooled[f]["by_block"][blk]["mse_pairs"].append((bs["rmse"] ** 2, bn))
                    pooled[f]["by_block"][blk]["mae_pairs"].append((bs["mae"], bn))
                    pooled[f]["by_block"][blk]["bias_pairs"].append((bs["bias"], bn))

    def wmean(pairs):
        if not pairs:
            return None
        tw = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / tw if tw else None

    def wrmse(mse_pairs):
        m = wmean(mse_pairs)
        return float(math.sqrt(m)) if m is not None else None

    result = {}
    for f in FIELDS:
        p = pooled[f]
        n_cells_total = int(sum(w for _, w in p["mse_pairs"]))
        by_block = {}
        for blk in lead_blocks:
            bb = p["by_block"][blk]
            by_block[blk] = {
                "n_cells_pooled": int(sum(w for _, w in bb["mse_pairs"])),
                "rmse":  wrmse(bb["mse_pairs"]),
                "mae":   wmean(bb["mae_pairs"]),
                "bias":  wmean(bb["bias_pairs"]),
            }
        result[f] = {
            "n_cases": len(included_cases),
            "total_cells_pooled": n_cells_total,
            "rmse":  wrmse(p["mse_pairs"]),
            "mae":   wmean(p["mae_pairs"]),
            "bias":  wmean(p["bias_pairs"]),
            "pearson_r": float(np.mean(p["r_vals"])) if p["r_vals"] else None,
            "frac_within_tol": float(np.mean(p["frac_vals"])) if p["frac_vals"] else None,
            "cell_tol": CELL_TOL[f],
            "by_lead_block": by_block,
        }
    return result


# ── Markdown report ────────────────────────────────────────────────────────────

def build_report(tost_result: dict, cell_pooled: dict) -> str:
    lines = []
    lines.append("# v0.12.0 Powered n=15 TOST + Cell-Level Equivalence Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now(timezone.utc).isoformat()}*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## VERDICT BOX")
    lines.append("")
    pvf   = tost_result["per_field_verdict"]
    all_eq = tost_result["all_fields_equivalent"]
    n      = tost_result["n_included"]
    dtype_ok = tost_result.get("dtype_fp64_confirmed", "unknown")
    lines.append(f"**Achieved n**: {n} (out of 15 planned; {tost_result['n_excluded']} excluded)")
    lines.append(f"**fp64 dtype confirmed**: {dtype_ok}")
    lines.append(f"**Total GPU wall-clock**: {tost_result.get('total_gpu_wall_s',0)/3600:.2f} h")
    lines.append("")
    lines.append("### PILLAR A — ADR-029 Paired TOST (T2/U10/V10 vs AEMET)")
    lines.append("")
    lines.append("| Field | Margin | Mean delta | SD | 90% CI | p_lower | p_upper | n | Verdict |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for f in SCORE_VARS:
        v = pvf[f]
        ci = v.get("ci90")
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "N/A"
        md = v.get("mean_delta")
        sd = v.get("sd_delta")
        lines.append(
            f"| {f} | ±{v['margin']:.4f} | "
            f"{md:.4f} | {sd:.4f} | {ci_str} | "
            f"{v.get('p_lower', 'N/A'):.4f} | {v.get('p_upper', 'N/A'):.4f} | "
            f"{v.get('n', 'N/A')} | **{v['verdict']}** |"
        )
    lines.append("")
    lines.append(f"**Overall TOST**: {'ALL FIELDS EQUIVALENT' if all_eq else 'NOT ALL EQUIVALENT'}")
    lines.append("")

    lines.append("### PILLAR B — Cell-level numerical equivalence (pooled grid, GPU vs CPU-WRF)")
    lines.append("")
    lines.append("| Field | Total cells | RMSE | Bias | MAE | Pearson r | Frac within tol | Tol |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    def fmt(x): return f"{x:.4f}" if x is not None else "N/A"
    for f in FIELDS:
        cs = cell_pooled.get(f, {})
        lines.append(
            f"| {f} | {cs.get('total_cells_pooled', 0):,} | {fmt(cs.get('rmse'))} | "
            f"{fmt(cs.get('bias'))} | {fmt(cs.get('mae'))} | {fmt(cs.get('pearson_r'))} | "
            f"{fmt(cs.get('frac_within_tol'))} | {cs.get('cell_tol')} K/ms |"
        )
    lines.append("")

    lines.append("#### By lead block")
    lines.append("")
    for blk in ["0-6h", "6-12h", "12-24h"]:
        lines.append(f"**{blk}:**")
        lines.append("")
        lines.append("| Field | n cells | RMSE | Bias | MAE |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for f in FIELDS:
            b = cell_pooled.get(f, {}).get("by_lead_block", {}).get(blk, {})
            lines.append(
                f"| {f} | {b.get('n_cells_pooled', 0):,} | {fmt(b.get('rmse'))} | "
                f"{fmt(b.get('bias'))} | {fmt(b.get('mae'))} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## POWER CAVEAT (ADR-029 mandatory)")
    lines.append("")
    lines.append(
        "n=15 is underpowered for a 10% RMSE difference (ADR-029 planning value: n≈27 at α=0.05, β=0.20). "
        "Empirical SD determines actual power:"
    )
    planning_sd = {
        "T2": 0.20 * 2.148692978020805,
        "U10": 0.20 * 2.3064713972582307,
        "V10": 0.20 * 2.752320537920854,
    }
    lines.append("")
    lines.append("| Field | Empirical SD | Planning SD (20% benchmark) | Power status |")
    lines.append("| --- | ---: | ---: | --- |")
    for f in SCORE_VARS:
        emp_sd = pvf[f].get("sd_delta")
        psd = planning_sd[f]
        status = "adequate for this effect size" if emp_sd is not None and emp_sd < psd else "underpowered"
        lines.append(f"| {f} | {fmt(emp_sd)} | {psd:.4f} | {status} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Included / Excluded Cases")
    lines.append("")
    lines.append("### Included:")
    for cid in tost_result["included_cases"]:
        lines.append(f"- {cid}")
    lines.append("")
    lines.append("### Excluded:")
    for exc in tost_result["excluded_cases"]:
        lines.append(f"- {exc['run_id']}: {exc.get('reason', 'unknown')}")
    if not tost_result["excluded_cases"]:
        lines.append("- (none)")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Per-case paired deltas (ADR-029)")
    lines.append("")
    lines.append("| Case | T2 delta | U10 delta | V10 delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    deltas = tost_result.get("aggregate", {}).get("per_case_deltas", {})
    for i, cid in enumerate(tost_result["included_cases"]):
        t2d  = deltas.get("T2",  [])[i] if i < len(deltas.get("T2",  [])) else "N/A"
        u10d = deltas.get("U10", [])[i] if i < len(deltas.get("U10", [])) else "N/A"
        v10d = deltas.get("V10", [])[i] if i < len(deltas.get("V10", [])) else "N/A"
        lines.append(
            f"| {cid[:32]}... | {fmt(t2d) if isinstance(t2d, float) else t2d} | "
            f"{fmt(u10d) if isinstance(u10d, float) else u10d} | "
            f"{fmt(v10d) if isinstance(v10d, float) else v10d} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Proof Objects")
    lines.append("")
    lines.append(f"- `{PROOF_DIR}/powered_tost_result.json`")
    lines.append(f"- `{PROOF_DIR}/cell_level_stats.json`")
    lines.append(f"- `{PROOF_DIR}/case_<RUN_ID>.json` (15 files)")
    lines.append(f"- `{PROOF_DIR}/manifest.json`")
    lines.append(f"- `/tmp/v0120_powered_tost.done`")
    lines.append("")
    lines.append("## Data Wiring")
    lines.append("")
    lines.append("- GPU init: `/mnt/data/canairy_meteo/runs/wrf_l2/<RUN_ID>/`")
    lines.append("- CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/<RUN_ID>/` (73 d02 + 73 d01 hourly wrfout each)")
    lines.append("- AEMET obs: `/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations/`")
    lines.append("- GPU outputs: `/tmp/v0120_powered_tost_runs/l2_d02_<RUN_ID>/`")
    lines.append("- Precision: fp64 (JAX_ENABLE_X64=true)")
    lines.append("- Complete-pair deletion: rows missing GPU/CPU/obs dropped before RMSE.")
    lines.append("")
    return "\n".join(lines) + "\n"


# ── Main campaign loop ─────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-gpu",  action="store_true", help="Score from existing GPU wrfouts")
    ap.add_argument("--resume",    action="store_true", help="Skip cases with existing proof JSON")
    ap.add_argument("--case",      default=None,        help="Run only this case ID (debug)")
    ap.add_argument("--dry-run",   action="store_true", help="Print plan, no GPU/scoring")
    ap.add_argument("--allow-single", action="store_true",
                    help="Treat a 1-case scored run as success (rc=0); the "
                         "cross-case TOST aggregate is skipped (needs n>=2).")
    args = ap.parse_args(argv)

    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    GPU_RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[setup] Worktree root: {ROOT}", flush=True)
    print(f"[setup] SRC: {SRC}", flush=True)
    print(f"[setup] gpuwrf version check: ", end="", flush=True)
    import gpuwrf
    print(getattr(gpuwrf, "__version__", "no __version__"), flush=True)

    print("[setup] Preparing merged run root...", flush=True)
    merged_root = prepare_merged_run_root()
    print(f"[setup] Merged run root: {merged_root}", flush=True)

    case_ids = CASE_IDS if args.case is None else [args.case]

    if args.dry_run:
        print("\n[DRY-RUN] Plan:")
        for run_id in case_ids:
            merged_dir = merged_root / run_id
            cpu_dir    = L2_CPU_ROOT / run_id
            init_time  = parse_init_time(run_id)
            gpu_dir    = GPU_RUNS_ROOT / f"l2_d02_{run_id}"
            n_links    = len(list(merged_dir.iterdir())) if merged_dir.is_dir() else 0
            print(f"  {run_id}: merged={n_links} links  cpu_d02={len(list(cpu_dir.glob('wrfout_d02_*')))}  gpu_exists={gpu_dir.is_dir()}")
        print(f"\n[DRY-RUN] Proof dir: {PROOF_DIR}")
        print("[DRY-RUN] Done (no GPU runs).")
        return 0

    included_cases: list[str] = []
    excluded_cases: list[dict] = []
    all_tost_scores: list[dict] = []
    total_gpu_wall_s = 0.0
    dtype_fp64_confirmed = None

    for run_id in case_ids:
        print(f"\n{'='*60}", flush=True)
        print(f"CASE: {run_id}", flush=True)
        print(f"{'='*60}", flush=True)

        case_proof_path = PROOF_DIR / f"case_{run_id}.json"

        # Resume: load existing proof
        if args.resume and case_proof_path.is_file():
            print(f"[resume] Loading existing: {case_proof_path}", flush=True)
            existing = json.loads(case_proof_path.read_text())
            included_cases.append(run_id)
            all_tost_scores.append(existing["tost_pairs"])
            total_gpu_wall_s += existing.get("gpu_validation", {}).get("elapsed_s", 0.0)
            if dtype_fp64_confirmed is None:
                dtype_fp64_confirmed = existing.get("gpu_validation", {}).get("dtype_fp64_first_file")
            continue

        merged_dir = merged_root / run_id
        cpu_dir    = L2_CPU_ROOT / run_id
        gpu_dir    = GPU_RUNS_ROOT / f"l2_d02_{run_id}"
        proof_subdir = PROOF_DIR / "pipeline_proofs" / run_id

        # Prerequisite checks
        if not merged_dir.is_dir():
            reason = f"Merged dir missing: {merged_dir}"
            print(f"[EXCLUDE] {run_id}: {reason}", flush=True)
            excluded_cases.append({"run_id": run_id, "reason": reason})
            continue

        if not cpu_dir.is_dir():
            reason = f"CPU backfill dir missing: {cpu_dir}"
            print(f"[EXCLUDE] {run_id}: {reason}", flush=True)
            excluded_cases.append({"run_id": run_id, "reason": reason})
            continue

        n_cpu_d02 = len(list(cpu_dir.glob("wrfout_d02_*")))
        if n_cpu_d02 < FORECAST_HOURS + 1:
            reason = f"CPU d02 incomplete: {n_cpu_d02} files (need {FORECAST_HOURS+1})"
            print(f"[EXCLUDE] {run_id}: {reason}", flush=True)
            excluded_cases.append({"run_id": run_id, "reason": reason})
            continue

        init_time = parse_init_time(run_id)

        # ── GPU forecast ──
        gpu_forecast_result: dict = {"elapsed_s": 0.0, "returncode": 0}
        if not args.skip_gpu:
            existing_gpu = list(gpu_dir.glob("wrfout_d02_*"))
            if len(existing_gpu) >= FORECAST_HOURS + 1:
                print(f"[skip-run] GPU output exists ({len(existing_gpu)} files)", flush=True)
                gpu_forecast_result = {"run_id": run_id, "returncode": 0, "elapsed_s": 0.0, "skipped": True}
            else:
                gpu_forecast_result = run_gpu_forecast(run_id, gpu_dir, proof_subdir)
                total_gpu_wall_s += gpu_forecast_result.get("elapsed_s", 0.0)
                if gpu_forecast_result["returncode"] != 0:
                    # One retry
                    print(f"[RETRY] {run_id}: rc={gpu_forecast_result['returncode']}", flush=True)
                    gpu_forecast_result = run_gpu_forecast(run_id, gpu_dir, proof_subdir)
                    total_gpu_wall_s += gpu_forecast_result.get("elapsed_s", 0.0)
                    if gpu_forecast_result["returncode"] != 0:
                        br = gpu_forecast_result.get("blocked_reason")
                        reason = (f"GPU forecast failed (2 attempts) "
                                  f"rc={gpu_forecast_result['returncode']}"
                                  + (f": {br}" if br else ""))
                        print(f"[EXCLUDE] {run_id}: {reason}", flush=True)
                        excluded_cases.append({"run_id": run_id, "reason": reason,
                                               "blocked_reason": br})
                        continue
        else:
            if not gpu_dir.is_dir():
                reason = f"--skip-gpu but GPU dir missing: {gpu_dir}"
                print(f"[EXCLUDE] {run_id}: {reason}", flush=True)
                excluded_cases.append({"run_id": run_id, "reason": reason})
                continue

        # ── Validate GPU output ──
        validation = validate_gpu_output(gpu_dir, run_id)
        validation["elapsed_s"] = gpu_forecast_result.get("elapsed_s", 0.0)
        if dtype_fp64_confirmed is None:
            dtype_fp64_confirmed = validation["dtype_fp64_first_file"]
            print(f"[dtype] fp64 confirmed = {dtype_fp64_confirmed} (first case)", flush=True)

        if not validation["valid"]:
            reason = (f"GPU output invalid: n={validation['n_files']}, "
                      f"nonfinite={validation['nonfinite_hour_files'][:3]}")
            print(f"[EXCLUDE] {run_id}: {reason}", flush=True)
            excluded_cases.append({"run_id": run_id, "reason": reason, "validation": validation})
            continue

        # ── Score ──
        print(f"[score] Scoring {run_id}...", flush=True)
        try:
            case_result = score_one_case(run_id, gpu_dir, cpu_dir, init_time, validation)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            reason = f"Scoring error: {type(exc).__name__}: {exc}"
            print(f"[EXCLUDE] {run_id}: {reason}", flush=True)
            excluded_cases.append({"run_id": run_id, "reason": reason})
            continue

        write_json(case_proof_path, case_result)
        included_cases.append(run_id)
        all_tost_scores.append(case_result["tost_pairs"])

        # Commit per-case proof
        subprocess.run(
            ["git", "-C", str(ROOT), "add", str(case_proof_path)],
            capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(ROOT), "commit", "-m",
             f"[powered-tost-v0120] case {run_id[:16]} scored"],
            capture_output=True
        )
        print(f"[commit] {run_id[:16]}", flush=True)

    # ── Aggregate TOST ──
    print(f"\n{'='*60}", flush=True)
    print(f"AGGREGATE TOST (n={len(all_tost_scores)} cases)", flush=True)
    print(f"{'='*60}", flush=True)

    n_scored = len(all_tost_scores)
    # rc semantics (FIXED — the v0.12.0 rc=2 abort conflated three states):
    #   0 cases scored  -> genuine failure (forecast/scoring all blocked) -> rc=2
    #   1 case  scored  -> SCORING PATH PROVEN; cross-case TOST needs n>=2.
    #                      In single-case debug (--case / --allow-single) this is
    #                      a SUCCESS (rc=0); in a full 15-case campaign it means
    #                      14 cases were excluded -> still report rc=2 so the
    #                      manager sees an under-powered campaign, but the
    #                      per-case proof was already written + committed.
    #   >=2 cases       -> full TOST aggregate -> rc=0
    if n_scored == 0:
        print("[ABORT] 0 cases scored — every case was excluded. "
              "rc=2. Exclusion reasons:", flush=True)
        for exc in excluded_cases:
            print(f"  - {exc['run_id']}: {exc.get('reason', 'unknown')}", flush=True)
        # Persist a machine-readable abort summary for the manager.
        write_json(PROOF_DIR / "powered_tost_abort.json", {
            "schema": "PoweredTOSTN15Abort",
            "schema_version": 1,
            "verdict": "ABORT_NO_CASES_SCORED",
            "n_included": 0,
            "n_excluded": len(excluded_cases),
            "excluded_cases": excluded_cases,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        })
        return 2

    if n_scored < 2:
        # One case scored cleanly: the SCORING path is proven end-to-end.
        single = included_cases[0]
        print(f"[SCORING-OK] 1 case scored end-to-end: {single}", flush=True)
        print("[INFO] cross-case TOST aggregate needs n>=2 — skipped.", flush=True)
        write_json(PROOF_DIR / "powered_tost_single_case.json", {
            "schema": "PoweredTOSTN15SingleCase",
            "schema_version": 1,
            "verdict": "SCORING_PATH_PROVEN_N1",
            "scored_case": single,
            "n_included": 1,
            "n_excluded": len(excluded_cases),
            "excluded_cases": excluded_cases,
            "note": ("Per-case TOST + cell-level scoring completed rc=0. The "
                     "cross-case equivalence TOST requires n>=2 cases; rerun "
                     "with more cases to compute it."),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        })
        # Single-case debug runs (--case or --allow-single) are a SUCCESS;
        # a full campaign that collapsed to 1 case is reported as rc=2.
        if args.case is not None or args.allow_single:
            return 0
        print("[WARN] full campaign yielded only 1 scored case (14 excluded); "
              "returning rc=2 (under-powered).", flush=True)
        return 2

    agg = aggregate_tost(all_tost_scores)

    per_field_verdict = {}
    for f in SCORE_VARS:
        res = agg["tost"][f]
        equiv   = res.get("equivalent", False)
        mean_d  = res.get("mean_delta")
        margin  = MARGINS[f]
        if res.get("status") == "INSUFFICIENT_N":
            verdict = "INSUFFICIENT_N"
        elif equiv:
            verdict = "EQUIVALENT"
        elif mean_d is not None and abs(mean_d) > margin:
            verdict = "NOT_EQUIVALENT_OUTSIDE_MARGIN"
        else:
            verdict = "NOT_EQUIVALENT_CI_NOT_INSIDE_MARGIN"
        per_field_verdict[f] = {**res, "margin": margin, "verdict": verdict}

    tost_result = {
        "schema": "PoweredTOSTN15Result",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "adr": "ADR-029",
        "code_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
        ).decode().strip(),
        "margins": MARGINS,
        "included_cases":       included_cases,
        "excluded_cases":       excluded_cases,
        "n_included":           len(included_cases),
        "n_excluded":           len(excluded_cases),
        "dtype_fp64_confirmed": dtype_fp64_confirmed,
        "total_gpu_wall_s":     total_gpu_wall_s,
        "aggregate":            agg,
        "per_field_verdict":    per_field_verdict,
        "all_fields_equivalent": all(v["verdict"] == "EQUIVALENT" for v in per_field_verdict.values()),
    }
    write_json(PROOF_DIR / "powered_tost_result.json", tost_result)

    # Pool cell-level stats
    cell_pooled = pool_cell_stats(included_cases)
    write_json(PROOF_DIR / "cell_level_stats.json", cell_pooled)

    # Write markdown report
    report = build_report(tost_result, cell_pooled)
    report_path = PROOF_DIR / "POWERED_TOST_AND_CELL_STATS.md"
    report_path.write_text(report, encoding="utf-8")

    # Done file
    done_line = (
        f"T2={per_field_verdict['T2']['verdict']} "
        f"U10={per_field_verdict['U10']['verdict']} "
        f"V10={per_field_verdict['V10']['verdict']} "
        f"n={len(included_cases)}"
    )
    Path("/tmp/v0120_powered_tost.done").write_text(done_line + "\n", encoding="utf-8")
    print(f"\n[DONE] {done_line}", flush=True)
    print(f"Report: {report_path}", flush=True)

    # Final commit
    subprocess.run(
        ["git", "-C", str(ROOT), "add",
         str(PROOF_DIR / "powered_tost_result.json"),
         str(PROOF_DIR / "cell_level_stats.json"),
         str(report_path)],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(ROOT), "commit", "-m",
         f"[powered-tost-v0120] n15 TOST+cell final — {done_line[:80]}"],
        capture_output=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
