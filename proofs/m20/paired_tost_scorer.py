#!/usr/bin/env python3
"""M20/M21 paired-TOST scorer (ADR-029).

This is the piece the existing M7 scorer (src/gpuwrf/validation/forecast_vs_obs.py) does
NOT provide: it forms CPU/GPU/obs COMPLETE PAIRS on the same station x valid_time mask,
computes per-(case, lead-block, variable) RMSE for both sides, takes the paired delta
RMSE_GPU - RMSE_CPU, and runs paired TOST against the ADR-029 predeclared margins.

It is forward-looking infrastructure: it runs the moment a GPU forecast wrfout exists for a
case. Until then it can be exercised in a CPU-vs-CPU self-check (delta should be ~0).

Reuses the frozen M7 interpolation/score primitives (no model code touched):
  interpolate_to_stations, load_aemet_observations, compute_station_scores

CLI:
  taskset -c 0-3 python3 proofs/m20/paired_tost_scorer.py \
      --cpu-run <cpu_wrfout_dir> --gpu-run <gpu_wrfout_dir> --case-id <id> \
      --out proofs/m20/paired_scores_<id>.json
  (omit --gpu-run for a CPU-vs-CPU plumbing self-test)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.validation.forecast_vs_obs import (  # noqa: E402
    DEFAULT_AEMET_ROOT,
    interpolate_to_stations,
    load_aemet_observations,
)

SCORE_VARS = ("T2", "U10", "V10")
LEAD_BLOCKS = {"0-24h": (0, 24), "24-48h": (24, 48), "48-72h": (48, 72)}

# ADR-029 predeclared TOST equivalence margins (10% of CPU WRF RMSE benchmark).
MARGINS = {"T2": 0.2148692978020805, "U10": 0.23064713972582307, "V10": 0.2752320537920854}
MIN_PAIRS_PER_BLOCK = 30  # predeclared exclusion threshold

WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")


def list_frames(run_dir: Path, domain: str) -> list[tuple[datetime, Path]]:
    out = []
    for p in sorted(run_dir.glob(f"wrfout_{domain}_*")):
        m = WRFOUT_RE.match(p.name)
        if m and p.is_file():
            vt = datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
            out.append((vt, p))
    return sorted(out)


def station_metadata_from_obs(obs: pd.DataFrame) -> pd.DataFrame:
    md = obs.dropna(subset=["lat", "lon"]).drop_duplicates("station_id")
    return md[["station_id", "lat", "lon"] + (["elev_m"] if "elev_m" in md.columns else [])].reset_index(drop=True)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def score_run(run_dir: Path, domain: str, station_md: pd.DataFrame,
              init: datetime, fh: int) -> pd.DataFrame:
    """Interpolate every frame to stations -> long table [station_id,time,lead,var,value]."""
    frames = list_frames(run_dir, domain)
    rows = []
    for vt, path in frames:
        lead = int(round((vt - init).total_seconds() / 3600.0))
        if lead < 0 or lead > fh:
            continue
        interp = interpolate_to_stations(path, station_md, variables=SCORE_VARS, valid_time=vt)
        for var in SCORE_VARS:
            sub = interp[["station_id", "time", var]].rename(columns={var: "value"})
            sub["lead"] = lead
            sub["var"] = var
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["station_id", "time", "value", "lead", "var"])


def paired_score(case_id: str, cpu_dir: Path, gpu_dir: Path | None, domain: str,
                 init: datetime, fh: int, aemet_root: Path) -> dict:
    obs = load_aemet_observations(aemet_root, variables=SCORE_VARS,
                                  start_time=init.isoformat(),
                                  end_time=(init + timedelta(hours=fh)).isoformat())
    obs = obs[obs["time"].dt.minute == 0]  # hourly obs land on the hour
    station_md = station_metadata_from_obs(obs)

    cpu = score_run(cpu_dir, domain, station_md, init, fh)
    gpu = score_run(gpu_dir, domain, station_md, init, fh) if gpu_dir else cpu.copy()
    self_test = gpu_dir is None

    obs_long = []
    for var in SCORE_VARS:
        if var in obs.columns:
            o = obs[["station_id", "time", var]].rename(columns={var: "obs"})
            o["var"] = var
            obs_long.append(o)
    obs_l = pd.concat(obs_long, ignore_index=True) if obs_long else pd.DataFrame(
        columns=["station_id", "time", "obs", "var"])
    obs_l["time"] = pd.to_datetime(obs_l["time"], utc=True).dt.floor("h")
    cpu["time"] = pd.to_datetime(cpu["time"], utc=True).dt.floor("h")
    gpu["time"] = pd.to_datetime(gpu["time"], utc=True).dt.floor("h")

    cpu = cpu.rename(columns={"value": "cpu"})
    gpu = gpu.rename(columns={"value": "gpu"})
    merged = (cpu.merge(gpu, on=["station_id", "time", "lead", "var"], how="inner")
                 .merge(obs_l, on=["station_id", "time", "var"], how="inner"))
    # COMPLETE-PAIR DELETION: drop any row missing any of cpu/gpu/obs
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["cpu", "gpu", "obs"])

    per_block = {}
    for var in SCORE_VARS:
        per_block[var] = {}
        for blk, (lo, hi) in LEAD_BLOCKS.items():
            m = merged[(merged["var"] == var) & (merged["lead"] > lo) & (merged["lead"] <= hi)]
            n = len(m)
            if n < MIN_PAIRS_PER_BLOCK:
                per_block[var][blk] = {"status": "EXCLUDED_LOW_N", "n_pairs": int(n)}
                continue
            cpu_rmse = rmse(m["cpu"].to_numpy(), m["obs"].to_numpy())
            gpu_rmse = rmse(m["gpu"].to_numpy(), m["obs"].to_numpy())
            per_block[var][blk] = {
                "status": "OK",
                "n_pairs": int(n),
                "cpu_rmse": cpu_rmse,
                "gpu_rmse": gpu_rmse,
                "paired_delta_rmse": gpu_rmse - cpu_rmse,
            }
    return {
        "schema": "M20PairedCaseScore",
        "schema_version": 1,
        "case_id": case_id,
        "self_test_cpu_vs_cpu": self_test,
        "domain": domain,
        "init_time_utc": init.isoformat(),
        "forecast_hours": fh,
        "total_complete_pairs": int(len(merged)),
        "per_block": per_block,
    }


def tost_on_deltas(deltas: list[float], margin: float, alpha: float = 0.05) -> dict:
    """Paired TOST: H0a: mu <= -margin, H0b: mu >= +margin. Equivalence iff BOTH reject."""
    from scipy import stats
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    n = d.size
    if n < 2:
        return {"status": "INSUFFICIENT_N", "n": int(n)}
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0:
        equivalent = abs(mean) < margin
        return {"status": "ZERO_VARIANCE", "n": n, "mean": mean,
                "equivalent": bool(equivalent), "p_lower": 0.0 if mean > -margin else 1.0,
                "p_upper": 0.0 if mean < margin else 1.0}
    t_lower = (mean - (-margin)) / se      # test H0a (reject if t_lower large +)
    t_upper = (mean - margin) / se         # test H0b (reject if t_upper large -)
    p_lower = float(stats.t.sf(t_lower, df))   # P(T > t_lower)
    p_upper = float(stats.t.cdf(t_upper, df))  # P(T < t_upper)
    ci_lo = mean - stats.t.ppf(1 - alpha, df) * se
    ci_hi = mean + stats.t.ppf(1 - alpha, df) * se
    equivalent = (p_lower < alpha) and (p_upper < alpha)
    return {"status": "OK", "n": int(n), "mean_delta": mean, "sd_delta": sd,
            "margin": margin, "p_lower": p_lower, "p_upper": p_upper,
            "tost_p": max(p_lower, p_upper), "ci90": [ci_lo, ci_hi],
            "equivalent": bool(equivalent)}


def aggregate_tost(case_scores: list[dict]) -> dict:
    """Collect per-case paired deltas (averaged over lead blocks per variable) -> TOST."""
    by_var = {v: [] for v in SCORE_VARS}
    for cs in case_scores:
        for var in SCORE_VARS:
            blocks = cs["per_block"].get(var, {})
            ok = [b["paired_delta_rmse"] for b in blocks.values() if b.get("status") == "OK"]
            if ok:
                by_var[var].append(float(np.mean(ok)))
    result = {v: tost_on_deltas(by_var[v], MARGINS[v]) for v in SCORE_VARS}
    sigma_emp = {v: (float(np.std(by_var[v], ddof=1)) if len(by_var[v]) > 1 else None) for v in SCORE_VARS}
    all_equiv = all(result[v].get("equivalent") for v in SCORE_VARS)
    return {
        "schema": "M21TostResult",
        "schema_version": 1,
        "n_cases": {v: len(by_var[v]) for v in SCORE_VARS},
        "per_case_deltas": by_var,
        "empirical_sigma": sigma_emp,
        "tost": result,
        "all_variables_equivalent": bool(all_equiv),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-run", required=True, type=Path)
    ap.add_argument("--gpu-run", type=Path, default=None,
                    help="GPU wrfout dir; omit for CPU-vs-CPU plumbing self-test")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--domain", default="d02")
    ap.add_argument("--init", required=True, help="ISO init time, e.g. 2026-05-21T18:00:00+00:00")
    ap.add_argument("--fh", type=int, required=True)
    ap.add_argument("--aemet-root", type=Path, default=DEFAULT_AEMET_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    init = datetime.fromisoformat(a.init)
    res = paired_score(a.case_id, a.cpu_run, a.gpu_run, a.domain, init, a.fh, a.aemet_root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({"case_id": a.case_id, "total_pairs": res["total_complete_pairs"],
                      "per_block": {v: {k: b.get("paired_delta_rmse", b.get("status"))
                                        for k, b in res["per_block"][v].items()} for v in SCORE_VARS}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
