#!/usr/bin/env python3
"""v0.2.0 PARQUET-grounded paired-TOST scorer (ADR-029 margins).

TRUTH SOURCE (the honest one)
-----------------------------
The CPU-WRF station truth is the 31-case POINT-SHADOW PARQUET corpus:

    <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank/_all_cases_point_shadows.parquet

It carries CPU-WRF ``wrf_phase14c_{t2_k,u10,v10}`` interpolated to 278 station
points (4 nests: d02, d03, d04, d05; 278 stations; d02 73 lead-h, d03/d04/d05
25 lead-h), with **0% NaN on T2/U10/V10**.  Critically it ALSO stores, per row,
the CPU run's ``nearest_grid_iy``/``nearest_grid_ix`` (interpolation_method ==
``nearest_grid_cell``) -- the exact grid cell the CPU extraction read.  This is
the key to a NON-GAMEABLE GPU-vs-CPU pairing: the GPU forecast is sampled at the
SAME (iy, ix) cell, so both sides use the identical nearest-grid-cell rule and
the difference is purely model-vs-model -- no interpolation-method mismatch, no
obs-sampling noise, no synthetic truth.

This is the literal reading of "score GPU-replay-at-stations vs the parquet
corpus": the parquet IS the CPU-WRF truth at the stations, and the TOST tests
whether the GPU port REPRODUCES that CPU-WRF station forecast within the
predeclared ADR-029 margins.

WHY DIRECT GPU-vs-CPU (not RMSE-vs-obs)
---------------------------------------
ADR-029's nominal delta is ``RMSE_GPU_vs_obs - RMSE_CPU_vs_obs``.  That requires
a common OBS reference.  The parquet has NO obs (obs live in the AEMET station
parquets, largely daily for the high-quality pool, ~106 of the 278 stations).
The parquet's value is the CPU-WRF point shadow itself.  The defensible,
lossless, full-278-station equivalence question the parquet answers is the
DIRECT one: does the GPU reproduce CPU-WRF at the stations?  We therefore form
the paired statistic on ``GPU - CPU`` directly and apply the FROZEN ADR-029
margins as the equivalence band.  The ADR-029 paired STRUCTURE is preserved
(per case x domain x lead-block x variable; complete-pair deletion; case-level
delta -> TOST over cases).

PAIRED STATISTIC + TOST
-----------------------
For each (case, domain, lead-block, variable) we form complete pairs on
(station, valid_time) where BOTH the parquet CPU value and the GPU sample are
finite, then:
  * mean_bias  = mean(GPU - CPU)              (signed; for descriptive context)
  * repro_rmse = sqrt(mean((GPU - CPU)^2))    (the reproduction error)
The per-case TOST delta for a variable is the lead-block-averaged repro_rmse.
TOST (two one-sided t-tests, alpha=0.05) on the per-case repro_rmse against the
ADR-029 margin: equivalence is accepted iff the per-case reproduction RMSE is
statistically below the predeclared margin for every required variable.

HONEST n-CAP (load-bearing -- read the FINDINGS report)
-------------------------------------------------------
The parquet truth is n=31 cases, but the GPU REPLAY forecast must be initialised
from the t=0 ``wrfout`` snapshot + the parent hourly ``wrfout`` boundary history
(gpuwrf.integration.d02_replay.build_replay_case), which was PURGED from all but
3 d02 cases (0509, 0521, 0530) + a few partial d03.  So the *scoreable* GPU-side
n is init-capped at the wrfout-retaining cases UNLESS CPU-WRF is re-run from the
retained met_em forcing to regenerate wrfout.  This scorer scores EVERY case for
which a GPU emit exists; the harness (tost_parquet_runner.py) only runs the GPU
forecast where the replay init data survives.

CLI (scoring only -- no GPU; pairs an existing GPU emit dir against the parquet):
  JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 python3 \
    proofs/m20/tost_parquet_scorer.py \
      --gpu-root proofs/m20/tost_run/gpu_wrfout_parquet \
      --out proofs/m20/tost_run/tost_parquet.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

PARQUET = Path("<DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank/"
               "_all_cases_point_shadows.parquet")

# ADR-029 predeclared TOST equivalence margins (10% of CPU WRF RMSE benchmark).
# T2 +/-0.2148692978020805 K, U10 +/-0.23064713972582307 m/s,
# V10 +/-0.2752320537920854 m/s.  FROZEN.  NOT loosened.
MARGINS = {"T2": 0.2148692978020805,
           "U10": 0.23064713972582307,
           "V10": 0.2752320537920854}
# parquet CPU column -> our variable name
CPU_COL = {"T2": "wrf_phase14c_t2_k", "U10": "wrf_phase14c_u10",
           "V10": "wrf_phase14c_v10"}
SCORE_VARS = ("T2", "U10", "V10")
# d02 spans 72 h; the nested nests span 24 h.  Use the same 24-h block grid the
# v0.1.0 / ADR-029 design used; d02 adds the 24-48 and 48-72 blocks.
LEAD_BLOCKS = {"0-24h": (0, 24), "24-48h": (24, 48), "48-72h": (48, 72)}
MIN_PAIRS_PER_BLOCK = 30  # ADR-029 predeclared exclusion threshold

WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")


# ---------------------------------------------------------------------------
# Parquet CPU truth.
# ---------------------------------------------------------------------------
def load_cpu_truth(parquet: Path = PARQUET) -> pd.DataFrame:
    """Load the CPU-WRF point-shadow truth, one row per (case, domain, station,
    lead).  Keeps the nearest-grid index so the GPU side can sample the SAME
    grid cell.  Returns a long [case_id, domain, station_id, lead_hour,
    valid_time_utc, iy, ix, T2, U10, V10] frame."""
    cols = ["case_id", "domain", "station_id", "lead_hour", "issue_time_utc",
            "valid_time_utc", "nearest_grid_iy", "nearest_grid_ix",
            "inside_domain_bbox", *CPU_COL.values()]
    df = pd.read_parquet(parquet, columns=cols)
    df = df.rename(columns={v: k for k, v in CPU_COL.items()})
    df = df.rename(columns={"nearest_grid_iy": "iy", "nearest_grid_ix": "ix"})
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True)
    df["lead_hour"] = df["lead_hour"].astype(int)
    return df


# ---------------------------------------------------------------------------
# GPU emit (full-grid NetCDF per lead) -> sample at the parquet's (iy, ix).
# ---------------------------------------------------------------------------
def _emit(out_dir: Path, domain: str, valid: datetime,
          lat: np.ndarray, lon: np.ndarray,
          t2: np.ndarray, u10: np.ndarray, v10: np.ndarray) -> Path:
    """Write a minimal full-grid WRF-style NetCDF (T2/U10/V10/XLAT/XLONG) the
    parquet scorer can sample at (iy, ix).  Leading Time axis matches real
    wrfout layout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"wrfout_{domain}_{valid:%Y-%m-%d_%H:%M:%S}"
    ny, nx = np.asarray(t2).shape
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.Times = valid.strftime("%Y-%m-%d_%H:%M:%S")

        def _v(name, arr):
            var = ds.createVariable(name, "f4",
                                    ("Time", "south_north", "west_east"))
            var[0, :, :] = np.asarray(arr, dtype=np.float32)

        _v("XLAT", lat); _v("XLONG", lon)
        _v("T2", t2); _v("U10", u10); _v("V10", v10)
    return path


def list_gpu_frames(gpu_dir: Path, domain: str) -> dict[datetime, Path]:
    frames: dict[datetime, Path] = {}
    if not gpu_dir.is_dir():
        return frames
    for p in sorted(gpu_dir.glob(f"wrfout_{domain}_*")):
        m = WRFOUT_RE.match(p.name)
        if m and p.is_file():
            vt = datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(
                tzinfo=timezone.utc)
            frames[vt] = p
    return frames


def _read_2d(ds: Dataset, name: str) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    arr = np.asarray(ds.variables[name][:], dtype=np.float64)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def sample_gpu_at_stations(gpu_dir: Path, domain: str,
                           cpu_case: pd.DataFrame) -> pd.DataFrame:
    """For each GPU frame (valid_time), sample T2/U10/V10 at the (iy, ix) cells
    used by the CPU parquet for THIS case.  Returns a long
    [station_id, valid_time_utc, lead_hour, var, gpu] frame."""
    gpu_frames = list_gpu_frames(gpu_dir, domain)
    if not gpu_frames:
        return pd.DataFrame(columns=["station_id", "valid_time_utc",
                                     "lead_hour", "var", "gpu"])
    rows = []
    for vt, path in gpu_frames.items():
        sub = cpu_case[cpu_case["valid_time_utc"] == vt]
        if sub.empty:
            continue
        with Dataset(path) as ds:
            grids = {v: _read_2d(ds, v) for v in SCORE_VARS}
        if any(g is None for g in grids.values()):
            continue
        ny, nx = next(iter(grids.values())).shape
        iy = sub["iy"].to_numpy()
        ix = sub["ix"].to_numpy()
        # only cells inside the GPU grid (defensive; aligned cases satisfy this)
        ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        ss = sub[ok]
        iy, ix = iy[ok], ix[ok]
        for var in SCORE_VARS:
            vals = grids[var][iy, ix]
            rows.append(pd.DataFrame({
                "station_id": ss["station_id"].to_numpy(),
                "valid_time_utc": ss["valid_time_utc"].to_numpy(),
                "lead_hour": ss["lead_hour"].to_numpy(),
                "var": var, "gpu": vals}))
    return (pd.concat(rows, ignore_index=True) if rows else
            pd.DataFrame(columns=["station_id", "valid_time_utc", "lead_hour",
                                  "var", "gpu"]))


# ---------------------------------------------------------------------------
# Per-case pairing + block statistics (ADR-029 structure).
# ---------------------------------------------------------------------------
def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def score_case(case_id: str, domain: str, gpu_dir: Path,
               cpu_truth: pd.DataFrame) -> dict:
    cpu_case = cpu_truth[(cpu_truth["case_id"] == case_id) &
                         (cpu_truth["domain"] == domain)].copy()
    gpu_long = sample_gpu_at_stations(gpu_dir, domain, cpu_case)

    cpu_long = []
    for var in SCORE_VARS:
        c = cpu_case[["station_id", "valid_time_utc", "lead_hour", var]].rename(
            columns={var: "cpu"})
        c["var"] = var
        cpu_long.append(c)
    cpu_l = (pd.concat(cpu_long, ignore_index=True) if cpu_long else
             pd.DataFrame(columns=["station_id", "valid_time_utc", "lead_hour",
                                   "var", "cpu"]))

    merged = cpu_l.merge(gpu_long,
                         on=["station_id", "valid_time_utc", "lead_hour", "var"],
                         how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["cpu", "gpu"])

    per_block: dict[str, dict] = {}
    for var in SCORE_VARS:
        per_block[var] = {}
        for blk, (lo, hi) in LEAD_BLOCKS.items():
            m = merged[(merged["var"] == var) &
                       (merged["lead_hour"] > lo) & (merged["lead_hour"] <= hi)]
            n = len(m)
            if n < MIN_PAIRS_PER_BLOCK:
                per_block[var][blk] = {"status": "EXCLUDED_LOW_N",
                                       "n_pairs": int(n)}
                continue
            d = (m["gpu"] - m["cpu"]).to_numpy()
            per_block[var][blk] = {
                "status": "OK", "n_pairs": int(n),
                "mean_bias": float(np.mean(d)),
                "repro_rmse": _rmse(d),
                "cpu_mean": float(m["cpu"].mean()),
                "gpu_mean": float(m["gpu"].mean()),
            }
    return {
        "schema": "V020ParquetCaseScore", "schema_version": 1,
        "case_id": case_id, "domain": domain,
        "truth_source": "wrf_case_bank parquet point-shadow (CPU-WRF "
                        "nearest_grid_cell)",
        "pairing": "GPU sampled at the parquet's nearest_grid_iy/ix; "
                   "complete-pair on (station, valid_time, var)",
        "total_complete_pairs": int(len(merged)),
        "per_block": per_block,
    }


def tost_on_values(values: list[float], margin: float,
                   alpha: float = 0.05) -> dict:
    """One-sided-equivalence TOST on a strictly-non-negative case statistic
    (the per-case reproduction RMSE).  H0: mu >= margin; equivalence iff the
    upper one-sided test rejects (mean + t*se < margin), i.e. the per-case
    reproduction error is significantly below the predeclared band.  Reports
    the symmetric-band lower test too for ADR-029 parity."""
    from scipy import stats
    d = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = d.size
    if n < 2:
        return {"status": "INSUFFICIENT_N", "n": int(n)}
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    if se == 0:
        return {"status": "ZERO_VARIANCE", "n": int(n), "mean": mean,
                "equivalent": bool(mean < margin), "margin": margin}
    t_upper = (mean - margin) / se
    p_upper = float(stats.t.cdf(t_upper, df))
    t_lower = (mean - (-margin)) / se
    p_lower = float(stats.t.sf(t_lower, df))
    ci_lo = mean - stats.t.ppf(1 - alpha, df) * se
    ci_hi = mean + stats.t.ppf(1 - alpha, df) * se
    equivalent = (p_upper < alpha) and (p_lower < alpha)
    return {"status": "OK", "n": int(n), "mean_repro_rmse": mean,
            "sd": sd, "margin": margin,
            "p_upper": p_upper, "p_lower": p_lower,
            "tost_p": max(p_upper, p_lower),
            "ci90": [ci_lo, ci_hi], "equivalent": bool(equivalent)}


def aggregate_tost(case_scores: list[dict]) -> dict:
    by_var: dict[str, list[float]] = {v: [] for v in SCORE_VARS}
    for cs in case_scores:
        for var in SCORE_VARS:
            blocks = cs["per_block"].get(var, {})
            ok = [b["repro_rmse"] for b in blocks.values()
                  if b.get("status") == "OK"]
            if ok:
                by_var[var].append(float(np.mean(ok)))
    tost = {v: tost_on_values(by_var[v], MARGINS[v]) for v in SCORE_VARS}
    sigma = {v: (float(np.std(by_var[v], ddof=1)) if len(by_var[v]) > 1 else None)
             for v in SCORE_VARS}
    all_equiv = all(tost[v].get("equivalent") for v in SCORE_VARS)
    return {
        "schema": "V020ParquetTostResult", "schema_version": 1,
        "n_cases": {v: len(by_var[v]) for v in SCORE_VARS},
        "per_case_repro_rmse": by_var,
        "empirical_sigma": sigma,
        "margins_frozen_adr029": MARGINS,
        "tost": tost,
        "all_variables_equivalent": bool(all_equiv),
    }


# ---------------------------------------------------------------------------
# Driver (scoring only; no GPU).
# ---------------------------------------------------------------------------
def discover_units(gpu_root: Path) -> list[str]:
    if not gpu_root.is_dir():
        return []
    return sorted(d.name for d in gpu_root.iterdir() if d.is_dir())


def parse_unit(uid: str) -> tuple[str, str]:
    """unit dir name '<case_id>__<domain>' -> (case_id, domain)."""
    if "__" in uid:
        cid, dom = uid.rsplit("__", 1)
        return cid, dom
    return uid, "d02"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-root", type=Path, required=True,
                    help="dir of <case_id>__<domain>/ GPU emit subdirs")
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--units", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    cpu_truth = load_cpu_truth(a.parquet)
    units = a.units if a.units else discover_units(a.gpu_root)

    case_scores = []
    per_unit = []
    for uid in units:
        gpu_dir = a.gpu_root / uid
        cid, dom = parse_unit(uid)
        cs = score_case(cid, dom, gpu_dir, cpu_truth)
        per_unit.append({"unit_id": uid, "case_id": cid, "domain": dom,
                         "total_complete_pairs": cs["total_complete_pairs"]})
        case_scores.append(cs)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        (a.out.parent / f"paired_parquet_{uid}.json").write_text(
            json.dumps(cs, indent=2, default=str) + "\n")

    agg = aggregate_tost(case_scores)
    payload = {
        "schema": "V020ParquetTostCampaign", "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "truth_source": str(a.parquet),
        "truth_kind": "31-case CPU-WRF point-shadow parquet "
                      "(nearest_grid_cell), 278 stations, 4 nests, 0% NaN",
        "gpu_root": str(a.gpu_root),
        "margins_frozen_adr029": MARGINS,
        "min_pairs_per_block": MIN_PAIRS_PER_BLOCK,
        "lead_blocks": LEAD_BLOCKS,
        "per_unit": per_unit,
        "tost_aggregate": agg,
        "verdict": ("EQUIVALENT" if agg["all_variables_equivalent"]
                    else "NOT_EQUIVALENT_OR_UNDERPOWERED"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"parquet-TOST -> {a.out}")
    print(f"  units: {[u['unit_id'] for u in per_unit]}")
    print(f"  n_cases per var: {agg['n_cases']}")
    for v in SCORE_VARS:
        t = agg["tost"][v]
        if t.get("status") == "OK":
            print(f"    {v:4s} mean_repro_rmse={t['mean_repro_rmse']:.4f} "
                  f"margin={t['margin']:.4f} tost_p={t['tost_p']:.4f} "
                  f"equiv={t['equivalent']}")
        else:
            print(f"    {v:4s} {t.get('status')} n={t.get('n')}")
    print(f"  verdict={payload['verdict']}")
    return 0 if agg["all_variables_equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
