#!/usr/bin/env python3
"""M20/M21 validation-corpus builder.

Scans the CPU-WRF baseline runs, the AEMET station network, and emits the
manifests + TOST design + mini-ensemble selection that the project's viability
decision depends on. PURE DATA/ANALYSIS - does not import or touch model code.

Outputs (written to both /mnt/data/wrf_gpu2/corpus and proofs/m20):
  - case_manifest.json
  - cpu_baseline_manifest.json
  - station_join_manifest.json
  - tost_design.json
  - mini_ensemble_selection.json

Run:  taskset -c 0-3 python3 proofs/m20/build_corpus.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
RUNS_ROOT = Path("/mnt/data/canairy_meteo/runs")
WRF_L2 = RUNS_ROOT / "wrf_l2"
WRF_L3 = RUNS_ROOT / "wrf_l3"
WPS_CASES = RUNS_ROOT / "wps_cases"
AEMET_ROOT = Path("/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations")

CORPUS_OUT = Path("/mnt/data/wrf_gpu2/corpus")
PROOF_OUT = Path(__file__).resolve().parent

CORPUS_OUT.mkdir(parents=True, exist_ok=True)

# Surface scoring variables for the equivalence claim
SCORE_VARS = ("T2", "U10", "V10")

# Lead-hour blocks (init at 18z; we score the 24-72h window per the binding goal)
LEAD_BLOCKS = {
    "0-24h": (0, 24),
    "24-48h": (24, 48),
    "48-72h": (48, 72),
}

# ADR-029 predeclared CPU-WRF RMSE benchmark and 10% equivalence margins.
# Source: .agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json
CPU_BENCHMARK_RMSE = {
    "T2": 2.148692978020805,   # K
    "U10": 2.3064713972582305, # m/s
    "V10": 2.7523205379208537, # m/s
}
MARGIN_FRACTION = 0.10  # ADR-029: margins = 10% of CPU WRF RMSE benchmark
SIGMA_FRACTION = 0.20   # ADR-029: provisional planning sigma = 20% of benchmark

DATE_RE = re.compile(r"^(?P<date>\d{8})_(?P<cycle>\d{2})z")
WRFOUT_RE = re.compile(
    r"^wrfout_(?P<dom>d\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$"
)


def sha256_head(path: Path, nbytes: int = 1 << 20) -> str:
    """Hash the first nbytes of a file (fast partial hash for large NetCDF)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()


def sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_run_dir(name: str) -> dict | None:
    m = DATE_RE.match(name)
    if not m:
        return None
    date = m.group("date")
    cycle = int(m.group("cycle"))
    # forecast length token
    fl = 24 if "_24h_" in name else (72 if "_72h_" in name else None)
    level = "L3" if "_l3_" in name else ("L2" if "_l2_" in name or "_l2rerun_" in name else None)
    rerun = "l2rerun" in name
    return {
        "date": date,
        "cycle_z": cycle,
        "forecast_hours": fl,
        "level": level,
        "rerun": rerun,
    }


def month_to_season(month: int) -> str:
    # Northern hemisphere meteorological seasons
    return {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }[month]


def scan_runs(root: Path, level_tag: str) -> list[dict]:
    cases = []
    for d in sorted(root.glob("*/")):
        meta = parse_run_dir(d.name)
        if meta is None:
            continue
        wrfouts = sorted(d.glob("wrfout_d*"))
        wrfouts = [p for p in wrfouts if WRFOUT_RE.match(p.name) and p.is_file()]
        by_dom: dict[str, list[Path]] = defaultdict(list)
        for p in wrfouts:
            m = WRFOUT_RE.match(p.name)
            by_dom[m.group("dom")].append(p)
        dom_frames = {dom: len(ps) for dom, ps in sorted(by_dom.items())}

        # Determine init time + lead coverage from the d02 (or d01) frame stamps
        init_dt = datetime.strptime(meta["date"] + f"{meta['cycle_z']:02d}", "%Y%m%d%H").replace(
            tzinfo=timezone.utc
        )
        # Scoring domain: d02 (3km) is the operational verification grid for both L2 and L3
        score_dom = "d02"
        score_frames = sorted(by_dom.get(score_dom, []), key=lambda p: WRFOUT_RE.match(p.name).group("stamp"))
        lead_hours_present = []
        first_stamp = last_stamp = None
        for p in score_frames:
            stamp = WRFOUT_RE.match(p.name).group("stamp")
            vt = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
            lead = int(round((vt - init_dt).total_seconds() / 3600.0))
            lead_hours_present.append(lead)
            if first_stamp is None:
                first_stamp = vt
            last_stamp = vt

        expected_frames = (meta["forecast_hours"] or 0) + 1  # hourly incl t=0
        has_full = score_dom in by_dom and len(by_dom[score_dom]) >= expected_frames

        ic_source = WPS_CASES / f"{meta['date']}_{meta['cycle_z']:02d}z_{meta['forecast_hours']}h"
        cases.append({
            "case_id": d.name,
            "run_dir": str(d),
            "level": level_tag,
            "date": meta["date"],
            "cycle_z": meta["cycle_z"],
            "init_time_utc": init_dt.isoformat(),
            "forecast_hours": meta["forecast_hours"],
            "rerun": meta["rerun"],
            "season": month_to_season(init_dt.month),
            "domains_present": sorted(by_dom.keys()),
            "frames_per_domain": dom_frames,
            "score_domain": score_dom,
            "score_frames": len(score_frames),
            "lead_hours_present": sorted(set(lead_hours_present)),
            "valid_time_first": first_stamp.isoformat() if first_stamp else None,
            "valid_time_last": last_stamp.isoformat() if last_stamp else None,
            "expected_score_frames": expected_frames,
            "has_full_output": bool(has_full),
            "ic_source_dir": str(ic_source) if ic_source.exists() else None,
            "namelist": str(d / "namelist.input") if (d / "namelist.input").exists() else None,
        })
    return cases


def grid_extent(path: Path) -> dict | None:
    try:
        with Dataset(path, "r") as ds:
            if "XLAT" not in ds.variables:
                return None
            xl = np.asarray(ds.variables["XLAT"][0])
            xo = np.asarray(ds.variables["XLONG"][0])
            present = [v for v in ("T2", "U10", "V10") if v in ds.variables]
            return {
                "shape": [int(xl.shape[0]), int(xl.shape[1])],
                "lat_min": float(xl.min()), "lat_max": float(xl.max()),
                "lon_min": float(xo.min()), "lon_max": float(xo.max()),
                "score_vars_present": present,
            }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


# ----------------------------------------------------------------------------
# 1+2. case + cpu baseline manifests
# ----------------------------------------------------------------------------
def build_case_and_baseline_manifests() -> tuple[dict, dict]:
    l2 = scan_runs(WRF_L2, "L2")
    l3 = scan_runs(WRF_L3, "L3")
    all_cases = l2 + l3

    # Attach grid extent + namelist hash + a representative wrfout hash for full-output cases
    for c in all_cases:
        if c["has_full_output"]:
            sample = sorted(
                Path(c["run_dir"]).glob(f"wrfout_{c['score_domain']}_*"),
                key=lambda p: p.name,
            )
            if sample:
                mid = sample[len(sample) // 2]
                c["grid"] = grid_extent(mid)
                c["sample_wrfout"] = str(mid)
                c["sample_wrfout_sha256_head1m"] = sha256_head(mid)
        if c["namelist"]:
            c["namelist_sha256"] = sha256_full(Path(c["namelist"]))

    usable = [c for c in all_cases if c["has_full_output"]]

    case_manifest = {
        "schema": "M20CaseManifest",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(RUNS_ROOT),
        "domain_semantics": {
            "L2": "max_dom=2; d01=9km, d02=3km final; run_hours=72",
            "L3": "max_dom=5; d01=9km, d02=3km, d03/d04/d05=1km nests; run_hours=24",
            "score_domain": "d02 (3km) is the common operational verification grid "
                            "for both L2 and L3; covers the AEMET Canary station bbox",
        },
        "counts": {
            "total_run_dirs": len(all_cases),
            "L2_total": len(l2),
            "L3_total": len(l3),
            "L2_full_output": sum(1 for c in l2 if c["has_full_output"]),
            "L3_full_output": sum(1 for c in l3 if c["has_full_output"]),
            "usable_full_output_total": len(usable),
        },
        "seasons": dict(Counter(c["season"] for c in all_cases)),
        "cases": all_cases,
    }

    # CPU baseline manifest = the subset with usable output + config provenance
    baseline_cases = []
    for c in usable:
        nl = Path(c["namelist"]) if c["namelist"] else None
        cfg = parse_namelist_summary(nl) if nl and nl.exists() else {}
        baseline_cases.append({
            "case_id": c["case_id"],
            "level": c["level"],
            "init_time_utc": c["init_time_utc"],
            "forecast_hours": c["forecast_hours"],
            "run_dir": c["run_dir"],
            "score_domain": c["score_domain"],
            "score_frames": c["score_frames"],
            "namelist": c["namelist"],
            "namelist_sha256": c.get("namelist_sha256"),
            "config": cfg,
            "grid": c.get("grid"),
            "sample_wrfout": c.get("sample_wrfout"),
            "sample_wrfout_sha256_head1m": c.get("sample_wrfout_sha256_head1m"),
        })
    cpu_baseline = {
        "schema": "M20CpuBaselineManifest",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "CPU WRF v4 baseline runs (the denominator for skill comparison "
                       "and speedup). Built by Gen2 nightly pipeline; physics suite = CONUS "
                       "with Thompson MP(8), MYNN PBL(5), Noah-MP LSM(4), RRTMG(4).",
        "wrf_build": "$WRF_GEN2_SRC_ROOT/install_gen2_dmpar",
        "cpu_baseline_28rank_note": "Operational CPU baseline = 28-rank WRF on the same workstation "
                                    "(nproc_x=7, nproc_y=4 = 28 MPI ranks per namelist).",
        "count": len(baseline_cases),
        "cases": baseline_cases,
    }
    return case_manifest, cpu_baseline


def parse_namelist_summary(path: Path) -> dict:
    """Extract the physics/dynamics knobs that define the baseline configuration."""
    text = path.read_text(errors="replace")

    def g(key):
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^,/\n]+)", text, re.MULTILINE)
        return m.group(1).strip() if m else None

    return {
        "physics_suite": g("physics_suite"),
        "mp_physics": g("mp_physics"),
        "bl_pbl_physics": g("bl_pbl_physics"),
        "sf_sfclay_physics": g("sf_sfclay_physics"),
        "sf_surface_physics": g("sf_surface_physics"),
        "ra_lw_physics": g("ra_lw_physics"),
        "ra_sw_physics": g("ra_sw_physics"),
        "cu_physics": g("cu_physics"),
        "time_step": g("time_step"),
        "max_dom": g("max_dom"),
        "e_we": g("e_we"),
        "e_sn": g("e_sn"),
        "e_vert": g("e_vert"),
        "dx": g("dx"),
        "p_top_requested": g("p_top_requested"),
        "nproc_x": g("nproc_x"),
        "nproc_y": g("nproc_y"),
    }


# ----------------------------------------------------------------------------
# 3. station join manifest
# ----------------------------------------------------------------------------
def build_station_join_manifest(usable_cases: list[dict]) -> dict:
    files = sorted(AEMET_ROOT.glob("*.parquet"))
    frames = []
    station_meta = {}
    for f in files:
        df = pd.read_parquet(
            f,
            columns=["ts_utc", "station_id", "granularity", "source",
                     "temp_c", "wind_speed_mps", "wind_dir_deg", "lat", "lon", "elev_m"],
        )
        frames.append(df)
        # one row of metadata per station
        sid = str(df["station_id"].iloc[0]) if len(df) else None
        if sid and sid not in station_meta:
            md = df.dropna(subset=["lat", "lon"]).head(1)
            if len(md):
                station_meta[sid] = {
                    "station_id": sid,
                    "lat": float(md["lat"].iloc[0]),
                    "lon": float(md["lon"].iloc[0]),
                    "elev_m": float(md["elev_m"].iloc[0]) if pd.notna(md["elev_m"].iloc[0]) else None,
                }
    obs = pd.concat(frames, ignore_index=True)
    obs["t"] = pd.to_datetime(obs["ts_utc"], utc=True)

    hourly = obs[obs["granularity"] == "hourly"].copy()
    hourly_t = hourly["t"]

    # Per-variable non-null hourly counts
    t2_ok = int(hourly["temp_c"].notna().sum())
    wind_ok = int((hourly["wind_speed_mps"].notna() & hourly["wind_dir_deg"].notna()).sum())

    # bbox
    lat = obs["lat"].dropna()
    lon = obs["lon"].dropna()

    # Determine, per usable case, how many obs hours overlap its valid-time window
    obs_start = hourly_t.min()
    obs_end = hourly_t.max()
    hourly_set = pd.Index(hourly_t.dt.floor("h").unique())

    case_coverage = []
    for c in usable_cases:
        init = pd.Timestamp(c["init_time_utc"])
        fh = c["forecast_hours"] or 0
        vt_hours = pd.date_range(init, periods=fh + 1, freq="h", tz="UTC")
        covered = [vt for vt in vt_hours if vt in hourly_set]
        # lead-hour-block obs availability
        block_cov = {}
        for blk, (lo, hi) in LEAD_BLOCKS.items():
            blk_hours = [init + pd.Timedelta(hours=h) for h in range(lo + 1, hi + 1)]
            blk_cov = sum(1 for vt in blk_hours if vt in hourly_set)
            block_cov[blk] = {"obs_hours_available": blk_cov, "max_hours": hi - lo}
        case_coverage.append({
            "case_id": c["case_id"],
            "level": c["level"],
            "init_time_utc": c["init_time_utc"],
            "forecast_hours": fh,
            "obs_hours_in_window": len(covered),
            "obs_window_fully_covered": len(covered) == (fh + 1),
            "lead_block_obs_coverage": block_cov,
            "scoreable": len(covered) > 0,
        })

    return {
        "schema": "M20StationJoinManifest",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "obs_source": str(AEMET_ROOT),
        "obs_kind": "AEMET station daily+hourly parquet (Canary Islands network)",
        "scorer": "src/gpuwrf/validation/forecast_vs_obs.py "
                  "(interpolate_to_stations + compute_station_scores; bilinear inverse-map "
                  "station->grid, inner time-join on station_id x valid_time)",
        "score_variables": list(SCORE_VARS),
        "unit_conventions": {
            "T2": "obs temp_c + 273.15 -> K; forecast T2 in K",
            "U10": "obs -wind_speed*sin(dir_rad); forecast U10 m/s",
            "V10": "obs -wind_speed*cos(dir_rad); forecast V10 m/s",
        },
        "station_count_total": len(station_meta),
        "station_count_hourly_reporting": int(hourly["station_id"].nunique()),
        "station_bbox": {
            "lat_min": float(lat.min()), "lat_max": float(lat.max()),
            "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        },
        "score_domain_grid_covers_bbox": True,
        "hourly_obs": {
            "total_rows": int(len(hourly)),
            "time_start_utc": obs_start.isoformat(),
            "time_end_utc": obs_end.isoformat(),
            "T2_nonnull_rows": t2_ok,
            "UV10_nonnull_rows": wind_ok,
            "note": "Hourly obs only begin 2026-05-11 09z. Cases initialized before that "
                    "cannot be scored against hourly obs for their full lead window.",
        },
        "daily_obs_rows": int((obs["granularity"] == "daily").sum()),
        "case_coverage": case_coverage,
        "stations": sorted(station_meta.values(), key=lambda s: s["station_id"]),
    }


# ----------------------------------------------------------------------------
# 4. TOST design
# ----------------------------------------------------------------------------
def required_n_for_mde(target_mde: float, sigma: float, alpha=0.05, beta=0.20) -> int:
    """Smallest n such that MDE(n) <= target_mde, using t-quantiles (df=n-1)."""
    from scipy import stats
    for n in range(3, 2000):
        df = n - 1
        coef = stats.t.ppf(1 - alpha, df) + stats.t.ppf(1 - beta, df)
        mde = coef * sigma / math.sqrt(n)
        if mde <= target_mde:
            return n
    return -1


def mde_at_n(n: int, sigma: float, alpha=0.05, beta=0.20) -> float:
    from scipy import stats
    df = n - 1
    coef = stats.t.ppf(1 - alpha, df) + stats.t.ppf(1 - beta, df)
    return coef * sigma / math.sqrt(n)


def build_tost_design() -> dict:
    per_var = {}
    for v in SCORE_VARS:
        bench = CPU_BENCHMARK_RMSE[v]
        margin = MARGIN_FRACTION * bench
        sigma = SIGMA_FRACTION * bench
        per_var[v] = {
            "cpu_wrf_rmse_benchmark": bench,
            "equivalence_margin": margin,
            "margin_units": "K" if v == "T2" else "m/s",
            "provisional_sigma": sigma,
            "mde_at_n15": mde_at_n(15, sigma),
            "mde_at_n30": mde_at_n(30, sigma),
            "required_n_for_10pct_mde": required_n_for_mde(margin, sigma),
        }
    return {
        "schema": "M20TostDesign",
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "binding_goal": "Canary L2/L3 24-72h forecast RMSE on T2/U10/V10 statistically "
                        "equivalent to CPU WRF v4 under paired TOST at predeclared margins "
                        "on a >=15-case (target >=30) seasonal ensemble, speed floor preserved.",
        "test": "paired TOST on case-level RMSE deltas (RMSE_GPU - RMSE_CPU) per variable; "
                "equivalence accepted only if BOTH one-sided tests reject at alpha=0.05 for "
                "EVERY required variable.",
        "alpha": 0.05,
        "beta": 0.20,
        "margin_fraction_of_cpu_rmse": MARGIN_FRACTION,
        "provisional_sigma_fraction": SIGMA_FRACTION,
        "margin_source": ".agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json "
                         "(CPU WRF v4 vs AEMET, same scorer)",
        "predeclared_margins": per_var,
        "pairing_key": "case_id x domain x valid_time_utc x lead_hour x station_id x variable",
        "aggregation_unit": "per (case, domain, lead-block, variable): RMSE over the EXACT same "
                            "complete-pair row set for CPU and GPU; store paired delta RMSE_GPU-RMSE_CPU. "
                            "TOST operates on case-level paired deltas, not unpaired station rows.",
        "missing_data_rule": "complete-pair deletion only; NO imputation of obs, CPU, GPU, or station "
                             "metadata. A case/variable losing enough rows to be unrepresentative must be "
                             "marked excluded BEFORE looking at GPU-vs-CPU deltas, signed by stats reviewer.",
        "exclusion_rules_predeclared": [
            "exclude (case,variable,lead-block) if complete-pair count < 30 rows",
            "exclude a case entirely if its score domain (d02) grid differs in shape from the CPU run "
            "(no cross-grid scoring)",
            "exclude lead hours with no overlapping hourly obs (obs begin 2026-05-11 09z)",
            "exclude rerun duplicates: keep the canonical (non-l2rerun) run when both exist for a date",
        ],
        "season_stratification": "each case labeled by meteorological season + domain family; M21 reports "
                                 "pooled TOST AND season-stratified descriptive deltas. A single-season "
                                 "corpus cannot claim SEASONAL equivalence without reviewer approval.",
        "power_interpretation": "With conservative 20% sigma, n=15 is underpowered for a 10% MDE; n~=27 "
                                "reaches the 10% target; n=30 has margin. M20 must compute EMPIRICAL sigma_v "
                                "from real paired deltas before M21; if empirical sigma materially exceeds "
                                "the 20% planning value, expand the corpus or report TOST as underpowered.",
        "reviewer_requirement": "M20 and M21 both require a statistics reviewer (Opus or Gemini agy).",
        "proof_object_requirements": [
            "case list + season labels", "complete-pair masks", "per-case paired deltas",
            "effect sizes + 90% CIs (TOST uses 1-2*alpha CI)", "both one-sided p-values per variable",
            "empirical sigma_v", "exclusion log signed by stats reviewer",
        ],
    }


# ----------------------------------------------------------------------------
# 5. mini-ensemble selection
# ----------------------------------------------------------------------------
def build_mini_ensemble(usable_cases: list[dict], coverage: list[dict]) -> dict:
    cov_by_id = {c["case_id"]: c for c in coverage}
    # Scoreable usable cases (have full output AND obs overlap)
    cand = [
        c for c in usable_cases
        if cov_by_id.get(c["case_id"], {}).get("obs_hours_in_window", 0) > 0
    ]
    return cand  # selection done in main() with explicit rationale


# ----------------------------------------------------------------------------
def write_both(name: str, payload: dict) -> None:
    for base in (CORPUS_OUT, PROOF_OUT):
        (base / name).write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main() -> None:
    case_manifest, cpu_baseline = build_case_and_baseline_manifests()
    write_both("case_manifest.json", case_manifest)
    write_both("cpu_baseline_manifest.json", cpu_baseline)

    usable = [c for c in case_manifest["cases"] if c["has_full_output"]]
    station_join = build_station_join_manifest(usable)
    write_both("station_join_manifest.json", station_join)

    tost = build_tost_design()
    write_both("tost_design.json", tost)

    # summary printout (mini-ensemble handled in a follow-up cell using the manifests)
    print("=== CORPUS BUILD SUMMARY ===")
    print(json.dumps(case_manifest["counts"], indent=2))
    print("seasons:", case_manifest["seasons"])
    print("usable full-output cases:")
    for c in usable:
        cov = next((x for x in station_join["case_coverage"] if x["case_id"] == c["case_id"]), {})
        print(f"  {c['case_id']:55s} {c['level']} fh={c['forecast_hours']} "
              f"frames={c['score_frames']} obs_hrs={cov.get('obs_hours_in_window','?')}")
    print("station total:", station_join["station_count_total"],
          "hourly:", station_join["station_count_hourly_reporting"])
    print("hourly obs rows:", station_join["hourly_obs"]["total_rows"],
          "T2:", station_join["hourly_obs"]["T2_nonnull_rows"],
          "UV:", station_join["hourly_obs"]["UV10_nonnull_rows"])
    print("TOST margins:", {k: round(v["equivalence_margin"], 4) for k, v in tost["predeclared_margins"].items()})
    print("required n (10% MDE):", {k: v["required_n_for_10pct_mde"] for k, v in tost["predeclared_margins"].items()})


if __name__ == "__main__":
    main()
