#!/usr/bin/env python3
"""CPU-only V10 grid-difference diagnostics for powered TOST cases.

This is an attribution aid, not an equivalence gate. It separates the station
TOST result from direct GPU-vs-CPU grid-field divergence and summarizes where
V10 errors are largest over the retained d02 wrfouts.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=src \
      python proofs/v014/v10_grid_diagnostics.py
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "proofs/v0120/powered_tost_n15"
OUT_JSON = ROOT / "proofs/v014/v10_grid_diagnostics.json"
OUT_MD = ROOT / "proofs/v014/v10_grid_diagnostics.md"
CPU_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output")
GPU_ROOT = Path("/tmp/v0120_powered_tost_runs")
WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")
FIELDS = ("T2", "U10", "V10", "PSFC")
TOST_MARGINS = {"T2": 0.215, "U10": 0.231, "V10": 0.275}


def parse_init_time(run_id: str) -> datetime:
    parts = run_id.split("_")
    ds = parts[0]
    hour = int(parts[1].replace("z", ""))
    return datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]), hour, tzinfo=timezone.utc)


def wrfout_map(path: Path, domain: str = "d02") -> dict[datetime, Path]:
    out = {}
    for p in sorted(path.glob(f"wrfout_{domain}_*")):
        m = WRFOUT_RE.match(p.name)
        if not m or not p.is_file():
            continue
        vt = datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
        out[vt] = p
    return out


def read2(path: Path, name: str) -> np.ndarray | None:
    with Dataset(path) as ds:
        if name not in ds.variables:
            return None
        v = ds.variables[name]
        arr = v[0] if v.dimensions and v.dimensions[0] == "Time" else v[:]
    return np.asarray(np.ma.filled(arr, np.nan), dtype=np.float64)


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "status": "NO_DATA"}
    abs_values = np.abs(values)
    return {
        "n": int(values.size),
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mae": float(np.mean(abs_values)),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "p99_abs": float(np.percentile(abs_values, 99)),
        "max_abs": float(np.max(abs_values)),
        "frac_abs_le_2p5": float(np.mean(abs_values <= 2.5)),
    }


def corr(a: list[float], b: list[float]) -> float | None:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(m)) < 2:
        return None
    aa = aa[m]
    bb = bb[m]
    if float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def load_case_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize_json_only(case: dict) -> dict:
    fields = case.get("cell_level", {}).get("field_stats", {})
    return {
        "run_id": case.get("run_id"),
        "source": "case_json_only",
        "reason_spatial_unavailable": "GPU wrfout directory is not available; using stored aggregate stats only.",
        "cell_level": {
            f: {
                "rmse": fields.get(f, {}).get("rmse"),
                "bias": fields.get(f, {}).get("bias"),
                "p95_abs": fields.get(f, {}).get("p95"),
                "p99_abs": fields.get(f, {}).get("p99"),
                "max_abs": fields.get(f, {}).get("max"),
                "frac_within_tol": fields.get(f, {}).get("frac_within_tol"),
                "pearson_r": fields.get(f, {}).get("pearson_r"),
            }
            for f in ("T2", "U10", "V10")
        },
        "station_tost_0_24h": station_tost_0_24h(case),
    }


def station_tost_0_24h(case: dict) -> dict:
    tost = case.get("tost_pairs", {}).get("per_block", {})
    return {
        f: {
            **tost.get(f, {}).get("0-24h", {}),
            "equivalence_margin": TOST_MARGINS[f],
        }
        for f in ("T2", "U10", "V10")
    }


def analyze_spatial(run_id: str, gpu_dir: Path, cpu_dir: Path) -> dict:
    init = parse_init_time(run_id)
    gm = wrfout_map(gpu_dir)
    cm = wrfout_map(cpu_dir)
    common = sorted(t for t in set(gm) & set(cm) if 0 < (t - init).total_seconds() / 3600 <= 24)
    if not common:
        return {
            "run_id": run_id,
            "source": "spatial_attempt",
            "status": "NO_COMMON_WRFOUTS",
            "gpu_dir": str(gpu_dir),
            "cpu_dir": str(cpu_dir),
        }

    first_cpu = cm[common[0]]
    hgt = read2(first_cpu, "HGT")
    land = read2(first_cpu, "LANDMASK")
    lat = read2(first_cpu, "XLAT")
    lon = read2(first_cpu, "XLONG")
    if hgt is None:
        hgt = np.zeros_like(read2(first_cpu, "V10"))
    if land is None:
        land = np.zeros_like(hgt)
    if lat is None:
        lat = np.zeros_like(hgt)
    if lon is None:
        lon = np.zeros_like(hgt)

    elev_labels = np.full(hgt.shape, "ocean", dtype=object)
    land_mask = land > 0.5
    elev_labels[land_mask & (hgt < 300.0)] = "land_0_300m"
    elev_labels[land_mask & (hgt >= 300.0) & (hgt < 1000.0)] = "land_300_1000m"
    elev_labels[land_mask & (hgt >= 1000.0)] = "land_gt_1000m"
    lat_mid = float(np.nanmedian(lat))
    lon_mid = float(np.nanmedian(lon))
    quad = np.full(hgt.shape, "SW", dtype=object)
    quad[(lat >= lat_mid) & (lon < lon_mid)] = "NW"
    quad[(lat >= lat_mid) & (lon >= lon_mid)] = "NE"
    quad[(lat < lat_mid) & (lon >= lon_mid)] = "SE"

    all_d = {f: [] for f in FIELDS}
    by_lead = []
    v10_by_block = {"0-6h": [], "6-12h": [], "12-24h": []}
    v10_by_elev = {}
    v10_by_quad = {}

    for t in common:
        lead_h = int(round((t - init).total_seconds() / 3600.0))
        diffs = {}
        for f in FIELDS:
            g = read2(gm[t], f)
            c = read2(cm[t], f)
            if g is None or c is None:
                continue
            d = g - c
            diffs[f] = d
            all_d[f].extend(d[np.isfinite(d)].ravel().tolist())

        v10 = diffs.get("V10")
        if v10 is None:
            continue
        if lead_h <= 6:
            v10_by_block["0-6h"].extend(v10[np.isfinite(v10)].ravel().tolist())
        elif lead_h <= 12:
            v10_by_block["6-12h"].extend(v10[np.isfinite(v10)].ravel().tolist())
        else:
            v10_by_block["12-24h"].extend(v10[np.isfinite(v10)].ravel().tolist())
        by_lead.append({
            "lead_h": lead_h,
            "V10": stats(v10),
            "U10": stats(diffs["U10"]) if "U10" in diffs else {"n": 0},
            "T2": stats(diffs["T2"]) if "T2" in diffs else {"n": 0},
            "PSFC": stats(diffs["PSFC"]) if "PSFC" in diffs else {"n": 0},
        })
        for label in sorted(set(elev_labels.ravel().tolist())):
            mask = elev_labels == label
            v10_by_elev.setdefault(label, []).extend(v10[mask & np.isfinite(v10)].ravel().tolist())
        for label in sorted(set(quad.ravel().tolist())):
            mask = quad == label
            v10_by_quad.setdefault(label, []).extend(v10[mask & np.isfinite(v10)].ravel().tolist())

    corr_inputs = {f: all_d[f] for f in ("T2", "U10", "PSFC") if all_d[f]}
    return {
        "run_id": run_id,
        "source": "spatial_grid_wrfouts",
        "gpu_dir": str(gpu_dir),
        "cpu_dir": str(cpu_dir),
        "n_common_leads": len(common),
        "field_stats": {f: stats(np.asarray(v)) for f, v in all_d.items() if v},
        "V10_by_lead": by_lead,
        "V10_by_block": {k: stats(np.asarray(v)) for k, v in v10_by_block.items()},
        "V10_by_elevation": {k: stats(np.asarray(v)) for k, v in v10_by_elev.items()},
        "V10_by_quadrant": {k: stats(np.asarray(v)) for k, v in v10_by_quad.items()},
        "V10_error_correlations": {
            f"corr_dV10_d{f}": corr(all_d["V10"], vals)
            for f, vals in corr_inputs.items()
        },
        "worst_V10_leads_by_rmse": sorted(
            (x for x in by_lead if x["V10"].get("n", 0) > 0),
            key=lambda x: x["V10"].get("rmse", -1.0),
            reverse=True,
        )[:5],
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# V10 Grid Diagnostics",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "This is a diagnosis artifact, not an equivalence pass.",
        "",
        "## Cases",
        "",
    ]
    v10_summary = report.get("summary", {}).get("V10", {})
    if v10_summary:
        lines.extend(
            [
                "## V10 Summary",
                "",
                f"- grid cases with V10 RMSE > 1.5 m/s: `{v10_summary.get('grid_rmse_gt_1p5')}` / `{v10_summary.get('grid_n_cases')}`",
                f"- grid V10 bias signs: `{', '.join(v10_summary.get('grid_bias_signs', []))}`",
                f"- station TOST V10 outside ADR-029 margin: `{v10_summary.get('station_outside_margin')}` / `{v10_summary.get('station_n_cases')}`",
                "",
            ]
        )
    for case in report["cases"]:
        lines.append(f"### {case['run_id']}")
        lines.append("")
        lines.append(f"- source: `{case['source']}`")
        if case["source"] == "spatial_grid_wrfouts":
            fs = case["field_stats"].get("V10", {})
            tost = case.get("station_tost_0_24h", {}).get("V10", {})
            lines.append(
                f"- V10 grid RMSE: `{fs.get('rmse'):.3f}` m/s; "
                f"bias: `{fs.get('bias'):.3f}` m/s; "
                f"p95 abs: `{fs.get('p95_abs'):.3f}` m/s"
            )
            lines.append(
                f"- station V10 paired delta RMSE: `{tost.get('paired_delta_rmse')}` "
                f"(margin `{tost.get('equivalence_margin')}`)"
            )
            lines.append("- V10 by block:")
            for block, st in case["V10_by_block"].items():
                lines.append(
                    f"  - `{block}`: RMSE `{st.get('rmse'):.3f}`, "
                    f"bias `{st.get('bias'):.3f}`, p95 `{st.get('p95_abs'):.3f}`"
                )
            lines.append("- V10 correlations:")
            for key, value in case["V10_error_correlations"].items():
                lines.append(f"  - `{key}`: `{value}`")
        else:
            v10 = case["cell_level"].get("V10", {})
            tost = case["station_tost_0_24h"].get("V10", {})
            lines.append(
                f"- stored V10 grid RMSE: `{v10.get('rmse')}`; "
                f"bias: `{v10.get('bias')}`; p95 abs: `{v10.get('p95_abs')}`"
            )
            lines.append(
                f"- station V10 paired delta RMSE: `{tost.get('paired_delta_rmse')}` "
                f"(margin `{tost.get('equivalence_margin')}`)"
            )
            lines.append(f"- note: {case.get('reason_spatial_unavailable')}")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n")


def summarize_v10(cases: list[dict]) -> dict:
    grid = []
    station = []
    for case in cases:
        if case.get("source") == "spatial_grid_wrfouts":
            v10 = case.get("field_stats", {}).get("V10", {})
            grid.append(v10)
        elif case.get("source") == "case_json_only":
            v10 = case.get("cell_level", {}).get("V10", {})
            grid.append(v10)
        st = case.get("station_tost_0_24h", {}).get("V10")
        if st is not None:
            station.append(st)

    def sign(x: float | None) -> str:
        if x is None:
            return "missing"
        if x > 0:
            return "+"
        if x < 0:
            return "-"
        return "0"

    return {
        "grid_n_cases": len(grid),
        "grid_rmse_gt_1p5": sum(1 for x in grid if (x.get("rmse") or 0.0) > 1.5),
        "grid_bias_signs": [sign(x.get("bias")) for x in grid],
        "station_n_cases": len(station),
        "station_outside_margin": sum(
            1
            for x in station
            if abs(x.get("paired_delta_rmse") or 0.0) > (x.get("equivalence_margin") or float("inf"))
        ),
        "station_delta_signs": [sign(x.get("paired_delta_rmse")) for x in station],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args(argv)

    cases = []
    for case_path in sorted(CASE_DIR.glob("case_*.json")):
        case = load_case_json(case_path)
        run_id = case["run_id"]
        gpu_dir = Path(case.get("gpu_dir", "")) if case.get("gpu_dir") else GPU_ROOT / f"l2_d02_{run_id}"
        cpu_dir = Path(case.get("cpu_dir", "")) if case.get("cpu_dir") else CPU_ROOT / run_id
        if gpu_dir.is_dir() and cpu_dir.is_dir():
            item = analyze_spatial(run_id, gpu_dir, cpu_dir)
            item["station_tost_0_24h"] = station_tost_0_24h(case)
            cases.append(item)
        else:
            cases.append(summarize_json_only(case))

    report = {
        "schema": "v014-v10-grid-diagnostics",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_dir": str(CASE_DIR),
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_spatial_cases": sum(1 for c in cases if c.get("source") == "spatial_grid_wrfouts"),
            "n_json_only_cases": sum(1 for c in cases if c.get("source") == "case_json_only"),
        },
    }
    report["summary"]["V10"] = summarize_v10(cases)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(report, args.out_md)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
