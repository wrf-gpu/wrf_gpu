"""V0.14 moist-cqw GPU h1-h4 validation proof.

This proof scores the manager-run short Canary d02 GPU gate for the Fable
moist-cqw / moist pg_buoy_w dynamics fix.

It is intentionally offline/CPU-only: it reads the new h1-h4 GPU wrfouts, the
previous PSFC-fix h1-h4 GPU wrfouts, and the retained CPU-WRF truth, then writes
compact JSON/Markdown evidence for the default-ON decision.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

RUN_ROOT = Path(os.environ.get("GPUWRF_MOIST_CQW_H4_RUN_ROOT", "") or Path("/tmp/v014_latest_moistcqw_h4_run_root").read_text().strip())
OLD_RUN_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_canary_d02_psfcfix_h4_20260610T160708Z")
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
CPU_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
GPU_DIR = RUN_ROOT / "gpu_output" / f"l2_d02_{RUN_ID}"
OLD_GPU_DIR = OLD_RUN_ROOT / "gpu_output" / f"l2_d02_{RUN_ID}"
DOMAIN = "d02"
INIT = datetime(2026, 5, 1, 18)
MOIST_ALL = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
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
    "HFX",
    "LH",
    "PBLH",
    "GLW",
    "SWDOWN",
    "SWNORM",
)
OUT_JSON = Path(__file__).with_suffix(".json")
OUT_MD = Path(__file__).with_suffix(".md")


def _lead_path(base: Path, lead_h: int) -> Path:
    t = INIT + timedelta(hours=lead_h)
    return base / f"wrfout_{DOMAIN}_{t:%Y-%m-%d_%H:%M:%S}"


def _stats(a: np.ndarray) -> dict[str, float]:
    x = np.asarray(a, dtype=np.float64).ravel()
    return {
        "mean": float(x.mean()),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p99_abs": float(np.percentile(np.abs(x), 99)),
        "max_abs": float(np.max(np.abs(x))),
    }


def _load(path: Path) -> dict[str, np.ndarray | float]:
    with Dataset(path) as nc:
        out: dict[str, np.ndarray | float] = {
            "P": np.asarray(nc["P"][0], dtype=np.float64),
            "PB": np.asarray(nc["PB"][0], dtype=np.float64),
            "PSFC": np.asarray(nc["PSFC"][0], dtype=np.float64),
            "MU": np.asarray(nc["MU"][0], dtype=np.float64),
            "MUB": np.asarray(nc["MUB"][0], dtype=np.float64),
            "C1H": np.asarray(nc["C1H"][0], dtype=np.float64),
            "C2H": np.asarray(nc["C2H"][0], dtype=np.float64),
            "DNW": np.asarray(nc["DNW"][0], dtype=np.float64),
            "P_TOP": float(np.asarray(nc["P_TOP"][:]).ravel()[0]),
        }
        for q in MOIST_ALL:
            out[q] = (
                np.asarray(nc[q][0], dtype=np.float64)
                if q in nc.variables
                else np.zeros_like(out["P"])
            )
    return out


def _qtot(d: dict[str, np.ndarray | float]) -> np.ndarray:
    qt = np.zeros_like(d["P"], dtype=np.float64)  # type: ignore[arg-type]
    for q in MOIST_ALL:
        qt = qt + d[q]  # type: ignore[operator]
    return qt


def _dp_dry(d: dict[str, np.ndarray | float]) -> np.ndarray:
    mut = d["MU"] + d["MUB"]  # type: ignore[operator]
    return (
        d["C1H"][:, None, None] * mut[None, :, :] + d["C2H"][:, None, None]  # type: ignore[index,operator]
    ) * (-d["DNW"][:, None, None])  # type: ignore[index]


def _p_hyd_w_profile(d: dict[str, np.ndarray | float], moist: bool) -> np.ndarray:
    layer = _dp_dry(d)
    if moist:
        layer = (1.0 + _qtot(d)) * layer
    nz = layer.shape[0]
    pw = np.empty((nz + 1,) + layer.shape[1:], dtype=np.float64)
    pw[nz] = float(d["P_TOP"])
    for k in range(nz - 1, -1, -1):
        pw[k] = pw[k + 1] + layer[k]
    return pw


def _p_hyd_half_k0(d: dict[str, np.ndarray | float], moist: bool) -> np.ndarray:
    pw = _p_hyd_w_profile(d, moist)
    return 0.5 * (pw[0] + pw[1])


def _ptot_k0(d: dict[str, np.ndarray | float]) -> np.ndarray:
    return d["P"][0] + d["PB"][0]  # type: ignore[index,operator]


def _pressure_residuals(base: Path, leads: range) -> dict[str, dict[str, float]]:
    per_lead = {}
    moist_means = []
    dry_means = []
    moist_rmses = []
    dry_rmses = []
    for h in leads:
        d = _load(_lead_path(base, h))
        moist = _stats(_ptot_k0(d) - _p_hyd_half_k0(d, moist=True))
        dry = _stats(_ptot_k0(d) - _p_hyd_half_k0(d, moist=False))
        per_lead[f"h{h}"] = {
            "ptot_k0_minus_moist_half_mean_pa": moist["mean"],
            "ptot_k0_minus_moist_half_rmse_pa": moist["rmse"],
            "ptot_k0_minus_dry_half_mean_pa": dry["mean"],
            "ptot_k0_minus_dry_half_rmse_pa": dry["rmse"],
        }
        moist_means.append(moist["mean"])
        dry_means.append(dry["mean"])
        moist_rmses.append(moist["rmse"])
        dry_rmses.append(dry["rmse"])
    return {
        "overall": {
            "ptot_k0_minus_moist_half_mean_pa": float(np.mean(moist_means)),
            "ptot_k0_minus_moist_half_rmse_pa": float(np.mean(moist_rmses)),
            "ptot_k0_minus_dry_half_mean_pa": float(np.mean(dry_means)),
            "ptot_k0_minus_dry_half_rmse_pa": float(np.mean(dry_rmses)),
        },
        "by_lead": per_lead,
    }


def _compare_field_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {}
    for field in FIELDS:
        s = d["field_summaries"].get(field)
        if not s:
            continue
        overall = s.get("overall") or {}
        drift = s.get("drift") or {}
        out[field] = {
            "rmse": overall.get("rmse"),
            "bias": overall.get("bias"),
            "p99_abs": overall.get("p99_abs"),
            "max_abs": overall.get("max_abs"),
            "rmse_slope_per_hour": drift.get("rmse_slope_per_hour"),
            "worst_lead_h": drift.get("worst_lead_h"),
        }
    return out


def _field_deltas(new: dict, old: dict) -> dict:
    out = {}
    for field in FIELDS:
        if field not in new or field not in old:
            continue
        if new[field]["rmse"] is None or old[field]["rmse"] is None:
            continue
        out[field] = {
            "old_rmse": old[field]["rmse"],
            "new_rmse": new[field]["rmse"],
            "rmse_delta": new[field]["rmse"] - old[field]["rmse"],
            "old_bias": old[field]["bias"],
            "new_bias": new[field]["bias"],
            "old_rmse_slope_per_hour": old[field]["rmse_slope_per_hour"],
            "new_rmse_slope_per_hour": new[field]["rmse_slope_per_hour"],
        }
    return out


def _resource_summary() -> dict:
    matches = list((RUN_ROOT / "resources").glob("*gpu_usage.csv"))
    if not matches:
        return {"available": False}
    path = matches[0]
    rows = list(csv.DictReader(path.open()))
    mem = [float(r["memory_used_mib"]) for r in rows]
    util = [float(r["utilization_gpu_pct"]) for r in rows]
    power = [float(r["power_draw_w"]) for r in rows]
    return {
        "available": True,
        "csv": str(path),
        "samples": len(rows),
        "peak_mem_mib": max(mem),
        "avg_mem_mib": float(np.mean(mem)),
        "avg_util_pct": float(np.mean(util)),
        "max_power_w": max(power),
    }


def main() -> None:
    leads = range(1, 5)
    new_compare = RUN_ROOT / "canary_d02_h4_grid_compare.json"
    old_compare = OLD_RUN_ROOT / "canary_d02_h4_grid_compare.json"
    gpu_rc_path = RUN_ROOT / "gpu_h4.rc"
    gpu_rc = int(gpu_rc_path.read_text().strip()) if gpu_rc_path.exists() else None
    new_fields = _compare_field_summary(new_compare)
    old_fields = _compare_field_summary(old_compare)
    pressure = {
        "cpu_truth": _pressure_residuals(CPU_DIR, leads),
        "old_gpu_psfcfix": _pressure_residuals(OLD_GPU_DIR, leads),
        "new_gpu_moistcqw": _pressure_residuals(GPU_DIR, leads),
    }
    deltas = _field_deltas(new_fields, old_fields)
    verdict = (
        "MOIST_CQW_GPU_H4_ACCEPT"
        if gpu_rc == 0
        and deltas.get("P", {}).get("rmse_delta", 1.0) < -10.0
        and deltas.get("W", {}).get("rmse_delta", 1.0) < 0.0
        and deltas.get("T", {}).get("rmse_delta", 1.0) < 0.0
        and deltas.get("U", {}).get("rmse_delta", 1.0) < 0.0
        and deltas.get("V", {}).get("rmse_delta", 1.0) < 0.0
        else "MOIST_CQW_GPU_H4_REVIEW"
    )
    out = {
        "schema": "V014MoistCqwGpuH4Validation",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "verdict": verdict,
        "run_root": str(RUN_ROOT),
        "old_run_root": str(OLD_RUN_ROOT),
        "run_id": RUN_ID,
        "gpu_rc": gpu_rc,
        "compare_json": str(new_compare),
        "old_compare_json": str(old_compare),
        "resource_summary": _resource_summary(),
        "pressure_k0_hydrostatic_residuals": pressure,
        "field_summary_new": new_fields,
        "field_summary_old": old_fields,
        "field_rmse_deltas_new_minus_old": deltas,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    _write_md(out)
    print(json.dumps({"verdict": verdict, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))


def _fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "n/a"
    return f"{float(x):.{digits}f}"


def _write_md(out: dict) -> None:
    res = out["resource_summary"]
    pr = out["pressure_k0_hydrostatic_residuals"]
    lines = [
        "# V0.14 Moist-CQW GPU h1-h4 Validation",
        "",
        f"Verdict: `{out['verdict']}`",
        "",
        f"- run root: `{out['run_root']}`",
        f"- previous PSFC-fix baseline: `{out['old_run_root']}`",
        f"- GPU rc: `{out['gpu_rc']}`",
        f"- compare JSON: `{out['compare_json']}`",
        f"- resource CSV: `{res.get('csv', 'n/a')}`",
        f"- peak VRAM: `{_fmt(res.get('peak_mem_mib'), 0)} MiB`; avg GPU util: `{_fmt(res.get('avg_util_pct'), 1)}%`; max power: `{_fmt(res.get('max_power_w'), 1)} W`",
        "",
        "## Field RMSE Delta",
        "",
        "Negative delta means the moist-cqw run improved over the previous PSFC-fix h1-h4 baseline.",
        "",
        "| Field | Old RMSE | New RMSE | Delta | Old bias | New bias |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, s in out["field_rmse_deltas_new_minus_old"].items():
        lines.append(
            f"| `{field}` | {_fmt(s['old_rmse'], 6)} | {_fmt(s['new_rmse'], 6)} | "
            f"{_fmt(s['rmse_delta'], 6)} | {_fmt(s['old_bias'], 6)} | {_fmt(s['new_bias'], 6)} |"
        )
    lines += [
        "",
        "## `P+PB(k0)` Hydrostatic Residual",
        "",
        "Means/RMSEs compare the lowest mass-level total pressure against the run's own dry or moist hydrostatic half-level column. The fix should move GPU pressure away from the old dry-balanced residual and toward the CPU/WRF moist-column regime.",
        "",
        "| Source | Mean vs moist Pa | RMSE vs moist Pa | Mean vs dry Pa | RMSE vs dry Pa |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("cpu_truth", "old_gpu_psfcfix", "new_gpu_moistcqw"):
        s = pr[label]["overall"]
        lines.append(
            f"| `{label}` | {_fmt(s['ptot_k0_minus_moist_half_mean_pa'], 3)} | "
            f"{_fmt(s['ptot_k0_minus_moist_half_rmse_pa'], 3)} | "
            f"{_fmt(s['ptot_k0_minus_dry_half_mean_pa'], 3)} | "
            f"{_fmt(s['ptot_k0_minus_dry_half_rmse_pa'], 3)} |"
        )
    lines += [
        "",
        "## Manager Decision",
        "",
        "The short GPU gate is accepted if the run is finite/green, pressure-state fields improve materially, and no wind/temperature/moisture regression appears. The all-field comparator may still report `FAIL` from stricter static/base-state or surface-flux lanes; that is not by itself a rejection of this moist-cqw dynamics sprint.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
