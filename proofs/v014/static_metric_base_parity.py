#!/usr/bin/env python3
"""V0.14 static metric/base-state parity probe.

CPU-only attribution probe for Case 3 d02. It compares CPU WRF wrfinput,
CPU WRF h0/h1 wrfouts, retained GPU h1 wrfout, the production GridSpec metrics
payload, and the loaded DycoreMetrics payload.

Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
    python proofs/v014/static_metric_base_parity.py
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DOMAIN = "d02"
INIT_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2") / RUN_ID
CPU_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
GPU_DIR = Path("/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z")
WRFINPUT = INIT_DIR / "wrfinput_d02"
CPU_H0 = CPU_DIR / "wrfout_d02_2026-05-01_18:00:00"
CPU_H1 = CPU_DIR / "wrfout_d02_2026-05-01_19:00:00"
GPU_H1 = GPU_DIR / "wrfout_d02_2026-05-01_19:00:00"
OUT_JSON = ROOT / "proofs/v014/static_metric_base_parity.json"
OUT_MD = ROOT / "proofs/v014/static_metric_base_parity.md"

HORIZONTAL_FIELDS = (
    "XLAT",
    "XLONG",
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "HGT",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "MAPFAC_MX",
    "MAPFAC_MY",
    "MAPFAC_UX",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_VY",
    "F",
    "E",
    "SINALPHA",
    "COSALPHA",
    "LANDMASK",
    "LU_INDEX",
    "RDX",
    "RDY",
)

VERTICAL_FIELDS = (
    "ZNU",
    "ZNW",
    "DN",
    "DNW",
    "RDN",
    "RDNW",
    "FNM",
    "FNP",
    "CFN",
    "CFN1",
    "CF1",
    "CF2",
    "CF3",
    "C1H",
    "C2H",
    "C3H",
    "C4H",
    "C1F",
    "C2F",
    "C3F",
    "C4F",
    "P_TOP",
)

BASE_FIELDS = ("PB", "PHB", "MUB")
REQUIRED_FIELDS = tuple(dict.fromkeys((*HORIZONTAL_FIELDS, *VERTICAL_FIELDS, *BASE_FIELDS)))

# Predeclared tolerances. Most fields are expected to be exact because the
# source files store fp32 values and the runtime loader preserves them exactly
# when materialized back to fp32. Scalar recomputations get a tiny absolute
# tolerance because they are computed in Python/JAX fp64 before fp32 output.
FIELD_TOLERANCE: dict[str, float] = {
    "ZNU": 5.0e-8,
    "ZNW": 5.0e-8,
    "RDX": 1.0e-11,
    "RDY": 1.0e-11,
    "CFN": 2.0e-7,
    "CFN1": 2.0e-7,
}

METRIC_FIELD_ATTR = {
    "MAPFAC_M": "msftx",
    "MAPFAC_U": "msfux",
    "MAPFAC_V": "msfvx",
    "MAPFAC_MX": "msftx",
    "MAPFAC_MY": "msfty",
    "MAPFAC_UX": "msfux",
    "MAPFAC_UY": "msfuy",
    "MAPFAC_VX": "msfvx",
    "MAPFAC_VY": "msfvy",
    "F": "f",
    "E": "e",
    "SINALPHA": "sina",
    "COSALPHA": "cosa",
    "DN": "dn",
    "DNW": "dnw",
    "RDN": "rdn",
    "RDNW": "rdnw",
    "FNM": "fnm",
    "FNP": "fnp",
    "CF1": "cf1",
    "CF2": "cf2",
    "CF3": "cf3",
    "C1H": "c1h",
    "C2H": "c2h",
    "C3H": "c3h",
    "C4H": "c4h",
    "C1F": "c1f",
    "C2F": "c2f",
    "C3F": "c3f",
    "C4F": "c4f",
    "P_TOP": "p_top",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_var(path: Path, name: str) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Dataset(path, "r") as ds:
        if name not in ds.variables:
            return None
        var = ds.variables[name]
        data = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
        return np.asarray(np.ma.filled(data, np.nan))


def nc_attrs(path: Path) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    with Dataset(path, "r") as ds:
        for name in ("DX", "DY", "MAP_PROJ", "CEN_LAT", "CEN_LON", "TRUELAT1", "TRUELAT2", "STAND_LON"):
            if hasattr(ds, name):
                value = getattr(ds, name)
                attrs[name] = value.item() if hasattr(value, "item") else value
    return attrs


def numeric(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.dtype.kind in {"S", "U", "O"}:
        return None
    return np.asarray(arr, dtype=np.float64)


def compare_arrays(left: np.ndarray | None, right: np.ndarray | None, *, field: str) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "status": "MISSING",
            "left_present": left is not None,
            "right_present": right is not None,
        }
    lnum = numeric(left)
    rnum = numeric(right)
    if lnum is None or rnum is None:
        return {"status": "NON_NUMERIC", "left_shape": list(left.shape), "right_shape": list(right.shape)}
    if lnum.shape != rnum.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "left_shape": list(lnum.shape),
            "right_shape": list(rnum.shape),
        }
    diff = lnum - rnum
    mask = np.isfinite(diff)
    total = int(diff.size)
    if not np.any(mask):
        return {"status": "NO_FINITE", "shape": list(diff.shape), "n": 0, "total": total}
    vals = diff[mask]
    abs_vals = np.abs(vals)
    max_abs = float(np.max(abs_vals))
    tol = float(FIELD_TOLERANCE.get(field, 0.0))
    exact = bool(np.array_equal(lnum, rnum, equal_nan=True))
    within_tol = bool(max_abs <= tol)
    status = "EXACT" if exact else ("WITHIN_TOL" if within_tol else "DIFF")
    idx = np.unravel_index(int(np.nanargmax(np.where(mask, np.abs(diff), np.nan))), diff.shape)
    return {
        "status": status,
        "shape": list(diff.shape),
        "n": int(vals.size),
        "total": total,
        "finite_fraction": float(vals.size / total) if total else None,
        "bias": float(np.mean(vals)),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
        "mae": float(np.mean(abs_vals)),
        "p95_abs": float(np.percentile(abs_vals, 95.0)),
        "max_abs": max_abs,
        "tolerance": tol,
        "max_index": [int(i) for i in idx],
        "left_at_max": float(lnum[idx]),
        "right_at_max": float(rnum[idx]),
    }


def is_pass(cmp: Mapping[str, Any]) -> bool:
    return cmp.get("status") in {"EXACT", "WITHIN_TOL"}


def is_fail(cmp: Mapping[str, Any]) -> bool:
    return cmp.get("status") not in {"EXACT", "WITHIN_TOL", "MISSING", "NOT_CHECKED"}


def max_abs(cmp: Mapping[str, Any]) -> float | None:
    value = cmp.get("max_abs")
    return None if value is None else float(value)


def read_source_fields(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray | None]:
    return {name: read_var(path, name) for name in fields}


def unstagger_x(arr: np.ndarray) -> np.ndarray:
    return 0.5 * (arr[..., :-1] + arr[..., 1:])


def unstagger_y(arr: np.ndarray) -> np.ndarray:
    return 0.5 * (arr[..., :-1, :] + arr[..., 1:, :])


def make_writer_like_state(wrfinput_fields: Mapping[str, np.ndarray | None]) -> SimpleNamespace:
    """Minimal NumPy state shaped like the production State for writer materialization."""

    p = numeric(wrfinput_fields.get("P"))
    pb = numeric(wrfinput_fields.get("PB"))
    ph = numeric(wrfinput_fields.get("PH"))
    phb = numeric(wrfinput_fields.get("PHB"))
    mu = numeric(wrfinput_fields.get("MU"))
    mub = numeric(wrfinput_fields.get("MUB"))
    t = numeric(wrfinput_fields.get("T"))
    qv = numeric(wrfinput_fields.get("QVAPOR"))
    u = numeric(wrfinput_fields.get("U"))
    v = numeric(wrfinput_fields.get("V"))
    w = numeric(wrfinput_fields.get("W"))
    landmask = numeric(wrfinput_fields.get("LANDMASK"))
    xland = numeric(wrfinput_fields.get("XLAND"))
    lu_index = numeric(wrfinput_fields.get("LU_INDEX"))

    if xland is None and landmask is not None:
        xland = np.where(landmask > 0.5, 1.0, 2.0)
    ns = SimpleNamespace()
    if u is not None:
        ns.u = u
    if v is not None:
        ns.v = v
    if w is not None:
        ns.w = w
    if t is not None:
        ns.theta = t + 300.0
    if qv is not None:
        ns.qv = qv
    if p is not None:
        ns.p_perturbation = p
    if pb is not None:
        ns.pb = pb
    if p is not None and pb is not None:
        ns.p_total = p + pb
    if ph is not None:
        ns.ph_perturbation = ph
    if phb is not None:
        ns.phb = phb
    if ph is not None and phb is not None:
        ns.ph_total = ph + phb
    if mu is not None:
        ns.mu_perturbation = mu
    if mub is not None:
        ns.mub = mub
    if mu is not None and mub is not None:
        ns.mu_total = mu + mub
    if xland is not None:
        ns.xland = xland
    if lu_index is not None:
        ns.lu_index = lu_index
    # Intentionally do not attach XLAT/XLONG/HGT. The production State does not
    # carry lat/lon arrays; HGT comes from GridSpec. This mirrors the writer path.
    return ns


def grid_metric_payload(grid: Any) -> dict[str, np.ndarray]:
    metrics = grid.metrics
    out: dict[str, np.ndarray] = {}
    eta = np.asarray(grid.eta_levels, dtype=np.float64)
    out["ZNW"] = eta
    out["ZNU"] = 0.5 * (eta[:-1] + eta[1:])
    out["HGT"] = np.asarray(grid.terrain_height, dtype=np.float64)
    out["RDX"] = np.asarray(1.0 / float(grid.projection.dx_m), dtype=np.float64)
    out["RDY"] = np.asarray(1.0 / float(grid.projection.dy_m), dtype=np.float64)
    for field, attr in METRIC_FIELD_ATTR.items():
        if field in {"P_TOP"}:
            out[field] = np.asarray(metrics.p_top, dtype=np.float64)
        elif hasattr(metrics, attr):
            out[field] = np.asarray(getattr(metrics, attr), dtype=np.float64)
    dn = np.asarray(metrics.dn, dtype=np.float64)
    dnw = np.asarray(metrics.dnw, dtype=np.float64)
    dn_top = float(dn.reshape(-1)[-1])
    dnw_top = float(dnw.reshape(-1)[-1])
    out["CFN"] = np.asarray((0.5 * dnw_top + dn_top) / dn_top, dtype=np.float64)
    out["CFN1"] = np.asarray(-0.5 * dnw_top / dn_top, dtype=np.float64)
    return out


def loaded_metric_payload(metrics: Any, grid: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    eta = np.asarray(grid.eta_levels, dtype=np.float64)
    out["ZNW"] = eta
    out["ZNU"] = 0.5 * (eta[:-1] + eta[1:])
    out["RDX"] = np.asarray(1.0 / float(grid.projection.dx_m), dtype=np.float64)
    out["RDY"] = np.asarray(1.0 / float(grid.projection.dy_m), dtype=np.float64)
    for field, attr in METRIC_FIELD_ATTR.items():
        if field == "P_TOP":
            out[field] = np.asarray(metrics.p_top, dtype=np.float64)
        elif hasattr(metrics, attr):
            out[field] = np.asarray(getattr(metrics, attr), dtype=np.float64)
    dn = np.asarray(metrics.dn, dtype=np.float64)
    dnw = np.asarray(metrics.dnw, dtype=np.float64)
    dn_top = float(dn.reshape(-1)[-1])
    dnw_top = float(dnw.reshape(-1)[-1])
    out["CFN"] = np.asarray((0.5 * dnw_top + dn_top) / dn_top, dtype=np.float64)
    out["CFN1"] = np.asarray(-0.5 * dnw_top / dn_top, dtype=np.float64)
    return out


def prepare_writer_payload(grid: Any, wrfinput_fields: Mapping[str, np.ndarray | None]) -> dict[str, np.ndarray]:
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload

    state = make_writer_like_state(wrfinput_fields)
    prepared = prepare_wrfout_payload(
        state,
        grid,
        SimpleNamespace(),
        Path("/tmp/static_metric_base_parity_synthetic_wrfout"),
        valid_time=datetime(2026, 5, 1, 18, tzinfo=timezone.utc),
        lead_hours=0.0,
        run_start=datetime(2026, 5, 1, 18, tzinfo=timezone.utc),
        diagnostics=None,
    )
    return dict(prepared.fields)


def compare_sources(
    left: Mapping[str, np.ndarray | None],
    right: Mapping[str, np.ndarray | None],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    return {name: compare_arrays(left.get(name), right.get(name), field=name) for name in fields}


def status_counts(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cmp in comparisons.values():
        status = str(cmp.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def top_diffs(comparisons: Mapping[str, Mapping[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    rows = []
    for field, cmp in comparisons.items():
        if cmp.get("status") not in {"DIFF", "WITHIN_TOL"}:
            continue
        rows.append(
            {
                "field": field,
                "status": cmp.get("status"),
                "max_abs": cmp.get("max_abs"),
                "rmse": cmp.get("rmse"),
                "bias": cmp.get("bias"),
            }
        )
    rows.sort(key=lambda row: -float(row["max_abs"] or 0.0))
    return rows[:limit]


def field_origin(field: str, comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    c = {name: group.get(field, {"status": "NOT_CHECKED"}) for name, group in comparisons.items()}
    cpu_input_vs_h0 = c["cpu_wrfinput_vs_cpu_wrfout_h0"]
    cpu_h0_vs_h1 = c["cpu_wrfout_h0_vs_cpu_wrfout_h1"]
    namelist_vs_input = c["wrfinput_vs_loaded_namelist_metrics"]
    raw_grid_vs_input = c["wrfinput_vs_raw_grid_metrics_without_attach"]
    current_grid_vs_input = c["wrfinput_vs_current_grid_metrics"]
    current_writer_vs_input = c["wrfinput_vs_current_synthetic_writer_payload"]
    current_writer_vs_gpu = c["current_synthetic_writer_payload_vs_retained_gpu_h1"]
    prefix_writer_vs_input = c["wrfinput_vs_prefix_synthetic_writer_payload"]
    prefix_writer_vs_gpu = c["prefix_synthetic_writer_payload_vs_retained_gpu_h1"]
    cpu_h1_vs_gpu = c["cpu_h1_vs_gpu_h1"]

    if is_pass(cpu_h1_vs_gpu):
        return {
            "origin": "no_mismatch",
            "confidence": "high",
            "evidence": "CPU h1 and retained GPU h1 match within the predeclared tolerance.",
        }

    if field in BASE_FIELDS and not is_pass(cpu_input_vs_h0) and is_pass(current_writer_vs_input):
        gpu_residual = max_abs(current_writer_vs_gpu) or 0.0
        cpu_gpu = max_abs(cpu_h1_vs_gpu) or 0.0
        if gpu_residual <= max(1.0e-3, 0.01 * cpu_gpu):
            return {
                "origin": "cpu_wrfinput_vs_cpu_wrfout",
                "confidence": "high" if is_pass(cpu_h0_vs_h1) else "medium",
                "evidence": (
                    "Current writer reconstruction follows wrfinput, retained GPU h1 is also "
                    "near wrfinput, and CPU wrfout differs from wrfinput"
                    + (" while staying stable from h0 to h1." if is_pass(cpu_h0_vs_h1) else ".")
                ),
                "cpu_input_vs_h0_max_abs": max_abs(cpu_input_vs_h0),
                "current_writer_vs_retained_gpu_h1_max_abs": gpu_residual,
                "cpu_h1_vs_gpu_max_abs": max_abs(cpu_h1_vs_gpu),
            }
        return {
            "origin": "cpu_wrfinput_vs_cpu_wrfout_plus_retained_gpu_h1_forecast_or_writer_reconstruction",
            "confidence": "medium",
            "evidence": (
                "CPU wrfout h0/h1 differ from wrfinput, but retained GPU h1 also differs from "
                "a zero-step writer reconstruction. With no retained GPU h0 frame, the GPU-side "
                "component cannot be split between forecast-step state drift and h1 writer "
                "base-field reconstruction."
            ),
            "cpu_input_vs_h0_max_abs": max_abs(cpu_input_vs_h0),
            "current_writer_vs_retained_gpu_h1_max_abs": max_abs(current_writer_vs_gpu),
            "cpu_h1_vs_gpu_max_abs": max_abs(cpu_h1_vs_gpu),
        }

    if not is_pass(cpu_input_vs_h0) and is_pass(current_writer_vs_input):
        return {
            "origin": "cpu_wrfinput_vs_cpu_wrfout",
            "confidence": "high" if is_pass(cpu_h0_vs_h1) else "medium",
            "evidence": (
                "GPU/writer payload follows wrfinput, while CPU wrfout differs from CPU wrfinput"
                + (" and is stable from h0 to h1." if is_pass(cpu_h0_vs_h1) else ".")
            ),
            "cpu_input_vs_h0_max_abs": max_abs(cpu_input_vs_h0),
            "cpu_h1_vs_gpu_max_abs": max_abs(cpu_h1_vs_gpu),
        }

    if is_fail(namelist_vs_input):
        return {
            "origin": "runtime_namelist_metrics",
            "confidence": "high",
            "evidence": "Loaded DycoreMetrics differ from wrfinput before any forecast step.",
            "wrfinput_vs_loaded_metrics_max_abs": max_abs(namelist_vs_input),
        }

    if (
        field in METRIC_FIELD_ATTR or field in {"CFN", "CFN1"}
    ) and is_fail(raw_grid_vs_input) and is_pass(current_grid_vs_input) and is_pass(current_writer_vs_input):
        if is_pass(prefix_writer_vs_gpu) and is_fail(current_writer_vs_gpu):
            return {
                "origin": "writer_payload_grid_metrics_prefix_fixed_current_source",
                "confidence": "high",
                "evidence": (
                    "Raw pre-fix GridSpec.metrics uses the analytic flat fallback and reproduces "
                    "the retained GPU h1 writer payload, while the patched GridSpec.metrics and "
                    "current synthetic writer payload match wrfinput. Runtime namelist metrics "
                    "also match wrfinput, so this is an emitted-static-field bug in the retained "
                    "artifact, not forecast-step dynamics."
                ),
                "wrfinput_vs_raw_grid_metrics_max_abs": max_abs(raw_grid_vs_input),
                "wrfinput_vs_current_grid_metrics_max_abs": max_abs(current_grid_vs_input),
                "current_writer_vs_retained_gpu_h1_max_abs": max_abs(current_writer_vs_gpu),
            }
        return {
            "origin": "writer_payload_grid_metrics_fixed_current_source",
            "confidence": "high",
            "evidence": (
                "Raw pre-fix GridSpec.metrics differs from wrfinput, but the patched "
                "GridSpec.metrics and current synthetic writer payload match wrfinput."
            ),
            "wrfinput_vs_raw_grid_metrics_max_abs": max_abs(raw_grid_vs_input),
        }

    if field.startswith("XLAT") or field.startswith("XLONG"):
        if is_fail(current_writer_vs_input) and is_pass(current_writer_vs_gpu):
            return {
                "origin": "writer_payload_latlon_fallback",
                "confidence": "high",
                "evidence": (
                    "XLAT/XLONG are not DycoreMetrics fields. The synthetic writer payload "
                    "differs from wrfinput and matches retained GPU h1, indicating the writer "
                    "projection fallback was emitted because the runtime State does not carry "
                    "lat/lon arrays."
                ),
                "wrfinput_vs_current_writer_max_abs": max_abs(current_writer_vs_input),
            }

    if is_fail(raw_grid_vs_input) and is_pass(prefix_writer_vs_gpu):
        return {
            "origin": "writer_payload_grid_metrics",
            "confidence": "high",
            "evidence": (
                "Loaded namelist metrics match wrfinput, but the raw pre-fix GridSpec.metrics "
                "differ and the pre-fix synthetic writer payload matches retained GPU h1."
            ),
            "wrfinput_vs_raw_grid_metrics_max_abs": max_abs(raw_grid_vs_input),
            "cpu_h1_vs_gpu_max_abs": max_abs(cpu_h1_vs_gpu),
        }

    if field in BASE_FIELDS and is_pass(current_writer_vs_input) and is_fail(current_writer_vs_gpu):
        return {
            "origin": "forecast_step_or_h1_writer_reconstruction",
            "confidence": "medium",
            "evidence": (
                "Input and zero-step writer reconstruction match wrfinput, but retained h1 GPU "
                "does not. The writer reconstructs PB/PHB/MUB from total-minus-perturbation "
                "state at output time, so this is not a true base input mismatch. A retained "
                "GPU h0 frame would be needed to split post-step state drift from h1 writer "
                "reconstruction."
            ),
            "current_writer_vs_retained_gpu_h1_max_abs": max_abs(current_writer_vs_gpu),
        }

    if is_fail(current_writer_vs_input) and is_pass(current_writer_vs_gpu):
        return {
            "origin": "writer_payload",
            "confidence": "high",
            "evidence": "Synthetic writer payload differs from wrfinput and matches retained GPU h1.",
            "wrfinput_vs_writer_max_abs": max_abs(current_writer_vs_input),
        }

    return {
        "origin": "narrowed_unresolved",
        "confidence": "low",
        "evidence": "Available retained CPU-only artifacts do not isolate one layer.",
        "cpu_h1_vs_gpu_max_abs": max_abs(cpu_h1_vs_gpu),
        "wrfinput_vs_current_writer_max_abs": max_abs(current_writer_vs_input),
        "current_writer_vs_retained_gpu_h1_max_abs": max_abs(current_writer_vs_gpu),
    }


def write_markdown(payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    key = payload["key_field_origins"]
    origins = payload["field_origins"]
    lines: list[str] = []
    lines.append("# V0.14 Static Metric/Base-State Parity")
    lines.append("")
    lines.append(f"Generated UTC: `{payload['generated_utc']}`")
    lines.append("")
    lines.append("CPU-only probe over Case 3 d02 retained artifacts. No GPU run was launched.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "- Source touched: yes. `src/gpuwrf/integration/d02_replay.py` now attaches the loaded WRF `DycoreMetrics` to `GridSpec.metrics` before the case is handed to runtime/writer code."
    )
    lines.append(
        "- Pre-fix vertical C/DN/RDN and MAPFAC mismatches are writer-only: raw `GridSpec.metrics` used the analytic flat fallback, while `case.metrics`/namelist metrics loaded WRF values. The current patched synthetic writer payload matches wrfinput for those fields; the retained GPU h1 artifact still shows the old writer payload."
    )
    lines.append(
        "- `XLAT/XLONG` remain a writer payload fallback issue, separate from metric plumbing: the runtime State lacks lat/lon arrays, so the writer emits projection-derived coordinates that match the retained GPU h1 artifact."
    )
    lines.append(
        "- `HGT` mismatch is CPU wrfinput-vs-CPU-wrfout terrain convention: retained GPU/current writer follows wrfinput, while CPU wrfout h0/h1 differ from wrfinput."
    )
    lines.append(
        "- `PB/PHB/MUB` are not caused by flat metrics. `PHB` is dominated by CPU wrfinput-vs-CPU-wrfout convention; `PB/MUB` also have a retained-GPU-h1 component that cannot be split between forecast-step state drift and h1 writer base-field reconstruction without a retained GPU h0."
    )
    lines.append("")
    lines.append("## Layer Counts")
    lines.append("")
    for name, counts in summary["comparison_status_counts"].items():
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"- `{name}`: {parts}")
    lines.append("")
    lines.append("## Key Field Origins")
    lines.append("")
    for field in ("C2H", "C2F", "C4H", "C4F", "RDN", "HGT", "XLAT", "XLONG", "MAPFAC_M", "PB", "PHB", "MUB"):
        item = key.get(field, {})
        lines.append(
            f"- `{field}`: `{item.get('origin')}` ({item.get('confidence')}); {item.get('evidence')}"
        )
    lines.append("")
    lines.append("## Worst CPU h1 vs GPU h1 Differences")
    lines.append("")
    lines.append("| field | max abs | RMSE | bias | origin |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for row in summary["top_cpu_h1_vs_gpu_h1_diffs"]:
        origin = key.get(row["field"], origins.get(row["field"], {})).get("origin", "")
        lines.append(
            f"| `{row['field']}` | {float(row['max_abs']):.6g} | {float(row['rmse']):.6g} | {float(row['bias']):.6g} | `{origin}` |"
        )
    lines.append("")
    lines.append("## Runtime/Writer Split")
    lines.append("")
    lines.append("- Pre-fix witness: `Gen2Run.grid(...).as_grid_spec()` constructs `GridSpec` before loaded metrics are available; `GridSpec.__post_init__` fills missing metrics with `DycoreMetrics.flat`.")
    lines.append("- Current source path: `build_replay_case` loads `DycoreMetrics` from the same metrics source and replaces `grid.metrics` with that payload before `State.zeros` and before writer bundles are built.")
    lines.append("- Runtime dynamics: nested pipeline still passes `metrics=case.metrics` into `OperationalNamelist.from_grid`; this proof found no evidence that dynamics consumed flat vertical metrics.")
    lines.append("- The nested writer calls `write_wrfout_netcdf(state, grid, namelist, ...)`, and `_add_grid_coordinate_fields` reads `grid.metrics`, not `namelist.metrics`.")
    lines.append("- The retained GPU h1 wrfout predates this local source patch, so it is used as a stale-artifact witness, not as a post-fix output claim.")
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    for limit in payload["limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    missing = [str(path) for path in (WRFINPUT, CPU_H0, CPU_H1, GPU_H1) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required artifact(s) missing: {missing}")

    from gpuwrf.dynamics.metrics import load_wrfinput_metrics
    from gpuwrf.io.gen2_accessor import Gen2Run

    run = Gen2Run(INIT_DIR)
    raw_grid = run.grid(DOMAIN).as_grid_spec()
    metrics_source = Path(run.history_files(DOMAIN)[0])
    loaded_metrics = load_wrfinput_metrics(metrics_source)
    grid = dataclass_replace(raw_grid, metrics=loaded_metrics)

    wrfinput_fields = read_source_fields(WRFINPUT, REQUIRED_FIELDS + ("U", "V", "W", "T", "P", "PH", "MU", "QVAPOR", "XLAND"))
    cpu_h0_fields = read_source_fields(CPU_H0, REQUIRED_FIELDS)
    cpu_h1_fields = read_source_fields(CPU_H1, REQUIRED_FIELDS)
    gpu_h1_fields = read_source_fields(GPU_H1, REQUIRED_FIELDS)

    raw_grid_metric_fields = grid_metric_payload(raw_grid)
    current_grid_metric_fields = grid_metric_payload(grid)
    loaded_metric_fields = loaded_metric_payload(loaded_metrics, grid)
    prefix_synthetic_writer_fields = prepare_writer_payload(raw_grid, wrfinput_fields)
    current_synthetic_writer_fields = prepare_writer_payload(grid, wrfinput_fields)

    # Add source-level HGT and base-state fields that are not part of DycoreMetrics.
    raw_grid_payload_for_compare: dict[str, np.ndarray | None] = dict(raw_grid_metric_fields)
    raw_grid_payload_for_compare["HGT"] = np.asarray(raw_grid.terrain_height, dtype=np.float64)
    current_grid_payload_for_compare: dict[str, np.ndarray | None] = dict(current_grid_metric_fields)
    current_grid_payload_for_compare["HGT"] = np.asarray(grid.terrain_height, dtype=np.float64)
    loaded_payload_for_compare: dict[str, np.ndarray | None] = dict(loaded_metric_fields)
    loaded_payload_for_compare["HGT"] = np.asarray(grid.terrain_height, dtype=np.float64)

    comparisons = {
        "cpu_wrfinput_vs_cpu_wrfout_h0": compare_sources(wrfinput_fields, cpu_h0_fields, REQUIRED_FIELDS),
        "cpu_wrfout_h0_vs_cpu_wrfout_h1": compare_sources(cpu_h0_fields, cpu_h1_fields, REQUIRED_FIELDS),
        "cpu_h1_vs_gpu_h1": compare_sources(cpu_h1_fields, gpu_h1_fields, REQUIRED_FIELDS),
        "wrfinput_vs_loaded_namelist_metrics": compare_sources(wrfinput_fields, loaded_payload_for_compare, REQUIRED_FIELDS),
        "wrfinput_vs_raw_grid_metrics_without_attach": compare_sources(wrfinput_fields, raw_grid_payload_for_compare, REQUIRED_FIELDS),
        "wrfinput_vs_current_grid_metrics": compare_sources(wrfinput_fields, current_grid_payload_for_compare, REQUIRED_FIELDS),
        "wrfinput_vs_prefix_synthetic_writer_payload": compare_sources(wrfinput_fields, prefix_synthetic_writer_fields, REQUIRED_FIELDS),
        "prefix_synthetic_writer_payload_vs_retained_gpu_h1": compare_sources(prefix_synthetic_writer_fields, gpu_h1_fields, REQUIRED_FIELDS),
        "wrfinput_vs_current_synthetic_writer_payload": compare_sources(wrfinput_fields, current_synthetic_writer_fields, REQUIRED_FIELDS),
        "current_synthetic_writer_payload_vs_retained_gpu_h1": compare_sources(current_synthetic_writer_fields, gpu_h1_fields, REQUIRED_FIELDS),
    }

    key_fields = tuple(dict.fromkeys(("C2H", "C2F", "C4H", "C4F", "RDN", "DN", "HGT", "XLAT", "XLONG", "MAPFAC_M", "MAPFAC_MX", "PB", "PHB", "MUB")))
    origins = {field: field_origin(field, comparisons) for field in REQUIRED_FIELDS}
    key_origins = {field: origins[field] for field in key_fields if field in origins}

    c_examples = {}
    for field in ("C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F", "DN", "RDN", "ZNU", "ZNW"):
        c_examples[field] = {
            "wrfinput_first5": np.asarray(wrfinput_fields.get(field), dtype=np.float64).reshape(-1)[:5].tolist()
            if wrfinput_fields.get(field) is not None
            else None,
            "raw_grid_metrics_first5": np.asarray(raw_grid_payload_for_compare.get(field), dtype=np.float64).reshape(-1)[:5].tolist()
            if raw_grid_payload_for_compare.get(field) is not None
            else None,
            "current_grid_metrics_first5": np.asarray(current_grid_payload_for_compare.get(field), dtype=np.float64).reshape(-1)[:5].tolist()
            if current_grid_payload_for_compare.get(field) is not None
            else None,
            "loaded_metrics_first5": np.asarray(loaded_payload_for_compare.get(field), dtype=np.float64).reshape(-1)[:5].tolist()
            if loaded_payload_for_compare.get(field) is not None
            else None,
            "gpu_h1_first5": np.asarray(gpu_h1_fields.get(field), dtype=np.float64).reshape(-1)[:5].tolist()
            if gpu_h1_fields.get(field) is not None
            else None,
        }

    payload: dict[str, Any] = {
        "schema": "v014_static_metric_base_parity",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only": True,
            "gpu_used": False,
        },
        "inputs": {
            "run_id": RUN_ID,
            "domain": DOMAIN,
            "wrfinput": str(WRFINPUT),
            "cpu_h0": str(CPU_H0),
            "cpu_h1": str(CPU_H1),
            "gpu_h1": str(GPU_H1),
            "init_dir": str(INIT_DIR),
            "cpu_dir": str(CPU_DIR),
            "gpu_dir": str(GPU_DIR),
        },
        "source_attrs": {
            "wrfinput": nc_attrs(WRFINPUT),
            "cpu_h0": nc_attrs(CPU_H0),
            "gpu_h1": nc_attrs(GPU_H1),
        },
        "runtime_provenance": {
            "grid_source_wrfout": run.grid(DOMAIN).source_wrfout,
            "metrics_source": str(metrics_source),
            "raw_grid_metrics_provenance": str(raw_grid.metrics.provenance),
            "current_grid_metrics_provenance": str(grid.metrics.provenance),
            "loaded_metrics_provenance": str(loaded_metrics.provenance),
            "grid_shape": {"nz": int(grid.nz), "ny": int(grid.ny), "nx": int(grid.nx)},
            "code_path_evidence": [
                "Pre-fix witness: run.grid(domain).as_grid_spec() has raw_grid.metrics.provenance == analytic-flat.",
                "Patched d02_replay.build_replay_case loads metrics_source = run.history_files(domain)[0], then dataclass_replace(grid, metrics=metrics).",
                "nested_pipeline._make_namelist passes metrics=case.metrics into OperationalNamelist.from_grid.",
                "nested_pipeline._PerDomainWrfoutWriter passes grid, not namelist.metrics, to write_wrfout_netcdf.",
                "wrfout_writer._add_grid_coordinate_fields reads metrics = grid.metrics.",
            ],
        },
        "source_change": {
            "source_touched": True,
            "file": "src/gpuwrf/integration/d02_replay.py",
            "diff_rationale": (
                "Attach load_wrfinput_metrics(metrics_source) to GridSpec.metrics with dataclasses.replace "
                "immediately after GridSpec creation, preserving case.metrics / namelist.metrics behavior "
                "while fixing shared grid metadata consumed by the wrfout writer."
            ),
            "read_only_preserved": [
                "src/gpuwrf/contracts/grid.py",
                "runtime dycore",
                "src/gpuwrf/io/wrfout_writer.py",
                "radiation",
            ],
        },
        "comparisons": comparisons,
        "field_origins": origins,
        "key_field_origins": key_origins,
        "vertical_examples": c_examples,
        "summary": {
            "comparison_status_counts": {name: status_counts(group) for name, group in comparisons.items()},
            "top_cpu_h1_vs_gpu_h1_diffs": top_diffs(comparisons["cpu_h1_vs_gpu_h1"], limit=16),
            "top_wrfinput_vs_raw_grid_metrics_diffs": top_diffs(comparisons["wrfinput_vs_raw_grid_metrics_without_attach"], limit=16),
            "top_wrfinput_vs_current_grid_metrics_diffs": top_diffs(comparisons["wrfinput_vs_current_grid_metrics"], limit=16),
            "top_wrfinput_vs_loaded_namelist_metrics_diffs": top_diffs(comparisons["wrfinput_vs_loaded_namelist_metrics"], limit=16),
            "top_prefix_writer_vs_retained_gpu_h1_diffs": top_diffs(comparisons["prefix_synthetic_writer_payload_vs_retained_gpu_h1"], limit=16),
            "top_current_writer_vs_retained_gpu_h1_diffs": top_diffs(comparisons["current_synthetic_writer_payload_vs_retained_gpu_h1"], limit=16),
            "conclusion": {
                "vertical_metric_mismatch": "pre_fix_writer_payload_grid_metrics; fixed in current synthetic writer payload",
                "runtime_consumed_vertical_metrics": "loaded_namelist_metrics_match_wrfinput",
                "mapfac_mismatch": "pre_fix_writer_payload_grid_metrics; fixed in current synthetic writer payload",
                "xlat_xlong_mismatch": "writer_payload_latlon_fallback_still_present",
                "hgt_mismatch": "cpu_wrfinput_vs_cpu_wrfout_convention",
                "base_state_mismatch": "PHB dominated by CPU output convention; PB/MUB also include retained h1 forecast_or_writer_reconstruction residual",
            },
        },
        "limits": [
            "No retained GPU h0 wrfout exists, so PB/PHB/MUB h1 differences cannot be split into post-step state drift versus h1 writer reconstruction with only retained artifacts.",
            "build_replay_case itself cannot run under JAX_PLATFORMS=cpu because State.zeros requires a visible GPU; this proof reconstructs the relevant GridSpec and loaded DycoreMetrics CPU-only instead.",
            "No fresh GPU writer smoke was launched. Retained GPU h1 predates the source patch and cannot demonstrate grid-cell-envelope improvement without a new GPU/writer artifact.",
        ],
    }

    write_json(OUT_JSON, payload)
    write_markdown(payload)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
