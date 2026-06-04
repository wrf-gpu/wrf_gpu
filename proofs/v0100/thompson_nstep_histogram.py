"""Thompson sedimentation nstep histogram over WRF/GPU wrfout files.

This probe is CPU-safe: it reads wrfout NetCDF fields and reuses the current
Thompson fall-speed and nstep formulas without running a forecast. It does not
edit kernel code and it does not need a CUDA device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.coupling.physics_couplers import _temperature_from_theta
from gpuwrf.physics import thompson_column as tc

GRAVITY = 9.81
R1 = float(tc.R1)


def _var(ds: Dataset, name: str, default: np.ndarray | None = None) -> np.ndarray:
    if name in ds.variables:
        return np.asarray(ds.variables[name][0])
    if default is None:
        raise KeyError(name)
    return np.asarray(default)


def _cols(arr: np.ndarray) -> jax.Array:
    return jnp.moveaxis(jnp.asarray(arr), 0, -1)


def _state_from_wrfout(path: Path) -> tc.ThompsonColumnState:
    with Dataset(path, "r") as ds:
        theta = _var(ds, "T") + 300.0
        p = _var(ds, "P") + _var(ds, "PB")
        qv = _var(ds, "QVAPOR", np.zeros_like(theta))
        qc = _var(ds, "QCLOUD", np.zeros_like(theta))
        qr = _var(ds, "QRAIN", np.zeros_like(theta))
        qi = _var(ds, "QICE", np.zeros_like(theta))
        qs = _var(ds, "QSNOW", np.zeros_like(theta))
        qg = _var(ds, "QGRAUP", np.zeros_like(theta))
        ni = _var(ds, "QNICE", np.zeros_like(theta))
        nr = _var(ds, "QNRAIN", np.zeros_like(theta))
        ns = _var(ds, "QNSNOW", np.zeros_like(theta))
        ng = _var(ds, "QNGRAUPEL", np.zeros_like(theta))
        ph = _var(ds, "PH") + _var(ds, "PHB")
        dz = np.maximum((ph[1:] - ph[:-1]) / GRAVITY, 1.0)
        if "W" in ds.variables:
            w_stag = _var(ds, "W")
            w = 0.5 * (w_stag[:-1] + w_stag[1:])
        else:
            w = np.zeros_like(theta)
    t_abs = _temperature_from_theta(jnp.asarray(theta), jnp.asarray(p))
    rho = tc.density_from_pressure_temperature(jnp.asarray(p), t_abs, jnp.asarray(qv))
    return tc.ThompsonColumnState(
        _cols(qv),
        _cols(qc),
        _cols(qr),
        _cols(qi),
        _cols(qs),
        _cols(qg),
        _cols(ni),
        _cols(nr),
        _cols(np.asarray(t_abs)),
        _cols(p),
        _cols(np.asarray(rho)),
        Ns=_cols(ns),
        Ng=_cols(ng),
        dz=_cols(dz),
        w=_cols(w),
    )


def _percentile(values: np.ndarray, pct: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values.astype(np.float64), pct))


def _summarize_nstep(nstep: np.ndarray, wet: np.ndarray, caps: tuple[int, ...]) -> tuple[dict[str, Any], np.ndarray]:
    wet_vals = nstep[wet]
    all_vals = nstep.reshape(-1)
    summary = {
        "columns": int(all_vals.size),
        "wet_columns": int(wet_vals.size),
        "wet_fraction": float(wet_vals.size / max(all_vals.size, 1)),
        "max": None if wet_vals.size == 0 else float(np.max(wet_vals)),
        "p99": _percentile(wet_vals, 99.0),
        "p99_9": _percentile(wet_vals, 99.9),
        "clip_count_by_cap": {str(cap): int(np.count_nonzero(wet_vals > float(cap))) for cap in caps},
        "all_column_max": float(np.max(all_vals)) if all_vals.size else None,
    }
    return summary, wet_vals.astype(np.float64, copy=False)


def _one_file(path: Path, dt_s: float, caps: tuple[int, ...]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = _state_from_wrfout(path)
    vt_r_mass, vt_r_num, vt_i_mass, vt_i_num, vt_s_mass, vt_g_mass, vt_g_num = tc._fall_speeds(state)
    dz = jnp.maximum(state.dz, 1.0)
    species = {
        "rain": (tc._nstep_per_column(vt_r_mass, vt_r_num, dz, dt_s), state.qr > R1),
        "ice": (tc._nstep_per_column(vt_i_mass, vt_i_mass, dz, dt_s), state.qi > R1),
        "snow": (tc._nstep_per_column(vt_s_mass, vt_s_mass, dz, dt_s), state.qs > R1),
        "graupel": (tc._nstep_per_column(vt_g_mass, vt_g_num, dz, dt_s), state.qg > R1),
    }
    out: dict[str, Any] = {}
    wet_values: dict[str, np.ndarray] = {}
    for name, (nstep, active3d) in species.items():
        wet = np.asarray(jnp.any(active3d, axis=-1))
        out[name], wet_values[name] = _summarize_nstep(np.asarray(nstep), wet, caps)
    return out, wet_values


def _merge(
    records: list[dict[str, Any]],
    wet_value_records: list[dict[str, np.ndarray]],
    caps: tuple[int, ...],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for species in ("rain", "ice", "snow", "graupel"):
        vals = []
        wet_series: list[np.ndarray] = []
        wet_count = 0
        col_count = 0
        clips = {str(cap): 0 for cap in caps}
        for rec in records:
            s = rec["species"][species]
            wet_count += int(s["wet_columns"])
            col_count += int(s["columns"])
            for cap in caps:
                clips[str(cap)] += int(s["clip_count_by_cap"][str(cap)])
            if s["wet_columns"] and s["max"] is not None:
                vals.append(s)
        for wet_values in wet_value_records:
            if wet_values[species].size:
                wet_series.append(wet_values[species])
        merged_wet = np.concatenate(wet_series) if wet_series else np.asarray([], dtype=np.float64)
        merged[species] = {
            "columns": col_count,
            "wet_columns": wet_count,
            "wet_fraction": float(wet_count / max(col_count, 1)),
            "max": None if merged_wet.size == 0 else float(np.max(merged_wet)),
            "p99": _percentile(merged_wet, 99.0),
            "p99_9": _percentile(merged_wet, 99.9),
            "max_file_p99": max((float(v["p99"]) for v in vals if v["p99"] is not None), default=None),
            "max_file_p99_9": max((float(v["p99_9"]) for v in vals if v["p99_9"] is not None), default=None),
            "clip_count_by_cap": clips,
        }
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt-s", type=float, default=10.0)
    parser.add_argument("--caps", default="16,32,64")
    parser.add_argument("--out-json", type=Path, default=Path("proofs/v0100/thompson_nstep_histogram.json"))
    parser.add_argument("wrfout", nargs="+", type=Path)
    args = parser.parse_args()

    caps = tuple(int(item) for item in args.caps.split(",") if item.strip())
    records = []
    wet_value_records = []
    for path in args.wrfout:
        one, wet_values = _one_file(path, float(args.dt_s), caps)
        records.append({"path": str(path), "species": one})
        wet_value_records.append(wet_values)
        print(json.dumps({"path": str(path), "species": one}, sort_keys=True), flush=True)

    merged = _merge(records, wet_value_records, caps)
    safe_caps = {
        species: [
            cap for cap in caps
            if int(summary["wet_columns"]) > 0 and int(summary["clip_count_by_cap"][str(cap)]) == 0
        ]
        for species, summary in merged.items()
    }
    common_safe = [
        cap for cap in caps
        if all(cap in safe_caps[species] for species in ("rain", "ice", "snow", "graupel"))
    ]
    payload = {
        "schema": "V0100Phase0ThompsonNstepHistogram",
        "schema_version": 1,
        "status": "PASS",
        "dt_s": float(args.dt_s),
        "caps": list(caps),
        "files": records,
        "merged": merged,
        "zero_clip_safe_caps_by_species": safe_caps,
        "common_zero_clip_safe_caps": common_safe,
        "recommended_nsed_safe_cap": max(common_safe) if common_safe else None,
        "note": (
            "Histogram uses existing wrfout hydrometeor columns and current JAX Thompson fall-speed/nstep formulas. "
            "It does not run the GPU forecast."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"common_zero_clip_safe_caps": common_safe}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
