#!/usr/bin/env python3
"""Exact analytic STATE-storage VRAM model per precision regime (value-independent).

The state-storage VRAM is fully determined by (field -> shape) x (field -> dtype).
Shapes come from the production contract ``_state_field_shapes`` (reads only
nz/ny/nx). Dtypes come from the three precision regimes the operational pipeline
can actually take:

  * fp64_default        : force_fp64=True path  -> every leaf float64.
  * mixed_perturb_fp32  : S4 (this sprint)      -> {p',ph',mu',w} float32, rest float64.
  * aggressive_fp32     : DEFAULT_DTYPES matrix -> ADR-007 PRECISION_MATRIX
                          (u/v/theta/qv + all moisture/number/bdy fp32; the
                          cancellation-sensitive w/mu/p/ph/totals/accum fp64).
  * combined_fp32       : aggressive matrix AND the 4 acoustic perturbations as
                          fp32 (the S4 perturbation-authoritative idea grafted
                          onto the bulk matrix) -- the maximal faithful storage win.

This is the EXACT resident-state VRAM. Peak VRAM additionally includes the
per-step transient (acoustic + physics/RRTMG scratch); that is measured on the
real GPU run and combined with this model in the report.
"""
from __future__ import annotations

import argparse
import json
from collections import namedtuple

import numpy as np

import jax.numpy as jnp

from gpuwrf.contracts.state import _state_field_shapes
from gpuwrf.contracts.precision import (
    PRECISION_MATRIX,
    MIXED_PERTURB_FP32_STORAGE_FIELDS,
)

_GridStub = namedtuple("_GridStub", ["nz", "ny", "nx"])

_BYTES = {jnp.float64: 8, jnp.float32: 4, jnp.int32: 4}


def _dtype_bytes(dt) -> int:
    name = jnp.dtype(dt).name
    return {"float64": 8, "float32": 4, "int32": 4}.get(name, 8)


def field_dtype(field: str, regime: str):
    """Return the storage dtype for a field under one precision regime."""
    matrix_dt = PRECISION_MATRIX.get(field, (jnp.float64, False))[0]
    if regime == "fp64_default":
        # force_fp64 upcasts everything except integer index leaves.
        return jnp.int32 if jnp.dtype(matrix_dt).name == "int32" else jnp.float64
    if regime == "mixed_perturb_fp32":
        if field in MIXED_PERTURB_FP32_STORAGE_FIELDS:
            return jnp.float32
        return jnp.int32 if jnp.dtype(matrix_dt).name == "int32" else jnp.float64
    if regime == "aggressive_fp32":
        return matrix_dt
    if regime == "combined_fp32":
        if field in MIXED_PERTURB_FP32_STORAGE_FIELDS:
            return jnp.float32
        return matrix_dt
    raise ValueError(regime)


REGIMES = ("fp64_default", "mixed_perturb_fp32", "aggressive_fp32", "combined_fp32")


def state_bytes(nz: int, ny: int, nx: int, regime: str, mp_physics: int = 8) -> dict:
    grid = _GridStub(nz, ny, nx)
    shapes = _state_field_shapes(grid, mp_physics=mp_physics)
    total = 0
    per_field = {}
    fp32_bytes = 0
    fp64_bytes = 0
    for field, shape in shapes.items():
        n = int(np.prod(shape))
        dt = field_dtype(field, regime)
        b = n * _dtype_bytes(dt)
        per_field[field] = {"shape": list(shape), "dtype": jnp.dtype(dt).name, "bytes": b}
        total += b
        if jnp.dtype(dt).name == "float32":
            fp32_bytes += b
        elif jnp.dtype(dt).name == "float64":
            fp64_bytes += b
    return {
        "regime": regime,
        "total_bytes": total,
        "total_mib": total / 2**20,
        "fp32_bytes": fp32_bytes,
        "fp64_bytes": fp64_bytes,
        "per_field": per_field,
    }


def scan(sizes, mp_physics: int = 8) -> dict:
    out = {"sizes": [], "mp_physics": mp_physics}
    for (nz, ny, nx, label) in sizes:
        cols = ny * nx
        entry = {"label": label, "nz": nz, "ny": ny, "nx": nx, "cols": cols, "regimes": {}}
        base = None
        for regime in REGIMES:
            sb = state_bytes(nz, ny, nx, regime, mp_physics=mp_physics)
            if base is None:
                base = sb["total_bytes"]
            entry["regimes"][regime] = {
                "state_mib": round(sb["total_mib"], 2),
                "fp32_frac_of_state": round(sb["fp32_bytes"] / sb["total_bytes"], 4),
                "state_reduction_vs_fp64": round(1.0 - sb["total_bytes"] / base, 4),
            }
        out["sizes"].append(entry)
    return out


def max_grid_that_fits(vram_gib: float, nz: int, transient_bytes_per_col: float,
                       mp_physics: int = 8) -> dict:
    """Largest square-ish grid (cols) that fits in vram_gib for each regime.

    peak = state_bytes(cols) + transient_bytes_per_col*cols.
    state_bytes is linear in cols (every leaf is (.,ny,nx) or (ny,nx)); compute
    bytes-per-col at a probe size then solve cols_max = vram / (state_bpc+trans_bpc).
    """
    budget = vram_gib * 2**30
    probe_ny, probe_nx = 200, 200
    probe_cols = probe_ny * probe_nx
    res = {}
    for regime in REGIMES:
        sb = state_bytes(nz, probe_ny, probe_nx, regime, mp_physics=mp_physics)
        state_bpc = sb["total_bytes"] / probe_cols
        cols_max = budget / (state_bpc + transient_bytes_per_col)
        side = int(np.sqrt(cols_max))
        res[regime] = {
            "state_bytes_per_col": round(state_bpc, 1),
            "cols_max": int(cols_max),
            "approx_side": side,
            "km1_area_km2": int(cols_max),  # 1 col == 1 km^2 at dx=1km
        }
    if res["fp64_default"]["cols_max"]:
        for regime in REGIMES:
            res[regime]["capability_gain_vs_fp64"] = round(
                res[regime]["cols_max"] / res["fp64_default"]["cols_max"], 3
            )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="proofs/v020/s4/ultracode/vram_model.json")
    ap.add_argument("--vram-gib", type=float, default=30.5,
                    help="usable VRAM budget (32.6 total minus ~2GiB desktop/driver)")
    ap.add_argument("--transient-bpc", type=float, default=0.0,
                    help="measured per-step transient bytes-per-column (from GPU run)")
    args = ap.parse_args()

    # Representative grids: bigswiss (measured case) + a ladder of 1km grids.
    sizes = [
        (44, 460, 460, "bigswiss_460x460x44_3km_211k"),
        (50, 300, 300, "1km_300x300x50_90k"),
        (50, 500, 500, "1km_500x500x50_250k"),
        (50, 700, 700, "1km_700x700x50_490k"),
        (50, 1000, 1000, "1km_1000x1000x50_1M"),
    ]
    model = scan(sizes)
    model["max_grid_that_fits_state_only"] = max_grid_that_fits(args.vram_gib, 50, 0.0)
    if args.transient_bpc > 0:
        model["max_grid_that_fits_with_measured_transient"] = max_grid_that_fits(
            args.vram_gib, 50, args.transient_bpc)
        model["transient_bytes_per_col"] = args.transient_bpc
    model["vram_gib_budget"] = args.vram_gib

    with open(args.out, "w") as fh:
        json.dump(model, fh, indent=2)

    # human summary
    print(f"=== STATE-storage VRAM model (exact; transient extra) -> {args.out} ===")
    for entry in model["sizes"]:
        print(f"\n{entry['label']}  ({entry['cols']:,} cols x {entry['nz']} lev)")
        for regime, r in entry["regimes"].items():
            print(f"  {regime:22s} state={r['state_mib']:9.1f} MiB  "
                  f"fp32_frac={r['fp32_frac_of_state']:.3f}  "
                  f"reduction_vs_fp64={r['state_reduction_vs_fp64']*100:5.1f}%")
    print(f"\n=== max 1km grid that fits state-only in {args.vram_gib} GiB ===")
    for regime, r in model["max_grid_that_fits_state_only"].items():
        print(f"  {regime:22s} cols_max={r['cols_max']:>12,}  side~{r['approx_side']:>5} km  "
              f"gain_vs_fp64={r.get('capability_gain_vs_fp64','-')}")


if __name__ == "__main__":
    main()
