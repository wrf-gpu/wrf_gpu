#!/usr/bin/env python3
"""Standalone RRTMG transient-memory reproducer.

CPU smoke:
  JAX_PLATFORMS=cpu PYTHONPATH=src python scripts/rrtmg_transient_reproducer.py \
    --kind lw --ncol 32 --nrep 1 --tile-cols 16 --emit-json

GPU validation must be serialized:
  scripts/with_gpu_lock.sh --label s1-rrtmg-transient -- \
    env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python scripts/rrtmg_transient_reproducer.py --kind both --ncol 144801 \
      --nrep 1 --tile-cols 1024 --out proofs/v020/oom_hardening/rrtmg_transient.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _enable_x64() -> None:
    from jax import config

    config.update("jax_enable_x64", True)


def _repeat_to_ncol(arr: np.ndarray, ncol: int, nrep: int) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim >= 2:
        reps = (ncol + out.shape[0] - 1) // out.shape[0]
        out = np.tile(out, (reps,) + (1,) * (out.ndim - 1))[:ncol]
        if nrep > 1:
            out = np.repeat(out, nrep, axis=-1)
        return out
    reps = (ncol + out.shape[0] - 1) // out.shape[0]
    return np.tile(out, reps)[:ncol]


def _sw_state(cls, arr: dict[str, np.ndarray], ncol: int, nrep: int):
    import jax.numpy as jnp

    def col(name: str):
        return jnp.asarray(_repeat_to_ncol(arr[name], ncol, nrep))

    return cls(
        T=col("input_T"),
        p=col("input_p"),
        qv=col("input_qv"),
        qc=col("input_qc"),
        qi=col("input_qi"),
        qs=col("input_qs"),
        qg=col("input_qg"),
        cloud_fraction=col("input_cloud_fraction"),
        surface_albedo=col("input_surface_albedo"),
        coszen=col("input_coszen"),
        dz=col("input_dz"),
        rho=col("input_rho"),
    )


def _lw_state(cls, arr: dict[str, np.ndarray], ncol: int, nrep: int):
    import jax.numpy as jnp

    def col(name: str):
        return jnp.asarray(_repeat_to_ncol(arr[name], ncol, nrep))

    return cls(
        T=col("input_T"),
        p=col("input_p"),
        qv=col("input_qv"),
        qc=col("input_qc"),
        qi=col("input_qi"),
        qs=col("input_qs"),
        qg=col("input_qg"),
        cloud_fraction=col("input_cloud_fraction"),
        surface_temperature=col("input_surface_temperature"),
        surface_emissivity=col("input_surface_emissivity"),
        dz=col("input_dz"),
        rho=col("input_rho"),
    )


def _memory_stats() -> dict[str, Any]:
    import jax

    devices = [dev for dev in jax.devices() if dev.platform == "gpu"]
    if not devices:
        return {"platform": "cpu", "available": False, "reason": "no JAX GPU device"}
    dev = devices[0]
    raw = {}
    if hasattr(dev, "memory_stats"):
        try:
            raw = dev.memory_stats() or {}
        except Exception as exc:  # noqa: BLE001
            return {"platform": "gpu", "available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "platform": "gpu",
        "available": bool(raw),
        "device": str(dev),
        "raw": raw,
        "peak_gib": round(int(raw.get("peak_bytes_in_use", 0)) / (1024**3), 3),
        "largest_alloc_gib": round(int(raw.get("largest_alloc_size", 0)) / (1024**3), 3),
        "bytes_in_use_gib": round(int(raw.get("bytes_in_use", 0)) / (1024**3), 3),
        "num_allocs": int(raw.get("num_allocs", 0) or 0),
    }


def run_one(kind: str, ncol: int, nrep: int, tile_cols: int, column_mode: str) -> dict[str, Any]:
    _enable_x64()
    import jax
    from gpuwrf.validation.tier1_rrtmg import LW_SAMPLE, SW_SAMPLE, _arrays

    enabled = column_mode == "tiled"
    if kind == "sw":
        import gpuwrf.physics.rrtmg_sw as swmod
        from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState

        swmod._SW_COLUMN_TILING = enabled
        swmod._SW_COLUMN_TILE_COLS = int(tile_cols if enabled else 0)
        clear = getattr(swmod.solve_rrtmg_sw_column, "clear_cache", None)
        if clear is not None:
            clear()
        state = _sw_state(RRTMGSWColumnState, _arrays(SW_SAMPLE), ncol, nrep)
        out = swmod.solve_rrtmg_sw_column(state, debug=False)
        fields = ("heating_rate", "flux_down", "flux_up", "surface_down", "surface_up")
    else:
        import gpuwrf.physics.rrtmg_lw as lwmod
        from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState

        lwmod._LW_COLUMN_TILING = enabled
        lwmod._LW_COLUMN_TILE_COLS = int(tile_cols if enabled else 0)
        clear = getattr(lwmod.solve_rrtmg_lw_column, "clear_cache", None)
        if clear is not None:
            clear()
        state = _lw_state(RRTMGLWColumnState, _arrays(LW_SAMPLE), ncol, nrep)
        out = lwmod.solve_rrtmg_lw_column(state, debug=False)
        fields = ("heating_rate", "flux_down", "flux_up", "surface_down", "surface_up")
    for field in fields:
        jax.block_until_ready(getattr(out, field))
    return {
        "kind": kind,
        "column_mode": column_mode,
        "ncol": int(ncol),
        "nrep": int(nrep),
        "nlev": int(np.asarray(state.p).shape[-1]),
        "tile_cols": int(tile_cols if enabled else 0),
        "memory": _memory_stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("lw", "sw", "both"), default="lw")
    parser.add_argument("--ncol", type=int, default=144801)
    parser.add_argument("--nrep", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=1024)
    parser.add_argument("--column-mode", choices=("tiled", "untiled"), default="tiled")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()

    kinds = ("lw", "sw") if args.kind == "both" else (args.kind,)
    rows = [run_one(kind, args.ncol, args.nrep, args.tile_cols, args.column_mode) for kind in kinds]
    payload = {
        "schema": "GpuwrfRrtmgTransientReproducer",
        "schema_version": 1,
        "env": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "rows": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.emit_json or args.out is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
